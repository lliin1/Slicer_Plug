import logging
import os
from typing import Annotated

import vtk
import qt

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode

from ctk import ctkCollapsibleButton

import numpy as np
import math
import struct

import itertools

import quaternion

# ----------------------------------------------------------------------
# 全局工具类: 用于平滑姿态
# 放在文件顶部,不依赖 UI 类
# ----------------------------------------------------------------------
class OrientationFilter:
    def __init__(self,alpha=0.3):
        """
        alpha: 滤波系数(0.0 ~ 1.0)
        0.1 = 非常平滑,滞后大(容易姿态跟不上)
        0.5 = 适中
        0.9 = 几乎实时,平滑度低
        针对颅骨表面跟随,建议 0.3 ~ 0.5
        """
        self.alpha = alpha
        self.last_q = np.quaternion(0,0,0,1) # 初始四元数（w,x,y,z）

    def update(self,target_q):
        # target_q 是 Slicer 发来的目标四元数 [qx,qy,qz,qw]
        # 转换为 numpy quaternion (w,x,y,z)
        q_target = np.quaternion(target_q[3],target_q[0],target_q[1],target_q[2])

        # 球面线性插值(Slerp)实现低通滤波
        # 这比简单的线性插值更准确,能实现单位四元数性质
        self.last_q = quaternion.slerp(
            self.last_q,
            q_target,
            0.0,        # t_start
            1.0,        # t_end
            self.alpha  # t_out (插值参数)
        )

        # 返回平滑后的四元数 [qx,qy,qz,qw]
        return [self.last_q.x,self.last_q.y,self.last_q.z,self.last_q.w]

# ----------------------------------------------------------------------
# 全局工具类: 用于平滑姿态
# 放在文件顶部,不依赖 UI 类
# ----------------------------------------------------------------------
class SimpleKalmanFilter:
    def __init__(self,Q=0.01,R=0.1):
        # Q: 过程噪声(越小代表越相信预测,平滑效果越强)
        # R: 测量噪声(越大代表越不信任原始数据,平滑效果越强)
        self.Q = np.array([[Q]])
        self.R = np.array([[R]])

        self.x = np.array([[0.0]]) # 估计值
        self.P = np.array([[1.0]]) # 误差协方差

    def update(self,measurement):
        # 1. 预测(假设速度不变,这里简化为保持上一状态)
        x_pred = self.x
        P_pred = self.P + self.Q\

        # 2. 计算卡尔曼增益 K
        K = P_pred @ np.linalg.inv(P_pred + self.R)

        # 3. 更新估计值
        z = np.array([[measurement]])
        self.x = x_pred + K@(z-x_pred)

        # 4. 更新协方差
        self.P = (np.eye(1) - K) @ P_pred

        return round(self.x[0,0],4)

# ----------------------------------------------------------------------
# 全局工具类: 用于高效计算轨迹法线和姿态
# 放在文件顶部,不依赖 UI 类
# ----------------------------------------------------------------------
class KukaTrajectoryProcessor:
    """
    生成用于 KUKA 机器人的平滑轨迹
    [前置条件]: 输入的 curve_node 必须已经在 Slicer 中经过 'Resample Curve' 处理,确保控制点分布均匀且密度足够(例如间距 0.5 mm)

    核心特性:
    1. 表面法线插值
    2. 参考向量传播 - 防止奇点
    3. 四元数符号一致性 - 防止 360 度自旋
    4. 轨迹后处理平滑
    """
    def __init__(self,model_node):
        """
        初始化:加载模型,计算平滑法线,构建定位器
        只需在路径处理前调用一次
        """
        self.model_node = model_node
        self.processed_data = None
        self.normal_array = None
        self.locator = None

        # 状态记忆变量
        self.prev_frame_x = np.array([1.0,0.0,0.0])
        self.prev_quat = np.array([0.0,0.0,0.0,1.0]) # w,x,y,z 顺序将在内部统一处理为 [x,y,z,w]
        self.is_first_point = True
        self._prepare_model_data()

        # 存储显示法线向量的列表
        self.debug_pose_data = []

    def _prepare_model_data(self):
        """预处理模型: 计算平滑法线并构建定位器"""
        poly_data = self.model_node.GetPolyData()

        # 添加平滑模型
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputData(poly_data)
        smoother.SetNumberOfIterations(500)      # 迭代次数,越多越平滑
        smoother.SetRelaxationFactor(0.1)       # 松弛因子,防止过度收缩
        smoother.FeatureEdgeSmoothingOff()     # 关闭特征角平滑,保留大轮廓
        smoother.BoundarySmoothingOff()         # 关闭边界平滑
        smoother.Update()
        smoothed_poly_data = smoother.GetOutput()

        process_source = smoothed_poly_data

        # 1. 计算法线
        normals_filter = vtk.vtkPolyDataNormals()
        #normals_filter.SetInputData(poly_data)
        normals_filter.SetInputData(process_source)

        # 设置特征角
        # 如果两个三角形夹角小于 60°,就平滑过渡;
        # 如果大于 60°,就断开,保留棱角
        normals_filter.SetFeatureAngle(60.0)

        normals_filter.ComputePointNormalsOn()  # 明确开启点法线
        normals_filter.ComputeCellNormalsOff()  # 明确关闭单元法线 (节省内存/时间)

        normals_filter.SplittingOff()           # 关闭分裂,保证锐利边缘法线平滑过渡

        normals_filter.Update()
        self.processed_data = normals_filter.GetOutput()

        self.normal_array = self.processed_data.GetPointData().GetNormals()

        if not self.normal_array:
            raise RuntimeError("无法从模型获取法线数据")

        # 2. 构建定位器(耗时操作,只做一次)
        # self.locator = vtk.vtkCellLocator()
        self.locator = vtk.vtkPointLocator()
        self.locator.SetDataSet(self.processed_data)
        self.locator.BuildLocator()

        print("[INFO] 模型法线已标准化并构建定位器")

    def process_single_point(self,point_xyz,next_point_xyz = None):
        """
        核心算法: 计算单点的位姿
        逻辑: 查找最近点 -> 插值法线 -> 传播参考向量 -> 构建矩阵 -> 校正四元数
        """
        # # A. 查找最近点(投影)
        # closest_point = [0.0,0.0,0.0]
        # cell_id = vtk.reference(0)
        # sub_cell_id = vtk.reference(0)
        # dist2 = vtk.reference(0.0)

        # self.locator.FindClosestPoint(point_xyz,closest_point,cell_id,sub_cell_id,dist2)

        # 直接使用传入的点坐标作为模型上的点
        closest_point = point_xyz

        # # 查找最近的单元格
        # cell_id = self.locator.FindCell(closest_point,0.0,vtk.vtkGenericCell(),[0.0]*3,[0.0]*3)

        # if cell_id == -1:
        #     return None

        # cell = self.processed_data.GetCell(cell_id)
        # if not cell:
        #     return None

        # # B. 重心坐标插值法线
        # pt_ids = cell.GetPointIds()
        # ids = [pt_ids.GetId(j) for j in range(3)]

        # n0 = np.array(self.normal_array.GetTuple(ids[0]))
        # n1 = np.array(self.normal_array.GetTuple(ids[1]))
        # n2 = np.array(self.normal_array.GetTuple(ids[2]))

        # weights = [0.0,0.0,0.0]
        # pcoords = [0.0,0.0,0.0]
        # dist_eval = vtk.reference(0.0)
        # sub_id_eval = vtk.reference(0)

        # cell.EvaluatePosition(closest_point,[0.0]*3,sub_id_eval,pcoords,dist_eval,weights)

        # interpolated_normal = (weights[0]*n0+weights[1]*n1+weights[2]*n2)

        # norm = np.linalg.norm(interpolated_normal)

        # z_axis = -interpolated_normal/norm if norm > 1e-6 else np.array([0.0,0.0,1.0])

        # 三角面片求法向量
        # polyData = self.model_node.GetPolyData()
        # points = polyData.GetPoints()
        # pt_ids = cell.GetPointIds()
        # ids = [pt_ids.GetId(j) for j in range(3)]

        # p0 = points.GetPoint(ids[0])
        # p1 = points.GetPoint(ids[1])
        # p2 = points.GetPoint(ids[2])

        # edge1 = np.array(p1) - np.array(p0)
        # edge2 = np.array(p2) - np.array(p0)

        # interpolated_normal = np.cross(edge1,edge2)
        # norm = np.linalg.norm(interpolated_normal)

        # z_axis = -interpolated_normal/norm if norm> 1e-6 else np.array([0.0,0.0,1.0])

        closest_point_id = self.locator.FindClosestPoint(point_xyz)
        interpolated_normal = np.array(self.normal_array.GetTuple(closest_point_id))
        norm = np.linalg.norm(interpolated_normal)

        z_axis = -interpolated_normal/norm if norm>1e-6 else np.array([0.0,0.0,1.0])

        # 构建 X 轴 (参考向量传播)
        candidate_x = None

        if next_point_xyz is not None:
            vec = np.array(next_point_xyz) - np.array(point_xyz)
            norm_vec = np.linalg.norm(vec)
            if norm_vec > 1e-6:
                geo_tangent = vec/norm_vec
                proj_tangent = geo_tangent - np.dot(geo_tangent,z_axis)*z_axis
                norm_proj = np.linalg.norm(proj_tangent)
                if norm_proj > 1e-6:
                    candidate_x = proj_tangent/norm_proj

        if candidate_x is None:
            prev_x_proj = self.prev_frame_x - np.dot(self.prev_frame_x,z_axis)*z_axis
            norm_prev = np.linalg.norm(prev_x_proj)

            if norm_prev > 1e-6:
                candidate_x = prev_x_proj/norm_prev
            else:
                aux = np.array([1.0,0.0,0.0]) if z_axis[0] < 0.9 else np.array([0.0,1.0,0.0])
                candidate_x = np.cross(z_axis,aux)
                candidate_x = candidate_x/(np.linalg.norm(candidate_x)+1e-9)

        x_axis = candidate_x
        y_axis = np.cross(x_axis,z_axis)
        y_axis = y_axis/(np.linalg.norm(y_axis)+1e-9)

        self.prev_frame_x = x_axis

        # D. 构造旋转矩阵 -> 四元数
        rot_mat = np.column_stack((x_axis,y_axis,z_axis))
        trace = np.trace(rot_mat)

        if(trace > 0):
            S = np.sqrt(1.0+trace)*2
            qw = 0.25*S
            qx = (rot_mat[2,1]-rot_mat[1,2])/S
            qy = (rot_mat[0,2]-rot_mat[2,0])/S
            qz = (rot_mat[1,0]-rot_mat[0,1])/S
        elif(rot_mat[0,0]>rot_mat[1,1] and rot_mat[0,0]>rot_mat[2,2]):
            S = np.sqrt(1.0 + rot_mat[0,0] - rot_mat[1,1] - rot_mat[2,2])*2
            qw = (rot_mat[2,1]-rot_mat[1,2])/S
            qx = 0.25*S
            qy = (rot_mat[0,1]+rot_mat[1,0])/S
            qz = (rot_mat[0,2]+rot_mat[2,0])/S
        elif(rot_mat[1,1]>rot_mat[0,0] and rot_mat[1,1]>rot_mat[2,2]):
            S = np.sqrt(1.0 + rot_mat[1,1] - rot_mat[0,0] - rot_mat[2,2])*2
            qw = (rot_mat[0,2]-rot_mat[2,0])/S
            qx = (rot_mat[0,1]+rot_mat[1,0])/S
            qy = 0.25*S
            qz = (rot_mat[1,2]+rot_mat[2,1])/S
        else:
            S = np.sqrt(1.0 + rot_mat[2,1] - rot_mat[0,0] - rot_mat[1,1])*2
            qw = (rot_mat[1,0]-rot_mat[0,1])*2
            qx = (rot_mat[0,2]+rot_mat[2,0])/S
            qy = (rot_mat[1,2]+rot_mat[2,1])/S
            qz = 0.25*S

        current_quat = np.array([qx,qy,qz,qw])

        # 平滑四元数        
        orientationFilter = OrientationFilter()
        orientationFilter.update(current_quat)

        norm_quat = np.linalg.norm(current_quat)
        if norm_quat > 1e-8:
            current_quat = current_quat/norm_quat

        if not self.is_first_point:
            if np.dot(self.prev_quat,current_quat) < 0:
                current_quat = -current_quat
        self.is_first_point = False

        self.prev_quat = current_quat

        # 将位姿添加到显示法向量的列表中
        self.debug_pose_data.append({
            "pos":list(point_xyz),
            "quat":current_quat.tolist(),
            "normal_vec":z_axis.tolist()
        })

        return current_quat.tolist()

    # def show_normals_in_slicer(self,node_name="Debug_Tool_Axis", scale=15.0):
    #     if not self.debug_pose_data:
    #         print("[WARNING] 没有缓存的位姿数据")
    #         return

    #     # 1. 创建/清空节点
    #     try:
    #         fiducial_node = slicer.util.getNode(node_name)
    #         if fiducial_node:
    #             fiducial_node.RemoveAllControlPoints()
    #     except slicer.util.MRMLNodeNotFoundException:
    #         fiducial_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode",node_name)

    #     # 2. 配置显示: 开启箭头(Glyph)
    #     # 配置显示节点
    #     display_node = fiducial_node.GetDisplayNode()

    #     # 设置 Glyph 类型(3 = Arrow, 0 = None, 1 = Sphere, 2 = Cube)
    #     # 设置非 0 类型,就会自动显示,无需 SetGlyphVisibility
    #     display_node.SetGlyphType(3) # 3 = Arrow

    #     # 设置 Glyph 大小
    #     display_node.SetGlyphScale(scale)

    #     # 启用独立方向
    #     # 默认是 False,导致所有箭头都朝向同一个方向
    #     if hasattr(display_node,"SetUseOrientation"):
    #         display_node.SetUseOrientation(True)
    #     else:
    #         print("[WARNING] No SetUseOrientation")

    #     # 设置颜色和样式
    #     display_node.SetColor(1,0,0) # 红色
    #     display_node.SetSelectedColor(1,0,0)
    #     display_node.SetActiveColor(1,0,0)
    #     # display_node.SetTextVisibility(False)
    #     display_node.SetTextScale(0.0)
    #     display_node.SetOpacity(1.0)

    #     # 3. 直接填入数据(无需任何数学转换)
    #     for item in self.debug_pose_data:
    #         pos = item["pos"]
    #         quat = item["quat"] # [x,y,z,w]

    #         # 添加点
    #         index = fiducial_node.AddControlPointWorld(pos,"")

    #         qx,qy,qz,qw = quat
    #         fiducial_node.SetNthControlPointOrientation(index,qw,qx,qy,qz)

    #     print(f"[INFO] 可视化完成: '{node_name}'")
    #     print(f"[INFO] 红色箭头 = 机械臂工具坐标系的 Z 轴(已包含取反逻辑)")
    def show_normals_in_slicer(self,node_name = "Normals_Lines",line_length = 10.0):
        if not self.debug_pose_data:
            print("[WARNING] 没有缓存的位姿数据")
            return

        try:
            fiducial_node = slicer.util.getNode(node_name)
        except slicer.util.MRMLNodeNotFoundException:
            fiducial_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode",node_name)

        # 配置显示
        display_node = fiducial_node.GetDisplayNode()

        # 设置颜色和样式
        display_node.SetColor(1,0,0)
        display_node.SetSelectedColor(1,0,0)
        display_node.SetActiveColor(1,0,0)
        display_node.SetTextScale(0.0)
        display_node.SetOpacity(1.0)

        for i,item in enumerate(self.debug_pose_data):
            pos = item["pos"]
            quat = item["quat"]

            z_axis = np.array(item["normal_vec"])
            # # 从四元数提取 Z 轴向量
            # qx,qy,qz,qw = quat

            # # 旋转矩阵第三列公式
            # zx = 2 * (qx * qz + qw * qy)
            # zy = 2 * (qy * qz - qw * qx)
            # zz = 1 - 2 * (qx * qx + qy * qy)

            # z_axis = np.array([zx,zy,zz])
            
            # # 归一化
            # z_norm = np.linalg.norm(z_axis)

            # if z_norm > 1e-6:
            #     z_axis = z_axis/z_norm

            # 计算线段的终点: 起点 + (方向 * 长度)
            end_pos = pos + z_axis*line_length 

            # Slicer 中以 “起点-终点-断点” 的方式绘制多条线段
            # 1. 添加起点
            start_point_id = fiducial_node.AddControlPointWorld(pos,"")
            fiducial_node.SetNthControlPointDescription(start_point_id,"StartLine")
            # 2. 添加终点 (形成线段)
            end_point_id = fiducial_node.AddControlPointWorld(end_pos,"")
            fiducial_node.SetNthControlPointDescription(end_point_id,"EndLine")
            # 3. 添加一个与终点重合的 “断点”
            # fiducial_node.AddControlPointWorld(end_pos,"")
        
        print("[INFO] 法线可视化完成")
        
#
# RS
#

class RS(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("RS")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#RS">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # RS1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="RS",
        sampleName="RS1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "RS1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="RS1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="RS1",
    )

    # RS2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="RS",
        sampleName="RS2",
        thumbnailFileName=os.path.join(iconsPath, "RS2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="RS2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="RS2",
    )


#
# RSParameterNode
#


@parameterNodeWrapper
class RSParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """

    inputVolume: vtkMRMLScalarVolumeNode
    imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
    invertThreshold: bool = False
    thresholdedVolume: vtkMRMLScalarVolumeNode
    invertedVolume: vtkMRMLScalarVolumeNode


#
# RSWidget
#

# IGTL_STATE_OFF = 0
# IGTL_STATE_WAIT_CONNECTION = 1
# IGTL_STATE_CONNECTED = 2

# CONNECTED_EVENT = 19001
# DISCONNECTED_EVENT = 19002

class RSWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation

        self.pointsSet = []
        # self.pathModelNode = None
        self.pathStringNode = None
        self._currentConnectionState = None
        self._hasEverConnected = False
        self.heartbeatTimer = None

        self.igtlConnector = None
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

        # 自动创建或获取 Transform 节点
        self.calibNode = None
        transform_node_name = "HandEyeCalibration"

        try:
            # 1. 尝试获取已存在的节点
            self.calibNode = slicer.util.getNode(transform_node_name)
            print(f"[INFO] 找到现有的变换节点: {transform_node_name}")
        except slicer.util.MRMLNodeNotFoundException:
            # 2. 如果不存在,创建一个新的 Transform 节点
            print(f"[INFO] 未找到节点,正在创建新的 Transform 节点: {transform_node_name}...")
            self.calibNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode",transform_node_name)

            # 3. 初始化为单位矩阵(需要后期标定)
            idensity_matrix = np.eye(4)

            # 4. 创建 VTK 矩阵对象并填充数据
            matrix_vtk = vtk.vtkMatrix4x4()
            
            # 将 numpy 数组转换为 vtkMatrix4x4 并设置到节点
            slicer.util.updateVTKMatrixFromArray(matrix_vtk,idensity_matrix)
            print(f"[WARNING] 节点已创建,但当前是单位矩阵(无变换)!")
            print(f"[WARNING] 机械臂将执行错误的坐标。请在'Transforms'模块中调整 {transform_node_name} 或进行手眼标定。")

            # 5. 将矩阵应用到节点
            self.calibNode.SetMatrixTransformToParent(matrix_vtk)

            # print(f"[INFO] 当前使用的变换矩阵:\n{self.slicerToKukaTransform}")

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # 创建可折叠面板
        connectionCollapsible = ctkCollapsibleButton()
        connectionCollapsible.text = "通信"
        self.layout.addWidget(connectionCollapsible)

        formLayout = qt.QFormLayout(connectionCollapsible)

        # 目标 IP
        self.targetIp = qt.QLineEdit("127.0.0.1")
        targetIP_Label = qt.QLabel("目标IP:")
        targetIP_Label.setStyleSheet("font-weight:bold;font-size:10pt;color:black")
        formLayout.addRow(targetIP_Label,self.targetIp)

        # 目标端口
        self.targetPort = qt.QLineEdit("18944")
        targetPort_Label = qt.QLabel("目标端口:")
        targetPort_Label.setStyleSheet("font-weight:bold;font-size:10pt;color:black")
        formLayout.addRow(targetPort_Label,self.targetPort)

        # 按钮区域
        buttonLayout = qt.QHBoxLayout()
        self.connectButton = qt.QPushButton("连接")
        self.disconnectButton = qt.QPushButton("关闭")
        self.disconnectButton.enabled = False
        buttonLayout.addWidget(self.connectButton)
        buttonLayout.addWidget(self.disconnectButton)
        formLayout.addRow(buttonLayout)

        # 连接状态
        self.statusLabel = qt.QLabel("连接断开")
        statusText_Label = qt.QLabel("状态:")
        self.statusLabel.setStyleSheet("font-weight:bold;font-size:10pt;color:red")
        statusText_Label.setStyleSheet("font-weight:bold;font-size:10pt;color:black")
        formLayout.addRow(statusText_Label,self.statusLabel)

        # 路径名
        pathLabel = qt.QLabel("路径名:")
        pathLabel.setStyleSheet("font-weight:bold;font-size:10pt;color:black")
        self.pathEdit = qt.QLineEdit("CC")
        formLayout.addRow(pathLabel,self.pathEdit)

        # 标定点
        # 1. 定义所有组名 (Calibration Groups)
        calibration_groups = ["calibration1","calibration2","calibration3"]

        # 2. 定义配置列: (后缀,标签文本)
        config = [
            ("x","X (mm):"),
            ("y","Y (mm):"),
            ("z","Z (mm):"),
            ("a","A (°):"),
            ("b","B (°):"),
            ("c","C (°):"),
        ]

        # 3. 嵌套循环生成所有控件
        for i,goup_name in enumerate(calibration_groups):
            # 为每一组添加一个分割标题,让界面更清洗
            title_name = qt.QLabel(f"标定点{i}")
            title_name.setStyleSheet("font-weight:bold;font-size:10pt;color:black")
            title_name.setAlignment(qt.Qt.AlignCenter)
            formLayout.addRow(title_name)

            # --- 创建网格布局 (2行3列) ---
            gridLayout = qt.QGridLayout()
            gridLayout.setContentsMargins(0,0,0,0) # 去除多余边框
            gridLayout.setSpacing(5) # 控件之间的间距

            # --- 循环填充网格 ---
            for index,(suffix,label_text) in enumerate(config):
                # 计算行和列
                row = index // 3
                col = index % 3

                # --- 动态构建属性名 ---
                label_name = f"{calibration_groups}_{suffix}Label"
                edit_name = f"{calibration_groups}_{suffix}Edit"

                label = qt.QLabel(label_text)
                label.setMaximumWidth(80)

                edit = qt.QLineEdit()
                # 让输入框填满单元格
                edit.setSizePolicy(qt.QSizePolicy.Expanding,qt.QSizePolicy.Preferred)

                # 绑定到 self(保持变量名可用)
                setattr(self,edit_name,edit)

                # --- 添加到网格的具体位置 ---
                gridLayout.addWidget(label,row,col*2)  # 标签放在偶数列 (0,2,4)
                gridLayout.addWidget(edit,row,col*2+1) # 输入框放在奇数列 (1,3,5)

            containerWidget = qt.QWidget()
            containerWidget.setLayout(gridLayout)

            formLayout.addRow(containerWidget)
            formLayout.setSpacing(15)

        # 坐标变换
        self.transformation_Button = qt.QPushButton("坐标变换")
        formLayout.addRow(self.transformation_Button)

        # 获取路径
        self.getPath_Button = qt.QPushButton("获取路径")
        formLayout.addRow(self.getPath_Button)

        # 发送路径
        self.sendPath_Button = qt.QPushButton("发送路径")
        formLayout.addRow(self.sendPath_Button)

        # 连接信号
        self.connectButton.connect('clicked()',self.onConnect)
        self.disconnectButton.connect('clicked()',self.onDisconnect)
        self.getPath_Button.connect('clicked()',self.onGetPath)
        self.sendPath_Button.connect('clicked()',self.onStorePath)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = RSLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def onConnect(self)->None:
        if self.igtlConnector:
            self.onDisconnect()
        try:
            host = self.targetIp.text.strip()
        except:
            self.showError("请输入目标地址")
        try:
            port = int(self.targetPort.text.strip())
        except:
            self.showError("请输入目标端口")

        # 创建 IGTL 连接器
        self.igtlConnector = slicer.vtkMRMLIGTLConnectorNode()
        self.igtlConnector.SetTypeClient(host,port)
        slicer.mrmlScene.AddNode(self.igtlConnector)

        # 创建 ModelNode 并注册为 outgoing
        # self.pathModelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode","RobotPath")
        self.pathStringNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTextNode","DeltaData")
        # self.pathModelNode.SetAttribute("HideFromEditors","true")
        self.pathStringNode.SetAttribute("HideFromEditors","true")

        # self.igtlConnector.RegisterOutgoingMRMLNode(self.pathModelNode)
        self.igtlConnector.RegisterOutgoingMRMLNode(self.pathStringNode)

        # 创建心跳节点
        self.heartbeatNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode","Heartbeat")
        self.heartbeatNode.SetAttribute("HideFromEditors","true")
        self.igtlConnector.RegisterOutgoingMRMLNode(self.heartbeatNode)

        self.igtlConnector.Start()

        self.heartbeatTimer = qt.QTimer()
        self.heartbeatTimer.setInterval(1000)
        self.heartbeatTimer.connect('timeout()',self._sendHeartbeat)
        self.heartbeatTimer.start()

        # 添加观察器: 监听连接器状态变化
        self.addObserver(self.igtlConnector,vtk.vtkCommand.ModifiedEvent,self._onConnectorModified)

        # 监听服务器发来的 STATUS 消息
        self.statusNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLIGTLStatusNode","ServerControl")
        self.statusNode.SetName("ServerControl")
        self.statusNode.SetAttribute("HideFromEditors","true")
        self.igtlConnector.RegisterIncomingMRMLNode(self.statusNode)
        self.addObserver(self.statusNode,vtk.vtkCommand.ModifiedEvent,self._onServerStatusUpdate)

        # 启动连接器后立即同步一次状态
        self._onConnectorModified(self.igtlConnector,vtk.vtkCommand.ModifiedEvent)

    def _onNodeAdded(self,caller,event,callData):
        node = callData
        if not node or not isinstance(node,slicer.vtkMRMLIGTLStatusNode):
            return

        # 检查是否是我们关心的 DeviceName
        device_name = node.GetIgtlDeviceName()
        print(f"[DEBUG] New IGTLStatusNode created: {node.GetName()},DeviceName='{device_name}")

        if device_name == "ServerControl":
            # 如果已有 statusNode,先移除旧观察器
            if hasattr(self,'statusNode') and self.statusNode:
                self.RemoveObserver(self.statusNode,vtk.vtkCommand.ModifiedEvent,self._onServerStatusUpdate)
            # 找到目标状态节点!保存引用并添加观测器
            self.statusNode = node
            self.statusNode.SetAttribute("HideFromEditors","true")
            self.addObserver(self.statusNode,vtk.vtkCommand.ModifiedEvent,self._onServerStatusUpdate)

    def _onServerStatusUpdate(self,caller,event):
        if not hasattr(self,'statusNode') or not self.statusNode:
            return
        status_string = self.statusNode.GetStatusString()
        if status_string and "shutting down" in status_string.lower():
            print("Received server shutdown notification:",status_string)
            self._handleDisconnection()

    def _onConnectorModified(self,caller,event):
        if not self.igtlConnector:
            print("被移除\n")
            return

        state = self.igtlConnector.GetState()
        print(f"[DEBUG] state={state}, _hasEverConnected={self._hasEverConnected}")

        if state == 2: # CONNECTED
            print("进入连接\n")
            if not self._hasEverConnected:
                print(f"{self._hasEverConnected}\n")
                self._hasEverConnected = True
                self.statusLabel.setText("已连接")
                self.statusLabel.setStyleSheet("color:green")
                self.connectButton.enabled = False
                self.disconnectButton.enabled = True
        elif state == 1: # WAIT
            print("等待连接\n")
            self.statusLabel.setText("连接中")
            self.statusLabel.setStyleSheet("color:orange")
            self.connectButton.enabled = False
            self.disconnectButton.enabled = True

        elif state == 0: # DISCONNECTED
            print("进入断开\n")
            if self._hasEverConnected:
                self._handleDisconnection()
            else:
                # 从未成功连接过
                self._cleanupConnectorResources()
                self.statusLabel.setText("连接失败")
                self.statusLabel.setStyleSheet("color:red")
                self.connectButton.enabled = True
                self.disconnectButton.enabled = False

    def _cleanupConnectorResources(self)->None:
        if hasattr(self,'heartbeatTimer') and self.heartbeatTimer:
            self.heartbeatTimer.stop()
            self.heartbeatTimer = None

        if self.igtlConnector:
            # 移除对连接器的观察器
            self.removeObserver(self.igtlConnector,vtk.vtkCommand.ModifiedEvent,self._onConnectorModified)
            self.igtlConnector.Stop()
            slicer.mrmlScene.RemoveNode(self.igtlConnector)
            self.igtlConnector = None

        # 移除可能残留的节点
        for nodeName in ["RobotPath","Heartbeat","ServerControl"]:
            node = slicer.mrmlScene.GetFirstNodeByName(nodeName)
            if node:
                slicer.mrmlScene.RemoveNode(node)

        self.pathModelNode = None

        # 移除场景节点添加观察器
        # self.removeObserver(slicer.mrmlScene,slicer.vtkMRMLScene.NodeAddedEvent,self._onNodeAdded)

        # 清理状态节点引用和观察器
        if hasattr(self,'statusNode') and self.statusNode:
            self.removeObserver(self.statusNode,vtk.vtkCommand.ModifiedEvent,self._onServerStatusUpdate)
            self.statusNode = None
        self._hasEverConnected = False

    def _handleDisconnection(self):
        self._cleanupConnectorResources()

        # 更新 UI
        self.statusLabel.setText("连接断开")
        self.statusLabel.setStyleSheet("font-weight:bold;font-size:10pt;color:red")
        self.connectButton.enabled = True
        self.disconnectButton.enabled = False

    def _sendHeartbeat(self):
        if not self.igtlConnector or self.igtlConnector.GetState() != 2:
            return

        poly = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        points.InsertNextPoint(0.0,0.0,0.0)
        poly.SetPoints(points)

        self.heartbeatNode.SetAndObservePolyData(poly)
        self.heartbeatNode.Modified()
        # self.igtlConnector.PushNode(self.heartbeatNode)

    def onDisconnect(self)->None:
        self._handleDisconnection()

    def get_normal_and_quaternion(model_node,point_coord=None,up_vector=[0,0,1]):
        """
        获取模型表面某点的内法线,并转换为四元数

        参数:
        - model_node: Slicer 模型节点
        - point_coord: [x,y,z] 列表. 如果提供,则计算该坐标最近点的法线
        - up_vector: 参考向上矢量 [x,y,z], 用于构建完整的旋转(解决万向节锁/滚动角问题).
                     默认为世界坐标系 Z 轴 [0,0,1]. 如果法线与 up_vector 平行, 需手动调整此值
        """

    def onGetPath(self)->None:
        pathName = self.pathEdit.text.strip()

        if pathName=="":
            self.showError("请输入路径名")
            return
        try:
            pathNode = slicer.util.getNode(pathName)
        except slicer.util.MRMLNodeNotFoundException:
            pathNode = None
            if pathName:
                self.showError("请导入模型文件")
            else:
                self.showError("请输入正确的路径名")
            return

        # 检查是否为 Markups 节点
        if not hasattr(pathNode,'GetNumberOfControlPoints'):
            self.showError("请选择一个 Markups 节点")
            return

        self.pointsSet.clear()

        # 遍历所有点,获取点坐标
        pointsNum = pathNode.GetNumberOfControlPoints()

        if pointsNum == 0:
            self.showError("该 Markups 节点没有控制点")
            return

        # 获取模型文件
        model_name = "FinalHoleImprove1"
        model_node = slicer.util.getNode(model_name)

        if model_node:
            print(f"找到模型文件: {model_node.GetName()}\n")
        else:
            print(f"没有找到模型文件: {model_name}\n")
        # 默认姿态(四元数)
        # default_quat = [0.0,0.0,0.0,1.0]

        orientationProcessor = KukaTrajectoryProcessor(model_node)

        if pointsNum > 0:
            for i in range(pointsNum):
                points = [0.0,0.0,0.0]
                quat = [0.0,0.0,0.0,1.0]
                pathNode.GetNthControlPointPositionWorld(i,points)
                quat = orientationProcessor.process_single_point(points)

                # self.pointsSet.append((points,default_quat))
                self.pointsSet.append((points,quat))
        else:
            print("没有足够的点位进行计算\n")
            return

        print("已获取路径点位")

        orientationProcessor.show_normals_in_slicer()

    def createPathPolyDataWithOrientation(self,points_with_orientation):
        if points_with_orientation is None:
            return

        polyData = vtk.vtkPolyData()
        vtk_points = vtk.vtkPoints()
        quat_array = vtk.vtkFloatArray()
        quat_array.SetName("Orientation") # 命名数组
        quat_array.SetNumberOfComponents(4) # 四元数
        quat_array.SetNumberOfTuples(len(points_with_orientation))

        for i,(pos,quat) in enumerate(points_with_orientation):
            vtk_points.InsertNextPoint(pos[0],pos[1],pos[2])
            quat_array.SetTuple4(i,quat[0],quat[1],quat[2],quat[3])

        polyData.SetPoints(vtk_points)
        polyData.GetPointData().AddArray(quat_array)
        return polyData

    def quaternion_to_kuka_abc(self,quat):
        """
        将四元数 [x, y, z, w] 转换为 KUKA 风格的 A, B, C (度)。

        数学定义 (Z-Y-X 顺序):
          1. 绕 Z 轴旋转 C 角 (Yaw)
          2. 绕 Y 轴旋转 B 角 (Pitch)
          3. 绕 X 轴旋转 A 角 (Roll)
        返回值: (A, B, C)
          A: 绕 X 轴角度 (Roll)
          B: 绕 Y 轴角度 (Pitch)
          C: 绕 Z 轴角度 (Yaw)
        """
        x,y,z,w = quat

        # 归一化
        norm = math.sqrt(x*x+y*y+z*z+w*w)
        if norm == 0: return 0.0,0.0,0.0
        x,y,z,w = x/norm,y/norm,z/norm,w/norm

        # --- 标准 Z-Y-X 欧拉角提取公式 ---
        # 1. 计算 B (Pitch,绕 Y 轴)
        # sin(B) = 2 * (w*y - z*x)
        sinp = 2*(w*y - z*x)
        if abs(sinp) >= 1:
            # 万向节死锁: Pitch = +/- 90 度
            B = math.copysign(math.pi/2,sinp)
            # 死锁时,A 和 C 耦合,通常设 A = 0,解算 C
            A = 0.0
            C = math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        else:
            B = math.asin(sinp)

            # 2. 计算 A (Roll,绕 X 轴)
            sinr_cosp = 2*(w*x+y*z)
            cosr_cosp = 1 - 2*(x*x+y*y)
            A = math.atan2(sinr_cosp,cosr_cosp)

            # 3. 计算 C (Yaw,绕 Z 轴)
            siny_cosp = 2*(w*z+x*y)
            cosy_cosp = 1-2*(y*y+z*z)
            C = math.atan2(siny_cosp,cosy_cosp)

        # 转换为角度 (度)
        A_deg = math.degrees(A)
        B_deg = math.degrees(B)
        C_deg = math.degrees(C)

        return A_deg,B_deg,C_deg

    def rotation_matrix_to_quaternion(self,R):
        """
        将 3x3 旋转矩阵转换为四元数 [x,y,z,w]
        """
        trace = np.trace(R)

        if trace > 0:
            S = math.sqrt(trace+1.0)*2
            qw = 0.25 * S
            qx = (R[2,1] - R[1,2])/S
            qy = (R[0,2] - R[2,0])/S
            qz = (R[1,0] - R[0,1])/S
        elif(R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = math.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2
            qw = (R[2,1]-R[1,2])/S
            qx = 0.25*S
            qy = (R[0,1]+R[1,0])/S
            qz = (R[0,2]+R[2,0])/S
        elif R[1,1] > R[2,2]:
            S = math.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2
            qw = (R[0,2] - R[2,0])/S
            qx = (R[0,1] + R[1,0])/S
            qy = 0.25*S
            qz = (R[1,2]+R[2,1])/S
        else:
            S = math.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2
            qw = (R[1,0]-R[0,1])/S
            qx = (R[0,2]+R[2,0])/S
            qy = (R[1,2]+R[2,1])/S
            qz = 0.25*S

        # 归一化结果
        norm = math.sqrt(qx*qx+qy*qy+qz*qz+qw*qw)
        if norm > 0:
            return [qx/norm,qy/norm,qz/norm,qw/norm]
        else:
            return [0.0,0.0,0.0,1.0]

    def apply_coordinate_transform(self,pos,quat,transform_matrix):
        """
        将点和姿态从 Slicer 坐标系变换到 KUKA (目标) 坐标系

        参数:
            pos: [x,y,z] (Slicer 坐标)
            quat: [x,y,z,w] (Slicer 姿态,四元数)
            transform_matrix: 4x4 numpy array (齐次变换矩阵,表示 Slicer -> KUKA 的变换)
                            即 P_kuka = T * P_slicer

        返回:
            new_pos: [x,y,z] (KUKA 坐标)
            new_quat: [x,y,z,w] (KUKA 姿态,四元数)
        """

        # --- 1. 变换位置 (Position) ---
        # 将 3D 点转换为齐次坐标 [x,y,z,1]
        p_homogeneous = np.array([pos[0],pos[1],pos[2],1.0])

        # 应用变换矩阵: P_new = T * P_old
        p_transformed = np.dot(transform_matrix,p_homogeneous)

        # 提取前三个分量作为新坐标
        new_pos = p_transformed[:3].tolist()

        # --- 2. 变换姿态 (Orientation) ---
        # 提取变换矩阵的旋转部分 (3x3)
        R_transform = transform_matrix[:3,:3]

        # 将输入四元数转换为旋转矩阵 (3x3)
        x,y,z,w = quat

        # 归一化
        norm = np.sqrt(x*x+y*y+z*z+w*w)
        if norm > 0:
            x,y,z,w = x/norm,y/norm,z/norm,w/norm

        R_Slicer = np.array([
            [1-2*y*y-2*z*z,2*x*y-2*z*w,2*x*z+2*y*w],
            [2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w],
            [2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y]
        ])

        # 计算新坐标系下的旋转矩阵
        R_kuka = np.dot(R_transform,R_Slicer)

        # 将新的旋转矩阵转换回四元数 [x,y,z,w]
        new_quat = self.rotation_matrix_to_quaternion(R_kuka)

        return new_pos,new_quat

    def onStorePath(self)->None:
        if not self.pointsSet:
            slicer.util.warningDisplay("请先获取路径点")
            return

        if self.igtlConnector is None or self.igtlConnector.GetState() != 2:
            slicer.util.warningDisplay("连接已断开,请重新连接!")
            return

        # 动态获取最新矩阵
        try:
            if not self.calibNode:
                self.calibNode = slicer.util.getNode("HandEyeCalibration")

        # 每次发送都重新读取,确保用户刚在 Transforms 模块的修改生效
            matrix_vtk = vtk.vtkMatrix4x4()
            self.calibNode.GetMatrixTransformToParent(matrix_vtk)

            current_transform_matrix = np.array([
                [matrix_vtk.GetElement(r,c) for c in range(4)]
                for r in range(4)
            ])

            # 定义一个绕 X 轴旋转 5° 的修正矩阵
            # 目的: 让工具坐标系的 Z 轴偏离默认方向 5°，迫使 A5 轴避开 0°
            tilt_angle_deg = 5.0
            tilt_angle_rad = math.radians(tilt_angle_deg)

            # 绕 X 轴旋转矩阵
            R_tilt = np.array(
                [
                    [1,0,0,0],
                    [0,math.cos(tilt_angle_rad),-math.sin(tilt_angle_rad),0],
                    [0,math.sin(tilt_angle_rad),math.cos(tilt_angle_rad),0],
                    [0,0,0,1]
                ]
            )
            # 将修正矩阵应用到工具变换中
            # 乘法顺序: 通常是 基础变换 * 修正变换
            # 按修正后的工具姿态来理解法线
            final_transform_matrix = np.dot(current_transform_matrix,R_tilt)    

        except Exception as e:
            slicer.util.errorDisplay(f"获取变换矩阵失败: {e}")
            return

        print(f"[INFO] 开始处理 {len(self.pointsSet)} 个路径点并生成二进制流...")

        # 卡尔曼滤波
        # kf_ABC = SimpleKalmanFilter(Q=0.001,R=0.1) # 调节 Q 和 R 来控制
        # smooth_da = 0.0
        # smooth_db = 0.0
        # smooth_dc = 0.0

        # 临时存储转换后的绝对坐标,用于增量计算
        abs_points = []

        # 2. 第一遍遍历: 坐标变换 & 四元数转 ABC
        for i,(pos,quat) in enumerate(self.pointsSet):
            # A. 坐标变换 (Slicer -> KUKA)
            # kuka_pos,kuka_quat = self.apply_coordinate_transform(pos,quat,current_transform_matrix)
            kuka_pos,kuka_quat = self.apply_coordinate_transform(pos,quat,final_transform_matrix)

            # B. 四元数转 KUKA 欧拉角 (A,B,C) -> 角度制
            A,B,C = self.quaternion_to_kuka_abc(kuka_quat)

            abs_points.append({
                'x':kuka_pos[0],
                'y':kuka_pos[1],
                'z':kuka_pos[2],
                'a':A,
                'b':B,
                'c':C
            })

        # 3. 第二遍遍历: 计算增量 -> 打包 -> 发送
        delta_groups = []

        for i,pt in enumerate(abs_points):
            if i == 0:
                # 第一个点:起始点采用绝对坐标
                # dx = pt['x']
                # dy = pt['y']
                # dz = pt['z']
                # da = pt['a']
                # db = pt['b']
                # dc = pt['c']
                dx = 0
                dy = 0
                dz = 0
                da = 0
                db = 0
                dc = 0
                # print(f"dx: {dx}; dy: {dy}; dz: {dz}; da: {da}; db: {db}; dc: {dc}\n")
            else:
                prev_pt = abs_points[i-1]

                dx = round(pt['x']-prev_pt['x'],4)
                dy = round(pt['y']-prev_pt['y'],4)
                dz = round(pt['z']-prev_pt['z'],4)
                da = round(pt['a']-prev_pt['a'],4)
                db = round(pt['b']-prev_pt['b'],4)
                dc = round(pt['c']-prev_pt['c'],4)

                # smooth_da = kf_ABC.update(da)
                # smooth_db = kf_ABC.update(db)
                # smooth_dc = kf_ABC.update(dc)

                # print(f"dx: {dx}; dy: {dy}; dz: {dz}; da: {da}; db: {db}; dc: {dc}\n")
            delta_groups.append([dx,dy,dz,da,db,dc])    # 更改为列表,便于修改

        pre_num_da,pre_num_db,pre_num_dc = -1,-1,-1
        none_zero_a,none_zero_b,none_zero_c = 0,0,0

        total_len = len(delta_groups)

        for i,item in enumerate(delta_groups):
            # --- 处理 A 列 (索引 3) ---
            if item[3] != 0:
                if none_zero_a == 0:  # 遇到第一个非零值
                    pre_num_da = i
                    none_zero_a = 1
                else: # 遇到第二个非零值,触发均匀覆盖
                    length_a = i - pre_num_da
                
                    if length_a == 0: continue # 防止除零

                    val = round(delta_groups[pre_num_da][3] / length_a,4)

                    # 覆盖 pre_num 及其后的 length_a 个格子
                    for j in range(length_a):
                        delta_groups[pre_num_da+j][3] = val

                    # 重置状态,当前值作为新的起点
                    none_zero_a = 1
                    pre_num_da = i
            elif i == total_len-1 and none_zero_a == 1: # 如果这是最后一个非零值
                length_a = i - pre_num_da

                if length_a == 0: continue # 防止除零

                val = round(delta_groups[pre_num_da][3] / length_a,4)

                for j in range(length_a):
                    delta_groups[pre_num_da+j][3] = val

            # --- 处理 B 列 (索引 4) ---
            if item[4] != 0:
                if none_zero_b == 0:  # 第一个非零值
                    pre_num_db = i
                    none_zero_b = 1
                else: # 遇到第二个非零值,触发均匀覆盖
                    length_b = i - pre_num_db
                    if length_b == 0: continue
                    val = round(delta_groups[pre_num_db][4] / length_b,4)
                    for k in range(length_b):
                        delta_groups[pre_num_db+k][4] = val

                    # 重置状态,当前值作为新的起点
                    pre_num_db = i
                    none_zero_b = 1
            elif i == total_len-1 and none_zero_b == 1: # 如果这是最后一个非零值
                length_b = i - pre_num_db

                if length_b == 0: continue # 防止除零

                val = round(delta_groups[pre_num_db][4] / length_b,4)

                for k in range(length_b):
                    delta_groups[pre_num_db+k][4] = val

            # --- 处理 C 列 (索引 5) ---
            if item[5] != 0:
                if none_zero_c == 0:  # 第一个非零值
                    pre_num_dc = i
                    none_zero_c = 1
                else: # 第二个非零值
                    length_c = i - pre_num_dc

                    if length_c == 0: continue
                    val = round(delta_groups[pre_num_dc][5] / length_c,4)
                    for l in range(length_c):
                        delta_groups[pre_num_dc+l][5] = val

                    pre_num_dc = i
                    none_zero_c = 1 # 重置当前作为新的第一个非零值

            elif i == total_len-1 and none_zero_c == 1:
                length_c = i - pre_num_dc

                if length_c == 0: continue
                val = round(delta_groups[pre_num_dc][5] / length_c,4)
                for l in range(length_c):
                    delta_groups[pre_num_dc+l][5] = val

        for i in range(total_len):
            print(f"dx: {delta_groups[i][0]}; dy: {delta_groups[i][1]}; dz: {delta_groups[i][2]}; da: {delta_groups[i][3]}; db: {delta_groups[i][4]}; dc: {delta_groups[i][5]}\n")

        # --- 打包过程 ---

        # 1. 准备头部: 总组数 (无符号整型,4 字节)
        count = len(delta_groups)

        # '<I' 表示: 小端序,无符号整型 (Unsigned Int, 4 字节)
        header = struct.pack('<I',count)

        # 2. 优化数据体
        flat_data = list(itertools.chain.from_iterable(delta_groups))

        # 动态构建格式字符串:例如 100 组 * 6 个 = 600 个 float -> '<600f'
        format_str = f'{count*6}f'

        body = struct.pack(format_str,*flat_data)

        # 3. 合并
        binary_data = header + body

        print(f"总大小: {len(binary_data)} 字节")

        # 4. 更新传输的数据
        # data_string = binary_data.decode('latin-1')
        # self.pathStringNode.SetText(data_string)
        import base64
        text_data = base64.b64encode(binary_data).decode('ascii')
        self.pathStringNode.SetText(text_data)
        # self.pathStringNode.SetText(binary_data)

        # 5. 触发更新
        self.pathStringNode.Modified()

    def showError(self,message)->None:
        slicer.util.errorDisplay(message)

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.onDisconnect()
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: RSParameterNode | None) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            # self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        pass

    def onApplyButton(self) -> None:
        pass


#
# RSLogic
#


class RSLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return RSParameterNode(super().getParameterNode())

