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

        # 默认姿态(四元数)
        default_quat = [0.0,0.0,0.0,1.0]

        if pointsNum > 0:
            for i in range(pointsNum):
                points = [0.0,0.0,0.0]
                pathNode.GetNthControlPointPositionWorld(i,points)
                self.pointsSet.append((points,default_quat))
        else:
            print("没有足够的点位进行计算\n")
            return

        print("已获取路径点位")

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
        if norm == 0:return 0.0,0.0,0.0
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

        except Exception as e:
            slicer.util.errorDisplay(f"获取变换矩阵失败: {e}")
            return

        print(f"[INFO] 开始处理 {len(self.pointsSet)} 个路径点并生成二进制流...")

        # 临时存储转换后的绝对坐标,用于增量计算
        abs_points = []

        # 2. 第一遍遍历: 坐标变换 & 四元数转 ABC
        for i,(pos,quat) in enumerate(self.pointsSet):
            # A. 坐标变换 (Slicer -> KUKA)
            kuka_pos,kuka_quat = self.apply_coordinate_transform(pos,quat,current_transform_matrix)

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
                print(f"dx: {dx}; dy: {dy}; dz: {dz}; da: {da}; db: {db}; dc: {dc}\n")
            else:
                prev_pt = abs_points[i-1]

                dx = round(pt['x']-prev_pt['x'],4)
                dy = round(pt['y']-prev_pt['y'],4)
                dz = round(pt['z']-prev_pt['z'],4)
                da = round(pt['a']-prev_pt['a'],4)
                db = round(pt['b']-prev_pt['b'],4)
                dc = round(pt['c']-prev_pt['c'],4)
                print(f"dx: {dx}; dy: {dy}; dz: {dz}; da: {da}; db: {db}; dc: {dc}\n")
            delta_groups.append((dx,dy,dz,da,db,dc))

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

