# 标注工具使用手册
作者 谢晨成 [z5237028@ad.unsw.edu.au](mailto:z5237028@ad.unsw.edu.au)

<img src="https://github.com/melmarsezio/Annotation-Tool/blob/master/siemens.png" width = "300" height = "300" alt="siemens" align=center />

**组成部分：Client, Openpose Server, RAFT Server.**

## 目录
1. [Purpose](#Purpose)
2. [流程图 Flowchart](#Flowchart)
3. [Openpose Server](#OpenposeServer)
4. [RAFT Server](#RAFTServer)
5. [Client](#Client)
6. [Function list](#FunctionList)
7. [Sample setup](#SampleSetup)

--------
<span id="Purpose"></span>
## 1.Purpose
&nbsp;&nbsp;&nbsp;&nbsp;Semi-automatic annotation tool on massive volumn of CT pose videoclips (not CT scan images), generates key points annotations align with the Common Objects in Context form ("COCO" from the Microsoft).  
&nbsp;&nbsp;&nbsp;&nbsp;Expected usage: After training with these annotation datas, CT machine can automatically allocate patients body parts without the intervention of physician.

--------
<span id="Flowchart"></span>
## 2.流程图 Flowchart

![Flowchart](https://github.com/melmarsezio/Annotation-Tool/blob/master/Flowchart.png "Flowchart")

--------
<span id="OpenposeServer"></span>
## 3.Openpose Server
+ File location: `intern@10.10.192.40:/home/intern/OPServer`
+ `python openposeServer.py --port 60001` (default --port 50051)

Openpose Server使用了CMU开发的人体姿态识别项目openpose，github开源地址：[CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose "CMU-Perceptual-Computing-Lab/openpose"), 论文链接：[OpenPose: Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields](https://github.com/melmarsezio/Annotation-Tool/blob/master/OPServer/OpenPose%20Realtime%20Multi-Person%202D%20Pose%20Estimation%20using%20Part%20Affinity%20Fields.pdf "OpenPose: Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields"), 论文介绍：[【人体姿态识别】 Openpose论文](https://zhuanlan.zhihu.com/p/48507352 "【人体姿态识别】 Openpose论文").

> OpenPose人体姿态识别项目是美国卡耐基梅隆大学（CMU）基于卷积神经网络和监督学习并以caffe为框架开发的开源库。可以实现人体动作、面部表情、手指运动等姿态估计。适用于单人和多人，具有极好的鲁棒性。是世界上首个基于深度学习的实时多人二维姿态估计应用，基于它的实例如雨后春笋般涌现。人体姿态估计技术在体育健身、动作采集、3D试衣、舆情监测等领域具有广阔的应用前景，人们更加熟悉的应用就是抖音尬舞机。

Openpose Server利用grpc协议将client发送的图片进行实时分析，并传回分析结果(关键点坐标)

--------
<span id="RAFTServer"></span>
## 4.RAFT Server
+ File location: `intern@10.10.192.40:/home/intern/RAFTServer`
+ `python raftServer.py --model raft/raft-small.pth --dir /home/intern/RAFTavi/ --port 60000` (default --port 50000)

RAFT Server使用了Zachary Teed 和 Jia Deng发表的RAFT: Recurrent All-Pairs Field Transforms for Optical Flow论文和开源模型代码. 论文链接：[RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://github.com/melmarsezio/Annotation-Tool/blob/master/RAFTServer/RAFT%20Recurrent%20All-Pairs%20Field%20Transforms%20for%20Optical%20Flow.pdf "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow")，论文介绍：[ECCV 2020最佳论文讲了啥？作者为ImageNet一作、李飞飞高徒邓嘉](https://zhuanlan.zhihu.com/p/205020999 "ECCV 2020最佳论文讲了啥？作者为ImageNet一作、李飞飞高徒邓嘉")，github开源模型地址：[princeton-vl/RAFT](https://github.com/CMU-Perceptual-Computing-Lab/openpose "princeton-vl/RAFT").

> RAFT由三个主要组件组成:
> + 特征编码器，该编码器为每个像素提取特征向量;
> + 相关层，对所有像素对产生4D相关卷，后续池化产生较低分辨率卷
> + 基于gru的循环更新操作符，从相关卷中检索值，迭代更新初始化为0的流场字段。

> 注：特征编码器用于提取每个像素的特征，相关层计算像素之间的视觉相似性，更新操作符模拟了迭代优化算法的步骤。
> RAFT模型有准确率高、泛化性好，效率高等特点。

RAFT Server利用grpc协议监听的信息有两种：
+ 视频文件名，必须提前由`~/RAFTServer/VideoDDP.py` 计算出结果并保存在指定路径，Server根据文件名，替换后缀名为`.avi`后查找相应文件并传回
+ 两帧图像，Server收到两帧图像后计算两帧之间每个像素点的运动向量并传回

--------
<span id="Client"></span>
## 5.Client
+ File location:`139.24.206.169: D:\Annotation Tool`
+ `python client.py` default OP port 60001 RAFT port 60000
+ `D:\Annotation Tool\UI` UI related files of this annotation tool: Use `Qt Designer` to design and generate `*.ui` file, than `pyuic5 -o *.py *.ui` to compile `*.ui` file into `*.py` which can be imported by main `client.py`
+ `D:\Annotation Tool\GRPC` grpc protocol related files. Message types are defined in `*.proto`, than `python -m grpc_tools.protoc -I ./  --python_out=. --grpc_python_out=.  ./*.proto` to compile into `*_pb2.py` and `*_pb2_grpc.py`. These two file should be imported by both client and server.
+ `D:\Annotation Tool\core` configuration files, utils files and algorithm files besides main `client.py`.

### Work flow in client
&nbsp;&nbsp; 0.`Ctrl+U` Load `*.yml` file for undistortion configuration _(optional, but recommanded)_

1. read material via one of below method:
   + `Ctrl+V` Load Video. _(support `*.mpeg *.mov *.wmv *.rmvb *.flv *.mp4 *.avi`)_
   + `Ctrl+I` Directly Load Images. _(disabled. Currently we design to convert video into `class Video` inherit from cv2.VideoCapture to avoid memory exceed, defined in `D:\Annotation Tool\core\fileIO.Video`. This design conflicts with Image input and need extra deisgn/options)_
   + `Ctrl+A` Load `*.annot` file which can resume from last saved points. _(Be careful with the directory of the video, assume unchanged, re-configurate if changed, still under development)_

2. obtain Key points we want to propogate via one of below method:
   + manually add key points by hold `A` and click on image.
   + Click `OPENPOSE` bottom, call server to estimate all keypoints for you.

3. adjust inaccurate keypoint location by two step:
   1. click on the key point to select.
   2. hold `Ctrl` and click new location to move it/ or press `D` to delete that point.

4. request for full RAFT result via one of below method:
   + `Ctrl+R` Load `*.avi` RAFT file. _(similar to video, converted into `class RAFT` inherit from cv2.VideoCapture, autometically remove third channel and take care of offsets and precision. The directory of RAFT file is assumed unchanged, re-configurate if changed, still under development)_
   + by click `Get RAFT` bottom. Receive `*.avi` file and saved in root directory of `client.py` If video filename is not matched with any RAFT result on server, message status shows `fail` from the server.
   Can not proceed to next step until RAFT result is loaded!

5. propogate keypoints using RAFT result by click `Propogate` bottom and select end page#. The client will propogate keypoints from current page to selected page. _(if did not load RAFT file, by click `Propogate` it will autometically call server to `Get RAFT` first)_

6. adjust any inaccurate keypoints in between and repropogate.

7. `Ctrl+S` save `*.annot` annotation file anytime you feel like or `Ctrl+Shift+S` to save as new file. _(optional, but recommanded)_

8. `Ctrl+C` Save keypoints results to `*.json` file in coco from.


--------
<span id="FunctionList"></span>
## 6.Function list
- [x] Load Video
- [x] Load Images
- [x] Load `*.annot` file
- [x] 设置各文件的默认位置
- [x] 设置边界（定义视野盲区）
- [ ] 批量控制关键点（锁死范围内的某几种关键点）
- [x] 检查RAFT文件是否和视频文件匹配
- [ ] 过滤过曝帧和异常帧
   - [ ] 显示Intensity窗口，由用户自主选择过曝范围，或
   - [ ] 设置阈值，自动过滤过曝帧
- [x] COCO格式输出

--------
<span id="SampleSetup"></span>
### 7.Sample setup
Server端-1 (Openpose Server):

```
ssh intern@10.10.192.40
ENTER PASSWORD
cd OPServer/
conda activate torch
python3 openposeServer.py --port 60000
```

Server端-2 (RAFT Server):

```
ssh intern@10.10.192.40
ENTER PASSWORD
cd RAFTServer/
conda activate torch
python3 raftServer.py --port 60001
```

Client端:

```
activate cv
D:
cd Annotation Tool
python client.py
```
--------

如有任何问题，可邮件至[z5237028@ad.unsw.edu.au](mailto:z5237028@ad.unsw.edu.au)
