###UI  
界面UI文件：annotationUI.ui #主界面 cocoUI.ui #输出coco文件弹出界面  
pyuic5 -o annotationUI.py annotationUI.ui 生成显示UI框架的py文件  
python client.py启动标注器client  

###服务器  
openpose服务器：openposeServer.py  
python openposeServer.py 等待命令行提示"Server all setup!!"后方可开始接收图片返回人体姿态预测结果  

###grpc  
grpc:  
格式协议：openpose.proto  
python -m grpc_tools.protoc -I ./  --python_out=. --grpc_python_out=.  ./openpose.proto  

格式协议：raft.proto  
python -m grpc_tools.protoc -I ./  --python_out=. --grpc_python_out=.  ./raft.proto  


###Summary  
client界面及功能： annotationUI.ui annotationUI.py/ cocoUI.ui cocoUI.py/ client.py  
client依赖功能文件： fileIO.py undistortion.py coco.py  
openpose服务器文件： openpose.proto openpose_pb2.py openpose_pb2_grpc.py openposeServer.py  
openpose依赖算法文件及模型： ./openpose/  
