# 性别识别api




#### 使用教程

1.  运行api.py，前提是有下载net.pkl
2.  如需自己训练，操作与[我这个工程](https://gitee.com/KareEnges/pytorch-CNN-SBATM)相同，pkl是训练好的
3.  启动后访问127.0.0.1:5000


#### 目录说明

1.  static和templates是存储flask文件的
2.  out是存放训练模型的（要使用模型请把out里的net *** .pkl改为net.pkl并移到根目录
3.  train是放训练集的
4.  runs里面是训练的tensorbosrd数据

