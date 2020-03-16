# CV_Stu
⏰ 计算机视觉学习总结、资料积累

## 1. Semantic Segmentation
💡 语义分割经典网络idea总结
### 1.1 FCN

- **全连接层转换成卷积层**
- **不同尺度下的信息融合: FCN-8S、16s、32s（skip connection）**

### 1.2 U-Net

- **Encoder-Decoder结构**
  - 前半部分为多层卷积池化，不断扩大感受野，用于提取特征。后半部分上采样恢复图片尺寸
- **更丰富的信息融合**
  - 更多的前后层之间的信息融合
  - 在串联之前，需要把前层的feature map crop到和后层一样的大小

### 1.3 SegNet

- **Encoder-Decoder结构**
  - 没有直接融合不同尺度的层的信息
  - 使用了带有坐标（index）的池化。在Max pooling时，选择最大像素的同时，记录下该像素在Feature map的位置。在反池化的时候，根据记录的坐标，把最大值复原到原来对应的位置，其他的位置补零。后面的卷积可以把0的元素给填上。这样一来，就解决了由于多次池化造成的位置信息的丢失

### 1.4 Deeplab v1

- **带孔卷积（Atrous conv）**
  - 不增加参数量的情况下增大感受野
- **CRF(条件随机场)**
  - 利用像素之间的关连信息：相邻的像素，或者颜色相近的像素有更大的可能属于同一个class

### 1.5 PSPNet

- **空间金字塔池化**
  - 金字塔池化得到一组感受野大小不同的feature map
  - 将这些感受野不同的map concat到一起，完成多层次的语义特征融合

### 1.6 Deeplab v2

-  在v1的基础上，**引入了ASPP(Atrous Spatial Pyramid Pooling)结构**
  - 选择不同扩张率的带孔卷积去处理Feature Map
  - ASPP层把这些不同层级的feature map concat到一起，进行信息融合

### 1.7  Deeplab v3

- **改进了ASPP模块**
  - 加入了BatchNorm
  - 加入特征的全局平均池化，**是对全局特征的强调、加强**
- **引入Resnet Block**

- **丢弃CRF**

### 1.8 Deeplab v3+

- **把Deeplab v3作为encoder**
- **decoder的过程中在此运用了不同层级特征的融合**
 
