# CV_Stu
⏰ 计算机视觉学习总结、资料积累

## 1. 2D Semantic Segmentation Idea 

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
 
## 2. 2D Semantic Segmentation Paper&Code

- [ ] [2019-CVPR oral] CLAN: Category-level Adversaries for Semantics Consistent [[`paper`]](https://arxiv.org/abs/1809.09478?context=cs) [[`code`]](https://github.com/RoyalVane/CLAN)

- [ ] [2019-CVPR] BRS: Interactive Image Segmentation via Backpropagating Refinement Scheme(***) [[`paper`]](https://vcg.seas.harvard.edu/publications/interactive-image-segmentation-via-backpropagating-refinement-scheme/paper) [[`code`]](https://github.com/wdjang/BRS-Interactive_segmentation)

- [x] [2019-CVPR] DFANet：Deep Feature Aggregation for Real-Time Semantic Segmentation(used in camera) [[`paper`]](https://share.weiyun.com/5NgHbWH) [[`code`]](https://github.com/j-a-lin/DFANet_PyTorch)

- [ ] [2019-CVPR] DeepCO3: Deep Instance Co-segmentation by Co-peak Search and Co-saliency [[`paper`]](http://cvlab.citi.sinica.edu.tw/images/paper/cvpr-hsu19.pdf) [[`code`]](https://github.com/KuangJuiHsu/DeepCO3)

- [ ] [2019-CVPR] Domain Adaptation(reducing the domain shif) [[`paper`]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Luo_Taking_a_Closer_Look_at_Domain_Shift_Category-Level_Adversaries_for_CVPR_2019_paper.pdf) 

- [ ] [2019-CVPR] ELKPPNet: An Edge-aware Neural Network with Large Kernel Pyramid Pooling for Learning Discriminative Features in Semantic- Segmentation [[`paper`]](https://arxiv.org/abs/1906.11428) [[`code`]](https://github.com/XianweiZheng104/ELKPPNet)

- [ ] [2019-CVPR oral] GLNet: Collaborative Global-Local Networks for Memory-Efficient Segmentation of Ultra-High Resolution Images[[`paper`]](https://arxiv.org/abs/1905.06368) [[`code`]](https://github.com/chenwydj/ultra_high_resolution_segmentation)

- [ ] [2019-CVPR] Instance Segmentation by Jointly Optimizing Spatial Embeddings and Clustering Bandwidth(***SOTA) [[`paper`]](https://arxiv.org/abs/1906.11109) [[`code`]](https://github.com/davyneven/SpatialEmbeddings)

- [ ] [2019-ECCV] ICNet: Real-Time Semantic Segmentation on High-Resolution Images [[`paper`]](https://arxiv.org/abs/1704.08545) [[`code`]](https://github.com/oandrienko/fast-semantic-segmentation)

- [ ] [2019-CVPR] LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation(***SOTA) [[`paper`]](https://arxiv.org/abs/1905.02423) [[`code`]](https://github.com/xiaoyufenfei/LEDNet)

- [ ] [2019-arXiv] LightNet++: Boosted Light-weighted Networks for Real-time Semantic Segmentation [[`paper`]](http://arxiv.org/abs/1605.02766) [[`code`]](https://github.com/ansleliu/LightNetPlusPlus)

- [ ] [2019-CVPR] PTSNet: A Cascaded Network for Video Object Segmentation [[`paper`]](https://arxiv.org/abs/1907.01203) [[`code`]](https://github.com/sydney0zq/PTSNet)

- [ ] [2019-CVPR] PPGNet: Learning Point-Pair Graph for Line Segment Detection [[`paper`]](https://www.aiyoggle.me/publication/ppgnet-cvpr19/ppgnet-cvpr19.pdf) [[`code`]](https://github.com/svip-lab/PPGNet)

- [ ] [2019-CVPR] Show, Match and Segment: Joint Learning of Semantic Matching and Object Co-segmentation [[`paper`]](https://arxiv.org/abs/1906.05857) [[`code`]](https://github.com/YunChunChen/MaCoSNet-pytorch)

- [ ] [2019-CVPR] Video Instance Segmentation [[`paper`]](https://arxiv.org/abs/1905.04804) [[`code`]](https://github.com/youtubevos/MaskTrackRCNN)
- [ ] Arxiv-2018 ExFuse: Enhancing Feature Fusion for Semantic Segmentation 87.9% mean Iou->voc2012 [[Paper]](https://arxiv.org/pdf/1804.03821.pdf)
- [ ] CVPR-2018 spotlight Learning to Adapt Structured Output Space for Semantic Segmentation  [[Paper]](https://arxiv.org/abs/1802.10349) [[Code]](https://github.com/wasidennis/AdaptSegNet)
- [ ] Arfix-2018 Adversarial Learning for Semi-supervised Semantic Segmentation [[Paper]](https://arxiv.org/abs/1802.07934) [[Code]](https://github.com/hfslyc/AdvSemiSeg)
- [ ] Arxiv-2018 Context Encoding for Semantic Segmentation [[Paper]](https://arxiv.org/pdf/1803.08904.pdf) [[Code]](https://github.com/zhanghang1989/MXNet-Gluon-SyncBN)
- [ ] CVPR-2018 Learning to Adapt Structured Output Space for Semantic Segmentation [[Paper]](https://arxiv.org/abs/1802.10349)[[Code]](https://github.com/wasidennis/AdaptSegNet)
- [ ] CVPR-2018 Dynamic-structured Semantic Propagation Network [[Paper]](https://arxiv.org/abs/1803.06067)
- [x] Deeplab v4: Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation [[Paper]](https://arxiv.org/pdf/1802.02611.pdf) [[Code]](https://github.com/tensorflow/models/tree/master/research/deeplab)
- [ ] Deep Value Networks Learn to Evaluate and Iteratively Refine Structured Outputs [[Paper]](https://arxiv.org/pdf/1703.04363.pdf)[[Code]](https://github.com/gyglim/dvn)
- [ ] ICCV-2017 Semantic Line Detection and Its Applications [[Paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Lee_Semantic_Line_Detection_ICCV_2017_paper.pdf)
- [ ] ICCV-2017 Attentive Semantic Video Generation Using Captions [[Paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Marwah_Attentive_Semantic_Video_ICCV_2017_paper.pdf)
- [ ] ICCV-2017 BlitzNet: A Real-Time Deep Network for Scene Understanding [[Paper]](https://arxiv.org/pdf/1708.02813.pdf) [[Code]](https://github.com/dvornikita/blitznet)
- [ ] ICCV-2017 SCNet: Learning Semantic Correspondence   [[Code]](https://github.com/k-han/SCNet)
- [ ] CVPR-2017 End-to-End Instance Segmentation with Recurrent Attention [[Code]](https://github.com/renmengye/rec-attend-public)
- [ ] CVPR-2017 Deep Watershed Transform for Instance Segmentation [[Code]](https://github.com/min2209/dwt)
- [ ] Piecewise Flat Embedding for Image Segmentation [[Paper]](https://pdfs.semanticscholar.org/4690/3c0ca5540e312b8f4c20c012f586e5071914.pdf)
- [ ] ICCV-2017 Curriculum Domain Adaptation for Semantic Segmentation of Urban Scenes [[Paper]](https://arxiv.org/abs/1707.09465)[[Code]](https://github.com/YangZhang4065/AdaptationSeg)
- [ ] CVPR-2017 Not All Pixels Are Equal: Difficulty-Aware Semantic Segmentation via Deep Layer Cascade-2017 [[Paper]](https://arxiv.org/abs/1704.01344)
- [ ] CVPR-2017 Annotating Object Instances with a Polygon-RNN-2017 [[Project]](http://www.cs.toronto.edu/polyrnn/) [[Paper]](https://arxiv.org/abs/1704.05548)
- [ ] CVPR-2017  Loss maxpooling for semantic image segmentation [[Paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Bulo_Loss_Max-Pooling_for_CVPR_2017_paper.pdf)
- [ ] ICCV-2017 Scale-adaptive convolutions for scene parsing [[Paper]](https://pdfs.semanticscholar.org/f984/781ccef66e6a6114707271b8bb29148ad45d.pdf)
- [ ] Towards End-to-End Lane Detection: an Instance Segmentation Approach [[Paper]](https://arxiv.org/pdf/1802.05591.pdf)arxiv-1802
- [ ] AAAI-2018 Mix-and-Match Tuning for Self-Supervised Semantic Segmentation [[Paper]](https://arxiv.org/pdf/1712.00661.pdf) arxiv-1712
- [ ] NIPS-2017-Learning Affinity via Spatial Propagation Networks [[Paper]](https://papers.nips.cc/paper/6750-learning-affinity-via-spatial-propagation-networks.pdf)
- [ ] AAAI-2018-Spatial As Deep: Spatial CNN for Traffic Scene Understanding [[Paper]](https://arxiv.org/pdf/1712.06080.pdf)
- [ ] Stacked Deconvolutional Network for Semantic Segmentation-2017 [[Paper]](https://arxiv.org/pdf/1708.04943.pdf)</br>
- [x] Deeplab v3: Rethinking Atrous Convolution for Semantic Image Segmentation-2017(DeeplabV3) [[Paper]](https://arxiv.org/pdf/1706.05587.pdf)</br>
- [ ] CVPR-2017 Learning Object Interactions and Descriptions for Semantic Image Segmentation-2017 [[Paper]](http://personal.ie.cuhk.edu.hk/~pluo/pdf/wangLLWcvpr17.pdf)</br>
- [ ] Pixel Deconvolutional Networks-2017 [[Code-Tensorflow]](https://github.com/HongyangGao/PixelDCN) [[Paper]](https://arxiv.org/abs/1705.06820)</br>
- [ ] Dilated Residual Networks-2017 [[Paper]](http://vladlen.info/papers/DRN.pdf)</br>
- [ ] A Review on Deep Learning Techniques Applied to Semantic Segmentation-2017 [[Paper]](https://arxiv.org/abs/1704.06857)</br>
- [ ] BiSeg: Simultaneous Instance Segmentation and Semantic Segmentation with Fully Convolutional Networks [[Paper]](https://arxiv.org/abs/1706.02135)</br>
- [ ] ICNet for Real-Time Semantic Segmentation on High-Resolution Images-2017 [[Project]](https://hszhao.github.io/projects/icnet/) [[Code]](https://github.com/hszhao/ICNet) [[Paper]](https://arxiv.org/abs/1704.08545) [[Video]](https://www.youtube.com/watch?v=qWl9idsCuLQ)</br>
- [ ]  Feature Forwarding: Exploiting Encoder Representations for Efficient Semantic Segmentation-2017 [[Project]](https://codeac29.github.io/projects/linknet/) [[Code-Torch7]](https://github.com/e-lab/LinkNet)</br>
- [ ]  Reformulating Level Sets as Deep Recurrent Neural Network Approach to Semantic Segmentation-2017 [[Paper]](https://arxiv.org/abs/1704.03593)</br>
- [ ]  Adversarial Examples for Semantic Image Segmentation-2017 [[Paper]](https://arxiv.org/abs/1703.01101)</br>
- [ ]  Large Kernel Matters - Improve Semantic Segmentation by Global Convolutional Network-2017 [[Paper]](https://arxiv.org/abs/1703.02719)</br>
- [ ] HyperNet: Towards Accurate Region Proposal Generation and Joint Object Detection [[Paper]](https://arxiv.org/pdf/1604.00600)
- [ ] Hypercolumns for Object Segmentation and Fine-grained Localization [[Paper]](https://arxiv.org/pdf/1411.5752)
- [ ] Matching-CNN meets KNN: Quasi-parametric human parsing[[Paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1B_034_ext.pdf)
- [ ] Deep Human Parsing with Active Template Regression [[Paper]](https://arxiv.org/pdf/1503.02391.pdf)
- [ ] TPAMI-2012 **Learning Hierarchical Features for Scene Labeling** The first paper for applying dl on semantic segmentation !!! [[Paper]](http://yann.lecun.com/exdb/publis/pdf/farabet-pami-13.pdf)
- [ ]  Label Refinement Network for Coarse-to-Fine Semantic Segmentation-2017 [[Paper]](https://www.arxiv.org/abs/1703.00551)
- [ ] Laplacian Pyramid Reconstruction and Refinement for Semantic Segmentation [[Paper]](https://arxiv.org/pdf/1605.02264.pdf)
- [ ] ParseNet: Looking Wider to See Better [[Paper]](https://www.cs.unc.edu/~wliu/papers/parsenet.pdf)
- [ ] CVPR-2016 Recombinator Networks: Learning Coarse-to-Fine Feature Aggregation [[Paper]](http://openaccess.thecvf.com/content_cvpr_2016/papers/Honari_Recombinator_Networks_Learning_CVPR_2016_paper.pdf)
- [ ]  **PixelNet: Representation of the pixels, by the pixels, and for the pixels-2017** [[Project]](http://www.cs.cmu.edu/~aayushb/pixelNet/) [[Code-Caffe]](https://github.com/aayushbansal/PixelNet) [[Paper]](https://arxiv.org/abs/1702.06506)</br>
- [ ]  LabelBank: Revisiting Global Perspectives for Semantic Segmentation-2017 [[Paper]](https://arxiv.org/abs/1703.09891)</br>
- [ ]  Progressively Diffused Networks for Semantic Image Segmentation-2017 [[Paper]](https://arxiv.org/abs/1702.05839)</br>
- [ ]  Understanding Convolution for Semantic Segmentation-2017 [[Model-Mxnet]](https://drive.google.com/drive/folders/0B72xLTlRb0SoREhISlhibFZTRmM) [[Paper]](https://arxiv.org/abs/1702.08502) [[Code]](https://github.com/TuSimple/TuSimple-DUC)</br>
- [ ]  ICCV-2017 Predicting Deeper into the Future of Semantic Segmentation-2017 [[Paper]](https://arxiv.org/abs/1703.07684)</br>
- [x]  CVPR-2017 **Pyramid Scene Parsing Network-2017** [[Project]](https://hszhao.github.io/projects/pspnet/) [[Code-Caffe]](https://github.com/hszhao/PSPNet) [[Paper]](https://arxiv.org/abs/1612.01105) [[Slides]](http://image-net.org/challenges/talks/2016/SenseCUSceneParsing.pdf)</br>
- [x]  FCNs in the Wild: Pixel-level Adversarial and Constraint-based Adaptation-2016 [[Paper]](https://arxiv.org/abs/1612.02649)</br>
- [ ]  FusionNet: A deep fully residual convolutional neural network for image segmentation in connectomics-2016 [[Code-PyTorch]](https://github.com/GunhoChoi/FusionNet_Pytorch) [[Paper]](https://arxiv.org/abs/1612.05360)</br>
- [ ]  RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation-2016 [[Code-MatConvNet]](https://github.com/guosheng/refinenet) [[Paper]](https://arxiv.org/abs/1611.06612)</br>
- [ ]  CVPRW-2017 The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation [[Code-Theano]](https://github.com/SimJeg/FC-DenseNet) [[Code-Keras1]](https://github.com/titu1994/Fully-Connected-DenseNets-Semantic-Segmentation) [[Code-Keras2]](https://github.com/0bserver07/One-Hundred-Layers-Tiramisu) [[Paper]](https://arxiv.org/abs/1611.09326)</br>
- [ ]  CVPR-2017 Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes [[Code-Theano]](https://github.com/TobyPDE/FRRN) [[Paper]](https://arxiv.org/abs/1611.08323)</br>
- [ ]  PixelNet: Towards a General Pixel-level Architecture-2016 [[Paper]](http://arxiv.org/abs/1609.06694)</br>
- [ ]  Recalling Holistic Information for Semantic Segmentation-2016 [[Paper]](https://arxiv.org/abs/1611.08061)</br>
- [ ]  Semantic Segmentation using Adversarial Networks-2016 [[Paper]](https://arxiv.org/abs/1611.08408) [[Code-Chainer]](https://github.com/oyam/Semantic-Segmentation-using-Adversarial-Networks)</br>
- [ ]  Region-based semantic segmentation with end-to-end training-2016 [[Paper]](http://arxiv.org/abs/1607.07671)</br>
- [ ]  Exploring Context with Deep Structured models for Semantic Segmentation-2016 [[Paper]](https://arxiv.org/abs/1603.03183)</
- [ ] **Multi-scale context aggregation by dilated convolutions** [[Paper]](https://arxiv.org/pdf/1511.07122.pdf)
- [ ]  Better Image Segmentation by Exploiting Dense Semantic Predictions-2016 [[Paper]](https://arxiv.org/abs/1606.01481)</br>
- [ ] Boundary-aware Instance Segmentation-2016 [[Paper]](https://infoscience.epfl.ch/record/227439/files/HayderHeSalzmannCVPR17.pdf)</br>
- [ ] Improving Fully Convolution Network for Semantic Segmentation-2016 [[Paper]](https://arxiv.org/abs/1611.08986)</br>
- [ ] Deep Structured Features for Semantic Segmentation-2016 [[Paper]](https://arxiv.org/abs/1609.07916)</br>
- [x] DeepLab v2:Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs-2016** [[Project]](http://liangchiehchen.com/projects/DeepLab.html) [[Code-Caffe]](https://bitbucket.org/deeplab/deeplab-public/) [[Code-Tensorflow]](https://github.com/DrSleep/tensorflow-deeplab-resnet) [[Code-PyTorch]](https://github.com/isht7/pytorch-deeplab-resnet) [[Paper]](https://arxiv.org/abs/1606.00915)</br>
- [x] DeepLab v1: Semantic Image Segmentation With Deep Convolutional Nets and Fully Connected CRFs-2014** [[Code-Caffe1]](https://bitbucket.org/deeplab/deeplab-public/) [[Code-Caffe2]](https://github.com/TheLegendAli/DeepLab-Context) [[Paper]](http://arxiv.org/abs/1412.7062)</br>
- [ ] Deep Learning Markov Random Field for Semantic Segmentation-2016 [[Project]](http://personal.ie.cuhk.edu.hk/~lz013/projects/DPN.html) [[Paper]](https://arxiv.org/abs/1606.07230)</br>
- [ ] ECCV2016 Salient Deconvolutional Networks  [[Code]](https://github.com/aravindhm/deconvnet_analysis)
- [ ]  Convolutional Random Walk Networks for Semantic Image Segmentation-2016 [[Paper]](https://arxiv.org/abs/1605.07681)</br>
- [ ]  ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation-2016 [[Code-Caffe1]](https://github.com/e-lab/ENet-training)[[Code-Caffe2]](https://github.com/TimoSaemann/ENet) [[Paper]](https://arxiv.org/abs/1606.02147) [[Blog]](https://culurciello.github.io/tech/2016/06/20/training-enet.html)</br>
- [ ]  High-performance Semantic Segmentation Using Very Deep Fully Convolutional Networks-2016 [[Paper]](https://arxiv.org/abs/1604.04339)</br>
- [ ]  CVPR-2016-oral ScribbleSup: Scribble-Supervised Convolutional Networks for Semantic Segmentation-2016 [[Paper]](http://arxiv.org/abs/1604.05144)</br>
- [ ]  Object Boundary Guided Semantic Segmentation-2016 [[Code-Caffe]](https://github.com/twtygqyy/obg_fcn) [[Paper]](http://arxiv.org/abs/1603.09742)</br>
- [ ]  Segmentation from Natural Language Expressions-2016 [[Project]](http://ronghanghu.com/text_objseg/) [[Code-Tensorflow]](https://github.com/ronghanghu/text_objseg) [[Code-Caffe]](https://github.com/Seth-Park/text_objseg_caffe) [[Paper]](http://arxiv.org/abs/1603.06180)</br>
- [ ]  Seed, Expand and Constrain: Three Principles for Weakly-Supervised Image Segmentation-2016 [[Code-Caffe]](https://github.com/kolesman/SEC) [[Paper]](https://arxiv.org/abs/1603.06098)</br>
- [ ]  Global Deconvolutional Networks for Semantic Segmentation-2016 [[Paper]](https://arxiv.org/abs/1602.03930)</br>
- [ ]  Learning Transferrable Knowledge for Semantic Segmentation with Deep Convolutional Neural Network-2015 [[Project]](http://cvlab.postech.ac.kr/research/transfernet/) [[Code-Caffe]](https://github.com/maga33/TransferNet) [[Paper]](http://arxiv.org/abs/1512.07928)</br>
- [ ]  Learning Dense Convolutional Embeddings for Semantic Segmentation-2015 [[Paper]](https://arxiv.org/abs/1511.04377)</br>
- [ ]  ParseNet: Looking Wider to See Better-2015 [[Code-Caffe]](https://github.com/weiliu89/caffe/tree/fcn) [[Model-Caffe]](https://github.com/BVLC/caffe/wiki/Model-Zoo#parsenet-looking-wider-to-see-better) [[Paper]](http://arxiv.org/abs/1506.04579)</br>
- [ ]  Decoupled Deep Neural Network for Semi-supervised Semantic Segmentation-2015 [[Project]](http://cvlab.postech.ac.kr/research/decouplednet/) [[Code-Caffe]](https://github.com/HyeonwooNoh/DecoupledNet) [[Paper]](http://arxiv.org/abs/1506.04924)</br>
- [ ] Bayesian segnet: Model uncertainty in deep convolutional encoder-decoder architectures for scene understanding [[Paper]](https://arxiv.org/pdf/1511.02680.pdf)
- [x]  **SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation-2015** [[Project]](http://mi.eng.cam.ac.uk/projects/segnet/) [[Code-Caffe]](https://github.com/alexgkendall/caffe-segnet) [[Paper]](http://arxiv.org/abs/1511.00561) [[Tutorial1]](http://mi.eng.cam.ac.uk/projects/segnet/tutorial.html) [[Tutorial2]](https://github.com/alexgkendall/SegNet-Tutorial)</br>
- [ ]  Semantic Image Segmentation with Task-Specific Edge Detection Using CNNs and a Discriminatively Trained Domain Transform-2015 [[Paper]](https://arxiv.org/abs/1511.03328)</br>
- [ ]  Semantic Segmentation with Boundary Neural Fields-2015 [[Code]](https://github.com/gberta/BNF_globalization) [[Paper]](https://arxiv.org/abs/1511.02674)</br>
- [ ]  Semantic Image Segmentation via Deep Parsing Network-2015 [[Project]](http://personal.ie.cuhk.edu.hk/~lz013/projects/DPN.html) [[Paper1]](http://arxiv.org/abs/1509.02634) [[Paper2]](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Liu_Semantic_Image_Segmentation_ICCV_2015_paper.pdf) [[Slides]](http://personal.ie.cuhk.edu.hk/~pluo/pdf/presentation_dpn.pdf)</br>
- [ ]  What’s the Point: Semantic Segmentation with Point Supervision-2015 [[Project]](http://vision.stanford.edu/whats_the_point/) [[Code-Caffe]](https://github.com/abearman/whats-the-point1) [[Model-Caffe]](http://vision.stanford.edu/whats_the_point/models.html) [[Paper]](https://arxiv.org/abs/1506.02106)</br>
- [x]  U-Net: Convolutional Networks for Biomedical Image Segmentation-2015 [[Project]](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) [[Code+Data]](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-release-2015-10-02.tar.gz) [[Code-Keras]](https://github.com/orobix/retina-unet) [[Code-Tensorflow]](https://github.com/jakeret/tf_unet) [[Paper]](http://arxiv.org/abs/1505.04597) [[Notes]](http://zongwei.leanote.com/post/Pa)</br>
- [ ]  Learning Deconvolution Network for Semantic Segmentation(DeconvNet)-2015 [[Project]](http://cvlab.postech.ac.kr/research/deconvnet/) [[Code-Caffe]](https://github.com/HyeonwooNoh/DeconvNet) [[Paper]](http://arxiv.org/abs/1505.04366) [[Slides]](http://web.cs.hacettepe.edu.tr/~aykut/classes/spring2016/bil722/slides/w06-deconvnet.pdf)</br>
- [ ]  Multi-scale Context Aggregation by Dilated Convolutions-2015 [[Project]](http://vladlen.info/publications/multi-scale-context-aggregation-by-dilated-convolutions/) [[Code-Caffe]](https://github.com/fyu/dilation) [[Code-Keras]](https://github.com/nicolov/segmentation_keras) [[Paper]](http://arxiv.org/abs/1511.07122) [[Notes]](http://www.inference.vc/dilated-convolutions-and-kronecker-factorisation/)</br>
- [ ]  ReSeg: A Recurrent Neural Network-based Model for Semantic Segmentation-2015 [[Code-Theano]](https://github.com/fvisin/reseg) [[Paper]](https://arxiv.org/abs/1511.07053)</br>
- [ ]  ICCV-2015 BoxSup: Exploiting Bounding Boxes to Supervise Convolutional Networks for Semantic Segmentation-2015 [[Paper]](https://arxiv.org/abs/1503.01640)</br>
- [ ]  Feedforward semantic segmentation with zoom-out features-2015 [[Code]](https://bitbucket.org/m_mostajabi/zoom-out-release) [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Mostajabi_Feedforward_Semantic_Segmentation_2015_CVPR_paper.pdf) [[Video]](https://www.youtube.com/watch?v=HvgvX1LXQa8)</br>
- [ ]  Conditional Random Fields as Recurrent Neural Networks-2015 [[Project]](http://www.robots.ox.ac.uk/~szheng/CRFasRNN.html) [[Code-Caffe1]](https://github.com/torrvision/crfasrnn) [[Code-Caffe2]](https://github.com/martinkersner/train-CRF-RNN) [[Demo]](http://www.robots.ox.ac.uk/~szheng/crfasrnndemo) [[Paper1]](http://www.robots.ox.ac.uk/~szheng/papers/CRFasRNN.pdf) [[Paper2]](http://arxiv.org/abs/1502.03240)</br>
- [ ]  Efficient Piecewise Training of Deep Structured Models for Semantic Segmentation-2015 [[Paper]](https://arxiv.org/abs/1504.01013)</br>
- [x] **Fully Convolutional Networks for Semantic Segmentation-2015** [[Code-Caffe]](https://github.com/shelhamer/fcn.berkeleyvision.org) [[Model-Caffe]](https://github.com/BVLC/caffe/wiki/Model-Zoo#fcn) [[Code-Tensorflow1]](https://github.com/MarvinTeichmann/tensorflow-fcn) [[Code-Tensorflow2]](https://github.com/shekkizh/FCN.tensorflow) [[Code-Chainer]](https://github.com/wkentaro/fcn) [[Code-PyTorch]](https://github.com/wkentaro/pytorch-fcn) [[Paper1]](http://arxiv.org/abs/1411.4038) [[Paper2]](http://arxiv.org/abs/1605.06211) [[Slides1]](https://docs.google.com/presentation/d/1VeWFMpZ8XN7OC3URZP4WdXvOGYckoFWGVN7hApoXVnc) [[Slides2]](http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-pixels.pdf)</br>
- [ ]  Deep Joint Task Learning for Generic Object Extraction-2014 [[Project]](http://vision.sysu.edu.cn/projects/deep-joint-task-learning/) [[Code-Caffe]](https://github.com/xiaolonw/nips14_loc_seg_testonly) [[Dataset]](http://objectextraction.github.io/) [[Paper]](http://ss.sysu.edu.cn/~ll/files/NIPS2014_JointTask.pdf)</br>
- [ ]  Highly Efficient Forward and Backward Propagation of Convolutional Neural Networks for Pixelwise Classification-2014 [[Code-Caffe]](https://dl.dropboxusercontent.com/u/6448899/caffe.zip) [[Paper]](https://arxiv.org/abs/1412.4526)</br>
- [ ]	 **Wider or deeper: Revisiting the resnet model for visual recognition** [[Paper]](https://arxiv.org/abs/1611.10080)</br>
- [ ]	 Describing the Scene as a Whole: Joint Object Detection, Scene Classification and Semantic Segmentation[[Paper]](https://ttic.uchicago.edu/~yaojian/Paper_Holistic.pdf)</br>
- [ ]	 Analyzing Semantic Segmentation Using Hybrid Human-Machine CRFs[[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Mottaghi_Analyzing_Semantic_Segmentation_2013_CVPR_paper.pdf)</br>
- [ ]	 Convolutional Patch Networks with Spatial Prior for Road Detection and Urban Scene Understanding[[Paper]](https://arxiv.org/abs/1502.06344.pdf)</br>
- [ ]	 Deep Deconvolutional Networks for Scene Parsing[[Paper]](https://arxiv.org/abs/1411.4101)</br>
- [ ]  FusionSeg: Learning to combine motion and appearance for fully automatic segmention of generic objects in videos[[Paper]](https://arxiv.org/pdf/1701.05384.pdf)[[Poject]](http://vision.cs.utexas.edu/projects/fusionseg/)</br>
- [ ] ICCV-2017 Deep Dual Learning for Semantic Image Segmentation [[Paper]](http://personal.ie.cuhk.edu.hk/~pluo/pdf/luoWLWiccv17.pdf)</br>
- [ ] From image-level to pixel level labeling with convolutional networks [[Paper]]()</br>
- [ ] Scene Segmentation with DAG-Recurrent Neural Networks [[Paper]](http://ieeexplore.ieee.org/abstract/document/7940028/)</br>
- [ ] Learning to Segment Every Thing [[Paper]](https://arxiv.org/pdf/1711.10370.pdf)</br>
- [ ] Panoptic Segmentation [[Paper]](https://arxiv.org/pdf/1801.00868.pdf)</br>
- [ ] The Devil is in the Decoder [[Paper]](https://arxiv.org/pdf/1707.05847.pdf)</br>
- [ ] Attention to Scale: Scale-aware Semantic Image Segmentation [[Paper]](http://arxiv.org/pdf/1511.03339)[[Project]](http://liangchiehchen.com/projects/DeepLab.html)</br>
- [ ] Convolutional Oriented Boundaries: From Image Segmentation to High-Level Tasks [[Paper]](https://arxiv.org/pdf/1701.04658.pdf) [[Project]](http://www.vision.ee.ethz.ch/~cvlsegmentation/)</br>
- [ ] Scale-Aware Alignment of Hierarchical Image Segmentation [[Paper]](https://www.vision.ee.ethz.ch/en/publications/papers/proceedings/eth_biwi_01271.pdf) [[Project]](http://www.vision.ee.ethz.ch/~cvlsegmentation/)</br>
- [ ] ICCV-2017 Semi Supervised Semantic Segmentation Using Generative Adversarial Network[[Paper]](https://arxiv.org/abs/1703.09695)</br>
- [ ] Object Region Mining with Adversarial Erasing: A Simple Classification to Semantic Segmentation Approach [[Paper]](https://arxiv.org/pdf/1703.08448.pdf)</br>
- [ ] CVPR-2016 Convolutional Feature Masking for Joint Object and Stuff Segmentation [[Paper]](http://arxiv.org/abs/1412.1283)
- [ ] ECCV-2016 Laplacian Pyramid Reconstruction and Refinement for Semantic Segmentation [[Paper]](https://arxiv.org/pdf/1411.5752.pdf)
- [ ]  FastMask: Segment Object Multi-scale Candidates in One Shot-2016 [[Code-Caffe]](https://github.com/voidrank/FastMask) [[Paper]](https://arxiv.org/abs/1612.08843)</br>
- [ ]  **Pixel Objectness-2017** [[Project]](http://vision.cs.utexas.edu/projects/pixelobjectness/) [[Code-Caffe]](https://github.com/suyogduttjain/pixelobjectness) [[Paper]](https://arxiv.org/abs/1701.05349)</br>
