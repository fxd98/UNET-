简体中文 | [English](release_notes.md)

## Release Notes
* 2020.02.26

  **`v2.0`**
  * 全新发布2.0版本，全面升级至动态图，支持20+分割模型，4个骨干网络，5个数据集，9种Loss：
    * 分割模型：ANN、BiSeNetV2、DANet、DeeplabV3、DeeplabV3+、FCN、FastSCNN、Gated-scnn、GCNet、HarDNet、OCRNet、PSPNet、UNet、UNet++、U<sup>2</sup>Net、Attention UNet、Decoupled SegNet、EMANet、DNLNet、ISANet
    * 骨干网络：ResNet, HRNet, MobileNetV3, Xception
    * 数据集：Cityscapes, ADE20K, Pascal VOC, Pascal Context, COCO Stuff
    * Loss：CrossEntropy Loss、BootstrappedCrossEntropy Loss、Dice Loss、BCE Loss、OhemCrossEntropyLoss、RelaxBoundaryLoss、OhemEdgeAttentionLoss、Lovasz Hinge Loss、Lovasz Softmax Loss
  * 提供基于Cityscapes和Pascal Voc数据集的高质量预训练模型 50+
  * 支持多卡GPU并行评估，提供了高效的指标计算功能。支持多尺度评估/翻转评估/滑动窗口评估等多种评估方式。
  * 支持XPU模型训练，包括DeepLabv3、HRNet、UNet。
  * 开源了基于Hierarchical Multi-Scale Attention结构的语义分割模型，在Cityscapes验证集上达到87% mIoU。
  * 动态图模式支持模型在线量化、剪枝等模型压缩功能。
  * 动态图下支持模型动转静，实现高性能部署。

* 2020.12.18

  **`v2.0.0-rc`**
  * 全新发布2.0-rc版本，全面升级至动态图，支持15+分割模型，4个骨干网络，3个数据集，4种Loss：
    * 分割模型：ANN, BiSeNetV2, DANet, DeeplabV3, DeeplabV3+, FCN, FastSCNN, Gated-scnn, GCNet, HarDNet, OCRNet, PSPNet, UNet, UNet++, U<sup>2</sup>-Net, Attention UNet
    * 骨干网络：ResNet, HRNet, MobileNetV3, Xception
    * 数据集：Cityscapes, ADE20K, Pascal VOC
    * Loss：CrossEntropy Loss、BootstrappedCrossEntropy Loss、Dice Loss、BCE Loss
  * 提供基于Cityscapes和Pascal Voc数据集的高质量预训练模型 40+。
  * 支持多卡GPU并行评估，提供了高效的指标计算功能。支持多尺度评估/翻转评估/滑动窗口评估等多种评估方式。

* 2020.12.02

  **`v0.8.0`**
  * 增加多尺度评估/翻转评估/滑动窗口评估等功能。
  * 支持多卡GPU并行评估，提供了高效的指标计算功能。
  * 增加Pascal VOC 2012数据集。
  * 新增在Pascal VOC 2012数据集上的高精度预训练模型，详见[模型库](../configs/)。
  * 支持对PNG格式的伪彩色图片进行预测可视化。

* 2020.10.28

  **`v0.7.0`**
  * 全面支持Paddle2.0-rc动态图模式，推出PaddleSeg[动态图体验版](../dygraph/)
  * 发布大量动态图模型，支持11个分割模型，4个骨干网络，3个数据集：
      * 分割模型：ANN, BiSeNetV2, DANet, DeeplabV3, DeeplabV3+, FCN, FastSCNN, GCNet, OCRNet, PSPNet, UNet
      * 骨干网络：ResNet, HRNet, MobileNetV3, Xception
      * 数据集：Cityscapes, ADE20K, Pascal VOC

  * 提供高精度骨干网络预训练模型以及基于Cityscapes数据集的语义分割[预训练模型](../dygraph/configs/)。Cityscapes精度超过**82%**。


* 2020.08.31

  **`v0.6.0`**
  * 丰富Deeplabv3p网络结构，新增ResNet-vd、MobileNetv3两种backbone，满足高性能与高精度场景，并提供基于Cityscapes和ImageNet的[预训练模型](./model_zoo.md)4个。
  * 新增高精度分割模型OCRNet，支持以HRNet作为backbone，提供基于Cityscapes的[预训练模型](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/docs/model_zoo.md#cityscapes%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B)，mIoU超过80%。
  * 新增proposal free的实例分割模型[Spatial Embedding](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/contrib/SpatialEmbeddings)，性能与精度均超越MaskRCNN。提供了基于kitti的预训练模型。

* 2020.05.12

  **`v0.5.0`**
  * 全面升级[HumanSeg人像分割模型](../contrib/PP-HumanSeg)，新增超轻量级人像分割模型HumanSeg-lite支持移动端实时人像分割处理，并提供基于光流的视频分割后处理提升分割流畅性。
  * 新增[气象遥感分割方案](../contrib/RemoteSensing)，支持积雪识别、云检测等气象遥感场景。
  * 新增[Lovasz Loss](lovasz_loss.md)，解决数据类别不均衡问题。
  * 使用VisualDL 2.0作为训练可视化工具

* 2020.02.25

  **`v0.4.0`**
  * 新增适用于实时场景且不需要预训练模型的分割网络Fast-SCNN，提供基于Cityscapes的[预训练模型](./model_zoo.md)1个
  * 新增LaneNet车道线检测网络，提供[预训练模型](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v0.4.0/contrib/LaneNet#%E4%B8%83-%E5%8F%AF%E8%A7%86%E5%8C%96)一个
  * 新增基于PaddleSlim的分割库压缩策略([量化](../slim/quantization/README.md), [蒸馏](../slim/distillation/README.md), [剪枝](../slim/prune/README.md), [搜索](../slim/nas/README.md))


* 2019.12.15

  **`v0.3.0`**
  * 新增HRNet分割网络，提供基于cityscapes和ImageNet的[预训练模型](./model_zoo.md)8个
  * 支持使用[伪彩色标签](./data_prepare.md#%E7%81%B0%E5%BA%A6%E6%A0%87%E6%B3%A8vs%E4%BC%AA%E5%BD%A9%E8%89%B2%E6%A0%87%E6%B3%A8)进行训练/评估/预测，提升训练体验，并提供将灰度标注图转为伪彩色标注图的脚本
  * 新增[学习率warmup](./configs/solver_group.md#lr_warmup)功能，支持与不同的学习率Decay策略配合使用
  * 新增图像归一化操作的GPU化实现，进一步提升预测速度。
  * 新增Python部署方案，更低成本完成工业级部署。
  * 新增Paddle-Lite移动端部署方案，支持人像分割模型的移动端部署。
  * 新增不同分割模型的预测[性能数据Benchmark](../deploy/python/docs/PaddleSeg_Infer_Benchmark.md), 便于开发者提供模型选型性能参考。


* 2019.11.04

  **`v0.2.0`**
  * 新增PSPNet分割网络，提供基于COCO和cityscapes数据集的[预训练模型](./model_zoo.md)4个。
  * 新增Dice Loss、BCE Loss以及组合Loss配置，支持样本不均衡场景下的[模型优化](./loss_select.md)。
  * 支持[FP16混合精度训练](./multiple_gpus_train_and_mixed_precision_train.md)以及动态Loss Scaling，在不损耗精度的情况下，训练速度提升30%+。
  * 支持[PaddlePaddle多卡多进程训练](./multiple_gpus_train_and_mixed_precision_train.md)，多卡训练时训练速度提升15%+。
  * 发布基于UNet的[工业标记表盘分割模型](../contrib#%E5%B7%A5%E4%B8%9A%E7%94%A8%E8%A1%A8%E5%88%86%E5%89%B2)。

* 2019.09.10

  **`v0.1.0`**
  * PaddleSeg分割库初始版本发布，包含DeepLabv3+, U-Net, ICNet三类分割模型, 其中DeepLabv3+支持Xception, MobileNet v2两种可调节的骨干网络。
  * CVPR19 LIP人体部件分割比赛冠军预测模型发布[ACE2P](../contrib/ACE2P)。
  * 预置基于DeepLabv3+网络的[人像分割](../contrib/HumanSeg/)和[车道线分割](../contrib/RoadLine)预测模型发布。

</br>
