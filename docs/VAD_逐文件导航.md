# VAD 逐文件导航

这是 [VAD 代码与算法详解](VAD_代码与算法详解.md) 的配套索引，覆盖当前提交原有的全部 **185 个 Git 跟踪文件**，并解释其所在目录的作用。主线文件优先详细阅读，通用模板和未启用实现可以按需查阅。

每组表格的路径均相对于仓库根目录。`__init__.py` 常常负责导入和注册；即使文件很短，也可能决定配置能否找到对应类。某个类被注册并不表示当前实验一定会使用它。

`.git/` 属于版本管理；`.agents/`、`.codex/`、`.aws/` 属于本地环境辅助目录，不属于算法源码索引。运行生成的 data、ckpts、work_dirs、test、val 目录及本次新增文档也不计入原始 185 个文件。

## 全部目录的作用

| 目录 | 作用 |
| --- | --- |
| `./` | 仓库说明、许可、依赖与版本控制规则。 |
| `VADv2/` | v2 部分核心模型与 CARLA 示例配置，未独立集成为完整可运行工程。 |
| `assets/` | 项目展示素材。 |
| `docs/` | 人阅读的安装、数据与实验操作说明。 |
| `projects/` | 配置和框架插件的顶层 Python 包。 |
| `projects/configs/` | 实验装配说明：模型、数据、损失和训练设置。 |
| `projects/configs/VAD/` | 六份 VAD v1 主实验配置，是选择 Tiny/Base 和训练阶段的入口。 |
| `projects/configs/_base_/` | 可继承的通用配置片段；当前 VAD 主要用其 default_runtime.py。 |
| `projects/configs/_base_/datasets/` | 不同数据集的通用加载与增强模板。 |
| `projects/configs/_base_/models/` | 通用检测/分割模型模板，不是 VAD 必须执行的子模型。 |
| `projects/configs/_base_/schedules/` | 优化器和学习率日程模板。 |
| `projects/configs/datasets/` | 自定义数据集基础配置；v1 继承 custom_nus-3d.py 后覆盖主要设置。 |
| `projects/mmdet3d_plugin/` | 把本仓库的自定义类注册到 MMDetection3D/MMCV 的插件。 |
| `projects/mmdet3d_plugin/VAD/` | VAD v1 核心模型、预测头和 Transformer 组织。 |
| `projects/mmdet3d_plugin/VAD/apis/` | 训练和多 GPU 测试的框架适配函数。 |
| `projects/mmdet3d_plugin/VAD/hooks/` | 训练生命周期回调，如将 epoch 传入模型。 |
| `projects/mmdet3d_plugin/VAD/modules/` | BEV encoder、空间/时间注意力、目标 decoder 和 CUDA 算子封装。 |
| `projects/mmdet3d_plugin/VAD/planner/` | 规划离线指标计算；这里的 planner 目录不等于完整在线控制器。 |
| `projects/mmdet3d_plugin/VAD/runner/` | 保留的时序 runner；当前六份配置使用标准 EpochBasedRunner。 |
| `projects/mmdet3d_plugin/VAD/utils/` | 地图点损失、规划约束、坐标归一化和可选损失调节。 |
| `projects/mmdet3d_plugin/bevformer/` | 保留的 BEVFormer 分支，当前插件不通过它构建 VAD 模型。 |
| `projects/mmdet3d_plugin/bevformer/apis/` | BEVFormer 的训练/测试适配入口，主 VAD 调用其 VAD 对应版本。 |
| `projects/mmdet3d_plugin/bevformer/dense_heads/` | BEVFormer 三维检测头，不包含 VAD 完整地图/规划任务。 |
| `projects/mmdet3d_plugin/bevformer/detectors/` | BEVFormer detector 与 fp16 变体。 |
| `projects/mmdet3d_plugin/bevformer/hooks/` | BEVFormer 回调。 |
| `projects/mmdet3d_plugin/bevformer/modules/` | BEVFormer 的 BEV 与注意力实现，与 VAD 同名模块大量相同。 |
| `projects/mmdet3d_plugin/bevformer/runner/` | BEVFormer 的视频训练 runner。 |
| `projects/mmdet3d_plugin/core/` | 框和评估的通用底层支持。 |
| `projects/mmdet3d_plugin/core/bbox/` | 三维框与矢量地图的匹配、编码和几何表示。 |
| `projects/mmdet3d_plugin/core/bbox/assigners/` | 将预测与 GT 做一对一匹配。 |
| `projects/mmdet3d_plugin/core/bbox/coders/` | 将网络输出解码并进行 top-k/范围筛选。 |
| `projects/mmdet3d_plugin/core/bbox/match_costs/` | 匹配时计算候选预测与 GT 的代价。 |
| `projects/mmdet3d_plugin/core/bbox/structures/` | 携带轨迹等字段的三维框对象和坐标变换。 |
| `projects/mmdet3d_plugin/core/evaluation/` | 评估 hook、运动指标和其他数据格式转换。 |
| `projects/mmdet3d_plugin/datasets/` | nuScenes 时序样本构建、地图 GT、结果格式化和离线评估。 |
| `projects/mmdet3d_plugin/datasets/map_utils/` | 矢量地图匹配、TP/FP 判定和 AP 统计。 |
| `projects/mmdet3d_plugin/datasets/pipelines/` | 图像读取、增强、目标过滤、张量打包。 |
| `projects/mmdet3d_plugin/datasets/samplers/` | 训练分组采样、分布式切分和测试采样顺序。 |
| `projects/mmdet3d_plugin/models/` | 可复用网络和训练辅助实现。 |
| `projects/mmdet3d_plugin/models/backbones/` | 可选 VoVNet 图像主干；v1 配置默认 ResNet-50。 |
| `projects/mmdet3d_plugin/models/hooks/` | 通用模型 hook。 |
| `projects/mmdet3d_plugin/models/opt/` | 自定义 AdamW2 实现；配置中的 AdamW 是另一个注册名。 |
| `projects/mmdet3d_plugin/models/utils/` | GridMask、MLP/嵌入、位置编码、调试绘图等可复用工具。 |
| `tools/` | 命令行入口和辅助脚本。 |
| `tools/analysis_tools/` | 日志曲线、速度/复杂度分析和 VAD 专用可视化。 |
| `tools/data_converter/` | 预先把数据转换成训练/评估需要的索引与标签。 |
| `tools/misc/` | 通用数据浏览、配置打印和推理优化等辅助脚本。 |
| `tools/model_converters/` | 不同模型权重格式转换和发布前裁剪工具。 |

## 逐文件说明

### 仓库根目录

仓库说明、许可、依赖与版本控制规则。

| 文件 | 作用与主线关系 |
| --- | --- |
| [.gitignore](../.gitignore) | Git 忽略规则，排除 Python 缓存、构建产物、虚拟环境等；不参与算法计算。 |
| [LICENSE](../LICENSE) | Apache License 2.0 许可文本，规定代码使用、修改和分发条件；不参与训练。 |
| [README.md](../README.md) | 项目总览、VAD v1/v2 更新、论文与模型链接、结果表和运行文档入口。结果是项目公布的实验信息，不是当前机器的测试记录。 |
| [requirements.txt](../requirements.txt) | 原 Linux 环境的 Conda 风格包版本清单，使用 name=version=build 格式；不是标准 pip requirements 格式。结合 docs/install.md 阅读。 |

### VADv2

v2 部分核心模型与 CARLA 示例配置，未独立集成为完整可运行工程。

| 文件 | 作用与主线关系 |
| --- | --- |
| [VADv2_config_voca4096.py](../VADv2/VADv2_config_voca4096.py) | VADv2 的 CARLA 示例配置：4096 轨迹词表，训练抽取 256 候选，测试使用 4096 候选，IterBasedRunner 120000 次迭代。引用的 detector/dataset/模块、数据和权重未全部提供，且基础配置相对路径需集成修正。 |
| [VADv2_head.py](../VADv2/VADv2_head.py) | 注册 v116ADTRHead，包含感知、运动、轨迹词表 query 交互、候选评分、交通信号任务及几何标签生成。当前输出候选几何直接取词表；PIPP 等依赖缺失，未接入 v1 插件主导入链。 |

### assets

项目展示素材。

| 文件 | 作用与主线关系 |
| --- | --- |
| [arch.png](../assets/arch.png) | README 展示的 VAD 架构图，帮助理解感知、运动和规划之间的关系；不是网络参数或训练数据。 |

### docs

人阅读的安装、数据与实验操作说明。

| 文件 | 作用与主线关系 |
| --- | --- |
| [install.md](../docs/install.md) | 原始安装步骤和依赖版本，包括 PyTorch、MMCV、MMDetection3D、nuScenes 开发包与图像预训练权重。 |
| [prepare_dataset.md](../docs/prepare_dataset.md) | 原始 nuScenes 数据和 CAN bus 准备说明，给出 VAD 专用 pkl 下载/生成方式与目录结构。 |
| [train_eval.md](../docs/train_eval.md) | 原始训练评估说明，介绍联合训练与两阶段训练、单 GPU 评估和旧发布权重的归一化设置。示例中的 VAD_base.py 在当前仓库不存在，应换为真实配置。 |
| [visualization.md](../docs/visualization.md) | 原始预测视频可视化用法。应结合实际脚本理解输入：专用 visualization.py 需要格式化后的 results_nusc.pkl 字典。 |

### projects

配置和框架插件的顶层 Python 包。

| 文件 | 作用与主线关系 |
| --- | --- |
| [__init__.py](../projects/__init__.py) | 空的 Python 包标记。它使 projects 可通过模块路径导入，本身没有算法或注册逻辑。 |

### projects/configs/VAD

六份 VAD v1 主实验配置，是选择 Tiny/Base 和训练阶段的入口。

| 文件 | 作用与主线关系 |
| --- | --- |
| [VAD_base_e2e.py](../projects/configs/VAD/VAD_base_e2e.py) | VAD v1 可用的主实验配置。Base：200×200 BEV、4 尺度特征、6 层感知 encoder/decoder、4 帧队列。60 epochs，从开始联合学习检测、地图、运动和规划。同时定义数据 pipeline、模型、损失、优化器和运行日程。 |
| [VAD_base_stage_1.py](../projects/configs/VAD/VAD_base_stage_1.py) | VAD v1 可用的主实验配置。Base：200×200 BEV、4 尺度特征、6 层感知 encoder/decoder、4 帧队列。48 epochs，感知和运动学习，四项规划损失权重置零。同时定义数据 pipeline、模型、损失、优化器和运行日程。 |
| [VAD_base_stage_2.py](../projects/configs/VAD/VAD_base_stage_2.py) | VAD v1 可用的主实验配置。Base：200×200 BEV、4 尺度特征、6 层感知 encoder/decoder、4 帧队列。12 epochs，加载对应 stage-1 权重，开启规划并继续联合学习感知/运动；未整体冻结上游。同时定义数据 pipeline、模型、损失、优化器和运行日程。 |
| [VAD_tiny_e2e.py](../projects/configs/VAD/VAD_tiny_e2e.py) | VAD v1 可用的主实验配置。Tiny：100×100 BEV、单尺度特征、3 层感知 encoder/decoder、3 帧队列。60 epochs，从开始联合学习检测、地图、运动和规划。同时定义数据 pipeline、模型、损失、优化器和运行日程。 |
| [VAD_tiny_stage_1.py](../projects/configs/VAD/VAD_tiny_stage_1.py) | VAD v1 可用的主实验配置。Tiny：100×100 BEV、单尺度特征、3 层感知 encoder/decoder、3 帧队列。48 epochs，感知和运动学习，四项规划损失权重置零。同时定义数据 pipeline、模型、损失、优化器和运行日程。 |
| [VAD_tiny_stage_2.py](../projects/configs/VAD/VAD_tiny_stage_2.py) | VAD v1 可用的主实验配置。Tiny：100×100 BEV、单尺度特征、3 层感知 encoder/decoder、3 帧队列。12 epochs，加载对应 stage-1 权重，开启规划并继续联合学习感知/运动；未整体冻结上游。同时定义数据 pipeline、模型、损失、优化器和运行日程。 |

### projects/configs/_base_

可继承的通用配置片段；当前 VAD 主要用其 default_runtime.py。

| 文件 | 作用与主线关系 |
| --- | --- |
| [default_runtime.py](../projects/configs/_base_/default_runtime.py) | VAD 主线实际继承的运行时默认值：日志、checkpoint、NCCL、工作流、加载/恢复权重等。部分值会被 VAD 顶层配置覆盖。 |

### projects/configs/_base_/datasets

不同数据集的通用加载与增强模板。

| 文件 | 作用与主线关系 |
| --- | --- |
| [coco_instance.py](../projects/configs/_base_/datasets/coco_instance.py) | COCO 二维图像目标检测/实例分割数据模板。定义路径、pipeline、类别与 dataloader 等；不属于当前 VAD 配置继承链。 |
| [kitti-3d-3class.py](../projects/configs/_base_/datasets/kitti-3d-3class.py) | KITTI 行人、骑行者、汽车三类点云检测数据模板。定义路径、pipeline、类别与 dataloader 等；不属于当前 VAD 配置继承链。 |
| [kitti-3d-car.py](../projects/configs/_base_/datasets/kitti-3d-car.py) | KITTI 仅汽车类别的点云检测数据模板。定义路径、pipeline、类别与 dataloader 等；不属于当前 VAD 配置继承链。 |
| [lyft-3d.py](../projects/configs/_base_/datasets/lyft-3d.py) | Lyft 九类点云检测数据模板。定义路径、pipeline、类别与 dataloader 等；不属于当前 VAD 配置继承链。 |
| [nuim_instance.py](../projects/configs/_base_/datasets/nuim_instance.py) | nuImages 图像实例标注的 COCO 风格数据模板。定义路径、pipeline、类别与 dataloader 等；不属于当前 VAD 配置继承链。 |
| [nus-3d.py](../projects/configs/_base_/datasets/nus-3d.py) | 标准 NuScenesDataset 十类点云检测模板，与 VAD 专用时序数据集不同。定义路径、pipeline、类别与 dataloader 等；不属于当前 VAD 配置继承链。 |
| [nus-mono3d.py](../projects/configs/_base_/datasets/nus-mono3d.py) | nuScenes 单目三维检测数据模板，使用 CustomNuScenesMonoDataset。定义路径、pipeline、类别与 dataloader 等；不属于当前 VAD 配置继承链。 |
| [range100_lyft-3d.py](../projects/configs/_base_/datasets/range100_lyft-3d.py) | Lyft 100 米范围的点云检测数据模板。定义路径、pipeline、类别与 dataloader 等；不属于当前 VAD 配置继承链。 |
| [s3dis-3d-5class.py](../projects/configs/_base_/datasets/s3dis-3d-5class.py) | S3DIS 室内五类三维物体检测数据模板。定义路径、pipeline、类别与 dataloader 等；不属于当前 VAD 配置继承链。 |
| [s3dis_seg-3d-13class.py](../projects/configs/_base_/datasets/s3dis_seg-3d-13class.py) | S3DIS 十三类点云语义分割数据模板。定义路径、pipeline、类别与 dataloader 等；不属于当前 VAD 配置继承链。 |
| [scannet-3d-18class.py](../projects/configs/_base_/datasets/scannet-3d-18class.py) | ScanNet 室内十八类三维检测数据模板。定义路径、pipeline、类别与 dataloader 等；不属于当前 VAD 配置继承链。 |
| [scannet_seg-3d-20class.py](../projects/configs/_base_/datasets/scannet_seg-3d-20class.py) | ScanNet 二十类点云语义分割数据模板。定义路径、pipeline、类别与 dataloader 等；不属于当前 VAD 配置继承链。 |
| [sunrgbd-3d-10class.py](../projects/configs/_base_/datasets/sunrgbd-3d-10class.py) | SUN RGB-D 室内十类三维检测数据模板。定义路径、pipeline、类别与 dataloader 等；不属于当前 VAD 配置继承链。 |
| [waymoD5-3d-3class.py](../projects/configs/_base_/datasets/waymoD5-3d-3class.py) | Waymo 三类点云检测模板，使用 LidarWaymoDataset 和抽样加载设置。定义路径、pipeline、类别与 dataloader 等；不属于当前 VAD 配置继承链。 |
| [waymoD5-3d-car.py](../projects/configs/_base_/datasets/waymoD5-3d-car.py) | Waymo 仅汽车类别的三维检测数据模板。定义路径、pipeline、类别与 dataloader 等；不属于当前 VAD 配置继承链。 |

### projects/configs/_base_/models

通用检测/分割模型模板，不是 VAD 必须执行的子模型。

| 文件 | 作用与主线关系 |
| --- | --- |
| [3dssd.py](../projects/configs/_base_/models/3dssd.py) | 3DSSD 点云单阶段三维检测模板，包含 PointNet2SAMSG、投票/聚合与 SSD3DHead。属于通用框架模板，六份 VAD 主配置不继承它。 |
| [cascade_mask_rcnn_r50_fpn.py](../projects/configs/_base_/models/cascade_mask_rcnn_r50_fpn.py) | ResNet-50 + FPN 的 Cascade Mask R-CNN 图像检测/实例分割模板，含多级 ROI 检测与 mask 分支。属于通用框架模板，六份 VAD 主配置不继承它。 |
| [centerpoint_01voxel_second_secfpn_nus.py](../projects/configs/_base_/models/centerpoint_01voxel_second_secfpn_nus.py) | nuScenes CenterPoint 体素检测模板，0.1 米平面体素、稀疏编码、SECOND、SECONDFPN、CenterHead。属于通用框架模板，六份 VAD 主配置不继承它。 |
| [centerpoint_02pillar_second_secfpn_nus.py](../projects/configs/_base_/models/centerpoint_02pillar_second_secfpn_nus.py) | nuScenes CenterPoint 柱状点云检测模板，0.2 米平面 pillar、PillarFeatureNet 和 CenterHead。属于通用框架模板，六份 VAD 主配置不继承它。 |
| [fcos3d.py](../projects/configs/_base_/models/fcos3d.py) | FCOSMono3D 单目三维检测模板，ResNet/FPN 后回归三维框、方向、速度和属性。属于通用框架模板，六份 VAD 主配置不继承它。 |
| [groupfree3d.py](../projects/configs/_base_/models/groupfree3d.py) | GroupFree3D 点云检测模板，用 PointNet++ 特征和 Transformer decoder 产生目标预测。属于通用框架模板，六份 VAD 主配置不继承它。 |
| [h3dnet.py](../projects/configs/_base_/models/h3dnet.py) | H3DNet 点云检测模板，组合多主干、VoteHead、几何 primitive 分支与 H3D ROI 头。属于通用框架模板，六份 VAD 主配置不继承它。 |
| [hv_pointpillars_fpn_lyft.py](../projects/configs/_base_/models/hv_pointpillars_fpn_lyft.py) | 继承 nuScenes PointPillars/FPN 模板并为 Lyft 调整范围、类别和锚框等。属于通用框架模板，六份 VAD 主配置不继承它。 |
| [hv_pointpillars_fpn_nus.py](../projects/configs/_base_/models/hv_pointpillars_fpn_nus.py) | nuScenes PointPillars 三维检测模板：点柱编码、BEV scatter、SECOND、FPN 和锚框头。属于通用框架模板，六份 VAD 主配置不继承它。 |
| [hv_pointpillars_fpn_range100_lyft.py](../projects/configs/_base_/models/hv_pointpillars_fpn_range100_lyft.py) | 继承 PointPillars/FPN 并适配 Lyft 100 米范围的检测模板。属于通用框架模板，六份 VAD 主配置不继承它。 |
| [hv_pointpillars_secfpn_kitti.py](../projects/configs/_base_/models/hv_pointpillars_secfpn_kitti.py) | KITTI PointPillars 模板，使用 PillarFeatureNet、scatter、SECOND 和 SECONDFPN。属于通用框架模板，六份 VAD 主配置不继承它。 |
| [hv_pointpillars_secfpn_waymo.py](../projects/configs/_base_/models/hv_pointpillars_secfpn_waymo.py) | Waymo PointPillars/SECONDFPN 检测模板，设定点柱、范围与 Anchor3DHead。属于通用框架模板，六份 VAD 主配置不继承它。 |
| [hv_second_secfpn_kitti.py](../projects/configs/_base_/models/hv_second_secfpn_kitti.py) | KITTI SECOND 体素检测模板，使用 HardSimpleVFE、SparseEncoder、SECOND 和 SECONDFPN。属于通用框架模板，六份 VAD 主配置不继承它。 |
| [hv_second_secfpn_waymo.py](../projects/configs/_base_/models/hv_second_secfpn_waymo.py) | Waymo SECOND 体素检测模板，配置对应范围、锚框和三维检测头。属于通用框架模板，六份 VAD 主配置不继承它。 |
| [imvotenet_image.py](../projects/configs/_base_/models/imvotenet_image.py) | ImVoteNet 的图像检测部分模板，包含 ResNet/FPN、RPN 与 ROI 头。属于通用框架模板，六份 VAD 主配置不继承它。 |
| [mask_rcnn_r50_fpn.py](../projects/configs/_base_/models/mask_rcnn_r50_fpn.py) | ResNet-50 + FPN 的 Mask R-CNN 图像检测及实例分割模板。属于通用框架模板，六份 VAD 主配置不继承它。 |
| [paconv_cuda_ssg.py](../projects/configs/_base_/models/paconv_cuda_ssg.py) | 继承 PAConv SSG 配置，把 set abstraction 模块替换为 CUDA 版本。属于通用框架模板，六份 VAD 主配置不继承它。 |
| [paconv_ssg.py](../projects/configs/_base_/models/paconv_ssg.py) | PAConv 点云语义分割模板，包含 PointNet++ 风格编码器、分割头和正则损失。属于通用框架模板，六份 VAD 主配置不继承它。 |
| [parta2.py](../projects/configs/_base_/models/parta2.py) | Part-A2 两阶段点云检测模板，使用稀疏编码、RPN 和部件聚合 ROI 头。属于通用框架模板，六份 VAD 主配置不继承它。 |
| [pointnet2_msg.py](../projects/configs/_base_/models/pointnet2_msg.py) | 继承 PointNet2 SSG，改用多尺度分组 MSG 的点云语义分割主干。属于通用框架模板，六份 VAD 主配置不继承它。 |
| [pointnet2_ssg.py](../projects/configs/_base_/models/pointnet2_ssg.py) | PointNet++ 单尺度分组 SSG 的点云语义分割模板。属于通用框架模板，六份 VAD 主配置不继承它。 |
| [votenet.py](../projects/configs/_base_/models/votenet.py) | VoteNet 点云三维检测模板，使用点特征投票和 VoteHead。属于通用框架模板，六份 VAD 主配置不继承它。 |

### projects/configs/_base_/schedules

优化器和学习率日程模板。

| 文件 | 作用与主线关系 |
| --- | --- |
| [cosine.py](../projects/configs/_base_/schedules/cosine.py) | 通用训练日程模板：AdamW、线性 warmup 与余弦退火，40 epochs。VAD 六份配置自行定义优化器与日程，不继承此文件。 |
| [cyclic_20e.py](../projects/configs/_base_/schedules/cyclic_20e.py) | 通用训练日程模板：AdamW 与周期性学习率/动量调整，20 epochs。VAD 六份配置自行定义优化器与日程，不继承此文件。 |
| [cyclic_40e.py](../projects/configs/_base_/schedules/cyclic_40e.py) | 通用训练日程模板：AdamW 与周期性学习率/动量调整，40 epochs。VAD 六份配置自行定义优化器与日程，不继承此文件。 |
| [mmdet_schedule_1x.py](../projects/configs/_base_/schedules/mmdet_schedule_1x.py) | 通用训练日程模板：SGD 与分段下降，12 epochs，通用二维检测 1x 模板。VAD 六份配置自行定义优化器与日程，不继承此文件。 |
| [schedule_2x.py](../projects/configs/_base_/schedules/schedule_2x.py) | 通用训练日程模板：AdamW 与分段下降，24 epochs。VAD 六份配置自行定义优化器与日程，不继承此文件。 |
| [schedule_3x.py](../projects/configs/_base_/schedules/schedule_3x.py) | 通用训练日程模板：AdamW 与分段下降，36 epochs。VAD 六份配置自行定义优化器与日程，不继承此文件。 |
| [seg_cosine_150e.py](../projects/configs/_base_/schedules/seg_cosine_150e.py) | 通用训练日程模板：语义分割 SGD 与余弦退火，150 epochs。VAD 六份配置自行定义优化器与日程，不继承此文件。 |
| [seg_cosine_200e.py](../projects/configs/_base_/schedules/seg_cosine_200e.py) | 通用训练日程模板：语义分割 Adam 与余弦退火，200 epochs。VAD 六份配置自行定义优化器与日程，不继承此文件。 |
| [seg_cosine_50e.py](../projects/configs/_base_/schedules/seg_cosine_50e.py) | 通用训练日程模板：语义分割 Adam 与余弦退火，50 epochs。VAD 六份配置自行定义优化器与日程，不继承此文件。 |

### projects/configs/datasets

自定义数据集基础配置；v1 继承 custom_nus-3d.py 后覆盖主要设置。

| 文件 | 作用与主线关系 |
| --- | --- |
| [custom_lyft-3d.py](../projects/configs/datasets/custom_lyft-3d.py) | Lyft 三维检测数据模板，配置点云加载、增强、类别和评估。六份 VAD 主配置不继承它，当前工作区不构成完整 Lyft 端到端方案。 |
| [custom_nus-3d.py](../projects/configs/datasets/custom_nus-3d.py) | VAD 主线继承的 nuScenes 基础配置。原内容偏通用点云检测，顶层 VAD 配置覆盖数据集类型、相机 pipeline、类别顺序、范围和标注路径；不能直接把基础文件当作最终配置。 |
| [custom_waymo-3d.py](../projects/configs/datasets/custom_waymo-3d.py) | Waymo 三维检测数据模板，配置数据路径、点云处理和采样。不是 VAD nuScenes 主训练配置。 |

### projects/mmdet3d_plugin

把本仓库的自定义类注册到 MMDetection3D/MMCV 的插件。

| 文件 | 作用与主线关系 |
| --- | --- |
| [__init__.py](../projects/mmdet3d_plugin/__init__.py) | 整个自定义插件的主注册入口。导入匹配器、解码器、评估 hook、数据集 pipeline、VoVNet、通用工具、AdamW2 和 VAD，使配置中的字符串类型能被框架 builder 找到；没有导入旧 bevformer 包。 |

### projects/mmdet3d_plugin/VAD

VAD v1 核心模型、预测头和 Transformer 组织。

| 文件 | 作用与主线关系 |
| --- | --- |
| [VAD.py](../projects/mmdet3d_plugin/VAD/VAD.py) | VAD 顶层模型：提取六相机图像特征，管理训练历史队列与测试 BEV 缓存，调用联合任务头，组织检测/地图/轨迹结果并计算运动与规划单样本指标。 |
| [VAD_head.py](../projects/mmdet3d_plugin/VAD/VAD_head.py) | VAD 联合任务核心：目标/地图 query，检测和矢量地图输出，目标六模式运动预测，自车三指令规划，匹配监督、损失组装与结果解码。 |
| [VAD_transformer.py](../projects/mmdet3d_plugin/VAD/VAD_transformer.py) | VADPerceptionTransformer 连接 BEV encoder 与检测/地图双 decoder；实现地图参考点迭代和通用交互 decoder，处理 CAN bus、自车运动对齐与相机/层级 embedding。 |
| [__init__.py](../projects/mmdet3d_plugin/VAD/__init__.py) | VAD 主插件包入口。导入 BEV 模块、runner、hook，再导入 VAD、VADHead、VADPerceptionTransformer 和两个自定义 decoder，触发 registry 注册。 |

### projects/mmdet3d_plugin/VAD/apis

训练和多 GPU 测试的框架适配函数。

| 文件 | 作用与主线关系 |
| --- | --- |
| [__init__.py](../projects/mmdet3d_plugin/VAD/apis/__init__.py) | 导出主线 custom_train_model 和 custom_train_detector。测试 API 导入被注释，tools/test.py 直接导入具体 test 文件。 |
| [mmdet_train.py](../projects/mmdet3d_plugin/VAD/apis/mmdet_train.py) | 训练管理主线。创建自定义 dataloader、并行模型、optimizer 和 runner，注册优化、学习率、日志、checkpoint、评估及自定义 hook，加载或恢复权重后调用 runner.run。 |
| [test.py](../projects/mmdet3d_plugin/VAD/apis/test.py) | 分布式测试主线。eval 和 no_grad 下调用模型，各 GPU 收集连续帧结果后按 rank 拼接；返回 bbox_results 字典，当前需要注意 evaluate 接口与 gpu_collect 无 return 的遗留问题。 |
| [train.py](../projects/mmdet3d_plugin/VAD/apis/train.py) | 训练主线的模型类别分派包装。VAD 作为检测框架模型转交 custom_train_detector，文件中另有调用框架原生训练 API 的旧 train_model 包装。 |

### projects/mmdet3d_plugin/VAD/hooks

训练生命周期回调，如将 epoch 传入模型。

| 文件 | 作用与主线关系 |
| --- | --- |
| [__init__.py](../projects/mmdet3d_plugin/VAD/hooks/__init__.py) | 导出 TransferWeight 和 CustomSetEpochInfoHook 并触发 hook 注册。当前配置使用后者传递训练 epoch，前者只为旧双模型流程保留。 |
| [custom_hooks.py](../projects/mmdet3d_plugin/VAD/hooks/custom_hooks.py) | 包含两个 hook：TransferWeight 同步训练与 eval_model 权重，CustomSetEpochInfoHook 在每轮开始时将 runner.epoch 传到 VADHead。当前训练主线启用 CustomSetEpochInfoHook，可供轨迹损失 warmup 使用。 |

### projects/mmdet3d_plugin/VAD/modules

BEV encoder、空间/时间注意力、目标 decoder 和 CUDA 算子封装。

| 文件 | 作用与主线关系 |
| --- | --- |
| [__init__.py](../projects/mmdet3d_plugin/VAD/modules/__init__.py) | 主线 BEV 组件导出及注册入口。导出 PerceptionTransformer、空间/时间注意力、BEVFormerEncoder/Layer 和 DetectionTransformerDecoder；同文件定义的注意力类也会随模块导入注册。 |
| [custom_base_transformer_layer.py](../projects/mmdet3d_plugin/VAD/modules/custom_base_transformer_layer.py) | MyCustomBaseTransformerLayer：按配置 operation_order 组织注意力、残差、归一化与 FFN 的通用底层层结构。 |
| [decoder.py](../projects/mmdet3d_plugin/VAD/modules/decoder.py) | DetectionTransformerDecoder 迭代更新目标参考中心；CustomMSDeformableAttention 从二维 BEV 中采样，供目标和地图 decoder 使用。 |
| [encoder.py](../projects/mmdet3d_plugin/VAD/modules/encoder.py) | BEVFormerEncoder/BEVFormerLayer：生成地面与高度参考点，使用 lidar2img 投影到相机，串联时间注意力、空间注意力和前馈网络。 |
| [multi_scale_deformable_attn_function.py](../projects/mmdet3d_plugin/VAD/modules/multi_scale_deformable_attn_function.py) | 为 MMCV 可变形注意力扩展算子提供 fp16/fp32 PyTorch autograd forward/backward 包装；不是独立训练任务。 |
| [spatial_cross_attention.py](../projects/mmdet3d_plugin/VAD/modules/spatial_cross_attention.py) | SpatialCrossAttention 与 MSDeformableAttention3D：按相机可见性重组 BEV query，进行多尺度可变形图像采样并聚合六相机信息。 |
| [temporal_self_attention.py](../projects/mmdet3d_plugin/VAD/modules/temporal_self_attention.py) | TemporalSelfAttention：通过可变形采样融合上一份 BEV 与当前 query，形成时间记忆。 |
| [transformer.py](../projects/mmdet3d_plugin/VAD/modules/transformer.py) | BEVFormer 通用 PerceptionTransformer：图像特征、CAN bus、历史 BEV 转换为 BEV 后解码目标；当前 VAD 配置实际用 VAD_transformer.py 的扩展类。 |

### projects/mmdet3d_plugin/VAD/planner

规划离线指标计算；这里的 planner 目录不等于完整在线控制器。

| 文件 | 作用与主线关系 |
| --- | --- |
| [metric_stp3.py](../projects/mmdet3d_plugin/VAD/planner/metric_stp3.py) | PlanningMetric 离线规划评估器：由 GT 未来车辆/行人框建立占用网格，评估平均位置 L2、轨迹点和车体覆盖碰撞。 |

### projects/mmdet3d_plugin/VAD/runner

保留的时序 runner；当前六份配置使用标准 EpochBasedRunner。

| 文件 | 作用与主线关系 |
| --- | --- |
| [__init__.py](../projects/mmdet3d_plugin/VAD/runner/__init__.py) | 导出并注册 EpochBasedRunner_video。虽然会被主插件导入，当前配置实际选择框架标准 EpochBasedRunner，所以导入不等于运行。 |
| [epoch_based_runner.py](../projects/mmdet3d_plugin/VAD/runner/epoch_based_runner.py) | 旧 video runner，先用外部 eval_model 处理序列历史，再训练最后一帧。当前 VAD.forward_train 已自行处理历史，六份配置不使用此类；其构造函数要求非空 eval_model。 |

### projects/mmdet3d_plugin/VAD/utils

地图点损失、规划约束、坐标归一化和可选损失调节。

| 文件 | 作用与主线关系 |
| --- | --- |
| [CD_loss.py](../projects/mmdet3d_plugin/VAD/utils/CD_loss.py) | 地图点集和点序的 L1/SmoothL1/Chamfer 损失、对应匹配成本、余弦方向损失与加权归约；主配置用 PtsL1Loss 和 PtsDirCosLoss。 |
| [__init__.py](../projects/mmdet3d_plugin/VAD/utils/__init__.py) | 主线地图与规划工具导出入口。导入地图坐标归一化、Chamfer/L1/方向代价和损失，以及规划边界、碰撞、方向损失，供 VADHead 构建和调用。 |
| [map_utils.py](../projects/mmdet3d_plugin/VAD/utils/map_utils.py) | 二维地图框和折线点在米坐标与 [0,1] 归一化坐标间转换。 |
| [plan_loss.py](../projects/mmdet3d_plugin/VAD/utils/plan_loss.py) | 规划边界距离、车辆未来位置距离、道路方向约束及线段相交工具；使用预测场景施加软约束。 |
| [traj_lr_warmup.py](../projects/mmdet3d_plugin/VAD/utils/traj_lr_warmup.py) | 根据 epoch 的 sigmoid 曲线调节轨迹回归损失权重，当前配置关闭；不直接修改优化器学习率。 |

### projects/mmdet3d_plugin/bevformer

保留的 BEVFormer 分支，当前插件不通过它构建 VAD 模型。

| 文件 | 作用与主线关系 |
| --- | --- |
| [__init__.py](../projects/mmdet3d_plugin/bevformer/__init__.py) | 旧 BEVFormer 包入口，导入检测 head、检测器、注意力模块、runner 和 hook。当前插件主入口不导入此包；与 VAD 同时额外导入可能触发同名类重复注册。 |

### projects/mmdet3d_plugin/bevformer/apis

BEVFormer 的训练/测试适配入口，主 VAD 调用其 VAD 对应版本。

| 文件 | 作用与主线关系 |
| --- | --- |
| [__init__.py](../projects/mmdet3d_plugin/bevformer/apis/__init__.py) | 旧 BEVFormer 训练 API 导出文件。测试 API 的导入被注释，当前训练入口实际转向 VAD/apis。 |
| [mmdet_train.py](../projects/mmdet3d_plugin/bevformer/apis/mmdet_train.py) | 旧 BEVFormer 训练组装流程，负责 dataloader、并行包装、optimizer、runner 和 hook。与 VAD 版本基本相同，但保留过时的 custom_build_dataset 导入，不属于当前主线。 |
| [test.py](../projects/mmdet3d_plugin/bevformer/apis/test.py) | 旧 BEVFormer 多卡推理及结果汇聚工具。与 VAD 版本除文件头外相同，当前入口未使用，并同样保留 collect_results_gpu 没有返回值的问题。 |
| [train.py](../projects/mmdet3d_plugin/bevformer/apis/train.py) | 旧 BEVFormer 训练类型分派包装。将检测任务转给 custom_train_detector，保留通用 train_model 分支，当前 VAD 主入口不调用这份代码。 |

### projects/mmdet3d_plugin/bevformer/dense_heads

BEVFormer 三维检测头，不包含 VAD 完整地图/规划任务。

| 文件 | 作用与主线关系 |
| --- | --- |
| [__init__.py](../projects/mmdet3d_plugin/bevformer/dense_heads/__init__.py) | 导出 BEVFormerHead，导入时触发 HEADS 注册。当前 VAD 采用 VADHead，未使用这份基础检测 head。 |
| [bevformer_head.py](../projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py) | 旧基础 3D 检测 head。定义 BEV/目标查询、各 decoder 层分类与框回归、匈牙利匹配监督和框解码，未包含 VAD 的地图、目标轨迹与自车规划分支。 |

### projects/mmdet3d_plugin/bevformer/detectors

BEVFormer detector 与 fp16 变体。

| 文件 | 作用与主线关系 |
| --- | --- |
| [__init__.py](../projects/mmdet3d_plugin/bevformer/detectors/__init__.py) | 导出 BEVFormer 与 BEVFormer_fp16 两个旧检测器。当前 VAD 配置 type=VAD，不使用这两个 detector。 |
| [bevformer.py](../projects/mmdet3d_plugin/bevformer/detectors/bevformer.py) | 基础时序多相机 3D 检测器。提取图像特征，利用历史帧生成 BEV，在 scene 切换时清空测试缓存，调用 BEVFormerHead；当前 VAD 在独立 VAD 类中扩展该思路。 |
| [bevformer_fp16.py](../projects/mmdet3d_plugin/bevformer/detectors/bevformer_fp16.py) | 旧 BEVFormer 半精度训练变体，接受由外部 runner 生成的 prev_bev。它把 val_step 特殊用作历史 BEV 推理，当前 VAD 配置未启用这条流程。 |

### projects/mmdet3d_plugin/bevformer/hooks

BEVFormer 回调。

| 文件 | 作用与主线关系 |
| --- | --- |
| [__init__.py](../projects/mmdet3d_plugin/bevformer/hooks/__init__.py) | 导出旧 TransferWeight hook。当前 VAD 另有同名实现，避免同时额外导入这份包。 |
| [custom_hooks.py](../projects/mmdet3d_plugin/bevformer/hooks/custom_hooks.py) | 旧 TransferWeight hook，周期性把训练模型权重复制到 eval_model。用于旧 video runner 的双模型模式，不属于当前标准 EpochBasedRunner 主线。 |

### projects/mmdet3d_plugin/bevformer/modules

BEVFormer 的 BEV 与注意力实现，与 VAD 同名模块大量相同。

| 文件 | 作用与主线关系 |
| --- | --- |
| [__init__.py](../projects/mmdet3d_plugin/bevformer/modules/__init__.py) | 导出旧 BEV transformer、空间与时间注意力、encoder 和 decoder。名称与 VAD/modules 相同，但当前实际注册和运行的是 VAD 那一份。 |
| [custom_base_transformer_layer.py](../projects/mmdet3d_plugin/bevformer/modules/custom_base_transformer_layer.py) | 可配置 transformer 层积木，根据 operation_order 组合 self-attention、cross-attention、norm 和 FFN。与 VAD 对应文件除头注释外相同，当前主线使用 VAD 副本。 |
| [decoder.py](../projects/mmdet3d_plugin/bevformer/modules/decoder.py) | 目标检测 decoder 与 BEV 上的 CustomMSDeformableAttention。逐层从 BEV 获取目标特征并更新、detach 参考点，与 VAD 副本实现相同。 |
| [encoder.py](../projects/mmdet3d_plugin/bevformer/modules/encoder.py) | BEV 编码器和 BEVFormerLayer，生成 3D/2D 参考点、按 lidar2img 投影，再执行时序与空间注意力。与 VAD 副本实现相同，并保留原作者注明的 shift_ref_2d 原地修改问题。 |
| [multi_scale_deformable_attn_function.py](../projects/mmdet3d_plugin/bevformer/modules/multi_scale_deformable_attn_function.py) | 多尺度可变形注意力的 autograd 包装，连接 MMCV 编译扩展的前向和反向运算。提供 fp16/fp32 版本，本仓并未在这里实现 CUDA 内核；主线使用 VAD 副本。 |
| [spatial_cross_attention.py](../projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py) | 空间交叉注意力及 MSDeformableAttention3D，仅处理每台相机可见的 BEV query，在多个高度参考点附近采样图像特征并跨相机合并。与 VAD 副本相同，旧目录不是当前入口。 |
| [temporal_self_attention.py](../projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py) | 时间自注意力，将历史融合 BEV 与当前 BEV query 作为两项内部队列，通过可变形采样融合。内部 queue=2 不代表训练原始图像序列只能有两帧；当前使用 VAD 副本。 |
| [transformer.py](../projects/mmdet3d_plugin/bevformer/modules/transformer.py) | 基础 PerceptionTransformer，编码相机与尺度信息、CAN bus 状态并对齐历史 BEV，然后串联 BEV encoder 和目标 decoder。与 VAD/modules 同名副本只有文件头区别，当前 VAD 主 transformer 进一步扩展地图等分支。 |

### projects/mmdet3d_plugin/bevformer/runner

BEVFormer 的视频训练 runner。

| 文件 | 作用与主线关系 |
| --- | --- |
| [__init__.py](../projects/mmdet3d_plugin/bevformer/runner/__init__.py) | 导出旧 EpochBasedRunner_video。当前 VAD 六份配置均选框架标准 EpochBasedRunner，未选择这个 runner。 |
| [epoch_based_runner.py](../projects/mmdet3d_plugin/bevformer/runner/epoch_based_runner.py) | 旧时序 runner，把序列拆帧，先用 eval_model 在 no_grad 下处理历史，再训练最后帧。它需要显式提供 eval_model，不能只改配置 type 就直接用于当前入口。 |

### projects/mmdet3d_plugin/core/bbox

三维框与矢量地图的匹配、编码和几何表示。

| 文件 | 作用与主线关系 |
| --- | --- |
| [util.py](../projects/mmdet3d_plugin/core/bbox/util.py) | 主线 3D 框数值编码/解码：尺寸取 log，yaw 变 sin/cos，并恢复为真实尺寸与角度。虽然函数名叫 normalize_bbox，其实现并未使用 pc_range 把中心缩放到 0..1。 |

### projects/mmdet3d_plugin/core/bbox/assigners

将预测与 GT 做一对一匹配。

| 文件 | 作用与主线关系 |
| --- | --- |
| [__init__.py](../projects/mmdet3d_plugin/core/bbox/assigners/__init__.py) | 导出并注册目标检测 HungarianAssigner3D 和地图 MapHungarianAssigner3D。两者在主线训练中负责把无固定顺序的预测查询与真实目标一对一匹配。 |
| [hungarian_assigner_3d.py](../projects/mmdet3d_plugin/core/bbox/assigners/hungarian_assigner_3d.py) | 主线检测一对一匹配器：以分类代价和编码框前 8 维 L1 代价建立矩阵，使用 SciPy Hungarian 求最优分配。未匹配 query 当背景；当前实现不把速度、轨迹或 IoU 加入最终匹配代价。 |
| [map_hungarian_assigner_3d.py](../projects/mmdet3d_plugin/core/bbox/assigners/map_hungarian_assigner_3d.py) | 主线地图一对一匹配器：对每对预测线和 GT 的多个等价点序寻找最低点代价，再用 Hungarian 分配实例。返回 order_index 告诉后续损失采用哪个 GT 起点/方向；主线框和 IoU 成本权重为 0。 |

### projects/mmdet3d_plugin/core/bbox/coders

将网络输出解码并进行 top-k/范围筛选。

| 文件 | 作用与主线关系 |
| --- | --- |
| [__init__.py](../projects/mmdet3d_plugin/core/bbox/coders/__init__.py) | 导出 NMSFreeCoder、带目标未来轨迹的 CustomNMSFreeCoder 和 MapNMSFreeCoder。主线使用配置选择相应解码器，把网络输出转换为检测框、轨迹和地图结果。 |
| [fut_nms_free_coder.py](../projects/mmdet3d_plugin/core/bbox/coders/fut_nms_free_coder.py) | 定义主线 CustomNMSFreeCoder，对最后一层检测分数取 top-k，解码筛选框，同时取出对应 query 的未来轨迹。输出框、分数、类别和轨迹，不执行传统 NMS。 |
| [map_nms_free_coder.py](../projects/mmdet3d_plugin/core/bbox/coders/map_nms_free_coder.py) | 定义主线 MapNMSFreeCoder，对地图分类分数取 top-k，并把地图框/点从归一化表示还原到局部米制坐标。按范围过滤后输出地图类别、置信度、包围框及折线点。 |
| [nms_free_coder.py](../projects/mmdet3d_plugin/core/bbox/coders/nms_free_coder.py) | 基础 NMS-free 检测后处理：最后一层分类分数 top-k、框解码、范围过滤。它不携带运动轨迹；VAD 当前配置使用的是同目录 fut_nms_free_coder.py 的扩展类。 |

### projects/mmdet3d_plugin/core/bbox/match_costs

匹配时计算候选预测与 GT 的代价。

| 文件 | 作用与主线关系 |
| --- | --- |
| [__init__.py](../projects/mmdet3d_plugin/core/bbox/match_costs/__init__.py) | 导出 BBox3DL1Cost 及框架 build_match_cost。使主线匈牙利匹配器能按配置建立框回归匹配代价。 |
| [match_cost.py](../projects/mmdet3d_plugin/core/bbox/match_costs/match_cost.py) | 定义主线 BBox3DL1Cost，实际计算 torch.cdist(p=1) 并乘权重。它提供预测框与 GT 框两两之间的 L1 代价矩阵，交给 Hungarian assigner 使用。 |

### projects/mmdet3d_plugin/core/bbox/structures

携带轨迹等字段的三维框对象和坐标变换。

| 文件 | 作用与主线关系 |
| --- | --- |
| [__init__.py](../projects/mmdet3d_plugin/core/bbox/structures/__init__.py) | 导出 CustomLiDARInstance3DBoxes。为需要自定义 3D 框操作的代码提供类型入口，不能仅凭导出就推断所有主线分支都使用此类型。 |
| [lidar_box3d.py](../projects/mmdet3d_plugin/core/bbox/structures/lidar_box3d.py) | 扩展 LiDAR 框容器以可附带 future trajectories/mask，提供角点、BEV 框、旋转翻转、范围判断和点框关系操作。属于备用容器，主线 dataset 并未实例化这个自定义类；不能假定所有继承操作自动正确处理额外轨迹字段。 |
| [nuscenes_box.py](../projects/mmdet3d_plugin/core/bbox/structures/nuscenes_box.py) | 定义 CustomNuscenesBox 和 CustomDetectionBox，用于保存框及未来轨迹、结果序列化和绘制框/多模式轨迹。主线格式化和可视化工具使用它；rotate/translate 更新框而不变换 fut_trajs，结果消费者需核对轨迹坐标语义。 |

### projects/mmdet3d_plugin/core/evaluation

评估 hook、运动指标和其他数据格式转换。

| 文件 | 作用与主线关系 |
| --- | --- |
| [__init__.py](../projects/mmdet3d_plugin/core/evaluation/__init__.py) | 导出 CustomDistEvalHook。训练主线分布式验证通过这个 hook 调用 VAD 多卡测试并交给数据集评估。 |
| [eval_hooks.py](../projects/mmdet3d_plugin/core/evaluation/eval_hooks.py) | 主线分布式验证 hook：支持动态验证间隔、同步 BatchNorm 统计量、调用 VAD custom_multi_gpu_test，并交给 dataset.evaluate 计算分数。还可根据返回指标保存最优 checkpoint。 |
| [kitti2waymo.py](../projects/mmdet3d_plugin/core/evaluation/kitti2waymo.py) | 把 KITTI 风格预测框转为 Waymo 的对象 protobuf 格式，包含坐标转换、逐样本转换与结果合并。属于其他数据集评估遗留工具，当前 nuScenes VAD 主线不调用。 |
| [metric_motion.py](../projects/mmdet3d_plugin/core/evaluation/metric_motion.py) | 提供 ADE、FDE 和按终点误差选择候选轨迹的通用辅助函数。主线逐帧运动指标在 VAD.py 中直接实现；get_best_preds 注释提 min ADE，但函数实际按最终点距离选模式。 |

### projects/mmdet3d_plugin/datasets

nuScenes 时序样本构建、地图 GT、结果格式化和离线评估。

| 文件 | 作用与主线关系 |
| --- | --- |
| [__init__.py](../projects/mmdet3d_plugin/datasets/__init__.py) | 导出和注册主线 VADCustomNuScenesDataset。它没有导出 custom_build_dataset，后者需要从 datasets.builder 导入。 |
| [builder.py](../projects/mmdet3d_plugin/datasets/builder.py) | 构造自定义 DataLoader、训练/测试 sampler 与 worker 随机种子，并支持 concat/repeat/class-balanced/CBGS 等数据集包装。主线使用其分布式加载器；文件中的 FUSERS/build_fuser 是额外通用接口。 |
| [detection_cvpr_2019.json](../projects/mmdet3d_plugin/datasets/detection_cvpr_2019.json) | 标准 nuScenes 风格的检测评估配置，按类别提供 30/40/50 米半径。主线 VAD 配置选择的是另一份 vad_nusc_detection_cvpr_2019.json，不能把两者范围混为一谈。 |
| [nuscenes_eval.py](../projects/mmdet3d_plugin/datasets/nuscenes_eval.py) | 较早的自定义 nuScenes 检测评估实现，包含 visibility、帧序号及相机重叠区分析。主线 dataset 实际导入 vad_custom_nuscenes_eval.py，因此本文件不是当前配置的直接评估主线。 |
| [nuscenes_vad_dataset.py](../projects/mmdet3d_plugin/datasets/nuscenes_vad_dataset.py) | 主线数据中枢：定义 LiDARInstanceLines、VectorizedLocalMap 和 VADCustomNuScenesDataset，负责框/轨迹标签、在线地图 GT、历史图像队列及 batch 元信息。还负责格式化检测/地图/规划结果并汇总评估；测试模式仍读取 GT，属于验证集离线评估接口。 |
| [vad_custom_nuscenes_eval.py](../projects/mmdet3d_plugin/datasets/vad_custom_nuscenes_eval.py) | VAD 实际调用的 nuScenes 检测评估器，沿用 devkit 的 AP/NDS 计算，并增加按 x/y 矩形范围过滤及可见性/重叠视野分析。支持 pkl 结果读取；评估范围与官方标准半径配置不同。 |
| [vad_nusc_detection_cvpr_2019.json](../projects/mmdet3d_plugin/datasets/vad_nusc_detection_cvpr_2019.json) | 主线检测评估参数：所有类别 x/y 阈值分别为 30/15 米，中心距离匹配阈值为 0.5、1、2、4 米。由 VADCustomNuScenesDataset 加载，决定当前检测数字的统计口径。 |

### projects/mmdet3d_plugin/datasets/map_utils

矢量地图匹配、TP/FP 判定和 AP 统计。

| 文件 | 作用与主线关系 |
| --- | --- |
| [mean_ap.py](../projects/mmdet3d_plugin/datasets/map_utils/mean_ap.py) | 主线地图评估组织器：按类别整理预测和 GT、重采样折线、并行计算 TP/FP、积分 precision-recall 曲线求 AP 并输出表格。评估时通常将 GT 和预测都重采样为 100 点，与训练的 20 点表示不同。 |
| [tpfp.py](../projects/mmdet3d_plugin/datasets/map_utils/tpfp.py) | 提供框/旋转框/折线的 TP、FP 判定。主线调用 custom_tpfp_gen，根据 Chamfer 或 buffer IoU、预测置信度及一条 GT 最多匹配一次的规则统计结果。 |
| [tpfp_chamfer.py](../projects/mmdet3d_plugin/datasets/map_utils/tpfp_chamfer.py) | 地图几何比较函数集，计算折线双向 Chamfer 距离或对折线加宽后的多边形 IoU。主线 custom_polyline_score 用 Shapely 空间索引筛候选，再构建预测线到 GT 线的分数矩阵；其他几何得分函数多为备用。 |

### projects/mmdet3d_plugin/datasets/pipelines

图像读取、增强、目标过滤、张量打包。

| 文件 | 作用与主线关系 |
| --- | --- |
| [__init__.py](../projects/mmdet3d_plugin/datasets/pipelines/__init__.py) | 主线数据处理组件导出入口。注册多相机增强、补齐、归一化、缩放、目标过滤、数据打包，以及备用点云加载组件，实际执行哪些由 pipeline 配置决定。 |
| [formating.py](../projects/mmdet3d_plugin/datasets/pipelines/formating.py) | 扩展默认格式化器，给自车历史/未来轨迹、mask、命令、状态增加维度，转 Tensor 并用 DataContainer 包装。主线由它约定 batch 的形状与哪些字段可以堆叠。 |
| [loading.py](../projects/mmdet3d_plugin/datasets/pipelines/loading.py) | 自定义单帧/多 sweep 点云读取器，支持虚拟点、PointPainting/MVP 增强及降低 LiDAR 线数。当前 VAD 主线训练不读取点云，测试使用外部普通 LoadPointsFromFile，因此这些自定义增强加载器属于备用能力。 |
| [transform_3d.py](../projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py) | 主线图像/标签预处理：光照颜色扰动、归一化、缩放并更新 lidar2img、补边、收集字段，以及同步过滤物体框/类别/未来轨迹。另含未被主线使用的点云范围过滤器；缩放类虽然名为 RandomScale，但当前实现断言只接受单个比例。 |

### projects/mmdet3d_plugin/datasets/samplers

训练分组采样、分布式切分和测试采样顺序。

| 文件 | 作用与主线关系 |
| --- | --- |
| [__init__.py](../projects/mmdet3d_plugin/datasets/samplers/__init__.py) | 导出分组分布式采样器、连续分布式采样器以及 SAMPLER 注册器和 build_sampler。主线 dataloader 据配置选择训练打乱或时序测试顺序。 |
| [distributed_sampler.py](../projects/mmdet3d_plugin/datasets/samplers/distributed_sampler.py) | 主线测试采样器：保持不打乱，将连续索引片段分给各 GPU，以支持每张卡上的时序 BEV 缓存。它有别于逐卡交错抽样，必须与自定义结果收集顺序配套。 |
| [group_sampler.py](../projects/mmdet3d_plugin/datasets/samplers/group_sampler.py) | 主线分布式训练采样器：按 dataset.flag 分组、随机打乱、补齐样本并分给各 GPU。set_epoch 参与随机种子，使每轮次序不同且可复现。 |
| [sampler.py](../projects/mmdet3d_plugin/datasets/samplers/sampler.py) | 创建 SAMPLER 注册表，并通过 build_from_cfg 按配置构建采样器。文件很小，负责名称到类的连接，不实现具体抽样算法。 |

### projects/mmdet3d_plugin/models/backbones

可选 VoVNet 图像主干；v1 配置默认 ResNet-50。

| 文件 | 作用与主线关系 |
| --- | --- |
| [__init__.py](../projects/mmdet3d_plugin/models/backbones/__init__.py) | 导出并触发 VoVNet 骨干注册。插件主入口会导入它，但当前 VAD 配置没有选择 VoVNet。 |
| [vovnet.py](../projects/mmdet3d_plugin/models/backbones/vovnet.py) | 备用 VoVNet 图像骨干实现，包含 OSA 多层特征拼接与 eSE 通道重标定，支持不同深度、阶段冻结和 BN eval。当前六份 VAD 配置使用框架 ResNet-50，因此它不是默认计算主线。 |

### projects/mmdet3d_plugin/models/hooks

通用模型 hook。

| 文件 | 作用与主线关系 |
| --- | --- |
| [__init__.py](../projects/mmdet3d_plugin/models/hooks/__init__.py) | 导出 GradChecker 调试 hook。插件主入口没有导入此 hooks 子包，当前训练配置也没有启用它。 |
| [hooks.py](../projects/mmdet3d_plugin/models/hooks/hooks.py) | GradChecker 调试 hook，检查需要梯度的参数是否没有 grad 并打印名称。可帮助排查未参与损失的分支，默认训练主线未启用。 |

### projects/mmdet3d_plugin/models/opt

自定义 AdamW2 实现；配置中的 AdamW 是另一个注册名。

| 文件 | 作用与主线关系 |
| --- | --- |
| [__init__.py](../projects/mmdet3d_plugin/models/opt/__init__.py) | 导出 AdamW2 并触发优化器注册。插件主入口会导入此包，为特定旧 PyTorch 版本的兼容分支做准备。 |
| [adamw.py](../projects/mmdet3d_plugin/models/opt/adamw.py) | AdamW2 兼容优化器，调整取出 beta 参数的位置并调用旧版 torch.optim._functional.adamw。仅 train.py 检测到 PyTorch 1.8.1 且配置使用 AdamW 时自动切换，不是 VAD 新提出的算法组件。 |

### projects/mmdet3d_plugin/models/utils

GridMask、MLP/嵌入、位置编码、调试绘图等可复用工具。

| 文件 | 作用与主线关系 |
| --- | --- |
| [__init__.py](../projects/mmdet3d_plugin/models/utils/__init__.py) | 导出 run_time、GridMask、RelPositionEmbedding、save_tensor、PatchEmbed 等工具。主插件导入这些定义，但导入不代表每一个工具都参与当前模型计算。 |
| [bricks.py](../projects/mmdet3d_plugin/models/utils/bricks.py) | 提供 run_time 装饰器，在函数前后 CUDA synchronize 并打印累计平均耗时。主要用于性能调试，额外同步会改变正常执行性能。 |
| [embed.py](../projects/mmdet3d_plugin/models/utils/embed.py) | PatchEmbed 工具，用卷积将图像划为 patch 并转换为 token 序列，可补齐尺寸和归一化。当前 ResNet 图像骨干主线没有使用它。 |
| [grid_mask.py](../projects/mmdet3d_plugin/models/utils/grid_mask.py) | 图像网格遮挡增强，包含普通 Grid 和 nn.Module 形式 GridMask。VAD 图像特征提取主线真正使用 GridMask，训练时随机遮挡，eval 时原样返回。 |
| [position_embedding.py](../projects/mmdet3d_plugin/models/utils/position_embedding.py) | 备用位置编码工具，RelPositionEmbedding 将二维坐标正余弦编码映射为特征，SineEmbedding 实现多频正余弦编码。当前 VAD BEV 配置采用框架 LearnedPositionalEncoding，不直接使用这两个类。 |
| [visual.py](../projects/mmdet3d_plugin/models/utils/visual.py) | 中间特征可视化辅助工具。save_tensor 将张量拼图保存，再用 convert_color 生成伪彩色，不是模型主线。 |

### tools

命令行入口和辅助脚本。

| 文件 | 作用与主线关系 |
| --- | --- |
| [create_data.py](../tools/create_data.py) | 遗留通用数据准备入口，包含 KITTI、nuScenes、Lyft、Waymo 和室内数据集分支。当前仓库缺少它导入的 nuscenes_converter、lyft_converter、kitti_converter、indoor_converter，不应当作可直接执行的 VAD 主线入口。 |
| [dist_test.sh](../tools/dist_test.sh) | 测试主线的多进程启动脚本。默认端口 29503，接收配置、checkpoint 和 GPU 数，固定附加 --eval bbox。 |
| [dist_train.sh](../tools/dist_train.sh) | 训练主线的多进程启动脚本。按 GPU 数启动 torch.distributed.launch，默认端口 28509，附加 --launcher pytorch 和 --deterministic。 |
| [test.py](../tools/test.py) | 测试主入口。读取配置和 checkpoint，构建数据加载器，选择单卡或分布式测试，支持原始结果保存、格式化与评估；当前分布式结果字典存在与 evaluate 不一致的接口问题。 |
| [train.py](../tools/train.py) | 训练主入口。读取和合并配置、导入插件、初始化分布式与随机种子、构建模型及数据集，调用 VAD 自定义训练 API，并准备日志和 checkpoint 元数据。 |

### tools/analysis_tools

日志曲线、速度/复杂度分析和 VAD 专用可视化。

| 文件 | 作用与主线关系 |
| --- | --- |
| [__init__.py](../tools/analysis_tools/__init__.py) | 空的包标记文件。将分析工具目录组织为 Python 包，本身没有训练或算法逻辑。 |
| [analyze_logs.py](../tools/analysis_tools/analyze_logs.py) | 辅助分析工具，不在模型前向主线中。读取按行 JSON 日志，绘制指定 loss/指标曲线，或者统计每轮、每次迭代训练耗时。 |
| [benchmark.py](../tools/analysis_tools/benchmark.py) | 辅助推理测速脚本，跳过前 5 次预热并同步 CUDA 计时。保留了错误的 custom_build_dataset 导入路径、命令行整数类型及重复计时问题，需要适配后使用。 |
| [get_flops.py](../tools/analysis_tools/get_flops.py) | 辅助复杂度估算脚本，为卷积、归一化、线性层等挂接 FLOPs 统计 hook，并从测试集取一条实际数据。当前调用与 VAD.forward_dummy 参数签名不一致，自定义注意力和功能式计算也不能假定已完整统计。 |
| [get_params.py](../tools/analysis_tools/get_params.py) | 临时参数统计脚本，不在主线中。需要手动替换 YOUR_CKPT_PATH=None，并累加 checkpoint 的 state_dict 全部张量元素；这还可能包含缓冲区，不严格等于可训练参数数目。 |
| [visualization.py](../tools/analysis_tools/visualization.py) | VAD 专用结果展示工具。读取 format_results 导出的检测、地图和规划结果，绘制六相机画面、BEV 框、目标多模态轨迹和自车规划，合成为 vis.mp4；默认硬编码 nuScenes trainval 与 ./data/nuscenes。 |

### tools/data_converter

预先把数据转换成训练/评估需要的索引与标签。

| 文件 | 作用与主线关系 |
| --- | --- |
| [__init__.py](../tools/data_converter/__init__.py) | 数据转换器 Python 包标记，仅保留版权注释。它本身不执行数据转换，也不定义算法。 |
| [create_gt_database.py](../tools/data_converter/create_gt_database.py) | 传统点云检测的数据增强工具：按 GT 框裁出物体点云，保存物体数据库与 dbinfos，亦支持图像/掩码块。当前 VAD 纯视觉配置及专用转换入口不调用它。 |
| [vad_nuscenes_converter.py](../tools/data_converter/vad_nuscenes_converter.py) | VAD 主线离线数据准备脚本：读取 nuScenes 和 CAN bus，生成带时间关系、六相机标定、物体/自车未来轨迹及 mask 的 temporal pkl。文件还带二维框导出辅助函数，但本文件主入口只调用 infos 生成；mini 与 full 使用相同后缀文件名。 |

### tools/misc

通用数据浏览、配置打印和推理优化等辅助脚本。

| 文件 | 作用与主线关系 |
| --- | --- |
| [browse_dataset.py](../tools/misc/browse_dataset.py) | 通用数据集浏览工具，不在 VAD 算法主线中。支持点云、3D 框、分割和投影图像，但只识别固定数据集类型，未直接适配 VADCustomNuScenesDataset 的时序多相机结构。 |
| [fuse_conv_bn.py](../tools/misc/fuse_conv_bn.py) | 推理模型整理辅助工具。把 BN 均值、方差和仿射参数折叠到相邻卷积中并保存新 checkpoint；采用旧版通用模型初始化接口，使用前需要核对兼容性。 |
| [print_config.py](../tools/misc/print_config.py) | 配置阅读辅助工具。展开 _base_ 继承并合并命令行 --options，打印最终配置，方便核查哪些设置真正生效。 |
| [visualize_results.py](../tools/misc/visualize_results.py) | 通用离线结果展示工具。加载 pickle 预测并调用 dataset.show，未提供 VAD 专用规划展示及完整插件导入流程。 |

### tools/model_converters

不同模型权重格式转换和发布前裁剪工具。

| 文件 | 作用与主线关系 |
| --- | --- |
| [convert_votenet_checkpoints.py](../tools/model_converters/convert_votenet_checkpoints.py) | 遗留 VoteNet 权重升级工具，与 VAD 主线无关。升级旧配置字段、重命名或拆分预测层参数，并严格加载新 state_dict 验证后保存。 |
| [publish_model.py](../tools/model_converters/publish_model.py) | 模型发布前的文件整理工具。删除 checkpoint 中 optimizer 状态、计算 SHA256 并在输出文件名加 8 位摘要，不执行模型训练或推理。 |
| [regnet2mmdet.py](../tools/model_converters/regnet2mmdet.py) | 遗留 RegNet 权重转换工具，与当前 ResNet VAD 主线无关。将 pycls 权重中 stem、head 和残差阶段的命名映射为 MMDetection 风格并保存。 |

## 本次新增文档

| 文件 | 用途 |
| --- | --- |
| [VAD_代码与算法详解.md](VAD_代码与算法详解.md) | 新手概念、算法实现、训练推理调用链、数据与评估、操作指南和常见问题。 |
| [VAD_逐文件导航.md](VAD_逐文件导航.md) | 当前文件，按目录覆盖全部原始文件。 |
| [figures/vad_learning/README.md](figures/vad_learning/README.md) | 2026-10-28 新增的 14 张 PNG 教学图索引，列出对应章节、源码依据与重绘方法。 |
| [figures/vad_learning/](figures/vad_learning/) 下的 14 个 PNG | 已嵌入主文档，解释场景表示、架构、BEV、注意力、query、轨迹、时序、匹配、训练、梯度、规划约束、评估及 v2 对照；逐图文件名见索引。 |
| [src/generate.py](figures/vad_learning/src/generate.py) | 重绘全部图片的统一入口，仅依赖绘图环境，不启动模型训练。 |
| [src/style.py](figures/vad_learning/src/style.py) | 统一中文字体、颜色、画布、框图组件和 PNG 保存设置。 |
| [src/core_figures.py](figures/vad_learning/src/core_figures.py) | 绘制图 01、02、03、04、11、12：表示、总体架构、BEV、注意力、梯度、规划约束。 |
| [src/model_figures.py](figures/vad_learning/src/model_figures.py) | 绘制图 05、06、08：query 尺寸、他车模式与自车分支、历史记忆。 |
| [src/data_figures.py](figures/vad_learning/src/data_figures.py) | 绘制图 07、09、13：坐标与 mask、地图点序和匹配、评估。 |
| [src/training_figures.py](figures/vad_learning/src/training_figures.py) | 绘制图 10、14：训练循环、阶段衔接和 v1/v2 规划对照。 |
