# VAD 学习图解：PNG 索引与绘图说明

这组图配套 [《VAD 代码与端到端自动驾驶算法详解》](../../VAD_代码与算法详解.md)，已嵌入正文对应章节。共 14 张 PNG，统一宽度为 **2880 像素**，高度为 1620、1800 或 1980 像素。点击下表中的图片名称可以打开原图。

整理日期：2026-10-28。代码基准：`1688c4b1c3a9e2e7873ca9700ff8058170c0e3c8`。所有图片均为根据本仓库实现重新绘制的教学示意；道路形状、车辆位置、轨迹、注意力权重、匹配成本和误差数字不是模型实测输出。图中的默认网络规模及阈值则来自当前配置和源码。

## 怎样用这些图学习

第一遍按 **01 → 02 → 04 → 03 → 05 → 06** 阅读：先知道模型怎样描述场景、最终想输出什么，再理解信息如何从相机传到自车规划。遇到维度时，先问“有多少份、每份装什么”，不必先记住全部数字。

第二遍按 **07 → 08 → 09 → 10 → 11 → 12 → 13** 阅读：理解轨迹标签、历史记忆、训练配对、参数更新、梯度边界和评价方式。最后再看图 14，对比另一种规划写法。图解编号按知识主题组织，在正文中则放在最相关的小节，因此正文里不一定按编号递增出现。

每张图都可用三个问题检查理解：输入是什么？箭头传递的是特征、坐标还是梯度？哪些内容是模型预测，哪些来自数据集标签？

## 14 张图与正文位置

| PNG 原图 | 对应正文 | 重点理解的问题 |
| --- | --- | --- |
| [01 矢量化场景表示](01_scene_representation.png) | [1.3 矢量化是什么意思](../../VAD_代码与算法详解.md#vad-figure-01) | 栅格、对象、折线怎样表达场景？为什么矢量化仍可使用密集 BEV？ |
| [02 VAD v1 总体流程](02_architecture.png) | [3.2 前向计算总图](../../VAD_代码与算法详解.md#vad-figure-02) | 图像怎样变成目标、地图、运动和自车未来位移？ |
| [03 BEV 投影与特征采样](03_bev_projection.png) | [6.1 第二步：建立 BEV](../../VAD_代码与算法详解.md#vad-figure-03) | 地面格子怎样找到图像里的相关信息？ |
| [04 Query 与 Attention](04_attention.png) | [第二章 术语](../../VAD_代码与算法详解.md#vad-figure-04) | Query、Key、Value 各做什么？加权取信息是什么意思？ |
| [05 Query 与张量尺寸](05_query_shapes.png) | [6.1 张量尺寸速查](../../VAD_代码与算法详解.md#vad-figure-05) | 300、100×20、300×6、1 和 3 分别是什么数量？ |
| [06 他车模式与自车分支](06_motion_branches.png) | [6.1 第六步：自车规划](../../VAD_代码与算法详解.md#vad-figure-06) | 六种未来候选为何不等于右转、左转、直行三个分支？ |
| [07 坐标、位移与 mask](07_offsets_masks.png) | [5.3 未来标签与 mask](../../VAD_代码与算法详解.md#vad-figure-07) | 怎样恢复未来位置？缺失标签为什么不能当作停车？ |
| [08 历史 BEV 与时序缓存](08_temporal_memory.png) | [6.2 历史帧与反向传播](../../VAD_代码与算法详解.md#vad-figure-08) | 历史帧怎样帮助预测？为什么历史前向没有梯度？ |
| [09 地图点序与匹配](09_map_matching.png) | [5.5 地图 GT](../../VAD_代码与算法详解.md#vad-figure-09) | 同形地图线为何有多种写法？无序预测怎样与真值配对？ |
| [10 训练循环与阶段](10_training_loop.png) | [7.2 真实训练调用链](../../VAD_代码与算法详解.md#vad-figure-10) | iteration 与 epoch 有何区别？48+12 与 60 怎样理解？ |
| [11 梯度回传路径](11_gradient_paths.png) | [6.2 梯度边界](../../VAD_代码与算法详解.md#vad-figure-11) | detach 切断哪条路径？规划如何影响上游特征和坐标？ |
| [12 三个规划软约束](12_planning_constraints.png) | [6.2 规划约束](../../VAD_代码与算法详解.md#vad-figure-12) | 路边距离、车辆距离和道路方向怎样参与训练？ |
| [13 开环评估与时间口径](13_evaluation.png) | [5.8 自车规划评估](../../VAD_代码与算法详解.md#vad-figure-13) | GT 用在哪里？1/2/3 秒 L2 为什么是前缀平均？ |
| [14 v1 与当前 v2 规划对照](14_v1_v2_planning.png) | [9.1 两种规划思路](../../VAD_代码与算法详解.md#vad-figure-14) | 直接回归与词表候选评分有什么区别？现有 v2 代码边界在哪？ |

颜色尽量统一：蓝色表示视觉和通用计算，绿色表示地图，橙色表示交通参与者与运动，紫色表示自车规划，红色表示损失、梯度或需要辨明的边界。对于训练循环、Q/K/V 等通用示意，以图中文字和局部图例为准。

## 源码依据与简化范围

图解用于帮助阅读实现，箭头展示主要关系，不穷举每个张量变换、残差连接、归一化或分支。下列入口可用于进一步核对。

| 图解 | 主要源码依据 | 阅读时需要保留的限定 |
| --- | --- | --- |
| 01、02、05、06 | [VAD_head.py](../../../projects/mmdet3d_plugin/VAD/VAD_head.py)、[Tiny 配置](../../../projects/configs/VAD/VAD_tiny_e2e.py)、[Base 配置](../../../projects/configs/VAD/VAD_base_e2e.py) | LaneNet 处理点级隐特征；query 数量是模型容量；六个运动模式没有预设的左右转语义。 |
| 03、04 | [VAD_transformer.py](../../../projects/mmdet3d_plugin/VAD/VAD_transformer.py)、[encoder.py](../../../projects/mmdet3d_plugin/VAD/modules/encoder.py)、[spatial_cross_attention.py](../../../projects/mmdet3d_plugin/VAD/modules/spatial_cross_attention.py) | 图 03 只展开一个格子与一个可见相机；图 04 是一般注意力的标量教学例子，不是整个可变形注意力实现。 |
| 07 | [vad_nuscenes_converter.py](../../../tools/data_converter/vad_nuscenes_converter.py) | 所有未来位置先转到当前 LiDAR 坐标系；自车参考原点也与该传感器有关。 |
| 08 | [VAD.py](../../../projects/mmdet3d_plugin/VAD/VAD.py)、[nuscenes_vad_dataset.py](../../../projects/mmdet3d_plugin/datasets/nuscenes_vad_dataset.py) | 图中 Base 的历史丢帧仅为一次采样例子；训练历史与当前共享权重；测试切换场景清空缓存。 |
| 09 | [地图点序生成](../../../projects/mmdet3d_plugin/datasets/nuscenes_vad_dataset.py)、[地图匹配器](../../../projects/mmdet3d_plugin/core/bbox/assigners/map_hungarian_assigner_3d.py)、[目标匹配器](../../../projects/mmdet3d_plugin/core/bbox/assigners/hungarian_assigner_3d.py) | 上半图采用当前 `map_gt_shift_pts_pattern='v2'`；下半图为检测成本示例，不是跨帧关联。 |
| 10 | [train.py](../../../tools/train.py)、[mmdet_train.py](../../../projects/mmdet3d_plugin/VAD/apis/mmdet_train.py)、[stage 1](../../../projects/configs/VAD/VAD_tiny_stage_1.py)、[stage 2](../../../projects/configs/VAD/VAD_tiny_stage_2.py) | 使用标准 EpochBasedRunner 和优化器 hook；阶段切换通过 `load_from`，不能等同于断点恢复。 |
| 11、12 | [VAD_head.py](../../../projects/mmdet3d_plugin/VAD/VAD_head.py)、[plan_loss.py](../../../projects/mmdet3d_plugin/VAD/utils/plan_loss.py) | 注意力位置副本与几何损失输入不同；边界相交清零、车辆类别筛选及 π 等价规则需结合正文看。 |
| 13 | [VAD.py](../../../projects/mmdet3d_plugin/VAD/VAD.py)、[metric_stp3.py](../../../projects/mmdet3d_plugin/VAD/planner/metric_stp3.py)、[dataset evaluate](../../../projects/mmdet3d_plugin/datasets/nuscenes_vad_dataset.py) | 这是当前 nuScenes 开环评估；图中前缀平均解释的是规划 L2；不能泛化到所有运动指标。 |
| 14 | [VADv2_head.py](../../../VADv2/VADv2_head.py)、[VADv2_config_voca4096.py](../../../VADv2/VADv2_config_voca4096.py) | 仅描述本仓库两个文件；完整 detector、候选选择和执行链未提供；不能代替完整 v2 论文系统。 |

总体架构、矢量表示和规划几何约束的概念参考 [VAD: Vectorized Scene Representation for Efficient Autonomous Driving，图 1—3 与第 3 节](https://arxiv.org/html/2303.12077v3)。图 14 的候选规划思路可对照 [VADv2: End-to-End Vectorized Autonomous Driving via Probabilistic Planning，第 3.2 节，2024 年版本](https://arxiv.org/html/2402.13243v1)。这些 PNG 是面向当前代码的原创教学重绘，没有复制论文原图；具体数字与实现细节以源码为准。

## 怎样重新生成或修改图片

绘图只需要 Python 3、Matplotlib、NumPy 和可用中文字体，不导入 PyTorch、MMCV 或 VAD 模型，不需要数据集与 GPU。在仓库根目录运行：

```bash
python3 docs/figures/vad_learning/src/generate.py
```

该命令会覆盖此目录下的 14 张同名 PNG。默认中文字体路径为 `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`。其他环境可以通过环境变量指定自己的中文字体文件，例如：

```bash
VAD_FIGURE_FONT=/path/to/your/CJK-font.ttf python3 docs/figures/vad_learning/src/generate.py
```

如果没有正确设置中文字体，Matplotlib 可能产生缺字警告；先修正字体再检查输出。分辨率与通用配色在 `style.py` 中统一管理。

| 绘图文件 | 作用 |
| --- | --- |
| [src/generate.py](src/generate.py) | 一条命令重绘全部 14 张图片；将 Matplotlib 缓存放入临时目录。 |
| [src/style.py](src/style.py) | 中文字体、画布、颜色、文本框、箭头、车辆符号、PNG 保存设置。 |
| [src/core_figures.py](src/core_figures.py) | 图 01、02、03、04、11、12：矢量表示、架构、BEV、注意力、梯度与规划约束。 |
| [src/model_figures.py](src/model_figures.py) | 图 05、06、08：query 尺寸、运动分支与时序记忆。 |
| [src/data_figures.py](src/data_figures.py) | 图 07、09、13：坐标与 mask、地图匹配、评估。 |
| [src/training_figures.py](src/training_figures.py) | 图 10、14：训练循环与 v1/v2 规划对照。 |

图片生成后应检查中文缺字、文字越界、箭头遮挡以及与源码的对应关系。修改模型规模或阈值时，正文、图内标注和本索引也需要同步更新。
