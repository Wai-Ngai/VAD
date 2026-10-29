"""Reproducible training and planning-comparison figures for the VAD guide.

These are teaching diagrams derived from the checked-in source, not measured
model outputs.  Run this module, or call generate(), to rebuild both PNGs.

Source anchors:
* tools/train.py and projects/mmdet3d_plugin/VAD/apis/mmdet_train.py
* projects/configs/VAD/VAD_{tiny,base}_{stage_1,stage_2,e2e}.py
* projects/mmdet3d_plugin/VAD/VAD_head.py
* VADv2/VADv2_head.py:719, 946-962, 1023-1051, 1119, 1126-1129
* VADv2/VADv2_config_voca4096.py:97-98, 336-360
"""

from style import P, plt, np, figure, text, panel, box, arrow, routed, note, car, save


CAPTIONS = {
    "10_training_loop.png": (
        "一次训练迭代从 batch、前向预测与损失计算开始，经反向传播和优化器更新后进入下一批。"
        "标准 EpochBasedRunner 与 hooks 管理循环、梯度更新、学习率、日志、保存和验证。"
        "两阶段配置先训练 48 轮，再加载权重训练 12 轮；第二阶段开启规划损失，"
        "感知与运动分支继续联合训练，并未整体冻结。e2e 配置从开始联合训练 60 轮。"
    ),
    "14_v1_v2_planning.png": (
        "VAD v1 直接回归右转、左转、直行三组自车未来位移，由给定导航命令选取分支并累加为轨迹。"
        "当前仓内 VADv2 head 将词表候选经过抽样和停车清零后构成 query，与图像、交通参与者、地图交互并评分；"
        "返回几何轨迹由 used_plan_anchors 给定。训练 256、测试 4096 是配置初值与意图，"
        "eval 后切回 train 不会自动恢复候选数。当前配置的四项规划评分损失权重依次为 0、0、0、200；"
        "仓库未提供完整的 v2 候选选择与执行链。"
    ),
}


def _readable_subtitle(ax):
    """Keep the shared header while using a readable 14 pt subtitle."""
    ax.texts[1].set_fontsize(14)


def training_loop():
    fig, ax = figure(
        "训练怎样重复，两个阶段怎样衔接？",
        "先理解一次迭代，再理解训练日程。框架负责调度，模型负责预测与损失。",
        10,
        height=10,
    )
    _readable_subtitle(ax)

    panel(ax, .5, 5.8, 15, 2.95, "一次 iteration：处理一个 batch", "blue")
    stages = [
        ("Batch", "取一批样本", "blue"),
        ("Forward", "模型预测", "blue"),
        ("Loss", "计算预测误差", "orange"),
        ("Backward", "求参数梯度", "purple"),
        ("Optimizer", "更新可训练参数", "teal"),
        ("下一 batch", "开始下一次", "blue"),
    ]
    x0, step, width, y, height = .86, 2.44, 2.0, 6.95, 1.05
    centers = []
    for i, (title, body, color) in enumerate(stages):
        x = x0 + i * step
        box(ax, x, y, width, height, title, body, color, fs=17, body_fs=14)
        centers.append(x + width / 2)
        if i:
            arrow(ax, (x - step + width + .06, y + height / 2),
                  (x - .07, y + height / 2))
    routed(ax, [(centers[-1], y - .06), (centers[-1], 6.48),
                (centers[0], 6.48), (centers[0], y - .06)], color="blue")
    text(ax, 8, 6.08, "一批数据完成一次参数更新；一个 epoch 遍历一轮训练集。",
         15, ha="center", color=P["muted"])

    panel(ax, .5, 4.1, 15, 1.4)
    helpers = [
        (.85, "标准 EpochBasedRunner", "管理 epoch、iter 与数据循环", "blue"),
        (5.94, "OptimizerHook", "执行反向、裁剪与参数更新", "purple"),
        (10.88, "其他 hooks", "学习率 · 日志 · 保存 · 验证", "teal"),
    ]
    for x, heading, body, color in helpers:
        text(ax, x, 5.03, heading, 17, weight="bold", color=P[color])
        text(ax, x, 4.51, body, 14)
    for x in (5.5, 10.5):
        ax.plot([x, x], [4.37, 5.25], color=P["line"], lw=1.2)

    panel(ax, .5, .85, 15, 2.98, "两种训练日程：总轮数均为 60", "teal")
    text(ax, .85, 2.72, "两阶段", 17, weight="bold")
    box(ax, 3.0, 2.36, 5.8, .73, "Stage 1 · 48 epochs",
        "检测 + 地图 + 运动；规划 loss = 0", "blue", fs=17, body_fs=14)
    arrow(ax, (8.9, 2.72), (9.39, 2.72), color="teal")
    box(ax, 9.5, 2.36, 5.45, .73, "Stage 2 · 12 epochs",
        "开启规划；上游继续联合训练", "teal", fs=17, body_fs=14)
    text(ax, .85, 1.76, "一次联合", 17, weight="bold")
    box(ax, 3.0, 1.4, 11.95, .73, "e2e · 60 epochs",
        "从开始联合优化：检测 + 地图 + 运动 + 规划", "purple", fs=17, body_fs=14)
    text(ax, 8, 1.04, "Stage 2 使用 load_from 载入 Stage 1；没有整体冻结感知与运动分支。",
         14, ha="center", color=P["muted"])
    return save(fig, "10_training_loop.png")


def _trajectory_card(ax, x, y, title, direction, color):
    """A deliberately schematic command branch, not a predicted trajectory."""
    panel(ax, x, y, 1.75, 1.7)
    text(ax, x + .875, y + 1.37, title, 16,
         ha="center", weight="bold", color=P[color])
    start_x, start_y = x + .875, y + .3
    t = np.linspace(0, 1, 40)
    xs = start_x + direction * .49 * t ** 1.8
    ys = start_y + .72 * t
    ax.plot(xs, ys, color=P[color], lw=2.6, zorder=3)
    arrow(ax, (xs[-6], ys[-6]), (xs[-1], ys[-1]), color=color, lw=2.3, ms=13)
    car(ax, start_x, start_y, w=.24, h=.36, color=color)


def planning_comparison():
    fig, ax = figure(
        "VAD v1 与当前 VADv2：规划的两种写法",
        "v1 回归三组位移；当前 v2 head 为词表候选评分。图中轨迹仅为示意。",
        14,
        height=10,
    )
    _readable_subtitle(ax)
    panel(ax, .5, 2.2, 7.25, 6.45, "VAD v1  ·  直接回归", "blue")
    panel(ax, 8.0, 2.2, 7.5, 6.45, "当前 VADv2  ·  词表候选 + 评分", "purple")

    box(ax, 1.17, 7.1, 5.91, .8, "环境特征 + 自车 query", color="blue", fs=17)
    arrow(ax, (4.125, 7.04), (4.125, 6.81), color="blue")
    box(ax, 1.17, 5.95, 5.91, .8, "轨迹回归头",
        "同时预测 3 组未来二维位移", "blue", fs=17, body_fs=14)
    card_specs = [
        (1.12, "右转分支", 1, "blue"),
        (3.25, "左转分支", -1, "purple"),
        (5.38, "直行分支", 0, "teal"),
    ]
    for x, title, direction, color in card_specs:
        center = x + .875
        routed(ax, [(4.125, 5.89), (4.125, 5.72),
                    (center, 5.72), (center, 5.60)], color=color, ms=13)
        _trajectory_card(ax, x, 3.85, title, direction, color)
        arrow(ax, (center, 3.79), (center, 3.37), color=color, ms=13)
    box(ax, 1.17, 2.5, 5.91, .8, "给定导航命令 → 选择对应分支",
        "位移累加 → 6 个未来轨迹点", "orange", fs=16, body_fs=14)

    vx, vw, center = 8.68, 6.13, 11.745
    right_stages = [
        (7.1, "轨迹词表与抽样", "配置意图：训练 256 / 测试 4096", "purple", 17),
        (5.95, "准备 used_plan_anchors", "近停车轨迹清零；第 0 条固定停车", "purple", 17),
        (4.8, "候选轨迹 → query 交互", "透视图像 → 交通参与者 → 地图", "blue", 17),
        (3.65, "碰撞 · 越界 · 中心线 · 专家", "当前评分损失权重：0 / 0 / 0 / 200", "orange", 16),
        (2.5, "输出：候选轨迹 + 各项分数", "轨迹 = 回归量 × 0 + used anchors", "purple", 17),
    ]
    for i, (y, heading, body, color, fs) in enumerate(right_stages):
        box(ax, vx, y, vw, .8, heading, body, color, fs=fs, body_fs=14)
        if i < len(right_stages) - 1:
            arrow(ax, (center, y - .06), (center, y - .29), color="purple")

    panel(ax, .5, .85, 15, 1.1)
    text(ax, .8, 1.56, "v1：轨迹输出 ≠ 油门 / 转向控制", 15,
         weight="bold", color=P["blue"])
    text(ax, 8.3, 1.56, "v2：未提供完整的候选选择 / 执行链", 15,
         weight="bold", color=P["red"])
    text(ax, .8, 1.12,
         "v2 注意：eval 会把候选数改为 4096；切回 train 不会自动恢复 256。",
         14, color=P["muted"])
    return save(fig, "14_v1_v2_planning.png")


def generate():
    """Build both 16-inch, 180-dpi PNGs and return their Paths."""
    return [training_loop(), planning_comparison()]


if __name__ == "__main__":
    for output in generate():
        print(output)
