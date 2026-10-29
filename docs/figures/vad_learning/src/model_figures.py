"""Query shapes, multi-modal motion and temporal memory for the VAD tutorial.

All geometry is a teaching illustration; no checkpoint outputs are used.
Run this file directly, or import and call generate().
"""
from style import figure, text, panel, box, arrow, note, car, save, P, plt, np
from matplotlib.patches import Rectangle


def canvas(title, subtitle, number, height):
    fig, ax = figure(title, subtitle, number, height=height)
    # These diagrams are meant to remain readable when embedded in the tutorial.
    for item in ax.texts:
        if item.get_fontsize() < 14:
            item.set_fontsize(14)
    return fig, ax


def slots(ax, x, y, rows, cols, color, w=.34, h=.24, gap=.09):
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Rectangle((x+c*(w+gap), y+r*(h+gap)), w, h,
                                   facecolor=P[color+'_bg'], edgecolor=P[color], lw=1.2))


def query_shapes():
    fig, ax = canvas('五类 query：每一种槽位负责什么？',
                     '默认 VAD v1 配置；B 是 batch size，D = 256。下方展示末层结果，不包含 decoder 层数维。',
                     5, 11)
    panel(ax, .5, 6.25, 4.8, 3.15, 'BEV query · 地面信息格', 'blue')
    panel(ax, 5.6, 6.25, 4.8, 3.15, '目标 query · 物体候选槽', 'orange')
    panel(ax, 10.7, 6.25, 4.8, 3.15, '地图 query · 折线上的点', 'teal')
    slots(ax, .95, 7.57, 4, 5, 'blue', .25, .18, .05)
    text(ax, 2.8, 8.13, 'Tiny：100 × 100', 16, weight='bold')
    text(ax, 2.8, 7.71, 'Base：200 × 200', 16, weight='bold')
    text(ax, .95, 7.13, '每格 256 维特征', 16, color=P['blue'])
    text(ax, .95, 6.73, '[B, 10 000 / 40 000, 256]', 16)

    slots(ax, 6.08, 7.72, 2, 7, 'orange', .31, .25, .09)
    text(ax, 8.0, 7.23, '300 个槽位 ≠ 300 个真实目标', 16, ha='center')
    text(ax, 8.0, 6.75, '类别 / 框编码：[B, 300, 10]', 16, ha='center')

    for i in range(3):
        xx = np.linspace(11.18, 12.78, 7)
        yy = 7.75 + i*.24 + .08*np.sin(np.linspace(0, np.pi*1.3, 7))
        ax.plot(xx, yy, '-o', color=P['teal'], lw=1.5, ms=4)
    text(ax, 13.18, 8.12, '100 条线', 16, weight='bold')
    text(ax, 13.18, 7.74, '每条 20 点', 16, weight='bold')
    text(ax, 13.10, 7.22, '共 2 000 个点 query', 16, ha='center')
    text(ax, 13.10, 6.74, '点坐标：[B, 100, 20, 2]', 16, ha='center')

    panel(ax, .5, 2.15, 7.3, 3.7, '运动 query · 每个目标的可能未来', 'orange')
    box(ax, .95, 4.24, 2.42, .70, '300 个目标特征', color='orange', fs=16)
    text(ax, 3.72, 4.60, '+', 24, ha='center', color=P['orange'])
    box(ax, 4.08, 4.24, 2.95, .70, '6 个模式 embedding', color='orange', fs=16)
    text(ax, 4.15, 3.80, '展开为 300 × 6 = 1 800 个 query，每个 256 维', 15, ha='center')
    text(ax, 4.15, 3.31, '目标交互 + 地图交互 → [B, 300, 6, 512]', 15, ha='center')
    text(ax, 4.15, 2.76, '轨迹：[B, 300, 6, 12]　其中 12 = 6 步 × xy', 15, ha='center', weight='bold')

    panel(ax, 8.1, 2.15, 7.4, 3.7, '自车 query · 汇总信息后规划', 'purple')
    box(ax, 8.53, 4.24, 2.3, .70, '1 个自车 query', color='purple', fs=16)
    arrow(ax, (10.96, 4.59), (11.48, 4.59), color='purple')
    box(ax, 11.65, 4.24, 3.40, .70, '先与目标、再与地图交互', color='purple', fs=15)
    text(ax, 11.80, 3.80, '初始特征 [B, 1, 256]；默认使用可学习参数', 15, ha='center')
    text(ax, 11.80, 3.31, '两份交互结果拼接 → MLP 输入 [B, 1, 512]', 15, ha='center')
    text(ax, 11.80, 2.76, '规划：[B, 3, 6, 2] = 3 指令 × 6 步 × xy', 15, ha='center', weight='bold')

    text(ax, .74, 1.60, '目标 / 地图的初始 embedding 为 512 维，进入 decoder 时拆为“内容 256 + 位置 256”。', 14, color=P['muted'])
    text(ax, .74, 1.12, '运动输出在 all_traj_preds 中复制出 L 维；Tiny 的 L=3，Base 的 L=6，不代表运行 L 套运动 decoder。', 14, color=P['muted'])
    return save(fig, '05_query_shapes.png')


def motion_branches():
    fig, ax = canvas('他车的六种可能未来，不等于自车的三个驾驶指令',
                     '左侧学习未来的不确定性；右侧一次生成三分支，再由给定指令选择。',
                     6, 10)
    panel(ax, .5, 2.25, 7.30, 6.2, '他车：6 个学习到的运动模式', 'orange')
    panel(ax, 8.1, 2.25, 7.40, 6.2, '自车：3 个有明确指令的分支', 'purple')
    u = np.linspace(0, 1, 7)
    start = np.array([3.95, 3.55])
    ends = [(-2.65, 2.98), (-1.72, 3.40), (-.63, 3.73),
            (.62, 3.75), (1.73, 3.42), (2.72, 3.02)]
    colors = ['#AD5811', '#D78223', '#E9AC50', '#B17847', '#B75A42', '#8F5C21']
    for i, ((dx, dy), color) in enumerate(zip(ends, colors)):
        xx = start[0] + dx*u**1.65
        yy = start[1] + dy*u
        ax.plot(xx, yy, color=color, lw=2.0, zorder=2)
        ax.scatter(xx[1:], yy[1:], s=32, color=color, edgecolor='white', linewidth=.6, zorder=3)
        text(ax, xx[-1], yy[-1]+.27, f'模式 {i}', 14, color=color, ha='center', weight='bold')
    car(ax, 3.95, 3.35, w=.48, h=.75, color='orange')
    text(ax, 4.15, 2.83, '每个目标：6 条候选轨迹 + 6 个模式分数', 15, ha='center')
    text(ax, 4.15, 2.48, '模式编号不预设“左转 / 直行”等动作语义', 14, ha='center', color=P['muted'])

    start = np.array([11.80, 3.55])
    for dx, dy, label, active in [(-2.55, 3.10, '1 · 左转', False),
                                  (0, 3.78, '2 · 直行', True),
                                  (2.55, 3.10, '0 · 右转', False)]:
        xx = start[0] + dx*u**1.8
        yy = start[1] + dy*u
        color = P['purple'] if active else '#A79CBC'
        ax.plot(xx, yy, color=color, lw=3.5 if active else 1.9,
                linestyle='-' if active else '--', zorder=2)
        ax.scatter(xx[1:], yy[1:], s=42 if active else 30,
                   color=color, edgecolor='white', linewidth=.6, zorder=3)
        text(ax, xx[-1], yy[-1]+.30, label, 16, color=color, ha='center', weight='bold')
    car(ax, 11.80, 3.35, w=.48, h=.75, color='purple')
    text(ax, 11.8, 2.83, 'ego_fut_preds：[B, 3, 6, 2]', 16, ha='center')
    text(ax, 11.8, 2.48, '示例命令 [0, 0, 1] → 选直行分支', 15, ha='center', color=P['purple'], weight='bold')

    note(ax, .72, 1.62, 14.56, '图中曲线已对逐步位移做 cumsum；每条轨迹的 6 个圆点对应未来约 3 秒。', 'blue', 15)
    text(ax, .88, .98, '默认 head 先生成三分支，命令用于监督 / 选分支；离线命令由 GT 提取，在线应由导航提供。', 14, color=P['muted'])
    return save(fig, '06_motion_branches.png')


def temporal_memory():
    fig, ax = canvas('训练队列与测试记忆：历史信息怎样传到当前帧？',
                     'Base 的 Q=4 包含当前帧；Tiny 的 Q=3。历史 BEV 与当前帧共用模型参数。',
                     8, 11)
    panel(ax, .5, 5.28, 15.0, 4.12, '训练：先抽历史帧，再只对当前帧任务计算损失', 'blue')
    text(ax, .95, 8.18, '历史候选', 15, color=P['muted'])
    times = ['t−4', 't−3', 't−2', 't−1']
    for i, label in enumerate(times):
        x=2.20+i*1.40
        box(ax, x, 7.88, 1.05, .58, label, color='red' if i==2 else 'gray', fs=16)
        if i==2:
            ax.plot([x+.06, x+.99], [7.96,8.38], color=P['red'], lw=2.6)
            ax.plot([x+.06, x+.99], [8.38,7.96], color=P['red'], lw=2.6)
    text(ax, 8.05, 8.18, '+', 22, ha='center')
    box(ax, 8.45, 7.88, 1.05, .58, 't', color='blue', fs=16)
    text(ax, 10.12, 8.18, '随机删 1 帧 + 当前帧 = 4 帧队列', 15)
    text(ax, 1.00, 7.43, '3 帧历史：eval + no_grad；仅生成 BEV', 15, color=P['gray'], weight='bold')
    text(ax, 10.75, 7.43, '当前帧：完整前向与损失', 15, color=P['blue'], weight='bold')
    node_x=[1.0,4.2,7.4,10.95]
    node_t=['t−4','t−3','t−1','t']
    for i,(x,label) in enumerate(zip(node_x,node_t)):
        box(ax,x,5.82,2.65,1.18,label,
            '图像 → BEV' if i<3 else '检测 / 地图 / 运动 / 规划',
            color='gray' if i<3 else 'blue',fs=18,body_fs=14)
        if i<3:
            arrow(ax,(x+2.72,6.41),(node_x[i+1]-.09,6.41),color='blue')
    text(ax, 8.7, 5.48, '历史传递的是汇总记忆；只保留当前帧标签进入 loss', 14, ha='center', color=P['muted'])
    text(ax,14.44,6.43,'梯度\n更新',15,color=P['blue'],ha='center',weight='bold')
    arrow(ax,(13.67,6.41),(14.05,6.41),color='blue')

    panel(ax, .5, 1.55, 15.0, 3.30, '测试：每帧读取并覆盖一份 prev_bev；换场景时清空', 'purple')
    xt=[1.0,4.40,7.80]
    for i,x in enumerate(xt):
        box(ax,x,2.55,2.65,1.16,['t','t+1','t+2'][i],
            '当前图像 + 上次 BEV',color='purple',fs=18,body_fs=14)
        if i<2:
            arrow(ax,(x+2.74,3.13),(xt[i+1]-.09,3.13),color='purple')
    arrow(ax,(10.54,3.13),(11.89,3.13),color='red',style='--')
    text(ax,11.22,3.57,'换场景',14,color=P['red'],ha='center')
    box(ax,12.0,2.55,2.85,1.16,'新 scene','prev_bev = None',color='red',fs=18,body_fs=15)
    text(ax,1.0,1.99,'同一场景按时间顺序运行；当前实现测试 batch size = 1',15,color=P['purple'])
    text(ax,11.8,1.99,'新场景不继承旧场景记忆',14,color=P['red'],ha='center')
    text(ax,.88,.98,'每次时间注意力处理“上一份汇总 BEV + 当前 query”，不是同时保存 Q 份独立 BEV 做注意力。',14,color=P['muted'])
    return save(fig, '08_temporal_memory.png')


def generate():
    return [query_shapes(), motion_branches(), temporal_memory()]


if __name__ == '__main__':
    for output in generate():
        print(output)
