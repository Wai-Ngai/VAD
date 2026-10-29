"""Data/geometry/evaluation teaching figures, checked against local VAD code.

Run directly, or call generate() from the shared figure-generation entry point.
All numeric examples are synthetic and are not measurements from the model.
"""
from style import figure, text, panel, box, arrow, routed, note, save, P, plt, np
from matplotlib.patches import Rectangle, Circle


def canvas(title, subtitle, number, height=11):
    fig, ax = figure(title, subtitle, number, height=height)
    # Keep even the common subtitle/footer legible in the document export.
    for item in ax.texts:
        if item.get_fontsize() < 14:
            item.set_fontsize(14)
    return fig, ax


def offsets_masks():
    fig, ax = canvas(
        '固定坐标中的轨迹：位移需要累加，缺失需要 mask',
        '所有未来位置先转到当前 LiDAR(t) 坐标；同一条轨迹始终使用这一套固定坐标轴。', 7)
    panel(ax, .5, 5.0, 7.05, 4.4, '① 同一坐标系里的四个已知位置', 'blue')
    panel(ax, 7.85, 5.0, 7.65, 4.4, '② 从逐步位移恢复未来位置', 'teal')

    ox, oy, scale = 1.25, 5.68, .80
    arrow(ax, (ox, oy), (6.85, oy), 'gray')
    arrow(ax, (ox, oy), (ox, 8.62), 'gray')
    text(ax, 6.93, oy, 'x', 16, color=P['muted'])
    text(ax, ox, 8.73, 'y', 16, color=P['muted'], ha='center')
    text(ax, 1.04, 5.37, 'O：当前 LiDAR 原点', 14, color=P['muted'])
    pts = np.array([[2, 1], [2, 2], [3, 3], [4, 3]], dtype=float)
    xy = pts * scale + np.array([ox, oy])
    for a, b in zip(xy[:-1], xy[1:]):
        d = (b-a) / np.linalg.norm(b-a)
        arrow(ax, a+d*.12, b-d*.11, 'blue', lw=2.5, ms=19)
    for i, (x,y) in enumerate(xy):
        ax.add_patch(Circle((x,y), .08, facecolor=P['orange'] if i==0 else P['blue'],
                            edgecolor=P['white'], linewidth=1, zorder=6))
    text(ax, xy[0,0]+.30, xy[0,1]-.05, 'p0 = (2, 1)\n当前物体中心', 15, color=P['orange'])
    text(ax, xy[1,0]-.27, xy[1,1], 'p1 = (2, 2)', 16, ha='right')
    text(ax, xy[2,0], xy[2,1]+.42, 'p2 = (3, 3)', 16, ha='center')
    text(ax, xy[3,0]+.25, xy[3,1], 'p3 = (4, 3)', 16)
    text(ax, 4.9, 6.9, '箭头：相邻时刻位移', 14, color=P['blue'])
    text(ax, 4.9, 6.47, '坐标以米为单位', 14, color=P['muted'])

    text(ax, 8.2, 8.55, '位置 = 当前中心 + 逐步位移的累积和', 18, weight='bold')
    box(ax, 8.2, 7.37, 6.95, .84, 'Δ = [(0, 1), (1, 1), (1, 0)]', color='blue', fs=18)
    arrow(ax, (11.68, 7.3), (11.68, 6.64), 'teal', lw=2.3)
    text(ax, 12.02, 6.99, 'cumsum + p0', 16, color=P['teal'])
    box(ax, 8.2, 5.68, 6.95, .84, 'p = [(2, 2), (3, 3), (4, 3)]', color='teal', fs=18)
    text(ax, 8.25, 5.24, '不能直接把 Δ 当成未来绝对位置。', 15, color=P['muted'])

    panel(ax, .5, 1.48, 15.0, 3.16, '③ 六个未来时刻：后三步没有标注，零只是占位', 'purple')
    start, colw = 3.83, 1.87
    for j in range(6):
        x = start + j*colw
        ax.add_patch(Rectangle((x-.04,1.73),colw-.08,2.13,
                               facecolor=P['blue_bg'] if j<3 else P['gray_bg'],edgecolor='none'))
    text(ax, .91, 3.52, '未来时间', 17, weight='bold')
    text(ax, .91, 2.8, '存储的 offset', 17, weight='bold')
    text(ax, .91, 2.08, '有效性 mask', 17, weight='bold')
    offsets = ['(0, 1)', '(1, 1)', '(1, 0)', '(0, 0)', '(0, 0)', '(0, 0)']
    for j in range(6):
        x = start + j*colw + (colw-.12)/2
        col = P['ink'] if j<3 else P['gray']
        text(ax,x,3.52,f'{(j+1)*.5:.1f} s',17,ha='center',color=col)
        text(ax,x,2.8,offsets[j],17,ha='center',color=col)
        text(ax,x,2.08,'1' if j<3 else '0',20,ha='center',weight='bold',color=P['blue'] if j<3 else P['gray'])
    note(ax,.5,.94,15,'mask = 0 表示未知，不表示物体停车；无效时刻应从相应监督与指标中排除。','orange',15)
    return save(fig, '07_offsets_masks.png')


def sample_polyline(vertices, n=20):
    vertices=np.asarray(vertices,dtype=float)
    lengths=np.linalg.norm(np.diff(vertices,axis=0),axis=1)
    arclength=np.r_[0,np.cumsum(lengths)]
    d=np.linspace(0,arclength[-1],n)
    return np.column_stack([np.interp(d,arclength,vertices[:,k]) for k in (0,1)])


def map_matching():
    fig, ax = canvas(
        '地图点序与一对一匹配：形状相同，写法可以不同',
        '地图点序模式 v2：每条线采样 20 点；点序维度为 19，含不足时的无效填充。',9)
    panel(ax,.5,5.75,15,3.65,'① 等价点序：保留几何形状与相邻连接关系','teal')
    text(ax,.94,8.55,'开放线：正序与反序',17,weight='bold')
    vertices=np.array([[1.35,7.72],[2.2,7.92],[3.0,7.55],[3.9,7.84],[4.85,7.64]])
    pts=sample_polyline(vertices)
    for row, (dy, left, right, label) in enumerate([(0,'1','20','正序：1 → 20'),(-1.02,'20','1','反序：20 → 1')]):
        p=pts+np.array([0,dy])
        ax.plot(p[:,0],p[:,1],color=P['teal'],lw=2.5,zorder=3)
        ax.scatter(p[:,0],p[:,1],s=27,c=P['teal'],edgecolors=P['white'],linewidths=.5,zorder=4)
        text(ax,1.03,p[0,1],left,15,ha='right',color=P['teal'],weight='bold')
        text(ax,5.05,p[-1,1],right,15,color=P['teal'],weight='bold')
        text(ax,5.65,7.7+dy,label,15,color=P['muted'])
    text(ax,8.15,8.55,'闭合线：同一方向，改换起点',17,weight='bold')
    polygon=np.array([[0,0],[1.32,.08],[1.48,1.03],[.74,1.4],[-.15,.94],[0,0]])
    for x, chosen in [(9.1,0),(12.8,3)]:
        p=polygon+np.array([x,6.8])
        ax.plot(p[:,0],p[:,1],color=P['purple'],lw=2.7)
        ax.scatter(p[:-1,0],p[:-1,1],s=28,c=P['purple'],zorder=4)
        # The direction stays identical; only the highlighted starting vertex changes.
        a,b=p[1],p[2]
        arrow(ax,a*.65+b*.35,a*.25+b*.75,'purple',lw=2,ms=17)
        sx,sy=p[chosen]
        ax.add_patch(Circle((sx,sy),.105,facecolor=P['orange'],edgecolor=P['white'],lw=1.2,zorder=5))
        if chosen==0:
            text(ax,sx-.12,sy-.28,'起点 A',14,ha='center',color=P['orange'])
        else:
            text(ax,sx+.18,sy+.05,'起点 B',14,color=P['orange'])
    text(ax,8.15,6.29,'原始顶点示意；换起点后重新等距采样。',14,color=P['muted'])
    text(ax,.95,6.04,'地图匹配：先为每对预测 / GT 选最低代价点序，再做实例的一对一匹配。',15,color=P['teal'])

    panel(ax,.5,1.48,15,3.94,'② 全局一对一分配：检测匹配示例（3 个预测槽 / 2 个 GT）','blue')
    text(ax,1.0,4.56,'综合成本',16,weight='bold')
    text(ax,3.23,4.56,'GT A',17,ha='center',weight='bold')
    text(ax,4.9,4.56,'GT B',17,ha='center',weight='bold')
    values=[[1,2],[1.1,100],[9,9]]
    for r in range(3):
        y=3.84-r*.76
        text(ax,1.02,y,f'Query {r+1}',16)
        for c in range(2):
            x=2.48+c*1.67
            picked=(r,c) in [(0,1),(1,0)]
            ax.add_patch(Rectangle((x,y-.31),1.5,.62,
                                  facecolor=P['blue_bg'] if picked else P['gray_bg'],
                                  edgecolor=P['blue'] if picked else P['line'],lw=1.7 if picked else 1))
            text(ax,x+.75,y,str(values[r][c]),19,ha='center',weight='bold' if picked else 'normal',
                 color=P['blue'] if picked else P['muted'])
    text(ax,6.74,4.55,'选择总成本最小的组合',18,weight='bold')
    for y,q,g,col in [(3.55,'Q1','GT B','blue'),(2.79,'Q2','GT A','blue'),(2.03,'Q3','背景','gray')]:
        box(ax,6.75,y,1.0,.56,q,color=col,fs=16)
        arrow(ax,(7.85,y+.28),(8.65,y+.28),col)
        box(ax,8.76,y,1.65,.56,g,color=col,fs=16)
    box(ax,11.32,2.74,3.55,1.17,'总成本 3.1','2.0 + 1.1',color='teal',fs=21,body_fs=17)
    text(ax,11.32,2.19,'不是逐行各自选最小值',15,color=P['muted'])
    note(ax,.5,.94,15,'示例成本＝分类＋框 L1（含权重）；这是训练阶段的 GT 配对，不是跨帧跟踪。','orange',15)
    return save(fig,'09_map_matching.png')


def evaluation():
    fig, ax = canvas(
        '怎样理解开环评估，以及 1 / 2 / 3 秒规划误差',
        '真实未来用于离线命令与打分，不进入视觉预测；下方误差数字均为教学示意。',13)
    panel(ax,.5,5.9,15,3.5,'① 开环：下一帧由录制日志决定，预测不改变后续观测','blue')
    rowy=8.05
    nodes=[(1.0,2.2,'录制图像 t'),(3.8,2.2,'视觉预测'),(6.6,2.3,'3 个规划分支'),
           (9.8,2.1,'按命令选择'),(12.8,2.,'预测轨迹')]
    for x,w,label in nodes:
        box(ax,x,rowy,w,.68,label,color='blue',fs=17)
    for (x,w,_),(nx,_,_) in zip(nodes[:-1],nodes[1:]):
        arrow(ax,(x+w+.08,rowy+.34),(nx-.08,rowy+.34),'blue')
    box(ax,1.0,6.98,2.2,.63,'录制图像 t+1',color='gray',fs=16)
    arrow(ax,(2.1,7.99),(2.1,7.67),'gray')
    text(ax,3.5,7.3,'下一次调用仍读取日志',15,color=P['muted'])
    text(ax,1.02,6.43,'GT 不送入视觉预测网络',15,color=P['orange'],weight='bold')

    box(ax,7.05,6.67,2.15,.66,'未来 GT / 标注',color='purple',fs=16)
    box(ax,9.8,6.67,2.1,.66,'离线命令',color='purple',fs=16)
    box(ax,12.8,6.67,2.,.66,'计算指标',color='teal',fs=16)
    arrow(ax,(9.27,7.),(9.72,7.),'purple')
    arrow(ax,(10.85,7.39),(10.85,7.98),'purple')
    arrow(ax,(13.8,7.98),(13.8,7.4),'blue')
    routed(ax,[(8.12,6.61),(8.12,6.19),(13.8,6.19),(13.8,6.61)],'purple')
    text(ax,10.9,6.03,'评分真值',14,color=P['purple'],ha='center')

    panel(ax,.5,3.52,15,1.94,'② 闭环概念对照：需要执行与环境反馈，本仓库的 nuScenes 评估不包含此环','gray')
    y=4.12
    closed=[(1.1,'观测'),(4.0,'规划'),(6.9,'控制 / 执行'),(9.8,'环境变化'),(12.7,'新观测')]
    for x,label in closed:
        box(ax,x,y,2.,.59,label,color='gray',fs=16)
    for (x,_),(nx,_) in zip(closed[:-1],closed[1:]):
        arrow(ax,(x+2.07,y+.295),(nx-.07,y+.295),'gray')
    routed(ax,[(13.7,4.06),(13.7,3.78),(2.1,3.78),(2.1,4.06)],'gray',style='--')

    panel(ax,.5,.83,15,2.32,'③ L2 计算前缀平均：不是只看恰好 1 / 2 / 3 秒的最后一点','teal')
    xs=np.array([1.45,3.15,4.85,6.55,8.25,9.95])
    for i,x in enumerate(xs):
        text(ax,x,2.42,f'{(i+1)*.5:.1f} s',14,ha='center',color=P['muted'])
        text(ax,x,2.02,f'{(i+1)*.2:.1f} m',17,ha='center',weight='bold')
    for end,y,color in [(1,1.61,'blue'),(3,1.32,'teal'),(5,1.03,'purple')]:
        ax.plot([xs[0]-.52,xs[end]+.52],[y,y],color=P[color],lw=2.1)
        ax.plot([xs[0]-.52,xs[0]-.52],[y,y+.15],color=P[color],lw=2.1)
        ax.plot([xs[end]+.52,xs[end]+.52],[y,y+.15],color=P[color],lw=2.1)
    text(ax,11.1,2.13,'1 s / 前 2 点 → 0.3 m',16,color=P['blue'],weight='bold')
    text(ax,11.1,1.64,'2 s / 前 4 点 → 0.5 m',16,color=P['teal'],weight='bold')
    text(ax,11.1,1.15,'3 s / 前 6 点 → 0.7 m',16,color=P['purple'],weight='bold')
    return save(fig,'13_evaluation.png')


def generate():
    return [offsets_masks(),map_matching(),evaluation()]


if __name__ == '__main__':
    for path in generate():
        print(path)
