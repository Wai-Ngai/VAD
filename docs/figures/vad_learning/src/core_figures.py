"""Original explanatory diagrams grounded in the local VAD v1 implementation."""
from style import *
from matplotlib.patches import Polygon, Arc


def scene():
    fig,ax=figure('矢量化：用对象和折线描述同一个场景',
                 '先理解场景表达，再区分网络内部的 BEV 特征。',1)
    panel(ax,.5,2.0,7.2,5.55,'栅格表示：逐格记录语义',color='blue')
    panel(ax,8.3,2.0,7.2,5.55,'矢量表示：保留实例与点序',color='teal')
    x0,y0,c=1.55,2.85,.26
    for i in range(19):
        for j in range(14):
            road=5+(j/13)**2*3
            fc=P['gray_bg'] if road-3<i<road+7 else P['white']
            if abs(i-(road-3))<.7 or abs(i-(road+7))<.7: fc='#93D3C5'
            if 8<=i<=9 and 7<=j<=10:fc='#EAC38F'
            if 8<=i<=9 and 1<=j<=3:fc='#CBB9E9'
            ax.add_patch(Rectangle((x0+i*c,y0+j*c),c,c,fc=fc,ec=P['line'],lw=.65))
    text(ax,4.0,2.40,'每个格子有一个标签；对象结构需要再解释',14,ha='center')
    yy=np.linspace(2.85,6.5,20)
    bend=.6*((yy-2.85)/3.65)**2
    for offset in (0,3.5):
        xx=9.25+bend+offset
        ax.plot(xx,yy,color=P['teal'],lw=2.3,marker='o',ms=3.5)
    ax.plot(11+bend,yy,color=P['teal'],ls='--',lw=1.4)
    car(ax,11.25,4.6,color='orange')
    car(ax,11.05,3.2,color='purple')
    for k in (-1,0,1):
        ty=np.linspace(4.9,6.4,6);tx=11.25+.15*(ty-4.9)+k*.33*(ty-4.9)**1.8
        ax.plot(tx,ty,color=P['orange'],lw=1.6,alpha=.7,marker='o',ms=3)
    ty=np.linspace(3.55,5.2,6);tx=11.05-.10*(ty-3.55)
    ax.plot(tx,ty,color=P['purple'],lw=2.4,marker='o',ms=4)
    text(ax,14.1,5.9,'道路折线',14,ha='center',color=P['teal'])
    text(ax,14.1,4.65,'目标 + 未来轨迹',14,ha='center',color=P['orange'])
    text(ax,14.1,3.35,'自车轨迹',14,ha='center',color=P['purple'])
    text(ax,11.9,2.40,'一条线、一辆车、一条轨迹都有独立表达',14,ha='center')
    note(ax,.7,1.22,14.6,'VAD 仍先建立密集 BEV 特征网格，再学习目标、地图折线与轨迹。',size=18)
    save(fig,'01_scene_representation.png')


def architecture():
    fig,ax=figure('VAD v1 总体流程：图像如何变成自车轨迹',
                 '颜色区分视觉 / 地图 / 他车运动 / 自车规划；箭头表示主要信息流。',2,height=10)
    box(ax,.5,7.05,2.1,1.2,'六路相机','当前帧图像',body_fs=15)
    box(ax,3.0,7.05,2.1,1.2,'ResNet + FPN','图像特征',body_fs=15)
    box(ax,5.5,7.05,2.1,1.2,'BEV Encoder','鸟瞰特征',body_fs=15)
    for a,b in ((2.6,3),(5.1,5.5)):
        arrow(ax,(a,7.65),(b,7.65),'blue')
    text(ax,6.55,8.65,'标定 + CAN bus + 历史 BEV',14,ha='center',color=P['blue'])
    arrow(ax,(6.55,8.43),(6.55,8.25),'blue')
    box(ax,8.2,7.05,2.6,1.2,'目标 Decoder','框 / 类别 / 目标特征',color='orange',body_fs=14)
    box(ax,8.2,5.15,2.6,1.25,'地图 Decoder','点级隐特征 → LaneNet 线特征',color='teal',body_fs=11.8)
    arrow(ax,(7.6,7.65),(8.2,7.65),'blue')
    routed(ax,[(7.85,7.65),(7.85,5.77),(8.2,5.77)],'blue')
    box(ax,11.65,7.05,3.85,1.2,'他车运动交互','6 个模式 / 目标 + 地图信息',color='orange',body_fs=14)
    arrow(ax,(10.8,7.65),(11.65,7.65),'orange')
    routed(ax,[(10.8,5.77),(11.2,5.77),(11.2,7.35),(11.65,7.35)],'teal')
    text(ax,13.57,6.56,'输出：他车未来轨迹与分数',14,ha='center',color=P['orange'])
    box(ax,.5,2.85,2.4,1.25,'1 个自车 query','可学习的初始特征',color='purple',body_fs=14)
    box(ax,3.65,2.85,2.7,1.25,'自车与目标交互','ego_agent_query',color='purple',body_fs=13.5)
    box(ax,7.15,2.85,2.9,1.25,'自车与地图交互','ego_map_query',color='purple',body_fs=13.5)
    box(ax,11.05,2.85,4.45,1.25,'拼接两份交互特征 → MLP','直接回归自车逐步位移',color='purple',body_fs=15)
    arrow(ax,(2.9,3.47),(3.65,3.47),'purple')
    arrow(ax,(6.35,3.47),(7.15,3.47),'purple')
    arrow(ax,(10.05,3.47),(11.05,3.47),'purple')
    routed(ax,[(15.5,7.65),(15.75,7.65),(15.75,4.65),(5,4.65),(5,4.10)],'orange')
    text(ax,5.2,4.90,'融合后的目标运动特征',14,color=P['orange'])
    arrow(ax,(8.6,5.15),(8.6,4.1),'teal')
    routed(ax,[(5,2.85),(5,2.35),(12,2.35),(12,2.85)],'purple')
    text(ax,6.5,2.10,'两份特征都进入规划头',14,color=P['purple'])
    box(ax,11.05,.85,4.45,1.1,'3 个分支 → 选择 1 条轨迹','每分支 6 步 × 2 维，再累加位移',color='purple',body_fs=14)
    arrow(ax,(14,2.85),(14,1.95),'purple')
    box(ax,7.15,.85,2.9,1.1,'高层指令','右 / 左 / 直',color='gray',body_fs=14)
    arrow(ax,(10.05,1.4),(11.05,1.4),'gray')
    box(ax,.5,.85,5.85,1.1,'训练时额外计算多任务损失','检测 + 地图 + 运动 + 规划',color='red',body_fs=15)
    save(fig,'02_architecture.png')


def bev():
    fig,ax=figure('BEV 如何从相机图像中取信息',
                 '空间交叉注意力的简化示意：三维参考点提供几何位置，采样偏移由网络学习。',3)
    for x,title in ((.5,'① 一个地面格子与高度柱'),(5.7,'② 投影到可见相机特征'),(10.9,'③ 融合成一个 BEV 特征')):
        panel(ax,x,2.15,4.6,5.42,title)
    # A schematic ground grid in an oblique view.
    origin=np.array([2.25,3.35]);u=np.array([.47,.17]);v=np.array([-.30,.23])
    for i in range(5):
        p=origin+i*u;q=p+4*v;ax.plot([p[0],q[0]],[p[1],q[1]],color=P['line'],lw=1.2)
        p=origin+i*v;q=p+4*u;ax.plot([p[0],q[0]],[p[1],q[1]],color=P['line'],lw=1.2)
    p=origin+2*u+2*v
    corners=np.array([p,p+u,p+u+v,p+v])
    ax.add_patch(Polygon(corners,fc=P['blue_bg'],ec=P['blue'],lw=1.8))
    cx,cy=(p+.5*u+.5*v)
    ax.plot([cx,cx],[cy,6.42],color=P['blue'],lw=2)
    heights=np.linspace(cy+.2,6.42,4)
    ax.scatter([cx]*4,heights,s=68,c=P['blue'],zorder=5)
    text(ax,.85,6.4,'4 个高度\n参考点',16,color=P['blue'],va='top')
    text(ax,2.8,2.62,'每个地面格子有自己的 query',14,ha='center')
    # Feature map plane. Neighbor offsets intentionally schematic.
    ax.add_patch(Rectangle((6.35,3.55),3.15,2.72,fc=P['blue_bg'],ec=P['blue'],lw=1.6))
    for x in np.arange(6.35,9.5,.35):ax.plot([x,x],[3.55,6.27],color='#CADDF0',lw=.7)
    for y in np.arange(3.55,6.27,.34):ax.plot([6.35,9.5],[y,y],color='#CADDF0',lw=.7)
    ys=np.linspace(4,5.8,4)
    for j,y in enumerate(ys):
        arrow(ax,(cx,heights[j]),(7.6,y),'blue',lw=1,style='--',ms=9)
        ax.scatter([7.6],[y],c=P['blue'],s=38,zorder=5)
        offsets=np.array([[.24,.08],[-.19,.12],[.10,-.18]])
        for dx,dy in offsets:
            ax.plot([7.6,7.6+dx],[y,y+dy],color=P['orange'],lw=.8)
            ax.scatter([7.6+dx],[y+dy],c=P['orange'],s=17,zorder=6)
    text(ax,8,6.62,'标定决定投影位置',15,ha='center')
    text(ax,8,3.10,'橙点：学习到的邻域采样位置',14,ha='center',color=P['orange'])
    text(ax,8,2.62,'不可见的相机不参与该点采样',14,ha='center',color=P['muted'])
    box(ax,11.45,5.48,3.5,.95,'可见相机 × 多尺度','',fs=17)
    arrow(ax,(9.5,4.9),(11.45,5.95),'blue')
    box(ax,11.45,3.95,3.5,1.05,'加权采样与融合','每格得到 256 维特征',body_fs=15)
    arrow(ax,(13.2,5.48),(13.2,5),'blue')
    text(ax,13.2,3.06,'不是把六张 RGB 图拼接起来',14,ha='center')
    text(ax,13.2,2.65,'4 个高度点 ≠ 只采样 4 个像素',14,ha='center')
    box(ax,.7,.85,4.4,.9,'时间注意力：读取历史 BEV','',fs=17)
    box(ax,5.8,.85,4.4,.9,'空间注意力：读取相机特征','',fs=17)
    box(ax,10.9,.85,4.4,.9,'FFN：进一步加工特征','',fs=17)
    arrow(ax,(5.1,1.3),(5.8,1.3),'blue');arrow(ax,(10.2,1.3),(10.9,1.3),'blue')
    save(fig,'03_bev_projection.png')


def attention():
    fig,ax=figure('Attention：按相关程度，汇总需要的信息',
                 '用自车向环境特征取信息作类比；以下权重与数值是人为构造的单维示例。',4)
    text(ax,2.05,7.35,'Query  发出查询',18,color=P['purple'],ha='center',weight='bold')
    text(ax,5.9,7.35,'Key  用于匹配',18,ha='center',weight='bold')
    text(ax,9.1,7.35,'注意力权重',18,ha='center',weight='bold')
    text(ax,11.25,7.35,'Value  携带信息',18,ha='center',weight='bold')
    box(ax,.55,4.1,3,1.8,'自车 query','“哪些场景信息\n与当前规划相关？”',color='purple',body_fs=16)
    rows=[(6.0,'目标 A','orange','0.6','1'),(4.35,'目标 B','orange','0.3','4'),(2.7,'目标 C','orange','0.1','8')]
    for y,label,color,weight,val in rows:
        box(ax,4.45,y,2.9,1.05,label,'',color=color,fs=18)
        box(ax,8.45,y,1.3,1.05,weight,'',color='blue',fs=20)
        box(ax,10.55,y,1.4,1.05,val,'',color=color,fs=20)
        arrow(ax,(3.55,5),(4.45,y+.52),'purple',lw=1.3)
        arrow(ax,(7.35,y+.52),(8.45,y+.52),'muted')
        arrow(ax,(9.75,y+.52),(10.55,y+.52),'muted')
        arrow(ax,(11.95,y+.52),(13.15,5),'muted',lw=1.3)
    box(ax,13.15,4.1,2.3,1.8,'加权汇总','得到新特征\n2.6',color='purple',body_fs=19)
    note(ax,.7,1.55,14.6,'示例：0.6 × 1 + 0.3 × 4 + 0.1 × 8 = 2.6',size=19)
    text(ax,8,.85,'真实实现使用多维 Q/K/V 投影、多头与位置编码；注意力权重不是目标类别概率。',14,ha='center',color=P['muted'])
    save(fig,'04_attention.png')


def gradients():
    fig,ax=figure('梯度从哪里回传：特征路径与坐标路径要分开看',
                 '灰/彩色实线表示前向；红色反向箭头表示可回传的梯度；红叉表示截断。',11,height=10)
    names=[('当前图像\n特征','blue'),('当前 BEV','blue'),('目标 / 地图\n隐特征','teal'),('运动特征','orange'),('自车特征','purple'),('规划损失','red')]
    for i,(name,c) in enumerate(names):
        x=.5+2.5*i;box(ax,x,6.6,2,1.2,name,color=c,fs=17)
        if i<5:arrow(ax,(x+2,7.2),(x+2.5,7.2),'muted')
    arrow(ax,(14,8.25),(1.5,8.25),'red',lw=2.2)
    text(ax,7.8,8.59,'当前帧的连续特征链保留梯度；上游没有被整体冻结',15,ha='center',color=P['red'])
    box(ax,.5,3.65,3.35,1.3,'历史图像 → 历史 BEV','eval + no_grad',color='gray',body_fs=16)
    routed(ax,[(3.85,4.3),(4.25,4.3),(4.25,6.6)],'gray',style='--')
    text(ax,.8,3.18,'历史计算图不展开反传',14,color=P['gray'])
    box(ax,4.85,3.65,3.1,1.3,'预测坐标','框中心 / 地图点 / 轨迹',color='teal',body_fs=14)
    arrow(ax,(6.5,6.6),(6.5,4.95),'teal')
    routed(ax,[(9,6.6),(9,6.12),(7.4,6.12),(7.4,4.95)],'orange')
    box(ax,8.85,4.65,2.8,1.15,'框中心 / 地图点副本','clone().detach()',color='gray',fs=15,body_fs=15)
    box(ax,12.55,4.65,2.95,1.15,'Attention 位置编码','此位置路径不回传',color='gray',body_fs=13.5)
    routed(ax,[(7.95,4.55),(8.30,4.55),(8.30,5.22),(8.85,5.22)],'gray')
    arrow(ax,(11.65,5.22),(12.55,5.22),'gray')
    text(ax,8.32,5.76,'×',29,color=P['red'],ha='center',weight='bold')
    box(ax,8.85,2.4,2.8,1.15,'原预测坐标','未统一 detach',color='teal',body_fs=15)
    box(ax,12.55,2.4,2.95,1.15,'几何约束损失','选中的数值可求导',color='red',body_fs=14)
    routed(ax,[(7.95,4),(8.30,4),(8.30,2.97),(8.85,2.97)],'teal')
    arrow(ax,(11.65,2.97),(12.55,2.97),'teal')
    routed(ax,[(14,2.4),(14,1.8),(6.4,1.8),(6.4,3.65)],'red',lw=2)
    text(ax,10,1.38,'连续坐标分支可以向上游回传',15,ha='center',color=P['red'])
    note(ax,.7,.88,14.6,'阈值筛选 / argmax / 匹配本身不可微；被选中的连续特征与坐标仍可能有梯度。',size=15)
    save(fig,'11_gradient_paths.png')


def constraints():
    fig,ax=figure('三个规划软约束：把几何关系加入训练损失',
                 '使用模型预测的地图与目标运动；以下是几何直觉，实际筛选与清零规则见正文。',12)
    for x,title,c in ((.5,'① 边界间距','teal'),(5.65,'② 车辆间距','orange'),(10.8,'③ 道路方向','purple')):
        panel(ax,x,1.65,4.7,5.9,title,color=c)
    # Boundary penalty: nearest sampled POINT, not perpendicular distance.
    by=np.linspace(3.4,6.65,9);bx=4.20+.15*np.sin(by)
    ax.plot(bx,by,'o-',c=P['teal'],lw=2,ms=5)
    tx=np.array([1.75,1.9,2.15,2.65,3.15,3.65]);ty=np.array([3.3,3.75,4.2,4.65,5.05,5.45])
    ax.plot(tx,ty,'o-',c=P['purple'],lw=2,ms=5)
    car(ax,1.75,3.03,color='purple')
    idx=int(np.argmin((bx-tx[-1])**2+(by-ty[-1])**2))
    ax.plot([tx[-1],bx[idx]],[ty[-1],by[idx]],color=P['red'],lw=2,ls='--')
    text(ax,2.25,6.25,'最近边界采样点',14,color=P['teal'])
    text(ax,2.65,5.7,'距离 d',15,color=P['red'])
    text(ax,2.85,2.30,'靠得太近就增加损失\n阈值 1 m：示意为 max(0, 1 − d)',14,ha='center')
    # Agent distances from predicted centers.
    ego=np.array([7.1,3.85]);agent=np.array([8.75,5.5])
    car(ax,*ego,color='purple');car(ax,*agent,color='orange')
    ax.scatter([ego[0],agent[0]],[ego[1],agent[1]],c=[P['purple'],P['orange']],s=35,zorder=7)
    ax.plot([ego[0],agent[0]],[ego[1],ego[1]],color=P['red'],ls='--',lw=1.8)
    ax.plot([agent[0],agent[0]],[ego[1],agent[1]],color=P['red'],ls='--',lw=1.8)
    text(ax,7.9,3.48,'|Δx|',17,ha='center',color=P['red'])
    text(ax,9.12,4.6,'|Δy|',17,color=P['red'])
    text(ax,7.2,6.35,'比较同一未来时刻的位置',14,color=P['orange'])
    text(ax,8,2.55,'x / y 阈值：1.5 m / 3 m\n先排除欧氏距离 > 3 m 的目标',14,ha='center')
    # Direction alignment, unoriented lane geometry.
    p=np.array([11.65,3.7]);q=np.array([14.8,6.2]);ax.plot([p[0],q[0]],[p[1],q[1]],color=P['teal'],lw=3)
    ax.scatter(np.linspace(p[0],q[0],8),np.linspace(p[1],q[1],8),c=P['teal'],s=20)
    start=np.array([12.3,3.7]);end=np.array([13.2,5.6])
    car(ax,*start,color='purple');arrow(ax,tuple(start+.15),tuple(end),'purple',lw=2.8,ms=22)
    arrow(ax,(12.55,4.05),(14.05,5.24),'teal',lw=2,ms=18)
    ax.add_patch(Arc((12.55,4.05),1.3,1.3,theta1=38,theta2=66,color=P['red'],lw=2))
    text(ax,13.45,4.65,'方向差',15,color=P['red'])
    text(ax,13.15,2.55,'关注 2 m 内的 divider\n方向按 π 等价处理',14,ha='center')
    note(ax,.7,.98,14.6,'这些是训练中的软约束；不等于精确车框碰撞检测，也不构成完整交通规则保证。',color='red',size=16)
    save(fig,'12_planning_constraints.png')


def generate():
    scene();architecture();bev();attention();gradients();constraints()

if __name__=='__main__':
    generate()
