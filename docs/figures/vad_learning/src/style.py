"""Shared, reproducible styling for the VAD learning PNG figures."""
from pathlib import Path
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
FONT = os.environ.get('VAD_FIGURE_FONT', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if Path(FONT).is_file():
    font_manager.fontManager.addfont(FONT)
    FAMILY = font_manager.FontProperties(fname=FONT).get_name()
else:
    FAMILY = 'sans-serif'
plt.rcParams.update({'font.family': FAMILY, 'axes.unicode_minus': False,
                     'font.size': 15, 'savefig.facecolor': '#F6F8FC'})
P = dict(bg='#F6F8FC', white='#FFFFFF', ink='#182C46', muted='#576B83',
         line='#DCE4EF', blue='#2378C8', blue_bg='#EAF3FC',
         teal='#008577', teal_bg='#E8F5F1', purple='#7651BF', purple_bg='#F1ECFA',
         orange='#C87516', orange_bg='#FFF2DF', red='#BE4B60', red_bg='#FCECF0',
         gray='#7A8799', gray_bg='#EDF0F5')

def figure(title, subtitle, number, height=9):
    fig, ax = plt.subplots(figsize=(16, height), dpi=180)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.patch.set_facecolor(P['bg'])
    ax.set(xlim=(0,16), ylim=(0,height)); ax.axis('off')
    text(ax, .5, height-.38, f'{number:02d}  {title}', 25, weight='bold', va='top')
    text(ax, .52, height-.97, subtitle, 13, color=P['muted'], va='top')
    ax.plot([.5,15.5],[.49,.49], color=P['line'], lw=1)
    text(ax,.52,.23,'VAD 学习图解  ·  教学示意，非模型实测输出',11,color=P['muted'])
    return fig, ax

def text(ax,x,y,s,size=16,color=None,ha='left',va='center',weight='normal',**kwargs):
    return ax.text(x,y,s,fontsize=size,color=color or P['ink'],ha=ha,va=va,
                   fontweight=weight,linespacing=1.45,**kwargs)

def panel(ax,x,y,w,h,title=None,color='blue'):
    patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.15',
                         facecolor=P['white'],edgecolor=P['line'],lw=1.15)
    ax.add_patch(patch)
    if title:
        text(ax,x+.25,y+h-.34,title,18,color=P[color],weight='bold')
    return patch

def box(ax,x,y,w,h,title,body='',color='blue',fs=17,body_fs=14):
    patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.025,rounding_size=0.11',
                         facecolor=P[color+'_bg'],edgecolor=P[color],lw=1.4)
    ax.add_patch(patch)
    if body:
        text(ax,x+w/2,y+h-.30,title,fs,ha='center',weight='bold',color=P[color])
        text(ax,x+w/2,y+(h-.40)/2,body,body_fs,ha='center')
    else:
        text(ax,x+w/2,y+h/2,title,fs,ha='center',weight='bold',color=P[color])
    return patch

def arrow(ax,a,b,color='muted',lw=1.8,style='-',rad=0,ms=16,zorder=3):
    c=P.get(color,color)
    ar=FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=ms,linewidth=lw,
                       color=c,linestyle=style,connectionstyle=f'arc3,rad={rad}',zorder=zorder)
    ax.add_patch(ar); return ar

def routed(ax,points,color='muted',lw=1.7,style='-',ms=16):
    for a,b in zip(points[:-2],points[1:-1]):
        ax.plot([a[0],b[0]],[a[1],b[1]],color=P.get(color,color),lw=lw,linestyle=style,zorder=2)
    return arrow(ax,points[-2],points[-1],color=color,lw=lw,style=style,ms=ms)

def note(ax,x,y,w,s,color='blue',size=14):
    ax.add_patch(FancyBboxPatch((x,y-.27),w,.54,boxstyle='round,pad=0.02,rounding_size=.08',
                               facecolor=P[color+'_bg'],edgecolor='none'))
    text(ax,x+.16,y,s,size,color=P[color])

def car(ax,x,y,w=.42,h=.72,color='orange',angle=0):
    from matplotlib.transforms import Affine2D
    trans=Affine2D().rotate_deg_around(x,y,angle)+ax.transData
    p=FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=.01,rounding_size=.08',
                     fc=P[color+'_bg'],ec=P[color],lw=2,transform=trans,zorder=4)
    ax.add_patch(p)
    win=Rectangle((x-w*.32,y+h*.03),w*.64,h*.20,facecolor=P[color],edgecolor='none',
                   transform=trans,zorder=5)
    ax.add_patch(win)

def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/name
    fig.savefig(path,dpi=180,facecolor=P['bg'],metadata={'Software':'VAD learning figures / Matplotlib'})
    plt.close(fig)
    return path
