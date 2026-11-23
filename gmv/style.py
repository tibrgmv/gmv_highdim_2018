"""Chart style of every figure: warm white, ink, one red accent, handwritten notes. Importing it applies it."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch

FONTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
SURFACE, INK, MUTED, GRIDC, EDGE = "#FAFAF8", "#1E1E1E", "#6B6862", "#ECE8E0", "#CFC9BE"
RED, NOTE, WARM, DARK, LIGHT = "#B5232A", "#C8372D", "#A39C90", "#5E5A55", "#CBC4B8"
TAN, SAND, WINE, ROSE = "#9C8F7A", "#B9AE9C", "#7A2A2E", "#D9A19C"
COLOURS = {"traditional": WARM, "Frahm-Memmel": DARK, "oracle": INK, "bona fide": RED, "equal weights": LIGHT}
STYLES = {"traditional": "-", "Frahm-Memmel": "-", "oracle": "-", "bona fide": "--", "equal weights": ":"}
SLIDE = (12, 5.4)

for _f in os.listdir(FONTS):
    if _f.endswith(".ttf"):
        font_manager.fontManager.addfont(os.path.join(FONTS, _f))
plt.rcParams.update({
    "font.family": "Caladea", "mathtext.fontset": "stix", "font.size": 14,
    "axes.prop_cycle": cycler(color=[INK, RED, WARM, DARK]), "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.edgecolor": EDGE, "axes.labelcolor": MUTED, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 1.0, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.titlesize": 17, "axes.titleweight": "normal", "axes.titlelocation": "left", "axes.titlepad": 14,
    "lines.linewidth": 2.6, "lines.markersize": 7, "legend.frameon": False, "legend.fontsize": 13,
})


def note(ax, x, y, text, size=24, ha="left", **kw):
    fp = FontProperties(fname=os.path.join(FONTS, "Caveat.ttf"), size=size)
    return ax.text(x, y, text, fontproperties=fp, color=NOTE, ha=ha, va="center", rotation=-2, linespacing=0.95, **kw)


def arrow(ax, start, end, rad=0.3):
    ax.add_patch(FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={rad}", color=NOTE, lw=1.8,
                                 arrowstyle="-|>,head_length=8,head_width=4.5", shrinkA=3, shrinkB=5))


def save(fig, path, dpi=160):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
