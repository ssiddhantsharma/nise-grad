"""Shared plotting style — figures4papers house style
(github.com/ChenLiu-1996/figures4papers): blue-green-red-neutral palette,
Helvetica/DejaVu stack, top+right spines off, thicker dark spines, frameless
legends, minimal grid, 300-dpi vector-editable export. Import and call apply().

The back-compat constant NAMES (GREEN/BLUE/RED/...) are kept so existing figure
scripts need no edits; each is remapped to its figures4papers ROLE below.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- figures4papers PALETTE (verbatim hexes) -------------------------------
BLUE_MAIN   = "#0F4D92"   # proposed / key method — emphasis
BLUE_SECOND = "#3775BA"
GREEN_1, GREEN_2, GREEN_3 = "#DDF3DE", "#AADCA9", "#8BCF8B"   # positives
RED_1, RED_2, RED_STRONG  = "#F6CFCB", "#E9A6A1", "#B64342"   # contrasts
NEUTRAL     = "#CFCECE"   # baselines / also-ran fills
INK         = "#272727"   # dark text / spines
MUTED       = "#7A7A7A"   # secondary labels
HIGHLIGHT   = "#FFD700"   # attention accent (e.g. "NEXT")
TEAL        = "#42949E"
VIOLET      = "#9A4D8E"
GRID        = "#E8E8E8"

# --- back-compat aliases (role -> f4p color), used across the figure scripts ---
GREEN  = BLUE_MAIN     # role: emphasized winner (iptm / ipSAE / structure / free)
BLUE   = BLUE_SECOND   # role: a secondary / comparison method (visible line)
RED    = RED_STRONG    # role: the fine-tune / failure / contrast
GOLD   = VIOLET        # role: 4th series accent (violet is visible on white; f4p gold is not)
PURPLE = VIOLET
GREY   = "#4A4A4A"     # role: dark neutral for reference lines / dark-fill boxes


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
        "font.size": 12.5,
        "axes.titlesize": 14, "axes.titleweight": "bold",
        "axes.labelsize": 12.5,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "legend.fontsize": 11, "figure.titlesize": 16, "figure.titleweight": "bold",
        # figures4papers structural conventions
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 1.6, "axes.edgecolor": INK,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": False,
        "axes.grid": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.dpi": 300, "figure.dpi": 150,
        "svg.fonttype": "none",   # editable text in vector exports
    })


def grid(ax):
    ax.grid(True, color=GRID, lw=0.8, axis="y")
    ax.set_axisbelow(True)
