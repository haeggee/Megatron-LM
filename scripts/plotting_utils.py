import seaborn as sns

def set_plot_style(rcParams):
    sns.set_theme(style="whitegrid")
    rcParams["text.usetex"] = False
    rcParams["figure.dpi"] = 300
    rcParams["font.size"] = "12.5"
    rcParams["axes.unicode_minus"] = False
    rcParams["font.family"] = "cmr10"
    rcParams["mathtext.fontset"] = "cm"
    rcParams["axes.formatter.use_mathtext"] = False
    # Set rcParams for the border color and ticks
    rcParams["axes.edgecolor"] = "black"  # Set border color
    rcParams["axes.linewidth"] = 1.5  # Set border width
    rcParams["xtick.color"] = "black"  # Set xtick color
    rcParams["ytick.color"] = "black"  # Set ytick color
    # set background color
    rcParams["axes.facecolor"] = "#F3E7CE"
    rcParams["axes.facecolor"] = "#F8EFDE"
    # rcParams["axes.facecolor"] = "#EEE7DD"
    # rcParams["axes.facecolor"] = "#F8F8F8"
    # rcParams["axes.facecolor"] = "#FFFAEB"
    rcParams["axes.facecolor"] = "#EDEDED"
    rcParams["axes.facecolor"] = "#EFEFEAFF"
    # set grid color
    rcParams["grid.color"] = "white"
    rcParams["grid.alpha"] = 0.7
    rcParams["grid.linewidth"] = 1.0
    # rcParams["grid.minor.linewidth"] = 0.5
    rcParams["grid.linestyle"] = "-"
    rcParams["axes.grid.which"] = "both"
    # make ticks show
    rcParams["xtick.bottom"] = True  # Ensure xticks are shown at the bottom
    rcParams["ytick.left"] = True  # Ensure yticks are shown on the left
    sns.set_context(context="talk", font_scale=0.9)
    return rcParams

def autotune_font_size_title(text: str, base_size=15, max_length=35):
    """Scale font size inversely with text length"""
    if len(text) <= max_length:
        return base_size
    else:
        scale_factor = max_length / len(text)
        return max(8, int(base_size * scale_factor))  # minimum size of 8