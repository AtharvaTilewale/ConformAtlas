"""ConformAtlas logo: a topographic 'atlas' of a two-basin free-energy landscape,
clipped to a disc, next to a two-weight wordmark. All geometry is vector; text is
converted to paths so the SVG needs no fonts."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

matplotlib.use("Agg")
plt.rcParams["svg.fonttype"] = "none"

OUT = Path(__file__).resolve().parent  # run: python docs/assets/make_logo.py

FONT_LIGHT = FontProperties(fname="/usr/share/fonts/truetype/noto/NotoSans-Light.ttf")
FONT_SEMI = FontProperties(fname="/usr/share/fonts/truetype/noto/NotoSans-SemiBold.ttf")
FONT_REG = FontProperties(fname="/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf")

NAVY = "#0F2743"
INK = "#15314F"
TEAL = "#0FA3A3"
TEAL_DEEP = "#0C7F8E"
GREY = "#6B7A8C"

# contour colours from outer (ink) to inner (teal)
LIGHT_LEVELS = ["#1C3A5E", "#1F4F78", "#157E92", TEAL]
WHITE_LEVELS = ["#FFFFFF", "#FFFFFF", "#D7F3F2", "#5FD1CC"]
WHITE_ALPHAS = [0.55, 0.75, 0.95, 1.0]


def landscape(n=600):
    x = np.linspace(-1.25, 1.25, n)
    X, Y = np.meshgrid(x, x)
    # two anisotropic wells, slightly rotated, of different depth
    def well(cx, cy, sx, sy, th, depth):
        c, s = np.cos(th), np.sin(th)
        u = c * (X - cx) + s * (Y - cy)
        v = -s * (X - cx) + c * (Y - cy)
        return depth * np.exp(-((u / sx) ** 2 + (v / sy) ** 2))

    Z = -(well(0.34, -0.14, 0.50, 0.36, 0.55, 1.0) + well(-0.42, 0.32, 0.40, 0.30, -0.35, 0.80))
    return X, Y, Z


def draw_icon(ax, cx, cy, r, mode="light", pts_per_unit=80.0, weight=1.0):
    """Draw the icon centred at (cx, cy) with disc radius r.

    pts_per_unit converts data units to points so stroke weights are stable across
    output sizes."""
    X, Y, Z = landscape()
    clip = Circle((cx, cy), r, transform=ax.transData, facecolor="none", edgecolor="none")
    ax.add_patch(clip)
    # outer levels are cut by the disc -> the landscape itself forms the globe edge
    levels = [-0.80, -0.56, -0.36, -0.20, -0.09, -0.035]
    if mode == "light":
        colors = [TEAL, "#149AA0", "#17809A", "#1E5F85", "#213F66", "#1C3A5E"]
        alphas = [1, 1, 1, 1, 0.9, 0.75]
    else:
        colors = ["#5FD1CC", "#8FE0DC", "#C9EFEE", "#FFFFFF", "#FFFFFF", "#FFFFFF"]
        alphas = [1, 1, 0.95, 0.9, 0.7, 0.5]
    widths_units = np.array([0.052, 0.048, 0.044, 0.040, 0.036, 0.032]) * r * weight
    cs = ax.contour(
        cx + X * r, cy + Y * r, Z, levels=levels, colors=colors,
        linewidths=list(widths_units * pts_per_unit),
    )
    colls = cs.collections if hasattr(cs, "collections") else [cs]
    for coll in colls:
        coll.set_clip_path(clip)
        coll.set_capstyle("round")
        coll.set_joinstyle("round")
    if hasattr(cs, "collections"):
        for coll, a in zip(colls, alphas):
            coll.set_alpha(a)
    else:
        # single collection (mpl >= 3.8): per-level alpha via RGBA edge colours
        from matplotlib.colors import to_rgba
        cs.set_edgecolor([to_rgba(c, a) for c, a in zip(colors, alphas)])
    # thin ring closes the disc
    ring_col = "#1C3A5E" if mode == "light" else "#FFFFFF"
    ax.add_patch(
        Circle((cx, cy), r, facecolor="none", edgecolor=ring_col,
               linewidth=0.026 * r * weight * pts_per_unit, alpha=0.9 if mode == "light" else 0.55)
    )
    # minima markers (the two states)
    dot = TEAL if mode == "light" else "#5FD1CC"
    for (mx, my, ms) in [(0.34, -0.14, 0.075), (-0.42, 0.32, 0.062)]:
        ax.add_patch(Circle((cx + mx * r, cy + my * r), ms * r, facecolor=dot, edgecolor="none"))


def text_path(s, prop, size, x, y):
    tp = TextPath((0, 0), s, size=size, prop=prop)
    return tp.transformed(Affine2D().translate(x, y)), tp.get_extents().width


def render(kind, mode, fname, fmt=("svg", "png"), dpi=300):
    """kind: 'logo' (icon + wordmark), 'icon', 'banner' (icon + wordmark + tagline)."""
    light = mode == "light"
    text_col = NAVY if light else "#FFFFFF"
    accent = TEAL if light else "#5FD1CC"
    tag_col = GREY if light else "#B9C6D3"

    r = 1.0
    if kind == "icon":
        span = 2.3
        fig_h = 4.0
        fig = plt.figure(figsize=(fig_h, fig_h))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(-span / 2, span / 2)
        ax.set_ylim(-span / 2, span / 2)
        draw_icon(ax, 0, 0, r, mode, pts_per_unit=fig_h * 72 / span, weight=1.25)
    else:
        size = 0.92
        gap = 0.46
        x0 = r + gap
        p1, w1 = text_path("Conform", FONT_LIGHT, size, x0, -0.33)
        p2, w2 = text_path("Atlas", FONT_SEMI, size, x0 + w1 + 0.13, -0.33)
        right = x0 + w1 + 0.13 + w2
        tag = None
        if kind == "banner":
            tag, wt = text_path(
                "Mapping conformational landscapes from molecular simulations", FONT_REG, 0.30, x0 + 0.03, -0.95
            )
            right = max(right, x0 + 0.03 + wt)
        xlim = (-1.22, right + 0.22)
        ylim = (-1.22 if kind == "logo" else -1.35, 1.22)
        span_y = ylim[1] - ylim[0]
        fig_h = 2.8
        fig_w = fig_h * (xlim[1] - xlim[0]) / span_y
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        draw_icon(ax, 0, 0, r, mode, pts_per_unit=fig_h * 72 / span_y)
        ax.add_patch(PathPatch(p1, facecolor=text_col, edgecolor="none"))
        ax.add_patch(PathPatch(p2, facecolor=text_col, edgecolor="none"))
        if kind == "banner":
            ax.add_patch(PathPatch(tag, facecolor=tag_col, edgecolor="none"))
            ax.plot([x0 + 0.03, x0 + 0.03 + 1.1], [-0.62, -0.62], color=accent,
                    lw=0.032 * fig_h * 72 / span_y, solid_capstyle="round")
    ax.set_aspect("equal")
    ax.axis("off")
    for f in fmt:
        fig.savefig(OUT / f"{fname}.{f}", dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


render("logo", "light", "conformatlas-logo")
render("logo", "dark", "conformatlas-logo-white")
render("banner", "light", "conformatlas-banner")
render("banner", "dark", "conformatlas-banner-white")
render("icon", "light", "conformatlas-icon", dpi=256)
render("icon", "dark", "conformatlas-icon-white", dpi=256)



def on_white(src: str, dst: str, pad_frac: float = 0.08) -> None:
    """Composite a transparent export on an opaque white card (README header:
    renders identically on GitHub's light and dark themes)."""
    from PIL import Image

    im = Image.open(OUT / f"{src}.png").convert("RGBA")
    pad_x, pad_y = int(im.width * pad_frac), int(im.height * pad_frac * 1.6)
    card = Image.new("RGBA", (im.width + 2 * pad_x, im.height + 2 * pad_y), (255, 255, 255, 255))
    card.alpha_composite(im, (pad_x, pad_y))
    card.convert("RGB").save(OUT / f"{dst}.png", optimize=True)


on_white("conformatlas-logo", "conformatlas-logo-on-white")
on_white("conformatlas-banner", "conformatlas-banner-on-white")

print(sorted(p.name for p in OUT.glob("conformatlas-*")))
