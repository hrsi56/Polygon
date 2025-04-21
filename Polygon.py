# polygon_drawer.py – enhanced
"""Streamlit app – precise polygon drawing with:
1.  Diagonal info (length + partial angles) in small grey / purple labels.
2.  Polygon area shown at centroid.
3.  Axis‑aligned bounding rectangle with width & height labels.
4.  PNG / SVG export buttons.
No external deps beyond numpy / matplotlib / streamlit.
"""

from __future__ import annotations

import io
import math
import string
from dataclasses import dataclass
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Arc

TOL = 1e-6  # closure tolerance

# ---------------------------------------------------------------------------
# Helper geometry
# ---------------------------------------------------------------------------

def vertex_names(n: int) -> List[str]:
    letters = string.ascii_uppercase
    out: list[str] = []
    for i in range(n):
        div, mod = divmod(i, 26)
        out.append(letters[mod] if div == 0 else letters[div - 1] + letters[mod])
    return out


def shoelace_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_centroid(pts: np.ndarray) -> np.ndarray:
    """Return centroid using shoelace formula (assumes non‑self‑intersecting)."""
    x = pts[:, 0]
    y = pts[:, 1]
    a = (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    A = 0.5 * a
    c_x = np.sum((x + np.roll(x, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y)) / (6 * A)
    c_y = np.sum((y + np.roll(y, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y)) / (6 * A)
    return np.array([c_x, c_y])


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    return math.degrees(math.acos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))


@dataclass
class PolygonData:
    pts: np.ndarray
    lengths: List[float]
    angles_int: List[float]

    @property
    def names(self):
        return vertex_names(len(self.pts))

# ---------------------------------------------------------------------------
# Angle repair & circumscribed construction
# ---------------------------------------------------------------------------

def repaired_angles(n: int, angs: Sequence[float] | None):
    if angs is None:
        return None
    factor = (n - 2) * 180.0 / sum(angs)
    return [a * factor for a in angs]


def circumscribed_polygon(lengths: Sequence[float]) -> PolygonData:
    n = len(lengths)
    L = np.asarray(lengths, float)
    R_lo = max(L) / 2 + 1e-9
    R_hi = 1e6

    def total(R):
        return np.sum(2 * np.arcsin(np.clip(L / (2 * R), -1 + 1e-12, 1 - 1e-12)))

    for _ in range(60):
        mid = 0.5 * (R_lo + R_hi)
        if total(mid) > 2 * math.pi:
            R_lo = mid
        else:
            R_hi = mid
    R = 0.5 * (R_lo + R_hi)
    central = 2 * np.arcsin(L / (2 * R))
    theta = np.concatenate(([0.0], np.cumsum(central)))[:-1]
    pts = np.stack([R * np.cos(theta), R * np.sin(theta)], axis=1)
    angles = [math.degrees(math.pi - 0.5 * (central[i - 1] + central[i])) for i in range(n)]
    return PolygonData(pts, list(L), angles)


# ---------------------------------------------------------------------------
# Build polygon from lengths + angles, ensuring closure
# ---------------------------------------------------------------------------

def polygon_from_lengths_angles(lengths: Sequence[float], angles: Sequence[float]):
    n = len(lengths)
    L = np.asarray(lengths, float)
    ext = np.radians(180.0 - np.asarray(angles))
    heads = np.zeros(n)
    heads[1:] = np.cumsum(ext[:-1])
    vecs = np.stack([L * np.cos(heads), L * np.sin(heads)], axis=1)

    gap = vecs.sum(axis=0)
    if np.hypot(*gap) > TOL:
        vecs += (-gap * (L / L.sum())[:, None])
        if np.hypot(*vecs.sum(axis=0)) > TOL:
            vecs[-1] -= vecs.sum(axis=0)

    pts = np.concatenate([[np.zeros(2)], np.cumsum(vecs, axis=0)])[:-1]
    lengths_corr = np.linalg.norm(vecs, axis=1).tolist()

    angles_corr = []
    for i in range(n):
        v1, v2 = pts[i - 1] - pts[i], pts[(i + 1) % n] - pts[i]
        angles_corr.append(angle_between(v1, v2))

    return PolygonData(pts, lengths_corr, angles_corr)

# ---------------------------------------------------------------------------
# Diagonals & bounding rectangle
# ---------------------------------------------------------------------------

def diagonals_info(pts: np.ndarray):
    n = len(pts)
    info = []
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            v = pts[j] - pts[i]
            length = float(np.linalg.norm(v))
            def partial(idx, vec):
                s1 = pts[idx - 1] - pts[idx]
                s2 = pts[(idx + 1) % n] - pts[idx]
                return min(angle_between(vec, s1), angle_between(vec, s2))
            info.append(dict(i=i, j=j, length=length,
                             ang_i=partial(i,  v),
                             ang_j=partial(j, -v)))
    return info


def bounding_rect(pts: np.ndarray):
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    rect = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    return rect, xmax - xmin, ymax - ymin


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_polygon(poly: PolygonData):
    n = len(poly.pts)
    names = poly.names
    pts_closed = np.vstack([poly.pts, poly.pts[0]])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect('equal'); ax.axis('off')

    # polygon edges
    ax.plot(pts_closed[:,0], pts_closed[:,1], '-o', lw=2)

    # diagonals with labels
    diags = diagonals_info(poly.pts)
    min_len = min(poly.lengths)
    for d in diags:
        p1, p2 = poly.pts[d['i']], poly.pts[d['j']]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '--', lw=0.8, color='gray', alpha=0.6)
        mid = 0.5*(p1+p2)
        ax.text(*mid, f"{d['length']:.2f}", fontsize=6, color='gray', ha='center', va='center')
        for idx, ang in ((d['i'], d['ang_i']), (d['j'], d['ang_j'])):
            base = poly.pts[idx]
            vec = (p2 if idx==d['i'] else p1) - base
            vec = vec / np.linalg.norm(vec)
            pos = base + vec * (0.12*min_len)
            ax.text(*pos, f"{ang:.1f}°", fontsize=6, color='purple', ha='center', va='center')

    # vertices & side lengths
    for i, (x,y) in enumerate(poly.pts):
        ax.text(x, y, names[i], fontsize=10, weight='bold', color='blue', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='circle,pad=0.25'))
        mid = 0.5*(poly.pts[i] + poly.pts[(i+1)%n])
        ax.text(*mid, f"{poly.lengths[i]:.2f}", fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'), ha='center', va='center')

    # internal angles arcs
    for i in range(n):
        p = poly.pts[i]
        v_prev = poly.pts[i-1] - p
        v_next = poly.pts[(i+1)%n] - p
        ang = poly.angles_int[i]
        bis = v_prev/np.linalg.norm(v_prev) + v_next/np.linalg.norm(v_next)
        if np.allclose(bis,0):
            bis = np.array([v_next[1], -v_next[0]])
        bis/=np.linalg.norm(bis)
        txt = p + bis*(0.18*min_len)
        start = math.degrees(math.atan2(v_prev[1], v_prev[0])); end = start - (180-ang)
        ax.add_patch(Arc(p, 0.36*min_len, 0.36*min_len, theta1=end, theta2=start, lw=1, color='red'))
        ax.text(*txt, f"{ang:.1f}°", fontsize=8, color='red', ha='center', va='center')

    # area label at centroid
    centroid = polygon_centroid(poly.pts)
    area_val = shoelace_area(poly.pts)
    ax.text(*centroid, f"Area = {area_val:.2f}", fontsize=10, color='green', ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # bounding rectangle
    rect, w, h = bounding_rect(poly.pts)
    rect_closed = np.vstack([rect, rect[0]])
    ax.plot(rect_closed[:,0], rect_closed[:,1], 'k-.', lw=1.0, alpha=0.5)
    # width & height labels
    bx, by = rect[0], rect[1]   # bottom left, bottom right
    tx, ty = rect[3], rect[2]   # top left, top right
    mid_w = 0.5*(bx+by)
    mid_h = 0.5*(by+ty)
    ax.text(*mid_w, f"w={w:.2f}", fontsize=8, color='black', ha='center', va='bottom')
    ax.text(*mid_h, f"h={h:.2f}", fontsize=8, color='black', ha='left', va='center')

    return fig, area_val, w, h, diags

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title='Polygon Drawer', layout='centered')
    st.title('📐 Polygon Drawer – enhanced')

    n = st.number_input('Number of sides', 3, 12, 4, 1)

    st.subheader('Side lengths')
    lengths = [st.number_input(f'Length {i+1}', 0.01, 1000.0, 1.0, 0.1, key=f'L{i}') for i in range(n)]

    use_ang = st.checkbox('Provide internal angles?')
    if use_ang:
        st.subheader('Internal angles (°)')
        angles = [st.number_input(f'∠ {vertex_names(n)[i]}', 1.0, 179.0, round(180*(n-2)/n,1), 1.0, key=f'A{i}') for i in range(n)]
        poly = polygon_from_lengths_angles(lengths, repaired_angles(n, angles))
    else:
        poly = circumscribed_polygon(lengths)

    if st.button('Draw polygon', use_container_width=True):
        fig, area_val, w, h, diags = draw_polygon(poly)
        st.pyplot(fig, use_container_width=True)

        st.markdown('### Numerical data')
        st.write({'Area': round(area_val,4), 'Bounding width': round(w,4), 'Bounding height': round(h,4)})

        st.markdown('### Diagonals')
        st.write({f"{poly.names[d['i']]}{poly.names[d['j']]}": {
            'Length': round(d['length'],3),
            f"∠ at {poly.names[d['i']]}": round(d['ang_i'],1),
            f"∠ at {poly.names[d['j']]}": round(d['ang_j'],1)} for d in diags})

        # export buttons
        png_buf = io.BytesIO(); fig.savefig(png_buf, format='png', dpi=300, bbox_inches='tight'); png_buf.seek(0)
        st.download_button('Download PNG', png_buf, 'polygon.png', 'image/png')

        svg_buf = io.BytesIO(); fig.savefig(svg_buf, format='svg', bbox_inches='tight'); svg_buf.seek(0)
        st.download_button('Download SVG', svg_buf, 'polygon.svg', 'image/svg+xml')


if __name__ == '__main__':
    main()
