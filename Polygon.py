# polygon_drawer.py – v3.1.1  (Python 3.9+)
# -----------------------------------------------------------
# Streamlit app – precise, always‑closed polygon drawing
#
# Features
#   • Diagonals: length + partial angles (small grey/purple labels)
#   • Polygon area (shoelace formula)
#   • Minimum‑area *rotated* bounding rectangle (needs Shapely)
#   • Live rotate & zoom sliders
#   • PNG / SVG export buttons
# -----------------------------------------------------------

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
from shapely.geometry import Polygon as ShpPoly     # Shapely ≥ 2.0 required

TOL = 1e-6           # geometric‑closure tolerance


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def names_seq(n: int) -> List[str]:
    """Return vertex labels A, B, … Z, AA, AB …"""
    letters = string.ascii_uppercase
    out: list[str] = []
    for i in range(n):
        div, mod = divmod(i, 26)
        out.append(letters[mod] if div == 0 else letters[div - 1] + letters[mod])
    return out


def shoelace_area(pts: np.ndarray) -> float:
    """Positive polygon area via shoelace formula."""
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


@dataclass
class PolyBase:
    pts: np.ndarray           # (n, 2)
    lengths: List[float]      # corrected side lengths
    angles_int: List[float]   # internal angles (deg)
    area: float               # unsigned area


# ---------------------------------------------------------------------------
# Polygon construction
# ---------------------------------------------------------------------------
def repair_angles(n: int, angs: Sequence[float] | None) -> List[float] | None:
    """Scale angles so their sum equals (n‑2)*180°."""
    if angs is None:
        return None
    factor = (n - 2) * 180.0 / sum(angs)
    return [a * factor for a in angs]


def circumscribed(lengths: Sequence[float]) -> PolyBase:
    """Place vertices on a circle ⇒ maximum‑area polygon for given sides."""
    n = len(lengths)
    L = np.asarray(lengths, float)

    R_lo = max(L) / 2 + 1e-9          # radius must satisfy L ≤ 2R
    R_hi = 1e6                        # large upper bound

    def total_angle(R: float) -> float:
        return np.sum(2 * np.arcsin(np.clip(L / (2 * R), -1 + 1e-12, 1 - 1e-12)))

    for _ in range(60):               # binary search for radius
        mid = 0.5 * (R_lo + R_hi)
        if total_angle(mid) > 2 * math.pi:
            R_lo = mid
        else:
            R_hi = mid
    R = 0.5 * (R_lo + R_hi)

    central = 2 * np.arcsin(L / (2 * R))          # central angles (rad)
    theta   = np.concatenate(([0.0], np.cumsum(central)))[:-1]
    pts     = np.stack([R * np.cos(theta), R * np.sin(theta)], axis=1)

    angles_int = [
        math.degrees(math.pi - 0.5 * (central[i - 1] + central[i]))
        for i in range(n)
    ]
    return PolyBase(pts, list(L), angles_int, shoelace_area(pts))


def build_polygon(lengths: Sequence[float], angs_deg: Sequence[float]) -> PolyBase:
    """General polygon from side lengths + internal angles (always closes)."""
    n = len(lengths)
    L = np.asarray(lengths, float)

    ext   = np.radians(180.0 - np.asarray(angs_deg))     # exterior turns
    heads = np.zeros(n)
    heads[1:] = np.cumsum(ext[:-1])

    vecs = np.stack([L * np.cos(heads), L * np.sin(heads)], axis=1)

    # Bowditch closure correction
    gap = vecs.sum(axis=0)
    if np.hypot(*gap) > TOL:
        vecs += (-gap * (L / L.sum())[:, None])          # distribute gap
        gap = vecs.sum(axis=0)
    if np.hypot(*gap) > TOL:                             # last‑resort tweak
        vecs[-1] -= gap

    pts = np.concatenate([[np.zeros(2)], np.cumsum(vecs, axis=0)])[:-1]
    lengths_corr = np.linalg.norm(vecs, axis=1).tolist()

    angles_corr = []
    for i in range(n):
        v1, v2 = pts[i - 1] - pts[i], pts[(i + 1) % n] - pts[i]
        ang = math.degrees(
            math.acos(np.clip(np.dot(v1, v2) /
                              (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
        )
        angles_corr.append(ang)

    return PolyBase(pts, lengths_corr, angles_corr, shoelace_area(pts))


# ---------------------------------------------------------------------------
# Extra geometry: diagonals & bounding rectangle
# ---------------------------------------------------------------------------
def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    return math.degrees(
        math.acos(np.clip(np.dot(u, v) /
                          (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))
    )


def diagonals_info(pts: np.ndarray) -> List[dict]:
    n = len(pts)
    diags = []
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:           # skip closing edge
                continue
            v = pts[j] - pts[i]
            length = float(np.linalg.norm(v))

            def part(idx, vec):
                s1, s2 = pts[idx - 1] - pts[idx], pts[(idx + 1) % n] - pts[idx]
                return min(angle_between(vec, s1), angle_between(vec, s2))

            diags.append(dict(
                i=i, j=j, length=length,
                ang_i=part(i,  v),
                ang_j=part(j, -v),
            ))
    return diags


def bounding_rectangle(pts: np.ndarray):
    rect = ShpPoly(pts).minimum_rotated_rectangle
    coords = np.array(rect.exterior.coords)[:-1]
    w = np.linalg.norm(coords[1] - coords[0])
    h = np.linalg.norm(coords[2] - coords[1])
    return coords, w, h, w * h


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw(poly: PolyBase, rot_deg: float, scale: float):
    # rotate & scale
    R = np.array([[math.cos(math.radians(rot_deg)), -math.sin(math.radians(rot_deg))],
                  [math.sin(math.radians(rot_deg)),  math.cos(math.radians(rot_deg))]])
    pts = poly.pts @ R.T * scale

    diags = diagonals_info(pts)
    rect, rw, rh, r_area = bounding_rectangle(pts)

    names = names_seq(len(pts))
    pts_closed = np.vstack([pts, pts[0]])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal"); ax.axis("off")
    ax.plot(pts_closed[:, 0], pts_closed[:, 1], "-o", lw=2)

    min_len = min(poly.lengths) * scale

    # -- diagonals ----------------------------------------------------------
    for d in diags:
        p1, p2 = pts[d["i"]], pts[d["j"]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "--", lw=0.8,
                color="gray", alpha=0.6)

        mid = 0.5 * (p1 + p2)
        ax.text(*mid, f"{d['length']:.2f}",
                fontsize=6, color="gray", ha="center", va="center")

        # partial‑angle labels at endpoints
        for idx, ang in ((d["i"], d["ang_i"]), (d["j"], d["ang_j"])):
            base = pts[idx]
            vec  = (p2 if idx == d["i"] else p1) - base
            vec /= np.linalg.norm(vec)
            pos  = base + vec * (0.12 * min_len)
            ax.text(*pos, f"{ang:.1f}°",
                    fontsize=6, color="purple", ha="center", va="center")

    # -- sides & vertices ---------------------------------------------------
    for i, (x, y) in enumerate(pts):
        ax.text(x, y, names[i], fontsize=10, weight="bold", color="blue",
                ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.8,
                          boxstyle="circle,pad=0.25"))
        mid = 0.5 * (pts[i] + pts[(i + 1) % len(pts)])
        ax.text(*mid, f"{poly.lengths[i] * scale:.2f}",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                ha="center", va="center")

    # -- internal angles ----------------------------------------------------
    for i in range(len(pts)):
        p = pts[i]
        v_prev = pts[i - 1] - p
        v_next = pts[(i + 1) % len(pts)] - p
        ang = poly.angles_int[i]

        bis = v_prev / np.linalg.norm(v_prev) + v_next / np.linalg.norm(v_next)
        if np.allclose(bis, 0):                       # straight – shouldn't occur
            bis = np.array([v_next[1], -v_next[0]])
        bis /= np.linalg.norm(bis)

        txt = p + bis * (0.18 * min_len)
        start = math.degrees(math.atan2(v_prev[1], v_prev[0]))
        end   = start - (180 - ang)
        ax.add_patch(Arc(p, 0.36 * min_len, 0.36 * min_len,
                         theta1=end, theta2=start, lw=1, color="red"))
        ax.text(*txt, f"{ang:.1f}°",
                fontsize=8, color="red", ha="center", va="center")

    # -- bounding rectangle -------------------------------------------------
    rect_c = np.vstack([rect, rect[0]])
    ax.plot(rect_c[:, 0], rect_c[:, 1], "k-.", lw=1, alpha=0.5)

    summary = {
        "area_scaled": poly.area * scale ** 2,
        "rect_w": rw,
        "rect_h": rh,
        "rect_area": r_area,
        "diagonals": diags,
    }
    return fig, summary


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config("Polygon Drawer", layout="centered")
    st.title("📐 Polygon Drawer – closed · interactive · exportable")

    n = st.number_input("Number of sides", 3, 12, 4, 1)

    st.subheader("Side lengths")
    lengths = [
        st.number_input(f"Length {i + 1}", 0.01, 1000.0, 1.0, 0.1, key=f"L{i}")
        for i in range(n)
    ]

    # optional angles
    if st.checkbox("Provide internal angles?"):
        st.subheader("Internal angles (°)")
        angles = [
            st.number_input(
                f"∠ {names_seq(n)[i]}", 1.0, 179.0,
                round(180 * (n - 2) / n, 1), 1.0, key=f"A{i}"
            )
            for i in range(n)
        ]
        poly = build_polygon(lengths, repair_angles(n, angles))
    else:
        poly = circumscribed(lengths)

    rot  = st.slider("Rotate (°)",      0.0, 360.0, 0.0, 1.0)
    zoom = st.slider("Zoom / scale ×",  0.2,   3.0, 1.0, 0.05)

    fig, info = draw(poly, rot, zoom)
    st.pyplot(fig, use_container_width=True)

    st.markdown("### Polygon data")
    st.write({
        "Scaled area": round(info["area_scaled"], 4),
        "Bounding rect W×H": f"{info['rect_w']:.2f} × {info['rect_h']:.2f}",
        "Bounding rect area": round(info["rect_area"], 4),
    })

    st.markdown("### Diagonals")
    st.write({
        f"{names_seq(n)[d['i']]}{names_seq(n)[d['j']]}": {
            "Length": round(d["length"], 3),
            f"∠ at {names_seq(n)[d['i']]}": round(d["ang_i"], 1),
            f"∠ at {names_seq(n)[d['j']]}": round(d["ang_j"], 1),
        }
        for d in info["diagonals"]
    })

    # -- downloads ----------------------------------------------------------
    png_buf = io.BytesIO()
    fig.savefig(png_buf, format="png", dpi=300, bbox_inches="tight")
    png_buf.seek(0)
    st.download_button("Download PNG", png_buf,
                       "polygon.png", "image/png")

    svg_buf = io.BytesIO()
    fig.savefig(svg_buf, format="svg", bbox_inches="tight")
    svg_buf.seek(0)
    st.download_button("Download SVG", svg_buf,
                       "polygon.svg", "image/svg+xml")


if __name__ == "__main__":
    main()