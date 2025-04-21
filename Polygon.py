# polygon_drawer.py – v3.1  (requires Shapely)
# -----------------------------------------------------------
# Streamlit app – precise, always‑closed polygon drawing
# Features
#   • Diagonals: length + partial angles (small grey/purple labels)
#   • Polygon area (shoelace)
#   • Smallest *rotated* bounding rectangle using Shapely
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
from shapely.geometry import Polygon as ShpPoly   # ← Shapely is now mandatory

TOL = 1e-6  # closure tolerance


# ---------- helpers ---------------------------------------------------------
def names_seq(n: int) -> List[str]:
    letters = string.ascii_uppercase
    return [(letters[i % 26] if i < 26 else letters[i // 26 - 1] + letters[i % 26])
            for i in range(n)]


def shoelace_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


@dataclass
class PolyBase:
    pts: np.ndarray
    lengths: List[float]
    angles_int: List[float]
    area: float


# ---------- construction ----------------------------------------------------
def repair_angles(n: int, angs: Sequence[float] | None) -> List[float] | None:
    if angs is None:
        return None
    target = (n - 2) * 180.0
    k = target / sum(angs)
    return [a * k for a in angs]


def circumscribed(lengths: Sequence[float]) -> PolyBase:
    n = len(lengths)
    L = np.asarray(lengths, float)
    Rlo, Rhi = max(L) / 2 + 1e-9, 1e6

    def total(R: float) -> float:
        return np.sum(2 * np.arcsin(np.clip(L / (2 * R), -1 + 1e-12, 1 - 1e-12)))

    for _ in range(60):
        mid = 0.5 * (Rlo + Rhi)
        (Rlo if total(mid) > 2 * math.pi else Rhi) = mid
    R = 0.5 * (Rlo + Rhi)

    central = 2 * np.arcsin(L / (2 * R))
    theta = np.concatenate(([0.0], np.cumsum(central)))[:-1]
    pts = np.stack([R * np.cos(theta), R * np.sin(theta)], axis=1)
    angles_int = [math.degrees(math.pi - 0.5 * (central[i - 1] + central[i])) for i in range(n)]
    return PolyBase(pts, list(L), angles_int, shoelace_area(pts))


def build_polygon(lengths: Sequence[float], angs_deg: Sequence[float]) -> PolyBase:
    n = len(lengths)
    L = np.asarray(lengths, float)
    ext = np.radians(180.0 - np.asarray(angs_deg, float))
    headings = np.zeros(n)
    headings[1:] = np.cumsum(ext[:-1])
    vecs = np.stack([L * np.cos(headings), L * np.sin(headings)], axis=1)

    gap = vecs.sum(axis=0)
    if np.hypot(*gap) > TOL:
        vecs += (-gap * (L / L.sum())[:, None])
        gap = vecs.sum(axis=0)
    if np.hypot(*gap) > TOL:
        vecs[-1] -= gap

    pts = np.concatenate([[np.zeros(2)], np.cumsum(vecs, axis=0)])[:-1]
    lengths_corr = np.linalg.norm(vecs, axis=1).tolist()

    angs_corr = []
    for i in range(n):
        v1 = pts[i - 1] - pts[i]
        v2 = pts[(i + 1) % n] - pts[i]
        ang = math.degrees(math.acos(np.clip(np.dot(v1, v2) /
                                             (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)))
        angs_corr.append(ang)

    return PolyBase(pts, lengths_corr, angs_corr, shoelace_area(pts))


# ---------- extra geometry --------------------------------------------------
def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    return math.degrees(math.acos(np.clip(np.dot(u, v) /
                                         (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))


def diagonals_info(pts: np.ndarray) -> List[dict]:
    n = len(pts)
    diags = []
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            v_ij = pts[j] - pts[i]
            length = float(np.linalg.norm(v_ij))

            def part(idx, vec):
                s1 = pts[idx - 1] - pts[idx]
                s2 = pts[(idx + 1) % n] - pts[idx]
                return min(angle_between(vec, s1), angle_between(vec, s2))

            diags.append(dict(
                i=i, j=j, length=length,
                ang_i=part(i, v_ij), ang_j=part(j, -v_ij)
            ))
    return diags


def bounding_rectangle(pts: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    rect = ShpPoly(pts).minimum_rotated_rectangle
    coords = np.array(rect.exterior.coords)[:-1]
    w = np.linalg.norm(coords[1] - coords[0])
    h = np.linalg.norm(coords[2] - coords[1])
    return coords, w, h, w * h


# ---------- drawing ---------------------------------------------------------
def draw(poly: PolyBase, rotation: float, scale: float):
    R = np.array([[math.cos(math.radians(rotation)), -math.sin(math.radians(rotation))],
                  [math.sin(math.radians(rotation)), math.cos(math.radians(rotation))]])
    pts = poly.pts @ R.T * scale
    diags = diagonals_info(pts)
    rect, rw, rh, r_area = bounding_rectangle(pts)

    names = names_seq(len(pts))
    pts_closed = np.vstack([pts, pts[0]])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.plot(pts_closed[:, 0], pts_closed[:, 1], "-o", lw=2)

    min_len = min(poly.lengths) * scale

    # diagonals
    for d in diags:
        p1, p2 = pts[d["i"]], pts[d["j"]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "--", lw=0.8, color="gray", alpha=0.6)
        mid = (p1 + p2) * 0.5
        ax.text(*mid, f"{d['length']:.2f}", fontsize=6, color="gray",
                ha="center", va="center")
        for idx, ang in ((d["i"], d["ang_i"]), (d["j"], d["ang_j"])):
            base = pts[idx]
            vec = (p2 if idx == d["i"] else p1) - base
            vec = vec / np.linalg.norm(vec)
            pos = base + vec * (0.12 * min_len)
            ax.text(*pos, f"{ang:.1f}°", fontsize=6, color="purple",
                    ha="center", va="center")

    # side labels & vertices
    for i, (x, y) in enumerate(pts):
        ax.text(x, y, names[i], fontsize=10, weight="bold", color="blue",
                ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.8, boxstyle="circle,pad=0.25"))
        mid = (pts[i] + pts[(i + 1) % len(pts)]) * 0.5
        ax.text(*mid, f"{poly.lengths[i] * scale:.2f}", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                ha="center", va="center")

    # internal angles
    for i in range(len(pts)):
        p = pts[i]
        v_prev = pts[i - 1] - p
        v_next = pts[(i + 1) % len(pts)] - p
        ang = poly.angles_int[i]
        bis = v_prev / np.linalg.norm(v_prev) + v_next / np.linalg.norm(v_next)
        if np.allclose(bis, 0):
            bis = np.array([v_next[1], -v_next[0]])
        bis = bis / np.linalg.norm(bis)
        txt = p + bis * (0.18 * min_len)
        start = math.degrees(math.atan2(v_prev[1], v_prev[0]))
        end = start - (180 - ang)
        ax.add_patch(Arc(p, 0.36 * min_len, 0.36 * min_len,
                         theta1=end, theta2=start, lw=1, color="red"))
        ax.text(*txt, f"{ang:.1f}°", fontsize=8, color="red",
                ha="center", va="center")

    # bounding rectangle
    rect_closed = np.vstack([rect, rect[0]])
    ax.plot(rect_closed[:, 0], rect_closed[:, 1], "k-.", lw=1, alpha=0.5)

    summary = {"area_scaled": poly.area * scale * scale,
               "rect_w": rw, "rect_h": rh, "rect_area": r_area,
               "diagonals": diags}
    return fig, summary


# ---------- Streamlit UI ----------------------------------------------------
def main():
    st.set_page_config("Polygon Drawer", layout="centered")
    st.title("📐 Polygon Drawer – closed, interactive, exportable")

    n = st.number_input("Number of sides", 3, 12, 4, 1, key="n")
    st.subheader("Side lengths")
    lengths = [st.number_input(f"Length {i+1}", 0.01, 1000.0, 1.0, 0.1, key=f"L{i}")
               for i in range(n)]

    if st.checkbox("Provide internal angles?"):
        st.subheader("Internal angles (°)")
        angles = [st.number_input(f"∠ {names_seq(n)[i]}", 1.0, 179.0,
                                  round(180*(n-2)/n, 1), 1.0, key=f"A{i}")
                  for i in range(n)]
        poly = build_polygon(lengths, repair_angles(n, angles))
    else:
        poly = circumscribed(lengths)

    rot = st.slider("Rotate (°)", 0.0, 360.0, 0.0, 1.0)
    zoom = st.slider("Zoom / scale", 0.2, 3.0, 1.0, 0.05)

    fig, info = draw(poly, rot, zoom)
    st.pyplot(fig, use_container_width=True)

    st.markdown("### Polygon data")
    st.write({
        "Scaled area": round(info["area_scaled"], 4),
        "Bounding rect W×H": f"{info['rect_w']:.2f} × {info['rect_h']:.2f}",
        "Bounding rect area": round(info["rect_area"], 4),
    })

    st.markdown("### Diagonals")
    st.write({
        f"{names_seq(n)[d['i']]}{names_seq(n)[d['j']]}": {
            "Length": round(d["length"], 3),
            f"∠ at {names_seq(n)[d['i']]}": round(d["ang_i"], 1),
            f"∠ at {names_seq(n)[d['j']]}": round(d["ang_j"], 1),
        } for d in info["diagonals"]
    })

    # exports
    png_buf = io.BytesIO()
    fig.savefig(png_buf, format="png", dpi=300, bbox_inches="tight")
    png_buf.seek(0)
    st.download_button("Download PNG", png_buf, "polygon.png", "image/png")

    svg_buf = io.BytesIO()
    fig.savefig(svg_buf, format="svg", bbox_inches="tight")
    svg_buf.seek(0)
    st.download_button("Download SVG", svg_buf, "polygon.svg", "image/svg+xml")


if __name__ == "__main__":
    main()