# polygon_drawer.py – v1.1 (2025‑04‑21)
# -----------------------------------------------------------
# Streamlit app – closed, proportional polygon drawing
#
# ✦  Validity check: stops with an error if the side‑length set cannot
#    form a polygon (e.g. 1 1 3 triangle, 5 1 2 1 quadrilateral).
# ✦  Diagonals: length + partial angles (tiny grey / purple labels).
# ✦  Area written at the centroid.
# ✦  Axis‑aligned bounding rectangle with width/height labels.
# ✦  Outward‑offset labels to reduce collisions.
# ✦  PNG / SVG export buttons.
# -----------------------------------------------------------
from __future__ import annotations

import datetime as dt
import io
import json
import math
import string
import zipfile
from dataclasses import dataclass
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Arc

TOL = 1e-6
LABEL_SHIFT = -0.05    # outward label offset (fraction of min side)

# ---------------------------------------------------------------------------
# Helper geometry
# ---------------------------------------------------------------------------
def vertex_names(n: int) -> List[str]:
    letters = string.ascii_uppercase
    return [(letters[i] if i < 26 else letters[i // 26 - 1] + letters[i % 26])
            for i in range(n)]


def shoelace_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def centroid(pts: np.ndarray) -> np.ndarray:
    x, y = pts[:, 0], pts[:, 1]
    a = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    A = 0.5 * a
    cx = np.sum((x + np.roll(x, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y)) / (6 * A)
    cy = np.sum((y + np.roll(y, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y)) / (6 * A)
    return np.array([cx, cy])


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    return math.degrees(
        math.acos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))
    )


def is_polygon_possible(lengths: Sequence[float]) -> bool:
    """Longest side must be strictly shorter than sum of the others."""
    L = sorted(lengths)
    return L[-1] < sum(L[:-1]) - 1e-9

# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------
@dataclass
class PolygonData:
    pts: np.ndarray
    lengths: List[float]
    angles_int: List[float]

    @property
    def names(self):
        return vertex_names(len(self.pts))

# ---------------------------------------------------------------------------
# Angle helpers
# ---------------------------------------------------------------------------
def repaired_angles(n: int, angs: Sequence[float] | None):
    if angs is None:
        return None
    factor = (n - 2) * 180.0 / sum(angs)
    return [a * factor for a in angs]

# ---------------------------------------------------------------------------
# Circumscribed (max‑area) polygon if no angles
# ---------------------------------------------------------------------------
def circumscribed_polygon(lengths: Sequence[float]) -> PolygonData:
    n = len(lengths)
    L = np.asarray(lengths, float)
    R_lo, R_hi = max(L) / 2 + 1e-9, 1e6

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
# Build polygon from lengths + angles (Bowditch correction)
# ---------------------------------------------------------------------------
def build_polygon(lengths: Sequence[float], angles: Sequence[float]) -> PolygonData:
    n = len(lengths)
    L = np.asarray(lengths, float)
    ext = np.radians(180.0 - np.asarray(angles))
    heads = np.zeros(n)
    heads[1:] = np.cumsum(ext[:-1])
    vecs = np.stack([L * np.cos(heads), L * np.sin(heads)], axis=1)

    gap = vecs.sum(axis=0)
    if np.hypot(*gap) > TOL:
        vecs += (-gap * (L / L.sum())[:, None])
        gap = vecs.sum(axis=0)
    if np.hypot(*gap) > TOL:
        vecs[-1] -= gap

    pts = np.concatenate([[np.zeros(2)], np.cumsum(vecs, axis=0)])[:-1]
    lengths_corr = np.linalg.norm(vecs, axis=1).tolist()
    angles_corr = [
        angle_between(pts[i - 1] - pts[i], pts[(i + 1) % n] - pts[i]) for i in range(n)
    ]
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

            def part(idx, vec):
                s1 = pts[idx - 1] - pts[idx]
                s2 = pts[(idx + 1) % n] - pts[idx]
                return min(angle_between(vec, s1), angle_between(vec, s2))

            info.append(
                dict(i=i, j=j, length=length, ang_i=part(i, v), ang_j=part(j, -v))
            )
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
    ax.set_aspect("equal")
    ax.axis("off")
    ax.plot(pts_closed[:, 0], pts_closed[:, 1], "-o", lw=2)

    # Diagonals ----------------------------------------------------------------
    diags = diagonals_info(poly.pts)
    min_len = min(poly.lengths)
    for d in diags:
        p1, p2 = poly.pts[d["i"]], poly.pts[d["j"]]
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            "--",
            lw=0.8,
            color="gray",
            alpha=0.6,
        )
        mid = 0.5 * (p1 + p2)
        ax.text(
            *mid,
            f"{d['length']:.2f}",
            fontsize=6,
            color="gray",
            ha="center",
            va="center",
        )
        for idx, ang in ((d["i"], d["ang_i"]), (d["j"], d["ang_j"])):
            base = poly.pts[idx]
            vec = (p2 if idx == d["i"] else p1) - base
            vec = vec / np.linalg.norm(vec)
            pos = base + vec * (0.12 * min_len)
            ax.text(
                *pos,
                f"{ang:.1f}°",
                fontsize=6,
                color="purple",
                ha="center",
                va="center",
            )

    # Vertex and side labels ---------------------------------------------------
    for i, (x, y) in enumerate(poly.pts):
        # outward normal (average of adjacent normals)
        e_prev = poly.pts[i] - poly.pts[i - 1]
        e_next = poly.pts[(i + 1) % n] - poly.pts[i]
        normal = np.array([-(e_prev[1] + e_next[1]), e_prev[0] + e_next[0]])
        normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) else normal
        ax.text(
            x + normal[0] * LABEL_SHIFT * min_len,
            y + normal[1] * LABEL_SHIFT * min_len,
            names[i],
            fontsize=9,
            weight="bold",
            color="blue",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.8, boxstyle="circle,pad=0.25"),
        )

        # side length
        mid = 0.5 * (poly.pts[i] + poly.pts[(i + 1) % n])
        edge = poly.pts[(i + 1) % n] - poly.pts[i]
        edge_normal = np.array([-edge[1], edge[0]])
        edge_normal = edge_normal / np.linalg.norm(edge_normal)
        ax.text(
            *(mid + edge_normal * LABEL_SHIFT * min_len),
            f"{poly.lengths[i]:.2f}",
            fontsize=7,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    # Internal angles ----------------------------------------------------------
    for i in range(n):
        p = poly.pts[i]
        v_prev = poly.pts[i - 1] - p
        v_next = poly.pts[(i + 1) % n] - p
        bis = v_prev / np.linalg.norm(v_prev) + v_next / np.linalg.norm(v_next)
        bis = bis / np.linalg.norm(bis) if np.linalg.norm(bis) else np.array(
            [v_next[1], -v_next[0]]
        )
        txt = p + bis * (0.23 * min_len)
        start = math.degrees(math.atan2(v_prev[1], v_prev[0]))
        end = start - (180 - poly.angles_int[i])
        ax.add_patch(
            Arc(
                p,
                0.36 * min_len,
                0.36 * min_len,
                theta1=end,
                theta2=start,
                lw=1,
                color="red",
            )
        )
        ax.text(
            *txt,
            f"{poly.angles_int[i]:.1f}°",
            fontsize=7,
            color="red",
            ha="center",
            va="center",
        )

    # Area at centroid ---------------------------------------------------------
    area_val = shoelace_area(poly.pts)
    ax.text(
        *(centroid(poly.pts)-[0,0.05]),
        f"Area = {area_val:.2f}",
        fontsize=9,
        color="green",
        ha="center",
        va="center",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    # Bounding rectangle -------------------------------------------------------
    rect, w, h = bounding_rect(poly.pts)
    rect_closed = np.vstack([rect, rect[0]])
    ax.plot(rect_closed[:, 0], rect_closed[:, 1], "k-.", lw=1.0, alpha=0.5)

    # width & height labels
    mid_w = 0.5 * (rect[0] + rect[1]) - [0.1 , 0.05]
    mid_h = 0.5 * (rect[1] + rect[2]) + [0.01 , 0.05]
    ax.text(
        *mid_w,
        f"w={w:.2f}",
        fontsize=8,
        ha="center",
        va="bottom",
        color="black",
    )
    ax.text(
        *mid_h,
        f"h={h:.2f}",
        fontsize=8,
        ha="left",
        va="center",
        color="black",
    )

    return fig, area_val, w, h, diags

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Polygon Drawer", layout="centered")
    st.title("📐 Polygon Drawer – enhanced")

    n = st.number_input("Number of sides", 3, 12, 4, 1)

    st.subheader("Side lengths")
    lengths = [
        st.number_input(f"Length {i + 1}", 0.01, 1000.0, 1.0, 0.1, key=f"L{i}")
        for i in range(n)
    ]

    # validity check (side‑length feasibility)
    if not is_polygon_possible(lengths):
        st.error(
            "⚠️  The side lengths cannot form a convex polygon "
            "(longest side ≥ sum of the others)."
        )
        st.stop()

    use_ang = st.checkbox("Provide internal angles?")
    if use_ang:
        st.subheader("Internal angles (°)")
        angles_in = [
            st.number_input(
                f"∠ {vertex_names(n)[i]}",
                1.0,
                179.0,
                round(180 * (n - 2) / n, 1),
                1.0,
                key=f"A{i}",
            )
            for i in range(n)
        ]
        poly = build_polygon(lengths, repaired_angles(n, angles_in))
    else:
        poly = circumscribed_polygon(lengths)

    if st.button("Draw polygon", use_container_width=True):
        fig, area_val, w, h, diags = draw_polygon(poly)
        st.pyplot(fig, use_container_width=True)

        # numeric data ------------------------------------
        area_val = shoelace_area(poly.pts)
        _, w, h = bounding_rect(poly.pts)
        diag_list = diagonals_info(poly.pts)

        num_data = {
            "Area": round(area_val, 4),
            "Bounding width": round(w, 4),
            "Bounding height": round(h, 4)
        }
        diag_data = {
            f"{poly.names[d['i']]}{poly.names[d['j']]}": {
                "Length": round(d["length"], 3),
                f"∠ at {poly.names[d['i']]}": round(d["ang_i"], 1),
                f"∠ at {poly.names[d['j']]}": round(d["ang_j"], 1),
            } for d in diag_list
        }

        st.markdown("### Numerical data")
        st.json(num_data, expanded=True)

        st.markdown("### Diagonals")
        st.json(diag_data, expanded=True)

        # prepare files ------------------------------
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M")
        base = f"YVD_Poligon_{ts}"

        # TXT (JSON)
        txt_content = json.dumps(
            {"Numerical data": num_data, "Diagonals": diag_data},
            indent=2
        ).encode()

        # PNG
        png_buf = io.BytesIO()
        fig.savefig(png_buf, format="png", dpi=300,
                    bbox_inches="tight")
        png_buf.seek(0)

        # PDF
        pdf_buf = io.BytesIO()
        fig.savefig(pdf_buf, format="pdf",
                    bbox_inches="tight")
        pdf_buf.seek(0)

        # SVG
        svg_buf = io.BytesIO()
        fig.savefig(svg_buf, format="svg",
                    bbox_inches="tight")
        svg_buf.seek(0)

        # ZIP archive
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w",
                             zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{base}.txt", txt_content)
            zf.writestr(f"{base}.png", png_buf.getvalue())
            zf.writestr(f"{base}.pdf", pdf_buf.getvalue())
            zf.writestr(f"{base}.svg", svg_buf.getvalue())
        zip_buf.seek(0)

        st.download_button("Download all (ZIP)",
                           zip_buf,
                           f"{base}.zip",
                           "application/zip")


if __name__ == "__main__":
    main()
