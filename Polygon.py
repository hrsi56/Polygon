from __future__ import annotations

"""
Polygon Drawer / Open Polyline Drawer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

• צורות סגורות (מצולעים) – חישוב אלכסונים, שטח, גבהים וכו׳  
• צורות פתוחות – קו שבור (polyline) – כל הצלעות נתונות, הזווית בראש A מתבטלת

בתיבה "צורה פתוחה" המשתמש קובע האם תחובר הצלע האחרונה לראשונה.

גרסה – April 2025
"""

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

TOL = 1e-6
LABEL_SHIFT = -0.05  # outward label offset (fraction of min side)

# ────────────────── Helpers ──────────────────────────────────────────────

def vertex_names(n: int) -> List[str]:
    letters = string.ascii_uppercase
    return [letters[i] if i < 26 else letters[i // 26 - 1] + letters[i % 26] for i in range(n)]


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """Return angle in degrees between two vectors."""
    return math.degrees(
        math.acos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))
    )


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


def is_polygon_possible(lengths: Sequence[float]) -> bool:
    """Basic polygon inequality – longest side shorter than sum of the rest."""
    L = sorted(lengths)
    return L[-1] < sum(L[:-1]) - 1e-9


# ────────────────── Data structure ───────────────────────────────────────
@dataclass
class PolygonData:
    pts: np.ndarray           # vertices in order (N×2)
    lengths: List[float]      # side‑length list (len = n_edges)
    angles_int: List[float]   # internal angles *per vertex* (len = len(pts)), None where not defined

    @property
    def names(self) -> List[str]:
        return vertex_names(len(self.pts))


# ────────────────── Builders ─────────────────────────────────────────────

def repaired_angles(n: int, angs: Sequence[float] | None):
    if angs is None:
        return None
    k = (n - 2) * 180.0 / sum(angs)
    return [a * k for a in angs]


# -- Closed polygon -------------------------------------------------------

def circumscribed_polygon(lengths: Sequence[float]) -> PolygonData:
    n = len(lengths)
    L = np.asarray(lengths, float)
    R_lo, R_hi = max(L) / 2 + 1e-9, 1e6

    def total(R: float) -> float:
        return np.sum(2 * np.arcsin(np.clip(L / (2 * R), -1 + 1e-12, 1 - 1e-12)))

    for _ in range(60):
        mid = 0.5 * (R_lo + R_hi)
        (R_lo, R_hi) = (mid, R_hi) if total(mid) > 2 * math.pi else (R_lo, mid)
    R = 0.5 * (R_lo + R_hi)

    central = 2 * np.arcsin(L / (2 * R))
    theta = np.concatenate(([0.0], np.cumsum(central)))[:-1]
    pts = np.stack([R * np.cos(theta), R * np.sin(theta)], axis=1)
    angles = [math.degrees(math.pi - 0.5 * (central[i - 1] + central[i])) for i in range(n)]
    return PolygonData(pts, list(L), angles)


# -- General closed polygon from lengths+angles --------------------------

def build_polygon(lengths: Sequence[float], angles: Sequence[float]) -> PolygonData:
    n = len(lengths)
    L = np.asarray(lengths, float)
    ext = np.radians(180.0 - np.asarray(angles))  # external turn
    heads = np.zeros(n)
    heads[1:] = np.cumsum(ext[:-1])
    vecs = np.stack([L * np.cos(heads), L * np.sin(heads)], axis=1)

    # adjust to close the gap
    gap = vecs.sum(axis=0)
    if np.hypot(*gap) > TOL:
        vecs += (-gap * (L / L.sum())[:, None])
        gap = vecs.sum(axis=0)
    if np.hypot(*gap) > TOL:
        vecs[-1] -= gap

    pts = np.concatenate([[np.zeros(2)], np.cumsum(vecs, axis=0)])[:-1]
    lengths_corr = np.linalg.norm(vecs, axis=1).tolist()
    angles_corr = [angle_between(pts[i - 1] - pts[i], pts[(i + 1) % n] - pts[i]) for i in range(n)]
    return PolygonData(pts, lengths_corr, angles_corr)


# -- Open polyline (first and last vertices are free) ---------------------

def build_polyline(lengths: Sequence[float], angles: Sequence[float]) -> PolygonData:
    """Build an open polyline given *segment* lengths and internal angles at all vertices
    (angle at the first vertex is ignored when drawing, but still used for geometry).
    The path starts at (0,0) and goes initially along the +x axis.
    """
    m = len(lengths)  # number of segments

    # heads[i] is heading (direction) of segment i in radians
    ext = np.radians(180.0 - np.asarray(angles))  # external turn angles
    heads = np.zeros(m)
    if m > 1:
        heads[1:] = np.cumsum(ext[:-1])

    L = np.asarray(lengths, float)
    vecs = np.stack([L * np.cos(heads), L * np.sin(heads)], axis=1)
    pts = np.concatenate([[np.zeros(2)], np.cumsum(vecs, axis=0)])  # m+1 vertices

    # compute internal angles for vertices 1..m-1; 0 and m are None
    int_angles = [None] * (m + 1)
    for i in range(1, m):
        int_angles[i] = angle_between(pts[i] - pts[i - 1], pts[i] - pts[i + 1])

    return PolygonData(pts, list(lengths), int_angles)


# ────────────────── Drawing ──────────────────────────────────────────────

def draw_shape(poly: PolygonData, closed: bool):
    """Draw closed polygon or open polyline."""
    pts = poly.pts
    n_edges = len(poly.lengths)
    names = poly.names

    # Build plotting array
    if closed:
        pts_plot = np.vstack([pts, pts[0]])
    else:
        pts_plot = pts

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.plot(pts_plot[:, 0], pts_plot[:, 1], "-o", lw=1.4, color="black", alpha=0.6)

    min_len = min(poly.lengths)

    # Vertices ----------------------------------------------------------------
    for i, (x, y) in enumerate(pts):
        prev_vec = pts[i] - pts[i - 1] if i > 0 else np.array([0.0, 0.0])
        next_vec = pts[i + 1] - pts[i] if (i < len(pts) - 1) else np.array([0.0, 0.0])
        normal = np.array([-(prev_vec[1] + next_vec[1]), prev_vec[0] + next_vec[0]])
        if np.linalg.norm(normal):
            normal /= np.linalg.norm(normal)
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

    # Edge length labels -------------------------------------------------------
    edge_iter = range(n_edges)
    for i in edge_iter:
        p1, p2 = pts[i], pts[i + 1] if not closed else pts[(i + 1) % len(pts)]
        mid = 0.5 * (p1 + p2)
        edge = p2 - p1
        edge_norm = np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)
        ax.text(
            *(mid + edge_norm * LABEL_SHIFT * min_len),
            f"{poly.lengths[i]:.2f}",
            fontsize=7,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            ha="center",
            va="center",
        )

    # Internal angles ----------------------------------------------------------
    if closed:
        angle_range = range(len(pts))
    else:
        angle_range = range(1, len(pts) - 1)  # skip first vertex A, skip last vertex

    for i in angle_range:
        if poly.angles_int[i] is None:
            continue
        p = pts[i]
        v_prev = pts[i - 1] - p
        v_next = pts[i + 1] - p if not closed else pts[(i + 1) % len(pts)] - p
        bis = v_prev / np.linalg.norm(v_prev) + v_next / np.linalg.norm(v_next)
        if not np.linalg.norm(bis):
            bis = np.array([v_next[1], -v_next[0]])
        else:
            bis /= np.linalg.norm(bis)
        txt = p + bis * (0.23 * min_len)
        ax.text(
            *txt,
            f"{poly.angles_int[i]:.1f}°",
            fontsize=7,
            color="red",
            ha="center",
            va="center",
        )

    # Additional annotations ---------------------------------------------------
    if closed:
        # area & bounding rectangle
        area_val = shoelace_area(pts)
        ax.text(
            *(centroid(pts) - np.array([0, 0.05])),
            f"Area = {area_val:.2f}",
            fontsize=9,
            color="green",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
    rect, w, h = bounding_rect(pts)
    rc = np.vstack([rect, rect[0]])
    ax.plot(rc[:, 0], rc[:, 1], "k-.", lw=1, alpha=0.5)
    HW = h * w
    mid_w = 0.5 * (rect[0] + rect[1]) - np.array([0.1, 0.05])
    mid_h = 0.5 * (rect[1] + rect[2]) + np.array([0.01, 0.05])
    ax.text(*mid_w, f"w={w:.2f}", fontsize=8, ha="center", va="bottom")
    ax.text(*mid_h, f"h={h:.2f}", fontsize=8, ha="left", va="center")
    ax.text(*(rect[0] + 0.03), f"Area={HW:.2f}", fontsize=8, ha="left", va="center")

    return fig


# Bounding rectangle ------------------------------------------------------

def bounding_rect(pts: np.ndarray):
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    rect = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    return rect, xmax - xmin, ymax - ymin


# ────────────────── Streamlit app ────────────────────────────────────────

def main():
    st.set_page_config(page_title="Polygon / Polyline Drawer", layout="centered")
    st.title("📐 Polygon & Polyline Drawer – לינקו")

    n_edges = st.number_input("Number of sides / segments", min_value=3, max_value=12, value=4, step=1)
    lengths = [
        st.number_input(f"Length {i + 1}", min_value=0.01, max_value=1000.0, value=1.0, step=0.1, key=f"L{i}")
        for i in range(int(n_edges))
    ]

    open_shape = st.checkbox("🔓 צורה פתוחה (הצלע הראשונה לא תתחבר לאחרונה)")

    provide_angles = st.checkbox("Provide internal angles?")
    if provide_angles:
        default_ang = round(180 * (n_edges - 2) / n_edges, 1)
        angles = [
            st.number_input(
                f"∠ {vertex_names(int(n_edges) if not open_shape else int(n_edges) + 1)[i]}",
                min_value=1.0,
                max_value=359.0,
                value=default_ang,
                step=1.0,
                key=f"A{i}",
            )
            for i in range(int(n_edges))
        ]
    else:
        angles = None

    if not open_shape and not is_polygon_possible(lengths):
        st.error("⚠️  Side lengths violate polygon inequality.")
        st.stop()

    # ـــ Build geometry ـــ
    poly: PolygonData | None = None
    if open_shape:
        if angles is None:
            st.error("For an open shape you must supply the angles, otherwise אין מספיק מידע לשרטט.")
            st.stop()
        poly = build_polyline(lengths, angles)
    else:
        if angles is None:
            poly = circumscribed_polygon(lengths)
        else:
            poly = build_polygon(lengths, repaired_angles(len(lengths), angles))

    # Draw button -------------------------------------------------------------
    if st.button("Draw", type="primary"):
        fig = draw_shape(poly, closed=not open_shape)
        st.pyplot(fig, use_container_width=True)

        # --- Output numerical data ------------------------------------------
        rect, w, h = bounding_rect(poly.pts)
        num_data = {"Bounding width": round(w, 4), "Bounding height": round(h, 4)}
        if not open_shape:
            num_data["Area"] = round(shoelace_area(poly.pts), 4)
            st.markdown("### Numerical data")
            st.json(num_data, expanded=True)
        else:
            st.info("הצורה פתוחה – אין שטח או אלכסונים לחשב.")
            st.markdown("### Bounding box")
            st.json(num_data, expanded=True)

        # --- Download ZIP ----------------------------------------------------
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M")
        base = f"YVD_{'OpenShape' if open_shape else 'Polygon'}_{ts}"

        png_buf, pdf_buf, svg_buf = io.BytesIO(), io.BytesIO(), io.BytesIO()
        fig.savefig(png_buf, format="png", dpi=300, bbox_inches="tight")
        fig.savefig(pdf_buf, format="pdf", bbox_inches="tight")
        fig.savefig(svg_buf, format="svg", bbox_inches="tight")
        png_buf.seek(0), pdf_buf.seek(0), svg_buf.seek(0)

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            if not open_shape:
                txt_bytes = json.dumps({"Numerical data": num_data}, indent=2).encode()
                zf.writestr(f"{base}.txt", txt_bytes)
            zf.writestr(f"{base}.png", png_buf.getvalue())
            zf.writestr(f"{base}.pdf", pdf_buf.getvalue())
            zf.writestr(f"{base}.svg", svg_buf.getvalue())
        zip_buf.seek(0)

        st.download_button("Download all (ZIP)", zip_buf, f"{base}.zip", "application/zip")


if __name__ == "__main__":
    main()
