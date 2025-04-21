# polygon_drawer.py – v1.2.1  (2025‑04‑21)
# -----------------------------------------------------------
# Streamlit app – clean polygon drawer + single ZIP download
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
LABEL_SHIFT = 0.04      # outward offset as fraction of min side


# ──────────────────────────  Geometry helpers  ────────────────────────────
def vertex_names(n: int) -> List[str]:
    letters = string.ascii_uppercase
    return [(letters[i] if i < 26 else letters[i // 26 - 1] + letters[i % 26])
            for i in range(n)]


def shoelace_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    return math.degrees(
        math.acos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))
    )


def is_polygon_possible(lengths: Sequence[float]) -> bool:
    L = sorted(lengths)
    return L[-1] < sum(L[:-1]) - 1e-9


# ──────────────────────────  Data class  ──────────────────────────────────
@dataclass
class PolygonData:
    pts: np.ndarray
    lengths: List[float]
    angles_int: List[float]

    @property
    def names(self):
        return vertex_names(len(self.pts))


# ──────────────────────────  Construction helpers  ───────────────────────
def repaired_angles(n: int, angs: Sequence[float] | None):
    if angs is None:
        return None
    k = (n - 2) * 180.0 / sum(angs)
    return [a * k for a in angs]


def circumscribed_polygon(lengths: Sequence[float]) -> PolygonData:
    n = len(lengths)
    L = np.asarray(lengths, float)
    R_lo, R_hi = max(L) / 2 + 1e-9, 1e6

    def total(R):
        return np.sum(2 * np.arcsin(np.clip(L / (2 * R), -1 + 1e-12, 1 - 1e-12)))

    for _ in range(60):                     # binary search
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


# ──────────────────────────  Extra geometry  ──────────────────────────────
def diagonals_info(pts: np.ndarray):
    n = len(pts)
    out = []
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

            out.append(
                dict(i=i, j=j, length=length,
                     ang_i=part(i, v), ang_j=part(j, -v))
            )
    return out


def bounding_rect(pts: np.ndarray):
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    rect = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    return rect, xmax - xmin, ymax - ymin


# ──────────────────────────  Drawing  ─────────────────────────────────────
def draw_polygon(poly: PolygonData):
    n = len(poly.pts)
    names = poly.names
    pts_closed = np.vstack([poly.pts, poly.pts[0]])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.plot(pts_closed[:, 0], pts_closed[:, 1], "-o", lw=2)

    # diagonals (plain, no labels)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            p1, p2 = poly.pts[i], poly.pts[j]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    "--", lw=0.8, color="gray", alpha=0.5)

    min_len = min(poly.lengths)

    # vertex & side labels (offset outward)
    for i, (x, y) in enumerate(poly.pts):
        e_prev = poly.pts[i] - poly.pts[i - 1]
        e_next = poly.pts[(i + 1) % n] - poly.pts[i]
        normal = np.array([-(e_prev[1] + e_next[1]),
                           e_prev[0] + e_next[0]])
        if np.linalg.norm(normal):
            normal /= np.linalg.norm(normal)
        ax.text(x + normal[0] * LABEL_SHIFT * min_len,
                y + normal[1] * LABEL_SHIFT * min_len,
                names[i], fontsize=9, weight="bold",
                color="blue", ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.8,
                          boxstyle="circle,pad=0.25"))

        mid = 0.5 * (poly.pts[i] + poly.pts[(i + 1) % n])
        edge = poly.pts[(i + 1) % n] - poly.pts[i]
        en = np.array([-edge[1], edge[0]])
        en /= np.linalg.norm(en)
        ax.text(*(mid + en * LABEL_SHIFT * min_len),
                f"{poly.lengths[i]:.2f}", fontsize=7,
                bbox=dict(facecolor="white", alpha=0.7,
                          edgecolor="none"),
                ha="center", va="center")

    # internal angle arcs (no numeric text)
    for i in range(n):
        p = poly.pts[i]
        v_prev = poly.pts[i - 1] - p
        v_next = poly.pts[(i + 1) % n] - p
        start = math.degrees(math.atan2(v_prev[1], v_prev[0]))
        end = start - (180 - poly.angles_int[i])
        ax.add_patch(
            Arc(p, 0.36 * min_len, 0.36 * min_len, theta1=end,
                theta2=start, lw=1, color="red")
        )

    # bounding rectangle outline
    rect, _, _ = bounding_rect(poly.pts)
    rc = np.vstack([rect, rect[0]])
    ax.plot(rc[:, 0], rc[:, 1], "k-.", lw=1, alpha=0.5)

    return fig


# ──────────────────────────  Streamlit UI  ────────────────────────────────
def main():
    st.set_page_config(page_title="Polygon Drawer", layout="centered")
    st.title("📐 Polygon Drawer – clean export")

    n = st.number_input("Number of sides", 3, 12, 4, 1)
    st.subheader("Side lengths")
    lengths = [st.number_input(f"Length {i + 1}", 0.01, 1000.0, 1.0, 0.1,
                               key=f"L{i}") for i in range(n)]

    # feasibility
    if not is_polygon_possible(lengths):
        st.error("⚠️  Side lengths violate polygon inequality.")
        st.stop()

    use_ang = st.checkbox("Provide internal angles?")
    if use_ang:
        st.subheader("Internal angles (°)")
        angs = [st.number_input(
            f"∠ {vertex_names(n)[i]}", 1.0, 179.0,
            round(180 * (n - 2) / n, 1), 1.0, key=f"A{i}"
        ) for i in range(n)]
        poly = build_polygon(lengths, repaired_angles(n, angs))
    else:
        poly = circumscribed_polygon(lengths)

    if st.button("Draw polygon", use_container_width=True):
        fig = draw_polygon(poly)
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
        st.json(num_data, expanded=False)

        st.markdown("### Diagonals")
        st.json(diag_data, expanded=False)

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