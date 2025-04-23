from __future__ import annotations

from scipy.optimize import minimize
import datetime as dt
import io
import json
import math
import string
import zipfile
from dataclasses import dataclass
from typing import List, Sequence , Sequence, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import matplotlib.image as mpimg

TOL = 1e-6
LABEL_SHIFT = -0.05        # outward label offset (fraction of min side)


# ────── geometry helpers ──────────────────────────────────────────────────
def vertex_names(n: int) -> List[str]:
    letters = string.ascii_uppercase
    return [(letters[i] if i < 26 else letters[i // 26 - 1] + letters[i % 26])
            for i in range(n)]


def build_polygon_with_extra(lengths: Sequence[float],
                             angles: Sequence[float]) -> PolygonData:
    """
    כמו build_polygon, אבל אם נשאר gap ‑ מוסיפים צלע חדשה במקום
    לחלק את התיקון בין הצלעות הקיימות.
    """
    n = len(lengths)
    L = np.asarray(lengths, float)

    # כיווני הצלעות עפ"י הזוויות החיצוניות
    ext = np.radians(180.0 - np.asarray(angles))
    heads = np.zeros(n)
    heads[1:] = np.cumsum(ext[:-1])
    vecs = np.stack([L * np.cos(heads), L * np.sin(heads)], axis=1)

    # ‑‑‑‑‑‑‑‑‑‑ gap -----------------------------------------------------
    gap = vecs.sum(axis=0)
    if np.hypot(*gap) > TOL:            # אם אכן נשאר פער
        vecs = np.vstack([vecs, -gap])  # מוסיפים צלע סוגרת

    # וֶרְטִיקְסִים וליסטים ממוזערים
    pts = np.concatenate([[np.zeros(2)],
                          np.cumsum(vecs, axis=0)])[:-1]
    lengths_corr = np.linalg.norm(vecs, axis=1).tolist()
    angles_corr = [angle_between(pts[i - 1] - pts[i],
                                 pts[(i + 1) % len(pts)] - pts[i])
                   for i in range(len(pts))]
    return PolygonData(pts, lengths_corr, angles_corr)

def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    return math.degrees(
        math.acos(np.clip(np.dot(u, v) /
                          (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))
    )


def shoelace_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def centroid(pts: np.ndarray) -> np.ndarray:
    x, y = pts[:, 0], pts[:, 1]
    a = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    A = 0.5 * a
    cx = np.sum((x + np.roll(x, -1)) *
                (x * np.roll(y, -1) - np.roll(x, -1) * y)) / (6 * A)
    cy = np.sum((y + np.roll(y, -1)) *
                (x * np.roll(y, -1) - np.roll(x, -1) * y)) / (6 * A)
    return np.array([cx, cy])


def is_polygon_possible(lengths: Sequence[float]) -> bool:
    L = sorted(lengths)
    return L[-1] < sum(L[:-1]) - 1e-9


# ────── data structure ────────────────────────────────────────────────────
@dataclass
class PolygonData:
    pts: np.ndarray
    lengths: List[float]
    angles_int: List[float]

    @property
    def names(self) -> List[str]:
        return vertex_names(len(self.pts))


# ────── construction functions ───────────────────────────────────────────
def repaired_angles(n: int, angs: Sequence[float] | None):
    if angs is None:
        return None
    k = (n - 2) * 180.0 / sum(angs)
    return [a * k for a in angs]


def circumscribed_polygon(lengths: Sequence[float]) -> PolygonData:
    n = len(lengths)
    lengths = np.asarray(lengths, float)

    def make_polygon(angles_rad):
        angle = 0
        pts = [np.array([0.0, 0.0])]
        for i in range(n):
            angle += angles_rad[i]
            dx = lengths[i] * np.cos(angle)
            dy = lengths[i] * np.sin(angle)
            pts.append(pts[-1] + np.array([dx, dy]))
        return np.array(pts)

    def objective(angles_rad):
        pts = make_polygon(angles_rad)
        return np.linalg.norm(pts[-1] - pts[0])**2

    initial_angles = np.full(n, -2 * np.pi / n)
    res = minimize(objective, initial_angles, method='BFGS')

    pts = make_polygon(res.x)
    if len(pts) != n + 1:
        raise ValueError("Polygon construction failed (wrong number of points)")
    pts = pts[:-1]  # Remove closing point

    angles_int = []
    for i in range(n):
        a = pts[i - 1] - pts[i]
        b = pts[(i + 1) % n] - pts[i]
        ang = np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))
        angles_int.append(np.degrees(ang))

    return PolygonData(pts, list(lengths), angles_int)


def build_polygon(lengths: Sequence[float],
                  angles: Sequence[float]) -> PolygonData:
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

    pts = np.concatenate([[np.zeros(2)],
                          np.cumsum(vecs, axis=0)])[:-1]
    lengths_corr = np.linalg.norm(vecs, axis=1).tolist()
    angles_corr = [
        angle_between(pts[i - 1] - pts[i],
                      pts[(i + 1) % n] - pts[i]) for i in range(n)
    ]
    return PolygonData(pts, lengths_corr, angles_corr)


# ────── diagonals with single‑reference angle ─────────────────────────────
def diagonals_info(poly: PolygonData):
    pts = poly.pts
    names = poly.names
    n = len(pts)
    info = []
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            v = pts[j] - pts[i]
            length = float(np.linalg.norm(v))

            def pick_angle(idx: int, vec: np.ndarray):
                # vectors of adjacent sides from vertex idx
                prev_vec = pts[idx - 1] - pts[idx]
                next_vec = pts[(idx + 1) % n] - pts[idx]
                ang_prev = angle_between(vec, prev_vec)
                ang_next = angle_between(vec, next_vec)
                if ang_prev <= ang_next:
                    return ang_prev, f"{names[idx - 1]}{names[idx]}"
                else:
                    return ang_next, f"{names[idx]}{names[(idx + 1) % n]}"

            ang_i, side_i = pick_angle(i, v)
            ang_j, side_j = pick_angle(j, -v)

            info.append(dict(
                i=i, j=j,
                length=length,
                end_i=dict(side=side_i, angle=ang_i),
                end_j=dict(side=side_j, angle=ang_j)
            ))
    return info


# ────── bounding rectangle ────────────────────────────────────────────────
def bounding_rect(pts: np.ndarray):
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    rect = np.array([[xmin, ymin], [xmax, ymin],
                     [xmax, ymax], [xmin, ymax]])
    return rect, xmax - xmin, ymax - ymin


# ────────── triangle altitudes ────────────────────────────────────────────
def triangle_altitudes(pts: np.ndarray):
    """Return list of dicts with keys: 'from', 'to', 'foot', 'length'."""
    alt = []
    names = vertex_names(3)
    for idx in range(3):
        A = pts[idx]
        B = pts[(idx + 1) % 3]
        C = pts[(idx + 2) % 3]
        BC_vec = C - B
        t = np.dot(A - B, BC_vec) / np.dot(BC_vec, BC_vec)
        foot = B + t * BC_vec
        length = float(np.linalg.norm(A - foot))
        alt.append(dict(
            from_v=idx,
            to_side=f"{names[(idx + 1) % 3]}{names[(idx + 2) % 3]}",
            foot=foot,
            length=length
        ))
    return alt

# ────── drawing routine ───────────────────────────────────────────────────
def draw_polygon(poly: PolygonData, show_altitudes: bool):
    rect, w, h = bounding_rect(poly.pts)
    img = mpimg.imread('Subject.PNG')  # or use plt.imread()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, extent=[rect[0][0] , rect[1][0] ,  rect[0][1], rect[2][1]], alpha=0.7)

    n = len(poly.pts)
    names = poly.names
    pts_closed = np.vstack([poly.pts, poly.pts[0]])

    ax.set_aspect("equal")
    ax.axis("off")
    ax.plot(pts_closed[:, 0], pts_closed[:, 1], "-o", lw=1.4, color= "green" , alpha=0.6 )

    min_len = min(poly.lengths)

    # ----- diagonals -------------------------------------------------------
    diags = diagonals_info(poly)
    for d in diags:
        p1, p2 = poly.pts[d["i"]], poly.pts[d["j"]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                "--", lw=0.8, color="gray", alpha=0.6)
        mid = 0.5 * (p1 + p2)
        ax.text(*mid, f"{d['length']:.2f}",
                fontsize=6, color="gray",
                ha="center", va="center")

        # endpoint i
        vec_i = (p2 - p1) / np.linalg.norm(p2 - p1)
        label_vec_i = vec_i
        ax.text(*(p1 + label_vec_i * 0.1 * min_len),
                f"{d['end_i']['angle']:.1f}°\n{d['end_i']['side']}",
                fontsize=5, color="brown",
                ha="center", va="center")

        # endpoint j
        vec_j = (p1 - p2) / np.linalg.norm(p1 - p2)
        ax.text(*(p2 + vec_j * 0.1 * min_len),
                f"{d['end_j']['angle']:.1f}°\n{d['end_j']['side']}",
                fontsize=5, color="brown",
                ha="center", va="center")

    # ----- vertices & sides ------------------------------------------------
    for i, (x, y) in enumerate(poly.pts):
        prev_vec = poly.pts[i] - poly.pts[i - 1]
        next_vec = poly.pts[(i + 1) % n] - poly.pts[i]
        normal = np.array([-(prev_vec[1] + next_vec[1]),
                           prev_vec[0] + next_vec[0]])
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
        edge_norm = np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)
        ax.text(*(mid + edge_norm * LABEL_SHIFT * min_len),
                f"{poly.lengths[i]:.2f}", fontsize=7,
                bbox=dict(facecolor="green", alpha=0.15,
                          edgecolor="none"),
                ha="center", va="center")

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
        ax.text(
            *txt,
            f"{poly.angles_int[i]:.1f}°",
            fontsize=7,
            color="red",
            ha="center",
            va="center",
        )




    # -------- altitudes for triangle --------------------------------------
    altitudes_data = None
    if show_altitudes and n == 3:
        altitudes_data = triangle_altitudes(poly.pts)
        for alt in altitudes_data:
            A = poly.pts[alt["from_v"]]
            foot = alt["foot"]
            ax.plot([A[0], foot[0]], [A[1], foot[1]],
                    ":", color="magenta", lw=1.2)
            ax.text(*foot, f"h={alt['length']:.2f}", fontsize=6,
                    color="magenta", ha="left", va="bottom")

    # ----- bounding rectangle ---------------------------------------------
    rc = np.vstack([rect, rect[0]])
    ax.plot(rc[:, 0], rc[:, 1], "-.", lw=1, alpha=0.5, color="purple")
    HW = h*w

    ax.text(*(rect[0]-[0,0.09] * (rect[3] - rect[0] )), f"w={w:.2f}", fontsize=8,
            ha="left", va="center" , color="purple")
    ax.text(*(rect[0]-[0,0.12] * (rect[3] - rect[0] )), f"h={h:.2f}", fontsize=8,
            ha="left", va="center" , color="purple")
    ax.text(*(rect[0]-[0,0.06] * (rect[3] - rect[0] )), f"Area REC={HW:.2f}", fontsize=8,
            ha="left", va="center" , color="purple")



    # ----- area label ------------------------------------------------------
    ax.text(*(rect[0]-[0,0.03] * (rect[3] - rect[0] ) ),
            f"Area Poligon = {shoelace_area(poly.pts):.2f}",
            fontsize=9, color="green",
            ha="left", va="center",
            bbox=dict(facecolor="white", alpha=0.7,
                      edgecolor="none"))


    ax.text(*(rect[1] - [0,0.06] * (rect[3] - rect[0]) -  [0.4,0] * (rect[1]- rect[0])),"Created by:\nYarden Viktor Dejorno",fontsize=9,
                ha="left", va="center")


    return fig, diags, altitudes_data



# ────── Streamlit UI  ───────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Polygon Drawer", layout="centered")
    st.title("📐 Polygon Drawer – לינקו. תמורת טובות הנעה")

    n = st.number_input("Number of sides", 3, 12, 4, 1)
    lengths = [st.number_input(f"Length {i + 1}", 0.1, 10000.0,
                               1.0, 0.5, key=f"L{i}") for i in range(n)]

    if not is_polygon_possible(lengths):
        st.error("⚠️  Side lengths violate polygon inequality.")
        st.stop()

    if st.checkbox("Provide internal angles?"):
        angs = [
            st.number_input(
                f"∠ {vertex_names(n)[i % n]} , {i}∠{(i % n) + 1 if (i % n) + 1 <= n else 1}",
                min_value=1.0,
                max_value=360.0,
                value=round(180 * (n - 2) / n, 1),
                step=1.0,
                key=f"A{i}"
            )
            for i in range(1, n + 1)
        ]
        add_extra = st.checkbox(f"אל תתקן את הנתונים. תסגור את הצורה כמו שנתתי לך. \n (תתווסף צלע. המצולע יהיה בעל {n+1} צלעות) ")
        if add_extra:
            poly = build_polygon_with_extra(lengths,
                                            angs)
        else:
            poly = build_polygon(lengths,
                                 repaired_angles(n, angs))
    else:
        poly = circumscribed_polygon(lengths)


    show_alt = False
    if n == 3:
        show_alt = True


    if st.button("Draw polygon", use_container_width=True):
        fig, diag_list, altitudes = draw_polygon(poly, show_alt)
        st.pyplot(fig, use_container_width=True)

        area_val = shoelace_area(poly.pts)
        _, w, h = bounding_rect(poly.pts)

        num_data = {
            "Area": round(area_val, 4),
            "Bounding width": round(w, 4),
            "Bounding height": round(h, 4)
        }
        if altitudes:
            num_data["Altitudes"] = [round(a["length"], 4)
                                     for a in altitudes]
            st.markdown("### Numerical data")
            st.json(num_data, expanded=True)

            txt_bytes = json.dumps(
                {"Numerical data": num_data},
                indent=2
            ).encode()

        else:
            diag_data = {
            f"{poly.names[d['i']]}{poly.names[d['j']]}": {
                "Length": round(d["length"], 3),
                poly.names[d["i"]]: d["end_i"],
                poly.names[d["j"]]: d["end_j"],
            } for d in diag_list
        }

            st.markdown("### Numerical data")
            st.json(num_data, expanded=True)

            st.markdown("### Diagonals")
            st.json(diag_data, expanded=True)

            txt_bytes = json.dumps(
                {"Numerical data": num_data, "Diagonals": diag_data},
                indent=2
            ).encode()

        # -------- create ZIP download --------------------------------------
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M")
        base = f"YVD_Poligon_{ts}"



        png_buf, pdf_buf, svg_buf = io.BytesIO(), io.BytesIO(), io.BytesIO()
        fig.savefig(png_buf, format="png", dpi=300, bbox_inches="tight")
        fig.savefig(pdf_buf, format="pdf", bbox_inches="tight")
        fig.savefig(svg_buf, format="svg", bbox_inches="tight")
        png_buf.seek(0); pdf_buf.seek(0); svg_buf.seek(0)

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{base}.txt", txt_bytes)
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