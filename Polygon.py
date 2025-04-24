from __future__ import annotations
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D
from matplotlib.font_manager import FontProperties
from scipy.optimize import minimize
import datetime as dt
import io
import json
import math
import string
import zipfile
from dataclasses import dataclass
from typing import List, Sequence , Sequence, NamedTuple
from matplotlib.path import Path as MplPath  # לוודא שייבאת

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import matplotlib.image as mpimg
from scipy.stats import alpha

TOL = 1e-6
LABEL_SHIFT = 0.05        # outward label offset (fraction of min side)


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
    ax.plot(pts_closed[:, 0], pts_closed[:, 1], "-o", lw=1.4, color= "green" , alpha=1 )

    min_len = min(poly.lengths)

    def segments_intersect(p1, p2, q1, q2) -> bool:
        """Check if segments (p1,p2) and (q1,q2) intersect (excluding endpoints)."""

        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))

    # הכנה לבדיקה פנימית
    polygon_path = MplPath(poly.pts)

    # --- distances from rect corners to polygon points, with smart edge comparison ---
    rect_edges = [rect[1] - rect[0], rect[3] - rect[0]]  # וקטורי צלעות: רוחב וגובה

    for corner in rect:
        dists = [np.linalg.norm(corner - p) for p in poly.pts]
        nearest_indices = np.argsort(dists)
        vecs = [poly.pts[idx] - corner for idx in nearest_indices]

        if all(np.linalg.norm(v) > 1e-8 for v in vecs[:2]):
            u = vecs[0] / np.linalg.norm(vecs[0])
            v = vecs[1] / np.linalg.norm(vecs[1])
            dot = abs(np.dot(u, v))
            if dot > 0.998:
                nearest_indices = [nearest_indices[0]]

        for idx in nearest_indices:
            p = poly.pts[idx]
            vec = p - corner
            dist = np.linalg.norm(vec)
            if dist < 1e-8:
                continue

            # --- בדיקה אם חוצה צלע או נמצא כולו בתוך המצולע ---
            intersects = False
            for i in range(n):
                a = poly.pts[i]
                b = poly.pts[(i + 1) % n]
                if np.allclose(a, corner) or np.allclose(b, corner) or np.allclose(a, p) or np.allclose(b, p):
                    continue
                if segments_intersect(corner, p, a, b):
                    intersects = True
                    break

            mid = 0.5 * (corner + p)
            if intersects or polygon_path.contains_point(mid):
                continue  # הקו לא תקף – חוצה או עובר דרך פנים המצולע

            # --- אם עברנו את כל הבדיקות, נחשב ונצייר ---
            vec_norm = vec / dist
            best_edge_len = None
            best_cos = -1

            for edge_vec in rect_edges:
                edge_len = np.linalg.norm(edge_vec)
                if edge_len < 0.5:
                    continue
                edge_dir = edge_vec / edge_len
                cos_angle = abs(np.dot(vec_norm, edge_dir))
                if cos_angle > best_cos:
                    best_cos = cos_angle
                    best_edge_len = edge_len

            if best_edge_len and dist < best_edge_len:
                ax.plot([p[0], corner[0]], [p[1], corner[1]],
                        color="orange", lw=4, alpha=0.3)

                ax.text(*mid, f"{dist:.2f}", fontsize=6, color="black",
                        ha="center", va="center")

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
                color="blue", ha="center", va="center")

        mid = 0.5 * (poly.pts[i] + poly.pts[(i + 1) % n])
        edge = poly.pts[(i + 1) % n] - poly.pts[i]
        edge_norm = np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)
        ax.text(*(mid + edge_norm * LABEL_SHIFT * min_len),
                f"{poly.lengths[i]:.2f}", fontsize=7,
                bbox=dict(facecolor="green", alpha=0.15,
                          edgecolor="none"),
                ha="center", va="center")

        edge_dir = edge / np.linalg.norm(edge)
        horizontal = np.array([1.0, 0.0])
        vertical = np.array([0.0, 1.0])

        angle_h = math.degrees(math.acos(np.clip(abs(np.dot(edge_dir, horizontal)), -1, 1)))
        angle_v = math.degrees(math.acos(np.clip(abs(np.dot(edge_dir, vertical)), -1, 1)))
        angle_to_rect = min(angle_h, angle_v)

        # מיקום: קרוב לקודקוד תחילת הצלע
        base_point = poly.pts[i]
        # כיוון הצלע בניצב – להזחה כלפי חוץ
        edge_norm = np.array([-edge[1], edge[0]])
        if np.linalg.norm(edge_norm) > 0:
            edge_norm /= np.linalg.norm(edge_norm)
        # סיבוב של הווקטור ° עם כיוון השעון
        theta = math.radians(75)
        rotation_matrix = np.array([
            [math.cos(theta), math.sin(theta)],
            [-math.sin(theta), math.cos(theta)]
        ])
        rotated_vec = rotation_matrix @ edge_norm

        # מיקום הטקסט אחרי סיבוב
        label_pos = base_point + rotated_vec * LABEL_SHIFT * min_len * 3.5

        # ציור הזווית ליד תחילת הצלע
        ax.text(*label_pos,
                f"∠{angle_to_rect:.1f}°",
                fontsize=7,
                color="green",
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


    # --- כתיבת אורכי צלעות המלבן החיצוני בצד החיצוני שלו ---
    edge_labels = [w, h, w, h]  # אורכים לפי סדר הקודקודים
    for i in range(4):
        p1 = rect[i]
        p2 = rect[(i + 1) % 4]
        mid = (p1 + p2) / 2
        edge_vec = p2 - p1
        normal = np.array([-edge_vec[1], edge_vec[0]])
        norm_len = np.linalg.norm(normal)
        if norm_len > 0:
            normal = normal / norm_len
        offset = -0.09 * max(w, h)  # להזיז החוצה ב-0.05 מאורך הצלע הגדולה יותר
        label_pos = mid + normal * offset

        ax.text(*label_pos,
                f"{edge_labels[i]:.2f}",
                fontsize=7,
                color="purple",
                ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.6))

    i=0
    p1 = rect[i]
    p2 = rect[(i + 1) % 4]
    mid = (p1 + p2) / 2
    edge_vec = p2 - p1
    normal = np.array([-edge_vec[1], edge_vec[0]])
    norm_len = np.linalg.norm(normal)
    if norm_len > 0:
        normal = normal / norm_len
    offset = -0.09 * max(w, h)  # להזיז החוצה ב-0.05 מאורך הצלע הגדולה יותר
    label_pos = mid + normal * offset

    
    ax.text(*([2,1.06] * label_pos  ), f"w={w:.2f}", fontsize=8,
            ha="left", va="center" , color="purple")
    ax.text(*([2,1.09] * label_pos  ), f"h={h:.2f}", fontsize=8,
            ha="left", va="center" , color="purple")
    ax.text(*([2,1.03] * label_pos ), f"Area REC={HW:.2f}", fontsize=8,
            ha="left", va="center" , color="purple")

    # ----- area label ------------------------------------------------------
    ax.text(*([2,1] * label_pos  ),
            f"Area Poligon = {shoelace_area(poly.pts):.2f}",
            fontsize=9, color="green",
            ha="left", va="center")



    i=0
    p1 = rect[i]
    p2 = rect[(i + 1) % 4]
    mid = (p1 + p2) / 2
    edge_vec = p2 - p1
    normal = np.array([-edge_vec[1], edge_vec[0]])
    norm_len = np.linalg.norm(normal)
    if norm_len > 0:
        normal = normal / norm_len
    offset = -0.09 * max(w, h)  # להזיז החוצה ב-0.05 מאורך הצלע הגדולה יותר
    label_pos = mid + normal * offset



    
    # --- לוגו טיפוגרפי של שם ---
    logo_text = "Created by Yarden Viktor Dejorno"
    font = FontProperties(fname="Pacifico-Regular.ttf")  # או שם אחר שהורדת
    tp = TextPath((0, 0), logo_text, size=0.03, prop=font)

    ax.text(*([0.4,1] * label_pos  ),"The App Created by:\nYarden Viktor Dejorno",fontsize=9, font=font)

    # אנכים מכל קודקוד של המצולע אל 2 הצלעות הקרובות של המלבן

    rect_sides = [(rect[i], rect[(i + 1) % 4]) for i in range(4)]

    for pt in poly.pts:
        distances = []

        for p1, p2 in rect_sides:
            edge_vec = p2 - p1
            edge_len_sq = np.dot(edge_vec, edge_vec)
            if edge_len_sq < 1e-8:
                continue

            t = np.dot(pt - p1, edge_vec) / edge_len_sq
            t_clamped = np.clip(t, 0, 1)
            foot = p1 + t_clamped * edge_vec
            dist = np.linalg.norm(pt - foot)

            distances.append((dist, foot, (p1, p2)))

        # בחר את שתי הצלעות הכי קרובות
        distances.sort(key=lambda x: x[0])
        for i in range(2):
            dist, foot, (p1, p2) = distances[i]
            ax.plot([pt[0], foot[0]], [pt[1], foot[1]],
                    linestyle="dashed", linewidth=0.8, color="green", alpha=0.7)

            # מיקום הטקסט – אמצע הקו + הזחה קטנה לצד
            mid = 0.5 * (pt + foot)
            offset_vec = foot - pt
            offset_vec = np.array([-offset_vec[1], offset_vec[0]])  # ניצב
            if np.linalg.norm(offset_vec) > 0:
                offset_vec /= np.linalg.norm(offset_vec)
            label_pos = mid + offset_vec * LABEL_SHIFT * 0.1 * min_len

            if dist > 0.01:

                ax.text(*label_pos, f"{dist:.2f}", fontsize=3,
                        color="green", ha="center", va="center")


    return fig, diags, altitudes_data



# ────── Streamlit UI  ───────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Polygon Drawer", layout="centered")
    st.title("📐 Polygon Drawer")
    st.title(" לינקו - תמורת טובות הנעה.")

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
