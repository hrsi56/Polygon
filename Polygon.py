# polygon_drawer.py
from __future__ import annotations

import datetime as dt
import io
import json
import math
import string
import zipfile
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


# ───────────────────────── קבועים ─────────────────────────
PREC = 1                       # מספר ספרות אחרי הנקודה
TOL  = 10 ** -(PREC + 1)       # רגישות סגירה (≈ 0.01)
LABEL_SHIFT = -0.05            # היסט‑תיוג יחסי


# ──────────────────────── עזרי‑גאומטריה ────────────────────
def rnd(x: float | np.ndarray) -> float | np.ndarray:
    """עיגול ל‑ספרה אחת אחרי הנקודה (או מערך שלם)."""
    return np.round(x, PREC)


def vertex_names(n: int) -> List[str]:
    letters = string.ascii_uppercase
    return [(letters[i] if i < 26 else letters[i // 26 - 1] + letters[i % 26])
            for i in range(n)]


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    return rnd(math.degrees(
        math.acos(np.clip(np.dot(u, v) /
                          (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))
    ))


def shoelace_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return rnd(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def centroid(pts: np.ndarray) -> np.ndarray:
    x, y = pts[:, 0], pts[:, 1]
    a = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    A = 0.5 * a
    cx = np.sum((x + np.roll(x, -1)) *
                (x * np.roll(y, -1) - np.roll(x, -1) * y)) / (6 * A)
    cy = np.sum((y + np.roll(y, -1)) *
                (x * np.roll(y, -1) - np.roll(x, -1) * y)) / (6 * A)
    return np.array([rnd(cx), rnd(cy)])


def bounding_rect(pts: np.ndarray) -> Tuple[np.ndarray, float, float]:
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    rect = np.array([[xmin, ymin], [xmax, ymin],
                     [xmax, ymax], [xmin, ymax]])
    return rect, rnd(xmax - xmin), rnd(ymax - ymin)


# ─────────────────────── מבנה‑נתונים מצולע ──────────────────────
@dataclass
class PolygonData:
    pts: np.ndarray
    lengths: List[float]
    angles_int: List[float]
    closed: bool = True           # True אם נסגר לגמרי

    @property
    def names(self) -> List[str]:
        return vertex_names(len(self.pts))


# ─────────────────────── פונקציות בנייה שונות ────────────────────
def repaired_angles(n: int, angs: Sequence[float] | None):
    """סקיילר אחיד כדי שסכום הזוויות יעמוד ב‑(n‑2)*180°."""
    if angs is None:
        return None
    k = (n - 2) * 180.0 / sum(angs)
    return [rnd(a * k) for a in angs]


# ---------- עוזר משותף ---------------------------------------------------
def _heads_from_angles(angles_deg: np.ndarray) -> np.ndarray:
    ext = np.radians(180.0 - angles_deg)          # זוויות חיצוניות
    heads = np.zeros_like(angles_deg)
    heads[1:] = np.cumsum(ext[:-1])
    return heads


def _vecs_from_L_heads(L: np.ndarray, heads: np.ndarray) -> np.ndarray:
    return np.stack([L * np.cos(heads), L * np.sin(heads)], axis=1)


def _poly_from_vecs(vecs: np.ndarray, lengths: np.ndarray, angles: np.ndarray,
                    closed: bool) -> PolygonData:
    pts = np.concatenate([[np.zeros(2)], np.cumsum(vecs, axis=0)])[:-1]
    return PolygonData(rnd(pts), rnd(lengths.tolist()), rnd(angles.tolist()), closed)


# ---------- 1. פיזור על כל הווקטורים ------------------------------------
def build_fix_both(lengths: Sequence[float],
                   angles: Sequence[float]) -> PolygonData:
    L = rnd(np.asarray(lengths, float))
    A = rnd(np.asarray(angles, float))
    heads = _heads_from_angles(A)
    vecs = _vecs_from_L_heads(L, heads)

    gap = vecs.sum(axis=0)
    if np.hypot(*gap) > TOL:
        vecs += (-gap * (L / L.sum())[:, None])     # פיזור יחסי
        gap = vecs.sum(axis=0)
    if np.hypot(*gap) > TOL:
        vecs[-1] -= gap                             # תיקון אחרון
    closed = np.hypot(*vecs.sum(axis=0)) <= TOL

    new_L = np.linalg.norm(vecs, axis=1)
    new_A = np.array([angle_between(vecs[i - 1], -vecs[i])
                      for i in range(len(L))])
    return _poly_from_vecs(vecs, new_L, new_A, closed)


# ---------- 2. מצב קשיח – לא מתקנים --------------------------------------
def build_strict_open(lengths: Sequence[float],
                      angles: Sequence[float]) -> PolygonData:
    L = rnd(np.asarray(lengths, float))
    A = rnd(np.asarray(angles, float))
    heads = _heads_from_angles(A)
    vecs = _vecs_from_L_heads(L, heads)
    closed = np.hypot(*vecs.sum(axis=0)) <= TOL
    return _poly_from_vecs(vecs, L, A, closed)      # closed= False ברוב המקרים


# ---------- 3. שמור אורכים, סגור בזוויות ----------------------------------
def build_fix_angles(lengths: Sequence[float],
                     angles: Sequence[float]) -> Optional[PolygonData]:
    L = rnd(np.asarray(lengths, float))
    A0 = rnd(np.asarray(angles, float))
    heads = _heads_from_angles(A0)
    vecs = _vecs_from_L_heads(L, heads)
    gap = vecs.sum(axis=0)
    if np.hypot(*gap) <= TOL:
        return _poly_from_vecs(vecs, L, A0, True)

    # מטריצה A: כל עמודה = R90(v_i)
    R90 = np.array([[0, -1], [1, 0]])
    A_mat = np.zeros((2, len(L)))
    for i, v in enumerate(vecs):
        A_mat[:, i] = R90 @ v
    try:
        delta = A_mat.T @ np.linalg.inv(A_mat @ A_mat.T) @ (-gap)
    except np.linalg.LinAlgError:
        return None

    ext = np.radians(180.0 - A0) + delta
    A_new = rnd(180.0 - np.degrees(ext))
    heads = _heads_from_angles(A_new)
    vecs = _vecs_from_L_heads(L, heads)
    if np.hypot(*vecs.sum(axis=0)) > TOL:
        return None
    return _poly_from_vecs(vecs, L, A_new, True)


# ---------- 4. שמור זוויות, סגור באורכים ----------------------------------
def build_fix_lengths(lengths: Sequence[float],
                      angles: Sequence[float]) -> Optional[PolygonData]:
    L0 = rnd(np.asarray(lengths, float))
    A = rnd(np.asarray(angles, float))
    heads = _heads_from_angles(A)
    dirs = np.stack([np.cos(heads), np.sin(heads)], axis=1)   # א־כיוונים
    vecs = (L0[:, None] * dirs)
    gap = vecs.sum(axis=0)
    if np.hypot(*gap) <= TOL:
        return _poly_from_vecs(vecs, L0, A, True)

    A_mat = dirs.T                                           # 2×n
    try:
        dL = A_mat.T @ np.linalg.inv(A_mat @ A_mat.T) @ (-gap)
    except np.linalg.LinAlgError:
        return None

    L_new = L0 + dL
    if (L_new <= 0).any():
        return None
    vecs = (L_new[:, None] * dirs)
    if np.hypot(*vecs.sum(axis=0)) > TOL:
        return None
    return _poly_from_vecs(vecs, L_new, A, True)


# ─────────────────────────── ציור מצולע ───────────────────────────
def draw_polygon(poly: PolygonData, show_altitudes: bool):
    n = len(poly.pts)
    names = poly.names
    pts_draw = (np.vstack([poly.pts, poly.pts[0]])
                if poly.closed else poly.pts)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.plot(pts_draw[:, 0], pts_draw[:, 1], "-o",
            lw=1.4, color="black", alpha=0.6)

    min_len = min(poly.lengths)

    # תיוג קודקודים וצלעות
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
                f"{poly.lengths[i]:.{PREC}f}", fontsize=7,
                bbox=dict(facecolor="white", alpha=0.7,
                          edgecolor="none"),
                ha="center", va="center")

    # זוויות פנימיות
    for i in range(n):
        p = poly.pts[i]
        v_prev = poly.pts[i - 1] - p
        v_next = poly.pts[(i + 1) % n] - p
        bis = v_prev / np.linalg.norm(v_prev) + v_next / np.linalg.norm(v_next)
        if np.linalg.norm(bis) == 0:
            bis = np.array([v_next[1], -v_next[0]])
        bis /= np.linalg.norm(bis)
        txt = p + bis * (0.23 * min_len)
        ax.text(*txt, f"{poly.angles_int[i]:.{PREC}f}°",
                fontsize=7, color="red", ha="center", va="center")

    return fig


# ─────────────────────────── ממשק Streamlit ───────────────────────────
def main():
    st.set_page_config(page_title="Polygon Drawer", layout="centered")
    st.title("📐 Polygon Drawer – לינקו")

    n = st.number_input("Number of sides", 3, 12, 4, 1)
    lengths = [rnd(st.number_input(f"Length {i + 1}", 0.1, 1000.0,
                                   1.0, 0.1, key=f"L{i}"))
               for i in range(n)]

    if st.checkbox("Provide internal angles?"):
        angs = [rnd(st.number_input(
            f"∠ {vertex_names(n)[i]}", 1.0, 359.9,
            rnd(180 * (n - 2) / n), 0.1, key=f"A{i}"
        )) for i in range(n)]
    else:
        st.error("⚠️  Without angles, לא ניתן להשתמש במצבי תיקון.")
        st.stop()

    # תיבת ברירה לשיטה
    option = st.selectbox(
        "בחר שיטת תיקון אם המצולע לא נסגר:",
        ("פיזור על כל הווקטורים (שינוי מינימלי)",
         "לא לתקן – צורה פתוחה",
         "שמור אורכים, סגור בזוויות",
         "שמור זוויות, סגור באורכים"),
        index=0
    )

    build_map = {0: build_fix_both,
                 1: build_strict_open,
                 2: build_fix_angles,
                 3: build_fix_lengths}
    build_idx = ("פיזור" in option) * 0 + ("לא לתקן" in option) * 1 + \
                ("שמור אורכים" in option) * 2 + ("שמור זוויות" in option) * 3

    if st.button("Draw polygon", use_container_width=True):
        # ניסיון בנייה
        builder = build_map[build_idx]
        poly = builder(lengths, repaired_angles(n, angs))

        if poly is None or (not poly.closed and build_idx != 1):
            st.error("⚠️  בחר שיטת תיקון אחרת – השיטה הזו לא הצליחה לסגור.")
            st.stop()

        show_alt = poly.closed and n == 3
        fig = draw_polygon(poly, show_alt)
        st.pyplot(fig, use_container_width=True)

        if poly.closed:
            area_val = shoelace_area(poly.pts)
            _, w, h = bounding_rect(poly.pts)
            st.markdown("### Numerical data")
            st.json({
                "Area": area_val,
                "Bounding width": w,
                "Bounding height": h
            }, expanded=True)
        else:
            st.warning("המצולע פתוח – לא חישבתי שטח / מלבן חוסם.")

        # קובצי הורדה (גם עבור מצולע פתוח)
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M")
        base = f"Polygon_{ts}"

        png_buf, pdf_buf, svg_buf = io.BytesIO(), io.BytesIO(), io.BytesIO()
        fig.savefig(png_buf, format="png", dpi=300, bbox_inches="tight")
        fig.savefig(pdf_buf, format="pdf", bbox_inches="tight")
        fig.savefig(svg_buf, format="svg", bbox_inches="tight")
        png_buf.seek(0); pdf_buf.seek(0); svg_buf.seek(0)

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
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