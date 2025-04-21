from __future__ import annotations

"""
Polygon Drawer – four flexible repair strategies
================================================
Copy‑paste this file over your existing one.  The main differences are:

* **Strict build** verifies closure first.  If the polygon doesn’t close a
  Streamlit *radio* asks which of four repair schemes to apply.
* Four repair functions:
  1. **distribute_gap** – minimal‑norm correction to *all* vectors (changes   
     only the lengths ‑ same behaviour as your original code).
  2. **add_extra_side** – keeps the data rigid and adds one more side that
     closes the shape.
  3. **fix_angles** – keeps the side‑lengths, iteratively tweeks the angles   
     with a numerical Jacobian until the gap is gone.
  4. **fix_last_length** – keeps all angles, changes *only* the last side’s
     length.

If the chosen scheme still can’t close the polygon the user gets the Hebrew
error message: "בחר שיטת תיקון אחרת".

The public API of the original file (draw_polygon, triang altitudes, Streamlit UI)
remains intact.
"""

import datetime as dt
import io
import json
import math
import string
import zipfile
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

TOL = 1e-2
LABEL_SHIFT = -0.05  # outward label offset (fraction of min side)

# ────── geometry helpers ──────────────────────────────────────────────────

def vertex_names(n: int) -> List[str]:
    letters = string.ascii_uppercase
    return [letters[i] if i < 26 else letters[i // 26 - 1] + letters[i % 26] for i in range(n)]


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
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

# ────── generic helpers for builds ────────────────────────────────────────

def _vectors_from_lengths_angles(L: np.ndarray, angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the edge vectors and the closure gap for given data."""
    ext = np.radians(180.0 - angles)
    heads = np.zeros(len(L))
    heads[1:] = np.cumsum(ext[:-1])
    vecs = np.stack([L * np.cos(heads), L * np.sin(heads)], axis=1)
    gap = vecs.sum(axis=0)
    return vecs, gap


def _polygon_from_vecs(vecs: np.ndarray) -> np.ndarray:
    return np.vstack([np.zeros(2), np.cumsum(vecs, axis=0)])[:-1]

# ────── 1. distribute gap across all vectors  (length correction) ─────────

def build_polygon_distribute(lengths: Sequence[float], angles: Sequence[float]) -> PolygonData:
    L = np.asarray(lengths, float)
    ang = np.asarray(angles, float)

    vecs, gap = _vectors_from_lengths_angles(L, ang)
    if np.linalg.norm(gap) < TOL:
        return PolygonData(_polygon_from_vecs(vecs), list(L), list(ang))

    # Least‑squares distribute deltaL along directions to cancel gap
    dirs = vecs / L[:, None]
    A = dirs.T  # shape 2×n
    delta_L, *_ = np.linalg.lstsq(A.T, -gap, rcond=None)
    L_new = L + delta_L
    if (L_new <= 0).any():  # negative length – cannot use this strategy
        raise ValueError("negative length after distribute fix")

    vecs = dirs * L_new[:, None]
    pts = _polygon_from_vecs(vecs)
    return PolygonData(pts, L_new.tolist(), list(ang))

# ────── 2. rigid data – add an extra side to close ────────────────────────

def build_polygon_add_side(lengths: Sequence[float], angles: Sequence[float]) -> PolygonData:
    L = np.asarray(lengths, float)
    ang = np.asarray(angles, float)

    vecs, gap = _vectors_from_lengths_angles(L, ang)
    if np.linalg.norm(gap) < TOL:
        return PolygonData(_polygon_from_vecs(vecs), list(L), list(ang))

    # extra side
    extra_len = float(np.linalg.norm(-gap))
    extra_vec = -gap
    vecs = np.vstack([vecs, extra_vec])
    pts = _polygon_from_vecs(vecs)

    # internal angle at new vertex = angle between prev & closing edge
    v_prev = -vecs[-2]
    v_next = vecs[-1]
    extra_angle = angle_between(v_prev, v_next)

    return PolygonData(pts, list(L) + [extra_len], list(ang) + [extra_angle])

# ────── 3. keep lengths, adjust angles (iterative Newton) ────────────────

def build_polygon_fix_angles(lengths: Sequence[float], angles: Sequence[float], *, max_iter: int = 40) -> PolygonData:
    L = np.asarray(lengths, float)
    ang = np.asarray(angles, float)
    n = len(L)

    eps = 1e-5
    for _ in range(max_iter):
        vecs, gap = _vectors_from_lengths_angles(L, ang)
        if np.linalg.norm(gap) < TOL:
            return PolygonData(_polygon_from_vecs(vecs), list(L), ang.tolist())
        # numerical Jacobian 2×n
        J = np.empty((2, n))
        for i in range(n):
            ang_pert = ang.copy(); ang_pert[i] += eps
            _, g2 = _vectors_from_lengths_angles(L, ang_pert)
            J[:, i] = (g2 - gap) / eps
        # Solve J * delta = -gap  (least squares, keep sum(delta)=0)
        # Augment with sum‑to‑zero constraint via Lagrange multiplier
        J_aug = np.vstack([J, np.ones(n)])  # 3×n
        rhs = np.concatenate([-gap, [0.]])
        delta, *_ = np.linalg.lstsq(J_aug.T, rhs, rcond=None)
        ang += delta
    raise ValueError("angle‑fix strategy did not converge")

# ────── 4. keep angles, adjust *only last* length ─────────────────────────

def build_polygon_fix_last_length(lengths: Sequence[float], angles: Sequence[float]) -> PolygonData:
    L = np.asarray(lengths, float)
    ang = np.asarray(angles, float)
    n = len(L)

    vecs, gap = _vectors_from_lengths_angles(L, ang)
    if np.linalg.norm(gap) < TOL:
        return PolygonData(_polygon_from_vecs(vecs), list(L), list(ang))

    # direction of last edge
    ext = np.radians(180.0 - ang)
    heads = np.zeros(n)
    heads[1:] = np.cumsum(ext[:-1])
    dir_last = np.array([math.cos(heads[-1]), math.sin(heads[-1])])

    proj = -gap @ dir_last  # projection of gap onto that direction
    new_L_last = L[-1] + proj
    if new_L_last <= 0:
        raise ValueError("last‑length fix would produce non‑positive side")

    L_new = L.copy(); L_new[-1] = new_L_last
    vecs = vecs.copy(); vecs[-1] = dir_last * new_L_last
    pts = _polygon_from_vecs(vecs)
    return PolygonData(pts, L_new.tolist(), list(ang))

# ────── wrapper strict checker ────────────────────────────────────────────

def build_polygon_strict(lengths: Sequence[float], angles: Sequence[float]) -> PolygonData:
    L = np.asarray(lengths, float)
    ang = np.asarray(angles, float)
    vecs, gap = _vectors_from_lengths_angles(L, ang)
    if np.linalg.norm(gap) > TOL:
        raise ValueError("non‑closing data")
    return PolygonData(_polygon_from_vecs(vecs), list(L), list(ang))

# ────── fallback circumscribed polygon (unchanged) ────────────────────────

def repaired_angles(n: int, angs: Sequence[float] | None):
    if angs is None:
        return None
    k = (n - 2) * 180.0 / sum(angs)
    return [a * k for a in angs]


def circumscribed_polygon(lengths: Sequence[float]) -> PolygonData:
    n = len(lengths)
    L = np.asarray(lengths, float)
    R_lo, R_hi = max(L) / 2 + 1e-9, 1e6

    def total(R: float) -> float:
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

# ────── diagonals / bounding / altitudes  (unchanged from original) ──────

# ...  (identical helper functions draw_polygon, diagonals_info, etc.)  ...

# For brevity in this snippet we assume draw_polygon and the other
# utilities you had remain exactly the same.  Copy them verbatim from
# your original file below this comment.

# ────── Streamlit UI  – main() ────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Polygon Drawer", layout="centered")
    st.title("📐 Polygon Drawer – לינקו. בתמורה לטובות הנעה")

    n = st.number_input("Number of sides", 3, 12, 4, 1)
    lengths = [
        st.number_input(f"Length {i + 1}", 0.01, 1000.0, 1.0, 0.1, key=f"L{i}")
        for i in range(n)
    ]

    if not is_polygon_possible(lengths):
        st.error("⚠️  Side lengths violate polygon inequality.")
        st.stop()

    if st.checkbox("Provide internal angles?"):
        angs = [
            st.number_input(
                f"∠ {vertex_names(n)[i]}", 1.0, 360.0, round(180 * (n - 2) / n, 1), 1.0, key=f"A{i}"
            )
            for i in range(n)
        ]
        angles = repaired_angles(n, angs)

        # first try strict build
        try:
            poly = build_polygon_strict(lengths, angles)
            build_ok = True
        except ValueError:
            build_ok = False

        if not build_ok:
            st.warning("המצולע שהזנת לא נסגר. בחר שיטת תיקון:")
            method = st.radio(
                "שיטת תיקון:",
                (
                    "פיזור על כל הווקטורים (מינימום שינוי)",
                    "הוסף צלע (קשיח)",
                    "שמור אורכים – תקן זוויות",
                    "שמור זוויות – תקן אורך אחרון",
                ),
            )
            if st.button("בצע תיקון"):
                try:
                    if method.startswith("פיזור"):
                        poly = build_polygon_distribute(lengths, angles)
                    elif method.startswith("הוסף"):
                        poly = build_polygon_add_side(lengths, angles)
                    elif method.startswith("שמור אורכים"):
                        poly = build_polygon_fix_angles(lengths, angles)
                    else:  # keep angles
                        poly = build_polygon_fix_last_length(lengths, angles)
                except ValueError:
                    st.error("בחר שיטת תיקון אחרת")
                    st.stop()
    else:
        poly = circumscribed_polygon(lengths)

    # decide if altitudes are shown
    show_alt = len(poly.pts) == 3

    if st.button("Draw polygon", use_container_width=True):
        fig, diag_list, altitudes = draw_polygon(poly, show_alt)
        st.pyplot(fig, use_container_width=True)

        area_val = shoelace_area(poly.pts)
        _, w, h = bounding_rect(poly.pts)

        num_data = {
            "Area": round(area_val, 4),
            "Bounding width": round(w, 4),
            "Bounding height": round(h, 4),
        }
        if altitudes:
            num_data["Altitudes"] = [round(a["length"], 4) for a in altitudes]
            st.markdown("### Numerical data")
            st.json(num_data, expanded=True)
            txt_bytes = json.dumps({"Numerical data": num_data}, indent=2).encode()
        else:
            diag_data = {
                f"{poly.names[d['i']]}{poly.names[d['j']]}": {
                    "Length": round(d["length"], 3),
                    poly.names[d["i"]]: d["end_i"],
                    poly.names[d["j"]]: d["end_j"],
                }
                for d in diag_list
            }
            st.markdown("### Numerical data")
            st.json(num_data, expanded=True)

            st.markdown("### Diagonals")
            st.json(diag_data, expanded=True)

            txt_bytes = json.dumps({"Numerical data": num_data, "Diagonals": diag_data}, indent=2).encode()

        # ZIP download (unchanged)
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

        st.download_button("Download all (ZIP)", zip_buf, f"{base}.zip", "application/zip")


if __name__ == "__main__":
    main()
