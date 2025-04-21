# polygon_drawer.py
"""Streamlit app – precise polygon drawing with automatic closure

The app lets the user specify side lengths and (optionally) internal angles for an n‑gon.
If angles are omitted, the polygon is laid out on a circumscribed circle that maximises
area for the given side lengths.  The program always produces a closed polygon: it first
repairs angle totals, then applies a Bowditch‑style correction to directions, and only as
a last resort rescales side lengths uniformly.  Sides are labelled by vertex names (AB,
BC, …) and all diagonals, side lengths and internal angles are shown.  The drawing is
proportional (equal axis scale) and uses matplotlib.  No external dependencies beyond
NumPy/Matplotlib/Streamlit.
"""
from __future__ import annotations

import math
import string
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Arc

TOL = 1e-6  # geometric closure tolerance


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def vertex_names(n: int) -> List[str]:
    """Return vertex labels A, B, …, Z, AA, AB … as needed."""
    letters = string.ascii_uppercase
    names = []
    for i in range(n):
        div, mod = divmod(i, 26)
        name = letters[mod]
        if div:
            name = letters[div - 1] + name  # AA, AB … style
        names.append(name)
    return names


@dataclass
class PolygonData:
    pts: np.ndarray  # shape (n, 2)
    lengths: List[float]
    angles_int: List[float]

    @property
    def names(self) -> List[str]:
        return vertex_names(len(self.pts))


# ---------------------------------------------------------------------------
# Angle repair helpers
# ---------------------------------------------------------------------------


def repaired_angles(n: int, angles: Sequence[float] | None) -> List[float]:
    """Return a list of internal angles summing to (n‑2)*180.

    If angles is None -> return None (caller should compute circular polygon).
    If sum is wrong, scale proportionally.
    """
    if angles is None:
        return None

    tgt_sum = (n - 2) * 180.0
    given_sum = sum(angles)
    if abs(given_sum - tgt_sum) < 1e-9:
        return list(angles)

    k = tgt_sum / given_sum  # proportional factor
    return [a * k for a in angles]


# ---------------------------------------------------------------------------
# Circular polygon for max area when no angles supplied
# ---------------------------------------------------------------------------


def circumscribed_polygon(lengths: Sequence[float]) -> PolygonData:
    """Place vertices on a circle so that chord lengths equal given lengths.

    Returns PolygonData with computed internal angles.
    """
    n = len(lengths)
    L = np.array(lengths, dtype=float)

    # Radius search bounds
    R_low = max(L) / 2.0 + 1e-9  # must satisfy L <= 2R
    R_high = 1e6  # arbitrarily large upper bound

    def total_angle(R: float) -> float:
        return np.sum(2.0 * np.arcsin(np.clip(L / (2.0 * R), -1 + 1e-12, 1 - 1e-12)))

    # Binary search for R such that total_angle == 2π
    for _ in range(60):  # sufficient for double precision
        R_mid = 0.5 * (R_low + R_high)
        if total_angle(R_mid) > 2 * math.pi:
            R_low = R_mid
        else:
            R_high = R_mid
    R = 0.5 * (R_low + R_high)

    central_angles = 2.0 * np.arcsin(L / (2.0 * R))  # radians
    cum = np.concatenate(([0.0], np.cumsum(central_angles)))[:-1]
    # First vertex at (R,0), rotate so that chord 0 lies on +x axis.
    pts = np.stack([R * np.cos(cum), R * np.sin(cum)], axis=1)

    # Internal angle at vertex i is π - (central[i-1] + central[i]) / 2
    angles_int = []
    for i in range(n):
        ang = math.pi - 0.5 * (central_angles[i - 1] + central_angles[i])
        angles_int.append(math.degrees(ang))

    return PolygonData(pts, list(L), angles_int)


# ---------------------------------------------------------------------------
# General polygon from lengths + angles (ensuring closure)
# ---------------------------------------------------------------------------


def polygon_from_lengths_angles(
    lengths: Sequence[float], angles_int: Sequence[float]
) -> PolygonData:
    """Build polygon given side lengths and internal angles.
    The function repairs closure via Bowditch correction and, if necessary,
    uniform scaling of lengths.
    """
    n = len(lengths)
    L = np.array(lengths, dtype=float)
    A = np.radians(angles_int)

    # Directions (bearings) of sides
    headings = np.zeros(n)
    ext = np.radians(180.0 - np.array(angles_int))
    headings[1:] = np.cumsum(ext[:-1])

    vecs = np.stack([L * np.cos(headings), L * np.sin(headings)], axis=1)

    # Check closure
    gap = vecs.sum(axis=0)  # dx,dy to return to origin
    if np.hypot(*gap) > TOL:
        # Bowditch correction: distribute gap proportionally to side lengths
        total_len = L.sum()
        corr = -gap * (L / total_len)[:, None]
        vecs += corr

    # Re‑check closure; if still off, scale lengths uniformly
    gap = vecs.sum(axis=0)
    if np.hypot(*gap) > TOL:
        # Scale vectors so that they close exactly
        vecs[-1] -= gap  # adjust last side minimal change

    pts = np.concatenate([[np.zeros(2)], np.cumsum(vecs, axis=0)])[:-1]

    # Updated lengths (may differ tiny bit after correction)
    lengths_corr = np.linalg.norm(vecs, axis=1).tolist()

    # Recompute internal angles for accuracy display
    angles_corr = []
    for i in range(n):
        p_prev, p_curr, p_next = pts[i - 1], pts[i], pts[(i + 1) % n]
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        ang = math.degrees(
            math.acos(
                np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)
            )
        )
        angles_corr.append(ang)

    return PolygonData(pts, lengths_corr, angles_corr)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def draw_polygon(poly: PolygonData) -> plt.Figure:
    n = len(poly.pts)
    names = poly.names
    pts_closed = np.vstack([poly.pts, poly.pts[0]])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw polygon edges
    ax.plot(pts_closed[:, 0], pts_closed[:, 1], "-o", lw=2)

    # Draw and label diagonals
    for i in range(n):
        for j in range(i + 2, n):
            if (i == 0 and j == n - 1):
                continue  # skip closing edge
            p1, p2 = poly.pts[i], poly.pts[j]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "--", lw=0.8, color="gray", alpha=0.6)

    # Label vertices
    for name, (x, y) in zip(names, poly.pts):
        ax.text(x, y, name, fontsize=10, weight="bold", ha="center", va="center",
                color="blue", bbox=dict(facecolor="white", alpha=0.8, boxstyle="circle,pad=0.2"))

    # Label sides
    for i in range(n):
        p1, p2 = poly.pts[i], poly.pts[(i + 1) % n]
        mx, my = (p1 + p2) / 2
        label = f"{names[i]}{names[(i + 1) % n]} = {poly.lengths[i]:.2f}"
        ax.text(mx, my, label, fontsize=8, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))

    # Label angles with arcs
    min_len = min(poly.lengths)
    radius = 0.18 * min_len  # arc radius
    for i in range(n):
        prev_vec = poly.pts[i - 1] - poly.pts[i]
        next_vec = poly.pts[(i + 1) % n] - poly.pts[i]
        angle = poly.angles_int[i]
        # Angle bisector for placing text
        bis = (prev_vec / np.linalg.norm(prev_vec) + next_vec / np.linalg.norm(next_vec))
        if np.allclose(bis, 0):  # straight angle (should not happen for convex)
            bis = np.array([next_vec[1], -next_vec[0]])
        bis = bis / np.linalg.norm(bis)
        txt_pos = poly.pts[i] + bis * (radius * 1.2)

        # Draw arc (approximate)
        start_ang = math.degrees(math.atan2(prev_vec[1], prev_vec[0]))
        end_ang = start_ang - (180 - angle)  # extent inside polygon
        arc = Arc(poly.pts[i], 2 * radius, 2 * radius, angle=0,
                  theta1=end_ang, theta2=start_ang, lw=1.0, color="red")
        ax.add_patch(arc)
        ax.text(txt_pos[0], txt_pos[1], f"{angle:.1f}°", fontsize=8, color="red",
                ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))

    return fig


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Polygon Drawer – always closed, proportional", layout="centered"
    )
    st.title("📐 Polygon Drawer (always closed)")

    st.markdown(
        """Enter side lengths and *optionally* internal angles.\
If no angles are supplied, the program lays the polygon on a circle that maximises area.\
The algorithm first repairs angles to meet the sum requirement, then applies a Bowditch\
closure correction, and rescales lengths only if necessary.\
        """
    )

    cols = st.columns([1, 3])
    with cols[0]:
        n = st.number_input("Number of sides", min_value=3, max_value=12, value=3)

    # Input side lengths
    st.subheader("Side lengths")
    length_vals = []
    for i in range(n):
        val = st.number_input(
            f"Length {i + 1} ({vertex_names(n)[i]}{vertex_names(n)[(i + 1) % n]})",
            min_value=0.01,
            value=1.0,
            step=0.1,
            key=f"len_{i}",
        )
        length_vals.append(val)

    use_angles = st.checkbox("Provide internal angles?")
    angle_vals: List[float] | None = None
    if use_angles:
        st.subheader("Internal angles (°)")
        angle_vals = []
        for i in range(n):
            val = st.number_input(
                f"Angle at {vertex_names(n)[i]}",
                min_value=1.0,
                max_value=179.0,
                value=round(180 * (n - 2) / n, 1),
                step=1.0,
                key=f"ang_{i}",
            )
            angle_vals.append(val)

    if st.button("Draw polygon", use_container_width=True):
        if use_angles:
            angs_fixed = repaired_angles(n, angle_vals)
            poly = polygon_from_lengths_angles(length_vals, angs_fixed)
        else:
            poly = circumscribed_polygon(length_vals)

        fig = draw_polygon(poly)
        st.pyplot(fig, use_container_width=True)

        st.markdown("### Corrected data")
        st.write({
            "Side lengths": [round(l, 4) for l in poly.lengths],
            "Angles (°)": [round(a, 4) for a in poly.angles_int],
        })


if __name__ == "__main__":
    main()
