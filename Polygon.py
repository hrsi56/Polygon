# poly_draw_with_diagonals.py
# -------------------------------------------------
# Streamlit app – שרטוט מצולעים + אלכסונים וחישוב אורכם
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

TOL = 1e-6


# ----------   חישובי עזר   ---------- #
def compute_internal_angle(p_prev, p_curr, p_next):
    v1 = np.array(p_prev) - np.array(p_curr)
    v2 = np.array(p_next) - np.array(p_curr)
    cos_t = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_t = np.clip(cos_t, -1, 1)
    return np.degrees(np.arccos(cos_t))


def all_diagonals(pts):
    """החזר [(i,j,Len), ...] לכל הזוגות שאינם צלעות."""
    n = len(pts)
    diags = []
    for i in range(n):
        for j in range(i + 1, n):
            # צלע משותפת? (i‑j צמודים במודולו n)
            if j == (i + 1) % n or (i == 0 and j == n - 1):
                continue
            length = np.linalg.norm(np.array(pts[j]) - np.array(pts[i]))
            diags.append((i + 1, j + 1, length))  # +1 להצגה אנושית
    return diags


# ----------   משולש   ---------- #
def draw_triangle(lengths):
    L1, L2, L3 = lengths
    A = (0.0, 0.0)
    B = (L1, 0.0)
    x = (L1 ** 2 + L2 ** 2 - L3 ** 2) / (2 * L1)
    y2 = L2 ** 2 - x ** 2
    if y2 < -TOL:
        st.error("לא ניתן לבנות משולש עם אורכים אלה.")
        return None, None, None
    y = np.sqrt(max(y2, 0.0))
    C = (x, y)

    pts = [A, B, C]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(*zip(*pts, pts[0]), "-o")
    ax.set_aspect("equal")
    ax.axis("off")

    # תוויות צלעות
    for i, (p1, p2) in enumerate([(A, B), (B, C), (C, A)]):
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(
            mx,
            my,
            f"{lengths[i]:.2f}",
            color="blue",
            fontsize=10,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.7),
        )

    # תוויות זוויות
    for i, curr in enumerate(pts):
        prev, nxt = pts[i - 1], pts[(i + 1) % 3]
        ang = compute_internal_angle(prev, curr, nxt)
        bis = (np.array(prev) - np.array(curr)) + (np.array(nxt) - np.array(curr))
        bis /= np.linalg.norm(bis)
        ax.text(
            curr[0] + bis[0] * 0.1 * min(lengths),
            curr[1] + bis[1] * 0.1 * min(lengths),
            f"{ang:.1f}°",
            color="green",
            fontsize=10,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.7),
        )

    return fig, lengths, []  # משולש: אין אלכסונים


# ----------   מצולע כללי   ---------- #
def draw_polygon(sides, lengths, int_angles):
    if sides == 3 and all(L is not None for L in lengths) and int_angles is None:
        return draw_triangle(lengths)

    missing = [i for i, L in enumerate(lengths) if L is None]

    # כיוונים לפי זוויות פנימיות → חיצוניות
    if int_angles:
        ext = [180 - a for a in int_angles]
        headings = np.cumsum([0] + ext[:-1])
    else:
        if len(missing) != 1:
            st.error("אם לא ניתנו זוויות, יש להשאיר צלע אחת ריקה בלבד.")
            return None, None, None
        headings = np.cumsum([0] + [0] * (sides - 1))

    # וקטורים
    vecs = []
    for hd, L in zip(headings, lengths):
        if L is not None:
            rad = np.radians(hd)
            vecs.append((L * np.cos(rad), L * np.sin(rad)))
        else:
            vecs.append(None)

    # השלמת צלע חסרה (אם צריך)
    if missing:
        dx = sum(v[0] for v in vecs if v)
        dy = sum(v[1] for v in vecs if v)
        L = np.hypot(dx, dy)
        i = missing[0]
        lengths[i] = L
        vecs[i] = (-dx, -dy)

    # נקודות – pts_closed כולל נקודת סגירה כפולה
    pts_closed = [(0, 0)]
    for dx, dy in vecs:
        x, y = pts_closed[-1]
        pts_closed.append((x + dx, y + dy))

    pts_unique = pts_closed[:-1]  # ללא הכפולה
    n = len(pts_unique)

    # ----- ציור -----
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(*zip(*pts_closed), "-o", lw=2)
    ax.set_aspect("equal")
    ax.axis("off")

    # ציור אלכסונים
    diag_list = all_diagonals(pts_unique)
    for i, j, _ in diag_list:
        p1, p2 = pts_unique[i - 1], pts_unique[j - 1]
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            "--",
            lw=1,
            color="gray",
            alpha=0.6,
        )

    # תוויות צלעות
    for i in range(sides):
        p1, p2 = pts_closed[i], pts_closed[i + 1]
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(
            mx,
            my,
            f"{lengths[i]:.2f}",
            fontsize=9,
            color="blue",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.7),
        )

    # תוויות זוויות
    for i in range(n):
        prev, curr, nxt = pts_unique[i - 1], pts_unique[i], pts_unique[(i + 1) % n]
        ang = compute_internal_angle(prev, curr, nxt)
        bis = (np.array(prev) - np.array(curr)) + (np.array(nxt) - np.array(curr))
        bis /= np.linalg.norm(bis)
        ax.text(
            curr[0] + bis[0] * 0.1 * min(lengths),
            curr[1] + bis[1] * 0.1 * min(lengths),
            f"{ang:.1f}°",
            fontsize=9,
            color="green",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.7),
        )

    return fig, lengths, diag_list


# ----------   UI Streamlit   ---------- #
st.set_page_config(page_title="🎯 מצולעים + אלכסונים", layout="centered")
st.title("🎯 שרטוט מצולעים (כולל אלכסונים)")

sides = st.number_input("מספר צלעות", 3, 12, 3, 1)

# צלעות
length_inputs = [st.text_input(f"צלע {i + 1}") for i in range(sides)]
lengths = [None if not L.strip() else float(L) for L in length_inputs]

# זוויות פנימיות
use_angles = st.checkbox("הזן זוויות פנימיות")
int_angles = None
if use_angles:
    angle_inputs = [st.text_input(f"זווית {i + 1}") for i in range(sides)]
    if "" in angle_inputs:
        st.error("חובה להזין את כל הזוויות.")
        st.stop()
    int_angles = [float(a) for a in angle_inputs]

if st.button("✏️ שרטוט"):
    fig, final_lengths, diag_list = draw_polygon(sides, lengths, int_angles)
    if fig:
        st.pyplot(fig)

        st.markdown("### אורכי צלעות")
        for i, L in enumerate(final_lengths, 1):
            st.write(f"צלע {i}: {L:.2f}")

        if diag_list:
            st.markdown("### אורכי אלכסונים")
            for i, j, L in diag_list:
                st.write(f"אלכסון {i}–{j}: {L:.2f}")
        else:
            st.markdown("⚪ למשולש אין אלכסונים.")




# -------------  פונקציית עזר חדשה ------------- #
def closure_check(sides, lengths, int_angles, tol=TOL):
    """
    מחזירה: (closed?, gap, suggested_last_len, suggested_last_angle or None)
    """
    # ►‑‑‑ בניית וקטורים בדיוק כמו draw_polygon (ללא תיקון צלע חסרה) ‑‑‑►
    if int_angles:
        ext = [180 - a for a in int_angles]
        headings = np.cumsum([0] + ext[:-1])
    else:
        headings = np.cumsum([0] + [0] * (sides - 1))

    vecs = [(L * np.cos(np.radians(hd)),
             L * np.sin(np.radians(hd))) for hd, L in zip(headings, lengths)]

    pts = [(0.0, 0.0)]
    for dx, dy in vecs:
        x, y = pts[-1]
        pts.append((x + dx, y + dy))

    gap_vec = np.array(pts[-1]) - np.array(pts[0])
    gap = np.linalg.norm(gap_vec)
    closed = gap < tol

    # ►‑‑‑ הצעות תיקון ‑‑‑►
    suggested_len = np.linalg.norm(np.array(pts[-2]) - np.array(pts[0]))
    suggested_ang = None
    if int_angles:
        ext_needed = (360 - sum(ext[:-1])) % 360
        suggested_ang = 180 - ext_needed

    return closed, gap, suggested_len, suggested_ang


# ----------   UI Streamlit   ---------- #
st.set_page_config(page_title="🎯 מצולעים + אלכסונים", layout="centered")
st.title("🎯 שרטוט מצולעים (כולל אלכסונים)")

sides = st.number_input("מספר צלעות", 3, 12, 3, 1)
length_inputs = [st.text_input(f"צלע {i + 1}") for i in range(sides)]
lengths = [None if not L.strip() else float(L) for L in length_inputs]

use_angles = st.checkbox("הזן זוויות פנימיות")
int_angles = None
if use_angles:
    angle_inputs = [st.text_input(f"זווית {i + 1}") for i in range(sides)]
    if "" in angle_inputs:
        st.error("חובה להזין את כל הזוויות.")
        st.stop()
    int_angles = [float(a) for a in angle_inputs]

if st.button("✏️ שרטוט"):
    fig, final_lengths, diag_list = draw_polygon(sides, lengths, int_angles)
    if fig:
        st.pyplot(fig)

        # ---- בדיקת סגירה ----
        closed, gap, sugg_len, sugg_ang = closure_check(
            sides, final_lengths, int_angles
        )
        if not closed:
            st.toast(
                f"⚠️ הצורה לא נסגרה (פער {gap:.2f}).",
                icon="⚠️",
            )
            st.warning(
                f"הצעה: שנה את **אורך הצלע האחרונה** ל‑{sugg_len:.2f}"
                + (
                    f" (או את **הזווית האחרונה** ל‑{sugg_ang:.1f}°)"
                    if sugg_ang is not None
                    else ""
                )
            )
