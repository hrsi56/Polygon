# poly_draw_with_check.py  (2025‑04‑21)
# -------------------------------------------------
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

TOL = 1e-6


# ----------  כלי‑עזר  ---------- #
def compute_internal_angle(p_prev, p_curr, p_next):
    v1 = np.array(p_prev) - np.array(p_curr)
    v2 = np.array(p_next) - np.array(p_curr)
    cos_t = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_t = np.clip(cos_t, -1, 1)
    return np.degrees(np.arccos(cos_t))


def all_diagonals(pts):
    n = len(pts)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            if j == (i + 1) % n or (i == 0 and j == n - 1):
                continue
            out.append((i + 1, j + 1, np.linalg.norm(np.array(pts[j]) - np.array(pts[i]))))
    return out


def closure_info(pts_closed):
    """מחזיר (is_closed, gap_vec, gap_len)."""
    gap_vec = np.array(pts_closed[0]) - np.array(pts_closed[-1])
    gap_len = np.linalg.norm(gap_vec)
    return gap_len <= TOL, gap_vec, gap_len


# ----------  משולש  ---------- #
def draw_triangle(lengths):
    L1, L2, L3 = lengths
    A = (0.0, 0.0)
    B = (L1, 0.0)
    x = (L1**2 + L2**2 - L3**2) / (2 * L1)
    y2 = L2**2 - x**2
    if y2 < -TOL:
        st.error("לא ניתן לבנות משולש עם אורכים אלה.")
        return None, None, [], True
    C = (x, np.sqrt(max(y2, 0.0)))

    pts = [A, B, C]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(*zip(*pts, pts[0]), "-o")
    ax.set_aspect("equal")
    ax.axis("off")

    # צלעות
    for i, (p1, p2) in enumerate([(A, B), (B, C), (C, A)]):
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my, f"{lengths[i]:.2f}", color="blue", fontsize=10,
                ha="center", va="center", bbox=dict(facecolor="white", alpha=0.7))

    # זוויות
    for i, curr in enumerate(pts):
        prev, nxt = pts[i - 1], pts[(i + 1) % 3]
        ang = compute_internal_angle(prev, curr, nxt)
        bis = (np.array(prev) - np.array(curr)) + (np.array(nxt) - np.array(curr))
        bis /= np.linalg.norm(bis)
        ax.text(curr[0] + bis[0] * 0.1 * min(lengths),
                curr[1] + bis[1] * 0.1 * min(lengths),
                f"{ang:.1f}°", color="green", fontsize=10,
                ha="center", va="center", bbox=dict(facecolor="white", alpha=0.7))

    return fig, lengths, [], True  # משולש תמיד סגור


# ----------  מצולע כללי  ---------- #
def draw_polygon(sides, lengths, int_angles):
    # מקרה פרטי – משולש שלם
    if sides == 3 and all(L is not None for L in lengths) and int_angles is None:
        return draw_triangle(lengths)

    missing = [i for i, L in enumerate(lengths) if L is None]

    # Headings לפי זוויות חיצוניות
    if int_angles:
        ext = [180 - a for a in int_angles]
        headings = np.cumsum([0] + ext[:-1])
    else:
        if len(missing) != 1:
            st.error("אם לא ניתנו זוויות, יש להשאיר צלע אחת ריקה בלבד.")
            return None, None, None, True
        headings = np.cumsum([0] + [0] * (sides - 1))  # 0° שרירותי

    # וקטורים
    vecs = []
    for hd, L in zip(headings, lengths):
        vecs.append((L * np.cos(np.radians(hd)), L * np.sin(np.radians(hd))) if L is not None else None)

    # השלמת צלע חסרה
    if missing:
        dx = sum(v[0] for v in vecs if v)
        dy = sum(v[1] for v in vecs if v)
        L = np.hypot(dx, dy)
        i = missing[0]
        lengths[i] = L
        vecs[i] = (-dx, -dy)

    # נקודות – כולל סגירה
    pts_closed = [(0.0, 0.0)]
    for dx, dy in vecs:
        x, y = pts_closed[-1]
        pts_closed.append((x + dx, y + dy))

    pts = pts_closed[:-1]  # ללא הכפולה
    n = len(pts)

    # -------- ציור --------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(*zip(*pts_closed), "-o", lw=2)
    ax.set_aspect("equal")
    ax.axis("off")

    # אלכסונים
    diag_list = all_diagonals(pts)
    for i, j, _ in diag_list:
        p1, p2 = pts[i - 1], pts[j - 1]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                "--", lw=1, color="gray", alpha=0.6)

    # תוויות צלעות
    for i in range(sides):
        p1, p2 = pts_closed[i], pts_closed[i + 1]
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my, f"{lengths[i]:.2f}", fontsize=9, color="blue",
                ha="center", va="center", bbox=dict(facecolor="white", alpha=0.7))

    # תוויות זוויות
    for i in range(n):
        prev, curr, nxt = pts[i - 1], pts[i], pts[(i + 1) % n]
        ang = compute_internal_angle(prev, curr, nxt)
        bis = (np.array(prev) - np.array(curr)) + (np.array(nxt) - np.array(curr))
        bis /= np.linalg.norm(bis)
        ax.text(curr[0] + bis[0] * 0.1 * min(lengths),
                curr[1] + bis[1] * 0.1 * min(lengths),
                f"{ang:.1f}°", fontsize=9, color="green",
                ha="center", va="center", bbox=dict(facecolor="white", alpha=0.7))

    # -------- בדיקת סגירה --------
    is_closed, gap_vec, gap_len = closure_info(pts_closed)
    if not is_closed:
        # וקטור מוצע לסגירה
        rec_len = gap_len
        rec_heading = (np.degrees(np.arctan2(gap_vec[1], gap_vec[0])) + 360) % 360
        v_last = np.array(pts_closed[-1]) - np.array(pts_closed[-2])
        curr_heading = (np.degrees(np.arctan2(v_last[1], v_last[0])) + 360) % 360
        delta_heading = ((rec_heading - curr_heading + 180) % 360) - 180
        delta_len = rec_len - lengths[-1]

        msg = (
            f"⚠️ המצולע לא נסגר (פער {gap_len:.2f}).\n\n"
            f"* אורך מומלץ לצלע האחרונה: **{rec_len:.2f}** "
            f"(שינוי {delta_len:+.2f})\n"
            f"* כיוון מומלץ לצלע האחרונה: **{rec_heading:.1f}°**\n"
            f"* שינוי זווית אחרונה: **{delta_heading:+.1f}°**"
        )
        if int_angles:
            # זווית‑פנים חדשה משוערת
            hd_prev = (np.degrees(np.arctan2(
                (pts_closed[-2][1] - pts_closed[-3][1]),
                (pts_closed[-2][0] - pts_closed[-3][0])
            )) + 360) % 360
            new_ext = (rec_heading - hd_prev + 360) % 360
            new_int = 180 - new_ext
            msg += f"\n* זווית‑פנים אחרונה מומלצת: **{new_int:.1f}°**"
        st.error(msg)
    else:
        st.success("✅ המצולע נסגר בהצלחה!")

    return fig, lengths, diag_list, is_closed


# ----------  UI  ---------- #
st.set_page_config("🎯 מצולעים + בדיקת סגירה", layout="centered")
st.title("🎯 שרטוט מצולעים (עם אלכסונים ובדיקת סגירה)")

sides = st.number_input("מספר צלעות", 3, 12, 3, 1)
length_inputs = [st.text_input(f"צלע {i + 1}") for i in range(sides)]
lengths = [None if not t.strip() else float(t) for t in length_inputs]

use_angles = st.checkbox("הזן זוויות פנימיות")
int_angles = None
if use_angles:
    angle_inputs = [st.text_input(f"זווית {i + 1}") for i in range(sides)]
    if "" in angle_inputs:
        st.error("חובה להזין את כל הזוויות.")
        st.stop()
    int_angles = [float(a) for a in angle_inputs]

if st.button("✏️ שרטוט"):
    fig, final_lengths, diag_list, _ = draw_polygon(sides, lengths, int_angles)
    if fig:
        st.pyplot(fig)
        st.markdown("### אורכי צלעות")
        for i, L in enumerate(final_lengths, 1):
            st.write(f"צלע {i}: {L:.2f}")
        if diag_list:
            st.markdown("### אורכי אלכסונים")
            for i, j, L in diag_list:
                st.write(f"אלכסון {i}–{j}: {L:.2f}")