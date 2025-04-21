# poly_draw_with_check.py
# -------------------------------------------------
# Streamlit app – שרטוט מצולעים + אלכסונים + בדיקת סגירה
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

TOL = 1e-6


# ----------   עזר: זווית פנימית   ---------- #
def compute_internal_angle(p_prev, p_curr, p_next):
    v1 = np.array(p_prev) - np.array(p_curr)
    v2 = np.array(p_next) - np.array(p_curr)
    cos_t = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_t = np.clip(cos_t, -1, 1)
    return np.degrees(np.arccos(cos_t))


# ----------   עזר: כל האלכסונים   ---------- #
def all_diagonals(pts):
    n = len(pts)
    diags = []
    for i in range(n):
        for j in range(i + 1, n):
            if j == (i + 1) % n or (i == 0 and j == n - 1):
                continue                      # צלע משותפת
            length = np.linalg.norm(np.array(pts[j]) - np.array(pts[i]))
            diags.append((i + 1, j + 1, length))  # +1 – ספירה “אנושית”
    return diags


# ----------   עזר: בדיקת סגירה   ---------- #
def check_closure(pts_closed):
    """בודק אם הנקודה האחרונה חופפת לראשונה (בתוך TOL)."""
    if np.allclose(pts_closed[0], pts_closed[-1], atol=TOL):
        return True, 0.0, 0.0, 0.0
    dx = pts_closed[0][0] - pts_closed[-1][0]
    dy = pts_closed[0][1] - pts_closed[-1][1]
    gap = np.hypot(dx, dy)
    return False, gap, dx, dy


# ----------   משולש   ---------- #
def draw_triangle(lengths):
    L1, L2, L3 = lengths
    A = (0.0, 0.0)
    B = (L1, 0.0)
    x = (L1**2 + L2**2 - L3**2) / (2 * L1)
    y2 = L2**2 - x**2
    if y2 < -TOL:
        st.error("לא ניתן לבנות משולש עם אורכים אלה.")
        return None, None, [], True
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

    return fig, lengths, [], True  # משולש תמיד סגור


# ----------   מצולע כללי   ---------- #
def draw_polygon(sides, lengths, int_angles):
    if sides == 3 and all(L is not None for L in lengths) and int_angles is None:
        return draw_triangle(lengths)

    missing = [i for i, L in enumerate(lengths) if L is None]

    # כיוונים (Headings) לפי זוויות חיצוניות
    if int_angles:
        ext = [180 - a for a in int_angles]
        headings = np.cumsum([0] + ext[:-1])
    else:
        if len(missing) != 1:
            st.error("אם לא ניתנו זוויות, יש להשאיר צלע אחת ריקה בלבד.")
            return None, None, None, True
        headings = np.cumsum([0] + [0] * (sides - 1))

    # וקטורים
    vecs = []
    for hd, L in zip(headings, lengths):
        if L is not None:
            rad = np.radians(hd)
            vecs.append((L * np.cos(rad), L * np.sin(rad)))
        else:
            vecs.append(None)

    # השלמת צלע חסרה
    if missing:
        dx = sum(v[0] for v in vecs if v)
        dy = sum(v[1] for v in vecs if v)
        L = np.hypot(dx, dy)
        i = missing[0]
        lengths[i] = L
        vecs[i] = (-dx, -dy)

    # נקודות (עם סגירה כפולה)
    pts_closed = [(0, 0)]
    for dx, dy in vecs:
        x, y = pts_closed[-1]
        pts_closed.append((x + dx, y + dy))

    pts_unique = pts_closed[:-1]
    n = len(pts_unique)

    # ----- ציור -----
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(*zip(*pts_closed), "-o", lw=2)
    ax.set_aspect("equal")
    ax.axis("off")

    # אלכסונים
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

    # ----- בדיקת סגירה -----
    closed, gap, dx, dy = check_closure(pts_closed)
    if not closed:
        # המלצות תיקון
        last_len = lengths[-1]
        new_len = last_len + gap
        v_last = vecs[-1]
        hd_last = (np.degrees(np.arctan2(v_last[1], v_last[0]))) % 360
        hd_needed = (np.degrees(np.arctan2(-dy, -dx))) % 360
        delta_angle = ((hd_needed - hd_last + 180) % 360) - 180

        st.error(
            f"⚠️ המצולע לא נסגר (פער {gap:.2f}).\n\n"
            f"* אורך מומלץ לצלע האחרונה: **{new_len:.2f}** (במקום {last_len:.2f})\n"
            f"* שינוי זווית אחרונה: **{delta_angle:+.1f}°**"
        )

    return fig, lengths, diag_list, closed


# ----------   UI Streamlit   ---------- #
st.set_page_config(page_title="🎯 מצולעים + בדיקת סגירה", layout="centered")
st.title("🎯 שרטוט מצולעים (עם אלכסונים ובדיקת סגירה)")

sides = st.number_input("מספר צלעות", 3, 12, 3, 1)

# צלעות
length_inputs = [st.text_input(f"צלע {i + 1}") for i in range(sides)]
lengths = [None if not L.strip() else float(L) for L in length_inputs]

# זוויות פנימיות (אופציונלי)
use_angles = st.checkbox("הזן זוויות פנימיות")
int_angles = None
if use_angles:
    angle_inputs = [st.text_input(f"זווית {i + 1}") for i in range(sides)]
    if "" in angle_inputs:
        st.error("חובה להזין את כל הזוויות.")
        st.stop()
    int_angles = [float(a) for a in angle_inputs]

if st.button("✏️ שרטוט"):
    fig, final_lengths, diag_list, closed = draw_polygon(
        sides, lengths, int_angles
    )
    if fig:
        st.pyplot(fig)

        st.markdown("### אורכי צלעות")
        for i, L in enumerate(final_lengths, 1):
            st.write(f"צלע {i}: {L:.2f}")

        if diag_list:
            st.markdown("### אורכי אלכסונים")
            for i, j, L in diag_list:
                st.write(f"אלכסון {i}–{j}: {L:.2f}")

        if closed:
            st.success("✅ המצולע נסגר בהצלחה!")