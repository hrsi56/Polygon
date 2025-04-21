# poly_draw_fixed.py
# -------------------------------------------------
# Streamlit app – שרטוט מצולעים עם תוויות צלע וזווית מדויקות
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

TOL = 1e-6


# ----------   חישובי עזר   ---------- #
def compute_internal_angle(p_prev, p_curr, p_next):
    """החזרת הזווית הפנימית (במעלות) בקודקוד p_curr."""
    v1 = np.array(p_prev) - np.array(p_curr)
    v2 = np.array(p_next) - np.array(p_curr)
    cos_t = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_t = np.clip(cos_t, -1, 1)              # הגנה על arccos
    return np.degrees(np.arccos(cos_t))


# ----------   משולש   ---------- #
def draw_triangle(lengths):
    L1, L2, L3 = lengths
    A = (0.0, 0.0)
    B = (L1, 0.0)

    # מציאת נקודה C לפי הקוסינוס
    x = (L1 ** 2 + L2 ** 2 - L3 ** 2) / (2 * L1)
    y2 = L2 ** 2 - x ** 2
    if y2 < -TOL:
        st.error("לא ניתן לבנות משולש עם אורכים אלה.")
        return None, None
    y = np.sqrt(max(y2, 0.0))
    C = (x, y)

    pts = [A, B, C]
    sides = [(A, B), (B, C), (C, A)]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(*zip(*pts, pts[0]), "-o")
    ax.set_aspect("equal")
    ax.axis("off")

    # תוויות צלעות
    for i, (p1, p2) in enumerate(sides):
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

    return fig, lengths


# ----------   מצולע כללי   ---------- #
def draw_polygon(sides, lengths, int_angles):
    # מקרה פרטי – משולש מלא ללא זוויות
    if sides == 3 and all(L is not None for L in lengths) and int_angles is None:
        return draw_triangle(lengths)

    # רשימת צלעות חסרות
    missing = [i for i, L in enumerate(lengths) if L is None]

    # חישוב כיוונים (headings) לפי זוויות חיצוניות
    if int_angles:
        ext = [180 - a for a in int_angles]
        headings = np.cumsum([0] + ext[:-1])
    else:
        if len(missing) != 1:
            st.error("אם לא ניתנו זוויות, יש להשאיר צלע אחת ריקה בלבד.")
            return None, None
        headings = np.cumsum([0] + [0] * (sides - 1))  # הנחייה השרירותית 0°

    # וקטורים
    vecs = []
    for hd, L in zip(headings, lengths):
        if L is not None:
            rad = np.radians(hd)
            vecs.append((L * np.cos(rad), L * np.sin(rad)))
        else:
            vecs.append(None)

    # חישוב הצלע החסרה (אם קיימת)
    if missing:
        dx = sum(v[0] for v in vecs if v)
        dy = sum(v[1] for v in vecs if v)
        L = np.hypot(dx, dy)
        i = missing[0]
        lengths[i] = L
        vecs[i] = (-dx, -dy)

    # רשימת נקודות: pts_closed כולל נקודת סגירה כפולה
    pts_closed = [(0, 0)]
    for dx, dy in vecs:
        x, y = pts_closed[-1]
        pts_closed.append((x + dx, y + dy))

    # ----- ציור -----
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(*zip(*pts_closed), "-o")
    ax.set_aspect("equal")
    ax.axis("off")

    # תוויות צלעות
    for i in range(sides):
        p1, p2 = pts_closed[i], pts_closed[i + 1]
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(
            mx,
            my,
            f"{lengths[i]:.2f}",
            fontsize=10,
            color="blue",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.7),
        )

    # תוויות זוויות – משתמשים ב‑pts_unique ללא נקודת‑הסגירה הכפולה
    pts_unique = (
        pts_closed[:-1]
        if np.allclose(pts_closed[0], pts_closed[-1], atol=TOL)
        else pts_closed
    )
    n = len(pts_unique)

    for i in range(n):
        prev = pts_unique[i - 1]
        curr = pts_unique[i]
        nxt = pts_unique[(i + 1) % n]

        ang = compute_internal_angle(prev, curr, nxt)
        bis = (np.array(prev) - np.array(curr)) + (np.array(nxt) - np.array(curr))
        bis /= np.linalg.norm(bis)
        ax.text(
            curr[0] + bis[0] * 0.1 * min(lengths),
            curr[1] + bis[1] * 0.1 * min(lengths),
            f"{ang:.1f}°",
            fontsize=10,
            color="green",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.7),
        )

    return fig, lengths


# ----------   UI Streamlit   ---------- #
st.set_page_config(page_title="🎯 שרטוט מצולעים מתוקן", layout="centered")
st.title("🎯 שרטוט מצולעים מתוקן")

sides = st.number_input("מספר צלעות", 3, 12, 3, 1)

# קלט צלעות
lengths = [st.text_input(f"צלע {i + 1}") for i in range(sides)]
lengths = [None if not L.strip() else float(L) for L in lengths]

# קלט זוויות פנימיות (רשות)
use_angles = st.checkbox("הזן זוויות פנימיות")
int_angles = None
if use_angles:
    int_angles = [st.text_input(f"זווית {i + 1}") for i in range(sides)]
    if "" in int_angles:
        st.error("חובה להזין את כל הזוויות.")
        st.stop()
    int_angles = [float(a) for a in int_angles]

# כפתור שרטוט
if st.button("✏️ שרטוט"):
    fig, final_lengths = draw_polygon(sides, lengths, int_angles)
    if fig:
        st.pyplot(fig)
        st.markdown("**אורכי צלעות סופיים:**")
        for i, L in enumerate(final_lengths, 1):
            st.write(f"צלע {i}: {L:.2f}")