# poly_draw_full.py
# -------------------------------------------------
# Streamlit app – שרטוט מצולעים, אלכסונים, בדיקת סגירה והצגת אורכים
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

TOL = 1e-6  # סף לטולרנס חישובי


# ----------   פונקציות עזר   ---------- #
def compute_internal_angle(p_prev, p_curr, p_next):
    """החזרת הזווית הפנימית (במעלות) בקודקוד p_curr."""
    v1 = np.array(p_prev) - np.array(p_curr)
    v2 = np.array(p_next) - np.array(p_curr)
    cos_t = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_t = np.clip(cos_t, -1, 1)
    return np.degrees(np.arccos(cos_t))


def all_diagonals(pts):
    """החזרת רשימת האלכסונים [(i,j,length), ...] למצולע סגור ב‑pts."""
    n = len(pts)
    diags = []
    for i in range(n):
        for j in range(i + 1, n):
            # אם הקודקודים צמודים (צלע) – לא אלכסון
            if j == (i + 1) % n or (i == 0 and j == n - 1):
                continue
            length = np.linalg.norm(np.array(pts[j]) - np.array(pts[i]))
            diags.append((i + 1, j + 1, length))  # +1 להצגה אנושית
    return diags


def check_closure(sides, lengths, int_angles):
    """
    בדוק אם המצולע נסגר – כלומר סכום הווקטורים חוזר (בקירוב) ל‑(0,0).
    מחזיר True אם נסגר, אחרת False.
    """
    # כיוונים (headings) זהים לאלה שמשמשים ב‑draw_polygon
    if int_angles:  # יש זוויות – משתמשים בזוויות חיצוניות
        ext = [180 - a for a in int_angles]
        headings = np.cumsum([0] + ext[:-1])
    else:           # אין זוויות → מניחים 0° לכל הצלעות מלבד הראשונה
        headings = np.cumsum([0] + [0] * (sides - 1))

    # סכום וקטורים
    dx = dy = 0.0
    for hd, L in zip(headings, lengths):
        if L is None:          # חסרה צלע → לא יכולים לקבוע סגירה
            return False
        rad = np.radians(hd)
        dx += L * np.cos(rad)
        dy += L * np.sin(rad)

    return np.hypot(dx, dy) < TOL


# ----------   שרטוט משולש   ---------- #
def draw_triangle(lengths):
    L1, L2, L3 = lengths
    A = (0.0, 0.0)
    B = (L1, 0.0)

    # מציאת נקודה C לפי משפט הקוסינוסים
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
        ax.text(mx, my, f"{lengths[i]:.2f}", color="blue", fontsize=10,
                ha="center", va="center", bbox=dict(facecolor="white", alpha=0.7))

    # תוויות זוויות
    for i, curr in enumerate(pts):
        prev, nxt = pts[i - 1], pts[(i + 1) % 3]
        ang = compute_internal_angle(prev, curr, nxt)
        bis = (np.array(prev) - np.array(curr)) + (np.array(nxt) - np.array(curr))
        bis /= np.linalg.norm(bis)
        ax.text(curr[0] + bis[0] * 0.1 * min(lengths),
                curr[1] + bis[1] * 0.1 * min(lengths),
                f"{ang:.1f}°", color="green", fontsize=10,
                ha="center", va="center", bbox=dict(facecolor="white", alpha=0.7))

    return fig, lengths, []     # אין אלכסונים במשולש


# ----------   שרטוט מצולע כללי   ---------- #
def draw_polygon(sides, lengths, int_angles):
    # מקרה פרטי – משולש מלא ללא זוויות
    if sides == 3 and all(L is not None for L in lengths) and int_angles is None:
        return draw_triangle(lengths)

    missing = [i for i, L in enumerate(lengths) if L is None]

    # כיווני הצלעות
    if int_angles:  # ידועות זוויות פנימיות
        ext = [180 - a for a in int_angles]          # זוויות חיצוניות
        headings = np.cumsum([0] + ext[:-1])         # כיוון מצטבר
    else:          # בלי זוויות – חייבת להיות צלע חסרה יחידה
        if len(missing) != 1:
            st.error("אם לא ניתנו זוויות, יש להשאיר צלע אחת בלבד ריקה.")
            return None, None, None
        headings = np.cumsum([0] + [0] * (sides - 1))

    # בניית וקטורים
    vecs = []
    for hd, L in zip(headings, lengths):
        if L is not None:
            rad = np.radians(hd)
            vecs.append((L * np.cos(rad), L * np.sin(rad)))
        else:
            vecs.append(None)

    # השלמת הצלע החסרה (אם קיימת)
    if missing:
        dx = sum(v[0] for v in vecs if v)
        dy = sum(v[1] for v in vecs if v)
        L = np.hypot(dx, dy)
        i = missing[0]
        lengths[i] = L
        vecs[i] = (-dx, -dy)

    # נקודות: pts_closed כולל את נקודת‑הסגירה הכפולה
    pts_closed = [(0, 0)]
    for dx, dy in vecs:
        x, y = pts_closed[-1]
        pts_closed.append((x + dx, y + dy))

    pts_unique = pts_closed[:-1]   # ללא הכפולה
    n = len(pts_unique)

    # --- ציור ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(*zip(*pts_closed), "-o", lw=2)
    ax.set_aspect("equal")
    ax.axis("off")

    # ציור אלכסונים
    diag_list = all_diagonals(pts_unique) if n >= 4 else []
    for i, j, _ in diag_list:
        p1, p2 = pts_unique[i - 1], pts_unique[j - 1]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "--", lw=1, color="gray", alpha=0.6)

    # תוויות צלעות
    for i in range(sides):
        p1, p2 = pts_closed[i], pts_closed[i + 1]
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my, f"{lengths[i]:.2f}", fontsize=9, color="blue",
                ha="center", va="center", bbox=dict(facecolor="white", alpha=0.7))

    # תוויות זוויות
    for i in range(n):
        prev, curr, nxt = pts_unique[i - 1], pts_unique[i], pts_unique[(i + 1) % n]
        ang = compute_internal_angle(prev, curr, nxt)
        bis = (np.array(prev) - np.array(curr)) + (np.array(nxt) - np.array(curr))
        bis /= np.linalg.norm(bis)
        ax.text(curr[0] + bis[0] * 0.1 * min(lengths),
                curr[1] + bis[1] * 0.1 * min(lengths),
                f"{ang:.1f}°", fontsize=9, color="green",
                ha="center", va="center", bbox=dict(facecolor="white", alpha=0.7))

    return fig, lengths, diag_list


# ----------   UI Streamlit   ---------- #
st.set_page_config(page_title="🎯 מצולעים עם אלכסונים ובדיקת סגירה",
                   layout="centered")

st.title("🎯 שרטוט מצולעים (אלכסונים + בדיקת סגירה)")

sides = st.number_input("מספר צלעות", min_value=3, max_value=12, value=4, step=1)

# קלט צלעות
length_inputs = [st.text_input(f"צלע {i + 1}") for i in range(sides)]
lengths = [None if not L.strip() else float(L) for L in length_inputs]

# קלט זוויות פנימיות (רשות)
use_angles = st.checkbox("הזן זוויות פנימיות")
int_angles = None
if use_angles:
    angle_inputs = [st.text_input(f"זווית {i + 1}") for i in range(sides)]
    if "" in angle_inputs:
        st.error("חובה להזין את כל הזוויות.")
        st.stop()
    int_angles = [float(a) for a in angle_inputs]

# כפתור שרטוט
if st.button("✏️ שרטוט"):
    fig, final_lengths, diag_list = draw_polygon(sides, lengths, int_angles)
    if fig:
        st.pyplot(fig)

        # ------- בדיקת סגירה -------
        if not check_closure(sides, final_lengths, int_angles):
            st.error("⚠️ הצורה אינה נסגרת כראוי (סכום הווקטורים שונה מאפס).")
        # ---------------------------

        st.markdown("### אורכי צלעות")
        for i, L in enumerate(final_lengths, 1):
            st.write(f"צלע {i}: {L:.2f}")

        if diag_list:
            st.markdown("### אורכי אלכסונים")
            for i, j, L in diag_list:
                st.write(f"אלכסון {i}–{j}: {L:.2f}")
        elif sides == 3:
            st.markdown("⚪ למשולש אין אלכסונים.")