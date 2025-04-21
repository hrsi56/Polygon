# poly_draw_with_diagonals_and_closure.py
# -------------------------------------------------
# Streamlit app – שרטוט מצולעים, אלכסונים, בדיקת סגירה

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

TOL = 1e-2


# ----------   חישובי עזר   ---------- #
def compute_internal_angle(p_prev, p_curr, p_next):
    v1 = np.array(p_prev) - np.array(p_curr)
    v2 = np.array(p_next) - np.array(p_curr)
    cos_t = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_t = np.clip(cos_t, -1, 1)
    return np.degrees(np.arccos(cos_t))


def all_diagonals(pts):
    n = len(pts)
    diags = []
    for i in range(n):
        for j in range(i + 1, n):
            if j == (i + 1) % n or (i == 0 and j == n - 1):
                continue
            length = np.linalg.norm(np.array(pts[j]) - np.array(pts[i]))
            diags.append((i + 1, j + 1, length))
    return diags


# ----------   מצולע כללי   ---------- #
def draw_polygon(sides, lengths, int_angles):
    missing = [i for i, L in enumerate(lengths) if L is None]

    if int_angles:
        ext = [180 - a for a in int_angles]
        headings = np.cumsum([0] + ext[:-1])
    else:
        if len(missing) != 1:
            st.error("אם לא ניתנו זוויות, יש להשאיר צלע אחת ריקה בלבד.")
            return None, None, None
        headings = np.cumsum([0] + [0] * (sides - 1))

    vecs = []
    for hd, L in zip(headings, lengths):
        if L is not None:
            rad = np.radians(hd)
            vecs.append((L * np.cos(rad), L * np.sin(rad)))
        else:
            vecs.append(None)

    if missing:
        dx = sum(v[0] for v in vecs if v)
        dy = sum(v[1] for v in vecs if v)
        L = np.hypot(dx, dy)
        i = missing[0]
        lengths[i] = L
        vecs[i] = (-dx, -dy)

    pts_closed = [(0, 0)]
    for dx, dy in vecs:
        x, y = pts_closed[-1]
        pts_closed.append((x + dx, y + dy))

    pts_unique = pts_closed[:-1]
    n = len(pts_unique)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(*zip(*pts_closed), "-o", lw=2)
    ax.set_aspect("equal")
    ax.axis("off")

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
st.set_page_config(page_title="🎯 מצולעים + אלכסונים + בדיקת סגירה", layout="centered")
st.title("🎯 שרטוט מצולעים (כולל אלכסונים + בדיקת סגירה)")

sides = st.number_input("מספר צלעות", 3, 12, 3, 1)

length_inputs = [st.text_input(f"צלע {i }") for i in range(sides)]
lengths = [None if not L.strip() else float(L) for L in length_inputs]

use_angles = st.checkbox("הזן זוויות פנימיות")
int_angles = None
if use_angles:
    angle_inputs = [st.text_input(f"זווית {i }") for i in range(sides)]
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

        # ----------   בדיקת סגירה   ---------- #
        # מחשבים את הווקטורים שוב כדי לוודא סגירה – בלי לשנות את הקוד הקיים
        if int_angles:
            ext = [180 - a for a in int_angles]
            headings = np.cumsum([0] + ext[:-1])
        else:
            headings = np.cumsum([0] + [0] * (sides - 1))

        vecs = []
        for hd, L in zip(headings, final_lengths):
            rad = np.radians(hd)
            vecs.append((L * np.cos(rad), L * np.sin(rad)))

        dx_total = sum(v[0] for v in vecs)
        dy_total = sum(v[1] for v in vecs)
        gap = np.hypot(dx_total, dy_total)

        if gap > TOL:
            # אורך נדרש של הצלע האחרונה לסגירה
            req_vec = (-sum(v[0] for v in vecs[:-1]), -sum(v[1] for v in vecs[:-1]))
            req_len = np.hypot(*req_vec)
            diff = abs(req_len - final_lengths[-1])

            # Pop‑up  (Toast)  – אם Streamlit < 1.25 השתמש ב‑st.warning
            try:
                st.toast(
                    f"❗  הצורה לא נסגרה כראוי.\n"
                    f"   אורך הצלע האחרונה הדרוש: {req_len:.2f}\n"
                    f"   סטייה: {diff:.2f}",
                    icon="⚠️",
                )
            except AttributeError:
                st.warning(
                    f"❗  הצורה לא נסגרה כראוי.\n"
                    f"  •  סטייה: {diff:.2f}"
                )