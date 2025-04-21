# poly_draw_close_fix.py
# -------------------------------------------------
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

TOL = 1e-6


# ----------   עזר   ---------- #
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
                continue          # זו צלע, לא אלכסון
            out.append((i + 1, j + 1,
                        np.linalg.norm(np.array(pts[i]) - np.array(pts[j]))))
    return out


# ----------   מצולע   ---------- #
def draw_polygon(sides, lengths, int_angles, auto_fix):
    # -------------------------------------------------
    # 1. כיוונים (headings)
    if int_angles:
        ext = [180 - a for a in int_angles]
        headings = np.cumsum([0] + ext[:-1])
    else:
        # ללא זוויות – חייבת להיות צלע אחת חסרה או Auto‑fix
        missing = [i for i, L in enumerate(lengths) if L is None]
        if not missing and not auto_fix:
            st.error("ללא זוויות יש להשאיר צלע אחת ריקה **או** לסמן 'סגירה אוטומטית'.")
            return None, None, None
        headings = np.cumsum([0] + [0] * (sides - 1))

    # -------------------------------------------------
    # 2. וקטורים ראשוניים
    vecs = []
    for hd, L in zip(headings, lengths):
        if L is None:
            vecs.append(None)
        else:
            rad = np.radians(hd)
            vecs.append((L * np.cos(rad), L * np.sin(rad)))

    # -------------------------------------------------
    # 3. צלע חסרה → משלים לסגירה
    if None in vecs:
        i_missing = vecs.index(None)
        dx = sum(v[0] for v in vecs if v)
        dy = sum(v[1] for v in vecs if v)
        vecs[i_missing] = (-dx, -dy)
        lengths[i_missing] = np.hypot(dx, dy)

    # -------------------------------------------------
    # 4. בניית נקודות
    pts = [(0, 0)]
    for vx, vy in vecs:
        x, y = pts[-1]
        pts.append((x + vx, y + vy))

    # -------------------------------------------------
    # 5. בדיקת סגירה
    gap = np.linalg.norm(np.array(pts[-1]) - np.array(pts[0]))
    if gap > TOL:
        if auto_fix:
            # מזיזים את **הצלע האחרונה** כדי לסגור
            vx, vy = vecs[-1]
            vx -= (pts[-1][0] - pts[0][0])
            vy -= (pts[-1][1] - pts[0][1])
            vecs[-1] = (vx, vy)
            lengths[-1] = np.hypot(vx, vy)

            # בונים נקודות מחדש
            pts = [(0, 0)]
            for vx, vy in vecs:
                x, y = pts[-1]
                pts.append((x + vx, y + vy))
            gap = 0.0
        else:
            st.error(f"המצולע לא נסגר: סטייה {gap:.3f}. "
                     "סמן 'סגירה אוטומטית' או תקן את הקלט.")
            return None, None, None

    # pts_closed לציור
    pts_closed = pts + [pts[0]]
    pts_u = pts  # ללא כפולה
    n = len(pts_u)

    # -------------------------------------------------
    # 6. ציור
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(*zip(*pts_closed), "-o", lw=2)
    ax.set_aspect("equal")
    ax.axis("off")

    # אלכסונים
    if sides >= 4:
        for i, j, _ in all_diagonals(pts_u):
            p1, p2 = pts_u[i - 1], pts_u[j - 1]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    "--", lw=1, color="gray", alpha=0.6)

    # תוויות צלע
    for i in range(sides):
        p1, p2 = pts_closed[i], pts_closed[i + 1]
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my, f"{lengths[i]:.2f}",
                fontsize=9, color="blue",
                ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7))

    # תוויות זווית
    for i in range(n):
        prev, curr, nxt = pts_u[i - 1], pts_u[i], pts_u[(i + 1) % n]
        ang = compute_internal_angle(prev, curr, nxt)
        bis = (np.array(prev) - np.array(curr)) + (np.array(nxt) - np.array(curr))
        bis /= np.linalg.norm(bis)
        ax.text(curr[0] + bis[0] * 0.1 * min(lengths),
                curr[1] + bis[1] * 0.1 * min(lengths),
                f"{ang:.1f}°",
                fontsize=9, color="green",
                ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7))

    return fig, lengths, all_diagonals(pts_u)


# ----------   UI   ---------- #
st.set_page_config(page_title="🎯 מצולעים – סגירה אוטומטית", layout="centered")
st.title("🎯 שרטוט מצולעים – כולל סגירה אוטומטית ואלכסונים")

sides = st.number_input("מספר צלעות", 3, 12, 4, 1)

length_inputs = [st.text_input(f"צלע {i + 1}") for i in range(sides)]
lengths = [None if not s.strip() else float(s) for s in length_inputs]

use_angles = st.checkbox("הזן זוויות פנימיות")
angles = None
if use_angles:
    a_inp = [st.text_input(f"זווית {i + 1}") for i in range(sides)]
    if "" in a_inp:
        st.error("יש למלא את כל הזוויות.")
        st.stop()
    angles = [float(a) for a in a_inp]

auto_fix = st.checkbox("סגירה אוטומטית (התאם צלע אחרונה)")

if st.button("✏️ שרטוט"):
    fig, final_lengths, diag = draw_polygon(sides, lengths, angles, auto_fix)
    if fig:
        st.pyplot(fig)

        st.markdown("### אורכי צלעות")
        for i, L in enumerate(final_lengths, 1):
            st.write(f"צלע {i}: {L:.2f}")

        if diag:
            st.markdown("### אורכי אלכסונים")
            for i, j, L in diag:
                st.write(f"אלכסון {i}–{j}: {L:.2f}")