import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

TOL = 1e-6

def compute_internal_angle(p_prev, p_curr, p_next):
    v1 = np.array(p_prev) - np.array(p_curr)
    v2 = np.array(p_next) - np.array(p_curr)
    cos_t = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
    cos_t = np.clip(cos_t, -1, 1)
    return np.degrees(np.arccos(cos_t))

def draw_triangle(lengths):
    L1, L2, L3 = lengths
    # נקודות A, B
    A = (0.0, 0.0)
    B = (L1, 0.0)
    # חוק הקוסינוסים כדי למצוא C
    x = (L1**2 + L2**2 - L3**2) / (2 * L1)
    y2 = L2**2 - x**2
    if y2 < -TOL:
        st.error("לא ניתן לבנות משולש עם אורכים אלה.")
        return None, None
    y = np.sqrt(max(y2, 0.0))
    C = (x, y)

    pts = [A, B, C]
    fig, ax = plt.subplots(figsize=(5,5))
    xs, ys = zip(*(pts + [pts[0]]))
    ax.plot(xs, ys, '-o')
    ax.set_aspect('equal')
    ax.axis('off')

    # סימון אורכים
    for i, (p1, p2) in enumerate(zip(pts, pts[1:]+[pts[0]])):
        mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
        ax.text(mx, my, f"{lengths[i]:.2f}", color='blue')

    # סימון זוויות
    for i, curr in enumerate(pts):
        prev = pts[i-1]
        nxt = pts[(i+1)%3]
        ang = compute_internal_angle(prev, curr, nxt)
        # וקטור ביסקטור קטן
        v1 = (np.array(prev)-np.array(curr))
        v2 = (np.array(nxt)-np.array(curr))
        bis = v1/np.linalg.norm(v1) + v2/np.linalg.norm(v2)
        bis = bis/np.linalg.norm(bis)*0.1
        ax.text(curr[0]+bis[0], curr[1]+bis[1], f"{ang:.1f}°",
                color='green', ha='center', va='center', fontsize=8)

    return fig, lengths

def draw_polygon(sides, lengths, int_angles):
    # משולש מיוחד
    if sides == 3 and all(L is not None for L in lengths) and int_angles is None:
        return draw_triangle(lengths)

    # חישוב זוויות חיצוניות וכיוונים
    ext = [180 - a for a in int_angles]
    headings = np.cumsum([0] + ext[:-1])

    # וקטורי צלעות
    vecs = []
    missing = [i for i,L in enumerate(lengths) if L is None]
    for i,(hd,L) in enumerate(zip(headings, lengths)):
        th = np.radians(hd)
        vecs.append(None if L is None else (L*np.cos(th), L*np.sin(th)))

    # חישוב צלע חסרה
    if missing:
        if len(missing)>1:
            st.error("אפשר להשאיר רק צלע אחת ריקה.")
            return None, None
        dx = sum(v[0] for v in vecs if v)
        dy = sum(v[1] for v in vecs if v)
        i = missing[0]
        L = np.hypot(-dx, -dy)
        lengths[i] = L
        vecs[i] = (L*np.cos(np.radians(headings[i])),
                   L*np.sin(np.radians(headings[i])))

    # בניית נקודות
    pts = [(0.,0.)]
    for dx,dy in vecs:
        x,y = pts[-1]
        pts.append((x+dx, y+dy))
    pts = pts[:-1]

    # ציור
    xs, ys = zip(*(pts + [pts[0]]))
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(xs, ys, '-o')
    ax.set_aspect('equal')
    ax.axis('off')

    # סימון אורכים וזוויות
    min_l = min(lengths)
    for i in range(sides):
        p1, p2 = pts[i], pts[(i+1)%sides]
        mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
        ax.text(mx, my, f"{lengths[i]:.2f}", color='blue')

        prev, curr, nxt = pts[i-1], pts[i], pts[(i+1)%sides]
        ang = compute_internal_angle(prev, curr, nxt)
        v1 = np.array(prev)-np.array(curr)
        v2 = np.array(nxt)-np.array(curr)
        bis = v1/np.linalg.norm(v1) + v2/np.linalg.norm(v2)
        bis = bis/np.linalg.norm(bis) * (0.1 * min_l)
        ax.text(curr[0]+bis[0], curr[1]+bis[1], f"{ang:.1f}°",
                color='green', ha='center', va='center', fontsize=8)

    return fig, lengths

# --- GUI Streamlit ---
st.title("🎯 שרטוט מצולעים מתוקן")

sides = st.number_input("מספר צלעות", 3, 12, 3, 1)

st.write("**קלט אורכי צלעות** (השאר ריק לצלע אחת חישובית)")
lengths = []
for i in range(sides):
    val = st.text_input(f"צלע {i+1}", key=f"len{i}")
    lengths.append(None if val.strip()=="" else float(val))

use_angles = st.checkbox("להזין זוויות פנימיות")
int_angles = None
if use_angles:
    st.write("**קלט זוויות פנימיות** (סכום: 180×(n-2))")
    tmp = []
    for i in range(sides):
        val = st.text_input(f"זווית {i+1}", key=f"ang{i}")
        tmp.append(None if val.strip()=="" else float(val))
    int_angles = tmp

if st.button("✏️ שרטוט"):
    # בדיקות
    if sum(1 for L in lengths if L is None) > 1:
        st.error("ניתן להשאיר רק צלע אחת ריקה.")
    elif use_angles:
        if None in int_angles:
            st.error("חסרה זווית פנימית.")
        elif abs(sum(int_angles)-180*(sides-2))>TOL:
            st.error(f"סכום הזוויות צריך להיות {180*(sides-2):.0f}°, הוזן {sum(int_angles):.1f}°.")
        else:
            fig, final = draw_polygon(sides, lengths, int_angles)
    else:
        fig, final = draw_polygon(sides, lengths, None)

    if fig:
        st.pyplot(fig)
        st.write("**אורכי צלעות סופיים:**")
        for i,L in enumerate(final,1):
            st.write(f"צלע {i}: {L:.2f}")

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        st.download_button("הורד PNG", buf.getvalue(), "polygon.png")
        buf2 = BytesIO()
        fig.savefig(buf2, format="pdf", bbox_inches='tight')
        st.download_button("הורד PDF", buf2.getvalue(), "polygon.pdf")