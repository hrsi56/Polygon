import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

TOL = 1e-6

def compute_internal_angle(p_prev, p_curr, p_next):
    # מחזיר זווית פנימית במעלות
    v1 = np.array(p_prev) - np.array(p_curr)
    v2 = np.array(p_next) - np.array(p_curr)
    cos_t = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
    cos_t = np.clip(cos_t, -1, 1)
    return np.degrees(np.arccos(cos_t))

def draw_polygon(sides, lengths, int_angles):
    # חישוב זוויות חיצוניות וכיוונים
    ext_angles = [180 - a for a in int_angles]
    headings = np.cumsum([0] + ext_angles[:-1])

    # בונים וקטורי צלעות, מחשבים צלע חסרה אם יש
    vecs = []
    missing = [i for i,L in enumerate(lengths) if L is None]
    for i,(hd,L) in enumerate(zip(headings, lengths)):
        th = np.radians(hd)
        if L is None:
            vecs.append(None)
        else:
            vecs.append((L*np.cos(th), L*np.sin(th)))

    if missing:
        if len(missing) > 1:
            st.error("שגיאה: אפשר להשאיר רק צלע אחת ריקה לחישוב אוטומטי.")
            return None, None
        # מחסירים את כל הווקטורים הידועים
        dx = sum(v[0] for v in vecs if v)
        dy = sum(v[1] for v in vecs if v)
        i = missing[0]
        L = np.hypot(-dx, -dy)
        lengths[i] = L
        vecs[i] = (L*np.cos(np.radians(headings[i])),
                   L*np.sin(np.radians(headings[i])))

    # בונים רשימת נקודות
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
    for i in range(sides):
        x1,y1 = pts[i]
        x2,y2 = pts[(i+1)%sides]
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my, f"{lengths[i]:.2f}", color='blue')

        # זווית פנימית
        prev, curr, nxt = pts[i-1], pts[i], pts[(i+1)%sides]
        ang = compute_internal_angle(prev,curr,nxt)
        # שרטוט הטקסט מעט פנימה
        bis = (np.array(prev)-np.array(curr))/np.linalg.norm(np.array(prev)-np.array(curr)) + \
              (np.array(nxt)-np.array(curr))/np.linalg.norm(np.array(nxt)-np.array(curr))
        bis = bis/np.linalg.norm(bis) * 0.1
        ax.text(curr[0]+bis[0], curr[1]+bis[1], f"{ang:.1f}°", color='green',
                ha='center', va='center', fontsize=8)

    return fig, lengths

# --- GUI ---
st.title("🎯 שרטוט מצולעים פשוט")

sides = st.number_input("מספר צלעות", min_value=3, max_value=12, value=4, step=1)

st.write("**קלט אורכי צלעות** – השאר ריק לצלע שתרצה לחשב אוטומטית")
lengths = []
for i in range(sides):
    val = st.text_input(f"צלע {i+1}", key=f"len{i}")
    lengths.append(None if val.strip()=="" else float(val))

use_angles = st.checkbox("להזין זוויות פנימיות ידנית")
int_angles = None
if use_angles:
    st.write("**קלט זוויות פנימיות** (בסכום של 180×(n-2)°)")
    tmp = []
    for i in range(sides):
        val = st.text_input(f"זווית פנימית {i+1}", key=f"ang{i}")
        tmp.append(None if val.strip()=="" else float(val))
    int_angles = tmp

if st.button("✏️ שרטוט"):
    # בדיקות בסיסיות
    if sum(1 for L in lengths if L is None) > 1:
        st.error("אפשר להשאיר ריק רק צלע אחת.")
    elif use_angles:
        if None in int_angles:
            st.error("חסרה זווית פנימית אחת לפחות.")
        elif abs(sum(int_angles) - 180*(sides-2)) > TOL:
            st.error(f"סכום הזוויות חייב להיות {180*(sides-2):.0f}°, כרגע {sum(int_angles):.1f}°.")
        else:
            fig, final_lengths = draw_polygon(sides, lengths, int_angles)
    else:
        # זוויות שוות
        equal = [180*(sides-2)/sides]*sides
        fig, final_lengths = draw_polygon(sides, lengths, equal)

    if fig:
        st.pyplot(fig)
        st.write("**אורכי צלעות סופיים:**")
        for i,L in enumerate(final_lengths, start=1):
            st.write(f"צלע {i}: {L:.2f}")

        # כפתורי הורדה
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        st.download_button("הורד כ-PNG", buf.getvalue(), file_name="polygon.png")
        buf_pdf = BytesIO()
        fig.savefig(buf_pdf, format="pdf", bbox_inches='tight')
        st.download_button("הורד כ-PDF", buf_pdf.getvalue(), file_name="polygon.pdf")