import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

def solve_quadrilateral(sides, angles):
    missing_sides = [i for i, s in enumerate(sides) if s is None]
    missing_angles = [i for i, a in enumerate(angles) if a is None]

    # ניתן להשאיר בדיוק זווית או צלע אחת ריקה
    if len(missing_sides) + len(missing_angles) > 1:
        return None, "🔴 חסרים יותר מדי נתונים – יש להשלים בדיוק צלע או זווית אחת."

    # השלמת זווית חסרה
    if len(missing_angles) == 1:
        idx = missing_angles[0]
        known_sum = sum(a for a in angles if a is not None)
        angles[idx] = 360 - known_sum
        # בדיקת תקינות זווית (למרובע קמור)
        if angles[idx] <= 0 or angles[idx] >= 180:
            return None, "🔴 הזווית שחושבה אינה תקפה – בדוק את הקלט."

    # השלמת צלע חסרה
    if len(missing_sides) == 1:
        idx = missing_sides[0]
        # סיכום וקטורים של שלוש הצלעות הידועות
        vec_sum = np.array([0.0, 0.0])
        angle_acc = 0.0
        for i in range(4):
            if sides[i] is not None:
                dx = sides[i] * np.cos(np.radians(angle_acc))
                dy = sides[i] * np.sin(np.radians(angle_acc))
                vec_sum += np.array([dx, dy])
            angle_acc += 180 - angles[i]

        missing_length = np.linalg.norm(vec_sum)
        if missing_length <= 1e-6:
            return None, "🔴 חישוב אורך הצלע החסרה נכשל – בדוק את הקלט."
        sides[idx] = missing_length

    return (sides, angles), "✅ המצולע הושלם בהצלחה עם פתרון יחיד."


def draw_quadrilateral(sides, angles):
    points = [(0.0, 0.0)]
    angle_acc = 0.0
    for i in range(4):
        dx = sides[i] * np.cos(np.radians(angle_acc))
        dy = sides[i] * np.sin(np.radians(angle_acc))
        points.append((points[-1][0] + dx, points[-1][1] + dy))
        angle_acc += 180 - angles[i]

    fig, ax = plt.subplots()
    xs, ys = zip(*points)
    ax.plot(xs, ys, 'b-')
    ax.set_aspect('equal')
    ax.axis('off')

    # סימון אורכי צלעות
    for i in range(4):
        mid_x = (points[i][0] + points[i+1][0]) / 2
        mid_y = (points[i][1] + points[i+1][1]) / 2
        ax.text(mid_x, mid_y, f'{sides[i]:.2f}', fontsize=10, color='blue')

    # סימון הזוויות מתוך הקלט
    for i in range(4):
        p_prev = points[i-1]
        p_curr = points[i]
        p_next = points[i+1]
        # כיוון חיצוני לציר
        v1 = np.array(p_prev) - np.array(p_curr)
        v2 = np.array(p_next) - np.array(p_curr)
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        # פותק הזווית (לא חובה לצייר קו)
        bis = v1 + v2
        bis /= np.linalg.norm(bis)
        offset = 0.1 * min(sides)
        lbl_x = p_curr[0] + bis[0] * offset
        lbl_y = p_curr[1] + bis[1] * offset
        ax.text(lbl_x, lbl_y, f'{angles[i]:.1f}°', fontsize=10, color='green', ha='center')

    return fig

# ממשק Streamlit
st.title("📐 מצייר מרובעים עם פתרון חכם")
st.markdown("הזן את אורכי הצלעות והזוויות (במעלות). אפשר להשאיר **אחת** מהן ריקה:")

# קריאת קלט
sides = [None]*4
angles = [None]*4
for i in range(4):
    inp = st.text_input(f"צלע {i+1}", key=f"s{i}")
    sides[i] = float(inp) if inp else None
for i in range(4):
    inp = st.text_input(f"זווית {i+1} (°)", key=f"a{i}")
    angles[i] = float(inp) if inp else None

if st.button("✏️ פתר והצג מצולע"):
    result, msg = solve_quadrilateral(sides.copy(), angles.copy())
    st.markdown(f"**{msg}**")
    if result:
        fig = draw_quadrilateral(*result)
        st.pyplot(fig)

        # הורדות
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        st.download_button("📥 הורד PNG", buf.getvalue(), file_name="quadrilateral.png", mime="image/png")
        buf_pdf = BytesIO()
        fig.savefig(buf_pdf, format='pdf', bbox_inches='tight')
        st.download_button("📄 הורד PDF", buf_pdf.getvalue(), file_name="quadrilateral.pdf", mime="application/pdf")
