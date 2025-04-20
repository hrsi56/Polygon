import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

def solve_quadrilateral(sides, angles):
    missing_sides = [i for i, s in enumerate(sides) if s is None]
    missing_angles = [i for i, a in enumerate(angles) if a is None]

    if len(missing_sides) + len(missing_angles) > 1:
        return None, "🔴 חסרים יותר מדי נתונים – יש להשלים צלע או זווית אחת לפחות."

    if len(missing_angles) == 1:
        known_sum = sum([a for a in angles if a is not None])
        missing_idx = missing_angles[0]
        angles[missing_idx] = 360 - known_sum
        if angles[missing_idx] <= 0 or angles[missing_idx] >= 180:
            return None, "🔴 הזווית שחושבה אינה תקפה – בדוק את הקלט."

    if len(missing_sides) == 1:
        points = [(0, 0)]
        angle = 0
        for i in range(4):
            if sides[i] is None:
                break
            dx = sides[i] * np.cos(np.radians(angle))
            dy = sides[i] * np.sin(np.radians(angle))
            new_point = (points[-1][0] + dx, points[-1][1] + dy)
            points.append(new_point)
            angle += 180 - angles[i]

        if len(points) == 4:
            end_vector = np.array(points[0]) - np.array(points[-1])
            missing_length = np.linalg.norm(end_vector)
            sides[missing_sides[0]] = missing_length
        else:
            return None, "🔴 לא הצלחנו לבנות מספיק נקודות לצורך השלמה."

    return (sides, angles), "✅ המצולע הושלם בהצלחה עם פתרון יחיד."

def draw_quadrilateral(sides, angles):
    points = [(0, 0)]
    angle = 0
    for i in range(4):
        dx = sides[i] * np.cos(np.radians(angle))
        dy = sides[i] * np.sin(np.radians(angle))
        new_point = (points[-1][0] + dx, points[-1][1] + dy)
        points.append(new_point)
        angle += 180 - angles[i]

    fig, ax = plt.subplots()
    xs, ys = zip(*points)
    ax.plot(xs, ys, 'b-')
    ax.set_aspect('equal')
    ax.axis('off')

    for i in range(4):
        x = (points[i][0] + points[i + 1][0]) / 2
        y = (points[i][1] + points[i + 1][1]) / 2
        ax.text(x, y, f'{sides[i]:.1f}', fontsize=10, color='blue')

        p_prev = points[i - 1]
        p_curr = points[i]
        p_next = points[i + 1]

        v1 = np.array([p_prev[0] - p_curr[0], p_prev[1] - p_curr[1]])
        v2 = np.array([p_next[0] - p_curr[0], p_next[1] - p_curr[1]])
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        bisector = v1 + v2
        bisector = bisector / np.linalg.norm(bisector)

        offset = 0.2 * min(sides)
        angle_x = p_curr[0] + bisector[0] * offset
        angle_y = p_curr[1] + bisector[1] * offset

        cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
        theta_rad = np.arccos(cos_theta)
        internal_angle = round(180 - np.degrees(theta_rad), 1)
        ax.text(angle_x, angle_y, f'{internal_angle}°', fontsize=10, color='green', ha='center')

    return fig

st.title("📐 מצייר מרובעים עם פתרון חכם")

st.markdown("הזן את אורכי הצלעות והזוויות (במעלות). אפשר להשאיר **אחת** מהן ריקה:")

sides = []
angles = []
for i in range(4):
    side = st.text_input(f"צלע {i + 1}", key=f"s{i}")
    sides.append(float(side) if side else None)
for i in range(4):
    angle = st.text_input(f"זווית {i + 1} (°)", key=f"a{i}")
    angles.append(float(angle) if angle else None)

if st.button("✏️ פתר והצג מצולע"):
    result, msg = solve_quadrilateral(sides.copy(), angles.copy())
    st.markdown(f"**{msg}**")
    if result:
        fig = draw_quadrilateral(result[0], result[1])
        st.pyplot(fig)

        # הורדה
        buf_png = BytesIO()
        fig.savefig(buf_png, format="png", dpi=300, bbox_inches='tight')
        st.download_button("📥 הורד כ-PNG", data=buf_png.getvalue(), file_name="quadrilateral.png", mime="image/png")

        buf_pdf = BytesIO()
        fig.savefig(buf_pdf, format="pdf", bbox_inches='tight')
        st.download_button("📄 הורד כ-PDF", data=buf_pdf.getvalue(), file_name="quadrilateral.pdf", mime="application/pdf")
