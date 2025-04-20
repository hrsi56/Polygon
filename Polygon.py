import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

TOLERANCE = 1e-2

def compute_internal_angle(p_prev, p_curr, p_next):
    v1 = np.array([p_prev[0] - p_curr[0], p_prev[1] - p_curr[1]])
    v2 = np.array([p_next[0] - p_curr[0], p_next[1] - p_curr[1]])

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    return round(180 - angle_deg, 1)

def draw_polygon(sides, lengths, custom_angles=None):
    total_angle_sum = 180 * (sides - 2)

    # השתמש בזוויות מותאמות אם ניתנו, אחרת זוויות שוות
    if custom_angles:
        angles = custom_angles
        sum_angles = sum(custom_angles)
        if abs(sum_angles - total_angle_sum) > 1:
            correction_factor = total_angle_sum / sum_angles
            angles = [a * correction_factor for a in custom_angles]
            correction_message = f"בוצע תיקון קל לזוויות כדי לסגור מצולע ({sum_angles:.1f}° → {total_angle_sum:.1f}°)"
        else:
            correction_message = None
    else:
        angle_value = total_angle_sum / sides
        angles = [angle_value] * sides
        correction_message = None

    # בניית נקודות לפי אורכי צלעות וזוויות פנימיות
    angle = 0
    points = [(0, 0)]
    missing_index = None
    vectors = []

    for i in range(sides):
        if lengths[i] is None:
            vectors.append(None)
            angle += 180 - angles[i]
            continue
        dx = lengths[i] * np.cos(np.radians(angle))
        dy = lengths[i] * np.sin(np.radians(angle))
        vectors.append((dx, dy))
        points.append((points[-1][0] + dx, points[-1][1] + dy))
        angle += 180 - angles[i]

    if None in lengths:
        missing_index = lengths.index(None)
        end_point = points[-1]
        back_vector = (-end_point[0], -end_point[1])
        length_missing = np.hypot(*back_vector)
        lengths[missing_index] = length_missing
        correction_message = f"בוצע חישוב לצלע החסרה (צלע {missing_index + 1}): {length_missing:.2f}"
    else:
        total_dx = sum(v[0] for v in vectors if v)
        total_dy = sum(v[1] for v in vectors if v)
        total_shift = np.hypot(total_dx, total_dy)
        if total_shift > TOLERANCE:
            correction_vector = (-total_dx, -total_dy)
            correction_length = np.hypot(*correction_vector)
            longest_index = np.argmax(lengths)
            old_length = lengths[longest_index]
            lengths[longest_index] = old_length + correction_length
            correction_message = f"בוצע תיקון קל בצלע {longest_index + 1} כדי לסגור את המצולע ({old_length:.2f} → {lengths[longest_index]:.2f})"

    # בנייה מחדש עם נקודות מדויקות
    angle = 0
    points = [(0, 0)]
    for i in range(sides):
        dx = lengths[i] * np.cos(np.radians(angle))
        dy = lengths[i] * np.sin(np.radians(angle))
        new_point = (points[-1][0] + dx, points[-1][1] + dy)
        points.append(new_point)
        angle += 180 - angles[i]

    xs, ys = zip(*points)
    fig, ax = plt.subplots()
    ax.plot(xs, ys, 'b-')
    ax.set_aspect('equal')
    ax.axis('off')

    for i in range(sides):
        # אמצע צלע
        x = (points[i][0] + points[i + 1][0]) / 2
        y = (points[i][1] + points[i + 1][1]) / 2
        ax.text(x, y, f'{lengths[i]:.2f}', fontsize=10, color='blue')

        # זווית גאומטרית אמיתית
        p_prev = points[i - 1]
        p_curr = points[i]
        p_next = points[i + 1]
        angle_val = compute_internal_angle(p_prev, p_curr, p_next)

        v1 = np.array([p_prev[0] - p_curr[0], p_prev[1] - p_curr[1]])
        v2 = np.array([p_next[0] - p_curr[0], p_next[1] - p_curr[1]])
        bisector = v1 / np.linalg.norm(v1) + v2 / np.linalg.norm(v2)
        bisector = bisector / np.linalg.norm(bisector)

        offset = 0.2 * min(lengths)
        text_x = p_curr[0] + bisector[0] * offset
        text_y = p_curr[1] + bisector[1] * offset

        ax.text(text_x, text_y, f'{angle_val:.1f}°', fontsize=10, color='green', ha='center', va='center')

    return fig, lengths, correction_message

# --- Streamlit UI ---
st.title("🎯 אפליקציית מצולעים חכמה עם זוויות מותאמות")

sides = st.number_input("🔺 כמה צלעות?", min_value=3, max_value=12, value=5)

st.subheader("📏 אורכי צלעות")
lengths = []
empty_count = 0
for i in range(sides):
    val = st.text_input(f"צלע {i + 1}", value="", key=f"len_{i}")
    if val.strip() == "":
        lengths.append(None)
        empty_count += 1
    else:
        try:
            lengths.append(float(val))
        except ValueError:
            st.error("יש להזין מספר או להשאיר ריק")

use_custom_angles = st.checkbox("אני רוצה להזין זוויות בעצמי")

angles = []
if use_custom_angles:
    st.subheader("🎛 הזנת זוויות פנימיות (במעלות)")
    for i in range(sides):
        val = st.text_input(f"זווית {i + 1}", value="108.0", key=f"angle_{i}")
        try:
            angle_val = float(val)
            angles.append(angle_val)
        except ValueError:
            st.error("יש להזין ערך מספרי לכל זווית")

if st.button("✏️ צייר מצולע"):
    if empty_count > 1:
        st.error("אפשר להשאיר ריק רק שדה אחד")
    elif use_custom_angles and len(angles) != sides:
        st.error("יש להזין את כל הזוויות")
    else:
        fig, final_lengths, msg = draw_polygon(sides, lengths, angles if use_custom_angles else None)
        st.pyplot(fig)

        st.info("📐 אורכי הצלעות:")
        for i, l in enumerate(final_lengths, 1):
            st.write(f"צלע {i}: {l:.2f}")

        if msg:
            st.warning(f"⚠️ {msg}")

        # הורדה
        buf_png = BytesIO()
        fig.savefig(buf_png, format="png", dpi=300, bbox_inches='tight')
        st.download_button("📥 הורד PNG", data=buf_png.getvalue(), file_name="polygon.png", mime="image/png")

        buf_pdf = BytesIO()
        fig.savefig(buf_pdf, format="pdf", bbox_inches='tight')
        st.download_button("📄 הורד PDF", data=buf_pdf.getvalue(), file_name="polygon.pdf", mime="application/pdf")