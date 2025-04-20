import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

TOLERANCE = 1e-2  # סף קטן של 0.01°

def compute_internal_angle(p_prev, p_curr, p_next):
    """
    מחשבת את הזווית הפנימית (במעלות) בנקודה p_curr,
    הנתונה שלוש נקודות עוקבות במצולע.
    """
    v1 = np.array(p_prev) - np.array(p_curr)
    v2 = np.array(p_next) - np.array(p_curr)

    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    return round(angle_deg, 1)


def draw_polygon(sides, lengths, custom_angles=None):
    total_angle_sum = 180 * (sides - 2)

    # 1. הכנת רשימת הזוויות (מותאמות או שוות)
    if custom_angles is not None:
        angles_list = custom_angles.copy()
        sum_angles = sum(angles_list)
        if abs(sum_angles - total_angle_sum) > TOLERANCE:
            factor = total_angle_sum / sum_angles
            angles_list = [a * factor for a in angles_list]
            correction_message = (
                f"בוצע תיקון קל לזוויות כדי לסגור מצולע "
                f"({sum_angles:.2f}° → {total_angle_sum:.2f}°)"
            )
        else:
            correction_message = None
    else:
        angle_value = total_angle_sum / sides
        angles_list = [angle_value] * sides
        correction_message = None

    # 2. בניית הווקטורים ושערוך צלע חסרה במידת הצורך
    angle_heading = 0.0
    points = [(0.0, 0.0)]
    vectors = []

    for i in range(sides):
        L = lengths[i]
        # אם אורך צלע חסר, נדלג על הוספת נקודה ונרשום None בווקטורים
        if L is None:
            vectors.append(None)
            angle_heading += 180 - angles_list[i]
            continue

        dx = L * np.cos(np.radians(angle_heading))
        dy = L * np.sin(np.radians(angle_heading))
        vectors.append((dx, dy))
        points.append((points[-1][0] + dx, points[-1][1] + dy))
        angle_heading += 180 - angles_list[i]

    # חשב צלע חסרה אם הייתה כזו
    if None in lengths:
        idx = lengths.index(None)
        end_pt = points[-1]
        missing_len = np.hypot(-end_pt[0], -end_pt[1])
        lengths[idx] = missing_len
        correction_message = (
            f"בוצע חישוב לצלע החסרה (צלע {idx + 1}): {missing_len:.2f}"
        )
    else:
        # בדוק סגירות מצולע ותיקן אם יש סטיה
        total_dx = sum(v[0] for v in vectors if v is not None)
        total_dy = sum(v[1] for v in vectors if v is not None)
        shift = np.hypot(total_dx, total_dy)
        if shift > TOLERANCE:
            # נוסיף את התיקון לצלע הארוכה ביותר
            corr_vec_len = shift
            long_idx = int(np.argmax(lengths))
            old = lengths[long_idx]
            lengths[long_idx] += corr_vec_len
            correction_message = (
                f"בוצע תיקון קל בצלע {long_idx + 1} כדי לסגור את המצולע "
                f"({old:.2f} → {lengths[long_idx]:.2f})"
            )

    # 3. בניה מחדש של נקודות למיקום מדוייק
    angle_heading = 0.0
    points = [(0.0, 0.0)]
    for i in range(sides):
        dx = lengths[i] * np.cos(np.radians(angle_heading))
        dy = lengths[i] * np.sin(np.radians(angle_heading))
        points.append((points[-1][0] + dx, points[-1][1] + dy))
        angle_heading += 180 - angles_list[i]

    # 4. ציור המתאר
    xs, ys = zip(*points)
    fig, ax = plt.subplots()
    ax.plot(xs, ys, 'b-')
    ax.set_aspect('equal')
    ax.axis('off')

    # 5. סימון אורכים וזוויות
    for i in range(sides):
        # סימון אורך אמצעי
        x_mid = (points[i][0] + points[i+1][0]) / 2
        y_mid = (points[i][1] + points[i+1][1]) / 2
        ax.text(x_mid, y_mid, f'{lengths[i]:.2f}', fontsize=10, color='blue')

        # סימון זווית פנימית
        p_prev = points[i-1]
        p_curr = points[i]
        p_next = points[i+1]
        ang = compute_internal_angle(p_prev, p_curr, p_next)

        # חישוב כיוון להצגת הטקסט
        v1 = (np.array(p_prev) - np.array(p_curr))
        v2 = (np.array(p_next) - np.array(p_curr))
        bis = v1/np.linalg.norm(v1) + v2/np.linalg.norm(v2)
        bis /= np.linalg.norm(bis)
        offset = 0.2 * min(lengths)
        tx = p_curr[0] + bis[0] * offset
        ty = p_curr[1] + bis[1] * offset

        ax.text(tx, ty, f'{ang:.1f}°', fontsize=10,
                color='green', ha='center', va='center')

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
            angles.append(float(val))
        except ValueError:
            st.error("יש להזין ערך מספרי לכל זווית")

if st.button("✏️ צייר מצולע"):
    if empty_count > 1:
        st.error("אפשר להשאיר ריק רק שדה אחד")
    elif use_custom_angles and len(angles) != sides:
        st.error("יש להזין את כל הזוויות")
    else:
        fig, final_lengths, msg = draw_polygon(
            sides,
            lengths,
            custom_angles=angles if use_custom_angles else None
        )
        st.pyplot(fig)

        st.info("📐 אורכי הצלעות:")
        for i, L in enumerate(final_lengths, start=1):
            st.write(f"צלע {i}: {L:.2f}")

        if msg:
            st.warning(f"⚠️ {msg}")

        # אפשרות הורדה כ־PNG ו־PDF
        buf_png = BytesIO()
        fig.savefig(buf_png, format="png", dpi=300, bbox_inches='tight')
        st.download_button("📥 הורד PNG", data=buf_png.getvalue(),
                           file_name="polygon.png", mime="image/png")

        buf_pdf = BytesIO()
        fig.savefig(buf_pdf, format="pdf", bbox_inches='tight')
        st.download_button("📄 הורד PDF", data=buf_pdf.getvalue(),
                           file_name="polygon.pdf", mime="application/pdf")