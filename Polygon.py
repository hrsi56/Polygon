import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

TOLERANCE = 1e-2  # טולרנס לפער קטן מאוד (0.01)

def draw_polygon(sides, lengths):
    internal_angle = (sides - 2) * 180 / sides
    angle = 0
    points = [(0, 0)]
    vectors = []

    missing_index = None
    if None in lengths:
        missing_index = lengths.index(None)

    # בניית וקטורים
    for i in range(sides):
        if i == missing_index:
            vectors.append(None)
            angle -= 180 - internal_angle
            continue
        dx = lengths[i] * np.cos(np.radians(angle))
        dy = lengths[i] * np.sin(np.radians(angle))
        vectors.append((dx, dy))
        new_point = (points[-1][0] + dx, points[-1][1] + dy)
        points.append(new_point)
        angle -= 180 - internal_angle

    # חישוב נקודת סיום
    if missing_index is not None:
        end_point = points[-1]
        back_vector = (-end_point[0], -end_point[1])
        length_missing = np.hypot(*back_vector)
        lengths[missing_index] = length_missing
        correction_message = f"בוצע חישוב לצלע החסרה (צלע {missing_index + 1}): {length_missing:.2f}"
    else:
        # בדיקה האם המצולע סגור
        total_dx = sum(l[0] for l in vectors if l is not None)
        total_dy = sum(l[1] for l in vectors if l is not None)
        total_shift = np.hypot(total_dx, total_dy)

        if total_shift > TOLERANCE:
            # תיקון קל בצלע הארוכה ביותר
            correction_vector = (-total_dx, -total_dy)
            correction_length = np.hypot(*correction_vector)

            longest_index = np.argmax(lengths)
            old_length = lengths[longest_index]
            new_length = old_length + correction_length

            lengths[longest_index] = new_length
            correction_message = f"בוצע תיקון קל בצלע {longest_index + 1} כדי לסגור את המצולע ({old_length:.2f} → {new_length:.2f})"
        else:
            correction_message = None

    # בנייה מחדש של הנקודות
    angle = 0
    points = [(0, 0)]
    for i in range(sides):
        dx = lengths[i] * np.cos(np.radians(angle))
        dy = lengths[i] * np.sin(np.radians(angle))
        new_point = (points[-1][0] + dx, points[-1][1] + dy)
        points.append(new_point)
        angle -= 180 - internal_angle

    # ציור
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

        # מרכז זווית פנימית
        p_prev = points[i - 1]
        p_curr = points[i]
        p_next = points[i + 1]

        v1 = np.array([p_prev[0] - p_curr[0], p_prev[1] - p_curr[1]])
        v2 = np.array([p_next[0] - p_curr[0], p_next[1] - p_curr[1]])

        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        bisector = v1 + v2
        bisector = bisector / np.linalg.norm(bisector)

        offset = 0.2 * min(lengths)
        angle_x = p_curr[0] + bisector[0] * offset
        angle_y = p_curr[1] + bisector[1] * offset

        ax.text(angle_x, angle_y, f'{internal_angle:.1f}°', fontsize=10, color='green', ha='center', va='center')

    return fig, lengths, correction_message

# --- Streamlit UI ---
st.title("🎯 מצייר מצולעים חכמים עם תיקון אוטומטי")

sides = st.number_input("🔺 כמה צלעות?", min_value=3, max_value=12, value=5)

lengths = []
empty_count = 0

for i in range(sides):
    val = st.text_input(f"אורך צלע {i + 1} (אפשר להשאיר אחד ריק)", value="", key=f"len_{i}")
    if val.strip() == "":
        lengths.append(None)
        empty_count += 1
    else:
        try:
            num = float(val)
            lengths.append(num)
        except ValueError:
            st.error("יש להזין מספר או להשאיר ריק")

if st.button("✏️ צייר מצולע"):
    if empty_count > 1:
        st.error("אפשר להשאיר ריק רק שדה אחד!")
    else:
        fig, final_lengths, msg = draw_polygon(sides, lengths)
        st.pyplot(fig)

        st.info("✏️ אורכי הצלעות הסופיים:")
        for idx, l in enumerate(final_lengths, 1):
            st.write(f"צלע {idx}: {l:.2f}")

        if msg:
            st.warning(f"⚠️ {msg}")

        # הורדה כ-PNG
        buf_png = BytesIO()
        fig.savefig(buf_png, format="png", dpi=300, bbox_inches='tight')
        st.download_button("📥 הורד PNG", data=buf_png.getvalue(), file_name="polygon.png", mime="image/png")

        # הורדה כ-PDF
        buf_pdf = BytesIO()
        fig.savefig(buf_pdf, format="pdf", bbox_inches='tight')
        st.download_button("📄 הורד PDF", data=buf_pdf.getvalue(), file_name="polygon.pdf", mime="application/pdf")