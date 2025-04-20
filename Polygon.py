import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

def draw_polygon(sides, lengths):
    internal_angle = (sides - 2) * 180 / sides
    points = [(0, 0)]
    angle = 0

    for i in range(sides):
        dx = lengths[i] * np.cos(np.radians(angle))
        dy = lengths[i] * np.sin(np.radians(angle))
        new_point = (points[-1][0] + dx, points[-1][1] + dy)
        points.append(new_point)
        angle -= 180 - internal_angle

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

        # מיקום פנימי לזווית
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

    return fig

# --- Streamlit UI ---
st.title("🎨 מצייר מצולעים לינקו")

sides = st.number_input("🔺 כמה צלעות?", min_value=3, max_value=12, value=4)

lengths = []
for i in range(sides):
    length = st.number_input(f"אורך צלע {i + 1}", min_value=1.0, value=100.0, key=f"length_{i}")
    lengths.append(length)

if st.button("צייר מצולע"):
    fig = draw_polygon(sides, lengths)
    st.pyplot(fig)

    # שמירה כ-PNG
    png_buf = BytesIO()
    fig.savefig(png_buf, format="png", dpi=300, bbox_inches='tight')
    st.download_button(
        label="📥 הורד כ-PNG",
        data=png_buf.getvalue(),
        file_name="polygon_output.png",
        mime="image/png"
    )

    # שמירה כ-PDF
    pdf_buf = BytesIO()
    fig.savefig(pdf_buf, format="pdf", bbox_inches='tight')
    st.download_button(
        label="📄 הורד כ-PDF",
        data=pdf_buf.getvalue(),
        file_name="polygon_output.pdf",
        mime="application/pdf"
    )