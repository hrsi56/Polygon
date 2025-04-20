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
        x = (points[i][0] + points[i + 1][0]) / 2
        y = (points[i][1] + points[i + 1][1]) / 2
        ax.text(x, y, f'{lengths[i]:.2f}', fontsize=10, color='blue')
        x0, y0 = points[i]
        ax.text(x0, y0, f'{internal_angle:.1f}°', fontsize=10, color='green')

    return fig

st.title("מצייר מצולעים")

sides = st.number_input("כמה צלעות?", min_value=3, max_value=12, value=4)

lengths = []
for i in range(sides):
    length = st.number_input(f"אורך צלע {i+1}", min_value=1.0, value=100.0)
    lengths.append(length)

if st.button("צייר מצולע"):
    fig = draw_polygon(sides, lengths)

    # הצגה בדף
    st.pyplot(fig)

    # הורדה כקובץ
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    st.download_button(
        label="הורד כקובץ PNG",
        data=buf.getvalue(),
        file_name="polygon_output.png",
        mime="image/png"
    )