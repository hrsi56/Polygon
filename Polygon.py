import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

TOLERANCE = 1e-6

def compute_internal_angle(p_prev, p_curr, p_next):
    """
    מחשבת את הזווית הפנימית (במעלות) בנקודה p_curr,
    על סמך שלוש נקודות עוקבות במצולע.
    """
    v1 = np.array(p_prev) - np.array(p_curr)
    v2 = np.array(p_next) - np.array(p_curr)
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))
    return round(angle, 1)


def draw_polygon(sides, lengths, custom_angles=None):
    """
    בונה ומשרטט מצולע לפי מספר צלעות, אורכים וזוויות פנימיות.
    מחזיר את הפיגורמה של matplotlib, רשימת אורכי הצלעות (כולל תיקונים)
    והודעת תיקון אם בוצע.
    """
    # 1. חישוב זוויות פנימיות
    total_int_sum = 180 * (sides - 2)
    if custom_angles is not None:
        int_angles = custom_angles.copy()
        sum_int = sum(int_angles)
        if abs(sum_int - total_int_sum) > TOLERANCE:
            factor = total_int_sum / sum_int
            int_angles = [a * factor for a in int_angles]
            correction = f"זוויות מותאמות: {sum_int:.1f}° → {total_int_sum:.1f}°"
        else:
            correction = None
    else:
        val = total_int_sum / sides
        int_angles = [val] * sides
        correction = None

    # 2. זוויות חיצוניות וכיווני ציור
    ext_angles = [180 - a for a in int_angles]
    headings = [sum(ext_angles[:i]) for i in range(sides)]

    # 3. וקטורי צלעות
    vectors = []
    missing_idx = None
    for i, L in enumerate(lengths):
        theta = np.radians(headings[i])
        if L is None:
            vectors.append(None)
            missing_idx = i
        else:
            dx = L * np.cos(theta)
            dy = L * np.sin(theta)
            vectors.append((dx, dy))

    # 4. טיפול בצלע חסרה או תיקון סגירה
    sum_dx = sum(v[0] for v in vectors if v is not None)
    sum_dy = sum(v[1] for v in vectors if v is not None)
    if missing_idx is not None:
        missing_len = np.hypot(-sum_dx, -sum_dy)
        lengths[missing_idx] = missing_len
        theta = np.radians(headings[missing_idx])
        vectors[missing_idx] = (missing_len * np.cos(theta), missing_len * np.sin(theta))
        correction = f"חישוב צלע חסרה #{missing_idx+1}: {missing_len:.2f}"
    else:
        if np.hypot(sum_dx, sum_dy) > TOLERANCE:
            idx_long = int(np.argmax(lengths))
            add_len = np.hypot(sum_dx, sum_dy)
            old = lengths[idx_long]
            lengths[idx_long] += add_len
            theta = np.radians(headings[idx_long])
            vectors[idx_long] = (lengths[idx_long] * np.cos(theta), lengths[idx_long] * np.sin(theta))
            correction = f"תיקון צלע #{idx_long+1}: {old:.2f} → {lengths[idx_long]:.2f}"

    # 5. בניית נקודות המצולע
    verts = [(0.0, 0.0)]
    for v in vectors:
        x, y = verts[-1]
        dx, dy = v
        verts.append((x + dx, y + dy))
    poly = verts[:-1]

    # 6. שרטוט
    xs = [p[0] for p in poly] + [poly[0][0]]
    ys = [p[1] for p in poly] + [poly[0][1]]
    fig, ax = plt.subplots()
    ax.plot(xs, ys, '-o')
    ax.set_aspect('equal')
    ax.axis('off')

    # 7. סימון אורכים וזוויות
    min_len = min([l for l in lengths if l is not None])
    for i in range(sides):
        p1 = poly[i]
        p2 = poly[(i + 1) % sides]
        mx = (p1[0] + p2[0]) / 2
        my = (p1[1] + p2[1]) / 2
        ax.text(mx, my, f"{lengths[i]:.2f}", fontsize=10, color='blue')

        prev = poly[(i - 1) % sides]
        curr = poly[i]
        nxt = poly[(i + 1) % sides]
        ang = compute_internal_angle(prev, curr, nxt)

        v1 = np.array(prev) - np.array(curr)
        v2 = np.array(nxt) - np.array(curr)
        bis = v1/np.linalg.norm(v1) + v2/np.linalg.norm(v2)
        bis /= np.linalg.norm(bis)
        offset = 0.1 * min_len
        tx = curr[0] + bis[0] * offset
        ty = curr[1] + bis[1] * offset
        ax.text(tx, ty, f"{ang:.1f}°", fontsize=10,
                color='green', ha='center', va='center')

    return fig, lengths, correction


# --- Streamlit UI ---
st.title("🎯 אפליקציית שרטוט מצולעים חכמה")

sides = st.number_input("כמה צלעות?", min_value=3, max_value=12, value=5)

# קלט אורכי צלעות
st.subheader("📏 אורכי צלעות (השאר ריק לצלע חישובית)")
lengths = []
empty_count = 0
for i in range(sides):
    val = st.text_input(f"צלע {i+1}", value="", key=f"len_{i}")
    if val.strip() == "":
        lengths.append(None)
        empty_count += 1
    else:
        try:
            lengths.append(float(val))
        except ValueError:
            st.error("יש להזין מספר או להשאיר ריק")
            lengths.append(None)
            empty_count += 1

# קלט זוויות פנימיות
use_custom = st.checkbox("הזנת זוויות פנימיות בעצמך")
custom_angles = None
if use_custom:
    st.subheader("🎛 הזן זוויות פנימיות")
    custom_angles = []
    for i in range(sides):
        prev_edge = sides if i == 0 else i
        next_edge = i+1
        prompt = f"זווית בפינה {i+1} (בין צלע {prev_edge} לצלע {next_edge})"
        val = st.text_input(prompt, value="", key=f"ang_{i}")
        if val.strip() == "":
            custom_angles.append(None)
        else:
            try:
                custom_angles.append(float(val))
            except ValueError:
                st.error("ערכים חייבים להיות מספריים")
                custom_angles.append(None)

# כפתור שרטוט
if st.button("✏️ שרטוט מצולע"):
    if empty_count > 1:
        st.error("ניתן להשאיר ריק רק צלע אחת")
    elif use_custom and (None in custom_angles):
        st.error("יש למלא את כל הזוויות או לבטל הזנה עצמית")
    else:
        fig, final_lens, msg = draw_polygon(
            sides,
            lengths.copy(),
            custom_angles if use_custom else None
        )
        st.pyplot(fig)

        st.info("📐 אורכי הצלעות:")
        for idx, L in enumerate(final_lens, start=1):
            st.write(f"צלע {idx}: {L:.2f}")

        if msg:
            st.warning(f"⚠️ {msg}")

        # טעינת הורדות
        buf_png = BytesIO()
        fig.savefig(buf_png, format="png", dpi=300, bbox_inches='tight')
        st.download_button("🖼 הורד PNG", buf_png.getvalue(), "polygon.png", "image/png")

        buf_pdf = BytesIO()
        fig.savefig(buf_pdf, format="pdf", bbox_inches='tight')
        st.download_button("📄 הורד PDF", buf_pdf.getvalue(), "polygon.pdf", "application/pdf")