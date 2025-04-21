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
    A = (0.0, 0.0)
    B = (L1, 0.0)
    x = (L1**2 + L2**2 - L3**2) / (2 * L1)
    y2 = L2**2 - x**2
    if y2 < -TOL:
        st.error("לא ניתן לבנות משולש עם אורכים אלה.")
        return None, None
    y = np.sqrt(max(y2, 0.0))
    C = (x, y)

    pts = [A, B, C, A]
    fig, ax = plt.subplots(figsize=(5,5))
    xs, ys = zip(*pts)
    ax.plot(xs, ys, '-o')
    ax.set_aspect('equal')
    ax.axis('off')

    for i, (p1, p2) in enumerate(zip(pts[:-1], pts[1:])):
        mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
        ax.text(mx, my, f"{lengths[i]:.2f}", color='blue', fontsize=10,
                ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    for i, curr in enumerate(pts[:-1]):
        prev, nxt = pts[i-1], pts[(i+1)%3]
        ang = compute_internal_angle(prev, curr, nxt)
        v1 = np.array(prev)-np.array(curr)
        v2 = np.array(nxt)-np.array(curr)
        bis = (v1/np.linalg.norm(v1) + v2/np.linalg.norm(v2))
        bis /= np.linalg.norm(bis)
        ax.text(curr[0]+bis[0]*0.2, curr[1]+bis[1]*0.2, f"{ang:.1f}°",
                color='green', ha='center', va='center', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    return fig, lengths

def draw_polygon(sides, lengths, int_angles):
    if sides == 3 and all(L is not None for L in lengths) and int_angles is None:
        return draw_triangle(lengths)

    vecs, missing = [], [i for i, L in enumerate(lengths) if L is None]

    if int_angles:
        ext = [180 - a for a in int_angles]
        headings = np.cumsum([0] + ext[:-1])
    else:
        if len(missing) != 1:
            st.error("אם לא ניתנו זוויות, יש להשאיר צלע אחת ריקה בלבד.")
            return None, None
        headings = np.cumsum([0] * sides)

    for hd, L in zip(headings, lengths):
        if L is not None:
            th = np.radians(hd)
            vecs.append((L * np.cos(th), L * np.sin(th)))
        else:
            vecs.append(None)

    if missing:
        dx = sum(v[0] for v in vecs if v)
        dy = sum(v[1] for v in vecs if v)
        L = np.hypot(dx, dy)
        i = missing[0]
        lengths[i] = L
        vecs[i] = (-dx, -dy)

    pts = [(0.,0.)]
    for dx, dy in vecs:
        x, y = pts[-1]
        pts.append((x+dx, y+dy))

    fig, ax = plt.subplots(figsize=(5,5))
    xs, ys = zip(*pts)
    ax.plot(xs, ys, '-o')
    ax.set_aspect('equal')
    ax.axis('off')

    min_l = min(lengths)
    for i in range(sides):
        p1, p2 = pts[i], pts[i+1]
        mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
        ax.text(mx, my, f"{lengths[i]:.2f}", color='blue', fontsize=10,
                ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    for i in range(sides):
        prev, curr, nxt = pts[i-1], pts[i], pts[(i+1)%sides]
        ang = compute_internal_angle(prev, curr, nxt)
        v1 = np.array(prev)-np.array(curr)
        v2 = np.array(nxt)-np.array(curr)
        bis = (v1/np.linalg.norm(v1) + v2/np.linalg.norm(v2))
        bis /= np.linalg.norm(bis)
        ax.text(curr[0]+bis[0]*0.15*min_l, curr[1]+bis[1]*0.15*min_l, f"{ang:.1f}°",
                color='green', ha='center', va='center', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    return fig, lengths

st.title("🎯 שרטוט מצולעים מתוקן")

sides = st.number_input("מספר צלעות", 3, 12, 3, 1)
lengths = [st.text_input(f"צלע {i+1}") for i in range(sides)]
lengths = [None if not L.strip() else float(L) for L in lengths]

use_angles = st.checkbox("הזן זוויות פנימיות")
int_angles = None
if use_angles:
    int_angles = [st.text_input(f"זווית {i+1}") for i in range(sides)]
    if "" in int_angles:
        st.error("חובה להזין את כל הזוויות.")
    else:
        int_angles = [float(a) for a in int_angles]

if st.button("✏️ שרטוט"):
    fig, final_lengths = draw_polygon(sides, lengths, int_angles)
    if fig:
        st.pyplot(fig)
        st.write("**אורכי צלעות סופיים:**")
        for i, L in enumerate(final_lengths, 1):
            st.write(f"צלע {i}: {L:.2f}")