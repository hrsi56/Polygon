# -------------------------------------------------
# הוסף במקום כלשהו למעלה (ליד פונקציות העזר)
def polygon_closes(sides, lengths, int_angles, tol=TOL):
    """
    החזר True אם וקטור הסכום של כל הצלעות מתאפס (המצולע נסגר).
    משתמש באותו חישוב headings כמו draw_polygon.
    """
    if int_angles:
        ext = [180 - a for a in int_angles]
        headings = np.cumsum([0] + ext[:-1])
    else:
        headings = np.cumsum([0] + [0] * (sides - 1))

    dx = dy = 0.0
    for hd, L in zip(headings, lengths):
        rad = np.radians(hd)
        dx += L * np.cos(rad)
        dy += L * np.sin(rad)

    return np.hypot(dx, dy) < tol
# -------------------------------------------------

# -------------------------------------------------
# בתוך בלוק הכפתור – מיד אחרי if fig: ...
# (כל מה שמתחת נשאר כמו שהיה)
    if fig:
        st.pyplot(fig)

        # בדיקת סגירה ✨
        if not polygon_closes(sides, final_lengths, int_angles):
            st.warning("⚠️ המצולע אינו נסגר עם הערכים שהוזנו.")

        st.markdown("### אורכי צלעות")
        ...
# -------------------------------------------------