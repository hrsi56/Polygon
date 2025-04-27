from __future__ import annotations
import io
import zipfile
import datetime as dt
import base64
import json
import math
import string
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from scipy.optimize import minimize

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, ctx
from dash.dependencies import ALL

# יצירת אפליקציית Dash עם Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server


# ────── קבועים ועזרי גאומטריה ─────────────────────────────────────
TOL = 1e-6
LABEL_SHIFT = 0.04  # הזחה חיצונית של תוויות (אחוז מאורך הצלע הקצרה)

def vertex_names(n: int) -> List[str]:
    letters = string.ascii_uppercase
    return [(letters[i] if i < 26 else letters[i // 26 - 1] + letters[i % 26]) for i in range(n)]

@dataclass
class PolygonData:
    pts: np.ndarray
    lengths: List[float]
    angles_int: List[float]

    @property
    def names(self) -> List[str]:
        return vertex_names(len(self.pts))

def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    return math.degrees(
        math.acos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))
    )

def shoelace_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def is_polygon_possible(lengths: Sequence[float]) -> bool:
    L = sorted(lengths)
    return L[-1] < sum(L[:-1]) - 1e-9

def repaired_angles(n: int, angs: Sequence[float] | None):
    if angs is None:
        return None
    k = (n - 2) * 180.0 / sum(angs)
    return [a * k for a in angs]

def build_polygon_with_extra(lengths: Sequence[float], angles: Sequence[float]) -> PolygonData:
    n = len(lengths)
    L = np.asarray(lengths, float)
    ext = np.radians(180.0 - np.asarray(angles))
    heads = np.zeros(n)
    heads[1:] = np.cumsum(ext[:-1])
    vecs = np.stack([L * np.cos(heads), L * np.sin(heads)], axis=1)
    gap = vecs.sum(axis=0)
    if np.hypot(*gap) > TOL:
        vecs = np.vstack([vecs, -gap])
    pts = np.concatenate([[np.zeros(2)], np.cumsum(vecs, axis=0)])[:-1]
    lengths_corr = np.linalg.norm(vecs, axis=1).tolist()
    angles_corr = [
        angle_between(pts[i - 1] - pts[i], pts[(i + 1) % len(pts)] - pts[i])
        for i in range(len(pts))
    ]
    return PolygonData(pts, lengths_corr, angles_corr)

def build_polygon(lengths: Sequence[float], angles: Sequence[float]) -> PolygonData:
    n = len(lengths)
    L = np.asarray(lengths, float)
    ext = np.radians(180.0 - np.asarray(angles))
    heads = np.zeros(n)
    heads[1:] = np.cumsum(ext[:-1])
    vecs = np.stack([L * np.cos(heads), L * np.sin(heads)], axis=1)
    gap = vecs.sum(axis=0)
    if np.hypot(*gap) > TOL:
        vecs += (-gap * (L / L.sum())[:, None])
        gap = vecs.sum(axis=0)
    if np.hypot(*gap) > TOL:
        vecs[-1] -= gap
    pts = np.concatenate([[np.zeros(2)], np.cumsum(vecs, axis=0)])[:-1]
    lengths_corr = np.linalg.norm(vecs, axis=1).tolist()
    angles_corr = [
        angle_between(pts[i - 1] - pts[i], pts[(i + 1) % n] - pts[i])
        for i in range(n)
    ]
    return PolygonData(pts, lengths_corr, angles_corr)

def circumscribed_polygon(lengths: Sequence[float]) -> PolygonData:
    n = len(lengths)
    lengths_arr = np.asarray(lengths, float)

    def make_polygon(angles_rad):
        angle = 0.0
        pts = [np.array([0.0, 0.0])]
        for i in range(n):
            angle += angles_rad[i]
            dx = lengths_arr[i] * math.cos(angle)
            dy = lengths_arr[i] * math.sin(angle)
            pts.append(pts[-1] + np.array([dx, dy]))
        return np.array(pts)

    def objective(angles_rad):
        pts = make_polygon(angles_rad)
        return np.linalg.norm(pts[-1] - pts[0]) ** 2

    initial = np.full(n, -2 * np.pi / n)
    res = minimize(objective, initial, method='BFGS')
    pts = make_polygon(res.x)[:-1]

    angles_int = []
    for i in range(n):
        a = pts[i - 1] - pts[i]
        b = pts[(i + 1) % n] - pts[i]
        ang = angle_between(a, b)
        angles_int.append(ang)

    return PolygonData(pts, list(lengths_arr), angles_int)

# ────── פונקציות עזר נוספות ─────────────────────────────────────

def diagonals_info(poly: PolygonData):
    pts = poly.pts
    names = poly.names
    n = len(pts)
    info = []
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # לא לקשר את הנקודה האחרונה לראשונה
            v = pts[j] - pts[i]
            length = float(np.linalg.norm(v))

            def pick_angle(idx: int, vec: np.ndarray):
                prev_vec = pts[idx - 1] - pts[idx]
                next_vec = pts[(idx + 1) % n] - pts[idx]
                ang_prev = angle_between(vec, prev_vec)
                ang_next = angle_between(vec, next_vec)
                if ang_prev <= ang_next:
                    return ang_prev, f"{names[idx - 1]}{names[idx]}"
                else:
                    return ang_next, f"{names[idx]}{names[(idx + 1) % n]}"

            ang_i, side_i = pick_angle(i, v)
            ang_j, side_j = pick_angle(j, -v)

            info.append({
                "i": i,
                "j": j,
                "length": length,
                "end_i": {"side": side_i, "angle": ang_i},
                "end_j": {"side": side_j, "angle": ang_j},
            })
    return info

def bounding_rect(pts: np.ndarray):
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    rect = np.array([
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax]
    ])
    return rect, xmax - xmin, ymax - ymin

def triangle_altitudes(pts: np.ndarray):
    """החזרת גבהים במשולש."""
    alt = []
    names = vertex_names(3)
    for idx in range(3):
        A = pts[idx]
        B = pts[(idx + 1) % 3]
        C = pts[(idx + 2) % 3]
        BC_vec = C - B
        t = np.dot(A - B, BC_vec) / np.dot(BC_vec, BC_vec)
        foot = B + t * BC_vec
        length = float(np.linalg.norm(A - foot))
        alt.append({
            "from_v": idx,
            "to_side": f"{names[(idx + 1) % 3]}{names[(idx + 2) % 3]}",
            "foot": foot,
            "length": length
        })
    return alt

# ────── ציור המצולע ─────────────────────────────────────
def draw_polygon_fig(poly: PolygonData, show_alt: bool):
    rect, w, h = bounding_rect(poly.pts)
    img = mpimg.imread('assets/Subject.PNG')  # רקע מותאם
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, extent=[rect[0][0], rect[1][0], rect[0][1], rect[2][1]], alpha=0.7)

    n = len(poly.pts)
    names = poly.names
    pts_closed = np.vstack([poly.pts, poly.pts[0]])

    ax.set_aspect('equal')
    ax.axis('off')
    ax.plot(pts_closed[:, 0], pts_closed[:, 1], '-o', lw=1.4, color='green', alpha=1)

    min_len = min(poly.lengths)

    # ציור קווים לפינות המלבן
    polygon_path = MplPath(poly.pts)
    rect_edges = [rect[1] - rect[0], rect[3] - rect[0]]

    def segments_intersect(p1, p2, q1, q2):
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
        return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))

    for corner in rect:
        dists = [np.linalg.norm(corner - p) for p in poly.pts]
        nearest = np.argsort(dists)
        vecs = [poly.pts[i] - corner for i in nearest]

        if all(np.linalg.norm(v) > 1e-8 for v in vecs[:2]):
            u = vecs[0] / np.linalg.norm(vecs[0])
            v = vecs[1] / np.linalg.norm(vecs[1])
            if abs(np.dot(u, v)) > 0.998:
                nearest = [nearest[0]]

        for i in nearest:
            p = poly.pts[i]
            vec = p - corner
            dist = np.linalg.norm(vec)
            if dist < 1e-8:
                continue

            intersects = False
            for k in range(n):
                a = poly.pts[k]
                b = poly.pts[(k + 1) % n]
                if np.allclose(a, corner) or np.allclose(b, corner) or np.allclose(a, p) or np.allclose(b, p):
                    continue
                if segments_intersect(corner, p, a, b):
                    intersects = True
                    break

            mid = 0.5 * (corner + p)
            if intersects or polygon_path.contains_point(mid):
                continue

            vec_norm = vec / dist
            best_edge_len = None
            best_cos = -1
            for edge_vec in rect_edges:
                elen = np.linalg.norm(edge_vec)
                if elen < 0.5:
                    continue
                edir = edge_vec / elen
                cang = abs(np.dot(vec_norm, edir))
                if cang > best_cos:
                    best_cos = cang
                    best_edge_len = elen

            if best_edge_len and dist < best_edge_len:
                ax.plot([p[0], corner[0]], [p[1], corner[1]], color='orange', lw=4, alpha=0.3)
                ax.text(*(0.5 * (p + corner)), f"{dist:.2f}", fontsize=6, color='black', ha='center', va='center')

    # ציור אלכסונים
    diags = diagonals_info(poly)
    for d in diags:
        p1, p2 = poly.pts[d['i']], poly.pts[d['j']]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '--', lw=0.8, color='brown', alpha=0.6)
        mid = 0.5 * (p1 + p2)
        ax.text(mid[0], mid[1], f"{d['length']:.2f}", fontsize=6, color='brown', ha='center', va='center')

    # תוויות קודקודים
    for i, (x, y) in enumerate(poly.pts):
        prev_v = poly.pts[i] - poly.pts[i - 1]
        next_v = poly.pts[(i + 1) % n] - poly.pts[i]
        norm_v = np.array([-(prev_v[1] + next_v[1]), prev_v[0] + next_v[0]])
        if np.linalg.norm(norm_v):
            norm_v = norm_v / np.linalg.norm(norm_v)
        ax.text(x + norm_v[0] * LABEL_SHIFT * min_len,
                y + norm_v[1] * LABEL_SHIFT * min_len,
                names[i], fontsize=9, weight='bold', color='blue', ha='center', va='center')

    # תוויות אורכים
    for i in range(n):
        mid = 0.5 * (poly.pts[i] + poly.pts[(i + 1) % n])
        edge = poly.pts[(i + 1) % n] - poly.pts[i]
        en = np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)
        ax.text(*(mid + en * LABEL_SHIFT * min_len),
                f"{poly.lengths[i]:.2f}", fontsize=7,
                bbox=dict(facecolor='green', alpha=0.15, edgecolor='none'), ha='center', va='center')

    # תוויות זוויות פנימיות
    for i in range(n):
        p = poly.pts[i]
        v1 = poly.pts[i - 1] - p
        v2 = poly.pts[(i + 1) % n] - p
        bis = v1 / np.linalg.norm(v1) + v2 / np.linalg.norm(v2)
        bis = bis / np.linalg.norm(bis) if np.linalg.norm(bis) else np.array([v2[1], -v2[0]])
        text_pos = p + bis * (0.23 * min_len)
        ax.text(text_pos[0], text_pos[1], f"{poly.angles_int[i]:.1f}°", fontsize=9, color='red', ha='center', va='center')

    # גבהים במשולש
    alt_data = None
    if show_alt and n == 3:
        alt_data = triangle_altitudes(poly.pts)
        for alt in alt_data:
            A = poly.pts[alt['from_v']]
            foot = alt['foot']
            ax.plot([A[0], foot[0]], [A[1], foot[1]], ':', color='magenta', lw=1.2)
            ax.text(foot[0], foot[1], f"h={alt['length']:.2f}", fontsize=6, color='magenta', ha='left', va='bottom')

    return fig, diags, alt_data


    font = FontProperties(fname="assets/Pacifico-Regular.ttf")
    ax.text(rect[0][0] + 0.05 * w, rect[2][1] + 0.05 * h,
            "Created by Yarden Viktor Dejorno",
            fontsize=10,
            fontproperties=font,
            color='black',
            ha='left',
            va='bottom')

# ────── Layout ראשי ─────────────────────────────────────

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1('📐 Polygon Drawer', className="text-center mt-4"),
            html.H5('לינקו - תמורת טובות הנעה.', className="text-center mb-4 text-muted"),
        ])
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Label('Number of sides:'),
            dcc.Input(id='num-sides', type='number', min=3, max=12, step=1, value=4,
                      className='form-control mb-3'),

            html.Div(id='length-inputs', className="mb-3"),

            dbc.Checklist(
                options=[{'label': 'Provide internal angles?', 'value': 'show'}],
                id='provide-angles',
                switch=True,
                className='mb-3'
            ),

            html.Div(id='angle-inputs', className="mb-3"),

            dbc.Checklist(
                options=[{'label': 'Add extra side if needed', 'value': 'extra'}],
                id='add-extra',
                switch=True,
                className='mb-3'
            ),

            dbc.Button('Draw Polygon', id='draw-button', color="success", className="w-100 mb-3 btn-lg"),
            html.Div(id='download-button-container'),  # נעדכן את כפתור ההורדה אחרי ציור

            dcc.Download(id='download-zip'),
        ], md=4),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        dcc.Loading(
                            id="loading-polygon",
                            type="circle",
                            children=html.Img(id='polygon-image', style={'width': '100%', 'height': 'auto'})
                        )
                    ]),
                    html.Hr(),
                    html.H5('Numerical Data', className="text-center"),
                    html.Div(id='numeric-data',
                             style={'fontFamily': 'monospace', 'backgroundColor': '#ffffff', 'padding': '10px',
                                    'borderRadius': '8px', 'boxShadow': '0px 2px 5px rgba(0,0,0,0.05)'}),
                    html.Hr(),
                    html.H5('Diagonals Info', className="text-center"),
                    html.Div(id='diag-data',
                             style={'fontFamily': 'monospace', 'backgroundColor': '#ffffff', 'padding': '10px',
                                    'borderRadius': '8px', 'boxShadow': '0px 2px 5px rgba(0,0,0,0.05)'}),
                ])
            ], className="shadow p-3 bg-white rounded")
        ], md=8),
    ])
], fluid=True)

# ────── עדכון שדות אורך ─────────────────────────────────────
@app.callback(
    Output('length-inputs', 'children'),
    Input('num-sides', 'value')
)
def update_length_inputs(n):
    return [
        dbc.InputGroup([
            dbc.InputGroupText(f"Length {i + 1}"),
            dbc.Input(id={'type': 'length', 'index': i}, type='number', min=0.1, max=10000, step=0.5, value=1.0)
        ], className="mb-2") for i in range(n)
    ]

# ────── עדכון שדות זווית ─────────────────────────────────────
@app.callback(
    Output('angle-inputs', 'children'),
    Input('provide-angles', 'value'),
    Input('num-sides', 'value')
)
def update_angle_inputs(show, n):
    if show and 'show' in show:
        return [
            dbc.InputGroup([
                dbc.InputGroupText(f"Angle {vertex_names(n)[i]}"),
                dbc.Input(id={'type': 'angle', 'index': i}, type='number', min=1, max=360,
                          step=1, value=round(180 * (n - 2) / n, 1))
            ], className="mb-2") for i in range(n)
        ]
    return []

# ────── ציור הפוליגון ונתונים מספריים ─────────────────────
@app.callback(
    Output('polygon-image', 'src'),
    Output('numeric-data', 'children'),
    Output('diag-data', 'children'),
    Output('download-button-container', 'children'),  # מוסיפים גם את זה
    Input('draw-button', 'n_clicks'),
    State('num-sides', 'value'),
    State({'type': 'length', 'index': ALL}, 'value'),
    State('provide-angles', 'value'),
    State({'type': 'angle', 'index': ALL}, 'value'),
    State('add-extra', 'value')
)
def draw_polygon_callback(n_clicks, n, lengths, show_angles, angles, add_extra):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    if not is_polygon_possible(lengths):
        return dash.no_update, dbc.Alert("⚠️ Side lengths violate polygon inequality.", color="danger", className="text-center"), dash.no_update, dash.no_update

    if show_angles and 'show' in show_angles:
        if add_extra and 'extra' in add_extra:
            poly = build_polygon_with_extra(lengths, angles)
        else:
            poly = build_polygon(lengths, repaired_angles(n, angles))
    else:
        poly = circumscribed_polygon(lengths)

    show_alt = (n == 3)
    fig, diags, altitudes = draw_polygon_fig(poly, show_alt)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    src = f"data:image/png;base64,{img_b64}"

    num_data = {
        'Area': round(shoelace_area(poly.pts), 4),
        'Width': round(bounding_rect(poly.pts)[1], 4),
        'Height': round(bounding_rect(poly.pts)[2], 4)
    }
    if altitudes:
        num_data['Altitudes'] = [round(a['length'], 4) for a in altitudes]

    diag_data = {
        f"{poly.names[d['i']]}{poly.names[d['j']]}": {
            'Length': round(d['length'], 3),
            poly.names[d['i']]: d['end_i'],
            poly.names[d['j']]: d['end_j'],
        }
        for d in diags
    }

    num_div = html.Pre(json.dumps(num_data, indent=2))
    diag_div = html.Pre(json.dumps(diag_data, indent=2))

    # יוצרים את כפתור ההורדה רק אחרי ציור
    download_btn = dbc.Button('Download ZIP', id='download-button', color="primary", className="w-100 btn-lg mt-2")

    return src, num_div, diag_div, download_btn

# ────── הורדת ZIP ─────────────────────────────────────
@app.callback(
    Output('download-zip', 'data'),
    Input('download-button', 'n_clicks'),
    State('num-sides', 'value'),
    State({'type': 'length', 'index': ALL}, 'value'),
    State('provide-angles', 'value'),
    State({'type': 'angle', 'index': ALL}, 'value'),
    State('add-extra', 'value'),
    prevent_initial_call=True
)
def download_zip(n_clicks, n, lengths, show_angles, angles, add_extra):
    if show_angles and 'show' in show_angles:
        if add_extra and 'extra' in add_extra:
            poly = build_polygon_with_extra(lengths, angles)
        else:
            poly = build_polygon(lengths, repaired_angles(n, angles))
    else:
        poly = circumscribed_polygon(lengths)

    show_alt = (n == 3)
    fig, diags, altitudes = draw_polygon_fig(poly, show_alt)

    num_data = {
        'Area': round(shoelace_area(poly.pts), 4),
        'Width': round(bounding_rect(poly.pts)[1], 4),
        'Height': round(bounding_rect(poly.pts)[2], 4)
    }
    if altitudes:
        num_data['Altitudes'] = [round(a['length'], 4) for a in altitudes]

    diag_data = {
        f"{poly.names[d['i']]}{poly.names[d['j']]}": {
            'Length': round(d['length'], 3),
            poly.names[d['i']]: d['end_i'],
            poly.names[d['j']]: d['end_j'],
        }
        for d in diags
    }

    txt = json.dumps({'Numerical data': num_data, 'Diagonals': diag_data}, indent=2).encode()

    png_buf = io.BytesIO()
    fig.savefig(png_buf, format='png', dpi=300, bbox_inches='tight')
    png_buf.seek(0)

    pdf_buf = io.BytesIO()
    fig.savefig(pdf_buf, format='pdf', bbox_inches='tight')
    pdf_buf.seek(0)

    svg_buf = io.BytesIO()
    fig.savefig(svg_buf, format='svg', bbox_inches='tight')
    svg_buf.seek(0)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        ts = dt.datetime.now().strftime('%Y%m%d_%H%M')
        base = f"YVD_Polygon_{ts}"
        zf.writestr(f"{base}.txt", txt)
        zf.writestr(f"{base}.png", png_buf.getvalue())
        zf.writestr(f"{base}.pdf", pdf_buf.getvalue())
        zf.writestr(f"{base}.svg", svg_buf.getvalue())

    zip_buf.seek(0)
    return dcc.send_bytes(lambda: zip_buf.getvalue(), filename=f"{base}.zip")

# ────── הרצת האפליקציה ─────────────────────────────────────
if __name__ == '__main__':
    app.run_server(debug=True)