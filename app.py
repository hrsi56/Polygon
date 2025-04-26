from __future__ import annotations
import io
import zipfile
import datetime as dt
from dataclasses import dataclass
from typing import List, Sequence
import numpy as np
import math
import string
import base64
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from scipy.optimize import minimize
import dash
from dash import html, dcc, Input, Output, State
from dash.dependencies import ALL

# ────── geometry helpers ──────────────────────────────────────────────────
TOL = 1e-6
LABEL_SHIFT = 0.04  # outward label offset (fraction of min side)

def vertex_names(n: int) -> List[str]:
    letters = string.ascii_uppercase
    return [(letters[i] if i < 26 else letters[i // 26 - 1] + letters[i % 26]) for i in range(n)]

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
    angles_corr = [angle_between(pts[i - 1] - pts[i], pts[(i + 1) % len(pts)] - pts[i]) for i in range(len(pts))]
    return PolygonData(pts, lengths_corr, angles_corr)

def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    return math.degrees(math.acos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))

def shoelace_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def is_polygon_possible(lengths: Sequence[float]) -> bool:
    L = sorted(lengths)
    return L[-1] < sum(L[:-1]) - 1e-9

@dataclass
class PolygonData:
    pts: np.ndarray
    lengths: List[float]
    angles_int: List[float]

    @property
    def names(self) -> List[str]:
        return vertex_names(len(self.pts))


def repaired_angles(n: int, angs: Sequence[float] | None):
    if angs is None:
        return None
    k = (n - 2) * 180.0 / sum(angs)
    return [a * k for a in angs]

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
        ang = math.degrees(math.acos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)))
        angles_int.append(ang)
    return PolygonData(pts, list(lengths_arr), angles_int)

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
    angles_corr = [angle_between(pts[i - 1] - pts[i], pts[(i + 1) % n] - pts[i]) for i in range(n)]
    return PolygonData(pts, lengths_corr, angles_corr)

def diagonals_info(poly: PolygonData):
    pts = poly.pts
    names = poly.names
    n = len(pts)
    info = []
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            v = pts[j] - pts[i]
            length = float(np.linalg.norm(v))
            def pick_angle(idx: int, vec: np.ndarray):
                prev_vec = pts[idx - 1] - pts[idx]
                next_vec = pts[(idx + 1) % n] - pts[idx]
                ang_prev = angle_between(vec, prev_vec)
                ang_next = angle_between(vec, next_vec)
                if ang_prev <= ang_next:
                    return ang_prev, f'{names[idx - 1]}{names[idx]}'
                return ang_next, f'{names[idx]}{names[(idx + 1) % n]}'
            ang_i, side_i = pick_angle(i, v)
            ang_j, side_j = pick_angle(j, -v)
            info.append({
                'i': i, 'j': j, 'length': length,
                'end_i': {'side': side_i, 'angle': ang_i},
                'end_j': {'side': side_j, 'angle': ang_j}
            })
    return info

def bounding_rect(pts: np.ndarray):
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    rect = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    return rect, xmax - xmin, ymax - ymin

def triangle_altitudes(pts: np.ndarray):
    alt = []
    names = vertex_names(3)
    for idx in range(3):
        A = pts[idx]
        B = pts[(idx + 1) % 3]
        C = pts[(idx + 2) % 3]
        BC = C - B
        t = np.dot(A - B, BC) / np.dot(BC, BC)
        foot = B + t * BC
        length = float(np.linalg.norm(A - foot))
        alt.append({'from_v': idx, 'to_side': f'{names[(idx + 1) % 3]}{names[(idx + 2) % 3]}', 'foot': foot, 'length': length})
    return alt

def draw_polygon_fig(poly: PolygonData, show_alt: bool):
    rect, w, h = bounding_rect(poly.pts)
    img = mpimg.imread('assets/Subject.PNG')
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, extent=[rect[0][0], rect[1][0], rect[0][1], rect[2][1]], alpha=0.7)
    n = len(poly.pts)
    names = poly.names
    pts_closed = np.vstack([poly.pts, poly.pts[0]])
    ax.set_aspect('equal')
    ax.axis('off')
    ax.plot(pts_closed[:, 0], pts_closed[:, 1], '-o', lw=1.4, color='green', alpha=1)
    min_len = min(poly.lengths)
    def segments_intersect(p1, p2, q1, q2):
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
        return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))
    polygon_path = MplPath(poly.pts)
    rect_edges = [rect[1] - rect[0], rect[3] - rect[0]]
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
                ax.text(*(0.5 * (p + corner)), f'{dist:.2f}', fontsize=6, color='black', ha='center', va='center')
    diags = diagonals_info(poly)
    for d in diags:
        p1, p2 = poly.pts[d['i']], poly.pts[d['j']]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '--', lw=0.8, color='brown', alpha=0.6)
        mid = 0.5 * (p1 + p2)
        ax.text(mid[0], mid[1], f'{d['length']:.2f}', fontsize=6, color='brown', ha='center', va='center')
        vec_i = (p2 - p1) / np.linalg.norm(p2 - p1)
        ax.text(*(p1 + vec_i * 0.1 * min_len), f'{d['end_i']['angle']:.1f}°\n{d['end_i']['side']}', fontsize=5, color='brown', ha='center', va='center')
        vec_j = (p1 - p2) / np.linalg.norm(p1 - p2)
        ax.text(*(p2 + vec_j * 0.1 * min_len), f'{d['end_j']['angle']:.1f}°\n{d['end_j']['side']}', fontsize=5, color='brown', ha='center', va='center')
    for i, (x, y) in enumerate(poly.pts):
        prev_v = poly.pts[i] - poly.pts[i - 1]
        next_v = poly.pts[(i + 1) % n] - poly.pts[i]
        norm_v = np.array([-(prev_v[1] + next_v[1]), prev_v[0] + next_v[0]])
        if np.linalg.norm(norm_v):
            norm_v = norm_v / np.linalg.norm(norm_v)
        ax.text(x + norm_v[0] * LABEL_SHIFT * min_len, y + norm_v[1] * LABEL_SHIFT * min_len,
                names[i], fontsize=9, weight='bold', color='blue', ha='center', va='center')
        mid = 0.5 * (poly.pts[i] + poly.pts[(i + 1) % n])
        edge = poly.pts[(i + 1) % n] - poly.pts[i]
        en = np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)
        ax.text(*(mid + en * LABEL_SHIFT * min_len), f'{poly.lengths[i]:.2f}', fontsize=7,
                bbox=dict(facecolor='green', alpha=0.15, edgecolor='none'), ha='center', va='center')
        ed = edge / np.linalg.norm(edge)
        angle_h = math.degrees(math.acos(np.clip(abs(np.dot(ed, [1, 0])), -1, 1)))
        angle_v = math.degrees(math.acos(np.clip(abs(np.dot(ed, [0, 1])), -1, 1)))
        angle_r = min(angle_h, angle_v)
        base = poly.pts[i]
        en2 = np.array([-edge[1], edge[0]])
        if np.linalg.norm(en2):
            en2 = en2 / np.linalg.norm(en2)
        theta = math.radians(75)
        rot = np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])
        rv = rot @ en2
        lbl = base + rv * LABEL_SHIFT * min_len * 3.5
        ax.text(lbl[0], lbl[1], f'∠{angle_r:.1f}°', fontsize=7, color='green', ha='center', va='center')
    for i in range(n):
        p = poly.pts[i]
        v1 = poly.pts[i - 1] - p
        v2 = poly.pts[(i + 1) % n] - p
        bis = v1 / np.linalg.norm(v1) + v2 / np.linalg.norm(v2)
        bis = bis / np.linalg.norm(bis) if np.linalg.norm(bis) else np.array([v2[1], -v2[0]])
        text_pos = p + bis * (0.23 * min_len)
        ax.text(text_pos[0], text_pos[1], f'{poly.angles_int[i]:.1f}°', fontsize=9, color='red', ha='center', va='center')
    alt_data = None
    if show_alt and n == 3:
        alt_data = triangle_altitudes(poly.pts)
        for alt in alt_data:
            A = poly.pts[alt['from_v']]
            foot = alt['foot']
            ax.plot([A[0], foot[0]], [A[1], foot[1]], ':', color='magenta', lw=1.2)
            ax.text(foot[0], foot[1], f'h={alt['length']:.2f}', fontsize=6, color='magenta', ha='left', va='bottom')
    rc = np.vstack([rect, rect[0]])
    ax.plot(rc[:, 0], rc[:, 1], '-.', lw=1, alpha=0.5, color='purple')
    HW = h * w
    edge_lbls = [w, h, w, h]
    for i in range(4):
        p1, p2 = rect[i], rect[(i + 1) % 4]
        mid = (p1 + p2) / 2
        ev = p2 - p1
        nv = np.array([-ev[1], ev[0]])
        if np.linalg.norm(nv):
            nv = nv / np.linalg.norm(nv)
        offs = -0.09 * max(w, h)
        pos = mid + nv * offs
        ax.text(pos[0], pos[1], f'{edge_lbls[i]:.2f}', fontsize=7, color='purple', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))
    base_mid = 0.5 * (rect[0] + rect[1])
    ev2 = rect[1] - rect[0]
    nv2 = np.array([-ev2[1], ev2[0]])
    if np.linalg.norm(nv2):
        nv2 = nv2 / np.linalg.norm(nv2)
    offs2 = -0.09 * max(w, h)
    pos2 = base_mid + nv2 * offs2
    area_pos = np.array([2 * pos2[0], 1.06 * pos2[1]])
    ax.text(area_pos[0], area_pos[1], f'Area Polygon = {shoelace_area(poly.pts):.2f}\nArea REC={HW:.2f}\nw={w:.2f}\nh={h:.2f}', fontsize=8, ha='left', va='center', color='black')
    logo_pos = np.array([0.4 * pos2[0], 1.1 * pos2[1]])
    font_prop = FontProperties(fname='assets/Pacifico-Regular.ttf')
    ax.text(logo_pos[0], logo_pos[1], 'The App Created by:\nYarden Viktor Dejorno', fontsize=9, font=font_prop)
    rect_sides = [(rect[i], rect[(i + 1) % 4]) for i in range(4)]
    for pt in poly.pts:
        dists = []
        for p1, p2 in rect_sides:
            ev = p2 - p1
            el2 = np.dot(ev, ev)
            if el2 < 1e-8:
                continue
            t = np.dot(pt - p1, ev) / el2
            t_clamped = np.clip(t, 0, 1)
            foot = p1 + t_clamped * ev
            dist = np.linalg.norm(pt - foot)
            dists.append((dist, foot))
        dists.sort(key=lambda x: x[0])
        for dist, foot in dists[:2]:
            ax.plot([pt[0], foot[0]], [pt[1], foot[1]], linestyle='dashed', linewidth=0.8, color='green', alpha=0.7)
            midp = 0.5 * (pt + foot)
            offv = foot - pt
            ort = np.array([-offv[1], offv[0]])
            if np.linalg.norm(ort):
                ort = ort / np.linalg.norm(ort)
            lblp = midp + ort * LABEL_SHIFT * 0.1 * min_len
            if dist > 0.01:
                ax.text(lblp[0], lblp[1], f'{dist:.2f}', fontsize=3, color='green', ha='center', va='center')
    return fig, diags, alt_data

# ────── Dash Application ─────────────────────────────────────────────────
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1('📐 Polygon Drawer'),
    html.H2('לינקו - תמורת טובות הנעה.'),
    html.Label('Number of sides'),
    dcc.Input(id='num-sides', type='number', min=3, max=12, step=1, value=4),
    html.Div(id='length-inputs'),
    dcc.Checklist(id='provide-angles', options=[{'label': 'Provide internal angles?', 'value': 'show'}]),
    html.Div(id='angle-inputs'),
    dcc.Checklist(id='add-extra', options=[{'label': 'Add extra closing side on gap', 'value': 'extra'}]),
    html.Button('Draw polygon', id='draw-button'),
    html.Div(html.Img(id='polygon-image', style={'maxWidth': '100%'})),
    html.Div(id='numeric-data'),
    html.Div(id='diag-data'),
    html.Button('Download all (ZIP)', id='download-button'),
    dcc.Download(id='download-zip')
])

@app.callback(
    Output('length-inputs', 'children'),
    Input('num-sides', 'value')
)
def update_length_inputs(n):
    return [html.Div([html.Label(f'Length {i+1}'), dcc.Input(id={'type': 'length', 'index': i}, type='number', min=0.1, max=10000, step=0.5, value=1.0)]) for i in range(n)]

@app.callback(
    Output('angle-inputs', 'children'),
    Input('provide-angles', 'value'),
    Input('num-sides', 'value')
)
def update_angle_inputs(show, n):
    if 'show' in show:
        return [html.Div([html.Label(f'Angle {vertex_names(n)[i]}'), dcc.Input(id={'type': 'angle', 'index': i}, type='number', min=1, max=360, step=1, value=round(180*(n-2)/n, 1))]) for i in range(n)]
    return []

@app.callback(
    Output('polygon-image', 'src'),
    Output('numeric-data', 'children'),
    Output('diag-data', 'children'),
    Input('draw-button', 'n_clicks'),
    State('num-sides', 'value'),
    State({'type': 'length', 'index': ALL}, 'value'),
    State('provide-angles', 'value'),
    State({'type': 'angle', 'index': ALL}, 'value'),
    State('add-extra', 'value')
)
def draw_polygon_callback(n_clicks, n, lengths, show_angles, angles, add_extra):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update
    if not is_polygon_possible(lengths):
        return dash.no_update, html.Div('⚠️ Side lengths violate polygon inequality.'), dash.no_update
    if 'show' in show_angles:
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
    src = f'data:image/png;base64,{img_b64}'
    num_data = {'Area': round(shoelace_area(poly.pts), 4), 'Width': round(bounding_rect(poly.pts)[1], 4), 'Height': round(bounding_rect(poly.pts)[2], 4)}
    if altitudes:
        num_data['Altitudes'] = [round(a['length'], 4) for a in altitudes]
    num_div = html.Pre(json.dumps(num_data, indent=2))
    diag_data = {}
    for d in diags:
        key = f'{poly.names[d['i']]}{poly.names[d['j']]}'
        diag_data[key] = {'Length': round(d['length'], 3), poly.names[d['i']]: d['end_i'], poly.names[d['j']]: d['end_j']}
    diag_div = html.Pre(json.dumps(diag_data, indent=2))
    return src, num_div, diag_div

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
    if 'show' in show_angles:
        if add_extra and 'extra' in add_extra:
            poly = build_polygon_with_extra(lengths, angles)
        else:
            poly = build_polygon(lengths, repaired_angles(n, angles))
    else:
        poly = circumscribed_polygon(lengths)
    show_alt = (n == 3)
    fig, diags, altitudes = draw_polygon_fig(poly, show_alt)
    num_data = {'Area': round(shoelace_area(poly.pts), 4), 'Width': round(bounding_rect(poly.pts)[1], 4), 'Height': round(bounding_rect(poly.pts)[2], 4)}
    diag_data = {}
    for d in diags:
        key = f'{poly.names[d['i']]}{poly.names[d['j']]}'
        diag_data[key] = {'Length': round(d['length'], 3), poly.names[d['i']]: d['end_i'], poly.names[d['j']]: d['end_j']}
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
        base = f'YVD_Polygon_{ts}'
        zf.writestr(f'{base}.txt', txt)
        zf.writestr(f'{base}.png', png_buf.getvalue())
        zf.writestr(f'{base}.pdf', pdf_buf.getvalue())
        zf.writestr(f='{base}.svg', svg_buf.getvalue())
    zip_buf.seek(0)
    return dcc.send_bytes(lambda: zip_buf.getvalue(), filename=f'{base}.zip')

if __name__ == '__main__':
    app.run_server(debug=True)
