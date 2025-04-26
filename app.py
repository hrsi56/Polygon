import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
import math
import base64
import io
from PIL import Image

# הגדרת האפליקציה
app = dash.Dash(__name__)
server = app.server  # כדי ש-Render ידע להריץ


# ----------------- פונקציות עזר -----------------

def vertex_names(n):
	letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	return [(letters[i] if i < 26 else letters[i // 26 - 1] + letters[i % 26]) for i in range(n)]


def build_polygon(lengths):
	n = len(lengths)
	ext_angle = 360 / n
	angles = np.cumsum([np.radians(ext_angle)] * n)
	pts = np.zeros((n, 2))
	for i in range(1, n):
		pts[i] = pts[i - 1] + [lengths[i - 1] * np.cos(angles[i - 1]),
		                       lengths[i - 1] * np.sin(angles[i - 1])]
	return pts


def shoelace_area(pts):
	x, y = pts[:, 0], pts[:, 1]
	return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def load_background_image(filepath):
	with open(filepath, "rb") as image_file:
		encoded = base64.b64encode(image_file.read()).decode()
	return f"data:image/png;base64,{encoded}"


# ----------------- מבנה עמוד -----------------

app.layout = html.Div([
	html.H1("📐 Polygon Drawer by Yarden", style={"textAlign": "center"}),

	html.Div([
		html.Label("Number of sides:"),
		dcc.Input(id="num-sides", type="number", min=3, max=12, value=4, step=1)
	], style={"marginBottom": "20px"}),

	html.Div(id="sides-inputs"),

	html.Button("Draw Polygon", id="draw-button", n_clicks=0, style={"marginTop": "20px"}),

	dcc.Graph(id="polygon-graph", style={"height": "80vh"}),

	html.Div(id="area-output", style={"marginTop": "20px", "textAlign": "center"})
])


# ----------------- Callbacks -----------------

@app.callback(
	Output("sides-inputs", "children"),
	Input("num-sides", "value")
)
def update_side_inputs(n):
	return [
		html.Div([
			html.Label(f"Length {i + 1}:"),
			dcc.Input(id=f"side-{i}", type="number", min=0.1, step=0.1, value=1.0)
		]) for i in range(n)
	]


@app.callback(
	Output("polygon-graph", "figure"),
	Output("area-output", "children"),
	Input("draw-button", "n_clicks"),
	State("num-sides", "value"),
	State({"type": "Input", "index": dash.ALL}, "value"),
	prevent_initial_call=True
)
def draw_polygon(n_clicks, n, sides_values):
	lengths = []
	for i in range(n):
		lengths.append(float(sides_values[i] if sides_values[i] else 1.0))

	pts = build_polygon(lengths)
	pts_closed = np.vstack([pts, pts[0]])

	fig = go.Figure()

	# ציור מצולע
	fig.add_trace(go.Scatter(
		x=pts_closed[:, 0],
		y=pts_closed[:, 1],
		mode="lines+markers",
		fill="toself",
		name="Polygon",
		line=dict(color="green"),
		marker=dict(size=8, color="blue")
	))

	# קווי אלכסון
	for i in range(n):
		for j in range(i + 2, n):
			if not (i == 0 and j == n - 1):
				fig.add_trace(go.Scatter(
					x=[pts[i, 0], pts[j, 0]],
					y=[pts[i, 1], pts[j, 1]],
					mode="lines",
					line=dict(dash="dash", color="orange"),
					showlegend=False
				))

	# תמונת רקע
	img_source = load_background_image("Subject.PNG")
	fig.update_layout(
		images=[dict(
			source=img_source,
			xref="x",
			yref="y",
			x=pts[:, 0].min() - 1,
			y=pts[:, 1].max() + 1,
			sizex=(pts[:, 0].max() - pts[:, 0].min()) + 2,
			sizey=(pts[:, 1].max() - pts[:, 1].min()) + 2,
			sizing="stretch",
			opacity=0.4,
			layer="below"
		)],
		xaxis=dict(visible=False),
		yaxis=dict(visible=False),
		margin=dict(t=20, b=20, l=20, r=20),
		height=700,
		plot_bgcolor="white"
	)

	# חתימה
	fig.add_annotation(
		xref="paper", yref="paper",
		x=0.5, y=-0.15,
		showarrow=False,
		text="Created by Yarden Viktor Dejorno",
		font=dict(size=12, color="black")
	)

	area = shoelace_area(pts)
	return fig, f"Polygon Area: {area:.2f}"


# ----------------- הרצה -----------------

if __name__ == "__main__":
	app.run_server(debug=True)