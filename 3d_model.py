import numpy as np
from scipy.spatial import ConvexHull
from ordered_set import OrderedSet
import noise
import plotly.graph_objects as go
import dash
from dash import dcc, html
import dash_daq as daq
from dash.dependencies import Input, Output
from utils.bresenham import line

""" octaves -- specifies the number of passes for generating fBm noise,
    defaults to 1 (simple noise).
    
    persistence -- specifies the amplitude of each successive octave relative
    to the one below it. Defaults to 0.5 (each higher octave's amplitude
    is halved). Note the amplitude of the first pass is always 1.0.
    
    lacunarity -- specifies the frequency of each successive octave relative
    to the one below it, similar to persistence. Defaults to 2.0.
    
    repeatx, repeaty, repeatz -- specifies the interval along each axis when 
    the noise values repeat. This can be used as the tile size for creating 
    tileable textures
    
    base -- specifies a fixed offset for the input coordinates. Useful for
    generating different noise textures with the same repeat interval
 """


class Environment:
    def __init__(
        self,
        dimensions=(1000, 1000),
        shape=(50, 50),
        scale=100.0,
        octaves=3,
        persistence=0.5,
        lacunarity=2.0,
    ):
        self.apply_cnt = 0
        self.selectMode = False
        self.removedPoints = []
        self.selectedPoints = OrderedSet()
        self.selectedArea = []
        self.cache = np.zeros(shape)
        self.dimensions = dimensions
        self.shape = shape
        self.terrain = np.zeros(shape)
        self.x_values = np.linspace(0, dimensions[0], shape[0])
        self.y_values = np.linspace(0, dimensions[1], shape[1])
        self.x, self.y = np.meshgrid(self.x_values, self.y_values)
        self.generate_perlin_noise(
            self.x, self.y, scale, octaves, persistence, lacunarity
        )

    def find_index(self, x_pos, y_pos):
        x_diff = np.abs(np.subtract.outer(x_pos, self.x_values))
        y_diff = np.abs(np.subtract.outer(y_pos, self.y_values))
        return np.column_stack((np.argmin(x_diff, axis=1), np.argmin(y_diff, axis=1)))

    def generate_perlin_noise(self, x, y, scale, octaves, persistence, lacunarity):
        for rx, ry in zip(x, y):
            idx = self.find_index(np.array(rx), np.array(ry))
            for i in idx:
                self.terrain[i[0]][i[1]] = noise.pnoise2(
                    rx[i[0]] / scale,
                    ry[i[1]] / scale,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    repeatx=1024,
                    repeaty=1024,
                    base=0,
                )

    def apply_value(self, value, kernel=None):
        if kernel is None:
            self.cache[self.selectedArea[:, 0], self.selectedArea[:, 1]] += value
    

    def find_selected_area(self):
        def point_inside_polygon(point, vertices):
            x, y = point
            n = len(vertices)
            inside = False
            p1x, p1y = vertices[0]
            for i in range(n + 1):
                p2x, p2y = vertices[i % n]
                if (y > min(p1y, p2y)) and (y <= max(p1y, p2y)):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xinters:
                                inside = not inside
                p1x, p1y = p2x, p2y
            return inside

        A = np.array(list(self.selectedPoints))
        if len(self.selectedPoints) <= 1:
            self.selectedArea = self.find_index(np.array(A[:, 0]), np.array(A[:, 1]))
        elif len(self.selectedPoints) == 2:
            idx = self.find_index(np.array(A[:, 0]), np.array(A[:, 1]))
            self.selectedArea = np.array(
                line(idx[0, 0], idx[0, 1], idx[1, 0], idx[1, 1])
            )
        else:
            A = np.array(list(self.selectedPoints))[:, :2]
            hull = ConvexHull(A)
            area_vertices = self.find_index(
                A[hull.vertices][:, 0], A[hull.vertices][:, 1]
            )
            grid = self.find_index(self.x.flatten(), self.y.flatten())
            is_inside = np.array(
                [point_inside_polygon(point, area_vertices) for point in grid]
            )
            self.selectedArea = np.column_stack((grid[:, 0], grid[:, 1]))[is_inside]


dimensions = (1000, 1000)
shape = (50, 50)
scale = 200.0
world = Environment(dimensions=dimensions, shape=shape, scale=scale)


# CREATING 3D TERRAIN MODEL
fig = go.Figure()
fig.add_surface(
    name="terrain",
    hoverinfo="none",
    z=world.terrain,
    x=world.x,
    y=world.y,
)

layout = go.Layout(
    scene=dict(
        aspectratio=dict(x=2, y=2, z=0.5),
        xaxis=dict(range=[0, dimensions[0]], showspikes=False),
        yaxis=dict(range=[0, dimensions[1]], showspikes=False),
        zaxis=dict(showspikes=False),
    ),
    uirevision="Don't change",
    autosize=True,
    hovermode=False,
    scattermode="group",
    spikedistance=0,
)
fig.update_layout(layout)


# app info
app = dash.Dash(__name__)
app.layout = html.Div(
    [
        dcc.Graph(
            id="figure1",
            figure=fig,
            style={"width": "100%", "height": "70%"},
            clear_on_unhover=True,
        ),
        html.Div(
            [
                html.Div(
                    daq.BooleanSwitch(id="select", on=False, label="Select"),
                    style={"display": "inline-block"},
                ),
                html.Div([
                    html.Button(
                        id="-",
                        children="<-",
                        style={"display": "flex"},
                    ),
                    html.Button(
                        id="+",
                        children="->",
                        style={"display": "flex"},
                    ),
                    ],
                    style={"display": "inline-block"}
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Button(
                                    id="apply",
                                    n_clicks=0,
                                    children="Apply",
                                ),
                                daq.BooleanSwitch(id="overview", on=False),
                            ],
                            style={"display": "flex"},
                        ),
                        dcc.Input(id="elevation", placeholder="", type="text"),
                    ],
                    style={"display": "inline-block"},
                ),
            ]
        ),
        html.Pre(id="data"),
    ],
    style={"width": "100vw", "height": "100vh"},
)


@app.callback(
    Output("figure1", "figure", allow_duplicate=True),
    Output("elevation", "value", allow_duplicate=True),
    Input("elevation", "value"),
    Input("apply", "n_clicks"),
    Input("overview", "on"),
    prevent_initial_call=True,
)
def show_changes(value, apply, overview):
    click = apply > world.apply_cnt
    world.apply_cnt = apply
    try:
        _value = float(value)
    except:
        if value == '-':
            return fig, '-'
        return fig, ""
    if len(world.selectedArea) >= 1:
        if overview:
            world.apply_value(_value)
            colors = None
            if world.selectMode:
                colors = np.full(world.shape, None)
                colors[world.selectedArea[:, 0], world.selectedArea[:, 1]] = "blue"
            fig.update_traces(
                surfacecolor=colors,
                z=world.terrain + world.cache,
                selector=dict(name="terrain"),
            )
            if click:
                world.terrain += world.cache
                world.cache = np.zeros(world.shape)
                value = ""
        else:
            colors = None
            world.cache = np.zeros(world.shape)
            value = ""
            if world.selectMode:
                colors = np.full(world.shape, None)
                colors[world.selectedArea[:, 0], world.selectedArea[:, 1]] = "blue"
            fig.update_traces(
                surfacecolor=colors,
                z=world.terrain,
                selector=dict(name="terrain"),
            )
    return fig, value


@app.callback(
    Output("figure1", "figure", allow_duplicate=True),
    Output("data", "children", allow_duplicate=True),
    Input("+", "n_clicks"),
    prevent_initial_call=True,
)
def redo_select(click):
    if world.selectMode and len(world.removedPoints) >= 1:
        removed = world.removedPoints.pop()
        world.selectedPoints.add(removed)
        world.find_selected_area()
        colors = np.full(world.shape, None)
        colors[world.selectedArea[:, 0], world.selectedArea[:, 1]] = "blue"
        fig.update_traces(surfacecolor=colors, selector=dict(name="terrain"))
    if len(world.selectedPoints) >= 1:
        return fig, str(world.selectedPoints[-1])
    return fig, ""


@app.callback(
    Output("figure1", "figure", allow_duplicate=True),
    Output("select", "on", allow_duplicate=True),
    Output("data", "children", allow_duplicate=True),
    Input("-", "n_clicks"),
    prevent_initial_call=True,
)
def undo_select(click):
    if world.selectMode and len(world.selectedPoints) >= 1:
        world.removedPoints.append(world.selectedPoints.pop())
        if len(world.selectedArea) > 1:
            world.find_selected_area()
            colors = np.full(world.shape, None)
            colors[world.selectedArea[:, 0], world.selectedArea[:, 1]] = "blue"
            fig.update_traces(surfacecolor=colors, selector=dict(name="terrain"))
        else:
            world.selectMode = False
            world.selectedArea = []
            fig.update_traces(
                surfacecolor=None,
                selector=dict(name="terrain"),
            )
    if world.selectMode:
        return fig, world.selectMode, str(world.selectedPoints[-1])
    return fig, world.selectMode, ''


@app.callback(
    Output("figure1", "figure", allow_duplicate=True),
    Output("select", "color"),
    Input("select", "on"),
    prevent_initial_call=True,
)
def select_toggle(on):
    if on:
        world.selectMode = True
        colors = None
        if len(world.selectedArea) >= 1:
            colors = np.full(world.shape, None)
            colors[world.selectedArea[:, 0], world.selectedArea[:, 1]] = "blue"
        fig.update_traces(surfacecolor=colors, selector=dict(name="terrain"))
        return fig, "green"
    world.selectMode = False
    fig.update_traces(
        surfacecolor=None,
        selector=dict(name="terrain"),
    )
    return fig, "gray"


@app.callback(
    Output("figure1", "figure", allow_duplicate=True),
    Output("data", "children", allow_duplicate=True),
    Input("figure1", "clickData"),
    prevent_initial_call=True,
)
def select_from_graph(clickData):
    if world.selectMode and clickData is not None:
        # x and y are inverted when getting from the graph ¯\_(ツ)_/¯
        x = clickData["points"][0]["y"]
        y = clickData["points"][0]["x"]
        z = clickData["points"][0]["z"]
        world.selectedPoints.add((x, y, z))
        world.find_selected_area()
        world.removedPoints = []
        if len(world.selectedArea) >= 1:
            colors = np.full(world.shape, None)
            colors[world.selectedArea[:, 0], world.selectedArea[:, 1]] = "blue"
            fig.update_traces(surfacecolor=colors, selector=dict(name="terrain"))

    if len(world.selectedPoints):
        return fig, str(world.selectedPoints[-1])
    return fig, ""


app.run(
    host="0.0.0.0",
    port="8050",
    dev_tools_ui=True,
    dev_tools_hot_reload=True,
    debug=True,
)
