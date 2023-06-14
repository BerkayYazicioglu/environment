import numpy as np
import plotly.graph_objects as go
from os import walk
import dash
from dash import dcc, html
import dash_daq as daq
from dash.dependencies import Input, Output
from environment import Environment

world = Environment(shape=(1, 1))
defaults = dict(
    size_x=1000,
    size_y=1000,
    grid_x=100,
    grid_y=100,
    scale=100,
    octaves=6,
    pers=0.5,
    lacu=0.5,
    filename="env",
)
create_clicked = 0
load_clicked = 0
save_clicked = 0
apply_clicked = 0
selectMode = False

# CREATING 3D TERRAIN MODEL
fig = go.Figure()
fig.add_surface(
    name="terrain",
    hoverinfo="none",
    z=[],
    x=[],
    y=[],
)

plot = go.Figure()
plot.add_heatmap(
    name="plot",
    hoverinfo="none",
    z=[],
    x=[],
    y=[],
)


# app info
app = dash.Dash(__name__)
app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    dcc.Graph(
                        id="figure1",
                        figure=fig,
                        style={
                            "display": "inline-block",
                            "width": "100%",
                            "height": "100%",
                            "margin": 0,
                        },
                        className="six columns",
                        clear_on_unhover=True,
                    ),
                    style={
                        "display": "inline-block",
                        "width": "50%",
                        "height": "100%",
                        "margin": 0,
                    },
                ),
                html.Div(
                    dcc.Graph(
                        id="plot",
                        figure=plot,
                        style={
                            "display": "inline-block",
                            "width": "100%",
                            "height": "100%",
                            "margin": 0,
                        },
                        className="six columns",
                        clear_on_unhover=True,
                    ),
                    style={
                        "display": "inline-block",
                        "width": "50%",
                        "height": "100%",
                        "margin": 0,
                    },
                ),
            ],
            className="row",
            style={"width": "100vw", "height": "70vh", "display": "flex"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("", id="filename", style={"display": "flex"}),
                        html.Button(
                            id="create",
                            n_clicks=0,
                            children="Create",
                            style={"display": "flex"},
                        ),
                    ],
                    style={
                        "display": "inline-block",
                        "font-size": "12px",
                    },
                ),
                html.Div(
                    [
                        dcc.Input(
                            id="file_input",
                            value=defaults["filename"],
                            style={
                                "display": "flex",
                                "width": 50,
                                "margin-left": "15px",
                            },
                            debounce=True,
                            type="text",
                        ),
                    ],
                    style={"display": "inline-block", "font-size": "12px"},
                ),
                html.Div(
                    [
                        html.Label(
                            "x(m)", style={"display": "flex", "margin-left": "15px"}
                        ),
                        dcc.Input(
                            id="size_x",
                            value=defaults["size_x"],
                            style={
                                "display": "flex",
                                "width": 50,
                                "margin-left": "15px",
                            },
                            debounce=True,
                            type="number",
                            min=0,
                        ),
                    ],
                    style={"display": "inline-block", "font-size": "12px"},
                ),
                html.Div(
                    [
                        html.Label("y(m)", style={"display": "flex"}),
                        dcc.Input(
                            id="size_y",
                            value=defaults["size_y"],
                            style={"display": "flex", "width": 50},
                            debounce=True,
                            type="number",
                            min=0,
                        ),
                    ],
                    style={"display": "inline-block", "font-size": "12px"},
                ),
                html.Div(
                    [
                        html.Label("x grid", style={"display": "flex"}),
                        dcc.Input(
                            id="grid_x",
                            value=defaults["grid_x"],
                            style={"display": "flex", "width": 50},
                            debounce=True,
                            type="number",
                            min=0,
                        ),
                    ],
                    style={"display": "inline-block", "font-size": "12px"},
                ),
                html.Div(
                    [
                        html.Label("y grid", style={"display": "flex"}),
                        dcc.Input(
                            id="grid_y",
                            value=defaults["grid_y"],
                            style={"display": "flex", "width": 50},
                            debounce=True,
                            type="number",
                            min=0,
                        ),
                    ],
                    style={"display": "inline-block", "font-size": "12px"},
                ),
                html.Div(
                    [
                        html.Label("scale", style={"display": "flex"}),
                        dcc.Input(
                            id="scale",
                            value=defaults["scale"],
                            style={"display": "flex", "width": 50},
                            debounce=True,
                            type="number",
                            min=0,
                            max=100,
                        ),
                    ],
                    style={"display": "inline-block", "font-size": "12px"},
                ),
                html.Div(
                    [
                        html.Label("octaves", style={"display": "flex"}),
                        dcc.Input(
                            id="octaves",
                            value=defaults["octaves"],
                            style={"display": "flex", "width": 50},
                            debounce=True,
                            type="number",
                            min=0,
                        ),
                    ],
                    style={"display": "inline-block", "font-size": "12px"},
                ),
                html.Div(
                    [
                        html.Label("pers", style={"display": "flex"}),
                        dcc.Input(
                            id="pers",
                            value=defaults["pers"],
                            style={"display": "flex", "width": 50},
                            debounce=True,
                            type="number",
                            step=0.1,
                            max=1,
                            min=0,
                        ),
                    ],
                    style={"display": "inline-block", "font-size": "12px"},
                ),
                html.Div(
                    [
                        html.Label("lacu", style={"display": "flex"}),
                        dcc.Input(
                            id="lacu",
                            value=defaults["lacu"],
                            style={"display": "flex", "width": 50},
                            debounce=True,
                            type="number",
                            step=0.1,
                            min=0,
                        ),
                    ],
                    style={"display": "inline-block", "font-size": "12px"},
                ),
            ],
        ),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Dropdown(
                            id="files", style={"display": "flex", "font-size": "12px"}
                        ),
                        html.Button(
                            id="load",
                            children="Load",
                            n_clicks=0,
                            style={"display": "flex"},
                        ),
                        html.Button(
                            id="save",
                            children="Save",
                            n_clicks=0,
                            style={"display": "flex"},
                        ),
                    ],
                    style={"display": "inline-block", "width": "100px"},
                ),
                html.Div(
                    [
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
                    style={"display": "inline-block"},
                ),
                html.Div(
                    daq.BooleanSwitch(id="select", on=False, label="Select"),
                    style={"display": "inline-block"},
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                daq.BooleanSwitch(id="overview", on=False),
                                html.Button(
                                    id="apply",
                                    n_clicks=0,
                                    children="Apply",
                                ),
                            ],
                            style={"display": "flex"},
                        ),
                        dcc.Input(
                            id="elevation", placeholder=0, debounce=True, type="number"
                        ),
                    ],
                    style={"display": "inline-block"},
                ),
                html.Div(
                    [
                        dcc.Slider(
                            min=0,
                            max=3,
                            value=0,
                            marks={
                                0: "elevation",
                                1: "roughness",
                                2: "destruction",
                                3: "population",
                            },
                            step=None,
                            id="mode",
                        ),
                        dcc.Slider(0, 1, marks=None, value=0, id="range"),
                    ],
                    style={"display": "inline_block", "width": "50vh"},
                ),
            ]
        ),
        html.Pre(id="data"),
    ],
    style={"width": "100vw", "height": "100vh"},
)


@app.callback(
    Output("plot", "figure", allow_duplicate=True),
    Input("mode", "value"),
    Input("range", "value"),
    prevent_initial_call=True,
)
def sliders(mode, range):
    global world, selectMode
    modes = ("terrain", "roughness", "destruction", "population")
    if mode != modes.index(world.mode):
        world.mode = modes[mode]
        plot.update_traces(z=world.data[world.mode], selector=dict(name="plot"))
    if selectMode and len(world.selectedArea) > 0:
        world.data[world.mode][world.selectedArea[:, 0], world.selectedArea[:, 1]] = range
        plot.update_traces(z=world.data[world.mode], selector=dict(name="plot"))
    return plot


@app.callback(
    Output("figure1", "figure", allow_duplicate=True),
    Output("plot", "figure", allow_duplicate=True),
    Input("scale", "value"),
    Input("octaves", "value"),
    Input("pers", "value"),
    Input("lacu", "value"),
    prevent_initial_call=True,
)
def change_noise(scale, octaves, pers, lacu):
    global world, selectMode
    if selectMode and len(world.selectedArea) > 0:
        world.generate_perlin_noise(
            world.x_values[world.selectedArea[:, 0]],
            world.y_values[world.selectedArea[:, 1]],
            scale,
            octaves,
            pers,
            lacu,
        )
        fig.update_traces(z=world.data["terrain"], selector=dict(name="terrain"))
        plot.update_traces(z=world.data[world.mode], selector=dict(name="plot"))
    return fig, plot


@app.callback(
    Output("figure1", "figure", allow_duplicate=True),
    Output("plot", "figure", allow_duplicate=True),
    Output("files", "options", allow_duplicate=True),
    Output("size_x", "value", allow_duplicate=True),
    Output("size_y", "value", allow_duplicate=True),
    Output("grid_x", "value", allow_duplicate=True),
    Output("grid_y", "value", allow_duplicate=True),
    Output("scale", "value", allow_duplicate=True),
    Output("octaves", "value", allow_duplicate=True),
    Output("pers", "value", allow_duplicate=True),
    Output("lacu", "value", allow_duplicate=True),
    Output("filename", "children", allow_duplicate=True),
    Input("files", "search_value"),
    Input("files", "value"),
    Input("load", "n_clicks"),
    Input("save", "n_clicks"),
    prevent_initial_call=True,
)
def select_file(search, value, load, save):
    global load_clicked, save_clicked, world

    if value and load > load_clicked:
        load_clicked = load
        world = Environment(filename=value.split(".")[0], load=True)
        fig.update_traces(
            x=world.x, y=world.y, z=world.data["terrain"], selector=dict(name="terrain")
        )
        plot.update_traces(z=world.data[world.mode], selector=dict(name="plot"))

        layout = go.Layout(
            scene=dict(
                aspectratio=dict(x=2, y=2, z=0.5),
                xaxis=dict(range=[0, world.dimensions[0]], showspikes=False),
                yaxis=dict(range=[0, world.dimensions[1]], showspikes=False),
                zaxis=dict(showspikes=False),
            ),
            uirevision="Don't change",
            autosize=True,
            hovermode=False,
            scattermode="group",
            spikedistance=0,
        )
        fig.update_layout(layout)

    if save > save_clicked and world.filename is not None:
        save_clicked = save
        world.save()

    return (
        fig,
        plot,
        [i for i in next(walk("data"), (None, None, []))[2]],
        world.dimensions[0],
        world.dimensions[1],
        world.shape[0],
        world.shape[1],
        world.scale,
        world.octaves,
        world.persistence,
        world.lacunarity,
        world.filename,
    )


@app.callback(
    Output("figure1", "figure", allow_duplicate=True),
    Output("plot", "figure", allow_duplicate=True),
    Output("filename", "children", allow_duplicate=True),
    Input("create", "n_clicks"),
    Input("file_input", "value"),
    Input("size_x", "value"),
    Input("size_y", "value"),
    Input("grid_x", "value"),
    Input("grid_y", "value"),
    Input("scale", "value"),
    Input("octaves", "value"),
    Input("pers", "value"),
    Input("lacu", "value"),
    prevent_initial_call=True,
)
def create(click, file, size_x, size_y, grid_x, grid_y, scale, octaves, pers, lacu):
    global create_clicked, world

    if click > create_clicked:
        create_clicked = click
        input = (size_x, size_y, grid_x, grid_y, scale, octaves, pers, lacu)
        if None in input:
            return fig, world.filename
        if file is None or file.replace(" ", "") == "":
            file = defaults["filename"]
        world = Environment(
            dimensions=(size_x, size_y),
            shape=(grid_x, grid_y),
            scale=scale,
            octaves=octaves,
            persistence=pers,
            lacunarity=lacu,
            filename=file.replace(" ", ""),
        )

        fig.update_traces(
            x=world.x, y=world.y, z=world.data["terrain"], selector=dict(name="terrain")
        )
        plot.update_traces(z=world.data[world.mode], selector=dict(name="plot"))

        layout = go.Layout(
            scene=dict(
                aspectratio=dict(x=2, y=2, z=0.5),
                xaxis=dict(range=[0, size_x], showspikes=False),
                yaxis=dict(range=[0, size_y], showspikes=False),
                zaxis=dict(showspikes=False),
            ),
            uirevision="Don't change",
            autosize=True,
            hovermode=False,
            scattermode="group",
            spikedistance=0,
        )
        fig.update_layout(layout)
    return fig, plot, world.filename


@app.callback(
    Output("figure1", "figure", allow_duplicate=True),
    Output("plot", "figure", allow_duplicate=True),
    Input("apply", "n_clicks"),
    prevent_initial_call=True,
)
def apply_changes(apply):
    global world, apply_clicked
    click = apply > apply_clicked
    apply_clicked = apply
    if click:
        world.data["terrain"] += world.cache
        world.cache = np.zeros(world.shape)
        fig.update_traces(
            z=world.data["terrain"],
            selector=dict(name="terrain"),
        )
        plot.update_traces(z=world.data[world.mode], selector=dict(name="plot"))
    return fig, plot


@app.callback(
    Output("figure1", "figure", allow_duplicate=True),
    Input("elevation", "value"),
    Input("overview", "on"),
    prevent_initial_call=True,
)
def show_changes(value, overview):
    global world, selectMode

    if len(world.selectedArea) >= 1:
        if overview and value is not None:
            world.apply_value(value)
            colors = None
            if selectMode:
                colors = np.full(world.shape, None)
                colors[world.selectedArea[:, 0], world.selectedArea[:, 1]] = "blue"
            fig.update_traces(
                surfacecolor=colors,
                z=world.data["terrain"] + world.cache,
                selector=dict(name="terrain"),
            )
        else:
            colors = None
            world.cache = np.zeros(world.shape)
            if selectMode:
                colors = np.full(world.shape, None)
                colors[world.selectedArea[:, 0], world.selectedArea[:, 1]] = "blue"
            fig.update_traces(
                surfacecolor=colors,
                z=world.data["terrain"],
                selector=dict(name="terrain"),
            )
    return fig


@app.callback(
    Output("figure1", "figure", allow_duplicate=True),
    Input("+", "n_clicks"),
    prevent_initial_call=True,
)
def redo_select(click):
    global world, selectMode
    if selectMode and len(world.removedPoints) >= 1:
        removed = world.removedPoints.pop()
        world.selectedPoints.add(removed)
        world.find_selected_area()
        colors = np.full(world.shape, None)
        colors[world.selectedArea[:, 0], world.selectedArea[:, 1]] = "blue"
        fig.update_traces(surfacecolor=colors, selector=dict(name="terrain"))
    return fig


@app.callback(
    Output("figure1", "figure", allow_duplicate=True),
    Output("select", "on", allow_duplicate=True),
    Input("-", "n_clicks"),
    prevent_initial_call=True,
)
def undo_select(click):
    global world, selectMode
    if selectMode and len(world.selectedPoints) >= 1:
        world.removedPoints.append(world.selectedPoints.pop())
        if len(world.selectedArea) > 1:
            world.find_selected_area()
            colors = np.full(world.shape, None)
            colors[world.selectedArea[:, 0], world.selectedArea[:, 1]] = "blue"
            fig.update_traces(surfacecolor=colors, selector=dict(name="terrain"))
        else:
            selectMode = False
            world.selectedArea = []
            fig.update_traces(
                surfacecolor=None,
                selector=dict(name="terrain"),
            )
    return fig, selectMode


@app.callback(
    Output("figure1", "figure", allow_duplicate=True),
    Output("select", "color"),
    Input("select", "on"),
    prevent_initial_call=True,
)
def select_toggle(on):
    global world, selectMode
    if on:
        selectMode = True
        colors = None
        if len(world.selectedArea) >= 1:
            colors = np.full(world.shape, None)
            colors[world.selectedArea[:, 0], world.selectedArea[:, 1]] = "blue"
        fig.update_traces(surfacecolor=colors, selector=dict(name="terrain"))
        return fig, "green"
    selectMode = False
    fig.update_traces(
        surfacecolor=None,
        selector=dict(name="terrain"),
    )
    return fig, "gray"


@app.callback(
    Output("figure1", "figure", allow_duplicate=True),
    Input("figure1", "clickData"),
    prevent_initial_call=True,
)
def select_from_graph(clickData):
    global world, selectMode
    if selectMode and clickData is not None:
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
    return fig


app.run(
    host="0.0.0.0",
    port="8050",
    dev_tools_ui=True,
    dev_tools_hot_reload=True,
    debug=True,
)
