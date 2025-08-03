import os
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State

root_dir_hist = "d:\\theta_data\\hist"

root_dir = r"d:\theta_data\hist\heatmap_600"

def generate_filename(ticker, start_date, end_date, leg1_exp, leg2_exp, ivl):
    return os.path.join(root_dir_hist, f"heatmap_{ivl}", f"{ticker}_{start_date}_{end_date}_leg_start-{leg1_exp}_leg_end-{leg2_exp}.pkl")

def mark_cells(df, step):
    value_cols = [c for c in df.columns if '-' in c]
    df_marked = df.copy()
    prev_row = None
    for idx, row in df[value_cols].iterrows():
        if prev_row is None:
            for col in value_cols:
                df_marked.at[idx, col + '_mark'] = 0
        else:
            for col in value_cols:
                diff = row[col] - prev_row[col]
                if diff > step:
                    df_marked.at[idx, col + '_mark'] = 1  # рост
                elif diff < -step:
                    df_marked.at[idx, col + '_mark'] = -1  # падение
                else:
                    df_marked.at[idx, col + '_mark'] = 0
        prev_row = row
    return df_marked

def plot_extrinsic_heatmap_monochrom(df, title_vars, step):
    value_cols = [c for c in df.columns if '-' in c and not c.endswith('_mark')]
    z_data = df[value_cols].astype(float).to_numpy()
    x_data = value_cols
    y_data = df['ms_of_day2']
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=x_data,
        y=y_data,
        colorscale=[[0, 'white'], [1, 'darkblue']],
        colorbar=dict(title='Value')
    ))
    mark_x_red, mark_y_red = [], []
    mark_x_green, mark_y_green = [], []
    for i, y in enumerate(y_data):
        for j, x in enumerate(x_data):
            mark_val = df.iloc[i][x + '_mark']
            if mark_val == 1:
                mark_x_red.append(x)
                mark_y_red.append(y)
            elif mark_val == -1:
                mark_x_green.append(x)
                mark_y_green.append(y)
    fig.add_trace(go.Scatter(x=mark_x_red, y=mark_y_red, mode='markers',
                             marker=dict(color='red', size=2),
                             name='Increase',
                             hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=mark_x_green, y=mark_y_green, mode='markers',
                             marker=dict(color='green', size=2),
                             name='Decrease',
                             hoverinfo='skip'))
    fig.update_layout(
        autosize=True,
        width=None,
        height=900,
        title=dict(text=f"Local slope of the IV term structure for call options <br>{title_vars}", x=0.5, xanchor='center'),
        xaxis_title="Difference",
        yaxis_title="ms_of_day2",
        margin=dict(l=40, r=40, t=80, b=80),
        xaxis=dict(showline=False, mirror=False, zeroline=False, tickangle=90),
        yaxis=dict(autorange='reversed', showline=False, mirror=False, zeroline=False)
    )
    return fig



import plotly.graph_objects as go
import numpy as np

def plot_extrinsic_3d_surface(df, title_vars, step):
    value_cols = [c for c in df.columns if '-' in c and not c.endswith('_mark')]
    z_data = df[value_cols].astype(float).to_numpy()
    x_data = np.arange(len(value_cols))  # numeric x-axis for 3D
    y_data = df['ms_of_day2'].to_numpy()

    # Create meshgrid for 3D surface
    X, Y = np.meshgrid(x_data, y_data)

    fig = go.Figure(data=[go.Surface(
        z=z_data,
        x=X,
        y=Y,
        colorscale=[[0, 'white'], [1, 'darkblue']],
        colorbar=dict(title='Value')
    )])

    # Markers for Increase/Decrease
    mark_x_red, mark_y_red, mark_z_red = [], [], []
    mark_x_green, mark_y_green, mark_z_green = [], [], []

    for i, y in enumerate(y_data):
        for j, x in enumerate(value_cols):
            mark_val = df.iloc[i][x + '_mark']
            if mark_val == 1:
                mark_x_red.append(j)
                mark_y_red.append(y)
                mark_z_red.append(z_data[i, j])
            elif mark_val == -1:
                mark_x_green.append(j)
                mark_y_green.append(y)
                mark_z_green.append(z_data[i, j])

    # Add 3D scatter for markers
    fig.add_trace(go.Scatter3d(
        x=mark_x_red, y=mark_y_red, z=mark_z_red,
        mode='markers', marker=dict(color='red', size=3),
        name='Increase'
    ))
    fig.add_trace(go.Scatter3d(
        x=mark_x_green, y=mark_y_green, z=mark_z_green,
        mode='markers', marker=dict(color='green', size=3),
        name='Decrease'
    ))

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Local slope of the IV term structure for call options <br>{title_vars}",
            x=0.5, xanchor='center'
        ),
        scene=dict(
            xaxis=dict(title='Difference', tickvals=np.arange(len(value_cols)), ticktext=value_cols),
            yaxis=dict(title='ms_of_day2', autorange='reversed'),
            zaxis=dict(title='Value')
        ),
        margin=dict(l=0, r=0, t=80, b=0),
        height=900
    )

    return fig



def create_app():
    app = Dash(__name__)
    files = sorted([f for f in os.listdir(root_dir) if f.endswith(".pkl")])

    app.layout = html.Div([
        html.Div([
            dcc.Dropdown(id='filename',
                         options=[{'label': f, 'value': f} for f in files],
                         placeholder='Select file',
                         style={'min-width': '600px'}),
            dcc.Input(id='step', type='number', value=0.25, placeholder='Step', step=0.05,
                      style={'width': '150px'})
        ], style={'display': 'flex', 'gap': '10px', 'flex-wrap': 'wrap', 'margin-bottom': '20px'}),
        dcc.Loading(dcc.Graph(id='heatmap', style={'height': '900px'}))
    ])

    @app.callback(Output('heatmap', 'figure'),
                  Input('filename', 'value'),
                  Input('step', 'value'))
    def update_plot(filename, step):
        if not filename:
            return {}
        path = os.path.join(root_dir, filename)
        df = pd.read_pickle(path)
        df = mark_cells(df, step)
        title_vars = filename.replace(".pkl", "")
        return plot_extrinsic_heatmap_monochrom(df, title_vars, step)
        # return plot_extrinsic_3d_surface(df, title_vars, step)

    return app
if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=8050)

