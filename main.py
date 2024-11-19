import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly

def make_3d_grid(size: int = 10, random: bool = False, coverage: float = 0.1) -> np.ndarray:
    grid = np.zeros((size, size, size), dtype=int)

    if not random and size > 5:
        grid[1,1,1] = 1
        # grid[4,5] = 1
        # grid[5,5] = 1

        # grid[8,8] = 1
        # grid[9,9] = 1

        # grid[2,8] = 1
        # grid[2,7] = 1
        # grid[1,8] = 1
        # grid[1,7] = 1
    else:
        total_cells = size * size * size
        num_obstacles = int(total_cells * coverage)

        obstacles = set()
        while len(obstacles) < num_obstacles:
            row = np.random.randint(0, size)
            col = np.random.randint(0, size)
            high = np.random.randint(0, size)
            obstacles.add((row, col, high))

        for row, col, high in obstacles:
            grid[row, col, high] = 1

    return grid



def plot_3d_grid(grid,):
    if not (grid.ndim == 3 and np.all((grid == 0) | (grid == 1))):
        raise ValueError("Input must be a 3D numpy array with values 0 or 1.")



    path = [(0,0,0), (1,1,1), (2,2,2), (3,3,3)]

    main = go.Scatter(x=[0, 1], y=[0, 1])

    layout = go.Layout(
        xaxis=dict(range=[0, 5], autorange=False),
        yaxis=dict(range=[0, 5], autorange=False),
        title=dict(text="Start Title"),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(
                label="Play",
                method="animate",
                args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)]
            )]
        )],
        sliders=[{
            'steps': [
                {
                    'args': [
                        [f'Frame {i+1}'], 
                        {'frame': {'duration': 500, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 300}}
                    ],
                    'label': f'Frame {i+1}',
                    'method': 'animate'
                } for i in range(3)  # Number of frames
            ],
            'currentvalue': {
                'prefix': 'Frame:',
                'visible': True,
                'xanchor': 'center',
                'font': {'size': 20}
            }
        }]
    )

    frames = [
        go.Frame(data=[go.Scatter(x=[1, 2], y=[1, 2])], name='Frame 1'),
        go.Frame(data=[go.Scatter(x=[1, 4], y=[1, 4])], name='Frame 2'),
        go.Frame(data=[go.Scatter(x=[3, 4], y=[3, 4])], layout=go.Layout(title_text="End Title"), name='Frame 3')
    ]

    

    fig = go.Figure(
    data=[main],
    layout= layout,
    frames=frames
    )

    fig.show()




    # Get the coordinates of 1s and 0s
    # dark_indices = np.argwhere(grid == 1)
    # light_indices = np.argwhere(grid == 0)

    # Initial figure
    # fig = go.Figure()
    
    # Add the points for 1s
    # if dark_indices.size > 0:
    #     dark_points = go.Scatter3d(
    #         x=dark_indices[:, 0],
    #         y=dark_indices[:, 1],
    #         z=dark_indices[:, 2],
    #         mode='markers+text',
    #         text=[f"({x}, {y}, {z})" for x, y, z in dark_indices],
    #         textposition='top center',
    #         marker=dict(size=5, color='black'),
    #         name='1'
    #     )
        # fig.add_trace(dark_points)

    # if light_indices.size > 0:
    #     light_points = go.Scatter3d(
    #         x=light_indices[:, 0],
    #         y=light_indices[:, 1],
    #         z=light_indices[:, 2],
    #         mode='markers+text',
    #         text=[f"({x}, {y}, {z})" for x, y, z in light_indices],
    #         textposition='top center',
    #         marker=dict(size=5, color='lightgray'),
    #         name='0'
    #     )
    # fig.add_trace(light_points)



    # Frames for animation
    # frames = []
    # for i in range(10):
    #     x_start, y_start, z_start = i, i, i
    #     x_end, y_end, z_end = i + 1, i + 1, i + 1

        # Add the cone (arrow)
        # cone = go.Cone(
        #     x=[x_start + 0.5], y=[y_start + 0.5], z=[z_start + 0.5],
        #     u=[1], v=[1], w=[1],
        #     colorscale=[[0, 'red'], [1, 'red']],
        #     sizemode="absolute", sizeref=0.3, anchor="tail",
        #     name="Arrow"
        # )
        # Add the connecting line
        # line_frame = go.Scatter3d(
        #     x=[x_start, x_end],
        #     y=[y_start, y_end],
        #     z=[z_start, z_end],
        #     mode='lines',
        #     line=dict(color='blue', width=4),
        #     name="Line"
        # )


if __name__=="__main__":
    grid = make_3d_grid(random=True)
    print(grid)

    plot_3d_grid(grid)
