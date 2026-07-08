import panel as pn
import numpy as np
import plotly.graph_objects as go
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, PointDrawTool, PolyDrawTool

pn.extension('plotly')

# -------------------------------------------------------------------------
# 1. SETUP BOKEH PLOT & DATA SOURCES
# -------------------------------------------------------------------------
# Set up a clean 2D coordinate space (0 to 100 for example)
p = figure(x_range=(0, 100), y_range=(0, 100), width=500, height=500,
           title="2D Scene Builder (Select element + Backspace to delete)",
           tools="pan,wheel_zoom,reset")

# Source for Points (Sinks / Point Sources)
point_source = ColumnDataSource(data=dict(x=[20, 50], y=[20, 80]))
point_renderer = p.scatter('x', 'y', source=point_source, color="red", size=12, alpha=0.8)

# Source for Lines (Mirrors / Slits)
# Bokeh expects a list of lists for multi-line coordinates: e.g., xs=[[x1, x2], [x3, x4]]
line_source = ColumnDataSource(data=dict(xs=[[30, 70]], ys=[[50, 50]]))
line_renderer = p.multi_line('xs', 'ys', source=line_source, color="blue", line_width=4, alpha=0.7)

# -------------------------------------------------------------------------
# 2. CONFIGURE EXCLUSIVE DRAW TOOLS
# -------------------------------------------------------------------------
# Point tool: Allows adding, dragging, and deleting single dots
draw_tool_points = PointDrawTool(renderers=[point_renderer], empty_value='point')

# Line tool: Allows adding lines, dragging endpoints, and deleting lines
# 'vertex_renderer' enables editing the endpoints of the lines after creation
draw_tool_lines = PolyDrawTool(renderers=[line_renderer], empty_value='line', 
                               vertex_renderer=point_renderer)

# Add tools to plot and activate them
p.add_tools(draw_tool_points, draw_tool_lines)
p.toolbar.active_tap = draw_tool_points 

# -------------------------------------------------------------------------
# 3. EXTRACT COORDINATES ON SIMULATE
# -------------------------------------------------------------------------
def get_scene_primitives():
    """Parses Bokeh structures into a clean list of dictionaries for your math engine."""
    primitives = []
    
    # Extract Points
    pt_data = point_source.data
    for x, y in zip(pt_data['x'], pt_data['y']):
        primitives.append({
            "type": "point",
            "coords": [float(x), float(y)]
        })
        
    # Extract Lines
    ln_data = line_source.data
    for xs, ys in zip(ln_data['xs'], ln_data['ys']):
        # Enforce that it must be a valid 2-point line segment
        if len(xs) == 2:
            primitives.append({
                "type": "line",
                "coords": [[float(xs[0]), float(xs[1])], [float(ys[0]), float(ys[1])]]
            })
            
    return primitives

# -------------------------------------------------------------------------
# 4. PANEL UI LAYOUT & CALLBACKS
# -------------------------------------------------------------------------
run_button = pn.widgets.Button(name='Simulate Wave Propagation', button_type='primary', sizing_mode='stretch_width')
output_text = pn.pane.Markdown("No data captured yet. Click Simulate.", styles={'background': '#f4f4f4', 'padding': '10px'})

def on_simulate(event):
    scene_data = get_scene_primitives()
    print("Extracted Scene Primitives:", scene_data) 

    import json
    formatted_json = json.dumps(scene_data, indent=2)

    # Format as a Markdown code block
    output_text.object = f"**Data Sent to Math Backend:**\n```json\n{formatted_json}\n```"

run_button.on_click(on_simulate)

# Placeholder Plotly figure
plotly_fig = go.Figure(data=go.Heatmap(z=np.zeros((100, 100))))
plotly_fig.update_layout(width=500, height=500, title="Wave Field Output")

# Constructing the Panel dashboard layout
layout = pn.Column(
    pn.Row(run_button),
    pn.Row(
        pn.Column("### 1. Design Scene", p),
        pn.Column("### 2. Output Array", output_text),
        pn.Column("### 3. Resulting Plot", pn.pane.Plotly(plotly_fig))
    )
)

layout.servable()
