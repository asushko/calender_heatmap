import dash
from dash import html
import os

# Create Dash app
app = dash.Dash(__name__)
server = app.server  # Expose Flask server for Gunicorn

# Layout with simple text
app.layout = html.Div("Gunicorn is running")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=True, host="0.0.0.0", port=port)
