



# Dash app initialization
app = Dash(__name__)

app.layout = html.Div([
    html.H1("GDM Risk Assessment Dashboard"),

    # Dropdown to select the column
    html.Label("Select a column to view box plot:"),
    dcc.Dropdown(
        id='column-dropdown',
        options=[{'label': col, 'value': col} for col in all_columns],
        value=all_columns[0],  # Default selection
        clearable=False
    ),

    # Boxplot for the selected column
    dcc.Graph(id='boxplot-graph')
])


# Callback to update the box plot based on dropdown selection
@app.callback(
    Output('boxplot-graph', 'figure'),
    [Input('column-dropdown', 'value')]
)
def update_boxplot(selected_column):
    # Create the box plot for the selected column
    boxplot = go.Figure(
        data=[go.Box(y=GDM7th[selected_column], name=selected_column)],
        layout=go.Layout(
            title=f"Box plot for {selected_column}",
            yaxis_title='Value',
            xaxis_title='Feature'
        )
    )
    return boxplot


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)