import plotly.offline as py
import plotly.graph_objects as go

# Your data
experiment_names = ["Exp 1", "Exp 2", "Exp 3", "Exp 4", "Exp 5", "Exp 6"]
experiment_desc = ["100% InVivo", "100% synthetic", "Mixed 25% Synth", "Mixed 50% Synth", "Mixed 75% Synth" , "Mixed 100% Synth"]
epe = [1.5204,	2.6669,	1.2810,	1.1612,	1.1843,	0.8904]
syntheticAmount = [0,10582,2645,5292,7937,10582]
invivoAmount = [5292,0,5292,5292,5292,5292]

# Create a bar chart
fig = go.Figure(data=[
 go.Bar(
     name='Synthetic',
     x=experiment_names,
     y=syntheticAmount,
     error_y=dict(
         type='data',
         array=epe
     ),
     marker=dict(color='blue')
 ),
 go.Bar(
     name='In Vivo',
     x=experiment_names,
     y=invivoAmount,
     error_y=dict(
         type='data',
         array=epe
     ),
     marker=dict(color='magenta')
 )
])

# Update layout for better visualization
fig.update_layout(
 title='Error over Synthetic and In Vivo Data Amount',
 xaxis_title='Experiment',
 yaxis_title='Total Sum of Images',
 legend_title='Experiment Type',
 hovermode='x unified',
 hovertemplate=
 '<i>Experiment:</i> %{x}<br>'+
 '<i>Amount:</i> %{y}<br>'+
 '<i>Error:</i> %{error_y.array}<extra></extra>'
)

py.plot(fig, filename='figure.html')
