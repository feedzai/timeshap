#  Copyright 2022 Feedzai
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import pandas as pd
import copy
import re
import altair as alt
from timeshap.plot.utils import multi_plot_wrapper


def plot_event_heatmap(event_data: pd.DataFrame,
                       ):
    """Plots local event explanations

    Parameters
    ----------
    event_data: pd.DataFrame
        Event global explanations
    """
    event_data = copy.deepcopy(event_data)
    # extract digit to order df by - this is redundant but gives security to the method
    event_data['idx'] = event_data['Feature'].apply(lambda x: event_data.shape[0] if x == 'Pruned Events' else int(re.findall(r'\d+', x)[0]) - 1)
    event_data = event_data.sort_values('idx')[['Shapley Value', 'Feature']]

    c_range = ["#5f8fd6",
               "#99c3fb",
               "#f5f5f5",
               "#ffaa92",
               "#d16f5b",
               ]

    event_data['row'] = event_data['Feature'].apply(lambda x: event_data.shape[0] if x == 'Pruned Events' else -eval(x.split(':')[0][6:]))
    event_data['column'] = event_data['Feature'].apply(lambda x: 1)
    event_data['rounded'] = event_data['Shapley Value'].apply(lambda x: round(x, 3))
    event_data['rounded_str'] = event_data['Shapley Value'].apply(lambda x: '0.000' if round(x, 3) == 0 else str(round(x, 3)))
    event_data['rounded_str'] = event_data['rounded_str'].apply(lambda x: f'{x}0' if len(x) == 4 else x)

    c = alt.Chart().encode(
        y=alt.Y('Feature:O',
                axis=alt.Axis(domain=False, labelFontSize=15, title='Event',
                              titleFontSize=15, titleX=-49),
                sort=list(event_data['Feature'].values), ),
    )

    a = c.mark_rect().encode(
        x=alt.X('column:O',
                axis=alt.Axis(title='Shapley Value', titleFontSize=15)),
        color=alt.Color('rounded', title=None,
                        legend=alt.Legend(gradientLength=225,
                                          gradientThickness=10, orient='right',
                                          labelFontSize=15),
                        scale=alt.Scale(domain=[-.5, .5], range=c_range))
    )
    b = c.mark_text(align='right', baseline='middle', dx=18, fontSize=15,
                    color='#798184').encode(
        x=alt.X('column:O',
                axis=alt.Axis(labels=False, title='Shapley Value', domain=False,
                              titleX=43)),
        text='rounded_str',
    )

    event_plot = alt.layer(a, b, data=event_data).properties(
        width=60,
        height=225
    )
    return event_plot


def plot_global_event(event_data: pd.DataFrame,
                      plot_parameters: dict = None,
                      ):
    """Plots global event explanations

    Parameters
    ----------
    event_data: pd.DataFrame
        Event global explanations

    plot_parameters: dict
        Dict containing optional plot parameters
            'height': height of the plot, default 150
            'width': width of the plot, default 360
            'axis_lims': plot Y domain, default [-0.3, 0.9]
            't_limit': number of events to plot, default -20
    """
    def plot(event_data: pd.DataFrame, plot_parameters: dict = None):
        event_data = copy.deepcopy(event_data)
        event_data = event_data[event_data['t (event index)'] < 1]
        event_data = event_data[['Shapley Value', 't (event index)']]



        avg_df = event_data.groupby('t (event index)').mean().reset_index()
     

        event_data['type'] = 'Shapley Value'
        avg_df['type'] = 'Mean'

        event_data = pd.concat([event_data, avg_df], axis=0, ignore_index=True)

        if plot_parameters is None:
            plot_parameters = {}

        height = plot_parameters.get('height', 150)
        width = plot_parameters.get('width', 360)
        axis_lims = plot_parameters.get('axis_lim', [-0.3, 0.9])
        t_limit = plot_parameters.get('axis_lim', -20)

        event_data = event_data[event_data['t (event index)'] >= t_limit]
        event_data = event_data[event_data['Shapley Value'] >= axis_lims[0]]
        event_data = event_data[event_data['Shapley Value'] <= axis_lims[1]]

        global_event = alt.Chart(event_data).mark_point(stroke='white',
                                                      strokeWidth=.6).encode(
            y=alt.Y('Shapley Value', axis=alt.Axis(grid=True, titleX=-23),
                    title="Shapley Value", scale=alt.Scale(domain=axis_lims, )),
            x=alt.X('t (event index):O', axis=alt.Axis(labelAngle=0)),
            color=alt.Color('type',
                            scale=alt.Scale(domain=['Shapley Value', 'Mean'],
                                            range=["#48caaa", '#d76d58']),
                            legend=alt.Legend(title=None, fillColor="white",
                                              symbolStrokeWidth=0, symbolSize=50,
                                              orient="top-left")),
            opacity=alt.condition(alt.datum.type == 'Mean', alt.value(1.0),
                                  alt.value(0.2)),
            size=alt.condition(alt.datum.type == 'Mean', alt.value(70),
                               alt.value(30)),
        ).properties(
            width=width,
            height=height,
        )

        return global_event

    return multi_plot_wrapper(event_data, plot, ((plot_parameters),))
