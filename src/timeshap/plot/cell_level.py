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
import numpy as np
import altair as alt
import re
import copy
from typing import List


def plot_cell_level(cell_data: pd.DataFrame,
                    model_features: List[str],
                    plot_features: dict,
                    ):
    """Plots local feature explanations

    Parameters
    ----------
    cell_data: pd.DataFrame
        Cell level explanations

    model_features: int
        The number of feature to display.

    plot_features: dict
        Dict containing mapping between model features and display features
    """
    c_range = ["#5f8fd6",
               "#99c3fb",
               "#f5f5f5",
               "#ffaa92",
               "#d16f5b",
               ]
    unique_events = [x for x in np.unique(cell_data['Event'].values) if x not in ['Other Events', 'Pruned Events']]
    sort_events = sorted(unique_events, key=lambda x:  re.findall(r'\d+', x)[0], reverse=True)
    unique_feats = [x for x in np.unique(cell_data['Feature'].values) if x not in ['Other Features', 'Pruned Events']]
    if plot_features:
        plot_features = copy.deepcopy(plot_features)
        sort_features = [plot_features[x] for x in model_features if x in unique_feats]
        if 'Other Features' in np.unique(cell_data['Feature'].values):
            plot_features['Other Features'] = 'Other Features'

        if 'Pruned Events' in np.unique(cell_data['Feature'].values):
            plot_features['Pruned Events'] = 'Pruned Events'

        cell_data['Feature'] = cell_data['Feature'].apply(lambda x: plot_features[x])
    else:
        sort_features = [x for x in model_features if x in unique_feats]

    cell_data['rounded'] = cell_data['Shapley Value'].apply(lambda x: round(x, 3))
    cell_data['rounded_str'] = cell_data['Shapley Value'].apply(lambda x: '0.000' if round(x, 3) == 0 else str(round(x, 3)))
    cell_data['rounded_str'] = cell_data['rounded_str'].apply(lambda x: f'{x}0' if len(x) == 4 else x)

    filtered_cell_data = cell_data[~np.logical_and(cell_data['Event'] == 'Pruned Events', cell_data['Feature'] == 'Pruned Events')]

    height = 225
    width = 200

    c = alt.Chart().encode(
        y=alt.Y('Feature', axis=alt.Axis(domain=False, labelFontSize=15, title=None), sort=sort_features),
    )

    a = c.mark_rect().encode(
        x=alt.X('Event', axis=alt.Axis(titleFontSize=15), sort=sort_events),
        color=alt.Color('rounded', title=None,
                        legend=alt.Legend(gradientLength=height,
                                          gradientThickness=10, orient='right',
                                          labelFontSize=15),
                        scale=alt.Scale(domain=[-.5, .5], range=c_range))
    )
    b = c.mark_text(align='right', baseline='middle', dx=18, fontSize=15,
                    color='#798184').encode(
            x=alt.X('Event', sort=sort_events,
                    axis=alt.Axis(orient="top", title='Shapley Value', domain=False,
                                  titleY=height + 20, titleX=172, labelAngle=30,
                                  labelFontSize=15, )),
            text='rounded_str',
    )

    cell_plot = alt.layer(a, b, data=filtered_cell_data).properties(
        width=160,
        height=height
    )

    if 'Pruned Events' in np.unique(cell_data['Event'].values):
        # isolate the pruned contribution
        df_prun = cell_data[np.logical_and(cell_data['Event'] == 'Pruned Events',cell_data['Feature'] == 'Pruned Events')]
        assert df_prun.shape == (1, 5)
        prun_rounded_str = df_prun.iloc[0]['rounded_str']
        prun_rounded = df_prun.iloc[0]['rounded']
        df_prun = pd.DataFrame([[["Pruned", "Events"], "Other features", prun_rounded, prun_rounded_str], ],
                               columns=['Event', 'Feature', 'rounded', 'rounded_str'])

        c = alt.Chart().encode(y=alt.Y('Feature',
                                       axis=alt.Axis(labels=False, domain=False,
                                                     title=None)), )

        a = c.mark_rect().encode(
            x=alt.X('Event', axis=alt.Axis(titleFontSize=15)),
            color=alt.Color('rounded', title=None, legend=None,
                            scale=alt.Scale(domain=[-.5, .5], range=c_range))
        )
        b = c.mark_text(align='right', dx=18, baseline='middle', fontSize=15,
                        color='#798184').encode(
            x=alt.X('Event',
                    axis=alt.Axis(labelOffset=24, labelPadding=30, orient="top",
                                  title=None, domain=False, labelAngle=0,
                                  labelFontSize=15, )),
            text='rounded_str',
        )

        cell_plot_prun = alt.layer(a, b, data=df_prun).properties(
            width=width / 3,
            height=height
        )

        cell_plot = alt.hconcat(cell_plot_prun, cell_plot).resolve_scale(color='independent')
    return cell_plot
