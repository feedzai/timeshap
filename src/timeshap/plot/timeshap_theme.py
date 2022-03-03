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


def timeshap_theme():
    # UI kit colors
    GREY_DARK_3 = "#384140"
    GREY_SEMI_1 = "#a2acb6"
    GREY_SEMI_2 = "#a2acb6"
    GREY_SEMI_3 = "#798184"
    GREY_LIGHT_3 = "#e1e1e1"

    # Font sizes
    FONT_LARGE = 20
    FONT_MEDIUM = 16
    FONT_SMALL = 12

    # Chart elements colors
    fontColor = GREY_DARK_3
    markColor = "#618FE0"
    axisColor = GREY_SEMI_3
    gridOpacity = 0.5
    backgroundColor = "white"
    font = "Roboto"

    # Color palettes
    feedzaiColorSchemes = {
        "category": {
            "default": [
                "#618FE0",
                "#50D0B0",
                "#E17560",
                "#E8B474",
                "#D889C4",
                "#34A3BC",
                "#A583A4",
            ],
            "status": [
                "#5cb85c",
                "#ea5f51",
                "#f2a638",
            ],
        },
        "sequential": {
            "blues": [
                "#001854",
                "#002f72",
                "#04478e",
                "#1e5fab",
                "#4477c5",
                "#6290e1",
                "#7ea9fd",
                "#9ec7ff",
                "#bde4ff",
            ],
            "greens": [
                "#002e19",
                "#00452f",
                "#005e45",
                "#00785d",
                "#009376",
                "#24af90",
                "#48caaa",
                "#67e5c4",
                "#90ffea",
            ],
            "oranges": [
                "#4a0000",
                "#6c0905",
                "#86261a",
                "#a13e2e",
                "#bc5543",
                "#d76d58",
                "#f3856f",
                "#ffa68e",
                "#ffcbb2",
            ],
        },
        "diverging": {
            "blueorange": [
                "#003073",
                "#2c5ea8",
                "#5f8fd6",
                "#99c3fb",
                "#f5f5f5",
                "#ffaa92",
                "#d16f5b",
                "#9e392c",
                "#650200",
            ],
            "greenpurple": [
                "#003b26",
                "#006b52",
                "#1aa082",
                "#55d5b5",
                "#f5f5f5",
                "#f5a5e1",
                "#be71ac",
                "#8a3f7a",
                "#560d49",
            ],
            "greenorange": [
                "#003b26",
                "#006b52",
                "#1aa082",
                "#55d5b5",
                "#f5f5f5",
                "#ffaa92",
                "#d16f5b",
                "#9e392c",
                "#650200",
            ],
        },
    }
    return {
        "config": {
            # Guides
            "axis": {
                "domain": True,
                "domainColor": axisColor,
                "grid": False,
                "gridColor": axisColor,
                "gridOpacity": gridOpacity,
                "gridDash": [3, 5],
                "gridWidth": 0.8,
                "gridCap": "round",
                "labelPadding": 3,
                "labelFont": font,
                "labelColor": axisColor,
                "tickSize": 5,
                "tickColor": axisColor,
                "tickOpacity": gridOpacity,
                "titleColor": fontColor,
                "titleFont": font,
                "titleFontSize": FONT_SMALL,
            },
            "axisBand": {
                "domain": True,
                "ticks": False,
                "labelPadding": 7,
            },
            "axisY": {
                "domain": False,
                "titleAlign": "left",
                "titleAngle": 0,
                "titleX": -20,
                "titleY": -10,
            },
            "legend": {
                "labelColor": axisColor,
                "labelFontSize": FONT_SMALL,
                "symbolSize": 40,
                "titleColor": fontColor,
                "titleFontSize": FONT_SMALL,
                "titlePadding": 10,
                "titleFont": font,
                "labelFont": font,
            },
            # Marks
            "line": {
                "stroke": markColor,
                "strokeWidth": 2,
            },
            "rule": {
                "stroke": axisColor,
            },
            "path": {"stroke": markColor, "strokeWidth": 0.5},
            "rect": {"fill": markColor},
            "point": {
                "filled": True,
                "shape": "circle",
            },
            "shape": {"stroke": markColor},
            "bar": {
                "fill": markColor,
                "stroke": None,
            },
            "text": {
                "font": font,
                "color": fontColor,
                "fontSize": FONT_SMALL,
            },
            "arc": {"stroke": "#fff", "strokeWidth": 1},
            # Colors
            "range": {
                "category": feedzaiColorSchemes["category"]["default"],
                "ramp": feedzaiColorSchemes["sequential"]["blues"],
                "heatmap": feedzaiColorSchemes["sequential"]["blues"],
                "diverging": feedzaiColorSchemes["diverging"]["greenorange"],
            },
            # Global chart elements
            "title": {
                "anchor": "start",
                "fontSize": FONT_LARGE,
                "color": fontColor,
                "fontWeight": "bold",
                "offset": 20,
                "font": font,
                "subtitleColor": fontColor,
                "subtitleFontSize": FONT_MEDIUM,
            },
            "header": {
                "labelFontSize": FONT_SMALL,
                "titleFontSize": FONT_MEDIUM,
                "labelColor": fontColor,
                "titleColor": fontColor,
                "titleFont": font,
                "labelFont": font,
            },
            "group": {
                "fill": backgroundColor,
            },
            "background": backgroundColor,
            "view": {
                "stroke": "transparent",
                "continuousHeight": 300,
                "continuousWidth": 400,
            },
        }
    }
