"""Shared chart styling: one palette, thin marks, tooltips, light/dark steps.

Colors follow the project's validated categorical order (blue, orange, aqua) and never
exceed three series per chart. Status colors (good / warning / critical) are reserved for
drift state and always appear with a text label. Every chart has a table view next to it.
"""

from __future__ import annotations

import altair as alt
import streamlit as st

LIGHT = {"s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "muted": "#8a8984"}
DARK = {"s1": "#3987e5", "s2": "#d95926", "s3": "#199e70", "muted": "#9a9993"}
STATUS = {"stable": "#0ca30c", "moderate": "#fab219", "major": "#d03b3b"}


def palette() -> dict:
    try:
        dark = st.context.theme.type == "dark"
    except Exception:  # noqa: BLE001 - older Streamlit / tests
        dark = False
    return DARK if dark else LIGHT


def series_scale(names: list[str]) -> alt.Scale:
    """Fixed slot per series name, in order: s1, s2, s3."""
    p = palette()
    return alt.Scale(domain=names, range=[p["s1"], p["s2"], p["s3"]][: len(names)])


def legend() -> alt.Legend:
    return alt.Legend(orient="top", title=None, symbolType="stroke", labelLimit=260)


def show(chart: alt.Chart, data, *, table_label: str = "Show data") -> None:
    st.altair_chart(chart, use_container_width=True, theme="streamlit")
    with st.expander(table_label):
        st.dataframe(data, width="stretch", hide_index=True)


def line(
    base: alt.Chart,
    x: str,
    y: str,
    color: str,
    names: list[str],
    *,
    x_title: str | None,
    y_title: str | None,
    y_format: str = ",.0f",
    tooltip=None,
    x_label_angle: int | None = None,
    x_zero: bool = True,
) -> alt.Chart:
    """Thin multi-series line with a hover rule and tooltip."""
    enc_color = alt.Color(f"{color}:N", scale=series_scale(names), legend=legend())
    x_axis = alt.Axis(labelAngle=x_label_angle) if x_label_angle is not None else alt.Axis()
    x_scale = alt.Scale(zero=False, nice=False) if not x_zero else alt.Undefined
    lines = base.mark_line(strokeWidth=2).encode(
        x=alt.X(x, title=x_title, axis=x_axis, scale=x_scale),
        y=alt.Y(y, title=y_title, axis=alt.Axis(format=y_format)),
        color=enc_color,
    )
    hover = alt.selection_point(
        fields=[x.split(":")[0]], nearest=True, on="pointerover", empty=False, clear="pointerout"
    )
    x_enc = alt.X(x, scale=x_scale)
    points = (
        base.mark_point(size=60, filled=True)
        .encode(
            x=x_enc,
            y=y,
            color=enc_color,
            opacity=alt.condition(hover, alt.value(1), alt.value(0)),
            tooltip=tooltip or [x, color, alt.Tooltip(y, format=y_format)],
        )
        .add_params(hover)
    )
    rule = (
        base.mark_rule(color=palette()["muted"], strokeWidth=1)
        .encode(x=x_enc, opacity=alt.condition(hover, alt.value(0.6), alt.value(0)))
        .transform_filter(hover)
    )
    return alt.layer(lines, points, rule)


def ref_rule(value, *, axis: str = "x", label: str = "", dash=(4, 4)) -> alt.Chart:
    """A labelled reference line (operating budget, split boundary, capacity)."""
    p = palette()
    import pandas as pd

    df = pd.DataFrame({"v": [value], "label": [label]})
    no_zero = alt.Scale(zero=False, nice=False)  # a reference line must not stretch the axis
    enc = {axis: alt.X("v", scale=no_zero) if axis == "x" else alt.Y("v", scale=no_zero)}
    rule = (
        alt.Chart(df)
        .mark_rule(color=p["muted"], strokeDash=list(dash), strokeWidth=1)
        .encode(**enc)
    )
    if not label:
        return rule
    text = (
        alt.Chart(df)
        .mark_text(align="left", dx=4, dy=-6 if axis == "y" else 0, baseline="top", fontSize=11)
        .encode(
            **enc, text="label", **({"y": alt.value(4)} if axis == "x" else {"x": alt.value(4)})
        )
    )
    return alt.layer(rule, text)
