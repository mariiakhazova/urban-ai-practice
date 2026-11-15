"""
Утилиты визуализации карт с помощью Folium.
"""

import branca
import folium
import geopandas as gpd
import numpy as np
import pandas as pd


PROVISION_COLORS = ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#91cf60", "#1a9850"]


def _color_scale(values, vmin=None, vmax=None, n_colors=256):
    if vmin is None:
        vmin = float(np.nanmin(values))
    if vmax is None:
        vmax = float(np.nanmax(values))
    if vmax <= vmin:
        vmax = vmin + 1
    return branca.colormap.LinearColormap(
        PROVISION_COLORS, vmin=vmin, vmax=vmax, caption=""
    )


def make_block_map(
    blocks: gpd.GeoDataFrame,
    values: pd.Series,
    title: str = "",
    value_label: str = "",
    color_label: str = "",
    schools_gdf: gpd.GeoDataFrame | None = None,
    boundary_gdf: gpd.GeoDataFrame | None = None,
    highlight_blocks: list[int] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    fixed_range: tuple | None = None,
) -> folium.Map:
    valid_vals = values.dropna()
    if len(valid_vals) == 0:
        center = blocks.geometry.unary_union.centroid
        m = folium.Map(location=[center.y, center.x], zoom_start=12)
        return m

    if fixed_range is not None:
        vmin, vmax = fixed_range
    else:
        if vmin is None:
            vmin = float(valid_vals.min())
        if vmax is None:
            vmax = float(valid_vals.max())

    cmap = _color_scale(valid_vals, vmin=vmin, vmax=vmax)
    cmap.caption = color_label or title

    center = blocks.geometry.unary_union.centroid
    m = folium.Map(
        location=[center.y, center.x],
        zoom_start=12,
        tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    )

    if boundary_gdf is not None and len(boundary_gdf) > 0:
        for _, row in boundary_gdf.iterrows():
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda x: {
                    "color": "#666666",
                    "weight": 2,
                    "fillOpacity": 0,
                    "dashArray": "5,5",
                },
            ).add_to(m)

    highlight_set = set(highlight_blocks or [])

    for idx, row in blocks.iterrows():
        val = values.get(idx, np.nan)
        if pd.isna(val):
            color = "#cccccc"
            opacity = 0.3
        else:
            color = cmap(val)
            opacity = 0.7

        weight = 1
        edge_color = "#444444"
        if idx in highlight_set:
            weight = 3
            edge_color = "#ff0000"
            opacity = 0.85

        tooltip_text = f"Квартал {idx}"
        if not pd.isna(val):
            tooltip_text += f"<br>{value_label}: {val:.4f}" if "обеспечен" in value_label.lower() or "доля" in value_label.lower() else f"<br>{value_label}: {val:.1f}"

        pop_val = row.get("population", None)
        if pop_val is not None and not pd.isna(pop_val):
            tooltip_text += f"<br>Население: {int(pop_val)}"

        cap_val = row.get("capacity", None)
        if cap_val is not None and not pd.isna(cap_val):
            tooltip_text += f"<br>Вместимость школ: {int(cap_val)}"

        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda x, c=color, w=weight, ec=edge_color, op=opacity: {
                "fillColor": c,
                "color": ec,
                "weight": w,
                "fillOpacity": op,
            },
            tooltip=tooltip_text,
        ).add_to(m)

    if schools_gdf is not None and len(schools_gdf) > 0:
        for _, school in schools_gdf.iterrows():
            geom = school.geometry
            point = geom if geom.geom_type == "Point" else geom.representative_point()
            cap = school.get("capacity", np.nan)
            radius = 5.0
            if not pd.isna(cap):
                radius = 4.0 + float(cap) / 200.0
            folium.CircleMarker(
                location=[point.y, point.x],
                radius=min(radius, 12),
                color="#1f4e9c",
                fill=True,
                fill_color="#3f6fb5",
                fill_opacity=0.8,
                tooltip=f"Школа · {int(cap) if not pd.isna(cap) else '?'} мест",
            ).add_to(m)

    cmap.add_to(m)
    title_html = f'<h3 style="text-align:center;font-family:sans-serif;">{title}</h3>'
    m.get_root().html.add_child(folium.Element(title_html))

    return m
