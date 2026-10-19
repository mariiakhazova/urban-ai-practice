"""Сборка автономного HTML-дашборда из готовых GeoJSON BlocksNet."""
import base64
import html
import time
from io import BytesIO
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from branca.colormap import LinearColormap
from folium.plugins import MiniMap

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "arzamas"
OUTPUT = BASE_DIR / "arzamas_dashboard.html"
COAT_PATH = BASE_DIR / "assets" / "arzamas_coat_of_arms.png"
GOOSE_PATH = BASE_DIR / "assets" / "arzamas_goose.png"

LAND_USE_LABELS = {
    "residential": "жилая",
    "business": "деловая",
    "recreation": "рекреационная",
    "industrial": "промышленная",
    "transport": "транспортная",
    "special": "специальная",
    "agriculture": "сельскохозяйственная",
}

MORPHOTYPE_LABELS = {
    "low-rise low-density residential": "малоэтажная низкоплотная жилая",
    "low-rise low-density non-residential": "малоэтажная низкоплотная нежилая",
    "low-rise low-density mixed-use": "малоэтажная низкоплотная смешанная",
    "mid-rise low-density residential": "среднеэтажная низкоплотная жилая",
    "mid-rise residential": "среднеэтажная жилая",
    "high-mid-rise residential": "повышенной этажности жилая",
    "low-rise model": "малоэтажная модель",
    "individual residential": "индивидуальная жилая",
    "low-rise non-residential": "малоэтажная нежилая",
    "mid-rise": "среднеэтажная",
    "central model": "центральная модель",
}

COLUMN_INFO = {
    "block_id": ("Квартал", "№", "Идентификаторы"),
    "population": ("Население", "чел.", "Население и спрос"),
    "demand": ("Спрос на школы", "мест", "Население и спрос"),
    "capacity": ("Ёмкость школ", "мест", "Население и спрос"),
    "provision_strong": ("Обеспеченность школами, авто (сильная)", "% норматива", "Школы"),
    "provision_weak": ("Обеспеченность школами, авто (слабая)", "% норматива", "Школы"),
    "provision_strong_intermodal": ("Обеспеченность школами, ОТ (сильная)", "% норматива", "Школы"),
    "provision_weak_intermodal": ("Обеспеченность школами, ОТ (слабая)", "% норматива", "Школы"),
    "area_accessibility_drive": ("Доступность по площади, авто", "мин", "Доступность"),
    "area_accessibility_intermodal": ("Доступность по площади, ОТ", "мин", "Доступность"),
    "fsi": ("Плотность застройки (FSI)", "индекс", "Плотность и морфология"),
    "gsi": ("Плотность покрытия (GSI)", "индекс", "Плотность и морфология"),
    "mxi": ("Смешанность использования (MXI)", "индекс", "Плотность и морфология"),
    "l": ("Связность улиц (L)", "индекс", "Плотность и морфология"),
    "osr": ("Открытые пространства (OSR)", "индекс", "Плотность и морфология"),
    "cluster": ("Кластер Spacematrix", "№", "Морфотипы"),
    "spacematrix_morphotype": ("Морфотип Spacematrix", "категория", "Морфотипы"),
    "strelka_morphotype": ("Морфотип Strelka", "категория", "Морфотипы"),
    "land_use": ("Главная функция", "категория", "Землепользование"),
    "share": ("Доля главной зоны", "доля", "Землепользование"),
    "residential": ("Жилая зона", "доля", "Землепользование"),
    "business": ("Деловая зона", "доля", "Землепользование"),
    "recreation": ("Рекреационная зона", "доля", "Землепользование"),
    "industrial": ("Промышленная зона", "доля", "Землепользование"),
    "transport": ("Транспортная зона", "доля", "Землепользование"),
    "special": ("Специальная зона", "доля", "Землепользование"),
    "agriculture": ("Сельскохозяйственная зона", "доля", "Землепользование"),
}

SUMMARY_COLUMNS = [
    "population",
    "demand",
    "capacity",
    "provision_strong",
    "provision_weak",
    "provision_strong_intermodal",
    "provision_weak_intermodal",
    "area_accessibility_drive",
    "area_accessibility_intermodal",
    "fsi",
    "gsi",
    "mxi",
    "l",
    "osr",
    "residential",
    "business",
    "recreation",
    "industrial",
    "transport",
    "special",
    "agriculture",
    "share",
]

FULL_COLUMNS = [
    "block_id",
    "land_use",
    "population",
    "demand",
    "capacity",
    "provision_strong",
    "provision_weak",
    "provision_strong_intermodal",
    "provision_weak_intermodal",
    "residential",
    "business",
    "recreation",
    "industrial",
    "transport",
    "special",
    "agriculture",
    "share",
    "fsi",
    "gsi",
    "mxi",
    "l",
    "osr",
    "cluster",
    "spacematrix_morphotype",
    "strelka_morphotype",
    "area_accessibility_drive",
    "area_accessibility_intermodal",
]

SHORT_LABELS = {
    "block_id": "Квартал",
    "land_use": "Функция",
    "population": "Население",
    "demand": "Спрос",
    "capacity": "Ёмкость",
    "provision_strong": "Обесп. сильн.",
    "provision_weak": "Обесп. слаб.",
    "provision_strong_intermodal": "Обесп. ОТ",
    "provision_weak_intermodal": "Обесп. ОТ слаб.",
    "residential": "Жилая",
    "business": "Деловая",
    "recreation": "Рекреация",
    "industrial": "Пром.",
    "transport": "Транспорт",
    "special": "Спец.",
    "agriculture": "Сельхоз",
    "share": "Доля гл. зоны",
    "fsi": "FSI",
    "gsi": "GSI",
    "mxi": "MXI",
    "l": "L",
    "osr": "OSR",
    "cluster": "Кластер",
    "spacematrix_morphotype": "Spacematrix",
    "strelka_morphotype": "Strelka",
    "area_accessibility_drive": "Доступность авто",
    "area_accessibility_intermodal": "Доступность ОТ",
}

PERCENT_COLUMNS = {"provision_strong", "provision_weak", "provision_strong_intermodal", "provision_weak_intermodal"}
INTEGER_COLUMNS = {"block_id", "population", "demand", "capacity", "cluster"}
DECIMALS = {
    "fsi": 3,
    "gsi": 3,
    "mxi": 3,
    "l": 2,
    "osr": 1,
    "area_accessibility_drive": 1,
    "area_accessibility_intermodal": 1,
    "share": 2,
    "residential": 2,
    "business": 2,
    "recreation": 2,
    "industrial": 2,
    "transport": 2,
    "special": 2,
    "agriculture": 2,
}

TOOLTIP_COLUMNS = [
    ("block_id", "Квартал"),
    ("land_use", "Функция"),
    ("population", "Население, чел."),
    ("demand", "Спрос, мест"),
    ("capacity", "Ёмкость, мест"),
    ("provision_strong", "Обеспеченность"),
    ("provision_weak", "Обеспеченность (слабая)"),
    ("provision_strong_intermodal", "Обеспеченность, ОТ"),
    ("provision_weak_intermodal", "Обеспеченность, ОТ (слабая)"),
    ("fsi", "FSI"),
    ("gsi", "GSI"),
    ("mxi", "MXI"),
    ("l", "L"),
    ("osr", "OSR"),
    ("area_accessibility_drive", "Доступность (авто), мин"),
    ("area_accessibility_intermodal", "Доступность (ОТ), мин"),
    ("spacematrix_morphotype", "Морфотип Spacematrix"),
    ("strelka_morphotype", "Морфотип Strelka"),
]

MAP_METRICS = [
    ("population", "Население, чел.", "YlOrRd", None),
    ("provision_strong", "Обеспеченность школами, авто", "RdYlGn", (0.0, 1.0)),
    ("provision_strong_intermodal", "Обеспеченность школами, ОТ", "RdYlGn", (0.0, 1.0)),
    ("provision_weak", "Обеспеченность (слабая), авто", "RdYlGn", (0.0, 1.0)),
    ("provision_weak_intermodal", "Обеспеченность (слабая), ОТ", "RdYlGn", (0.0, 1.0)),
    ("area_accessibility_drive", "Доступность по площади, авто, мин", "RdYlGn_r", None),
    ("area_accessibility_intermodal", "Доступность по площади, ОТ, мин", "RdYlGn_r", None),
    ("fsi", "FSI — плотность застройки", "BuPu", None),
    ("gsi", "GSI — плотность покрытия", "BuGn", None),
    ("mxi", "MXI — смешанность", "PuRd", None),
    ("l", "L — связность улиц", "OrRd", None),
    ("osr", "OSR — открытые пространства", "Greens", None),
]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.5,
        "figure.facecolor": "white",
    }
)


def fmt_int(value):
    return f"{float(value):,.0f}".replace(",", " ")


def fmt_decimal(value, digits=2):
    return f"{float(value):.{digits}f}".replace(".", ",")


def fmt_percent(value, digits=1):
    return f"{float(value) * 100:.{digits}f}%".replace(".", ",")


def fmt_measure(value, digits=1):
    text = f"{float(value):,.{digits}f}"
    return text.replace(",", " ").replace(".", ",")


def plural(number, one, few, many):
    remainder = abs(int(number)) % 100
    if 11 <= remainder <= 14:
        return many
    remainder %= 10
    if remainder == 1:
        return one
    if 2 <= remainder <= 4:
        return few
    return many


def fmt_value(column, value):
    if pd.isna(value):
        return "—"
    if column in PERCENT_COLUMNS:
        return fmt_percent(value)
    if column in INTEGER_COLUMNS:
        return fmt_int(value)
    return fmt_decimal(value, DECIMALS.get(column, 2))


def format_feature_value(column, value):
    if pd.isna(value):
        return "—"
    if column == "land_use":
        return land_use_label(value)
    if column in ("spacematrix_morphotype", "strelka_morphotype"):
        return morphotype_label(value)
    return fmt_value(column, value)


def land_use_label(value):
    return LAND_USE_LABELS.get(value, str(value))


def morphotype_label(value):
    return MORPHOTYPE_LABELS.get(value, str(value))


def read_layers():
    file_names = {
        "blocks": "blocks_metrics.geojson",
        "schools": "schools.geojson",
        "boundary": "boundary.geojson",
    }
    missing = [name for name in file_names.values() if not (DATA_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Не найдены готовые GeoJSON: " + ", ".join(missing) + ". Ожидаемая папка: " + str(DATA_DIR)
        )

    blocks = gpd.read_file(DATA_DIR / file_names["blocks"])
    schools = gpd.read_file(DATA_DIR / file_names["schools"])
    boundary = gpd.read_file(DATA_DIR / file_names["boundary"])

    if blocks.crs is not None and blocks.crs.to_epsg() != 4326:
        blocks = blocks.to_crs(4326)
    if schools.crs is not None and schools.crs.to_epsg() != 4326:
        schools = schools.to_crs(4326)
    if boundary.crs is not None and boundary.crs.to_epsg() != 4326:
        boundary = boundary.to_crs(4326)

    blocks["block_id"] = pd.to_numeric(blocks["block_id"], errors="coerce").astype(int)
    if "area_accessibility_drive" not in blocks.columns and "area_accessibility" in blocks.columns:
        blocks = blocks.rename(columns={"area_accessibility": "area_accessibility_drive"})
    blocks = blocks.sort_values("block_id").reset_index(drop=True)
    if "capacity" in schools.columns:
        schools["capacity"] = pd.to_numeric(schools["capacity"], errors="coerce")
    return blocks, schools, boundary


def read_matrix_means():
    means = {}
    for key in ("drive", "intermodal"):
        path = DATA_DIR / f"acc_mx_{key}.csv"
        if not path.exists():
            continue
        matrix = pd.read_csv(path, index_col=0)
        values = matrix.to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            means[key] = float(values.mean())
    return means


def chart_to_base64(figure):
    buffer = BytesIO()
    figure.tight_layout()
    figure.savefig(buffer, format="png", dpi=115, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def coat_data_uri():
    if not COAT_PATH.exists():
        return ""
    encoded = base64.b64encode(COAT_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def goose_data_uri():
    if not GOOSE_PATH.exists():
        return ""
    encoded = base64.b64encode(GOOSE_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def palette(name, count=7):
    colormap = plt.get_cmap(name)
    return [matplotlib.colors.to_hex(colormap(index / (count - 1))) for index in range(count)]


def color_scale(values, cmap_name, fixed=None):
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if fixed is not None:
        low, high = fixed
    elif len(numeric):
        low, high = numeric.quantile(0.02), numeric.quantile(0.98)
    else:
        low, high = 0.0, 1.0
    if not np.isfinite(low):
        low = float(numeric.min()) if len(numeric) else 0.0
    if not np.isfinite(high):
        high = float(numeric.max()) if len(numeric) else 1.0
    if high <= low:
        high = low + 1.0
    return LinearColormap(colors=palette(cmap_name), vmin=float(low), vmax=float(high))


def make_map(blocks, schools, boundary):
    bounds = blocks.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    city_map = folium.Map(
        location=center,
        zoom_start=13,
        tiles="CartoDB positron",
        control_scale=True,
        width="100%",
        height=700,
    )
    MiniMap(toggle_display=True).add_to(city_map)

    boundary_layer = folium.FeatureGroup(name="Граница города", show=True)
    folium.GeoJson(
        boundary.__geo_interface__,
        style_function=lambda feature: {
            "fillOpacity": 0.0,
            "color": "#1f2933",
            "weight": 2.4,
        },
    ).add_to(boundary_layer)
    boundary_layer.add_to(city_map)

    map_blocks = blocks.copy()
    for column, _ in TOOLTIP_COLUMNS:
        if column in map_blocks.columns:
            map_blocks[column] = map_blocks[column].map(
                lambda value, current=column: format_feature_value(current, value)
            )
    map_blocks["block_id"] = map_blocks["block_id"].astype(int).astype(str)

    tooltip_fields = [column for column, _ in TOOLTIP_COLUMNS if column in map_blocks.columns]
    tooltip_aliases = [alias for column, alias in TOOLTIP_COLUMNS if column in map_blocks.columns]

    for index, (metric, label, cmap_name, fixed) in enumerate(MAP_METRICS):
        if metric not in blocks.columns:
            continue
        scale = color_scale(blocks[metric], cmap_name, fixed)
        color_map = {
            int(block_id): (scale(value) if pd.notna(value) else "#d9d9d9")
            for block_id, value in zip(blocks["block_id"], blocks[metric])
        }

        def style_function(feature, colors=color_map):
            block_id = int(feature["properties"]["block_id"])
            return {
                "fillColor": colors.get(block_id, "#d9d9d9"),
                "fillOpacity": 0.78,
                "color": "#4a5560",
                "weight": 0.45,
            }

        layer = folium.FeatureGroup(name=label, show=index == 0)
        folium.GeoJson(
            map_blocks.__geo_interface__,
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=tooltip_aliases,
                localize=True,
                sticky=False,
                labels=True,
                max_width=360,
            ),
        ).add_to(layer)
        layer.add_to(city_map)

    if len(schools):
        school_layer = folium.FeatureGroup(name="Школы", show=True)
        for _, school in schools.iterrows():
            geometry = school.geometry
            if geometry is None or geometry.is_empty:
                continue
            point = geometry if geometry.geom_type == "Point" else geometry.representative_point()
            capacity = school.get("capacity", np.nan)
            radius = 5.0 + (float(capacity) / 190.0 if pd.notna(capacity) else 0.0)
            tooltip = "Школа"
            if pd.notna(capacity):
                tooltip += f" · {fmt_int(capacity)} мест"
            folium.CircleMarker(
                location=[point.y, point.x],
                radius=min(radius, 12.0),
                color="#1f4e9c",
                fill=True,
                fill_color="#3f6fb5",
                fill_opacity=0.85,
                weight=1.2,
                tooltip=tooltip,
            ).add_to(school_layer)
        school_layer.add_to(city_map)

    folium.LayerControl(collapsed=False).add_to(city_map)
    return city_map


def add_histogram(ax, series, color, bins, xlabel, log_y=False, clip_quantile=None):
    data = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clip_quantile is not None and len(data):
        data = data.clip(upper=data.quantile(clip_quantile))
    ax.hist(data, bins=bins, color=color, edgecolor="white", linewidth=0.4)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Кварталов")


def make_charts(blocks, matrix_means):
    charts = []

    figure, ax = plt.subplots(figsize=(4.8, 3.0))
    add_histogram(ax, blocks["population"], "#e07b39", 40, "чел.", log_y=True)
    ax.set_title("Население кварталов (лог. шкала)")
    charts.append(("Население кварталов", chart_to_base64(figure)))

    figure, ax = plt.subplots(figsize=(4.8, 3.0))
    add_histogram(ax, blocks["area_accessibility_drive"], "#4d8f4d", 30, "минут")
    ax.set_title("Доступность по площади, авто")
    charts.append(("Доступность по площади, авто", chart_to_base64(figure)))

    if "area_accessibility_intermodal" in blocks.columns:
        figure, ax = plt.subplots(figsize=(4.8, 3.0))
        add_histogram(ax, blocks["area_accessibility_intermodal"], "#2f6f8f", 30, "минут")
        ax.set_title("Доступность по площади, общественный транспорт")
        charts.append(("Доступность по площади, ОТ", chart_to_base64(figure)))

    figure, ax = plt.subplots(figsize=(4.8, 3.0))
    add_histogram(ax, blocks["provision_strong"], "#3f6fb5", np.linspace(0, 1, 21), "доля норматива")
    ax.set_title("Обеспеченность школами, авто")
    charts.append(("Обеспеченность школами, авто", chart_to_base64(figure)))

    if "provision_strong_intermodal" in blocks.columns:
        figure, ax = plt.subplots(figsize=(4.8, 3.0))
        add_histogram(
            ax, blocks["provision_strong_intermodal"], "#5a7fbf", np.linspace(0, 1, 21), "доля норматива"
        )
        ax.set_title("Обеспеченность школами, общественный транспорт")
        charts.append(("Обеспеченность школами, ОТ", chart_to_base64(figure)))

    if {"area_accessibility_drive", "area_accessibility_intermodal"} <= set(blocks.columns):
        figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
        access_values = [
            blocks["area_accessibility_drive"].mean(),
            blocks["area_accessibility_intermodal"].mean(),
        ]
        axes[0].bar(["авто", "ОТ"], access_values, color=["#4d8f4d", "#2f6f8f"])
        axes[0].set_title("Доступность по площади, мин")
        for position, value in enumerate(access_values):
            axes[0].text(position, value, fmt_decimal(value, 1), ha="center", va="bottom", fontsize=9)
        provision_values = [np.nan, np.nan]
        if {"provision_strong", "provision_strong_intermodal"} <= set(blocks.columns):
            provision_values = [
                blocks["provision_strong"].mean(),
                blocks["provision_strong_intermodal"].mean(),
            ]
        axes[1].bar(["авто", "ОТ"], provision_values, color=["#4d8f4d", "#2f6f8f"])
        axes[1].set_title("Обеспеченность школами, доля норматива")
        for position, value in enumerate(provision_values):
            if np.isfinite(value):
                axes[1].text(position, value, fmt_decimal(value, 2), ha="center", va="bottom", fontsize=9)
        figure.suptitle("Автомобиль против общественного транспорта", fontsize=11)
        charts.append(("Авто и общественный транспорт", chart_to_base64(figure)))

    figure, ax = plt.subplots(figsize=(4.8, 3.0))
    add_histogram(ax, blocks["fsi"], "#8a5aa8", 30, "FSI")
    ax.set_title("Плотность застройки (FSI)")
    charts.append(("FSI", chart_to_base64(figure)))

    figure, ax = plt.subplots(figsize=(4.8, 3.0))
    add_histogram(ax, blocks["gsi"], "#2f8f83", 30, "GSI")
    ax.set_title("Плотность покрытия (GSI)")
    charts.append(("GSI", chart_to_base64(figure)))

    figure, ax = plt.subplots(figsize=(4.8, 3.0))
    add_histogram(ax, blocks["mxi"], "#c05a8a", 30, "MXI")
    ax.set_title("Смешанность использования (MXI)")
    charts.append(("MXI", chart_to_base64(figure)))

    figure, ax = plt.subplots(figsize=(4.8, 3.0))
    add_histogram(ax, blocks["l"], "#b7652a", 30, "L")
    ax.set_title("Связность улиц (L)")
    charts.append(("Связность улиц", chart_to_base64(figure)))

    figure, ax = plt.subplots(figsize=(4.8, 3.0))
    add_histogram(ax, blocks["osr"], "#3f7f45", 30, "OSR", clip_quantile=0.98)
    ax.set_title("Открытые пространства (OSR), обрезка по 98-му перцентилю")
    charts.append(("OSR", chart_to_base64(figure)))

    figure, ax = plt.subplots(figsize=(4.8, 3.0))
    counts = blocks["land_use"].map(land_use_label).fillna("нет данных").value_counts().sort_values()
    ax.barh(counts.index, counts.values, color="#4f7cac")
    ax.set_xlabel("Кварталов")
    ax.set_title("Главная функция кварталов")
    ax.tick_params(axis="y", labelsize=7)
    charts.append(("Главная функция", chart_to_base64(figure)))

    figure, ax = plt.subplots(figsize=(5.2, 3.3))
    counts = blocks["spacematrix_morphotype"].map(morphotype_label).fillna("нет данных").value_counts().sort_values()
    ax.barh(counts.index, counts.values, color="#8a5aa8")
    ax.set_xlabel("Кварталов")
    ax.set_title("Морфотипы Spacematrix")
    ax.tick_params(axis="y", labelsize=7)
    charts.append(("Морфотипы Spacematrix", chart_to_base64(figure)))

    figure, ax = plt.subplots(figsize=(5.2, 3.3))
    counts = blocks["strelka_morphotype"].map(morphotype_label).fillna("нет данных").value_counts().sort_values()
    ax.barh(counts.index, counts.values, color="#4d8f8a")
    ax.set_xlabel("Кварталов")
    ax.set_title("Морфотипы Strelka")
    ax.tick_params(axis="y", labelsize=7)
    charts.append(("Морфотипы Strelka", chart_to_base64(figure)))

    valid = blocks.dropna(subset=["fsi", "gsi"])
    correlation = valid["fsi"].corr(valid["gsi"])
    figure, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.scatter(valid["fsi"], valid["gsi"], s=20, alpha=0.65, color="#7a5195", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("FSI")
    ax.set_ylabel("GSI")
    title = "FSI и GSI"
    if pd.notna(correlation):
        title += f" (r = {fmt_decimal(correlation, 2)})"
    ax.set_title(title)
    charts.append(("FSI и GSI", chart_to_base64(figure)))

    return charts


def build_cards(blocks, schools, matrix_means):
    population = blocks["population"].sum()
    populated = int((blocks["population"] > 0).sum())
    provision = blocks["provision_strong"].dropna()
    access = blocks["area_accessibility_drive"].dropna()
    provision_pt = blocks.get("provision_strong_intermodal")
    provision_pt = provision_pt.dropna() if provision_pt is not None else pd.Series(dtype=float)
    access_pt = blocks.get("area_accessibility_intermodal")
    access_pt = access_pt.dropna() if access_pt is not None else pd.Series(dtype=float)
    cards = [
        ("Кварталов", fmt_int(len(blocks))),
        ("Население", fmt_int(population) + " чел."),
        ("Жилых кварталов", fmt_int(populated)),
        ("Школ", fmt_int(len(schools)) + " · " + fmt_int(schools["capacity"].sum()) + " мест"),
        ("Обеспеченность, авто", fmt_percent(provision.mean()) if len(provision) else "—"),
        ("Обеспеченность, ОТ", fmt_percent(provision_pt.mean()) if len(provision_pt) else "—"),
        ("Доступность, авто", fmt_decimal(access.mean(), 1) + " мин" if len(access) else "—"),
        ("Доступность, ОТ", fmt_decimal(access_pt.mean(), 1) + " мин" if len(access_pt) else "—"),
    ]
    if "drive" in matrix_means:
        cards.append(("Среднее время в пути, авто", fmt_decimal(matrix_means["drive"], 1) + " мин"))
    if "intermodal" in matrix_means:
        cards.append(("Среднее время в пути, ОТ", fmt_decimal(matrix_means["intermodal"], 1) + " мин"))
    return "\n".join(
        f'<div class="card"><div class="card-value">{html.escape(value)}</div>'
        f'<div class="card-label">{html.escape(label)}</div></div>'
        for label, value in cards
    )


def build_conclusions(blocks, schools, matrix_means):
    total = len(blocks)
    population = blocks["population"].sum()
    populated = int((blocks["population"] > 0).sum())
    top10_share = blocks.nlargest(10, "population")["population"].sum() / population if population else 0
    top_block = blocks.loc[blocks["population"].idxmax()]
    land_counts = blocks["land_use"].value_counts()
    no_land = int(blocks["land_use"].isna().sum())
    land_parts = [
        f"{LAND_USE_LABELS[key]} — {fmt_int(land_counts.get(key, 0))}"
        for key in ["residential", "industrial", "business", "recreation", "special", "agriculture"]
        if land_counts.get(key, 0)
    ]
    provision = blocks["provision_strong"].dropna()
    provision_positive = int((provision > 0).sum())
    provision_full = int((provision >= 0.99).sum())
    demand_total = blocks["demand"].sum()
    capacity_total = schools["capacity"].sum() if "capacity" in schools.columns else 0
    coverage = capacity_total / demand_total if demand_total else 0
    access = blocks["area_accessibility_drive"].dropna()
    best = blocks.loc[blocks["area_accessibility_drive"].idxmin()]
    worst = blocks.loc[blocks["area_accessibility_drive"].idxmax()]
    density_count = int(blocks["mxi"].notna().sum())
    dominant_sm = blocks["spacematrix_morphotype"].dropna().value_counts()
    dominant_st = blocks["strelka_morphotype"].dropna().value_counts()
    correlation = blocks["fsi"].corr(blocks["gsi"])

    transport_parts = []
    if "drive" in matrix_means and "intermodal" in matrix_means:
        transport_parts.append(
            f"среднее время в пути — {fmt_decimal(matrix_means['drive'], 1)} мин на автомобиле "
            f"против {fmt_decimal(matrix_means['intermodal'], 1)} мин на общественном транспорте"
        )
    if {"area_accessibility_drive", "area_accessibility_intermodal"} <= set(blocks.columns):
        access_pt = blocks["area_accessibility_intermodal"].dropna()
        transport_parts.append(
            f"доступность по площади — {fmt_decimal(access.mean(), 1)} мин на автомобиле "
            f"против {fmt_decimal(access_pt.mean(), 1)} мин на общественном транспорте"
        )
    if {"provision_strong", "provision_strong_intermodal"} <= set(blocks.columns):
        provision_pt = blocks["provision_strong_intermodal"].dropna()
        transport_parts.append(
            f"средняя обеспеченность школами — {fmt_percent(provision.mean())} "
            f"против {fmt_percent(provision_pt.mean())} норматива"
        )

    items = [
        "<b>Население.</b> "
        f"В границах Арзамаса выделено {fmt_int(total)} "
        f"{plural(total, 'квартал', 'квартала', 'кварталов')}, суммарное население — "
        f"{fmt_int(population)} чел. Люди проживают в {fmt_int(populated)} "
        f"{plural(populated, 'квартале', 'кварталах', 'кварталах')} "
        f"({fmt_percent(populated / total)}). На 10 самых населённых кварталов приходится "
        f"{fmt_percent(top10_share)} населения; крупнейший квартал — №{int(top_block['block_id'])} "
        f"({fmt_int(top_block['population'])} чел.).",
        "<b>Землепользование.</b> "
        + (", ".join(land_parts) + ". " if land_parts else "")
        + f"Главная функция не определена у {fmt_int(no_land)} "
        f"{plural(no_land, 'квартала', 'кварталов', 'кварталов')} "
        f"({fmt_percent(no_land / total)}).",
        "<b>Школы.</b> "
        f"В данных {fmt_int(len(schools))} школ суммарной ёмкостью {fmt_int(capacity_total)} мест; "
        f"расчётный спрос — {fmt_int(demand_total)} мест, то есть ёмкость покрывает "
        f"{fmt_percent(coverage)} спроса. Обеспеченность рассчитана для {fmt_int(len(provision))} "
        f"{plural(len(provision), 'жилого квартала', 'жилых кварталов', 'жилых кварталов')}: "
        f"средняя — {fmt_percent(provision.mean())} норматива, ненулевая — "
        f"у {fmt_int(provision_positive)}, полная (≥99%) — у {fmt_int(provision_full)}.",
        "<b>Доступность.</b> "
        f"Средняя доступность по площади — {fmt_decimal(access.mean(), 1)} мин "
        f"(медиана {fmt_decimal(access.median(), 1)}), диапазон "
        f"{fmt_decimal(access.min(), 1)}–{fmt_decimal(access.max(), 1)} мин. "
        f"Самый доступный квартал — №{int(best['block_id'])} ({fmt_decimal(best['area_accessibility_drive'], 1)} мин), "
        f"наименее доступный — №{int(worst['block_id'])} ({fmt_decimal(worst['area_accessibility_drive'], 1)} мин).",
        *(
            [
                "<b>Общественный транспорт.</b> "
                + "; ".join(transport_parts)
                + "."
            ]
            if transport_parts
            else []
        ),
        "<b>Плотность и морфотипы.</b> "
        f"Средние FSI = {fmt_decimal(blocks['fsi'].mean(), 2)}, GSI = "
        f"{fmt_decimal(blocks['gsi'].mean(), 2)}; показатели MXI, L, OSR и морфотипы рассчитаны для "
        f"{fmt_int(density_count)} {plural(density_count, 'квартала', 'кварталов', 'кварталов')}. "
        + (
            f"Доминирующий морфотип Spacematrix — «{morphotype_label(dominant_sm.index[0])}» "
            f"({fmt_int(dominant_sm.iloc[0])} из {fmt_int(density_count)}; "
            f"{fmt_percent(dominant_sm.iloc[0] / density_count)}), "
            if len(dominant_sm)
            else ""
        )
        + (
            f"Strelka — «{morphotype_label(dominant_st.index[0])}» "
            f"({fmt_int(dominant_st.iloc[0])} из {fmt_int(density_count)}; "
            f"{fmt_percent(dominant_st.iloc[0] / density_count)})."
            if len(dominant_st)
            else ""
        ),
    ]

    if pd.notna(correlation):
        items.append(
            "<b>Связь показателей.</b> "
            f"Коэффициент корреляции FSI и GSI — {fmt_decimal(correlation, 2)}: "
            "более плотная застройка сопровождается большей долей застроенной территории."
            if correlation >= 0.5
            else "<b>Связь показателей.</b> "
            f"Коэффициент корреляции FSI и GSI — {fmt_decimal(correlation, 2)}: "
            "выраженной линейной связи между показателями нет."
        )

    return "\n".join(f"<li>{item}</li>" for item in items)


def build_limitations(blocks):
    total = len(blocks)
    no_density = int(blocks["mxi"].isna().sum())
    no_provision = int(blocks["provision_strong"].isna().sum())
    no_land = int(blocks["land_use"].isna().sum())
    osr_max = blocks["osr"].max()
    osr_p98 = blocks["osr"].quantile(0.98)
    parts = [
        f"показатели MXI, L, OSR и морфотипы отсутствуют у {fmt_int(no_density)} "
        f"{plural(no_density, 'квартала', 'кварталов', 'кварталов')} "
        f"({fmt_percent(no_density / total)}); у этих же кварталов FSI и GSI равны нулю",
        f"обеспеченность школами не рассчитана для {fmt_int(no_provision)} "
        f"{plural(no_provision, 'квартала', 'кварталов', 'кварталов')} "
        f"({fmt_percent(no_provision / total)})",
        f"главная функция не определена у {fmt_int(no_land)} "
        f"{plural(no_land, 'квартала', 'кварталов', 'кварталов')} "
        f"({fmt_percent(no_land / total)})",
    ]
    return (
        "<b>Ограничения данных.</b> "
        + "; ".join(parts)
        + ". "
        f"OSR содержит выбросы (максимум {fmt_measure(osr_max)}), поэтому на графике "
        f"использована обрезка по 98-му перцентилю ({fmt_measure(osr_p98)}). "
        "Обеспеченность — результат конкурентной модели BlocksNet, а не фактическая загрузка школ. "
        "Показатели для общественного транспорта рассчитаны по пешеходно-автобусному графу "
        "(маршруты автобусов OSM); для недостижимых пар кварталов время заменено максимальным."
    )


def build_summary_table(blocks):
    rows = []
    for column in SUMMARY_COLUMNS:
        label, unit, group = COLUMN_INFO[column]
        series = pd.to_numeric(blocks[column], errors="coerce")
        rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{html.escape(group)}</td>"
            f"<td>{html.escape(unit)}</td>"
            f"<td>{fmt_int(series.count())}</td>"
            f"<td>{fmt_value(column, series.mean())}</td>"
            f"<td>{fmt_value(column, series.median())}</td>"
            f"<td>{fmt_value(column, series.min())}</td>"
            f"<td>{fmt_value(column, series.max())}</td>"
            f"<td>{fmt_int(len(blocks) - series.count())}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def build_categorical_table(blocks, column, labels, title):
    temporary = blocks.copy()
    temporary["_label"] = [
        labels.get(value, str(value)) if pd.notna(value) else "нет данных" for value in temporary[column]
    ]
    stats = (
        temporary.groupby("_label")
        .agg(
            blocks=("block_id", "size"),
            population=("population", "sum"),
            fsi=("fsi", "mean"),
            access=("area_accessibility_drive", "mean"),
        )
        .sort_values("blocks", ascending=False)
    )
    rows = []
    for label, row in stats.iterrows():
        rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{fmt_int(row['blocks'])}</td>"
            f"<td>{fmt_percent(row['blocks'] / len(temporary))}</td>"
            f"<td>{fmt_int(row['population'])}</td>"
            f"<td>{fmt_value('fsi', row['fsi'])}</td>"
            f"<td>{fmt_value('area_accessibility_drive', row['access'])}</td>"
            "</tr>"
        )
    return (
        f'<div class="table-card"><h3>{html.escape(title)}</h3>'
        '<div class="table-wrap"><table class="data-table">'
        "<thead><tr><th>Категория</th><th>Кварталов</th><th>Доля</th>"
        "<th>Население, чел.</th><th>Средний FSI</th><th>Средняя доступность, мин</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div></div>"
    )


def build_ranking_table(title, frame):
    columns = ["block_id", "land_use", "population", "provision_strong", "fsi", "area_accessibility_drive"]
    headers = ["Квартал", "Функция", "Население", "Обеспеченность", "FSI", "Доступность, авто"]
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if column == "land_use":
                display = land_use_label(value) if pd.notna(value) else "—"
            elif column == "area_accessibility_drive":
                display = fmt_decimal(value, 1) if pd.notna(value) else "—"
            else:
                display = fmt_value(column, value)
            cells.append(f"<td>{html.escape(display)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    header_html = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    return (
        f'<div class="table-card"><h3>{html.escape(title)}</h3>'
        '<div class="table-wrap"><table class="data-table">'
        f"<thead><tr>{header_html}</tr></thead><tbody>{''.join(rows)}</tbody></table></div></div>"
    )


def build_full_table(blocks):
    header_cells = []
    for column in FULL_COLUMNS:
        label, unit, _ = COLUMN_INFO[column]
        header_cells.append(
            f'<th title="{html.escape(label + ", " + unit)}">{html.escape(SHORT_LABELS[column])}</th>'
        )

    rows = []
    for _, row in blocks.iterrows():
        cells = []
        for column in FULL_COLUMNS:
            value = row[column]
            if pd.isna(value):
                cells.append('<td data-value="">—</td>')
                continue
            if column == "land_use":
                cells.append(
                    f'<td data-value="{html.escape(str(value))}">{html.escape(land_use_label(value))}</td>'
                )
            elif column in ("spacematrix_morphotype", "strelka_morphotype"):
                cells.append(
                    f'<td data-value="{html.escape(str(value))}">{html.escape(morphotype_label(value))}</td>'
                )
            else:
                cells.append(
                    f'<td data-value="{float(value):.6f}">{html.escape(fmt_value(column, value))}</td>'
                )
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return (
        '<div class="table-tools">'
        '<label for="table-search">Поиск по таблице</label>'
        '<input id="table-search" type="search" placeholder="Квартал, функция, морфотип…">'
        f'<span class="row-count">Показано: <b id="row-count">{len(blocks)}</b> из {len(blocks)}</span>'
        "</div>"
        '<div class="table-wrap full-table-wrap"><table class="data-table full-table" id="full-table">'
        "<thead><tr>"
        + "".join(header_cells)
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


CSS = """
:root {
  --ink: #0d2b0d;
  --muted: #2f5b2f;
  --line: #a3d9a3;
  --panel: #f7fff2;
  --accent: #006600;
  --bg: #e8ffe8;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: "Segoe UI", Arial, sans-serif;
  line-height: 1.45;
}
header {
  background: #006600;
  color: white;
  padding: 34px clamp(18px, 4vw, 56px) 28px;
}
header h1 { margin: 0 0 8px; font-size: clamp(24px, 3vw, 34px); letter-spacing: -0.01em; }
header p { margin: 0; color: #d8f5d8; max-width: 980px; }
.header-inner { display: flex; align-items: center; gap: 18px; }
.header-inner > div { min-width: 0; }
.coat {
  height: 96px;
  width: auto;
  flex: 0 0 auto;
  background: #ffffff;
  border-radius: 12px;
  padding: 6px;
  box-shadow: 0 2px 8px rgba(0, 64, 0, 0.28);
}
.goose {
  height: 120px;
  width: auto;
  margin-left: auto;
  flex: 0 0 auto;
  opacity: 0.95;
  filter: brightness(0) invert(1) drop-shadow(0 2px 6px rgba(0, 50, 0, 0.35));
}
.toc {
  position: sticky;
  top: 0;
  z-index: 20;
  display: flex;
  flex-wrap: wrap;
  gap: 6px 18px;
  padding: 10px clamp(18px, 4vw, 56px);
  background: rgba(240, 255, 240, 0.95);
  border-bottom: 1px solid var(--line);
  backdrop-filter: blur(6px);
}
.toc a { color: var(--accent); text-decoration: none; font-size: 13px; font-weight: 600; }
.toc a:hover { text-decoration: underline; }
main { padding: 24px clamp(18px, 4vw, 56px) 10px; }
.cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(165px, 1fr));
  gap: 12px;
  margin-bottom: 26px;
}
.card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 15px 16px;
  box-shadow: 0 1px 3px rgba(0, 64, 0, 0.08);
}
.card-value { font-size: 22px; font-weight: 700; color: var(--accent); }
.card-label { margin-top: 4px; color: var(--muted); font-size: 12px; }
section { margin-bottom: 32px; }
h2 { margin: 0 0 12px; font-size: 20px; }
h3 { margin: 0 0 10px; font-size: 15px; }
p { margin: 0 0 10px; }
.lead { color: var(--muted); font-size: 13px; margin-bottom: 12px; }
.insights {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 20px 22px;
  box-shadow: 0 1px 3px rgba(0, 64, 0, 0.08);
}
.insights ul { margin: 0; padding-left: 20px; }
.insights li { margin-bottom: 10px; }
.note {
  margin-top: 15px;
  padding: 12px 14px;
  background: #e9ffd9;
  border: 1px solid #9fd89f;
  border-radius: 10px;
  color: #23491f;
  font-size: 13px;
}
.map-frame {
  display: block;
  width: 100%;
  height: 720px;
  border: 0;
  border-radius: 14px;
  background: #b9e6b9;
  box-shadow: 0 1px 4px rgba(0, 64, 0, 0.14);
}
.map-hint { color: var(--muted); font-size: 13px; margin: 8px 2px 0; }
.charts {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(310px, 1fr));
  gap: 14px;
}
.chart {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 10px;
  box-shadow: 0 1px 3px rgba(0, 64, 0, 0.08);
}
.chart img { display: block; width: 100%; height: auto; }
.table-card, .table-panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 1px 3px rgba(0, 64, 0, 0.08);
}
.table-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
  gap: 14px;
}
.table-wrap { overflow: auto; max-height: 560px; border-radius: 8px; }
.full-table-wrap { max-height: 720px; }
table.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12.5px;
  background: white;
}
table.data-table th, table.data-table td {
  padding: 8px 10px;
  border-bottom: 1px solid var(--line);
  text-align: left;
  white-space: nowrap;
}
table.data-table th {
  position: sticky;
  top: 0;
  z-index: 2;
  background: #d9f2d9;
  font-weight: 700;
  font-size: 12px;
  cursor: default;
}
table.full-table th { cursor: pointer; user-select: none; }
table.full-table th[data-order="asc"]::after { content: " ▲"; color: var(--accent); }
table.full-table th[data-order="desc"]::after { content: " ▼"; color: var(--accent); }
table.data-table tbody tr:nth-child(even) { background: #f2fff0; }
table.data-table tbody tr:hover { background: #e2f8dd; }
tfoot td { font-weight: 700; background: #eaf8e6; }
.table-tools {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}
.table-tools label { font-size: 13px; font-weight: 600; }
.table-tools input {
  flex: 1 1 260px;
  max-width: 430px;
  padding: 8px 11px;
  border: 1px solid #9ccc9c;
  border-radius: 8px;
  font: inherit;
  font-size: 13px;
  background: white;
}
.row-count { color: var(--muted); font-size: 13px; }
footer {
  padding: 10px clamp(18px, 4vw, 56px) 38px;
  color: var(--muted);
  font-size: 12px;
}
footer p { margin-bottom: 6px; }
@media (max-width: 980px) {
  .goose { display: none; }
}
@media (max-width: 720px) {
  .map-frame { height: 520px; }
  .table-grid { grid-template-columns: 1fr; }
  .coat { height: 64px; }
}
"""


JS = """
(() => {
  const table = document.getElementById("full-table");
  const search = document.getElementById("table-search");
  const counter = document.getElementById("row-count");
  if (!table || !search || !counter) return;

  const updateCount = () => {
    counter.textContent = Array.from(table.tBodies[0].rows).filter((row) => !row.hidden).length;
  };

  search.addEventListener("input", () => {
    const query = search.value.trim().toLowerCase();
    Array.from(table.tBodies[0].rows).forEach((row) => {
      row.hidden = query.length > 0 && !row.textContent.toLowerCase().includes(query);
    });
    updateCount();
  });

  Array.from(table.tHead.rows[0].cells).forEach((header, index) => {
    header.addEventListener("click", () => {
      const ascending = header.dataset.order !== "asc";
      header.dataset.order = ascending ? "asc" : "desc";
      const body = table.tBodies[0];
      const rows = Array.from(body.rows);
      rows.sort((left, right) => {
        const leftCell = left.cells[index];
        const rightCell = right.cells[index];
        const leftValue = leftCell.dataset.value || leftCell.textContent.trim();
        const rightValue = rightCell.dataset.value || rightCell.textContent.trim();
        const leftNumber = Number.parseFloat(leftValue);
        const rightNumber = Number.parseFloat(rightValue);
        if (!Number.isNaN(leftNumber) && !Number.isNaN(rightNumber)) {
          return ascending ? leftNumber - rightNumber : rightNumber - leftNumber;
        }
        return ascending
          ? leftValue.localeCompare(rightValue, "ru")
          : rightValue.localeCompare(leftValue, "ru");
      });
      rows.forEach((row) => body.appendChild(row));
    });
  });
})();
"""


def main():
    blocks, schools, boundary = read_layers()
    matrix_means = read_matrix_means()
    city_map = make_map(blocks, schools, boundary)
    map_document = city_map.get_root().render()
    map_srcdoc = html.escape(map_document, quote=True)

    cards_html = build_cards(blocks, schools, matrix_means)
    conclusions_html = build_conclusions(blocks, schools, matrix_means)
    limitations_html = build_limitations(blocks)
    charts_html = "\n".join(
        f'<figure class="chart"><img src="data:image/png;base64,{image}" alt="{html.escape(title)}"></figure>'
        for title, image in make_charts(blocks, matrix_means)
    )
    summary_html = build_summary_table(blocks)
    categorical_html = "\n".join(
        [
            build_categorical_table(blocks, "land_use", LAND_USE_LABELS, "Главная функция кварталов"),
            build_categorical_table(
                blocks, "spacematrix_morphotype", MORPHOTYPE_LABELS, "Морфотипы Spacematrix"
            ),
            build_categorical_table(
                blocks, "strelka_morphotype", MORPHOTYPE_LABELS, "Морфотипы Strelka"
            ),
        ]
    )
    rankings_html = "\n".join(
        [
            build_ranking_table("Топ-10 по населению", blocks.nlargest(10, "population")),
            build_ranking_table(
                "Максимальная обеспеченность школами",
                blocks[blocks["provision_strong"] > 0].nlargest(10, "provision_strong"),
            ),
            build_ranking_table("Самые доступные кварталы", blocks.nsmallest(10, "area_accessibility_drive")),
        ]
    )
    full_table_html = build_full_table(blocks)

    coat_uri = coat_data_uri()
    coat_html = (
        f'<img class="coat" src="{coat_uri}" alt="Герб Арзамаса">'
        if coat_uri
        else ""
    )
    goose_uri = goose_data_uri()
    goose_html = (
        f'<img class="goose" src="{goose_uri}" alt="Арзамасский гусь">'
        if goose_uri
        else ""
    )

    timestamp = time.strftime("%d.%m.%Y %H:%M")
    document = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Арзамас — дашборд кварталов</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <div class="header-inner">
    {coat_html}
    <div>
      <h1>Арзамас — дашборд кварталов</h1>
      <p>{fmt_int(len(blocks))} {plural(len(blocks), 'квартал', 'квартала', 'кварталов')} ·
      население, школы, доступность (авто и ОТ), плотность и морфотипы ·
      готовые GeoJSON BlocksNet · сгенерировано {timestamp}</p>
    </div>
    {goose_html}
  </div>
</header>
<nav class="toc">
  <a href="#conclusions">Выводы</a>
  <a href="#map">Карта</a>
  <a href="#charts">Графики</a>
  <a href="#summary">Все показатели</a>
  <a href="#categories">Категории</a>
  <a href="#rankings">Рейтинги</a>
  <a href="#full-table">Таблица кварталов</a>
</nav>
<main>
  <div class="cards">
  {cards_html}
  </div>

  <section id="conclusions">
    <h2>Выводы</h2>
    <div class="insights">
      <ul>
      {conclusions_html}
      </ul>
      <div class="note">{limitations_html}</div>
    </div>
  </section>

  <section id="map">
    <h2>Интерактивная карта</h2>
    <p class="lead">Слои переключаются в правом верхнем углу карты. Наведите курсор на квартал,
    чтобы увидеть все его показатели; размер круга школы пропорционален вместимости.</p>
    <iframe class="map-frame" title="Интерактивная карта кварталов Арзамаса"
            srcdoc="{map_srcdoc}"></iframe>
    <p class="map-hint">Чем насыщеннее цвет квартала, тем выше значение выбранного показателя.
    Слои доступности есть для автомобиля и общественного транспорта.
    Для доступности использована обратная шкала: зелёный — ближе, красный — дальше.</p>
  </section>

  <section id="charts">
    <h2>Графики</h2>
    <div class="charts">
    {charts_html}
    </div>
  </section>

  <section id="summary">
    <h2>Сводная таблица всех показателей</h2>
    <p class="lead">Описательные статистики по готовым данным. Для процентов указана доля
    норматива, для остальных величин — исходные единицы.</p>
    <div class="table-card">
      <div class="table-wrap">
        <table class="data-table">
          <thead><tr><th>Показатель</th><th>Группа</th><th>Ед.</th><th>N</th>
          <th>Среднее</th><th>Медиана</th><th>Мин.</th><th>Макс.</th><th>Пропуски</th></tr></thead>
          <tbody>{summary_html}</tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="categories">
    <h2>Категории</h2>
    <div class="table-grid">
    {categorical_html}
    </div>
  </section>

  <section id="rankings">
    <h2>Рейтинги кварталов</h2>
    <div class="table-grid">
    {rankings_html}
    </div>
  </section>

  <section id="full-table">
    <h2>Полная таблица кварталов</h2>
    <p class="lead">Все {len(FULL_COLUMNS)} колонок исходного файла <code>blocks_metrics.geojson</code> без геометрии.
    Поиск фильтрует строки, заголовки сортируют таблицу.</p>
    {full_table_html}
  </section>
</main>
<footer>
  <p>Источник данных: OpenStreetMap (© участники OSM, ODbL); расчёт показателей — BlocksNet.
  Скрипт читает готовые файлы <code>blocks_metrics.geojson</code>, <code>schools.geojson</code>,
  <code>boundary.geojson</code> и матрицы <code>acc_mx_drive.csv</code> / <code>acc_mx_intermodal.csv</code>;
  он не скачивает данные и не повторяет расчёты пайплайна.</p>
  <p>Сгенерировано {timestamp} · <code>{OUTPUT.name}</code></p>
</footer>
<script>{JS}</script>
</body>
</html>
"""
    OUTPUT.write_text(document, encoding="utf-8")
    print(f"Дашборд сохранён: {OUTPUT}")
    print(f"Размер: {OUTPUT.stat().st_size / 1e6:.2f} МБ")


if __name__ == "__main__":
    main()
