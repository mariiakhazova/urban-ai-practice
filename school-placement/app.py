"""
Интерактивное приложение для оценки вариантов размещения школ.

Streamlit-приложение с картой кварталов, расчётами обеспеченности
и автоматическим подбором оптимального квартала для школы.
"""

import io
import tempfile
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import folium_static

import auto_placement
import data_loader
import engine
import map_utils

st.set_page_config(page_title="Размещение школ", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "examples" / "data" / "arzamas"


@st.cache_resource
def load_all_data():
    try:
        blocks = data_loader.load_blocks()
        acc_mx = data_loader.load_accessibility_matrix()
        boundary = data_loader.load_boundary()
        schools = data_loader.load_schools()
        validation = data_loader.validate_data(blocks, acc_mx)
        if not validation["ok"]:
            st.error(f"Ошибки в данных: {validation['errors']}")
        if validation["warnings"]:
            st.warning(f"Предупреждения: {validation['warnings']}")
        return {
            "blocks": blocks,
            "acc_mx": acc_mx,
            "boundary": boundary,
            "schools": schools,
            "validation": validation,
        }
    except FileNotFoundError as e:
        st.error(f"Ошибка загрузки данных: {e}")
        st.info(
            "Папка с данными не найдена. "
            "Убедитесь, что проект python-package-boilerplate находится рядом с school-placement."
        )
        st.stop()
    except Exception as e:
        st.error(f"Неожиданная ошибка при загрузке данных: {e}")
        st.stop()


@st.cache_resource
def compute_base_provision(_blocks, _acc_mx):
    prov_input = data_loader.load_provision_input(_blocks)
    base_prov = engine.compute_provision(
        prov_input["population"], prov_input["capacity"], _acc_mx
    )
    return base_prov


def run_auto_placement(base_pop, base_cap, acc_mx, capacity):
    progress_bar = st.progress(0, text="Перебор кварталов...")
    status_text = st.empty()

    def progress_callback(checked, total, block_id, opt, n_errors):
        progress = checked / total
        progress_bar.progress(progress)
        remaining = total - checked
        status_text.markdown(
            f"Проверено: **{checked}**/{total} кварталов | Осталось: **{remaining}**"
        )

    result = auto_placement.auto_placement(
        base_pop, base_cap, acc_mx, capacity, progress_callback=progress_callback
    )

    progress_bar.progress(1.0, text="Завершено!")
    status_text.empty()

    return result


def render_scenario_selection(st, blocks, base_prov, default_cap=600):
    st.subheader("Ручной выбор кварталов для школ")

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        mode = st.radio("Сценарий размещения", ["Одна школа", "Две школы"])

    with col2:
        if mode == "Одна школа":
            capacity1 = st.number_input(
                "Вместимость школы, мест", min_value=1, value=default_cap, step=50
            )
            block1 = st.number_input(
                "ID квартала", min_value=0, max_value=len(blocks) - 1, value=0, step=1
            )
            additions = {int(block1): int(capacity1)}
        else:
            capacity1 = st.number_input(
                "Вместимость первой школы, мест",
                min_value=1,
                value=300,
                step=50,
                key="cap1",
            )
            block1 = st.number_input(
                "ID первой школы", min_value=0, max_value=len(blocks) - 1, value=0, step=1
            )
            capacity2 = st.number_input(
                "Вместимость второй школы, мест",
                min_value=1,
                value=300,
                step=50,
                key="cap2",
            )
            block2 = st.number_input(
                "ID второй школы", min_value=0, max_value=len(blocks) - 1, value=1, step=1
            )
            additions = {int(block1): int(capacity1), int(block2): int(capacity2)}

    with col3:
        reset = st.button("Сбросить")

    return additions, reset


def render_auto_placement(st, blocks, base_pop, base_cap, acc_mx, base_deficit):
    st.subheader("Автоматический подбор квартала")

    col1, col2 = st.columns([2, 1])

    with col1:
        capacity = st.number_input(
            "Вместимость новой школы, мест",
            min_value=1,
            value=600,
            step=50,
            key="auto_cap",
        )

    with col2:
        start_button = st.button("Разместить автоматически")

    if start_button or "auto_result" in st.session_state:
        if start_button:
            with st.spinner(
                f"Перебор всех {len(blocks)} кварталов... Это может занять несколько минут."
            ):
                result = run_auto_placement(
                    base_pop, base_cap, acc_mx, capacity
                )
                st.session_state.auto_result = result
                st.session_state.auto_capacity = capacity
                st.session_state.auto_base_deficit = base_deficit
        else:
            result = st.session_state.auto_result
            capacity = st.session_state.auto_capacity
            base_deficit = st.session_state.auto_base_deficit

        st.success(
            f"Проверено {result.n_checked}/{result.n_total} кварталов, "
            f"{result.n_errors} ошибок"
        )

        if result.incomplete:
            st.warning("Перебор не завершён. Найдены лучшие среди проверенных кварталов.")

        if result.best_options:
            best_ids = [o.block_id for o in result.best_options]
            if len(best_ids) == 1:
                st.info(
                    f"Лучший квартал: {best_ids[0]} "
                    f"(сокращение дефицита на {result.best_options[0].deficit_reduction:.1f} мест)"
                )
            else:
                st.info(
                    f"Лучшие кварталы ({len(best_ids)} шт., "
                    f"одинаковое сокращение дефицита на {result.best_options[0].deficit_reduction:.1f} мест): "
                    f"{', '.join(map(str, best_ids[:5]))}"
                    + (", ..." if len(best_ids) > 5 else "")
                )

        options_df = auto_placement.auto_placement_to_df(result)
        st.dataframe(
            options_df.head(20),
            use_container_width=True,
            column_config={
                "block_id": st.column_config.NumberColumn("ID квартала"),
                "capacity_added": st.column_config.NumberColumn("Добавлено мест"),
                "deficit": st.column_config.NumberColumn("Остаточный дефицит"),
                "deficit_reduction": st.column_config.NumberColumn("Сокращение дефицита"),
                "is_best": st.column_config.CheckboxColumn("Лучший"),
                "error": st.column_config.TextColumn("Ошибка"),
            },
        )

        if result.best_options:
            return result.best_options[0].block_id, capacity, result.best_options[0].deficit
    else:
        return None, capacity, base_deficit


def render_comparison_table(base_prov, base_cap, base_summ):
    st.subheader("Сравнение сценариев")

    if "scenarios" not in st.session_state:
        st.session_state.scenarios = []

    scenario_list = st.session_state.scenarios

    if len(scenario_list) > 0:
        comp_df = pd.DataFrame(scenario_list)
        comp_df["№"] = range(1, len(comp_df) + 1)
        comp_df = comp_df.set_index("№")
        st.dataframe(
            comp_df,
            use_container_width=True,
            column_config={
                "name": "Сценарий",
                "total_capacity": "Общая вместимость",
                "capacity_increase": "Прирост мест",
                "total_deficit": "Остаточный дефицит",
                "deficit_reduction": "Сокращение дефицита",
                "deficit_reduction_pct": "Сокращение, %",
            },
        )

        csv = comp_df.to_csv(index=True).encode("utf-8-sig")
        st.download_button(
            "Скачать таблицу (CSV)",
            data=csv,
            file_name="scenarios_comparison.csv",
            mime="text/csv",
        )


def main():
    st.title("Интерактивное приложение для оценки вариантов размещения школ")
    st.markdown("---")

    st.markdown(
        """
**Назначение**: Оценка обеспеченности населения местами в школах для разных вариантов размещения.

**Методика расчёта**: Конкурентное распределение спроса (competitive_provision) из библиотеки BlocksNet.
На основе населения, вместимости школ и матрицы времени в пути определяется,
какая часть спроса удовлетворяется в пределах доступности (15 минут) и без ограничения.

**Данные**:
- Кварталы: Арзамас, 403 квартала
- Население: 85 933 чел.
- Школы: 17 школ, 4 250 мест
- Дефицит мест: 6 199 мест
- Источник: результаты практики 2 (python-package-boilerplate/examples/data/arzamas/)

**Допущения**:
- Потребность в местах: 120 мест на 1000 жителей
- Доступность: 15 минут на автомобильном транспорте
- Размещение школы увеличивает вместимость соответствующего квартала
"""
    )

    data = load_all_data()
    blocks = data["blocks"]
    acc_mx = data["acc_mx"]
    boundary = data["boundary"]
    schools = data["schools"]
    validation = data["validation"]

    summ = validation["summary"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Кварталов", summ["n_blocks"])
    col2.metric("Население", f"{int(summ['total_population']):,}")
    col3.metric("Школ", summ["n_with_capacity"])
    col4.metric("Мест в школах", f"{int(summ['total_capacity']):,}")

    st.markdown("---")

    base_prov = compute_base_provision(blocks, acc_mx)
    base_deficit = engine.total_deficit(base_prov)
    base_summ = engine.scenario_summary(base_prov, base_deficit, summ["total_capacity"], {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Потребность", f"{base_summ['total_demand']:,} мест")
    col2.metric("Вместимость", f"{base_summ['total_capacity']:,} мест")
    col3.metric("Дефицит", f"{int(base_summ['total_deficit']):,} мест")
    col4.metric(
        "Обеспеченность", f"{base_summ['mean_provision_strong']:.1%} (15 мин)"
    )

    st.markdown("---")

    with st.expander("Информация о данных и источниках", expanded=False):
        st.markdown(
            f"""
- Исходные файлы: `{DATA_DIR}`
- Версия BlocksNet: 1.0.0a9
- Параметры школ:
  - Потребность: {data_loader.SCHOOL_DEMAND} мест на 1000 жителей
  - Доступность: {data_loader.SCHOOL_ACCESSIBILITY} минут
- Матрица доступности:
  - Размер: {acc_mx.shape}
  - Среднее время: {summ['mx_mean']:.2f} мин
  - Макс: {summ['mx_max']:.2f} мин
"""
        )
        if validation["warnings"]:
            st.warning("Предупреждения валидации:")
            for w in validation["warnings"]:
                st.warning(f"- {w}")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Ручной выбор", "Автоматический подбор", "Сравнение"])

    with tab1:
        base_pop = blocks["population"]
        base_cap = blocks["capacity"]
        additions, reset = render_scenario_selection(st, blocks, base_prov)

        if "reset_last" not in st.session_state:
            st.session_state.reset_last = False
        if reset or st.session_state.reset_last:
            st.session_state.additions = {}
            st.session_state.reset_last = False
        else:
            st.session_state.additions = additions

        current_additions = st.session_state.additions

        if current_additions:
            st.info(
                f"Добавлено школ: {len(current_additions)}, "
                f"вместимость: {sum(current_additions.values())} мест"
            )
            st.write(f"Кварталы: {list(current_additions.keys())}")

        calc_button = st.button("Рассчитать сценарий")

        if calc_button or "scen_prov" in st.session_state:
            if calc_button:
                st.session_state.last_additions = current_additions.copy()

            additions_to_use = st.session_state.get("last_additions", {})
            scen_prov = engine.compute_scenario(
                base_pop, base_cap, acc_mx, additions_to_use
            )
            st.session_state.scen_prov = scen_prov
            st.session_state.scen_additions = additions_to_use

            scen_summ = engine.scenario_summary(
                scen_prov, base_deficit, summ["total_capacity"], additions_to_use
            )

            st.info("Результаты расчёта:")

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Вместимость", f"{scen_summ['total_capacity']:,}")
            col2.metric("Прирост мест", f"+{scen_summ['capacity_increase']:,}")
            col3.metric(
                "Дефицит", f"{int(scen_summ['total_deficit']):,}", delta=-int(scen_summ['deficit_reduction'])
            )
            col4.metric("Сокращение, %", f"{scen_summ['deficit_reduction_pct']:.1f}%")
            col5.metric(
                "Обеспеченность", f"{scen_summ['mean_provision_strong']:.1%}", delta=f"+{scen_summ['mean_provision_strong'] - base_summ['mean_provision_strong']:.1%}"
            )

            st.markdown("---")

            details = engine.block_details(scen_prov, base_pop, base_cap)

            if additions_to_use:
                st.subheader("Карточки выбранных кварталов")
                for block_id, added_cap in additions_to_use.items():
                    if block_id in details.index:
                        row = details.loc[block_id]
                        with st.expander(f"Квартал {block_id} (добавлено {added_cap} мест)", expanded=True):
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Население", f"{int(row['population']):,}")
                                st.metric("Потребность", f"{int(row['demand']):,}")
                            with col_b:
                                st.metric("Баз. вместимость", f"{int(row['base_capacity']):,}")
                                st.metric("Вместимость всего", f"{int(row['capacity']):,}")
                            with col_c:
                                st.metric("Дефицит", f"{int(row['demand_left']):,}")
                                st.metric("Обеспеченность (15 мин)", f"{row['provision_strong']:.2%}")
                                st.metric("Обеспеченность (без огранич.)", f"{row['provision_weak']:.2%}")

            st.markdown("---")
            details["capacity_added"] = details.index.map(
                lambda x: additions_to_use.get(x, 0)
            )
            details = details.sort_values("population", ascending=False)
            st.dataframe(
                details.head(50),
                use_container_width=True,
                column_config={
                    "population": st.column_config.NumberColumn("Население"),
                    "base_capacity": st.column_config.NumberColumn("Баз. вместимость"),
                    "demand": st.column_config.NumberColumn("Потребность"),
                    "capacity": st.column_config.NumberColumn("Вместимость"),
                    "capacity_added": st.column_config.NumberColumn("Добавлено"),
                    "demand_left": st.column_config.NumberColumn("Дефицит"),
                    "provision_strong": st.column_config.NumberColumn(
                        "Обеспеченность (15 мин)", format="%.4f"
                    ),
                    "provision_weak": st.column_config.NumberColumn(
                        "Обеспеченность (без огранич.)", format="%.4f"
                    ),
                },
            )

            csv = details.to_csv(index=True).encode("utf-8-sig")
            st.download_button(
                "Скачать результаты (CSV)",
                data=csv,
                file_name="scenario_results.csv",
                mime="text/csv",
            )

            st.markdown("---")
            st.subheader("Карта")

            prov_display = details["provision_strong"]
            block_ids = [x for x in additions_to_use.keys() if additions_to_use[x] > 0]

            m = map_utils.make_block_map(
                blocks,
                prov_display,
                title="Обеспеченность местами в школах (в пределах 15 мин)",
                value_label="Обеспеченность (доля)",
                color_label="Обеспеченность",
                schools_gdf=schools,
                boundary_gdf=boundary,
                highlight_blocks=block_ids,
                vmin=0,
                vmax=1,
            )
            folium_static(m, width=1200, height=600)

            summ["name"] = f"Сценарий: {len(additions_to_use)} школ"
            scen_summ["name"] = f"Ручной выбор ({len(additions_to_use)} школ)"
            st.session_state.scenarios = [base_summ, scen_summ]

    with tab2:
        base_pop = blocks["population"]
        base_cap = blocks["capacity"]

        best_block_id, best_capacity, best_deficit = render_auto_placement(
            st, blocks, base_pop, base_cap, acc_mx, base_deficit
        )

        if best_block_id is not None:
            st.markdown("---")
            st.subheader("Карта лучшего варианта")

            additions = {best_block_id: best_capacity}
            scen_prov = engine.compute_scenario(base_pop, base_cap, acc_mx, additions)
            details = engine.block_details(scen_prov, base_pop, base_cap)

            prov_display = details["provision_strong"]
            m = map_utils.make_block_map(
                blocks,
                prov_display,
                title=f"Обеспеченность после размещения школы на {best_capacity} мест в квартале {best_block_id}",
                value_label="Обеспеченность (доля)",
                color_label="Обеспеченность",
                schools_gdf=schools,
                boundary_gdf=boundary,
                highlight_blocks=[best_block_id],
                vmin=0,
                vmax=1,
            )
            folium_static(m, width=1200, height=600)

            st.markdown("---")
            st.subheader("Карточка лучшего квартала")

            if best_block_id in details.index:
                row = details.loc[best_block_id]
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Население", f"{int(row['population']):,}")
                    st.metric("Потребность", f"{int(row['demand']):,}")
                with col_b:
                    st.metric("Баз. вместимость", f"{int(row['base_capacity']):,}")
                    st.metric("Вместимость всего", f"{int(row['capacity']):,}")
                with col_c:
                    st.metric("Дефицит", f"{int(row['demand_left']):,}")
                    st.metric("Обеспеченность (15 мин)", f"{row['provision_strong']:.2%}")
                    st.metric("Обеспеченность (без огранич.)", f"{row['provision_weak']:.2%}")

            scen_summ = engine.scenario_summary(
                scen_prov, base_deficit, summ["total_capacity"], additions
            )
            scen_summ["name"] = f"Авто: квартал {best_block_id}"
            if "scenarios" in st.session_state and len(st.session_state.scenarios) >= 2:
                st.session_state.scenarios = st.session_state.scenarios[:2] + [scen_summ]

    with tab3:
        render_comparison_table(base_prov, base_cap, base_summ)

    st.markdown("---")
    st.markdown(
        """
**Важно**: Выбор земельного участка и оценка возможности строительства требуют отдельного градостроительного обоснования.
Это приложение сравнивает только обеспеченность населения местами в школах для разных вариантов их размещения по кварталам.
"""
    )


if __name__ == "__main__":
    main()
