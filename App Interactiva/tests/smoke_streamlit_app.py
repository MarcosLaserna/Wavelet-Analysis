from streamlit.testing.v1 import AppTest


def main() -> None:
    app = AppTest.from_file("app.py")
    app.run(timeout=60)

    preferred_groups = ["Tecnologia", "Crypto", "Divisas", "Farmaceutica y Defensa"]
    available_groups = list(app.multiselect[0].options)
    selected_groups = [group for group in preferred_groups if group in available_groups]
    app.multiselect[0].set_value(selected_groups or available_groups[:2])
    app.run(timeout=60)

    selected_assets = list(app.multiselect[1].options)[:8]
    app.multiselect[1].set_value(selected_assets)
    app.run(timeout=120)

    if app.exception:
        raise RuntimeError(f"Streamlit exceptions: {app.exception}")

    expected_views = {
        "1. Arquetipoides",
        "2. K-Means",
        "3. Matrices",
        "4. Red",
        "5. Series",
        "6. Exportar",
    }
    rendered_views = set(app.radio[0].options)

    if not expected_views.issubset(rendered_views):
        missing = expected_views - rendered_views
        raise AssertionError(f"Missing views after automatic calculation: {sorted(missing)}")

    app.radio[0].set_value("4. Red")
    app.run(timeout=120)
    app.slider[0].set_value(0.55)
    app.run(timeout=120)

    if app.radio[0].value != "4. Red":
        raise AssertionError("Network view did not remain selected after rerun")

    if "7. Presentación" in app.radio[0].options:
        raise AssertionError("Presentation mode should not expose an extra view")

    print("OK - Streamlit smoke test passed")
    print(f"Dataset message: {app.success[0].value}")
    print(f"Metrics rendered: {len(app.metric)}")
    print(f"Dataframes rendered: {len(app.dataframe)}")


if __name__ == "__main__":
    main()
