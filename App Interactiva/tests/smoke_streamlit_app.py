from streamlit.testing.v1 import AppTest


def main() -> None:
    app = AppTest.from_file("app.py")
    app.run(timeout=60)

    selected_assets = list(app.multiselect[0].options)[:6]
    app.multiselect[0].set_value(selected_assets)
    app.slider[2].set_value(8)
    app.run(timeout=120)

    if app.exception:
        raise RuntimeError(f"Streamlit exceptions: {app.exception}")

    expected_views = {
        "1. Arquetipos",
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

    dataset_message = app.success[0].value
    model_message = app.success[1].value if len(app.success) > 1 else "Model view not selected"

    app.radio[0].set_value("4. Red")
    app.run(timeout=120)

    headers = {header.value for header in app.header}
    if "Red de interdependencia wavelet" not in headers:
        raise AssertionError("Network view did not render")

    app.slider[0].set_value(0.9)
    app.run(timeout=120)

    if app.radio[0].value != "4. Red":
        raise AssertionError("Network view was not preserved after threshold change")

    print("OK - Streamlit smoke test passed")
    print(f"Dataset message: {dataset_message}")
    print(f"Model message: {model_message}")
    print(f"Navigation views: {len(rendered_views)}")
    print(f"Metrics rendered: {len(app.metric)}")
    print(f"Dataframes rendered: {len(app.dataframe)}")


if __name__ == "__main__":
    main()
