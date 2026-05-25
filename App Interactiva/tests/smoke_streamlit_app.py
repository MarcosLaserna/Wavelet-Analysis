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

    expected_tabs = {
        "1. Arquetipos",
        "2. K-Means",
        "3. Matrices",
        "4. Series",
        "5. Exportar",
    }
    rendered_tabs = {tab.label for tab in app.tabs}

    if not expected_tabs.issubset(rendered_tabs):
        missing = expected_tabs - rendered_tabs
        raise AssertionError(f"Missing tabs after automatic calculation: {sorted(missing)}")

    print("OK - Streamlit smoke test passed")
    print(f"Dataset message: {app.success[0].value}")
    print(f"Model message: {app.success[1].value}")
    print(f"Metrics rendered: {len(app.metric)}")
    print(f"Dataframes rendered: {len(app.dataframe)}")


if __name__ == "__main__":
    main()
