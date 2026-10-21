"""Fraud analyst dashboard (Streamlit).

Talks only to the streaming fraud service API (api/fraud_service.py):
live alerts, case status workflow, feedback labels and drift status.

Run from the project root:
    streamlit run dashboard/app.py
FRAUD_SERVICE_URL points it at the service (default http://127.0.0.1:8001).
"""

import os

import pandas as pd
import requests
import streamlit as st

SERVICE_URL = os.environ.get("FRAUD_SERVICE_URL", "http://127.0.0.1:8001")
# Address of the service as the browser sees it (differs in Kubernetes).
PUBLIC_SERVICE_URL = os.environ.get("FRAUD_SERVICE_PUBLIC_URL", SERVICE_URL)
REFRESH_SECONDS = int(os.environ.get("DASHBOARD_REFRESH_SECONDS", "5"))

st.set_page_config(page_title="Fraud Analyst Dashboard", layout="wide")


def api(method, path, **kwargs):
    try:
        response = requests.request(
            method, f"{SERVICE_URL}{path}", timeout=10, **kwargs
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"Fraud service unavailable at {SERVICE_URL}: {exc}")
        st.stop()


def to_table(records):
    if not records:
        return pd.DataFrame()
    table = pd.DataFrame(records)
    columns = [
        "id", "transaction_id", "step", "type", "amount", "name_orig",
        "name_dest", "fraud_score", "xgboost_probability", "gnn_probability",
        "decision", "case_status", "analyst_label", "actual_is_fraud",
        "scored_at",
    ]
    return table[[c for c in columns if c in table.columns]]


health = api("GET", "/")
options = api("GET", "/options")

st.title("Fraud Analyst Dashboard")
st.caption(
    f"Service {SERVICE_URL} · model {health['model_version']} · "
    f"threshold {health['threshold']} · GNN weight {health['gnn_weight']} · "
    f"Kafka consumer: {health['kafka_consumer']}"
)


# --------------------------------------------------
# Live KPIs and alert feed (auto-refreshing)
# --------------------------------------------------

@st.fragment(run_every=REFRESH_SECONDS)
def live_overview():
    stats = api("GET", "/stats")

    previous_alerts = st.session_state.get("alert_count")
    if previous_alerts is not None and stats["alerts"] > previous_alerts:
        st.toast(f"🚨 {stats['alerts'] - previous_alerts} new fraud alert(s)")
    st.session_state["alert_count"] = stats["alerts"]

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Scored", f"{stats['scored']:,}")
    k2.metric("Alerts", f"{stats['alerts']:,}")
    k3.metric("Open cases", f"{stats['open_cases']:,}")
    k4.metric("Labeled fraud", f"{stats['labeled_fraud']:,}")
    k5.metric("Labeled legitimate", f"{stats['labeled_legitimate']:,}")
    alert_precision = (
        stats["true_alerts"] / stats["alerts"] if stats["alerts"] else 0
    )
    k6.metric("Alert precision (replay labels)", f"{alert_precision:.1%}")

    st.subheader("Latest alerts")
    alerts = to_table(api(
        "GET", "/transactions", params={"limit": 50, "alerts_only": True}
    ))
    if alerts.empty:
        st.info("No alerts yet. Start the replay producer to stream transactions.")
    else:
        st.dataframe(alerts, hide_index=True, width="stretch")


alerts_tab, review_tab, all_tab, drift_tab = st.tabs(
    ["Alerts", "Case review", "All transactions", "Drift monitor"]
)

with alerts_tab:
    live_overview()


# --------------------------------------------------
# Case review: status workflow and feedback labels
# --------------------------------------------------

with review_tab:
    status_filter = st.selectbox(
        "Show cases with status", ["(any)"] + options["case_statuses"]
    )
    cases = api("GET", "/transactions", params={
        "limit": 200,
        "alerts_only": True,
        **({"status": status_filter} if status_filter != "(any)" else {}),
    })

    if not cases:
        st.info("No cases match.")
    else:
        labels = {
            c["id"]: (
                f"#{c['id']} · {c['type']} {c['amount']:,.2f} · "
                f"score {c['fraud_score']:.3f} · {c['case_status']}"
            )
            for c in cases
        }
        case_id = st.selectbox(
            "Case", list(labels), format_func=labels.get
        )
        case = api("GET", f"/cases/{case_id}")

        left, right = st.columns(2)
        with left:
            st.markdown("**Transaction**")
            st.json(case["transaction"])
            st.markdown("**Scores**")
            st.write({
                "fraud_score": case["fraud_score"],
                "xgboost_probability": case["xgboost_probability"],
                "gnn_probability": case["gnn_probability"],
                "threshold": case["threshold"],
                "model_version": case["model_version"],
                "actual_is_fraud (replay label)": case["actual_is_fraud"],
            })
        with right:
            st.markdown("**Features the model saw**")
            st.dataframe(
                pd.Series(case["features"], name="value").to_frame(),
                width="stretch",
            )

        with st.form("case_update"):
            status = st.selectbox(
                "Case status",
                options["case_statuses"],
                index=options["case_statuses"].index(case["case_status"])
                if case["case_status"] in options["case_statuses"] else 0,
            )
            label_choices = ["(none)"] + options["feedback_labels"]
            label = st.radio(
                "Feedback label (used in the next retraining)",
                label_choices,
                index=label_choices.index(case["analyst_label"])
                if case["analyst_label"] in label_choices else 0,
                horizontal=True,
            )
            note = st.text_area("Analyst note", case["analyst_note"] or "")
            if st.form_submit_button("Save case"):
                api("PATCH", f"/cases/{case_id}", json={
                    "status": status,
                    "label": None if label == "(none)" else label,
                    "note": note,
                })
                st.success(f"Case #{case_id} updated.")
                st.rerun()


# --------------------------------------------------
# All scored transactions
# --------------------------------------------------

with all_tab:
    limit = st.slider("Rows", 50, 1000, 200, step=50)
    table = to_table(api("GET", "/transactions", params={"limit": limit}))
    if table.empty:
        st.info("No scored transactions yet.")
    else:
        st.dataframe(table, hide_index=True, width="stretch")


# --------------------------------------------------
# Drift monitor
# --------------------------------------------------

with drift_tab:
    drift = api("GET", "/drift/latest")
    st.write(f"**Status:** {drift.get('status')}")
    if "drift_share" in drift:
        d1, d2, d3 = st.columns(3)
        d1.metric("Drifted feature share", f"{drift['drift_share']:.0%}")
        d2.metric("Retrain threshold", f"{drift['drift_share_threshold']:.0%}")
        d3.metric("Retrain triggered", "yes" if drift["retrain_triggered"] else "no")
        st.write("Checked at", drift["checked_at"],
                 "· drifted features:", ", ".join(drift["drifted_columns"]) or "none")
        st.dataframe(
            pd.DataFrame(drift["columns"]).T, width="stretch"
        )
        st.markdown(
            f"[Open the full Evidently report]({PUBLIC_SERVICE_URL}/drift/report)"
        )
