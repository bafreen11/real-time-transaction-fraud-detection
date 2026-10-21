import time

import numpy as np
import pandas as pd

from fraud_detection.config import (
    FEAST_DATA_DIR,
    PAYSIM_START_TIME,
    RAW_DATA_PATH,
    RECIPIENT_STATS_PATH,
    REPLAY_START_STEP,
    ensure_output_dirs,
)
from fraud_detection.feature_store import FEATURE_VIEW_NAME, get_store
from fraud_detection.features.behavioral import (
    RECIPIENT_STATE_FIELDS,
    add_event_timestamps,
    build_recipient_snapshots,
    is_tracked_recipient,
    update_recipient_state,
)

ONLINE_WRITE_BATCH = 50_000

ensure_output_dirs()

# --------------------------------------------------
# 1. Load raw transactions in file order
# --------------------------------------------------

print("Loading raw PaySim transactions...")
df = pd.read_csv(
    RAW_DATA_PATH,
    usecols=["step", "amount", "nameDest"],
)
df = add_event_timestamps(df)
print("Transactions:", len(df))

# --------------------------------------------------
# 2. Recipient state after every transaction (offline source)
# --------------------------------------------------

snapshots = build_recipient_snapshots(df)
snapshots.to_parquet(RECIPIENT_STATS_PATH, index=False)

print("Recipient snapshots:", len(snapshots))
print("Tracked recipient accounts:", snapshots["account_id"].nunique())
print("Saved offline source:", RECIPIENT_STATS_PATH)

# --------------------------------------------------
# 3. Check the offline builder against the streaming update
# --------------------------------------------------

# Replay the first rows one by one with update_recipient_state(), as the
# scoring service does, and compare with the vectorized snapshots.
check_rows = df.head(200_000)
check_rows = check_rows[is_tracked_recipient(check_rows["nameDest"])]

state = {}
streamed = []

for account, step, amount in zip(
    check_rows["nameDest"], check_rows["step"], check_rows["amount"]
):
    state[account] = update_recipient_state(
        state.get(account, {}), step, amount
    )
    streamed.append(state[account])

streamed = pd.DataFrame(streamed)[RECIPIENT_STATE_FIELDS]
offline = snapshots.head(len(streamed))[RECIPIENT_STATE_FIELDS]

if not np.allclose(streamed.to_numpy(float), offline.to_numpy(float)):
    raise ValueError("Offline snapshots and streaming updates disagree.")

print(f"Offline and streaming state match on {len(streamed)} rows.")

# --------------------------------------------------
# 4. Register the feature view and bootstrap the online store
# --------------------------------------------------

# A fresh online store, so no state from an earlier replay survives.
for name in ("online_store.db", "registry.db"):
    (FEAST_DATA_DIR / name).unlink(missing_ok=True)

store = get_store(apply_definitions=True)
print("Feast feature view registered.")

# The online store gets each account's latest state from before
# REPLAY_START_STEP (what `feast materialize` up to that step would
# write). The producer replays from that step and the scoring service
# keeps the store up to date. Only accounts that receive a transaction
# in the replay window are written: no other account is ever looked
# up, and writing all ~2M accounts to SQLite takes hours.
replay_start = (
    pd.Timestamp(PAYSIM_START_TIME)
    + pd.Timedelta(hours=REPLAY_START_STEP - 1)
)
replay_accounts = df.loc[df["step"] >= REPLAY_START_STEP, "nameDest"].unique()

history = snapshots[
    (snapshots["event_timestamp"] < replay_start)
    & snapshots["account_id"].isin(replay_accounts)
]
latest = history.groupby("account_id", sort=False).tail(1)

print(f"Writing state of {len(latest)} recipient accounts "
      f"(history before step {REPLAY_START_STEP}) to the online store...")
started = time.time()
for start in range(0, len(latest), ONLINE_WRITE_BATCH):
    store.write_to_online_store(
        FEATURE_VIEW_NAME, latest.iloc[start:start + ONLINE_WRITE_BATCH]
    )
    print(f"  {min(start + ONLINE_WRITE_BATCH, len(latest))}/{len(latest)}")
print(f"Online store written in {time.time() - started:.0f}s.")
print("Feast feature store ready.")
