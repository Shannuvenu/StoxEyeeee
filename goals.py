# goals.py
import os
import json
import uuid
from datetime import datetime, date

GOALS_FILE = "data/goals.json"
os.makedirs("data", exist_ok=True)

def _ensure_file():
    if not os.path.exists(GOALS_FILE):
        with open(GOALS_FILE, "w") as f:
            json.dump({"goals": []}, f)

def load_goals():
    _ensure_file()
    with open(GOALS_FILE, "r") as f:
        return json.load(f).get("goals", [])

def save_goals(goals):
    with open(GOALS_FILE, "w") as f:
        json.dump({"goals": goals}, f, indent=2)

def add_goal(name: str, target_amount: float, target_date: str, notes: str = "", emoji: str = "🎯"):
    """
    target_date must be ISO string: YYYY-MM-DD
    """
    goals = load_goals()
    new_goal = {
        "id": str(uuid.uuid4()),
        "name": name.strip(),
        "emoji": emoji,
        "target_amount": float(target_amount),
        "target_date": target_date,  # ISO
        "created_at": datetime.now().date().isoformat(),
        "notes": notes.strip()
    }
    goals.append(new_goal)
    save_goals(goals)
    return new_goal

def remove_goal(goal_id: str):
    goals = load_goals()
    goals = [g for g in goals if g.get("id") != goal_id]
    save_goals(goals)

def _parse_date(d):
    if isinstance(d, date):
        return d
    try:
        return datetime.fromisoformat(d).date()
    except Exception:
        # last resort
        return datetime.strptime(str(d), "%Y-%m-%d").date()

def evaluate_goal(portfolio_value: float, goal: dict, today: date | None = None):
    """
    Returns a dict with:
    - progress_pct
    - remaining_amount
    - days_left
    - on_track (bool)
    - need_per_day, need_per_month
    - reason (string)
    """
    today = today or datetime.now().date()
    target_amount = float(goal.get("target_amount", 0.0))
    created_at = _parse_date(goal.get("created_at"))
    target_date = _parse_date(goal.get("target_date"))

    # Safety guards
    if target_amount <= 0:
        target_amount = 1.0
    if target_date <= created_at:
        # avoid zero/negative duration
        target_date = created_at.replace(year=created_at.year + 1)

    total_days = max(1, (target_date - created_at).days)
    elapsed_days = max(0, (today - created_at).days)
    days_left = max(0, (target_date - today).days)

    progress_pct = max(0.0, min(100.0, (portfolio_value / target_amount) * 100.0))
    remaining_amount = max(0.0, target_amount - portfolio_value)

    # Simple pace rule: expected completion % by now = elapsed_days / total_days
    expected_pct_by_now = (elapsed_days / total_days) * 100.0
    on_track = progress_pct + 1e-9 >= expected_pct_by_now  # tiny epsilon

    need_per_day = remaining_amount / days_left if days_left > 0 else 0.0
    need_per_month = need_per_day * 30.0

    # Reason text
    if on_track:
        if remaining_amount <= 0:
            reason = "Target reached! 🎉"
        else:
            reason = (
                f"On track. You’ve completed {progress_pct:.1f}% vs expected {expected_pct_by_now:.1f}% by now."
            )
    else:
        lag_pct = max(0.0, expected_pct_by_now - progress_pct)
        reason = (
            f"Lagging by {lag_pct:.1f}% of target pace. You need about ₹{need_per_month:,.0f}/month to catch up."
        )

    return {
        "progress_pct": progress_pct,
        "remaining_amount": remaining_amount,
        "days_left": days_left,
        "on_track": on_track,
        "need_per_day": need_per_day,
        "need_per_month": need_per_month,
        "reason": reason
    }
