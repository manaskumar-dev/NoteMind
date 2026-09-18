from datetime import datetime, timedelta

INTERVALS_DAYS = [1, 3, 7, 15]  # stage 0..3; after last stage, cycle stays at 15


def next_review_date(stage: int, from_time: datetime | None = None) -> datetime:
    from_time = from_time or datetime.utcnow()
    idx = min(stage, len(INTERVALS_DAYS) - 1)
    return from_time + timedelta(days=INTERVALS_DAYS[idx])


def advance_stage(stage: int) -> int:
    return min(stage + 1, len(INTERVALS_DAYS) - 1)
