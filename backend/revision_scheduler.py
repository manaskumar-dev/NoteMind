from datetime import datetime, timedelta

def calculate_next_revision(review_count: int):
    if review_count == 1:
        return datetime.utcnow() + timedelta(days=1)
    elif review_count == 2:
        return datetime.utcnow() + timedelta(days=3)
    elif review_count == 3:
        return datetime.utcnow() + timedelta(days=7)
    else:
        return datetime.utcnow() + timedelta(days=15)