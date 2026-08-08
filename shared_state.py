import threading

lock = threading.Lock()

latest_frame = None
latest_score = 0
latest_status = "Waiting"

score_history = []