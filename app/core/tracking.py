
from collections import defaultdict, deque


class TrafficCounter:
    """
    Track object movement and count line crossings.
    """

    def __init__(self, class_names, line_position_ratio=0.50):
        self.track_history = defaultdict(
            lambda: deque(maxlen=2)
        )

        self.counted_ids = set()

        self.counts = {
            name: {
                "Up": 0,
                "Down": 0
            }
            for name in class_names.values()
        }

        self.line_position_ratio = line_position_ratio

    def update(self, track_id, class_name, center_x, center_y, line_y):
        """
        Update an object's position and count its
        first crossing of the horizontal line.

        Returns:
            "Up", "Down", or None
        """

        if track_id is None:
            return None

        self.track_history[track_id].append(
            (center_x, center_y)
        )

        if len(self.track_history[track_id]) < 2:
            return None

        previous_y = self.track_history[track_id][0][1]
        current_y = self.track_history[track_id][1][1]

        if track_id in self.counted_ids:
            return None

        direction = None

        if previous_y < line_y <= current_y:
            direction = "Down"

        elif previous_y > line_y >= current_y:
            direction = "Up"

        if direction is not None:
            self.counts[class_name][direction] += 1
            self.counted_ids.add(track_id)

        return direction
