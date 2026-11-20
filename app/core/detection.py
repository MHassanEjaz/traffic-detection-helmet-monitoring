
# Vehicle and person class configuration

CLASS_NAMES = {
    0: "Person",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

CLASS_COLORS = {
    0: (255, 255, 0),
    2: (0, 255, 0),
    3: (255, 0, 255),
    5: (0, 165, 255),
    7: (255, 100, 0)
}


def is_motorcycle_rider(person_box, motorcycle_boxes):
    """
    Check whether a detected person is associated
    with a detected motorcycle.
    """

    px1, py1, px2, py2 = person_box

    person_center_x = (px1 + px2) / 2
    person_bottom_y = py2

    for bx1, by1, bx2, by2 in motorcycle_boxes:

        bike_center_x = (bx1 + bx2) / 2
        bike_width = bx2 - bx1

        horizontal_match = (
            bx1 - 0.25 * bike_width
            <= person_center_x
            <= bx2 + 0.25 * bike_width
        )

        vertical_match = (
            person_bottom_y >= by1
            and py1 <= by2
        )

        center_match = (
            abs(person_center_x - bike_center_x)
            <= max(bike_width * 0.75, 30)
        )

        if horizontal_match and (vertical_match or center_match):
            return True

    return False
