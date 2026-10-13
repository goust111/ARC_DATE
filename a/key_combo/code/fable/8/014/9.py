import random
from collections import Counter

import numpy as np


SIDE_ORDER = ("top", "bottom", "left", "right")
FRAME_MARGIN = 1
BACKGROUND_COLOR = 0
MAX_GENERATION_ATTEMPTS = 400
MIN_BBOX_GAP = 2
INPUT_SIZE_RANGE = (16, 26)
OBJECT_COUNT_CHOICES = (2, 3, 4)
OBJECT_COUNT_WEIGHTS = (2, 3, 5)
ALL_COLORS = tuple(range(1, 10))


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _unique_side_color(vals: np.ndarray, side_name: str) -> int:
    uniq = np.unique(vals)
    assert uniq.size == 1, f"{side_name} border must use exactly one color"
    return int(uniq[0])


def _frame_side_colors(grid: np.ndarray, bg: int) -> dict[str, int]:
    h, w = grid.shape
    assert h >= 5 and w >= 5, "frame puzzle needs room for an inner area"
    assert int(grid[0, 0]) == bg
    assert int(grid[0, w - 1]) == bg
    assert int(grid[h - 1, 0]) == bg
    assert int(grid[h - 1, w - 1]) == bg
    sides = {
        "top": _unique_side_color(grid[0, 1:-1], "top"),
        "bottom": _unique_side_color(grid[h - 1, 1:-1], "bottom"),
        "left": _unique_side_color(grid[1:-1, 0], "left"),
        "right": _unique_side_color(grid[1:-1, w - 1], "right"),
    }
    assert len(set(sides.values())) == 4, "frame sides must have distinct colors"
    return sides


def _analyze_object(
    grid: np.ndarray,
    coords: np.ndarray,
    frame_sides: dict[str, int],
) -> dict:
    bbox = _bbox_from_coords(coords)
    r0, c0, r1, c1 = bbox
    vals = grid[coords[:, 0], coords[:, 1]]
    colors = sorted(set(map(int, vals)))
    assert len(colors) == 2, "each fractured object must contain exactly two colors"

    fracture = None
    for color in colors:
        sub = coords[vals == color]
        rows = sorted(set(map(int, sub[:, 0])))
        cols = sorted(set(map(int, sub[:, 1])))
        if len(rows) == 1 and cols == list(range(c0, c1 + 1)):
            fracture = {"color": color, "orientation": "horizontal", "index": rows[0]}
            break
        if len(cols) == 1 and rows == list(range(r0, r1 + 1)):
            fracture = {"color": color, "orientation": "vertical", "index": cols[0]}
            break
    assert fracture is not None, "failed to find a full-width or full-height fracture line"

    body_coords = coords[vals != fracture["color"]]
    body_vals = grid[body_coords[:, 0], body_coords[:, 1]]
    body_colors = sorted(set(map(int, body_vals)))
    assert len(body_colors) == 1, "body must be monochrome"
    body_color = body_colors[0]

    matched_sides = [side for side, color in frame_sides.items() if color == fracture["color"]]
    assert len(matched_sides) == 1, "fracture color must match exactly one frame side"
    side = matched_sides[0]

    if fracture["orientation"] == "horizontal":
        fr = fracture["index"]
        top = body_coords[body_coords[:, 0] < fr]
        bottom = body_coords[body_coords[:, 0] > fr]
        assert top.size and bottom.size, "horizontal fracture must split object into two nonempty halves"
        if side == "top":
            moving = top
            delta = 1 - int(moving[:, 0].min())
        else:
            assert side == "bottom", "horizontal fracture must map to top or bottom"
            moving = bottom
            delta = (grid.shape[0] - 2) - int(moving[:, 0].max())
        dr, dc = delta, 0
    else:
        fc = fracture["index"]
        left = body_coords[body_coords[:, 1] < fc]
        right = body_coords[body_coords[:, 1] > fc]
        assert left.size and right.size, "vertical fracture must split object into two nonempty halves"
        if side == "left":
            moving = left
            delta = 1 - int(moving[:, 1].min())
        else:
            assert side == "right", "vertical fracture must map to left or right"
            moving = right
            delta = (grid.shape[1] - 2) - int(moving[:, 1].max())
        dr, dc = 0, delta

    moved = moving + np.array([dr, dc], dtype=int)
    assert moved.size > 0
    assert moved[:, 0].min() >= 1 and moved[:, 0].max() <= grid.shape[0] - 2
    assert moved[:, 1].min() >= 1 and moved[:, 1].max() <= grid.shape[1] - 2

    fracture_coords = coords[vals == fracture["color"]]
    return {
        "body_color": body_color,
        "moving": moving,
        "moved": moved,
        "fracture": fracture_coords,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase each colored fracture line and slide only the half that faces the matching frame side until it touches that side.

    Input structure:
    - The grid has a one-cell outer frame whose four sides have four different colors, with background-color corners.
    - Inside the frame are several single-color body objects, each containing one one-cell-wide fracture line of a second color.
    - Each fracture line is straight, spans the full width or height of its object's bbox, and its color matches exactly one frame side.

    Rule:
    - Keep the frame unchanged.
    - For each object, erase the fracture line to background.
    - Split the body into the two halves on opposite sides of the fracture, then move only the half that is closer to the matching frame side in a straight line perpendicular to the fracture until it touches the inner edge of that side; leave the other half in place.

    Key insight:
    - The fracture is a directional sign, not a symmetric crack: its color chooses the destination side, and only the half already facing that side migrates there.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    frame_sides = _frame_side_colors(g, bg)

    work = g.copy()
    work[0, :] = bg
    work[-1, :] = bg
    work[:, 0] = bg
    work[:, -1] = bg

    components = _find_components(work != bg)
    assert components, "expected at least one fractured object"

    plans = [_analyze_object(g, coords, frame_sides) for coords in components]

    out = g.copy()
    for plan in plans:
        out[plan["fracture"][:, 0], plan["fracture"][:, 1]] = bg
        out[plan["moving"][:, 0], plan["moving"][:, 1]] = bg

    occupied = out != bg
    for plan in plans:
        rr = plan["moved"][:, 0]
        cc = plan["moved"][:, 1]
        assert not np.any(occupied[rr, cc]), "moved halves must not overlap other preserved pixels"
        out[rr, cc] = plan["body_color"]
        occupied[rr, cc] = True
    return out


def _interval_sep(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return b0 - a1
    if b1 < a0:
        return a0 - b1
    return 0


def _cheb_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    dr = _interval_sep(a[0], a[2], b[0], b[2])
    dc = _interval_sep(a[1], a[3], b[1], b[3])
    return max(dr, dc)


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _sample_sides(count: int) -> list[str]:
    if count == 4:
        sides = list(SIDE_ORDER)
        random.shuffle(sides)
        return sides
    horizontal = random.sample(("top", "bottom"), 1)
    vertical = random.sample(("left", "right"), 1)
    sides = horizontal + vertical
    remaining = [side for side in SIDE_ORDER if side not in sides]
    while len(sides) < count:
        choice = random.choice(remaining)
        remaining.remove(choice)
        sides.append(choice)
    random.shuffle(sides)
    return sides


def _sample_object_shape(side: str) -> dict:
    if side in {"top", "bottom"}:
        width = random.randint(2, 4)
        total_h = random.randint(width + 1, min(width + 4, 8))
        body_h = total_h - 1
        top_h = random.randint(1, body_h - 1)
        bottom_h = body_h - top_h
        return {
            "orientation": "horizontal",
            "h": total_h,
            "w": width,
            "top_h": top_h,
            "bottom_h": bottom_h,
        }
    height = random.randint(2, 4)
    total_w = random.randint(height + 1, min(height + 4, 8))
    body_w = total_w - 1
    left_w = random.randint(1, body_w - 1)
    right_w = body_w - left_w
    return {
        "orientation": "vertical",
        "h": height,
        "w": total_w,
        "left_w": left_w,
        "right_w": right_w,
    }


def _object_output_boxes(spec: dict, h: int, w: int) -> list[tuple[int, int, int, int]]:
    r0, c0 = spec["r0"], spec["c0"]
    if spec["orientation"] == "horizontal":
        obj_h = spec["h"]
        obj_w = spec["w"]
        top_h = spec["top_h"]
        bottom_h = spec["bottom_h"]
        if spec["side"] == "top":
            moved = (1, c0, top_h, c0 + obj_w - 1)
            stay = (r0 + top_h + 1, c0, r0 + obj_h - 1, c0 + obj_w - 1)
        else:
            moved = (h - 1 - bottom_h, c0, h - 2, c0 + obj_w - 1)
            stay = (r0, c0, r0 + top_h - 1, c0 + obj_w - 1)
        return [moved, stay]

    obj_h = spec["h"]
    obj_w = spec["w"]
    left_w = spec["left_w"]
    right_w = spec["right_w"]
    if spec["side"] == "left":
        moved = (r0, 1, r0 + obj_h - 1, left_w)
        stay = (r0, c0 + left_w + 1, r0 + obj_h - 1, c0 + obj_w - 1)
    else:
        moved = (r0, w - 1 - right_w, r0 + obj_h - 1, w - 2)
        stay = (r0, c0, r0 + obj_h - 1, c0 + left_w - 1)
    return [moved, stay]


def _moving_sweep_box(spec: dict, h: int, w: int) -> tuple[int, int, int, int]:
    r0, c0 = spec["r0"], spec["c0"]
    if spec["orientation"] == "horizontal":
        obj_w = spec["w"]
        if spec["side"] == "top":
            start = (r0, c0, r0 + spec["top_h"] - 1, c0 + obj_w - 1)
            end = (1, c0, spec["top_h"], c0 + obj_w - 1)
        else:
            start = (r0 + spec["top_h"] + 1, c0, r0 + spec["h"] - 1, c0 + obj_w - 1)
            end = (h - 1 - spec["bottom_h"], c0, h - 2, c0 + obj_w - 1)
    else:
        obj_h = spec["h"]
        if spec["side"] == "left":
            start = (r0, c0, r0 + obj_h - 1, c0 + spec["left_w"] - 1)
            end = (r0, 1, r0 + obj_h - 1, spec["left_w"])
        else:
            start = (r0, c0 + spec["left_w"] + 1, r0 + obj_h - 1, c0 + spec["w"] - 1)
            end = (r0, w - 1 - spec["right_w"], r0 + obj_h - 1, w - 2)
    return (
        min(start[0], end[0]),
        min(start[1], end[1]),
        max(start[2], end[2]),
        max(start[3], end[3]),
    )


def _candidate_positions(shape: dict, side: str, h: int, w: int) -> list[tuple[int, int]]:
    obj_h = shape["h"]
    obj_w = shape["w"]
    if side == "top":
        row_range = range(max(3, h // 2 - 1), h - obj_h - 1)
        col_range = range(2, w - obj_w - 1)
    elif side == "bottom":
        row_range = range(2, min(h // 2, h - obj_h - 1))
        col_range = range(2, w - obj_w - 1)
    elif side == "left":
        row_range = range(2, h - obj_h - 1)
        col_range = range(max(3, w // 2 - 1), w - obj_w - 1)
    else:
        row_range = range(2, h - obj_h - 1)
        col_range = range(2, min(w // 2, w - obj_w - 1))
    candidates = [(r0, c0) for r0 in row_range for c0 in col_range]
    random.shuffle(candidates)
    return candidates


def _can_place(spec: dict, placed: list[dict], h: int, w: int) -> bool:
    input_box = (spec["r0"], spec["c0"], spec["r0"] + spec["h"] - 1, spec["c0"] + spec["w"] - 1)
    output_boxes = _object_output_boxes(spec, h, w)
    sweep = _moving_sweep_box(spec, h, w)
    for other in placed:
        if _cheb_gap(input_box, other["input_box"]) < MIN_BBOX_GAP:
            return False
        for box in output_boxes:
            if any(_cheb_gap(box, obox) < MIN_BBOX_GAP for obox in other["output_boxes"]):
                return False
        if _bbox_overlap(sweep, other["input_box"]):
            return False
        if any(_bbox_overlap(sweep, obox) for obox in other["output_boxes"]):
            return False
        if _bbox_overlap(sweep, other["sweep"]):
            return False
    spec["input_box"] = input_box
    spec["output_boxes"] = output_boxes
    spec["sweep"] = sweep
    return True


def _place_objects(h: int, w: int, frame_sides: dict[str, int], body_colors: list[int], sides: list[str]) -> list[dict] | None:
    placed: list[dict] = []
    for side, body_color in zip(sides, body_colors):
        shape = _sample_object_shape(side)
        spec = {
            **shape,
            "side": side,
            "body_color": body_color,
            "fracture_color": frame_sides[side],
        }
        success = False
        for r0, c0 in _candidate_positions(shape, side, h, w):
            candidate = dict(spec)
            candidate["r0"] = r0
            candidate["c0"] = c0
            if _can_place(candidate, placed, h, w):
                placed.append(candidate)
                success = True
                break
        if not success:
            return None
    return placed


def _render_input(h: int, w: int, frame_sides: dict[str, int], specs: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BACKGROUND_COLOR, dtype=int)
    grid[0, 1:-1] = frame_sides["top"]
    grid[h - 1, 1:-1] = frame_sides["bottom"]
    grid[1:-1, 0] = frame_sides["left"]
    grid[1:-1, w - 1] = frame_sides["right"]

    for spec in specs:
        r0, c0 = spec["r0"], spec["c0"]
        body = spec["body_color"]
        crack = spec["fracture_color"]
        if spec["orientation"] == "horizontal":
            grid[r0 : r0 + spec["h"], c0 : c0 + spec["w"]] = body
            grid[r0 + spec["top_h"], c0 : c0 + spec["w"]] = crack
        else:
            grid[r0 : r0 + spec["h"], c0 : c0 + spec["w"]] = body
            grid[r0 : r0 + spec["h"], c0 + spec["left_w"]] = crack
    return grid


def _render_output_from_specs(h: int, w: int, frame_sides: dict[str, int], specs: list[dict]) -> np.ndarray:
    out = np.full((h, w), BACKGROUND_COLOR, dtype=int)
    out[0, 1:-1] = frame_sides["top"]
    out[h - 1, 1:-1] = frame_sides["bottom"]
    out[1:-1, 0] = frame_sides["left"]
    out[1:-1, w - 1] = frame_sides["right"]

    for spec in specs:
        r0, c0 = spec["r0"], spec["c0"]
        body = spec["body_color"]
        if spec["orientation"] == "horizontal":
            top_h = spec["top_h"]
            bottom_h = spec["bottom_h"]
            width = spec["w"]
            if spec["side"] == "top":
                out[r0 + top_h + 1 : r0 + spec["h"], c0 : c0 + width] = body
                out[1 : top_h + 1, c0 : c0 + width] = body
            else:
                out[r0 : r0 + top_h, c0 : c0 + width] = body
                out[h - 1 - bottom_h : h - 1, c0 : c0 + width] = body
        else:
            left_w = spec["left_w"]
            right_w = spec["right_w"]
            height = spec["h"]
            if spec["side"] == "left":
                out[r0 : r0 + height, c0 + left_w + 1 : c0 + spec["w"]] = body
                out[r0 : r0 + height, 1 : left_w + 1] = body
            else:
                out[r0 : r0 + height, c0 : c0 + left_w] = body
                out[r0 : r0 + height, w - 1 - right_w : w - 1] = body
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*INPUT_SIZE_RANGE)
    w = random.randint(*INPUT_SIZE_RANGE)

    object_count = random.choices(OBJECT_COUNT_CHOICES, weights=OBJECT_COUNT_WEIGHTS, k=1)[0]
    sides = _sample_sides(object_count)

    frame_colors = random.sample(ALL_COLORS, 4)
    frame_sides = dict(zip(SIDE_ORDER, frame_colors))

    body_pool = [color for color in ALL_COLORS if color not in frame_colors]
    if len(body_pool) < object_count:
        return None
    body_colors = random.sample(body_pool, object_count)

    specs = _place_objects(h, w, frame_sides, body_colors, sides)
    if specs is None:
        return None

    inp = _render_input(h, w, frame_sides, specs)
    expected = _render_output_from_specs(h, w, frame_sides, specs)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fracture-sign input")
