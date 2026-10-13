from __future__ import annotations

import random
from collections import Counter

import numpy as np


Shape = tuple[tuple[int, int], ...]
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_separator_run(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = grid.shape
    solid = [
        c
        for c in range(w)
        if np.all(grid[:, c] == grid[0, c]) and int(grid[0, c]) != int(bg)
    ]
    runs: list[tuple[int, int, int]] = []
    i = 0
    while i < len(solid):
        start = solid[i]
        color = int(grid[0, start])
        end = start
        i += 1
        while i < len(solid) and solid[i] == end + 1 and int(grid[0, solid[i]]) == color:
            end = solid[i]
            i += 1
        runs.append((int(start), int(end), color))

    candidates = []
    for start, end, color in runs:
        left_w = int(start)
        right_w = int(w - end - 1)
        if left_w > 0 and left_w == right_w:
            candidates.append((int(start), int(end), int(color)))

    assert len(candidates) == 1, "expected one centered full-height separator bar"
    return candidates[0]


def _normalize_shape(cells: set[tuple[int, int]]) -> Shape:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in cells))


def _shape_dims(shape: Shape) -> tuple[int, int]:
    return max(r for r, _c in shape) + 1, max(c for _r, c in shape) + 1


def _find_components(panel: np.ndarray, bg: int) -> list[dict]:
    h, w = panel.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(panel[r, c]) == int(bg):
                continue

            color = int(panel[r, c])
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(panel[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            shape = _normalize_shape(set(coords))
            components.append(
                {
                    "coords": arr,
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                    "shape": shape,
                }
            )

    return components


def _panel_info(panel: np.ndarray, bg: int) -> dict:
    objects = _find_components(panel, bg)
    assert 3 <= len(objects) <= 4, "each panel must contain 3 or 4 objects"

    colors = {int(obj["color"]) for obj in objects}
    assert len(colors) == 1, "each panel must use a single foreground color"
    color = int(next(iter(colors)))

    counts = Counter(obj["shape"] for obj in objects)
    unique_shapes = [shape for shape, count in counts.items() if count == 1]
    common_shapes = [shape for shape, count in counts.items() if count == len(objects) - 1]
    assert len(counts) == 2 and len(unique_shapes) == 1 and len(common_shapes) == 1

    anomaly_shape = unique_shapes[0]
    anomaly = next(obj for obj in objects if obj["shape"] == anomaly_shape)
    return {"color": color, "objects": objects, "anomaly": anomaly}


def _erase_component(out: np.ndarray, obj: dict, col_offset: int, bg: int) -> None:
    rows = obj["coords"][:, 0]
    cols = obj["coords"][:, 1] + int(col_offset)
    out[rows, cols] = int(bg)


def _paint_shape(
    out: np.ndarray,
    shape: Shape,
    top: int,
    left: int,
    color: int,
    bg: int,
    col_lo: int,
    col_hi: int,
) -> None:
    h, w = out.shape
    for dr, dc in shape:
        rr = int(top + dr)
        cc = int(left + dc)
        assert 0 <= rr < h and col_lo <= cc < col_hi, "swapped shape must stay inside its target panel"
        assert int(out[rr, cc]) == int(bg), "swapped shape must not overlap preserved cells"
        out[rr, cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the unique odd-shape object across two separator-split panels.

    Input structure:
    - Black background.
    - One full-height grey vertical separator bar splitting the grid into two equal-width panels.
    - Each panel contains 3-4 monochrome 4-connected objects.
    - Within each panel, exactly one object's normalized shape differs from all others.

    Rule:
    - Identify the unique normalized shape in each panel.
    - Erase those two anomalous objects.
    - Place the left anomaly's shape at the right anomaly's original bbox anchor, recolored to the right panel color.
    - Place the right anomaly's shape at the left anomaly's original bbox anchor, recolored to the left panel color.
    - Keep the separator and all non-anomalous objects unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    sep_start, sep_end, sep_color = _find_separator_run(g, bg)
    assert np.all(g[:, sep_start : sep_end + 1] == sep_color), "separator bar must be solid"

    panel_w = int(sep_start)
    right_offset = int(sep_end + 1)
    left_panel = g[:, :panel_w]
    right_panel = g[:, right_offset:]
    assert left_panel.shape == right_panel.shape, "panels must have equal size"

    left = _panel_info(left_panel, bg)
    right = _panel_info(right_panel, bg)
    left_anomaly = left["anomaly"]
    right_anomaly = right["anomaly"]

    out = g.copy()
    _erase_component(out, left_anomaly, 0, bg)
    _erase_component(out, right_anomaly, right_offset, bg)

    left_r0, left_c0, _left_r1, _left_c1 = left_anomaly["bbox"]
    right_r0, right_c0, _right_r1, _right_c1 = right_anomaly["bbox"]
    _paint_shape(
        out,
        right_anomaly["shape"],
        left_r0,
        left_c0,
        left["color"],
        bg,
        0,
        panel_w,
    )
    _paint_shape(
        out,
        left_anomaly["shape"],
        right_r0,
        right_offset + right_c0,
        right["color"],
        bg,
        right_offset,
        g.shape[1],
    )
    return out


# --------------- generator ---------------

GEN_BG = 0
GEN_LEFT = 1
GEN_RIGHT = 2
GEN_SEPARATOR = 5

GEN_PANEL_HEIGHTS = (11, 12, 13, 14)
GEN_PANEL_WIDTHS = (11, 12, 13, 14)
GEN_SEPARATOR_WIDTHS = (1, 2)
GEN_OBJECT_COUNTS = (3, 4)

GEN_SHAPE_MIN_AREA = 2
GEN_SHAPE_MAX_AREA = 5
GEN_SHAPE_MAX_HEIGHT = 3
GEN_SHAPE_MAX_WIDTH = 3

GEN_OUTER_MARGIN = 1
GEN_OBJECT_PADDING = 1
GEN_RETRIES = 400
GEN_SHAPE_RETRIES = 200


def _random_connected_shape() -> Shape | None:
    for _ in range(GEN_SHAPE_RETRIES):
        area = random.randint(GEN_SHAPE_MIN_AREA, GEN_SHAPE_MAX_AREA)
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    cell = (int(r + dr), int(c + dc))
                    if cell not in cells:
                        frontier.add(cell)
            cells.add(random.choice(tuple(frontier)))

        shape = _normalize_shape(cells)
        h, w = _shape_dims(shape)
        if h > GEN_SHAPE_MAX_HEIGHT or w > GEN_SHAPE_MAX_WIDTH:
            continue
        return shape
    return None


def _sample_distinct_shapes(n: int) -> list[Shape] | None:
    seen: set[Shape] = set()
    shapes: list[Shape] = []
    attempts = 0
    while len(shapes) < n and attempts < GEN_SHAPE_RETRIES * 4:
        attempts += 1
        shape = _random_connected_shape()
        if shape is None or shape in seen:
            continue
        seen.add(shape)
        shapes.append(shape)
    return shapes if len(shapes) == n else None


def _place_rect(blocked: np.ndarray, height: int, width: int) -> tuple[int, int] | None:
    h, w = blocked.shape
    r_min = GEN_OUTER_MARGIN
    c_min = GEN_OUTER_MARGIN
    r_max = h - height - GEN_OUTER_MARGIN
    c_max = w - width - GEN_OUTER_MARGIN
    if r_max < r_min or c_max < c_min:
        return None

    positions = []
    for r0 in range(r_min, r_max + 1):
        for c0 in range(c_min, c_max + 1):
            if not bool(blocked[r0 : r0 + height, c0 : c0 + width].any()):
                positions.append((int(r0), int(c0)))
    return random.choice(positions) if positions else None


def _mark_blocked(blocked: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    h, w = blocked.shape
    r0 = max(0, int(top - GEN_OBJECT_PADDING))
    c0 = max(0, int(left - GEN_OBJECT_PADDING))
    r1 = min(h, int(top + height + GEN_OBJECT_PADDING))
    c1 = min(w, int(left + width + GEN_OBJECT_PADDING))
    blocked[r0:r1, c0:c1] = True


def _paint_shape_unchecked(panel: np.ndarray, shape: Shape, top: int, left: int, color: int) -> None:
    for dr, dc in shape:
        panel[int(top + dr), int(left + dc)] = int(color)


def _erase_shape_unchecked(panel: np.ndarray, shape: Shape, top: int, left: int, bg: int) -> None:
    for dr, dc in shape:
        panel[int(top + dr), int(left + dc)] = int(bg)


def _build_panel(
    panel_h: int,
    panel_w: int,
    color: int,
    count: int,
    common_shape: Shape,
    anomaly_shape: Shape,
    incoming_shape: Shape,
) -> dict | None:
    panel = np.full((panel_h, panel_w), GEN_BG, dtype=int)
    blocked = np.zeros((panel_h, panel_w), dtype=bool)

    common_h, common_w = _shape_dims(common_shape)
    anomaly_h, anomaly_w = _shape_dims(anomaly_shape)
    incoming_h, incoming_w = _shape_dims(incoming_shape)
    slot_h = max(anomaly_h, incoming_h)
    slot_w = max(anomaly_w, incoming_w)

    items = [("anomaly", slot_h, slot_w)] + [("common", common_h, common_w)] * (count - 1)
    random.shuffle(items)
    items.sort(key=lambda item: item[1] * item[2], reverse=True)

    anomaly_anchor: tuple[int, int] | None = None
    common_anchors: list[tuple[int, int]] = []
    for kind, height, width in items:
        pos = _place_rect(blocked, height, width)
        if pos is None:
            return None
        _mark_blocked(blocked, pos[0], pos[1], height, width)
        if kind == "anomaly":
            anomaly_anchor = pos
        else:
            common_anchors.append(pos)

    assert anomaly_anchor is not None
    _paint_shape_unchecked(panel, anomaly_shape, anomaly_anchor[0], anomaly_anchor[1], color)
    for top, left in common_anchors:
        _paint_shape_unchecked(panel, common_shape, top, left, color)

    return {
        "panel": panel,
        "anomaly_shape": anomaly_shape,
        "anomaly_anchor": anomaly_anchor,
    }


def _compose_input(left_panel: np.ndarray, right_panel: np.ndarray, sep_w: int) -> np.ndarray:
    h, panel_w = left_panel.shape
    grid = np.full((h, panel_w * 2 + sep_w), GEN_BG, dtype=int)
    grid[:, :panel_w] = left_panel
    grid[:, panel_w : panel_w + sep_w] = GEN_SEPARATOR
    grid[:, panel_w + sep_w :] = right_panel
    return grid


def _expected_output(grid: np.ndarray, left: dict, right: dict, panel_w: int, sep_w: int) -> np.ndarray:
    out = grid.copy()
    left_top, left_left = left["anomaly_anchor"]
    right_top, right_left = right["anomaly_anchor"]
    right_offset = panel_w + sep_w

    _erase_shape_unchecked(out[:, :panel_w], left["anomaly_shape"], left_top, left_left, GEN_BG)
    _erase_shape_unchecked(out[:, right_offset:], right["anomaly_shape"], right_top, right_left, GEN_BG)

    _paint_shape_unchecked(out[:, :panel_w], right["anomaly_shape"], left_top, left_left, GEN_LEFT)
    _paint_shape_unchecked(out[:, right_offset:], left["anomaly_shape"], right_top, right_left, GEN_RIGHT)
    return out


def _generate_candidate() -> np.ndarray | None:
    panel_h = random.choice(GEN_PANEL_HEIGHTS)
    panel_w = random.choice(GEN_PANEL_WIDTHS)
    sep_w = random.choice(GEN_SEPARATOR_WIDTHS)
    left_count = random.choice(GEN_OBJECT_COUNTS)
    right_count = random.choice(GEN_OBJECT_COUNTS)

    shapes = _sample_distinct_shapes(4)
    if shapes is None:
        return None
    left_common, left_anomaly, right_common, right_anomaly = shapes

    left = _build_panel(panel_h, panel_w, GEN_LEFT, left_count, left_common, left_anomaly, right_anomaly)
    if left is None:
        return None
    right = _build_panel(panel_h, panel_w, GEN_RIGHT, right_count, right_common, right_anomaly, left_anomaly)
    if right is None:
        return None

    grid = _compose_input(left["panel"], right["panel"], sep_w)
    expected = _expected_output(grid, left, right, panel_w, sep_w)
    if np.array_equal(grid, expected):
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
