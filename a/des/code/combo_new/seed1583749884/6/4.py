import random

import numpy as np


_DIR_BY_SIDE = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
_SIDE_BY_DIR = {value: key for key, value in _DIR_BY_SIDE.items()}


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(points: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        cells.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        cells.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        cells.append((r, c0))
    return cells


def _filled_rect_info(grid: np.ndarray, color: int) -> dict | None:
    points = np.argwhere(grid == color)
    if points.size == 0:
        return None
    bbox = _bbox(points)
    r0, c0, r1, c1 = bbox
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if np.all(patch == color):
        return {"color": int(color), "bbox": bbox}
    return None


def _gap_side(bbox: tuple[int, int, int, int], gap: tuple[int, int]) -> str:
    r0, c0, r1, c1 = bbox
    r, c = gap
    if r == r0 and c0 < c < c1:
        return "top"
    if r == r1 and c0 < c < c1:
        return "bottom"
    if c == c0 and r0 < r < r1:
        return "left"
    if c == c1 and r0 < r < r1:
        return "right"
    raise AssertionError("gap must lie on a non-corner side cell")


def _frame_info(grid: np.ndarray, color: int) -> dict | None:
    points = np.argwhere(grid == color)
    if points.size == 0:
        return None
    bbox = _bbox(points)
    r0, c0, r1, c1 = bbox
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    point_set = {tuple(map(int, pt)) for pt in points}
    if any(r not in (r0, r1) and c not in (c0, c1) for r, c in point_set):
        return None

    border = _border_cells(bbox)
    missing = [(r, c) for r, c in border if int(grid[r, c]) != color]
    if len(missing) != 1:
        return None
    gap = missing[0]
    side = _gap_side(bbox, gap)
    if len(point_set) != len(border) - 1:
        return None
    return {"color": int(color), "bbox": bbox, "gap": gap, "side": side}


def _inner_perimeter_cycle(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    top, left = r0 + 1, c0 + 1
    bottom, right = r1 - 1, c1 - 1
    if top > bottom or left > right:
        return []
    if top == bottom and left == right:
        return [(top, left)]
    if top == bottom:
        return [(top, c) for c in range(left, right + 1)]
    if left == right:
        return [(r, left) for r in range(top, bottom + 1)]

    cycle: list[tuple[int, int]] = []
    for c in range(left, right + 1):
        cycle.append((top, c))
    for r in range(top + 1, bottom + 1):
        cycle.append((r, right))
    for c in range(right - 1, left - 1, -1):
        cycle.append((bottom, c))
    for r in range(bottom - 1, top, -1):
        cycle.append((r, left))
    return cycle


def _inside_gap_cell(frame: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = frame["bbox"]
    r, c = frame["gap"]
    side = frame["side"]
    if side == "top":
        return r0 + 1, c
    if side == "bottom":
        return r1 - 1, c
    if side == "left":
        return r, c0 + 1
    return r, c1 - 1


def _entry_point(frame: dict, direction: tuple[int, int], current_gap: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = frame["bbox"]
    gr, gc = current_gap
    if direction == (-1, 0):
        return r0 + 1, gc
    if direction == (1, 0):
        return r1 - 1, gc
    if direction == (0, -1):
        return gr, c0 + 1
    if direction == (0, 1):
        return gr, c1 - 1
    raise AssertionError("direction must be cardinal")


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc >= ac else -1
        return [(ar, c) for c in range(ac, bc + step, step)]
    if ac == bc:
        step = 1 if br >= ar else -1
        return [(r, ac) for r in range(ar, br + step, step)]
    raise AssertionError("segment must be straight")


def _perimeter_arc(
    bbox: tuple[int, int, int, int], start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]]:
    cycle = _inner_perimeter_cycle(bbox)
    if len(cycle) <= 1:
        return [start]
    index = {cell: i for i, cell in enumerate(cycle)}
    i0 = index[start]
    i1 = index[end]
    n = len(cycle)
    clockwise = (i1 - i0) % n
    counter = (i0 - i1) % n
    if clockwise <= counter:
        return [cycle[(i0 + k) % n] for k in range(clockwise + 1)]
    return [cycle[(i0 - k) % n] for k in range(counter + 1)]


def _paint_cells(out: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Percolate the innermost filled color outward through the gaps of nested rectangular frames.

    Input structure:
    - The background is the dominant color.
    - There are 3-4 nested one-pixel-thick rectangular frames, each in a different color and each missing exactly one non-corner border cell.
    - The smallest frame surrounds a solid filled rectangle whose color is distinct from every frame color.

    Rule:
    - Identify the innermost filled rectangle and the ordered frame nest around it.
    - Starting at the innermost frame's gap, extend the filled color straight outward until reaching the next frame's inner wall.
    - If that outer frame has an aligned gap on the same outward ray, pass straight through; otherwise follow the frame's inner perimeter to its own gap, then continue outward from there.
    - Repeat layer by layer until the path exits beyond the outermost frame; recolor only background path cells and keep all frame pixels unchanged.

    Key insight:
    - Each gap is a portal in a nested corridor system: the transformation is determined by recovering the frame order and routing the fill through successive gap-alignment or wall-following steps.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _bg_color(g)
    object_colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert len(object_colors) >= 4, "expected at least three frames plus one fill color"

    fill_candidates = []
    frames = []
    for color in object_colors:
        fill_info = _filled_rect_info(g, color)
        if fill_info is not None:
            fill_candidates.append(fill_info)
            continue
        frame = _frame_info(g, color)
        if frame is not None:
            frames.append(frame)
            continue
        raise AssertionError("every non-background color must be a frame or the inner filled region")

    assert len(fill_candidates) == 1, "expected exactly one filled rectangle"
    assert 3 <= len(frames) <= 4, "expected 3-4 nested frames"

    fill = fill_candidates[0]
    frames.sort(key=lambda frame: (frame["bbox"][2] - frame["bbox"][0], frame["bbox"][3] - frame["bbox"][1]))

    fill_r0, fill_c0, fill_r1, fill_c1 = fill["bbox"]
    inner_bbox = frames[0]["bbox"]
    assert (fill_r0, fill_c0, fill_r1, fill_c1) == (
        inner_bbox[0] + 1,
        inner_bbox[1] + 1,
        inner_bbox[2] - 1,
        inner_bbox[3] - 1,
    ), "filled rectangle must occupy the innermost frame interior"

    for inner, outer in zip(frames, frames[1:]):
        ir0, ic0, ir1, ic1 = inner["bbox"]
        or0, oc0, or1, oc1 = outer["bbox"]
        assert or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1, "frames must be strictly nested"

    out = g.copy()
    fill_color = int(fill["color"])

    current_gap = frames[0]["gap"]
    current_direction = _DIR_BY_SIDE[frames[0]["side"]]
    out[current_gap] = fill_color

    for frame in frames[1:]:
        entry = _entry_point(frame, current_direction, current_gap)
        start = (current_gap[0] + current_direction[0], current_gap[1] + current_direction[1])
        _paint_cells(out, _line_cells(start, entry), fill_color)

        gap_inside = _inside_gap_cell(frame)
        if gap_inside != entry:
            _paint_cells(out, _perimeter_arc(frame["bbox"], entry, gap_inside), fill_color)

        out[frame["gap"]] = fill_color
        current_gap = frame["gap"]
        current_direction = _DIR_BY_SIDE[frame["side"]]

    r, c = current_gap[0] + current_direction[0], current_gap[1] + current_direction[1]
    h, w = out.shape
    while 0 <= r < h and 0 <= c < w:
        out[r, c] = fill_color
        r += current_direction[0]
        c += current_direction[1]

    return out


def _shift_bbox(bbox: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def _gap_candidates(bbox: tuple[int, int, int, int]) -> list[dict]:
    r0, c0, r1, c1 = bbox
    candidates = []
    for c in range(c0 + 1, c1):
        candidates.append({"gap": (r0, c), "side": "top"})
        candidates.append({"gap": (r1, c), "side": "bottom"})
    for r in range(r0 + 1, r1):
        candidates.append({"gap": (r, c0), "side": "left"})
        candidates.append({"gap": (r, c1), "side": "right"})
    return candidates


def _arc_is_unambiguous(bbox: tuple[int, int, int, int], start: tuple[int, int], end: tuple[int, int]) -> bool:
    cycle = _inner_perimeter_cycle(bbox)
    if len(cycle) <= 1:
        return True
    index = {cell: i for i, cell in enumerate(cycle)}
    i0 = index[start]
    i1 = index[end]
    n = len(cycle)
    return ((i1 - i0) % n) != ((i0 - i1) % n)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, gap: tuple[int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[gap] = 0


def _make_scene(rng: random.Random) -> np.ndarray | None:
    layer_count = rng.choice([3, 4])
    fill_h = rng.randint(2, 4)
    fill_w = rng.randint(2, 5)

    frames_local = [(0, 0, fill_h + 1, fill_w + 1)]
    for _ in range(layer_count - 1):
        r0, c0, r1, c1 = frames_local[-1]
        expand_top = rng.randint(2, 3)
        expand_bottom = rng.randint(2, 3)
        expand_left = rng.randint(2, 3)
        expand_right = rng.randint(2, 3)
        frames_local.append((r0 - expand_top, c0 - expand_left, r1 + expand_bottom, c1 + expand_right))

    outer = frames_local[-1]
    outer_h = outer[2] - outer[0] + 1
    outer_w = outer[3] - outer[1] + 1
    top_margin = rng.randint(1, 3)
    bottom_margin = rng.randint(1, 3)
    left_margin = rng.randint(1, 3)
    right_margin = rng.randint(1, 3)
    grid_h = top_margin + outer_h + bottom_margin
    grid_w = left_margin + outer_w + right_margin
    if grid_h > 30 or grid_w > 30:
        return None

    shift_r = top_margin - outer[0]
    shift_c = left_margin - outer[1]
    frames = [_shift_bbox(bbox, shift_r, shift_c) for bbox in frames_local]
    fill_bbox = (frames[0][0] + 1, frames[0][1] + 1, frames[0][2] - 1, frames[0][3] - 1)

    palette = list(range(1, 10))
    rng.shuffle(palette)
    fill_color = palette[0]
    frame_colors = palette[1 : 1 + layer_count]

    frame_infos = []
    first_bbox = frames[0]
    first_gap = rng.choice(_gap_candidates(first_bbox))
    frame_infos.append({"bbox": first_bbox, "gap": first_gap["gap"], "side": first_gap["side"]})

    current_gap = first_gap["gap"]
    current_direction = _DIR_BY_SIDE[first_gap["side"]]
    for bbox in frames[1:]:
        entry = _entry_point({"bbox": bbox}, current_direction, current_gap)
        aligned_gap = {
            "gap": (entry[0] - current_direction[0], entry[1] - current_direction[1]),
            "side": _SIDE_BY_DIR[current_direction],
        }

        branch = rng.choice(["straight", "turn"])
        if branch == "straight":
            chosen = aligned_gap
        else:
            turn_candidates = []
            for candidate in _gap_candidates(bbox):
                frame = {"bbox": bbox, "gap": candidate["gap"], "side": candidate["side"]}
                gap_inside = _inside_gap_cell(frame)
                if gap_inside == entry:
                    continue
                if not _arc_is_unambiguous(bbox, entry, gap_inside):
                    continue
                turn_candidates.append(candidate)
            if not turn_candidates:
                return None
            chosen = rng.choice(turn_candidates)

        frame_infos.append({"bbox": bbox, "gap": chosen["gap"], "side": chosen["side"]})
        current_gap = chosen["gap"]
        current_direction = _DIR_BY_SIDE[chosen["side"]]

    grid = np.zeros((grid_h, grid_w), dtype=int)
    fr0, fc0, fr1, fc1 = fill_bbox
    grid[fr0 : fr1 + 1, fc0 : fc1 + 1] = fill_color
    for color, frame in zip(frame_colors, frame_infos):
        _draw_frame(grid, frame["bbox"], color, frame["gap"])

    if _bg_color(grid) != 0:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    new_fill = (out == fill_color) & (grid != fill_color)
    if int(new_fill.sum()) < layer_count + 4:
        return None
    if not (new_fill[0, :].any() or new_fill[-1, :].any() or new_fill[:, 0].any() or new_fill[:, -1].any()):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = random.Random()
        grid = _make_scene(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame seepage puzzle")
