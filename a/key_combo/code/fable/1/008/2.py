from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
WHITE = 10
PATH_COLORS = tuple(range(1, 10))

GAP_CASES = (
    "top_left",
    "left_top",
    "top_right",
    "right_top",
    "bottom_right",
    "right_bottom",
    "bottom_left",
    "left_bottom",
)

GAP_CASE_CYCLE = (
    "top_left",
    "left_top",
    "bottom_right",
    "right_bottom",
    "top_right",
    "right_top",
    "bottom_left",
    "left_bottom",
)

INTERIOR_SIZE_OPTIONS = tuple(
    (h, w)
    for h in range(2, 7)
    for w in range(2, 7)
    if 4 <= h * w <= 18
)

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
MAX_GENERATION_ATTEMPTS = 400
_GEN_CALL_COUNT = 0


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _ordered_path(coords: np.ndarray, start: Coord) -> list[Coord]:
    cells = {tuple(map(int, coord)) for coord in coords}
    degrees = {
        cell: sum((neighbor in cells) for neighbor in _neighbors4(*cell))
        for cell in cells
    }
    endpoints = sorted(cell for cell, degree in degrees.items() if degree == 1)
    assert len(endpoints) == 2, "colored snake must have exactly two endpoints"
    assert start in endpoints, "white cap must be at one snake endpoint"
    assert all(degree in (1, 2) for degree in degrees.values()), "snake must not branch"

    path: list[Coord] = []
    prev = None
    cur = start
    while True:
        path.append(cur)
        next_cells = [n for n in _neighbors4(*cur) if n in cells and n != prev]
        if not next_cells:
            break
        assert len(next_cells) == 1, "snake traversal must be unique"
        prev, cur = cur, next_cells[0]

    assert len(path) == len(cells), "snake traversal must cover every snake pixel"
    return path


def _gap_coord_from_case(bbox: BBox, case: str) -> Coord:
    r0, c0, r1, c1 = bbox
    mapping = {
        "top_left": (r0, c0 + 1),
        "left_top": (r0 + 1, c0),
        "top_right": (r0, c1 - 1),
        "right_top": (r0 + 1, c1),
        "bottom_right": (r1, c1 - 1),
        "right_bottom": (r1 - 1, c1),
        "bottom_left": (r1, c0 + 1),
        "left_bottom": (r1 - 1, c0),
    }
    return mapping[case]


def _case_from_gap(bbox: BBox, gap: Coord) -> str:
    matches = [case for case in GAP_CASES if _gap_coord_from_case(bbox, case) == gap]
    assert len(matches) == 1, "gap position must identify exactly one corner-adjacent intake case"
    return matches[0]


def _fill_order_from_case(bbox: BBox, case: str, *, serpentine: bool = True) -> list[Coord]:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    assert inner_h >= 2 and inner_w >= 2

    order: list[Coord] = []
    if case == "top_left":
        start_r, start_c, along_sign, inward_sign, horizontal = r0 + 1, c0 + 1, 1, 1, True
    elif case == "top_right":
        start_r, start_c, along_sign, inward_sign, horizontal = r0 + 1, c1 - 1, -1, 1, True
    elif case == "bottom_right":
        start_r, start_c, along_sign, inward_sign, horizontal = r1 - 1, c1 - 1, -1, -1, True
    elif case == "bottom_left":
        start_r, start_c, along_sign, inward_sign, horizontal = r1 - 1, c0 + 1, 1, -1, True
    elif case == "left_top":
        start_r, start_c, along_sign, inward_sign, horizontal = r0 + 1, c0 + 1, 1, 1, False
    elif case == "right_top":
        start_r, start_c, along_sign, inward_sign, horizontal = r0 + 1, c1 - 1, 1, -1, False
    elif case == "right_bottom":
        start_r, start_c, along_sign, inward_sign, horizontal = r1 - 1, c1 - 1, -1, -1, False
    elif case == "left_bottom":
        start_r, start_c, along_sign, inward_sign, horizontal = r1 - 1, c0 + 1, -1, 1, False
    else:
        raise AssertionError("unknown gap case")

    if horizontal:
        for i in range(inner_h):
            rr = start_r + inward_sign * i
            line = [(rr, start_c + along_sign * k) for k in range(inner_w)]
            if serpentine and i % 2 == 1:
                line.reverse()
            order.extend(line)
    else:
        for i in range(inner_w):
            cc = start_c + inward_sign * i
            line = [(start_r + along_sign * k, cc) for k in range(inner_h)]
            if serpentine and i % 2 == 1:
                line.reverse()
            order.extend(line)

    assert len(order) == inner_h * inner_w
    return order


def _fill_pattern(bbox: BBox, case: str, seq: list[int], *, serpentine: bool = True) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    order = _fill_order_from_case(bbox, case, serpentine=serpentine)
    assert len(seq) == len(order)
    patch = np.full((inner_h, inner_w), -1, dtype=int)
    for (r, c), color in zip(order, seq):
        patch[r - (r0 + 1), c - (c0 + 1)] = int(color)
    assert np.all(patch >= 0)
    return patch


def _analyze_path(grid: np.ndarray, coords: np.ndarray) -> dict:
    white_cells = [tuple(map(int, cell)) for cell in coords if int(grid[tuple(cell)]) == WHITE]
    assert len(white_cells) == 1, "expected exactly one white cap"
    path = _ordered_path(coords, white_cells[0])
    payload = [int(grid[r, c]) for r, c in path[1:]]
    assert payload, "snake must have colored payload after the white cap"
    assert WHITE not in payload and BG not in payload
    assert len(set(payload)) >= 2, "snake payload must be visibly multicolor"
    return {"path": path, "payload": payload, "white": white_cells[0]}


def _analyze_frame(grid: np.ndarray, coords: np.ndarray) -> dict:
    colors = {int(grid[tuple(cell)]) for cell in coords}
    assert len(colors) == 1, "frame must be monochrome"
    color = next(iter(colors))
    assert color not in (BG, WHITE)

    bbox = _bbox_from_coords(coords)
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    inner_h = h - 2
    inner_w = w - 2
    assert inner_h >= 2 and inner_w >= 2, "interior must keep both row order and row direction meaningful"

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    assert np.all(sub[1:-1, 1:-1] == BG), "frame interior must be empty in the input"
    missing = np.argwhere(border & (sub == BG))
    assert missing.shape == (1, 2), "frame border must have exactly one missing cell"
    gap = (r0 + int(missing[0, 0]), c0 + int(missing[0, 1]))
    case = _case_from_gap(bbox, gap)

    border_values = sub[border]
    assert int(np.count_nonzero(border_values == BG)) == 1
    assert np.all(border_values[(border_values != BG)] == color)
    assert int(coords.shape[0]) == int(border.sum()) - 1
    return {
        "bbox": bbox,
        "color": color,
        "gap": gap,
        "case": case,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "area": inner_h * inner_w,
    }


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.count_nonzero(g == WHITE) == 1, "expected exactly one white cap in the whole grid"

    comps = _find_components(g != BG)
    assert len(comps) == 2, "expected exactly one snake component and one frame component"

    white_comp_idx = [i for i, comp in enumerate(comps) if any(int(g[tuple(cell)]) == WHITE for cell in comp)]
    assert len(white_comp_idx) == 1, "white cap must belong to exactly one component"
    path_comp = comps[white_comp_idx[0]]
    frame_comp = comps[1 - white_comp_idx[0]]

    path_info = _analyze_path(g, path_comp)
    frame_info = _analyze_frame(g, frame_comp)
    fill_order = _fill_order_from_case(frame_info["bbox"], frame_info["case"], serpentine=True)
    assert len(path_info["payload"]) == frame_info["area"], "snake payload length must equal frame interior area"

    return {
        "grid": g,
        "path": path_info,
        "frame": frame_info,
        "fill_order": fill_order,
    }


def _apply_fill(grid: np.ndarray, fill_order: list[Coord], payload: list[int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for (r, c), color in zip(fill_order, payload):
        out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the colored snake from its white-capped end and inject it through the frame notch to serpent-fill the hollow rectangle.

    Input structure:
    - Black is the background.
    - One 4-connected single-pixel-wide multicolor snake has a unique white endpoint that fixes the reading direction.
    - A separate monochrome hollow rectangular frame has exactly one missing border cell next to a corner, and its interior is black.

    Rule:
    - Starting at the cell after the white cap, read the snake colors along the unique endpoint-to-endpoint path.
    - The missing border cell identifies the interior start corner and the first writing direction along that same side.
    - Write the read colors into the frame interior with back-and-forth serpentine rows until the interior is full, while keeping the snake and the frame unchanged.

    Key insight:
    - The notch is an intake port, not damage: it simultaneously selects the start corner, the first row direction, and the row stacking direction that turns a 1D color tape into a 2D serpentine block.
    """
    info = _analyze_grid(np.asarray(grid, dtype=int))
    return _apply_fill(info["grid"], info["fill_order"], info["path"]["payload"])


def _path_turn_count(path: list[Coord]) -> int:
    turns = 0
    prev_dir = None
    for i in range(1, len(path)):
        dr = path[i][0] - path[i - 1][0]
        dc = path[i][1] - path[i - 1][1]
        cur_dir = (dr, dc)
        if prev_dir is not None and cur_dir != prev_dir:
            turns += 1
        prev_dir = cur_dir
    return turns


def _build_payload(length: int, forbidden: set[int]) -> list[int]:
    palette = [color for color in PATH_COLORS if color not in forbidden]
    min_palette = 3 if length >= 6 else 2
    max_palette = min(5, len(palette), length)
    while True:
        k = random.randint(min_palette, max_palette)
        chosen = random.sample(palette, k)
        seq = chosen[:]
        while len(seq) < length:
            seq.append(random.choice(chosen))
        random.shuffle(seq)
        if len(set(seq)) < 2:
            continue
        if seq == seq[::-1]:
            continue
        if seq[0] == seq[-1]:
            continue
        return seq


def _build_path(length: int) -> list[Coord] | None:
    side = 10
    starts = [(r, c) for r in range(side) for c in range(side)]
    random.shuffle(starts)
    min_turns = 2 if length >= 8 else 1

    def search(
        path: list[Coord],
        used: set[Coord],
        prev_dir: Coord | None,
        turns: int,
    ) -> list[Coord] | None:
        if len(path) == length:
            return path if turns >= min_turns else None

        remaining_after_move = length - len(path) - 1
        dirs = list(DIRS4)
        random.shuffle(dirs)
        if prev_dir is not None and turns < min_turns:
            dirs.sort(key=lambda d: d == prev_dir)

        r, c = path[-1]
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < side and 0 <= nc < side):
                continue
            if (nr, nc) in used:
                continue
            if any((ar, ac) in used and (ar, ac) != (r, c) for ar, ac in _neighbors4(nr, nc)):
                continue

            new_turns = turns + int(prev_dir is not None and (dr, dc) != prev_dir)
            if new_turns + remaining_after_move < min_turns:
                continue

            used.add((nr, nc))
            path.append((nr, nc))
            found = search(path, used, (dr, dc), new_turns)
            if found is not None:
                return found
            path.pop()
            used.remove((nr, nc))
        return None

    for start in starts:
        found = search([start], {start}, None, 0)
        if found is not None:
            arr = np.array(found, dtype=int)
            arr -= arr.min(axis=0)
            return [tuple(map(int, cell)) for cell in arr]
    return None


def _render_path_sprite(path: list[Coord], colors: list[int]) -> np.ndarray:
    arr = np.array(path, dtype=int)
    h = int(arr[:, 0].max()) + 1
    w = int(arr[:, 1].max()) + 1
    sprite = np.full((h, w), BG, dtype=int)
    for (r, c), color in zip(path, colors):
        sprite[r, c] = int(color)
    return sprite


def _render_frame_sprite(inner_h: int, inner_w: int, color: int, case: str) -> np.ndarray:
    h = inner_h + 2
    w = inner_w + 2
    sprite = np.full((h, w), BG, dtype=int)
    sprite[0, :] = color
    sprite[-1, :] = color
    sprite[:, 0] = color
    sprite[:, -1] = color
    gap = _gap_coord_from_case((0, 0, h - 1, w - 1), case)
    sprite[gap] = BG
    return sprite


def _paint_sprite(dst: np.ndarray, sprite: np.ndarray, top: int, left: int) -> None:
    h, w = sprite.shape
    mask = sprite != BG
    dst_view = dst[top : top + h, left : left + w]
    dst_view[mask] = sprite[mask]


def _compose_grid(frame_sprite: np.ndarray, path_sprite: np.ndarray) -> np.ndarray:
    fh, fw = frame_sprite.shape
    ph, pw = path_sprite.shape
    mode = random.choice(("frame_left", "frame_right", "frame_top", "frame_bottom"))

    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    gap = random.randint(2, 5)

    if mode in ("frame_left", "frame_right"):
        min_h = top_margin + bottom_margin + max(fh, ph)
        min_w = left_margin + right_margin + fw + pw + gap
        assert min_h <= 30 and min_w <= 30
        extra_h = random.randint(0, 30 - min_h)
        extra_w = random.randint(0, 30 - min_w)
        h = min_h + extra_h
        w = min_w + extra_w

        extra_left = random.randint(0, extra_w)
        extra_between = random.randint(0, extra_w - extra_left)
        extra_right = extra_w - extra_left - extra_between

        left = left_margin + extra_left
        right_gap = gap + extra_between
        _ = right_margin + extra_right

        frame_r = top_margin + random.randint(0, h - top_margin - bottom_margin - fh)
        path_r = top_margin + random.randint(0, h - top_margin - bottom_margin - ph)
        if mode == "frame_left":
            frame_c = left
            path_c = frame_c + fw + right_gap
        else:
            path_c = left
            frame_c = path_c + pw + right_gap
    else:
        min_h = top_margin + bottom_margin + fh + ph + gap
        min_w = left_margin + right_margin + max(fw, pw)
        assert min_h <= 30 and min_w <= 30
        extra_h = random.randint(0, 30 - min_h)
        extra_w = random.randint(0, 30 - min_w)
        h = min_h + extra_h
        w = min_w + extra_w

        extra_top = random.randint(0, extra_h)
        extra_between = random.randint(0, extra_h - extra_top)
        extra_bottom = extra_h - extra_top - extra_between

        top = top_margin + extra_top
        down_gap = gap + extra_between
        _ = bottom_margin + extra_bottom

        frame_c = left_margin + random.randint(0, w - left_margin - right_margin - fw)
        path_c = left_margin + random.randint(0, w - left_margin - right_margin - pw)
        if mode == "frame_top":
            frame_r = top
            path_r = frame_r + fh + down_gap
        else:
            path_r = top
            frame_r = path_r + ph + down_gap

    grid = np.full((h, w), BG, dtype=int)
    _paint_sprite(grid, frame_sprite, frame_r, frame_c)
    _paint_sprite(grid, path_sprite, path_r, path_c)
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    info = _analyze_grid(grid)
    out = main(grid)
    expected = _apply_fill(info["grid"], info["fill_order"], info["path"]["payload"])
    if not np.array_equal(out, expected):
        return False

    bbox = info["frame"]["bbox"]
    case = info["frame"]["case"]
    payload = info["path"]["payload"]
    target_patch = _fill_pattern(bbox, case, payload, serpentine=True)

    gap_matches = [
        other_case
        for other_case in GAP_CASES
        if np.array_equal(_fill_pattern(bbox, other_case, payload, serpentine=True), target_patch)
    ]
    if gap_matches != [case]:
        return False

    if np.array_equal(_fill_pattern(bbox, case, list(reversed(payload)), serpentine=True), target_patch):
        return False

    if np.array_equal(_fill_pattern(bbox, case, payload, serpentine=False), target_patch):
        return False

    with_white = [WHITE] + payload[:-1]
    if np.array_equal(_fill_pattern(bbox, case, with_white, serpentine=True), target_patch):
        return False

    if np.array_equal(out, grid):
        return False

    return True


def _attempt_generate(case: str) -> np.ndarray | None:
    inner_h, inner_w = random.choice(INTERIOR_SIZE_OPTIONS)
    area = inner_h * inner_w

    frame_color = random.choice(PATH_COLORS)
    payload = _build_payload(area, {frame_color})
    path = _build_path(area + 1)
    if path is None:
        return None
    if _path_turn_count(path) < (2 if len(path) >= 8 else 1):
        return None

    path_colors = [WHITE] + payload
    frame_sprite = _render_frame_sprite(inner_h, inner_w, frame_color, case)
    path_sprite = _render_path_sprite(path, path_colors)
    grid = _compose_grid(frame_sprite, path_sprite)

    try:
        return grid if _semantic_checks(grid) else None
    except AssertionError:
        return None


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    case = GAP_CASE_CYCLE[_GEN_CALL_COUNT % len(GAP_CASE_CYCLE)]
    _GEN_CALL_COUNT += 1

    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate(case)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid snake-to-notch rectangle input")
