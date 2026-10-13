from __future__ import annotations

import random

import numpy as np


BG = 0
GRAY = 5
WALL_COLOR = 8
PALETTE = [1, 2, 3, 4, 6, 7, 9]

DIR_STEP = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
WALL_TO_DIR = {
    "top": "U",
    "bottom": "D",
    "left": "L",
    "right": "R",
}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _paint_rect(grid: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = rect
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _rect_cells(rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _paint_wall(grid: np.ndarray, side: str, color: int) -> None:
    if side == "top":
        grid[0, :] = color
    elif side == "bottom":
        grid[-1, :] = color
    elif side == "left":
        grid[:, 0] = color
    elif side == "right":
        grid[:, -1] = color
    else:
        raise AssertionError(f"unknown side: {side}")


def _detect_wall(grid: np.ndarray) -> tuple[str, int]:
    H, W = grid.shape
    edges = {
        "top": grid[0, :],
        "bottom": grid[H - 1, :],
        "left": grid[:, 0],
        "right": grid[:, W - 1],
    }
    candidates: list[tuple[str, int]] = []
    for side, edge in edges.items():
        uniq = np.unique(edge)
        if len(uniq) == 1 and int(uniq[0]) != BG:
            candidates.append((side, int(uniq[0])))
    assert len(candidates) == 1, "expected exactly one full-wall border"
    return candidates[0]


def _extract_legend(
    grid: np.ndarray, wall_side: str, wall_color: int
) -> tuple[int, int, int, list[int]]:
    H, W = grid.shape
    special = {BG, GRAY, wall_color}
    for r in range(min(H, 4)):
        c = 0
        while wall_side == "left" and c < W and int(grid[r, c]) == wall_color:
            c += 1
        c0 = c
        colors: list[int] = []
        while c < W and int(grid[r, c]) not in special:
            colors.append(int(grid[r, c]))
            c += 1
        if colors:
            assert len(set(colors)) == len(colors), "legend colors must be distinct"
            return r, c0, c - 1, colors
    raise AssertionError("failed to locate the legend row")


def _find_components(grid: np.ndarray, colors: set[int]) -> list[dict]:
    H, W = grid.shape
    visited = np.zeros((H, W), dtype=bool)
    comps: list[dict] = []
    for r in range(H):
        for c in range(W):
            color = int(grid[r, c])
            if color not in colors or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"color": color, "coords": coords, "bbox": _bbox(coords)})
    return comps


def _is_solid_rectangle(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    return bool(np.all(patch == int(comp["color"])))


def _detect_direction(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> str:
    H, W = grid.shape
    r0, c0, r1, c1 = bbox
    candidates: list[str] = []
    if c0 > 0 and bool(np.all(grid[r0 : r1 + 1, :c0] == GRAY)):
        candidates.append("R")
    if c1 + 1 < W and bool(np.all(grid[r0 : r1 + 1, c1 + 1 :] == GRAY)):
        candidates.append("L")
    if r0 > 0 and bool(np.all(grid[:r0, c0 : c1 + 1] == GRAY)):
        candidates.append("D")
    if r1 + 1 < H and bool(np.all(grid[r1 + 1 :, c0 : c1 + 1] == GRAY)):
        candidates.append("U")
    assert len(candidates) == 1, "each slider must have one recoverable trail direction"
    return candidates[0]


def _front_cells(bbox: tuple[int, int, int, int], direction: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if direction == "L":
        return [(r, c0 - 1) for r in range(r0, r1 + 1)]
    if direction == "R":
        return [(r, c1 + 1) for r in range(r0, r1 + 1)]
    if direction == "U":
        return [(r0 - 1, c) for c in range(c0, c1 + 1)]
    if direction == "D":
        return [(r1 + 1, c) for c in range(c0, c1 + 1)]
    raise AssertionError(f"unknown direction: {direction}")


def _trail_rect(
    rect: tuple[int, int, int, int], direction: str, shape: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    H, W = shape
    r0, c0, r1, c1 = rect
    if direction == "L":
        return None if c1 + 1 > W - 1 else (r0, c1 + 1, r1, W - 1)
    if direction == "R":
        return None if c0 - 1 < 0 else (r0, 0, r1, c0 - 1)
    if direction == "U":
        return None if r1 + 1 > H - 1 else (r1 + 1, c0, H - 1, c1)
    if direction == "D":
        return None if r0 - 1 < 0 else (0, c0, r0 - 1, c1)
    raise AssertionError(f"unknown direction: {direction}")


def _trail_length(rect: tuple[int, int, int, int], direction: str, shape: tuple[int, int]) -> int:
    trail = _trail_rect(rect, direction, shape)
    if trail is None:
        return -1
    r0, c0, r1, c1 = trail
    if direction in {"L", "R"}:
        return c1 - c0 + 1
    return r1 - r0 + 1


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the arrival chain of sliding rectangles from their gray trails, then recolor each rank by the legend.

    Input structure
    - Black background, one full teal wall on a grid edge, and a short legend row near the upper-left corner.
    - Several differently colored solid rectangles rest in the field; each rectangle has a gray straight trail from some border to its rear side.
    - Every rectangle's front side is flush with either the wall or the flank of an earlier rectangle.

    Rule
    - Use the full gray strip behind each rectangle to infer its motion direction.
    - The object directly touching its front side is the stopper: the wall marks rank 1, the rectangle resting on rank 1 is rank 2, and so on along the unique dependency chain.
    - Recolor rank i with the i-th legend color, erase the legend and all gray trails, and keep only the wall plus the recolored rectangles.

    Key insight
    - Resting contacts fossilize time: the final geometry alone does not reveal arrival order, but trail direction plus "who stopped whom" reconstruct a unique causal chain.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    wall_side, wall_color = _detect_wall(g)
    legend_r, legend_c0, legend_c1, legend_colors = _extract_legend(g, wall_side, wall_color)

    work = g.copy()
    work[legend_r, legend_c0 : legend_c1 + 1] = BG

    slider_colors = set(map(int, np.unique(work))) - {BG, GRAY, wall_color}
    sliders = _find_components(work, slider_colors)
    assert sliders, "expected at least one slider"
    assert len(sliders) == len(legend_colors), "legend length must match the number of sliders"

    slider_at: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(sliders):
        assert _is_solid_rectangle(work, comp), "sliders must be solid rectangles"
        for cell in comp["coords"]:
            slider_at[cell] = idx

    stoppers: list[int | None] = []
    directions: list[str] = []
    for comp in sliders:
        direction = _detect_direction(work, comp["bbox"])
        directions.append(direction)
        front = _front_cells(comp["bbox"], direction)
        vals = [int(work[r, c]) for r, c in front]
        if all(val == wall_color for val in vals):
            stoppers.append(None)
            continue
        ids = {slider_at.get((r, c)) for r, c in front}
        assert None not in ids, "front contact must be another slider or the wall"
        assert len(ids) == 1, "each slider must be stopped by exactly one object"
        stoppers.append(next(iter(ids)))

    roots = [idx for idx, stopper in enumerate(stoppers) if stopper is None]
    assert len(roots) == 1, "expected one wall-touching root slider"

    children: dict[int | None, list[int]] = {}
    for idx, stopper in enumerate(stoppers):
        children.setdefault(stopper, []).append(idx)
    assert len(children[None]) == 1
    for stopper, items in children.items():
        if stopper is not None:
            assert len(items) <= 1, "resting dependencies must form a single chain"

    order: list[int] = []
    cur = children[None][0]
    while True:
        order.append(cur)
        nxt = children.get(cur, [])
        if not nxt:
            break
        cur = nxt[0]
    assert len(order) == len(sliders), "dependency chain must cover all sliders"

    out = np.full_like(g, BG)
    _paint_wall(out, wall_side, wall_color)
    for color, idx in zip(legend_colors, order):
        _paint_rect(out, sliders[idx]["bbox"], color)
    return out


GEN_HEIGHT_RANGE = (18, 26)
GEN_WIDTH_RANGE = (18, 26)
GEN_WALL_SIDES = ("left", "right", "bottom")
GEN_SLIDER_COUNT_RANGE = (3, 5)
GEN_MIN_SIDE = 2
GEN_MAX_SHORT = 4
GEN_MAX_LONG = 5
GEN_MIN_TRAIL = 2
GEN_ATTEMPTS = 400
GEN_LOCAL_TRIES = 160


def _legend_slots(H: int, W: int, wall_side: str, n: int) -> tuple[int, int, int]:
    _ = H
    row = 0
    c0 = 1 if wall_side == "left" else 0
    c1 = c0 + n - 1
    assert c1 < W
    return row, c0, c1


def _sample_legend_colors(slider_colors: list[int]) -> list[int]:
    for _ in range(200):
        colors = random.sample(PALETTE, len(slider_colors))
        if all(a != b for a, b in zip(colors, slider_colors)):
            return colors
    raise RuntimeError("failed to sample a nontrivial legend recoloring")


def _candidate_rect_against_wall(
    H: int, W: int, wall_side: str
) -> tuple[tuple[int, int, int, int], str] | None:
    direction = WALL_TO_DIR[wall_side]
    if direction in {"L", "R"}:
        h = random.randint(GEN_MIN_SIDE, GEN_MAX_SHORT)
        w = random.randint(GEN_MIN_SIDE, min(GEN_MAX_LONG, W - 2))
        row_lo = GEN_MIN_TRAIL
        row_hi = H - h - 2
        if row_lo > row_hi:
            return None
        r0 = random.randint(row_lo, row_hi)
        if direction == "L":
            rect = (r0, 1, r0 + h - 1, w)
        else:
            rect = (r0, W - 1 - w, r0 + h - 1, W - 2)
        return rect, direction

    w = random.randint(GEN_MIN_SIDE, GEN_MAX_SHORT)
    h = random.randint(GEN_MIN_SIDE, min(GEN_MAX_LONG, H - 2))
    col_lo = 1
    col_hi = W - w - 2
    if col_lo > col_hi:
        return None
    c0 = random.randint(col_lo, col_hi)
    rect = (H - 1 - h, c0, H - 2, c0 + w - 1)
    return rect, direction


def _attachment_sides(prev_dir: str) -> list[str]:
    if prev_dir in {"L", "R"}:
        return ["top", "bottom"]
    return ["left", "right"]


def _candidate_rect_against_slider(
    prev_bbox: tuple[int, int, int, int], attach_side: str, shape: tuple[int, int]
) -> tuple[tuple[int, int, int, int], str] | None:
    H, W = shape
    pr0, pc0, pr1, pc1 = prev_bbox
    prev_h = pr1 - pr0 + 1
    prev_w = pc1 - pc0 + 1

    if attach_side in {"left", "right"}:
        max_h = min(prev_h, GEN_MAX_SHORT)
        if max_h < GEN_MIN_SIDE:
            return None
        h = random.randint(GEN_MIN_SIDE, max_h)
        width_cap = GEN_MAX_LONG
        if attach_side == "left":
            width_cap = min(width_cap, pc0 - GEN_MIN_TRAIL)
            direction = "R"
        else:
            width_cap = min(width_cap, W - 1 - pc1 - GEN_MIN_TRAIL)
            direction = "L"
        if width_cap < GEN_MIN_SIDE:
            return None
        w = random.randint(GEN_MIN_SIDE, width_cap)
        r0 = random.randint(pr0, pr1 - h + 1)
        if attach_side == "left":
            rect = (r0, pc0 - w, r0 + h - 1, pc0 - 1)
        else:
            rect = (r0, pc1 + 1, r0 + h - 1, pc1 + w)
        return rect, direction

    max_w = min(prev_w, GEN_MAX_SHORT)
    if max_w < GEN_MIN_SIDE:
        return None
    w = random.randint(GEN_MIN_SIDE, max_w)
    height_cap = GEN_MAX_LONG
    if attach_side == "top":
        height_cap = min(height_cap, pr0 - GEN_MIN_TRAIL)
        direction = "D"
    else:
        height_cap = min(height_cap, H - 1 - pr1 - GEN_MIN_TRAIL)
        direction = "U"
    if height_cap < GEN_MIN_SIDE:
        return None
    h = random.randint(GEN_MIN_SIDE, height_cap)
    c0 = random.randint(pc0, pc1 - w + 1)
    if attach_side == "top":
        rect = (pr0 - h, c0, pr0 - 1, c0 + w - 1)
    else:
        rect = (pr1 + 1, c0, pr1 + h - 1, c0 + w - 1)
    return rect, direction


def _expected_target_cells(
    rect: tuple[int, int, int, int], direction: str
) -> list[tuple[int, int]]:
    return _front_cells(rect, direction)


def _candidate_ok(
    grid: np.ndarray,
    reserved: np.ndarray,
    rect: tuple[int, int, int, int],
    direction: str,
    allowed_target_cells: set[tuple[int, int]],
) -> bool:
    H, W = grid.shape
    r0, c0, r1, c1 = rect
    if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
        return False
    if _trail_length(rect, direction, grid.shape) < GEN_MIN_TRAIL:
        return False
    trail = _trail_rect(rect, direction, grid.shape)
    if trail is None:
        return False

    candidate = set(_rect_cells(rect)) | set(_rect_cells(trail))
    occupied = (grid != BG) | reserved
    for r, c in candidate:
        if occupied[r, c]:
            return False

    allowed_neighbors = set()
    if direction == "L":
        for r in range(r0, r1 + 1):
            allowed_neighbors.add((r, c0, r, c0 - 1))
    elif direction == "R":
        for r in range(r0, r1 + 1):
            allowed_neighbors.add((r, c1, r, c1 + 1))
    elif direction == "U":
        for c in range(c0, c1 + 1):
            allowed_neighbors.add((r0, c, r0 - 1, c))
    else:
        for c in range(c0, c1 + 1):
            allowed_neighbors.add((r1, c, r1 + 1, c))

    for tr, tc in allowed_target_cells:
        if not (0 <= tr < H and 0 <= tc < W) or grid[tr, tc] == BG or reserved[tr, tc]:
            return False

    for r, c in candidate:
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if (nr, nc) in candidate:
                continue
            if occupied[nr, nc] and (r, c, nr, nc) not in allowed_neighbors:
                return False

    return True


def _place_slider(
    grid: np.ndarray,
    rect: tuple[int, int, int, int],
    direction: str,
    color: int,
) -> None:
    trail = _trail_rect(rect, direction, grid.shape)
    assert trail is not None
    _paint_rect(grid, trail, GRAY)
    _paint_rect(grid, rect, color)


def _build_expected_output(
    shape: tuple[int, int], wall_side: str, placed: list[dict], legend_colors: list[int]
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    _paint_wall(out, wall_side, WALL_COLOR)
    for color, item in zip(legend_colors, placed):
        _paint_rect(out, item["bbox"], color)
    return out


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GEN_HEIGHT_RANGE)
    W = random.randint(*GEN_WIDTH_RANGE)
    n = random.randint(*GEN_SLIDER_COUNT_RANGE)
    wall_side = random.choice(GEN_WALL_SIDES)

    legend_r, legend_c0, legend_c1 = _legend_slots(H, W, wall_side, n)
    if legend_c1 >= W:
        return None

    slider_colors = random.sample(PALETTE, n)
    legend_colors = _sample_legend_colors(slider_colors)

    grid = np.full((H, W), BG, dtype=int)
    _paint_wall(grid, wall_side, WALL_COLOR)
    reserved = np.zeros((H, W), dtype=bool)
    reserved[legend_r, legend_c0 : legend_c1 + 1] = True

    placed: list[dict] = []
    first = None
    for _ in range(GEN_LOCAL_TRIES):
        candidate = _candidate_rect_against_wall(H, W, wall_side)
        if candidate is None:
            continue
        rect, direction = candidate
        if wall_side == "left":
            target_cells = {(r, 0) for r in range(rect[0], rect[2] + 1)}
        elif wall_side == "right":
            target_cells = {(r, W - 1) for r in range(rect[0], rect[2] + 1)}
        else:
            target_cells = {(H - 1, c) for c in range(rect[1], rect[3] + 1)}
        if _candidate_ok(grid, reserved, rect, direction, target_cells):
            first = {"bbox": rect, "dir": direction, "color": slider_colors[0]}
            break
    if first is None:
        return None

    _place_slider(grid, first["bbox"], first["dir"], first["color"])
    placed.append(first)

    for color in slider_colors[1:]:
        prev = placed[-1]
        sides = _attachment_sides(prev["dir"])
        random.shuffle(sides)
        item = None
        for _ in range(GEN_LOCAL_TRIES):
            side = random.choice(sides)
            candidate = _candidate_rect_against_slider(prev["bbox"], side, grid.shape)
            if candidate is None:
                continue
            rect, direction = candidate
            target_cells = set(_expected_target_cells(rect, direction))
            if _candidate_ok(grid, reserved, rect, direction, target_cells):
                item = {"bbox": rect, "dir": direction, "color": color}
                break
        if item is None:
            return None
        _place_slider(grid, item["bbox"], item["dir"], item["color"])
        placed.append(item)

    grid[legend_r, legend_c0 : legend_c1 + 1] = np.array(legend_colors, dtype=int)
    expected = _build_expected_output(grid.shape, wall_side, placed, legend_colors)

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for fable/3/054/5")
