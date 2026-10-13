from __future__ import annotations

import random

import numpy as np


BLACK = 0
COLORS = tuple(range(1, 10))


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _trace_polyline(mask: np.ndarray) -> list[tuple[int, int]]:
    coords = [tuple(map(int, p)) for p in np.argwhere(mask)]
    if not coords:
        return []
    if len(coords) == 1:
        return coords

    h, w = mask.shape
    coord_set = set(coords)
    neighbors = {
        pos: [nxt for nxt in _neighbors4(pos[0], pos[1], h, w) if nxt in coord_set]
        for pos in coords
    }
    endpoints = [pos for pos, nxts in neighbors.items() if len(nxts) == 1]
    assert len(endpoints) == 2, "each colored object must be a single open polyline"

    path = [endpoints[0]]
    prev: tuple[int, int] | None = None
    cur = endpoints[0]
    while True:
        nxts = [nxt for nxt in neighbors[cur] if nxt != prev]
        assert len(nxts) <= 1, "polyline must not branch"
        if not nxts:
            break
        nxt = nxts[0]
        path.append(nxt)
        prev, cur = cur, nxt

    assert len(path) == len(coords), "polyline must be connected without cycles"
    return path


def _corner_positions(mask: np.ndarray) -> list[tuple[int, int]]:
    path = _trace_polyline(mask)
    corners: list[tuple[int, int]] = []
    for i in range(1, len(path) - 1):
        r0, c0 = path[i - 1]
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        d1 = (r1 - r0, c1 - c0)
        d2 = (r2 - r1, c2 - c1)
        if d1 != d2:
            corners.append((r1, c1))
    return corners


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the right-edge legend and compress each colored polyline to one dot per corner.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert w >= 2

    out = np.zeros_like(g)
    out[:, w - 1] = g[:, w - 1]
    legend = [(int(r), int(g[r, w - 1])) for r in range(h) if int(g[r, w - 1]) != BLACK]
    main_area = g[:, : w - 1]

    for row, color in legend:
        mask = main_area == int(color)
        turns = _corner_positions(mask)
        if turns:
            out[row, : len(turns)] = int(color)

    return out


GEN_ATTEMPTS = 400
GRID_HEIGHT_RANGE = (12, 28)
GRID_WIDTH_RANGE = (14, 32)
LINE_COUNT_RANGE = (2, 5)
TURN_COUNT_RANGE = (0, 6)
MAJOR_STEP_RANGE = (1, 5)
MINOR_STEP_RANGE = (1, 4)
MINOR_SWING_LIMIT = 5
BOX_PADDING = 1


def _build_patch_from_coords(coords: list[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        patch[r - r0, c - c0] = True
    return patch


def _sample_minor_moves(count: int) -> list[int]:
    pos = 0
    moves: list[int] = []
    for _ in range(count):
        step = random.randint(*MINOR_STEP_RANGE)
        if pos <= -MINOR_SWING_LIMIT:
            move = step
        elif pos >= MINOR_SWING_LIMIT:
            move = -step
        else:
            move = random.choice((-step, step))
        moves.append(move)
        pos += move
    return moves


def _canonical_polyline_patch(turns: int, start_horizontal: bool) -> np.ndarray:
    segments = turns + 1
    if start_horizontal:
        horizontal_count = (segments + 1) // 2
        vertical_count = segments // 2
    else:
        horizontal_count = segments // 2
        vertical_count = (segments + 1) // 2

    major_low = MAJOR_STEP_RANGE[0] if turns == 0 else max(2, MAJOR_STEP_RANGE[0])
    horizontal_steps = [random.randint(major_low, MAJOR_STEP_RANGE[1]) for _ in range(horizontal_count)]
    vertical_moves = _sample_minor_moves(vertical_count)

    r = 0
    c = 0
    coords = [(r, c)]
    h_idx = 0
    v_idx = 0
    for seg_idx in range(segments):
        is_horizontal = (seg_idx % 2 == 0) == start_horizontal
        if is_horizontal:
            steps = horizontal_steps[h_idx]
            h_idx += 1
            for _ in range(steps):
                c += 1
                coords.append((r, c))
        else:
            move = vertical_moves[v_idx]
            v_idx += 1
            dr = 1 if move > 0 else -1
            for _ in range(abs(move)):
                r += dr
                coords.append((r, c))

    return _build_patch_from_coords(coords)


def _make_polyline_patch(turns: int) -> np.ndarray:
    for _ in range(500):
        patch = _canonical_polyline_patch(turns, start_horizontal=random.choice((True, False)))
        if random.random() < 0.5:
            patch = patch.T
        if random.random() < 0.5:
            patch = np.flipud(patch)
        if random.random() < 0.5:
            patch = np.fliplr(patch)
        try:
            corner_count = len(_corner_positions(patch))
        except AssertionError:
            continue
        if corner_count == turns:
            return patch
    raise RuntimeError("failed to construct a valid polyline patch")


def _pick_legend_rows(height: int, count: int) -> list[int]:
    span = 2 * count - 1
    start = random.randint(0, height - span)
    return [start + 2 * i for i in range(count)]


def _find_free_positions(blocked: np.ndarray, patch_shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = blocked.shape
    ph, pw = patch_shape
    positions: list[tuple[int, int]] = []
    for top in range(h - ph + 1):
        for left in range(w - pw + 1):
            if not blocked[top : top + ph, left : left + pw].any():
                positions.append((top, left))
    return positions


def _place_patch(canvas: np.ndarray, blocked: np.ndarray, patch: np.ndarray, color: int) -> bool:
    positions = _find_free_positions(blocked, patch.shape)
    if not positions:
        return False

    top, left = random.choice(positions)
    rr, cc = np.where(patch)
    canvas[top + rr, left + cc] = int(color)

    ph, pw = patch.shape
    r0 = max(0, top - BOX_PADDING)
    r1 = min(canvas.shape[0], top + ph + BOX_PADDING)
    c0 = max(0, left - BOX_PADDING)
    c1 = min(canvas.shape[1], left + pw + BOX_PADDING)
    blocked[r0:r1, c0:c1] = True
    return True


def _expected_output(height: int, width: int, legend_rows: list[int], items: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros((height, width), dtype=int)
    for row, (color, turns) in zip(legend_rows, items, strict=False):
        out[row, width - 1] = int(color)
        if turns > 0:
            out[row, :turns] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    line_count = random.randint(*LINE_COUNT_RANGE)
    colors = random.sample(COLORS, line_count)
    turn_counts = [random.randint(*TURN_COUNT_RANGE) for _ in range(line_count)]
    if max(turn_counts) == 0:
        turn_counts[random.randrange(line_count)] = random.randint(1, TURN_COUNT_RANGE[1])

    patches = [ _make_polyline_patch(turns) for turns in turn_counts ]
    max_patch_h = max(int(p.shape[0]) for p in patches)
    max_patch_w = max(int(p.shape[1]) for p in patches)
    padded_area = sum((int(p.shape[0]) + 2 * BOX_PADDING) * (int(p.shape[1]) + 2 * BOX_PADDING) for p in patches)

    min_height = max(GRID_HEIGHT_RANGE[0], 2 * line_count - 1, max_patch_h)
    min_width = max(GRID_WIDTH_RANGE[0], max(turn_counts) + 1, max_patch_w + 2)
    if min_height > GRID_HEIGHT_RANGE[1] or min_width > GRID_WIDTH_RANGE[1]:
        return None

    height = random.randint(min_height, GRID_HEIGHT_RANGE[1])
    width = random.randint(min_width, GRID_WIDTH_RANGE[1])
    main_width = width - 2
    if main_width <= 0 or max_patch_w > main_width:
        return None

    while height * main_width < padded_area * 2:
        if width < GRID_WIDTH_RANGE[1]:
            width += 1
            main_width = width - 2
        elif height < GRID_HEIGHT_RANGE[1]:
            height += 1
        else:
            return None
        if max_patch_w > main_width:
            return None

    legend_rows = _pick_legend_rows(height, line_count)
    items = list(zip(colors, turn_counts, patches, strict=False))
    random.shuffle(items)
    items.sort(key=lambda item: item[2].shape[0] * item[2].shape[1], reverse=True)

    canvas = np.zeros((height, main_width), dtype=int)
    blocked = np.zeros_like(canvas, dtype=bool)
    for color, _turns, patch in items:
        if not _place_patch(canvas, blocked, patch, color):
            return None

    grid = np.zeros((height, width), dtype=int)
    grid[:, :main_width] = canvas
    for row, color in zip(legend_rows, colors, strict=False):
        grid[row, width - 1] = int(color)

    expected = _expected_output(height, width, legend_rows, list(zip(colors, turn_counts, strict=False)))
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
