from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
SIDE_NAMES = ("top", "bottom", "left", "right")
DIR_NAMES = ("up", "down", "left", "right")
DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ARROW_OFFSETS = {
    "up": ((0, 0), (1, -1), (1, 1)),
    "down": ((0, 0), (-1, -1), (-1, 1)),
    "left": ((0, 0), (-1, 1), (1, 1)),
    "right": ((0, 0), (-1, -1), (1, -1)),
}

GRID_RANGE = (20, 30)
FRAME_OUTER_RANGE = (8, 16)
FRAME_MARGIN = 6
HEAD_EDGE_MARGIN = 2
COLOR_COUNT_RANGE = (2, 4)
LAYER_SIZE_RANGE = (3, 7)
SECOND_COMPONENT_PROB = 0.35
SECOND_COMPONENT_SIZE_RANGE = (1, 3)
MAX_STEP = 4
GEN_ATTEMPTS = 500


def _background_color(g: np.ndarray) -> int:
    if bool(np.any(g == BG)):
        return int(BG)
    vals, counts = np.unique(np.asarray(g, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _find_frame(g: np.ndarray, bg: int) -> tuple[int, int, int, int, int]:
    best = None
    for raw_color in np.unique(g):
        color = int(raw_color)
        if color == int(bg):
            continue
        coords = np.argwhere(g == int(color))
        if coords.size == 0:
            continue
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 3 or w < 3:
            continue
        perimeter = int(2 * (h + w) - 4)
        if int(coords.shape[0]) != perimeter:
            continue
        if not bool((g[r0, c0 : c1 + 1] == int(color)).all()):
            continue
        if not bool((g[r1, c0 : c1 + 1] == int(color)).all()):
            continue
        if not bool((g[r0 : r1 + 1, c0] == int(color)).all()):
            continue
        if not bool((g[r0 : r1 + 1, c1] == int(color)).all()):
            continue
        area = int(h * w)
        cand = (area, int(r0), int(c0), int(r1), int(c1), int(color))
        if best is None or cand[0] > best[0]:
            best = cand
    assert best is not None, "expected a single-pixel-thick rectangular frame"
    _area, r0, c0, r1, c1, color = best
    return int(r0), int(c0), int(r1), int(c1), int(color)


def _first_seen_colors(inner: np.ndarray, bg: int, frame_color: int) -> list[int]:
    order = []
    seen = set()
    for r in range(int(inner.shape[0])):
        for c in range(int(inner.shape[1])):
            color = int(inner[r, c])
            if color in (int(bg), int(frame_color)) or color in seen:
                continue
            seen.add(int(color))
            order.append(int(color))
    return order


def _decode_arrow(coords: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    pts = np.asarray(coords, dtype=int)
    assert pts.shape == (3, 2), "each arrow must contain exactly three pixels"
    rows = [int(v) for v in pts[:, 0]]
    cols = [int(v) for v in pts[:, 1]]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)

    if h == 3 and w == 2:
        col_counts = {int(c): int(sum(int(v) == int(c) for v in cols)) for c in set(cols)}
        head_cols = [c for c, cnt in col_counts.items() if cnt == 1]
        tail_cols = [c for c, cnt in col_counts.items() if cnt == 2]
        assert len(head_cols) == 1 and len(tail_cols) == 1
        head_col = int(head_cols[0])
        head_pts = [(int(r), int(c)) for r, c in pts if int(c) == int(head_col)]
        assert len(head_pts) == 1 and int(head_pts[0][0]) == int(r0 + 1)
        direction = (0, 1) if int(head_col) == int(c1) else (0, -1)
        return (int(head_pts[0][0]), int(head_pts[0][1])), direction

    if h == 2 and w == 3:
        row_counts = {int(r): int(sum(int(v) == int(r) for v in rows)) for r in set(rows)}
        head_rows = [r for r, cnt in row_counts.items() if cnt == 1]
        tail_rows = [r for r, cnt in row_counts.items() if cnt == 2]
        assert len(head_rows) == 1 and len(tail_rows) == 1
        head_row = int(head_rows[0])
        head_pts = [(int(r), int(c)) for r, c in pts if int(r) == int(head_row)]
        assert len(head_pts) == 1 and int(head_pts[0][1]) == int(c0 + 1)
        direction = (1, 0) if int(head_row) == int(r1) else (-1, 0)
        return (int(head_pts[0][0]), int(head_pts[0][1])), direction

    raise AssertionError("arrow pixels must form a cardinal 3-pixel wedge")


def _arrow_side(coords: np.ndarray, frame: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = map(int, frame)
    pts = np.asarray(coords, dtype=int)
    rows = [int(v) for v in pts[:, 0]]
    cols = [int(v) for v in pts[:, 1]]
    if max(rows) < int(r0):
        return "top"
    if min(rows) > int(r1):
        return "bottom"
    if max(cols) < int(c0):
        return "left"
    if min(cols) > int(c1):
        return "right"

    head, _direction = _decode_arrow(pts)
    hr, hc = map(int, head)
    dists = []
    if hr < int(r0):
        dists.append((int(r0 - hr), "top"))
    if hr > int(r1):
        dists.append((int(hr - r1), "bottom"))
    if hc < int(c0):
        dists.append((int(c0 - hc), "left"))
    if hc > int(c1):
        dists.append((int(hc - c1), "right"))
    assert dists, "arrow must lie outside the frame"
    dists.sort(key=lambda item: item[0])
    return dists[0][1]


def _step_from_head(head: tuple[int, int], side: str, frame: tuple[int, int, int, int]) -> int:
    hr, hc = map(int, head)
    r0, c0, r1, c1 = map(int, frame)
    if side == "top":
        return int(r0 - hr)
    if side == "bottom":
        return int(hr - r1)
    if side == "left":
        return int(c0 - hc)
    if side == "right":
        return int(hc - c1)
    raise AssertionError(f"unknown side: {side}")


def _apply_layers(
    inner_shape: tuple[int, int],
    layers: dict[int, list[tuple[int, int]]],
    order: list[int],
    moves: dict[int, tuple[int, int]],
    bg: int,
) -> tuple[np.ndarray, dict[tuple[int, int], int], bool]:
    out = np.full(inner_shape, int(bg), dtype=int)
    hit_counts: dict[tuple[int, int], int] = {}
    clipped = False
    h, w = map(int, inner_shape)

    for color in order:
        dr, dc = map(int, moves[int(color)])
        for r, c in layers[int(color)]:
            rr = int(r + dr)
            cc = int(c + dc)
            if not (0 <= rr < h and 0 <= cc < w):
                clipped = True
                continue
            key = (int(rr), int(cc))
            hit_counts[key] = int(hit_counts.get(key, 0) + 1)
            out[int(rr), int(cc)] = int(color)
    return out, hit_counts, bool(clipped)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift each color layer inside a grey frame by the vector encoded in its matching outer arrow, then crop to the frame interior.

    Input structure:
    - bg is black.
    - A single-pixel-thick grey rectangular frame sits near the center of the grid.
    - The frame interior contains a multi-color pattern made of non-black, non-grey colors.
    - Outside the frame, each interior color has exactly one same-colored 3-pixel arrow marker on one frame side.

    Rule:
    - For each interior color, isolate all pixels of that color inside the frame and find the matching same-colored arrow outside.
    - Decode the arrow direction from its 3-pixel wedge shape, and decode the step count from the head pixel's distance to the frame edge.
    - Translate that whole color layer by the decoded vector, clip pixels that leave the frame interior, and overlay the shifted layers in the order their colors first appear inside the frame from top to bottom then left to right.
    - Return only the frame interior after all shifted layers have been repainted.

    Key insight:
    - Each arrow carries two controls at once: its wedge orientation gives the movement direction, and its head-to-frame gap gives the movement distance for that color layer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(_background_color(g))
    r0, c0, r1, c1, frame_color = _find_frame(g, bg)
    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    order = _first_seen_colors(inner, bg, int(frame_color))

    layers: dict[int, list[tuple[int, int]]] = {}
    moves: dict[int, tuple[int, int]] = {}
    for color in order:
        layer_coords = [(int(r), int(c)) for r, c in np.argwhere(inner == int(color))]
        assert layer_coords, "each ordered color must appear in the frame interior"
        layers[int(color)] = layer_coords

        outside_coords = np.array(
            [
                (int(r), int(c))
                for r, c in np.argwhere(g == int(color))
                if not (int(r0) < int(r) < int(r1) and int(c0) < int(c) < int(c1))
            ],
            dtype=int,
        )
        head, direction = _decode_arrow(outside_coords)
        side = _arrow_side(outside_coords, (int(r0), int(c0), int(r1), int(c1)))
        step = _step_from_head(head, side, (int(r0), int(c0), int(r1), int(c1)))
        moves[int(color)] = (int(direction[0] * step), int(direction[1] * step))

    out, _hit_counts, _clipped = _apply_layers(inner.shape, layers, order, moves, bg)
    return out


def _centered_origin(total: int, size: int, margin: int) -> int:
    lo = int(margin)
    hi = int(total - margin - size)
    assert lo <= hi
    center = int((total - size) // 2)
    close = [v for v in range(lo, hi + 1) if abs(int(v - center)) <= 2]
    return int(random.choice(close if close else list(range(lo, hi + 1))))


def _min_step(side: str, direction: str) -> int:
    offsets = ARROW_OFFSETS[direction]
    if side == "top":
        return int(1 + max(int(dr) for dr, _dc in offsets))
    if side == "bottom":
        return int(1 - min(int(dr) for dr, _dc in offsets))
    if side == "left":
        return int(1 + max(int(dc) for _dr, dc in offsets))
    if side == "right":
        return int(1 - min(int(dc) for _dr, dc in offsets))
    raise AssertionError(f"unknown side: {side}")


def _safe_seed_candidates(
    inner_h: int,
    inner_w: int,
    move: tuple[int, int],
    occupied: set[tuple[int, int]],
    *,
    boundary_only: bool,
) -> list[tuple[int, int]]:
    dr, dc = map(int, move)
    row_lo = int(max(0, -dr))
    row_hi = int(min(inner_h - 1, inner_h - 1 - dr))
    col_lo = int(max(0, -dc))
    col_hi = int(min(inner_w - 1, inner_w - 1 - dc))
    if row_lo > row_hi or col_lo > col_hi:
        return []

    if boundary_only:
        row_vals = [row_lo] if dr < 0 else [row_hi] if dr > 0 else list(range(row_lo, row_hi + 1))
        col_vals = [col_lo] if dc < 0 else [col_hi] if dc > 0 else list(range(col_lo, col_hi + 1))
    else:
        row_vals = list(range(row_lo, row_hi + 1))
        col_vals = list(range(col_lo, col_hi + 1))

    cands = [(int(r), int(c)) for r in row_vals for c in col_vals if (int(r), int(c)) not in occupied]
    random.shuffle(cands)
    return cands


def _clip_neighbor(
    seed: tuple[int, int],
    move: tuple[int, int],
    inner_h: int,
    inner_w: int,
    blocked: set[tuple[int, int]],
) -> tuple[int, int] | None:
    r, c = map(int, seed)
    dr, dc = map(int, move)
    step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
    step_c = 0 if dc == 0 else (1 if dc > 0 else -1)
    cand = (int(r + step_r), int(c + step_c))
    rr, cc = map(int, cand)
    if not (0 <= rr < inner_h and 0 <= cc < inner_w):
        return None
    if cand in blocked:
        return None
    return cand


def _grow_component(
    seed: tuple[int, int],
    inner_h: int,
    inner_w: int,
    blocked: set[tuple[int, int]],
    target_size: int,
    *,
    extra_first: tuple[int, int] | None = None,
) -> set[tuple[int, int]] | None:
    sr, sc = map(int, seed)
    reach_r = random.randint(2, 4)
    reach_c = random.randint(2, 4)
    box_r0 = int(max(0, sr - reach_r))
    box_r1 = int(min(inner_h - 1, sr + reach_r))
    box_c0 = int(max(0, sc - reach_c))
    box_c1 = int(min(inner_w - 1, sc + reach_c))

    cells = {seed}
    frontier: list[tuple[int, int]] = []

    def push_neighbors(cell: tuple[int, int]) -> None:
        r, c = map(int, cell)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cand = (int(r + dr), int(c + dc))
            rr, cc = map(int, cand)
            if not (box_r0 <= rr <= box_r1 and box_c0 <= cc <= box_c1):
                continue
            if not (0 <= rr < inner_h and 0 <= cc < inner_w):
                continue
            if cand in blocked or cand in cells:
                continue
            frontier.append(cand)

    push_neighbors(seed)
    if extra_first is not None:
        if extra_first in blocked or extra_first in cells:
            return None
        cells.add(extra_first)
        push_neighbors(extra_first)

    while len(cells) < int(target_size) and frontier:
        idx = random.randrange(len(frontier))
        cand = frontier.pop(idx)
        if cand in blocked or cand in cells:
            continue
        cells.add(cand)
        push_neighbors(cand)

    if len(cells) < int(target_size):
        return None
    return cells


def _choose_overlap_pair(
    specs: list[dict],
    inner_h: int,
    inner_w: int,
) -> tuple[tuple[int, int], dict[int, tuple[int, int]]] | None:
    if len(specs) < 2:
        return None
    pair = tuple(random.sample(range(len(specs)), 2))
    idx_a, idx_b = map(int, pair)
    move_a = tuple(map(int, specs[idx_a]["move"]))
    move_b = tuple(map(int, specs[idx_b]["move"]))
    candidates = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    random.shuffle(candidates)
    for out_r, out_c in candidates:
        seed_a = (int(out_r - move_a[0]), int(out_c - move_a[1]))
        seed_b = (int(out_r - move_b[0]), int(out_c - move_b[1]))
        if seed_a == seed_b:
            continue
        for rr, cc in (seed_a, seed_b):
            if not (0 <= rr < inner_h and 0 <= cc < inner_w):
                break
        else:
            return (idx_a, idx_b), {idx_a: seed_a, idx_b: seed_b}
    return None


def _place_layers(
    inner_h: int,
    inner_w: int,
    specs: list[dict],
    *,
    force_overlap: bool,
    force_clip: bool,
) -> tuple[dict[int, list[tuple[int, int]]], bool, bool] | None:
    overlap_info = _choose_overlap_pair(specs, inner_h, inner_w) if force_overlap else None
    overlap_seeds: dict[int, tuple[int, int]] = {}
    overlap_pair: tuple[int, int] | None = None
    if force_overlap:
        if overlap_info is None:
            return None
        overlap_pair, overlap_seeds = overlap_info

    clip_candidates = [idx for idx in range(len(specs)) if overlap_pair is None or idx not in overlap_pair]
    clip_idx = random.choice(clip_candidates) if force_clip and clip_candidates else None

    occupied: set[tuple[int, int]] = set()
    layers: dict[int, list[tuple[int, int]]] = {}

    for idx, spec in enumerate(specs):
        color = int(spec["color"])
        move = tuple(map(int, spec["move"]))
        want_clip = clip_idx is not None and int(idx) == int(clip_idx)
        blocked = set(occupied)

        if idx in overlap_seeds:
            seed = overlap_seeds[idx]
            if seed in blocked:
                return None
        else:
            cands = _safe_seed_candidates(inner_h, inner_w, move, blocked, boundary_only=want_clip)
            if not cands:
                return None
            seed = cands[0]

        extra_first = None
        target_size = random.randint(*LAYER_SIZE_RANGE)
        if want_clip:
            extra_first = _clip_neighbor(seed, move, inner_h, inner_w, blocked | {seed})
            if extra_first is None:
                return None
            target_size = max(int(target_size), 4)

        cells = _grow_component(seed, inner_h, inner_w, blocked, target_size, extra_first=extra_first)
        if cells is None:
            return None

        if random.random() < SECOND_COMPONENT_PROB:
            extra_blocked = set(occupied) | cells
            extra_cands = _safe_seed_candidates(inner_h, inner_w, move, extra_blocked, boundary_only=False)
            if extra_cands:
                extra_seed = extra_cands[0]
                extra_size = random.randint(*SECOND_COMPONENT_SIZE_RANGE)
                extra_cells = _grow_component(extra_seed, inner_h, inner_w, extra_blocked, extra_size)
                if extra_cells is not None:
                    cells |= extra_cells

        occupied |= cells
        layers[int(color)] = sorted((int(r), int(c)) for r, c in cells)

    inner = np.full((inner_h, inner_w), int(BG), dtype=int)
    for color, coords in layers.items():
        for r, c in coords:
            inner[int(r), int(c)] = int(color)

    order = _first_seen_colors(inner, int(BG), int(FRAME))
    moves = {int(spec["color"]): tuple(map(int, spec["move"])) for spec in specs}
    _out, hits, clipped = _apply_layers(inner.shape, layers, order, moves, int(BG))
    overlap_happened = any(int(count) > 1 for count in hits.values())
    return layers, bool(overlap_happened), bool(clipped)


def _place_arrow(
    grid: np.ndarray,
    frame: tuple[int, int, int, int],
    spec: dict,
) -> None:
    r0, c0, r1, c1 = map(int, frame)
    side = str(spec["side"])
    direction = str(spec["direction"])
    step = int(spec["step"])

    if side == "top":
        head = (int(r0 - step), int(random.randint(c0 + HEAD_EDGE_MARGIN, c1 - HEAD_EDGE_MARGIN)))
    elif side == "bottom":
        head = (int(r1 + step), int(random.randint(c0 + HEAD_EDGE_MARGIN, c1 - HEAD_EDGE_MARGIN)))
    elif side == "left":
        head = (int(random.randint(r0 + HEAD_EDGE_MARGIN, r1 - HEAD_EDGE_MARGIN)), int(c0 - step))
    elif side == "right":
        head = (int(random.randint(r0 + HEAD_EDGE_MARGIN, r1 - HEAD_EDGE_MARGIN)), int(c1 + step))
    else:
        raise AssertionError(f"unknown side: {side}")

    for dr, dc in ARROW_OFFSETS[direction]:
        rr = int(head[0] + dr)
        cc = int(head[1] + dc)
        grid[rr, cc] = int(spec["color"])


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    grid_h = random.randint(*GRID_RANGE)
    grid_w = random.randint(*GRID_RANGE)
    max_frame_h = min(FRAME_OUTER_RANGE[1], int(grid_h - 2 * FRAME_MARGIN))
    max_frame_w = min(FRAME_OUTER_RANGE[1], int(grid_w - 2 * FRAME_MARGIN))
    if max_frame_h < FRAME_OUTER_RANGE[0] or max_frame_w < FRAME_OUTER_RANGE[0]:
        return None

    n_colors = random.randint(*COLOR_COUNT_RANGE)
    frame_h = random.randint(FRAME_OUTER_RANGE[0], max_frame_h)
    frame_w = random.randint(FRAME_OUTER_RANGE[0], max_frame_w)
    r0 = _centered_origin(grid_h, frame_h, FRAME_MARGIN)
    c0 = _centered_origin(grid_w, frame_w, FRAME_MARGIN)
    r1 = int(r0 + frame_h - 1)
    c1 = int(c0 + frame_w - 1)
    inner_h = int(frame_h - 2)
    inner_w = int(frame_w - 2)
    if inner_h < 4 or inner_w < 4:
        return None

    colors = random.sample(PALETTE, n_colors)
    directions = random.sample(DIR_NAMES, n_colors)
    sides = random.sample(SIDE_NAMES, n_colors)
    specs = []
    for color, direction, side in zip(colors, directions, sides):
        min_step = _min_step(side, direction)
        if min_step > MAX_STEP:
            return None
        step = random.randint(min_step, MAX_STEP)
        dr, dc = map(int, DIR_VECTORS[direction])
        specs.append(
            {
                "color": int(color),
                "direction": str(direction),
                "side": str(side),
                "step": int(step),
                "move": (int(dr * step), int(dc * step)),
            }
        )

    force_overlap = bool(random.random() < 0.5 and n_colors >= 2)
    force_clip = bool(random.random() < 0.5)
    placed = _place_layers(inner_h, inner_w, specs, force_overlap=force_overlap, force_clip=force_clip)
    if placed is None:
        return None
    layers, overlap_happened, clipped = placed
    if force_overlap and not overlap_happened:
        return None
    if force_clip and not clipped:
        return None

    grid = np.full((grid_h, grid_w), int(BG), dtype=int)
    grid[r0, c0 : c1 + 1] = int(FRAME)
    grid[r1, c0 : c1 + 1] = int(FRAME)
    grid[r0 : r1 + 1, c0] = int(FRAME)
    grid[r0 : r1 + 1, c1] = int(FRAME)

    inner = np.full((inner_h, inner_w), int(BG), dtype=int)
    for color, coords in layers.items():
        for rr, cc in coords:
            inner[int(rr), int(cc)] = int(color)
    grid[r0 + 1 : r1, c0 + 1 : c1] = inner

    for spec in specs:
        _place_arrow(grid, (int(r0), int(c0), int(r1), int(c1)), spec)

    order = _first_seen_colors(inner, int(BG), int(FRAME))
    moves = {int(spec["color"]): tuple(map(int, spec["move"])) for spec in specs}
    expected, _hits, _clipped = _apply_layers(inner.shape, layers, order, moves, int(BG))

    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_case()
        if built is None:
            continue
        grid, _expected = built
        return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
