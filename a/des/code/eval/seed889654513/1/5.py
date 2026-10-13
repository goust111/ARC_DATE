import random
from collections import deque

import numpy as np


BG = 0
GRAY = 5
STEP_COLOR_TO_COUNT = {2: 1, 3: 2, 4: 3}
COUNT_TO_STEP_COLOR = {v: k for k, v in STEP_COLOR_TO_COUNT.items()}
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Avoid using gray in the input so overlap remains visually unambiguous.
FENCE_COLOR_POOL = (1, 6, 7, 8, 9)
SEED_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(coords)
    return out


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _parse_legend(row: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    c = 0
    w = len(row)
    while c < w:
        if int(row[c]) == BG:
            c += 1
            continue
        assert c + 1 < w and int(row[c + 1]) != BG, "legend entries must be horizontal 1x2 pairs"
        fence_color = int(row[c])
        step_color = int(row[c + 1])
        assert step_color in STEP_COLOR_TO_COUNT, "legend step color must be red, green, or yellow"
        assert fence_color not in mapping, "each fence color must have exactly one legend entry"
        mapping[fence_color] = STEP_COLOR_TO_COUNT[step_color]
        c += 2
    assert mapping, "expected at least one legend entry on the bottom row"
    return mapping


def _is_rect_frame(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    want = np.zeros_like(mask, dtype=bool)
    want[0, :] = True
    want[-1, :] = True
    want[:, 0] = True
    want[:, -1] = True
    return bool(np.array_equal(mask, want))


def _detect_fences(body: np.ndarray, legend_map: dict[int, int]):
    fences = []
    for color, steps in sorted(legend_map.items()):
        comps = _components(body == color)
        frames = []
        for coords in comps:
            r0, c0, r1, c1 = _bbox(coords)
            sub = body[r0 : r1 + 1, c0 : c1 + 1] == color
            if _is_rect_frame(sub):
                frames.append(
                    {
                        "color": int(color),
                        "steps": int(steps),
                        "bbox": (int(r0), int(c0), int(r1), int(c1)),
                        "coords": [(int(r), int(c)) for r, c in coords],
                    }
                )
        assert len(frames) == 1, f"expected one rectangular fence for color {color}"
        fences.append(frames[0])
    return fences


def _grow(seed_cells: list[tuple[int, int]], steps: int, base: np.ndarray):
    h, w = base.shape
    reached = set(seed_cells)
    frontier = set(seed_cells)
    for _ in range(steps):
        nxt = set()
        for r, c in frontier:
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in reached:
                    continue
                if int(base[nr, nc]) != BG:
                    continue
                reached.add((nr, nc))
                nxt.add((nr, nc))
        if not nxt:
            break
        frontier = nxt
    return reached


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each seed outward by the number of layers encoded in the bottom legend row,
    using fence color -> step-color lookup and painting overlaps gray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 4 and g.shape[1] >= 6, "expected a non-empty 2D grid"

    legend_map = _parse_legend(g[-1])
    body = g[:-1].copy()
    fences = _detect_fences(body, legend_map)

    cleared = body.copy()
    seeds = []
    for fence in fences:
        r0, c0, r1, c1 = fence["bbox"]
        for r, c in fence["coords"]:
            cleared[r, c] = BG

        inner = body[r0 + 1 : r1, c0 + 1 : c1]
        seed_colors = [int(color) for color in np.unique(inner) if int(color) != BG]
        assert len(seed_colors) == 1, f"fence {fence['color']} must contain exactly one seed color"
        seed_color = seed_colors[0]
        seed_cells = [
            (r, c)
            for r in range(r0 + 1, r1)
            for c in range(c0 + 1, c1)
            if int(body[r, c]) == seed_color
        ]
        assert seed_cells, f"fence {fence['color']} must contain at least one seed pixel"
        seeds.append({"color": seed_color, "steps": fence["steps"], "cells": seed_cells})

    out = g.copy()
    out[:-1] = cleared
    reach_count = np.zeros_like(cleared, dtype=np.int16)

    for seed in seeds:
        reached = _grow(seed["cells"], seed["steps"], cleared)
        for r, c in reached:
            if reach_count[r, c] == 0:
                out[r, c] = seed["color"]
            else:
                out[r, c] = GRAY
            reach_count[r, c] += 1

    out[-1] = g[-1]
    return out


def _bboxes_too_close(a, b, pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _random_connected_shape(
    rng: random.Random,
    h: int,
    w: int,
    min_cells: int,
    max_cells: int,
    anchor: str | None = None,
):
    area = rng.randint(min_cells, min(max_cells, h * w))
    if anchor == "left":
        start = (rng.randrange(h), 0)
    elif anchor == "right":
        start = (rng.randrange(h), w - 1)
    elif anchor == "up":
        start = (0, rng.randrange(w))
    elif anchor == "down":
        start = (h - 1, rng.randrange(w))
    else:
        start = (rng.randrange(h), rng.randrange(w))

    shape = {start}
    while len(shape) < area:
        frontier = []
        for r, c in shape:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in shape:
                    frontier.append((nr, nc))
        frontier = list(dict.fromkeys(frontier))
        if not frontier:
            break
        shape.add(rng.choice(frontier))
    return shape


def _paint_frame(grid: np.ndarray, bbox, color: int):
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _place_random_bbox(
    rng: random.Random,
    grid_h: int,
    grid_w: int,
    outer_h: int,
    outer_w: int,
    placed: list[tuple[int, int, int, int]],
    pad: int,
):
    max_top = grid_h - outer_h - 2
    max_left = grid_w - outer_w - 1
    if max_top < 1 or max_left < 1:
        return None
    for _ in range(80):
        r0 = rng.randint(1, max_top)
        c0 = rng.randint(1, max_left)
        bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
        if any(_bboxes_too_close(bbox, other, pad) for other in placed):
            continue
        return bbox
    return None


def _place_overlap_bbox(
    rng: random.Random,
    grid_h: int,
    grid_w: int,
    outer_h: int,
    outer_w: int,
    anchor_bbox,
    placed: list[tuple[int, int, int, int]],
):
    ar0, ac0, ar1, ac1 = anchor_bbox
    ah = ar1 - ar0 + 1
    aw = ac1 - ac0 + 1
    max_top = grid_h - outer_h - 2
    max_left = grid_w - outer_w - 1
    choices = []

    for gap in (1, 2):
        # Right / left.
        c0 = ac1 + gap + 1
        if c0 <= max_left:
            for offset in (-1, 0, 1):
                r0 = min(max(1, ar0 + offset), max_top)
                choices.append((r0, c0, "right"))
        c0 = ac0 - gap - outer_w
        if c0 >= 1:
            for offset in (-1, 0, 1):
                r0 = min(max(1, ar0 + offset), max_top)
                choices.append((r0, c0, "left"))
        # Below / above.
        r0 = ar1 + gap + 1
        if r0 <= max_top:
            for offset in (-1, 0, 1):
                c0 = min(max(1, ac0 + offset), max_left)
                choices.append((r0, c0, "down"))
        r0 = ar0 - gap - outer_h
        if r0 >= 1:
            for offset in (-1, 0, 1):
                c0 = min(max(1, ac0 + offset), max_left)
                choices.append((r0, c0, "up"))

    rng.shuffle(choices)
    for r0, c0, side in choices:
        bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
        if any(_bboxes_too_close(bbox, other, 1) for other in placed):
            continue
        if side in {"right", "left"}:
            if min(ar1, bbox[2]) - max(ar0, bbox[0]) + 1 < min(ah, outer_h) - 1:
                continue
        else:
            if min(ac1, bbox[3]) - max(ac0, bbox[1]) + 1 < min(aw, outer_w) - 1:
                continue
        return bbox, side
    return None, None


def _paint_seed(grid: np.ndarray, bbox, seed_color: int, shape: set[tuple[int, int]]):
    r0, c0, _, _ = bbox
    for rr, cc in shape:
        grid[r0 + 1 + rr, c0 + 1 + cc] = seed_color


def _build_legend(
    rng: random.Random,
    width: int,
    fence_colors: list[int],
    steps: list[int],
):
    row = np.zeros(width, dtype=int)
    order = list(range(len(fence_colors)))
    rng.shuffle(order)
    gaps = [rng.randint(1, 2) for _ in range(max(0, len(order) - 1))]
    span = 2 * len(order) + sum(gaps)
    assert span <= width, "legend does not fit"
    start = rng.randint(0, width - span)
    c = start
    for idx_i, idx in enumerate(order):
        row[c] = fence_colors[idx]
        row[c + 1] = COUNT_TO_STEP_COLOR[steps[idx]]
        c += 2
        if idx_i != len(order) - 1:
            c += gaps[idx_i]
    return row


def _sample_specs(rng: random.Random, count: int, grid_h: int, grid_w: int):
    specs = []
    for _ in range(count):
        inner_h = rng.randint(3, 5)
        inner_w = rng.randint(3, 6)
        if grid_h >= 20 and rng.random() < 0.25:
            inner_h = min(inner_h + 1, 6)
        if grid_w >= 24 and rng.random() < 0.25:
            inner_w = min(inner_w + 1, 7)
        specs.append((inner_h + 2, inner_w + 2))
    return specs


def _gen_attempt(rng: random.Random):
    h = rng.randint(15, 24)
    w = rng.randint(18, 28)
    require_overlap = bool(rng.randrange(2))

    max_objects = 4 if h >= 18 and w >= 22 else 3
    count = rng.randint(2, max_objects)
    fence_colors = rng.sample(FENCE_COLOR_POOL, count)
    seed_choices = [color for color in SEED_COLOR_POOL if color not in fence_colors]
    seed_colors = rng.sample(seed_choices, count)
    steps = [rng.randint(1, 3) for _ in range(count)]
    specs = _sample_specs(rng, count, h, w)

    grid = np.zeros((h, w), dtype=int)
    bboxes: list[tuple[int, int, int, int]] = []
    anchor_sides: list[str | None] = [None] * count

    first_bbox = _place_random_bbox(rng, h, w, specs[0][0], specs[0][1], [], 1 if require_overlap else 4)
    if first_bbox is None:
        return None
    bboxes.append(first_bbox)

    if count >= 2 and require_overlap:
        second_bbox, side = _place_overlap_bbox(rng, h, w, specs[1][0], specs[1][1], first_bbox, bboxes)
        if second_bbox is None:
            return None
        bboxes.append(second_bbox)
        if side == "right":
            anchor_sides[0] = "right"
            anchor_sides[1] = "left"
        elif side == "left":
            anchor_sides[0] = "left"
            anchor_sides[1] = "right"
        elif side == "down":
            anchor_sides[0] = "down"
            anchor_sides[1] = "up"
        else:
            anchor_sides[0] = "up"
            anchor_sides[1] = "down"
        start_idx = 2
    else:
        start_idx = 1

    for idx in range(start_idx, count):
        pad = 2 if require_overlap else 4
        bbox = _place_random_bbox(rng, h, w, specs[idx][0], specs[idx][1], bboxes, pad)
        if bbox is None:
            return None
        bboxes.append(bbox)

    for idx, bbox in enumerate(bboxes):
        _paint_frame(grid, bbox, fence_colors[idx])
        inner_h = bbox[2] - bbox[0] - 1
        inner_w = bbox[3] - bbox[1] - 1
        max_cells = min(5, inner_h * inner_w)
        min_cells = 1 if max_cells == 1 else 2
        shape = _random_connected_shape(
            rng,
            inner_h,
            inner_w,
            min_cells=min_cells,
            max_cells=max_cells,
            anchor=anchor_sides[idx],
        )
        _paint_seed(grid, bbox, seed_colors[idx], shape)

    grid[-1] = _build_legend(rng, w, fence_colors, steps)

    try:
        out = main(grid)
    except AssertionError:
        return None

    has_overlap = bool(np.any(out[:-1] == GRAY))
    if has_overlap != require_overlap:
        return None

    # Fence removal should reveal at least one grown cell beyond the original seed.
    if np.array_equal(out[:-1], grid[:-1]):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: exceeded {GEN_RETRIES} retries")
