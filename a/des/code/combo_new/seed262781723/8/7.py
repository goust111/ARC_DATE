from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
GREEN = 3
GREY = 5
SEED_COLORS = (1, 4, 6, 7, 8, 9)

GEN_ATTEMPTS = 500
SHAPE_ATTEMPTS = 250


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(coords)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _in_bbox(r: int, c: int, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _corner_touch_count(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> int:
    r0, c0, r1, c1 = bbox
    h, w = shape
    return int(r0 == 0 and c0 == 0) + int(r0 == 0 and c1 == w - 1) + int(r1 == h - 1 and c0 == 0) + int(
        r1 == h - 1 and c1 == w - 1
    )


def _extract_demo(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    candidates: list[tuple[tuple[int, int, int, int], list[tuple[int, int]]]] = []
    for comp in _components(g == GREY):
        bbox = _bbox(comp)
        r0, c0, r1, c1 = bbox
        if _corner_touch_count(bbox, g.shape) != 1:
            continue
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        frame = np.zeros_like(patch, dtype=bool)
        frame[0, :] = True
        frame[-1, :] = True
        frame[:, 0] = True
        frame[:, -1] = True
        if not np.all(patch[frame] == GREY):
            continue
        if np.any(patch[~frame] == GREY):
            continue
        if len(comp) != int(frame.sum()):
            continue

        inner = patch[1:-1, 1:-1]
        if not np.all((inner == BG) | (inner == GREEN) | (inner == RED)):
            continue
        if np.count_nonzero(inner == RED) != 1:
            continue
        colored = np.argwhere((inner == GREEN) | (inner == RED))
        if colored.shape[0] < 2 or np.count_nonzero(inner == GREEN) < 1:
            continue
        anchor = tuple(map(int, np.argwhere(inner == RED)[0]))
        offsets = sorted((int(r - anchor[0]), int(c - anchor[1])) for r, c in colored)
        candidates.append((bbox, offsets))

    assert len(candidates) == 1, "expected exactly one corner-anchored grey demo frame"
    return candidates[0]


def _find_seeds(grid: np.ndarray, demo_bbox: tuple[int, int, int, int]) -> list[tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    seeds: list[tuple[int, int, int]] = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if _in_bbox(r, c, demo_bbox):
                continue
            value = int(g[r, c])
            if value not in {BG, RED, GREEN, GREY}:
                seeds.append((r, c, value))
    return seeds


def _stamp_all(
    base: np.ndarray,
    demo_bbox: tuple[int, int, int, int],
    offsets: list[tuple[int, int]],
    seeds: list[tuple[int, int, int]],
) -> np.ndarray:
    out = np.asarray(base, dtype=int).copy()
    height, width = out.shape
    for r, c, color in sorted(seeds):
        for dr, dc in offsets:
            rr = r + dr
            cc = c + dc
            if not (0 <= rr < height and 0 <= cc < width):
                continue
            if _in_bbox(rr, cc, demo_bbox):
                continue
            out[rr, cc] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each scattered seed pixel into the corner demo's anchored template while keeping the demo unchanged.

    Input structure:
    - The background is black.
    - Exactly one grid corner contains a grey rectangular frame whose interior shows a small template in green, with one template cell marked red as the anchor.
    - Outside that demo frame, the main area contains isolated single-pixel seeds in colors other than grey, green, and red.

    Rule:
    - Detect the corner grey frame and read all green/red template cells inside it.
    - Use the red cell as the anchor and record every template cell's offset from that anchor.
    - For each seed, scanned row by row from top to bottom and left to right within each row, stamp the whole template in the seed's color with the anchor aligned to the seed.
    - Clip stamped cells that fall outside the grid, let later seeds overwrite earlier ones, and never modify the demo frame area.

    Key insight:
    - The framed corner patch is a stamping legend: the red pixel is not an output color cue but the alignment point for replaying the green shape at every seed.
    """
    g = np.asarray(grid, dtype=int)
    demo_bbox, offsets = _extract_demo(g)
    seeds = _find_seeds(g, demo_bbox)
    return _stamp_all(g, demo_bbox, offsets, seeds)


def _normalize(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return sorted((r - min_r, c - min_c) for r, c in coords)


def _overlap_shifts(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    shifts = set()
    for r1, c1 in cells:
        for r2, c2 in cells:
            dr = r1 - r2
            dc = c1 - c2
            if dr == 0 and dc == 0:
                continue
            shifts.add((dr, dc))
    return sorted(shifts)


def _random_shape(rng: random.Random, require_overlap_pair: bool) -> list[tuple[int, int]]:
    for _ in range(SHAPE_ATTEMPTS):
        target = rng.randint(3, 7)
        cells = {(0, 0)}
        while len(cells) < target:
            r, c = rng.choice(tuple(cells))
            dr, dc = rng.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
            cells.add((r + dr, c + dc))

        shape = _normalize(cells)
        height = max(r for r, _ in shape) + 1
        width = max(c for _, c in shape) + 1
        if height > 5 or width > 5:
            continue
        if height == 1 or width == 1:
            if rng.random() < 0.65:
                continue
        if require_overlap_pair:
            far_shifts = [shift for shift in _overlap_shifts(shape) if abs(shift[0]) + abs(shift[1]) != 1]
            if not far_shifts:
                continue
        return shape
    raise RuntimeError("failed to sample a non-degenerate template shape")


def _demo_bbox_for_corner(corner: str, height: int, width: int, demo_h: int, demo_w: int) -> tuple[int, int, int, int]:
    if corner == "TL":
        return 0, 0, demo_h - 1, demo_w - 1
    if corner == "TR":
        return 0, width - demo_w, demo_h - 1, width - 1
    if corner == "BL":
        return height - demo_h, 0, height - 1, demo_w - 1
    if corner == "BR":
        return height - demo_h, width - demo_w, height - 1, width - 1
    raise ValueError(corner)


def _paint_demo(
    grid: np.ndarray,
    demo_bbox: tuple[int, int, int, int],
    shape: list[tuple[int, int]],
    anchor_cell: tuple[int, int],
    pad_top: int,
    pad_left: int,
) -> None:
    r0, c0, r1, c1 = demo_bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = BG
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    for cell in shape:
        rr = r0 + 1 + pad_top + cell[0]
        cc = c0 + 1 + pad_left + cell[1]
        grid[rr, cc] = RED if cell == anchor_cell else GREEN


def _expand_bbox(bbox: tuple[int, int, int, int], pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    h, w = shape
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _seed_ok(
    pos: tuple[int, int],
    existing: list[tuple[int, int, int]],
    blocked_bbox: tuple[int, int, int, int],
) -> bool:
    r, c = pos
    if _in_bbox(r, c, blocked_bbox):
        return False
    for sr, sc, _ in existing:
        if abs(sr - r) + abs(sc - c) <= 1:
            return False
    return True


def _stamp_cells(
    pos: tuple[int, int],
    offsets: list[tuple[int, int]],
    shape: tuple[int, int],
    demo_bbox: tuple[int, int, int, int],
) -> tuple[set[tuple[int, int]], bool, bool]:
    height, width = shape
    cells: set[tuple[int, int]] = set()
    clipped = False
    hits_demo = False
    for dr, dc in offsets:
        rr = pos[0] + dr
        cc = pos[1] + dc
        if not (0 <= rr < height and 0 <= cc < width):
            clipped = True
            continue
        if _in_bbox(rr, cc, demo_bbox):
            hits_demo = True
            continue
        cells.add((rr, cc))
    return cells, clipped, hits_demo


def _place_non_overlapping_seed(
    rng: random.Random,
    grid_shape: tuple[int, int],
    demo_bbox: tuple[int, int, int, int],
    blocked_bbox: tuple[int, int, int, int],
    offsets: list[tuple[int, int]],
    existing_seeds: list[tuple[int, int, int]],
    used_cells: set[tuple[int, int]],
    require_clipped: bool,
) -> tuple[tuple[int, int], set[tuple[int, int]], bool] | None:
    height, width = grid_shape
    positions = [(r, c) for r in range(height) for c in range(width)]
    rng.shuffle(positions)
    for pos in positions:
        if not _seed_ok(pos, existing_seeds, blocked_bbox):
            continue
        cells, clipped, hits_demo = _stamp_cells(pos, offsets, grid_shape, demo_bbox)
        if hits_demo or clipped != require_clipped:
            continue
        if cells & used_cells:
            continue
        return pos, cells, clipped
    return None


def _place_overlap_pair(
    rng: random.Random,
    grid_shape: tuple[int, int],
    demo_bbox: tuple[int, int, int, int],
    blocked_bbox: tuple[int, int, int, int],
    offsets: list[tuple[int, int]],
    require_unclipped: bool,
) -> tuple[tuple[int, int], set[tuple[int, int]], bool, tuple[int, int], set[tuple[int, int]], bool] | None:
    height, width = grid_shape
    shifts = [shift for shift in _overlap_shifts(offsets) if abs(shift[0]) + abs(shift[1]) != 1]
    rng.shuffle(shifts)
    positions = [(r, c) for r in range(height) for c in range(width)]
    rng.shuffle(positions)
    for pos_a in positions:
        if not _seed_ok(pos_a, [], blocked_bbox):
            continue
        cells_a, clipped_a, hits_demo_a = _stamp_cells(pos_a, offsets, grid_shape, demo_bbox)
        if hits_demo_a or (require_unclipped and clipped_a):
            continue
        for dr, dc in shifts:
            pos_b = (pos_a[0] + dr, pos_a[1] + dc)
            if not (0 <= pos_b[0] < height and 0 <= pos_b[1] < width):
                continue
            if not _seed_ok(pos_b, [(pos_a[0], pos_a[1], -1)], blocked_bbox):
                continue
            cells_b, clipped_b, hits_demo_b = _stamp_cells(pos_b, offsets, grid_shape, demo_bbox)
            if hits_demo_b or (require_unclipped and clipped_b):
                continue
            if not (cells_a & cells_b):
                continue
            return pos_a, cells_a, clipped_a, pos_b, cells_b, clipped_b
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    force_clipped = rng.random() < 0.5
    force_overlap = rng.random() < 0.5

    shape = _random_shape(rng, require_overlap_pair=force_overlap)
    anchor_cell = rng.choice(shape)
    offsets = sorted((r - anchor_cell[0], c - anchor_cell[1]) for r, c in shape)

    shape_h = max(r for r, _ in shape) + 1
    shape_w = max(c for _, c in shape) + 1
    pad_top = rng.randint(0, 2)
    pad_bottom = rng.randint(0, 2)
    pad_left = rng.randint(0, 2)
    pad_right = rng.randint(0, 2)
    demo_h = shape_h + pad_top + pad_bottom + 2
    demo_w = shape_w + pad_left + pad_right + 2

    min_h = max(12, demo_h + 4)
    min_w = max(12, demo_w + 4)
    if min_h > 30 or min_w > 30:
        return None
    height = rng.randint(min_h, 30)
    width = rng.randint(min_w, 30)

    demo_corner = rng.choice(("TL", "TR", "BL", "BR"))
    demo_bbox = _demo_bbox_for_corner(demo_corner, height, width, demo_h, demo_w)
    blocked_bbox = _expand_bbox(demo_bbox, 1, (height, width))

    min_seed_count = 3 if force_clipped and force_overlap else 2
    seed_count = rng.randint(min_seed_count, min(len(SEED_COLORS), 5))
    colors = list(rng.sample(SEED_COLORS, seed_count))

    seeds: list[tuple[int, int, int]] = []
    used_cells: set[tuple[int, int]] = set()
    has_clipped = False

    if force_overlap:
        pair = _place_overlap_pair(
            rng,
            (height, width),
            demo_bbox,
            blocked_bbox,
            offsets,
            require_unclipped=not force_clipped,
        )
        if pair is None:
            return None
        pos_a, cells_a, clipped_a, pos_b, cells_b, clipped_b = pair
        seeds.append((pos_a[0], pos_a[1], colors.pop()))
        seeds.append((pos_b[0], pos_b[1], colors.pop()))
        used_cells |= cells_a | cells_b
        has_clipped = clipped_a or clipped_b

    while len(seeds) < seed_count:
        need_clipped = force_clipped and not has_clipped and (seed_count - len(seeds) == 1 or not force_overlap)
        placed = _place_non_overlapping_seed(
            rng,
            (height, width),
            demo_bbox,
            blocked_bbox,
            offsets,
            seeds,
            used_cells,
            require_clipped=need_clipped,
        )
        if placed is None:
            if need_clipped:
                return None
            placed = _place_non_overlapping_seed(
                rng,
                (height, width),
                demo_bbox,
                blocked_bbox,
                offsets,
                seeds,
                used_cells,
                require_clipped=False,
            )
            if placed is None:
                return None
        pos, cells, clipped = placed
        seeds.append((pos[0], pos[1], colors.pop()))
        used_cells |= cells
        has_clipped = has_clipped or clipped

    if force_clipped != has_clipped:
        return None

    overlap_pairs = 0
    footprints = []
    for r, c, _ in seeds:
        cells, clipped, hits_demo = _stamp_cells((r, c), offsets, (height, width), demo_bbox)
        if hits_demo:
            return None
        footprints.append(cells)
    for i in range(len(footprints)):
        for j in range(i + 1, len(footprints)):
            if footprints[i] & footprints[j]:
                overlap_pairs += 1
    if force_overlap and overlap_pairs == 0:
        return None
    if not force_overlap and overlap_pairs != 0:
        return None

    grid = np.full((height, width), BG, dtype=int)
    _paint_demo(grid, demo_bbox, shape, anchor_cell, pad_top, pad_left)
    for r, c, color in seeds:
        grid[r, c] = int(color)

    expected = _stamp_all(grid, demo_bbox, offsets, seeds)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if np.array_equal(actual, grid):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamping-template input")
