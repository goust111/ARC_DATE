from __future__ import annotations

import random

import numpy as np


BG = 0
SEED = 3
FRAME_COLORS = (1, 2, 4, 5, 6, 7, 8, 9)

GRID_H_RANGE = (12, 26)
GRID_W_RANGE = (12, 26)
FRAME_OUTER_MIN = 5
FRAME_OUTER_MAX = 11
FRAME_PADDING = 1
FRAME_TRIES = 120
GEN_RETRIES = 400
EDGE_TOUCH_PROB = 0.3
ALIGNMENT_PROB = 0.4


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _frame_bbox(component: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in component]
    cols = [c for _, c in component]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    expected = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    if set(component) != expected:
        return None
    return (r0, c0, r1, c1)


def _extract_regions(grid: np.ndarray) -> list[dict[str, object]]:
    regions: list[dict[str, object]] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (BG, SEED):
            continue
        for component in _components(grid == color):
            bbox = _frame_bbox(component)
            if bbox is None:
                continue
            r0, c0, r1, c1 = bbox
            seeds = [
                (int(r), int(c))
                for r, c in np.argwhere(grid[r0 + 1 : r1, c0 + 1 : c1] == SEED)
            ]
            seeds = [(r + r0 + 1, c + c0 + 1) for r, c in seeds]
            if seeds:
                regions.append(
                    {
                        "bbox": bbox,
                        "color": color,
                        "seeds": sorted(seeds, key=lambda pos: (pos[0], pos[1])),
                    }
                )
    return regions


def _paint_cross_from_input(
    out: np.ndarray,
    inp: np.ndarray,
    center: tuple[int, int],
    arm_len: int,
    color: int,
) -> None:
    h, w = inp.shape
    r0, c0 = center
    out[r0, c0] = color
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        for step in range(1, arm_len + 1):
            r = r0 + dr * step
            c = c0 + dc * step
            if not (0 <= r < h and 0 <= c < w):
                break
            if int(inp[r, c]) not in (BG, SEED):
                break
            out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each green seed inside a rectangular framed region with a cross whose arm length
    equals the number of seeds in that region and whose color matches the frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    seed_records: list[tuple[int, int, int, int]] = []
    for region in _extract_regions(g):
        seeds = region["seeds"]
        assert isinstance(seeds, list) and seeds, "every region should contain at least one seed"
        count = len(seeds)
        color = int(region["color"])
        for r, c in seeds:
            seed_records.append((r, c, count, color))

    for r, c, count, color in sorted(seed_records, key=lambda item: (item[0], item[1])):
        _paint_cross_from_input(out, g, (r, c), count, color)

    return out


def _boxes_conflict(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    pad: int = FRAME_PADDING,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _pick_origin(limit: int, span: int) -> int:
    if limit == span:
        return 0
    if random.random() < EDGE_TOUCH_PROB:
        return random.choice([0, limit - span])
    return random.randint(0, limit - span)


def _sample_bbox(
    height: int,
    width: int,
    existing: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    for _ in range(FRAME_TRIES):
        outer_h = random.randint(FRAME_OUTER_MIN, min(FRAME_OUTER_MAX, height))
        outer_w = random.randint(FRAME_OUTER_MIN, min(FRAME_OUTER_MAX, width))
        r0 = _pick_origin(height, outer_h)
        c0 = _pick_origin(width, outer_w)
        bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
        if any(_boxes_conflict(bbox, other) for other in existing):
            continue
        return bbox
    return None


def _seed_independent_set_capacity(inner_h: int, inner_w: int) -> int:
    return ((inner_h + 1) // 2) * ((inner_w + 1) // 2)


def _non_adjacent_positions(
    cells: list[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]] | None:
    random.shuffle(cells)
    chosen: list[tuple[int, int]] = []
    for r, c in cells:
        if any(abs(r - rr) <= 1 and abs(c - cc) <= 1 for rr, cc in chosen):
            continue
        chosen.append((r, c))
        if len(chosen) == count:
            return chosen
    return None


def _aligned_seed_layout(
    r0: int,
    c0: int,
    inner_h: int,
    inner_w: int,
    count: int,
) -> list[tuple[int, int]] | None:
    layouts: list[list[tuple[int, int]]] = []
    if inner_w >= 2 * count - 1:
        for local_r in range(inner_h):
            cols = list(range(inner_w))
            random.shuffle(cols)
            cells = [(r0 + local_r, c0 + c) for c in cols]
            picked = _non_adjacent_positions(cells, count)
            if picked is not None:
                layouts.append(picked)
    if inner_h >= 2 * count - 1:
        for local_c in range(inner_w):
            rows = list(range(inner_h))
            random.shuffle(rows)
            cells = [(r0 + r, c0 + local_c) for r in rows]
            picked = _non_adjacent_positions(cells, count)
            if picked is not None:
                layouts.append(picked)
    if not layouts:
        return None
    return random.choice(layouts)


def _sample_seeds_for_region(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    max_count = min(5, _seed_independent_set_capacity(inner_h, inner_w), max(1, min(inner_h, inner_w)))
    counts = [n for n in range(1, max_count + 1)]
    weights = [4, 4, 3, 2, 1][: len(counts)]
    count = random.choices(counts, weights=weights, k=1)[0]

    inner_r0 = r0 + 1
    inner_c0 = c0 + 1
    if count > 1 and random.random() < ALIGNMENT_PROB:
        aligned = _aligned_seed_layout(inner_r0, inner_c0, inner_h, inner_w, count)
        if aligned is not None:
            return sorted(aligned, key=lambda pos: (pos[0], pos[1]))

    all_cells = [(r, c) for r in range(inner_r0, r1) for c in range(inner_c0, c1)]
    picked = _non_adjacent_positions(all_cells, count)
    if picked is None:
        return None
    return sorted(picked, key=lambda pos: (pos[0], pos[1]))


def _render_input(
    shape: tuple[int, int],
    regions: list[dict[str, object]],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for region in regions:
        r0, c0, r1, c1 = region["bbox"]
        color = int(region["color"])
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color
        for r, c in region["seeds"]:
            grid[r, c] = SEED
    return grid


def _render_expected(
    shape: tuple[int, int],
    regions: list[dict[str, object]],
) -> np.ndarray:
    inp = _render_input(shape, regions)
    out = inp.copy()
    seed_records: list[tuple[int, int, int, int, tuple[int, int, int, int]]] = []
    for region in regions:
        bbox = region["bbox"]
        color = int(region["color"])
        seeds = sorted(region["seeds"], key=lambda pos: (pos[0], pos[1]))
        count = len(seeds)
        for r, c in seeds:
            seed_records.append((r, c, count, color, bbox))

    for r, c, count, color, bbox in sorted(seed_records, key=lambda item: (item[0], item[1])):
        r0, c0, r1, c1 = bbox
        out[r, c] = color
        for rr in range(max(r0 + 1, r - count), r):
            out[rr, c] = color
        for rr in range(r + 1, min(r1 - 1, r + count) + 1):
            out[rr, c] = color
        for cc in range(max(c0 + 1, c - count), c):
            out[r, cc] = color
        for cc in range(c + 1, min(c1 - 1, c + count) + 1):
            out[r, cc] = color
    return out


def _sample_regions(height: int, width: int) -> list[dict[str, object]] | None:
    area = height * width
    max_regions = 1 + (area >= 220) + (area >= 360) + (area >= 520)
    target_count = random.randint(1, max_regions)

    bboxes: list[tuple[int, int, int, int]] = []
    for _ in range(target_count):
        bbox = _sample_bbox(height, width, bboxes)
        if bbox is None:
            return None
        bboxes.append(bbox)

    colors = random.sample(FRAME_COLORS, len(bboxes))
    regions: list[dict[str, object]] = []
    for bbox, color in zip(bboxes, colors):
        seeds = _sample_seeds_for_region(bbox)
        if seeds is None:
            return None
        regions.append({"bbox": bbox, "color": int(color), "seeds": seeds})
    return regions


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    regions = _sample_regions(height, width)
    if regions is None:
        return None

    inp = _render_input((height, width), regions)
    expected = _render_expected((height, width), regions)

    if not any(len(region["seeds"]) > 1 for region in regions):
        return None
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
