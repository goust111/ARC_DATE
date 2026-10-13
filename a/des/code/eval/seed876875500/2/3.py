from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


FRAME_COLOR = 5
GRID_H_RANGE = (20, 30)
GRID_W_RANGE = (20, 30)
REGION_COUNT_NO_OVERLAP = (2, 4)
REGION_COUNT_OVERLAP = (2, 3)
INTERIOR_H_RANGE = (3, 6)
INTERIOR_W_RANGE = (3, 7)
MAX_RING_LAYERS = 3
PLACE_RETRIES = 100
GEN_RETRIES = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _same_color_components(grid: np.ndarray, bg: int) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, list[tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append((color, pts))
    return comps


def _frame_bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    r1 = max(r for r, _ in pts)
    c1 = max(c for _, c in pts)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    expected = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    if len(pts) != expected:
        return None
    pset = set(pts)
    for c in range(c0, c1 + 1):
        if (r0, c) not in pset or (r1, c) not in pset:
            return None
    for r in range(r0, r1 + 1):
        if (r, c0) not in pset or (r, c1) not in pset:
            return None
    return r0, c0, r1, c1


def _ring_cells(r0: int, c0: int, r1: int, c1: int, layer: int, h: int, w: int) -> list[tuple[int, int]]:
    rr0 = r0 - layer
    cc0 = c0 - layer
    rr1 = r1 + layer
    cc1 = c1 + layer
    cells: list[tuple[int, int]] = []
    for c in range(cc0, cc1 + 1):
        if 0 <= rr0 < h and 0 <= c < w:
            cells.append((rr0, c))
        if rr1 != rr0 and 0 <= rr1 < h and 0 <= c < w:
            cells.append((rr1, c))
    for r in range(rr0 + 1, rr1):
        if 0 <= r < h and 0 <= cc0 < w:
            cells.append((r, cc0))
        if cc1 != cc0 and 0 <= r < h and 0 <= cc1 < w:
            cells.append((r, cc1))
    return cells


def _seed_summary(interior: np.ndarray, bg: int) -> tuple[list[int], int]:
    counts: Counter[int] = Counter()
    first_seen: dict[int, int] = {}
    idx = 0
    for r in range(interior.shape[0]):
        for c in range(interior.shape[1]):
            color = int(interior[r, c])
            if color == bg:
                idx += 1
                continue
            counts[color] += 1
            if color not in first_seen:
                first_seen[color] = idx
            idx += 1
    ordered = sorted(counts, key=lambda color: (-counts[color], first_seen[color]))
    return ordered, int(sum(counts.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow outward concentric rectangular rings from each framed region.

    Input structure:
    - The most common color is background.
    - One-cell-thick monochrome rectangular frames enclose scattered seed pixels.
    - Each region interior contains background plus 2-3 seed colors.

    Rule:
    - For each frame, count distinct seed colors in the interior.
    - Order those colors by seed count descending; ties use first row-major appearance.
    - Grow one outer rectangular ring per distinct color, using that ordered color list.
    - If rings overlap, the region with more total seeds wins the overlap.
    - Original frames and seed pixels remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    bg = _most_common_color(g)

    frames = []
    for color, pts in _same_color_components(g, bg):
        box = _frame_bbox(pts)
        if box is None:
            continue
        r0, c0, r1, c1 = box
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        ordered_colors, seed_total = _seed_summary(interior, bg)
        assert len(ordered_colors) >= 2, "each frame must contain at least two seed colors"
        frames.append(
            {
                "frame_color": int(color),
                "bbox": (r0, c0, r1, c1),
                "ordered_colors": ordered_colors,
                "seed_total": seed_total,
            }
        )

    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["bbox"][2], item["bbox"][3]))
    out = g.copy()
    best_owner: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}

    for order, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]
        priority = (int(frame["seed_total"]), -order)
        for layer, color in enumerate(frame["ordered_colors"], start=1):
            for rr, cc in _ring_cells(r0, c0, r1, c1, layer, h, w):
                if int(g[rr, cc]) != bg:
                    continue
                prev = best_owner.get((rr, cc))
                if prev is None or priority > prev[0]:
                    best_owner[(rr, cc)] = (priority, int(color))

    for (r, c), (_priority, color) in best_owner.items():
        out[r, c] = color
    return out


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_box(box: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    return box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad


def _count_options(distinct: int, tie_case: bool, scale: str | None) -> list[list[int]]:
    options = {
        (2, False): [[2, 1], [3, 1], [3, 2], [4, 1], [4, 2]],
        (2, True): [[1, 1], [2, 2], [3, 3]],
        (3, False): [[2, 1, 1], [3, 2, 1], [4, 2, 1], [4, 3, 1]],
        (3, True): [[2, 2, 1], [3, 3, 1], [2, 2, 2], [3, 2, 2]],
    }[(distinct, tie_case)]
    if scale == "small":
        options = [counts for counts in options if sum(counts) <= 4]
    elif scale == "large":
        options = [counts for counts in options if sum(counts) >= 5]
    return options


def _choose_counts(distinct: int, tie_case: bool, scale: str | None = None) -> list[int]:
    options = _count_options(distinct, tie_case, scale)
    if not options:
        options = _count_options(distinct, tie_case, None)
    return list(random.choice(options))


def _seed_layout(
    interior_h: int,
    interior_w: int,
    counts: list[int],
    colors: list[int],
) -> list[tuple[int, int, int]]:
    total = sum(counts)
    cells = [(r, c) for r in range(interior_h) for c in range(interior_w)]
    chosen = sorted(random.sample(cells, total))

    order: list[int] = []
    for count in sorted(set(counts), reverse=True):
        idxs = [i for i, value in enumerate(counts) if value == count]
        random.shuffle(idxs)
        order.extend(idxs)

    assigned: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(colors))}
    for pos, color_idx in zip(chosen[: len(colors)], order, strict=False):
        assigned[color_idx].append(pos)

    remaining = chosen[len(colors) :]
    random.shuffle(remaining)
    cursor = 0
    for i, count in enumerate(counts):
        need = count - 1
        assigned[i].extend(remaining[cursor : cursor + need])
        cursor += need

    seeds: list[tuple[int, int, int]] = []
    for i, color in enumerate(colors):
        for r, c in assigned[i]:
            seeds.append((r, c, int(color)))
    return seeds


def _make_region_spec(scale: str | None = None) -> dict:
    distinct = random.choice([2, 3])
    tie_case = random.choice([False, True])
    counts = _choose_counts(distinct, tie_case, scale)
    total = sum(counts)

    dims = [
        (ih, iw)
        for ih in range(INTERIOR_H_RANGE[0], INTERIOR_H_RANGE[1] + 1)
        for iw in range(INTERIOR_W_RANGE[0], INTERIOR_W_RANGE[1] + 1)
        if ih * iw >= total + 2
    ]
    interior_h, interior_w = random.choice(dims)

    palette = [color for color in range(10) if color != FRAME_COLOR]
    bg = 0  # placeholder filtered by caller
    colors = random.sample([color for color in palette if color != bg], distinct)
    seeds = _seed_layout(interior_h, interior_w, counts, colors)
    return {
        "distinct": distinct,
        "tie_case": tie_case,
        "counts": counts,
        "seed_total": total,
        "interior_h": interior_h,
        "interior_w": interior_w,
        "bbox_h": interior_h + 2,
        "bbox_w": interior_w + 2,
        "seeds_local": seeds,
    }


def _with_bg_colors(spec: dict, bg: int) -> dict:
    palette = [color for color in range(10) if color not in {bg, FRAME_COLOR}]
    colors = random.sample(palette, spec["distinct"])
    updated = dict(spec)
    updated["seed_colors"] = colors
    updated["seeds_local"] = _seed_layout(spec["interior_h"], spec["interior_w"], spec["counts"], colors)
    return updated


def _spec_box(spec: dict) -> tuple[int, int, int, int]:
    return spec["r0"], spec["c0"], spec["r0"] + spec["bbox_h"] - 1, spec["c0"] + spec["bbox_w"] - 1


def _union_ring_cells(spec: dict, h: int, w: int) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = _spec_box(spec)
    cells: set[tuple[int, int]] = set()
    for layer in range(1, spec["distinct"] + 1):
        cells.update(_ring_cells(r0, c0, r1, c1, layer, h, w))
    return cells


def _can_place_far(spec: dict, placed: list[dict], pad: int) -> bool:
    box = _spec_box(spec)
    for other in placed:
        if _boxes_intersect(_expand_box(box, pad), _expand_box(_spec_box(other), pad)):
            return False
    return True


def _place_overlap_pair(specs: list[dict], h: int, w: int) -> bool:
    left, right = specs[0], specs[1]
    orient = random.choice(["h", "v"])
    max_gap = left["distinct"] + right["distinct"] - 1
    gap = random.randint(1, max_gap)

    if orient == "h":
        total_h = max(left["bbox_h"], right["bbox_h"])
        total_w = left["bbox_w"] + gap + right["bbox_w"]
        if total_h > h or total_w > w:
            return False
        top = random.randint(0, h - total_h)
        c0 = random.randint(0, w - total_w)
        left["r0"] = top
        left["c0"] = c0
        right["r0"] = top
        right["c0"] = c0 + left["bbox_w"] + gap
    else:
        total_h = left["bbox_h"] + gap + right["bbox_h"]
        total_w = max(left["bbox_w"], right["bbox_w"])
        if total_h > h or total_w > w:
            return False
        r0 = random.randint(0, h - total_h)
        left_col = random.randint(0, w - total_w)
        left["r0"] = r0
        left["c0"] = left_col
        right["r0"] = r0 + left["bbox_h"] + gap
        right["c0"] = left_col
    return True


def _place_far_frames(specs: list[dict], start_idx: int, h: int, w: int) -> bool:
    placed = specs[:start_idx]
    for idx in range(start_idx, len(specs)):
        spec = specs[idx]
        found = False
        for _ in range(PLACE_RETRIES):
            spec["r0"] = random.randint(0, h - spec["bbox_h"])
            spec["c0"] = random.randint(0, w - spec["bbox_w"])
            if _can_place_far(spec, placed, MAX_RING_LAYERS):
                placed.append(spec)
                found = True
                break
        if not found:
            return False
    return True


def _draw_input(h: int, w: int, bg: int, specs: list[dict]) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = _spec_box(spec)
        grid[r0, c0 : c1 + 1] = FRAME_COLOR
        grid[r1, c0 : c1 + 1] = FRAME_COLOR
        grid[r0 : r1 + 1, c0] = FRAME_COLOR
        grid[r0 : r1 + 1, c1] = FRAME_COLOR
        for lr, lc, color in spec["seeds_local"]:
            grid[r0 + 1 + lr, c0 + 1 + lc] = int(color)
    return grid


def _intended_overlap_ok(specs: list[dict], h: int, w: int, overlap_case: bool) -> bool:
    unions = [_union_ring_cells(spec, h, w) for spec in specs]
    pair_hits = set()
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            if unions[i] & unions[j]:
                pair_hits.add((i, j))
    if overlap_case:
        return pair_hits == {(0, 1)}
    return not pair_hits


def _frame_detectable(inp: np.ndarray) -> bool:
    bg = _most_common_color(inp)
    count = 0
    for _color, pts in _same_color_components(inp, bg):
        if _frame_bbox(pts) is not None:
            count += 1
    return count >= 2


def _build_example() -> np.ndarray | None:
    overlap_case = random.choice([False, True])
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    bg = random.choice([color for color in range(5) if color != FRAME_COLOR])

    if overlap_case:
        n_regions = random.randint(*REGION_COUNT_OVERLAP)
        big = _make_region_spec(scale="large")
        small = _make_region_spec(scale="small")
        if big["seed_total"] <= small["seed_total"]:
            return None
        primary = [big, small]
        if random.choice([False, True]):
            primary.reverse()
        if primary[0]["seed_total"] == primary[1]["seed_total"]:
            return None
        specs = primary + [_make_region_spec() for _ in range(n_regions - 2)]
    else:
        n_regions = random.randint(*REGION_COUNT_NO_OVERLAP)
        specs = [_make_region_spec() for _ in range(n_regions)]

    specs = [_with_bg_colors(spec, bg) for spec in specs]
    if overlap_case and specs[0]["seed_total"] == specs[1]["seed_total"]:
        return None

    if overlap_case:
        if not _place_overlap_pair(specs, h, w):
            return None
        if len(specs) > 2 and not _place_far_frames(specs, 2, h, w):
            return None
    else:
        if not _place_far_frames(specs, 0, h, w):
            return None

    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            if _boxes_intersect(_expand_box(_spec_box(specs[i]), 1), _expand_box(_spec_box(specs[j]), 1)):
                return None

    if not _intended_overlap_ok(specs, h, w, overlap_case):
        return None

    inp = _draw_input(h, w, bg, specs)
    if not _frame_detectable(inp):
        return None
    try:
        out = main(inp)
    except Exception:
        return None
    if out.shape != inp.shape:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_example()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
