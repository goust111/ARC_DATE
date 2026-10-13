from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
GEN_ATTEMPTS = 500
REGION_COUNT_CHOICES = (1, 1, 2, 2, 3)
SEED_COUNT_CHOICES = (1, 2, 3, 4)
EDGE_MARGIN = 1
BASE_GAP_CHOICES = (2, 3, 4)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _round_half_up(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_edge(comp: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)


def _infer_border_color(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    counts = Counter(int(v) for v in grid.ravel() if int(v) != int(bg))
    best: tuple[tuple[int, int, int], int] | None = None
    for color, count in counts.items():
        enclosed = 0
        bg_inside = 0
        for comp in _components(grid != int(color)):
            if _touches_edge(comp, h, w):
                continue
            bg_cells = sum(1 for r, c in comp if int(grid[r, c]) == int(bg))
            if bg_cells == 0:
                continue
            enclosed += 1
            bg_inside += bg_cells
        if enclosed == 0:
            continue
        key = (int(enclosed), int(bg_inside), int(count))
        if best is None or key > best[0]:
            best = (key, int(color))
    if best is not None:
        return int(best[1])
    return int(counts.most_common(1)[0][0])


def _region_center(
    bg_cells: list[tuple[int, int]], region_set: set[tuple[int, int]]
) -> tuple[int, int]:
    mean_r = sum(r for r, _ in bg_cells) / len(bg_cells)
    mean_c = sum(c for _, c in bg_cells) / len(bg_cells)
    center = (_round_half_up(mean_r), _round_half_up(mean_c))
    if center in region_set:
        return center
    return min(
        bg_cells,
        key=lambda p: (
            (float(p[0]) - float(mean_r)) ** 2 + (float(p[1]) - float(mean_c)) ** 2,
            int(p[0]),
            int(p[1]),
        ),
    )


def _enclosed_regions(
    grid: np.ndarray, bg: int, border: int
) -> list[tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int, int]]]]:
    h, w = grid.shape
    regions = []
    for comp in _components(grid != int(border)):
        if _touches_edge(comp, h, w):
            continue
        bg_cells = [(int(r), int(c)) for r, c in comp if int(grid[r, c]) == int(bg)]
        seeds = [(int(r), int(c), int(grid[r, c])) for r, c in comp if int(grid[r, c]) != int(bg)]
        if bg_cells and seeds:
            regions.append((comp, bg_cells, seeds))
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint each enclosed region as centered Chebyshev rings whose depth and color order come from the interior seeds.

    Input structure:
    - The background is the dominant color, typically black.
    - One shared non-background color draws several closed polygonal borders.
    - Each enclosed border region contains isolated colored seed pixels on an otherwise background interior.

    Rule:
    - Infer the border color and find every enclosed non-border region.
    - For each region, count its seed pixels, then compute the visual center by averaging the rows and columns of the region's background cells and rounding to the nearest cell.
    - Sort the seeds by increasing distance from that center, using row-major order to break distance ties.
    - Clear the region interior back to background, then fill Chebyshev layer 0 (the center cell), layer 1 (its 8-neighborhood cells still inside the region), and so on up to layer N-1 with the sorted seed colors, where N is the seed count.

    Key insight:
    - The seeds encode two things at once: how many centered layers to draw, and which seed color goes on each layer according to the seed's own distance from the region center.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    border = _infer_border_color(g, bg)
    out = g.copy()

    for region, bg_cells, seeds in _enclosed_regions(g, bg, border):
        region_set = {(int(r), int(c)) for r, c in region}
        center = _region_center(bg_cells, region_set)
        ordered = sorted(
            seeds,
            key=lambda item: (_chebyshev((int(item[0]), int(item[1])), center), int(item[0]), int(item[1])),
        )

        for r, c in region:
            out[int(r), int(c)] = int(bg)

        for layer, (_r, _c, color) in enumerate(ordered):
            for rr, cc in region:
                if _chebyshev((int(rr), int(cc)), center) == int(layer):
                    out[int(rr), int(cc)] = int(color)

    return out


def _split_nonnegative(total: int, parts: int) -> list[int]:
    buckets = [0] * int(parts)
    for _ in range(int(total)):
        buckets[random.randrange(int(parts))] += 1
    return buckets


def _normalize(cells: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(int(r - min_r), int(c - min_c)) for r, c in cells}, (int(min_r), int(min_c))


def _core_rect(seed_count: int) -> set[tuple[int, int]]:
    half_h = random.randint(max(2, seed_count), seed_count + 2)
    half_w = random.randint(max(2, seed_count), seed_count + 2)
    return {
        (int(r), int(c))
        for r in range(-half_h, half_h + 1)
        for c in range(-half_w, half_w + 1)
    }


def _add_protrusions(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    updated = set(cells)
    sides = ["top", "bottom", "left", "right"]
    random.shuffle(sides)
    for side in sides:
        if random.random() < 0.35:
            continue
        if side in ("top", "bottom"):
            span = random.randint(2, max(2, c1 - c0))
            start = random.randint(c0, c1 - span + 1)
            depth = random.randint(1, 2)
            row_range = range(r0 - depth, r0) if side == "top" else range(r1 + 1, r1 + depth + 1)
            for r in row_range:
                for c in range(start, start + span):
                    updated.add((int(r), int(c)))
        else:
            span = random.randint(2, max(2, r1 - r0))
            start = random.randint(r0, r1 - span + 1)
            depth = random.randint(1, 2)
            col_range = range(c0 - depth, c0) if side == "left" else range(c1 + 1, c1 + depth + 1)
            for r in range(start, start + span):
                for c in col_range:
                    updated.add((int(r), int(c)))
    return updated


def _make_interior(seed_count: int) -> set[tuple[int, int]]:
    cells = _core_rect(seed_count)
    cells = _add_protrusions(cells)
    norm_cells, _ = _normalize(cells)
    return norm_cells


def _border_shell(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    border: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr, nc in _neighbors8(int(r), int(c)):
            if (int(nr), int(nc)) not in cells:
                border.add((int(nr), int(nc)))
    return border


def _center_for_cells(
    cells: set[tuple[int, int]], seeds: set[tuple[int, int]]
) -> tuple[int, int] | None:
    bg_cells = [cell for cell in cells if cell not in seeds]
    if not bg_cells:
        return None
    return _region_center(bg_cells, set(cells))


def _deep_cells(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    region = set(cells)
    out = []
    for r, c in sorted(region):
        if all((int(nr), int(nc)) in region for nr, nc in _neighbors4(int(r), int(c))):
            out.append((int(r), int(c)))
    return out


def _pick_seed_positions(cells: set[tuple[int, int]], seed_count: int) -> list[tuple[int, int]] | None:
    region = set(cells)
    approx_center = _center_for_cells(region, set())
    if approx_center is None:
        return None
    core_candidates = [
        cell for cell in _deep_cells(region) if _chebyshev(cell, approx_center) >= int(seed_count)
    ]
    if len(core_candidates) < int(seed_count):
        return None

    for _ in range(200):
        random.shuffle(core_candidates)
        chosen: list[tuple[int, int]] = []
        for cell in core_candidates:
            if all(abs(cell[0] - other[0]) + abs(cell[1] - other[1]) > 1 for other in chosen):
                chosen.append((int(cell[0]), int(cell[1])))
            if len(chosen) == int(seed_count):
                break
        if len(chosen) != int(seed_count):
            continue

        seed_set = set(chosen)
        center = _center_for_cells(region, seed_set)
        if center is None or center in seed_set:
            continue
        if any(_chebyshev(seed, center) < int(seed_count) for seed in chosen):
            continue
        if any(not any(_chebyshev(cell, center) == layer for cell in region) for layer in range(int(seed_count))):
            continue
        return chosen
    return None


def _region_patch(seed_count: int, border_color: int) -> np.ndarray | None:
    for _ in range(200):
        cells = _make_interior(seed_count)
        seeds = _pick_seed_positions(cells, seed_count)
        if seeds is None:
            continue

        border = _border_shell(cells)
        if not border:
            continue
        all_cells = set(cells) | set(border)
        norm_all, offset = _normalize(all_cells)
        off_r, off_c = offset
        norm_cells = {(r - off_r, c - off_c) for r, c in cells}
        norm_border = {(r - off_r, c - off_c) for r, c in border}
        norm_seeds = [(r - off_r, c - off_c) for r, c in seeds]

        h = max(r for r, _ in norm_all) + 1
        w = max(c for _, c in norm_all) + 1
        if h > 14 or w > 14:
            continue

        palette = [c for c in range(1, 10) if c != int(border_color)]
        seed_colors = random.sample(palette, int(seed_count))
        patch = np.full((int(h), int(w)), int(BG), dtype=int)
        for r, c in norm_border:
            patch[int(r), int(c)] = int(border_color)
        for (r, c), color in zip(norm_seeds, seed_colors):
            patch[int(r), int(c)] = int(color)

        if np.array_equal(main(patch), patch):
            continue
        return patch
    return None


def _pack_rows(patches: list[np.ndarray]) -> np.ndarray | None:
    row_count = 1 if len(patches) == 1 else random.choice((1, 2))
    gap = random.choice(BASE_GAP_CHOICES)
    order = patches[:]
    random.shuffle(order)
    rows: list[list[np.ndarray]] = [[] for _ in range(int(row_count))]
    row_widths = [0] * int(row_count)
    row_heights = [0] * int(row_count)

    for patch in order:
        idx = min(range(int(row_count)), key=lambda i: (row_widths[i], row_heights[i], i))
        rows[idx].append(patch)
        row_widths[idx] += int(patch.shape[1]) + (int(gap) if len(rows[idx]) > 1 else 0)
        row_heights[idx] = max(int(row_heights[idx]), int(patch.shape[0]))

    required_h = sum(int(h) for h in row_heights if h > 0) + int(gap) * max(0, int(row_count) - 1) + 2 * EDGE_MARGIN
    required_w = max(row_widths) + 2 * EDGE_MARGIN
    if required_h > 30 or required_w > 30:
        return None

    extra_h = random.randint(0, min(30 - int(required_h), 4))
    extra_w = random.randint(0, min(30 - int(required_w), 6))
    height = int(required_h + extra_h)
    width = int(required_w + extra_w)
    grid = np.full((int(height), int(width)), int(BG), dtype=int)

    row_sizes = [h for h in row_heights if h > 0]
    if not row_sizes:
        return None
    v_slack = _split_nonnegative(int(height - required_h), len(row_sizes) + 1)
    cur_r = EDGE_MARGIN + int(v_slack[0])

    active_row = 0
    for row, row_h in zip(rows, row_heights):
        if not row:
            continue
        row_required_w = sum(int(p.shape[1]) for p in row) + int(gap) * (len(row) - 1) + 2 * EDGE_MARGIN
        h_slack = _split_nonnegative(int(width - row_required_w), len(row) + 1)
        cur_c = EDGE_MARGIN + int(h_slack[0])

        for i, patch in enumerate(row):
            ph, pw = map(int, patch.shape)
            top = int(cur_r + random.randint(0, int(row_h - ph)))
            view = grid[int(top) : int(top + ph), int(cur_c) : int(cur_c + pw)]
            mask = patch != int(BG)
            if bool((view[mask] != int(BG)).any()):
                return None
            view[mask] = patch[mask]
            cur_c += int(pw)
            if i < len(row) - 1:
                cur_c += int(gap) + int(h_slack[i + 1])

        cur_r += int(row_h)
        active_row += 1
        if active_row < len(row_sizes):
            cur_r += int(gap) + int(v_slack[active_row])

    return grid


def _gen_attempt() -> np.ndarray | None:
    border_color = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
    region_count = random.choice(REGION_COUNT_CHOICES)
    patches: list[np.ndarray] = []
    for _ in range(int(region_count)):
        seed_count = random.choice(SEED_COUNT_CHOICES)
        patch = _region_patch(int(seed_count), int(border_color))
        if patch is None:
            return None
        patches.append(patch)

    grid = _pack_rows(patches)
    if grid is None:
        return None
    if _get_bg(grid) != int(BG):
        return None
    if _infer_border_color(grid, BG) != int(border_color):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
