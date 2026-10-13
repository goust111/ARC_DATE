import random

import numpy as np


GRAY = 5
BLACK = 0
RED = 2
MAROON = 9
BASE_COLORS = (1, 3, 4, 6, 7, 8)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GENERATE_RETRIES = 400


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _normalize_shape(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _shape_size(cells: set[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return max(rows) + 1, max(cols) + 1


def _shift_shape(cells: set[tuple[int, int]], top: int, left: int) -> set[tuple[int, int]]:
    return {(top + r, left + c) for r, c in cells}


def _random_shape(
    rng: random.Random,
    max_h: int,
    max_w: int,
    area_min: int = 1,
    area_max: int = 4,
    min_span_h: int = 1,
    min_span_w: int = 1,
) -> set[tuple[int, int]]:
    max_h = max(1, int(max_h))
    max_w = max(1, int(max_w))
    area_max = max(1, min(int(area_max), max_h * max_w))
    area_min = max(1, min(int(area_min), area_max))
    for _ in range(120):
        box_h = rng.randint(1, max_h)
        box_w = rng.randint(1, max_w)
        local_area_max = min(area_max, box_h * box_w)
        local_area_min = min(area_min, local_area_max)
        area = rng.randint(local_area_min, local_area_max)
        cells = {(rng.randrange(box_h), rng.randrange(box_w))}
        while len(cells) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < box_h and 0 <= cc < box_w and (rr, cc) not in cells:
                        frontier.add((rr, cc))
            if not frontier:
                break
            cells.add(rng.choice(sorted(frontier)))
        if len(cells) != area:
            continue
        cells = _normalize_shape(cells)
        shape_h, shape_w = _shape_size(cells)
        if shape_h > max_h or shape_w > max_w:
            continue
        if shape_h < min_span_h or shape_w < min_span_w:
            continue
        return cells
    return {(0, 0)}


def _find_separators(grid: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    sep_rows = [r for r in range(h) if int(np.count_nonzero(grid[r] == GRAY)) >= w - 3]
    sep_cols = [c for c in range(w) if int(np.count_nonzero(grid[:, c] == GRAY)) >= h - 3]
    return sep_rows, sep_cols


def _parse_regions(grid: np.ndarray) -> tuple[list[dict], np.ndarray, set[int], set[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sep_rows, sep_cols = _find_separators(g)
    row_bands = [(sep_rows[i] + 1, sep_rows[i + 1] - 1) for i in range(len(sep_rows) - 1) if sep_rows[i + 1] - sep_rows[i] > 1]
    col_bands = [(sep_cols[i] + 1, sep_cols[i + 1] - 1) for i in range(len(sep_cols) - 1) if sep_cols[i + 1] - sep_cols[i] > 1]
    region_map = np.full((h, w), -1, dtype=int)
    regions: list[dict] = []

    def row_gaps(row: int, c0: int, c1: int) -> list[dict]:
        mask = g[row, c0 : c1 + 1] != GRAY
        gaps = []
        for s, e in _runs(mask):
            gaps.append({"width": e - s + 1, "positions": tuple(range(c0 + s, c0 + e + 1))})
        return gaps

    def col_gaps(col: int, r0: int, r1: int) -> list[dict]:
        mask = g[r0 : r1 + 1, col] != GRAY
        gaps = []
        for s, e in _runs(mask):
            gaps.append({"width": e - s + 1, "positions": tuple(range(r0 + s, r0 + e + 1))})
        return gaps

    for ri, (r0, r1) in enumerate(row_bands):
        for ci, (c0, c1) in enumerate(col_bands):
            rid = len(regions)
            region_map[r0 : r1 + 1, c0 : c1 + 1] = rid
            gaps = []
            for gap in row_gaps(sep_rows[ri], c0, c1):
                gaps.append({"dir": (-1, 0), "dist": gap["width"], "open": set(gap["positions"]), "side": "top"})
            for gap in row_gaps(sep_rows[ri + 1], c0, c1):
                gaps.append({"dir": (1, 0), "dist": gap["width"], "open": set(gap["positions"]), "side": "bottom"})
            for gap in col_gaps(sep_cols[ci], r0, r1):
                gaps.append({"dir": (0, -1), "dist": gap["width"], "open": set(gap["positions"]), "side": "left"})
            for gap in col_gaps(sep_cols[ci + 1], r0, r1):
                gaps.append({"dir": (0, 1), "dist": gap["width"], "open": set(gap["positions"]), "side": "right"})
            regions.append(
                {
                    "id": rid,
                    "row": ri,
                    "col": ci,
                    "r0": r0,
                    "r1": r1,
                    "c0": c0,
                    "c1": c1,
                    "gaps": gaps,
                }
            )
    return regions, region_map, set(sep_rows), set(sep_cols)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each black seed toward a nearby gap on its region boundary.

    The gap side selects the translation direction, the gap width selects the
    translation distance, and only pixels aligned with the gap can cross into
    the neighboring region. Crossed pixels become maroon, while collisions with
    another region's original seed become red.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    regions, region_map, sep_rows, sep_cols = _parse_regions(g)
    out = g.copy()
    original_black = g == BLACK
    priority = {MAROON: 1, BLACK: 2, RED: 3}
    paint: dict[tuple[int, int], int] = {}

    def record(r: int, c: int, color: int) -> None:
        prev = paint.get((r, c))
        if prev is None or priority[color] > priority[prev]:
            paint[(r, c)] = color

    for region in regions:
        r0, r1, c0, c1 = region["r0"], region["r1"], region["c0"], region["c1"]
        seed_local = np.argwhere(original_black[r0 : r1 + 1, c0 : c1 + 1])
        if seed_local.size == 0:
            continue
        seeds = [(r0 + int(rr), c0 + int(cc)) for rr, cc in seed_local]
        for gap in region["gaps"]:
            dr, dc = gap["dir"]
            step = int(gap["dist"])
            open_positions = gap["open"]
            for r, c in seeds:
                rr = r + dr * step
                cc = c + dc * step
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if r0 <= rr <= r1 and c0 <= cc <= c1:
                    record(rr, cc, BLACK)
                    continue
                if gap["side"] in {"top", "bottom"}:
                    if c not in open_positions:
                        continue
                else:
                    if r not in open_positions:
                        continue
                if rr in sep_rows or cc in sep_cols:
                    continue
                if int(region_map[rr, cc]) < 0:
                    continue
                color = RED if bool(original_black[rr, cc]) else MAROON
                record(rr, cc, color)

    for (r, c), color in paint.items():
        out[r, c] = color
    return out


def _build_layout(rng: random.Random) -> tuple[np.ndarray, list[dict], dict[tuple[int, int], dict], list[int], list[int]]:
    for _ in range(80):
        n_region_rows, n_region_cols = rng.choice([(2, 2), (2, 3), (3, 2)])
        heights = [rng.randint(4, 7) for _ in range(n_region_rows)]
        widths = [rng.randint(4, 7) for _ in range(n_region_cols)]
        h = sum(heights) + n_region_rows + 1
        w = sum(widths) + n_region_cols + 1
        if h > 26 or w > 26:
            continue

        sep_rows = [0]
        row_bands = []
        for band_h in heights:
            r0 = sep_rows[-1] + 1
            r1 = r0 + band_h - 1
            row_bands.append((r0, r1))
            sep_rows.append(r1 + 1)

        sep_cols = [0]
        col_bands = []
        for band_w in widths:
            c0 = sep_cols[-1] + 1
            c1 = c0 + band_w - 1
            col_bands.append((c0, c1))
            sep_cols.append(c1 + 1)

        grid = np.full((h, w), GRAY, dtype=int)
        colors = rng.sample(list(BASE_COLORS), n_region_rows * n_region_cols)
        regions: list[dict] = []
        region_by_key: dict[tuple[int, int], dict] = {}
        color_idx = 0
        for ri, (r0, r1) in enumerate(row_bands):
            for ci, (c0, c1) in enumerate(col_bands):
                color = colors[color_idx]
                color_idx += 1
                grid[r0 : r1 + 1, c0 : c1 + 1] = color
                region = {
                    "key": (ri, ci),
                    "r0": r0,
                    "r1": r1,
                    "c0": c0,
                    "c1": c1,
                    "color": color,
                }
                regions.append(region)
                region_by_key[(ri, ci)] = region
        return grid, regions, region_by_key, sep_rows, sep_cols
    raise RuntimeError("failed to build layout")


def _select_gap_edges(rng: random.Random, n_region_rows: int, n_region_cols: int) -> list[dict]:
    candidates = []
    for ri in range(n_region_rows - 1):
        for ci in range(n_region_cols):
            candidates.append({"orientation": "h", "a": (ri, ci), "b": (ri + 1, ci), "line": ri + 1})
    for ri in range(n_region_rows):
        for ci in range(n_region_cols - 1):
            candidates.append({"orientation": "v", "a": (ri, ci), "b": (ri, ci + 1), "line": ci + 1})
    rng.shuffle(candidates)

    used_regions: set[tuple[int, int]] = set()
    used_h_lines: set[int] = set()
    used_v_lines: set[int] = set()
    selected: list[dict] = []
    target = rng.randint(1, min(3, len(candidates), (n_region_rows * n_region_cols + 1) // 2))

    for edge in candidates:
        if len(selected) >= target:
            break
        if edge["a"] in used_regions or edge["b"] in used_regions:
            continue
        if edge["orientation"] == "h" and edge["line"] in used_h_lines:
            continue
        if edge["orientation"] == "v" and edge["line"] in used_v_lines:
            continue
        if rng.random() < 0.72 or not selected:
            selected.append(dict(edge))
            used_regions.update((edge["a"], edge["b"]))
            if edge["orientation"] == "h":
                used_h_lines.add(edge["line"])
            else:
                used_v_lines.add(edge["line"])

    if not selected:
        edge = candidates[0]
        selected.append(dict(edge))

    for edge in candidates:
        if len(selected) >= target:
            break
        if edge["a"] in used_regions or edge["b"] in used_regions:
            continue
        if edge["orientation"] == "h" and edge["line"] in used_h_lines:
            continue
        if edge["orientation"] == "v" and edge["line"] in used_v_lines:
            continue
        selected.append(dict(edge))
        used_regions.update((edge["a"], edge["b"]))
        if edge["orientation"] == "h":
            used_h_lines.add(edge["line"])
        else:
            used_v_lines.add(edge["line"])
    return selected


def _place_free_shape(
    rng: random.Random,
    region: dict,
    forbidden: set[tuple[int, int]] | None = None,
) -> set[tuple[int, int]] | None:
    forbidden = forbidden or set()
    height = region["r1"] - region["r0"] + 1
    width = region["c1"] - region["c0"] + 1
    for _ in range(120):
        shape = _random_shape(
            rng,
            max_h=min(3, height),
            max_w=min(3, width),
            area_min=1,
            area_max=min(4, height * width),
        )
        shape_h, shape_w = _shape_size(shape)
        top = rng.randint(region["r0"], region["r1"] - shape_h + 1)
        left = rng.randint(region["c0"], region["c1"] - shape_w + 1)
        cells = _shift_shape(shape, top, left)
        if cells.isdisjoint(forbidden):
            return cells
    return None


def _place_stay_shape(
    rng: random.Random,
    region: dict,
    direction: tuple[int, int],
    step: int,
    forbidden: set[tuple[int, int]] | None = None,
) -> set[tuple[int, int]] | None:
    forbidden = forbidden or set()
    height = region["r1"] - region["r0"] + 1
    width = region["c1"] - region["c0"] + 1
    if direction[0] != 0:
        max_h = min(3, height - step)
        max_w = min(3, width)
    else:
        max_h = min(3, height)
        max_w = min(3, width - step)
    if max_h < 1 or max_w < 1:
        return None
    for _ in range(120):
        shape = _random_shape(
            rng,
            max_h=max_h,
            max_w=max_w,
            area_min=1,
            area_max=min(4, max_h * max_w),
        )
        shape_h, shape_w = _shape_size(shape)
        if direction == (-1, 0):
            top_lo = region["r0"] + step
            top_hi = region["r1"] - shape_h + 1
            left_lo = region["c0"]
            left_hi = region["c1"] - shape_w + 1
        elif direction == (1, 0):
            top_lo = region["r0"]
            top_hi = region["r1"] - shape_h + 1 - step
            left_lo = region["c0"]
            left_hi = region["c1"] - shape_w + 1
        elif direction == (0, -1):
            top_lo = region["r0"]
            top_hi = region["r1"] - shape_h + 1
            left_lo = region["c0"] + step
            left_hi = region["c1"] - shape_w + 1
        else:
            top_lo = region["r0"]
            top_hi = region["r1"] - shape_h + 1
            left_lo = region["c0"]
            left_hi = region["c1"] - shape_w + 1 - step
        if top_lo > top_hi or left_lo > left_hi:
            continue
        top = rng.randint(top_lo, top_hi)
        left = rng.randint(left_lo, left_hi)
        cells = _shift_shape(shape, top, left)
        if cells.isdisjoint(forbidden):
            return cells
    return None


def _place_cross_shape(
    rng: random.Random,
    region: dict,
    direction: tuple[int, int],
    step: int,
    gap_start: int,
    gap_end: int,
) -> set[tuple[int, int]] | None:
    height = region["r1"] - region["r0"] + 1
    width = region["c1"] - region["c0"] + 1
    if direction[0] != 0:
        max_h = min(3, height)
        max_w = min(width, gap_end - gap_start + 1)
        min_span_h = 2 if max_h >= 2 else 1
        min_span_w = 1
    else:
        max_h = min(height, gap_end - gap_start + 1)
        max_w = min(3, width)
        min_span_h = 1
        min_span_w = 2 if max_w >= 2 else 1
    for _ in range(120):
        shape = _random_shape(
            rng,
            max_h=max_h,
            max_w=max_w,
            area_min=1,
            area_max=min(4, max_h * max_w),
            min_span_h=min_span_h,
            min_span_w=min_span_w,
        )
        shape_h, shape_w = _shape_size(shape)
        if direction[0] != 0:
            left_lo = gap_start
            left_hi = gap_end - shape_w + 1
            if left_lo > left_hi:
                continue
            left = rng.randint(left_lo, left_hi)
            top = region["r0"] if direction == (-1, 0) else region["r1"] - shape_h + 1
        else:
            top_lo = gap_start
            top_hi = gap_end - shape_h + 1
            if top_lo > top_hi:
                continue
            top = rng.randint(top_lo, top_hi)
            left = region["c0"] if direction == (0, -1) else region["c1"] - shape_w + 1
        return _shift_shape(shape, top, left)
    return None


def _cross_targets(
    cells: set[tuple[int, int]],
    direction: tuple[int, int],
    step: int,
    gap_start: int,
    gap_end: int,
    neighbor: dict,
) -> set[tuple[int, int]]:
    targets: set[tuple[int, int]] = set()
    dr, dc = direction
    for r, c in cells:
        rr = r + dr * step
        cc = c + dc * step
        if dr != 0 and not (gap_start <= c <= gap_end):
            continue
        if dc != 0 and not (gap_start <= r <= gap_end):
            continue
        if neighbor["r0"] <= rr <= neighbor["r1"] and neighbor["c0"] <= cc <= neighbor["c1"]:
            targets.add((rr, cc))
    return targets


def _place_overlap_pair(
    rng: random.Random,
    region_a: dict,
    region_b: dict,
    orientation: str,
    step: int,
    gap_start: int,
    gap_end: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    if step < 2:
        return None
    if orientation == "h":
        max_h = min(step - 1, region_a["r1"] - region_a["r0"] + 1, region_b["r1"] - region_b["r0"] + 1)
        max_w = min(step, gap_end - gap_start + 1, region_a["c1"] - region_a["c0"] + 1, region_b["c1"] - region_b["c0"] + 1)
        if max_h < 1 or max_w < 1:
            return None
        for _ in range(120):
            shape = _random_shape(rng, max_h=max_h, max_w=max_w, area_min=1, area_max=min(4, max_h * max_w))
            shape_h, shape_w = _shape_size(shape)
            left_lo = gap_start
            left_hi = gap_end - shape_w + 1
            top_lo = max(region_a["r0"], region_b["r0"] - step)
            top_hi = min(region_a["r1"] - shape_h + 1, region_b["r1"] - shape_h + 1 - step)
            if left_lo > left_hi or top_lo > top_hi:
                continue
            left = rng.randint(left_lo, left_hi)
            top = rng.randint(top_lo, top_hi)
            a_cells = _shift_shape(shape, top, left)
            b_cells = _shift_shape(shape, top + step, left)
            return a_cells, b_cells
    else:
        max_h = min(step, gap_end - gap_start + 1, region_a["r1"] - region_a["r0"] + 1, region_b["r1"] - region_b["r0"] + 1)
        max_w = min(step - 1, region_a["c1"] - region_a["c0"] + 1, region_b["c1"] - region_b["c0"] + 1)
        if max_h < 1 or max_w < 1:
            return None
        for _ in range(120):
            shape = _random_shape(rng, max_h=max_h, max_w=max_w, area_min=1, area_max=min(4, max_h * max_w))
            shape_h, shape_w = _shape_size(shape)
            top_lo = gap_start
            top_hi = gap_end - shape_h + 1
            left_lo = max(region_a["c0"], region_b["c0"] - step)
            left_hi = min(region_a["c1"] - shape_w + 1, region_b["c1"] - shape_w + 1 - step)
            if top_lo > top_hi or left_lo > left_hi:
                continue
            top = rng.randint(top_lo, top_hi)
            left = rng.randint(left_lo, left_hi)
            a_cells = _shift_shape(shape, top, left)
            b_cells = _shift_shape(shape, top, left + step)
            return a_cells, b_cells
    return None


def _draw_gap(grid: np.ndarray, edge: dict, region_a: dict, region_b: dict, sep_rows: list[int], sep_cols: list[int], rng: random.Random) -> None:
    fill_color = rng.choice([region_a["color"], region_b["color"]])
    gap_start = int(edge["gap_start"])
    gap_end = gap_start + int(edge["step"]) - 1
    if edge["orientation"] == "h":
        row = sep_rows[int(edge["line"])]
        grid[row, gap_start : gap_end + 1] = fill_color
    else:
        col = sep_cols[int(edge["line"])]
        grid[gap_start : gap_end + 1, col] = fill_color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid, regions, region_by_key, sep_rows, sep_cols = _build_layout(rng)
    n_region_rows = max(key[0] for key in region_by_key) + 1
    n_region_cols = max(key[1] for key in region_by_key) + 1
    selected_edges = _select_gap_edges(rng, n_region_rows, n_region_cols)
    seeds: dict[tuple[int, int], set[tuple[int, int]]] = {}
    used_modes: set[str] = set()

    for edge in selected_edges:
        region_a = region_by_key[edge["a"]]
        region_b = region_by_key[edge["b"]]
        if edge["orientation"] == "h":
            band_lo, band_hi = region_a["c0"], region_a["c1"]
        else:
            band_lo, band_hi = region_a["r0"], region_a["r1"]
        max_step = min(3, band_hi - band_lo + 1)
        step = rng.randint(1, max_step)
        gap_start = rng.randint(band_lo, band_hi - step + 1)
        gap_end = gap_start + step - 1
        edge["step"] = step
        edge["gap_start"] = gap_start
        direction_a = (1, 0) if edge["orientation"] == "h" else (0, 1)
        direction_b = (-1, 0) if edge["orientation"] == "h" else (0, -1)

        feasible_modes = ["stay"]
        if step >= 2:
            feasible_modes.append("cross")
            feasible_modes.append("overlap")
        preferred = [mode for mode in ("stay", "cross", "overlap") if mode in feasible_modes and mode not in used_modes]
        mode = rng.choice(preferred if preferred else feasible_modes)

        placed = None
        if mode == "overlap":
            placed = _place_overlap_pair(rng, region_a, region_b, edge["orientation"], step, gap_start, gap_end)
        elif mode == "cross":
            active_is_a = rng.random() < 0.5
            active_region = region_a if active_is_a else region_b
            passive_region = region_b if active_is_a else region_a
            active_dir = direction_a if active_is_a else direction_b
            passive_dir = direction_b if active_is_a else direction_a
            active_cells = _place_cross_shape(rng, active_region, active_dir, step, gap_start, gap_end)
            if active_cells is not None:
                targets = _cross_targets(active_cells, active_dir, step, gap_start, gap_end, passive_region)
                if targets:
                    passive_cells = _place_stay_shape(rng, passive_region, passive_dir, step, forbidden=targets)
                    if passive_cells is not None:
                        placed = (active_cells, passive_cells) if active_is_a else (passive_cells, active_cells)
        if placed is None:
            a_cells = _place_stay_shape(rng, region_a, direction_a, step)
            b_cells = _place_stay_shape(rng, region_b, direction_b, step)
            if a_cells is None or b_cells is None:
                return None
            placed = (a_cells, b_cells)
            mode = "stay"

        seeds[region_a["key"]] = placed[0]
        seeds[region_b["key"]] = placed[1]
        used_modes.add(mode)
        _draw_gap(grid, edge, region_a, region_b, sep_rows, sep_cols, rng)

    for region in regions:
        if region["key"] in seeds:
            continue
        cells = _place_free_shape(rng, region)
        if cells is None:
            return None
        seeds[region["key"]] = cells

    for cells in seeds.values():
        for r, c in cells:
            grid[r, c] = BLACK

    try:
        parsed_regions, _, _, _ = _parse_regions(grid)
    except Exception:
        return None
    if len(parsed_regions) != len(regions):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
