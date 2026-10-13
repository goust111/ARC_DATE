import random
from collections import Counter

import numpy as np


SEP = 5
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)
PERP = {
    UP: (LEFT, RIGHT),
    DOWN: (LEFT, RIGHT),
    LEFT: (UP, DOWN),
    RIGHT: (UP, DOWN),
}
ARROW_TEMPLATES = {
    (RIGHT, UP): {(0, 0), (1, 0), (1, 1)},
    (RIGHT, DOWN): {(0, 0), (0, 1), (1, 0)},
    (LEFT, UP): {(0, 1), (1, 0), (1, 1)},
    (LEFT, DOWN): {(0, 0), (0, 1), (1, 1)},
    (UP, RIGHT): {(0, 0), (1, 0), (1, 1)},
    (UP, LEFT): {(0, 1), (1, 0), (1, 1)},
    (DOWN, RIGHT): {(0, 0), (0, 1), (1, 0)},
    (DOWN, LEFT): {(0, 0), (0, 1), (1, 1)},
}

GRID_BANDS = (3, 4, 5)
REGION_SIZE_CHOICES = (4, 5)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _contiguous_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = prev = idx
    runs.append((start, prev))
    return runs


def _first_seen_order(values: np.ndarray) -> list[int]:
    seen = set()
    order = []
    for value in values.ravel():
        value = int(value)
        if value not in seen:
            seen.add(value)
            order.append(value)
    return order


def _majority_color(patch: np.ndarray) -> int:
    values = [int(v) for v in patch.ravel()]
    counts = Counter(values)
    best = max(counts.values())
    for value in values:
        if counts[value] == best:
            return int(value)
    raise AssertionError("unreachable majority lookup")


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _arrow_candidate_dirs(coords: np.ndarray) -> list[tuple[int, int]]:
    if len(coords) != 3:
        return []
    r0, c0, r1, c1 = _bbox(coords)
    if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
        return []
    cells = {(int(r), int(c)) for r, c in coords}
    degrees = {}
    for r, c in cells:
        degree = 0
        for dr, dc in DIRS:
            if (r + dr, c + dc) in cells:
                degree += 1
        degrees[(r, c)] = degree
    deg_hist = sorted(degrees.values())
    if deg_hist != [1, 1, 2]:
        return []
    corner = next(cell for cell, degree in degrees.items() if degree == 2)
    endpoints = sorted(cell for cell, degree in degrees.items() if degree == 1)
    return [(er - corner[0], ec - corner[1]) for er, ec in endpoints]


def _find_separator_color(grid: np.ndarray) -> int:
    order = _first_seen_order(grid)
    best_color = None
    best_score = -1
    for color in order:
        full_rows = int(np.sum(np.all(grid == color, axis=1)))
        full_cols = int(np.sum(np.all(grid == color, axis=0)))
        if full_rows == 0 or full_cols == 0:
            continue
        score = full_rows + full_cols
        if score > best_score:
            best_score = score
            best_color = color
    assert best_color is not None, "expected a separator color with full rows and columns"
    return int(best_color)


def _region_bands(grid: np.ndarray, sep: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    rows = [r for r in range(grid.shape[0]) if not np.all(grid[r] == sep)]
    cols = [c for c in range(grid.shape[1]) if not np.all(grid[:, c] == sep)]
    row_bands = _contiguous_runs(rows)
    col_bands = _contiguous_runs(cols)
    assert row_bands and col_bands
    return row_bands, col_bands


def _first_object_in_direction(
    start: tuple[int, int],
    direction: tuple[int, int],
    object_regions: set[tuple[int, int]],
    n_rows: int,
    n_cols: int,
) -> tuple[int, int] | None:
    r, c = start
    dr, dc = direction
    while True:
        r += dr
        c += dc
        if not (0 <= r < n_rows and 0 <= c < n_cols):
            return None
        if (r, c) in object_regions:
            return (r, c)


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    sep = _find_separator_color(g)
    row_bands, col_bands = _region_bands(g, sep)
    n_rows = len(row_bands)
    n_cols = len(col_bands)

    infos: dict[tuple[int, int], dict] = {}
    maybe_regions: list[tuple[int, int]] = []
    object_regions: set[tuple[int, int]] = set()

    for ri, (r0, r1) in enumerate(row_bands):
        for ci, (c0, c1) in enumerate(col_bands):
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            bg = _majority_color(patch)
            fg_mask = patch != bg
            fg_cells = np.argwhere(fg_mask).astype(int)

            info = {
                "bg": int(bg),
                "fg_cells": fg_cells + np.array([[r0, c0]], dtype=int) if len(fg_cells) else fg_cells,
                "candidate_dirs": [],
                "kind": "empty",
            }
            if len(fg_cells):
                comps = _components(fg_mask)
                if len(comps) == 1:
                    candidate_dirs = _arrow_candidate_dirs(comps[0])
                    if candidate_dirs:
                        info["kind"] = "maybe_arrow"
                        info["candidate_dirs"] = candidate_dirs
                        maybe_regions.append((ri, ci))
                    else:
                        info["kind"] = "object"
                        object_regions.add((ri, ci))
                else:
                    info["kind"] = "object"
                    object_regions.add((ri, ci))
            infos[(ri, ci)] = info

    arrows: list[dict] = []
    for idx in maybe_regions:
        info = infos[idx]
        hits = []
        for direction in info["candidate_dirs"]:
            target = _first_object_in_direction(idx, direction, object_regions, n_rows, n_cols)
            if target is not None:
                hits.append((direction, target))
        if len(hits) == 1:
            direction, target = hits[0]
            arrows.append(
                {
                    "region": idx,
                    "target": target,
                    "dir": direction,
                    "bg": info["bg"],
                    "color": int(g[tuple(info["fg_cells"][0])]),
                }
            )
            info["kind"] = "arrow"
        else:
            info["kind"] = "object"
            object_regions.add(idx)

    out = g.copy()
    for arrow in arrows:
        cells = infos[arrow["region"]]["fg_cells"]
        out[cells[:, 0], cells[:, 1]] = int(infos[arrow["region"]]["bg"])

    by_target: dict[tuple[int, int], list[dict]] = {}
    for arrow in arrows:
        by_target.setdefault(arrow["target"], []).append(arrow)

    for target, target_arrows in by_target.items():
        if len(target_arrows) == 1:
            paint = int(target_arrows[0]["color"])
        else:
            counts = Counter(int(arrow["bg"]) for arrow in target_arrows)
            rare_count = min(counts.values())
            paint = next(int(arrow["bg"]) for arrow in target_arrows if counts[int(arrow["bg"])] == rare_count)
        cells = infos[target]["fg_cells"]
        out[cells[:, 0], cells[:, 1]] = paint

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each object pointed to by a regional arrow, then erase the arrows.

    Input structure:
    - Full gray rows and columns partition the grid into rectangular regions.
    - Every region has one solid non-gray background color, and some regions also contain a small 4-connected foreground object.
    - Some regions instead contain a 3-pixel L-shaped arrow whose color differs from that region's background.

    Rule:
    - Detect each arrow's direction by testing which of its two geometric candidate rays reaches the first object-containing region after crossing the separators.
    - If an object is pointed to by exactly one arrow, recolor the whole object to that arrow's color.
    - If multiple arrows point to the same object, ignore the arrow colors and recolor the object to the rarest background color among those arrows' source regions.
    - Remove every arrow and keep the gray separators and all region backgrounds unchanged.

    Key insight:
    - The 3-cell L shape alone is directionally ambiguous, so the real arrow direction is the unique candidate whose row-or-column ray actually lands on an object region.
    """
    return _transform(grid)


def _bands_from_sizes(sizes: list[int]) -> tuple[list[tuple[int, int]], int]:
    bands = []
    start = 0
    for i, size in enumerate(sizes):
        bands.append((start, start + size - 1))
        start += size
        if i + 1 < len(sizes):
            start += 1
    return bands, start


def _step_toward(src: tuple[int, int], dst: tuple[int, int]) -> tuple[int, int]:
    if src[0] == dst[0]:
        return RIGHT if src[1] < dst[1] else LEFT
    if src[1] == dst[1]:
        return DOWN if src[0] < dst[0] else UP
    raise ValueError("source and target must share a row or column")


def _between_cells(src: tuple[int, int], dst: tuple[int, int]) -> list[tuple[int, int]]:
    if src[0] == dst[0]:
        row = src[0]
        lo, hi = sorted((src[1], dst[1]))
        return [(row, col) for col in range(lo + 1, hi)]
    if src[1] == dst[1]:
        col = src[1]
        lo, hi = sorted((src[0], dst[0]))
        return [(row, col) for row in range(lo + 1, hi)]
    return []


def _decoy_options(
    src: tuple[int, int],
    true_dir: tuple[int, int],
    object_positions: set[tuple[int, int]],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    n_rows, n_cols = shape
    options = []
    for decoy in PERP[true_dir]:
        r, c = src
        dr, dc = decoy
        clear = True
        while True:
            r += dr
            c += dc
            if not (0 <= r < n_rows and 0 <= c < n_cols):
                break
            if (r, c) in object_positions:
                clear = False
                break
        if clear:
            options.append(decoy)
    return options


def _valid_sources(
    target: tuple[int, int],
    object_positions: set[tuple[int, int]],
    shape: tuple[int, int],
) -> list[dict]:
    n_rows, n_cols = shape
    choices = []
    for r in range(n_rows):
        for c in range(n_cols):
            src = (r, c)
            if src == target or src in object_positions:
                continue
            if src[0] != target[0] and src[1] != target[1]:
                continue
            between = _between_cells(src, target)
            if any(cell in object_positions for cell in between):
                continue
            true_dir = _step_toward(src, target)
            decoys = _decoy_options(src, true_dir, object_positions, shape)
            if decoys:
                choices.append({"src": src, "dir": true_dir, "decoys": decoys})
    return choices


def _choose_layout(n_rows: int, n_cols: int) -> dict | None:
    cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    for _ in range(400):
        t_multi, t_single = random.sample(cells, 2)
        object_positions = {t_multi, t_single}
        multi_sources = _valid_sources(t_multi, object_positions, (n_rows, n_cols))
        if len(multi_sources) < 3:
            continue
        chosen_multi = random.sample(multi_sources, 3)
        used = {entry["src"] for entry in chosen_multi}
        single_sources = [
            entry
            for entry in _valid_sources(t_single, object_positions, (n_rows, n_cols))
            if entry["src"] not in used
        ]
        if not single_sources:
            continue
        chosen_single = random.choice(single_sources)
        return {
            "multi_target": t_multi,
            "single_target": t_single,
            "multi_sources": chosen_multi,
            "single_source": chosen_single,
        }
    return None


def _assign_backgrounds(
    n_rows: int,
    n_cols: int,
    layout: dict,
    bg_dom: int,
    bg_rare: int,
    bg_other: int,
) -> np.ndarray | None:
    grid = np.full((n_rows, n_cols), -1, dtype=int)
    multi_sources = layout["multi_sources"]
    single_source = layout["single_source"]["src"]

    rare_idx = random.randrange(3)
    for i, entry in enumerate(multi_sources):
        grid[entry["src"]] = bg_rare if i == rare_idx else bg_dom
    grid[single_source] = bg_other

    fixed = {(entry["src"]) for entry in multi_sources}
    fixed.add(single_source)

    for _ in range(200):
        bg_grid = grid.copy()
        for r in range(n_rows):
            for c in range(n_cols):
                if (r, c) in fixed:
                    continue
                choices = [bg_dom, bg_rare, bg_other]
                if (r, c) == layout["multi_target"]:
                    choices = [bg_dom, bg_other]
                bg_grid[r, c] = random.choice(choices)
        row_ok = all(len({int(bg_grid[r, c]) for c in range(n_cols)}) >= 2 for r in range(n_rows))
        col_ok = all(len({int(bg_grid[r, c]) for r in range(n_rows)}) >= 2 for c in range(n_cols))
        if row_ok and col_ok:
            return bg_grid
    return None


def _grow_shape(max_h: int, max_w: int) -> list[tuple[int, int]]:
    area = random.randint(4, min(7, max_h * max_w))
    cells = {(random.randrange(max_h), random.randrange(max_w))}
    for _ in range(200):
        if len(cells) >= area:
            break
        r, c = random.choice(tuple(cells))
        neighbors = []
        for dr, dc in DIRS:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < max_h and 0 <= cc < max_w and (rr, cc) not in cells:
                neighbors.append((rr, cc))
        if neighbors:
            cells.add(random.choice(neighbors))
    if len(cells) != area:
        raise RuntimeError("failed to grow shape")
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return sorted((r - r0, c - c0) for r, c in cells)


def _place_object(region: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = region
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    shape = _grow_shape(h - 2, w - 2)
    bbox_h = 1 + max(r for r, _ in shape)
    bbox_w = 1 + max(c for _, c in shape)
    top = random.randint(1, h - bbox_h - 1)
    left = random.randint(1, w - bbox_w - 1)
    return [(r0 + top + r, c0 + left + c) for r, c in shape]


def _place_arrow(
    region: tuple[int, int, int, int],
    true_dir: tuple[int, int],
    decoy_dir: tuple[int, int],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = region
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    top = random.randint(1, h - 3)
    left = random.randint(1, w - 3)
    template = ARROW_TEMPLATES[(true_dir, decoy_dir)]
    return [(r0 + top + r, c0 + left + c) for r, c in sorted(template)]


def _expected_output(
    grid: np.ndarray,
    region_infos: dict[tuple[int, int], dict],
    arrows: list[dict],
    layout: dict,
    bg_rare: int,
    single_arrow_color: int,
) -> np.ndarray:
    out = grid.copy()
    for arrow in arrows:
        cells = np.array(arrow["cells"], dtype=int)
        out[cells[:, 0], cells[:, 1]] = int(arrow["bg"])

    multi_cells = np.array(region_infos[layout["multi_target"]]["cells"], dtype=int)
    out[multi_cells[:, 0], multi_cells[:, 1]] = int(bg_rare)

    single_cells = np.array(region_infos[layout["single_target"]]["cells"], dtype=int)
    out[single_cells[:, 0], single_cells[:, 1]] = int(single_arrow_color)
    return out


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.choice(GRID_BANDS)
    n_cols = random.choice(GRID_BANDS)
    row_sizes = [random.choice(REGION_SIZE_CHOICES) for _ in range(n_rows)]
    col_sizes = [random.choice(REGION_SIZE_CHOICES) for _ in range(n_cols)]
    row_bands, height = _bands_from_sizes(row_sizes)
    col_bands, width = _bands_from_sizes(col_sizes)
    if height > 30 or width > 30:
        return None

    layout = _choose_layout(n_rows, n_cols)
    if layout is None:
        return None

    palette = [c for c in range(10) if c != SEP]
    random.shuffle(palette)
    bg_dom, bg_rare, bg_other = palette[:3]
    obj_color_multi, obj_color_single = palette[3:5]
    arrow_colors = palette[5:9]
    multi_arrow_colors = arrow_colors[:3]
    single_arrow_color = arrow_colors[3]

    bg_grid = _assign_backgrounds(n_rows, n_cols, layout, bg_dom, bg_rare, bg_other)
    if bg_grid is None:
        return None

    grid = np.full((height, width), SEP, dtype=int)
    region_bounds: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    region_infos: dict[tuple[int, int], dict] = {}
    for ri, (r0, r1) in enumerate(row_bands):
        for ci, (c0, c1) in enumerate(col_bands):
            bg = int(bg_grid[ri, ci])
            grid[r0 : r1 + 1, c0 : c1 + 1] = bg
            region_bounds[(ri, ci)] = (r0, c0, r1, c1)
            region_infos[(ri, ci)] = {"bg": bg, "cells": []}

    multi_target = layout["multi_target"]
    single_target = layout["single_target"]
    multi_cells = _place_object(region_bounds[multi_target])
    single_cells = _place_object(region_bounds[single_target])
    region_infos[multi_target]["cells"] = multi_cells
    region_infos[single_target]["cells"] = single_cells
    for r, c in multi_cells:
        grid[r, c] = obj_color_multi
    for r, c in single_cells:
        grid[r, c] = obj_color_single

    arrows: list[dict] = []
    rare_bg_assigned = False
    for arrow_color, entry in zip(multi_arrow_colors, layout["multi_sources"]):
        src = entry["src"]
        src_bg = int(bg_grid[src])
        if src_bg == bg_rare:
            rare_bg_assigned = True
        decoy = random.choice(entry["decoys"])
        cells = _place_arrow(region_bounds[src], entry["dir"], decoy)
        for r, c in cells:
            grid[r, c] = arrow_color
        arrows.append(
            {
                "region": src,
                "target": multi_target,
                "bg": src_bg,
                "color": arrow_color,
                "cells": cells,
            }
        )

    if not rare_bg_assigned:
        return None

    single_entry = layout["single_source"]
    single_src = single_entry["src"]
    single_decoy = random.choice(single_entry["decoys"])
    single_cells_arrow = _place_arrow(region_bounds[single_src], single_entry["dir"], single_decoy)
    for r, c in single_cells_arrow:
        grid[r, c] = single_arrow_color
    arrows.append(
        {
            "region": single_src,
            "target": single_target,
            "bg": int(bg_grid[single_src]),
            "color": single_arrow_color,
            "cells": single_cells_arrow,
        }
    )

    expected = _expected_output(
        grid,
        region_infos,
        arrows,
        layout,
        bg_rare,
        single_arrow_color,
    )
    actual = _transform(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-region ARC input")
