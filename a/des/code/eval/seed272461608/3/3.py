from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import random


GREY = 5
ARROW = 9
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

ARROW_PATTERNS = {
    "right": frozenset({(0, 0), (1, 0), (1, 1)}),
    "left": frozenset({(0, 0), (0, 1), (1, 1)}),
    "up": frozenset({(0, 1), (1, 0), (1, 1)}),
    "down": frozenset({(0, 0), (0, 1), (1, 0)}),
}

DIR_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GEN_RETRIES = 400
ROWS_RANGE = (2, 4)
COLS_RANGE = (2, 4)
CELL_H_RANGE = (4, 7)
CELL_W_RANGE = (4, 7)
MARGIN_RANGE = (1, 3)
EXTRA_ARROW_PROB = 0.45
EMPTY_REGION_PROB = 0.28


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_edge(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0, c0, _, _ = _bbox(coords)
    return frozenset((r - r0, c - c0) for r, c in coords)


def _border_bg(grid: np.ndarray) -> int:
    border = np.concatenate([grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]])
    return int(Counter(map(int, border.tolist())).most_common(1)[0][0])


def _extract_regions(grid: np.ndarray) -> tuple[list[dict], list[int], list[int], dict[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _border_bg(g)
    regions: list[dict] = []
    for coords in _components(g != GREY):
        base_colors = {int(g[r, c]) for r, c in coords if int(g[r, c]) != ARROW}
        assert len(base_colors) == 1, "each region must have exactly one base color"
        if _touches_edge(coords, h, w) and base_colors == {bg}:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        regions.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "area": len(coords),
                "orig_color": int(next(iter(base_colors))),
            }
        )

    assert regions, "expected enclosed regions"
    row_starts = sorted({region["r0"] for region in regions})
    col_starts = sorted({region["c0"] for region in regions})
    assert len(regions) == len(row_starts) * len(col_starts), "expected a full rectangular region table"

    row_index = {r0: idx for idx, r0 in enumerate(row_starts)}
    col_index = {c0: idx for idx, c0 in enumerate(col_starts)}
    anchor_to_idx: dict[tuple[int, int], int] = {}
    for idx, region in enumerate(regions):
        anchor = (region["r0"], region["c0"])
        anchor_to_idx[anchor] = idx
        region["row_idx"] = row_index[region["r0"]]
        region["col_idx"] = col_index[region["c0"]]

    return regions, row_starts, col_starts, anchor_to_idx


def _decode_arrow(
    grid: np.ndarray,
    region: dict,
    row_starts: list[int],
    col_starts: list[int],
    anchor_to_idx: dict[tuple[int, int], int],
) -> int | None:
    arrow_cells = [(r, c) for r, c in region["coords"] if int(grid[r, c]) == ARROW]
    if not arrow_cells:
        return None

    assert len(arrow_cells) == 3, "each arrow marker must use exactly 3 pixels"
    r0, c0, r1, c1 = _bbox(arrow_cells)
    assert r1 - r0 == 1 and c1 - c0 == 1, "arrow marker must fit in a 2x2 box"
    assert len(_components(np.isin(grid, [ARROW]) & _mask_from_coords(grid.shape, arrow_cells))) == 1

    shape = _normalize(arrow_cells)
    directions = [name for name, pattern in ARROW_PATTERNS.items() if shape == pattern]
    assert len(directions) == 1, "arrow marker must be an L-shape in one of four rotations"
    direction = directions[0]

    touches = {
        "up": any(r == region["r0"] for r, _ in arrow_cells),
        "down": any(r == region["r1"] for r, _ in arrow_cells),
        "left": any(c == region["c0"] for _, c in arrow_cells),
        "right": any(c == region["c1"] for _, c in arrow_cells),
    }
    assert touches[direction], "arrow rotation must agree with the region edge it touches"
    assert sum(bool(v) for v in touches.values()) == 1, "arrow must be placed away from region corners"

    dr, dc = DIR_DELTA[direction]
    nr = region["row_idx"] + dr
    nc = region["col_idx"] + dc
    assert 0 <= nr < len(row_starts) and 0 <= nc < len(col_starts), "arrow must point to an orthogonal neighbor"
    return anchor_to_idx[(row_starts[nr], col_starts[nc])]


def _mask_from_coords(shape: tuple[int, int], coords: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Regions are separated by grey lines. Maroon 3-pixel L markers inside a region
    point to one orthogonal neighboring region. Each region's output color is
    determined by the colors of regions whose arrows point into it.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    regions, row_starts, col_starts, anchor_to_idx = _extract_regions(g)
    incoming: dict[int, list[int]] = defaultdict(list)

    for src_idx, region in enumerate(regions):
        dst_idx = _decode_arrow(g, region, row_starts, col_starts, anchor_to_idx)
        if dst_idx is not None:
            incoming[dst_idx].append(src_idx)

    for idx, region in enumerate(regions):
        sources = incoming.get(idx, [])
        if not sources:
            fill = region["orig_color"]
        elif len(sources) == 1:
            fill = regions[sources[0]]["orig_color"]
        else:
            best = max(sources, key=lambda src: (regions[src]["area"], -src))
            fill = regions[best]["orig_color"]
        for r, c in region["coords"]:
            out[r, c] = fill

    return out


def _neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell
    result = []
    if r > 0:
        result.append((r - 1, c))
    if r + 1 < rows:
        result.append((r + 1, c))
    if c > 0:
        result.append((r, c - 1))
    if c + 1 < cols:
        result.append((r, c + 1))
    return result


def _direction_between(src: tuple[int, int], dst: tuple[int, int]) -> str:
    dr = dst[0] - src[0]
    dc = dst[1] - src[1]
    if dr == -1 and dc == 0:
        return "up"
    if dr == 1 and dc == 0:
        return "down"
    if dr == 0 and dc == -1:
        return "left"
    if dr == 0 and dc == 1:
        return "right"
    raise AssertionError("cells must be orthogonal neighbors")


def _layout_bboxes(
    margins: tuple[int, int, int, int],
    row_heights: list[int],
    col_widths: list[int],
) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    top_margin, _bottom_margin, left_margin, _right_margin = margins
    top = top_margin
    left = left_margin

    row_starts = []
    pos = top + 1
    for height in row_heights:
        row_starts.append(pos)
        pos += height + 1

    col_starts = []
    pos = left + 1
    for width in col_widths:
        col_starts.append(pos)
        pos += width + 1

    bboxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for r_idx, r0 in enumerate(row_starts):
        for c_idx, c0 in enumerate(col_starts):
            bboxes[(r_idx, c_idx)] = (r0, c0, r0 + row_heights[r_idx] - 1, c0 + col_widths[c_idx] - 1)
    return bboxes


def _paint_arrow(grid: np.ndarray, bbox: tuple[int, int, int, int], direction: str) -> None:
    r0, c0, r1, c1 = bbox
    if direction == "right":
        rr = random.randint(r0 + 1, r1 - 2)
        cells = [(rr, c1 - 1), (rr + 1, c1 - 1), (rr + 1, c1)]
    elif direction == "left":
        rr = random.randint(r0 + 1, r1 - 2)
        cells = [(rr, c0), (rr, c0 + 1), (rr + 1, c0 + 1)]
    elif direction == "up":
        cc = random.randint(c0 + 1, c1 - 2)
        cells = [(r0, cc + 1), (r0 + 1, cc), (r0 + 1, cc + 1)]
    else:
        cc = random.randint(c0 + 1, c1 - 2)
        cells = [(r1 - 1, cc), (r1 - 1, cc + 1), (r1, cc)]

    for r, c in cells:
        grid[r, c] = ARROW


def _render_input(
    bg: int,
    margins: tuple[int, int, int, int],
    row_heights: list[int],
    col_widths: list[int],
    region_colors: dict[tuple[int, int], int],
    outgoing: dict[tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    top_margin, bottom_margin, left_margin, right_margin = margins
    table_h = sum(row_heights) + len(row_heights) + 1
    table_w = sum(col_widths) + len(col_widths) + 1
    height = top_margin + table_h + bottom_margin
    width = left_margin + table_w + right_margin

    grid = np.full((height, width), bg, dtype=int)
    top = top_margin
    left = left_margin
    grid[top : top + table_h, left : left + table_w] = GREY

    bboxes = _layout_bboxes(margins, row_heights, col_widths)
    for cell, (r0, c0, r1, c1) in bboxes.items():
        grid[r0 : r1 + 1, c0 : c1 + 1] = region_colors[cell]

    for src, dst in outgoing.items():
        _paint_arrow(grid, bboxes[src], _direction_between(src, dst))

    return grid


def _render_expected(
    bg: int,
    margins: tuple[int, int, int, int],
    row_heights: list[int],
    col_widths: list[int],
    region_colors: dict[tuple[int, int], int],
    outgoing: dict[tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    incoming: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for src, dst in outgoing.items():
        incoming[dst].append(src)

    table_h = sum(row_heights) + len(row_heights) + 1
    table_w = sum(col_widths) + len(col_widths) + 1
    top_margin, bottom_margin, left_margin, right_margin = margins
    height = top_margin + table_h + bottom_margin
    width = left_margin + table_w + right_margin
    out = np.full((height, width), bg, dtype=int)

    top = top_margin
    left = left_margin
    out[top : top + table_h, left : left + table_w] = GREY
    bboxes = _layout_bboxes(margins, row_heights, col_widths)
    areas = {(r, c): row_heights[r] * col_widths[c] for r in range(len(row_heights)) for c in range(len(col_widths))}

    for cell, (r0, c0, r1, c1) in bboxes.items():
        sources = incoming.get(cell, [])
        if not sources:
            fill = region_colors[cell]
        elif len(sources) == 1:
            fill = region_colors[sources[0]]
        else:
            winner = max(sources, key=lambda src: (areas[src], -src[0], -src[1]))
            fill = region_colors[winner]
        out[r0 : r1 + 1, c0 : c1 + 1] = fill

    return out


def _choose_branch_cells(
    rows: int,
    cols: int,
    areas: dict[tuple[int, int], int],
) -> tuple[
    tuple[int, int],
    tuple[tuple[int, int], ...],
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
]:
    cells = [(r, c) for r in range(rows) for c in range(cols)]

    branch_candidates = []
    for multi_target in cells:
        neighbors = _neighbors(multi_target, rows, cols)
        if len(neighbors) < 2:
            continue
        max_k = 3 if len(cells) >= 7 and len(neighbors) >= 3 else 2
        ks = [2, 3] if max_k == 3 else [2]
        for k in ks:
            for subset in combinations(neighbors, k):
                max_area = max(areas[cell] for cell in subset)
                if sum(areas[cell] == max_area for cell in subset) != 1:
                    continue
                branch_candidates.append((multi_target, subset))
    if not branch_candidates:
        raise AssertionError("need at least one valid multi-arrow target")

    weighted = []
    for item in branch_candidates:
        weighted.extend([item] * (3 if len(item[1]) == 2 else 2))
    multi_target, multi_sources = random.choice(weighted)
    used = set(multi_sources)
    used.add(multi_target)

    single_options = []
    for single_target in cells:
        if single_target in used:
            continue
        for single_source in _neighbors(single_target, rows, cols):
            if single_source in used:
                continue
            single_options.append((single_source, single_target))
    if not single_options:
        raise AssertionError("need a disjoint single-arrow pair")

    single_source, single_target = random.choice(single_options)
    used.add(single_source)
    used.add(single_target)

    zero_options = [cell for cell in cells if cell not in used]
    if not zero_options:
        raise AssertionError("need a distinct zero-arrow target")
    zero_target = random.choice(zero_options)

    return multi_target, tuple(multi_sources), single_source, single_target, zero_target


def _assign_colors(
    rows: int,
    cols: int,
    bg: int,
    areas: dict[tuple[int, int], int],
    multi_target: tuple[int, int],
    multi_sources: tuple[tuple[int, int], ...],
    single_source: tuple[int, int],
    single_target: tuple[int, int],
    zero_target: tuple[int, int],
) -> dict[tuple[int, int], int]:
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    palette = [c for c in range(10) if c not in (bg, GREY, ARROW)]
    assert len(palette) >= 4

    colors = {cell: (bg if random.random() < EMPTY_REGION_PROB else random.choice(palette)) for cell in cells}

    winner_source = max(multi_sources, key=lambda cell: areas[cell])
    winner_color = random.choice(palette)
    colors[winner_source] = winner_color

    remaining_palette = [c for c in palette if c != winner_color]
    for src in multi_sources:
        if src == winner_source:
            continue
        colors[src] = random.choice(remaining_palette)

    single_color_choices = [c for c in palette if c != winner_color]
    colors[single_source] = random.choice(single_color_choices)

    target_choices = [bg] + palette
    colors[multi_target] = random.choice([c for c in target_choices if c != winner_color])
    colors[single_target] = random.choice([c for c in target_choices if c != colors[single_source]])
    colors[zero_target] = random.choice(palette)

    spare = [cell for cell in cells if cell not in {multi_target, single_target, zero_target, single_source, *multi_sources}]
    if spare and random.random() < 0.7:
        colors[random.choice(spare)] = bg

    if all(color == bg for color in colors.values()):
        colors[random.choice(cells)] = random.choice(palette)
    return colors


def _make_case() -> tuple[np.ndarray, np.ndarray]:
    rows = random.randint(*ROWS_RANGE)
    cols = random.randint(*COLS_RANGE)
    while rows * cols < 6:
        rows = random.randint(*ROWS_RANGE)
        cols = random.randint(*COLS_RANGE)

    row_heights = [random.randint(*CELL_H_RANGE) for _ in range(rows)]
    col_widths = [random.randint(*CELL_W_RANGE) for _ in range(cols)]
    margins = (
        random.randint(*MARGIN_RANGE),
        random.randint(*MARGIN_RANGE),
        random.randint(*MARGIN_RANGE),
        random.randint(*MARGIN_RANGE),
    )

    bg = random.choice([c for c in range(10) if c not in (GREY, ARROW)])
    areas = {(r, c): row_heights[r] * col_widths[c] for r in range(rows) for c in range(cols)}
    multi_target, multi_sources, single_source, single_target, zero_target = _choose_branch_cells(rows, cols, areas)

    outgoing: dict[tuple[int, int], tuple[int, int]] = {src: multi_target for src in multi_sources}
    outgoing[single_source] = single_target

    blocked_targets = {multi_target, single_target, zero_target}
    used_sources = set(outgoing)
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    for src in cells:
        if src in used_sources or src in blocked_targets:
            continue
        if random.random() >= EXTRA_ARROW_PROB:
            continue
        options = [dst for dst in _neighbors(src, rows, cols) if dst not in blocked_targets]
        if options:
            outgoing[src] = random.choice(options)

    region_colors = _assign_colors(
        rows,
        cols,
        bg,
        areas,
        multi_target,
        multi_sources,
        single_source,
        single_target,
        zero_target,
    )

    inp = _render_input(bg, margins, row_heights, col_widths, region_colors, outgoing)
    expected = _render_expected(bg, margins, row_heights, col_widths, region_colors, outgoing)
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            inp, expected = _make_case()
            actual = main(inp)
            if not np.array_equal(actual, expected):
                continue
            return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input")
