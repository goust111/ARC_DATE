from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
DIVIDER_PALETTE = (1, 2, 3, 6, 7, 8, 9)

GEN_ATTEMPTS = 400
REGION_ATTEMPTS = 160

BASE_SHAPES = (
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _normalize(cells: set[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    pts = list(cells)
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return tuple(sorted((r - min_r, c - min_c) for r, c in pts))


def _transform_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    cells = set(shape)
    for _ in range(random.randint(0, 3)):
        cells = {(c, -r) for r, c in cells}
    if random.random() < 0.5:
        cells = {(r, -c) for r, c in cells}
    return _normalize(cells)


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _anchor_offsets(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    anchor_r, anchor_c = min(shape)
    return tuple(sorted((r - anchor_r, c - anchor_c) for r, c in shape))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(sorted(comp))
    return comps


def _intervals(size: int, lines: list[int]) -> list[tuple[int, int]]:
    cuts = [-1] + sorted(lines) + [size]
    spans: list[tuple[int, int]] = []
    for left, right in zip(cuts, cuts[1:]):
        start = left + 1
        end = right - 1
        if start <= end:
            spans.append((start, end))
    return spans


def _dominant_non_black(values: list[int]) -> int:
    counts = Counter(v for v in values if v != BLACK)
    assert counts, "expected non-black line colors"
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    assert len(winners) == 1, "divider line color must be unambiguous"
    return int(winners[0])


def _detect_dividers(grid: np.ndarray) -> tuple[list[int], list[int], dict[int, int], dict[int, int]]:
    h, w = grid.shape
    line_rows = [r for r in range(h) if np.all(grid[r, :] != BLACK)]
    line_cols = [c for c in range(w) if np.all(grid[:, c] != BLACK)]
    row_colors: dict[int, int] = {}
    col_colors: dict[int, int] = {}

    line_row_set = set(line_rows)
    line_col_set = set(line_cols)

    for r in line_rows:
        vals = [int(grid[r, c]) for c in range(w) if c not in line_col_set]
        if not vals:
            vals = [int(v) for v in grid[r, :]]
        row_colors[r] = _dominant_non_black(vals)

    for c in line_cols:
        vals = [int(grid[r, c]) for r in range(h) if r not in line_row_set]
        if not vals:
            vals = [int(v) for v in grid[:, c]]
        col_colors[c] = _dominant_non_black(vals)

    assert line_rows or line_cols, "expected at least one divider line"
    return line_rows, line_cols, row_colors, col_colors


def _region_stamp_color(
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    row_colors: dict[int, int],
    col_colors: dict[int, int],
) -> int:
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    candidates: list[tuple[float, int]] = []

    if r0 - 1 in row_colors:
        candidates.append((abs(center_r - (r0 - 1)), int(row_colors[r0 - 1])))
    if r1 + 1 in row_colors:
        candidates.append((abs(center_r - (r1 + 1)), int(row_colors[r1 + 1])))
    if c0 - 1 in col_colors:
        candidates.append((abs(center_c - (c0 - 1)), int(col_colors[c0 - 1])))
    if c1 + 1 in col_colors:
        candidates.append((abs(center_c - (c1 + 1)), int(col_colors[c1 + 1])))

    assert candidates, "every region must touch at least one divider line"
    min_dist = min(dist for dist, _ in candidates)
    winners = {color for dist, color in candidates if abs(dist - min_dist) < 1e-9}
    assert len(winners) == 1, "nearest divider line color must be unique or color-consistent"
    return int(next(iter(winners)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each region's grey template onto its yellow seeds using the color of the nearest adjacent divider line.

    Input structure:
    - The background is black.
    - Full-row and/or full-column colored divider lines split the canvas into rectangular regions.
    - Each region contains exactly one small 4-connected grey template and one to three yellow seed pixels.

    Rule:
    - For each region, inspect the divider lines directly touching its boundary and select the color of the one whose row/column is nearest to the region center.
    - Extract the grey template as offsets relative to its top-leftmost pixel.
    - For every yellow seed, use the seed as that anchor pixel and repaint the same offsets in the selected divider color, clipping any stamped pixels that would cross the region boundary; keep the original grey template unchanged.

    Key insight:
    - The final pattern in each region combines three separate cues: shape from the grey template, position from the yellow seeds, and color from the nearest surrounding divider line.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    out = g.copy()

    line_rows, line_cols, row_colors, col_colors = _detect_dividers(g)
    row_spans = _intervals(h, line_rows)
    col_spans = _intervals(w, line_cols)

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            region = g[r0 : r1 + 1, c0 : c1 + 1]
            grey_components = _components(region == GREY)
            assert len(grey_components) == 1, "each region must contain exactly one grey template"
            offsets = _anchor_offsets(tuple(grey_components[0]))

            seeds = [tuple(map(int, coord)) for coord in np.argwhere(region == YELLOW)]
            assert 1 <= len(seeds) <= 3, "each region must contain one to three yellow seeds"

            stamp_color = _region_stamp_color(r0, r1, c0, c1, row_colors, col_colors)

            for sr, sc in seeds:
                for dr, dc in offsets:
                    rr = r0 + sr + dr
                    cc = c0 + sc + dc
                    if r0 <= rr <= r1 and c0 <= cc <= c1 and int(g[rr, cc]) != GREY:
                        out[rr, cc] = stamp_color

    return out


def _choose_partition() -> tuple[int, int, dict[int, int], dict[int, int]] | None:
    mode = random.choice(("horizontal", "vertical", "grid"))

    if mode == "horizontal":
        line_count = random.choice((1, 2))
        heights = [random.randint(6, 9) for _ in range(line_count + 1)]
        h = sum(heights) + line_count
        w = random.randint(14, 24)
        if h > 30:
            return None
        color = random.choice(DIVIDER_PALETTE)
        row_colors: dict[int, int] = {}
        cursor = heights[0]
        for idx in range(line_count):
            row_colors[cursor] = color
            cursor += 1 + heights[idx + 1]
        return h, w, row_colors, {}

    if mode == "vertical":
        line_count = random.choice((1, 2))
        widths = [random.randint(6, 9) for _ in range(line_count + 1)]
        h = random.randint(14, 24)
        w = sum(widths) + line_count
        if w > 30:
            return None
        color = random.choice(DIVIDER_PALETTE)
        col_colors: dict[int, int] = {}
        cursor = widths[0]
        for idx in range(line_count):
            col_colors[cursor] = color
            cursor += 1 + widths[idx + 1]
        return h, w, {}, col_colors

    for _ in range(80):
        top_h = random.randint(6, 10)
        bottom_h = random.randint(6, 10)
        left_w = random.randint(6, 10)
        right_w = random.randint(6, 10)
        h = top_h + 1 + bottom_h
        w = left_w + 1 + right_w
        if h > 30 or w > 30:
            continue
        if len({top_h, bottom_h} & {left_w, right_w}) > 0:
            continue
        horiz_color, vert_color = random.sample(DIVIDER_PALETTE, 2)
        row_colors = {top_h: horiz_color}
        col_colors = {left_w: vert_color}
        region_colors = set()
        for r0, r1 in _intervals(h, list(row_colors)):
            for c0, c1 in _intervals(w, list(col_colors)):
                region_colors.add(_region_stamp_color(r0, r1, c0, c1, row_colors, col_colors))
        if len(region_colors) < 2:
            continue
        return h, w, row_colors, col_colors
    return None


def _paint_dividers(h: int, w: int, row_colors: dict[int, int], col_colors: dict[int, int]) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    for r, color in row_colors.items():
        grid[int(r), :] = int(color)
    for c, color in col_colors.items():
        grid[:, int(c)] = int(color)
    for r, r_color in row_colors.items():
        for c, c_color in col_colors.items():
            grid[int(r), int(c)] = random.choice((int(r_color), int(c_color)))
    return grid


def _touches_occupied(cell: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    if cell in occupied:
        return True
    r, c = cell
    return any((nr, nc) in occupied for nr, nc in _neighbors4(r, c))


def _seed_candidates(
    region_h: int,
    region_w: int,
    offsets: tuple[tuple[int, int], ...],
    template_cells: set[tuple[int, int]],
    seed_cells: set[tuple[int, int]],
    painted_cells: set[tuple[int, int]],
    require_clipped: bool | None,
) -> list[tuple[int, int, set[tuple[int, int]]]]:
    candidates: list[tuple[int, int, set[tuple[int, int]]]] = []
    occupied = set(template_cells) | set(seed_cells)
    for sr in range(region_h):
        for sc in range(region_w):
            if _touches_occupied((sr, sc), occupied):
                continue
            visible: set[tuple[int, int]] = set()
            clipped = False
            for dr, dc in offsets:
                rr = sr + dr
                cc = sc + dc
                if 0 <= rr < region_h and 0 <= cc < region_w:
                    visible.add((rr, cc))
                else:
                    clipped = True
            if require_clipped is True and not clipped:
                continue
            if require_clipped is False and clipped:
                continue
            if len(visible) < 2:
                continue
            if visible & template_cells:
                continue
            if visible & seed_cells:
                continue
            if visible & painted_cells:
                continue
            candidates.append((sr, sc, visible))
    random.shuffle(candidates)
    return candidates


def _desired_clip_flags(seed_count: int) -> list[bool]:
    if seed_count == 1:
        return [random.random() < 0.5]
    flags = [True, False]
    while len(flags) < seed_count:
        flags.append(random.random() < 0.5)
    random.shuffle(flags)
    return flags[:seed_count]


def _place_region_objects(region_h: int, region_w: int) -> tuple[
    set[tuple[int, int]],
    tuple[tuple[int, int], ...],
    list[tuple[int, int]],
] | None:
    for _ in range(REGION_ATTEMPTS):
        shape = _transform_shape(random.choice(BASE_SHAPES))
        shape_h, shape_w = _shape_dims(shape)
        if shape_h > region_h or shape_w > region_w:
            continue

        top_margin = 1 if region_h - shape_h >= 2 else 0
        left_margin = 1 if region_w - shape_w >= 2 else 0
        max_r = region_h - shape_h - top_margin
        max_c = region_w - shape_w - left_margin
        if max_r < top_margin or max_c < left_margin:
            continue
        origin_r = random.randint(top_margin, max_r)
        origin_c = random.randint(left_margin, max_c)

        template_cells = {(origin_r + r, origin_c + c) for r, c in shape}
        offsets = _anchor_offsets(shape)

        seed_count = random.randint(1, 3)
        seeds: list[tuple[int, int]] = []
        seed_cells: set[tuple[int, int]] = set()
        painted_cells: set[tuple[int, int]] = set()

        ok = True
        for want_clipped in _desired_clip_flags(seed_count):
            choices = _seed_candidates(region_h, region_w, offsets, template_cells, seed_cells, painted_cells, want_clipped)
            if not choices:
                choices = _seed_candidates(region_h, region_w, offsets, template_cells, seed_cells, painted_cells, None)
            if not choices:
                ok = False
                break
            sr, sc, visible = choices[0]
            seeds.append((sr, sc))
            seed_cells.add((sr, sc))
            painted_cells |= visible

        if ok:
            return template_cells, offsets, seeds
    return None


def _render_expected(
    grid: np.ndarray,
    specs: list[dict[str, object]],
) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        r0, r1, c0, c1 = spec["bounds"]  # type: ignore[misc]
        stamp_color = int(spec["color"])  # type: ignore[index]
        offsets = spec["offsets"]  # type: ignore[assignment]
        seeds = spec["seeds"]  # type: ignore[assignment]
        for sr, sc in seeds:
            for dr, dc in offsets:
                rr = sr + dr
                cc = sc + dc
                if int(r0) <= rr <= int(r1) and int(c0) <= cc <= int(c1) and int(grid[rr, cc]) != GREY:
                    out[rr, cc] = stamp_color
    return out


def _generate_attempt() -> np.ndarray | None:
    partition = _choose_partition()
    if partition is None:
        return None
    h, w, row_colors, col_colors = partition
    grid = _paint_dividers(h, w, row_colors, col_colors)

    specs: list[dict[str, object]] = []
    for r0, r1 in _intervals(h, list(row_colors)):
        for c0, c1 in _intervals(w, list(col_colors)):
            region_h = r1 - r0 + 1
            region_w = c1 - c0 + 1
            placed = _place_region_objects(region_h, region_w)
            if placed is None:
                return None
            template_cells, offsets, seeds = placed
            stamp_color = _region_stamp_color(r0, r1, c0, c1, row_colors, col_colors)

            for tr, tc in template_cells:
                grid[r0 + tr, c0 + tc] = GREY
            abs_seeds: list[tuple[int, int]] = []
            for sr, sc in seeds:
                ar = r0 + sr
                ac = c0 + sc
                grid[ar, ac] = YELLOW
                abs_seeds.append((ar, ac))

            specs.append(
                {
                    "bounds": (r0, r1, c0, c1),
                    "color": stamp_color,
                    "offsets": offsets,
                    "seeds": abs_seeds,
                }
            )

    expected = _render_expected(grid, specs)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
