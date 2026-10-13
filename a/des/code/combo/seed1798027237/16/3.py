from __future__ import annotations

from collections import deque
from functools import lru_cache
from itertools import product

import numpy as np


BG = 0
WALL = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400
LAYOUT_POOL = (
    (1, 2),
    (2, 1),
    (2, 3),
    (2, 3),
    (3, 2),
    (3, 2),
    (2, 4),
    (2, 4),
    (4, 2),
    (4, 2),
    (3, 3),
    (3, 3),
)
RNG = np.random.default_rng()


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _dominant_non_bg_color(grid: np.ndarray, bg: int) -> int:
    values, counts = np.unique(grid[grid != bg], return_counts=True)
    assert values.size > 0, "expected at least one non-background color"
    return int(values[np.argmax(counts)])


def _partition_spans(size: int, line_positions: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    start = 0
    for pos in line_positions:
        assert start <= pos - 1, "lines must leave a non-empty region between them"
        spans.append((start, pos - 1))
        start = pos + 1
    assert start <= size - 1, "lines must stay away from the grid border"
    spans.append((start, size - 1))
    return tuple(spans)


def _band_lookup(size: int, spans: tuple[tuple[int, int], ...]) -> np.ndarray:
    lookup = np.full(size, -1, dtype=int)
    for idx, (start, end) in enumerate(spans):
        lookup[start : end + 1] = idx
    return lookup


def _component(start: tuple[int, int], adj: dict[tuple[int, int], set[tuple[int, int]]]) -> set[tuple[int, int]]:
    seen = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for nxt in adj[node]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def _build_graph(
    region_rows: int,
    region_cols: int,
    horizontal_choices: tuple[int, ...],
    vertical_choices: tuple[int, ...],
) -> dict[tuple[int, int], set[tuple[int, int]]]:
    adj = {(r, c): set() for r in range(region_rows) for c in range(region_cols)}
    for row_boundary, col_band in enumerate(horizontal_choices):
        a = (row_boundary, col_band)
        b = (row_boundary + 1, col_band)
        adj[a].add(b)
        adj[b].add(a)
    for col_boundary, row_band in enumerate(vertical_choices):
        a = (row_band, col_boundary)
        b = (row_band, col_boundary + 1)
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _sorted_components(adj: dict[tuple[int, int], set[tuple[int, int]]]) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[int, int]] = set()
    comps: list[tuple[tuple[int, int], ...]] = []
    for node in sorted(adj):
        if node in seen:
            continue
        comp = tuple(sorted(_component(node, adj)))
        seen.update(comp)
        comps.append(comp)
    return comps


@lru_cache(maxsize=None)
def _feasible_configs(region_rows: int, region_cols: int) -> tuple[tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[tuple[int, int], ...], ...]], ...]:
    configs = []
    for horizontal_choices in product(range(region_cols), repeat=max(0, region_rows - 1)):
        for vertical_choices in product(range(region_rows), repeat=max(0, region_cols - 1)):
            adj = _build_graph(region_rows, region_cols, horizontal_choices, vertical_choices)
            comps = tuple(comp for comp in _sorted_components(adj) if len(comp) == 2)
            if comps:
                configs.append((tuple(horizontal_choices), tuple(vertical_choices), comps))
    return tuple(configs)


def _spans_from_lengths(lengths: np.ndarray) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...], int]:
    spans: list[tuple[int, int]] = []
    lines: list[int] = []
    start = 0
    for idx, length in enumerate(lengths.tolist()):
        end = start + int(length) - 1
        spans.append((start, end))
        start = end + 1
        if idx < len(lengths) - 1:
            lines.append(start)
            start += 1
    return tuple(spans), tuple(lines), start


def _analyze_grid(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    bg = _most_common_color(g)
    wall = _dominant_non_bg_color(g, bg)
    wall_mask = g == wall

    line_rows = tuple(int(r) for r in np.flatnonzero(np.count_nonzero(wall_mask, axis=1) > (w / 2)))
    line_cols = tuple(int(c) for c in np.flatnonzero(np.count_nonzero(wall_mask, axis=0) > (h / 2)))
    assert line_rows or line_cols, "expected at least one partition line"

    row_spans = _partition_spans(h, line_rows)
    col_spans = _partition_spans(w, line_cols)
    assert len(row_spans) * len(col_spans) > 1, "expected multiple subregions"

    row_of = _band_lookup(h, row_spans)
    col_of = _band_lookup(w, col_spans)
    line_row_set = set(line_rows)
    line_col_set = set(line_cols)

    adj = {(r, c): set() for r in range(len(row_spans)) for c in range(len(col_spans))}

    for row_boundary, line_r in enumerate(line_rows):
        off_crossing = g[line_r, [c for c in range(w) if c not in line_col_set]]
        assert np.all((off_crossing == bg) | (off_crossing == wall))
        gap_cols = [c for c in range(w) if c not in line_col_set and int(g[line_r, c]) == bg]
        assert len(gap_cols) == 1, "each horizontal line must have exactly one non-intersection gap"
        gap_c = gap_cols[0]
        col_band = int(col_of[gap_c])
        assert col_band >= 0
        a = (row_boundary, col_band)
        b = (row_boundary + 1, col_band)
        adj[a].add(b)
        adj[b].add(a)

    for col_boundary, line_c in enumerate(line_cols):
        off_crossing = g[[r for r in range(h) if r not in line_row_set], line_c]
        assert np.all((off_crossing == bg) | (off_crossing == wall))
        gap_rows = [r for r in range(h) if r not in line_row_set and int(g[r, line_c]) == bg]
        assert len(gap_rows) == 1, "each vertical line must have exactly one non-intersection gap"
        gap_r = gap_rows[0]
        row_band = int(row_of[gap_r])
        assert row_band >= 0
        a = (row_band, col_boundary)
        b = (row_band, col_boundary + 1)
        adj[a].add(b)
        adj[b].add(a)

    seed_by_region: dict[tuple[int, int], int] = {}
    target_region: tuple[int, int] | None = None
    all_seed_colors: list[int] = []

    for region_r, (r0, r1) in enumerate(row_spans):
        for region_c, (c0, c1) in enumerate(col_spans):
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            assert not np.any(patch == wall), "partition walls must not spill into region interiors"
            seed_cells = np.argwhere((patch != bg) & (patch != wall))
            assert seed_cells.shape[0] <= 1, "each region may contain at most one seed"
            if seed_cells.shape[0] == 0:
                assert target_region is None, "expected a unique empty target region"
                target_region = (region_r, region_c)
            else:
                sr, sc = map(int, seed_cells[0])
                color = int(patch[sr, sc])
                seed_by_region[(region_r, region_c)] = color
                all_seed_colors.append(color)

    assert target_region is not None, "expected one empty region"
    assert len(seed_by_region) + 1 == len(row_spans) * len(col_spans), "every non-target region must contain one seed"
    assert len(all_seed_colors) == len(set(all_seed_colors)), "seed colors must be distinct"

    target_component = _component(target_region, adj)
    reachable_seed_regions = [node for node in target_component if node in seed_by_region]
    assert len(reachable_seed_regions) == 1, "the target must connect to exactly one seed region"
    source_region = reachable_seed_regions[0]

    return {
        "bg": bg,
        "wall": wall,
        "row_spans": row_spans,
        "col_spans": col_spans,
        "target_region": target_region,
        "source_region": source_region,
        "fill_color": seed_by_region[source_region],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the unique empty subregion with the color of the only seed that can
    reach it through the chain of single-cell gaps in the grey partition lines.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_grid(g)

    target_r, target_c = info["target_region"]
    row_spans = info["row_spans"]
    col_spans = info["col_spans"]
    r0, r1 = row_spans[target_r]
    c0, c1 = col_spans[target_c]
    fill_color = int(info["fill_color"])
    bg = int(info["bg"])

    out = g.copy()
    patch = out[r0 : r1 + 1, c0 : c1 + 1]
    patch[patch == bg] = fill_color
    return out


def _gen_attempt() -> np.ndarray | None:
    region_rows, region_cols = LAYOUT_POOL[int(RNG.integers(len(LAYOUT_POOL)))]
    configs = _feasible_configs(region_rows, region_cols)
    if not configs:
        return None

    horizontal_choices, vertical_choices, size2_components = configs[int(RNG.integers(len(configs)))]
    pair = size2_components[int(RNG.integers(len(size2_components)))]
    target_region = pair[int(RNG.integers(2))]
    source_region = pair[1] if pair[0] == target_region else pair[0]

    heights = RNG.integers(3, 6, size=region_rows)
    widths = RNG.integers(3, 6, size=region_cols)
    row_spans, line_rows, height = _spans_from_lengths(heights)
    col_spans, line_cols, width = _spans_from_lengths(widths)

    grid = np.full((height, width), BG, dtype=int)
    if line_rows:
        grid[np.array(line_rows), :] = WALL
    if line_cols:
        grid[:, np.array(line_cols)] = WALL

    for row_boundary, line_r in enumerate(line_rows):
        c0, c1 = col_spans[horizontal_choices[row_boundary]]
        gap_c = int(RNG.integers(c0, c1 + 1))
        grid[line_r, gap_c] = BG

    for col_boundary, line_c in enumerate(line_cols):
        r0, r1 = row_spans[vertical_choices[col_boundary]]
        gap_r = int(RNG.integers(r0, r1 + 1))
        grid[gap_r, line_c] = BG

    all_regions = [(r, c) for r in range(region_rows) for c in range(region_cols)]
    seeded_regions = [node for node in all_regions if node != target_region]
    color_order = RNG.permutation(np.array(SEED_COLORS, dtype=int))[: len(seeded_regions)]
    color_by_region = {region: int(color) for region, color in zip(seeded_regions, color_order.tolist())}

    for region, color in color_by_region.items():
        r0, r1 = row_spans[region[0]]
        c0, c1 = col_spans[region[1]]
        seed_r = int(RNG.integers(r0, r1 + 1))
        seed_c = int(RNG.integers(c0, c1 + 1))
        grid[seed_r, seed_c] = color

    try:
        info = _analyze_grid(grid)
        if info["target_region"] != target_region:
            return None
        if info["source_region"] != source_region:
            return None
        if int(info["fill_color"]) != color_by_region[source_region]:
            return None
        out = main(grid)
    except AssertionError:
        return None

    tr, tc = target_region
    r0, r1 = row_spans[tr]
    c0, c1 = col_spans[tc]
    expected = color_by_region[source_region]
    if not np.all(out[r0 : r1 + 1, c0 : c1 + 1] == expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
