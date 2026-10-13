import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

SPECIAL_COLORS = {BLACK, RED, GRAY}
PATTERN_COLORS = [BLUE, GREEN, YELLOW, PINK, ORANGE, TEAL, MAROON]
GEN_ATTEMPTS = 400


class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _region_spans(obstacle_cols: Sequence[int], width: int) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    left = 0
    for c in obstacle_cols:
        spans.append((left, c - 1))
        left = c + 1
    spans.append((left, width - 1))
    return spans


def _locate_region(col: int, spans: Sequence[Tuple[int, int]]) -> Optional[int]:
    for idx, (c0, c1) in enumerate(spans):
        if c0 <= col <= c1:
            return idx
    return None


def _find_obstacles(grid: np.ndarray) -> Tuple[List[int], Dict[int, int]]:
    h, w = grid.shape
    obstacle_cols: List[int] = []
    gap_rows: Dict[int, int] = {}
    for c in range(w):
        non_gray_rows = np.flatnonzero(grid[:, c] != GRAY)
        if len(non_gray_rows) != 1:
            continue
        r = int(non_gray_rows[0])
        if int(grid[r, c]) == BLACK:
            obstacle_cols.append(c)
            gap_rows[c] = r
    assert 3 <= len(obstacle_cols) <= 4, "expected 3-4 vertical gray obstacle columns"
    return obstacle_cols, gap_rows


def _crossed_edge_for_control(control_col: int, spans: Sequence[Tuple[int, int]], obstacle_cols: Sequence[int]) -> Optional[int]:
    region = _locate_region(control_col, spans)
    assert region is not None, "control pixel must lie inside a region"
    _, right = spans[region]
    if region < len(obstacle_cols) and control_col == right:
        return region
    return None


def _connected_groups(n_regions: int, edges: Iterable[int]) -> List[List[int]]:
    dsu = DSU(n_regions)
    for edge in edges:
        dsu.union(edge, edge + 1)
    groups: Dict[int, List[int]] = {}
    for region in range(n_regions):
        root = dsu.find(region)
        groups.setdefault(root, []).append(region)
    return list(groups.values())


def _observed_mask(grid: np.ndarray, spans: Sequence[Tuple[int, int]], group: Sequence[int]) -> np.ndarray:
    mask = np.zeros(grid.shape, dtype=bool)
    for region in group:
        c0, c1 = spans[region]
        mask[:, c0 : c1 + 1] = True
    return mask & ~np.isin(grid, list(SPECIAL_COLORS))


def _infer_min_tile(grid: np.ndarray, mask: np.ndarray) -> Tuple[int, int, np.ndarray]:
    coords = np.argwhere(mask)
    assert len(coords) > 0, "expected observed pattern pixels"
    h, w = grid.shape
    candidates: List[Tuple[int, int, int, np.ndarray]] = []
    for pr in range(1, h + 1):
        for pc in range(1, w + 1):
            if pr * pc > len(coords):
                continue
            tile = np.full((pr, pc), -1, dtype=int)
            seen = np.zeros((pr, pc), dtype=bool)
            ok = True
            for r, c in coords:
                rr = int(r % pr)
                cc = int(c % pc)
                value = int(grid[r, c])
                if tile[rr, cc] == -1:
                    tile[rr, cc] = value
                    seen[rr, cc] = True
                elif tile[rr, cc] != value:
                    ok = False
                    break
            if ok and seen.all():
                candidates.append((pr * pc, pr, pc, tile.copy()))
    assert candidates, "no valid periodic tile found"
    min_area = min(item[0] for item in candidates)
    best = [(pr, pc, tile) for area, pr, pc, tile in candidates if area == min_area]
    assert len(best) == 1, "minimal periodic tile must be unique"
    return best[0]


def _render_output_from_groups(
    grid: np.ndarray,
    spans: Sequence[Tuple[int, int]],
    groups: Sequence[Sequence[int]],
    crossed_edges: Sequence[int],
    obstacle_cols: Sequence[int],
    gap_rows: Dict[int, int],
) -> np.ndarray:
    out = grid.copy()
    edge_to_group: Dict[int, Sequence[int]] = {}
    for group in groups:
        group_set = set(group)
        for edge in crossed_edges:
            if edge in group_set and edge + 1 in group_set:
                edge_to_group[edge] = group

    for group in groups:
        mask = _observed_mask(grid, spans, group)
        pr, pc, tile = _infer_min_tile(grid, mask)
        for region in group:
            c0, c1 = spans[region]
            for r in range(out.shape[0]):
                for c in range(c0, c1 + 1):
                    out[r, c] = tile[r % pr, c % pc]
        for edge, edge_group in edge_to_group.items():
            if list(edge_group) != list(group):
                continue
            barrier_col = obstacle_cols[edge]
            gap_row = gap_rows[barrier_col]
            out[gap_row, barrier_col] = tile[gap_row % pr, barrier_col % pc]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect path-linked regions so they share one periodic tile when repairing holes.

    Input structure:
    - The background is black, with 3-4 vertical gray obstacle columns; each gray
      column has exactly one black single-cell gap.
    - The open column intervals between obstacle columns are rectangular pattern
      regions containing periodic non-black colors plus black missing cells.
    - Red control pixels appear on the bottom row inside different regions.

    Rule:
    - A control pixel grows upward through its own region.
    - If that control pixel sits on the region's rightmost column, then at the
      adjacent obstacle's gap row it detours through that single black gap into the
      next region; otherwise it never crosses an obstacle.
    - Regions joined by these crossings form a connected group under transitive
      closure.
    - For each connected group, collect all non-black, non-gray, non-red pixels in
      the group's regions, infer the unique smallest periodic tile consistent with
      their absolute row/column residues, then repaint every cell in those regions
      from that tile.
    - Any obstacle gap actually used by a crossing path is also repainted from the
      same tile; the gray obstacle cells remain unchanged.

    Key insight:
    - The red paths are only a hidden grouping mechanism: they decide which
      separated regions must pool their partial observations before the shared
      periodic tile becomes uniquely recoverable.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h, _w = g.shape
    assert np.any(g[h - 1] == RED), "expected bottom-row red control pixels"

    obstacle_cols, gap_rows = _find_obstacles(g)
    spans = _region_spans(obstacle_cols, g.shape[1])

    control_cols = [int(c) for c in np.flatnonzero(g[h - 1] == RED)]
    assert len(control_cols) >= 2, "expected 2-3 control pixels"
    edges = sorted({edge for c in control_cols if (edge := _crossed_edge_for_control(c, spans, obstacle_cols)) is not None})
    groups = _connected_groups(len(spans), edges)
    out = _render_output_from_groups(g, spans, groups, edges, obstacle_cols, gap_rows)
    out[g == GRAY] = GRAY
    return out


def _primitive_period(tile: np.ndarray) -> Tuple[int, int]:
    h, w = tile.shape
    for pr in range(1, h + 1):
        for pc in range(1, w + 1):
            if h % pr != 0 or w % pc != 0:
                continue
            ok = True
            for r in range(h):
                for c in range(w):
                    if tile[r, c] != tile[r % pr, c % pc]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return pr, pc
    raise AssertionError("unreachable")


def _random_tile(rng: random.Random, min_residues: int) -> Tuple[int, int, np.ndarray]:
    choices = [(2, 2), (2, 3), (3, 2), (3, 3)]
    rng.shuffle(choices)
    for pr, pc in choices:
        if pr * pc < min_residues:
            continue
        for _ in range(80):
            n_colors = rng.choice([2, 3])
            colors = rng.sample(PATTERN_COLORS, n_colors)
            tile = np.array([[rng.choice(colors) for _ in range(pc)] for _ in range(pr)], dtype=int)
            if len(set(map(int, tile.flatten()))) < 2:
                continue
            if _primitive_period(tile) == (pr, pc):
                return pr, pc, tile
    raise RuntimeError("failed to generate a primitive tile")


def _make_layout(rng: random.Random) -> Tuple[int, int, List[int], Dict[int, int], List[Tuple[int, int]]]:
    h = rng.randint(12, 18)
    n_obstacles = rng.choice([3, 4])
    widths = [rng.randint(4, 6) for _ in range(n_obstacles + 1)]
    w = sum(widths) + n_obstacles
    if w > 30:
        return _make_layout(rng)
    obstacle_cols: List[int] = []
    cursor = widths[0]
    for idx in range(n_obstacles):
        obstacle_cols.append(cursor + idx)
        cursor += widths[idx + 1]
    gap_rows = {c: rng.randint(2, h - 3) for c in obstacle_cols}
    spans = _region_spans(obstacle_cols, w)
    return h, w, obstacle_cols, gap_rows, spans


def _choose_controls(
    rng: random.Random,
    spans: Sequence[Tuple[int, int]],
    obstacle_cols: Sequence[int],
) -> Tuple[List[int], List[int]]:
    n_regions = len(spans)
    chain_len = 2 if n_regions >= 4 else 1
    max_start = n_regions - chain_len - 1
    chain_start = rng.randint(0, max_start)
    edges = list(range(chain_start, chain_start + chain_len))

    controls = [spans[edge][1] for edge in edges]
    used_regions = set(edges)

    extra_regions = [r for r in range(n_regions) if r not in used_regions]
    if rng.random() < 0.6 and extra_regions:
        region = rng.choice(extra_regions)
        c0, c1 = spans[region]
        interior = [c for c in range(c0, c1 + 1) if c != c1 or region == len(obstacle_cols)]
        if interior:
            controls.append(rng.choice(interior))
            used_regions.add(region)

    if len(set(_locate_region(c, spans) for c in controls)) < len(controls):
        return _choose_controls(rng, spans, obstacle_cols)
    return controls, edges


def _render_full_pattern(
    h: int,
    w: int,
    obstacle_cols: Sequence[int],
    gap_rows: Dict[int, int],
    spans: Sequence[Tuple[int, int]],
    region_tiles: Sequence[Tuple[int, int, np.ndarray]],
    crossed_edges: Sequence[int],
) -> np.ndarray:
    out = np.full((h, w), BLACK, dtype=int)
    for c in obstacle_cols:
        out[:, c] = GRAY
        out[gap_rows[c], c] = BLACK
    for region, (pr, pc, tile) in enumerate(region_tiles):
        c0, c1 = spans[region]
        for r in range(h):
            for c in range(c0, c1 + 1):
                out[r, c] = tile[r % pr, c % pc]
    for edge in crossed_edges:
        barrier_col = obstacle_cols[edge]
        gap_row = gap_rows[barrier_col]
        pr, pc, tile = region_tiles[edge]
        out[gap_row, barrier_col] = tile[gap_row % pr, barrier_col % pc]
    return out


def _apply_holes(
    rng: random.Random,
    full: np.ndarray,
    spans: Sequence[Tuple[int, int]],
    groups: Sequence[Sequence[int]],
    region_tiles: Sequence[Tuple[int, int, np.ndarray]],
    controls: Sequence[int],
) -> np.ndarray:
    inp = full.copy()
    control_set = set((inp.shape[0] - 1, c) for c in controls)

    for group in groups:
        if len(group) == 1:
            region = group[0]
            pr, pc, _tile = region_tiles[region]
            c0, c1 = spans[region]
            keep_one: Dict[Tuple[int, int], Tuple[int, int]] = {}
            for r in range(inp.shape[0]):
                for c in range(c0, c1 + 1):
                    residue = (r % pr, c % pc)
                    keep_one.setdefault(residue, (r, c))
            for r in range(inp.shape[0]):
                for c in range(c0, c1 + 1):
                    if (r, c) in control_set:
                        continue
                    residue = (r % pr, c % pc)
                    if keep_one[residue] == (r, c):
                        continue
                    if rng.random() < 0.15:
                        inp[r, c] = BLACK
            continue

        residues = [(rr, cc) for rr in range(region_tiles[group[0]][0]) for cc in range(region_tiles[group[0]][1])]
        missing: Dict[int, set] = {region: set() for region in group}
        for idx, region in enumerate(group):
            missing[region].add(residues[idx % len(residues)])
            if len(residues) >= 2 * len(group):
                missing[region].add(residues[(idx + len(group)) % len(residues)])

        for region in group:
            pr, pc, _tile = region_tiles[region]
            c0, c1 = spans[region]
            for r in range(inp.shape[0]):
                for c in range(c0, c1 + 1):
                    if (r, c) in control_set:
                        continue
                    residue = (r % pr, c % pc)
                    if residue in missing[region]:
                        inp[r, c] = BLACK
                    elif rng.random() < 0.08:
                        inp[r, c] = BLACK
    return inp


def _build_region_tiles(
    rng: random.Random,
    groups: Sequence[Sequence[int]],
    n_regions: int,
) -> List[Tuple[int, int, np.ndarray]]:
    region_tiles: List[Optional[Tuple[int, int, np.ndarray]]] = [None] * n_regions
    used_signatures = set()
    for group in groups:
        pr, pc, tile = _random_tile(rng, max(4, len(group) + 1))
        signature = (pr, pc, tuple(map(int, tile.flatten())))
        while signature in used_signatures:
            pr, pc, tile = _random_tile(rng, max(4, len(group) + 1))
            signature = (pr, pc, tuple(map(int, tile.flatten())))
        used_signatures.add(signature)
        for region in group:
            region_tiles[region] = (pr, pc, tile.copy())
    return [tile for tile in region_tiles if tile is not None]


def _all_groups_from_edges(n_regions: int, edges: Sequence[int]) -> List[List[int]]:
    return [sorted(group) for group in _connected_groups(n_regions, edges)]


def _independent_output(grid: np.ndarray, spans: Sequence[Tuple[int, int]]) -> Optional[np.ndarray]:
    out = grid.copy()
    for region, span in enumerate(spans):
        mask = _observed_mask(grid, spans, [region])
        try:
            pr, pc, tile = _infer_min_tile(grid, mask)
        except AssertionError:
            return None
        c0, c1 = span
        for r in range(out.shape[0]):
            for c in range(c0, c1 + 1):
                out[r, c] = tile[r % pr, c % pc]
    out[grid == GRAY] = GRAY
    return out


def _validate_instance(inp: np.ndarray, expected: np.ndarray, crossed_edges: Sequence[int]) -> bool:
    h, _w = inp.shape
    if len(set((int(c)) for c in np.flatnonzero(inp[h - 1] == RED))) < 2:
        return False

    obstacle_cols, gap_rows = _find_obstacles(inp)
    spans = _region_spans(obstacle_cols, inp.shape[1])
    groups = _all_groups_from_edges(len(spans), crossed_edges)
    if not any(len(group) > 1 for group in groups):
        return False
    if not any(len(group) == 1 for group in groups):
        return False

    if not np.array_equal(main(inp), expected):
        return False

    multi_groups = [group for group in groups if len(group) > 1]
    if not multi_groups:
        return False

    informative_group = False
    for group in multi_groups:
        group_mask = _observed_mask(inp, spans, group)
        try:
            _infer_min_tile(inp, group_mask)
        except AssertionError:
            return False
        if all(_infer_region_fails(inp, spans, region) for region in group):
            informative_group = True
    if not informative_group:
        return False

    independent = _independent_output(inp, spans)
    if independent is not None and np.array_equal(independent, expected):
        return False
    return True


def _infer_region_fails(inp: np.ndarray, spans: Sequence[Tuple[int, int]], region: int) -> bool:
    try:
        _infer_min_tile(inp, _observed_mask(inp, spans, [region]))
    except AssertionError:
        return True
    return False


def _gen_attempt(rng: random.Random) -> Optional[np.ndarray]:
    h, w, obstacle_cols, gap_rows, spans = _make_layout(rng)
    controls, crossed_edges = _choose_controls(rng, spans, obstacle_cols)
    groups = _all_groups_from_edges(len(spans), crossed_edges)
    if not any(len(group) > 1 for group in groups) or not any(len(group) == 1 for group in groups):
        return None

    region_tiles = _build_region_tiles(rng, groups, len(spans))
    full = _render_full_pattern(h, w, obstacle_cols, gap_rows, spans, region_tiles, crossed_edges)
    inp = full.copy()
    for c in obstacle_cols:
        inp[gap_rows[c], c] = BLACK
    inp = _apply_holes(rng, inp, spans, groups, region_tiles, controls)
    for c in controls:
        inp[h - 1, c] = RED
    expected = full.copy()

    if np.array_equal(inp, expected):
        return None
    if not _validate_instance(inp, expected, crossed_edges):
        return None
    return inp


def generate_input() -> np.ndarray:
    seed = random.randrange(1 << 30)
    rng = random.Random(seed)
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid path-connected periodic-tile puzzle")
