# Combo variant of 58490d8a (index 31)
# combo_spec: D1/3+D14/0
# Name: 连接器匹配+排序重排

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, label_mask, paste, random_free_location


# Design analysis
# ----------------
# a) Concrete fused behavior and trace
#    - The legend is still a black rectangle with one colored marker pixel on every
#      other row. Those marker colors define both the connector tags to look for and
#      the colors used when drawing the final tally dots.
#    - Outside the legend, each counted source is no longer "all objects of the same
#      visible color". Instead, each source is a compact cluster containing:
#        1. several disconnected body sprites of one body color
#        2. exactly one detached single-pixel connector whose color matches one legend row
#    - Example trace:
#        input legend rows top->bottom = [red, teal, yellow]
#        connector-tagged cluster body counts = [2, 5, 3]
#        original task would try to count red/teal/yellow outside objects directly
#        fused task first matches clusters by connector color, obtaining counts [2, 5, 3]
#        then sorts rows by count descending, so output rows become [teal x5, yellow x3, red x2]
#      Difference 1: body color is not the matching key anymore.
#      Difference 2: even after counting, the row order is not preserved.
#
# b) Complexity calibration
#    - Body colors are sampled from a small pool disjoint from the legend colors, and
#      are often reused across different clusters. This makes direct color matching fail
#      visibly and forces the connector step.
#    - Counts are sampled as distinct values and the input legend order is rejected if it
#      is already descending by count. This keeps the reorder stage visible instead of
#      collapsing back to the single-variant behavior.
#    - There is at most one distractor cluster with an unkeyed connector color, which
#      adds pressure on the connector-matching rule without introducing many branches.
#
# c) Original-solver assumptions that break
#    - `mask = (g == col) & outside` no longer isolates the counted bodies, because a
#      legend color only appears on the single connector, not on the bodies.
#    - `label_mask(mask)` no longer gives the desired tally; it would usually return 1
#      for each keyed legend color because it sees the connector only.
#    - The original drawing loop reused the input row immediately. That breaks after
#      sorting: row identity must be separated from `(marker color, count)` records.
#    - Combining both changes means the solver must now do:
#        detect legend -> detect connectors/body sprites -> assign sprites to connectors
#        -> count per connector color -> sort records -> redraw into ranked row slots
#
# d) Placement legality
#    - Each outside group is generated as one reserved rectangular patch. The patch holds
#      one isolated connector and up to five disconnected 2x2 body sprites.
#    - Patches are placed by reserving their full rectangles with extra padding, so groups
#      do not overlap each other or the legend box and remain inside bounds.
#    - Generator acceptance runs the final solver and rejects any sample where the intended
#      counts are not recovered, so nearest-connector grouping is not left to chance.
#
# e) Visual observability
#    - The only singleton outside components are connectors, making them easy to identify.
#    - Because body colors repeat and do not match legend colors, a viewer can see that
#      body color alone cannot explain the output.
#    - Sorting is based on visibly different tally lengths, not on hidden color ids.
#
# f) Visual cleanliness
#    - The legend region stays visually simple: just black rows with marker/tally dots.
#    - Outside objects appear as compact, well-spaced local clusters rather than being
#      scattered everywhere, so "which connector belongs to which nearby bodies" stays readable.
#
# g) Change inferability
#    - Colors are not numerically transformed. A marker color reappears exactly as the
#      connector color and as the final tally color.
#    - The only row motion is rank-by-count motion. The same legend colors move together
#      with their tally lengths, which provides the structural cue needed to infer sorting.
#
# h) Anti-degeneracy targets for generation
#    - Connector behavior must matter: the fused output should differ from a solver that
#      merely sorts direct same-color counts.
#    - Sorting behavior must matter: the fused output should differ from a solver that
#      uses connector matching but keeps the original legend-row order.
#
# i) New solver assumptions to enforce in generation
#    - Every counted body sprite is a non-singleton connected component.
#    - Every keyed connector color appears on exactly one outside singleton connector.
#    - Distinct group counts fit within the legend width and are not already descending
#      in the input legend order.


COLOR_POOL = tuple(range(1, 10))
BLACK = 0
DISTRACTOR_PROB = 0.5

KEY_COUNT_RANGE = (2, 4)
TARGET_COUNT_RANGE = (1, 5)
KEY_ROW_FIRST = 1
KEY_ROW_STEP = 2
KEY_COLUMN = 1
TALLY_COL_STEP = 2
LEGEND_HEIGHT_TAIL = 1
LEGEND_WIDTH_TAIL = 1
LEGEND_SLACK_RANGE = (0, 3)

GRID_HEIGHT_RANGE = (20, 25)
GRID_WIDTH_RANGE = (20, 25)
LEGEND_EDGE_MARGIN = 2

SPRITE_AREA_RANGE = (4, 4)
SPRITE_BBOXES = ((2, 2),)
SPRITE_CONNECTIVITY = 8

SINGLE_BODY_POOL_PROB = 0.7
GROUP_RECT_PADDING = 3
GENERATION_ATTEMPTS = 300

GROUP_SLOTS = ((1, 1), (0, 0), (0, 2), (1, 0), (1, 2))
CONNECTOR_SLOT = (0, 1)


def _randint(bounds: tuple[int, int]) -> int:
    return random.randint(bounds[0], bounds[1])


@lru_cache(maxsize=None)
def _valid_masks(h: int, w: int, area: int) -> tuple[np.ndarray, ...]:
    masks = []
    for cells in combinations(range(h * w), area):
        mask = np.zeros((h, w), dtype=bool)
        for cell in cells:
            mask[cell // w, cell % w] = True
        spans_edges = mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()
        if not spans_edges:
            continue
        _labels, n = label_mask(mask, connectivity=SPRITE_CONNECTIVITY)
        if int(n) == 1:
            masks.append(mask)
    return tuple(masks)


def _find_legend(grid: np.ndarray) -> tuple[int, int, int, int]:
    best = None
    for coords in components_mask(grid == int(BLACK), connectivity=8):
        if coords.size == 0:
            continue
        bbox = bbox_coords(coords)
        sub = crop_bbox(grid, bbox)
        if (sub != int(BLACK)).any():
            area = int(coords.shape[0])
            cand = (area, tuple(map(int, bbox)))
            if best is None or area > best[0]:
                best = cand
    assert best is not None
    return tuple(map(int, best[1]))


def _read_legend_keys(box: np.ndarray) -> tuple[int | None, list[tuple[int, int]]]:
    bh, bw = map(int, box.shape)
    key_col = None
    for cc in range(bw):
        if (box[:, int(cc)] != int(BLACK)).any():
            key_col = int(cc)
            break
    if key_col is None:
        return None, []
    keys = [(int(rr), int(box[int(rr), int(key_col)])) for rr in range(bh) if int(box[int(rr), int(key_col)]) != int(BLACK)]
    keys.sort(key=lambda t: int(t[0]))
    return key_col, keys


def _outside_connectors_and_sprites(
    grid: np.ndarray,
    outside: np.ndarray,
    bg: int,
) -> tuple[list[tuple[int, int, int]], list[np.ndarray]]:
    outside_fg = outside & (grid != int(bg)) & (grid != int(BLACK))
    connectors: list[tuple[int, int, int]] = []
    sprites: list[np.ndarray] = []
    for coords in components_mask(outside_fg, connectivity=8):
        if coords.size == 0:
            continue
        if int(coords.shape[0]) == 1:
            rr, cc = map(int, coords[0])
            connectors.append((rr, cc, int(grid[rr, cc])))
        else:
            sprites.append(coords)
    return connectors, sprites


def _counts_by_connector_color(grid: np.ndarray) -> tuple[list[tuple[int, int]], int | None, dict[int, int], tuple[int, int]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    r0, c0, r1, c1 = _find_legend(g)
    box = crop_bbox(g, (r0, c0, r1, c1))
    bh, bw = map(int, box.shape)
    key_col, keys = _read_legend_keys(box)
    if key_col is None:
        return [], None, {}, (bh, bw)

    outside = np.ones((H, W), dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False
    connectors, sprites = _outside_connectors_and_sprites(g, outside, bg)

    counts_by_color: dict[int, int] = {}
    if connectors:
        conn_arr = np.array([(rr, cc) for rr, cc, _col in connectors], dtype=int)
        conn_cols = [int(col) for _rr, _cc, col in connectors]
        for coords in sprites:
            diffs = np.abs(coords[:, None, :] - conn_arr[None, :, :])
            dists = np.max(diffs, axis=2).min(axis=0)
            idx = int(np.argmin(dists))
            connector_color = int(conn_cols[idx])
            counts_by_color[connector_color] = int(counts_by_color.get(connector_color, 0)) + 1

    return keys, key_col, counts_by_color, (bh, bw)


def _sorted_records(keys: list[tuple[int, int]], counts_by_color: dict[int, int]) -> list[tuple[int, int, int]]:
    records = [(int(rr), int(color), int(counts_by_color.get(int(color), 0))) for rr, color in keys]
    records.sort(key=lambda t: (-int(t[2]), int(t[0])))
    return records


def _draw_tally(bh: int, bw: int, key_col: int, slot_rows: list[int], records: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.zeros((bh, bw), dtype=int)
    for new_idx, (_orig_rr, color, count) in enumerate(records):
        rr = int(slot_rows[new_idx])
        for i in range(int(count)):
            cc = int(key_col + TALLY_COL_STEP * i)
            if 0 <= cc < bw:
                out[int(rr), int(cc)] = int(color)
    return out


def _solve_connector_unsorted(grid: np.ndarray) -> np.ndarray:
    keys, key_col, counts_by_color, (bh, bw) = _counts_by_connector_color(grid)
    if key_col is None:
        return np.zeros((bh, bw), dtype=int)
    records = [(int(rr), int(color), int(counts_by_color.get(int(color), 0))) for rr, color in keys]
    slot_rows = [int(rr) for rr, _color in keys]
    return _draw_tally(bh, bw, int(key_col), slot_rows, records)


def _solve_direct_sorted(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    r0, c0, r1, c1 = _find_legend(g)
    box = crop_bbox(g, (r0, c0, r1, c1))
    bh, bw = map(int, box.shape)
    key_col, keys = _read_legend_keys(box)
    if key_col is None:
        return np.zeros((bh, bw), dtype=int)

    outside = np.ones((H, W), dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False
    records: list[tuple[int, int, int]] = []
    for rr, color in keys:
        mask = (g == int(color)) & outside
        _labels, count = label_mask(mask, connectivity=8)
        records.append((int(rr), int(color), int(count)))
    records.sort(key=lambda t: (-int(t[2]), int(t[0])))
    slot_rows = [int(rr) for rr, _color in keys]
    return _draw_tally(bh, bw, int(key_col), slot_rows, records)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group outside sprites by connector color, then rank the legend rows by those counts.

    Input structure:
    - A black legend rectangle contains one colored marker pixel on each tally row.
    - Outside the legend, each countable source is a compact cluster made of several detached body sprites plus one isolated single-pixel connector.

    Rule:
    - Match each legend row to the outside cluster whose connector pixel has the same color as that row's marker, and count the non-singleton body sprites assigned to that connector.
    - Sort the legend rows by those counts from largest to smallest, breaking ties by the original top-to-bottom legend order, then draw evenly spaced tally dots in the ranked row slots using the marker color.

    Key insight: The legend markers do not match body colors directly and they do not stay on their original rows; they first identify connector-tagged clusters, then the resulting counts determine a second-stage row permutation.
    """
    g = np.array(grid, dtype=int)
    keys, key_col, counts_by_color, (bh, bw) = _counts_by_connector_color(g)
    if key_col is None:
        return np.zeros((bh, bw), dtype=int)
    slot_rows = [int(rr) for rr, _color in keys]
    records = _sorted_records(keys, counts_by_color)
    return _draw_tally(bh, bw, int(key_col), slot_rows, records)


def _sample_palette() -> tuple[int, list[int], list[int], int | None]:
    colors = list(COLOR_POOL)
    bg = random.choice(colors)

    k = _randint(KEY_COUNT_RANGE)
    remaining = [c for c in colors if c != bg]
    marker_colors = random.sample(remaining, k)
    remaining = [c for c in remaining if c not in marker_colors]

    pool_size = 1 if random.random() < SINGLE_BODY_POOL_PROB else min(2, len(remaining))
    pool_size = max(1, pool_size)
    body_pool = random.sample(remaining, pool_size)
    body_colors = [random.choice(body_pool) for _ in range(k)]

    leftover = [c for c in remaining if c not in body_pool]
    distractor_marker = random.choice(leftover) if leftover and random.random() < DISTRACTOR_PROB else None
    return bg, marker_colors, body_colors, distractor_marker


def _sample_counts(k: int) -> list[int]:
    choices = list(range(TARGET_COUNT_RANGE[0], TARGET_COUNT_RANGE[1] + 1))
    counts = random.sample(choices, k)
    while counts == sorted(counts, reverse=True):
        random.shuffle(counts)
    return [int(v) for v in counts]


def _legend_shape(k: int, max_count: int) -> tuple[int, int]:
    bh = KEY_ROW_STEP * k + LEGEND_HEIGHT_TAIL
    bw = TALLY_COL_STEP * max_count + LEGEND_WIDTH_TAIL + _randint(LEGEND_SLACK_RANGE)
    return bh, bw


def _key_row(i: int) -> int:
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _sorted_key_records(marker_colors: list[int], counts: list[int]) -> list[tuple[int, int, int]]:
    records = [(int(i), int(color), int(count)) for i, (color, count) in enumerate(zip(marker_colors, counts))]
    records.sort(key=lambda t: (-int(t[2]), int(t[0])))
    return records


def _build_expected(marker_colors: list[int], counts: list[int]) -> np.ndarray:
    max_count = max(counts)
    bh, bw = _legend_shape(len(marker_colors), max_count)
    expected = np.full((bh, bw), BLACK, dtype=int)
    for out_i, (_orig_i, color, count) in enumerate(_sorted_key_records(marker_colors, counts)):
        rr = _key_row(out_i)
        for j in range(count):
            expected[rr, KEY_COLUMN + TALLY_COL_STEP * j] = int(color)
    return expected


def _build_legend(shape: tuple[int, int], marker_colors: list[int]) -> np.ndarray:
    legend = np.full(shape, BLACK, dtype=int)
    for i, color in enumerate(marker_colors):
        legend[_key_row(i), KEY_COLUMN] = int(color)
    return legend


def _make_sprite(color: int, bg: int) -> np.ndarray:
    area = _randint(SPRITE_AREA_RANGE)
    dims = []
    for h, w in SPRITE_BBOXES:
        if h * w >= area and _valid_masks(h, w, area):
            dims.append((h, w))
    h, w = random.choice(dims)
    mask = random.choice(_valid_masks(h, w, area))
    sprite = np.full((h, w), int(bg), dtype=int)
    sprite[mask] = int(color)
    return sprite


def _build_group_patch(connector_color: int, body_color: int, count: int, bg: int) -> np.ndarray:
    sprite = _make_sprite(body_color, bg)
    sh, sw = map(int, sprite.shape)
    cell_h, cell_w = int(sh + 1), int(sw + 1)

    patch = np.full((2 * cell_h, 3 * cell_w), int(bg), dtype=int)
    for slot_r, slot_c in GROUP_SLOTS[:count]:
        paste(patch, sprite, int(slot_r * cell_h), int(slot_c * cell_w), transparent=bg)

    conn_r = int(CONNECTOR_SLOT[0] * cell_h + sh // 2)
    conn_c = int(CONNECTOR_SLOT[1] * cell_w + sw // 2)
    patch[conn_r, conn_c] = int(connector_color)

    nz = np.argwhere(patch != int(bg))
    r0, c0, r1, c1 = bbox_coords(nz)
    return crop_bbox(patch, (int(r0), int(c0), int(r1), int(c1)))


def _new_grid_with_legend(bg: int, legend: np.ndarray, group_patches: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    bh, bw = map(int, legend.shape)
    legend_area = int(bh * bw)
    patch_area = int(sum(int(p.shape[0]) * int(p.shape[1]) for p in group_patches))

    for _ in range(20):
        H = _randint(GRID_HEIGHT_RANGE)
        W = _randint(GRID_WIDTH_RANGE)
        if H < bh + 2 * LEGEND_EDGE_MARGIN or W < bw + 2 * LEGEND_EDGE_MARGIN:
            continue
        if H * W < legend_area + patch_area + 20:
            continue

        grid = np.full((H, W), int(bg), dtype=int)
        occ = np.zeros((H, W), dtype=int)
        r0 = random.randint(LEGEND_EDGE_MARGIN, H - bh - LEGEND_EDGE_MARGIN)
        c0 = random.randint(LEGEND_EDGE_MARGIN, W - bw - LEGEND_EDGE_MARGIN)
        paste(grid, legend, r0, c0)
        occ[int(r0) : int(r0) + bh, int(c0) : int(c0) + bw] = 1
        return grid, occ
    return None


def _place_group_patches(grid: np.ndarray, occ: np.ndarray, group_patches: list[np.ndarray], bg: int) -> None:
    ordered = sorted(group_patches, key=lambda p: int(p.shape[0]) * int(p.shape[1]), reverse=True)
    for patch in ordered:
        reserve = np.ones(patch.shape, dtype=int)
        sr, sc = random_free_location(
            occ,
            reserve,
            transparent=0,
            padding=GROUP_RECT_PADDING,
            padding_connectivity=8,
        )
        paste(occ, reserve, sr, sc)
        paste(grid, patch, sr, sc, transparent=bg)


def _gen_attempt() -> np.ndarray | None:
    bg, marker_colors, body_colors, distractor_marker = _sample_palette()
    counts = _sample_counts(len(marker_colors))
    expected = _build_expected(marker_colors, counts)
    legend = _build_legend(expected.shape, marker_colors)

    group_patches = [
        _build_group_patch(connector_color=marker, body_color=body, count=count, bg=bg)
        for marker, body, count in zip(marker_colors, body_colors, counts)
    ]
    if distractor_marker is not None:
        distractor_body = random.choice(body_colors)
        distractor_count = _randint((1, 3))
        group_patches.append(
            _build_group_patch(
                connector_color=distractor_marker,
                body_color=distractor_body,
                count=distractor_count,
                bg=bg,
            )
        )

    base = _new_grid_with_legend(bg, legend, group_patches)
    if base is None:
        return None
    grid, occ = base

    try:
        _place_group_patches(grid, occ, group_patches, bg)
    except ValueError:
        return None

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(_solve_connector_unsorted(grid), expected):
        return None
    if np.array_equal(_solve_direct_sorted(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid combo 58490d8a input")
