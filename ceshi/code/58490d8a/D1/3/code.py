# Variant of 58490d8a
# Dimension: D1 — 匹配条件：颜色匹配 (match_cond)
# Change: s.color == d.color — 图例框内的关键像素颜色与框外精灵颜色一致即为匹配组... → 连接器颜色匹配

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, label_mask, paste, random_free_location

# Design analysis
# ----------------
# a) Concrete variant:
#    - The legend still is a black box with one colored marker pixel per tally row.
#    - Outside the legend, each counted source is now a small spatial cluster:
#      several body sprites plus exactly one detached single-pixel connector.
#    - The connector color matches one legend-row marker; the body sprite color does not.
#      Output dots are painted in the legend marker color, so the row-to-cluster match
#      is visible through the connector rather than the sprite body.
#    - Small trace:
#      original: row color 8 -> count all outside 8-components -> draw 8,8,8
#      variant: row color 8 -> find the lone outside 8-pixel connector -> count nearby
#      non-singleton sprites in that connector's cluster -> draw 8,8,8
#      Difference: direct color counting on outside pixels would now only see the single
#      connector pixel, not the number of body sprites.
# b) Complexity calibration:
#    - Each key color appears both in the legend and on exactly one outside connector.
#    - Body colors are sampled independently from a small shared pool, often repeated
#      across different keys, so the connector match is necessary and visually supported.
#    - There is at most one distractor cluster with an unkeyed connector color, mirroring
#      the original task's distractor role without adding extra rule branches.
# c) Original-solver assumptions that break:
#    - `mask = (g == col) & outside` used to isolate the counted sprites. That no longer
#      works because `col` now marks only the single connector pixel, not the bodies.
#    - `label_mask(mask)` used to turn color groups directly into counts. In the variant
#      this would count connectors, usually yielding 1 instead of the true tally.
#    - The legend key pixel used to double as both "match feature" and "output color".
#      Here it remains the output color, but the match feature is now the connector.
# d) Placement legality:
#    - Each outside cluster is built as one reserved rectangular patch containing several
#      disconnected body sprites and one isolated connector pixel.
#    - Patches are placed with rectangle reservation, so clusters do not overlap each
#      other or the legend box, and solver-side nearest-connector grouping stays stable.
# e) Visual observability:
#    - The connector is the only single-pixel object in each cluster, so humans can spot
#      it quickly and match it to the legend color.
#    - Repeated body colors across different groups make it visible that body color alone
#      is not the rule.
# f) Visual cleanliness:
#    - Groups are compact local clusters; the legend remains unchanged and visually
#      separated from sources.
#    - Connectors are isolated by at least one blank cell, so they remain readable as
#      intentional markers rather than noise.
# g) Color inference:
#    - Colors are not sorted or transformed by id.
#    - Connector colors are reused exactly as legend marker colors, giving an explicit
#      perceptual cue for the new match relation.
# h) Variant-effect check target:
#    - On generated inputs, the old solver would count connector-colored singleton
#      components, while the new solver counts body sprites assigned to that connector.
#      Therefore the two solvers differ whenever any target count is not 1.
# i) New solver assumptions to enforce in generation:
#    - Outside singleton components are connectors, never body sprites.
#    - Each counted body sprite is closer to its own connector than to any other connector.
#    - Every keyed connector color appears on exactly one outside connector.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count clustered sprites by connector color and write the tallies into the legend.

    Input structure:
    - A black legend box contains one colored marker pixel on each tally row.
    - Outside the box, each counted source is a compact cluster of same-body-color sprites plus one isolated single-pixel connector.

    Rule:
    - Match each legend row to the outside cluster whose connector pixel has the same color as that row's marker.
    - Count the non-singleton sprites in that connector's cluster, then paint that many evenly spaced dots across the legend row using the marker color.

    Key insight: The relevant outside color is not the sprite body color anymore; the lone connector pixel is the tag that identifies which cluster a legend row should count.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    black = 0

    # STEP 1) Find the legend bbox: largest black CC whose bbox contains some non-black pixels.
    best = None
    for coords in components_mask(g == int(black), connectivity=8):
        if coords.size == 0:
            continue
        b = bbox_coords(coords)
        sub = crop_bbox(g, b)
        if (sub != int(black)).any():
            area = int(coords.shape[0])
            cand = (area, tuple(map(int, b)))
            if best is None or cand[0] > best[0]:
                best = cand

    assert best is not None
    r0, c0, r1, c1 = map(int, best[1])
    box = crop_bbox(g, (r0, c0, r1, c1))
    bh, bw = map(int, box.shape)

    # STEP 2) Read the legend markers from the leftmost non-black column.
    key_col = None
    for cc in range(bw):
        if (box[:, int(cc)] != int(black)).any():
            key_col = int(cc)
            break
    if key_col is None:
        return np.zeros_like(box)

    keys = [(int(rr), int(box[int(rr), int(key_col)])) for rr in range(bh) if int(box[int(rr), int(key_col)]) != int(black)]
    keys.sort(key=lambda t: int(t[0]))

    outside = np.ones((H, W), dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False
    outside_fg = outside & (g != int(bg)) & (g != int(black))

    connectors: list[tuple[int, int, int]] = []
    sprites: list[np.ndarray] = []
    for coords in components_mask(outside_fg, connectivity=8):
        if coords.size == 0:
            continue
        if int(coords.shape[0]) == 1:
            rr, cc = map(int, coords[0])
            connectors.append((rr, cc, int(g[rr, cc])))
        else:
            sprites.append(coords)

    counts_by_color: dict[int, int] = {}
    if connectors:
        conn_arr = np.array([(rr, cc) for rr, cc, _col in connectors], dtype=int)
        conn_cols = [int(col) for _rr, _cc, col in connectors]
        for coords in sprites:
            # Assign each body sprite to the nearest connector using Chebyshev distance.
            diffs = np.abs(coords[:, None, :] - conn_arr[None, :, :])
            dists = np.max(diffs, axis=2).min(axis=0)
            idx = int(np.argmin(dists))
            col = int(conn_cols[idx])
            counts_by_color[col] = int(counts_by_color.get(col, 0)) + 1

    out = np.zeros((bh, bw), dtype=int)
    # STEP 3/4) Draw one marker-colored tally dot per sprite assigned to that connector color.
    for rr, marker_col in keys:
        count = int(counts_by_color.get(int(marker_col), 0))
        for i in range(count):
            cc = int(key_col + 2 * i)
            if 0 <= cc < bw:
                out[int(rr), int(cc)] = int(marker_col)

    return out


# --------------- generator (connector-match variant) ---------------

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

# Body sprites are kept compact so several connector-tagged clusters can fit on the
# original canvas scale without changing the legend mechanics.
SPRITE_AREA_RANGE = (4, 4)
SPRITE_BBOXES = ((2, 2),)
SPRITE_CONNECTIVITY = 8

SINGLE_BODY_POOL_PROB = 0.7
GROUP_RECT_PADDING = 2
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
        _lab, n = label_mask(mask, connectivity=SPRITE_CONNECTIVITY)
        if int(n) == 1:
            masks.append(mask)
    return tuple(masks)


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
    return [_randint(TARGET_COUNT_RANGE) for _ in range(k)]


def _legend_shape(k: int, max_count: int) -> tuple[int, int]:
    bh = KEY_ROW_STEP * k + LEGEND_HEIGHT_TAIL
    bw = TALLY_COL_STEP * max_count + LEGEND_WIDTH_TAIL + _randint(LEGEND_SLACK_RANGE)
    return bh, bw


def _key_row(i: int) -> int:
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _build_expected(marker_colors: list[int], counts: list[int]) -> np.ndarray:
    max_count = max(counts)
    bh, bw = _legend_shape(len(marker_colors), max_count)
    expected = np.full((bh, bw), BLACK, dtype=int)
    for i, (color, count) in enumerate(zip(marker_colors, counts)):
        rr = _key_row(i)
        for j in range(count):
            expected[rr, KEY_COLUMN + TALLY_COL_STEP * j] = int(color)
    return expected


def _build_legend(shape: tuple[int, int], marker_colors: list[int]) -> np.ndarray:
    legend = np.full(shape, BLACK, dtype=int)
    for i, color in enumerate(marker_colors):
        legend[_key_row(i), KEY_COLUMN] = int(color)
    return legend


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

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid connector-match 58490d8a input")
