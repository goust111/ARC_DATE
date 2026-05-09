# Combo variant of 58490d8a (index 8)
# combo_spec: D1/3+D7/3
# Name: 连接器匹配+图例内干扰行

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, label_mask, paste, random_free_location

# Design analysis
# ----------------
# a) Concrete fused behavior:
#    - The legend remains one black box with one shared marker column, but now every colored
#      row is only a candidate row, not automatically a true key.
#    - Outside the box, each candidate row color appears on exactly one detached single-pixel
#      connector. Nearby to that connector sits a compact patch containing 1-5 body sprites.
#    - True rows have one connector-matched cluster; distractor rows have no matching outside
#      connector at all and therefore produce no dots.
#    - Trace:
#      original task: row color 8 -> count outside color-8 components -> draw 3 dots.
#      D1 only: row color 8 -> find outside connector color 8 -> count nearby body sprites -> draw 3 dots.
#      fused task: row color 8 -> find outside connector color 8 -> count nearby body sprites;
#      if no same-colored connector exists, the row is fake and stays black.
#      Difference: the solver must first recover connector grouping, then reject extra legend rows.
# b) Complexity calibration:
#    - Distractor rows are mandatory (1-2 rows) so the legend really mixes valid and invalid rows.
#    - True counts span 1-5 exactly like the base task, while distractor rows are ruled out only by
#      the missing connector, not by a hidden threshold on the count.
#    - Body colors are drawn from a tiny shared pool and often repeat across different connectors, so
#      body color alone is not enough to solve the task.
# c) Implicit-assumption review against the original solver:
#    - "Legend row color equals outside sprite color" is broken: legend colors now match only connectors.
#    - "Every colored legend row is valid" is broken: some rows intentionally have no outside match.
#    - "Counting color-connected components is enough" is broken twice: it would count connectors instead
#      of bodies, and it would also keep fake legend rows that have no outside match.
#    - The remaining safe assumptions are preserved: one legend box, one shared marker column, odd-row
#      spacing, and black output canvas with every-other-column tally spacing.
# d) Placement legality:
#    - Each outside cluster is rendered as one compact patch with disconnected body sprites and one
#      isolated connector pixel; patches are rectangle-reserved so they do not overlap or touch.
#    - The legend bbox is reserved in the occupancy map, so outside patches stay separated from it.
# e) Visual observability:
#    - The connector is the only singleton object in each cluster, so humans can identify it as a tag.
#    - Reused body colors across multiple clusters make the connector-based match visible.
#    - Because at least one legend row has no same-colored connector outside, the "fake row" idea is
#      observable from a few examples rather than hidden in edge cases.
# f) Visual cleanliness:
#    - The legend keeps the original tidy odd-row rhythm.
#    - Outside information is localized into compact cluster patches instead of scattered ambiguous noise.
# g) Color inference:
#    - No sorting or arithmetic on color ids is used.
#    - Connector colors are copied directly into legend markers, providing an explicit visible cue for
#      which outside cluster a row should inspect.
# h) Fusion-effect check target:
#    - Compared on the same generated input, the original solver would mostly count connector singletons
#      or miss repeated bodies entirely, while the D1-only solver without legend filtering would still
#      incorrectly keep the extra rows. The fused solver differs from both baselines when generation
#      works as intended.
# i) New solver assumptions enforced by the generator:
#    - Every outside singleton component is a connector, never a body sprite.
#    - Each real connector color appears on exactly one outside connector and exactly one legend row.
#    - Each body sprite is closer to its own connector than to any other connector.
#    - Fake legend rows have no same-colored outside connector at all.

COLOR_POOL = tuple(range(1, 10))
BLACK = 0

KEY_COUNT_RANGE = (2, 4)
DISTRACTOR_ROW_COUNT_RANGE = (1, 2)
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

SINGLE_BODY_POOL_PROB = 0.75
GROUP_RECT_PADDING = 2
GENERATION_ATTEMPTS = 500

GROUP_SLOTS = ((1, 1), (0, 0), (0, 2), (1, 0), (1, 2))
CONNECTOR_SLOT = (0, 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count connector-tagged sprite clusters and keep only the real legend rows.

    Input structure:
    - A black legend box contains one shared marker column with several colored candidate rows.
    - Outside the box, each candidate color appears on exactly one isolated connector pixel beside a compact cluster of body sprites.

    Rule:
    - Match a legend row to the outside cluster whose connector pixel has the same color as that row's marker.
    - Count the non-singleton body sprites in that matched cluster and draw that many evenly spaced tally dots in the marker color.
    - Rows whose marker color has no outside connector are distractors and stay black.

    Key insight: A colored legend row is not automatically valid anymore; you must use the connector to find its cluster first, and some legend rows are fake because no same-colored connector exists outside.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    black = 0

    best = None
    for coords in components_mask(g == int(black), connectivity=8):
        if coords.size == 0:
            continue
        bbox = bbox_coords(coords)
        sub = crop_bbox(g, bbox)
        if (sub != int(black)).any():
            area = int(coords.shape[0])
            cand = (area, tuple(map(int, bbox)))
            if best is None or cand[0] > best[0]:
                best = cand

    assert best is not None
    r0, c0, r1, c1 = map(int, best[1])
    box = crop_bbox(g, (r0, c0, r1, c1))
    bh, bw = map(int, box.shape)

    key_col = None
    for cc in range(bw):
        if (box[:, int(cc)] != int(black)).any():
            key_col = int(cc)
            break
    if key_col is None:
        return np.zeros_like(box)

    candidates = [(int(rr), int(box[int(rr), int(key_col)])) for rr in range(bh) if int(box[int(rr), int(key_col)]) != int(black)]
    candidates.sort(key=lambda item: int(item[0]))

    outside = np.ones((H, W), dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False
    outside_fg = outside & (g != int(bg)) & (g != int(black))

    connectors: list[tuple[int, int, int]] = []
    bodies: list[np.ndarray] = []
    for coords in components_mask(outside_fg, connectivity=8):
        if coords.size == 0:
            continue
        if int(coords.shape[0]) == 1:
            rr, cc = map(int, coords[0])
            connectors.append((rr, cc, int(g[rr, cc])))
        else:
            bodies.append(coords)

    counts_by_connector: dict[int, int] = {int(col): 0 for _rr, _cc, col in connectors}
    if connectors:
        connector_points = np.array([(rr, cc) for rr, cc, _col in connectors], dtype=int)
        connector_colors = [int(col) for _rr, _cc, col in connectors]
        for coords in bodies:
            diffs = np.abs(coords[:, None, :] - connector_points[None, :, :])
            dists = np.max(diffs, axis=2).min(axis=0)
            idx = int(np.argmin(dists))
            color = int(connector_colors[idx])
            counts_by_connector[color] = int(counts_by_connector.get(color, 0)) + 1

    out = np.zeros((bh, bw), dtype=int)
    for rr, marker_color in candidates:
        if int(marker_color) not in counts_by_connector:
            continue
        count = int(counts_by_connector[int(marker_color)])
        for i in range(count):
            cc = int(key_col + TALLY_COL_STEP * i)
            if 0 <= cc < bw:
                out[int(rr), int(cc)] = int(marker_color)

    return out


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


def _sample_palette() -> tuple[int, list[int], list[int], list[int]]:
    colors = list(COLOR_POOL)
    bg = random.choice(colors)

    real_count = _randint(KEY_COUNT_RANGE)
    distractor_count = _randint(DISTRACTOR_ROW_COUNT_RANGE)
    remaining = [c for c in colors if c != bg]
    connector_colors = random.sample(remaining, real_count + distractor_count)
    real_connector_colors = connector_colors[:real_count]
    distractor_connector_colors = connector_colors[real_count:]

    leftover = [c for c in remaining if c not in connector_colors]
    pool_size = 1 if random.random() < SINGLE_BODY_POOL_PROB else min(2, len(leftover))
    pool_size = max(1, pool_size)
    body_pool = random.sample(leftover, pool_size)
    return bg, real_connector_colors, distractor_connector_colors, body_pool


def _sample_counts(k: int) -> list[int]:
    return [_randint(TARGET_COUNT_RANGE) for _ in range(k)]


def _legend_shape(total_rows: int, max_count: int) -> tuple[int, int]:
    bh = KEY_ROW_STEP * total_rows + LEGEND_HEIGHT_TAIL
    bw = TALLY_COL_STEP * max_count + LEGEND_WIDTH_TAIL + _randint(LEGEND_SLACK_RANGE)
    return bh, bw


def _key_row(i: int) -> int:
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _row_specs(
    real_connector_colors: list[int],
    real_counts: list[int],
    distractor_connector_colors: list[int],
    body_pool: list[int],
) -> list[dict[str, int]]:
    total = len(real_connector_colors) + len(distractor_connector_colors)
    slots = list(range(total))
    random.shuffle(slots)

    specs: list[dict[str, int]] = []
    cursor = 0
    for color, count in zip(real_connector_colors, real_counts):
        specs.append(
            {
                "row": _key_row(slots[cursor]),
                "marker": int(color),
                "body": int(random.choice(body_pool)),
                "count": int(count),
                "is_real": 1,
            }
        )
        cursor += 1
    for color in distractor_connector_colors:
        specs.append(
            {
                "row": _key_row(slots[cursor]),
                "marker": int(color),
                "body": int(random.choice(body_pool)),
                "count": 0,
                "is_real": 0,
            }
        )
        cursor += 1

    specs.sort(key=lambda item: int(item["row"]))
    return specs


def _build_expected(shape: tuple[int, int], row_specs: list[dict[str, int]]) -> np.ndarray:
    expected = np.full(shape, BLACK, dtype=int)
    for spec in row_specs:
        count = int(spec["count"])
        if not int(spec["is_real"]):
            continue
        rr = int(spec["row"])
        color = int(spec["marker"])
        for j in range(count):
            expected[rr, KEY_COLUMN + TALLY_COL_STEP * j] = int(color)
    return expected


def _build_legend(shape: tuple[int, int], row_specs: list[dict[str, int]]) -> np.ndarray:
    legend = np.full(shape, BLACK, dtype=int)
    for spec in row_specs:
        legend[int(spec["row"]), KEY_COLUMN] = int(spec["marker"])
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

    for _ in range(25):
        H = _randint(GRID_HEIGHT_RANGE)
        W = _randint(GRID_WIDTH_RANGE)
        if H < bh + 2 * LEGEND_EDGE_MARGIN or W < bw + 2 * LEGEND_EDGE_MARGIN:
            continue
        if H * W < legend_area + patch_area + 32:
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
    bg, real_connector_colors, distractor_connector_colors, body_pool = _sample_palette()
    real_counts = _sample_counts(len(real_connector_colors))
    row_specs = _row_specs(real_connector_colors, real_counts, distractor_connector_colors, body_pool)

    max_count = max(int(spec["count"]) for spec in row_specs if int(spec["is_real"]))
    shape = _legend_shape(len(row_specs), max_count)
    expected = _build_expected(shape, row_specs)
    legend = _build_legend(shape, row_specs)

    group_patches = [
        _build_group_patch(
            connector_color=int(spec["marker"]),
            body_color=int(spec["body"]),
            count=int(spec["count"]),
            bg=bg,
        )
        for spec in row_specs
        if int(spec["is_real"])
    ]

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
    raise RuntimeError("failed to generate a valid combo 58490d8a input")
