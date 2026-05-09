# Combo variant of 58490d8a (index 27)
# combo_spec: D1/3+D2/1+D11/0
# Name: 连接器匹配+异形精灵+4连通

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, label_mask, paste, random_free_location

# Design analysis
# ---------------
# a) Concreteization and trace:
#    - The legend remains a black tally box with one colored marker pixel on each odd row.
#    - Outside the box, the counted sources are no longer "all pixels of the same legend color".
#      Instead each legend row matches exactly one detached single-pixel connector of that color.
#    - Around each connector sits a local cluster of monochrome body motifs. Those body motifs use
#      a body color that is independent from the legend color, and several different motifs in the
#      same cluster may have different silhouettes.
#    - Each body motif is one 8-connected patch, but some motifs split into 2 or 3 parts under
#      4-connectivity because their same-color pixels only meet diagonally.
#    - Small trace:
#        original: legend color 8 -> count all outside 8-connected color-8 objects -> draw 8,8,8
#        fused:    legend color 8 -> find the lone outside connector pixel colored 8
#                  -> collect nearby body motifs assigned to that connector
#                  -> count their 4-connected fragments (for example 1 + 2 = 3)
#                  -> draw 8,8,8
#      The difference is threefold: matching goes through a connector, body shape is not fixed,
#      and one visible body motif may contribute more than one tally dot.
# b) Complexity calibration:
#    - Body colors are sampled from a tiny shared pool, often repeated across multiple connector
#      groups, so body color is not a reliable group id.
#    - Motifs are limited to 2x2..3x3 boxes with area 4..7, matching the source-task scale.
#    - Per cluster we allow only 1..3 body motifs, and each motif contributes 1..3 four-connected
#      pieces; that keeps the search space small while still making D2 and D11 visible.
#    - Generation forces at least one keyed cluster to contain multiple distinct body shapes and at
#      least one motif with four-count > 1, so the sample set does not collapse back to the source.
# c) Hidden-assumption audit of the original solver:
#    - Legend detection via "largest black 8-connected component whose bbox contains non-black
#      pixels" still holds because the legend geometry stays unchanged and outside objects are never black.
#    - Key-column detection still holds because the only non-black pixels inside the legend remain the
#      marker pixels in column 1.
#    - The original `mask = (g == key_color) & outside` breaks: the legend color now appears only on
#      the connector singleton, not on the counted bodies.
#    - The original `label_mask(..., connectivity=8)` also breaks twice: it would count only the
#      connector singleton for matching colors, and it would merge diagonal body contacts that must now
#      split under 4-connectivity.
#    - The fused solver therefore needs a new chain: isolate outside foreground -> separate singleton
#      connectors from non-singleton bodies -> assign each body to its nearest connector -> count 4-CCs.
# d) Placement legality:
#    - Every cluster is constructed inside a reserved rectangular patch that contains exactly one
#      singleton connector and 1..3 body motifs separated by at least one background cell.
#    - Whole patches, not just visible pixels, are reserved with 8-neighborhood padding, so clusters
#      cannot overlap each other or touch the legend box, and nearest-connector assignment stays stable.
# e) Visual observability:
#    - The connector is the only singleton in a cluster, so humans can identify it as a tag.
#    - Reused body colors across different connectors show that connector color, not body color,
#      determines which legend row receives the tally.
#    - At least one keyed cluster has two or more visibly different same-color motifs, so the D2
#      shape-variation signal is present in the raw pixels.
#    - At least one keyed cluster contains a motif whose diagonal-only links split under 4-connectivity,
#      making the output larger than the visible number of body blobs in that cluster.
# f) Visual cleanliness:
#    - The legend and clusters occupy separate regions with a background moat.
#    - Cluster internals use a tiny fixed slot layout, so the image reads as a few compact tagged groups
#      rather than a noisy scatter of unrelated fragments.
# g) Color inference:
#    - No ordering or numeric property of colors is used.
#    - Marker colors are reused exactly on connectors, while body colors come from a small independent
#      pool; this gives a perceptual cue for the new matching rule without introducing hidden metadata.
# h) New solver assumptions that generation must enforce:
#    - Outside singleton components are connectors only.
#    - Each keyed connector color appears on exactly one outside connector.
#    - Every body motif is a monochrome 8-connected component with known 4-connected fragment count.
#    - Each body motif is closer to its own connector than to any other connector.
#    - At least one keyed cluster exhibits both fusion effects: multiple body shapes and a nontrivial
#      four-connected split, so the generated task truly instantiates the combo rather than the source rule.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match legend rows to connector pixels, then tally 4-connected body fragments inside each tagged cluster.

    Input structure:
    - A black legend rectangle contains one colored marker pixel on each tally row in its leftmost non-black column.
    - Outside the box, each relevant group contains exactly one isolated connector pixel whose color matches a legend marker, plus 1-3 nearby same-body-color motifs; motifs of that body color may have different shapes.

    Rule:
    - For each legend row, find the outside connector with the same color as the row marker.
    - Assign nearby non-singleton body motifs to that connector, count their same-color pieces using 4-connectivity, and draw that many evenly spaced dots across the legend row using the marker color.

    Key insight: The legend color identifies a cluster through its detached connector, while the actual tally comes from splitting that cluster's body motifs by 4-connectivity rather than by visible blob count.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    black = 0

    best = None
    for coords in components_mask(g == int(black), connectivity=8):
        if coords.size == 0:
            continue
        box = bbox_coords(coords)
        sub = crop_bbox(g, box)
        if (sub != int(black)).any():
            cand = (int(coords.shape[0]), tuple(map(int, box)))
            if best is None or cand[0] > best[0]:
                best = cand

    assert best is not None
    r0, c0, r1, c1 = map(int, best[1])
    legend = crop_bbox(g, (r0, c0, r1, c1))
    bh, bw = map(int, legend.shape)

    key_col = None
    for cc in range(bw):
        if (legend[:, int(cc)] != int(black)).any():
            key_col = int(cc)
            break
    if key_col is None:
        return np.zeros_like(legend)

    keys = [(int(rr), int(legend[int(rr), int(key_col)])) for rr in range(bh) if int(legend[int(rr), int(key_col)]) != int(black)]
    keys.sort(key=lambda item: int(item[0]))

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
            connectors.append((int(rr), int(cc), int(g[int(rr), int(cc)])))
        else:
            bodies.append(coords)

    counts_by_connector: dict[int, int] = {}
    if connectors:
        conn_pts = np.array([(rr, cc) for rr, cc, _color in connectors], dtype=int)
        conn_colors = [int(color) for _rr, _cc, color in connectors]
        for coords in bodies:
            diffs = np.abs(coords[:, None, :] - conn_pts[None, :, :])
            cheb = np.max(diffs, axis=2).min(axis=0)
            idx = int(np.argmin(cheb))

            comp_mask = np.zeros((H, W), dtype=bool)
            comp_mask[coords[:, 0], coords[:, 1]] = True
            _lab4, n4 = label_mask(comp_mask, connectivity=4)
            counts_by_connector[int(conn_colors[idx])] = int(counts_by_connector.get(int(conn_colors[idx]), 0)) + int(n4)

    out = np.zeros((bh, bw), dtype=int)
    for rr, marker_color in keys:
        count = int(counts_by_connector.get(int(marker_color), 0))
        for i in range(count):
            cc = int(key_col + 2 * i)
            if 0 <= cc < bw:
                out[int(rr), int(cc)] = int(marker_color)
    return out


# --------------- generator (combo variant) ---------------

COLOR_POOL = tuple(range(1, 10))
BLACK = 0
DISTRACTOR_PROB = 0.4

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

SPRITE_AREA_RANGE = (4, 7)
SPRITE_BBOXES = ((2, 2), (2, 3), (3, 2), (3, 3))
SPRITE_CONNECTIVITY = 8

BODY_FOUR_COUNTS = (1, 2, 3)
MAX_BODY_MOTIFS = 3
GROUP_RECT_PADDING = 2
GENERATION_ATTEMPTS = 400
GRID_SIZE_HEADROOM = 4

CELL_H = 4
CELL_W = 4
CONNECTOR_SLOT = (0, 1)
BODY_SLOT_LAYOUTS: dict[int, tuple[tuple[int, int], ...]] = {
    1: ((1, 1),),
    2: ((0, 0), (1, 1)),
    3: ((0, 0), (1, 0), (1, 1)),
}


def _randint(bounds: tuple[int, int]) -> int:
    return random.randint(bounds[0], bounds[1])


def _key_row(i: int) -> int:
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _legend_shape(k: int, max_count: int) -> tuple[int, int]:
    bh = KEY_ROW_STEP * k + LEGEND_HEIGHT_TAIL
    bw = TALLY_COL_STEP * max_count + LEGEND_WIDTH_TAIL + _randint(LEGEND_SLACK_RANGE)
    return bh, bw


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


@lru_cache(maxsize=None)
def _valid_masks(h: int, w: int, area: int) -> tuple[np.ndarray, ...]:
    masks = []
    for cells in combinations(range(h * w), area):
        mask = np.zeros((h, w), dtype=bool)
        for cell in cells:
            mask[cell // w, cell % w] = True
        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        _lab8, n8 = label_mask(mask, connectivity=SPRITE_CONNECTIVITY)
        if int(n8) == 1:
            masks.append(mask)
    return tuple(masks)


def _mask_signature(mask: np.ndarray) -> tuple[int, ...]:
    return (int(mask.shape[0]), int(mask.shape[1]), *(int(v) for v in mask.astype(int).flatten()))


@lru_cache(maxsize=1)
def _mask_catalog() -> dict[int, tuple[tuple[np.ndarray, tuple[int, ...]], ...]]:
    catalog: dict[int, list[tuple[np.ndarray, tuple[int, ...]]]] = {count: [] for count in BODY_FOUR_COUNTS}
    for area in range(SPRITE_AREA_RANGE[0], SPRITE_AREA_RANGE[1] + 1):
        for h, w in SPRITE_BBOXES:
            if h * w < area:
                continue
            for mask in _valid_masks(h, w, area):
                _lab4, n4 = label_mask(mask, connectivity=4)
                four_count = int(n4)
                if four_count in catalog:
                    catalog[four_count].append((mask.copy(), _mask_signature(mask)))
    return {count: tuple(options) for count, options in catalog.items()}


def _make_sprite_from_mask(color: int, bg: int, mask: np.ndarray) -> np.ndarray:
    sprite = np.full(mask.shape, int(bg), dtype=int)
    sprite[mask] = int(color)
    return sprite


def _sample_palette(forced_k: int | None = None, *, want_distractor: bool | None = None) -> tuple[int, list[int], list[int], int | None]:
    colors = list(COLOR_POOL)
    bg = random.choice(colors)

    k = int(forced_k) if forced_k is not None else _randint(KEY_COUNT_RANGE)
    remaining = [c for c in colors if c != bg]
    marker_colors = random.sample(remaining, k)
    remaining = [c for c in remaining if c not in marker_colors]

    pool_size = 1 if len(remaining) == 1 or random.random() < 0.7 else 2
    pool_size = min(pool_size, len(remaining))
    body_pool = random.sample(remaining, pool_size)
    body_colors = [random.choice(body_pool) for _ in range(k)]
    if k >= 2 and len(set(body_colors)) == k:
        j = random.randrange(k)
        i = (j + random.randrange(1, k)) % k
        body_colors[i] = body_colors[j]

    leftover = [c for c in remaining if c not in body_pool]
    use_distractor = bool(want_distractor) if want_distractor is not None else random.random() < DISTRACTOR_PROB
    distractor_marker = random.choice(leftover) if leftover and use_distractor else None
    return bg, marker_colors, body_colors, distractor_marker


def _sample_counts(k: int) -> list[int]:
    while True:
        counts = [_randint(TARGET_COUNT_RANGE) for _ in range(k)]
        if max(counts) >= 3:
            return counts


@lru_cache(maxsize=None)
def _unit_partitions(total: int) -> tuple[tuple[int, ...], ...]:
    options: set[tuple[int, ...]] = set()
    allowed = BODY_FOUR_COUNTS
    for a in allowed:
        if a == total:
            options.add((a,))
        for b in allowed:
            if a + b == total:
                options.add(tuple(sorted((a, b))))
            for c in allowed:
                if a + b + c == total:
                    options.add(tuple(sorted((a, b, c))))
    filtered = [parts for parts in options if 1 <= len(parts) <= MAX_BODY_MOTIFS]
    filtered.sort(key=lambda parts: (len(parts), parts))
    return tuple(filtered)


def _choose_group_units(total: int, *, force_fusion: bool) -> tuple[int, ...] | None:
    options = list(_unit_partitions(int(total)))
    if force_fusion:
        options = [parts for parts in options if len(parts) >= 2 and max(parts) > 1]
        return random.choice(options) if options else None

    rich = [parts for parts in options if len(parts) >= 2 and max(parts) > 1]
    multi = [parts for parts in options if len(parts) >= 2]
    if rich and random.random() < 0.55:
        return random.choice(rich)
    if multi and random.random() < 0.4:
        return random.choice(multi)
    return random.choice(options) if options else None


def _choose_sprite_variant(four_count: int, used: set[tuple[int, ...]] | None = None) -> tuple[np.ndarray, tuple[int, ...]]:
    used = used or set()
    options = [(mask, sig) for mask, sig in _mask_catalog()[int(four_count)] if sig not in used]
    if not options:
        options = list(_mask_catalog()[int(four_count)])
    if not options:
        raise ValueError(f"no mask options for four-count {four_count}")
    mask, sig = random.choice(options)
    return mask.copy(), sig


def _build_group_patch(
    connector_color: int,
    body_color: int,
    units: tuple[int, ...],
    bg: int,
) -> tuple[np.ndarray, dict[str, object]]:
    motif_units = list(units)
    random.shuffle(motif_units)
    slots = list(BODY_SLOT_LAYOUTS[len(motif_units)])
    random.shuffle(slots)

    patch = np.full((2 * CELL_H, 2 * CELL_W), int(bg), dtype=int)
    used_signatures: set[tuple[int, ...]] = set()
    signatures: list[tuple[int, ...]] = []

    for unit, (slot_r, slot_c) in zip(motif_units, slots):
        mask, sig = _choose_sprite_variant(int(unit), used=used_signatures)
        used_signatures.add(sig)
        signatures.append(sig)

        sprite = _make_sprite_from_mask(body_color, bg, mask)
        sh, sw = map(int, sprite.shape)
        top = int(slot_r * CELL_H + (CELL_H - sh) // 2)
        left = int(slot_c * CELL_W + (CELL_W - sw) // 2)
        paste(patch, sprite, top, left, transparent=bg)

    conn_r = int(CONNECTOR_SLOT[0] * CELL_H + CELL_H // 2)
    conn_c = int(CONNECTOR_SLOT[1] * CELL_W + CELL_W // 2)
    patch[conn_r, conn_c] = int(connector_color)

    nz = np.argwhere(patch != int(bg))
    r0, c0, r1, c1 = bbox_coords(nz)
    cropped = crop_bbox(patch, (int(r0), int(c0), int(r1), int(c1)))
    meta = {
        "unit_sum": int(sum(motif_units)),
        "motif_count": int(len(motif_units)),
        "shape_variants": int(len(set(signatures))),
        "signatures": tuple(signatures),
    }
    return cropped, meta


def _new_grid_with_legend(bg: int, legend: np.ndarray, patches: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    bh, bw = map(int, legend.shape)
    legend_area = int(bh * bw)
    patch_area = int(sum(int(p.shape[0]) * int(p.shape[1]) for p in patches))
    patch_count = int(len(patches))
    min_h = max(GRID_HEIGHT_RANGE[0], bh + 2 * LEGEND_EDGE_MARGIN + CELL_H + patch_count)
    min_w = max(GRID_WIDTH_RANGE[0], bw + 2 * LEGEND_EDGE_MARGIN + CELL_W + patch_count)
    max_h = max(GRID_HEIGHT_RANGE[1], min_h + GRID_SIZE_HEADROOM)
    max_w = max(GRID_WIDTH_RANGE[1], min_w + GRID_SIZE_HEADROOM)

    for _ in range(30):
        H = random.randint(min_h, max_h)
        W = random.randint(min_w, max_w)
        if H < bh + 2 * LEGEND_EDGE_MARGIN or W < bw + 2 * LEGEND_EDGE_MARGIN:
            continue
        if H * W < legend_area + patch_area + 12 * len(patches):
            continue

        grid = np.full((H, W), int(bg), dtype=int)
        occ = np.zeros((H, W), dtype=int)
        r0 = random.randint(LEGEND_EDGE_MARGIN, H - bh - LEGEND_EDGE_MARGIN)
        c0 = random.randint(LEGEND_EDGE_MARGIN, W - bw - LEGEND_EDGE_MARGIN)
        paste(grid, legend, r0, c0)
        occ[int(r0) : int(r0) + bh, int(c0) : int(c0) + bw] = 1
        return grid, occ
    return None


def _place_group_patches(grid: np.ndarray, occ: np.ndarray, patches: list[np.ndarray], bg: int) -> None:
    ordered = sorted(patches, key=lambda patch: int(patch.shape[0]) * int(patch.shape[1]), reverse=True)
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


def _gen_attempt(forced_k: int | None = None, *, want_distractor: bool | None = None) -> np.ndarray | None:
    bg, marker_colors, body_colors, distractor_marker = _sample_palette(forced_k, want_distractor=want_distractor)
    counts = _sample_counts(len(marker_colors))
    expected = _build_expected(marker_colors, counts)
    legend = _build_legend(expected.shape, marker_colors)

    eligible_anchor = [i for i, total in enumerate(counts) if _choose_group_units(int(total), force_fusion=True) is not None]
    if not eligible_anchor:
        return None
    anchor = random.choice(eligible_anchor)

    keyed_patches: list[np.ndarray] = []
    keyed_meta: list[dict[str, object]] = []
    for i, (marker_color, body_color, total) in enumerate(zip(marker_colors, body_colors, counts)):
        units = _choose_group_units(int(total), force_fusion=(i == anchor))
        if units is None:
            return None
        patch, meta = _build_group_patch(int(marker_color), int(body_color), units, int(bg))
        keyed_patches.append(patch)
        keyed_meta.append(meta)

    all_patches = list(keyed_patches)
    if distractor_marker is not None:
        distractor_total = _randint((1, 4))
        distractor_units = _choose_group_units(int(distractor_total), force_fusion=False)
        if distractor_units is None:
            return None
        distractor_body = random.choice(body_colors)
        distractor_patch, _meta = _build_group_patch(int(distractor_marker), int(distractor_body), distractor_units, int(bg))
        all_patches.append(distractor_patch)

    base = _new_grid_with_legend(int(bg), legend, all_patches)
    if base is None:
        return None
    grid, occ = base

    try:
        _place_group_patches(grid, occ, all_patches, int(bg))
    except ValueError:
        return None

    if not np.array_equal(main(grid), expected):
        return None

    if max(int(meta["shape_variants"]) for meta in keyed_meta) < 2:
        return None
    if max(int(meta["unit_sum"]) - int(meta["motif_count"]) for meta in keyed_meta) < 1:
        return None

    return grid


def generate_input() -> np.ndarray:
    target_k = _randint(KEY_COUNT_RANGE)
    want_distractor = random.random() < DISTRACTOR_PROB
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(target_k, want_distractor=want_distractor)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid combo 58490d8a input")
