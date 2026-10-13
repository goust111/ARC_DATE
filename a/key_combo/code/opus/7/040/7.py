from __future__ import annotations

import numpy as np


_QUADRANT_ORDER = (0, 3, 2, 1)  # TL -> BR -> BL -> TR
_MATCHINGS = (
    ((0, 1), (2, 3)),
    ((0, 2), (1, 3)),
    ((0, 3), (1, 2)),
)
_DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_INPUT_COLORS = (1, 2, 3, 4)
_MIX_COLOR = 5
_GEN_RETRY_LIMIT = 400
_GEN_W_RANGE = (6, 9)  # exclusive high end
_GEN_MIN_AREA = 4
_GEN_MAX_AREA = 10
_GEN_MAX_SHAPE_SIDE = 4
_GEN_MAX_PAIR_OVERLAP = 4
_GEN_MAX_TOTAL_OVERLAP = 6


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in _DIR4]


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _component_coords(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"

    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))

    assert len(seen) == int(mask.sum()), "expected exactly one 4-connected object"
    return np.array(sorted(seen), dtype=int)


def _d4_variants(mask: np.ndarray) -> list[np.ndarray]:
    base = _tight_crop(mask)
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rotated = np.rot90(base, k)
        for variant in (rotated, np.fliplr(rotated)):
            key = (tuple(map(int, variant.shape)), variant.tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append(variant.copy())
    return variants


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min((tuple(map(int, v.shape)), v.tobytes()) for v in _d4_variants(mask))


def _partner_map_from_matching(matching: tuple[tuple[int, int], tuple[int, int]]) -> list[int]:
    partner = [-1, -1, -1, -1]
    for a, b in matching:
        partner[a] = b
        partner[b] = a
    assert all(idx >= 0 for idx in partner)
    return partner


def _extract_quadrant(region: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    foreground = sorted(int(v) for v in np.unique(region) if int(v) != 0)
    assert len(foreground) == 1, "each quadrant must contain exactly one foreground color"
    color = foreground[0]
    full_mask = region == color
    _component_coords(full_mask)
    tight_mask = _tight_crop(full_mask)
    return color, full_mask, tight_mask


def _infer_partner_map(tight_masks: list[np.ndarray]) -> list[int]:
    groups: dict[tuple[tuple[int, int], bytes], list[int]] = {}
    for idx, mask in enumerate(tight_masks):
        groups.setdefault(_canonical_key(mask), []).append(idx)

    assert len(groups) == 2, "expected exactly two shape classes"
    assert sorted(len(indices) for indices in groups.values()) == [2, 2], "expected two matching pairs"

    partner = [-1, -1, -1, -1]
    for indices in groups.values():
        a, b = indices
        partner[a] = b
        partner[b] = a
    return partner


def _overlay_masks(
    local_masks: list[np.ndarray],
    colors: list[int],
    partner: list[int],
) -> tuple[np.ndarray, int, int]:
    w = int(local_masks[0].shape[0])
    out = np.zeros((w, w), dtype=int)
    contributors = np.zeros((w, w), dtype=np.uint8)
    same_pair_hits = 0
    different_pair_hits = 0

    for idx in _QUADRANT_ORDER:
        mask = local_masks[idx]
        color = int(colors[idx])
        mate_bit = np.uint8(1 << partner[idx])
        occupied = mask & (contributors != 0)
        same_pair = occupied & ((contributors & mate_bit) != 0)
        different_pair = occupied & ~same_pair

        out[mask & ~occupied] = color
        out[different_pair] = color
        out[same_pair] = _MIX_COLOR

        same_pair_hits += int(same_pair.sum())
        different_pair_hits += int(different_pair.sum())
        contributors[mask] |= np.uint8(1 << idx)

    return out, same_pair_hits, different_pair_hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the four quadrant shapes onto one canvas and color overlaps by shape-based pairing.

    Input structure:
    - The grid is an even-by-even 2x2 arrangement of equal square quadrants on a black background.
    - Each quadrant contains exactly one 4-connected monochrome shape, and the four foreground colors are distinct.
    - Across the four shapes, exactly two disjoint pairs match up to rotation and reflection.

    Rule:
    - Extract the one shape from each quadrant and pair the quadrants by D4 shape equivalence.
    - Fold all four quadrant masks onto one half-size square canvas, preserving each shape's local coordinates.
    - Paint in fixed back-to-front order TL, BR, BL, TR; later shapes normally overwrite earlier ones, except a later shape paints grey where it overlaps its matched partner.

    Key insight:
    - The visible stacking order is fixed, but deciding where grey appears requires solving the hidden pairing relation induced by rotation/flip shape matching.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h == w and h % 2 == 0, "expected a square grid split into four equal quadrants"

    side = h // 2
    assert side > 0

    regions = [
        g[0:side, 0:side],
        g[0:side, side:],
        g[side:, 0:side],
        g[side:, side:],
    ]

    colors: list[int] = []
    local_masks: list[np.ndarray] = []
    tight_masks: list[np.ndarray] = []
    seen_colors: set[int] = set()

    for region in regions:
        color, full_mask, tight_mask = _extract_quadrant(region)
        assert color not in seen_colors, "foreground colors must be distinct across quadrants"
        seen_colors.add(color)
        colors.append(color)
        local_masks.append(full_mask)
        tight_masks.append(tight_mask)

    partner = _infer_partner_map(tight_masks)
    out, _, _ = _overlay_masks(local_masks, colors, partner)
    return out


def _random_connected_shape(rng: np.random.Generator) -> np.ndarray | None:
    area = int(rng.integers(_GEN_MIN_AREA, _GEN_MAX_AREA + 1))
    canvas = np.zeros((_GEN_MAX_SHAPE_SIDE + 1, _GEN_MAX_SHAPE_SIDE + 1), dtype=bool)
    start = (int(rng.integers(0, canvas.shape[0])), int(rng.integers(0, canvas.shape[1])))
    canvas[start] = True
    cells = {start}

    while len(cells) < area:
        frontier = []
        for r, c in list(cells):
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < canvas.shape[0] and 0 <= nc < canvas.shape[1] and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cell = frontier[int(rng.integers(0, len(frontier)))]
        cells.add(cell)
        canvas[cell] = True

    mask = _tight_crop(canvas)
    mh, mw = map(int, mask.shape)
    if not (2 <= mh <= _GEN_MAX_SHAPE_SIDE and 2 <= mw <= _GEN_MAX_SHAPE_SIDE):
        return None
    if int(mask.sum()) != area:
        return None
    if area == mh * mw:
        return None
    _component_coords(mask)
    return mask


def _sample_distinct_base_shapes(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    first = None
    for _ in range(60):
        first = _random_connected_shape(rng)
        if first is not None:
            break
    if first is None:
        return None

    first_key = _canonical_key(first)
    for _ in range(120):
        second = _random_connected_shape(rng)
        if second is None:
            continue
        if _canonical_key(second) != first_key:
            return first, second
    return None


def _feasible_positions(mask: np.ndarray, side: int) -> list[tuple[int, int]]:
    h, w = map(int, mask.shape)
    if h > side - 2 or w > side - 2:
        return []
    return [(r, c) for r in range(1, side - h) for c in range(1, side - w)]


def _placed_mask(mask: np.ndarray, top_left: tuple[int, int], side: int) -> np.ndarray:
    r0, c0 = map(int, top_left)
    h, w = map(int, mask.shape)
    placed = np.zeros((side, side), dtype=bool)
    placed[r0 : r0 + h, c0 : c0 + w] = mask
    return placed


def _pair_placement_options(
    first: np.ndarray,
    second: np.ndarray,
    side: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    options: list[tuple[np.ndarray, np.ndarray]] = []
    positions_first = _feasible_positions(first, side)
    positions_second = _feasible_positions(second, side)
    for pos_first in positions_first:
        placed_first = _placed_mask(first, pos_first, side)
        for pos_second in positions_second:
            placed_second = _placed_mask(second, pos_second, side)
            overlap = int((placed_first & placed_second).sum())
            if 1 <= overlap <= _GEN_MAX_PAIR_OVERLAP:
                options.append((placed_first, placed_second))
    return options


def _quadrant_slices(side: int, idx: int) -> tuple[slice, slice]:
    row_offset = 0 if idx < 2 else side
    col_offset = 0 if idx in (0, 2) else side
    return slice(row_offset, row_offset + side), slice(col_offset, col_offset + side)


def _render_input(side: int, local_masks: list[np.ndarray], colors: list[int]) -> np.ndarray:
    grid = np.zeros((side * 2, side * 2), dtype=int)
    for idx, mask in enumerate(local_masks):
        rs, cs = _quadrant_slices(side, idx)
        region = grid[rs, cs]
        region[mask] = int(colors[idx])
    return grid


def _case_metrics(local_masks: list[np.ndarray]) -> tuple[int, int]:
    union = np.logical_or.reduce(local_masks)
    area_sum = sum(int(mask.sum()) for mask in local_masks)
    union_count = int(union.sum())
    return union_count, int(area_sum - union_count)


def _pick_case(rng: np.random.Generator) -> tuple[int, list[np.ndarray], list[int], list[int]] | None:
    side = int(rng.integers(*_GEN_W_RANGE))
    base_shapes = _sample_distinct_base_shapes(rng)
    if base_shapes is None:
        return None

    matching = _MATCHINGS[int(rng.integers(0, len(_MATCHINGS)))]
    partner = _partner_map_from_matching(matching)
    pair_bases = list(base_shapes)
    if int(rng.integers(0, 2)) == 1:
        pair_bases.reverse()

    tight_masks: list[np.ndarray | None] = [None, None, None, None]
    pair_options: list[tuple[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]]] = []
    for pair, base_mask in zip(matching, pair_bases):
        variants = _d4_variants(base_mask)
        first_mask = variants[int(rng.integers(0, len(variants)))]
        second_mask = variants[int(rng.integers(0, len(variants)))]
        tight_masks[pair[0]] = first_mask
        tight_masks[pair[1]] = second_mask
        options = _pair_placement_options(first_mask, second_mask, side)
        if not options:
            return None
        pair_options.append((pair, options))

    colors = list(_INPUT_COLORS)
    perm = rng.permutation(len(colors))
    colors = [colors[int(i)] for i in perm]
    need_diff_overlap = bool(rng.random() < 0.7)

    (pair_a, options_a), (pair_b, options_b) = pair_options
    sample_a = rng.permutation(len(options_a))[: min(len(options_a), 80)]
    sample_b = rng.permutation(len(options_b))[: min(len(options_b), 120)]

    for idx_a in sample_a:
        placed_a0, placed_a1 = options_a[int(idx_a)]
        for idx_b in sample_b:
            placed_b0, placed_b1 = options_b[int(idx_b)]
            local_masks = [np.zeros((side, side), dtype=bool) for _ in range(4)]
            local_masks[pair_a[0]] = placed_a0
            local_masks[pair_a[1]] = placed_a1
            local_masks[pair_b[0]] = placed_b0
            local_masks[pair_b[1]] = placed_b1

            union_count, total_overlap = _case_metrics(local_masks)
            if union_count < 10 or union_count > int(0.62 * side * side):
                continue
            if total_overlap > _GEN_MAX_TOTAL_OVERLAP:
                continue

            expected, same_hits, diff_hits = _overlay_masks(local_masks, colors, partner)
            if same_hits < 1:
                continue
            if need_diff_overlap:
                if diff_hits < 1:
                    continue
            elif diff_hits != 0:
                continue
            if int(np.count_nonzero(expected == _MIX_COLOR)) > 4:
                continue

            inp = _render_input(side, local_masks, colors)
            try:
                actual = main(inp)
            except AssertionError:
                continue
            if not np.array_equal(actual, expected):
                continue
            return side, local_masks, colors, partner

    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRY_LIMIT):
        case = _pick_case(rng)
        if case is None:
            continue
        side, local_masks, colors, _partner = case
        return _render_input(side, local_masks, colors)
    raise RuntimeError("failed to generate a valid input for puzzle 7")
