import random
from collections import deque

import numpy as np


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PALETTE = tuple(range(10))
SHADOW_COLOR = 5

LEFT_EXTRA_RANGE = (2, 5)
FRAGMENT_COUNT_RANGE = (3, 5)
SHADOW_BOX_H_RANGE = (5, 8)
SHADOW_BOX_W_RANGE = (5, 8)
SHADOW_FILL_RANGE = (0.58, 0.82)
MIN_FRAGMENT_AREA = 3
MAX_OVERLAP_ADD = 3
RIGHT_W_MAX = 18
PLACEMENT_RETRIES = 180
GENERATION_RETRIES = 400

_RNG = random.Random()


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    total = int(np.sum(mask))
    return total > 0 and len(_components(mask)) == 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _mask_from_coords(coords: set[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _bitcount(value: int) -> int:
    return int(value.bit_count())


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for c in range(1, w - 1):
        value = int(grid[0, c])
        if value == bg:
            continue
        if np.all(grid[:, c] == value) and np.any(grid[:, :c] == value):
            candidates.append((c, value))
    assert len(candidates) == 1, "expected exactly one full-height separator"
    return candidates[0]


def _extract_fragments(right: np.ndarray, bg: int) -> list[dict]:
    comps = _components(right != bg)
    assert comps, "expected fragments in the right panel"
    fragments: list[dict] = []
    for comp in comps:
        colors = {int(right[r, c]) for r, c in comp}
        assert len(colors) == 1, "each fragment must be monochrome"
        color = colors.pop()
        r0, c0, r1, c1 = _bbox(comp)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            mask[r - r0, c - c0] = True
        fragments.append({"color": color, "mask": mask, "area": int(mask.sum())})
    return fragments


def _placement_options(shadow_mask: np.ndarray, fragment_masks: list[np.ndarray]) -> tuple[list[list[dict]], int]:
    shadow_cells = np.argwhere(shadow_mask)
    assert shadow_cells.size > 0, "shadow must be non-empty"
    index_grid = np.full(shadow_mask.shape, -1, dtype=int)
    for idx, (r, c) in enumerate(shadow_cells):
        index_grid[int(r), int(c)] = int(idx)
    all_bits = (1 << len(shadow_cells)) - 1

    options: list[list[dict]] = []
    for frag_mask in fragment_masks:
        fh, fw = frag_mask.shape
        frag_coords = [(int(r), int(c)) for r, c in np.argwhere(frag_mask)]
        frag_options: list[dict] = []
        for r0 in range(shadow_mask.shape[0] - fh + 1):
            for c0 in range(shadow_mask.shape[1] - fw + 1):
                fits = True
                bits = 0
                for dr, dc in frag_coords:
                    rr, cc = r0 + dr, c0 + dc
                    if not shadow_mask[rr, cc]:
                        fits = False
                        break
                    bits |= 1 << int(index_grid[rr, cc])
                if fits:
                    frag_options.append({"origin": (r0, c0), "bits": bits})
        options.append(frag_options)
    return options, all_bits


def _solve_placements(
    shadow_mask: np.ndarray,
    fragment_masks: list[np.ndarray],
    limit: int = 2,
) -> list[list[tuple[int, int]]]:
    options, all_bits = _placement_options(shadow_mask, fragment_masks)
    assert all(options), "each fragment must fit somewhere inside the shadow"

    areas = [int(mask.sum()) for mask in fragment_masks]
    order = sorted(range(len(fragment_masks)), key=lambda i: (len(options[i]), -areas[i]))
    ordered_options = [options[i] for i in order]

    suffix_union = [0] * (len(order) + 1)
    for i in range(len(order) - 1, -1, -1):
        union_bits = 0
        for option in ordered_options[i]:
            union_bits |= int(option["bits"])
        suffix_union[i] = suffix_union[i + 1] | union_bits

    solutions: list[list[tuple[int, int]]] = []

    def dfs(idx: int, covered: int, chosen: list[tuple[int, int]]) -> None:
        if len(solutions) >= limit:
            return
        if (covered | suffix_union[idx]) != all_bits:
            return
        if idx == len(order):
            if covered == all_bits:
                result = [(-1, -1)] * len(order)
                for j, origin in enumerate(chosen):
                    result[order[j]] = origin
                solutions.append(result)
            return

        candidate_ids = sorted(
            range(len(ordered_options[idx])),
            key=lambda k: _bitcount(int(ordered_options[idx][k]["bits"]) & ~covered),
            reverse=True,
        )
        for option_id in candidate_ids:
            option = ordered_options[idx][option_id]
            dfs(idx + 1, covered | int(option["bits"]), chosen + [option["origin"]])
            if len(solutions) >= limit:
                return

    dfs(0, 0, [])
    return solutions


def _render_solution(
    canvas_shape: tuple[int, int],
    bg: int,
    fragments: list[dict],
    origins: list[tuple[int, int]],
) -> np.ndarray:
    out = np.full(canvas_shape, bg, dtype=int)
    order = sorted(range(len(fragments)), key=lambda i: (fragments[i]["area"], i), reverse=True)
    for idx in order:
        mask = fragments[idx]["mask"]
        r0, c0 = origins[idx]
        for dr, dc in np.argwhere(mask):
            out[r0 + int(dr), c0 + int(dc)] = int(fragments[idx]["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble right-side fragments into the left shadow, then resolve overlap by fragment area.
    Smaller fragments are painted last and therefore appear on top.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    sep_c, shadow_color = _find_separator(g, bg)

    left = g[:, :sep_c]
    right = g[:, sep_c + 1 :]
    shadow_mask = left == shadow_color
    assert np.any(shadow_mask), "left panel must contain the shadow"

    fragments = _extract_fragments(right, bg)
    fragment_masks = [frag["mask"] for frag in fragments]
    solutions = _solve_placements(shadow_mask, fragment_masks, limit=2)
    assert len(solutions) == 1, "fragment placement must be unique"

    return _render_solution(left.shape, bg, fragments, solutions[0])


def _random_shadow(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*SHADOW_BOX_H_RANGE)
    w = rng.randint(*SHADOW_BOX_W_RANGE)
    target_area = rng.randint(
        max(10, int(np.ceil(h * w * SHADOW_FILL_RANGE[0]))),
        max(10, int(np.floor(h * w * SHADOW_FILL_RANGE[1]))),
    )
    target_area = min(target_area, h * w - 2)
    if target_area < 10:
        return None

    mask = np.ones((h, w), dtype=bool)
    cells_to_remove = h * w - target_area
    for _ in range(cells_to_remove):
        removable: list[tuple[int, int]] = []
        for r in range(h):
            for c in range(w):
                if not mask[r, c]:
                    continue
                if all(
                    0 <= r + dr < h and 0 <= c + dc < w and mask[r + dr, c + dc]
                    for dr, dc in NEIGHBORS4
                ):
                    continue
                mask[r, c] = False
                if _is_connected(mask):
                    removable.append((r, c))
                mask[r, c] = True
        if not removable:
            return None
        rr, cc = rng.choice(removable)
        mask[rr, cc] = False

    mask = _crop_mask(mask)
    if mask.shape[0] < 4 or mask.shape[1] < 4:
        return None
    if int(mask.sum()) == int(mask.size):
        return None
    return mask


def _partition_shadow(mask: np.ndarray, count: int, rng: random.Random) -> list[set[tuple[int, int]]] | None:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if len(cells) < count * MIN_FRAGMENT_AREA:
        return None

    seeds = rng.sample(cells, count)
    regions = [{seed} for seed in seeds]
    unassigned = set(cells) - set(seeds)

    while unassigned:
        frontiers: list[list[tuple[int, int]]] = []
        active: list[int] = []
        for idx, region in enumerate(regions):
            frontier = []
            for r, c in region:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in unassigned:
                        frontier.append((nr, nc))
            dedup = list(dict.fromkeys(frontier))
            frontiers.append(dedup)
            if dedup:
                active.append(idx)
        if not active:
            return None
        min_size = min(len(regions[idx]) for idx in active)
        choices = [idx for idx in active if len(regions[idx]) <= min_size + 1]
        region_id = rng.choice(choices)
        cell = rng.choice(frontiers[region_id])
        regions[region_id].add(cell)
        unassigned.remove(cell)

    for region in regions:
        if len(region) < MIN_FRAGMENT_AREA:
            return None
    return regions


def _add_overlap(
    regions: list[set[tuple[int, int]]],
    rng: random.Random,
) -> list[set[tuple[int, int]]] | None:
    pairs: list[tuple[int, int]] = []
    for top_idx, top in enumerate(regions):
        for bottom_idx, bottom in enumerate(regions):
            if top_idx == bottom_idx or len(top) >= len(bottom) - 1:
                continue
            if any(any((r + dr, c + dc) in top for dr, dc in NEIGHBORS4) for r, c in bottom):
                pairs.append((top_idx, bottom_idx))
    if not pairs:
        return None

    top_idx, bottom_idx = rng.choice(pairs)
    top = set(regions[top_idx])
    bottom = set(regions[bottom_idx])
    max_add = min(MAX_OVERLAP_ADD, len(bottom) - len(top) - 1)
    if max_add < 1:
        return None

    add_size = rng.randint(1, max_add)
    addition: set[tuple[int, int]] = set()
    while len(addition) < add_size:
        frontier = [
            cell
            for cell in bottom - addition
            if any((cell[0] + dr, cell[1] + dc) in top or (cell[0] + dr, cell[1] + dc) in addition for dr, dc in NEIGHBORS4)
        ]
        if not frontier:
            return None
        addition.add(rng.choice(frontier))

    updated = [set(region) for region in regions]
    updated[top_idx] |= addition
    return updated


def _embed_shadow(mask: np.ndarray, rng: random.Random) -> tuple[np.ndarray, tuple[int, int]]:
    sh, sw = mask.shape
    left_h = sh + rng.randint(*LEFT_EXTRA_RANGE)
    left_w = sw + rng.randint(*LEFT_EXTRA_RANGE)

    top_margin = left_h - sh
    left_margin = left_w - sw
    r0 = rng.randint(1, top_margin - 1) if top_margin >= 2 else 0
    c0 = rng.randint(1, left_margin - 1) if left_margin >= 2 else 0

    out = np.zeros((left_h, left_w), dtype=bool)
    out[r0 : r0 + sh, c0 : c0 + sw] = mask
    return out, (r0, c0)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def _expand_box(box: tuple[int, int, int, int], pad: int, h: int, w: int) -> tuple[int, int, int, int]:
    return (
        max(0, box[0] - pad),
        max(0, box[1] - pad),
        min(h - 1, box[2] + pad),
        min(w - 1, box[3] + pad),
    )


def _place_right_fragments(
    fragment_masks: list[np.ndarray],
    height: int,
    rng: random.Random,
) -> tuple[int, list[tuple[int, int]]] | None:
    widths = [mask.shape[1] for mask in fragment_masks]
    base_width = max(max(widths) + 2, (sum(widths) + len(widths)) // 2 + 4)
    right_w = min(RIGHT_W_MAX, base_width + rng.randint(0, 3))
    if right_w < max(widths):
        return None

    placements: list[tuple[int, int] | None] = [None] * len(fragment_masks)
    placed_boxes: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(fragment_masks)), key=lambda i: fragment_masks[i].shape[0] * fragment_masks[i].shape[1], reverse=True)

    for idx in order:
        mask = fragment_masks[idx]
        mh, mw = mask.shape
        row_lo = 0 if height == mh else 1 if height - mh >= 2 else 0
        row_hi = height - mh if height == mh else height - mh - (1 if height - mh >= 2 else 0)
        col_lo = 0 if right_w == mw else 1 if right_w - mw >= 2 else 0
        col_hi = right_w - mw if right_w == mw else right_w - mw - (1 if right_w - mw >= 2 else 0)
        if row_hi < row_lo or col_hi < col_lo:
            return None

        for _ in range(PLACEMENT_RETRIES):
            r0 = rng.randint(row_lo, row_hi)
            c0 = rng.randint(col_lo, col_hi)
            box = (r0, c0, r0 + mh - 1, c0 + mw - 1)
            if any(_boxes_overlap(_expand_box(box, 1, height, right_w), placed) for placed in placed_boxes):
                continue
            placements[idx] = (r0, c0)
            placed_boxes.append(box)
            break
        else:
            return None

    return right_w, [placement for placement in placements if placement is not None]


def _make_expected_output(
    left_shape: tuple[int, int],
    bg: int,
    fragments: list[dict],
    origins: list[tuple[int, int]],
) -> np.ndarray:
    return _render_solution(left_shape, bg, fragments, origins)


def _gen_attempt(rng: random.Random, overlap_requested: bool) -> np.ndarray | None:
    shadow_local = _random_shadow(rng)
    if shadow_local is None:
        return None

    shadow_area = int(shadow_local.sum())
    fragment_count = rng.randint(FRAGMENT_COUNT_RANGE[0], FRAGMENT_COUNT_RANGE[1])
    fragment_count = min(fragment_count, shadow_area // MIN_FRAGMENT_AREA)
    if fragment_count < FRAGMENT_COUNT_RANGE[0]:
        return None

    regions = _partition_shadow(shadow_local, fragment_count, rng)
    if regions is None:
        return None

    if overlap_requested:
        regions = _add_overlap(regions, rng)
        if regions is None:
            return None

    fragment_local_masks = [_mask_from_coords(region, shadow_local.shape) for region in regions]
    if any(not _is_connected(mask) for mask in fragment_local_masks):
        return None

    shadow_mask, shadow_offset = _embed_shadow(shadow_local, rng)
    bg = rng.choice([color for color in PALETTE if color != SHADOW_COLOR])
    palette = [color for color in PALETTE if color not in {bg, SHADOW_COLOR}]
    if len(palette) < len(regions):
        return None
    colors = rng.sample(palette, len(regions))

    fragments: list[dict] = []
    true_origins: list[tuple[int, int]] = []
    for color, region in zip(colors, regions):
        r0, c0, r1, c1 = _bbox(list(region))
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in region:
            mask[r - r0, c - c0] = True
        fragments.append({"color": color, "mask": mask, "area": int(mask.sum())})
        true_origins.append((shadow_offset[0] + r0, shadow_offset[1] + c0))

    fragment_masks = [frag["mask"] for frag in fragments]
    solutions = _solve_placements(shadow_mask, fragment_masks, limit=2)
    if len(solutions) != 1 or solutions[0] != true_origins:
        return None

    right_layout = _place_right_fragments(fragment_masks, shadow_mask.shape[0], rng)
    if right_layout is None:
        return None
    right_w, right_origins = right_layout

    grid = np.full((shadow_mask.shape[0], shadow_mask.shape[1] + 1 + right_w), bg, dtype=int)
    left = grid[:, : shadow_mask.shape[1]]
    left[shadow_mask] = SHADOW_COLOR
    grid[:, shadow_mask.shape[1]] = SHADOW_COLOR

    for frag, (r0, c0) in zip(fragments, right_origins):
        for dr, dc in np.argwhere(frag["mask"]):
            grid[r0 + int(dr), shadow_mask.shape[1] + 1 + c0 + int(dc)] = int(frag["color"])

    if _most_common_color(grid) != bg:
        return None

    expected = _make_expected_output(shadow_mask.shape, bg, fragments, true_origins)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    overlap_requested = bool(_RNG.getrandbits(1))
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(_RNG, overlap_requested)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
