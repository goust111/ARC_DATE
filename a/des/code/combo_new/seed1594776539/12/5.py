from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
REGION_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)

    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the grey seed hidden inside each colored rectangle, recolor the seeds by their parent rectangles, and center-stack them in left-to-right region order.

    Input structure:
    - The grid is fully tiled by distinct non-black rectangular color regions.
    - Inside each region, some cells are replaced by grey, and those grey cells form one 4-connected seed shape.
    - Each seed stays strictly inside its region, so the visible cells of the region color still reveal the region's full rectangle.

    Rule:
    - For each non-grey region color, recover the containing rectangle from the bounding box of that color's visible cells.
    - Inside that rectangle, isolate the grey seed, crop it to its own bounding box, and remember both the seed's parent color and the rectangle's left-to-right position.
    - Create a black output canvas whose size matches the largest seed bounding box, then place all cropped seeds centered on that canvas, repainting grey cells with the parent region color and drawing from leftmost region to rightmost region.

    Key insight:
    - The grey pixels must be grouped by the colored rectangle that contains them rather than treated as one global object; once each seed is assigned to its parent region, the task becomes ordered centered overlay with color restoration.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.all(g != BLACK), "input should contain only colored regions and grey seeds"

    regions: list[tuple[int, int, int, np.ndarray]] = []
    covered = np.zeros(g.shape, dtype=bool)

    for color in (int(c) for c in np.unique(g) if int(c) != GREY):
        color_mask = g == color
        r0, c0, r1, c1 = _bbox_from_mask(color_mask)
        window = g[r0 : r1 + 1, c0 : c1 + 1]

        assert np.all((window == color) | (window == GREY)), "region bbox must contain only its color and grey"
        assert np.all(window[0, :] == color), "seed must not touch the region's top edge"
        assert np.all(window[-1, :] == color), "seed must not touch the region's bottom edge"
        assert np.all(window[:, 0] == color), "seed must not touch the region's left edge"
        assert np.all(window[:, -1] == color), "seed must not touch the region's right edge"
        assert not covered[r0 : r1 + 1, c0 : c1 + 1].any(), "region rectangles must not overlap"

        seed_mask = window == GREY
        assert seed_mask.any(), "every region must contain a grey seed"
        assert len(_components4(seed_mask)) == 1, "each region must contain exactly one 4-connected seed"

        sr0, sc0, sr1, sc1 = _bbox_from_mask(seed_mask)
        seed = seed_mask[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
        regions.append((c0, r0, color, seed))
        covered[r0 : r1 + 1, c0 : c1 + 1] = True

    assert covered.all(), "region rectangles must tile the entire grid"

    max_h = max(seed.shape[0] for _, _, _, seed in regions)
    max_w = max(seed.shape[1] for _, _, _, seed in regions)
    out = np.full((max_h, max_w), BLACK, dtype=int)

    for _left, _top, color, seed in sorted(regions, key=lambda item: (item[0], item[1])):
        h, w = seed.shape
        r_off = (max_h - h) // 2
        c_off = (max_w - w) // 2
        patch = out[r_off : r_off + h, c_off : c_off + w]
        patch[seed] = color

    return out


def _mask_from_cells(cells: set[tuple[int, int]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _stair_base(h: int, w: int, left_to_right: bool) -> np.ndarray:
    r = 0
    c = 0 if left_to_right else w - 1
    cells = {(r, c)}
    steps = [(1, 0)] * (h - 1) + ([(0, 1)] * (w - 1) if left_to_right else [(0, -1)] * (w - 1))
    random.shuffle(steps)
    for dr, dc in steps:
        r += dr
        c += dc
        cells.add((r, c))
    return _mask_from_cells(cells, h, w)


def _frame_base(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _cross_base(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[random.randrange(h), :] = True
    mask[:, random.randrange(w)] = True
    return mask


def _comb_base(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    spine = 0 if random.random() < 0.5 else w - 1
    mask[:, spine] = True

    rows = {0, h - 1}
    mid_rows = list(range(1, h - 1))
    random.shuffle(mid_rows)
    extra_count = random.randint(0, min(2, len(mid_rows)))
    rows.update(mid_rows[:extra_count])
    for r in rows:
        mask[r, :] = True
    return mask


def _hook_base(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    if random.random() < 0.5:
        mask[:, 0] = True
    else:
        mask[:, -1] = True
    return mask


def _grow_connected(mask: np.ndarray, target_area: int) -> np.ndarray:
    h, w = mask.shape
    cells = {tuple(map(int, coord)) for coord in np.argwhere(mask)}

    while len(cells) < target_area:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(tuple(frontier)))

    return _mask_from_cells(cells, h, w)


def _random_seed_mask(h: int, w: int) -> np.ndarray:
    bases = [
        _frame_base(h, w),
        _cross_base(h, w),
        _comb_base(h, w),
        _hook_base(h, w),
        _stair_base(h, w, True),
        _stair_base(h, w, False),
    ]
    base = random.choice(bases)
    base_area = int(base.sum())
    max_area = h * w if h * w <= 4 else h * w - 1
    target_hi = max(base_area, min(max_area, int(np.ceil(0.8 * h * w))))
    target_area = random.randint(base_area, target_hi)
    mask = _grow_connected(base, target_area)
    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    return mask


def _stack_expected(seeds: list[tuple[int, np.ndarray]]) -> np.ndarray:
    max_h = max(mask.shape[0] for _, mask in seeds)
    max_w = max(mask.shape[1] for _, mask in seeds)
    out = np.full((max_h, max_w), BLACK, dtype=int)
    for color, mask in seeds:
        h, w = mask.shape
        r_off = (max_h - h) // 2
        c_off = (max_w - w) // 2
        patch = out[r_off : r_off + h, c_off : c_off + w]
        patch[mask] = color
    return out


def _distribute(total: int, parts: int, cap: int) -> list[int]:
    values = [0] * parts
    for _ in range(total):
        choices = [i for i, value in enumerate(values) if value < cap]
        if not choices:
            break
        values[random.choice(choices)] += 1
    return values


def _sample_layout() -> tuple[int, list[tuple[int, tuple[int, int], int]]] | None:
    for _ in range(300):
        n_regions = random.randint(3, 5)
        grid_h = random.randint(max(8, n_regions + 3), 14)

        heights = random.sample(list(range(2, grid_h - 1)), n_regions)
        widths = random.sample(list(range(2, 8)), n_regions)
        if sum(widths) + 2 * n_regions > 30:
            continue

        for _ in range(300):
            paired_dims = list(zip(heights, random.sample(widths, n_regions)))
            areas = [h * w for h, w in paired_dims]
            if len(set(areas)) != n_regions:
                continue

            max_h = max(h for h, _ in paired_dims)
            max_w = max(w for _, w in paired_dims)
            has_exact_nonmax = any(
                (h != max_h or w != max_w)
                and (max_h - h) % 2 == 0
                and (max_w - w) % 2 == 0
                for h, w in paired_dims
            )
            has_biased = any(
                (max_h - h) % 2 != 0 or (max_w - w) % 2 != 0 for h, w in paired_dims
            )
            if not has_exact_nonmax or not has_biased:
                continue

            order = random.sample(range(n_regions), n_regions)
            ordered_dims = [paired_dims[i] for i in order]
            ordered_areas = [areas[i] for i in order]
            if ordered_areas == sorted(ordered_areas) or ordered_areas == sorted(
                ordered_areas, reverse=True
            ):
                continue

            min_region_widths = [w + 2 for _, w in ordered_dims]
            min_total_w = sum(min_region_widths)
            remaining = 30 - min_total_w
            used_extra = 0 if remaining == 0 else random.randint(1, min(remaining, 3 * n_regions))
            extras = _distribute(used_extra, n_regions, cap=3)
            colors = random.sample(REGION_COLORS, n_regions)

            return grid_h, [
                (color, dims, base_w + extra)
                for color, dims, base_w, extra in zip(colors, ordered_dims, min_region_widths, extras)
            ]

    return None


def _generate_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None

    grid_h, region_specs = layout
    grid_w = sum(region_w for _, _, region_w in region_specs)
    inp = np.zeros((grid_h, grid_w), dtype=int)
    ordered_seeds: list[tuple[int, np.ndarray]] = []

    c0 = 0
    for color, (seed_h, seed_w), region_w in region_specs:
        inp[:, c0 : c0 + region_w] = color
        seed = _random_seed_mask(seed_h, seed_w)

        r0 = random.randint(1, grid_h - seed_h - 1)
        c_seed = random.randint(c0 + 1, c0 + region_w - seed_w - 1)
        inp[r0 : r0 + seed_h, c_seed : c_seed + seed_w][seed] = GREY

        ordered_seeds.append((color, seed))
        c0 += region_w

    expected = _stack_expected(ordered_seeds)
    try:
        actual = main(inp)
    except Exception:
        return None

    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
