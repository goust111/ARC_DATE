from __future__ import annotations

import numpy as np


BG = 0
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
COLOR_POOL = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
RNG = np.random.default_rng()
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop_to_bbox(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _stack_layers(main_crop: np.ndarray, order: list[int]) -> np.ndarray:
    colors = [int(v) for v in np.unique(main_crop) if int(v) != BG]
    assert colors, "expected a non-empty multicolor figure"
    assert set(order) == set(colors), "indicator bar colors must match the main figure colors"
    assert len(order) == len(colors), "indicator bar colors must be unique"

    out = np.zeros_like(main_crop)
    for color in order:
        layer = main_crop == int(color)
        assert layer.any(), "every bar color must appear in the main figure"
        out[_dilate4(layer)] = int(color)
    return out


def _bar_candidate(grid: np.ndarray, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if c0 != c1:
        return False
    if r1 - r0 + 1 != int(coords.shape[0]):
        return False
    column = grid[r0 : r1 + 1, c0]
    if np.any(column == BG):
        return False
    colors = [int(v) for v in column]
    return 3 <= len(colors) <= 5 and len(set(colors)) == len(colors)


def _extract_parts(grid: np.ndarray) -> tuple[np.ndarray, list[int]]:
    comps = _components(grid != BG)
    assert len(comps) == 2, "expected one multicolor figure and one indicator bar"

    bar_indices = [idx for idx, coords in enumerate(comps) if _bar_candidate(grid, coords)]
    assert len(bar_indices) == 1, "expected a unique single-column indicator bar"
    bar_idx = int(bar_indices[0])
    main_idx = 1 - bar_idx

    main_coords = comps[main_idx]
    bar_coords = comps[bar_idx]

    main_r0, main_c0, main_r1, main_c1 = _bbox(main_coords)
    bar_r0, bar_c0, bar_r1, bar_c1 = _bbox(bar_coords)
    assert main_c1 < bar_c0, "the main figure must be left of the indicator bar"
    assert main_c1 + 1 < bar_c0, "the main figure and bar must be separated by black cells"
    assert main_r1 - main_r0 + 1 >= 4 and main_c1 - main_c0 + 1 >= 4, "main figure is too small"
    assert int(main_coords.shape[0]) > int(bar_coords.shape[0]), "main figure must be larger than the bar"

    main_crop = grid[main_r0 : main_r1 + 1, main_c0 : main_c1 + 1].copy()
    order = [int(v) for v in grid[bar_r0 : bar_r1 + 1, bar_c1]]

    colors = [int(v) for v in np.unique(main_crop) if int(v) != BG]
    assert 3 <= len(colors) <= 5, "main figure must use 3-5 colors"
    assert set(order) == set(colors), "bar must mention each main-figure color exactly once"
    return main_crop, order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split the multicolor connected figure into monochrome layers, dilate each layer by
    one 4-neighborhood step, and restack them using the order shown by the right-side bar.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    main_crop, order = _extract_parts(g)
    return _stack_layers(main_crop, order)


def _adjacent_open_cells(mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: list[tuple[int, int]] = []
    for dr, dc in NEIGHBORS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
            cells.append((nr, nc))
    return cells


def _grow_blob(height: int, width: int, target: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (int(rng.integers(height)), int(rng.integers(width)))
    mask[start] = True
    frontier = set(_adjacent_open_cells(mask, start[0], start[1]))

    while int(mask.sum()) < target and frontier:
        choices = list(frontier)
        weights = np.array(
            [1 + sum(mask[nr, nc] for nr, nc in _neighbors_in_bounds(r, c, height, width)) ** 2 for r, c in choices],
            dtype=float,
        )
        weights /= weights.sum()
        idx = int(rng.choice(len(choices), p=weights))
        cell = choices[idx]
        frontier.remove(cell)
        mask[cell] = True
        for nxt in _adjacent_open_cells(mask, cell[0], cell[1]):
            frontier.add(nxt)
        if rng.random() < 0.30:
            extra = [nxt for nxt in _adjacent_open_cells(mask, cell[0], cell[1]) if nxt in frontier]
            if extra and int(mask.sum()) < target:
                er, ec = extra[int(rng.integers(len(extra)))]
                frontier.discard((er, ec))
                mask[er, ec] = True
                for nxt in _adjacent_open_cells(mask, er, ec):
                    frontier.add(nxt)
    return mask


def _neighbors_in_bounds(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for dr, dc in NEIGHBORS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            cells.append((nr, nc))
    return cells


def _union_rectangles(height: int, width: int, target: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)

    rh = int(rng.integers(2, min(height, 5) + 1))
    rw = int(rng.integers(2, min(width, 5) + 1))
    r0 = int(rng.integers(0, height - rh + 1))
    c0 = int(rng.integers(0, width - rw + 1))
    mask[r0 : r0 + rh, c0 : c0 + rw] = True

    for _ in range(12):
        if int(mask.sum()) >= target:
            break
        coords = np.argwhere(mask)
        anchor_r, anchor_c = coords[int(rng.integers(len(coords)))]
        rh = int(rng.integers(2, min(height, 6) + 1))
        rw = int(rng.integers(2, min(width, 6) + 1))

        r_lo = max(0, int(anchor_r) - rh + 1)
        r_hi = min(int(anchor_r), height - rh)
        c_lo = max(0, int(anchor_c) - rw + 1)
        c_hi = min(int(anchor_c), width - rw)
        r0 = int(rng.integers(r_lo, r_hi + 1))
        c0 = int(rng.integers(c_lo, c_hi + 1))
        mask[r0 : r0 + rh, c0 : c0 + rw] = True

    return mask


def _make_shape(mode: str, color_count: int, rng: np.random.Generator) -> np.ndarray | None:
    raw_h = int(rng.integers(7, 14))
    raw_w = int(rng.integers(7, 14))
    max_area = int(raw_h * raw_w * 0.72)
    min_area = max(color_count * 6, raw_h + raw_w)
    if max_area <= min_area:
        return None

    target = int(rng.integers(min_area, max_area + 1))
    if mode == "grow":
        mask = _grow_blob(raw_h, raw_w, target, rng)
    else:
        mask = _union_rectangles(raw_h, raw_w, target, rng)

    if not mask.any():
        return None
    mask = _crop_to_bbox(mask)
    h, w = mask.shape
    area = int(mask.sum())
    if h < 5 or w < 5:
        return None
    if area < color_count * 6:
        return None
    if area >= h * w:
        return None
    if len(_components(mask)) != 1:
        return None
    return mask


def _colorize_seeds(mask: np.ndarray, colors: list[int], rng: np.random.Generator) -> np.ndarray:
    coords = np.argwhere(mask)
    n_colors = len(colors)
    seed_count = int(n_colors + rng.integers(0, n_colors + 1))
    seed_coords = coords[rng.choice(len(coords), size=seed_count, replace=False)]
    extra_colors = rng.choice(np.asarray(colors, dtype=int), size=seed_count - n_colors, replace=True)
    seed_colors = np.concatenate([np.asarray(colors, dtype=int), extra_colors])
    seed_colors = seed_colors[rng.permutation(seed_count)]
    tie_bias = rng.permutation(seed_count)

    distances = np.abs(coords[:, None, 0] - seed_coords[None, :, 0]) + np.abs(coords[:, None, 1] - seed_coords[None, :, 1])
    scores = distances * (seed_count + 1) + tie_bias[None, :]
    owners = scores.argmin(axis=1)

    out = np.zeros(mask.shape, dtype=int)
    out[coords[:, 0], coords[:, 1]] = seed_colors[owners]
    return out


def _colorize_stripes(mask: np.ndarray, colors: list[int], rng: np.random.Generator) -> np.ndarray:
    coords = np.argwhere(mask)
    palette = np.asarray(colors, dtype=int)[rng.permutation(len(colors))]
    coeffs = ((1, 0), (0, 1), (1, 1), (1, 2), (2, 1))
    a, b = coeffs[int(rng.integers(len(coeffs)))]
    band = int(rng.integers(1, 3))
    phase = int(rng.integers(0, len(colors) * band))
    parity_boost = int(rng.integers(0, 2))

    values = a * coords[:, 0] + b * coords[:, 1] + phase
    if parity_boost:
        values += (coords[:, 0] + coords[:, 1]) % 2
    ids = (values // band) % len(colors)

    out = np.zeros(mask.shape, dtype=int)
    out[coords[:, 0], coords[:, 1]] = palette[ids]
    return out


def _color_counts(main_crop: np.ndarray, colors: list[int]) -> list[int]:
    return [int(np.count_nonzero(main_crop == color)) for color in colors]


def _random_order(colors: list[int], counts: list[int], rng: np.random.Generator) -> list[int] | None:
    ascending = [color for _count, color in sorted(zip(counts, colors))]
    descending = list(reversed(ascending))
    for _ in range(40):
        order = list(np.asarray(colors, dtype=int)[rng.permutation(len(colors))])
        if order != ascending and order != descending:
            return [int(v) for v in order]
    return None


def _make_main_crop(mask: np.ndarray, colors: list[int], color_mode: str, rng: np.random.Generator) -> np.ndarray | None:
    if color_mode == "seeds":
        main_crop = _colorize_seeds(mask, colors, rng)
    else:
        main_crop = _colorize_stripes(mask, colors, rng)

    counts = _color_counts(main_crop, colors)
    if min(counts) < 3:
        return None
    if len(set(counts)) != len(counts):
        return None
    if not np.array_equal(main_crop != BG, mask):
        return None
    return main_crop


def _render_input(main_crop: np.ndarray, order: list[int], rng: np.random.Generator) -> np.ndarray:
    crop_h, crop_w = main_crop.shape
    left_pad = int(rng.integers(1, 4))
    right_pad = int(rng.integers(1, 4))
    gap = int(rng.integers(2, 6))
    height = max(crop_h + int(rng.integers(2, 7)), len(order) + int(rng.integers(2, 7)))
    width = left_pad + crop_w + gap + 1 + right_pad

    grid = np.zeros((height, width), dtype=int)
    main_r0 = int(rng.integers(1, height - crop_h))
    main_c0 = left_pad
    bar_r0 = int(rng.integers(1, height - len(order)))
    bar_c = left_pad + crop_w + gap

    grid[main_r0 : main_r0 + crop_h, main_c0 : main_c0 + crop_w] = main_crop
    grid[bar_r0 : bar_r0 + len(order), bar_c] = np.asarray(order, dtype=int)
    return grid


def _valid_scene(main_crop: np.ndarray, order: list[int]) -> bool:
    output = _stack_layers(main_crop, order)
    if np.array_equal(output, main_crop):
        return False
    if len(np.unique(output[output != BG])) < 2:
        return False
    if np.array_equal(output, _stack_layers(main_crop, list(reversed(order)))):
        return False
    rotated = order[1:] + order[:1]
    if np.array_equal(output, _stack_layers(main_crop, rotated)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = RNG
    color_count = int(rng.integers(3, 6))
    colors = [int(v) for v in rng.choice(COLOR_POOL, size=color_count, replace=False)]
    shape_mode = ("grow", "rects")[int(rng.integers(2))]
    color_mode = ("seeds", "stripes")[int(rng.integers(2))]

    mask = _make_shape(shape_mode, color_count, rng)
    if mask is None:
        return None

    main_crop = _make_main_crop(mask, colors, color_mode, rng)
    if main_crop is None:
        return None

    counts = _color_counts(main_crop, colors)
    order = _random_order(colors, counts, rng)
    if order is None:
        return None

    if not _valid_scene(main_crop, order):
        return None

    grid = _render_input(main_crop, order, rng)
    try:
        expected = _stack_layers(main_crop, order)
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
