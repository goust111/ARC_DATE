from __future__ import annotations

import random

import numpy as np


BG = 0
GEN_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _overlap_slices(shape: tuple[int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    h, w = shape
    r0 = max(0, -dr)
    r1 = min(h, h - dr)
    c0 = max(0, -dc)
    c1 = min(w, w - dc)
    return r0, r1, c0, c1


def _best_translation(
    g: np.ndarray, missing: int = BG
) -> tuple[tuple[int, int] | None, list[tuple[int, int]], int]:
    h, w = g.shape
    best_support = -1
    best: list[tuple[int, int]] = []
    for dr in range(-(h - 1), h):
        for dc in range(-(w - 1), w):
            if dr == 0 and dc == 0:
                continue
            r0, r1, c0, c1 = _overlap_slices(g.shape, dr, dc)
            if r0 >= r1 or c0 >= c1:
                continue
            a = g[r0:r1, c0:c1]
            b = g[r0 + dr : r1 + dr, c0 + dc : c1 + dc]
            visible = (a != missing) & (b != missing)
            support = int(np.count_nonzero(visible))
            if support == 0:
                continue
            if not np.all(a[visible] == b[visible]):
                continue
            if support > best_support:
                best_support = support
                best = [(int(dr), int(dc))]
            elif support == best_support:
                best.append((int(dr), int(dc)))
    assert best_support > 0
    positive = [v for v in best if v[0] > 0 or (v[0] == 0 and v[1] > 0)]
    return (positive[0] if len(positive) == 1 else None), best, best_support


def _propagate_with_shift(g: np.ndarray, shift: tuple[int, int], missing: int = BG) -> np.ndarray:
    h, w = g.shape
    dr, dc = shift
    parent = list(range(h * w))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    r0, r1, c0, c1 = _overlap_slices(g.shape, dr, dc)
    for r in range(r0, r1):
        rr = r + dr
        for c in range(c0, c1):
            union(r * w + c, rr * w + c + dc)

    colors_by_root: dict[int, set[int]] = {}
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == missing:
                continue
            root = find(r * w + c)
            colors_by_root.setdefault(root, set()).add(color)

    assert all(len(colors) == 1 for colors in colors_by_root.values())
    out = g.copy()
    for r in range(h):
        for c in range(w):
            if int(out[r, c]) != missing:
                continue
            colors = colors_by_root.get(find(r * w + c), set())
            if len(colors) == 1:
                out[r, c] = next(iter(colors))
    return out


def _read_marker_colors(g: np.ndarray) -> list[int]:
    k = 0
    lim = min(g.shape)
    while k < lim and int(g[k, k]) != BG:
        k += 1
    assert k >= 2
    for r in range(k):
        for c in range(k):
            if r != c:
                assert int(g[r, c]) == BG
    return [int(g[i, i]) for i in range(k)]


def _layer_index_grid(h: int, w: int) -> np.ndarray:
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    return np.minimum(np.minimum(rows, cols), np.minimum(h - 1 - rows, w - 1 - cols))


def _paint_concentric(cell_h: int, cell_w: int, colors: list[int]) -> np.ndarray:
    layers = _layer_index_grid(cell_h, cell_w)
    out = np.zeros((cell_h, cell_w), dtype=int)
    for i, color in enumerate(colors):
        out[layers == i] = int(color)
    return out


def _repair_missing_bands(cell: np.ndarray, marker_colors: list[int]) -> np.ndarray:
    out = cell.copy()
    layers = _layer_index_grid(*cell.shape)
    assert len(marker_colors) == int(layers.max()) + 1
    for i, color in enumerate(marker_colors):
        mask = layers == i
        visible = {int(v) for v in out[mask] if int(v) != BG}
        if visible:
            assert visible == {int(color)}
        else:
            out[mask] = int(color)
    return out


def _tile_strip(cell: np.ndarray, orientation: str, copies: int) -> np.ndarray:
    if orientation == "H":
        return np.tile(cell, (1, copies))
    return np.tile(cell, (copies, 1))


def _fill_output(
    base_shape: tuple[int, int],
    region_bbox: tuple[int, int, int, int],
    region: np.ndarray,
) -> np.ndarray:
    out = np.zeros(base_shape, dtype=int)
    r0, c0, r1, c1 = region_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = region
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Repair a repeated concentric-band strip by combining translation propagation with diagonal palette recovery.

    Input structure:
    - The grid contains a down-right diagonal of singleton marker pixels in the top-left corner.
    - Below that diagonal sits one periodic strip made of repeated concentric rectangular-band cells, but many strip pixels are black.
    - Some black cells are local occlusions, while one or two whole concentric bands are erased in every repeated copy.

    Rule:
    - Remove the diagonal markers from consideration and find the unique nonzero translation with maximum visible, color-consistent support inside the strip.
    - Propagate visible colors along that translation to restore every locally missing pixel that still has a visible periodic mate.
    - Read the diagonal marker colors from outermost to innermost band order, then fill any still-all-black band of the recovered unit cell with its corresponding marker color.
    - Tile the fully repaired unit cell back across the strip and drop the diagonal markers from the output.

    Key insight:
    - Translation propagation repairs random per-copy damage, but it cannot recover bands that are absent from every copy; the diagonal palette supplies exactly those systematic missing bands.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    marker_colors = _read_marker_colors(g)
    work = g.copy()
    for i in range(len(marker_colors)):
        work[i, i] = BG

    region_bbox = _bbox(work != BG)
    assert region_bbox is not None
    r0, c0, r1, c1 = region_bbox
    crop = work[r0 : r1 + 1, c0 : c1 + 1]

    shift, best_pair, _support = _best_translation(crop)
    assert shift is not None
    assert len(best_pair) == 2 and (-shift[0], -shift[1]) in best_pair
    assert (shift[0] == 0) ^ (shift[1] == 0)

    propagated = _propagate_with_shift(crop, shift)

    if shift[0] == 0:
        orientation = "H"
        cell_h = crop.shape[0]
        cell_w = abs(shift[1])
        assert crop.shape[1] % cell_w == 0
        copies = crop.shape[1] // cell_w
        unit = propagated[:, :cell_w]
    else:
        orientation = "V"
        cell_h = abs(shift[0])
        cell_w = crop.shape[1]
        assert crop.shape[0] % cell_h == 0
        copies = crop.shape[0] // cell_h
        unit = propagated[:cell_h, :]

    repaired_unit = _repair_missing_bands(unit, marker_colors)
    repaired_region = _tile_strip(repaired_unit, orientation, copies)
    return _fill_output(g.shape, region_bbox, repaired_region)


def _random_palette(k: int) -> list[int]:
    return random.sample(range(1, 10), k)


def _cell_dims(layer_count: int) -> tuple[int, int]:
    choices = [2 * layer_count - 1, 2 * layer_count]
    return random.choice(choices), random.choice(choices)


def _erase_systemic_layers(region: np.ndarray, layers: np.ndarray, missing_layers: list[int]) -> None:
    for idx in missing_layers:
        region[layers == idx] = BG


def _random_missing_count(copies: int) -> int:
    if copies == 3:
        return random.choices([0, 1, 2], weights=[2, 5, 3])[0]
    return random.choices([0, 1, 2, 3], weights=[2, 4, 3, 1])[0]


def _apply_local_occlusion(
    region: np.ndarray,
    orientation: str,
    copies: int,
    missing_layers: set[int],
) -> int:
    cell_h, cell_w = region.shape if orientation == "H" else (region.shape[0] // copies, region.shape[1])
    if orientation == "H":
        cell_h, cell_w = region.shape[0], region.shape[1] // copies
    layers = _layer_index_grid(cell_h, cell_w)
    random_holes = 0
    if orientation == "H":
        for r in range(cell_h):
            for c in range(cell_w):
                if int(layers[r, c]) in missing_layers:
                    continue
                miss = min(_random_missing_count(copies), copies - 1)
                if miss == 0:
                    continue
                chosen = random.sample(range(copies), miss)
                for i in chosen:
                    region[r, i * cell_w + c] = BG
                    random_holes += 1
    else:
        for r in range(cell_h):
            for c in range(cell_w):
                if int(layers[r, c]) in missing_layers:
                    continue
                miss = min(_random_missing_count(copies), copies - 1)
                if miss == 0:
                    continue
                chosen = random.sample(range(copies), miss)
                for i in chosen:
                    region[i * cell_h + r, c] = BG
                    random_holes += 1
    return random_holes


def _band_only_repair(masked_region: np.ndarray, shift: tuple[int, int], marker_colors: list[int]) -> np.ndarray:
    if shift[0] == 0:
        cell = masked_region[:, : abs(shift[1])].copy()
        copies = masked_region.shape[1] // abs(shift[1])
        return _tile_strip(_repair_missing_bands(cell, marker_colors), "H", copies)
    cell = masked_region[: abs(shift[0]), :].copy()
    copies = masked_region.shape[0] // abs(shift[0])
    return _tile_strip(_repair_missing_bands(cell, marker_colors), "V", copies)


def _build_attempt() -> np.ndarray | None:
    layers = random.choice([3, 4])
    palette = _random_palette(layers)
    cell_h, cell_w = _cell_dims(layers)
    cell = _paint_concentric(cell_h, cell_w, palette)
    orientation = random.choice(["H", "V"])

    copies_choices = [3, 4]
    random.shuffle(copies_choices)
    copies = None
    for candidate in copies_choices:
        region_h = cell_h if orientation == "H" else cell_h * candidate
        region_w = cell_w * candidate if orientation == "H" else cell_w
        top = layers + random.randint(1, 2)
        left = random.randint(0, 3)
        bottom = random.randint(1, 3)
        right = random.randint(1, 3)
        grid_h = top + region_h + bottom
        grid_w = left + region_w + right
        if grid_h <= 30 and grid_w <= 30:
            copies = candidate
            break
    if copies is None:
        return None

    region = _tile_strip(cell, orientation, copies)
    layer_grid = _layer_index_grid(cell_h, cell_w)
    repeated_layers = _tile_strip(layer_grid, orientation, copies)

    missing_candidates = list(range(1, layers))
    if not missing_candidates:
        return None
    n_missing = random.choice([1, 2]) if len(missing_candidates) >= 2 else 1
    missing_layers = random.sample(missing_candidates, n_missing)
    _erase_systemic_layers(region, repeated_layers, missing_layers)

    random_holes = _apply_local_occlusion(region, orientation, copies, set(missing_layers))
    if random_holes == 0:
        return None

    if _bbox(region != BG) != (0, 0, region.shape[0] - 1, region.shape[1] - 1):
        return None

    expected_shift = (0, cell_w) if orientation == "H" else (cell_h, 0)
    shift, best_pair, _support = _best_translation(region)
    if shift != expected_shift or set(best_pair) != {expected_shift, (-expected_shift[0], -expected_shift[1])}:
        return None

    propagated = _propagate_with_shift(region, expected_shift)
    if not np.any(propagated == BG):
        return None

    final_unit = _repair_missing_bands(
        propagated[:, :cell_w] if orientation == "H" else propagated[:cell_h, :],
        palette,
    )
    expected_region = _tile_strip(final_unit, orientation, copies)
    if np.any(expected_region == BG):
        return None

    band_only = _band_only_repair(region, expected_shift, palette)
    if np.array_equal(band_only, expected_region):
        return None

    top = layers + random.randint(1, 2)
    left = random.randint(0, 3)
    bottom = random.randint(1, 3)
    right = random.randint(1, 3)
    grid_h = top + region.shape[0] + bottom
    grid_w = left + region.shape[1] + right
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for i, color in enumerate(palette):
        grid[i, i] = int(color)
    grid[top : top + region.shape[0], left : left + region.shape[1]] = region

    out = main(grid)
    expected = _fill_output(
        grid.shape,
        (top, left, top + region.shape[0] - 1, left + region.shape[1] - 1),
        expected_region,
    )
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid periodic concentric-band repair input")
