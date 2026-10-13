from __future__ import annotations

from collections import Counter

import numpy as np


_RNG = np.random.default_rng()
_GEN_RETRY_LIMIT = 500
_TRANSFORM_NAMES = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flip_lr",
    "flip_ud",
    "transpose",
    "anti_transpose",
)


def get_bg(grid: np.ndarray) -> int:
    """Return the dominant background color, following arc_utils semantics."""
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a grid to an inclusive bbox, mirroring arc_utils."""
    r0, c0, r1, c1 = map(int, bbox)
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(
    dst: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    *,
    transparent: int | None = None,
) -> np.ndarray:
    """Paste a patch at top-left (r0, c0), reusing arc_utils' canonical API shape."""
    r0 = int(r0)
    c0 = int(c0)
    h, w = map(int, patch.shape)
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:, :] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]
    return dst


def _apply_transform(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if name == "identity":
        return arr.copy()
    if name == "rot90":
        return np.rot90(arr, 1).copy()
    if name == "rot180":
        return np.rot90(arr, 2).copy()
    if name == "rot270":
        return np.rot90(arr, 3).copy()
    if name == "flip_lr":
        return np.fliplr(arr).copy()
    if name == "flip_ud":
        return np.flipud(arr).copy()
    if name == "transpose":
        return arr.T.copy()
    if name == "anti_transpose":
        return np.fliplr(np.rot90(arr, 1)).copy()
    raise ValueError(f"unknown transform {name}")


def _patch_key(patch: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in np.asarray(patch).ravel())


def _contiguous_groups(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask_1d)
    if idx.size == 0:
        return []
    groups: list[tuple[int, int]] = []
    start = prev = int(idx[0])
    for raw_v in idx[1:]:
        v = int(raw_v)
        if v == prev + 1:
            prev = v
            continue
        groups.append((start, prev))
        start = prev = v
    groups.append((start, prev))
    return groups


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(int(v) for v in coords[0])
    stack = [start]
    seen = {start}
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _random_pattern_mask(tile_size: int) -> np.ndarray:
    for _ in range(300):
        density = float(_RNG.uniform(0.35, 0.7))
        mask = _RNG.random((tile_size, tile_size)) < density
        area = int(mask.sum())
        if area < tile_size + 1 or area > tile_size * tile_size - 2:
            continue
        if not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1)):
            continue
        if not _is_connected(mask):
            continue
        variants = {_patch_key(_apply_transform(mask.astype(int), name)) for name in _TRANSFORM_NAMES}
        if len(variants) != len(_TRANSFORM_NAMES):
            continue
        return mask
    raise RuntimeError("failed to generate an asymmetric tile pattern")


def _random_layout_mask(n: int, transform_name: str) -> np.ndarray:
    min_occupied = max(3, n)
    for _ in range(300):
        density = float(_RNG.uniform(0.35, 0.7))
        layout = _RNG.random((n, n)) < density
        occupied = int(layout.sum())
        if occupied < min_occupied or occupied >= n * n:
            continue
        if not np.all(layout.any(axis=0)) or not np.all(layout.any(axis=1)):
            continue
        if np.array_equal(_apply_transform(layout, transform_name), layout):
            continue
        return layout
    raise RuntimeError("failed to generate a non-symmetric macro layout")


def _build_grid(
    layout: np.ndarray,
    tile_patch: np.ndarray,
    anomaly_transform: str,
    anomaly_site: tuple[int, int],
    gap: int,
    margins: tuple[int, int, int, int],
    bg: int,
) -> np.ndarray:
    n = int(layout.shape[0])
    tile_size = int(tile_patch.shape[0])
    top, bottom, left, right = map(int, margins)
    base = n * tile_size + (n - 1) * int(gap)
    h = top + base + bottom
    w = left + base + right
    grid = np.full((h, w), int(bg), dtype=int)
    anomaly_patch = _apply_transform(tile_patch, anomaly_transform)
    for r in range(n):
        for c in range(n):
            if not layout[r, c]:
                continue
            rr = top + r * (tile_size + gap)
            cc = left + c * (tile_size + gap)
            patch = anomaly_patch if (r, c) == anomaly_site else tile_patch
            paste(grid, patch, rr, cc)
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the uniquely reoriented tile, apply that same transform to the whole tile layout, and redraw every occupied site with the standard tile.

    Input structure:
    - A dominant background surrounds equally sized square tile windows laid out on a regular square lattice with fixed gaps.
    - Every occupied window contains the same monochrome pattern, except for one anomalous window whose pattern is a rotated or reflected copy.
    - Empty lattice sites are pure background, and each lattice row and column has at least one occupied site.

    Rule:
    - Recover the lattice by grouping occupied rows and columns into same-sized tile bands.
    - Identify the majority tile as the standard pattern and the lone different tile as the transformed anomaly.
    - Determine the unique dihedral transform from the standard tile to the anomaly, apply that same transform to the macro occupancy mask, and redraw the transformed layout using only the standard tile orientation.

    Key insight:
    - The odd tile is not a defect to repair locally; it is a cross-scale instruction whose local orientation tells you how to move the entire macro layout.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    row_groups = _contiguous_groups(np.any(g != bg, axis=1))
    col_groups = _contiguous_groups(np.any(g != bg, axis=0))
    assert row_groups and col_groups, "expected at least one occupied tile band"
    assert len(row_groups) == len(col_groups), "macro lattice must be square"

    row_heights = {r1 - r0 + 1 for r0, r1 in row_groups}
    col_widths = {c1 - c0 + 1 for c0, c1 in col_groups}
    assert len(row_heights) == 1 and len(col_widths) == 1, "tile bands must have uniform size"
    tile_size = row_heights.pop()
    assert tile_size == col_widths.pop(), "tile windows must be square"

    n = len(row_groups)
    occupancy = np.zeros((n, n), dtype=bool)
    patches: list[tuple[tuple[int, int], np.ndarray]] = []
    for r_idx, (r0, r1) in enumerate(row_groups):
        for c_idx, (c0, c1) in enumerate(col_groups):
            patch = crop_bbox(g, (r0, c0, r1, c1))
            if np.any(patch != bg):
                occupancy[r_idx, c_idx] = True
                patches.append(((r_idx, c_idx), patch))

    assert len(patches) >= 3, "expected at least three occupied tiles"
    assert np.all(occupancy.any(axis=0)) and np.all(occupancy.any(axis=1)), "every macro row/column must be represented"

    counts = Counter(_patch_key(patch) for _site, patch in patches)
    assert len(counts) == 2, "expected one standard tile type and one anomaly"
    standard_key, standard_count = counts.most_common(1)[0]
    anomaly_keys = [key for key, count in counts.items() if count == 1 and key != standard_key]
    assert len(anomaly_keys) == 1, "expected exactly one anomalous tile"
    assert standard_count == len(patches) - 1, "all non-anomalous tiles must be identical"
    anomaly_key = anomaly_keys[0]

    standard_patch = next(patch for _site, patch in patches if _patch_key(patch) == standard_key)
    anomaly_patch = next(patch for _site, patch in patches if _patch_key(patch) == anomaly_key)

    matches = [
        name
        for name in _TRANSFORM_NAMES[1:]
        if np.array_equal(_apply_transform(standard_patch, name), anomaly_patch)
    ]
    assert len(matches) == 1, "anomaly transform must be uniquely identifiable"
    transform_name = matches[0]

    transformed_layout = _apply_transform(occupancy, transform_name).astype(bool)
    out = np.full_like(g, bg)
    for r_idx, (r0, _r1) in enumerate(row_groups):
        for c_idx, (c0, _c1) in enumerate(col_groups):
            if transformed_layout[r_idx, c_idx]:
                paste(out, standard_patch, r0, c0, transparent=None)
    return out


def _gen_attempt() -> np.ndarray | None:
    n = int(_RNG.integers(2, 6))
    tile_size = int(_RNG.integers(3, 6))
    gap = int(_RNG.integers(1, 4))
    base = n * tile_size + (n - 1) * gap
    if base > 30:
        return None

    extra_h = 30 - base
    extra_w = 30 - base
    top = int(_RNG.integers(0, min(4, extra_h) + 1))
    bottom = int(_RNG.integers(0, min(4, extra_h - top) + 1))
    left = int(_RNG.integers(0, min(4, extra_w) + 1))
    right = int(_RNG.integers(0, min(4, extra_w - left) + 1))

    bg = int(_RNG.integers(0, 10))
    fg_choices = [color for color in range(10) if color != bg]
    fg = int(fg_choices[int(_RNG.integers(0, len(fg_choices)))])

    transform_name = _TRANSFORM_NAMES[int(_RNG.integers(1, len(_TRANSFORM_NAMES)))]
    pattern_mask = _random_pattern_mask(tile_size)
    tile_patch = np.full((tile_size, tile_size), bg, dtype=int)
    tile_patch[pattern_mask] = fg

    layout = _random_layout_mask(n, transform_name)
    sites = np.argwhere(layout)
    anomaly_site = tuple(int(v) for v in sites[int(_RNG.integers(0, len(sites)))])
    grid = _build_grid(layout, tile_patch, transform_name, anomaly_site, gap, (top, bottom, left, right), bg)

    if max(grid.shape) > 30:
        return None
    if int(np.sum(grid == bg)) <= int(np.sum(grid != bg)):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cross-scale anomaly sample")
