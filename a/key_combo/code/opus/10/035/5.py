from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
FG = 3
BRANCHES = (
    "same_bbox_same_area",
    "same_bbox_diff_area",
    "diff_bbox_same_area",
    "diff_bbox_diff_area",
)


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    height, width = mask.shape
    stack = [tuple(map(int, coords[0]))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c, height, width):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _split_bands(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    occupied_rows = [not np.all(grid[r] == bg) for r in range(grid.shape[0])]
    bands: list[tuple[int, int]] = []
    r = 0
    while r < len(occupied_rows):
        if not occupied_rows[r]:
            r += 1
            continue
        start = r
        while r < len(occupied_rows) and occupied_rows[r]:
            r += 1
        bands.append((start, r))
    return bands


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return mask.shape, np.packbits(mask.astype(np.uint8), bitorder="little").tobytes()


def _extract_band_object(
    grid: np.ndarray, band_start: int, band_end: int, bg: int
) -> dict[str, object]:
    band = grid[band_start:band_end]
    mask = band != bg
    assert np.any(mask), "each band must contain a sprite"
    assert _is_connected(mask), "each band must contain one 4-connected sprite"
    colors = np.unique(band[mask])
    assert len(colors) == 1, "sprite must be monochrome"
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)
    shape = mask[r0 : r1 + 1, c0 : c1 + 1].copy()
    return {
        "band_start": band_start,
        "band_end": band_end,
        "color": int(colors[0]),
        "bbox": (r0, c0, r1, c1),
        "shape": shape,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift the unique band sprite right by the repeated-shape count and erase the repeated sprites.

    Input structure:
    - A black background grid is split into 4-6 horizontal bands by full-width black separator rows.
    - Each non-separator band contains exactly one small 4-connected monochrome sprite on the same left anchor column.
    - Exactly one band has a different sprite shape; the other bands share one identical shape.

    Rule:
    - Split the grid into horizontal bands using the full-width background rows.
    - Compare the band sprites by their cropped shape masks, find the one unique shape, and let K be the number of repeated-shape bands.
    - Erase every repeated sprite, then move the unique sprite K columns to the right without changing its rows.

    Key insight:
    - The repeated sprites are not the output target; their count is converted into the horizontal shift distance for the odd one out.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _dominant_color(g)
    bands = _split_bands(g, bg)
    assert 4 <= len(bands) <= 6, "expected 4-6 horizontal bands"
    band_heights = {end - start for start, end in bands}
    assert len(band_heights) == 1, "expected equal-height bands"

    objects = [_extract_band_object(g, start, end, bg) for start, end in bands]
    colors = {obj["color"] for obj in objects}
    assert len(colors) == 1, "expected one shared sprite color across bands"

    keys = [_shape_key(obj["shape"]) for obj in objects]
    counts = Counter(keys)
    repeated = [key for key, count in counts.items() if count > 1]
    unique = [key for key, count in counts.items() if count == 1]
    assert len(repeated) == 1 and len(unique) == 1, "expected one repeated shape and one unique shape"
    shift = counts[repeated[0]]

    unique_index = next(i for i, key in enumerate(keys) if key == unique[0])
    unique_obj = objects[unique_index]
    r0, c0, r1, c1 = unique_obj["bbox"]
    shape = unique_obj["shape"]
    out = np.full_like(g, bg)
    for dr, dc in np.argwhere(shape):
        rr = int(unique_obj["band_start"]) + r0 + int(dr)
        cc = c0 + int(dc) + shift
        assert cc < g.shape[1], "shifted unique sprite must stay inside the grid"
        out[rr, cc] = int(unique_obj["color"])
    return out


def _enumerate_shapes(height: int) -> list[np.ndarray]:
    shapes: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for width in (2, 3):
        total = height * width
        min_cells = max(3, height)
        for bits in range(1 << total):
            if bits.bit_count() < min_cells or bits.bit_count() > 6:
                continue
            mask = np.array(
                [(bits >> idx) & 1 for idx in range(total)],
                dtype=bool,
            ).reshape(height, width)
            if not np.all(mask.any(axis=1)):
                continue
            if not np.all(mask.any(axis=0)):
                continue
            if not _is_connected(mask):
                continue
            key = _shape_key(mask)
            if key not in seen:
                seen.add(key)
                shapes.append(mask)
    return shapes


SHAPES_BY_HEIGHT = {height: _enumerate_shapes(height) for height in (3, 4)}


def _pick_shape_pair(height: int, branch: str) -> tuple[np.ndarray, np.ndarray]:
    shapes = SHAPES_BY_HEIGHT[height]
    order = list(range(len(shapes)))
    random.shuffle(order)
    for idx in order:
        repeated = shapes[idx]
        rep_width = repeated.shape[1]
        rep_area = int(repeated.sum())
        candidates = []
        for unique in shapes:
            if np.array_equal(unique, repeated):
                continue
            same_width = unique.shape[1] == rep_width
            same_area = int(unique.sum()) == rep_area
            if branch == "same_bbox_same_area" and same_width and same_area:
                candidates.append(unique)
            if branch == "same_bbox_diff_area" and same_width and not same_area:
                candidates.append(unique)
            if branch == "diff_bbox_same_area" and not same_width and same_area:
                candidates.append(unique)
            if branch == "diff_bbox_diff_area" and not same_width and not same_area:
                candidates.append(unique)
        if candidates:
            return repeated.copy(), random.choice(candidates).copy()
    raise RuntimeError(f"no shape pair available for {height=}, {branch=}")


def _expected_output(
    shape: tuple[int, int],
    band_height: int,
    unique_band: int,
    left: int,
    shift: int,
    unique: np.ndarray,
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    band_start = unique_band * (band_height + 1)
    for dr, dc in np.argwhere(unique):
        grid[band_start + int(dr), left + int(dc) + shift] = FG
    return grid


def _validate_branch(repeated: np.ndarray, unique: np.ndarray, branch: str) -> None:
    rep_width = repeated.shape[1]
    uniq_width = unique.shape[1]
    rep_area = int(repeated.sum())
    uniq_area = int(unique.sum())
    if branch == "same_bbox_same_area":
        assert rep_width == uniq_width and rep_area == uniq_area
    elif branch == "same_bbox_diff_area":
        assert rep_width == uniq_width and rep_area != uniq_area
    elif branch == "diff_bbox_same_area":
        assert rep_width != uniq_width and rep_area == uniq_area
    elif branch == "diff_bbox_diff_area":
        assert rep_width != uniq_width and rep_area != uniq_area
    else:
        raise AssertionError(f"unknown branch {branch}")


def _generate_one() -> np.ndarray:
    band_height = random.choice((3, 4))
    band_count = random.randint(4, 6)
    branch = random.choice(BRANCHES)
    repeated, unique = _pick_shape_pair(band_height, branch)
    unique_band = random.randrange(band_count)
    shift = band_count - 1

    max_width = max(repeated.shape[1], unique.shape[1])
    min_width = max(10, max_width + shift + 4)
    width = random.randint(min_width, min(20, min_width + 6))
    max_left = width - max_width - shift - 1
    left = random.randint(1, max_left)
    height = band_count * band_height + (band_count - 1)

    grid = np.full((height, width), BG, dtype=int)
    for band_idx in range(band_count):
        band_start = band_idx * (band_height + 1)
        sprite = unique if band_idx == unique_band else repeated
        grid[band_start : band_start + band_height, left : left + sprite.shape[1]][sprite] = FG

    bands = _split_bands(grid, BG)
    assert len(bands) == band_count
    assert all(end - start == band_height for start, end in bands)
    objects = [_extract_band_object(grid, start, end, BG) for start, end in bands]
    assert all(int(obj["bbox"][1]) == left for obj in objects), "sprites must share one left anchor column"
    counts = Counter(_shape_key(obj["shape"]) for obj in objects)
    assert sorted(counts.values()) == [1, band_count - 1]
    _validate_branch(repeated, unique, branch)

    expected = _expected_output(grid.shape, band_height, unique_band, left, shift, unique)
    actual = main(grid)
    assert np.array_equal(actual, expected), "solver output must match the intended semantic output"

    actual_mask = actual == FG
    assert _is_connected(actual_mask), "output must contain one connected sprite"
    coords = np.argwhere(actual_mask)
    _, out_c0 = coords.min(axis=0)
    assert int(out_c0) == left + shift, "unique sprite must move right by exactly the repeated count"
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            return _generate_one()
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
