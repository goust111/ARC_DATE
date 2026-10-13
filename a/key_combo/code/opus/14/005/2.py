from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIVIDER = 0
REGION_BG = 5
SPRITE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400
LAYOUTS = ((2, 2), (2, 3), (3, 2), (2, 4), (4, 2))
CELL_SIZES = (3, 4)
TEMPLATES = (
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 1)),
)


def _normalize_coords(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _c in coords)
    c0 = min(c for _r, c in coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _rotate_coords(coords: tuple[tuple[int, int], ...], k: int) -> tuple[tuple[int, int], ...]:
    pts = np.array(coords, dtype=int)
    h = int(pts[:, 0].max()) + 1
    w = int(pts[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[pts[:, 0], pts[:, 1]] = True
    rotated = np.rot90(mask, int(k))
    return _normalize_coords([tuple(map(int, p)) for p in np.argwhere(rotated)])


def _crop_binary(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise AssertionError("expected non-empty sprite")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _non_divider_runs(flags: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, is_divider in enumerate(flags.tolist()):
        if not is_divider and start is None:
            start = int(idx)
        elif is_divider and start is not None:
            runs.append((int(start), int(idx)))
            start = None
    if start is not None:
        runs.append((int(start), int(len(flags))))
    return runs


def _region_slices(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_dividers = np.all(grid == int(DIVIDER), axis=1)
    col_dividers = np.all(grid == int(DIVIDER), axis=0)
    row_runs = _non_divider_runs(row_dividers)
    col_runs = _non_divider_runs(col_dividers)
    assert row_runs and col_runs, "expected at least one partition row and column"
    return row_runs, col_runs


def _extract_sprite(region: np.ndarray) -> tuple[int, np.ndarray]:
    vals = [int(v) for v in region.flatten() if int(v) != int(DIVIDER)]
    assert vals, "region must contain non-divider cells"
    bg = Counter(vals).most_common(1)[0][0]
    colors = sorted({int(v) for v in vals if int(v) != int(bg)})
    assert len(colors) == 1, "expected exactly one sprite color per region"
    color = int(colors[0])
    sprite_mask = region == int(color)
    return color, _crop_binary(sprite_mask)


def _decode_regions(grid: np.ndarray) -> list[tuple[int, int]]:
    row_runs, col_runs = _region_slices(grid)
    entries: list[tuple[int, int]] = []

    ref_region = grid[row_runs[0][0] : row_runs[0][1], col_runs[0][0] : col_runs[0][1]]
    ref_color, ref_sprite = _extract_sprite(ref_region)
    del ref_color

    rotations = [np.rot90(ref_sprite, k) for k in range(4)]
    assert len({(rot.shape, rot.tobytes()) for rot in rotations}) == 4, "reference sprite must be asymmetric under rotation"

    read_index = 0
    for rr in row_runs:
        for cc in col_runs:
            region = grid[rr[0] : rr[1], cc[0] : cc[1]]
            color, sprite = _extract_sprite(region)
            matches = [k for k, rotated in enumerate(rotations) if sprite.shape == rotated.shape and np.array_equal(sprite, rotated)]
            assert len(matches) == 1, "sprite rotation must match the reference uniquely"
            entries.append((read_index, int(matches[0]), int(color)))
            read_index += 1

    ordered = sorted(entries, key=lambda item: (item[1], item[0]))
    return [(int(color), int(code)) for _idx, code, color in ordered]


def _build_output(colors_in_order: list[int]) -> np.ndarray:
    assert colors_in_order, "expected at least one color"
    cur = np.full((1, 1), int(colors_in_order[0]), dtype=int)
    for color in colors_in_order[1:]:
        nxt = np.full((cur.shape[0] + 2, cur.shape[1] + 2), int(color), dtype=int)
        nxt[1:-1, 1:-1] = cur
        cur = nxt
    return cur


def main(grid: np.ndarray) -> np.ndarray:
    """Decode per-region sprite rotations into an inner-to-outer nested wrapping order.

    Input structure:
    - The grid is partitioned by single-cell black divider rows and columns into equal square regions.
    - Each region contains a gray background and exactly one small connected monochrome sprite.
    - All sprites share one template up to quarter-turn rotation; the top-left region provides the reference orientation.

    Rule:
    - Locate the rectangular regions between the black divider lines.
    - Read regions in row-major order and extract each region's sprite color and its rotation relative to the top-left reference: 0, 90, 180, or 270 degrees counterclockwise.
    - Stable-sort the regions by that rotation code, keeping row-major order among ties.
    - Build the output from the sorted colors: the first color is the 1x1 center square, and each later color wraps the current shape with a one-cell square frame.

    Key insight:
    - The rotated sprite is not the visual target itself; it is a hidden ordering signal that determines how the region colors are nested from the center outward.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    decoded = _decode_regions(g)
    ordered_colors = [int(color) for color, _code in decoded]
    return _build_output(ordered_colors)


def _choose_codes(count: int) -> list[int]:
    assert count >= 4
    codes = [0, 1, 2, 3]
    while len(codes) < int(count):
        codes.append(int(random.randint(0, 3)))
    tail = codes[1:]
    random.shuffle(tail)
    return [0] + tail


def _sprite_patch(template: tuple[tuple[int, int], ...], rotation: int, cell_size: int, color: int) -> np.ndarray:
    coords = _rotate_coords(template, rotation)
    h = max(r for r, _c in coords) + 1
    w = max(c for _r, c in coords) + 1
    assert h <= int(cell_size) and w <= int(cell_size)
    patch = np.full((int(cell_size), int(cell_size)), int(REGION_BG), dtype=int)
    r0 = int(random.randint(0, int(cell_size) - h))
    c0 = int(random.randint(0, int(cell_size) - w))
    for r, c in coords:
        patch[int(r0 + r), int(c0 + c)] = int(color)
    return patch


def _assemble_grid(rows: int, cols: int, cell_size: int, template: tuple[tuple[int, int], ...], colors: list[int], codes: list[int]) -> np.ndarray:
    h = int(rows) * int(cell_size) + (int(rows) - 1)
    w = int(cols) * int(cell_size) + (int(cols) - 1)
    grid = np.full((h, w), int(REGION_BG), dtype=int)

    idx = 0
    for r in range(int(rows)):
        for c in range(int(cols)):
            top = r * (int(cell_size) + 1)
            left = c * (int(cell_size) + 1)
            patch = _sprite_patch(template, int(codes[idx]), int(cell_size), int(colors[idx]))
            grid[top : top + int(cell_size), left : left + int(cell_size)] = patch
            idx += 1

    for r in range(1, int(rows)):
        divider_row = r * int(cell_size) + (r - 1)
        grid[divider_row, :] = int(DIVIDER)
    for c in range(1, int(cols)):
        divider_col = c * int(cell_size) + (c - 1)
        grid[:, divider_col] = int(DIVIDER)
    return grid


def _expected_from_layout(colors: list[int], codes: list[int]) -> np.ndarray:
    order = sorted(range(len(colors)), key=lambda idx: (int(codes[idx]), int(idx)))
    ordered_colors = [int(colors[idx]) for idx in order]
    return _build_output(ordered_colors)


def _gen_attempt() -> np.ndarray | None:
    rows, cols = random.choice(LAYOUTS)
    count = int(rows) * int(cols)
    cell_size = int(random.choice(CELL_SIZES))
    template = _normalize_coords(random.choice(TEMPLATES))
    if len({_rotate_coords(template, k) for k in range(4)}) != 4:
        return None

    colors = random.sample(list(SPRITE_COLORS), count)
    codes = _choose_codes(count)
    grid = _assemble_grid(rows, cols, cell_size, template, colors, codes)

    if max(grid.shape) > 30:
        return None

    row_runs, col_runs = _region_slices(grid)
    if len(row_runs) * len(col_runs) != count:
        return None
    if any((r1 - r0) != int(cell_size) for r0, r1 in row_runs):
        return None
    if any((c1 - c0) != int(cell_size) for c0, c1 in col_runs):
        return None

    expected = _expected_from_layout(colors, codes)
    if not np.array_equal(main(grid), expected):
        return None

    decoded_codes = [code for _color, code in _decode_regions(grid)]
    if set(decoded_codes) != {0, 1, 2, 3}:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
