from __future__ import annotations

from collections import Counter
import math
import random

import numpy as np


BACKGROUND = 0
FRAME_COLORS = (1, 2, 3)
MAX_ATTEMPTS = 400


def _runs(bits: np.ndarray) -> list[tuple[int, int]]:
    bits = np.asarray(bits, dtype=bool)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(bits):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(bits) - 1))
    return runs


def _tile_band(mask: np.ndarray, width: int) -> np.ndarray:
    tile_w = mask.shape[1]
    reps = int(math.ceil(width / tile_w))
    return np.tile(mask, (1, reps))[:, :width]


def _extract_frames(grid: np.ndarray) -> tuple[int, list[tuple[int, int, int, np.ndarray]], dict[int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    frame_mask = np.zeros_like(g, dtype=bool)
    frame_mask[:, 1:] = g[:, 1:] != BACKGROUND

    row_runs = _runs(np.any(frame_mask, axis=1))
    col_runs = _runs(np.any(frame_mask, axis=0))
    assert row_runs and col_runs, "expected an upper frame region"

    frame_size = row_runs[0][1] - row_runs[0][0] + 1
    assert frame_size >= 5, "expected square frames with nontrivial interiors"
    assert all(r1 - r0 + 1 == frame_size for r0, r1 in row_runs)
    assert all(c1 - c0 + 1 == frame_size for c0, c1 in col_runs)

    frames: list[tuple[int, int, int, np.ndarray]] = []
    unions: dict[int, np.ndarray] = {}

    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            block = g[r0 : r1 + 1, c0 : c1 + 1]
            if not np.any(block):
                continue

            colors = sorted(set(map(int, block[block != BACKGROUND])))
            assert len(colors) == 1, "each frame must use exactly one visible color"
            color = colors[0]
            assert np.all(block[0, :] == color)
            assert np.all(block[-1, :] == color)
            assert np.all(block[:, 0] == color)
            assert np.all(block[:, -1] == color)

            interior = block[1:-1, 1:-1]
            assert np.all((interior == BACKGROUND) | (interior == color))
            mask = interior == color

            if color not in unions:
                unions[color] = np.zeros_like(mask, dtype=bool)
            unions[color] |= mask
            frames.append((r0, c0, color, mask))

    assert 4 <= len(frames) <= 6, "expected 4-6 frames"
    counts = Counter(color for _, _, color, _ in frames)
    assert len(counts) in (2, 3), "expected 2-3 frame colors"
    assert all(2 <= count <= 3 for count in counts.values())
    return frame_size, frames, unions


def main(grid: np.ndarray) -> np.ndarray:
    """Group same-colored frames, union their interior motifs, then tile each guide-selected band with the matching motif.

    Input structure:
    A black background grid has an upper region of equal-size square frames and a lower region separated by one empty row.
    Each frame has a solid one-pixel border in one of a few colors, and its interior contains only black or that same color.
    The lower region contains one guide pixel per used frame color in the leftmost column; each guide row marks a horizontal band.

    Rule:
    Detect every square frame, group frames by border color, and union the relative interior colored cells within each color group.
    That union becomes the color's canonical tile. For every guide pixel, fill the band starting at that row and extending down by
    one interior height using the matching color's tile repeated horizontally from column 1 onward. Everything else becomes black.

    Key insight:
    The upper frames define motif content while the lower guide pixels define placement; color matching is the only bridge between them.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)

    frame_size, frames, unions = _extract_frames(g)
    tile_size = frame_size - 2

    frame_bottom = max(r0 + frame_size - 1 for r0, _, _, _ in frames)
    guides = [(r, int(g[r, 0])) for r in range(h) if int(g[r, 0]) != BACKGROUND]
    assert len(guides) == len(unions), "expected one guide per color"
    assert {color for _, color in guides} == set(unions), "guide colors must match frame colors"
    guide_rows = [r for r, _ in guides]
    assert min(guide_rows) - frame_bottom == 2, "expected one blank separator row"
    assert all(b - a == tile_size for a, b in zip(guide_rows, guide_rows[1:])), "guide bands must be evenly spaced"
    assert all(r + tile_size <= h for r in guide_rows), "guide band out of bounds"
    assert not np.any(g[min(guide_rows) :, 1:]), "lower region should only contain guide pixels"

    out = np.zeros_like(g)
    for start_row, color in guides:
        band_mask = _tile_band(unions[color], w - 1)
        band = out[start_row : start_row + tile_size, 1:]
        band[band_mask] = color

    return out


def _random_tile(tile_size: int, frame_count: int, existing: list[np.ndarray]) -> np.ndarray | None:
    area = tile_size * tile_size
    min_fill = max(frame_count + 1, tile_size + 1)
    max_fill = area - 1
    if min_fill > max_fill:
        return None

    for _ in range(80):
        fill = random.randint(min_fill, max_fill)
        mask = np.zeros((tile_size, tile_size), dtype=bool)
        picks = random.sample(range(area), fill)
        for idx in picks:
            mask[idx // tile_size, idx % tile_size] = True

        if any(np.array_equal(mask, prev) for prev in existing):
            continue
        if not all(mask[r].any() and (~mask[r]).any() for r in range(tile_size)):
            continue
        if not all(mask[:, c].any() and (~mask[:, c]).any() for c in range(tile_size)):
            continue
        return mask

    return None


def _split_tile(mask: np.ndarray, frame_count: int) -> list[np.ndarray] | None:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if len(coords) <= frame_count:
        return None

    for _ in range(120):
        random.shuffle(coords)
        parts = [np.zeros_like(mask, dtype=bool) for _ in range(frame_count)]
        unique_cells: list[tuple[int, int]] = []
        for idx in range(frame_count):
            rc = coords[idx]
            parts[idx][rc] = True
            unique_cells.append(rc)

        for rc in coords[frame_count:]:
            parts[random.randrange(frame_count)][rc] = True

        for rc in coords[frame_count:]:
            if random.random() < 0.28:
                extra = random.randrange(frame_count)
                parts[extra][rc] = True

        if any(not part.any() for part in parts):
            continue
        if any(np.array_equal(part, mask) for part in parts):
            continue
        if len({part.tobytes() for part in parts}) != frame_count:
            continue
        if not np.array_equal(np.logical_or.reduce(parts), mask):
            continue

        essential = True
        for idx, part in enumerate(parts):
            others = [parts[j] for j in range(frame_count) if j != idx]
            other_union = np.logical_or.reduce(others) if others else np.zeros_like(mask, dtype=bool)
            if not np.any(part & ~other_union):
                essential = False
                break
        if essential:
            return parts

    return None


def _build_expected(shape: tuple[int, int], guides: list[tuple[int, int]], tiles: dict[int, np.ndarray]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    _, w = shape
    tile_size = next(iter(tiles.values())).shape[0]
    for start_row, color in guides:
        band_mask = _tile_band(tiles[color], w - 1)
        out[start_row : start_row + tile_size, 1:][band_mask] = color
    return out


def _frame_block(color: int, interior_mask: np.ndarray) -> np.ndarray:
    tile_size = interior_mask.shape[0]
    frame = np.zeros((tile_size + 2, tile_size + 2), dtype=int)
    frame[0, :] = color
    frame[-1, :] = color
    frame[:, 0] = color
    frame[:, -1] = color
    frame[1:-1, 1:-1][interior_mask] = color
    return frame


def _choose_color_counts(total_frames: int) -> list[int]:
    if total_frames == 4:
        return [2, 2]
    if total_frames == 5:
        return random.choice(([2, 3], [3, 2]))
    return random.choice(([3, 3], [2, 2, 2]))


def _gen_attempt() -> np.ndarray | None:
    tile_size = random.choice((3, 4))
    frame_size = tile_size + 2
    total_frames = random.choice((4, 5, 6))
    color_counts = _choose_color_counts(total_frames)
    color_count = len(color_counts)
    colors = random.sample(FRAME_COLORS, color_count)

    tiles: dict[int, np.ndarray] = {}
    parts_by_color: dict[int, list[np.ndarray]] = {}
    existing_tiles: list[np.ndarray] = []
    for color, count in zip(colors, color_counts):
        tile = _random_tile(tile_size, count, existing_tiles)
        if tile is None:
            return None
        parts = _split_tile(tile, count)
        if parts is None:
            return None
        tiles[color] = tile
        parts_by_color[color] = parts
        existing_tiles.append(tile)

    cols = 2 if total_frames == 4 else 3
    rows = 2
    row_gap = 1
    col_gap = random.choice((1, 2))
    top_pad = random.choice((1, 2))
    left_pad = random.choice((1, 2))
    right_pad = random.randint(0, tile_size)
    bottom_pad = random.choice((0, 1))

    frame_region_h = rows * frame_size + (rows - 1) * row_gap
    frame_region_w = cols * frame_size + (cols - 1) * col_gap
    height = top_pad + frame_region_h + 1 + color_count * tile_size + bottom_pad
    width = 1 + left_pad + frame_region_w + right_pad
    if height > 30 or width > 30:
        return None

    grid = np.zeros((height, width), dtype=int)

    slots = [(r, c) for r in range(rows) for c in range(cols)]
    chosen_slots = random.sample(slots, total_frames)
    frame_specs: list[tuple[int, np.ndarray]] = []
    for color in colors:
        for part in parts_by_color[color]:
            frame_specs.append((color, part))
    random.shuffle(frame_specs)
    random.shuffle(chosen_slots)

    for (slot_r, slot_c), (color, part) in zip(chosen_slots, frame_specs):
        r0 = top_pad + slot_r * (frame_size + row_gap)
        c0 = 1 + left_pad + slot_c * (frame_size + col_gap)
        grid[r0 : r0 + frame_size, c0 : c0 + frame_size] = _frame_block(color, part)

    first_band_row = top_pad + frame_region_h + 1
    guide_order = random.sample(colors, len(colors))
    guides = [(first_band_row + idx * tile_size, color) for idx, color in enumerate(guide_order)]
    for row, color in guides:
        grid[row, 0] = color

    if np.any(grid[first_band_row:, 1:]):
        return None

    for color in colors:
        parts = parts_by_color[color]
        tile = tiles[color]
        if not np.array_equal(np.logical_or.reduce(parts), tile):
            return None
        if any(np.array_equal(part, tile) for part in parts):
            return None
        if len({part.tobytes() for part in parts}) != len(parts):
            return None

    expected = _build_expected(grid.shape, guides, tiles)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    # Reject trivial instances where every band is a solid fill, which would make the frame union step redundant.
    if any(tile.all() or (~tile).all() for tile in tiles.values()):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
