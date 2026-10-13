from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
MARKER = 8
PALETTE = [1, 2, 3, 4, 5, 6, 7, 9]
MAX_GENERATION_ATTEMPTS = 600


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _nonblack_bbox(grid: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(grid != BG)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _dominant_nonblack_color(row: np.ndarray) -> int | None:
    counts = Counter(int(v) for v in row if int(v) != BG)
    if not counts:
        return None
    max_count = max(counts.values())
    winners = [color for color, count in counts.items() if count == max_count]
    if len(winners) != 1:
        return None
    return int(winners[0])


def _extract_target_color(grid: np.ndarray) -> tuple[int, tuple[int, int]]:
    marker_coords = np.argwhere(grid == MARKER)
    if marker_coords.shape[0] != 1:
        raise ValueError("expected exactly one teal marker")
    mr, mc = map(int, marker_coords[0])
    neighbors = []
    for nr, nc in _neighbors4(mr, mc):
        if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) not in (BG, MARKER):
            neighbors.append((nr, nc, int(grid[nr, nc])))
    if len(neighbors) != 1:
        raise ValueError("marker must touch exactly one non-black neighbor")
    return int(neighbors[0][2]), (mr, mc)


def _transform_full(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int).copy()
    target_color, (mr, mc) = _extract_target_color(g)
    g[mr, mc] = BG

    out = g.copy()
    for r in range(out.shape[0]):
        if np.any(out[r] == target_color):
            transformed = np.full(out.shape[1], target_color, dtype=int)
            transformed[out[r] == target_color] = BG
            out[r] = transformed
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Use the teal-selected color to invert only its rows, then crop to the surviving foreground.

    Input structure:
    - A black-background grid contains scattered non-black pixels across several rows, plus optional empty rows.
    - Exactly one teal marker pixel is 4-neighbor-adjacent to exactly one non-black pixel; that touched pixel's color is the hidden target color.
    - Non-empty rows may mix the target color with other distractor colors, and some non-empty rows contain no target color at all.

    Rule:
    - Find the teal marker and read the unique adjacent non-black pixel color as target color C, then erase the marker to black.
    - Process rows independently: rows without C stay unchanged; rows with at least one C are inverted into a binary C-versus-black row.
    - In an inverted row, every C pixel becomes black, while every black or other non-C pixel becomes C.
    - After all rows are processed, crop the result to the minimal bounding box of all non-black pixels.

    Key insight:
    - The teal pixel does not select an object; it selects a color, and that color gates which rows undergo the inversion.
    """
    full = _transform_full(grid)
    box = _nonblack_bbox(full)
    if box is None:
        raise ValueError("transformation removed all non-black pixels")
    return _crop_bbox(full, box)


def _split_positive(total: int, parts: int, cap: int) -> list[int]:
    if total < parts or parts <= 0:
        raise ValueError("invalid split request")
    values = [1] * parts
    remaining = total - parts
    while remaining:
        choices = [i for i, value in enumerate(values) if value < cap]
        if not choices:
            raise ValueError("split cap too small")
        idx = random.choice(choices)
        values[idx] += 1
        remaining -= 1
    random.shuffle(values)
    return values


def _scatter_row(width: int, counts: dict[int, int]) -> np.ndarray:
    row = np.full(width, BG, dtype=int)
    total = sum(counts.values())
    interior_cols = list(range(1, width - 1))
    if total > len(interior_cols):
        raise ValueError("row payload is too dense")
    cols = random.sample(interior_cols, total)
    random.shuffle(cols)
    colors = list(counts.items())
    random.shuffle(colors)

    start = 0
    for color, count in colors:
        for col in cols[start : start + count]:
            row[col] = int(color)
        start += count
    return row


def _make_target_dominant_row(width: int, target_color: int, other_colors: list[int]) -> np.ndarray:
    target_count = random.randint(2, 4)
    extra_pixels = random.randint(1, min(3, width - 3 - target_count))
    extra_color_count = 1 if extra_pixels == 1 else random.randint(1, min(2, len(other_colors), extra_pixels))
    extra_counts = _split_positive(extra_pixels, extra_color_count, target_count - 1)
    chosen = random.sample(other_colors, extra_color_count)

    counts = {target_color: target_count}
    for color, count in zip(chosen, extra_counts):
        counts[int(color)] = counts.get(int(color), 0) + int(count)
    return _scatter_row(width, counts)


def _make_target_minor_row(width: int, target_color: int, dominant_color: int, other_colors: list[int]) -> np.ndarray:
    dominant_count = random.randint(2, 4)
    target_count = random.randint(1, dominant_count - 1)
    max_extra = min(2, width - 3 - dominant_count - target_count)
    extra_pixels = random.randint(0, max_extra)

    counts = {int(dominant_color): dominant_count, target_color: target_count}
    if extra_pixels:
        pool = [color for color in other_colors if color != dominant_color]
        if not pool:
            pool = [dominant_color]
        extra_color = random.choice(pool)
        counts[int(extra_color)] = counts.get(int(extra_color), 0) + extra_pixels
    return _scatter_row(width, counts)


def _make_immune_row(width: int, dominant_color: int, other_colors: list[int], force_multicolor: bool) -> np.ndarray:
    dominant_count = random.randint(2, 4)
    max_extra = min(2, width - 3 - dominant_count)
    min_extra = 1 if force_multicolor and max_extra > 0 else 0
    extra_pixels = random.randint(min_extra, max_extra)

    counts = {int(dominant_color): dominant_count}
    if extra_pixels:
        pool = [color for color in other_colors if color != dominant_color]
        if not pool:
            pool = [dominant_color]
        extra_color = random.choice(pool)
        counts[int(extra_color)] = counts.get(int(extra_color), 0) + extra_pixels
    return _scatter_row(width, counts)


def _pick_active_rows(height: int) -> list[int]:
    candidates = list(range(1, height - 1, 2))
    if len(candidates) < 4:
        raise ValueError("height too small for spaced active rows")
    count = random.randint(4, min(6, len(candidates)))
    return sorted(random.sample(candidates, count))


def _place_marker(grid: np.ndarray, target_color: int) -> bool:
    candidates = []
    for r in range(1, grid.shape[0] - 1):
        for c in range(1, grid.shape[1] - 1):
            if int(grid[r, c]) != BG:
                continue
            neighbors = []
            for nr, nc in _neighbors4(r, c):
                if int(grid[nr, nc]) not in (BG, MARKER):
                    neighbors.append((nr, nc, int(grid[nr, nc])))
            if len(neighbors) == 1 and neighbors[0][2] == target_color:
                candidates.append((r, c))
    if not candidates:
        return False
    mr, mc = random.choice(candidates)
    grid[mr, mc] = MARKER
    return True


def _build_attempt() -> np.ndarray:
    height = random.randint(10, 16)
    width = random.randint(10, 16)
    grid = np.full((height, width), BG, dtype=int)

    palette_size = random.randint(3, 4)
    colors = random.sample(PALETTE, palette_size)
    target_color = random.choice(colors)
    other_colors = [color for color in colors if color != target_color]

    active_rows = _pick_active_rows(height)
    target_row_count = random.randint(2, min(4, len(active_rows) - 1))
    target_rows = sorted(random.sample(active_rows, target_row_count))
    immune_rows = [row for row in active_rows if row not in target_rows]

    shared_dominant = random.choice(other_colors)

    target_modes = ["dominant", "minor"]
    while len(target_modes) < len(target_rows):
        target_modes.append(random.choice(["dominant", "minor"]))
    random.shuffle(target_modes)

    immune_force_multicolor_index = random.randrange(len(immune_rows))
    for idx, row_idx in enumerate(target_rows):
        mode = target_modes[idx]
        if mode == "dominant":
            row = _make_target_dominant_row(width, target_color, other_colors)
        else:
            row = _make_target_minor_row(width, target_color, shared_dominant, other_colors)
        grid[row_idx] = row

    for idx, row_idx in enumerate(immune_rows):
        if idx == 0:
            dominant = shared_dominant
        else:
            dominant = random.choice(other_colors)
        row = _make_immune_row(
            width,
            dominant,
            other_colors,
            force_multicolor=(idx == immune_force_multicolor_index),
        )
        grid[row_idx] = row

    if not _place_marker(grid, target_color):
        raise ValueError("failed to place a unique marker")
    return grid


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    if inp.ndim != 2 or out.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False

    try:
        target_color, _marker = _extract_target_color(inp)
    except ValueError:
        return False

    nonempty_rows = []
    rows_with_target = []
    immune_rows = []
    for r in range(inp.shape[0]):
        row = inp[r]
        payload = [int(v) for v in row if int(v) not in (BG, MARKER)]
        if not payload:
            continue
        nonempty_rows.append(r)
        if _dominant_nonblack_color(row) is None:
            return False
        if any(v == target_color for v in payload):
            rows_with_target.append(r)
        else:
            immune_rows.append(r)

    if len(rows_with_target) < 2 or len(immune_rows) < 1:
        return False
    if not any(np.all(inp[r] == BG) for r in range(inp.shape[0])):
        return False
    if not any(_dominant_nonblack_color(inp[r]) != target_color for r in rows_with_target):
        return False
    if not any(
        np.any(inp[r] == target_color) and np.any((inp[r] != BG) & (inp[r] != MARKER) & (inp[r] != target_color))
        for r in rows_with_target
    ):
        return False

    full = _transform_full(inp)
    if not any(np.any((full[r] != BG) & (full[r] != target_color)) for r in immune_rows):
        return False
    if not all(set(map(int, np.unique(full[r]))).issubset({BG, target_color}) for r in rows_with_target):
        return False

    expected = main(inp)
    if not np.array_equal(out, expected):
        return False
    if out.shape == inp.shape:
        return False
    if np.array_equal(out, inp[: out.shape[0], : out.shape[1]]):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            inp = _build_attempt()
            out = main(inp)
        except Exception:
            continue
        if _semantic_checks(inp, out):
            return inp
    raise RuntimeError("failed to generate a valid gated row-inversion sample")
