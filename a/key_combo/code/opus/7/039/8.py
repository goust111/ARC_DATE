from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6

REFERENCE_COLORS = (BLUE, RED, GREEN, YELLOW, PINK)
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(coords)
    return parts


def _reference_rows(grid: np.ndarray, bg: int) -> list[int]:
    rows: list[int] = []
    for r in range(grid.shape[0]):
        if np.any(grid[r] != bg):
            rows.append(r)
        elif rows:
            break
        else:
            break
    assert rows and rows[0] == 0 and len(rows) <= 2
    return rows


def _extract_reference_sequence(grid: np.ndarray, rows: list[int], bg: int) -> list[int]:
    sequence: list[int] = []
    last = None
    for c in range(grid.shape[1]):
        colors = {int(grid[r, c]) for r in rows if int(grid[r, c]) != bg}
        assert len(colors) <= 1
        color = next(iter(colors)) if colors else None
        if color is None:
            last = None
            continue
        if color != last:
            sequence.append(color)
            last = color
    assert 4 <= len(sequence) <= 5
    assert len(set(sequence)) == len(sequence)
    return sequence


def _boundary_of_component(component: list[tuple[int, int]], grid: np.ndarray, bg: int) -> set[tuple[int, int]]:
    h, w = grid.shape
    boundary: set[tuple[int, int]] = set()
    for r, c in component:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != bg:
                boundary.add((nr, nc))
    return boundary


def _object_from_seed(start: tuple[int, int], grid: np.ndarray, bg: int, min_row: int) -> set[tuple[int, int]]:
    h, w = grid.shape
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (
                0 <= nr < h
                and 0 <= nc < w
                and nr > min_row
                and int(grid[nr, nc]) != bg
                and (nr, nc) not in seen
            ):
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each closed region with the reference color that immediately follows its matched top-boundary prefix.

    Input structure:
    - A black background grid has a 1-2 row reference color band at the top.
    - Below it are several one-pixel-thick closed boundary objects in grey, with some top-edge boundary pixels recolored.
    - Each closed region encloses a background interior, and its recolored top-edge pixels form a left-to-right color sequence.

    Rule:
    - Read the top reference band as an ordered color sequence.
    - For each enclosed background region, extract the non-grey colors on the highest boundary row from left to right.
    - If that sequence is a proper prefix of the reference sequence, flood-fill the region interior with the next reference color; otherwise leave it unchanged.

    Key insight:
    - The top color band is a lookup table, and each region's top boundary encodes how far into that table to advance before choosing the fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    ref_rows = _reference_rows(g, bg)
    reference = _extract_reference_sequence(g, ref_rows, bg)
    cutoff_row = ref_rows[-1]

    out = g.copy()
    for component in _components(g == bg):
        rows = [r for r, _c in component]
        cols = [c for _r, c in component]
        if min(rows) <= cutoff_row:
            continue
        if min(rows) == 0 or max(rows) == g.shape[0] - 1 or min(cols) == 0 or max(cols) == g.shape[1] - 1:
            continue

        boundary = _boundary_of_component(component, g, bg)
        if not boundary:
            continue
        obj = _object_from_seed(next(iter(boundary)), g, bg, cutoff_row)
        top_r = min(r for r, _c in obj)
        top_cols = sorted(c for r, c in obj if r == top_r)
        prefix = [int(g[top_r, c]) for c in top_cols if int(g[top_r, c]) not in (bg, GREY)]
        k = len(prefix)
        if 0 < k < len(reference) and prefix == reference[:k]:
            fill_color = reference[k]
            for r, c in component:
                out[r, c] = fill_color

    return out


def _shape_masks() -> tuple[np.ndarray, np.ndarray]:
    base_h, base_w = random.choice(
        [
            (4, 4),
            (4, 5),
            (4, 6),
            (5, 4),
            (5, 5),
            (5, 6),
            (6, 4),
            (6, 5),
        ]
    )
    cells = {(r, c) for r in range(base_h) for c in range(base_w)}
    sides = ["left", "right", "bottom"]
    random.shuffle(sides)
    for side in sides[: random.randint(0, 2)]:
        length = random.randint(1, 2)
        if side == "left":
            r0 = random.randint(1, base_h - 2)
            for d in range(1, length + 1):
                cells.add((r0, -d))
        elif side == "right":
            r0 = random.randint(1, base_h - 2)
            for d in range(length):
                cells.add((r0, base_w + d))
        else:
            c0 = random.randint(1, base_w - 2)
            for d in range(length):
                cells.add((base_h + d, c0))

    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    normalized = {(r - min_r, c - min_c) for r, c in cells}
    h = max(r for r, _c in normalized) + 1
    w = max(c for _r, c in normalized) + 1

    filled = np.zeros((h, w), dtype=bool)
    for r, c in normalized:
        filled[r, c] = True

    boundary = np.zeros_like(filled)
    for r, c in np.argwhere(filled):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if nr < 0 or nr >= h or nc < 0 or nc >= w or not filled[nr, nc]:
                boundary[rr, cc] = True
                break
    interior = filled & ~boundary
    if not (4 <= int(interior.sum()) <= 12):
        raise ValueError("interior area out of range")
    if len(_components(interior)) != 1:
        raise ValueError("interior must stay connected")
    return boundary, interior


def _sample_prefix_positions(top_cols: list[int], k: int) -> list[int]:
    indices = sorted(random.sample(range(len(top_cols)), k))
    return [top_cols[i] for i in indices]


def _separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _paint_reference_band(grid: np.ndarray, colors: list[int], rows: list[int], start_col: int, widths: list[int]) -> None:
    c = start_col
    for color, width in zip(colors, widths):
        grid[np.ix_(rows, np.arange(c, c + width))] = color
        c += width


def _attempt_generate() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(16, 22)
    width = random.randint(17, 25)
    band_height = random.choice([1, 2])
    gap_after_band = random.choice([1, 2])

    sequence_len = random.choice([4, 5])
    reference = random.sample(list(REFERENCE_COLORS), sequence_len)
    segment_widths = [random.randint(2, 3) for _ in range(sequence_len)]
    band_width = sum(segment_widths)
    if band_width + 2 > width:
        return None

    region_count = random.randint(3, 5)
    prefix_lengths = list(range(1, sequence_len))
    random.shuffle(prefix_lengths)
    while len(prefix_lengths) < region_count:
        prefix_lengths.append(random.randint(1, sequence_len - 1))
    random.shuffle(prefix_lengths)

    inp = np.full((height, width), BLACK, dtype=int)
    out = inp.copy()

    band_rows = list(range(band_height))
    band_start = random.randint(1, width - band_width - 1)
    _paint_reference_band(inp, reference, band_rows, band_start, segment_widths)
    _paint_reference_band(out, reference, band_rows, band_start, segment_widths)

    occupied_boxes: list[tuple[int, int, int, int]] = []
    min_row = band_height + gap_after_band

    for prefix_len in prefix_lengths:
        placed = False
        for _ in range(120):
            try:
                boundary, interior = _shape_masks()
            except ValueError:
                continue
            h, w = boundary.shape
            if min_row + h > height - 1 or w > width - 2:
                continue
            r0 = random.randint(min_row, height - h - 1)
            c0 = random.randint(1, width - w - 1)
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(not _separated(box, other, gap=2) for other in occupied_boxes):
                continue

            top_cols = sorted(int(c) for c in np.flatnonzero(boundary[0]))
            if prefix_len > len(top_cols):
                continue
            color_cols = set(_sample_prefix_positions(top_cols, prefix_len))

            for rr, cc in np.argwhere(boundary):
                color = GREY
                if int(rr) == 0 and int(cc) in color_cols:
                    rank = sum(col <= int(cc) for col in sorted(color_cols)) - 1
                    color = reference[rank]
                inp[r0 + int(rr), c0 + int(cc)] = color
                out[r0 + int(rr), c0 + int(cc)] = color

            fill_color = reference[prefix_len]
            for rr, cc in np.argwhere(interior):
                out[r0 + int(rr), c0 + int(cc)] = fill_color

            occupied_boxes.append(box)
            placed = True
            break
        if not placed:
            return None

    if len({p for p in prefix_lengths}) < 2:
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        pair = _attempt_generate()
        if pair is None:
            continue
        inp, expected = pair
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
