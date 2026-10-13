from __future__ import annotations

from collections import Counter
import random

import numpy as np


YELLOW = 4
SEPARATOR = 5
MAROON = 9

BG_CHOICES = [0, 1]
BORDER_CHOICES = [2, 3, 6, 7, 8]
GEN_RETRIES = 400


def _shape(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=bool)


STAMP_CATALOG: dict[int, list[np.ndarray]] = {
    1: [
        _shape([[1]]),
    ],
    2: [
        _shape([[1, 1]]),
        _shape([[1], [1]]),
    ],
    3: [
        _shape([[1, 1, 1]]),
        _shape([[1], [1], [1]]),
        _shape([[1, 0], [1, 1]]),
        _shape([[0, 1], [1, 1]]),
        _shape([[1, 1], [1, 0]]),
        _shape([[1, 1], [0, 1]]),
    ],
    4: [
        _shape([[1, 1, 1, 1]]),
        _shape([[1], [1], [1], [1]]),
        _shape([[1, 1], [1, 1]]),
        _shape([[1, 1, 1], [0, 1, 0]]),
        _shape([[0, 1, 0], [1, 1, 1]]),
        _shape([[1, 0], [1, 1], [1, 0]]),
        _shape([[0, 1], [1, 1], [0, 1]]),
        _shape([[1, 0], [1, 0], [1, 1]]),
        _shape([[0, 1], [0, 1], [1, 1]]),
        _shape([[1, 1], [0, 1], [0, 1]]),
        _shape([[1, 1], [1, 0], [1, 0]]),
        _shape([[1, 1, 0], [0, 1, 1]]),
        _shape([[0, 1, 1], [1, 1, 0]]),
        _shape([[1, 0], [1, 1], [0, 1]]),
        _shape([[0, 1], [1, 1], [1, 0]]),
    ],
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _extract_library_patterns(grid: np.ndarray) -> dict[int, np.ndarray]:
    lib_mask = (grid == SEPARATOR) | (grid == MAROON)
    coords = np.argwhere(lib_mask)
    assert len(coords) > 0, "expected a grey/maroon stamp library"

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    box = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]

    separator_rows = [r for r in range(box.shape[0]) if bool(np.all(box[r, :] == SEPARATOR))]
    assert len(separator_rows) >= 5, "stamp library must be split by separator rows"

    patterns: dict[int, np.ndarray] = {}
    for top, bottom in zip(separator_rows, separator_rows[1:]):
        if int(bottom - top) <= 1:
            continue
        cell = box[int(top) + 1 : int(bottom), :]
        stamp_pts = np.argwhere(cell == MAROON)
        if len(stamp_pts) == 0:
            continue
        sr0, sc0 = stamp_pts.min(axis=0)
        sr1, sc1 = stamp_pts.max(axis=0)
        stamp = cell[int(sr0) : int(sr1) + 1, int(sc0) : int(sc1) + 1] == MAROON
        area = int(stamp.sum())
        patterns[area] = stamp

    assert set(patterns) == {1, 2, 3, 4}, "library must contain one stamp of each area 1..4"
    return patterns


def _is_frame(subgrid: np.ndarray, color: int) -> bool:
    h, w = subgrid.shape
    if h < 3 or w < 3:
        return False
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not bool(np.all(subgrid[border] == color)):
        return False
    if bool(np.any(subgrid[~border] == color)):
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace yellow counts inside framed regions with same-area library stamps in the frame color.

    Input structure:
    - bg is the most common color.
    - Several 1-pixel-thick colored rectangular frames enclose bg interiors containing 1 to 4 yellow marker pixels.
    - A stamp library sits on one side of the grid; full-width grey separator rows split it into cells, and each cell contains a maroon pattern.

    Rule:
    - Count the yellow markers inside each framed rectangle.
    - Read the maroon library pattern whose pixel area matches that count.
    - Clear the yellow markers, then place the chosen pattern at the centered position of the frame interior, recolored to the frame's border color.
    - If exact centering is impossible, use the upper-left centered placement.

    Key insight:
    - Marker count chooses the shape, while the enclosing frame color independently chooses the stamp color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    patterns = _extract_library_patterns(g)
    out = g.copy()

    role_colors = {bg, YELLOW, SEPARATOR, MAROON}
    for color in sorted(set(map(int, g.reshape(-1))) - role_colors):
        coords = np.argwhere(g == color)
        if len(coords) == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        sub = g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        if not _is_frame(sub, int(color)):
            continue

        inner = sub[1:-1, 1:-1]
        if not bool(np.all((inner == bg) | (inner == YELLOW))):
            continue

        count = int(np.sum(inner == YELLOW))
        if count not in patterns:
            continue

        stamp = patterns[count]
        sh, sw = map(int, stamp.shape)
        ih, iw = map(int, inner.shape)
        assert sh <= ih and sw <= iw, "stamp must fit inside the frame interior"

        placed_r0 = int(r0) + 1 + (ih - sh) // 2
        placed_c0 = int(c0) + 1 + (iw - sw) // 2

        region = out[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
        region[region == YELLOW] = bg
        patch = out[placed_r0 : placed_r0 + sh, placed_c0 : placed_c0 + sw]
        patch[stamp] = int(color)

    return out


def _choose_stamps(rng: random.Random) -> dict[int, np.ndarray]:
    return {area: rng.choice(STAMP_CATALOG[area]).copy() for area in range(1, 5)}


def _region_counts(rng: random.Random) -> list[int]:
    n_regions = rng.choices([2, 3, 4], weights=[2, 3, 5], k=1)[0]
    counts = rng.sample([1, 2, 3, 4], k=n_regions)
    if n_regions >= 3 and rng.random() < 0.3:
        duplicate_idx = rng.randrange(n_regions)
        counts[duplicate_idx] = rng.choice([1, 2, 3, 4])
    return counts


def _region_spec(rng: random.Random, border_color: int, count: int, stamp: np.ndarray) -> dict[str, object]:
    sh, sw = map(int, stamp.shape)
    inner_h_min = max(3, sh + 1)
    inner_w_min = max(3, sw + 1)
    inner_h_max = min(7, sh + 4)
    inner_w_max = min(7, sw + 4)
    inner_h = rng.randint(inner_h_min, inner_h_max)
    inner_w = rng.randint(inner_w_min, inner_w_max)
    return {
        "color": border_color,
        "count": count,
        "stamp": stamp,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "outer_h": inner_h + 2,
        "outer_w": inner_w + 2,
    }


def _row_lengths_for_count(rng: random.Random, n_regions: int) -> list[int]:
    if n_regions == 2:
        return rng.choice([[2], [1, 1]])
    if n_regions == 3:
        return rng.choice([[2, 1], [1, 2]])
    return [2, 2]


def _layout_regions(rng: random.Random, specs: list[dict[str, object]]) -> tuple[list[tuple[dict[str, object], int, int]], int, int] | None:
    order = specs[:]
    rng.shuffle(order)
    row_lengths = _row_lengths_for_count(rng, len(order))

    rows: list[list[dict[str, object]]] = []
    idx = 0
    for length in row_lengths:
        rows.append(order[idx : idx + length])
        idx += length

    row_meta = []
    field_w = 0
    field_h = 0
    row_gaps = [rng.randint(1, 3) for _ in range(max(0, len(rows) - 1))]
    for row in rows:
        horiz_gaps = [rng.randint(1, 3) for _ in range(max(0, len(row) - 1))]
        row_h = max(int(spec["outer_h"]) for spec in row)
        row_w = sum(int(spec["outer_w"]) for spec in row) + sum(horiz_gaps)
        row_meta.append((row, row_h, row_w, horiz_gaps))
        field_w = max(field_w, row_w)
        field_h += row_h
    field_h += sum(row_gaps)

    placements: list[tuple[dict[str, object], int, int]] = []
    cursor_y = 0
    for row_index, (row, row_h, row_w, horiz_gaps) in enumerate(row_meta):
        cursor_x = rng.randint(0, field_w - row_w)
        for item_index, spec in enumerate(row):
            max_dy = row_h - int(spec["outer_h"])
            dy = rng.randint(0, max_dy)
            placements.append((spec, cursor_y + dy, cursor_x))
            cursor_x += int(spec["outer_w"])
            if item_index < len(horiz_gaps):
                cursor_x += horiz_gaps[item_index]
        if row_index < len(row_gaps):
            cursor_y += row_h + row_gaps[row_index]

    return placements, field_h, field_w


def _build_library(rng: random.Random, bg: int, stamps: dict[int, np.ndarray]) -> np.ndarray:
    max_w = max(int(stamp.shape[1]) for stamp in stamps.values())
    lib_w = max_w + rng.choice([2, 3])

    cell_heights = {area: int(stamps[area].shape[0]) + rng.choice([2, 2, 3]) for area in range(1, 5)}
    lib_h = 5 + sum(cell_heights.values())
    lib = np.full((lib_h, lib_w), bg, dtype=int)

    row = 0
    lib[row, :] = SEPARATOR
    row += 1
    for area in range(1, 5):
        cell_h = cell_heights[area]
        stamp = stamps[area]
        sh, sw = map(int, stamp.shape)
        top = row + (cell_h - sh) // 2
        left = (lib_w - sw) // 2
        lib[top : top + sh, left : left + sw][stamp] = MAROON
        row += cell_h
        lib[row, :] = SEPARATOR
        row += 1

    return lib


def _centered_stamp_cells(inner_h: int, inner_w: int, stamp: np.ndarray) -> set[tuple[int, int]]:
    sh, sw = map(int, stamp.shape)
    top = (inner_h - sh) // 2
    left = (inner_w - sw) // 2
    pts = np.argwhere(stamp)
    return {(top + int(r), left + int(c)) for r, c in pts}


def _sample_markers(
    rng: random.Random,
    inner_h: int,
    inner_w: int,
    count: int,
    centered_stamp: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    for _ in range(80):
        chosen = set(rng.sample(cells, count))
        if count > 1 and chosen == centered_stamp:
            continue
        rows = {r for r, _ in chosen}
        cols = {c for _, c in chosen}
        if count >= 3 and (len(rows) == 1 or len(cols) == 1):
            continue
        return sorted(chosen)
    return sorted(rng.sample(cells, count))


def _draw_region(
    grid: np.ndarray,
    top: int,
    left: int,
    spec: dict[str, object],
    rng: random.Random,
) -> None:
    color = int(spec["color"])
    count = int(spec["count"])
    stamp = np.asarray(spec["stamp"], dtype=bool)
    inner_h = int(spec["inner_h"])
    inner_w = int(spec["inner_w"])
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])

    grid[top : top + outer_h, left] = color
    grid[top : top + outer_h, left + outer_w - 1] = color
    grid[top, left : left + outer_w] = color
    grid[top + outer_h - 1, left : left + outer_w] = color

    centered_stamp = _centered_stamp_cells(inner_h, inner_w, stamp)
    for r, c in _sample_markers(rng, inner_h, inner_w, count, centered_stamp):
        grid[top + 1 + r, left + 1 + c] = YELLOW


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.choice(BG_CHOICES)
    stamps = _choose_stamps(rng)
    library = _build_library(rng, bg, stamps)
    lib_h, lib_w = map(int, library.shape)

    counts = _region_counts(rng)
    border_colors = rng.sample(BORDER_CHOICES, k=len(counts))
    specs = [_region_spec(rng, color, count, stamps[count]) for color, count in zip(border_colors, counts)]
    layout = _layout_regions(rng, specs)
    if layout is None:
        return None
    placements, field_h, field_w = layout

    gap = rng.randint(1, 3)
    base_h = max(field_h, lib_h)
    base_w = field_w + gap + lib_w
    if base_h > 30 or base_w > 30:
        return None

    extra_h = rng.randint(0, min(5, 30 - base_h))
    extra_w = rng.randint(0, min(4, 30 - base_w))
    height = base_h + extra_h
    width = base_w + extra_w

    grid = np.full((height, width), bg, dtype=int)
    side = rng.choice(["left", "right"])

    library_top = rng.randint(0, height - lib_h)
    field_top = rng.randint(0, height - field_h)

    if side == "left":
        library_left = 0
        field_left = lib_w + gap + rng.randint(0, extra_w)
    else:
        field_left = rng.randint(0, extra_w)
        library_left = width - lib_w

    grid[library_top : library_top + lib_h, library_left : library_left + lib_w] = library
    for spec, local_top, local_left in placements:
        _draw_region(grid, field_top + local_top, field_left + local_left, spec, rng)

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
