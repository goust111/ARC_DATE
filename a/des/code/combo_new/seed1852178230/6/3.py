from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
OBJECT_COLOR = 5
STRIP_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_HEIGHT_RANGE = (10, 24)
GRID_WIDTH_RANGE = (10, 24)
MIN_OBJECTS = 2
MAX_OBJECTS = 5
MAX_BBOX_SIDE = 7
GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _majority_color(g: np.ndarray) -> int:
    values, counts = np.unique(g, return_counts=True)
    max_count = int(counts.max())
    winners = values[counts == max_count]
    assert len(winners) == 1, "background must be the unique most common color"
    return int(winners[0])


def _partition_positive(total: int, parts: int) -> list[int]:
    assert total >= parts >= 1
    sizes = [1] * parts
    for _ in range(total - parts):
        sizes[random.randrange(parts)] += 1
    random.shuffle(sizes)
    return sizes


def _band_sequence(length: int, palette: list[int]) -> np.ndarray:
    assert length >= 2
    assert len(palette) >= 2

    n_used = random.randint(2, min(4, len(palette)))
    used = random.sample(palette, n_used)
    max_runs = min(length, n_used + 3)
    n_runs = random.randint(n_used, max_runs)
    run_lengths = _partition_positive(length, n_runs)

    seq: list[int] = []
    remaining = set(used)
    for i in range(n_runs):
        choices = [c for c in used if not seq or c != seq[-1]]
        runs_left = n_runs - i
        if len(remaining) >= runs_left:
            forced = [c for c in choices if c in remaining]
            assert forced
            choices = forced
        elif remaining and random.random() < 0.6:
            preferred = [c for c in choices if c in remaining]
            if preferred:
                choices = preferred
        color = int(random.choice(choices))
        seq.append(color)
        remaining.discard(color)

    out = np.repeat(np.array(seq, dtype=int), run_lengths)
    assert out.shape == (length,)
    assert len(set(map(int, out.tolist()))) >= 2
    return out


def _sample_bands(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for _ in range(100):
        palette = random.sample(list(STRIP_COLORS), random.randint(2, 4))
        col_band = _band_sequence(width, palette)
        row_band = _band_sequence(height - 1, palette)
        same = row_band[:, None] == col_band[None, 1:]
        same_ratio = float(np.mean(same))
        if 0.2 <= same_ratio <= 0.8 and same.any() and (~same).any():
            return col_band, row_band, same
    raise RuntimeError("failed to sample balanced row/column color bands")


def _grow_connected_mask(height: int, width: int, target_area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    seed = (random.randrange(height), random.randrange(width))
    mask[seed] = True
    frontier = set(_neighbors4(seed[0], seed[1], height, width))

    while int(mask.sum()) < target_area:
        assert frontier, "frontier should not become empty while space remains"
        r, c = random.choice(tuple(frontier))
        frontier.remove((r, c))
        mask[r, c] = True
        for nr, nc in _neighbors4(r, c, height, width):
            if not mask[nr, nc]:
                frontier.add((nr, nc))

    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _random_object_mask(max_h: int, max_w: int) -> np.ndarray:
    bbox_h = random.randint(2, min(MAX_BBOX_SIDE, max_h))
    bbox_w = random.randint(2, min(MAX_BBOX_SIDE, max_w))
    area = bbox_h * bbox_w
    min_area = max(3, int(np.ceil(area * 0.35)))
    max_area = max(min_area, int(np.floor(area * 0.75)))
    target_area = random.randint(min_area, max_area)
    return _grow_connected_mask(bbox_h, bbox_w, target_area)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            src_r1 = h - max(0, dr)
            src_c0 = max(0, -dc)
            src_c1 = w - max(0, dc)
            dst_r0 = max(0, dr)
            dst_r1 = h - max(0, -dr)
            dst_c0 = max(0, dc)
            dst_c1 = w - max(0, -dc)
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _find_positions(
    occupied: np.ndarray, mask: np.ndarray, required_cells: np.ndarray | None
) -> list[tuple[int, int]]:
    h, w = occupied.shape
    mh, mw = mask.shape
    blocked = _dilate8(occupied)
    positions: list[tuple[int, int]] = []
    for r0 in range(h - mh + 1):
        for c0 in range(w - mw + 1):
            window = blocked[r0 : r0 + mh, c0 : c0 + mw]
            if np.any(window & mask):
                continue
            if required_cells is not None:
                required_window = required_cells[r0 : r0 + mh, c0 : c0 + mw]
                if not np.any(required_window & mask):
                    continue
            positions.append((r0, c0))
    return positions


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each interior monochrome object pixel using the edge color bands as row/column lookup rules.

    Input structure:
    - The background is the dominant black color.
    - The top row is a full-width non-background color band that encodes the column color for each column.
    - The left column, excluding the top-left cell, is a non-background color band that encodes the row color for each interior row.
    - The interior region contains one or more 4-connected monochrome foreground objects on the background.

    Rule:
    - Clear the top row and the left column to the background color in the output.
    - For each non-background interior pixel, read the column-band color from the same column and the row-band color from the same row.
    - If the two colors match, paint the pixel with that shared color; otherwise paint it with the column-band color.
    - Leave all background interior pixels unchanged.

    Key insight:
    The edge strips are coordinate-indexed color lookup tables, and when the row and column instructions disagree the column strip has priority.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"
    h, w = map(int, g.shape)
    assert h >= 3 and w >= 3, "grid must contain edge bands plus an interior"

    bg = _majority_color(g)
    col_band = g[0]
    row_band = g[1:, 0]
    assert bool(np.all(col_band != bg)), "top edge band must avoid the background color"
    assert bool(np.all(row_band != bg)), "left edge band must avoid the background color"

    inner = g[1:, 1:]
    object_mask = inner != bg
    object_colors = np.unique(inner[object_mask]) if object_mask.any() else np.array([], dtype=int)
    assert len(object_colors) == 1, "interior foreground objects must share one color"

    row_lookup = row_band[:, None]
    col_lookup = col_band[None, 1:]
    recolor = np.where(col_lookup == row_lookup, row_lookup, col_lookup)

    out = np.full_like(g, bg)
    out_inner = out[1:, 1:]
    out_inner[object_mask] = recolor[object_mask]
    return out


def _build_expected(col_band: np.ndarray, row_band: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
    h, w = object_mask.shape
    out = np.full((h + 1, w + 1), BACKGROUND, dtype=int)
    row_lookup = row_band[:, None]
    col_lookup = col_band[None, 1:]
    recolor = np.where(col_lookup == row_lookup, row_lookup, col_lookup)
    out_inner = out[1:, 1:]
    out_inner[object_mask] = recolor[object_mask]
    return out


def _build_example() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    inner_h, inner_w = height - 1, width - 1

    col_band, row_band, same = _sample_bands(height, width)
    diff = ~same

    max_objects = min(MAX_OBJECTS, max(MIN_OBJECTS, inner_h * inner_w // 22))
    object_count = random.randint(MIN_OBJECTS, max_objects)

    object_mask = np.zeros((inner_h, inner_w), dtype=bool)
    required_kinds = [same, diff]
    for obj_index in range(object_count):
        required = required_kinds[obj_index] if obj_index < len(required_kinds) else None
        placed = False
        for _ in range(80):
            mask = _random_object_mask(inner_h, inner_w)
            positions = _find_positions(object_mask, mask, required)
            if not positions:
                continue
            r0, c0 = random.choice(positions)
            mh, mw = mask.shape
            object_mask[r0 : r0 + mh, c0 : c0 + mw] |= mask
            placed = True
            break
        if not placed:
            return None

    eq_count = int(np.sum(object_mask & same))
    diff_count = int(np.sum(object_mask & diff))
    total_fg = int(object_mask.sum())
    if eq_count == 0 or diff_count == 0 or total_fg < 8:
        return None
    eq_ratio = eq_count / total_fg
    if not 0.25 <= eq_ratio <= 0.75:
        return None

    comps = _components(object_mask)
    if len(comps) != object_count:
        return None

    inp = np.full((height, width), BACKGROUND, dtype=int)
    inp[0, :] = col_band
    inp[1:, 0] = row_band
    inp[1:, 1:][object_mask] = OBJECT_COLOR

    expected = _build_expected(col_band, row_band, object_mask)

    values, counts = np.unique(inp, return_counts=True)
    bg_count = int(counts[np.where(values == BACKGROUND)[0][0]])
    if any(int(count) >= bg_count for value, count in zip(values.tolist(), counts.tolist(), strict=False) if int(value) != BACKGROUND):
        return None

    inner_values = set(int(v) for v in np.unique(inp[1:, 1:]) if int(v) != BACKGROUND)
    if inner_values != {OBJECT_COLOR}:
        return None

    if len(set(map(int, col_band.tolist()))) < 2 or len(set(map(int, row_band.tolist()))) < 2:
        return None

    if not np.array_equal(main(inp), expected):
        return None

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        pair = _build_example()
        if pair is not None:
            return pair[0]
    raise RuntimeError("failed to generate a valid edge-band recoloring input")
