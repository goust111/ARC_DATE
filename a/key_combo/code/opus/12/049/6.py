from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
GREEN = 3

HEIGHT_CHOICES = (11, 13, 15)
WIDTH_CHOICES = (9, 11, 13, 15)
PAIR_COUNT_RANGE = (4, 6)
GENERATION_ATTEMPTS = 600


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_pairs(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    mid = w // 2
    pairs: list[tuple[int, int, int, int]] = []
    for r in range(h):
        cols = np.flatnonzero(grid[r] == GREEN)
        if len(cols) == 0:
            continue
        assert len(cols) == 2, "each marker row must contain exactly two green pixels"
        c0, c1 = map(int, cols)
        assert c0 < c1 and c0 + c1 == 2 * mid, "green pairs must mirror across the vertical middle column"
        assert grid[r, mid] == BG, "input markers must stay off the middle column"
        assert not np.any(grid[r] == RED), "red decorations cannot share a marker row"
        pairs.append((r, c0, c1, c1 - c0 - 1))
    assert 4 <= len(pairs) <= 6, "expected 4-6 marker rows"
    return pairs


def _axis_info(grid: np.ndarray) -> tuple[int, int, int]:
    pairs = _extract_pairs(grid)
    gaps = [gap for _r, _c0, _c1, gap in pairs]
    min_gap = min(gaps)
    assert gaps.count(min_gap) == 1, "minimum green-pair gap must be unique"
    axis_row, c0, c1, _gap = next(item for item in pairs if item[3] == min_gap)
    return axis_row, c0, c1


def _filled_base(grid: np.ndarray) -> tuple[np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))).issubset({BG, RED, GREEN})
    h, w = g.shape
    assert h <= 30 and w <= 30 and w % 2 == 1
    assert np.all(g[:, w // 2] == BG), "input must not place colored pixels on the vertical middle column"

    axis_row, c0, c1 = _axis_info(g)
    base = g.copy()
    base[axis_row, c0 : c1 + 1] = GREEN
    return base, axis_row


def _mirror_full(base: np.ndarray, axis_row: int) -> np.ndarray:
    h, w = base.shape
    out = base.copy()
    src = np.argwhere(base != BG)
    for r, c in src:
        r = int(r)
        c = int(c)
        if r == axis_row:
            continue
        rr = 2 * axis_row - r
        if 0 <= rr < h and out[rr, c] == BG:
            out[rr, c] = int(base[r, c])
    return out


def _edge_padded_crop(grid: np.ndarray) -> np.ndarray:
    box = _bbox(grid != BG)
    assert box is not None, "transformed pattern must be non-empty"
    r0, c0, r1, c1 = box
    core = grid[r0 : r1 + 1, c0 : c1 + 1]
    padded = np.pad(core, 1, mode="edge")
    rs = 0 if r0 > 0 else 1
    re = padded.shape[0] if r1 < grid.shape[0] - 1 else padded.shape[0] - 1
    cs = 0 if c0 > 0 else 1
    ce = padded.shape[1] if c1 < grid.shape[1] - 1 else padded.shape[1] - 1
    return padded[rs:re, cs:ce]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the closest mirrored green pair into an axis, reflect all markings across it, then crop with clipped edge padding.

    Input structure:
    - A black background contains 4-6 rows of green marker pairs.
    - In every marker row, exactly two green pixels sit symmetrically around the grid's vertical middle column.
    - Optional red decoration pixels may appear one row above or below some green markers, but no colored input pixel lies on the middle column.

    Rule:
    - Find the unique green pair with the smallest horizontal gap and fill the inclusive segment between its endpoints, making a horizontal green axis.
    - Treat that filled segment as a mirror axis: copy every non-black pixel above it to the reflected position below, and every non-black pixel below it to the reflected position above; never overwrite an already non-black target cell.
    - Take the minimal bounding box of the completed mirrored pattern and apply a one-cell edge padding to that box, clipping away any part of the padding that would fall outside the original grid.

    Key insight:
    - The row chosen by the minimum-gap connection is not just edited; the new green segment becomes the geometric mirror axis that determines the final cropped and padded symmetric pattern.
    """
    base, axis_row = _filled_base(np.asarray(grid, dtype=int))
    mirrored = _mirror_full(base, axis_row)
    return _edge_padded_crop(mirrored)


def _analyze_input(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    pairs = _extract_pairs(g)
    h, w = g.shape
    mid = w // 2
    markers = {r: (c0, c1) for r, c0, c1, _gap in pairs}
    axis_row, axis_c0, axis_c1 = _axis_info(g)

    reds = [tuple(map(int, rc)) for rc in np.argwhere(g == RED)]
    for r, c in reds:
        assert c != mid
        assert any(abs(r - mr) == 1 and c in markers[mr] for mr in markers), "red pixels must sit one row from a matching marker"

    pair_rows = sorted(markers)
    assert any(r < axis_row for r in pair_rows) and any(r > axis_row for r in pair_rows), "axis row must have markers on both sides"

    return {
        "shape": g.shape,
        "mid": mid,
        "pairs": pairs,
        "markers": markers,
        "axis_row": axis_row,
        "axis_span": (axis_c0, axis_c1),
        "reds": reds,
    }


def _semantic_check(grid: np.ndarray) -> None:
    info = _analyze_input(grid)
    g = np.asarray(grid, dtype=int)
    base, axis_row = _filled_base(g)
    mirrored = _mirror_full(base, axis_row)
    solved = main(g)

    for r, c in np.argwhere(base != BG):
        r = int(r)
        c = int(c)
        if r == axis_row:
            continue
        rr = 2 * axis_row - r
        if not (0 <= rr < base.shape[0]):
            continue
        if base[rr, c] == BG:
            assert mirrored[rr, c] == base[r, c], "empty mirror targets must receive the reflected color"
        else:
            assert mirrored[rr, c] == base[rr, c], "occupied mirror targets must keep their original color"

    box = _bbox(mirrored != BG)
    assert box is not None
    r0, c0, r1, c1 = box
    padded = np.pad(mirrored[r0 : r1 + 1, c0 : c1 + 1], 1, mode="edge")
    rs = 0 if r0 > 0 else 1
    re = padded.shape[0] if r1 < mirrored.shape[0] - 1 else padded.shape[0] - 1
    cs = 0 if c0 > 0 else 1
    ce = padded.shape[1] if c1 < mirrored.shape[1] - 1 else padded.shape[1] - 1
    assert np.array_equal(solved, padded[rs:re, cs:ce]), "output must equal the clipped one-cell edge pad of the mirrored bbox"

    input_nonblack = set(map(tuple, np.argwhere(g != BG)))
    output_nonblack = set(map(tuple, np.argwhere(mirrored != BG)))
    assert input_nonblack <= output_nonblack, "the mirror stage must preserve all original colored pixels"
    assert info["axis_row"] == axis_row


def _select_rows(height: int, pair_count: int, vertical_clip_side: str | None) -> tuple[list[int], int]:
    for _ in range(200):
        parity = 1 if vertical_clip_side is not None else random.randint(0, 1)
        candidates = [r for r in range(height) if r % 2 == parity]
        axis_choices = []
        for axis_row in candidates[1:-1]:
            radius = min(axis_row, height - 1 - axis_row)
            allowed = [r for r in candidates if abs(r - axis_row) <= radius]
            if len(allowed) < pair_count:
                continue
            above = [r for r in allowed if r < axis_row]
            below = [r for r in allowed if r > axis_row]
            if not above or not below:
                continue
            if vertical_clip_side == "top" and 1 not in above:
                continue
            if vertical_clip_side == "bottom" and (height - 2) not in below:
                continue
            axis_choices.append((axis_row, allowed))

        if not axis_choices:
            continue

        axis_row, allowed = random.choice(axis_choices)
        above = [r for r in allowed if r < axis_row]
        below = [r for r in allowed if r > axis_row]

        chosen = {axis_row}
        if vertical_clip_side == "top":
            chosen.add(1)
        elif vertical_clip_side == "bottom":
            chosen.add(height - 2)

        if not any(r < axis_row for r in chosen):
            chosen.add(random.choice(above))
        if not any(r > axis_row for r in chosen):
            chosen.add(random.choice(below))

        remaining = [r for r in allowed if r not in chosen]
        need = pair_count - len(chosen)
        if need < 0 or need > len(remaining):
            continue
        if need:
            chosen.update(random.sample(remaining, need))
        rows = sorted(chosen)
        if len(rows) == pair_count:
            return rows, axis_row
    raise RuntimeError("failed to choose valid marker rows")


def _assign_spans(width: int, rows: list[int], axis_row: int, horizontal_clip: bool) -> dict[int, int]:
    mid = width // 2
    axis_k = 1 if random.random() < 0.4 else random.randint(1, mid - 1)
    others = [r for r in rows if r != axis_row]
    row_to_k = {axis_row: axis_k}

    available = list(range(axis_k + 1, mid + 1))
    assert available, "non-axis rows need a wider pair than the axis row"

    if horizontal_clip:
        forced_row = random.choice(others)
        row_to_k[forced_row] = mid

    distinct_pool = available[:]
    random.shuffle(distinct_pool)
    distinct_count = random.randint(1, min(3, len(distinct_pool)))
    palette = distinct_pool[:distinct_count]
    if horizontal_clip and mid not in palette:
        palette.append(mid)

    for r in others:
        if r in row_to_k:
            continue
        row_to_k[r] = random.choice(palette)
    return row_to_k


def _place_reds(
    grid: np.ndarray,
    rows: list[int],
    row_to_cols: dict[int, tuple[int, int]],
    vertical_clip_side: str | None,
) -> None:
    h, _w = grid.shape
    used: set[tuple[int, int]] = set()

    if vertical_clip_side == "top":
        c = random.choice(row_to_cols[1])
        grid[0, c] = RED
        used.add((0, c))
    elif vertical_clip_side == "bottom":
        anchor = h - 2
        c = random.choice(row_to_cols[anchor])
        grid[h - 1, c] = RED
        used.add((h - 1, c))

    candidates: list[tuple[int, int]] = []
    for r in rows:
        c0, c1 = row_to_cols[r]
        for rr in (r - 1, r + 1):
            if 0 <= rr < h:
                for c in (c0, c1):
                    if (rr, c) not in used:
                        candidates.append((rr, c))

    random.shuffle(candidates)
    want_extra = random.randint(0, min(4 - len(used), len(candidates)))
    if not used and random.random() < 0.8 and candidates:
        want_extra = max(1, want_extra)

    for r, c in candidates:
        if want_extra == 0:
            break
        if grid[r, c] != BG:
            continue
        grid[r, c] = RED
        used.add((r, c))
        want_extra -= 1


def _generate_attempt() -> np.ndarray | None:
    height = random.choice(HEIGHT_CHOICES)
    width = random.choice(WIDTH_CHOICES)
    pair_count = random.randint(*PAIR_COUNT_RANGE)

    vertical_clip_side = None
    if random.random() < 0.4:
        vertical_clip_side = random.choice(("top", "bottom"))
    horizontal_clip = random.random() < 0.4

    try:
        rows, axis_row = _select_rows(height, pair_count, vertical_clip_side)
        row_to_k = _assign_spans(width, rows, axis_row, horizontal_clip)
    except Exception:
        return None

    mid = width // 2
    grid = np.zeros((height, width), dtype=int)
    row_to_cols: dict[int, tuple[int, int]] = {}
    for r in rows:
        k = row_to_k[r]
        c0 = mid - k
        c1 = mid + k
        row_to_cols[r] = (c0, c1)
        grid[r, c0] = GREEN
        grid[r, c1] = GREEN

    _place_reds(grid, rows, row_to_cols, vertical_clip_side)

    try:
        _semantic_check(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
