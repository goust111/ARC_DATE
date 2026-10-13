import math
import random
from typing import Iterable

import numpy as np


BG = 0
RED = 2
GREEN = 3
GREY = 5
TEAL = 8

LEFT_W = 6
DIVIDER_COLOR = GREY
AREA_TO_SHIFT = {3: -2, 4: -1, 5: 0, 6: 1, 7: 2}

GLYPH_TEMPLATES = {
    3: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int),
    4: np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=int),
    5: np.array([[1, 1, 0], [0, 1, 0], [1, 0, 1]], dtype=int),
    6: np.array([[1, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=int),
    7: np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=int),
}

MAX_GEN_ATTEMPTS = 400


def _neighbors(r: int, c: int, conn8: bool) -> Iterable[tuple[int, int]]:
    if conn8:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                yield r + dr, c + dc
        return
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray, conn8: bool = False) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, conn8):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _find_divider_col(grid: np.ndarray) -> int:
    matches = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == DIVIDER_COLOR)]
    assert len(matches) == 1, "expected a unique full-height grey divider"
    return int(matches[0])


def _decode_glyph_offsets(left: np.ndarray) -> list[int]:
    glyph_parts = []
    for coords in _components(left == TEAL, conn8=True):
        r0, c0, r1, c1 = _bbox(coords)
        if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
            continue
        area = int(coords.shape[0])
        assert area in AREA_TO_SHIFT, f"unsupported glyph area: {area}"
        glyph_parts.append((r0, area))
    glyph_parts.sort()
    assert 3 <= len(glyph_parts) <= 5, "expected 3-5 glyphs"
    return [AREA_TO_SHIFT[area] for _, area in glyph_parts]


def _marker_height(right: np.ndarray) -> int:
    h = 0
    while h < right.shape[0] and int(right[h, 0]) == GREY:
        h += 1
    assert 3 <= h <= 4, "expected a top grey marker run of height 3 or 4"
    assert np.count_nonzero(right[:, 0] == GREY) == h, "marker column must be a single top run"
    return h


def _clean_right_zone(right: np.ndarray) -> np.ndarray:
    cleaned = np.array(right, dtype=int, copy=True)
    cleaned[cleaned == GREY] = BG
    return cleaned


def _render_cycles(cleaned: np.ndarray, motif_h: int, offsets: list[int], start_row: int) -> np.ndarray:
    out = np.array(cleaned, dtype=int, copy=True)
    h, w = out.shape
    motif = out[:motif_h].copy()
    assert np.all(np.any(motif != BG, axis=1)), "every motif row must contain content"
    assert np.all(out[start_row:] == BG), "suffix must be blank before filling"
    for r in range(start_row, h):
        cycle_idx = (r - start_row) // motif_h
        src_row = motif[(r - start_row) % motif_h]
        shift = int(offsets[cycle_idx % len(offsets)])
        for c in np.flatnonzero(src_row != BG):
            tc = int(c + shift)
            if 0 <= tc < w:
                out[r, tc] = int(src_row[c])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode left glyph areas into per-cycle horizontal shifts and fill the right suffix with shifted motif repeats.

    Input structure:
    - The grid has black background, a full-height grey vertical divider, teal 3x3 glyphs on the left, and a motif panel on the right.
    - On the right side, the first column has a top contiguous grey run marking the motif rows; below it may be 0-2 rows of existing red/green content, then an empty black suffix.
    - Each left glyph is a separate 3x3 teal component, read from top to bottom.

    Rule:
    - Count each glyph's teal area and map it to a horizontal shift: 3 -> left 2, 4 -> left 1, 5 -> 0, 6 -> right 1, 7 -> right 2.
    - Remove the left area and divider; keep only the right panel, replacing all grey marker pixels with black.
    - Extract the top marker-delimited motif rows and, starting below the lowest existing right-side foreground row, copy those rows cyclically into the empty suffix.
    - For cycle 1 use the first decoded shift, for cycle 2 the second, and continue cyclically through the glyph sequence; shifted pixels that leave the panel are clipped.

    Key insight:
    - The left glyphs do not encode what to draw; their pixel areas encode a repeating sequence of per-cycle offsets applied to the same right-side motif as it is copied downward.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) > 0

    divider_c = _find_divider_col(g)
    left = g[:, :divider_c]
    right = g[:, divider_c + 1 :]
    assert left.shape[1] >= 3 and right.shape[1] >= 6

    offsets = _decode_glyph_offsets(left)
    motif_h = _marker_height(right)
    cleaned = _clean_right_zone(right)

    foreground_rows = np.flatnonzero(np.any(cleaned != BG, axis=1))
    assert foreground_rows.size > 0, "right panel must contain a motif"
    start_row = int(foreground_rows[-1]) + 1
    if start_row >= cleaned.shape[0]:
        return cleaned
    return _render_cycles(cleaned, motif_h, offsets, start_row)


def _sample_area_sequence(rng: random.Random, count: int) -> list[int]:
    negatives = [3, 4]
    positives = [6, 7]
    areas = [rng.choice(negatives), rng.choice(positives)]
    if count >= 3 and rng.random() < 0.7:
        areas.append(5)
    unused = [a for a in AREA_TO_SHIFT if a not in areas]
    rng.shuffle(unused)
    while len(areas) < count:
        areas.append(unused.pop())
    rng.shuffle(areas)
    return areas


def _pick_cycle_count(
    rng: random.Random,
    glyph_count: int,
    glyph_h: int,
    start_row: int,
    motif_h: int,
) -> tuple[int, bool]:
    min_cycles = max(2, math.ceil(max(0, glyph_h - start_row) / motif_h))
    max_cycles = min(6, (30 - start_row) // motif_h)
    assert min_cycles <= max_cycles

    wrap_min = max(min_cycles, glyph_count + 1)
    can_wrap = wrap_min <= max_cycles
    want_wrap = can_wrap and rng.random() < 0.55
    if want_wrap:
        return rng.randint(wrap_min, max_cycles), True

    non_wrap_max = min(max_cycles, glyph_count)
    if min_cycles <= non_wrap_max:
        return rng.randint(min_cycles, non_wrap_max), False
    return rng.randint(wrap_min, max_cycles), True


def _build_motif(
    rng: random.Random,
    right_w: int,
    motif_h: int,
    mode: str,
) -> tuple[np.ndarray, int, int]:
    assert mode in {"left_clip", "right_clip", "center"}
    max_w = min(10, right_w - 1)
    min_w = min(6, max_w)
    motif_w = rng.randint(min_w, max_w)

    if mode == "left_clip":
        anchor = rng.randint(1, min(2, right_w - motif_w))
    elif mode == "right_clip":
        max_anchor = right_w - motif_w
        anchor = rng.randint(max(1, max_anchor - 1), max_anchor)
    else:
        low = 2
        high = right_w - motif_w - 2
        if high < low:
            anchor = max(1, (right_w - motif_w) // 2)
        else:
            anchor = rng.randint(low, high)

    local = np.zeros((motif_h, motif_w), dtype=int)
    for r in range(motif_h):
        k = rng.randint(2, min(4, motif_w))
        cols = sorted(rng.sample(range(motif_w), k))
        local[r, cols] = 1
    local[0, 0] = 1
    local[-1, -1] = 1

    if len({tuple(row.tolist()) for row in local}) == 1:
        local[-1, max(0, motif_w - 2)] = 1 - local[-1, max(0, motif_w - 2)]
        if not local[-1].any():
            local[-1, -1] = 1

    motif = np.zeros((motif_h, right_w), dtype=int)
    phase = rng.randint(0, 1)
    for r in range(motif_h):
        for c in range(motif_w):
            if not local[r, c]:
                continue
            motif[r, anchor + c] = RED if (r + c + phase) % 2 == 0 else GREEN

    colors = set(map(int, motif[motif != BG].tolist()))
    assert colors == {RED, GREEN}, "motif must use both colors"
    return motif, anchor, motif_w


def _add_middle_rows(
    rng: random.Random,
    right: np.ndarray,
    start_row: int,
    anchor: int,
    motif_w: int,
) -> None:
    for r in range(_marker_height(right), start_row):
        k = rng.randint(1, min(4, motif_w))
        span = list(range(anchor, anchor + motif_w))
        cols = sorted(rng.sample(span, k))
        for c in cols:
            right[r, c] = RED if rng.random() < 0.5 else GREEN


def _expected_output(
    right: np.ndarray,
    motif_h: int,
    offsets: list[int],
) -> np.ndarray:
    cleaned = _clean_right_zone(right)
    foreground_rows = np.flatnonzero(np.any(cleaned != BG, axis=1))
    start_row = int(foreground_rows[-1]) + 1
    if start_row >= cleaned.shape[0]:
        return cleaned
    return _render_cycles(cleaned, motif_h, offsets, start_row)


def _clip_happens(motif: np.ndarray, shift: int) -> bool:
    h, w = motif.shape
    for r in range(h):
        for c in np.flatnonzero(motif[r] != BG):
            tc = int(c + shift)
            if tc < 0 or tc >= w:
                return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    glyph_count = rng.randint(3, 5)
    areas = _sample_area_sequence(rng, glyph_count)
    offsets = [AREA_TO_SHIFT[a] for a in areas]
    glyph_h = 4 * glyph_count - 1

    right_w = rng.randint(12, 16)
    motif_h = rng.randint(3, 4)
    middle_h = rng.choice([0, 1, 1, 2, 2])
    start_row = motif_h + middle_h
    cycle_count, wraps = _pick_cycle_count(rng, glyph_count, glyph_h, start_row, motif_h)
    h = start_row + motif_h * cycle_count
    if h > 30:
        return None

    mode = rng.choice(["left_clip", "right_clip", "center"])
    motif, anchor, motif_w = _build_motif(rng, right_w, motif_h, mode)

    right = np.zeros((h, right_w), dtype=int)
    right[:motif_h] = motif
    right[:motif_h, 0] = GREY
    _add_middle_rows(rng, right, start_row, anchor, motif_w)

    left = np.zeros((h, LEFT_W), dtype=int)
    glyph_c0 = rng.randint(1, LEFT_W - 3)
    for i, area in enumerate(areas):
        r0 = 4 * i
        tile = GLYPH_TEMPLATES[area]
        left[r0 : r0 + 3, glyph_c0 : glyph_c0 + 3][tile == 1] = TEAL

    grid = np.zeros((h, LEFT_W + 1 + right_w), dtype=int)
    grid[:, :LEFT_W] = left
    grid[:, LEFT_W] = GREY
    grid[:, LEFT_W + 1 :] = right

    expected = _expected_output(right, motif_h, offsets)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    if np.count_nonzero(expected[start_row:] != BG) == 0:
        return None
    if len(set(offsets)) != len(offsets):
        return None
    if not any(s < 0 for s in offsets) or not any(s > 0 for s in offsets):
        return None
    if 0 not in offsets and rng.random() < 0.25:
        return None
    if wraps and cycle_count <= glyph_count:
        return None
    if not wraps and cycle_count > glyph_count:
        return None

    if mode == "left_clip" and not any(_clip_happens(motif, s) for s in offsets if s < 0):
        return None
    if mode == "right_clip" and not any(_clip_happens(motif, s) for s in offsets if s > 0):
        return None
    if mode == "center" and any(_clip_happens(motif, s) for s in offsets):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GEN_ATTEMPTS} attempts")
