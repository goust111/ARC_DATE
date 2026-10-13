from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

MOTIF_COLORS = [3, 4, 6, 7, 8, 9]
STRUCTURE_COLORS = {BLACK, BLUE, RED, GREY}

GEN_RETRIES = 600


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _contiguous_cols(row: np.ndarray, color: int) -> list[int]:
    cols = np.flatnonzero(row == color).tolist()
    assert cols, f"expected color {color} in row"
    assert cols[-1] - cols[0] + 1 == len(cols), "expected one contiguous segment"
    return [int(c) for c in cols]


def _find_separator(g: np.ndarray) -> int:
    rows = [r for r in range(1, g.shape[0] - 1) if np.all(g[r] == RED)]
    assert len(rows) == 1, "expected exactly one internal red separator row"
    return int(rows[0])


def _find_ruler(g: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(g == GREY)
    assert len(coords) >= 2, "expected one vertical grey ruler"
    cols = sorted({int(c) for _, c in coords})
    assert len(cols) == 1 and cols[0] in (0, g.shape[1] - 1), "ruler must lie on one side edge"
    rows = sorted(int(r) for r, _ in coords)
    assert rows[-1] - rows[0] + 1 == len(rows), "ruler must be contiguous"
    return cols[0], rows[0], rows[-1], len(rows)


def _find_motif(g: np.ndarray, ruler_col: int, r0: int, r1: int, overlap_w: int) -> np.ndarray:
    mask = ~np.isin(g, list(STRUCTURE_COLORS))
    coords = np.argwhere(mask)
    assert len(coords) > 0, "expected one attached motif"

    mr0, mc0 = map(int, coords.min(axis=0))
    mr1, mc1 = map(int, coords.max(axis=0))
    assert len(coords) == (mr1 - mr0 + 1) * (mc1 - mc0 + 1), "motif must be rectangular"
    assert 1 <= mr1 - mr0 + 1 <= 3, "motif height out of range"
    assert mc1 - mc0 + 1 == overlap_w, "motif width must match overlap width"
    assert mr0 == r0 or mr1 == r1, "motif must attach to one ruler end"

    if ruler_col == 0:
        assert mc0 == 1, "left ruler motif must be immediately to the right"
    else:
        assert mc1 == g.shape[1] - 2, "right ruler motif must be immediately to the left"

    return g[mr0 : mr1 + 1, mc0 : mc1 + 1].copy()


def _paint_output(g: np.ndarray, separator_r: int, overlap_left: int, motif: np.ndarray, repeats: int, active_top: bool) -> np.ndarray:
    out = g.copy()
    tile_stack = np.tile(motif, (repeats, 1))
    fill_h = tile_stack.shape[0]
    fill_w = tile_stack.shape[1]

    if active_top:
        row0 = 1
        row1 = row0 + fill_h
        assert row1 <= separator_r, "tiled block exceeds upper half"
    else:
        row1 = g.shape[0] - 1
        row0 = row1 - fill_h
        assert row0 > separator_r, "tiled block exceeds lower half"

    target = out[row0:row1, overlap_left : overlap_left + fill_w]
    assert np.all(target == BLACK), "fill region must be empty in the input"
    out[row0:row1, overlap_left : overlap_left + fill_w] = tile_stack
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Tile the ruler-end motif inside the overlap columns on the side of the longer edge bar.

    Input structure:
    - A black-background grid has one blue horizontal segment on the top row and one blue horizontal segment on the bottom row.
    - One internal row is a full-width red separator splitting the grid into upper and lower halves.
    - A vertical grey ruler sits on the left or right edge, and one non-structural rectangular motif is attached to its top end or bottom end.

    Rule:
    - Compute the column overlap of the top and bottom blue segments.
    - The longer blue segment chooses the active half: top if the top segment is longer, bottom if the bottom segment is longer.
    - The ruler length gives the number of repeats, and the attached motif gives the tile.
    - Repeat that motif vertically `ruler_length` times inside the overlap columns, starting just below the top bar for the upper half or ending just above the bottom bar for the lower half.

    Key insight:
    - The output needs three independent cues at once: overlap columns from the two bars, side selection from the longer bar, and repeat count plus tile content from the ruler object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert get_bg(g) == BLACK, "expected black background"

    separator_r = _find_separator(g)

    top_cols = _contiguous_cols(g[0], BLUE)
    bottom_cols = _contiguous_cols(g[-1], BLUE)
    assert len(top_cols) != len(bottom_cols), "top and bottom bars must differ in length"

    overlap = sorted(set(top_cols) & set(bottom_cols))
    assert len(overlap) >= 2, "overlap must be at least two columns wide"
    assert overlap[-1] - overlap[0] + 1 == len(overlap), "overlap must be contiguous"

    ruler_col, ruler_r0, ruler_r1, repeats = _find_ruler(g)
    motif = _find_motif(g, ruler_col, ruler_r0, ruler_r1, len(overlap))

    active_top = len(top_cols) > len(bottom_cols)
    return _paint_output(g, separator_r, overlap[0], motif, repeats, active_top)


def _motif_palette(area: int) -> int:
    if area == 1:
        return 1
    return random.randint(2, min(3, area))


def _make_motif(height: int, width: int) -> np.ndarray:
    area = height * width
    for _ in range(100):
        n_colors = _motif_palette(area)
        palette = random.sample(MOTIF_COLORS, n_colors)
        motif = np.array(random.choices(palette, k=area), dtype=int).reshape(height, width)
        for idx, color in zip(random.sample(range(area), n_colors), palette):
            motif.flat[idx] = color
        if area > 1 and len(np.unique(motif)) < 2:
            continue
        if height > 1 and all(np.array_equal(motif[0], motif[r]) for r in range(1, height)):
            continue
        if width > 1 and all(np.array_equal(motif[:, 0], motif[:, c]) for c in range(1, width)):
            continue
        return motif
    raise RuntimeError("failed to build a non-degenerate motif")


def _candidate_intervals(overlap_left: int, overlap_w: int, length: int, left_guard: int, right_guard: int) -> list[tuple[int, int]]:
    extra = length - overlap_w
    out: list[tuple[int, int]] = []
    for left_ext in range(extra + 1):
        start = overlap_left - left_ext
        end = overlap_left + overlap_w - 1 + (extra - left_ext)
        if start >= left_guard and end <= right_guard:
            out.append((start, end))
    return out


def _choose_bar_layout(width: int, overlap_w: int, active_top: bool, ruler_left: bool) -> tuple[tuple[int, int], tuple[int, int], int]:
    left_guard = overlap_w + 2 if ruler_left else 1
    right_guard = width - overlap_w - 3 if not ruler_left else width - 2
    if left_guard + overlap_w - 1 > right_guard:
        raise RuntimeError("not enough width for disjoint ruler and overlap")

    lengths = [n for n in range(max(3, overlap_w + 1), 9)]
    if active_top:
        len_pairs = [(t, b) for t in lengths for b in lengths if t > b]
    else:
        len_pairs = [(t, b) for t in lengths for b in lengths if b > t]
    random.shuffle(len_pairs)

    overlap_starts = list(range(left_guard, right_guard - overlap_w + 2))
    random.shuffle(overlap_starts)

    for overlap_left in overlap_starts:
        for top_len, bottom_len in len_pairs:
            top_cands = _candidate_intervals(overlap_left, overlap_w, top_len, left_guard, right_guard)
            bot_cands = _candidate_intervals(overlap_left, overlap_w, bottom_len, left_guard, right_guard)
            pairs = [
                (top_seg, bottom_seg)
                for top_seg in top_cands
                for bottom_seg in bot_cands
                if top_seg[0] != bottom_seg[0] and top_seg[1] != bottom_seg[1]
            ]
            if not pairs:
                continue
            top_seg, bottom_seg = random.choice(pairs)
            return top_seg, bottom_seg, overlap_left
    raise RuntimeError("failed to choose partial-overlap edge bars")


def _valid_repeat_specs(available_rows: int) -> list[tuple[int, int]]:
    specs: list[tuple[int, int]] = []
    for repeats in range(2, min(5, available_rows) + 1):
        for motif_h in range(1, min(3, repeats, available_rows // repeats) + 1):
            specs.append((repeats, motif_h))
    return specs


def _place_ruler(height: int, separator_r: int, repeats: int) -> tuple[int, int]:
    regions: list[tuple[int, int]] = []
    if separator_r - 1 >= repeats:
        regions.append((1, separator_r - repeats))
    if height - separator_r - 2 >= repeats:
        regions.append((separator_r + 1, height - 1 - repeats))
    assert regions, "ruler must fit in at least one half"

    lo, hi = random.choice(regions)
    start = random.randint(lo, hi)
    return start, start + repeats - 1


def _build_instance() -> tuple[np.ndarray, np.ndarray]:
    height = random.randint(14, 20)
    width = random.randint(16, 24)
    separator_r = random.randint(4, height - 5)
    active_top = random.choice([True, False])

    active_rows = separator_r - 1 if active_top else height - separator_r - 2
    repeat_specs = _valid_repeat_specs(active_rows)
    if not repeat_specs:
        raise RuntimeError("active half too small")
    tall_specs = [spec for spec in repeat_specs if spec[1] == 3]
    medium_specs = [spec for spec in repeat_specs if spec[1] == 2]
    if tall_specs and random.random() < 0.4:
        repeats, motif_h = random.choice(tall_specs)
    elif medium_specs and random.random() < 0.5:
        repeats, motif_h = random.choice(medium_specs)
    else:
        repeats, motif_h = random.choice(repeat_specs)

    ruler_left = random.choice([True, False])
    max_overlap = min(6, width - 10)
    if max_overlap < 2:
        raise RuntimeError("width too small for overlap")
    overlap_w = random.randint(2, max_overlap)

    top_seg, bottom_seg, overlap_left = _choose_bar_layout(width, overlap_w, active_top, ruler_left)
    motif = _make_motif(motif_h, overlap_w)

    ruler_r0, ruler_r1 = _place_ruler(height, separator_r, repeats)
    attach_top = random.choice([True, False])

    g = np.full((height, width), BLACK, dtype=int)
    g[separator_r, :] = RED
    g[0, top_seg[0] : top_seg[1] + 1] = BLUE
    g[-1, bottom_seg[0] : bottom_seg[1] + 1] = BLUE

    ruler_col = 0 if ruler_left else width - 1
    g[ruler_r0 : ruler_r1 + 1, ruler_col] = GREY

    if ruler_left:
        motif_c0 = 1
    else:
        motif_c0 = width - 1 - overlap_w
    motif_c1 = motif_c0 + overlap_w

    if attach_top:
        motif_r0 = ruler_r0
    else:
        motif_r0 = ruler_r1 - motif_h + 1
    motif_r1 = motif_r0 + motif_h
    g[motif_r0:motif_r1, motif_c0:motif_c1] = motif

    expected = _paint_output(g, separator_r, overlap_left, motif, repeats, active_top)
    return g, expected


def _validate_semantics(inp: np.ndarray, out: np.ndarray) -> None:
    assert inp.shape == out.shape
    assert inp.ndim == 2 and max(inp.shape) <= 30
    assert out.ndim == 2 and max(out.shape) <= 30
    assert not np.array_equal(inp, out), "output must change the input"

    separator_r = _find_separator(inp)
    top_cols = _contiguous_cols(inp[0], BLUE)
    bottom_cols = _contiguous_cols(inp[-1], BLUE)
    overlap = sorted(set(top_cols) & set(bottom_cols))
    assert len(overlap) >= 2
    assert len(overlap) < min(len(top_cols), len(bottom_cols)), "overlap should not equal the full shorter segment"
    assert top_cols[0] != bottom_cols[0] and top_cols[-1] != bottom_cols[-1], "bars should differ on both sides"

    ruler_col, ruler_r0, ruler_r1, repeats = _find_ruler(inp)
    motif = _find_motif(inp, ruler_col, ruler_r0, ruler_r1, len(overlap))
    motif_h = motif.shape[0]
    active_top = len(top_cols) > len(bottom_cols)

    if active_top:
        fill_rows = slice(1, 1 + motif_h * repeats)
        assert 1 + motif_h * repeats <= separator_r
    else:
        fill_rows = slice(inp.shape[0] - 1 - motif_h * repeats, inp.shape[0] - 1)
        assert inp.shape[0] - 1 - motif_h * repeats > separator_r

    in_target = inp[fill_rows, overlap[0] : overlap[-1] + 1]
    out_target = out[fill_rows, overlap[0] : overlap[-1] + 1]
    assert np.all(in_target == BLACK), "target region must start empty"
    assert np.array_equal(out_target, np.tile(motif, (repeats, 1))), "output target must be a pure repeated tile"

    untouched = inp.copy()
    untouched[fill_rows, overlap[0] : overlap[-1] + 1] = out_target
    assert np.array_equal(untouched, out), "only the target region may change"


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            inp, expected = _build_instance()
            actual = main(inp)
            _validate_semantics(inp, expected)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 4")
