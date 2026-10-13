from __future__ import annotations

from collections import Counter
from importlib.machinery import SourceFileLoader
import random

import numpy as np


try:
    _ARC = SourceFileLoader(
        "_shared_arc_utils_for_fable_054_5",
        "/home/lixinyu/snap/new/solvers/arc_utils.py",
    ).load_module()
    Color = _ARC.Color
    bbox = _ARC.bbox
    crop_bbox = _ARC.crop_bbox
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9

    def bbox(mask: np.ndarray):
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def crop_bbox(grid: np.ndarray, box):
        r0, c0, r1, c1 = box
        return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


SPECIAL_COLORS = {int(Color.BLACK), int(Color.YELLOW)}
PATTERN_COLORS = [c for c in range(10) if c not in SPECIAL_COLORS]
MAX_GENERATION_ATTEMPTS = 800


def _visible_mask(grid: np.ndarray) -> np.ndarray:
    return (grid != int(Color.BLACK)) & (grid != int(Color.YELLOW))


def _infer_missing_corner(markers: list[tuple[int, int]]) -> tuple[int, int]:
    row_counts = Counter(r for r, _ in markers)
    col_counts = Counter(c for _, c in markers)
    assert len(row_counts) == 2 and sorted(row_counts.values()) == [1, 2]
    assert len(col_counts) == 2 and sorted(col_counts.values()) == [1, 2]
    missing_r = next(r for r, count in row_counts.items() if count == 1)
    missing_c = next(c for c, count in col_counts.items() if count == 1)
    return int(missing_r), int(missing_c)


def _target_bbox_from_markers(markers: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    missing = _infer_missing_corner(markers)
    corners = markers + [missing]
    rows = [r for r, _ in corners]
    cols = [c for _, c in corners]
    return missing, (min(rows), min(cols), max(rows), max(cols))


def _point_in_bbox(pt: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    r, c = pt
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _bbox_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _bbox_intersection(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _residue_signature(
    grid: np.ndarray,
    visible: np.ndarray,
    row_period: int,
    col_period: int,
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]] | None:
    residue_map: dict[tuple[int, int], int] = {}
    counts: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(visible):
        key = (int(r) % row_period, int(c) % col_period)
        color = int(grid[int(r), int(c)])
        if key in residue_map and residue_map[key] != color:
            return None
        residue_map[key] = color
        counts[key] = counts.get(key, 0) + 1
    return residue_map, counts


def _period_candidates(
    grid: np.ndarray,
    *,
    min_visible_per_class: int = 1,
) -> list[tuple[int, int, dict[tuple[int, int], int], dict[tuple[int, int], int]]]:
    visible = _visible_mask(grid)
    h, w = grid.shape
    out = []
    for row_period in range(1, h):
        if h < 2 * row_period:
            continue
        for col_period in range(1, w):
            if w < 2 * col_period:
                continue
            sig = _residue_signature(grid, visible, row_period, col_period)
            if sig is None:
                continue
            residue_map, counts = sig
            if len(residue_map) != row_period * col_period:
                continue
            if min_visible_per_class > 1 and min(counts.values()) < min_visible_per_class:
                continue
            out.append((row_period, col_period, residue_map, counts))
    out.sort(key=lambda item: (item[0] * item[1], item[0], item[1]))
    return out


def _detect_periods(grid: np.ndarray) -> tuple[int, int, dict[tuple[int, int], int]]:
    candidates = _period_candidates(grid, min_visible_per_class=1)
    assert candidates, "could not infer a valid translational period"
    row_period, col_period, residue_map, _counts = candidates[0]
    return int(row_period), int(col_period), residue_map


def _restore_window(
    grid: np.ndarray,
    target_bbox: tuple[int, int, int, int],
    row_period: int,
    col_period: int,
    residue_map: dict[tuple[int, int], int],
) -> np.ndarray:
    r0, c0, _r1, _c1 = target_bbox
    window = crop_bbox(grid, target_bbox)
    for wr in range(window.shape[0]):
        for wc in range(window.shape[1]):
            if int(window[wr, wc]) in SPECIAL_COLORS:
                key = ((r0 + wr) % row_period, (c0 + wc) % col_period)
                window[wr, wc] = residue_map[key]
    return window


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the periodic pattern inside the yellow-cornered rectangle after inferring its missing corner.

    Input structure:
    - The whole grid is a row-periodic and column-periodic multicolor pattern whose palette excludes black and yellow.
    - One solid black rectangle occludes part of that pattern.
    - Three yellow singleton pixels mark three corners of an axis-aligned target rectangle; the fourth corner is absent.

    Rule:
    - Infer the missing fourth rectangle corner from the repeated row and column coordinates of the three yellow markers.
    - Ignore black and yellow cells, then find the smallest row and column periods whose residue classes explain every visible pattern cell.
    - Restore the yellow-marker cells and any black-covered target cells from the periodic residue map, and output only the recovered target rectangle.

    Key insight:
    - Two hidden pieces must be completed at once: the rectangle geometry is completed from three corners, while the unseen content is completed from the grid's translational symmetry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    black_box = bbox(g == int(Color.BLACK))
    assert black_box is not None, "expected a black occluding rectangle"
    br0, bc0, br1, bc1 = black_box
    assert np.all(g[br0 : br1 + 1, bc0 : bc1 + 1] == int(Color.BLACK))

    yellow_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == int(Color.YELLOW))]
    assert len(yellow_coords) == 3, "expected exactly three yellow corner markers"
    missing_corner, target_bbox = _target_bbox_from_markers(yellow_coords)
    assert not any(y == missing_corner for y in yellow_coords)

    row_period, col_period, residue_map = _detect_periods(g)
    return _restore_window(g, target_bbox, row_period, col_period, residue_map)


def _axis_period(tile: np.ndarray, axis: int) -> int:
    n = tile.shape[axis]
    idx = np.arange(n)
    for period in range(1, n + 1):
        if np.array_equal(tile, np.take(tile, idx % period, axis=axis)):
            return int(period)
    raise AssertionError("unreachable")


def _sample_tile(row_period: int, col_period: int) -> np.ndarray:
    for _ in range(500):
        palette_size = random.randint(3, min(6, len(PATTERN_COLORS)))
        palette = random.sample(PATTERN_COLORS, k=palette_size)
        tile = np.random.choice(palette, size=(row_period, col_period)).astype(int)
        if len(np.unique(tile)) < 3:
            continue
        if _axis_period(tile, 0) != row_period:
            continue
        if _axis_period(tile, 1) != col_period:
            continue
        row_patterns = {tuple(map(int, row)) for row in tile}
        col_patterns = {tuple(map(int, col)) for col in tile.T}
        if len(row_patterns) < 2 or len(col_patterns) < 2:
            continue
        return tile
    raise RuntimeError("failed to sample a primitive periodic tile")


def _sample_clean_pattern() -> tuple[np.ndarray, int, int]:
    row_period = random.randint(2, 5)
    col_period = random.randint(2, 5)
    row_repeats = random.randint(3, min(5, 30 // row_period))
    col_repeats = random.randint(3, min(5, 30 // col_period))
    h = row_period * row_repeats
    w = col_period * col_repeats
    tile = _sample_tile(row_period, col_period)
    tiled = np.tile(tile, (row_repeats + 1, col_repeats + 1))
    row_offset = random.randrange(row_period)
    col_offset = random.randrange(col_period)
    clean = tiled[row_offset : row_offset + h, col_offset : col_offset + w].copy()
    return clean, row_period, col_period


def _sample_target_bbox(h: int, w: int, row_period: int, col_period: int) -> tuple[int, int, int, int]:
    min_h = max(3, row_period)
    min_w = max(3, col_period)
    max_h = min(h - 1, row_period * 2 + 3)
    max_w = min(w - 1, col_period * 2 + 3)
    if min_h > max_h:
        min_h = 3
        max_h = h - 1
    if min_w > max_w:
        min_w = 3
        max_w = w - 1
    target_h = random.randint(min_h, max_h)
    target_w = random.randint(min_w, max_w)
    r0 = random.randint(0, h - target_h)
    c0 = random.randint(0, w - target_w)
    return int(r0), int(c0), int(r0 + target_h - 1), int(c0 + target_w - 1)


def _rectangle_corners(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]


def _choose_cross_boundary_black(
    clean: np.ndarray,
    target_bbox: tuple[int, int, int, int],
    visible_corners: list[tuple[int, int]],
    missing_corner: tuple[int, int],
    cover_missing: bool,
) -> tuple[int, int, int, int] | None:
    h, w = clean.shape
    tr0, tc0, tr1, tc1 = target_bbox
    target_h = tr1 - tr0 + 1
    target_w = tc1 - tc0 + 1
    for _ in range(400):
        bh = random.randint(2, min(h, max(3, target_h + 2)))
        bw = random.randint(2, min(w, max(3, target_w + 2)))
        r_lo = max(0, tr0 - bh + 1)
        r_hi = min(h - bh, tr1)
        c_lo = max(0, tc0 - bw + 1)
        c_hi = min(w - bw, tc1)
        if r_lo > r_hi or c_lo > c_hi:
            continue
        br0 = random.randint(r_lo, r_hi)
        bc0 = random.randint(c_lo, c_hi)
        box = (br0, bc0, br0 + bh - 1, bc0 + bw - 1)
        if tr0 <= box[0] and box[2] <= tr1 and tc0 <= box[1] and box[3] <= tc1:
            continue
        if any(_point_in_bbox(pt, box) for pt in visible_corners):
            continue
        if _point_in_bbox(missing_corner, box) != cover_missing:
            continue
        if _bbox_area(_bbox_intersection(box, target_bbox) or (0, 0, -1, -1)) == 0:
            continue
        hidden = [
            (r, c)
            for r in range(max(box[0], tr0), min(box[2], tr1) + 1)
            for c in range(max(box[1], tc0), min(box[3], tc1) + 1)
            if (r, c) not in _rectangle_corners(target_bbox)
        ]
        if not hidden:
            continue
        if len({int(clean[r, c]) for r, c in hidden}) < 2:
            continue
        return box
    return None


def _choose_black_bbox(
    clean: np.ndarray,
    target_bbox: tuple[int, int, int, int],
    missing_corner: tuple[int, int],
    visible_corners: list[tuple[int, int]],
) -> tuple[int, int, int, int] | None:
    tr0, tc0, tr1, tc1 = target_bbox
    target_h = tr1 - tr0 + 1
    target_w = tc1 - tc0 + 1
    cover_missing = random.random() < 0.6
    inside_target = cover_missing or (target_h >= 4 and target_w >= 4 and random.random() < 0.5)

    if inside_target:
        for _ in range(400):
            bh = random.randint(2, target_h - 1)
            bw = random.randint(2, target_w - 1)
            br0 = random.randint(tr0, tr1 - bh + 1)
            bc0 = random.randint(tc0, tc1 - bw + 1)
            box = (br0, bc0, br0 + bh - 1, bc0 + bw - 1)
            if any(_point_in_bbox(pt, box) for pt in visible_corners):
                continue
            if _point_in_bbox(missing_corner, box) != cover_missing:
                continue
            hidden = [
                (r, c)
                for r in range(box[0], box[2] + 1)
                for c in range(box[1], box[3] + 1)
                if (r, c) not in _rectangle_corners(target_bbox)
            ]
            if not hidden:
                continue
            if len({int(clean[r, c]) for r, c in hidden}) < 2:
                continue
            return box
    return _choose_cross_boundary_black(clean, target_bbox, visible_corners, missing_corner, cover_missing)


def _paint_black(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(Color.BLACK)


def _expected_output(clean: np.ndarray, target_bbox: tuple[int, int, int, int]) -> np.ndarray:
    return crop_bbox(clean, target_bbox)


def _validate_instance(
    inp: np.ndarray,
    clean: np.ndarray,
    target_bbox: tuple[int, int, int, int],
    row_period: int,
    col_period: int,
) -> bool:
    expected = _expected_output(clean, target_bbox)
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    yellow_coords = [tuple(map(int, rc)) for rc in np.argwhere(inp == int(Color.YELLOW))]
    if len(yellow_coords) != 3:
        return False
    black_box = bbox(inp == int(Color.BLACK))
    if black_box is None:
        return False
    br0, bc0, br1, bc1 = black_box
    if not np.all(inp[br0 : br1 + 1, bc0 : bc1 + 1] == int(Color.BLACK)):
        return False

    candidates = _period_candidates(inp, min_visible_per_class=2)
    if not candidates:
        return False
    best_rp, best_cp, residue_map, counts = candidates[0]
    if (best_rp, best_cp) != (row_period, col_period):
        return False
    if sum(1 for rp, cp, _rm, _ct in candidates if rp * cp == row_period * col_period) != 1:
        return False
    if min(counts.values()) < 2 or len(residue_map) != row_period * col_period:
        return False

    raw_window = crop_bbox(inp, target_bbox)
    if np.count_nonzero(raw_window == int(Color.BLACK)) == 0:
        return False
    if np.count_nonzero(raw_window == int(Color.YELLOW)) != 3:
        return False
    if np.array_equal(raw_window, expected):
        return False

    corners = set(_rectangle_corners(target_bbox))
    hidden_noncorner = [
        (r, c)
        for r in range(target_bbox[0], target_bbox[2] + 1)
        for c in range(target_bbox[1], target_bbox[3] + 1)
        if int(inp[r, c]) == int(Color.BLACK) and (r, c) not in corners
    ]
    if not hidden_noncorner:
        return False
    if len({int(clean[r, c]) for r, c in hidden_noncorner}) < 2:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        clean, row_period, col_period = _sample_clean_pattern()
        h, w = clean.shape
        target_bbox = _sample_target_bbox(h, w, row_period, col_period)
        corners = _rectangle_corners(target_bbox)
        missing_index = random.randrange(4)
        missing_corner = corners[missing_index]
        visible_corners = [pt for i, pt in enumerate(corners) if i != missing_index]

        black_bbox = _choose_black_bbox(clean, target_bbox, missing_corner, visible_corners)
        if black_bbox is None:
            continue

        inp = clean.copy()
        _paint_black(inp, black_bbox)
        for r, c in visible_corners:
            inp[r, c] = int(Color.YELLOW)

        if _validate_instance(inp, clean, target_bbox, row_period, col_period):
            return inp

    raise RuntimeError(f"failed to generate a valid sample after {MAX_GENERATION_ATTEMPTS} attempts")
