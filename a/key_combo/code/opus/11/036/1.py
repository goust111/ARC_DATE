from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

HORIZONTAL = "H"
VERTICAL = "V"

MIN_SIZE = 13
MAX_SIZE = 16
ODD_LENGTHS = (3, 5, 7)
GEN_ATTEMPTS = 600


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    return int(counts.most_common(1)[0][0])


def _axis_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return b0 - a1 - 1
    if b1 < a0:
        return a0 - b1 - 1
    return 0


def _boxes_have_blank_gap(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> bool:
    ra0, ca0, ra1, ca1 = box_a
    rb0, cb0, rb1, cb1 = box_b
    row_gap = _axis_gap(ra0, ra1, rb0, rb1)
    col_gap = _axis_gap(ca0, ca1, cb0, cb1)
    return row_gap >= 1 or col_gap >= 1


def _bar_pixels(center: tuple[int, int], length: int, orient: str) -> list[tuple[int, int]]:
    r, c = center
    half = length // 2
    if orient == HORIZONTAL:
        return [(r, cc) for cc in range(c - half, c + half + 1)]
    return [(rr, c) for rr in range(r - half, r + half + 1)]


def _bar_bbox(center: tuple[int, int], length: int, orient: str) -> tuple[int, int, int, int]:
    pixels = _bar_pixels(center, length, orient)
    rows = [r for r, _ in pixels]
    cols = [c for _, c in pixels]
    return min(rows), min(cols), max(rows), max(cols)


def _fits(shape: tuple[int, int], center: tuple[int, int], length: int) -> bool:
    h, w = shape
    half = length // 2
    r, c = center
    return half <= r < h - half and half <= c < w - half


def _find_bars(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    h, w = g.shape
    seen = np.zeros_like(g, dtype=bool)
    bars: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or g[r, c] == bg:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = sorted({rr for rr, _ in coords})
            cols = sorted({cc for _, cc in coords})
            assert len(rows) == 1 or len(cols) == 1, "each component must be a straight bar"
            orient = HORIZONTAL if len(rows) == 1 else VERTICAL
            length = len(coords)
            assert length >= 3 and length % 2 == 1, "bars must have odd length >= 3"

            if orient == HORIZONTAL:
                row = rows[0]
                expected = {(row, cc) for cc in range(cols[0], cols[-1] + 1)}
                assert expected == set(coords)
                center = (row, (cols[0] + cols[-1]) // 2)
            else:
                col = cols[0]
                expected = {(rr, col) for rr in range(rows[0], rows[-1] + 1)}
                assert expected == set(coords)
                center = ((rows[0] + rows[-1]) // 2, col)

            bars.append(
                {
                    "color": color,
                    "length": length,
                    "orient": orient,
                    "center": center,
                    "bbox": _bar_bbox(center, length, orient),
                }
            )

    assert bars, "expected at least one bar"
    return bars


def _rotate_orient(orient: str) -> str:
    return VERTICAL if orient == HORIZONTAL else HORIZONTAL


def _transform_bars(
    bars: list[dict],
    shape: tuple[int, int],
    rotate_colors: set[int],
) -> np.ndarray:
    h, w = shape
    out = np.full(shape, BLACK, dtype=int)
    seen: dict[tuple[int, int], int] = {}
    for bar in bars:
        orient = _rotate_orient(bar["orient"]) if bar["color"] in rotate_colors else bar["orient"]
        for cell in _bar_pixels(bar["center"], bar["length"], orient):
            r, c = cell
            assert 0 <= r < h and 0 <= c < w
            prev = seen.get(cell)
            assert prev is None, "transformed bars must not overlap"
            seen[cell] = int(bar["color"])
            out[r, c] = int(bar["color"])
    return out


def _apply_connectors(base: np.ndarray) -> np.ndarray:
    out = np.array(base, copy=True)
    h, w = base.shape
    colors = sorted(int(c) for c in np.unique(base) if c not in (BLACK, GREEN))
    for color in colors:
        mask = base == color
        for r in range(h):
            cols = np.flatnonzero(mask[r])
            if cols.size >= 2:
                c0, c1 = int(cols[0]), int(cols[-1])
                for c in range(c0, c1 + 1):
                    if base[r, c] == BLACK:
                        out[r, c] = GREEN
        for c in range(w):
            rows = np.flatnonzero(mask[:, c])
            if rows.size >= 2:
                r0, r1 = int(rows[0]), int(rows[-1])
                for r in range(r0, r1 + 1):
                    if base[r, c] == BLACK:
                        out[r, c] = GREEN
    return out


def _solve_with_colors(grid: np.ndarray, rotate_colors: set[int]) -> np.ndarray:
    bars = _find_bars(grid)
    base = _transform_bars(bars, grid.shape, rotate_colors=rotate_colors)
    return _apply_connectors(base)


def _try_solve_with_colors(grid: np.ndarray, rotate_colors: set[int]) -> np.ndarray | None:
    try:
        return _solve_with_colors(grid, rotate_colors)
    except Exception:
        return None


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate only the red bars, then draw green row/column spans between same-color bars that align after that rotation.

    Input structure:
    - The grid has a black background with several 4-connected one-cell-thick straight bars.
    - Foreground bars use exactly the puzzle colors blue and red, with odd lengths and either horizontal or vertical orientation.
    - Bars are separated from one another so each foreground component is a single bar.

    Rule:
    - Identify every bar and keep its center and length.
    - Rotate each red bar 90 degrees around its own center; blue bars stay fixed.
    - On the transformed canvas, for each color independently, fill every blank cell between the leftmost and rightmost same-color pixels in any occupied row with green.
    - Do the same for columns, filling every blank cell between the topmost and bottommost same-color pixels with green while preserving any existing bar pixels.

    Key insight:
    - The crucial step is color-conditional rotation: only red bars change orientation, and those rotations create the row/column alignments that determine where the later span fills appear.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _solve_with_colors(g, rotate_colors={RED})


def _line_bar_sets(bars: list[dict], shape: tuple[int, int], rotate_colors: set[int]) -> tuple[list[set[int]], list[set[int]]]:
    h, w = shape
    rows = [set() for _ in range(h)]
    cols = [set() for _ in range(w)]
    for idx, bar in enumerate(bars):
        orient = _rotate_orient(bar["orient"]) if bar["color"] in rotate_colors else bar["orient"]
        for r, c in _bar_pixels(bar["center"], bar["length"], orient):
            rows[r].add(idx)
            cols[c].add(idx)
    return rows, cols


def _has_cross_color_span(base: np.ndarray) -> bool:
    h, w = base.shape
    colors = [BLUE, RED]
    for color in colors:
        other = RED if color == BLUE else BLUE
        mask = base == color
        for r in range(h):
            cols = np.flatnonzero(mask[r])
            if cols.size >= 2:
                c0, c1 = int(cols[0]), int(cols[-1])
                if np.any(base[r, c0 : c1 + 1] == other):
                    return True
        for c in range(w):
            rows = np.flatnonzero(mask[:, c])
            if rows.size >= 2:
                r0, r1 = int(rows[0]), int(rows[-1])
                if np.any(base[r0 : r1 + 1, c] == other):
                    return True
    return False


def _render_input(shape: tuple[int, int], bars: list[dict]) -> np.ndarray:
    g = np.zeros(shape, dtype=int)
    for bar in bars:
        for r, c in _bar_pixels(bar["center"], bar["length"], bar["orient"]):
            assert g[r, c] == BLACK
            g[r, c] = int(bar["color"])
    return g


def _scene_geometry_ok(shape: tuple[int, int], bars: list[dict]) -> bool:
    try:
        _render_input(shape, bars)
        _transform_bars(bars, shape, rotate_colors={RED})
        return True
    except Exception:
        return False


def _validate_scene(shape: tuple[int, int], bars: list[dict]) -> bool:
    if sum(bar["color"] == RED for bar in bars) < 3 or sum(bar["color"] == BLUE for bar in bars) < 3:
        return False

    for i, bar_a in enumerate(bars):
        if not _fits(shape, bar_a["center"], bar_a["length"]):
            return False
        for bar_b in bars[i + 1 :]:
            if not _boxes_have_blank_gap(bar_a["bbox"], bar_b["bbox"]):
                return False

    try:
        inp = _render_input(shape, bars)
        parsed = _find_bars(inp)
        if len(parsed) != len(bars):
            return False
        base = _transform_bars(parsed, shape, rotate_colors={RED})
        out = _apply_connectors(base)
    except Exception:
        return False

    if not np.any(out == GREEN):
        return False

    pre_rows, pre_cols = _line_bar_sets(parsed, shape, rotate_colors=set())
    post_rows, post_cols = _line_bar_sets(parsed, shape, rotate_colors={RED})
    red_ids = {i for i, bar in enumerate(parsed) if bar["color"] == RED}
    blue_ids = {i for i, bar in enumerate(parsed) if bar["color"] == BLUE}

    red_new_rows = [
        r for r in range(shape[0])
        if len(post_rows[r] & red_ids) >= 2 and len(pre_rows[r] & red_ids) < 2
    ]
    red_new_cols = [
        c for c in range(shape[1])
        if len(post_cols[c] & red_ids) >= 2 and len(pre_cols[c] & red_ids) < 2
    ]
    blue_pre_aligned = any(len(pre_rows[r] & blue_ids) >= 2 for r in range(shape[0])) or any(
        len(pre_cols[c] & blue_ids) >= 2 for c in range(shape[1])
    )

    if not red_new_rows or not red_new_cols or not blue_pre_aligned:
        return False
    if not _has_cross_color_span(base):
        return False

    alt_none = _try_solve_with_colors(inp, rotate_colors=set())
    alt_blue = _try_solve_with_colors(inp, rotate_colors={BLUE})
    alt_both = _try_solve_with_colors(inp, rotate_colors={BLUE, RED})
    for alt in (alt_none, alt_blue, alt_both):
        if alt is not None and np.array_equal(out, alt):
            return False
    if np.array_equal(out, base):
        return False

    return True


def _try_add_bar(shape: tuple[int, int], bars: list[dict], color: int) -> bool:
    h, w = shape
    for _ in range(50):
        length = random.choice(ODD_LENGTHS)
        half = length // 2
        orient = random.choice((HORIZONTAL, VERTICAL))
        center = (
            random.randint(half, h - 1 - half),
            random.randint(half, w - 1 - half),
        )
        candidate = {
            "color": color,
            "length": length,
            "orient": orient,
            "center": center,
            "bbox": _bar_bbox(center, length, orient),
        }
        if all(_boxes_have_blank_gap(candidate["bbox"], bar["bbox"]) for bar in bars):
            bars.append(candidate)
            if _scene_geometry_ok(shape, bars):
                return True
            else:
                bars.pop()
    return False


def _mandatory_scene(shape: tuple[int, int]) -> list[dict] | None:
    h, w = shape
    def place_bar(
        bars: list[dict],
        *,
        color: int,
        orient: str,
        length: int,
        row_values: list[int],
        col_values: list[int],
    ) -> dict | None:
        candidates = [(r, c) for r in row_values for c in col_values]
        random.shuffle(candidates)
        for center in candidates:
            candidate = {
                "color": color,
                "length": length,
                "orient": orient,
                "center": center,
                "bbox": _bar_bbox(center, length, orient),
            }
            if not all(_boxes_have_blank_gap(candidate["bbox"], bar["bbox"]) for bar in bars):
                continue
            bars.append(candidate)
            if _scene_geometry_ok(shape, bars):
                return candidate
            bars.pop()
        return None

    bars: list[dict] = []
    red_h_lens = [random.choice((5, 7)), random.choice((5, 7))]
    red_v_lens = [random.choice((5, 7)), random.choice((5, 7))]
    blue_lens = [random.choice((3, 5)), random.choice((3, 5))]

    blue_row = random.randint(h // 2, h - 4)
    rv1_len = red_v_lens[0]
    rv2_len = red_v_lens[1]
    rv1_half = rv1_len // 2
    rv2_half = rv2_len // 2

    v1 = place_bar(
        bars,
        color=RED,
        orient=VERTICAL,
        length=rv1_len,
        row_values=[blue_row],
        col_values=list(range(max(rv1_half, w // 2 - 2), min(w - rv1_half, w // 2 + 3))),
    )
    if v1 is None:
        return None

    c_v1 = v1["center"][1]
    rv2_row_values = [
        r for r in (blue_row - 3, blue_row - 2, blue_row + 2, blue_row + 3)
        if rv2_half <= r < h - rv2_half
    ]
    rv2_col_values = [
        c for c in (c_v1 - 3, c_v1 - 2, c_v1 + 2, c_v1 + 3)
        if rv2_half <= c < w - rv2_half
    ]
    if not rv2_row_values or not rv2_col_values:
        return None
    if place_bar(
        bars,
        color=RED,
        orient=VERTICAL,
        length=rv2_len,
        row_values=rv2_row_values,
        col_values=rv2_col_values,
    ) is None:
        return None

    bh1_len, bh2_len = blue_lens
    bh1_half = bh1_len // 2
    bh2_half = bh2_len // 2
    left_cols = list(range(bh1_half, max(bh1_half + 1, c_v1 - bh1_half - 2)))
    right_cols = list(range(min(w - bh2_half - 1, c_v1 + bh2_half + 3), w - bh2_half))
    if not left_cols or not right_cols:
        return None
    if place_bar(
        bars,
        color=BLUE,
        orient=HORIZONTAL,
        length=bh1_len,
        row_values=[blue_row],
        col_values=left_cols,
    ) is None:
        return None
    if place_bar(
        bars,
        color=BLUE,
        orient=HORIZONTAL,
        length=bh2_len,
        row_values=[blue_row],
        col_values=right_cols,
    ) is None:
        return None

    rh1_len, rh2_len = red_h_lens
    rh1_half = rh1_len // 2
    rh2_half = rh2_len // 2
    top_rows = list(range(2, max(3, blue_row - 3)))
    if len(top_rows) < 4:
        return None
    row_pairs = []
    for r1 in top_rows:
        for r2 in top_rows:
            if r2 - r1 in (2, 3):
                row_pairs.append((r1, r2))
    random.shuffle(row_pairs)
    placed_h = False
    for r1, r2 in row_pairs:
        snapshot = [dict(bar) for bar in bars]
        left_cols = list(range(rh1_half, max(rh1_half + 1, w // 3 + 1)))
        right_cols = list(range(max(rh2_half, w // 2 + 1), w - rh2_half))
        h1 = place_bar(
            bars,
            color=RED,
            orient=HORIZONTAL,
            length=rh1_len,
            row_values=[r1],
            col_values=left_cols,
        )
        if h1 is None:
            bars = snapshot
            continue
        h2 = place_bar(
            bars,
            color=RED,
            orient=HORIZONTAL,
            length=rh2_len,
            row_values=[r2],
            col_values=right_cols,
        )
        if h2 is not None:
            placed_h = True
            break
        bars = snapshot
    if not placed_h:
        return None

    if not _try_add_bar(shape, bars, BLUE):
        return None

    for _ in range(random.randint(0, 2)):
        _try_add_bar(shape, bars, random.choice((BLUE, RED)))

    return bars


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        shape = (
            random.randint(MIN_SIZE, MAX_SIZE),
            random.randint(MIN_SIZE, MAX_SIZE),
        )
        scene = _mandatory_scene(shape)
        if scene is None:
            continue
        if _validate_scene(shape, scene):
            return _render_input(shape, scene)
    raise RuntimeError("failed to generate a valid conditional-rotation scene")
