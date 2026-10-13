from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
TEAL = 8

LEFT_MAX_COL = 7
SEP_COL = 8
RIGHT_MIN_COL = 9
SQUARE = 5
BAR_HEIGHT_MAX = 6

GEN_RETRIES = 600


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _scan_bars(grid: np.ndarray) -> list[dict[str, int]]:
    h, _w = grid.shape
    bars: list[dict[str, int]] = []
    for c in range(LEFT_MAX_COL + 1):
        col = grid[:, c]
        nz = np.where(col != BG)[0]
        if nz.size == 0:
            continue
        color = int(col[nz[0]])
        assert color in (TEAL, RED), "left zone may only contain teal/red bars"
        r0 = int(nz[0])
        assert bool(np.all(col[r0:] == color)), "bar must be contiguous and bottom-aligned"
        height = h - r0
        assert 1 <= height <= BAR_HEIGHT_MAX
        bars.append({"color": color, "col": c, "height": height})
    assert 4 <= len(bars) <= 5, "expected 4-5 operand bars"
    assert any(bar["color"] == TEAL for bar in bars)
    assert any(bar["color"] == RED for bar in bars)
    return bars


def _scan_squares(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    squares: list[dict[str, object]] = []
    for r0 in range(h - SQUARE + 1):
        for c0 in range(RIGHT_MIN_COL, w - SQUARE + 1):
            patch = grid[r0 : r0 + SQUARE, c0 : c0 + SQUARE]
            border = np.concatenate(
                [patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]]
            )
            if int(border.min()) != int(border.max()):
                continue
            frame = int(border[0])
            if frame not in (BLUE, GREEN):
                continue
            if not bool(np.all(patch[1:-1, 1:-1] == YELLOW)):
                continue
            squares.append(
                {
                    "bbox": (int(r0), int(c0), int(r0 + SQUARE - 1), int(c0 + SQUARE - 1)),
                    "frame": frame,
                    "patch": patch.copy(),
                }
            )
    assert 3 <= len(squares) <= 4, "expected 3-4 bordered squares"
    return squares


def _n_from_bars(bars: list[dict[str, int]]) -> int:
    total = 0
    for bar in bars:
        total += int(bar["height"]) if int(bar["color"]) == TEAL else -int(bar["height"])
    assert total != 0 and abs(total) in (1, 2)
    return int(total)


def _sort_squares(
    squares: list[dict[str, object]], *, target_left: bool, positive: bool
) -> list[dict[str, object]]:
    def key(square: dict[str, object]) -> tuple[int, int, int]:
        r0, c0, _r1, _c1 = square["bbox"]  # type: ignore[misc]
        dist = int(c0 - RIGHT_MIN_COL) if target_left else int(r0)
        return dist, int(r0), int(c0)

    ordered = sorted(squares, key=key)
    if not positive:
        ordered.reverse()
    return ordered


def _render_with_modes(
    grid: np.ndarray, *, target_left: bool, positive: bool
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    squares = _scan_squares(g)

    out = g.copy()
    for square in squares:
        r0, c0, r1, c1 = square["bbox"]  # type: ignore[misc]
        out[r0 : r1 + 1, c0 : c1 + 1] = BG

    ordered = _sort_squares(squares, target_left=target_left, positive=positive)
    if target_left:
        for i, square in enumerate(ordered):
            dest_r = i * SQUARE
            dest_c = RIGHT_MIN_COL
            assert dest_r + SQUARE <= h
            _paste(out, square["patch"], dest_r, dest_c)  # type: ignore[arg-type]
    else:
        for i, square in enumerate(ordered):
            dest_r = 0
            dest_c = RIGHT_MIN_COL + i * SQUARE
            assert dest_c + SQUARE <= w
            _paste(out, square["patch"], dest_r, dest_c)  # type: ignore[arg-type]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Use the signed teal-minus-red bar total to choose how the bordered squares are reordered onto an edge.

    Input structure:
    - A black background grid is split by a full-height grey separator at column 8.
    - In the left zone, 4-5 bottom-aligned one-cell-wide bars use teal for positive height and red for negative height.
    - In the right zone, 3-4 scattered 5x5 bordered squares have blue or green frames and solid yellow 3x3 interiors.

    Rule:
    - Compute N = total teal-bar height minus total red-bar height.
    - If |N| = 1, the target is the right zone's left edge (column 9); if |N| = 2, the target is the top edge (row 0).
    - If N > 0, sort the squares by distance to that target edge from near to far; if N < 0, sort from far to near.
    - Distance is left-column minus 9 for the left-edge case, or top row for the top-edge case.
    - Erase the original squares and place them in sorted order flush against the chosen edge, stacked along that edge with no gaps; keep the bars and separator unchanged.

    Key insight:
    - The same signed arithmetic result controls two independent decisions at once: its magnitude selects the destination edge, and its sign selects the distance-order direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert g.shape[1] >= RIGHT_MIN_COL + SQUARE
    assert bool(np.all(g[:, SEP_COL] == GREY)), "expected a full-height grey separator"

    bars = _scan_bars(g)
    n = _n_from_bars(bars)
    return _render_with_modes(g, target_left=(abs(n) == 1), positive=(n > 0))


def _bounded_parts(total: int, parts: int, upper: int) -> list[int] | None:
    if parts == 0:
        return [] if total == 0 else None
    if not (parts <= total <= parts * upper):
        return None
    values: list[int] = []
    remaining = total
    for i in range(parts):
        slots = parts - i - 1
        lo = max(1, remaining - slots * upper)
        hi = min(upper, remaining - slots)
        if lo > hi:
            return None
        value = random.randint(lo, hi)
        values.append(value)
        remaining -= value
    random.shuffle(values)
    return values


def _sample_bars(target_n: int) -> list[tuple[int, int]]:
    for _ in range(300):
        bar_count = random.randint(4, 5)
        teal_count = random.randint(1, bar_count - 1)
        red_count = bar_count - teal_count
        if target_n > 0:
            red_lo = red_count
            red_hi = min(6 * red_count, 6 * teal_count - target_n)
            if red_lo > red_hi:
                continue
            red_sum = random.randint(red_lo, red_hi)
            teal_sum = red_sum + target_n
        else:
            teal_lo = teal_count
            teal_hi = min(6 * teal_count, 6 * red_count + target_n)
            if teal_lo > teal_hi:
                continue
            teal_sum = random.randint(teal_lo, teal_hi)
            red_sum = teal_sum - target_n

        teal_heights = _bounded_parts(teal_sum, teal_count, BAR_HEIGHT_MAX)
        red_heights = _bounded_parts(red_sum, red_count, BAR_HEIGHT_MAX)
        if teal_heights is None or red_heights is None:
            continue
        bars = [(TEAL, h) for h in teal_heights] + [(RED, h) for h in red_heights]
        random.shuffle(bars)
        return bars
    raise RuntimeError("failed to sample operand bars")


def _rect_clear(
    placed: list[tuple[int, int]], r0: int, c0: int, *, gap: int = 2
) -> bool:
    for pr, pc in placed:
        if not (r0 + SQUARE - 1 + gap < pr or pr + SQUARE - 1 + gap < r0 or c0 + SQUARE - 1 + gap < pc or pc + SQUARE - 1 + gap < c0):
            return False
    return True


def _sample_sorted_color_sequence(n_squares: int) -> list[int]:
    choices = [BLUE, GREEN]
    for _ in range(200):
        seq = [random.choice(choices) for _ in range(n_squares)]
        if len(set(seq)) < 2:
            continue
        if seq == list(reversed(seq)):
            continue
        return seq
    raise RuntimeError("failed to sample a non-palindromic color order")


def _make_patch(frame: int) -> np.ndarray:
    patch = np.full((SQUARE, SQUARE), frame, dtype=int)
    patch[1:-1, 1:-1] = YELLOW
    return patch


def _sample_input_positions(
    h: int,
    w: int,
    n_squares: int,
    *,
    target_left: bool,
    positive: bool,
) -> list[tuple[int, int]]:
    if target_left:
        dist_candidates = list(range(0, w - RIGHT_MIN_COL - SQUARE + 1))
        random.shuffle(dist_candidates)
        chosen = sorted(dist_candidates[:n_squares])
        c0s = chosen if positive else list(reversed(chosen))
        c0s = [RIGHT_MIN_COL + dc for dc in c0s]
        for _ in range(300):
            rows = random.sample(range(0, h - SQUARE + 1), n_squares)
            placed: list[tuple[int, int]] = []
            ok = True
            result: list[tuple[int, int]] = []
            for r0, c0 in zip(rows, c0s):
                if not _rect_clear(placed, r0, c0):
                    ok = False
                    break
                placed.append((r0, c0))
                result.append((r0, c0))
            if ok:
                return result
    else:
        dist_candidates = list(range(0, h - SQUARE + 1))
        random.shuffle(dist_candidates)
        chosen = sorted(dist_candidates[:n_squares])
        r0s = chosen if positive else list(reversed(chosen))
        for _ in range(300):
            cols = random.sample(range(RIGHT_MIN_COL, w - SQUARE + 1), n_squares)
            placed: list[tuple[int, int]] = []
            ok = True
            result: list[tuple[int, int]] = []
            for r0, c0 in zip(r0s, cols):
                if not _rect_clear(placed, r0, c0):
                    ok = False
                    break
                placed.append((r0, c0))
                result.append((r0, c0))
            if ok:
                return result
    raise RuntimeError("failed to place input squares")


def _build_case(
    h: int, w: int, n_value: int, n_squares: int, sorted_frames: list[int]
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    grid[:, SEP_COL] = GREY

    bars = _sample_bars(n_value)
    bar_cols = sorted(random.sample(range(LEFT_MAX_COL + 1), len(bars)))
    for (color, bar_h), col in zip(bars, bar_cols):
        grid[h - bar_h : h, col] = color

    target_left = abs(n_value) == 1
    positive = n_value > 0
    input_positions = _sample_input_positions(
        h, w, n_squares, target_left=target_left, positive=positive
    )

    patches = [_make_patch(frame) for frame in sorted_frames]

    for (r0, c0), patch in zip(input_positions, patches):
        _paste(grid, patch, r0, c0)

    return grid


def _wrong_order_output(grid: np.ndarray) -> np.ndarray:
    squares = _scan_squares(grid)
    out = grid.copy()
    for square in squares:
        r0, c0, r1, c1 = square["bbox"]  # type: ignore[misc]
        out[r0 : r1 + 1, c0 : c1 + 1] = BG
    for i, square in enumerate(sorted(squares, key=lambda s: tuple(int(x) for x in s["bbox"]))):
        dest_r = i * SQUARE
        dest_c = RIGHT_MIN_COL
        if dest_r + SQUARE <= grid.shape[0]:
            _paste(out, square["patch"], dest_r, dest_c)  # type: ignore[arg-type]
    return out


def _case_is_non_degenerate(grid: np.ndarray) -> bool:
    bars = _scan_bars(grid)
    n = _n_from_bars(bars)
    squares = _scan_squares(grid)
    dists: list[int] = []
    for square in squares:
        r0, c0, _r1, _c1 = square["bbox"]  # type: ignore[misc]
        dists.append(int(c0 - RIGHT_MIN_COL) if abs(n) == 1 else int(r0))
    if len(set(dists)) != len(dists):
        return False

    true_out = main(grid)
    if np.array_equal(true_out, grid):
        return False

    alt_abs_out = _render_with_modes(grid, target_left=(abs(n) != 1), positive=(n > 0))
    alt_sign_out = _render_with_modes(grid, target_left=(abs(n) == 1), positive=(n < 0))
    if np.array_equal(true_out, alt_abs_out):
        return False
    if np.array_equal(true_out, alt_sign_out):
        return False

    naive_out = _wrong_order_output(grid)
    if np.array_equal(true_out, naive_out):
        return False

    frames = [int(square["frame"]) for square in squares]
    if len(set(frames)) < 2:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_value = random.choice([-2, -1, 1, 2])
    n_squares = random.randint(3, 4)
    h = random.randint(max(20, n_squares * SQUARE), 26)
    w = random.randint(max(24, RIGHT_MIN_COL + n_squares * SQUARE), 30)
    sorted_frames = _sample_sorted_color_sequence(n_squares)

    try:
        grid = _build_case(h, w, n_value, n_squares, sorted_frames)
    except Exception:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None
    if not _case_is_non_degenerate(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
