import hashlib
import random

import numpy as np


SIZE_RANGE = (10, 18)
VERTICAL_COUNT_RANGE = (1, 3)
HORIZONTAL_COUNT_RANGE = (1, 3)
POSITION_GAP = 3
MAX_GENERATION_ATTEMPTS = 400
PALETTE = tuple(range(1, 10))
MODE_CYCLE = ("mixed", "same_only", "diff_only")

_GEN_CALL_COUNT = 0
_SEEN_OUTPUT_SIGNATURES: set[str] = set()


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _signature(grid: np.ndarray) -> str:
    g = np.asarray(grid, dtype=int)
    return hashlib.md5(g.tobytes() + str(g.shape).encode()).hexdigest()


def _pick_positions(length: int, count: int) -> list[int] | None:
    choices = list(range(1, length - 1))
    random.shuffle(choices)
    picked: list[int] = []
    for pos in choices:
        if all(abs(pos - prev) >= POSITION_GAP for prev in picked):
            picked.append(pos)
            if len(picked) == count:
                return sorted(picked)
    return None


def _collect_lines(grid: np.ndarray) -> tuple[int, dict[int, int], dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    h, w = g.shape
    verticals: dict[int, int] = {}
    horizontals: dict[int, int] = {}

    for r, c in np.argwhere(g != bg):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        assert r in {0, h - 1} or c in {0, w - 1}, "all markers must lie on the boundary"
        assert not (r in {0, h - 1} and c in {0, w - 1}), "corner markers are undefined"
        if r in {0, h - 1}:
            assert 0 < c < w - 1, "vertical markers must not use corners"
            if c in verticals:
                assert verticals[c] == color, "a column cannot have conflicting marker colors"
            verticals[c] = color
        else:
            assert 0 < r < h - 1, "horizontal markers must not use corners"
            if r in horizontals:
                assert horizontals[r] == color, "a row cannot have conflicting marker colors"
            horizontals[r] = color

    assert verticals and horizontals, "expected at least one vertical and one horizontal line"
    return bg, verticals, horizontals


def _draw_base_lines(shape: tuple[int, int], bg: int, verticals: dict[int, int], horizontals: dict[int, int]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for c, color in verticals.items():
        out[:, int(c)] = int(color)
    for r, color in horizontals.items():
        out[int(r), :] = int(color)
    return out


def _apply_crossings(
    grid: np.ndarray,
    bg: int,
    verticals: dict[int, int],
    horizontals: dict[int, int],
    *,
    apply_same: bool = True,
    apply_diff: bool = True,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    h, w = out.shape
    for r, h_color in horizontals.items():
        assert 0 < r < h - 1
        for c, v_color in verticals.items():
            assert 0 < c < w - 1
            if h_color == v_color:
                if apply_same:
                    out[r - 1 : r + 2, c - 1 : c + 2] = h_color
            else:
                if apply_diff:
                    out[r, c] = bg
                    out[r - 1, c] = bg
                    out[r + 1, c] = bg
                    out[r, c - 1] = bg
                    out[r, c + 1] = bg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Extend every boundary marker into a full line, then turn crossings into blooms or blast holes.

    Input structure:
    - The background dominates the grid, and every foreground pixel lies on the boundary.
    - Markers on the top and bottom edges come in same-color column pairs; each pair marks one full vertical line.
    - Markers on the left and right edges come in same-color row pairs; each pair marks one full horizontal line.

    Rule:
    - Draw every marked column and row across the whole grid in the marker color.
    - At each row/column crossing, compare the horizontal and vertical line colors.
    - If the colors match, replace the 3x3 neighborhood centered on the crossing with a solid block of that color.
    - If the colors differ, clear the crossing cell and its four orthogonal neighbors back to background.

    Key insight:
    - The result is determined by interactions between full-span boundary-induced lines, not by the isolated markers themselves.
    """
    g = np.asarray(grid, dtype=int)
    bg, verticals, horizontals = _collect_lines(g)
    base = _draw_base_lines(g.shape, bg, verticals, horizontals)
    return _apply_crossings(base, bg, verticals, horizontals)


def _choose_mode() -> str:
    global _GEN_CALL_COUNT
    mode = MODE_CYCLE[_GEN_CALL_COUNT % len(MODE_CYCLE)]
    _GEN_CALL_COUNT += 1
    return mode


def _sample_line_counts(mode: str) -> tuple[int, int]:
    if mode == "mixed":
        return random.randint(2, 3), random.randint(2, 3)
    return random.randint(*VERTICAL_COUNT_RANGE), random.randint(*HORIZONTAL_COUNT_RANGE)


def _assign_colors(mode: str, v_count: int, h_count: int) -> tuple[list[int], list[int]]:
    if mode == "same_only":
        color = random.choice(PALETTE)
        return [color] * v_count, [color] * h_count

    if mode == "diff_only":
        pool = random.sample(PALETTE, random.randint(2, min(5, len(PALETTE))))
        split = random.randint(1, len(pool) - 1)
        random.shuffle(pool)
        v_palette = pool[:split]
        h_palette = pool[split:]
        if not v_palette or not h_palette:
            return _assign_colors(mode, v_count, h_count)
        return [random.choice(v_palette) for _ in range(v_count)], [random.choice(h_palette) for _ in range(h_count)]

    shared = random.choice(PALETTE)
    extra_choices = [c for c in PALETTE if c != shared]
    extras = random.sample(extra_choices, random.randint(1, min(3, len(extra_choices))))
    v_colors = [shared]
    h_colors = [shared]
    while len(v_colors) < v_count:
        v_colors.append(random.choice(extras))
    while len(h_colors) < h_count:
        h_colors.append(random.choice(extras))
    random.shuffle(v_colors)
    random.shuffle(h_colors)
    return v_colors, h_colors


def _build_input(height: int, width: int, rows: list[int], cols: list[int], row_colors: list[int], col_colors: list[int]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for c, color in zip(cols, col_colors):
        grid[0, c] = color
        grid[height - 1, c] = color
    for r, color in zip(rows, row_colors):
        grid[r, 0] = color
        grid[r, width - 1] = color
    return grid


def _crossing_counts(verticals: dict[int, int], horizontals: dict[int, int]) -> tuple[int, int]:
    same = 0
    diff = 0
    for h_color in horizontals.values():
        for v_color in verticals.values():
            if h_color == v_color:
                same += 1
            else:
                diff += 1
    return same, diff


def _validate_semantics(grid: np.ndarray, mode: str) -> bool:
    bg, verticals, horizontals = _collect_lines(grid)
    rows = sorted(horizontals)
    cols = sorted(verticals)
    if any(b - a < POSITION_GAP for a, b in zip(rows, rows[1:])):
        return False
    if any(b - a < POSITION_GAP for a, b in zip(cols, cols[1:])):
        return False

    same_count, diff_count = _crossing_counts(verticals, horizontals)
    if mode == "same_only" and diff_count != 0:
        return False
    if mode == "diff_only" and same_count != 0:
        return False
    if mode == "mixed" and (same_count == 0 or diff_count == 0):
        return False

    base = _draw_base_lines(grid.shape, bg, verticals, horizontals)
    out = main(grid)
    if np.array_equal(out, base):
        return False

    same_only = _apply_crossings(base, bg, verticals, horizontals, apply_same=True, apply_diff=False)
    diff_only = _apply_crossings(base, bg, verticals, horizontals, apply_same=False, apply_diff=True)
    if same_count > 0:
        if np.array_equal(out, diff_only):
            return False
        for r, h_color in horizontals.items():
            for c, v_color in verticals.items():
                if h_color == v_color and not np.all(out[r - 1 : r + 2, c - 1 : c + 2] == h_color):
                    return False
    if diff_count > 0:
        if np.array_equal(out, same_only):
            return False
        for r, h_color in horizontals.items():
            for c, v_color in verticals.items():
                if h_color != v_color:
                    cross = {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
                    if any(out[rr, cc] != bg for rr, cc in cross):
                        return False

    if mode == "mixed":
        if np.array_equal(out, same_only) or np.array_equal(out, diff_only):
            return False

    if _signature(out) in _SEEN_OUTPUT_SIGNATURES:
        return False
    return True


def _gen_attempt(mode: str) -> np.ndarray | None:
    height = random.randint(*SIZE_RANGE)
    width = random.randint(*SIZE_RANGE)
    v_count, h_count = _sample_line_counts(mode)
    rows = _pick_positions(height, h_count)
    cols = _pick_positions(width, v_count)
    if rows is None or cols is None:
        return None

    col_colors, row_colors = _assign_colors(mode, v_count, h_count)
    grid = _build_input(height, width, rows, cols, row_colors, col_colors)
    if not _validate_semantics(grid, mode):
        return None
    return grid


def generate_input() -> np.ndarray:
    mode = _choose_mode()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(mode)
        if grid is None:
            continue
        _SEEN_OUTPUT_SIGNATURES.add(_signature(main(grid)))
        return grid
    raise RuntimeError(f"failed to generate a valid input for mode={mode}")
