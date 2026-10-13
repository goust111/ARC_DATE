import random
from collections import deque

import numpy as np


BLACK = 0
GREEN = 3
YELLOW = 4
GRAY = 5
PINK = 6
ORANGE = 7

MAX_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize_bar(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    pts = sorted((int(r), int(c)) for r, c in comp)
    assert len(pts) == 2, "green components must be size-2 bars"
    (r0, c0), (r1, c1) = pts
    assert (r0 == r1 and c1 == c0 + 1) or (c0 == c1 and r1 == r0 + 1), "green components must be straight micro-bars"
    return pts[0], pts[1]


def _collect_micro_bars(grid: np.ndarray) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    return {_normalize_bar(comp) for comp in _components_of_color(grid, GREEN)}


def _has_left_bar(bars, r: int, c: int) -> bool:
    for offset in (2, 3):
        bar = tuple(sorted(((r, c - offset - 1), (r, c - offset))))
        if bar in bars:
            return True
    return False


def _has_right_bar(bars, r: int, c: int) -> bool:
    for offset in (2, 3):
        bar = tuple(sorted(((r, c + offset), (r, c + offset + 1))))
        if bar in bars:
            return True
    return False


def _has_up_bar(bars, r: int, c: int) -> bool:
    for offset in (2, 3):
        bar = tuple(sorted(((r - offset - 1, c), (r - offset, c))))
        if bar in bars:
            return True
    return False


def _has_down_bar(bars, r: int, c: int) -> bool:
    for offset in (2, 3):
        bar = tuple(sorted(((r + offset, c), (r + offset + 1, c))))
        if bar in bars:
            return True
    return False


def _find_hidden_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    bars = _collect_micro_bars(grid)
    centers: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) == GREEN:
                continue
            if _has_left_bar(bars, r, c) and _has_right_bar(bars, r, c) and _has_up_bar(bars, r, c) and _has_down_bar(bars, r, c):
                centers.append((r, c))
    return centers


def _infer_row_span(row: np.ndarray) -> tuple[int, int]:
    gray_cols = np.flatnonzero(row == GRAY)
    assert len(gray_cols) > 0, "every row must retain visible gray cells"
    left = int(gray_cols[0])
    right = int(gray_cols[-1])
    pink_cols = np.flatnonzero(row == PINK)
    orange_cols = np.flatnonzero(row == ORANGE)
    assert len(pink_cols) > 0 and len(orange_cols) > 0, "every row must retain visible side colors"
    assert int(pink_cols[-1]) < left, "visible pink pixels must stay left of the gray run"
    assert int(orange_cols[0]) > right, "visible orange pixels must stay right of the gray run"
    assert not np.any(row[:left] == ORANGE), "orange cannot appear before the gray run"
    assert not np.any(row[right + 1 :] == PINK), "pink cannot appear after the gray run"
    return left, right


def _render_aligned_rows(shape: tuple[int, int], widths: list[int], target_col: int) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), ORANGE, dtype=int)
    for r, width in enumerate(widths):
        start = max(0, int(target_col) - int(width) + 1)
        end = start + int(width) - 1
        out[r, :start] = PINK
        out[r, start : end + 1] = GRAY
    return out


def _overlay_bars(grid: np.ndarray, bars: list[tuple[tuple[int, int], tuple[int, int]]]) -> np.ndarray:
    out = grid.copy()
    for (r0, c0), (r1, c1) in bars:
        out[r0, c0] = GREEN
        out[r1, c1] = GREEN
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Detect a hidden cross of green micro-bars, mark its center, and align every gray row segment to that center column.

    Input structure:
    - The grid is built from row-wise tricolor bands: pink on the left, one contiguous gray segment in the middle, and orange on the right.
    - Green size-2 micro-bars are overlaid on top of those rows; each micro-bar is exactly horizontal or vertical.
    - Exactly one non-green cell is the hidden cross center implied by two horizontal bars and two vertical bars placed 2 or 3 cells away in the four cardinal directions.

    Rule:
    - Find the unique hidden-cross center from the arrangement of green size-2 bars and repaint that center yellow.
    - Use the center column as the target alignment column T.
    - For each row, keep the gray segment width unchanged and slide the segment so its right edge reaches T when possible; if that would push the left edge past column 0, clamp the segment to start at column 0.
    - Refill cells left of the shifted gray segment with pink and cells right of it with orange, then restore every green micro-bar in place and keep the yellow center marker on top.

    Key insight:
    - The alignment target is not encoded by the row segments themselves; it is hidden in a separate micro-bar system whose implied cross center supplies the common anchor column.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"

    centers = _find_hidden_centers(grid)
    assert len(centers) == 1, "expected exactly one hidden cross center"
    center_r, center_c = centers[0]

    spans = [_infer_row_span(grid[r]) for r in range(grid.shape[0])]
    widths = [right - left + 1 for left, right in spans]

    out = _render_aligned_rows(grid.shape, widths, center_c)
    out = _overlay_bars(out, sorted(_collect_micro_bars(grid)))
    out[center_r, center_c] = YELLOW
    return out


def _cross_bars(center_r: int, center_c: int, offsets: dict[str, int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return [
        tuple(sorted(((center_r, center_c - offsets["left"] - 1), (center_r, center_c - offsets["left"])))),
        tuple(sorted(((center_r, center_c + offsets["right"]), (center_r, center_c + offsets["right"] + 1)))),
        tuple(sorted(((center_r - offsets["up"] - 1, center_c), (center_r - offsets["up"], center_c)))),
        tuple(sorted(((center_r + offsets["down"], center_c), (center_r + offsets["down"] + 1, center_c)))),
    ]


def _bar_cells(bars: list[tuple[tuple[int, int], tuple[int, int]]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in bars:
        cells.add(a)
        cells.add(b)
    return cells


def _bar_touches_existing(existing: set[tuple[int, int]], new_bar: tuple[tuple[int, int], tuple[int, int]]) -> bool:
    for r, c in new_bar:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in existing:
                    return True
    return False


def _sample_row_spec(
    width: int,
    target_col: int,
    blocked_cols: set[int],
    relation: str | None = None,
    force_clipped: bool = False,
) -> tuple[int, int] | None:
    min_run = 3
    max_run = width - 6
    if max_run < min_run:
        return None
    for _ in range(200):
        if force_clipped:
            if target_col + 2 > max_run:
                return None
            run_w = random.randint(target_col + 2, max_run)
        else:
            run_w = random.randint(min_run, max_run)
        lo = 3
        hi = width - run_w - 3
        if relation == "magenta":
            lo = max(lo, target_col + 1)
        elif relation == "gray":
            lo = max(lo, target_col - run_w + 1)
            hi = min(hi, target_col)
        elif relation == "orange":
            hi = min(hi, target_col - run_w)
        if lo > hi:
            continue
        start = random.randint(lo, hi)
        end = start + run_w - 1
        if start in blocked_cols or end in blocked_cols:
            continue
        return start, end
    return None


def _render_input(rows: list[tuple[int, int]], shape: tuple[int, int], bars: list[tuple[tuple[int, int], tuple[int, int]]]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), ORANGE, dtype=int)
    for r, (start, end) in enumerate(rows):
        grid[r, :start] = PINK
        grid[r, start : end + 1] = GRAY
    return _overlay_bars(grid, bars)


def _expected_output(
    rows: list[tuple[int, int]],
    shape: tuple[int, int],
    bars: list[tuple[tuple[int, int], tuple[int, int]]],
    center: tuple[int, int],
    target_col: int,
) -> np.ndarray:
    widths = [end - start + 1 for start, end in rows]
    out = _render_aligned_rows(shape, widths, target_col)
    out = _overlay_bars(out, bars)
    out[center] = YELLOW
    return out


def _visible_row_colors_ok(grid: np.ndarray) -> bool:
    for r in range(grid.shape[0]):
        row = grid[r]
        if not np.any(row == PINK) or not np.any(row == GRAY) or not np.any(row == ORANGE):
            return False
        try:
            _infer_row_span(row)
        except AssertionError:
            return False
    return True


def _choose_rows(
    height: int,
    width: int,
    center_r: int,
    target_col: int,
    cross_bars: list[tuple[tuple[int, int], tuple[int, int]]],
) -> list[tuple[int, int]] | None:
    blocked_by_row: dict[int, set[int]] = {}
    for a, b in cross_bars:
        for r, c in (a, b):
            blocked_by_row.setdefault(r, set()).add(c)

    center_relation = random.choice(["magenta", "gray", "orange"])
    clipped_row = None
    if target_col + 2 <= width - 6 and random.random() < 0.35:
        choices = [r for r in range(height) if r != center_r]
        if choices:
            clipped_row = random.choice(choices)

    rows: list[tuple[int, int]] = []
    for r in range(height):
        relation = center_relation if r == center_r else None
        force_clipped = r == clipped_row
        spec = _sample_row_spec(width, target_col, blocked_by_row.get(r, set()), relation=relation, force_clipped=force_clipped)
        if spec is None:
            return None
        rows.append(spec)

    rights = [end for _, end in rows]
    if len(set(rights)) < 3:
        return None
    if not any(end < target_col for _, end in rows):
        return None
    if not any(end > target_col for _, end in rows):
        return None
    return rows


def _random_distractor_bar(
    height: int,
    width: int,
    rows: list[tuple[int, int]],
    existing_bars: list[tuple[tuple[int, int], tuple[int, int]]],
    center: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    center_r, center_c = center
    existing_cells = _bar_cells(existing_bars)
    for _ in range(250):
        horizontal = random.choice([False, True])
        if horizontal:
            r = random.randint(0, height - 1)
            c = random.randint(0, width - 2)
            bar = tuple(sorted(((r, c), (r, c + 1))))
        else:
            r = random.randint(0, height - 2)
            c = random.randint(0, width - 1)
            bar = tuple(sorted(((r, c), (r + 1, c))))
        if center in bar:
            continue
        if _bar_touches_existing(existing_cells, bar):
            continue
        ok = True
        for rr, cc in bar:
            start, end = rows[rr]
            if cc in (start, end):
                ok = False
                break
        if not ok:
            continue
        if horizontal and bar[0][0] == center_r and abs(bar[0][1] - center_c) <= 4:
            continue
        if (not horizontal) and bar[0][1] == center_c and abs(bar[0][0] - center_r) <= 4:
            continue
        return bar
    return None


def _wrong_targets(target_col: int, rows: list[tuple[int, int]]) -> list[int]:
    rights = sorted(end for _, end in rows)
    guesses = {rights[0], rights[-1], rights[len(rights) // 2], int(round(sum(rights) / len(rights)))}
    return sorted(g for g in guesses if 0 <= g and g != target_col)


def _semantic_checks(
    grid: np.ndarray,
    rows: list[tuple[int, int]],
    bars: list[tuple[tuple[int, int], tuple[int, int]]],
    center: tuple[int, int],
    target_col: int,
) -> bool:
    centers = _find_hidden_centers(grid)
    if centers != [center]:
        return False
    if len(_collect_micro_bars(grid)) != len(bars):
        return False
    inferred = []
    for r in range(grid.shape[0]):
        try:
            inferred.append(_infer_row_span(grid[r]))
        except AssertionError:
            return False
    if inferred != rows:
        return False
    expected = _expected_output(rows, grid.shape, bars, center, target_col)
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    for wrong_target in _wrong_targets(target_col, rows):
        wrong = _expected_output(rows, grid.shape, bars, center, wrong_target)
        if np.array_equal(wrong, expected):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(11, 16)
    width = random.randint(13, 19)
    center_r = random.randint(4, height - 5)
    target_col = random.randint(4, width - 5)
    offsets = {key: random.choice([2, 3]) for key in ("left", "right", "up", "down")}
    cross_bars = _cross_bars(center_r, target_col, offsets)

    rows = _choose_rows(height, width, center_r, target_col, cross_bars)
    if rows is None:
        return None

    bars = list(cross_bars)
    distractor_count = random.choice([1, 2])
    for _ in range(distractor_count):
        distractor = _random_distractor_bar(height, width, rows, bars, (center_r, target_col))
        if distractor is None:
            return None
        bars.append(distractor)

    bars = sorted(set(bars))
    grid = _render_input(rows, (height, width), bars)
    if not _visible_row_colors_ok(grid):
        return None
    if not _semantic_checks(grid, rows, bars, (center_r, target_col), target_col):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-cross alignment input")
