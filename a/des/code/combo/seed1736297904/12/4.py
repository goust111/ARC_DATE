import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

GEN_RETRIES = 300
PATCH_RETRIES = 120


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _shape_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    shape_mask = (grid == BLUE) | (grid == RED) | (grid == YELLOW)
    coords = np.argwhere(shape_mask)
    assert coords.size > 0, "expected a colored shape"
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _line_side_horizontal(grid: np.ndarray, row: int, c0: int, c1: int) -> int:
    h, _ = grid.shape
    sides: set[int] = set()
    for c in (c0, c1):
        if row > 0 and int(grid[row - 1, c]) == GREEN:
            sides.add(-1)
        if row + 1 < h and int(grid[row + 1, c]) == GREEN:
            sides.add(1)
    assert sides and len(sides) == 1, "horizontal fold line must have exactly one green-directed side"
    return next(iter(sides))


def _line_side_vertical(grid: np.ndarray, col: int, r0: int, r1: int) -> int:
    _, w = grid.shape
    sides: set[int] = set()
    for r in (r0, r1):
        if col > 0 and int(grid[r, col - 1]) == GREEN:
            sides.add(-1)
        if col + 1 < w and int(grid[r, col + 1]) == GREEN:
            sides.add(1)
    assert sides and len(sides) == 1, "vertical fold line must have exactly one green-directed side"
    return next(iter(sides))


def _extract_lines(grid: np.ndarray) -> list[dict[str, int]]:
    top, left, bottom, right = _shape_bbox(grid)
    lines: list[dict[str, int]] = []

    for comp in _components(grid == RED):
        rows = sorted({r for r, _ in comp})
        cols = sorted({c for _, c in comp})
        if len(rows) == 1:
            row = rows[0]
            c0, c1 = cols[0], cols[-1]
            assert len(comp) == c1 - c0 + 1
            assert np.all(grid[row, c0 : c1 + 1] == RED)
            side = _line_side_horizontal(grid, row, c0, c1)
            depth = row - top if side == -1 else bottom - row
            lines.append(
                {
                    "orientation": 0,
                    "axis": row,
                    "start": c0,
                    "end": c1,
                    "source_side": side,
                    "depth": depth,
                }
            )
        elif len(cols) == 1:
            col = cols[0]
            r0, r1 = rows[0], rows[-1]
            assert len(comp) == r1 - r0 + 1
            assert np.all(grid[r0 : r1 + 1, col] == RED)
            side = _line_side_vertical(grid, col, r0, r1)
            depth = col - left if side == -1 else right - col
            lines.append(
                {
                    "orientation": 1,
                    "axis": col,
                    "start": r0,
                    "end": r1,
                    "source_side": side,
                    "depth": depth,
                }
            )
        else:
            raise AssertionError("each red component must be a straight line segment")

    assert 1 <= len(lines) <= 2, "expected one or two fold lines"
    lines.sort(key=lambda item: (item["depth"], item["orientation"], item["axis"]))
    return lines


def _apply_fold(counts: np.ndarray, greens: np.ndarray, line: dict[str, int]) -> None:
    h, w = counts.shape
    if line["orientation"] == 0:
        axis = line["axis"]
        if line["source_side"] == -1:
            coords = np.argwhere(counts[:axis, :] > 0)
        else:
            coords = np.argwhere(counts[axis + 1 :, :] > 0)
            coords[:, 0] += axis + 1
        values = counts[coords[:, 0], coords[:, 1]].copy()
        counts[coords[:, 0], coords[:, 1]] = 0
        for (r, c), val in zip(coords, values):
            tr, tc = 2 * axis - int(r), int(c)
            if 0 <= tr < h and 0 <= tc < w and not greens[tr, tc]:
                if counts[tr, tc] == 0:
                    counts[tr, tc] = val
                else:
                    counts[tr, tc] = 2
    else:
        axis = line["axis"]
        if line["source_side"] == -1:
            coords = np.argwhere(counts[:, :axis] > 0)
        else:
            coords = np.argwhere(counts[:, axis + 1 :] > 0)
            coords[:, 1] += axis + 1
        values = counts[coords[:, 0], coords[:, 1]].copy()
        counts[coords[:, 0], coords[:, 1]] = 0
        for (r, c), val in zip(coords, values):
            tr, tc = int(r), 2 * axis - int(c)
            if 0 <= tr < h and 0 <= tc < w and not greens[tr, tc]:
                if counts[tr, tc] == 0:
                    counts[tr, tc] = val
                else:
                    counts[tr, tc] = 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat the red line segments as fold axes. The green marker indicates which
    side folds across each axis; overlapping blue layers become yellow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    counts = np.zeros_like(g)
    counts[g == BLUE] = 1
    counts[g == YELLOW] = 2
    greens = g == GREEN

    for line in _extract_lines(g):
        _apply_fold(counts, greens, line)

    out = np.full_like(g, BLACK)
    out[counts == 1] = BLUE
    out[counts >= 2] = YELLOW
    out[greens] = GREEN
    return out


def _step_profile(
    rng: random.Random,
    depth: int,
    width: int,
    min_width: int,
    start: tuple[int, int] | None = None,
    require_gap: bool = False,
) -> list[tuple[int, int]]:
    left, right = (0, width - 1) if start is None else start
    profile: list[tuple[int, int]] = []
    for i in range(depth):
        options = [(0, 0), (1, 0), (0, 1), (1, 1)]
        rng.shuffle(options)
        if require_gap and i == 0:
            options = [opt for opt in options if opt != (0, 0)] + [(0, 0)]
        step_left, step_right = 0, 0
        for cand_left, cand_right in options:
            if right - left + 1 - cand_left - cand_right >= min_width:
                step_left, step_right = cand_left, cand_right
                break
        left += step_left
        right -= step_right
        profile.append((left, right))
    return profile


def _profiles_have_overlap(source: list[tuple[int, int]], target: list[tuple[int, int]]) -> bool:
    return any(max(sl, tl) <= min(sr, tr) for (sl, sr), (tl, tr) in zip(source, target))


def _profiles_need_bg_fill(source: list[tuple[int, int]], target: list[tuple[int, int]]) -> bool:
    return any(tl > sl or tr < sr for (sl, sr), (tl, tr) in zip(source, target))


def _make_interval_rows(
    intervals: list[tuple[int, int]],
    line_rows: list[int],
    source_sides: dict[int, int],
    rng: random.Random,
) -> np.ndarray:
    h = len(intervals)
    w = max(right for _, right in intervals) + 1
    patch = np.full((h, w), BLACK, dtype=int)

    for r, (left, right) in enumerate(intervals):
        patch[r, left : right + 1] = BLUE

    for row in line_rows:
        left, right = intervals[row]
        patch[row, left : right + 1] = RED

    for row in line_rows:
        side = source_sides[row]
        adj_row = row - 1 if side == -1 else row + 1
        adj_left, adj_right = intervals[adj_row]
        options: list[tuple[int, int]] = []
        if adj_left > 0:
            options.append((adj_row, 0))
        if adj_right < w - 1:
            options.append((adj_row, w - 1))
        assert options, "generator must leave a visible green-marker slot"
        gr, gc = rng.choice(options)
        patch[gr, gc] = GREEN

    return patch


def _build_one_line_patch(rng: random.Random) -> np.ndarray:
    for _ in range(PATCH_RETRIES):
        top_depth = rng.randint(2, 4)
        target_depth = rng.randint(top_depth + 1, top_depth + 4)
        width = rng.randint(8, 15)
        min_width = max(4, width // 2)

        source = _step_profile(rng, top_depth, width, min_width, require_gap=True)
        target = _step_profile(rng, top_depth, width, min_width)
        if not (_profiles_have_overlap(source, target) and _profiles_need_bg_fill(source, target)):
            continue

        extra = _step_profile(
            rng,
            target_depth - top_depth,
            width,
            min_width,
            start=target[-1],
        )
        intervals = list(reversed(source)) + [(0, width - 1)] + target + extra
        line_row = top_depth
        return _make_interval_rows(intervals, [line_row], {line_row: -1}, rng)
    raise RuntimeError("failed to generate one-line patch")


def _build_two_line_patch(rng: random.Random) -> np.ndarray:
    for _ in range(PATCH_RETRIES):
        top_depth = rng.randint(2, 4)
        bottom_depth = rng.randint(2, 4)
        gap_rows = rng.randint(1, 3)
        width = rng.randint(9, 16)
        min_width = max(4, width // 2)

        top_source = _step_profile(rng, top_depth, width, min_width, require_gap=True)
        top_target = _step_profile(rng, top_depth, width, min_width)
        bottom_source = _step_profile(rng, bottom_depth, width, min_width, require_gap=True)
        bottom_target = _step_profile(rng, bottom_depth, width, min_width)

        top_ok = _profiles_have_overlap(top_source, top_target) and _profiles_need_bg_fill(top_source, top_target)
        bottom_ok = _profiles_have_overlap(bottom_source, bottom_target) and _profiles_need_bg_fill(bottom_source, bottom_target)
        if not (top_ok or bottom_ok):
            continue

        center_gap = [(0, width - 1)] * gap_rows
        intervals = (
            list(reversed(top_source))
            + [(0, width - 1)]
            + top_target
            + center_gap
            + list(reversed(bottom_target))
            + [(0, width - 1)]
            + bottom_source
        )
        top_line = top_depth
        bottom_line = top_depth + 1 + top_depth + gap_rows + bottom_depth
        return _make_interval_rows(intervals, [top_line, bottom_line], {top_line: -1, bottom_line: 1}, rng)
    raise RuntimeError("failed to generate two-line patch")


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    patch = _build_one_line_patch(rng) if rng.random() < 0.5 else _build_two_line_patch(rng)

    if rng.random() < 0.5:
        patch = np.fliplr(patch)
    if rng.random() < 0.5:
        patch = np.flipud(patch)
    if rng.random() < 0.5:
        patch = patch.T

    ph, pw = patch.shape
    pad_top = rng.randint(1, 4)
    pad_bottom = rng.randint(1, 4)
    pad_left = rng.randint(1, 4)
    pad_right = rng.randint(1, 4)
    grid = np.full((ph + pad_top + pad_bottom, pw + pad_left + pad_right), BLACK, dtype=int)
    grid[pad_top : pad_top + ph, pad_left : pad_left + pw] = patch

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if np.any(out == RED):
        return None
    if not np.any(out == YELLOW):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid folding input")
