from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
ORANGE = 7
SOURCE_COLORS = (1, 2, 3, 6, 8, 9)

GRID_H_RANGE = (16, 22)
GRID_W_RANGE = (16, 22)
SOURCE_LENGTHS = (2, 3, 4, 5)
GENERATION_ATTEMPTS = 400
SCENARIOS = ("triple_merge", "dual_merge", "vertical_mix", "separate")


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_sources(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    sources: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BLACK, GREY) or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))

            arr = np.array(sorted(coords), dtype=int)
            sources.append(
                {
                    "color": int(color),
                    "coords": arr,
                    "area": int(arr.shape[0]),
                }
            )

    return sources


def _source_span(source: dict[str, object]) -> tuple[str, int, int, int]:
    coords = np.asarray(source["coords"], dtype=int)
    rows = np.unique(coords[:, 0])
    cols = np.unique(coords[:, 1])
    area = int(source["area"])

    if rows.size == 1:
        row = int(rows[0])
        lo = int(cols.min())
        hi = int(cols.max())
        assert hi - lo + 1 == area, "horizontal source must occupy consecutive rail cells"
        return "horizontal", row, lo, hi

    if cols.size == 1:
        col = int(cols[0])
        lo = int(rows.min())
        hi = int(rows.max())
        assert hi - lo + 1 == area, "vertical source must occupy consecutive rail cells"
        return "vertical", col, lo, hi

    raise AssertionError("each source must be a straight 1-cell-wide segment")


def _propagation_cells(grid: np.ndarray, source: dict[str, object]) -> set[tuple[int, int]]:
    h, w = grid.shape
    area = int(source["area"])
    axis, fixed, lo, hi = _source_span(source)

    if axis == "horizontal":
        starts = ((fixed, lo - 1), (fixed, hi + 1))
    else:
        starts = ((lo - 1, fixed), (hi + 1, fixed))

    seen: set[tuple[int, int]] = set()
    q: deque[tuple[int, int, int]] = deque()

    for r, c in starts:
        if 0 <= r < h and 0 <= c < w and int(grid[r, c]) == GREY and (r, c) not in seen:
            seen.add((int(r), int(c)))
            q.append((int(r), int(c), 1))

    while q:
        r, c, dist = q.popleft()
        if dist >= area:
            continue
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == GREY and (nr, nc) not in seen:
                seen.add((int(nr), int(nc)))
                q.append((int(nr), int(nc), dist + 1))

    return seen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each colored source through the grey rail network by a number of steps equal to its area, mixing overlaps.

    Input structure:
    - The background is black.
    - Grey 1-pixel rails run horizontally or vertically and may intersect at shared grey cells.
    - Colored sources are 2-5 pixel monochrome segments that lie on top of one rail and replace consecutive rail cells.
    - Each rail carries at most one source.

    Rule:
    - Identify every source as a straight segment and let its pixel count be N.
    - From the grey cell just beyond each source endpoint, expand through 4-connected grey rail cells for at most N steps; source pixels themselves stay unchanged.
    - A grey cell reached by one source takes that source color, a grey cell reached by two different signals becomes yellow, and a grey cell reached by three or more signals becomes orange.
    - Grey cells reached by no signal remain grey, and black background remains black.

    Key insight:
    - The source area is the propagation budget, and rail intersections let independently started signals meet so overlap color depends on how many signals reach the same grey rail cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sources = _find_sources(g)
    assert sources, "expected at least one colored source"

    coverage: dict[tuple[int, int], list[int]] = {}
    for idx, source in enumerate(sources):
        area = int(source["area"])
        assert 2 <= area <= 5, "source area must encode a distance from 2 to 5"
        _source_span(source)
        for cell in _propagation_cells(g, source):
            coverage.setdefault(cell, []).append(int(idx))

    out = g.copy()
    for (r, c), ids in coverage.items():
        assert int(g[r, c]) == GREY, "propagation may only recolor grey rail cells"
        colors = {int(sources[idx]["color"]) for idx in ids}
        if len(ids) >= 3:
            out[r, c] = ORANGE
        elif len(colors) >= 2:
            out[r, c] = YELLOW
        else:
            out[r, c] = next(iter(colors))

    return out


def _sample_dims() -> tuple[int, int, int, int, int]:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    top = random.randint(1, 2)
    bottom = h - 1 - random.randint(1, 2)
    trunk_c = random.randint(6, w - 7)
    return h, w, top, bottom, trunk_c


def _sample_spaced_rows(lo: int, hi: int, count: int, min_gap: int) -> list[int] | None:
    if hi - lo + 1 < count:
        return None
    for _ in range(80):
        rows = sorted(random.sample(range(lo, hi + 1), count))
        if all(rows[i + 1] - rows[i] >= min_gap for i in range(len(rows) - 1)):
            return rows
    return None


def _draw_base(h: int, w: int, top: int, bottom: int, trunk_c: int, rows: list[int]) -> tuple[np.ndarray, dict[int, tuple[int, int]]]:
    grid = np.full((h, w), BLACK, dtype=int)
    grid[top : bottom + 1, trunk_c] = GREY

    spans: dict[int, tuple[int, int]] = {}
    for row in rows:
        left = random.randint(1, 2)
        right = w - 1 - random.randint(1, 2)
        spans[int(row)] = (int(left), int(right))
        grid[int(row), int(left) : int(right) + 1] = GREY

    return grid, spans


def _horizontal_candidates(
    row: int,
    left: int,
    right: int,
    trunk_c: int,
    target_row: int | None,
    must_reach_target: bool | None,
    must_reach_trunk: bool | None,
    force_truncate: bool | None,
    allowed_sides: tuple[str, ...],
) -> list[dict[str, int | str | bool]]:
    candidates: list[dict[str, int | str | bool]] = []

    for length in SOURCE_LENGTHS:
        if "left" in allowed_sides:
            for start in range(left, trunk_c - length + 1):
                end = start + length - 1
                if end >= trunk_c:
                    continue
                dist_trunk = trunk_c - end
                dist_target = dist_trunk if target_row is None else dist_trunk + abs(row - target_row)
                truncated = start - left < length
                if must_reach_trunk is not None and (dist_trunk <= length) != must_reach_trunk:
                    continue
                if must_reach_target is not None and (dist_target <= length) != must_reach_target:
                    continue
                if force_truncate is not None and truncated != force_truncate:
                    continue
                candidates.append(
                    {
                        "orientation": "horizontal",
                        "row": int(row),
                        "start": int(start),
                        "end": int(end),
                        "length": int(length),
                        "side": "left",
                        "truncated": bool(truncated),
                    }
                )

        if "right" in allowed_sides:
            for start in range(trunk_c + 1, right - length + 2):
                end = start + length - 1
                if start <= trunk_c or end > right:
                    continue
                dist_trunk = start - trunk_c
                dist_target = dist_trunk if target_row is None else dist_trunk + abs(row - target_row)
                truncated = right - end < length
                if must_reach_trunk is not None and (dist_trunk <= length) != must_reach_trunk:
                    continue
                if must_reach_target is not None and (dist_target <= length) != must_reach_target:
                    continue
                if force_truncate is not None and truncated != force_truncate:
                    continue
                candidates.append(
                    {
                        "orientation": "horizontal",
                        "row": int(row),
                        "start": int(start),
                        "end": int(end),
                        "length": int(length),
                        "side": "right",
                        "truncated": bool(truncated),
                    }
                )

    return candidates


def _vertical_candidates(
    top: int,
    bottom: int,
    trunk_c: int,
    cross_rows: list[int],
    target_row: int,
    force_truncate: bool | None,
) -> list[dict[str, int | str | bool]]:
    candidates: list[dict[str, int | str | bool]] = []
    topmost = min(cross_rows)
    bottommost = max(cross_rows)

    for length in SOURCE_LENGTHS:
        for start in range(top, topmost - length + 1):
            end = start + length - 1
            if end >= topmost:
                continue
            dist_target = target_row - end
            truncated = start - top < length
            if dist_target <= 0 or dist_target > length:
                continue
            if force_truncate is not None and truncated != force_truncate:
                continue
            candidates.append(
                {
                    "orientation": "vertical",
                    "col": int(trunk_c),
                    "start": int(start),
                    "end": int(end),
                    "length": int(length),
                    "side": "top",
                    "truncated": bool(truncated),
                }
            )

        for start in range(bottommost + 1, bottom - length + 2):
            end = start + length - 1
            if start <= bottommost or end > bottom:
                continue
            dist_target = start - target_row
            truncated = bottom - end < length
            if dist_target <= 0 or dist_target > length:
                continue
            if force_truncate is not None and truncated != force_truncate:
                continue
            candidates.append(
                {
                    "orientation": "vertical",
                    "col": int(trunk_c),
                    "start": int(start),
                    "end": int(end),
                    "length": int(length),
                    "side": "bottom",
                    "truncated": bool(truncated),
                }
            )

    return candidates


def _paint_source(grid: np.ndarray, source: dict[str, int | str | bool], color: int) -> dict[str, int | str | bool]:
    if source["orientation"] == "horizontal":
        row = int(source["row"])
        start = int(source["start"])
        end = int(source["end"])
        assert np.all(grid[row, start : end + 1] == GREY), "source must overwrite only grey rail cells"
        grid[row, start : end + 1] = int(color)
    else:
        col = int(source["col"])
        start = int(source["start"])
        end = int(source["end"])
        assert np.all(grid[start : end + 1, col] == GREY), "source must overwrite only grey rail cells"
        grid[start : end + 1, col] = int(color)

    placed = dict(source)
    placed["color"] = int(color)
    return placed


def _build_triple_merge() -> tuple[np.ndarray, list[dict[str, int | str | bool]], str] | None:
    h, w, top, bottom, trunk_c = _sample_dims()
    gap = random.choice((1, 2))
    center = random.randint(top + 4, bottom - 4)
    rows = [center - gap, center, center + gap]
    grid, spans = _draw_base(h, w, top, bottom, trunk_c, rows)

    colors = random.sample(SOURCE_COLORS, 3)
    trunc_idx = random.randrange(3)
    placed: list[dict[str, int | str | bool]] = []

    for idx, row in enumerate(rows):
        left, right = spans[row]
        candidates = _horizontal_candidates(
            row,
            left,
            right,
            trunk_c,
            center,
            True,
            None,
            True if idx == trunc_idx and random.random() < 0.7 else None,
            ("left", "right"),
        )
        if not candidates:
            return None
        placed.append(_paint_source(grid, random.choice(candidates), colors[idx]))

    return grid, placed, "triple_merge"


def _build_dual_merge() -> tuple[np.ndarray, list[dict[str, int | str | bool]], str] | None:
    h, w, top, bottom, trunk_c = _sample_dims()
    gap = random.choice((1, 2))
    center = random.randint(top + 4, bottom - 4)
    rows = [center - gap, center + gap]
    grid, spans = _draw_base(h, w, top, bottom, trunk_c, rows)

    colors = random.sample(SOURCE_COLORS, 2)
    placed: list[dict[str, int | str | bool]] = []
    for idx, row in enumerate(rows):
        left, right = spans[row]
        candidates = _horizontal_candidates(
            row,
            left,
            right,
            trunk_c,
            center,
            True,
            None,
            True if idx == 0 and random.random() < 0.6 else None,
            ("left", "right"),
        )
        if not candidates:
            return None
        placed.append(_paint_source(grid, random.choice(candidates), colors[idx]))

    return grid, placed, "dual_merge"


def _build_vertical_mix() -> tuple[np.ndarray, list[dict[str, int | str | bool]], str] | None:
    h, w, top, bottom, trunk_c = _sample_dims()
    gap = random.choice((1, 2))
    center = random.randint(top + 4, bottom - 4)
    rows = [center, center + gap]
    if rows[-1] > bottom - 1:
        rows = [center - gap, center]
    grid, spans = _draw_base(h, w, top, bottom, trunk_c, rows)

    target_row = random.choice(rows)
    colors = random.sample(SOURCE_COLORS, 2)

    vertical_candidates = _vertical_candidates(top, bottom, trunk_c, rows, target_row, None)
    if not vertical_candidates:
        return None
    placed = [_paint_source(grid, random.choice(vertical_candidates), colors[0])]

    left, right = spans[target_row]
    horizontal_candidates = _horizontal_candidates(
        target_row,
        left,
        right,
        trunk_c,
        target_row,
        True,
        None,
        True if random.random() < 0.5 else None,
        ("left", "right"),
    )
    if not horizontal_candidates:
        return None
    placed.append(_paint_source(grid, random.choice(horizontal_candidates), colors[1]))

    return grid, placed, "vertical_mix"


def _build_separate() -> tuple[np.ndarray, list[dict[str, int | str | bool]], str] | None:
    h, w, top, bottom, trunk_c = _sample_dims()
    row_count = random.choice((2, 3))
    rows = _sample_spaced_rows(top + 1, bottom - 1, row_count, 3)
    if rows is None:
        return None
    grid, spans = _draw_base(h, w, top, bottom, trunk_c, rows)

    colors = random.sample(SOURCE_COLORS, row_count)
    trunc_idx = random.randrange(row_count)
    placed: list[dict[str, int | str | bool]] = []

    for idx, row in enumerate(rows):
        left, right = spans[row]
        candidates = _horizontal_candidates(
            row,
            left,
            right,
            trunk_c,
            None,
            None,
            False,
            True if idx == trunc_idx else None,
            ("left", "right"),
        )
        if not candidates:
            return None
        placed.append(_paint_source(grid, random.choice(candidates), colors[idx]))

    return grid, placed, "separate"


def _build_scene() -> tuple[np.ndarray, list[dict[str, int | str | bool]], str] | None:
    scenario = random.choice(SCENARIOS)
    if scenario == "triple_merge":
        return _build_triple_merge()
    if scenario == "dual_merge":
        return _build_dual_merge()
    if scenario == "vertical_mix":
        return _build_vertical_mix()
    return _build_separate()


def _validate_scene(grid: np.ndarray, sources: list[dict[str, int | str | bool]], scenario: str) -> bool:
    out = main(grid)
    if np.array_equal(grid, out):
        return False
    if np.count_nonzero(out == GREY) == 0:
        return False

    yellow_count = int(np.count_nonzero(out == YELLOW))
    orange_count = int(np.count_nonzero(out == ORANGE))
    truncated_count = sum(bool(src["truncated"]) for src in sources)
    vertical_count = sum(src["orientation"] == "vertical" for src in sources)

    if scenario == "triple_merge":
        return orange_count >= 1
    if scenario == "dual_merge":
        return yellow_count >= 1 and orange_count == 0
    if scenario == "vertical_mix":
        return vertical_count == 1 and yellow_count >= 1
    return yellow_count == 0 and orange_count == 0 and truncated_count >= 1


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        built = _build_scene()
        if built is None:
            continue
        grid, sources, scenario = built
        if _validate_scene(grid, sources, scenario):
            return grid
    raise RuntimeError("failed to generate a valid signal-rail scene")
