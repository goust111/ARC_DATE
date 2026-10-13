from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
Segment = tuple[int, int]
PALETTE = tuple(range(10))
MAX_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _row_segments(mask: np.ndarray) -> list[Segment]:
    spans: list[Segment] = []
    c = 0
    w = int(mask.shape[0])
    while c < w:
        if not bool(mask[c]):
            c += 1
            continue
        left = c
        while c + 1 < w and bool(mask[c + 1]):
            c += 1
        spans.append((int(left), int(c)))
        c += 1
    return spans


def _segment_containing(segments: list[Segment], col: int) -> Segment | None:
    c = int(col)
    for left, right in segments:
        if int(left) <= c <= int(right):
            return int(left), int(right)
    return None


def _role_colors(grid: np.ndarray) -> tuple[int, int, int]:
    bg = _get_bg(grid)
    top_non_bg = [int(c) for c in np.unique(grid[0]) if int(c) != bg]
    assert len(top_non_bg) == 1, "expected a unique water color on the top row"
    water = int(top_non_bg[0])

    shelf_colors = [int(c) for c in np.unique(grid) if int(c) not in {bg, water}]
    assert len(shelf_colors) == 1, "expected exactly one shelf color besides water"
    return int(bg), int(water), int(shelf_colors[0])


def _phase_interval(cols: list[int], period: int) -> tuple[int, int] | None:
    phases = sorted({int(c) % int(period) for c in cols})
    if not phases:
        return None

    p = int(period)
    best: tuple[int, int, int] | None = None
    for i, start in enumerate(phases):
        end = start
        for phase in phases:
            value = phase if phase >= start else phase + p
            end = max(end, value)
        prev = phases[i - 1] if i > 0 else phases[-1] - p
        gap_before = start - prev - 1
        score = (end - start + 1, -gap_before, start)
        if best is None or score < best:
            best = score

    assert best is not None
    width, _gap, start = best
    end = start + width - 1
    if end >= p:
        return None
    if width < 2 or width >= p:
        return None
    return int(start), int(end)


def _candidate_shelves(
    shape: tuple[int, int],
    shelf_coords: np.ndarray,
    shelf_rows: list[int],
    period: int,
) -> tuple[np.ndarray, int] | None:
    h, w = shape
    intended = np.zeros((h, w), dtype=bool)
    observed = np.zeros((h, w), dtype=bool)
    for r, c in shelf_coords:
        observed[int(r), int(c)] = True

    p = int(period)
    for r in shelf_rows:
        cols = sorted(int(c) for rr, c in shelf_coords if int(rr) == int(r))
        interval = _phase_interval(cols, p)
        if interval is None:
            return None
        start, end = interval

        possible = 0
        visible = 0
        for k in range((-p - end) // p, (w + p) // p + 1):
            left = start + k * p
            right = end + k * p
            if left < 0 or right >= w:
                continue
            possible += 1
            intended[int(r), left : right + 1] = True
            if bool(np.any(observed[int(r), left : right + 1])):
                visible += 1
        if possible < 2 or visible < 2:
            return None

    if bool(np.any(observed & ~intended)):
        return None
    holes = int(np.count_nonzero(intended & ~observed))
    return intended, holes


def _infer_intended_shelves(grid: np.ndarray, shelf: int) -> tuple[int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    shelf_coords = np.argwhere(g == int(shelf))
    assert shelf_coords.size > 0, "expected shelf pixels"
    shelf_rows = sorted({int(r) for r, _c in shelf_coords})

    candidates: list[tuple[int, int, np.ndarray]] = []
    for period in range(3, max(4, w // 2 + 1)):
        result = _candidate_shelves((h, w), shelf_coords, shelf_rows, period)
        if result is None:
            continue
        intended, holes = result
        if holes == 0:
            continue
        candidates.append((int(holes), int(period), intended))

    assert candidates, "expected a uniquely inferable horizontal shelf period"
    candidates.sort(key=lambda item: (item[0], item[1]))
    best_holes, best_period, best_intended = candidates[0]
    if len(candidates) > 1:
        next_holes, next_period, _next = candidates[1]
        assert best_holes < next_holes, f"ambiguous shelf period: {best_period} and {next_period}"
    return int(best_period), best_intended


def _paint_open(out: np.ndarray, original: np.ndarray, r: int, c: int, water: int, shelf: int) -> None:
    if 0 <= r < out.shape[0] and 0 <= c < out.shape[1] and int(original[r, c]) != int(shelf):
        out[r, c] = int(water)


def _simulate_water(
    grid: np.ndarray,
    water: int,
    shelf: int,
    intended: np.ndarray,
    *,
    collect_events: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    actual_shelf = g == int(shelf)
    row_spans = [_row_segments(intended[r]) for r in range(h)]
    out = g.copy()
    events: dict[str, object] = {"hole_leaks": 0, "overflows": 0, "hit_rows": set()}

    sources = [(0, int(c)) for c in np.flatnonzero(g[0] == int(water))]
    assert sources, "expected at least one top-row source"
    q: deque[Coord] = deque(sources)
    seen_starts: set[Coord] = set()

    while q:
        r, c = q.popleft()
        r = int(r)
        c = int(c)
        if not (0 <= r < h and 0 <= c < w):
            continue
        if (r, c) in seen_starts:
            continue
        seen_starts.add((r, c))
        if int(g[r, c]) == int(shelf):
            continue

        _paint_open(out, g, r, c, water, shelf)
        while True:
            nr = r + 1
            if nr >= h:
                break
            if int(out[nr, c]) == int(water) and int(g[nr, c]) != int(shelf):
                break

            if bool(actual_shelf[nr, c]):
                segment = _segment_containing(row_spans[nr], c)
                assert segment is not None, "actual shelf must lie inside inferred span"
                left, right = segment
                spread_r = r
                cast_rows = events["hit_rows"]
                assert isinstance(cast_rows, set)
                cast_rows.add(int(nr))

                _paint_open(out, g, spread_r, c, water, shelf)
                for dc in (-1, 1):
                    x = c
                    while True:
                        x += dc
                        if x < left or x > right:
                            overflow_c = left - 1 if dc < 0 else right + 1
                            if 0 <= overflow_c < w:
                                _paint_open(out, g, spread_r, overflow_c, water, shelf)
                                events["overflows"] = int(events["overflows"]) + 1
                                q.append((int(spread_r), int(overflow_c)))
                            break

                        _paint_open(out, g, spread_r, x, water, shelf)
                        if bool(intended[nr, x]) and not bool(actual_shelf[nr, x]):
                            out[nr, x] = int(water)
                            events["hole_leaks"] = int(events["hole_leaks"]) + 1
                            q.append((int(nr), int(x)))
                            break
                break

            if int(g[nr, c]) != int(_get_bg(g)) and int(g[nr, c]) != int(water):
                break

            r = nr
            _paint_open(out, g, r, c, water, shelf)

    if not collect_events:
        events = {}
    return out, events


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let top-edge water fall over phase-repaired shelves, leaking through inferred missing shelf pixels.

    Input structure:
    - The background is the most common color; a unique water color appears on the top row.
    - The other non-background color forms horizontal shelf fragments on several rows.
    - All shelf rows share one smallest horizontal period, and some periodic shelf cells are erased to background as holes.

    Rule:
    - Infer the unique smallest horizontal period from the remaining shelf pixels, then reconstruct each row's intended repeated shelf spans.
    - Water falls straight down through background cells until it hits a real shelf pixel.
    - Above a hit shelf span, water spreads left and right; the first inferred hole on a side is filled and leaks downward, while a side with no hole overflows one cell beyond the true endpoint.
    - Continue every leaked or overflowed stream recursively until it reaches the bottom or existing water.

    Key insight:
    - A background cell inside the periodic shelf phase is a leak, while an identical-looking background cell beyond the true endpoint is an overflow point.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0 and max(g.shape) <= 30, "expected an ARC-sized 2D grid"
    _bg, water, shelf = _role_colors(g)
    _period, intended = _infer_intended_shelves(g, shelf)
    out, _events = _simulate_water(g, water, shelf, intended)
    return out


def _segments_for_interval(width: int, period: int, start: int, end: int) -> list[Segment]:
    spans: list[Segment] = []
    for k in range((-period - end) // period, (width + period) // period + 1):
        left = start + k * period
        right = end + k * period
        if 0 <= left and right < width:
            spans.append((int(left), int(right)))
    return spans


def _intended_from_layout(shape: tuple[int, int], rows: list[int], period: int, start: int, end: int) -> np.ndarray:
    h, w = shape
    intended = np.zeros((h, w), dtype=bool)
    for r in rows:
        for left, right in _segments_for_interval(w, period, start, end):
            intended[int(r), left : right + 1] = True
    return intended


def _choose_rows(height: int, count: int) -> list[int] | None:
    choices = list(range(3, height - 1))
    for _ in range(80):
        rows = sorted(random.sample(choices, int(count)))
        if all(rows[i + 1] - rows[i] >= 3 for i in range(len(rows) - 1)):
            return [int(r) for r in rows]
    return None


def _build_candidate() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(14, 24)
    width = random.randint(21, 30)
    period = random.randint(6, min(8, width // 3))
    span_len = random.randint(3, period - 2)
    start = random.randint(1, period - span_len - 1)
    end = start + span_len - 1
    row_count = random.randint(3, min(5, max(3, (height - 4) // 3)))
    rows = _choose_rows(height, row_count)
    if rows is None:
        return None

    bg, water, shelf = random.sample(list(PALETTE), 3)
    intended = _intended_from_layout((height, width), rows, period, start, end)
    segments = _segments_for_interval(width, period, start, end)
    segments = [seg for seg in segments if seg[0] > 0 and seg[1] < width - 1]
    segments = [seg for seg in segments if seg[1] - seg[0] + 1 >= 3]
    if not segments:
        return None

    top_row = rows[0]
    top_left, top_right = random.choice(segments)
    source_col = random.randint(top_left, top_right)
    left_choices = [c for c in range(top_left, source_col)]
    right_choices = [c for c in range(source_col + 1, top_right + 1)]
    if not left_choices and not right_choices:
        return None
    if not left_choices:
        hole_col = random.choice(right_choices)
    elif not right_choices:
        hole_col = random.choice(left_choices)
    else:
        hole_col = random.choice(left_choices if random.random() < 0.5 else right_choices)

    protected: set[Coord] = {(int(top_row), int(source_col))}
    for r in rows[1:]:
        protected.add((int(r), int(hole_col)))

    holes: set[Coord] = {(int(top_row), int(hole_col))}
    second_row = rows[1]
    second_seg = _segment_containing(_row_segments(intended[second_row]), hole_col)
    if second_seg is not None and random.random() < 0.8:
        options = [c for c in range(second_seg[0], second_seg[1] + 1) if c != hole_col]
        if options:
            holes.add((int(second_row), int(random.choice(options))))

    all_intended = [tuple(map(int, rc)) for rc in np.argwhere(intended)]
    random.shuffle(all_intended)
    extra = random.randint(1, 5)
    for r, c in all_intended:
        if extra <= 0:
            break
        if (r, c) in holes or (r, c) in protected:
            continue
        phase = c % period
        if phase in {start, end} and random.random() < 0.8:
            continue
        holes.add((int(r), int(c)))
        extra -= 1

    grid = np.full((height, width), int(bg), dtype=int)
    grid[intended] = int(shelf)
    for r, c in holes:
        grid[int(r), int(c)] = int(bg)
    grid[0, source_col] = int(water)

    if random.random() < 0.4:
        choices = [
            c
            for left, right in segments
            for c in range(left, right + 1)
            if c != source_col and (top_row, c) not in holes
        ]
        if choices:
            grid[0, int(random.choice(choices))] = int(water)

    counts = Counter(map(int, grid.ravel()))
    if counts[int(bg)] <= max(counts.get(int(water), 0), counts.get(int(shelf), 0)):
        return None

    try:
        inferred_period, inferred = _infer_intended_shelves(grid, shelf)
    except Exception:
        return None
    if inferred_period != period or not np.array_equal(inferred, intended):
        return None

    expected, events = _simulate_water(grid, water, shelf, intended, collect_events=True)
    hit_rows = events.get("hit_rows", set())
    if not isinstance(hit_rows, set):
        return None
    if int(events.get("hole_leaks", 0)) < 1 or int(events.get("overflows", 0)) < 1:
        return None
    if len(hit_rows) < 2:
        return None
    if np.array_equal(expected, grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        candidate = _build_candidate()
        if candidate is not None:
            grid, _expected = candidate
            return grid
    raise RuntimeError(f"failed to generate a valid phase-leak waterfall input after {MAX_ATTEMPTS} attempts")
