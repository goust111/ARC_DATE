from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
ARC_COLORS = tuple(range(10))
GEN_RETRIES = 400
TURN_SPAN_MIN = 3
SEGMENT_MIN = 3


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _rotate_left(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return (-dc, dr)


def _rotate_right(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return (dc, -dr)


def _neighbors4(r: int, c: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if _in_bounds(shape, rr, cc):
            out.append((rr, cc))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, mask.shape):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _infer_bg_and_path(g: np.ndarray) -> tuple[int, int, set[int]]:
    counts = Counter(int(v) for v in g.ravel())
    bg = int(counts.most_common(1)[0][0])
    non_bg = [int(color) for color in counts if int(color) != bg]
    assert non_bg, "expected non-background content"

    candidates: list[int] = []
    for color in non_bg:
        comps = _components(g == color)
        if any(len(comp) > 1 for comp in comps):
            candidates.append(int(color))

    assert len(candidates) == 1, "expected exactly one multi-cell path color"
    path = int(candidates[0])
    signals = {int(color) for color in non_bg if int(color) != path}
    assert signals, "expected at least one signal color"
    return bg, path, signals


def _path_neighbors(mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    return [(rr, cc) for rr, cc in _neighbors4(r, c, mask.shape) if bool(mask[rr, cc])]


def _signal_adjacent_path_count(path_mask: np.ndarray, r: int, c: int) -> int:
    return sum(bool(path_mask[rr, cc]) for rr, cc in _neighbors4(r, c, path_mask.shape))


def _active_turn(
    g: np.ndarray,
    path_mask: np.ndarray,
    signal_colors: set[int],
    r: int,
    c: int,
) -> tuple[int, int] | None:
    neighbors = _path_neighbors(path_mask, r, c)
    if len(neighbors) != 1:
        return None

    pr, pc = neighbors[0]
    heading = (r - pr, c - pc)
    options = (_rotate_left(heading), _rotate_right(heading))
    found: list[tuple[int, int]] = []
    for dr, dc in options:
        sr, sc = r + dr, c + dc
        if not _in_bounds(g.shape, sr, sc):
            continue
        if int(g[sr, sc]) not in signal_colors:
            continue
        if _signal_adjacent_path_count(path_mask, sr, sc) == 1:
            found.append((dr, dc))

    assert len(found) <= 1, "each endpoint should have at most one unresolved signal"
    return found[0] if found else None


def main(grid: np.ndarray) -> np.ndarray:
    """Extend broken route segments by turning at side signals and continuing to the next stop.

    Input structure:
    - The background is the dominant color.
    - One non-background color forms thin axis-aligned route fragments that may
      already include several joined straight segments.
    - Other non-background colors are single-pixel side signals placed next to
      an open route endpoint, on the left or right of the local travel direction.
    - The signal occupies the corner cell itself, so the resumed route continues
      beyond that fixed marker while leaving the marker unchanged.

    Rule:
    - Infer the route color and find each open endpoint of the current route.
    - For an endpoint with a side signal, determine the endpoint heading from its
      connected interior neighbor, then read whether the signal is on the left or
      on the right relative to that heading.
    - Turn 90 degrees toward the signal and extend the route in that direction,
      starting beyond the signal cell, until the new line reaches the grid edge
      or the next existing route fragment.
    - Repeat until no unresolved signal-adjacent open endpoint remains; keep all
      signal pixels unchanged.

    Key insight:
    - The signal's meaning is purely geometric: left versus right must be read
      relative to the route's current heading at that specific endpoint, not by
      absolute grid direction and not by the signal color itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()
    bg, path_color, signal_colors = _infer_bg_and_path(out)

    for _ in range(out.size):
        path_mask = out == path_color
        actions: list[tuple[int, int, int, int]] = []
        for r, c in np.argwhere(path_mask):
            turn = _active_turn(out, path_mask, signal_colors, int(r), int(c))
            if turn is not None:
                actions.append((int(r), int(c), int(turn[0]), int(turn[1])))

        if not actions:
            break

        actions.sort()
        r, c, dr, dc = actions[0]
        rr, cc = r + 2 * dr, c + 2 * dc
        painted = False
        while _in_bounds(out.shape, rr, cc) and int(out[rr, cc]) == bg:
            out[rr, cc] = path_color
            painted = True
            rr += dr
            cc += dc

        assert painted, "active signal should allow at least one new path cell"
    else:
        raise AssertionError("extension loop exceeded safety bound")

    return out


def _bounded_composition(total: int, parts: int, lo: int, hi: int) -> list[int] | None:
    if parts == 0:
        return [] if total == 0 else None
    if not (parts * lo <= total <= parts * hi):
        return None

    values: list[int] = []
    remaining = total
    for i in range(parts):
        slots = parts - i - 1
        cur_lo = max(lo, remaining - slots * hi)
        cur_hi = min(hi, remaining - slots * lo)
        if cur_lo > cur_hi:
            return None
        value = random.randint(cur_lo, cur_hi)
        values.append(value)
        remaining -= value
    random.shuffle(values)
    return values


def _sample_turn_positions(limit: int, count: int) -> list[int] | None:
    for _ in range(80):
        values = [random.randint(1, limit - 2)]
        ok = True
        for _step in range(count - 1):
            cur = values[-1]
            signs: list[int] = []
            if cur - TURN_SPAN_MIN >= 1:
                signs.append(-1)
            if cur + TURN_SPAN_MIN <= limit - 2:
                signs.append(1)
            if not signs:
                ok = False
                break
            sign = random.choice(signs)
            max_span = cur - 1 if sign < 0 else (limit - 2 - cur)
            span = random.randint(TURN_SPAN_MIN, max_span)
            values.append(cur + sign * span)
        if ok:
            return values
    return None


def _build_horizontal_polyline(height: int, width: int, n_segments: int) -> list[tuple[int, int]] | None:
    h_count = (n_segments + 1) // 2
    rows = _sample_turn_positions(height, h_count)
    if rows is None:
        return None

    lengths = _bounded_composition(width - 1, h_count, SEGMENT_MIN, width - 1)
    if lengths is None:
        return None

    cumulative = [0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)

    left_to_right = random.choice((True, False))
    if left_to_right:
        xs = cumulative
    else:
        xs = [width - 1 - value for value in cumulative]

    points: list[tuple[int, int]] = [(rows[0], xs[0])]
    for i in range(h_count):
        points.append((rows[i], xs[i + 1]))
        if i + 1 < h_count:
            points.append((rows[i + 1], xs[i + 1]))
    return points


def _build_vertical_polyline(height: int, width: int, n_segments: int) -> list[tuple[int, int]] | None:
    v_count = (n_segments + 1) // 2
    cols = _sample_turn_positions(width, v_count)
    if cols is None:
        return None

    lengths = _bounded_composition(height - 1, v_count, SEGMENT_MIN, height - 1)
    if lengths is None:
        return None

    cumulative = [0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)

    top_to_bottom = random.choice((True, False))
    if top_to_bottom:
        ys = cumulative
    else:
        ys = [height - 1 - value for value in cumulative]

    points: list[tuple[int, int]] = [(ys[0], cols[0])]
    for i in range(v_count):
        points.append((ys[i + 1], cols[i]))
        if i + 1 < v_count:
            points.append((ys[i + 1], cols[i + 1]))
    return points


def _segment_direction(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    assert (dr == 0) ^ (dc == 0), "segments must be axis-aligned"
    if dr != 0:
        return (1 if dr > 0 else -1, 0)
    return (0, 1 if dc > 0 else -1)


def _segment_length(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _choose_missing_indices(points: list[tuple[int, int]]) -> list[int] | None:
    eligible = [
        idx
        for idx in range(1, len(points) - 2)
        if _segment_length(points[idx], points[idx + 1]) >= TURN_SPAN_MIN
    ]
    valid: list[list[int]] = []
    n = len(eligible)
    for mask in range(1, 1 << n):
        chosen = [eligible[i] for i in range(n) if (mask >> i) & 1]
        if any(b == a + 1 for a, b in zip(chosen, chosen[1:])):
            continue
        valid.append(chosen)
    if not valid:
        return None
    return random.choice(valid)


def _paint_segment(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: int, start_step: int) -> None:
    dr, dc = _segment_direction(a, b)
    steps = _segment_length(a, b)
    for step in range(start_step, steps + 1):
        rr = a[0] + dr * step
        cc = a[1] + dc * step
        grid[rr, cc] = color


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(10, 24)
    width = random.randint(10, 24)
    mode = random.choice(("horizontal", "vertical"))

    possible_segments = [
        n
        for n in (3, 5, 7)
        if ((n + 1) // 2) * SEGMENT_MIN <= (width - 1 if mode == "horizontal" else height - 1)
    ]
    if not possible_segments:
        return None

    n_segments = random.choice(possible_segments)
    if mode == "horizontal":
        points = _build_horizontal_polyline(height, width, n_segments)
    else:
        points = _build_vertical_polyline(height, width, n_segments)
    if points is None:
        return None

    missing = _choose_missing_indices(points)
    if not missing:
        return None
    missing_set = set(missing)

    bg, path_color, sig_a, sig_b = random.sample(ARC_COLORS, 4)
    signal_choices = (sig_a, sig_b)

    inp = np.full((height, width), bg, dtype=int)
    out = np.full((height, width), bg, dtype=int)

    for idx in range(len(points) - 1):
        a = points[idx]
        b = points[idx + 1]
        if idx in missing_set:
            _paint_segment(out, a, b, path_color, start_step=2)
            dr, dc = _segment_direction(a, b)
            sr, sc = a[0] + dr, a[1] + dc
            signal_color = random.choice(signal_choices)
            inp[sr, sc] = signal_color
            out[sr, sc] = signal_color
        else:
            _paint_segment(inp, a, b, path_color, start_step=0)
            _paint_segment(out, a, b, path_color, start_step=0)

    return inp, out


def _valid_input(inp: np.ndarray) -> bool:
    bg, path_color, signal_colors = _infer_bg_and_path(inp)
    path_mask = inp == path_color
    endpoint_count = 0
    for r, c in np.argwhere(path_mask):
        neighbors = _path_neighbors(path_mask, int(r), int(c))
        if len(neighbors) != 1:
            continue
        endpoint_count += 1
        turn = _active_turn(inp, path_mask, signal_colors, int(r), int(c))
        if turn is not None:
            sr = int(r) + turn[0]
            sc = int(c) + turn[1]
            if _signal_adjacent_path_count(path_mask, sr, sc) != 1:
                return False
    return endpoint_count >= 2 and bg != path_color


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_case()
        if built is None:
            continue
        inp, expected = built
        if not _valid_input(inp):
            continue
        if not np.array_equal(main(inp), expected):
            continue
        return inp
    raise RuntimeError("failed to generate a valid signal-turn route puzzle")
