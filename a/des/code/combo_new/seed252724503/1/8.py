from __future__ import annotations

import random

import numpy as np


BG = 0
GATE = 8
PATH_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
GEN_RETRIES = 400


def _detect_gates(g: np.ndarray) -> tuple[str, list[int]]:
    h, w = g.shape
    gate_cols = [c for c in range(w) if np.all(g[:, c] != BG) and np.any(g[:, c] == GATE)]
    gate_rows = [r for r in range(h) if np.all(g[r, :] != BG) and np.any(g[r, :] == GATE)]

    if gate_cols and not gate_rows:
        return "vertical", gate_cols
    if gate_rows and not gate_cols:
        return "horizontal", gate_rows
    raise AssertionError("expected exactly one orientation of full-span gates")


def _parse_gate_rules(line: np.ndarray) -> dict[int, int]:
    line = np.asarray(line, dtype=int)
    rules: dict[int, int] = {}
    i = 0
    n = len(line)
    while i < n:
        if int(line[i]) == GATE:
            i += 1
            continue

        run: list[int] = []
        while i < n and int(line[i]) != GATE:
            value = int(line[i])
            assert value != BG
            run.append(value)
            i += 1

        assert len(run) % 2 == 0 and run, "gate markers must be 2x1 color pairs"
        for j in range(0, len(run), 2):
            a = int(run[j])
            b = int(run[j + 1])
            assert a not in rules, "each gate input color must be unambiguous"
            rules[a] = b

    assert rules, "each gate must carry at least one color rule"
    return rules


def _trace_line(line: np.ndarray, gate_positions: list[int], start_color: int) -> tuple[list[tuple[int, int]], list[int]]:
    line = np.asarray(line, dtype=int)
    n = len(line)
    assert n > 0 and int(line[0]) == int(start_color)

    segments: list[tuple[int, int]] = []
    crossings: list[int] = []
    pos = 0

    for gate in gate_positions:
        if gate <= pos:
            continue

        if (
            bool(np.all(line[pos:gate] == int(start_color)))
            and gate + 1 < n
            and int(line[gate]) == GATE
            and int(line[gate + 1]) == int(start_color)
        ):
            segments.append((int(pos), int(gate - 1)))
            crossings.append(int(gate))
            pos = int(gate + 1)
            continue

        end = pos
        while end + 1 < n and int(line[end + 1]) == int(start_color):
            end += 1
        segments.append((int(pos), int(end)))
        return segments, crossings

    end = pos
    while end + 1 < n and int(line[end + 1]) == int(start_color):
        end += 1
    segments.append((int(pos), int(end)))
    return segments, crossings


def _path_segment_colors(start_color: int, crossings: list[int], gate_rules: dict[int, dict[int, int]]) -> list[int]:
    colors = [int(start_color)]
    current = int(start_color)
    for gate in crossings:
        current = int(gate_rules[int(gate)].get(current, current))
        colors.append(current)
    return colors


def _paint_horizontal_path(
    out: np.ndarray,
    row: int,
    segments: list[tuple[int, int]],
    segment_colors: list[int],
    mirrored: bool,
) -> None:
    width = out.shape[1]
    for (a, b), color in zip(segments, segment_colors):
        if mirrored:
            c0 = int(width - 1 - b)
            c1 = int(width - 1 - a)
        else:
            c0 = int(a)
            c1 = int(b)
        out[int(row), c0 : c1 + 1] = int(color)


def _paint_vertical_path(
    out: np.ndarray,
    col: int,
    segments: list[tuple[int, int]],
    segment_colors: list[int],
    mirrored: bool,
) -> None:
    height = out.shape[0]
    for (a, b), color in zip(segments, segment_colors):
        if mirrored:
            r0 = int(height - 1 - b)
            r1 = int(height - 1 - a)
        else:
            r0 = int(a)
            r1 = int(b)
        out[r0 : r1 + 1, int(col)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each straight path after it passes through full-span color-conversion gates.

    Input structure:
    - The background is black.
    - Several single-pixel-wide horizontal or vertical colored paths start from a grid edge
      and run straight inward without touching each other.
    - Several parallel azure full-span lines perpendicular to the paths act as gates.
    - On each gate, away from path crossings, small 2x1 color-pair markers lie along the
      gate direction: the upper/left color A means "incoming A becomes outgoing B".

    Rule:
    - Detect the gate orientation and read every gate's A->B rules from its 2-cell markers.
    - For each path, start from its border-touching endpoint and follow the straight line inward.
    - Every time the path crosses a gate, update its current color if that gate has a rule for
      the current color; otherwise keep the color unchanged.
    - Recolor each path segment after the crossing using the accumulated current color.
    - Leave the azure gate cells and the colored gate markers unchanged.

    Key insight:
    - The 2-cell markers are local if-then rules, but the visible output requires chaining them
      in traversal order because the color produced by one gate becomes the input to the next.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out = g.copy()

    orientation, gate_positions = _detect_gates(g)
    gate_rules: dict[int, dict[int, int]] = {}
    if orientation == "vertical":
        for c in gate_positions:
            gate_rules[int(c)] = _parse_gate_rules(g[:, int(c)])

        width = g.shape[1]
        for r in range(g.shape[0]):
            left = int(g[r, 0])
            right = int(g[r, width - 1])
            left_start = left not in (BG, GATE)
            right_start = right not in (BG, GATE)
            if not left_start and not right_start:
                continue

            if left_start:
                line = g[r, :]
                trace_gates = list(gate_positions)
                start_color = left
                segments, crossings = _trace_line(line, trace_gates, start_color)
                segment_colors = _path_segment_colors(start_color, crossings, gate_rules)
                _paint_horizontal_path(out, r, segments, segment_colors, mirrored=False)
            else:
                line = g[r, ::-1]
                trace_gates = sorted(width - 1 - int(c) for c in gate_positions)
                start_color = int(line[0])
                segments, crossings = _trace_line(line, trace_gates, start_color)
                original_crossings = [int(width - 1 - gate) for gate in crossings]
                segment_colors = _path_segment_colors(start_color, original_crossings, gate_rules)
                _paint_horizontal_path(out, r, segments, segment_colors, mirrored=True)
        return out

    for r in gate_positions:
        gate_rules[int(r)] = _parse_gate_rules(g[int(r), :])

    height = g.shape[0]
    for c in range(g.shape[1]):
        top = int(g[0, c])
        bottom = int(g[height - 1, c])
        top_start = top not in (BG, GATE)
        bottom_start = bottom not in (BG, GATE)
        if not top_start and not bottom_start:
            continue

        if top_start:
            line = g[:, c]
            trace_gates = list(gate_positions)
            start_color = top
            segments, crossings = _trace_line(line, trace_gates, start_color)
            segment_colors = _path_segment_colors(start_color, crossings, gate_rules)
            _paint_vertical_path(out, c, segments, segment_colors, mirrored=False)
        else:
            line = g[::-1, c]
            trace_gates = sorted(height - 1 - int(r) for r in gate_positions)
            start_color = int(line[0])
            segments, crossings = _trace_line(line, trace_gates, start_color)
            original_crossings = [int(height - 1 - gate) for gate in crossings]
            segment_colors = _path_segment_colors(start_color, original_crossings, gate_rules)
            _paint_vertical_path(out, c, segments, segment_colors, mirrored=True)

    return out


def _sample_positions(count: int, lo: int, hi: int, min_gap: int) -> list[int] | None:
    span = int(hi - lo + 1)
    needed = int(count + (count - 1) * (min_gap - 1))
    if span < needed:
        return None
    compressed = span - (count - 1) * (min_gap - 1)
    picks = sorted(random.sample(range(compressed), count))
    return [int(lo + picks[i] + i * (min_gap - 1)) for i in range(count)]


def _pair_starts(length: int, blocked: set[int], count: int) -> list[int] | None:
    candidates = [i for i in range(length - 1) if i not in blocked and i + 1 not in blocked]
    random.shuffle(candidates)
    starts: list[int] = []
    for start in candidates:
        if any(abs(start - prev) < 3 for prev in starts):
            continue
        starts.append(int(start))
        if len(starts) == count:
            return sorted(starts)
    return None


def _base_case() -> np.ndarray | None:
    gate_count = random.randint(2, 4)
    path_count = random.randint(3, 5)

    height = random.randint(max(14, path_count * 2 + 4), 24)
    width = random.randint(max(16, gate_count * 4 + 6), 28)

    gate_cols = _sample_positions(gate_count, 2, width - 3, min_gap=3)
    path_rows = _sample_positions(path_count, 1, height - 2, min_gap=2)
    if gate_cols is None or path_rows is None:
        return None

    colors = random.sample(PATH_COLORS, path_count)
    cross_counts = [random.randint(1, gate_count) for _ in range(path_count)]
    if all(count == 1 for count in cross_counts):
        cross_counts[random.randrange(path_count)] = random.randint(2, gate_count)

    paths: list[dict] = []
    for row, color, cross_count in zip(path_rows, colors, cross_counts):
        side = random.choice(("left", "right"))
        if side == "left":
            crossed = gate_cols[:cross_count]
            last_gate = crossed[-1]
            next_gate = gate_cols[cross_count] if cross_count < gate_count else width - 1
            end_lo = int(last_gate + 1)
            end_hi = int(next_gate - 1 if cross_count < gate_count else width - 2)
        else:
            crossed = list(reversed(gate_cols))[:cross_count]
            last_gate = crossed[-1]
            next_gate = list(reversed(gate_cols))[cross_count] if cross_count < gate_count else 0
            end_lo = int(next_gate + 1 if cross_count < gate_count else 1)
            end_hi = int(last_gate - 1)

        if end_lo > end_hi:
            return None

        paths.append(
            {
                "row": int(row),
                "color": int(color),
                "side": side,
                "crossed": [int(c) for c in crossed],
                "end": int(random.randint(end_lo, end_hi)),
            }
        )

    feature_candidates = [path for path in paths if len(path["crossed"]) >= 2]
    if not feature_candidates:
        return None
    featured = random.choice(feature_candidates)

    gate_rules: dict[int, dict[int, int]] = {int(c): {} for c in gate_cols}
    first_gate = int(featured["crossed"][0])
    second_gate = int(featured["crossed"][1])
    first_in = int(featured["color"])
    first_out_choices = [color for color in PATH_COLORS if color != first_in]
    first_out = int(random.choice(first_out_choices))
    gate_rules[first_gate][first_in] = first_out

    blocked_inputs: dict[int, set[int]] = {int(c): set() for c in gate_cols}
    blocked_inputs[second_gate].add(first_out)

    if len(featured["crossed"]) >= 3 and random.random() < 0.7:
        third_gate = int(featured["crossed"][2])
        third_out_choices = [color for color in PATH_COLORS if color != first_out]
        gate_rules[third_gate][first_out] = int(random.choice(third_out_choices))

    for gate in gate_cols:
        rules = gate_rules[int(gate)]
        target_count = random.randint(max(1, len(rules)), 3)
        available_inputs = [
            color
            for color in PATH_COLORS
            if color not in rules and color not in blocked_inputs[int(gate)]
        ]
        random.shuffle(available_inputs)
        while len(rules) < target_count and available_inputs:
            a = int(available_inputs.pop())
            b_choices = [color for color in PATH_COLORS if color != a]
            rules[a] = int(random.choice(b_choices))
        if not rules:
            return None

    marker_starts: dict[int, list[int]] = {}
    blocked_rows = set(int(row) for row in path_rows)
    for gate in gate_cols:
        starts = _pair_starts(height, blocked_rows, len(gate_rules[int(gate)]))
        if starts is None:
            return None
        marker_starts[int(gate)] = starts

    grid = np.zeros((height, width), dtype=int)
    for gate in gate_cols:
        grid[:, int(gate)] = GATE

    gate_set = set(int(c) for c in gate_cols)
    for path in paths:
        row = int(path["row"])
        color = int(path["color"])
        end = int(path["end"])
        if path["side"] == "left":
            for c in range(0, end + 1):
                if c not in gate_set:
                    grid[row, c] = color
        else:
            for c in range(end, width):
                if c not in gate_set:
                    grid[row, c] = color

    for gate in gate_cols:
        items = list(gate_rules[int(gate)].items())
        random.shuffle(items)
        for start, (a, b) in zip(marker_starts[int(gate)], items):
            grid[int(start), int(gate)] = int(a)
            grid[int(start + 1), int(gate)] = int(b)

    if random.random() < 0.5:
        grid = grid.T.copy()

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _base_case()
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if np.array_equal(out, grid):
            continue
        return np.asarray(grid, dtype=int)
    raise RuntimeError("failed to generate a valid filter-gate ARC input")
