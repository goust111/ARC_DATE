from __future__ import annotations

import hashlib
import random
from collections import Counter

import numpy as np


BG = 0
OBSTACLE = 8
LEGEND_COLORS = (1, 2, 3, 4, 6, 7)

DIRS = {
    "N": (-1, 0),
    "E": (0, 1),
    "S": (1, 0),
    "W": (0, -1),
}
RIGHT_OF = {"N": "E", "E": "S", "S": "W", "W": "N"}
LEFT_OF = {value: key for key, value in RIGHT_OF.items()}
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}

PATTERNS_BY_SIDE = {
    "left": [["E", "S", "E"], ["E", "N", "E"], ["E"]],
    "right": [["W", "S", "W"], ["W", "N", "W"], ["W"]],
    "top": [["S", "E", "S"], ["S", "W", "S"], ["S"]],
    "bottom": [["N", "E"], ["N", "W"], ["N"]],
}

MAX_GENERATION_ATTEMPTS = 500
_SEEN_OUTPUT_SIGS: set[str] = set()


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    h, w = shape
    r, c = cell
    return 0 <= r < h and 0 <= c < w


def _add(cell: tuple[int, int], delta: tuple[int, int], scale: int = 1) -> tuple[int, int]:
    return cell[0] + delta[0] * scale, cell[1] + delta[1] * scale


def _nearest_inward_direction(shape: tuple[int, int], cell: tuple[int, int]) -> tuple[str, tuple[int, int]]:
    h, w = shape
    r, c = cell
    distances = {
        "top": r,
        "bottom": h - 1 - r,
        "left": c,
        "right": w - 1 - c,
    }
    best = min(distances.values())
    sides = [side for side, dist in distances.items() if dist == best]
    assert len(sides) == 1, "marker must have a unique nearest boundary"
    inward = {
        "top": "S",
        "bottom": "N",
        "left": "E",
        "right": "W",
    }[sides[0]]
    return inward, DIRS[inward]


def _parse_legends(grid: np.ndarray) -> list[dict]:
    top = list(map(int, grid[0]))
    legends: list[dict] = []
    c = 0
    while c < len(top):
        if top[c] in (BG, OBSTACLE):
            c += 1
            continue
        c0 = c
        seq = []
        while c < len(top) and top[c] not in (BG, OBSTACLE):
            seq.append(int(top[c]))
            c += 1
        legends.append({"seq": tuple(seq), "c0": c0, "c1": c - 1})
    assert 1 <= len(legends) <= 2, "expected 1-2 top-edge legends"
    for legend in legends:
        seq = legend["seq"]
        assert 3 <= len(seq) <= 5, "legend length must be 3-5"
        assert len(set(seq)) == len(seq), "colors within a legend must be distinct"
        assert all(color in LEGEND_COLORS for color in seq), "invalid legend color"
    return legends


def _find_markers(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    markers: list[tuple[int, int]] = []
    for r in range(1, h):
        for c in range(w):
            value = int(grid[r, c])
            if value in (BG, OBSTACLE):
                continue
            assert value in LEGEND_COLORS, "marker colors must come from the legend palette"
            for dr, dc in DIRS.values():
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) not in (BG, OBSTACLE):
                    raise AssertionError("markers must be isolated single pixels")
            markers.append((r, c))
    assert 1 <= len(markers) <= 2, "expected 1-2 markers"
    return markers


def _legend_occurrence(legends: list[dict], color: int) -> tuple[tuple[int, ...], int]:
    choices = []
    for legend in legends:
        for idx, value in enumerate(legend["seq"]):
            if int(value) == int(color):
                choices.append((legend["seq"], idx))
    assert len(choices) == 1, "marker color must identify a unique legend position"
    return choices[0]


def _trace_backward_prefix(grid: np.ndarray, start: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    prefix: list[tuple[int, int]] = []
    reverse = (-direction[0], -direction[1])
    cell = start
    while True:
        nxt = _add(cell, reverse)
        if not _in_bounds(grid.shape, nxt):
            break
        if int(grid[nxt]) == OBSTACLE:
            break
        prefix.append(nxt)
        cell = nxt
    return prefix


def _trace_forward_path(grid: np.ndarray, start: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    path = [start]
    cell = start
    heading = direction
    seen_states = set()

    while True:
        state = (cell, heading)
        assert state not in seen_states, "path must not loop"
        seen_states.add(state)

        nxt = _add(cell, heading)
        if not _in_bounds(grid.shape, nxt):
            break

        if int(grid[nxt]) == OBSTACLE:
            right = (heading[1], -heading[0])
            left = (-heading[1], heading[0])
            right_cell = _add(cell, right)
            left_cell = _add(cell, left)
            right_free = _in_bounds(grid.shape, right_cell) and int(grid[right_cell]) != OBSTACLE
            left_free = _in_bounds(grid.shape, left_cell) and int(grid[left_cell]) != OBSTACLE
            if right_free:
                heading = right
            elif left_free:
                heading = left
            else:
                break
            continue

        cell = nxt
        path.append(cell)

    return path


def _paint_legend(
    out: np.ndarray,
    base: np.ndarray,
    seq: tuple[int, ...],
    anchor_idx: int,
    prefix: list[tuple[int, int]],
    path: list[tuple[int, int]],
) -> None:
    anchor = path[0]
    for idx, color in enumerate(seq):
        step = idx - anchor_idx
        target = None
        if step == 0:
            target = anchor
        elif step > 0:
            if step < len(path):
                target = path[step]
        else:
            back = -step
            if back <= len(prefix):
                target = prefix[back - 1]
        if target is None:
            continue
        assert int(base[target]) != OBSTACLE, "legend cannot paint onto an obstacle"
        existing = int(out[target])
        if target != anchor and existing not in (BG, color):
            raise AssertionError("legend placement must remain conflict-free")
        out[target] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each marker's matched top-edge legend along its obstacle-turning path, anchored at the marker color.

    Input structure:
    - A black background grid contains 1-2 horizontal legend strings on the top edge.
    - Interior teal pixels are barrier-turn triggers and 1-2 isolated non-teal marker pixels sit away from the edge.
    - Each marker color appears in exactly one legend position, which fixes both the legend choice and the anchor index.

    Rule:
    - For each marker, take the direction opposite its unique nearest boundary so the walk starts toward the interior.
    - Build the usable path as a straight backward prefix toward that boundary plus a forward trace that goes straight and only turns when a teal pixel blocks the next step, preferring a right turn and otherwise taking a left turn.
    - Copy the matched legend onto this bent path so the anchor-colored legend cell stays on the marker, clipping any legend cells that would extend beyond the available prefix or beyond the traced forward path.
    - Leave the original legends and teal obstacles in place after painting.

    Key insight:
    - The marker does not spawn a straight row copy; the legend is aligned by color index and then folded along the exact obstacle-governed trajectory.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    legends = _parse_legends(g)
    markers = _find_markers(g)

    out = g.copy()
    for marker in markers:
        color = int(g[marker])
        seq, anchor_idx = _legend_occurrence(legends, color)
        _, direction = _nearest_inward_direction(g.shape, marker)
        prefix = _trace_backward_prefix(g, marker, direction)
        path = _trace_forward_path(g, marker, direction)
        _paint_legend(out, g, seq, anchor_idx, prefix, path)

    return out


def _random_legend_layout(width: int, count: int) -> list[tuple[int, int]]:
    if count == 1:
        length = random.randint(3, 5)
        return [(random.randint(0, width - length), length)]

    for _ in range(100):
        lengths = [random.randint(3, 5), random.randint(3, 5)]
        total = lengths[0] + lengths[1]
        if total + 2 > width:
            continue
        gap = random.randint(2, width - total)
        left = random.randint(0, width - total - gap)
        return [(left, lengths[0]), (left + lengths[0] + gap, lengths[1])]
    raise AssertionError("failed to place legends")


def _random_legend_sequences(lengths: list[int], marker_count: int) -> list[tuple[int, ...]]:
    for _ in range(200):
        seqs = [tuple(random.sample(LEGEND_COLORS, length)) for length in lengths]
        counts = Counter(color for seq in seqs for color in seq)
        if sum(1 for count in counts.values() if count == 1) >= marker_count:
            return seqs
    raise AssertionError("failed to create enough unique legend colors")


def _select_marker_specs(seqs: list[tuple[int, ...]], marker_count: int) -> list[dict]:
    occurrences = []
    counts = Counter(color for seq in seqs for color in seq)
    for legend_idx, seq in enumerate(seqs):
        for anchor_idx, color in enumerate(seq):
            if counts[color] == 1:
                occurrences.append(
                    {
                        "legend_idx": legend_idx,
                        "seq": seq,
                        "anchor_idx": anchor_idx,
                        "color": color,
                    }
                )

    assert len(occurrences) >= marker_count
    if marker_count == 2 and len(seqs) == 2:
        by_legend = {0: [], 1: []}
        for occ in occurrences:
            by_legend[occ["legend_idx"]].append(occ)
        if by_legend[0] and by_legend[1] and random.random() < 0.7:
            return [random.choice(by_legend[0]), random.choice(by_legend[1])]
    return random.sample(occurrences, marker_count)


def _candidate_start_positions(
    shape: tuple[int, int],
    side: str,
    back_len: int,
    top_legend_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    starts: list[tuple[int, int]] = []
    if side == "left":
        for r in range(2, h - 2):
            starts.append((r, back_len))
    elif side == "right":
        for r in range(2, h - 2):
            starts.append((r, w - 1 - back_len))
    elif side == "top":
        for c in range(2, w - 2):
            if (0, c) not in top_legend_cells:
                starts.append((back_len, c))
    else:
        for c in range(2, w - 2):
            starts.append((h - 1 - back_len, c))
    return [cell for cell in starts if _nearest_inward_direction(shape, cell)[0] == PATTERNS_BY_SIDE[side][0][0]]


def _prefix_cells(start: tuple[int, int], direction_name: str, count: int) -> list[tuple[int, int]]:
    reverse = DIRS[OPPOSITE[direction_name]]
    return [_add(start, reverse, step) for step in range(1, count + 1)]


def _realize_pattern(
    shape: tuple[int, int],
    start: tuple[int, int],
    pattern: list[str],
    blocked: set[tuple[int, int]],
    seg_lengths: list[int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    assert len(seg_lengths) == max(0, len(pattern) - 1)
    path = [start]
    used = {start}
    obstacles: list[tuple[int, int]] = []
    blocked_now = set(blocked)

    cell = start
    for idx, heading_name in enumerate(pattern[:-1]):
        heading = DIRS[heading_name]
        next_name = pattern[idx + 1]
        seg_cells = []
        for step in range(1, seg_lengths[idx] + 1):
            nxt = _add(cell, heading, step)
            if not _in_bounds(shape, nxt) or nxt in blocked_now or nxt in used:
                return None
            seg_cells.append(nxt)

        turn_cell = seg_cells[-1]
        ahead = _add(turn_cell, heading)
        if not _in_bounds(shape, ahead) or ahead in blocked_now or ahead in used:
            return None

        extras = [ahead]
        if next_name == RIGHT_OF[heading_name]:
            pass
        elif next_name == LEFT_OF[heading_name]:
            right_block = _add(turn_cell, DIRS[RIGHT_OF[heading_name]])
            if not _in_bounds(shape, right_block) or right_block in blocked_now or right_block in used or right_block == ahead:
                return None
            extras.append(right_block)
        else:
            return None

        first_after_turn = _add(turn_cell, DIRS[next_name])
        if not _in_bounds(shape, first_after_turn) or first_after_turn in blocked_now or first_after_turn in used:
            return None

        path.extend(seg_cells)
        used.update(seg_cells)
        for obstacle in extras:
            blocked_now.add(obstacle)
            obstacles.append(obstacle)
        cell = turn_cell

    final_heading = DIRS[pattern[-1]]
    final_cells: list[tuple[int, int]] = []
    cursor = cell
    while True:
        nxt = _add(cursor, final_heading)
        if not _in_bounds(shape, nxt):
            break
        if nxt in blocked_now or nxt in used:
            return None
        final_cells.append(nxt)
        used.add(nxt)
        cursor = nxt
    if not final_cells:
        return None

    path.extend(final_cells)
    return path, obstacles


def _place_marker_path(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    top_legend_cells: set[tuple[int, int]],
    anchor_idx: int,
    legend_len: int,
    require_turn: bool,
    clip_prefix: bool,
) -> dict | None:
    sides = list(PATTERNS_BY_SIDE)
    random.shuffle(sides)

    for side in sides:
        patterns = [pattern for pattern in PATTERNS_BY_SIDE[side] if (len(pattern) > 1) == require_turn]
        if not patterns:
            patterns = PATTERNS_BY_SIDE[side]
        random.shuffle(patterns)

        back_lengths = list(range(1, 5))
        random.shuffle(back_lengths)
        for back_len in back_lengths:
            if clip_prefix and not (anchor_idx > back_len):
                continue
            if not clip_prefix and anchor_idx > back_len and random.random() < 0.9:
                continue

            starts = _candidate_start_positions(shape, side, back_len, top_legend_cells)
            random.shuffle(starts)
            for start in starts:
                direction_name, _ = _nearest_inward_direction(shape, start)
                prefix = _prefix_cells(start, direction_name, back_len)
                if any(cell in occupied for cell in prefix):
                    continue

                pattern_choices = patterns[:]
                random.shuffle(pattern_choices)
                for pattern in pattern_choices:
                    seg_count = len(pattern) - 1
                    if seg_count == 0:
                        built = _realize_pattern(shape, start, pattern, occupied | set(prefix), [])
                        if built is None:
                            continue
                        path, obstacles = built
                    else:
                        candidates = [2, 3, 4, 5]
                        tried = set()
                        path = obstacles = None
                        for _ in range(60):
                            seg_lengths = tuple(random.choice(candidates) for _ in range(seg_count))
                            if seg_lengths in tried:
                                continue
                            tried.add(seg_lengths)
                            built = _realize_pattern(shape, start, pattern, occupied | set(prefix), list(seg_lengths))
                            if built is not None:
                                path, obstacles = built
                                break
                        if path is None:
                            continue

                    if len(path) - 1 < len(pattern) - 1:
                        continue
                    if len(prefix) + len(path) < legend_len:
                        continue

                    return {
                        "side": side,
                        "start": start,
                        "direction_name": direction_name,
                        "prefix": prefix,
                        "path": path,
                        "obstacles": obstacles,
                        "pattern": pattern,
                    }
    return None


def _distance1(cell1: tuple[int, int], cell2: tuple[int, int]) -> int:
    return abs(cell1[0] - cell2[0]) + abs(cell1[1] - cell2[1])


def _render_expected(shape: tuple[int, int], legends: list[dict], markers: list[dict], extra_obstacles: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for legend in legends:
        c0 = int(legend["c0"])
        for offset, color in enumerate(legend["seq"]):
            grid[0, c0 + offset] = int(color)
    for marker in markers:
        for obstacle in marker["obstacles"]:
            grid[obstacle] = OBSTACLE
        grid[marker["start"]] = int(marker["color"])
    for obstacle in extra_obstacles:
        grid[obstacle] = OBSTACLE

    out = grid.copy()
    for marker in markers:
        _paint_legend(out, grid, marker["seq"], marker["anchor_idx"], marker["prefix"], marker["path"])
    return grid, out


def _validate_case(grid: np.ndarray, expected: np.ndarray, meta: dict) -> None:
    legends = _parse_legends(grid)
    assert [legend["seq"] for legend in legends] == [legend["seq"] for legend in meta["legends"]]
    markers = _find_markers(grid)
    assert sorted(markers) == sorted(marker["start"] for marker in meta["markers"])

    for marker in meta["markers"]:
        seq, anchor_idx = _legend_occurrence(legends, marker["color"])
        assert seq == marker["seq"]
        assert anchor_idx == marker["anchor_idx"]
        _, direction = _nearest_inward_direction(grid.shape, marker["start"])
        prefix = _trace_backward_prefix(grid, marker["start"], direction)
        path = _trace_forward_path(grid, marker["start"], direction)
        assert prefix == marker["prefix"], "backward prefix mismatch"
        assert path == marker["path"], "forward path mismatch"
        assert len(prefix) + len(path) >= len(seq), "path too short for its legend"

    actual = main(grid)
    assert np.array_equal(actual, expected), "solver output mismatch"


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    h = random.randint(12, 16)
    w = random.randint(12, 16)
    marker_count = random.randint(1, 2)
    legend_count = random.randint(1, 2)

    layout = _random_legend_layout(w, legend_count)
    lengths = [length for _c0, length in layout]
    seqs = _random_legend_sequences(lengths, marker_count)
    legends = [{"c0": c0, "seq": seq, "c1": c0 + len(seq) - 1} for (c0, _), seq in zip(layout, seqs)]
    top_legend_cells = {(0, c) for legend in legends for c in range(legend["c0"], legend["c1"] + 1)}

    marker_specs = _select_marker_specs(seqs, marker_count)
    random.shuffle(marker_specs)

    occupied = set(top_legend_cells)
    markers: list[dict] = []
    turned_indices = set()
    must_turn_index = random.randrange(marker_count)

    for idx, spec in enumerate(marker_specs):
        require_turn = idx == must_turn_index or (marker_count == 2 and random.random() < 0.6)
        clip_prefix = spec["anchor_idx"] > 0 and random.random() < 0.55
        placed = _place_marker_path(
            (h, w),
            occupied,
            top_legend_cells,
            spec["anchor_idx"],
            len(spec["seq"]),
            require_turn=require_turn,
            clip_prefix=clip_prefix,
        )
        if placed is None:
            raise AssertionError("failed to place marker path")

        marker = {
            **placed,
            "legend_idx": spec["legend_idx"],
            "seq": spec["seq"],
            "anchor_idx": spec["anchor_idx"],
            "color": spec["color"],
            "clip_prefix": clip_prefix,
        }
        if len(marker["pattern"]) > 1:
            turned_indices.add(idx)
        markers.append(marker)
        occupied.update(marker["prefix"])
        occupied.update(marker["path"])
        occupied.update(marker["obstacles"])

    assert turned_indices, "at least one marker path should turn"

    extra_obstacles = []
    all_reserved = set(occupied)
    candidates = []
    for r in range(1, h):
        for c in range(w):
            cell = (r, c)
            if cell in all_reserved:
                continue
            if all(_distance1(cell, other) > 1 for other in all_reserved):
                candidates.append(cell)
    random.shuffle(candidates)
    for cell in candidates[: random.randint(0, 3)]:
        extra_obstacles.append(cell)
        all_reserved.add(cell)

    grid, expected = _render_expected((h, w), legends, markers, extra_obstacles)
    meta = {"legends": legends, "markers": markers}
    _validate_case(grid, expected, meta)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid, expected = _generate_case()
        except AssertionError:
            continue
        sig = hashlib.md5(expected.tobytes() + str(expected.shape).encode()).hexdigest()
        if sig in _SEEN_OUTPUT_SIGS:
            continue
        _SEEN_OUTPUT_SIGS.add(sig)
        return grid
    raise RuntimeError("failed to generate a valid non-degenerate sample")
