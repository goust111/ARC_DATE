from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
MAROON = 9
INPUT_COLORS = tuple(range(1, 9))
MAX_GENERATION_RETRIES = 400


def _parse_edge_runs(line: np.ndarray, offset: int) -> dict[int, tuple[int, int, int]]:
    runs: dict[int, tuple[int, int, int]] = {}
    i = int(offset)
    n = int(line.size)
    while i < n:
        color = int(line[i])
        if color == BG:
            i += 1
            continue
        start = i
        while i + 1 < n and int(line[i + 1]) == color:
            i += 1
        end = i
        runs[color] = (start, end, start + end)  # store doubled axis coordinate
        i += 1
    return runs


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _find_rectangles(grid: np.ndarray, colors: set[int]) -> list[dict[str, int]]:
    interior = grid[1:, 1:]
    rects: list[dict[str, int]] = []
    for color in sorted(colors):
        comps = _components(interior == color)
        for comp in comps:
            rows = [r + 1 for r, _ in comp]
            cols = [c + 1 for _, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            assert area == len(comp), "interior objects must be solid rectangles"
            assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color), "object bbox must be monochrome"
            rects.append(
                {
                    "color": int(color),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "area": int(area),
                }
            )
    return rects


def _reflect_interval(start: int, end: int, axis2: int) -> tuple[int, int]:
    return int(axis2 - end), int(axis2 - start)


def _copy_bboxes(rect: dict[str, int], row_axis2: int, col_axis2: int) -> list[tuple[int, int, int, int]]:
    row_pairs = [(rect["r0"], rect["r1"]), _reflect_interval(rect["r0"], rect["r1"], row_axis2)]
    col_pairs = [(rect["c0"], rect["c1"]), _reflect_interval(rect["c0"], rect["c1"], col_axis2)]
    out: list[tuple[int, int, int, int]] = []
    for rr0, rr1 in row_pairs:
        for cc0, cc1 in col_pairs:
            out.append((int(rr0), int(cc0), int(rr1), int(cc1)))
    return out


def _object_counts(shape: tuple[int, int], rect: dict[str, int], row_axis2: int, col_axis2: int) -> np.ndarray:
    h, w = shape
    counts = np.zeros((h, w), dtype=int)
    for r0, c0, r1, c1 in _copy_bboxes(rect, row_axis2, col_axis2):
        assert 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w, "reflected copies must stay inside the grid"
        counts[r0 : r1 + 1, c0 : c1 + 1] += 1
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each interior rectangle across the two fold axes indicated by same-color edge bands.

    Input structure:
    - Background is black.
    - Row 0 (excluding the top-left corner) is split into solid color bands, and column 0
      (excluding the top-left corner) is split into solid color bands.
    - A repeated color appears exactly once on the top band row and exactly once on the left
      band column; the interior contains solid monochrome rectangles in such repeated colors.

    Rule:
    - For each interior rectangle, use the center of its matching top band as a vertical fold
      axis and the center of its matching left band as a horizontal fold axis.
    - Draw the original rectangle together with its left-right reflection, up-down reflection,
      and double reflection.
    - Any pixel covered by two or more copies of the same rectangle becomes maroon; pixels
      covered by exactly one copy keep the rectangle's own color.
    - Process rectangles by descending area, breaking ties by smaller top row, and keep the
      edge bands unchanged in the final output.

    Key insight:
    The border color bands are legends rather than decoration: each shared color supplies two
    reflection axes, and only self-overlap within one rectangle changes color to maroon.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3 and g.shape[1] >= 3

    top_runs = _parse_edge_runs(g[0], offset=1)
    left_runs = _parse_edge_runs(g[:, 0], offset=1)
    repeated = set(top_runs) & set(left_runs)

    rects = _find_rectangles(g, repeated)
    rects.sort(key=lambda item: (-item["area"], item["r0"], item["c0"]))

    out = g.copy()
    for rect in rects:
        color = int(rect["color"])
        row_axis2 = int(left_runs[color][2])
        col_axis2 = int(top_runs[color][2])
        counts = _object_counts(g.shape, rect, row_axis2, col_axis2)
        out[counts > 0] = color
        out[counts > 1] = MAROON

    out[0, :] = g[0, :]
    out[:, 0] = g[:, 0]
    return out


def _random_odd_partition(count: int, min_total: int, max_total: int, min_len: int = 3, max_len: int = 9) -> list[int] | None:
    lo = max(min_total, count * min_len)
    hi = min(max_total, count * max_len)
    totals = [t for t in range(lo, hi + 1) if t % 2 == count % 2]
    if not totals:
        return None

    total = random.choice(totals)
    remaining = (total - count * min_len) // 2
    cap = (max_len - min_len) // 2
    extra_units = [0] * count
    for _ in range(remaining):
        choices = [i for i, v in enumerate(extra_units) if v < cap]
        if not choices:
            return None
        extra_units[random.choice(choices)] += 1

    lengths = [min_len + 2 * v for v in extra_units]
    random.shuffle(lengths)
    return lengths


def _segment_layout(lengths: list[int], colors: list[int]) -> tuple[list[tuple[int, int, int, int]], dict[int, tuple[int, int, int]]]:
    pos = 1
    seq: list[tuple[int, int, int, int]] = []
    lookup: dict[int, tuple[int, int, int]] = {}
    for length, color in zip(lengths, colors):
        start = pos
        end = pos + length - 1
        axis2 = start + end
        seq.append((int(color), int(start), int(end), int(axis2)))
        lookup[int(color)] = (int(start), int(end), int(axis2))
        pos = end + 1
    return seq, lookup


def _interval_catalog(axis2: int, low: int, high: int, max_len: int = 6) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = {"outer": [], "partial": [], "centered": []}
    for start in range(low, high + 1):
        for end in range(start, min(high, start + max_len - 1) + 1):
            rs, re = _reflect_interval(start, end, axis2)
            if rs < low or re > high:
                continue
            overlap = min(end, re) - max(start, rs) + 1
            if overlap <= 0:
                kind = "outer"
            elif start == rs and end == re:
                kind = "centered"
            else:
                kind = "partial"
            out[kind].append((int(start), int(end)))
    return out


def _input_safe(intervals: list[tuple[int, int]], min_start: int = 2) -> list[tuple[int, int]]:
    return [it for it in intervals if it[0] >= min_start]


def _case_templates(n_repeat: int) -> list[tuple[str, str]]:
    disjoint = ("outer", "outer")
    row_overlap = ("overlap", "outer")
    col_overlap = ("outer", "overlap")
    both_overlap = ("overlap", "overlap")
    if n_repeat == 2:
        cases = random.choice(
            [
                [disjoint, row_overlap],
                [disjoint, col_overlap],
                [disjoint, both_overlap],
            ]
        )
    elif n_repeat == 3:
        cases = random.choice(
            [
                [disjoint, row_overlap, col_overlap],
                [disjoint, row_overlap, both_overlap],
                [disjoint, col_overlap, both_overlap],
            ]
        )
    else:
        cases = [disjoint, row_overlap, col_overlap, both_overlap]
    random.shuffle(cases)
    return list(cases)


def _has_requested_class(spec: str, available: dict[str, list[tuple[int, int]]]) -> bool:
    if spec == "outer":
        return bool(available["outer"])
    return bool(available["partial"] or available["centered"])


def _candidate_count(spec: str, available: dict[str, list[tuple[int, int]]]) -> int:
    if spec == "outer":
        return len(available["outer"])
    return len(available["partial"]) + len(available["centered"])


def _assign_cases(
    colors: list[int],
    cases: list[tuple[str, str]],
    catalogs: dict[int, dict[str, dict[str, list[tuple[int, int]]]]],
) -> list[tuple[int, tuple[str, str]]] | None:
    perms = list(itertools.permutations(colors))
    random.shuffle(perms)
    for perm in perms:
        pairs = list(zip(perm, cases))
        if all(
            _has_requested_class(case[0], catalogs[color]["rows"])
            and _has_requested_class(case[1], catalogs[color]["cols"])
            for color, case in pairs
        ):
            pairs.sort(
                key=lambda item: _candidate_count(item[1][0], catalogs[item[0]]["rows"])
                * _candidate_count(item[1][1], catalogs[item[0]]["cols"])
            )
            return pairs
    return None


def _actual_class(spec: str, available: dict[str, list[tuple[int, int]]]) -> str | None:
    if spec == "outer":
        return "outer" if available["outer"] else None
    choices = [kind for kind in ("partial", "centered") if available[kind]]
    if not choices:
        return None
    return random.choice(choices)


def _placement_ok(occupied: np.ndarray, rect: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = rect
    rr0 = max(1, r0 - 1)
    cc0 = max(1, c0 - 1)
    rr1 = min(occupied.shape[0] - 1, r1 + 1)
    cc1 = min(occupied.shape[1] - 1, c1 + 1)
    return not np.any(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1])


def _render_input(
    shape: tuple[int, int],
    top_segments: list[tuple[int, int, int, int]],
    left_segments: list[tuple[int, int, int, int]],
    objects: list[dict[str, int]],
) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    for color, start, end, _axis2 in top_segments:
        grid[0, start : end + 1] = int(color)
    for color, start, end, _axis2 in left_segments:
        grid[start : end + 1, 0] = int(color)
    for obj in objects:
        grid[obj["r0"] : obj["r1"] + 1, obj["c0"] : obj["c1"] + 1] = int(obj["color"])
    return grid


def _objects_cross_overlap(shape: tuple[int, int], objects: list[dict[str, int]], axes: dict[int, tuple[int, int]]) -> bool:
    masks = []
    for obj in objects:
        row_axis2, col_axis2 = axes[obj["color"]]
        counts = _object_counts(shape, obj, row_axis2, col_axis2)
        masks.append(counts > 0)
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if np.any(masks[i] & masks[j]):
                return True
    return False


def _gen_attempt() -> np.ndarray | None:
    n_repeat = random.randint(2, 4)
    max_top_extra = min(2, 8 - n_repeat)
    n_top_extra = random.randint(0, max_top_extra)
    max_left_extra = min(2, 8 - n_repeat - n_top_extra)
    n_left_extra = random.randint(0, max_left_extra)

    top_count = n_repeat + n_top_extra
    left_count = n_repeat + n_left_extra
    top_lengths = _random_odd_partition(top_count, min_total=16, max_total=29)
    left_lengths = _random_odd_partition(left_count, min_total=16, max_total=29)
    if top_lengths is None or left_lengths is None:
        return None

    pool = list(INPUT_COLORS)
    random.shuffle(pool)
    repeated = pool[:n_repeat]
    cursor = n_repeat
    top_extras = pool[cursor : cursor + n_top_extra]
    cursor += n_top_extra
    left_extras = pool[cursor : cursor + n_left_extra]

    top_colors = repeated + top_extras
    left_colors = repeated + left_extras
    random.shuffle(top_colors)
    random.shuffle(left_colors)

    top_segments, top_lookup = _segment_layout(top_lengths, top_colors)
    left_segments, left_lookup = _segment_layout(left_lengths, left_colors)
    h = 1 + sum(left_lengths)
    w = 1 + sum(top_lengths)

    repeated_colors = [color for color in repeated if color in top_lookup and color in left_lookup]
    if len(repeated_colors) != n_repeat:
        return None

    catalogs: dict[int, dict[str, dict[str, list[tuple[int, int]]]]] = {}
    for color in repeated_colors:
        row_axis2 = left_lookup[color][2]
        col_axis2 = top_lookup[color][2]
        rows = {k: _input_safe(v) for k, v in _interval_catalog(row_axis2, 1, h - 1, max_len=6).items()}
        cols = {k: _input_safe(v) for k, v in _interval_catalog(col_axis2, 1, w - 1, max_len=6).items()}
        catalogs[color] = {"rows": rows, "cols": cols}

    cases = _case_templates(n_repeat)
    assignments = _assign_cases(repeated_colors, cases, catalogs)
    if assignments is None:
        return None

    occupied = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, int]] = []
    for color, case in assignments:
        placed = False
        for _ in range(80):
            row_class = _actual_class(case[0], catalogs[color]["rows"])
            col_class = _actual_class(case[1], catalogs[color]["cols"])
            if row_class is None or col_class is None:
                continue
            row_interval = random.choice(catalogs[color]["rows"][row_class])
            col_interval = random.choice(catalogs[color]["cols"][col_class])
            r0, r1 = row_interval
            c0, c1 = col_interval
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            if area < 4 or area > 20:
                continue
            if max(r1 - r0 + 1, c1 - c0 + 1) < 2:
                continue
            rect = (r0, c0, r1, c1)
            if not _placement_ok(occupied, rect):
                continue
            occupied[r0 : r1 + 1, c0 : c1 + 1] = True
            objects.append(
                {
                    "color": int(color),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "area": int(area),
                }
            )
            placed = True
            break
        if not placed:
            return None

    axes = {color: (left_lookup[color][2], top_lookup[color][2]) for color in repeated_colors}
    want_cross_overlap = len(objects) >= 3 and random.random() < 0.35
    if want_cross_overlap and not _objects_cross_overlap((h, w), objects, axes):
        return None

    grid = _render_input((h, w), top_segments, left_segments, objects)
    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(grid, out):
        return None
    if np.count_nonzero(out == MAROON) == 0:
        return None
    if not np.array_equal(out[0, :], grid[0, :]) or not np.array_equal(out[:, 0], grid[:, 0]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
