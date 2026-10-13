import numpy as np


BG = 0
RED = 2
BLUE = 1
GREEN = 3
YELLOW = 4

DIR_BY_BREAK = {
    RED: (-1, 0),
    BLUE: (1, 0),
    GREEN: (0, -1),
    YELLOW: (0, 1),
}
BREAK_COLORS = tuple(DIR_BY_BREAK.keys())
BRIDGE_COLOR_POOL = (5, 6, 7, 8, 9)

MIN_GRID = 18
MAX_GRID = 30
MIN_BRIDGES = 3
MAX_BRIDGES = len(BRIDGE_COLOR_POOL)
MAX_GEN_ATTEMPTS = 500
MAX_CANDIDATE_ATTEMPTS = 120
MAX_PLACEMENT_ATTEMPTS = 200


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bridge_line(color: int, orientation: str, length: int) -> list[tuple[int, int]]:
    if orientation == "h":
        return [(0, idx) for idx in range(length)]
    return [(idx, 0) for idx in range(length)]


def _segment_runs(seq: list[int], color: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    idx = 0
    while idx < len(seq):
        if int(seq[idx]) != int(color):
            idx += 1
            continue
        start = idx
        while idx + 1 < len(seq) and int(seq[idx + 1]) == int(color):
            idx += 1
        runs.append((start, idx))
        idx += 1
    return runs


def _nearest_break_color(seq: list[int], color: int, start: int, end: int) -> int:
    for idx in range(end + 1, len(seq)):
        if int(seq[idx]) != int(color):
            return int(seq[idx])
    for idx in range(start - 1, -1, -1):
        if int(seq[idx]) != int(color):
            return int(seq[idx])
    raise AssertionError("broken bridge segment must have a nearby breakpoint")


def _local_output_list(seq: list[int], color: int, orientation: str) -> list[tuple[int, int]]:
    length = len(seq)
    positions = _bridge_line(color, orientation, length)
    if all(int(v) == int(color) for v in seq):
        return list(positions)

    out = [positions[0], positions[-1]]
    for start, end in _segment_runs(seq, color):
        seg_len = end - start + 1
        brk = _nearest_break_color(seq, color, start, end)
        dr, dc = DIR_BY_BREAK[int(brk)]
        for idx in range(start, end + 1):
            if idx == 0 or idx == length - 1:
                continue
            r, c = positions[idx]
            out.append((r + dr * seg_len, c + dc * seg_len))
    return out


def _color_set(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    exclude = exclude or set()
    return [int(v) for v in np.unique(grid) if int(v) not in exclude]


def _extract_bridge(grid: np.ndarray, color: int) -> tuple[str, list[tuple[int, int]], list[int]]:
    coords = np.argwhere(grid == int(color))
    rows = np.unique(coords[:, 0])
    cols = np.unique(coords[:, 1])

    if len(rows) == 1:
        r = int(rows[0])
        c0 = int(cols.min())
        c1 = int(cols.max())
        positions = [(r, c) for c in range(c0, c1 + 1)]
        seq = [int(grid[r, c]) for c in range(c0, c1 + 1)]
        orientation = "h"
    elif len(cols) == 1:
        c = int(cols[0])
        r0 = int(rows.min())
        r1 = int(rows.max())
        positions = [(r, c) for r in range(r0, r1 + 1)]
        seq = [int(grid[r, c]) for r in range(r0, r1 + 1)]
        orientation = "v"
    else:
        raise AssertionError("bridge color must occupy a single row or column")

    if int(seq[0]) != int(color) or int(seq[-1]) != int(color):
        raise AssertionError("anchors must sit at the bridge endpoints")
    for value in seq:
        if int(value) != int(color) and int(value) not in DIR_BY_BREAK:
            raise AssertionError("bridge interior may only contain bridge color or breakpoint colors")
    return orientation, positions, seq


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detach bridge segments according to breakpoint-color directions while keeping anchor pixels fixed.

    Input structure:
    - A black-background grid contains several monochrome bridge colors that are not the breakpoint colors.
    - Each bridge color has exactly two endpoint anchor pixels connected by a 1-cell-wide straight horizontal or vertical line.
    - Some bridge interiors replace one or more line cells with breakpoint colors: red, blue, green, or yellow.

    Rule:
    - Any bridge with no breakpoint stays unchanged.
    - For a broken bridge, erase every breakpoint cell and clear the original bridge cells except the two anchors.
    - Split the bridge into maximal same-color segments. Each segment uses the nearest breakpoint on its right or lower side; if none exists there, use the nearest breakpoint on its left or upper side.
    - Translate the non-anchor pixels of that segment by exactly the segment length in the direction encoded by that breakpoint color: red up, blue down, green left, yellow right. Paint translated pixels in the bridge color and keep both anchors in place.

    Key insight:
    Breakpoint colors are a visible direction code, and each maximal bridge fragment moves independently by a distance equal to its own length.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    bridge_colors = _color_set(g, exclude={bg, *BREAK_COLORS})
    for color in bridge_colors:
        _orientation, positions, seq = _extract_bridge(g, int(color))
        if all(int(v) == int(color) for v in seq):
            continue

        for idx, (r, c) in enumerate(positions):
            out[r, c] = int(color) if idx in (0, len(positions) - 1) else bg

        for start, end in _segment_runs(seq, int(color)):
            seg_len = end - start + 1
            brk = _nearest_break_color(seq, int(color), start, end)
            dr, dc = DIR_BY_BREAK[int(brk)]
            for idx in range(start, end + 1):
                if idx == 0 or idx == len(seq) - 1:
                    continue
                r, c = positions[idx]
                rr = r + dr * seg_len
                cc = c + dc * seg_len
                if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                    out[rr, cc] = int(color)

    return out


def _sample_break_counts(rng: np.random.Generator, broken_count: int) -> list[int]:
    counts = [1 + int(rng.random() < 0.45) for _ in range(broken_count)]
    if sum(counts) < 2 and broken_count >= 1:
        counts[0] = 2
    return counts


def _assign_break_colors(rng: np.random.Generator, total_breaks: int) -> list[int]:
    if total_breaks == 0:
        return []
    colors: list[int] = []
    distinct = min(4, total_breaks)
    colors.extend(list(rng.permutation(np.array(BREAK_COLORS, dtype=int)))[:distinct])
    while len(colors) < total_breaks:
        colors.append(int(rng.choice(BREAK_COLORS)))
    rng.shuffle(colors)
    return colors


def _build_sequence(
    rng: np.random.Generator,
    color: int,
    broken: bool,
    break_colors: list[int],
) -> list[int] | None:
    if not broken:
        length = int(rng.integers(5, 11))
        return [int(color)] * length

    n_breaks = len(break_colors)
    if n_breaks == 0:
        return None

    seg_lengths: list[int] = []
    for idx in range(n_breaks + 1):
        if idx == 0 or idx == n_breaks:
            seg_lengths.append(int(rng.integers(2, 5)))
        else:
            seg_lengths.append(int(rng.integers(1, 4)))

    total_length = sum(seg_lengths) + n_breaks
    if total_length > 12:
        return None

    seq: list[int] = []
    for idx, seg_len in enumerate(seg_lengths):
        seq.extend([int(color)] * seg_len)
        if idx < n_breaks:
            seq.append(int(break_colors[idx]))
    return seq


def _build_candidate(
    rng: np.random.Generator,
    color: int,
    orientation: str,
    broken: bool,
    break_colors: list[int],
) -> dict | None:
    for _ in range(MAX_CANDIDATE_ATTEMPTS):
        seq = _build_sequence(rng, int(color), bool(broken), break_colors)
        if seq is None:
            continue

        input_cells = {(r, c): int(v) for (r, c), v in zip(_bridge_line(int(color), orientation, len(seq)), seq)}
        output_list = _local_output_list(seq, int(color), orientation)
        output_cells = set(output_list)

        if len(output_cells) == 0 or len(output_list) != len(output_cells):
            continue

        if broken:
            anchor_a = (0, 0)
            anchor_b = (0, len(seq) - 1) if orientation == "h" else (len(seq) - 1, 0)
            moved_only = set(output_cells) - {anchor_a, anchor_b}
            if not moved_only:
                continue

        all_coords = set(input_cells.keys()) | set(output_cells)
        rows = [r for r, _ in all_coords]
        cols = [c for _, c in all_coords]
        span_h = max(rows) - min(rows) + 1
        span_w = max(cols) - min(cols) + 1
        if span_h > 14 or span_w > 14:
            continue

        return {
            "color": int(color),
            "orientation": orientation,
            "seq": seq,
            "input_cells": input_cells,
            "output_cells": output_cells,
            "bbox": (min(rows), min(cols), max(rows), max(cols)),
            "size_hint": max(span_h, span_w),
        }
    return None


def _dilate_points(points: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    blocked: set[tuple[int, int]] = set()
    for r, c in points:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    blocked.add((rr, cc))
    return blocked


def _shift_points(points: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in points}


def _shift_colored(points: dict[tuple[int, int], int], dr: int, dc: int) -> dict[tuple[int, int], int]:
    return {(r + dr, c + dc): int(v) for (r, c), v in points.items()}


def _place_candidates(rng: np.random.Generator, h: int, w: int, candidates: list[dict]) -> tuple[dict, dict] | None:
    input_colored: dict[tuple[int, int], int] = {}
    output_colored: dict[tuple[int, int], int] = {}
    input_blocked: set[tuple[int, int]] = set()
    output_blocked: set[tuple[int, int]] = set()

    ordered = sorted(candidates, key=lambda item: int(item["size_hint"]), reverse=True)
    for cand in ordered:
        r0, c0, r1, c1 = cand["bbox"]
        min_r = -int(r0)
        max_r = h - 1 - int(r1)
        min_c = -int(c0)
        max_c = w - 1 - int(c1)
        if min_r > max_r or min_c > max_c:
            return None

        placed = False
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            dr = int(rng.integers(min_r, max_r + 1))
            dc = int(rng.integers(min_c, max_c + 1))
            shifted_input = _shift_colored(cand["input_cells"], dr, dc)
            shifted_output = _shift_points(cand["output_cells"], dr, dc)
            input_points = set(shifted_input.keys())

            if input_points & input_blocked:
                continue
            if shifted_output & output_blocked:
                continue

            for key, value in shifted_input.items():
                input_colored[key] = int(value)
            for key in shifted_output:
                output_colored[key] = int(cand["color"])

            input_blocked |= _dilate_points(input_points, (h, w))
            output_blocked |= _dilate_points(shifted_output, (h, w))
            placed = True
            break

        if not placed:
            return None

    return input_colored, output_colored


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = int(rng.integers(MIN_GRID, MAX_GRID + 1))
    w = int(rng.integers(MIN_GRID, MAX_GRID + 1))

    n_bridges = int(rng.integers(MIN_BRIDGES, MAX_BRIDGES + 1))
    colors = list(map(int, rng.choice(BRIDGE_COLOR_POOL, size=n_bridges, replace=False)))

    broken_count = int(rng.integers(1, n_bridges))
    broken_flags = [True] * broken_count + [False] * (n_bridges - broken_count)
    rng.shuffle(broken_flags)

    orientations = ["h"] * (n_bridges // 2) + ["v"] * (n_bridges - n_bridges // 2)
    rng.shuffle(orientations)
    if "h" not in orientations:
        orientations[0] = "h"
    if "v" not in orientations:
        orientations[-1] = "v"

    break_counts = _sample_break_counts(rng, broken_count)
    flat_break_colors = _assign_break_colors(rng, sum(break_counts))

    candidates: list[dict] = []
    color_idx = 0
    break_idx = 0
    broken_seen = 0
    for broken, orientation in zip(broken_flags, orientations):
        if broken:
            count = break_counts[broken_seen]
            brk_colors = flat_break_colors[break_idx : break_idx + count]
            break_idx += count
            broken_seen += 1
        else:
            brk_colors = []

        cand = _build_candidate(
            rng,
            int(colors[color_idx]),
            orientation,
            bool(broken),
            list(brk_colors),
        )
        color_idx += 1
        if cand is None:
            return None
        candidates.append(cand)

    placed = _place_candidates(rng, h, w, candidates)
    if placed is None:
        return None
    input_cells, output_cells = placed

    inp = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)

    for (r, c), value in input_cells.items():
        inp[r, c] = int(value)
    for (r, c), value in output_cells.items():
        expected[r, c] = int(value)

    if len(_color_set(inp, exclude={BG})) != n_bridges + len(set(flat_break_colors)):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
