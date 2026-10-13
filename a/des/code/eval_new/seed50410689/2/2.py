from __future__ import annotations

import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
GEN_ATTEMPTS = 400
FRAME_PLACE_TRIES = 160


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _smallest_period(seq: list[int]) -> list[int]:
    n = len(seq)
    for p in range(1, n + 1):
        if all(seq[i] == seq[i % p] for i in range(n)):
            return [int(x) for x in seq[:p]]
    raise AssertionError("period must exist")


def _is_legend_row(row: np.ndarray, bg: int) -> bool:
    if row.size < 3 or int(row[0]) == bg:
        return False
    strip = [int(x) for x in row[1:]]
    if any(v == bg for v in strip):
        return False
    if len(set(strip)) != 2:
        return False
    period = _smallest_period(strip)
    return len(period) == 2 and period[0] != period[1]


def _legend_mapping(grid: np.ndarray, bg: int) -> tuple[dict[int, list[int]], int]:
    h, _ = grid.shape
    rows: list[int] = []
    for r in range(h - 1, -1, -1):
        if _is_legend_row(grid[r], bg):
            rows.append(int(r))
        elif rows:
            break
    assert rows, "bottom legend rows must be present"
    rows.reverse()

    mapping: dict[int, list[int]] = {}
    for r in rows:
        label = int(grid[r, 0])
        assert label not in mapping, "legend label colors must be unique"
        mapping[label] = _smallest_period([int(x) for x in grid[r, 1:]])
    return mapping, rows[0]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not bool(mask[r0, c0]) or bool(seen[r0, c0]):
                continue
            stack = [(int(r0), int(c0))]
            seen[r0, c0] = True
            comp: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                comp.append((int(r), int(c)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    coords: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        coords.add((int(r0), int(c)))
        coords.add((int(r1), int(c)))
    for r in range(r0, r1 + 1):
        coords.add((int(r), int(c0)))
        coords.add((int(r), int(c1)))
    return coords


def _frame_specs(grid: np.ndarray, legend_top: int, patterns: dict[int, list[int]]) -> list[dict[str, int]]:
    upper = grid[:legend_top]
    specs: list[dict[str, int]] = []
    for color in patterns:
        for comp in _components(upper == int(color)):
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            comp_set = {(int(r), int(c)) for r, c in comp}
            border = _border_coords(r0, c0, r1, c1)
            if not comp_set.issubset(border):
                continue
            missing = list(border - comp_set)
            if len(missing) != 1:
                continue
            gap_r, gap_c = missing[0]
            if gap_r == r0:
                offset = gap_c - c0
            elif gap_r == r1:
                offset = gap_c - c0
            elif gap_c == c0:
                offset = gap_r - r0
            elif gap_c == c1:
                offset = gap_r - r0
            else:
                continue
            specs.append(
                {
                    "color": int(color),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "offset": int(offset),
                }
            )
    specs.sort(key=lambda spec: (spec["r0"], spec["c0"], spec["color"]))
    return specs


def _fill_from_specs(grid: np.ndarray, specs: list[dict[str, int]], patterns: dict[int, list[int]]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        r0 = spec["r0"]
        c0 = spec["c0"]
        r1 = spec["r1"]
        c1 = spec["c1"]
        offset = spec["offset"]
        pattern = patterns[spec["color"]]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        if inner_h <= 0 or inner_w <= 0:
            continue
        row = np.array([pattern[(offset + i) % len(pattern)] for i in range(inner_w)], dtype=int)
        out[r0 + 1 : r1, c0 + 1 : c1] = np.tile(row, (inner_h, 1))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the inside of gapped rectangular frames using bottom-legend stripe patterns and the gap position as phase.

    Input structure:
    - The background is black.
    - The bottom of the grid contains a horizontal legend made of several full rows; each row starts with one label-color cell, and the remaining cells form an alternating repeating stripe pattern.
    - Above the legend are several single-color rectangular border frames, each with exactly one missing border pixel; a frame's border color matches one legend label color.

    Rule:
    - Read the legend as a mapping from border color to that row's 1D repeating pattern.
    - For each frame, recover its full bounding rectangle and locate the one missing border pixel.
    - Convert the missing pixel position into an offset along its edge, counting left-to-right on horizontal edges or top-to-bottom on vertical edges, starting at 0.
    - Fill the frame interior row-by-row with the matched pattern, starting each row from pattern[offset] and repeating cyclically; keep the border, the gap, and the legend unchanged.

    Key insight:
    The border color selects which pattern to use, but the gap location independently encodes the phase shift that determines how the pattern begins inside that frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    patterns, legend_top = _legend_mapping(g, bg)
    specs = _frame_specs(g, legend_top, patterns)
    return _fill_from_specs(g, specs, patterns)


def _draw_legend(grid: np.ndarray, start_row: int, labels: list[int], patterns: dict[int, tuple[int, int]]) -> None:
    h, w = grid.shape
    assert start_row + len(labels) <= h
    for idx, label in enumerate(labels):
        r = start_row + idx
        a, b = patterns[label]
        grid[r, 0] = int(label)
        for c in range(1, w):
            grid[r, c] = int(a if (c - 1) % 2 == 0 else b)


def _sample_pattern_pairs(labels: list[int]) -> dict[int, tuple[int, int]]:
    used: set[tuple[int, int]] = set()
    pairs: dict[int, tuple[int, int]] = {}
    for label in labels:
        choices = [
            (a, b)
            for a in COLORS
            for b in COLORS
            if a != b and a != label and b != label and (a, b) not in used
        ]
        pair = random.choice(choices)
        used.add(pair)
        pairs[int(label)] = (int(pair[0]), int(pair[1]))
    return pairs


def _can_place(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(occupied.shape[0] - 1, r1 + 1)
    cc1 = min(occupied.shape[1] - 1, c1 + 1)
    return not bool(np.any(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1]))


def _paint_frame(
    grid: np.ndarray,
    color: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    edge: str,
    offset: int,
) -> None:
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)
    if edge == "top":
        grid[r0, c0 + offset] = BG
    elif edge == "bottom":
        grid[r1, c0 + offset] = BG
    elif edge == "left":
        grid[r0 + offset, c0] = BG
    else:
        grid[r0 + offset, c1] = BG


def _build_expected(
    inp: np.ndarray,
    specs: list[dict[str, int]],
    pattern_pairs: dict[int, tuple[int, int]],
) -> np.ndarray:
    patterns = {color: [pair[0], pair[1]] for color, pair in pattern_pairs.items()}
    return _fill_from_specs(inp, specs, patterns)


def _generate_attempt() -> np.ndarray | None:
    n_labels = random.randint(2, 4)
    grid_h = random.randint(max(16, n_labels + 10), 30)
    grid_w = random.randint(14, 30)
    upper_h = grid_h - n_labels
    if upper_h < 8:
        return None

    labels = random.sample(COLORS, n_labels)
    random.shuffle(labels)
    pattern_pairs = _sample_pattern_pairs(labels)

    frame_count = random.randint(n_labels, min(5, n_labels + 2))
    frame_colors = labels + random.choices(labels, k=frame_count - n_labels)
    random.shuffle(frame_colors)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    _draw_legend(grid, upper_h, labels, pattern_pairs)

    occupied = np.zeros((upper_h, grid_w), dtype=bool)
    specs: list[dict[str, int]] = []
    for color in frame_colors:
        placed = False
        for _ in range(FRAME_PLACE_TRIES):
            max_h = min(9, upper_h - 2)
            max_w = min(12, grid_w - 2)
            if max_h < 4 or max_w < 4:
                return None
            outer_h = random.randint(4, max_h)
            outer_w = random.randint(4, max_w)
            max_r0 = upper_h - outer_h - 1
            max_c0 = grid_w - outer_w - 1
            if max_r0 < 1 or max_c0 < 1:
                continue
            r0 = random.randint(1, max_r0)
            c0 = random.randint(1, max_c0)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            if not _can_place(occupied, r0, c0, r1, c1):
                continue

            edge = random.choice(("top", "bottom", "left", "right"))
            if edge in ("top", "bottom"):
                offset = random.randint(1, outer_w - 2)
            else:
                offset = random.randint(1, outer_h - 2)

            _paint_frame(grid, color, r0, c0, r1, c1, edge, offset)
            occupied[max(0, r0 - 1) : min(upper_h, r1 + 2), max(0, c0 - 1) : min(grid_w, c1 + 2)] = True
            specs.append(
                {
                    "color": int(color),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "offset": int(offset),
                }
            )
            placed = True
            break
        if not placed:
            return None

    if _most_common_color(grid) != BG:
        return None

    expected = _build_expected(grid, specs, pattern_pairs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
