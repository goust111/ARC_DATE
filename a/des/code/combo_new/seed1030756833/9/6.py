from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


BG = 0
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
HORIZONTAL = {"left", "right"}
VERTICAL = {"up", "down"}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _marker_cells(r: int, c: int) -> list[tuple[int, int]]:
    return [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]


def _find_markers(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    markers: list[dict] = []
    for r in range(h - 1):
        for c in range(w - 1):
            color = int(g[r, c])
            if color == BG:
                continue
            if np.all(g[r : r + 2, c : c + 2] == color):
                markers.append(
                    {
                        "r": int(r),
                        "c": int(c),
                        "color": color,
                        "cells": set(_marker_cells(r, c)),
                    }
                )
    return markers


def _side_slots(marker: dict) -> list[tuple[str, list[tuple[int, int]]]]:
    r, c = int(marker["r"]), int(marker["c"])
    return [
        ("up", [(r - 1, c), (r - 1, c + 1)]),
        ("down", [(r + 2, c), (r + 2, c + 1)]),
        ("left", [(r, c - 1), (r + 1, c - 1)]),
        ("right", [(r, c + 2), (r + 1, c + 2)]),
    ]


def _segment_from_marker(g: np.ndarray, marker: dict, all_marker_cells: set[tuple[int, int]]) -> dict:
    color = int(marker["color"])
    matches: list[tuple[str, tuple[int, int]]] = []
    for side, slots in _side_slots(marker):
        own = [
            (r, c)
            for r, c in slots
            if _in_bounds(g.shape, r, c) and int(g[r, c]) == color
        ]
        assert len(own) <= 1, "a start block side cannot contain a two-pixel-thick pipe"
        if own:
            matches.append((side, own[0]))

    assert len(matches) == 1, "each 2x2 start block must identify exactly one outgoing pipe"
    direction, start = matches[0]
    dr, dc = DIRS[direction]
    line: list[tuple[int, int]] = []
    r, c = start
    while _in_bounds(g.shape, r, c):
        if (r, c) in all_marker_cells:
            break
        if int(g[r, c]) == BG:
            break
        line.append((int(r), int(c)))
        r += dr
        c += dc

    assert line, "pipe must contain at least one visible cell outside its start block"
    orient = "h" if direction in HORIZONTAL else "v"
    return {
        "color": color,
        "marker": marker,
        "dir": direction,
        "delta": (dr, dc),
        "orient": orient,
        "line": line,
        "line_set": set(line),
    }


def _segments_from_grid(g: np.ndarray) -> list[dict]:
    markers = _find_markers(g)
    assert markers, "expected at least one 2x2 start block"

    all_marker_cells: set[tuple[int, int]] = set()
    for marker in markers:
        assert not (all_marker_cells & marker["cells"]), "start blocks must not overlap"
        all_marker_cells |= marker["cells"]

    segments = [_segment_from_marker(g, marker, all_marker_cells) for marker in markers]
    return segments


def _crossing_sets(segments: list[dict]) -> list[set[tuple[int, int]]]:
    crosses = [set() for _ in segments]
    for i, a in enumerate(segments):
        for j in range(i + 1, len(segments)):
            b = segments[j]
            if int(a["color"]) == int(b["color"]):
                continue
            if a["orient"] == b["orient"]:
                continue
            if a["orient"] == "h":
                h_seg, v_seg = a, b
            else:
                h_seg, v_seg = b, a
            row = int(h_seg["line"][0][0])
            col = int(v_seg["line"][0][1])
            point = (row, col)
            if point in h_seg["line_set"] and point in v_seg["line_set"]:
                crosses[i].add(point)
                crosses[j].add(point)
    return crosses


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shrink each colored pipe toward its marked start by the number of crossings it passes through.

    Input structure:
    - The background is black.
    - The foreground consists of several one-pixel-wide horizontal or vertical colored pipe segments.
    - Each pipe has a visible 2x2 same-color square at its start end and a natural unmarked end.
    - Pipes of different colors may cross; at a crossing the later drawn pipe hides the lower pipe color.

    Rule:
    - For each pipe, read away from its 2x2 start block to recover the straight pipe path, including covered crossing cells.
    - Count the unique crossing positions where that pipe intersects pipes of other colors.
    - Keep the 2x2 start block and the first K+1 pipe cells from the start end, where K is the crossing count.
    - Clear the rest of every pipe to black; if two retained pipe cells still cross, use the visible color from the input at that crossing.

    Key insight:
    - The 2x2 block gives each pipe a direction, and the small count of perpendicular crossings is the length instruction for the shrunken remnant.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    segments = _segments_from_grid(g)
    crosses = _crossing_sets(segments)

    out = np.full(g.shape, BG, dtype=int)
    marker_positions: set[tuple[int, int]] = set()
    for seg in segments:
        color = int(seg["color"])
        for r, c in seg["marker"]["cells"]:
            out[r, c] = color
            marker_positions.add((r, c))

    retained: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    for seg, pts in zip(segments, crosses):
        keep_len = len(pts) + 1
        assert keep_len <= len(seg["line"]), "pipe must be longer than its shrunken remnant"
        for point in seg["line"][:keep_len]:
            if point not in marker_positions:
                retained[point].append(int(seg["color"]))

    for (r, c), colors in retained.items():
        unique = list(dict.fromkeys(colors))
        if len(unique) == 1:
            out[r, c] = unique[0]
        else:
            visible = int(g[r, c])
            out[r, c] = visible if visible in unique else unique[-1]

    return out


# ----------------------------- generator -----------------------------

COLOR_POOL = list(range(1, 10))
GRID_MIN = 16
GRID_MAX = 30
LINE_COUNT_RANGE = (5, 8)
LINE_LEN_MIN = 6
LINE_LEN_MAX = 18
MAX_CROSSES_PER_LINE = 4
MAX_GENERATION_ATTEMPTS = 1200
MAX_PLACEMENT_ATTEMPTS = 140


def _orientation(direction: str) -> str:
    return "h" if direction in HORIZONTAL else "v"


def _random_spec(h: int, w: int, color: int, orient: str) -> dict | None:
    directions = ("left", "right") if orient == "h" else ("up", "down")
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        direction = random.choice(directions)
        dr, dc = DIRS[direction]
        mr = random.randint(0, h - 2)
        mc = random.randint(0, w - 2)
        offset = random.randint(0, 1)

        if direction == "right":
            start = (mr + offset, mc + 2)
        elif direction == "left":
            start = (mr + offset, mc - 1)
        elif direction == "down":
            start = (mr + 2, mc + offset)
        else:
            start = (mr - 1, mc + offset)

        line: list[tuple[int, int]] = []
        r, c = start
        while _in_bounds((h, w), r, c):
            line.append((int(r), int(c)))
            r += dr
            c += dc

        max_len = min(LINE_LEN_MAX, len(line))
        if max_len < LINE_LEN_MIN:
            continue
        length = random.randint(LINE_LEN_MIN, max_len)
        line = line[:length]
        end_after = (line[-1][0] + dr, line[-1][1] + dc)

        return {
            "color": int(color),
            "marker": {"r": mr, "c": mc, "color": int(color), "cells": set(_marker_cells(mr, mc))},
            "dir": direction,
            "delta": (dr, dc),
            "orient": _orientation(direction),
            "line": line,
            "line_set": set(line),
            "after_end": end_after,
        }
    return None


def _line_index(spec: dict, point: tuple[int, int]) -> int:
    return spec["line"].index(point)


def _spec_fits(spec: dict, specs: list[dict]) -> bool:
    marker = set(spec["marker"]["cells"])
    line = set(spec["line"])
    after = spec["after_end"]
    if marker & line:
        return False

    line_occupants: dict[tuple[int, int], list[dict]] = defaultdict(list)
    occupied: set[tuple[int, int]] = set()
    for old in specs:
        old_marker = set(old["marker"]["cells"])
        old_line = set(old["line"])
        occupied |= old_marker | old_line
        for point in old_line:
            line_occupants[point].append(old)

        if marker & old_marker:
            return False
        if marker & old_line or line & old_marker:
            return False
        if _in_bounds((10**9, 10**9), after[0], after[1]) and after in (old_marker | old_line):
            return False
        if old["after_end"] in (marker | line):
            return False

    for point in line:
        occupants = line_occupants.get(point, [])
        if len(occupants) > 1:
            return False
        if not occupants:
            continue
        old = occupants[0]
        if old["orient"] == spec["orient"]:
            return False
        if int(old["color"]) == int(spec["color"]):
            return False
        old_idx = _line_index(old, point)
        new_idx = _line_index(spec, point)
        if old_idx < 2 or new_idx < 2:
            return False
        if old_idx >= len(old["line"]) - 1 or new_idx >= len(spec["line"]) - 1:
            return False

    # Avoid a same-color side-by-side accident that would look like another 2x2 start block.
    for old in specs:
        if int(old["color"]) != int(spec["color"]):
            continue
        old_cells = set(old["marker"]["cells"]) | set(old["line"])
        new_cells = marker | line
        for r, c in new_cells:
            for rr in (r - 1, r):
                for cc in (c - 1, c):
                    block = {(rr, cc), (rr + 1, cc), (rr, cc + 1), (rr + 1, cc + 1)}
                    if (r, c) in block and block <= (old_cells | new_cells):
                        if not (block == marker or block == set(old["marker"]["cells"])):
                            return False

    return True


def _sample_line_colors(total: int) -> list[int]:
    n_colors = random.randint(3, min(5, total))
    counts = [1] * n_colors
    for _ in range(total - n_colors):
        choices = [i for i, count in enumerate(counts) if count < 3]
        counts[random.choice(choices)] += 1
    colors = random.sample(COLOR_POOL, n_colors)
    line_colors: list[int] = []
    for color, count in zip(colors, counts):
        line_colors.extend([color] * count)
    random.shuffle(line_colors)
    return line_colors


def _sample_orientations(total: int) -> list[str]:
    h_count = random.randint(2, total - 2)
    orientations = ["h"] * h_count + ["v"] * (total - h_count)
    random.shuffle(orientations)
    return orientations


def _crossing_sets_for_specs(specs: list[dict]) -> list[set[tuple[int, int]]]:
    return _crossing_sets(specs)


def _render_specs(
    h: int,
    w: int,
    specs: list[dict],
    draw_order: list[int],
    crosses: list[set[tuple[int, int]]] | None = None,
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for idx in draw_order:
        spec = specs[idx]
        color = int(spec["color"])
        for r, c in spec["marker"]["cells"]:
            grid[r, c] = color
        if crosses is None:
            line = spec["line"]
        else:
            line = spec["line"][: len(crosses[idx]) + 1]
        for r, c in line:
            grid[r, c] = color
    return grid


def _background_dominates(grid: np.ndarray) -> bool:
    return int(np.count_nonzero(grid == BG)) > int(grid.size // 2)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    total = random.randint(*LINE_COUNT_RANGE)
    colors = _sample_line_colors(total)
    orientations = _sample_orientations(total)

    specs: list[dict] = []
    for color, orient in zip(colors, orientations):
        placed = False
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            spec = _random_spec(h, w, color, orient)
            if spec is None:
                continue
            if _spec_fits(spec, specs):
                specs.append(spec)
                placed = True
                break
        if not placed:
            return None

    crosses = _crossing_sets_for_specs(specs)
    counts = [len(points) for points in crosses]
    if not any(count == 0 for count in counts):
        return None
    if not any(count > 0 for count in counts):
        return None
    if max(counts) > MAX_CROSSES_PER_LINE:
        return None
    if any(len(spec["line"]) <= count + 1 for spec, count in zip(specs, counts)):
        return None

    crossing_points = set().union(*crosses) if crosses else set()
    if not (1 <= len(crossing_points) <= 8):
        return None

    draw_order = list(range(len(specs)))
    random.shuffle(draw_order)
    inp = _render_specs(h, w, specs, draw_order)
    expected = _render_specs(h, w, specs, draw_order, crosses)

    if not _background_dominates(inp):
        return None
    if len(_find_markers(inp)) != len(specs):
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-network puzzle input")
