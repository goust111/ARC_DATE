import random
from collections import Counter
from itertools import permutations

import numpy as np


LINE_KINDS = ("H", "V", "D", "U")
MIN_RAW_LINE_LEN = 7
MIN_VISIBLE_LINE_PIXELS = 3
MIN_GRID_SIZE = 12
MAX_GRID_SIZE = 18
MIN_LINE_COUNT = 3
MAX_LINE_COUNT = 5
GENERATION_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _line_value(kind: str, r: int, c: int) -> int:
    if kind == "H":
        return int(r)
    if kind == "V":
        return int(c)
    if kind == "D":
        return int(r - c)
    if kind == "U":
        return int(r + c)
    raise ValueError(kind)


def _line_cells(shape: tuple[int, int], kind: str, value: int) -> frozenset[tuple[int, int]]:
    h, w = shape
    cells: list[tuple[int, int]] = []
    if kind == "H":
        if 0 <= value < h:
            cells = [(value, c) for c in range(w)]
    elif kind == "V":
        if 0 <= value < w:
            cells = [(r, value) for r in range(h)]
    elif kind == "D":
        for r in range(h):
            c = r - value
            if 0 <= c < w:
                cells.append((r, c))
    elif kind == "U":
        for r in range(h):
            c = value - r
            if 0 <= c < w:
                cells.append((r, c))
    else:
        raise ValueError(kind)
    return frozenset(cells)


def _candidate_lines(shape: tuple[int, int]) -> list[dict]:
    h, w = shape
    candidates: list[dict] = []

    for row in range(h):
        cells = _line_cells(shape, "H", row)
        if len(cells) >= MIN_RAW_LINE_LEN:
            candidates.append({"kind": "H", "value": row, "cells": cells})
    for col in range(w):
        cells = _line_cells(shape, "V", col)
        if len(cells) >= MIN_RAW_LINE_LEN:
            candidates.append({"kind": "V", "value": col, "cells": cells})

    for value in range(-(w - 1), h):
        cells = _line_cells(shape, "D", value)
        if len(cells) >= MIN_RAW_LINE_LEN:
            candidates.append({"kind": "D", "value": value, "cells": cells})
    for value in range(h + w - 1):
        cells = _line_cells(shape, "U", value)
        if len(cells) >= MIN_RAW_LINE_LEN:
            candidates.append({"kind": "U", "value": value, "cells": cells})

    return candidates


def _crosses(line_a: dict, line_b: dict) -> bool:
    return bool(line_a["cells"] & line_b["cells"])


def _render(shape: tuple[int, int], bg: int, ordered_lines: list[dict]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for line in ordered_lines:
        for r, c in line["cells"]:
            out[r, c] = int(line["color"])
    return out


def _valid_orders(colors: list[int], below_edges: set[tuple[int, int]]) -> list[tuple[int, ...]]:
    valid = []
    for order in permutations(colors):
        pos = {color: i for i, color in enumerate(order)}
        if all(pos[lower] < pos[upper] for lower, upper in below_edges):
            valid.append(order)
    return valid


def _infer_line(points: list[tuple[int, int]], shape: tuple[int, int]) -> dict:
    if len(points) < MIN_VISIBLE_LINE_PIXELS:
        raise AssertionError("each line needs at least three visible pixels")

    candidates: list[dict] = []
    for kind in LINE_KINDS:
        grouped: dict[int, list[tuple[int, int]]] = {}
        for pt in points:
            grouped.setdefault(_line_value(kind, pt[0], pt[1]), []).append(pt)
        for value, on_line in grouped.items():
            off_line = [pt for pt in points if pt not in on_line]
            if len(on_line) < MIN_VISIBLE_LINE_PIXELS or len(off_line) > 1:
                continue
            full = _line_cells(shape, kind, value)
            if not all(pt in full for pt in on_line):
                continue
            candidates.append(
                {
                    "kind": kind,
                    "value": value,
                    "line_points": tuple(sorted(on_line)),
                    "marker": off_line[0] if off_line else None,
                }
            )

    if not candidates:
        raise AssertionError("failed to infer a line")

    best_count = max(len(c["line_points"]) for c in candidates)
    best = [c for c in candidates if len(c["line_points"]) == best_count]
    keyset = {(c["kind"], c["value"], c["marker"]) for c in best}
    if len(keyset) != 1:
        raise AssertionError("ambiguous line inference")
    return best[0]


def _recover_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    lines: dict[int, frozenset[tuple[int, int]]] = {}
    target_color = None
    marker = None

    for color in colors:
        pts = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        info = _infer_line(pts, g.shape)
        line_cells = _line_cells(g.shape, info["kind"], info["value"])
        lines[color] = line_cells
        if info["marker"] is not None:
            assert target_color is None, "expected exactly one marked color"
            target_color = color
            marker = tuple(map(int, info["marker"]))

    assert target_color is not None and marker is not None, "missing isolated marker"
    assert all(marker not in cells for cells in lines.values()), "marker cannot lie on any line"

    below_edges: set[tuple[int, int]] = set()
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if (r, c) == marker:
                assert g[r, c] == target_color
                continue
            covering = [color for color, cells in lines.items() if (r, c) in cells]
            visible = int(g[r, c])
            if not covering:
                assert visible == bg
                continue
            assert visible in covering
            for color in covering:
                if color != visible:
                    below_edges.add((color, visible))

    valid = _valid_orders(colors, below_edges)
    assert len(valid) == 1, "line order must be uniquely recoverable"
    return {
        "bg": bg,
        "lines": lines,
        "target_color": target_color,
        "marker": marker,
        "order": valid[0],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Rollback the overlaid line scene to the marked line's layer, restoring hidden pixels below deleted layers.

    Input structure:
    - The background is the dominant color.
    - Every foreground color is the visible part of one full one-cell-thick horizontal, vertical, or 45-degree diagonal line, with higher lines covering lower ones at crossings.
    - Exactly one extra isolated marker pixel has the same color as one line and does not lie on any line.

    Rule:
    - Recover every full line from its same-color visible collinear pixels, treating the lone off-line pixel of the marked color as the marker.
    - Use visible crossing colors to infer the unique bottom-to-top layer order: at each crossing, the visible line is above every other line through that cell.
    - Delete all lines strictly above the marked line and erase the marker pixel itself.
    - Redraw the remaining lines in bottom-to-top order so the marked line becomes the new top layer and any pixels previously hidden only by deleted layers reappear.

    Key insight:
    - The task is not local erasure: you must reconstruct full line geometry and global occlusion order first, then rerender the surviving stack to restore covered line segments.
    """
    scene = _recover_scene(np.asarray(grid, dtype=int))
    order = list(scene["order"])
    target_index = order.index(scene["target_color"])
    kept_colors = order[: target_index + 1]
    ordered_lines = [
        {"color": color, "cells": scene["lines"][color]}
        for color in kept_colors
    ]
    return _render(grid.shape, scene["bg"], ordered_lines)


def _choose_colors(line_count: int) -> tuple[int, list[int]]:
    palette = list(range(10))
    bg = random.choice(palette)
    others = [c for c in palette if c != bg]
    random.shuffle(others)
    return bg, others[:line_count]


def _build_line_order(shape: tuple[int, int], line_count: int) -> list[dict] | None:
    candidates = _candidate_lines(shape)
    random.shuffle(candidates)

    for _ in range(120):
        first = random.choice(candidates)
        order = [first]
        used = {(first["kind"], first["value"])}
        cover_counts = Counter(first["cells"])

        while len(order) < line_count:
            prev = order[-1]
            pool = []
            for cand in candidates:
                key = (cand["kind"], cand["value"])
                if key in used:
                    continue
                if not _crosses(prev, cand):
                    continue
                if any(cover_counts[cell] >= 2 for cell in cand["cells"]):
                    continue
                pool.append(cand)
            if not pool:
                break
            random.shuffle(pool)
            chosen = pool[0]
            order.append(chosen)
            used.add((chosen["kind"], chosen["value"]))
            cover_counts.update(chosen["cells"])

        if len(order) == line_count:
            return order
    return None


def _visible_count(grid: np.ndarray, color: int, cells: frozenset[tuple[int, int]]) -> int:
    return sum(1 for r, c in cells if int(grid[r, c]) == color)


def _marker_candidates(shape: tuple[int, int], occupied: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = shape
    candidates = []
    for r in range(h):
        for c in range(w):
            if (r, c) in occupied:
                continue
            isolated = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) in occupied:
                        isolated = False
                        break
                if not isolated:
                    break
            if isolated:
                candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _expected_output(shape: tuple[int, int], bg: int, ordered_lines: list[dict], target_index: int) -> np.ndarray:
    keep = ordered_lines[: target_index + 1]
    return _render(shape, bg, keep)


def _crossing_edges_from_order(ordered_lines: list[dict]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for i, lower in enumerate(ordered_lines):
        for upper in ordered_lines[i + 1 :]:
            if lower["cells"] & upper["cells"]:
                edges.add((lower["color"], upper["color"]))
    return edges


def _semantic_checks(inp: np.ndarray, ordered_lines: list[dict], target_index: int, marker: tuple[int, int], bg: int) -> bool:
    expected = _expected_output(inp.shape, bg, ordered_lines, target_index)
    target_color = ordered_lines[target_index]["color"]
    removed_colors = {line["color"] for line in ordered_lines[target_index + 1 :]}

    for line in ordered_lines:
        if _visible_count(inp, line["color"], line["cells"]) < MIN_VISIBLE_LINE_PIXELS:
            return False

    edges = _crossing_edges_from_order(ordered_lines)
    valid = _valid_orders([line["color"] for line in ordered_lines], edges)
    if len(valid) != 1 or valid[0] != tuple(line["color"] for line in ordered_lines):
        return False

    restored_lower = []
    for i in range(target_index):
        for j in range(target_index + 1, len(ordered_lines)):
            inter = ordered_lines[i]["cells"] & ordered_lines[j]["cells"]
            for cell in inter:
                restored_lower.append((cell, ordered_lines[i]["color"]))
    if not restored_lower:
        return False

    naive = inp.copy()
    for color in removed_colors:
        naive[naive == color] = bg
    naive[marker] = bg
    if np.array_equal(naive, expected):
        return False
    if not any(int(inp[r, c]) != lower_color and int(expected[r, c]) == lower_color for (r, c), lower_color in restored_lower):
        return False

    try:
        recovered = _recover_scene(inp)
    except AssertionError:
        return False
    if recovered["target_color"] != target_color:
        return False
    if recovered["marker"] != marker:
        return False
    if recovered["order"] != tuple(line["color"] for line in ordered_lines):
        return False
    if not np.array_equal(main(inp), expected):
        return False
    return True


def _generate_case() -> np.ndarray | None:
    size = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    shape = (size, size)
    line_count = random.randint(MIN_LINE_COUNT, MAX_LINE_COUNT)
    ordered_templates = _build_line_order(shape, line_count)
    if ordered_templates is None:
        return None

    valid_targets = []
    for target_index in range(1, line_count - 1):
        has_lower_restoration = any(
            ordered_templates[i]["cells"] & ordered_templates[j]["cells"]
            for i in range(target_index)
            for j in range(target_index + 1, line_count)
        )
        if has_lower_restoration:
            valid_targets.append(target_index)
    if not valid_targets:
        return None
    target_index = random.choice(valid_targets)

    bg, colors = _choose_colors(line_count)
    ordered_lines = []
    for template, color in zip(ordered_templates, colors):
        ordered_lines.append(
            {
                "kind": template["kind"],
                "value": template["value"],
                "cells": template["cells"],
                "color": color,
            }
        )

    inp = _render(shape, bg, ordered_lines)
    occupied = set().union(*(line["cells"] for line in ordered_lines))
    target_line = ordered_lines[target_index]

    marker = None
    for cell in _marker_candidates(shape, occupied):
        if _line_value(target_line["kind"], cell[0], cell[1]) == target_line["value"]:
            continue
        trial = inp.copy()
        trial[cell] = target_line["color"]
        if _semantic_checks(trial, ordered_lines, target_index, cell, bg):
            marker = cell
            inp = trial
            break
    if marker is None:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        case = _generate_case()
        if case is not None:
            return case
    raise RuntimeError("failed to generate a valid rollback-line puzzle")
