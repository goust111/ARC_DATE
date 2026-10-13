import random
from collections import Counter, deque

import numpy as np


MARKER_COLOR = 5
DIR_VECS = {
    "U": (-1, 0),
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
}
DIR_ORDER = ("U", "R", "D", "L")

GRID_H_RANGE = (12, 20)
GRID_W_RANGE = (12, 20)
PAIR_COUNT_RANGE = (2, 4)
SEGMENT_LEN_RANGE = (2, 6)
GAP_RANGE = (1, 5)
GEN_RETRIES = 400
PLACEMENT_RETRIES = 120


def _bg_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).flatten())).most_common(1)[0][0])


def _move(point: tuple[int, int], direction: str, steps: int = 1) -> tuple[int, int]:
    dr, dc = DIR_VECS[direction]
    return int(point[0] + dr * steps), int(point[1] + dc * steps)


def _opposite(direction: str) -> str:
    return DIR_ORDER[(DIR_ORDER.index(direction) + 2) % 4]


def _left_of(direction: str) -> str:
    return DIR_ORDER[(DIR_ORDER.index(direction) - 1) % 4]


def _right_of(direction: str) -> str:
    return DIR_ORDER[(DIR_ORDER.index(direction) + 1) % 4]


def _turn_name(dir1: str, dir2: str) -> str:
    delta = (DIR_ORDER.index(dir2) - DIR_ORDER.index(dir1)) % 4
    if delta == 1:
        return "right"
    if delta == 3:
        return "left"
    raise AssertionError(f"Expected a bend, got {dir1} -> {dir2}")


def _side_cell(point: tuple[int, int], travel_dir: str, side: str) -> tuple[int, int]:
    assert side in ("left", "right")
    side_dir = _left_of(travel_dir) if side == "left" else _right_of(travel_dir)
    return _move(point, side_dir)


def _components_for_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = np.asarray(grid) == int(color)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for dr, dc in DIR_VECS.values():
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(pts)
    return comps


def _segment_info(grid: np.ndarray, color: int, points: list[tuple[int, int]]) -> dict:
    rows = sorted({int(r) for r, _ in points})
    cols = sorted({int(c) for _, c in points})
    if len(rows) == 1:
        orient = "H"
        row = rows[0]
        c0, c1 = cols[0], cols[-1]
        expected = {(int(row), int(c)) for c in range(int(c0), int(c1) + 1)}
        assert expected == set(points)
        endpoints = [(int(row), int(c0)), (int(row), int(c1))]
    elif len(cols) == 1:
        orient = "V"
        col = cols[0]
        r0, r1 = rows[0], rows[-1]
        expected = {(int(r), int(col)) for r in range(int(r0), int(r1) + 1)}
        assert expected == set(points)
        endpoints = [(int(r0), int(col)), (int(r1), int(col))]
    else:
        raise AssertionError(f"Color {color} component is not a straight segment")

    h, w = grid.shape
    marker_flags = []
    marker_neighbors = []
    for r, c in endpoints:
        neighbors = []
        for dr, dc in DIR_VECS.values():
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == int(MARKER_COLOR):
                neighbors.append((int(nr), int(nc)))
        marker_flags.append(bool(neighbors))
        marker_neighbors.append(neighbors)
    assert sum(marker_flags) == 1, f"Color {color} needs exactly one marked endpoint"

    outer_idx = marker_flags.index(True)
    inner_idx = 1 - outer_idx
    outer = endpoints[outer_idx]
    inner = endpoints[inner_idx]

    if orient == "H":
        outer_dir = "R" if outer[1] < inner[1] else "L"
    else:
        outer_dir = "D" if outer[0] < inner[0] else "U"

    return {
        "color": int(color),
        "orient": orient,
        "points": sorted(points),
        "outer": outer,
        "inner": inner,
        "outer_dir": outer_dir,
        "marker_neighbor": marker_neighbors[outer_idx][0],
    }


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        row = a[0]
        c0, c1 = sorted((a[1], b[1]))
        return [(int(row), int(c)) for c in range(int(c0), int(c1) + 1)]
    if a[1] == b[1]:
        col = a[1]
        r0, r1 = sorted((a[0], b[0]))
        return [(int(r), int(col)) for r in range(int(r0), int(r1) + 1)]
    raise AssertionError(f"Cannot draw non-axis-aligned line from {a} to {b}")


def _pair_plan(seg_a: dict, seg_b: dict) -> dict:
    if seg_a["orient"] == seg_b["orient"]:
        raise AssertionError("Paired segments must be perpendicular")

    ra, ca = seg_a["inner"]
    rb, cb = seg_b["inner"]
    if ra == rb and ca != cb:
        if seg_a["orient"] == "V":
            first, second = seg_a, seg_b
        else:
            first, second = seg_b, seg_a
    elif ca == cb and ra != rb:
        if seg_a["orient"] == "H":
            first, second = seg_a, seg_b
        else:
            first, second = seg_b, seg_a
    else:
        raise AssertionError("Paired inner endpoints must align on exactly one axis")

    corner = tuple(first["inner"])
    connector = _line_cells(corner, tuple(second["inner"]))
    dir1 = str(first["outer_dir"])
    dir2 = _opposite(str(second["outer_dir"]))
    turn = _turn_name(dir1, dir2)
    return {
        "color": int(first["color"]),
        "first": first,
        "second": second,
        "corner": corner,
        "connector": connector,
        "dir1": dir1,
        "dir2": dir2,
        "turn": turn,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconnect each color's two path fragments into an L path, then move both grey
    endpoint markers to the side determined by the bend handedness.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    out = g.copy()
    out[out == int(MARKER_COLOR)] = int(bg)

    path_colors = [
        int(color)
        for color in sorted(set(map(int, g.flatten())))
        if int(color) not in (int(bg), int(MARKER_COLOR))
    ]

    for color in path_colors:
        comps = _components_for_color(g, int(color))
        assert len(comps) == 2, f"Color {color} should have exactly two fragments"
        seg_a = _segment_info(g, int(color), comps[0])
        seg_b = _segment_info(g, int(color), comps[1])
        plan = _pair_plan(seg_a, seg_b)

        for r, c in plan["connector"]:
            out[int(r), int(c)] = int(color)

        target_side = "right" if plan["turn"] == "left" else "left"
        p1 = _side_cell(tuple(plan["first"]["outer"]), plan["dir1"], target_side)
        p2 = _side_cell(tuple(plan["second"]["outer"]), plan["dir2"], target_side)
        out[int(p1[0]), int(p1[1])] = int(MARKER_COLOR)
        out[int(p2[0]), int(p2[1])] = int(MARKER_COLOR)

    return out


def _expand_cells(cells: set[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    expanded.add((int(rr), int(cc)))
    return expanded


def _object_template(dir1: str, turn: str, len1: int, len2: int, gap: int) -> dict:
    dir2 = _left_of(dir1) if turn == "left" else _right_of(dir1)
    corner = (0, 0)

    seg1 = [_move(corner, _opposite(dir1), k) for k in range(len1)]
    seg1.reverse()
    seg2 = [_move(corner, dir2, gap + i) for i in range(1, len2 + 1)]
    connector = [_move(corner, dir2, i) for i in range(1, gap + 1)]

    outer1 = seg1[0]
    outer2 = seg2[-1]

    source_side = "left" if turn == "left" else "right"
    target_side = "right" if turn == "left" else "left"

    src1 = _side_cell(outer1, dir1, source_side)
    src2 = _side_cell(outer2, dir2, source_side)
    dst1 = _side_cell(outer1, dir1, target_side)
    dst2 = _side_cell(outer2, dir2, target_side)

    input_cells = set(seg1) | set(seg2) | {src1, src2}
    all_cells = set(seg1) | set(connector) | set(seg2) | {src1, src2, dst1, dst2}
    assert len(input_cells) == len(seg1) + len(seg2) + 2
    assert len({src1, src2, dst1, dst2}) == 4
    assert not ({src1, src2, dst1, dst2} & (set(seg1) | set(connector) | set(seg2)))

    return {
        "dir1": dir1,
        "dir2": dir2,
        "turn": turn,
        "seg1": seg1,
        "seg2": seg2,
        "connector": connector,
        "input_cells": input_cells,
        "all_cells": all_cells,
        "sources": [src1, src2],
        "targets": [dst1, dst2],
    }


def _translate_cells(cells, offset_r: int, offset_c: int) -> list[tuple[int, int]]:
    return [(int(r + offset_r), int(c + offset_c)) for r, c in cells]


def _place_object(
    h: int,
    w: int,
    blocked: set[tuple[int, int]],
    color: int,
    turn: str,
) -> dict | None:
    dir1 = random.choice(DIR_ORDER)
    len1 = random.randint(*SEGMENT_LEN_RANGE)
    len2 = random.randint(*SEGMENT_LEN_RANGE)
    gap = random.randint(*GAP_RANGE)
    template = _object_template(dir1, turn, len1, len2, gap)

    rs = [r for r, _ in template["all_cells"]]
    cs = [c for _, c in template["all_cells"]]
    min_r, max_r = min(rs), max(rs)
    min_c, max_c = min(cs), max(cs)

    row_lo = -min_r
    row_hi = h - 1 - max_r
    col_lo = -min_c
    col_hi = w - 1 - max_c
    if row_lo > row_hi or col_lo > col_hi:
        return None

    for _ in range(PLACEMENT_RETRIES):
        off_r = random.randint(int(row_lo), int(row_hi))
        off_c = random.randint(int(col_lo), int(col_hi))
        translated_all = {tuple(cell) for cell in _translate_cells(template["all_cells"], off_r, off_c)}
        if translated_all & blocked:
            continue
        translated = {
            "color": int(color),
            "turn": turn,
            "dir1": template["dir1"],
            "dir2": template["dir2"],
            "seg1": _translate_cells(template["seg1"], off_r, off_c),
            "seg2": _translate_cells(template["seg2"], off_r, off_c),
            "connector": _translate_cells(template["connector"], off_r, off_c),
            "sources": _translate_cells(template["sources"], off_r, off_c),
            "targets": _translate_cells(template["targets"], off_r, off_c),
            "all_cells": translated_all,
        }
        return translated
    return None


def _expected_output(grid: np.ndarray, objects: list[dict], bg: int) -> np.ndarray:
    out = np.array(grid, copy=True)
    out[out == int(MARKER_COLOR)] = int(bg)
    for obj in objects:
        for r, c in obj["connector"]:
            out[int(r), int(c)] = int(obj["color"])
        for r, c in obj["targets"]:
            out[int(r), int(c)] = int(MARKER_COLOR)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    bg_choices = [c for c in range(10) if c != int(MARKER_COLOR)]
    bg = random.choice(bg_choices)
    grid = np.full((h, w), bg, dtype=int)

    pair_count = random.randint(*PAIR_COUNT_RANGE)
    colors = [c for c in range(10) if c not in (int(bg), int(MARKER_COLOR))]
    if len(colors) < pair_count:
        return None
    path_colors = random.sample(colors, pair_count)

    if pair_count >= 2 and random.random() < 0.7:
        turns = ["left", "right"] + [random.choice(["left", "right"]) for _ in range(pair_count - 2)]
        random.shuffle(turns)
    else:
        turns = [random.choice(["left", "right"]) for _ in range(pair_count)]

    blocked: set[tuple[int, int]] = set()
    objects: list[dict] = []
    for color, turn in zip(path_colors, turns):
        placed = None
        for _ in range(PLACEMENT_RETRIES):
            candidate = _place_object(h, w, blocked, int(color), turn)
            if candidate is not None:
                placed = candidate
                break
        if placed is None:
            return None
        blocked |= _expand_cells(set(placed["all_cells"]), h, w, pad=1)
        objects.append(placed)

    for obj in objects:
        for r, c in obj["seg1"] + obj["seg2"]:
            grid[int(r), int(c)] = int(obj["color"])
        for r, c in obj["sources"]:
            grid[int(r), int(c)] = int(MARKER_COLOR)

    expected = _expected_output(grid, objects, bg)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    sig_colors = {int(c) for c in np.unique(grid)}
    if len(sig_colors) < 4:
        return None
    if int((grid == int(MARKER_COLOR)).sum()) != 2 * pair_count:
        return None
    if _bg_color(grid) != int(bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate a valid input after {GEN_RETRIES} attempts")
