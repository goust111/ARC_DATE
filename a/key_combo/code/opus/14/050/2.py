from __future__ import annotations

from itertools import combinations, permutations
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells |= {(r1, c) for c in range(c0, c1 + 1)}
    cells |= {(r, c0) for r in range(r0, r1 + 1)}
    cells |= {(r, c1) for r in range(r0, r1 + 1)}
    return cells


def _marker_specs(box: tuple[int, int, int, int]) -> dict[str, dict[str, tuple[int, int]]]:
    r0, c0, r1, c1 = box
    return {
        "tl": {"src": (r0 + 1, c0 + 1), "outer": (r0 - 1, c0 - 1), "dir": (-1, -1)},
        "tr": {"src": (r0 + 1, c1 - 1), "outer": (r0 - 1, c1 + 1), "dir": (-1, 1)},
        "bl": {"src": (r1 - 1, c0 + 1), "outer": (r1 + 1, c0 - 1), "dir": (1, -1)},
        "br": {"src": (r1 - 1, c1 - 1), "outer": (r1 + 1, c1 + 1), "dir": (1, 1)},
    }


def _destination_for_depth(frame: dict, depth: int) -> tuple[int, int]:
    spec = frame["marker_spec"]
    rr = int(spec["outer"][0] + (depth - 1) * spec["dir"][0])
    cc = int(spec["outer"][1] + (depth - 1) * spec["dir"][1])
    return rr, cc


def _infer_frame(grid: np.ndarray, color: int, bg: int) -> dict:
    coords = np.argwhere(grid == color)
    assert coords.size > 0, f"missing color {color}"
    box = _bbox_from_coords(coords)
    r0, c0, r1, c1 = box
    assert r1 - r0 + 1 >= 4 and c1 - c0 + 1 >= 4

    border = _border_cells(box)
    interior = []
    for r, c in coords:
        cell = (int(r), int(c))
        if cell not in border:
            interior.append(cell)
    assert len(interior) == 1, f"color {color} must have exactly one interior marker"

    marker = interior[0]
    specs = _marker_specs(box)
    corner = None
    for name, spec in specs.items():
        if marker == spec["src"]:
            corner = name
            marker_spec = spec
            break
    assert corner is not None, f"marker for color {color} must occupy an interior corner"

    visible_cells = {tuple(map(int, rc)) for rc in coords}
    assert visible_cells - ({marker} | border) == set()
    assert any((r0, c) in visible_cells for c in range(c0, c1 + 1))
    assert any((r1, c) in visible_cells for c in range(c0, c1 + 1))
    assert any((r, c0) in visible_cells for r in range(r0, r1 + 1))
    assert any((r, c1) in visible_cells for r in range(r0, r1 + 1))

    return {
        "color": int(color),
        "bbox": box,
        "border": border,
        "marker": marker,
        "corner": corner,
        "marker_spec": marker_spec,
    }


def _infer_frames(grid: np.ndarray) -> dict[int, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    frames = {}
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        frames[color] = _infer_frame(g, color, bg)
    assert 3 <= len(frames) <= 4
    return frames


def _overlap_constraints(grid: np.ndarray, frames: dict[int, dict]) -> set[tuple[int, int]]:
    constraints: set[tuple[int, int]] = set()
    for color_a, color_b in combinations(sorted(frames), 2):
        shared = frames[color_a]["border"] & frames[color_b]["border"]
        assert shared, f"frames {color_a} and {color_b} must overlap"
        winners = {
            int(grid[r, c])
            for r, c in shared
            if int(grid[r, c]) in {color_a, color_b}
        }
        assert winners, f"pair {color_a},{color_b} lacks a visible pairwise overlap"
        assert len(winners) == 1, f"pair {color_a},{color_b} must imply a unique order"
        winner = next(iter(winners))
        loser = color_b if winner == color_a else color_a
        constraints.add((loser, winner))
    return constraints


def _unique_order(colors: list[int], constraints: set[tuple[int, int]]) -> tuple[int, ...]:
    valid = []
    for order in permutations(colors):
        pos = {color: idx for idx, color in enumerate(order)}
        if all(pos[below] < pos[above] for below, above in constraints):
            valid.append(order)
    assert len(valid) == 1, "layer order must be uniquely determined"
    return valid[0]


def _paint_frame(canvas: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    canvas[r0 : r1 + 1, c0] = color
    canvas[r0 : r1 + 1, c1] = color
    canvas[r0, c0 : c1 + 1] = color
    canvas[r1, c0 : c1 + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the overlap depth order of several rectangular frames and move each same-color corner marker outward by its depth.

    Input structure:
    - The background is black.
    - The foreground contains 3 or 4 one-pixel-thick rectangular frames, each in a distinct non-black color.
    - Every frame has exactly one same-color marker at one interior corner cell, and frame borders may overlap so only the top color is visible at shared border cells.

    Rule:
    - Recover each frame's rectangle and its marked interior corner.
    - Use the visible colors on shared border cells to infer the unique bottom-to-top layer order.
    - Erase every interior marker, then move it to the corresponding exterior corner on the same diagonal, with an extra outward shift of depth-1 where depth 1 is the bottom frame.
    - Keep the visible frame borders unchanged.

    Key insight:
    - The overlap colors determine a global depth ranking, and that ranking is reused as the per-frame scalar that controls how far its marker moves outward.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    assert bg == 0, "expected black background"

    frames = _infer_frames(g)
    order = _unique_order(list(frames), _overlap_constraints(g, frames))

    out = g.copy()
    for frame in frames.values():
        out[frame["marker"]] = bg

    for depth, color in enumerate(order, start=1):
        frame = frames[color]
        dst = _destination_for_depth(frame, depth)
        assert 0 <= dst[0] < g.shape[0] and 0 <= dst[1] < g.shape[1]
        assert out[dst] == bg, "destination must be empty"
        out[dst] = color

    return out


_BG = 0
_COLORS = (1, 2, 3, 4)
_FRAME_COUNT_CHOICES = (3, 4)
_GRID_RANGE = (15, 20)
_FRAME_MIN_SIZE = 4
_FRAME_MAX_SIZE = 8
_MAX_DEPTH = max(_FRAME_COUNT_CHOICES)
_OUTER_MARGIN = _MAX_DEPTH
_GEN_ATTEMPTS = 600


def _assign_distinct_ends(starts: list[int], cap: int) -> list[int] | None:
    max_start = max(starts)
    candidates = []
    for start in starts:
        lo = max(start + _FRAME_MIN_SIZE - 1, max_start)
        hi = min(start + _FRAME_MAX_SIZE - 1, cap)
        vals = list(range(lo, hi + 1))
        if not vals:
            return None
        candidates.append(vals)

    order = sorted(range(len(starts)), key=lambda idx: len(candidates[idx]))
    used: set[int] = set()
    assigned = [None] * len(starts)

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        vals = candidates[idx][:]
        random.shuffle(vals)
        for val in vals:
            if val in used:
                continue
            used.add(val)
            assigned[idx] = val
            if backtrack(pos + 1):
                return True
            used.remove(val)
            assigned[idx] = None
        return False

    if not backtrack(0):
        return None
    return [int(v) for v in assigned]


def _sample_rectangles(height: int, width: int, count: int) -> dict[int, tuple[int, int, int, int]] | None:
    top_start_max = height - count - 7
    left_start_max = width - count - 7
    if top_start_max < _OUTER_MARGIN or left_start_max < _OUTER_MARGIN:
        return None

    top_start = random.randint(_OUTER_MARGIN, top_start_max)
    left_start = random.randint(_OUTER_MARGIN, left_start_max)
    tops = list(range(top_start, top_start + count))
    lefts = list(range(left_start, left_start + count))
    random.shuffle(tops)
    random.shuffle(lefts)

    bottoms = _assign_distinct_ends(tops, height - 1 - _OUTER_MARGIN)
    rights = _assign_distinct_ends(lefts, width - 1 - _OUTER_MARGIN)
    if bottoms is None or rights is None:
        return None

    return {
        idx: (int(tops[idx]), int(lefts[idx]), int(bottoms[idx]), int(rights[idx]))
        for idx in range(count)
    }


def _choose_marker_specs(
    boxes_by_color: dict[int, tuple[int, int, int, int]],
    order: tuple[int, ...],
    shape: tuple[int, int],
) -> dict[int, dict] | None:
    border_union = {
        color: _border_cells(box)
        for color, box in boxes_by_color.items()
    }
    all_borders = set().union(*border_union.values())

    colors = list(order)
    random.shuffle(colors)
    chosen: dict[int, dict] = {}
    used_src: set[tuple[int, int]] = set()
    used_dst: set[tuple[int, int]] = set()

    def backtrack(pos: int) -> bool:
        if pos == len(colors):
            return True
        color = colors[pos]
        depth = order.index(color) + 1
        box = boxes_by_color[color]
        options = list(_marker_specs(box).items())
        random.shuffle(options)
        for corner, spec in options:
            src = spec["src"]
            if src in used_src:
                continue
            if src in all_borders - border_union[color]:
                continue
            dst = (
                spec["outer"][0] + (depth - 1) * spec["dir"][0],
                spec["outer"][1] + (depth - 1) * spec["dir"][1],
            )
            if not (0 <= dst[0] < shape[0] and 0 <= dst[1] < shape[1]):
                continue
            if dst in used_dst or dst in used_src or dst in all_borders:
                continue
            chosen[color] = {"corner": corner, "marker_spec": spec, "marker": src}
            used_src.add(src)
            used_dst.add(dst)
            if backtrack(pos + 1):
                return True
            used_src.remove(src)
            used_dst.remove(dst)
            del chosen[color]
        return False

    if not backtrack(0):
        return None
    return chosen


def _render_input(
    shape: tuple[int, int],
    frames: dict[int, dict],
    order: tuple[int, ...],
) -> np.ndarray:
    grid = np.full(shape, _BG, dtype=int)
    for color in order:
        _paint_frame(grid, frames[color]["bbox"], color)
        grid[frames[color]["marker"]] = color
    return grid


def _render_output_from_truth(
    inp: np.ndarray,
    frames: dict[int, dict],
    order: tuple[int, ...],
) -> np.ndarray:
    out = inp.copy()
    for frame in frames.values():
        out[frame["marker"]] = _BG
    for depth, color in enumerate(order, start=1):
        out[_destination_for_depth(frames[color], depth)] = color
    return out


def _proxy_orders(frames: dict[int, dict]) -> list[tuple[int, ...]]:
    colors = list(frames)

    def area(color: int) -> int:
        r0, c0, r1, c1 = frames[color]["bbox"]
        return (r1 - r0 + 1) * (c1 - c0 + 1)

    def height(color: int) -> int:
        r0, _, r1, _ = frames[color]["bbox"]
        return r1 - r0 + 1

    def width(color: int) -> int:
        _, c0, _, c1 = frames[color]["bbox"]
        return c1 - c0 + 1

    proxies = [
        tuple(sorted(colors)),
        tuple(sorted(colors, reverse=True)),
        tuple(sorted(colors, key=lambda c: frames[c]["bbox"][0])),
        tuple(sorted(colors, key=lambda c: frames[c]["bbox"][1])),
        tuple(sorted(colors, key=lambda c: frames[c]["bbox"][2])),
        tuple(sorted(colors, key=lambda c: frames[c]["bbox"][3])),
        tuple(sorted(colors, key=lambda c: (frames[c]["bbox"][0], frames[c]["bbox"][1]))),
        tuple(sorted(colors, key=area)),
        tuple(sorted(colors, key=area, reverse=True)),
        tuple(sorted(colors, key=height)),
        tuple(sorted(colors, key=width)),
    ]
    return proxies


def _fixed_distance_output(inp: np.ndarray, frames: dict[int, dict]) -> np.ndarray:
    out = inp.copy()
    for frame in frames.values():
        out[frame["marker"]] = _BG
        out[frame["marker_spec"]["outer"]] = frame["color"]
    return out


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    true_frames: dict[int, dict],
    true_order: tuple[int, ...],
) -> bool:
    try:
        inferred = _infer_frames(inp)
        constraints = _overlap_constraints(inp, inferred)
        inferred_order = _unique_order(list(inferred), constraints)
        solved = main(inp)
    except Exception:
        return False

    if inferred_order != true_order:
        return False
    if not np.array_equal(solved, expected):
        return False
    if np.array_equal(_fixed_distance_output(inp, inferred), expected):
        return False
    if any(proxy == true_order for proxy in _proxy_orders(inferred)):
        return False

    for color, frame in inferred.items():
        if frame["bbox"] != true_frames[color]["bbox"]:
            return False
        if frame["marker"] != true_frames[color]["marker"]:
            return False
        coords = np.argwhere(inp == color)
        if _bbox_from_coords(coords) != true_frames[color]["bbox"]:
            return False

    seen = set()
    for depth, color in enumerate(true_order, start=1):
        dst = _destination_for_depth(true_frames[color], depth)
        if dst in seen or inp[dst] != _BG:
            return False
        seen.add(dst)
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*_GRID_RANGE)
    width = random.randint(*_GRID_RANGE)
    count = random.choice(_FRAME_COUNT_CHOICES)

    boxes = _sample_rectangles(height, width, count)
    if boxes is None:
        return None

    colors = random.sample(_COLORS, count)
    order = tuple(random.sample(colors, count))
    boxes_by_color = {color: boxes[idx] for idx, color in enumerate(colors)}
    marker_info = _choose_marker_specs(boxes_by_color, order, (height, width))
    if marker_info is None:
        return None

    frames = {}
    for color, box in boxes_by_color.items():
        frames[color] = {
            "color": color,
            "bbox": box,
            "border": _border_cells(box),
            "corner": marker_info[color]["corner"],
            "marker_spec": marker_info[color]["marker_spec"],
            "marker": marker_info[color]["marker"],
        }

    inp = _render_input((height, width), frames, order)
    expected = _render_output_from_truth(inp, frames, order)
    if not _semantic_checks(inp, expected, frames, order):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid overlapping-frame task input")
