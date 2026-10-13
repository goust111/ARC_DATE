from __future__ import annotations

from importlib.machinery import SourceFileLoader
import random
from typing import Iterable

import numpy as np


_ARC_UTILS_PATH = "/home/lixinyu/snap/new/solvers/arc_utils.py"

try:
    _arc_utils = SourceFileLoader("_arc_utils_task_7", _ARC_UTILS_PATH).load_module()
    get_bg = _arc_utils.get_bg
    in_bounds = _arc_utils.in_bounds
    neighbors4 = _arc_utils.neighbors4
    components_mask = _arc_utils.components_mask
except Exception:
    def get_bg(grid: np.ndarray) -> int:
        values, counts = np.unique(grid, return_counts=True)
        return int(values[np.argmax(counts)])

    def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
        h, w = shape
        return 0 <= r < h and 0 <= c < w

    def neighbors4(r: int, c: int):
        yield r - 1, c
        yield r + 1, c
        yield r, c - 1
        yield r, c + 1

    def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
        assert connectivity == 4
        h, w = mask.shape
        seen = np.zeros((h, w), dtype=bool)
        out: list[np.ndarray] = []
        for r in range(h):
            for c in range(w):
                if seen[r, c] or not mask[r, c]:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                coords: list[tuple[int, int]] = []
                while stack:
                    rr, cc = stack.pop()
                    coords.append((rr, cc))
                    for nr, nc in neighbors4(rr, cc):
                        if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                out.append(np.array(coords, dtype=int))
        return out


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
MARKER_COLORS = (RED, GREEN, BLUE, ORANGE)

COLOR_TO_PATTERN: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {
    RED: ((0, YELLOW), (YELLOW, YELLOW)),
    GREEN: ((YELLOW, 0), (YELLOW, YELLOW)),
    BLUE: ((YELLOW, YELLOW), (0, YELLOW)),
    ORANGE: ((YELLOW, YELLOW), (YELLOW, 0)),
}
PATTERN_TO_COLOR = {pattern: color for color, pattern in COLOR_TO_PATTERN.items()}

_SIDE_ORDER = ("west", "north", "east", "south")
_GEN_COUNTER = 0


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _turn_left(direction: Coord) -> Coord:
    dr, dc = direction
    return -dc, dr


def _turn_right(direction: Coord) -> Coord:
    dr, dc = direction
    return dc, -dr


def _turn_back(direction: Coord) -> Coord:
    dr, dc = direction
    return -dr, -dc


def _has_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    stack: list[Coord] = []
    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors4(r, c):
            if in_bounds(mask.shape, nr, nc) and not mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return bool(np.any((~mask) & (~seen)))


def _extract_symbols(grid: np.ndarray) -> list[dict]:
    yellow_components = components_mask(grid == YELLOW, connectivity=4)
    symbols: list[dict] = []
    for comp in yellow_components:
        if comp.shape[0] != 3:
            continue
        bbox = _bbox_from_coords(comp)
        r0, c0, r1, c1 = bbox
        if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(np.isin(patch, [BG, YELLOW])):
            continue
        if int(np.sum(patch == YELLOW)) != 3 or int(np.sum(patch == BG)) != 1:
            continue
        pattern = tuple(tuple(int(v) for v in row) for row in patch.tolist())
        color = PATTERN_TO_COLOR.get(pattern)
        if color is None:
            continue
        symbols.append({"bbox": bbox, "pattern": pattern, "color": int(color)})

    symbols.sort(key=lambda sym: sym["bbox"])
    assert symbols, "expected at least one 2x2 contour symbol"

    for i, first in enumerate(symbols):
        fr0, fc0, fr1, fc1 = first["bbox"]
        for second in symbols[i + 1 :]:
            sr0, sc0, sr1, sc1 = second["bbox"]
            disjoint = fr1 < sr0 or sr1 < fr0 or fc1 < sc0 or sc1 < fc0
            assert disjoint, "symbols should not overlap"
    return symbols


def _restored_obstacle_mask(grid: np.ndarray, symbols: Iterable[dict]) -> np.ndarray:
    mask = grid == GREY
    for symbol in symbols:
        r0, c0, r1, c1 = symbol["bbox"]
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    assert len(components_mask(mask, connectivity=4)) == 1, "expected one restored obstacle region"
    assert not _has_hole(mask), "restored obstacle should be hole-free"
    return mask


def _find_marker(grid: np.ndarray) -> tuple[Coord, int]:
    marker_coords = np.argwhere(np.isin(grid, MARKER_COLORS))
    assert marker_coords.shape == (1, 2), "expected one marker pixel"
    marker_pos = tuple(map(int, marker_coords[0]))
    return marker_pos, int(grid[marker_pos])


def _sightline_to_obstacle(mask: np.ndarray, start: Coord) -> tuple[Coord, Coord]:
    hits: list[tuple[Coord, Coord]] = []
    sr, sc = start
    for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        dr, dc = direction
        r, c = sr + dr, sc + dc
        previous_bg: Coord | None = None
        while in_bounds(mask.shape, r, c):
            if mask[r, c]:
                assert previous_bg is not None, "marker must not touch obstacle directly"
                hits.append((direction, previous_bg))
                break
            previous_bg = (r, c)
            r += dr
            c += dc
    assert len(hits) == 1, "expected one clear cardinal sightline to the obstacle"
    return hits[0]


def _boundary_nodes(mask: np.ndarray) -> set[Coord]:
    h, w = mask.shape
    nodes: set[Coord] = set()
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            if any(in_bounds(mask.shape, nr, nc) and mask[nr, nc] for nr, nc in neighbors4(r, c)):
                nodes.add((r, c))
    return nodes


def _boundary_neighbors(mask: np.ndarray, nodes: set[Coord], pos: Coord) -> list[Coord]:
    r, c = pos
    neighbors: list[Coord] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if nxt not in nodes:
                continue
            if abs(dr) + abs(dc) == 1:
                neighbors.append(nxt)
                continue
            bridge_a = (r + dr, c)
            bridge_b = (r, c + dc)
            obstacle_count = int(in_bounds(mask.shape, *bridge_a) and mask[bridge_a]) + int(
                in_bounds(mask.shape, *bridge_b) and mask[bridge_b]
            )
            if obstacle_count == 1:
                neighbors.append(nxt)
    return neighbors


def _boundary_walk(mask: np.ndarray, start: Coord, sight_direction: Coord) -> list[Coord]:
    nodes = _boundary_nodes(mask)
    assert start in nodes, "contour start must lie on the outer boundary ring"

    graph = {node: _boundary_neighbors(mask, nodes, node) for node in nodes}
    assert all(len(nbrs) == 2 for nbrs in graph.values()), "boundary ring should be a simple 8-neighbor cycle"

    first_step = (start[0] + _turn_left(sight_direction)[0], start[1] + _turn_left(sight_direction)[1])
    assert first_step in graph[start], "clockwise walk should leave the sightline by a left turn"

    path = [start]
    prev = start
    curr = first_step
    limit = len(nodes) + 1
    for _ in range(limit):
        if curr == start:
            return path
        path.append(curr)
        a, b = graph[curr]
        nxt = a if a != prev else b
        prev, curr = curr, nxt
    raise AssertionError("boundary walk did not close")


def _has_simple_boundary_cycle(mask: np.ndarray) -> bool:
    work = np.pad(mask.astype(bool), 1, constant_values=False)
    nodes = _boundary_nodes(work)
    if not nodes:
        return False
    graph = {node: _boundary_neighbors(work, nodes, node) for node in nodes}
    if any(len(nbrs) != 2 for nbrs in graph.values()):
        return False

    start = next(iter(nodes))
    seen = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for nxt in graph[node]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(nodes)


def _adjacent_to_bbox(pos: Coord, bbox: BBox) -> bool:
    r, c = pos
    r0, c0, r1, c1 = bbox
    return (
        (r0 <= r <= r1 and (c == c0 - 1 or c == c1 + 1))
        or (c0 <= c <= c1 and (r == r0 - 1 or r == r1 + 1))
    )


def _analyze_case(grid: np.ndarray) -> dict:
    symbols = _extract_symbols(grid)
    obstacle_mask = _restored_obstacle_mask(grid, symbols)
    marker_pos, marker_color = _find_marker(grid)
    sight_direction, contour_start = _sightline_to_obstacle(obstacle_mask, marker_pos)
    path = _boundary_walk(obstacle_mask, contour_start, sight_direction)

    first_hits: dict[int, int] = {}
    for step, pos in enumerate(path):
        for idx, symbol in enumerate(symbols):
            if idx not in first_hits and _adjacent_to_bbox(pos, symbol["bbox"]):
                first_hits[idx] = step

    assert len(first_hits) == len(symbols), "every symbol should be encountered along the contour"
    encounter_order = sorted(first_hits, key=first_hits.get)
    matching = [idx for idx in encounter_order if symbols[idx]["color"] == marker_color]
    assert matching, "marker color should match at least one decoded symbol"
    match_idx = matching[0]
    stop_step = first_hits[match_idx]

    return {
        "symbols": symbols,
        "obstacle_mask": obstacle_mask,
        "marker_pos": marker_pos,
        "marker_color": marker_color,
        "sight_direction": sight_direction,
        "contour_start": contour_start,
        "path": path,
        "first_hits": first_hits,
        "encounter_order": encounter_order,
        "matching_symbol": match_idx,
        "stop": path[stop_step],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the marker clockwise around a grey obstacle until the first decoded contour symbol matches its color.

    Input structure:
    - The background is black, with one connected grey obstacle in the middle.
    - Several 2x2 yellow/black symbol masks are embedded on the obstacle surface, and one single marker pixel
      outside the obstacle is colored red, green, blue, or orange.
    - The marker has exactly one unobstructed horizontal or vertical sightline to the obstacle.

    Rule:
    - Move the marker along its unique sightline to the last background cell next to the obstacle.
    - From there, walk clockwise while staying 4-adjacent to the restored obstacle contour.
    - Whenever the walk reaches a cell adjacent to a 2x2 symbol mask, decode that mask by its yellow/black pattern;
      stop at the first encountered symbol whose decoded color equals the marker color.
    - Erase the original marker position and restore every symbol mask back to solid grey obstacle cells.

    Key insight:
    - The geometry only determines the traversal order; the stopping point is selected by reading the embedded 2x2
      symbols and matching their hidden color code to the marker color.
    """
    g = np.asarray(grid, dtype=int)
    analysis = _analyze_case(g)

    out = g.copy()
    bg = get_bg(g)
    out[analysis["marker_pos"]] = bg
    for symbol in analysis["symbols"]:
        r0, c0, r1, c1 = symbol["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = GREY
    out[analysis["stop"]] = analysis["marker_color"]
    return out


def _side_boxes(local_h: int, local_w: int) -> dict[str, BBox]:
    west_r = random.randint(2, local_h - 4)
    east_r = random.randint(2, local_h - 4)
    top_c = random.randint(2, local_w - 4)
    bottom_c = random.randint(2, local_w - 4)
    return {
        "west": (west_r, 0, west_r + 1, 1),
        "north": (0, top_c, 1, top_c + 1),
        "east": (east_r, local_w - 2, east_r + 1, local_w - 1),
        "south": (local_h - 2, bottom_c, local_h - 1, bottom_c + 1),
    }


def _bbox_mask(shape: tuple[int, int], bbox: BBox) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = True
    return out


def _carve_notch(mask: np.ndarray, reserved: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    side = random.choice(_SIDE_ORDER)
    trial = mask.copy()

    if side in {"west", "east"}:
        depth = random.randint(1, min(2, max(1, w // 3)))
        length = random.randint(2, h - 3)
        r0 = random.randint(1, h - length - 1)
        c0 = 0 if side == "west" else w - depth
        candidate = np.zeros_like(mask, dtype=bool)
        candidate[r0 : r0 + length, c0 : c0 + depth] = True
    else:
        depth = random.randint(1, min(2, max(1, h // 3)))
        length = random.randint(2, w - 3)
        c0 = random.randint(1, w - length - 1)
        r0 = 0 if side == "north" else h - depth
        candidate = np.zeros_like(mask, dtype=bool)
        candidate[r0 : r0 + depth, c0 : c0 + length] = True

    if np.any(candidate & reserved):
        return mask

    trial[candidate] = False
    if np.sum(trial) < 30:
        return mask
    if len(components_mask(trial, connectivity=4)) != 1 or _has_hole(trial):
        return mask
    return trial


def _generate_obstacle_mask() -> tuple[np.ndarray, dict[str, BBox]]:
    for _ in range(200):
        local_h = random.randint(7, 9)
        local_w = random.randint(7, 9)
        mask = np.ones((local_h, local_w), dtype=bool)
        side_boxes = _side_boxes(local_h, local_w)
        reserved = np.zeros_like(mask, dtype=bool)
        for bbox in side_boxes.values():
            reserved |= _bbox_mask(mask.shape, bbox)

        notch_count = random.randint(1, 4)
        for _ in range(notch_count):
            mask = _carve_notch(mask, reserved)

        area = int(np.sum(mask))
        if not (30 <= area <= 50):
            continue
        if not _has_simple_boundary_cycle(mask):
            continue

        start_candidates = _start_side_candidates(mask, side_boxes)
        if all(start_candidates[side] for side in _SIDE_ORDER):
            return mask, side_boxes
    raise RuntimeError("failed to generate obstacle mask")


def _start_side_candidates(mask: np.ndarray, side_boxes: dict[str, BBox]) -> dict[str, list[int]]:
    h, w = mask.shape
    west_rows = [
        r
        for r in range(1, h - 1)
        if mask[r, 0] and not (side_boxes["west"][0] <= r <= side_boxes["west"][2])
    ]
    east_rows = [
        r
        for r in range(1, h - 1)
        if mask[r, w - 1] and not (side_boxes["east"][0] <= r <= side_boxes["east"][2])
    ]
    north_cols = [
        c
        for c in range(1, w - 1)
        if mask[0, c] and not (side_boxes["north"][1] <= c <= side_boxes["north"][3])
    ]
    south_cols = [
        c
        for c in range(1, w - 1)
        if mask[h - 1, c] and not (side_boxes["south"][1] <= c <= side_boxes["south"][3])
    ]
    return {"west": west_rows, "east": east_rows, "north": north_cols, "south": south_cols}


def _place_obstacle(mask: np.ndarray, margin_top: int, margin_left: int, shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    h, w = mask.shape
    grid[margin_top : margin_top + h, margin_left : margin_left + w][mask] = GREY
    return grid


def _global_side_boxes(side_boxes: dict[str, BBox], top: int, left: int) -> dict[str, BBox]:
    out: dict[str, BBox] = {}
    for side, bbox in side_boxes.items():
        r0, c0, r1, c1 = bbox
        out[side] = (r0 + top, c0 + left, r1 + top, c1 + left)
    return out


def _build_marker_position(side: str, index: int, top: int, left: int, local_h: int, local_w: int, gap: int) -> Coord:
    if side == "west":
        return top + index, left - 1 - gap
    if side == "east":
        return top + index, left + local_w + gap
    if side == "north":
        return top - 1 - gap, left + index
    if side == "south":
        return top + local_h + gap, left + index
    raise AssertionError(f"unknown side: {side}")


def _paint_symbols(grid: np.ndarray, side_to_color: dict[str, int], global_boxes: dict[str, BBox]) -> None:
    for side, color in side_to_color.items():
        pattern = np.array(COLOR_TO_PATTERN[color], dtype=int)
        r0, c0, r1, c1 = global_boxes[side]
        grid[r0 : r1 + 1, c0 : c1 + 1] = pattern


def _encounter_sides(path: list[Coord], global_boxes: dict[str, BBox]) -> list[str]:
    hits: list[tuple[int, str]] = []
    for side, bbox in global_boxes.items():
        for step, pos in enumerate(path):
            if _adjacent_to_bbox(pos, bbox):
                hits.append((step, side))
                break
    assert len(hits) == 4, "all four side symbols should be encountered"
    hits.sort()
    return [side for _step, side in hits]


def _validate_generated_case(grid: np.ndarray, target_color: int, target_index: int) -> bool:
    analysis = _analyze_case(grid)
    symbols = analysis["symbols"]
    if len(symbols) != 4:
        return False
    if sorted(symbol["color"] for symbol in symbols) != sorted(MARKER_COLORS):
        return False
    encounter_colors = [symbols[idx]["color"] for idx in analysis["encounter_order"]]
    if encounter_colors[target_index] != target_color:
        return False
    if analysis["encounter_order"].index(analysis["matching_symbol"]) != target_index:
        return False
    if analysis["stop"] == analysis["marker_pos"]:
        return False

    out = main(grid)
    if int(np.sum(np.isin(out, MARKER_COLORS))) != 1:
        return False
    if out[analysis["marker_pos"]] != BG:
        return False
    if out[analysis["stop"]] != target_color:
        return False
    for symbol in symbols:
        r0, c0, r1, c1 = symbol["bbox"]
        if not np.all(out[r0 : r1 + 1, c0 : c1 + 1] == GREY):
            return False
    return True


def _gen_attempt(target_color: int, target_index: int) -> np.ndarray | None:
    try:
        mask, side_boxes = _generate_obstacle_mask()
        local_h, local_w = mask.shape
        start_candidates = _start_side_candidates(mask, side_boxes)
        start_side = random.choice(_SIDE_ORDER)
        start_index = random.choice(start_candidates[start_side])

        gap = random.randint(2, 4)
        margin_top = random.randint(3, 5)
        margin_bottom = random.randint(3, 5)
        margin_left = random.randint(3, 5)
        margin_right = random.randint(3, 5)
        if start_side == "west":
            margin_left = max(margin_left, gap + 2)
        elif start_side == "east":
            margin_right = max(margin_right, gap + 2)
        elif start_side == "north":
            margin_top = max(margin_top, gap + 2)
        else:
            margin_bottom = max(margin_bottom, gap + 2)

        grid_h = local_h + margin_top + margin_bottom
        grid_w = local_w + margin_left + margin_right
        if max(grid_h, grid_w) > 30:
            return None

        grid = _place_obstacle(mask, margin_top, margin_left, (grid_h, grid_w))
        global_boxes = _global_side_boxes(side_boxes, margin_top, margin_left)
        marker_pos = _build_marker_position(start_side, start_index, margin_top, margin_left, local_h, local_w, gap)
        grid[marker_pos] = target_color

        obstacle_mask = grid == GREY
        sight_direction, contour_start = _sightline_to_obstacle(obstacle_mask, marker_pos)
        preview_path = _boundary_walk(obstacle_mask, contour_start, sight_direction)
        encounter_sides = _encounter_sides(preview_path, global_boxes)

        remaining_colors = [color for color in MARKER_COLORS if color != target_color]
        remaining_sides = [side for side in encounter_sides if side != encounter_sides[target_index]]
        random.shuffle(remaining_colors)
        side_to_color = {encounter_sides[target_index]: target_color}
        for side, color in zip(remaining_sides, remaining_colors):
            side_to_color[side] = color

        _paint_symbols(grid, side_to_color, global_boxes)
        if not _validate_generated_case(grid, target_color, target_index):
            return None
        return grid
    except AssertionError:
        return None


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    target_color = MARKER_COLORS[_GEN_COUNTER % len(MARKER_COLORS)]
    target_index = 1 + ((_GEN_COUNTER // len(MARKER_COLORS)) % 3)
    _GEN_COUNTER += 1

    for _ in range(400):
        grid = _gen_attempt(target_color, target_index)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 7")
