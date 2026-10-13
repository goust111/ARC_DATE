from __future__ import annotations

from collections import deque
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np


Coord = Tuple[int, int]

BG = 0
WALL = 5
SEGMENT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 500


def _in_bounds(shape: Tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors8(r: int, c: int) -> Sequence[Coord]:
    return (
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    )


def _find_components(mask: np.ndarray) -> List[List[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: List[List[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: List[Coord] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _segment_cells(start: Coord, direction: Coord, length: int) -> List[Coord]:
    sr, sc = start
    dr, dc = direction
    return [(sr + i * dr, sc + i * dc) for i in range(length)]


def _paint(grid: np.ndarray, cells: Sequence[Coord], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _parse_segment(color: int, comp: Sequence[Coord]) -> Dict[str, object]:
    coords = sorted((int(r), int(c)) for r, c in comp)
    if len(coords) < 2:
        raise ValueError("segment too short")

    if len({r - c for r, c in coords}) == 1:
        ordered = sorted(coords, key=lambda p: (p[0], p[1]))
        direction = (1, 1)
    elif len({r + c for r, c in coords}) == 1:
        ordered = sorted(coords, key=lambda p: (p[0], -p[1]))
        direction = (1, -1)
    else:
        raise ValueError("component is not diagonal")

    for a, b in zip(ordered, ordered[1:]):
        if (b[0] - a[0], b[1] - a[1]) != direction:
            raise ValueError("segment has gaps")

    return {
        "color": int(color),
        "coords": tuple(ordered),
        "top": ordered[0],
        "tail": ordered[-1],
        "direction": direction,
        "length": len(ordered),
    }


def _extract_segments(grid: np.ndarray) -> List[Dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    segments: List[Dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(g) if v not in (BG, WALL)):
        comps = _find_components(g == color)
        for comp in comps:
            segments.append(_parse_segment(color, comp))
    segments.sort(key=lambda seg: (int(seg["length"]), seg["top"][0], seg["top"][1]))
    for idx, seg in enumerate(segments):
        seg["index"] = idx
    return segments


def _simulate(grid: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, object]], List[Dict[str, object]]]:
    original = np.asarray(grid, dtype=int)
    out = original.copy()
    segments = _extract_segments(original)

    coord_to_segment: Dict[Coord, int] = {}
    for seg in segments:
        for cell in seg["coords"]:
            coord_to_segment[cell] = int(seg["index"])

    added_by: Dict[Coord, int] = {}
    traces: List[Dict[str, object]] = []

    for seg in segments:
        tail_r, tail_c = seg["tail"]
        dr, dc = seg["direction"]
        added: List[Coord] = []
        blocker: Dict[str, object] | None = None

        for step in range(1, int(seg["length"]) + 2):
            rr = tail_r + dr * step
            cc = tail_c + dc * step
            if not _in_bounds(out.shape, rr, cc):
                blocker = {"type": "boundary", "coord": (rr, cc), "step": step}
                break
            value = int(out[rr, cc])
            if value == BG:
                out[rr, cc] = int(seg["color"])
                added.append((rr, cc))
                added_by[(rr, cc)] = int(seg["index"])
                continue

            if value == WALL:
                blocker = {"type": "wall", "coord": (rr, cc), "step": step}
            elif (rr, cc) in added_by:
                blocker = {
                    "type": "ray",
                    "coord": (rr, cc),
                    "step": step,
                    "source": int(added_by[(rr, cc)]),
                }
            elif (rr, cc) in coord_to_segment:
                blocker = {
                    "type": "segment",
                    "coord": (rr, cc),
                    "step": step,
                    "source": int(coord_to_segment[(rr, cc)]),
                }
            else:
                blocker = {"type": "pixel", "coord": (rr, cc), "step": step}
            break

        erased = False
        if not added:
            for rr, cc in seg["coords"]:
                out[rr, cc] = BG
            erased = True

        traces.append(
            {
                "segment": seg,
                "added": tuple(added),
                "erased": erased,
                "blocker": blocker,
            }
        )

    return out, traces, segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend diagonal segments by one-more-than-their-length while walls and earlier segments can stop them.

    Input structure:
    - The grid is mostly black background, with grey wall cells and 2-4 colored diagonal segments.
    - Each colored object is a monochromatic maximal straight segment on one diagonal family.
    - Segment colors are distinct from the grey wall color and from the black background.

    Rule:
    - Identify every diagonal segment, measure its length L, and orient it from its top endpoint toward its lower endpoint on that diagonal.
    - Process segments in ascending length order, breaking ties by the row and then column of the top endpoint.
    - For the current segment, try to paint L+1 new cells forward from its tail; stop before any out-of-bounds cell, grey wall, surviving original segment, or previously painted extension.
    - If the very first attempted extension cell is blocked, erase that segment's original cells; otherwise keep the original segment and any newly painted cells.

    Key insight:
    - The obstacle map is dynamic: erased short segments can free longer ones, while earlier extensions can create fresh blockers for later segments.
    """
    out, _, _ = _simulate(np.asarray(grid, dtype=int))
    return out


def _build_chain_group(shape: Tuple[int, int], len_a: int, len_b: int) -> Dict[str, object] | None:
    h, w = shape
    total_span = len_a + len_b
    if h < total_span + 1 or w < total_span + 1:
        return None

    row_lo = 0
    row_hi = h - total_span - 1
    col_lo = 0
    col_hi = max(0, w // 2 - total_span - 1)
    if row_lo > row_hi or col_lo > col_hi:
        return None

    b_start = (random.randint(row_lo, row_hi), random.randint(col_lo, col_hi))
    direction = (1, 1)
    b_cells = _segment_cells(b_start, direction, len_b)
    a_start = (b_start[0] + len_b, b_start[1] + len_b)
    a_cells = _segment_cells(a_start, direction, len_a)
    wall = (a_start[0] + len_a, a_start[1] + len_a)
    return {"a": a_cells, "b": b_cells, "wall": wall}


def _build_cross_group(
    shape: Tuple[int, int],
    len_c: int,
    len_d: int,
    forbidden: Sequence[Coord],
) -> Dict[str, object] | None:
    h, w = shape
    occupied = set(forbidden)

    row_lo = max(len_d + 1, len_c + 1)
    row_hi = h - len_c
    col_lo = len_c + 1
    col_hi = w - len_d - 2
    if row_lo > row_hi or col_lo > col_hi:
        return None

    for _ in range(100):
        x = (random.randint(row_lo, row_hi), random.randint(col_lo, col_hi))
        dir_c = (1, 1)
        dir_d = (1, -1)
        c_start = (x[0] - (len_c + 1), x[1] - (len_c + 1))
        d_start = (x[0] - (len_d + 1), x[1] + (len_d + 1))
        c_cells = _segment_cells(c_start, dir_c, len_c)
        d_cells = _segment_cells(d_start, dir_d, len_d)
        c_future = [
            (c_cells[-1][0] + dir_c[0] * step, c_cells[-1][1] + dir_c[1] * step)
            for step in range(1, len_c + 2)
        ]
        d_future = [
            (d_cells[-1][0] + dir_d[0] * step, d_cells[-1][1] + dir_d[1] * step)
            for step in range(1, len_d + 2)
        ]
        all_cells = c_cells + d_cells + c_future + d_future
        if len(set(all_cells)) != len(all_cells) - 1:
            # The only intended duplicate is the crossing cell.
            counts: Dict[Coord, int] = {}
            for cell in all_cells:
                counts[cell] = counts.get(cell, 0) + 1
            if sorted(counts.values(), reverse=True) != [2] + [1] * (len(counts) - 1):
                continue
            if counts.get(x, 0) != 2:
                continue
        if any(not _in_bounds(shape, r, c) for r, c in all_cells):
            continue
        if occupied.intersection(c_cells) or occupied.intersection(d_cells):
            continue
        if occupied.intersection(c_future) or occupied.intersection(d_future):
            continue
        return {"c": c_cells, "d": d_cells, "cross": x}
    return None


def _wall_cluster(blocker: Coord, shape: Tuple[int, int], reserved: Sequence[Coord]) -> List[Coord]:
    h, w = shape
    reserved_set = set(reserved)
    cells = [blocker]
    candidates = [
        (blocker[0], blocker[1] + 1),
        (blocker[0] + 1, blocker[1]),
        (blocker[0] - 1, blocker[1]),
        (blocker[0], blocker[1] - 1),
    ]
    random.shuffle(candidates)
    for cell in candidates:
        if _in_bounds((h, w), cell[0], cell[1]) and cell not in reserved_set:
            cells.append(cell)
            break
    return cells


def _extra_wall_clusters(shape: Tuple[int, int], reserved: Sequence[Coord]) -> List[List[Coord]]:
    h, w = shape
    reserved_set = set(reserved)
    clusters: List[List[Coord]] = []
    cluster_shapes = [
        [(0, 0)],
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (1, 0), (1, 1)],
    ]
    cluster_count = random.randint(1, 3)
    for _ in range(cluster_count):
        for _ in range(80):
            shape_cells = random.choice(cluster_shapes)
            r0 = random.randint(0, h - 1)
            c0 = random.randint(0, w - 1)
            cluster = [(r0 + dr, c0 + dc) for dr, dc in shape_cells]
            if any(not _in_bounds((h, w), r, c) for r, c in cluster):
                continue
            if reserved_set.intersection(cluster):
                continue
            if any(any(abs(r - rr) <= 1 and abs(c - cc) <= 1 for rr, cc in reserved_set) for r, c in cluster):
                continue
            reserved_set.update(cluster)
            clusters.append(cluster)
            break
    return clusters


def _make_case() -> np.ndarray | None:
    h = random.randint(12, 18)
    w = random.randint(15, 21)
    shape = (h, w)

    len_a = 2
    len_b = random.choice((3, 4))
    len_c = random.choice((2, 3))
    len_d = 4

    chain = _build_chain_group(shape, len_a, len_b)
    if chain is None:
        return None

    reserved = list(chain["a"]) + list(chain["b"]) + [chain["wall"]]
    cross = _build_cross_group(shape, len_c, len_d, reserved)
    if cross is None:
        return None

    colors = random.sample(list(SEGMENT_COLORS), 4)
    grid = np.zeros(shape, dtype=int)
    segments = {
        "a": {"cells": chain["a"], "color": colors[0]},
        "b": {"cells": chain["b"], "color": colors[1]},
        "c": {"cells": cross["c"], "color": colors[2]},
        "d": {"cells": cross["d"], "color": colors[3]},
    }

    important = []
    for name, info in segments.items():
        important.extend(info["cells"])
        direction = (1, 1) if name in ("a", "b", "c") else (1, -1)
        important.extend(
            (info["cells"][-1][0] + direction[0] * step, info["cells"][-1][1] + direction[1] * step)
            for step in range(1, len(info["cells"]) + 2)
            if _in_bounds(shape, info["cells"][-1][0] + direction[0] * step, info["cells"][-1][1] + direction[1] * step)
        )

    wall_cells = _wall_cluster(chain["wall"], shape, important)
    important.extend(wall_cells)
    extra_walls = _extra_wall_clusters(shape, important)

    for info in segments.values():
        _paint(grid, info["cells"], int(info["color"]))
    _paint(grid, wall_cells, WALL)
    for cluster in extra_walls:
        _paint(grid, cluster, WALL)

    if random.random() < 0.5:
        grid = np.fliplr(grid)

    return grid


def _validate_case(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False

    segments = _extract_segments(g)
    if not (2 <= len(segments) <= 4):
        return False
    lengths = [int(seg["length"]) for seg in segments]
    if len(set(lengths)) < 2:
        return False
    if len({int(seg["color"]) for seg in segments}) != len(segments):
        return False

    out, traces, _ = _simulate(g)
    if out.shape != g.shape:
        return False

    erased_by_wall = False
    any_free = False
    unblock_found = False
    dynamic_ray_block = False

    erased_segments = {int(t["segment"]["index"]) for t in traces if t["erased"]}
    coord_sets = {int(seg["index"]): set(seg["coords"]) for seg in segments}

    for trace in traces:
        blocker = trace["blocker"]
        if trace["erased"] and blocker and blocker["type"] == "wall" and blocker["step"] == 1:
            erased_by_wall = True
        if len(trace["added"]) > 0:
            any_free = True
        if blocker and blocker["type"] == "ray":
            dynamic_ray_block = True
        if len(trace["added"]) > 0:
            for source in erased_segments:
                if any(cell in coord_sets[source] for cell in trace["added"]):
                    unblock_found = True
                    break

    return erased_by_wall and any_free and unblock_found and dynamic_ray_block


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _make_case()
        if grid is not None and _validate_case(grid):
            return grid
    raise RuntimeError("failed to generate a valid obstacle-aware diagonal extension task")
