from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8

PALETTE = (BLUE, RED, GREEN, YELLOW, PINK, ORANGE, TEAL)
MAX_ATTEMPTS = 600

Coord = Tuple[int, int]
Rect = Tuple[int, int, int, int]


def _neighbors4(r: int, c: int) -> Iterable[Coord]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> List[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: List[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: List[Coord] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> Rect:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_partitions(grid: np.ndarray) -> List[Tuple[Rect, int]]:
    g = np.asarray(grid, dtype=int)
    parts: List[Tuple[Rect, int]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BLACK):
        for coords in _components(g == color):
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            block = g[r0 : r1 + 1, c0 : c1 + 1]
            if np.any(block != color):
                raise ValueError("non-rectangular or non-monochromatic partition")
            parts.append(((r0, c0, r1, c1), color))
    if not parts:
        raise ValueError("expected at least one colored partition")
    parts.sort(key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3]))
    return parts


def _compress_partitions(parts: Sequence[Tuple[Rect, int]]) -> np.ndarray:
    rows = sorted({rect[0] for rect, _color in parts})
    cols = sorted({rect[1] for rect, _color in parts})
    row_index = {r: i for i, r in enumerate(rows)}
    col_index = {c: i for i, c in enumerate(cols)}
    out = np.zeros((len(rows), len(cols)), dtype=int)
    for (r0, c0, _r1, _c1), color in parts:
        rr = row_index[r0]
        cc = col_index[c0]
        if out[rr, cc] != BLACK:
            raise ValueError("duplicate compressed coordinate")
        out[rr, cc] = int(color)
    return out


def _route_cells(a: Coord, b: Coord) -> List[Coord]:
    (r1, c1), (r2, c2) = a, b
    cells: List[Coord] = []
    step_c = 1 if c2 > c1 else -1
    for c in range(c1 + step_c, c2 + step_c, step_c):
        cells.append((r1, c))
    step_r = 1 if r2 > r1 else -1
    for r in range(r1 + step_r, r2 + step_r, step_r):
        cells.append((r, c2))
    return cells


def _draw_routes(compressed: np.ndarray) -> np.ndarray:
    out = np.asarray(compressed, dtype=int).copy()
    counts = Counter(int(v) for v in out.flat if int(v) != BLACK)
    for color in sorted(c for c, n in counts.items() if n == 2):
        pts = [tuple(map(int, coord)) for coord in np.argwhere(out == color)]
        pts.sort()
        start, end = pts
        for r, c in _route_cells(start, end)[:-1]:
            if out[r, c] == BLACK:
                out[r, c] = GREY
    return out


def _analyze_routes(compressed: np.ndarray) -> List[dict]:
    occupied = {tuple(map(int, coord)) for coord in np.argwhere(compressed != BLACK)}
    counts = Counter(int(v) for v in compressed.flat if int(v) != BLACK)
    infos: List[dict] = []
    for color in sorted(c for c, n in counts.items() if n == 2):
        pts = [tuple(map(int, coord)) for coord in np.argwhere(compressed == color)]
        pts.sort()
        start, end = pts
        if start[0] == end[0] or start[1] == end[1]:
            raise ValueError("paired endpoints must differ in both row and column")
        intermediates = _route_cells(start, end)[:-1]
        blocked = [cell for cell in intermediates if cell in occupied]
        visible = [cell for cell in intermediates if cell not in occupied]
        infos.append(
            {
                "color": color,
                "start": start,
                "end": end,
                "intermediates": tuple(intermediates),
                "blocked": tuple(blocked),
                "visible": tuple(visible),
            }
        )
    return infos


def _physical_adjacency(rect_a: Rect, rect_b: Rect) -> bool:
    a_r0, a_c0, a_r1, a_c1 = rect_a
    b_r0, b_c0, b_r1, b_c1 = rect_b
    row_overlap = max(0, min(a_r1, b_r1) - max(a_r0, b_r0) + 1)
    col_overlap = max(0, min(a_c1, b_c1) - max(a_c0, b_c0) + 1)
    if a_r1 + 1 == b_r0 or b_r1 + 1 == a_r0:
        return col_overlap > 0
    if a_c1 + 1 == b_c0 or b_c1 + 1 == a_c0:
        return row_overlap > 0
    return False


def _band_tiling(row_sizes: Sequence[int], col_sizes: Sequence[int], rng: np.random.Generator) -> List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]]:
    rows = len(row_sizes)
    cols = len(col_sizes)
    used = np.zeros((rows, cols), dtype=bool)
    placements: List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]] = []
    row_starts = np.cumsum([0, *row_sizes[:-1]])
    col_starts = np.cumsum([0, *col_sizes[:-1]])

    def rec() -> bool:
        free = np.argwhere(~used)
        if free.size == 0:
            return True
        r, c = map(int, free[0])
        candidates: List[Tuple[int, int, int, int]] = []
        for hr in (2, 1):
            if r + hr > rows:
                continue
            height = int(sum(row_sizes[r : r + hr]))
            if not (2 <= height <= 4):
                continue
            for wc in (2, 1):
                if c + wc > cols:
                    continue
                width = int(sum(col_sizes[c : c + wc]))
                if not (2 <= width <= 4):
                    continue
                if np.any(used[r : r + hr, c : c + wc]):
                    continue
                candidates.append((r, c, hr, wc))
        if not candidates:
            return False
        rng.shuffle(candidates)
        candidates.sort(
            key=lambda item: (
                0.45 * (item[2] != item[3])
                + 0.20 * (item[2] * item[3] > 1)
                + rng.random()
            ),
            reverse=True,
        )
        for br, bc, hr, wc in candidates:
            used[br : br + hr, bc : bc + wc] = True
            pr0 = int(row_starts[br])
            pc0 = int(col_starts[bc])
            pr1 = int(row_starts[br] + sum(row_sizes[br : br + hr]) - 1)
            pc1 = int(col_starts[bc] + sum(col_sizes[bc : bc + wc]) - 1)
            placements.append(((br, bc, hr, wc), (pr0, pc0, pr1, pc1)))
            if rec():
                return True
            placements.pop()
            used[br : br + hr, bc : bc + wc] = False
        return False

    if not rec():
        raise RuntimeError("failed to tile band grid")
    return placements


def _candidate_pairs(positions: Sequence[Coord]) -> List[dict]:
    occupied = set(positions)
    out: List[dict] = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            a = positions[i]
            b = positions[j]
            if a[0] == b[0] or a[1] == b[1]:
                continue
            start, end = sorted((a, b))
            intermediates = _route_cells(start, end)[:-1]
            if not intermediates:
                continue
            visible = tuple(cell for cell in intermediates if cell not in occupied)
            blocked = tuple(cell for cell in intermediates if cell in occupied)
            if not visible:
                continue
            out.append(
                {
                    "i": i,
                    "j": j,
                    "start": start,
                    "end": end,
                    "route": tuple(intermediates),
                    "visible": visible,
                    "blocked": blocked,
                }
            )
    return out


def _select_pairs(positions: Sequence[Coord], rng: np.random.Generator) -> List[dict] | None:
    candidates = _candidate_pairs(positions)
    if not candidates:
        return None
    rng.shuffle(candidates)
    occupied_positions = set(positions)

    def compatible(candidate: dict, chosen: Sequence[dict]) -> bool:
        candidate_route = set(candidate["route"])
        candidate_endpoints = {candidate["start"], candidate["end"]}
        for other in chosen:
            if candidate["i"] in (other["i"], other["j"]) or candidate["j"] in (other["i"], other["j"]):
                return False
            other_route = set(other["route"])
            other_endpoints = {other["start"], other["end"]}
            if candidate_route & other_route:
                return False
            if candidate_route & other_endpoints:
                return False
            if other_route & candidate_endpoints:
                return False
        return True

    for target_pairs in rng.permutation([3, 2]):
        if len(positions) < 2 * target_pairs:
            continue

        def dfs(start_idx: int, chosen: List[dict]) -> List[dict] | None:
            if len(chosen) == target_pairs:
                blocked_count = sum(bool(item["blocked"]) for item in chosen)
                clear_count = sum(not item["blocked"] for item in chosen)
                if blocked_count >= 1 and clear_count >= 1:
                    return list(chosen)
                return None
            remaining_slots = target_pairs - len(chosen)
            if len(candidates) - start_idx < remaining_slots:
                return None
            for idx in range(start_idx, len(candidates)):
                cand = candidates[idx]
                if not compatible(cand, chosen):
                    continue
                chosen.append(cand)
                result = dfs(idx + 1, chosen)
                if result is not None:
                    return result
                chosen.pop()
            return None

        result = dfs(0, [])
        if result is None:
            continue
        endpoints = {pair["start"] for pair in result} | {pair["end"] for pair in result}
        for pair in result:
            blockers = set(pair["blocked"])
            if blockers - (occupied_positions - endpoints):
                break
        else:
            return result
    return None


def _remaining_count_plan(available_colors: Sequence[int], total_needed: int, rng: np.random.Generator) -> Dict[int, int] | None:
    colors = list(available_colors)
    rng.shuffle(colors)

    def rec(index: int, remaining: int, used: int) -> Dict[int, int] | None:
        if remaining == 0:
            return {} if used >= 2 else None
        if index == len(colors):
            return None
        color = colors[index]
        choices = [0, 1] + list(range(3, remaining + 1))
        rng.shuffle(choices)
        for count in choices:
            if count > remaining:
                continue
            sub = rec(index + 1, remaining - count, used + (1 if count > 0 else 0))
            if sub is not None:
                if count > 0:
                    sub = dict(sub)
                    sub[color] = count
                return sub
        return None

    return rec(0, total_needed, 0)


def _assign_colors(
    physical_rects: Sequence[Rect],
    positions: Sequence[Coord],
    pairs: Sequence[dict],
    rng: np.random.Generator,
) -> List[int] | None:
    n = len(physical_rects)
    adjacency = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _physical_adjacency(physical_rects[i], physical_rects[j]):
                adjacency[i].add(j)
                adjacency[j].add(i)
    for pair in pairs:
        if pair["j"] in adjacency[pair["i"]]:
            return None

    remaining_cells = n - 2 * len(pairs)
    for _ in range(16):
        pair_colors = list(rng.choice(PALETTE, size=len(pairs), replace=False))
        assigned = [-1] * n
        for color, pair in zip(pair_colors, pairs):
            assigned[pair["i"]] = int(color)
            assigned[pair["j"]] = int(color)

        free_colors = [c for c in PALETTE if c not in pair_colors]
        count_plan = _remaining_count_plan(free_colors, remaining_cells, rng)
        if count_plan is None:
            continue
        remaining_counts = dict(count_plan)

        unassigned = [idx for idx in range(n) if assigned[idx] == -1]
        unassigned.sort(key=lambda idx: len(adjacency[idx]), reverse=True)

        def rec(ptr: int) -> bool:
            if ptr == len(unassigned):
                return all(v == 0 for v in remaining_counts.values())
            idx = unassigned[ptr]
            neighbor_colors = {assigned[nbr] for nbr in adjacency[idx] if assigned[nbr] != -1}
            choices = [color for color, count in remaining_counts.items() if count > 0 and color not in neighbor_colors]
            rng.shuffle(choices)
            choices.sort(key=lambda color: (remaining_counts[color], rng.random()), reverse=True)
            for color in choices:
                assigned[idx] = color
                remaining_counts[color] -= 1
                if rec(ptr + 1):
                    return True
                remaining_counts[color] += 1
                assigned[idx] = -1
            return False

        if not rec(0):
            continue

        counts = Counter(assigned)
        if sorted(count for count in counts.values() if count == 2) != [2] * len(pairs):
            continue
        if sum(count == 2 for count in counts.values()) != len(pairs):
            continue
        return assigned
    return None


def _build_input_from_layout(
    rects: Sequence[Rect],
    colors: Sequence[int],
    mosaic_h: int,
    mosaic_w: int,
    rng: np.random.Generator,
) -> np.ndarray:
    pad_top = int(rng.integers(1, 4))
    pad_bottom = int(rng.integers(1, 4))
    pad_left = int(rng.integers(1, 4))
    pad_right = int(rng.integers(1, 4))
    total_h = mosaic_h + pad_top + pad_bottom
    total_w = mosaic_w + pad_left + pad_right
    if total_h > 30 or total_w > 30:
        raise ValueError("grid exceeds ARC size bound")
    grid = np.zeros((total_h, total_w), dtype=int)
    for (r0, c0, r1, c1), color in zip(rects, colors):
        grid[pad_top + r0 : pad_top + r1 + 1, pad_left + c0 : pad_left + c1 + 1] = int(color)
    return grid


def _validate_generated(grid: np.ndarray) -> None:
    parts = _extract_partitions(grid)
    for (r0, c0, r1, c1), _color in parts:
        if not (2 <= r1 - r0 + 1 <= 4):
            raise ValueError("partition height out of bounds")
        if not (2 <= c1 - c0 + 1 <= 4):
            raise ValueError("partition width out of bounds")
    compressed = _compress_partitions(parts)
    route_info = _analyze_routes(compressed)
    if not (2 <= len(route_info) <= 3):
        raise ValueError("need exactly 2-3 paired colors")
    if not any(info["visible"] for info in route_info):
        raise ValueError("at least one route must paint visible grey cells")
    if not any(info["blocked"] for info in route_info):
        raise ValueError("at least one route must hit an occupied compressed cell")
    if not any(not info["blocked"] for info in route_info):
        raise ValueError("at least one route should stay clear for branch coverage")
    used_route_cells = set()
    for info in route_info:
        route_cells = set(info["intermediates"])
        if used_route_cells & route_cells:
            raise ValueError("pair routes must not cross")
        used_route_cells |= route_cells
    counts = Counter(int(v) for v in compressed.flat if int(v) != BLACK)
    if sum(count == 2 for count in counts.values()) != len(route_info):
        raise ValueError("compressed pair counting mismatch")
    if compressed.shape[0] < 4 or compressed.shape[1] < 4 or compressed.shape[0] > 6 or compressed.shape[1] > 6:
        raise ValueError("compressed grid should stay within 4x4 to 6x6")
    out = main(grid)
    if out.shape != compressed.shape:
        raise ValueError("output shape mismatch")
    if not np.any(out == GREY):
        raise ValueError("output should contain grey path pixels")


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    row_count = int(rng.integers(5, 7))
    col_count = int(rng.integers(5, 7))
    row_sizes = [2] * row_count
    col_sizes = [2] * col_count
    mosaic_h = int(sum(row_sizes))
    mosaic_w = int(sum(col_sizes))
    if mosaic_h > 24 or mosaic_w > 24:
        return None

    try:
        tiled = _band_tiling(row_sizes, col_sizes, rng)
    except RuntimeError:
        return None

    has_vertical_merge = any(band_rect[2] > 1 for band_rect, _phys in tiled)
    has_horizontal_merge = any(band_rect[3] > 1 for band_rect, _phys in tiled)
    if not (has_vertical_merge and has_horizontal_merge):
        return None

    physical_rects = [phys for _band, phys in tiled]
    parts = [((r0, c0, r1, c1), idx) for idx, (r0, c0, r1, c1) in enumerate(physical_rects, start=1)]
    compressed = _compress_partitions(parts)
    if compressed.shape[0] < 4 or compressed.shape[1] < 4 or compressed.shape[0] > 6 or compressed.shape[1] > 6:
        return None
    if not np.any(compressed == BLACK):
        return None

    positions = [tuple(map(int, pos)) for pos in np.argwhere(compressed != BLACK)]
    pair_selection = _select_pairs(positions, rng)
    if pair_selection is None:
        return None

    colors = _assign_colors(physical_rects, positions, pair_selection, rng)
    if colors is None:
        return None

    grid = _build_input_from_layout(physical_rects, colors, mosaic_h, mosaic_w, rng)
    try:
        _validate_generated(grid)
    except Exception:
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress a rectangular partition mosaic to sparse top-left pixels and route grey horizontal-then-vertical links between colors that appear exactly twice.

    Input structure:
    - A black background surrounds one compact mosaic made of edge-touching monochromatic rectangles.
    - Every partition is a solid rectangle, and orthogonally adjacent partitions use different colors.
    - The relevant compressed coordinates are the sorted unique top-left rows and columns of those rectangles.

    Rule:
    - Detect every non-black rectangular partition and record its top-left coordinate.
    - Build a compressed sparse grid whose rows and columns are the unique top-left row and column sets; place each partition color at its own top-left slot and leave all unused slots black.
    - Find the colors that occur exactly twice in that compressed grid.
    - For each such color, sort its two endpoints by row then column, move first horizontally to the target column and then vertically to the target row, and paint only black intermediate cells grey.

    Key insight:
    - The endpoints to connect do not exist as special markers in the original image; they emerge only after converting the partition layout into its compressed top-left-coordinate representation.
    """
    parts = _extract_partitions(np.asarray(grid, dtype=int))
    compressed = _compress_partitions(parts)
    return _draw_routes(compressed)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        try:
            _validate_generated(grid)
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
