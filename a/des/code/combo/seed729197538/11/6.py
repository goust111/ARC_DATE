from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
MAROON = 9

CARD4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

# E, NE, N, NW, W, SW, S, SE in counter-clockwise order.
DIRS8 = [
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
]
DIR_TO_INDEX = {d: i for i, d in enumerate(DIRS8)}
TURN_PATTERNS = [("L", "L"), ("L", "R"), ("R", "L"), ("R", "R")]
NON_SPECIAL_COLORS = [1, 2, 3, 4, 6, 7, 8]


def _components(mask: np.ndarray, connectivity: int = 8) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    steps = CARD4 if connectivity == 4 else NEIGHBORS8
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _build_adj8(cells: list[tuple[int, int]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    cell_set = set(cells)
    adj = {cell: [] for cell in cell_set}
    for r, c in cell_set:
        for dr, dc in NEIGHBORS8:
            nxt = (r + dr, c + dc)
            if nxt in cell_set:
                adj[(r, c)].append(nxt)
    return adj


def _path_endpoints(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    adj = _build_adj8(cells)
    endpoints = [cell for cell, nxts in adj.items() if len(nxts) == 1]
    return sorted(endpoints)


def _order_path(cells: list[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    adj = _build_adj8(cells)
    assert start in adj, "path start must belong to the path"
    ordered = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxts = [nxt for nxt in adj[cur] if nxt != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path must not branch"
        nxt = nxts[0]
        ordered.append(nxt)
        prev, cur = cur, nxt
    assert len(ordered) == len(cells), "ordered path must cover the whole component"
    return ordered


def _segment_dirs(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    dirs: list[tuple[int, int]] = []
    for (r0, c0), (r1, c1) in zip(points, points[1:], strict=False):
        step = (r1 - r0, c1 - c0)
        assert step in DIR_TO_INDEX, "steps must be horizontal, vertical, or diagonal"
        if not dirs or dirs[-1] != step:
            dirs.append(step)
    return dirs


def _relative_turn(d1: tuple[int, int], d2: tuple[int, int]) -> str:
    x1, y1 = d1[1], -d1[0]
    x2, y2 = d2[1], -d2[0]
    cross = x1 * y2 - y1 * x2
    assert cross != 0, "turn must not be straight or reversed"
    return "L" if cross > 0 else "R"


def _turn_pattern(points: list[tuple[int, int]]) -> tuple[str, str]:
    dirs = _segment_dirs(points)
    assert len(dirs) >= 3, "path must contain at least two turns"
    turns = [_relative_turn(dirs[i], dirs[i + 1]) for i in range(len(dirs) - 1)]
    return turns[0], turns[1]


def _is_rect_frame(g: np.ndarray, cells: list[tuple[int, int]], color: int) -> bool:
    r0, c0, r1, c1 = _bbox(cells)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    border_count = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    if len(cells) != border_count:
        return False
    for r in range(r0, r1 + 1):
        if g[r, c0] != color or g[r, c1] != color:
            return False
    for c in range(c0, c1 + 1):
        if g[r0, c] != color or g[r1, c] != color:
            return False
    return True


def _find_legend_bbox(g: np.ndarray) -> tuple[int, int, int, int]:
    h, w = g.shape
    best: tuple[int, int, int, int] | None = None
    best_area = -1
    for comp in _components(g == GREY, connectivity=4):
        if not _is_rect_frame(g, comp, GREY):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if not ((r0 == 0 or r1 == h - 1) and (c0 == 0 or c1 == w - 1)):
            continue
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0:
            continue
        if not np.any(inner == MAROON):
            continue
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area > best_area:
            best_area = area
            best = (r0, c0, r1, c1)
    assert best is not None, "legend frame not found"
    return best


def _legend_mapping(g: np.ndarray, legend_bbox: tuple[int, int, int, int]) -> dict[tuple[str, str], int]:
    r0, c0, r1, c1 = legend_bbox
    inner_mask = np.zeros_like(g, dtype=bool)
    inner_mask[r0 + 1 : r1, c0 + 1 : c1] = True

    maroon_mask = inner_mask & (g == MAROON)
    maroon_comps = _components(maroon_mask, connectivity=8)

    blocks: list[tuple[float, int, list[tuple[int, int]]]] = []
    for color in sorted(int(v) for v in np.unique(g)):
        if color in (BG, GREY, MAROON):
            continue
        for comp in _components(inner_mask & (g == color), connectivity=4):
            br0, bc0, br1, bc1 = _bbox(comp)
            if br1 - br0 == 1 and bc1 - bc0 == 1 and len(comp) == 4:
                blocks.append(((br0 + br1) / 2.0, color, comp))

    maroon_comps.sort(key=lambda comp: (_bbox(comp)[0] + _bbox(comp)[2]) / 2.0)
    blocks.sort(key=lambda item: item[0])
    assert maroon_comps and len(maroon_comps) == len(blocks), "legend rows must pair paths with 2x2 blocks"

    mapping: dict[tuple[str, str], int] = {}
    for comp, (_row, color, block) in zip(maroon_comps, blocks, strict=False):
        endpoints = _path_endpoints(comp)
        assert len(endpoints) == 2, "legend sample must be a simple path"
        block_center = np.mean(np.array(block, dtype=float), axis=0)

        def dist2(p: tuple[int, int]) -> float:
            pr, pc = p
            return float((pr - block_center[0]) ** 2 + (pc - block_center[1]) ** 2)

        start = endpoints[0] if dist2(endpoints[0]) >= dist2(endpoints[1]) else endpoints[1]
        ordered = _order_path(comp, start)
        mapping[_turn_pattern(ordered)] = color
    assert mapping, "legend mapping must be non-empty"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each path endpoint cross using the legend row whose maroon turn sketch
    matches the path's first two relative turns.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(np.bincount(g.ravel()).argmax())
    out = g.copy()

    legend_bbox = _find_legend_bbox(g)
    mapping = _legend_mapping(g, legend_bbox)

    r0, c0, r1, c1 = legend_bbox
    outside_legend = np.ones_like(g, dtype=bool)
    outside_legend[r0 : r1 + 1, c0 : c1 + 1] = False

    for color in sorted(int(v) for v in np.unique(g)):
        if color in (bg, GREY, MAROON):
            continue
        mask = (g == color) & outside_legend
        for comp in _components(mask, connectivity=8):
            endpoints = _path_endpoints(comp)
            assert len(endpoints) == 2, "every path must have exactly two endpoints"
            border_endpoints = [
                p
                for p in endpoints
                if p[0] == 0 or p[0] == g.shape[0] - 1 or p[1] == 0 or p[1] == g.shape[1] - 1
            ]
            assert len(border_endpoints) == 1, "each path must start on the outer border exactly once"
            ordered = _order_path(comp, border_endpoints[0])
            pattern = _turn_pattern(ordered)
            fill = mapping[pattern]
            er, ec = ordered[-1]
            out[er, ec] = fill
            for dr, dc in CARD4:
                nr, nc = er + dr, ec + dc
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and g[nr, nc] == bg:
                    out[nr, nc] = fill

    return out


def _build_polyline(
    start_dir_idx: int,
    turns: list[str],
    lengths: list[int],
) -> list[tuple[int, int]] | None:
    dirs = [start_dir_idx]
    for turn in turns:
        step = 1 if turn == "L" else -1
        dirs.append((dirs[-1] + step) % 8)
    assert len(dirs) == len(lengths)

    points = [(0, 0)]
    occupied = {(0, 0)}
    for dir_idx, seg_len in zip(dirs, lengths, strict=False):
        dr, dc = DIRS8[dir_idx]
        for _ in range(seg_len):
            nr = points[-1][0] + dr
            nc = points[-1][1] + dc
            if (nr, nc) in occupied:
                return None
            for pr, pc in occupied:
                if (pr, pc) == points[-1]:
                    continue
                if max(abs(nr - pr), abs(nc - pc)) <= 1:
                    return None
            points.append((nr, nc))
            occupied.add((nr, nc))
    return points


def _normalize(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in points)
    min_c = min(c for _, c in points)
    return [(r - min_r, c - min_c) for r, c in points]


def _legend_sample(pattern: tuple[str, str]) -> list[tuple[int, int]]:
    for _ in range(200):
        start_dir = random.choice([0, 1, 7])
        lengths = [random.randint(2, 3) for _ in range(3)]
        points = _build_polyline(start_dir, list(pattern), lengths)
        if points is None:
            continue
        if points[-1][1] <= points[0][1]:
            continue
        norm = _normalize(points)
        r0, c0, r1, c1 = _bbox(norm)
        if r1 - r0 + 1 <= 3 and c1 - c0 + 1 <= 7:
            return norm
    raise RuntimeError("failed to build legend sample")


def _build_legend_rows(pattern_to_color: dict[tuple[str, str], int]) -> np.ndarray:
    row_order = TURN_PATTERNS[:]
    random.shuffle(row_order)
    row_patches: list[np.ndarray] = []
    row_height = 4

    for pattern in row_order:
        sample = _legend_sample(pattern)
        sample_h = max(r for r, _ in sample) + 1
        sample_w = max(c for _, c in sample) + 1
        top_pad = random.randint(0, row_height - sample_h)
        left_pad = random.randint(0, 1)
        gap = random.randint(1, 2)
        block_col = left_pad + sample_w + gap
        row_w = block_col + 2 + 1
        patch = np.full((row_height, row_w), BG, dtype=int)
        for r, c in sample:
            patch[top_pad + r, left_pad + c] = MAROON
        patch[1:3, block_col : block_col + 2] = pattern_to_color[pattern]
        row_patches.append(patch)

    inner_h = len(row_patches) * row_height + (len(row_patches) - 1)
    inner_w = max(p.shape[1] for p in row_patches)
    legend = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    legend[0, :] = GREY
    legend[-1, :] = GREY
    legend[:, 0] = GREY
    legend[:, -1] = GREY

    r = 1
    for patch in row_patches:
        legend[r : r + patch.shape[0], 1 : 1 + patch.shape[1]] = patch
        r += patch.shape[0] + 1
    return legend


def _place_legend(grid: np.ndarray, legend: np.ndarray, corner: str) -> tuple[int, int, int, int]:
    h, w = grid.shape
    lh, lw = legend.shape
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - lw
    elif corner == "bl":
        r0, c0 = h - lh, 0
    else:
        r0, c0 = h - lh, w - lw
    grid[r0 : r0 + lh, c0 : c0 + lw] = legend
    return r0, c0, r0 + lh - 1, c0 + lw - 1


def _mark_blocked(blocked: np.ndarray, cells: set[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _start_dir_for_side(side: str) -> int:
    return {"top": 6, "bottom": 2, "left": 0, "right": 4}[side]


def _side_shape_ok(points: list[tuple[int, int]], side: str) -> bool:
    if side == "top":
        return all(r >= 0 for r, _ in points) and all(r > 0 for r, _ in points[1:])
    if side == "bottom":
        return all(r <= 0 for r, _ in points) and all(r < 0 for r, _ in points[1:])
    if side == "left":
        return all(c >= 0 for _, c in points) and all(c > 0 for _, c in points[1:])
    return all(c <= 0 for _, c in points) and all(c < 0 for _, c in points[1:])


def _endpoint_cross_clear(points: list[tuple[int, int]]) -> bool:
    path_set = set(points)
    er, ec = points[-1]
    for dr, dc in CARD4:
        if (er + dr, ec + dc) in path_set:
            return False
    return True


def _actual_path(side: str, pattern: tuple[str, str]) -> list[tuple[int, int]]:
    start_dir = _start_dir_for_side(side)
    for _ in range(400):
        turns = [pattern[0], pattern[1], random.choice(["L", "R"])]
        lengths = [
            random.randint(3, 6),
            random.randint(2, 5),
            random.randint(2, 5),
            random.randint(2, 5),
        ]
        points = _build_polyline(start_dir, turns, lengths)
        if points is None:
            continue
        if not _side_shape_ok(points, side):
            continue
        if not _endpoint_cross_clear(points):
            continue
        br0, bc0, br1, bc1 = _bbox(points)
        if br1 - br0 + 1 > 12 or bc1 - bc0 + 1 > 12:
            continue
        if tuple(map(abs, points[-1])) == (0, 0):
            continue
        if _turn_pattern(points) == pattern:
            return points
    raise RuntimeError("failed to build actual path")


def _path_translation_candidates(
    points: list[tuple[int, int]],
    side: str,
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    min_r, min_c, max_r, max_c = _bbox(points)
    candidates: list[tuple[int, int]] = []
    if side in ("top", "bottom"):
        row_off = 0 if side == "top" else h - 1
        for col_off in range(-min_c, w - max_c):
            candidates.append((row_off, col_off))
    else:
        col_off = 0 if side == "left" else w - 1
        for row_off in range(-min_r, h - max_r):
            candidates.append((row_off, col_off))
    random.shuffle(candidates)
    return candidates


def _translate(points: list[tuple[int, int]], off_r: int, off_c: int) -> list[tuple[int, int]]:
    return [(r + off_r, c + off_c) for r, c in points]


def _can_place_path(
    blocked: np.ndarray,
    abs_points: list[tuple[int, int]],
    side: str,
) -> bool:
    h, w = blocked.shape
    start = abs_points[0]
    if side == "top" and start[0] != 0:
        return False
    if side == "bottom" and start[0] != h - 1:
        return False
    if side == "left" and start[1] != 0:
        return False
    if side == "right" and start[1] != w - 1:
        return False

    for r, c in abs_points:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if blocked[r, c]:
            return False
    for r, c in abs_points[1:]:
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            return False

    end_r, end_c = abs_points[-1]
    if not (0 < end_r < h - 1 and 0 < end_c < w - 1):
        return False

    path_set = set(abs_points)
    for dr, dc in CARD4:
        nr, nc = end_r + dr, end_c + dc
        if blocked[nr, nc] or (nr, nc) in path_set:
            return False
    return True


def _expected_output(
    grid: np.ndarray,
    paths: list[dict[str, object]],
    pattern_to_color: dict[tuple[str, str], int],
) -> np.ndarray:
    out = grid.copy()
    for info in paths:
        end_r, end_c = info["end"]
        fill = pattern_to_color[info["pattern"]]
        out[end_r, end_c] = fill
        for dr, dc in CARD4:
            nr, nc = end_r + dr, end_c + dc
            if grid[nr, nc] == BG:
                out[nr, nc] = fill
    return out


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(26, 34)
    grid_w = random.randint(26, 34)
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    legend_colors = random.sample(NON_SPECIAL_COLORS, 4)
    path_colors = [c for c in NON_SPECIAL_COLORS if c not in legend_colors]
    random.shuffle(path_colors)
    path_count = random.randint(2, min(3, len(path_colors)))
    pattern_to_color = {pattern: color for pattern, color in zip(TURN_PATTERNS, legend_colors, strict=False)}

    legend = _build_legend_rows(pattern_to_color)
    if legend.shape[0] > grid_h or legend.shape[1] > grid_w:
        return None
    legend_bbox = _place_legend(grid, legend, random.choice(["tl", "tr", "bl", "br"]))

    blocked = np.zeros_like(grid, dtype=bool)
    lr0, lc0, lr1, lc1 = legend_bbox
    legend_cells = {(r, c) for r in range(lr0, lr1 + 1) for c in range(lc0, lc1 + 1)}
    _mark_blocked(blocked, legend_cells)

    patterns = TURN_PATTERNS[:]
    random.shuffle(patterns)
    chosen_patterns = patterns[:path_count]
    paths: list[dict[str, object]] = []

    for path_color, pattern in zip(path_colors[:path_count], chosen_patterns, strict=False):
        placed = False
        sides = ["top", "bottom", "left", "right"]
        random.shuffle(sides)
        for side in sides:
            rel_points = _actual_path(side, pattern)
            for off_r, off_c in _path_translation_candidates(rel_points, side, grid.shape):
                abs_points = _translate(rel_points, off_r, off_c)
                if not _can_place_path(blocked, abs_points, side):
                    continue
                for r, c in abs_points:
                    grid[r, c] = path_color
                end_r, end_c = abs_points[-1]
                reserved = set(abs_points)
                reserved.update((end_r + dr, end_c + dc) for dr, dc in CARD4)
                _mark_blocked(blocked, reserved)
                paths.append({"pattern": pattern, "end": (end_r, end_c)})
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    expected = _expected_output(grid, paths, pattern_to_color)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
