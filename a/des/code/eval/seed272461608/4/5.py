from __future__ import annotations

from collections import deque
from itertools import combinations
import random

import numpy as np


BG = 0
YELLOW = 4
WIRE_COLORS = (1, 2, 3, 5, 6, 8, 9)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _segment_endpoints(coords: np.ndarray) -> list[tuple[int, int]]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0 = int(rows.min())
    r1 = int(rows.max())
    c0 = int(cols.min())
    c1 = int(cols.max())

    if r0 == r1:
        return [(r0, c0), (r0, c1)]
    if c0 == c1:
        return [(r0, c0), (r1, c0)]
    raise AssertionError("each input fragment must be a straight segment")


def _path_between(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]] | None:
    ra, ca = a
    rb, cb = b
    if ra == rb:
        lo, hi = sorted((ca, cb))
        return [(ra, c) for c in range(lo, hi + 1)]
    if ca == cb:
        lo, hi = sorted((ra, rb))
        return [(r, ca) for r in range(lo, hi + 1)]
    return None


def _clear_path(grid: np.ndarray, path: list[tuple[int, int]], color: int) -> bool:
    for i, (r, c) in enumerate(path):
        value = int(grid[r, c])
        if i == 0 or i == len(path) - 1:
            if value != int(color):
                return False
        elif value != BG:
            return False
    return True


def _candidate_connections(grid: np.ndarray, color: int) -> list[dict[str, object]]:
    comps = _components(grid == int(color))
    endpoints: list[dict[str, object]] = []
    for comp_id, coords in enumerate(comps):
        for endpoint in _segment_endpoints(coords):
            endpoints.append({"comp": comp_id, "pos": endpoint})

    candidates: list[dict[str, object]] = []
    for i, j in combinations(range(len(endpoints)), 2):
        ep_i = endpoints[i]
        ep_j = endpoints[j]
        if int(ep_i["comp"]) == int(ep_j["comp"]):
            continue
        path = _path_between(ep_i["pos"], ep_j["pos"])
        if path is None or not _clear_path(grid, path, color):
            continue
        candidates.append(
            {
                "a": i,
                "b": j,
                "length": len(path),
                "path": path,
                "key": (
                    min(ep_i["pos"], ep_j["pos"]),
                    max(ep_i["pos"], ep_j["pos"]),
                ),
            }
        )
    return candidates


def _best_matching(candidates: list[dict[str, object]], endpoint_count: int) -> list[dict[str, object]]:
    edges_by_node: dict[int, list[int]] = {i: [] for i in range(endpoint_count)}
    for edge_idx, edge in enumerate(candidates):
        edges_by_node[int(edge["a"])].append(edge_idx)
        edges_by_node[int(edge["b"])].append(edge_idx)

    def better(
        first: tuple[int, int, tuple[tuple[tuple[int, int], tuple[int, int]], ...]],
        second: tuple[int, int, tuple[tuple[tuple[int, int], tuple[int, int]], ...]],
    ) -> bool:
        if first[0] != second[0]:
            return first[0] > second[0]
        if first[1] != second[1]:
            return first[1] < second[1]
        return first[2] < second[2]

    def dfs(
        used_mask: int,
    ) -> tuple[int, int, tuple[tuple[tuple[int, int], tuple[int, int]], ...], list[int]]:
        first_free = None
        for node in range(endpoint_count):
            if (used_mask >> node) & 1 == 0:
                first_free = node
                break
        if first_free is None:
            return (0, 0, tuple(), [])

        best = dfs(used_mask | (1 << first_free))
        for edge_idx in edges_by_node[first_free]:
            edge = candidates[edge_idx]
            a = int(edge["a"])
            b = int(edge["b"])
            other = b if a == first_free else a
            if (used_mask >> other) & 1:
                continue
            sub = dfs(used_mask | (1 << first_free) | (1 << other))
            score = (
                sub[0] + 1,
                sub[1] + int(edge["length"]),
                tuple(sorted(sub[2] + (edge["key"],))),
                [edge_idx] + sub[3],
            )
            if better(score[:3], best[:3]):
                best = score
        return best

    return [candidates[i] for i in dfs(0)[3]]


def _connect_color(grid: np.ndarray, color: int) -> np.ndarray:
    out = grid.copy()
    comps = _components(grid == int(color))
    endpoint_count = 2 * len(comps)
    candidates = _candidate_connections(grid, color)
    for edge in _best_matching(candidates, endpoint_count):
        for r, c in edge["path"]:
            out[r, c] = int(color)
    return out


def _enclosed_cells(grid: np.ndarray, color: int) -> np.ndarray:
    h, w = grid.shape
    reachable = np.zeros((h, w), dtype=bool)
    q = deque()

    for r in range(h):
        for c in (0, w - 1):
            if int(grid[r, c]) != int(color) and not bool(reachable[r, c]):
                reachable[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if int(grid[r, c]) != int(color) and not bool(reachable[r, c]):
                reachable[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != int(color) and not bool(reachable[nr, nc]):
                reachable[nr, nc] = True
                q.append((nr, nc))

    return (grid != int(color)) & ~reachable


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color segment endpoints along empty straight lines, then light
    yellow bulbs enclosed by any completed same-color loop.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    colors = sorted(int(color) for color in np.unique(g) if int(color) not in {BG, YELLOW})
    for color in colors:
        out = _connect_color(out, color)

    for color in colors:
        inside = _enclosed_cells(out, color)
        out[(out == YELLOW) & inside] = int(color)

    return out


GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
CIRCUIT_COUNT_CHOICES = (1, 2, 3)
PATCH_SIDE_MIN = 9
PATCH_SIDE_MAX = 13
GLOBAL_BULB_COUNT_RANGE = (1, 5)
GENERATION_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 120


def _draw_h(grid: np.ndarray, row: int, c0: int, c1: int, color: int) -> None:
    grid[row, c0 : c1 + 1] = int(color)


def _draw_v(grid: np.ndarray, col: int, r0: int, r1: int, color: int) -> None:
    grid[r0 : r1 + 1, col] = int(color)


def _sample_shortened(span: int) -> tuple[int, int]:
    while True:
        start_gap = random.randint(2, max(2, span - 4))
        end_gap = random.randint(2, max(2, span - start_gap - 2))
        if span - start_gap - end_gap >= 2:
            return start_gap, end_gap


def _segment_list(count: int, closed: bool, h: int, w: int) -> list[tuple[str, int, int, int]]:
    r0, r1 = 1, h - 2
    c0, c1 = 1, w - 2
    span_h = r1 - r0 + 1
    span_w = c1 - c0 + 1

    segments: list[tuple[str, int, int, int]] = [("h", r0, c0, c1)]

    if closed:
        segments.append(("h", r1, c0, c1))
        if count >= 3:
            side = random.choice(("left", "right"))
            top_gap, bottom_gap = _sample_shortened(span_h)
            col = c0 if side == "left" else c1
            segments.append(("v", col, r0 + top_gap, r1 - bottom_gap))
        if count == 4:
            side = "right" if any(seg[1] == c0 for seg in segments if seg[0] == "v") else "left"
            top_gap, bottom_gap = _sample_shortened(span_h)
            col = c0 if side == "left" else c1
            segments.append(("v", col, r0 + top_gap, r1 - bottom_gap))
    else:
        align_side = random.choice(("left", "right"))
        shift = random.randint(2, span_w - 2)
        if align_side == "left":
            segments.append(("h", r1, c0, c1 - shift))
            open_side = "right"
        else:
            segments.append(("h", r1, c0 + shift, c1))
            open_side = "left"

        if count >= 3:
            top_gap, bottom_gap = _sample_shortened(span_h)
            col = c0 if align_side == "left" else c1
            segments.append(("v", col, r0 + top_gap, r1 - bottom_gap))
        if count == 4:
            top_gap, bottom_gap = _sample_shortened(span_h)
            col = c0 if open_side == "left" else c1
            segments.append(("v", col, r0 + top_gap, r1 - bottom_gap))

    return segments


def _interior_cells(h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(2, h - 2) for c in range(2, w - 2)]


def _outside_cells(h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if r in {0, h - 1} or c in {0, w - 1}:
                cells.append((r, c))
    return cells


def _sample_cells(cells: list[tuple[int, int]], count_range: tuple[int, int]) -> list[tuple[int, int]]:
    if not cells:
        return []
    lo = min(count_range[0], len(cells))
    hi = min(count_range[1], len(cells))
    if hi <= 0:
        return []
    if lo > hi:
        lo = hi
    return random.sample(cells, random.randint(lo, hi))


def _build_patch(color: int, count: int, closed: bool) -> np.ndarray:
    h = random.randint(PATCH_SIDE_MIN, PATCH_SIDE_MAX)
    w = random.randint(PATCH_SIDE_MIN, PATCH_SIDE_MAX)
    patch = np.full((h, w), BG, dtype=int)

    for orient, fixed, start, end in _segment_list(count, closed, h, w):
        if orient == "h":
            _draw_h(patch, fixed, start, end, color)
        else:
            _draw_v(patch, fixed, start, end, color)

    for r, c in _sample_cells(_interior_cells(h, w), (1, 3)):
        if int(patch[r, c]) == BG:
            patch[r, c] = YELLOW

    for r, c in _sample_cells(_outside_cells(h, w), (0, 2)):
        if int(patch[r, c]) == BG:
            patch[r, c] = YELLOW

    if random.random() < 0.5:
        patch = patch.T.copy()

    return patch


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    block = patch != BG
    grid[top : top + h, left : left + w][block] = patch[block]


def _place_patches(grid: np.ndarray, patches: list[np.ndarray]) -> bool:
    h, w = grid.shape
    occupied = np.zeros((h, w), dtype=bool)

    for patch in patches:
        ph, pw = patch.shape
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(0, h - ph)
            left = random.randint(0, w - pw)
            r0 = max(0, top - 1)
            c0 = max(0, left - 1)
            r1 = min(h, top + ph + 1)
            c1 = min(w, left + pw + 1)
            if bool(occupied[r0:r1, c0:c1].any()):
                continue
            _paste_patch(grid, patch, top, left)
            occupied[r0:r1, c0:c1] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _add_global_bulbs(grid: np.ndarray) -> None:
    bg_cells = [(int(r), int(c)) for r, c in np.argwhere(grid == BG)]
    if not bg_cells:
        return
    count = min(len(bg_cells), random.randint(*GLOBAL_BULB_COUNT_RANGE))
    for r, c in random.sample(bg_cells, count):
        grid[r, c] = YELLOW


def _scene_signature(grid: np.ndarray) -> tuple[int, ...]:
    non_bg = tuple(sorted(int(v) for v in np.unique(grid) if int(v) != BG))
    return (grid.shape[0], grid.shape[1], len(non_bg), int((grid == YELLOW).sum()))


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        h = random.randint(*GRID_H_RANGE)
        w = random.randint(*GRID_W_RANGE)
        grid = np.full((h, w), BG, dtype=int)

        circuit_count = random.choice(CIRCUIT_COUNT_CHOICES)
        colors = random.sample(WIRE_COLORS, circuit_count)

        closed_flags = [random.choice((False, True)) for _ in range(circuit_count)]
        if circuit_count >= 2 and all(closed_flags):
            closed_flags[random.randrange(circuit_count)] = False
        if circuit_count >= 2 and not any(closed_flags):
            closed_flags[random.randrange(circuit_count)] = True

        fragment_counts = [random.choice((2, 3, 4)) for _ in range(circuit_count)]
        patches = [
            _build_patch(color, count, closed)
            for color, count, closed in zip(colors, fragment_counts, closed_flags)
        ]

        if not _place_patches(grid, patches):
            continue
        _add_global_bulbs(grid)

        out = main(grid)
        if not isinstance(out, np.ndarray) or out.shape != grid.shape:
            continue
        if np.array_equal(out, grid):
            continue
        if _scene_signature(grid) == _scene_signature(out):
            continue
        return grid

    raise RuntimeError("failed to generate valid ARC circuit input")
