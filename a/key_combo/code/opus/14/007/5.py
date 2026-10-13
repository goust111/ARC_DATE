from __future__ import annotations

from collections import deque
import random

import numpy as np


_NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_ARM_DELTAS = [(-2, 0), (-1, 0), (0, -2), (0, -1), (0, 1), (0, 2), (1, 0), (2, 0)]
_CENTER_DELTAS = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_MOTIF_DELTAS = _ARM_DELTAS + _CENTER_DELTAS
_PATTERNS = [
    [(-2, 1), (-1, 6), (1, -3), (2, 3)],
    [(-2, 1), (-1, 6), (1, -4), (2, 3)],
    [(-2, 1), (-1, 6), (0, -2), (2, 3)],
    [(-2, 1), (-1, 6), (0, -4), (2, 3)],
    [(-2, 1), (-1, 6), (-1, -3), (2, 3)],
    [(-2, 1), (-1, 6), (-1, -4), (2, 3)],
    [(-2, 1), (-1, 5), (1, -2), (2, 3)],
]


def _transform_point(pt: tuple[int, int], turns: int, flip: bool) -> tuple[int, int]:
    r, c = pt
    for _ in range(turns % 4):
        r, c = c, -r
    if flip:
        c = -c
    return int(r), int(c)


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(coords)
    return components


def _extract_crosses(grid: np.ndarray) -> list[dict]:
    crosses: list[dict] = []
    for coords in _component_coords(grid != 0):
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 != 2 or c1 - c0 != 2:
            raise ValueError("expected 3x3 crosses")
        cr, cc = r0 + 1, c0 + 1
        center_color = int(grid[cr, cc])
        arm_positions = [(cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)]
        arm_colors = {int(grid[r, c]) for r, c in arm_positions}
        if len(arm_colors) != 1:
            raise ValueError("cross arm colors must match")
        arm_color = int(next(iter(arm_colors)))
        crosses.append(
            {
                "center": (cr, cc),
                "arm_color": arm_color,
                "center_color": center_color,
            }
        )
    crosses.sort(key=lambda item: item["center"])
    return crosses


def _paint_motif(out: np.ndarray, owner: np.ndarray, idx: int, cross: dict) -> None:
    cr, cc = cross["center"]
    for dr, dc in _ARM_DELTAS:
        rr, cc2 = cr + dr, cc + dc
        out[rr, cc2] = cross["arm_color"]
        owner[rr, cc2] = idx
    for dr, dc in _CENTER_DELTAS:
        rr, cc2 = cr + dr, cc + dc
        out[rr, cc2] = cross["center_color"]
        owner[rr, cc2] = idx


def _expanded_grid(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    crosses = _extract_crosses(grid)
    out = np.array(grid, dtype=int, copy=True)
    owner = np.full(grid.shape, -1, dtype=int)
    for idx, cross in enumerate(crosses):
        _paint_motif(out, owner, idx, cross)
    return out, owner, crosses


def _enclosed_regions(expanded: np.ndarray, owner: np.ndarray, crosses: list[dict]) -> list[dict]:
    h, w = expanded.shape
    regions: list[dict] = []
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if expanded[r, c] != 0 or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            touches_border = False
            adjacent: set[int] = set()
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for dr, dc in _NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if expanded[nr, nc] == 0 and not seen[nr, nc]:
                            seen[nr, nc] = True
                            q.append((nr, nc))
                        elif expanded[nr, nc] != 0:
                            adjacent.add(int(owner[nr, nc]))
            if touches_border:
                continue
            centroid_r = sum(r0 for r0, _ in coords) / len(coords)
            centroid_c = sum(c0 for _, c0 in coords) / len(coords)
            distances = []
            for idx, cross in enumerate(crosses):
                cr, cc = cross["center"]
                dist2 = (cr - centroid_r) ** 2 + (cc - centroid_c) ** 2
                distances.append((dist2, cr, cc, idx))
            distances.sort()
            unique_nearest = abs(distances[0][0] - distances[1][0]) > 1e-9 if len(distances) > 1 else True
            regions.append(
                {
                    "coords": coords,
                    "adjacent_ids": adjacent,
                    "k": len(adjacent),
                    "nearest_idx": int(distances[0][3]),
                    "unique_nearest": unique_nearest,
                }
            )
    return regions


def _solve(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    expanded, owner, crosses = _expanded_grid(grid)
    out = expanded.copy()
    regions = _enclosed_regions(expanded, owner, crosses)
    for region in regions:
        cross = crosses[region["nearest_idx"]]
        fill_color = cross["center_color"] if region["k"] % 2 == 1 else cross["arm_color"]
        for r, c in region["coords"]:
            out[r, c] = fill_color
    return out, regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each two-color cross, then fill each enclosed corridor by the parity of its neighboring motifs.

    Input structure:
    - A black background contains 3x3 cross objects as separate 4-connected components.
    - Each cross has one arm color on its four cardinal arm cells and a different center color on its middle cell.
    - The crosses are spaced so their 5x5 expansions do not overlap, but some of the black gaps between expansions become enclosed regions.

    Rule:
    - Expand every cross into a 5x5 motif: keep the arm color on the orthogonal line and extend it one more step north, south, east, and west; keep the center color at the center and copy it to the four diagonal neighbors.
    - Find every 4-connected black region in the expanded grid that does not touch the grid boundary.
    - For each enclosed region, count how many distinct expanded cross motifs are 4-neighbor adjacent to that region, then find the nearest cross center to the region centroid by Euclidean distance.
    - Fill the region with that nearest cross's center color when the adjacency count is odd, or with that cross's arm color when the adjacency count is even; leave the boundary-connected exterior black region unchanged.

    Key insight:
    - The cross already encodes two roles, arm versus center, and the enclosed-region parity decides which of those two role colors from the nearest cross should be propagated into each cavity.
    """
    out, _regions = _solve(np.asarray(grid, dtype=int))
    return out


def _non_overlapping(center: tuple[int, int], others: list[tuple[int, int]]) -> bool:
    cr, cc = center
    this_cells = {(cr + dr, cc + dc) for dr, dc in _MOTIF_DELTAS}
    for orow, ocol in others:
        other_cells = {(orow + dr, ocol + dc) for dr, dc in _MOTIF_DELTAS}
        if this_cells & other_cells:
            return False
    return True


def _sample_color_pairs(count: int) -> list[tuple[int, int]]:
    colors = list(range(1, 10))
    if count <= 4:
        chosen = random.sample(colors, 2 * count)
        return [(chosen[2 * i], chosen[2 * i + 1]) for i in range(count)]
    pairs: list[tuple[int, int]] = []
    used_pairs: set[tuple[int, int]] = set()
    while len(pairs) < count:
        arm = random.choice(colors)
        center = random.choice(colors)
        if arm == center or (arm, center) in used_pairs:
            continue
        used_pairs.add((arm, center))
        pairs.append((arm, center))
    return pairs


def _draw_cross(grid: np.ndarray, center: tuple[int, int], arm_color: int, center_color: int) -> None:
    r, c = center
    grid[r, c] = center_color
    grid[r - 1, c] = arm_color
    grid[r + 1, c] = arm_color
    grid[r, c - 1] = arm_color
    grid[r, c + 1] = arm_color


def _pattern_centers(h: int, w: int) -> list[tuple[int, int]]:
    base = random.choice(_PATTERNS)
    turns = random.randint(0, 3)
    flip = random.choice([False, True])
    transformed = [_transform_point(pt, turns, flip) for pt in base]
    min_r = min(r for r, _ in transformed)
    max_r = max(r for r, _ in transformed)
    min_c = min(c for _, c in transformed)
    max_c = max(c for _, c in transformed)
    r_shift = random.randint(3 - min_r, h - 4 - max_r)
    c_shift = random.randint(3 - min_c, w - 4 - max_c)
    centers = [(r + r_shift, c + c_shift) for r, c in transformed]
    if random.random() < 0.35:
        candidates = [(r, c) for r in range(3, h - 3) for c in range(3, w - 3)]
        random.shuffle(candidates)
        for extra in candidates:
            if _non_overlapping(extra, centers):
                cluster_rows = [r for r, _ in centers]
                cluster_cols = [c for _, c in centers]
                if (
                    extra[0] < min(cluster_rows) - 5
                    or extra[0] > max(cluster_rows) + 5
                    or extra[1] < min(cluster_cols) - 5
                    or extra[1] > max(cluster_cols) + 5
                ):
                    centers.append(extra)
                    break
    centers.sort()
    return centers


def generate_input() -> np.ndarray:
    for attempt in range(300):
        h = random.randint(17, 24)
        w = random.randint(17, 24)
        centers = _pattern_centers(h, w)
        grid = np.zeros((h, w), dtype=int)
        for center, (arm_color, center_color) in zip(centers, _sample_color_pairs(len(centers))):
            _draw_cross(grid, center, arm_color, center_color)

        out, regions = _solve(grid)
        if len(regions) < 2:
            continue
        if {region["k"] % 2 for region in regions} != {0, 1}:
            continue
        if attempt < 120 and not all(region["unique_nearest"] for region in regions):
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle input")
