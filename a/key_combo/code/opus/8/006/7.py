from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
LINE = 1
MARKER_COLORS = (2, 3, 4, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _endpoints(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    pts: list[tuple[int, int, int, int]] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if grid[r, c] != LINE:
                continue
            neighbors = []
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] == LINE:
                    neighbors.append((dr, dc))
            if len(neighbors) == 1:
                dr, dc = neighbors[0]
                pts.append((r, c, -dr, -dc))
    return pts


def _fill_facing_gaps(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    endpoints = _endpoints(grid)
    horiz: dict[int, list[tuple[int, int, int]]] = {}
    vert: dict[int, list[tuple[int, int, int]]] = {}
    for r, c, dr, dc in endpoints:
        if dr == 0:
            horiz.setdefault(r, []).append((c, dr, dc))
        else:
            vert.setdefault(c, []).append((r, dr, dc))

    fills: set[tuple[int, int]] = set()

    for r, items in horiz.items():
        items.sort()
        for i in range(len(items)):
            c1, dr1, dc1 = items[i]
            if (dr1, dc1) != (0, 1):
                continue
            for j in range(i + 1, len(items)):
                c2, dr2, dc2 = items[j]
                gap = c2 - c1 - 1
                if gap > 3:
                    break
                if (dr2, dc2) != (0, -1) or gap < 1:
                    continue
                if np.all(grid[r, c1 + 1 : c2] == BG):
                    fills.update((r, cc) for cc in range(c1 + 1, c2))
                    break

    for c, items in vert.items():
        items.sort()
        for i in range(len(items)):
            r1, dr1, dc1 = items[i]
            if (dr1, dc1) != (1, 0):
                continue
            for j in range(i + 1, len(items)):
                r2, dr2, dc2 = items[j]
                gap = r2 - r1 - 1
                if gap > 3:
                    break
                if (dr2, dc2) != (-1, 0) or gap < 1:
                    continue
                if np.all(grid[r1 + 1 : r2, c] == BG):
                    fills.update((rr, c) for rr in range(r1 + 1, r2))
                    break

    for r, c in fills:
        out[r, c] = LINE
    return out


def _closed_cavities(grid: np.ndarray) -> list[set[tuple[int, int]]]:
    mask = grid != LINE
    regions: list[set[tuple[int, int]]] = []
    h, w = grid.shape
    for comp in _components(mask):
        pts = set(comp)
        if any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in pts):
            continue
        regions.append(pts)
    return regions


def _bg_has_internal_pixel(grid: np.ndarray, bg_cells: set[tuple[int, int]]) -> bool:
    for r, c in bg_cells:
        ok = True
        for dr, dc in DIRS8:
            nr, nc = r + dr, c + dc
            if not _in_bounds(grid.shape, nr, nc):
                ok = False
                break
            if grid[nr, nc] != LINE and (nr, nc) not in bg_cells:
                ok = False
                break
        if ok:
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill short blue gaps so open line fragments become closed rooms, then only paint rooms that contain a true interior pixel.

    Input structure:
    - The background is black (0).
    - Blue (1) pixels form a sparse one-cell-thick orthogonal line network with short missing segments.
    - Non-blue singleton marker pixels lie inside regions that would become enclosed after the blue gaps are repaired.

    Rule:
    - Find blue endpoints, meaning blue pixels with exactly one blue 4-neighbor.
    - For endpoint pairs in the same row or column that face each other across 1 to 3 background cells, fill the whole gap with blue.
    - In the repaired grid, find every 4-connected non-blue region that does not touch the outer border.
    - If such a region contains at least one background pixel whose full 3x3 neighborhood stays inside that region or on blue border pixels, fill that region's background pixels with its unique marker color; otherwise leave the region unfilled.

    Key insight:
    - The decisive step is not the original fragments but the enclosed regions that emerge only after repairing short facing gaps; only geometrically thick regions receive the marker color.
    """
    g = np.asarray(grid, dtype=int)
    repaired = _fill_facing_gaps(g)
    out = repaired.copy()

    for cavity in _closed_cavities(repaired):
        marker_colors = {int(repaired[r, c]) for r, c in cavity if repaired[r, c] not in (BG, LINE)}
        if len(marker_colors) != 1:
            continue
        bg_cells = {(r, c) for r, c in cavity if repaired[r, c] == BG}
        if not _bg_has_internal_pixel(repaired, bg_cells):
            continue
        fill_color = next(iter(marker_colors))
        for r, c in bg_cells:
            if out[r, c] == BG:
                out[r, c] = fill_color
    return out


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _qualifies(cavity: np.ndarray, wall: np.ndarray, marker: tuple[int, int]) -> bool:
    patch = np.zeros(cavity.shape, dtype=int)
    patch[wall] = LINE
    bg_cells = set(map(tuple, np.argwhere(cavity)))
    bg_cells.remove(marker)
    return _bg_has_internal_pixel(patch, bg_cells)


def _interior_candidates(cavity: np.ndarray) -> list[tuple[int, int]]:
    pts = set(map(tuple, np.argwhere(cavity)))
    candidates = []
    for r, c in pts:
        if all((r + dr, c + dc) in pts for dr, dc in DIRS4):
            candidates.append((r, c))
    return candidates


def _large_cavity_mask() -> np.ndarray:
    h = random.randint(4, 6)
    w = random.randint(4, 6)
    return np.ones((h, w), dtype=bool)


def _large_gap_options(shape: tuple[int, int]) -> list[set[tuple[int, int]]]:
    h, w = shape
    options: list[set[tuple[int, int]]] = []
    max_h_gap = max(0, min(3, w - 4))
    for gap_len in range(1, max_h_gap + 1):
        for start in range(2, w - 2 - gap_len + 1):
            options.append({(0, c) for c in range(start, start + gap_len)})
            options.append({(h - 1, c) for c in range(start, start + gap_len)})
    max_v_gap = max(0, min(3, h - 4))
    for gap_len in range(1, max_v_gap + 1):
        for start in range(2, h - 2 - gap_len + 1):
            options.append({(r, 0) for r in range(start, start + gap_len)})
            options.append({(r, w - 1) for r in range(start, start + gap_len)})
    return options


def _open_gaps(
    wall: np.ndarray, gap_options: list[set[tuple[int, int]]], gap_count: int
) -> np.ndarray | None:
    candidates = gap_options[:]
    random.shuffle(candidates)

    chosen: list[set[tuple[int, int]]] = []
    used: set[tuple[int, int]] = set()
    for cand in candidates:
        halo = {(r + dr, c + dc) for r, c in cand for dr, dc in DIRS8 + [(0, 0)]}
        if halo & used:
            continue
        chosen.append(cand)
        used |= halo
        if len(chosen) == gap_count:
            break
    if len(chosen) != gap_count:
        return None

    out = wall.copy()
    for cand in chosen:
        for r, c in cand:
            out[r, c] = False
    return out


def _region_shape(kind: str) -> dict:
    if kind == "small":
        cavity = np.zeros((5, 5), dtype=bool)
        cavity[1, 2] = True
        cavity[1, 3] = True
        cavity[2, 1] = True
        cavity[2, 2] = True
        cavity[2, 3] = True
        cavity[3, 2] = True
        cavity[3, 3] = True
        wall = np.zeros((5, 5), dtype=bool)
        wall[0, :] = True
        wall[-1, :] = True
        wall[:, 0] = True
        wall[:, -1] = True
        wall[1, 1] = True
        wall[3, 1] = True
        rot = random.randrange(4)
        cavity = np.rot90(cavity, rot)
        wall = np.rot90(wall, rot)
        base_gap = np.zeros((5, 5), dtype=int)
        base_gap[2, 4] = 1
        gap_mask = np.rot90(base_gap, rot)
        gap = {tuple(map(int, rc)) for rc in np.argwhere(gap_mask == 1)}
        return {
            "kind": kind,
            "cavity": cavity,
            "wall": wall,
            "marker": (2, 2),
            "qualifies": False,
            "gap_options": [gap],
        }

    cavity_mask = _large_cavity_mask()
    h, w = cavity_mask.shape
    cavity = np.zeros((h + 2, w + 2), dtype=bool)
    cavity[1:-1, 1:-1] = True
    wall = np.zeros((h + 2, w + 2), dtype=bool)
    wall[0, :] = True
    wall[-1, :] = True
    wall[:, 0] = True
    wall[:, -1] = True
    marker = random.choice(_interior_candidates(cavity))
    return {
        "kind": kind,
        "cavity": cavity,
        "wall": wall,
        "marker": marker,
        "qualifies": _qualifies(cavity, wall, marker),
        "gap_options": _large_gap_options(wall.shape),
    }


def _can_place(occ: np.ndarray, patch_mask: np.ndarray, r0: int, c0: int, pad: int = 2) -> bool:
    h, w = patch_mask.shape
    if r0 < 0 or c0 < 0 or r0 + h > occ.shape[0] or c0 + w > occ.shape[1]:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occ.shape[0], r0 + h + pad)
    cc1 = min(occ.shape[1], c0 + w + pad)
    return not occ[rr0:rr1, cc0:cc1].any()


def _place_shapes(shapes: list[dict], height: int, width: int) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    occ = np.zeros((height, width), dtype=bool)
    colors = random.sample(MARKER_COLORS, len(shapes))

    for shape, color in zip(shapes, colors, strict=False):
        patch_mask = shape["cavity"] | shape["wall"]
        ph, pw = patch_mask.shape
        options = [(r, c) for r in range(height - ph + 1) for c in range(width - pw + 1)]
        random.shuffle(options)
        placed = False
        for r0, c0 in options:
            if not _can_place(occ, patch_mask, r0, c0):
                continue
            wall_cells = np.argwhere(shape["wall"])
            for rr, cc in wall_cells:
                grid[r0 + rr, c0 + cc] = LINE
            mr, mc = shape["marker"]
            grid[r0 + mr, c0 + mc] = color
            occ[r0 : r0 + ph, c0 : c0 + pw] |= patch_mask
            placed = True
            break
        if not placed:
            return None
    return grid


def _make_input_attempt() -> np.ndarray | None:
    height = random.randint(16, 24)
    width = random.randint(16, 24)
    n_shapes = random.choice([2, 2, 3])
    kinds = ["large", "small"]
    if n_shapes == 3:
        kinds.append(random.choice(["large", "small"]))

    shapes = [_region_shape(kind) for kind in kinds]
    if not any(shape["qualifies"] for shape in shapes):
        return None
    if not any(not shape["qualifies"] for shape in shapes):
        return None

    total_gaps = random.randint(n_shapes, min(4, 2 * n_shapes))
    gap_counts = [1] * n_shapes
    large_indices = [i for i, shape in enumerate(shapes) if shape["kind"] == "large"]
    for _ in range(total_gaps - n_shapes):
        gap_counts[random.choice(large_indices)] += 1

    opened_shapes = []
    for shape, gap_count in zip(shapes, gap_counts, strict=False):
        opened = _open_gaps(shape["wall"], shape["gap_options"], gap_count)
        if opened is None:
            return None
        opened_shapes.append({**shape, "wall": opened})

    grid = _place_shapes(opened_shapes, height, width)
    if grid is None:
        return None

    repaired = _fill_facing_gaps(grid)
    regions = _closed_cavities(repaired)
    if len(regions) < 2:
        return None

    large_found = False
    small_found = False
    for region in regions:
        marker_cells = [(r, c) for r, c in region if repaired[r, c] not in (BG, LINE)]
        if len(marker_cells) != 1:
            return None
        bg_cells = {(r, c) for r, c in region if repaired[r, c] == BG}
        if _bg_has_internal_pixel(repaired, bg_cells):
            large_found = True
        else:
            small_found = True
    if not (large_found and small_found):
        return None

    if _closed_cavities(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _make_input_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed")
