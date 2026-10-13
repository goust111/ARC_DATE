from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GRID_MIN = 18
GRID_MAX = 32
OBJECT_COUNT_CHOICES = (2, 2, 3, 3, 4)
SHAPE_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 180
OBJECT_RETRIES = 40
GEN_ATTEMPTS = 300


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _outline(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                out[r, c] = True
                continue
            for nr, nc in _neighbors4(r, c, h, w):
                if not mask[nr, nc]:
                    out[r, c] = True
                    break
    return out


def _gap_on_row(grey_mask: np.ndarray, row: int, cols: list[int]) -> list[tuple[int, int]]:
    if len(cols) < 3:
        return []
    lo = min(cols)
    hi = max(cols)
    missing = [c for c in range(lo, hi + 1) if not grey_mask[row, c]]
    if not missing:
        return []
    if missing[0] == lo or missing[-1] == hi:
        return []
    if missing != list(range(missing[0], missing[-1] + 1)):
        return []
    return [(row, c) for c in missing]


def _gap_on_col(grey_mask: np.ndarray, col: int, rows: list[int]) -> list[tuple[int, int]]:
    if len(rows) < 3:
        return []
    lo = min(rows)
    hi = max(rows)
    missing = [r for r in range(lo, hi + 1) if not grey_mask[r, col]]
    if not missing:
        return []
    if missing[0] == lo or missing[-1] == hi:
        return []
    if missing != list(range(missing[0], missing[-1] + 1)):
        return []
    return [(r, col) for r in missing]


def _find_gap(grey_mask: np.ndarray, coords: list[tuple[int, int]]) -> tuple[str, list[tuple[int, int]]]:
    r0, c0, r1, c1 = _bbox(coords)
    by_row: dict[int, list[int]] = {}
    by_col: dict[int, list[int]] = {}
    for r, c in coords:
        by_row.setdefault(r, []).append(c)
        by_col.setdefault(c, []).append(r)

    candidates: list[tuple[str, list[tuple[int, int]]]] = []
    top_gap = _gap_on_row(grey_mask, r0, by_row.get(r0, []))
    if top_gap:
        candidates.append(("up", top_gap))
    bottom_gap = _gap_on_row(grey_mask, r1, by_row.get(r1, []))
    if bottom_gap:
        candidates.append(("down", bottom_gap))
    left_gap = _gap_on_col(grey_mask, c0, by_col.get(c0, []))
    if left_gap:
        candidates.append(("left", left_gap))
    right_gap = _gap_on_col(grey_mask, c1, by_col.get(c1, []))
    if right_gap:
        candidates.append(("right", right_gap))

    assert len(candidates) == 1, "expected exactly one directional gap"
    return candidates[0]


def _interior_with_filled_gap(
    grid: np.ndarray, coords: list[tuple[int, int]], gap: list[tuple[int, int]]
) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    pr0 = max(0, r0 - 1)
    pc0 = max(0, c0 - 1)
    pr1 = min(grid.shape[0] - 1, r1 + 1)
    pc1 = min(grid.shape[1] - 1, c1 + 1)

    local = grid[pr0 : pr1 + 1, pc0 : pc1 + 1]
    walls = local == GREY
    for r, c in gap:
        walls[r - pr0, c - pc0] = True

    h, w = walls.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not walls[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not walls[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if walls[nr, nc] or outside[nr, nc]:
                continue
            outside[nr, nc] = True
            q.append((nr, nc))

    return ~walls & ~outside


def _paint_ray(out: np.ndarray, gap: list[tuple[int, int]], direction: str, color: int) -> None:
    h, w = out.shape
    dr, dc = DIRS[direction]
    for r, c in gap:
        rr = r
        cc = c
        while 0 <= rr < h and 0 <= cc < w:
            if out[rr, cc] == BLACK:
                out[rr, cc] = color
            rr += dr
            cc += dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer each outline's interior seed color, detect the side and width of its unique
    boundary gap, then spray that color outward from the gap to the grid edge.
    """
    g = np.asarray(grid, dtype=int)
    grey_mask = g == GREY
    components = _components(grey_mask)
    assert components, "expected at least one grey outline"

    out = g.copy()
    for comp in components:
        direction, gap = _find_gap(grey_mask, comp)
        interior = _interior_with_filled_gap(g, comp, gap)
        r0, c0, r1, c1 = _bbox(comp)
        pr0 = max(0, r0 - 1)
        pc0 = max(0, c0 - 1)
        pr1 = min(g.shape[0] - 1, r1 + 1)
        pc1 = min(g.shape[1] - 1, c1 + 1)
        local = g[pr0 : pr1 + 1, pc0 : pc1 + 1]
        colors = {
            int(v)
            for v in np.unique(local[interior])
            if int(v) not in (BLACK, GREY)
        }
        assert len(colors) == 1, "each region must contain exactly one seed color"
        color = next(iter(colors))
        _paint_ray(out, gap, direction, color)
    return out


def _build_filled_shape(h: int, w: int, cuts: tuple[int, int, int, int]) -> np.ndarray:
    tl, tr, bl, br = cuts
    rr, cc = np.indices((h, w))
    return (
        (rr + cc >= tl)
        & (rr + (w - 1 - cc) >= tr)
        & ((h - 1 - rr) + cc >= bl)
        & ((h - 1 - rr) + (w - 1 - cc) >= br)
    )


def _side_cells(mask: np.ndarray, direction: str) -> list[tuple[int, int]]:
    h, w = mask.shape
    if direction == "up":
        cols = np.flatnonzero(mask[0])
        return [(0, int(c)) for c in cols]
    if direction == "down":
        cols = np.flatnonzero(mask[h - 1])
        return [(h - 1, int(c)) for c in cols]
    if direction == "left":
        rows = np.flatnonzero(mask[:, 0])
        return [(int(r), 0) for r in rows]
    rows = np.flatnonzero(mask[:, w - 1])
    return [(int(r), w - 1) for r in rows]


def _sample_region(color: int, direction: str):
    for _ in range(SHAPE_ATTEMPTS):
        h = random.randint(7, 13)
        w = random.randint(7, 13)
        max_cut = min(3, h - 4, w - 4)
        cuts = tuple(random.randint(0, max_cut) for _ in range(4))
        filled = _build_filled_shape(h, w, cuts)
        if not filled[0].any() or not filled[h - 1].any() or not filled[:, 0].any() or not filled[:, w - 1].any():
            continue

        side = _side_cells(filled, direction)
        if len(side) < 4:
            continue
        ordered = [cell[1] if direction in ("up", "down") else cell[0] for cell in side]
        if ordered != list(range(ordered[0], ordered[-1] + 1)):
            continue

        closed_outline = _outline(filled)
        interior = filled & ~closed_outline
        interior_cells = [tuple(map(int, rc)) for rc in np.argwhere(interior)]
        if len(interior_cells) < 4:
            continue

        gap_max = min(3, len(side) - 2)
        gap_width = random.randint(1, gap_max)
        start = random.randint(1, len(side) - gap_width - 1)
        gap = side[start : start + gap_width]

        open_outline = closed_outline.copy()
        for r, c in gap:
            open_outline[r, c] = False
        if len(_components(open_outline)) != 1:
            continue

        seed_count = random.randint(2, min(5, len(interior_cells)))
        seeds = random.sample(interior_cells, seed_count)
        patch = np.full((h, w), BLACK, dtype=int)
        patch[open_outline] = GREY
        for r, c in seeds:
            patch[r, c] = color

        return {
            "patch": patch,
            "gap": gap,
            "direction": direction,
            "color": color,
            "nonblack": [tuple(map(int, rc)) for rc in np.argwhere(patch != BLACK)],
        }
    return None


def _expand8(coords: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _global_ray_cells(
    gap: list[tuple[int, int]], direction: str, top: int, left: int, h: int, w: int
) -> set[tuple[int, int]]:
    dr, dc = DIRS[direction]
    cells: set[tuple[int, int]] = set()
    for r, c in gap:
        rr = top + r
        cc = left + c
        while 0 <= rr < h and 0 <= cc < w:
            cells.add((rr, cc))
            rr += dr
            cc += dc
    return cells


def _placement_ranges(direction: str, h: int, w: int, H: int, W: int):
    edge_pad = 1
    ray_pad = 2
    r_lo = edge_pad
    r_hi = H - h - edge_pad
    c_lo = edge_pad
    c_hi = W - w - edge_pad
    if direction == "up":
        r_lo = ray_pad
    elif direction == "down":
        r_hi = H - h - ray_pad
    elif direction == "left":
        c_lo = ray_pad
    else:
        c_hi = W - w - ray_pad
    return r_lo, r_hi, c_lo, c_hi


def _try_place_region(
    region,
    reserved_output: set[tuple[int, int]],
    occupied_nonblack: set[tuple[int, int]],
    spaced_nonblack: set[tuple[int, int]],
):
    H, W = region["canvas_shape"]
    patch = region["patch"]
    h, w = patch.shape
    r_lo, r_hi, c_lo, c_hi = _placement_ranges(region["direction"], h, w, H, W)
    if r_lo > r_hi or c_lo > c_hi:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        top = random.randint(r_lo, r_hi)
        left = random.randint(c_lo, c_hi)
        nonblack = {(top + r, left + c) for r, c in region["nonblack"]}
        if nonblack & spaced_nonblack:
            continue
        if nonblack & reserved_output:
            continue

        ray = _global_ray_cells(region["gap"], region["direction"], top, left, H, W)
        if ray & reserved_output:
            continue
        if ray & occupied_nonblack:
            continue

        placed = {
            "patch": patch,
            "top": top,
            "left": left,
            "gap": [(top + r, left + c) for r, c in region["gap"]],
            "direction": region["direction"],
            "color": region["color"],
            "nonblack": nonblack,
            "ray": ray,
        }
        return placed
    return None


def _render_input(shape: tuple[int, int], placed_regions: list[dict]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for region in placed_regions:
        patch = region["patch"]
        h, w = patch.shape
        top = region["top"]
        left = region["left"]
        sub = grid[top : top + h, left : left + w]
        mask = patch != BLACK
        sub[mask] = patch[mask]
    return grid


def _render_output(inp: np.ndarray, placed_regions: list[dict]) -> np.ndarray:
    out = inp.copy()
    for region in placed_regions:
        _paint_ray(out, region["gap"], region["direction"], region["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.choice(OBJECT_COUNT_CHOICES)
    side_min = GRID_MIN + 3 * (n_objects - 2)
    H = random.randint(side_min, GRID_MAX)
    W = random.randint(side_min, GRID_MAX)

    directions = random.sample(list(DIRS), n_objects)
    colors = random.sample(PALETTE, n_objects)

    placed_regions: list[dict] = []
    reserved_output: set[tuple[int, int]] = set()
    occupied_nonblack: set[tuple[int, int]] = set()
    spaced_nonblack: set[tuple[int, int]] = set()

    for color, direction in zip(colors, directions):
        placed = None
        for _ in range(OBJECT_RETRIES):
            region = _sample_region(color, direction)
            if region is None:
                continue
            region["canvas_shape"] = (H, W)
            placed = _try_place_region(
                region,
                reserved_output,
                occupied_nonblack,
                spaced_nonblack,
            )
            if placed is not None:
                break
        if placed is None:
            return None
        placed_regions.append(placed)
        reserved_output |= placed["nonblack"] | placed["ray"]
        occupied_nonblack |= placed["nonblack"]
        spaced_nonblack |= _expand8(placed["nonblack"], H, W)

    grid = _render_input((H, W), placed_regions)
    expected = _render_output(grid, placed_regions)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
