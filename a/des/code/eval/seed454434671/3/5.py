import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7
SEED_COLORS = (RED, BLUE, YELLOW, ORANGE)

SIDES = ("top", "bottom", "left", "right")
DIR_BY_SIDE = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 120


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    return border


def _branch_dir(trunk_dir: tuple[int, int], branch_index: int) -> tuple[int, int]:
    if trunk_dir[0] != 0:
        return (0, -1) if branch_index % 2 == 0 else (0, 1)
    return (-1, 0) if branch_index % 2 == 0 else (1, 0)


def _parse_fences(g: np.ndarray) -> list[dict]:
    fences: list[dict] = []
    for comp in _components_mask(g == GREEN):
        comp_set = set(comp)
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        border = _border_cells(r0, c0, r1, c1)
        if not comp_set.issubset(border):
            continue

        missing = list(border - comp_set)
        if len(missing) != 1:
            continue
        gap = missing[0]
        if g[gap] != BLACK:
            continue
        if gap in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
            continue

        if gap[0] == r0:
            side = "top"
        elif gap[0] == r1:
            side = "bottom"
        elif gap[1] == c0:
            side = "left"
        else:
            side = "right"

        if r1 - r0 <= 1 or c1 - c0 <= 1:
            continue
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        if np.any(interior == GREEN):
            continue

        seed_coords = np.argwhere((interior != BLACK) & (interior != GREEN))
        if not (1 <= len(seed_coords) <= 4):
            continue

        seeds: list[tuple[int, int, int]] = []
        seed_colors = set()
        for lr, lc in sorted((int(r), int(c)) for r, c in seed_coords):
            color = int(interior[lr, lc])
            seed_colors.add(color)
            seeds.append((r0 + 1 + lr, c0 + 1 + lc, color))
        if len(seed_colors) != len(seeds):
            continue

        fences.append(
            {
                "gap": gap,
                "trunk_dir": DIR_BY_SIDE[side],
                "seeds": seeds,
            }
        )
    return fences


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a green trunk through each rectangular fence gap and add alternating colored branches.

    The black gap acts as a connector cell. The counted trunk segment starts at the first cell
    outside the fence so the first branch does not overlap the fence border.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape

    for fence in _parse_fences(g):
        gap_r, gap_c = fence["gap"]
        dr, dc = fence["trunk_dir"]
        seeds = fence["seeds"]
        trunk_len = 2 * len(seeds)

        out[gap_r, gap_c] = GREEN

        trunk: list[tuple[int, int]] = []
        r, c = gap_r + dr, gap_c + dc
        for _ in range(trunk_len):
            if 0 <= r < h and 0 <= c < w:
                out[r, c] = GREEN
                trunk.append((r, c))
            r += dr
            c += dc

        for idx, (_sr, _sc, color) in enumerate(seeds):
            if 2 * idx >= len(trunk):
                break
            br, bc = trunk[2 * idx]
            sdr, sdc = _branch_dir((dr, dc), idx)
            for step in (1, 2):
                rr = br + sdr * step
                cc = bc + sdc * step
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = color

    return out


def _build_object_spec(side: str) -> dict | None:
    seed_count = random.randint(1, 4)
    interior_h = random.randint(2, 5)
    interior_w = random.randint(2, 5)
    if interior_h * interior_w < seed_count:
        return None

    outer_h = interior_h + 2
    outer_w = interior_w + 2
    trunk_dir = DIR_BY_SIDE[side]

    if side in ("top", "bottom"):
        gap_rel = (0 if side == "top" else outer_h - 1, random.randint(1, outer_w - 2))
    else:
        gap_rel = (random.randint(1, outer_h - 2), 0 if side == "left" else outer_w - 1)

    border = _border_cells(0, 0, outer_h - 1, outer_w - 1)
    border.remove(gap_rel)

    interior_cells = [(r, c) for r in range(1, outer_h - 1) for c in range(1, outer_w - 1)]
    seed_cells = sorted(random.sample(interior_cells, seed_count))
    seed_colors = random.sample(SEED_COLORS, seed_count)
    seeds = [(r, c, color) for (r, c), color in zip(seed_cells, seed_colors, strict=True)]

    reserve = {(r, c) for r in range(outer_h) for c in range(outer_w)}
    trunk: list[tuple[int, int]] = []
    r, c = gap_rel[0] + trunk_dir[0], gap_rel[1] + trunk_dir[1]
    for _ in range(2 * seed_count):
        trunk.append((r, c))
        reserve.add((r, c))
        r += trunk_dir[0]
        c += trunk_dir[1]

    for idx in range(seed_count):
        br, bc = trunk[2 * idx]
        bdr, bdc = _branch_dir(trunk_dir, idx)
        for step in (1, 2):
            reserve.add((br + bdr * step, bc + bdc * step))

    rel_rows = [r for r, _ in reserve]
    rel_cols = [c for _, c in reserve]
    return {
        "border": border,
        "gap": gap_rel,
        "seeds": seeds,
        "reserve": reserve,
        "min_r": min(rel_rows),
        "max_r": max(rel_rows),
        "min_c": min(rel_cols),
        "max_c": max(rel_cols),
    }


def _pad_cells(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    padded = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    padded.add((nr, nc))
    return padded


def _draw_object(grid: np.ndarray, spec: dict, r0: int, c0: int) -> None:
    for r, c in spec["border"]:
        grid[r0 + r, c0 + c] = GREEN
    for r, c, color in spec["seeds"]:
        grid[r0 + r, c0 + c] = color


def _gen_attempt() -> np.ndarray | None:
    object_count = random.randint(1, 4)
    min_dim = 12 + 4 * object_count
    h = random.randint(min_dim, min_dim + 8)
    w = random.randint(min_dim, min_dim + 8)
    grid = np.full((h, w), BLACK, dtype=int)

    blocked: set[tuple[int, int]] = set()
    sides = random.sample(SIDES, k=4)[:object_count]

    for side in sides:
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            spec = _build_object_spec(side)
            if spec is None:
                continue

            row_lo = -spec["min_r"]
            row_hi = h - 1 - spec["max_r"]
            col_lo = -spec["min_c"]
            col_hi = w - 1 - spec["max_c"]
            if row_lo > row_hi or col_lo > col_hi:
                continue

            r0 = random.randint(row_lo, row_hi)
            c0 = random.randint(col_lo, col_hi)
            reserve = {(r0 + r, c0 + c) for r, c in spec["reserve"]}
            if reserve & blocked:
                continue

            _draw_object(grid, spec, r0, c0)
            blocked |= _pad_cells(reserve, h, w)
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
