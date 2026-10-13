from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
YELLOW = 4
MAGENTA = 6
ORANGE = 7
AZURE = 8

COPY_COLOR_BY_HOLES = {0: ORANGE, 1: MAGENTA, 2: AZURE}
BODY_COLORS = (1, 2, 3, 5, 9)
DIR_TO_VEC = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ARROW_TEMPLATES = {
    "up": {(0, 1), (1, 0), (1, 2)},
    "down": {(0, 0), (0, 2), (1, 1)},
    "left": {(0, 1), (1, 0), (2, 1)},
    "right": {(0, 0), (1, 1), (2, 0)},
}
SHAPE_RETRIES = 120
GEN_RETRIES = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _count_holes(mask: np.ndarray) -> int:
    bg = ~mask
    h, w = bg.shape
    holes = 0
    for comp in _components(bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in comp):
            continue
        holes += 1
    return holes


def _norm_points(points: list[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0 = min(r for r, _ in points)
    c0 = min(c for _, c in points)
    return frozenset((r - r0, c - c0) for r, c in points)


def _arrow_direction(points: list[tuple[int, int]]) -> tuple[int, int]:
    norm = _norm_points(points)
    for name, template in ARROW_TEMPLATES.items():
        if norm == frozenset(template):
            return DIR_TO_VEC[name]
    raise AssertionError("expected one internal 3-pixel yellow arrow per object")


def _component_info(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict:
    coords = np.array(comp, dtype=int)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = crop != BLACK
    arrow = [tuple(map(int, cell)) for cell in np.argwhere(crop == YELLOW)]
    assert len(arrow) == 3, "each object must contain exactly three yellow arrow pixels"
    direction = _arrow_direction(arrow)
    holes = _count_holes(mask)
    assert holes in COPY_COLOR_BY_HOLES, "hole count must be 0, 1, or 2"
    return {
        "coords": coords,
        "direction": direction,
        "holes": holes,
        "mask": mask,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each colored object in its arrow direction, with distance and copy color set by hole count.

    Input structure:
    - The background is black.
    - The grid contains several separated 4-connected foreground objects made from one body color plus an internal yellow arrow marker.
    - Each object contains exactly one 3-pixel yellow arrow that points up, down, left, or right.
    - Each object may contain 0, 1, or 2 enclosed 1-pixel black holes.

    Rule:
    - For each foreground object, detect the direction of its yellow arrow.
    - Count the number of enclosed black holes N inside that object.
    - Translate the whole object by N+1 cells in the arrow direction to create a copy.
    - Recolor the translated copy by hole count: 0 -> orange, 1 -> magenta, 2 -> azure.
    - The copy does not keep a yellow arrow; every translated foreground cell uses the copy color.
    - Keep the original input objects unchanged and clip translated cells that fall outside the grid.

    Key insight:
    - The arrow and the holes encode independent parameters: the arrow gives the projection direction, while hole count simultaneously controls both the translation distance and the copy color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    copies = np.zeros_like(g)
    for comp in _components(g != BLACK):
        info = _component_info(g, comp)
        dr, dc = info["direction"]
        shift = info["holes"] + 1
        color = COPY_COLOR_BY_HOLES[info["holes"]]
        for r, c in info["coords"]:
            rr = int(r) + dr * shift
            cc = int(c) + dc * shift
            if 0 <= rr < h and 0 <= cc < w:
                copies[rr, cc] = color
    out = copies
    out[g != BLACK] = g[g != BLACK]
    return out


def _is_connected(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _arrow_bbox(direction: str) -> tuple[int, int]:
    if direction in ("up", "down"):
        return (2, 3)
    return (3, 2)


def _pick_holes(
    rng: random.Random,
    h: int,
    w: int,
    hole_count: int,
    arrow_cells: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    if hole_count == 0:
        return []
    candidates = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if (r, c) in arrow_cells:
                continue
            if any(abs(r - ar) + abs(c - ac) <= 1 for ar, ac in arrow_cells):
                continue
            candidates.append((r, c))
    rng.shuffle(candidates)
    holes: list[tuple[int, int]] = []
    for cell in candidates:
        if any(max(abs(cell[0] - hr), abs(cell[1] - hc)) <= 1 for hr, hc in holes):
            continue
        holes.append(cell)
        if len(holes) == hole_count:
            return holes
    return None


def _generate_shape(
    rng: random.Random,
    direction: str,
    hole_count: int,
) -> tuple[np.ndarray, set[tuple[int, int]]]:
    for _ in range(SHAPE_RETRIES):
        min_side = 6 if hole_count == 2 else 5
        h = rng.randint(min_side, 7)
        w = rng.randint(min_side, 7)
        ah, aw = _arrow_bbox(direction)
        if h < ah + 2 or w < aw + 2:
            continue
        ar = rng.randint(1, h - ah - 1)
        ac = rng.randint(1, w - aw - 1)
        arrow_cells = {(ar + rr, ac + cc) for rr, cc in ARROW_TEMPLATES[direction]}
        holes = _pick_holes(rng, h, w, hole_count, arrow_cells)
        if holes is None:
            continue

        mask = np.ones((h, w), dtype=bool)
        for hr, hc in holes:
            mask[hr, hc] = False

        target_cuts = rng.choice([0, 0, 1, 1, 2])
        boundary = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if r in (0, h - 1) or c in (0, w - 1)
        ]
        rng.shuffle(boundary)
        cuts = 0
        for r, c in boundary:
            if cuts == target_cuts:
                break
            if not mask[r, c]:
                continue
            trial = mask.copy()
            trial[r, c] = False
            if not _is_connected(trial):
                continue
            if _count_holes(trial) != hole_count:
                continue
            mask = trial
            cuts += 1

        if not _is_connected(mask):
            continue
        if _count_holes(mask) != hole_count:
            continue
        if not all(mask[r, c] for r, c in arrow_cells):
            continue
        return mask, arrow_cells
    raise RuntimeError("failed to generate a valid object shape")


def _render_object(mask: np.ndarray, arrow_cells: set[tuple[int, int]], color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = color
    for r, c in arrow_cells:
        patch[r, c] = YELLOW
    return patch


def _cells_from_mask(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    cells = set()
    for r, c in np.argwhere(mask):
        cells.add((top + int(r), left + int(c)))
    return cells


def _translated_cells(
    mask: np.ndarray,
    top: int,
    left: int,
    shift: tuple[int, int],
    grid_shape: tuple[int, int],
) -> set[tuple[int, int]]:
    h, w = grid_shape
    dr, dc = shift
    cells = set()
    for r, c in np.argwhere(mask):
        rr = top + int(r) + dr
        cc = left + int(c) + dc
        if 0 <= rr < h and 0 <= cc < w:
            cells.add((rr, cc))
    return cells


def _touches_any(cells: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        if (r, c) in occupied:
            return True
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied:
                return True
    return False


def _place_objects(
    rng: random.Random,
    specs: list[dict],
    h: int,
    w: int,
) -> np.ndarray | None:
    grid = np.zeros((h, w), dtype=int)
    occupied_orig: set[tuple[int, int]] = set()
    occupied_proj: set[tuple[int, int]] = set()
    mid_r = h // 2
    mid_c = w // 2
    gap = 2
    quadrants = [
        (0, mid_r - gap, 0, mid_c - gap),
        (0, mid_r - gap, mid_c + gap, w),
        (mid_r + gap, h, 0, mid_c - gap),
        (mid_r + gap, h, mid_c + gap, w),
    ]
    rng.shuffle(quadrants)

    for spec, (r_lo, r_hi, c_lo, c_hi) in zip(specs, quadrants):
        patch = spec["patch"]
        mask = spec["mask"]
        ph, pw = patch.shape
        if r_hi - r_lo < ph or c_hi - c_lo < pw:
            return None
        candidates = [
            (r, c)
            for r in range(r_lo, r_hi - ph + 1)
            for c in range(c_lo, c_hi - pw + 1)
        ]
        rng.shuffle(candidates)
        placed = False
        dr, dc = spec["direction"]
        shift = (dr * (spec["holes"] + 1), dc * (spec["holes"] + 1))
        for top, left in candidates:
            orig_cells = _cells_from_mask(mask, top, left)
            proj_cells = _translated_cells(mask, top, left, shift, (h, w))
            if _touches_any(orig_cells, occupied_orig):
                continue
            if orig_cells & occupied_proj:
                continue
            if _touches_any(proj_cells, occupied_orig):
                continue
            if proj_cells & occupied_proj:
                continue
            patch_rows = slice(top, top + ph)
            patch_cols = slice(left, left + pw)
            target = grid[patch_rows, patch_cols]
            if np.any((patch != BLACK) & (target != BLACK)):
                continue
            target[patch != BLACK] = patch[patch != BLACK]
            occupied_orig |= orig_cells
            occupied_proj |= proj_cells
            placed = True
            break
        if not placed:
            return None
    return grid


def _valid_generated_grid(grid: np.ndarray) -> bool:
    try:
        comps = _components(grid != BLACK)
        if len(comps) != 4:
            return False
        infos = [_component_info(grid, comp) for comp in comps]
    except AssertionError:
        return False
    hole_multiset = sorted(info["holes"] for info in infos)
    dir_set = {tuple(info["direction"]) for info in infos}
    return hole_multiset.count(0) >= 1 and hole_multiset.count(1) >= 1 and hole_multiset.count(2) >= 1 and dir_set == set(DIR_TO_VEC.values())


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(22, 28)
    w = rng.randint(22, 28)
    directions = rng.sample(list(DIR_TO_VEC), 4)
    hole_counts = [0, 1, 2, rng.randrange(3)]
    rng.shuffle(hole_counts)
    colors = rng.sample(BODY_COLORS, 4)

    specs: list[dict] = []
    for direction, holes, color in zip(directions, hole_counts, colors):
        mask, arrow_cells = _generate_shape(rng, direction, holes)
        patch = _render_object(mask, arrow_cells, color)
        specs.append(
            {
                "patch": patch,
                "mask": mask,
                "direction": DIR_TO_VEC[direction],
                "holes": holes,
            }
        )

    grid = _place_objects(rng, specs, h, w)
    if grid is None:
        return None
    if not _valid_generated_grid(grid):
        return None
    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
