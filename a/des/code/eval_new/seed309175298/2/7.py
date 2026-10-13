import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
BROWN = 9

PRIMARY_COLORS = (BLUE, RED, YELLOW)
INPUT_COLORS = {BLACK, BLUE, RED, GREEN, YELLOW, GRAY}
PRIMARY_BITS = {BLUE: 1, RED: 2, YELLOW: 4}
BIT_TO_COLOR = {
    1: BLUE,
    2: RED,
    3: MAGENTA,
    4: YELLOW,
    5: AZURE,
    6: ORANGE,
    7: BROWN,
}
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _turns(dr: int, dc: int) -> tuple[tuple[int, int], tuple[int, int]]:
    return (-dc, dr), (dc, -dr)


def _detect_towers(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    h, w = grid.shape
    towers: list[dict] = []
    tower_mask = np.zeros((h, w), dtype=bool)
    for color in PRIMARY_COLORS:
        for coords in _components(grid == color):
            if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in coords):
                continue
            rows = sorted(r for r, _ in coords)
            cols = sorted(c for _, c in coords)
            r0, r1 = rows[0], rows[-1]
            c0, c1 = cols[0], cols[-1]
            assert len(coords) == 2, "edge towers must contain exactly two cells"
            if r0 == r1 and c1 == c0 + 1 and r0 in (0, h - 1):
                direction = (1, 0) if r0 == 0 else (-1, 0)
            elif c0 == c1 and r1 == r0 + 1 and c0 in (0, w - 1):
                direction = (0, 1) if c0 == 0 else (0, -1)
            else:
                raise AssertionError("edge primary components must be 1x2 or 2x1 towers")
            front = tuple(sorted((int(r), int(c)) for r, c in coords))
            for r, c in front:
                tower_mask[r, c] = True
            towers.append({"color": int(color), "front": front, "dir": direction})

    primary_mask = np.isin(grid, PRIMARY_COLORS)
    assert np.array_equal(primary_mask, tower_mask), "primary colors must appear only in towers"
    assert towers, "expected at least one signal tower"
    return towers, tower_mask


def _solve(grid: np.ndarray, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h <= 30 and w <= 30, "ARC grids must fit within 30x30"
    assert set(map(int, np.unique(g))) <= INPUT_COLORS, "unexpected input colors"

    towers, tower_mask = _detect_towers(g)
    wall_mask = g == GRAY
    prism_mask = g == GREEN
    ray_bits = np.zeros((h, w), dtype=np.uint8)
    reached_prisms: set[tuple[int, int]] = set()

    queue: deque[tuple[tuple[tuple[int, int], ...], int, int, int]] = deque()
    for tower in towers:
        dr, dc = tower["dir"]
        queue.append((tower["front"], dr, dc, tower["color"]))

    seen_states: set[tuple[tuple[tuple[int, int], ...], int, int, int]] = set()
    while queue:
        front, dr, dc, color = queue.popleft()
        state = (front, dr, dc, color)
        if state in seen_states:
            continue
        seen_states.add(state)

        bit = PRIMARY_BITS[color]
        current = front
        while True:
            nxt = tuple(sorted((r + dr, c + dc) for r, c in current))
            if any(not (0 <= nr < h and 0 <= nc < w) for nr, nc in nxt):
                break
            if any(wall_mask[nr, nc] for nr, nc in nxt):
                break

            for nr, nc in nxt:
                if not tower_mask[nr, nc]:
                    ray_bits[nr, nc] |= bit
                if prism_mask[nr, nc]:
                    reached_prisms.add((nr, nc))
                    for side_dr, side_dc in _turns(dr, dc):
                        queue.append((((nr, nc),), side_dr, side_dc, color))
            current = nxt

    out = g.copy()
    for r in range(h):
        for c in range(w):
            if tower_mask[r, c] or wall_mask[r, c]:
                continue
            mask = int(ray_bits[r, c])
            if mask:
                out[r, c] = BIT_TO_COLOR[mask]

    if return_meta:
        return out, {
            "towers": towers,
            "tower_mask": tower_mask,
            "ray_bits": ray_bits,
            "reached_prisms": reached_prisms,
            "prism_count": int(prism_mask.sum()),
        }
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the colored tower rays, split them at green prisms, and paint the final overlap colors.

    Input structure:
    - The background is black.
    - Red, blue, and yellow signal towers are 1x2 or 2x1 rectangles attached to the grid edge.
    - Green single pixels are prisms, and gray pixels are walls inside the grid.

    Rule:
    - Each tower emits a two-cell-wide ray straight inward, perpendicular to its edge, and the ray stops before a wall or outside the grid.
    - When a ray reaches a green prism, it keeps going through that prism and also emits two one-cell-wide side rays in the perpendicular directions.
    - Different ray colors overlay on the same cell as magenta (red+blue), orange (red+yellow), or azure (blue+yellow), while tower cells keep their original colors.

    Key insight:
    - The puzzle is a ray-tracing task on an implicit network: edge rectangles determine the launch directions, prisms recursively branch the paths, and the output depends on tracking every colored coverage overlap.
    """
    return _solve(grid)


def _cells_empty(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    return all(int(grid[r, c]) == BLACK for r, c in cells)


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> bool:
    if not _cells_empty(grid, cells):
        return False
    for r, c in cells:
        grid[r, c] = color
    return True


def _pick_vertical_prism_row(h: int, cross_r: int, blocked_rows: set[int] | None = None) -> int | None:
    blocked_rows = blocked_rows or set()
    candidates = [
        r
        for r in range(2, h - 2)
        if r not in blocked_rows and r not in {cross_r, cross_r + 1}
    ]
    return random.choice(candidates) if candidates else None


def _pick_horizontal_prism_col(w: int, cross_c: int, blocked_cols: set[int] | None = None) -> int | None:
    blocked_cols = blocked_cols or set()
    candidates = [
        c
        for c in range(2, w - 2)
        if c not in blocked_cols and c not in {cross_c, cross_c + 1}
    ]
    return random.choice(candidates) if candidates else None


def _pick_side_col(w: int, start_c: int, blocked_cols: set[int] | None = None):
    blocked_cols = blocked_cols or set()
    choices: list[tuple[int, list[int]]] = []
    left = [c for c in range(2, start_c - 1) if c not in blocked_cols]
    right = [c for c in range(start_c + 2, w - 2) if c not in blocked_cols]
    if left:
        choices.append((-1, left))
    if right:
        choices.append((1, right))
    if not choices:
        return None
    side, options = random.choice(choices)
    return side, random.choice(options)


def _pick_side_row(h: int, start_r: int, blocked_rows: set[int] | None = None):
    blocked_rows = blocked_rows or set()
    choices: list[tuple[int, list[int]]] = []
    up = [r for r in range(2, start_r - 1) if r not in blocked_rows]
    down = [r for r in range(start_r + 2, h - 2) if r not in blocked_rows]
    if up:
        choices.append((-1, up))
    if down:
        choices.append((1, down))
    if not choices:
        return None
    side, options = random.choice(choices)
    return side, random.choice(options)


def _place_vertical_wall(
    grid: np.ndarray, cross_c: int, cross_r: int, side: str, prism_rows: list[int]
) -> bool:
    h, _ = grid.shape
    if side == "top":
        lo = max([cross_r + 1] + prism_rows) + 2
        candidates = list(range(lo, h - 1))
    else:
        hi = min([cross_r] + prism_rows) - 2
        candidates = list(range(1, hi + 1))
    random.shuffle(candidates)
    for row in candidates:
        cells = [(row, cross_c), (row, cross_c + 1)]
        if _paint_cells(grid, cells, GRAY):
            return True
    return False


def _place_horizontal_wall(
    grid: np.ndarray, cross_r: int, cross_c: int, side: str, prism_cols: list[int]
) -> bool:
    _, w = grid.shape
    if side == "left":
        lo = max([cross_c + 1] + prism_cols) + 2
        candidates = list(range(lo, w - 1))
    else:
        hi = min([cross_c] + prism_cols) - 2
        candidates = list(range(1, hi + 1))
    random.shuffle(candidates)
    for col in candidates:
        cells = [(cross_r, col), (cross_r + 1, col)]
        if _paint_cells(grid, cells, GRAY):
            return True
    return False


def _place_row_wall(
    grid: np.ndarray,
    row: int,
    anchor_c: int,
    side: int,
    blocked_cols: set[int] | None = None,
) -> bool:
    blocked_cols = blocked_cols or set()
    _, w = grid.shape
    if side < 0:
        candidates = [c for c in range(1, anchor_c - 1) if c not in blocked_cols]
    else:
        candidates = [c for c in range(anchor_c + 2, w - 1) if c not in blocked_cols]
    random.shuffle(candidates)
    for col in candidates:
        if _paint_cells(grid, [(row, col)], GRAY):
            return True
    return False


def _place_col_wall(
    grid: np.ndarray,
    col: int,
    anchor_r: int,
    side: int,
    blocked_rows: set[int] | None = None,
) -> bool:
    blocked_rows = blocked_rows or set()
    h, _ = grid.shape
    if side < 0:
        candidates = [r for r in range(1, anchor_r - 1) if r not in blocked_rows]
    else:
        candidates = [r for r in range(anchor_r + 2, h - 1) if r not in blocked_rows]
    random.shuffle(candidates)
    for row in candidates:
        if _paint_cells(grid, [(row, col)], GRAY):
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 26)
    w = random.randint(14, 26)
    cross_r = random.randint(4, h - 6)
    cross_c = random.randint(4, w - 6)
    v_side = random.choice(["top", "bottom"])
    h_side = random.choice(["left", "right"])
    v_color, h_color = random.sample(PRIMARY_COLORS, 2)
    mode = random.choice(["single", "double", "cascade"])

    grid = np.full((h, w), BLACK, dtype=int)
    if not _paint_cells(grid, [(0 if v_side == "top" else h - 1, cross_c), (0 if v_side == "top" else h - 1, cross_c + 1)], v_color):
        return None
    if not _paint_cells(grid, [(cross_r, 0 if h_side == "left" else w - 1), (cross_r + 1, 0 if h_side == "left" else w - 1)], h_color):
        return None

    prisms: list[tuple[int, int]] = []
    v_prism: tuple[int, int] | None = None
    h_prism: tuple[int, int] | None = None
    v_cascade: tuple[int, int] | None = None
    h_cascade: tuple[int, int] | None = None
    v_cascade_side: int | None = None
    h_cascade_side: int | None = None

    if mode == "single":
        if random.choice(["v", "h"]) == "v":
            row = _pick_vertical_prism_row(h, cross_r)
            if row is None:
                return None
            v_prism = (row, random.choice([cross_c, cross_c + 1]))
            grid[v_prism] = GREEN
            prisms.append(v_prism)
        else:
            col = _pick_horizontal_prism_col(w, cross_c)
            if col is None:
                return None
            h_prism = (random.choice([cross_r, cross_r + 1]), col)
            grid[h_prism] = GREEN
            prisms.append(h_prism)
    elif mode == "double":
        row = _pick_vertical_prism_row(h, cross_r)
        col = _pick_horizontal_prism_col(w, cross_c)
        if row is None or col is None:
            return None
        v_prism = (row, random.choice([cross_c, cross_c + 1]))
        h_prism = (random.choice([cross_r, cross_r + 1]), col)
        if v_prism == h_prism:
            return None
        grid[v_prism] = GREEN
        grid[h_prism] = GREEN
        prisms.extend([v_prism, h_prism])
    else:
        if random.choice(["v", "h"]) == "v":
            row = _pick_vertical_prism_row(h, cross_r)
            if row is None:
                return None
            v_prism = (row, random.choice([cross_c, cross_c + 1]))
            grid[v_prism] = GREEN
            prisms.append(v_prism)
            side_pick = _pick_side_col(w, v_prism[1], blocked_cols={cross_c, cross_c + 1})
            if side_pick is None:
                return None
            v_cascade_side, cascade_col = side_pick
            v_cascade = (v_prism[0], cascade_col)
            if int(grid[v_cascade]) != BLACK:
                return None
            grid[v_cascade] = GREEN
            prisms.append(v_cascade)
            if random.random() < 0.5:
                col = _pick_horizontal_prism_col(w, cross_c, blocked_cols={cascade_col})
                if col is None:
                    return None
                h_prism = (random.choice([cross_r, cross_r + 1]), col)
                if int(grid[h_prism]) != BLACK:
                    return None
                grid[h_prism] = GREEN
                prisms.append(h_prism)
        else:
            col = _pick_horizontal_prism_col(w, cross_c)
            if col is None:
                return None
            h_prism = (random.choice([cross_r, cross_r + 1]), col)
            grid[h_prism] = GREEN
            prisms.append(h_prism)
            side_pick = _pick_side_row(h, h_prism[0], blocked_rows={cross_r, cross_r + 1})
            if side_pick is None:
                return None
            h_cascade_side, cascade_row = side_pick
            h_cascade = (cascade_row, h_prism[1])
            if int(grid[h_cascade]) != BLACK:
                return None
            grid[h_cascade] = GREEN
            prisms.append(h_cascade)
            if random.random() < 0.5:
                row = _pick_vertical_prism_row(h, cross_r, blocked_rows={cascade_row})
                if row is None:
                    return None
                v_prism = (row, random.choice([cross_c, cross_c + 1]))
                if int(grid[v_prism]) != BLACK:
                    return None
                grid[v_prism] = GREEN
                prisms.append(v_prism)

    wall_style = random.choice(["none", "one", "several"])
    wall_kinds = ["v_main", "h_main"]
    if v_prism is not None:
        wall_kinds.append("v_branch")
    if h_prism is not None:
        wall_kinds.append("h_branch")

    if wall_style != "none":
        random.shuffle(wall_kinds)
        target_walls = 1 if wall_style == "one" else min(len(wall_kinds), random.randint(2, 3))
        placed_walls = 0
        for kind in wall_kinds:
            if kind == "v_main":
                ok = _place_vertical_wall(
                    grid,
                    cross_c=cross_c,
                    cross_r=cross_r,
                    side=v_side,
                    prism_rows=[p[0] for p in prisms if p[1] in {cross_c, cross_c + 1}],
                )
            elif kind == "h_main":
                ok = _place_horizontal_wall(
                    grid,
                    cross_r=cross_r,
                    cross_c=cross_c,
                    side=h_side,
                    prism_cols=[p[1] for p in prisms if p[0] in {cross_r, cross_r + 1}],
                )
            elif kind == "v_branch":
                anchor = v_cascade[1] if v_cascade is not None else v_prism[1]
                side = v_cascade_side if v_cascade is not None else random.choice([-1, 1])
                ok = _place_row_wall(
                    grid,
                    row=v_prism[0],
                    anchor_c=anchor,
                    side=side,
                    blocked_cols={cross_c, cross_c + 1},
                )
            else:
                anchor = h_cascade[0] if h_cascade is not None else h_prism[0]
                side = h_cascade_side if h_cascade is not None else random.choice([-1, 1])
                ok = _place_col_wall(
                    grid,
                    col=h_prism[1],
                    anchor_r=anchor,
                    side=side,
                    blocked_rows={cross_r, cross_r + 1},
                )
            placed_walls += int(ok)
            if placed_walls >= target_walls:
                break

    try:
        out, meta = _solve(grid, return_meta=True)
    except AssertionError:
        return None

    if len(prisms) != meta["prism_count"]:
        return None
    if len(meta["reached_prisms"]) != len(prisms):
        return None
    if np.any(meta["ray_bits"] == 7):
        return None

    painted = int(np.count_nonzero(meta["ray_bits"]))
    mixed = int(np.isin(out, (MAGENTA, ORANGE, AZURE)).sum())
    walls = int((grid == GRAY).sum())
    if painted < 12 or painted > (h * w * 3) // 4:
        return None
    if mixed < 4:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-prism ARC input")
