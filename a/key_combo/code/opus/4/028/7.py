from __future__ import annotations

import random
from collections import Counter, deque

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
MAROON = 9

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OPEN_DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GEN_ATTEMPTS = 400
DIVIDER_ROW = 3
LOWER_START = DIVIDER_ROW + 1
WALL_SIZE_CHOICES = [(5, 5), (5, 6), (5, 7), (6, 6), (6, 7), (7, 6), (7, 7)]
CONTROL_COLORS = [BLUE, YELLOW, ORANGE, TEAL]
SEED_COLORS = [RED, GREEN, YELLOW, ORANGE, TEAL, BLUE]


def get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid).ravel()))
    return int(counts.most_common(1)[0][0])


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    neigh = DIR4 if connectivity == 4 else [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in neigh:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_divider_row(grid: np.ndarray) -> tuple[int, int]:
    for r in range(grid.shape[0]):
        row = grid[r]
        if np.all(row == row[0]) and int(row[0]) != BLACK:
            return int(r), int(row[0])
    raise AssertionError("expected one solid divider row")


def _opening_direction(local_wall: np.ndarray) -> str:
    counts = {
        "up": int(local_wall[0, :].sum()),
        "down": int(local_wall[-1, :].sum()),
        "left": int(local_wall[:, 0].sum()),
        "right": int(local_wall[:, -1].sum()),
    }
    return min(counts, key=lambda side: (counts[side], side))


def _scanline_coords(shape: tuple[int, int], direction: str) -> list[list[tuple[int, int]]]:
    h, w = shape
    if direction == "up":
        return [[(r, c) for r in range(h - 1, -1, -1)] for c in range(w)]
    if direction == "down":
        return [[(r, c) for r in range(h)] for c in range(w)]
    if direction == "left":
        return [[(r, c) for c in range(w - 1, -1, -1)] for r in range(h)]
    if direction == "right":
        return [[(r, c) for c in range(w)] for r in range(h)]
    raise ValueError(direction)


def _shadow_mask(fill_mask: np.ndarray, obstacle_mask: np.ndarray, direction: str) -> np.ndarray:
    keep = fill_mask.copy()
    for line in _scanline_coords(fill_mask.shape, direction):
        started = False
        blocked = False
        for r, c in line:
            if keep[r, c]:
                started = True
            if started and obstacle_mask[r, c]:
                blocked = True
                continue
            if blocked and keep[r, c]:
                keep[r, c] = False
    return keep


def _flood_fill(
    grid: np.ndarray,
    seed: tuple[int, int],
    seed_color: int,
    wall_color: int,
    active_cols: set[int],
    use_active: bool,
) -> np.ndarray:
    h, w = grid.shape
    passable = ((grid == BLACK) | (grid == seed_color)) & (grid != wall_color) & (grid != MAROON)
    if use_active:
        col_mask = np.zeros(w, dtype=bool)
        for c in active_cols:
            col_mask[int(c)] = True
        passable &= col_mask[None, :]
    sr, sc = seed
    passable[sr, sc] = True

    seen = np.zeros((h, w), dtype=bool)
    if not passable[sr, sc]:
        return seen
    q = deque([(sr, sc)])
    seen[sr, sc] = True
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen & (grid == BLACK)


def _solve(
    grid: np.ndarray,
    use_active: bool = True,
    use_shadow: bool = True,
    return_details: bool = False,
    strict_conflicts: bool = True,
):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    divider_row, wall_color = _find_divider_row(g)
    active_cols = {int(c) for c in np.where(g[0] != BLACK)[0]}

    out = g.copy()
    if h > 2:
        out[2, :] = BLACK
        for c in active_cols:
            out[2, c] = PINK

    lower_wall_mask = np.zeros_like(g, dtype=bool)
    lower_wall_mask[divider_row + 1 :, :] = g[divider_row + 1 :, :] == wall_color
    wall_components = components_mask(lower_wall_mask, connectivity=4)
    details = []

    for coords in wall_components:
        r0, c0, r1, c1 = bbox_coords(coords)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        local_wall = sub == wall_color
        direction = _opening_direction(local_wall)

        seed_locs = np.argwhere((sub != BLACK) & (sub != wall_color) & (sub != MAROON))
        assert seed_locs.shape[0] == 1, "each wall must contain exactly one seed"
        sr, sc = map(int, seed_locs[0])
        seed = (r0 + sr, c0 + sc)
        seed_color = int(sub[sr, sc])

        raw_fill = _flood_fill(g, seed, seed_color, wall_color, active_cols, use_active=use_active)
        final_fill = raw_fill
        if use_shadow:
            final_fill = _shadow_mask(raw_fill, g == MAROON, direction)

        overlap = final_fill & (out != g)
        if strict_conflicts:
            assert not np.any(overlap), "wall fills must not conflict"
        out[final_fill] = seed_color

        details.append(
            {
                "bbox": (r0, c0, r1, c1),
                "direction": direction,
                "seed": seed,
                "seed_color": seed_color,
                "raw_fill": raw_fill,
                "final_fill": final_fill,
            }
        )

    if return_details:
        return out, details
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the top compressed mask row and use the same active columns to constrain lower flood fills.

    Input structure:
    - Black background with a control strip above a full gray divider row.
    - Row 0 contains scattered foreground pixels whose columns define the active mask; rows 1-2 contain a single matching marker at column 0.
    - Below the divider are several gray U-shaped walls, each containing one colored seed, plus isolated maroon obstacle pixels.

    Rule:
    - Read the active columns from the non-black pixels in row 0.
    - Replace row 2 with a pink mask row: pink in active columns, black elsewhere, while leaving row 0, row 1, walls, divider, and obstacles unchanged.
    - For each lower U-shaped wall, flood-fill from its seed through black cells and out of the wall opening, but only inside active columns; inactive columns behave like invisible walls.
    - Along each wall's opening direction, maroon obstacles cast a shadow: any filled cells behind the first obstacle on that scanline are removed.

    Key insight:
    The top-row column mask is reused twice: once as a visible compressed pink row and once as a global spatial gate that restricts every lower flood fill.
    """
    return _solve(grid, use_active=True, use_shadow=True)


def _draw_wall(grid: np.ndarray, spec: dict) -> None:
    r0, c0, r1, c1 = spec["bbox"]
    direction = spec["direction"]
    if direction == "up":
        grid[r0:r1 + 1, c0] = GREY
        grid[r0:r1 + 1, c1] = GREY
        grid[r1, c0:c1 + 1] = GREY
    elif direction == "down":
        grid[r0:r1 + 1, c0] = GREY
        grid[r0:r1 + 1, c1] = GREY
        grid[r0, c0:c1 + 1] = GREY
    elif direction == "left":
        grid[r0, c0:c1 + 1] = GREY
        grid[r1, c0:c1 + 1] = GREY
        grid[r0:r1 + 1, c1] = GREY
    else:
        grid[r0, c0:c1 + 1] = GREY
        grid[r1, c0:c1 + 1] = GREY
        grid[r0:r1 + 1, c0] = GREY
    sr, sc = spec["seed"]
    grid[sr, sc] = spec["seed_color"]


def _slot_bounds(h: int, w: int) -> dict[str, tuple[int, int, int, int]]:
    mid_r = LOWER_START + (h - LOWER_START) // 2
    mid_c = w // 2
    return {
        "tl": (LOWER_START, mid_r - 1, 0, mid_c - 1),
        "tr": (LOWER_START, mid_r - 1, mid_c, w - 1),
        "bl": (mid_r, h - 1, 0, mid_c - 1),
        "br": (mid_r, h - 1, mid_c, w - 1),
    }


def _slot_dirs(slot: str) -> list[str]:
    return {
        "tl": ["up", "left"],
        "tr": ["up", "right"],
        "bl": ["down", "left"],
        "br": ["down", "right"],
    }[slot]


def _place_wall(h: int, w: int, slot: str, seed_color: int) -> dict | None:
    rs0, rs1, cs0, cs1 = _slot_bounds(h, w)[slot]
    rs0 += 1
    cs0 += 1
    rs1 -= 1
    cs1 -= 1
    directions = _slot_dirs(slot)
    random.shuffle(directions)
    for direction in directions:
        for _ in range(25):
            wh, ww = random.choice(WALL_SIZE_CHOICES)
            if rs1 - rs0 + 1 < wh or cs1 - cs0 + 1 < ww:
                continue
            r0_min = rs0
            r0_max = rs1 - wh + 1
            c0_min = cs0
            c0_max = cs1 - ww + 1
            if direction == "up":
                r0_min = max(r0_min, LOWER_START + 3)
            elif direction == "down":
                r0_max = min(r0_max, h - wh - 3)
            elif direction == "left":
                c0_min = max(c0_min, 3)
            else:
                c0_max = min(c0_max, w - ww - 3)
            if r0_min > r0_max or c0_min > c0_max:
                continue
            r0 = random.randint(r0_min, r0_max)
            c0 = random.randint(c0_min, c0_max)
            r1 = r0 + wh - 1
            c1 = c0 + ww - 1

            if direction in ("up", "down"):
                path_col = random.randint(c0 + 1, c1 - 1)
                if direction == "up":
                    seed_r = random.randint(max(r0 + 1, r1 - 2), r1 - 1)
                else:
                    seed_r = random.randint(r0 + 1, min(r1 - 1, r0 + 2))
                seed_c = path_col
                ray_key = path_col
            else:
                path_row = random.randint(r0 + 1, r1 - 1)
                seed_r = path_row
                seed_c = c0 + 1 if direction == "left" else c1 - 1
                ray_key = path_row

            return {
                "bbox": (r0, c0, r1, c1),
                "direction": direction,
                "seed": (seed_r, seed_c),
                "seed_color": int(seed_color),
                "ray_key": int(ray_key),
            }
    return None


def _place_obstacle(grid: np.ndarray, spec: dict) -> tuple[int, int] | None:
    h, w = grid.shape
    r0, c0, r1, c1 = spec["bbox"]
    direction = spec["direction"]
    if direction == "up":
        rows = list(range(LOWER_START + 1, r0 - 1))
        if not rows:
            return None
        return random.choice(rows), spec["ray_key"]
    if direction == "down":
        rows = list(range(r1 + 2, h - 1))
        if not rows:
            return None
        return random.choice(rows), spec["ray_key"]
    if direction == "left":
        cols = list(range(1, c0 - 1))
        if not cols:
            return None
        return spec["ray_key"], random.choice(cols)
    cols = list(range(c1 + 2, w - 1))
    if not cols:
        return None
    return spec["ray_key"], random.choice(cols)


def _choose_active_columns(w: int, wall_specs: list[dict]) -> set[int] | None:
    must_active = {int(spec["seed"][1]) for spec in wall_specs}
    for spec in wall_specs:
        r0, c0, r1, c1 = spec["bbox"]
        if spec["direction"] in ("up", "down"):
            must_active.add(int(spec["ray_key"]))
        else:
            must_active.add(c0 + 1 if spec["direction"] == "left" else c1 - 1)
    if len(must_active) >= w:
        return None

    for _ in range(120):
        target = random.randint(max(w // 2 + 1, len(must_active)), min(w - 1, max(w // 2 + 2, int(0.78 * w))))
        active = set(must_active)
        pool = [c for c in range(w) if c not in active]
        random.shuffle(pool)
        active.update(pool[: max(0, target - len(active))])
        if len(active) < 3:
            continue
        if max(active) - min(active) + 1 < (w // 2 + 1):
            continue

        good = True
        for spec in wall_specs:
            _r0, c0, _r1, c1 = spec["bbox"]
            interior_cols = set(range(c0 + 1, c1))
            if not (active & interior_cols):
                good = False
                break
            if interior_cols <= active:
                good = False
                break
        if good:
            return active
    return None


def _semantic_checks(grid: np.ndarray) -> bool:
    out, details = _solve(grid, use_active=True, use_shadow=True, return_details=True, strict_conflicts=False)
    out_nomask = _solve(grid, use_active=False, use_shadow=True, strict_conflicts=False)
    out_noshadow, no_shadow_details = _solve(
        grid, use_active=True, use_shadow=False, return_details=True, strict_conflicts=False
    )

    occupied = np.zeros_like(grid, dtype=int)
    for d in details:
        occupied += d["final_fill"].astype(int)
    if np.any(occupied > 1):
        return False

    if np.array_equal(out, grid):
        return False
    if np.array_equal(out, out_nomask):
        return False
    if np.array_equal(out, out_noshadow):
        return False
    if int(np.sum(out[2] == PINK)) < 3:
        return False
    if len(details) < 2:
        return False
    if sum(int(d["final_fill"].sum()) for d in details) < 12:
        return False
    if any(int(d["final_fill"].sum()) < 3 for d in details):
        return False
    if sum(int(d["raw_fill"].sum() > d["final_fill"].sum()) for d in details) < 1:
        return False

    active_cols = {int(c) for c in np.where(grid[0] != BLACK)[0]}
    for d, d_ns in zip(details, no_shadow_details):
        _r0, c0, _r1, c1 = d["bbox"]
        interior_cols = set(range(c0 + 1, c1))
        if not (interior_cols & active_cols):
            return False
        if interior_cols <= active_cols:
            return False
        if int(d_ns["raw_fill"].sum()) <= int(d["final_fill"].sum()):
            return False

    lower_changed = out[LOWER_START:, :] != out_nomask[LOWER_START:, :]
    if int(lower_changed.sum()) < 4:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    grid = np.full((h, w), BLACK, dtype=int)

    control_color = random.choice(CONTROL_COLORS)
    grid[1, 0] = control_color
    grid[2, 0] = control_color
    grid[DIVIDER_ROW, :] = GREY

    slot_names = list(_slot_bounds(h, w))
    random.shuffle(slot_names)
    num_walls = random.choice([2, 3])
    chosen_slots = slot_names[:num_walls]

    seed_palette = [c for c in SEED_COLORS if c != control_color]
    random.shuffle(seed_palette)
    wall_specs = []
    for slot in chosen_slots:
        if not seed_palette:
            return None
        spec = _place_wall(h, w, slot, seed_palette.pop())
        if spec is None:
            return None
        wall_specs.append(spec)

    active_cols = _choose_active_columns(w, wall_specs)
    if active_cols is None:
        return None
    for c in active_cols:
        grid[0, c] = control_color

    for spec in wall_specs:
        _draw_wall(grid, spec)

    obstacle_walls = random.sample(wall_specs, k=random.randint(1, min(2, len(wall_specs))))
    for spec in obstacle_walls:
        pos = _place_obstacle(grid, spec)
        if pos is None:
            return None
        r, c = pos
        if grid[r, c] != BLACK:
            return None
        grid[r, c] = MAROON

    for spec in wall_specs:
        r0, c0, r1, c1 = spec["bbox"]
        if spec["seed"][1] not in active_cols:
            return None
        interior_cols = set(range(c0 + 1, c1))
        if not (interior_cols & active_cols):
            return None
        if interior_cols <= active_cols:
            return None

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
