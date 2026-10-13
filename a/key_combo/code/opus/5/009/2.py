from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
YELLOW = 4
MAGENTA = 6
CYAN = 8


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                nbrs = neighbors4 if connectivity == 4 else neighbors8
                for nr, nc in nbrs(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def object_infos(grid: np.ndarray) -> list[dict]:
    infos: list[dict] = []
    for pts in components(grid == BLUE, connectivity=4):
        r0, c0, r1, c1 = bbox_from_coords(pts)
        infos.append(
            {
                "coords": pts,
                "bbox": (r0, c0, r1, c1),
                "cavities": cavity_regions(pts),
            }
        )
    return infos


def cavity_regions(coords: np.ndarray) -> list[set[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox_from_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    obj = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        obj[r - r0, c - c0] = True

    regions: list[set[tuple[int, int]]] = []
    for pts in components(~obj, connectivity=4):
        rr = pts[:, 0]
        cc = pts[:, 1]
        touches_top = bool(np.any(rr == 0))
        touches_side = bool(np.any(cc == 0) or np.any(cc == w - 1))
        touches_bottom = bool(np.any(rr == h - 1))
        if touches_top and not touches_side and not touches_bottom:
            regions.append({(int(r + r0), int(c + c0)) for r, c in pts})
    return regions


def draw_outline(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    blue_mask = out == BLUE
    outline = np.zeros_like(blue_mask)
    for r, c in np.argwhere(blue_mask):
        for nr, nc in neighbors8(int(r), int(c)):
            if in_bounds(out.shape, nr, nc) and out[nr, nc] == BG:
                outline[nr, nc] = True
    out[outline] = RED
    return out


def gravity_drop(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    h, w = out.shape
    yellow_positions = [tuple(map(int, rc)) for rc in np.argwhere(out == YELLOW)]
    out[out == YELLOW] = BG
    by_col: dict[int, list[int]] = {}
    for r, c in yellow_positions:
        by_col.setdefault(c, []).append(r)
    for c, rows in by_col.items():
        for start_r in sorted(rows, reverse=True):
            rr = int(start_r)
            while rr + 1 < h and out[rr + 1, c] == BG:
                rr += 1
            out[rr, c] = YELLOW
    return out


def solve(grid: np.ndarray, *, use_outline: bool = True) -> tuple[np.ndarray, list[dict], list[bool]]:
    stage = draw_outline(grid) if use_outline else np.array(grid, dtype=int, copy=True)
    dropped = gravity_drop(stage)
    infos = object_infos(np.asarray(grid, dtype=int))
    hits: list[bool] = []
    final = np.array(dropped, dtype=int, copy=True)
    for info in infos:
        cavity_union = set().union(*info["cavities"]) if info["cavities"] else set()
        hit = any(final[r, c] == YELLOW for r, c in cavity_union)
        hits.append(hit)
        if not hit:
            continue
        for r, c in info["coords"]:
            final[int(r), int(c)] = CYAN
        for cavity in info["cavities"]:
            for r, c in cavity:
                final[int(r), int(c)] = MAGENTA
    return final, infos, hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw red fences around blue objects, drop yellow markers vertically, then recolor only the cavity objects that catch a marker.

    Input structure:
    - Black background contains 4-5 separated blue foreground objects and several yellow single-pixel markers above them.
    - Some blue objects are container-like 4-connected components whose bounding boxes include a top-entry internal cavity; the others are solid.
    - Yellow markers are placed in columns above blue objects or their future red outline positions.

    Rule:
    - For every blue component, paint red on every background cell in its 8-neighborhood, keeping blue and yellow cells unchanged.
    - Then let all yellow pixels fall straight down within their own columns, resolving each column from bottom to top; a yellow stops above the first red cell, blue cell, settled yellow, or the grid bottom.
    - For each blue object that has at least one internal top-entry cavity inside its original bounding box, check whether any settled yellow ends inside that cavity region.
    - If such a cavity object is hit, recolor its blue body to cyan and fill all of its cavity cells with magenta, covering any yellow there. All other pixels stay as they are after the fall.

    Key insight:
    - The red outline is both decoration and a physical fence: it narrows or seals cavity entrances, so hit detection must be computed after outlining and falling, not from object shape alone.
    """
    return solve(np.asarray(grid, dtype=int))[0]


def cavity_patch(height: int, width: int, neck_width: int, shift: int) -> tuple[np.ndarray, list[int], int]:
    patch = np.zeros((height, width), dtype=int)
    patch[:, 0] = BLUE
    patch[:, -1] = BLUE
    patch[-1, :] = BLUE
    patch[0, :] = BLUE
    open_start = (width - neck_width) // 2 + shift
    open_start = max(1, min(open_start, width - neck_width - 1))
    patch[0, open_start : open_start + neck_width] = BG
    target_col = open_start + neck_width // 2
    blocked_cols = [open_start, open_start + neck_width - 1]
    return patch, blocked_cols, target_col


def solid_patch(kind: int) -> np.ndarray:
    if kind == 0:
        return np.ones((3, 4), dtype=int) * BLUE
    if kind == 1:
        return np.ones((4, 3), dtype=int) * BLUE
    if kind == 2:
        patch = np.ones((4, 4), dtype=int) * BLUE
        patch[0, 3] = BG
        patch[3, 0] = BG
        return patch
    patch = np.ones((4, 4), dtype=int) * BLUE
    patch[1, 0] = BG
    patch[2, 3] = BG
    return patch


def scatter_gaps(count: int, total: int, minimum: int) -> list[int]:
    vals = [minimum] * count
    extra = total - count * minimum
    for _ in range(extra):
        vals[random.randrange(count)] += 1
    return vals


def build_attempt() -> np.ndarray | None:
    n_objects = random.choice([4, 5])
    core_count = 4
    cavity_count = random.choice([2, 3])
    solid_kinds = [random.choice([0, 1, 2, 3]) for _ in range(core_count - cavity_count)]

    cavity_modes = ["hit", "miss"]
    while len(cavity_modes) < cavity_count:
        cavity_modes.append(random.choice(["hit", "miss"]))
    random.shuffle(cavity_modes)

    specs: list[dict] = []
    for mode in cavity_modes:
        height = random.choice([5, 6])
        width = 6
        neck_width = 3 if mode == "hit" else 2
        shift = random.choice([-1, 0, 1])
        patch, blocked_cols, open_col = cavity_patch(height, width, neck_width, shift)
        specs.append(
            {
                "patch": patch,
                "kind": "cavity",
                "mode": mode,
                "open_col": open_col,
                "blocked_cols": blocked_cols,
            }
        )
    for kind in solid_kinds:
        patch = solid_patch(kind)
        specs.append(
            {
                "patch": patch,
                "kind": "solid",
                "mode": "solid",
            }
        )
    random.shuffle(specs)

    widths = [spec["patch"].shape[1] for spec in specs]
    total_width = sum(widths)
    min_total = total_width + 2 * (core_count - 1) + 2
    if min_total > 30:
        return None
    width = random.randint(max(20, min_total), min(30, min_total + 4))
    base_gaps = [1] + [2] * (core_count - 1) + [1]
    gaps = list(base_gaps)
    extra_gap = width - total_width - sum(base_gaps)
    for _ in range(extra_gap):
        gaps[random.randrange(len(gaps))] += 1

    max_height = max(spec["patch"].shape[0] for spec in specs)
    height = random.randint(20, 27) if n_objects == 4 else random.randint(24, 30)
    top_hi = height - max_height - (2 if n_objects == 4 else 7)
    if top_hi < 8:
        return None
    top = random.randint(8, top_hi)
    grid = np.zeros((height, width), dtype=int)

    col = gaps[0]
    placements: list[dict] = []
    for idx, spec in enumerate(specs):
        patch = spec["patch"]
        ph, pw = patch.shape
        row = top + random.choice([0, 0, 1])
        if row + ph > height - 1:
            row = height - ph - 1
        area = grid[row : row + ph, col : col + pw]
        if np.any(area != BG):
            return None
        mask = patch != BG
        area[mask] = patch[mask]
        spec = dict(spec)
        spec["row"] = row
        spec["col"] = col
        placements.append(spec)
        col += pw + gaps[idx + 1]

    if n_objects == 5:
        extra_patch = solid_patch(random.choice([0, 1, 2, 3]))
        eph, epw = extra_patch.shape
        row_lo = top + max_height + 2
        row_hi = height - eph - 1
        if row_lo > row_hi:
            return None
        placed_extra = False
        for _ in range(80):
            row = random.randint(row_lo, row_hi)
            col0 = random.randint(1, width - epw - 1)
            r_lo = max(0, row - 2)
            r_hi = min(height, row + eph + 2)
            c_lo = max(0, col0 - 2)
            c_hi = min(width, col0 + epw + 2)
            if np.any(grid[r_lo:r_hi, c_lo:c_hi] == BLUE):
                continue
            if np.any(grid[row : row + eph, col0 : col0 + epw] != BG):
                continue
            area = grid[row : row + eph, col0 : col0 + epw]
            mask = extra_patch != BG
            area[mask] = extra_patch[mask]
            placements.append(
                {
                    "patch": extra_patch,
                    "kind": "solid",
                    "mode": "solid",
                    "row": row,
                    "col": col0,
                }
            )
            placed_extra = True
            break
        if not placed_extra:
            return None

    top_limit = min(p["row"] for p in placements) - 2
    if top_limit < 2:
        return None

    used_yellows: set[tuple[int, int]] = set()

    def add_yellow(c: int, preferred_rows: list[int]) -> bool:
        choices = [r for r in preferred_rows if 0 <= r < top_limit and (r, c) not in used_yellows]
        if not choices:
            choices = [r for r in range(top_limit) if (r, c) not in used_yellows]
        if not choices:
            return False
        r = random.choice(choices)
        used_yellows.add((r, c))
        grid[r, c] = YELLOW
        return True

    for spec in placements:
        if spec["kind"] == "cavity":
            open_c = spec["col"] + spec["open_col"]
            blocked_abs = [spec["col"] + cc for cc in spec["blocked_cols"]]
            if spec["mode"] == "hit":
                if not add_yellow(open_c, [0, 1, 2, 3, 4]):
                    return None
                if random.random() < 0.5:
                    if not add_yellow(open_c, [5, 6, 7]):
                        return None
                if random.random() < 0.65:
                    if not add_yellow(random.choice(blocked_abs), [0, 2, 4, 6]):
                        return None
            else:
                if not add_yellow(open_c, [1, 2, 3, 4, 5]):
                    return None
                if random.random() < 0.4:
                    if not add_yellow(random.choice(blocked_abs), [0, 2, 4, 6]):
                        return None
        else:
            pw = spec["patch"].shape[1]
            target_c = spec["col"] + random.randint(0, pw - 1)
            if not add_yellow(target_c, [0, 1, 3, 5, 7]):
                return None
            if random.random() < 0.35:
                if not add_yellow(target_c, [2, 4, 6]):
                    return None

    return grid


def validate_instance(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    solved, infos, hits = solve(grid, use_outline=True)
    raw_solved, raw_infos, raw_hits = solve(grid, use_outline=False)

    if len(infos) < 4 or len(infos) > 5:
        return False

    cavity_ids = [i for i, info in enumerate(infos) if info["cavities"]]
    if len(cavity_ids) < 2 or len(cavity_ids) > 3:
        return False
    if not any(hits[i] for i in cavity_ids):
        return False
    if not any(not hits[i] for i in cavity_ids):
        return False
    if not any(raw_hits[i] != hits[i] for i in cavity_ids):
        return False

    if not np.any(solved == RED):
        return False
    if not np.any(solved == CYAN):
        return False
    if not np.any(solved == MAGENTA):
        return False

    for i, info in enumerate(infos):
        body_cells = [tuple(map(int, rc)) for rc in info["coords"]]
        body_vals = {int(solved[r, c]) for r, c in body_cells}
        if hits[i]:
            if body_vals != {CYAN}:
                return False
            for cavity in info["cavities"]:
                if any(int(solved[r, c]) != MAGENTA for r, c in cavity):
                    return False
        else:
            if body_vals != {BLUE}:
                return False

    if np.array_equal(grid, solved):
        return False
    if np.array_equal(raw_solved, solved):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = build_attempt()
        if grid is not None and validate_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
