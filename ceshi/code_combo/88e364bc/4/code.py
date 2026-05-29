# Combo variant of 88e364bc (index 4)
# combo_spec: D2/1+D12/1
# Name: 象限位置匹配+边缘中部图例
#
# Design analysis
# ===============
# 本融合保留原题的运动本体：黄色像素仍在封闭区域内按 legend 指示方向滑动并向远端堆叠。
# 但同时修改两个维度：
# 1. D2/1：区域不再按墙色匹配 legend，而是按几何象限匹配；
# 2. D12/1：legend 不再放在角落，而是放在四条边的中部槽位。
#
# 为了让这两个变化都可观察，generator 强制：
# - legend 填充色与墙色取自不同调色板，颜色匹配必然失败；
# - 至少有一组同墙色区域位于不同象限且方向不同，阻断退化回原题；
# - 3-4 个边中 legend 同时出现，使“边槽位 -> 象限”的新索引规则可见。
#
# 边槽位与象限的对应使用顺时针保序映射：
#   top -> TL, right -> TR, bottom -> BR, left -> BL
# 这相当于把原本四个角的 legend 沿边界顺时针平移到相邻的边中位置。

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


GRID_H, GRID_W = 20, 20
MOVER = 4

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
ALL_DIRS = CARDINAL_DIRS + DIAGONAL_DIRS
EDGE_SLOTS = ["top", "right", "bottom", "left"]
EDGE_TO_QUADRANT = {
    "top": "TL",
    "right": "TR",
    "bottom": "BR",
    "left": "BL",
}
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
CARDINAL_PROB = 0.5
LEGEND_BUFFER = 1

BLOB_GROW_RETRIES = 40
YELLOWS_PER_REGION = [1, 2]
MAX_SLIDE_STEPS = 5
GEN_RETRIES = 500


def _quadrant_from_points(pts: np.ndarray | list[tuple[int, int]], h: int, w: int) -> str:
    arr = np.asarray(pts, dtype=float)
    cr, cc = arr.mean(axis=0)
    return ("T" if cr < h / 2 else "B") + ("L" if cc < w / 2 else "R")


def _legend_5x5(fill: int, dr: int, dc: int) -> np.ndarray:
    g = np.full((5, 5), fill, dtype=int)
    g[1, 1] = 1
    g[1, 2] = 1
    g[1, 3] = 2
    g[3, 1] = 1
    g[3, 2] = 1
    g[3, 3] = 2
    rot = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}
    return np.rot90(g, rot[(dr, dc)])


def _legend_6x7(fill: int, dr: int, dc: int) -> np.ndarray:
    g = np.full((6, 7), fill, dtype=int)
    g[1, 3] = 1
    g[2, 1] = 1
    g[2, 4] = 1
    g[3, 2] = 1
    g[3, 5] = 2
    g[4, 3] = 2
    if (dr, dc) == (1, 1):
        return g
    if (dr, dc) == (-1, 1):
        return np.flipud(g)
    if (dr, dc) == (1, -1):
        return np.fliplr(g)
    if (dr, dc) == (-1, -1):
        return np.rot90(g, 2)
    raise ValueError((dr, dc))


def _edge_origin(slot: str, h: int, w: int, ph: int, pw: int) -> tuple[int, int]:
    if slot == "top":
        return 0, (w - pw) // 2
    if slot == "bottom":
        return h - ph, (w - pw) // 2
    if slot == "left":
        return (h - ph) // 2, 0
    if slot == "right":
        return (h - ph) // 2, w - pw
    raise ValueError(slot)


def _edge_slot_from_bbox(r0: int, c0: int, r1: int, c1: int, h: int, w: int) -> str | None:
    touches = [
        r0 == 0,
        r1 == h - 1,
        c0 == 0,
        c1 == w - 1,
    ]
    if sum(bool(x) for x in touches) != 1:
        return None
    if r0 == 0:
        return "top"
    if r1 == h - 1:
        return "bottom"
    if c0 == 0:
        return "left"
    if c1 == w - 1:
        return "right"
    return None


def _legend_directions_by_quadrant(g: np.ndarray, bg: int) -> dict[str, tuple[int, int]]:
    h, w = g.shape
    best: dict[str, tuple[int, tuple[int, int]]] = {}
    for pts in components_mask(g != bg, connectivity=4):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        r0, c0, r1, c1 = bbox_coords(pts)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area != len(pts):
            continue

        slot = _edge_slot_from_bbox(r0, c0, r1, c1, h, w)
        if slot is None:
            continue
        quadrant = EDGE_TO_QUADRANT[slot]

        counts: dict[int, int] = {}
        for r, c in pts:
            v = int(g[r, c])
            counts[v] = counts.get(v, 0) + 1
        if len(counts) != 3:
            continue

        fill = max(counts, key=counts.get)
        others = [(cnt, col) for col, cnt in counts.items() if col != fill]
        if len(others) != 2:
            continue
        others.sort()
        if others[0][0] == others[1][0]:
            rc = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            scored = []
            for _cnt, col in others:
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                scored.append((np.sum((pts_col.mean(axis=0) - rc) ** 2), col))
            scored.sort()
            center_col = scored[0][1]
            indicator_col = scored[1][1]
        else:
            indicator_col = others[0][1]
            center_col = others[1][1]

        center = np.array([(r, c) for r, c in pts if int(g[r, c]) == center_col], dtype=float)
        indic = np.array([(r, c) for r, c in pts if int(g[r, c]) == indicator_col], dtype=float)
        if len(center) == 0 or len(indic) == 0:
            continue

        delta = indic.mean(axis=0) - center.mean(axis=0)
        dr = 0 if abs(float(delta[0])) < 1e-6 else (1 if delta[0] > 0 else -1)
        dc = 0 if abs(float(delta[1])) < 1e-6 else (1 if delta[1] > 0 else -1)
        if dr == 0 and dc == 0:
            continue

        prev = best.get(quadrant)
        if prev is None or len(pts) > prev[0]:
            best[quadrant] = (len(pts), (dr, dc))

    return {quadrant: direction for quadrant, (_sz, direction) in best.items()}


def _slide_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
            return False
    return True


def _forward_slide(
    region_set: set[tuple[int, int]],
    start: tuple[int, int],
    dr: int,
    dc: int,
    blocked: set[tuple[int, int]],
) -> tuple[int, int]:
    r, c = start
    while True:
        if not _slide_step_valid(region_set, r, c, dr, dc):
            break
        nr, nc = r + dr, c + dc
        if (nr, nc) in blocked:
            break
        r, c = nr, nc
    return (r, c)


def _slide_points(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
    if not points:
        return []
    points_sorted = sorted(points, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    occupied: set[tuple[int, int]] = set()
    out_pts: list[tuple[int, int]] = []
    for start in points_sorted:
        final = _forward_slide(region_set, start, dr, dc, occupied)
        occupied.add(final)
        out_pts.append(final)
    return out_pts


def main(grid: np.ndarray) -> np.ndarray:
    """
    边缘中部的 legend 按位置编码象限，而不是按颜色匹配区域。

    - `top/right/bottom/left` 四个边中槽位分别控制 `TL/TR/BR/BL` 四个象限；
    - region 的墙色只用于显示封闭边界，不再决定与哪个 legend 配对；
    - legend 内部仍沿用原题的 center -> indicator 方向编码。
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    dirs = _legend_directions_by_quadrant(g, bg)
    assert dirs, "expected at least one border legend in an edge-middle slot"

    passable = (g == bg) | (g == MOVER)
    out = g.copy()

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == MOVER]
        if not movers:
            continue

        quadrant = _quadrant_from_points(pts, h, w)
        if quadrant not in dirs:
            continue
        dr, dc = dirs[quadrant]

        pset = {(int(r), int(c)) for r, c in pts}
        for r, c in movers:
            out[r, c] = bg
        for r, c in _slide_points(pset, movers, dr, dc):
            out[r, c] = MOVER

    return out


def _grow_blob(
    canvas_mask: np.ndarray, seed: tuple[int, int], target_size: int, rng: random.Random
) -> set[tuple[int, int]] | None:
    h, w = canvas_mask.shape
    sr, sc = seed
    if not canvas_mask[sr, sc]:
        return None
    blob = {(sr, sc)}
    frontier: list[tuple[int, int]] = []

    def push_neighbors(r: int, c: int) -> None:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and canvas_mask[nr, nc] and (nr, nc) not in blob:
                frontier.append((nr, nc))

    push_neighbors(sr, sc)
    while len(blob) < target_size and frontier:
        idx = rng.randrange(len(frontier))
        r, c = frontier.pop(idx)
        if (r, c) in blob or not canvas_mask[r, c]:
            continue
        blob.add((r, c))
        push_neighbors(r, c)
    if len(blob) < target_size:
        return None
    return blob


def _blob_wall_cells(blob: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    wall = set()
    for r, c in blob:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in blob:
                wall.add((nr, nc))
    return wall


def _fill_holes(blob: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    walls = _blob_wall_cells(blob, h, w)
    barrier = np.zeros((h, w), dtype=bool)
    for r, c in blob | walls:
        barrier[r, c] = True

    reachable = np.zeros((h, w), dtype=bool)
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if not barrier[r, c]:
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not barrier[r, c]:
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        if reachable[r, c] or barrier[r, c]:
            continue
        reachable[r, c] = True
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not reachable[nr, nc] and not barrier[nr, nc]:
                stack.append((nr, nc))

    trapped = (~reachable) & (~barrier)
    if not trapped.any():
        return blob
    filled = set(blob)
    for r, c in np.argwhere(trapped):
        filled.add((int(r), int(c)))
    return filled


def _random_direction(rng: random.Random) -> tuple[int, int]:
    if rng.random() < CARDINAL_PROB:
        return rng.choice(CARDINAL_DIRS)
    return rng.choice(DIAGONAL_DIRS)


def _place_legends(rng: random.Random, grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    h, w = grid.shape
    n_legends = rng.randint(3, 4)
    slots = rng.sample(EDGE_SLOTS, n_legends)
    fills = rng.sample(WALL_COLOR_CHOICES, n_legends)

    while True:
        dirs = [_random_direction(rng) for _ in range(n_legends)]
        if len(set(dirs)) >= 2:
            break

    legend_occupied = np.zeros((h, w), dtype=bool)
    legends = []
    for slot, fill, (dr, dc) in zip(slots, fills, dirs):
        patch = _legend_5x5(fill, dr, dc) if (dr, dc) in CARDINAL_DIRS else _legend_6x7(fill, dr, dc)
        ph, pw = patch.shape
        r0, c0 = _edge_origin(slot, h, w, ph, pw)
        paste(grid, patch, r0, c0)

        r_lo = max(0, r0 - LEGEND_BUFFER)
        r_hi = min(h, r0 + ph + LEGEND_BUFFER)
        c_lo = max(0, c0 - LEGEND_BUFFER)
        c_hi = min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append(
            {
                "slot": slot,
                "quadrant": EDGE_TO_QUADRANT[slot],
                "fill": fill,
                "dir": (dr, dc),
            }
        )

    return legend_occupied, legends


def _assign_wall_colors(rng: random.Random, legends: list[dict]) -> dict[str, int] | None:
    legend_fills = {int(leg["fill"]) for leg in legends}
    palette = [c for c in WALL_COLOR_CHOICES if c not in legend_fills]
    if len(palette) < 2:
        return None
    shared_color, alt_color = rng.sample(palette, 2)

    quadrants = [str(leg["quadrant"]) for leg in legends]
    dirs = {str(leg["quadrant"]): tuple(leg["dir"]) for leg in legends}
    differing_pairs = [
        (a, b)
        for i, a in enumerate(quadrants)
        for b in quadrants[i + 1 :]
        if dirs[a] != dirs[b]
    ]
    if not differing_pairs:
        return None
    a, b = rng.choice(differing_pairs)

    wall_by_quadrant = {a: shared_color, b: shared_color}
    remaining = [q for q in quadrants if q not in wall_by_quadrant]
    rng.shuffle(remaining)
    if remaining:
        wall_by_quadrant[remaining[0]] = alt_color
    for q in remaining[1:]:
        wall_by_quadrant[q] = rng.choice([shared_color, alt_color])
    return wall_by_quadrant


def _quadrant_frame(quadrant: str) -> tuple[int, int, int, int]:
    if quadrant == "TL":
        return (2, 6, 2, 6)
    if quadrant == "TR":
        return (2, 6, 13, 17)
    if quadrant == "BL":
        return (13, 17, 2, 6)
    if quadrant == "BR":
        return (13, 17, 13, 17)
    raise ValueError(quadrant)


def _try_make_region_rect(
    rng: random.Random,
    quadrant: str,
    occupied: np.ndarray,
) -> set[tuple[int, int]] | None:
    h, w = occupied.shape
    r0_min, r0_max, c0_min, c0_max = _quadrant_frame(quadrant)
    for _ in range(BLOB_GROW_RETRIES):
        rect_h = rng.randint(3, 4)
        rect_w = rng.randint(3, 4)
        r0 = rng.randint(r0_min, r0_max - rect_h + 1)
        c0 = rng.randint(c0_min, c0_max - rect_w + 1)
        rect = {(r, c) for r in range(r0, r0 + rect_h) for c in range(c0, c0 + rect_w)}
        walls = _blob_wall_cells(rect, h, w)
        if any(occupied[r, c] for r, c in rect | walls):
            continue
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in walls):
            continue
        return rect
    return None


def _grow_regions(rng: random.Random, legends: list[dict], legend_occupied: np.ndarray) -> list[dict] | None:
    h, w = legend_occupied.shape
    occupied = legend_occupied.copy()
    wall_by_quadrant = _assign_wall_colors(rng, legends)
    if wall_by_quadrant is None:
        return None

    regions = []
    for leg in legends:
        quadrant = str(leg["quadrant"])
        blob = _try_make_region_rect(rng, quadrant, occupied)
        if blob is None:
            return None
        if _quadrant_from_points(list(blob), h, w) != quadrant:
            return None

        walls = _blob_wall_cells(blob, h, w)
        regions.append(
            {
                "quadrant": quadrant,
                "color": wall_by_quadrant[quadrant],
                "dir": tuple(leg["dir"]),
                "interior": blob,
                "walls": walls,
            }
        )
        for r, c in blob | walls:
            occupied[r, c] = True

    same_color_dirs: dict[int, list[tuple[int, int]]] = {}
    for reg in regions:
        same_color_dirs.setdefault(int(reg["color"]), []).append(tuple(reg["dir"]))
    if not any(len(set(v)) >= 2 for v in same_color_dirs.values() if len(v) >= 2):
        return None
    return regions


def _validate_topology(grid: np.ndarray, regions: list[dict]) -> bool:
    h, w = grid.shape
    labeled, n_comp = label_mask(grid == 0, connectivity=4)
    bounded: list[set[tuple[int, int]]] = []
    for comp_id in range(1, n_comp + 1):
        coords = np.argwhere(labeled == comp_id)
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in coords):
            continue
        bounded.append({(int(r), int(c)) for r, c in coords})
    region_sets = [reg["interior"] for reg in regions]
    if len(bounded) != len(region_sets):
        return False
    return all(comp in region_sets for comp in bounded)


def _find_reverse_slide(
    blob: set[tuple[int, int]],
    out_pos: tuple[int, int],
    dr: int,
    dc: int,
    already_stopped: set[tuple[int, int]],
    existing_inputs: list[tuple[int, int]],
    rng: random.Random,
) -> tuple[int, int] | None:
    target_steps = rng.randint(1, MAX_SLIDE_STEPS)
    for steps in range(target_steps, 0, -1):
        r, c = out_pos
        ok = True
        for _ in range(steps):
            pr, pc = r - dr, c - dc
            if (pr, pc) not in blob or (pr, pc) in already_stopped or (pr, pc) in existing_inputs:
                ok = False
                break
            if dr != 0 and dc != 0:
                if (pr + dr, pc) not in blob or (pr, pc + dc) not in blob:
                    ok = False
                    break
            r, c = pr, pc
        if ok and _forward_slide(blob, (r, c), dr, dc, already_stopped) == out_pos:
            return (r, c)
    return None


def _place_yellows(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
    for reg in regions:
        blob = reg["interior"]
        dr, dc = reg["dir"]
        n_yellows = rng.choice(YELLOWS_PER_REGION)

        cells_sorted = sorted(blob, key=lambda p: dr * p[0] + dc * p[1], reverse=True)
        far_key = dr * cells_sorted[0][0] + dc * cells_sorted[0][1]
        tier1 = [p for p in cells_sorted if dr * p[0] + dc * p[1] == far_key]
        first = rng.choice(tier1)
        output_positions = [first]

        if n_yellows == 2:
            pr, pc = first[0] - dr, first[1] - dc
            cands = ([(pr, pc)] if (pr, pc) in blob else []) + [p for p in tier1 if p != first]
            if cands:
                second = rng.choice(cands)
                if second != first:
                    output_positions.append(second)

        out_sorted = sorted(output_positions, key=lambda p: dr * p[0] + dc * p[1], reverse=True)
        already_stopped: set[tuple[int, int]] = set()
        yellows_input: list[tuple[int, int]] = []
        for out_pos in out_sorted:
            candidate = _find_reverse_slide(blob, out_pos, dr, dc, already_stopped, yellows_input, rng)
            if candidate is None:
                return False
            yellows_input.append(candidate)
            already_stopped.add(out_pos)

        for r, c in yellows_input:
            grid[r, c] = MOVER
    return True


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    h, w = input_grid.shape
    try:
        solver_out = main(input_grid)
    except Exception:
        return False

    struct_mask = (input_grid != MOVER) & (input_grid != 0)
    if not np.array_equal(solver_out[struct_mask], input_grid[struct_mask]):
        return False
    if int((solver_out == MOVER).sum()) != int((input_grid == MOVER).sum()):
        return False

    regions_mask = np.zeros((h, w), dtype=bool)
    for reg in regions:
        for r, c in reg["interior"]:
            regions_mask[r, c] = True
    if not bool(np.all(regions_mask[solver_out == MOVER])):
        return False

    same_color_dirs: dict[int, list[tuple[int, int]]] = {}
    for reg in regions:
        same_color_dirs.setdefault(int(reg["color"]), []).append(tuple(reg["dir"]))
    if not any(len(set(v)) >= 2 for v in same_color_dirs.values() if len(v) >= 2):
        return False

    active_quadrants = {str(reg["quadrant"]) for reg in regions}
    return len(active_quadrants) >= 3


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    legend_occupied, legends = _place_legends(rng, grid)
    regions = _grow_regions(rng, legends, legend_occupied)
    if regions is None:
        return None

    for reg in regions:
        for r, c in reg["walls"]:
            grid[r, c] = reg["color"]

    if not _validate_topology(grid, regions):
        return None

    input_grid = grid.copy()
    if not _place_yellows(rng, regions, input_grid):
        return None
    if not _oracle_check(input_grid, regions):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            g = _gen_attempt(rng)
        except Exception:
            continue
        if g is not None:
            return g
    raise RuntimeError("generate_input: exceeded retries")
