# Variant of 88e364bc
# Dimension: D15 — 多步处理与操作顺序 (multi_step)
# Change: 单步处理——所有区域的黄色像素独立滑动一次即完成... → 迭代滑动直到稳定

# Design analysis
# ===============
# source task note:
# - The prompt asks to read solver code for source task 28a6681f first.
# - In this workspace I could find
#   /home/lixinyu/snap/new/ARC-data/ARC-AGI-2/data/evaluation/28a6681f.json
#   but no local solver source file named 28a6681f.py under the allowed paths.
# - I therefore use the requested source-task idea only at the mechanism level:
#   repeated synchronous rounds whose intermediate states matter.
#
# variant_spec:
#   preserved_structure:
#     grid: 20x20, bg=0
#     legends:
#       - border rectangles still encode direction through center-vs-indicator centroids
#       - legend fill color still matches the wall color of the controlled region
#     regions:
#       - still enclosed by 1-cell-thick colored walls
#       - still contain yellow movers on bg interior cells
#   d15_core_change:
#     old_rule:
#       - per region, yellow pixels were processed once and each one slid all the way to the far wall
#     new_rule:
#       - per round, every yellow attempts to advance exactly one cell in its legend direction
#       - movement decisions use the occupancy at the start of that round
#       - after applying all allowed one-cell moves simultaneously, another round starts
#       - repeat until no yellow can move
#   necessary_adaptation_for_observability:
#     - If the task only changes from "slide to wall in one pass" to "repeat one-cell steps",
#       the final packed state stays the same on monotone straight chains, so the D15 change is
#       invisible in ARC input/output pairs.
#     - To make the iterative dependency visible while staying close to the original diagonal
#       corner-cut rule, diagonal steps now also require the two orthogonal side cells to be free
#       of yellow at the start of the round.
#     - This means one yellow can temporarily block another's diagonal passage; after several
#       rounds the front yellow may clear, but the final arrangement can differ from the old
#       slide-to-wall solver.
#     - Generator samples are biased to diagonal legends and include at least one region whose
#       two yellows form this exact "wait for passage to clear" pattern.
#
# behavior_trace:
#   local_example:
#     region_interior: 4x4 open square
#     direction: down-right
#     input_yellows: [(top-left), (top-left+right)]
#   old_solver:
#     - far yellow slides diagonally to the far corner band
#     - rear yellow then keeps sliding through the newly cleared path and ends deeper
#   new_solver:
#     round_1:
#       - front yellow moves one step
#       - rear yellow is blocked because its east side-neighbor is occupied at round start
#     round_2+:
#       - rear yellow starts moving only after that side cell is clear
#       - later it can again be blocked by the front yellow occupying the side cell of a diagonal step
#     result:
#       - final positions differ from the old one-pass slide-to-wall result
#
# implicit_assumption_audit_for_main:
#   _legend_directions:
#     assumption: legends remain border-touching solid rectangles with exactly 3 colors.
#     status: preserved.
#   passable region extraction:
#     assumption: movers live only on bg interior cells and walls remain static.
#     status: preserved.
#   old _slide_points loop:
#     assumption: one pass is sufficient because the movement relation can be compressed into
#       a far-to-near stacking operation.
#     status: no longer valid once diagonal side-cells can be yellow blockers whose occupancy
#       changes across rounds.
#   new round simulator:
#     assumptions:
#       - all movers in a region share one direction, still determined by the majority border color
#       - same-round movement is evaluated against the round-start occupancy
#       - translation by a fixed direction is injective, so simultaneous one-cell moves cannot merge
#     generator guarantees:
#       - all controlled regions are clean enclosed rectangles, so border-color detection stays unambiguous
#       - at least one region uses a diagonal interaction pattern that needs more than one round
#
# placement_legality:
#   - Legends are placed at distinct corners and reserve a one-cell buffer.
#   - Regions are axis-aligned closed rectangles with a one-cell buffer from legends and from each other.
#   - Each special region is large enough to host a guaranteed multi-round diagonal interaction pattern.
#   - Yellow inputs are only placed on interior cells and never overlap.
#
# visual_observability:
#   - Multiple training examples can show the same visual motif: two yellows near the leading edge of
#     a diagonal-controlled chamber do not both end up in the farthest possible cells.
#   - Because the walls, legends, and yellows remain visually simple, the new sequential dependency
#     is readable from a few examples rather than hidden in generator-only metadata.

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    h, w = g.shape
    non_bg = g != bg
    best: dict[int, tuple[int, tuple[int, int]]] = {}
    for pts in components_mask(non_bg, connectivity=4):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area != len(pts):
            continue

        counts: dict[int, int] = {}
        for r, c in pts:
            v = int(g[r, c])
            counts[v] = counts.get(v, 0) + 1
        if len(counts) != 3:
            continue

        bg_col = max(counts, key=counts.get)
        others = [(cnt, col) for col, cnt in counts.items() if col != bg_col]
        others.sort()
        if len(others) < 2:
            continue

        if others[0][0] == others[1][0]:
            rc = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            cols = [others[0][1], others[1][1]]
            cents = []
            for col in cols:
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                cents.append((np.sum((pts_col.mean(axis=0) - rc) ** 2), col))
            cents.sort()
            center_col = cents[0][1]
            indicator_col = cents[1][1]
        else:
            indicator_col = others[0][1]
            center_col = others[-1][1]

        center = [(r, c) for r, c in pts if int(g[r, c]) == center_col]
        indic = [(r, c) for r, c in pts if int(g[r, c]) == indicator_col]
        if not center or not indic:
            continue

        cc = np.mean(np.array(center, dtype=float), axis=0)
        ci = np.mean(np.array(indic, dtype=float), axis=0)
        vr, vc = (ci - cc).tolist()
        dr = 0 if abs(vr) < 1e-6 else (1 if vr > 0 else -1)
        dc = 0 if abs(vc) < 1e-6 else (1 if vc > 0 else -1)
        if dr == 0 and dc == 0:
            continue

        prev = best.get(bg_col)
        if prev is None or len(pts) > prev[0]:
            best[bg_col] = (len(pts), (dr, dc))

    return {k: v for k, (_, v) in best.items()}


def _step_can_move(
    region_set: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    r: int,
    c: int,
    dr: int,
    dc: int,
) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if (nr, nc) in occupied:
        return False
    if dr != 0 and dc != 0:
        side1 = (r + dr, c)
        side2 = (r, c + dc)
        if side1 not in region_set or side2 not in region_set:
            return False
        # Variant-specific dynamic gate: a yellow cannot brush diagonally past another yellow.
        if side1 in occupied or side2 in occupied:
            return False
    return True


def _iterate_until_stable(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> tuple[list[tuple[int, int]], int]:
    occupied = {(int(r), int(c)) for r, c in points}
    rounds = 0
    while True:
        movers = {
            (r, c)
            for r, c in occupied
            if _step_can_move(region_set, occupied, r, c, dr, dc)
        }
        if not movers:
            break
        occupied = {
            (r + dr, c + dc) if (r, c) in movers else (r, c)
            for r, c in occupied
        }
        rounds += 1
    return sorted(occupied), rounds


def _slide_points_old(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
    if not points:
        return []
    points_sorted = sorted(points, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    occupied = set()
    out_pts = []
    for r0, c0 in points_sorted:
        r, c = r0, c0
        while True:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in region_set:
                break
            if dr != 0 and dc != 0:
                if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
                    break
            if (nr, nc) in occupied:
                break
            r, c = nr, nc
        occupied.add((r, c))
        out_pts.append((r, c))
    return out_pts


def _collect_regions(
    g: np.ndarray,
    bg: int,
    mover: int,
    dirs: dict[int, tuple[int, int]],
) -> list[dict]:
    h, w = g.shape
    passable = (g == bg) | (g == mover)
    infos = []
    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == mover]
        if not movers:
            continue

        border_counts: dict[int, int] = {}
        region_set = {(int(r), int(c)) for r, c in pts}
        for r, c in pts:
            for nr, nc in neighbors4(int(r), int(c)):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in region_set:
                    continue
                v = int(g[nr, nc])
                if v == bg or v == mover:
                    continue
                border_counts[v] = border_counts.get(v, 0) + 1
        if not border_counts:
            continue
        border_col = max(border_counts, key=border_counts.get)
        if border_col not in dirs:
            continue
        infos.append(
            {
                "region_set": region_set,
                "movers": movers,
                "dir": dirs[border_col],
                "border_color": border_col,
            }
        )
    return infos


def _solve(grid: np.ndarray, *, iterative: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    mover = 4
    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one legend"

    out = g.copy()
    for info in _collect_regions(g, bg, mover, dirs):
        for r, c in info["movers"]:
            out[r, c] = bg
        dr, dc = info["dir"]
        if iterative:
            new_pts, _ = _iterate_until_stable(info["region_set"], info["movers"], dr, dc)
        else:
            new_pts = _slide_points_old(info["region_set"], info["movers"], dr, dc)
        for r, c in new_pts:
            out[r, c] = mover
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move yellow pixels inside each walled region according to its border legend.

    Variant rule:
    - Yellows advance only one cell per round.
    - All move decisions in a round are based on the round-start occupancy.
    - Diagonal steps also require the two orthogonal side cells to be free of yellow.
    - Repeat rounds until no yellow moves anywhere.
    """
    return _solve(grid, iterative=True)


def _old_main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, iterative=False)


# --------------- generator ---------------

GRID_H, GRID_W = 20, 20

CORNERS = ["TL", "TR", "BL", "BR"]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
LEGEND_BUFFER = 1

REGIONS_PER_COLOR = (1, 2)
MAX_TOTAL_REGIONS = 3
SPECIAL_INTERIOR_SIZES = [4, 5, 6]
PLAIN_INTERIOR_H = [3, 4, 5, 6]
PLAIN_INTERIOR_W = [4, 5, 6, 7]
PLACEMENT_RETRIES = 120
GEN_RETRIES = 400


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


def _corner_origin(corner: str, h: int, w: int, ph: int, pw: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, w - pw
    if corner == "BL":
        return h - ph, 0
    if corner == "BR":
        return h - ph, w - pw
    raise ValueError(corner)


def _place_legends(
    rng: random.Random,
    grid: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, tuple[int, int]]]]:
    h, w = grid.shape
    n_legends = rng.choice([1, 2])
    corners = rng.sample(CORNERS, n_legends)
    wall_colors = rng.sample(WALL_COLOR_CHOICES, n_legends)

    reserved = np.zeros((h, w), dtype=bool)
    legends: list[tuple[int, tuple[int, int]]] = []
    for corner, wall_color in zip(corners, wall_colors):
        dr, dc = rng.choice(DIAGONAL_DIRS)
        patch = _legend_6x7(wall_color, dr, dc)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo = max(0, r0 - LEGEND_BUFFER)
        r_hi = min(h, r0 + ph + LEGEND_BUFFER)
        c_lo = max(0, c0 - LEGEND_BUFFER)
        c_hi = min(w, c0 + pw + LEGEND_BUFFER)
        reserved[r_lo:r_hi, c_lo:c_hi] = True
        legends.append((wall_color, (dr, dc)))
    return reserved, legends


def _can_place_box(
    reserved: np.ndarray,
    outer_r0: int,
    outer_c0: int,
    outer_h: int,
    outer_w: int,
) -> bool:
    h, w = reserved.shape
    if outer_r0 <= 0 or outer_c0 <= 0:
        return False
    if outer_r0 + outer_h >= h or outer_c0 + outer_w >= w:
        return False
    r_lo = outer_r0 - 1
    c_lo = outer_c0 - 1
    r_hi = outer_r0 + outer_h + 1
    c_hi = outer_c0 + outer_w + 1
    return not reserved[r_lo:r_hi, c_lo:c_hi].any()


def _paint_rect_region(
    grid: np.ndarray,
    reserved: np.ndarray,
    outer_r0: int,
    outer_c0: int,
    inner_h: int,
    inner_w: int,
    color: int,
    direction: tuple[int, int],
    *,
    special: bool,
) -> dict:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    walls = set()
    for c in range(outer_c0, outer_c0 + outer_w):
        walls.add((outer_r0, c))
        walls.add((outer_r0 + outer_h - 1, c))
    for r in range(outer_r0 + 1, outer_r0 + outer_h - 1):
        walls.add((r, outer_c0))
        walls.add((r, outer_c0 + outer_w - 1))
    for r, c in walls:
        grid[r, c] = color

    r_lo = outer_r0 - 1
    c_lo = outer_c0 - 1
    r_hi = outer_r0 + outer_h + 1
    c_hi = outer_c0 + outer_w + 1
    reserved[r_lo:r_hi, c_lo:c_hi] = True

    interior = {
        (r, c)
        for r in range(outer_r0 + 1, outer_r0 + 1 + inner_h)
        for c in range(outer_c0 + 1, outer_c0 + 1 + inner_w)
    }
    return {
        "color": color,
        "dir": direction,
        "interior": interior,
        "walls": walls,
        "special": special,
        "inner_r0": outer_r0 + 1,
        "inner_c0": outer_c0 + 1,
        "inner_h": inner_h,
        "inner_w": inner_w,
    }


def _place_one_region(
    rng: random.Random,
    grid: np.ndarray,
    reserved: np.ndarray,
    color: int,
    direction: tuple[int, int],
    *,
    special: bool,
) -> dict | None:
    if special:
        side = rng.choice(SPECIAL_INTERIOR_SIZES)
        sizes = [(side, side)]
    else:
        sizes = [(rng.choice(PLAIN_INTERIOR_H), rng.choice(PLAIN_INTERIOR_W))]

    h, w = grid.shape
    for inner_h, inner_w in sizes:
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        for _ in range(PLACEMENT_RETRIES):
            outer_r0 = rng.randint(1, h - outer_h - 1)
            outer_c0 = rng.randint(1, w - outer_w - 1)
            if not _can_place_box(reserved, outer_r0, outer_c0, outer_h, outer_w):
                continue
            return _paint_rect_region(
                grid,
                reserved,
                outer_r0,
                outer_c0,
                inner_h,
                inner_w,
                color,
                direction,
                special=special,
            )
    return None


def _generate_regions(
    rng: random.Random,
    grid: np.ndarray,
    reserved: np.ndarray,
    legends: list[tuple[int, tuple[int, int]]],
) -> list[dict] | None:
    regions: list[dict] = []
    region_specs: list[tuple[int, tuple[int, int], bool]] = []
    for color, direction in legends:
        region_specs.append((color, direction, True))
        extra = rng.randint(REGIONS_PER_COLOR[0], REGIONS_PER_COLOR[1]) - 1
        for _ in range(extra):
            region_specs.append((color, direction, False))
    if len(region_specs) > MAX_TOTAL_REGIONS:
        specials = [spec for spec in region_specs if spec[2]]
        plains = [spec for spec in region_specs if not spec[2]]
        rng.shuffle(plains)
        region_specs = specials + plains[: max(0, MAX_TOTAL_REGIONS - len(specials))]
    rng.shuffle(region_specs)
    region_specs.sort(key=lambda spec: 0 if spec[2] else 1)

    for color, direction, special in region_specs:
        reg = _place_one_region(rng, grid, reserved, color, direction, special=special)
        if reg is None:
            return None
        regions.append(reg)
    return regions


def _validate_topology(grid: np.ndarray, regions: list[dict]) -> bool:
    h, w = grid.shape
    labeled, n_comp = label_mask(grid == 0, connectivity=4)
    bounded = []
    for comp_id in range(1, n_comp + 1):
        coords = np.argwhere(labeled == comp_id)
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in coords):
            continue
        bounded.append({(int(r), int(c)) for r, c in coords})
    expected = [reg["interior"] for reg in regions]
    if len(bounded) != len(expected):
        return False
    return all(comp in expected for comp in bounded)


def _special_pair_positions(reg: dict) -> list[tuple[int, int]]:
    r0 = reg["inner_r0"]
    c0 = reg["inner_c0"]
    h = reg["inner_h"]
    w = reg["inner_w"]
    dr, dc = reg["dir"]
    if (dr, dc) == (1, 1):
        return [(r0, c0), (r0, c0 + 1)]
    if (dr, dc) == (1, -1):
        return [(r0, c0 + w - 1), (r0, c0 + w - 2)]
    if (dr, dc) == (-1, 1):
        return [(r0 + h - 1, c0), (r0 + h - 1, c0 + 1)]
    if (dr, dc) == (-1, -1):
        return [(r0 + h - 1, c0 + w - 1), (r0 + h - 1, c0 + w - 2)]
    raise ValueError((dr, dc))


def _single_mover_candidates(reg: dict, occupied: set[tuple[int, int]]) -> list[tuple[int, int]]:
    dr, dc = reg["dir"]
    region_set = reg["interior"]
    out = []
    for r, c in sorted(region_set):
        if (r, c) in occupied:
            continue
        if _step_can_move(region_set, occupied | {(r, c)}, r, c, dr, dc):
            out.append((r, c))
    return out


def _place_yellows(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
    total = 0
    special_regions = [reg for reg in regions if reg["special"]]
    if not special_regions:
        return False

    for reg in special_regions:
        pts = _special_pair_positions(reg)
        new_pts, rounds = _iterate_until_stable(reg["interior"], pts, *reg["dir"])
        old_pts = _slide_points_old(reg["interior"], pts, *reg["dir"])
        if set(new_pts) == set(old_pts) or rounds < 2:
            return False
        for r, c in pts:
            grid[r, c] = 4
        total += len(pts)

    if total > 6:
        return False

    spare_regions = [reg for reg in regions if not reg["special"]]
    rng.shuffle(spare_regions)
    for reg in spare_regions:
        if total >= 6 or rng.random() < 0.45:
            continue
        candidates = _single_mover_candidates(reg, set())
        if not candidates:
            continue
        r, c = rng.choice(candidates)
        grid[r, c] = 4
        total += 1
    return total >= 2


def _variant_metrics(input_grid: np.ndarray) -> dict:
    g = np.asarray(input_grid, dtype=int)
    bg = get_bg(g)
    mover = 4
    dirs = _legend_directions(g, bg)
    infos = _collect_regions(g, bg, mover, dirs)
    region_rounds = []
    differing_regions = 0
    for info in infos:
        dr, dc = info["dir"]
        new_pts, rounds = _iterate_until_stable(info["region_set"], info["movers"], dr, dc)
        old_pts = _slide_points_old(info["region_set"], info["movers"], dr, dc)
        region_rounds.append(rounds)
        if set(new_pts) != set(old_pts):
            differing_regions += 1
    return {
        "region_rounds": region_rounds,
        "multi_round_regions": sum(1 for r in region_rounds if r >= 2),
        "total_rounds": sum(region_rounds),
        "differing_regions": differing_regions,
    }


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    h, w = input_grid.shape
    try:
        solver_out = main(input_grid)
    except Exception:
        return False

    struct_mask = (input_grid != 4) & (input_grid != 0)
    if not np.array_equal(solver_out[struct_mask], input_grid[struct_mask]):
        return False
    if int((solver_out == 4).sum()) != int((input_grid == 4).sum()):
        return False

    regions_mask = np.zeros((h, w), dtype=bool)
    for reg in regions:
        for r, c in reg["interior"]:
            regions_mask[r, c] = True
    if not bool(np.all(regions_mask[solver_out == 4])):
        return False

    metrics = _variant_metrics(input_grid)
    if metrics["multi_round_regions"] < 1:
        return False
    if metrics["differing_regions"] < 1:
        return False
    if metrics["total_rounds"] < 2:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    reserved, legends = _place_legends(rng, grid)
    regions = _generate_regions(rng, grid, reserved, legends)
    if regions is None:
        return None
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
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries")
