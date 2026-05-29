# Variant of b9e38dc0
# Dimension: D6 — 半包围精灵的形状结构 (obstacle)
# Change: 半包围区域由U形或C形边框围成，有一个明确的开口方向（上/下/左/右）... → 完全包围但有可破坏的薄壁

import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import components_mask, get_bg, neighbors4

# ---------------------------------------------------------------------------
# Design analysis for the D6 variant
#
# a) Concrete behavior:
#    - The old task used a partial enclosure whose endpoints visibly exposed the
#      radiate direction. This variant replaces that with a fully closed wall.
#    - The wall is now a rectangular shell around a cavity. Exactly one side of
#      the shell is 1 cell thick; the other three sides are 2 or 3 cells thick.
#    - A fill-colored seed sits inside the cavity, near the side opposite the
#      thin wall. In the output, fill first floods the cavity, then breaks
#      through the thin side and continues as a straight rectangular beam toward
#      the corresponding grid edge. Obstacles in that beam cast background
#      shadows behind themselves along the beam direction.
#    - Small trace example:
#      original: seed inside U-shape -> endpoints reveal "up" -> fill exits the
#      open top and spreads upward between the two arms.
#      variant: seed inside closed box -> no endpoints exist -> solver must
#      flood only the sealed cavity, measure wall thickness on all four sides,
#      find the unique thickness-1 side, replace that side's wall strip with
#      fill, then extend the beam from that side.
#
# b) Complexity calibration:
#    - The shape is not reduced to an arbitrary blob; it is still a single wall
#      object with clear inside/outside structure and a visible directional cue.
#    - Only four directional cases exist, matching the original cardinal
#      radiate choices. The new discriminating signal is a unique minimum wall
#      thickness, which appears as repeated evidence across the whole thin side.
#
# c) Hidden assumptions from the original solver that break here:
#    - Degree-1 endpoint detection no longer works because the wall is a loop.
#    - Endpoint-derived rays no longer define the exit direction.
#    - Assuming the wall is only a barrier is no longer enough; one side must be
#      converted from wall to fill in the output.
#    - The fill region cannot be inferred from a partially open shape anymore;
#      the solver must first recover the sealed interior via flood-fill from the
#      seed before thickness can be measured.
#
# d) Placement legality guaranteed by the generator:
#    - The shell is an axis-aligned rectangle ring, so it never self-overlaps.
#    - The thin side always has at least 3 cells of outside margin before the
#      relevant grid border, so external radiation is visible.
#    - Obstacles are placed only on background cells outside the shell and at
#      Manhattan distance > 1 from the wall, so they never steal the "closest
#      non-wall color" role from the fill seed.
#
# e) Visual observability:
#    - Humans can infer the direction by comparing visible wall thicknesses.
#      The only 1-cell-thick side is the side that gets breached.
#    - Across samples, the thin side varies over the four cardinal directions so
#      the rule is not overfit to one placement.
#
# f) Visual cleanliness:
#    - The output has only three active layers in the interesting region:
#      preserved wall, fill region, and sparse obstacle pixels with shadows.
#    - Obstacles are kept outside the cavity to avoid mixing "what fills the
#      room" with "what blocks the beam".
#
# g) Color inferability:
#    - No color-id ordering is used. The fill color is still selected by spatial
#      proximity to the wall, and obstacle colors remain arbitrary preserved
#      singleton colors.
# ---------------------------------------------------------------------------

# -- Configuration constants -------------------------------------------------
GRID_H_VERT = (11, 18)
GRID_H_HORIZ = (8, 18)
GRID_W = (15, 19)

INNER_H_VERT = (3, 6)
INNER_W_VERT = (5, 8)
INNER_H_HORIZ = (4, 7)
INNER_W_HORIZ = (3, 6)

OPEN_END_MARGIN = 3
FAR_SIDE_MARGIN = 2
SIDE_MARGIN = 1

THICK_OTHER = (2, 3)

SEED_SIZE_WEIGHTS = [1, 1, 1, 1, 2, 2, 3, 4, 5, 6]
SEED_DEPTH_LOOKBACK = 2

MAX_OBSTACLES = 5
OBSTACLE_IN_FILL_PROB = 0.65
OBSTACLE_MIN_SEP = 2

MIN_NEW_FILL = 20
PALETTE_SIZE = (3, 5)
GEN_RETRIES = 300


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _seed_fill(mask: np.ndarray, starts: list[tuple[int, int]]) -> np.ndarray:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    q = []
    for r, c in starts:
        if 0 <= r < h and 0 <= c < w and mask[r, c] and not seen[r, c]:
            seen[r, c] = True
            q.append((r, c))
    while q:
        r, c = q.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return seen


def _trim_dead_ends(points: set[tuple[int, int]]) -> set[tuple[int, int]]:
    pts = set(points)
    changed = True
    while changed:
        changed = False
        drop = {
            (r, c)
            for r, c in pts
            if sum((rr, cc) in pts for rr, cc in neighbors4(r, c)) <= 1
        }
        if drop and len(drop) < len(pts):
            pts -= drop
            changed = True
    return pts


def _min_dist_to_wall(grid: np.ndarray, color: int, wall_points: list[tuple[int, int]]) -> tuple[int, int]:
    pts = [(int(r), int(c)) for r, c in zip(*np.where(grid == color))]
    best_d = 10**9
    for r, c in pts:
        for wr, wc in wall_points:
            d = abs(r - wr) + abs(c - wc)
            if d < best_d:
                best_d = d
                if best_d == 0:
                    return 0, len(pts)
    return best_d, len(pts)


def _side_samples(interior: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    rows, cols = np.where(interior)
    top = int(rows.min())
    bottom = int(rows.max())
    left = int(cols.min())
    right = int(cols.max())
    return {
        (-1, 0): [(top, c) for c in range(left, right + 1) if interior[top, c]],
        (1, 0): [(bottom, c) for c in range(left, right + 1) if interior[bottom, c]],
        (0, -1): [(r, left) for r in range(top, bottom + 1) if interior[r, left]],
        (0, 1): [(r, right) for r in range(top, bottom + 1) if interior[r, right]],
    }


def _count_wall_run(comp_mask: np.ndarray, start: tuple[int, int], d: tuple[int, int]) -> int:
    h, w = comp_mask.shape
    r, c = start
    rr, cc = r + d[0], c + d[1]
    count = 0
    while 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]:
        count += 1
        rr += d[0]
        cc += d[1]
    return count


def _measure_side_thicknesses(comp_mask: np.ndarray, interior: np.ndarray) -> dict[tuple[int, int], int]:
    samples = _side_samples(interior)
    thicknesses = {}
    for d, pts in samples.items():
        runs = [_count_wall_run(comp_mask, p, d) for p in pts]
        assert runs and min(runs) > 0, "expected interior side to face wall"
        thicknesses[d] = min(runs)
    return thicknesses


def _choose_radiate_dir(thicknesses: dict[tuple[int, int], int]) -> tuple[int, int]:
    items = sorted(thicknesses.items(), key=lambda kv: kv[1])
    assert len(items) == 4, "expected four side thicknesses"
    assert items[0][1] < items[1][1], "expected a unique thinnest wall side"
    return items[0][0]


def _build_breach_and_beam(
    comp_mask: np.ndarray,
    interior: np.ndarray,
    rad: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = comp_mask.shape
    breach = np.zeros_like(comp_mask, dtype=bool)
    beam = np.zeros_like(comp_mask, dtype=bool)
    rows, cols = np.where(interior)
    top = int(rows.min())
    bottom = int(rows.max())
    left = int(cols.min())
    right = int(cols.max())

    if rad == (-1, 0):
        for c in range(left, right + 1):
            if not interior[top, c]:
                continue
            rr = top - 1
            while rr >= 0 and comp_mask[rr, c]:
                breach[rr, c] = True
                rr -= 1
        beam[:top, left : right + 1] = True
    elif rad == (1, 0):
        for c in range(left, right + 1):
            if not interior[bottom, c]:
                continue
            rr = bottom + 1
            while rr < h and comp_mask[rr, c]:
                breach[rr, c] = True
                rr += 1
        beam[bottom + 1 :, left : right + 1] = True
    elif rad == (0, -1):
        for r in range(top, bottom + 1):
            if not interior[r, left]:
                continue
            cc = left - 1
            while cc >= 0 and comp_mask[r, cc]:
                breach[r, cc] = True
                cc -= 1
        beam[top : bottom + 1, :left] = True
    else:
        for r in range(top, bottom + 1):
            if not interior[r, right]:
                continue
            cc = right + 1
            while cc < w and comp_mask[r, cc]:
                breach[r, cc] = True
                cc += 1
        beam[top : bottom + 1, right + 1 :] = True

    return breach, beam


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, copy=False)
    h, w = g.shape
    bg = get_bg(g)
    vals = [int(v) for v in np.unique(g) if int(v) != bg]
    assert vals, "expected at least one non-background color"

    wall = max(vals, key=lambda c: int((g == c).sum()))
    wall_mask = g == wall
    comps = [set((int(r), int(c)) for r, c in pts) for pts in components_mask(wall_mask, connectivity=8)]
    assert comps, "expected at least one wall connected component"

    other_colors = [c for c in vals if c != wall]
    assert other_colors, "expected at least one non-wall color"
    wall_points = [p for comp in comps for p in comp]
    fill = min(other_colors, key=lambda c: _min_dist_to_wall(g, c, wall_points))
    fill_pos = [(int(r), int(c)) for r, c in zip(*np.where(g == fill))]
    assert fill_pos, "expected at least one fill seed pixel"

    def comp_min_dist(comp: set[tuple[int, int]]) -> int:
        best = 10**9
        for r, c in fill_pos:
            for wr, wc in comp:
                d = abs(r - wr) + abs(c - wc)
                if d < best:
                    best = d
                    if best == 0:
                        return 0
        return best

    wall_comp = min(comps, key=lambda comp: (comp_min_dist(comp), -len(comp)))
    wall_comp = _trim_dead_ends(wall_comp)
    assert wall_comp, "expected a closed core wall after trimming auxiliaries"
    comp_mask = np.zeros_like(wall_mask, dtype=bool)
    for r, c in wall_comp:
        comp_mask[r, c] = True

    interior_passable = (~comp_mask) & ((g == bg) | (g == fill))
    interior = _seed_fill(interior_passable, fill_pos)
    assert interior.any(), "expected fill seed to reach an enclosed interior"
    assert not _touches_border(interior), "seed region leaked to the border; wall must stay sealed in input"

    thicknesses = _measure_side_thicknesses(comp_mask, interior)
    rad = _choose_radiate_dir(thicknesses)
    breach, beam = _build_breach_and_beam(comp_mask, interior, rad)
    assert breach.any(), "expected a thin wall segment to breach"

    rows, cols = np.where(interior)
    bbox = (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))
    return {
        "bg": bg,
        "wall": wall,
        "fill": fill,
        "fill_pos": fill_pos,
        "comp_mask": comp_mask,
        "interior": interior,
        "thicknesses": thicknesses,
        "rad": rad,
        "breach": breach,
        "beam": beam,
        "interior_bbox": bbox,
        "shape": (h, w),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the sealed chamber and burst outward through its uniquely thinnest wall.

    Input structure:
    - bg is the most common color. The wall is the most frequent non-bg color and forms one closed shell around a cavity.
    - A fill-color seed lies inside that cavity. One wall side is exactly 1 pixel thick; the other three sides are thicker. Other non-bg pixels are obstacles.
    Rule:
    - Flood-fill the chamber interior with the fill color, then break the thin wall and continue filling straight outward from that side to the grid edge.
    - Obstacles stay fixed and cast background shadows behind themselves along the same outward direction.

    Key insight: Because the wall is fully closed, the exit direction is no longer visible as an opening; it must be inferred by comparing the four wall-side thicknesses around the enclosed fill region.
    """
    g = np.array(grid, copy=True)
    info = _analyze_scene(g)
    bg = info["bg"]
    wall = info["wall"]
    fill = info["fill"]
    fill_pos = info["fill_pos"]
    interior = info["interior"]
    rad = info["rad"]
    breach = info["breach"]
    beam = info["beam"]

    domain = interior | beam | breach
    passable = domain & ((g == bg) | (g == fill) | breach)
    seen = _seed_fill(passable, fill_pos)

    write_mask = seen & ((g == bg) | breach)
    out = g.copy()
    out[write_mask] = fill

    filled_mask = write_mask.copy()
    obstacles = (g != bg) & (g != fill) & (g != wall)

    def shadow_process(coords: list[tuple[int, int]]) -> None:
        started = False
        blocked = False
        for r, c in coords:
            if filled_mask[r, c]:
                started = True
            if started and obstacles[r, c]:
                blocked = True
                continue
            if blocked and filled_mask[r, c]:
                out[r, c] = bg
                filled_mask[r, c] = False

    dr, dc = rad
    if dr != 0:
        for c in range(g.shape[1]):
            rows = range(g.shape[0]) if dr > 0 else range(g.shape[0] - 1, -1, -1)
            shadow_process([(r, c) for r in rows])
    else:
        for r in range(g.shape[0]):
            cols = range(g.shape[1]) if dc > 0 else range(g.shape[1] - 1, -1, -1)
            shadow_process([(r, c) for c in cols])

    return out


def _grow_seed_blob(seed: tuple[int, int], interior_cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    blob = {seed}
    frontier = [seed]
    target_size = random.choice(SEED_SIZE_WEIGHTS)
    while len(blob) < target_size and frontier:
        r, c = frontier.pop(random.randrange(len(frontier)))
        nbrs = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        for q in random.sample(nbrs, len(nbrs)):
            if q in interior_cells and q not in blob:
                blob.add(q)
                frontier.append(q)
                if len(blob) >= target_size:
                    break
    return blob


def _obstacle_fits(pos: tuple[int, int], placed: list[tuple[int, int]]) -> bool:
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) >= OBSTACLE_MIN_SEP for rr, cc in placed)


def _choose_thicknesses(rad: tuple[int, int]) -> dict[str, int]:
    top = random.randint(*THICK_OTHER)
    bottom = random.randint(*THICK_OTHER)
    left = random.randint(*THICK_OTHER)
    right = random.randint(*THICK_OTHER)
    if rad == (-1, 0):
        top = 1
    elif rad == (1, 0):
        bottom = 1
    elif rad == (0, -1):
        left = 1
    else:
        right = 1
    return {"top": top, "bottom": bottom, "left": left, "right": right}


def _choose_inner_dims(rad: tuple[int, int], h: int, w: int, t: dict[str, int]) -> tuple[int, int] | None:
    if rad[0]:
        hs = list(range(INNER_H_VERT[0], INNER_H_VERT[1] + 1))
        ws = list(range(INNER_W_VERT[0], INNER_W_VERT[1] + 1))
    else:
        hs = list(range(INNER_H_HORIZ[0], INNER_H_HORIZ[1] + 1))
        ws = list(range(INNER_W_HORIZ[0], INNER_W_HORIZ[1] + 1))
    random.shuffle(hs)
    random.shuffle(ws)
    for inner_h in hs:
        for inner_w in ws:
            outer_h = t["top"] + inner_h + t["bottom"]
            outer_w = t["left"] + inner_w + t["right"]
            if outer_h + FAR_SIDE_MARGIN <= h and outer_w + SIDE_MARGIN * 2 <= w:
                if rad[0]:
                    if outer_h + OPEN_END_MARGIN + FAR_SIDE_MARGIN <= h:
                        return inner_h, inner_w
                else:
                    if outer_w + OPEN_END_MARGIN + FAR_SIDE_MARGIN <= w:
                        return inner_h, inner_w
    return None


def _place_outer_box(
    rad: tuple[int, int],
    h: int,
    w: int,
    outer_h: int,
    outer_w: int,
) -> tuple[int, int]:
    if rad == (-1, 0):
        r0 = random.randint(OPEN_END_MARGIN, h - outer_h - FAR_SIDE_MARGIN)
        c0 = random.randint(SIDE_MARGIN, w - outer_w - SIDE_MARGIN)
    elif rad == (1, 0):
        r0 = random.randint(FAR_SIDE_MARGIN, h - outer_h - OPEN_END_MARGIN)
        c0 = random.randint(SIDE_MARGIN, w - outer_w - SIDE_MARGIN)
    elif rad == (0, -1):
        r0 = random.randint(SIDE_MARGIN, h - outer_h - SIDE_MARGIN)
        c0 = random.randint(OPEN_END_MARGIN, w - outer_w - FAR_SIDE_MARGIN)
    else:
        r0 = random.randint(SIDE_MARGIN, h - outer_h - SIDE_MARGIN)
        c0 = random.randint(FAR_SIDE_MARGIN, w - outer_w - OPEN_END_MARGIN)
    return r0, c0


def _render_shell(
    grid: np.ndarray,
    rad: tuple[int, int],
    r0: int,
    c0: int,
    inner_h: int,
    inner_w: int,
    t: dict[str, int],
    wall_color: int,
) -> tuple[np.ndarray, np.ndarray]:
    outer_h = t["top"] + inner_h + t["bottom"]
    outer_w = t["left"] + inner_w + t["right"]
    grid[r0 : r0 + outer_h, c0 : c0 + outer_w] = wall_color

    ir0 = r0 + t["top"]
    ic0 = c0 + t["left"]
    interior = np.zeros_like(grid, dtype=bool)
    interior[ir0 : ir0 + inner_h, ic0 : ic0 + inner_w] = True
    grid[interior] = get_bg(grid)

    wall_mask = np.zeros_like(grid, dtype=bool)
    wall_mask[r0 : r0 + outer_h, c0 : c0 + outer_w] = True
    wall_mask[interior] = False

    mid_r = ir0 + inner_h // 2
    mid_c = ic0 + inner_w // 2
    if rad == (-1, 0):
        spur_cells = [(r0 - 1, mid_c), (r0 - 2, mid_c)]
    elif rad == (1, 0):
        spur_cells = [(r0 + outer_h, mid_c), (r0 + outer_h + 1, mid_c)]
    elif rad == (0, -1):
        spur_cells = [(mid_r, c0 - 1), (mid_r, c0 - 2)]
    else:
        spur_cells = [(mid_r, c0 + outer_w), (mid_r, c0 + outer_w + 1)]
    for sr, sc in spur_cells:
        grid[sr, sc] = wall_color
        wall_mask[sr, sc] = True
    return wall_mask, interior


def _seed_pool(interior: np.ndarray, rad: tuple[int, int]) -> list[tuple[int, int]]:
    rows, cols = np.where(interior)
    top = int(rows.min())
    bottom = int(rows.max())
    left = int(cols.min())
    right = int(cols.max())
    pool = []
    for r, c in zip(rows, cols):
        r = int(r)
        c = int(c)
        if rad == (-1, 0):
            if r >= bottom - SEED_DEPTH_LOOKBACK:
                pool.append((r, c))
        elif rad == (1, 0):
            if r <= top + SEED_DEPTH_LOOKBACK:
                pool.append((r, c))
        elif rad == (0, -1):
            if c >= right - SEED_DEPTH_LOOKBACK:
                pool.append((r, c))
        else:
            if c <= left + SEED_DEPTH_LOOKBACK:
                pool.append((r, c))
    return pool


def _find_obstacle_candidates(
    grid: np.ndarray,
    bg: int,
    wall_color: int,
    nominal_fill_zone: np.ndarray,
    exterior_bg: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    wall_mask = grid == wall_color
    dist = distance_transform_cdt(~wall_mask, metric="taxicab")
    eligible = exterior_bg & (grid == bg) & (dist > 1)
    all_coords = [(int(r), int(c)) for r, c in zip(*np.where(eligible))]
    fill_coords = [p for p in all_coords if nominal_fill_zone[p[0], p[1]]]
    random.shuffle(all_coords)
    random.shuffle(fill_coords)
    return all_coords, fill_coords


def _place_obstacles(
    n_obstacles: int,
    candidates: list[tuple[int, int]],
    fill_candidates: list[tuple[int, int]],
    obstacle_colors: list[int],
    grid: np.ndarray,
) -> None:
    placed: list[tuple[int, int]] = []
    n_fill = random.randint(0, min(n_obstacles, len(fill_candidates)))
    if fill_candidates and random.random() < OBSTACLE_IN_FILL_PROB:
        n_fill = max(1, n_fill)
    for p in fill_candidates:
        if len(placed) >= n_fill:
            break
        if _obstacle_fits(p, placed):
            placed.append(p)
    placed_set = set(placed)
    for p in candidates:
        if len(placed) >= n_obstacles:
            break
        if p not in placed_set and _obstacle_fits(p, placed):
            placed.append(p)
            placed_set.add(p)
    for i, (r, c) in enumerate(placed):
        grid[r, c] = obstacle_colors[i % len(obstacle_colors)]


def _gen_attempt() -> np.ndarray | None:
    rad = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    h = random.randint(*GRID_H_VERT) if rad[0] else random.randint(*GRID_H_HORIZ)
    w = random.randint(*GRID_W)

    t = _choose_thicknesses(rad)
    dims = _choose_inner_dims(rad, h, w, t)
    if dims is None:
        return None
    inner_h, inner_w = dims
    outer_h = t["top"] + inner_h + t["bottom"]
    outer_w = t["left"] + inner_w + t["right"]
    r0, c0 = _place_outer_box(rad, h, w, outer_h, outer_w)

    palette = random.sample(range(10), random.randint(*PALETTE_SIZE))
    bg, wall_color, fill_color = palette[:3]
    obstacle_colors = palette[3:]

    grid = np.full((h, w), bg, dtype=int)
    wall_mask, interior = _render_shell(grid, rad, r0, c0, inner_h, inner_w, t, wall_color)

    pool = _seed_pool(interior, rad)
    if not pool:
        return None
    interior_cells = {tuple(map(int, p)) for p in zip(*np.where(interior))}
    blob = _grow_seed_blob(random.choice(pool), interior_cells)
    for r, c in blob:
        if grid[r, c] != bg:
            return None
        grid[r, c] = fill_color

    try:
        nominal_out = main(grid)
    except Exception:
        return None

    nominal_fill_zone = (nominal_out == fill_color) & (grid == bg)
    exterior_bg = (~interior) & (~wall_mask) & (grid == bg)

    n_obstacles = random.randint(0, MAX_OBSTACLES)
    if n_obstacles and not obstacle_colors:
        available = [c for c in range(10) if c not in (bg, wall_color, fill_color)]
        obstacle_colors = random.sample(available, random.randint(1, 2))
    if n_obstacles:
        candidates, fill_candidates = _find_obstacle_candidates(
            grid, bg, wall_color, nominal_fill_zone, exterior_bg
        )
        _place_obstacles(n_obstacles, candidates, fill_candidates, obstacle_colors, grid)

    try:
        out = main(grid)
    except Exception:
        return None
    new_fill = (out == fill_color) & (grid == bg)
    if int(new_fill.sum()) < MIN_NEW_FILL or not _touches_border(new_fill):
        return None

    info = _analyze_scene(grid)
    if info["thicknesses"][info["rad"]] != 1:
        return None
    other = [v for d, v in info["thicknesses"].items() if d != info["rad"]]
    if not all(v >= 2 for v in other):
        return None
    if int((new_fill & info["beam"]).sum()) < 8:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 D6 variant input")
