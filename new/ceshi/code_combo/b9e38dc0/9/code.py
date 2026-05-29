# Combo variant of b9e38dc0 (index 9)
# combo_spec: D6/3+D8/2
# Name: 薄壁突破+边框延伸管道

import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import components_mask, get_bg, neighbors4

# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete behavior and trace:
#    - Base task: an open funnel already reveals its radiate direction through
#      two endpoints, and the outgoing corridor is bounded by invisible rays.
#    - D6/3 replaces that visible opening with a sealed shell. The solver must
#      first flood the closed cavity, then compare the four wall-side
#      thicknesses and break through the unique 1-cell-thick side.
#    - D8/2 makes the previously implicit corridor boundary visible as wall.
#    - Combined behavior here:
#      1. Input shows a closed rectangular shell with one thin side and three
#         thicker sides, plus a seed inside and sparse exterior obstacles.
#      2. Output floods the interior, recolors the thin side into a doorway,
#         sends a straight beam to the grid edge, and extends the thick side
#         walls along that same direction so the beam sits inside a pipe.
#      3. Obstacles inside the beam erase fill behind themselves, but the pipe
#         walls remain.
#    - Small example: if the top wall is the only thickness-1 side, the seed
#      fills the chamber, the top strip becomes fill, rows above the chamber
#      fill within the interior width, and the left/right wall bands continue
#      upward as wall-colored rails.
#
# b) Complexity calibration:
#    - The wall family is restricted to one axis-aligned shell with exactly four
#      candidate sides and a unique thinnest side. This keeps inference finite.
#    - The pipe uses the same side-wall thicknesses already visible in the
#      input, so the new transform is strong visually without adding a second
#      unrelated geometry system.
#    - Cardinal radiate directions are the only directional cases. Each sample
#      repeats the crucial signal twice: thin-side comparison inside the shell
#      and thick-wall continuation outside it.
#
# c) Hidden assumptions that changed:
#    - Endpoint detection from the original solver is invalid because the wall
#      is now a loop, not a degree-1 chain.
#    - The old invisible-ray barrier is insufficient: the solver must explicitly
#      recover the enclosed interior, measure side thicknesses, and synthesize
#      both the breach mask and the future pipe-wall mask.
#    - Obstacles cannot be allowed on future pipe walls, otherwise the wall
#      extension would overwrite them ambiguously. The generator forbids that.
#    - Fill-color detection still relies on "closest non-wall color to the
#      wall", so obstacle placement stays at taxicab distance > 1 from the wall.
#
# d) Placement legality:
#    - The generator builds one rectangular shell from an outer box minus an
#      inner box, guaranteeing a single closed component with no overlaps.
#    - Margins ensure the thin side has visible runway to the relevant grid edge
#      and the opposite side has room for the seed.
#    - Future pipe walls and obstacle candidates are separated before coloring.
#
# e) Visual observability:
#    - Humans can compare wall thicknesses directly in the input. The only
#      1-cell side is always the exit side.
#    - The output adds a conspicuous second cue: the thicker perpendicular walls
#      continue outward as a pipe. Across examples, both the chosen side and the
#      carried wall thickness vary, making the fusion rule visible from samples.
#
# f) Visual cleanliness:
#    - Only three active layers share the interesting area: wall, fill, and
#      sparse obstacle/shadow holes. Pipe walls occupy dedicated border strips,
#      so fill and wall are not mixed in the same cells.
#    - Obstacles live on exterior background cells only; they never clutter the
#      sealed cavity or the wall itself.
#
# g) Inferability of changes:
#    - No color-id ordering is used. Colors are assigned arbitrarily; roles come
#      from frequency and spatial position.
#    - The fill color changes only by structural reachability, and the wall
#      extension uses the same wall color, so no extra color mapping rule must
#      be learned.
# ---------------------------------------------------------------------------

# Canvas
GRID_H_VERT = (11, 18)
GRID_H_HORIZ = (8, 18)
GRID_W = (15, 19)

# Inner chamber size
INNER_H_VERT = (3, 6)
INNER_W_VERT = (5, 8)
INNER_H_HORIZ = (4, 7)
INNER_W_HORIZ = (3, 6)

# Placement margins
OPEN_END_MARGIN = 3
FAR_SIDE_MARGIN = 2
SIDE_MARGIN = 1

# Wall thickness
THICK_OTHER = (2, 3)

# Fill seed
SEED_SIZE_WEIGHTS = [1, 1, 1, 1, 2, 2, 3, 4, 5, 6]
SEED_DEPTH_LOOKBACK = 2

# Obstacles
MAX_OBSTACLES = 5
OBSTACLE_IN_FILL_PROB = 0.65
OBSTACLE_MIN_SEP = 2

# Validation
MIN_NEW_FILL = 20
MIN_PIPE_WALL_CELLS = 6
MIN_PIPE_LENGTH = 3
PALETTE_SIZE = (3, 5)
GEN_RETRIES = 400


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _seed_fill(mask: np.ndarray, starts: list[tuple[int, int]]) -> np.ndarray:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    q: list[tuple[int, int]] = []
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


def _min_dist_to_wall(
    grid: np.ndarray,
    color: int,
    wall_points: list[tuple[int, int]],
) -> tuple[int, int]:
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
    rr, cc = start[0] + d[0], start[1] + d[1]
    count = 0
    while 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]:
        count += 1
        rr += d[0]
        cc += d[1]
    return count


def _measure_side_thicknesses(comp_mask: np.ndarray, interior: np.ndarray) -> dict[tuple[int, int], int]:
    samples = _side_samples(interior)
    thicknesses: dict[tuple[int, int], int] = {}
    for d, pts in samples.items():
        runs = [_count_wall_run(comp_mask, p, d) for p in pts]
        assert runs and min(runs) > 0, "expected each interior side to face wall"
        thicknesses[d] = min(runs)
    return thicknesses


def _choose_radiate_dir(thicknesses: dict[tuple[int, int], int]) -> tuple[int, int]:
    items = sorted(thicknesses.items(), key=lambda kv: kv[1])
    assert len(items) == 4, "expected four side thicknesses"
    assert items[0][1] == 1, "expected breach side thickness to be 1"
    assert items[0][1] < items[1][1], "expected a unique thinnest side"
    return items[0][0]


def _build_geometry(
    comp_mask: np.ndarray,
    interior: np.ndarray,
    rad: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    breach = np.zeros_like(comp_mask, dtype=bool)
    beam = np.zeros_like(comp_mask, dtype=bool)
    pipe_wall = np.zeros_like(comp_mask, dtype=bool)

    irows, icols = np.where(interior)
    wrows, wcols = np.where(comp_mask)
    top = int(irows.min())
    bottom = int(irows.max())
    left = int(icols.min())
    right = int(icols.max())
    outer_top = int(wrows.min())
    outer_bottom = int(wrows.max())
    outer_left = int(wcols.min())
    outer_right = int(wcols.max())

    if rad == (-1, 0):
        breach[outer_top:top, left : right + 1] = comp_mask[outer_top:top, left : right + 1]
        beam[:top, left : right + 1] = True
        pipe_wall[:outer_top, outer_left:left] = True
        pipe_wall[:outer_top, right + 1 : outer_right + 1] = True
    elif rad == (1, 0):
        breach[bottom + 1 : outer_bottom + 1, left : right + 1] = comp_mask[
            bottom + 1 : outer_bottom + 1, left : right + 1
        ]
        beam[bottom + 1 :, left : right + 1] = True
        pipe_wall[outer_bottom + 1 :, outer_left:left] = True
        pipe_wall[outer_bottom + 1 :, right + 1 : outer_right + 1] = True
    elif rad == (0, -1):
        breach[top : bottom + 1, outer_left:left] = comp_mask[top : bottom + 1, outer_left:left]
        beam[top : bottom + 1, :left] = True
        pipe_wall[outer_top:top, :outer_left] = True
        pipe_wall[bottom + 1 : outer_bottom + 1, :outer_left] = True
    else:
        breach[top : bottom + 1, right + 1 : outer_right + 1] = comp_mask[
            top : bottom + 1, right + 1 : outer_right + 1
        ]
        beam[top : bottom + 1, right + 1 :] = True
        pipe_wall[outer_top:top, outer_right + 1 :] = True
        pipe_wall[bottom + 1 : outer_bottom + 1, outer_right + 1 :] = True

    pipe_wall &= ~comp_mask
    pipe_wall &= ~breach
    return breach, beam, pipe_wall


def _analyze_scene(grid: np.ndarray) -> dict:
    # Solver assumptions audited against the generator:
    # - The chosen wall component is one sealed shell nearest to the seed blob.
    # - The seed-reachable non-wall area is exactly the chamber interior and does
    #   not touch the border before the breach is synthesized.
    # - Every chamber side faces a contiguous wall run, so side thickness can be
    #   measured by straight scans normal to the interior bbox.
    # - The uniquely thin side has thickness 1, which lets the solver recolor a
    #   doorway and infer the outward pipe direction from the same measurement.
    g = np.array(grid, copy=False)
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
    comp_mask = np.zeros_like(wall_mask, dtype=bool)
    for r, c in wall_comp:
        comp_mask[r, c] = True

    passable_inside = (~comp_mask) & ((g == bg) | (g == fill))
    interior = _seed_fill(passable_inside, fill_pos)
    assert interior.any(), "expected enclosed interior reachable from the seed"
    assert not _touches_border(interior), "closed shell leaked to border"

    thicknesses = _measure_side_thicknesses(comp_mask, interior)
    rad = _choose_radiate_dir(thicknesses)
    breach, beam, pipe_wall = _build_geometry(comp_mask, interior, rad)
    assert breach.any(), "expected a non-empty breach strip"

    irows, icols = np.where(interior)
    wrows, wcols = np.where(comp_mask)
    return {
        "grid": g,
        "bg": int(bg),
        "wall": int(wall),
        "fill": int(fill),
        "fill_pos": fill_pos,
        "comp_mask": comp_mask,
        "interior": interior,
        "thicknesses": thicknesses,
        "rad": rad,
        "breach": breach,
        "beam": beam,
        "pipe_wall": pipe_wall,
        "interior_bbox": (
            int(irows.min()),
            int(icols.min()),
            int(irows.max()),
            int(icols.max()),
        ),
        "wall_bbox": (
            int(wrows.min()),
            int(wcols.min()),
            int(wrows.max()),
            int(wcols.max()),
        ),
    }


def _variant_metrics(grid: np.ndarray) -> dict:
    info = _analyze_scene(grid)
    rad = info["rad"]
    h, w = grid.shape
    beam_only = info["beam"] & (~info["breach"])
    if rad == (-1, 0):
        pipe_len = info["wall_bbox"][0]
    elif rad == (1, 0):
        pipe_len = h - 1 - info["wall_bbox"][2]
    elif rad == (0, -1):
        pipe_len = info["wall_bbox"][1]
    else:
        pipe_len = w - 1 - info["wall_bbox"][3]
    return {
        "rad": rad,
        "thin_thickness": info["thicknesses"][rad],
        "other_thicknesses": tuple(
            info["thicknesses"][d] for d in [(-1, 0), (1, 0), (0, -1), (0, 1)] if d != rad
        ),
        "pipe_length": int(pipe_len),
        "pipe_wall_cells": int(info["pipe_wall"].sum()),
        "beam_cells": int(beam_only.sum()),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a sealed chamber, punch through its thinnest wall, and extend the surrounding frame into a pipe.

    Input structure:
    - The background is the dominant color, and the most common non-background color is one closed wall shell around a cavity.
    - A smaller non-wall color blob sits inside that cavity. Exactly one wall side is 1 cell thick; the other three sides are thicker. Any remaining non-background cells are sparse exterior obstacles.

    Rule:
    - Flood-fill the chamber interior with the seed color, turn the uniquely thin wall side into a breach, and continue the fill straight outward to the corresponding grid edge.
    - At the same time, extend the two thicker border bands that flank that exit direction so they become visible pipe walls around the outgoing beam. Obstacles remain fixed and carve background shadows behind themselves along the beam direction.

    Key insight:
    - The exit direction is no longer exposed by an opening. It must be inferred by comparing wall thicknesses around the enclosed seed region, and that same choice also determines which border strips are prolonged into the pipe.
    """
    info = _analyze_scene(grid)
    g = np.array(info["grid"], copy=True)
    bg = info["bg"]
    wall = info["wall"]
    fill = info["fill"]
    fill_pos = info["fill_pos"]

    domain = info["interior"] | info["beam"] | info["breach"]
    passable = domain & ((g == bg) | (g == fill) | info["breach"])
    seen = _seed_fill(passable, fill_pos)

    out = g.copy()
    write_mask = seen & ((g == bg) | info["breach"])
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

    dr, dc = info["rad"]
    if dr != 0:
        for c in range(g.shape[1]):
            rows = range(g.shape[0]) if dr > 0 else range(g.shape[0] - 1, -1, -1)
            shadow_process([(r, c) for r in rows])
    else:
        for r in range(g.shape[0]):
            cols = range(g.shape[1]) if dc > 0 else range(g.shape[1] - 1, -1, -1)
            shadow_process([(r, c) for c in cols])

    out[info["pipe_wall"]] = wall
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
            if outer_h + SIDE_MARGIN * 2 > h or outer_w + SIDE_MARGIN * 2 > w:
                continue
            if rad == (-1, 0):
                if outer_h + OPEN_END_MARGIN + FAR_SIDE_MARGIN <= h:
                    return inner_h, inner_w
            elif rad == (1, 0):
                if outer_h + OPEN_END_MARGIN + FAR_SIDE_MARGIN <= h:
                    return inner_h, inner_w
            elif rad == (0, -1):
                if outer_w + OPEN_END_MARGIN + FAR_SIDE_MARGIN <= w:
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
    bg: int,
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
    grid[interior] = bg

    wall_mask = np.zeros_like(grid, dtype=bool)
    wall_mask[r0 : r0 + outer_h, c0 : c0 + outer_w] = True
    wall_mask[interior] = False
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
        if rad == (-1, 0) and r >= bottom - SEED_DEPTH_LOOKBACK:
            pool.append((r, c))
        elif rad == (1, 0) and r <= top + SEED_DEPTH_LOOKBACK:
            pool.append((r, c))
        elif rad == (0, -1) and c >= right - SEED_DEPTH_LOOKBACK:
            pool.append((r, c))
        elif rad == (0, 1) and c <= left + SEED_DEPTH_LOOKBACK:
            pool.append((r, c))
    return pool


def _find_obstacle_candidates(
    grid: np.ndarray,
    bg: int,
    wall_color: int,
    nominal_fill_zone: np.ndarray,
    forbidden_mask: np.ndarray,
    interior: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    wall_mask = grid == wall_color
    dist = distance_transform_cdt(~wall_mask, metric="taxicab")
    eligible = (grid == bg) & (dist > 1) & (~forbidden_mask) & (~interior)
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
    wall_mask, interior = _render_shell(grid, bg, r0, c0, inner_h, inner_w, t, wall_color)

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
        layout = _analyze_scene(grid)
        nominal_out = main(grid)
    except Exception:
        return None

    nominal_fill_zone = (nominal_out == fill_color) & (grid == bg)
    forbidden = layout["pipe_wall"] | layout["breach"]

    n_obstacles = random.randint(0, MAX_OBSTACLES)
    if n_obstacles and not obstacle_colors:
        available = [c for c in range(10) if c not in (bg, wall_color, fill_color)]
        obstacle_colors = random.sample(available, random.randint(1, 2))
    if n_obstacles:
        candidates, fill_candidates = _find_obstacle_candidates(
            grid,
            bg,
            wall_color,
            nominal_fill_zone,
            forbidden,
            interior,
        )
        _place_obstacles(n_obstacles, candidates, fill_candidates, obstacle_colors, grid)

    try:
        layout = _analyze_scene(grid)
        out = main(grid)
        metrics = _variant_metrics(grid)
    except Exception:
        return None

    new_fill = (out == fill_color) & (grid == bg)
    new_wall = (out == wall_color) & (grid == bg)
    if int(new_fill.sum()) < MIN_NEW_FILL or not _touches_border(new_fill):
        return None
    if metrics["thin_thickness"] != 1 or any(v < 2 for v in metrics["other_thicknesses"]):
        return None
    if metrics["pipe_length"] < MIN_PIPE_LENGTH or metrics["pipe_wall_cells"] < MIN_PIPE_WALL_CELLS:
        return None
    if int((new_fill & layout["beam"]).sum()) < 8:
        return None
    if int((new_wall & layout["pipe_wall"]).sum()) != int(layout["pipe_wall"].sum()):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 combo input")
