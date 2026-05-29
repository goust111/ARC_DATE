# Combo variant of b9e38dc0 (index 33)
# combo_spec: D1/2+D6/3+D8/2
# Name: 边界追踪+薄壁突破+管道延伸

import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import components_mask, get_bg, neighbors4

# -----------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concretization and trace:
#    - The original task used an open U/funnel wall. This combo replaces it with
#      one closed rectangular shell containing a seed blob.
#    - Exactly one shell side is 1 cell thick; the other three sides are 2 or 3
#      cells thick. That unique minimum thickness determines the breakout
#      direction.
#    - Original behavior at the changed steps:
#        1. Flood the whole interior.
#        2. Exit through an already-open mouth.
#        3. Treat endpoint rays as invisible beam boundaries.
#    - Combo behavior:
#        1. Recover the sealed interior from the seed, but do not flood it.
#        2. Paint only the interior cells that touch the wall in 4-neighborhood,
#           leaving a visible hollow core.
#        3. Find the uniquely thinnest wall side, convert that side's wall strip
#           into fill as a breach, and extend the beam straight outward.
#        4. Extend the orthogonal shell borders alongside that beam in wall
#           color, so the outgoing fill sits inside a visible pipe.
#    - Small mental example: seed in a closed box whose top wall is 1 thick and
#      left/right/bottom are thicker. The old solver would have filled the whole
#      cavity if there were an opening. The combo solver instead draws only the
#      cavity outline, breaks the top strip, sends fill upward, and grows the
#      left/right walls upward as a tube.
#
# b) Complexity calibration:
#    - The shape family is restricted to axis-aligned shells with four possible
#      thin-wall directions. That keeps inference discrete and repeated across
#      examples.
#    - The interior must retain a nontrivial untouched core, so the boundary
#      tracing effect is visibly different from full flood-fill.
#    - The pipe wall additions are repeated on both sides of the beam, which
#      gives the solver enough evidence that this is border extension rather
#      than arbitrary extra painting.
#
# c) Implicit assumptions that changed from the original:
#    - Endpoint detection is invalid because the wall is now a loop.
#    - The old global flood mask is invalid because the interior is no longer
#      supposed to become solid fill.
#    - The radiate direction now comes from wall-thickness comparison, not from
#      an already-visible opening.
#    - The outside beam is no longer just fill geometry; some of its boundary
#      cells must be recolored as wall to form the pipe.
#    - Obstacle shadows should only erase the outward beam fill, not the traced
#      interior contour or the pipe walls.
#
# d) Placement legality enforced by the generator:
#    - The shell is rendered as a rectangle ring with one thin side and three
#      thicker sides, so it never self-intersects.
#    - Grid margins guarantee visible space outside the thin wall for the beam
#      and pipe extension.
#    - Obstacles are placed only on background cells outside the shell, away
#      from the wall, and never on future pipe-wall cells.
#
# e) Visual observability:
#    - Human-visible cues are structural only: thin side location, hollow traced
#      cavity, filled breach strip, and pipe walls around the beam.
#    - No color-id ordering is used anywhere. The fill color is inferred by
#      proximity to the wall, and the thin side is inferred by repeated wall
#      thickness, not by a hidden numeric rule.
#
# f) Visual cleanliness:
#    - The output separates roles cleanly:
#        wall color = shell + pipe border,
#        fill color = traced cavity edge + breach + beam interior,
#        obstacle colors = sparse blockers only.
#    - The generator rejects cramped cavities and short beams so the three combo
#      effects do not visually collapse into one another.
#
# g) New main() assumptions and how generate_input() satisfies them:
#    - The seed must lie inside one sealed cavity. The generator builds exactly
#      one shell and places the seed inside it.
#    - The thinnest side must be unique. The generator fixes one side to
#      thickness 1 and all others to 2-3.
#    - The cavity boundary must have background cells left to paint. The
#      generator validates trace size after placing the seed blob.
#    - The beam corridor and pipe walls must have visible positive length. The
#      generator validates both before accepting a sample.
# -----------------------------------------------------------------------------


GRID_H_VERT = (12, 18)
GRID_H_HORIZ = (8, 18)
GRID_W = (15, 19)

INNER_H_VERT = (4, 6)
INNER_W_VERT = (5, 8)
INNER_H_HORIZ = (5, 7)
INNER_W_HORIZ = (4, 6)

OPEN_END_MARGIN = 3
FAR_SIDE_MARGIN = 2
SIDE_MARGIN = 1

THICK_OTHER = (2, 3)

SEED_SIZE_WEIGHTS = [1, 1, 1, 1, 2, 2, 3, 4, 5]
SEED_DEPTH_LOOKBACK = 2

MAX_OBSTACLES = 5
OBSTACLE_IN_BEAM_PROB = 0.8
OBSTACLE_MIN_SEP = 2

MIN_NEW_FILL = 20
MIN_TRACE = 8
MIN_CORE = 4
MIN_BEAM_FILL = 10
MIN_PIPE_WALL = 8
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
    rr = start[0] + d[0]
    cc = start[1] + d[1]
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
        assert runs and min(runs) > 0, "expected each interior side to face wall"
        thicknesses[d] = min(runs)
    return thicknesses


def _choose_radiate_dir(thicknesses: dict[tuple[int, int], int]) -> tuple[int, int]:
    ordered = sorted(thicknesses.items(), key=lambda kv: kv[1])
    assert len(ordered) == 4, "expected four side thicknesses"
    assert ordered[0][1] < ordered[1][1], "expected a unique thinnest side"
    return ordered[0][0]


def _build_combo_masks(
    comp_mask: np.ndarray,
    interior: np.ndarray,
    thicknesses: dict[tuple[int, int], int],
    rad: tuple[int, int],
) -> dict:
    h, w = comp_mask.shape
    rows, cols = np.where(interior)
    top = int(rows.min())
    bottom = int(rows.max())
    left = int(cols.min())
    right = int(cols.max())

    t_top = thicknesses[(-1, 0)]
    t_bottom = thicknesses[(1, 0)]
    t_left = thicknesses[(0, -1)]
    t_right = thicknesses[(0, 1)]

    outer_top = top - t_top
    outer_bottom = bottom + t_bottom
    outer_left = left - t_left
    outer_right = right + t_right
    assert outer_top >= 0 and outer_left >= 0 and outer_bottom < h and outer_right < w

    trace_geom = np.zeros_like(comp_mask, dtype=bool)
    for r, c in zip(rows, cols):
        r = int(r)
        c = int(c)
        if any(
            0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]
            for rr, cc in neighbors4(r, c)
        ):
            trace_geom[r, c] = True

    core_geom = interior & (~trace_geom)
    breach = np.zeros_like(comp_mask, dtype=bool)
    beam_fill = np.zeros_like(comp_mask, dtype=bool)
    pipe_wall = np.zeros_like(comp_mask, dtype=bool)

    if rad == (-1, 0):
        for c in range(left, right + 1):
            rr = top - 1
            while rr >= 0 and comp_mask[rr, c]:
                breach[rr, c] = True
                rr -= 1
        beam_fill[:outer_top, left : right + 1] = True
        pipe_wall[:top, outer_left:left] = True
        pipe_wall[:top, right + 1 : outer_right + 1] = True
    elif rad == (1, 0):
        for c in range(left, right + 1):
            rr = bottom + 1
            while rr < h and comp_mask[rr, c]:
                breach[rr, c] = True
                rr += 1
        beam_fill[outer_bottom + 1 :, left : right + 1] = True
        pipe_wall[bottom + 1 :, outer_left:left] = True
        pipe_wall[bottom + 1 :, right + 1 : outer_right + 1] = True
    elif rad == (0, -1):
        for r in range(top, bottom + 1):
            cc = left - 1
            while cc >= 0 and comp_mask[r, cc]:
                breach[r, cc] = True
                cc -= 1
        beam_fill[top : bottom + 1, :outer_left] = True
        pipe_wall[outer_top:top, :left] = True
        pipe_wall[bottom + 1 : outer_bottom + 1, :left] = True
    else:
        for r in range(top, bottom + 1):
            cc = right + 1
            while cc < w and comp_mask[r, cc]:
                breach[r, cc] = True
                cc += 1
        beam_fill[top : bottom + 1, outer_right + 1 :] = True
        pipe_wall[outer_top:top, right + 1 :] = True
        pipe_wall[bottom + 1 : outer_bottom + 1, right + 1 :] = True

    return {
        "trace_geom": trace_geom,
        "core_geom": core_geom,
        "breach": breach,
        "beam_fill": beam_fill,
        "pipe_wall": pipe_wall,
        "interior_bbox": (top, left, bottom, right),
        "outer_bbox": (outer_top, outer_left, outer_bottom, outer_right),
    }


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, copy=False)
    h, w = g.shape
    bg = get_bg(g)
    vals = [int(v) for v in np.unique(g) if int(v) != bg]
    assert vals, "expected at least one non-background color"

    wall = max(vals, key=lambda c: int((g == c).sum()))
    wall_mask = g == wall
    comps = [
        set((int(r), int(c)) for r, c in pts)
        for pts in components_mask(wall_mask, connectivity=8)
    ]
    assert comps, "expected at least one wall component"

    other_colors = [c for c in vals if c != wall]
    assert other_colors, "expected at least one non-wall color"
    wall_points = [p for comp in comps for p in comp]
    fill = min(other_colors, key=lambda c: _min_dist_to_wall(g, c, wall_points))
    fill_pos = [(int(r), int(c)) for r, c in zip(*np.where(g == fill))]
    if not fill_pos:
        return {"grid": g, "bg": bg, "wall": wall, "fill": fill, "fill_pos": []}

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

    interior_passable = (~comp_mask) & ((g == bg) | (g == fill))
    interior = _seed_fill(interior_passable, fill_pos)
    assert interior.any(), "expected seed to reach an enclosed cavity"
    assert not _touches_border(interior), "expected the input cavity to stay sealed"

    thicknesses = _measure_side_thicknesses(comp_mask, interior)
    rad = _choose_radiate_dir(thicknesses)
    combo_masks = _build_combo_masks(comp_mask, interior, thicknesses, rad)

    return {
        "grid": g,
        "shape": (h, w),
        "bg": int(bg),
        "wall": int(wall),
        "fill": int(fill),
        "fill_pos": fill_pos,
        "comp_mask": comp_mask,
        "interior": interior,
        "thicknesses": thicknesses,
        "rad": rad,
        **combo_masks,
    }


def _shadow_beam(
    out: np.ndarray,
    beam_mask: np.ndarray,
    obstacles: np.ndarray,
    bg: int,
    rad: tuple[int, int],
) -> np.ndarray:
    remaining = beam_mask.copy()

    def shadow_process(coords: list[tuple[int, int]]) -> None:
        started = False
        blocked = False
        for r, c in coords:
            if remaining[r, c]:
                started = True
            if started and obstacles[r, c]:
                blocked = True
                continue
            if blocked and remaining[r, c]:
                out[r, c] = bg
                remaining[r, c] = False

    dr, dc = rad
    if dr != 0:
        for c in range(out.shape[1]):
            rows = range(out.shape[0]) if dr > 0 else range(out.shape[0] - 1, -1, -1)
            shadow_process([(r, c) for r in rows])
    else:
        for r in range(out.shape[0]):
            cols = range(out.shape[1]) if dc > 0 else range(out.shape[1] - 1, -1, -1)
            shadow_process([(r, c) for c in cols])
    return remaining


def _variant_metrics(grid: np.ndarray) -> dict:
    info = _analyze_scene(grid)
    g = info["grid"]
    bg = info["bg"]
    out = main(grid)
    return {
        "rad": info["rad"],
        "thin_side": info["thicknesses"][info["rad"]],
        "other_thicknesses": tuple(v for d, v in info["thicknesses"].items() if d != info["rad"]),
        "trace_added": int((info["trace_geom"] & (g == bg)).sum()),
        "core_bg_kept": int((info["core_geom"] & (out == bg)).sum()),
        "breach_added": int((info["breach"] & (out == info["fill"])).sum()),
        "beam_fill_added": int((info["beam_fill"] & (out == info["fill"])).sum()),
        "pipe_wall_added": int((info["pipe_wall"] & (g == bg) & (out == info["wall"])).sum()),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the inside rim of a sealed chamber, break its thinnest wall, and wrap the outgoing beam in an extended pipe.

    Input structure:
    - The dominant color is background. The most frequent non-background color is one closed rectangular shell around a cavity.
    - A smaller non-wall color blob sits inside that cavity. Exactly one shell side is 1 cell thick, while the other three sides are thicker. Any further non-background pixels are outside obstacles.
    Rule:
    - Recover the sealed cavity from the seed, but do not flood its whole interior. Color only the cavity cells that touch the wall, creating an inner outline.
    - Find the uniquely thinnest wall side, turn that wall strip into a breach of fill color, continue the fill straight outward, and keep obstacles as blockers that cast background shadows only in that outward beam.
    - Extend the shell's side borders along the beam in wall color so the beam is visibly enclosed inside a pipe.

    Key insight: The solver must combine three cues at once: closed-room reasoning to find the cavity, thickness comparison to choose the exit side, and separation of fill-vs-wall roles so the beam interior fills while the beam border becomes wall.
    """
    info = _analyze_scene(grid)
    g = info["grid"]
    if not info["fill_pos"]:
        return g.copy()

    bg = info["bg"]
    wall = info["wall"]
    fill = info["fill"]

    domain = info["interior"] | info["breach"] | info["beam_fill"]
    passable = domain & ((g == bg) | (g == fill) | info["breach"])
    seen = _seed_fill(passable, info["fill_pos"])

    trace_mask = info["trace_geom"] & seen & (g == bg)
    breach_mask = info["breach"] & seen
    beam_mask = info["beam_fill"] & seen & (g == bg)

    out = g.copy()
    out[trace_mask | breach_mask | beam_mask] = fill

    obstacles = (g != bg) & (g != fill) & (g != wall)
    beam_mask = _shadow_beam(out, beam_mask, obstacles, bg, info["rad"])

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
    t = {
        "top": random.randint(*THICK_OTHER),
        "bottom": random.randint(*THICK_OTHER),
        "left": random.randint(*THICK_OTHER),
        "right": random.randint(*THICK_OTHER),
    }
    if rad == (-1, 0):
        t["top"] = 1
    elif rad == (1, 0):
        t["bottom"] = 1
    elif rad == (0, -1):
        t["left"] = 1
    else:
        t["right"] = 1
    return t


def _choose_inner_dims(rad: tuple[int, int], h: int, w: int, t: dict[str, int]) -> tuple[int, int] | None:
    hs = list(range(INNER_H_VERT[0], INNER_H_VERT[1] + 1)) if rad[0] else list(range(INNER_H_HORIZ[0], INNER_H_HORIZ[1] + 1))
    ws = list(range(INNER_W_VERT[0], INNER_W_VERT[1] + 1)) if rad[0] else list(range(INNER_W_HORIZ[0], INNER_W_HORIZ[1] + 1))
    random.shuffle(hs)
    random.shuffle(ws)
    for inner_h in hs:
        for inner_w in ws:
            outer_h = t["top"] + inner_h + t["bottom"]
            outer_w = t["left"] + inner_w + t["right"]
            if rad[0]:
                if outer_h + OPEN_END_MARGIN + FAR_SIDE_MARGIN <= h and outer_w + SIDE_MARGIN * 2 <= w:
                    return inner_h, inner_w
            else:
                if outer_w + OPEN_END_MARGIN + FAR_SIDE_MARGIN <= w and outer_h + SIDE_MARGIN * 2 <= h:
                    return inner_h, inner_w
    return None


def _place_outer_box(rad: tuple[int, int], h: int, w: int, outer_h: int, outer_w: int) -> tuple[int, int]:
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
    beam_fill: np.ndarray,
    interior: np.ndarray,
    wall_mask: np.ndarray,
    pipe_wall: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    dist = distance_transform_cdt(~wall_mask, metric="taxicab")
    exterior_bg = (~interior) & (~wall_mask) & (grid == bg)
    eligible = exterior_bg & (dist > 1) & (~pipe_wall)
    all_coords = [(int(r), int(c)) for r, c in zip(*np.where(eligible))]
    beam_coords = [p for p in all_coords if beam_fill[p[0], p[1]]]
    random.shuffle(all_coords)
    random.shuffle(beam_coords)
    return all_coords, beam_coords


def _place_obstacles(
    n_obstacles: int,
    candidates: list[tuple[int, int]],
    beam_candidates: list[tuple[int, int]],
    obstacle_colors: list[int],
    grid: np.ndarray,
) -> None:
    placed: list[tuple[int, int]] = []
    n_beam = random.randint(0, min(n_obstacles, len(beam_candidates)))
    if beam_candidates and random.random() < OBSTACLE_IN_BEAM_PROB:
        n_beam = max(1, n_beam)
    for p in beam_candidates:
        if len(placed) >= n_beam:
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

    thicknesses = _choose_thicknesses(rad)
    dims = _choose_inner_dims(rad, h, w, thicknesses)
    if dims is None:
        return None
    inner_h, inner_w = dims
    outer_h = thicknesses["top"] + inner_h + thicknesses["bottom"]
    outer_w = thicknesses["left"] + inner_w + thicknesses["right"]
    r0, c0 = _place_outer_box(rad, h, w, outer_h, outer_w)

    palette = random.sample(range(10), random.randint(*PALETTE_SIZE))
    bg, wall_color, fill_color = palette[:3]
    obstacle_colors = palette[3:]

    grid = np.full((h, w), bg, dtype=int)
    wall_mask, interior = _render_shell(grid, r0, c0, inner_h, inner_w, thicknesses, wall_color)

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
        info = _analyze_scene(grid)
    except Exception:
        return None

    trace_added = int((info["trace_geom"] & (grid == bg)).sum())
    core_bg = int((info["core_geom"] & (grid == bg)).sum())
    beam_fill = int(info["beam_fill"].sum())
    pipe_wall = int((info["pipe_wall"] & (grid == bg)).sum())
    if trace_added < MIN_TRACE or core_bg < MIN_CORE or beam_fill < MIN_BEAM_FILL or pipe_wall < MIN_PIPE_WALL:
        return None

    n_obstacles = random.randint(0, MAX_OBSTACLES)
    if n_obstacles and not obstacle_colors:
        available = [c for c in range(10) if c not in (bg, wall_color, fill_color)]
        obstacle_colors = random.sample(available, random.randint(1, 2))
    if n_obstacles:
        candidates, beam_candidates = _find_obstacle_candidates(
            grid,
            bg,
            wall_color,
            info["beam_fill"],
            info["interior"],
            info["comp_mask"],
            info["pipe_wall"],
        )
        _place_obstacles(n_obstacles, candidates, beam_candidates, obstacle_colors, grid)

    try:
        info = _analyze_scene(grid)
        out = main(grid)
        metrics = _variant_metrics(grid)
    except Exception:
        return None

    new_fill = (out == fill_color) & (grid == bg)
    if int(new_fill.sum()) < MIN_NEW_FILL or not _touches_border(new_fill):
        return None
    if metrics["trace_added"] < MIN_TRACE or metrics["core_bg_kept"] < MIN_CORE:
        return None
    if metrics["beam_fill_added"] < MIN_BEAM_FILL or metrics["pipe_wall_added"] < MIN_PIPE_WALL:
        return None
    if metrics["thin_side"] != 1 or not all(v >= 2 for v in metrics["other_thicknesses"]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 combo input")
