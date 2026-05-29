# Combo variant of b9e38dc0 (index 31)
# combo_spec: D8/0+D17/1
# Name: 矩形框缺一边+填充后擦除边框

from __future__ import annotations

import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import components_mask, get_bg, neighbors4


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    The original task used a generic half-enclosure whose two endpoints defined the opening
#    direction. This combo narrows that geometry to one specific family: a 1-pixel rectangular
#    frame with exactly one side missing as an edge segment between the two corner pixels.
#    Example: if the input contains
#      x . . x
#      x s . x
#      x . . x
#      x x x x
#    then the missing side is the top edge, so the fill can occupy the 2x2 interior and also
#    continue upward between the two vertical side walls. The combo then adds a second step:
#    after that fill/radiation result is finalized, every wall pixel of the chosen frame is
#    erased back to background. Relative to the base task, the geometry becomes more regular,
#    and relative to the D8-only variant, the erased object is now always a clean rectangular
#    three-sided frame rather than an arbitrary zigzag enclosure.
#
# b) Complexity calibration
#    The visual change is strong but still simple enough for ARC. Each sample still contains
#    only one dominant wall color, one nearby fill blob, and a few sparse singleton obstacles.
#    The new geometry has only four opening directions and one clear family of shapes, so the
#    examples repeat the key cue. At the same time, the erased wall contributes 17-25 pixels,
#    which is large enough that the post-fill disappearance is obvious by eye.
#
# c) Implicit-assumption audit against the original main()
#    - The base solver inferred the opening from endpoint tangents of a free-form component.
#      That is unnecessarily brittle once the wall is constrained to an axis-aligned frame.
#      The combo solver instead identifies the wall bbox and asks which bbox side is absent.
#    - The base barrier logic relied on diagonal or 1:2 endpoint rays. Those assumptions no
#      longer hold; the combo always uses straight cardinal rays extending the two side walls.
#    - "Fill color = nearest non-wall color to the wall" still works, but only if obstacles are
#      kept farther away than the seed. The generator preserves that.
#    - "Wall = most frequent non-bg color" still works because the frame remains much larger
#      than the seed blob and obstacle set.
#
# d) Placement legality
#    The generator places a full rectangular bbox first, chooses which side is open, and then
#    renders the three remaining sides without overlapping any other object. The frame bbox is
#    kept in-bounds, the open side has at least a 3-cell margin to the radiate edge, and the
#    closed side has at least a 2-cell margin to its opposite edge. Obstacles are sampled only
#    from background cells, so they never overwrite the frame or the seed blob.
#
# e) Visual observability
#    A human can infer the rule from a few examples because two linked regularities repeat:
#    the enclosure is always a rectangular three-sided frame, and the entire frame disappears
#    after the same fill/radiate action. The radiating corridor always continues through the
#    missing side toward one grid edge, so the opening direction is explicit in the pixels.
#
# f) Visual cleanliness
#    The output is cleaner than the original: there is one filled rectangle-like mass plus one
#    outward strip, with sparse obstacle holes cut out by shadows. Removing the frame prevents
#    the output from stacking a thick border on top of the newly filled region.
#
# g) Change inferability
#    No hidden color ordering is introduced. The only color change is structural: wall pixels
#    become background after they have already served as barriers. Because the removed pixels
#    outline a regular rectangular frame seen in the input, the bg conversion is easy to infer.
#
# h) Generator-side anti-degeneration check
#    The generator validates three properties on each candidate:
#    1. the detected wall component is exactly a rectangle missing one side,
#    2. the filled result reaches the intended radiate border with at least 20 new fill cells,
#    3. the erased-output differs from the non-erased-output exactly on the wall component.
#    This prevents accidental fallback to the base task or to a sample where erasing has no
#    visible effect.
#
# i) New solver assumptions and generator guarantees
#    `_detect_open_rect_frame` assumes the chosen wall component equals the border of one bbox
#    with one side segment removed and no extra protrusions. The generator constructs exactly
#    that. The shadow pass assumes the radiate direction is cardinal, which is guaranteed by
#    the four opening-side choices. The BFS assumes the seed lies inside the frame near the
#    closed side; seed placement is restricted to interior cells satisfying that condition.


GRID_H_VERT = (11, 18)
GRID_H_HORIZ = (8, 18)
GRID_W = (15, 19)

FRAME_H_RANGE = (6, 9)
FRAME_W_RANGE = (6, 10)
WALL_PIXELS_MIN = 17
WALL_PIXELS_MAX = 25

OPEN_END_MARGIN = 3
CLOSED_END_MARGIN = 2

SEED_SIZE_WEIGHTS = [1, 1, 1, 1, 2, 2, 3, 4, 5, 6]
SEED_DEPTH_LOOKBACK = 3

MAX_OBSTACLES = 5
OBSTACLE_IN_FILL_PROB = 0.65
OBSTACLE_MIN_SEP = 2

MIN_NEW_FILL = 20
PALETTE_SIZE = (3, 5)
GEN_RETRIES = 400


def _wall_count(frame_h: int, frame_w: int, missing_side: str) -> int:
    if missing_side in ("top", "bottom"):
        return 2 * frame_h + frame_w - 2
    return frame_h + 2 * frame_w - 2


def _render_frame_mask(frame_h: int, frame_w: int, missing_side: str) -> np.ndarray:
    mask = np.zeros((frame_h, frame_w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    if missing_side == "top":
        mask[0, 1:-1] = False
    elif missing_side == "bottom":
        mask[-1, 1:-1] = False
    elif missing_side == "left":
        mask[1:-1, 0] = False
    else:
        mask[1:-1, -1] = False
    return mask


def _detect_open_rect_frame(comp_mask: np.ndarray) -> tuple[int, int, int, int, str, tuple[int, int], list[tuple[int, int]]] | None:
    coords = np.argwhere(comp_mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    frame_h = int(r1 - r0 + 1)
    frame_w = int(c1 - c0 + 1)
    if frame_h < 3 or frame_w < 3:
        return None

    top = comp_mask[r0, c0 : c1 + 1]
    bottom = comp_mask[r1, c0 : c1 + 1]
    left = comp_mask[r0 : r1 + 1, c0]
    right = comp_mask[r0 : r1 + 1, c1]

    def state(line: np.ndarray) -> str:
        if bool(line.all()):
            return "full"
        if len(line) >= 2 and bool(line[0]) and bool(line[-1]) and not bool(line[1:-1].any()):
            return "gap"
        return "other"

    states = {
        "top": state(top),
        "bottom": state(bottom),
        "left": state(left),
        "right": state(right),
    }
    missing = [side for side, kind in states.items() if kind == "gap"]
    present = [side for side, kind in states.items() if kind == "full"]
    if len(missing) != 1 or len(present) != 3:
        return None

    missing_side = missing[0]
    expected_local = _render_frame_mask(frame_h, frame_w, missing_side)
    expected = np.zeros_like(comp_mask, dtype=bool)
    expected[r0 : r1 + 1, c0 : c1 + 1] = expected_local
    if not np.array_equal(comp_mask, expected):
        return None

    if missing_side == "top":
        rad = (-1, 0)
        endpoints = [(r0, c0), (r0, c1)]
    elif missing_side == "bottom":
        rad = (1, 0)
        endpoints = [(r1, c0), (r1, c1)]
    elif missing_side == "left":
        rad = (0, -1)
        endpoints = [(r0, c0), (r1, c0)]
    else:
        rad = (0, 1)
        endpoints = [(r0, c1), (r1, c1)]
    return r0, c0, r1, c1, missing_side, rad, endpoints


def _solve_core(
    grid: np.ndarray,
    *,
    erase_wall: bool,
    return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    g = np.array(grid, copy=True)
    h, w = g.shape

    bg = int(get_bg(g))
    vals = np.unique(g)
    non_bg = [int(v) for v in vals if int(v) != bg]
    assert non_bg, "expected at least one non-background color"

    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    wall_mask = g == wall
    comps = [set((int(r), int(c)) for r, c in pts) for pts in components_mask(wall_mask, connectivity=8)]
    assert comps, "expected at least one wall connected component"

    other_colors = [c for c in non_bg if c != wall]
    assert other_colors, "expected at least one non-wall color"

    wall_points = [p for comp in comps for p in comp]

    def min_dist_to_wall(color: int) -> tuple[int, int]:
        pts = list(zip(*np.where(g == color)))
        best = 10**9
        for r, c in pts:
            rr, cc = int(r), int(c)
            for wr, wc in wall_points:
                d = abs(rr - wr) + abs(cc - wc)
                if d < best:
                    best = d
                    if best == 0:
                        return 0, len(pts)
        return best, len(pts)

    fill_stats = {c: min_dist_to_wall(c) for c in other_colors}
    fill = min(other_colors, key=lambda c: (fill_stats[c][0], fill_stats[c][1]))

    fill_pos = [(int(r), int(c)) for r, c in zip(*np.where(g == fill))]
    if not fill_pos:
        out = g.copy()
        meta = {
            "bg": bg,
            "wall": wall,
            "fill": fill,
            "wall_comp_mask": np.zeros_like(g, dtype=bool),
            "filled_mask": np.zeros_like(g, dtype=bool),
            "rad": (0, 0),
            "missing_side": None,
            "frame_bbox": None,
        }
        return (out, meta) if return_meta else out

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
    comp_mask = np.zeros_like(g, dtype=bool)
    for r, c in wall_comp:
        comp_mask[r, c] = True

    found = _detect_open_rect_frame(comp_mask)
    assert found is not None, "expected a rectangle frame missing one side"
    r0, c0, r1, c1, missing_side, rad, endpoints = found

    barrier = comp_mask.copy()
    dr, dc = rad
    for start_r, start_c in endpoints:
        rr, cc = start_r, start_c
        while True:
            rr += dr
            cc += dc
            if not (0 <= rr < h and 0 <= cc < w):
                break
            barrier[rr, cc] = True

    passable = (~barrier) & ((g == bg) | (g == fill))
    seen = np.zeros_like(passable, dtype=bool)
    stack = fill_pos[:]
    for r, c in stack:
        if passable[r, c]:
            seen[r, c] = True
    while stack:
        r, c = stack.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                stack.append((rr, cc))

    filled_mask = seen & (g == bg)
    out = g.copy()
    out[filled_mask] = fill

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

    if rad != (0, 0):
        if dr != 0:
            rows = range(h) if dr > 0 else range(h - 1, -1, -1)
            for c in range(w):
                shadow_process([(r, c) for r in rows])
        else:
            cols = range(w) if dc > 0 else range(w - 1, -1, -1)
            for r in range(h):
                shadow_process([(r, c) for c in cols])

    if erase_wall:
        out[comp_mask] = bg

    meta = {
        "bg": bg,
        "wall": wall,
        "fill": fill,
        "wall_comp_mask": comp_mask,
        "filled_mask": filled_mask.copy(),
        "rad": rad,
        "missing_side": missing_side,
        "frame_bbox": (r0, c0, r1, c1),
    }
    return (out, meta) if return_meta else out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the inside of a three-sided rectangular frame, radiate through its missing edge, then erase the frame itself.

    Input structure:
    - The background is the most common color, and the dominant non-background color forms one axis-aligned 1-pixel rectangle border with exactly one side missing.
    - A small blob of another color sits inside that frame near the closed side; any remaining non-background colors are obstacle pixels on the background.

    Rule:
    - Detect which side of the rectangle is missing, use the remaining three sides plus two straight outward wall extensions as barriers, and flood-fill all reachable background from the seed color through the opening.
    - Apply obstacle shadows along that same opening direction, then erase the full rectangular wall component back to background so only the fill mass, outward strip, and untouched obstacles remain.

    Key insight: The missing side does two jobs at once: it tells you the radiate direction and it identifies a temporary frame that must first constrain the fill and then disappear completely from the final image.
    """
    return _solve_core(grid, erase_wall=True)


def _choose_frame_dims(missing_side: str, h: int, w: int) -> tuple[int, int]:
    candidates = []
    for frame_h in range(FRAME_H_RANGE[0], FRAME_H_RANGE[1] + 1):
        for frame_w in range(FRAME_W_RANGE[0], FRAME_W_RANGE[1] + 1):
            wall_pixels = _wall_count(frame_h, frame_w, missing_side)
            if not (WALL_PIXELS_MIN <= wall_pixels <= WALL_PIXELS_MAX):
                continue
            interior_area = (frame_h - 2) * (frame_w - 2)
            if interior_area < 12:
                continue
            if frame_h > h or frame_w > w:
                continue
            if missing_side in ("top", "bottom") and frame_h > h - OPEN_END_MARGIN - CLOSED_END_MARGIN:
                continue
            if missing_side in ("left", "right") and frame_w > w - OPEN_END_MARGIN - CLOSED_END_MARGIN:
                continue
            candidates.append((frame_h, frame_w))
    assert candidates, f"no valid frame dimensions for {missing_side}"
    return random.choice(candidates)


def _place_frame_bbox(h: int, w: int, frame_h: int, frame_w: int, missing_side: str) -> tuple[int, int, int, int]:
    if missing_side == "top":
        r0 = random.randint(OPEN_END_MARGIN, h - frame_h - CLOSED_END_MARGIN)
        c0 = random.randint(0, w - frame_w)
    elif missing_side == "bottom":
        r0 = random.randint(CLOSED_END_MARGIN, h - frame_h - OPEN_END_MARGIN)
        c0 = random.randint(0, w - frame_w)
    elif missing_side == "left":
        r0 = random.randint(0, h - frame_h)
        c0 = random.randint(OPEN_END_MARGIN, w - frame_w - CLOSED_END_MARGIN)
    else:
        r0 = random.randint(0, h - frame_h)
        c0 = random.randint(CLOSED_END_MARGIN, w - frame_w - OPEN_END_MARGIN)
    return int(r0), int(c0), int(r0 + frame_h - 1), int(c0 + frame_w - 1)


def _frame_interior_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _seed_pool(interior: list[tuple[int, int]], wall_mask: np.ndarray, missing_side: str, frame_bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    pool = []
    for r, c in interior:
        adjacent_to_wall = (
            wall_mask[r - 1, c]
            or wall_mask[r + 1, c]
            or wall_mask[r, c - 1]
            or wall_mask[r, c + 1]
        )
        if not adjacent_to_wall:
            continue
        if missing_side == "top" and r < max(r0 + 1, r1 - SEED_DEPTH_LOOKBACK):
            continue
        if missing_side == "bottom" and r > min(r1 - 1, r0 + SEED_DEPTH_LOOKBACK):
            continue
        if missing_side == "left" and c < max(c0 + 1, c1 - SEED_DEPTH_LOOKBACK):
            continue
        if missing_side == "right" and c > min(c1 - 1, c0 + SEED_DEPTH_LOOKBACK):
            continue
        pool.append((r, c))
    return pool


def _grow_seed_blob(seed: tuple[int, int], interior_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    blob = {seed}
    frontier = [seed]
    target_size = random.choice(SEED_SIZE_WEIGHTS)
    while len(blob) < target_size and frontier:
        r, c = frontier.pop(random.randrange(len(frontier)))
        neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        random.shuffle(neighbors)
        for rr, cc in neighbors:
            if (rr, cc) in interior_set and (rr, cc) not in blob:
                blob.add((rr, cc))
                frontier.append((rr, cc))
                if len(blob) >= target_size:
                    break
    return blob


def _obstacle_fits(pos: tuple[int, int], placed: list[tuple[int, int]]) -> bool:
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) >= OBSTACLE_MIN_SEP for rr, cc in placed)


def _find_obstacle_candidates(grid: np.ndarray, bg: int, wall_color: int, nominal_fill_zone: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    wall_mask = grid == wall_color
    dist = distance_transform_cdt(~wall_mask, metric="taxicab")
    eligible = (grid == bg) & (dist > 1)
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

    for pos in fill_candidates:
        if len(placed) >= n_fill:
            break
        if _obstacle_fits(pos, placed):
            placed.append(pos)

    placed_set = set(placed)
    for pos in candidates:
        if len(placed) >= n_obstacles:
            break
        if pos not in placed_set and _obstacle_fits(pos, placed):
            placed.append(pos)
            placed_set.add(pos)

    for idx, (r, c) in enumerate(placed):
        grid[r, c] = obstacle_colors[idx % len(obstacle_colors)]


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _touches_radiate_border(mask: np.ndarray, rad: tuple[int, int]) -> bool:
    if rad == (-1, 0):
        return bool(mask[0, :].any())
    if rad == (1, 0):
        return bool(mask[-1, :].any())
    if rad == (0, -1):
        return bool(mask[:, 0].any())
    if rad == (0, 1):
        return bool(mask[:, -1].any())
    return False


def _gen_attempt() -> np.ndarray | None:
    missing_side = random.choice(["top", "bottom", "left", "right"])
    rad = {
        "top": (-1, 0),
        "bottom": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }[missing_side]
    h = random.randint(*GRID_H_VERT) if rad[0] else random.randint(*GRID_H_HORIZ)
    w = random.randint(*GRID_W)

    frame_h, frame_w = _choose_frame_dims(missing_side, h, w)
    r0, c0, r1, c1 = _place_frame_bbox(h, w, frame_h, frame_w, missing_side)

    palette = random.sample(range(10), random.randint(*PALETTE_SIZE))
    bg, wall_color, fill_color = palette[:3]
    obstacle_colors = palette[3:]

    grid = np.full((h, w), bg, dtype=int)
    wall_local = _render_frame_mask(frame_h, frame_w, missing_side)
    wall_mask = np.zeros((h, w), dtype=bool)
    wall_mask[r0 : r1 + 1, c0 : c1 + 1] = wall_local
    grid[wall_mask] = wall_color

    interior = _frame_interior_cells(r0, c0, r1, c1)
    pool = _seed_pool(interior, wall_mask, missing_side, (r0, c0, r1, c1))
    if not pool:
        return None
    blob = _grow_seed_blob(random.choice(pool), set(interior))
    for r, c in blob:
        if grid[r, c] != bg:
            return None
        grid[r, c] = fill_color

    try:
        nominal_out = _solve_core(grid, erase_wall=True)
    except Exception:
        return None
    nominal_fill_zone = (nominal_out == fill_color) & (grid == bg)

    n_obstacles = random.randint(0, MAX_OBSTACLES)
    if n_obstacles and not obstacle_colors:
        available = [c for c in range(10) if c not in (bg, wall_color, fill_color)]
        obstacle_colors = random.sample(available, random.randint(1, 2))
    if n_obstacles:
        candidates, fill_candidates = _find_obstacle_candidates(grid, bg, wall_color, nominal_fill_zone)
        _place_obstacles(n_obstacles, candidates, fill_candidates, obstacle_colors, grid)

    try:
        legacy_out, legacy_meta = _solve_core(grid, erase_wall=False, return_meta=True)
        out, meta = _solve_core(grid, erase_wall=True, return_meta=True)
    except Exception:
        return None

    new_fill = (out == fill_color) & (grid == bg)
    diff_mask = out != legacy_out
    wall_comp_mask = legacy_meta["wall_comp_mask"]
    found = _detect_open_rect_frame(wall_comp_mask)
    if found is None:
        return None
    if meta["missing_side"] != missing_side or meta["rad"] != rad:
        return None
    if int(new_fill.sum()) < MIN_NEW_FILL:
        return None
    if not _touches_border(new_fill) or not _touches_radiate_border(new_fill, rad):
        return None
    if int(wall_comp_mask.sum()) < WALL_PIXELS_MIN:
        return None
    if int(diff_mask.sum()) != int(wall_comp_mask.sum()):
        return None
    if not np.array_equal(diff_mask, wall_comp_mask):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid combo b9e38dc0 input")
