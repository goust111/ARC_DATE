# Combo variant of 53fb4810 (index 3)
# combo_spec: D1/0+D9/0
# Name: 对角线无线索射线

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# Design analysis for the combo variant:
# a) Concrete behavior and behavior trace:
# - Original task: one compact seed sits on one cardinal side of each plus, and the
#   seed repeats in tile-sized jumps along that same cardinal axis. One ray is left
#   pre-expanded in the input as an in-grid clue.
# - New combo: every seed is compact, there is no pre-expanded clue ray anywhere,
#   and the path itself changes from cardinal to diagonal.
# - To make the diagonal direction locally recoverable, each seed is anchored in one
#   of the four "corner mouth" cells of the plus bbox:
#   - NW seed: its bbox bottom-right corner is the plus bbox top-left corner.
#   - NE seed: its bbox bottom-left corner is the plus bbox top-right corner.
#   - SW seed: its bbox top-right corner is the plus bbox bottom-left corner.
#   - SE seed: its bbox top-left corner is the plus bbox bottom-right corner.
# - Those corner mouth cells are background cells in the chosen plus templates, but
#   they are 4-neighbor-adjacent to two perpendicular plus arms. Therefore a single
#   compact marker still yields two contact signs, e.g. {north, west} -> NW.
# - Example: if a 2x2 seed is anchored at the top-left mouth of a plus, the original
#   task would have used one side-attached seed and repeated it straight upward or
#   leftward in jumps of 2. This combo instead keeps only that 2x2 corner seed in
#   the input, and the output stamps the same 2x2 tile at shifts 0, (-1,-1),
#   (-2,-2), ... until the grid edge.
# - Because diagonal stamping moves by one pixel per step, consecutive copies can
#   overlap. The chosen convention is "later stamps win" while plus pixels remain
#   protected. The generator uses the same rule, so solver and generator agree.
#
# b) Complexity calibration:
# - Seed sizes stay in a small, repeated family: 1x2, 2x1, 2x2. This is large enough
#   to make the diagonal repetition visually distinctive, but small enough to avoid
#   noisy payloads.
# - Plus count remains 2-3 and all pluses share one shape/color, so training pairs
#   still have repeated support for the same rule within a grid.
# - The visual change is substantial: compared with the original task, inputs now
#   have zero long clue rays and every completed output ray slants diagonally.
#
# c) Implicit-assumption audit for main():
# - `detect_dominant_big_component_color(...)` still assumes pluses are the only
#   large repeated components. This remains true because all marker seeds are tiny
#   and there is no pre-expanded clue ray in the input.
# - `mark_lab = label_mask(non_mask, connectivity=4)` assumes each compact seed is a
#   separate 4-connected component. The generator enforces non-touching marker/ray
#   placement so different seeds never merge.
# - The original solver assumed one cardinal contact direction. That no longer holds:
#   a diagonal seed intentionally touches two perpendicular arms. The new solver
#   therefore recovers one row sign and one column sign from all plus-marker contacts.
# - The old nozzle-span assertions are invalid after the path change. They are
#   replaced with corner-alignment assertions that pin each seed bbox to one unique
#   plus bbox corner mouth.
# - The extension step can no longer jump by tile height/width. It must shift the
#   whole tile by one diagonal pixel per step, with deterministic overwrite order.
#
# d) Placement legality:
# - Generation is still output-first. It first plans the full diagonal ray union for
#   each plus, rejects collisions or unwanted adjacencies, paints the output, and
#   then erases every non-seed ray cell to form the input.
# - Rays are forbidden from crossing other rays, touching other marker components,
#   or approaching other pluses closely enough to look attached.
#
# e) Visual observability:
# - A human sees only compact corner seeds in the input and complete diagonal rays in
#   the output, so both fused changes are directly observable from 2-4 examples.
# - No hidden ordering by color id is used. Direction is recoverable from visible
#   geometry only: which plus corner mouth contains the seed.
#
# f) Visual cleanliness:
# - The generator blocks marker-marker adjacency and keeps rays away from unrelated
#   pluses. This avoids merged components and ambiguous ownership.
# - Using a single shared plus shape per grid keeps the layout readable even though
#   diagonal rays can sweep larger areas than the original straight rays.
#
# g) Change inferability:
# - There is no color remapping rule to infer; colors are only payload values copied
#   by repetition. The only transformation is spatial, and every sample reinforces
#   the same corner-seed -> diagonal-ray relation.


def detect_dominant_big_component_color(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    min_h: int = 3,
    min_w: int = 3,
    connectivity: int = 4,
) -> int | None:
    """
    Local helper (only used by this solver): choose a color that has the most "big" components.

    Big component = connected component whose bbox is at least (min_h x min_w).
    Tie-break: total area of big components (larger wins).
    """
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    best_key: tuple[int, int] | None = None
    best_col: int | None = None

    for col in sorted(set(map(int, grid.flatten())) - {int(bg)}):
        lab, n = label_mask(grid == int(col), connectivity=int(connectivity))
        big_areas: list[int] = []
        for k in range(1, int(n) + 1):
            coords = np.argwhere(lab == int(k))
            if coords.size == 0:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h >= int(min_h) and w >= int(min_w):
                big_areas.append(int(coords.shape[0]))
        if not big_areas:
            continue
        key = (int(len(big_areas)), int(sum(big_areas)))
        if best_key is None or key > best_key:
            best_key = key
            best_col = int(col)

    return int(best_col) if best_col is not None else None


def _rect_intersects_grid(r0: int, c0: int, h: int, w: int, H: int, W: int) -> bool:
    return int(r0) < int(H) and int(r0) + int(h) > 0 and int(c0) < int(W) and int(c0) + int(w) > 0


def _corner_alignment_ok(
    plus_bbox: tuple[int, int, int, int],
    seed_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
) -> bool:
    pr0, pc0, pr1, pc1 = map(int, plus_bbox)
    r0, c0, r1, c1 = map(int, seed_bbox)
    dr, dc = map(int, direction)
    if (dr, dc) == (-1, -1):
        return r1 == pr0 and c1 == pc0
    if (dr, dc) == (-1, 1):
        return r1 == pr0 and c0 == pc1
    if (dr, dc) == (1, -1):
        return r0 == pr1 and c1 == pc0
    if (dr, dc) == (1, 1):
        return r0 == pr1 and c0 == pc1
    return False


def _diagonal_direction_from_contacts(
    plus_mask: np.ndarray,
    pts: np.ndarray,
    H: int,
    W: int,
) -> tuple[int, int] | None:
    row_signs: set[int] = set()
    col_signs: set[int] = set()
    for pr, pc in pts:
        pr = int(pr)
        pc = int(pc)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = pr + dr
            cc = pc + dc
            if 0 <= rr < H and 0 <= cc < W and bool(plus_mask[rr, cc]):
                drow = pr - rr
                dcol = pc - cc
                if drow != 0:
                    row_signs.add(1 if drow > 0 else -1)
                if dcol != 0:
                    col_signs.add(1 if dcol > 0 else -1)
    if len(row_signs) != 1 or len(col_signs) != 1:
        return None
    return int(next(iter(row_signs))), int(next(iter(col_signs)))


def extend_repeating_tile_diagonal(
    out: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    bg: int,
    protect: set[int] | None = None,
) -> None:
    """Stamp a tile repeatedly by unit diagonal shifts until it leaves the grid."""
    protect = set() if protect is None else set(map(int, protect))
    H, W = out.shape
    h, w = tile.shape
    dr, dc = map(int, direction)
    if abs(dr) != 1 or abs(dc) != 1:
        raise ValueError("extend_repeating_tile_diagonal: direction must be one of (±1,±1)")
    bg = int(bg)
    r0 = int(r0)
    c0 = int(c0)

    def put(rr: int, cc: int, v: int) -> None:
        if v == bg:
            return
        if int(out[rr, cc]) in protect:
            return
        out[rr, cc] = int(v)

    shift = 1
    while _rect_intersects_grid(r0 + shift * dr, c0 + shift * dc, h, w, H, W):
        rr0 = r0 + shift * dr
        cc0 = c0 + shift * dc
        for i in range(h):
            for j in range(w):
                rr = rr0 + i
                cc = cc0 + j
                if 0 <= rr < H and 0 <= cc < W:
                    put(rr, cc, int(tile[i, j]))
        shift += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow every compact corner seed into a diagonal repeating ray until the grid edge.

    Input structure:
    - The background is the most common color.
    - One dominant non-background color forms several identical plus-like anchors.
    - Each plus has exactly one compact marker component, anchored in one corner mouth
      of the plus bbox; there is no fully pre-expanded demonstration ray anywhere.

    Rule:
    - For each marker touching a plus, recover one vertical sign and one horizontal
      sign from the plus-marker contacts; together they define one diagonal direction.
    - Repeat the marker's bounding-box tile by shifting it one pixel per step along
      that diagonal until the tile has fully left the grid.
    - Later diagonal stamps may overwrite earlier ray cells, but plus pixels are
      never overwritten.

    Key insight:
    - The seed is no longer on a single side of the plus. Instead it sits in one
      specific corner mouth, so the two perpendicular contacts jointly encode the
      diagonal direction that the compact tile must follow.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    H, W = g.shape

    bg = int(get_bg(g))
    plus_col = detect_dominant_big_component_color(g, bg=bg, min_h=3, min_w=3, connectivity=4)
    if plus_col is None:
        return out

    plus_lab, pn = label_mask(g == int(plus_col), connectivity=4)
    non_mask = (g != int(bg)) & (g != int(plus_col))
    mark_lab, _mn = label_mask(non_mask, connectivity=4)

    for pk in range(1, int(pn) + 1):
        plus_mask = plus_lab == int(pk)
        plus_coords = np.argwhere(plus_mask)
        if plus_coords.size == 0:
            continue

        adj_ids: set[int] = set()
        for r, c in plus_coords:
            r = int(r)
            c = int(c)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and int(mark_lab[rr, cc]) > 0:
                    adj_ids.add(int(mark_lab[rr, cc]))

        plus_bbox = bbox_coords(plus_coords)
        for cid in sorted(adj_ids):
            pts = np.argwhere(mark_lab == int(cid))
            if pts.size == 0:
                continue
            r0, c0, r1, c1 = bbox_coords(pts)
            tile = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))

            direction = _diagonal_direction_from_contacts(plus_mask, pts, H, W)
            if direction is None:
                continue

            assert _corner_alignment_ok(plus_bbox, (int(r0), int(c0), int(r1), int(c1)), direction)

            extend_repeating_tile_diagonal(
                out,
                tile,
                int(r0),
                int(c0),
                direction,
                bg=int(bg),
                protect={int(plus_col)},
            )

    return out


# --------------- generator ---------------

# New implicit assumptions introduced by the diagonal combo, and how generation
# guarantees them:
# - Each seed must touch its plus in exactly one diagonal corner pattern, not in an
#   arbitrary way. `_seed_bbox_for_direction(...)` constructs only those four corner
#   alignments, and `main()` asserts the same geometry.
# - Distinct marker components must remain separate under color-agnostic 4-connectivity.
#   The generator blocks 4-neighbor contact between different ray unions via
#   `ray_guard`, so `mark_lab` never merges two seeds.
# - Rays must not brush unrelated pluses, or the local contact logic could attribute
#   a marker to the wrong anchor. The generator blocks all ray cells from entering the
#   8-neighborhood guard of already-placed pluses.
# - The diagonal overwrite order is now semantically meaningful because repeated
#   stamps can overlap. `_project_diagonal_tile_coords(...)` uses the same "later
#   stamps win" rule as the solver helper, so oracle validation is exact.

_H_RANGE = (13, 29)
_W_RANGE = (13, 23)
_N_PLUS_RANGE = (2, 4)
_SEED_SHAPES = [(1, 2), (2, 1), (2, 2)]
_SEED_MAX_COLORS = 4
_MIN_DIAG_STEPS = 4
_MIN_DISTINCT_DIRS = 2
_PLACEMENT_TRIES = 200
_GEN_RETRIES = 600

_PLUS_SHAPES = [
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[0, 1, 1, 0], [1, 1, 1, 1], [0, 1, 1, 0]], dtype=int),
    np.array(
        [
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 0],
        ],
        dtype=int,
    ),
]

_DIAGONAL_DIRECTIONS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _build_seed_tile(palette: list[int], rng: np.random.Generator) -> np.ndarray:
    th, tw = _SEED_SHAPES[int(rng.integers(0, len(_SEED_SHAPES)))]
    n_cells = int(th * tw)
    max_k = min(_SEED_MAX_COLORS, n_cells, len(palette))
    n_colors = int(rng.integers(1, max_k + 1))
    chosen = list(rng.choice(palette, size=n_colors, replace=False))
    vals = list(chosen) + list(rng.choice(chosen, size=n_cells - n_colors, replace=True))
    rng.shuffle(vals)
    return np.array(vals, dtype=int).reshape(th, tw)


def _seed_bbox_for_direction(
    plus_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
    tile_shape: tuple[int, int],
) -> tuple[int, int]:
    pr0, pc0, pr1, pc1 = map(int, plus_bbox)
    th, tw = map(int, tile_shape)
    dr, dc = map(int, direction)
    if (dr, dc) == (-1, -1):
        return int(pr0 - th + 1), int(pc0 - tw + 1)
    if (dr, dc) == (-1, 1):
        return int(pr0 - th + 1), int(pc1)
    if (dr, dc) == (1, -1):
        return int(pr1), int(pc0 - tw + 1)
    if (dr, dc) == (1, 1):
        return int(pr1), int(pc1)
    raise ValueError("direction must be one of (±1,±1)")


def _rect_in_bounds(r0: int, c0: int, h: int, w: int, H: int, W: int) -> bool:
    return 0 <= int(r0) and 0 <= int(c0) and int(r0) + int(h) <= int(H) and int(c0) + int(w) <= int(W)


def _max_diag_steps(
    r0: int,
    c0: int,
    h: int,
    w: int,
    H: int,
    W: int,
    direction: tuple[int, int],
) -> int:
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r0 + h - 1)
    c1 = int(c0 + w - 1)
    dr, dc = map(int, direction)
    if (dr, dc) == (-1, -1):
        return int(min(r1, c1))
    if (dr, dc) == (-1, 1):
        return int(min(r1, W - 1 - c0))
    if (dr, dc) == (1, -1):
        return int(min(H - 1 - r0, c1))
    if (dr, dc) == (1, 1):
        return int(min(H - 1 - r0, W - 1 - c0))
    raise ValueError("direction must be one of (±1,±1)")


def _project_diagonal_tile_coords(
    H: int,
    W: int,
    seed_r0: int,
    seed_c0: int,
    direction: tuple[int, int],
    tile: np.ndarray,
) -> tuple[list[tuple[int, int, int]], int]:
    """Compute the final visible union after repeated unit diagonal shifts."""
    th, tw = tile.shape
    max_steps = _max_diag_steps(seed_r0, seed_c0, th, tw, H, W, direction)
    dr, dc = map(int, direction)
    painted: dict[tuple[int, int], int] = {}

    for k in range(max_steps + 1):
        rr0 = int(seed_r0 + k * dr)
        cc0 = int(seed_c0 + k * dc)
        for i in range(th):
            for j in range(tw):
                rr = rr0 + i
                cc = cc0 + j
                if 0 <= rr < H and 0 <= cc < W:
                    painted[(int(rr), int(cc))] = int(tile[i, j])

    cells = [(int(r), int(c), int(v)) for (r, c), v in sorted(painted.items())]
    return cells, int(max_steps)


def _rand_start(lo: int, hi: int, rng: np.random.Generator) -> int:
    lo = int(lo)
    hi = int(hi)
    if lo > hi:
        lo, hi = 0, hi
    return int(rng.integers(lo, hi + 1))


def _place_plus_and_ray(
    rng: np.random.Generator,
    plus_shape: np.ndarray,
    palette: list[int],
    H: int,
    W: int,
    occ: np.ndarray,
    plus_guard: np.ndarray,
    ray_guard: np.ndarray,
    out: np.ndarray,
    plus_color: int,
) -> dict | None:
    ph, pw = plus_shape.shape

    for _ in range(_PLACEMENT_TRIES):
        r0 = _rand_start(1, H - ph - 1, rng) if H - ph >= 2 else _rand_start(0, H - ph, rng)
        c0 = _rand_start(1, W - pw - 1, rng) if W - pw >= 2 else _rand_start(0, W - pw, rng)
        plus_cells = [
            (r0 + dr, c0 + dc)
            for dr in range(ph)
            for dc in range(pw)
            if plus_shape[dr, dc]
        ]
        if any(occ[r, c] != 0 for r, c in plus_cells):
            continue
        if any(plus_guard[r, c] for r, c in plus_cells):
            continue
        if any(ray_guard[r, c] for r, c in plus_cells):
            continue

        plus_bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        dirs = list(_DIAGONAL_DIRECTIONS)
        rng.shuffle(dirs)
        for direction in dirs:
            tile = _build_seed_tile(palette, rng)
            sr0, sc0 = _seed_bbox_for_direction(plus_bbox, direction, tile.shape)
            th, tw = tile.shape
            if not _rect_in_bounds(sr0, sc0, th, tw, H, W):
                continue
            steps = _max_diag_steps(sr0, sc0, th, tw, H, W, direction)
            if steps < _MIN_DIAG_STEPS:
                continue

            cells, _max_steps = _project_diagonal_tile_coords(H, W, sr0, sc0, direction, tile)
            if any(occ[r, c] != 0 for r, c, _ in cells):
                continue
            if any(plus_guard[r, c] for r, c, _ in cells):
                continue
            if any(ray_guard[r, c] for r, c, _ in cells):
                continue

            for r, c in plus_cells:
                occ[r, c] = 1
                out[r, c] = int(plus_color)
                for dr2 in (-1, 0, 1):
                    for dc2 in (-1, 0, 1):
                        rr = r + dr2
                        cc = c + dc2
                        if 0 <= rr < H and 0 <= cc < W:
                            plus_guard[rr, cc] = 1

            for r, c, v in cells:
                occ[r, c] = 2
                out[r, c] = int(v)
            for r, c, _ in cells:
                ray_guard[r, c] = 1
                for dr2, dc2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr = r + dr2
                    cc = c + dc2
                    if 0 <= rr < H and 0 <= cc < W:
                        ray_guard[rr, cc] = 1

            return {
                "tile": tile,
                "cells": cells,
                "seed_r0": int(sr0),
                "seed_c0": int(sc0),
                "direction": tuple(map(int, direction)),
                "steps": int(_max_steps),
            }

    return None


def _build_input_from_output(out: np.ndarray, pluses: list[dict], bg: int) -> np.ndarray:
    """Erase every ray tail so the input keeps only compact corner seeds."""
    inp = out.copy()
    for p in pluses:
        th, tw = p["tile"].shape
        sr0 = int(p["seed_r0"])
        sc0 = int(p["seed_c0"])
        seed_set = {(sr0 + ii, sc0 + jj) for ii in range(th) for jj in range(tw)}
        for r, c, _ in p["cells"]:
            if (int(r), int(c)) not in seed_set:
                inp[int(r), int(c)] = int(bg)
    return inp


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [c for c in range(10) if c != bg]
    plus_color = int(rng.choice(others))
    palette = [c for c in others if c != plus_color]

    plus_shape = _PLUS_SHAPES[int(rng.integers(0, len(_PLUS_SHAPES)))]
    # Keep 2-4 anchors per grid so the diagonal rule is supported by multiple objects.
    n_plus = int(rng.integers(_N_PLUS_RANGE[0], _N_PLUS_RANGE[1] + 1))

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    plus_guard = np.zeros((H, W), dtype=np.uint8)
    ray_guard = np.zeros((H, W), dtype=np.uint8)

    pluses: list[dict] = []
    for _ in range(n_plus):
        info = _place_plus_and_ray(
            rng,
            plus_shape,
            palette,
            H,
            W,
            occ,
            plus_guard,
            ray_guard,
            out,
            plus_color,
        )
        if info is None:
            return None
        pluses.append(info)

    # Avoid grids that collapse to a single diagonal branch. Requiring several
    # distinct directions makes the fused diagonal rule more directly observable
    # from one example and gives more even coverage across the four branches.
    required_distinct = min(max(_MIN_DISTINCT_DIRS, n_plus - 1), n_plus)
    distinct_dirs = {tuple(map(int, p["direction"])) for p in pluses}
    if len(distinct_dirs) < required_distinct:
        return None

    inp = _build_input_from_output(out, pluses, bg)

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after %d attempts" % _GEN_RETRIES)
