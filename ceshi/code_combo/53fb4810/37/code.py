# Combo variant of 53fb4810 (index 37)
# combo_spec: D12/0+D17/0
# Name: 墙壁障碍射线碰撞

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, get_bg, label_mask


# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior trace:
#    Original 53fb4810 lets every seed tile repeat to the boundary independently.
#    Here one long ray is already visible in the input and must be treated as
#    "paint that already exists". Example: a vertical sample ray A is present,
#    a horizontal seed B would hit A, and a lower vertical seed C would hit B
#    if B is painted first. Static wall bars may also lie farther along the same
#    path, so the actual stop cell is whichever blocker is encountered first in
#    the chosen processing order.
# b) Complexity calibration:
#    The fusion would be too weak if walls existed but no ray ever met another
#    ray, or if collisions existed but walls never mattered. The generator
#    therefore always creates:
#    - one sample ray stopped by a wall,
#    - one later ray stopped by the sample ray,
#    - one later ray that visibly extends, then hits an earlier ray while a
#      farther wall remains on the same path.
# c) Implicit assumptions broken by the combo:
#    - Original solver assumed every ray can ignore earlier rays. That is false:
#      the output now depends on the evolving painted state.
#    - Original solver assumed a marker component's whole bbox is the seed. That
#      fails for the pre-extended clue ray, so we recover the seed from the
#      nozzle-adjacent 2-step slab only.
#    - Original solver ignored obstacle colors. We now detect a dedicated wall
#      color and exclude it from marker components.
# d) Placement legality:
#    The generator builds the output first from a clean board, places pluses,
#    straight wall bars, then simulates rays in the intended order. It only
#    keeps layouts where all objects fit, seeds stay attached, and no accidental
#    seed/plus overlap occurs.
# e) Visual observability:
#    Order is based on visible top-to-bottom / left-to-right anchor positions
#    after the already-drawn clue ray. No color-id ordering or hidden metadata is
#    used. Walls are monochrome bars; rays are alternating 2-step patterns, so
#    the two roles are visually distinct.
# f) Visual cleanliness:
#    The canonical layout uses separated lanes: the sample ray, the colliding
#    horizontal ray, and the dependent vertical ray occupy clear corridors. Wall
#    bars are offset from plus bodies and do not sit on top of seed tiles.
# g) New solver assumptions and generator guarantees:
#    - all pluses share one dominant color and classic cross-style topology;
#    - every seed period is exactly 2 cells along the ray axis;
#    - exactly one attached marker component is longer than its seed slab;
#    - the wall color appears as straight 1-cell-thick bars of length >= 4;
#    - the dependent third ray paints at least one cell before colliding.
#    The generator enforces each of these so `main()` never needs ambiguous
#    tie-breaking on hidden factors.


def detect_dominant_big_component_color(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    min_h: int = 3,
    min_w: int = 3,
    connectivity: int = 4,
) -> int | None:
    """Choose the non-bg color with the most large connected components."""
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    best_key: tuple[int, int] | None = None
    best_col: int | None = None

    for col in sorted(set(map(int, grid.flatten())) - {bg}):
        lab, n = label_mask(grid == int(col), connectivity=int(connectivity))
        big_areas: list[int] = []
        for k in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(k))
            if pts.size == 0:
                continue
            r0, c0, r1, c1 = bbox_coords(pts)
            if int(r1 - r0 + 1) >= int(min_h) and int(c1 - c0 + 1) >= int(min_w):
                big_areas.append(int(pts.shape[0]))
        if not big_areas:
            continue
        key = (int(len(big_areas)), int(sum(big_areas)))
        if best_key is None or key > best_key:
            best_key = key
            best_col = int(col)

    return int(best_col) if best_col is not None else None


def detect_wall_color(grid: np.ndarray, bg: int, plus_col: int) -> int | None:
    """Detect the dedicated wall color: straight 1-cell-thick bars."""
    best: tuple[int, int] | None = None
    best_col: int | None = None

    for col in sorted(set(map(int, grid.flatten())) - {int(bg), int(plus_col)}):
        lab, n = label_mask(grid == int(col), connectivity=4)
        total_len = 0
        n_lines = 0
        for k in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(k))
            if pts.size == 0:
                continue
            r0, c0, r1, c1 = bbox_coords(pts)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            area = int(pts.shape[0])
            if area == h * w and ((h == 1 and w >= 4) or (w == 1 and h >= 4)):
                total_len += area
                n_lines += 1
        if n_lines == 0:
            continue
        key = (int(total_len), int(n_lines))
        if best is None or key > best:
            best = key
            best_col = int(col)

    return int(best_col) if best_col is not None else None


def _adjacent_component_ids(
    plus_mask: np.ndarray, mark_lab: np.ndarray
) -> set[int]:
    ids: set[int] = set()
    H, W = plus_mask.shape
    for r, c in np.argwhere(plus_mask):
        r = int(r)
        c = int(c)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < H and 0 <= cc < W and int(mark_lab[rr, cc]) > 0:
                ids.add(int(mark_lab[rr, cc]))
    return ids


def _direction_from_component(plus_mask: np.ndarray, pts: np.ndarray) -> tuple[int, int] | None:
    dirs: set[tuple[int, int]] = set()
    H, W = plus_mask.shape
    for pr, pc in pts:
        pr = int(pr)
        pc = int(pc)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = pr + dr
            cc = pc + dc
            if 0 <= rr < H and 0 <= cc < W and bool(plus_mask[rr, cc]):
                dirs.add((-int(dr), -int(dc)))
    if len(dirs) != 1:
        return None
    return next(iter(dirs))


def _seed_bbox_for_plus(
    plus_mask: np.ndarray, direction: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Return the 2-step seed slab: (r0, c0, h, w)."""
    pts = np.argwhere(plus_mask)
    pr0, pc0, pr1, pc1 = bbox_coords(pts)
    dr, dc = map(int, direction)

    if dc == 0:
        nozzle_row = int(pr1) if dr == 1 else int(pr0)
        nozzle_cols = [int(c) for c in range(int(pc0), int(pc1) + 1) if bool(plus_mask[nozzle_row, c])]
        c0 = int(min(nozzle_cols))
        w = int(len(nozzle_cols))
        h = 2
        r0 = int(pr1 + 1) if dr == 1 else int(pr0 - h)
    else:
        nozzle_col = int(pc1) if dc == 1 else int(pc0)
        nozzle_rows = [int(r) for r in range(int(pr0), int(pr1) + 1) if bool(plus_mask[r, nozzle_col])]
        r0 = int(min(nozzle_rows))
        h = int(len(nozzle_rows))
        w = 2
        c0 = int(pc1 + 1) if dc == 1 else int(pc0 - w)

    return int(r0), int(c0), int(h), int(w)


def _extend_repeating_seed(
    out: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    bg: int,
) -> dict:
    """Extend one ray until the next strip would hit any non-bg cell."""
    H, W = out.shape
    h, w = tile.shape
    dr, dc = map(int, direction)
    painted: list[tuple[int, int]] = []
    stop_kind = "boundary"
    stop_cell: tuple[int, int] | None = None

    def stripe_clear_vertical(rr: int) -> bool:
        for cc in range(int(c0), int(c0 + w)):
            if int(out[rr, cc]) != int(bg):
                return False
        return True

    def stripe_clear_horizontal(cc: int) -> bool:
        for rr in range(int(r0), int(r0 + h)):
            if int(out[rr, cc]) != int(bg):
                return False
        return True

    if (dr, dc) == (1, 0):
        k = 0
        rr = int(r0 + h)
        while rr < H:
            if not stripe_clear_vertical(int(rr)):
                stop_kind = "occupied"
                stop_cell = (int(rr), int(c0))
                break
            row_vals = tile[int(k % h), :]
            for j in range(w):
                out[int(rr), int(c0 + j)] = int(row_vals[j])
                painted.append((int(rr), int(c0 + j)))
            rr += 1
            k += 1
    elif (dr, dc) == (-1, 0):
        k = 1
        rr = int(r0 - 1)
        while rr >= 0:
            if not stripe_clear_vertical(int(rr)):
                stop_kind = "occupied"
                stop_cell = (int(rr), int(c0))
                break
            row_vals = tile[int((-k) % h), :]
            for j in range(w):
                out[int(rr), int(c0 + j)] = int(row_vals[j])
                painted.append((int(rr), int(c0 + j)))
            rr -= 1
            k += 1
    elif (dr, dc) == (0, 1):
        k = 0
        cc = int(c0 + w)
        while cc < W:
            if not stripe_clear_horizontal(int(cc)):
                stop_kind = "occupied"
                stop_cell = (int(r0), int(cc))
                break
            col_vals = tile[:, int(k % w)]
            for i in range(h):
                out[int(r0 + i), int(cc)] = int(col_vals[i])
                painted.append((int(r0 + i), int(cc)))
            cc += 1
            k += 1
    elif (dr, dc) == (0, -1):
        k = 1
        cc = int(c0 - 1)
        while cc >= 0:
            if not stripe_clear_horizontal(int(cc)):
                stop_kind = "occupied"
                stop_cell = (int(r0), int(cc))
                break
            col_vals = tile[:, int((-k) % w)]
            for i in range(h):
                out[int(r0 + i), int(cc)] = int(col_vals[i])
                painted.append((int(r0 + i), int(cc)))
            cc -= 1
            k += 1
    else:
        raise ValueError("direction must be cardinal")

    return {"painted": painted, "stop_kind": stop_kind, "stop_cell": stop_cell}


def _collect_ray_specs(
    g: np.ndarray, bg: int, plus_col: int, wall_col: int | None
) -> list[dict]:
    plus_lab, pn = label_mask(g == int(plus_col), connectivity=4)
    mask = (g != int(bg)) & (g != int(plus_col))
    if wall_col is not None:
        mask &= g != int(wall_col)
    mark_lab, _ = label_mask(mask, connectivity=4)

    specs: list[dict] = []
    for pk in range(1, int(pn) + 1):
        plus_mask = plus_lab == int(pk)
        plus_pts = np.argwhere(plus_mask)
        if plus_pts.size == 0:
            continue
        adj = sorted(_adjacent_component_ids(plus_mask, mark_lab))
        if len(adj) != 1:
            continue
        cid = int(adj[0])
        pts = np.argwhere(mark_lab == int(cid))
        direction = _direction_from_component(plus_mask, pts)
        if direction is None:
            continue
        sr0, sc0, sh, sw = _seed_bbox_for_plus(plus_mask, direction)
        tile = g[int(sr0) : int(sr0 + sh), int(sc0) : int(sc0 + sw)].copy()
        pr0, pc0, pr1, pc1 = bbox_coords(plus_pts)
        specs.append(
            {
                "plus_bbox": (int(pr0), int(pc0), int(pr1), int(pc1)),
                "direction": tuple(map(int, direction)),
                "seed_r0": int(sr0),
                "seed_c0": int(sc0),
                "tile": tile,
                "component_area": int(pts.shape[0]),
            }
        )
    return specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat each attached marker pattern away from its '+' source, but now walls and earlier rays can terminate later rays.

    Input structure:
    - The most common color is background, and one dominant large color forms several identical '+' anchors.
    - A separate color forms static straight wall bars.
    - Each '+' has one attached 2-step seed slab, and exactly one attached component is already a long fully drawn ray.

    Rule:
    - Treat the long pre-drawn ray as already existing paint.
    - For every remaining seed, use its touching side to determine the outward direction and keep repeating the 2-step seed periodically.
    - Process the remaining '+' anchors in top-to-bottom, then left-to-right order.
    - A ray stops before the next strip that would hit a wall, a '+' pixel, or any ray cells already present from the input or from earlier processed seeds.

    Key insight:
    - The visible long ray is both a pattern clue and the first occupied obstacle: later rays are not independent, so wall blocking and ray-ray collisions must be resolved against the evolving painted grid.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    plus_col = detect_dominant_big_component_color(g, bg=bg, min_h=3, min_w=3, connectivity=4)
    if plus_col is None:
        return out

    wall_col = detect_wall_color(g, bg=int(bg), plus_col=int(plus_col))
    specs = _collect_ray_specs(g, bg=int(bg), plus_col=int(plus_col), wall_col=wall_col)
    if not specs:
        return out

    pre_idx = None
    for i, spec in enumerate(specs):
        if int(spec["component_area"]) > int(spec["tile"].size):
            if pre_idx is None or int(spec["component_area"]) > int(specs[pre_idx]["component_area"]):
                pre_idx = int(i)

    order = [specs[i] for i in range(len(specs)) if i != pre_idx]
    order.sort(key=lambda s: (int(s["plus_bbox"][0]), int(s["plus_bbox"][1])))

    for spec in order:
        _extend_repeating_seed(
            out,
            np.asarray(spec["tile"], dtype=int),
            int(spec["seed_r0"]),
            int(spec["seed_c0"]),
            tuple(spec["direction"]),
            bg=int(bg),
        )

    return out


# --------------- generator ---------------

_PLUS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
_H_RANGE = (18, 25)
_W_RANGE = (18, 25)
_GEN_RETRIES = 200


def _paint_plus(grid: np.ndarray, r0: int, c0: int, color: int) -> None:
    rr, cc = np.where(_PLUS == 1)
    grid[rr + int(r0), cc + int(c0)] = int(color)


def _seed_bbox_from_plus_rc(
    plus_r0: int, plus_c0: int, direction: tuple[int, int], cross: int = 1
) -> tuple[int, int, int, int]:
    plus_mask = np.zeros((plus_r0 + 3, plus_c0 + 3), dtype=bool)
    rr, cc = np.where(_PLUS == 1)
    plus_mask[rr + int(plus_r0), cc + int(plus_c0)] = True
    return _seed_bbox_for_plus(plus_mask, direction)


def _stamp_seed(
    grid: np.ndarray,
    plus_r0: int,
    plus_c0: int,
    direction: tuple[int, int],
    tile: np.ndarray,
) -> tuple[int, int]:
    sr0, sc0, sh, sw = _seed_bbox_from_plus_rc(int(plus_r0), int(plus_c0), direction)
    assert tile.shape == (sh, sw)
    grid[int(sr0) : int(sr0 + sh), int(sc0) : int(sc0 + sw)] = tile
    return int(sr0), int(sc0)


def _occupied_reason(v: int, bg: int, plus_color: int, wall_color: int) -> str:
    if int(v) == int(bg):
        return "bg"
    if int(v) == int(plus_color):
        return "plus"
    if int(v) == int(wall_color):
        return "wall"
    return "ray"


def _build_scene(
    rng: np.random.Generator, *, want_meta: bool = False
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    if H < 18 or W < 18:
        return None

    palette = list(range(10))
    bg = int(rng.choice(palette))
    palette.remove(bg)
    plus_color = int(rng.choice(palette))
    palette.remove(plus_color)
    wall_color = int(rng.choice(palette))
    palette.remove(wall_color)

    if len(palette) < 6:
        return None
    ray_cols = list(rng.choice(palette, size=6, replace=False))
    tile_a = np.array([[int(ray_cols[0])], [int(ray_cols[1])]], dtype=int)
    tile_b = np.array([[int(ray_cols[2]), int(ray_cols[3])]], dtype=int)
    tile_c = np.array([[int(ray_cols[4])], [int(ray_cols[5])]], dtype=int)

    a_col = int(rng.integers(8, W - 3))
    rb_lo = 6
    rb_hi = H - 11
    if rb_lo > rb_hi:
        return None
    rb = int(rng.integers(rb_lo, rb_hi + 1))
    # Leave one empty stripe above the third seed so its collision is visible.
    rc = int(rb + 5)
    cc = int(a_col - 4)
    if cc < 4 or rc + 2 >= H - 3:
        return None

    plus_a = (1, int(a_col - 1))
    plus_b = (int(rb), 1)
    plus_c = (int(rc), int(cc))

    wall_top_row = int(rb - 2)
    wall_bottom_row = int(H - 3)
    wall_top = [(int(wall_top_row), int(c)) for c in range(int(cc - 1), int(a_col))]
    wall_bottom = [(int(wall_bottom_row), int(c)) for c in range(int(a_col - 2), int(a_col + 3))]

    out = np.full((H, W), bg, dtype=int)
    for r, c in wall_top + wall_bottom:
        out[int(r), int(c)] = int(wall_color)

    _paint_plus(out, plus_a[0], plus_a[1], plus_color)
    _paint_plus(out, plus_b[0], plus_b[1], plus_color)
    _paint_plus(out, plus_c[0], plus_c[1], plus_color)

    seed_a = _stamp_seed(out, plus_a[0], plus_a[1], (1, 0), tile_a)
    seed_b = _stamp_seed(out, plus_b[0], plus_b[1], (0, 1), tile_b)
    seed_c = _stamp_seed(out, plus_c[0], plus_c[1], (-1, 0), tile_c)

    ray_a = _extend_repeating_seed(out, tile_a, seed_a[0], seed_a[1], (1, 0), bg=bg)
    stop_a = ray_a["stop_cell"]
    if stop_a is None or _occupied_reason(out[stop_a], bg, plus_color, wall_color) != "wall":
        return None

    ray_b = _extend_repeating_seed(out, tile_b, seed_b[0], seed_b[1], (0, 1), bg=bg)
    stop_b = ray_b["stop_cell"]
    if stop_b is None or _occupied_reason(out[stop_b], bg, plus_color, wall_color) != "ray":
        return None

    ray_c = _extend_repeating_seed(out, tile_c, seed_c[0], seed_c[1], (-1, 0), bg=bg)
    stop_c = ray_c["stop_cell"]
    if stop_c is None or _occupied_reason(out[stop_c], bg, plus_color, wall_color) != "ray":
        return None
    if len(ray_c["painted"]) < 1:
        return None

    latent_wall_hit = (int(wall_top_row), int(seed_c[1]))
    if latent_wall_hit[0] >= stop_c[0] or int(out[latent_wall_hit]) != int(wall_color):
        return None

    inp = out.copy()
    for r, c in ray_b["painted"] + ray_c["painted"]:
        inp[int(r), int(c)] = int(bg)

    if rng.random() < 0.5:
        inp = np.fliplr(inp)
        out = np.fliplr(out)

    got = main(inp)
    if not np.array_equal(got, out):
        return None

    meta = {
        "wall_stop_count": 1,
        "ray_stop_count": 2,
        "latent_wall_behind_collision": 1,
        "visible_dependent_collision": 1,
    }
    return inp, out, meta if want_meta else {}


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        built = _build_scene(rng, want_meta=False)
        if built is not None:
            inp, _out, _meta = built
            return inp
    raise RuntimeError(f"generate_input: failed after {_GEN_RETRIES} attempts")
