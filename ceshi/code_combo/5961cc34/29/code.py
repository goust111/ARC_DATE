# Combo variant of 5961cc34 (index 29)
# combo_spec: D6/0+D10/1+D17/2
# Name: 穿透激活+多干扰色块+干扰阻挡

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste


# -----------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concretization and behavior trace
# - Visual vocabulary stays close to the source task: azure background, blue
#   diamond/octagon blobs, one green edge per blob, and a yellow cap on a short
#   red bottom post.
# - The fused behavior is:
#   * D6: rays no longer stop on the first activatable blob. Once a ray enters a
#     blob through blue cells, that blob joins the relay network and the same ray
#     keeps moving.
#   * D17: a non-chain distractor blob blocks the ray if the ray reaches that
#     blob through its green side first.
#   * D10: there are now 2-3 distractor blobs, so the solver must separate
#     "pass-through active relay blob" from "blue-green object that only blocks
#     or distracts".
# - Concrete trace used by the generator:
#   * The start ray climbs one column and crosses two trunk blobs in sequence.
#     Under the original task, only the lower trunk blob would activate; here
#     both activate because the start ray penetrates the first one.
#   * Each activated trunk blob emits a horizontal thick ray.
#   * That horizontal ray ends right before a blocker blob whose green edge faces
#     the incoming ray, so the blocker disappears from the output instead of
#     turning red.
#   * An optional third distractor sits off every ray, forcing the presence of
#     multiple irrelevant blue-green blobs without adding ambiguity to which ones
#     actually participate.
#
# b) Complexity calibration
# - The generator fixes the active chain to two vertically aligned relay blobs.
#   This makes D6 visible in nearly every sample instead of relying on rare
#   incidental multi-hit lines.
# - Only trunk blobs can activate; side rays terminate in blockers rather than
#   branching again. That keeps the scene readable while still requiring the
#   solver to combine all three dimensions.
# - Distractor count is locked to 2 or 3. This is enough to make D10 observable
#   without flooding the grid with too many near-identical objects.
#
# c) Implicit-assumption review against the original main()
# - Original start-ray loop assumed "first blue hit ends the ray". That break is
#   invalid under D6 and must be removed.
# - Original thick-ray loop also stopped at first blue hit. That also breaks D6.
# - Original D17-only logic assumed any non-source green hit is a blocker. In
#   this combo that is too strong, because once a blob has already activated, its
#   green edge must be traversable as part of a pass-through chain. The solver
#   therefore treats green cells of already-activated blobs as transparent.
# - Original final assert assumed at most one distractor. That is incompatible
#   with D10 and is removed.
#
# d) Placement legality
# - The generator first places the two trunk blobs with a recursive vertical
#   layout helper so they do not overlap and the start column enters each trunk
#   blob through blue, not green.
# - Each side blocker is placed by matching the blocker green edge to the source
#   ray thickness; the red path is terminated one cell before that edge.
# - A free distractor, when present, is placed strictly outside every ray cell
#   and at least one pixel away from existing blobs via dilated occupancy.
#
# e) Visual observability
# - Across samples, the user can see one vertical red line turning two stacked
#   blobs red. That is the direct evidence for D6.
# - The same samples also show horizontal red rays stopping flush against other
#   blue-green blobs that never turn red. That is the direct evidence for D17.
# - Because 2-3 distractors exist while only two blobs activate, D10 is visible:
#   not every blue-green object on the board belongs to the relay chain.
#
# f) Visualization cleanliness
# - Trunk activation is vertical; emitted rays are horizontal only. This avoids
#   tangled multi-directional overlaps around the start column.
# - Each blocker is attached to a single emitted ray, and the optional extra
#   distractor is off-ray. The important contrast is therefore easy to parse.
#
# g) New solver assumptions and generator guarantees
# - Solver assumption: a ray may safely pass through green only if that green
#   belongs to the emitting source blob or to a blob already known to be active.
#   Generator guarantee: only relay blobs can become activated before a later ray
#   crosses their green edge.
# - Solver assumption: trunk blobs are entered from blue cells, while blockers
#   are first encountered on their green edge. Generator guarantee: trunk ports
#   are restricted to LEFT/RIGHT, and blocker entry sides are matched explicitly.
# - Solver assumption: the color roles remain recoverable from pixel counts and
#   adjacency. Generator guarantee: blue still dominates other non-background
#   colors, and green only appears adjacent to blue.
# -----------------------------------------------------------------------------


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a red relay network where chain blobs are penetrable but distractor blobs can stop rays.

    Input structure:
    - The grid is mostly background. Large blue diamond/octagon blobs each carry one short green edge that encodes a port direction and thickness.
    - A yellow cap above a short bottom red segment marks the launch point of the initial upward ray.

    Rule:
    - Repaint the start marker red and trace the upward ray. When a ray enters a blob through blue cells, that blob activates, turns red, and the same ray keeps travelling instead of stopping.
    - Activated blobs emit thick rays through their own green ports. A ray may pass through already-activated blobs, but if it first reaches an inactive distractor blob on that blob's green side, it stops immediately before the blocker.
    - Continue breadth-first until no new blobs are activated; inactive distractors are erased to background.

    Key insight: The solver must separate relay membership from raw appearance: blue-green blobs hit through blue become transparent red relays, while blue-green blobs met on their green face behave as stopping shields.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- STEP 1) Detect the bottom start marker: a vertical red post with one distinct cap pixel ---
    start_candidates: list[tuple[int, int, int, list[tuple[int, int]]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=8):
            if coords.size == 0:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            if c0 != c1 or (r1 - r0 + 1) < 2 or r1 != H - 1:
                continue
            if any(int(g[rr, c0]) != int(col) for rr in range(r0, r1 + 1)):
                continue
            if r0 - 1 < 0:
                continue
            cap = int(g[r0 - 1, c0])
            if cap == bg or cap == int(col):
                continue
            start_candidates.append(
                (
                    int(col),
                    int(cap),
                    int(c0),
                    [(int(rr), int(c0)) for rr in range(int(r0), int(r1) + 1)],
                )
            )

    assert len(start_candidates) == 1
    red_col, yellow_col, start_col, start_seg = start_candidates[0]

    # --- STEP 2) Recover blob-body color and port color from the remaining palette ---
    counts = Counter(map(int, g.flatten()))
    other_cols = [(int(cnt), int(col)) for col, cnt in counts.items() if int(col) not in (bg, red_col, yellow_col)]
    assert other_cols
    max_count = max(cnt for cnt, _col in other_cols)
    blue_candidates = [int(col) for cnt, col in other_cols if cnt == max_count]
    assert len(blue_candidates) == 1
    blue_col = int(blue_candidates[0])

    green_candidates: list[int] = []
    for col in colors_present(g, exclude={bg, red_col, yellow_col, blue_col}):
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue
        if any(
            0 <= int(r + dr) < H
            and 0 <= int(c + dc) < W
            and int(g[int(r + dr), int(c + dc)]) == int(blue_col)
            for r, c in pts
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
        ):
            green_candidates.append(int(col))

    assert len(green_candidates) == 1
    green_col = int(green_candidates[0])

    out = np.full_like(g, int(bg))

    # --- STEP 3) Extract each blue component and recover its green-edge geometry ---
    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []
    for coords in components_mask(g == int(blue_col), connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = int(len(blobs))
        for r, c in pts:
            blob_id[int(r), int(c)] = bi
        blobs.append({"pts": pts, "ports": []})

    green_owner: dict[tuple[int, int], int] = {}
    for bi, blob in enumerate(blobs):
        pts = blob["pts"]
        seen_dirs: set[tuple[int, int]] = set()
        for r, c in pts:
            for ddr, ddc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = int(r + ddr)
                cc = int(c + ddc)
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(green_col):
                    seen_dirs.add((int(ddr), int(ddc)))
        for ddr, ddc in seen_dirs:
            port_pixels = [
                (int(r), int(c))
                for r, c in pts
                if 0 <= int(r + ddr) < H and 0 <= int(c + ddc) < W and int(g[int(r + ddr), int(c + ddc)]) == int(green_col)
            ]
            green_pixels = [(int(r + ddr), int(c + ddc)) for r, c in port_pixels]
            assert port_pixels
            blob["ports"].append({"dir": (int(ddr), int(ddc)), "pixels": port_pixels, "green": green_pixels})
            for gp in green_pixels:
                green_owner[(int(gp[0]), int(gp[1]))] = int(bi)

    activated: set[int] = set()
    queue: list[int] = []

    def activate_blob_at(r: int, c: int) -> int:
        """Turn a blob red once; activated blobs become traversable on later rays."""
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in activated:
            return bi
        activated.add(int(bi))
        queue.append(int(bi))
        for pr, pc in blobs[bi]["pts"]:
            out[int(pr), int(pc)] = int(red_col)
        return bi

    def trace_ray(
        starts: list[tuple[int, int]],
        d: tuple[int, int],
        *,
        include_origin: bool,
        source_blob: int | None,
    ) -> None:
        """
        Trace one thin or thick ray.

        Source-port green pixels are always traversable. Any green cell belonging
        to an already activated blob is also traversable; other green-edge hits
        are blockers and terminate the ray before that cell.
        """
        ddr, ddc = map(int, d)
        allowed_green: set[tuple[int, int]] = set()
        if source_blob is not None:
            for port in blobs[source_blob]["ports"]:
                if tuple(port["dir"]) == (ddr, ddc):
                    allowed_green.update((int(r), int(c)) for r, c in port["green"])

        for sr, sc in starts:
            r, c = int(sr), int(sc)
            if include_origin:
                out[r, c] = int(red_col)
            while True:
                nr = int(r + ddr)
                nc = int(c + ddc)
                if not (0 <= nr < H and 0 <= nc < W):
                    break

                cell = int(g[nr, nc])
                owner = green_owner.get((nr, nc))
                if cell == int(green_col):
                    if (nr, nc) in allowed_green or (owner is not None and int(owner) in activated):
                        out[nr, nc] = int(red_col)
                        r, c = nr, nc
                        continue
                    break

                out[nr, nc] = int(red_col)
                if cell == int(blue_col):
                    activate_blob_at(nr, nc)
                r, c = nr, nc

    # --- STEP 4) Paint the start marker and launch the initial thin ray upward ---
    for r, c in start_seg:
        out[int(r), int(c)] = int(red_col)
    top_r = min(int(r) for r, _c in start_seg)
    assert top_r > 0 and int(g[top_r - 1, int(start_col)]) == int(yellow_col)
    out[top_r - 1, int(start_col)] = int(red_col)
    trace_ray([(int(top_r - 1), int(start_col))], (-1, 0), include_origin=False, source_blob=None)

    # --- STEP 5) BFS propagation: each activated blob emits from its own green side ---
    while queue:
        bi = int(queue.pop(0))
        for port in blobs[bi]["ports"]:
            trace_ray(port["pixels"], tuple(port["dir"]), include_origin=False, source_blob=bi)

    return out


# --------------- generator ---------------

BG = 8
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)
OPP = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

SHAPES = ((3, 4), (4, 4), (5, 4), (5, 6), (6, 6), (8, 8))
_SHAPE_WIDTHS = {
    (3, 4): (2, 4, 2),
    (4, 4): (2, 4, 4, 2),
    (5, 4): (2, 4, 4, 4, 2),
    (5, 6): (2, 4, 6, 4, 2),
    (6, 6): (2, 4, 6, 6, 4, 2),
    (8, 8): (2, 4, 6, 8, 8, 6, 4, 2),
}

H_MIN, H_MAX = 18, 27
W_MIN, W_MAX = 18, 27
START_COL_PAD = 5
START_H_MIN, START_H_MAX = 2, 3
START_BG_GAP = 2
TRUNK_BG_GAP = 1
ACTIVE_COUNT = 2
DISTRACTOR_COUNT_CHOICES = (2, 3)
GEN_RETRIES = 500


def _shape_mask(h: int, w: int) -> np.ndarray:
    widths = _SHAPE_WIDTHS[(h, w)]
    mask = np.zeros((h, w), dtype=bool)
    for r, ww in enumerate(widths):
        c0 = (w - ww) // 2
        mask[r, c0 : c0 + ww] = True
    return mask


def _perp_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    return rc[1] if d[0] else rc[0]


def _along_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    return rc[0] if d[0] else rc[1]


def _sort_on_ray(coords: list[tuple[int, int]], d: tuple[int, int]) -> list[tuple[int, int]]:
    return sorted(coords, key=lambda p: _perp_value(p, d))


def _edge_offsets(mask: np.ndarray, d: tuple[int, int]) -> list[tuple[int, int]]:
    pts = np.argwhere(mask)
    axis = 0 if d in (UP, DOWN) else 1
    extreme = pts[:, axis].min() if d in (UP, LEFT) else pts[:, axis].max()
    keep = pts[:, axis] == extreme
    return _sort_on_ray([(int(r), int(c)) for r, c in pts[keep]], d)


def _blob_spec(size: tuple[int, int], port_dir: tuple[int, int]) -> dict | None:
    h, w = size
    mask = _shape_mask(h, w)
    green = _edge_offsets(mask, port_dir)
    if len(green) not in (1, 2):
        return None

    green_set = set(green)
    patch = np.full((h, w), BG, dtype=int)
    patch[mask] = BLUE
    for r, c in green:
        patch[r, c] = GREEN

    port_pixels: list[tuple[int, int]] = []
    for gr, gc in green:
        br = int(gr - port_dir[0])
        bc = int(gc - port_dir[1])
        if not (0 <= br < h and 0 <= bc < w):
            return None
        if not mask[br, bc] or (br, bc) in green_set:
            return None
        port_pixels.append((int(br), int(bc)))

    blue = [(int(r), int(c)) for r, c in np.argwhere(mask) if (int(r), int(c)) not in green_set]
    full = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    return {
        "size": (h, w),
        "mask": mask,
        "patch": patch,
        "blue": blue,
        "full": full,
        "port_dir": port_dir,
        "green": _sort_on_ray([(int(r), int(c)) for r, c in green], port_dir),
        "port_pixels": _sort_on_ray(port_pixels, port_dir),
    }


def _all_specs(port_dir: tuple[int, int] | None = None) -> list[dict]:
    dirs = [port_dir] if port_dir is not None else list(DIRS)
    out = [spec for size in SHAPES for d in dirs if (spec := _blob_spec(size, d)) is not None]
    random.shuffle(out)
    return out


def _trunk_specs() -> list[dict]:
    return [spec for spec in _all_specs() if spec["port_dir"] in (LEFT, RIGHT)]


def _dilate8(mask: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r0, r1 = max(0, dr), H + min(0, dr)
            c0, c1 = max(0, dc), W + min(0, dc)
            out[r0:r1, c0:c1] |= mask[r0 - dr : r1 - dr, c0 - dc : c1 - dc]
    return out


def _offset(offsets: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(int(r0 + r), int(c0 + c)) for r, c in offsets]


def _in_bounds(coords: list[tuple[int, int]], H: int, W: int) -> bool:
    return all(0 <= int(r) < H and 0 <= int(c) < W for r, c in coords)


def _trace_to_boundary(
    starts: list[tuple[int, int]], d: tuple[int, int], include_origin: bool, H: int, W: int
) -> list[tuple[int, int]]:
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for r, c in starts:
        r = int(r)
        c = int(c)
        if include_origin:
            cells.append((r, c))
        while True:
            r += int(dr)
            c += int(dc)
            if not (0 <= r < H and 0 <= c < W):
                break
            cells.append((int(r), int(c)))
    return cells


def _path_to_targets(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    targets: list[tuple[int, int]],
    include_origin: bool,
    H: int,
    W: int,
) -> list[tuple[int, int]] | None:
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for (r, c), (tr, tc) in zip(starts, targets):
        r = int(r)
        c = int(c)
        tr = int(tr)
        tc = int(tc)
        if include_origin:
            cells.append((r, c))
        guard = H + W + 10
        while (r, c) != (tr, tc):
            r += int(dr)
            c += int(dc)
            guard -= 1
            if guard < 0 or not (0 <= r < H and 0 <= c < W):
                return None
            cells.append((int(r), int(c)))
    return cells


def _mark_ray(out: np.ndarray, ray_mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    paint_coords(out, cells, RED)
    paint_coords(ray_mask, cells, True)


def _place_blob(inp: np.ndarray, occupied: np.ndarray, blue_mask: np.ndarray, blob: dict) -> None:
    paste(inp, blob["spec"]["patch"], blob["r0"], blob["c0"], transparent=BG)
    for r, c in blob["full"]:
        occupied[int(r), int(c)] = True
    for r, c in blob["blue"]:
        blue_mask[int(r), int(c)] = True


def _place_free_distractor(
    inp: np.ndarray, occupied: np.ndarray, blue_mask: np.ndarray, ray_mask: np.ndarray
) -> bool:
    H, W = inp.shape
    occ_pad = _dilate8(occupied)
    candidates: list[dict] = []
    for spec in _all_specs():
        h, w = spec["mask"].shape
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                full = _offset(spec["full"], r0, c0)
                if any(occ_pad[r, c] or ray_mask[r, c] for r, c in full):
                    continue
                candidates.append(
                    {
                        "spec": spec,
                        "r0": int(r0),
                        "c0": int(c0),
                        "full": full,
                        "blue": _offset(spec["blue"], r0, c0),
                    }
                )
    if not candidates:
        return False
    _place_blob(inp, occupied, blue_mask, random.choice(candidates))
    return True


def _find_trunk_layout(
    H: int, W: int, start_col: int, cap_r: int, active_count: int, occupied: np.ndarray
) -> list[dict] | None:
    specs = _trunk_specs()
    initial_limit = cap_r - START_BG_GAP - 1
    if initial_limit < 0:
        return None

    def dfs(entry_limit: int, remaining: int, occ: np.ndarray) -> list[dict] | None:
        if remaining == 0:
            return []
        occ_pad = _dilate8(occ)
        candidates: list[dict] = []
        for spec in specs:
            h, w = spec["mask"].shape
            for entry_local in _edge_offsets(spec["mask"], DOWN):
                c0 = int(start_col - entry_local[1])
                if not (0 <= c0 <= W - w):
                    continue
                min_entry_row = int(entry_local[0])
                if min_entry_row > entry_limit:
                    continue
                rows = list(range(min_entry_row, entry_limit + 1))
                random.shuffle(rows)
                for entry_row in rows:
                    r0 = int(entry_row - entry_local[0])
                    full = _offset(spec["full"], r0, c0)
                    if any(occ_pad[r, c] for r, c in full):
                        continue
                    candidates.append(
                        {
                            "spec": spec,
                            "r0": int(r0),
                            "c0": int(c0),
                            "full": full,
                            "blue": _offset(spec["blue"], r0, c0),
                            "ports": _sort_on_ray(_offset(spec["port_pixels"], r0, c0), spec["port_dir"]),
                        }
                    )
        random.shuffle(candidates)
        for cand in candidates:
            occ2 = occ.copy()
            for r, c in cand["full"]:
                occ2[int(r), int(c)] = True
            next_limit = int(cand["r0"] - TRUNK_BG_GAP - 1)
            rest = dfs(next_limit, remaining - 1, occ2)
            if rest is not None:
                return [cand] + rest
        return None

    return dfs(initial_limit, active_count, occupied.copy())


def _try_place_blocker_along_axis(
    spec: dict,
    window: list[tuple[int, int]],
    d: tuple[int, int],
    lane_vals: list[int],
    starts: list[tuple[int, int]],
    include_origin: bool,
    occ_pad: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
    H: int,
    W: int,
) -> list[dict]:
    h, w = spec["mask"].shape
    win_vals = [_perp_value(p, d) for p in window]
    edge_line = _along_value(window[0], d)
    start_line = _along_value(starts[0], d)
    is_vertical = bool(d[0])

    fixed = int(lane_vals[0] - win_vals[0])
    extent = H - h if is_vertical else W - w
    low = int(edge_line)
    high = int(extent + edge_line)
    if d in (UP, LEFT):
        high = min(high, int(start_line - 3))
    else:
        low = max(low, int(start_line + 3))
    if low > high:
        return []

    entry_lines = list(range(low, high + 1))
    random.shuffle(entry_lines)
    candidates: list[dict] = []

    for entry_line in entry_lines:
        origin_along = int(entry_line - edge_line)
        r0 = int(origin_along if is_vertical else fixed)
        c0 = int(fixed if is_vertical else origin_along)
        if is_vertical:
            entry_cells = [(int(entry_line), int(c0 + c)) for _, c in window]
        else:
            entry_cells = [(int(r0 + r), int(entry_line)) for r, _ in window]
        stop_cells = [(int(r - d[0]), int(c - d[1])) for r, c in entry_cells]
        full = _offset(spec["full"], r0, c0)
        if not _in_bounds(full, H, W) or not _in_bounds(stop_cells, H, W):
            continue
        if any(occ_pad[r, c] or ray_mask[r, c] for r, c in full):
            continue
        if any(occ_pad[r, c] for r, c in stop_cells):
            continue
        path = _path_to_targets(starts, d, stop_cells, include_origin, H, W)
        if path is None:
            continue
        if any(blue_mask[r, c] for r, c in path):
            continue
        candidates.append(
            {
                "spec": spec,
                "r0": int(r0),
                "c0": int(c0),
                "entry": entry_cells,
                "stop": stop_cells,
                "path": path,
                "blue": _offset(spec["blue"], r0, c0),
                "full": full,
            }
        )

    return candidates


def _candidate_blocker(
    H: int,
    W: int,
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
) -> dict | None:
    starts = _sort_on_ray(starts, d)
    thickness = len(starts)
    lane_vals = [_perp_value(p, d) for p in starts]
    lane_delta = [v - lane_vals[0] for v in lane_vals]
    entry_side = OPP[d]
    occ_pad = _dilate8(occupied)
    candidates: list[dict] = []

    for spec in _all_specs(entry_side):
        edge = spec["green"]
        if len(edge) < thickness:
            continue
        for k in range(len(edge) - thickness + 1):
            window = edge[k : k + thickness]
            win_delta = [_perp_value(p, d) - _perp_value(window[0], d) for p in window]
            if win_delta != lane_delta:
                continue
            candidates.extend(
                _try_place_blocker_along_axis(
                    spec,
                    window,
                    d,
                    lane_vals,
                    starts,
                    include_origin,
                    occ_pad,
                    blue_mask,
                    ray_mask,
                    H,
                    W,
                )
            )

    return random.choice(candidates) if candidates else None


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    start_h = random.randint(START_H_MIN, START_H_MAX)
    cap_r = int(H - start_h - 1)
    start_col = random.randint(START_COL_PAD, W - START_COL_PAD - 1)
    inp[cap_r, start_col] = YELLOW
    out[cap_r, start_col] = RED
    occupied[cap_r, start_col] = True
    for r in range(cap_r + 1, H):
        inp[r, start_col] = RED
        out[r, start_col] = RED
        occupied[r, start_col] = True

    trunk = _find_trunk_layout(H, W, start_col, cap_r, ACTIVE_COUNT, occupied)
    if trunk is None:
        return None

    for blob in trunk:
        _place_blob(inp, occupied, blue_mask, blob)
        paint_coords(out, blob["full"], RED)

    start_path = _trace_to_boundary([(cap_r, start_col)], UP, True, H, W)
    _mark_ray(out, ray_mask, start_path)

    blockers: list[dict] = []
    for blob in trunk:
        blocker = _candidate_blocker(
            H,
            W,
            blob["ports"],
            blob["spec"]["port_dir"],
            False,
            occupied,
            blue_mask,
            ray_mask,
        )
        if blocker is None:
            return None
        blockers.append(blocker)
        _mark_ray(out, ray_mask, blocker["path"])
        _place_blob(inp, occupied, blue_mask, blocker)

    total_distractors = random.choice(DISTRACTOR_COUNT_CHOICES)
    extra_distractors = int(total_distractors - len(blockers))
    for _ in range(extra_distractors):
        if not _place_free_distractor(inp, occupied, blue_mask, ray_mask):
            return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5961cc34 combo input")
