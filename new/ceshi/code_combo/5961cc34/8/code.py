# Combo variant of 5961cc34 (index 8)
# combo_spec: D6/1+D17/2
# Name: 碰撞双击激活+干扰色块阻挡

from __future__ import annotations

from collections import Counter
from typing import Callable
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paste


# -----------------------------------------------------------------------------
# Design analysis for the combo variant D6/1 + D17/2
#
# a) Concrete behavior
# - We keep the original vocabulary: azure bg, blue blob bodies, green port edge,
#   yellow starter cap, red start stem / rays / activated results.
# - New collision semantics:
#   * Blue-side contact no longer activates immediately. The ray stops there and
#     only the struck blue edge of that blob becomes red.
#   * A blob becomes fully active only after two distinct ray lanes have struck
#     it. Once active, the whole blue body turns red and it emits from its own
#     green side as before.
#   * Green-side contact is different again: the ray stops immediately before
#     the blob, and that blob is not marked at all in the output.
# - Bootstrap issue:
#   * A strict "every blob needs two hits" rule is impossible here because the
#     task has only one thin start ray. Without one initial seed activation, the
#     cascade would always die on the first collision.
#   * The fused design therefore treats the first start-ray hit as a required
#     seed exception. All later blue-side hits obey the real two-lane rule.
#
# b) Complexity calibration
# - The visible distinctions remain structural rather than numeric:
#   blue-side single hit -> only one red edge strip,
#   double hit -> whole blob turns red and keeps propagating,
#   green-side hit -> the ray halts in front of an erased blob.
# - To keep this readable, the generator always includes at least one true
#   downstream double-hit activation after the seed.
# - Because these blob shapes are wide and adjacent thick-ray lanes cannot cleanly
#   branch into separate terminal targets, each grid shows one terminal special
#   case at a time: either a single-hit end blob or a blocker end blob. The
#   generator alternates those modes across calls so a small example set exposes
#   both halves of the combo.
#
# c) Original assumptions that break
# - Original main() assumed first blue contact always activates. That must now
#   split into "seed hit" versus later "count hits and maybe only mark edge".
# - Original ray loops painted through any non-blue cell until a blue cell was
#   found. That breaks once foreign green edges become blockers.
# - Original generator only needed one incoming lane per activated target. The
#   combo now needs a guaranteed thickness-2 contact somewhere downstream so the
#   "needs two rays" rule is actually exercised.
#
# d) Placement legality
# - The generator still uses path-first placement with 8-neighborhood spacing.
# - The seed blob is placed above the start column and emits upward with a
#   thickness-2 port. A second blob is then placed above it so both lanes strike
#   the same blue side and genuinely satisfy the double-hit rule.
# - The final horizontal ray is then terminated by either:
#   * a normal blue-side blob that receives only one hit, or
#   * a green-side blocker blob.
#
# e) Visual observability
# - Across 2-4 examples, a human can see:
#   * one blob after the seed turning fully red only when reached by a thick
#     two-lane beam,
#   * some examples where the last target shows only a red edge strip,
#   * other examples where the last red line stops short of a vanished blob.
# - No hidden ordering by color id or off-grid metadata is needed.
#
# f) Clean visualization
# - The layout is intentionally sparse: vertical seed chain, then one horizontal
#   terminal event. This avoids clutter while still surfacing the fused logic.
# - We avoid off-ray distractors so the only non-red-end behavior is the intended
#   partial-hit or blocker phenomenon.
#
# g) Generator guarantees for solver assumptions
# - Exactly one start marker exists.
# - Exactly one seed activation comes from the initial upward ray.
# - Every downstream full activation is produced by two distinct ray lanes.
# - Partial targets are always hit through a blue side, never a green side.
# - Blockers are always hit through their green side, never their blue side.
# -----------------------------------------------------------------------------


# --- Colors ---
BG = 8
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

# --- Directions ---
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)
OPP = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

# --- Diamond/octagon shapes (h, w) ---
SHAPES = ((3, 4), (4, 4), (5, 4), (5, 6), (6, 6), (8, 8))
_SHAPE_WIDTHS = {
    (3, 4): (2, 4, 2),
    (4, 4): (2, 4, 4, 2),
    (5, 4): (2, 4, 4, 4, 2),
    (5, 6): (2, 4, 6, 4, 2),
    (6, 6): (2, 4, 6, 6, 4, 2),
    (8, 8): (2, 4, 6, 8, 8, 6, 4, 2),
}

# --- Canvas bounds ---
H_MIN, H_MAX = 13, 27
W_MIN, W_MAX = 15, 27
START_COL_PAD = 3

# --- Start segment ---
START_H_MIN, START_H_MAX = 2, 3

# --- Placement ---
MIN_GAP_FIRST = 4
MIN_GAP_CHAIN = 3

# --- Retry ---
GEN_RETRIES = 500

# --- Alternating terminal mode for observability across examples ---
_GEN_MODE_COUNTER = 0


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
        port_pixels.append((br, bc))

    blue = [(int(r), int(c)) for r, c in np.argwhere(mask) if (int(r), int(c)) not in green_set]
    full = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    return {
        "size": (h, w),
        "mask": mask,
        "patch": patch,
        "blue": blue,
        "full": full,
        "port_dir": port_dir,
        "green": _sort_on_ray(green, port_dir),
        "port_pixels": _sort_on_ray(port_pixels, port_dir),
    }


def _all_specs(
    port_dir: tuple[int, int] | None = None,
    spec_filter: Callable[[dict], bool] | None = None,
) -> list[dict]:
    dirs = [port_dir] if port_dir is not None else list(DIRS)
    out = []
    for size in SHAPES:
        for d in dirs:
            spec = _blob_spec(size, d)
            if spec is None:
                continue
            if spec_filter is not None and not spec_filter(spec):
                continue
            out.append(spec)
    random.shuffle(out)
    return out


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
    return [(r0 + r, c0 + c) for r, c in offsets]


def _in_bounds(coords: list[tuple[int, int]], H: int, W: int) -> bool:
    return all(0 <= r < H and 0 <= c < W for r, c in coords)


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
        if include_origin:
            cells.append((r, c))
        guard = H + W + 10
        while (r, c) != (tr, tc):
            r = int(r + dr)
            c = int(c + dc)
            guard -= 1
            if guard < 0 or not (0 <= r < H and 0 <= c < W):
                return None
            cells.append((r, c))
    return cells


def _mark_ray_mask(ray_mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        ray_mask[int(r), int(c)] = True


def _edge_blue_from_pts(pts: list[tuple[int, int]], d: tuple[int, int]) -> list[tuple[int, int]]:
    axis = 0 if d in (UP, DOWN) else 1
    values = [p[axis] for p in pts]
    extreme = min(values) if d in (UP, LEFT) else max(values)
    return _sort_on_ray([p for p in pts if p[axis] == extreme], d)


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- STEP 1) Detect the unique bottom start stem with a distinct cap above ---
    start_candidates: list[tuple[int, int, int, list[tuple[int, int]]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=8):
            if coords.size == 0:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            if c0 != c1 or (r1 - r0 + 1) < 2:
                continue
            if r1 != H - 1:
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
                    [(int(rr), int(c0)) for rr in range(r0, r1 + 1)],
                )
            )

    assert len(start_candidates) == 1
    red_col, yellow_col, start_col, start_seg = start_candidates[0]

    # --- STEP 2) Identify blue body color and green port color ---
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
        ok = False
        for r, c in pts:
            r = int(r)
            c = int(c)
            for dr, dc in DIRS:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == blue_col:
                    ok = True
                    break
            if ok:
                break
        if ok:
            green_candidates.append(int(col))

    assert len(green_candidates) == 1
    green_col = int(green_candidates[0])

    out = np.full_like(g, bg)

    # --- STEP 3) Extract blobs and their port geometry ---
    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []
    for coords in components_mask(g == blue_col, connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = int(len(blobs))
        for r, c in pts:
            blob_id[r, c] = bi
        edge_blue = {d: _edge_blue_from_pts(pts, d) for d in DIRS}
        blobs.append({"pts": pts, "ports": [], "edge_blue": edge_blue})

    green_owner: dict[tuple[int, int], int] = {}
    for bi, blob in enumerate(blobs):
        seen_dirs: set[tuple[int, int]] = set()
        for r, c in blob["pts"]:
            for ddr, ddc in DIRS:
                rr = int(r + ddr)
                cc = int(c + ddc)
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == green_col:
                    seen_dirs.add((int(ddr), int(ddc)))
        for ddr, ddc in seen_dirs:
            port_pixels = [
                (int(r), int(c))
                for (r, c) in blob["pts"]
                if 0 <= int(r + ddr) < H
                and 0 <= int(c + ddc) < W
                and int(g[int(r + ddr), int(c + ddc)]) == green_col
            ]
            green_pixels = [(int(r + ddr), int(c + ddc)) for r, c in port_pixels]
            assert port_pixels
            blob["ports"].append({"dir": (int(ddr), int(ddc)), "pixels": port_pixels, "green": green_pixels})
            for gp in green_pixels:
                green_owner[gp] = bi

    activated: set[int] = set()
    seed_activated: set[int] = set()
    queue: list[int] = []
    hit_rays: list[set[tuple[int, int, int, int, int]]] = [set() for _ in blobs]
    hit_sides: list[set[tuple[int, int]]] = [set() for _ in blobs]
    blocked_blobs: set[int] = set()

    def fully_activate(bi: int, *, seeded: bool) -> None:
        if bi in activated:
            return
        activated.add(int(bi))
        queue.append(int(bi))
        if seeded:
            seed_activated.add(int(bi))
        for pr, pc in blobs[bi]["pts"]:
            out[int(pr), int(pc)] = int(red_col)

    def record_blue_hit(
        bi: int,
        ray_id: tuple[int, int, int, int, int],
        entry_side: tuple[int, int],
        *,
        allow_seed: bool,
    ) -> None:
        if bi in activated:
            return
        hit_rays[bi].add(ray_id)
        hit_sides[bi].add(entry_side)
        for pr, pc in blobs[bi]["edge_blue"][entry_side]:
            out[int(pr), int(pc)] = int(red_col)
        if allow_seed:
            fully_activate(bi, seeded=True)
            return
        if len(hit_rays[bi]) >= 2:
            fully_activate(bi, seeded=False)

    def trace_ray(
        starts: list[tuple[int, int]],
        d: tuple[int, int],
        *,
        include_origin: bool,
        source_blob: int | None,
        allow_seed_hit: bool,
    ) -> None:
        ddr, ddc = map(int, d)
        allowed_green: set[tuple[int, int]] = set()
        if source_blob is not None:
            for port in blobs[source_blob]["ports"]:
                if tuple(port["dir"]) == (ddr, ddc):
                    allowed_green.update((int(r), int(c)) for r, c in port["green"])

        for sr, sc in starts:
            ray_id = (
                -1 if source_blob is None else int(source_blob),
                int(sr),
                int(sc),
                int(ddr),
                int(ddc),
            )
            r = int(sr)
            c = int(sc)
            if include_origin:
                out[r, c] = int(red_col)
            while True:
                nr = int(r + ddr)
                nc = int(c + ddc)
                if not (0 <= nr < H and 0 <= nc < W):
                    break
                if (nr, nc) in allowed_green:
                    out[nr, nc] = int(red_col)
                    r, c = nr, nc
                    continue
                cell = int(g[nr, nc])
                if cell == green_col:
                    if (nr, nc) in green_owner:
                        blocked_blobs.add(int(green_owner[(nr, nc)]))
                    break
                out[nr, nc] = int(red_col)
                if cell == blue_col:
                    bi = int(blob_id[nr, nc])
                    assert bi >= 0
                    record_blue_hit(bi, ray_id, OPP[(ddr, ddc)], allow_seed=allow_seed_hit and not activated)
                    break
                r, c = nr, nc

    # --- STEP 4) Paint the starter and launch the unique seed ray ---
    for r, c in start_seg:
        out[int(r), int(c)] = int(red_col)
    top_r = min(int(r) for r, _c in start_seg)
    assert top_r > 0 and int(g[top_r - 1, start_col]) == int(yellow_col)
    out[top_r - 1, start_col] = int(red_col)
    trace_ray([(top_r - 1, start_col)], UP, include_origin=False, source_blob=None, allow_seed_hit=True)

    # --- STEP 5) BFS propagation from fully activated blobs ---
    while queue:
        bi = int(queue.pop(0))
        for port in blobs[bi]["ports"]:
            trace_ray(port["pixels"], tuple(port["dir"]), include_origin=False, source_blob=bi, allow_seed_hit=False)

    classifications: list[str] = []
    counts_out = Counter()
    for bi in range(len(blobs)):
        if bi in seed_activated:
            label = "seed_activated"
        elif bi in activated:
            label = "double_activated"
        elif bi in blocked_blobs:
            label = "blocker"
        elif len(hit_rays[bi]) >= 1:
            label = "single_hit"
        else:
            label = "unused"
        classifications.append(label)
        counts_out[label] += 1

    meta = {
        "classifications": classifications,
        "counts": {k: int(v) for k, v in counts_out.items()},
        "hit_counts": [len(s) for s in hit_rays],
        "hit_sides": [sorted(s) for s in hit_sides],
        "activated": sorted(activated),
        "seed_activated": sorted(seed_activated),
        "blocked": sorted(blocked_blobs),
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a red cascade where downstream blobs need two ray lanes, while green-edge blobs can simply stop the beam.

    Input structure:
    - The dominant background surrounds blue diamond/octagon blobs, each carrying one green edge that shows its outgoing port side.
    - A yellow cap above a short red stem at the bottom marks the unique starting beam.

    Rule:
    - The start beam still seeds the cascade by activating the first blue blob it meets.
    - After that, blue-side collisions only mark the struck edge red; a blob emits further only after two distinct ray lanes have hit it.
    - If a ray reaches a blob through its green edge, it stops there and that blob is not marked at all.

    Key insight: The same blue-green blob can end up in three visibly different states: fully red after two hits, only a red edge after one blue-side hit, or completely absent when it acts as a green-side blocker.
    """
    out, _meta = _simulate(grid)
    return out


def summarize_behaviors(grid: np.ndarray) -> dict[str, int]:
    _out, meta = _simulate(grid)
    return dict(meta["counts"])


def _try_place_activator_along_axis(
    spec: dict,
    window: list[tuple[int, int]],
    d: tuple[int, int],
    lane_vals: list[int],
    starts: list[tuple[int, int]],
    include_origin: bool,
    min_gap: int,
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

    fixed = lane_vals[0] - win_vals[0]
    extent = H - h if is_vertical else W - w
    low = edge_line
    high = extent + edge_line
    if d in (UP, LEFT):
        high = min(high, start_line - min_gap)
    else:
        low = max(low, start_line + min_gap)

    candidates: list[dict] = []
    entry_lines = list(range(low, high + 1))
    random.shuffle(entry_lines)
    for entry_line in entry_lines:
        origin_along = entry_line - edge_line
        r0 = origin_along if is_vertical else fixed
        c0 = fixed if is_vertical else origin_along
        if is_vertical:
            entry_cells = [(entry_line, c0 + c) for _, c in window]
        else:
            entry_cells = [(r0 + r, entry_line) for r, _ in window]
        full = _offset(spec["full"], r0, c0)
        if not _in_bounds(full, H, W):
            continue
        if any(occ_pad[r, c] or ray_mask[r, c] for r, c in full):
            continue
        path = _path_to_targets(starts, d, entry_cells, include_origin, H, W)
        if path is None:
            continue
        entry_set = set(entry_cells)
        if any(blue_mask[r, c] for r, c in path if (r, c) not in entry_set):
            continue
        candidates.append(
            {
                "spec": spec,
                "r0": int(r0),
                "c0": int(c0),
                "entry": entry_cells,
                "path": path,
                "blue": _offset(spec["blue"], r0, c0),
                "full": full,
                "ports": _sort_on_ray(_offset(spec["port_pixels"], r0, c0), spec["port_dir"]),
            }
        )
    return candidates


def _try_place_blocker_along_axis(
    spec: dict,
    window: list[tuple[int, int]],
    d: tuple[int, int],
    lane_vals: list[int],
    starts: list[tuple[int, int]],
    include_origin: bool,
    min_gap: int,
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

    fixed = lane_vals[0] - win_vals[0]
    extent = H - h if is_vertical else W - w
    low = edge_line
    high = extent + edge_line
    if d in (UP, LEFT):
        high = min(high, start_line - min_gap)
    else:
        low = max(low, start_line + min_gap)

    candidates: list[dict] = []
    entry_lines = list(range(low, high + 1))
    random.shuffle(entry_lines)
    for entry_line in entry_lines:
        origin_along = entry_line - edge_line
        r0 = origin_along if is_vertical else fixed
        c0 = fixed if is_vertical else origin_along
        if is_vertical:
            entry_cells = [(entry_line, c0 + c) for _, c in window]
        else:
            entry_cells = [(r0 + r, entry_line) for r, _ in window]
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
                "ports": _sort_on_ray(_offset(spec["port_pixels"], r0, c0), spec["port_dir"]),
            }
        )
    return candidates


def _candidate_activator(
    H: int,
    W: int,
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
    spec_filter: Callable[[dict], bool] | None = None,
) -> dict | None:
    starts = _sort_on_ray(starts, d)
    thickness = len(starts)
    lane_vals = [_perp_value(p, d) for p in starts]
    lane_delta = [v - lane_vals[0] for v in lane_vals]
    entry_side = OPP[d]
    occ_pad = _dilate8(occupied)
    min_gap = MIN_GAP_FIRST if include_origin else MIN_GAP_CHAIN
    candidates: list[dict] = []

    for spec in _all_specs(spec_filter=spec_filter):
        if spec["port_dir"] == entry_side:
            continue
        edge = _edge_offsets(spec["mask"], entry_side)
        if len(edge) < thickness:
            continue
        for k in range(len(edge) - thickness + 1):
            window = edge[k : k + thickness]
            win_delta = [_perp_value(p, d) - _perp_value(window[0], d) for p in window]
            if win_delta != lane_delta:
                continue
            candidates.extend(
                _try_place_activator_along_axis(
                    spec,
                    window,
                    d,
                    lane_vals,
                    starts,
                    include_origin,
                    min_gap,
                    occ_pad,
                    blue_mask,
                    ray_mask,
                    H,
                    W,
                )
            )

    return random.choice(candidates) if candidates else None


def _candidate_blocker(
    H: int,
    W: int,
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
    spec_filter: Callable[[dict], bool] | None = None,
) -> dict | None:
    starts = _sort_on_ray(starts, d)
    thickness = len(starts)
    lane_vals = [_perp_value(p, d) for p in starts]
    lane_delta = [v - lane_vals[0] for v in lane_vals]
    entry_side = OPP[d]
    occ_pad = _dilate8(occupied)
    min_gap = MIN_GAP_FIRST if include_origin else MIN_GAP_CHAIN
    candidates: list[dict] = []

    def blocker_filter(spec: dict) -> bool:
        if spec["port_dir"] != entry_side:
            return False
        return True if spec_filter is None else spec_filter(spec)

    for spec in _all_specs(port_dir=entry_side, spec_filter=blocker_filter):
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
                    min_gap,
                    occ_pad,
                    blue_mask,
                    ray_mask,
                    H,
                    W,
                )
            )

    return random.choice(candidates) if candidates else None


def _place_blob(inp: np.ndarray, occupied: np.ndarray, blue_mask: np.ndarray, blob: dict) -> None:
    paste(inp, blob["spec"]["patch"], blob["r0"], blob["c0"], transparent=BG)
    for r, c in blob["full"]:
        occupied[int(r), int(c)] = True
    for r, c in blob["blue"]:
        blue_mask[int(r), int(c)] = True


def _seed_spec(spec: dict) -> bool:
    return spec["port_dir"] == UP and len(spec["port_pixels"]) == 2


def _double_hit_spec(spec: dict) -> bool:
    return spec["port_dir"] in (LEFT, RIGHT) and len(spec["port_pixels"]) == 1


def _partial_terminal_spec(spec: dict) -> bool:
    return spec["port_dir"] in (UP, DOWN)


def _next_terminal_mode() -> str:
    global _GEN_MODE_COUNTER
    mode = "partial" if (_GEN_MODE_COUNTER % 2 == 0) else "blocker"
    _GEN_MODE_COUNTER += 1
    return mode


def _gen_attempt(mode: str) -> np.ndarray | None:
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    inp = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    # Place start marker.
    start_h = random.randint(START_H_MIN, START_H_MAX)
    cap_r = H - start_h - 1
    start_col = random.randint(START_COL_PAD, W - START_COL_PAD - 1)
    inp[cap_r, start_col] = YELLOW
    occupied[cap_r, start_col] = True
    ray_mask[cap_r, start_col] = True
    for r in range(cap_r + 1, H):
        inp[r, start_col] = RED
        occupied[r, start_col] = True

    seed = _candidate_activator(
        H,
        W,
        [(cap_r, start_col)],
        UP,
        True,
        occupied,
        blue_mask,
        ray_mask,
        spec_filter=_seed_spec,
    )
    if seed is None:
        return None
    _mark_ray_mask(ray_mask, seed["path"])
    _place_blob(inp, occupied, blue_mask, seed)

    double_hit = _candidate_activator(
        H,
        W,
        seed["ports"],
        seed["spec"]["port_dir"],
        False,
        occupied,
        blue_mask,
        ray_mask,
        spec_filter=_double_hit_spec,
    )
    if double_hit is None:
        return None
    _mark_ray_mask(ray_mask, double_hit["path"])
    _place_blob(inp, occupied, blue_mask, double_hit)

    if mode == "partial":
        terminal = _candidate_activator(
            H,
            W,
            double_hit["ports"],
            double_hit["spec"]["port_dir"],
            False,
            occupied,
            blue_mask,
            ray_mask,
            spec_filter=_partial_terminal_spec,
        )
    else:
        terminal = _candidate_blocker(
            H,
            W,
            double_hit["ports"],
            double_hit["spec"]["port_dir"],
            False,
            occupied,
            blue_mask,
            ray_mask,
        )
    if terminal is None:
        return None
    _mark_ray_mask(ray_mask, terminal["path"])
    _place_blob(inp, occupied, blue_mask, terminal)

    try:
        _out, meta = _simulate(inp)
    except Exception:
        return None

    counts = meta["counts"]
    if counts.get("seed_activated", 0) != 1:
        return None
    if counts.get("double_activated", 0) < 1:
        return None
    if mode == "partial":
        if counts.get("single_hit", 0) < 1:
            return None
        if counts.get("blocker", 0) != 0:
            return None
    else:
        if counts.get("blocker", 0) < 1:
            return None
    return inp


def generate_input() -> np.ndarray:
    mode = _next_terminal_mode()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(mode)
        if grid is not None:
            return grid
    # Rare fallback: do not fail the whole task if one mode is temporarily tight.
    fallback = "blocker" if mode == "partial" else "partial"
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(fallback)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5961cc34 combo input")
