# Combo variant of 53fb4810 (index 8)
# combo_spec: D4/0+D11/0
# Name: 矩形框架多方向射线

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and behavioral trace
# - Original task: one solid '+'-like sprite owns exactly one attached marker; the marker's
#   attachment side both identifies the source and gives one ray direction.
# - Fused task: the source is now a hollow rectangular frame, and one frame may own 2-4
#   separate markers on different sides. Each side-attached marker is its own ray seed.
# - Concrete trace:
#   old: plus with one right seed [[2, 3]] -> repeat [[2, 3]] to the right.
#   new: hollow frame with a top seed [[2, 3]] and a left seed [[4], [6]] ->
#   repeat [[2, 3]] upward and repeat [[4], [6]] leftward, independently.
# - The key behavioral delta is that the solver must first recognize a frame component, then
#   decompose the puzzle by frame side, not by plus arm. One frame can therefore trigger
#   several rays in the same output.
#
# b) Complexity calibration
# - Marker count is capped at 2-4 because a rectangle has four sides; this keeps the variant
#   visibly multi-directional without making placement chaotic.
# - Each grid shares one frame color and one outer frame size, so the "frame instead of plus"
#   change is easy to spot across examples.
# - Seed tiles stay small (cross-axis 1-2, along-axis 1-2, avoiding 1x1) and sit flush with
#   a side's inner span. This preserves the original "small local pattern gets repeated"
#   logic instead of introducing a second abstract mechanism.
#
# c) Implicit-assumption audit of the original main()
# - The dominant big-component-color heuristic still works, but it now points to hollow frames
#   instead of dense pluses. Generator guarantees frame bboxes are at least 5x5, while rays are
#   thin strips whose bbox is never simultaneously tall and wide.
# - The old "arm-tip nozzle span" checks no longer make sense. The new solver replaces them
#   with frame validation plus side-flush checks: every marker component must sit wholly on one
#   side of exactly one frame and remain inside that side's non-corner span.
# - `mark_lab` still assumes each logical marker/ray is one isolated 4-connected component.
#   Generator therefore forbids overlap and 4-neighbor contact between distinct rays, even when
#   they belong to the same frame.
# - Because one frame now owns multiple markers, the solver must keep all adjacent component ids
#   and process each independently; it cannot short-circuit after the first match.
#
# d) Placement legality
# - Generation remains output-first: place frames, then all full rays, then erase every ray
#   except one exemplar back to its seed tile.
# - Frames do not overlap or 4-touch foreign structures. Rays may touch only their own frame
#   at the seed boundary; otherwise they cannot overlap or 4-touch any foreign frame/ray.
# - Each frame uses distinct sides for its markers, so ownership is unambiguous.
#
# e) Visual observability
# - A human sees hollow frames rather than pluses, and also sees some frames with several seed
#   patches on different sides. One long pre-expanded strip demonstrates that the local tile
#   repeats away from the touched side.
# - Direction is purely geometric: which side of the frame the marker hugs. No color ordering,
#   no numeric color semantics, and no hidden coordinate sorting are used.
#
# f) Visual cleanliness
# - Only 2-3 frames are placed, and each frame's rays are kept spatially isolated from other
#   objects. This avoids accidental mergers into cluttered bands.
# - Seeds stay off the corners, so top/left/right/bottom ownership remains visually legible.
#
# g) Inferability of the change
# - Colors are copied verbatim inside each repeated strip. The only rule to infer is spatial:
#   identify the frame, identify which side a seed sits on, then repeat that seed away from the
#   frame. The multi-marker extension is visible because the same frame can show several seeds.
#
# New solver assumptions introduced here and guaranteed by the generator
# - Every frame is a perfect 1-cell-thick hollow rectangle with no gaps.
# - Every marker/ray component is a filled rectangle whose bbox equals the component itself.
# - Every such component touches exactly one side of exactly one frame and is flush with that
#   side: top components end at `frame_top - 1`, right components start at `frame_right + 1`,
#   and analogously for the other sides.


_DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIR_TO_SIDE = {(-1, 0): "up", (1, 0): "down", (0, -1): "left", (0, 1): "right"}
_SIDE_TO_DIR = {v: k for k, v in _DIR_TO_SIDE.items()}


def detect_dominant_big_component_color(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    min_h: int = 5,
    min_w: int = 5,
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


def extend_repeating_tile(
    out: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    bg: int,
    protect: set[int] | None = None,
) -> None:
    """Extend a tile pattern from (r0,c0) until the boundary."""
    protect = set() if protect is None else set(map(int, protect))
    H, W = out.shape
    h, w = tile.shape
    dr, dc = map(int, direction)
    dr = 0 if dr == 0 else int(dr / abs(dr))
    dc = 0 if dc == 0 else int(dc / abs(dc))
    bg = int(bg)
    r0 = int(r0)
    c0 = int(c0)

    def put(rr: int, cc: int, v: int) -> None:
        if v == bg:
            return
        if int(out[rr, cc]) in protect:
            return
        out[rr, cc] = int(v)

    if (dr, dc) == (0, 1):
        start_c = c0 + w
        for k, tc in enumerate(range(start_c, W)):
            col_idx = k % w
            for i in range(h):
                put(r0 + i, tc, int(tile[i, col_idx]))
    elif (dr, dc) == (0, -1):
        for k, tc in enumerate(range(c0 - 1, -1, -1), start=1):
            col_idx = (-k) % w
            for i in range(h):
                put(r0 + i, tc, int(tile[i, col_idx]))
    elif (dr, dc) == (1, 0):
        start_r = r0 + h
        for k, tr in enumerate(range(start_r, H)):
            row_idx = k % h
            for j in range(w):
                put(tr, c0 + j, int(tile[row_idx, j]))
    elif (dr, dc) == (-1, 0):
        for k, tr in enumerate(range(r0 - 1, -1, -1), start=1):
            row_idx = (-k) % h
            for j in range(w):
                put(tr, c0 + j, int(tile[row_idx, j]))
    else:
        raise ValueError("extend_repeating_tile: direction must be one of (±1,0),(0,±1)")


def detect_hollow_frame(frame_mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the bbox of a perfect 1-cell-thick hollow rectangle, or None."""
    pts = np.argwhere(frame_mask)
    if pts.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(pts)
    local = frame_mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    h, w = local.shape
    if h < 5 or w < 5:
        return None
    if np.any(local[1:-1, 1:-1]):
        return None
    if not np.all(local[0, :]) or not np.all(local[-1, :]):
        return None
    if not np.all(local[:, 0]) or not np.all(local[:, -1]):
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _adjacent_marker_ids(
    mark_lab: np.ndarray,
    frame_mask: np.ndarray,
) -> set[int]:
    H, W = mark_lab.shape
    ids: set[int] = set()
    for r, c in np.argwhere(frame_mask):
        r = int(r)
        c = int(c)
        for dr, dc in _DIRECTIONS:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < H and 0 <= cc < W and int(mark_lab[rr, cc]) > 0:
                ids.add(int(mark_lab[rr, cc]))
    return ids


def _component_side_relative_to_frame(
    pts: np.ndarray,
    frame_mask: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
) -> tuple[str, tuple[int, int]] | None:
    dirs: set[tuple[int, int]] = set()
    H, W = frame_mask.shape
    for pr, pc in pts:
        pr = int(pr)
        pc = int(pc)
        for dr, dc in _DIRECTIONS:
            rr = pr + dr
            cc = pc + dc
            if 0 <= rr < H and 0 <= cc < W and bool(frame_mask[rr, cc]):
                dirs.add((-int(dr), -int(dc)))
    if len(dirs) != 1:
        return None

    direction = next(iter(dirs))
    side = _DIR_TO_SIDE[direction]
    r0, c0, r1, c1 = bbox_coords(pts)
    fr0, fc0, fr1, fc1 = map(int, frame_bbox)

    area = int((r1 - r0 + 1) * (c1 - c0 + 1))
    if int(pts.shape[0]) != area:
        return None

    if side == "up":
        if int(r1) != int(fr0 - 1):
            return None
        if int(c0) < int(fc0 + 1) or int(c1) > int(fc1 - 1):
            return None
    elif side == "down":
        if int(r0) != int(fr1 + 1):
            return None
        if int(c0) < int(fc0 + 1) or int(c1) > int(fc1 - 1):
            return None
    elif side == "left":
        if int(c1) != int(fc0 - 1):
            return None
        if int(r0) < int(fr0 + 1) or int(r1) > int(fr1 - 1):
            return None
    else:
        if int(c0) != int(fc1 + 1):
            return None
        if int(r0) < int(fr0 + 1) or int(r1) > int(fr1 - 1):
            return None

    return side, direction


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend side markers outward from hollow frames by periodic repetition to the grid edge.

    Input structure:
    - The background is the most frequent color.
    - One dominant non-background color forms several hollow rectangular frames with 1-cell-thick borders.
    - Each frame can have several separate marker components, each sitting flush outside a different side;
      exactly one of those markers somewhere in the grid may already appear as a long repeated strip.

    Rule:
    - Detect each hollow frame, then collect every non-frame component that touches that frame on exactly one side.
    - Treat the touching component's bounding box as the seed tile and repeat that rectangular pattern away from
      the frame along the corresponding outward direction until the boundary.
    - Frame pixels are protected, so rays may fill background but never overwrite the frame itself.

    Key insight: the frame side is the directional cue, and one frame can contribute multiple independent rays.
    Solve per attached component, not per whole object, using the frame geometry to separate ownership.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()

    bg = int(get_bg(g))
    frame_col = detect_dominant_big_component_color(g, bg=bg, min_h=5, min_w=5, connectivity=4)
    if frame_col is None:
        return out

    frame_lab, fn = label_mask(g == int(frame_col), connectivity=4)
    non_mask = (g != int(bg)) & (g != int(frame_col))
    mark_lab, _ = label_mask(non_mask, connectivity=4)

    for fk in range(1, int(fn) + 1):
        frame_mask = frame_lab == int(fk)
        frame_bbox = detect_hollow_frame(frame_mask)
        if frame_bbox is None:
            continue

        adj_ids = _adjacent_marker_ids(mark_lab, frame_mask)
        for cid in sorted(adj_ids):
            pts = np.argwhere(mark_lab == int(cid))
            if pts.size == 0:
                continue
            side_info = _component_side_relative_to_frame(pts, frame_mask, frame_bbox)
            if side_info is None:
                continue

            _side, direction = side_info
            r0, c0, r1, c1 = bbox_coords(pts)
            tile = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))
            extend_repeating_tile(
                out,
                tile,
                int(r0),
                int(c0),
                direction,
                bg=int(bg),
                protect={int(frame_col)},
            )

    return out


# --------------- generator (output-first fused frame multi-ray variant) ---------------

_H_RANGE = (20, 33)
_W_RANGE = (20, 29)

_FRAME_H_RANGE = (5, 9)
_FRAME_W_RANGE = (5, 9)

_MIN_MARKERS_PER_FRAME = 2
_MAX_MARKERS_PER_FRAME = 4
_SEED_ALONG_CHOICES = [1, 2]
_SEED_CROSS_CHOICES = [1, 2]
_SEED_MAX_COLORS = 4
_MIN_RAY_UNITS = 2
_PRE_EXT_IDEAL_UNITS = 5

_PLACEMENT_TRIES = 220
_SUBSET_TRIES = 16
_TILE_TRIES = 18
_GEN_RETRIES = 500


def _build_frame_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((int(h), int(w)), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _build_seed_tile(
    direction: tuple[int, int],
    palette: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    along = int(rng.choice(_SEED_ALONG_CHOICES))
    cross = int(rng.choice(_SEED_CROSS_CHOICES))
    if along == 1 and cross == 1:
        along = 2

    dr, dc = map(int, direction)
    th, tw = (along, cross) if dc == 0 else (cross, along)

    n_cells = int(th * tw)
    max_k = min(_SEED_MAX_COLORS, n_cells, len(palette))
    n_colors = int(rng.integers(1, max_k + 1))
    chosen = list(rng.choice(palette, size=n_colors, replace=False))
    vals = list(chosen) + list(rng.choice(chosen, size=n_cells - n_colors, replace=True))
    rng.shuffle(vals)
    tile = np.array(vals, dtype=int).reshape(int(th), int(tw))

    if n_cells > 1 and np.all(tile == tile.flat[0]) and len(palette) >= 2:
        alt_choices = [c for c in palette if c != int(tile.flat[0])]
        tile.flat[-1] = int(rng.choice(alt_choices))
    return tile


def _candidate_seed_origins(
    frame_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
    tile: np.ndarray,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    fr0, fc0, fr1, fc1 = map(int, frame_bbox)
    fh = int(fr1 - fr0 + 1)
    fw = int(fc1 - fc0 + 1)
    th, tw = tile.shape
    dr, dc = map(int, direction)
    out: list[tuple[int, int]] = []

    if dc == 0:
        sr0 = int(fr0 - th) if dr == -1 else int(fr1 + 1)
        if sr0 < 0 or sr0 + th > H:
            return out
        for local_c0 in range(1, int(fw - tw)):
            sc0 = int(fc0 + local_c0)
            if 0 <= sc0 and sc0 + tw <= W:
                out.append((int(sr0), int(sc0)))
    else:
        sc0 = int(fc0 - tw) if dc == -1 else int(fc1 + 1)
        if sc0 < 0 or sc0 + tw > W:
            return out
        for local_r0 in range(1, int(fh - th)):
            sr0 = int(fr0 + local_r0)
            if 0 <= sr0 and sr0 + th <= H:
                out.append((int(sr0), int(sc0)))

    return out


def _project_tile_coords(
    H: int,
    W: int,
    seed_r0: int,
    seed_c0: int,
    tile: np.ndarray,
    direction: tuple[int, int],
) -> list[tuple[int, int, int]]:
    th, tw = tile.shape
    dr, dc = map(int, direction)
    cells: list[tuple[int, int, int]] = []

    if dc == 0:
        step = int(dr * th)
        r = int(seed_r0)
        while 0 <= r and r + th <= H:
            for i in range(th):
                for j in range(tw):
                    cells.append((int(r + i), int(seed_c0 + j), int(tile[i, j])))
            r += step
    else:
        step = int(dc * tw)
        c = int(seed_c0)
        while 0 <= c and c + tw <= W:
            for i in range(th):
                for j in range(tw):
                    cells.append((int(seed_r0 + i), int(c + j), int(tile[i, j])))
            c += step

    return cells


def _compute_ray_units(
    H: int,
    W: int,
    seed_r0: int,
    seed_c0: int,
    tile: np.ndarray,
    direction: tuple[int, int],
) -> int:
    th, tw = tile.shape
    dr, dc = map(int, direction)
    if dr == 1:
        return int((H - seed_r0) // th)
    if dr == -1:
        return int(seed_r0 // th) + 1
    if dc == 1:
        return int((W - seed_c0) // tw)
    return int(seed_c0 // tw) + 1


def _frame_cells_fit(occ: np.ndarray, frame_cells: list[tuple[int, int]]) -> bool:
    H, W = occ.shape
    for r, c in frame_cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(occ[r, c]) != 0:
            return False
        for dr, dc in _DIRECTIONS:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < H and 0 <= cc < W and int(occ[rr, cc]) != 0:
                return False
    return True


def _ray_cells_fit(
    trial_occ: np.ndarray,
    cells: list[tuple[int, int, int]],
    own_frame_cells: set[tuple[int, int]],
) -> bool:
    H, W = trial_occ.shape
    cell_pos = {(int(r), int(c)) for r, c, _ in cells}
    if len(cell_pos) != len(cells):
        return False

    for r, c, _ in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(trial_occ[r, c]) != 0:
            return False
        for dr, dc in _DIRECTIONS:
            rr = r + dr
            cc = c + dc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if (rr, cc) in cell_pos:
                continue
            val = int(trial_occ[rr, cc])
            if val == 2:
                return False
            if val == 1 and (rr, cc) not in own_frame_cells:
                return False
    return True


def _tile_signature(tile: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return tuple(map(int, tile.shape)), tuple(map(int, tile.flatten()))


def _sample_direction_subset(rng: np.random.Generator) -> list[tuple[int, int]]:
    dirs = _DIRECTIONS.copy()
    rng.shuffle(dirs)
    k = int(rng.choice([2, 3, 3, 4]))
    return dirs[:k]


def _place_frame_and_rays(
    rng: np.random.Generator,
    frame_h: int,
    frame_w: int,
    palette: list[int],
    H: int,
    W: int,
    occ: np.ndarray,
    out: np.ndarray,
    frame_color: int,
) -> dict[str, object] | None:
    frame_mask = _build_frame_mask(frame_h, frame_w)

    for _ in range(_PLACEMENT_TRIES):
        r0 = int(rng.integers(1, H - frame_h))
        c0 = int(rng.integers(1, W - frame_w))
        frame_bbox = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
        frame_cells = [
            (int(r0 + rr), int(c0 + cc))
            for rr in range(frame_h)
            for cc in range(frame_w)
            if bool(frame_mask[rr, cc])
        ]
        if not _frame_cells_fit(occ, frame_cells):
            continue

        own_frame_cells = set((int(r), int(c)) for r, c in frame_cells)

        for _ in range(_SUBSET_TRIES):
            chosen_dirs = _sample_direction_subset(rng)
            trial_occ = occ.copy()
            for rr, cc in frame_cells:
                trial_occ[rr, cc] = 1

            rays: list[dict[str, object]] = []
            used_tiles: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
            success = True

            for direction in chosen_dirs:
                ray_info: dict[str, object] | None = None
                for _ in range(_TILE_TRIES):
                    tile = _build_seed_tile(direction, palette, rng)
                    sig = _tile_signature(tile)
                    if sig in used_tiles:
                        continue

                    origins = _candidate_seed_origins(frame_bbox, direction, tile, H, W)
                    if not origins:
                        continue
                    rng.shuffle(origins)

                    for seed_r0, seed_c0 in origins:
                        units = _compute_ray_units(H, W, seed_r0, seed_c0, tile, direction)
                        if units < _MIN_RAY_UNITS:
                            continue

                        cells = _project_tile_coords(H, W, seed_r0, seed_c0, tile, direction)
                        if not _ray_cells_fit(trial_occ, cells, own_frame_cells):
                            continue

                        for rr, cc, _vv in cells:
                            trial_occ[rr, cc] = 2
                        ray_info = {
                            "direction": tuple(map(int, direction)),
                            "tile": tile,
                            "cells": cells,
                            "seed_r0": int(seed_r0),
                            "seed_c0": int(seed_c0),
                            "units": int(units),
                        }
                        used_tiles.add(sig)
                        break

                    if ray_info is not None:
                        break

                if ray_info is None:
                    success = False
                    break
                rays.append(ray_info)

            if not success:
                continue

            for rr, cc in frame_cells:
                occ[rr, cc] = 1
                out[rr, cc] = int(frame_color)
            for ray in rays:
                for rr, cc, vv in ray["cells"]:
                    occ[rr, cc] = 2
                    out[rr, cc] = int(vv)

            return {
                "frame_bbox": frame_bbox,
                "marker_count": len(rays),
                "rays": rays,
            }

    return None


def _build_input_from_output(
    out: np.ndarray,
    frames: list[dict[str, object]],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    all_rays: list[tuple[int, int, dict[str, object]]] = []
    for fi, frame in enumerate(frames):
        for ri, ray in enumerate(frame["rays"]):
            all_rays.append((fi, ri, ray))

    best_marker_count = max(int(frame["marker_count"]) for frame in frames)
    strong = [
        i
        for i, (fi, _ri, ray) in enumerate(all_rays)
        if int(frames[fi]["marker_count"]) == best_marker_count and int(ray["units"]) >= _PRE_EXT_IDEAL_UNITS
    ]
    if strong:
        keep_idx = int(rng.choice(strong))
    else:
        keep_idx = max(
            range(len(all_rays)),
            key=lambda i: (int(frames[all_rays[i][0]]["marker_count"]), int(all_rays[i][2]["units"])),
        )

    keep_fi, keep_ri, _keep_ray = all_rays[keep_idx]
    inp = out.copy()
    for fi, frame in enumerate(frames):
        for ri, ray in enumerate(frame["rays"]):
            if fi == keep_fi and ri == keep_ri:
                continue
            tile = np.asarray(ray["tile"], dtype=int)
            th, tw = tile.shape
            sr0 = int(ray["seed_r0"])
            sc0 = int(ray["seed_c0"])
            seed_set = {(sr0 + rr, sc0 + cc) for rr in range(th) for cc in range(tw)}
            for r, c, _ in ray["cells"]:
                if (int(r), int(c)) not in seed_set:
                    inp[int(r), int(c)] = int(bg)
    return inp


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [c for c in range(10) if c != bg]
    frame_color = int(rng.choice(others))
    palette = [c for c in others if c != frame_color]

    frame_h = int(rng.integers(*_FRAME_H_RANGE))
    frame_w = int(rng.integers(*_FRAME_W_RANGE))
    n_frames = int(rng.choice([2, 2, 3]))

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    frames: list[dict[str, object]] = []

    for _ in range(n_frames):
        info = _place_frame_and_rays(
            rng, frame_h, frame_w, palette, H, W, occ, out, frame_color
        )
        if info is None:
            return None
        frames.append(info)

    marker_counts = [int(frame["marker_count"]) for frame in frames]
    if max(marker_counts) < 3:
        return None
    inp = _build_input_from_output(out, frames, bg, rng)

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
    raise RuntimeError(f"generate_input: failed after {_GEN_RETRIES} attempts")
