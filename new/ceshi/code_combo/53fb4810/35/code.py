# Combo variant of 53fb4810 (index 35)
# combo_spec: D4/0+D6/0+D7/0
# Name: 矩形框架遇障源消失

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    The base task used solid '+' anchors, boundary-bounded rays, and source retention.
#    Here each anchor is a hollow 1-cell-thick rectangle with one 1-cell gap on exactly
#    one side. A small seed tile sits outside that same side, close enough to the gap that
#    the opening is visibly the directional cue. A shared obstacle color appears farther
#    along every ray. Example:
#      original: plus -> seed AB -> AB AB AB ... boundary, plus remains
#      combo   : frame-with-top-gap, seed AB above it -> AB AB [obstacle bar], then the
#                whole frame footprint is erased in the output
#    So three stages change together: direction comes from frame geometry instead of plus
#    arms, extension stops at the first blocker instead of the edge, and the anchor itself
#    disappears after the ray is completed.
# b) Complexity calibration:
#    - Gap width is fixed to 1 cell, and obstacle color is shared globally. This keeps both
#      cues repeated and easy to notice across 2-3 objects.
#    - Seed tiles are still only 1x2, 2x1, or 2x2, which preserves the original periodic
#      reasoning class without introducing hard period inference.
#    - Each grid is required to use at least two gap directions, so the frame opening is a
#      meaningful signal rather than a constant orientation.
# c) Hidden-assumption audit against the original solver:
#    - "dominant big component color = plus color" is invalid in shape but still usable if
#      retargeted to hollow frames; min bbox thresholds are raised accordingly.
#    - "adjacency side alone gives direction" is no longer reliable. The solver must read
#      the unique missing border cell and then only accept payload touching that side.
#    - "non-bg/non-source components are all markers" breaks once obstacle bars are added.
#      The solver therefore tries each non-frame color as a possible obstacle color and
#      chooses the one whose removal makes every frame adjacent to exactly one valid marker.
#    - "out = input copy means source is preserved" must be explicitly inverted at the end
#      by erasing every frame cell back to the background.
# d) Placement legality:
#    Generation is output-first over a full scene containing frames, rays, and obstacles.
#    Candidate placements are rejected if they overlap, leave the grid, or 4-touch earlier
#    objects. Within one object, only the seed may touch its own frame and only the obstacle
#    may touch its own ray at the stopping face.
# e) Visual observability:
#    Across a few examples, a human sees the same one-color rectangular outlines, the same
#    detached obstacle color, and the same disappearance of the outlines in the outputs. The
#    clue ray terminates exactly at the obstacle, making the stop rule visible.
# f) Visual cleanliness:
#    Rays are axis-aligned rectangles, obstacle bars are thin aligned stripes, and different
#    objects are separated by a margin. This avoids merged components and layered clutter.
# g) Change inferability:
#    No hidden color ordering is used. Colors only play structural roles that are visible:
#    frame color, obstacle color, and seed colors. The important logic comes from geometry.
# h) New solver assumptions checked against the generator:
#    - Every frame is a valid hollow rectangle with exactly one gap side.
#    - After removing the obstacle color, each frame has exactly one marker/ray component on
#      its gap side and that component stays entirely outside the frame bbox on that side.
#    - Frame color never appears in rays, so clearing frame-colored cells cannot delete ray
#      content. Obstacle color never appears inside seeds, so structural obstacle inference
#      does not remove marker pixels.


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
    Local helper: choose the color that has the most "big" components.

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


def detect_hollow_frame_gap(frame_mask: np.ndarray) -> dict[str, object] | None:
    """Return hollow-frame geometry with its unique gap side, or None if invalid."""
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
    if not (bool(local[0, 0]) and bool(local[0, -1]) and bool(local[-1, 0]) and bool(local[-1, -1])):
        return None

    side_arrays = {
        "up": local[0, 1:-1],
        "down": local[-1, 1:-1],
        "left": local[1:-1, 0],
        "right": local[1:-1, -1],
    }
    gaps: list[tuple[str, int, int]] = []
    for side, arr in side_arrays.items():
        miss = np.where(~arr)[0]
        if miss.size == 0:
            continue
        if not np.all(np.diff(miss) == 1):
            return None
        start = int(miss[0] + 1)
        end = int(miss[-1] + 1)
        if end - start + 1 > 2:
            return None
        gaps.append((side, start, end))

    if len(gaps) != 1:
        return None

    gap_side, gap_start, gap_end = gaps[0]
    for side, arr in side_arrays.items():
        for idx, val in enumerate(arr, start=1):
            should_be_filled = not (side == gap_side and gap_start <= idx <= gap_end)
            if bool(val) != bool(should_be_filled):
                return None

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "side": str(gap_side),
        "direction": tuple(map(int, _SIDE_TO_DIR[str(gap_side)])),
        "gap_span": (int(gap_start), int(gap_end)),
    }


def _adjacent_marker_ids_on_gap_side(
    mark_lab: np.ndarray,
    frame_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    side: str,
) -> set[int]:
    """Collect marker components that touch the frame specifically on the gapped side."""
    H, W = mark_lab.shape
    r0, c0, r1, c1 = map(int, bbox)
    ids: set[int] = set()
    if side == "up":
        rr = r0 - 1
        if rr < 0:
            return ids
        for cc in range(c0, c1 + 1):
            if bool(frame_mask[r0, cc]) and int(mark_lab[rr, cc]) > 0:
                ids.add(int(mark_lab[rr, cc]))
    elif side == "down":
        rr = r1 + 1
        if rr >= H:
            return ids
        for cc in range(c0, c1 + 1):
            if bool(frame_mask[r1, cc]) and int(mark_lab[rr, cc]) > 0:
                ids.add(int(mark_lab[rr, cc]))
    elif side == "left":
        cc = c0 - 1
        if cc < 0:
            return ids
        for rr in range(r0, r1 + 1):
            if bool(frame_mask[rr, c0]) and int(mark_lab[rr, cc]) > 0:
                ids.add(int(mark_lab[rr, cc]))
    else:
        cc = c1 + 1
        if cc >= W:
            return ids
        for rr in range(r0, r1 + 1):
            if bool(frame_mask[rr, c1]) and int(mark_lab[rr, cc]) > 0:
                ids.add(int(mark_lab[rr, cc]))
    return ids


def _marker_component_valid_for_frame(
    pts: np.ndarray,
    info: dict[str, object],
) -> tuple[int, int, int, int] | None:
    """Check that a marker/ray component lies fully outside the frame on the gap side."""
    if pts.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(pts)
    fr0, fc0, fr1, fc1 = map(int, info["bbox"])
    side = str(info["side"])
    gap_start, gap_end = map(int, info["gap_span"])

    if side == "up":
        if int(r1) >= int(fr0):
            return None
        local0 = int(c0 - fc0)
        local1 = int(c1 - fc0)
        if local0 < 1 or local1 > (fc1 - fc0 - 1):
            return None
        if local0 > gap_end + 1 or local1 < gap_start - 1:
            return None
    elif side == "down":
        if int(r0) <= int(fr1):
            return None
        local0 = int(c0 - fc0)
        local1 = int(c1 - fc0)
        if local0 < 1 or local1 > (fc1 - fc0 - 1):
            return None
        if local0 > gap_end + 1 or local1 < gap_start - 1:
            return None
    elif side == "left":
        if int(c1) >= int(fc0):
            return None
        local0 = int(r0 - fr0)
        local1 = int(r1 - fr0)
        if local0 < 1 or local1 > (fr1 - fr0 - 1):
            return None
        if local0 > gap_end + 1 or local1 < gap_start - 1:
            return None
    else:
        if int(c0) <= int(fc1):
            return None
        local0 = int(r0 - fr0)
        local1 = int(r1 - fr0)
        if local0 < 1 or local1 > (fr1 - fr0 - 1):
            return None
        if local0 > gap_end + 1 or local1 < gap_start - 1:
            return None

    return int(r0), int(c0), int(r1), int(c1)


def _infer_obstacle_color_and_mark_lab(
    grid: np.ndarray,
    frame_lab: np.ndarray,
    frame_infos: dict[int, dict[str, object]],
    *,
    bg: int,
    frame_col: int,
) -> tuple[int | None, np.ndarray]:
    """
    Find the color whose removal makes every frame adjacent to exactly one valid marker/ray.
    Returns (obstacle_color, mark_lab).
    """
    colors = sorted(set(map(int, grid.flatten())) - {int(bg), int(frame_col)})

    for obstacle_col in colors:
        non_mask = (grid != int(bg)) & (grid != int(frame_col)) & (grid != int(obstacle_col))
        mark_lab, _ = label_mask(non_mask, connectivity=4)
        ok = True
        for fk, info in frame_infos.items():
            frame_mask = frame_lab == int(fk)
            adj_ids = _adjacent_marker_ids_on_gap_side(mark_lab, frame_mask, info["bbox"], str(info["side"]))
            if len(adj_ids) != 1:
                ok = False
                break
            cid = next(iter(adj_ids))
            pts = np.argwhere(mark_lab == int(cid))
            if _marker_component_valid_for_frame(pts, info) is None:
                ok = False
                break
        if ok:
            return int(obstacle_col), mark_lab

    non_mask = (grid != int(bg)) & (grid != int(frame_col))
    mark_lab, _ = label_mask(non_mask, connectivity=4)
    return None, mark_lab


def extend_repeating_tile_until_blocked(
    out: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    bg: int,
) -> None:
    """Extend a repeating tile from (r0,c0) until the next stripe would hit a non-bg cell."""
    H, W = out.shape
    h, w = tile.shape
    dr, dc = map(int, direction)
    dr = 0 if dr == 0 else int(dr / abs(dr))
    dc = 0 if dc == 0 else int(dc / abs(dc))
    bg = int(bg)
    r0 = int(r0)
    c0 = int(c0)

    def can_paint(cells: list[tuple[int, int, int]]) -> bool:
        active = [(rr, cc) for rr, cc, v in cells if int(v) != int(bg)]
        return all(int(out[rr, cc]) == int(bg) for rr, cc in active)

    def paint(cells: list[tuple[int, int, int]]) -> None:
        for rr, cc, v in cells:
            if int(v) != int(bg):
                out[rr, cc] = int(v)

    if (dr, dc) == (0, 1):
        start_c = c0 + w
        for k, tc in enumerate(range(start_c, W)):
            stripe = [(r0 + i, tc, int(tile[i, k % w])) for i in range(h)]
            if not can_paint(stripe):
                break
            paint(stripe)
    elif (dr, dc) == (0, -1):
        for k, tc in enumerate(range(c0 - 1, -1, -1), start=1):
            stripe = [(r0 + i, tc, int(tile[i, (-k) % w])) for i in range(h)]
            if not can_paint(stripe):
                break
            paint(stripe)
    elif (dr, dc) == (1, 0):
        start_r = r0 + h
        for k, tr in enumerate(range(start_r, H)):
            stripe = [(tr, c0 + j, int(tile[k % h, j])) for j in range(w)]
            if not can_paint(stripe):
                break
            paint(stripe)
    elif (dr, dc) == (-1, 0):
        for k, tr in enumerate(range(r0 - 1, -1, -1), start=1):
            stripe = [(tr, c0 + j, int(tile[(-k) % h, j])) for j in range(w)]
            if not can_paint(stripe):
                break
            paint(stripe)
    else:
        raise ValueError("extend_repeating_tile_until_blocked: direction must be axial")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend marker tiles outward from hollow frames until blockers, then erase the frames.

    Input structure:
    - The background is the most common color.
    - One dominant non-background color forms several hollow rectangular frames with a
      single 1-cell gap on one side.
    - Each frame has one adjacent seed/ray component on that same side, and one extra
      shared color appears only in detached obstacle bars farther along the rays.

    Rule:
    - For each frame, detect which side contains the unique gap and locate the payload
      component touching that side.
    - Repeat that component outward in the gap direction, but stop as soon as the next
      stripe would hit a non-background cell, which is the obstacle bar.
    - After all rays are completed, erase every frame pixel back to the background.

    Key insight: The frame opening determines direction, the obstacle bars determine
    how far the repetition goes, and the frame itself is only an input scaffold that
    disappears in the final output.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()

    bg = int(get_bg(g))
    frame_col = detect_dominant_big_component_color(g, bg=bg, min_h=5, min_w=5, connectivity=4)
    if frame_col is None:
        return out

    frame_lab, fn = label_mask(g == int(frame_col), connectivity=4)
    frame_infos: dict[int, dict[str, object]] = {}
    for fk in range(1, int(fn) + 1):
        frame_mask = frame_lab == int(fk)
        info = detect_hollow_frame_gap(frame_mask)
        if info is not None:
            frame_infos[int(fk)] = info
    if not frame_infos:
        return out

    _obstacle_col, mark_lab = _infer_obstacle_color_and_mark_lab(
        g,
        frame_lab,
        frame_infos,
        bg=int(bg),
        frame_col=int(frame_col),
    )

    for fk, info in frame_infos.items():
        frame_mask = frame_lab == int(fk)
        adj_ids = _adjacent_marker_ids_on_gap_side(mark_lab, frame_mask, info["bbox"], str(info["side"]))
        for cid in sorted(adj_ids):
            pts = np.argwhere(mark_lab == int(cid))
            bbox = _marker_component_valid_for_frame(pts, info)
            if bbox is None:
                continue
            r0, c0, r1, c1 = map(int, bbox)
            tile = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))
            extend_repeating_tile_until_blocked(
                out,
                tile,
                int(r0),
                int(c0),
                tuple(map(int, info["direction"])),
                bg=int(bg),
            )

    out[frame_lab > 0] = int(bg)
    return out


# --------------- generator (output-first combo variant) ---------------

# --- grid / counts ---
_H_RANGE = (14, 29)
_W_RANGE = (14, 24)
_N_FRAME_RANGE = (2, 5)

# --- frame geometry ---
_FRAME_H_RANGE = (5, 9)
_FRAME_W_RANGE = (5, 9)
_FRAME_GAP_WIDTH = 1

# --- seed / ray geometry ---
_ALONG_CHOICES = [1, 2]
_CROSS_CHOICES = [1, 2]
_SEED_MAX_COLORS = 4
_MIN_STOP_UNITS = 2
_PRE_EXT_IDEAL_UNITS = 5

# --- retries ---
_PLACEMENT_TRIES = 220
_GEN_RETRIES = 500


def _build_frame_mask(h: int, w: int, side: str) -> np.ndarray:
    mask = np.zeros((int(h), int(w)), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    if side in ("up", "down"):
        gap = int(w // 2)
        row = 0 if side == "up" else h - 1
        mask[row, gap : gap + _FRAME_GAP_WIDTH] = False
    else:
        gap = int(h // 2)
        col = 0 if side == "left" else w - 1
        mask[gap : gap + _FRAME_GAP_WIDTH, col] = False
    return mask


def _gap_index(frame_h: int, frame_w: int, side: str) -> int:
    return int(frame_w // 2) if side in ("up", "down") else int(frame_h // 2)


def _build_seed_tile(
    direction: tuple[int, int],
    palette: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    along = int(rng.choice(_ALONG_CHOICES))
    cross = int(rng.choice(_CROSS_CHOICES))
    if along == 1 and cross == 1:
        along = 2
    _dr, dc = map(int, direction)
    th, tw = (along, cross) if dc == 0 else (cross, along)

    n_cells = int(th * tw)
    max_k = min(_SEED_MAX_COLORS, n_cells, len(palette))
    n_colors = int(rng.integers(1, max_k + 1))
    chosen = list(rng.choice(palette, size=n_colors, replace=False))
    vals = list(chosen) + list(rng.choice(chosen, size=n_cells - n_colors, replace=True))
    rng.shuffle(vals)
    return np.array(vals, dtype=int).reshape(int(th), int(tw))


def _candidate_seed_origins(
    frame_bbox: tuple[int, int, int, int],
    side: str,
    tile: np.ndarray,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, frame_bbox)
    th, tw = tile.shape
    fh = int(r1 - r0 + 1)
    fw = int(c1 - c0 + 1)
    gap = _gap_index(fh, fw, side)
    out: list[tuple[int, int]] = []

    if side in ("up", "down"):
        sr0 = int(r0 - th) if side == "up" else int(r1 + 1)
        if sr0 < 0 or sr0 + th > H:
            return out
        for lc0 in range(1, fw - tw):
            cols = list(range(lc0, lc0 + tw))
            if all(col == gap for col in cols):
                continue
            if min(abs(col - gap) for col in cols) > 1:
                continue
            sc0 = int(c0 + lc0)
            if 0 <= sc0 and sc0 + tw <= W:
                out.append((int(sr0), int(sc0)))
    else:
        sc0 = int(c0 - tw) if side == "left" else int(c1 + 1)
        if sc0 < 0 or sc0 + tw > W:
            return out
        for lr0 in range(1, fh - th):
            rows = list(range(lr0, lr0 + th))
            if all(row == gap for row in rows):
                continue
            if min(abs(row - gap) for row in rows) > 1:
                continue
            sr0 = int(r0 + lr0)
            if 0 <= sr0 and sr0 + th <= H:
                out.append((int(sr0), int(sc0)))

    return out


def _compute_full_units(
    H: int,
    W: int,
    seed_r0: int,
    seed_c0: int,
    direction: tuple[int, int],
    tile: np.ndarray,
) -> int:
    """Maximum number of full tile copies that fit before the boundary, including the seed."""
    dr, dc = map(int, direction)
    th, tw = tile.shape
    if dr == 1:
        return int(1 + (H - (seed_r0 + th)) // th)
    if dr == -1:
        return int(1 + seed_r0 // th)
    if dc == 1:
        return int(1 + (W - (seed_c0 + tw)) // tw)
    return int(1 + seed_c0 // tw)


def _project_tile_coords_for_units(
    seed_r0: int,
    seed_c0: int,
    direction: tuple[int, int],
    tile: np.ndarray,
    units: int,
) -> list[tuple[int, int, int]]:
    """Return cells for the first `units` full tile copies, including the seed."""
    dr, dc = map(int, direction)
    th, tw = tile.shape
    cells: list[tuple[int, int, int]] = []
    for u in range(int(units)):
        rr0 = int(seed_r0 + u * th * dr)
        cc0 = int(seed_c0 + u * tw * dc)
        for i in range(th):
            for j in range(tw):
                cells.append((int(rr0 + i), int(cc0 + j), int(tile[i, j])))
    return cells


def _obstacle_cells_after_units(
    seed_r0: int,
    seed_c0: int,
    direction: tuple[int, int],
    tile: np.ndarray,
    stop_units: int,
    obstacle_color: int,
) -> list[tuple[int, int, int]]:
    """Place a 1-stripe obstacle bar immediately after `stop_units` full tiles."""
    dr, dc = map(int, direction)
    th, tw = tile.shape
    cells: list[tuple[int, int, int]] = []
    if dr == 1:
        row = int(seed_r0 + stop_units * th)
        for j in range(tw):
            cells.append((row, int(seed_c0 + j), int(obstacle_color)))
    elif dr == -1:
        row = int(seed_r0 - 1 - (stop_units - 1) * th)
        for j in range(tw):
            cells.append((row, int(seed_c0 + j), int(obstacle_color)))
    elif dc == 1:
        col = int(seed_c0 + stop_units * tw)
        for i in range(th):
            cells.append((int(seed_r0 + i), col, int(obstacle_color)))
    else:
        col = int(seed_c0 - 1 - (stop_units - 1) * tw)
        for i in range(th):
            cells.append((int(seed_r0 + i), col, int(obstacle_color)))
    return cells


def _touches_existing(occ: np.ndarray, r: int, c: int) -> bool:
    H, W = occ.shape
    if bool(occ[r, c]):
        return True
    for dr, dc in _DIRECTIONS:
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < H and 0 <= cc < W and bool(occ[rr, cc]):
            return True
    return False


def _all_in_bounds(H: int, W: int, cells: list[tuple[int, int, int]]) -> bool:
    return all(0 <= int(r) < H and 0 <= int(c) < W for r, c, _ in cells)


def _place_frame_ray_and_obstacle(
    rng: np.random.Generator,
    frame_h: int,
    frame_w: int,
    palette: list[int],
    H: int,
    W: int,
    occ: np.ndarray,
    scene: np.ndarray,
    *,
    frame_color: int,
    obstacle_color: int,
) -> dict[str, object] | None:
    """Try to place one frame, its repeating ray, and a blocking obstacle."""
    for _ in range(_PLACEMENT_TRIES):
        r0 = int(rng.integers(1, H - frame_h))
        c0 = int(rng.integers(1, W - frame_w))

        dirs = list(_DIRECTIONS)
        rng.shuffle(dirs)
        for direction in dirs:
            side = _DIR_TO_SIDE[tuple(direction)]
            frame_mask = _build_frame_mask(frame_h, frame_w, side)
            frame_cells = [
                (int(r0 + rr), int(c0 + cc))
                for rr in range(frame_h)
                for cc in range(frame_w)
                if bool(frame_mask[rr, cc])
            ]
            if any(_touches_existing(occ, rr, cc) for rr, cc in frame_cells):
                continue

            tile = _build_seed_tile(direction, palette, rng)
            seed_origins = _candidate_seed_origins(
                (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1), side, tile, H, W
            )
            if not seed_origins:
                continue
            rng.shuffle(seed_origins)

            for seed_r0, seed_c0 in seed_origins:
                full_units = _compute_full_units(H, W, seed_r0, seed_c0, direction, tile)
                if full_units <= _MIN_STOP_UNITS:
                    continue
                candidates = list(range(_MIN_STOP_UNITS, full_units))
                rng.shuffle(candidates)
                for stop_units in candidates:
                    ray_cells = _project_tile_coords_for_units(seed_r0, seed_c0, direction, tile, stop_units)
                    obstacle_cells = _obstacle_cells_after_units(
                        seed_r0,
                        seed_c0,
                        direction,
                        tile,
                        stop_units,
                        int(obstacle_color),
                    )
                    if not _all_in_bounds(H, W, ray_cells) or not _all_in_bounds(H, W, obstacle_cells):
                        continue
                    ray_coords = [(int(r), int(c)) for r, c, _ in ray_cells]
                    obstacle_coords = [(int(r), int(c)) for r, c, _ in obstacle_cells]
                    all_coords = frame_cells + ray_coords + obstacle_coords
                    if len(set(all_coords)) != len(all_coords):
                        continue
                    if any(_touches_existing(occ, rr, cc) for rr, cc in ray_coords):
                        continue
                    if any(_touches_existing(occ, rr, cc) for rr, cc in obstacle_coords):
                        continue

                    for rr, cc in frame_cells:
                        occ[rr, cc] = True
                        scene[rr, cc] = int(frame_color)
                    for rr, cc, v in ray_cells:
                        occ[rr, cc] = True
                        scene[rr, cc] = int(v)
                    for rr, cc, v in obstacle_cells:
                        occ[rr, cc] = True
                        scene[rr, cc] = int(v)

                    return {
                        "direction": tuple(map(int, direction)),
                        "side": str(side),
                        "tile": np.array(tile, dtype=int),
                        "cells": ray_cells,
                        "seed_r0": int(seed_r0),
                        "seed_c0": int(seed_c0),
                        "units": int(stop_units),
                        "full_units": int(full_units),
                        "frame_cells": frame_cells,
                        "obstacle_cells": obstacle_cells,
                    }

    return None


def _erase_frames(scene: np.ndarray, frames: list[dict[str, object]], bg: int) -> np.ndarray:
    out = scene.copy()
    for item in frames:
        for r, c in item["frame_cells"]:
            out[int(r), int(c)] = int(bg)
    return out


def _build_input_from_scene(
    scene: np.ndarray,
    frames: list[dict[str, object]],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Keep full frames and obstacles, but erase the repeated suffix of every non-demo ray."""
    pre_idx = max(range(len(frames)), key=lambda i: int(frames[i]["units"]))
    if int(frames[pre_idx]["units"]) < _PRE_EXT_IDEAL_UNITS:
        candidates = [i for i, item in enumerate(frames) if int(item["units"]) >= _PRE_EXT_IDEAL_UNITS]
        if candidates:
            pre_idx = int(rng.choice(candidates))

    inp = scene.copy()
    for i, item in enumerate(frames):
        if i == pre_idx:
            continue
        tile = np.asarray(item["tile"], dtype=int)
        th, tw = tile.shape
        sr0 = int(item["seed_r0"])
        sc0 = int(item["seed_c0"])
        seed_set = {(sr0 + rr, sc0 + cc) for rr in range(th) for cc in range(tw)}
        for r, c, _ in item["cells"]:
            if (int(r), int(c)) not in seed_set:
                inp[int(r), int(c)] = int(bg)
    return inp


def _gen_attempt(
    rng: np.random.Generator,
    *,
    return_meta: bool = False,
) -> np.ndarray | dict[str, object] | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [c for c in range(10) if c != bg]
    frame_color, obstacle_color = map(int, rng.choice(others, size=2, replace=False))
    palette = [c for c in others if c not in {frame_color, obstacle_color}]

    frame_h = int(rng.integers(*_FRAME_H_RANGE))
    frame_w = int(rng.integers(*_FRAME_W_RANGE))
    N = int(rng.integers(*_N_FRAME_RANGE))

    scene = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=bool)

    frames: list[dict[str, object]] = []
    for _ in range(N):
        info = _place_frame_ray_and_obstacle(
            rng,
            frame_h,
            frame_w,
            palette,
            H,
            W,
            occ,
            scene,
            frame_color=int(frame_color),
            obstacle_color=int(obstacle_color),
        )
        if info is None:
            return None
        frames.append(info)

    if len({tuple(item["direction"]) for item in frames}) < 2:
        return None
    if max(int(item["units"]) for item in frames) < _PRE_EXT_IDEAL_UNITS:
        return None

    inp = _build_input_from_scene(scene, frames, bg, rng)
    out = _erase_frames(scene, frames, bg)

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, out):
        return None
    if return_meta:
        return {
            "input": inp,
            "output": out,
            "scene": scene,
            "bg": int(bg),
            "frame_color": int(frame_color),
            "obstacle_color": int(obstacle_color),
            "frames": frames,
        }
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_GEN_RETRIES} attempts")
