# Variant of 53fb4810
# Dimension: D4 — 源对象形状 (src)
# Change: '+'精灵为小型中心对称色块（3x3十字、3x4胖十字、5x4胖菱形等），bbox至少3x3，左右+上下对称... → 矩形框架作为源

from __future__ import annotations

import numpy as np
from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# Design analysis for the D4 variant
# ----------------------------------
# spec:
#   src:
#     old: solid centrally-symmetric plus/blob; marker side itself gave the ray direction
#     new: one-color hollow rectangular frame with a single 1-cell gap on exactly one side
#   visible behavior:
#     - each frame is a 1-cell-thick border; its interior stays background
#     - a small non-frame marker tile sits just outside the same side that contains the gap
#     - the gap side, not a plus arm, is the structural cue that says whether the ray goes
#       up / down / left / right
#   concrete example:
#     - old flow: detect plus component -> find adjacent seed -> use arm-tip span as nozzle
#     - new flow: detect hollow frame -> read which side of the perimeter has the unique gap
#       -> find the marker component touching that side -> repeat that component outward
#   complexity calibration:
#     - only four direction categories remain, matching the original task's ray logic
#     - frames share one outer size per grid, but gap side varies across frames so the D4
#       change is visible within a single example
#     - gap width is fixed to 1 cell; this keeps detection unambiguous and prevents noisy,
#       weakly attached seeds
#   hidden assumptions audited from the original main():
#     - dominant big-component color is still valid, but it now denotes sparse hollow frames
#       rather than dense pluses
#     - adjacency-to-source no longer reveals direction reliably; direction must come from
#       frame geometry, specifically the unique missing border cell on one side
#     - the old arm-tip alignment asserts are invalid because frames have no arm nozzles;
#       generator instead guarantees one marker component touching only the gapped side
#   placement legality:
#     - frame border cells, seed cells, and projected rays are generated output-first
#     - different objects must neither overlap nor 4-touch each other, otherwise multiple
#       marker rays could merge into one connected component and break solver assumptions
#   observability / cleanliness:
#     - the frame outline plus its single gap is easy to spot by eye
#     - using at least two gap directions per generated grid avoids degenerating back into a
#       single repeated orientation
#     - colors are never used for ordering or direction; geometry alone controls the rule


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

    best_key: tuple[int, int] | None = None  # (num_big, total_big_area)
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
    """Local helper (only used by this solver): extend a tile pattern from (r0,c0) until boundary."""
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

    if (dr, dc) == (0, 1):  # right
        start_c = c0 + w
        for k, tc in enumerate(range(start_c, W)):
            col_idx = k % w
            for i in range(h):
                put(r0 + i, tc, int(tile[i, col_idx]))
    elif (dr, dc) == (0, -1):  # left
        for k, tc in enumerate(range(c0 - 1, -1, -1), start=1):
            col_idx = (-k) % w
            for i in range(h):
                put(r0 + i, tc, int(tile[i, col_idx]))
    elif (dr, dc) == (1, 0):  # down
        start_r = r0 + h
        for k, tr in enumerate(range(start_r, H)):
            row_idx = k % h
            for j in range(w):
                put(tr, c0 + j, int(tile[row_idx, j]))
    elif (dr, dc) == (-1, 0):  # up
        for k, tr in enumerate(range(r0 - 1, -1, -1), start=1):
            row_idx = (-k) % h
            for j in range(w):
                put(tr, c0 + j, int(tile[row_idx, j]))
    else:
        raise ValueError("extend_repeating_tile: direction must be one of (±1,0),(0,±1)")


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
        "side": gap_side,
        "direction": _SIDE_TO_DIR[gap_side],
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
    else:  # right
        cc = c1 + 1
        if cc >= W:
            return ids
        for rr in range(r0, r1 + 1):
            if bool(frame_mask[rr, c1]) and int(mark_lab[rr, cc]) > 0:
                ids.add(int(mark_lab[rr, cc]))
    return ids


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend marker tiles outward from hollow frames by repeating them to the grid edge.

    Input structure:
    - bg = most common color.
    - A dominant non-bg color forms several hollow rectangular frames with 1-cell-thick borders.
    - Each frame has exactly one missing border cell on one side, and a small colored tile sits
      just outside that same side; one frame already shows its full repeated ray as an example.

    Rule:
    - For each frame, detect which side contains the unique border gap.
    - Take the adjacent marker component on that side and project its rectangular pattern away
      from the frame, repeating periodically until the grid boundary.
    - Frame pixels are protected and must never be overwritten.

    Key insight: The source is no longer a symmetric plus whose arm picks the direction; the
    frame's unique opening side is the directional cue, and the attached marker supplies the tile.
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
        info = detect_hollow_frame_gap(frame_mask)
        if info is None:
            continue

        bbox = info["bbox"]
        side = str(info["side"])
        direction = tuple(map(int, info["direction"]))
        r0f, c0f, r1f, c1f = map(int, bbox)

        adj_ids = _adjacent_marker_ids_on_gap_side(mark_lab, frame_mask, bbox, side)
        for cid in sorted(adj_ids):
            pts = np.argwhere(mark_lab == int(cid))
            if pts.size == 0:
                continue
            r0, c0, r1, c1 = bbox_coords(pts)
            if side == "up" and int(r1) >= int(r0f):
                continue
            if side == "down" and int(r0) <= int(r1f):
                continue
            if side == "left" and int(c1) >= int(c0f):
                continue
            if side == "right" and int(c0) <= int(c1f):
                continue

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


# --------------- generator (output-first hollow-frame variant) ---------------

# --- grid / counts ---
_H_RANGE = (14, 29)            # [min, max)
_W_RANGE = (14, 24)            # [min, max)
_N_FRAME_RANGE = (2, 4)        # 2 or 3

# --- frame geometry ---
_FRAME_H_RANGE = (5, 9)        # [min, max)
_FRAME_W_RANGE = (5, 9)        # [min, max)
_FRAME_GAP_WIDTH = 1

# --- seed / ray geometry ---
_ALONG_CHOICES = [1, 2]
_CROSS_CHOICES = [1, 2]
_SEED_MAX_COLORS = 4
_MIN_RAY_UNITS = 2
_PRE_EXT_IDEAL_UNITS = 6

# --- retries ---
_PLACEMENT_TRIES = 180
_GEN_RETRIES = 400


def _build_frame_mask(h: int, w: int, side: str) -> np.ndarray:
    mask = np.zeros((int(h), int(w)), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    if side in ("up", "down"):
        gap = w // 2
        row = 0 if side == "up" else h - 1
        mask[row, gap : gap + _FRAME_GAP_WIDTH] = False
    else:
        gap = h // 2
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
    dr, dc = map(int, direction)
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
        if side == "up":
            sr0 = r0 - th
        else:
            sr0 = r1 + 1
        if sr0 < 0 or sr0 + th > H:
            return out
        for lc0 in range(1, fw - tw):
            cols = list(range(lc0, lc0 + tw))
            if all(col == gap for col in cols):
                continue
            if min(abs(col - gap) for col in cols) > 1:
                continue
            sc0 = c0 + lc0
            if 0 <= sc0 and sc0 + tw <= W:
                out.append((int(sr0), int(sc0)))
    else:
        if side == "left":
            sc0 = c0 - tw
        else:
            sc0 = c1 + 1
        if sc0 < 0 or sc0 + tw > W:
            return out
        for lr0 in range(1, fh - th):
            rows = list(range(lr0, lr0 + th))
            if all(row == gap for row in rows):
                continue
            if min(abs(row - gap) for row in rows) > 1:
                continue
            sr0 = r0 + lr0
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
    direction: tuple[int, int],
    tile: np.ndarray,
    seed_r0: int,
    seed_c0: int,
    H: int,
    W: int,
) -> int:
    th, tw = tile.shape
    dr, dc = map(int, direction)
    if dr == 1:
        return int((H - (seed_r0 + th)) // th)
    if dr == -1:
        return int(seed_r0 // th)
    if dc == 1:
        return int((W - (seed_c0 + tw)) // tw)
    return int(seed_c0 // tw)


def _touches_existing(occ: np.ndarray, r: int, c: int) -> bool:
    H, W = occ.shape
    if bool(occ[r, c]):
        return True
    for dr, dc in _DIRECTIONS:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < H and 0 <= cc < W and bool(occ[rr, cc]):
            return True
    return False


def _place_frame_and_ray(
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

            for sr0, sc0 in seed_origins:
                units = _compute_ray_units(direction, tile, sr0, sc0, H, W)
                if units < _MIN_RAY_UNITS:
                    continue
                ray_cells = _project_tile_coords(H, W, sr0, sc0, tile, direction)
                ray_coords = [(int(r), int(c)) for r, c, _ in ray_cells]
                all_coords = frame_cells + ray_coords
                if len(set(all_coords)) != len(all_coords):
                    continue
                if any(_touches_existing(occ, rr, cc) for rr, cc in ray_coords):
                    continue

                for rr, cc in frame_cells:
                    occ[rr, cc] = True
                    out[rr, cc] = int(frame_color)
                for rr, cc, v in ray_cells:
                    occ[rr, cc] = True
                    out[rr, cc] = int(v)

                return {
                    "direction": tuple(map(int, direction)),
                    "tile": tile,
                    "cells": ray_cells,
                    "seed_r0": int(sr0),
                    "seed_c0": int(sc0),
                    "units": int(units),
                }

    return None


def _build_input_from_output(
    out: np.ndarray,
    frames: list[dict[str, object]],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    pre_idx = max(range(len(frames)), key=lambda i: int(frames[i]["units"]))
    if int(frames[pre_idx]["units"]) < _PRE_EXT_IDEAL_UNITS:
        candidates = [i for i, item in enumerate(frames) if int(item["units"]) >= _PRE_EXT_IDEAL_UNITS]
        if candidates:
            pre_idx = int(rng.choice(candidates))

    inp = out.copy()
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


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [c for c in range(10) if c != bg]
    frame_color = int(rng.choice(others))
    palette = [c for c in others if c != frame_color]

    frame_h = int(rng.integers(*_FRAME_H_RANGE))
    frame_w = int(rng.integers(*_FRAME_W_RANGE))
    N = int(rng.integers(*_N_FRAME_RANGE))

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=bool)

    frames: list[dict[str, object]] = []
    for _ in range(N):
        info = _place_frame_and_ray(
            rng, frame_h, frame_w, palette, H, W, occ, out, frame_color
        )
        if info is None:
            return None
        frames.append(info)

    if len({tuple(item["direction"]) for item in frames}) < 2:
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
