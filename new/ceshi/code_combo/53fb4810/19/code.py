# Combo variant of 53fb4810 (index 19)
# combo_spec: D4/0+D6/2+D11/0
# Name: 矩形框架多射线精灵间停止

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
# a) Concrete behavior and trace
# - Original 53fb4810: a solid '+'-like sprite had one adjacent seed tile, and that tile
#   repeated to the boundary. Only one ray was already expanded in the input.
# - D4/0 changes the anchor object into a hollow rectangular frame. The direction cue is no
#   longer an arm tip; it is the frame side that contains an outer-rim gap.
# - D11/0 requires several independent seeds per source object. For frames, the coherent
#   generalization is: one frame may have multiple gapped sides, and each gapped side owns
#   one seed tile on that side.
# - D6/2 changes the endpoint: repetition stops at the next frame on the same corridor,
#   not at the canvas border. The one pre-expanded teaching example is therefore a bridge
#   that touches its source frame on one end and another frame on the other.
# - Small trace:
#   old: plus + right seed [2,3] -> fill rightward until the border.
#   combo: top-left hollow frame has a gap on its right side and another on its bottom side;
#   a 1x2 seed on the right side repeats rightward only until the top-right frame, while a
#   second seed on the bottom side repeats downward only until the bottom-left frame.
#
# b) Complexity calibration
# - Every generated grid contains four identical frames arranged on the corners of a larger
#   rectangle. Each frame has exactly two marked sides, so "multi-marker per source" is
#   obvious but not noisy.
# - Opposite directions on the same corridor use parallel lanes. This prevents ray overlap
#   while still producing repeated evidence of the stop-before-frame rule on all four sides.
# - Gap width stays fixed at one cell, tile cross-span stays in {1,2}, and along-span stays
#   in {1,2}; the rule remains human-readable instead of introducing too many degrees of
#   freedom at once.
#
# c) Hidden assumptions audited from the original main()
# - "Dominant big component color" still works, but it now denotes sparse hollow borders.
# - The original solver inferred direction from marker adjacency to a plus arm. That is
#   invalid here: adjacency only tells which frame a component belongs to, while the side's
#   gap tells which outward normal to use.
# - The original solver could safely treat every adjacent component as a seed. That fails
#   once the input contains a fully expanded bridge, because that bridge touches two frames.
#   The fix is to only treat components adjacent to exactly one frame as compact seeds.
# - The old endpoint helper assumed the boundary is the stop. Here the next frame on the lane
#   is the stop condition, so a "find later frame" search replaces the boundary-only logic.
#
# d) Placement legality
# - The generator is output-first. It places four disjoint thick-rim frames, then validates
#   eight rays (two per frame) before committing.
# - Rays never overlap and never 4-touch a foreign ray; otherwise multiple logical markers
#   could merge into one connected component and break seed extraction.
# - Each active frame side has an outer-rim gap, but ray lanes are placed away from the gap
#   cell so that incoming bridges stop on the target frame border rather than leaking through
#   the opening.
#
# e) Visual observability
# - A human can see: the source objects are frames, not pluses; each frame has multiple
#   attached seeds; and the one long example ray ends exactly at another frame instead of
#   running to the border.
# - The same frame-to-frame stopping interaction appears on multiple corridors in the output,
#   so the changed endpoint rule is not a one-off special case.
#
# f) Visual cleanliness
# - The rectangle-corner layout keeps the eight rays in separate horizontal/vertical bands.
#   There is no heavy overlap and the empty frame interiors remain visible.
# - Keeping only one full bridge in the input avoids turning the prompt into a dense grid of
#   stripes while still exposing the periodic repetition mechanism.
#
# g) Color inferability
# - Colors are copied directly from each seed tile; there is no color remapping.
# - Palette size per tile stays small, so the task remains about geometric propagation and
#   stopping conditions, not hidden color arithmetic.


_DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_SIDE_TO_DIR = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


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


def detect_hollow_frame_gap_sides(frame_mask: np.ndarray) -> dict[str, object] | None:
    """Validate a 2-cell-thick hollow frame and return its bbox plus all gapped sides."""
    pts = np.argwhere(frame_mask)
    if pts.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(pts)
    local = frame_mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    h, w = local.shape
    if h < 7 or w < 7:
        return None
    if np.any(local[2:-2, 2:-2]):
        return None
    if not np.all(local[1, :]):
        return None
    if not np.all(local[-2, :]):
        return None
    if not np.all(local[:, 1]):
        return None
    if not np.all(local[:, -2]):
        return None
    if not (bool(local[0, 0]) and bool(local[0, -1]) and bool(local[-1, 0]) and bool(local[-1, -1])):
        return None

    side_arrays = {
        "up": local[0, 1:-1],
        "down": local[-1, 1:-1],
        "left": local[1:-1, 0],
        "right": local[1:-1, -1],
    }

    gap_sides: list[str] = []
    gap_index: dict[str, int] = {}
    for side, arr in side_arrays.items():
        miss = np.where(~arr)[0]
        if miss.size == 0:
            continue
        if miss.size != 1:
            return None
        gap_sides.append(side)
        gap_index[side] = int(miss[0] + 1)

    if not gap_sides:
        return None

    for side, arr in side_arrays.items():
        miss_pos = gap_index.get(side)
        for idx, val in enumerate(arr, start=1):
            should_be_filled = miss_pos != idx
            if bool(val) != bool(should_be_filled):
                return None

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "gap_sides": tuple(gap_sides),
        "gap_index": gap_index,
    }


def _adjacent_frame_ids(component_mask: np.ndarray, frame_lab: np.ndarray) -> set[int]:
    ids: set[int] = set()
    H, W = frame_lab.shape
    for r, c in np.argwhere(component_mask):
        r = int(r)
        c = int(c)
        for dr, dc in _DIRECTIONS:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < H and 0 <= cc < W and int(frame_lab[rr, cc]) > 0:
                ids.add(int(frame_lab[rr, cc]))
    return ids


def _adjacent_marker_ids_on_side(
    mark_lab: np.ndarray,
    frame_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    side: str,
) -> set[int]:
    """Collect non-frame components that touch one specific frame side."""
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


def _component_touch_sides(
    pts: np.ndarray,
    frame_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> set[str]:
    """Return which side(s) of the frame a component is adjacent to."""
    H, W = frame_mask.shape
    r0, c0, r1, c1 = map(int, bbox)
    sides: set[str] = set()

    for pr, pc in pts:
        pr = int(pr)
        pc = int(pc)
        for dr, dc in _DIRECTIONS:
            rr = pr + dr
            cc = pc + dc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if not bool(frame_mask[rr, cc]):
                continue
            if rr == r0:
                sides.add("up")
            if rr == r1:
                sides.add("down")
            if cc == c0:
                sides.add("left")
            if cc == c1:
                sides.add("right")
    return sides


def _find_stop_before_frame(
    frame_mask_all: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    direction: tuple[int, int],
) -> int | None:
    """Return the first row/col occupied by a later frame on the ray lane."""
    H, W = frame_mask_all.shape
    dr, dc = map(int, direction)

    if (dr, dc) == (0, 1):
        for cc in range(int(c0 + w), W):
            if bool(np.any(frame_mask_all[int(r0) : int(r0 + h), int(cc)])):
                return int(cc)
    elif (dr, dc) == (0, -1):
        for cc in range(int(c0 - 1), -1, -1):
            if bool(np.any(frame_mask_all[int(r0) : int(r0 + h), int(cc)])):
                return int(cc)
    elif (dr, dc) == (1, 0):
        for rr in range(int(r0 + h), H):
            if bool(np.any(frame_mask_all[int(rr), int(c0) : int(c0 + w)])):
                return int(rr)
    elif (dr, dc) == (-1, 0):
        for rr in range(int(r0 - 1), -1, -1):
            if bool(np.any(frame_mask_all[int(rr), int(c0) : int(c0 + w)])):
                return int(rr)
    else:
        raise ValueError("_find_stop_before_frame: direction must be axial")
    return None


def extend_repeating_tile_until_frame(
    out: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    bg: int,
    frame_mask_all: np.ndarray,
    protect: set[int] | None = None,
) -> None:
    """Repeat a tile away from its source until the next frame on that corridor."""
    protect = set() if protect is None else set(map(int, protect))
    h, w = tile.shape
    dr, dc = map(int, direction)
    dr = 0 if dr == 0 else int(dr / abs(dr))
    dc = 0 if dc == 0 else int(dc / abs(dc))
    bg = int(bg)
    r0 = int(r0)
    c0 = int(c0)

    stop = _find_stop_before_frame(frame_mask_all, r0, c0, int(h), int(w), (dr, dc))
    if stop is None:
        return

    def put(rr: int, cc: int, v: int) -> None:
        if v == bg:
            return
        if int(out[rr, cc]) in protect:
            return
        out[rr, cc] = int(v)

    if (dr, dc) == (0, 1):
        for k, tc in enumerate(range(int(c0 + w), int(stop))):
            col_idx = k % int(w)
            for i in range(int(h)):
                put(int(r0 + i), int(tc), int(tile[i, col_idx]))
    elif (dr, dc) == (0, -1):
        for k, tc in enumerate(range(int(c0 - 1), int(stop), -1), start=1):
            col_idx = (-k) % int(w)
            for i in range(int(h)):
                put(int(r0 + i), int(tc), int(tile[i, col_idx]))
    elif (dr, dc) == (1, 0):
        for k, tr in enumerate(range(int(r0 + h), int(stop))):
            row_idx = k % int(h)
            for j in range(int(w)):
                put(int(tr), int(c0 + j), int(tile[row_idx, j]))
    elif (dr, dc) == (-1, 0):
        for k, tr in enumerate(range(int(r0 - 1), int(stop), -1), start=1):
            row_idx = (-k) % int(h)
            for j in range(int(w)):
                put(int(tr), int(c0 + j), int(tile[row_idx, j]))
    else:
        raise ValueError("extend_repeating_tile_until_frame: direction must be axial")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend every compact seed on a gapped side of a hollow frame until it reaches another frame.

    Input structure:
    - The background is the most frequent color.
    - One dominant non-background color forms several hollow rectangular frames whose walls are thick enough to stay connected even when some outer-rim cells are missing.
    - A frame may have multiple gap-bearing sides, each with one small colored seed just outside it; one seed somewhere in the grid may already appear as a full bridge between two frames.

    Rule:
    - Detect each frame and read every side that contains a border gap.
    - For each compact seed component that touches exactly one frame on one of those sides, repeat its rectangular pattern outward along that side's normal direction.
    - Stop the repetition when the corridor reaches the next frame, and never overwrite frame pixels.

    Key insight:
    - The frame side with the gap, not the seed's color, determines the direction; and the long example bridge must be ignored as a seed because it touches two different frames rather than one.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()

    bg = int(get_bg(g))
    frame_col = detect_dominant_big_component_color(g, bg=bg, min_h=5, min_w=5, connectivity=4)
    if frame_col is None:
        return out

    frame_lab, fn = label_mask(g == int(frame_col), connectivity=4)
    frame_mask_all = frame_lab > 0

    frame_infos: dict[int, dict[str, object]] = {}
    for fk in range(1, int(fn) + 1):
        info = detect_hollow_frame_gap_sides(frame_lab == int(fk))
        if info is not None:
            frame_infos[int(fk)] = info

    non_mask = (g != int(bg)) & (g != int(frame_col))
    mark_lab, mn = label_mask(non_mask, connectivity=4)
    comp_frame_ids: dict[int, set[int]] = {}
    for cid in range(1, int(mn) + 1):
        comp_frame_ids[int(cid)] = _adjacent_frame_ids(mark_lab == int(cid), frame_lab)

    for fk, info in frame_infos.items():
        frame_mask = frame_lab == int(fk)
        bbox = tuple(map(int, info["bbox"]))
        fr0, fc0, fr1, fc1 = bbox

        for side in info["gap_sides"]:
            direction = _SIDE_TO_DIR[str(side)]
            adj_ids = _adjacent_marker_ids_on_side(mark_lab, frame_mask, bbox, str(side))
            for cid in sorted(adj_ids):
                if comp_frame_ids.get(int(cid), set()) != {int(fk)}:
                    continue

                pts = np.argwhere(mark_lab == int(cid))
                if pts.size == 0:
                    continue
                if _component_touch_sides(pts, frame_mask, bbox) != {str(side)}:
                    continue

                r0, c0, r1, c1 = bbox_coords(pts)
                if side == "up" and int(r1) >= int(fr0):
                    continue
                if side == "down" and int(r0) <= int(fr1):
                    continue
                if side == "left" and int(c1) >= int(fc0):
                    continue
                if side == "right" and int(c0) <= int(fc1):
                    continue

                tile = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))
                extend_repeating_tile_until_frame(
                    out,
                    tile,
                    int(r0),
                    int(c0),
                    tuple(map(int, direction)),
                    bg=int(bg),
                    frame_mask_all=frame_mask_all,
                    protect={int(frame_col)},
                )

    return out


# --------------- generator (combo output-first frame-bridge variant) ---------------

_FRAME_H_RANGE = (8, 11)
_FRAME_W_RANGE = (8, 11)
_CORRIDOR_GAP_RANGE = (8, 13)
_OUTER_MARGIN_CHOICES = [1, 2, 3]
_ALONG_CHOICES = [1, 2]
_CROSS_CHOICES = [1, 2]
_SEED_MAX_COLORS = 4
_GEN_RETRIES = 400

_FRAME_GAP_SIDES = {
    0: ("right", "down"),  # top-left
    1: ("left", "down"),   # top-right
    2: ("left", "up"),     # bottom-right
    3: ("right", "up"),    # bottom-left
}


def _build_frame_mask(h: int, w: int, gap_sides: tuple[str, ...]) -> np.ndarray:
    mask = np.ones((int(h), int(w)), dtype=bool)
    mask[2:-2, 2:-2] = False

    gap_row = int(h // 2)
    gap_col = int(w // 2)
    for side in gap_sides:
        if side == "up":
            mask[0, gap_col] = False
        elif side == "down":
            mask[-1, gap_col] = False
        elif side == "left":
            mask[gap_row, 0] = False
        elif side == "right":
            mask[gap_row, -1] = False
    return mask


def _build_seed_tile(
    direction: tuple[int, int],
    palette: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    cross = int(rng.choice(_CROSS_CHOICES))
    along = int(rng.choice(_ALONG_CHOICES))
    if cross == 1 and along == 1:
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
        tile.flat[-1] = int(rng.choice([c for c in palette if c != int(tile.flat[0])]))
    return tile


def _seed_origin(
    frame_bbox: tuple[int, int, int, int],
    side: str,
    slot: str,
    tile: np.ndarray,
) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, frame_bbox)
    th, tw = tile.shape

    if side == "right":
        sr0 = r0 + 1 if slot == "top" else r1 - th
        sc0 = c1 + 1
        return int(sr0), int(sc0)
    if side == "left":
        sr0 = r0 + 1 if slot == "top" else r1 - th
        sc0 = c0 - tw
        return int(sr0), int(sc0)
    if side == "down":
        sr0 = r1 + 1
        sc0 = c0 + 1 if slot == "left" else c1 - tw
        return int(sr0), int(sc0)
    if side == "up":
        sr0 = r0 - th
        sc0 = c0 + 1 if slot == "left" else c1 - tw
        return int(sr0), int(sc0)
    raise ValueError("_seed_origin: invalid side")


def _find_stop_before_frame_generator(
    frame_mask_all: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    direction: tuple[int, int],
) -> int | None:
    return _find_stop_before_frame(frame_mask_all, r0, c0, h, w, direction)


def _collect_ray_cells(
    frame_mask_all: np.ndarray,
    seed_r0: int,
    seed_c0: int,
    tile: np.ndarray,
    direction: tuple[int, int],
) -> tuple[list[tuple[int, int, int]], int] | None:
    """Return full ray cells including seed, plus its tail length in cells."""
    th, tw = tile.shape
    stop = _find_stop_before_frame_generator(
        frame_mask_all, int(seed_r0), int(seed_c0), int(th), int(tw), tuple(map(int, direction))
    )
    if stop is None:
        return None

    cells: list[tuple[int, int, int]] = []
    for i in range(int(th)):
        for j in range(int(tw)):
            cells.append((int(seed_r0 + i), int(seed_c0 + j), int(tile[i, j])))

    dr, dc = map(int, direction)
    tail_len = 0
    if (dr, dc) == (0, 1):
        for k, tc in enumerate(range(int(seed_c0 + tw), int(stop))):
            tail_len += 1
            col_idx = k % int(tw)
            for i in range(int(th)):
                cells.append((int(seed_r0 + i), int(tc), int(tile[i, col_idx])))
    elif (dr, dc) == (0, -1):
        for k, tc in enumerate(range(int(seed_c0 - 1), int(stop), -1), start=1):
            tail_len += 1
            col_idx = (-k) % int(tw)
            for i in range(int(th)):
                cells.append((int(seed_r0 + i), int(tc), int(tile[i, col_idx])))
    elif (dr, dc) == (1, 0):
        for k, tr in enumerate(range(int(seed_r0 + th), int(stop))):
            tail_len += 1
            row_idx = k % int(th)
            for j in range(int(tw)):
                cells.append((int(tr), int(seed_c0 + j), int(tile[row_idx, j])))
    elif (dr, dc) == (-1, 0):
        for k, tr in enumerate(range(int(seed_r0 - 1), int(stop), -1), start=1):
            tail_len += 1
            row_idx = (-k) % int(th)
            for j in range(int(tw)):
                cells.append((int(tr), int(seed_c0 + j), int(tile[row_idx, j])))
    else:
        raise ValueError("_collect_ray_cells: direction must be axial")

    return cells, int(tail_len)


def _ray_cells_fit(
    ray_occ: np.ndarray,
    cells: list[tuple[int, int, int]],
) -> bool:
    """Require distinct ray components to stay separated by at least one background cell."""
    H, W = ray_occ.shape
    cell_pos = {(int(r), int(c)) for r, c, _ in cells}
    if len(cell_pos) != len(cells):
        return False

    for r, c, _ in cells:
        rr = int(r)
        cc = int(c)
        if not (0 <= rr < H and 0 <= cc < W):
            return False
        if bool(ray_occ[rr, cc]):
            return False
        for dr, dc in _DIRECTIONS:
            nr = rr + dr
            nc = cc + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if (nr, nc) in cell_pos:
                continue
            if bool(ray_occ[nr, nc]):
                return False
    return True


def _build_input_from_output(
    out: np.ndarray,
    rays: list[dict[str, object]],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Keep one full bridge as the exemplar and collapse the rest back to seeds."""
    # Every accepted ray already has a visible tail, so choose the teaching bridge uniformly
    # instead of biasing toward the longest corridor. This improves branch coverage across
    # all four directions and all four source frames.
    pre_idx = int(rng.integers(len(rays)))

    inp = out.copy()
    for i, ray in enumerate(rays):
        if i == pre_idx:
            continue
        tile = np.asarray(ray["tile"], dtype=int)
        th, tw = tile.shape
        sr0 = int(ray["seed_r0"])
        sc0 = int(ray["seed_c0"])
        seed_set = {(int(sr0 + ii), int(sc0 + jj)) for ii in range(int(th)) for jj in range(int(tw))}
        for r, c, _ in ray["cells"]:
            if (int(r), int(c)) not in seed_set:
                inp[int(r), int(c)] = int(bg)
    return inp


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    frame_h = int(rng.integers(*_FRAME_H_RANGE))
    frame_w = int(rng.integers(*_FRAME_W_RANGE))
    gap_y = int(rng.integers(*_CORRIDOR_GAP_RANGE))
    gap_x = int(rng.integers(*_CORRIDOR_GAP_RANGE))
    top_margin = int(rng.choice(_OUTER_MARGIN_CHOICES))
    bottom_margin = int(rng.choice(_OUTER_MARGIN_CHOICES))
    left_margin = int(rng.choice(_OUTER_MARGIN_CHOICES))
    right_margin = int(rng.choice(_OUTER_MARGIN_CHOICES))

    H = int(top_margin + frame_h + gap_y + frame_h + bottom_margin)
    W = int(left_margin + frame_w + gap_x + frame_w + right_margin)

    bg = int(rng.integers(0, 10))
    others = [c for c in range(10) if c != bg]
    frame_color = int(rng.choice(others))
    palette = [c for c in others if c != frame_color]

    out = np.full((H, W), bg, dtype=int)
    frame_mask_all = np.zeros((H, W), dtype=bool)

    top = int(top_margin)
    left = int(left_margin)
    bottom = int(top + frame_h + gap_y)
    right = int(left + frame_w + gap_x)

    frame_boxes = [
        (top, left, top + frame_h - 1, left + frame_w - 1),
        (top, right, top + frame_h - 1, right + frame_w - 1),
        (bottom, right, bottom + frame_h - 1, right + frame_w - 1),
        (bottom, left, bottom + frame_h - 1, left + frame_w - 1),
    ]

    for idx, bbox in enumerate(frame_boxes):
        r0, c0, _r1, _c1 = map(int, bbox)
        mask = _build_frame_mask(frame_h, frame_w, _FRAME_GAP_SIDES[idx])
        for rr in range(frame_h):
            for cc in range(frame_w):
                if bool(mask[rr, cc]):
                    ar = int(r0 + rr)
                    ac = int(c0 + cc)
                    frame_mask_all[ar, ac] = True
                    out[ar, ac] = int(frame_color)

    ray_specs = [
        {"src_idx": 0, "side": "right", "direction": (0, 1), "slot": "top"},
        {"src_idx": 1, "side": "left", "direction": (0, -1), "slot": "bottom"},
        {"src_idx": 3, "side": "right", "direction": (0, 1), "slot": "top"},
        {"src_idx": 2, "side": "left", "direction": (0, -1), "slot": "bottom"},
        {"src_idx": 0, "side": "down", "direction": (1, 0), "slot": "left"},
        {"src_idx": 3, "side": "up", "direction": (-1, 0), "slot": "right"},
        {"src_idx": 1, "side": "down", "direction": (1, 0), "slot": "left"},
        {"src_idx": 2, "side": "up", "direction": (-1, 0), "slot": "right"},
    ]

    ray_occ = np.zeros((H, W), dtype=bool)
    rays: list[dict[str, object]] = []
    for spec in ray_specs:
        tile = _build_seed_tile(tuple(map(int, spec["direction"])), palette, rng)
        seed_r0, seed_c0 = _seed_origin(frame_boxes[int(spec["src_idx"])], str(spec["side"]), str(spec["slot"]), tile)
        ray_info = _collect_ray_cells(
            frame_mask_all,
            int(seed_r0),
            int(seed_c0),
            tile,
            tuple(map(int, spec["direction"])),
        )
        if ray_info is None:
            return None

        cells, tail_len = ray_info
        if int(tail_len) < max(int(tile.shape[0] + tile.shape[1]) - 1, 4):
            return None
        if any(bool(frame_mask_all[int(r), int(c)]) for r, c, _ in cells):
            return None
        if not _ray_cells_fit(ray_occ, cells):
            return None

        for r, c, v in cells:
            ray_occ[int(r), int(c)] = True
            out[int(r), int(c)] = int(v)
        rays.append(
            {
                "src_idx": int(spec["src_idx"]),
                "side": str(spec["side"]),
                "direction": tuple(map(int, spec["direction"])),
                "slot": str(spec["slot"]),
                "tile": tile,
                "cells": cells,
                "seed_r0": int(seed_r0),
                "seed_c0": int(seed_c0),
                "tail_len": int(tail_len),
            }
        )

    inp = _build_input_from_output(out, rays, bg, rng)

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
