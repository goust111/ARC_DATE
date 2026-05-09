# Combo variant of 5961cc34 (index 19)
# combo_spec: D2/0+D5/0+D14/2
# Name: DFS激活+多端口色块+交替方向处理

from __future__ import annotations

from collections import Counter
from itertools import combinations
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste


# ---------------------------------------------------------------------------
# Design analysis for the combo variant D2/0 + D5/0 + D14/2
#
# a) concretization + behavior trace:
#    - Concrete scene:
#      The board still contains one bottom start marker, blue diamond/octagon blobs,
#      and green edge bars. The fusion change is that every blob has 2 or 3 distinct
#      ports, ports are explored in DFS order, and the processing step alternates
#      axes globally: step 1 allows only horizontal ports, step 2 only vertical,
#      step 3 only horizontal again, and so on.
#    - Original trace:
#      start ray -> first blob -> that blob emits one ray -> BFS expands by layers.
#      With one port per blob the output is close to a single chain.
#    - Fused trace used here:
#      start ray -> root blob -> root fires both horizontal ports on step 1 and
#      activates a left child and a right child. Ports are processed in visible
#      row-major order of their green markers, so the lower side port is hit later.
#      Because we use DFS, that later-hit child is processed first on step 2 and
#      gets the vertical turn. The earlier sibling is postponed to step 3, where
#      vertical ports are no longer legal, so it stays red but emits nothing.
#    - Small mental example:
#      Root has left/right/(optional up) ports. Both side children visibly carry
#      up+down green ports. In the output both children turn red, but only the
#      later-hit side child gets vertical rays because DFS reaches it at the even
#      step. Under a BFS alternating schedule both children would have emitted
#      vertically, so this is a real D2 x D14 interaction, not just two edits.
#
# b) complexity calibration:
#    - The generator forces one root plus two side children, so multi-port branching
#      is always visible.
#    - Both children are constructed with legal vertical ports and empty vertical
#      corridors. One child is intentionally suppressed by the DFS step schedule,
#      which makes the fusion behavior observable from the pixels alone.
#    - Port counts are restricted to 2 or 3 and all rays remain axis-aligned, which
#      keeps the task learnable without collapsing back to the original linear rule.
#
# c) implicit assumption review:
#    - Original code assumed blue-only connected components were sufficient. That is
#      unsafe with several green sides, so the new solver groups blobs by the combined
#      blue+green footprint and then records blue pixels separately for activation.
#    - Original queue logic assumed every activated blob eventually emits. That is
#      false now: a blob is processed exactly once, and only the ports whose axis
#      matches that blob's global DFS step are allowed to fire.
#    - Original propagation order was FIFO and layer-based. The new solver must use
#      a LIFO stack so that "latest activated blob first" is real, otherwise the
#      sibling-suppression effect disappears.
#    - Port parsing can no longer assume exactly one direction; all 2-3 distinct
#      green-bearing sides must be collected in a deterministic visible order.
#
# d) placement legality:
#    - The generator is still output-first. It first fixes a legal root hit, then
#      places one child on the left root port and one on the right root port.
#    - Blob footprints are kept 8-neighbor separated, rays are forbidden to cross
#      existing rays or foreign blob cells, and all placements stay in bounds.
#    - The suppressed child is also required to have clear vertical corridors, so
#      its missing rays are caused by the fused schedule rather than by blockage.
#
# e) visual observability:
#    - A human sees a branching root with multiple green exits, two activated side
#      children, and only one of those children actually continuing vertically.
#    - When the optional third root port points upward, the input also shows a green
#      upward exit that remains unused because step 1 only allows horizontal firing.
#    - No rule depends on numeric color ordering or hidden metadata; the evidence is
#      entirely geometric.
#
# f) cleanliness:
#    - The generator uses one root and two side children on opposite sides, which
#      keeps rays separated and easy to track.
#    - The root's left/right green bars are intentionally placed at different
#      heights, so the DFS tie-break is visible from the input instead of looking
#      like a hard-coded "left wins" or "right wins" positional rule.
#    - The continuing child emits only vertical rays, so the picture remains readable
#      instead of becoming a dense crossing mesh.
#
# g) inferability of the change:
#    - Blue+green still means "inactive blob with encoded exits", red still means
#      "activated body or traversed ray". The new part is the process schedule:
#      which exits actually fire depends on the DFS order and the odd/even axis turn.
#    - The generator guarantees at least one suppressed-but-otherwise-legal branch,
#      so samples do not degenerate into outputs that would also be explained by a
#      plain BFS multi-port solver.
#
# New main() assumptions audited against generate_input():
#    1. There is exactly one bottom red post with a non-red cap.
#    2. Blue is the dominant non-bg/start color and green is the unique color
#       4-adjacent to blue.
#    3. Every blob is one 8-connected blue+green component with 2 or 3 distinct
#       ports, each port width 1 or 2.
#    4. Incoming rays always enter through blue on the entry side, never through
#       a green marker.
#    5. Port processing order is deterministic and visible: ports are handled by
#       the row-major position of their green markers.
#    6. The intended suppressed child has legal vertical lanes even though our
#       fused solver will not draw them.
#    The generator explicitly enforces all six assumptions.
# ---------------------------------------------------------------------------


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
DIR_ORDER = (LEFT, RIGHT, UP, DOWN)
OPP = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

# --- Diamond/octagon shapes (h, w) ---
SHAPES = ((3, 4), (4, 4), (5, 4), (5, 6), (6, 6), (8, 8))
ROOT_SIZES = ((4, 4), (5, 4), (5, 6), (6, 6))
CHILD_SIZES = ((3, 4), (4, 4), (5, 4), (5, 6))
_SHAPE_WIDTHS = {
    (3, 4): (2, 4, 2),
    (4, 4): (2, 4, 4, 2),
    (5, 4): (2, 4, 4, 4, 2),
    (5, 6): (2, 4, 6, 4, 2),
    (6, 6): (2, 4, 6, 6, 4, 2),
    (8, 8): (2, 4, 6, 8, 8, 6, 4, 2),
}

# --- Canvas bounds ---
H_MIN, H_MAX = 15, 27
W_MIN, W_MAX = 20, 27
START_COL_PAD = 6

# --- Start segment ---
START_H_MIN, START_H_MAX = 2, 3

# --- Placement ---
MIN_GAP_FIRST = 4
MIN_GAP_BRANCH = 3
MAX_DISTRACTORS = 1

# --- Retry ---
GEN_RETRIES = 500


def _axis_of(d: tuple[int, int]) -> str:
    return "V" if int(d[0]) else "H"


def _step_axis(step_idx: int) -> str:
    return "H" if int(step_idx) % 2 == 1 else "V"


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


def _port_anchor(coords: list[tuple[int, int]]) -> tuple[int, int]:
    return min((int(r), int(c)) for r, c in coords)


def _ordered_port_dirs(ports_by_dir: dict[tuple[int, int], list[tuple[int, int]]]) -> list[tuple[int, int]]:
    return [
        d
        for d, _pixels in sorted(
            ports_by_dir.items(),
            key=lambda item: (_port_anchor(item[1])[0], _port_anchor(item[1])[1], DIR_ORDER.index(item[0])),
        )
    ]


def _edge_offsets(mask: np.ndarray, d: tuple[int, int]) -> list[tuple[int, int]]:
    pts = np.argwhere(mask)
    axis = 0 if d in (UP, DOWN) else 1
    extreme = pts[:, axis].min() if d in (UP, LEFT) else pts[:, axis].max()
    keep = pts[:, axis] == extreme
    return _sort_on_ray([(int(r), int(c)) for r, c in pts[keep]], d)


def _center_port_window(edge: list[tuple[int, int]], d: tuple[int, int]) -> list[tuple[int, int]]:
    width = min(2, len(edge))
    start = (len(edge) - width) // 2
    return _sort_on_ray(edge[start : start + width], d)


def _is_contiguous_edge_window(edge: list[tuple[int, int]], d: tuple[int, int]) -> bool:
    edge = _sort_on_ray(edge, d)
    vals = [_perp_value(p, d) for p in edge]
    return all(vals[i + 1] == vals[i] + 1 for i in range(len(vals) - 1))


def _blob_spec_from_windows(
    size: tuple[int, int],
    port_windows: dict[tuple[int, int], list[tuple[int, int]]],
) -> dict | None:
    h, w = size
    mask = _shape_mask(h, w)

    ports_local: dict[tuple[int, int], list[tuple[int, int]]] = {}
    green_set: set[tuple[int, int]] = set()
    for d, raw_green in port_windows.items():
        green = _sort_on_ray(list(raw_green), d)
        edge_set = set(_edge_offsets(mask, d))
        if not green or len(green) > 2:
            return None
        if not set(green) <= edge_set:
            return None
        if not _is_contiguous_edge_window(green, d):
            return None
        if green_set & set(green):
            return None

        port_pixels: list[tuple[int, int]] = []
        for gr, gc in green:
            br = gr - d[0]
            bc = gc - d[1]
            if not (0 <= br < h and 0 <= bc < w):
                return None
            if not mask[br, bc] or (br, bc) in green_set:
                return None
            port_pixels.append((int(br), int(bc)))
        if len(set(port_pixels)) != len(port_pixels):
            return None

        green_set |= set(green)
        ports_local[d] = _sort_on_ray(port_pixels, d)

    blue_mask = mask.copy()
    for r, c in green_set:
        blue_mask[r, c] = False
    if len(components_mask(blue_mask, connectivity=8)) != 1:
        return None

    patch = np.full((h, w), BG, dtype=int)
    patch[mask] = BLUE
    for r, c in green_set:
        patch[r, c] = GREEN

    blue = [(int(r), int(c)) for r, c in np.argwhere(blue_mask)]
    full = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    return {
        "size": (h, w),
        "mask": mask,
        "patch": patch,
        "blue": blue,
        "full": full,
        "port_dirs": tuple(_ordered_port_dirs(ports_local)),
        "ports_by_dir": ports_local,
    }


def _blob_spec(size: tuple[int, int], port_dirs: tuple[tuple[int, int], ...]) -> dict | None:
    h, w = size
    mask = _shape_mask(h, w)
    port_windows: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for d in port_dirs:
        edge = _edge_offsets(mask, d)
        port_windows[d] = _center_port_window(edge, d)
    return _blob_spec_from_windows(size, port_windows)


def _build_all_specs() -> list[dict]:
    specs: list[dict] = []
    for size in SHAPES:
        for k in (2, 3):
            for dirs in combinations(DIRS, k):
                spec = _blob_spec(size, tuple(dirs))
                if spec is not None:
                    specs.append(spec)
    return specs


ALL_SPECS = _build_all_specs()


def _build_root_specs() -> list[dict]:
    specs: list[dict] = []
    for size in ROOT_SIZES:
        mask = _shape_mask(*size)
        left_edge = _edge_offsets(mask, LEFT)
        right_edge = _edge_offsets(mask, RIGHT)
        up_edge = _edge_offsets(mask, UP)
        up_window = _center_port_window(up_edge, UP)
        for left_green in left_edge:
            for right_green in right_edge:
                if int(left_green[0]) == int(right_green[0]):
                    continue
                base = {
                    LEFT: [tuple(map(int, left_green))],
                    RIGHT: [tuple(map(int, right_green))],
                }
                for include_up in (False, True):
                    port_windows = dict(base)
                    if include_up:
                        port_windows[UP] = up_window
                    spec = _blob_spec_from_windows(size, port_windows)
                    if spec is not None:
                        specs.append(spec)
    return specs


ROOT_SPECS = _build_root_specs()


def _specs_for(
    dir_sets: set[frozenset[tuple[int, int]]] | None = None,
    sizes: tuple[tuple[int, int], ...] | None = None,
) -> list[dict]:
    allowed_sizes = set(sizes) if sizes is not None else None
    out = []
    for spec in ALL_SPECS:
        if allowed_sizes is not None and tuple(spec["size"]) not in allowed_sizes:
            continue
        if dir_sets is not None and frozenset(spec["port_dirs"]) not in dir_sets:
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


def _path_to_entries(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    entries: list[tuple[int, int]],
    include_origin: bool,
    H: int,
    W: int,
) -> list[tuple[int, int]] | None:
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for (r, c), (er, ec) in zip(starts, entries):
        if include_origin:
            cells.append((r, c))
        guard = H + W + 10
        while (r, c) != (er, ec):
            r += dr
            c += dc
            guard -= 1
            if guard < 0 or not (0 <= r < H and 0 <= c < W):
                return None
            cells.append((r, c))
    return cells


def _path_to_boundary(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for r, c in starts:
        if include_origin:
            cells.append((r, c))
        while True:
            r += dr
            c += dc
            if not (0 <= r < H and 0 <= c < W):
                break
            cells.append((r, c))
    return cells


def _mark_ray(out: np.ndarray, ray_mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    paint_coords(out, cells, RED)
    paint_coords(ray_mask, cells, True)


def _mark_mask(mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    paint_coords(mask, cells, True)


def _path_is_clear(
    path: list[tuple[int, int]],
    occupied: np.ndarray,
    ray_mask: np.ndarray,
    allowed: set[tuple[int, int]],
) -> bool:
    for r, c in path:
        if (r, c) in allowed:
            continue
        if occupied[r, c] or ray_mask[r, c]:
            return False
    return True


def _terminal_paths(
    blob: dict,
    occupied: np.ndarray,
    ray_mask: np.ndarray,
    H: int,
    W: int,
    dirs: tuple[tuple[int, int], ...] | None = None,
) -> dict[tuple[int, int], list[tuple[int, int]]] | None:
    allowed = set(blob["full"])
    tmp_ray = ray_mask.copy()
    out: dict[tuple[int, int], list[tuple[int, int]]] = {}
    use_dirs = dirs if dirs is not None else tuple(blob["ports_by_dir"].keys())
    for d in DIR_ORDER:
        if d not in use_dirs:
            continue
        path = _path_to_boundary(blob["ports_by_dir"][d], d, False, H, W)
        if not _path_is_clear(path, occupied, tmp_ray, allowed):
            return None
        _mark_mask(tmp_ray, path)
        out[d] = path
    return out


def _try_place_along_axis(
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

    entry_lines = list(range(low, high + 1))
    random.shuffle(entry_lines)
    candidates: list[dict] = []

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

        path = _path_to_entries(starts, d, entry_cells, include_origin, H, W)
        if path is None:
            continue
        entry_set = set(entry_cells)
        if any(blue_mask[r, c] for r, c in path if (r, c) not in entry_set):
            continue

        candidates.append(
            {
                "spec": spec,
                "r0": r0,
                "c0": c0,
                "entry": entry_cells,
                "path": path,
                "blue": _offset(spec["blue"], r0, c0),
                "full": full,
                "ports_by_dir": {
                    dd: _sort_on_ray(_offset(local, r0, c0), dd)
                    for dd, local in spec["ports_by_dir"].items()
                },
            }
        )

    return candidates


def _candidate_blob_options(
    H: int,
    W: int,
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
    dir_sets: set[frozenset[tuple[int, int]]] | None = None,
    sizes: tuple[tuple[int, int], ...] | None = None,
) -> list[dict]:
    starts = _sort_on_ray(starts, d)
    thickness = len(starts)
    lane_vals = [_perp_value(p, d) for p in starts]
    lane_delta = [v - lane_vals[0] for v in lane_vals]
    entry_side = OPP[d]
    occ_pad = _dilate8(occupied)
    min_gap = MIN_GAP_FIRST if include_origin else MIN_GAP_BRANCH
    candidates: list[dict] = []

    for spec in _specs_for(dir_sets=dir_sets, sizes=sizes):
        if entry_side in spec["port_dirs"]:
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
                _try_place_along_axis(
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

    random.shuffle(candidates)
    return candidates


def _place_blob(inp: np.ndarray, occupied: np.ndarray, blue_mask: np.ndarray, blob: dict) -> None:
    paste(inp, blob["spec"]["patch"], blob["r0"], blob["c0"], transparent=BG)
    for r, c in blob["full"]:
        occupied[r, c] = True
    for r, c in blob["blue"]:
        blue_mask[r, c] = True


def _paint_blob_red(out: np.ndarray, blob: dict) -> None:
    paint_coords(out, blob["blue"], RED)


def _place_distractor(
    inp: np.ndarray,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
) -> bool:
    H, W = inp.shape
    occ_pad = _dilate8(occupied)
    specs = ALL_SPECS.copy()
    random.shuffle(specs)
    candidates: list[dict] = []
    for spec in specs:
        h, w = spec["mask"].shape
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                full = _offset(spec["full"], r0, c0)
                if any(occ_pad[r, c] or ray_mask[r, c] for r, c in full):
                    continue
                candidates.append(
                    {
                        "spec": spec,
                        "r0": r0,
                        "c0": c0,
                        "full": full,
                        "blue": _offset(spec["blue"], r0, c0),
                    }
                )
    if not candidates:
        return False
    _place_blob(inp, occupied, blue_mask, random.choice(candidates))
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a red relay tree where multi-port blobs are explored by DFS under an
    alternating horizontal/vertical schedule.

    Input structure:
    - The background is the dominant color. Each blob is a blue diamond/octagon with
      2 or 3 green edge bars, and each green side marks one possible emission
      direction for that blob.
    - The start marker is still a yellow cap above a short red post that reaches the
      bottom edge.

    Rule:
    - Draw the start post and cap in red, shoot a thin red ray upward, and activate
      the first blue blob it hits.
    - Activated blobs are processed in DFS order: the most recently activated blob is
      handled next. Within a blob, ports are considered in the visible row-major
      order of their green markers. Processing step 1 allows only horizontal ports,
      step 2 only vertical ports, step 3 only horizontal ports again, and so on.
    - When a blob is handled, only the ports whose axis matches that step fire. Their
      thick rays travel until they leave the grid or hit a new blob, which turns red
      and is pushed onto the DFS stack. Unused ports never fire later.

    Key insight: Green edges now encode several possible exits, but a blob gets only
    one DFS turn, and that turn is axis-restricted by the global odd/even step index.
    The final red pattern therefore depends on both branching structure and activation
    order, not just on static reachability.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

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
            if cap in (bg, int(col)):
                continue
            start_candidates.append(
                (int(col), int(cap), int(c0), [(int(rr), int(c0)) for rr in range(r0, r1 + 1)])
            )

    assert len(start_candidates) == 1
    red_col, yellow_col, start_col, start_seg = start_candidates[0]

    counts = Counter(map(int, g.flatten()))
    other_cols = [
        (int(cnt), int(col))
        for col, cnt in counts.items()
        if int(col) not in (bg, red_col, yellow_col)
    ]
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
                rr = r + dr
                cc = c + dc
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

    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []
    combo_mask = (g == blue_col) | (g == green_col)
    for coords in components_mask(combo_mask, connectivity=8):
        full = [(int(r), int(c)) for r, c in coords]
        blue_pts = [(r, c) for r, c in full if int(g[r, c]) == blue_col]
        green_pts = [(r, c) for r, c in full if int(g[r, c]) == green_col]
        assert blue_pts and green_pts
        bi = len(blobs)
        for r, c in blue_pts:
            blob_id[r, c] = bi
        blobs.append({"full": full, "blue": blue_pts, "green": green_pts, "ports": []})

    for blob in blobs:
        full = blob["full"]
        blue_set = set(blob["blue"])
        green_pts = blob["green"]
        min_r = min(r for r, _c in full)
        max_r = max(r for r, _c in full)
        min_c = min(c for _r, c in full)
        max_c = max(c for _r, c in full)
        edge_greens = {
            UP: [(r, c) for r, c in green_pts if r == min_r],
            DOWN: [(r, c) for r, c in green_pts if r == max_r],
            LEFT: [(r, c) for r, c in green_pts if c == min_c],
            RIGHT: [(r, c) for r, c in green_pts if c == max_c],
        }
        ports = []
        for d in DIR_ORDER:
            greens = edge_greens[d]
            if not greens:
                continue
            port_pixels = []
            for gr, gc in greens:
                br = gr - d[0]
                bc = gc - d[1]
                assert (br, bc) in blue_set
                port_pixels.append((int(br), int(bc)))
            port_pixels = _sort_on_ray(port_pixels, d)
            assert 1 <= len(port_pixels) <= 2
            ports.append({"dir": d, "pixels": port_pixels, "order_key": _port_anchor(greens)})
        ports.sort(key=lambda port: (port["order_key"][0], port["order_key"][1], DIR_ORDER.index(port["dir"])))
        blob["ports"] = ports
        assert 2 <= len(ports) <= 3

    activated: set[int] = set()
    stack: list[int] = []

    def activate_blob_at(r: int, c: int, collector: list[int]) -> None:
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in activated:
            return
        activated.add(bi)
        collector.append(bi)
        paint_coords(out, blobs[bi]["blue"], red_col)

    for r, c in start_seg:
        out[r, c] = red_col
    top_r = min(int(r) for r, _c in start_seg)
    assert top_r > 0 and int(g[top_r - 1, start_col]) == yellow_col
    out[top_r - 1, start_col] = red_col

    first_hits: list[int] = []
    for r in range(top_r - 1, -1, -1):
        out[r, start_col] = red_col
        if int(g[r, start_col]) == blue_col:
            activate_blob_at(int(r), int(start_col), first_hits)
            break
    stack.extend(first_hits)

    def extend_thick(
        port_pixels: list[tuple[int, int]],
        d: tuple[int, int],
        collector: list[int],
    ) -> None:
        dr, dc = d
        for pr, pc in _sort_on_ray(port_pixels, d):
            r, c = int(pr), int(pc)
            while True:
                r += dr
                c += dc
                if not (0 <= r < H and 0 <= c < W):
                    break
                out[r, c] = red_col
                if int(g[r, c]) == blue_col:
                    activate_blob_at(int(r), int(c), collector)
                    break

    step_idx = 1
    while stack:
        bi = int(stack.pop())
        wanted_axis = _step_axis(step_idx)
        step_idx += 1
        newly: list[int] = []
        for port in blobs[bi]["ports"]:
            if _axis_of(port["dir"]) != wanted_axis:
                continue
            extend_thick(port["pixels"], port["dir"], newly)
        stack.extend(newly)

    return out


def _place_start(
    H: int,
    W: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    start_h = random.randint(START_H_MIN, START_H_MAX)
    cap_r = H - start_h - 1
    start_col = random.randint(START_COL_PAD, W - START_COL_PAD - 1)
    inp[cap_r, start_col] = YELLOW
    out[cap_r, start_col] = RED
    occupied[cap_r, start_col] = True
    ray_mask[cap_r, start_col] = True
    for r in range(cap_r + 1, H):
        inp[r, start_col] = RED
        out[r, start_col] = RED
        occupied[r, start_col] = True

    return inp, out, occupied, blue_mask, ray_mask, cap_r, start_col


def _manual_root_candidates(
    H: int,
    W: int,
    cap_r: int,
    start_col: int,
) -> list[dict]:
    candidates: list[dict] = []
    for spec in ROOT_SPECS:
        h, w = spec["mask"].shape
        down_edge = _edge_offsets(spec["mask"], DOWN)
        entry = down_edge[len(down_edge) // 2]
        c0 = start_col - entry[1]
        if not (0 <= c0 <= W - w):
            continue
        max_r0 = cap_r - MIN_GAP_FIRST - entry[0]
        if max_r0 < 1:
            continue
        r0_choices = list(range(1, max_r0 + 1))
        random.shuffle(r0_choices)
        for r0 in r0_choices[:8]:
            full = _offset(spec["full"], r0, c0)
            blue = _offset(spec["blue"], r0, c0)
            path = [(r, start_col) for r in range(cap_r, r0 + entry[0] + 1)]
            candidates.append(
                {
                    "spec": spec,
                    "r0": r0,
                    "c0": c0,
                    "full": full,
                    "blue": blue,
                    "entry": [(r0 + entry[0], start_col)],
                    "path": [(r, start_col) for r in range(cap_r, r0 + entry[0] - 1, -1)],
                    "ports_by_dir": {
                        d: _sort_on_ray(_offset(local, r0, c0), d)
                        for d, local in spec["ports_by_dir"].items()
                    },
                }
            )
    random.shuffle(candidates)
    for candidate in candidates:
        candidate["port_order"] = tuple(_ordered_port_dirs(candidate["ports_by_dir"]))
    return candidates


def _manual_child_candidates(
    H: int,
    W: int,
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
    source_full: set[tuple[int, int]],
) -> list[dict]:
    entry_side = OPP[d]
    thickness = len(starts)
    starts = _sort_on_ray(starts, d)
    lane_delta = [_perp_value(p, d) - _perp_value(starts[0], d) for p in starts]
    occ_pad = _dilate8(occupied)
    candidates: list[dict] = []

    for spec in _specs_for(dir_sets={frozenset((UP, DOWN))}, sizes=CHILD_SIZES):
        h, w = spec["mask"].shape
        edge = _edge_offsets(spec["mask"], entry_side)
        if len(edge) < thickness:
            continue
        windows = []
        for k in range(len(edge) - thickness + 1):
            window = edge[k : k + thickness]
            deltas = [_perp_value(p, d) - _perp_value(window[0], d) for p in window]
            if deltas == lane_delta:
                windows.append(window)
        random.shuffle(windows)

        for window in windows:
            r0 = starts[0][0] - window[0][0]
            if not (0 <= r0 <= H - h):
                continue

            if d == LEFT:
                max_c0 = starts[0][1] - MIN_GAP_BRANCH - window[0][1]
                c0_choices = list(range(0, min(max_c0, W - w) + 1))
                c0_choices.sort(reverse=True)
            else:
                min_c0 = starts[0][1] + MIN_GAP_BRANCH - window[0][1]
                c0_choices = list(range(max(0, min_c0), W - w + 1))
            random.shuffle(c0_choices)

            for c0 in c0_choices[:12]:
                full = _offset(spec["full"], r0, c0)
                if any(occ_pad[r, c] or ray_mask[r, c] for r, c in full):
                    continue

                entry_cells = [(r0 + r, c0 + c) for r, c in window]
                path = _path_to_entries(starts, d, entry_cells, False, H, W)
                if path is None:
                    continue
                entry_set = set(entry_cells)
                if any(blue_mask[r, c] for r, c in path if (r, c) not in entry_set):
                    continue
                allowed = set(source_full) | entry_set
                if not _path_is_clear(path, occupied, ray_mask, allowed):
                    continue

                blob = {
                    "spec": spec,
                    "r0": r0,
                    "c0": c0,
                    "entry": entry_cells,
                    "path": path,
                    "blue": _offset(spec["blue"], r0, c0),
                    "full": full,
                    "ports_by_dir": {
                        dd: _sort_on_ray(_offset(local, r0, c0), dd)
                        for dd, local in spec["ports_by_dir"].items()
                    },
                }
                terms = _terminal_paths(blob, occupied, ray_mask, H, W, dirs=(UP, DOWN))
                if terms is None:
                    continue
                blob["terminal_paths"] = terms
                candidates.append(blob)
    random.shuffle(candidates)
    return candidates


def _build_scene(
    H: int,
    W: int,
    inp: np.ndarray,
    out: np.ndarray,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
    cap_r: int,
    start_col: int,
) -> dict | None:
    root_candidates = _manual_root_candidates(H, W, cap_r, start_col)

    for root in root_candidates:
        inp_root = inp.copy()
        out_root = out.copy()
        occupied_root = occupied.copy()
        blue_root = blue_mask.copy()
        ray_root = ray_mask.copy()

        _mark_ray(out_root, ray_root, root["path"])
        _place_blob(inp_root, occupied_root, blue_root, root)
        _paint_blob_red(out_root, root)

        ignored_root: dict[tuple[int, int], list[tuple[int, int]]] = {}
        if UP in root["ports_by_dir"]:
            root_term = _terminal_paths(root, occupied_root, ray_root, H, W, dirs=(UP,))
            if root_term is None:
                continue
            ignored_root = root_term

        right_options = _manual_child_candidates(
            H,
            W,
            root["ports_by_dir"][RIGHT],
            RIGHT,
            occupied_root,
            blue_root,
            ray_root,
            set(root["full"]),
        )
        if not right_options:
            continue

        left_options = _manual_child_candidates(
            H,
            W,
            root["ports_by_dir"][LEFT],
            LEFT,
            occupied_root,
            blue_root,
            ray_root,
            set(root["full"]),
        )
        if not left_options:
            continue

        horiz_dirs = [d for d in root["port_order"] if _axis_of(d) == "H"]
        assert set(horiz_dirs) == {LEFT, RIGHT}
        suppress_dir = horiz_dirs[0]
        continue_dir = horiz_dirs[-1]
        options_by_dir = {LEFT: left_options, RIGHT: right_options}

        for continue_blob in options_by_dir[continue_dir]:
            inp_cont = inp_root.copy()
            out_cont = out_root.copy()
            occupied_cont = occupied_root.copy()
            blue_cont = blue_root.copy()
            ray_cont = ray_root.copy()

            _mark_ray(out_cont, ray_cont, continue_blob["path"])
            _place_blob(inp_cont, occupied_cont, blue_cont, continue_blob)
            _paint_blob_red(out_cont, continue_blob)
            for d in DIR_ORDER:
                if d in continue_blob["terminal_paths"]:
                    _mark_ray(out_cont, ray_cont, continue_blob["terminal_paths"][d])

            for suppress_blob in options_by_dir[suppress_dir]:
                if any(occupied_cont[r, c] or ray_cont[r, c] for r, c in suppress_blob["full"]):
                    continue
                suppress_terms = _terminal_paths(suppress_blob, occupied_cont, ray_cont, H, W, dirs=(UP, DOWN))
                if suppress_terms is None:
                    continue

                suppress_allowed = set(root["full"]) | set(suppress_blob["entry"])
                if not _path_is_clear(suppress_blob["path"], occupied_cont, ray_cont, suppress_allowed):
                    continue

                inp_final = inp_cont.copy()
                out_final = out_cont.copy()
                occupied_final = occupied_cont.copy()
                blue_final = blue_cont.copy()
                ray_final = ray_cont.copy()

                _mark_ray(out_final, ray_final, suppress_blob["path"])
                _place_blob(inp_final, occupied_final, blue_final, suppress_blob)
                _paint_blob_red(out_final, suppress_blob)

                inp[:, :] = inp_final
                out[:, :] = out_final
                occupied[:, :] = occupied_final
                blue_mask[:, :] = blue_final
                ray_mask[:, :] = ray_final

                suppressed = set()
                for d in (UP, DOWN):
                    suppressed.update(suppress_terms[d])
                ignored_root_cells = set()
                if UP in ignored_root:
                    ignored_root_cells.update(ignored_root[UP])

                return {
                    "root_port_count": len(root["spec"]["port_dirs"]),
                    "suppressed_cells": len(suppressed),
                    "ignored_root_cells": len(ignored_root_cells),
                    "activated_blobs": 3,
                    "continue_dir": continue_dir,
                }

    return None


def _gen_attempt(with_meta: bool = False) -> np.ndarray | tuple[np.ndarray, dict] | None:
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    inp, out, occupied, blue_mask, ray_mask, cap_r, start_col = _place_start(H, W)
    meta = _build_scene(H, W, inp, out, occupied, blue_mask, ray_mask, cap_r, start_col)
    if meta is None:
        return None

    for _ in range(random.randint(0, MAX_DISTRACTORS)):
        if not _place_distractor(inp, occupied, blue_mask, ray_mask):
            return None

    if meta["suppressed_cells"] <= 0:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return (inp, meta) if with_meta else inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _gen_attempt(with_meta=False)
        if built is not None:
            return built
    raise RuntimeError("failed to generate a valid 5961cc34 combo input")
