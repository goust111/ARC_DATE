# Variant of 5961cc34
# Dimension: D16 — 端口方向约束 (meta)
# Change: 色块端口方向与射线进入方向不同（否则色块是终端节点），允许上/下/左/右四个方向... → 转向规则由图例决定

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste


# Design analysis for the D16 variant
# a) Concrete behavior:
#    The original task lets each activated blue blob emit through the one green-marked side it carries.
#    Here each reachable blob instead shows BOTH orthogonal exits for the direction from which it will be hit.
#    Example: the start ray travels upward, so the first chain blob shows green bars on its left and right edges.
#    A detached 2x2 orange elbow legend in a top corner decides whether every activation takes the clockwise or
#    counterclockwise 90-degree turn. The output keeps the legend unchanged, turns activated blobs red, and paints
#    only the legend-selected branch rays red. The behavioral difference sits exactly at the “which port emits?”
#    step: original logic emits from the blob’s unique green side, whereas the variant first reads the legend and
#    then chooses one of two candidate side-bars on every hit.
# b) Complexity calibration:
#    The legend has only two states (`cw`, `ccw`), which is enough to create a visible global rule without adding
#    too many branches. Every chain sample uses at least one blob so the legend is never dead. The chain length
#    remains small (1-3) so the task still reads like the seed task rather than a new puzzle family.
# c) Hidden assumptions audited from the original solver:
#    - Blue is identified by dominant pixel count among non-bg/non-start colors: still valid because the legend is tiny.
#    - Green is the unique non-start color adjacent to blue: still valid because the legend is detached from blobs.
#    - One blob == one 8-connected blue component: checked during design; replacing two orthogonal edge bars with green
#      still leaves the blue body connected for the shapes we keep.
#    - Original BFS enqueued one outgoing direction per blob. That no longer holds, so activation now depends on the
#      incoming ray direction plus the global legend rule.
#    - The old “emit every detected port” pattern would be wrong here, because each blob exposes two candidate turns.
# d) Placement legality:
#    The generator still works output-first, but specs are keyed by incoming direction. A chain blob is only placed if
#    its entry edge can accept the current ray thickness and if its two orthogonal green sides both have bar length 1-2.
#    The legend occupies a reserved corner mask; rays are forbidden from crossing those legend cells, and blob placement
#    keeps at least one-cell clearance from the legend via the existing occupied-mask dilation.
# e) Visual observability:
#    The detached orange elbow is the only new symbol, and its chirality is directly mirrored by the network turning
#    left or right at every activation. Because each active blob shows both candidate exits in the input, humans cannot
#    solve the puzzle by reading a blob locally; they must infer the global rule from the repeated turn direction.
# f) Visual neatness:
#    The legend is isolated in a corner, not embedded into the blob field. Unused blobs are still erased in the output,
#    so the result remains the original clean red-network picture plus one small legend glyph.
# g) Color-change inference:
#    Only one extra color is introduced (orange=7) and it is used solely for the legend. There is no hidden sorting by
#    color id and no ambiguous recoloring rule: orange stays orange, while activated blue/green structure becomes red.
# Reference-task note:
#    The local workspace includes `e87109e9.json` but not its solver source. I inspected only the provided I/O examples
#    and borrowed the minimal idea relevant here: a small detached legend whose shape globally controls turn behavior.


# --- Colors ---
BG = 8
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

# --- Directions ---
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)
OPP = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
CW_TURN = {UP: RIGHT, RIGHT: DOWN, DOWN: LEFT, LEFT: UP}
CCW_TURN = {UP: LEFT, LEFT: DOWN, DOWN: RIGHT, RIGHT: UP}
TURN_MAPS = {"cw": CW_TURN, "ccw": CCW_TURN}

# --- Legend ---
LEGEND_PATTERNS = {
    "cw": {(0, 0), (0, 1), (1, 1)},
    "ccw": {(0, 0), (0, 1), (1, 0)},
}
LEGEND_ROW = 1

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
START_COL_PAD = 2

# --- Start segment ---
START_H_MIN, START_H_MAX = 2, 3

# --- Chain / distractor ---
CHAIN_LEN_WEIGHTS = [0, 2, 3, 3]  # chain_len in [0,1,2,3]; zero disables legend-irrelevant cases
MAX_BLOB_COUNT = 4
MAX_DISTRACTORS = 1

# --- Placement ---
MIN_GAP_FIRST = 4
MIN_GAP_CHAIN = 3

# --- Retry ---
GEN_RETRIES = 500


def _turn(d: tuple[int, int], rule: str) -> tuple[int, int]:
    return TURN_MAPS[rule][d]


def _perp_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    return rc[1] if d[0] else rc[0]


def _along_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    return rc[0] if d[0] else rc[1]


def _sort_on_ray(coords: list[tuple[int, int]], d: tuple[int, int]) -> list[tuple[int, int]]:
    return sorted(coords, key=lambda p: _perp_value(p, d))


def _shape_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, ww in enumerate(_SHAPE_WIDTHS[(h, w)]):
        c0 = (w - ww) // 2
        mask[r, c0 : c0 + ww] = True
    return mask


def _edge_offsets(mask: np.ndarray, d: tuple[int, int]) -> list[tuple[int, int]]:
    pts = np.argwhere(mask)
    axis = 0 if d in (UP, DOWN) else 1
    extreme = pts[:, axis].min() if d in (UP, LEFT) else pts[:, axis].max()
    keep = pts[:, axis] == extreme
    return _sort_on_ray([(int(r), int(c)) for r, c in pts[keep]], d)


def _offset(offsets: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(r0 + r, c0 + c) for r, c in offsets]


def _in_bounds(coords: list[tuple[int, int]], H: int, W: int) -> bool:
    return all(0 <= r < H and 0 <= c < W for r, c in coords)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r0, r1 = max(0, dr), H + min(0, dr)
            c0, c1 = max(0, dc), W + min(0, dc)
            out[r0:r1, c0:c1] |= mask[r0 - dr : r1 - dr, c0 - dc : c1 - dc]
    return out


def _legend_origin(W: int, side: str) -> tuple[int, int]:
    if side == "left":
        return LEGEND_ROW, 1
    if side == "right":
        return LEGEND_ROW, W - 3
    raise ValueError(side)


def _legend_cells(rule: str, origin: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = origin
    return [(r0 + dr, c0 + dc) for dr, dc in sorted(LEGEND_PATTERNS[rule])]


def _detect_legend(g: np.ndarray, used_colors: set[int]) -> tuple[str, int, list[tuple[int, int]]]:
    candidates: list[tuple[str, int, list[tuple[int, int]]]] = []
    for col in colors_present(g, exclude=used_colors):
        for coords in components_mask(g == int(col), connectivity=8):
            pts = [(int(r), int(c)) for r, c in coords]
            if len(pts) != 3:
                continue
            r0 = min(r for r, _ in pts)
            c0 = min(c for _, c in pts)
            r1 = max(r for r, _ in pts)
            c1 = max(c for _, c in pts)
            if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
                continue
            rel = {(r - r0, c - c0) for r, c in pts}
            for rule, pattern in LEGEND_PATTERNS.items():
                if rel == pattern:
                    candidates.append((rule, int(col), pts))
                    break
    assert len(candidates) == 1
    return candidates[0]


def _solve_core(grid: np.ndarray, forced_rule: str | None = None, emit_all_ports: bool = False) -> np.ndarray:
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
            if cap == bg or cap == int(col):
                continue
            start_candidates.append(
                (int(col), cap, c0, [(rr, c0) for rr in range(r0, r1 + 1)])
            )

    assert len(start_candidates) == 1
    red_col, yellow_col, start_col, start_seg = start_candidates[0]

    counts = Counter(map(int, g.flatten()))
    other_cols = [(int(cnt), int(col)) for col, cnt in counts.items() if int(col) not in (bg, red_col, yellow_col)]
    assert other_cols
    max_count = max(cnt for cnt, _col in other_cols)
    blue_candidates = [int(col) for cnt, col in other_cols if cnt == max_count]
    assert len(blue_candidates) == 1
    blue_col = blue_candidates[0]

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
    green_col = green_candidates[0]

    legend_rule, legend_col, legend_pts = _detect_legend(
        g, used_colors={bg, red_col, yellow_col, blue_col, green_col}
    )
    if forced_rule is not None:
        legend_rule = forced_rule

    out = np.full_like(g, bg)
    for r, c in legend_pts:
        out[r, c] = legend_col

    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []
    for coords in components_mask(g == blue_col, connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = len(blobs)
        for r, c in pts:
            blob_id[r, c] = bi
        blobs.append({"pts": pts, "ports": {}})

    for blob in blobs:
        pts = blob["pts"]
        seen_dirs: set[tuple[int, int]] = set()
        for r, c in pts:
            for ddr, ddc in DIRS:
                rr = r + ddr
                cc = c + ddc
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == green_col:
                    seen_dirs.add((ddr, ddc))
        port_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for ddr, ddc in seen_dirs:
            port_pixels = [
                (int(r), int(c))
                for (r, c) in pts
                if 0 <= int(r + ddr) < H
                and 0 <= int(c + ddc) < W
                and int(g[int(r + ddr), int(c + ddc)]) == green_col
            ]
            assert port_pixels
            port_map[(ddr, ddc)] = _sort_on_ray(port_pixels, (ddr, ddc))
        assert len(port_map) == 2
        blob["ports"] = port_map

    activated: set[int] = set()
    queue: deque[tuple[int, tuple[int, int]]] = deque()

    def activate_blob_at(r: int, c: int, incoming_dir: tuple[int, int]) -> None:
        bi = int(blob_id[r, c])
        assert bi >= 0
        if bi in activated:
            return
        activated.add(bi)
        for pr, pc in blobs[bi]["pts"]:
            out[pr, pc] = red_col
        if emit_all_ports:
            for d in blobs[bi]["ports"]:
                queue.append((bi, d))
        else:
            out_dir = _turn(incoming_dir, legend_rule)
            assert out_dir in blobs[bi]["ports"]
            queue.append((bi, out_dir))

    for r, c in start_seg:
        out[r, c] = red_col
    top_r = min(r for r, _c in start_seg)
    assert top_r > 0 and int(g[top_r - 1, start_col]) == yellow_col
    out[top_r - 1, start_col] = red_col
    for r in range(top_r - 1, -1, -1):
        out[r, start_col] = red_col
        if int(g[r, start_col]) == blue_col:
            activate_blob_at(r, start_col, UP)
            break

    def extend_thick(port_pixels: list[tuple[int, int]], d: tuple[int, int]) -> None:
        ddr, ddc = d
        for pr, pc in port_pixels:
            r, c = pr, pc
            while True:
                r += ddr
                c += ddc
                if not (0 <= r < H and 0 <= c < W):
                    break
                out[r, c] = red_col
                if int(g[r, c]) == blue_col:
                    activate_blob_at(r, c, d)
                    break

    while queue:
        bi, d = queue.popleft()
        extend_thick(blobs[bi]["ports"][d], d)

    assert len(blobs) - len(activated) <= 1, "at most 1 distractor blob"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a red cascade by reading a detached turn legend before following blob exits.

    Input structure:
    - The background is dominant, the start marker is still a yellow cap above a short red column at the bottom.
    - Blue blobs now carry two green side-bars: the clockwise and counterclockwise 90-degree exits for the ray that will hit them.
    - A tiny detached orange elbow legend in a top corner tells whether all turns are clockwise or counterclockwise.

    Rule:
    - Launch the thin start ray upward from the bottom marker.
    - Whenever a ray reaches a blue blob, recolor that blob red and continue only through the green side obtained by applying the legend's turn rule to the incoming direction.
    - Propagate breadth-first until no new blobs are hit; keep the legend, erase inactive blobs, and color every used ray red.

    Key insight:
    - The green marks are no longer a complete local instruction: each activated blob exposes both possible 90-degree turns, so the solver must decode the detached legend first and apply that single global turning rule at every hit.
    """
    return _solve_core(grid)


def _blob_spec(size: tuple[int, int], incoming_dir: tuple[int, int]) -> dict | None:
    h, w = size
    mask = _shape_mask(h, w)
    turn_dirs = (_turn(incoming_dir, "cw"), _turn(incoming_dir, "ccw"))
    green_by_dir: dict[tuple[int, int], list[tuple[int, int]]] = {}
    green_set: set[tuple[int, int]] = set()

    for d in turn_dirs:
        green = _edge_offsets(mask, d)
        if len(green) not in (1, 2):
            return None
        for cell in green:
            if cell in green_set:
                return None
            green_set.add(cell)
        green_by_dir[d] = green

    lens = {len(v) for v in green_by_dir.values()}
    if len(lens) != 1:
        return None

    patch = np.full((h, w), BG, dtype=int)
    patch[mask] = BLUE
    for r, c in green_set:
        patch[r, c] = GREEN

    ports_by_dir: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for d, green in green_by_dir.items():
        port_pixels: list[tuple[int, int]] = []
        for gr, gc in green:
            br = gr - d[0]
            bc = gc - d[1]
            if not (0 <= br < h and 0 <= bc < w):
                return None
            if not mask[br, bc] or (br, bc) in green_set:
                return None
            port_pixels.append((br, bc))
        ports_by_dir[d] = _sort_on_ray(port_pixels, d)

    blue = [(int(r), int(c)) for r, c in np.argwhere(mask) if (int(r), int(c)) not in green_set]
    full = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    return {
        "size": (h, w),
        "mask": mask,
        "patch": patch,
        "blue": blue,
        "full": full,
        "incoming_dir": incoming_dir,
        "entry_side": OPP[incoming_dir],
        "ports_by_dir": ports_by_dir,
    }


def _all_specs(incoming_dir: tuple[int, int] | None = None) -> list[dict]:
    dirs = [incoming_dir] if incoming_dir is not None else list(DIRS)
    specs = [s for size in SHAPES for d in dirs if (s := _blob_spec(size, d)) is not None]
    random.shuffle(specs)
    return specs


def _path_to_entries(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    entries: list[tuple[int, int]],
    include_origin: bool,
    H: int,
    W: int,
    blocked_mask: np.ndarray,
) -> list[tuple[int, int]] | None:
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for (r, c), (er, ec) in zip(starts, entries):
        if include_origin:
            if blocked_mask[r, c]:
                return None
            cells.append((r, c))
        guard = H + W + 10
        while (r, c) != (er, ec):
            r += dr
            c += dc
            guard -= 1
            if guard < 0 or not (0 <= r < H and 0 <= c < W):
                return None
            if blocked_mask[r, c]:
                return None
            cells.append((r, c))
    return cells


def _path_to_boundary(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    H: int,
    W: int,
    blocked_mask: np.ndarray,
) -> list[tuple[int, int]] | None:
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for r, c in starts:
        if include_origin:
            if blocked_mask[r, c]:
                return None
            cells.append((r, c))
        while True:
            r += dr
            c += dc
            if not (0 <= r < H and 0 <= c < W):
                break
            if blocked_mask[r, c]:
                return None
            cells.append((r, c))
    return cells


def _mark_ray(out: np.ndarray, ray_mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    paint_coords(out, cells, RED)
    paint_coords(ray_mask, cells, True)


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
    blocked_mask: np.ndarray,
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
        if any(occ_pad[r, c] or ray_mask[r, c] or blocked_mask[r, c] for r, c in full):
            continue
        path = _path_to_entries(starts, d, entry_cells, include_origin, H, W, blocked_mask)
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
                    pd: _sort_on_ray(_offset(port_pixels, r0, c0), pd)
                    for pd, port_pixels in spec["ports_by_dir"].items()
                },
            }
        )

    return candidates


def _candidate_next_blob(
    H: int,
    W: int,
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
    blocked_mask: np.ndarray,
) -> dict | None:
    starts = _sort_on_ray(starts, d)
    thickness = len(starts)
    lane_vals = [_perp_value(p, d) for p in starts]
    lane_delta = [v - lane_vals[0] for v in lane_vals]
    entry_side = OPP[d]
    occ_pad = _dilate8(occupied)
    min_gap = MIN_GAP_FIRST if include_origin else MIN_GAP_CHAIN
    candidates: list[dict] = []

    for spec in _all_specs(d):
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
                    blocked_mask,
                    H,
                    W,
                )
            )

    return random.choice(candidates) if candidates else None


def _place_blob(inp: np.ndarray, occupied: np.ndarray, blue_mask: np.ndarray, blob: dict) -> None:
    paste(inp, blob["spec"]["patch"], blob["r0"], blob["c0"], transparent=BG)
    for r, c in blob["full"]:
        occupied[r, c] = True
    for r, c in blob["blue"]:
        blue_mask[r, c] = True


def _place_distractor(
    inp: np.ndarray,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
    blocked_mask: np.ndarray,
) -> bool:
    H, W = inp.shape
    occ_pad = _dilate8(occupied)
    candidates: list[dict] = []
    for spec in _all_specs():
        h, w = spec["mask"].shape
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                full = _offset(spec["full"], r0, c0)
                if any(occ_pad[r, c] or ray_mask[r, c] or blocked_mask[r, c] for r, c in full):
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


def _place_legend(
    inp: np.ndarray,
    out: np.ndarray,
    occupied: np.ndarray,
    blocked_mask: np.ndarray,
    rule: str,
    side: str,
) -> None:
    legend = _legend_cells(rule, _legend_origin(inp.shape[1], side))
    for r, c in legend:
        inp[r, c] = ORANGE
        out[r, c] = ORANGE
        occupied[r, c] = True
        blocked_mask[r, c] = True


def _variant_signal(inp: np.ndarray) -> int:
    expected = _solve_core(inp)
    all_ports = _solve_core(inp, emit_all_ports=True)
    return int(np.count_nonzero(expected != all_ports))


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)
    blocked_mask = np.zeros((H, W), dtype=bool)

    turn_rule = random.choice(["cw", "ccw"])
    legend_side = random.choice(["left", "right"])
    _place_legend(inp, out, occupied, blocked_mask, turn_rule, legend_side)

    start_h = random.randint(START_H_MIN, START_H_MAX)
    cap_r = H - start_h - 1
    start_col = random.randint(START_COL_PAD, W - START_COL_PAD - 1)
    inp[cap_r, start_col] = YELLOW
    out[cap_r, start_col] = RED
    ray_mask[cap_r, start_col] = True
    occupied[cap_r, start_col] = True
    for r in range(cap_r + 1, H):
        inp[r, start_col] = RED
        out[r, start_col] = RED
        occupied[r, start_col] = True

    chain_len = random.choices(range(len(CHAIN_LEN_WEIGHTS)), weights=CHAIN_LEN_WEIGHTS)[0]
    assert chain_len >= 1
    max_dist = min(MAX_DISTRACTORS, MAX_BLOB_COUNT - chain_len)
    num_distractors = random.randint(0, max_dist)

    starts = [(cap_r, start_col)]
    direction = UP
    include_origin = True
    for _ in range(chain_len):
        blob = _candidate_next_blob(
            H,
            W,
            starts,
            direction,
            include_origin,
            occupied,
            blue_mask,
            ray_mask,
            blocked_mask,
        )
        if blob is None:
            return None
        _mark_ray(out, ray_mask, blob["path"])
        _place_blob(inp, occupied, blue_mask, blob)
        paint_coords(out, blob["blue"], RED)
        direction = _turn(direction, turn_rule)
        starts = blob["ports_by_dir"][direction]
        include_origin = False

    final_path = _path_to_boundary(starts, direction, include_origin, H, W, blocked_mask)
    if final_path is None:
        return None
    if any(blue_mask[r, c] for r, c in final_path):
        return None
    _mark_ray(out, ray_mask, final_path)

    for _ in range(num_distractors):
        if not _place_distractor(inp, occupied, blue_mask, ray_mask, blocked_mask):
            return None

    if not np.array_equal(main(inp), out):
        return None
    if _variant_signal(inp) == 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5961cc34 D16 input")
