# Combo variant of 5961cc34 (index 32)
# combo_spec: D7/2+D8/2+D14/2
# Name: 交替方向处理+深度颜色+轨迹变色

from __future__ import annotations

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
# - The input grammar stays close to the source task: a bottom start marker
#   (yellow cap over a short red stem), plus blue diamond/octagon blobs with one
#   green edge-port each.
# - The output no longer has one uniform "active" color. Two interacting color
#   systems are visible:
#   1) Blob bodies are colored by activation depth:
#      depth 1 -> red, depth 2 -> orange, depth 3 -> yellow.
#   2) Ray segments are colored by the global processing step on which they are
#      emitted. Odd steps process horizontal rays; even steps process vertical
#      rays. The emitted segment color therefore encodes both time and axis:
#      step 1/H -> pink, step 2/V -> green, step 3/H -> maroon, step 4/V -> grey.
# - Trace example:
#   start hits root blob -> root becomes red.
#   If root port is horizontal, it emits on step 1, so that segment is pink.
#   The next blob becomes orange. Its vertical port emits on step 2, so that
#   segment is green. The third blob becomes yellow. Its horizontal port emits
#   on step 3, so the terminal segment is maroon.
#   If root port is vertical instead, step 1 has no horizontal ray, so it is
#   skipped; the first emitted segment appears on step 2 and is green.
#
# b) Complexity calibration
# - Blob depth colors use only three warm colors because the generator caps the
#   activated chain at three blobs. This preserves repeated evidence for the
#   depth mapping instead of introducing a long one-off palette.
# - Ray-step colors use only four post-start colors, enough to distinguish
#   horizontal vs vertical emissions and early vs late emissions without making
#   the output visually noisy.
# - The generator biases toward chain lengths 2-3 and mixes root horizontal /
#   root vertical cases so both depth colors and skipped-step behavior appear
#   frequently across a small example set.
#
# c) Implicit-assumption audit
# - The source solver assumed every activated blob immediately joins a BFS queue
#   and eventually emits in arrival order. That breaks here because a blob may
#   need to wait one step for the matching axis. The combo solver therefore uses
#   a "pending but not yet emitted" set keyed by blob id.
# - The source code painted a ray cell before checking whether it hit blue. That
#   would corrupt depth-colored blob entry pixels and is no longer valid. The hit
#   test must happen first; only non-blue traversed cells are painted directly.
# - The old "all active structure is red" assumption also breaks in two places:
#   blob painting must be depth-aware, and ray painting must be step-aware.
# - The fusion adds one more assumption not present before: every blob has a
#   single logical port side. The generator already constructs exactly one side,
#   and the solver asserts that after port extraction.
#
# d) Placement legality
# - The generator still uses output-first construction and the original padded
#   non-overlap checks. Every candidate blob must fit in bounds, keep one-cell
#   8-neighborhood separation from prior blobs, and be reached by a clear ray
#   lane before any unintended blue cell.
# - To keep ray-step colors within a compact palette, the generator constrains
#   the activated chain to alternate port axes after the root. This keeps the
#   actual emitted steps contiguous, except for the optional initial skip when
#   the root port is vertical.
#
# e) Visual observability
# - Humans can infer the depth palette because the first / second / third hit
#   blobs always occupy obvious positions along the same cascade chain.
# - Humans can infer the step-axis palette because horizontal emitted segments
#   repeat one family (pink then maroon) while vertical segments repeat another
#   (green then grey), and root-vertical examples visibly omit any step-1
#   horizontal segment before the first emitted ray appears.
# - No ordering uses hidden numeric color ids; all state changes are anchored to
#   visible traversal order and visible axis orientation.
#
# f) Visual cleanliness
# - Blob colors and ray colors are separated by geometry: blobs stay as compact
#   solid bodies, rays stay thin or 2-lane straight lines.
# - The generator keeps mostly linear cascades and places distractors off all
#   actual ray lanes, so the colored signal remains easy to parse.
#
# g) Change inferability
# - The blob-depth mapping has only three values with repeated structural support.
# - The ray-step mapping has only four emitted-step values, split by axis parity.
# - The generator deliberately produces both root-horizontal and root-vertical
#   cases, so the "skip empty step" part is not latent; it changes which ray
#   color appears first on the activated chain.

from collections import Counter
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste


# --- Colors ---
BG = 8
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
MAROON = 9

DEPTH_PALETTE = (RED, ORANGE, YELLOW)
STEP_RAY_COLORS = {
    0: RED,      # start ray below/above the cap before any step scheduling
    1: PINK,     # odd -> horizontal
    2: GREEN,    # even -> vertical
    3: MAROON,   # odd -> horizontal
    4: GREY,     # even -> vertical
}

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
START_COL_PAD = 2

# --- Start segment ---
START_H_MIN, START_H_MAX = 2, 3

# --- Chain / distractor ---
CHAIN_LEN_VALUES = (2, 3)
CHAIN_LEN_WEIGHTS = (4, 5)
ROOT_AXIS_WEIGHTS = {"H": 3, "V": 4}
MAX_BLOB_COUNT = 4
MAX_DISTRACTORS = 1

# --- Placement ---
MIN_GAP_FIRST = 4
MIN_GAP_CHAIN = 3

# --- Retry ---
GEN_RETRIES = 400


def _depth_color(depth: int) -> int:
    """Map activation depth 1..3 to the blob-body palette."""
    idx = min(max(int(depth), 1), len(DEPTH_PALETTE)) - 1
    return int(DEPTH_PALETTE[idx])


def _step_color(step_idx: int) -> int:
    """Map the global emitted-step index to the trajectory palette."""
    if int(step_idx) in STEP_RAY_COLORS:
        return int(STEP_RAY_COLORS[int(step_idx)])
    return int(STEP_RAY_COLORS[max(STEP_RAY_COLORS)])


def _axis_of(d: tuple[int, int]) -> str:
    return "V" if int(d[0]) else "H"


def _round_axis(step_idx: int) -> str:
    return "H" if int(step_idx) % 2 == 1 else "V"


def _dir_choices(axis: str) -> tuple[tuple[int, int], ...]:
    return (LEFT, RIGHT) if axis == "H" else (UP, DOWN)


def _other_axis(axis: str) -> str:
    return "V" if axis == "H" else "H"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild an activation network whose blob colors show depth and whose rays show scheduled step/axis.

    Input structure:
    - The background is the dominant color. Blue diamond/octagon blobs each have one green edge-port that fixes an outgoing axis-aligned ray and its thickness.
    - A unique bottom start marker is a short red vertical stem capped by a yellow pixel.

    Rule:
    - Fire the thin start ray upward and activate the first blue blob it reaches; activated blob bodies are recolored by depth (1=red, 2=orange, 3=yellow).
    - Afterwards advance through global steps: odd steps emit only horizontal pending rays, even steps emit only vertical pending rays. If a step has no pending ray of the required axis, skip it and continue.
    - Every emitted ray segment is colored by its step index (so color also reveals axis parity), and newly hit blobs join the pending set for later eligible steps. Unhit blobs and unused green ports disappear into the background.

    Key insight: The green port still determines direction and thickness, but emission is no longer immediate BFS; blobs wait for the matching horizontal/vertical time slot, while blob bodies and ray segments encode two different parts of the cascade state.
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
                (
                    int(col),
                    int(cap),
                    int(c0),
                    [(int(rr), int(c0)) for rr in range(r0, r1 + 1)],
                )
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
                rr = r + int(dr)
                cc = c + int(dc)
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(blue_col):
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

    for coords in components_mask(g == int(blue_col), connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = int(len(blobs))
        for r, c in pts:
            blob_id[int(r), int(c)] = int(bi)
        blobs.append({"pts": pts, "ports": []})

    for blob in blobs:
        pts = blob["pts"]
        seen_dirs: set[tuple[int, int]] = set()
        for r, c in pts:
            for ddr, ddc in DIRS:
                rr = r + int(ddr)
                cc = c + int(ddc)
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(green_col):
                    seen_dirs.add((int(ddr), int(ddc)))
        for ddr, ddc in seen_dirs:
            port_pixels = [
                (int(r), int(c))
                for (r, c) in pts
                if 0 <= int(r + ddr) < H
                and 0 <= int(c + ddc) < W
                and int(g[int(r + ddr), int(c + ddc)]) == int(green_col)
            ]
            assert port_pixels
            blob["ports"].append({"dir": (int(ddr), int(ddc)), "pixels": port_pixels})
        assert len(blob["ports"]) == 1
        blob["port"] = blob["ports"][0]

    activated: set[int] = set()
    pending: dict[int, int] = {}

    def activate_blob_at(r: int, c: int, depth: int, new_pending: dict[int, int]) -> None:
        """Color a newly hit blob by depth and register it for a later eligible step."""
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in activated:
            return
        activated.add(int(bi))
        new_pending[int(bi)] = int(depth)
        fill = _depth_color(int(depth))
        for pr, pc in blobs[bi]["pts"]:
            out[int(pr), int(pc)] = int(fill)

    def extend_thick(
        port_pixels: list[tuple[int, int]],
        d: tuple[int, int],
        emitter_depth: int,
        step_idx: int,
        new_pending: dict[int, int],
    ) -> None:
        """Emit one step-colored ray and activate the first blue blob reached on each lane."""
        dr, dc = map(int, d)
        ray_color = _step_color(int(step_idx))
        for pr, pc in port_pixels:
            r, c = int(pr), int(pc)
            while True:
                r += dr
                c += dc
                if not (0 <= r < H and 0 <= c < W):
                    break
                if int(g[int(r), int(c)]) == int(blue_col):
                    activate_blob_at(int(r), int(c), int(emitter_depth) + 1, new_pending)
                    break
                out[int(r), int(c)] = int(ray_color)

    for r, c in start_seg:
        out[int(r), int(c)] = int(red_col)
    top_r = min(int(r) for r, _c in start_seg)
    assert top_r > 0 and int(g[top_r - 1, int(start_col)]) == int(yellow_col)
    out[int(top_r) - 1, int(start_col)] = int(red_col)

    boot_pending: dict[int, int] = {}
    for r in range(int(top_r) - 1, -1, -1):
        if int(g[int(r), int(start_col)]) == int(blue_col):
            activate_blob_at(int(r), int(start_col), 1, boot_pending)
            break
        out[int(r), int(start_col)] = int(red_col)
    pending.update(boot_pending)

    step_idx = 1
    while pending:
        wanted_axis = _round_axis(int(step_idx))
        eligible = [
            (int(bi), int(depth), blobs[int(bi)]["port"])
            for bi, depth in pending.items()
            if _axis_of(blobs[int(bi)]["port"]["dir"]) == wanted_axis
        ]
        if not eligible:
            step_idx += 1
            continue

        new_pending: dict[int, int] = {}
        for bi, depth, port in eligible:
            extend_thick(port["pixels"], port["dir"], depth, step_idx, new_pending)
            pending.pop(int(bi), None)
        for bi, depth in new_pending.items():
            pending[int(bi)] = int(depth)
        step_idx += 1

    return out


def _shape_mask(h: int, w: int) -> np.ndarray:
    """Build the diamond/octagon footprint for a blob size."""
    widths = _SHAPE_WIDTHS[(h, w)]
    mask = np.zeros((h, w), dtype=bool)
    for r, ww in enumerate(widths):
        c0 = (w - ww) // 2
        mask[r, c0 : c0 + ww] = True
    return mask


def _perp_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    return int(rc[1] if d[0] else rc[0])


def _along_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    return int(rc[0] if d[0] else rc[1])


def _sort_on_ray(coords: list[tuple[int, int]], d: tuple[int, int]) -> list[tuple[int, int]]:
    return sorted(coords, key=lambda p: _perp_value(p, d))


def _edge_offsets(mask: np.ndarray, d: tuple[int, int]) -> list[tuple[int, int]]:
    """Return mask-local cells on the edge facing direction d."""
    pts = np.argwhere(mask)
    axis = 0 if d in (UP, DOWN) else 1
    extreme = pts[:, axis].min() if d in (UP, LEFT) else pts[:, axis].max()
    keep = pts[:, axis] == extreme
    return _sort_on_ray([(int(r), int(c)) for r, c in pts[keep]], d)


def _blob_spec(size: tuple[int, int], port_dir: tuple[int, int]) -> dict | None:
    """Build one legal blob template for a shape and a port direction."""
    h, w = size
    mask = _shape_mask(h, w)
    green = _edge_offsets(mask, port_dir)
    if len(green) not in (1, 2):
        return None

    green_set = set(green)
    patch = np.full((h, w), BG, dtype=int)
    patch[mask] = BLUE
    for r, c in green:
        patch[int(r), int(c)] = GREEN

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
        "port_pixels": _sort_on_ray(port_pixels, port_dir),
    }


def _all_specs(port_dirs: tuple[tuple[int, int], ...] | None = None) -> list[dict]:
    dirs = list(port_dirs) if port_dirs is not None else list(DIRS)
    out = [spec for size in SHAPES for d in dirs if (spec := _blob_spec(size, d)) is not None]
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
    return [(int(r0 + r), int(c0 + c)) for r, c in offsets]


def _in_bounds(coords: list[tuple[int, int]], H: int, W: int) -> bool:
    return all(0 <= int(r) < H and 0 <= int(c) < W for r, c in coords)


def _path_to_entries(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    entries: list[tuple[int, int]],
    include_origin: bool,
    H: int,
    W: int,
) -> list[tuple[int, int]] | None:
    """Trace all lanes from starts to the aligned entry cells."""
    dr, dc = map(int, d)
    cells: list[tuple[int, int]] = []
    for (r, c), (er, ec) in zip(starts, entries):
        r = int(r)
        c = int(c)
        er = int(er)
        ec = int(ec)
        if include_origin:
            cells.append((int(r), int(c)))
        guard = H + W + 10
        while (r, c) != (er, ec):
            r += dr
            c += dc
            guard -= 1
            if guard < 0 or not (0 <= r < H and 0 <= c < W):
                return None
            cells.append((int(r), int(c)))
    return cells


def _path_to_boundary(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    """Trace all lanes from starts until they leave the grid."""
    dr, dc = map(int, d)
    cells: list[tuple[int, int]] = []
    for r, c in starts:
        r = int(r)
        c = int(c)
        if include_origin:
            cells.append((int(r), int(c)))
        while True:
            r += dr
            c += dc
            if not (0 <= r < H and 0 <= c < W):
                break
            cells.append((int(r), int(c)))
    return cells


def _mark_ray(out: np.ndarray, ray_mask: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    paint_coords(out, cells, int(color))
    paint_coords(ray_mask, cells, True)


def _paint_blob_depth(out: np.ndarray, blob: dict, depth: int) -> None:
    paint_coords(out, blob["blue"], _depth_color(int(depth)))


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
    """Enumerate placements whose entry edge aligns with the incoming lanes."""
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
        r0 = int(origin_along if is_vertical else fixed)
        c0 = int(fixed if is_vertical else origin_along)
        if is_vertical:
            entry_cells = [(int(entry_line), int(c0 + c)) for _, c in window]
        else:
            entry_cells = [(int(r0 + r), int(entry_line)) for r, _ in window]
        full = _offset(spec["full"], r0, c0)
        if not _in_bounds(full, H, W):
            continue
        if any(bool(occ_pad[r, c]) or bool(ray_mask[r, c]) for r, c in full):
            continue
        path = _path_to_entries(starts, d, entry_cells, include_origin, H, W)
        if path is None:
            continue
        entry_set = set(entry_cells)
        if any(bool(blue_mask[r, c]) for r, c in path if (r, c) not in entry_set):
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
                "ports": _sort_on_ray(_offset(spec["port_pixels"], r0, c0), spec["port_dir"]),
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
    port_dirs: tuple[tuple[int, int], ...] | None = None,
) -> dict | None:
    """Choose one legal next blob that the current ray lanes can reach."""
    starts = _sort_on_ray(starts, d)
    thickness = len(starts)
    lane_vals = [_perp_value(p, d) for p in starts]
    lane_delta = [v - lane_vals[0] for v in lane_vals]
    entry_side = OPP[d]
    occ_pad = _dilate8(occupied)
    min_gap = MIN_GAP_FIRST if include_origin else MIN_GAP_CHAIN
    candidates: list[dict] = []

    for spec in _all_specs(port_dirs=port_dirs):
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

    return random.choice(candidates) if candidates else None


def _place_blob(inp: np.ndarray, occupied: np.ndarray, blue_mask: np.ndarray, blob: dict) -> None:
    paste(inp, blob["spec"]["patch"], blob["r0"], blob["c0"], transparent=BG)
    for r, c in blob["full"]:
        occupied[int(r), int(c)] = True
    for r, c in blob["blue"]:
        blue_mask[int(r), int(c)] = True


def _place_distractor(
    inp: np.ndarray,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
) -> bool:
    H, W = inp.shape
    occ_pad = _dilate8(occupied)
    candidates: list[dict] = []
    for spec in _all_specs():
        h, w = spec["mask"].shape
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                full = _offset(spec["full"], r0, c0)
                if any(bool(occ_pad[r, c]) or bool(ray_mask[r, c]) for r, c in full):
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


def _place_start(H: int, W: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    start_h = random.randint(START_H_MIN, START_H_MAX)
    cap_r = H - start_h - 1
    start_col = random.randint(START_COL_PAD, W - START_COL_PAD - 1)
    inp[int(cap_r), int(start_col)] = YELLOW
    out[int(cap_r), int(start_col)] = RED
    ray_mask[int(cap_r), int(start_col)] = True
    occupied[int(cap_r), int(start_col)] = True
    for r in range(cap_r + 1, H):
        inp[int(r), int(start_col)] = RED
        out[int(r), int(start_col)] = RED
        occupied[int(r), int(start_col)] = True

    return inp, out, occupied, blue_mask, ray_mask, int(cap_r), int(start_col)


def _gen_attempt(with_meta: bool = False) -> np.ndarray | tuple[np.ndarray, dict] | None:
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    inp, out, occupied, blue_mask, ray_mask, cap_r, start_col = _place_start(H, W)

    chain_len = random.choices(CHAIN_LEN_VALUES, weights=CHAIN_LEN_WEIGHTS)[0]
    root_axis = random.choices(
        ["H", "V"],
        weights=[ROOT_AXIS_WEIGHTS["H"], ROOT_AXIS_WEIGHTS["V"]],
    )[0]
    axis_seq = [root_axis]
    for _ in range(1, chain_len):
        axis_seq.append(_other_axis(axis_seq[-1]))

    max_dist = min(MAX_DISTRACTORS, MAX_BLOB_COUNT - chain_len)
    num_distractors = random.randint(0, max_dist)

    root = _candidate_next_blob(
        H,
        W,
        [(cap_r, start_col)],
        UP,
        True,
        occupied,
        blue_mask,
        ray_mask,
        port_dirs=_dir_choices(axis_seq[0]),
    )
    if root is None:
        return None

    _mark_ray(out, ray_mask, root["path"], RED)
    _place_blob(inp, occupied, blue_mask, root)
    _paint_blob_depth(out, root, depth=1)

    starts = root["ports"]
    direction = root["spec"]["port_dir"]
    step_idx = 1
    emission_steps: list[int] = []

    for depth in range(2, chain_len + 1):
        while _round_axis(step_idx) != _axis_of(direction):
            step_idx += 1
        nxt = _candidate_next_blob(
            H,
            W,
            starts,
            direction,
            False,
            occupied,
            blue_mask,
            ray_mask,
            port_dirs=_dir_choices(axis_seq[depth - 1]),
        )
        if nxt is None:
            return None
        _mark_ray(out, ray_mask, nxt["path"], _step_color(step_idx))
        emission_steps.append(int(step_idx))
        _place_blob(inp, occupied, blue_mask, nxt)
        _paint_blob_depth(out, nxt, depth=depth)
        starts = nxt["ports"]
        direction = nxt["spec"]["port_dir"]
        step_idx += 1

    while _round_axis(step_idx) != _axis_of(direction):
        step_idx += 1
    final_path = _path_to_boundary(starts, direction, False, H, W)
    if any(bool(blue_mask[r, c]) for r, c in final_path):
        return None
    _mark_ray(out, ray_mask, final_path, _step_color(step_idx))
    emission_steps.append(int(step_idx))

    for _ in range(num_distractors):
        if not _place_distractor(inp, occupied, blue_mask, ray_mask):
            return None

    if not np.array_equal(main(inp), out):
        return None

    meta = {
        "chain_len": int(chain_len),
        "root_axis": root_axis,
        "root_emit_step": int(emission_steps[0]),
        "emission_steps": tuple(int(s) for s in emission_steps),
        "skip_before_root_emit": int(emission_steps[0] > 1),
        "has_depth3": int(chain_len >= 3),
        "num_distractors": int(num_distractors),
    }
    return (inp, meta) if with_meta else inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _gen_attempt(with_meta=False)
        if built is not None:
            return built
    raise RuntimeError("failed to generate a valid 5961cc34 combo input")
