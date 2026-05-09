# Combo variant of 5961cc34 (index 27)
# combo_spec: D2/1+D4/0+D8/3
# Name: 边缘控制像素+单步无级联+无轨迹

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import components_mask, get_bg, paint_coords, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - Original task: a bottom stem launched an upward ray; the first hit blob
#      turned red, then green ports emitted visible rays breadth-first until the
#      cascade stopped.
#    - This combo changes three linked steps at once:
#      1) the source is now one red border control pixel,
#      2) only the first blob directly hit by that inward start ray may activate,
#      3) no ray cells are colored in the output.
#    - Example trace: a left-edge control pixel fires rightward and would first
#      cross blob A, whose green port points toward blob B. In the original task
#      A and B would both activate and a thick red trajectory would be drawn. In
#      this combo only A is recolored red; B stays absent in the output and the
#      whole path remains background. The red border pixel stays as the lone
#      source cue.
# b) Complexity calibration
#    - If every sample had only one blob, the D2/D8 changes would be hard to see.
#      To keep the variant legible, the generator always creates a latent chain
#      of length 2-3, so every input contains a plausible downstream blob that is
#      not activated in the output.
#    - The control edge varies over all four borders, making the D4 change
#      directly observable instead of collapsing back to the original bottom-only
#      source layout.
# c) Implicit-assumption audit against the original main()
#    - Bottom stem parsing is invalid: there is no yellow cap or vertical red
#      segment anymore, so start detection must become border scanning.
#    - The fixed initial direction UP is invalid: launch direction now depends on
#      which border holds the control pixel.
#    - Queue/BFS propagation is invalid: a hit blob no longer emits anything in
#      the solver, so port parsing is irrelevant to output construction.
#    - Painting rays is invalid: the output must not include either the start ray
#      or any thick port trajectory. Only the control pixel and the activated
#      blob body remain red.
#    - The old "at most one distractor remains inactive" assert is invalid
#      because the combo intentionally leaves all downstream latent blobs
#      inactive.
# d) Placement legality
#    - The generator keeps all blobs strictly inside the border so the unique
#      non-background border cell is always the control pixel.
#    - It still reserves the full latent ray network with a boolean mask, even
#      though those rays are invisible in the target output. This guarantees that
#      implied start/port lines are clear and do not overlap blobs or distractors.
#    - Standard in-bounds and one-cell separation checks are preserved.
# e) Visual observability
#    - Across a few examples a human can see: the source moved to varying edges,
#      only one blob ever turns red, and aligned downstream blobs suggested by
#      green ports do not activate. That jointly signals D4 + D2 + D8.
# f) Visual cleanliness
#    - Output is intentionally sparse: one border control pixel plus one red blob.
#      Because no trajectories are drawn, the generator offsets complexity into
#      the input arrangement rather than cluttering the target grid.
# g) Change inferability
#    - No new color semantics are introduced. Blue still denotes blob bodies,
#      green still marks a port side in the input, and red still marks the source
#      plus the activated result. The missing red trajectories are therefore a
#      structural clue, not a hidden color remapping.
# h) Quantifiable variant signal
#    - Useful checks are: control-edge distribution across the four borders, the
#      balance between latent chain lengths 2 and 3, and the fraction whose
#      latent BFS would activate more than one blob under the old rule.
# i) New solver assumptions that the generator must guarantee
#    - Exactly one non-background border cell exists and it is the red control.
#    - Blue is still the dominant non-background/non-control color by area.
#    - At least two blue blobs are present and the first lies on the inward
#      control ray, so the output always contains exactly one activated blob
#      while still exposing a suppressed downstream continuation.


# --- Colors ---
BG = 8
BLUE = 1
RED = 2
GREEN = 3

# --- Directions ---
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)
OPP = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

EDGE_TO_DIR = {
    "top": DOWN,
    "bottom": UP,
    "left": RIGHT,
    "right": LEFT,
}

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
EDGE_PAD = 2

# --- Latent chain / distractor ---
CHAIN_LENGTHS = (2, 3)
CHAIN_LEN_WEIGHTS = [4, 4]
MAX_BLOB_COUNT = 4
MAX_DISTRACTORS = 1

# --- Placement ---
MIN_GAP_FIRST = 4
MIN_GAP_CHAIN = 3

# --- Retry ---
GEN_RETRIES = 400


def _control_direction(r: int, c: int, H: int, W: int) -> tuple[int, int]:
    if int(r) == 0:
        return DOWN
    if int(r) == int(H - 1):
        return UP
    if int(c) == 0:
        return RIGHT
    if int(c) == int(W - 1):
        return LEFT
    raise AssertionError("control pixel must lie on the border")


def _touches_border(coords: list[tuple[int, int]], H: int, W: int) -> bool:
    return any(int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1)) for r, c in coords)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor only the first blob hit by a border-launched invisible ray.

    Input structure:
    - The dominant color is the background. Blue diamond/octagon blobs sit away from the border, each with one green edge bar marking a latent port direction.
    - Exactly one red control pixel lies on the border; its border side defines the inward launch direction.

    Rule:
    - Start from the red control pixel and scan straight inward until the first blue blob is reached.
    - Recolor that single hit blob red and stop immediately; do not emit any follow-up port ray and do not color any traversed path cells.

    Key insight: Green ports are only a misleading continuation cue here; the true output is determined solely by which blob the border control would hit first along its invisible inward line.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    border_pts: list[tuple[int, int]] = []
    for c in range(W):
        if int(g[0, c]) != int(bg):
            border_pts.append((0, int(c)))
        if int(H - 1) != 0 and int(g[H - 1, c]) != int(bg):
            border_pts.append((int(H - 1), int(c)))
    for r in range(1, H - 1):
        if int(g[r, 0]) != int(bg):
            border_pts.append((int(r), 0))
        if int(W - 1) != 0 and int(g[r, W - 1]) != int(bg):
            border_pts.append((int(r), int(W - 1)))

    assert len(border_pts) == 1
    start_r, start_c = border_pts[0]
    red_col = int(g[int(start_r), int(start_c)])
    start_dir = _control_direction(int(start_r), int(start_c), H, W)

    counts = Counter(map(int, g.flatten()))
    other_cols = [(int(cnt), int(col)) for col, cnt in counts.items() if int(col) not in (bg, red_col)]
    assert other_cols
    max_count = max(cnt for cnt, _col in other_cols)
    blue_candidates = [int(col) for cnt, col in other_cols if cnt == max_count]
    assert len(blue_candidates) == 1
    blue_col = int(blue_candidates[0])

    out = np.full_like(g, int(bg))
    out[int(start_r), int(start_c)] = int(red_col)

    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[list[tuple[int, int]]] = []
    for coords in components_mask(g == int(blue_col), connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = int(len(blobs))
        for r, c in pts:
            blob_id[int(r), int(c)] = bi
        blobs.append(pts)

    dr, dc = start_dir
    r, c = int(start_r), int(start_c)
    while 0 <= r < H and 0 <= c < W:
        if int(g[int(r), int(c)]) == int(blue_col):
            bi = int(blob_id[int(r), int(c)])
            assert bi >= 0
            for pr, pc in blobs[bi]:
                out[int(pr), int(pc)] = int(red_col)
            break
        r += int(dr)
        c += int(dc)

    return out


def _shape_mask(h: int, w: int) -> np.ndarray:
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
        patch[int(r), int(c)] = GREEN

    port_pixels: list[tuple[int, int]] = []
    for gr, gc in green:
        br = int(gr - port_dir[0])
        bc = int(gc - port_dir[1])
        if not (0 <= br < h and 0 <= bc < w):
            return None
        if not mask[int(br), int(bc)] or (int(br), int(bc)) in green_set:
            return None
        port_pixels.append((int(br), int(bc)))

    blue = [(int(r), int(c)) for r, c in np.argwhere(mask) if (int(r), int(c)) not in green_set]
    full = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    return {
        "size": (int(h), int(w)),
        "mask": mask,
        "patch": patch,
        "blue": blue,
        "full": full,
        "port_dir": port_dir,
        "port_pixels": _sort_on_ray(port_pixels, port_dir),
    }


def _all_specs(port_dir: tuple[int, int] | None = None) -> list[dict]:
    dirs = [port_dir] if port_dir is not None else list(DIRS)
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
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for (r, c), (er, ec) in zip(starts, entries):
        if include_origin:
            cells.append((int(r), int(c)))
        guard = H + W + 10
        while (int(r), int(c)) != (int(er), int(ec)):
            r = int(r + dr)
            c = int(c + dc)
            guard -= 1
            if guard < 0 or not (0 <= int(r) < H and 0 <= int(c) < W):
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
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for r, c in starts:
        if include_origin:
            cells.append((int(r), int(c)))
        while True:
            r = int(r + dr)
            c = int(c + dc)
            if not (0 <= int(r) < H and 0 <= int(c) < W):
                break
            cells.append((int(r), int(c)))
    return cells


def _mark_ray(ray_mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
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
    H: int,
    W: int,
) -> list[dict]:
    h, w = spec["mask"].shape
    win_vals = [_perp_value(p, d) for p in window]
    edge_line = _along_value(window[0], d)
    start_line = _along_value(starts[0], d)
    is_vertical = bool(d[0])

    fixed = int(lane_vals[0] - win_vals[0])
    extent = int(H - h if is_vertical else W - w)
    low = int(edge_line)
    high = int(extent + edge_line)
    if d in (UP, LEFT):
        high = min(high, int(start_line - min_gap))
    else:
        low = max(low, int(start_line + min_gap))

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
        full = _offset(spec["full"], r0, c0)
        if not _in_bounds(full, H, W):
            continue
        if _touches_border(full, H, W):
            continue
        if any(bool(occ_pad[r, c]) or bool(ray_mask[r, c]) for r, c in full):
            continue
        path = _path_to_entries(starts, d, entry_cells, include_origin, H, W)
        if path is None:
            continue
        entry_set = set(entry_cells)
        if any(bool(blue_mask[r, c]) for r, c in path if (int(r), int(c)) not in entry_set):
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


def _candidate_next_blob(
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
    lane_delta = [int(v - lane_vals[0]) for v in lane_vals]
    entry_side = OPP[d]
    occ_pad = _dilate8(occupied)
    min_gap = MIN_GAP_FIRST if include_origin else MIN_GAP_CHAIN
    candidates: list[dict] = []

    for spec in _all_specs():
        if spec["port_dir"] == entry_side:
            continue
        edge = _edge_offsets(spec["mask"], entry_side)
        if len(edge) < thickness:
            continue
        for k in range(len(edge) - thickness + 1):
            window = edge[k : k + thickness]
            win_delta = [int(_perp_value(p, d) - _perp_value(window[0], d)) for p in window]
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
                if _touches_border(full, H, W):
                    continue
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


def _sample_control(H: int, W: int) -> tuple[tuple[int, int], tuple[int, int], str]:
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge == "top":
        coord = (0, random.randint(EDGE_PAD, W - EDGE_PAD - 1))
    elif edge == "bottom":
        coord = (int(H - 1), random.randint(EDGE_PAD, W - EDGE_PAD - 1))
    elif edge == "left":
        coord = (random.randint(EDGE_PAD, H - EDGE_PAD - 1), 0)
    else:
        coord = (random.randint(EDGE_PAD, H - EDGE_PAD - 1), int(W - 1))
    return (int(coord[0]), int(coord[1])), EDGE_TO_DIR[edge], edge


def _build_variant_output(
    H: int,
    W: int,
    chain: list[dict],
    start_r: int,
    start_c: int,
) -> np.ndarray:
    out = np.full((H, W), BG, dtype=int)
    out[int(start_r), int(start_c)] = RED
    assert chain, "combo generator must always place a first-hit blob"
    paint_coords(out, chain[0]["blue"], RED)
    return out


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    inp = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    (start_r, start_c), direction, _edge = _sample_control(H, W)
    inp[int(start_r), int(start_c)] = RED
    occupied[int(start_r), int(start_c)] = True
    ray_mask[int(start_r), int(start_c)] = True

    chain_len = random.choices(CHAIN_LENGTHS, weights=CHAIN_LEN_WEIGHTS)[0]
    max_dist = min(MAX_DISTRACTORS, MAX_BLOB_COUNT - chain_len)
    num_distractors = random.randint(0, max_dist)

    chain: list[dict] = []
    starts = [(int(start_r), int(start_c))]
    include_origin = True
    for _ in range(chain_len):
        blob = _candidate_next_blob(H, W, starts, direction, include_origin, occupied, blue_mask, ray_mask)
        if blob is None:
            return None
        chain.append(blob)
        _mark_ray(ray_mask, blob["path"])
        _place_blob(inp, occupied, blue_mask, blob)
        starts = blob["ports"]
        direction = blob["spec"]["port_dir"]
        include_origin = False

    final_path = _path_to_boundary(starts, direction, include_origin, H, W)
    if any(bool(blue_mask[r, c]) for r, c in final_path):
        return None
    _mark_ray(ray_mask, final_path)

    for _ in range(num_distractors):
        if not _place_distractor(inp, occupied, blue_mask, ray_mask):
            return None

    out = _build_variant_output(H, W, chain, start_r, start_c)
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5961cc34 combo input")
