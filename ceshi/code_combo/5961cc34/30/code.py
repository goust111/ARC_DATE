# Combo variant of 5961cc34 (index 30)
# combo_spec: D15/0+D20/0
# Name: 颜色角色推断+边缘图例
#
# Design analysis for the fused variant:
# - Concrete embodiment:
#   The original task used fixed colors to encode four roles directly in the arena.
#   Here the arena is moved into rows 2..H-1, while rows 0..1 become a top-edge
#   legend strip. The legend contains three tiny exemplars:
#   1) a vertical cap-over-ray marker, which identifies the start-cap color and
#      the emitted-ray / activated color;
#   2) a body->ray pair, which shows that blob bodies recolor into the ray color;
#   3) a body-port pair, which shows which color marks the emission edge.
#   Example trace: if the legend shows pink over grey, orange->grey, orange-next-to-green,
#   then the bottom start marker is pink over grey, the blobs are orange, their ports
#   are green, and every activated structure/ray becomes grey. The cascade geometry
#   is unchanged; only the role lookup stage is new.
# - Complexity calibration:
#   The legend changes only color semantics, not geometry. This keeps the original
#   reasoning core intact while making the new behavior visible in every sample.
#   The legend uses repeated body and ray colors across two icons, which prevents
#   underconstrained "one-off clue" ambiguity.
# - Hidden assumptions audited:
#   Original main() assumed: red start segment is the only thin bottom-touching
#   column; yellow sits directly above it; blue is the dominant non-start color;
#   green is the unique body-adjacent auxiliary color; the whole grid is the arena.
#   After fusion these assumptions break, so the new solver:
#   1) reads colors from fixed legend positions instead of pixel-count heuristics;
#   2) ignores the top legend strip when extracting blobs and tracing rays;
#   3) preserves the legend unchanged in the output so it functions as instructions,
#      not as part of the cascade state.
# - Placement legality:
#   The generator first builds the entire cascade inside a clean playfield with the
#   original non-overlap logic. Only after that does it add the legend in reserved
#   rows above the playfield, so legend pixels can never collide with blobs or rays.
# - Visual observability / tidiness:
#   The legend is flush with the top edge and separated from the arena by a full-row
#   boundary, so humans can visually isolate it as metadata. The arena itself stays
#   uncluttered because no legend symbol is mixed into the blob field.
# - Inferability of the change:
#   The generator keeps every semantic role different from its original fixed
#   palette id, and varies those ids across samples. That makes the legend necessary:
#   a fixed-role color interpretation cannot explain the generated examples.

from __future__ import annotations

import random
from arc_utils import components_mask, get_bg, paint_coords, paste
import numpy as np

# --- Legend layout ---
LEGEND_H = 2
LEGEND_W = 11
CAP_ICON = (0, 1)
RAY_ICON = (1, 1)
ACT_BODY_ICON = (0, 4)
ACT_RAY_ICON = (0, 5)
PORT_BODY_ICON = (0, 8)
PORT_ICON = (0, 9)
LEGEND_CELLS = {CAP_ICON, RAY_ICON, ACT_BODY_ICON, ACT_RAY_ICON, PORT_BODY_ICON, PORT_ICON}

# --- Fixed palette from the original task; used only to avoid degenerate regressions ---
ORIG_BG = 8
ORIG_BODY = 1
ORIG_RAY = 2
ORIG_PORT = 3
ORIG_CAP = 4

# --- Dynamic color pool ---
COLOR_POOL = tuple(range(1, 10))

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

# --- Canvas bounds for the playfield below the legend ---
PLAY_H_MIN, PLAY_H_MAX = 13, 27
W_MIN, W_MAX = 15, 27
START_COL_PAD = 2

# --- Start segment ---
START_H_MIN, START_H_MAX = 2, 3

# --- Chain / distractor ---
CHAIN_LEN_CHOICES = (1, 2, 3)
CHAIN_LEN_WEIGHTS = (3, 3, 3)
MAX_BLOB_COUNT = 4
MAX_DISTRACTORS = 1

# --- Placement ---
MIN_GAP_FIRST = 4
MIN_GAP_CHAIN = 3

# --- Retry ---
GEN_RETRIES = 350


def _read_legend(grid: np.ndarray, bg: int) -> dict[str, int]:
    """Decode role colors from the fixed top-edge legend strip."""
    H, W = map(int, grid.shape)
    assert H > LEGEND_H and W >= LEGEND_W

    cap_col = int(grid[CAP_ICON])
    ray_col = int(grid[RAY_ICON])
    body_col = int(grid[ACT_BODY_ICON])
    ray_col_2 = int(grid[ACT_RAY_ICON])
    body_col_2 = int(grid[PORT_BODY_ICON])
    port_col = int(grid[PORT_ICON])

    assert ray_col == ray_col_2
    assert body_col == body_col_2
    assert len({cap_col, ray_col, body_col, port_col}) == 4
    assert bg not in {cap_col, ray_col, body_col, port_col}

    for rr in range(LEGEND_H):
        for cc in range(LEGEND_W):
            if (rr, cc) in LEGEND_CELLS:
                continue
            assert int(grid[rr, cc]) == int(bg)

    return {"bg": int(bg), "cap": cap_col, "ray": ray_col, "body": body_col, "port": port_col}


def _paint_legend(grid: np.ndarray, roles: dict[str, int]) -> None:
    """Draw the fixed legend icons into the top strip of a full grid."""
    grid[:LEGEND_H, :] = int(roles["bg"])
    grid[CAP_ICON] = int(roles["cap"])
    grid[RAY_ICON] = int(roles["ray"])
    grid[ACT_BODY_ICON] = int(roles["body"])
    grid[ACT_RAY_ICON] = int(roles["ray"])
    grid[PORT_BODY_ICON] = int(roles["body"])
    grid[PORT_ICON] = int(roles["port"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build an activated ray network after decoding the color roles from a top-edge legend.

    Input structure:
    - Rows 0..1 form a small legend strip: cap-over-ray, body-to-ray, and body-next-to-port examples.
    - Rows 2..H-1 are the actual arena: body-colored blobs carry port-colored edge markers, and a bottom start marker uses the legend's cap/ray colors.

    Rule:
    - First read the legend to determine which colors mean body, port, start cap, and emitted/activated ray.
    - Then run the original cascade logic inside the arena only: shoot the thin start ray upward, activate the first body hit, and BFS through all port emissions until no new body is reached.

    Key insight: The geometry still determines the cascade, but the top strip now supplies the semantic key, so the same shapes can encode different roles under different palettes.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    roles = _read_legend(g, bg)

    play = np.array(g[LEGEND_H:, :], dtype=int)
    H, W = map(int, play.shape)
    out_play = np.full_like(play, int(bg))

    start_candidates: list[tuple[int, int, list[tuple[int, int]]]] = []
    for coords in components_mask(play == int(roles["ray"]), connectivity=8):
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        if c0 != c1 or (r1 - r0 + 1) < 2 or r1 != int(H - 1):
            continue
        if any(int(play[rr, c0]) != int(roles["ray"]) for rr in range(int(r0), int(r1) + 1)):
            continue
        if int(r0) - 1 < 0 or int(play[int(r0) - 1, int(c0)]) != int(roles["cap"]):
            continue
        start_candidates.append((int(c0), [(int(rr), int(c0)) for rr in range(int(r0), int(r1) + 1)]))

    assert len(start_candidates) == 1
    start_col, start_seg = start_candidates[0]

    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict[str, object]] = []
    for coords in components_mask(play == int(roles["body"]), connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = int(len(blobs))
        for r, c in pts:
            blob_id[int(r), int(c)] = bi
        blobs.append({"pts": pts, "ports": []})

    for blob in blobs:
        pts = blob["pts"]
        seen_dirs: set[tuple[int, int]] = set()
        for r, c in pts:
            for ddr, ddc in DIRS:
                rr = int(r + ddr)
                cc = int(c + ddc)
                if 0 <= rr < H and 0 <= cc < W and int(play[rr, cc]) == int(roles["port"]):
                    seen_dirs.add((int(ddr), int(ddc)))
        for ddr, ddc in seen_dirs:
            port_pixels = [
                (int(r), int(c))
                for (r, c) in pts
                if 0 <= int(r + ddr) < H and 0 <= int(c + ddc) < W and int(play[int(r + ddr), int(c + ddc)]) == int(roles["port"])
            ]
            assert port_pixels
            blob["ports"].append({"dir": (int(ddr), int(ddc)), "pixels": port_pixels})

    activated: set[int] = set()
    queue: list[int] = []

    def activate_blob_at(r: int, c: int) -> None:
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in activated:
            return
        activated.add(int(bi))
        queue.append(int(bi))
        for pr, pc in blobs[bi]["pts"]:
            out_play[int(pr), int(pc)] = int(roles["ray"])

    for r, c in start_seg:
        out_play[int(r), int(c)] = int(roles["ray"])
    top_r = min(int(r) for r, _c in start_seg)
    assert int(top_r) > 0 and int(play[int(top_r) - 1, int(start_col)]) == int(roles["cap"])
    out_play[int(top_r) - 1, int(start_col)] = int(roles["ray"])

    for r in range(int(top_r) - 1, -1, -1):
        out_play[int(r), int(start_col)] = int(roles["ray"])
        if int(play[int(r), int(start_col)]) == int(roles["body"]):
            activate_blob_at(int(r), int(start_col))
            break

    def extend_thick(port_pixels: list[tuple[int, int]], d: tuple[int, int]) -> None:
        ddr, ddc = map(int, d)
        for pr, pc in port_pixels:
            r, c = int(pr), int(pc)
            while True:
                r += int(ddr)
                c += int(ddc)
                if not (0 <= r < H and 0 <= c < W):
                    break
                out_play[int(r), int(c)] = int(roles["ray"])
                if int(play[int(r), int(c)]) == int(roles["body"]):
                    activate_blob_at(int(r), int(c))
                    break

    while queue:
        bi = int(queue.pop(0))
        for port in blobs[bi]["ports"]:
            extend_thick(port["pixels"], port["dir"])

    assert len(blobs) - len(activated) <= 1, "at most 1 distractor blob"

    out = np.full_like(g, int(bg))
    out[LEGEND_H:, :] = out_play
    _paint_legend(out, roles)
    return out


def _shape_mask(h: int, w: int) -> np.ndarray:
    """Build a boolean diamond/octagon mask for the given (h, w) shape."""
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


def _blob_spec(size: tuple[int, int], port_dir: tuple[int, int], roles: dict[str, int]) -> dict[str, object] | None:
    """Build one blob template in the sampled body/port colors."""
    h, w = size
    mask = _shape_mask(h, w)
    green = _edge_offsets(mask, port_dir)
    if len(green) not in (1, 2):
        return None

    green_set = set(green)
    patch = np.full((h, w), int(roles["bg"]), dtype=int)
    patch[mask] = int(roles["body"])
    for r, c in green:
        patch[r, c] = int(roles["port"])

    port_pixels: list[tuple[int, int]] = []
    for gr, gc in green:
        br, bc = gr - port_dir[0], gc - port_dir[1]
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
        "port_pixels": _sort_on_ray(port_pixels, port_dir),
    }


def _all_specs(roles: dict[str, int], port_dir: tuple[int, int] | None = None) -> list[dict[str, object]]:
    dirs = [port_dir] if port_dir is not None else list(DIRS)
    out = [s for size in SHAPES for d in dirs if (s := _blob_spec(size, d, roles)) is not None]
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
            r, c = r + dr, c + dc
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
            r, c = r + dr, c + dc
            if not (0 <= r < H and 0 <= c < W):
                break
            cells.append((r, c))
    return cells


def _mark_ray(out: np.ndarray, ray_mask: np.ndarray, cells: list[tuple[int, int]], ray_col: int) -> None:
    paint_coords(out, cells, int(ray_col))
    paint_coords(ray_mask, cells, True)


def _try_place_along_axis(
    spec: dict[str, object],
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
) -> list[dict[str, object]]:
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
    candidates: list[dict[str, object]] = []

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
    roles: dict[str, int],
) -> dict[str, object] | None:
    starts = _sort_on_ray(starts, d)
    thickness = len(starts)
    lane_vals = [_perp_value(p, d) for p in starts]
    lane_delta = [v - lane_vals[0] for v in lane_vals]
    entry_side = OPP[d]
    occ_pad = _dilate8(occupied)
    min_gap = MIN_GAP_FIRST if include_origin else MIN_GAP_CHAIN
    candidates: list[dict[str, object]] = []

    for spec in _all_specs(roles):
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


def _place_blob(inp: np.ndarray, occupied: np.ndarray, blue_mask: np.ndarray, blob: dict[str, object], bg: int) -> None:
    paste(inp, blob["spec"]["patch"], blob["r0"], blob["c0"], transparent=bg)
    for r, c in blob["full"]:
        occupied[r, c] = True
    for r, c in blob["blue"]:
        blue_mask[r, c] = True


def _place_distractor(
    inp: np.ndarray,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
    roles: dict[str, int],
) -> bool:
    H, W = inp.shape
    occ_pad = _dilate8(occupied)
    candidates: list[dict[str, object]] = []
    for spec in _all_specs(roles):
        h, w = spec["mask"].shape
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                full = _offset(spec["full"], r0, c0)
                if any(occ_pad[r, c] or ray_mask[r, c] for r, c in full):
                    continue
                candidates.append({"spec": spec, "r0": r0, "c0": c0, "full": full, "blue": _offset(spec["blue"], r0, c0)})
    if not candidates:
        return False
    _place_blob(inp, occupied, blue_mask, random.choice(candidates), int(roles["bg"]))
    return True


def _sample_roles() -> dict[str, int]:
    """Sample five distinct role colors, all different from the original fixed ids."""
    while True:
        bg, body, ray, port, cap = random.sample(COLOR_POOL, 5)
        if bg == ORIG_BG or body == ORIG_BODY or ray == ORIG_RAY or port == ORIG_PORT or cap == ORIG_CAP:
            continue
        return {"bg": int(bg), "body": int(body), "ray": int(ray), "port": int(port), "cap": int(cap)}


def _embed_full_grid(play: np.ndarray, roles: dict[str, int]) -> np.ndarray:
    H, W = map(int, play.shape)
    full = np.full((H + LEGEND_H, W), int(roles["bg"]), dtype=int)
    full[LEGEND_H:, :] = play
    _paint_legend(full, roles)
    return full


def _gen_attempt() -> np.ndarray | None:
    roles = _sample_roles()
    H = random.randint(PLAY_H_MIN, PLAY_H_MAX)
    W = random.randint(max(W_MIN, LEGEND_W), W_MAX)
    inp = np.full((H, W), int(roles["bg"]), dtype=int)
    out = np.full((H, W), int(roles["bg"]), dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    start_h = random.randint(START_H_MIN, START_H_MAX)
    cap_r = H - start_h - 1
    start_col = random.randint(START_COL_PAD, W - START_COL_PAD - 1)
    inp[cap_r, start_col] = int(roles["cap"])
    out[cap_r, start_col] = int(roles["ray"])
    ray_mask[cap_r, start_col] = True
    occupied[cap_r, start_col] = True
    for r in range(cap_r + 1, H):
        inp[r, start_col] = int(roles["ray"])
        out[r, start_col] = int(roles["ray"])
        occupied[r, start_col] = True

    chain_len = random.choices(CHAIN_LEN_CHOICES, weights=CHAIN_LEN_WEIGHTS)[0]
    min_dist = 0
    max_dist = min(MAX_DISTRACTORS, MAX_BLOB_COUNT - chain_len)
    num_distractors = random.randint(min_dist, max_dist)

    starts = [(cap_r, start_col)]
    direction = UP
    include_origin = True
    for _ in range(chain_len):
        blob = _candidate_next_blob(H, W, starts, direction, include_origin, occupied, blue_mask, ray_mask, roles)
        if blob is None:
            return None
        _mark_ray(out, ray_mask, blob["path"], int(roles["ray"]))
        _place_blob(inp, occupied, blue_mask, blob, int(roles["bg"]))
        paint_coords(out, blob["blue"], int(roles["ray"]))
        starts = blob["ports"]
        direction = blob["spec"]["port_dir"]
        include_origin = False

    final_path = _path_to_boundary(starts, direction, include_origin, H, W)
    if any(blue_mask[r, c] for r, c in final_path):
        return None
    _mark_ray(out, ray_mask, final_path, int(roles["ray"]))

    for _ in range(num_distractors):
        if not _place_distractor(inp, occupied, blue_mask, ray_mask, roles):
            return None

    full_inp = _embed_full_grid(inp, roles)
    full_out = _embed_full_grid(out, roles)

    if not np.array_equal(main(full_inp), full_out):
        return None
    return full_inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5961cc34 combo input")
