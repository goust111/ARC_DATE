# Combo variant of 5961cc34 (index 28)
# combo_spec: D4/3+D6/2
# Name: 起始段高度编码距离+距离编码停止

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste

# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concretization and behavior trace
# - The visible objects stay close to the base task: blue diamond/octagon blobs,
#   one intended green port edge per blob, and one bottom start marker made of a yellow
#   cap above a red vertical stem.
# - The fusion changes one cue into a dual code. The stem height h in {2, 3}
#   now means:
#   * every emitted ray travels for exactly h steps, and
#   * at most h blobs may activate in total.
# - Small trace:
#   * Base task: stem height 2 is decorative; start ray rises until collision,
#     hit blob A emits until collision, then blob B emits until collision.
#   * Combo task with h=2: the start ray paints exactly 2 cells above the cap.
#     If it touches blob A on step 1, step 2 is still painted. Blob A activates.
#     Blob A's port ray also paints exactly 2 cells. If that hits blob B, blob B
#     activates but does not emit, because the activation budget is exhausted.
#   * Difference points:
#     1. collision no longer determines ray length;
#     2. stem height no longer controls only depth, but also every ray's range;
#     3. the last allowed hit can still sit inside an incoming ray that keeps
#        painting its remaining fixed steps.
#
# b) Complexity calibration
# - We keep only two stem heights, 2 and 3. That is enough to support a human
#   hypothesis while keeping the rule legible.
# - All rays use the same global budget h instead of mixing start-based and
#   blob-size-based distances. This prevents the fusion from becoming a pile of
#   unrelated codes.
# - The generator oversamples one extra latent hop beyond the cap where the
#   geometry supports it reliably. In practice h=3 often carries a latent fourth
#   blob, while both heights still prefer early hits so fixed-distance overshoot
#   is visible.
#
# c) Implicit assumption audit against the original main()
# - Start detection still assumes exactly one bottom-touching 1-wide red segment
#   with a non-bg cap above it. The generator preserves that.
# - Blue is still inferred as the dominant non-start color, so every sample must
#   contain at least one blob. Chain-length-0 samples therefore force a distractor.
# - Green is still the unique non-start color 4-adjacent to blue; generator keeps
#   exactly one intended green side per blob, with an h=2 terminal exception
#   where the previous blob's exit green may also touch the incoming side.
# - The original "stop ray on first blue hit" assumption is invalid. Rays must
#   continue until their counted budget is spent.
# - The original "every activated blob emits" assumption is invalid. Once h blobs
#   are activated, newly hit blobs must stay inactive; the h-th blob turns red
#   but is not enqueued.
# - A new fusion-specific contract appears: after the cap is exhausted, no extra
#   blob may lie later on the same already-drawn fixed-distance corridor, because
#   the solver paints ray cells before deciding whether activation is allowed.
#   The generator enforces one relevant blob per corridor.
#
# d) Placement legality
# - The generator uses fixed-step tracing before placing a target blob, so every
#   active chain hop is guaranteed to lie on the exact corridor the solver will
#   paint.
# - Candidate blobs must fit entirely in bounds, remain separated from existing
#   blobs by a padded occupancy mask, and be the only blue object on that ray.
# - Truncation samples place one latent continuation blob on the next would-be
#   emission corridor, but they do not draw that corridor in the output.
#
# e) Visual observability
# - Fixed-distance behavior is visible because rays often stop mid-grid and often
#   continue beyond the first activated blob by 1 extra cell.
# - Depth behavior is visible because many height-3 starts show a reachable
#   fourth blob in the input that never turns red in the output.
# - Both effects reuse the same visible cue, so the examples teach "count the
#   red stem" rather than two unrelated measurements.
#
# f) Visual cleanliness
# - Distractors are forbidden in truncated cases, so an unactivated on-path blob
#   can be attributed to the depth cap rather than off-path noise.
# - Ray corridors are short and object spacing is still controlled by the padded
#   occupancy mask, so short-range samples remain readable rather than cramped.
#
# g) Change inferability and new-assumption audit
# - No new colors are introduced. Red still means "result": start marker,
#   activated blob bodies, and traversed ray cells.
# - Each loop/branch in the solver now assumes:
#   * the stem height is 2 or 3 and can be read from a solid bottom segment;
#   * each non-terminal blob has one intended outgoing port edge;
#   * every intended ray has enough in-bounds room for its full step budget;
#   * any latent extra blob appears only as the blocked next emission, never as a
#     second forbidden hit later on an already-painted ray.
# - The generator explicitly constructs samples to satisfy all of these, except
#   that h=2 terminal blobs may also sit against the previous green exit cell.
# ---------------------------------------------------------------------------


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a red activation cascade whose range and depth are both encoded by the start stem.

    Input structure:
    - The background is the dominant color. Blue diamond/octagon blobs each carry one intended green edge port.
    - A bottom-touching red stem of height 2-3 stands below a single yellow cap; that same height is the only numeric cue.

    Rule:
    - Turn the start marker red and fire the upward start ray for exactly as many steps as the stem height.
    - Any touched blue blob activates, turns red, and would emit from its green side for that same fixed distance, but the total number of activated blobs may not exceed the stem height.
    - Rays do not stop on impact; they keep painting until the fixed budget is used up, while blobs beyond the activation budget stay inactive and untouched ports disappear into background.

    Key insight:
    - One visual feature does double duty: counting the red stem gives both the global ray length and the maximum BFS depth, so obstacle-based stopping is no longer the right model.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- STEP 1) Detect the start marker and read its height as the shared budget ---
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
            if cap in (int(bg), int(col)):
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
    ray_steps = int(len(start_seg))
    max_activations = int(len(start_seg))

    # --- STEP 2) Identify blob colors ---
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
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(blue_col):
                    ok = True
                    break
            if ok:
                break
        if ok:
            green_candidates.append(int(col))

    assert len(green_candidates) == 1
    green_col = int(green_candidates[0])

    out = np.full_like(g, int(bg))

    # --- STEP 3) Extract blue blobs and index them ---
    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []
    for coords in components_mask(g == int(blue_col), connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = int(len(blobs))
        for r, c in pts:
            blob_id[int(r), int(c)] = bi
        blobs.append({"pts": pts, "ports": []})

    # --- STEP 4) Recover each blob's visible green-adjacent port sides ---
    for blob in blobs:
        pts = blob["pts"]
        seen_dirs: set[tuple[int, int]] = set()
        for r, c in pts:
            for ddr, ddc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = int(r + ddr)
                cc = int(c + ddc)
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(green_col):
                    seen_dirs.add((int(ddr), int(ddc)))
        for ddr, ddc in seen_dirs:
            port_pixels = [
                (int(r), int(c))
                for (r, c) in pts
                if 0 <= int(r + ddr) < H and 0 <= int(c + ddc) < W and int(g[int(r + ddr), int(c + ddc)]) == int(green_col)
            ]
            assert port_pixels
            blob["ports"].append({"dir": (int(ddr), int(ddc)), "pixels": port_pixels})
        # In h=2 terminal-chain layouts, the last candidate can sit adjacent to
        # the previous blob's green exit cell, which makes the incoming side
        # look like an extra port. We keep all observed sides here; only blobs
        # that remain under the activation budget will actually emit.

    activated: set[int] = set()
    queue: list[int] = []

    def activate_blob_at(r: int, c: int) -> None:
        """Activate a blob once; the last allowed activation does not emit further."""
        if len(activated) >= max_activations:
            return
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in activated:
            return
        activated.add(int(bi))
        if len(activated) < max_activations:
            queue.append(int(bi))
        for pr, pc in blobs[bi]["pts"]:
            out[int(pr), int(pc)] = int(red_col)

    def extend_fixed(starts: list[tuple[int, int]], d: tuple[int, int], steps: int, include_origin: bool) -> None:
        """Paint exactly `steps` cells from each start, activating any blue cell crossed."""
        ddr, ddc = map(int, d)
        for sr, sc in starts:
            r, c = int(sr), int(sc)
            if include_origin:
                out[int(r), int(c)] = int(red_col)
                if int(g[int(r), int(c)]) == int(blue_col):
                    activate_blob_at(int(r), int(c))
            for _ in range(int(steps)):
                r += int(ddr)
                c += int(ddc)
                if not (0 <= r < H and 0 <= c < W):
                    break
                out[int(r), int(c)] = int(red_col)
                if int(g[int(r), int(c)]) == int(blue_col):
                    activate_blob_at(int(r), int(c))

    # --- STEP 5) Fire the start ray for the shared fixed distance ---
    for r, c in start_seg:
        out[int(r), int(c)] = int(red_col)
    top_r = min(int(r) for r, _c in start_seg)
    cap_r = int(top_r) - 1
    assert cap_r >= 0 and int(g[int(cap_r), int(start_col)]) == int(yellow_col)
    extend_fixed([(int(cap_r), int(start_col))], (-1, 0), int(ray_steps), include_origin=True)

    # --- STEP 6) BFS cascade: every emitting blob uses the same fixed distance ---
    while queue:
        bi = int(queue.pop(0))
        for port in blobs[bi]["ports"]:
            extend_fixed(port["pixels"], port["dir"], int(ray_steps), include_origin=False)

    assert len(blobs) - len(activated) <= 2, "at most one distractor plus one cap-blocked continuation blob"
    return out


# --------------- generator (fixed-distance + height-capped combo) ---------------

BG = 8
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)
OPP = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

SHAPES = ((3, 4), (4, 4), (5, 4), (5, 6), (6, 6), (8, 8))
_SHAPE_WIDTHS = {
    (3, 4): (2, 4, 2),
    (4, 4): (2, 4, 4, 2),
    (5, 4): (2, 4, 4, 4, 2),
    (5, 6): (2, 4, 6, 4, 2),
    (6, 6): (2, 4, 6, 6, 4, 2),
    (8, 8): (2, 4, 6, 8, 8, 6, 4, 2),
}

H_MIN, H_MAX = 13, 27
W_MIN, W_MAX = 15, 27
START_COL_PAD = 2

START_H_MIN, START_H_MAX = 2, 3

LATENT_CHAIN_CHOICES = {
    2: [0, 1, 2, 3],
    3: [0, 1, 2, 3, 4],
}
LATENT_CHAIN_WEIGHTS = {
    2: [1, 2, 4, 4],
    3: [1, 2, 3, 3, 8],
}
MAX_BLOB_COUNT = 4
MAX_DISTRACTORS = 1

MIN_GAP_FIRST = 1
MIN_GAP_CHAIN = 1

GEN_RETRIES = 600


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
        patch[r, c] = GREEN

    port_pixels: list[tuple[int, int]] = []
    for gr, gc in green:
        br, bc = gr - port_dir[0], gc - port_dir[1]
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
        "green": green,
        "port_dir": port_dir,
        "port_pixels": _sort_on_ray(port_pixels, port_dir),
    }


def _all_specs(port_dir: tuple[int, int] | None = None) -> list[dict]:
    dirs = [port_dir] if port_dir is not None else list(DIRS)
    out = [s for size in SHAPES for d in dirs if (s := _blob_spec(size, d)) is not None]
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


def _trace_steps(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    steps: int,
    include_origin: bool,
    H: int,
    W: int,
) -> list[tuple[int, int]] | None:
    dr, dc = map(int, d)
    cells: list[tuple[int, int]] = []
    for sr, sc in starts:
        r, c = int(sr), int(sc)
        if include_origin:
            if not (0 <= r < H and 0 <= c < W):
                return None
            cells.append((int(r), int(c)))
        for _ in range(int(steps)):
            r += int(dr)
            c += int(dc)
            if not (0 <= r < H and 0 <= c < W):
                return None
            cells.append((int(r), int(c)))
    return cells


def _mark_ray(out: np.ndarray, ray_mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    paint_coords(out, cells, RED)
    paint_coords(ray_mask, cells, True)


def _step_index_on_ray(cell: tuple[int, int], lane_origin: dict[int, int], d: tuple[int, int]) -> int:
    lane = _perp_value(cell, d)
    return int(abs(_along_value(cell, d) - lane_origin[int(lane)]))


def _candidate_next_blob(
    H: int,
    W: int,
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    steps_limit: int,
    allowed_overlap: set[tuple[int, int]],
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
    prefer_passthrough: bool,
) -> dict | None:
    starts = _sort_on_ray(starts, d)
    full_path = _trace_steps(starts, d, steps_limit, include_origin, H, W)
    if full_path is None:
        return None

    thickness = len(starts)
    lane_vals = [_perp_value(p, d) for p in starts]
    lane_delta = [int(v - lane_vals[0]) for v in lane_vals]
    lane_origin = {int(_perp_value(p, d)): int(_along_value(p, d)) for p in starts}
    entry_side = OPP[d]
    occ_pad = _dilate8(occupied)
    corridor_set = set(full_path) if not include_origin else set()
    min_gap = MIN_GAP_FIRST if include_origin else MIN_GAP_CHAIN
    sign = -1 if d in (UP, LEFT) else 1
    start_line = _along_value(starts[0], d)

    candidates: list[dict] = []
    passthrough: list[dict] = []

    for spec in _all_specs():
        if spec["port_dir"] == entry_side:
            continue
        edge = _edge_offsets(spec["mask"], entry_side)
        if len(edge) < thickness:
            continue
        edge_line = _along_value(edge[0], d)
        for k in range(len(edge) - thickness + 1):
            window = edge[k : k + thickness]
            win_delta = [_perp_value(p, d) - _perp_value(window[0], d) for p in window]
            if win_delta != lane_delta:
                continue
            fixed = lane_vals[0] - _perp_value(window[0], d)
            for hit_step in range(int(min_gap), int(steps_limit) + 1):
                entry_line = start_line + sign * int(hit_step)
                origin_along = int(entry_line - edge_line)
                r0 = int(origin_along if d[0] else fixed)
                c0 = int(fixed if d[0] else origin_along)
                full = _offset(spec["full"], r0, c0)
                if not _in_bounds(full, H, W):
                    continue
                blue = _offset(spec["blue"], r0, c0)
                tight_blue = corridor_set.intersection(blue) if (steps_limit == 2 and not include_origin) else set()
                # Allow tight spacing only for the entry blue cells that the
                # current fixed-distance ray will actually hit.
                if any(
                    bool((((r, c) not in tight_blue) and occ_pad[r, c]) or ray_mask[r, c])
                    for r, c in full
                ):
                    continue

                full_set = set(full)
                if any(
                    bool(occupied[r, c] or ray_mask[r, c])
                    for r, c in full_path
                    if (r, c) not in allowed_overlap and (r, c) not in full_set
                ):
                    continue
                if any(bool(blue_mask[r, c]) for r, c in full_path if (r, c) not in full_set):
                    continue

                blue_set = set(blue)
                touched_blue = [cell for cell in full_path if cell in blue_set]
                if not touched_blue:
                    continue
                first_hit = min(_step_index_on_ray(cell, lane_origin, d) for cell in touched_blue)
                overshoot = int(steps_limit - first_hit)

                cand = {
                    "spec": spec,
                    "r0": int(r0),
                    "c0": int(c0),
                    "path": list(full_path),
                    "blue": blue,
                    "full": full,
                    "ports": _sort_on_ray(_offset(spec["port_pixels"], r0, c0), spec["port_dir"]),
                    "green": _sort_on_ray(_offset(spec["green"], r0, c0), spec["port_dir"]),
                    "hit_step": int(first_hit),
                    "overshoot": int(overshoot),
                }
                candidates.append(cand)
                if int(overshoot) > 0:
                    passthrough.append(cand)

    if prefer_passthrough and passthrough:
        return random.choice(passthrough)
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
                if any(bool(occ_pad[r, c] or ray_mask[r, c]) for r, c in full):
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


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    start_h = random.choices((2, 3), weights=(2, 3))[0]
    cap_r = H - start_h - 1
    if cap_r <= 0:
        return None
    start_col = random.randint(START_COL_PAD, W - START_COL_PAD - 1)
    inp[int(cap_r), int(start_col)] = YELLOW
    occupied[int(cap_r), int(start_col)] = True
    for r in range(int(cap_r) + 1, H):
        inp[int(r), int(start_col)] = RED
        out[int(r), int(start_col)] = RED
        occupied[int(r), int(start_col)] = True

    latent_chain_len = random.choices(
        LATENT_CHAIN_CHOICES[int(start_h)],
        weights=LATENT_CHAIN_WEIGHTS[int(start_h)],
    )[0]
    active_chain_len = min(int(start_h), int(latent_chain_len))
    truncated = bool(latent_chain_len > active_chain_len)

    if truncated:
        num_distractors = 0
    else:
        min_dist = 1 if latent_chain_len == 0 else 0
        max_dist = min(MAX_DISTRACTORS, MAX_BLOB_COUNT - latent_chain_len)
        num_distractors = random.randint(min_dist, max_dist)

    starts = [(int(cap_r), int(start_col))]
    direction = UP
    include_origin = True
    allowed_overlap = {(int(cap_r), int(start_col))}
    ray_steps = int(start_h)

    for idx in range(int(active_chain_len)):
        prefer_passthrough = bool(random.random() < 0.8 or idx == 0)
        blob = _candidate_next_blob(
            H,
            W,
            starts,
            direction,
            include_origin,
            ray_steps,
            allowed_overlap,
            occupied,
            blue_mask,
            ray_mask,
            prefer_passthrough,
        )
        if blob is None:
            return None
        _mark_ray(out, ray_mask, blob["path"])
        _place_blob(inp, occupied, blue_mask, blob)
        paint_coords(out, blob["blue"], RED)

        starts = blob["ports"]
        direction = blob["spec"]["port_dir"]
        include_origin = False
        allowed_overlap = set(blob["green"])

    for _ in range(int(latent_chain_len - active_chain_len)):
        blocked_blob = _candidate_next_blob(
            H,
            W,
            starts,
            direction,
            include_origin,
            ray_steps,
            allowed_overlap,
            occupied,
            blue_mask,
            ray_mask,
            prefer_passthrough=False,
        )
        if blocked_blob is None:
            return None
        _place_blob(inp, occupied, blue_mask, blocked_blob)
        starts = blocked_blob["ports"]
        direction = blocked_blob["spec"]["port_dir"]
        include_origin = False
        allowed_overlap = set(blocked_blob["green"])

    if not truncated:
        final_path = _trace_steps(starts, direction, ray_steps, include_origin, H, W)
        if final_path is None:
            return None
        if any(bool(occupied[r, c] or ray_mask[r, c]) for r, c in final_path if (r, c) not in allowed_overlap):
            return None
        if any(bool(blue_mask[r, c]) for r, c in final_path if (r, c) not in allowed_overlap):
            return None
        _mark_ray(out, ray_mask, final_path)

    for _ in range(int(num_distractors)):
        if not _place_distractor(inp, occupied, blue_mask, ray_mask):
            return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5961cc34 combo input")
