# Combo variant of 5961cc34 (index 15)
# combo_spec: D11/1+D16/0
# Name: 等间距排布+端口正交约束

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste

# Design analysis for the fused variant.
# a) Concrete behavior:
#    The original task already shoots a start ray upward, but successive blobs can
#    be placed at irregular distances and may continue straight.  Here the chain is
#    forced into a regular staircase: every hit occurs after the same number of ray
#    steps, and each blob must turn the ray by 90 degrees relative to its incoming
#    direction.  Example: cap -> 5 cells up -> blob A -> 5 cells right -> blob B ->
#    5 cells up -> blob C.  The output rule remains "activate blobs and paint rays
#    red", but the input geometry now visibly advertises equal spacing plus strict
#    axis alternation.
# b) Complexity calibration:
#    The variant keeps only a small set of common gaps (5-7) and chain lengths (2-3)
#    so the pattern repeats within one grid.  That is enough for human-visible
#    regularity without creating dozens of nearly-equivalent cases.
# c) Hidden assumptions reviewed:
#    Original `main()` did not care which port was compatible with an entering ray.
#    The fused solver therefore tracks the incoming direction per activated blob and
#    only emits from the unique orthogonal port.  It also records the first hit
#    distance and requires later blob-to-blob hits to match that same distance.
# d) Placement legality:
#    The generator places blobs output-first with a fixed `gap_steps`; each next
#    entry cell is computed exactly `gap_steps` away from the current ray lanes.
#    Candidates are rejected if they leave the grid, touch padded occupied cells,
#    cross older rays, or put stray blue cells in a ray corridor.
# e) Visual observability:
#    Because chains are almost always length 2-3, one sample already shows repeated
#    equal jumps and the vertical/horizontal alternation.  This is much easier to
#    infer than a latent arithmetic property that appears only once.
# f) Cleanliness:
#    Rays are kept non-self-intersecting by rejecting placements that reuse the
#    existing ray mask, and distractors are placed away from both rays and padded
#    blob footprints.
# g) Inferability:
#    No extra color semantics are introduced.  The same original palette is reused,
#    so the new information is carried by visible geometry: equal hop size and
#    right-angle turning.


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
ORTH = {
    UP: (LEFT, RIGHT),
    DOWN: (LEFT, RIGHT),
    LEFT: (UP, DOWN),
    RIGHT: (UP, DOWN),
}
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
H_MIN, H_MAX = 15, 27
W_MIN, W_MAX = 17, 27
START_COL_PAD = 2

# --- Start segment ---
START_H_MIN, START_H_MAX = 2, 3

# --- Fused chain control ---
CHAIN_LENGTHS = (2, 3)
CHAIN_LEN_WEIGHTS = (3, 2)
GAP_CHOICES = (5, 6, 7)
MAX_BLOB_COUNT = 4
MAX_DISTRACTORS = 1
DISTRACTOR_WEIGHTS = (3, 2)  # 0 or 1

# --- Retry ---
GEN_RETRIES = 800


def _is_orthogonal(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] * b[0] + a[1] * b[1] == 0


def _shape_mask(h: int, w: int) -> np.ndarray:
    """Build a boolean diamond/octagon mask for the given (h, w) shape."""
    widths = _SHAPE_WIDTHS[(h, w)]
    mask = np.zeros((h, w), dtype=bool)
    for r, ww in enumerate(widths):
        c0 = (w - ww) // 2
        mask[r, c0 : c0 + ww] = True
    return mask


def _perp_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    """Coordinate perpendicular to direction d (column for vertical, row for horizontal)."""
    return rc[1] if d[0] else rc[0]


def _sort_on_ray(coords: list[tuple[int, int]], d: tuple[int, int]) -> list[tuple[int, int]]:
    """Sort coordinates by their perpendicular value relative to direction d."""
    return sorted(coords, key=lambda p: _perp_value(p, d))


def _edge_offsets(mask: np.ndarray, d: tuple[int, int]) -> list[tuple[int, int]]:
    """Return sorted mask-local coordinates on the edge facing direction d."""
    pts = np.argwhere(mask)
    axis = 0 if d in (UP, DOWN) else 1
    extreme = pts[:, axis].min() if d in (UP, LEFT) else pts[:, axis].max()
    keep = pts[:, axis] == extreme
    return _sort_on_ray([(int(r), int(c)) for r, c in pts[keep]], d)


def _blob_spec(size: tuple[int, int], port_dir: tuple[int, int]) -> dict | None:
    """Build a blob specification dict for a given shape and port direction."""
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
        br = int(gr - port_dir[0])
        bc = int(gc - port_dir[1])
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


def _all_specs(port_dirs: tuple[tuple[int, int], ...] | None = None) -> list[dict]:
    """Generate all valid blob specs, optionally filtered by port direction."""
    dirs = list(port_dirs) if port_dirs is not None else list(DIRS)
    out = [s for size in SHAPES for d in dirs if (s := _blob_spec(size, d)) is not None]
    random.shuffle(out)
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    """Dilate a boolean mask by 1 pixel in all 8 directions."""
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
    return all(0 <= r < H and 0 <= c < W for r, c in coords)


def _step_coords(coords: list[tuple[int, int]], d: tuple[int, int], k: int) -> list[tuple[int, int]]:
    dr, dc = d
    return [(int(r + dr * k), int(c + dc * k)) for r, c in coords]


def _path_to_entries(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    entries: list[tuple[int, int]],
    H: int,
    W: int,
) -> list[tuple[int, int]] | None:
    """Trace each ray lane from a source pixel to its entry cell, excluding the source pixel."""
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for (r, c), (er, ec) in zip(starts, entries):
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
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    """Trace ray lanes from the source pixels to the boundary, excluding the sources."""
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for r, c in starts:
        while True:
            r += dr
            c += dc
            if not (0 <= r < H and 0 <= c < W):
                break
            cells.append((int(r), int(c)))
    return cells


def _mark_ray(out: np.ndarray, ray_mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    paint_coords(out, cells, RED)
    paint_coords(ray_mask, cells, True)


def _candidate_next_blob(
    H: int,
    W: int,
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    gap_steps: int,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
) -> dict | None:
    """Place the next blob exactly `gap_steps` away along the current lanes."""
    starts = _sort_on_ray(starts, d)
    thickness = len(starts)
    lane_vals = [_perp_value(p, d) for p in starts]
    lane_delta = [v - lane_vals[0] for v in lane_vals]
    target_entries = _step_coords(starts, d, gap_steps)
    entry_side = OPP[d]
    occ_pad = _dilate8(occupied)
    candidates: list[dict] = []

    for spec in _all_specs(ORTH[d]):
        edge = _edge_offsets(spec["mask"], entry_side)
        if len(edge) < thickness:
            continue
        for k in range(len(edge) - thickness + 1):
            window = edge[k : k + thickness]
            win_delta = [_perp_value(p, d) - _perp_value(window[0], d) for p in window]
            if win_delta != lane_delta:
                continue
            r0 = int(target_entries[0][0] - window[0][0])
            c0 = int(target_entries[0][1] - window[0][1])
            entry_cells = _offset(window, r0, c0)
            if entry_cells != target_entries:
                continue
            full = _offset(spec["full"], r0, c0)
            if not _in_bounds(full, H, W):
                continue
            if any(occ_pad[r, c] or ray_mask[r, c] for r, c in full):
                continue
            path = _path_to_entries(starts, d, entry_cells, H, W)
            if path is None:
                continue
            entry_set = set(entry_cells)
            if any(ray_mask[r, c] for r, c in path):
                continue
            if any(blue_mask[r, c] for r, c in path if (r, c) not in entry_set):
                continue
            candidates.append({
                "spec": spec,
                "r0": r0,
                "c0": c0,
                "entry": entry_cells,
                "path": path,
                "blue": _offset(spec["blue"], r0, c0),
                "full": full,
                "ports": _sort_on_ray(_offset(spec["port_pixels"], r0, c0), spec["port_dir"]),
            })

    return random.choice(candidates) if candidates else None


def _place_blob(
    inp: np.ndarray,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    blob: dict,
) -> None:
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
) -> bool:
    """Try to place a single off-ray blob in free space."""
    H, W = inp.shape
    occ_pad = _dilate8(occupied)
    candidates: list[dict] = []
    for spec in _all_specs():
        h, w = spec["mask"].shape
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                full = _offset(spec["full"], r0, c0)
                if any(occ_pad[r, c] or ray_mask[r, c] for r, c in full):
                    continue
                candidates.append({
                    "spec": spec,
                    "r0": r0,
                    "c0": c0,
                    "full": full,
                    "blue": _offset(spec["blue"], r0, c0),
                })
    if not candidates:
        return False
    _place_blob(inp, occupied, blue_mask, random.choice(candidates))
    return True


def _choose_dims(chain_len: int, gap_steps: int) -> tuple[int, int]:
    """Bias dimensions upward enough that a 2-3 hop staircase fits comfortably."""
    min_h = max(H_MIN, 7 + 2 * gap_steps + 2 * max(0, chain_len - 2))
    min_w = max(W_MIN, 9 + gap_steps)
    H = random.randint(min(min_h, H_MAX), H_MAX)
    W = random.randint(min(min_w, W_MAX), W_MAX)
    return H, W


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a red staircase network by activating blue blobs with fixed-length right-angle hops.

    Input structure:
    - The background is the dominant color.  Blue diamond/octagon blobs each have one green edge-port.
    - A yellow cap above a short red segment at the bottom marks the unique start column.
    - Along the active chain, every hit blob sits the same ray-distance from the previous emitter.

    Rule:
    - Fire the start ray upward.  When it reaches a blue blob, repaint that blob red.
    - An activated blob may emit only through the green port whose direction is orthogonal to the
      ray that entered it; the resulting thick ray follows that port until it hits the next blob or
      exits the grid.
    - All blob-to-blob hit distances in the cascade must match the first observed hop length.

    Key insight:
    - Green marks tell you the outgoing direction and width, but the fused variant also encodes a
      strict alternating axis pattern and a repeated hop size, producing a regular zigzag cascade.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- Step 1: locate the start marker (vertical red segment capped by a non-red pixel). ---
    start_candidates: list[tuple[int, int, int, list[tuple[int, int]]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=8):
            if coords.size == 0:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            if c0 != c1 or (r1 - r0 + 1) < 2 or r1 != H - 1:
                continue
            if any(int(g[rr, c0]) != int(col) for rr in range(r0, r1 + 1)):
                continue
            if r0 - 1 < 0:
                continue
            cap = int(g[r0 - 1, c0])
            if cap == bg or cap == int(col):
                continue
            start_candidates.append((int(col), int(cap), int(c0), [(int(rr), int(c0)) for rr in range(r0, r1 + 1)]))

    assert len(start_candidates) == 1
    red_col, yellow_col, start_col, start_seg = start_candidates[0]

    # --- Step 2: infer blue and green. ---
    counts = Counter(map(int, g.flatten()))
    other_cols = [(int(cnt), int(col)) for col, cnt in counts.items() if int(col) not in (bg, red_col, yellow_col)]
    assert other_cols
    max_count = max(cnt for cnt, _ in other_cols)
    blue_candidates = [int(col) for cnt, col in other_cols if cnt == max_count]
    assert len(blue_candidates) == 1
    blue_col = int(blue_candidates[0])

    green_candidates: list[int] = []
    for col in colors_present(g, exclude={bg, red_col, yellow_col, blue_col}):
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue
        if any(
            0 <= int(r + dr) < H and 0 <= int(c + dc) < W and int(g[int(r + dr), int(c + dc)]) == blue_col
            for r, c in pts
            for dr, dc in DIRS
        ):
            green_candidates.append(int(col))
    assert len(green_candidates) == 1
    green_col = int(green_candidates[0])

    out = np.full_like(g, bg)

    # --- Step 3: collect blue blobs and their unique port side. ---
    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []
    for coords in components_mask(g == blue_col, connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = int(len(blobs))
        for r, c in pts:
            blob_id[r, c] = bi
        blobs.append({"pts": pts, "ports": []})

    for blob in blobs:
        pts = blob["pts"]
        seen_dirs: set[tuple[int, int]] = set()
        for r, c in pts:
            for ddr, ddc in DIRS:
                rr = int(r + ddr)
                cc = int(c + ddc)
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == green_col:
                    seen_dirs.add((int(ddr), int(ddc)))
        for ddr, ddc in seen_dirs:
            port_pixels = [
                (int(r), int(c))
                for (r, c) in pts
                if 0 <= int(r + ddr) < H and 0 <= int(c + ddc) < W and int(g[int(r + ddr), int(c + ddc)]) == green_col
            ]
            assert port_pixels
            blob["ports"].append({"dir": (int(ddr), int(ddc)), "pixels": _sort_on_ray(port_pixels, (int(ddr), int(ddc)))})
        assert len(blob["ports"]) == 1

    activated: set[int] = set()
    queue: list[tuple[int, tuple[int, int]]] = []
    common_gap: int | None = None

    def activate_blob_at(r: int, c: int, incoming_dir: tuple[int, int], hit_steps: int | None) -> None:
        """Record an activated blob and enforce the equal-spacing invariant."""
        nonlocal common_gap
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if hit_steps is not None:
            if common_gap is None:
                common_gap = int(hit_steps)
            else:
                assert int(hit_steps) == int(common_gap)
        if bi in activated:
            return
        activated.add(bi)
        queue.append((bi, incoming_dir))
        for pr, pc in blobs[bi]["pts"]:
            out[pr, pc] = red_col

    # --- Step 4: draw the start ray upward. ---
    for r, c in start_seg:
        out[r, c] = red_col
    top_r = min(int(r) for r, _ in start_seg)
    cap_r = int(top_r - 1)
    assert cap_r >= 0 and int(g[cap_r, start_col]) == yellow_col
    out[cap_r, start_col] = red_col

    r = int(cap_r)
    steps = 0
    while True:
        r -= 1
        if r < 0:
            break
        steps += 1
        out[r, start_col] = red_col
        if int(g[r, start_col]) == blue_col:
            activate_blob_at(r, start_col, UP, steps)
            break

    def extend_thick(port_pixels: list[tuple[int, int]], d: tuple[int, int]) -> None:
        """Extend a thick ray lane-by-lane; all lanes must agree on hit distance and target blob."""
        hit_blob: int | None = None
        hit_coord: tuple[int, int] | None = None
        hit_steps: int | None = None

        for pr, pc in port_pixels:
            r = int(pr)
            c = int(pc)
            steps_local = 0
            while True:
                r += int(d[0])
                c += int(d[1])
                if not (0 <= r < H and 0 <= c < W):
                    assert hit_blob is None
                    break
                steps_local += 1
                out[r, c] = red_col
                if int(g[r, c]) == blue_col:
                    bi = int(blob_id[r, c])
                    if hit_blob is None:
                        hit_blob = bi
                        hit_coord = (int(r), int(c))
                        hit_steps = int(steps_local)
                    else:
                        assert bi == hit_blob and int(steps_local) == int(hit_steps)
                    break

        if hit_blob is not None:
            assert hit_coord is not None and hit_steps is not None
            activate_blob_at(hit_coord[0], hit_coord[1], d, hit_steps)

    # --- Step 5: BFS through the chain, enforcing orthogonal exits. ---
    while queue:
        bi, incoming_dir = queue.pop(0)
        valid_ports = [port for port in blobs[bi]["ports"] if _is_orthogonal(port["dir"], incoming_dir)]
        assert len(valid_ports) == 1
        extend_thick(valid_ports[0]["pixels"], valid_ports[0]["dir"])

    assert len(blobs) - len(activated) <= 1, "at most 1 distractor blob"
    assert common_gap is not None, "fused generator always provides at least one hit"
    return out


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid fused-variant input grid."""
    chain_len = random.choices(CHAIN_LENGTHS, weights=CHAIN_LEN_WEIGHTS)[0]
    gap_steps = random.choice(GAP_CHOICES)
    H, W = _choose_dims(chain_len, gap_steps)

    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    # Place the start segment first.
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

    # Build the equal-gap orthogonal chain.
    starts = [(int(cap_r), int(start_col))]
    direction = UP
    for _ in range(chain_len):
        blob = _candidate_next_blob(H, W, starts, direction, gap_steps, occupied, blue_mask, ray_mask)
        if blob is None:
            return None
        _mark_ray(out, ray_mask, blob["path"])
        _place_blob(inp, occupied, blue_mask, blob)
        paint_coords(out, blob["blue"], RED)
        starts = blob["ports"]
        direction = blob["spec"]["port_dir"]

    final_path = _path_to_boundary(starts, direction, H, W)
    if any(blue_mask[r, c] or ray_mask[r, c] for r, c in final_path):
        return None
    _mark_ray(out, ray_mask, final_path)

    max_dist = min(MAX_DISTRACTORS, MAX_BLOB_COUNT - chain_len)
    if max_dist < 0:
        return None
    num_distractors = random.choices(range(max_dist + 1), weights=DISTRACTOR_WEIGHTS[: max_dist + 1])[0]
    for _ in range(num_distractors):
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
    raise RuntimeError("failed to generate a valid fused 5961cc34 input")
