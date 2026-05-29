# Combo variant of 5961cc34 (index 1)
# combo_spec: D2/1+D4/1+D7/2
# Name: 多起始点+单步激活+颜色按源编码

from __future__ import annotations

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
# - Baseline 5961cc34 has one bottom start, upward start ray, blue blobs with
#   green ports, and a BFS cascade that keeps emitting through ports.
# - This combo changes three linked stages at once:
#   1) there are 2-3 bottom starts instead of one;
#   2) each start only cares about the first blue blob on its own vertical ray;
#   3) the hit blob color depends on which source hit it, not on cascade depth.
# - Small trace:
#     original: one start -> blob A turns red -> A emits -> blob B turns red -> ...
#     combo   : left/middle/right starts fire upward independently.
#               Each source stops at its own first blue hit.
#               The left hit blob becomes red, the middle hit blob orange
#               (when a middle source exists), and the right hit blob yellow.
#               No green-port ray is emitted at all.
# - To make the "first hit only" rule observable, the generator often places an
#   extra blue+green blob higher on the same source column. The lower blob is
#   the one that changes color; the upper one stays erased because the ray never
#   goes past the first collision.
#
# b) Complexity calibration
# - If every source column contained only one blob, the "first hit" part would
#   be too weak. The generator therefore injects hidden upper blobs on the same
#   column in many samples.
# - Source-color mapping is kept small and repeated:
#     * 2 starts: left -> red, right -> yellow
#     * 3 starts: left -> red, middle -> orange, right -> yellow
#   This keeps the mapping human-visible without adding too many cases.
# - Blob geometry, start-marker shape, and blue/green input semantics stay close
#   to the source task, so the new difficulty is the fused logic itself.
#
# c) Implicit-assumption audit of the original solver
# - `assert len(start_candidates) == 1` breaks and must become 2-3 starts.
# - The old queue / BFS cascade is invalid because activated blobs no longer
#   emit any follow-up rays.
# - Port discovery is no longer needed for propagation, but the solver still
#   infers the blue/green roles from the unchanged input palette so that the
#   object parsing assumptions remain explicit.
# - The original solver assumed every reachable blob should become red. That is
#   replaced by a source-index color map on only the directly hit first blob.
# - Multi-source overlap could make source-color assignment ambiguous if two
#   sources hit the same blob. The generator forbids that by ensuring each blob
#   intersects at most one source column.
#
# d) Placement legality
# - Start columns are spaced apart so stems stay visually separate.
# - Target blobs are placed so each source has a clear vertical path to its own
#   first hit, and no blob crosses another source column.
# - Hidden upper blobs on a source column are allowed only above the already-hit
#   blocker blob, so they are unreachable but still physically valid.
# - Distractors never overlap objects, start rays, or the reserved one-pixel
#   halo around existing blobs.
#
# e) Visual observability
# - Humans can see multiple identical bottom launchers, so the multi-source
#   change is explicit.
# - The input sometimes shows two stacked blobs on one source column, while the
#   output colors only the lower one. That makes the "first hit only" rule
#   directly inferable from a few examples.
# - The source-color mapping is tied to horizontal role, which is visible in the
#   grid itself rather than to hidden numeric color ids.
#
# f) Visual cleanliness
# - Output contains only thin vertical red rays plus a few colored blob bodies;
#   there are no emitted port rays, which keeps the grids sparser than the
#   original cascade task.
# - Hidden upper blobs are used sparingly, typically at most one per sample, so
#   the input remains readable.
#
# g) Color-change inferability
# - Only three activation colors are used overall: red, orange, yellow.
# - Their assignment follows a repeated positional rule:
#     left source -> red
#     middle source -> orange (only in 3-source cases)
#     right source -> yellow
# - Because all source stems share the same appearance, the solver must infer
#   that output color comes from source position rather than blob shape or size.
#
# h) Quantifiable non-degeneracy target
# - Every generated sample has at least 2 starts and every start hits a blob, so
#   a "single-source" or "all-hit-blobs-red" interpretation fails.
# - Many samples also contain a higher hidden blob on at least one source
#   column, so a solver that colors every blob on a source column also fails.
#
# i) New solver assumptions audited against the generator
# - `_find_starts()` assumes each source is a 1-cell-wide bottom stem of height
#   2 or 3 with a single cap directly above; the generator enforces this.
# - The solver assumes blue is the dominant non-marker body color and green is
#   the unique non-marker color 4-adjacent to blue; every blob patch preserves
#   that property.
# - The solver assumes each directly hit blob belongs to exactly one source. The
#   generator guarantees no blob spans multiple start columns.

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
ORANGE = 7

# --- Directions ---
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)

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
W_MIN, W_MAX = 18, 27

# --- Start markers ---
START_COL_PAD = 2
START_H_MIN, START_H_MAX = 2, 3
MIN_START_SEP = 5
START_COUNT_WEIGHTS = ((2, 1), (3, 3))
THREE_START_W_MIN = 20
THREE_START_H_MIN = 16

# --- Placement ---
MIN_GAP_FIRST = 4
MIN_GAP_STACK = 3
MAX_HIDDEN_STACKED = 1
MAX_FREE_DISTRACTORS = 1

# --- Retry ---
GEN_RETRIES = 500


def _source_color(index: int, total: int) -> int:
    """Map the source's left-to-right role to its output blob color."""
    if total == 2:
        return (RED, YELLOW)[int(index)]
    assert total == 3
    return (RED, ORANGE, YELLOW)[int(index)]


def _find_starts(grid: np.ndarray, bg: int) -> list[dict]:
    """Find every bottom stem with a differently colored cap above it."""
    H, W = grid.shape
    starts: list[dict] = []
    for c in range(W):
        stem_color = int(grid[H - 1, c])
        if stem_color == int(bg):
            continue
        r0 = H - 1
        while r0 >= 0 and int(grid[r0, c]) == int(stem_color):
            r0 -= 1
        top_r = int(r0 + 1)
        height = int(H - top_r)
        if height not in (2, 3):
            continue
        cap_r = int(top_r - 1)
        if cap_r < 0:
            continue
        cap_color = int(grid[cap_r, c])
        if cap_color in (int(bg), int(stem_color)):
            continue
        starts.append(
            {
                "stem_color": int(stem_color),
                "cap_color": int(cap_color),
                "col": int(c),
                "cap_r": int(cap_r),
                "seg": [(int(rr), int(c)) for rr in range(int(top_r), H)],
            }
        )
    starts.sort(key=lambda item: item["col"])
    return starts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the first blob hit by each bottom launcher, using a source-dependent palette.

    Input structure:
    - The background is dominant; blue diamond/octagon blobs still carry one green side port just like the source task.
    - There are 2-3 bottom launchers, each made of a short red stem with a yellow cap directly above it.

    Rule:
    - Convert every stem and cap to red, then fire a thin vertical ray upward from each cap.
    - Each source stops at the first blue blob it meets; that blob alone is activated, and no port ray is emitted.
    - Activated blob bodies are recolored by source position: left source -> red, middle -> orange, right source -> yellow.

    Key insight: The green ports are now only part of the unchanged input vocabulary; the real task is to match each bottom source to its first vertical hit and apply the source-position color code.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- STEP 1) Detect all start markers and sort them left-to-right ---
    starts = _find_starts(g, bg)
    assert 2 <= len(starts) <= 3
    red_colors = {int(start["stem_color"]) for start in starts}
    yellow_colors = {int(start["cap_color"]) for start in starts}
    assert len(red_colors) == 1 and len(yellow_colors) == 1
    red_col = int(next(iter(red_colors)))
    yellow_col = int(next(iter(yellow_colors)))

    # --- STEP 2) Infer blue body and green port colors from the unchanged input palette ---
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
            for dr, dc in DIRS:
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

    out = np.full_like(g, int(bg))

    # --- STEP 3) Extract blue blobs and build a pixel->blob index ---
    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[list[tuple[int, int]]] = []
    for coords in components_mask(g == int(blue_col), connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = int(len(blobs))
        for r, c in pts:
            blob_id[int(r), int(c)] = int(bi)
        blobs.append(pts)

    painted_blobs: set[int] = set()

    def paint_blob_at(r: int, c: int, color: int) -> None:
        """Color the whole blue component containing (r, c) once."""
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in painted_blobs:
            return
        painted_blobs.add(int(bi))
        for pr, pc in blobs[bi]:
            out[int(pr), int(pc)] = int(color)

    # --- STEP 4) Launch every start ray independently; only the first hit matters ---
    total = int(len(starts))
    for index, start in enumerate(starts):
        source_fill = int(_source_color(index, total))
        paint_coords(out, start["seg"], int(red_col))
        cap_r = int(start["cap_r"])
        start_col = int(start["col"])
        assert int(g[cap_r, start_col]) == int(yellow_col)
        out[int(cap_r), int(start_col)] = int(red_col)
        for r in range(int(cap_r) - 1, -1, -1):
            if int(g[int(r), int(start_col)]) == int(blue_col):
                paint_blob_at(int(r), int(start_col), source_fill)
                break
            out[int(r), int(start_col)] = int(red_col)

    return out


def _shape_mask(h: int, w: int) -> np.ndarray:
    """Build the source-task diamond/octagon mask."""
    widths = _SHAPE_WIDTHS[(h, w)]
    mask = np.zeros((h, w), dtype=bool)
    for r, ww in enumerate(widths):
        c0 = (w - ww) // 2
        mask[r, c0 : c0 + ww] = True
    return mask


def _offset(offsets: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(int(r0 + r), int(c0 + c)) for r, c in offsets]


def _in_bounds(coords: list[tuple[int, int]], H: int, W: int) -> bool:
    return all(0 <= int(r) < H and 0 <= int(c) < W for r, c in coords)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    """Dilate by one pixel in 8-neighborhood to keep blobs visually separate."""
    H, W = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r0, r1 = max(0, dr), H + min(0, dr)
            c0, c1 = max(0, dc), W + min(0, dc)
            out[r0:r1, c0:c1] |= mask[r0 - dr : r1 - dr, c0 - dc : c1 - dc]
    return out


def _edge_offsets(mask: np.ndarray, d: tuple[int, int]) -> list[tuple[int, int]]:
    pts = np.argwhere(mask)
    axis = 0 if d in (UP, DOWN) else 1
    extreme = pts[:, axis].min() if d in (UP, LEFT) else pts[:, axis].max()
    keep = pts[:, axis] == extreme
    return [(int(r), int(c)) for r, c in pts[keep]]


def _hit_lanes_up(mask: np.ndarray, blue: list[tuple[int, int]], full: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return local columns where an upward ray first hits a blue cell, not a green cell."""
    full_by_col: dict[int, list[int]] = {}
    blue_by_col: dict[int, list[int]] = {}
    for r, c in full:
        full_by_col.setdefault(int(c), []).append(int(r))
    for r, c in blue:
        blue_by_col.setdefault(int(c), []).append(int(r))

    lanes: list[tuple[int, int]] = []
    for c, rows in blue_by_col.items():
        if max(full_by_col[int(c)]) == max(rows):
            lanes.append((int(c), int(max(rows))))
    return lanes


def _blob_spec(size: tuple[int, int], port_dir: tuple[int, int]) -> dict | None:
    """Build one legal input blob patch plus valid upward hit lanes."""
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

    blue = [
        (int(r), int(c))
        for r, c in np.argwhere(mask)
        if (int(r), int(c)) not in green_set
    ]
    full = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    hit_lanes = _hit_lanes_up(mask, blue, full)
    if not hit_lanes:
        return None

    return {
        "mask": mask,
        "patch": patch,
        "blue": blue,
        "full": full,
        "port_dir": port_dir,
        "hit_lanes_up": hit_lanes,
    }


def _all_specs() -> list[dict]:
    specs = [spec for size in SHAPES for d in DIRS if (spec := _blob_spec(size, d)) is not None]
    random.shuffle(specs)
    return specs


def _choose_start_cols(W: int, num_starts: int) -> list[int] | None:
    cols = list(range(START_COL_PAD, W - START_COL_PAD))
    choices: list[list[int]] = []

    def backtrack(idx: int, chosen: list[int]) -> None:
        if len(chosen) == num_starts:
            choices.append(chosen.copy())
            return
        for j in range(idx, len(cols)):
            c = int(cols[j])
            if chosen and c - chosen[-1] < MIN_START_SEP:
                continue
            chosen.append(c)
            backtrack(j + 1, chosen)
            chosen.pop()

    backtrack(0, [])
    return random.choice(choices) if choices else None


def _vertical_cells(cap_r: int, col: int, stop_exclusive: int) -> list[tuple[int, int]]:
    """Cells on the red vertical ray from the cap upward, excluding the hit row."""
    return [(int(r), int(col)) for r in range(int(cap_r), int(stop_exclusive), -1)]


def _target_blob_candidate(
    H: int,
    W: int,
    start: dict,
    all_start_cols: list[int],
    occupied: np.ndarray,
    blob_mask: np.ndarray,
    ray_mask: np.ndarray,
) -> dict | None:
    """Place the directly hit blob for one source."""
    start_col = int(start["col"])
    cap_r = int(start["cap_r"])
    other_cols = {int(c) for c in all_start_cols if int(c) != start_col}
    occ_pad = _dilate8(occupied)
    candidates: list[dict] = []

    for spec in _all_specs():
        for local_c, entry_local_r in spec["hit_lanes_up"]:
            c0 = int(start_col - local_c)
            max_r0 = int(cap_r - MIN_GAP_FIRST - entry_local_r)
            if max_r0 < 0:
                continue
            r0_values = list(range(max_r0 + 1))
            random.shuffle(r0_values)
            for r0 in r0_values:
                full = _offset(spec["full"], int(r0), int(c0))
                if not _in_bounds(full, H, W):
                    continue
                if any(occ_pad[int(r), int(c)] or ray_mask[int(r), int(c)] for r, c in full):
                    continue
                if any(int(c) in other_cols for _r, c in full):
                    continue
                entry_r = int(r0 + entry_local_r)
                ray_cells = _vertical_cells(cap_r, start_col, entry_r)
                if any(blob_mask[int(r), int(c)] for r, c in ray_cells if int(r) != cap_r):
                    continue
                candidates.append(
                    {
                        "spec": spec,
                        "r0": int(r0),
                        "c0": int(c0),
                        "full": full,
                        "blue": _offset(spec["blue"], int(r0), int(c0)),
                        "entry_r": int(entry_r),
                        "ray_cells": ray_cells,
                    }
                )

    return random.choice(candidates) if candidates else None


def _stacked_hidden_candidate(
    H: int,
    W: int,
    source_col: int,
    max_entry_row: int,
    other_start_cols: set[int],
    occupied: np.ndarray,
    ray_mask: np.ndarray,
) -> dict | None:
    """Optionally place an unseen higher blob on the same source column."""
    occ_pad = _dilate8(occupied)
    candidates: list[dict] = []

    for spec in _all_specs():
        for local_c, entry_local_r in spec["hit_lanes_up"]:
            c0 = int(source_col - local_c)
            max_r0 = int(max_entry_row - entry_local_r)
            if max_r0 < 0:
                continue
            r0_values = list(range(max_r0 + 1))
            random.shuffle(r0_values)
            for r0 in r0_values:
                full = _offset(spec["full"], int(r0), int(c0))
                if not _in_bounds(full, H, W):
                    continue
                if any(occ_pad[int(r), int(c)] or ray_mask[int(r), int(c)] for r, c in full):
                    continue
                if any(int(c) in other_start_cols for _r, c in full):
                    continue
                entry_r = int(r0 + entry_local_r)
                candidates.append(
                    {
                        "spec": spec,
                        "r0": int(r0),
                        "c0": int(c0),
                        "full": full,
                        "blue": _offset(spec["blue"], int(r0), int(c0)),
                        "entry_r": int(entry_r),
                    }
                )

    return random.choice(candidates) if candidates else None


def _place_blob(inp: np.ndarray, occupied: np.ndarray, blob_mask: np.ndarray, blob: dict) -> None:
    paste(inp, blob["spec"]["patch"], int(blob["r0"]), int(blob["c0"]), transparent=BG)
    for r, c in blob["full"]:
        occupied[int(r), int(c)] = True
        blob_mask[int(r), int(c)] = True


def _place_free_distractor(
    inp: np.ndarray,
    occupied: np.ndarray,
    blob_mask: np.ndarray,
    ray_mask: np.ndarray,
    forbidden_cols: set[int],
) -> bool:
    H, W = inp.shape
    occ_pad = _dilate8(occupied)
    candidates: list[dict] = []
    for spec in _all_specs():
        h, w = spec["patch"].shape
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                full = _offset(spec["full"], int(r0), int(c0))
                if any(occ_pad[int(r), int(c)] or ray_mask[int(r), int(c)] for r, c in full):
                    continue
                if any(int(c) in forbidden_cols for _r, c in full):
                    continue
                candidates.append({"spec": spec, "r0": int(r0), "c0": int(c0), "full": full})
    if not candidates:
        return False
    _place_blob(inp, occupied, blob_mask, random.choice(candidates))
    return True


def _choose_start_count(H: int, W: int) -> int:
    # Allow 3-source layouts on moderately sized boards so both source-count
    # branches appear often enough in training examples.
    if W < THREE_START_W_MIN or H < THREE_START_H_MIN:
        return 2
    values, weights = zip(*START_COUNT_WEIGHTS)
    return int(random.choices(values, weights=weights)[0])


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    num_starts = _choose_start_count(H, W)
    start_cols = _choose_start_cols(W, num_starts)
    if start_cols is None:
        return None

    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blob_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    starts: list[dict] = []
    for col in start_cols:
        start_h = random.randint(START_H_MIN, START_H_MAX)
        cap_r = int(H - start_h - 1)
        inp[int(cap_r), int(col)] = YELLOW
        out[int(cap_r), int(col)] = RED
        occupied[int(cap_r), int(col)] = True
        ray_mask[int(cap_r), int(col)] = True
        seg = [(int(r), int(col)) for r in range(int(cap_r) + 1, H)]
        paint_coords(inp, seg, RED)
        paint_coords(out, seg, RED)
        paint_coords(occupied, seg, True)
        starts.append({"col": int(col), "cap_r": int(cap_r), "seg": seg})

    targets: list[dict] = []
    for start in starts:
        target = _target_blob_candidate(H, W, start, start_cols, occupied, blob_mask, ray_mask)
        if target is None:
            return None
        targets.append(target)
        _place_blob(inp, occupied, blob_mask, target)
        paint_coords(out, target["ray_cells"], RED)
        paint_coords(ray_mask, target["ray_cells"], True)

    for index, target in enumerate(targets):
        paint_coords(out, target["blue"], _source_color(index, num_starts))

    hidden_candidates = list(range(num_starts))
    random.shuffle(hidden_candidates)
    stacked_done = 0
    for idx in hidden_candidates:
        if stacked_done >= MAX_HIDDEN_STACKED:
            break
        if random.random() < 0.35:
            continue
        source_col = int(starts[idx]["col"])
        max_entry_row = int(targets[idx]["entry_r"] - MIN_GAP_STACK)
        if max_entry_row < 0:
            continue
        hidden = _stacked_hidden_candidate(
            H,
            W,
            source_col,
            max_entry_row,
            {int(c) for c in start_cols if int(c) != source_col},
            occupied,
            ray_mask,
        )
        if hidden is None:
            continue
        _place_blob(inp, occupied, blob_mask, hidden)
        stacked_done += 1

    num_free_distractors = random.randint(0, MAX_FREE_DISTRACTORS)
    for _ in range(num_free_distractors):
        if not _place_free_distractor(inp, occupied, blob_mask, ray_mask, set(map(int, start_cols))):
            return None

    starts_found = _find_starts(inp, BG)
    if len(starts_found) != num_starts:
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
