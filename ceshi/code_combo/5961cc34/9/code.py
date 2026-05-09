from __future__ import annotations

# Combo variant of 5961cc34 (index 9)
# combo_spec: D1/2+D4/2
# Name: 螺旋射线+箭头形起始标记

from collections import Counter
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste


# Design analysis for combo D1/2 + D4/2
# a) Concrete behavior trace
#    - Original start object: a bottom red stem with one yellow cap, so Step 1 could hard-code
#      "shoot upward from that cap". Original rays were straight, so path placement depended only
#      on one axis at a time.
#    - New start object: a fixed 3x3 arrow sprite. Four red body pixels make the shaft/wedge and
#      one yellow pixel protrudes on the firing side. The arrow tail is anchored on the boundary
#      opposite the firing direction, so the protruding yellow tip is the visible cue for the
#      initial direction.
#    - New path rule: after launch, a ray keeps advancing cell by cell. If the next step would
#      leave the grid, it turns 90 degrees clockwise and tries again. To make "forms a spiral"
#      executable on a finite rectangle, the same ray also refuses to re-enter its own visited
#      cells; otherwise a boundary-only rule would just loop around the perimeter forever. This
#      yields the intended clockwise inward spiral.
#    - Example sketch: if a right-pointing arrow starts near the left edge, the first leg goes
#      right, the next leg turns downward at the right border, then left at the bottom border,
#      then up, and so on. If that spiral first reaches a blue entry-edge cell of a blob, that
#      blob activates and emits its own spiral from the green port direction. The BFS cascade over
#      blobs is unchanged; only the path geometry and start cue changed.
# b) Complexity calibration
#    - Direction freedom is limited to the four cardinals and one arrow family, so the start cue is
#      rich enough to matter but still easy to recognize across 2-4 ARC examples.
#    - Spiral motion is deterministic: always clockwise, always self-avoiding within one ray. There
#      is no extra hidden parity or left/right switch, so the case count stays controlled.
#    - To keep corners readable, the generator only uses single-pixel ports for chain blobs. The
#      solver still supports multi-pixel ports by tracing one spiral lane per port pixel, but the
#      training distribution avoids visually messy turning ribbons.
# c) Implicit-assumption audit
#    - Original Step 1 assumed a bottom-touching vertical red segment with a distinct cap directly
#      above it. That is false now, so start detection must match the arrow geometry instead.
#    - Original straight-ray code assumed each ray can be described by one direction and one axis
#      interval. That breaks under spiral turning, so both solver and generator must trace rays
#      cell-by-cell.
#    - Original generator assumed candidate blobs could be placed by matching a straight lane to an
#      edge window. That is no longer valid; the new generator enumerates the actual spiral cells
#      and asks which blob placements would be hit first along that ordered path.
#    - New solver assumption: exactly one arrow exists; its red body has area 4; it has exactly one
#      non-red 4-neighbor tip; the full 5-cell shape matches one of four templates; and the tail is
#      on the boundary opposite the inferred direction. The generator enforces all of these.
# d) Placement legality
#    - Objects are still placed output-first. The generator paints the arrow, then for each active
#      ray it computes the exact spiral order before placing the next blob on the first-hit cell of
#      some candidate placement.
#    - Blob footprints are kept in-bounds, non-overlapping, and 8-neighbor separated by an occupied
#      dilation mask. Distractors are placed only after all active rays are fixed and must avoid the
#      full ray mask as well as existing objects.
# e) Visual observability
#    - The arrow makes the initial direction visible without reading color ids. Horizontal starts
#      appear regularly, so the task cannot be mistaken for the original always-up launch.
#    - The generator enforces several clockwise turns somewhere in the visible network. Depending on
#      where the first blob sits, the initial leg may still hit before the first corner, but the
#      solved output as a whole always shows nontrivial spiral turning.
# f) Visual cleanliness
#    - Rays are prevented from intersecting pre-existing objects during generation, and distractors
#      stay off all ray cells. This avoids ambiguous overlaps between blob bodies, ports, and spiral
#      turns.
#    - Restricting chain ports to width 1 keeps turns looking like a single coherent spiral instead
#      of a kinked multi-lane ribbon.
# g) Change inferability
#    - Color roles remain stable from the base task: blue body, green port, red activation/rays,
#      yellow start tip, azure background. The new inference burden is geometric only:
#      "read direction from the arrow, then follow repeated clockwise turns."


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
OPP = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
CW = {UP: RIGHT, RIGHT: DOWN, DOWN: LEFT, LEFT: UP}

# --- Start arrow ---
START_FREE_PAD = 1
START_TEMPLATES = {
    UP: {
        "body": {(1, 0), (1, 1), (1, 2), (2, 1)},
        "tip": (0, 1),
        "tail": (2, 1),
    },
    DOWN: {
        "body": {(0, 1), (1, 0), (1, 1), (1, 2)},
        "tip": (2, 1),
        "tail": (0, 1),
    },
    LEFT: {
        "body": {(0, 1), (1, 1), (1, 2), (2, 1)},
        "tip": (1, 0),
        "tail": (1, 2),
    },
    RIGHT: {
        "body": {(0, 1), (1, 0), (1, 1), (2, 1)},
        "tip": (1, 2),
        "tail": (1, 0),
    },
}
for _spec in START_TEMPLATES.values():
    _spec["full"] = set(_spec["body"]) | {_spec["tip"]}


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

# --- Chain / distractor ---
# Keep exactly one activation hop in every sample:
# 0 would hide the blob-hit / port-emission rule, while 2 is too generation-fragile here.
CHAIN_LENGTHS = (1,)
MAX_BLOB_COUNT = 4
MAX_DISTRACTORS = 1

# --- Placement / visibility ---
MIN_STEPS_FIRST = 6
MIN_STEPS_CHAIN = 3
MIN_FIRST_TURNS = 0
MIN_TOTAL_TURNS = 3

# --- Retry ---
GEN_RETRIES = 400


def _normalize(coords: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return {(r - r0, c - c0) for r, c in coords}, (r0, c0)


def _tail_on_required_boundary(
    tail: tuple[int, int], d: tuple[int, int], H: int, W: int,
) -> bool:
    r, c = tail
    if d == UP:
        return r == H - 1
    if d == DOWN:
        return r == 0
    if d == LEFT:
        return c == W - 1
    return c == 0


def _find_start_marker(g: np.ndarray, bg: int) -> dict:
    H, W = g.shape
    candidates: list[dict] = []

    for body_col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(body_col), connectivity=8):
            body = {(int(r), int(c)) for r, c in coords}
            if len(body) != 4:
                continue

            tip_neighbors: set[tuple[int, int]] = set()
            for r, c in body:
                for dr, dc in DIRS:
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if 0 <= rr < H and 0 <= cc < W:
                        v = int(g[rr, cc])
                        if v != int(bg) and v != int(body_col):
                            tip_neighbors.add((rr, cc))
            if len(tip_neighbors) != 1:
                continue

            tip = next(iter(tip_neighbors))
            tip_col = int(g[tip])
            full = set(body) | {tip}
            if len(full) != 5:
                continue

            norm_full, origin = _normalize(full)
            norm_body = {(r - origin[0], c - origin[1]) for r, c in body}

            matches: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
            for d, spec in START_TEMPLATES.items():
                if norm_full != spec["full"] or norm_body != spec["body"]:
                    continue
                expected_tip = (origin[0] + spec["tip"][0], origin[1] + spec["tip"][1])
                tail = (origin[0] + spec["tail"][0], origin[1] + spec["tail"][1])
                if tip != expected_tip:
                    continue
                if not _tail_on_required_boundary(tail, d, H, W):
                    continue
                matches.append((d, expected_tip, tail))

            if len(matches) != 1:
                continue
            d, tip_rc, tail_rc = matches[0]
            candidates.append(
                {
                    "body_col": int(body_col),
                    "tip_col": int(tip_col),
                    "dir": d,
                    "tip": tip_rc,
                    "tail": tail_rc,
                    "full": sorted(full),
                }
            )

    assert len(candidates) == 1
    return candidates[0]


def _shape_mask(h: int, w: int) -> np.ndarray:
    widths = _SHAPE_WIDTHS[(h, w)]
    mask = np.zeros((h, w), dtype=bool)
    for r, ww in enumerate(widths):
        c0 = (w - ww) // 2
        mask[r, c0 : c0 + ww] = True
    return mask


def _perp_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    return rc[1] if d[0] else rc[0]


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
        br = int(gr - port_dir[0])
        bc = int(gc - port_dir[1])
        if not (0 <= br < h and 0 <= bc < w):
            return None
        if not mask[br, bc] or (br, bc) in green_set:
            return None
        port_pixels.append((br, bc))

    blue = [
        (int(r), int(c))
        for r, c in np.argwhere(mask)
        if (int(r), int(c)) not in green_set
    ]
    full = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    edge_blue_by_side = {
        d: _sort_on_ray([p for p in _edge_offsets(mask, d) if p not in green_set], d)
        for d in DIRS
    }
    return {
        "size": (h, w),
        "mask": mask,
        "patch": patch,
        "blue": blue,
        "full": full,
        "port_dir": port_dir,
        "port_pixels": _sort_on_ray(port_pixels, port_dir),
        "edge_blue_by_side": edge_blue_by_side,
        "port_width": len(port_pixels),
    }


ALL_SPECS = [s for size in SHAPES for d in DIRS if (s := _blob_spec(size, d)) is not None]
CHAIN_SPECS = [s for s in ALL_SPECS if int(s["port_width"]) == 1]


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


def _placed_blob(spec: dict, r0: int, c0: int) -> dict:
    return {
        "spec": spec,
        "r0": int(r0),
        "c0": int(c0),
        "full": _offset(spec["full"], r0, c0),
        "blue": _offset(spec["blue"], r0, c0),
        "ports": _sort_on_ray(_offset(spec["port_pixels"], r0, c0), spec["port_dir"]),
    }


def _spiral_records(
    start: tuple[int, int], direction: tuple[int, int], H: int, W: int, include_origin: bool,
) -> list[dict]:
    start = (int(start[0]), int(start[1]))
    cur = start
    d = (int(direction[0]), int(direction[1]))
    visited = {start}
    records: list[dict] = []
    if include_origin:
        records.append({"cell": start, "dir": d})

    blocked_turns = 0
    while blocked_turns < 4:
        nr = int(cur[0] + d[0])
        nc = int(cur[1] + d[1])
        nxt = (nr, nc)
        if 0 <= nr < H and 0 <= nc < W and nxt not in visited:
            cur = nxt
            visited.add(nxt)
            records.append({"cell": nxt, "dir": d})
            blocked_turns = 0
        else:
            d = CW[d]
            blocked_turns += 1
    return records


def _turn_prefix(records: list[dict]) -> list[int]:
    turns = 0
    prev = None
    out: list[int] = []
    for rec in records:
        d = rec["dir"]
        if prev is not None and d != prev:
            turns += 1
        out.append(int(turns))
        prev = d
    return out


def _trace_spiral_runtime(
    start: tuple[int, int],
    direction: tuple[int, int],
    H: int,
    W: int,
    include_origin: bool,
    visit_cell,
) -> None:
    start = (int(start[0]), int(start[1]))
    cur = start
    d = (int(direction[0]), int(direction[1]))
    visited = {start}
    if include_origin and visit_cell(start):
        return

    blocked_turns = 0
    while blocked_turns < 4:
        nr = int(cur[0] + d[0])
        nc = int(cur[1] + d[1])
        nxt = (nr, nc)
        if 0 <= nr < H and 0 <= nc < W and nxt not in visited:
            cur = nxt
            visited.add(nxt)
            blocked_turns = 0
            if visit_cell(nxt):
                return
        else:
            d = CW[d]
            blocked_turns += 1


def _mark_ray(out: np.ndarray, ray_mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    paint_coords(out, cells, RED)
    paint_coords(ray_mask, cells, True)


def _place_blob(inp: np.ndarray, occupied: np.ndarray, blue_mask: np.ndarray, blob: dict) -> None:
    paste(inp, blob["spec"]["patch"], blob["r0"], blob["c0"], transparent=BG)
    for r, c in blob["full"]:
        occupied[r, c] = True
    for r, c in blob["blue"]:
        blue_mask[r, c] = True


def _make_start_patch(direction: tuple[int, int]) -> np.ndarray:
    patch = np.full((3, 3), BG, dtype=int)
    spec = START_TEMPLATES[direction]
    for r, c in spec["body"]:
        patch[r, c] = RED
    tr, tc = spec["tip"]
    patch[tr, tc] = YELLOW
    return patch


def _start_placement(
    H: int, W: int, direction: tuple[int, int],
) -> tuple[np.ndarray, int, int, tuple[int, int], list[tuple[int, int]]]:
    if direction in (UP, DOWN):
        lane = random.randint(START_FREE_PAD, W - START_FREE_PAD - 1)
        r0 = H - 3 if direction == UP else 0
        c0 = lane - 1
    else:
        lane = random.randint(START_FREE_PAD, H - START_FREE_PAD - 1)
        r0 = lane - 1
        c0 = W - 3 if direction == LEFT else 0

    patch = _make_start_patch(direction)
    spec = START_TEMPLATES[direction]
    tip = (r0 + spec["tip"][0], c0 + spec["tip"][1])
    full = _offset(sorted(spec["full"]), r0, c0)
    return patch, r0, c0, tip, full


def _candidate_next_blob(
    H: int,
    W: int,
    start: tuple[int, int],
    direction: tuple[int, int],
    include_origin: bool,
    occupied: np.ndarray,
    ray_mask: np.ndarray,
    pick_one: bool = True,
) -> dict | list[dict] | None:
    records = _spiral_records(start, direction, H, W, include_origin)
    if not records:
        return None
    turns = _turn_prefix(records)
    forbidden = occupied
    allowed = {tuple(map(int, start))} if include_origin else set()
    first_block = len(records)
    for idx, rec in enumerate(records):
        cell = rec["cell"]
        if cell not in allowed and bool(forbidden[cell[0], cell[1]]):
            first_block = idx
            break

    occ_pad = _dilate8(occupied)
    min_steps = MIN_STEPS_FIRST if include_origin else MIN_STEPS_CHAIN
    specs = CHAIN_SPECS[:]
    random.shuffle(specs)
    candidates: list[dict] = []

    for spec in specs:
        if spec["port_dir"] == OPP[direction]:
            continue
        h, w = spec["mask"].shape
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                blob = _placed_blob(spec, r0, c0)
                if not _in_bounds(blob["full"], H, W):
                    continue
                if any(bool(occ_pad[r, c]) for r, c in blob["full"]):
                    continue

                full_set = set(blob["full"])
                blue_set = set(blob["blue"])
                hit_idx = None
                good = True
                for idx, rec in enumerate(records[:first_block]):
                    cell = rec["cell"]
                    if cell in full_set:
                        if cell in blue_set:
                            hit_idx = idx
                        else:
                            good = False
                        break
                if not good or hit_idx is None or hit_idx < min_steps:
                    continue
                if include_origin and turns[hit_idx] < MIN_FIRST_TURNS:
                    continue

                hit_cell = records[hit_idx]["cell"]
                incoming = records[hit_idx]["dir"]
                entry_side = OPP[incoming]
                local_hit = (hit_cell[0] - r0, hit_cell[1] - c0)
                if local_hit not in spec["edge_blue_by_side"][entry_side]:
                    continue

                candidates.append(
                    {
                        **blob,
                        "path": [rec["cell"] for rec in records[: hit_idx + 1]],
                        "turns": int(turns[hit_idx]),
                    }
                )

    if not pick_one:
        return candidates
    return random.choice(candidates) if candidates else None


def _final_spiral(
    H: int,
    W: int,
    start: tuple[int, int],
    direction: tuple[int, int],
    include_origin: bool,
    blue_hits: np.ndarray,
    ray_mask: np.ndarray,
) -> dict | None:
    records = _spiral_records(start, direction, H, W, include_origin)
    turns = _turn_prefix(records)
    cells: list[tuple[int, int]] = []
    start_cell = tuple(map(int, start))
    for idx, rec in enumerate(records):
        cell = rec["cell"]
        cells.append(cell)
        if cell != start_cell and bool(blue_hits[cell[0], cell[1]]):
            return {"cells": cells, "turns": int(turns[idx])}
    return {"cells": cells, "turns": int(turns[-1]) if turns else 0}


def _place_distractor(
    inp: np.ndarray,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
) -> bool:
    H, W = inp.shape
    occ_pad = _dilate8(occupied)
    specs = ALL_SPECS[:]
    random.shuffle(specs)
    candidates: list[dict] = []
    for spec in specs:
        h, w = spec["mask"].shape
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                blob = _placed_blob(spec, r0, c0)
                if any(bool(occ_pad[r, c] or ray_mask[r, c]) for r, c in blob["full"]):
                    continue
                candidates.append(blob)
    if not candidates:
        return False
    _place_blob(inp, occupied, blue_mask, random.choice(candidates))
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a red activation network whose first launch comes from a small arrow and whose rays curl clockwise into spirals.

    Input structure:
    - The background is the dominant color. Blue diamond/octagon blobs still carry one green edge bar
      that marks the emission side of that blob.
    - The start marker is no longer a bottom stem; it is a boundary-anchored red arrow sprite with one
      yellow protruding tip. The tip side reveals the initial firing direction.

    Rule:
    - Recolor the whole start arrow red, then launch a ray from the yellow tip in the direction implied
      by the arrow shape.
    - Every ray advances one cell at a time. If its next step would leave the grid or re-enter a cell
      already visited by that same ray, it turns 90 degrees clockwise and continues, yielding a spiral.
    - When a ray first reaches a blue blob, that blob turns red and emits new spiral rays through its
      green-marked port direction. Continue breadth-first until no new blobs are reached; untouched blobs
      and ports disappear.

    Key insight:
    - Two cues must be read together: the arrow geometry gives the initial direction, and the repeated
      clockwise turns determine the actual path. After that, the old blob-activation cascade still applies.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- STEP 1) Detect the start arrow sprite and infer the initial direction ---
    start = _find_start_marker(g, bg)
    red_col = int(start["body_col"])
    yellow_col = int(start["tip_col"])
    start_dir = start["dir"]
    start_tip = tuple(map(int, start["tip"]))
    start_full = [(int(r), int(c)) for r, c in start["full"]]

    # --- STEP 2) Identify blob colors ---
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

    # --- STEP 3) Extract blue blobs and build a pixel->blob map ---
    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []
    for coords in components_mask(g == int(blue_col), connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = int(len(blobs))
        for r, c in pts:
            blob_id[r, c] = bi
        blobs.append({"pts": pts, "ports": []})

    # --- STEP 4) Recover each blob's green-marked emission ports ---
    for blob in blobs:
        pts = blob["pts"]
        seen_dirs: set[tuple[int, int]] = set()
        for r, c in pts:
            for ddr, ddc in DIRS:
                rr = int(r + ddr)
                cc = int(c + ddc)
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
            out[int(pr), int(pc)] = int(red_col)

    # --- STEP 5) Trace the arrow-launched spiral until it hits a blob or exhausts itself ---
    for r, c in start_full:
        out[r, c] = int(red_col)

    def visit_ray(cell: tuple[int, int]) -> bool:
        r, c = cell
        out[r, c] = int(red_col)
        if int(g[r, c]) == int(blue_col):
            activate_blob_at(r, c)
            return True
        return False

    _trace_spiral_runtime(start_tip, start_dir, H, W, True, visit_ray)

    def extend_spiral(port_pixels: list[tuple[int, int]], d: tuple[int, int]) -> None:
        for pr, pc in port_pixels:
            _trace_spiral_runtime((int(pr), int(pc)), d, H, W, False, visit_ray)

    # --- STEP 6) BFS propagation over activated blobs ---
    while queue:
        bi = int(queue.pop(0))
        for port in blobs[bi]["ports"]:
            extend_spiral(port["pixels"], port["dir"])

    assert len(blobs) - len(activated) <= 1, "at most 1 distractor blob"
    return out


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    ray_block = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    # Place the directional start arrow.
    start_dir = random.choice(DIRS)
    start_patch, start_r0, start_c0, start_tip, start_full = _start_placement(H, W, start_dir)
    paste(inp, start_patch, start_r0, start_c0, transparent=BG)
    paint_coords(out, start_full, RED)
    for r, c in start_full:
        occupied[r, c] = True

    # Decide chain length and distractor count.
    chain_len = random.choice(CHAIN_LENGTHS)
    max_dist = min(MAX_DISTRACTORS, MAX_BLOB_COUNT - chain_len)
    num_distractors = random.randint(0, max_dist)

    # Build the activation chain.
    current_start = start_tip
    current_dir = start_dir
    include_origin = True
    total_turns = 0

    for _ in range(chain_len):
        candidate_blobs = _candidate_next_blob(
            H, W, current_start, current_dir, include_origin, ray_block, ray_mask, pick_one=False,
        )
        if not candidate_blobs:
            return None
        feasible: list[tuple[dict, dict]] = []
        for blob in candidate_blobs:
            next_blue = blue_mask.copy()
            for r, c in blob["blue"]:
                next_blue[r, c] = True
            final = _final_spiral(H, W, blob["ports"][0], blob["spec"]["port_dir"], False, next_blue, ray_mask)
            if final is None:
                continue
            if int(blob["turns"]) + int(final["turns"]) < MIN_TOTAL_TURNS:
                continue
            feasible.append((blob, final))
        if not feasible:
            return None
        blob, final = random.choice(feasible)
        _mark_ray(out, ray_mask, blob["path"])
        _place_blob(inp, occupied, blue_mask, blob)
        for r, c in blob["full"]:
            ray_block[r, c] = True
        paint_coords(out, blob["blue"], RED)
        total_turns += int(blob["turns"])
        current_start = blob["ports"][0]
        current_dir = blob["spec"]["port_dir"]
        include_origin = False

    # Final spiral for the last active source.
    final = _final_spiral(H, W, current_start, current_dir, include_origin, blue_mask, ray_mask)
    if final is None:
        return None
    _mark_ray(out, ray_mask, final["cells"])
    total_turns += int(final["turns"])

    # Fusion visibility guardrails.
    if int(total_turns) < MIN_TOTAL_TURNS:
        return None

    # Place distractors off all rays.
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
    raise RuntimeError("failed to generate a valid 5961cc34 combo input")
