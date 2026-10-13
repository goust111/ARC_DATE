from __future__ import annotations

import random

import numpy as np


GREY = 5
BACKGROUND = 0
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
SIDES = ("up", "down", "left", "right")
SIDE_TO_DIR = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GAP_LOCAL = {
    "up": (0, 1),
    "down": (2, 1),
    "left": (1, 0),
    "right": (1, 2),
}
DIR_TO_HIT_LOCALS = {
    (0, 1): ((0, 0), (1, 0), (2, 0)),
    (0, -1): ((0, 2), (1, 2), (2, 2)),
    (1, 0): ((0, 0), (0, 1), (0, 2)),
    (-1, 0): ((2, 0), (2, 1), (2, 2)),
}
MIDPOINT_TO_SIDE = {
    (0, 1): "up",
    (2, 1): "down",
    (1, 0): "left",
    (1, 2): "right",
}


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _grow8(mask: np.ndarray) -> np.ndarray:
    grown = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            src_r1 = mask.shape[0] - max(0, dr)
            src_c0 = max(0, -dc)
            src_c1 = mask.shape[1] - max(0, dc)
            dst_r0 = max(0, dr)
            dst_r1 = dst_r0 + (src_r1 - src_r0)
            dst_c0 = max(0, dc)
            dst_c1 = dst_c0 + (src_c1 - src_c0)
            grown[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return grown


def _segment_cells(start: tuple[int, int], hit: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    r, c = start
    cells = []
    while True:
        r += dr
        c += dc
        if (r, c) == hit:
            return cells
        cells.append((r, c))


def _ray_to_boundary(start: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    r, c = start
    cells = [(r, c)]
    while True:
        r += dr
        c += dc
        if not in_bounds(shape, r, c):
            return cells
        cells.append((r, c))


def _frame_patch(center_color: int, gap_side: str, bg: int = BACKGROUND) -> np.ndarray:
    patch = np.full((3, 3), GREY, dtype=int)
    patch[1, 1] = int(center_color)
    gr, gc = GAP_LOCAL[gap_side]
    patch[gr, gc] = int(bg)
    return patch


def _parse_frames(grid: np.ndarray, bg: int) -> tuple[list[dict], dict[tuple[int, int], int], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    frames: list[dict] = []
    wall_to_frame: dict[tuple[int, int], int] = {}
    centers: set[tuple[int, int]] = set()
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            patch = g[r0 : r0 + 3, c0 : c0 + 3]
            center = int(patch[1, 1])
            if center in (bg, GREY):
                continue
            corners = [int(patch[0, 0]), int(patch[0, 2]), int(patch[2, 0]), int(patch[2, 2])]
            if any(val != GREY for val in corners):
                continue
            side_mids = {
                "up": int(patch[0, 1]),
                "down": int(patch[2, 1]),
                "left": int(patch[1, 0]),
                "right": int(patch[1, 2]),
            }
            gap_sides = [side for side, val in side_mids.items() if val == bg]
            if len(gap_sides) != 1:
                continue
            gap_side = gap_sides[0]
            if any(val != GREY for side, val in side_mids.items() if side != gap_side):
                continue

            frame_id = len(frames)
            gap = (r0 + GAP_LOCAL[gap_side][0], c0 + GAP_LOCAL[gap_side][1])
            walls = []
            for dr in range(3):
                for dc in range(3):
                    rr, cc = r0 + dr, c0 + dc
                    if (dr, dc) == (1, 1) or (dr, dc) == GAP_LOCAL[gap_side]:
                        continue
                    walls.append((rr, cc))
                    wall_to_frame[(rr, cc)] = frame_id
            frames.append(
                {
                    "top": r0,
                    "left": c0,
                    "gap_side": gap_side,
                    "gap": gap,
                    "center_color": center,
                    "out_dir": SIDE_TO_DIR[gap_side],
                    "walls": tuple(walls),
                }
            )
            centers.add((r0 + 1, c0 + 1))
    return frames, wall_to_frame, centers


def _find_seed(grid: np.ndarray, bg: int, centers: set[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int], int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates = []
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            color = int(g[r, c])
            if color in (bg, GREY) or (r, c) in centers:
                continue
            dirs = []
            if r == 0:
                dirs.append((1, 0))
            if r == h - 1:
                dirs.append((-1, 0))
            if c == 0:
                dirs.append((0, 1))
            if c == w - 1:
                dirs.append((0, -1))
            if len(dirs) == 1:
                candidates.append(((r, c), dirs[0], color))
    assert len(candidates) == 1, "expected exactly one edge seed"
    return candidates[0]


def _trace_ray(
    grid: np.ndarray,
    out: np.ndarray,
    bg: int,
    wall_to_frame: dict[tuple[int, int], int],
    start: tuple[int, int],
    direction: tuple[int, int],
    color: int,
    *,
    paint_start: bool,
) -> int | None:
    r, c = start
    if paint_start and int(grid[r, c]) == bg:
        out[r, c] = int(color)
    dr, dc = direction
    while True:
        nr, nc = r + dr, c + dc
        if not in_bounds(grid.shape, nr, nc):
            return None
        if (nr, nc) in wall_to_frame:
            return wall_to_frame[(nr, nc)]
        if int(grid[nr, nc]) == bg:
            out[nr, nc] = int(color)
            r, c = nr, nc
            continue
        return None


def _simulate(grid: np.ndarray, *, return_trace: bool = False):
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frames, wall_to_frame, centers = _parse_frames(g, bg)
    seed_pos, seed_dir, seed_color = _find_seed(g, bg, centers)

    out = g.copy()
    current_start = seed_pos
    current_dir = seed_dir
    current_color = seed_color
    paint_start = False
    activated: set[int] = set()
    trace: list[int] = []

    while True:
        hit = _trace_ray(
            g,
            out,
            bg,
            wall_to_frame,
            current_start,
            current_dir,
            current_color,
            paint_start=paint_start,
        )
        if hit is None or hit in activated:
            break
        activated.add(hit)
        trace.append(hit)
        frame = frames[hit]
        current_start = frame["gap"]
        current_dir = frame["out_dir"]
        current_color = int(frame["center_color"])
        paint_start = True

    if return_trace:
        return out, trace, frames
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the edge seed as a relay of rays through gapped grey frames.

    Input structure:
    - The grid has a dominant background, exactly one non-grey seed pixel on one border,
      and several 3x3 grey hollow frames whose border misses one side midpoint.
    - Every such frame has a colored center pixel and one background gap cell on its border.

    Rule:
    - The seed fires inward perpendicular to its border and paints background cells in its own color.
    - A ray stops when it first reaches a grey frame wall; that struck frame then fires a new ray
      from its gap outward, using its center color, and the gap cell itself also becomes painted.
    - The relay repeats until a ray exits the grid; untouched frames remain unchanged.

    Key insight:
    - Each 3x3 grey frame acts as a dormant nozzle: being hit matters, and the gap direction plus
      center color uniquely determine the next painted segment.
    """
    return _simulate(np.asarray(grid, dtype=int))


def _seed_choices(h: int, w: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    choices = []
    for c in range(2, w - 2):
        choices.append(((0, c), (1, 0)))
        choices.append(((h - 1, c), (-1, 0)))
    for r in range(2, h - 2):
        choices.append(((r, 0), (0, 1)))
        choices.append(((r, w - 1), (0, -1)))
    random.shuffle(choices)
    return choices


def _mark(mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        mask[r, c] = True


def _candidate_chain_step(
    shape: tuple[int, int],
    emitter: tuple[int, int],
    direction: tuple[int, int],
    footprint_mask: np.ndarray,
    padded_mask: np.ndarray,
    path_mask: np.ndarray,
    remaining: int,
) -> list[dict]:
    h, w = shape
    candidates: list[dict] = []
    distances = list(range(3, max(h, w)))
    random.shuffle(distances)
    for dist in distances:
        hit = (emitter[0] + direction[0] * dist, emitter[1] + direction[1] * dist)
        if not in_bounds(shape, hit[0], hit[1]):
            continue
        path_cells = _segment_cells(emitter, hit, direction)
        if len(path_cells) < 2:
            continue
        if any(footprint_mask[r, c] or path_mask[r, c] for r, c in path_cells):
            continue
        locals_for_hit = list(DIR_TO_HIT_LOCALS[direction])
        random.shuffle(locals_for_hit)
        for local_hit in locals_for_hit:
            top = hit[0] - local_hit[0]
            left = hit[1] - local_hit[1]
            if top < 0 or left < 0 or top + 2 >= h or left + 2 >= w:
                continue
            footprint = [(top + dr, left + dc) for dr in range(3) for dc in range(3)]
            if any(padded_mask[r, c] or path_mask[r, c] for r, c in footprint):
                continue
            blocked_gap_side = MIDPOINT_TO_SIDE.get(local_hit)
            gap_sides = [side for side in SIDES if side != blocked_gap_side]
            random.shuffle(gap_sides)
            for gap_side in gap_sides:
                gap = (top + GAP_LOCAL[gap_side][0], left + GAP_LOCAL[gap_side][1])
                out_dir = SIDE_TO_DIR[gap_side]
                final_cells = _ray_to_boundary(gap, out_dir, shape)
                if remaining == 1:
                    if len(final_cells) < 2:
                        continue
                    blocked = set(path_cells)
                    if any(footprint_mask[r, c] or path_mask[r, c] or (r, c) in blocked for r, c in final_cells[1:]):
                        continue
                candidates.append(
                    {
                        "top": top,
                        "left": left,
                        "hit": hit,
                        "path_cells": path_cells,
                        "footprint": footprint,
                        "gap_side": gap_side,
                        "gap": gap,
                        "out_dir": out_dir,
                        "final_cells": final_cells,
                    }
                )
    random.shuffle(candidates)
    return candidates


def _build_chain(
    shape: tuple[int, int],
    emitter: tuple[int, int],
    direction: tuple[int, int],
    remaining: int,
    footprint_mask: np.ndarray,
    padded_mask: np.ndarray,
    path_mask: np.ndarray,
) -> tuple[list[dict], np.ndarray] | None:
    candidates = _candidate_chain_step(shape, emitter, direction, footprint_mask, padded_mask, path_mask, remaining)
    for cand in candidates[:120]:
        new_footprint = footprint_mask.copy()
        _mark(new_footprint, cand["footprint"])
        new_padded = _grow8(new_footprint)
        new_path = path_mask.copy()
        _mark(new_path, cand["path_cells"])

        frame = {
            "top": cand["top"],
            "left": cand["left"],
            "gap_side": cand["gap_side"],
        }
        if remaining == 1:
            _mark(new_path, cand["final_cells"])
            return [frame], new_path

        result = _build_chain(
            shape,
            cand["gap"],
            cand["out_dir"],
            remaining - 1,
            new_footprint,
            new_padded,
            new_path,
        )
        if result is not None:
            frames, final_path = result
            return [frame] + frames, final_path
    return None


def _place_decoys(
    grid: np.ndarray,
    footprint_mask: np.ndarray,
    path_mask: np.ndarray,
    min_count: int,
    max_count: int,
) -> list[dict]:
    h, w = grid.shape
    decoys: list[dict] = []
    occupied = footprint_mask.copy()
    padded = _grow8(occupied)
    target = random.randint(min_count, max_count)
    slots = [(r, c) for r in range(h - 2) for c in range(w - 2)]
    random.shuffle(slots)
    for top, left in slots:
        if len(decoys) >= target:
            break
        footprint = [(top + dr, left + dc) for dr in range(3) for dc in range(3)]
        if any(padded[r, c] or path_mask[r, c] for r, c in footprint):
            continue
        decoys.append(
            {
                "top": top,
                "left": left,
                "gap_side": random.choice(SIDES),
            }
        )
        _mark(occupied, footprint)
        padded = _grow8(occupied)
    return decoys


def _compose_grid(shape: tuple[int, int], seed: tuple[int, int], seed_color: int, frames: list[dict]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    grid[seed] = int(seed_color)
    colors = random.sample(PALETTE, k=min(len(PALETTE), len(frames) + 1))
    if len(colors) < len(frames) + 1:
        while len(colors) < len(frames) + 1:
            colors.append(random.choice(PALETTE))
    for frame, center_color in zip(frames, colors[1:]):
        patch = _frame_patch(center_color, frame["gap_side"])
        r0, c0 = frame["top"], frame["left"]
        grid[r0 : r0 + 3, c0 : c0 + 3] = patch
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 28)
    w = random.randint(18, 28)
    shape = (h, w)
    chain_len = random.randint(1, 4)

    for seed, seed_dir in _seed_choices(h, w):
        footprint_mask = np.zeros(shape, dtype=bool)
        padded_mask = np.zeros(shape, dtype=bool)
        path_mask = np.zeros(shape, dtype=bool)
        footprint_mask[seed] = True
        padded_mask = _grow8(footprint_mask)

        built = _build_chain(shape, seed, seed_dir, chain_len, footprint_mask, padded_mask, path_mask)
        if built is None:
            continue
        chain_frames, full_path = built

        footprint_mask = np.zeros(shape, dtype=bool)
        footprint_mask[seed] = True
        for frame in chain_frames:
            cells = [(frame["top"] + dr, frame["left"] + dc) for dr in range(3) for dc in range(3)]
            _mark(footprint_mask, cells)

        decoys = _place_decoys(np.zeros(shape, dtype=int), footprint_mask, full_path, 1, 4)
        all_frames = chain_frames + decoys
        seed_color = random.choice(PALETTE)
        grid = _compose_grid(shape, seed, seed_color, all_frames)

        try:
            out, trace, parsed = _simulate(grid, return_trace=True)
        except AssertionError:
            continue
        if len(parsed) != len(all_frames):
            continue
        if len(trace) != chain_len:
            continue
        expected_hits = {(frame["top"], frame["left"]) for frame in chain_frames}
        actual_hits = {(parsed[idx]["top"], parsed[idx]["left"]) for idx in trace}
        if actual_hits != expected_hits:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
