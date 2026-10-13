from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
CYAN = 8

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
SIDE_TO_OUT = {
    "top": (-1, 0),
    "right": (0, 1),
    "bottom": (1, 0),
    "left": (0, -1),
}
ENTRY_SIDE_FOR_DIR = {
    (-1, 0): "bottom",
    (0, 1): "left",
    (1, 0): "top",
    (0, -1): "right",
}
OPPOSITE_SIDE = {
    "top": "bottom",
    "bottom": "top",
    "left": "right",
    "right": "left",
}
ORTHOGONAL_SIDES = {
    "top": ("left", "right"),
    "bottom": ("left", "right"),
    "left": ("top", "bottom"),
    "right": ("top", "bottom"),
}
GEN_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    return mask.astype(np.int16).cumsum(axis=0).cumsum(axis=1)


def _rect_sum(pref: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    if r0 > r1 or c0 > c1:
        return 0
    total = int(pref[r1, c1])
    if r0:
        total -= int(pref[r0 - 1, c1])
    if c0:
        total -= int(pref[r1, c0 - 1])
    if r0 and c0:
        total += int(pref[r0 - 1, c0 - 1])
    return total


def _gap_coord(r0: int, c0: int, h: int, w: int, side: str, offset: int) -> tuple[int, int]:
    if side == "top":
        return r0, c0 + offset
    if side == "bottom":
        return r0 + h - 1, c0 + offset
    if side == "left":
        return r0 + offset, c0
    if side == "right":
        return r0 + offset, c0 + w - 1
    raise ValueError(side)


def _bbox_from_side_gap(
    gap: tuple[int, int], side: str, offset: int, h: int, w: int
) -> tuple[int, int, int, int]:
    gr, gc = gap
    if side == "top":
        r0, c0 = gr, gc - offset
    elif side == "bottom":
        r0, c0 = gr - (h - 1), gc - offset
    elif side == "left":
        r0, c0 = gr - offset, gc
    elif side == "right":
        r0, c0 = gr - offset, gc - (w - 1)
    else:
        raise ValueError(side)
    return r0, c0, r0 + h - 1, c0 + w - 1


def _iter_bbox_cells(bbox: tuple[int, int, int, int]) -> Iterable[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            yield r, c


def _bbox_separated(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1
) -> bool:
    return a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1]


def _segment_cells(start: tuple[int, int], direction: tuple[int, int], steps: int) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    return [(r + dr * k, c + dc * k) for k in range(1, steps + 1)]


def _make_frame(
    r0: int,
    c0: int,
    h: int,
    w: int,
    gap1_side: str,
    gap1_offset: int,
    gap2_side: str,
    gap2_offset: int,
) -> dict:
    gap1 = _gap_coord(r0, c0, h, w, gap1_side, gap1_offset)
    gap2 = _gap_coord(r0, c0, h, w, gap2_side, gap2_offset)
    assert gap1 != gap2
    return {
        "bbox": (r0, c0, r0 + h - 1, c0 + w - 1),
        "gaps": (gap1, gap2),
        "gap_sides": {
            gap1: gap1_side,
            gap2: gap2_side,
        },
    }


def _draw_frame(grid: np.ndarray, frame: dict) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    grid[r0, c0 : c1 + 1] = BLUE
    grid[r1, c0 : c1 + 1] = BLUE
    grid[r0 : r1 + 1, c0] = BLUE
    grid[r0 : r1 + 1, c1] = BLUE
    for gr, gc in frame["gaps"]:
        grid[gr, gc] = BLACK


def _find_emitter(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    yellow_pts = np.argwhere(grid == YELLOW)
    assert yellow_pts.shape == (1, 2), "expected exactly one yellow start pixel"
    yr, yc = map(int, yellow_pts[0])

    red_neighbors = []
    for dr, dc in DIRS:
        rr, cc = yr + dr, yc + dc
        if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == RED:
            red_neighbors.append((rr, cc))
    assert len(red_neighbors) == 1, "yellow must have exactly one adjacent red lead cell"

    rr, cc = red_neighbors[0]
    direction = (yr - rr, yc - cc)
    seg = []
    cur_r, cur_c = rr, cc
    while _in_bounds(grid.shape, cur_r, cur_c) and int(grid[cur_r, cur_c]) == RED:
        seg.append((cur_r, cur_c))
        cur_r -= direction[0]
        cur_c -= direction[1]
    red_coords = {tuple(map(int, p)) for p in np.argwhere(grid == RED)}
    assert red_coords == set(seg), "red lead must be a single straight segment"
    return (yr, yc), direction


def _find_frames(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    blue_pref = _prefix_sum(grid == BLUE)
    nonblack_pref = _prefix_sum(grid != BLACK)
    frames = []

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            if int(grid[r0, c0]) != BLUE:
                continue
            for r1 in range(r0 + 2, h):
                if int(grid[r1, c0]) != BLUE:
                    continue
                for c1 in range(c0 + 2, w):
                    if int(grid[r0, c1]) != BLUE or int(grid[r1, c1]) != BLUE:
                        continue

                    border_len = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
                    if _rect_sum(nonblack_pref, r0 + 1, c0 + 1, r1 - 1, c1 - 1) != 0:
                        continue
                    blue_count = _rect_sum(blue_pref, r0, c0, r1, c1)
                    nonblack_count = _rect_sum(nonblack_pref, r0, c0, r1, c1)
                    if blue_count != border_len - 2 or nonblack_count != blue_count:
                        continue

                    gaps = []
                    ok = True
                    for c in range(c0 + 1, c1):
                        if int(grid[r0, c]) == BLACK:
                            gaps.append((r0, c))
                        elif int(grid[r0, c]) != BLUE:
                            ok = False
                        if int(grid[r1, c]) == BLACK:
                            gaps.append((r1, c))
                        elif int(grid[r1, c]) != BLUE:
                            ok = False
                    for r in range(r0 + 1, r1):
                        if int(grid[r, c0]) == BLACK:
                            gaps.append((r, c0))
                        elif int(grid[r, c0]) != BLUE:
                            ok = False
                        if int(grid[r, c1]) == BLACK:
                            gaps.append((r, c1))
                        elif int(grid[r, c1]) != BLUE:
                            ok = False
                    if not ok or len(gaps) != 2:
                        continue

                    gap_sides = {}
                    for gr, gc in gaps:
                        if gr == r0:
                            gap_sides[(gr, gc)] = "top"
                        elif gr == r1:
                            gap_sides[(gr, gc)] = "bottom"
                        elif gc == c0:
                            gap_sides[(gr, gc)] = "left"
                        else:
                            gap_sides[(gr, gc)] = "right"
                    frames.append(
                        {
                            "bbox": (r0, c0, r1, c1),
                            "gaps": tuple(gaps),
                            "gap_sides": gap_sides,
                        }
                    )
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Send a cyan ray through hollow frames, using each border gap as a doorway and relay.

    Input structure:
    - The grid has a black background, several blue one-cell-thick hollow rectangular frames, and each frame has exactly two non-corner border gaps.
    - A single yellow pixel is attached to a short straight red lead segment; together they mark the ray origin and its initial direction.
    - Frame interiors and all empty space outside the frames are black in the input.

    Rule:
    - Emit a cyan ray from the yellow pixel, continuing in the same direction away from the red lead.
    - Every black cell the ray traverses becomes cyan; if the next cell is a solid blue border cell, the ray stops there.
    - If the ray enters a frame through one of its gaps, fill that frame's interior cyan, then continue from the frame's other gap in the outward direction perpendicular to that gap's side.
    - Keep all blue border pixels, the red lead, the yellow start pixel, and every untouched frame unchanged.

    Key insight:
    - A border gap is a usable door, not a defect: entering exactly through the doorway activates the whole frame and redirects the ray according to the other doorway's side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    start, direction = _find_emitter(g)
    frames = _find_frames(g)
    assert frames, "expected at least one frame"

    gap_map = {}
    for i, frame in enumerate(frames):
        gap_a, gap_b = frame["gaps"]
        side_a = frame["gap_sides"][gap_a]
        side_b = frame["gap_sides"][gap_b]
        gap_map[gap_a] = {
            "frame": i,
            "gap_out": SIDE_TO_OUT[side_a],
            "other_gap": gap_b,
            "other_out": SIDE_TO_OUT[side_b],
        }
        gap_map[gap_b] = {
            "frame": i,
            "gap_out": SIDE_TO_OUT[side_b],
            "other_gap": gap_a,
            "other_out": SIDE_TO_OUT[side_a],
        }

    out = g.copy()
    entered = set()
    r, c = start
    dr, dc = direction
    seen = set()

    while True:
        state = (r, c, dr, dc)
        assert state not in seen, "ray path must not cycle"
        seen.add(state)

        nr, nc = r + dr, c + dc
        if not _in_bounds(g.shape, nr, nc):
            break
        if int(g[nr, nc]) == BLUE:
            break
        assert int(g[nr, nc]) == BLACK, "ray may travel only through black cells and frame gaps"

        out[nr, nc] = CYAN
        info = gap_map.get((nr, nc))
        if info is not None and (dr, dc) == (-info["gap_out"][0], -info["gap_out"][1]):
            fi = int(info["frame"])
            if fi not in entered:
                entered.add(fi)
                r0, c0, r1, c1 = frames[fi]["bbox"]
                out[r0 + 1 : r1, c0 + 1 : c1] = CYAN
            other_gap = info["other_gap"]
            other_out = info["other_out"]
            out[other_gap] = CYAN
            r, c = other_gap
            dr, dc = other_out
            continue

        r, c = nr, nc

    return out


def _random_frame_dims() -> tuple[int, int]:
    return random.randint(4, 8), random.randint(4, 8)


def _random_offset(side: str, h: int, w: int) -> int:
    span = w if side in ("top", "bottom") else h
    return random.randint(1, span - 2)


def _random_exit_side(entry_side: str) -> str:
    straight = OPPOSITE_SIDE[entry_side]
    if random.random() < 0.35:
        return straight
    return random.choice(ORTHOGONAL_SIDES[entry_side])


def _frame_cells_conflict(
    bbox: tuple[int, int, int, int],
    frames: list[dict],
    occupied: set[tuple[int, int]],
) -> bool:
    for other in frames:
        if not _bbox_separated(bbox, other["bbox"], pad=1):
            return True
    for cell in _iter_bbox_cells(bbox):
        if cell in occupied:
            return True
    return False


def _try_place_relay(
    shape: tuple[int, int],
    current: tuple[int, int],
    direction: tuple[int, int],
    frames: list[dict],
    occupied: set[tuple[int, int]],
    ray_cells: set[tuple[int, int]],
) -> dict | None:
    h_grid, w_grid = shape
    entry_side = ENTRY_SIDE_FOR_DIR[direction]
    for _ in range(80):
        h, w = _random_frame_dims()
        dist = random.randint(2, 6)
        entry_offset = _random_offset(entry_side, h, w)
        entry_gap = (current[0] + direction[0] * dist, current[1] + direction[1] * dist)
        bbox = _bbox_from_side_gap(entry_gap, entry_side, entry_offset, h, w)
        r0, c0, r1, c1 = bbox
        if r0 < 1 or c0 < 1 or r1 >= h_grid - 1 or c1 >= w_grid - 1:
            continue
        seg = _segment_cells(current, direction, dist)
        if any(not _in_bounds(shape, r, c) or (r, c) in occupied or (r, c) in ray_cells for r, c in seg):
            continue
        if _frame_cells_conflict(bbox, frames, occupied):
            continue

        exit_side = _random_exit_side(entry_side)
        exit_offset = _random_offset(exit_side, h, w)
        frame = _make_frame(r0, c0, h, w, entry_side, entry_offset, exit_side, exit_offset)
        exit_gap = frame["gaps"][1]
        exit_out = SIDE_TO_OUT[exit_side]
        step_out = (exit_gap[0] + exit_out[0], exit_gap[1] + exit_out[1])
        if not _in_bounds(shape, *step_out):
            continue
        if step_out in occupied or step_out in ray_cells:
            continue
        return {
            "frame": frame,
            "entry_gap": entry_gap,
            "exit_gap": exit_gap,
            "exit_dir": exit_out,
            "segment": seg,
        }
    return None


def _random_two_gaps(side_forbidden: str, h: int, w: int) -> tuple[tuple[str, int], tuple[str, int]]:
    sides = ["top", "right", "bottom", "left"]
    gap1_side = random.choice([s for s in sides if s != side_forbidden])
    gap2_choices = [s for s in sides if s != side_forbidden]
    gap2_side = random.choice(gap2_choices)
    gap1 = (gap1_side, _random_offset(gap1_side, h, w))
    for _ in range(20):
        gap2 = (gap2_side, _random_offset(gap2_side, h, w))
        if gap2 != gap1:
            return gap1, gap2
        gap2_side = random.choice(gap2_choices)
    return gap1, (OPPOSITE_SIDE[gap1_side], _random_offset(OPPOSITE_SIDE[gap1_side], h, w))


def _try_place_blocker(
    shape: tuple[int, int],
    current: tuple[int, int],
    direction: tuple[int, int],
    frames: list[dict],
    occupied: set[tuple[int, int]],
    ray_cells: set[tuple[int, int]],
) -> dict | None:
    h_grid, w_grid = shape
    hit_side = ENTRY_SIDE_FOR_DIR[direction]
    for _ in range(80):
        h, w = _random_frame_dims()
        dist = random.randint(2, 6)
        hit_offset = _random_offset(hit_side, h, w)
        hit = (current[0] + direction[0] * dist, current[1] + direction[1] * dist)
        bbox = _bbox_from_side_gap(hit, hit_side, hit_offset, h, w)
        r0, c0, r1, c1 = bbox
        if r0 < 1 or c0 < 1 or r1 >= h_grid - 1 or c1 >= w_grid - 1:
            continue
        seg = _segment_cells(current, direction, dist - 1)
        if any(not _in_bounds(shape, r, c) or (r, c) in occupied or (r, c) in ray_cells for r, c in seg):
            continue
        if hit in occupied or hit in ray_cells:
            continue
        if _frame_cells_conflict(bbox, frames, occupied):
            continue

        gap1, gap2 = _random_two_gaps(hit_side, h, w)
        frame = _make_frame(r0, c0, h, w, gap1[0], gap1[1], gap2[0], gap2[1])
        if hit in frame["gaps"]:
            continue
        return {"frame": frame, "segment": seg}
    return None


def _final_edge_segment(
    shape: tuple[int, int],
    current: tuple[int, int],
    direction: tuple[int, int],
    occupied: set[tuple[int, int]],
    ray_cells: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    cells = []
    r, c = current
    dr, dc = direction
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            break
        if (r, c) in occupied or (r, c) in ray_cells:
            return None
        cells.append((r, c))
    return cells if cells else None


def _try_place_distractor(
    shape: tuple[int, int],
    frames: list[dict],
    occupied: set[tuple[int, int]],
    ray_cells: set[tuple[int, int]],
) -> dict | None:
    h_grid, w_grid = shape
    for _ in range(120):
        h, w = _random_frame_dims()
        r0 = random.randint(1, h_grid - h - 1)
        c0 = random.randint(1, w_grid - w - 1)
        sides = random.sample(["top", "right", "bottom", "left"], 2)
        frame = _make_frame(
            r0,
            c0,
            h,
            w,
            sides[0],
            _random_offset(sides[0], h, w),
            sides[1],
            _random_offset(sides[1], h, w),
        )
        if _frame_cells_conflict(frame["bbox"], frames, occupied):
            continue
        if any(cell in ray_cells for cell in _iter_bbox_cells(frame["bbox"])):
            continue
        return frame
    return None


def _build_attempt() -> np.ndarray | None:
    h = random.randint(20, 28)
    w = random.randint(20, 28)
    grid = np.zeros((h, w), dtype=int)

    direction = random.choice(DIRS)
    lead_len = random.randint(1, 3)
    row_lo = 4 + (lead_len if direction[0] == 1 else 0)
    row_hi = h - 5 - (lead_len if direction[0] == -1 else 0)
    col_lo = 4 + (lead_len if direction[1] == 1 else 0)
    col_hi = w - 5 - (lead_len if direction[1] == -1 else 0)
    if row_lo > row_hi or col_lo > col_hi:
        return None
    start = (random.randint(row_lo, row_hi), random.randint(col_lo, col_hi))

    occupied = {start}
    ray_cells: set[tuple[int, int]] = set()
    for k in range(1, lead_len + 1):
        rr = start[0] - direction[0] * k
        cc = start[1] - direction[1] * k
        if not _in_bounds(grid.shape, rr, cc):
            return None
        occupied.add((rr, cc))

    chain_len = random.choices([0, 1, 2, 3, 4], weights=[1, 2, 3, 3, 2], k=1)[0]
    collision_terminal = random.random() < 0.45

    frames: list[dict] = []
    current = start
    current_dir = direction
    for _ in range(chain_len):
        relay = _try_place_relay(grid.shape, current, current_dir, frames, occupied, ray_cells)
        if relay is None:
            return None
        frames.append(relay["frame"])
        ray_cells.update(relay["segment"])
        ray_cells.add(relay["exit_gap"])
        occupied.update(_iter_bbox_cells(relay["frame"]["bbox"]))
        current = relay["exit_gap"]
        current_dir = relay["exit_dir"]

    if collision_terminal:
        blocker = _try_place_blocker(grid.shape, current, current_dir, frames, occupied, ray_cells)
        if blocker is None:
            return None
        frames.append(blocker["frame"])
        ray_cells.update(blocker["segment"])
        occupied.update(_iter_bbox_cells(blocker["frame"]["bbox"]))
    else:
        tail = _final_edge_segment(grid.shape, current, current_dir, occupied, ray_cells)
        if tail is None:
            return None
        ray_cells.update(tail)

    min_extra = max(0, 3 - len(frames))
    max_extra = max(min_extra, min(6 - len(frames), 2))
    extra_count = random.randint(min_extra, max_extra) if max_extra >= min_extra else 0
    for _ in range(extra_count):
        distractor = _try_place_distractor(grid.shape, frames, occupied, ray_cells)
        if distractor is None:
            return None
        frames.append(distractor)
        occupied.update(_iter_bbox_cells(distractor["bbox"]))

    grid[start] = YELLOW
    for k in range(1, lead_len + 1):
        rr = start[0] - direction[0] * k
        cc = start[1] - direction[1] * k
        grid[rr, cc] = RED
    for frame in frames:
        _draw_frame(grid, frame)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if len(_find_frames(grid)) != len(frames):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid doorway-relay frame puzzle")
