from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

DIR_UP = (-1, 0)
DIR_LEFT = (0, -1)
DIR_RIGHT = (0, 1)

ODD_INTERIOR_OPTIONS = ((3, 3), (3, 5), (5, 3), (5, 5))
EVEN_INTERIOR_OPTIONS = ((3, 4), (4, 3), (4, 4), (5, 4), (4, 5))
GEN_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _detect_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for comp in _components(grid == BLUE):
        r0, c0, r1, c1 = _bbox_from_coords(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = (
            {(r0, c) for c in range(c0, c1 + 1)}
            | {(r1, c) for c in range(c0, c1 + 1)}
            | {(r, c0) for r in range(r0, r1 + 1)}
            | {(r, c1) for r in range(r0, r1 + 1)}
        )
        if set(comp) != border:
            continue
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "area": (r1 - r0 - 1) * (c1 - c0 - 1),
            }
        )
    frames.sort(key=lambda frame: frame["bbox"])
    return frames


def _frame_index_for_interior_cell(frames: list[dict], r: int, c: int) -> int | None:
    for idx, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]
        if r0 < r < r1 and c0 < c < c1:
            return idx
    return None


def _solve(
    grid: np.ndarray,
    *,
    odd_turn_left: bool = True,
    even_turn_right: bool = True,
    previous_paths_block: bool = True,
    fill_touched_frames: bool = True,
    return_meta: bool = False,
):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    frames = _detect_frames(g)
    out = g.copy()
    starts = sorted(int(c) for c in np.where(g[-1] == RED)[0])

    previous_cells: set[tuple[int, int]] = set()
    frame_hits = [False] * len(frames)
    paths: list[list[tuple[int, int]]] = []

    for start_c in starts:
        r = h - 1
        c = start_c
        path = [(r, c)]

        while r > 0:
            nr = r + DIR_UP[0]
            nc = c + DIR_UP[1]
            blocked_up = g[nr, nc] == GREY or (previous_paths_block and (nr, nc) in previous_cells)
            if blocked_up:
                frame_idx = _frame_index_for_interior_cell(frames, r, c)
                dc = 1
                if frame_idx is not None:
                    area_is_odd = frames[frame_idx]["area"] % 2 == 1
                    if area_is_odd:
                        dc = -1 if odd_turn_left else 1
                    else:
                        dc = 1 if even_turn_right else -1
                tr, tc = r, c + dc
                if not _in_bounds(g.shape, tr, tc):
                    break
                if g[tr, tc] in (GREY, BLUE):
                    break
                if previous_paths_block and (tr, tc) in previous_cells:
                    break
                r, c = tr, tc
            else:
                r, c = nr, nc

            path.append((r, c))

        for rr, cc in path:
            out[rr, cc] = RED
            previous_cells.add((rr, cc))

        touched = set()
        for rr, cc in path:
            frame_idx = _frame_index_for_interior_cell(frames, rr, cc)
            if frame_idx is not None:
                touched.add(frame_idx)
        for frame_idx in touched:
            frame_hits[frame_idx] = True

        paths.append(path)

    if fill_touched_frames:
        for hit, frame in zip(frame_hits, frames):
            if not hit:
                continue
            r0, c0, r1, c1 = frame["bbox"]
            interior = out[r0 + 1 : r1, c0 + 1 : c1]
            interior[interior == BLACK] = YELLOW

    if return_meta:
        return out, {"frames": frames, "paths": paths, "frame_hits": frame_hits, "starts": starts}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow red paths upward through blue frames, using frame-area parity to choose in-frame detours, then fill traversed interiors yellow.

    Input structure:
    - Black background with 2-3 red start pixels on the bottom row.
    - Blue 1-pixel-thick hollow rectangles with grey single-cell obstacles both inside and outside them.
    - Rectangles do not overlap, and their interiors may be entered only by vertical path growth plus obstacle-driven sidesteps.

    Rule:
    - Process red starts from left to right, growing each path upward one cell at a time.
    - When the next upward cell is blocked by a grey obstacle or an earlier path, detour horizontally by one cell.
    - Outside every frame, detours go right; inside a frame interior, odd interior area turns left and even interior area turns right.
    - If a detour target is out of bounds, grey, or on a blue frame border, that path stops; after all paths are drawn, every touched frame fills its remaining black interior cells yellow.

    Key insight:
    - The blue rectangles are not just containers: their hidden interior area parity controls the in-frame detour direction, which changes which frames are actually traversed and therefore which interiors get filled.
    """
    return _solve(grid)


def _draw_frame(grid: np.ndarray, r0: int, c0: int, inner_h: int, inner_w: int) -> dict:
    h = inner_h + 2
    w = inner_w + 2
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0, c0 : c1 + 1] = BLUE
    grid[r1, c0 : c1 + 1] = BLUE
    grid[r0 : r1 + 1, c0] = BLUE
    grid[r0 : r1 + 1, c1] = BLUE
    return {"bbox": (r0, c0, r1, c1), "area": inner_h * inner_w}


def _place_shift_stair(grid: np.ndarray, start_col: int, shift_count: int) -> None:
    h = grid.shape[0]
    for offset in range(shift_count):
        grid[h - 2 - offset, start_col + offset] = GREY


def _legal_obstacle_cell(grid: np.ndarray, frames: list[dict], r: int, c: int) -> bool:
    if not _in_bounds(grid.shape, r, c) or grid[r, c] != BLACK:
        return False
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        if r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1:
            return False
    return True


def _sprinkle_distractors(grid: np.ndarray, frames: list[dict], occupied: set[tuple[int, int]], rng: random.Random) -> None:
    candidates: list[tuple[int, int]] = []
    h, w = grid.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if (r, c) in occupied:
                continue
            if _legal_obstacle_cell(grid, frames, r, c):
                candidates.append((r, c))
    rng.shuffle(candidates)
    for r, c in candidates[: rng.randint(0, 3)]:
        grid[r, c] = GREY


def _candidate_grid(rng: random.Random) -> np.ndarray | None:
    odd_inner_h, odd_inner_w = rng.choice(ODD_INTERIOR_OPTIONS)
    even_inner_h, even_inner_w = rng.choice(EVEN_INTERIOR_OPTIONS)

    odd_outer_w = odd_inner_w + 2
    even_outer_w = even_inner_w + 2
    odd_c0 = rng.randint(5, 9)
    frame_gap = rng.randint(4, 6)
    even_c0 = odd_c0 + odd_outer_w + frame_gap
    right_margin = rng.randint(2, 4)
    width = even_c0 + even_outer_w + right_margin
    if width > 30:
        return None

    odd_r0 = rng.randint(6, 10)
    even_r0 = rng.randint(4, 9)
    odd_outer_h = odd_inner_h + 2
    even_outer_h = even_inner_h + 2
    min_height = max(odd_r0 + odd_outer_h + 7, even_r0 + even_outer_h + 7, 18)
    max_height = min(30, min_height + 5)
    if min_height > max_height:
        return None
    height = rng.randint(min_height, max_height)

    odd_entry_c = odd_c0 + 2
    start_a = odd_c0 - 3
    start_b = odd_c0 - 1
    shift_c = rng.randint(3, 5)
    even_entry_c = even_c0 + 1
    start_c = even_entry_c - shift_c

    starts = [start_a, start_b, start_c]
    if min(starts) < 0 or len(set(starts)) != 3:
        return None
    if not (start_a < start_b < start_c):
        return None

    grid = np.zeros((height, width), dtype=int)
    frames = [
        _draw_frame(grid, odd_r0, odd_c0, odd_inner_h, odd_inner_w),
        _draw_frame(grid, even_r0, even_c0, even_inner_h, even_inner_w),
    ]

    for c in starts:
        if any(frame["bbox"][1] <= c <= frame["bbox"][3] for frame in frames):
            return None
        grid[-1, c] = RED

    if frames[0]["bbox"][2] > height - 7 or frames[1]["bbox"][2] > height - 5:
        return None

    _place_shift_stair(grid, start_a, odd_entry_c - start_a)
    _place_shift_stair(grid, start_c, shift_c)

    odd_r0_bbox, odd_c0_bbox, odd_r1_bbox, _ = frames[0]["bbox"]
    even_r0_bbox, even_c0_bbox, even_r1_bbox, _ = frames[1]["bbox"]
    odd_obs_r = rng.randint(odd_r0_bbox + 1, odd_r1_bbox - 2)
    even_obs_r = rng.randint(even_r0_bbox + 1, even_r1_bbox - 2)
    grid[odd_obs_r, odd_entry_c] = GREY
    grid[even_obs_r, even_entry_c] = GREY

    occupied = {(height - 2 - offset, start_a + offset) for offset in range(odd_entry_c - start_a)}
    occupied |= {(height - 2 - offset, start_c + offset) for offset in range(shift_c)}
    occupied |= {(odd_obs_r, odd_entry_c), (even_obs_r, even_entry_c)}
    _sprinkle_distractors(grid, frames, occupied, rng)

    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    correct, meta = _solve(grid, return_meta=True)
    frames = meta["frames"]
    if len(frames) != 2:
        return False
    if not all(meta["frame_hits"]):
        return False

    if np.array_equal(correct, grid):
        return False

    no_fill = _solve(grid, fill_touched_frames=False)
    if np.array_equal(correct, no_fill):
        return False

    no_order = _solve(grid, previous_paths_block=False)
    if np.array_equal(correct, no_order):
        return False

    always_right = _solve(grid, odd_turn_left=False, even_turn_right=True)
    if np.array_equal(correct, always_right):
        return False

    always_left = _solve(grid, odd_turn_left=True, even_turn_right=False)
    if np.array_equal(correct, always_left):
        return False

    for frame, path in zip(frames, (meta["paths"][0], meta["paths"][-1])):
        r0, c0, r1, c1 = frame["bbox"]
        interior_cols = {c for r, c in path if r0 < r < r1 and c0 < c < c1}
        if len(interior_cols) < 2:
            return False

    yellow_count = int(np.count_nonzero(correct == YELLOW))
    if yellow_count < 2:
        return False

    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _candidate_grid(rng)
        if grid is None:
            continue
        if not _semantic_checks(grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 1")
