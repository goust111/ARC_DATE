from __future__ import annotations

import random
from collections import deque
from typing import Any

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
PALETTE = [BLUE, RED, GREEN, YELLOW]

DIRECTIONS = [
    ("N", -1, 0),
    ("NE", -1, 1),
    ("E", 0, 1),
    ("SE", 1, 1),
    ("S", 1, 0),
    ("SW", 1, -1),
    ("W", 0, -1),
    ("NW", -1, -1),
]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_in_bounds(shape: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return 0 <= r0 <= r1 < shape[0] and 0 <= c0 <= c1 < shape[1]


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _frame_border_cells(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    for r, c in _frame_border_cells(box):
        grid[r, c] = GRAY


def _gray_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(grid.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(grid == GRAY):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q = deque([(r, c)])
        coords: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            coords.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == GRAY:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(coords)
    return comps


def _find_block_center(grid: np.ndarray) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    candidates: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            block = g[r - 1 : r + 2, c - 1 : c + 2]
            if int(block[1, 1]) != GRAY:
                continue
            surround = [int(block[rr, cc]) for rr in range(3) for cc in range(3) if (rr, cc) != (1, 1)]
            if all(color in PALETTE for color in surround):
                candidates.append((r, c))
    assert len(candidates) == 1, f"expected exactly one 3x3 seed block, got {candidates}"
    return candidates[0]


def _find_frames(grid: np.ndarray, center: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _gray_components(grid):
        if comp == [center]:
            continue
        box = _bbox_from_coords(comp)
        r0, c0, r1, c1 = box
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 4 or w < 4:
            continue
        border = set(_frame_border_cells(box))
        coords = set(comp)
        if coords != border:
            continue
        if not np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == BLACK):
            continue
        frames.append(box)
    assert 3 <= len(frames) <= 5, f"expected 3-5 frames, got {frames}"
    return sorted(frames)


def _trace_layout(
    shape: tuple[int, int],
    center: tuple[int, int],
    frames: list[tuple[int, int, int, int]],
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    border_owner: dict[tuple[int, int], int] = {}
    for idx, box in enumerate(frames):
        for cell in _frame_border_cells(box):
            assert cell not in border_owner
            border_owner[cell] = idx

    rays: list[dict[str, Any]] = []
    hits_by_frame: list[list[dict[str, Any]]] = [[] for _ in frames]
    cr, cc = center
    for dir_idx, (name, dr, dc) in enumerate(DIRECTIONS):
        sr, sc = cr + dr, cc + dc
        assert _in_bounds(shape, sr, sc)
        path = [(sr, sc)]
        r, c = sr, sc
        hit_frame = None
        while True:
            nr = r + dr
            nc = c + dc
            if not _in_bounds(shape, nr, nc):
                break
            owner = border_owner.get((nr, nc))
            if owner is not None:
                hit_frame = owner
                break
            r, c = nr, nc
            path.append((r, c))
        stop = (r, c)
        info = {
            "dir_idx": dir_idx,
            "name": name,
            "dr": dr,
            "dc": dc,
            "start": (sr, sc),
            "path": path,
            "stop": stop,
            "dist": abs(stop[0] - sr) + abs(stop[1] - sc),
            "hit_frame": hit_frame,
        }
        rays.append(info)
        if hit_frame is not None:
            hits_by_frame[hit_frame].append(info)
    return rays, hits_by_frame


def _winner(hit_infos: list[dict[str, Any]]) -> dict[str, Any]:
    assert hit_infos
    return min(hit_infos, key=lambda info: (int(info["dist"]), int(info["dir_idx"])))


def _primary_frame(
    center: tuple[int, int], orient: str, want_tie: bool, rng: random.Random
) -> tuple[int, int, int, int]:
    cr, cc = center
    if want_tie:
        depth = rng.randint(4, 6)
        span = rng.randint(4, 6)
        if orient == "N":
            return cr - 2 - depth + 1, cc - 2, cr - 2, cc + 2
        if orient == "S":
            return cr + 2, cc - 2, cr + 2 + depth - 1, cc + 2
        if orient == "E":
            return cr - 2, cc + 2, cr + 2, cc + 2 + span - 1
        return cr - 2, cc - 2 - span + 1, cr + 2, cc - 2

    extra = rng.randint(1, 2)
    reach = 2 + extra
    long_side = rng.randint(4, 6)
    if orient == "N":
        r1 = cr - 2 - extra
        if rng.random() < 0.5:
            return r1 - long_side + 1, cc, r1, cc + reach
        return r1 - long_side + 1, cc - reach, r1, cc
    if orient == "S":
        r0 = cr + 2 + extra
        if rng.random() < 0.5:
            return r0, cc, r0 + long_side - 1, cc + reach
        return r0, cc - reach, r0 + long_side - 1, cc
    if orient == "E":
        c0 = cc + 2 + extra
        if rng.random() < 0.5:
            return cr - reach, c0, cr, c0 + long_side - 1
        return cr, c0, cr + reach, c0 + long_side - 1
    c1 = cc - 2 - extra
    if rng.random() < 0.5:
        return cr - reach, c1 - long_side + 1, cr, c1
    return cr, c1 - long_side + 1, cr + reach, c1


def _sample_box(shape: tuple[int, int], rng: random.Random) -> tuple[int, int, int, int]:
    H, W = shape
    h = rng.randint(4, 6)
    w = rng.randint(4, 6)
    r0 = rng.randint(0, H - h)
    c0 = rng.randint(0, W - w)
    return r0, c0, r0 + h - 1, c0 + w - 1


def _find_candidate_frame(
    shape: tuple[int, int],
    center: tuple[int, int],
    current_frames: list[tuple[int, int, int, int]],
    block_box: tuple[int, int, int, int],
    desired_class: str,
    rng: random.Random,
    want_tie: bool,
    forbid_winner_dir: int | None = None,
) -> tuple[int, int, int, int] | None:
    for _ in range(400):
        box = _sample_box(shape, rng)
        if not _boxes_separated(block_box, box, pad=0):
            continue
        if any(not _boxes_separated(prev, box, pad=1) for prev in current_frames):
            continue

        solo_hits = _trace_layout(shape, center, [box])[1][0]
        full_frames = current_frames + [box]
        full_hits = _trace_layout(shape, center, full_frames)[1]
        actual_hits = full_hits[-1]

        if desired_class == "offpath":
            if solo_hits:
                continue
        elif desired_class == "shadowed":
            if not solo_hits or actual_hits:
                continue
        elif desired_class == "visible_hit":
            if not actual_hits:
                continue
            if forbid_winner_dir is not None and _winner(actual_hits)["dir_idx"] == forbid_winner_dir:
                continue
        else:
            raise ValueError(desired_class)

        primary_hits = full_hits[0]
        if len(primary_hits) < 2:
            continue
        best_dist = min(int(info["dist"]) for info in primary_hits)
        tie_count = sum(int(info["dist"]) == best_dist for info in primary_hits)
        if want_tie and tie_count < 2:
            continue
        if not want_tie and tie_count != 1:
            continue
        return box
    return None


def _assign_seed_colors(
    hit_map: list[list[dict[str, Any]]], rng: random.Random
) -> list[int]:
    colors = [rng.choice(PALETTE) for _ in range(8)]
    winner_dirs: list[int] = []
    for hits in hit_map:
        if hits:
            d = int(_winner(hits)["dir_idx"])
            if d not in winner_dirs:
                winner_dirs.append(d)
    shuffled = PALETTE[:]
    rng.shuffle(shuffled)
    for i, d in enumerate(winner_dirs[: len(shuffled)]):
        colors[d] = shuffled[i]
    if len(set(colors)) == 1:
        colors[rng.randrange(8)] = shuffled[0]
        colors[(rng.randrange(7) + 1) % 8] = shuffled[1]
    return colors


def _render_grid(
    shape: tuple[int, int],
    center: tuple[int, int],
    frames: list[tuple[int, int, int, int]],
    seed_colors: list[int],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    cr, cc = center
    grid[cr, cc] = GRAY
    for idx, (_name, dr, dc) in enumerate(DIRECTIONS):
        grid[cr + dr, cc + dc] = int(seed_colors[idx])
    for box in frames:
        _draw_frame(grid, box)
    return grid


def _validate_generated_scene(grid: np.ndarray, want_tie: bool, require_shadow: bool) -> bool:
    center = _find_block_center(grid)
    frames = _find_frames(grid, center)
    rays, hits = _trace_layout(grid.shape, center, frames)

    if sum(bool(h) for h in hits) < 2:
        return False
    if not any(len(h) >= 2 for h in hits):
        return False

    path_flags = []
    shadowed = False
    for idx, box in enumerate(frames):
        solo = _trace_layout(grid.shape, center, [box])[1][0]
        on_path = bool(solo)
        path_flags.append(on_path)
        if on_path and not hits[idx]:
            shadowed = True
    if sum(path_flags) < 2:
        return False
    if not any(not flag for flag in path_flags):
        return False
    if require_shadow and not shadowed:
        return False

    multi_hit_frames = [frame_hits for frame_hits in hits if len(frame_hits) >= 2]
    if not multi_hit_frames:
        return False
    tie_counts = []
    for frame_hits in multi_hit_frames:
        best_dist = min(int(info["dist"]) for info in frame_hits)
        tie_counts.append(sum(int(info["dist"]) == best_dist for info in frame_hits))
    if want_tie and max(tie_counts) < 2:
        return False
    if not want_tie and min(tie_counts) != 1 and all(count != 1 for count in tie_counts):
        return False

    winner_dirs = {int(_winner(frame_hits)["dir_idx"]) for frame_hits in hits if frame_hits}
    if len(winner_dirs) < 2:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    filled_pixels = 0
    for idx, box in enumerate(frames):
        r0, c0, r1, c1 = box
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        if hits[idx]:
            if np.any(interior == BLACK):
                return False
            filled_pixels += int(np.count_nonzero(interior))
        else:
            if not np.all(interior == BLACK):
                return False
    return filled_pixels > 0 and any(len(info["path"]) > 2 for info in rays)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the 3x3 block's eight rays until they hit gray frames, then fill each hit frame with the nearest ray color.

    Input structure:
    - A black background contains one isolated solid 3x3 color block and 3-5 separate gray hollow rectangles.
    - The 3x3 block has a gray center and eight non-black surround cells whose relative positions define the compass directions.
    - Each gray component other than the block center is a one-pixel-thick rectangular frame with a black interior.

    Rule:
    - From each non-center block cell, cast a straight ray in its compass direction using that seed color.
    - Paint the ray on black cells, but if the next cell would be a gray frame border then stop on the current cell and mark that frame as hit.
    - For every hit frame, compare all hitting rays by start-to-stop Manhattan distance; fill the frame interior with the shortest ray's color, breaking ties by clockwise direction order N, NE, E, SE, S, SW, W, NW.

    Key insight:
    - The gray rectangles act both as ray stoppers and as deferred fill targets, so the winning fill color is determined by the first ray geometry rather than by the frame itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    center = _find_block_center(g)
    frames = _find_frames(g, center)
    rays, hits_by_frame = _trace_layout(g.shape, center, frames)

    out = g.copy()
    for info in rays:
        color = int(g[info["start"]])
        for r, c in info["path"]:
            out[r, c] = color

    for frame_idx, hit_infos in enumerate(hits_by_frame):
        if not hit_infos:
            continue
        winner = _winner(hit_infos)
        color = int(g[winner["start"]])
        r0, c0, r1, c1 = frames[frame_idx]
        out[r0 + 1 : r1, c0 + 1 : c1] = color
    return out


def generate_input() -> np.ndarray:
    for _ in range(500):
        rng = random.Random()
        H = rng.randint(18, 22)
        W = rng.randint(18, 22)
        shape = (H, W)
        center = (rng.randint(6, H - 7), rng.randint(6, W - 7))
        block_box = (center[0] - 1, center[1] - 1, center[0] + 1, center[1] + 1)

        want_tie = rng.random() < 0.6
        require_shadow = rng.random() < 0.7
        frame_count = rng.randint(4 if require_shadow else 3, 5)

        orient = rng.choice(["N", "E", "S", "W"])
        primary = _primary_frame(center, orient, want_tie, rng)
        if not _bbox_in_bounds(shape, primary):
            continue
        if not _boxes_separated(block_box, primary, pad=0):
            continue
        frames = [primary]

        primary_hits = _trace_layout(shape, center, frames)[1][0]
        if len(primary_hits) < 2:
            continue
        primary_winner_dir = int(_winner(primary_hits)["dir_idx"])

        secondary = _find_candidate_frame(
            shape, center, frames, block_box, "visible_hit", rng, want_tie, forbid_winner_dir=primary_winner_dir
        )
        if secondary is None:
            continue
        frames.append(secondary)

        decoy = _find_candidate_frame(shape, center, frames, block_box, "offpath", rng, want_tie)
        if decoy is None:
            continue
        frames.append(decoy)

        if require_shadow:
            shadow = _find_candidate_frame(shape, center, frames, block_box, "shadowed", rng, want_tie)
            if shadow is None:
                continue
            frames.append(shadow)

        while len(frames) < frame_count:
            klass = "visible_hit" if rng.random() < 0.5 else "offpath"
            extra = _find_candidate_frame(shape, center, frames, block_box, klass, rng, want_tie)
            if extra is None:
                break
            frames.append(extra)
        if len(frames) != frame_count:
            continue

        _, hit_map = _trace_layout(shape, center, frames)
        seed_colors = _assign_seed_colors(hit_map, rng)
        grid = _render_grid(shape, center, frames, seed_colors)
        if _validate_generated_scene(grid, want_tie=want_tie, require_shadow=require_shadow):
            return grid

    raise RuntimeError("failed to generate a valid input")
