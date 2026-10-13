from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, components_mask, find_components, in_bounds


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
CYCLE = (Color.GREEN, Color.YELLOW, Color.ORANGE)


def _turn_left(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return -dc, dr


def _turn_right(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return dc, -dr


def _rect_mask(shape: tuple[int, int], r: int, c: int, h: int, w: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[r : r + h, c : c + w] = True
    return mask


def _expanded_rect_mask(shape: tuple[int, int], r: int, c: int, h: int, w: int) -> np.ndarray:
    """Return the obstacle plus its cross-shaped one-cell extrusion."""
    H, W = shape
    mask = _rect_mask(shape, r, c, h, w)
    if r - 1 >= 0:
        mask[r - 1, c : c + w] = True
    if r + h < H:
        mask[r + h, c : c + w] = True
    if c - 1 >= 0:
        mask[r : r + h, c - 1] = True
    if c + w < W:
        mask[r : r + h, c + w] = True
    return mask


def _extract_obstacles(grid: np.ndarray, *, expand: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    mask = np.zeros((H, W), dtype=bool)
    obstacles = find_components(
        g,
        bg=Color.BLACK,
        colors=[Color.RED],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    for comp in obstacles:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if expand:
            mask |= _expanded_rect_mask((H, W), r0, c0, h, w)
        else:
            mask |= _rect_mask((H, W), r0, c0, h, w)
    return mask


def _initial_direction(grid: np.ndarray, start: tuple[int, int]) -> tuple[int, int]:
    r, c = start
    options = []
    for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        nr, nc = r + dr, c + dc
        if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == Color.BLUE:
            options.append((dr, dc))
    assert len(options) == 1, "expected a unique channel-facing direction from the start marker"
    return options[0]


def _simulate_particle(
    canvas: np.ndarray,
    traversable: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    end_mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    out = np.asarray(canvas, dtype=int).copy()
    H, W = out.shape

    r, c = map(int, start)
    dr, dc = map(int, direction)
    color_idx = 0
    collisions = 0
    out[r, c] = int(CYCLE[color_idx])

    visited_states: set[tuple[int, int, int, int, int, int]] = set()
    path = [(r, c)]
    stop_reason = "stuck"
    reached_end = False

    while True:
        state = (r, c, dr, dc, color_idx, collisions % 2)
        if state in visited_states:
            stop_reason = "loop"
            break
        visited_states.add(state)

        nr, nc = r + dr, c + dc
        if in_bounds((H, W), nr, nc) and bool(end_mask[nr, nc]):
            out[nr, nc] = Color.MAROON
            path.append((nr, nc))
            reached_end = True
            stop_reason = "end"
            break

        blocked = (not in_bounds((H, W), nr, nc)) or (not bool(traversable[nr, nc]))
        if not blocked:
            color_idx = (color_idx + 1) % len(CYCLE)
            r, c = nr, nc
            out[r, c] = int(CYCLE[color_idx])
            path.append((r, c))
            continue

        out[r, c] = Color.MAROON
        collisions += 1
        next_dir = _turn_left((dr, dc)) if collisions % 2 == 1 else _turn_right((dr, dc))
        tr, tc = r + next_dir[0], c + next_dir[1]
        turn_blocked = (not in_bounds((H, W), tr, tc)) or (
            not bool(traversable[tr, tc]) and not bool(end_mask[tr, tc])
        )
        if turn_blocked:
            stop_reason = "stuck"
            break
        dr, dc = next_dir

    meta = {
        "path": path,
        "path_set": set(path),
        "collisions": collisions,
        "reached_end": reached_end,
        "stop_reason": stop_reason,
    }
    return out, meta


def _solve(grid: np.ndarray, *, expand_obstacles: bool, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30

    start_cells = np.argwhere(g == Color.GREEN)
    assert start_cells.shape[0] == 1, "expected exactly one start marker"
    start = tuple(map(int, start_cells[0]))
    direction = _initial_direction(g, start)

    channel_mask = g == Color.BLUE
    obstacle_mask = _extract_obstacles(g, expand=expand_obstacles)
    traversable = channel_mask & ~obstacle_mask
    end_mask = g == Color.YELLOW

    out = g.copy()
    out[obstacle_mask] = Color.RED

    simulated, meta = _simulate_particle(out, traversable, start, direction, end_mask)
    if return_meta:
        meta = dict(meta)
        meta["traversable"] = traversable
        meta["obstacle_mask"] = obstacle_mask
        return simulated, meta
    return simulated


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extrude each red obstacle into a cross-shaped shell, then trace an alternating-turn particle through the narrowed blue channel.

    Input structure:
    - Black background surrounds one blue winding channel that runs from the left side toward the right side.
    - Small red rectangular obstacles sit inside the channel without touching the channel wall.
    - A single green start pixel sits just outside the left entrance, and a yellow 2x2 end block sits at the right exit.

    Rule:
    - Expand every red obstacle by one cell on each of its four sides, but do not fill the four diagonal corner cells of that added ring.
    - Treat the remaining blue cells as traversable channel; launch a particle from the green pixel toward its unique adjacent blue cell.
    - The particle paints visited cells with the cycle green, yellow, orange, starting on the start pixel itself; when the next step would hit black, red, or the grid edge, mark the current cell maroon and alternate 90-degree turns: odd collisions turn left, even collisions turn right.
    - If the turned direction is also blocked, the particle stops; if the next step enters the yellow end block, stop there and paint that reached end cell maroon while leaving the other end cells yellow.

    Key insight:
    - The red extrusion is not decorative: it changes where collisions happen, and those shifted collision points change the entire alternating-turn trajectory.
    """
    return _solve(grid, expand_obstacles=True, return_meta=False)


def _build_layout() -> dict | None:
    width = random.randint(4, 5)
    if width == 5:
        n_rises = 2
    else:
        n_rises = random.choice([2, 2, 3])

    rise_gaps = [random.randint(width + 1, width + 3) for _ in range(n_rises)]
    run_gaps = [random.randint(4, 7) for _ in range(n_rises)]
    final_run = random.randint(5, 8)

    top_row_last = random.randint(1, 3)
    r_tops = [0] * (n_rises + 1)
    r_tops[-1] = top_row_last
    for i in range(n_rises - 1, -1, -1):
        r_tops[i] = r_tops[i + 1] + rise_gaps[i]

    H = r_tops[0] + width + random.randint(1, 3)
    W = 1 + sum(run_gaps) + final_run + width + 2
    if not (18 <= H <= 30 and 18 <= W <= 30):
        return None

    channel = np.zeros((H, W), dtype=bool)
    h_segments = []
    v_segments = []

    c_start = 1
    final_right = None
    for i, r_top in enumerate(r_tops):
        if i < n_rises:
            turn_col = c_start + run_gaps[i]
            c_end = turn_col + width - 1
        else:
            turn_col = None
            c_end = c_start + final_run + width - 1
            final_right = c_end

        if c_end >= W - 2:
            return None

        channel[r_top : r_top + width, c_start : c_end + 1] = True
        h_segments.append({"r_top": r_top, "c0": c_start, "c1": c_end})

        if i < n_rises:
            next_r_top = r_tops[i + 1]
            channel[next_r_top : r_top + width, turn_col : turn_col + width] = True
            v_segments.append(
                {"r0": next_r_top, "r1": r_top + width - 1, "c0": turn_col, "c1": turn_col + width - 1}
            )
            c_start = turn_col

    assert final_right is not None
    start = (r_tops[0] + width - 3, 0)
    return {
        "shape": (H, W),
        "width": width,
        "channel": channel,
        "h_segments": h_segments,
        "v_segments": v_segments,
        "start": start,
        "final_right": final_right,
        "r_tops": r_tops,
    }


def _can_place_obstacle(
    channel: np.ndarray,
    occupied: np.ndarray,
    expanded_taken: np.ndarray,
    r: int,
    c: int,
    h: int,
    w: int,
) -> bool:
    H, W = channel.shape
    if r < 1 or c < 1 or r + h >= H or c + w >= W:
        return False
    if not bool(channel[r : r + h, c : c + w].all()):
        return False
    if not bool(channel[r - 1, c : c + w].all()):
        return False
    if not bool(channel[r + h, c : c + w].all()):
        return False
    if not bool(channel[r : r + h, c - 1].all()):
        return False
    if not bool(channel[r : r + h, c + w].all()):
        return False

    rect = _rect_mask(channel.shape, r, c, h, w)
    expanded = _expanded_rect_mask(channel.shape, r, c, h, w)
    if bool((rect & occupied).any()):
        return False
    if bool((expanded & expanded_taken).any()):
        return False
    if len(components_mask(channel & ~(expanded_taken | expanded), connectivity=4)) != 1:
        return False
    return True


def _place_obstacle(
    occupied: np.ndarray,
    expanded_taken: np.ndarray,
    r: int,
    c: int,
    h: int,
    w: int,
) -> tuple[np.ndarray, np.ndarray]:
    rect = _rect_mask(occupied.shape, r, c, h, w)
    expanded = _expanded_rect_mask(occupied.shape, r, c, h, w)
    occupied = occupied.copy()
    expanded_taken = expanded_taken.copy()
    occupied |= rect
    expanded_taken |= expanded
    return occupied, expanded_taken


def _try_guide_obstacle(layout: dict, occupied: np.ndarray, expanded_taken: np.ndarray):
    width = int(layout["width"])
    seg = layout["h_segments"][0]
    row = int(seg["r_top"] + width - 2)
    candidates = []
    for ow in random.sample([1, 2], k=2):
        c_min = int(seg["c0"] + 3)
        c_max = int(seg["c1"] - ow - 2)
        for c in range(c_min, c_max + 1):
            if _can_place_obstacle(layout["channel"], occupied, expanded_taken, row, c, 1, ow):
                candidates.append((row, c, 1, ow))
    if not candidates:
        return None
    return random.choice(candidates)


def _try_stopper_obstacle(layout: dict, occupied: np.ndarray, expanded_taken: np.ndarray):
    if len(layout["h_segments"]) < 2:
        return None
    seg = layout["h_segments"][1]
    row = int(seg["r_top"] + 1)
    candidates = []
    for ow in random.sample([1, 2], k=2):
        c_min = int(seg["c0"] + 3)
        c_max = int(seg["c1"] - ow - 3)
        for c in range(c_min, c_max + 1):
            if _can_place_obstacle(layout["channel"], occupied, expanded_taken, row, c, 1, ow):
                candidates.append((row, c, 1, ow))
    if not candidates:
        return None
    return random.choice(candidates)


def _filler_candidates(layout: dict, occupied: np.ndarray, expanded_taken: np.ndarray) -> list[tuple[int, int, int, int]]:
    width = int(layout["width"])
    channel = layout["channel"]
    candidates: list[tuple[int, int, int, int]] = []
    size_pool = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for idx, seg in enumerate(layout["h_segments"]):
        if idx == 0:
            continue
        for h, w in size_pool:
            row = int(seg["r_top"] + width - h - 1)
            if row <= seg["r_top"]:
                continue
            c_min = int(seg["c0"] + 2)
            c_max = int(seg["c1"] - w - 2)
            for c in range(c_min, c_max + 1):
                if _can_place_obstacle(channel, occupied, expanded_taken, row, c, h, w):
                    candidates.append((row, c, h, w))
    return candidates


def _render_input(layout: dict, obstacles: list[tuple[int, int, int, int]], want_reach: bool) -> np.ndarray:
    H, W = map(int, layout["shape"])
    g = np.zeros((H, W), dtype=int)
    g[layout["channel"]] = Color.BLUE

    for r, c, h, w in obstacles:
        g[r : r + h, c : c + w] = Color.RED

    sr, sc = map(int, layout["start"])
    g[sr, sc] = Color.GREEN

    end_c = int(layout["final_right"] + 1)
    if want_reach:
        end_r = int(layout["r_tops"][-1])
    else:
        end_r = int(layout["r_tops"][-1] + layout["width"] - 2)
    g[end_r : end_r + 2, end_c : end_c + 2] = Color.YELLOW
    return g


def _traversable_connected(mask: np.ndarray) -> bool:
    comps = components_mask(mask, connectivity=4)
    return len(comps) == 1


def _base_properties_ok(grid: np.ndarray) -> bool:
    H, W = grid.shape
    if H > 30 or W > 30:
        return False
    if np.count_nonzero(grid == Color.GREEN) != 1:
        return False
    if np.count_nonzero(grid == Color.YELLOW) != 4:
        return False
    start = tuple(map(int, np.argwhere(grid == Color.GREEN)[0]))
    neighbors = 0
    for dr, dc in DIR4:
        nr, nc = start[0] + dr, start[1] + dc
        if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == Color.BLUE:
            neighbors += 1
    if neighbors != 1:
        return False
    red = find_components(
        grid,
        bg=Color.BLACK,
        colors=[Color.RED],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    if not (2 <= len(red) <= 4):
        return False
    for comp in red:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not bool((patch == Color.RED).all()):
            return False
        if (r1 - r0 + 1) > 2 or (c1 - c0 + 1) > 2:
            return False
    return True


def _gen_attempt(want_reach: bool) -> np.ndarray | None:
    layout = _build_layout()
    if layout is None:
        return None

    occupied = np.zeros(layout["shape"], dtype=bool)
    expanded_taken = np.zeros(layout["shape"], dtype=bool)
    obstacles: list[tuple[int, int, int, int]] = []

    guide = _try_guide_obstacle(layout, occupied, expanded_taken)
    if guide is None:
        return None
    occupied, expanded_taken = _place_obstacle(occupied, expanded_taken, *guide)
    obstacles.append(guide)

    if not want_reach:
        stopper = _try_stopper_obstacle(layout, occupied, expanded_taken)
        if stopper is None:
            return None
        occupied, expanded_taken = _place_obstacle(occupied, expanded_taken, *stopper)
        obstacles.append(stopper)

    target_count = random.randint(2, 4)
    for _ in range(target_count - len(obstacles)):
        filler = _filler_candidates(layout, occupied, expanded_taken)
        if not filler:
            break
        choice = random.choice(filler)
        occupied, expanded_taken = _place_obstacle(occupied, expanded_taken, *choice)
        obstacles.append(choice)

    if not (2 <= len(obstacles) <= 4):
        return None

    grid = _render_input(layout, obstacles, want_reach)
    if not _base_properties_ok(grid):
        return None

    try:
        expanded_out, expanded_meta = _solve(grid, expand_obstacles=True, return_meta=True)
        base_out, base_meta = _solve(grid, expand_obstacles=False, return_meta=True)
    except Exception:
        return None

    if not _traversable_connected(expanded_meta["traversable"]):
        return None
    if expanded_meta["stop_reason"] == "loop" or base_meta["stop_reason"] == "loop":
        return None
    if expanded_meta["collisions"] < 2:
        return None
    if len(expanded_meta["path"]) < 8:
        return None
    if want_reach != bool(expanded_meta["reached_end"]):
        return None
    if not want_reach and expanded_meta["stop_reason"] != "stuck":
        return None
    if np.array_equal(expanded_out, base_out):
        return None
    if expanded_meta["path"] == base_meta["path"]:
        return None

    return grid


def generate_input() -> np.ndarray:
    want_reach = random.random() < 0.55
    for _ in range(5000):
        grid = _gen_attempt(want_reach)
        if grid is not None:
            return grid
    # Fallback: switch branch so both terminal behaviors remain generatable.
    for _ in range(5000):
        grid = _gen_attempt(not want_reach)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
