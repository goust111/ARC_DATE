from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg


BG = Color.BLACK
OBSTACLE = Color.GREY
TRACK_POOL = tuple(c for c in Color.NOT_BLACK if c != OBSTACLE)
GEN_MAX_ATTEMPTS = 800
PLAY_H_RANGE = (8, 17)
PLAY_W_RANGE = (10, 18)
OBSTACLE_H_OPTIONS = (1, 2, 2, 3)
OBSTACLE_W_OPTIONS = (1, 2, 2, 3, 4)


def _is_corner(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return int(r) in {0, h - 1} and int(c) in {0, w - 1}


def _start_direction(shape: tuple[int, int], start: tuple[int, int]) -> tuple[int, int]:
    h, w = map(int, shape)
    r, c = map(int, start)
    if (r, c) == (0, 0):
        return 1, 1
    if (r, c) == (0, w - 1):
        return 1, -1
    if (r, c) == (h - 1, 0):
        return -1, 1
    if (r, c) == (h - 1, w - 1):
        return -1, -1
    raise ValueError("start marker must be at a playfield corner")


def _legend_colors_from_row(row: np.ndarray, bg: int, divider: int) -> list[int]:
    colors: list[int] = []
    prev = None
    for value in row:
        color = int(value)
        if color in {int(bg), int(divider)}:
            prev = None
            continue
        if color != prev:
            colors.append(color)
        prev = color
    return colors


def _layout_candidate(
    grid: np.ndarray,
    bg: int,
    divider_r: int,
    legend_r: int,
    play_r0: int,
    play_r1: int,
) -> dict | None:
    h, _ = map(int, grid.shape)
    if not (0 <= divider_r < h and 0 <= legend_r < h and 0 <= play_r0 < play_r1 <= h):
        return None

    divider_row = grid[divider_r]
    divider = int(divider_row[0])
    if divider == int(bg) or not np.all(divider_row == divider):
        return None

    legend = _legend_colors_from_row(grid[legend_r], bg, divider)
    if len(legend) < 2:
        return None

    playfield = grid[play_r0:play_r1].copy()
    marker_coords = np.argwhere((playfield != int(bg)) & (playfield != int(OBSTACLE)))
    if marker_coords.shape[0] != 1:
        return None
    start = tuple(map(int, marker_coords[0]))
    if not _is_corner(playfield.shape, *start):
        return None

    return {
        "divider": divider,
        "legend": legend,
        "playfield": playfield,
        "play_r0": play_r0,
        "start": start,
    }


def _parse_layout(grid: np.ndarray, bg: int) -> dict:
    h, _ = map(int, grid.shape)
    candidates = [
        _layout_candidate(grid, bg, 1, 0, 2, h),
        _layout_candidate(grid, bg, h - 2, h - 1, 0, h - 2),
    ]
    valid = [cand for cand in candidates if cand is not None]
    if len(valid) != 1:
        raise ValueError("failed to identify the divider/legend layout")
    return valid[0]


def _obstacle_data(playfield: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bg = int(get_bg(playfield))
    components = find_components(
        playfield,
        bg=bg,
        colors=[OBSTACLE],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    obstacle_mask = np.zeros(playfield.shape, dtype=bool)
    obstacle_ids = np.full(playfield.shape, -1, dtype=int)

    for idx, comp in enumerate(components):
        patch = crop_bbox(playfield, comp["bbox"])
        if not np.all(patch == int(OBSTACLE)):
            raise ValueError("grey obstacles must be solid rectangles")
        coords = comp["coords"]
        obstacle_mask[coords[:, 0], coords[:, 1]] = True
        obstacle_ids[coords[:, 0], coords[:, 1]] = idx

    if not np.array_equal(obstacle_mask, playfield == int(OBSTACLE)):
        raise ValueError("unexpected non-rectangular obstacle geometry")
    return obstacle_mask, obstacle_ids


def _blocked(mask: np.ndarray, r: int, c: int) -> bool:
    h, w = map(int, mask.shape)
    return not (0 <= int(r) < h and 0 <= int(c) < w) or bool(mask[int(r), int(c)])


def _reflect_from_wall(shape: tuple[int, int], r: int, c: int, dr: int, dc: int) -> tuple[int, int]:
    h, w = map(int, shape)
    nr = int(r + dr)
    nc = int(c + dc)
    if not (0 <= nr < h):
        dr = -dr
    if not (0 <= nc < w):
        dc = -dc
    return int(dr), int(dc)


def _reflect_from_obstacle(
    obstacle_mask: np.ndarray,
    r: int,
    c: int,
    dr: int,
    dc: int,
) -> tuple[int, int, str]:
    block_r = _blocked(obstacle_mask, r + dr, c)
    block_c = _blocked(obstacle_mask, r, c + dc)
    if block_r and not block_c:
        return -int(dr), int(dc), "flip_r"
    if block_c and not block_r:
        return int(dr), -int(dc), "flip_c"
    return -int(dr), -int(dc), "flip_both"


def _trace_path(
    playfield: np.ndarray,
    start: tuple[int, int],
    legend_colors: list[int],
) -> dict:
    obstacle_mask, obstacle_ids = _obstacle_data(playfield)
    h, w = map(int, playfield.shape)
    r, c = map(int, start)
    dr, dc = _start_direction((h, w), (r, c))
    color_index = 0
    path: list[tuple[int, int, int]] = []
    wall_hits = 0
    obstacle_hits = 0
    obstacle_kinds: list[str] = []
    distinct_obstacles: set[int] = set()
    seen_states = {(r, c, dr, dc)}

    for _ in range(8 * h * w):
        nr = r + dr
        nc = c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            dr, dc = _reflect_from_wall((h, w), r, c, dr, dc)
            wall_hits += 1
        elif obstacle_mask[nr, nc]:
            dr, dc, kind = _reflect_from_obstacle(obstacle_mask, r, c, dr, dc)
            obstacle_hits += 1
            obstacle_kinds.append(kind)
            distinct_obstacles.add(int(obstacle_ids[nr, nc]))
            color_index += 1
            if color_index >= len(legend_colors):
                raise ValueError("legend exhausted before the trajectory terminated")
        else:
            r, c = int(nr), int(nc)
            path.append((r, c, int(legend_colors[color_index])))
            if _is_corner((h, w), r, c):
                return {
                    "path": path,
                    "wall_hits": wall_hits,
                    "obstacle_hits": obstacle_hits,
                    "obstacle_kinds": obstacle_kinds,
                    "distinct_obstacles": distinct_obstacles,
                    "end": (r, c),
                }

        state = (r, c, dr, dc)
        if state in seen_states:
            raise ValueError("trajectory looped before reaching a corner")
        seen_states.add(state)

    raise ValueError("trajectory exceeded the step budget")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the 45-degree ball path and recolor each post-obstacle segment with the next legend color.

    Input structure:
    - The grid contains one large playfield plus a single legend row at the top or bottom, separated by a full-width divider row.
    - Inside the playfield, one unique non-grey corner marker marks the launch corner, and every grey connected component is a solid rectangular obstacle block.
    - The legend row contains a left-to-right strip of trajectory colors separated by background gaps.

    Rule:
    - Launch from the marked corner toward the interior along the matching diagonal, while keeping the marker, divider, legend, and grey obstacles unchanged.
    - Move diagonally through empty playfield cells; when the next step would leave the playfield, reflect on the outer wall and keep the current trajectory color.
    - When the next diagonal step would enter a grey obstacle, reflect on that obstacle and advance to the next legend color from that bounce onward.
    - Stop when the ball reaches any playfield corner, and paint each visited playfield cell in traversal order with the active trajectory color.

    Key insight:
    - The divider isolates a color program: only obstacle collisions consume that program, while wall bounces are color-neutral decoys.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    layout = _parse_layout(g, bg)
    trace = _trace_path(layout["playfield"], layout["start"], layout["legend"])

    out = g.copy()
    play_r0 = int(layout["play_r0"])
    start = tuple(map(int, layout["start"]))
    for r, c, color in trace["path"]:
        if (int(r), int(c)) == start:
            continue
        out[play_r0 + int(r), int(c)] = int(color)
    return out


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _random_corner(rng: np.random.Generator, h: int, w: int) -> tuple[int, int]:
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    return corners[int(rng.integers(0, len(corners)))]


def _place_obstacles(
    rng: np.random.Generator,
    h: int,
    w: int,
) -> list[tuple[int, int, int, int]] | None:
    count = int(rng.choice([1, 2, 2, 3, 3, 4]))
    boxes: list[tuple[int, int, int, int]] = []

    for _ in range(count):
        placed = False
        for _ in range(80):
            oh = int(rng.choice(OBSTACLE_H_OPTIONS))
            ow = int(rng.choice(OBSTACLE_W_OPTIONS))
            if oh >= h - 1 or ow >= w - 1:
                continue
            r0 = int(rng.integers(1, h - oh))
            c0 = int(rng.integers(1, w - ow))
            box = (r0, c0, r0 + oh - 1, c0 + ow - 1)
            if any(_boxes_too_close(box, other, pad=1) for other in boxes):
                continue
            boxes.append(box)
            placed = True
            break
        if not placed:
            return None

    return boxes


def _render_legend_row(rng: np.random.Generator, width: int, colors: list[int]) -> np.ndarray:
    row = np.full(width, BG, dtype=int)
    n = len(colors)
    widths = [1] * n
    spare = width - (n + max(0, n - 1))
    order = list(rng.permutation(n))
    for idx in order:
        if spare <= 0:
            break
        if rng.random() < 0.6:
            widths[idx] += 1
            spare -= 1

    total = sum(widths) + max(0, n - 1)
    left = int(rng.integers(0, width - total + 1))
    c = left
    for idx, color in enumerate(colors):
        row[c : c + widths[idx]] = int(color)
        c += widths[idx]
        if idx + 1 < n:
            c += 1
    return row


def _build_playfield(
    h: int,
    w: int,
    start: tuple[int, int],
    start_color: int,
    boxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    playfield = np.full((h, w), BG, dtype=int)
    for r0, c0, r1, c1 in boxes:
        playfield[r0 : r1 + 1, c0 : c1 + 1] = int(OBSTACLE)
    playfield[start] = int(start_color)
    return playfield


def _trajectory_is_clear(playfield: np.ndarray, start: tuple[int, int], legend_len: int) -> bool:
    try:
        stats = _trace_path(playfield, start, list(range(legend_len)))
    except ValueError:
        return False

    coords = [(int(r), int(c)) for r, c, _ in stats["path"]]
    repeated = len(coords) - len(set(coords))
    return (
        1 <= int(stats["obstacle_hits"]) <= 4
        and int(stats["wall_hits"]) >= 1
        and len(coords) >= max(6, min(playfield.shape))
        and repeated <= 1
        and stats["end"] != start
        and all(kind != "flip_both" for kind in stats["obstacle_kinds"])
    )


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(PLAY_H_RANGE[0], PLAY_H_RANGE[1] + 1))
    w = int(rng.integers(PLAY_W_RANGE[0], PLAY_W_RANGE[1] + 1))
    start = _random_corner(rng, h, w)
    boxes = _place_obstacles(rng, h, w)
    if boxes is None:
        return None

    probe = np.full((h, w), BG, dtype=int)
    for r0, c0, r1, c1 in boxes:
        probe[r0 : r1 + 1, c0 : c1 + 1] = int(OBSTACLE)

    try:
        stats = _trace_path(probe, start, list(range(10)))
    except ValueError:
        return None

    if not (
        1 <= int(stats["obstacle_hits"]) <= 4
        and int(stats["wall_hits"]) >= 1
        and stats["end"] != start
        and all(kind != "flip_both" for kind in stats["obstacle_kinds"])
    ):
        return None

    coords = [(int(r), int(c)) for r, c, _ in stats["path"]]
    if len(coords) < max(6, min(h, w)) or len(coords) - len(set(coords)) > 1:
        return None

    available = list(TRACK_POOL)
    start_color = int(available.pop(int(rng.integers(0, len(available)))))
    divider_color = int(available.pop(int(rng.integers(0, len(available)))))
    if len(available) < int(stats["obstacle_hits"]) + 1:
        return None
    pick = rng.choice(available, size=int(stats["obstacle_hits"]) + 1, replace=False)
    legend_colors = [int(v) for v in pick]

    playfield = _build_playfield(h, w, start, start_color, boxes)
    legend_row = _render_legend_row(rng, w, legend_colors)
    divider_row = np.full(w, divider_color, dtype=int)

    if bool(rng.integers(0, 2)):
        grid = np.vstack([legend_row[None, :], divider_row[None, :], playfield])
    else:
        grid = np.vstack([playfield, divider_row[None, :], legend_row[None, :]])

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None
    if not _trajectory_is_clear(playfield, start, len(legend_colors)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid recoloring pinball input")
