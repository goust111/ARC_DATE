from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

_ARC_ROOT = Path("/home/lixinyu/snap/new")
if str(_ARC_ROOT) not in sys.path:
    sys.path.append(str(_ARC_ROOT))

from arc_utils import Color, bbox_h, bbox_w, find_components, neighbors4


BG = Color.BLACK
CHANNEL = Color.BLUE
MOVER = Color.RED
MASK_TOP = Color.GREEN
TARGET = Color.YELLOW
MASK_BOTTOM = Color.GREY

GEN_RETRIES = 400


def _panel_height(grid: np.ndarray) -> int:
    height, _width = grid.shape
    assert height % 4 == 0
    panel_h = height // 4
    assert panel_h >= 2
    return panel_h


def _xor_target(top_panel: np.ndarray, bottom_panel: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    xor_mask = (top_panel != BG) ^ (bottom_panel != BG)
    coords = np.argwhere(xor_mask)
    assert coords.shape[0] == 1
    r, c = map(int, coords[0])
    return xor_mask, (r, c)


def _find_mover(lower: np.ndarray) -> tuple[int, int]:
    movers: list[tuple[int, int]] = []
    for comp in find_components(
        lower,
        bg=BG,
        colors=[MOVER],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    ):
        bbox = comp["bbox"]
        if bbox_h(bbox) != 2 or bbox_w(bbox) != 2 or int(comp["area"]) != 4:
            continue
        r0, c0, r1, c1 = map(int, bbox)
        if bool((lower[r0 : r1 + 1, c0 : c1 + 1] == MOVER).all()):
            movers.append((r0, c0))
    assert len(movers) == 1
    return movers[0]


def _feasible_placements(lower: np.ndarray) -> np.ndarray:
    h, w = lower.shape
    traversable = (lower == CHANNEL) | (lower == MOVER)
    feasible = np.zeros((h - 1, w - 1), dtype=bool)
    for r in range(h - 1):
        for c in range(w - 1):
            feasible[r, c] = bool(traversable[r : r + 2, c : c + 2].all())
    return feasible


def _distance_map(feasible: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    dist = -np.ones(feasible.shape, dtype=int)
    sr, sc = start
    assert feasible[sr, sc]
    dist[sr, sc] = 0
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < feasible.shape[0] and 0 <= cc < feasible.shape[1]:
                if feasible[rr, cc] and dist[rr, cc] == -1:
                    dist[rr, cc] = dist[r, c] + 1
                    queue.append((rr, cc))
    return dist


def _choose_stop(dist: np.ndarray, feasible: np.ndarray, target: tuple[int, int]) -> tuple[int, int]:
    tr, tc = target
    if 0 <= tr < feasible.shape[0] and 0 <= tc < feasible.shape[1]:
        if feasible[tr, tc] and dist[tr, tc] >= 0:
            return tr, tc

    best_pos = None
    best_key = None
    for r in range(feasible.shape[0]):
        for c in range(feasible.shape[1]):
            d = int(dist[r, c])
            if d < 0:
                continue
            md = abs(r - tr) + abs(c - tc)
            key = (d, -md, -r, -c)
            if best_key is None or key > best_key:
                best_key = key
                best_pos = (r, c)
    assert best_pos is not None
    return best_pos


def _final_lower(lower: np.ndarray, target: tuple[int, int], stop: tuple[int, int]) -> np.ndarray:
    sr, sc = _find_mover(lower)
    tr, tc = target
    er, ec = stop
    out = lower.copy()
    out[sr : sr + 2, sc : sc + 2] = CHANNEL
    out[tr, tc] = TARGET
    out[er : er + 2, ec : ec + 2] = MOVER
    return out


def _solve_lower(lower: np.ndarray, target: tuple[int, int]) -> tuple[tuple[int, int], np.ndarray]:
    start = _find_mover(lower)
    feasible = _feasible_placements(lower)
    dist = _distance_map(feasible, start)
    stop = _choose_stop(dist, feasible, target)
    return stop, _final_lower(lower, target, stop)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the unique xor marker and slide the 2x2 block through the lower channel toward that anchor.

    Input structure:
    - The grid is split into three stacked parts: a top foreground mask panel, a second aligned mask panel,
      and a lower winding blue channel containing one red 2x2 slider.
    - The two upper panels have equal size and use background plus one foreground color each.
    - Exactly one aligned cell is foreground in exactly one upper panel; the mapped lower-half cell lies on the channel.

    Rule:
    - Compute the xor of the two upper panels and keep only the unique yellow cell in the first output panel;
      clear the second upper panel to background.
    - Map that xor coordinate to the lower section as a yellow target anchor at the same column and row offset.
    - Move the red 2x2 block through blue cells using only placements where the full 2x2 window fits inside the channel.
    - If the target anchor is a reachable 2x2 top-left placement, the block lands there; otherwise it stops at the farthest
      reachable placement and the uncovered yellow anchor remains visible.

    Key insight:
    - The upper xor is not the final answer by itself; it decodes the exact lower-channel destination that the size-constrained
      slider must then try to reach.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    panel_h = _panel_height(g)
    top_panel = g[:panel_h, :]
    bottom_panel = g[panel_h : 2 * panel_h, :]
    lower = g[2 * panel_h :, :]

    xor_mask, target = _xor_target(top_panel, bottom_panel)
    _stop, lower_out = _solve_lower(lower, target)

    out = np.full_like(g, BG)
    out[:panel_h, :] = np.where(xor_mask, TARGET, BG)
    out[2 * panel_h :, :] = lower_out
    return out


def _expand_polyline(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    path = [points[0]]
    for (r0, c0), (r1, c1) in zip(points, points[1:]):
        dr = 0 if r1 == r0 else (1 if r1 > r0 else -1)
        dc = 0 if c1 == c0 else (1 if c1 > c0 else -1)
        r, c = r0, c0
        while (r, c) != (r1, c1):
            r += dr
            c += dc
            path.append((r, c))
    return path


def _path_joints(panel_h: int, width: int) -> list[tuple[int, int]] | None:
    place_h = 2 * panel_h - 1
    place_w = width - 1
    if place_h < 6 or place_w < 6:
        return None

    upper_rows = list(range(0, panel_h - 1))
    lower_rows = list(range(panel_h, place_h))
    if not upper_rows or not lower_rows:
        return None

    r0 = random.choice(upper_rows)
    r1 = random.choice(lower_rows)
    r2 = random.choice(upper_rows)
    c1 = random.randint(1, place_w - 4)
    c2 = random.randint(c1 + 2, place_w - 2)

    return [(r0, 0), (r0, c1), (r1, c1), (r1, c2), (r2, c2), (r2, place_w - 1)]


def _paint_base_channel(lower: np.ndarray, path: list[tuple[int, int]]) -> None:
    for r, c in path:
        lower[r : r + 2, c : c + 2] = CHANNEL


def _straight_orientation(path: list[tuple[int, int]], idx: int) -> tuple[int, int] | None:
    prev_r, prev_c = path[idx - 1]
    cur_r, cur_c = path[idx]
    next_r, next_c = path[idx + 1]
    dr1, dc1 = cur_r - prev_r, cur_c - prev_c
    dr2, dc2 = next_r - cur_r, next_c - cur_c
    if (dr1, dc1) == (dr2, dc2):
        return dr1, dc1
    return None


def _add_bulges(lower: np.ndarray, path: list[tuple[int, int]], banned: set[tuple[int, int]]) -> None:
    height, width = lower.shape
    candidates = list(range(1, len(path) - 1))
    random.shuffle(candidates)
    additions = random.randint(0, 4)
    used = 0
    for idx in candidates:
        if used >= additions:
            break
        orientation = _straight_orientation(path, idx)
        if orientation is None:
            continue
        r, c = path[idx]
        cells: list[tuple[int, int]] = []
        if orientation[0] == 0:
            col = c + random.randint(0, 1)
            cells.extend([(r - 1, col), (r + 2, col)])
        else:
            row = r + random.randint(0, 1)
            cells.extend([(row, c - 1), (row, c + 2)])
        random.shuffle(cells)
        take = random.randint(1, 2)
        painted = False
        for rr, cc in cells[:take]:
            if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in banned:
                lower[rr, cc] = CHANNEL
                painted = True
        if painted:
            used += 1


def _apply_choke(lower: np.ndarray, path: list[tuple[int, int]], step_idx: int, side: int) -> tuple[int, int]:
    r0, c0 = path[step_idx]
    r1, c1 = path[step_idx + 1]
    if r0 == r1:
        rr = r0 + side
        cc = max(c0, c1)
    else:
        rr = max(r0, r1)
        cc = c0 + side
    lower[rr, cc] = BG
    return rr, cc


def _build_masks(panel_h: int, width: int, target: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    total = panel_h * width
    shared_count = random.randint(4, min(10, total - 1))
    all_cells = [(r, c) for r in range(panel_h) for c in range(width) if (r, c) != target]
    shared = set(random.sample(all_cells, shared_count))

    top = np.zeros((panel_h, width), dtype=int)
    bottom = np.zeros((panel_h, width), dtype=int)
    for r, c in shared:
        top[r, c] = MASK_TOP
        bottom[r, c] = MASK_BOTTOM
    if random.random() < 0.5:
        top[target] = MASK_TOP
    else:
        bottom[target] = MASK_BOTTOM
    return top, bottom


def _compose_input(top: np.ndarray, bottom: np.ndarray, lower: np.ndarray) -> np.ndarray:
    return np.vstack([top, bottom, lower]).astype(int)


def _segment_index(path: list[tuple[int, int]], coord: tuple[int, int]) -> int:
    return path.index(coord)


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.randint(4, 7)
    width = random.randint(8, 13)
    blocked = random.random() < 0.5

    joints = _path_joints(panel_h, width)
    if joints is None:
        return None
    path = _expand_polyline(joints)
    if len(path) < 10:
        return None

    lower_h = 2 * panel_h
    lower = np.full((lower_h, width), BG, dtype=int)
    _paint_base_channel(lower, path)

    upper_turn_idx = _segment_index(path, joints[4])
    final_segment_start = upper_turn_idx
    if final_segment_start >= len(path) - 1:
        return None

    choke_step = None
    banned: set[tuple[int, int]] = set()
    if blocked:
        middle_start = _segment_index(path, joints[2])
        middle_end = _segment_index(path, joints[3])
        if middle_end - middle_start < 2:
            return None
        choke_step = random.randint(middle_start + 1, middle_end - 1)
        banned.add(_apply_choke(lower, path, choke_step, random.randint(0, 1)))

    _add_bulges(lower, path, banned)

    start_r, start_c = path[0]
    lower[start_r : start_r + 2, start_c : start_c + 2] = MOVER

    target_idx = random.randint(final_segment_start, len(path) - 2)
    target = path[target_idx]
    if lower[target] != CHANNEL and lower[target] != MOVER:
        return None

    top, bottom = _build_masks(panel_h, width, target)
    inp = _compose_input(top, bottom, lower)

    xor_mask, decoded_target = _xor_target(top, bottom)
    if decoded_target != target:
        return None

    stop, lower_out = _solve_lower(lower, target)
    reached = stop == target
    if blocked == reached:
        return None

    out = np.full_like(inp, BG)
    out[:panel_h, :] = np.where(xor_mask, TARGET, BG)
    out[2 * panel_h :, :] = lower_out

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid xor-guided slider input")
