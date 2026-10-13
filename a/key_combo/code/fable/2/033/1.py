from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_RETRIES = 500
GRID_MIN = 16
GRID_MAX = 28
OBSTACLE_SIDE_MIN = 4
OBSTACLE_SIDE_MAX = 8
APPROACH_SIDES = ("left", "right", "top", "bottom")
MIN_OUTER_MARGIN = 3
MIN_GAP = 2
MAX_GAP = 6
MIN_REMAIN = 2
MIN_RIBBON_COLORS = 3
MAX_RIBBON_COLORS = 5


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def add(p: tuple[int, int], d: tuple[int, int]) -> tuple[int, int]:
    return int(p[0] + d[0]), int(p[1] + d[1])


def turn_left(d: tuple[int, int]) -> tuple[int, int]:
    dr, dc = d
    return -dc, dr


def turn_right(d: tuple[int, int]) -> tuple[int, int]:
    dr, dc = d
    return dc, -dr


def reverse_dir(d: tuple[int, int]) -> tuple[int, int]:
    return -d[0], -d[1]


def bbox_of(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in neighbors4(cr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _ray_hits_obstacle(
    grid: np.ndarray,
    obstacle_mask: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    bg: int,
) -> bool:
    r, c = add(start, direction)
    while in_bounds(grid.shape, r, c):
        if obstacle_mask[r, c]:
            return True
        if int(grid[r, c]) != int(bg):
            return False
        r, c = add((r, c), direction)
    return False


def _face_run_length(
    obstacle_mask: np.ndarray,
    contact: tuple[int, int],
    approach: tuple[int, int],
    turn: tuple[int, int],
) -> int:
    pos = contact
    count = 0
    while True:
        nxt = add(pos, turn)
        if not in_bounds(obstacle_mask.shape, nxt[0], nxt[1]):
            return count
        if obstacle_mask[nxt]:
            return count
        inner = add(nxt, approach)
        if not in_bounds(obstacle_mask.shape, inner[0], inner[1]) or not obstacle_mask[inner]:
            return count
        count += 1
        pos = nxt


def _choose_contour_step(
    obstacle_mask: np.ndarray,
    pos: tuple[int, int],
    move: tuple[int, int],
    hand: str,
) -> tuple[int, int]:
    if hand == "right":
        options = [turn_right(move), move, turn_left(move), reverse_dir(move)]
    else:
        options = [turn_left(move), move, turn_right(move), reverse_dir(move)]
    for cand in options:
        nxt = add(pos, cand)
        if in_bounds(obstacle_mask.shape, nxt[0], nxt[1]) and not obstacle_mask[nxt]:
            return cand
    raise AssertionError("no contour continuation found")


def _trace_track(
    obstacle_mask: np.ndarray,
    head: tuple[int, int],
    approach: tuple[int, int],
    length: int,
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int], dict[tuple[int, int], int]]:
    assert length >= 1
    track = [head]
    current = head
    while len(track) < length:
        nxt = add(current, approach)
        if not in_bounds(obstacle_mask.shape, nxt[0], nxt[1]) or obstacle_mask[nxt]:
            break
        current = nxt
        track.append(current)

    if len(track) == length:
        left_len = _face_run_length(obstacle_mask, current, approach, turn_left(approach))
        right_len = _face_run_length(obstacle_mask, current, approach, turn_right(approach))
        return track, current, approach, {turn_left(approach): left_len, turn_right(approach): right_len}

    obstacle_ahead = add(current, approach)
    assert in_bounds(obstacle_mask.shape, obstacle_ahead[0], obstacle_ahead[1]) and obstacle_mask[obstacle_ahead]

    left_turn = turn_left(approach)
    right_turn = turn_right(approach)
    left_len = _face_run_length(obstacle_mask, current, approach, left_turn)
    right_len = _face_run_length(obstacle_mask, current, approach, right_turn)
    if left_len > right_len:
        first_turn = left_turn
        hand = "right"
    elif right_len > left_len:
        first_turn = right_turn
        hand = "left"
    else:
        first_turn = left_turn
        hand = "right"

    first_step = add(current, first_turn)
    assert in_bounds(obstacle_mask.shape, first_step[0], first_step[1]) and not obstacle_mask[first_step]
    track.append(first_step)
    current = first_step
    move = first_turn

    max_steps = obstacle_mask.size * 8
    for _ in range(max_steps):
        if len(track) >= length:
            break
        move = _choose_contour_step(obstacle_mask, current, move, hand)
        current = add(current, move)
        track.append(current)
    assert len(track) == length, "track did not reach ribbon length"
    return track, track[min(len(track) - 1, 0)], first_turn, {left_turn: left_len, right_turn: right_len}


def _parse_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)

    comps = components_mask(g != bg)
    assert len(comps) == 2, "expected exactly one obstacle and one ribbon"

    ribbon = None
    obstacle = None
    for comp in comps:
        r0, c0, r1, c1 = bbox_of(comp)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        colors = {int(g[r, c]) for r, c in comp}
        if min(h, w) == 1 and len(comp) == max(h, w) and len(colors) >= 2:
            assert ribbon is None, "expected a unique multicolor ribbon"
            ribbon = comp
        else:
            assert obstacle is None, "expected a unique obstacle"
            obstacle = comp

    assert ribbon is not None and obstacle is not None

    obstacle_colors = {int(g[r, c]) for r, c in obstacle}
    assert len(obstacle_colors) == 1, "obstacle must be monochrome"
    obstacle_mask = np.zeros_like(g, dtype=bool)
    for r, c in obstacle:
        obstacle_mask[r, c] = True

    rr0, rc0, rr1, rc1 = bbox_of(ribbon)
    horizontal = rr0 == rr1
    if horizontal:
        ordered = sorted(ribbon, key=lambda p: p[1])
        end_a, end_b = ordered[0], ordered[-1]
        dir_a, dir_b = (0, -1), (0, 1)
    else:
        ordered = sorted(ribbon, key=lambda p: p[0])
        end_a, end_b = ordered[0], ordered[-1]
        dir_a, dir_b = (-1, 0), (1, 0)

    see_a = _ray_hits_obstacle(g, obstacle_mask, end_a, dir_a, bg)
    see_b = _ray_hits_obstacle(g, obstacle_mask, end_b, dir_b, bg)
    assert see_a ^ see_b, "exactly one ribbon end must see the obstacle"

    if see_a:
        head = end_a
        approach = dir_a
        ribbon_order = ordered
    else:
        head = end_b
        approach = dir_b
        ribbon_order = list(reversed(ordered))

    colors = [int(g[r, c]) for r, c in ribbon_order]
    track, _contact_or_last, _first_turn, face_lengths = _trace_track(
        obstacle_mask, head, approach, len(colors)
    )

    return {
        "bg": int(bg),
        "grid": g,
        "obstacle_mask": obstacle_mask,
        "obstacle_color": next(iter(obstacle_colors)),
        "ribbon_order": ribbon_order,
        "colors": colors,
        "track": track,
        "approach": approach,
        "face_lengths": face_lengths,
    }


def _turn_count(path: list[tuple[int, int]]) -> int:
    if len(path) < 3:
        return 0
    dirs = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    return sum(dirs[i] != dirs[i - 1] for i in range(1, len(dirs)))


def _sample_ribbon_colors(length: int, used_colors: set[int]) -> list[int]:
    palette = [color for color in range(10) if color not in used_colors]
    random.shuffle(palette)
    palette = palette[: random.randint(MIN_RIBBON_COLORS, min(MAX_RIBBON_COLORS, len(palette)))]
    assert len(palette) >= 3

    for _ in range(200):
        seq = [random.choice(palette) for _ in range(length)]
        if len(set(seq)) < 3:
            continue
        if all(seq[i] == seq[0] for i in range(length)):
            continue
        if seq == list(reversed(seq)):
            continue
        if len(set(seq[i] for i in range(0, length, 2))) == 1 and len(set(seq[i] for i in range(1, length, 2))) == 1:
            continue
        if sum(seq[i] != seq[i - 1] for i in range(1, length)) < 3:
            continue
        return seq
    raise RuntimeError("failed to sample ribbon colors")


def _paint_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _place_ribbon(
    grid: np.ndarray,
    side: str,
    obstacle_box: tuple[int, int, int, int],
    impact_index: int,
    gap: int,
    length: int,
    colors: list[int],
) -> None:
    r0, c0, r1, c1 = obstacle_box
    if side == "left":
        head = (r0 + impact_index, c0 - 1 - gap)
        ribbon_cells = [(head[0], head[1] - k) for k in range(length)]
    elif side == "right":
        head = (r0 + impact_index, c1 + 1 + gap)
        ribbon_cells = [(head[0], head[1] + k) for k in range(length)]
    elif side == "top":
        head = (r0 - 1 - gap, c0 + impact_index)
        ribbon_cells = [(head[0] - k, head[1]) for k in range(length)]
    else:
        head = (r1 + 1 + gap, c0 + impact_index)
        ribbon_cells = [(head[0] + k, head[1]) for k in range(length)]

    for (r, c), color in zip(ribbon_cells, colors):
        grid[r, c] = int(color)


def _space_available(side: str, obstacle_box: tuple[int, int, int, int], shape: tuple[int, int]) -> int:
    r0, c0, r1, c1 = obstacle_box
    h, w = shape
    if side == "left":
        return int(c0)
    if side == "right":
        return int(w - 1 - c1)
    if side == "top":
        return int(r0)
    return int(h - 1 - r1)


def _valid_impacts(span: int) -> list[int]:
    mid = span // 2
    values = []
    for idx in range(span):
        if span % 2 == 1 and idx == mid:
            continue
        values.append(idx)
    return values


def _build_candidate() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid = np.full((h, w), random.randint(0, 9), dtype=int)
    bg = int(grid[0, 0])

    obstacle_h = random.randint(OBSTACLE_SIDE_MIN, OBSTACLE_SIDE_MAX)
    obstacle_w = random.randint(OBSTACLE_SIDE_MIN, OBSTACLE_SIDE_MAX)
    if obstacle_h + 2 * MIN_OUTER_MARGIN >= h or obstacle_w + 2 * MIN_OUTER_MARGIN >= w:
        return None

    r0 = random.randint(MIN_OUTER_MARGIN, h - obstacle_h - MIN_OUTER_MARGIN - 1)
    c0 = random.randint(MIN_OUTER_MARGIN, w - obstacle_w - MIN_OUTER_MARGIN - 1)
    obstacle_box = (r0, c0, r0 + obstacle_h - 1, c0 + obstacle_w - 1)

    obstacle_color_choices = [color for color in range(10) if color != bg]
    obstacle_color = random.choice(obstacle_color_choices)
    _paint_rect(grid, obstacle_box, obstacle_color)

    valid_sides = [side for side in APPROACH_SIDES if _space_available(side, obstacle_box, grid.shape) >= 9]
    if not valid_sides:
        return None
    side = random.choice(valid_sides)
    span = obstacle_h if side in ("left", "right") else obstacle_w
    impacts = _valid_impacts(span)
    if not impacts:
        return None
    impact_index = random.choice(impacts)

    available = _space_available(side, obstacle_box, grid.shape)
    gap_hi = min(MAX_GAP, max(MIN_GAP, (available - MIN_REMAIN - 1) // 2))
    if gap_hi < MIN_GAP:
        return None
    gap = random.randint(MIN_GAP, gap_hi)

    ring_len = 2 * obstacle_h + 2 * obstacle_w + 4
    max_remaining = min(ring_len - 2, available - 2 * gap - 1)
    if max_remaining < MIN_REMAIN:
        return None
    min_turns = random.choice([2, 2, 2, 3])
    remaining = random.randint(MIN_REMAIN, max_remaining)
    length = gap + 1 + remaining

    colors = _sample_ribbon_colors(length, {bg, obstacle_color})
    _place_ribbon(grid, side, obstacle_box, impact_index, gap, length, colors)

    try:
        parsed = _parse_instance(grid)
    except AssertionError:
        return None

    track = parsed["track"]
    if len(track) != length:
        return None
    if len(set(track)) != len(track):
        return None
    if _turn_count(track) < min_turns:
        return None

    face_lengths = parsed["face_lengths"]
    if len(set(face_lengths.values())) != 2:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    moved_cells = [(r, c) for r, c in track if (r, c) not in parsed["ribbon_order"]]
    if not moved_cells:
        return None

    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the straight multicolor ribbon toward the obstacle, then relay the remaining colors around its outside contour.

    Input structure:
    - The dominant color is the background.
    - One connected monochrome obstacle sits away from the ribbon.
    - One 1-pixel-wide multicolor ribbon is perfectly straight, horizontal or vertical, and exactly one of its two ends can see the obstacle by extending the ribbon's own axis.

    Rule:
    - The visible end is the ribbon head; read the ribbon's color sequence from that head to the far tail.
    - Lay a track starting at the head: first march straight toward the obstacle until the last background cell before impact, then compare the two possible side turns and choose the one whose obstacle-touching side run is longer; continue from there by wall-following the obstacle's exterior.
    - Erase the original ribbon and rewrite its colors, in the same head-to-tail order, onto the new track; keep the obstacle and all other background cells unchanged.

    Key insight:
    - The ribbon behaves like a train whose cargo is the color order: geometry is rebuilt from straight approach plus contour-hugging motion, and the turning side is chosen by the longer immediately continuable obstacle-adjacent direction rather than by distance alone.
    """
    parsed = _parse_instance(np.asarray(grid, dtype=int))
    out = parsed["grid"].copy()
    bg = int(parsed["bg"])

    for r, c in parsed["ribbon_order"]:
        out[r, c] = bg
    for (r, c), color in zip(parsed["track"], parsed["colors"]):
        out[r, c] = int(color)
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ribbon-around-obstacle input")
