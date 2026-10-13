from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8)

DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
CARDINALS = tuple(d for d in DIRS8 if 0 in d and d != (0, 0))
DIAGONALS = tuple(d for d in DIRS8 if 0 not in d)

FULL9 = {(r, c) for r in range(3) for c in range(3)}
RING8 = FULL9 - {(1, 1)}
MISS_BL8 = FULL9 - {(2, 0)}
SUBSET7 = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 2)}
PARTIAL7 = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 1)}
CARDINAL7 = {(0, 0), (0, 1), (0, 2), (1, 1), (2, 0), (2, 1), (2, 2)}
VERT_BIG7 = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 1)}
VERT_SMALL7 = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 2)}
LEFT_BIG7 = {(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 1), (2, 2)}
LEFT_SMALL7 = {(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)}
RIGHT_BIG7 = {(0, 0), (0, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2)}
RIGHT_SMALL7 = {(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)}
STAIR6 = {(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)}
STAIR6_M = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)}
SUPPORT_LIBRARY = (FULL9, RING8, MISS_BL8, SUBSET7, PARTIAL7, CARDINAL7, STAIR6, STAIR6_M)


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    comps.sort(key=lambda comp: _bbox(comp)[:2])
    return comps


def _infer_direction_from_cells(grey_cells: list[tuple[int, int]], color_cells: list[tuple[int, int]]) -> tuple[int, int] | None:
    if not color_cells:
        return None
    grey_center = np.mean(np.asarray(grey_cells, dtype=float), axis=0)
    color_center = np.mean(np.asarray(color_cells, dtype=float), axis=0)
    direction = (_sign(color_center[0] - grey_center[0]), _sign(color_center[1] - grey_center[1]))
    if direction == (0, 0):
        return None
    return direction


def _parse_sprite(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict:
    values = {pos: int(grid[pos]) for pos in comp}
    r0, c0, r1, c1 = _bbox(comp)
    local_cells: dict[tuple[int, int], int] = {}
    grey_local: list[tuple[int, int]] = []
    color_local: list[tuple[int, int]] = []
    for (r, c), value in values.items():
        local = (r - r0, c - c0)
        local_cells[local] = value
        if value == GREY:
            grey_local.append(local)
        else:
            color_local.append(local)
    return {
        "bbox": (r0, c0, r1, c1),
        "cells": values,
        "local_cells": local_cells,
        "grey_local": grey_local,
        "color_local": color_local,
        "grey_area": len(grey_local),
        "direction": _infer_direction_from_cells(grey_local, color_local),
    }


def _translate_bbox(bbox: tuple[int, int, int, int], direction: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    dr, dc = direction
    shift_r = 0
    shift_c = 0
    if dr < 0:
        shift_r = -r0
    elif dr > 0:
        shift_r = h - 1 - r1
    if dc < 0:
        shift_c = -c0
    elif dc > 0:
        shift_c = w - 1 - c1
    return shift_r, shift_c


def _solve_with_trace(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    comps = _component_coords(g != BLACK)
    sprites = [_parse_sprite(g, comp) for comp in comps]

    anchors = [sprite for sprite in sprites if sprite["direction"] is None]
    movers = [sprite for sprite in sprites if sprite["direction"] is not None]
    assert len(anchors) == 1, "expected one all-grey anchor"
    anchor = anchors[0]

    moved: list[dict] = []
    for idx, sprite in enumerate(movers):
        shift_r, shift_c = _translate_bbox(sprite["bbox"], sprite["direction"], g.shape)
        translated: dict[tuple[int, int], int] = {}
        grey_positions: set[tuple[int, int]] = set()
        for (lr, lc), value in sprite["local_cells"].items():
            rr = sprite["bbox"][0] + lr + shift_r
            cc = sprite["bbox"][1] + lc + shift_c
            translated[(rr, cc)] = value
            if value == GREY:
                grey_positions.add((rr, cc))
        moved.append(
            {
                "idx": idx,
                "bbox": sprite["bbox"],
                "direction": sprite["direction"],
                "grey_area": sprite["grey_area"],
                "translated": translated,
                "grey_positions": grey_positions,
            }
        )

    order = sorted(range(len(moved)), key=lambda i: (moved[i]["grey_area"], moved[i]["bbox"][0], moved[i]["bbox"][1]))
    owner = np.full((h, w), -1, dtype=int)
    final_color = np.full((h, w), BLACK, dtype=int)
    for idx in order:
        for (r, c), value in moved[idx]["translated"].items():
            owner[r, c] = idx
            final_color[r, c] = value

    out = np.full_like(g, BLACK)
    trace_movers: list[dict] = []
    for idx, sprite in enumerate(moved):
        visible = {pos: value for pos, value in sprite["translated"].items() if owner[pos] == idx}
        visible_grey = {pos for pos, value in visible.items() if value == GREY}
        visible_grey_count = len(visible_grey)
        partial = 0 < visible_grey_count < sprite["grey_area"]
        hidden = visible_grey_count == 0
        if not hidden:
            for pos, value in visible.items():
                out[pos] = MAROON if partial and value == GREY else value
        trace_movers.append(
            {
                "bbox": sprite["bbox"],
                "direction": sprite["direction"],
                "grey_area": sprite["grey_area"],
                "occupied": set(sprite["translated"]),
                "visible": set(visible),
                "visible_grey_count": visible_grey_count,
                "hidden": hidden,
                "partial": partial,
            }
        )

    for (r, c), value in anchor["cells"].items():
        out[r, c] = value

    return out, {"anchor": anchor, "movers": trace_movers}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each direction-encoded grey sprite to the matching canvas edge and resolve overlaps by grey-area priority.

    Input structure:
    - The grid is a black background containing several isolated grey-led sprites.
    - One sprite is an all-grey 3x3 anchor that never moves.
    - Every other sprite is a 4-connected irregular shape inside a 3x3 frame, made of grey cells plus 1-2 colored marker cells whose bias indicates one of eight directions.

    Rule:
    - Detect every non-black connected sprite; the unique sprite without colored cells is the stationary anchor.
    - For each moving sprite, compare the centroid of its colored marker cells to the centroid of its grey body to decode one of the eight directions.
    - Translate that whole sprite until the corresponding side or corner of its bounding box is flush with the canvas edge.
    - When translated sprites overlap, the sprite with larger grey area covers the smaller one on the overlapping cells.
    - Any sprite that keeps some but not all of its original grey cells recolors all remaining visible grey cells to maroon; a sprite with no visible grey cells disappears. Non-colliding sprites and the anchor keep their original colors.

    Key insight:
    - Direction markers determine only the slide destination; the visible recoloring pattern is a second-stage effect caused by edge collisions, and those collisions are resolved by comparing sprite grey areas rather than by color identity.
    """
    return _solve_with_trace(np.asarray(grid, dtype=int))[0]


def _transform_support(coords: set[tuple[int, int]], rot: int, flip: bool) -> set[tuple[int, int]]:
    out = set()
    for r, c in coords:
        if flip:
            c = 2 - c
        for _ in range(rot % 4):
            r, c = c, 2 - r
        out.add((r, c))
    return out


def _sprite_from_support(
    support: set[tuple[int, int]],
    direction: tuple[int, int],
    color_count: int,
    color: int,
) -> np.ndarray | None:
    if len(support) - color_count < 5:
        return None
    dr, dc = direction
    cells = sorted(support)
    combos = []
    for combo in itertools.combinations(cells, color_count):
        grey_cells = [pos for pos in cells if pos not in combo]
        if _infer_direction_from_cells(grey_cells, list(combo)) == direction:
            score = sum((pos[0] - 1) * dr + (pos[1] - 1) * dc for pos in combo)
            score += 0.1 * sum(abs(pos[0] - 1) + abs(pos[1] - 1) for pos in combo)
            combos.append((score, combo))
    if not combos:
        return None
    best_score = max(score for score, _ in combos)
    chosen = random.choice([combo for score, combo in combos if score == best_score])

    sprite = np.zeros((3, 3), dtype=int)
    for pos in cells:
        sprite[pos] = GREY
    for pos in chosen:
        sprite[pos] = color
    grey_local = [tuple(pos) for pos in np.argwhere(sprite == GREY)]
    color_local = [tuple(pos) for pos in np.argwhere((sprite != GREY) & (sprite != BLACK))]
    if _infer_direction_from_cells(grey_local, color_local) != direction:
        return None
    return sprite


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0] - pad
        or b[2] + pad < a[0] - pad
        or a[3] + pad < b[1] - pad
        or b[3] + pad < a[1] - pad
    )


def _paste_sprite(grid: np.ndarray, sprite: np.ndarray, top: int, left: int) -> None:
    for r in range(3):
        for c in range(3):
            value = int(sprite[r, c])
            if value != BLACK:
                grid[top + r, left + c] = value


def _destination_box(direction: tuple[int, int], lane: int | None, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    dr, dc = direction
    if dr < 0:
        r0 = 0
    elif dr > 0:
        r0 = h - 3
    else:
        assert lane is not None
        r0 = lane
    if dc < 0:
        c0 = 0
    elif dc > 0:
        c0 = w - 3
    else:
        assert lane is not None
        c0 = lane
    return r0, c0, r0 + 2, c0 + 2


def _candidate_origins(shape: tuple[int, int], direction: tuple[int, int], lane: int | None) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = direction
    if dr < 0:
        row_range = range(1, h - 3)
    elif dr > 0:
        row_range = range(1, h - 3)
    else:
        assert lane is not None
        row_range = [lane]
    if dc < 0:
        col_range = range(1, w - 3)
    elif dc > 0:
        col_range = range(1, w - 3)
    else:
        assert lane is not None
        col_range = [lane]
    return [(r, c) for r in row_range for c in col_range]


def _collision_profile(trace: dict) -> tuple[list[list[int]], list[str]]:
    movers = trace["movers"]
    n = len(movers)
    graph = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if movers[i]["occupied"] & movers[j]["occupied"]:
                graph[i].append(j)
                graph[j].append(i)
    groups = []
    seen = [False] * n
    for i in range(n):
        if seen[i] or not graph[i]:
            continue
        stack = [i]
        seen[i] = True
        group = []
        while stack:
            cur = stack.pop()
            group.append(cur)
            for nxt in graph[cur]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
        groups.append(sorted(group))

    kinds = []
    for group in groups:
        if len(group) != 2:
            kinds.append("invalid")
            continue
        a, b = group
        lo, hi = sorted((a, b), key=lambda idx: movers[idx]["grey_area"])
        if movers[lo]["grey_area"] == movers[hi]["grey_area"]:
            kinds.append("invalid")
        elif movers[lo]["visible_grey_count"] == 0:
            kinds.append("full")
        elif movers[lo]["partial"]:
            kinds.append("partial")
        else:
            kinds.append("invalid")
    return groups, kinds


def _build_anchor() -> np.ndarray:
    return np.full((3, 3), GREY, dtype=int)


def _make_collision_pair(
    big_support: set[tuple[int, int]],
    small_support: set[tuple[int, int]],
    direction: tuple[int, int],
    colors: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    big_options = [k for k in (1, 2) if len(big_support) - k >= 5]
    small_options = [k for k in (1, 2) if len(small_support) - k >= 5]
    random.shuffle(big_options)
    random.shuffle(small_options)
    for kb in big_options:
        for ks in small_options:
            big = _sprite_from_support(big_support, direction, kb, colors[0])
            small = _sprite_from_support(small_support, direction, ks, colors[1])
            if big is None or small is None:
                continue
            if int(np.count_nonzero(big == GREY)) > int(np.count_nonzero(small == GREY)):
                return big, small
    return None


def _sample_unique_colors(n: int) -> list[int]:
    palette = list(MARKER_COLORS)
    random.shuffle(palette)
    return palette[:n]


def _make_partial_pair(direction: tuple[int, int], colors: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    if direction in {(-1, 0), (1, 0)}:
        big_support = VERT_BIG7
        small_support = VERT_SMALL7
    elif direction == (0, -1):
        big_support = LEFT_BIG7
        small_support = LEFT_SMALL7
    else:
        big_support = RIGHT_BIG7
        small_support = RIGHT_SMALL7
    big = _sprite_from_support(big_support, direction, 1, colors[0])
    small = _sprite_from_support(small_support, direction, 2, colors[1])
    if big is None or small is None:
        return None
    return big, small


def _build_extra_sprite(direction: tuple[int, int], color: int) -> np.ndarray | None:
    supports = list(SUPPORT_LIBRARY)
    random.shuffle(supports)
    for support in supports:
        k_choices = [k for k in (1, 2) if len(support) - k >= 5]
        random.shuffle(k_choices)
        for k in k_choices:
            sprite = _sprite_from_support(support, direction, k, color)
            if sprite is not None:
                return sprite
    return None


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(16, 20)
    w = random.randint(16, 20)
    canvas = np.zeros((h, w), dtype=int)

    anchor = _build_anchor()
    anchor_top = random.randint(h // 2 - 2, h // 2)
    anchor_left = random.randint(w // 2 - 2, w // 2)
    boxes = [(anchor_top, anchor_left, anchor_top + 2, anchor_left + 2)]

    colors = _sample_unique_colors(6)

    rot_full = random.randrange(4)
    flip_full = bool(random.getrandbits(1))
    full_pair = _make_collision_pair(
        _transform_support(FULL9, rot_full, flip_full),
        _transform_support(SUBSET7, rot_full, flip_full),
        random.choice(DIAGONALS),
        (colors[0], colors[1]),
    )
    if full_pair is None:
        return None
    full_big, full_small = full_pair
    full_dir = _infer_direction_from_cells(
        [tuple(pos) for pos in np.argwhere(full_big == GREY)],
        [tuple(pos) for pos in np.argwhere((full_big != GREY) & (full_big != BLACK))],
    )
    assert full_dir is not None

    edge_dir = random.choice(CARDINALS)
    partial_pair = _make_partial_pair(edge_dir, (colors[2], colors[3]))
    if partial_pair is None:
        return None
    part_big, part_small = partial_pair

    used_dest_boxes: list[tuple[int, int, int, int]] = []
    full_dest = _destination_box(full_dir, None, (h, w))
    used_dest_boxes.append(full_dest)

    part_lane_candidates = list(range(1, (w - 3) if edge_dir[0] else (h - 3)))
    random.shuffle(part_lane_candidates)
    part_lane = None
    for lane in part_lane_candidates:
        dest = _destination_box(edge_dir, lane, (h, w))
        if not any(_bbox_overlap(dest, other) for other in used_dest_boxes):
            part_lane = lane
            used_dest_boxes.append(dest)
            break
    if part_lane is None:
        return None

    extra_count = random.choice((1, 1, 2))
    extras: list[tuple[np.ndarray, tuple[int, int], int | None]] = []
    for extra_idx in range(extra_count):
        for _ in range(80):
            direction = random.choice(DIRS8)
            lane = None
            if direction in CARDINALS:
                lane_choices = list(range(1, (w - 3) if direction[0] else (h - 3)))
                random.shuffle(lane_choices)
                lane = None
                for maybe_lane in lane_choices:
                    dest = _destination_box(direction, maybe_lane, (h, w))
                    if any(_bbox_overlap(dest, other) for other in used_dest_boxes):
                        continue
                    lane = maybe_lane
                    break
                if lane is None:
                    continue
                dest_box = _destination_box(direction, lane, (h, w))
            else:
                dest_box = _destination_box(direction, None, (h, w))
                if any(_bbox_overlap(dest_box, other) for other in used_dest_boxes):
                    continue
            sprite = _build_extra_sprite(direction, colors[4 + extra_idx])
            if sprite is None:
                continue
            extras.append((sprite, direction, lane))
            used_dest_boxes.append(dest_box)
            break
        else:
            return None

    objects = [
        (anchor, None, None, "anchor"),
        (full_big, full_dir, None, "full_big"),
        (full_small, full_dir, None, "full_small"),
        (part_big, edge_dir, part_lane, "part_big"),
        (part_small, edge_dir, part_lane, "part_small"),
    ]
    for idx, (sprite, direction, lane) in enumerate(extras):
        objects.append((sprite, direction, lane, f"extra_{idx}"))

    placements: dict[str, tuple[int, int]] = {"anchor": (anchor_top, anchor_left)}
    for sprite, direction, lane, name in objects[1:]:
        candidates = _candidate_origins((h, w), direction, lane)
        random.shuffle(candidates)
        placed = False
        for top, left in candidates:
            box = (top, left, top + 2, left + 2)
            if any(_bbox_overlap(box, other, pad=1) for other in boxes):
                continue
            placements[name] = (top, left)
            boxes.append(box)
            placed = True
            break
        if not placed:
            return None

    _paste_sprite(canvas, anchor, anchor_top, anchor_left)
    for sprite, _direction, _lane, name in objects[1:]:
        _paste_sprite(canvas, sprite, *placements[name])

    try:
        out, trace = _solve_with_trace(canvas)
    except AssertionError:
        return None

    groups, kinds = _collision_profile(trace)
    if sorted(kinds) != ["full", "partial"]:
        return None
    if len(groups) != 2 or any(len(group) != 2 for group in groups):
        return None
    if np.count_nonzero(out == MAROON) == 0:
        return None
    non_colliding = [i for i, mover in enumerate(trace["movers"]) if all(i not in group for group in groups)]
    if not non_colliding:
        return None
    if not all(not trace["movers"][i]["partial"] and not trace["movers"][i]["hidden"] for i in non_colliding):
        return None

    anchor_cells = list(trace["anchor"]["cells"])
    if not all(out[r, c] == GREY for r, c in anchor_cells):
        return None

    for comp in _component_coords(canvas != BLACK):
        sprite = _parse_sprite(canvas, comp)
        if sprite["direction"] is None:
            continue
        if sprite["grey_area"] < 5 or sprite["direction"] not in DIRS8:
            return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid collision-arbitration ARC input")
