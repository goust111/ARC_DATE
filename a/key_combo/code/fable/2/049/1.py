from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
PATH = 5
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_TO_IDX = {d: i for i, d in enumerate(DIRS)}
SPRITE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
TURN_COUNT_CHOICES = [0, 1, 2, 3, 4]
TURN_COUNT_WEIGHTS = [1, 2, 4, 3, 1]
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int, copy=True)


def _follow_path(path_cells: set[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    ordered = [start]
    prev: tuple[int, int] | None = None
    current = start
    while True:
        next_cells = [p for p in _neighbors4(*current) if p in path_cells and p != prev]
        if not next_cells:
            break
        assert len(next_cells) == 1, "path must be a simple polyline"
        nxt = next_cells[0]
        ordered.append(nxt)
        prev, current = current, nxt
    assert len(ordered) == len(path_cells), "path traversal did not consume all path cells"
    return ordered


def _path_turns(ordered: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], list[int]]:
    dirs = [
        (int(r1 - r0), int(c1 - c0))
        for (r0, c0), (r1, c1) in zip(ordered, ordered[1:])
    ]
    corners: list[tuple[int, int]] = []
    rotations: list[int] = []
    rot = 0
    for idx in range(1, len(ordered) - 1):
        incoming = dirs[idx - 1]
        outgoing = dirs[idx]
        if incoming == outgoing:
            continue
        diff = (DIR_TO_IDX[outgoing] - DIR_TO_IDX[incoming]) % 4
        assert diff in (1, 3), "path may only turn left or right"
        if diff == 3:
            rot = (rot + 1) % 4
        else:
            rot = (rot - 1) % 4
        corners.append(ordered[idx])
        rotations.append(rot)
    return corners, rotations


def _stamp(out: np.ndarray, sprite: np.ndarray, center: tuple[int, int], color: int, rotation: int) -> None:
    patch = np.rot90(sprite, k=int(rotation) % 4)
    side = int(patch.shape[0])
    half = side // 2
    r0 = int(center[0]) - half
    c0 = int(center[1]) - half
    r1 = r0 + side
    c1 = c0 + side
    assert 0 <= r0 and 0 <= c0 and r1 <= out.shape[0] and c1 <= out.shape[1], "stamp out of bounds"
    view = out[r0:r1, c0:c1]
    view[patch != BG] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the sprite onto each gray-path corner using the cumulative turn rotation, then erase the path.

    Input structure:
    - The background is black.
    - One non-gray monochrome sprite has an odd square bbox whose geometric center is well defined.
    - One gray 4-connected polyline has exactly two endpoints, and exactly one endpoint is edge-adjacent to the sprite.

    Rule:
    - Keep the original sprite in place.
    - Traverse the gray path starting from the endpoint next to the sprite.
    - Each left turn adds one 90-degree counterclockwise rotation; each right turn adds one 90-degree clockwise rotation.
    - Stamp the rotated sprite centered on every corner using the cumulative rotation after that turn, then stamp the endpoint once more with the final rotation.
    - Erase the entire gray path to black.

    Key insight:
    - The gray polyline is only a disposable instruction tape: its corner sequence encodes a running orientation state, while straight lengths matter only for where the rotated sprite copies are placed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sprite_colors = [int(c) for c in np.unique(g) if int(c) not in (BG, PATH)]
    assert len(sprite_colors) == 1, "expected exactly one sprite color"
    sprite_color = sprite_colors[0]

    sprite_components = _component_coords(g == sprite_color)
    assert len(sprite_components) == 1, "expected exactly one sprite component"
    sprite_coords = sprite_components[0]
    sprite_set = {tuple(map(int, p)) for p in sprite_coords}
    sprite_bbox = _bbox_coords(sprite_coords)
    sprite = _crop_bbox(g, sprite_bbox)
    assert sprite.shape[0] == sprite.shape[1] and sprite.shape[0] % 2 == 1, "sprite bbox must be odd square"
    assert set(map(int, np.unique(sprite))) <= {BG, sprite_color}
    half = sprite.shape[0] // 2
    assert int(sprite[half, half]) == sprite_color, "sprite center must be colored"

    path_coords = {tuple(map(int, p)) for p in np.argwhere(g == PATH)}
    assert path_coords, "expected a gray path"

    degrees = {
        p: sum((nr, nc) in path_coords for nr, nc in _neighbors4(*p))
        for p in path_coords
    }
    endpoints = [p for p, deg in degrees.items() if int(deg) == 1]
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    assert all(int(deg) == 2 for p, deg in degrees.items() if p not in endpoints), "path must be a simple polyline"

    start_candidates = [
        p for p in endpoints
        if any((nr, nc) in sprite_set for nr, nc in _neighbors4(*p))
    ]
    assert len(start_candidates) == 1, "exactly one path endpoint must touch the sprite"
    start = start_candidates[0]

    ordered = _follow_path(path_coords, start)
    corners, rotations = _path_turns(ordered)

    out = np.array(g, copy=True)
    out[out == PATH] = BG

    for center, rot in zip(corners, rotations):
        _stamp(out, sprite, center, sprite_color, rot)
    final_rotation = rotations[-1] if rotations else 0
    _stamp(out, sprite, ordered[-1], sprite_color, final_rotation)
    return out


def _edge_start_options(mask: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    side = mask.shape[0]
    options: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    for c in range(side):
        if bool(mask[0, c]):
            options.append(((0, c), (-1, c), (-1, 0)))
        if bool(mask[side - 1, c]):
            options.append(((side - 1, c), (side, c), (1, 0)))
    for r in range(side):
        if bool(mask[r, 0]):
            options.append(((r, 0), (r, -1), (0, -1)))
        if bool(mask[r, side - 1]):
            options.append(((r, side - 1), (r, side), (0, 1)))
    return options


def _rotation_variants(mask: np.ndarray) -> list[np.ndarray]:
    return [np.rot90(mask, k=k) for k in range(4)]


def _sample_sprite_mask() -> np.ndarray:
    for _ in range(300):
        side = random.choices([3, 5], weights=[3, 5])[0]
        half = side // 2
        target = random.randint(side + 2, min(side * side - 1, side * side // 2 + side))
        cells = {(half, half)}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < side and 0 <= nc < side and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        edges = {
            "top": lambda p: p[0] == 0,
            "bottom": lambda p: p[0] == side - 1,
            "left": lambda p: p[1] == 0,
            "right": lambda p: p[1] == side - 1,
        }
        for edge_name, pred in edges.items():
            if any(pred(p) for p in cells):
                continue
            if edge_name == "top":
                target_cell = (0, random.randrange(side))
            elif edge_name == "bottom":
                target_cell = (side - 1, random.randrange(side))
            elif edge_name == "left":
                target_cell = (random.randrange(side), 0)
            else:
                target_cell = (random.randrange(side), side - 1)
            cr, cc = random.choice(list(cells))
            rr, cc2 = cr, cc
            while (rr, cc2) != target_cell:
                step_choices = []
                if rr != target_cell[0]:
                    step_choices.append((rr + (1 if target_cell[0] > rr else -1), cc2))
                if cc2 != target_cell[1]:
                    step_choices.append((rr, cc2 + (1 if target_cell[1] > cc2 else -1)))
                rr, cc2 = random.choice(step_choices)
                cells.add((rr, cc2))

        mask = np.zeros((side, side), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if len(_component_coords(mask)) != 1:
            continue
        variants = _rotation_variants(mask)
        if any(np.array_equal(variants[0], other) for other in variants[1:]):
            continue
        if not all(bool(mask[0, :].any()) and bool(mask[-1, :].any()) and bool(mask[:, 0].any()) and bool(mask[:, -1].any()) for _ in [0]):
            continue
        return mask
    raise RuntimeError("failed to generate a non-symmetric sprite")


def _trace_path(
    start: tuple[int, int],
    initial_dir: tuple[int, int],
    turns: list[int],
    lengths: list[int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[int]]:
    dirs = [initial_dir]
    current_dir = initial_dir
    for turn in turns:
        idx = DIR_TO_IDX[current_dir]
        current_dir = DIRS[(idx + turn) % 4]
        dirs.append(current_dir)

    path = [start]
    current = start
    corners: list[tuple[int, int]] = []
    for seg_idx, (direction, seg_len) in enumerate(zip(dirs, lengths)):
        dr, dc = direction
        for _ in range(seg_len):
            current = (current[0] + dr, current[1] + dc)
            path.append(current)
        if seg_idx < len(dirs) - 1:
            corners.append(current)
    return path, corners, dirs


def _pairwise_nonoverlapping_boxes(boxes: list[tuple[int, int, int, int]]) -> bool:
    for i, (r0, c0, r1, c1) in enumerate(boxes):
        for rr0, cc0, rr1, cc1 in boxes[i + 1 :]:
            if not (r1 < rr0 or rr1 < r0 or c1 < cc0 or cc1 < c0):
                return False
    return True


def _ordered_turns_from_dirs(dirs: list[tuple[int, int]]) -> list[int]:
    out: list[int] = []
    for prev_dir, next_dir in zip(dirs, dirs[1:]):
        diff = (DIR_TO_IDX[next_dir] - DIR_TO_IDX[prev_dir]) % 4
        assert diff in (1, 3)
        out.append(diff)
    return out


def _render_case(
    shape: tuple[int, int],
    sprite_mask: np.ndarray,
    sprite_color: int,
    sprite_offset: tuple[int, int],
    path: list[tuple[int, int]],
    stamp_centers: list[tuple[int, int]],
    turns: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    sr, sc = sprite_offset
    side = sprite_mask.shape[0]
    sprite_patch = np.where(sprite_mask, sprite_color, BG)
    grid[sr : sr + side, sc : sc + side][sprite_mask] = sprite_color
    for r, c in path:
        grid[int(r), int(c)] = PATH

    out = np.array(grid, copy=True)
    out[out == PATH] = BG
    rot = 0
    for center, diff in zip(stamp_centers[:-1], turns):
        if diff == 3:
            rot = (rot + 1) % 4
        else:
            rot = (rot - 1) % 4
        _stamp(out, sprite_patch, center, sprite_color, rot)
    _stamp(out, sprite_patch, stamp_centers[-1], sprite_color, rot)
    return grid, out


def _make_unrotated_output(
    grid: np.ndarray,
    sprite_patch: np.ndarray,
    sprite_color: int,
    centers: list[tuple[int, int]],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    out[out == PATH] = BG
    for center in centers:
        _stamp(out, sprite_patch, center, sprite_color, 0)
    return out


def _make_noncumulative_output(
    grid: np.ndarray,
    sprite_patch: np.ndarray,
    sprite_color: int,
    centers: list[tuple[int, int]],
    turns: list[int],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    out[out == PATH] = BG
    for center, diff in zip(centers[:-1], turns):
        rot = 1 if diff == 3 else 3
        _stamp(out, sprite_patch, center, sprite_color, rot)
    final_rot = 1 if turns and turns[-1] == 3 else 3 if turns else 0
    _stamp(out, sprite_patch, centers[-1], sprite_color, final_rot)
    return out


def _generate_case(return_meta: bool = False):
    for _ in range(MAX_GENERATION_ATTEMPTS):
        sprite_mask = _sample_sprite_mask()
        side = int(sprite_mask.shape[0])
        sprite_color = random.choice(SPRITE_COLORS)
        start_pixel, start_cell, initial_dir = random.choice(_edge_start_options(sprite_mask))

        turn_count = random.choices(TURN_COUNT_CHOICES, weights=TURN_COUNT_WEIGHTS)[0]
        turns = [random.choice([-1, 1]) for _ in range(turn_count)]
        min_seg = side + 2
        max_seg = side + 5
        lengths = [random.randint(min_seg, max_seg) for _ in range(turn_count + 1)]
        path_rel, corners_rel, dirs = _trace_path(start_cell, initial_dir, turns, lengths)

        if len(set(path_rel)) != len(path_rel):
            continue
        path_set = set(path_rel)
        degrees = {
            p: sum((nr, nc) in path_set for nr, nc in _neighbors4(*p))
            for p in path_set
        }
        deg_values = sorted(degrees.values())
        if deg_values != [1, 1] + [2] * (len(path_set) - 2):
            continue

        sprite_cells = set(map(tuple, np.argwhere(sprite_mask)))
        if any(
            any((nr, nc) in sprite_cells for nr, nc in _neighbors4(*p))
            for p in path_rel[1:]
        ):
            continue

        stamp_centers_rel = corners_rel + [path_rel[-1]]
        half = side // 2
        sprite_box = (0, 0, side - 1, side - 1)
        stamp_boxes = [
            (r - half, c - half, r + half, c + half)
            for r, c in stamp_centers_rel
        ]
        if not _pairwise_nonoverlapping_boxes([sprite_box] + stamp_boxes):
            continue

        all_boxes = [sprite_box] + stamp_boxes
        min_r = min(min(r for r, _ in path_rel), min(b[0] for b in all_boxes))
        max_r = max(max(r for r, _ in path_rel), max(b[2] for b in all_boxes))
        min_c = min(min(c for _, c in path_rel), min(b[1] for b in all_boxes))
        max_c = max(max(c for _, c in path_rel), max(b[3] for b in all_boxes))

        scene_h = max_r - min_r + 1
        scene_w = max_c - min_c + 1
        if scene_h > 28 or scene_w > 28:
            continue

        extra_h = random.randint(2, 30 - scene_h) if scene_h <= 28 else 0
        extra_w = random.randint(2, 30 - scene_w) if scene_w <= 28 else 0
        top = random.randint(1, extra_h - 1)
        left = random.randint(1, extra_w - 1)
        bottom = extra_h - top
        right = extra_w - left
        height = scene_h + top + bottom
        width = scene_w + left + right

        shift_r = top - min_r
        shift_c = left - min_c
        sprite_offset = (shift_r, shift_c)
        path = [(r + shift_r, c + shift_c) for r, c in path_rel]
        stamp_centers = [(r + shift_r, c + shift_c) for r, c in stamp_centers_rel]
        diff_turns = _ordered_turns_from_dirs(dirs)

        grid, expected = _render_case(
            (height, width),
            sprite_mask,
            sprite_color,
            sprite_offset,
            path,
            stamp_centers,
            diff_turns,
        )

        sprite_patch = np.where(sprite_mask, sprite_color, BG)
        if turn_count > 0:
            if np.array_equal(expected, _make_unrotated_output(grid, sprite_patch, sprite_color, stamp_centers)):
                continue
        if turn_count > 1:
            if np.array_equal(expected, _make_noncumulative_output(grid, sprite_patch, sprite_color, stamp_centers, diff_turns)):
                continue
        if not np.array_equal(main(grid), expected):
            continue

        meta = {
            "turn_count": turn_count,
            "turns": diff_turns,
            "shape": grid.shape,
            "sprite_side": side,
        }
        return (grid, meta) if return_meta else grid
    raise RuntimeError("failed to generate a valid corner-stamp input")


def generate_input() -> np.ndarray:
    return _generate_case(return_meta=False)
