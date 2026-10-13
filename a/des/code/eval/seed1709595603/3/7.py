from __future__ import annotations

import random

import numpy as np


FRAME_COLOR = 8
EXIT_SIDES = ("up", "down", "left", "right")
OPPOSITE_SIDE = {"up": "bottom", "down": "top", "left": "right", "right": "left"}
SIDE_TO_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
TURN_SIDES = {"left", "top"}


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors(r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, connectivity):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_ring(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    ring: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        ring.add((r0, c))
        ring.add((r1, c))
    for r in range(r0, r1 + 1):
        ring.add((r, c0))
        ring.add((r, c1))
    return ring


def _detect_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for comp in _components(grid == FRAME_COLOR, connectivity=4):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
            continue
        ring = _frame_ring(r0, c0, r1, c1)
        comp_set = set(comp)
        if comp_set != ring:
            continue
        if not np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == bg):
            continue
        frames.append({"bbox": (r0, c0, r1, c1), "cells": ring})
    frames.sort(key=lambda item: item["bbox"])
    assert frames, "expected at least one Azure envelope"
    return frames


def _find_connection(grid: np.ndarray, frame: dict, bg: int) -> tuple[str, tuple[int, int], tuple[int, int]]:
    h, w = grid.shape
    r0, c0, r1, c1 = frame["bbox"]
    candidates: list[tuple[str, tuple[int, int], tuple[int, int]]] = []

    for c in range(c0 + 1, c1):
        if r0 > 0 and grid[r0 - 1, c] != bg and grid[r0 - 1, c] != FRAME_COLOR:
            candidates.append(("top", (r0, c), (r0 - 1, c)))
        if r1 + 1 < h and grid[r1 + 1, c] != bg and grid[r1 + 1, c] != FRAME_COLOR:
            candidates.append(("bottom", (r1, c), (r1 + 1, c)))
    for r in range(r0 + 1, r1):
        if c0 > 0 and grid[r, c0 - 1] != bg and grid[r, c0 - 1] != FRAME_COLOR:
            candidates.append(("left", (r, c0), (r, c0 - 1)))
        if c1 + 1 < w and grid[r, c1 + 1] != bg and grid[r, c1 + 1] != FRAME_COLOR:
            candidates.append(("right", (r, c1), (r, c1 + 1)))

    assert len(candidates) == 1, "each envelope must have exactly one guide-line connection"
    return candidates[0]


def _orthogonal_dirs(dr: int, dc: int) -> list[tuple[int, int]]:
    if dr == 0:
        return [(-1, 0), (1, 0)]
    return [(0, -1), (0, 1)]


def _collect_line_and_stamp(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    frame_cells: set[tuple[int, int]],
    bg: int,
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    dr, dc = direction
    axis: list[tuple[int, int]] = []
    r, c = start
    while _in_bounds(grid.shape, r, c) and grid[r, c] != bg and (r, c) not in frame_cells:
        axis.append((r, c))
        r += dr
        c += dc

    assert axis, "guide line must extend away from the frame"
    line_color = int(grid[axis[0]])
    stamp_start_idx: int | None = None
    ortho = _orthogonal_dirs(dr, dc)

    for idx, (r, c) in enumerate(axis):
        if int(grid[r, c]) != line_color:
            stamp_start_idx = idx
            break
        branch = False
        for odr, odc in ortho:
            nr, nc = r + odr, c + odc
            if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] != bg and (nr, nc) not in frame_cells:
                branch = True
                break
        if branch:
            stamp_start_idx = idx
            break

    if stamp_start_idx is None:
        line_cells = axis[:-1]
        seed_candidates = [axis[-1]]
    else:
        line_cells = axis[:stamp_start_idx]
        seed_candidates = [axis[stamp_start_idx]]
        sr, sc = axis[stamp_start_idx]
        for odr, odc in ortho:
            nr, nc = sr + odr, sc + odc
            if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] != bg and (nr, nc) not in frame_cells:
                seed_candidates.append((nr, nc))

    assert line_cells, "guide line must have positive length"
    excluded = set(frame_cells) | set(line_cells)
    stamp: set[tuple[int, int]] = set()
    stack = [pt for pt in seed_candidates if pt not in excluded and grid[pt] != bg]
    seen = set(stack)
    while stack:
        r, c = stack.pop()
        stamp.add((r, c))
        for nr, nc in _neighbors(r, c, connectivity=8):
            if (
                _in_bounds(grid.shape, nr, nc)
                and grid[nr, nc] != bg
                and (nr, nc) not in excluded
                and (nr, nc) not in seen
            ):
                seen.add((nr, nc))
                stack.append((nr, nc))

    assert stamp, "failed to isolate the stamp cluster"
    return line_cells, stamp


def _stamp_patch(grid: np.ndarray, stamp_cells: set[tuple[int, int]], bg: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(stamp_cells)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in stamp_cells:
        patch[r - r0, c - c0] = int(grid[r, c])
    return patch


def _placement_in_envelope(
    bbox: tuple[int, int, int, int], patch_shape: tuple[int, int], side: str
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    ph, pw = patch_shape
    assert ph <= inner_h and pw <= inner_w, "stamp must fit inside the envelope"

    if side == "left":
        return r0 + 1 + (inner_h - ph) // 2, c0 + 1
    if side == "right":
        return r0 + 1 + (inner_h - ph) // 2, c1 - pw
    if side == "top":
        return r0 + 1, c0 + 1 + (inner_w - pw) // 2
    if side == "bottom":
        return r1 - ph, c0 + 1 + (inner_w - pw) // 2
    raise ValueError(side)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each multi-color stamp into its connected Azure envelope.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames = _detect_frames(g, bg)

    out = np.full_like(g, bg)
    for frame in frames:
        for r, c in frame["cells"]:
            out[r, c] = FRAME_COLOR

        side, _frame_pt, start = _find_connection(g, frame, bg)
        line_dir = SIDE_TO_DIR[side]
        _line_cells, stamp_cells = _collect_line_and_stamp(g, start, line_dir, frame["cells"], bg)
        patch = _stamp_patch(g, stamp_cells, bg)
        if side in TURN_SIDES:
            patch = np.rot90(patch, 2)

        top, left = _placement_in_envelope(frame["bbox"], patch.shape, side)
        ph, pw = patch.shape
        out[top : top + ph, left : left + pw] = patch

    return out


def _choose_palette(bg: int, count: int) -> list[int]:
    colors = [c for c in range(10) if c not in (bg, FRAME_COLOR)]
    return random.sample(colors, count)


def _attachment(shape: tuple[int, int], side: str) -> tuple[int, int]:
    h, w = shape
    if side == "up":
        return 0, random.randrange(w)
    if side == "down":
        return h - 1, random.randrange(w)
    if side == "left":
        return random.randrange(h), 0
    if side == "right":
        return random.randrange(h), w - 1
    raise ValueError(side)


def _make_stamp(exit_side: str, bg: int) -> tuple[np.ndarray, int, tuple[int, int]]:
    for _ in range(200):
        h = random.randint(2, 3)
        w = random.randint(2, 3)
        area = h * w
        n_colors = random.randint(2, min(3, area))
        palette = _choose_palette(bg, n_colors)
        dominant = palette[0]
        attach = _attachment((h, w), exit_side)

        flat = [dominant] * area
        attach_idx = attach[0] * w + attach[1]
        other_indices = [i for i in range(area) if i != attach_idx]
        random.shuffle(other_indices)
        n_non_dom = random.randint(max(1, n_colors - 1), area - 1)

        chosen = [attach_idx] + other_indices[: n_non_dom - 1]
        assignments = palette[1:].copy()
        while len(assignments) < len(chosen):
            assignments.append(random.choice(palette[1:]))
        random.shuffle(assignments)
        for idx, color in zip(chosen, assignments, strict=False):
            flat[idx] = color
        patch = np.array(flat, dtype=int).reshape(h, w)

        values, counts = np.unique(patch, return_counts=True)
        count_map = {int(v): int(c) for v, c in zip(values, counts, strict=False)}
        if count_map[dominant] != max(count_map.values()):
            continue
        if list(count_map.values()).count(count_map[dominant]) != 1:
            continue
        if patch[attach] == dominant:
            continue
        if np.array_equal(patch, np.rot90(patch, 2)):
            continue
        return patch, dominant, attach

    raise RuntimeError("failed to create a non-degenerate stamp")


def _build_object(bg: int, exit_side: str) -> tuple[np.ndarray, np.ndarray]:
    patch, line_color, attach = _make_stamp(exit_side, bg)
    conn_side = OPPOSITE_SIDE[exit_side]
    out_patch = np.rot90(patch, 2) if exit_side in {"right", "down"} else patch.copy()

    inner_h = out_patch.shape[0] + random.randint(0, 3)
    inner_w = out_patch.shape[1] + random.randint(0, 3)
    env_h = inner_h + 2
    env_w = inner_w + 2
    line_len = random.randint(2, 6)

    if conn_side in {"left", "right"}:
        conn_offset = random.randint(1, env_h - 2)
    else:
        conn_offset = random.randint(1, env_w - 2)

    stamp_h, stamp_w = patch.shape
    if exit_side == "right":
        env_r = attach[0] - conn_offset
        env_c = stamp_w + line_len
        line_coords = [(attach[0], stamp_w + k) for k in range(line_len)]
    elif exit_side == "left":
        env_r = attach[0] - conn_offset
        env_c = -(env_w + line_len)
        line_coords = [(attach[0], -1 - k) for k in range(line_len)]
    elif exit_side == "down":
        env_r = stamp_h + line_len
        env_c = attach[1] - conn_offset
        line_coords = [(stamp_h + k, attach[1]) for k in range(line_len)]
    else:
        env_r = -(env_h + line_len)
        env_c = attach[1] - conn_offset
        line_coords = [(-1 - k, attach[1]) for k in range(line_len)]

    all_points = [(0, 0), (stamp_h - 1, stamp_w - 1), (env_r, env_c), (env_r + env_h - 1, env_c + env_w - 1)]
    all_points.extend(line_coords)
    min_r = min(r for r, _ in all_points)
    min_c = min(c for _, c in all_points)
    max_r = max(r for r, _ in all_points)
    max_c = max(c for _, c in all_points)

    shift_r = -min_r
    shift_c = -min_c
    obj_h = max_r - min_r + 1
    obj_w = max_c - min_c + 1
    inp = np.full((obj_h, obj_w), bg, dtype=int)
    out = np.full((obj_h, obj_w), bg, dtype=int)

    stamp_r = shift_r
    stamp_c = shift_c
    inp[stamp_r : stamp_r + stamp_h, stamp_c : stamp_c + stamp_w] = patch

    for r, c in line_coords:
        inp[r + shift_r, c + shift_c] = line_color

    env_r += shift_r
    env_c += shift_c
    env_bbox = (env_r, env_c, env_r + env_h - 1, env_c + env_w - 1)
    for r, c in _frame_ring(*env_bbox):
        inp[r, c] = FRAME_COLOR
        out[r, c] = FRAME_COLOR

    place_r, place_c = _placement_in_envelope(env_bbox, out_patch.shape, conn_side)
    ph, pw = out_patch.shape
    out[place_r : place_r + ph, place_c : place_c + pw] = out_patch
    return inp, out


def _can_place(occupied: np.ndarray, top: int, left: int, h: int, w: int) -> bool:
    h0, w0 = occupied.shape
    if top < 0 or left < 0 or top + h > h0 or left + w > w0:
        return False
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h0, top + h + 1)
    c1 = min(w0, left + w + 1)
    return not occupied[r0:r1, c0:c1].any()


def _mark_occupied(occupied: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(occupied.shape[0], top + h + 1)
    c1 = min(occupied.shape[1], left + w + 1)
    occupied[r0:r1, c0:c1] = True


def _try_place_objects(objects: list[tuple[np.ndarray, np.ndarray]], bg: int) -> tuple[np.ndarray, np.ndarray] | None:
    max_h = max(inp.shape[0] for inp, _ in objects)
    max_w = max(inp.shape[1] for inp, _ in objects)
    n_objects = len(objects)

    low = max(18, 12 + 4 * n_objects, max_h + 2, max_w + 2)
    high = max(low, 30)
    canvas_h = random.randint(low, high)
    canvas_w = random.randint(low, high)

    inp = np.full((canvas_h, canvas_w), bg, dtype=int)
    out = np.full((canvas_h, canvas_w), bg, dtype=int)
    occupied = np.zeros((canvas_h, canvas_w), dtype=bool)

    for obj_in, obj_out in sorted(objects, key=lambda item: item[0].shape[0] * item[0].shape[1], reverse=True):
        oh, ow = obj_in.shape
        placed = False
        positions = [(r, c) for r in range(canvas_h - oh + 1) for c in range(canvas_w - ow + 1)]
        random.shuffle(positions)
        for top, left in positions[:300]:
            if not _can_place(occupied, top, left, oh, ow):
                continue
            view_in = inp[top : top + oh, left : left + ow]
            view_out = out[top : top + oh, left : left + ow]
            mask_in = obj_in != bg
            mask_out = obj_out != bg
            view_in[mask_in] = obj_in[mask_in]
            view_out[mask_out] = obj_out[mask_out]
            _mark_occupied(occupied, top, left, oh, ow)
            placed = True
            break
        if not placed:
            return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(400):
        bg = random.choice([c for c in range(10) if c != FRAME_COLOR])
        n_objects = random.randint(1, 4)
        exits = [random.choice(EXIT_SIDES) for _ in range(n_objects)]
        objects = [_build_object(bg, exit_side) for exit_side in exits]
        placed = _try_place_objects(objects, bg)
        if placed is None:
            continue
        inp, expected = placed
        if np.array_equal(main(inp), expected):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
