from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BG = 0
SEPARATOR = 5
TRAIL = 7
OBJECT_COLORS = [1, 2, 3, 4, 6, 8, 9]

PANEL_H_RANGE = (8, 12)
PANEL_W_RANGE = (9, 15)
OBJECT_COUNT_RANGE = (3, 5)
MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)


def find_components(grid: np.ndarray, *, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for color in sorted(int(v) for v in np.unique(grid) if int(v) != int(bg)):
        for r0, c0 in np.argwhere(grid == color):
            r0 = int(r0)
            c0 = int(c0)
            if seen[r0, c0]:
                continue

            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[Coord] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            bbox = bbox_coords(arr)
            patch = crop_bbox(grid, bbox)
            mask = patch == color
            objects.append(
                {
                    "color": color,
                    "coords": arr,
                    "coord_set": {(int(r), int(c)) for r, c in coords},
                    "bbox": bbox,
                    "sig": (color, patch.shape, mask.tobytes()),
                }
            )

    objects.sort(key=lambda obj: (obj["color"], obj["bbox"]))
    return objects


def _find_separator_row(grid: np.ndarray) -> int:
    h, _w = grid.shape
    candidates = [r for r in range(h) if r == h - r - 1 and np.all(grid[r] == int(grid[r, 0]))]
    assert len(candidates) == 1, "expected one centered solid divider row"
    return int(candidates[0])


def _translate_set(coords: set[Coord], dr: int, dc: int) -> set[Coord]:
    return {(r + dr, c + dc) for r, c in coords}


def _movement_between(src: dict, dst: dict) -> tuple[int, int] | None:
    sr0, sc0, _sr1, _sc1 = src["bbox"]
    dr0, dc0, _dr1, _dc1 = dst["bbox"]
    move_r = int(dr0 - sr0)
    move_c = int(dc0 - sc0)
    if move_r != 0 and move_c != 0:
        return None
    if _translate_set(src["coord_set"], move_r, move_c) != dst["coord_set"]:
        return None
    return move_r, move_c


def _sweep_cells(coord_set: set[Coord], step_r: int, step_c: int, steps: int, shape: tuple[int, int]) -> set[Coord]:
    h, w = shape
    swept: set[Coord] = set()
    for t in range(steps):
        for r, c in coord_set:
            rr = r + t * step_r
            cc = c + t * step_c
            if 0 <= rr < h and 0 <= cc < w:
                swept.add((rr, cc))
    return swept


def _nearest_exit_direction(coord_set: set[Coord], shape: tuple[int, int]) -> tuple[Coord, int] | None:
    h, w = shape
    rows = [r for r, _c in coord_set]
    cols = [c for _r, c in coord_set]
    distances = {
        (-1, 0): max(rows) + 1,
        (1, 0): h - min(rows),
        (0, -1): max(cols) + 1,
        (0, 1): w - min(cols),
    }
    best = min(distances.values())
    dirs = [direction for direction, distance in distances.items() if distance == best]
    if len(dirs) != 1:
        return None
    return dirs[0], int(best)


def _infer_exit_direction(coord_set: set[Coord], shape: tuple[int, int]) -> tuple[Coord, int]:
    result = _nearest_exit_direction(coord_set, shape)
    assert result is not None, "missing object must have a unique nearest exit edge"
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare the upper and lower snapshot panels, then draw the later panel with orange movement residue.

    Input structure:
    - A single solid divider row splits the grid into two equal-height panels.
    - Each panel shows monochrome connected objects on the same background color.
    - Most objects appear in both panels with the same shape and color; some objects appear only above.

    Rule:
    - Remove the divider and keep only the lower panel size as the output canvas.
    - For each object that appears in both panels, place it at the lower-panel position and keep its color.
    - Paint orange every cell swept during the straight horizontal or vertical slide from the upper position to the lower one, except cells occupied in the final lower snapshot.
    - If an object appears only in the upper panel, treat it as having continued straight to the unique nearest grid edge and paint that full in-bounds swept trail orange without keeping the object itself.

    Key insight:
    - The two halves are before/after frames of the same scene: the answer is the later frame plus motion trails, not a direct overlay or difference mask.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep = _find_separator_row(g)
    top = g[:sep]
    bottom = g[sep + 1 :]
    assert top.shape == bottom.shape

    bg = int(get_bg(np.vstack([top, bottom])))
    top_objects = find_components(top, bg=bg)
    bottom_objects = find_components(bottom, bg=bg)

    bottom_by_sig: dict[tuple, list[dict]] = defaultdict(list)
    for obj in bottom_objects:
        bottom_by_sig[obj["sig"]].append(obj)

    matched_bottom: set[int] = set()
    final_occupied = {(int(r), int(c)) for r, c in np.argwhere(bottom != bg)}
    trail_cells: set[Coord] = set()

    for top_obj in top_objects:
        candidates: list[tuple[int, dict, int, int]] = []
        for idx, bottom_obj in enumerate(bottom_by_sig[top_obj["sig"]]):
            token = id(bottom_obj)
            if token in matched_bottom:
                continue
            movement = _movement_between(top_obj, bottom_obj)
            if movement is None:
                continue
            candidates.append((token, bottom_obj, movement[0], movement[1]))

        if len(candidates) == 1:
            token, _bottom_obj, move_r, move_c = candidates[0]
            matched_bottom.add(token)
            steps = abs(move_r) + abs(move_c)
            if steps:
                step_r = 0 if move_r == 0 else int(move_r / abs(move_r))
                step_c = 0 if move_c == 0 else int(move_c / abs(move_c))
                trail_cells |= _sweep_cells(top_obj["coord_set"], step_r, step_c, steps, top.shape)
            continue

        assert not candidates, "object matching must be unambiguous"
        direction, steps = _infer_exit_direction(top_obj["coord_set"], top.shape)
        trail_cells |= _sweep_cells(top_obj["coord_set"], direction[0], direction[1], steps, top.shape)

    assert len(matched_bottom) == len(bottom_objects), "every lower-panel object must have a top-panel source"

    out = bottom.copy()
    for r, c in trail_cells:
        if (r, c) not in final_occupied:
            out[r, c] = TRAIL
    return out


def _shape_coords(mask: np.ndarray) -> list[Coord]:
    return [(int(r), int(c)) for r, c in np.argwhere(mask)]


def _paint_mask(grid: np.ndarray, mask: np.ndarray, origin: Coord, color: int) -> None:
    r0, c0 = origin
    h, w = mask.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = int(color)


def _absolute_cells(mask: np.ndarray, origin: Coord) -> set[Coord]:
    r0, c0 = origin
    return {(r0 + r, c0 + c) for r, c in _shape_coords(mask)}


def _halo(cells: set[Coord], shape: tuple[int, int], pad: int = 1) -> set[Coord]:
    h, w = shape
    out: set[Coord] = set()
    for r, c in cells:
        for rr in range(max(0, r - pad), min(h, r + pad + 1)):
            for cc in range(max(0, c - pad), min(w, c + pad + 1)):
                out.add((rr, cc))
    return out


def _random_shape() -> np.ndarray:
    box_h = random.randint(1, 3)
    box_w = random.randint(1, 3)
    target = random.choices([1, 2, 3, 4, 5], weights=[1, 3, 3, 2, 1])[0]
    target = min(target, box_h * box_w)

    start = (random.randrange(box_h), random.randrange(box_w))
    cells = {start}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return _random_shape()
        cells.add(random.choice(frontier))

    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    mask = np.zeros((max(rows) - min(rows) + 1, max(cols) - min(cols) + 1), dtype=bool)
    for r, c in cells:
        mask[r - min(rows), c - min(cols)] = True
    return mask


def _state_list(count: int) -> list[str]:
    states = ["move"]
    if count >= 2 and random.random() < 0.7:
        states.append("exit")
    if count >= 3 and random.random() < 0.6:
        states.append("static")
    while len(states) < count:
        states.append(random.choices(["move", "static", "exit"], weights=[5, 2, 3])[0])
    while states.count("exit") > 2:
        states[states.index("exit")] = "move"
    random.shuffle(states)
    return states


def _candidate_specs(
    mask: np.ndarray,
    state: str,
    panel_shape: tuple[int, int],
    top_blocked: set[Coord],
    bottom_blocked: set[Coord],
    output_occ: set[Coord],
) -> list[dict]:
    panel_h, panel_w = panel_shape
    mask_h, mask_w = mask.shape
    coords_rel = _shape_coords(mask)
    candidates: list[dict] = []

    for r0 in range(panel_h - mask_h + 1):
        for c0 in range(panel_w - mask_w + 1):
            top_origin = (r0, c0)
            top_cells = {(r0 + r, c0 + c) for r, c in coords_rel}
            if top_cells & top_blocked:
                continue

            if state == "static":
                bottom_cells = top_cells
                if bottom_cells & bottom_blocked:
                    continue
                if bottom_cells & output_occ:
                    continue
                candidates.append(
                    {
                        "top": top_origin,
                        "bottom": top_origin,
                        "top_cells": top_cells,
                        "bottom_cells": bottom_cells,
                        "paint_cells": set(),
                    }
                )
                continue

            if state == "move":
                for step_r, step_c in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    max_steps = panel_h if step_r else panel_w
                    for steps in range(1, max_steps):
                        br = r0 + step_r * steps
                        bc = c0 + step_c * steps
                        if not (0 <= br and 0 <= bc and br + mask_h <= panel_h and bc + mask_w <= panel_w):
                            continue
                        bottom_origin = (br, bc)
                        bottom_cells = {(br + r, bc + c) for r, c in coords_rel}
                        if bottom_cells & bottom_blocked:
                            continue
                        swept = _sweep_cells(top_cells, step_r, step_c, steps, panel_shape)
                        paint_cells = swept - bottom_cells
                        if paint_cells & output_occ:
                            continue
                        if bottom_cells & output_occ:
                            continue
                        candidates.append(
                            {
                                "top": top_origin,
                                "bottom": bottom_origin,
                                "top_cells": top_cells,
                                "bottom_cells": bottom_cells,
                                "paint_cells": paint_cells,
                            }
                        )
                continue

            exit_info = _nearest_exit_direction(top_cells, panel_shape)
            if exit_info is None:
                continue
            direction, steps = exit_info
            if steps <= 1:
                continue
            swept = _sweep_cells(top_cells, direction[0], direction[1], steps, panel_shape)
            if swept & output_occ:
                continue
            candidates.append(
                {
                    "top": top_origin,
                    "bottom": None,
                    "top_cells": top_cells,
                    "bottom_cells": set(),
                    "paint_cells": swept,
                    "exit_dir": direction,
                    "exit_steps": steps,
                }
            )

    return candidates


def _render_scene(panel_shape: tuple[int, int], specs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    panel_h, panel_w = panel_shape
    top = np.full((panel_h, panel_w), BG, dtype=int)
    bottom = np.full((panel_h, panel_w), BG, dtype=int)

    for spec in specs:
        _paint_mask(top, spec["mask"], spec["top"], spec["color"])
        if spec["bottom"] is not None:
            _paint_mask(bottom, spec["mask"], spec["bottom"], spec["color"])

    final_occupied = {(int(r), int(c)) for r, c in np.argwhere(bottom != BG)}
    out = bottom.copy()
    for spec in specs:
        for r, c in spec["paint_cells"]:
            if (r, c) not in final_occupied:
                out[r, c] = TRAIL

    sep = np.full((1, panel_w), SEPARATOR, dtype=int)
    return np.vstack([top, sep, bottom]), out


def _attempt_generate() -> np.ndarray | None:
    panel_shape = (random.randint(*PANEL_H_RANGE), random.randint(*PANEL_W_RANGE))
    object_count = random.randint(*OBJECT_COUNT_RANGE)
    states = _state_list(object_count)
    colors = random.sample(OBJECT_COLORS, object_count)

    order = list(range(object_count))
    order.sort(key=lambda idx: {"exit": 0, "move": 1, "static": 2}[states[idx]])

    top_blocked: set[Coord] = set()
    bottom_blocked: set[Coord] = set()
    output_occ: set[Coord] = set()
    specs: list[dict | None] = [None] * object_count

    for idx in order:
        state = states[idx]
        mask = _random_shape()
        candidates = _candidate_specs(mask, state, panel_shape, top_blocked, bottom_blocked, output_occ)
        if not candidates:
            return None
        choice = random.choice(candidates)
        spec = {
            "color": colors[idx],
            "mask": mask,
            "state": state,
            **choice,
        }
        specs[idx] = spec
        top_blocked |= _halo(spec["top_cells"], panel_shape)
        if spec["bottom"] is not None:
            bottom_blocked |= _halo(spec["bottom_cells"], panel_shape)
            output_occ |= spec["bottom_cells"]
        output_occ |= spec["paint_cells"]

    concrete_specs = [spec for spec in specs if spec is not None]
    inp, expected = _render_scene(panel_shape, concrete_specs)

    sep = _find_separator_row(inp)
    top = inp[:sep]
    bottom = inp[sep + 1 :]
    if top.shape != bottom.shape:
        return None
    if len(find_components(top, bg=BG)) != object_count:
        return None
    if len(find_components(bottom, bg=BG)) != sum(spec["bottom"] is not None for spec in concrete_specs):
        return None
    for spec in concrete_specs:
        if spec["state"] == "exit":
            direction, steps = _infer_exit_direction(spec["top_cells"], panel_shape)
            if direction != spec["exit_dir"] or steps != spec["exit_steps"]:
                return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid motion-trail input")
