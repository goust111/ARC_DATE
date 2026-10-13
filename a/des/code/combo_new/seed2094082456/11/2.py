from __future__ import annotations

from collections import deque
import random

import numpy as np


DIRS: tuple[str, ...] = ("up", "down", "left", "right")
DIR_STEPS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ARROW_PATTERNS: dict[str, tuple[np.ndarray, ...]] = {
    "right": (
        np.array([[0, 1], [1, 1]], dtype=bool),
        np.array([[1, 1], [0, 1]], dtype=bool),
    ),
    "left": (
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1], [1, 0]], dtype=bool),
    ),
    "down": (
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[0, 1], [1, 1]], dtype=bool),
    ),
    "up": (
        np.array([[1, 1], [1, 0]], dtype=bool),
        np.array([[1, 1], [0, 1]], dtype=bool),
    ),
}

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_PADDING = 1
SIZE_MIN_BY_COUNT = {2: 12, 3: 15, 4: 18}
SIZE_MAX = 28
PALETTE = tuple(range(10))


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _mask_to_cells(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple((int(r), int(c)) for r, c in np.argwhere(mask))


def _shift_cells(cells: tuple[tuple[int, int], ...], dr: int, dc: int) -> tuple[tuple[int, int], ...]:
    return tuple((int(r + dr), int(c + dc)) for r, c in cells)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _find_components(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    owner = np.full((h, w), -1, dtype=int)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                cr, cc = queue.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            arr = np.array(coords, dtype=int)
            cid = len(components)
            owner[arr[:, 0], arr[:, 1]] = cid
            components.append(
                {
                    "id": cid,
                    "color": color,
                    "coords": tuple((int(rr), int(cc)) for rr, cc in coords),
                    "bbox": _bbox(arr),
                    "area": int(arr.shape[0]),
                }
            )

    return components, owner


def _is_arrow_component(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    return int(component["area"]) == 3 and int(r1 - r0) == 1 and int(c1 - c0) == 1


def _target_is_on_side(arrow_bbox: tuple[int, int, int, int], target_bbox: tuple[int, int, int, int], direction: str) -> bool:
    ar0, ac0, ar1, ac1 = arrow_bbox
    tr0, tc0, tr1, tc1 = target_bbox
    if direction == "up":
        return tr1 < ar0 and not (tc1 < ac0 or tc0 > ac1)
    if direction == "down":
        return tr0 > ar1 and not (tc1 < ac0 or tc0 > ac1)
    if direction == "left":
        return tc1 < ac0 and not (tr1 < ar0 or tr0 > ar1)
    if direction == "right":
        return tc0 > ac1 and not (tr1 < ar0 or tr0 > ar1)
    raise AssertionError(direction)


def _first_hits_by_direction(owner: np.ndarray, bbox: tuple[int, int, int, int], direction: str) -> list[int]:
    h, w = owner.shape
    r0, c0, r1, c1 = bbox
    hits: list[int] = []

    if direction == "up":
        for c in range(c0, c1 + 1):
            rr = r0 - 1
            while rr >= 0 and int(owner[rr, c]) == -1:
                rr -= 1
            hits.append(-1 if rr < 0 else int(owner[rr, c]))
    elif direction == "down":
        for c in range(c0, c1 + 1):
            rr = r1 + 1
            while rr < h and int(owner[rr, c]) == -1:
                rr += 1
            hits.append(-1 if rr >= h else int(owner[rr, c]))
    elif direction == "left":
        for r in range(r0, r1 + 1):
            cc = c0 - 1
            while cc >= 0 and int(owner[r, cc]) == -1:
                cc -= 1
            hits.append(-1 if cc < 0 else int(owner[r, cc]))
    elif direction == "right":
        for r in range(r0, r1 + 1):
            cc = c1 + 1
            while cc < w and int(owner[r, cc]) == -1:
                cc += 1
            hits.append(-1 if cc >= w else int(owner[r, cc]))
    else:
        raise AssertionError(direction)

    return hits


def _match_arrow(arrow: dict, components: list[dict], owner: np.ndarray, target_ids: set[int]) -> tuple[str, int]:
    candidates: list[tuple[str, int]] = []
    for direction in DIRS:
        hits = _first_hits_by_direction(owner, arrow["bbox"], direction)
        nonempty = [cid for cid in hits if cid != -1]
        if not nonempty:
            continue
        if any(cid not in target_ids for cid in nonempty):
            continue
        unique = set(nonempty)
        if len(unique) != 1:
            continue
        target_id = next(iter(unique))
        if _target_is_on_side(arrow["bbox"], components[target_id]["bbox"], direction):
            candidates.append((direction, int(target_id)))

    assert len(candidates) == 1, f"expected one arrow-target match, got {candidates}"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each L-shaped arrow away from its target and stamp a boundary copy in the arrow color.

    Input structure:
    - The background is the dominant color.
    - Several monochrome irregular target objects of distinct colors are scattered in the grid.
    - Each target has one adjacent 3-pixel L marker of a different unique color placed directly above, below, left, or right of it.

    Rule:
    - Identify every 3-pixel 2x2-bbox L component as an arrow marker and pair it with the unique target component found in the direction the marker faces.
    - Remove the arrow pixels, then take the matched target shape and project a recolored copy in the opposite direction until that copy's bbox touches the corresponding grid edge.
    - Keep the original target objects unchanged and preserve the copied shape's orientation.

    Key insight:
    - The marker's side relative to the object identifies both which object to copy and which opposite edge receives the recolored copy.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    components, owner = _find_components(g, bg)

    arrows = [comp for comp in components if _is_arrow_component(comp)]
    targets = [comp for comp in components if not _is_arrow_component(comp)]
    assert arrows and targets, "expected both arrows and targets"

    target_ids = {int(comp["id"]) for comp in targets}
    out = g.copy()

    for arrow in arrows:
        for r, c in arrow["coords"]:
            out[r, c] = bg

    h, w = g.shape
    for arrow in arrows:
        direction, target_id = _match_arrow(arrow, components, owner, target_ids)
        target = components[target_id]
        tr0, tc0, tr1, tc1 = target["bbox"]
        dr = 0
        dc = 0

        if direction == "up":
            dr = h - 1 - tr1
        elif direction == "down":
            dr = -tr0
        elif direction == "left":
            dc = w - 1 - tc1
        elif direction == "right":
            dc = -tc0
        else:
            raise AssertionError(direction)

        for r, c in target["coords"]:
            out[int(r + dr), int(c + dc)] = int(arrow["color"])

    return out


def _random_shape() -> np.ndarray:
    for _ in range(200):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        max_area = min(h * w - 1, 7)
        if max_area < 4:
            continue
        area = random.randint(4, max_area)
        cells = {(random.randrange(h), random.randrange(w))}

        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        mask = mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if bool(np.all(mask)):
            continue
        if int(mask.sum()) == 3 and mask.shape == (2, 2):
            continue
        return mask

    raise RuntimeError("failed to sample a target shape")


def _valid_arrow_offsets(mask: np.ndarray, direction: str) -> list[int]:
    h, w = mask.shape
    offsets: list[int] = []
    if direction == "right":
        for r0 in range(h - 1):
            if bool(mask[r0, 0] or mask[r0 + 1, 0]):
                offsets.append(r0)
    elif direction == "left":
        for r0 in range(h - 1):
            if bool(mask[r0, w - 1] or mask[r0 + 1, w - 1]):
                offsets.append(r0)
    elif direction == "down":
        for c0 in range(w - 1):
            if bool(mask[0, c0] or mask[0, c0 + 1]):
                offsets.append(c0)
    elif direction == "up":
        for c0 in range(w - 1):
            if bool(mask[h - 1, c0] or mask[h - 1, c0 + 1]):
                offsets.append(c0)
    else:
        raise AssertionError(direction)
    return offsets


def _expanded(cells: tuple[tuple[int, int], ...], h: int, w: int, pad: int = PLACEMENT_PADDING) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - pad), min(h, r + pad + 1)):
            for cc in range(max(0, c - pad), min(w, c + pad + 1)):
                out.add((int(rr), int(cc)))
    return out


def _separated(a: tuple[tuple[int, int], ...], b: tuple[tuple[int, int], ...], h: int, w: int, pad: int = PLACEMENT_PADDING) -> bool:
    return set(b).isdisjoint(_expanded(a, h, w, pad=pad))


def _placement_candidates(mask: np.ndarray, direction: str, grid_h: int, grid_w: int) -> list[dict]:
    h, w = mask.shape
    shape_cells = _mask_to_cells(mask)
    arrow_offsets = _valid_arrow_offsets(mask, direction)
    candidates: list[dict] = []

    for tr in range(grid_h - h + 1):
        for tc in range(grid_w - w + 1):
            if direction == "right":
                copy_top = (tr, 0)
            elif direction == "left":
                copy_top = (tr, grid_w - w)
            elif direction == "down":
                copy_top = (0, tc)
            elif direction == "up":
                copy_top = (grid_h - h, tc)
            else:
                raise AssertionError(direction)

            target_cells = _shift_cells(shape_cells, tr, tc)
            copy_cells = _shift_cells(shape_cells, copy_top[0], copy_top[1])
            if not set(target_cells).isdisjoint(copy_cells):
                continue
            if not _separated(target_cells, copy_cells, grid_h, grid_w):
                continue

            for offset in arrow_offsets:
                if direction == "right":
                    arrow_top = (tr + offset, tc - 2)
                elif direction == "left":
                    arrow_top = (tr + offset, tc + w)
                elif direction == "down":
                    arrow_top = (tr - 2, tc + offset)
                else:
                    arrow_top = (tr + h, tc + offset)

                ar, ac = arrow_top
                if ar < 0 or ac < 0 or ar + 2 > grid_h or ac + 2 > grid_w:
                    continue

                for arrow_mask in ARROW_PATTERNS[direction]:
                    arrow_cells = _shift_cells(_mask_to_cells(arrow_mask), ar, ac)
                    if not set(arrow_cells).isdisjoint(target_cells):
                        continue
                    candidates.append(
                        {
                            "direction": direction,
                            "target_cells": target_cells,
                            "copy_cells": copy_cells,
                            "arrow_cells": arrow_cells,
                        }
                    )

    random.shuffle(candidates)
    return candidates


def _paint(grid: np.ndarray, cells: tuple[tuple[int, int], ...], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _validate_generated(inp: np.ndarray, expected: np.ndarray, n_objects: int) -> bool:
    try:
        solved = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(solved, expected):
        return False
    bg = _bg_color(inp)
    components, _owner = _find_components(inp, bg)
    arrows = [comp for comp in components if _is_arrow_component(comp)]
    targets = [comp for comp in components if not _is_arrow_component(comp)]
    if len(arrows) != n_objects or len(targets) != n_objects:
        return False
    arrow_colors = {int(comp["color"]) for comp in arrows}
    target_colors = {int(comp["color"]) for comp in targets}
    return len(arrow_colors) == n_objects and len(target_colors) == n_objects and arrow_colors.isdisjoint(target_colors)


def _generate_attempt() -> np.ndarray | None:
    n_objects = random.randint(2, 4)
    min_side = SIZE_MIN_BY_COUNT[n_objects]
    grid_h = random.randint(min_side, SIZE_MAX)
    grid_w = random.randint(min_side, SIZE_MAX)
    bg = random.choice(PALETTE)
    usable = [color for color in PALETTE if color != bg]
    target_colors = random.sample(usable, n_objects)
    remaining = [color for color in usable if color not in target_colors]
    arrow_colors = random.sample(remaining, n_objects)

    inp = np.full((grid_h, grid_w), bg, dtype=int)
    expected = np.full((grid_h, grid_w), bg, dtype=int)

    input_forbidden: set[tuple[int, int]] = set()
    output_forbidden: set[tuple[int, int]] = set()

    directions = [random.choice(DIRS) for _ in range(n_objects)]
    random.shuffle(directions)

    for target_color, arrow_color, direction in zip(target_colors, arrow_colors, directions):
        placed = False
        for _ in range(80):
            shape = _random_shape()
            for candidate in _placement_candidates(shape, direction, grid_h, grid_w):
                input_cells = candidate["target_cells"] + candidate["arrow_cells"]
                if any(cell in input_forbidden for cell in input_cells):
                    continue
                if any(cell in output_forbidden for cell in candidate["target_cells"]):
                    continue
                if any(cell in output_forbidden for cell in candidate["copy_cells"]):
                    continue

                _paint(inp, candidate["target_cells"], target_color)
                _paint(inp, candidate["arrow_cells"], arrow_color)
                _paint(expected, candidate["target_cells"], target_color)
                _paint(expected, candidate["copy_cells"], arrow_color)

                input_forbidden.update(_expanded(input_cells, grid_h, grid_w))
                output_forbidden.update(_expanded(candidate["target_cells"], grid_h, grid_w))
                output_forbidden.update(_expanded(candidate["copy_cells"], grid_h, grid_w))
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    if _validate_generated(inp, expected, n_objects):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-projection puzzle")
