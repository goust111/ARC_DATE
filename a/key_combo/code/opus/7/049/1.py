from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
CONTOUR_COLOR = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 8, 9)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.asarray(cells, dtype=int)
            out.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def _can_shift_down(coords: np.ndarray, dr: int, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    shifted = coords + np.array([dr, 0], dtype=int)
    if not bool(np.all((0 <= shifted[:, 0]) & (shifted[:, 0] < h) & (0 <= shifted[:, 1]) & (shifted[:, 1] < w))):
        return False
    return not bool(occupied[shifted[:, 0], shifted[:, 1]].any())


def _drop_contours(contours: list[dict], markers: list[tuple[int, int, int]], shape: tuple[int, int]) -> list[dict]:
    occupied = np.zeros(shape, dtype=bool)
    for r, c, _color in markers:
        occupied[int(r), int(c)] = True

    placed: list[dict] = []
    order = sorted(contours, key=lambda comp: (int(comp["bbox"][2]), int(comp["bbox"][0]), int(comp["bbox"][1])), reverse=True)
    for comp in order:
        dr = 0
        while _can_shift_down(comp["coords"], dr + 1, occupied):
            dr += 1
        landed = comp["coords"] + np.array([dr, 0], dtype=int)
        occupied[landed[:, 0], landed[:, 1]] = True
        placed.append(
            {
                "id": comp["id"],
                "color": int(comp["color"]),
                "coords": landed,
                "bbox": _bbox(landed),
                "area": int(comp["area"]),
                "drop": dr,
            }
        )
    return placed


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.asarray(cells, dtype=int))
    return out


def _local_cavity_regions(grid: np.ndarray, contour: dict) -> list[np.ndarray]:
    r0, c0, r1, c1 = contour["bbox"]
    local = grid[r0 : r1 + 1, c0 : c1 + 1]
    hh, ww = local.shape
    wall = np.zeros((hh, ww), dtype=bool)
    rel = contour["coords"] - np.array([r0, c0], dtype=int)
    wall[rel[:, 0], rel[:, 1]] = True
    passable = ~wall

    seen = np.zeros((hh, ww), dtype=bool)
    q = deque()

    def seed(rr: int, cc: int) -> None:
        if 0 <= rr < hh and 0 <= cc < ww and passable[rr, cc] and not seen[rr, cc]:
            seen[rr, cc] = True
            q.append((rr, cc))

    for cc in range(ww):
        seed(0, cc)
    for rr in range(hh):
        seed(rr, 0)
        seed(rr, ww - 1)

    while q:
        rr, cc = q.popleft()
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < hh and 0 <= nc < ww and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    interior = passable & ~seen
    regions = _mask_components(interior)
    return [coords + np.array([r0, c0], dtype=int) for coords in regions]


def _paint(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop large open grey contours, then fill each gravity-sealed cavity with its internal marker color.

    Input structure:
    - The background is black.
    - Large 4-connected objects (area > 3) are grey open-bottom contour sprites suspended in the upper half.
    - Small 4-connected objects (area <= 3) are fixed marker pixels placed inside the cavities that will appear after the drop.

    Rule:
    - Split foreground components by area into moving contours and fixed markers.
    - Drop contours straight downward in descending input-bottom order until the next step would leave the grid or hit a previously landed contour or marker.
    - For each landed contour, find the passable cells inside its bounding box that are unreachable from the top/side boundary without crossing the contour; this is the sealed cavity formed by gravity plus the support below.
    - Recolor every cavity cell, including its marker pixel, with the unique marker color inside it, while keeping the grey contours unchanged.

    Key insight:
    - The input outlines are still open, but after falling they use the support beneath them to turn their local bottom opening into a closed basin whose marker supplies the fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    components = _find_components(g)
    contours = []
    markers = []
    for idx, comp in enumerate(components):
        item = dict(comp)
        item["id"] = idx
        if int(comp["area"]) > 3:
            contours.append(item)
        else:
            for r, c in comp["coords"]:
                markers.append((int(r), int(c), int(comp["color"])))

    assert contours, "expected at least one contour"
    assert markers, "expected at least one marker"

    landed = _drop_contours(contours, markers, g.shape)

    dropped = np.full_like(g, BG)
    for r, c, color in markers:
        dropped[int(r), int(c)] = int(color)
    for comp in landed:
        _paint(dropped, comp["coords"], comp["color"])

    out = dropped.copy()
    filled = 0
    for comp in landed:
        regions = _local_cavity_regions(dropped, comp)
        for region in regions:
            colors = sorted({int(dropped[r, c]) for r, c in region if int(dropped[r, c]) != BG})
            assert len(colors) == 1, "each cavity must contain exactly one marker color"
            _paint(out, region, colors[0])
            filled += 1
        _paint(out, comp["coords"], comp["color"])

    assert filled > 0, "expected at least one fillable cavity"
    return out


def _make_contour_sprite(width: int | None = None) -> np.ndarray:
    w = int(width if width is not None else random.randint(3, 6))
    h = random.randint(3, 5)
    sprite = np.zeros((h, w), dtype=int)
    sprite[0, :] = CONTOUR_COLOR
    sprite[:, 0] = CONTOUR_COLOR
    sprite[:, -1] = CONTOUR_COLOR

    variant = random.choice(["plain", "left_hook", "right_hook", "tooth"])
    if variant == "left_hook" and w >= 4:
        sprite[h - 1, 1] = CONTOUR_COLOR
        if h >= 4 and random.random() < 0.5:
            sprite[h - 2, 1] = CONTOUR_COLOR
    elif variant == "right_hook" and w >= 4:
        sprite[h - 1, w - 2] = CONTOUR_COLOR
        if h >= 4 and random.random() < 0.5:
            sprite[h - 2, w - 2] = CONTOUR_COLOR
    elif variant == "tooth" and w >= 5 and h >= 4:
        sprite[1, w // 2] = CONTOUR_COLOR
    return sprite


def _sprite_coords(sprite: np.ndarray, top: int, left: int, idx: int) -> dict:
    rr, cc = np.nonzero(sprite == CONTOUR_COLOR)
    coords = np.column_stack([rr + int(top), cc + int(left)]).astype(int)
    return {
        "id": idx,
        "color": CONTOUR_COLOR,
        "coords": coords,
        "bbox": _bbox(coords),
        "area": int(coords.shape[0]),
    }


def _path_union(original: list[dict], landed: list[dict], shape: tuple[int, int]) -> np.ndarray:
    path = np.zeros(shape, dtype=bool)
    landed_by_id = {int(comp["id"]): comp for comp in landed}
    for comp in original:
        final = landed_by_id[int(comp["id"])]
        dr = int(final["drop"])
        for step in range(dr + 1):
            shifted = comp["coords"] + np.array([step, 0], dtype=int)
            path[shifted[:, 0], shifted[:, 1]] = True
    return path


def _place_input_contours(h: int, w: int) -> list[dict] | None:
    count = random.randint(1, 3)
    make_stack = count >= 2 and random.random() < 0.45

    shapes = []
    if make_stack:
        shared_width = random.randint(4, 6)
        shapes.append(_make_contour_sprite(shared_width))
        shapes.append(_make_contour_sprite(shared_width))
        for _ in range(count - 2):
            shapes.append(_make_contour_sprite())
        lane_groups = [[0, 1]] + [[idx] for idx in range(2, count)]
    else:
        for _ in range(count):
            shapes.append(_make_contour_sprite())
        lane_groups = [[idx] for idx in range(count)]

    lane_pos: list[tuple[int, int]] = []
    for group in lane_groups:
        lane_width = max(int(shapes[idx].shape[1]) for idx in group)
        for _ in range(50):
            left = random.randint(1, w - lane_width - 1)
            right = left + lane_width - 1
            if all(right + 2 < other_left or other_right + 2 < left for other_left, other_right in lane_pos):
                lane_pos.append((left, right))
                break
        else:
            return None

    contours: list[dict] = []
    for group, (left, right) in zip(lane_groups, lane_pos):
        if len(group) == 1:
            idx = group[0]
            sprite = shapes[idx]
            max_top = min(h // 2, h - int(sprite.shape[0]) - 4)
            if max_top < 1:
                return None
            top = random.randint(1, max_top)
            offset = left + (right - left + 1 - int(sprite.shape[1])) // 2
            contours.append(_sprite_coords(sprite, top, offset, idx))
            continue

        top_idx, lower_idx = group
        top_sprite = shapes[top_idx]
        lower_sprite = shapes[lower_idx]
        top_row = random.randint(1, max(1, h // 5))
        lower_min = top_row + int(top_sprite.shape[0]) + random.randint(1, 3)
        lower_max = min(h // 2, h - int(lower_sprite.shape[0]) - 4)
        if lower_min > lower_max:
            return None
        lower_row = random.randint(lower_min, lower_max)
        top_left = left + (right - left + 1 - int(top_sprite.shape[1])) // 2
        lower_left = left + (right - left + 1 - int(lower_sprite.shape[1])) // 2
        contours.append(_sprite_coords(top_sprite, top_row, top_left, top_idx))
        contours.append(_sprite_coords(lower_sprite, lower_row, lower_left, lower_idx))

    return contours


def _render_contours(shape: tuple[int, int], contours: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for comp in contours:
        _paint(grid, comp["coords"], comp["color"])
    return grid


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(13, 18)
    w = random.randint(13, 18)
    contours = _place_input_contours(h, w)
    if contours is None:
        return None

    input_grid = _render_contours((h, w), contours)
    if len(_find_components(input_grid)) != len(contours):
        return None

    landed = _drop_contours(contours, [], (h, w))
    landed_grid = _render_contours((h, w), landed)
    path = _path_union(contours, landed, (h, w))

    marker_positions = []
    used_colors = random.sample(list(MARKER_COLORS), k=len(landed))
    for comp, color in zip(sorted(landed, key=lambda x: x["id"]), used_colors):
        regions = _local_cavity_regions(landed_grid, comp)
        if len(regions) != 1:
            return None
        region = regions[0]
        candidates = [
            (int(r), int(c))
            for r, c in region
            if int(landed_grid[r, c]) == BG and not path[r, c]
        ]
        lower_half = [(r, c) for r, c in candidates if r >= h // 2]
        pick_from = lower_half if lower_half else candidates
        if not pick_from:
            return None
        marker_positions.append((*random.choice(pick_from), int(color)))

    for r, c, color in marker_positions:
        input_grid[int(r), int(c)] = int(color)

    components = _find_components(input_grid)
    marker_count = sum(1 for comp in components if int(comp["area"]) <= 3)
    contour_count = sum(1 for comp in components if int(comp["area"]) > 3)
    if marker_count != len(marker_positions) or contour_count != len(contours):
        return None

    output = main(input_grid)
    if np.array_equal(output, input_grid):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
