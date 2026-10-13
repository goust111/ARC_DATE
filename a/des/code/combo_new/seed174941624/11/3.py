from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
AXIS_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(coords)
    return out


def _line_component_info(coords: list[tuple[int, int]], color: int) -> dict:
    rows = sorted({r for r, _c in coords})
    cols = sorted({c for _r, c in coords})
    if len(rows) == 1:
        orientation = "h"
        axis = rows[0]
        span = cols
    else:
        assert len(cols) == 1, "mirror axis must be horizontal or vertical"
        orientation = "v"
        axis = cols[0]
        span = rows
    assert span == list(range(span[0], span[-1] + 1)), "mirror axis must be contiguous"
    return {
        "coords": coords,
        "coord_set": set(coords),
        "color": int(color),
        "orientation": orientation,
        "axis": int(axis),
    }


def _reflect_coords(
    coords: list[tuple[int, int]], orientation: str, axis: int
) -> list[tuple[int, int]]:
    if orientation == "v":
        return [(r, 2 * axis - c) for r, c in coords]
    return [(2 * axis - r, c) for r, c in coords]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each grey object across its adjacent colored axis segment and recolor the mirrored copy with the axis color.

    Input structure:
    - The background is black.
    - The grid contains one-cell-thick horizontal or vertical colored line segments that act as mirror axes.
    - The grid also contains one or more 4-connected grey objects.
    - Each grey object is adjacent by one side to exactly one mirror-axis segment, while each axis may touch zero or more grey objects.

    Rule:
    - Keep every colored axis segment and every original grey object unchanged.
    - For each grey object, find the unique axis segment it touches by 4-neighborhood adjacency.
    - Reflect that grey object's shape across the infinite row or column containing the touched axis segment.
    - Draw the reflected copy in the touched axis segment's color, overwriting any non-black pixels at the mirrored locations.

    Key insight:
    - The grey object's adjacency identifies which colored segment provides the mirror line, and the new copy inherits the axis color rather than the grey source color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    grey_components = _components(g == GREY)
    assert grey_components, "expected at least one grey object"

    line_components: list[dict] = []
    coord_to_line: dict[tuple[int, int], int] = {}
    for color in sorted(int(c) for c in np.unique(g) if c not in (BG, GREY)):
        for comp in _components(g == color):
            info = _line_component_info(comp, color)
            idx = len(line_components)
            line_components.append(info)
            for coord in comp:
                coord_to_line[coord] = idx
    assert line_components, "expected at least one mirror axis"

    out = g.copy()
    for comp in grey_components:
        line_ids = set()
        for r, c in comp:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                idx = coord_to_line.get((nr, nc))
                if idx is not None:
                    line_ids.add(idx)
        assert len(line_ids) == 1, "each grey object must touch exactly one mirror axis"
        line = line_components[next(iter(line_ids))]
        mirrored = _reflect_coords(comp, line["orientation"], line["axis"])
        for rr, cc in mirrored:
            assert 0 <= rr < h and 0 <= cc < w, "mirrored copy must stay inside the grid"
            out[rr, cc] = int(line["color"])
    return out


def _neighbors_in_box(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _random_polyomino(
    h: int, w: int, area: int, touch_side: str, attempts: int = 80
) -> np.ndarray | None:
    assert area <= h * w
    if touch_side == "left":
        starters = [(r, 0) for r in range(h)]
    elif touch_side == "right":
        starters = [(r, w - 1) for r in range(h)]
    elif touch_side == "top":
        starters = [(0, c) for c in range(w)]
    else:
        starters = [(h - 1, c) for c in range(w)]

    for _ in range(attempts):
        cells = {random.choice(starters)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                frontier.update(_neighbors_in_box(r, c, h, w))
            frontier -= cells
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return mask
    return None


def _mask_coords(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    coords: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        coords.add((top + int(r), left + int(c)))
    return coords


def _touch_mapping(orientation: str, side: str) -> str:
    if orientation == "v":
        return "right" if side == "left" else "left"
    return "bottom" if side == "above" else "top"


def _side_bbox(
    orientation: str, side: str, depth: int, line_len: int
) -> tuple[int, int, int, int] | None:
    if orientation == "v":
        obj_h = random.randint(2, min(5, line_len))
        obj_w = random.randint(1, depth)
        if obj_h * obj_w < 3:
            return None
        top = random.randint(0, line_len - obj_h)
        left = depth - obj_w if side == "left" else depth + 1
        return top, left, obj_h, obj_w
    obj_h = random.randint(1, depth)
    obj_w = random.randint(2, min(5, line_len))
    if obj_h * obj_w < 3:
        return None
    top = depth - obj_h if side == "above" else depth + 1
    left = random.randint(0, line_len - obj_w)
    return top, left, obj_h, obj_w


def _reflect_local(
    coords: set[tuple[int, int]], orientation: str, axis: int
) -> set[tuple[int, int]]:
    if orientation == "v":
        return {(r, 2 * axis - c) for r, c in coords}
    return {(2 * axis - r, c) for r, c in coords}


def _adjacent_to_set(coords: set[tuple[int, int]], others: set[tuple[int, int]]) -> bool:
    for r, c in coords:
        for dr, dc in DIR4:
            if (r + dr, c + dc) in others:
                return True
    return False


def _sample_object_count(allow_empty: bool) -> int:
    if allow_empty:
        return random.choices([0, 1, 2, 3], weights=[1, 4, 3, 1], k=1)[0]
    return random.choices([1, 2, 3], weights=[4, 3, 1], k=1)[0]


def _build_patch(
    orientation: str, require_object: bool, overlap_pair: bool
) -> dict | None:
    for _ in range(120):
        axis_color = random.choice(AXIS_COLORS)
        line_len = random.randint(6, 12)
        depth = random.randint(2, 5)
        patch_h = line_len if orientation == "v" else 2 * depth + 1
        patch_w = 2 * depth + 1 if orientation == "v" else line_len
        axis = depth
        if orientation == "v":
            axis_cells = {(r, axis) for r in range(patch_h)}
            sides = ("left", "right")
        else:
            axis_cells = {(axis, c) for c in range(patch_w)}
            sides = ("above", "below")

        grey_cells: set[tuple[int, int]] = set()
        used_output = set(axis_cells)
        object_count = 0

        if overlap_pair:
            pair_side = sides[0]
            bbox = _side_bbox(orientation, pair_side, depth, line_len)
            if bbox is None:
                continue
            top, left, obj_h, obj_w = bbox
            area = random.randint(3, min(obj_h * obj_w, 7))
            mask = _random_polyomino(
                obj_h, obj_w, area, _touch_mapping(orientation, pair_side)
            )
            if mask is None:
                continue
            first = _mask_coords(mask, top, left)
            second = _reflect_local(first, orientation, axis)
            if first & second:
                continue
            grey_cells |= first | second
            used_output |= first | second
            object_count = 2

        target_objects = object_count + _sample_object_count(
            allow_empty=(not require_object and not overlap_pair)
        )
        if require_object and target_objects == 0:
            target_objects = 1

        placed = object_count
        while placed < target_objects:
            success = False
            for _ in range(80):
                side = random.choice(sides)
                bbox = _side_bbox(orientation, side, depth, line_len)
                if bbox is None:
                    continue
                top, left, obj_h, obj_w = bbox
                area = random.randint(3, min(obj_h * obj_w, 7))
                mask = _random_polyomino(
                    obj_h, obj_w, area, _touch_mapping(orientation, side)
                )
                if mask is None:
                    continue
                coords = _mask_coords(mask, top, left)
                mirror = _reflect_local(coords, orientation, axis)
                footprint = coords | mirror
                if footprint & used_output:
                    continue
                if _adjacent_to_set(coords, grey_cells):
                    continue
                grey_cells |= coords
                used_output |= footprint
                placed += 1
                success = True
                break
            if not success:
                break

        if placed < target_objects:
            continue
        if require_object and not grey_cells:
            continue

        patch = np.full((patch_h, patch_w), BG, dtype=int)
        for r, c in axis_cells:
            patch[r, c] = axis_color
        for r, c in grey_cells:
            patch[r, c] = GREY
        return {
            "grid": patch,
            "height": patch_h,
            "width": patch_w,
            "object_count": placed,
        }
    return None


def _make_patches() -> list[dict] | None:
    overlap_mode = random.random() < 0.45
    total = random.randint(2, 4) if overlap_mode else random.randint(1, 4)
    patches: list[dict] = []

    if overlap_mode:
        overlap_patch = _build_patch(
            orientation=random.choice(("v", "h")),
            require_object=True,
            overlap_pair=True,
        )
        if overlap_patch is None:
            return None
        patches.append(overlap_patch)
        normal = _build_patch(
            orientation=random.choice(("v", "h")),
            require_object=True,
            overlap_pair=False,
        )
        if normal is None:
            return None
        patches.append(normal)

    while len(patches) < total:
        patch = _build_patch(
            orientation=random.choice(("v", "h")),
            require_object=(len(patches) == 0),
            overlap_pair=False,
        )
        if patch is None:
            return None
        patches.append(patch)

    return patches


def _place_patch(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    region = dst[top : top + h, left : left + w]
    mask = patch != BG
    region[mask] = patch[mask]


def _pack_rows(patches: list[dict]) -> np.ndarray | None:
    patches = patches[:]
    random.shuffle(patches)
    if len(patches) == 1:
        rows = [patches]
    elif random.random() < 0.5:
        split = random.randint(1, len(patches) - 1)
        rows = [patches[:split], patches[split:]]
    else:
        rows = [patches]

    row_gap = random.randint(2, 4)
    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    row_widths = []
    row_heights = []
    col_gaps = []
    for row in rows:
        gap = random.randint(1, 3)
        col_gaps.append(gap)
        row_width = sum(p["width"] for p in row) + gap * (len(row) - 1)
        row_height = max(p["height"] for p in row)
        row_widths.append(row_width)
        row_heights.append(row_height)

    total_h = top_margin + bottom_margin + sum(row_heights) + row_gap * (len(rows) - 1)
    total_w = left_margin + right_margin + max(row_widths)
    if total_h > 30 or total_w > 30:
        return None

    grid = np.full((total_h, total_w), BG, dtype=int)
    y = top_margin
    for row, row_height, row_width, gap in zip(rows, row_heights, row_widths, col_gaps):
        start_x = left_margin + random.randint(0, total_w - left_margin - right_margin - row_width)
        x = start_x
        for patch in row:
            y_off = y + random.randint(0, row_height - patch["height"])
            _place_patch(grid, patch["grid"], y_off, x)
            x += patch["width"] + gap
        y += row_height + row_gap
    return grid


def _gen_attempt() -> np.ndarray | None:
    patches = _make_patches()
    if patches is None:
        return None
    grid = _pack_rows(patches)
    if grid is None:
        return None
    out = main(grid)
    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
