from __future__ import annotations

import random

import numpy as np


_DIRECTIONS = ("up", "right", "down", "left")
_CENTER_OFFSETS = {
    "up": [(-1, 0), (-1, -1), (-1, 1), (0, 0), (1, 0)],
    "right": [(0, 1), (-1, 1), (1, 1), (0, 0), (0, -1)],
    "down": [(1, 0), (1, -1), (1, 1), (0, 0), (-1, 0)],
    "left": [(0, -1), (-1, -1), (1, -1), (0, 0), (0, 1)],
}
_HEAD_OFFSETS = {
    "up": [(0, 0), (0, -1), (0, 1), (1, 0), (2, 0)],
    "right": [(0, 0), (-1, 0), (1, 0), (0, -1), (0, -2)],
    "down": [(0, 0), (0, -1), (0, 1), (-1, 0), (-2, 0)],
    "left": [(0, 0), (-1, 0), (1, 0), (0, 1), (0, 2)],
}
_ARROW_SHAPES = {
    frozenset(
        (int(r - pts[:, 0].min()), int(c - pts[:, 1].min()))
        for r, c in pts
    )
    for pts in (
        np.array(offsets, dtype=int)
        for offsets in _CENTER_OFFSETS.values()
    )
}


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask, dtype=int)
    return np.pad(arr, ((1, 0), (1, 0)), constant_values=0).cumsum(axis=0).cumsum(axis=1)


def _rect_sum(prefix: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    if r0 > r1 or c0 > c1:
        return 0
    return int(
        prefix[r1 + 1, c1 + 1]
        - prefix[r0, c1 + 1]
        - prefix[r1 + 1, c0]
        + prefix[r0, c0]
    )


def _is_arrow_component(coords: np.ndarray) -> bool:
    if int(coords.shape[0]) != 5:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
        return False
    normalized = frozenset((int(r - r0), int(c - c0)) for r, c in coords)
    return normalized in _ARROW_SHAPES


def _most_common_color(g: np.ndarray) -> int:
    values, counts = np.unique(g, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _infer_colors(g: np.ndarray) -> tuple[int, int, int, list[np.ndarray]]:
    bg = _most_common_color(g)
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    arrow_candidates: list[tuple[int, list[np.ndarray]]] = []
    for color in colors:
        comps = _components4(g == int(color))
        if comps and all(_is_arrow_component(comp) for comp in comps):
            arrow_candidates.append((int(color), comps))
    assert len(arrow_candidates) == 1, "expected one arrow color"
    arrow_color, arrow_comps = arrow_candidates[0]
    wall_colors = [color for color in colors if color != arrow_color]
    assert len(wall_colors) == 1, "expected one wall color"
    return bg, int(wall_colors[0]), int(arrow_color), arrow_comps


def _single_gap_on_bbox(g: np.ndarray, wall_color: int, bbox: tuple[int, int, int, int]) -> tuple[int, int] | None:
    r0, c0, r1, c1 = bbox
    gap: tuple[int, int] | None = None

    for c in range(c0, c1 + 1):
        if int(g[r0, c]) != wall_color:
            if gap is not None:
                return None
            gap = (r0, c)
        if int(g[r1, c]) != wall_color:
            if gap is not None:
                return None
            gap = (r1, c)

    for r in range(r0 + 1, r1):
        if int(g[r, c0]) != wall_color:
            if gap is not None:
                return None
            gap = (r, c0)
        if int(g[r, c1]) != wall_color:
            if gap is not None:
                return None
            gap = (r, c1)

    if gap is None:
        return None
    gr, gc = gap
    if (gr in (r0, r1)) and (gc in (c0, c1)):
        return None
    return gap


def _frame_rooms(
    g: np.ndarray,
    wall_color: int,
    arrow_comps: list[np.ndarray],
) -> list[dict] | None:
    rooms: list[dict] = []
    for wall_comp in _components4(g == int(wall_color)):
        r0, c0, r1, c1 = _bbox(wall_comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 6 or w < 6:
            continue

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        wall_mask = patch == int(wall_color)
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        if bool(np.any(wall_mask[1:-1, 1:-1])):
            continue
        if int(wall_mask.sum()) != int(border.sum()) - 1:
            continue

        gap = _single_gap_on_bbox(g, wall_color, (r0, c0, r1, c1))
        if gap is None:
            continue

        inside = []
        for idx, arrow_comp in enumerate(arrow_comps):
            ar0, ac0, ar1, ac1 = _bbox(arrow_comp)
            if r0 < ar0 and ar1 < r1 and c0 < ac0 and ac1 < c1:
                inside.append(idx)
        if len(inside) != 1:
            continue

        rooms.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gap": gap,
                "arrow_idx": int(inside[0]),
            }
        )

    if len(rooms) != len(arrow_comps):
        return None
    if len({room["arrow_idx"] for room in rooms}) != len(arrow_comps):
        return None
    return sorted(rooms, key=lambda room: int(room["arrow_idx"]))


def _scan_room_for_arrow(
    g: np.ndarray,
    wall_color: int,
    arrow_color: int,
    arrow_comp: np.ndarray,
    wall_prefix: np.ndarray,
    arrow_prefix: np.ndarray,
) -> dict:
    h, w = g.shape
    ar0, ac0, ar1, ac1 = _bbox(arrow_comp)
    best: dict | None = None

    for r0 in range(ar0 - 1, -1, -1):
        for r1 in range(ar1 + 1, h):
            room_h = r1 - r0 + 1
            if room_h < 6:
                continue
            for c0 in range(ac0 - 1, -1, -1):
                for c1 in range(ac1 + 1, w):
                    room_w = c1 - c0 + 1
                    if room_w < 6:
                        continue

                    perimeter = 2 * (room_h + room_w) - 4
                    if _rect_sum(wall_prefix, r0, c0, r1, c1) != perimeter - 1:
                        continue
                    if _rect_sum(wall_prefix, r0 + 1, c0 + 1, r1 - 1, c1 - 1) != 0:
                        continue
                    if _rect_sum(arrow_prefix, r0 + 1, c0 + 1, r1 - 1, c1 - 1) != 5:
                        continue

                    gap = _single_gap_on_bbox(g, wall_color, (r0, c0, r1, c1))
                    if gap is None or int(g[gap]) == arrow_color:
                        continue

                    area = room_h * room_w
                    candidate = {"bbox": (r0, c0, r1, c1), "gap": gap}
                    if best is None or area < int(best["area"]):
                        best = {"bbox": candidate["bbox"], "gap": candidate["gap"], "area": area}

    assert best is not None, "failed to locate room around arrow"
    return {"bbox": best["bbox"], "gap": best["gap"]}


def _locate_rooms(
    g: np.ndarray,
    wall_color: int,
    arrow_color: int,
    arrow_comps: list[np.ndarray],
) -> list[dict]:
    rooms = _frame_rooms(g, wall_color, arrow_comps)
    if rooms is not None:
        return rooms

    wall_prefix = _prefix_sum(g == int(wall_color))
    arrow_prefix = _prefix_sum(g == int(arrow_color))
    scanned = []
    for idx, arrow_comp in enumerate(arrow_comps):
        room = _scan_room_for_arrow(g, wall_color, arrow_color, arrow_comp, wall_prefix, arrow_prefix)
        room["arrow_idx"] = int(idx)
        scanned.append(room)
    return sorted(scanned, key=lambda room: int(room["arrow_idx"]))


def _gap_direction(bbox: tuple[int, int, int, int], gap: tuple[int, int]) -> str:
    r0, c0, r1, c1 = bbox
    gr, gc = gap
    if gr == r0:
        return "up"
    if gr == r1:
        return "down"
    if gc == c0:
        return "left"
    if gc == c1:
        return "right"
    raise AssertionError("gap must lie on one room wall")


def _cells_from_center(center_r: int, center_c: int, direction: str) -> list[tuple[int, int]]:
    return [(int(center_r + dr), int(center_c + dc)) for dr, dc in _CENTER_OFFSETS[direction]]


def _cells_from_head(head_r: int, head_c: int, direction: str) -> list[tuple[int, int]]:
    return [(int(head_r + dr), int(head_c + dc)) for dr, dc in _HEAD_OFFSETS[direction]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect each rectangular room, rotate its arrow toward the unique wall gap,
    then translate the arrow so its head lands exactly on that gap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, wall_color, arrow_color, arrow_comps = _infer_colors(g)
    rooms = _locate_rooms(g, wall_color, arrow_color, arrow_comps)
    assert len(rooms) == len(arrow_comps), "each arrow must belong to one room"

    out = g.copy()
    for room in rooms:
        coords = arrow_comps[int(room["arrow_idx"])]
        out[coords[:, 0], coords[:, 1]] = int(bg)

    for room in rooms:
        direction = _gap_direction(room["bbox"], room["gap"])
        for rr, cc in _cells_from_head(int(room["gap"][0]), int(room["gap"][1]), direction):
            assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]
            out[rr, cc] = int(arrow_color)

    return out


# --------------- generator ---------------

GEN_BG = 0
GEN_WALL = 1
GEN_ARROW = 2
GEN_HEIGHT_RANGE = (18, 31)
GEN_WIDTH_RANGE = (18, 31)
GEN_ROOM_SIDE_RANGE = (6, 11)
GEN_ROOM_PAD = 2
GEN_PLACE_ATTEMPTS = 200
GEN_ATTEMPTS = 300


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _draw_room(grid: np.ndarray, bbox: tuple[int, int, int, int], gap: tuple[int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = GEN_WALL
    grid[r1, c0 : c1 + 1] = GEN_WALL
    grid[r0 : r1 + 1, c0] = GEN_WALL
    grid[r0 : r1 + 1, c1] = GEN_WALL
    grid[gap] = GEN_BG


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(color)


def _sample_room(
    grid_shape: tuple[int, int],
    placed_boxes: list[tuple[int, int, int, int]],
) -> dict | None:
    height, width = grid_shape
    for _ in range(GEN_PLACE_ATTEMPTS):
        room_h = random.randint(*GEN_ROOM_SIDE_RANGE)
        room_w = random.randint(*GEN_ROOM_SIDE_RANGE)
        if room_h > height or room_w > width:
            continue

        r0 = random.randint(0, height - room_h)
        c0 = random.randint(0, width - room_w)
        bbox = (r0, c0, r0 + room_h - 1, c0 + room_w - 1)
        if any(_boxes_overlap(bbox, other, GEN_ROOM_PAD) for other in placed_boxes):
            continue

        exit_dir = random.choice(_DIRECTIONS)
        if exit_dir in ("up", "down"):
            gap_r = r0 if exit_dir == "up" else r0 + room_h - 1
            gap_c = random.randint(c0 + 2, c0 + room_w - 3)
        else:
            gap_r = random.randint(r0 + 2, r0 + room_h - 3)
            gap_c = c0 if exit_dir == "left" else c0 + room_w - 1
        gap = (gap_r, gap_c)

        input_dir = random.choice(_DIRECTIONS)
        center_r = random.randint(r0 + 2, r0 + room_h - 3)
        center_c = random.randint(c0 + 2, c0 + room_w - 3)
        input_cells = _cells_from_center(center_r, center_c, input_dir)
        output_cells = _cells_from_head(gap_r, gap_c, exit_dir)
        if set(input_cells) == set(output_cells):
            continue

        return {
            "bbox": bbox,
            "gap": gap,
            "exit_dir": exit_dir,
            "input_dir": input_dir,
            "input_cells": input_cells,
            "output_cells": output_cells,
        }
    return None


def _expected_from_specs(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        _paint_cells(out, spec["input_cells"], GEN_BG)
    for spec in specs:
        _paint_cells(out, spec["output_cells"], GEN_ARROW)
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GEN_HEIGHT_RANGE)
    width = random.randint(*GEN_WIDTH_RANGE)
    grid = np.full((height, width), GEN_BG, dtype=int)

    max_rooms = min(5, max(1, (height * width) // 70))
    room_count = random.randint(1, max_rooms)

    specs: list[dict] = []
    boxes: list[tuple[int, int, int, int]] = []
    for _ in range(room_count):
        spec = _sample_room((height, width), boxes)
        if spec is None:
            return None
        specs.append(spec)
        boxes.append(spec["bbox"])

    for spec in specs:
        _draw_room(grid, spec["bbox"], spec["gap"])
    for spec in specs:
        _paint_cells(grid, spec["input_cells"], GEN_ARROW)

    expected = _expected_from_specs(grid, specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-arrow puzzle")
