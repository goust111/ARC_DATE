from __future__ import annotations

import random

import numpy as np


BG = 0
FOUR_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))
ALL_COLORS = tuple(range(1, 10))

# Generator configuration.
OBJECT_COUNT_RANGE = (2, 4)
GRID_HEIGHT_RANGE = (16, 24)
GRID_WIDTH_RANGE = (16, 24)
LEGEND_LENGTH_MIN = 3
LEGEND_LENGTH_MAX = 6
PATCH_RETRIES = 80
GENERATION_RETRIES = 300


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in FOUR_NEIGHBORS:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _color_components(grid: np.ndarray, bg: int = BG) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in FOUR_NEIGHBORS:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(pts, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return comps


def _count_holes(mask: np.ndarray) -> int:
    holes = 0
    for coords in _mask_components(~mask):
        touches_boundary = any(
            int(r) == 0
            or int(c) == 0
            or int(r) == mask.shape[0] - 1
            or int(c) == mask.shape[1] - 1
            for r, c in coords
        )
        if not touches_boundary:
            holes += 1
    return holes


def _legend_colors(grid: np.ndarray) -> list[int]:
    colors = [int(v) for v in grid[0] if int(v) != BG]
    assert colors, "expected at least one legend color"
    assert len(colors) == len(set(colors)), "legend colors must be unique"
    return colors


def _object_records(grid: np.ndarray) -> list[dict[str, int]]:
    scene = np.asarray(grid[2:, :], dtype=int)
    records: list[dict[str, int]] = []
    for comp in _color_components(scene, bg=BG):
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]  # scene-relative
        sub = scene[r0 : r1 + 1, c0 : c1 + 1]
        hole_count = _count_holes(sub == color)
        assert hole_count >= 1, "every object must contain at least one hole"
        records.append(
            {
                "holes": int(hole_count),
                "row": int(r0),
                "col": int(c0),
            }
        )
    assert records, "expected at least one scene object"
    return records


def _render_concentric(grid: np.ndarray, ordered_holes: list[int]) -> np.ndarray:
    out = np.zeros_like(grid)
    out[0, :] = grid[0, :]

    legend = _legend_colors(grid)
    colors = [int(legend[k - 1]) for k in ordered_holes]
    side = 4 * len(colors) - 3

    height, width = grid.shape
    scene_height = height - 2
    assert side <= scene_height and side <= width, "scene is too small for the canonical output"

    top = 2 + (scene_height - side) // 2
    left = (width - side) // 2

    for layer, color in enumerate(colors):
        inset = 2 * layer
        r0 = top + inset
        c0 = left + inset
        span = side - 4 * layer
        r1 = r0 + span - 1
        c1 = c0 + span - 1
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each object by its hole count, sort objects by hole count, and redraw them
    as centered concentric square frames below the unchanged top legend.
    """
    g = np.asarray(grid, dtype=int)
    records = _object_records(g)
    ordered = sorted(records, key=lambda rec: (-int(rec["holes"]), int(rec["col"]), int(rec["row"])))
    return _render_concentric(g, [int(rec["holes"]) for rec in ordered])


def _crop_nonzero_bbox(patch: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.asarray(patch[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _checker_capacity(height: int, width: int) -> int:
    return ((height - 1) // 2) * ((width - 1) // 2)


def _side_extensions(mode: str) -> tuple[int, int, int, int]:
    if mode == "rect":
        return 0, 0, 0, 0
    if mode == "one_side":
        values = {"top": 0, "bottom": 0, "left": 0, "right": 0}
        values[random.choice(tuple(values))] = random.randint(1, 2)
        return values["top"], values["bottom"], values["left"], values["right"]
    if mode == "opposite":
        if random.random() < 0.5:
            return 0, 0, random.randint(1, 2), random.randint(1, 2)
        return random.randint(1, 2), random.randint(1, 2), 0, 0
    top = random.randint(0, 2)
    bottom = random.randint(0, 2)
    left = random.randint(0, 2)
    right = random.randint(0, 2)
    if top == bottom == left == right == 0:
        left = 1
    return top, bottom, left, right


def _paint_vertical_arm(patch: np.ndarray, row0: int, col0: int, core_h: int, ext: int, place_left: bool, color: int) -> None:
    if ext == 0:
        return
    arm_h = random.randint(max(2, core_h // 3), core_h)
    start = random.randint(0, core_h - arm_h)
    rr0 = row0 + start
    rr1 = rr0 + arm_h
    if place_left:
        patch[rr0:rr1, :col0] = color
    else:
        patch[rr0:rr1, patch.shape[1] - ext :] = color


def _paint_horizontal_arm(patch: np.ndarray, row0: int, col0: int, core_w: int, ext: int, place_top: bool, color: int) -> None:
    if ext == 0:
        return
    arm_w = random.randint(max(2, core_w // 3), core_w)
    start = random.randint(0, core_w - arm_w)
    cc0 = col0 + start
    cc1 = cc0 + arm_w
    if place_top:
        patch[:row0, cc0:cc1] = color
    else:
        patch[patch.shape[0] - ext :, cc0:cc1] = color


def _build_object_patch(color: int, hole_count: int) -> np.ndarray | None:
    for _ in range(PATCH_RETRIES):
        core_h = random.randint(5, 8 + hole_count)
        core_w = random.randint(5, 8 + hole_count)
        if _checker_capacity(core_h, core_w) < hole_count:
            continue

        mode = random.choice(("rect", "one_side", "opposite", "multi_side"))
        ext_top, ext_bottom, ext_left, ext_right = _side_extensions(mode)
        patch = np.zeros((core_h + ext_top + ext_bottom, core_w + ext_left + ext_right), dtype=int)

        r0 = ext_top
        c0 = ext_left
        patch[r0 : r0 + core_h, c0 : c0 + core_w] = int(color)

        candidates = [(r0 + r, c0 + c) for r in range(1, core_h - 1, 2) for c in range(1, core_w - 1, 2)]
        holes = random.sample(candidates, hole_count)
        for hr, hc in holes:
            patch[hr, hc] = BG

        _paint_vertical_arm(patch, r0, c0, core_h, ext_left, True, int(color))
        _paint_vertical_arm(patch, r0, c0, core_h, ext_right, False, int(color))
        _paint_horizontal_arm(patch, r0, c0, core_w, ext_top, True, int(color))
        _paint_horizontal_arm(patch, r0, c0, core_w, ext_bottom, False, int(color))

        patch = _crop_nonzero_bbox(patch)
        if _count_holes(patch == int(color)) == int(hole_count):
            return patch
    return None


def _free_positions(blocked: np.ndarray, patch: np.ndarray) -> list[tuple[int, int]]:
    scene_h, scene_w = blocked.shape
    h, w = patch.shape
    positions: list[tuple[int, int]] = []
    for r in range(1, scene_h - h):
        for c in range(1, scene_w - w):
            if not blocked[r - 1 : r + h + 1, c - 1 : c + w + 1].any():
                positions.append((r, c))
    return positions


def _paste_patch(scene: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    scene[top : top + h, left : left + w][patch != BG] = patch[patch != BG]


def _build_expected(grid: np.ndarray, placements: list[dict[str, int]]) -> np.ndarray:
    ordered = sorted(placements, key=lambda rec: (-int(rec["holes"]), int(rec["col"]), int(rec["row"])))
    return _render_concentric(grid, [int(rec["holes"]) for rec in ordered])


def _gen_attempt() -> np.ndarray | None:
    object_count = random.randint(*OBJECT_COUNT_RANGE)
    max_legend = min(LEGEND_LENGTH_MAX, 9 - object_count)
    min_legend = max(LEGEND_LENGTH_MIN, object_count)
    if min_legend > max_legend:
        return None
    legend_length = random.randint(min_legend, max_legend)
    hole_counts = random.sample(list(range(1, legend_length + 1)), object_count)

    min_side = 4 * object_count - 3
    min_height = max(GRID_HEIGHT_RANGE[0], min_side + 4)
    min_width = max(GRID_WIDTH_RANGE[0], min_side + 2, legend_length + 2)
    if min_height > GRID_HEIGHT_RANGE[1] or min_width > GRID_WIDTH_RANGE[1]:
        return None

    height = random.randint(min_height, GRID_HEIGHT_RANGE[1])
    width = random.randint(min_width, GRID_WIDTH_RANGE[1])
    grid = np.zeros((height, width), dtype=int)

    legend_colors = random.sample(list(ALL_COLORS), legend_length)
    object_palette = [color for color in ALL_COLORS if color not in set(legend_colors)]
    object_colors = random.sample(object_palette, object_count)

    legend_start = random.randint(0, width - legend_length)
    grid[0, legend_start : legend_start + legend_length] = np.array(legend_colors, dtype=int)

    scene = grid[2:, :]
    blocked = np.zeros_like(scene, dtype=bool)
    placements: list[dict[str, int]] = []

    order = list(range(object_count))
    random.shuffle(order)
    for idx in order:
        patch = _build_object_patch(object_colors[idx], hole_counts[idx])
        if patch is None:
            return None
        positions = _free_positions(blocked, patch)
        if not positions:
            return None
        top, left = random.choice(positions)
        _paste_patch(scene, patch, top, left)
        h, w = patch.shape
        blocked[top - 1 : top + h + 1, left - 1 : left + w + 1] = True
        placements.append({"holes": int(hole_counts[idx]), "row": int(top), "col": int(left)})

    expected = _build_expected(grid, placements)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
