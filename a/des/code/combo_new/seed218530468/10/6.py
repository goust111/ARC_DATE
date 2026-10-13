from __future__ import annotations

from collections import Counter
import random

import numpy as np


MAX_GRID_SIZE = 30
MAX_GENERATION_ATTEMPTS = 200
MIN_MARGIN = 2
CORE_SIZE_CHOICES = (1, 2)
MAX_FRAME_COUNT = 5
ALL_COLORS = tuple(range(10))


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    assert connectivity == 4
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for dr, dc in dirs:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(pts, dtype=int))
    return comps


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _paint_square_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _classify_structure_component(mask: np.ndarray, coords: np.ndarray) -> dict[str, int | tuple[int, int, int, int] | str] | None:
    r0, c0, r1, c1 = bbox_coords(coords)
    side_h = int(r1 - r0 + 1)
    side_w = int(c1 - c0 + 1)
    if side_h != side_w:
        return None
    side = int(side_h)
    area = int(coords.shape[0])
    sub = np.zeros((side, side), dtype=bool)
    sub[coords[:, 0] - r0, coords[:, 1] - c0] = True
    if area == side * side and bool(sub.all()):
        return {"kind": "core", "bbox": (r0, c0, r1, c1), "side": side}
    if side < 3:
        return None
    expected = np.zeros((side, side), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    if area == int(expected.sum()) and bool(np.array_equal(sub, expected)):
        return {"kind": "frame", "bbox": (r0, c0, r1, c1), "side": side}
    return None


def _detect_structure(
    grid: np.ndarray,
) -> tuple[int, dict[str, int], int, tuple[int, int, int, int], list[tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 7 and w >= 7
    bg = get_bg(g)
    corner_colors = {
        "tl": int(g[0, 0]),
        "tr": int(g[0, w - 1]),
        "bl": int(g[h - 1, 0]),
        "br": int(g[h - 1, w - 1]),
    }
    assert len(set(corner_colors.values())) == 4
    assert all(int(col) != bg for col in corner_colors.values())

    colors = sorted(int(col) for col in np.unique(g) if int(col) != bg)
    candidates: list[tuple[int, tuple[int, int, int, int], list[tuple[int, int, int, int]]]] = []
    for color in colors:
        mask = g == int(color)
        comps = components_mask(mask, connectivity=4)
        if not comps:
            continue
        classified = [_classify_structure_component(mask, comp) for comp in comps]
        if any(item is None for item in classified):
            continue
        cores = [item for item in classified if item["kind"] == "core"]
        frames = [item for item in classified if item["kind"] == "frame"]
        if len(cores) != 1 or not frames or len(cores) + len(frames) != len(classified):
            continue
        core_bbox = cores[0]["bbox"]
        core_side = int(cores[0]["side"])
        center_row_sum = int(core_bbox[0] + core_bbox[2])
        center_col_sum = int(core_bbox[1] + core_bbox[3])
        frame_boxes = [item["bbox"] for item in frames]
        frame_sides = sorted(int(item["side"]) for item in frames)
        if any(int(box[0] + box[2]) != center_row_sum or int(box[1] + box[3]) != center_col_sum for box in frame_boxes):
            continue
        expected_sides = [int(core_side + 4 * idx) for idx in range(1, len(frame_sides) + 1)]
        if frame_sides != expected_sides:
            continue
        frame_boxes = sorted(frame_boxes, key=lambda box: int(box[2] - box[0]))
        candidates.append((int(color), core_bbox, frame_boxes))

    assert len(candidates) == 1, "expected exactly one concentric frame/core structure"
    structure_color, core_bbox, frame_boxes = candidates[0]
    return bg, corner_colors, structure_color, core_bbox, frame_boxes


def _recolor_frame(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    corner_colors: dict[str, int],
) -> None:
    r0, c0, r1, c1 = bbox
    side = int(r1 - r0 + 1)
    mid_r = int(r0 + (side - 1) // 2)
    mid_c = int(c0 + (side - 1) // 2)

    for c in range(c0, c1 + 1):
        out[r0, c] = int(corner_colors["tl"] if c <= mid_c else corner_colors["tr"])
        out[r1, c] = int(corner_colors["bl"] if c <= mid_c else corner_colors["br"])
    for r in range(r0 + 1, r1):
        out[r, c0] = int(corner_colors["tl"] if r <= mid_r else corner_colors["bl"])
        out[r, c1] = int(corner_colors["tr"] if r <= mid_r else corner_colors["br"])


def _draw_rays(
    out: np.ndarray,
    core_bbox: tuple[int, int, int, int],
    outer_bbox: tuple[int, int, int, int],
    corner_colors: dict[str, int],
) -> None:
    cr0, cc0, cr1, cc1 = core_bbox
    or0, oc0, or1, oc1 = outer_bbox
    if cr0 > or0:
        out[or0:cr0, cc0 : cc1 + 1] = int(corner_colors["tl"])
    if cr1 < or1:
        out[cr1 + 1 : or1 + 1, cc0 : cc1 + 1] = int(corner_colors["br"])
    if cc0 > oc0:
        out[cr0 : cr1 + 1, oc0:cc0] = int(corner_colors["bl"])
    if cc1 < oc1:
        out[cr0 : cr1 + 1, cc1 + 1 : oc1 + 1] = int(corner_colors["tr"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor concentric square frames by corner markers and extend four cardinal rays from the inner core.

    Input structure:
    - The background is the dominant color.
    - One foreground color forms several disconnected concentric 1-pixel square frames plus one solid square core.
    - The four grid corners contain four different single-pixel marker colors.

    Rule:
    - Identify the shared frame/core color, the nested frame bboxes, and the solid inner core.
    - Recolor each frame perimeter into four corner-controlled segments: top and bottom edges split at the horizontal midpoint, left and right edges split at the vertical midpoint, and midpoint ties go to the upper or left segment.
    - Extend four rays from the four sides of the core out to the outermost frame, using the top-left marker upward, top-right marker rightward, bottom-right marker downward, and bottom-left marker leftward.
    - Rays overwrite both frame pixels and background gaps, while the four corner markers stay unchanged.

    Key insight:
    - The corner markers encode a direction-to-color map that must be applied twice: once to partition every square frame into quadrant-colored edge segments, and again to color the cross-shaped rays emitted by the core.
    """
    g = np.asarray(grid, dtype=int)
    _bg, corner_colors, _structure_color, core_bbox, frame_boxes = _detect_structure(g)
    out = g.copy()
    for box in frame_boxes:
        _recolor_frame(out, box, corner_colors)
    _draw_rays(out, core_bbox, frame_boxes[-1], corner_colors)

    h, w = out.shape
    out[0, 0] = int(corner_colors["tl"])
    out[0, w - 1] = int(corner_colors["tr"])
    out[h - 1, 0] = int(corner_colors["bl"])
    out[h - 1, w - 1] = int(corner_colors["br"])
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = int(random.choice(ALL_COLORS))
    palette = [int(col) for col in ALL_COLORS if int(col) != bg]
    structure_color, tl_color, tr_color, bl_color, br_color = random.sample(palette, 5)

    core_size = int(random.choice(CORE_SIZE_CHOICES))
    max_frames = min(MAX_FRAME_COUNT, (MAX_GRID_SIZE - 2 * MIN_MARGIN - core_size) // 4)
    if max_frames < 1:
        return None
    frame_count = int(random.randint(1, max_frames))
    outer_side = int(core_size + 4 * frame_count)

    max_margin_r = (MAX_GRID_SIZE - outer_side) // 2
    max_margin_c = (MAX_GRID_SIZE - outer_side) // 2
    if max_margin_r < MIN_MARGIN or max_margin_c < MIN_MARGIN:
        return None

    margin_r = int(random.randint(MIN_MARGIN, max_margin_r))
    margin_c = int(random.randint(MIN_MARGIN, max_margin_c))
    h = int(outer_side + 2 * margin_r)
    w = int(outer_side + 2 * margin_c)

    grid = np.full((h, w), int(bg), dtype=int)
    grid[0, 0] = int(tl_color)
    grid[0, w - 1] = int(tr_color)
    grid[h - 1, 0] = int(bl_color)
    grid[h - 1, w - 1] = int(br_color)

    outer_r0 = int(margin_r)
    outer_c0 = int(margin_c)
    core_r0 = int(outer_r0 + 2 * frame_count)
    core_c0 = int(outer_c0 + 2 * frame_count)
    grid[core_r0 : core_r0 + core_size, core_c0 : core_c0 + core_size] = int(structure_color)

    for depth in range(1, frame_count + 1):
        side = int(core_size + 4 * depth)
        r0 = int(core_r0 - 2 * depth)
        c0 = int(core_c0 - 2 * depth)
        bbox = (r0, c0, int(r0 + side - 1), int(c0 + side - 1))
        _paint_square_frame(grid, bbox, int(structure_color))

    return grid


def _validate_generated_input(grid: np.ndarray) -> bool:
    try:
        bg, corner_colors, structure_color, core_bbox, frame_boxes = _detect_structure(grid)
    except AssertionError:
        return False
    if int(bg) == int(structure_color):
        return False
    if int(max(grid.shape)) > MAX_GRID_SIZE:
        return False
    if len(frame_boxes) < 1:
        return False
    if len(set(corner_colors.values())) != 4:
        return False
    core_side = int(core_bbox[2] - core_bbox[0] + 1)
    if core_side not in CORE_SIZE_CHOICES:
        return False
    try:
        out = main(grid)
    except AssertionError:
        return False
    if bool(np.array_equal(out, grid)):
        return False
    if out.shape != grid.shape:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None and _validate_generated_input(grid):
            return grid
    raise RuntimeError("failed to generate a valid concentric-frame input")
