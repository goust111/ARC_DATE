import random
from collections import deque

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GRID_SIZE_RANGE = (22, 30)
OBJECT_COUNT_CHOICES = [2, 3]
HOLE_COUNT_CHOICES = [0, 1, 2, 3, 4]
GEN_ATTEMPTS = 400
PAIR_ATTEMPTS = 160
OBJECT_PLACEMENT_ATTEMPTS = 160

DIR_TO_PATTERN = {
    (-1, 0): frozenset({(0, 0), (1, 0), (1, 1)}),  # up: left edge is the full side
    (0, 1): frozenset({(0, 0), (0, 1), (1, 0)}),   # right: top edge is the full side
    (1, 0): frozenset({(0, 0), (0, 1), (1, 1)}),   # down: right edge is the full side
    (0, -1): frozenset({(0, 1), (1, 0), (1, 1)}),  # left: bottom edge is the full side
}
PATTERN_TO_DIR = {pattern: direction for direction, pattern in DIR_TO_PATTERN.items()}
CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in CARDINALS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    return _mask_components(grid == int(color))


def _is_rect_frame(cells: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(cells)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    boundary = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    return len(boundary) == len(cells) and set(cells) == boundary


def _marker_direction(cells: list[tuple[int, int]]) -> tuple[int, int] | None:
    if len(cells) != 3:
        return None
    r0, c0, r1, c1 = _bbox(cells)
    if r1 - r0 != 1 or c1 - c0 != 1:
        return None
    norm = frozenset((r - r0, c - c0) for r, c in cells)
    return PATTERN_TO_DIR.get(norm)


def _body_encloses_marker(
    body_cells: list[tuple[int, int]], marker_cells: list[tuple[int, int]]
) -> bool:
    r0, c0, r1, c1 = _bbox(body_cells)
    if not all(r0 < r < r1 and c0 < c < c1 for r, c in marker_cells):
        return False

    local_h = r1 - r0 + 1
    local_w = c1 - c0 + 1
    blocked = np.zeros((local_h, local_w), dtype=bool)
    for r, c in body_cells:
        blocked[r - r0, c - c0] = True

    seen = np.zeros_like(blocked)
    q = deque()
    for r, c in marker_cells:
        rr, cc = r - r0, c - c0
        seen[rr, cc] = True
        q.append((rr, cc))

    while q:
        rr, cc = q.popleft()
        if rr in (0, local_h - 1) or cc in (0, local_w - 1):
            return False
        for dr, dc in CARDINALS:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < local_h and 0 <= nc < local_w and not blocked[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return True


def _count_background_components(mask: np.ndarray) -> int:
    return len(_mask_components(mask))


def _find_entities(grid: np.ndarray, bg: int) -> list[dict]:
    entities: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        comp_infos = []
        for cells in _color_components(grid, color):
            comp_infos.append(
                {
                    "cells": cells,
                    "bbox": _bbox(cells),
                    "is_frame": _is_rect_frame(cells),
                    "direction": _marker_direction(cells),
                }
            )

        frames = [info for info in comp_infos if info["is_frame"]]
        markers = [info for info in comp_infos if info["direction"] is not None]
        if len(frames) != 1 or not markers:
            continue

        frame = frames[0]
        remainder = [info for info in comp_infos if info is not frame and info["direction"] is None]
        match = None
        for marker in markers:
            bodies = [info for info in remainder if _body_encloses_marker(info["cells"], marker["cells"])]
            if len(bodies) == 1:
                match = {"color": color, "frame": frame, "body": bodies[0], "marker": marker}
                break
        if match is not None:
            entities.append(match)
    return entities


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each colored object by the distance encoded as the number of holes inside its same-color fence.

    Input structure:
    - The background is the dominant color.
    - Each active foreground color appears as three disconnected same-color parts: a 1-pixel-wide rectangular fence, a larger body shell, and an interior 3-pixel L marker enclosed by that shell.
    - Fence interiors may contain background regions split by other colors; the number of enclosed 4-connected background regions is the movement distance.
    - The L marker is a rotated 2x2-minus-one-corner shape whose orientation indicates one cardinal translation direction.

    Rule:
    - For each active color, identify its rectangular fence, its body shell, and the enclosed 3-pixel L marker.
    - Count the number K of 4-connected background regions inside the fence interior.
    - Decode the L orientation to a cardinal direction and translate the body shell together with the marker by K cells along that direction.
    - Clear the object's original body and marker pixels to background while leaving every fence and every fence-interior obstacle unchanged.

    Key insight:
    - Direction and distance come from different same-color cues: the enclosed L marker encodes the motion axis, while the matching fence encodes the step count through its enclosed background-region count.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _mode_color(g)
    out = g.copy()

    for entity in _find_entities(g, bg):
        color = int(entity["color"])
        r0, c0, r1, c1 = entity["frame"]["bbox"]
        hole_count = _count_background_components(g[r0 + 1 : r1, c0 + 1 : c1] == bg)
        dr, dc = entity["marker"]["direction"]

        moving_cells = entity["body"]["cells"] + entity["marker"]["cells"]
        for r, c in moving_cells:
            out[r, c] = bg
        for r, c in moving_cells:
            nr = r + dr * hole_count
            nc = c + dc * hole_count
            assert 0 <= nr < h and 0 <= nc < w
            out[nr, nc] = color

    return out


def _crop_non_bg(patch: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return patch[r0 : r1 + 1, c0 : c1 + 1]


def _partition_positive(total: int, parts: int) -> list[int]:
    out = [1] * parts
    for _ in range(total - parts):
        out[random.randrange(parts)] += 1
    return out


def _make_frame_patch(color: int, blocker: int, hole_count: int) -> np.ndarray:
    if hole_count == 0:
        inner_h = random.randint(3, 7)
        inner_w = random.randint(3, 7)
        inner = np.full((inner_h, inner_w), blocker, dtype=int)
    elif hole_count == 1:
        inner_h = random.randint(4, 8)
        inner_w = random.randint(4, 8)
        inner = np.full((inner_h, inner_w), BG, dtype=int)
    else:
        split_vertical = random.choice([False, True])
        if split_vertical:
            inner_h = random.randint(4, 8)
            inner_w = random.randint(2 * hole_count - 1, 2 * hole_count + 2)
            widths = _partition_positive(inner_w - (hole_count - 1), hole_count)
            inner = np.full((inner_h, inner_w), BG, dtype=int)
            cursor = 0
            for idx, width in enumerate(widths):
                cursor += width
                if idx < hole_count - 1:
                    inner[:, cursor] = blocker
                    cursor += 1
        else:
            inner_h = random.randint(2 * hole_count - 1, 2 * hole_count + 2)
            inner_w = random.randint(4, 8)
            heights = _partition_positive(inner_h - (hole_count - 1), hole_count)
            inner = np.full((inner_h, inner_w), BG, dtype=int)
            cursor = 0
            for idx, height in enumerate(heights):
                cursor += height
                if idx < hole_count - 1:
                    inner[cursor, :] = blocker
                    cursor += 1

    patch = np.full((inner.shape[0] + 2, inner.shape[1] + 2), BG, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    patch[1:-1, 1:-1] = inner
    return patch


def _make_object_patch(color: int, direction: tuple[int, int]) -> np.ndarray:
    inner_h = random.randint(4, 6)
    inner_w = random.randint(4, 6)
    canvas = np.full((inner_h + 6, inner_w + 6), BG, dtype=int)

    r0, c0 = 2, 2
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    canvas[r0 : r1 + 1, c0] = color
    canvas[r0 : r1 + 1, c1] = color
    canvas[r0, c0 : c1 + 1] = color
    canvas[r1, c0 : c1 + 1] = color

    side_count = random.randint(1, 2)
    for side in random.sample(["top", "bottom", "left", "right"], side_count):
        length = random.randint(1, 2)
        if side in ("top", "bottom"):
            cc = random.randint(c0 + 1, c1 - 1)
            if side == "top":
                canvas[r0 - length : r0, cc] = color
            else:
                canvas[r1 + 1 : r1 + 1 + length, cc] = color
        else:
            rr = random.randint(r0 + 1, r1 - 1)
            if side == "left":
                canvas[rr, c0 - length : c0] = color
            else:
                canvas[rr, c1 + 1 : c1 + 1 + length] = color

    marker_r = random.randint(r0 + 2, r1 - 2)
    marker_c = random.randint(c0 + 2, c1 - 2)
    for dr, dc in DIR_TO_PATTERN[direction]:
        canvas[marker_r + dr, marker_c + dc] = color

    return _crop_non_bg(canvas)


def _rect_clear(mask: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = mask.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not mask[r0:r1, c0:c1].any()


def _mark_rect(mask: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    mask[top : top + height, left : left + width] = True


def _paste(canvas: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    region = canvas[top : top + h, left : left + w]
    keep = patch != BG
    region[keep] = patch[keep]


def _try_place_pair(
    inp: np.ndarray,
    expected: np.ndarray,
    input_mask: np.ndarray,
    output_mask: np.ndarray,
    color: int,
    blocker: int,
    direction: tuple[int, int],
    hole_count: int,
) -> bool:
    h, w = inp.shape
    frame = _make_frame_patch(color, blocker, hole_count)
    obj = _make_object_patch(color, direction)
    fh, fw = frame.shape
    oh, ow = obj.shape
    dr, dc = direction

    for _ in range(PAIR_ATTEMPTS):
        frame_top = random.randint(0, h - fh)
        frame_left = random.randint(0, w - fw)
        if not _rect_clear(input_mask, frame_top, frame_left, fh, fw):
            continue
        if not _rect_clear(output_mask, frame_top, frame_left, fh, fw):
            continue

        temp_in = input_mask.copy()
        temp_out = output_mask.copy()
        _mark_rect(temp_in, frame_top, frame_left, fh, fw)
        _mark_rect(temp_out, frame_top, frame_left, fh, fw)

        for _ in range(OBJECT_PLACEMENT_ATTEMPTS):
            obj_top = random.randint(0, h - oh)
            obj_left = random.randint(0, w - ow)
            out_top = obj_top + dr * hole_count
            out_left = obj_left + dc * hole_count
            if out_top < 0 or out_left < 0 or out_top + oh > h or out_left + ow > w:
                continue
            if not _rect_clear(temp_in, obj_top, obj_left, oh, ow):
                continue
            if not _rect_clear(temp_out, out_top, out_left, oh, ow):
                continue

            _mark_rect(temp_in, obj_top, obj_left, oh, ow)
            _mark_rect(temp_out, out_top, out_left, oh, ow)
            input_mask[:, :] = temp_in
            output_mask[:, :] = temp_out
            _paste(inp, frame, frame_top, frame_left)
            _paste(expected, frame, frame_top, frame_left)
            _paste(inp, obj, obj_top, obj_left)
            _paste(expected, obj, out_top, out_left)
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    inp = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)
    input_mask = np.zeros((h, w), dtype=bool)
    output_mask = np.zeros((h, w), dtype=bool)

    color_count = random.choice(OBJECT_COUNT_CHOICES)
    colors = random.sample(PALETTE, color_count + 1)
    object_colors = colors[:color_count]
    blocker = colors[-1]

    metas = []
    for color in object_colors:
        metas.append(
            {
                "color": color,
                "direction": random.choice(list(DIR_TO_PATTERN)),
                "holes": random.choice(HOLE_COUNT_CHOICES),
            }
        )
    random.shuffle(metas)

    for meta in metas:
        if not _try_place_pair(
            inp,
            expected,
            input_mask,
            output_mask,
            meta["color"],
            blocker,
            meta["direction"],
            meta["holes"],
        ):
            return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 4")
