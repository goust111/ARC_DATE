from __future__ import annotations

import random
from collections import Counter
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


try:
    _ARC_UTILS = SourceFileLoader(
        "arc_utils_fable_7_5",
        "/home/lixinyu/snap/new/solvers/arc_utils.py",
    ).load_module()
    Color = _ARC_UTILS.Color
    bbox = _ARC_UTILS.bbox
    crop_bbox = _ARC_UTILS.crop_bbox
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9
        ALL_COLORS = list(range(10))

    def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = box
        return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


POSES = ("identity", "fliplr", "flipud", "rot180")
UL = (0, 0)
UR = (0, 1)
LL = (1, 0)
LR = (1, 1)
CORNER_ORDER = (UL, UR, LL, LR)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _apply_pose(arr: np.ndarray, pose: str) -> np.ndarray:
    if pose == "identity":
        return arr.copy()
    if pose == "fliplr":
        return np.fliplr(arr)
    if pose == "flipud":
        return np.flipud(arr)
    if pose == "rot180":
        return np.rot90(arr, 2)
    raise ValueError(f"unknown pose: {pose}")


def _pose_corner(corner: tuple[int, int], pose: str) -> tuple[int, int]:
    r, c = map(int, corner)
    if pose == "identity":
        return r, c
    if pose == "fliplr":
        return r, 1 - c
    if pose == "flipud":
        return 1 - r, c
    if pose == "rot180":
        return 1 - r, 1 - c
    raise ValueError(f"unknown pose: {pose}")


def _corner_to_rc(corner: tuple[int, int], h: int, w: int) -> tuple[int, int]:
    return (0 if corner[0] == 0 else h - 1, 0 if corner[1] == 0 else w - 1)


def _rc_to_corner(r: int, c: int, h: int, w: int) -> tuple[int, int]:
    mapping = {
        (0, 0): UL,
        (0, w - 1): UR,
        (h - 1, 0): LL,
        (h - 1, w - 1): LR,
    }
    if (r, c) not in mapping:
        raise AssertionError("marker must sit on a slot corner")
    return mapping[(r, c)]


def _find_divider(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    candidates: list[int] = []
    for c in range(1, w - 1):
        col = grid[:, c]
        color = int(col[0])
        if color == bg:
            continue
        if not np.all(col == color):
            continue
        if int(np.sum(grid == color)) != h:
            continue
        candidates.append(c)
    assert len(candidates) == 1, "expected exactly one full-height divider column"
    return candidates[0]


def _extract_template(left: np.ndarray, bg: int) -> np.ndarray:
    box = bbox(left != bg)
    assert box is not None, "left side must contain a template"
    return crop_bbox(left, box)


def _analyze_template(template: np.ndarray, bg: int) -> tuple[int, int, tuple[int, int]]:
    colors = [int(color) for color in np.unique(template) if int(color) != bg]
    assert len(colors) == 2, "template must use exactly one body color plus one anchor color"

    counts = {color: int(np.sum(template == color)) for color in colors}
    anchor_candidates = [color for color, count in counts.items() if count == 1]
    assert len(anchor_candidates) == 1, "template must have exactly one anchor pixel"
    anchor_color = anchor_candidates[0]
    body_color = next(color for color in colors if color != anchor_color)

    ar, ac = map(int, np.argwhere(template == anchor_color)[0])
    h, w = template.shape
    assert (ar, ac) in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    return body_color, anchor_color, _rc_to_corner(ar, ac, h, w)


def _pose_for_target(anchor_corner: tuple[int, int], target_corner: tuple[int, int]) -> str:
    matches = [pose for pose in POSES if _pose_corner(anchor_corner, pose) == target_corner]
    assert len(matches) == 1
    return matches[0]


def _stamp_slot(template: np.ndarray, anchor_corner: tuple[int, int], target_corner: tuple[int, int]) -> np.ndarray:
    return _apply_pose(template, _pose_for_target(anchor_corner, target_corner))


def _slot_slices(
    right_shape: tuple[int, int],
    slot_h: int,
    slot_w: int,
) -> list[tuple[int, int, int, int]]:
    h, w = map(int, right_shape)
    assert h % slot_h == 0 and w % slot_w == 0
    out = []
    for r0 in range(0, h, slot_h):
        for c0 in range(0, w, slot_w):
            out.append((r0, c0, r0 + slot_h, c0 + slot_w))
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Stamp the left template into each right-side slot using the flip pose that aligns the template anchor to the slot marker corner.

    Input structure:
    - A single full-height divider column splits the grid into a left template area and a right slot area.
    - The left side contains one rectangular template whose non-background cells use one body color plus one unique anchor color.
    - The right side is tiled into slots the same size as the template; each non-empty slot contains exactly one marker pixel on one of its four corners.

    Rule:
    - Crop the left template and locate its unique anchor pixel, which lies on a template corner.
    - Partition the right side into equal template-sized slots.
    - For every marked slot, choose from identity, left-right flip, up-down flip, and 180-degree rotation the unique pose that moves the template anchor onto the marker corner, then stamp the full posed template over that slot.
    - Leave empty slots, the divider, and the original left template unchanged.

    Key insight:
    - The corner marker is a pose address: it does not describe a color, only which corner the anchor pixel must occupy after the allowed mirror/rotation operations.
    """
    g = np.asarray(input_grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    divider_c = _find_divider(g, bg)
    left = g[:, :divider_c]
    right = g[:, divider_c + 1 :]

    template = _extract_template(left, bg)
    slot_h, slot_w = map(int, template.shape)
    _body_color, _anchor_color, anchor_corner = _analyze_template(template, bg)
    assert right.shape[0] % slot_h == 0 and right.shape[1] % slot_w == 0

    out = g.copy()
    for r0, c0, r1, c1 in _slot_slices(right.shape, slot_h, slot_w):
        slot = right[r0:r1, c0:c1]
        coords = np.argwhere(slot != bg)
        if coords.size == 0:
            continue
        assert coords.shape == (1, 2), "each non-empty slot must contain exactly one marker"
        mr, mc = map(int, coords[0])
        target_corner = _rc_to_corner(mr, mc, slot_h, slot_w)
        stamped = _stamp_slot(template, anchor_corner, target_corner)
        out[r0:r1, divider_c + 1 + c0 : divider_c + 1 + c1] = stamped

    return out


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                cells.add((nr, nc))
    return list(cells)


def _base_ul_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[:, w - 1] = True
    return mask


def _generate_template(h: int, w: int, bg: int, body: int, anchor: int) -> tuple[np.ndarray, tuple[int, int]]:
    anchor_corner = random.choice(CORNER_ORDER)
    base = _base_ul_mask(h, w)
    max_area = max(h + w - 1, int(round(0.72 * h * w)))
    target_area = random.randint(h + w - 1, max_area)
    while int(base.sum()) < target_area:
        choices = _frontier(base)
        if not choices:
            break
        base[random.choice(choices)] = True

    pose = _pose_for_target(UL, anchor_corner)
    mask = _apply_pose(base.astype(int), pose).astype(bool)

    template = np.full((h, w), bg, dtype=int)
    template[mask] = body
    ar, ac = _corner_to_rc(anchor_corner, h, w)
    template[ar, ac] = anchor

    variants = {_apply_pose(template, pose_name).tobytes() for pose_name in POSES}
    if len(variants) < 4:
        return _generate_template(h, w, bg, body, anchor)
    return template, anchor_corner


def _layout_options() -> list[tuple[int, int, int, int, int]]:
    options: list[tuple[int, int, int, int, int]] = []
    for slot_h in range(3, 7):
        for slot_w in range(3, 7):
            for slot_rows in range(2, min(4, 30 // slot_h) + 1):
                for slot_cols in range(2, min(4, 30 // slot_w) + 1):
                    if slot_rows * slot_cols < 6:
                        continue
                    for left_w in range(slot_w + 2, slot_w + 6):
                        total_w = left_w + 1 + slot_cols * slot_w
                        total_h = slot_rows * slot_h
                        if total_w <= 30 and total_h <= 30:
                            options.append((slot_h, slot_w, slot_rows, slot_cols, left_w))
    return options


LAYOUT_OPTIONS = _layout_options()


def _corner_assignments(slot_count: int) -> list[tuple[int, int]]:
    assert slot_count >= 6
    marked = random.randint(4, slot_count - 1)
    corners = list(CORNER_ORDER)
    random.shuffle(corners)
    assignments = corners[:]
    while len(assignments) < marked:
        assignments.append(random.choice(CORNER_ORDER))
    random.shuffle(assignments)
    return assignments


def _render_case() -> tuple[np.ndarray, np.ndarray]:
    slot_h, slot_w, slot_rows, slot_cols, left_w = random.choice(LAYOUT_OPTIONS)
    h = slot_rows * slot_h
    w = left_w + 1 + slot_cols * slot_w

    bg, body, anchor, marker, divider = random.sample(Color.ALL_COLORS, 5)
    template, anchor_corner = _generate_template(slot_h, slot_w, bg, body, anchor)

    inp = np.full((h, w), bg, dtype=int)
    divider_c = left_w
    inp[:, divider_c] = divider

    top = random.randint(0, h - slot_h)
    left = random.randint(0, left_w - slot_w)
    inp[top : top + slot_h, left : left + slot_w] = template

    expected = inp.copy()
    slot_boxes = _slot_slices((h, slot_cols * slot_w), slot_h, slot_w)
    slot_order = list(range(len(slot_boxes)))
    random.shuffle(slot_order)
    assignments = _corner_assignments(len(slot_boxes))

    for slot_idx, corner in zip(slot_order, assignments):
        r0, c0, r1, c1 = slot_boxes[slot_idx]
        mr, mc = _corner_to_rc(corner, slot_h, slot_w)
        inp[r0:r1, divider_c + 1 + c0 : divider_c + 1 + c1] = bg
        inp[r0 + mr, divider_c + 1 + c0 + mc] = marker
        expected[r0:r1, divider_c + 1 + c0 : divider_c + 1 + c1] = _stamp_slot(
            template,
            anchor_corner,
            corner,
        )

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        inp, expected = _render_case()
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid corner-stamp input")
