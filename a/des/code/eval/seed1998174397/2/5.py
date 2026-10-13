from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5
AZURE = 8

TRANSFORMS = ("flip_h", "flip_v", "rot_cw", "rot_ccw")
ANCHOR_COLOR_POOL = (1, 4, 6, 7, 9)


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    coords = np.argwhere(mask)
    assert coords.size > 0
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    cropped = _tight_crop(mask)
    return tuple("".join("1" if cell else "0" for cell in row) for row in cropped)


def _connected(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                nxt = (int(nr), int(nc))
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return len(seen) == int(mask.sum())


def _apply_transform(mask: np.ndarray, kind: str) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if kind == "flip_h":
        out = np.fliplr(mask)
    elif kind == "flip_v":
        out = np.flipud(mask)
    elif kind == "rot_cw":
        out = np.rot90(mask, -1)
    elif kind == "rot_ccw":
        out = np.rot90(mask, 1)
    else:
        raise ValueError(kind)
    return _tight_crop(out)


def _is_azure_frame(sub: np.ndarray) -> bool:
    if sub.shape != (3, 3):
        return False
    border = sub.copy()
    center = int(border[1, 1])
    border[1, 1] = AZURE
    return bool(np.all(border == AZURE) and center != AZURE)


def _infer_transform(before: np.ndarray, after: np.ndarray) -> str:
    before = _tight_crop(before)
    after_key = _mask_key(after)
    matches = [kind for kind in TRANSFORMS if _mask_key(_apply_transform(before, kind)) == after_key]
    assert len(matches) == 1
    return matches[0]


def _find_divider_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    assert len(rows) == 1
    return int(rows[0])


def _parse_template_mapping(grid: np.ndarray, divider_row: int) -> dict[int, str]:
    upper = grid[:divider_row, :]
    h, w = upper.shape
    mapping: dict[int, str] = {}
    for r in range(h - 2):
        for c in range(w - 2):
            if not _is_azure_frame(upper[r : r + 3, c : c + 3]):
                continue
            assert r + 5 < h and c >= 2 and c + 4 < w
            assert int(upper[r + 4, c + 1]) == RED
            color = int(upper[r + 1, c + 1])
            before = upper[r + 3 : r + 6, c - 2 : c + 1] != BLACK
            after = upper[r + 3 : r + 6, c + 2 : c + 5] != BLACK
            assert before.any() and after.any()
            assert color not in mapping
            mapping[color] = _infer_transform(before, after)
    assert mapping
    return mapping


def _extract_green_pattern(grid: np.ndarray, top: int, left: int) -> np.ndarray:
    assert top + 3 <= grid.shape[0] and left + 3 <= grid.shape[1]
    window = grid[top : top + 3, left : left + 3]
    assert int(window[0, 0]) == GREEN
    coords = np.argwhere(window == GREEN)
    assert coords.size > 0
    r1, c1 = coords.max(axis=0)
    mask = window[: int(r1) + 1, : int(c1) + 1] == GREEN
    assert _connected(mask)
    return _tight_crop(mask)


def _stamp_mask(canvas: np.ndarray, top: int, left: int, color: int, mask: np.ndarray) -> None:
    h, w = mask.shape
    patch = canvas[top : top + h, left : left + w]
    patch[mask] = int(color)


def _transform_workspace(grid: np.ndarray, divider_row: int, mapping: dict[int, str]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    h, w = out.shape
    anchor_count = 0
    for r in range(divider_row + 1, h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in mapping:
                continue
            right = c + 1 < w and int(grid[r, c + 1]) == GREEN
            down = r + 1 < h and int(grid[r + 1, c]) == GREEN
            assert right != down
            if right:
                top, left = r, c + 1
            else:
                top, left = r + 1, c
            pattern = _extract_green_pattern(grid, top, left)
            transformed = _apply_transform(pattern, mapping[color])
            out[top : top + 3, left : left + 3] = BLACK
            _stamp_mask(out, top, left, GREEN, transformed)
            anchor_count += 1
    assert anchor_count > 0
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer color-coded transforms from template examples, then apply them to each green motif beside
    the matching colored anchor in the workspace.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    divider_row = _find_divider_row(g)
    mapping = _parse_template_mapping(g, divider_row)
    return _transform_workspace(g, divider_row, mapping)


def _enumerate_valid_masks() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    seen: set[tuple[str, ...]] = set()
    for bits in range(1, 1 << 9):
        if bits.bit_count() not in (3, 4):
            continue
        arr = np.zeros((3, 3), dtype=bool)
        for idx in range(9):
            if bits & (1 << idx):
                arr[idx // 3, idx % 3] = True
        cropped = _tight_crop(arr)
        if cropped.shape[0] > 3 or cropped.shape[1] > 3:
            continue
        if not _connected(cropped):
            continue
        transformed = {_mask_key(_apply_transform(cropped, kind)) for kind in TRANSFORMS}
        if len(transformed) != 4:
            continue
        key = _mask_key(cropped)
        if key in seen:
            continue
        seen.add(key)
        masks.append(cropped)
    assert masks
    return masks


VALID_MASKS = _enumerate_valid_masks()


def _render_mask(mask: np.ndarray, color: int) -> np.ndarray:
    out = np.full((3, 3), BLACK, dtype=int)
    _stamp_mask(out, 0, 0, color, mask)
    return out


def _can_place_rect(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not bool(occupied[r0:r1, c0:c1].any())


def _mark_rect(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    occupied[top : top + height, left : left + width] = True


def _draw_frame(grid: np.ndarray, top: int, left: int, center_color: int) -> None:
    grid[top : top + 3, left : left + 3] = AZURE
    grid[top + 1, left + 1] = int(center_color)


def _paint_workspace_pattern(grid: np.ndarray, top: int, left: int, mask: np.ndarray) -> None:
    _stamp_mask(grid, top, left, GREEN, mask)


def _make_case() -> tuple[np.ndarray, np.ndarray] | None:
    selected_colors = random.sample(list(ANCHOR_COLOR_POOL), 4)
    transform_order = list(TRANSFORMS)
    random.shuffle(transform_order)
    color_to_transform = dict(zip(selected_colors, transform_order, strict=True))

    template_order = selected_colors[:]
    random.shuffle(template_order)
    top_margin = random.randint(0, 1)
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 4)
    gaps = [random.randint(1, 3) for _ in range(3)]
    tile_width = 7
    upper_h = top_margin + 6

    tile_starts: list[int] = []
    cursor = left_margin
    for idx in range(4):
        tile_starts.append(cursor)
        cursor += tile_width
        if idx < 3:
            cursor += gaps[idx]
    template_w = cursor + right_margin

    work_h = random.randint(10, 15)
    total_w = template_w + random.randint(0, 5)
    divider_row = upper_h
    total_h = upper_h + 1 + work_h

    grid = np.full((total_h, total_w), BLACK, dtype=int)
    grid[divider_row, :] = GREY

    box_top = top_margin
    for tile_left, color in zip(tile_starts, template_order, strict=True):
        sample_mask = random.choice(VALID_MASKS)
        result_mask = _apply_transform(sample_mask, color_to_transform[color])
        _draw_frame(grid, box_top, tile_left + 2, color)
        before_patch = _render_mask(sample_mask, color)
        after_patch = _render_mask(result_mask, color)
        grid[box_top + 3 : box_top + 6, tile_left : tile_left + 3] = before_patch
        grid[box_top + 4, tile_left + 3] = RED
        grid[box_top + 3 : box_top + 6, tile_left + 4 : tile_left + 7] = after_patch

    n_anchors = random.randint(4, 8)
    anchor_colors = selected_colors[:] + [random.choice(selected_colors) for _ in range(n_anchors - 4)]
    random.shuffle(anchor_colors)
    orientations = ["right"] * (n_anchors // 2) + ["down"] * (n_anchors - n_anchors // 2)
    random.shuffle(orientations)

    occupied = np.zeros((work_h, total_w), dtype=bool)
    objects: list[tuple[int, int, np.ndarray, str]] = []
    for color, orientation in zip(anchor_colors, orientations, strict=True):
        mask = random.choice(VALID_MASKS)
        foot_h, foot_w = ((3, 4) if orientation == "right" else (4, 3))
        placed = False
        for _ in range(300):
            top = random.randint(0, work_h - foot_h)
            left = random.randint(0, total_w - foot_w)
            if not _can_place_rect(occupied, top, left, foot_h, foot_w, pad=1):
                continue
            _mark_rect(occupied, top, left, foot_h, foot_w)
            anchor_r = divider_row + 1 + top
            anchor_c = left
            grid[anchor_r, anchor_c] = color
            if orientation == "right":
                pattern_top, pattern_left = anchor_r, anchor_c + 1
            else:
                pattern_top, pattern_left = anchor_r + 1, anchor_c
            _paint_workspace_pattern(grid, pattern_top, pattern_left, mask)
            objects.append((pattern_top, pattern_left, mask, color_to_transform[color]))
            placed = True
            break
        if not placed:
            return None

    expected = grid.copy()
    for top, left, mask, transform_name in objects:
        expected[top : top + 3, left : left + 3] = BLACK
        _stamp_mask(expected, top, left, GREEN, _apply_transform(mask, transform_name))
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        case = _make_case()
        if case is None:
            continue
        inp, expected = case
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
