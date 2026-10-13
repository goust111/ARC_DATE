from __future__ import annotations

import random

import numpy as np


FRAME_COLOR = 3
SIDES = ("top", "right", "bottom", "left")

GRID_H_RANGE = (22, 30)
GRID_W_RANGE = (22, 30)
INNER_H_RANGE = (11, 17)
INNER_W_RANGE = (11, 17)
BASE_DIM_CHOICES = (3, 4, 5)
GEN_RETRIES = 500
LAYOUT_RETRIES = 120


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _extract_patch(grid: np.ndarray, coords: np.ndarray, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = grid[coords[:, 0], coords[:, 1]]
    return patch


def _component_boundary(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    boundary = np.zeros_like(mask, dtype=bool)
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            boundary[r, c] = True
            continue
        if not (mask[r - 1, c] and mask[r + 1, c] and mask[r, c - 1] and mask[r, c + 1]):
            boundary[r, c] = True
    return boundary


def _frame_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return None
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    if np.array_equal(sub, expected):
        return int(r0), int(c0), int(r1), int(c1)
    return None


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        bbox = _frame_bbox(grid == color)
        if bbox is not None:
            candidates.append((color, bbox))
    assert len(candidates) == 1, "expected a unique rectangular fence"
    return candidates[0]


def _key_side(coord: tuple[int, int], frame_bbox: tuple[int, int, int, int]) -> str:
    r, c = coord
    r0, c0, r1, c1 = frame_bbox
    if r == r0 + 1:
        return "top"
    if c == c1 - 1:
        return "right"
    if r == r1 - 1:
        return "bottom"
    if c == c0 + 1:
        return "left"
    raise AssertionError("key must lie one cell inside a fence side")


def _keys_by_color(
    grid: np.ndarray, frame_bbox: tuple[int, int, int, int], bg: int, frame_color: int
) -> dict[int, dict[str, tuple[int, int] | str]]:
    r0, c0, r1, c1 = frame_bbox
    keys: dict[int, dict[str, tuple[int, int] | str]] = {}
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            color = int(grid[r, c])
            if color in (bg, frame_color):
                continue
            side = _key_side((r, c), frame_bbox)
            keys[color] = {"coord": (r, c), "side": side}
    assert len(keys) == 4, "expected four keys"
    assert {info["side"] for info in keys.values()} == set(SIDES)
    return keys


def _parse_object_colors(patch: np.ndarray, bg: int) -> tuple[int, int]:
    mask = patch != bg
    boundary = _component_boundary(mask)
    outer_values = np.unique(patch[boundary])
    outer_values = outer_values[outer_values != bg]
    inner_values = np.unique(patch[mask & ~boundary])
    inner_values = inner_values[inner_values != bg]
    assert outer_values.size == 1, "boundary must be monochrome"
    assert inner_values.size == 1, "interior must be monochrome"
    return int(outer_values[0]), int(inner_values[0])


def _orientation_side(patch: np.ndarray, bg: int) -> str:
    mask = patch != bg
    boundary = _component_boundary(mask)
    inner = np.argwhere(mask & ~boundary)
    outer = np.argwhere(boundary)
    assert inner.size > 0, "object must contain interior pixels"
    inner_center = inner.mean(axis=0)
    outer_center = outer.mean(axis=0)
    dr = float(inner_center[0] - outer_center[0])
    dc = float(inner_center[1] - outer_center[1])
    if abs(dr) >= abs(dc):
        return "top" if dr < 0 else "bottom"
    return "left" if dc < 0 else "right"


def _rotate_to_side(patch: np.ndarray, bg: int, target_side: str) -> np.ndarray:
    for turns in range(4):
        candidate = np.rot90(patch, turns)
        if _orientation_side(candidate, bg) == target_side:
            return candidate
    raise AssertionError("failed to rotate object to target side")


def _recolor_outer(patch: np.ndarray, bg: int, new_color: int) -> np.ndarray:
    out = np.array(patch, copy=True)
    boundary = _component_boundary(out != bg)
    out[boundary] = int(new_color)
    return out


def _frontier_anchor(mask: np.ndarray, side: str) -> int:
    if side == "top":
        cols = np.where(mask[0])[0]
        return int((int(cols.min()) + int(cols.max())) // 2)
    if side == "bottom":
        cols = np.where(mask[-1])[0]
        return int((int(cols.min()) + int(cols.max())) // 2)
    if side == "left":
        rows = np.where(mask[:, 0])[0]
        return int((int(rows.min()) + int(rows.max())) // 2)
    rows = np.where(mask[:, -1])[0]
    return int((int(rows.min()) + int(rows.max())) // 2)


def _placement_origin(
    patch: np.ndarray,
    bg: int,
    side: str,
    key_coord: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
) -> tuple[int, int]:
    mask = patch != bg
    h, w = patch.shape
    r0, c0, r1, c1 = frame_bbox
    kr, kc = key_coord
    if side == "top":
        return int(r0 + 1), int(kc - _frontier_anchor(mask, side))
    if side == "bottom":
        return int(r1 - h), int(kc - _frontier_anchor(mask, side))
    if side == "left":
        return int(kr - _frontier_anchor(mask, side)), int(c0 + 1)
    return int(kr - _frontier_anchor(mask, side)), int(c1 - w)


def _paste_non_bg(dst: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    view = dst[top : top + h, left : left + w]
    mask = patch != bg
    view[mask] = patch[mask]


def _draw_frame(dst: np.ndarray, frame_bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = frame_bbox
    dst[r0, c0 : c1 + 1] = int(color)
    dst[r1, c0 : c1 + 1] = int(color)
    dst[r0 : r1 + 1, c0] = int(color)
    dst[r0 : r1 + 1, c1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frame_color, frame_bbox = _find_frame(g, bg)
    keys = _keys_by_color(g, frame_bbox, bg, frame_color)

    out = np.array(g, copy=True)
    for info in keys.values():
        kr, kc = info["coord"]
        out[kr, kc] = int(bg)

    object_mask = (g != bg) & (g != frame_color)
    for coords in _find_components(object_mask):
        if coords.shape[0] <= 1:
            continue
        patch = _extract_patch(g, coords, bg)
        outer_color, _inner_color = _parse_object_colors(patch, bg)
        key_info = keys[outer_color]
        side = str(key_info["side"])
        rotated = _rotate_to_side(patch, bg, side)
        placed = _recolor_outer(rotated, bg, frame_color)
        top, left = _placement_origin(placed, bg, side, tuple(key_info["coord"]), frame_bbox)
        out[coords[:, 0], coords[:, 1]] = int(bg)
        _paste_non_bg(out, placed, top, left, bg)

    return out


def _centered_bar_lengths(axis_len: int) -> list[int]:
    return [n for n in range(1, axis_len - 1) if n % 2 == axis_len % 2]


def _make_mask(side: str) -> np.ndarray:
    base_h = random.choice(BASE_DIM_CHOICES)
    base_w = random.choice(BASE_DIM_CHOICES)
    base = np.ones((base_h, base_w), dtype=bool)

    if side in ("top", "bottom"):
        bar_len = random.choice(_centered_bar_lengths(base_w))
        start = (base_w - bar_len) // 2
        mask = np.zeros((base_h + 1, base_w), dtype=bool)
        if side == "top":
            mask[1:, :] = base
            mask[0, start : start + bar_len] = True
        else:
            mask[:-1, :] = base
            mask[-1, start : start + bar_len] = True
    else:
        bar_len = random.choice(_centered_bar_lengths(base_h))
        start = (base_h - bar_len) // 2
        mask = np.zeros((base_h, base_w + 1), dtype=bool)
        if side == "left":
            mask[:, 1:] = base
            mask[start : start + bar_len, 0] = True
        else:
            mask[:, :-1] = base
            mask[start : start + bar_len, -1] = True
    return mask


def _paint_object(mask: np.ndarray, outer_color: int, inner_color: int, bg: int) -> np.ndarray:
    boundary = _component_boundary(mask)
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(inner_color)
    patch[boundary] = int(outer_color)
    return patch


def _boxes_separated(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _build_output(
    shape: tuple[int, int],
    bg: int,
    frame_bbox: tuple[int, int, int, int],
    key_positions: dict[str, tuple[int, int]],
    placed_patches: dict[str, np.ndarray],
) -> np.ndarray | None:
    h, w = shape
    out = np.full((h, w), int(bg), dtype=int)
    _draw_frame(out, frame_bbox, FRAME_COLOR)
    occupied = out != bg
    r0, c0, r1, c1 = frame_bbox

    for side in SIDES:
        patch = placed_patches[side]
        top, left = _placement_origin(patch, bg, side, key_positions[side], frame_bbox)
        ph, pw = patch.shape
        if top < r0 + 1 or left < c0 + 1 or top + ph - 1 > r1 - 1 or left + pw - 1 > c1 - 1:
            return None
        mask = patch != bg
        view = occupied[top : top + ph, left : left + pw]
        if np.any(view[mask]):
            return None
        _paste_non_bg(out, patch, top, left, bg)
        view[mask] = True
    return out


def _place_external_objects(
    grid: np.ndarray, bg: int, frame_bbox: tuple[int, int, int, int], patches: list[np.ndarray]
) -> bool:
    h, w = grid.shape
    occupied_boxes = [frame_bbox]
    order = list(range(len(patches)))
    random.shuffle(order)
    order.sort(key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)

    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        candidates: list[tuple[int, int]] = []
        for r in range(h - ph + 1):
            for c in range(w - pw + 1):
                box = (r, c, r + ph - 1, c + pw - 1)
                if all(_boxes_separated(box, other, pad=1) for other in occupied_boxes):
                    candidates.append((r, c))
        if not candidates:
            return False
        top, left = random.choice(candidates)
        _paste_non_bg(grid, patch, top, left, bg)
        occupied_boxes.append((top, left, top + ph - 1, left + pw - 1))
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    bg = random.choice([c for c in range(10) if c != FRAME_COLOR])

    palette = [c for c in range(10) if c not in {bg, FRAME_COLOR}]
    random.shuffle(palette)
    key_colors = palette[:4]
    inner_colors = palette[4:]
    random.shuffle(key_colors)
    random.shuffle(inner_colors)
    key_color_by_side = {side: key_colors[i] for i, side in enumerate(SIDES)}

    input_patches: dict[str, np.ndarray] = {}
    placed_patches: dict[str, np.ndarray] = {}
    for i, side in enumerate(SIDES):
        source_side = random.choice(SIDES)
        raw = _paint_object(_make_mask(source_side), key_color_by_side[side], inner_colors[i], bg)
        input_patches[side] = raw
        rotated = _rotate_to_side(raw, bg, side)
        placed_patches[side] = _recolor_outer(rotated, bg, FRAME_COLOR)

    for _ in range(LAYOUT_RETRIES):
        inner_h = random.randint(*INNER_H_RANGE)
        inner_w = random.randint(*INNER_W_RANGE)
        frame_h = inner_h + 2
        frame_w = inner_w + 2
        if frame_h >= height - 1 or frame_w >= width - 1:
            continue
        r0 = random.randint(1, height - frame_h - 1)
        c0 = random.randint(1, width - frame_w - 1)
        frame_bbox = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
        key_positions = {
            "top": (r0 + 1, c0 + 1 + inner_w // 2),
            "right": (r0 + 1 + inner_h // 2, c0 + frame_w - 2),
            "bottom": (r0 + frame_h - 2, c0 + 1 + inner_w // 2),
            "left": (r0 + 1 + inner_h // 2, c0 + 1),
        }

        expected = _build_output((height, width), bg, frame_bbox, key_positions, placed_patches)
        if expected is None:
            continue

        grid = np.full((height, width), int(bg), dtype=int)
        _draw_frame(grid, frame_bbox, FRAME_COLOR)
        for side in SIDES:
            kr, kc = key_positions[side]
            grid[kr, kc] = int(key_color_by_side[side])

        if not _place_external_objects(grid, bg, frame_bbox, [input_patches[side] for side in SIDES]):
            continue
        if np.array_equal(main(grid), expected):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
