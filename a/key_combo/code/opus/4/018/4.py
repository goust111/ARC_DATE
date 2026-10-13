from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg, paste, random_free_location


BG = Color.BLACK
RED = Color.RED
DIVIDER = Color.GREY
SPRITE_COLORS = [Color.BLUE, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
SLOT_COUNT = 6

SLOT_HEIGHT_OPTIONS = [3, 4, 5]
LEFT_WIDTH_RANGE = (5, 8)
RIGHT_WIDTH_RANGE = (5, 8)
MAX_GENERATION_ATTEMPTS = 400


def _mask_signature(arr: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(map(int, row)) for row in arr)


def _mask_from_cells(h: int, w: int, cells: tuple[int, ...]) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    idx = np.array(cells, dtype=int)
    mask[idx // w, idx % w] = True
    return mask


def _is_connected_cells(h: int, w: int, cells: tuple[int, ...]) -> bool:
    cellset = set(cells)
    stack = [cells[0]]
    seen = {cells[0]}
    while stack:
        cell = stack.pop()
        r, c = divmod(cell, w)
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w:
                neighbor = nr * w + nc
                if neighbor in cellset and neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
    return len(seen) == len(cells)


def _build_mask_catalog(
    *,
    min_h: int,
    max_h: int,
    min_w: int,
    max_w: int,
    min_area: int,
    max_area: int,
) -> dict[tuple[int, int, int], list[np.ndarray]]:
    catalog: dict[tuple[int, int, int], list[np.ndarray]] = {}
    for h in range(min_h, max_h + 1):
        for w in range(min_w, max_w + 1):
            total = h * w
            for area in range(min_area, min(max_area, total) + 1):
                accepted: list[np.ndarray] = []
                for cells in combinations(range(total), area):
                    rows = [cell // w for cell in cells]
                    cols = [cell % w for cell in cells]
                    if min(rows) != 0 or max(rows) != h - 1:
                        continue
                    if min(cols) != 0 or max(cols) != w - 1:
                        continue
                    if not _is_connected_cells(h, w, cells):
                        continue
                    accepted.append(_mask_from_cells(h, w, cells))
                if accepted:
                    catalog[(h, w, area)] = accepted
    return catalog


RED_MASKS = _build_mask_catalog(
    min_h=1,
    max_h=3,
    min_w=1,
    max_w=4,
    min_area=1,
    max_area=5,
)
SPRITE_MASKS = _build_mask_catalog(
    min_h=2,
    max_h=4,
    min_w=2,
    max_w=4,
    min_area=4,
    max_area=8,
)
RED_KEYS = list(RED_MASKS)
SPRITE_KEYS = list(SPRITE_MASKS)


def _find_divider_col(grid: np.ndarray) -> int:
    divider_cols = [c for c in range(grid.shape[1]) if bool(np.all(grid[:, c] == DIVIDER))]
    assert len(divider_cols) == 1, "expected one full-height grey divider column"
    divider_col = divider_cols[0]
    assert 0 < divider_col < grid.shape[1] - 1
    return divider_col


def _extract_sprite_patch(left: np.ndarray) -> np.ndarray:
    components = find_components(
        left,
        bg=BG,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    assert len(components) == 1, "expected one multicolor sprite in the left region"
    sprite = crop_bbox(left, components[0]["bbox"])
    colors = set(map(int, np.unique(sprite))) - {int(BG)}
    assert len(colors) >= 2, "sprite must contain at least two visible colors"
    return sprite


def _slot_red_component(slot: np.ndarray) -> dict:
    components = find_components(
        slot,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[RED],
        return_format="dict",
    )
    assert len(components) == 1, "each slot must contain exactly one red component"
    component = components[0]
    assert 1 <= int(component["area"]) <= 5
    return component


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert int(get_bg(g)) == int(BG), "expected black background"

    h, w = g.shape
    divider_col = _find_divider_col(g)
    left = g[:, :divider_col]
    right = g[:, divider_col + 1 :]
    assert right.size > 0 and left.size > 0
    assert h % SLOT_COUNT == 0, "grid height must split evenly into six slots"
    slot_h = h // SLOT_COUNT
    sprite_patch = _extract_sprite_patch(left)
    sh, sw = sprite_patch.shape
    assert sh <= slot_h and sw <= right.shape[1], "sprite must fit inside a right-side slot"

    slot_info: list[dict] = []
    for slot_idx in range(SLOT_COUNT):
        slot_top = slot_idx * slot_h
        slot = right[slot_top : slot_top + slot_h, :]
        comp = _slot_red_component(slot)
        crop = crop_bbox(slot, comp["bbox"])
        slot_info.append(
            {
                "slot_idx": slot_idx,
                "area": int(comp["area"]),
                "bbox": tuple(map(int, comp["bbox"])),
                "crop": crop,
                "signature": _mask_signature(crop),
                "coords": np.asarray(comp["coords"], dtype=int)
                + np.array([slot_top, divider_col + 1], dtype=int),
            }
        )

    counts = Counter(info["signature"] for info in slot_info)
    unique_signatures = [signature for signature, n in counts.items() if n == 1]
    repeated_signatures = [signature for signature, n in counts.items() if n == SLOT_COUNT - 1]
    assert len(unique_signatures) == 1, "expected exactly one outlier slot pattern"
    assert len(repeated_signatures) == 1, "expected exactly one repeated normal pattern"

    outlier_idx = next(
        info["slot_idx"] for info in slot_info if info["signature"] == unique_signatures[0]
    )
    area = slot_info[outlier_idx]["area"]
    target_idx = outlier_idx + area
    assert target_idx < SLOT_COUNT, "encoded slot offset must stay inside the six slots"

    target_top = target_idx * slot_h
    stamp_r = target_top + (slot_h - sh) // 2
    stamp_c = divider_col + 1 + (right.shape[1] - sw) // 2

    return {
        "grid": g,
        "divider_col": divider_col,
        "left_width": divider_col,
        "right_width": int(right.shape[1]),
        "slot_h": int(slot_h),
        "sprite_patch": sprite_patch,
        "slot_info": slot_info,
        "outlier_idx": int(outlier_idx),
        "outlier_area": int(area),
        "target_idx": int(target_idx),
        "stamp_r": int(stamp_r),
        "stamp_c": int(stamp_c),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the unique right-side red outlier as an offset signal, then stamp the left sprite into the encoded slot.

    Input structure:
    - The background is black, and one full-height grey column divides the grid into left and right regions.
    - The left region contains one multicolor 4-connected sprite.
    - The right region is split into six equal-height slots, each with one centered red 4-connected shape.

    Rule:
    - Crop each slot's red shape to its minimal bounding box and compare the exact cropped patterns, including black cells inside the crop.
    - Find the only slot whose cropped red pattern occurs once; let its top-to-bottom index be S and its red area be A.
    - Compute the target slot T = S + A, erase the outlier red shape from slot S, and stamp the cropped left sprite into the center of slot T.
    - Keep the left region, the divider column, and every other normal slot unchanged.

    Key insight:
    - The odd red shape is not the destination itself: its slot gives the starting point, and its pixel count encodes how far downward to move before stamping the left sprite.
    """
    info = _analyze_grid(np.asarray(grid, dtype=int))
    out = info["grid"].copy()

    outlier = info["slot_info"][info["outlier_idx"]]
    coords = outlier["coords"]
    out[coords[:, 0], coords[:, 1]] = BG

    paste(out, info["sprite_patch"], info["stamp_r"], info["stamp_c"])
    return out


def _colorize_sprite(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    color_count = random.randint(2, min(3, len(coords)))
    colors = random.sample(SPRITE_COLORS, color_count)
    random.shuffle(colors)
    order = coords.copy()
    np.random.shuffle(order)

    if color_count == 1:
        groups = [order]
    else:
        cuts = sorted(random.sample(range(1, len(order)), color_count - 1))
        groups = np.split(order, cuts)

    patch = np.full(mask.shape, BG, dtype=int)
    for color, group in zip(colors, groups):
        patch[group[:, 0], group[:, 1]] = int(color)
    return patch


def _make_red_patch(mask: np.ndarray) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = RED
    return patch


def _choose_sprite_patch(slot_h: int, left_w: int, right_w: int) -> np.ndarray | None:
    max_w = min(left_w - 2, right_w, 4)
    keys = [key for key in SPRITE_KEYS if key[0] <= slot_h and key[1] <= max_w]
    if not keys:
        return None
    key = random.choice(keys)
    mask = random.choice(SPRITE_MASKS[key]).copy()
    return _colorize_sprite(mask)


def _choose_normal_red_patch(sprite_patch: np.ndarray) -> np.ndarray | None:
    sh, sw = sprite_patch.shape
    keys = [key for key in RED_KEYS if key[0] <= sh and key[1] <= sw and 2 <= key[2] <= 5]
    if not keys:
        return None
    key = random.choice(keys)
    return _make_red_patch(random.choice(RED_MASKS[key]).copy())


def _choose_outlier_red_patch(slot_h: int, right_w: int, area: int, normal_signature: tuple[tuple[int, ...], ...]) -> np.ndarray | None:
    keys = [key for key in RED_KEYS if key[0] <= slot_h and key[1] <= right_w and key[2] == area]
    random.shuffle(keys)
    for key in keys:
        masks = RED_MASKS[key][:]
        random.shuffle(masks)
        for mask in masks:
            patch = _make_red_patch(mask.copy())
            if _mask_signature(patch) != normal_signature:
                return patch
    return None


def _slot_origin(slot_h: int, right0: int, right_w: int, slot_idx: int, patch: np.ndarray) -> tuple[int, int]:
    ph, pw = patch.shape
    return (
        slot_idx * slot_h + (slot_h - ph) // 2,
        right0 + (right_w - pw) // 2,
    )


def _render_input(
    *,
    slot_h: int,
    left_w: int,
    right_w: int,
    sprite_patch: np.ndarray,
    normal_patch: np.ndarray,
    outlier_patch: np.ndarray,
    outlier_idx: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    h = slot_h * SLOT_COUNT
    w = left_w + 1 + right_w
    grid = np.full((h, w), BG, dtype=int)
    grid[:, left_w] = DIVIDER

    left_canvas = np.full((h, left_w), BG, dtype=int)
    sprite_r, sprite_c = random_free_location(
        left_canvas,
        sprite_patch,
        transparent=BG,
        border=1,
        padding=0,
    )
    paste(left_canvas, sprite_patch, sprite_r, sprite_c)
    grid[:, :left_w] = left_canvas

    right0 = left_w + 1
    for slot_idx in range(SLOT_COUNT):
        patch = outlier_patch if slot_idx == outlier_idx else normal_patch
        r0, c0 = _slot_origin(slot_h, right0, right_w, slot_idx, patch)
        paste(grid, patch, r0, c0, transparent=BG)

    return grid, (sprite_r, sprite_c)


def _expected_output_from_metadata(
    grid: np.ndarray,
    *,
    slot_h: int,
    left_w: int,
    right_w: int,
    sprite_patch: np.ndarray,
    outlier_patch: np.ndarray,
    outlier_idx: int,
    target_idx: int,
) -> np.ndarray:
    out = grid.copy()
    right0 = left_w + 1

    outlier_r0, outlier_c0 = _slot_origin(slot_h, right0, right_w, outlier_idx, outlier_patch)
    red_coords = np.argwhere(outlier_patch == RED)
    out[outlier_r0 + red_coords[:, 0], outlier_c0 + red_coords[:, 1]] = BG

    stamp_r = target_idx * slot_h + (slot_h - sprite_patch.shape[0]) // 2
    stamp_c = right0 + (right_w - sprite_patch.shape[1]) // 2
    paste(out, sprite_patch, stamp_r, stamp_c)
    return out


def _semantic_validation(
    grid: np.ndarray,
    *,
    sprite_patch: np.ndarray,
    slot_h: int,
    left_w: int,
    right_w: int,
    outlier_idx: int,
    outlier_area: int,
    target_idx: int,
    expected: np.ndarray,
) -> bool:
    try:
        info = _analyze_grid(grid)
    except Exception:
        return False

    if info["slot_h"] != slot_h or info["left_width"] != left_w or info["right_width"] != right_w:
        return False
    if info["outlier_idx"] != outlier_idx or info["outlier_area"] != outlier_area:
        return False
    if info["target_idx"] != target_idx:
        return False
    if not np.array_equal(info["sprite_patch"], sprite_patch):
        return False
    if not np.array_equal(main(grid), expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    slot_h = random.choice(SLOT_HEIGHT_OPTIONS)
    left_w = random.randint(*LEFT_WIDTH_RANGE)
    right_w = random.randint(*RIGHT_WIDTH_RANGE)

    sprite_patch = _choose_sprite_patch(slot_h, left_w, right_w)
    if sprite_patch is None:
        return None

    normal_patch = _choose_normal_red_patch(sprite_patch)
    if normal_patch is None:
        return None

    outlier_area = random.randint(1, 5)
    outlier_idx = random.randint(0, SLOT_COUNT - outlier_area - 1)
    target_idx = outlier_idx + outlier_area

    outlier_patch = _choose_outlier_red_patch(
        slot_h,
        right_w,
        outlier_area,
        _mask_signature(normal_patch),
    )
    if outlier_patch is None:
        return None

    grid, _sprite_loc = _render_input(
        slot_h=slot_h,
        left_w=left_w,
        right_w=right_w,
        sprite_patch=sprite_patch,
        normal_patch=normal_patch,
        outlier_patch=outlier_patch,
        outlier_idx=outlier_idx,
    )
    expected = _expected_output_from_metadata(
        grid,
        slot_h=slot_h,
        left_w=left_w,
        right_w=right_w,
        sprite_patch=sprite_patch,
        outlier_patch=outlier_patch,
        outlier_idx=outlier_idx,
        target_idx=target_idx,
    )

    if np.array_equal(grid, expected):
        return None
    if not _semantic_validation(
        grid,
        sprite_patch=sprite_patch,
        slot_h=slot_h,
        left_w=left_w,
        right_w=right_w,
        outlier_idx=outlier_idx,
        outlier_area=outlier_area,
        target_idx=target_idx,
        expected=expected,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
