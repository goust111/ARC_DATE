from __future__ import annotations

import random
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, label_mask


BG = 0
RED = 2
INPUT_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
GRID_ROWS = (18, 24)
GRID_COLS = (18, 24)
TOTAL_SPRITES = (8, 12)
TYPE_COUNTS = (2, 3)
WINNER_K_VALUES = (3, 4, 5)
MAX_ATTEMPTS = 500


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(x) for x in row) for row in mask.astype(int))


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, _r1, _c1 = map(int, component["bbox"])
    rel = component["coords"] - np.array([r0, c0], dtype=int)
    mask = np.zeros((rel[:, 0].max() + 1, rel[:, 1].max() + 1), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _coords_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(mask)]


def _build_mask_pool() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for h, w in ((2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2)):
        area = h * w
        for cells in range(3, min(6, area) + 1):
            for picks in combinations(range(area), cells):
                mask = np.zeros((h, w), dtype=bool)
                for idx in picks:
                    mask[idx // w, idx % w] = True
                if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
                    continue
                _, n = label_mask(mask, connectivity=8)
                if n == 1:
                    masks.append(mask)
    return masks


_ALL_MASKS = _build_mask_pool()
_WINNER_MASKS = [m for m in _ALL_MASKS if int(m.sum()) <= 4 and m.shape[0] <= 3 and m.shape[1] <= 3]
_LOSER_MASKS = [m for m in _ALL_MASKS if int(m.sum()) >= 4]


def _paint_stamp(
    out: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
    color: int,
    *,
    red_override: bool,
    bg: int = BG,
) -> None:
    h, w = mask.shape
    for dr, dc in _coords_from_mask(mask):
        rr = int(top + dr)
        cc = int(left + dc)
        if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
            if red_override:
                out[rr, cc] = RED
            elif int(out[rr, cc]) == int(bg):
                out[rr, cc] = int(color)


def _stamp_sequence(
    out: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
    color: int,
    period: int,
    dc_step: int,
    *,
    bg: int = BG,
) -> None:
    h, w = mask.shape
    step = 1
    rr = top + h
    cc = left + dc_step
    while rr < out.shape[0] and cc < out.shape[1] and cc + w > 0:
        _paint_stamp(out, mask, rr, cc, color, red_override=(step % period == 0), bg=bg)
        rr += h
        cc += dc_step
        step += 1


def _render_output(
    grid_shape: tuple[int, int],
    originals: list[dict],
    winner_mask: np.ndarray,
    winner_color: int,
    period: int,
    *,
    include_left: bool = True,
    include_right: bool = True,
    forced_period: int | None = None,
) -> np.ndarray:
    out = np.full(grid_shape, BG, dtype=int)
    h, w = winner_mask.shape
    for item in originals:
        top = int(item["top"])
        left = int(item["left"])
        _paint_stamp(out, winner_mask, top, left, winner_color, red_override=False)
        if include_right:
            _stamp_sequence(
                out,
                winner_mask,
                top,
                left,
                winner_color,
                forced_period if forced_period is not None else period,
                w,
            )
        if include_left:
            _stamp_sequence(
                out,
                winner_mask,
                top,
                left,
                winner_color,
                forced_period if forced_period is not None else period,
                -w,
            )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the uniquely most frequent colored sprite type, then stamp it down both diagonals with a recolor period equal to its frequency.

    Input structure:
    - A black background contains 8-connected monochrome sprites that do not touch each other.
    - There are 2-3 sprite types, where type identity is the tight bbox mask together with the sprite color.
    - Exactly one type appears more often than every other type, and its tight bbox has height H and width W.

    Rule:
    - Find every monochrome 8-connected component and normalize it to its tight bbox mask.
    - Count identical `(color, mask)` types and select the unique most frequent type; let its frequency be K.
    - Erase every non-winning sprite.
    - Keep each winning original in place, then from its upper-left bbox corner stamp the same mask at offsets `(H, W), (2H, 2W), ...` and `(H, -W), (2H, -2W), ...`, clipping anything outside the grid.
    - Along each diagonal direction, number copies from 1 after the original; copies whose index is a multiple of K are recolored red, while the others keep the winning sprite color.

    Key insight:
    - The same frequency count K determines both which sprite survives and which diagonal copies turn red, while the winning sprite bbox sets the diagonal stride.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(get_bg(g))
    components = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert components, "expected at least one foreground sprite"

    groups: dict[tuple[int, tuple[tuple[int, ...], ...]], list[dict]] = defaultdict(list)
    masks: dict[tuple[int, tuple[tuple[int, ...], ...]], np.ndarray] = {}
    for component in components:
        color = int(component["color"])
        mask = _component_mask(component)
        key = (color, _mask_signature(mask))
        groups[key].append(component)
        masks[key] = mask

    max_count = max(len(items) for items in groups.values())
    winners = [key for key, items in groups.items() if len(items) == max_count]
    assert len(winners) == 1, "winner must be unique"

    winner_key = winners[0]
    winner_color = int(winner_key[0])
    winner_mask = masks[winner_key]
    period = len(groups[winner_key])

    originals = []
    for component in groups[winner_key]:
        r0, c0, _r1, _c1 = map(int, component["bbox"])
        originals.append({"top": r0, "left": c0})

    return _render_output(g.shape, originals, winner_mask, winner_color, period, include_left=True, include_right=True)


def _neighbor_block(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    blocked = set()
    h, w = shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    blocked.add((rr, cc))
    return blocked


def _sprite_cells(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    return {(int(top + r), int(left + c)) for r, c in _coords_from_mask(mask)}


def _trajectory_info(mask: np.ndarray, top: int, left: int, grid_shape: tuple[int, int], period: int) -> dict:
    h, w = mask.shape
    original = _sprite_cells(mask, top, left)
    copies: set[tuple[int, int]] = set()
    left_cells: set[tuple[int, int]] = set()
    right_cells: set[tuple[int, int]] = set()
    red_cells: set[tuple[int, int]] = set()
    left_len = 0
    right_len = 0

    for dc_step, bucket in ((w, right_cells), (-w, left_cells)):
        step = 1
        rr = top + h
        cc = left + dc_step
        while rr < grid_shape[0] and cc < grid_shape[1] and cc + w > 0:
            visible = _sprite_cells(mask, rr, cc)
            visible = {cell for cell in visible if 0 <= cell[0] < grid_shape[0] and 0 <= cell[1] < grid_shape[1]}
            copies.update(visible)
            bucket.update(visible)
            if step % period == 0:
                red_cells.update(visible)
            if dc_step > 0:
                right_len += 1
            else:
                left_len += 1
            rr += h
            cc += dc_step
            step += 1

    return {
        "top": int(top),
        "left": int(left),
        "original": original,
        "all_output": original | copies,
        "copy_cells": copies,
        "left_cells": left_cells,
        "right_cells": right_cells,
        "red_cells": red_cells,
        "left_len": left_len,
        "right_len": right_len,
    }


def _count_options(type_count: int) -> list[list[int]]:
    options = []
    for total in range(TOTAL_SPRITES[0], TOTAL_SPRITES[1] + 1):
        if type_count == 2:
            for k in WINNER_K_VALUES:
                other = total - k
                if 1 <= other < k:
                    options.append([k, other])
        else:
            for k in WINNER_K_VALUES:
                for a in range(1, k):
                    b = total - k - a
                    if 1 <= b < k:
                        if max(a, b) < k and max(a, b) >= 2:
                            options.append([k, a, b])
    return options


_COUNT_OPTIONS = {count: _count_options(count) for count in TYPE_COUNTS}


def _pick_type_specs(type_count: int) -> tuple[list[int], list[np.ndarray], list[int]] | None:
    counts = random.choice(_COUNT_OPTIONS[type_count]).copy()
    winner_count = counts[0]

    for _ in range(200):
        winner_mask = random.choice(_WINNER_MASKS)
        used = {_mask_signature(winner_mask)}
        masks = [winner_mask]
        ok = True
        for _ in range(type_count - 1):
            choices = [m for m in _LOSER_MASKS if _mask_signature(m) not in used]
            if not choices:
                ok = False
                break
            picked = random.choice(choices)
            used.add(_mask_signature(picked))
            masks.append(picked)
        if not ok:
            continue

        winner_total = int(winner_mask.sum()) * winner_count
        loser_totals = [int(mask.sum()) * counts[idx] for idx, mask in enumerate(masks[1:], start=1)]
        if max(loser_totals) <= winner_total:
            continue

        colors = random.sample(INPUT_COLORS, type_count)
        return counts, masks, colors
    return None


def _winner_candidates(grid_shape: tuple[int, int], winner_mask: np.ndarray, period: int) -> list[dict]:
    h, w = winner_mask.shape
    candidates = []
    for top in range(1, grid_shape[0] - h):
        for left in range(1, grid_shape[1] - w):
            info = _trajectory_info(winner_mask, top, left, grid_shape, period)
            if info["left_len"] < 1 or info["right_len"] < 1:
                continue
            candidates.append(info)
    random.shuffle(candidates)
    return candidates


def _pick_winner_layout(grid_shape: tuple[int, int], winner_mask: np.ndarray, period: int) -> list[dict] | None:
    candidates = _winner_candidates(grid_shape, winner_mask, period)
    if not candidates:
        return None

    target_count = period
    for _ in range(150):
        chosen: list[dict] = []
        blocked_originals: set[tuple[int, int]] = set()
        blocked_output: set[tuple[int, int]] = set()
        pool = candidates.copy()
        random.shuffle(pool)
        for cand in pool:
            if cand["original"] & blocked_originals:
                continue
            if cand["all_output"] & blocked_output:
                continue
            chosen.append(cand)
            blocked_originals |= _neighbor_block(cand["original"], grid_shape)
            blocked_output |= cand["all_output"]
            if len(chosen) == target_count:
                break

        if len(chosen) != target_count:
            continue
        if not any(item["red_cells"] for item in chosen):
            continue
        if not any(max(item["left_len"], item["right_len"]) >= period for item in chosen):
            continue
        return chosen
    return None


def _place_other_instances(
    grid_shape: tuple[int, int],
    masks: list[np.ndarray],
    counts: list[int],
    colors: list[int],
    winner_layout: list[dict],
) -> np.ndarray | None:
    grid = np.full(grid_shape, BG, dtype=int)
    blocked = set()
    future_output = set()

    for item in winner_layout:
        future_output |= item["all_output"]
        for r, c in item["original"]:
            grid[r, c] = colors[0]
        blocked |= _neighbor_block(item["original"], grid_shape)

    for type_idx in range(1, len(masks)):
        mask = masks[type_idx]
        color = colors[type_idx]
        coords = _coords_from_mask(mask)
        h, w = mask.shape

        for _ in range(counts[type_idx]):
            legal = []
            for top in range(1, grid_shape[0] - h):
                for left in range(1, grid_shape[1] - w):
                    cells = {(top + r, left + c) for r, c in coords}
                    if cells & blocked:
                        continue
                    if cells & future_output:
                        continue
                    legal.append((top, left, cells))
            if not legal:
                return None
            top, left, cells = random.choice(legal)
            for r, c in cells:
                grid[r, c] = color
            blocked |= _neighbor_block(cells, grid_shape)

    return grid


def _validate_semantics(
    grid: np.ndarray,
    counts: list[int],
    masks: list[np.ndarray],
    colors: list[int],
    winner_layout: list[dict],
) -> bool:
    expected = _render_output(
        grid.shape,
        [{"top": item["top"], "left": item["left"]} for item in winner_layout],
        masks[0],
        colors[0],
        counts[0],
    )
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, grid):
        return False
    if not np.any(actual == RED):
        return False
    if not any(item["left_cells"] for item in winner_layout):
        return False
    if not any(item["right_cells"] for item in winner_layout):
        return False

    pixel_totals = [int(mask.sum()) * counts[idx] for idx, mask in enumerate(masks)]
    pixel_winner = int(np.argmax(pixel_totals))
    if pixel_winner == 0:
        return False
    wrong_by_pixels = _render_output(
        grid.shape,
        [{"top": item["top"], "left": item["left"]} for item in winner_layout],
        masks[pixel_winner],
        colors[pixel_winner],
        counts[pixel_winner],
    )
    if np.array_equal(wrong_by_pixels, expected):
        return False

    if counts[0] != 3:
        wrong_period = _render_output(
            grid.shape,
            [{"top": item["top"], "left": item["left"]} for item in winner_layout],
            masks[0],
            colors[0],
            counts[0],
            forced_period=3,
        )
        if np.array_equal(wrong_period, expected):
            return False

    if np.array_equal(
        _render_output(
            grid.shape,
            [{"top": item["top"], "left": item["left"]} for item in winner_layout],
            masks[0],
            colors[0],
            counts[0],
            include_left=False,
        ),
        expected,
    ):
        return False
    if np.array_equal(
        _render_output(
            grid.shape,
            [{"top": item["top"], "left": item["left"]} for item in winner_layout],
            masks[0],
            colors[0],
            counts[0],
            include_right=False,
        ),
        expected,
    ):
        return False

    comps = find_components(grid, bg=BG, connectivity=8, monochromatic=True, return_format="dict")
    if len(comps) != sum(counts):
        return False
    seen = defaultdict(int)
    for comp in comps:
        key = (int(comp["color"]), _mask_signature(_component_mask(comp)))
        seen[key] += 1
    target_keys = {(colors[idx], _mask_signature(masks[idx])): counts[idx] for idx in range(len(counts))}
    return seen == target_keys


def _attempt_generate() -> np.ndarray | None:
    grid_shape = (
        random.randint(*GRID_ROWS),
        random.randint(*GRID_COLS),
    )
    type_count = random.choice(TYPE_COUNTS)
    specs = _pick_type_specs(type_count)
    if specs is None:
        return None
    counts, masks, colors = specs

    winner_layout = _pick_winner_layout(grid_shape, masks[0], counts[0])
    if winner_layout is None:
        return None

    grid = _place_other_instances(grid_shape, masks, counts, colors, winner_layout)
    if grid is None:
        return None

    if not _validate_semantics(grid, counts, masks, colors, winner_layout):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
