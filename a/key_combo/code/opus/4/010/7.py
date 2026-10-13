from __future__ import annotations

import random

import numpy as np


BG = 0
LEFT_FG = 1
RIGHT_FG = 2
OUT_FG = 3
LEFT_MARK = 4
SEP = 5
RIGHT_MARK = 7

MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for c in range(grid.shape[1]):
        color = int(grid[0, c])
        if color == bg:
            continue
        if not np.all(grid[:, c] == color):
            continue
        if c == 0 or c == grid.shape[1] - 1:
            continue
        if c == grid.shape[1] - 1 - c:
            candidates.append((c, color))
    assert candidates, "expected a full-height separator column splitting two equal panels"
    preferred = [item for item in candidates if item[1] == SEP]
    chosen = preferred[0] if preferred else candidates[0]
    return int(chosen[0]), int(chosen[1])


def _find_marker_color_and_rect(panel: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in np.unique(panel):
        value = int(color)
        if value == bg:
            continue
        mask = panel == value
        box = _bbox(mask)
        assert box is not None
        r0, c0, r1, c1 = box
        area = int(mask.sum())
        rect_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        if area > 1 and area == rect_area:
            candidates.append((value, box))
    assert len(candidates) == 1, "expected exactly one solid marker rectangle per panel"
    return candidates[0]


def _candidate_fill_value(
    restored: np.ndarray,
    marker_mask: np.ndarray,
    r: int,
    c: int,
) -> int | None:
    height, width = restored.shape
    mirror = (r, width - 1 - c)
    candidates = [mirror]
    if marker_mask[mirror]:
        candidates.extend(
            [
                (height - 1 - r, width - 1 - c),
                (height - 1 - mirror[0], width - 1 - mirror[1]),
            ]
        )

    values: list[int] = []
    for rr, cc in candidates:
        val = int(restored[rr, cc])
        if val != -1:
            values.append(val)

    if not values:
        return None
    if len(set(values)) != 1:
        return None
    return int(values[0])


def _restore_panel(panel: np.ndarray, marker_color: int, fg_color: int) -> np.ndarray:
    restored = panel.copy().astype(int)
    marker_mask = restored == int(marker_color)
    restored[marker_mask] = -1

    unknown = {tuple(map(int, coord)) for coord in np.argwhere(marker_mask)}
    for _ in range(len(unknown) + 1):
        progress = False
        for r, c in list(unknown):
            value = _candidate_fill_value(restored, marker_mask, r, c)
            if value is None:
                continue
            restored[r, c] = int(value)
            unknown.remove((r, c))
            progress = True
        if not unknown or not progress:
            break

    assert not unknown, "marker-covered cells must be recoverable from symmetry clues"
    return np.where(restored == int(fg_color), int(fg_color), BG)


def main(grid: np.ndarray) -> np.ndarray:
    """Recover each panel's masked mirror-symmetric foreground, then output their positional foreground intersection.

    Input structure:
    - The grid has black background and one full-height separator column splitting it into two equal-width panels.
    - Each panel contains one vertically symmetric foreground pattern plus one solid marker rectangle of a unique color.
    - The marker hides part of the panel pattern and does not define the output by itself.

    Rule:
    - Find the separator and crop the left and right panels.
    - In each panel, identify the unique solid marker rectangle and the panel's foreground color.
    - For every marker-covered cell, recover its hidden value from the vertical mirror cell; if that mirror is also covered, use the 180-degree counterparts of the covered pair.
    - Remove both marker rectangles, then compare the restored panels cell by cell and paint green exactly where both restored panels contain foreground.

    Key insight:
    - The green output is the intersection of two latent symmetric shapes, so both independent restorations must happen before the aligned comparison.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    sep_col, _sep_color = _find_separator(g, bg)
    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]
    assert left.shape == right.shape, "panels must have equal size"

    left_marker, _left_rect = _find_marker_color_and_rect(left, bg)
    right_marker, _right_rect = _find_marker_color_and_rect(right, bg)

    left_colors = set(map(int, np.unique(left))) - {bg, left_marker}
    right_colors = set(map(int, np.unique(right))) - {bg, right_marker}
    assert len(left_colors) == 1 and len(right_colors) == 1, "each panel must have one foreground color"
    left_fg = left_colors.pop()
    right_fg = right_colors.pop()

    left_restored = _restore_panel(left, left_marker, left_fg)
    right_restored = _restore_panel(right, right_marker, right_fg)
    return np.where((left_restored != BG) & (right_restored != BG), OUT_FG, BG).astype(int)


def _empty_mask(shape: tuple[int, int]) -> np.ndarray:
    return np.zeros(shape, dtype=bool)


def _mirror_rect_pair_mask(height: int, width: int, r0: int, h: int, c0: int, w: int) -> np.ndarray:
    mask = _empty_mask((height, width))
    mask[r0 : r0 + h, c0 : c0 + w] = True
    rc0 = width - (c0 + w)
    mask[r0 : r0 + h, rc0 : rc0 + w] = True
    return mask


def _center_pair_mask(height: int, width: int, r0: int, h: int) -> np.ndarray:
    mid = width // 2
    mask = _empty_mask((height, width))
    for r in range(r0, r0 + h):
        mask[r, mid] = True
        mask[height - 1 - r, mid] = True
    return mask


def _primitive_catalog(height: int, width: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    left_half = width // 2
    rects: list[np.ndarray] = []
    seen: set[bytes] = set()
    max_h = min(4, height)
    max_w = min(3, left_half)
    for h in range(1, max_h + 1):
        for w in range(1, max_w + 1):
            for r0 in range(0, height - h + 1):
                for c0 in range(0, left_half - w + 1):
                    mask = _mirror_rect_pair_mask(height, width, r0, h, c0, w)
                    key = mask.tobytes()
                    if key not in seen:
                        seen.add(key)
                        rects.append(mask)

    centers: list[np.ndarray] = []
    if width % 2 == 1:
        top_limit = height // 2
        for h in range(1, min(3, top_limit) + 1):
            for r0 in range(0, top_limit - h + 1):
                mask = _center_pair_mask(height, width, r0, h)
                key = mask.tobytes()
                if key not in seen:
                    seen.add(key)
                    centers.append(mask)
    return rects, centers


def _mask_overlaps(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.any(a & b))


def _pick_disjoint_subset(
    masks: list[np.ndarray],
    blocked: np.ndarray,
    count_lo: int,
    count_hi: int,
) -> list[np.ndarray] | None:
    if not masks:
        return None
    target = random.randint(count_lo, count_hi)
    order = list(range(len(masks)))
    random.shuffle(order)

    chosen: list[np.ndarray] = []
    used = blocked.copy()
    for idx in order:
        mask = masks[idx]
        if _mask_overlaps(mask, used):
            continue
        chosen.append(mask)
        used |= mask
        if len(chosen) == target:
            return chosen

    return None


def _is_solid_rectangle(mask: np.ndarray) -> bool:
    box = _bbox(mask)
    if box is None:
        return False
    r0, c0, r1, c1 = box
    return int(mask.sum()) == int((r1 - r0 + 1) * (c1 - c0 + 1))


def _build_scene_masks(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool] | None:
    rect_prims, center_prims = _primitive_catalog(height, width)
    want_center = width % 2 == 1 and center_prims and random.random() < 0.55

    for _ in range(120):
        common_parts: list[np.ndarray] = []
        blocked = _empty_mask((height, width))

        if want_center:
            center_choices = [mask for mask in center_prims if int(mask.sum()) >= 4]
            if not center_choices:
                center_choices = center_prims
            center_mask = random.choice(center_choices)
            common_parts.append(center_mask)
            blocked |= center_mask

        rect_count_lo = 1 if common_parts else 2
        rect_count_hi = 3
        common_rects = _pick_disjoint_subset(rect_prims, blocked, rect_count_lo, rect_count_hi)
        if common_rects is None:
            continue
        common_parts.extend(common_rects)
        common = np.zeros((height, width), dtype=bool)
        for mask in common_parts:
            common |= mask

        if len(common_parts) < 2:
            continue
        if not (4 <= int(common.sum()) <= max(8, (height * width) // 3)):
            continue
        if _is_solid_rectangle(common):
            continue

        extra_a_parts = _pick_disjoint_subset(rect_prims, common, 1, 3)
        if extra_a_parts is None:
            continue
        extra_a = np.zeros_like(common)
        for mask in extra_a_parts:
            if _mask_overlaps(mask, extra_a):
                break
            extra_a |= mask
        else:
            blocked_b = common | extra_a
            extra_b_parts = _pick_disjoint_subset(rect_prims, blocked_b, 1, 3)
            if extra_b_parts is None:
                continue
            extra_b = np.zeros_like(common)
            for mask in extra_b_parts:
                if _mask_overlaps(mask, extra_b):
                    break
                extra_b |= mask
            else:
                left_mask = common | extra_a
                right_mask = common | extra_b
                if np.array_equal(left_mask, right_mask):
                    continue
                if _is_solid_rectangle(left_mask) or _is_solid_rectangle(right_mask):
                    continue
                density = max(int(left_mask.sum()), int(right_mask.sum()))
                if density >= (height * width * 3) // 5:
                    continue
                return common, left_mask, right_mask, want_center

    return None


def _coords_set(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, coord)) for coord in np.argwhere(mask)}


def _rect_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _side_marker_candidates(
    panel_mask: np.ndarray,
    common_mask: np.ndarray,
) -> list[dict]:
    height, width = panel_mask.shape
    mid = width // 2
    candidates: list[dict] = []
    side_ranges = [
        ("left", 0, (mid - 1) if width % 2 == 1 else (mid - 1)),
        ("right", (mid + 1) if width % 2 == 1 else mid, width - 1),
    ]
    for mode, c_lo, c_hi in side_ranges:
        if c_lo > c_hi:
            continue
        max_w = min(4, c_hi - c_lo + 1)
        for h in range(1, min(4, height) + 1):
            for w in range(1, max_w + 1):
                for r0 in range(0, height - h + 1):
                    for c0 in range(c_lo, c_hi - w + 2):
                        r1, c1 = r0 + h - 1, c0 + w - 1
                        box = panel_mask[r0 : r1 + 1, c0 : c1 + 1]
                        fg_count = int(box.sum())
                        if fg_count < 2:
                            continue
                        common_count = int(common_mask[r0 : r1 + 1, c0 : c1 + 1].sum())
                        if common_count < 1:
                            continue
                        cells = _rect_cells(r0, c0, r1, c1)
                        candidates.append(
                            {
                                "mode": mode,
                                "rect": (r0, c0, r1, c1),
                                "cells": cells,
                                "hidden_common": cells & _coords_set(common_mask),
                                "fg_count": fg_count,
                                "area": len(cells),
                            }
                        )
    return candidates


def _axis_marker_candidates(
    panel_mask: np.ndarray,
    common_mask: np.ndarray,
) -> list[dict]:
    height, width = panel_mask.shape
    if width % 2 == 0:
        return []
    mid = width // 2
    candidates: list[dict] = []
    half = height // 2
    for top in (True, False):
        start_lo = 0 if top else height - half
        start_hi = half - 1 if top else height - 1
        for h in range(2, min(4, half) + 1):
            for r0 in range(start_lo, start_hi - h + 2):
                rows = list(range(r0, r0 + h))
                if top:
                    if rows[-1] >= half:
                        continue
                else:
                    if rows[0] < height - half:
                        continue
                ok = True
                for r in rows:
                    rr = height - 1 - r
                    if rr in rows:
                        ok = False
                        break
                    if bool(panel_mask[r, mid]) != bool(panel_mask[rr, mid]):
                        ok = False
                        break
                if not ok:
                    continue
                fg_count = int(panel_mask[rows, mid].sum())
                common_count = int(common_mask[rows, mid].sum())
                if fg_count < 2 or common_count < 1:
                    continue
                rect = (rows[0], mid, rows[-1], mid)
                cells = {(r, mid) for r in rows}
                candidates.append(
                    {
                        "mode": "axis",
                        "rect": rect,
                        "cells": cells,
                        "hidden_common": cells & _coords_set(common_mask),
                        "fg_count": fg_count,
                        "area": len(cells),
                    }
                )
    return candidates


def _choose_marker_pair(
    common_mask: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    want_axis: bool,
) -> tuple[dict, dict] | None:
    left_side = _side_marker_candidates(left_mask, common_mask)
    right_side = _side_marker_candidates(right_mask, common_mask)
    left_axis = _axis_marker_candidates(left_mask, common_mask)
    right_axis = _axis_marker_candidates(right_mask, common_mask)
    if not left_side or not right_side:
        return None

    common_cells = _coords_set(common_mask)
    left_modes = [left_side]
    right_modes = [right_side]
    if left_axis:
        left_modes.append(left_axis)
    if right_axis:
        right_modes.append(right_axis)

    require_axis = want_axis and (left_axis or right_axis) and random.random() < 0.7
    for _ in range(300):
        left_pool = random.choice(left_modes)
        right_pool = random.choice(right_modes)
        left_cand = random.choice(left_pool)
        right_cand = random.choice(right_pool)
        if require_axis and left_cand["mode"] != "axis" and right_cand["mode"] != "axis":
            continue

        hidden_left = set(left_cand["hidden_common"])
        hidden_right = set(right_cand["hidden_common"])
        visible_common = common_cells - hidden_left - hidden_right
        if not hidden_left or not hidden_right:
            continue
        if not (hidden_left - hidden_right):
            continue
        if not (hidden_right - hidden_left):
            continue
        if len(visible_common) < 2:
            continue
        return left_cand, right_cand
    return None


def _render_panel(mask: np.ndarray, fg_color: int, mark_color: int, rect: tuple[int, int, int, int]) -> np.ndarray:
    panel = np.where(mask, int(fg_color), BG).astype(int)
    r0, c0, r1, c1 = rect
    panel[r0 : r1 + 1, c0 : c1 + 1] = int(mark_color)
    return panel


def _naive_intersection(left_panel: np.ndarray, right_panel: np.ndarray) -> np.ndarray:
    left_visible = left_panel == LEFT_FG
    right_visible = right_panel == RIGHT_FG
    return np.where(left_visible & right_visible, OUT_FG, BG).astype(int)


def _restored_intersection(left_mask: np.ndarray, right_mask: np.ndarray) -> np.ndarray:
    return np.where(left_mask & right_mask, OUT_FG, BG).astype(int)


def _scene_semantics_ok(
    inp: np.ndarray,
    expected: np.ndarray,
    common_mask: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    left_rect: tuple[int, int, int, int],
    right_rect: tuple[int, int, int, int],
) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    left_panel = inp[:, : common_mask.shape[1]]
    right_panel = inp[:, common_mask.shape[1] + 1 :]
    naive = _naive_intersection(left_panel, right_panel)
    if np.array_equal(naive, expected):
        return False

    left_visible = left_panel == LEFT_FG
    right_visible = right_panel == RIGHT_FG
    if np.array_equal(_restored_intersection(left_visible, right_mask), expected):
        return False
    if np.array_equal(_restored_intersection(left_mask, right_visible), expected):
        return False

    left_hidden = int((common_mask & ~left_visible).sum())
    right_hidden = int((common_mask & ~right_visible).sum())
    if left_hidden < 1 or right_hidden < 1:
        return False

    if int(common_mask.sum()) < 4:
        return False
    if int(common_mask.sum()) >= int(left_mask.sum()) or int(common_mask.sum()) >= int(right_mask.sum()):
        return False

    lr0, lc0, lr1, lc1 = left_rect
    rr0, rc0, rr1, rc1 = right_rect
    if int(left_mask[lr0 : lr1 + 1, lc0 : lc1 + 1].sum()) < 2:
        return False
    if int(right_mask[rr0 : rr1 + 1, rc0 : rc1 + 1].sum()) < 2:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    width = random.randint(8, 13)
    if 2 * width + 1 > 30:
        return None
    height = random.randint(8, 14)

    built = _build_scene_masks(height, width)
    if built is None:
        return None
    common_mask, left_mask, right_mask, want_axis = built

    chosen = _choose_marker_pair(common_mask, left_mask, right_mask, want_axis)
    if chosen is None:
        return None
    left_marker, right_marker = chosen

    left_panel = _render_panel(left_mask, LEFT_FG, LEFT_MARK, left_marker["rect"])
    right_panel = _render_panel(right_mask, RIGHT_FG, RIGHT_MARK, right_marker["rect"])

    inp = np.full((height, 2 * width + 1), BG, dtype=int)
    inp[:, width] = SEP
    inp[:, :width] = left_panel
    inp[:, width + 1 :] = right_panel

    expected = np.where(common_mask, OUT_FG, BG).astype(int)
    if not _scene_semantics_ok(
        inp,
        expected,
        common_mask,
        left_mask,
        right_mask,
        left_marker["rect"],
        right_marker["rect"],
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-panel symmetric intersection task")
