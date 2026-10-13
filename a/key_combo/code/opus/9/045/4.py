from __future__ import annotations

import random
from typing import Any

import numpy as np


BACKGROUND = 0
SEPARATOR_COLOR = 3
INPUT_COLORS = (1, 2, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400

_GEN_CALL_COUNT = 0


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_separator(crop: np.ndarray) -> tuple[str, int]:
    h, w = crop.shape
    empty_rows = [r for r in range(1, h - 1) if np.all(crop[r, :] == BACKGROUND)]
    empty_cols = [c for c in range(1, w - 1) if np.all(crop[:, c] == BACKGROUND)]
    separators = [("row", r) for r in empty_rows] + [("col", c) for c in empty_cols]
    assert len(separators) == 1, "expected exactly one interior separator"
    return separators[0]


def _paint_output(grid: np.ndarray, rotate_first: bool, rotate_second: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    box = _bbox(g != BACKGROUND)
    assert box is not None, "expected non-background foreground"
    r0, c0, r1, c1 = box
    crop = g[r0 : r1 + 1, c0 : c1 + 1]
    axis, index = _find_separator(crop)

    out = g.copy()
    out_crop = crop.copy()
    if axis == "row":
        first = crop[:index, :]
        second = crop[index + 1 :, :]
        if rotate_first:
            out_crop[:index, :] = np.rot90(first, 2)
        if rotate_second:
            out_crop[index + 1 :, :] = np.rot90(second, 2)
        out[r0 : r1 + 1, c0 : c1 + 1] = out_crop
        out[r0 + index, :] = SEPARATOR_COLOR
    else:
        first = crop[:, :index]
        second = crop[:, index + 1 :]
        if rotate_first:
            out_crop[:, :index] = np.rot90(first, 2)
        if rotate_second:
            out_crop[:, index + 1 :] = np.rot90(second, 2)
        out[r0 : r1 + 1, c0 : c1 + 1] = out_crop
        out[:, c0 + index] = SEPARATOR_COLOR
    return out


def _analyze(grid: np.ndarray) -> dict[str, Any]:
    g = np.asarray(grid, dtype=int)
    box = _bbox(g != BACKGROUND)
    assert box is not None, "expected non-empty foreground"
    r0, c0, r1, c1 = box
    crop = g[r0 : r1 + 1, c0 : c1 + 1]
    axis, index = _find_separator(crop)

    if axis == "row":
        first = crop[:index, :]
        second = crop[index + 1 :, :]
    else:
        first = crop[:, :index]
        second = crop[:, index + 1 :]

    first_count = int(np.count_nonzero(first))
    second_count = int(np.count_nonzero(second))
    assert first_count > 0 and second_count > 0

    if first_count < second_count:
        smaller = "first"
    elif second_count < first_count:
        smaller = "second"
    else:
        smaller = "both"

    return {
        "box": box,
        "axis": axis,
        "index": int(index),
        "first": first.copy(),
        "second": second.copy(),
        "first_count": first_count,
        "second_count": second_count,
        "smaller": smaller,
    }


def _rotate_tight_bbox(region: np.ndarray) -> np.ndarray:
    out = region.copy()
    box = _bbox(out != BACKGROUND)
    if box is None:
        return out
    r0, c0, r1, c1 = box
    out[r0 : r1 + 1, c0 : c1 + 1] = np.rot90(out[r0 : r1 + 1, c0 : c1 + 1], 2)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Locate the unique empty divider inside the foreground bbox, rotate the smaller side 180 degrees, and paint the divider green.

    Input structure:
    - A black background contains one multicolor foreground pattern with a tight bounding box near the center.
    - Strictly inside that foreground bounding box there is exactly one all-black row or all-black column.
    - Foreground pixels exist on both sides of that empty divider, and the two side regions may be irregular or disconnected.

    Rule:
    - Find the tight bounding box of all non-black pixels.
    - Inside that box, detect the unique fully black interior row or column and use it as the separator.
    - Split the bounding box into the two rectangular side regions defined by that separator.
    - Compare the numbers of non-black pixels in the two side regions; rotate the smaller region 180 degrees inside its own side rectangle, leave the larger region unchanged, and rotate both if the counts tie.
    - Finally repaint the separator across the entire grid with green.

    Key insight:
    - The empty interior line is both the partition that defines the comparison regions and the final marked axis, so solving requires separator detection, area comparison, and region-local rotation in that exact order.
    """
    info = _analyze(grid)
    if info["smaller"] == "first":
        return _paint_output(grid, rotate_first=True, rotate_second=False)
    if info["smaller"] == "second":
        return _paint_output(grid, rotate_first=False, rotate_second=True)
    return _paint_output(grid, rotate_first=True, rotate_second=True)


def _required_side_rows(side_h: int, width: int, forbidden_cols: set[int]) -> set[tuple[int, int]]:
    allowed_cols = [c for c in range(width) if c not in forbidden_cols]
    assert allowed_cols
    req: set[tuple[int, int]] = set()
    for r in range(side_h):
        req.add((r, random.choice(allowed_cols)))
    return req


def _choose_counts(required_small: int, allowed_small: int, required_large: int, allowed_large: int) -> tuple[int, int] | None:
    small_lo = max(required_small, int(np.ceil(allowed_small * 0.35)))
    small_hi = min(allowed_small, int(np.floor(allowed_small * 0.62)))
    large_lo = max(required_large, int(np.ceil(allowed_large * 0.42)))
    large_hi = min(allowed_large, int(np.floor(allowed_large * 0.82)))
    if small_lo > small_hi or large_lo > large_hi:
        return None

    small_choices = list(range(small_lo, small_hi + 1))
    random.shuffle(small_choices)
    for small_count in small_choices:
        gap = random.randint(2, 6)
        lo = max(large_lo, small_count + gap)
        if lo <= large_hi:
            return small_count, random.randint(lo, large_hi)
    return None


def _fill_side(side_h: int, width: int, required: set[tuple[int, int]], target_count: int, forbidden_cols: set[int]) -> np.ndarray:
    mask = np.zeros((side_h, width), dtype=bool)
    for r, c in required:
        mask[r, c] = True

    remaining = [
        (r, c)
        for r in range(side_h)
        for c in range(width)
        if c not in forbidden_cols and not mask[r, c]
    ]
    random.shuffle(remaining)
    need = target_count - int(mask.sum())
    assert need >= 0
    for r, c in remaining[:need]:
        mask[r, c] = True
    return mask


def _colorize(mask: np.ndarray) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    assert len(cells) >= 2
    color_count = min(random.randint(2, 4), len(cells))
    palette = random.sample(INPUT_COLORS, color_count)
    values = list(palette)
    while len(values) < len(cells):
        values.append(random.choice(palette))
    random.shuffle(values)
    for (r, c), value in zip(cells, values):
        patch[r, c] = int(value)
    return patch


def _build_local_row_case(small_on_top: bool) -> np.ndarray | None:
    total_h = random.randint(8, 10)
    width = random.randint(8, 10)
    splits = [(t, total_h - 1 - t) for t in range(3, total_h - 3)]
    if not splits:
        return None
    top_h, bottom_h = random.choice(splits)

    small_is_top = bool(small_on_top)
    forbidden_small_col = random.choice([0, width - 1])
    top_forbidden = {forbidden_small_col} if small_is_top else set()
    bottom_forbidden = {forbidden_small_col} if not small_is_top else set()

    top_req = _required_side_rows(top_h, width, top_forbidden)
    bottom_req = _required_side_rows(bottom_h, width, bottom_forbidden)

    for c in range(width):
        if c == forbidden_small_col:
            owner = "bottom" if small_is_top else "top"
        else:
            owner = random.choice(["top", "bottom"])
        if owner == "top":
            top_req.add((random.randrange(top_h), c))
        else:
            bottom_req.add((random.randrange(bottom_h), c))

    top_allowed = top_h * (width - len(top_forbidden))
    bottom_allowed = bottom_h * (width - len(bottom_forbidden))
    if small_is_top:
        counts = _choose_counts(len(top_req), top_allowed, len(bottom_req), bottom_allowed)
    else:
        counts = _choose_counts(len(bottom_req), bottom_allowed, len(top_req), top_allowed)
    if counts is None:
        return None

    if small_is_top:
        top_count, bottom_count = counts
    else:
        bottom_count, top_count = counts

    top_mask = _fill_side(top_h, width, top_req, top_count, top_forbidden)
    bottom_mask = _fill_side(bottom_h, width, bottom_req, bottom_count, bottom_forbidden)

    if any(not top_mask[r].any() for r in range(top_h)) or any(not bottom_mask[r].any() for r in range(bottom_h)):
        return None
    if any(not (top_mask[:, c].any() or bottom_mask[:, c].any()) for c in range(width)):
        return None

    top_patch = _colorize(top_mask)
    bottom_patch = _colorize(bottom_mask)
    small_patch = top_patch if small_is_top else bottom_patch
    if np.array_equal(small_patch, np.rot90(small_patch, 2)):
        return None

    local = np.zeros((top_h + 1 + bottom_h, width), dtype=int)
    local[:top_h, :] = top_patch
    local[top_h + 1 :, :] = bottom_patch
    return local


def _place_local(local: np.ndarray) -> np.ndarray:
    h, w = local.shape
    outer_h = h + random.randint(2, 6)
    outer_w = w + random.randint(2, 6)
    grid = np.zeros((outer_h, outer_w), dtype=int)
    r0 = random.randint(1, outer_h - h - 1)
    c0 = random.randint(1, outer_w - w - 1)
    grid[r0 : r0 + h, c0 : c0 + w] = local
    return grid


def _build_candidate(branch: int) -> np.ndarray | None:
    if branch == 0:
        local = _build_local_row_case(small_on_top=True)
        return None if local is None else _place_local(local)
    if branch == 1:
        local = _build_local_row_case(small_on_top=False)
        return None if local is None else _place_local(local)
    if branch == 2:
        local = _build_local_row_case(small_on_top=True)
        return None if local is None else _place_local(local.T)
    local = _build_local_row_case(small_on_top=False)
    return None if local is None else _place_local(local.T)


def _validate_case(inp: np.ndarray) -> bool:
    try:
        info = _analyze(inp)
    except AssertionError:
        return False

    if info["first_count"] == info["second_count"]:
        return False

    out = main(inp)
    if np.array_equal(out, inp):
        return False

    green_only = _paint_output(inp, rotate_first=False, rotate_second=False)
    if np.array_equal(out, green_only):
        return False

    if info["smaller"] == "first":
        wrong_side = _paint_output(inp, rotate_first=False, rotate_second=True)
        tight_shortcut = _paint_output(inp, rotate_first=False, rotate_second=False)
        box = _bbox(inp != BACKGROUND)
        assert box is not None
        r0, c0, r1, c1 = box
        crop = inp[r0 : r1 + 1, c0 : c1 + 1].copy()
        if info["axis"] == "row":
            idx = info["index"]
            crop[:idx, :] = _rotate_tight_bbox(crop[:idx, :])
            tight_shortcut[r0 : r1 + 1, c0 : c1 + 1] = crop
            tight_shortcut[r0 + idx, :] = SEPARATOR_COLOR
        else:
            idx = info["index"]
            crop[:, :idx] = _rotate_tight_bbox(crop[:, :idx])
            tight_shortcut[r0 : r1 + 1, c0 : c1 + 1] = crop
            tight_shortcut[:, c0 + idx] = SEPARATOR_COLOR
    else:
        wrong_side = _paint_output(inp, rotate_first=True, rotate_second=False)
        tight_shortcut = _paint_output(inp, rotate_first=False, rotate_second=False)
        box = _bbox(inp != BACKGROUND)
        assert box is not None
        r0, c0, r1, c1 = box
        crop = inp[r0 : r1 + 1, c0 : c1 + 1].copy()
        if info["axis"] == "row":
            idx = info["index"]
            crop[idx + 1 :, :] = _rotate_tight_bbox(crop[idx + 1 :, :])
            tight_shortcut[r0 : r1 + 1, c0 : c1 + 1] = crop
            tight_shortcut[r0 + idx, :] = SEPARATOR_COLOR
        else:
            idx = info["index"]
            crop[:, idx + 1 :] = _rotate_tight_bbox(crop[:, idx + 1 :])
            tight_shortcut[r0 : r1 + 1, c0 : c1 + 1] = crop
            tight_shortcut[:, c0 + idx] = SEPARATOR_COLOR

    if np.array_equal(out, wrong_side):
        return False
    if np.array_equal(out, tight_shortcut):
        return False

    axis = info["axis"]
    box = info["box"]
    r0, c0, r1, c1 = box
    if axis == "row":
        sep_r = r0 + info["index"]
        if not np.all(out[sep_r, :] == SEPARATOR_COLOR):
            return False
        if np.any(out[:, c0 : c1 + 1] == SEPARATOR_COLOR) and not np.all(out[sep_r, c0 : c1 + 1] == SEPARATOR_COLOR):
            return False
    else:
        sep_c = c0 + info["index"]
        if not np.all(out[:, sep_c] == SEPARATOR_COLOR):
            return False
        if np.any(out[r0 : r1 + 1, :] == SEPARATOR_COLOR) and not np.all(out[r0 : r1 + 1, sep_c] == SEPARATOR_COLOR):
            return False

    return True


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    branch = _GEN_CALL_COUNT % 4
    _GEN_CALL_COUNT += 1

    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _build_candidate(branch)
        if inp is not None and _validate_case(inp):
            return inp
    raise RuntimeError("failed to generate a valid separator-rotation case")
