import itertools
import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from arc_utils import (
    Color,
    bbox_h,
    bbox_w,
    components_mask,
    crop_bbox,
    find_components,
    get_bg,
    paste,
)


BLUE = Color.BLUE
RED = Color.RED
GREEN = Color.GREEN
BG = Color.BLACK

_DIM_OPTIONS = (
    (2, 3),
    (3, 2),
    (3, 3),
    (2, 4),
    (4, 2),
    (3, 4),
    (4, 3),
)
_AREA_RANGE = range(4, 9)
_GRID_MIN = 14
_GRID_MAX = 24
_RETRY_LIMIT = 400
_MASK_CACHE: dict[str, list[np.ndarray]] = {}


def _valid_template_mask(mask: np.ndarray, flip_axis: str) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    if not (4 <= int(mask.sum()) <= 8):
        return False
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    if (int(r0), int(c0), int(r1), int(c1)) != (0, 0, mask.shape[0] - 1, mask.shape[1] - 1):
        return False
    if len(components_mask(mask, connectivity=4)) != 1:
        return False
    if flip_axis == "lr" and np.array_equal(mask, np.fliplr(mask)):
        return False
    if flip_axis == "ud" and np.array_equal(mask, np.flipud(mask)):
        return False
    return True


def _template_masks(flip_axis: str) -> list[np.ndarray]:
    if flip_axis in _MASK_CACHE:
        return _MASK_CACHE[flip_axis]

    masks: list[np.ndarray] = []
    for h, w in _DIM_OPTIONS:
        for area in _AREA_RANGE:
            if area > h * w:
                continue
            for cells in itertools.combinations(range(h * w), area):
                mask = np.zeros((h, w), dtype=bool)
                for idx in cells:
                    mask[idx // w, idx % w] = True
                if _valid_template_mask(mask, flip_axis):
                    masks.append(mask.copy())

    assert masks, f"no valid masks for {flip_axis}"
    _MASK_CACHE[flip_axis] = masks
    return masks


def _blue_patch(mask: np.ndarray) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = BLUE
    return patch


def _green_mirror(patch: np.ndarray, axis: str) -> np.ndarray:
    mirrored = np.fliplr(patch) if axis == "horizontal" else np.flipud(patch)
    out = mirrored.copy()
    out[out == BLUE] = GREEN
    return out


def _infer_missing_value(values: list[int]) -> tuple[int, int]:
    diffs = [int(b - a) for a, b in zip(values, values[1:])]
    assert diffs, "need at least two observed positions"
    step = min(diffs)
    assert step > 0
    gap_inds = [i for i, d in enumerate(diffs) if d == 2 * step]
    assert len(gap_inds) == 1, "expected exactly one doubled gap"
    assert all(d in {step, 2 * step} for d in diffs), "sequence must be arithmetic with one hole"
    gap_i = gap_inds[0]
    return int(values[gap_i] + step), int(step)


def _blue_components(grid: np.ndarray) -> list[dict]:
    comps = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[BLUE],
        return_format="dict",
    )
    assert len(comps) in {3, 4}, "expected 3-4 blue sprites"
    return comps


def _anchor_component(grid: np.ndarray) -> dict:
    comps = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[RED],
        return_format="dict",
    )
    assert len(comps) == 1, "expected one red anchor"
    comp = comps[0]
    h = bbox_h(comp["bbox"])
    w = bbox_w(comp["bbox"])
    assert (h, w) in {(2, 1), (1, 2)}, "anchor must be a domino"
    assert int(comp["area"]) == 2
    return comp


def _template_patch(grid: np.ndarray, comps: list[dict]) -> np.ndarray:
    patch = crop_bbox(grid, comps[0]["bbox"])
    assert set(map(int, np.unique(patch))) <= {BG, BLUE}
    for comp in comps[1:]:
        other = crop_bbox(grid, comp["bbox"])
        assert other.shape == patch.shape
        assert np.array_equal(other, patch), "all blue sprites must share one template"
    return patch


def _sequence_info(grid: np.ndarray) -> dict:
    comps = _blue_components(grid)
    patch = _template_patch(grid, comps)
    h, w = patch.shape

    r0s = [int(comp["bbox"][0]) for comp in comps]
    c0s = [int(comp["bbox"][1]) for comp in comps]
    horizontal = len(set(r0s)) == 1 and len(set(c0s)) == len(comps)
    vertical = len(set(c0s)) == 1 and len(set(r0s)) == len(comps)
    assert horizontal ^ vertical, "blue sprites must form one horizontal or vertical sequence"

    if horizontal:
        fixed_r = int(r0s[0])
        ordered = sorted(c0s)
        missing_c0, step = _infer_missing_value(ordered)
        return {
            "axis": "horizontal",
            "patch": patch,
            "slot_r0": fixed_r,
            "slot_c0": missing_c0,
            "step": step,
            "h": h,
            "w": w,
        }

    fixed_c = int(c0s[0])
    ordered = sorted(r0s)
    missing_r0, step = _infer_missing_value(ordered)
    return {
        "axis": "vertical",
        "patch": patch,
        "slot_r0": missing_r0,
        "slot_c0": fixed_c,
        "step": step,
        "h": h,
        "w": w,
    }


def _destination_from_anchor(seq: dict, anchor: dict) -> tuple[int, int]:
    slot_r0 = int(seq["slot_r0"])
    slot_c0 = int(seq["slot_c0"])
    h = int(seq["h"])
    w = int(seq["w"])
    ar0, ac0, ar1, ac1 = map(int, anchor["bbox"])

    if seq["axis"] == "horizontal":
        assert (bbox_h(anchor["bbox"]), bbox_w(anchor["bbox"])) == (2, 1)
        axis_c = ac0
        assert slot_r0 <= ar0 <= ar1 <= slot_r0 + h - 1
        if axis_c == slot_c0:
            return slot_r0, axis_c - (w - 1)
        assert axis_c == slot_c0 + w - 1
        return slot_r0, axis_c

    assert (bbox_h(anchor["bbox"]), bbox_w(anchor["bbox"])) == (1, 2)
    axis_r = ar0
    assert slot_c0 <= ac0 <= ac1 <= slot_c0 + w - 1
    if axis_r == slot_r0:
        return axis_r - (h - 1), slot_c0
    assert axis_r == slot_r0 + h - 1
    return axis_r, slot_c0


def _expected_output_from_grid(grid: np.ndarray) -> np.ndarray:
    seq = _sequence_info(grid)
    anchor = _anchor_component(grid)
    green = _green_mirror(seq["patch"], seq["axis"])
    dst_r0, dst_c0 = _destination_from_anchor(seq, anchor)
    h, w = green.shape

    out = np.asarray(grid, dtype=int).copy()
    out[out == RED] = BG

    assert 0 <= dst_r0 <= out.shape[0] - h
    assert 0 <= dst_c0 <= out.shape[1] - w
    target = out[dst_r0 : dst_r0 + h, dst_c0 : dst_c0 + w]
    assert bool(np.all((target == BG) | (green == BG))), "mirrored sprite must land on empty background"
    paste(out, green, dst_r0, dst_c0, transparent=BG)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the blue arithmetic sequence to locate the missing slot, then place a green mirror copy across the red edge anchor.

    Input structure:
    - The background is black.
    - There are 3-4 identical blue 4-connected sprites with the same cropped bbox patch.
    - Their bbox top-left coordinates form one horizontal or vertical arithmetic progression with exactly one interior term missing.
    - A single red domino sits on one edge of that missing slot; its long axis is perpendicular to the progression axis.

    Rule:
    - Infer the absent slot from the doubled step in the blue bbox sequence.
    - Treat the red domino's row or column as a mirror line attached to one edge of the absent slot.
    - Take the blue sprite template, reflect it across that line onto the opposite side of the marked slot edge, recolor the reflected copy green, and erase the red anchor.
    - Keep all original blue sprites unchanged.

    Key insight:
    - The blue arithmetic progression says which slot is missing, while the red domino says which edge of that slot is the reflection line; together they determine where the green mirror copy must appear.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert int(get_bg(g)) == BG
    return _expected_output_from_grid(g)


def _slot_copy_output(inp: np.ndarray, patch: np.ndarray, axis: str, slot_r0: int, slot_c0: int, *, mirror: bool) -> np.ndarray:
    out = inp.copy()
    out[out == RED] = BG
    placed = patch.copy()
    if mirror:
        placed = np.fliplr(placed) if axis == "horizontal" else np.flipud(placed)
    placed[placed == BLUE] = GREEN
    paste(out, placed, slot_r0, slot_c0, transparent=BG)
    return out


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        seq = _sequence_info(inp)
        anchor = _anchor_component(inp)
        actual = main(inp)
    except Exception:
        return False

    if not np.array_equal(actual, expected):
        return False

    slot_r0 = int(seq["slot_r0"])
    slot_c0 = int(seq["slot_c0"])
    patch = seq["patch"]
    axis = str(seq["axis"])

    if np.array_equal(_slot_copy_output(inp, patch, axis, slot_r0, slot_c0, mirror=False), expected):
        return False
    if np.array_equal(_slot_copy_output(inp, patch, axis, slot_r0, slot_c0, mirror=True), expected):
        return False

    alt_green = _green_mirror(patch, axis)
    dst_r0, dst_c0 = _destination_from_anchor(seq, anchor)
    if axis == "horizontal":
        alt_r0, alt_c0 = slot_r0, slot_c0
    else:
        alt_r0, alt_c0 = slot_r0, slot_c0
    if (alt_r0, alt_c0) == (dst_r0, dst_c0):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    axis = random.choice(["horizontal", "vertical"])
    flip_axis = "lr" if axis == "horizontal" else "ud"
    mask = random.choice(_template_masks(flip_axis))
    patch = _blue_patch(mask)
    green = _green_mirror(patch, axis)
    h, w = patch.shape

    total = random.choice([4, 5])
    missing_idx = random.randint(1, total - 2)
    if axis == "horizontal":
        step = random.randint(2 * w, 2 * w + 2)
        span_h = h
        span_w = w + (total - 1) * step
    else:
        step = random.randint(2 * h, 2 * h + 2)
        span_h = h + (total - 1) * step
        span_w = w

    if span_h + 2 > _GRID_MAX or span_w + 2 > _GRID_MAX:
        return None

    H = random.randint(max(_GRID_MIN, span_h + 2), _GRID_MAX)
    W = random.randint(max(_GRID_MIN, span_w + 2), _GRID_MAX)
    if axis == "horizontal":
        base_r0 = random.randint(1, H - h - 1)
        base_c0 = random.randint(1, W - span_w - 1)
    else:
        base_r0 = random.randint(1, H - span_h - 1)
        base_c0 = random.randint(1, W - w - 1)

    inp = np.zeros((H, W), dtype=int)
    for idx in range(total):
        r0 = base_r0 + (idx * step if axis == "vertical" else 0)
        c0 = base_c0 + (idx * step if axis == "horizontal" else 0)
        if idx != missing_idx:
            paste(inp, patch, r0, c0, transparent=BG)

    slot_r0 = base_r0 + (missing_idx * step if axis == "vertical" else 0)
    slot_c0 = base_c0 + (missing_idx * step if axis == "horizontal" else 0)

    expected = inp.copy()
    side = random.choice(["left", "right"] if axis == "horizontal" else ["top", "bottom"])
    if axis == "horizontal":
        anchor_r0 = random.randint(slot_r0, slot_r0 + h - 2)
        anchor_c0 = slot_c0 if side == "left" else slot_c0 + w - 1
        paste(inp, np.full((2, 1), RED, dtype=int), anchor_r0, anchor_c0)
        expected[inp == RED] = BG
        dst_r0 = slot_r0
        dst_c0 = anchor_c0 - (w - 1) if side == "left" else anchor_c0
    else:
        anchor_r0 = slot_r0 if side == "top" else slot_r0 + h - 1
        anchor_c0 = random.randint(slot_c0, slot_c0 + w - 2)
        paste(inp, np.full((1, 2), RED, dtype=int), anchor_r0, anchor_c0)
        expected[inp == RED] = BG
        dst_r0 = anchor_r0 - (h - 1) if side == "top" else anchor_r0
        dst_c0 = slot_c0

    target = expected[dst_r0 : dst_r0 + h, dst_c0 : dst_c0 + w]
    if target.shape != green.shape:
        return None
    if not bool(np.all((target == BG) | (green == BG))):
        return None
    paste(expected, green, dst_r0, dst_c0, transparent=BG)

    if not _semantic_checks(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
