from __future__ import annotations

from importlib.machinery import SourceFileLoader
from typing import Callable, Literal

import numpy as np
import random


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_key_combo_3_013_3",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC_UTILS.Color

OP_UNION = "union"
OP_INTERSECTION = "intersection"
OP_SYMMETRIC_DIFFERENCE = "symmetric_difference"
Operator = Literal["union", "intersection", "symmetric_difference"]

OP_WEIGHTS = {
    OP_UNION: 0.38,
    OP_INTERSECTION: 0.31,
    OP_SYMMETRIC_DIFFERENCE: 0.31,
}

PANEL_H_MIN, PANEL_H_MAX = 6, 14
PANEL_W_MIN, PANEL_W_MAX = 6, 14
MAX_GENERATION_ATTEMPTS = 400
MASK_DENSITY_RANGE = (0.18, 0.62)
ASYMMETRIC_NOISE_DENSITY_RANGE = (0.08, 0.28)
SYMMETRIC_BASE_DENSITY = 0.35


def _is_vertically_symmetric(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, np.fliplr(mask)))


def _symmetric_candidates(top_mask: np.ndarray, bottom_mask: np.ndarray) -> list[Operator]:
    candidates: list[Operator] = []
    if _is_vertically_symmetric(top_mask | bottom_mask):
        candidates.append(OP_UNION)
    if _is_vertically_symmetric(top_mask & bottom_mask):
        candidates.append(OP_INTERSECTION)
    if _is_vertically_symmetric(top_mask ^ bottom_mask):
        candidates.append(OP_SYMMETRIC_DIFFERENCE)
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compute the single symmetric combine-mask operator between top/bottom panels and paint the result yellow.

    Input structure:
    - The input is a 2D grid whose height is even and split into equal top and bottom panels.
    - The top panel contains only black and green pixels as a foreground mask.
    - The bottom panel contains only black and gray pixels as a foreground mask.
    - Exactly one of union/intersection/symmetric-difference of the two masks is left-right symmetric.

    Rule:
    - Extract green mask from top panel and gray mask from bottom panel.
    - Compute element-wise union, intersection, and symmetric-difference masks on aligned pixels.
    - Identify the unique operator whose result is vertically mirror symmetric.
    - Return one panel-sized black grid with that symmetric mask painted yellow.

    Key insight:
    - The operator itself is ambiguous in appearance, but the symmetry constraint singles out one operator for each sample.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    height, width = g.shape
    assert height % 2 == 0, "expected an even height"
    h = height // 2

    top = g[:h]
    bottom = g[h:]

    top_colors = set(map(int, np.unique(top)))
    bottom_colors = set(map(int, np.unique(bottom)))
    assert top_colors <= {int(Color.BLACK), int(Color.GREEN)}, f"top panel has unexpected colors: {top_colors}"
    assert bottom_colors <= {int(Color.BLACK), int(Color.GRAY)}, f"bottom panel has unexpected colors: {bottom_colors}"

    top_mask = top == int(Color.GREEN)
    bottom_mask = bottom == int(Color.GRAY)

    candidates = _symmetric_candidates(top_mask, bottom_mask)
    assert len(candidates) == 1, "expected exactly one symmetric combine-mask operator"

    if OP_UNION in candidates:
        combined = top_mask | bottom_mask
    elif OP_INTERSECTION in candidates:
        combined = top_mask & bottom_mask
    else:
        combined = top_mask ^ bottom_mask

    out = np.full((h, width), int(Color.BLACK), dtype=int)
    out[combined] = int(Color.YELLOW)
    return out


def _rand_bool_mask(shape: tuple[int, int], density: float) -> np.ndarray:
    h, w = shape
    return np.random.random((h, w)) < float(density)


def _make_symmetric_mask(h: int, w: int, density: float) -> np.ndarray:
    density = float(density)
    for _ in range(64):
        if w == 1:
            left = np.array([[_rand_bool_mask((h, 1), density)]], dtype=object)
        half = np.random.random((h, w // 2)) < density
        right = np.fliplr(half)
        if w % 2 == 1:
            mid = np.random.random((h, 1)) < density
            mask = np.concatenate([half, mid, right], axis=1)
        else:
            mask = np.concatenate([half, right], axis=1)

        area = int(mask.sum())
        if 0 < area < h * w:
            return mask
    # Fallback: small symmetric bar plus side mirror copy (deterministic enough for generator progress).
    r0 = random.randint(0, max(0, h - 2))
    r1 = min(h - 1, r0 + random.randint(1, max(1, h // 2)))
    left_w = max(1, w // 4)
    c0 = random.randint(0, max(0, w // 2 - left_w))
    c1 = c0 + left_w - 1
    mask = np.zeros((h, w), dtype=bool)
    if w % 2 == 1:
        center = w // 2
        c_pairs = list(range(-left_w, left_w + 1))
        for c in c_pairs:
            cc_left = center + c
            cc_right = center - c
            if 0 <= cc_left < w:
                mask[r0 : r1 + 1, cc_left] = True
            if 0 <= cc_right < w:
                mask[r0 : r1 + 1, cc_right] = True
    else:
        cols = np.arange(c0, c1 + 1)
        mirror = w - 1 - cols
        mirrored_cols = np.concatenate([cols, mirror[mirror > mirror.min()]])
        for cc in np.unique(mirrored_cols):
            if 0 <= cc < w:
                mask[r0 : r1 + 1, cc] = True
    return mask


def _random_subset_of(mask: np.ndarray, density: float) -> np.ndarray | None:
    idx = np.flatnonzero(mask.ravel())
    if len(idx) == 0:
        return None

    for _ in range(128):
        keep = np.random.random(len(idx)) < float(density)
        if not keep.any():
            continue
        if keep.all():
            continue
        sub = np.zeros_like(mask, dtype=bool)
        sub.flat[idx[keep]] = True
        if not sub.any():
            continue
        if _is_vertically_symmetric(sub):
            continue
        if _is_vertically_symmetric(mask & (~sub)):
            continue
        return sub
    return None


def _random_asymmetric_mask_in_space(space: np.ndarray, density: float) -> np.ndarray | None:
    idx = np.flatnonzero(space.ravel())
    if len(idx) == 0:
        return None

    for _ in range(256):
        keep = np.random.random(len(idx)) < float(density)
        if not keep.any():
            continue
        mask = np.zeros_like(space, dtype=bool)
        mask.flat[idx[keep]] = True
        if mask.sum() == 0:
            continue
        if _is_vertically_symmetric(mask):
            continue
        return mask
    return None


def _build_union_case(h: int, w: int) -> tuple[np.ndarray, np.ndarray] | None:
    base = _make_symmetric_mask(h, w, SYMMETRIC_BASE_DENSITY)
    for _ in range(100):
        top = _random_subset_of(base, random.uniform(*MASK_DENSITY_RANGE))
        if top is None:
            continue
        bottom = base
        if _symmetric_candidates(top, bottom) == [OP_UNION]:
            return top, bottom
    return None


def _build_intersection_case(h: int, w: int) -> tuple[np.ndarray, np.ndarray] | None:
    base = _make_symmetric_mask(h, w, SYMMETRIC_BASE_DENSITY)
    space = ~base
    for _ in range(100):
        noise = _random_asymmetric_mask_in_space(space, random.uniform(*ASYMMETRIC_NOISE_DENSITY_RANGE))
        if noise is None:
            continue
        top = base
        bottom = np.logical_or(base, noise)
        if _symmetric_candidates(top, bottom) == [OP_INTERSECTION]:
            return top, bottom
    return None


def _build_xor_case(h: int, w: int) -> tuple[np.ndarray, np.ndarray] | None:
    base = _make_symmetric_mask(h, w, SYMMETRIC_BASE_DENSITY)
    space = ~base
    for _ in range(100):
        top = _random_asymmetric_mask_in_space(space, random.uniform(*ASYMMETRIC_NOISE_DENSITY_RANGE))
        if top is None:
            continue
        bottom = np.logical_xor(base, top)
        if _symmetric_candidates(top, bottom) == [OP_SYMMETRIC_DIFFERENCE]:
            return top, bottom
    return None


def _build_case(op: Operator, h: int, w: int) -> tuple[np.ndarray, np.ndarray] | None:
    builders: dict[Operator, Callable[[int, int], tuple[np.ndarray, np.ndarray] | None]] = {
        OP_UNION: _build_union_case,
        OP_INTERSECTION: _build_intersection_case,
        OP_SYMMETRIC_DIFFERENCE: _build_xor_case,
    }
    return builders[op](h, w)


def _colorize_panels(top_mask: np.ndarray, bottom_mask: np.ndarray) -> np.ndarray:
    h, w = top_mask.shape
    top = np.full((h, w), int(Color.BLACK), dtype=int)
    bottom = np.full((h, w), int(Color.BLACK), dtype=int)
    top[top_mask] = int(Color.GREEN)
    bottom[bottom_mask] = int(Color.GRAY)
    return np.vstack([top, bottom])


def generate_input() -> np.ndarray:
    """
    Randomly generate valid two-panel inputs where exactly one combine-mask operator is vertically symmetric.

    Input structure:
    - Grid height is even and split into equal top and bottom panels.
    - Top uses a green mask on black; bottom uses a gray mask on black.
    - The three operator results are arranged so that one and only one of union/intersection/symmetric-difference is vertically symmetric.

    Rule:
    - Build cases for one random target operator.
    - Keep retrying geometry/placement parameters until that operator is uniquely symmetric.
    - Return the stacked colored panel grid.
    """
    branches = [OP_UNION, OP_INTERSECTION, OP_SYMMETRIC_DIFFERENCE]
    weights = [OP_WEIGHTS[b] for b in branches]
    for _ in range(MAX_GENERATION_ATTEMPTS):
        h = random.randint(PANEL_H_MIN, PANEL_H_MAX)
        w = random.randint(PANEL_W_MIN, PANEL_W_MAX)
        op: Operator = random.choices(branches, weights=weights, k=1)[0]

        pair = _build_case(op, h, w)
        if pair is None:
            continue
        top_mask, bottom_mask = pair
        selected = _symmetric_candidates(top_mask, bottom_mask)
        if selected != [op]:
            continue
        return _colorize_panels(top_mask, bottom_mask)

    raise RuntimeError("failed to generate a valid 3-key-combo input")
