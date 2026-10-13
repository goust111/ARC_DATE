from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
PALETTE = (1, 3, 4, 6, 7, 8, 9)
HEIGHT_CHOICES = tuple(range(10, 29, 2))
WIDTH_RANGE = (8, 16)
MULTI_COUNT_RANGE = (2, 5)
NOISE_COUNT_RANGE = (2, 4)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as coordinate arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _clean_lower(lower: np.ndarray) -> np.ndarray:
    """Erase singleton 4-connected foreground components from the lower half."""
    cleaned = lower.copy()
    for coords in _find_components(lower != BACKGROUND):
        if len(coords) == 1:
            r, c = map(int, coords[0])
            cleaned[r, c] = BACKGROUND
    return cleaned


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase singleton noise in the lower half, then mirror the cleaned pattern into the blank upper half.

    Input structure:
    - The grid has even height and a black background.
    - The upper half is completely black.
    - The lower half contains several 4-connected foreground objects plus isolated
      single foreground pixels acting as noise.

    Rule:
    - Split the grid into its empty upper half and populated lower half.
    - In the lower half, find all 4-connected foreground components and erase every
      component of size 1.
    - Reverse the cleaned lower half by row order and copy it into the upper half.
    - Keep the cleaned lower half in place.

    Key insight:
    - The upper half is not a mirror of the raw input: singleton foreground pixels in
      the lower half are traps and must be deleted before the vertical completion step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert height % 2 == 0, "expected an even-height grid"
    assert max(height, width) <= 30, "ARC grids must fit within 30x30"

    mid = height // 2
    assert np.all(g[:mid] == BACKGROUND), "expected the upper half to be empty"

    lower = g[mid:]
    cleaned = _clean_lower(lower)

    out = g.copy()
    out[:mid] = cleaned[::-1]
    out[mid:] = cleaned
    return out


def _shape_patch(kind: str, color: int, h: int, w: int) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    if kind == "rect":
        patch[:, :] = color
    elif kind == "lshape":
        patch[:, 0] = color
        patch[-1, :] = color
        if h >= 3 and w >= 3 and random.random() < 0.5:
            patch[0, : max(2, w - 1)] = color
    elif kind == "tee":
        center = random.randrange(w)
        patch[0, :] = color
        patch[:, center] = color
        if h >= 4:
            patch[-1, max(0, center - 1) : min(w, center + 2)] = color
    elif kind == "step":
        for rr in range(h):
            span = min(w, 2 + rr)
            patch[rr, :span] = color
        if w >= 4 and random.random() < 0.5:
            patch[:, -1] = color
    elif kind == "zigzag":
        col = random.randrange(w)
        for rr in range(h):
            patch[rr, col] = color
            if rr + 1 < h:
                nxt = min(w - 1, col + 1) if rr % 2 == 0 else max(0, col - 1)
                patch[rr + 1, nxt] = color
                lo, hi = sorted((col, nxt))
                patch[rr + 1, lo : hi + 1] = color
                col = nxt
        patch[-1, :] = np.where(patch[-1, :] == 0, 0, color)
    else:
        raise ValueError(f"unknown shape kind: {kind}")

    if random.random() < 0.5:
        patch = np.rot90(patch, k=random.randint(1, 3))
    return patch


def _random_patch(color: int) -> np.ndarray:
    kind = random.choice(("rect", "lshape", "tee", "step", "zigzag"))
    if kind == "rect":
        h = random.randint(2, 4)
        w = random.randint(2, 5)
    elif kind == "lshape":
        h = random.randint(3, 5)
        w = random.randint(3, 5)
    elif kind == "tee":
        h = random.randint(3, 5)
        w = random.randint(3, 6)
    elif kind == "step":
        h = random.randint(3, 5)
        w = random.randint(3, 5)
    else:
        h = random.randint(3, 5)
        w = random.randint(3, 5)

    patch = _shape_patch(kind, color, h, w)
    coords = np.argwhere(patch != BACKGROUND)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    patch = patch[r0 : r1 + 1, c0 : c1 + 1]
    assert np.count_nonzero(patch) >= 2
    return patch


def _can_place(lower: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    h, w = patch.shape
    hh, ww = lower.shape
    if top < 1 or left < 1 or top + h > hh or left + w > ww:
        return False
    if top + h >= hh:
        return False
    if left + w >= ww:
        return False

    for rr in range(h):
        for cc in range(w):
            if patch[rr, cc] == BACKGROUND:
                continue
            gr = top + rr
            gc = left + cc
            if lower[gr, gc] != BACKGROUND:
                return False
            for nr, nc in _neighbors4(gr, gc):
                if 0 <= nr < hh and 0 <= nc < ww and lower[nr, nc] != BACKGROUND:
                    return False
    return True


def _place_patch(lower: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    region = lower[top : top + h, left : left + w]
    region[patch != BACKGROUND] = patch[patch != BACKGROUND]


def _free_noise_positions(lower: np.ndarray) -> list[tuple[int, int]]:
    hh, ww = lower.shape
    candidates: list[tuple[int, int]] = []
    for r in range(hh):
        for c in range(ww):
            if lower[r, c] != BACKGROUND:
                continue
            ok = True
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < hh and 0 <= nc < ww and lower[nr, nc] != BACKGROUND:
                    ok = False
                    break
            if ok:
                candidates.append((r, c))
    return candidates


def _component_stats(lower: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    multi: list[np.ndarray] = []
    single: list[np.ndarray] = []
    for coords in _find_components(lower != BACKGROUND):
        (single if len(coords) == 1 else multi).append(coords)
    return multi, single


def _is_valid_example(inp: np.ndarray) -> bool:
    height = inp.shape[0]
    mid = height // 2
    lower = inp[mid:]
    cleaned = _clean_lower(lower)
    out = main(inp)

    if np.array_equal(lower, cleaned):
        return False
    if np.array_equal(cleaned, cleaned[::-1]):
        return False
    if not np.array_equal(out[:mid], cleaned[::-1]):
        return False
    if not np.array_equal(out[mid:], cleaned):
        return False

    multi, single = _component_stats(lower)
    if not (MULTI_COUNT_RANGE[0] <= len(multi) <= MULTI_COUNT_RANGE[1]):
        return False
    if not (NOISE_COUNT_RANGE[0] <= len(single) <= NOISE_COUNT_RANGE[1]):
        return False
    if any(lower[r, c] == BACKGROUND for comp in multi for r, c in comp):
        return False

    naive = inp.copy()
    naive[:mid] = lower[::-1]
    if np.array_equal(naive, out):
        return False

    unique_colors = {int(lower[r, c]) for comp in multi for r, c in comp}
    if len(unique_colors) < 2:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    height = random.choice(HEIGHT_CHOICES)
    width = random.randint(*WIDTH_RANGE)
    mid = height // 2

    lower = np.zeros((mid, width), dtype=int)
    multi_count = random.randint(*MULTI_COUNT_RANGE)
    noise_count = random.randint(*NOISE_COUNT_RANGE)

    multi_colors = random.sample(PALETTE, k=random.randint(2, min(4, len(PALETTE))))
    patch_specs: list[tuple[np.ndarray, int]] = []
    for _ in range(multi_count):
        color = random.choice(multi_colors)
        patch_specs.append((_random_patch(color), color))

    random.shuffle(patch_specs)
    for patch, _color in patch_specs:
        positions = [
            (top, left)
            for top in range(1, mid - patch.shape[0])
            for left in range(1, width - patch.shape[1])
            if _can_place(lower, patch, top, left)
        ]
        if not positions:
            return None
        top, left = random.choice(positions)
        _place_patch(lower, patch, top, left)

    multi, _ = _component_stats(lower)
    if len(multi) != multi_count:
        return None

    free = _free_noise_positions(lower)
    if len(free) < noise_count:
        return None

    same_color_noise = random.random() < 0.6
    noise_colors = []
    for idx in range(noise_count):
        if same_color_noise and idx == 0:
            noise_colors.append(random.choice(sorted({int(lower[r, c]) for comp in multi for r, c in comp})))
        else:
            noise_colors.append(random.choice(PALETTE))

    random.shuffle(free)
    for (r, c), color in zip(free[:noise_count], noise_colors):
        lower[r, c] = color

    grid = np.zeros((height, width), dtype=int)
    grid[mid:] = lower

    if not _is_valid_example(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to construct a valid example")
