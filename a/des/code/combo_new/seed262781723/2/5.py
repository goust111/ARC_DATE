from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
CONTAINER = 5
LIQUID_COLORS = tuple(color for color in range(1, 10) if color != CONTAINER)
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q: deque[tuple[int, int]] = deque([(r, c)])
        comp: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            comp.append((cr, cc))
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _u_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[:, 0] = True
    mask[:, -1] = True
    mask[-1, :] = True
    return mask


def _liquid_height(inner: np.ndarray, bg: int, wall_color: int) -> int:
    colors = {
        int(color)
        for color in np.unique(inner)
        if int(color) not in {int(bg), int(wall_color)}
    }
    assert len(colors) == 1, "each container must contain exactly one liquid color"
    liquid = colors.pop()
    liquid_mask = inner == liquid

    rows = np.flatnonzero(np.any(liquid_mask, axis=1))
    cols = np.flatnonzero(np.any(liquid_mask, axis=0))
    assert len(rows) >= 1 and len(cols) == inner.shape[1], "liquid must span the full inner width"
    assert np.array_equal(rows, np.arange(rows[0], rows[-1] + 1)), "liquid rows must be contiguous"
    assert rows[-1] == inner.shape[0] - 1, "liquid must rest on the container floor"
    assert np.all(liquid_mask[rows[0] : rows[-1] + 1, :]), "liquid must form a solid rectangle"
    return int(len(rows))


def _find_containers(g: np.ndarray) -> list[dict]:
    bg = BG
    containers: list[dict] = []

    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(g == color):
            r0, c0, r1, c1 = _bbox(coords)
            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for r, c in coords:
                local[r - r0, c - c0] = True
            if local.shape[0] < 2 or local.shape[1] < 3:
                continue
            if not np.array_equal(local, _u_mask(local.shape)):
                continue

            inner = g[r0:r1, c0 + 1 : c1]
            assert inner.size > 0, "container must have an interior"
            height = _liquid_height(inner, bg=bg, wall_color=color)
            containers.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "height": height,
                    "patch": g[r0 : r1 + 1, c0 : c1 + 1].copy(),
                }
            )

    assert len(containers) >= 2, "expected multiple containers"
    containers.sort(key=lambda item: item["bbox"][1])
    patch_shapes = {tuple(item["patch"].shape) for item in containers}
    assert len(patch_shapes) == 1, "all containers must share one size"
    return containers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder U-shaped containers by ascending liquid height while keeping each container intact.

    Input structure:
    - Black background with several disconnected U-shaped containers arranged left to right.
    - Each container is made from one color of outline pixels: two vertical walls and one bottom floor.
    - Inside every container is one solid non-black, non-outline rectangle touching the floor, with a unique color and a possibly different height.

    Rule:
    - Detect every U-shaped container and measure the height of its inner colored rectangle.
    - Stably sort the containers from lowest liquid level to highest liquid level.
    - Rebuild the grid by placing the sorted containers back into the original left-to-right container slots.

    Key insight:
    - The colored rectangles are not transformed locally; their visible height is the global sorting key that determines a whole-container permutation.
    """
    g = np.asarray(grid, dtype=int)
    bg = BG
    containers = _find_containers(g)
    ordered = sorted(containers, key=lambda item: item["height"])

    out = g.copy()
    for item in containers:
        r0, c0, r1, c1 = item["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = bg

    for slot, item in zip(containers, ordered, strict=False):
        r0, c0, r1, c1 = slot["bbox"]
        patch = item["patch"]
        assert patch.shape == (r1 - r0 + 1, c1 - c0 + 1)
        out[r0 : r1 + 1, c0 : c1 + 1] = patch

    return out


def _build_container_patch(inner_h: int, inner_w: int, liquid_h: int, color: int) -> np.ndarray:
    h = inner_h + 1
    w = inner_w + 2
    patch = np.full((h, w), BG, dtype=int)
    patch[:, 0] = CONTAINER
    patch[:, -1] = CONTAINER
    patch[-1, :] = CONTAINER
    patch[h - 1 - liquid_h : h - 1, 1:-1] = int(color)
    return patch


def _sample_heights(count: int, max_height: int, allow_ties: bool) -> list[int] | None:
    if allow_ties:
        for _ in range(100):
            heights = [random.randint(1, max_height) for _ in range(count)]
            if 1 < len(set(heights)) < count:
                return heights
        return None
    if max_height < count:
        return None
    return random.sample(range(1, max_height + 1), count)


def _sorted_index_order(heights: list[int]) -> list[int]:
    return sorted(range(len(heights)), key=lambda idx: (heights[idx], idx))


def _build_expected(inp: np.ndarray, top: int, left: int, step: int, patches: list[np.ndarray], order: list[int]) -> np.ndarray:
    out = np.full_like(inp, BG)
    for slot_idx, patch_idx in enumerate(order):
        patch = patches[patch_idx]
        h, w = patch.shape
        c0 = left + slot_idx * step
        out[top : top + h, c0 : c0 + w] = patch
    return out


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(3, 6)
    min_gap = 1
    max_inner_w = min(5, (30 - (count - 1) * min_gap) // count - 2)
    if max_inner_w < 2:
        return None
    inner_w = random.randint(2, max_inner_w)
    container_w = inner_w + 2

    max_gap = min(3, (30 - count * container_w) // max(1, count - 1))
    if max_gap < 1:
        return None
    gap = random.randint(1, max_gap)

    inner_h = random.randint(max(3, min(count, 6)), 8)
    container_h = inner_h + 1
    if container_h > 30:
        return None

    remaining_w = 30 - (count * container_w + (count - 1) * gap)
    if remaining_w < 0:
        return None
    left = random.randint(0, min(4, remaining_w))
    right = random.randint(0, min(4, remaining_w - left))

    remaining_h = 30 - container_h
    top = random.randint(0, min(4, remaining_h))
    bottom = random.randint(0, min(4, remaining_h - top))

    rows = container_h + top + bottom
    cols = count * container_w + (count - 1) * gap + left + right
    if rows > 30 or cols > 30:
        return None

    allow_ties = random.random() < 0.5
    heights = _sample_heights(count, inner_h, allow_ties)
    if heights is None:
        return None
    random.shuffle(heights)

    order = _sorted_index_order(heights)
    if order == list(range(count)):
        return None

    colors = random.sample(LIQUID_COLORS, count)
    patches = [
        _build_container_patch(inner_h=inner_h, inner_w=inner_w, liquid_h=height, color=color)
        for height, color in zip(heights, colors, strict=False)
    ]

    inp = np.full((rows, cols), BG, dtype=int)
    step = container_w + gap
    for idx, patch in enumerate(patches):
        c0 = left + idx * step
        inp[top : top + container_h, c0 : c0 + container_w] = patch

    expected = _build_expected(inp, top=top, left=left, step=step, patches=patches, order=order)
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid container-sorting ARC input")
