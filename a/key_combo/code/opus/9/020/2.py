from __future__ import annotations

import random
from collections import deque

import numpy as np


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(
    dst: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    *,
    transparent: int = 0,
    clip: bool = False,
) -> None:
    h, w = patch.shape
    rr0 = int(r0)
    cc0 = int(c0)
    rr1 = rr0 + h
    cc1 = cc0 + w

    if not clip:
        if rr0 < 0 or cc0 < 0 or rr1 > dst.shape[0] or cc1 > dst.shape[1]:
            raise ValueError("paste out of bounds")
        src = patch
        dst_view = dst[rr0:rr1, cc0:cc1]
    else:
        dr0 = max(0, rr0)
        dc0 = max(0, cc0)
        dr1 = min(dst.shape[0], rr1)
        dc1 = min(dst.shape[1], cc1)
        if dr0 >= dr1 or dc0 >= dc1:
            return
        sr0 = dr0 - rr0
        sc0 = dc0 - cc0
        sr1 = sr0 + (dr1 - dr0)
        sc1 = sc0 + (dc1 - dc0)
        src = patch[sr0:sr1, sc0:sc1]
        dst_view = dst[dr0:dr1, dc0:dc1]

    mask = src != int(transparent)
    dst_view[mask] = src[mask]


def _component_masks(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comp = np.zeros((h, w), dtype=bool)
            rows, cols = np.array(cells, dtype=int).T
            comp[rows, cols] = True
            comps.append(comp)
    return comps


def _count_enclosed_black_pixels(crop: np.ndarray) -> int:
    empty = crop == 0
    h, w = crop.shape
    outside = np.zeros((h, w), dtype=bool)
    q = deque()

    for r in range(h):
        for c in (0, w - 1):
            if empty[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if empty[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and empty[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    return int(np.count_nonzero(empty & ~outside))


def _extract_sprite(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int], np.ndarray]:
    fg_colors = sorted(int(v) for v in np.unique(grid) if int(v) != 0)
    assert len(fg_colors) == 1, "expected exactly one foreground color"
    color = fg_colors[0]
    mask = grid == color
    comps = _component_masks(mask)
    assert len(comps) == 1, "expected one 4-connected foreground sprite"
    bbox = _bbox(mask)
    assert bbox is not None
    return color, bbox, _crop_bbox(grid, bbox)


def _periodic_stamp(
    grid_shape: tuple[int, int],
    sprite: np.ndarray,
    top: int,
    left: int,
    base_color: int,
    hole_count: int,
) -> np.ndarray:
    out = np.zeros(grid_shape, dtype=int)
    step = sprite.shape[1] + 1
    k = 0
    while left + k * step < grid_shape[1]:
        color = base_color if k == 0 or k % (hole_count + 1) != hole_count else 6
        patch = np.where(sprite != 0, color, 0)
        _paste(out, patch, top, left + k * step, transparent=0, clip=True)
        k += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the single holed sprite repeatedly to the right, making every topology-timed copy magenta.

    Input structure:
    - The grid has black background and exactly one 4-connected monochrome foreground sprite.
    - The sprite sits near the left side and may contain enclosed black pixels inside its tight bbox.
    - No other foreground objects or guide marks are present.

    Rule:
    - Find the sprite's tight bbox and count the black pixels inside that bbox that are not 4-connected to the bbox border through black pixels.
    - Let that enclosed-black count be H, and reuse the sprite's exact bbox crop as the stamping patch.
    - Place copies at columns `left + k * (bbox_width + 1)` for `k = 0, 1, 2, ...`, clipping any part that extends past the right border.
    - Copy `k = 0` stays in its original color; for `k >= 1`, the copy is magenta (6) exactly when `k mod (H + 1) == H`, otherwise it keeps the original color.

    Key insight:
    - The recolor period is not fixed in space; it is derived from the sprite's enclosed black topology, so different hole counts produce different magenta rhythms.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    color, bbox, sprite = _extract_sprite(g)
    hole_count = _count_enclosed_black_pixels(sprite)
    assert hole_count >= 1, "sprite must enclose at least one black pixel"

    r0, c0, _r1, _c1 = bbox
    stamped = _periodic_stamp(g.shape, sprite, r0, c0, color, hole_count)
    return stamped


_HOLE_BLOCK: list[int] = []

_BASE_DIMS = {
    1: [(3, 3), (3, 4), (4, 3), (4, 4), (5, 3), (3, 5)],
    2: [(3, 4), (4, 3), (3, 5), (5, 3), (4, 4), (4, 5), (5, 4)],
    3: [(3, 5), (5, 3), (4, 4), (4, 5), (5, 4)],
}
_GEN_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]
_GEN_ATTEMPTS = 500


def _next_hole_count() -> int:
    global _HOLE_BLOCK
    if not _HOLE_BLOCK:
        _HOLE_BLOCK = random.sample([1, 2, 3], 3)
    return _HOLE_BLOCK.pop()


def _trim_patch(patch: np.ndarray) -> np.ndarray:
    bbox = _bbox(patch != 0)
    assert bbox is not None
    return _crop_bbox(patch, bbox)


def _patch_valid(patch: np.ndarray, color: int, hole_count: int) -> bool:
    if not np.any(patch == color):
        return False
    if np.any((patch != 0) & (patch != color)):
        return False
    trimmed = _trim_patch(patch)
    mask = trimmed == color
    if len(_component_masks(mask)) != 1:
        return False
    if _count_enclosed_black_pixels(trimmed) != hole_count:
        return False
    return max(trimmed.shape) <= 6


def _random_base_patch(color: int, hole_count: int) -> np.ndarray:
    h, w = random.choice(_BASE_DIMS[hole_count])
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color

    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    keep_black = set(random.sample(interior, hole_count))
    for cell in interior:
        if cell not in keep_black:
            patch[cell] = color
    return patch


def _mutate_patch(patch: np.ndarray, color: int, hole_count: int) -> np.ndarray:
    canvas = np.pad(patch, 1)
    canvas[canvas != 0] = color
    for _ in range(random.randint(1, 4)):
        if random.random() < 0.55:
            candidates = []
            for r in range(canvas.shape[0]):
                for c in range(canvas.shape[1]):
                    if canvas[r, c] != 0:
                        continue
                    if any(
                        0 <= nr < canvas.shape[0]
                        and 0 <= nc < canvas.shape[1]
                        and canvas[nr, nc] == color
                        for nr, nc in _neighbors4(r, c)
                    ):
                        trial = canvas.copy()
                        trial[r, c] = color
                        if _patch_valid(trial, color, hole_count):
                            candidates.append(trial)
            if candidates:
                canvas = random.choice(candidates)
        else:
            candidates = []
            for r in range(canvas.shape[0]):
                for c in range(canvas.shape[1]):
                    if canvas[r, c] != color:
                        continue
                    trial = canvas.copy()
                    trial[r, c] = 0
                    if _patch_valid(trial, color, hole_count):
                        candidates.append(trial)
            if candidates:
                canvas = random.choice(candidates)
    return _trim_patch(canvas)


def _build_sprite_patch(color: int, hole_count: int) -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        patch = _random_base_patch(color, hole_count)
        patch = _mutate_patch(patch, color, hole_count)
        if _patch_valid(patch, color, hole_count):
            return patch
    raise RuntimeError("failed to build a valid sprite patch")


def _render_expected(
    grid_shape: tuple[int, int],
    patch: np.ndarray,
    top: int,
    left: int,
    color: int,
    hole_count: int,
) -> np.ndarray:
    out = np.zeros(grid_shape, dtype=int)
    step = patch.shape[1] + 1
    for k in range(100):
        col = left + k * step
        if col >= grid_shape[1]:
            break
        copy_color = color if k == 0 or k % (hole_count + 1) != hole_count else 6
        colored_patch = np.where(patch != 0, copy_color, 0)
        _paste(out, colored_patch, top, col, transparent=0, clip=True)
    return out


def _sample_grid_width(sprite_w: int, left: int, hole_count: int) -> int:
    min_width = left + hole_count * (sprite_w + 1) + sprite_w
    min_width = max(min_width, 12)
    max_width = 30
    assert min_width <= max_width
    return random.randint(min_width, max_width)


def _sample_input() -> np.ndarray | None:
    hole_count = _next_hole_count()
    color = random.choice(_GEN_COLORS)
    patch = _build_sprite_patch(color, hole_count)
    ph, pw = patch.shape

    left = random.randint(0, 2)
    width = _sample_grid_width(pw, left, hole_count)
    height = random.randint(max(ph + 2, 10), 18)
    top = random.randint(1, height - ph - 1)

    inp = np.zeros((height, width), dtype=int)
    _paste(inp, patch, top, left, transparent=0, clip=False)

    expected = _render_expected(inp.shape, patch, top, left, color, hole_count)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    actual_holes = _count_enclosed_black_pixels(_extract_sprite(inp)[2])
    if actual_holes != hole_count:
        return None

    step = pw + 1
    magenta_left = left + hole_count * step
    if magenta_left + pw > width:
        return None
    if not np.any(actual[:, magenta_left : magenta_left + pw] == 6):
        return None
    if hole_count >= 2 and not np.any(actual[:, left + step : left + step + pw] == color):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _sample_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
