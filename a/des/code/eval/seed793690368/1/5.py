from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
ARROW_COLOR = 9
PALETTE = tuple(color for color in range(1, 9))
DIRECTIONS = ("up", "right", "down", "left")
SIZE_CHOICES = (5, 7, 9, 11)
SHAFT_LENGTHS = (2, 3)
MAX_GENERATION_ATTEMPTS = 300


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(np.min(coords[:, 0])),
        int(np.min(coords[:, 1])),
        int(np.max(coords[:, 0])),
        int(np.max(coords[:, 1])),
    )


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((int(cr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = int(cr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _pattern_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = (grid != BG) & (grid != ARROW_COLOR)
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected one nested square pattern"
    r0, c0, r1, c1 = _bbox(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert patch.shape[0] == patch.shape[1], "pattern bbox must be square"
    assert patch.shape[0] % 2 == 1, "pattern size must be odd"
    assert np.all(patch != BG), "pattern bbox must be fully filled"
    assert np.all(patch != ARROW_COLOR), "arrow color must stay outside the pattern"
    return r0, c0, r1, c1


def _arrow_direction(grid: np.ndarray, pattern_bbox: tuple[int, int, int, int]) -> str:
    arrow_mask = grid == ARROW_COLOR
    arrow_coords = np.argwhere(arrow_mask)
    assert arrow_coords.size > 0, "expected one maroon arrow"
    assert len(_components(arrow_mask)) == 1, "expected one connected arrow"

    r0, c0, r1, c1 = pattern_bbox
    candidates = []
    if np.all(arrow_coords[:, 0] < r0):
        candidates.append("up")
    if np.all(arrow_coords[:, 1] > c1):
        candidates.append("right")
    if np.all(arrow_coords[:, 0] > r1):
        candidates.append("down")
    if np.all(arrow_coords[:, 1] < c0):
        candidates.append("left")
    assert len(candidates) == 1, "arrow must lie on exactly one side of the pattern"
    return candidates[0]


def _layers_from_bbox(pattern_bbox: tuple[int, int, int, int]) -> int:
    r0, _c0, r1, _c1 = pattern_bbox
    return (int(r1 - r0 + 1) + 1) // 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unfold the arrow-indicated side of each concentric square layer.

    The nested square stays fixed. The maroon arrow is removed. For each ring,
    copy the side pointed at by the arrow and lay those sides outward from the
    pattern, ordered outermost to innermost with one-cell background gaps.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    pattern_bbox = _pattern_bbox(g)
    direction = _arrow_direction(g, pattern_bbox)
    pr0, pc0, pr1, pc1 = pattern_bbox
    layers = _layers_from_bbox(pattern_bbox)

    out = g.copy()
    out[out == ARROW_COLOR] = BG

    for layer in range(layers):
        if direction == "right":
            rows = slice(pr0 + layer, pr1 - layer + 1)
            src_col = pc1 - layer
            dst_col = pc1 + 1 + 2 * layer
            assert dst_col < g.shape[1]
            out[rows, dst_col] = g[rows, src_col]
        elif direction == "left":
            rows = slice(pr0 + layer, pr1 - layer + 1)
            src_col = pc0 + layer
            dst_col = pc0 - 1 - 2 * layer
            assert dst_col >= 0
            out[rows, dst_col] = g[rows, src_col]
        elif direction == "down":
            cols = slice(pc0 + layer, pc1 - layer + 1)
            src_row = pr1 - layer
            dst_row = pr1 + 1 + 2 * layer
            assert dst_row < g.shape[0]
            out[dst_row, cols] = g[src_row, cols]
        else:
            cols = slice(pc0 + layer, pc1 - layer + 1)
            src_row = pr0 + layer
            dst_row = pr0 - 1 - 2 * layer
            assert dst_row >= 0
            out[dst_row, cols] = g[src_row, cols]

    return out


def _nested_patch(size: int, colors: list[int]) -> np.ndarray:
    patch = np.empty((size, size), dtype=int)
    for layer, color in enumerate(colors):
        patch[layer : size - layer, layer : size - layer] = int(color)
    return patch


def _arrow_offsets(direction: str, shaft_len: int) -> list[tuple[int, int]]:
    if direction == "right":
        return (
            [(0, dc) for dc in range(1, shaft_len + 1)]
            + [(-1, shaft_len), (0, shaft_len + 1), (1, shaft_len)]
        )
    if direction == "left":
        return (
            [(0, -dc) for dc in range(1, shaft_len + 1)]
            + [(-1, -shaft_len), (0, -shaft_len - 1), (1, -shaft_len)]
        )
    if direction == "down":
        return (
            [(dr, 0) for dr in range(1, shaft_len + 1)]
            + [(shaft_len, -1), (shaft_len + 1, 0), (shaft_len, 1)]
        )
    assert direction == "up"
    return (
        [(-dr, 0) for dr in range(1, shaft_len + 1)]
        + [(-shaft_len, -1), (-shaft_len - 1, 0), (-shaft_len, 1)]
    )


def _margin_requirement(direction: str, layers: int, shaft_len: int) -> tuple[int, int, int, int]:
    reach = max(2 * layers - 1, shaft_len + 1)
    if direction == "up":
        return reach, 0, 0, 0
    if direction == "right":
        return 0, reach, 0, 0
    if direction == "down":
        return 0, 0, reach, 0
    return 0, 0, 0, reach


def _place_arrow(
    grid: np.ndarray,
    pattern_bbox: tuple[int, int, int, int],
    direction: str,
    shaft_len: int,
) -> bool:
    pr0, pc0, pr1, pc1 = pattern_bbox
    size = pr1 - pr0 + 1

    if direction in {"left", "right"}:
        anchor_r = pr0 + random.randint(1, size - 2)
        anchor_c = pc0 if direction == "left" else pc1
    else:
        anchor_r = pr0 if direction == "up" else pr1
        anchor_c = pc0 + random.randint(1, size - 2)

    cells = [(anchor_r + dr, anchor_c + dc) for dr, dc in _arrow_offsets(direction, shaft_len)]
    h, w = grid.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BG:
            return False

    for r, c in cells:
        grid[r, c] = ARROW_COLOR
    return True


def _gen_attempt() -> np.ndarray | None:
    size = random.choice(SIZE_CHOICES)
    layers = (size + 1) // 2
    direction = random.choice(DIRECTIONS)
    shaft_len = random.choice(SHAFT_LENGTHS)

    need_top, need_right, need_bottom, need_left = _margin_requirement(direction, layers, shaft_len)
    top_margin = random.randint(max(1, need_top), max(1, need_top) + 4)
    right_margin = random.randint(max(1, need_right), max(1, need_right) + 4)
    bottom_margin = random.randint(max(1, need_bottom), max(1, need_bottom) + 4)
    left_margin = random.randint(max(1, need_left), max(1, need_left) + 4)

    h = top_margin + size + bottom_margin
    w = left_margin + size + right_margin
    grid = np.full((h, w), BG, dtype=int)

    colors = random.sample(PALETTE, layers)
    patch = _nested_patch(size, colors)
    pr0, pc0 = top_margin, left_margin
    grid[pr0 : pr0 + size, pc0 : pc0 + size] = patch
    pattern_bbox = (pr0, pc0, pr0 + size - 1, pc0 + size - 1)

    if not _place_arrow(grid, pattern_bbox, direction, shaft_len):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
