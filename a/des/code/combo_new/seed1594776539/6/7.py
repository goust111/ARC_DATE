from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
AZURE = 8
MAROON = 9

WARM_COLORS = (RED, ORANGE, YELLOW, MAROON)
COLD_COLORS = (BLUE, GREEN, AZURE)
SUPPORTED_COLORS = set(WARM_COLORS) | set(COLD_COLORS)

GRID_H_RANGE = (10, 22)
GRID_W_RANGE = (10, 22)
STAMP_COUNT_RANGE = (3, 7)
PLACEMENT_TRIES = 160
GEN_ATTEMPTS = 400

DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _checkerboard(shape: tuple[int, int]) -> np.ndarray:
    rows, cols = np.indices(shape)
    return np.where((rows + cols) % 2 == 0, GREY, BLACK).astype(int)


def _cell_bg(r: int, c: int) -> int:
    return GREY if (r + c) % 2 == 0 else BLACK


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _mask_is_connected(mask: np.ndarray) -> bool:
    coords = list(zip(*np.where(mask)))
    if not coords:
        return False
    h, w = mask.shape
    seen = {coords[0]}
    q: deque[tuple[int, int]] = deque([coords[0]])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _touches_bbox_edges(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _stamp_masks(axis: str) -> list[np.ndarray]:
    assert axis in {"horizontal", "vertical"}
    flip = np.flipud if axis == "horizontal" else np.fliplr
    masks: list[np.ndarray] = []
    for h in range(1, 4):
        for w in range(1, 4):
            if axis == "horizontal" and h < 2:
                continue
            if axis == "vertical" and w < 2:
                continue
            for bits in range(1, 1 << (h * w)):
                mask = np.zeros((h, w), dtype=bool)
                for idx in range(h * w):
                    if bits & (1 << idx):
                        mask[idx // w, idx % w] = True
                if not _touches_bbox_edges(mask):
                    continue
                if not _mask_is_connected(mask):
                    continue
                if np.array_equal(mask, flip(mask)):
                    continue
                masks.append(mask)
    return masks


WARM_MASKS = _stamp_masks("horizontal")
COLD_MASKS = _stamp_masks("vertical")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each colored stamp across the axis class encoded by its visible color while preserving the checkerboard.

    Input structure:
    - The background is a grey/black checkerboard with grey in the top-left corner.
    - Several small 4-connected monochromatic stamps of size at most 3x3 sit on top of that checkerboard.
    - Stamp colors are drawn from two visible categories: warm colors and cold colors.
    - Every stamp is asymmetric with respect to the flip axis assigned to its color category.

    Rule:
    - Split every non-grey, non-black stamp into its own 4-connected component and take that component's bounding box.
    - Warm colors (red, orange, yellow, maroon) are flipped across the horizontal axis through the bbox center, so the stamp is mirrored top-to-bottom in place.
    - Cold colors (blue, green, azure) are flipped across the vertical axis through the bbox center, so the stamp is mirrored left-to-right in place.
    - Replace each original stamp with its flipped version and leave every uncovered checkerboard cell unchanged.

    Key insight:
    - The color itself does not move the stamp; it selects which mirror axis to use, and the mirror is local to each stamp's own bounding box rather than to the whole grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    expected_bg = _checkerboard(g.shape)
    bg_mask = (g == GREY) | (g == BLACK)
    assert np.array_equal(g[bg_mask], expected_bg[bg_mask]), "background must be the canonical checkerboard"

    out = g.copy()
    colors = [int(color) for color in np.unique(g) if int(color) not in (BLACK, GREY)]
    assert colors, "expected at least one colored stamp"
    assert all(color in SUPPORTED_COLORS for color in colors), "found unsupported stamp color"

    for color in colors:
        axis = "horizontal" if color in WARM_COLORS else "vertical"
        for comp in _components_of_color(g, color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0 = min(rows)
            r1 = max(rows)
            c0 = min(cols)
            c1 = max(cols)
            assert r1 - r0 + 1 <= 3 and c1 - c0 + 1 <= 3, "stamp bbox must be at most 3x3"

            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for r, c in comp:
                mask[r - r0, c - c0] = True
            flipped = np.flipud(mask) if axis == "horizontal" else np.fliplr(mask)
            assert not np.array_equal(mask, flipped), "stamp must change under its category flip"

            source = set(comp)
            dest = {
                (r0 + int(rr), c0 + int(cc))
                for rr, cc in zip(*np.where(flipped))
            }
            for r, c in dest:
                assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1]
                if (r, c) not in source:
                    assert int(g[r, c]) in (BLACK, GREY), "flipped stamp may not overwrite another stamp"

            for r, c in source:
                out[r, c] = _cell_bg(r, c)
            for r, c in dest:
                out[r, c] = color

    return out


def _sample_color_counts(total: int) -> tuple[int, int]:
    options = []
    for warm_count in range(1, min(len(WARM_COLORS), total - 1) + 1):
        cold_count = total - warm_count
        if 1 <= cold_count <= len(COLD_COLORS):
            options.append((warm_count, cold_count))
    near_balanced = [pair for pair in options if abs(pair[0] - pair[1]) <= 1]
    return random.choice(near_balanced or options)


def _expanded_box(top: int, left: int, h: int, w: int, grid_h: int, grid_w: int) -> tuple[int, int, int, int]:
    return (
        max(0, top - 1),
        max(0, left - 1),
        min(grid_h - 1, top + h),
        min(grid_w - 1, left + w),
    )


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_stamps(
    grid: np.ndarray,
    specs: list[tuple[int, np.ndarray]],
) -> bool:
    h, w = grid.shape
    blocked: list[tuple[int, int, int, int]] = []
    ordered = sorted(specs, key=lambda item: item[1].shape[0] * item[1].shape[1], reverse=True)

    for color, mask in ordered:
        mh, mw = map(int, mask.shape)
        placed = False
        for _ in range(PLACEMENT_TRIES):
            top = random.randint(0, h - mh)
            left = random.randint(0, w - mw)
            box = _expanded_box(top, left, mh, mw, h, w)
            if any(_boxes_overlap(box, other) for other in blocked):
                continue
            coords = np.argwhere(mask)
            for rr, cc in coords:
                grid[top + int(rr), left + int(cc)] = color
            blocked.append(box)
            placed = True
            break
        if not placed:
            return False
    return True


def _all_branch_colors_present(grid: np.ndarray) -> bool:
    colors = {int(color) for color in np.unique(grid) if int(color) not in (BLACK, GREY)}
    return bool(colors & set(WARM_COLORS)) and bool(colors & set(COLD_COLORS))


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    grid_w = random.randint(*GRID_W_RANGE)
    grid = _checkerboard((grid_h, grid_w))

    total = random.randint(*STAMP_COUNT_RANGE)
    warm_count, cold_count = _sample_color_counts(total)
    warm_colors = random.sample(list(WARM_COLORS), warm_count)
    cold_colors = random.sample(list(COLD_COLORS), cold_count)

    specs: list[tuple[int, np.ndarray]] = []
    for color in warm_colors:
        specs.append((color, random.choice(WARM_MASKS).copy()))
    for color in cold_colors:
        specs.append((color, random.choice(COLD_MASKS).copy()))
    random.shuffle(specs)

    if not _place_stamps(grid, specs):
        return None
    if not _all_branch_colors_present(grid):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid checkerboard stamp reflection input")
