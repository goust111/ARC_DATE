from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4

DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_ATTEMPTS = 400
BBOX_H_RANGE = (5, 9)
BBOX_W_RANGE = (5, 9)
AREA_RATIO_RANGE = (0.38, 0.68)
GAP_RANGE = (2, 7)
MARGIN_RANGE = (0, 5)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for start_r, start_c in np.argwhere(mask):
        sr = int(start_r)
        sc = int(start_c)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        stack = [(sr, sc)]
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for dr, dc in DIRS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _single_component_mask(grid: np.ndarray, color: int) -> np.ndarray:
    comps = _components(grid == color)
    assert len(comps) == 1, f"expected exactly one {color}-colored object"
    return _tight_mask(comps[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align the two same-sized objects by bbox and keep only exclusive pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    fg_colors = set(map(int, np.unique(g))) - {BLACK}
    assert fg_colors == {BLUE, RED}, "expected blue and red objects on a black background"
    assert len(_components(g != BLACK)) == 2, "expected exactly two foreground objects"

    blue_mask = _single_component_mask(g, BLUE)
    red_mask = _single_component_mask(g, RED)
    assert blue_mask.shape == red_mask.shape, "object bboxes must have the same shape"

    out = np.full(blue_mask.shape, BLACK, dtype=int)
    out[np.logical_xor(blue_mask, red_mask)] = YELLOW
    return out


def _random_path(
    rng: np.random.Generator,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    r, c = start
    gr, gc = goal
    path = [(r, c)]
    while (r, c) != (gr, gc):
        steps: list[tuple[int, int]] = []
        if r != gr:
            steps.append((r + (1 if gr > r else -1), c))
        if c != gc:
            steps.append((r, c + (1 if gc > c else -1)))
        r, c = steps[int(rng.integers(len(steps)))]
        path.append((r, c))
    return path


def _frontier(cells: set[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]]:
    frontier: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))
    return list(frontier)


def _random_connected_mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray | None:
    pivot = (int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1)))
    anchors = [
        (0, int(rng.integers(0, w))),
        (h - 1, int(rng.integers(0, w))),
        (int(rng.integers(0, h)), 0),
        (int(rng.integers(0, h)), w - 1),
    ]

    cells: set[tuple[int, int]] = set()
    for anchor in anchors:
        cells.update(_random_path(rng, anchor, pivot))

    min_area = max(len(cells) + 1, int(np.ceil(AREA_RATIO_RANGE[0] * h * w)))
    max_area = min(h * w - 2, int(np.floor(AREA_RATIO_RANGE[1] * h * w)))
    if min_area > max_area:
        return None

    target_area = int(rng.integers(min_area, max_area + 1))
    while len(cells) < target_area:
        candidates = _frontier(cells, h, w)
        if not candidates:
            return None
        cells.add(candidates[int(rng.integers(len(candidates)))])

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _random_orient(rng: np.random.Generator, mask: np.ndarray) -> np.ndarray:
    oriented = mask
    if mask.shape[0] == mask.shape[1]:
        oriented = np.rot90(oriented, int(rng.integers(4)))
    if bool(rng.integers(2)):
        oriented = oriented[::-1, :]
    if bool(rng.integers(2)):
        oriented = oriented[:, ::-1]
    return np.array(oriented, copy=True)


def _valid_pair(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    if mask_a.shape != mask_b.shape or np.array_equal(mask_a, mask_b):
        return False
    overlap = int(np.count_nonzero(mask_a & mask_b))
    blue_only = int(np.count_nonzero(mask_a & ~mask_b))
    red_only = int(np.count_nonzero(mask_b & ~mask_a))
    both_bg = int(np.count_nonzero(~(mask_a | mask_b)))
    return overlap > 0 and blue_only > 0 and red_only > 0 and both_bg > 0


def _place_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    patch = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    patch[mask] = color


def _make_canvas(
    rng: np.random.Generator,
    blue_mask: np.ndarray,
    red_mask: np.ndarray,
) -> np.ndarray:
    h, w = blue_mask.shape
    layout = "horizontal" if bool(rng.integers(2)) else "vertical"

    if layout == "horizontal":
        top_a = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))
        top_b = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))
        bottom_a = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))
        bottom_b = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))
        left = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))
        right = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))
        gap = int(rng.integers(GAP_RANGE[0], GAP_RANGE[1] + 1))

        height = max(top_a + h + bottom_a, top_b + h + bottom_b)
        width = left + w + gap + w + right
        slots = [(top_a, left), (top_b, left + w + gap)]
    else:
        left_a = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))
        left_b = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))
        right_a = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))
        right_b = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))
        top = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))
        bottom = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))
        gap = int(rng.integers(GAP_RANGE[0], GAP_RANGE[1] + 1))

        height = top + h + gap + h + bottom
        width = max(left_a + w + right_a, left_b + w + right_b)
        slots = [(top, left_a), (top + h + gap, left_b)]

    grid = np.full((height, width), BLACK, dtype=int)
    if bool(rng.integers(2)):
        _place_mask(grid, blue_mask, slots[0][0], slots[0][1], BLUE)
        _place_mask(grid, red_mask, slots[1][0], slots[1][1], RED)
    else:
        _place_mask(grid, red_mask, slots[0][0], slots[0][1], RED)
        _place_mask(grid, blue_mask, slots[1][0], slots[1][1], BLUE)
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = int(rng.integers(BBOX_H_RANGE[0], BBOX_H_RANGE[1] + 1))
    w = int(rng.integers(BBOX_W_RANGE[0], BBOX_W_RANGE[1] + 1))

    blue_mask = _random_connected_mask(rng, h, w)
    red_mask = _random_connected_mask(rng, h, w)
    if blue_mask is None or red_mask is None:
        return None

    blue_mask = _random_orient(rng, blue_mask)
    red_mask = _random_orient(rng, red_mask)
    if not _valid_pair(blue_mask, red_mask):
        return None

    grid = _make_canvas(rng, blue_mask, red_mask)
    expected = np.full((h, w), BLACK, dtype=int)
    expected[np.logical_xor(blue_mask, red_mask)] = YELLOW

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
