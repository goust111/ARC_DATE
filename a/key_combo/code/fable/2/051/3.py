from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
TEAL = 8
MAROON = 9

_TRANSFORM_NAMES = (
    "id",
    "rot90",
    "rot180",
    "rot270",
    "flip_lr",
    "flip_ud",
    "flip_main",
    "flip_anti",
)

_SHAPE_LIBRARY = (
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 0], [0, 0, 1, 1]], dtype=bool),
)

_GRID_SIZE_RANGE = (7, 14)
_COMPONENT_COUNT_RANGE = (3, 5)
_AREA_RATIO_RANGE = (0.14, 0.36)
_PLACEMENT_TRIES = 60
_BUILD_RETRIES = 60
_GENERATE_RETRIES = 400


def _apply_transform(mask: np.ndarray, name: str) -> np.ndarray:
    if name == "id":
        out = mask
    elif name == "rot90":
        out = np.rot90(mask, 1)
    elif name == "rot180":
        out = np.rot90(mask, 2)
    elif name == "rot270":
        out = np.rot90(mask, 3)
    elif name == "flip_lr":
        out = np.fliplr(mask)
    elif name == "flip_ud":
        out = np.flipud(mask)
    elif name == "flip_main":
        out = mask.T
    elif name == "flip_anti":
        out = np.fliplr(np.rot90(mask, 1))
    else:
        raise ValueError(f"unknown transform: {name}")
    return np.array(out, dtype=bool, copy=True)


def _all_transforms(mask: np.ndarray) -> list[np.ndarray]:
    return [_apply_transform(mask, name) for name in _TRANSFORM_NAMES]


def _split_masks(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    full_blue_cols = np.flatnonzero(np.all(grid == BLUE, axis=0))
    full_blue_rows = np.flatnonzero(np.all(grid == BLUE, axis=1))

    if len(full_blue_cols) == 1 and len(full_blue_rows) == 0:
        sep = int(full_blue_cols[0])
        assert np.count_nonzero(grid == BLUE) == grid.shape[0], "separator must be a single full blue column"
        first = grid[:, :sep]
        second = grid[:, sep + 1 :]
    elif len(full_blue_rows) == 1 and len(full_blue_cols) == 0:
        sep = int(full_blue_rows[0])
        assert np.count_nonzero(grid == BLUE) == grid.shape[1], "separator must be a single full blue row"
        first = grid[:sep, :]
        second = grid[sep + 1 :, :]
    else:
        raise AssertionError("expected one full blue separator bar")

    assert first.shape == second.shape, "the two masks must have the same size"
    assert first.ndim == 2 and first.shape[0] == first.shape[1], "the hidden-transform masks must be square"
    for block in (first, second):
        colors = set(map(int, np.unique(block)))
        assert colors <= {BLACK, MAROON}, "mask blocks must be binary maroon/black"
    return first == MAROON, second == MAROON


def _best_alignment(first: np.ndarray, second: np.ndarray) -> tuple[int, np.ndarray, list[int]]:
    variants = _all_transforms(first)
    diffs = [variant != second for variant in variants]
    counts = [int(diff.sum()) for diff in diffs]
    best = min(counts)
    winners = [idx for idx, count in enumerate(counts) if count == best]
    assert len(winners) == 1, "expected a unique best rigid transform"
    winner = winners[0]
    return winner, diffs[winner], counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align the left/top binary mask to the right/bottom one by the unique best rigid transform and output the leftover mismatches.

    Input structure:
    - A single blue separator bar splits the canvas into two equal-sized maroon/black binary masks.
    - Maroon cells are active pixels and black cells are inactive pixels.
    - The second mask is a slightly edited rigid transform of the first, but the transform itself is not marked.

    Rule:
    - Try all eight square rigid transforms of the first mask: identity, quarter-turn rotations, and the four reflections.
    - Compare each transformed version to the second mask in the second mask's coordinate system.
    - Select the unique transform with the strictly fewest disagreements and output a same-sized grid whose mismatching cells are teal and whose matching cells are black.

    Key insight:
    - The visible left/right or top/bottom placement is only a carrier; the answer is the residual after undoing the hidden best alignment, not the raw side-by-side difference.
    """
    g = np.asarray(grid, dtype=int)
    first, second = _split_masks(g)
    _best_idx, diff, _counts = _best_alignment(first, second)
    return np.where(diff, TEAL, BLACK).astype(int)


def _component_patch() -> np.ndarray:
    patch = random.choice(_SHAPE_LIBRARY)
    return _apply_transform(patch, random.choice(_TRANSFORM_NAMES))


def _can_place(occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    n = occupied.shape[0]
    for pr, pc in np.argwhere(patch):
        rr = int(r0 + pr)
        cc = int(c0 + pc)
        if occupied[max(0, rr - 1) : min(n, rr + 2), max(0, cc - 1) : min(n, cc + 2)].any():
            return False
    return True


def _place_patch(mask: np.ndarray, occupied: np.ndarray, patch: np.ndarray) -> bool:
    n = mask.shape[0]
    h, w = patch.shape
    if h > n or w > n:
        return False
    for _ in range(_PLACEMENT_TRIES):
        r0 = random.randint(0, n - h)
        c0 = random.randint(0, n - w)
        if not _can_place(occupied, patch, r0, c0):
            continue
        coords = np.argwhere(patch)
        rows = coords[:, 0] + r0
        cols = coords[:, 1] + c0
        mask[rows, cols] = True
        occupied[rows, cols] = True
        return True
    return False


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _build_first_mask() -> np.ndarray | None:
    n = random.randint(*_GRID_SIZE_RANGE)
    mask = np.zeros((n, n), dtype=bool)
    occupied = np.zeros_like(mask, dtype=bool)
    component_count = random.randint(*_COMPONENT_COUNT_RANGE)

    for _ in range(component_count):
        if not _place_patch(mask, occupied, _component_patch()):
            return None

    area = int(mask.sum())
    lo = int(np.ceil(n * n * _AREA_RATIO_RANGE[0]))
    hi = int(np.floor(n * n * _AREA_RATIO_RANGE[1]))
    if not (lo <= area <= hi):
        return None

    r0, c0, r1, c1 = _bbox(mask)
    if r1 - r0 + 1 < max(4, n // 2) or c1 - c0 + 1 < max(4, n // 2):
        return None

    variants = _all_transforms(mask)
    sigs = {variant.astype(np.uint8).tobytes() for variant in variants}
    if len(sigs) != len(_TRANSFORM_NAMES):
        return None

    min_pairwise = min(
        int(np.count_nonzero(variants[i] != variants[j]))
        for i in range(len(variants))
        for j in range(i + 1, len(variants))
    )
    if min_pairwise < 5:
        return None
    return mask


def _toggle_cells(mask: np.ndarray, count: int) -> tuple[np.ndarray, set[tuple[int, int]]]:
    out = mask.copy()
    active = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
    inactive = [tuple(map(int, coord)) for coord in np.argwhere(~mask)]
    chosen: set[tuple[int, int]] = set()

    if count >= 2 and active and inactive and random.random() < 0.7:
        chosen.add(random.choice(active))
        chosen.add(random.choice(inactive))

    all_cells = [(r, c) for r in range(mask.shape[0]) for c in range(mask.shape[1])]
    random.shuffle(all_cells)
    for cell in all_cells:
        if len(chosen) >= count:
            break
        chosen.add(cell)

    assert len(chosen) == count
    for r, c in chosen:
        out[r, c] = ~out[r, c]
    return out, chosen


def _render_input(first: np.ndarray, second: np.ndarray, orientation: str) -> np.ndarray:
    first_grid = np.where(first, MAROON, BLACK).astype(int)
    second_grid = np.where(second, MAROON, BLACK).astype(int)
    n = first.shape[0]

    if orientation == "vertical":
        out = np.full((n, 2 * n + 1), BLACK, dtype=int)
        out[:, :n] = first_grid
        out[:, n] = BLUE
        out[:, n + 1 :] = second_grid
        return out

    out = np.full((2 * n + 1, n), BLACK, dtype=int)
    out[:n, :] = first_grid
    out[n, :] = BLUE
    out[n + 1 :, :] = second_grid
    return out


def _expected_output(first: np.ndarray, second: np.ndarray) -> tuple[int, np.ndarray, list[int]]:
    best_idx, diff, counts = _best_alignment(first, second)
    expected = np.where(diff, TEAL, BLACK).astype(int)
    return best_idx, expected, counts


def _validate_instance(
    grid: np.ndarray,
    expected_best_idx: int,
    expected_diff_cells: set[tuple[int, int]],
    expected_orientation: str,
) -> bool:
    first, second = _split_masks(grid)
    best_idx, diff, counts = _best_alignment(first, second)
    if best_idx != expected_best_idx:
        return False
    if counts[best_idx] != len(expected_diff_cells):
        return False
    if {tuple(map(int, coord)) for coord in np.argwhere(diff)} != expected_diff_cells:
        return False

    full_blue_cols = np.flatnonzero(np.all(grid == BLUE, axis=0))
    full_blue_rows = np.flatnonzero(np.all(grid == BLUE, axis=1))
    orientation = "vertical" if len(full_blue_cols) == 1 else "horizontal"
    if orientation != expected_orientation:
        return False

    if best_idx != 0:
        direct = int(np.count_nonzero(first != second))
        if direct <= counts[best_idx]:
            return False
        if np.array_equal(first != second, diff):
            return False

    return bool(np.any(diff))


def _attempt() -> np.ndarray | None:
    for _ in range(_BUILD_RETRIES):
        first = _build_first_mask()
        if first is None:
            continue

        variants = _all_transforms(first)
        best_idx = random.randrange(len(variants))
        aligned = variants[best_idx]
        min_alt = min(
            int(np.count_nonzero(aligned != variants[idx]))
            for idx in range(len(variants))
            if idx != best_idx
        )
        max_edits = min(4, (min_alt - 1) // 2)
        if max_edits < 1:
            continue

        edit_count = random.randint(1, max_edits)
        second, edited_cells = _toggle_cells(aligned, edit_count)
        solved_idx, expected, counts = _expected_output(first, second)
        if solved_idx != best_idx or counts[solved_idx] != edit_count:
            continue

        orientation = random.choice(("vertical", "horizontal"))
        grid = _render_input(first, second, orientation)
        if not _validate_instance(grid, best_idx, edited_cells, orientation):
            continue

        try:
            actual = main(grid)
        except AssertionError:
            continue
        if not np.array_equal(actual, expected):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GENERATE_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-alignment XOR task")
