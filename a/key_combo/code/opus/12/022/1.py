import random
from typing import Iterable

import numpy as np


BG = 0
CROSS = 1
MASK = 2
MARKER = 3

SIZE_CHOICES = (11, 13, 15, 17)
MASK_SIZE_RANGE = (4, 9)
RETRY_LIMIT = 400

ARM_PAIRS = (
    ("top", "bottom"),
    ("left", "right"),
    ("top", "right"),
    ("right", "bottom"),
    ("bottom", "left"),
    ("left", "top"),
)
ARM_PAIR_WEIGHTS = (2, 2, 1, 1, 1, 1)


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _ray_spec(q: int, arm: str, distance: int) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    n = 2 * q + 1
    d = int(distance)
    if arm == "top":
        marker = (q - d, q)
        cells = [(r, q - d) for r in range(0, q - d + 1)]
    elif arm == "bottom":
        marker = (q + d, q)
        cells = [(r, q + d) for r in range(q + d, n)]
    elif arm == "left":
        marker = (q, q - d)
        cells = [(q + d, c) for c in range(0, q - d + 1)]
    elif arm == "right":
        marker = (q, q + d)
        cells = [(q - d, c) for c in range(q + d, n)]
    else:
        raise ValueError(f"unknown arm: {arm}")
    return marker, cells


def _render_blue_mirror(mask: np.ndarray) -> np.ndarray:
    q = mask.shape[0]
    n = 2 * q + 1
    out = np.zeros((n, n), dtype=int)
    out[:q, :q][mask] = CROSS
    out[:q, q + 1 :][np.fliplr(mask)] = CROSS
    out[q + 1 :, :q][np.flipud(mask)] = CROSS
    out[q + 1 :, q + 1 :][np.flipud(np.fliplr(mask))] = CROSS
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the central cross, mirror the upper-left mask into four blue quadrants, and carve directional green rays through selected copies.

    Input structure:
    - The grid is an odd square with black background.
    - A blue horizontal line and a blue vertical line cross at the center, except that two green arm markers overwrite two cross cells.
    - The upper-left quadrant contains a red 4-connected mask that stays at least one cell away from the cross.

    Rule:
    - Treat the central row and column as the deleted cross and rebuild the output from the upper-left red mask alone.
    - Recolor that mask to blue and mirror it into the other three quadrants around the vanished cross, leaving the former cross row and column empty.
    - For each green arm marker, keep the marker itself green, then cast a green ray parallel to the deleted arm at the same offset from the center inside the corresponding mirrored quadrant, extending away from the center to the edge.
    - Along each ray, erase blue pixels to black, paint black pixels green, and leave any other color unchanged.

    Key insight:
    - The cross supplies both the symmetry frame and the ray offsets: after the cross disappears, each marker's distance from the center selects which mirrored row or column gets eroded.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1] and g.shape[0] % 2 == 1

    n = g.shape[0]
    q = n // 2
    source_mask = g[:q, :q] == MASK
    out = _render_blue_mirror(source_mask)

    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == MARKER)]
    assert len(markers) == 2

    for r, c in markers:
        out[r, c] = MARKER
        if r == q and c > q:
            arm = "right"
            distance = c - q
        elif r == q and c < q:
            arm = "left"
            distance = q - c
        elif c == q and r < q:
            arm = "top"
            distance = q - r
        elif c == q and r > q:
            arm = "bottom"
            distance = r - q
        else:
            raise AssertionError("green markers must lie on distinct cross arms")

        _, cells = _ray_spec(q, arm, distance)
        for rr, cc in cells:
            if out[rr, cc] == CROSS:
                out[rr, cc] = BG
            elif out[rr, cc] == BG:
                out[rr, cc] = MARKER

    return out


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _random_connected_mask(q: int, k: int) -> np.ndarray:
    usable = [(r, c) for r in range(q - 1) for c in range(q - 1)]
    seed = random.choice(usable)
    cells = {seed}
    while len(cells) < k:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < q - 1 and 0 <= nc < q - 1 and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))

    mask = np.zeros((q, q), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _mask_requirements(mask: np.ndarray, arm_dists: list[tuple[str, int]]) -> bool:
    q = mask.shape[0]
    if not _connected(mask):
        return False
    rows, cols = np.where(mask)
    if rows.ptp() < 1 or cols.ptp() < 1:
        return False
    if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)):
        return False

    for arm, distance in arm_dists:
        idx = q - distance
        if arm in ("left", "right"):
            if mask[idx, : idx + 1].sum() == 0:
                return False
        else:
            if mask[: idx + 1, idx].sum() == 0:
                return False
    return True


def _sample_mask(q: int, arm_dists: list[tuple[str, int]]) -> np.ndarray | None:
    for _ in range(120):
        k = random.randint(*MASK_SIZE_RANGE)
        mask = _random_connected_mask(q, k)
        if _mask_requirements(mask, arm_dists):
            return mask
    return None


def _build_input(mask: np.ndarray, arm_dists: list[tuple[str, int]]) -> np.ndarray:
    q = mask.shape[0]
    n = 2 * q + 1
    grid = np.zeros((n, n), dtype=int)
    grid[q, :] = CROSS
    grid[:, q] = CROSS
    grid[:q, :q][mask] = MASK
    for arm, distance in arm_dists:
        marker, _ = _ray_spec(q, arm, distance)
        grid[marker] = MARKER
    return grid


def _passes_semantic_checks(mask: np.ndarray, arm_dists: list[tuple[str, int]]) -> bool:
    q = mask.shape[0]
    base = _render_blue_mirror(mask)
    total_erased = 0
    total_painted = 0
    for arm, distance in arm_dists:
        _, cells = _ray_spec(q, arm, distance)
        values = np.array([base[r, c] for r, c in cells], dtype=int)
        if not np.any(values == CROSS):
            return False
        if not np.any(values == BG):
            return False
        total_erased += int(np.sum(values == CROSS))
        total_painted += int(np.sum(values == BG))

    if total_erased < 2 or total_painted < 2:
        return False

    inp = _build_input(mask, arm_dists)
    out = main(inp)
    if np.array_equal(inp, out):
        return False
    if np.array_equal(out, base):
        return False
    if int(np.sum(out == CROSS)) >= int(np.sum(base == CROSS)):
        return False
    if int(np.sum(out == MARKER)) <= len(arm_dists):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(SIZE_CHOICES)
    q = n // 2
    arm_pair = random.choices(ARM_PAIRS, weights=ARM_PAIR_WEIGHTS, k=1)[0]
    arm_dists = [(arm, random.randint(2, q)) for arm in arm_pair]
    mask = _sample_mask(q, arm_dists)
    if mask is None:
        return None
    if not _passes_semantic_checks(mask, arm_dists):
        return None
    return _build_input(mask, arm_dists)


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {RETRY_LIMIT} attempts")
