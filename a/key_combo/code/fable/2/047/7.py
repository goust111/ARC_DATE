from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
INPUT_COLORS = (3, 4, 5, 6, 7, 8, 9)

MAX_ATTEMPTS = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    queue: deque[tuple[int, int]] = deque([start])
    h, w = mask.shape

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))

    return len(seen) == int(mask.sum())


def _has_interior_violation(mask: np.ndarray) -> bool:
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if 0 < rr < h - 1 and 0 < cc < w - 1:
            if mask[rr - 1, cc] and mask[rr + 1, cc] and mask[rr, cc - 1] and mask[rr, cc + 1]:
                return True
    return False


def _phase_analysis(grid: np.ndarray) -> dict[str, np.ndarray | int | tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    colors = np.unique(g)
    assert len(colors) == 2, "expected exactly two checkerboard colors"
    color0, color1 = map(int, colors.tolist())

    rr, cc = np.indices(g.shape)
    even = ((rr + cc) % 2 == 0)

    score0 = int(np.count_nonzero(g[even] == color0) + np.count_nonzero(g[~even] == color1))
    score1 = int(np.count_nonzero(g[even] == color1) + np.count_nonzero(g[~even] == color0))
    assert score0 != score1, "checkerboard phase must be uniquely determined"

    if score0 > score1:
        even_color, odd_color = color0, color1
    else:
        even_color, odd_color = color1, color0

    expected = np.where(even, even_color, odd_color)
    violations = g != expected
    assert violations.any(), "expected at least one checkerboard violation"

    box = _bbox(violations)
    assert box is not None

    return {
        "grid": g,
        "even_mask": even,
        "expected": expected,
        "violations": violations,
        "bbox": box,
        "even_color": even_color,
        "odd_color": odd_color,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden checkerboard violations, recolor them by flip direction, and crop to their bbox.

    Input structure:
    - The grid contains exactly two visible colors arranged as an almost-perfect checkerboard.
    - One checkerboard phase fits the whole grid better than the other, so the expected color of each
      coordinate is recoverable from the global parity frame.
    - A small set of cells is flipped to the opposite checkerboard color; those flipped cells form the
      hidden figure.

    Rule:
    - Infer which color belongs on even `(r+c)%2` positions by choosing the checkerboard phase with the
      fewest mismatches.
    - Mark every cell whose actual color disagrees with its expected checkerboard color.
    - Return the minimal bbox of all marked cells, using blue for cells that should be the even-parity
      color, red for cells that should be the odd-parity color, and black elsewhere inside the crop.

    Key insight:
    - The figure is not any one visible input color; it is the set of parity violations, so the global
      checkerboard frame must be reconstructed before the hidden shape and its two polarities appear.
    """
    info = _phase_analysis(grid)
    violations = info["violations"]
    even_mask = info["even_mask"]
    r0, c0, r1, c1 = info["bbox"]

    out = np.full((r1 - r0 + 1, c1 - c0 + 1), BLACK, dtype=int)
    crop_violations = violations[r0 : r1 + 1, c0 : c1 + 1]
    crop_even = even_mask[r0 : r1 + 1, c0 : c1 + 1]

    out[crop_violations & crop_even] = BLUE
    out[crop_violations & ~crop_even] = RED
    return out


def _grow_shape() -> np.ndarray:
    work_h = random.randint(6, 12)
    work_w = random.randint(6, 12)
    mask = np.zeros((work_h, work_w), dtype=bool)

    seed_h = random.randint(3, min(5, work_h - 1))
    seed_w = random.randint(3, min(5, work_w - 1))
    seed_r = random.randint(0, work_h - seed_h)
    seed_c = random.randint(0, work_w - seed_w)
    mask[seed_r : seed_r + seed_h, seed_c : seed_c + seed_w] = True

    max_area = min(work_h * work_w - 3, 34)
    target_area = random.randint(max(seed_h * seed_w + 2, 11), max_area)

    while int(mask.sum()) < target_area:
        candidates: list[tuple[int, int, int]] = []
        for r in range(work_h):
            for c in range(work_w):
                if mask[r, c]:
                    continue
                neighbors = sum(
                    1
                    for nr, nc in _neighbors4(r, c)
                    if 0 <= nr < work_h and 0 <= nc < work_w and mask[nr, nc]
                )
                if neighbors:
                    candidates.append((neighbors, r, c))

        assert candidates, "shape growth must retain a frontier"
        candidates.sort(reverse=True)
        best = candidates[: max(3, len(candidates) // 3)]
        _neighbors, r, c = random.choice(best)
        mask[r, c] = True

    for _ in range(random.randint(1, 4)):
        removable: list[tuple[int, int]] = []
        for r, c in np.argwhere(mask):
            rr = int(r)
            cc = int(c)
            if all(
                0 <= nr < work_h and 0 <= nc < work_w and mask[nr, nc]
                for nr, nc in _neighbors4(rr, cc)
            ):
                continue

            trial = mask.copy()
            trial[rr, cc] = False
            if trial.sum() < 11:
                continue
            if _is_connected(trial) and _has_interior_violation(trial):
                removable.append((rr, cc))

        if not removable:
            break
        rr, cc = random.choice(removable)
        mask[rr, cc] = False

    box = _bbox(mask)
    assert box is not None
    mask = _crop(mask, box)

    if mask.all():
        return _grow_shape()
    if not _is_connected(mask):
        return _grow_shape()
    if not _has_interior_violation(mask):
        return _grow_shape()

    return mask


def _parity_counts(mask: np.ndarray, offset_parity: int) -> tuple[int, int]:
    rr, cc = np.indices(mask.shape)
    local_even = ((rr + cc) % 2 == 0)
    if offset_parity % 2 == 0:
        even_count = int(np.count_nonzero(mask & local_even))
        odd_count = int(np.count_nonzero(mask & ~local_even))
    else:
        even_count = int(np.count_nonzero(mask & ~local_even))
        odd_count = int(np.count_nonzero(mask & local_even))
    return even_count, odd_count


def _candidate_positions(grid_shape: tuple[int, int], mask_shape: tuple[int, int], desired_parity: int) -> list[tuple[int, int]]:
    gh, gw = grid_shape
    mh, mw = mask_shape
    positions = []
    for r0 in range(gh - mh + 1):
        for c0 in range(gw - mw + 1):
            if (r0 + c0) % 2 == desired_parity:
                positions.append((r0, c0))
    return positions


def _random_odd(low: int, high: int) -> int:
    if low % 2 == 0:
        low += 1
    if high % 2 == 0:
        high -= 1
    assert low <= high, "expected an odd size range"
    return random.randrange(low, high + 1, 2)


def _render_case(shape: np.ndarray, even_color: int, odd_color: int, grid_h: int, grid_w: int, top: int, left: int) -> np.ndarray:
    rr, cc = np.indices((grid_h, grid_w))
    grid = np.where((rr + cc) % 2 == 0, even_color, odd_color).astype(int)
    for r, c in np.argwhere(shape):
        rr_abs = top + int(r)
        cc_abs = left + int(c)
        grid[rr_abs, cc_abs] = odd_color if (rr_abs + cc_abs) % 2 == 0 else even_color
    return grid


def _semantic_checks(grid: np.ndarray) -> None:
    info = _phase_analysis(grid)
    g = info["grid"]
    violations = info["violations"]
    box = info["bbox"]
    even_color = int(info["even_color"])
    odd_color = int(info["odd_color"])

    assert g.shape[0] % 2 == 1 and g.shape[1] % 2 == 1, "generator must use odd-by-odd boards"
    assert _is_connected(violations), "violations must form one hidden figure"
    assert _has_interior_violation(violations), "figure must contain locally hidden interior cells"

    counts = {int(color): int(np.count_nonzero(g == int(color))) for color in np.unique(g)}
    assert counts[even_color] > counts[odd_color], "even-parity checker color must remain globally dominant"

    blue_count = int(np.count_nonzero(violations & info["even_mask"]))
    red_count = int(np.count_nonzero(violations & ~info["even_mask"]))
    assert blue_count > 0 and red_count > 0, "both violation directions must appear"

    r0, c0, r1, c1 = box
    bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
    assert int(violations.sum()) < bbox_area, "crop must retain black negative space"

    out = main(g)
    assert out.shape == (r1 - r0 + 1, c1 - c0 + 1)
    assert np.count_nonzero(out == BLUE) == blue_count
    assert np.count_nonzero(out == RED) == red_count
    assert np.count_nonzero(out == BLACK) == bbox_area - blue_count - red_count


def _gen_attempt() -> np.ndarray | None:
    shape = _grow_shape()
    sh, sw = shape.shape

    grid_h = _random_odd(max(11, sh + 3), 29)
    grid_w = _random_odd(max(11, sw + 3), 29)

    local_even, local_odd = _parity_counts(shape, offset_parity=0)
    desired_offset_parity = 0 if local_odd >= local_even else 1
    positions = _candidate_positions((grid_h, grid_w), (sh, sw), desired_offset_parity)
    if not positions:
        return None

    positions.sort(key=lambda pos: min(pos[0], pos[1], grid_h - sh - pos[0], grid_w - sw - pos[1]))
    interior = positions[max(0, len(positions) // 3) :]
    edgeish = positions[: max(1, len(positions) // 4)]
    roll = random.random()
    if roll < 0.75 and interior:
        top, left = random.choice(interior)
    elif roll < 0.9:
        top, left = random.choice(positions)
    else:
        top, left = random.choice(edgeish)

    even_count, odd_count = _parity_counts(shape, offset_parity=(top + left) % 2)
    if odd_count < even_count:
        return None

    even_color, odd_color = random.sample(INPUT_COLORS, 2)
    grid = _render_case(shape, even_color, odd_color, grid_h, grid_w, top, left)

    try:
        _semantic_checks(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid checkerboard-violation puzzle")
