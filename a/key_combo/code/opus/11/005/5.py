import numpy as np


BLACK = 0
RED = 2
GREEN = 3
SIDES = ("left", "right", "top", "bottom")
RNG = np.random.default_rng()

MIN_SIDE = 12
MAX_SIDE = 15
MIN_AREA = 8
MAX_AREA = 20
GEN_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    if box is None:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _touch_side(mask: np.ndarray) -> str | None:
    touches = {
        "left": bool(np.any(mask[:, 0])),
        "right": bool(np.any(mask[:, -1])),
        "top": bool(np.any(mask[0, :])),
        "bottom": bool(np.any(mask[-1, :])),
    }
    active = [side for side in SIDES if touches[side]]
    if len(active) != 1:
        return None
    return active[0]


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    height, width = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < height and 0 <= nc < width and mask[nr, nc]:
                nxt = (int(nr), int(nc))
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return len(seen) == int(mask.sum())


def _count_runs(line: np.ndarray) -> int:
    runs = 0
    inside = False
    for value in line:
        if bool(value):
            if not inside:
                runs += 1
                inside = True
        else:
            inside = False
    return runs


def _is_nonconvex(mask: np.ndarray) -> bool:
    height, width = mask.shape
    for r in range(height):
        if _count_runs(mask[r]) >= 2:
            return True
    for c in range(width):
        if _count_runs(mask[:, c]) >= 2:
            return True
    return False


def _is_asymmetric(mask: np.ndarray) -> bool:
    crop = _crop_mask(mask)
    return not np.array_equal(crop, np.fliplr(crop)) and not np.array_equal(crop, np.flipud(crop))


def _mirror_shared(grid: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return np.concatenate((np.fliplr(grid[:, 1:]), grid), axis=1)
    if side == "right":
        return np.concatenate((grid, np.fliplr(grid[:, :-1])), axis=1)
    if side == "top":
        return np.concatenate((np.flipud(grid[1:, :]), grid), axis=0)
    if side == "bottom":
        return np.concatenate((grid, np.flipud(grid[:-1, :])), axis=0)
    raise AssertionError(f"unknown side: {side}")


def _corridor_masks(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    hmask = np.zeros_like(g, dtype=bool)
    vmask = np.zeros_like(g, dtype=bool)
    row_count = 0
    col_count = 0

    for r in range(height):
        cols = np.flatnonzero(g[r] == RED)
        if cols.size < 2:
            continue
        c0 = int(cols[0])
        c1 = int(cols[-1])
        if c1 - c0 >= 2 and np.all(g[r, c0 + 1 : c1] == BLACK):
            hmask[r, c0 + 1 : c1] = True
            row_count += 1

    for c in range(width):
        rows = np.flatnonzero(g[:, c] == RED)
        if rows.size < 2:
            continue
        r0 = int(rows[0])
        r1 = int(rows[-1])
        if r1 - r0 >= 2 and np.all(g[r0 + 1 : r1, c] == BLACK):
            vmask[r0 + 1 : r1, c] = True
            col_count += 1

    return hmask, vmask, row_count, col_count


def _build_left_template(height: int, width: int) -> np.ndarray | None:
    max_anchor = min(4, width - 4)
    if max_anchor < 1:
        return None

    for _ in range(80):
        span = int(RNG.integers(4, 8))
        if span > height - 2:
            continue
        r0 = int(RNG.integers(1, height - span))
        r2 = r0 + span - 1
        r_mid = int(RNG.integers(r0 + 1, r2))
        anchor = int(RNG.integers(1, max_anchor + 1))
        min_tip = anchor + 2
        max_tip = min(width - 2, anchor + 4)
        if min_tip > max_tip:
            continue
        tip = int(RNG.integers(min_tip, max_tip + 1))

        mask = np.zeros((height, width), dtype=bool)
        mask[r0 : r2 + 1, anchor] = True
        mask[r_mid, : anchor + 1] = True
        mask[r0, anchor : tip + 1] = True
        mask[r2, anchor : tip + 1] = True

        candidates = []
        if anchor + 1 <= width - 2:
            if r0 + 1 < r2 and r0 + 1 != r_mid:
                candidates.append((r0 + 1, anchor + 1))
            if r2 - 1 > r0 and r2 - 1 != r_mid:
                candidates.append((r2 - 1, anchor + 1))
            candidates.append((r_mid, anchor + 1))
        if tip + 1 <= width - 2:
            candidates.append((r0, tip + 1))
            candidates.append((r2, tip + 1))

        if not candidates:
            continue

        extra_count = 1
        base_area = int(mask.sum())
        if base_area <= MAX_AREA - 2 and len(candidates) >= 2 and float(RNG.random()) < 0.35:
            extra_count = 2
        for index in RNG.choice(len(candidates), size=extra_count, replace=False):
            rr, cc = candidates[int(index)]
            mask[rr, cc] = True

        area = int(mask.sum())
        if MIN_AREA <= area <= MAX_AREA and _is_connected(mask):
            return mask
    return None


def _orient_from_left(mask: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return mask
    if side == "right":
        return np.fliplr(mask)
    if side == "top":
        return np.rot90(mask, 3)
    if side == "bottom":
        return np.rot90(mask, 1)
    raise AssertionError(f"unknown side: {side}")


def _validate_generated(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if not np.all(np.isin(g, (BLACK, RED))):
        return False

    mask = g == RED
    area = int(mask.sum())
    if not (MIN_AREA <= area <= MAX_AREA):
        return False
    if not _is_connected(mask):
        return False

    side = _touch_side(mask)
    if side is None:
        return False
    if not _is_asymmetric(mask):
        return False
    if not _is_nonconvex(_crop_mask(mask)):
        return False

    _, _, input_rows, input_cols = _corridor_masks(g)

    mirrored = _mirror_shared(g, side)
    if max(mirrored.shape) > 30:
        return False
    hmask, vmask, row_count, col_count = _corridor_masks(mirrored)
    if row_count < 1 or col_count < 1:
        return False
    if not np.any(hmask) or not np.any(vmask):
        return False
    if side in ("left", "right"):
        if input_rows != 0 or row_count <= input_rows:
            return False
    else:
        if input_cols != 0 or col_count <= input_cols:
            return False

    solved = main(g)
    expected = mirrored.copy()
    expected[hmask | vmask] = GREEN
    if not np.array_equal(solved, expected):
        return False
    if np.array_equal(solved, mirrored):
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the single border-touching red shape across its touched side, then fill every edge-to-edge empty corridor in the mirrored figure green.

    Input structure:
    - The grid is black except for one 4-connected red shape.
    - That red shape touches exactly one outer border of the grid and does not touch the other three borders.
    - The shape is interpreted in the standard ARC `(row, col)` coordinate system, and corridor checks are performed only after the mirror is complete.

    Rule:
    - Detect which border the red shape touches and reflect the whole grid across that border, sharing the seam row or seam column once, so the mirrored output has size `2W-1` by `H` or `W` by `2H-1`.
    - Keep all original and mirrored red pixels unchanged.
    - In the mirrored grid, for each row whose leftmost red pixel and rightmost red pixel have only black cells between them, recolor those interior black cells green.
    - Apply the same test to every column using its topmost and bottommost red pixels; recolor the qualifying interior black cells green, including row/column intersections.

    Key insight:
    - The red shape does not expose the target corridors until the border-based reflection creates a larger symmetric figure, so the mirror must happen before corridor detection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.all(np.isin(g, (BLACK, RED))), "expected a black grid with a single red shape"

    mask = g == RED
    assert np.any(mask), "expected red foreground pixels"

    side = _touch_side(mask)
    assert side is not None, "expected the shape to touch exactly one border"

    mirrored = _mirror_shared(g, side)
    hmask, vmask, _row_count, _col_count = _corridor_masks(mirrored)
    out = mirrored.copy()
    out[hmask | vmask] = GREEN
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        height = int(RNG.integers(MIN_SIDE, MAX_SIDE + 1))
        width = int(RNG.integers(MIN_SIDE, MAX_SIDE + 1))
        left_mask = _build_left_template(height, width)
        if left_mask is None:
            continue

        side = str(RNG.choice(SIDES))
        oriented_mask = _orient_from_left(left_mask, side)
        grid = np.zeros(oriented_mask.shape, dtype=int)
        grid[oriented_mask] = RED

        if _validate_generated(grid):
            return grid

    raise RuntimeError("failed to generate a valid border-mirror corridor sample")
