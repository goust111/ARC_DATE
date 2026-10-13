from __future__ import annotations

import random

import numpy as np


BG = 0
ARROW_COLOR = 4
STRIPE_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
ARC_MAX = 30
MAX_GENERATION_ATTEMPTS = 400

RIGHT_ARROW_MASK = np.array(
    [
        [0, 1],
        [1, 0],
        [0, 1],
    ],
    dtype=int,
)
LEFT_ARROW_MASK = np.array(
    [
        [1, 0],
        [0, 1],
        [1, 0],
    ],
    dtype=int,
)


def _neighbors8(r: int, c: int, h: int, w: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                yield rr, cc


def _components8(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for rr, cc2 in _neighbors8(cr, cc, h, w):
                    if mask[rr, cc2] and not seen[rr, cc2]:
                        seen[rr, cc2] = True
                        stack.append((rr, cc2))
            comps.append(np.array(cells, dtype=int))
    return comps


def _find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _find_arrows(grid: np.ndarray) -> list[dict[str, object]]:
    arrows: list[dict[str, object]] = []
    for coords in _components8(grid == ARROW_COLOR):
        if coords.shape[0] != 3:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if (r1 - r0 + 1, c1 - c0 + 1) != (3, 2):
            continue
        patch = np.zeros((3, 2), dtype=int)
        for r, c in coords:
            patch[r - r0, c - c0] = 1
        if np.array_equal(patch, RIGHT_ARROW_MASK):
            direction = 1
        elif np.array_equal(patch, LEFT_ARROW_MASK):
            direction = -1
        else:
            continue
        arrows.append(
            {
                "coords": coords,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "row": int(r0 + 1),
                "direction": int(direction),
                "shift": int(coords.shape[0]),
            }
        )
    arrows.sort(key=lambda item: (item["row"], item["bbox"][1]))  # type: ignore[index]
    return arrows


def _find_stripe_run(grid: np.ndarray, arrow: dict[str, object]) -> tuple[int, int]:
    row = int(arrow["row"])
    _r0, c0, _r1, c1 = arrow["bbox"]  # type: ignore[misc]
    direction = int(arrow["direction"])
    row_mask = grid[row] != BG
    for rr, cc in arrow["coords"]:  # type: ignore[index]
        if int(rr) == row:
            row_mask[int(cc)] = False
    runs = _find_runs(row_mask)
    if direction > 0:
        candidates = [(s, e) for s, e in runs if s >= int(c1) + 2]
        assert candidates, "missing stripe to the right of a right-pointing arrow"
        return min(candidates, key=lambda item: item[0])
    candidates = [(s, e) for s, e in runs if e <= int(c0) - 1]
    assert candidates, "missing stripe to the left of a left-pointing arrow"
    return max(candidates, key=lambda item: item[1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each striped row cyclically according to the direction and size of its yellow arrow marker.

    Input structure:
    - Background is black.
    - Each active row contains one contiguous multicolor horizontal stripe segment.
    - The same logical row also has a 3-pixel yellow arrow on its left or right, separated from the stripe by one black column.
    - A `>` arrow points right and a `<` arrow points left; each arrow spans three grid rows around the stripe row.

    Rule:
    - Detect every 3-pixel yellow arrow and read its orientation.
    - Use the arrow's middle row to locate the stripe segment associated with that arrow.
    - Treat the stripe as a circular 1D sequence and rotate it by the arrow's pixel count, which is 3.
    - Rotate right for `>` arrows, rotate left for `<` arrows, and erase the arrow markers to black.

    Key insight:
    - The arrow encodes two parameters at once: orientation gives the rotation direction, and arrow area gives the rotation amount.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    arrows = _find_arrows(g)
    assert arrows, "expected at least one valid arrow marker"

    for arrow in arrows:
        for r, c in arrow["coords"]:  # type: ignore[index]
            out[int(r), int(c)] = BG

    for arrow in arrows:
        row = int(arrow["row"])
        start, end = _find_stripe_run(g, arrow)
        segment = g[row, start:end].copy()
        shift = int(arrow["shift"])
        if int(arrow["direction"]) > 0:
            rotated = np.roll(segment, shift)
        else:
            rotated = np.roll(segment, -shift)
        out[row, start:end] = rotated

    return out


def _balanced_directions(count: int) -> list[int]:
    directions = [1, -1] * (count // 2)
    if count % 2:
        directions.append(random.choice([1, -1]))
    random.shuffle(directions)
    return directions


def _sample_centers(count: int) -> tuple[list[int], int] | None:
    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    centers = [top_margin]
    for _ in range(count - 1):
        centers.append(centers[-1] + random.randint(4, 5))
    height = centers[-1] + 1 + bottom_margin + 1
    if height > ARC_MAX:
        return None
    return centers, height


def _make_stripe(length: int) -> np.ndarray | None:
    for _ in range(100):
        run_count = random.randint(2, min(4, length))
        cuts = sorted(random.sample(range(1, length), run_count - 1))
        bounds = [0] + cuts + [length]
        run_lengths = [bounds[i + 1] - bounds[i] for i in range(run_count)]
        colors = random.sample(STRIPE_COLORS, run_count)
        parts = [np.full(run_len, color, dtype=int) for run_len, color in zip(run_lengths, colors)]
        stripe = np.concatenate(parts)
        if np.array_equal(stripe, np.roll(stripe, 3)):
            continue
        return stripe
    return None


def _draw_right_arrow(grid: np.ndarray, row: int, left_col: int) -> None:
    grid[row, left_col] = ARROW_COLOR
    grid[row - 1, left_col + 1] = ARROW_COLOR
    grid[row + 1, left_col + 1] = ARROW_COLOR


def _draw_left_arrow(grid: np.ndarray, row: int, left_col: int) -> None:
    grid[row - 1, left_col] = ARROW_COLOR
    grid[row, left_col + 1] = ARROW_COLOR
    grid[row + 1, left_col] = ARROW_COLOR


def _gen_attempt() -> np.ndarray | None:
    row_count = random.randint(2, 6)
    centers_and_height = _sample_centers(row_count)
    if centers_and_height is None:
        return None
    centers, height = centers_and_height

    width = random.randint(12, ARC_MAX)
    max_len = min(18, width - 3)
    if max_len < 5:
        return None

    grid = np.full((height, width), BG, dtype=int)
    directions = _balanced_directions(row_count)
    changed_rows = 0

    for row, direction in zip(centers, directions):
        length = random.randint(5, max_len)
        stripe = _make_stripe(length)
        if stripe is None:
            return None

        if direction > 0:
            max_start = width - length
            if max_start < 3:
                return None
            start = random.randint(3, max_start)
            arrow_left = start - 3
            _draw_right_arrow(grid, row, arrow_left)
        else:
            max_start = width - length - 3
            if max_start < 0:
                return None
            start = random.randint(0, max_start)
            arrow_left = start + length + 1
            _draw_left_arrow(grid, row, arrow_left)

        grid[row, start : start + length] = stripe
        if not np.array_equal(stripe, np.roll(stripe, 3 * direction)):
            changed_rows += 1

    arrows = _find_arrows(grid)
    if len(arrows) != row_count:
        return None
    if {int(arrow["direction"]) for arrow in arrows} != {1, -1}:
        return None
    if changed_rows != row_count:
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    for arrow in arrows:
        for r, c in arrow["coords"]:  # type: ignore[index]
            if out[int(r), int(c)] != BG:
                return None
        row = int(arrow["row"])
        start, end = _find_stripe_run(grid, arrow)
        if np.array_equal(grid[row, start:end], out[row, start:end]):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
