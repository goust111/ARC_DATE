from __future__ import annotations

import numpy as np


BACKGROUND = 0
MAX_GRID_SIDE = 30
GENERATION_ATTEMPTS = 300


def _read_run(values: np.ndarray) -> list[int]:
    run: list[int] = []
    for value in values:
        color = int(value)
        if color == BACKGROUND:
            break
        run.append(color)
    assert run, "expected a non-black ruler run"
    if len(run) >= 2:
        assert run[0] != run[1], "ruler must alternate between two visible colors"
    for i, color in enumerate(run):
        assert color == run[i % 2], "ruler colors must alternate regularly"
    return run


def _edge_rulers(grid: np.ndarray) -> tuple[list[int], list[int]]:
    """Return horizontal and vertical ruler runs."""
    if int(grid[0, 0]) == BACKGROUND:
        horizontal = _read_run(grid[0, 1:])
        vertical = _read_run(grid[1:, 0])
    else:
        horizontal = _read_run(grid[0, :])
        vertical = _read_run(grid[:, 0])
    return horizontal, vertical


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the interior window encoded by edge rulers, then flip the crop according to ruler parity.

    Input structure:
    - The background is black.
    - The top edge has one continuous alternating-color ruler, and the left edge has one continuous alternating-color ruler.
    - The rulers use colors not used by the non-black pattern objects in the interior.
    - The interior area begins one row and one column in from the outer edge and contains one or more colored objects.

    Rule:
    - The horizontal ruler length is the crop width W; the vertical ruler length is the crop height H.
    - Crop the H x W interior window whose upper-left cell is at row 1, column 1.
    - If the horizontal ruler's last color matches its first color, flip the crop left-right.
    - If the vertical ruler's last color matches its first color, flip the crop top-bottom.

    Key insight: Each ruler encodes two facts at once: its length gives the crop size, while its alternating-color parity gives the independent flip instruction for that axis.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    rows, cols = map(int, g.shape)
    assert rows >= 2 and cols >= 2, "grid must contain an edge and an interior"

    horizontal, vertical = _edge_rulers(g)
    width = int(len(horizontal))
    height = int(len(vertical))
    assert height + 1 <= rows and width + 1 <= cols, "crop must fit inside the grid interior"

    out = g[1 : 1 + height, 1 : 1 + width].copy()
    if int(horizontal[0]) == int(horizontal[-1]):
        out = np.fliplr(out)
    if int(vertical[0]) == int(vertical[-1]):
        out = np.flipud(out)
    return out


# --------------- generator ---------------

def _length_with_parity(rng: np.random.Generator, want_same_ends: bool, lo: int, hi: int) -> int:
    parity = 1 if bool(want_same_ends) else 0
    choices = [n for n in range(int(lo), int(hi) + 1) if n % 2 == parity]
    assert choices
    return int(choices[int(rng.integers(0, len(choices)))])


def _ruler(length: int, first: int, second: int) -> np.ndarray:
    return np.array([first if i % 2 == 0 else second for i in range(int(length))], dtype=int)


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _connected_mask(rng: np.random.Generator, h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((int(h), int(w)), dtype=bool)
    r = int(rng.integers(0, h))
    c = int(rng.integers(0, w))
    mask[r, c] = True

    while int(mask.sum()) < int(area):
        frontier: set[tuple[int, int]] = set()
        for rr, cc in np.argwhere(mask):
            for nr, nc in _neighbors4(int(rr), int(cc), int(h), int(w)):
                if not bool(mask[nr, nc]):
                    frontier.add((int(nr), int(nc)))
        assert frontier
        cells = tuple(frontier)
        nr, nc = cells[int(rng.integers(0, len(cells)))]
        mask[nr, nc] = True
    return mask


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _paint_random_objects(
    rng: np.random.Generator,
    canvas: np.ndarray,
    colors: list[int],
    min_objects: int,
    max_objects: int,
) -> bool:
    h, w = map(int, canvas.shape)
    occupied = canvas != BACKGROUND
    target_count = int(rng.integers(int(min_objects), int(max_objects) + 1))
    placed = 0

    for _ in range(target_count):
        for _attempt in range(80):
            mh = int(rng.integers(2, min(6, h) + 1))
            mw = int(rng.integers(2, min(6, w) + 1))
            max_area = min(mh * mw, 10)
            min_area = min(3, max_area)
            area = int(rng.integers(min_area, max_area + 1))
            mask = _trim_mask(_connected_mask(rng, mh, mw, area))
            mh, mw = map(int, mask.shape)
            if mh > h or mw > w:
                continue

            blocked = _dilate8(occupied)
            positions = []
            for r0 in range(h - mh + 1):
                for c0 in range(w - mw + 1):
                    if not bool((blocked[r0 : r0 + mh, c0 : c0 + mw] & mask).any()):
                        positions.append((r0, c0))
            if not positions:
                continue

            r0, c0 = positions[int(rng.integers(0, len(positions)))]
            patch = np.zeros((mh, mw), dtype=int)
            base = int(colors[int(rng.integers(0, len(colors)))])
            patch[mask] = base

            if len(colors) >= 2 and int(mask.sum()) >= 4 and rng.random() < 0.45:
                accent_choices = [c for c in colors if int(c) != base]
                accent = int(accent_choices[int(rng.integers(0, len(accent_choices)))])
                pts = np.argwhere(mask)
                accent_count = int(rng.integers(1, max(2, len(pts) // 2) + 1))
                picks = rng.choice(len(pts), size=accent_count, replace=False)
                for idx in picks:
                    rr, cc = pts[int(idx)]
                    patch[int(rr), int(cc)] = accent

            region = canvas[r0 : r0 + mh, c0 : c0 + mw]
            region[mask] = patch[mask]
            occupied[r0 : r0 + mh, c0 : c0 + mw] |= mask
            placed += 1
            break

    return int(placed) >= int(min_objects)


def _add_outside_noise(rng: np.random.Generator, grid: np.ndarray, crop_h: int, crop_w: int, colors: list[int]) -> None:
    rows, cols = map(int, grid.shape)
    outside = []
    for r in range(1, rows):
        for c in range(1, cols):
            if 1 <= r <= crop_h and 1 <= c <= crop_w:
                continue
            outside.append((r, c))
    if not outside or rng.random() >= 0.55:
        return

    count = int(rng.integers(1, min(6, len(outside)) + 1))
    picks = rng.choice(len(outside), size=count, replace=False)
    for idx in picks:
        r, c = outside[int(idx)]
        grid[int(r), int(c)] = int(colors[int(rng.integers(0, len(colors)))])


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    horizontal_flip = bool(rng.integers(0, 2))
    vertical_flip = bool(rng.integers(0, 2))
    crop_w = _length_with_parity(rng, horizontal_flip, 5, 15)
    crop_h = _length_with_parity(rng, vertical_flip, 5, 15)

    extra_bottom = int(rng.integers(0, min(7, MAX_GRID_SIDE - crop_h - 1) + 1))
    extra_right = int(rng.integers(0, min(7, MAX_GRID_SIDE - crop_w - 1) + 1))
    rows = 1 + crop_h + extra_bottom
    cols = 1 + crop_w + extra_right

    palette = list(map(int, rng.choice(np.arange(1, 10), size=4, replace=False)))
    h_first, h_second, v_first, v_second = palette
    object_colors = [c for c in range(1, 10) if c not in palette]
    rng.shuffle(object_colors)

    window = np.zeros((crop_h, crop_w), dtype=int)
    max_objects = min(4, max(1, (crop_h * crop_w) // 18))
    if not _paint_random_objects(rng, window, object_colors, 1, max_objects):
        return None

    # Avoid examples where a requested flip is visually invisible.
    if np.array_equal(window, np.fliplr(window)) or np.array_equal(window, np.flipud(window)):
        return None

    grid = np.zeros((rows, cols), dtype=int)
    grid[0, 1 : 1 + crop_w] = _ruler(crop_w, h_first, h_second)
    grid[1 : 1 + crop_h, 0] = _ruler(crop_h, v_first, v_second)
    grid[1 : 1 + crop_h, 1 : 1 + crop_w] = window
    _add_outside_noise(rng, grid, crop_h, crop_w, object_colors)

    try:
        expected = window.copy()
        if horizontal_flip:
            expected = np.fliplr(expected)
        if vertical_flip:
            expected = np.flipud(expected)
        if not np.array_equal(main(grid), expected):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid edge-ruler crop/flip sample")
