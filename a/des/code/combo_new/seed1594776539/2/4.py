from __future__ import annotations

import numpy as np


BACKGROUND = 0
BORDER = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 400


def _border_values(g: np.ndarray) -> np.ndarray:
    return np.concatenate((g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]))


def _majority_color(values: np.ndarray) -> int:
    colors, counts = np.unique(values, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _find_colored_run(line: np.ndarray, frame_color: int) -> tuple[int, int]:
    runs: list[tuple[int, int]] = []
    i = 1
    while i < len(line) - 1:
        value = int(line[i])
        if value in (BACKGROUND, frame_color):
            i += 1
            continue
        j = i
        while j + 1 < len(line) - 1 and int(line[j + 1]) == value:
            j += 1
        runs.append((int(i), int(j)))
        i = j + 1
    assert len(runs) == 1
    start, end = runs[0]
    assert end - start + 1 >= 2
    return start, end


def _window_bounds(g: np.ndarray) -> tuple[int, int, int, int]:
    frame_color = _majority_color(_border_values(g))
    c1, c2 = _find_colored_run(g[0, :], frame_color)
    r1, r2 = _find_colored_run(g[:, 0], frame_color)
    return r1, r2, c1, c2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the border-selected window and repair the missing pixels by left-right mirroring.

    Input structure:
    - The background is black.
    - The outermost border is mostly one frame color, with one non-black colored run on the top edge and one on the left edge.
    - The top run marks a contiguous column interval and the left run marks a contiguous row interval.
    - Inside the marked window there is one main foreground pattern on black, almost perfectly symmetric across the window's vertical midline.

    Rule:
    - Read the top-edge colored run to get the window columns, and the left-edge colored run to get the window rows.
    - Crop that rectangular window from the input.
    - For each pair of horizontally mirrored cells inside the crop, if one side is foreground and the other side is black, copy the foreground color into the missing side.

    Key insight:
    The border markings are coordinate selectors, and the object inside the selected crop is repaired by enforcing its intended left-right mirror symmetry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert min(g.shape) >= 5

    r1, r2, c1, c2 = _window_bounds(g)
    out = g[r1 : r2 + 1, c1 : c2 + 1].copy()
    h, w = out.shape

    for c in range(w // 2):
        mc = w - 1 - c
        left = out[:, c].copy()
        right = out[:, mc].copy()

        fill_left = (left == BACKGROUND) & (right != BACKGROUND)
        fill_right = (right == BACKGROUND) & (left != BACKGROUND)
        out[fill_left, c] = right[fill_left]
        out[fill_right, mc] = left[fill_right]

    return out


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(int(r), int(c))]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
    return count


def _orbit_cells(r: int, c: int, w: int) -> tuple[tuple[int, int], ...]:
    mc = w - 1 - c
    if mc == c:
        return ((int(r), int(c)),)
    c0, c1 = sorted((int(c), int(mc)))
    return ((int(r), int(c0)), (int(r), int(c1)))


def _canonical_orbit(r: int, c: int, w: int) -> tuple[int, int]:
    return int(r), int(min(c, w - 1 - c))


def _grow_symmetric_mask(height: int, width: int) -> np.ndarray | None:
    min_area = max(8, int(np.ceil(height * width * 0.30)))
    max_area = max(min_area + 2, int(np.floor(height * width * 0.65)))

    for _ in range(200):
        mask = np.zeros((height, width), dtype=bool)
        orbit_set: set[tuple[int, int]] = set()

        seed_r = int(RNG.integers(height))
        seed_c = width // 2 if width % 2 else width // 2 - 1
        seed = (seed_r, seed_c)
        orbit_set.add(seed)
        for rr, cc in _orbit_cells(seed_r, seed_c, width):
            mask[rr, cc] = True

        while True:
            area = int(mask.sum())
            if area >= min_area and (area >= max_area or (area > min_area and RNG.random() < 0.25)):
                break

            frontier: set[tuple[int, int]] = set()
            for rr, cc in np.argwhere(mask):
                for nr, nc in _neighbors4(int(rr), int(cc), height, width):
                    orbit = _canonical_orbit(int(nr), int(nc), width)
                    if orbit not in orbit_set:
                        frontier.add(orbit)

            if not frontier:
                break

            options = list(frontier)
            RNG.shuffle(options)
            chosen = None
            for nr, nc in options:
                orbit_size = len(_orbit_cells(nr, nc, width))
                if area + orbit_size <= max_area or area < min_area:
                    chosen = (nr, nc)
                    break
            if chosen is None:
                break

            orbit_set.add(chosen)
            for rr, cc in _orbit_cells(chosen[0], chosen[1], width):
                mask[rr, cc] = True

        if int(mask.sum()) < min_area:
            continue

        pair_orbits = [
            (r, c)
            for r in range(height)
            for c in range(width // 2)
            if mask[r, c] and mask[r, width - 1 - c]
        ]
        if len(pair_orbits) < 3:
            continue
        return mask

    return None


def _paint_sprite(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=int)
    n_colors = int(RNG.integers(1, 4))
    colors = np.asarray(RNG.choice(PALETTE, size=n_colors, replace=False), dtype=int)
    style = int(RNG.integers(3))
    phase = int(RNG.integers(n_colors))

    for r in range(h):
        for c in range((w + 1) // 2):
            orbit = _orbit_cells(r, c, w)
            if not all(mask[rr, cc] for rr, cc in orbit):
                continue
            key0 = r + c
            key1 = r // 2 + 2 * c
            key2 = 2 * r + c // 2
            key = (key0, key1, key2)[style]
            color = int(colors[(key + phase) % n_colors])
            for rr, cc in orbit:
                out[rr, cc] = color
    return out


def _corrupt_sprite(clean_sprite: np.ndarray, missing_count: int) -> np.ndarray | None:
    h, w = clean_sprite.shape
    pair_cells = [
        (r, c, w - 1 - c)
        for r in range(h)
        for c in range(w // 2)
        if clean_sprite[r, c] != BACKGROUND and clean_sprite[r, w - 1 - c] != BACKGROUND
    ]
    if len(pair_cells) < missing_count:
        return None

    for _ in range(120):
        broken = clean_sprite.copy()
        chosen = RNG.choice(len(pair_cells), size=missing_count, replace=False)
        for idx in chosen:
            r, c0, c1 = pair_cells[int(idx)]
            if bool(RNG.integers(2)):
                broken[r, c0] = BACKGROUND
            else:
                broken[r, c1] = BACKGROUND
        if _component_count(broken != BACKGROUND) == 1:
            return broken
    return None


def _build_window() -> tuple[np.ndarray, np.ndarray] | None:
    want_odd = bool(RNG.integers(2))
    width_options = [w for w in range(7, 16) if (w % 2 == 1) == want_odd]
    window_w = int(RNG.choice(width_options))
    window_h = int(RNG.integers(6, 15))

    max_side_margin = min(2, (window_w - 5) // 2)
    side_margin = int(RNG.integers(max_side_margin + 1))
    obj_w = window_w - 2 * side_margin

    for _ in range(160):
        top_margin = int(RNG.integers(0, min(3, window_h - 4)))
        bottom_margin = int(RNG.integers(0, min(3, window_h - top_margin - 4)))
        obj_h = window_h - top_margin - bottom_margin
        if obj_h < 5:
            continue

        mask = _grow_symmetric_mask(obj_h, obj_w)
        if mask is None or _component_count(mask) != 1:
            continue

        clean_sprite = _paint_sprite(mask)
        missing_count = int(RNG.integers(1, 4))
        broken_sprite = _corrupt_sprite(clean_sprite, missing_count)
        if broken_sprite is None:
            continue

        clean_window = np.zeros((window_h, window_w), dtype=int)
        broken_window = np.zeros((window_h, window_w), dtype=int)
        clean_window[top_margin : top_margin + obj_h, side_margin : side_margin + obj_w] = clean_sprite
        broken_window[top_margin : top_margin + obj_h, side_margin : side_margin + obj_w] = broken_sprite

        if _component_count(clean_window != BACKGROUND) != 1:
            continue
        if _component_count(broken_window != BACKGROUND) != 1:
            continue
        if np.array_equal(clean_window, broken_window):
            continue
        return clean_window, broken_window

    return None


def _place_distractors(grid: np.ndarray, blocked: np.ndarray) -> None:
    shapes = [
        np.array([[1]], dtype=bool),
        np.array([[1, 1]], dtype=bool),
        np.array([[1], [1]], dtype=bool),
        np.array([[1, 1], [1, 0]], dtype=bool),
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1], [0, 1]], dtype=bool),
        np.array([[1, 1], [1, 1]], dtype=bool),
    ]

    inner_h = grid.shape[0] - 2
    inner_w = grid.shape[1] - 2
    n_objects = int(RNG.integers(0, 5))

    for _ in range(n_objects):
        shape = shapes[int(RNG.integers(len(shapes)))]
        sh, sw = shape.shape
        if sh > inner_h or sw > inner_w:
            continue

        color = int(RNG.choice(PALETTE))
        for _ in range(60):
            r0 = int(RNG.integers(1, grid.shape[0] - sh))
            c0 = int(RNG.integers(1, grid.shape[1] - sw))
            region = blocked[r0 : r0 + sh, c0 : c0 + sw]
            if bool(np.any(region & shape)):
                continue
            if bool(np.any((grid[r0 : r0 + sh, c0 : c0 + sw] != BACKGROUND) & shape)):
                continue
            grid[r0 : r0 + sh, c0 : c0 + sw][shape] = color
            blocked[r0 : r0 + sh, c0 : c0 + sw] |= shape
            break


def _gen_attempt() -> np.ndarray | None:
    window = _build_window()
    if window is None:
        return None
    clean_window, broken_window = window
    window_h, window_w = clean_window.shape

    grid_h = int(RNG.integers(window_h + 3, min(31, window_h + 11)))
    grid_w = int(RNG.integers(window_w + 3, min(31, window_w + 11)))
    if grid_h > 30 or grid_w > 30:
        return None

    r1 = int(RNG.integers(1, grid_h - window_h))
    c1 = int(RNG.integers(1, grid_w - window_w))
    r2 = r1 + window_h - 1
    c2 = c1 + window_w - 1

    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)
    grid[0, :] = BORDER
    grid[-1, :] = BORDER
    grid[:, 0] = BORDER
    grid[:, -1] = BORDER

    top_color = int(RNG.choice(PALETTE))
    if bool(RNG.integers(2)):
        left_color = top_color
    else:
        left_color = int(RNG.choice(PALETTE[PALETTE != top_color]))
    grid[0, c1 : c2 + 1] = top_color
    grid[r1 : r2 + 1, 0] = left_color
    grid[r1 : r2 + 1, c1 : c2 + 1] = broken_window

    blocked = np.zeros_like(grid, dtype=bool)
    blocked[max(1, r1 - 1) : min(grid_h - 1, r2 + 2), max(1, c1 - 1) : min(grid_w - 1, c2 + 2)] = True
    _place_distractors(grid, blocked)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, clean_window):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
