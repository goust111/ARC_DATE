from __future__ import annotations

import random
from typing import Iterable

import numpy as np


FRAME_COLOR = 5
BACKGROUND = 0
PATTERN_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SEEN_INPUTS: set[bytes] = set()
SEEN_OUTPUTS: set[bytes] = set()


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(sprite, np.ones((factor, factor), dtype=int))


def _find_largest_frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, int, int, int, int]] = []

    for r0 in range(h - 2):
        for r1 in range(r0 + 2, h):
            for c0 in range(w - 2):
                for c1 in range(c0 + 2, w):
                    if not (
                        np.all(g[r0, c0 : c1 + 1] == FRAME_COLOR)
                        and np.all(g[r1, c0 : c1 + 1] == FRAME_COLOR)
                        and np.all(g[r0 : r1 + 1, c0] == FRAME_COLOR)
                        and np.all(g[r0 : r1 + 1, c1] == FRAME_COLOR)
                    ):
                        continue
                    if np.any(g[r0 + 1 : r1, c0 + 1 : c1] == FRAME_COLOR):
                        continue
                    area = (r1 - r0 - 1) * (c1 - c0 - 1)
                    candidates.append((area, r0, c0, r1, c1))

    assert candidates, "expected at least one grey hollow rectangle"
    max_area = max(area for area, *_ in candidates)
    largest = [item for item in candidates if item[0] == max_area]
    assert len(largest) == 1, "expected a unique largest frame"
    _area, r0, c0, r1, c1 = largest[0]
    return int(r0), int(c0), int(r1), int(c1)


def _outside_isolated_pixels(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    r0, c0, r1, c1 = frame_bbox
    isolated: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            if int(g[r, c]) == BACKGROUND:
                continue
            if all(
                not (0 <= nr < h and 0 <= nc < w) or int(g[nr, nc]) == BACKGROUND
                for nr, nc in _neighbors4(r, c)
            ):
                isolated.append((int(r), int(c)))
    return isolated


def main(grid: np.ndarray) -> np.ndarray:
    """Extract the interior of the unique grey frame, then magnify it by the count of isolated outer pixels.

    Input structure:
    - Black background with one unique grey (5) one-cell-thick hollow rectangle.
    - The frame interior contains a small multi-color pattern that does not touch the frame border.
    - Outside the frame are several non-black pixels and optional small multi-pixel distractor blobs.

    Rule:
    - Find the unique largest grey hollow rectangle and crop its strict interior.
    - Count only the non-black pixels outside the entire frame rectangle whose 4-neighbors are all black or out of bounds.
    - Use that count as a uniform pixel-scaling factor for the cropped interior.

    Key insight:
    - The frame simultaneously separates the image to keep from the outer region that encodes the scale, so both cropping and counting depend on the same spatial partition.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    r0, c0, r1, c1 = _find_largest_frame_bbox(g)
    interior = g[r0 + 1 : r1, c0 + 1 : c1].copy()
    k = len(_outside_isolated_pixels(g, (r0, c0, r1, c1)))
    assert k > 0, "expected at least one isolated outer pixel"
    return _scale_sprite(interior, k)


def _pattern_checker(h: int, w: int, colors: list[int]) -> np.ndarray:
    arr = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            arr[r, c] = colors[(r + c) % 2]
    if len(colors) == 3:
        arr[h // 2, w // 2] = colors[2]
    return arr


def _pattern_stripes(h: int, w: int, colors: list[int]) -> np.ndarray:
    arr = np.zeros((h, w), dtype=int)
    horizontal = random.random() < 0.5
    if horizontal:
        for r in range(h):
            arr[r, :] = colors[(r * len(colors)) // h]
    else:
        for c in range(w):
            arr[:, c] = colors[(c * len(colors)) // w]
    if len(colors) == 3 and h >= 2 and w >= 2:
        arr[-1, 0] = colors[2]
    return arr


def _pattern_blocks(h: int, w: int, colors: list[int]) -> np.ndarray:
    arr = np.full((h, w), colors[0], dtype=int)
    if h >= 2:
        arr[h // 2 :, :] = colors[1]
    else:
        arr[:, w // 2 :] = colors[1]
    if len(colors) == 3:
        rr = slice(0, max(1, h // 2))
        cc = slice(max(0, w - max(1, w // 2)), w)
        arr[rr, cc] = colors[2]
    return arr


def _pattern_plus(h: int, w: int, colors: list[int]) -> np.ndarray:
    arr = np.full((h, w), colors[0], dtype=int)
    arr[h // 2, :] = colors[1]
    arr[:, w // 2] = colors[1]
    if len(colors) == 3:
        arr[h // 2, w // 2] = colors[2]
    return arr


def _make_interior(inner_h: int, inner_w: int) -> np.ndarray:
    inner = np.zeros((inner_h, inner_w), dtype=int)
    motif_h = random.randint(2, inner_h - 2)
    motif_w = random.randint(2, inner_w - 2)
    top = random.randint(1, inner_h - motif_h - 1)
    left = random.randint(1, inner_w - motif_w - 1)

    color_count = random.choice((2, 3))
    colors = random.sample(PATTERN_COLORS, color_count)
    mode = random.choice(("checker", "stripes", "blocks", "plus"))
    if mode == "checker":
        motif = _pattern_checker(motif_h, motif_w, colors)
    elif mode == "stripes":
        motif = _pattern_stripes(motif_h, motif_w, colors)
    elif mode == "blocks":
        motif = _pattern_blocks(motif_h, motif_w, colors)
    else:
        motif = _pattern_plus(motif_h, motif_w, colors)

    if random.random() < 0.45 and motif_h >= 3 and motif_w >= 3:
        holes = random.randint(1, min(2, motif_h * motif_w - 1))
        options = [(r, c) for r in range(motif_h) for c in range(motif_w) if (r, c) != (motif_h // 2, motif_w // 2)]
        for rr, cc in random.sample(options, holes):
            motif[rr, cc] = BACKGROUND

    nonzero_colors = sorted(set(int(v) for v in motif.flat if int(v) != BACKGROUND))
    if len(nonzero_colors) < 2:
        motif[0, 0] = colors[0]
        motif[-1, -1] = colors[1]
        nonzero_colors = sorted(set(int(v) for v in motif.flat if int(v) != BACKGROUND))
    assert 2 <= len(nonzero_colors) <= 3

    inner[top : top + motif_h, left : left + motif_w] = motif
    assert np.all(inner[0, :] == 0) and np.all(inner[-1, :] == 0)
    assert np.all(inner[:, 0] == 0) and np.all(inner[:, -1] == 0)
    assert np.count_nonzero(inner) >= 4
    return inner


def _reserve_halo(mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = mask.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                mask[rr, cc] = True


def _choose_singletons(
    grid: np.ndarray,
    blocked: np.ndarray,
    count: int,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    placed: list[tuple[int, int]] = []
    for _ in range(count):
        candidates = [(r, c) for r in range(h) for c in range(w) if not blocked[r, c]]
        random.shuffle(candidates)
        pos = None
        for r, c in candidates:
            if all(
                not (0 <= nr < h and 0 <= nc < w) or grid[nr, nc] == BACKGROUND
                for nr, nc in _neighbors4(r, c)
            ):
                pos = (r, c)
                break
        assert pos is not None, "failed to place isolated singleton"
        placed.append(pos)
        blocked[pos] = True
        _reserve_halo(blocked, [pos])
    return placed


def _distractor_shapes() -> list[list[tuple[int, int]]]:
    return [
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1)],
    ]


def _place_distractor(
    grid: np.ndarray,
    blocked: np.ndarray,
    shape: list[tuple[int, int]],
    color: int,
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    shape_h = max(r for r, _ in shape) + 1
    shape_w = max(c for _, c in shape) + 1
    anchors = [(r0, c0) for r0 in range(h - shape_h + 1) for c0 in range(w - shape_w + 1)]
    random.shuffle(anchors)
    for r0, c0 in anchors:
        cells = [(r0 + dr, c0 + dc) for dr, dc in shape]
        if any(blocked[r, c] for r, c in cells):
            continue
        if any(grid[r, c] != BACKGROUND for r, c in cells):
            continue
        for r, c in cells:
            grid[r, c] = color
        _reserve_halo(blocked, cells)
        return cells
    return None


def _sample_grid_shape(outer_h: int, outer_w: int) -> tuple[int, int, int, int]:
    extra_h = random.randint(4, 8)
    extra_w = random.randint(4, 8)
    h = outer_h + extra_h
    w = outer_w + extra_w
    top = random.randint(1, extra_h - 1)
    left = random.randint(1, extra_w - 1)
    return h, w, top, left


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR


def _expected_output(inner: np.ndarray, scale: int) -> np.ndarray:
    return _scale_sprite(inner, scale)


def _validate(grid: np.ndarray, bbox: tuple[int, int, int, int], inner: np.ndarray, scale: int) -> bool:
    try:
        found_bbox = _find_largest_frame_bbox(grid)
        if found_bbox != bbox:
            return False
        isolated = _outside_isolated_pixels(grid, bbox)
        if len(isolated) != scale:
            return False
        if not np.array_equal(main(grid), _expected_output(inner, scale)):
            return False
    except AssertionError:
        return False

    r0, c0, r1, c1 = bbox
    outside = grid.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = 0
    for r, c in np.argwhere(outside != 0):
        if (int(r), int(c)) not in isolated:
            if all(
                not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]) or int(grid[nr, nc]) == 0
                for nr, nc in _neighbors4(int(r), int(c))
            ):
                return False

    inner_colors = {int(v) for v in inner.flat if int(v) != 0}
    if not (2 <= len(inner_colors) <= 3):
        return False
    if scale not in (2, 3):
        return False
    if np.max(_expected_output(inner, scale).shape) > 30:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    inner_h = random.randint(4, 6)
    inner_w = random.randint(4, 6)
    scale = random.choice((2, 3))
    inner = _make_interior(inner_h, inner_w)

    outer_h = inner_h + 2
    outer_w = inner_w + 2
    grid_h, grid_w, top, left = _sample_grid_shape(outer_h, outer_w)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    bbox = (top, left, top + outer_h - 1, left + outer_w - 1)

    _paint_frame(grid, bbox)
    grid[top + 1 : top + 1 + inner_h, left + 1 : left + 1 + inner_w] = inner

    blocked = np.zeros_like(grid, dtype=bool)
    r0, c0, r1, c1 = bbox
    for rr in range(max(0, r0 - 1), min(grid_h, r1 + 2)):
        for cc in range(max(0, c0 - 1), min(grid_w, c1 + 2)):
            blocked[rr, cc] = True

    singleton_positions = _choose_singletons(grid, blocked, scale)
    inner_colors = sorted({int(v) for v in inner.flat if int(v) != 0})
    for idx, (r, c) in enumerate(singleton_positions):
        if inner_colors and random.random() < 0.55:
            color = random.choice(inner_colors)
        else:
            color = random.choice([v for v in PATTERN_COLORS if v not in inner_colors] or list(PATTERN_COLORS))
        grid[r, c] = color

    n_distractors = random.choice((0, 1, 1, 2, 2))
    distractor_pixels = 0
    if n_distractors == 0 and random.random() < 0.75:
        n_distractors = 1
    for _ in range(n_distractors):
        color_pool = [v for v in PATTERN_COLORS if v != FRAME_COLOR]
        color = random.choice(color_pool)
        shape = random.choice(_distractor_shapes())
        cells = _place_distractor(grid, blocked, shape, color)
        if cells is None:
            return None
        distractor_pixels += len(cells)

    if distractor_pixels == 0 and random.random() < 0.6:
        return None
    if distractor_pixels and distractor_pixels + scale == scale:
        return None
    if not _validate(grid, bbox, inner, scale):
        return None

    input_sig = grid.tobytes() + str(grid.shape).encode()
    output = _expected_output(inner, scale)
    output_sig = output.tobytes() + str(output.shape).encode()
    if input_sig in SEEN_INPUTS or output_sig in SEEN_OUTPUTS:
        return None
    SEEN_INPUTS.add(input_sig)
    SEEN_OUTPUTS.add(output_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 4")
