from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BACKGROUND = 0
GAP_COLOR = 5
FRAME_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_size(mask: np.ndarray) -> int:
    box = _bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    return max(int(r1 - r0 + 1), int(c1 - c0 + 1))


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _count_components(mask: np.ndarray, connectivity: int = 4) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    steps = _neighbors4 if connectivity == 4 else None
    assert steps is not None
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for nr, nc in steps(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _ring_coords(size: int) -> list[tuple[int, int]]:
    if size == 1:
        return [(0, 0)]
    coords: list[tuple[int, int]] = []
    hi = size - 1
    for c in range(size):
        coords.append((0, c))
        coords.append((hi, c))
    for r in range(1, hi):
        coords.append((r, 0))
        coords.append((r, hi))
    return coords


def _draw_centered_ring(canvas: np.ndarray, size: int, color: int) -> None:
    n = canvas.shape[0]
    inset = (n - size) // 2
    hi = inset + size - 1
    canvas[inset, inset : hi + 1] = color
    canvas[hi, inset : hi + 1] = color
    canvas[inset : hi + 1, inset] = color
    canvas[inset : hi + 1, hi] = color


def _expected_output(outer_size: int, gap: int, colors_in_order: list[int]) -> np.ndarray:
    out = np.full((outer_size, outer_size), BACKGROUND, dtype=int)
    for layer, color in enumerate(colors_in_order):
        size = outer_size - 2 * layer * gap
        if size <= 0:
            continue
        _draw_centered_ring(out, size, int(color))
    return out


def _rebuild_using_observed_sizes(sorted_layers: list[tuple[int, int]]) -> np.ndarray:
    out = np.full((sorted_layers[0][0], sorted_layers[0][0]), BACKGROUND, dtype=int)
    for size, color in sorted_layers:
        _draw_centered_ring(out, int(size), int(color))
    return out


def _compact_object_area(grid: np.ndarray, bg: int) -> int:
    if GAP_COLOR != bg and np.any(grid == GAP_COLOR):
        mask = grid == GAP_COLOR
        count = int(mask.sum())
        assert 1 <= count <= 4, "gap object must have 1-4 pixels"
        assert _count_components(mask, connectivity=4) == 1, "gap object must be 4-connected"
        return count

    foreground = [int(c) for c in np.unique(grid) if int(c) != int(bg)]
    candidates = []
    for color in foreground:
        mask = grid == color
        area = int(mask.sum())
        if 1 <= area <= 4 and _count_components(mask, connectivity=4) == 1:
            candidates.append((area, color))
    assert len(candidates) == 1, "expected a unique compact gap object"
    return int(candidates[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort fragmented frame colors by inferred side length, then rebuild centered rings whose spacing is encoded by a compact grey object.

    Input structure:
    - Black background with 3-5 differently colored square-frame fragments.
    - Each fragment color shows only part of one 1-pixel-wide square frame; its intended full side length is the larger dimension of that color's visible bbox.
    - A separate compact grey object is present, uses a color not used by the frames, and does not overlap them.

    Rule:
    - For each frame color, compute its visible bbox and take the larger bbox dimension as its intended side length.
    - Sort the frame colors by that side length from largest to smallest.
    - Count the grey object's pixels to get the gap width G.
    - Create a square output whose side equals the largest inferred frame side, then draw centered square rings in the sorted color order.
    - The outer ring uses the largest side; the k-th ring side is largest_side - 2*k*G.

    Key insight:
    - The frame fragments only determine the nesting order, while the compact grey object's area independently determines the spacing between rings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    gap = _compact_object_area(g, bg)

    frame_colors = [
        int(color)
        for color in np.unique(g)
        if int(color) not in (int(bg), int(GAP_COLOR))
    ]
    assert 3 <= len(frame_colors) <= 5, "expected 3-5 frame colors"

    layers: list[tuple[int, int]] = []
    for color in frame_colors:
        mask = g == color
        size = _bbox_size(mask)
        layers.append((size, int(color)))

    sizes = [size for size, _color in layers]
    assert len(set(sizes)) == len(sizes), "frame side lengths must be unique"

    layers.sort(reverse=True)
    largest = layers[0][0]
    out = np.full((largest, largest), bg, dtype=int)
    for layer, (_observed_size, color) in enumerate(layers):
        size = largest - 2 * layer * gap
        if size <= 0:
            continue
        _draw_centered_ring(out, size, int(color))
    return out


def _random_polyomino(area: int) -> np.ndarray:
    templates: dict[int, list[list[tuple[int, int]]]] = {
        1: [[(0, 0)]],
        2: [[(0, 0), (0, 1)], [(0, 0), (1, 0)]],
        3: [
            [(0, 0), (0, 1), (0, 2)],
            [(0, 0), (1, 0), (2, 0)],
            [(0, 0), (1, 0), (1, 1)],
            [(0, 1), (1, 0), (1, 1)],
        ],
        4: [
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [(0, 0), (0, 1), (0, 2), (0, 3)],
            [(0, 0), (1, 0), (2, 0), (3, 0)],
            [(0, 0), (1, 0), (2, 0), (2, 1)],
            [(0, 1), (1, 1), (2, 1), (2, 0)],
            [(0, 0), (0, 1), (0, 2), (1, 1)],
        ],
    }
    cells = random.choice(templates[int(area)])
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    norm = [(r - r0, c - c0) for r, c in cells]
    h = max(r for r, _c in norm) + 1
    w = max(c for _r, c in norm) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c in norm:
        patch[r, c] = GAP_COLOR
    if random.random() < 0.5:
        patch = np.flipud(patch)
    if random.random() < 0.5:
        patch = np.fliplr(patch)
    if random.random() < 0.5:
        patch = patch.T
    return patch


def _sample_case_parameters() -> tuple[int, int, int, list[int], list[int], np.ndarray]:
    feasible: list[tuple[int, int]] = []
    for n_frames in range(3, 6):
        for gap in range(1, 5):
            min_outer = 2 * (n_frames - 1) * gap + 1
            if min_outer <= 22:
                feasible.append((n_frames, gap))
    n_frames, gap = random.choice(feasible)
    min_outer = max(7, 2 * (n_frames - 1) * gap + 1)
    outer_size = random.randint(min_outer, min(22, min_outer + 7))

    formula_sizes = [outer_size - 2 * k * gap for k in range(n_frames)]
    inner_pool = list(range(3, outer_size))
    for _ in range(100):
        inner_sizes = sorted(random.sample(inner_pool, n_frames - 1), reverse=True)
        if any(inner_sizes[idx - 1] != formula_sizes[idx] for idx in range(1, n_frames)):
            break
    else:
        raise RuntimeError("failed to sample non-degenerate frame sizes")

    observed_sizes = [outer_size] + inner_sizes
    colors = random.sample(FRAME_COLORS, n_frames)
    output = _expected_output(outer_size, gap, colors)
    return n_frames, gap, outer_size, observed_sizes, colors, output


def _placement_for_size(side: int, size: int) -> tuple[int, int]:
    mode = random.choice(("top", "bottom", "left", "right"))
    clip = random.randint(1, size - 2)
    if mode == "top":
        return -clip, random.randint(0, side - size)
    if mode == "bottom":
        return side - size + clip, random.randint(0, side - size)
    if mode == "left":
        return random.randint(0, side - size), -clip
    return random.randint(0, side - size), side - size + clip


def _render_frames(side: int, sizes: list[int], colors: list[int]) -> np.ndarray:
    grid = np.full((side, side), BACKGROUND, dtype=int)
    placements = [_placement_for_size(side, size) for size in sizes]
    draw_order = random.sample(range(len(sizes)), len(sizes))
    ring_cache = {size: _ring_coords(size) for size in sizes}
    for idx in draw_order:
        r0, c0 = placements[idx]
        color = colors[idx]
        for dr, dc in ring_cache[sizes[idx]]:
            rr = r0 + dr
            cc = c0 + dc
            if 0 <= rr < side and 0 <= cc < side:
                grid[rr, cc] = color
    return grid


def _place_gap_object(grid: np.ndarray, area: int) -> bool:
    patch = _random_polyomino(area)
    h, w = patch.shape
    gh, gw = grid.shape
    if h + 2 > gh or w + 2 > gw:
        return False

    positions = [
        (r, c)
        for r in range(1, gh - h)
        for c in range(1, gw - w)
    ]
    random.shuffle(positions)
    for r0, c0 in positions:
        region = grid[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1]
        if np.any(region != BACKGROUND):
            continue
        mask = patch == GAP_COLOR
        grid[r0 : r0 + h, c0 : c0 + w][mask] = GAP_COLOR
        return True
    return False


def _all_pixels_on_bbox_border(mask: np.ndarray) -> bool:
    box = _bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if rr not in (r0, r1) and cc not in (c0, c1):
            return False
    return True


def _semantic_validation(
    inp: np.ndarray,
    expected: np.ndarray,
    observed_sizes: list[int],
    colors: list[int],
    gap: int,
) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    if int((inp == GAP_COLOR).sum()) != gap:
        return False
    if _count_components(inp == GAP_COLOR, connectivity=4) != 1:
        return False

    layers: list[tuple[int, int]] = []
    for size, color in zip(observed_sizes, colors):
        mask = inp == color
        if not np.any(mask):
            return False
        if _bbox_size(mask) != size:
            return False
        if int(mask.sum()) <= 4:
            return False
        if int(mask.sum()) >= max(1, 4 * size - 4):
            return False
        if not _all_pixels_on_bbox_border(mask):
            return False
        layers.append((size, color))

    layers.sort(reverse=True)
    if np.array_equal(_rebuild_using_observed_sizes(layers), expected):
        return False

    outer = observed_sizes[0]
    derived_sizes = [outer - 2 * k * gap for k in range(len(observed_sizes))]
    if any(size <= 0 for size in derived_sizes):
        return False
    if observed_sizes[1:] == derived_sizes[1:]:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_frames, gap, outer_size, observed_sizes, colors, expected = _sample_case_parameters()
    side = random.randint(max(13, outer_size + 3), min(28, outer_size + 8))
    if side < outer_size:
        return None

    inp = _render_frames(side, observed_sizes, colors)
    if not _place_gap_object(inp, gap):
        return None

    if not _semantic_validation(inp, expected, observed_sizes, colors, gap):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fragmented-frame gap-coded input")
