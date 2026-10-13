from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400

# Small 4-connected fragment shapes with 1-4 pixels.
BASE_SHAPES = [
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (1, 0), (2, 0), (3, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 1), (2, 1), (2, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
]


def _normalize_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    pts = sorted((r - min_r, c - min_c) for r, c in shape)
    return tuple((int(r), int(c)) for r, c in pts)


def _transform_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    pts = np.array(shape, dtype=int)
    if random.random() < 0.5:
        pts = pts[:, ::-1]
    turns = random.randrange(4)
    for _ in range(turns):
        pts = np.stack((pts[:, 1], -pts[:, 0]), axis=1)
    return _normalize_shape(tuple(map(tuple, pts.tolist())))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_divider_column(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates = []
    for c in range(w):
        if int(g[0, c]) == BLACK:
            continue
        if not np.all(g[:, c] == g[0, c]):
            continue
        if c == 0 or c == w - 1:
            continue
        if not np.any(g[:, :c] != BLACK):
            continue
        if not np.any(g[:, c + 1 :] != BLACK):
            continue
        candidates.append(int(c))
    assert len(candidates) == 1, "expected a unique full-height divider column"
    return int(candidates[0])


def _count_components(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    total = 0
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            total += 1
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
    return int(total)


def _count_components_by_color(left: np.ndarray) -> dict[int, int]:
    counts: dict[int, int] = {}
    for color in sorted(set(map(int, left.reshape(-1)))):
        if color == BLACK:
            continue
        n = _count_components(left == int(color))
        if n > 0:
            counts[int(color)] = int(n)
    return counts


def _ring_pixels(height: int, width: int, center_r: int, center_c: int, layers: int) -> set[tuple[int, int]]:
    pixels = {(int(center_r), int(center_c))}
    for layer in range(1, int(layers) + 1):
        radius = 2 * int(layer) - 1
        top = int(center_r - radius)
        bottom = int(center_r + radius)
        left = int(center_c - radius)
        right = int(center_c + radius)
        for c in range(max(0, left), min(width - 1, right) + 1):
            if 0 <= top < height:
                pixels.add((int(top), int(c)))
            if 0 <= bottom < height:
                pixels.add((int(bottom), int(c)))
        for r in range(max(0, top), min(height - 1, bottom) + 1):
            if 0 <= left < width:
                pixels.add((int(r), int(left)))
            if 0 <= right < width:
                pixels.add((int(r), int(right)))
    return pixels


def _paint_rings(canvas: np.ndarray, center_r: int, center_c: int, color: int, layers: int) -> None:
    for r, c in _ring_pixels(canvas.shape[0], canvas.shape[1], int(center_r), int(center_c), int(layers)):
        canvas[int(r), int(c)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count left-side fragments per color, then expand each matching right-side seed
    into that many concentric hollow square rings, and finally keep only the right half.
    """
    g = np.asarray(grid, dtype=int)
    divider = _find_divider_column(g)
    left = g[:, :divider]
    right = g[:, divider + 1 :].copy()

    counts = _count_components_by_color(left)

    for r, c in np.argwhere(right != BLACK):
        color = int(right[int(r), int(c)])
        _paint_rings(right, int(r), int(c), int(color), counts.get(int(color), 0))
    return right


def _allocate_counts(n_colors: int, total_components: int, max_layers: int) -> list[int]:
    counts = [1] * int(n_colors)
    remaining = int(total_components - n_colors)
    while remaining > 0:
        choices = [i for i, v in enumerate(counts) if v < int(max_layers)]
        if not choices:
            break
        idx = random.choice(choices)
        counts[int(idx)] += 1
        remaining -= 1
    return [int(v) for v in counts]


def _expand_pixels(pixels: set[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in pixels:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < height and 0 <= cc < width:
                    expanded.add((int(rr), int(cc)))
    return expanded


def _choose_seed_centers(
    height: int,
    width: int,
    counts_by_color: dict[int, int],
    require_clipping: bool,
) -> dict[int, tuple[int, int]] | None:
    colors = sorted(counts_by_color, key=lambda color: (-counts_by_color[color], color))
    clipped_color = random.choice(colors) if require_clipping else None
    blocked: set[tuple[int, int]] = set()
    centers: dict[int, tuple[int, int]] = {}

    for color in colors:
        layers = int(counts_by_color[int(color)])
        radius = 2 * layers - 1
        cells = [(r, c) for r in range(height) for c in range(width)]
        random.shuffle(cells)
        found = None
        for r, c in cells:
            clipped = bool(
                r - radius < 0
                or r + radius >= height
                or c - radius < 0
                or c + radius >= width
            )
            if require_clipping and color == clipped_color and not clipped:
                continue
            if not require_clipping and clipped:
                continue
            pixels = _ring_pixels(height, width, int(r), int(c), layers)
            if any(pixel in blocked for pixel in pixels):
                continue
            found = (int(r), int(c), pixels)
            break
        if found is None:
            return None
        r, c, pixels = found
        centers[int(color)] = (int(r), int(c))
        blocked |= _expand_pixels(pixels, height, width)

    return centers


def _place_left_fragments(height: int, width: int, counts_by_color: dict[int, int]) -> np.ndarray | None:
    left = np.full((height, width), BLACK, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    specs = []
    for color, count in counts_by_color.items():
        for _ in range(int(count)):
            shape = _transform_shape(random.choice(BASE_SHAPES))
            h, w = _shape_dims(shape)
            specs.append((h * w, len(shape), int(color), shape))
    random.shuffle(specs)
    specs.sort(key=lambda item: (-item[0], -item[1]))

    for _bbox_area, _area, color, shape in specs:
        h, w = _shape_dims(shape)
        options = [
            (r0, c0)
            for r0 in range(height - h + 1)
            for c0 in range(width - w + 1)
        ]
        random.shuffle(options)
        placed = False
        for r0, c0 in options:
            rr0 = max(0, int(r0 - 1))
            rr1 = min(height, int(r0 + h + 1))
            cc0 = max(0, int(c0 - 1))
            cc1 = min(width, int(c0 + w + 1))
            if bool(blocked[rr0:rr1, cc0:cc1].any()):
                continue
            for dr, dc in shape:
                left[int(r0 + dr), int(c0 + dc)] = int(color)
            blocked[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return None

    return left


def _expected_output(
    height: int,
    width: int,
    centers: dict[int, tuple[int, int]],
    counts_by_color: dict[int, int],
) -> np.ndarray:
    out = np.full((height, width), BLACK, dtype=int)
    for color, (r, c) in centers.items():
        _paint_rings(out, int(r), int(c), int(color), counts_by_color[int(color)])
    return out


def _sample_dimensions() -> tuple[int, int, int]:
    height = random.randint(13, 20)
    left_width = random.randint(8, 12)
    right_width = random.randint(12, 18)
    return int(height), int(left_width), int(right_width)


def _sample_color_plan(height: int, right_width: int, require_clipping: bool) -> dict[int, int] | None:
    if min(height, right_width) >= 17 and random.random() < 0.35:
        n_colors = 4
    else:
        n_colors = random.randint(2, 3)

    max_layers = 4 if require_clipping else min(4, max(1, (min(height, right_width) + 1) // 4))
    colors = random.sample(PALETTE, int(n_colors))
    max_extra = min(4, int(n_colors) * (int(max_layers) - 1))
    total_components = int(n_colors) + random.randint(0, int(max_extra))
    counts = _allocate_counts(int(n_colors), int(total_components), int(max_layers))
    plan = {int(color): int(count) for color, count in zip(colors, counts)}
    return plan if plan else None


def _gen_attempt() -> np.ndarray | None:
    height, left_width, right_width = _sample_dimensions()
    require_clipping = random.choice([False, True])
    counts_by_color = _sample_color_plan(height, right_width, require_clipping)
    if counts_by_color is None:
        return None

    centers = _choose_seed_centers(height, right_width, counts_by_color, require_clipping)
    if centers is None:
        return None

    left = _place_left_fragments(height, left_width, counts_by_color)
    if left is None:
        return None

    if _count_components_by_color(left) != counts_by_color:
        return None

    right = np.full((height, right_width), BLACK, dtype=int)
    for color, (r, c) in centers.items():
        right[int(r), int(c)] = int(color)

    grid = np.full((height, left_width + 1 + right_width), BLACK, dtype=int)
    grid[:, :left_width] = left
    grid[:, left_width] = GREY
    grid[:, left_width + 1 :] = right

    expected = _expected_output(height, right_width, centers, counts_by_color)
    try:
        out = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(out, expected):
        return None
    if out.shape != (height, right_width):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
