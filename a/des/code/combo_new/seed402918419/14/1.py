from __future__ import annotations

import random

import numpy as np


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _mode_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _valid_frame_bbox(
    grid: np.ndarray, points: list[tuple[int, int]], color: int, bg: int
) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(points)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return None

    point_set = set(points)
    expected_area = 2 * (h + w) - 4
    if len(points) != expected_area:
        return None

    for r in range(r0, r1 + 1):
        if (r, c0) not in point_set or (r, c1) not in point_set:
            return None
    for c in range(c0, c1 + 1):
        if (r0, c) not in point_set or (r1, c) not in point_set:
            return None

    inside = grid[r0 + 1 : r1, c0 + 1 : c1]
    if inside.size == 0:
        return None
    if np.any(inside == color):
        return None
    if int(np.count_nonzero(inside != bg)) != 1:
        return None
    return r0, c0, r1, c1


def _find_frames(
    grid: np.ndarray, bg: int
) -> tuple[int, list[tuple[int, int, int, int]]]:
    candidates: list[tuple[int, int, int, list[tuple[int, int, int, int]]]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        boxes: list[tuple[int, int, int, int]] = []
        area = 0
        for comp in _components(grid == color):
            box = _valid_frame_bbox(grid, comp, color, bg)
            if box is not None:
                boxes.append(box)
                area += len(comp)
        if boxes:
            candidates.append((len(boxes), area, color, boxes))

    assert candidates, "expected at least one framed color"
    candidates.sort(key=lambda item: (item[0], item[1], -item[2]), reverse=True)
    _, _, color, boxes = candidates[0]
    boxes = sorted(boxes)
    return color, boxes


def _parse_legend_entry(
    grid: np.ndarray, points: list[tuple[int, int]], bg: int, frame_color: int
) -> tuple[int, np.ndarray] | None:
    r0, c0, r1, c1 = _bbox(points)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h not in (2, 3) or w != h + 1:
        return None
    if len(points) != h * h + 1:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    left_non_bg = [(r, c) for r in range(h) for c in range(1) if int(sub[r, c]) != bg]
    if len(left_non_bg) != 1:
        return None

    pattern = sub[:, 1:]
    if np.any(pattern == bg) or np.any(pattern == frame_color):
        return None

    key_r, key_c = left_non_bg[0]
    key_color = int(sub[key_r, key_c])
    return key_color, pattern.copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each framed region by tiling the legend pattern selected by its seed color.

    Input structure:
    - The background is black. Several same-colored one-cell-thick rectangular frames appear in the grid.
    - Each frame encloses exactly one non-black, non-frame single-pixel seed.
    - In one corner, separate legend entries each contain one index pixel and an attached 2x2 or 3x3 color pattern block.

    Rule:
    - Identify every legend entry and map its index-pixel color to the pattern block on its right.
    - For each rectangular frame, read the enclosed seed color and look up the corresponding legend pattern.
    - Tile that pattern across the whole frame interior from the interior's top-left corner, replacing the seed, while leaving the frame border unchanged.

    Key insight:
    - The seed pixel is only a lookup key: the real work is extracting the corner color-to-pattern legend and then repeating the matched tile inside each closed rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _mode_color(g)

    frame_color, frame_boxes = _find_frames(g, bg)
    frame_lookup: list[tuple[int, int, int, int]] = [
        (r0 + 1, c0 + 1, r1 - 1, c1 - 1) for r0, c0, r1, c1 in frame_boxes
    ]

    seed_by_frame: dict[int, int] = {}
    pattern_by_color: dict[int, np.ndarray] = {}
    feature_mask = (g != bg) & (g != frame_color)
    for comp in _components(feature_mask):
        containing = [
            idx
            for idx, (ir0, ic0, ir1, ic1) in enumerate(frame_lookup)
            if all(ir0 <= r <= ir1 and ic0 <= c <= ic1 for r, c in comp)
        ]
        if containing:
            assert len(containing) == 1, "foreground component should belong to one frame interior"
            assert len(comp) == 1, "each frame should contain exactly one single-pixel seed component"
            idx = containing[0]
            seed_color = int(g[comp[0]])
            assert idx not in seed_by_frame, "each frame should contain exactly one seed"
            seed_by_frame[idx] = seed_color
            continue

        parsed = _parse_legend_entry(g, comp, bg, frame_color)
        assert parsed is not None, "non-frame foreground outside regions should be a legend entry"
        key_color, pattern = parsed
        assert key_color not in pattern_by_color, "legend keys must be unique"
        pattern_by_color[key_color] = pattern

    assert len(seed_by_frame) == len(frame_boxes), "every frame should contribute one seed"
    assert pattern_by_color, "expected at least one legend mapping"

    out = g.copy()
    for idx, (r0, c0, r1, c1) in enumerate(frame_boxes):
        seed_color = seed_by_frame[idx]
        assert seed_color in pattern_by_color, "seed color missing from legend"
        tile = pattern_by_color[seed_color]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        row_ids = np.arange(inner_h)[:, None] % tile.shape[0]
        col_ids = np.arange(inner_w)[None, :] % tile.shape[1]
        out[r0 + 1 : r1, c0 + 1 : c1] = tile[row_ids, col_ids]
    return out


BG = 0
GRID_MIN = 18
GRID_MAX = 30
LEGEND_GAP = 1
MAX_GENERATION_ATTEMPTS = 400


def _random_pattern(size: int, palette: list[int], existing: list[np.ndarray]) -> np.ndarray:
    for _ in range(200):
        color_count = random.randint(2, min(4, len(palette)))
        colors = random.sample(palette, color_count)
        arr = np.array(random.choices(colors, k=size * size), dtype=int).reshape(size, size)
        if len(set(map(int, arr.ravel()))) < 2:
            continue
        if any(np.array_equal(arr, prev) for prev in existing):
            continue
        return arr
    raise RuntimeError("failed to sample a distinct legend pattern")


def _entry_patch(key_color: int, pattern: np.ndarray) -> np.ndarray:
    size = int(pattern.shape[0])
    patch = np.full((size, size + 1), BG, dtype=int)
    patch[random.randrange(size), 0] = key_color
    patch[:, 1:] = pattern
    return patch


def _reserve(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, margin: int) -> None:
    h, w = mask.shape
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(h - 1, r1 + margin)
    cc1 = min(w - 1, c1 + margin)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_boxes(
    shape: tuple[int, int],
    legend_box: tuple[int, int, int, int],
    box_sizes: list[tuple[int, int]],
) -> list[tuple[int, int, int, int]] | None:
    h, w = shape
    blocked = np.zeros((h, w), dtype=bool)
    _reserve(blocked, *legend_box, margin=1)

    placements: list[tuple[int, int, int, int]] = [(-1, -1, -1, -1)] * len(box_sizes)
    order = sorted(range(len(box_sizes)), key=lambda idx: box_sizes[idx][0] * box_sizes[idx][1], reverse=True)
    for idx in order:
        box_h, box_w = box_sizes[idx]
        candidates = []
        for r0 in range(1, h - box_h):
            for c0 in range(1, w - box_w):
                r1 = r0 + box_h - 1
                c1 = c0 + box_w - 1
                if np.any(blocked[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2]):
                    continue
                candidates.append((r0, c0, r1, c1))
        if not candidates:
            return None
        choice = random.choice(candidates)
        placements[idx] = choice
        _reserve(blocked, *choice, margin=1)
    return placements


def _render_case(
    shape: tuple[int, int],
    corner: str,
    legend_order: list[int],
    patterns: dict[int, np.ndarray],
    border_color: int,
    frames: list[tuple[int, int, int, int]],
    seeds: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    size = int(next(iter(patterns.values())).shape[0])
    legend_h = len(legend_order) * size + (len(legend_order) - 1) * LEGEND_GAP
    legend_w = size + 1

    if corner.startswith("T"):
        legend_r0 = 0
    else:
        legend_r0 = h - legend_h
    if corner.endswith("L"):
        legend_c0 = 0
    else:
        legend_c0 = w - legend_w

    inp = np.full((h, w), BG, dtype=int)
    for i, key_color in enumerate(legend_order):
        patch = _entry_patch(key_color, patterns[key_color])
        rr = legend_r0 + i * (size + LEGEND_GAP)
        cc = legend_c0
        inp[rr : rr + size, cc : cc + size + 1] = patch

    for (r0, c0, r1, c1), seed_color in zip(frames, seeds, strict=False):
        inp[r0 : r1 + 1, c0] = border_color
        inp[r0 : r1 + 1, c1] = border_color
        inp[r0, c0 : c1 + 1] = border_color
        inp[r1, c0 : c1 + 1] = border_color
        sr = random.randint(r0 + 1, r1 - 1)
        sc = random.randint(c0 + 1, c1 - 1)
        inp[sr, sc] = seed_color

    out = inp.copy()
    for (r0, c0, r1, c1), seed_color in zip(frames, seeds, strict=False):
        tile = patterns[seed_color]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        row_ids = np.arange(inner_h)[:, None] % tile.shape[0]
        col_ids = np.arange(inner_w)[None, :] % tile.shape[1]
        out[r0 + 1 : r1, c0 + 1 : c1] = tile[row_ids, col_ids]
    return inp, out


def _sample_case() -> tuple[np.ndarray, np.ndarray] | None:
    pattern_size = random.choice([2, 3])
    legend_count = random.randint(2, 4)
    region_count = random.randint(legend_count, min(5, legend_count + 2))

    all_colors = list(range(1, 10))
    border_color = random.choice(all_colors)
    remaining = [c for c in all_colors if c != border_color]
    key_colors = random.sample(remaining, legend_count)
    pattern_palette = [c for c in remaining if c not in key_colors]
    if len(pattern_palette) < 2:
        return None

    patterns: dict[int, np.ndarray] = {}
    used_patterns: list[np.ndarray] = []
    for key_color in key_colors:
        pattern = _random_pattern(pattern_size, pattern_palette, used_patterns)
        patterns[key_color] = pattern
        used_patterns.append(pattern)

    region_keys = key_colors.copy()
    while len(region_keys) < region_count:
        region_keys.append(random.choice(key_colors))
    random.shuffle(region_keys)

    if region_count <= 2:
        inner_max = 8
    elif region_count == 3:
        inner_max = 7
    elif region_count == 4:
        inner_max = 6
    else:
        inner_max = 5
    box_sizes = []
    for _ in range(region_count):
        inner_h = random.randint(3, inner_max)
        inner_w = random.randint(3, inner_max + 1)
        box_sizes.append((inner_h + 2, inner_w + 2))

    legend_h = legend_count * pattern_size + (legend_count - 1) * LEGEND_GAP
    legend_w = pattern_size + 1
    required_area = legend_h * (legend_w + 2) + sum((bh + 2) * (bw + 2) for bh, bw in box_sizes)
    min_side = max(GRID_MIN, int(np.ceil(np.sqrt(required_area * 1.35))))
    if min_side > GRID_MAX:
        return None

    h = random.randint(min_side, GRID_MAX)
    w = random.randint(min_side, GRID_MAX)
    corner = random.choice(["TL", "TR", "BL", "BR"])
    legend_box = (
        0 if corner.startswith("T") else h - legend_h,
        0 if corner.endswith("L") else w - legend_w,
        (legend_h - 1) if corner.startswith("T") else h - 1,
        (legend_w - 1) if corner.endswith("L") else w - 1,
    )
    placements = _place_boxes((h, w), legend_box, box_sizes)
    if placements is None:
        return None

    legend_order = key_colors[:]
    random.shuffle(legend_order)
    return _render_case((h, w), corner, legend_order, patterns, border_color, placements, region_keys)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        sampled = _sample_case()
        if sampled is None:
            continue
        inp, expected = sampled
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid ARC input")
