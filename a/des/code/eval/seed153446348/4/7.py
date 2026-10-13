from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GREEN = 3
AZURE = 8
WORK_COLORS = [1, 2, 4, 5, 6, 7, 9]
MAX_GENERATION_ATTEMPTS = 300


def _background_color(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    return Counter(corners).most_common(1)[0][0]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _uniform_color(patch: np.ndarray, forbidden: set[int]) -> int | None:
    vals = np.unique(patch)
    if len(vals) != 1:
        return None
    color = int(vals[0])
    if color in forbidden:
        return None
    return color


def _split_top_bottom(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    content_rows = np.flatnonzero(np.any(grid != bg, axis=1))
    assert len(content_rows) > 0

    gaps: list[tuple[int, int, int]] = []
    for prev, nxt in zip(content_rows, content_rows[1:]):
        gap = int(nxt - prev - 1)
        if gap > 0:
            gaps.append((gap, int(prev), int(nxt)))
    assert gaps, "expected a blank band between the legend and the lower scene"

    best_gap = max(gaps, key=lambda item: item[0])[0]
    best = [item for item in gaps if item[0] == best_gap]
    assert len(best) == 1, "expected a unique largest blank row gap"
    _, top_end, bottom_start = best[0]
    return grid[: top_end + 1], grid[bottom_start:]


def _extract_mapping(top: np.ndarray, bg: int) -> dict[int, int]:
    mapping: dict[int, int] = {}
    recognized = 0
    forbidden = {bg, GREEN, AZURE}

    for comp in _components(top == GREEN):
        r0, c0 = comp.min(axis=0)
        r1, c1 = comp.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        if c0 != c1:
            continue
        side = int(r1 - r0 + 1)
        left_c0 = c0 - side
        right_c1 = c0 + side
        if left_c0 < 0 or right_c1 >= top.shape[1]:
            continue

        src_patch = top[r0 : r1 + 1, left_c0:c0]
        dst_patch = top[r0 : r1 + 1, c0 + 1 : c0 + 1 + side]
        if src_patch.shape != (side, side) or dst_patch.shape != (side, side):
            continue

        src = _uniform_color(src_patch, forbidden)
        dst = _uniform_color(dst_patch, forbidden)
        if src is None or dst is None or src == dst:
            continue

        if src in mapping:
            assert mapping[src] == dst, "source color mapping must be unique"
        else:
            mapping[src] = dst
        recognized += 1

    assert recognized == len(mapping), "each green separator must describe exactly one pair"
    assert len(mapping) >= 2, "expected multiple color mapping examples"
    return mapping


def _extract_lower_objects(bottom: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    mask_box = _bbox(bottom == AZURE)
    pattern_box = _bbox((bottom != bg) & (bottom != AZURE))
    assert mask_box is not None and pattern_box is not None

    mr0, mc0, mr1, mc1 = mask_box
    pr0, pc0, pr1, pc1 = pattern_box
    mask = bottom[mr0 : mr1 + 1, mc0 : mc1 + 1]
    pattern = bottom[pr0 : pr1 + 1, pc0 : pc1 + 1]

    assert mask.shape == pattern.shape, "mask and mixed pattern must share the same shape"
    assert bool(np.any(mask == AZURE)), "mask must contain azure pixels"
    assert not bool(np.any(pattern == bg)), "mixed pattern should be fully visible"
    assert not bool(np.any(pattern == AZURE)), "mixed pattern must be distinct from the mask color"
    return pattern.copy(), mask


def _apply_mapping(pattern: np.ndarray, mask: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = pattern.copy()
    active = mask == AZURE
    for src, dst in mapping.items():
        out[active & (pattern == src)] = int(dst)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn source->target color mappings from upper demo pairs, then apply them to the
    lower mixed-color rectangle only at positions selected by the azure mask.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    top, bottom = _split_top_bottom(g, bg)
    mapping = _extract_mapping(top, bg)
    pattern, mask = _extract_lower_objects(bottom, bg)
    return _apply_mapping(pattern, mask, mapping)


def _random_segments(total: int, min_parts: int = 2, max_parts: int = 4) -> list[int]:
    parts = random.randint(min_parts, min(max_parts, total))
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    bounds = [0] + cuts + [total]
    return [bounds[i + 1] - bounds[i] for i in range(parts)]


def _build_pattern(height: int, width: int, colors: list[int]) -> np.ndarray:
    row_segments = _random_segments(height)
    col_segments = _random_segments(width)
    blocks = len(row_segments) * len(col_segments)

    palette = list(colors)
    block_colors = palette[:]
    while len(block_colors) < blocks:
        block_colors.append(random.choice(palette))
    random.shuffle(block_colors)

    pattern = np.empty((height, width), dtype=int)
    idx = 0
    r0 = 0
    for rh in row_segments:
        c0 = 0
        for cw in col_segments:
            pattern[r0 : r0 + rh, c0 : c0 + cw] = int(block_colors[idx])
            idx += 1
            c0 += cw
        r0 += rh
    return pattern


def _build_mask(height: int, width: int, min_true: int, min_false: int) -> np.ndarray:
    mode = random.choice(["scatter", "bands", "rects"])
    mask = np.zeros((height, width), dtype=bool)

    if mode == "scatter":
        density = random.uniform(0.30, 0.70)
        mask = np.random.random((height, width)) < density
    elif mode == "bands":
        row_count = random.randint(1, max(1, height - 1))
        col_count = random.randint(1, max(1, width - 1))
        rows = random.sample(range(height), row_count)
        cols = random.sample(range(width), col_count)
        mask[rows, :] = True
        mask[:, cols] = True
        noise = np.random.random((height, width)) < random.uniform(0.05, 0.15)
        mask ^= noise
    else:
        rects = random.randint(1, 3)
        for _ in range(rects):
            r0 = random.randint(0, height - 1)
            r1 = random.randint(r0, height - 1)
            c0 = random.randint(0, width - 1)
            c1 = random.randint(c0, width - 1)
            mask[r0 : r1 + 1, c0 : c1 + 1] = True
        extras = random.randint(0, max(1, (height * width) // 6))
        for _ in range(extras):
            mask[random.randrange(height), random.randrange(width)] = True

    mask[0, random.randrange(width)] = True
    mask[height - 1, random.randrange(width)] = True
    mask[random.randrange(height), 0] = True
    mask[random.randrange(height), width - 1] = True

    false_cells = list(zip(*np.where(~mask)))
    random.shuffle(false_cells)
    while int(mask.sum()) < min_true and false_cells:
        r, c = false_cells.pop()
        mask[r, c] = True

    true_cells = [(r, c) for r, c in zip(*np.where(mask))]
    random.shuffle(true_cells)
    protected = {
        (0, c) for c in np.flatnonzero(mask[0])
    } | {
        (height - 1, c) for c in np.flatnonzero(mask[height - 1])
    } | {
        (r, 0) for r in np.flatnonzero(mask[:, 0])
    } | {
        (r, width - 1) for r in np.flatnonzero(mask[:, width - 1])
    }
    while mask.size - int(mask.sum()) < min_false:
        changed = False
        for r, c in true_cells:
            if (r, c) in protected:
                continue
            mask[r, c] = False
            changed = True
            break
        if changed:
            true_cells = [(r, c) for r, c in zip(*np.where(mask))]
            random.shuffle(true_cells)
        else:
            break

    assert bool(mask[0].any()) and bool(mask[-1].any()) and bool(mask[:, 0].any()) and bool(mask[:, -1].any())
    assert int(mask.sum()) >= min_true
    assert mask.size - int(mask.sum()) >= min_false
    return mask


def _enforce_branch_coverage(
    pattern: np.ndarray,
    mask: np.ndarray,
    sources: list[int],
    fillers: list[int],
) -> None:
    true_cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    false_cells = [tuple(map(int, rc)) for rc in np.argwhere(~mask)]
    random.shuffle(true_cells)
    random.shuffle(false_cells)

    for src in sources:
        r, c = true_cells.pop()
        pattern[r, c] = int(src)

    r, c = false_cells.pop()
    pattern[r, c] = int(random.choice(sources))

    filler = int(random.choice(fillers))
    r, c = true_cells.pop()
    pattern[r, c] = filler

    r, c = false_cells.pop()
    pattern[r, c] = filler


def _render_pair(grid: np.ndarray, top: int, left: int, size: int, src: int, dst: int) -> None:
    grid[top : top + size, left : left + size] = int(src)
    grid[top : top + size, left + size] = GREEN
    grid[top : top + size, left + size + 1 : left + 2 * size + 1] = int(dst)


def _pair_row_layout(sizes: list[int], row_indices: list[list[int]]) -> tuple[list[int], list[int], list[list[int]]]:
    row_heights = [max(sizes[i] for i in row) for row in row_indices]
    row_widths: list[int] = []
    gaps_by_row: list[list[int]] = []
    for row in row_indices:
        gaps = [random.randint(1, 3) for _ in range(max(0, len(row) - 1))]
        width = sum(2 * sizes[i] + 1 for i in row) + sum(gaps)
        row_widths.append(width)
        gaps_by_row.append(gaps)
    return row_heights, row_widths, gaps_by_row


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice([2, 3])
    filler_count = 1 if pair_count == 3 else random.choice([1, 2, 3])
    picked = random.sample(WORK_COLORS, 2 * pair_count + filler_count)
    sources = picked[:pair_count]
    targets = picked[pair_count : 2 * pair_count]
    fillers = picked[2 * pair_count :]
    mapping = dict(zip(sources, targets))

    pattern_h = random.randint(4, 8)
    pattern_w = random.randint(5, 9)
    pattern = _build_pattern(pattern_h, pattern_w, sources + fillers)

    min_true = pair_count + 1
    min_false = 2
    mask = _build_mask(pattern_h, pattern_w, min_true=min_true, min_false=min_false)
    _enforce_branch_coverage(pattern, mask, sources, fillers)
    expected = _apply_mapping(pattern, np.where(mask, AZURE, BLACK), mapping)
    if np.array_equal(expected, pattern):
        return None

    sizes = [random.randint(2, 4) for _ in range(pair_count)]
    row_count = random.choice([1, 2]) if pair_count > 1 else 1
    row_count = min(row_count, pair_count)
    order = list(range(pair_count))
    random.shuffle(order)
    if row_count == 1:
        row_indices = [order]
    else:
        split = random.randint(1, pair_count - 1)
        row_indices = [order[:split], order[split:]]
        row_indices.sort(key=len, reverse=True)

    row_heights, row_widths, gaps_by_row = _pair_row_layout(sizes, row_indices)
    top_pad = random.randint(1, 2)
    inter_row_gap = random.randint(1, 2) if len(row_indices) == 2 else 0
    sep_rows = random.randint(3, 4)
    bottom_pad_top = random.randint(0, 2)
    bottom_pad_bottom = random.randint(1, 3)
    mask_gap = random.randint(1, 4)

    top_min_width = max(row_widths) + 2
    bottom_min_width = 2 * pattern_w + mask_gap + 2
    width = max(top_min_width, bottom_min_width) + random.randint(0, 5)
    top_content_height = sum(row_heights) + inter_row_gap
    height = top_pad + top_content_height + sep_rows + bottom_pad_top + pattern_h + bottom_pad_bottom

    grid = np.full((height, width), BLACK, dtype=int)

    row_tops: list[int] = []
    cursor = top_pad
    for i, row_height in enumerate(row_heights):
        row_tops.append(cursor)
        cursor += row_height
        if i + 1 < len(row_heights):
            cursor += inter_row_gap

    for row_top, row, row_width, gaps in zip(row_tops, row_indices, row_widths, gaps_by_row):
        x = random.randint(1, width - row_width - 1)
        gap_iter = iter(gaps)
        for j, idx in enumerate(row):
            size = sizes[idx]
            y = row_top + random.randint(0, row_heights[row_tops.index(row_top)] - size)
            _render_pair(grid, y, x, size, sources[idx], targets[idx])
            x += 2 * size + 1
            if j + 1 < len(row):
                x += next(gap_iter)

    bottom_top = top_pad + top_content_height + sep_rows + bottom_pad_top
    left = random.randint(1, width - (2 * pattern_w + mask_gap) - 1)
    mask_left = left + pattern_w + mask_gap
    grid[bottom_top : bottom_top + pattern_h, left : left + pattern_w] = pattern
    grid[bottom_top : bottom_top + pattern_h, mask_left : mask_left + pattern_w] = np.where(mask, AZURE, BLACK)

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
