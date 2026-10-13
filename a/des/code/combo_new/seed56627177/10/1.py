import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _infer_periodic_pattern(tail: list[int]) -> tuple[int, ...]:
    for period in (2, 3):
        if len(tail) < 2 * period or len(tail) % period != 0:
            continue
        motif = tuple(int(v) for v in tail[:period])
        if len(set(motif)) != period:
            continue
        if all(tuple(int(v) for v in tail[i : i + period]) == motif for i in range(0, len(tail), period)):
            return motif
    raise AssertionError("legend segment must contain a repeated 2- or 3-color pattern")


def _parse_legend(grid: np.ndarray) -> dict[int, tuple[int, ...]]:
    h, w = grid.shape
    legend_rows: list[int] = []
    for r in range(min(2, h)):
        if np.any(grid[r] != BLACK):
            legend_rows.append(r)
        else:
            break
    assert legend_rows, "expected one or two legend rows at the top"
    if len(legend_rows) == 2:
        assert np.any(grid[1] != BLACK), "a two-row legend must use the second row"

    mapping: dict[int, tuple[int, ...]] = {}
    for r in legend_rows:
        row = grid[r]
        c = 0
        found_segment = False
        while c < w:
            while c < w and int(row[c]) == BLACK:
                c += 1
            if c >= w:
                break
            start = c
            while c < w and int(row[c]) != BLACK:
                c += 1
            segment = row[start:c]
            assert segment.size >= 5, "each legend segment needs a label plus at least two repeats"
            label = int(segment[0])
            assert label != GREY, "grey is reserved for rectangle borders"
            pattern = _infer_periodic_pattern([int(v) for v in segment[1:]])
            assert label not in mapping, "legend label colors must be unique"
            mapping[label] = pattern
            found_segment = True
        assert found_segment, "each legend row must contain at least one segment"
    return mapping


def _find_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    rects: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == GREY):
        coords = np.array(comp, dtype=int)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "grey frame must enclose a non-empty interior"

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(patch, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        assert int(border.sum()) == len(comp), "grey component must be exactly a rectangular border"
        assert np.all(patch[border] == GREY), "frame border must be fully grey"
        assert np.all(patch[~border] != GREY), "frame interior must not contain grey pixels"
        rects.append((int(r0), int(c0), int(r1), int(c1)))

    rects.sort()
    assert rects, "expected at least one grey rectangle"
    return rects


def _tile_pattern(pattern: tuple[int, ...], height: int, width: int) -> np.ndarray:
    row = np.array([pattern[c % len(pattern)] for c in range(width)], dtype=int)
    return np.tile(row, (height, 1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey rectangular region with the periodic color band selected by its interior marker.

    Input structure:
    - The background is black.
    - The top of the grid contains one or two rows of horizontal legend bands. Each legend segment is a single label pixel followed by a repeated 2- or 3-color periodic strip, and neighboring segments are separated by black gaps.
    - Below the legend are several disjoint grey rectangular frames. Each frame encloses a black interior containing exactly one non-black marker pixel whose color matches one legend label.

    Rule:
    - Read every legend segment as a mapping from its label color to the periodic strip that follows it.
    - For each grey rectangle, find the unique non-black interior marker and use its color as the lookup key into the legend.
    - Starting from the rectangle interior's top-left corner, repeat the matched strip across columns and copy that same row pattern down every interior row, overwriting the original marker.

    Key insight:
    - The colored interior pixel is not part of the final image; it is only an index into the top legend, which supplies the periodic fill template for the whole enclosed rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    legend = _parse_legend(g)
    out = g.copy()

    for r0, c0, r1, c1 in _find_rectangles(g):
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_locs = np.argwhere(interior != BLACK)
        assert len(marker_locs) == 1, "each rectangle must contain exactly one marker pixel"
        mr, mc = map(int, marker_locs[0])
        label = int(interior[mr, mc])
        assert label in legend, "rectangle marker must match a legend label"
        out[r0 + 1 : r1, c0 + 1 : c1] = _tile_pattern(legend[label], r1 - r0 - 1, c1 - c0 - 1)

    return out


def _make_segment(rng: random.Random, label: int) -> tuple[np.ndarray, tuple[int, ...]]:
    period = rng.choice([2, 3])
    motif = tuple(rng.sample([c for c in COLORS if c != label], period))
    repeats = rng.randint(2, 4)
    segment = np.array((label,) + motif * repeats, dtype=int)
    return segment, motif


def _build_legend_rows(rng: random.Random) -> tuple[list[np.ndarray], dict[int, tuple[int, ...]]] | None:
    for _ in range(80):
        row_count = rng.choice([1, 2])
        if row_count == 1:
            counts = [rng.randint(2, 3)]
        else:
            total = rng.randint(2, 4)
            first = rng.randint(1, total - 1)
            counts = [first, total - first]

        total_labels = sum(counts)
        labels = rng.sample(COLORS, total_labels)
        label_to_pattern: dict[int, tuple[int, ...]] = {}
        rows: list[np.ndarray] = []
        idx = 0
        ok = True
        for count in counts:
            pieces: list[int] = []
            for j in range(count):
                label = labels[idx]
                idx += 1
                segment, pattern = _make_segment(rng, label)
                label_to_pattern[label] = pattern
                if j > 0:
                    pieces.extend([BLACK] * rng.randint(1, 2))
                pieces.extend(int(v) for v in segment)
            row = np.array(pieces, dtype=int)
            if row.size > 26:
                ok = False
                break
            rows.append(row)
        if ok:
            return rows, label_to_pattern
    return None


def _assign_labels(rng: random.Random, labels: list[int], count: int) -> list[int]:
    if count == 1:
        return [rng.choice(labels)]
    used_count = rng.randint(2, min(len(labels), count))
    used = rng.sample(labels, used_count)
    assigned = used[:]
    while len(assigned) < count:
        assigned.append(rng.choice(used))
    rng.shuffle(assigned)
    return assigned


def _pack_rectangles(
    sizes: list[tuple[int, int]],
    width_limit: int,
    rng: random.Random,
) -> tuple[list[tuple[int, int]], int, int] | None:
    x = 0
    y = 0
    row_height = 0
    max_right = 0
    placements: list[tuple[int, int]] = []
    for height, width in sizes:
        gap_x = 0 if x == 0 else rng.randint(2, 3)
        if x > 0 and x + gap_x + width > width_limit:
            y += row_height + rng.randint(2, 3)
            x = 0
            row_height = 0
            gap_x = 0
        if width > width_limit:
            return None
        left = x + gap_x
        placements.append((y, left))
        x = left + width
        row_height = max(row_height, height)
        max_right = max(max_right, x)
    return placements, y + row_height, max_right


def _draw_rectangle(
    grid: np.ndarray,
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    label: int,
    rng: random.Random,
) -> None:
    bottom = top + inner_h + 1
    right = left + inner_w + 1
    grid[top : bottom + 1, left] = GREY
    grid[top : bottom + 1, right] = GREY
    grid[top, left : right + 1] = GREY
    grid[bottom, left : right + 1] = GREY
    mr = rng.randint(top + 1, bottom - 1)
    mc = rng.randint(left + 1, right - 1)
    grid[mr, mc] = label


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    built_legend = _build_legend_rows(rng)
    if built_legend is None:
        return None
    legend_rows, legend = built_legend
    labels = list(legend)

    rect_count = rng.randint(2, 5)
    assigned_labels = _assign_labels(rng, labels, rect_count)
    sizes: list[tuple[int, int]] = []
    for _ in range(rect_count):
        inner_h = rng.randint(2, 6)
        inner_w = rng.randint(3, 8)
        sizes.append((inner_h + 2, inner_w + 2))

    width_floor = max(max(len(row) for row in legend_rows), 12)
    width_limit = rng.randint(width_floor, 28)
    packed = _pack_rectangles(sizes, width_limit, rng)
    if packed is None:
        return None
    placements, pack_h, pack_w = packed

    separator = rng.randint(1, 2)
    bottom_pad = rng.randint(0, 2)
    grid_h = len(legend_rows) + separator + pack_h + bottom_pad
    grid_w = max(max(len(row) for row in legend_rows), pack_w)
    grid_w += rng.randint(0, min(2, 30 - grid_w))
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    for r, row in enumerate(legend_rows):
        start = rng.randint(0, grid_w - len(row))
        grid[r, start : start + len(row)] = row

    rect_top = len(legend_rows) + separator
    pack_shift = rng.randint(0, grid_w - pack_w)
    for (y, x), (height, width), label in zip(placements, sizes, assigned_labels):
        inner_h = height - 2
        inner_w = width - 2
        _draw_rectangle(grid, rect_top + y, pack_shift + x, inner_h, inner_w, label, rng)

    try:
        solved = main(grid)
    except Exception:
        return None

    for (y, x), (height, width), label in zip(placements, sizes, assigned_labels):
        pattern = legend[label]
        top = rect_top + y
        left = pack_shift + x
        expected = _tile_pattern(pattern, height - 2, width - 2)
        if not np.array_equal(solved[top + 1 : top + height - 1, left + 1 : left + width - 1], expected):
            return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {MAX_RETRIES} attempts")
