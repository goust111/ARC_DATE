from __future__ import annotations

from collections import deque

import numpy as np


BACKGROUND = 0
AXIS_COLOR = 2
PALETTE = [1, 3, 4, 5, 6, 7, 8, 9]
_RNG = np.random.default_rng()


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _axis_info(grid: np.ndarray) -> tuple[int, int]:
    red = np.argwhere(grid == AXIS_COLOR)
    assert red.size > 0, "expected a red vertical axis"
    cols = np.unique(red[:, 1])
    assert len(cols) == 1, "expected the red axis to occupy one column"
    rows = np.sort(red[:, 0])
    assert np.all(rows == np.arange(rows[0], rows[-1] + 1)), "expected a contiguous vertical axis"
    return int(cols[0]), int(rows.size)


def _extract_rectangles(grid: np.ndarray, bg: int, axis_col: int) -> list[dict[str, int]]:
    h, _w = grid.shape
    mask = (grid != bg) & (grid != AXIS_COLOR)
    rectangles: list[dict[str, int]] = []
    for comp in _components4(mask):
        r0, c0, r1, c1 = _bbox(comp)
        color = int(grid[comp[0, 0], comp[0, 1]])
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch == color), "expected each foreground object to be a solid rectangle"
        assert r1 == h - 1, "expected rectangles to touch the bottom row"
        if c1 < axis_col:
            group = 1
        elif c0 > axis_col:
            group = 2
        else:
            raise AssertionError("rectangles must lie entirely on one side of the axis")
        rectangles.append(
            {
                "color": color,
                "left": int(c0),
                "height": int(r1 - r0 + 1),
                "width": int(c1 - c0 + 1),
                "group": group,
            }
        )

    assert len(rectangles) >= 2, "expected multiple rectangles"
    assert any(rect["group"] == 1 for rect in rectangles), "expected at least one left-side rectangle"
    assert any(rect["group"] == 2 for rect in rectangles), "expected at least one right-side rectangle"
    left_edges = [rect["left"] for rect in rectangles]
    assert len(set(left_edges)) == len(left_edges), "expected unique left edges"
    return sorted(rectangles, key=lambda rect: rect["left"])


def _render_chain(rectangles: list[dict[str, int]], shape: tuple[int, int], bg: int) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    owner = np.zeros(shape, dtype=int)

    row = 0
    col = 0
    h, w = shape
    for rect in rectangles:
        rh = int(rect["height"])
        rw = int(rect["width"])
        color = int(rect["color"])
        group = int(rect["group"])
        assert row + rh <= h and col + rw <= w, "diagonal chain does not fit in the canvas"
        for rr in range(row, row + rh):
            for cc in range(col, col + rw):
                current = int(owner[rr, cc])
                if current == 0 or current == group or (current == 1 and group == 2):
                    out[rr, cc] = color
                    owner[rr, cc] = group
        row += rh - 1
        col += rw - 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack the bottom rectangles into a diagonal chain while giving mirrored-axis-group pixels priority over left-group pixels.

    Input structure
    - The grid uses a dominant background, one red vertical axis segment, and 4-5 solid monochrome rectangles.
    - Every rectangle touches the bottom row, has a unique left edge, and lies fully on either side of the red axis.
    - Rectangles on the right side are the mirrored-tag group; rectangles on the left side are the unmirrored group.

    Rule
    - Ignore the red axis itself after using it to split rectangles into left and right groups.
    - Sort all rectangles by original left edge and place them onto a blank background canvas as a diagonal corner-overlap chain starting from the upper-left.
    - When two pixels overlap, a later rectangle overwrites an earlier one if both belong to the same side-group; across groups, the right-side mirrored group always wins.

    Key insight
    - The red axis is not copied: it only labels which rectangles belong to the mirrored priority group before the usual diagonal stacking is performed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _most_common_color(g)
    axis_col, _axis_height = _axis_info(g)
    rectangles = _extract_rectangles(g, bg, axis_col)
    return _render_chain(rectangles, g.shape, bg)


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = np.sort(_RNG.integers(0, total + 1, size=parts - 1))
    points = np.concatenate(([0], cuts, [total]))
    return [int(points[i + 1] - points[i]) for i in range(parts)]


def _split_choices(n: int) -> np.ndarray:
    weights = np.array([1 if i in {1, n - 1} else 2 for i in range(1, n)], dtype=float)
    return weights / weights.sum()


def _sample_layout() -> tuple[int, int, int, np.ndarray, np.ndarray] | None:
    for _ in range(200):
        height = int(_RNG.integers(10, 14))
        width = int(_RNG.integers(18, 27))
        count = int(_RNG.choice([4, 5]))
        heights = _RNG.integers(2, 5, size=count)
        widths = _RNG.integers(2, 5, size=count)
        split = int(_RNG.choice(np.arange(1, count), p=_split_choices(count)))

        chain_h = 1 + int(np.sum(heights - 1))
        chain_w = 1 + int(np.sum(widths - 1))
        if chain_h > height or chain_w > width:
            continue

        if not np.any(heights[:split] != widths[:split]):
            continue
        if not np.any(heights[split:] != widths[split:]):
            continue

        min_input_width = int(np.sum(widths)) + count + 1
        if min_input_width > width:
            continue

        return height, width, split, heights, widths
    return None


def _build_input(
    height: int, width: int, split: int, heights: np.ndarray, widths: np.ndarray
) -> np.ndarray:
    count = len(heights)
    slack = width - (int(np.sum(widths)) + count + 1)
    extras = _random_composition(slack, count + 2)

    left_margin = extras[0]
    right_margin = extras[1]
    gap_values = [1 + extra for extra in extras[2:]]

    internal_left = gap_values[: max(0, split - 1)]
    axis_left_gap = gap_values[max(0, split - 1)]
    axis_right_gap = gap_values[max(0, split - 1) + 1]
    internal_right = gap_values[max(0, split - 1) + 2 :]

    lefts: list[int] = []
    cursor = left_margin
    for i in range(split):
        lefts.append(cursor)
        cursor += int(widths[i])
        if i < split - 1:
            cursor += internal_left[i]

    cursor += axis_left_gap
    axis_col = cursor
    cursor += 1 + axis_right_gap

    for i in range(split, count):
        lefts.append(cursor)
        cursor += int(widths[i])
        if i < count - 1:
            cursor += internal_right[i - split]

    assert cursor + right_margin == width

    colors = list(_RNG.choice(PALETTE, size=count, replace=False))
    grid = np.full((height, width), BACKGROUND, dtype=int)
    for i in range(count):
        rect_h = int(heights[i])
        rect_w = int(widths[i])
        top = height - rect_h
        left = lefts[i]
        grid[top:height, left : left + rect_w] = colors[i]

    axis_height = int(_RNG.integers(int(heights.max()), min(height, int(heights.max()) + 2) + 1))
    grid[height - axis_height : height, axis_col] = AXIS_COLOR

    return grid


def _valid_generated_sample(grid: np.ndarray) -> bool:
    try:
        axis_col, axis_height = _axis_info(grid)
        rects = _extract_rectangles(grid, BACKGROUND, axis_col)
        if len(rects) not in {4, 5}:
            return False
        if axis_height < max(rect["height"] for rect in rects):
            return False
        if not any(rects[i]["group"] != rects[i + 1]["group"] for i in range(len(rects) - 1)):
            return False
        out = _render_chain(rects, grid.shape, BACKGROUND)
        if max(out.shape) > 30:
            return False
        if not np.any(out != BACKGROUND):
            return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(300):
        layout = _sample_layout()
        if layout is None:
            continue
        grid = _build_input(*layout)
        if _valid_generated_sample(grid):
            return grid
    raise RuntimeError("failed to generate a valid sample")
