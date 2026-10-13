from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_mask(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            count += 1
            stack = [(r0, c0)]
            seen[r0, c0] = True
            while stack:
                r, c = stack.pop()
                for rr, cc in _neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
    return count


def _is_connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and _component_count(mask) == 1


def _hole_count(mask: np.ndarray) -> int:
    """
    Count enclosed 4-connected background regions in a cropped binary object mask.
    """
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r0 in range(h):
        for c0 in range(w):
            if not bg[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            touches_border = False
            while stack:
                r, c = stack.pop()
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches_border = True
                for rr, cc in _neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and bg[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            if not touches_border:
                holes += 1
    return holes


def _consecutive_groups(indices: list[int]) -> list[list[int]]:
    groups: list[list[int]] = []
    for idx in indices:
        if not groups or idx != groups[-1][-1] + 1:
            groups.append([idx])
        else:
            groups[-1].append(idx)
    return groups


def _separator_color(grid: np.ndarray) -> int:
    h, w = grid.shape
    line_colors: list[int] = []
    for r in range(h):
        if np.all(grid[r] == grid[r, 0]):
            line_colors.append(int(grid[r, 0]))
    for c in range(w):
        if np.all(grid[:, c] == grid[0, c]):
            line_colors.append(int(grid[0, c]))
    assert line_colors, "expected full-row or full-column separator lines"
    return int(Counter(line_colors).most_common(1)[0][0])


def _spans_between_separator_groups(indices: list[int]) -> list[tuple[int, int]]:
    groups = _consecutive_groups(indices)
    spans: list[tuple[int, int]] = []
    for left, right in zip(groups, groups[1:]):
        start = left[-1] + 1
        end = right[0] - 1
        if start <= end:
            spans.append((start, end))
    return spans


def _parse_layout(grid: np.ndarray) -> tuple[int, int, tuple[int, int], list[tuple[int, int]]]:
    sep = _separator_color(grid)
    h, w = grid.shape

    full_sep_rows = [r for r in range(h) if np.all(grid[r] == sep)]
    full_sep_cols = [c for c in range(w) if np.all(grid[:, c] == sep)]
    row_spans = _spans_between_separator_groups(full_sep_rows)
    col_spans = _spans_between_separator_groups(full_sep_cols)

    assert len(row_spans) == 1, "expected exactly one row of cells"
    assert len(col_spans) >= 2, "expected multiple cell columns"

    cell_heights = {r1 - r0 + 1 for r0, r1 in row_spans}
    cell_widths = {c1 - c0 + 1 for c0, c1 in col_spans}
    assert len(cell_heights) == 1, "cell row heights must be uniform"
    assert len(cell_widths) == 1, "cell widths must be uniform"

    non_sep = grid[grid != sep]
    assert non_sep.size, "grid must contain background/object pixels"
    bg = int(Counter(int(v) for v in non_sep.flat).most_common(1)[0][0])
    return sep, bg, row_spans[0], col_spans


def _extract_object(cell: np.ndarray, bg: int) -> dict[str, object]:
    fg = cell != bg
    assert fg.any(), "each cell must contain one object"
    assert _is_connected(fg), "cell object must be 4-connected"
    colors = np.unique(cell[fg])
    assert colors.size == 1, "cell object must be monochrome"
    mask = _crop_mask(fg)
    return {
        "color": int(colors[0]),
        "mask": mask,
        "holes": _hole_count(mask),
    }


def _paint_centered(
    out: np.ndarray,
    row_span: tuple[int, int],
    col_span: tuple[int, int],
    obj: dict[str, object],
) -> None:
    r0, r1 = row_span
    c0, c1 = col_span
    cell_h = r1 - r0 + 1
    cell_w = c1 - c0 + 1
    mask = obj["mask"]
    assert isinstance(mask, np.ndarray)
    h, w = mask.shape
    assert h <= cell_h and w <= cell_w, "object must fit inside its destination cell"
    top = r0 + (cell_h - h) // 2
    left = c0 + (cell_w - w) // 2
    patch = out[top : top + h, left : left + w]
    patch[mask] = int(obj["color"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder monochrome cell objects by increasing enclosed-hole count.

    Input structure:
    - Black background with a grey frame and grey vertical separators that divide the grid into one row of equal-width cells.
    - Each cell contains exactly one monochrome 4-connected object on the black background.
    - Some objects are solid, while others contain one or more enclosed black holes.

    Rule:
    - Detect the separator grid, isolate each cell, and crop the single object inside it.
    - Count the object's enclosed black 4-connected regions that do not touch the cropped object's bounding-box border.
    - Stable-sort the objects from fewest holes to most holes, then place them back from left to right into the same cell layout, centered in each destination cell.

    Key insight:
    - The sortable attribute is topological: the number of enclosed background holes, not the object's color, area, or outline.
    """
    g = np.asarray(grid, dtype=int)
    sep, bg, row_span, col_spans = _parse_layout(g)

    objects = []
    for col_span in col_spans:
        r0, r1 = row_span
        c0, c1 = col_span
        cell = g[r0 : r1 + 1, c0 : c1 + 1]
        objects.append(_extract_object(cell, bg))

    ordered = sorted(objects, key=lambda obj: int(obj["holes"]))

    out = np.full(g.shape, bg, dtype=int)
    out[g == sep] = sep
    for col_span, obj in zip(col_spans, ordered):
        _paint_centered(out, row_span, col_span, obj)
    return out


def _random_transform(mask: np.ndarray, max_h: int, max_w: int) -> np.ndarray:
    variants: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))

    unique: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for variant in variants:
        key = (tuple(int(x) for x in variant.shape), variant.tobytes())
        if key in seen:
            continue
        seen.add(key)
        if variant.shape[0] <= max_h and variant.shape[1] <= max_w:
            unique.append(variant.copy())
    return random.choice(unique)


def _decorate_shape(mask: np.ndarray, hole_count: int, max_h: int, max_w: int) -> np.ndarray:
    h, w = mask.shape
    canvas = np.zeros((max_h, max_w), dtype=bool)
    top = random.randint(0, max_h - h)
    left = random.randint(0, max_w - w)
    canvas[top : top + h, left : left + w] = mask

    for _ in range(random.randint(0, 3)):
        r0, c0, r1, c1 = _bbox_from_mask(canvas)
        options: list[tuple[str, int]] = []
        if r0 > 0:
            options.append(("top", r0))
        if r1 < max_h - 1:
            options.append(("bottom", max_h - 1 - r1))
        if c0 > 0:
            options.append(("left", c0))
        if c1 < max_w - 1:
            options.append(("right", max_w - 1 - c1))
        if not options:
            break

        side, depth_limit = random.choice(options)
        trial = canvas.copy()
        if side in ("top", "bottom"):
            span = random.randint(1, c1 - c0 + 1)
            start = random.randint(c0, c1 - span + 1)
            depth = random.randint(1, depth_limit)
            if side == "top":
                trial[r0 - depth : r0, start : start + span] = True
            else:
                trial[r1 + 1 : r1 + 1 + depth, start : start + span] = True
        else:
            span = random.randint(1, r1 - r0 + 1)
            start = random.randint(r0, r1 - span + 1)
            depth = random.randint(1, depth_limit)
            if side == "left":
                trial[start : start + span, c0 - depth : c0] = True
            else:
                trial[start : start + span, c1 + 1 : c1 + 1 + depth] = True

        cropped = _crop_mask(trial)
        if _is_connected(cropped) and _hole_count(cropped) == hole_count:
            canvas = trial

    return _crop_mask(canvas)


def _random_polyomino(max_h: int, max_w: int) -> np.ndarray | None:
    max_area = min(max_h * max_w, 18)
    min_area = min(6, max_area)
    if min_area < 4:
        return None

    for _ in range(80):
        target = random.randint(min_area, max_area)
        cells = {(random.randrange(max_h), random.randrange(max_w))}
        while len(cells) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for rr, cc in _neighbors4(r, c):
                    if 0 <= rr < max_h and 0 <= cc < max_w and (rr, cc) not in cells:
                        frontier.append((rr, cc))
            if not frontier:
                break
            rr, cc = random.choice(frontier)
            cells.add((rr, cc))

        mask = np.zeros((max_h, max_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        cropped = _crop_mask(mask)
        if min(cropped.shape) < 2:
            continue
        if _is_connected(cropped) and _hole_count(cropped) == 0:
            return _random_transform(cropped, max_h, max_w)
    return None


def _base_holey_shape(hole_count: int, max_h: int, max_w: int) -> np.ndarray | None:
    layouts = {
        1: [(1, 1, {(0, 0)})],
        2: [(1, 2, {(0, 0), (0, 1)}), (2, 1, {(0, 0), (1, 0)})],
        3: [
            (1, 3, {(0, 0), (0, 1), (0, 2)}),
            (3, 1, {(0, 0), (1, 0), (2, 0)}),
            (2, 2, {(0, 0), (0, 1), (1, 0)}),
            (2, 2, {(0, 0), (0, 1), (1, 1)}),
            (2, 2, {(0, 0), (1, 0), (1, 1)}),
            (2, 2, {(0, 1), (1, 0), (1, 1)}),
        ],
        4: [(2, 2, {(0, 0), (0, 1), (1, 0), (1, 1)})],
    }

    choices = layouts[hole_count][:]
    random.shuffle(choices)
    for rows, cols, carved in choices:
        for _ in range(40):
            row_heights = [random.randint(1, 2) for _ in range(rows)]
            col_widths = [random.randint(1, 2) for _ in range(cols)]
            h = 2 + sum(row_heights) + (rows - 1)
            w = 2 + sum(col_widths) + (cols - 1)
            if h > max_h or w > max_w:
                continue

            mask = np.ones((h, w), dtype=bool)
            y = 1
            for rr in range(rows):
                x = 1
                for cc in range(cols):
                    if (rr, cc) in carved:
                        mask[y : y + row_heights[rr], x : x + col_widths[cc]] = False
                    x += col_widths[cc] + 1
                y += row_heights[rr] + 1
            return mask
    return None


def _make_shape(hole_count: int, max_h: int, max_w: int) -> np.ndarray | None:
    for _ in range(120):
        if hole_count == 0:
            mask = _random_polyomino(max_h, max_w)
        else:
            base = _base_holey_shape(hole_count, max_h, max_w)
            if base is None:
                continue
            mask = _decorate_shape(base, hole_count, max_h, max_w)
            mask = _random_transform(mask, max_h, max_w)
        if mask is None:
            continue
        if mask.shape[0] > max_h or mask.shape[1] > max_w:
            continue
        if _is_connected(mask) and _hole_count(mask) == hole_count:
            return mask
    return None


def _max_supported_holes(max_h: int, max_w: int) -> int:
    minima = {
        4: ((5, 5),),
        3: ((5, 5), (3, 7), (7, 3)),
        2: ((3, 5), (5, 3)),
        1: ((3, 3),),
        0: ((2, 2),),
    }
    for hole_count in range(4, -1, -1):
        if any(max_h >= need_h and max_w >= need_w for need_h, need_w in minima[hole_count]):
            return hole_count
    return 0


def _sample_hole_counts(n_cells: int, max_holes: int) -> list[int] | None:
    allowed = list(range(max_holes + 1))
    if len(allowed) < 2:
        return None

    want_ties = len(allowed) < n_cells or random.random() < 0.5
    for _ in range(100):
        if want_ties:
            counts = [random.choice(allowed) for _ in range(n_cells)]
            if len(set(counts)) == n_cells or len(set(counts)) < 2:
                continue
        else:
            counts = random.sample(allowed, n_cells)
        if counts != sorted(counts):
            return counts
    return None


def _build_grid(cell_h: int, cell_w: int, objects: list[dict[str, object]]) -> np.ndarray:
    h = cell_h + 2
    w = len(objects) * cell_w + len(objects) + 1
    grid = np.full((h, w), BLACK, dtype=int)
    grid[0, :] = GREY
    grid[-1, :] = GREY
    for idx in range(len(objects) + 1):
        grid[:, idx * (cell_w + 1)] = GREY

    row_span = (1, cell_h)
    for idx, obj in enumerate(objects):
        col_start = 1 + idx * (cell_w + 1)
        col_span = (col_start, col_start + cell_w - 1)
        _paint_centered(grid, row_span, col_span, obj)
    return grid


def _object_signature(obj: dict[str, object]) -> tuple[int, tuple[int, int], bytes, int]:
    mask = obj["mask"]
    assert isinstance(mask, np.ndarray)
    return (
        int(obj["color"]),
        tuple(int(x) for x in mask.shape),
        mask.tobytes(),
        int(obj["holes"]),
    )


def _gen_attempt() -> np.ndarray | None:
    n_cells = random.choice((3, 4))
    if n_cells == 3:
        cell_h = random.randint(7, 11)
        cell_w = random.randint(7, 8)
    else:
        cell_h = random.randint(7, 10)
        cell_w = random.randint(5, 6)

    max_obj_h = cell_h - 2
    max_obj_w = cell_w - 2
    max_holes = _max_supported_holes(max_obj_h, max_obj_w)
    if max_holes < 2:
        return None

    hole_counts = _sample_hole_counts(n_cells, max_holes)
    if hole_counts is None:
        return None

    colors = random.sample(PALETTE, n_cells)
    objects: list[dict[str, object]] = []
    seen_shapes: set[tuple[tuple[int, int], bytes]] = set()
    for color, hole_count in zip(colors, hole_counts):
        mask = _make_shape(hole_count, max_obj_h, max_obj_w)
        if mask is None:
            return None
        sig = (tuple(int(x) for x in mask.shape), mask.tobytes())
        if sig in seen_shapes:
            return None
        seen_shapes.add(sig)
        objects.append({"color": color, "mask": mask, "holes": hole_count})

    grid = _build_grid(cell_h, cell_w, objects)
    if max(grid.shape) > 30:
        return None

    sep, bg, row_span, col_spans = _parse_layout(grid)
    if sep != GREY or bg != BLACK:
        return None

    parsed_in = []
    r0, r1 = row_span
    for col_span in col_spans:
        c0, c1 = col_span
        parsed_in.append(_extract_object(grid[r0 : r1 + 1, c0 : c1 + 1], bg))

    if [int(obj["holes"]) for obj in parsed_in] != hole_counts:
        return None

    expected = sorted(objects, key=lambda obj: int(obj["holes"]))
    out = main(grid)
    parsed_out = []
    for col_span in col_spans:
        c0, c1 = col_span
        parsed_out.append(_extract_object(out[r0 : r1 + 1, c0 : c1 + 1], bg))

    if [_object_signature(obj) for obj in parsed_out] != [_object_signature(obj) for obj in expected]:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hole-sorting input")
