from __future__ import annotations

from collections import Counter
import random

import numpy as np


PALETTE = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 400
OBJECT_COUNT_RANGE = (2, 5)
RECT_SIDE_RANGE = (3, 5)
ROW_GAP_RANGE = (1, 3)
COL_GAP_RANGE = (1, 3)
CANVAS_SLACK_MAX = 6


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = map(int, mask.shape)
    seen = np.zeros((height, width), dtype=bool)
    out: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))

    return out


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = map(int, patch.shape)
    dst[int(top) : int(top) + h, int(left) : int(left) + w] = patch


def _parse_rectangles(grid: np.ndarray, bg: int) -> list[dict]:
    objects: list[dict] = []
    for coords in _components(grid != bg):
        r0, c0, r1, c1 = _bbox_coords(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        h, w = map(int, patch.shape)
        assert int(coords.shape[0]) == h * w, "each object must fill its bounding box"

        counts = Counter(map(int, patch.ravel()))
        ranked = counts.most_common()
        assert 1 <= len(ranked) <= 2, "each rectangle must have at most one pointer color"

        body_color = int(ranked[0][0])
        marker_color: int | None = None
        if len(ranked) == 2:
            assert ranked[1][1] == 1, "pointer must be a singleton pixel"
            assert ranked[0][1] == h * w - 1, "rectangle body must be solid except for one pointer"
            marker_color = int(ranked[1][0])
            rows, cols = np.where(patch != body_color)
            assert len(rows) == 1
            mr, mc = int(rows[0]), int(cols[0])
            assert 0 < mr < h - 1 and 0 < mc < w - 1, "pointer pixel must be strictly inside the rectangle"
            assert marker_color != body_color, "pointer must reference another rectangle color"

        objects.append(
            {
                "body": body_color,
                "next": marker_color,
                "shape": (h, w),
            }
        )

    assert objects, "expected at least one rectangle"
    return objects


def _chain_order(objects: list[dict]) -> list[dict]:
    body_to_obj = {int(obj["body"]): obj for obj in objects}
    assert len(body_to_obj) == len(objects), "rectangle body colors must be unique"

    incoming = Counter()
    tail_count = 0
    for obj in objects:
        nxt = obj["next"]
        if nxt is None:
            tail_count += 1
            continue
        assert int(nxt) in body_to_obj, "pointer must target an existing rectangle color"
        incoming[int(nxt)] += 1

    assert tail_count == 1, "expected exactly one tail rectangle without a pointer"
    assert all(count == 1 for count in incoming.values()), "each rectangle can have at most one predecessor"

    heads = [body for body in body_to_obj if incoming[int(body)] == 0]
    assert len(heads) == 1, "expected exactly one unreferenced head rectangle"

    ordered: list[dict] = []
    seen: set[int] = set()
    cur = int(heads[0])
    while True:
        assert cur not in seen, "pointer chain must not cycle"
        seen.add(cur)
        obj = body_to_obj[cur]
        ordered.append(obj)
        nxt = obj["next"]
        if nxt is None:
            break
        cur = int(nxt)

    assert len(ordered) == len(objects), "pointer chain must visit every rectangle exactly once"
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the rectangles' internal color pointers and repack pure copies into an overlapping diagonal chain.

    Input structure:
    - The most common color is the background.
    - Every foreground object is a filled axis-aligned rectangle, and the rectangles are separated by background cells.
    - Each rectangle has a unique body color; all but one contain exactly one internal singleton pixel whose color equals the next rectangle's body color.

    Rule:
    - Treat each rectangle's singleton non-body pixel as a pointer to the next rectangle in a linked list of body colors.
    - Find the unique head whose body color is never pointed to, traverse the chain to the unique tail with no pointer, and ignore the rectangles' original positions.
    - Clear the canvas to background, then place pure body-color copies in chain order: the first starts at the upper-left corner, and each later rectangle's upper-left cell overlaps the previous rectangle's lower-right cell.

    Key insight:
    - The off-color pixel is not paint to preserve; it is a color-key pointer that defines the only valid ordering of the rectangles.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    bg = _get_bg(g)

    ordered = _chain_order(_parse_rectangles(g, bg))

    out = np.full((height, width), bg, dtype=int)
    row = 0
    col = 0
    for obj in ordered:
        h, w = map(int, obj["shape"])
        assert row + h <= height and col + w <= width, "packed chain must fit inside the canvas"
        patch = np.full((h, w), int(obj["body"]), dtype=int)
        _paste(out, patch, row, col)
        row += h - 1
        col += w - 1

    return out


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    values: list[int] = []
    prev = 0
    for cut in cuts + [total]:
        values.append(int(cut - prev))
        prev = int(cut)
    return values


def _expected_output(shape: tuple[int, int], bg: int, ordered: list[dict]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    row = 0
    col = 0
    for obj in ordered:
        h, w = map(int, obj["shape"])
        _paste(out, np.full((h, w), int(obj["body"]), dtype=int), row, col)
        row += h - 1
        col += w - 1
    return out


def _input_layout_dims(rows: list[list[dict]]) -> tuple[int, int, list[int], list[list[int]], list[int]]:
    row_heights = [max(int(obj["shape"][0]) for obj in row) for row in rows]
    row_gaps = [random.randint(*ROW_GAP_RANGE) for _ in range(len(rows) - 1)]

    col_gaps_by_row: list[list[int]] = []
    row_widths: list[int] = []
    for row in rows:
        gaps = [random.randint(*COL_GAP_RANGE) for _ in range(len(row) - 1)]
        width = sum(int(obj["shape"][1]) for obj in row) + sum(gaps)
        col_gaps_by_row.append(gaps)
        row_widths.append(int(width))

    compact_h = sum(row_heights) + sum(row_gaps)
    compact_w = max(row_widths)
    return int(compact_h), int(compact_w), row_heights, col_gaps_by_row, row_gaps


def _build_input(
    canvas_shape: tuple[int, int],
    bg: int,
    rows: list[list[dict]],
    row_heights: list[int],
    col_gaps_by_row: list[list[int]],
    row_gaps: list[int],
) -> np.ndarray:
    height, width = map(int, canvas_shape)
    grid = np.full((height, width), bg, dtype=int)

    compact_h = sum(row_heights) + sum(row_gaps)
    top = random.randint(0, height - compact_h)
    row_top = int(top)
    for row_idx, row in enumerate(rows):
        row_height = int(row_heights[row_idx])
        row_width = sum(int(obj["shape"][1]) for obj in row) + sum(col_gaps_by_row[row_idx])
        left = random.randint(0, width - row_width)
        cursor = int(left)
        for obj_idx, obj in enumerate(row):
            h, w = map(int, obj["shape"])
            patch = np.full((h, w), int(obj["body"]), dtype=int)
            nxt = obj["next"]
            if nxt is not None:
                mr = random.randint(1, h - 2)
                mc = random.randint(1, w - 2)
                patch[mr, mc] = int(nxt)
            local_top = row_top + random.randint(0, row_height - h)
            _paste(grid, patch, local_top, cursor)
            if obj_idx < len(row) - 1:
                cursor += w + int(col_gaps_by_row[row_idx][obj_idx])
        if row_idx < len(rows) - 1:
            row_top += row_height + int(row_gaps[row_idx])

    return grid


def _gen_attempt() -> np.ndarray | None:
    object_count = random.randint(*OBJECT_COUNT_RANGE)
    bg = random.choice(PALETTE)
    body_colors = random.sample([color for color in PALETTE if color != bg], object_count)
    chain_colors = body_colors[:]
    random.shuffle(chain_colors)

    objects: list[dict] = []
    for idx, body in enumerate(chain_colors):
        h = random.randint(*RECT_SIDE_RANGE)
        w = random.randint(*RECT_SIDE_RANGE)
        nxt = chain_colors[idx + 1] if idx + 1 < object_count else None
        objects.append({"body": int(body), "next": None if nxt is None else int(nxt), "shape": (h, w)})

    output_h = 1 + sum(int(obj["shape"][0]) - 1 for obj in objects)
    output_w = 1 + sum(int(obj["shape"][1]) - 1 for obj in objects)
    if output_h > 30 or output_w > 30:
        return None

    placement_objects = objects[:]
    while True:
        random.shuffle(placement_objects)
        if [int(obj["body"]) for obj in placement_objects] != chain_colors:
            break

    row_count = random.randint(2, min(3, object_count))
    row_sizes = _random_composition(object_count, row_count)
    rows: list[list[dict]] = []
    offset = 0
    for size in row_sizes:
        rows.append(placement_objects[offset : offset + size])
        offset += size

    compact_h, compact_w, row_heights, col_gaps_by_row, row_gaps = _input_layout_dims(rows)
    min_h = max(output_h, compact_h)
    min_w = max(output_w, compact_w)
    if min_h > 30 or min_w > 30:
        return None

    height = random.randint(min_h, min(30, min_h + CANVAS_SLACK_MAX))
    width = random.randint(min_w, min(30, min_w + CANVAS_SLACK_MAX))

    grid = _build_input((height, width), bg, rows, row_heights, col_gaps_by_row, row_gaps)
    expected = _expected_output((height, width), bg, objects)

    if _get_bg(grid) != bg:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
