from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
MARKER = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

RECT_COUNT_OPTIONS = (2, 3)
GRID_ROWS_BY_COUNT = {2: (13, 20), 3: (16, 22)}
GRID_COLS_BY_COUNT = {2: (15, 22), 3: (18, 26)}
RECT_H_RANGE = (5, 9)
RECT_W_RANGE = (5, 10)
GRID_BORDER = 1
LEFT_MARGIN = 2
RECT_GAP = 1
GEN_RETRY_LIMIT = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _x_mask(height: int, width: int) -> np.ndarray:
    """Mirror the 8cb8642d interior-X primitive, including the center connector."""
    assert height > 0 and width > 0
    if height > width:
        return _x_mask(width, height).T

    mask = np.zeros((height, width), dtype=bool)
    center_rows = {(height - 1) // 2, height // 2}
    for r in range(height):
        d = min(r, height - 1 - r)
        left = d
        right = width - 1 - d
        assert left <= right
        if r in center_rows:
            mask[r, left : right + 1] = True
        else:
            mask[r, left] = True
            mask[r, right] = True
    return mask


def _analyze_rectangles(grid: np.ndarray) -> list[dict[str, int | tuple[int, int, int, int]]]:
    work = np.asarray(grid, dtype=int).copy()
    work[:, 0] = BACKGROUND
    comps = _components(work != BACKGROUND)
    assert 2 <= len(comps) <= 3, "expected 2 or 3 rectangles"

    rects = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        block = work[r0 : r1 + 1, c0 : c1 + 1]
        h, w = block.shape
        assert h >= 5 and w >= 5, "rectangle must be at least 5x5"
        assert np.all(block != BACKGROUND), "rectangle component must fill its bbox"

        border = np.concatenate(
            [block[0, :], block[-1, :], block[1:-1, 0], block[1:-1, -1]]
        )
        frame_colors = np.unique(border)
        assert frame_colors.size == 1, "rectangle border must be monochrome"
        frame = int(frame_colors[0])

        marker_coords = np.argwhere(block == MARKER)
        assert marker_coords.shape == (1, 2), "each rectangle must contain one grey marker"
        mr, mc = map(int, marker_coords[0])
        assert 0 < mr < h - 1 and 0 < mc < w - 1, "marker must be strictly inside"

        other_colors = set(map(int, np.unique(block)))
        other_colors.discard(frame)
        assert other_colors == {MARKER}, "rectangle may only differ from frame at the grey marker"

        key_row = r0 + mr
        key_color = int(grid[key_row, 0])
        assert key_color not in (BACKGROUND, MARKER), "marker row must have a non-grey key color"

        rects.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "frame": frame,
                "key_row": key_row,
                "key_color": key_color,
            }
        )

    key_rows = [int(rect["key_row"]) for rect in rects]
    assert len(set(key_rows)) == len(rects), "rectangle markers must lie on distinct keyed rows"
    return sorted(rects, key=lambda rect: (int(rect["r0"]), int(rect["c0"])))


def _build_output(shape: tuple[int, int], rects: list[dict[str, int | tuple[int, int, int, int]]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for rect in rects:
        r0 = int(rect["r0"])
        c0 = int(rect["c0"])
        r1 = int(rect["r1"])
        c1 = int(rect["c1"])
        frame = int(rect["frame"])
        key_color = int(rect["key_color"])

        out[r0, c0 : c1 + 1] = frame
        out[r1, c0 : c1 + 1] = frame
        out[r0 : r1 + 1, c0] = frame
        out[r0 : r1 + 1, c1] = frame

        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[:] = key_color
        interior[_x_mask(r1 - r0 - 1, c1 - c0 - 1)] = MARKER
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Hollow each filled rectangle, draw a grey X inside it, and fill the remaining interior by the key color from the marker's row.

    Input structure:
    - The background is black, and column 0 contains sparse colored key pixels.
    - Away from column 0 there are 2 or 3 solid rectangular blocks, each with a uniform border color and one grey marker pixel strictly inside.
    - For every rectangle, the marker's absolute row has exactly one non-black key pixel in column 0.

    Rule:
    - For each solid rectangle, keep its 1-cell border and erase its interior to black.
    - Inside that hollow interior, draw the canonical 45-degree X pattern in grey, including the center connector where the two diagonals meet.
    - Read the marker row's column-0 key color and recolor every remaining black interior pixel of that rectangle with that key color.
    - After all rectangles are processed, clear the entire first column to black.

    Key insight:
    - The grey marker is a dual-purpose control: it identifies the rectangle that gets the X primitive and, through its row index, selects a separate fill color from column 0.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    rects = _analyze_rectangles(g)
    return _build_output(g.shape, rects)


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = RECT_GAP) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _size_pattern_ok(sizes: list[tuple[int, int]]) -> bool:
    has_odd_odd = any((h % 2 == 1) and (w % 2 == 1) for h, w in sizes)
    has_even = any((h % 2 == 0) or (w % 2 == 0) for h, w in sizes)
    has_nonsquare = any(h != w for h, w in sizes)
    has_tall = any(h > w for h, w in sizes)
    has_wide = any(w > h for h, w in sizes)
    return has_odd_odd and has_even and has_nonsquare and (len(sizes) == 2 or (has_tall and has_wide))


def _choose_sizes(h: int, w: int, n_rects: int) -> list[tuple[int, int]] | None:
    max_h = min(RECT_H_RANGE[1], h - 2 * GRID_BORDER)
    max_w = min(RECT_W_RANGE[1], w - LEFT_MARGIN - GRID_BORDER)
    if max_h < RECT_H_RANGE[0] or max_w < RECT_W_RANGE[0]:
        return None

    area_cap = int(h * w * (0.34 if n_rects == 2 else 0.42))
    for _ in range(120):
        sizes = [
            (
                random.randint(RECT_H_RANGE[0], max_h),
                random.randint(RECT_W_RANGE[0], max_w),
            )
            for _ in range(n_rects)
        ]
        if sum(sh * sw for sh, sw in sizes) > area_cap:
            continue
        if _size_pattern_ok(sizes):
            return sizes
    return None


def _place_rectangles(h: int, w: int, sizes: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
    placed: dict[int, tuple[int, int, int, int]] = {}

    def search(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        rh, rw = sizes[idx]
        candidates = []
        for r0 in range(GRID_BORDER, h - rh - GRID_BORDER + 1):
            for c0 in range(LEFT_MARGIN, w - rw - GRID_BORDER + 1):
                box = (r0, c0, r0 + rh - 1, c0 + rw - 1)
                if all(_boxes_separated(box, other) for other in placed.values()):
                    candidates.append(box)
        random.shuffle(candidates)
        for box in candidates:
            placed[idx] = box
            if search(pos + 1):
                return True
            del placed[idx]
        return False

    if not search(0):
        return None
    return [placed[i] for i in range(len(sizes))]


def _assign_marker_rows(boxes: list[tuple[int, int, int, int]]) -> list[int] | None:
    choices = [list(range(r0 + 1, r1)) for r0, _c0, r1, _c1 in boxes]
    order = sorted(range(len(boxes)), key=lambda i: len(choices[i]))
    rows: dict[int, int] = {}
    used: set[int] = set()

    def search(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        candidates = choices[idx][:]
        random.shuffle(candidates)
        for row in candidates:
            if row in used:
                continue
            rows[idx] = row
            used.add(row)
            if search(pos + 1):
                return True
            used.remove(row)
            del rows[idx]
        return False

    if not search(0):
        return None
    return [rows[i] for i in range(len(boxes))]


def _build_case() -> tuple[np.ndarray, list[dict[str, int]]] | None:
    n_rects = random.choice(RECT_COUNT_OPTIONS)
    h = random.randint(*GRID_ROWS_BY_COUNT[n_rects])
    w = random.randint(*GRID_COLS_BY_COUNT[n_rects])

    sizes = _choose_sizes(h, w, n_rects)
    if sizes is None:
        return None
    boxes = _place_rectangles(h, w, sizes)
    if boxes is None:
        return None
    marker_rows = _assign_marker_rows(boxes)
    if marker_rows is None:
        return None

    colors = PALETTE[:]
    random.shuffle(colors)
    frame_colors = colors[:n_rects]
    key_colors = colors[n_rects : 2 * n_rects]

    grid = np.zeros((h, w), dtype=int)
    specs: list[dict[str, int]] = []
    for i, box in enumerate(boxes):
        r0, c0, r1, c1 = box
        frame = frame_colors[i]
        key_color = key_colors[i]
        marker_row = marker_rows[i]
        marker_col = c0 + (c1 - c0) // 2

        grid[r0 : r1 + 1, c0 : c1 + 1] = frame
        grid[marker_row, marker_col] = MARKER
        grid[marker_row, 0] = key_color

        specs.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "frame": frame,
                "key_row": marker_row,
                "key_color": key_color,
            }
        )

    return grid, specs


def _validate_case(grid: np.ndarray, specs: list[dict[str, int]]) -> bool:
    h, w = grid.shape
    if max(h, w) > 30:
        return False
    if len(specs) not in (2, 3):
        return False

    key_rows = [spec["key_row"] for spec in specs]
    key_colors = [spec["key_color"] for spec in specs]
    frame_colors = [spec["frame"] for spec in specs]
    if len(set(key_rows)) != len(key_rows):
        return False
    if len(set(key_colors)) != len(key_colors):
        return False
    if len(set(frame_colors)) != len(frame_colors):
        return False
    if set(key_colors) & set(frame_colors):
        return False

    if np.count_nonzero(grid[:, 0]) != len(specs):
        return False

    output = main(grid)
    expected = _build_output(grid.shape, specs)
    if not np.array_equal(output, expected):
        return False
    if np.array_equal(output, grid):
        return False

    if np.count_nonzero(output == BACKGROUND) < (h * w) // 3:
        return False

    has_odd_odd = False
    has_even = False
    has_tall = False
    has_wide = False
    for spec in specs:
        r0, c0, r1, c1 = spec["r0"], spec["c0"], spec["r1"], spec["c1"]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        xmask = _x_mask(inner_h, inner_w)
        fill_count = inner_h * inner_w - int(xmask.sum())
        if fill_count <= 0 or np.all(xmask):
            return False
        interior = output[r0 + 1 : r1, c0 + 1 : c1]
        if int(np.sum(interior == spec["key_color"])) != fill_count:
            return False
        if int(np.sum(interior == MARKER)) != int(xmask.sum()):
            return False
        if spec["frame"] == spec["key_color"] or spec["frame"] == MARKER:
            return False
        has_odd_odd |= (inner_h % 2 == 1) and (inner_w % 2 == 1)
        has_even |= (inner_h % 2 == 0) or (inner_w % 2 == 0)
        has_tall |= inner_h > inner_w
        has_wide |= inner_w > inner_h

    if not (has_odd_odd and has_even):
        return False
    if len(specs) == 3 and not (has_tall and has_wide):
        return False

    parsed = _analyze_rectangles(grid)
    if len(parsed) != len(specs):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        built = _build_case()
        if built is None:
            continue
        grid, specs = built
        if _validate_case(grid, specs):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
