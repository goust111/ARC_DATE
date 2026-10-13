import numpy as np


BG = 0
BLUE = 1
GREEN = 3
RECT_COLORS = [2, 4, 5, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _area(rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = rect
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] <= inner[0] <= inner[2] <= outer[2] and outer[1] <= inner[1] <= inner[3] <= outer[3]


def _inset(rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return rect[0] + 1, rect[1] + 1, rect[2] - 1, rect[3] - 1


def _side_matches(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    return int(a[0] == b[0]) + int(a[1] == b[1]) + int(a[2] == b[2]) + int(a[3] == b[3])


def _paint_border(grid: np.ndarray, rect: tuple[int, int, int, int], color: int | bool) -> None:
    r0, c0, r1, c1 = rect
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _inside(rect: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = rect
    return r0 <= r <= r1 and c0 <= c <= c1


def _complete_layers(grid: np.ndarray) -> list[dict[str, int | tuple[int, int, int, int]]]:
    layers: list[dict[str, int | tuple[int, int, int, int]]] = []
    colors = sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, BLUE, GREEN))
    for color in colors:
        rect = _bbox(grid == color)
        assert rect is not None
        layers.append({"color": color, "raw": rect, "rect": rect, "area": _area(rect)})

    layers.sort(key=lambda layer: int(layer["area"]), reverse=True)
    completed: list[dict[str, int | tuple[int, int, int, int]]] = []
    for layer in layers:
        raw = layer["raw"]
        assert isinstance(raw, tuple)
        parents = [parent for parent in completed if _contains(parent["rect"], raw) and parent["rect"] != raw]
        if parents:
            parent = min(parents, key=lambda item: _area(item["rect"]))
            parent_inset = _inset(parent["rect"])
            if parent_inset[0] <= parent_inset[2] and parent_inset[1] <= parent_inset[3] and _contains(parent_inset, raw):
                if _side_matches(raw, parent_inset) >= 3:
                    layer["rect"] = parent_inset
        completed.append(layer)
    return layers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the folded paths' center-column trace and recolor it by nested-rectangle depth.

    Input structure:
    - A black odd-width grid contains 2-3 colors of sparse pixels, each color hinting one nested axis-aligned rectangle boundary.
    - Green marker pixels sit away from the center column and blue is absent from the input.
    - The rectangle colors are distinct from the marker color and are nested from outer to inner.

    Rule:
    - Complete each sparse rectangle color into its full boundary using its bbox, with the same 3-side inset expansion used for sparse nested rectangles.
    - For every green marker, paint a blue folded route: fill its whole column, then connect that column's top and bottom ends to opposite top/bottom corners according to whether the marker is left or right of the center column.
    - Erase every blue pixel outside the center column, then recolor the remaining center-column blue pixels by the deepest completed rectangle containing that cell; if none contains it, keep it blue.

    Key insight:
    - The markers decide which center-axis cells are activated by the folded routes, while the completed nesting decides the color of those surviving center-axis cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert w % 2 == 1
    mid = w // 2

    layers = _complete_layers(g)
    markers = np.argwhere(g == GREEN)

    path = np.zeros_like(g, dtype=bool)
    for _r, c in markers:
        c = int(c)
        path[:, c] = True
        if c < mid:
            path[0, c:] = True
            path[h - 1, : c + 1] = True
        elif c > mid:
            path[0, : c + 1] = True
            path[h - 1, c:] = True
        else:
            path[0, c:] = True
            path[h - 1, : c + 1] = True

    out = np.zeros_like(g)
    for layer in layers:
        rect = layer["rect"]
        assert isinstance(rect, tuple)
        _paint_border(out, rect, int(layer["color"]))

    for r in range(h):
        if not path[r, mid]:
            continue
        color = BLUE
        for layer in reversed(layers):
            rect = layer["rect"]
            assert isinstance(rect, tuple)
            if _inside(rect, r, mid):
                color = int(layer["color"])
                break
        out[r, mid] = color

    out[g == GREEN] = GREEN
    return out


def _choose_rectangles(rng: np.random.Generator, h: int, w: int) -> list[tuple[int, int, int, int]]:
    mid = w // 2
    num_layers = int(rng.integers(2, 4))
    mode = str(rng.choice(["top", "bottom"]))
    touch_far_edge = bool(rng.integers(0, 2))

    for _ in range(200):
        left_gaps = [int(rng.integers(1, 3)) for _ in range(num_layers - 1)]
        right_gaps = [int(rng.integers(1, 3)) for _ in range(num_layers - 1)]
        total_left = sum(left_gaps)
        total_right = sum(right_gaps)
        min_c0_max = mid - total_left - 2
        max_c1_min = mid + total_right + 2
        if min_c0_max < 1 or max_c1_min > w - 2:
            continue
        c0 = int(rng.integers(1, min_c0_max + 1))
        c1 = int(rng.integers(max_c1_min, w - 1))
        if c1 - c0 + 1 < 5:
            continue

        vertical_gaps = [int(rng.integers(1, 4)) for _ in range(num_layers - 1)]
        if mode == "top":
            outer_r0 = 0
            outer_r1_min = 4 + sum(vertical_gaps)
            outer_r1_max = h - 1 if touch_far_edge else h - 2
            if outer_r1_min > outer_r1_max:
                continue
            r0 = outer_r0
            r1 = int(rng.integers(outer_r1_min, outer_r1_max + 1))
            rects = [(r0, c0, r1, c1)]
            cur_c0, cur_c1, cur_r1 = c0, c1, r1
            valid = True
            for i in range(num_layers - 1):
                cur_c0 += left_gaps[i]
                cur_c1 -= right_gaps[i]
                cur_r1 -= vertical_gaps[i]
                if cur_c1 - cur_c0 + 1 < 3 or cur_r1 - r0 + 1 < 3 or not (cur_c0 < mid < cur_c1):
                    valid = False
                    break
                rects.append((r0, cur_c0, cur_r1, cur_c1))
            if valid:
                return rects
        else:
            outer_r1 = h - 1
            outer_r0_max = 0 if touch_far_edge else 1
            outer_r0_min = 0
            r0 = int(rng.integers(outer_r0_min, outer_r0_max + 1))
            cur_r0 = r0
            if outer_r1 - cur_r0 + 1 < 4 + sum(vertical_gaps):
                continue
            rects = [(cur_r0, c0, outer_r1, c1)]
            cur_c0, cur_c1 = c0, c1
            valid = True
            for i in range(num_layers - 1):
                cur_c0 += left_gaps[i]
                cur_c1 -= right_gaps[i]
                cur_r0 += vertical_gaps[i]
                if cur_c1 - cur_c0 + 1 < 3 or outer_r1 - cur_r0 + 1 < 3 or not (cur_c0 < mid < cur_c1):
                    valid = False
                    break
                rects.append((cur_r0, cur_c0, outer_r1, cur_c1))
            if valid:
                return rects
    raise RuntimeError("failed to sample nested rectangles")


def _sample_side_points(cells: list[tuple[int, int]], rng: np.random.Generator, count: int) -> list[tuple[int, int]]:
    picks = rng.choice(len(cells), size=count, replace=False)
    return [cells[int(i)] for i in np.atleast_1d(picks)]


def _sample_rectangle(rect: tuple[int, int, int, int], mid: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    top = [(r0, c) for c in range(c0, c1 + 1) if c != mid]
    bottom = [(r1, c) for c in range(c0, c1 + 1) if c != mid]
    left = [(r, c0) for r in range(r0, r1 + 1)]
    right = [(r, c1) for r in range(r0, r1 + 1)]

    pts = set()
    for side in (top, bottom, left, right):
        need = 2 if len(side) >= 2 else 1
        pts.update(_sample_side_points(side, rng, need))

    border = set(top) | set(bottom) | set(left) | set(right)
    extras = [cell for cell in border if cell not in pts and cell[1] != mid]
    if extras:
        extra_count = int(rng.integers(0, min(4, len(extras)) + 1))
        if extra_count:
            pts.update(_sample_side_points(extras, rng, extra_count))
    return list(pts)


def _cells_between_rectangles(rects: list[tuple[int, int, int, int]], h: int, w: int) -> list[tuple[int, int]]:
    layer_masks = []
    for rect in rects:
        r0, c0, r1, c1 = rect
        mask = np.zeros((h, w), dtype=bool)
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
        layer_masks.append(mask)

    border_mask = np.zeros((h, w), dtype=bool)
    for rect in rects:
        _paint_border(border_mask, rect, True)

    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if border_mask[r, c]:
                continue
            depth = sum(int(mask[r, c]) for mask in layer_masks)
            if depth == 0 or 0 < depth < len(rects):
                candidates.append((r, c))
    return candidates

def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    h, w = inp.shape
    mid = w // 2
    if np.any(inp == BLUE):
        return False
    if np.any((out == BLUE) & (np.arange(w)[None, :] != mid)):
        return False
    if np.count_nonzero(inp == GREEN) < 2:
        return False
    if not np.any(inp[:, :mid] == GREEN) or not np.any(inp[:, mid + 1 :] == GREEN):
        return False
    if out[0, mid] == BG and out[h - 1, mid] == BG:
        return False
    if not np.array_equal(out[inp == GREEN], np.full(np.count_nonzero(inp == GREEN), GREEN)):
        return False
    rect_colors = [int(v) for v in np.unique(inp) if int(v) not in (BG, BLUE, GREEN)]
    if len(rect_colors) < 2 or len(rect_colors) > 3:
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        h = int(rng.choice([13, 15, 17]))
        w = int(rng.choice([13, 15, 17]))
        mid = w // 2
        rects = _choose_rectangles(rng, h, w)
        colors = [int(c) for c in rng.choice(RECT_COLORS, size=len(rects), replace=False)]

        grid = np.zeros((h, w), dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        for rect, color in zip(rects, colors):
            for r, c in _sample_rectangle(rect, mid, rng):
                if occupied[r, c]:
                    break
                grid[r, c] = color
                occupied[r, c] = True
            else:
                continue
            break
        else:
            candidates = [
                (r, c)
                for r, c in _cells_between_rectangles(rects, h, w)
                if 0 < r < h - 1 and 0 < c < w - 1 and c != mid and grid[r, c] == BG
            ]
            rows = {}
            for r, c in candidates:
                rows.setdefault(r, []).append((r, c))
            usable_rows = [r for r, cells in rows.items() if any(c < mid for _, c in cells) and any(c > mid for _, c in cells)]
            if len(usable_rows) < 2:
                continue

            marker_count = int(rng.choice([2, 2, 3, 4]))
            chosen_rows = list(rng.choice(usable_rows, size=min(marker_count, len(usable_rows)), replace=False))
            if len(chosen_rows) < 2:
                continue
            left_row = int(chosen_rows[0])
            right_row = int(chosen_rows[1])
            left_cells = [cell for cell in rows[left_row] if cell[1] < mid]
            right_cells = [cell for cell in rows[right_row] if cell[1] > mid]
            if not left_cells or not right_cells:
                continue
            lr, lc = left_cells[int(rng.integers(len(left_cells)))]
            rr, rc = right_cells[int(rng.integers(len(right_cells)))]
            grid[lr, lc] = GREEN
            grid[rr, rc] = GREEN

            for row in chosen_rows[2:]:
                row = int(row)
                side = str(rng.choice(["left", "right"]))
                cells = [cell for cell in rows[row] if (cell[1] < mid) == (side == "left")]
                if not cells:
                    cells = rows[row]
                mr, mc = cells[int(rng.integers(len(cells)))]
                grid[mr, mc] = GREEN

            out = main(grid)
            if _semantic_checks(grid, out):
                return grid
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPTS} attempts")
