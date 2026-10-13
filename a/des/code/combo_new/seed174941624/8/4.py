import random
from collections import deque

import numpy as np


BG = 0
DIVIDER = 5
MAROON = 9
LEGEND_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8]
GENERATION_RETRIES = 300
RIGHT_MAX_H = 23
RIGHT_MAX_W = 19

SHAPE_LIBRARY = [
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
]


def _most_common_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(cells)
    return components


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _cropped_mask(points: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(points)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        mask[r - r0, c - c0] = True
    return mask


def _shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    cropped = np.asarray(mask, dtype=np.uint8)
    return int(cropped.shape[0]), int(cropped.shape[1]), cropped.tobytes()


def _find_divider(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    candidates = []
    for c in range(1, w - 1):
        color = int(grid[0, c])
        if color == bg:
            continue
        if not np.all(grid[:, c] == color):
            continue
        if not np.any(grid[:, :c] != bg):
            continue
        if not np.any(grid[:, c + 1 :] != bg):
            continue
        candidates.append(c)
    assert len(candidates) == 1, "expected a unique full-height divider column"
    return candidates[0]


def _extract_legend(left: np.ndarray, bg: int) -> dict[tuple[int, int, bytes], int]:
    legend: dict[tuple[int, int, bytes], int] = {}
    for cells in _connected_components(left != bg):
        colors = {int(left[r, c]) for r, c in cells}
        assert len(colors) == 1, "legend components must be monochrome"
        color = next(iter(colors))
        mask = _cropped_mask(cells)
        key = _shape_key(mask)
        assert key not in legend, "legend shapes must be unique"
        legend[key] = color
    assert legend, "legend must contain at least one shape"
    return legend


def _interior_holes(component_mask: np.ndarray) -> list[np.ndarray]:
    h, w = component_mask.shape
    bg_mask = ~component_mask
    seen = np.zeros((h, w), dtype=bool)
    holes: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bg_mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            touches_border = False
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if bg_mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            if not touches_border:
                holes.append(_cropped_mask(cells))
    return holes


def _exterior_background(bg_mask: np.ndarray) -> np.ndarray:
    h, w = bg_mask.shape
    outside = np.zeros((h, w), dtype=bool)
    queue = deque()
    for r in range(h):
        for c in (0, w - 1):
            if bg_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if bg_mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                queue.append((nr, nc))
    return outside


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each maroon blob's outer outline using the legend color of its internal hole shape.

    Input structure:
    - The background is a black grid split into a left legend area and a right main area by one full-height grey vertical divider.
    - The left area contains several separated solid colored shapes, one color per shape, acting as a shape legend.
    - The right area contains several larger maroon connected components whose interiors include one or more black holes.

    Rule:
    - Remove the legend area and divider from the output, keeping only the right main area.
    - For each maroon component, extract every enclosed black hole and identify the unique legend shape that matches the hole shape exactly up to translation only.
    - Recolor only the component's outer contour with the matched legend color, while leaving the rest of the maroon component and all holes unchanged.

    Key insight:
    - The hole shape does not tell you what to fill; it tells you which legend color should be applied to the blob's external boundary.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    divider = _find_divider(g, bg)

    left = g[:, :divider]
    right = g[:, divider + 1 :]
    legend = _extract_legend(left, bg)

    right_colors = set(map(int, np.unique(right))) - {bg}
    assert len(right_colors) == 1, "right side must contain a single foreground color"
    maroon = next(iter(right_colors))

    outside = _exterior_background(right == bg)
    out = right.copy()
    h, w = right.shape

    for cells in _connected_components(right == maroon):
        component_mask = _cropped_mask(cells)
        holes = _interior_holes(component_mask)
        assert holes, "each maroon component must contain at least one hole"
        hole_keys = {_shape_key(hole) for hole in holes}
        assert len(hole_keys) == 1, "all holes inside a blob must share the same shape"
        match_key = next(iter(hole_keys))
        assert match_key in legend, "every hole shape must appear in the legend"
        color = legend[match_key]

        for r, c in cells:
            if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                out[r, c] = color
                continue
            for nr, nc in _neighbors4(r, c, h, w):
                if outside[nr, nc]:
                    out[r, c] = color
                    break
    return out


# --------------- generator ---------------


def _add_protrusion(mask: np.ndarray) -> np.ndarray:
    if random.random() >= 0.5:
        return mask

    h, w = mask.shape
    side = random.choice(["top", "bottom", "left", "right"])
    if side in {"left", "right"}:
        protrusion_h = random.randint(2, min(4, h))
        protrusion_w = random.randint(1, 2)
        top = 0 if protrusion_h == h else random.randint(0, h - protrusion_h)
        new = np.zeros((h, w + protrusion_w), dtype=bool)
        if side == "left":
            new[:, protrusion_w:] = mask
            new[top : top + protrusion_h, :protrusion_w] = True
        else:
            new[:, :w] = mask
            new[top : top + protrusion_h, w:] = True
        return new

    protrusion_h = random.randint(1, 2)
    protrusion_w = random.randint(2, min(4, w))
    left = 0 if protrusion_w == w else random.randint(0, w - protrusion_w)
    new = np.zeros((h + protrusion_h, w), dtype=bool)
    if side == "top":
        new[protrusion_h:, :] = mask
        new[:protrusion_h, left : left + protrusion_w] = True
    else:
        new[:h, :] = mask
        new[h:, left : left + protrusion_w] = True
    return new


def _build_blob(shape: np.ndarray) -> tuple[np.ndarray, int, str]:
    sh, sw = shape.shape
    hole_count = random.choice([1, 2, 3])
    if max(sh, sw) == 3 and hole_count == 3 and random.random() < 0.5:
        hole_count = 2
    layout = random.choice(["row", "col"])
    gap = 1

    if layout == "row":
        extra_h = random.randint(0, 2)
        extra_w = random.randint(0, 1)
        required_w = hole_count * sw + (hole_count - 1) * gap
        h = sh + 2 + extra_h
        w = required_w + 2 + extra_w
        top = random.randint(1, h - sh - 1)
        left = random.randint(1, w - required_w - 1)
        starts = [(top, left + i * (sw + gap)) for i in range(hole_count)]
    else:
        extra_h = random.randint(0, 1)
        extra_w = random.randint(0, 2)
        required_h = hole_count * sh + (hole_count - 1) * gap
        h = required_h + 2 + extra_h
        w = sw + 2 + extra_w
        top = random.randint(1, h - required_h - 1)
        left = random.randint(1, w - sw - 1)
        starts = [(top + i * (sh + gap), left) for i in range(hole_count)]

    mask = np.ones((h, w), dtype=bool)
    for rr, cc in starts:
        window = mask[rr : rr + sh, cc : cc + sw]
        window[shape] = False

    return _add_protrusion(mask), hole_count, layout


def _pack_horizontal(items: list[dict]) -> tuple[int, int, list[tuple[int, int]]] | None:
    widths = [item["mask"].shape[1] for item in items]
    heights = [item["mask"].shape[0] for item in items]
    gaps = [random.randint(2, 3) for _ in range(max(0, len(items) - 1))]
    width = 1 + sum(widths) + sum(gaps) + 1
    height = max(heights) + random.randint(2, 5)
    if width > RIGHT_MAX_W or height > RIGHT_MAX_H:
        return None

    positions = []
    x = 1
    for idx, item in enumerate(items):
        h, w = item["mask"].shape
        y = random.randint(1, height - h - 1)
        positions.append((y, x))
        x += w
        if idx < len(gaps):
            x += gaps[idx]
    return height, width, positions


def _pack_vertical(items: list[dict]) -> tuple[int, int, list[tuple[int, int]]] | None:
    widths = [item["mask"].shape[1] for item in items]
    heights = [item["mask"].shape[0] for item in items]
    gaps = [random.randint(2, 3) for _ in range(max(0, len(items) - 1))]
    width = max(widths) + random.randint(2, 5)
    height = 1 + sum(heights) + sum(gaps) + 1
    if width > RIGHT_MAX_W or height > RIGHT_MAX_H:
        return None

    positions = []
    y = 1
    for idx, item in enumerate(items):
        h, w = item["mask"].shape
        x = random.randint(1, width - w - 1)
        positions.append((y, x))
        y += h
        if idx < len(gaps):
            y += gaps[idx]
    return height, width, positions


def _row_width(items: list[dict], gaps: list[int]) -> int:
    return sum(item["mask"].shape[1] for item in items) + sum(gaps)


def _pack_grid(items: list[dict]) -> tuple[int, int, list[tuple[int, int]]] | None:
    if len(items) < 2:
        return None
    split = random.randint(1, len(items) - 1)
    row1 = items[:split]
    row2 = items[split:]
    gaps1 = [random.randint(2, 3) for _ in range(max(0, len(row1) - 1))]
    gaps2 = [random.randint(2, 3) for _ in range(max(0, len(row2) - 1))]
    row_gap = random.randint(2, 3)
    row1_h = max(item["mask"].shape[0] for item in row1)
    row2_h = max(item["mask"].shape[0] for item in row2)
    row1_w = _row_width(row1, gaps1)
    row2_w = _row_width(row2, gaps2)
    width = max(row1_w, row2_w) + 2
    height = row1_h + row_gap + row2_h + 2
    if width > RIGHT_MAX_W or height > RIGHT_MAX_H:
        return None

    positions: list[tuple[int, int]] = []
    row1_left = random.randint(1, width - row1_w - 1)
    row2_left = random.randint(1, width - row2_w - 1)

    x = row1_left
    for idx, item in enumerate(row1):
        h, w = item["mask"].shape
        y = random.randint(1, row1_h - h + 1)
        positions.append((y, x))
        x += w
        if idx < len(gaps1):
            x += gaps1[idx]

    x = row2_left
    y0 = 1 + row1_h + row_gap
    for idx, item in enumerate(row2):
        h, w = item["mask"].shape
        y = y0 + random.randint(0, row2_h - h)
        positions.append((y, x))
        x += w
        if idx < len(gaps2):
            x += gaps2[idx]

    return height, width, positions


def _pack_right(items: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
    order = items[:]
    random.shuffle(order)
    packers = [_pack_horizontal, _pack_vertical, _pack_grid]
    random.shuffle(packers)

    for packer in packers:
        packed = packer(order)
        if packed is None:
            continue
        height, width, positions = packed
        right = np.full((height, width), BG, dtype=int)
        placed: list[dict] = []
        occupied = np.zeros((height, width), dtype=bool)
        ok = True
        for item, (top, left) in zip(order, positions):
            mask = item["mask"]
            h, w = mask.shape
            region = occupied[top : top + h, left : left + w]
            if np.any(region & mask):
                ok = False
                break
            right[top : top + h, left : left + w][mask] = MAROON
            region[mask] = True
            placed.append(
                {
                    "mask": mask,
                    "top": top,
                    "left": left,
                    "color": item["color"],
                }
            )
        if ok:
            return right, placed
    return None


def _paint_expected(right: np.ndarray, placed: list[dict]) -> np.ndarray:
    out = right.copy()
    outside = _exterior_background(right == BG)
    h, w = right.shape
    for item in placed:
        mask = item["mask"]
        top = item["top"]
        left = item["left"]
        color = item["color"]
        mh, mw = mask.shape
        for r in range(mh):
            for c in range(mw):
                if not mask[r, c]:
                    continue
                gr, gc = top + r, left + c
                if gr == 0 or gc == 0 or gr == h - 1 or gc == w - 1:
                    out[gr, gc] = color
                    continue
                for nr, nc in _neighbors4(gr, gc, h, w):
                    if outside[nr, nc]:
                        out[gr, gc] = color
                        break
    return out


def _embed_right(right_small: np.ndarray, placed_small: list[dict], total_h: int) -> tuple[np.ndarray, list[dict]]:
    if right_small.shape[0] == total_h:
        return right_small, placed_small

    top_offset = random.randint(0, total_h - right_small.shape[0])
    right = np.full((total_h, right_small.shape[1]), BG, dtype=int)
    right[top_offset : top_offset + right_small.shape[0], :] = right_small
    placed = []
    for item in placed_small:
        placed.append(
            {
                "mask": item["mask"],
                "top": item["top"] + top_offset,
                "left": item["left"],
                "color": item["color"],
            }
        )
    return right, placed


def _build_legend(
    shapes: list[np.ndarray],
    colors: list[int],
    total_h: int,
    max_left_w: int,
) -> np.ndarray | None:
    max_shape_w = max(shape.shape[1] for shape in shapes)
    min_left_w = max_shape_w + 2
    if min_left_w > max_left_w:
        return None

    left_w = random.randint(min_left_w, min(max_left_w, max_shape_w + 5))
    gaps = [random.randint(1, 2) for _ in range(max(0, len(shapes) - 1))]
    content_h = sum(shape.shape[0] for shape in shapes) + sum(gaps)
    if content_h > total_h:
        return None

    legend = np.full((total_h, left_w), BG, dtype=int)
    order = list(range(len(shapes)))
    random.shuffle(order)
    y = random.randint(0, total_h - content_h)
    for idx, shape_idx in enumerate(order):
        shape = shapes[shape_idx]
        color = colors[shape_idx]
        sh, sw = shape.shape
        x = random.randint(1, left_w - sw - 1)
        legend[y : y + sh, x : x + sw][shape] = color
        y += sh
        if idx < len(gaps):
            y += gaps[idx]
    return legend


def _gen_attempt() -> np.ndarray | None:
    blob_count = random.randint(2, 4)
    indices = random.sample(range(len(SHAPE_LIBRARY)), blob_count)
    shapes = [SHAPE_LIBRARY[idx].copy() for idx in indices]
    colors = random.sample(LEGEND_COLOR_POOL, blob_count)

    items = []
    for shape, color in zip(shapes, colors):
        blob_mask, _, _ = _build_blob(shape)
        items.append({"mask": blob_mask, "color": color, "shape": shape})

    packed = _pack_right(items)
    if packed is None:
        return None
    right_small, placed_small = packed

    legend_gaps = blob_count - 1
    legend_min_h = sum(shape.shape[0] for shape in shapes) + legend_gaps
    total_h = max(right_small.shape[0], legend_min_h)
    total_h += random.randint(0, min(3, 30 - total_h))
    if total_h > 30:
        return None

    right, placed = _embed_right(right_small, placed_small, total_h)
    expected = _paint_expected(right, placed)

    max_left_w = 30 - 1 - right.shape[1]
    if max_left_w < 4:
        return None
    legend = _build_legend(shapes, colors, total_h, max_left_w)
    if legend is None:
        return None

    grid = np.full((total_h, legend.shape[1] + 1 + right.shape[1]), BG, dtype=int)
    grid[:, : legend.shape[1]] = legend
    grid[:, legend.shape[1]] = DIVIDER
    grid[:, legend.shape[1] + 1 :] = right

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_RETRIES} attempts")
