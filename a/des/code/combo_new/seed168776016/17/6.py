from __future__ import annotations

import random

import numpy as np


BG = 0
SEP_COLOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 300


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(mask)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _find_separator_row(grid: np.ndarray) -> tuple[int, int]:
    rows = [
        int(r)
        for r in range(grid.shape[0])
        if int(grid[r, 0]) != BG and bool(np.all(grid[r, :] == grid[r, 0]))
    ]
    assert len(rows) == 1, "expected one full-width non-background separator row"
    row = rows[0]
    return row, int(grid[row, 0])


def _paint_mask(grid: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> None:
    h, w = mask.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    assert patch.shape == mask.shape, "paint target must be in bounds"
    assert not np.any((patch != BG) & mask), "painted objects may not overlap"
    patch[mask] = int(color)


def _extract_top_shapes(top: np.ndarray, sep_color: int) -> dict[int, tuple[int, int, np.ndarray]]:
    colors = sorted(set(map(int, np.unique(top))) - {BG, int(sep_color)})
    shapes: dict[int, tuple[int, int, np.ndarray]] = {}
    for color in colors:
        mask = top == int(color)
        comps = _components4(mask)
        assert len(comps) == 1, "each top color must form exactly one 4-connected shape"
        cropped = _crop_mask(mask)
        r0, c0, _r1, _c1 = _bbox(mask)
        shapes[int(color)] = (int(r0), int(c0), cropped)
    return shapes


def _extract_bottom_templates(bottom: np.ndarray, sep_color: int) -> dict[int, np.ndarray]:
    colors = sorted(set(map(int, np.unique(bottom))) - {BG, int(sep_color)})
    templates: dict[int, np.ndarray] = {}
    for color in colors:
        mask = bottom == int(color)
        cropped = _crop_mask(mask)
        assert cropped[0, 0], "template bbox must start at a colored pixel"
        assert int(cropped.sum()) >= 2, "template must contain at least two colored pixels"
        assert cropped.shape[0] <= 4 and cropped.shape[1] <= 4, "template bbox must be at most 4x4"
        templates[int(color)] = cropped
    return templates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate each top shape into an array whose placement pattern is encoded by the same-colored template below.

    Input structure:
    - The background is black.
    - One full-width grey row separates a top sprite region from a lower template region.
    - Above the separator, each non-black color forms exactly one 4-connected shape, and different shape bounding boxes do not overlap.
    - Below the separator, each of those colors appears as a small template mask on black, with a bounding box no larger than 4x4.

    Rule:
    - Pair every top shape with the lower template of the same color.
    - Crop the shape to its bounding box and crop the template to its bounding box.
    - For every colored template pixel at relative coordinate (dr, dc), stamp one copy of the shape with its bounding-box top-left shifted by dr * (H + 1) rows and dc * (W + 1) columns, where H and W are the shape bbox dimensions.
    - Clear the separator row and the entire lower template region to black in the output.

    Key insight:
    - The lower pattern is not copied directly; it is a placement instruction map whose pixel coordinates are expanded by the size of the matched top shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    sep_row, sep_color = _find_separator_row(g)
    assert 0 < sep_row < g.shape[0] - 1, "separator must split the grid into non-empty top and bottom regions"

    top = g[:sep_row, :]
    bottom = g[sep_row + 1 :, :]
    shapes = _extract_top_shapes(top, sep_color)
    templates = _extract_bottom_templates(bottom, sep_color)
    assert shapes and templates and set(shapes) == set(templates), "top shapes and bottom templates must match by color"

    out = np.zeros_like(g)
    top_h, top_w = top.shape

    for color, (base_r, base_c, shape_mask) in shapes.items():
        template_mask = templates[color]
        assert template_mask[0, 0], "the original shape must correspond to the template origin"
        shape_h, shape_w = shape_mask.shape
        for dr, dc in np.argwhere(template_mask):
            r0 = int(base_r) + int(dr) * (shape_h + 1)
            c0 = int(base_c) + int(dc) * (shape_w + 1)
            assert r0 + shape_h <= top_h and c0 + shape_w <= top_w, "stamped copy must stay in the top region"
            _paint_mask(out, shape_mask, color, r0, c0)

    return out


def _shape_dims(area: int) -> list[tuple[int, int]]:
    dims = []
    for h in range(1, 5):
        for w in range(1, 5):
            if area > h * w:
                continue
            if h * w > area + 5:
                continue
            dims.append((h, w))
    return dims


def _random_shape_mask() -> np.ndarray:
    area = random.randint(3, 12)
    dims = _shape_dims(area)
    assert dims
    h, w = random.choice(dims)
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return _random_shape_mask()
        cells.add(random.choice(frontier))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    mask = _crop_mask(mask)
    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    return np.asarray(mask, dtype=bool)


def _random_template_mask() -> np.ndarray:
    while True:
        h = random.randint(1, 4)
        w = random.randint(1, 4)
        if h * w < 2:
            continue
        cells = {(0, 0)}
        candidates = [(r, c) for r in range(h) for c in range(w) if (r, c) != (0, 0)]
        count = random.randint(2, min(h * w, 6))
        cells.update(random.sample(candidates, count - 1))
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return _crop_mask(mask)


def _output_patch(shape_mask: np.ndarray, template_mask: np.ndarray) -> np.ndarray:
    h, w = shape_mask.shape
    coords = np.argwhere(template_mask)
    max_r = int(coords[:, 0].max())
    max_c = int(coords[:, 1].max())
    out = np.zeros((max_r * (h + 1) + h, max_c * (w + 1) + w), dtype=bool)
    for dr, dc in coords:
        r0 = int(dr) * (h + 1)
        c0 = int(dc) * (w + 1)
        patch = out[r0 : r0 + h, c0 : c0 + w]
        assert not np.any(patch & shape_mask), "copies should not overlap inside one output patch"
        patch |= shape_mask
    return out


def _random_partition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    prev = 0
    sizes = []
    for cut in cuts + [total]:
        sizes.append(cut - prev)
        prev = cut
    return sizes


def _pack_rectangles(
    rects: list[tuple[int, int]],
    *,
    max_width: int,
    max_height: int,
    max_rows: int,
    margin_range: tuple[int, int],
    gap_range: tuple[int, int],
    attempts: int = 200,
) -> tuple[int, int, list[tuple[int, int]]] | None:
    count = len(rects)
    assert count > 0
    for _ in range(attempts):
        row_count = random.randint(1, min(max_rows, count))
        order = list(range(count))
        random.shuffle(order)
        groups: list[list[int]] = []
        pos = 0
        for size in _random_partition(count, row_count):
            groups.append(order[pos : pos + size])
            pos += size

        top = random.randint(*margin_range)
        bottom = random.randint(*margin_range)
        left = random.randint(*margin_range)
        right = random.randint(*margin_range)
        row_gap = random.randint(*gap_range) if row_count > 1 else 0
        row_heights = [max(rects[i][0] for i in group) for group in groups]
        height = top + bottom + sum(row_heights) + row_gap * (row_count - 1)
        if height > max_height:
            continue

        positions: list[tuple[int, int] | None] = [None] * count
        width = 0
        y = top
        for group, row_height in zip(groups, row_heights, strict=False):
            gaps = [random.randint(*gap_range) for _ in range(max(0, len(group) - 1))]
            x = left
            for i, rect_idx in enumerate(group):
                rect_h, rect_w = rects[rect_idx]
                dy = random.randint(0, row_height - rect_h)
                positions[rect_idx] = (y + dy, x)
                x += rect_w
                if i < len(gaps):
                    x += gaps[i]
            width = max(width, x + right)
            y += row_height + row_gap

        assert all(pos is not None for pos in positions)
        if width <= max_width:
            return height, width, [(int(r), int(c)) for r, c in positions]
    return None


def _random_object(color: int) -> dict[str, object]:
    for _ in range(200):
        shape_mask = _random_shape_mask()
        template_mask = _random_template_mask()
        patch = _output_patch(shape_mask, template_mask)
        if patch.shape[0] > 11 or patch.shape[1] > 11:
            continue
        return {
            "color": int(color),
            "shape": shape_mask,
            "template": template_mask,
            "patch": patch,
        }
    raise RuntimeError("failed to sample a compact object-template pair")


def _build_example() -> tuple[np.ndarray, np.ndarray] | None:
    count = random.randint(2, 4)
    colors = random.sample(PALETTE, count)
    objects = [_random_object(color) for color in colors]

    top_rects = [tuple(map(int, obj["patch"].shape)) for obj in objects]
    top_layout = _pack_rectangles(
        top_rects,
        max_width=30,
        max_height=23,
        max_rows=2,
        margin_range=(0, 2),
        gap_range=(1, 2),
    )
    if top_layout is None:
        return None
    top_h0, top_w0, top_positions = top_layout

    template_rects = [tuple(map(int, obj["template"].shape)) for obj in objects]
    bottom_layout = _pack_rectangles(
        template_rects,
        max_width=30,
        max_height=6,
        max_rows=1,
        margin_range=(1, 2),
        gap_range=(1, 3),
    )
    if bottom_layout is None:
        return None
    bottom_h0, bottom_w0, bottom_positions = bottom_layout

    base_width = max(top_w0, bottom_w0)
    base_height = top_h0 + 1 + bottom_h0
    if base_height > 30 or base_width > 30:
        return None

    extra_w = random.randint(0, min(3, 30 - base_width))
    extra_h = random.randint(0, min(3, 30 - base_height))
    final_w = base_width + extra_w

    top_extra = random.randint(0, extra_h)
    bottom_extra = extra_h - top_extra
    final_top_h = top_h0 + top_extra
    final_bottom_h = bottom_h0 + bottom_extra
    sep_row = final_top_h

    top_x_shift = random.randint(0, final_w - top_w0)
    bottom_x_shift = random.randint(0, final_w - bottom_w0)
    top_y_shift = random.randint(0, final_top_h - top_h0)
    bottom_y_shift = random.randint(0, final_bottom_h - bottom_h0)

    inp = np.zeros((final_top_h + 1 + final_bottom_h, final_w), dtype=int)
    out = np.zeros_like(inp)
    inp[sep_row, :] = SEP_COLOR

    for obj, (r0, c0) in zip(objects, top_positions, strict=False):
        color = int(obj["color"])
        shape_mask = np.asarray(obj["shape"], dtype=bool)
        patch_mask = np.asarray(obj["patch"], dtype=bool)
        shape_r = top_y_shift + int(r0)
        shape_c = top_x_shift + int(c0)
        _paint_mask(inp, shape_mask, color, shape_r, shape_c)
        _paint_mask(out, patch_mask, color, shape_r, shape_c)

    bottom_origin = sep_row + 1
    for obj, (r0, c0) in zip(objects, bottom_positions, strict=False):
        color = int(obj["color"])
        template_mask = np.asarray(obj["template"], dtype=bool)
        rr = bottom_origin + bottom_y_shift + int(r0)
        cc = bottom_x_shift + int(c0)
        _paint_mask(inp, template_mask, color, rr, cc)

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        example = _build_example()
        if example is None:
            continue
        inp, out = example
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("failed to generate a valid input")
