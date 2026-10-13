from __future__ import annotations

import numpy as np


OUTPUT_SIZE = 15
BACKGROUND = 0
PALETTE = np.arange(1, 10, dtype=int)
GENERATION_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rectangle_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _border_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _count_components(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _extract_layers(grid: np.ndarray) -> tuple[int, list[dict[str, int | tuple[int, int, int, int]]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert 2 <= len(colors) <= 4, "expected 2 to 4 nested colored rectangles"

    layers = []
    for color in colors:
        mask = g == color
        bbox = _bbox_from_mask(mask)
        r0, c0, r1, c1 = bbox
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height >= 3 and width >= 3, "each rectangle must be at least 3x3"
        layers.append(
            {
                "color": color,
                "bbox": bbox,
                "area": height * width,
            }
        )

    layers.sort(key=lambda layer: int(layer["area"]), reverse=True)

    center_r = None
    center_c = None
    for i, layer in enumerate(layers):
        r0, c0, r1, c1 = layer["bbox"]
        if center_r is None:
            center_r = r0 + r1
            center_c = c0 + c1
        assert r0 + r1 == center_r and c0 + c1 == center_c, "rectangles must be concentric"
        if i + 1 < len(layers):
            nr0, nc0, nr1, nc1 = layers[i + 1]["bbox"]
            assert r0 < nr0 < nr1 < r1, "nested rectangles must be strictly contained"
            assert c0 < nc0 < nc1 < c1, "nested rectangles must be strictly contained"

    expected_union = np.zeros_like(g, dtype=bool)
    for i, layer in enumerate(layers):
        outer = _rectangle_mask(g.shape, layer["bbox"])
        if i + 1 < len(layers):
            inner = _rectangle_mask(g.shape, layers[i + 1]["bbox"])
            expected = outer & ~inner
        else:
            expected = outer
        color = int(layer["color"])
        assert np.array_equal(g == color, expected), "visible colors must match nested solid rectangles"
        expected_union |= expected

    assert np.array_equal(g != bg, expected_union), "foreground must be exactly the nested rectangle structure"
    return bg, layers


def _hollow_layers(shape: tuple[int, int], bg: int, layers: list[dict[str, int | tuple[int, int, int, int]]]) -> np.ndarray:
    hollowed = np.full(shape, bg, dtype=int)
    for layer in layers:
        border = _border_mask(shape, layer["bbox"])
        hollowed[border] = int(layer["color"])
    return hollowed


def _build_output(layer_count: int, line_color: int) -> np.ndarray:
    out = np.full((OUTPUT_SIZE, OUTPUT_SIZE), BACKGROUND, dtype=int)
    separators = np.arange(layer_count, OUTPUT_SIZE, layer_count + 1, dtype=int)
    assert separators.size > 0, "expected at least one separator line"
    out[separators, :] = line_color
    out[:, separators] = line_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert nested solid rectangles into a lattice whose spacing is driven by the number of hollow rings.

    Input structure:
    - The background is black.
    - The foreground is 2 to 4 strictly concentric, differently colored solid axis-aligned rectangles.
    - Each inner rectangle lies strictly inside the previous one, so the visible input is a stack of colored rectangular layers with one solid innermost rectangle.

    Rule:
    - Recover each original rectangle bbox from the visible nested layers.
    - Hollow every rectangle by keeping only its 1-pixel border, then count how many separate border rings appear.
    - Take the innermost ring color and draw a 15x15 black lattice with that color on rows and columns N, N+(N+1), N+2(N+1), ... where N is the ring count.

    Key insight:
    - The rectangles are only a parameter carrier: hollowing reveals the ring depth, and that topological depth becomes the lattice spacing while the innermost color becomes the lattice color.
    """
    bg, layers = _extract_layers(np.asarray(grid, dtype=int))
    hollowed = _hollow_layers(np.asarray(grid, dtype=int).shape, bg, layers)

    ring_count = _count_components(hollowed != bg)
    assert ring_count == len(layers), "each hollowed rectangle should form one ring component"
    line_color = int(layers[-1]["color"])
    return _build_output(ring_count, line_color)


def _parity_values(lo: int, hi: int, parity: int) -> list[int]:
    return [value for value in range(lo, hi + 1) if value % 2 == parity]


def _sample_gaps(n_layers: int, mode: int) -> tuple[list[int], list[int]]:
    gap_count = n_layers - 1
    if gap_count == 1:
        return [int(np.random.choice([1, 2, 3]))], [int(np.random.choice([1, 2, 3]))]

    if mode == 0:
        gap_r = int(np.random.choice([1, 2, 3]))
        gap_c = int(np.random.choice([1, 2, 3]))
        return [gap_r] * gap_count, [gap_c] * gap_count

    if mode == 1:
        rows = np.random.choice([1, 2, 3], size=gap_count, replace=True).astype(int).tolist()
        if len(set(rows)) == 1:
            rows[-1] = 1 if rows[-1] != 1 else 2
        gap_c = int(np.random.choice([1, 2, 3]))
        return rows, [gap_c] * gap_count

    rows = np.random.choice([1, 2, 3], size=gap_count, replace=True).astype(int).tolist()
    cols = np.random.choice([1, 2, 3], size=gap_count, replace=True).astype(int).tolist()
    if len(set(rows + cols)) == 1:
        cols[-1] = 1 if cols[-1] != 1 else 2
    return rows, cols


def _build_boxes(
    outer_bbox: tuple[int, int, int, int],
    gap_rows: list[int],
    gap_cols: list[int],
) -> list[tuple[int, int, int, int]]:
    boxes = [outer_bbox]
    r0, c0, r1, c1 = outer_bbox
    for dr, dc in zip(gap_rows, gap_cols):
        r0 += dr
        c0 += dc
        r1 -= dr
        c1 -= dc
        boxes.append((r0, c0, r1, c1))
    return boxes


def _paint_nested_rectangles(
    shape: tuple[int, int],
    boxes: list[tuple[int, int, int, int]],
    colors: list[int],
) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for bbox, color in zip(boxes, colors):
        r0, c0, r1, c1 = bbox
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
    return grid


def _expected_from_params(n_layers: int, inner_color: int) -> np.ndarray:
    return _build_output(n_layers, inner_color)


def _gen_attempt() -> np.ndarray | None:
    n_layers = int(np.random.choice([2, 3, 4]))
    row_parity = int(np.random.randint(0, 2))
    col_parity = int(np.random.randint(0, 2))
    gap_mode = int(np.random.randint(0, 3))
    gap_rows, gap_cols = _sample_gaps(n_layers, gap_mode)

    inner_h_choices = _parity_values(3, 7, row_parity)
    inner_w_choices = _parity_values(3, 7, col_parity)
    if not inner_h_choices or not inner_w_choices:
        return None

    if row_parity == col_parity and np.random.rand() < 0.35:
        side_choices = sorted(set(inner_h_choices) & set(inner_w_choices))
        if not side_choices:
            return None
        inner_h = inner_w = int(np.random.choice(side_choices))
    else:
        inner_h = int(np.random.choice(inner_h_choices))
        inner_w = int(np.random.choice(inner_w_choices))

    outer_h = inner_h + 2 * sum(gap_rows)
    outer_w = inner_w + 2 * sum(gap_cols)
    pad_r = int(np.random.randint(1, 4))
    pad_c = int(np.random.randint(1, 4))
    height = outer_h + 2 * pad_r
    width = outer_w + 2 * pad_c

    if not (11 <= height <= 24 and 11 <= width <= 24):
        return None

    outer_bbox = (pad_r, pad_c, pad_r + outer_h - 1, pad_c + outer_w - 1)
    boxes = _build_boxes(outer_bbox, gap_rows, gap_cols)

    for bbox in boxes:
        r0, c0, r1, c1 = bbox
        if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
            return None

    colors = np.random.choice(PALETTE, size=n_layers, replace=False).astype(int).tolist()
    grid = _paint_nested_rectangles((height, width), boxes, colors)

    try:
        expected = _expected_from_params(n_layers, colors[-1])
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-ring input")
