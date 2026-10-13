from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
MARKER = 6
PALETTE = np.array([1, 2, 3, 4, 5, 7, 8, 9], dtype=int)
MAX_GENERATION_ATTEMPTS = 500


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for row in range(height):
        for col in range(width):
            if not bool(mask[row, col]) or bool(seen[row, col]):
                continue

            queue: deque[tuple[int, int]] = deque([(row, col)])
            seen[row, col] = True
            coords: list[tuple[int, int]] = []

            while queue:
                r, c = queue.popleft()
                coords.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = int(r + dr), int(c + dc)
                    if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            components.append(np.array(coords, dtype=int))

    return components


def _border_is_color(block: np.ndarray, color: int, thickness: int) -> bool:
    height, width = map(int, block.shape)
    if thickness <= 0 or thickness > min(height, width):
        return False

    border = np.zeros_like(block, dtype=bool)
    border[:thickness, :] = True
    border[height - thickness :, :] = True
    border[:, :thickness] = True
    border[:, width - thickness :] = True
    return bool(np.all(block[border] == int(color)))


def _extract_layers(block: np.ndarray) -> list[tuple[int, int]]:
    current = np.asarray(block, dtype=int)
    layers_outer_to_inner: list[tuple[int, int]] = []

    while True:
        height, width = map(int, current.shape)
        color = int(current[0, 0])
        thickness = 0
        for candidate in range(1, min(height, width) + 1):
            if _border_is_color(current, color, candidate):
                thickness = int(candidate)
            else:
                break

        assert thickness > 0, "every layer must have positive thickness"
        layers_outer_to_inner.append((color, thickness))

        if height <= 2 * thickness or width <= 2 * thickness:
            break

        current = current[thickness : height - thickness, thickness : width - thickness]

    return list(reversed(layers_outer_to_inner))


def _parse_structures(grid: np.ndarray) -> list[dict[str, object]]:
    markers = [tuple(map(int, pos)) for pos in np.argwhere(grid == MARKER)]
    assert markers, "expected at least one magenta marker"

    shape_mask = (grid != BG) & (grid != MARKER)
    components = _connected_components(shape_mask)
    assert components, "expected at least one concentric structure"

    marker_set = set(markers)
    used_markers: set[tuple[int, int]] = set()
    structures: list[dict[str, object]] = []

    for coords in components:
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        block = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)

        hits = [(r, c) for r, c in markers if r == r1 + 1 and c0 <= c <= c1]
        assert len(hits) == 1, "each structure must have one marker directly below it"
        marker_row, marker_col = hits[0]
        used_markers.add((marker_row, marker_col))

        layers = _extract_layers(block)
        structures.append(
            {
                "marker_col": int(marker_col),
                "width": int(block.shape[1]),
                "layers": layers,
            }
        )

    assert used_markers == marker_set, "every marker must belong to exactly one structure"
    structures.sort(key=lambda item: int(item["marker_col"]))
    return structures


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unfold each nested rectangular ring target into stacked full-width horizontal color bands.

    Input structure:
    - The background is black.
    - Several disjoint filled concentric-rectangle structures appear in the grid; each one is made of nested rectangular color rings with a solid colored center.
    - Exactly one magenta marker pixel sits directly below each structure.

    Rule:
    - Treat each non-black, non-magenta connected component as one concentric structure.
    - Match each structure with the unique magenta marker directly beneath its outer bounding box.
    - Recover that structure's layers from the innermost solid rectangle outward by repeatedly peeling off uniform rectangular borders.
    - Turn each recovered layer into a full-width horizontal band whose height equals that layer's single-side thickness.
    - Stack the structures' bands vertically in left-to-right marker order, and use the maximum outer structure width as the output width.

    Key insight:
    - The objects are not ordinary rectangles to crop; they encode an ordered thickness sequence, and the magenta dots only specify which structure's unfolded bands come first.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    structures = _parse_structures(g)
    out_height = sum(int(thickness) for s in structures for _color, thickness in s["layers"])
    out_width = max(int(s["width"]) for s in structures)
    out = np.zeros((out_height, out_width), dtype=int)

    row = 0
    for structure in structures:
        for color, thickness in structure["layers"]:
            out[row : row + thickness, :] = int(color)
            row += int(thickness)

    return out


def _build_structure(rng: np.random.Generator) -> tuple[np.ndarray, int]:
    layer_count = int(rng.integers(2, 5))
    colors = rng.choice(PALETTE, size=layer_count, replace=False).tolist()

    center_thickness = int(rng.integers(1, 4))
    elongation = int(rng.integers(0, 5))
    orientation = int(rng.integers(0, 3))
    if orientation == 0:
        center_shape = (center_thickness, center_thickness + elongation)
    elif orientation == 1:
        center_shape = (center_thickness + elongation, center_thickness)
    else:
        center_shape = (center_thickness, center_thickness)

    structure = np.full(center_shape, colors[0], dtype=int)
    for color in colors[1:]:
        thickness = int(rng.integers(1, 4))
        inner_h, inner_w = map(int, structure.shape)
        expanded = np.full((inner_h + 2 * thickness, inner_w + 2 * thickness), int(color), dtype=int)
        expanded[thickness : thickness + inner_h, thickness : thickness + inner_w] = structure
        structure = expanded

    marker_col = int(rng.integers(0, structure.shape[1]))
    return structure, marker_col


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    structure_count = int(rng.integers(1, 5))
    structures = [_build_structure(rng) for _ in range(structure_count)]

    parsed = [
        {
            "grid": block,
            "marker_col": marker_col,
            "layers": _extract_layers(block),
        }
        for block, marker_col in structures
    ]

    out_height = sum(int(thickness) for item in parsed for _color, thickness in item["layers"])
    if out_height > 30:
        return None

    widths = [int(item["grid"].shape[1]) for item in parsed]
    heights = [int(item["grid"].shape[0]) + 1 for item in parsed]
    left_margin = int(rng.integers(0, 3))
    right_margin = int(rng.integers(0, 3))
    gaps = [int(rng.integers(1, 5)) for _ in range(max(0, structure_count - 1))]
    total_width = int(left_margin + sum(widths) + sum(gaps) + right_margin)
    if total_width > 30:
        return None

    max_height = max(heights)
    if max_height > 30:
        return None
    extra_height = int(rng.integers(0, min(30 - max_height, 8) + 1))
    canvas_height = int(max_height + extra_height)
    if canvas_height > 30:
        return None

    grid = np.zeros((canvas_height, total_width), dtype=int)
    col = int(left_margin)
    for index, item in enumerate(parsed):
        block = np.asarray(item["grid"], dtype=int)
        block_h, block_w = map(int, block.shape)
        top = int(rng.integers(0, canvas_height - (block_h + 1) + 1))

        grid[top : top + block_h, col : col + block_w] = block
        grid[top + block_h, col + int(item["marker_col"])] = MARKER
        col += block_w + (gaps[index] if index < len(gaps) else 0)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
