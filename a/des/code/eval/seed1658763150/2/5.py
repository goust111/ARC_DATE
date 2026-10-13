from __future__ import annotations

import random

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))

# --------------- solver helpers ---------------


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_monochromatic_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    return components


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _is_ring(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False
    if component["area"] != 2 * height + 2 * width - 4:
        return False

    mask = np.zeros((height, width), dtype=bool)
    for r, c in component["coords"]:
        mask[r - r0, c - c0] = True
    return bool(np.array_equal(mask, _border_mask(height, width)))


def _center_key(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r0 + r1, c0 + c1


def _read_center_sequence(
    grid: np.ndarray, inner_bbox: tuple[int, int, int, int], bg: int
) -> list[int]:
    r0, c0, r1, c1 = inner_bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    cells = np.argwhere(interior != bg)
    assert 1 <= cells.shape[0] <= 2, "expected a 1- or 2-cell center instruction block"

    rr0 = int(cells[:, 0].min())
    rr1 = int(cells[:, 0].max())
    cc0 = int(cells[:, 1].min())
    cc1 = int(cells[:, 1].max())
    block = interior[rr0 : rr1 + 1, cc0 : cc1 + 1]

    height, width = block.shape
    assert height * width == cells.shape[0], "center block must fill its bbox"
    assert (height, width) in {(1, 1), (1, 2), (2, 1)}, "center block must be 1x1, 1x2, or 2x1"
    assert _center_key((r0 + 1 + rr0, c0 + 1 + cc0, r0 + rr1 + 1, c0 + cc1 + 1)) == _center_key(
        inner_bbox
    ), "center block must share the rings' center"

    if height == 1:
        return [int(v) for v in block[0, :]]
    return [int(v) for v in block[:, 0]]


def _extract_structures(grid: np.ndarray) -> tuple[int, list[dict]]:
    bg = int(np.bincount(grid.ravel()).argmax())
    components = _find_monochromatic_components(grid, bg)
    rings = [component for component in components if _is_ring(component)]
    assert rings, "expected at least one ring"

    grouped: dict[tuple[int, int], list[dict]] = {}
    for ring in rings:
        grouped.setdefault(_center_key(ring["bbox"]), []).append(ring)

    structures: list[dict] = []
    for key, group in grouped.items():
        ordered = sorted(
            group,
            key=lambda ring: (
                ring["bbox"][2] - ring["bbox"][0] + 1,
                ring["bbox"][3] - ring["bbox"][1] + 1,
            ),
        )
        assert 2 <= len(ordered) <= 4, "each structure must contain 2-4 rings"

        colors = [int(ring["color"]) for ring in ordered]
        assert len(set(colors)) == len(colors), "ring colors must be unique within a structure"

        for inner, outer in zip(ordered, ordered[1:]):
            ir0, ic0, ir1, ic1 = inner["bbox"]
            or0, oc0, or1, oc1 = outer["bbox"]
            assert (or0, oc0, or1, oc1) == (ir0 - 2, ic0 - 2, ir1 + 2, ic1 + 2), (
                "adjacent rings must be separated by one background cell"
            )

        sequence = _read_center_sequence(grid, ordered[0]["bbox"], bg)
        ring_by_color = {int(ring["color"]): ring for ring in ordered}
        used: set[int] = set()
        output_rings: list[dict] = []
        for color in sequence:
            assert color in ring_by_color, "center colors must refer to ring colors"
            if color not in used:
                output_rings.append(ring_by_color[color])
                used.add(color)
        for ring in ordered:
            if int(ring["color"]) not in used:
                output_rings.append(ring)

        outer_bbox = ordered[-1]["bbox"]
        structures.append(
            {
                "x": int(outer_bbox[1]),
                "y": int(outer_bbox[0]),
                "ordered_rings": output_rings,
            }
        )

    structures.sort(key=lambda structure: (structure["x"], structure["y"]))
    return bg, structures


def _render_ring(height: int, width: int, color: int, bg: int) -> np.ndarray:
    ring = np.full((height, width), bg, dtype=int)
    ring[0, :] = color
    ring[-1, :] = color
    ring[:, 0] = color
    ring[:, -1] = color
    return ring


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each concentric-ring structure into standalone rings, then line the rings up using the center colors as order hints.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, structures = _extract_structures(g)
    flat_rings: list[tuple[int, int, int]] = []
    for structure in structures:
        for ring in structure["ordered_rings"]:
            r0, c0, r1, c1 = ring["bbox"]
            flat_rings.append((r1 - r0 + 1, c1 - c0 + 1, int(ring["color"])))

    assert flat_rings, "expected output rings"
    out_h = max(height for height, _width, _color in flat_rings)
    out_w = sum(width for _height, width, _color in flat_rings) + len(flat_rings) - 1
    out = np.full((out_h, out_w), bg, dtype=int)

    col = 0
    for height, width, color in flat_rings:
        out[:height, col : col + width] = _render_ring(height, width, color, bg)
        col += width + 1
    return out


# --------------- generator helpers ---------------


STRUCTURE_COUNT_OPTIONS = (2, 3)
RING_COUNT_OPTIONS = (2, 3, 4)
CENTER_KIND_OPTIONS = ("single", "horizontal", "vertical")
MARGIN_RANGE = (1, 4)
GAP_RANGE = (2, 5)
VERTICAL_EXTRA_RANGE = (0, 5)
CENTER_MOAT = 1
MAX_GENERATION_ATTEMPTS = 400


def _sample_structure_spec() -> dict:
    ring_count = random.choice(RING_COUNT_OPTIONS)
    center_kind = random.choice(CENTER_KIND_OPTIONS)
    if center_kind == "single":
        center_h, center_w = 1, 1
        sequence_len = 1
    elif center_kind == "horizontal":
        center_h, center_w = 1, 2
        sequence_len = 2
    else:
        center_h, center_w = 2, 1
        sequence_len = 2

    ring_colors = random.sample(COLOR_POOL, ring_count)
    sequence = random.sample(ring_colors, sequence_len)
    inner_h = center_h + 2 * (CENTER_MOAT + 1)
    inner_w = center_w + 2 * (CENTER_MOAT + 1)
    outer_h = inner_h + 4 * (ring_count - 1)
    outer_w = inner_w + 4 * (ring_count - 1)

    return {
        "ring_count": ring_count,
        "ring_colors": ring_colors,  # inner -> outer
        "sequence": sequence,
        "center_h": center_h,
        "center_w": center_w,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "outer_h": outer_h,
        "outer_w": outer_w,
    }


def _paint_ring(grid: np.ndarray, r0: int, c0: int, height: int, width: int, color: int) -> None:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _render_structure(grid: np.ndarray, top: int, left: int, spec: dict) -> None:
    n = int(spec["ring_count"])
    for layer_index, color in enumerate(spec["ring_colors"]):
        shrink = 2 * (n - 1 - layer_index)
        height = int(spec["inner_h"] + 4 * layer_index)
        width = int(spec["inner_w"] + 4 * layer_index)
        _paint_ring(grid, top + shrink, left + shrink, height, width, int(color))

    inner_top = top + 2 * (n - 1)
    inner_left = left + 2 * (n - 1)
    center_top = inner_top + CENTER_MOAT + 1
    center_left = inner_left + CENTER_MOAT + 1
    if spec["center_h"] == 1:
        for offset, color in enumerate(spec["sequence"]):
            grid[center_top, center_left + offset] = int(color)
    else:
        for offset, color in enumerate(spec["sequence"]):
            grid[center_top + offset, center_left] = int(color)


def _expected_output(specs: list[dict]) -> np.ndarray:
    ordered_rings: list[tuple[int, int, int]] = []
    for spec in specs:
        ring_map = {
            int(color): (
                int(spec["inner_h"] + 4 * index),
                int(spec["inner_w"] + 4 * index),
                int(color),
            )
            for index, color in enumerate(spec["ring_colors"])
        }
        used: set[int] = set()
        for color in spec["sequence"]:
            if int(color) not in used:
                ordered_rings.append(ring_map[int(color)])
                used.add(int(color))
        for index, color in enumerate(spec["ring_colors"]):
            if int(color) not in used:
                ordered_rings.append(
                    (
                        int(spec["inner_h"] + 4 * index),
                        int(spec["inner_w"] + 4 * index),
                        int(color),
                    )
                )

    out_h = max(height for height, _width, _color in ordered_rings)
    out_w = sum(width for _height, width, _color in ordered_rings) + len(ordered_rings) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    col = 0
    for height, width, color in ordered_rings:
        out[:height, col : col + width] = _render_ring(height, width, color, BG)
        col += width + 1
    return out


def _generate_attempt() -> np.ndarray | None:
    structure_count = random.choice(STRUCTURE_COUNT_OPTIONS)
    specs = [_sample_structure_spec() for _ in range(structure_count)]

    top_margin = random.randint(*MARGIN_RANGE)
    bottom_margin = random.randint(*MARGIN_RANGE)
    left_margin = random.randint(*MARGIN_RANGE)
    right_margin = random.randint(*MARGIN_RANGE)
    gaps = [random.randint(*GAP_RANGE) for _ in range(structure_count - 1)]
    extra_height = random.randint(*VERTICAL_EXTRA_RANGE)

    max_h = max(int(spec["outer_h"]) for spec in specs)
    grid_h = top_margin + bottom_margin + max_h + extra_height
    grid_w = (
        left_margin
        + right_margin
        + sum(int(spec["outer_w"]) for spec in specs)
        + sum(gaps)
    )
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    left = left_margin
    for index, spec in enumerate(specs):
        max_top = grid_h - bottom_margin - int(spec["outer_h"])
        top = random.randint(top_margin, max_top)
        _render_structure(grid, top, left, spec)
        left += int(spec["outer_w"])
        if index < len(gaps):
            left += gaps[index]

    expected = _expected_output(specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {MAX_GENERATION_ATTEMPTS} attempts")
