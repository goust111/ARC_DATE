import random

import numpy as np


MAX_GENERATION_ATTEMPTS = 300
BACKGROUND = 0
CORE_SIZE_CHOICES = (2, 3)
RING_COUNT_CHOICES = (2, 3, 4, 5)
MARGIN_MIN = 1
MARGIN_MAX = 4
COLOR_POOL = tuple(range(1, 10))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    if rows.size == 0:
        raise AssertionError("expected at least one foreground cell")
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _is_solid_core(grid: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> bool:
    r0, c0, r1, c1 = bbox
    block = grid[r0 : r1 + 1, c0 : c1 + 1]
    color = int(block[0, 0])
    return color != bg and bool(np.all(block == color))


def _read_ring(
    grid: np.ndarray, bbox: tuple[int, int, int, int], bg: int
) -> dict[str, object] | None:
    r0, c0, r1, c1 = bbox
    if r1 - r0 < 4 or c1 - c0 < 4:
        return None

    top = grid[r0, c0 : c1 + 1]
    bottom = grid[r1, c0 : c1 + 1]
    left = grid[r0 : r1 + 1, c0]
    right = grid[r0 : r1 + 1, c1]

    corners = (top[0], top[-1], bottom[0], bottom[-1])
    if any(int(v) == bg for v in corners):
        return None

    gap_top = np.flatnonzero(top == bg)
    gap_bottom = np.flatnonzero(bottom == bg)
    gap_left = np.flatnonzero(left == bg)
    gap_right = np.flatnonzero(right == bg)
    if not all(gaps.size == 1 for gaps in (gap_top, gap_bottom, gap_left, gap_right)):
        return None

    gap_top_idx = int(gap_top[0])
    gap_bottom_idx = int(gap_bottom[0])
    gap_left_idx = int(gap_left[0])
    gap_right_idx = int(gap_right[0])
    if gap_top_idx in (0, top.size - 1) or gap_bottom_idx in (0, bottom.size - 1):
        return None
    if gap_left_idx in (0, left.size - 1) or gap_right_idx in (0, right.size - 1):
        return None

    boundary_values = []
    for c in range(c0, c1 + 1):
        if c != c0 + gap_top_idx:
            boundary_values.append(int(grid[r0, c]))
        if c != c0 + gap_bottom_idx:
            boundary_values.append(int(grid[r1, c]))
    for r in range(r0 + 1, r1):
        if r != r0 + gap_left_idx:
            boundary_values.append(int(grid[r, c0]))
        if r != r0 + gap_right_idx:
            boundary_values.append(int(grid[r, c1]))
    colors = set(boundary_values)
    if bg in colors or len(colors) != 1:
        return None

    if np.any(grid[r0 + 1, c0 + 1 : c1] != bg):
        return None
    if np.any(grid[r1 - 1, c0 + 1 : c1] != bg):
        return None
    if np.any(grid[r0 + 1 : r1, c0 + 1] != bg):
        return None
    if np.any(grid[r0 + 1 : r1, c1 - 1] != bg):
        return None

    return {
        "bbox": bbox,
        "color": int(boundary_values[0]),
        "gaps": {
            "top": int(c0 + gap_top_idx),
            "bottom": int(c0 + gap_bottom_idx),
            "left": int(r0 + gap_left_idx),
            "right": int(r0 + gap_right_idx),
        },
    }


def _parse_structure(grid: np.ndarray) -> tuple[int, dict[str, object], list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _most_common_color(g)
    outer_bbox = _bbox(g != bg)
    rings_outer_to_inner: list[dict[str, object]] = []
    current = outer_bbox

    while True:
        if _is_solid_core(g, current, bg):
            r0, c0, r1, c1 = current
            core = {
                "bbox": current,
                "color": int(g[r0, c0]),
            }
            return bg, core, rings_outer_to_inner

        ring = _read_ring(g, current, bg)
        assert ring is not None, "expected a concentric ring layer"
        rings_outer_to_inner.append(ring)

        r0, c0, r1, c1 = current
        current = (r0 + 2, c0 + 2, r1 - 2, c1 - 2)
        assert current[0] <= current[2] and current[1] <= current[3], "invalid nested structure"


def _mirror_gaps(
    bbox: tuple[int, int, int, int], gaps: dict[str, int]
) -> dict[str, int]:
    r0, c0, r1, c1 = bbox
    return {
        "top": int(c0 + c1 - gaps["top"]),
        "bottom": int(c0 + c1 - gaps["bottom"]),
        "left": int(r0 + r1 - gaps["left"]),
        "right": int(r0 + r1 - gaps["right"]),
    }


def _paint_ring(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
    gaps: dict[str, int],
    bg: int,
) -> None:
    r0, c0, r1, c1 = bbox
    out[r0, c0 : c1 + 1] = color
    out[r1, c0 : c1 + 1] = color
    out[r0 : r1 + 1, c0] = color
    out[r0 : r1 + 1, c1] = color

    out[r0, gaps["top"]] = bg
    out[r1, gaps["bottom"]] = bg
    out[gaps["left"], c0] = bg
    out[gaps["right"], c1] = bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror odd-depth ring gaps across their side midpoints and recolor all rings
    to the innermost core color.
    """
    g = np.asarray(grid, dtype=int)
    bg, core, rings_outer_to_inner = _parse_structure(g)
    core_color = int(core["color"])

    out = g.copy()
    for layer, ring in enumerate(reversed(rings_outer_to_inner), start=1):
        bbox = ring["bbox"]
        gaps = ring["gaps"]
        if layer % 2 == 1:
            gaps = _mirror_gaps(bbox, gaps)
        _paint_ring(out, bbox, core_color, gaps, bg)
    return out


def _near_center_margins() -> tuple[int, int]:
    first = random.randint(MARGIN_MIN, MARGIN_MAX)
    second = random.randint(max(MARGIN_MIN, first - 1), min(MARGIN_MAX, first + 1))
    if random.random() < 0.5:
        first, second = second, first
    return first, second


def _choose_gap(start: int, end: int, avoid_center: bool) -> int:
    candidates = list(range(start + 1, end))
    if avoid_center and (start + end) % 2 == 0:
        center = (start + end) // 2
        if center in candidates and len(candidates) > 1:
            candidates.remove(center)
    return random.choice(candidates)


def _build_ring_spec(
    core_bbox: tuple[int, int, int, int], ring_count: int
) -> list[dict[str, object]]:
    r0, c0, r1, c1 = core_bbox
    rings = []
    for layer in range(1, ring_count + 1):
        bbox = (r0 - 2 * layer, c0 - 2 * layer, r1 + 2 * layer, c1 + 2 * layer)
        br0, bc0, br1, bc1 = bbox
        avoid_center = layer % 2 == 1
        rings.append(
            {
                "bbox": bbox,
                "gaps": {
                    "top": _choose_gap(bc0, bc1, avoid_center),
                    "bottom": _choose_gap(bc0, bc1, avoid_center),
                    "left": _choose_gap(br0, br1, avoid_center),
                    "right": _choose_gap(br0, br1, avoid_center),
                },
            }
        )
    return rings


def _gen_attempt() -> np.ndarray | None:
    ring_count = random.choice(RING_COUNT_CHOICES)
    core_height = random.choice(CORE_SIZE_CHOICES)
    core_width = random.choice(CORE_SIZE_CHOICES)
    outer_height = core_height + 4 * ring_count
    outer_width = core_width + 4 * ring_count

    top_margin, bottom_margin = _near_center_margins()
    left_margin, right_margin = _near_center_margins()
    grid_height = outer_height + top_margin + bottom_margin
    grid_width = outer_width + left_margin + right_margin

    color_count = ring_count + 1
    if color_count > len(COLOR_POOL):
        return None
    colors = random.sample(COLOR_POOL, color_count)
    core_color = colors[0]
    ring_colors_inner_to_outer = colors[1:]

    outer_bbox = (
        top_margin,
        left_margin,
        top_margin + outer_height - 1,
        left_margin + outer_width - 1,
    )
    core_bbox = (
        outer_bbox[0] + 2 * ring_count,
        outer_bbox[1] + 2 * ring_count,
        outer_bbox[2] - 2 * ring_count,
        outer_bbox[3] - 2 * ring_count,
    )

    grid = np.full((grid_height, grid_width), BACKGROUND, dtype=int)
    cr0, cc0, cr1, cc1 = core_bbox
    grid[cr0 : cr1 + 1, cc0 : cc1 + 1] = core_color

    rings_inner_to_outer = _build_ring_spec(core_bbox, ring_count)
    for ring, ring_color in zip(rings_inner_to_outer, ring_colors_inner_to_outer):
        _paint_ring(grid, ring["bbox"], ring_color, ring["gaps"], BACKGROUND)

    try:
        bg, core, parsed_rings = _parse_structure(grid)
    except AssertionError:
        return None
    if bg != BACKGROUND or int(core["color"]) != core_color or len(parsed_rings) != ring_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid concentric-ring input")
