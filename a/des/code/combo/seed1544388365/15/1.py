from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, Any]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, Any]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                }
            )

    return components


def _component_mask(comp: dict[str, Any]) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rows = comp["coords"][:, 0] - r0
    cols = comp["coords"][:, 1] - c0
    mask[rows, cols] = True
    return mask


def _is_solid_square(comp: dict[str, Any]) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return h == w and len(comp["coords"]) == h * w


def _is_rectangular_frame(comp: dict[str, Any]) -> bool:
    mask = _component_mask(comp)
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    if not mask[0, :].all() or not mask[-1, :].all():
        return False
    if not mask[:, 0].all() or not mask[:, -1].all():
        return False
    return not mask[1:-1, 1:-1].any()


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        outer[0] < inner[0]
        and outer[1] < inner[1]
        and outer[2] > inner[2]
        and outer[3] > inner[3]
    )


def _find_legend(
    components: list[dict[str, Any]], shape: tuple[int, int]
) -> list[dict[str, Any]]:
    h, w = shape
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = {}

    for comp in components:
        if not _is_solid_square(comp):
            continue
        r0, c0, r1, c1 = comp["bbox"]
        size = r1 - r0 + 1
        if c0 == 0:
            key = ("left", size, c0)
        elif c1 == w - 1:
            key = ("right", size, c1)
        else:
            continue
        groups.setdefault(key, []).append(comp)

    candidates: list[list[dict[str, Any]]] = []
    for comps in groups.values():
        if len(comps) != 3:
            continue
        comps = sorted(comps, key=lambda comp: comp["bbox"][0])
        colors = {int(comp["color"]) for comp in comps}
        if len(colors) != 3:
            continue
        separated = all(comps[i]["bbox"][2] < comps[i + 1]["bbox"][0] for i in range(2))
        if separated:
            candidates.append(comps)

    assert len(candidates) == 1, "expected a unique 3-block edge legend"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each nested rectangular frame group's innermost region using the
    legend color indexed by its nesting depth, then erase the legend.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    components = _find_components(g, bg)

    legend = _find_legend(components, g.shape)
    depth_to_color = {
        depth: int(comp["color"]) for depth, comp in enumerate(legend, start=1)
    }

    rings = [comp for comp in components if _is_rectangular_frame(comp)]
    assert rings, "expected at least one rectangular frame"

    out = g.copy()
    for comp in legend:
        rows, cols = comp["coords"].T
        out[rows, cols] = bg

    for comp in rings:
        bbox = comp["bbox"]
        contains_other = any(
            other is not comp and _strictly_contains(bbox, other["bbox"]) for other in rings
        )
        if contains_other:
            continue

        depth = 1 + sum(
            other is not comp and _strictly_contains(other["bbox"], bbox) for other in rings
        )
        assert depth in depth_to_color, "nesting depth must be 1..3"
        r0, c0, r1, c1 = bbox
        out[r0 + 1 : r1, c0 + 1 : c1] = depth_to_color[depth]

    return out


GEN_RETRIES = 500
GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
LEGEND_SIZE_RANGE = (1, 3)
LEGEND_GAP_RANGE = (1, 2)
GROUP_COUNT_RANGE = (2, 5)
OUTER_PADDING = 1
LEGEND_PADDING = 2


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _draw_frame(
    grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int
) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _reserve(
    occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int
) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _sample_depths(count: int, rng: np.random.Generator) -> list[int]:
    if count >= 3 and rng.random() < 0.7:
        depths = [1, 2, 3]
        while len(depths) < count:
            depths.append(_randint(rng, 1, 3))
    elif count == 2 and rng.random() < 0.7:
        depths = rng.choice(np.array([1, 2, 3]), size=2, replace=False).tolist()
    else:
        depths = rng.integers(1, 4, size=count).tolist()
    rng.shuffle(depths)
    return [int(depth) for depth in depths]


def _sample_group_specs(
    count: int, rng: np.random.Generator
) -> list[dict[str, int]]:
    specs: list[dict[str, int]] = []
    for depth in _sample_depths(count, rng):
        max_inner = {1: 6, 2: 4, 3: 3}[depth]
        inner_h = _randint(rng, 1, max_inner)
        inner_w = _randint(rng, 1, max_inner)
        outer_h = inner_h + 4 * depth - 2
        outer_w = inner_w + 4 * depth - 2
        specs.append(
            {
                "depth": depth,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": outer_h,
                "outer_w": outer_w,
            }
        )
    specs.sort(key=lambda spec: spec["outer_h"] * spec["outer_w"], reverse=True)
    return specs


def _place_groups(
    occupied: np.ndarray,
    specs: list[dict[str, int]],
    rng: np.random.Generator,
) -> bool:
    h, w = occupied.shape
    for spec in specs:
        ph = spec["outer_h"]
        pw = spec["outer_w"]
        positions = [
            (r, c)
            for r in range(OUTER_PADDING, h - ph - OUTER_PADDING + 1)
            for c in range(OUTER_PADDING, w - pw - OUTER_PADDING + 1)
            if not occupied[r : r + ph, c : c + pw].any()
        ]
        if not positions:
            return False
        rng.shuffle(positions)
        r0, c0 = positions[0]
        spec["r0"] = int(r0)
        spec["c0"] = int(c0)
        _reserve(occupied, r0, c0, ph, pw, OUTER_PADDING)
    return True


def _build_expected_output(
    grid: np.ndarray,
    legend_boxes: list[tuple[int, int, int]],
    legend_colors: list[int],
    specs: list[dict[str, int]],
) -> np.ndarray:
    out = grid.copy()
    for r0, c0, size in legend_boxes:
        out[r0 : r0 + size, c0 : c0 + size] = 0
    for spec in specs:
        depth = spec["depth"]
        inner_r0 = spec["r0"] + 2 * (depth - 1) + 1
        inner_c0 = spec["c0"] + 2 * (depth - 1) + 1
        inner_r1 = inner_r0 + spec["inner_h"]
        inner_c1 = inner_c0 + spec["inner_w"]
        out[inner_r0:inner_r1, inner_c0:inner_c1] = legend_colors[depth - 1]
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    legend_size = _randint(rng, *LEGEND_SIZE_RANGE)
    legend_gap = _randint(rng, *LEGEND_GAP_RANGE)
    legend_height = 3 * legend_size + 2 * legend_gap

    min_h = max(GRID_H_RANGE[0], legend_height + 2)
    if min_h > GRID_H_RANGE[1]:
        return None

    h = _randint(rng, min_h, GRID_H_RANGE[1])
    w = _randint(rng, *GRID_W_RANGE)

    colors = rng.choice(np.arange(1, 10), size=4, replace=False).tolist()
    frame_color = int(colors[0])
    legend_colors = [int(color) for color in colors[1:]]

    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    top = _randint(rng, 1, h - legend_height - 1)
    side = "left" if rng.random() < 0.5 else "right"
    legend_col = 0 if side == "left" else w - legend_size
    legend_boxes: list[tuple[int, int, int]] = []
    for idx, color in enumerate(legend_colors):
        r0 = top + idx * (legend_size + legend_gap)
        c0 = legend_col
        grid[r0 : r0 + legend_size, c0 : c0 + legend_size] = color
        legend_boxes.append((r0, c0, legend_size))
    _reserve(occupied, top, legend_col, legend_height, legend_size, LEGEND_PADDING)

    group_count = _randint(rng, *GROUP_COUNT_RANGE)
    specs = _sample_group_specs(group_count, rng)
    if not _place_groups(occupied, specs, rng):
        return None

    for spec in specs:
        r0 = spec["r0"]
        c0 = spec["c0"]
        outer_h = spec["outer_h"]
        outer_w = spec["outer_w"]
        for layer in range(spec["depth"]):
            inset = 2 * layer
            _draw_frame(
                grid,
                r0 + inset,
                c0 + inset,
                r0 + outer_h - 1 - inset,
                c0 + outer_w - 1 - inset,
                frame_color,
            )

    expected = _build_expected_output(grid, legend_boxes, legend_colors, specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
