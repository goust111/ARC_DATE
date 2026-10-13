from __future__ import annotations

from collections import Counter
import random

import numpy as np


RED = 2
PALETTE = [0, 1, 3, 4, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _component_bboxes(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    mask = (grid != bg) & (grid != RED)
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []

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
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if _in_bounds((h, w), nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            assert len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1), "expected solid rectangular objects"
            boxes.append((r0, c0, r1, c1))

    boxes.sort()
    return boxes


def _border_is_color(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> bool:
    return bool(
        np.all(grid[r0, c0 : c1 + 1] == color)
        and np.all(grid[r1, c0 : c1 + 1] == color)
        and np.all(grid[r0 : r1 + 1, c0] == color)
        and np.all(grid[r0 : r1 + 1, c1] == color)
    )


def _extract_layers(
    grid: np.ndarray, bbox: tuple[int, int, int, int]
) -> list[tuple[int, int, int, int, int, int]]:
    r0, c0, r1, c1 = map(int, bbox)
    layers: list[tuple[int, int, int, int, int, int]] = []

    while r0 <= r1 and c0 <= c1:
        color = int(grid[r0, c0])
        thickness = 0
        while True:
            rr0 = r0 + thickness
            cc0 = c0 + thickness
            rr1 = r1 - thickness
            cc1 = c1 - thickness
            if rr0 > rr1 or cc0 > cc1:
                break
            if not _border_is_color(grid, rr0, cc0, rr1, cc1, color):
                break
            thickness += 1
        assert thickness > 0, "failed to recover nested layers"
        layers.append((r0, c0, r1, c1, thickness, color))
        r0 += thickness
        c0 += thickness
        r1 -= thickness
        c1 -= thickness

    return layers


def _count_corner_markers(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = map(int, bbox)
    total = 0
    for cr, cc, dr, dc in (
        (r0, c0, -1, -1),
        (r0, c1, -1, 1),
        (r1, c0, 1, -1),
        (r1, c1, 1, 1),
    ):
        rr = cr + dr
        cc2 = cc + dc
        while _in_bounds(grid.shape, rr, cc2) and int(grid[rr, cc2]) == RED:
            total += 1
            rr += dr
            cc2 += dc
    return total


def _paint_ring(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _paint_layer(
    grid: np.ndarray,
    layer: tuple[int, int, int, int, int, int],
    color: int,
) -> None:
    r0, c0, r1, c1, thickness, _old_color = layer
    for offset in range(thickness):
        rr0 = r0 + offset
        cc0 = c0 + offset
        rr1 = r1 - offset
        cc1 = c1 - offset
        if rr0 > rr1 or cc0 > cc1:
            break
        _paint_ring(grid, rr0, cc0, rr1, cc1, color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the red corner markers for each nested rectangle and cyclically rotate its layer colors inward.

    Input structure:
    - A dominant background surrounds one or more solid axis-aligned rectangles.
    - Each rectangle is partitioned into single-color nested rectangular layers whose thicknesses may differ.
    - Red pixels appear only on the outward diagonals of rectangle corners and act as per-object markers.

    Rule:
    - Ignore the red pixels when locating each rectangular object.
    - For one rectangle, count all red pixels on its four outward corner diagonals to obtain k.
    - Keep the layer geometry fixed, replace layer i by the original layer (i-k) cyclically, and erase all red markers.

    Key insight:
    - The red corner dots are not decorations to add or preserve; they encode a private cyclic color shift for the adjacent nested rectangle.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()
    out[out == RED] = bg

    for bbox in _component_bboxes(g, bg):
        layers = _extract_layers(g, bbox)
        colors = [layer[-1] for layer in layers]
        shift = _count_corner_markers(g, bbox) % len(layers)
        rotated = [colors[(i - shift) % len(colors)] for i in range(len(colors))]
        for layer, color in zip(layers, rotated):
            _paint_layer(out, layer, color)

    return out


def _render_patch(
    ring_thicknesses: list[int],
    core_h: int,
    core_w: int,
    colors: list[int],
    bg: int,
) -> np.ndarray:
    total_ring = sum(ring_thicknesses)
    h = core_h + 2 * total_ring
    w = core_w + 2 * total_ring
    patch = np.full((h, w), bg, dtype=int)

    r0, c0, r1, c1 = 0, 0, h - 1, w - 1
    for thickness, color in zip(ring_thicknesses, colors[:-1]):
        _paint_layer(patch, (r0, c0, r1, c1, thickness, color), color)
        r0 += thickness
        c0 += thickness
        r1 -= thickness
        c1 -= thickness
    patch[r0 : r1 + 1, c0 : c1 + 1] = int(colors[-1])
    return patch


def _sample_marker_lengths(layer_count: int) -> list[int]:
    for _ in range(200):
        lengths = [random.randint(0, 3) for _ in range(4)]
        total = sum(lengths)
        if total > 0 and total % layer_count != 0:
            return lengths
    raise RuntimeError("failed to sample informative corner markers")


def _sample_object(bg: int) -> dict:
    for _ in range(200):
        layer_count = random.randint(2, 5)
        ring_thicknesses = [random.choice([1, 1, 2, 2, 3]) for _ in range(layer_count - 1)]
        core_h = random.randint(1, 5)
        core_w = random.randint(1, 5)
        total_ring = sum(ring_thicknesses)
        outer_h = core_h + 2 * total_ring
        outer_w = core_w + 2 * total_ring
        if outer_h > 13 or outer_w > 13:
            continue
        colors = random.sample([c for c in PALETTE if c != bg], layer_count)
        return {
            "ring_thicknesses": ring_thicknesses,
            "core_h": core_h,
            "core_w": core_w,
            "colors": colors,
            "marker_lengths": _sample_marker_lengths(layer_count),
        }
    raise RuntimeError("failed to sample an object")


def _reserve_bounds(spec: dict, r0: int, c0: int) -> tuple[int, int, int, int]:
    patch = spec["patch"]
    h, w = map(int, patch.shape)
    tl, tr, bl, br = map(int, spec["marker_lengths"])
    top_need = max(tl, tr)
    left_need = max(tl, bl)
    bottom_need = max(bl, br)
    right_need = max(tr, br)
    return (
        r0 - top_need - 1,
        c0 - left_need - 1,
        r0 + h + bottom_need,
        c0 + w + right_need,
    )


def _viable_positions(spec: dict, reserved: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, reserved.shape)
    patch_h, patch_w = map(int, spec["patch"].shape)
    tl, tr, bl, br = map(int, spec["marker_lengths"])
    top_need = max(tl, tr)
    left_need = max(tl, bl)
    bottom_need = max(bl, br)
    right_need = max(tr, br)

    r_min = top_need + 1
    r_max = h - patch_h - bottom_need - 1
    c_min = left_need + 1
    c_max = w - patch_w - right_need - 1
    if r_min > r_max or c_min > c_max:
        return []

    positions = []
    for r0 in range(r_min, r_max + 1):
        for c0 in range(c_min, c_max + 1):
            rr0, cc0, rr1, cc1 = _reserve_bounds(spec, r0, c0)
            if np.any(reserved[rr0 : rr1 + 1, cc0 : cc1 + 1]):
                continue
            positions.append((r0, c0))
    return positions


def _draw_object(grid: np.ndarray, spec: dict) -> None:
    r0, c0 = map(int, spec["pos"])
    patch = spec["patch"]
    h, w = map(int, patch.shape)
    grid[r0 : r0 + h, c0 : c0 + w] = patch

    tl, tr, bl, br = map(int, spec["marker_lengths"])
    for step in range(1, tl + 1):
        grid[r0 - step, c0 - step] = RED
    for step in range(1, tr + 1):
        grid[r0 - step, c0 + w - 1 + step] = RED
    for step in range(1, bl + 1):
        grid[r0 + h - 1 + step, c0 - step] = RED
    for step in range(1, br + 1):
        grid[r0 + h - 1 + step, c0 + w - 1 + step] = RED


def _expected_grid(grid: np.ndarray, specs: list[dict], bg: int) -> np.ndarray:
    out = grid.copy()
    out[out == RED] = bg
    for spec in specs:
        shifted = sum(spec["marker_lengths"]) % len(spec["colors"])
        new_colors = [spec["colors"][(i - shifted) % len(spec["colors"])] for i in range(len(spec["colors"]))]
        rotated_patch = _render_patch(
            spec["ring_thicknesses"],
            spec["core_h"],
            spec["core_w"],
            new_colors,
            bg,
        )
        r0, c0 = map(int, spec["pos"])
        h, w = map(int, rotated_patch.shape)
        out[r0 : r0 + h, c0 : c0 + w] = rotated_patch
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(PALETTE)
    h = random.randint(19, 30)
    w = random.randint(19, 30)
    grid = np.full((h, w), bg, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    object_count = random.choice([1, 2, 2, 3, 3])
    specs = []
    for _ in range(object_count):
        spec = _sample_object(bg)
        spec["patch"] = _render_patch(
            spec["ring_thicknesses"],
            spec["core_h"],
            spec["core_w"],
            spec["colors"],
            bg,
        )
        specs.append(spec)

    specs.sort(key=lambda item: item["patch"].shape[0] * item["patch"].shape[1], reverse=True)

    for spec in specs:
        positions = _viable_positions(spec, reserved)
        if not positions:
            return None
        spec["pos"] = random.choice(positions)
        rr0, cc0, rr1, cc1 = _reserve_bounds(spec, spec["pos"][0], spec["pos"][1])
        reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        _draw_object(grid, spec)

    if _get_bg(grid) != bg:
        return None

    expected = _expected_grid(grid, specs, bg)
    actual = main(grid)
    if np.array_equal(actual, grid):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
