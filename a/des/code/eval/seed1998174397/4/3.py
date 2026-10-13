from __future__ import annotations

import random
from collections import deque

import numpy as np


DIVIDER_COLOR = 9
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _paint_ring(grid: np.ndarray, top: int, left: int, side: int, color: int) -> None:
    bottom = top + side - 1
    right = left + side - 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color


def _ring_values(square: np.ndarray, offset: int) -> np.ndarray:
    side = square.shape[0] - 2 * offset
    ring = np.zeros(square.shape, dtype=bool)
    top = offset
    left = offset
    bottom = top + side - 1
    right = left + side - 1
    ring[top, left : right + 1] = True
    ring[bottom, left : right + 1] = True
    ring[top : bottom + 1, left] = True
    ring[top : bottom + 1, right] = True
    return square[ring]


def _parse_legend(left: np.ndarray, bg: int) -> list[int] | None:
    if left.size == 0:
        return None
    comps = _components(left != bg)
    if not comps:
        return None

    entries: list[tuple[int, int, int]] = []
    for pts in comps:
        r0, c0, r1, c1 = _bbox(pts)
        if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
            return None
        block = left[r0 : r1 + 1, c0 : c1 + 1]
        colors = set(map(int, block.flatten()))
        if len(colors) != 1 or bg in colors:
            return None
        if len(pts) != 4:
            return None
        entries.append((r0, c0, int(block[0, 0])))

    entries.sort()
    for i in range(1, len(entries)):
        if entries[i][0] - entries[i - 1][0] != 3:
            return None
    return [color for _r, _c, color in entries]


def _parse_group(square: np.ndarray, bg: int) -> list[tuple[int, int]] | None:
    side = square.shape[0]
    if square.shape[0] != square.shape[1] or side < 5:
        return None

    layers: list[tuple[int, int]] = []
    recon = np.full(square.shape, bg, dtype=int)

    for offset in range((side + 1) // 2):
        vals = _ring_values(square, offset)
        uniq = {int(v) for v in vals}
        if len(uniq) != 1:
            return None
        color = uniq.pop()
        if color == bg:
            break
        layers.append((offset, color))
        _paint_ring(recon, offset, offset, side - 2 * offset, color)

    if not (2 <= len(layers) <= 4):
        return None
    if len({color for _offset, color in layers}) != len(layers):
        return None
    if side - 2 * len(layers) < 1:
        return None
    if not np.array_equal(square, recon):
        return None
    return layers


def _parse_right(right: np.ndarray, bg: int) -> list[dict[str, object]] | None:
    comps = _components(right != bg)
    if not comps:
        return None

    groups: list[dict[str, object]] = []
    for pts in comps:
        r0, c0, r1, c1 = _bbox(pts)
        sub = right[r0 : r1 + 1, c0 : c1 + 1]
        layers = _parse_group(sub, bg)
        if layers is None:
            return None
        groups.append({"bbox": (r0, c0, r1, c1), "layers": layers})
    return groups


def _find_structure(
    grid: np.ndarray,
) -> tuple[int, int, np.ndarray, list[int], list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)

    candidates: list[tuple[int, list[int], list[dict[str, object]]]] = []
    for c in range(1, w - 1):
        column = g[:, c]
        if int(column[0]) == bg or not np.all(column == column[0]):
            continue
        legend = _parse_legend(g[:, :c], bg)
        if legend is None:
            continue
        groups = _parse_right(g[:, c + 1 :], bg)
        if groups is None:
            continue
        rank = {color: idx for idx, color in enumerate(legend)}
        if any(color not in rank for group in groups for _offset, color in group["layers"]):
            continue
        candidates.append((c, legend, groups))

    assert candidates, "expected a valid divider with legend and nested frame groups"
    divider_col, legend, groups = candidates[0]
    return bg, divider_col, g[:, divider_col + 1 :], legend, groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the highest-priority ring from each nested frame group.

    The priority order is read from the left legend: top block has highest
    priority, then the next block below it, and so on. The vertical divider and
    the entire legend area are removed from the output.
    """
    bg, _divider_col, right, legend, groups = _find_structure(grid)
    out = np.full(right.shape, bg, dtype=int)
    rank = {color: idx for idx, color in enumerate(legend)}

    for group in groups:
        r0, c0, r1, c1 = group["bbox"]
        side = r1 - r0 + 1
        chosen_offset, chosen_color = min(group["layers"], key=lambda item: rank[item[1]])
        _paint_ring(out, r0 + chosen_offset, c0 + chosen_offset, side - 2 * chosen_offset, chosen_color)

    return out


def _valid_positions(occupied: np.ndarray, side: int, border_gap: int, object_gap: int) -> list[tuple[int, int]]:
    h, w = occupied.shape
    positions: list[tuple[int, int]] = []
    for r in range(border_gap, h - side - border_gap + 1):
        for c in range(border_gap, w - side - border_gap + 1):
            r0 = max(0, r - object_gap)
            c0 = max(0, c - object_gap)
            r1 = min(h, r + side + object_gap)
            c1 = min(w, c + side + object_gap)
            if not occupied[r0:r1, c0:c1].any():
                positions.append((r, c))
    return positions


def _place_groups(
    height: int,
    width: int,
    group_specs: list[dict[str, object]],
    border_gap: int = 1,
    object_gap: int = 1,
) -> list[dict[str, object]] | None:
    occupied = np.zeros((height, width), dtype=bool)
    placed: list[dict[str, object]] = []

    order = sorted(range(len(group_specs)), key=lambda idx: int(group_specs[idx]["side"]), reverse=True)
    for idx in order:
        spec = group_specs[idx]
        side = int(spec["side"])
        positions = _valid_positions(occupied, side, border_gap, object_gap)
        if not positions:
            return None
        top, left = random.choice(positions)
        occupied[top : top + side, left : left + side] = True
        placed.append(
            {
                "side": side,
                "layers": list(spec["layers"]),
                "top": top,
                "left": left,
            }
        )
    return placed


def _build_expected(height: int, width: int, bg: int, legend: list[int], placed: list[dict[str, object]]) -> np.ndarray:
    out = np.full((height, width), bg, dtype=int)
    rank = {color: idx for idx, color in enumerate(legend)}
    for spec in placed:
        side = int(spec["side"])
        top = int(spec["top"])
        left = int(spec["left"])
        chosen_offset, chosen_color = min(spec["layers"], key=lambda item: rank[item[1]])
        _paint_ring(out, top + chosen_offset, left + chosen_offset, side - 2 * chosen_offset, chosen_color)
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c != DIVIDER_COLOR])
    usable_colors = [c for c in range(10) if c not in (bg, DIVIDER_COLOR)]

    group_count = random.randint(2, 4)
    layer_counts = [random.choice([2, 3, 4]) for _ in range(group_count)]
    max_layers = max(layer_counts)
    legend_count = random.randint(max_layers, min(len(usable_colors), max_layers + 2))
    legend = random.sample(usable_colors, legend_count)

    group_specs: list[dict[str, object]] = []
    for layer_count in layer_counts:
        inner_side = random.randint(3, 7)
        side = inner_side + 2 * (layer_count - 1)
        layer_colors = random.sample(legend, layer_count)
        random.shuffle(layer_colors)
        group_specs.append(
            {
                "side": side,
                "layers": [(offset, color) for offset, color in enumerate(layer_colors)],
            }
        )

    max_side = max(int(spec["side"]) for spec in group_specs)
    legend_span = 3 * legend_count - 1
    min_height = max(legend_span, max_side + 2, 12)

    for _ in range(120):
        height = random.randint(min_height, min(30, min_height + 8))
        right_width_min = max(max_side + 2, 14)
        right_width_max = min(36, max(right_width_min, right_width_min + group_count * 5))
        if right_width_min > right_width_max:
            continue
        right_width = random.randint(right_width_min, right_width_max)
        placed = _place_groups(height, right_width, group_specs)
        if placed is None:
            continue

        left_width = random.randint(3, 6)
        total_width = left_width + 1 + right_width
        grid = np.full((height, total_width), bg, dtype=int)
        divider_col = left_width
        grid[:, divider_col] = DIVIDER_COLOR

        legend_col = random.randint(0, left_width - 2)
        legend_top = random.randint(0, height - legend_span)
        for idx, color in enumerate(legend):
            r0 = legend_top + 3 * idx
            grid[r0 : r0 + 2, legend_col : legend_col + 2] = color

        for spec in placed:
            top = int(spec["top"])
            left = int(spec["left"]) + divider_col + 1
            side = int(spec["side"])
            for offset, color in spec["layers"]:
                _paint_ring(grid, top + offset, left + offset, side - 2 * offset, color)

        expected = _build_expected(height, right_width, bg, legend, placed)
        actual = main(grid)
        if np.array_equal(actual, expected):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
