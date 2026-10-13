from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
LEGEND_COLORS = (2, 3, 4, 6, 7, 8, 9)

TRANSFORM_TEMPLATES: dict[int, tuple[tuple[tuple[int, int], ...], ...]] = {
    0: (
        ((0, 0), (0, 1), (0, 2), (1, 0)),
        ((0, 0), (0, 1), (1, 1), (1, 2)),
        ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)),
    ),
    2: (
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 1), (2, 2)),
        ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 2), (2, 0), (2, 1)),
    ),
}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(sorted(coords), dtype=int))
    parts.sort(key=lambda pts: (int(pts[:, 0].min()), int(pts[:, 1].min()), len(pts)))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _local_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not bg[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_border = False
            while q:
                rr, cc = q.popleft()
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _find_axis_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == BLUE)]
    assert len(cols) == 1, "expected exactly one full blue reflection axis"
    axis_c = int(cols[0])
    assert axis_c == grid.shape[1] // 2, "axis must be the center column"
    return axis_c


def _mirror_coords(coords: np.ndarray, axis_c: int) -> np.ndarray:
    mirrored = coords.copy()
    mirrored[:, 1] = 2 * int(axis_c) - mirrored[:, 1]
    return mirrored


def _merged_hole_count(coords: np.ndarray, axis_c: int) -> int:
    merged = np.vstack([coords, _mirror_coords(coords, axis_c)])
    return _hole_count(_local_mask(merged))


def _legend_mapping(grid: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for color in sorted(int(v) for v in np.unique(grid)):
        if color in (BLACK, BLUE, GREY):
            continue
        for coords in _components(grid == color):
            holes = _hole_count(_local_mask(coords))
            assert holes not in mapping, "legend hole counts must be unique"
            mapping[holes] = int(color)
    assert mapping, "expected at least one legend shape"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the grey sprites across the blue axis, count holes in each mirrored union, and recolor by the legend.

    Input structure:
    - The background is black.
    - A full blue vertical line in the center column is the reflection axis.
    - Grey 4-connected sprites lie strictly on the left side, while non-blue non-grey colored legend shapes sit in the lower-right area.

    Rule:
    - Keep the blue axis and every colored legend shape unchanged.
    - Reflect each grey sprite across the blue axis and add the mirrored copy on the right.
    - For each sprite, take the union of its original pixels and mirrored pixels, then count the number of enclosed 4-connected background holes inside that union's bounding box.
    - Find the legend shape with the same hole count and recolor the whole mirrored union to that legend color.

    Key insight:
    - The determining feature is the topology of the post-mirror union: mirror first, then count holes, then use the legend color for that hole count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    axis_c = _find_axis_col(g)
    legend = _legend_mapping(g)
    grey_parts = _components(g == GREY)
    assert grey_parts, "expected grey sprites"

    out = g.copy()
    for coords in grey_parts:
        assert int(coords[:, 1].max()) <= axis_c - 2, "grey sprite must stay left of the axis gap"
        mirrored = _mirror_coords(coords, axis_c)
        holes = _merged_hole_count(coords, axis_c)
        assert holes in legend, "every mirrored union must match a legend entry"
        color = int(legend[holes])

        for rr, cc in mirrored:
            assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]
            assert int(g[rr, cc]) in (BLACK, GREY), "mirrored sprite may only land on empty space"

        out[coords[:, 0], coords[:, 1]] = color
        out[mirrored[:, 0], mirrored[:, 1]] = color

    return out


def _legend_patch(hole_count: int, color: int) -> np.ndarray:
    width = 3 if hole_count == 0 else 2 * hole_count + 1
    patch = np.full((3, width), int(color), dtype=int)
    if hole_count:
        patch[1, 1::2] = BLACK
    return patch


def _paint_coords(grid: np.ndarray, coords: tuple[tuple[int, int], ...], r0: int, c0: int, color: int) -> None:
    for dr, dc in coords:
        grid[r0 + dr, c0 + dc] = int(color)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0] - pad
        or b[2] + pad < a[0] - pad
        or a[3] + pad < b[1] - pad
        or b[3] + pad < a[1] - pad
    )


def _normalized(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _is_horizontally_symmetric(coords: tuple[tuple[int, int], ...]) -> bool:
    norm = _normalized(coords)
    max_c = max(c for _, c in norm)
    mirrored = _normalized(tuple((r, max_c - c) for r, c in norm))
    return norm == mirrored


def _place_legends(
    grid: np.ndarray, axis_c: int, legend_holes: list[int], legend_colors: list[int]
) -> list[tuple[int, int, int, int]]:
    specs = list(zip(legend_holes, legend_colors))
    random.shuffle(specs)
    patches = [_legend_patch(holes, color) for holes, color in specs]
    total_h = sum(p.shape[0] for p in patches) + (len(patches) - 1)
    max_w = max(p.shape[1] for p in patches)

    start_r = grid.shape[0] - total_h - 1
    start_c = grid.shape[1] - max_w - 1
    assert start_r >= 1 and start_c >= axis_c + 2

    boxes: list[tuple[int, int, int, int]] = []
    cur_r = start_r
    for patch in patches:
        r0 = cur_r
        c0 = grid.shape[1] - patch.shape[1] - 1
        grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch
        boxes.append((r0, c0, r0 + patch.shape[0] - 1, c0 + patch.shape[1] - 1))
        cur_r += patch.shape[0] + 1
    return boxes


def _component_box(coords: tuple[tuple[int, int], ...], r0: int, c0: int) -> tuple[int, int, int, int]:
    rows = [r0 + r for r, _ in coords]
    cols = [c0 + c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mirror_box(box: tuple[int, int, int, int], axis_c: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return r0, 2 * axis_c - c1, r1, 2 * axis_c - c0


def _expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    axis_c = _find_axis_col(g)
    legend = _legend_mapping(g)
    out = g.copy()
    for coords in _components(g == GREY):
        mirrored = _mirror_coords(coords, axis_c)
        color = legend[_merged_hole_count(coords, axis_c)]
        out[coords[:, 0], coords[:, 1]] = color
        out[mirrored[:, 0], mirrored[:, 1]] = color
    return out


def _validate_generated(grid: np.ndarray, legend_holes: set[int]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    axis_c = _find_axis_col(grid)
    if int(np.count_nonzero(grid == BLUE)) != grid.shape[0]:
        return False

    legend = _legend_mapping(grid)
    if set(legend) != legend_holes:
        return False

    colored_components = 0
    for color in legend.values():
        comps = _components(grid == color)
        if len(comps) != 1:
            return False
        colored_components += 1
        r0, c0, r1, c1 = _bbox(comps[0])
        if r0 == 0 or c0 == 0 or r1 == grid.shape[0] - 1 or c1 == grid.shape[1] - 1:
            return False
    if colored_components != len(legend_holes):
        return False

    grey_parts = _components(grid == GREY)
    if not 2 <= len(grey_parts) <= 4:
        return False

    merged_counts: list[int] = []
    changed_topology = False
    for coords in grey_parts:
        if not (4 <= len(coords) <= 8):
            return False
        if int(coords[:, 1].max()) > axis_c - 2:
            return False
        if _is_horizontally_symmetric(tuple(map(tuple, coords.tolist()))):
            return False
        original_holes = _hole_count(_local_mask(coords))
        if original_holes not in (0, 1):
            return False
        merged_holes = _merged_hole_count(coords, axis_c)
        if merged_holes not in legend:
            return False
        if merged_holes == 0 and original_holes != 0:
            return False
        if merged_holes == 2 and original_holes != 1:
            return False
        if merged_holes != original_holes:
            changed_topology = True
        merged_counts.append(merged_holes)

    if 0 not in merged_counts or 2 not in merged_counts:
        return False
    if not changed_topology:
        return False

    expected = _expected_output(grid)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    recolor_colors = {int(v) for v in np.unique(actual[(grid == GREY) | ((grid == BLACK) & (actual != BLACK))])}
    return len(recolor_colors) >= 2


def _choose_sprite_holes(count: int) -> list[int]:
    holes = [0, 2]
    while len(holes) < count:
        holes.append(random.choice((0, 2)))
    random.shuffle(holes)
    return holes


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(13, 19)
    width = random.choice([w for w in range(17, 24) if w % 2 == 1])
    axis_c = width // 2

    grid = np.zeros((height, width), dtype=int)
    grid[:, axis_c] = BLUE

    legend_holes = [0, 2]
    legend_colors = random.sample(LEGEND_COLORS, len(legend_holes))
    legend_boxes = _place_legends(grid, axis_c, legend_holes, legend_colors)

    sprite_count = random.randint(2, 4)
    sprite_holes = _choose_sprite_holes(sprite_count)
    placed_boxes: list[tuple[int, int, int, int]] = []

    order = list(range(sprite_count))
    order.sort(key=lambda idx: max(r for r, _ in TRANSFORM_TEMPLATES[sprite_holes[idx]][0]), reverse=True)
    for idx in order:
        target_holes = sprite_holes[idx]
        variants = list(TRANSFORM_TEMPLATES[target_holes])
        random.shuffle(variants)
        placed = False

        for base in variants:
            coords = base
            if random.random() < 0.5:
                max_r = max(r for r, _ in coords)
                coords = tuple(sorted((max_r - r, c) for r, c in coords))
            h = max(r for r, _ in coords) + 1
            w = max(c for _, c in coords) + 1

            for _ in range(80):
                r0 = random.randint(1, height - h - 2)
                c0 = random.randint(1, axis_c - w - 1)
                box = _component_box(coords, r0, c0)
                mbox = _mirror_box(box, axis_c)

                if any(_boxes_overlap(box, other, pad=1) for other in placed_boxes):
                    continue
                if any(_boxes_overlap(mbox, lbox, pad=1) for lbox in legend_boxes):
                    continue

                coords_arr = np.array([(r0 + r, c0 + c) for r, c in coords], dtype=int)
                if _merged_hole_count(coords_arr, axis_c) != target_holes:
                    continue

                _paint_coords(grid, coords, r0, c0, GREY)
                placed_boxes.append(box)
                placed = True
                break
            if placed:
                break

        if not placed:
            return None

    if not _validate_generated(grid, set(legend_holes)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid mirrored-topology sample")


for _holes, _variants in TRANSFORM_TEMPLATES.items():
    for _coords in _variants:
        assert not _is_horizontally_symmetric(_coords)
        _coords_arr = np.array(_coords, dtype=int)
        assert _hole_count(_local_mask(_coords_arr)) == (_holes // 2)
        for _axis in (4, 5, 6):
            shifted = _coords_arr.copy()
            shifted[:, 1] += 1
            assert _merged_hole_count(shifted, _axis) == _holes
