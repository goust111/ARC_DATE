from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
FILL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.asarray(grid).ravel()))
    if exclude is not None:
        colors -= set(exclude)
    return sorted(colors)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    assert bg == BLACK, "expected black background"

    rectangles: list[dict] = []
    marker_counts: set[int] = set()
    for coords in _components(g != bg):
        rr, cc = coords[:, 0], coords[:, 1]
        box = int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max())
        crop = crop_bbox(g, box)
        h, w = crop.shape
        assert h >= 3 and w >= 3, "expected rectangles of at least 3x3"
        assert np.all(crop != bg), "foreground component must fill its bbox"

        fill_colors = colors_present(crop, exclude={bg, GREY})
        assert len(fill_colors) == 1, "each rectangle must have exactly one non-grey fill color"
        fill_color = int(fill_colors[0])

        count = int(np.count_nonzero(crop == GREY))
        assert count not in marker_counts, "marker counts must be unique"
        marker_counts.add(count)

        rectangles.append(
            {
                "bbox": box,
                "crop": crop,
                "color": fill_color,
                "marker_count": count,
            }
        )

    assert rectangles, "expected at least one rectangle"
    rectangles.sort(key=lambda item: item["marker_count"])
    return rectangles


def _panel_from_crop(crop: np.ndarray, rank: int) -> np.ndarray:
    if rank % 2 == 1:
        return np.concatenate([crop, np.fliplr(crop)], axis=1)
    return np.concatenate([crop, np.flipud(crop)], axis=0)


def _assemble_panels(panels: list[np.ndarray]) -> np.ndarray:
    max_h = max(panel.shape[0] for panel in panels)
    total_w = sum(panel.shape[1] for panel in panels) + len(panels) - 1
    out = np.full((max_h, total_w), BLACK, dtype=int)

    c0 = 0
    for panel in panels:
        h, w = panel.shape
        out[:h, c0 : c0 + w] = panel
        c0 += w + 1
    return out


def _render_expected(specs: list[dict]) -> np.ndarray:
    ordered = sorted(specs, key=lambda item: item["marker_count"])
    panels = [_panel_from_crop(item["patch"], rank) for rank, item in enumerate(ordered, start=1)]
    return _assemble_panels(panels)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the cropped rectangles by grey-marker sparsity, then mirror-concatenate by rank parity.

    Input structure:
    - A black background contains 4 to 5 separated filled rectangles in distinct non-grey colors.
    - Each rectangle is at least 3x3 and may contain several grey marker pixels inside it.
    - The grey marker counts are pairwise distinct, so the rectangles have an unambiguous global order.

    Rule:
    - Detect every foreground rectangle, crop its tight bounding box, and count the grey markers inside it.
    - Sort the cropped rectangles from the fewest grey markers to the most; ranks start at 1.
    - For odd ranks, place a left-right mirror copy immediately to the right of the crop; for even ranks, place an up-down mirror copy immediately below the crop.
    - Arrange the transformed results from left to right in rank order with one black column between neighbors, padding shorter results only at the bottom with black.

    Key insight:
    - The grey markers do not directly choose a mirror axis; they first determine the global ranking, and only the parity of that ranking decides whether each rectangle expands horizontally or vertically.
    """
    rectangles = _extract_rectangles(grid)
    panels = [_panel_from_crop(item["crop"], rank) for rank, item in enumerate(rectangles, start=1)]
    return _assemble_panels(panels)


def _candidate_shapes(marker_count: int) -> list[tuple[int, int]]:
    return [
        (h, w)
        for h in range(3, 7)
        for w in range(3, 5)
        if (h - 2) * (w - 2) >= marker_count
    ]


def _build_patch(color: int, marker_count: int, *, prefer_narrow: bool = False) -> np.ndarray:
    shapes = _candidate_shapes(marker_count)
    if prefer_narrow:
        narrow_shapes = [shape for shape in shapes if shape[1] == 3]
        if narrow_shapes:
            shapes = narrow_shapes
    h, w = random.choice(shapes)
    patch = np.full((h, w), color, dtype=int)

    if marker_count:
        interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
        for r, c in random.sample(interior, marker_count):
            patch[r, c] = GREY
    return patch


def _sample_marker_counts(n_rects: int) -> list[int]:
    if n_rects == 5:
        counts = [0, 1, 2, 3, 4]
    else:
        missing = random.choice([1, 2, 3])
        counts = [count for count in range(5) if count != missing]
    random.shuffle(counts)
    return counts


def _order(specs: list[dict], key) -> tuple[int, ...]:
    return tuple(sorted(range(len(specs)), key=lambda idx: key(specs[idx])))


def _size_and_color_order_is_nontrivial(specs: list[dict]) -> bool:
    count_order = _order(specs, lambda item: (item["marker_count"], item["color"]))
    alternative_orders = [
        _order(specs, lambda item: (item["color"], item["shape"][0], item["shape"][1])),
        _order(specs, lambda item: (item["shape"][0] * item["shape"][1], item["shape"][0], item["shape"][1])),
        _order(specs, lambda item: (item["shape"][0], item["shape"][1], item["color"])),
        _order(specs, lambda item: (item["shape"][1], item["shape"][0], item["color"])),
    ]
    return all(order != count_order for order in alternative_orders)


def _position_order_is_nontrivial(specs: list[dict]) -> bool:
    count_order = _order(specs, lambda item: (item["marker_count"], item["color"]))
    alternative_orders = [
        _order(specs, lambda item: (item["top_left"][1], item["top_left"][0])),
        _order(specs, lambda item: (item["top_left"][0], item["top_left"][1])),
    ]
    return all(order != count_order for order in alternative_orders)


def _output_fits(specs: list[dict]) -> bool:
    ordered = sorted(specs, key=lambda item: item["marker_count"])
    heights: list[int] = []
    widths: list[int] = []
    for rank, item in enumerate(ordered, start=1):
        h, w = item["shape"]
        heights.append(h if rank % 2 == 1 else 2 * h)
        widths.append(2 * w if rank % 2 == 1 else w)
    return max(heights) <= 30 and sum(widths) + len(widths) - 1 <= 30


def _place_specs(specs: list[dict], grid_shape: tuple[int, int]) -> np.ndarray | None:
    h, w = grid_shape
    grid = np.full((h, w), BLACK, dtype=int)
    order = sorted(
        range(len(specs)),
        key=lambda idx: specs[idx]["shape"][0] * specs[idx]["shape"][1],
        reverse=True,
    )

    for idx in order:
        patch = specs[idx]["patch"]
        ph, pw = patch.shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, h - ph):
            for c0 in range(1, w - pw):
                r1 = r0 + ph
                c1 = c0 + pw
                if np.any(grid[r0 - 1 : r1 + 1, c0 - 1 : c1 + 1] != BLACK):
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        grid[r0 : r0 + ph, c0 : c0 + pw] = patch
        specs[idx]["top_left"] = (r0, c0)
    return grid


def _generate_case() -> tuple[np.ndarray, list[dict]] | None:
    n_rects = 4 if random.random() < 0.55 else 5
    marker_counts = _sample_marker_counts(n_rects)
    colors = random.sample(FILL_COLORS, n_rects)

    specs: list[dict] = []
    for color, marker_count in zip(colors, marker_counts):
        prefer_narrow = n_rects == 5 and marker_count in {0, 2, 4}
        patch = _build_patch(color, marker_count, prefer_narrow=prefer_narrow)
        specs.append(
            {
                "color": color,
                "marker_count": marker_count,
                "patch": patch,
                "shape": tuple(map(int, patch.shape)),
            }
        )

    if not _output_fits(specs):
        return None
    if not _size_and_color_order_is_nontrivial(specs):
        return None

    grid_shape = (random.randint(18, 24), random.randint(18, 24))
    grid = _place_specs(specs, grid_shape)
    if grid is None:
        return None
    if not _position_order_is_nontrivial(specs):
        return None

    expected = _render_expected(specs)
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, specs


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid input for puzzle 10")
