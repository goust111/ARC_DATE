from __future__ import annotations

from collections import Counter, deque

import numpy as np


BG = 0
BLUE = 1
MARKER_COLORS = (2, 3, 4)
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(cells, dtype=int))

    return components


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _contains_strict(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _blue_rectangles(grid: np.ndarray) -> list[dict[str, np.ndarray | tuple[int, int, int, int] | int]]:
    comps = _components_mask(grid == BLUE)
    assert 2 <= len(comps) <= 3, "expected one nested family of 2-3 blue rectangles"

    infos: list[dict[str, np.ndarray | tuple[int, int, int, int] | int]] = []
    for coords in comps:
        bbox = _bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "each rectangle needs a strict interior"

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(sub[0, :] == BLUE)
        assert np.all(sub[-1, :] == BLUE)
        assert np.all(sub[:, 0] == BLUE)
        assert np.all(sub[:, -1] == BLUE)

        infos.append({"coords": coords, "bbox": bbox})

    bboxes = [info["bbox"] for info in infos]
    depths = [sum(_contains_strict(other, bbox) for other in bboxes if other != bbox) for bbox in bboxes]
    assert max(depths) == len(infos) - 1, "rectangles must form one strictly nested chain"
    assert len(set(depths)) == len(depths), "nesting depths must be unique"

    for info, depth in zip(infos, depths):
        info["depth"] = depth
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform nested blue rectangles into hollow frames and fill only the deepest enclosure with the marker's color.

    Input structure:
    - The background is black.
    - Blue pixels form one family of 2 or 3 strictly nested axis-aligned rectangles with a full blue perimeter at every level.
    - Outer levels may already look like thick rectangular rings because inner levels and 1-cell black moats are visible inside them.
    - The strict interior of the deepest rectangle contains exactly one non-blue marker pixel, colored red, green, or yellow.

    Rule:
    - Treat each blue component's bounding box as one rectangle in the nesting chain.
    - Hollow every blue rectangle locally: keep its 1-cell outer perimeter and change only that rectangle's own strict-interior blue pixels to black.
    - Find the deepest rectangle by strict bbox containment.
    - Read the single marker color inside that deepest rectangle, then fill the deepest strict interior with that color, replacing the marker itself as part of the fill.
    - Leave all non-deepest enclosed gaps black.

    Key insight:
    - The blue shapes reveal multiple rectangle levels only through nesting and hollowing; the only region that gets recolored is the strict interior of the deepest bbox, and its fill color is taken from the marker trapped inside it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert _get_bg(g) == BG, "expected black background"

    infos = _blue_rectangles(g)
    deepest = max(infos, key=lambda info: int(info["depth"]))
    inner_r0, inner_c0, inner_r1, inner_c1 = deepest["bbox"]

    inner_patch = g[inner_r0 + 1 : inner_r1, inner_c0 + 1 : inner_c1]
    marker_coords = np.argwhere((inner_patch != BG) & (inner_patch != BLUE))
    assert marker_coords.shape[0] == 1, "expected exactly one marker in the deepest interior"
    mr, mc = map(int, marker_coords[0])
    marker_color = int(inner_patch[mr, mc])
    assert marker_color in MARKER_COLORS

    out = g.copy()
    for info in infos:
        r0, c0, r1, c1 = info["bbox"]
        coords = info["coords"]
        keep = (
            (coords[:, 0] == r0)
            | (coords[:, 0] == r1)
            | (coords[:, 1] == c0)
            | (coords[:, 1] == c1)
        )
        removable = coords[~keep]
        if removable.size:
            out[removable[:, 0], removable[:, 1]] = BG

    out[inner_r0 + 1 : inner_r1, inner_c0 + 1 : inner_c1] = marker_color
    return out


def _build_expected(
    grid: np.ndarray,
    component_masks: list[np.ndarray],
    bboxes: list[tuple[int, int, int, int]],
    marker_color: int,
) -> np.ndarray:
    expected = grid.copy()
    for comp_mask, (r0, c0, r1, c1) in zip(component_masks, bboxes):
        rr, cc = np.nonzero(comp_mask)
        keep = (rr == r0) | (rr == r1) | (cc == c0) | (cc == c1)
        expected[rr[~keep], cc[~keep]] = BG
    inner_r0, inner_c0, inner_r1, inner_c1 = bboxes[-1]
    expected[inner_r0 + 1 : inner_r1, inner_c0 + 1 : inner_c1] = marker_color
    return expected


def _build_nested_sample(
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[np.ndarray], list[tuple[int, int, int, int]], int]:
    n_layers = int(rng.integers(2, 4))

    if float(rng.random()) < 0.35:
        inner_h = inner_w = 3
    else:
        inner_h = int(rng.integers(3, 8))
        inner_w = int(rng.integers(3, 8))

    sizes_inner_to_outer = [(inner_h, inner_w)]
    margins_inner_to_outer: list[tuple[int, int]] = []
    for _ in range(n_layers - 1):
        margin_r = int(rng.integers(3, 5))
        margin_c = int(rng.integers(3, 5))
        prev_h, prev_w = sizes_inner_to_outer[-1]
        sizes_inner_to_outer.append((prev_h + 2 * margin_r, prev_w + 2 * margin_c))
        margins_inner_to_outer.append((margin_r, margin_c))

    bboxes_sizes = list(reversed(sizes_inner_to_outer))
    outer_h, outer_w = bboxes_sizes[0]
    assert outer_h <= 24 and outer_w <= 24

    max_extra_h = 30 - outer_h
    max_extra_w = 30 - outer_w
    total_extra_h = int(rng.integers(2, min(8, max_extra_h) + 1))
    total_extra_w = int(rng.integers(2, min(8, max_extra_w) + 1))
    top = int(rng.integers(1, total_extra_h))
    left = int(rng.integers(1, total_extra_w))
    bottom = total_extra_h - top
    right = total_extra_w - left

    h = outer_h + top + bottom
    w = outer_w + left + right
    grid = np.zeros((h, w), dtype=int)

    bboxes: list[tuple[int, int, int, int]] = []
    outer_bbox = (top, left, top + outer_h - 1, left + outer_w - 1)
    bboxes.append(outer_bbox)
    for child_h, child_w in bboxes_sizes[1:]:
        pr0, pc0, pr1, pc1 = bboxes[-1]
        parent_h = pr1 - pr0 + 1
        parent_w = pc1 - pc0 + 1
        offset_r = (parent_h - child_h) // 2
        offset_c = (parent_w - child_w) // 2
        bboxes.append(
            (pr0 + offset_r, pc0 + offset_c, pr0 + offset_r + child_h - 1, pc0 + offset_c + child_w - 1)
        )

    component_masks: list[np.ndarray] = []
    for i, bbox in enumerate(bboxes):
        r0, c0, r1, c1 = bbox
        comp_mask = np.zeros_like(grid, dtype=bool)
        comp_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        if i + 1 < len(bboxes):
            cr0, cc0, cr1, cc1 = bboxes[i + 1]
            comp_mask[cr0 - 1 : cr1 + 2, cc0 - 1 : cc1 + 2] = False
        component_masks.append(comp_mask)
        grid[comp_mask] = BLUE

    inner_r0, inner_c0, inner_r1, inner_c1 = bboxes[-1]
    interior_cells = [
        (r, c)
        for r in range(inner_r0 + 1, inner_r1)
        for c in range(inner_c0 + 1, inner_c1)
    ]
    assert interior_cells, "deepest rectangle must have an interior"

    marker_r, marker_c = interior_cells[int(rng.integers(0, len(interior_cells)))]
    marker_color = int(rng.choice(MARKER_COLORS))
    grid[marker_r, marker_c] = marker_color
    return grid, component_masks, bboxes, marker_color


def _validate_sample(
    grid: np.ndarray,
    component_masks: list[np.ndarray],
    bboxes: list[tuple[int, int, int, int]],
    marker_color: int,
) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if marker_color not in MARKER_COLORS:
        return False

    if len(component_masks) != len(bboxes) or len(bboxes) not in {2, 3}:
        return False

    blue_components = _components_mask(grid == BLUE)
    if len(blue_components) != len(bboxes):
        return False

    if sorted(_bbox_coords(coords) for coords in blue_components) != sorted(bboxes):
        return False

    if sum(int(color in MARKER_COLORS) for color in grid.ravel()) != 1:
        return False

    removable_blue = 0
    for comp_mask, (r0, c0, r1, c1) in zip(component_masks, bboxes):
        rr, cc = np.nonzero(comp_mask)
        if rr.size == 0:
            return False
        keep = (rr == r0) | (rr == r1) | (cc == c0) | (cc == c1)
        removable_blue += int((~keep).sum())

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not (
            np.all(sub[0, :] == BLUE)
            and np.all(sub[-1, :] == BLUE)
            and np.all(sub[:, 0] == BLUE)
            and np.all(sub[:, -1] == BLUE)
        ):
            return False

    if removable_blue == 0:
        return False

    expected = _build_expected(grid, component_masks, bboxes, marker_color)
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, grid):
        return False

    inner_r0, inner_c0, inner_r1, inner_c1 = bboxes[-1]
    if not np.all(actual[inner_r0 + 1 : inner_r1, inner_c0 + 1 : inner_c1] == marker_color):
        return False

    for child in bboxes[1:]:
        cr0, cc0, cr1, cc1 = child
        moat = actual[cr0 - 1 : cr1 + 2, cc0 - 1 : cc1 + 2].copy()
        moat[1:-1, 1:-1] = -1
        if np.any(moat == marker_color):
            return False

    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        sample = _build_nested_sample(rng)
        if _validate_sample(*sample):
            return sample[0]
    raise RuntimeError("failed to generate a valid nested-hollow input")
