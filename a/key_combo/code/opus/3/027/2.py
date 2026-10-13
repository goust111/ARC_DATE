from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    neighbors = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in neighbors:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _stamp_centered(out: np.ndarray, patch: np.ndarray, center_r: int, center_c: int, *, transparent: int = BG) -> None:
    h, w = patch.shape
    anchor_r = h // 2
    anchor_c = w // 2
    for pr in range(h):
        for pc in range(w):
            value = int(patch[pr, pc])
            if value == transparent:
                continue
            rr = center_r + pr - anchor_r
            cc = center_c + pc - anchor_c
            if _in_bounds(out.shape, rr, cc):
                out[rr, cc] = value


def _random_dihedral(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, copy=True)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.rot90(out, random.randint(0, 3)).copy()


def _mask_coords(mask: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    coords = np.argwhere(mask != 0)
    return [(int(r0 + r), int(c0 + c)) for r, c in coords]


def _clipped_rect_cells(
    shape: tuple[int, int],
    r0: int,
    c0: int,
    h: int,
    w: int,
    *,
    pad: int = 0,
) -> list[tuple[int, int]]:
    hh, ww = shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(hh - 1, r0 + h - 1 + pad)
    cc1 = min(ww - 1, c0 + w - 1 + pad)
    if rr0 > rr1 or cc0 > cc1:
        return []
    return [(r, c) for r in range(rr0, rr1 + 1) for c in range(cc0, cc1 + 1)]


def _can_place_component(cells: list[tuple[int, int]], occupied: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    for r, c in cells:
        if not _in_bounds(shape, r, c) or (r, c) in occupied:
            return False
    for r, c in cells:
        for dr, dc in NEIGHBORS8 + [(0, 0)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) in occupied:
                return False
    return True


def _can_place_single(
    center: tuple[int, int],
    stamp_shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    grid_shape: tuple[int, int],
) -> bool:
    r, c = center
    if not _in_bounds(grid_shape, r, c) or (r, c) in occupied:
        return False
    h, w = stamp_shape
    r0 = r - h // 2
    c0 = c - w // 2
    for cell in _clipped_rect_cells(grid_shape, r0, c0, h, w, pad=1):
        if cell in occupied or cell in reserved:
            return False
    return True


def _mark_reserved(center: tuple[int, int], stamp_shape: tuple[int, int], reserved: set[tuple[int, int]], grid_shape: tuple[int, int]) -> None:
    r, c = center
    h, w = stamp_shape
    r0 = r - h // 2
    c0 = c - w // 2
    reserved.update(_clipped_rect_cells(grid_shape, r0, c0, h, w, pad=1))


def _connected_under(connectivity: int, mask: np.ndarray) -> bool:
    comps = _find_components(mask.astype(bool), connectivity=connectivity)
    return len(comps) == 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the unique largest 8-connected component as a template, stamp it at isolated single pixels, and erase all other smaller components.

    Input structure:
    - The background is black.
    - All foreground pixels share one non-black color.
    - The foreground splits into one unique largest 8-connected component, several isolated single pixels, and one or more smaller 8-connected multi-pixel components.

    Rule:
    - Label all non-black foreground pixels by 8-connectivity and choose the unique largest component as the template.
    - Crop that template to its bounding box, preserving background holes inside the box.
    - For every size-1 component, center the cropped template on that pixel and overlay only the template's foreground cells, clipping parts that fall outside the grid.
    - For every remaining component whose size is between 2 and the template size minus 1, erase all of its pixels to black.
    - Keep the original template component in place.

    Key insight:
    - Component size is a three-way gate: the largest component defines the reusable stamp, isolated single pixels trigger copies, and all other smaller multi-pixel components are the pieces to delete.
    """
    g = np.asarray(grid, dtype=int)
    fg_colors = sorted({int(v) for v in g.flat if int(v) != BG})
    assert len(fg_colors) == 1, "expected exactly one foreground color"
    color = fg_colors[0]

    components = _find_components(g == color, connectivity=8)
    assert components, "expected at least one foreground component"

    areas = [int(comp.shape[0]) for comp in components]
    template_area = max(areas)
    assert areas.count(template_area) == 1, "largest component must be unique"

    template_coords = components[areas.index(template_area)]
    template_bbox = _bbox_from_coords(template_coords)
    template_patch = _crop_bbox(g, template_bbox)

    singles = [comp for comp in components if int(comp.shape[0]) == 1]
    middles = [comp for comp in components if 1 < int(comp.shape[0]) < template_area]
    assert singles, "expected at least one single-pixel component"
    assert middles, "expected at least one intermediate-size component"

    out = g.copy()
    for comp in middles:
        out[comp[:, 0], comp[:, 1]] = BG
    for comp in singles:
        r, c = map(int, comp[0])
        _stamp_centered(out, template_patch, r, c, transparent=BG)
    return out


TEMPLATE_MASKS = [
    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 1], [1, 1, 0]], dtype=int),
    np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=int),
    np.array([[1, 0, 1], [1, 1, 1], [1, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 1, 0], [0, 1, 1]], dtype=int),
]

DIAGONAL_MIDDLE_MASKS = [
    np.array([[1, 0], [0, 1]], dtype=int),
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int),
    np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]], dtype=int),
]

OTHER_MIDDLE_MASKS = [
    np.array([[1, 1]], dtype=int),
    np.array([[1], [1]], dtype=int),
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[0, 1], [1, 1], [0, 1]], dtype=int),
]

GRID_H_RANGE = (14, 18)
GRID_W_RANGE = (14, 18)
SINGLE_COUNT_RANGE = (2, 4)
MIDDLE_COUNT_RANGE = (1, 3)
GEN_ATTEMPTS = 500


def _build_input_from_plan(plan: dict) -> np.ndarray:
    h, w = plan["shape"]
    color = int(plan["color"])
    grid = np.zeros((h, w), dtype=int)

    tr0, tc0 = plan["template_origin"]
    for r, c in _mask_coords(plan["template_mask"], tr0, tc0):
        grid[r, c] = color

    for item in plan["middles"]:
        for r, c in _mask_coords(item["mask"], item["origin"][0], item["origin"][1]):
            grid[r, c] = color

    for r, c in plan["single_centers"]:
        grid[r, c] = color
    return grid


def _build_expected_output(plan: dict) -> np.ndarray:
    inp = _build_input_from_plan(plan)
    out = inp.copy()
    color = int(plan["color"])
    patch = np.where(plan["template_mask"] != 0, color, BG).astype(int)

    for item in plan["middles"]:
        coords = _mask_coords(item["mask"], item["origin"][0], item["origin"][1])
        for r, c in coords:
            out[r, c] = BG

    for center in plan["single_centers"]:
        _stamp_centered(out, patch, int(center[0]), int(center[1]), transparent=BG)
    return out


def _component_sizes(grid: np.ndarray) -> list[int]:
    color = int(next(v for v in grid.flat if int(v) != BG))
    comps = _find_components(grid == color, connectivity=8)
    return sorted((int(comp.shape[0]) for comp in comps), reverse=True)


def _validate_semantics(plan: dict, inp: np.ndarray, out: np.ndarray) -> None:
    color = int(plan["color"])
    assert inp.ndim == 2 and out.ndim == 2
    assert max(inp.shape) <= 30 and max(out.shape) <= 30
    assert set(map(int, np.unique(inp))) <= {BG, color}
    assert np.count_nonzero(inp == color) > 0
    assert np.any(plan["template_mask"] == 0), "template must not be a solid block"
    assert int(plan["template_mask"].sum()) >= 6

    comps8 = _find_components(inp == color, connectivity=8)
    areas8 = sorted((int(comp.shape[0]) for comp in comps8), reverse=True)
    assert areas8[0] == int(plan["template_mask"].sum())
    assert areas8.count(areas8[0]) == 1
    assert sum(area == 1 for area in areas8) == len(plan["single_centers"]) >= 2
    middle_areas = [area for area in areas8 if 1 < area < areas8[0]]
    assert len(middle_areas) == len(plan["middles"]) >= 1
    assert all(2 <= area <= 4 for area in middle_areas)

    assert any(
        len(_find_components(item["mask"].astype(bool), connectivity=4)) > 1
        for item in plan["middles"]
    ), "at least one middle component must expose 8-connectivity"

    stamp_h, stamp_w = plan["template_mask"].shape
    for r, c in plan["single_centers"]:
        r0 = int(r) - stamp_h // 2
        c0 = int(c) - stamp_w // 2
        fg_cells = [
            (rr, cc)
            for rr, cc in _clipped_rect_cells(inp.shape, r0, c0, stamp_h, stamp_w)
            if int(inp[rr, cc]) == color
        ]
        assert fg_cells == [(int(r), int(c))], "single target footprint must be otherwise empty"

    expected = _build_expected_output(plan)
    assert np.array_equal(out, expected)
    assert not np.array_equal(inp, out), "output must differ from input"


def _try_place_template(plan: dict, occupied: set[tuple[int, int]]) -> bool:
    h, w = plan["shape"]
    mask = plan["template_mask"]
    mh, mw = mask.shape
    row_lo = 2
    row_hi = h - mh - 2
    col_lo = 2
    col_hi = w - mw - 2
    if row_lo > row_hi:
        row_lo, row_hi = 1, h - mh - 1
    if col_lo > col_hi:
        col_lo, col_hi = 1, w - mw - 1
    if row_lo > row_hi or col_lo > col_hi:
        return False

    candidates = [(r0, c0) for r0 in range(row_lo, row_hi + 1) for c0 in range(col_lo, col_hi + 1)]
    random.shuffle(candidates)
    for r0, c0 in candidates:
        cells = _mask_coords(mask, r0, c0)
        if _can_place_component(cells, occupied, plan["shape"]):
            occupied.update(cells)
            plan["template_origin"] = (r0, c0)
            return True
    return False


def _try_place_middles(plan: dict, occupied: set[tuple[int, int]]) -> bool:
    middle_count = random.randint(*MIDDLE_COUNT_RANGE)
    masks = [_random_dihedral(random.choice(DIAGONAL_MIDDLE_MASKS))]
    for _ in range(middle_count - 1):
        masks.append(_random_dihedral(random.choice(DIAGONAL_MIDDLE_MASKS + OTHER_MIDDLE_MASKS)))

    middles = []
    for mask in masks:
        mh, mw = mask.shape
        candidates = [(r0, c0) for r0 in range(plan["shape"][0] - mh + 1) for c0 in range(plan["shape"][1] - mw + 1)]
        random.shuffle(candidates)
        placed = False
        for r0, c0 in candidates:
            cells = _mask_coords(mask, r0, c0)
            if _can_place_component(cells, occupied, plan["shape"]):
                occupied.update(cells)
                middles.append({"mask": mask, "origin": (r0, c0)})
                placed = True
                break
        if not placed:
            return False
    plan["middles"] = middles
    return True


def _try_place_singles(plan: dict, occupied: set[tuple[int, int]]) -> bool:
    single_count = random.randint(*SINGLE_COUNT_RANGE)
    stamp_shape = tuple(map(int, plan["template_mask"].shape))
    reserved: set[tuple[int, int]] = set()
    singles: list[tuple[int, int]] = []

    want_edge = random.random() < 0.35
    for i in range(single_count):
        candidates = []
        for r in range(plan["shape"][0]):
            for c in range(plan["shape"][1]):
                is_edge = r in (0, plan["shape"][0] - 1) or c in (0, plan["shape"][1] - 1)
                if i == 0 and want_edge and not is_edge:
                    continue
                if (i > 0 and is_edge) or (not want_edge and is_edge):
                    continue
                if _can_place_single((r, c), stamp_shape, occupied, reserved, plan["shape"]):
                    candidates.append((r, c))
        if not candidates:
            return False
        center = random.choice(candidates)
        singles.append(center)
        occupied.add(center)
        _mark_reserved(center, stamp_shape, reserved, plan["shape"])

    plan["single_centers"] = singles
    return True


def _gen_attempt() -> np.ndarray | None:
    plan = {
        "shape": (
            random.randint(*GRID_H_RANGE),
            random.randint(*GRID_W_RANGE),
        ),
        "color": random.randint(1, 9),
        "template_mask": _random_dihedral(random.choice(TEMPLATE_MASKS)),
        "template_origin": None,
        "middles": [],
        "single_centers": [],
    }
    occupied: set[tuple[int, int]] = set()

    if not _connected_under(8, plan["template_mask"]) or not _connected_under(4, plan["template_mask"]):
        return None
    if int(plan["template_mask"].sum()) < 6:
        return None
    if not _try_place_template(plan, occupied):
        return None
    if not _try_place_middles(plan, occupied):
        return None
    if not _try_place_singles(plan, occupied):
        return None

    inp = _build_input_from_plan(plan)
    out = main(inp)
    _validate_semantics(plan, inp, out)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
