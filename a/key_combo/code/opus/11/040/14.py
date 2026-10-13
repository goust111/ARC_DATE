from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components


BG = Color.BLACK
INPUT_COLOR = Color.BLUE
SMALL_ONLY = Color.RED
LARGE_ONLY = Color.GREEN
OVERLAP = Color.YELLOW

HEIGHT_RANGE = (10, 14)
WIDTH_RANGE = (18, 24)
SMALL_AREA_CHOICES = [3, 4, 5]
MAX_GENERATION_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 160
SHAPE_ATTEMPTS = 120


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_area(coords: np.ndarray) -> int:
    r0, c0, r1, c1 = _bbox(coords)
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _bbox_width(coords: np.ndarray) -> int:
    _r0, c0, _r1, c1 = _bbox(coords)
    return int(c1 - c0 + 1)


def _normalize(coords: np.ndarray) -> np.ndarray:
    pts = np.asarray(coords, dtype=int).copy()
    pts[:, 0] -= int(pts[:, 0].min())
    pts[:, 1] -= int(pts[:, 1].min())
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    return pts[order]


def _shift(coords: np.ndarray, dr: int = 0, dc: int = 0) -> np.ndarray:
    pts = np.asarray(coords, dtype=int).copy()
    pts[:, 0] += int(dr)
    pts[:, 1] += int(dc)
    return pts


def _coord_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.asarray(coords, dtype=int)}


def _random_transform(coords: np.ndarray) -> np.ndarray:
    pts = np.asarray(coords, dtype=int).copy()
    for _ in range(random.randrange(4)):
        pts = np.column_stack((pts[:, 1], -pts[:, 0]))
    if random.random() < 0.5:
        pts[:, 1] *= -1
    if random.random() < 0.5:
        pts[:, 0] *= -1
    return _normalize(pts)


def _random_connected_shape(area: int) -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        cells = {(0, 0)}
        while len(cells) < int(area):
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                frontier.update(
                    {
                        (r - 1, c),
                        (r + 1, c),
                        (r, c - 1),
                        (r, c + 1),
                    }
                )
            frontier.difference_update(cells)
            next_cell = random.choice(tuple(frontier))
            cells.add(next_cell)
        pts = _random_transform(_normalize(np.array(sorted(cells), dtype=int)))
        if _bbox_area(pts) == int(area):
            continue
        r0, c0, r1, c1 = _bbox(pts)
        if r1 == r0 or c1 == c0:
            continue
        if r1 - r0 + 1 > 4 or c1 - c0 + 1 > 4:
            continue
        return pts
    raise RuntimeError(f"failed to build non-rectangular shape of area {area}")


def _paint_mask(mask: np.ndarray, coords: np.ndarray) -> None:
    mask[coords[:, 0], coords[:, 1]] = True


def _render_from_inputs(
    shape_small: np.ndarray,
    shape_large: np.ndarray,
    shift_small: int,
    shift_large: int,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    H, W = map(int, grid_shape)
    small_target = _shift(shape_small, dc=shift_small)
    large_target = _shift(shape_large, dc=-shift_large)
    assert np.all((0 <= small_target[:, 0]) & (small_target[:, 0] < H))
    assert np.all((0 <= large_target[:, 0]) & (large_target[:, 0] < H))
    assert np.all((0 <= small_target[:, 1]) & (small_target[:, 1] < W))
    assert np.all((0 <= large_target[:, 1]) & (large_target[:, 1] < W))

    small_mask = np.zeros((H, W), dtype=bool)
    large_mask = np.zeros((H, W), dtype=bool)
    _paint_mask(small_mask, small_target)
    _paint_mask(large_mask, large_target)

    out = np.full((H, W), BG, dtype=int)
    out[small_mask & ~large_mask] = SMALL_ONLY
    out[large_mask & ~small_mask] = LARGE_ONLY
    out[small_mask & large_mask] = OVERLAP
    return out


def _sorted_blue_components(grid: np.ndarray) -> list[dict[str, object]]:
    comps = find_components(
        np.asarray(grid, dtype=int),
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[INPUT_COLOR],
    )
    return sorted(comps, key=lambda comp: (int(comp["bbox"][1]), int(comp["bbox"][0])))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate the two blue components toward each other by their own areas and recolor overlap versus differences.

    Input structure:
    - The grid has black background and exactly two blue 4-connected foreground components with a clear horizontal gap.
    - Each component is a small irregular shape, and the two shapes have different pixel counts.
    - The left and right components are the only foreground objects.

    Rule:
    - Identify the two blue components and compute their areas.
    - Let S be the smaller-area component and L be the larger-area component; if the areas tie, the left component is S.
    - Erase the originals, move S horizontally right by |S| columns, and move L horizontally left by |L| columns.
    - Repaint the translated result so S-only pixels become red, L-only pixels become green, overlap pixels become yellow, and all other cells stay black.

    Key insight:
    - Each component's area is not just a label: it is the exact displacement that determines where the two translated shapes intersect and where they remain separate.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components = _sorted_blue_components(g)
    assert len(components) == 2, "expected exactly two blue components"
    left, right = components

    if int(left["area"]) <= int(right["area"]):
        small, large = left, right
    else:
        small, large = right, left

    return _render_from_inputs(
        np.asarray(small["coords"], dtype=int),
        np.asarray(large["coords"], dtype=int),
        int(small["area"]),
        int(large["area"]),
        g.shape,
    )


def _validate_input_components(grid: np.ndarray) -> tuple[dict[str, object], dict[str, object]]:
    comps = _sorted_blue_components(grid)
    assert len(comps) == 2, "input must contain exactly two blue components"
    left, right = comps
    assert 3 <= int(left["area"]) <= 6
    assert 3 <= int(right["area"]) <= 6
    assert int(left["area"]) < int(right["area"]), "generator must keep the left component smaller"
    assert _bbox_area(np.asarray(left["coords"], dtype=int)) > int(left["area"])
    assert _bbox_area(np.asarray(right["coords"], dtype=int)) > int(right["area"])
    gap = int(right["bbox"][1]) - int(left["bbox"][3]) - 1
    assert gap >= 4, "input components need a visible gap"
    H, W = map(int, grid.shape)
    assert int(left["bbox"][3]) <= W // 2 - 2
    assert int(right["bbox"][1]) >= W // 2 + 1
    return left, right


def _wrong_shift_variants(
    small: dict[str, object],
    large: dict[str, object],
    grid_shape: tuple[int, int],
) -> list[np.ndarray]:
    shape_small = np.asarray(small["coords"], dtype=int)
    shape_large = np.asarray(large["coords"], dtype=int)
    area_small = int(small["area"])
    area_large = int(large["area"])
    width_small = _bbox_width(shape_small)
    width_large = _bbox_width(shape_large)

    shifts: list[tuple[int, int]] = []
    delta_small = area_small - 1 if area_small > 1 else area_small + 1
    delta_large = area_large - 1 if area_large > 1 else area_large + 1
    shifts.append((delta_small, area_large))
    shifts.append((area_small, delta_large))
    shifts.append((width_small, width_large))
    shifts.append((area_small, area_small))
    shifts.append((width_small, area_large))

    variants: list[np.ndarray] = []
    for shift_small, shift_large in shifts:
        try:
            variants.append(
                _render_from_inputs(shape_small, shape_large, shift_small, shift_large, grid_shape)
            )
        except AssertionError:
            continue
    return variants


def _validate_instance(inp: np.ndarray) -> None:
    g = np.asarray(inp, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert set(map(int, np.unique(g))).issubset({BG, INPUT_COLOR})

    left, right = _validate_input_components(g)
    out = main(g)
    assert out.shape == g.shape
    assert set(map(int, np.unique(out))).issubset({BG, SMALL_ONLY, LARGE_ONLY, OVERLAP})
    assert np.count_nonzero(out == SMALL_ONLY) >= 1
    assert np.count_nonzero(out == LARGE_ONLY) >= 1
    assert np.count_nonzero(out == OVERLAP) >= 1
    assert not np.array_equal(out, g), "transformation must be non-trivial"

    actual = _render_from_inputs(
        np.asarray(left["coords"], dtype=int),
        np.asarray(right["coords"], dtype=int),
        int(left["area"]),
        int(right["area"]),
        g.shape,
    )
    assert np.array_equal(out, actual)

    for wrong in _wrong_shift_variants(left, right, g.shape):
        assert not np.array_equal(out, wrong), "example should depend on the exact encoded areas"


def _candidate_alignment(
    shape_small: np.ndarray,
    shape_large: np.ndarray,
    cell_small: np.ndarray,
    cell_large: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    final_small_rel = _shift(shape_small, dr=-int(cell_small[0]), dc=-int(cell_small[1]))
    final_large_rel = _shift(shape_large, dr=-int(cell_large[0]), dc=-int(cell_large[1]))

    overlap = _coord_set(final_small_rel) & _coord_set(final_large_rel)
    overlap_size = len(overlap)
    if overlap_size == 0 or overlap_size > min(3, len(shape_small) - 1):
        return None

    input_small_rel = _shift(final_small_rel, dc=-len(shape_small))
    input_large_rel = _shift(final_large_rel, dc=len(shape_large))
    if _coord_set(input_small_rel) & _coord_set(input_large_rel):
        return None

    _, _cs0, _rs1, cs1 = _bbox(input_small_rel)
    _rl0, cl0, _rl1, _cl1 = _bbox(input_large_rel)
    gap = int(cl0) - int(cs1) - 1
    if gap < 4:
        return None

    return input_small_rel, input_large_rel


def _anchor_ranges(
    input_small_rel: np.ndarray,
    input_large_rel: np.ndarray,
    H: int,
    W: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    all_rel = np.vstack((input_small_rel, input_large_rel))
    min_r = int(all_rel[:, 0].min())
    max_r = int(all_rel[:, 0].max())
    min_c = int(all_rel[:, 1].min())
    max_c = int(all_rel[:, 1].max())

    row_lo = -min_r
    row_hi = H - 1 - max_r
    col_lo = -min_c
    col_hi = W - 1 - max_c

    _rs0, _cs0, _rs1, cs1 = _bbox(input_small_rel)
    _rl0, cl0, _rl1, _cl1 = _bbox(input_large_rel)
    col_hi = min(col_hi, W // 2 - 2 - int(cs1))
    col_lo = max(col_lo, W // 2 + 1 - int(cl0))

    if row_lo > row_hi or col_lo > col_hi:
        return None
    return (int(row_lo), int(row_hi)), (int(col_lo), int(col_hi))


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*HEIGHT_RANGE)
    W = random.randint(*WIDTH_RANGE)

    area_small = random.choice(SMALL_AREA_CHOICES)
    area_large = random.randint(area_small + 1, 6)
    shape_small = _random_connected_shape(area_small)
    shape_large = _random_connected_shape(area_large)

    small_cells = list(np.asarray(shape_small, dtype=int))
    large_cells = list(np.asarray(shape_large, dtype=int))
    random.shuffle(small_cells)
    random.shuffle(large_cells)

    trials = 0
    for cell_small in small_cells:
        for cell_large in large_cells:
            trials += 1
            if trials > PLACEMENT_ATTEMPTS:
                return None
            candidate = _candidate_alignment(shape_small, shape_large, cell_small, cell_large)
            if candidate is None:
                continue
            input_small_rel, input_large_rel = candidate
            anchor = _anchor_ranges(input_small_rel, input_large_rel, H, W)
            if anchor is None:
                continue
            (row_lo, row_hi), (col_lo, col_hi) = anchor
            row_anchor = random.randint(row_lo, row_hi)
            col_anchor = random.randint(col_lo, col_hi)

            input_small = _shift(input_small_rel, dr=row_anchor, dc=col_anchor)
            input_large = _shift(input_large_rel, dr=row_anchor, dc=col_anchor)
            grid = np.full((H, W), BG, dtype=int)
            grid[input_small[:, 0], input_small[:, 1]] = INPUT_COLOR
            grid[input_large[:, 0], input_large[:, 1]] = INPUT_COLOR
            try:
                _validate_instance(grid)
            except AssertionError:
                continue
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-14 input")
