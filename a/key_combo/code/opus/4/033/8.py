from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


def _load_arc_utils():
    arc_utils_path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("arc_utils_shared", arc_utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load arc_utils from {arc_utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC = _load_arc_utils()
Color = _ARC.Color
find_components = _ARC.find_components
bbox_coords = _ARC.bbox_coords


def _outline_bbox_perimeter(out: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    h, w = out.shape
    r0, c0, r1, c1 = bbox
    er0 = max(0, r0 - 1)
    ec0 = max(0, c0 - 1)
    er1 = min(h - 1, r1 + 1)
    ec1 = min(w - 1, c1 + 1)
    for r in range(er0, er1 + 1):
        for c in range(ec0, ec1 + 1):
            if r in (er0, er1) or c in (ec0, ec1):
                if int(out[r, c]) == Color.BLACK:
                    out[r, c] = Color.GREEN


def _gray_components(grid: np.ndarray) -> list[dict]:
    return find_components(
        np.asarray(grid, dtype=int),
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=True,
        colors=[Color.GREY],
        return_format="dict",
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase single-pixel grey noise, recolor the unique largest and smallest surviving components, then outline all survivors.

    Input structure:
    - The grid is a black background containing several grey 4-connected components.
    - Some grey components are single-pixel noise, while the others are multi-pixel objects of varied shapes and sizes.
    - Components are spaced so that their expanded bounding-box outlines do not overlap.

    Rule:
    - Find every grey 4-connected component and erase each area-1 component to black.
    - Among the remaining components, recolor the unique largest-area component blue and the unique smallest-area component red; keep the other surviving components grey.
    - For every surviving component, expand its axis-aligned bounding box by one cell in each direction, clip to the grid, and paint the perimeter green wherever that perimeter cell is currently black.
    - Apply the outline step in row-major order of the surviving components' bounding-box top-left corners.

    Key insight:
    - The single-pixel grey noise must be removed before ranking areas, because that filtering changes which surviving component becomes the red minimum-area extreme.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    components = _gray_components(g)
    survivors = [comp for comp in components if int(comp["area"]) > 1]

    for comp in components:
        if int(comp["area"]) == 1:
            coords = comp["coords"]
            out[coords[:, 0], coords[:, 1]] = Color.BLACK

    assert len(survivors) >= 2, "expected at least two surviving multi-pixel components"
    areas = [int(comp["area"]) for comp in survivors]
    assert len(set(areas)) == len(areas), "surviving component areas must be unique"

    max_comp = max(survivors, key=lambda comp: int(comp["area"]))
    min_comp = min(survivors, key=lambda comp: int(comp["area"]))
    out[max_comp["coords"][:, 0], max_comp["coords"][:, 1]] = Color.BLUE
    out[min_comp["coords"][:, 0], min_comp["coords"][:, 1]] = Color.RED

    for comp in sorted(survivors, key=lambda comp: (int(comp["bbox"][0]), int(comp["bbox"][1]))):
        _outline_bbox_perimeter(out, tuple(map(int, comp["bbox"])))

    return out


def _shape_line(area: int) -> np.ndarray:
    if random.random() < 0.5:
        return np.ones((1, area), dtype=bool)
    return np.ones((area, 1), dtype=bool)


def _shape_rectangle(area: int) -> np.ndarray:
    factors = [(h, area // h) for h in range(2, area + 1) if area % h == 0 and area // h >= 2]
    if not factors:
        return _shape_line(area)
    h, w = random.choice(factors)
    return np.ones((h, w), dtype=bool)


def _shape_blob(area: int) -> np.ndarray:
    cells = {(0, 0)}
    frontier = {(0, 1), (1, 0), (-1, 0), (0, -1)}
    while len(cells) < area:
        r, c = random.choice(tuple(frontier))
        frontier.remove((r, c))
        cells.add((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if (nr, nc) not in cells:
                frontier.add((nr, nc))

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    rmin, rmax = min(rows), max(rows)
    cmin, cmax = min(cols), max(cols)
    mask = np.zeros((rmax - rmin + 1, cmax - cmin + 1), dtype=bool)
    for r, c in cells:
        mask[r - rmin, c - cmin] = True
    return mask


def _make_shape(area: int) -> np.ndarray:
    choices = ["blob", "blob", "rect", "line"]
    if area <= 3:
        choices.extend(["line", "line"])
    if area in {4, 6, 8, 9, 10, 12, 14, 15}:
        choices.append("rect")
    kind = random.choice(choices)
    if kind == "line":
        return _shape_line(area)
    if kind == "rect":
        return _shape_rectangle(area)
    return _shape_blob(area)


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int]:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_gap = max(0, max(br0 - ar1 - 1, ar0 - br1 - 1))
    col_gap = max(0, max(bc0 - ac1 - 1, ac0 - bc1 - 1))
    return row_gap, col_gap


def _can_place(
    grid: np.ndarray,
    shape: np.ndarray,
    top: int,
    left: int,
    boxes: list[tuple[int, int, int, int]],
    min_gap: int = 2,
) -> bool:
    h, w = shape.shape
    if np.any(grid[top : top + h, left : left + w][shape] != Color.BLACK):
        return False
    new_box = (top, left, top + h - 1, left + w - 1)
    for box in boxes:
        row_gap, col_gap = _bbox_gap(new_box, box)
        if row_gap < min_gap and col_gap < min_gap:
            return False
    return True


def _paste_shape(grid: np.ndarray, shape: np.ndarray, top: int, left: int, color: int) -> None:
    sub = grid[top : top + shape.shape[0], left : left + shape.shape[1]]
    sub[shape] = color


def _find_between_singleton_spot(
    grid: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int] | None:
    multi_boxes = boxes[:]
    random.shuffle(multi_boxes)
    for i in range(len(multi_boxes)):
        for j in range(i + 1, len(multi_boxes)):
            a = multi_boxes[i]
            b = multi_boxes[j]
            ar0, ac0, ar1, ac1 = a
            br0, bc0, br1, bc1 = b
            row_overlap0 = max(ar0, br0)
            row_overlap1 = min(ar1, br1)
            if row_overlap0 <= row_overlap1:
                left_box, right_box = (a, b) if ac0 < bc0 else (b, a)
                gap = right_box[1] - left_box[3] - 1
                if gap >= 3:
                    candidate_rows = list(range(row_overlap0, row_overlap1 + 1))
                    random.shuffle(candidate_rows)
                    for r in candidate_rows:
                        c = (left_box[3] + right_box[1]) // 2
                        if grid[r, c] != Color.BLACK:
                            continue
                        if _singleton_clear(grid, r, c):
                            return r, c
            col_overlap0 = max(ac0, bc0)
            col_overlap1 = min(ac1, bc1)
            if col_overlap0 <= col_overlap1:
                top_box, bottom_box = (a, b) if ar0 < br0 else (b, a)
                gap = bottom_box[0] - top_box[2] - 1
                if gap >= 3:
                    candidate_cols = list(range(col_overlap0, col_overlap1 + 1))
                    random.shuffle(candidate_cols)
                    for c in candidate_cols:
                        r = (top_box[2] + bottom_box[0]) // 2
                        if grid[r, c] != Color.BLACK:
                            continue
                        if _singleton_clear(grid, r, c):
                            return r, c
    return None


def _singleton_clear(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    if grid[r, c] != Color.BLACK:
        return False
    for nr in range(max(0, r - 1), min(h, r + 2)):
        for nc in range(max(0, c - 1), min(w, c + 2)):
            if (nr != r or nc != c) and grid[nr, nc] == Color.GREY:
                return False
    return True


def _place_random_singletons(
    grid: np.ndarray,
    count: int,
    reserved: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    candidates = [(r, c) for r in range(h) for c in range(w) if (r, c) not in reserved]
    random.shuffle(candidates)
    placed = 0
    for r, c in candidates:
        if _singleton_clear(grid, r, c):
            grid[r, c] = Color.GREY
            placed += 1
            if placed == count:
                return True
    return False


def _has_between_singleton(grid: np.ndarray, multi_boxes: list[tuple[int, int, int, int]]) -> bool:
    singles = [tuple(map(int, comp["coords"][0])) for comp in _gray_components(grid) if int(comp["area"]) == 1]
    for sr, sc in singles:
        for i in range(len(multi_boxes)):
            for j in range(i + 1, len(multi_boxes)):
                a = multi_boxes[i]
                b = multi_boxes[j]
                ar0, ac0, ar1, ac1 = a
                br0, bc0, br1, bc1 = b
                row_overlap0 = max(ar0, br0)
                row_overlap1 = min(ar1, br1)
                if row_overlap0 <= sr <= row_overlap1:
                    left_box, right_box = (a, b) if ac0 < bc0 else (b, a)
                    if left_box[3] < sc < right_box[1]:
                        return True
                col_overlap0 = max(ac0, bc0)
                col_overlap1 = min(ac1, bc1)
                if col_overlap0 <= sc <= col_overlap1:
                    top_box, bottom_box = (a, b) if ar0 < br0 else (b, a)
                    if top_box[2] < sr < bottom_box[0]:
                        return True
    return False


def _semantic_checks(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not set(map(int, np.unique(grid))).issubset({Color.BLACK, Color.GREY}):
        return False

    components = _gray_components(grid)
    if not 6 <= len(components) <= 10:
        return False

    singles = [comp for comp in components if int(comp["area"]) == 1]
    multis = [comp for comp in components if int(comp["area"]) > 1]
    if not 2 <= len(singles) <= 4:
        return False
    if len(multis) < 4:
        return False

    areas = [int(comp["area"]) for comp in multis]
    if len(set(areas)) != len(areas):
        return False
    if min(areas) <= 1 or max(areas) > 15:
        return False

    for i, comp_a in enumerate(components):
        for comp_b in components[i + 1 :]:
            row_gap, col_gap = _bbox_gap(tuple(map(int, comp_a["bbox"])), tuple(map(int, comp_b["bbox"])))
            if row_gap < 2 and col_gap < 2:
                return False

    if not _has_between_singleton(grid, [tuple(map(int, comp["bbox"])) for comp in multis]):
        return False

    smallest_all = min(int(comp["area"]) for comp in components)
    smallest_survivor = min(areas)
    if smallest_all != 1 or smallest_survivor == 1:
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if np.array_equal(out, grid):
        return False
    if np.count_nonzero(out == Color.RED) != smallest_survivor:
        return False
    if np.count_nonzero(out == Color.BLUE) != max(areas):
        return False
    if np.count_nonzero((grid == Color.GREY) & (out == Color.BLACK)) != len(singles):
        return False
    if np.count_nonzero(out == Color.GREEN) == 0:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(15, 22)
    w = random.randint(15, 22)
    grid = np.full((h, w), Color.BLACK, dtype=int)

    multi_count = random.randint(4, 6)
    single_count = random.randint(2, 4)
    total_count = multi_count + single_count
    if total_count > 10:
        return None

    possible_areas = list(range(2, 16))
    multi_areas = sorted(random.sample(possible_areas, multi_count))
    random.shuffle(multi_areas)

    boxes: list[tuple[int, int, int, int]] = []
    for area in multi_areas:
        shape = _make_shape(area)
        sh, sw = shape.shape
        positions = [(r, c) for r in range(h - sh + 1) for c in range(w - sw + 1)]
        random.shuffle(positions)
        placed = False
        for top, left in positions:
            if _can_place(grid, shape, top, left, boxes):
                _paste_shape(grid, shape, top, left, Color.GREY)
                boxes.append((top, left, top + sh - 1, left + sw - 1))
                placed = True
                break
        if not placed:
            return None

    spot = _find_between_singleton_spot(grid, boxes)
    if spot is None:
        return None
    reserved = {spot}
    grid[spot] = Color.GREY

    if not _place_random_singletons(grid, single_count - 1, reserved):
        return None

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 8")
