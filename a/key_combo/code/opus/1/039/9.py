from __future__ import annotations

import random

import numpy as np


BG = 0
OBJ = 1
LINE = 2
RECOLOR = 3
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Generator configuration.
GRID_H_RANGE = (16, 19)
GRID_W_RANGE = (16, 19)
OBJECT_COUNT_RANGE = (5, 7)
HOLED_COUNT_CHOICES = (2, 3, 4)
HOLED_COUNT_WEIGHTS = (0.2, 0.5, 0.3)
OBJECT_BBOX_GAP = 1
MAX_PATCH_ATTEMPTS = 300
MAX_CASE_ATTEMPTS = 600

HOLE_SEEDS = (
    frozenset({(0, 0), (0, 1)}),
    frozenset({(0, 0), (0, 1), (0, 2)}),
    frozenset({(0, 0), (1, 0), (1, 1)}),
    frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
)


def _neighbors4(row: int, col: int) -> tuple[tuple[int, int], ...]:
    return tuple((row + dr, col + dc) for dr, dc in DIR4)


def _component_sets(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    height, width = map(int, mask.shape)
    seen = np.zeros((height, width), dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for row in range(height):
        for col in range(width):
            if not bool(mask[row, col]) or bool(seen[row, col]):
                continue
            stack = [(row, col)]
            seen[row, col] = True
            comp: set[tuple[int, int]] = set()
            while stack:
                r, c = stack.pop()
                comp.add((int(r), int(c)))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _round_half_up(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def _normalize_coords(coords: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> set[tuple[int, int]]:
    rows = [row for row, _col in coords]
    cols = [col for _row, col in coords]
    min_row = min(rows)
    min_col = min(cols)
    return {(int(row) - min_row, int(col) - min_col) for row, col in coords}


def _coords_to_patch(coords: set[tuple[int, int]] | frozenset[tuple[int, int]], color: int = OBJ) -> np.ndarray:
    norm = _normalize_coords(coords)
    height = max(row for row, _col in norm) + 1
    width = max(col for _row, col in norm) + 1
    patch = np.zeros((height, width), dtype=int)
    for row, col in norm:
        patch[row, col] = int(color)
    return patch


def _patch_to_coords(patch: np.ndarray) -> set[tuple[int, int]]:
    return {(int(row), int(col)) for row, col in np.argwhere(np.asarray(patch) != 0)}


def _random_dihedral(coords: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> set[tuple[int, int]]:
    patch = _coords_to_patch(coords)
    if random.random() < 0.5:
        patch = np.fliplr(patch)
    patch = np.rot90(patch, random.randrange(4))
    return _patch_to_coords(patch)


def _bbox_of_coords(coords: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [row for row, _col in coords]
    cols = [col for _row, col in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _analyze_object(coords: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> dict[str, object]:
    coords = set(coords)
    bbox = _bbox_of_coords(coords)
    r0, c0, r1, c1 = bbox
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    patch = np.zeros((height, width), dtype=bool)
    local_coords: set[tuple[int, int]] = set()
    for row, col in coords:
        local = (int(row) - r0, int(col) - c0)
        patch[local] = True
        local_coords.add(local)

    holes_local: list[set[tuple[int, int]]] = []
    for comp in _component_sets(~patch):
        touches_edge = any(
            row == 0 or row == height - 1 or col == 0 or col == width - 1
            for row, col in comp
        )
        if not touches_edge:
            holes_local.append(comp)

    assert len(holes_local) <= 1, "expected at most one enclosed hole per object"

    hole_coords: frozenset[tuple[int, int]] | None = None
    center: tuple[int, int] | None = None
    if holes_local:
        local_hole = holes_local[0]
        hole_coords = frozenset((row + r0, col + c0) for row, col in local_hole)
        center = _hole_center(hole_coords)

    obj_center = (
        _round_half_up(sum(row for row, _col in coords) / len(coords)),
        _round_half_up(sum(col for _row, col in coords) / len(coords)),
    )
    bbox_center = (
        _round_half_up((r0 + r1) / 2),
        _round_half_up((c0 + c1) / 2),
    )
    return {
        "coords": frozenset((int(row), int(col)) for row, col in coords),
        "bbox": tuple(map(int, bbox)),
        "hole_coords": hole_coords,
        "center": center,
        "obj_center": obj_center,
        "bbox_center": bbox_center,
    }


def _hole_center(hole_coords: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> tuple[int, int]:
    rows = [row for row, _col in hole_coords]
    cols = [col for _row, col in hole_coords]
    target = (
        _round_half_up(sum(rows) / len(rows)),
        _round_half_up(sum(cols) / len(cols)),
    )
    if target in hole_coords:
        return target
    return min(
        hole_coords,
        key=lambda cell: ((cell[0] - target[0]) ** 2 + (cell[1] - target[1]) ** 2, cell[0], cell[1]),
    )


def _extract_blue_objects(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    return [_analyze_object(comp) for comp in _component_sets(g == OBJ)]


def _aligned(a: tuple[int, int], b: tuple[int, int]) -> bool:
    dr = abs(int(a[0]) - int(b[0]))
    dc = abs(int(a[1]) - int(b[1]))
    return dr == 0 or dc == 0 or dr == dc


def _mode_matches(mode: str, a: tuple[int, int], b: tuple[int, int]) -> bool:
    if a == b:
        return False
    if mode == "row":
        return int(a[0]) == int(b[0]) and int(a[1]) != int(b[1])
    if mode == "col":
        return int(a[1]) == int(b[1]) and int(a[0]) != int(b[0])
    if mode == "diag":
        return abs(int(a[0]) - int(b[0])) == abs(int(a[1]) - int(b[1]))
    raise ValueError(f"unknown mode: {mode}")


def _bresenham_aligned(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    assert _aligned(a, b), "line endpoints must be row/col/45-degree aligned"
    dr = int(np.sign(int(b[0]) - int(a[0])))
    dc = int(np.sign(int(b[1]) - int(a[1])))
    steps = max(abs(int(b[0]) - int(a[0])), abs(int(b[1]) - int(a[1])))
    return [(int(a[0]) + step * dr, int(a[1]) + step * dc) for step in range(steps + 1)]


def _rules_out_shortcut(
    proxy_a: tuple[int, int],
    proxy_b: tuple[int, int],
    hole_a: tuple[int, int],
    hole_b: tuple[int, int],
) -> bool:
    if not _aligned(proxy_a, proxy_b):
        return True
    return _bresenham_aligned(proxy_a, proxy_b) != _bresenham_aligned(hole_a, hole_b)


def main(grid: np.ndarray) -> np.ndarray:
    """Connect aligned hole centers with red lines and recolor every hole-bearing object green.

    Input structure:
    - The grid has black background and several separated blue 4-connected monochrome objects.
    - Some blue objects contain exactly one enclosed 4-connected black hole, while the rest are solid.
    - Red and green are absent from the input and are reserved for the output overlays.

    Rule:
    - Find every blue 4-connected object and detect whether its bounding-box background contains one enclosed hole.
    - For each hole-bearing object, compute the hole center by rounding the mean row and mean column of the hole pixels; if that rounded point is not itself a hole pixel, snap to the nearest hole pixel with row/column tie-breaks.
    - Recolor all pixels of every hole-bearing object to green, then connect every pair of hole centers that share a row, share a column, or lie on a 45-degree diagonal by drawing a red endpoint-inclusive Bresenham line.
    - Leave solid blue objects unchanged and keep all hole interiors black unless a red line passes through them.

    Key insight:
    - The important anchors are not object centroids or bounding-box centers but the centers of the enclosed negative-space holes hidden inside the qualifying objects.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    infos = _extract_blue_objects(g)
    hole_infos = [info for info in infos if info["center"] is not None]

    out = g.copy()
    for info in hole_infos:
        coords = np.array(sorted(info["coords"]), dtype=int)
        out[coords[:, 0], coords[:, 1]] = RECOLOR

    centers = [tuple(info["center"]) for info in hole_infos]
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            if not _aligned(centers[i], centers[j]):
                continue
            for row, col in _bresenham_aligned(centers[i], centers[j]):
                out[row, col] = LINE

    return out


def _shift_coords(coords: set[tuple[int, int]] | frozenset[tuple[int, int]], top: tuple[int, int]) -> frozenset[tuple[int, int]]:
    tr, tc = map(int, top)
    return frozenset((int(row) + tr, int(col) + tc) for row, col in coords)


def _bbox_from_top(shape: tuple[int, int], top: tuple[int, int]) -> tuple[int, int, int, int]:
    row, col = map(int, top)
    height, width = map(int, shape)
    return row, col, row + height - 1, col + width - 1


def _bbox_clear(existing: list[tuple[int, int, int, int]], bbox: tuple[int, int, int, int], gap: int = OBJECT_BBOX_GAP) -> bool:
    r0, c0, r1, c1 = map(int, bbox)
    for er0, ec0, er1, ec1 in existing:
        if not (r1 + gap < er0 or er1 + gap < r0 or c1 + gap < ec0 or ec1 + gap < c0):
            return False
    return True


def _frontier4(coords: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> set[tuple[int, int]]:
    frontier: set[tuple[int, int]] = set()
    for row, col in coords:
        for nbr in _neighbors4(row, col):
            if nbr not in coords:
                frontier.add(nbr)
    return frontier


def _is_single_component(coords: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> bool:
    patch = _coords_to_patch(coords, color=1)
    return len(_component_sets(patch == 1)) == 1


def _patch_meets_hole_rules(coords: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> bool:
    coords = _normalize_coords(coords)
    if not (5 <= len(coords) <= 12):
        return False
    r0, c0, r1, c1 = _bbox_of_coords(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if max(height, width) > 6:
        return False
    if len(coords) == height * width:
        return False
    if not _is_single_component(coords):
        return False
    info = _analyze_object(coords)
    if info["center"] is None:
        return False
    return len(info["hole_coords"]) in (2, 3, 4)


def _patch_meets_solid_rules(coords: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> bool:
    coords = _normalize_coords(coords)
    if not (5 <= len(coords) <= 12):
        return False
    r0, c0, r1, c1 = _bbox_of_coords(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if max(height, width) > 6 or min(height, width) < 2:
        return False
    if len(coords) == height * width:
        return False
    if not _is_single_component(coords):
        return False
    info = _analyze_object(coords)
    return info["center"] is None


def _generate_holed_patch() -> dict[str, object]:
    for _ in range(MAX_PATCH_ATTEMPTS):
        hole = set(random.choice(HOLE_SEEDS))
        object_cells: set[tuple[int, int]] = set()
        for row, col in hole:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    cell = (row + dr, col + dc)
                    if cell not in hole:
                        object_cells.add(cell)
        object_cells = _normalize_coords(object_cells)

        for _edit in range(random.randint(1, 4)):
            info = _analyze_object(object_cells)
            current_hole = set(info["hole_coords"]) if info["hole_coords"] is not None else set()
            protected = set()
            for row, col in current_hole:
                for nbr in _neighbors4(row, col):
                    if nbr in object_cells:
                        protected.add(nbr)

            if random.random() < 0.55:
                removable = list(object_cells - protected)
                random.shuffle(removable)
                for cell in removable:
                    trial = _normalize_coords(object_cells - {cell})
                    if _patch_meets_hole_rules(trial):
                        object_cells = trial
                        break
            else:
                if len(object_cells) >= 12:
                    continue
                additions = list(_frontier4(object_cells) - current_hole)
                random.shuffle(additions)
                for cell in additions:
                    trial = _normalize_coords(object_cells | {cell})
                    if _patch_meets_hole_rules(trial):
                        object_cells = trial
                        break

        object_cells = _random_dihedral(object_cells)
        object_cells = _normalize_coords(object_cells)
        if not _patch_meets_hole_rules(object_cells):
            continue

        info = _analyze_object(object_cells)
        patch = _coords_to_patch(object_cells)
        return {
            "patch": patch,
            "coords": frozenset(object_cells),
            "center": tuple(info["center"]),
            "obj_center": tuple(info["obj_center"]),
            "bbox_center": tuple(info["bbox_center"]),
            "hole_coords": frozenset(info["hole_coords"]),
        }
    raise RuntimeError("failed to generate a holed patch")


def _generate_solid_patch() -> dict[str, object]:
    for _ in range(MAX_PATCH_ATTEMPTS):
        target_area = random.randint(5, 12)
        coords: set[tuple[int, int]] = {(0, 0)}
        while len(coords) < target_area:
            frontier = list(_frontier4(coords))
            random.shuffle(frontier)
            added = False
            for cell in frontier:
                trial = _normalize_coords(coords | {cell})
                r0, c0, r1, c1 = _bbox_of_coords(trial)
                if max(r1 - r0 + 1, c1 - c0 + 1) <= 6:
                    coords = trial
                    added = True
                    break
            if not added:
                break
        coords = _random_dihedral(coords)
        coords = _normalize_coords(coords)
        if not _patch_meets_solid_rules(coords):
            continue
        patch = _coords_to_patch(coords)
        return {"patch": patch, "coords": frozenset(coords)}
    raise RuntimeError("failed to generate a solid patch")


def _shift_point(point: tuple[int, int], top: tuple[int, int]) -> tuple[int, int]:
    return int(point[0]) + int(top[0]), int(point[1]) + int(top[1])


def _choose_aligned_pair_positions(
    patch_a: dict[str, object],
    patch_b: dict[str, object],
    height: int,
    width: int,
    mode: str,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    shape_a = tuple(map(int, np.asarray(patch_a["patch"]).shape))
    shape_b = tuple(map(int, np.asarray(patch_b["patch"]).shape))
    hole_off_a = tuple(map(int, patch_a["center"]))
    hole_off_b = tuple(map(int, patch_b["center"]))

    row_marks = sorted(
        {
            3,
            4,
            max(3, height // 2 - 1),
            min(height - 4, height // 2 + 1),
            height - 5,
            height - 4,
        }
    )
    col_marks = sorted(
        {
            3,
            4,
            max(3, width // 2 - 1),
            min(width - 4, width // 2 + 1),
            width - 5,
            width - 4,
        }
    )

    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    if mode == "row":
        left_cols = col_marks[: max(2, len(col_marks) // 2)]
        right_cols = col_marks[max(1, len(col_marks) // 2) :]
        for row in row_marks:
            for col_a in left_cols:
                for col_b in right_cols:
                    if abs(col_a - col_b) >= 5:
                        candidates.append(((row, col_a), (row, col_b)))
    elif mode == "col":
        top_rows = row_marks[: max(2, len(row_marks) // 2)]
        bottom_rows = row_marks[max(1, len(row_marks) // 2) :]
        for col in col_marks:
            for row_a in top_rows:
                for row_b in bottom_rows:
                    if abs(row_a - row_b) >= 5:
                        candidates.append(((row_a, col), (row_b, col)))
    elif mode == "diag":
        for row in row_marks:
            for col in col_marks:
                for delta in range(4, min(height, width)):
                    for dr, dc in ((delta, delta), (delta, -delta)):
                        row_b = row + dr
                        col_b = col + dc
                        if 0 <= row_b < height and 0 <= col_b < width:
                            candidates.append(((row, col), (row_b, col_b)))
    else:
        raise ValueError(f"unknown mode: {mode}")

    random.shuffle(candidates)
    for hole_a, hole_b in candidates:
        top_a = (int(hole_a[0]) - hole_off_a[0], int(hole_a[1]) - hole_off_a[1])
        top_b = (int(hole_b[0]) - hole_off_b[0], int(hole_b[1]) - hole_off_b[1])
        if top_a[0] < 0 or top_a[1] < 0 or top_b[0] < 0 or top_b[1] < 0:
            continue
        if top_a[0] + shape_a[0] > height or top_a[1] + shape_a[1] > width:
            continue
        if top_b[0] + shape_b[0] > height or top_b[1] + shape_b[1] > width:
            continue

        bbox_a = _bbox_from_top(shape_a, top_a)
        bbox_b = _bbox_from_top(shape_b, top_b)
        if not _bbox_clear([bbox_a], bbox_b):
            continue

        obj_a = _shift_point(tuple(patch_a["obj_center"]), top_a)
        obj_b = _shift_point(tuple(patch_b["obj_center"]), top_b)
        box_a = _shift_point(tuple(patch_a["bbox_center"]), top_a)
        box_b = _shift_point(tuple(patch_b["bbox_center"]), top_b)
        if not _rules_out_shortcut(obj_a, obj_b, hole_a, hole_b):
            continue
        if not _rules_out_shortcut(box_a, box_b, hole_a, hole_b):
            continue
        return top_a, top_b
    return None


def _place_patch_on_grid(grid: np.ndarray, patch: np.ndarray, top: tuple[int, int]) -> frozenset[tuple[int, int]]:
    row0, col0 = map(int, top)
    coords = []
    for row, col in np.argwhere(patch == OBJ):
        rr = row0 + int(row)
        cc = col0 + int(col)
        grid[rr, cc] = OBJ
        coords.append((rr, cc))
    return frozenset(coords)


def _metadata_expected_output(grid: np.ndarray, objects: list[dict[str, object]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    holed = [obj for obj in objects if obj["kind"] == "hole"]
    for obj in holed:
        coords = np.array(sorted(obj["coords"]), dtype=int)
        out[coords[:, 0], coords[:, 1]] = RECOLOR
    centers = [tuple(obj["center"]) for obj in holed]
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            if not _aligned(centers[i], centers[j]):
                continue
            for row, col in _bresenham_aligned(centers[i], centers[j]):
                out[row, col] = LINE
    return out


def _validate_case(grid: np.ndarray, objects: list[dict[str, object]], forced_pair: tuple[int, int]) -> bool:
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert max(grid.shape) <= 30
    assert set(map(int, np.unique(grid))) <= {BG, OBJ}

    object_bboxes = [tuple(obj["bbox"]) for obj in objects]
    for i in range(len(object_bboxes)):
        for j in range(i + 1, len(object_bboxes)):
            assert _bbox_clear([object_bboxes[i]], object_bboxes[j], gap=0), "object bboxes must not overlap"

    extracted = _extract_blue_objects(grid)
    assert len(extracted) == len(objects)
    by_bbox = {tuple(info["bbox"]): info for info in extracted}
    for obj in objects:
        info = by_bbox[tuple(obj["bbox"])]
        assert info["coords"] == obj["coords"]
        if obj["kind"] == "hole":
            assert info["center"] == obj["center"]
            assert len(info["hole_coords"]) in (2, 3, 4)
        else:
            assert info["center"] is None

    holed = [obj for obj in objects if obj["kind"] == "hole"]
    assert 2 <= len(holed) <= 4
    pair_centers = [tuple(obj["center"]) for obj in holed]
    aligned_pairs: list[tuple[int, int]] = []
    unaligned_pairs = 0
    for i in range(len(pair_centers)):
        for j in range(i + 1, len(pair_centers)):
            if _aligned(pair_centers[i], pair_centers[j]):
                aligned_pairs.append((i, j))
            else:
                unaligned_pairs += 1
    assert aligned_pairs, "at least one hole-center pair must be aligned"
    if len(holed) >= 3:
        assert unaligned_pairs >= 1, "need an unaligned hole-center pair to exercise the filter"

    fi, fj = forced_pair
    forced_a = holed[fi]
    forced_b = holed[fj]
    assert _aligned(tuple(forced_a["center"]), tuple(forced_b["center"]))
    assert _rules_out_shortcut(
        tuple(forced_a["obj_proxy"]),
        tuple(forced_b["obj_proxy"]),
        tuple(forced_a["center"]),
        tuple(forced_b["center"]),
    )
    assert _rules_out_shortcut(
        tuple(forced_a["bbox_proxy"]),
        tuple(forced_b["bbox_proxy"]),
        tuple(forced_a["center"]),
        tuple(forced_b["center"]),
    )

    all_object_pixels = set().union(*(set(obj["coords"]) for obj in objects))
    for i, j in aligned_pairs:
        line_cells = set(_bresenham_aligned(pair_centers[i], pair_centers[j]))
        allowed = set(holed[i]["coords"]) | set(holed[j]["coords"])
        assert not (line_cells & (all_object_pixels - allowed)), "lines should not cross unrelated objects"

    expected = _metadata_expected_output(grid, objects)
    solved = main(grid)
    assert np.array_equal(solved, expected)
    assert set(map(int, np.unique(solved))) <= {BG, OBJ, LINE, RECOLOR}
    return True


def _generate_case() -> tuple[np.ndarray, list[dict[str, object]]]:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    object_count = random.randint(*OBJECT_COUNT_RANGE)
    holed_count = random.choices(HOLED_COUNT_CHOICES, weights=HOLED_COUNT_WEIGHTS, k=1)[0]
    holed_count = min(int(holed_count), object_count - 1)
    solid_count = object_count - holed_count

    hole_patches = [_generate_holed_patch() for _ in range(holed_count)]
    solid_patches = [_generate_solid_patch() for _ in range(solid_count)]

    guaranteed_pair = None
    modes = ["row", "col", "diag"]
    random.shuffle(modes)
    for i in range(holed_count):
        for j in range(i + 1, holed_count):
            random.shuffle(modes)
            for mode in modes:
                pair = _choose_aligned_pair_positions(hole_patches[i], hole_patches[j], height, width, mode)
                if pair is not None:
                    guaranteed_pair = (i, j, pair[0], pair[1])
                    break
            if guaranteed_pair is not None:
                break
        if guaranteed_pair is not None:
            break
    if guaranteed_pair is None:
        raise RuntimeError("failed to place the guaranteed aligned pair")

    grid = np.zeros((height, width), dtype=int)
    objects: list[dict[str, object]] = []
    placed_bboxes: list[tuple[int, int, int, int]] = []
    forced_centers: list[tuple[int, int]] = []

    i0, i1, top0, top1 = guaranteed_pair
    for patch_index, top in ((i0, top0), (i1, top1)):
        patch = hole_patches[patch_index]
        coords = _place_patch_on_grid(grid, np.asarray(patch["patch"], dtype=int), top)
        info = _analyze_object(coords)
        obj = {
            "kind": "hole",
            "coords": coords,
            "bbox": tuple(info["bbox"]),
            "center": tuple(info["center"]),
            "obj_proxy": _shift_point(tuple(patch["obj_center"]), top),
            "bbox_proxy": _shift_point(tuple(patch["bbox_center"]), top),
        }
        objects.append(obj)
        placed_bboxes.append(tuple(info["bbox"]))
        forced_centers.append(tuple(info["center"]))

    for idx, patch in enumerate(hole_patches):
        if idx in (i0, i1):
            continue
        shape = tuple(map(int, np.asarray(patch["patch"]).shape))
        positions = [(row, col) for row in range(height - shape[0] + 1) for col in range(width - shape[1] + 1)]
        random.shuffle(positions)
        placed = False
        for top in positions:
            bbox = _bbox_from_top(shape, top)
            if not _bbox_clear(placed_bboxes, bbox):
                continue
            coords = _place_patch_on_grid(grid, np.asarray(patch["patch"], dtype=int), top)
            info = _analyze_object(coords)
            objects.append(
                {
                    "kind": "hole",
                    "coords": coords,
                    "bbox": tuple(info["bbox"]),
                    "center": tuple(info["center"]),
                    "obj_proxy": _shift_point(tuple(patch["obj_center"]), top),
                    "bbox_proxy": _shift_point(tuple(patch["bbox_center"]), top),
                }
            )
            placed_bboxes.append(tuple(info["bbox"]))
            placed = True
            break
        if not placed:
            raise RuntimeError("failed to place an extra holed patch")

    for patch in solid_patches:
        shape = tuple(map(int, np.asarray(patch["patch"]).shape))
        positions = [(row, col) for row in range(height - shape[0] + 1) for col in range(width - shape[1] + 1)]
        random.shuffle(positions)
        placed = False
        for top in positions:
            bbox = _bbox_from_top(shape, top)
            if not _bbox_clear(placed_bboxes, bbox):
                continue
            coords = _place_patch_on_grid(grid, np.asarray(patch["patch"], dtype=int), top)
            objects.append(
                {
                    "kind": "solid",
                    "coords": coords,
                    "bbox": tuple(_bbox_of_coords(coords)),
                }
            )
            placed_bboxes.append(tuple(_bbox_of_coords(coords)))
            placed = True
            break
        if not placed:
            raise RuntimeError("failed to place a solid patch")

    random.shuffle(objects)
    holed = [obj for obj in objects if obj["kind"] == "hole"]
    forced_pair_actual = [idx for idx, obj in enumerate(holed) if tuple(obj["center"]) in set(forced_centers)]
    assert len(forced_pair_actual) == 2
    _validate_case(grid, objects, tuple(sorted(forced_pair_actual)))
    return grid, objects


def generate_input() -> np.ndarray:
    for _ in range(MAX_CASE_ATTEMPTS):
        try:
            grid, _objects = _generate_case()
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle input")
