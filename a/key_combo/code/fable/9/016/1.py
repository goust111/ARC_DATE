import random
from collections import defaultdict

import numpy as np


BLACK = 0
DIVIDER = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
UNKNOWN = -1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _rectangles_from_mask(mask: np.ndarray) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    rects = []
    for coords in _components(mask):
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert coords.shape[0] == h * w, "expected dense rectangle"
        rects.append((box, coords))
    return rects


def _tile_for_period(grid: np.ndarray, known: np.ndarray, pr: int, pc: int) -> np.ndarray | None:
    tile = np.full((pr, pc), UNKNOWN, dtype=int)
    for r, c in known:
        rr = int(r % pr)
        cc = int(c % pc)
        color = int(grid[r, c])
        if tile[rr, cc] == UNKNOWN:
            tile[rr, cc] = color
        elif tile[rr, cc] != color:
            return None
    if np.any(tile == UNKNOWN):
        return None
    return tile


def _infer_periodic_tile(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    known = np.argwhere(g != BLACK)
    assert known.size > 0, "expected known periodic cells"

    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for pr in range(1, h + 1):
        for pc in range(1, w + 1):
            tile = _tile_for_period(g, known, pr, pc)
            if tile is not None:
                candidates.append((pr * pc, pr, pc, tile))

    assert candidates, "no periodic explanation found"
    min_area = min(area for area, _pr, _pc, _tile in candidates)
    best = [(pr, pc, tile) for area, pr, pc, tile in candidates if area == min_area]
    assert len(best) == 1, "expected unique minimum-area tile"
    return best[0]


def _fill_by_tile(shape: tuple[int, int], pr: int, pc: int, tile: np.ndarray) -> np.ndarray:
    h, w = shape
    rr = np.arange(h)[:, None] % pr
    cc = np.arange(w)[None, :] % pc
    return tile[rr, cc].astype(int)


def _predicted_patch(full_prediction: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    return _crop(full_prediction, box)


def _find_divider_row(grid: np.ndarray) -> int:
    hits = [r for r in range(grid.shape[0]) if np.all(grid[r] == DIVIDER)]
    assert len(hits) == 1, "expected one grey divider row"
    return int(hits[0])


def main(grid: np.ndarray) -> np.ndarray:
    """Return the periodic top area with tray patches restored into their phase-matching holes.

    Input structure:
    - A single full-width grey divider row splits the grid into an upper periodic pattern area and a lower black tray.
    - The upper area contains several rectangular black holes cut out of a non-black 2D periodic pattern.
    - The tray contains separated dense rectangular patch blocks, each with the same size as some hole.

    Rule:
    - Remove the divider and tray; the output canvas is exactly the upper area.
    - Infer the unique smallest rectangular period from the upper area's non-black cells, then use it to predict every hole.
    - For each tray patch, find the unique same-sized hole whose periodic prediction differs from the patch in exactly one cell, and paste that patch back unchanged; any remaining holes are filled directly from the periodic prediction.

    Key insight:
    - The tray blocks are not generic fillers: each must be matched by phase against the periodic pattern, and its one defect pixel must survive in the final image.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    divider = _find_divider_row(g)
    top = g[:divider]
    tray = g[divider + 1 :]
    assert top.size > 0 and tray.size > 0

    pr, pc, tile = _infer_periodic_tile(top)
    periodic = _fill_by_tile(top.shape, pr, pc, tile)

    holes = []
    for box, _coords in _rectangles_from_mask(top == BLACK):
        holes.append({"box": box, "shape": (box[2] - box[0] + 1, box[3] - box[1] + 1)})
    assert holes, "expected at least one hole"

    patches = []
    for box, _coords in _rectangles_from_mask(tray != BLACK):
        patch = _crop(tray, box)
        assert np.all(patch != BLACK), "tray patch must be dense and non-black"
        patches.append({"box": box, "shape": patch.shape, "grid": patch})
    assert patches, "expected at least one tray patch"

    assignments: dict[int, np.ndarray] = {}
    used_holes: set[int] = set()
    for patch in patches:
        matches = []
        for idx, hole in enumerate(holes):
            if hole["shape"] != patch["shape"]:
                continue
            predicted = _predicted_patch(periodic, hole["box"])
            if int(np.count_nonzero(predicted != patch["grid"])) == 1:
                matches.append(idx)
        assert len(matches) == 1, "patch must have one phase-consistent hole"
        hole_idx = matches[0]
        assert hole_idx not in used_holes, "two patches matched the same hole"
        used_holes.add(hole_idx)
        assignments[hole_idx] = patch["grid"]

    out = periodic.copy()
    for idx, patch in assignments.items():
        r0, c0, r1, c1 = holes[idx]["box"]
        out[r0 : r1 + 1, c0 : c1 + 1] = patch
    return out


def _random_tile(pr: int, pc: int, colors: list[int]) -> np.ndarray:
    return np.array([[random.choice(colors) for _ in range(pc)] for _ in range(pr)], dtype=int)


def _generate_periodic_output() -> tuple[np.ndarray, int, int, np.ndarray]:
    for _ in range(200):
        h = random.randint(10, 16)
        w = random.randint(12, 18)
        pr = random.randint(2, min(5, h - 2))
        pc = random.randint(2, min(5, w - 2))
        n_colors = random.randint(3, min(6, len(COLOR_POOL)))
        colors = random.sample(COLOR_POOL, n_colors)
        tile = _random_tile(pr, pc, colors)
        full = _fill_by_tile((h, w), pr, pc, tile)
        try:
            got_pr, got_pc, got_tile = _infer_periodic_tile(full)
        except AssertionError:
            continue
        if got_pr == pr and got_pc == pc and np.array_equal(got_tile, tile):
            return full, pr, pc, tile
    raise RuntimeError("failed to generate a unique periodic output")


def _holes_obey_residue_budget(top: np.ndarray, pr: int, pc: int, min_count: int = 2) -> bool:
    counts = np.zeros((pr, pc), dtype=int)
    h, w = top.shape
    for r in range(h):
        for c in range(w):
            if top[r, c] != BLACK:
                counts[r % pr, c % pc] += 1
    return int(counts.min()) >= min_count


def _place_holes(full: np.ndarray, pr: int, pc: int) -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    h, w = full.shape
    hole_count = random.randint(2, 4)
    base_shape = (random.randint(2, 4), random.randint(2, 4))
    shapes = [base_shape, base_shape]
    while len(shapes) < hole_count:
        shapes.append((random.randint(2, 4), random.randint(2, 5)))
    random.shuffle(shapes)

    boxes: list[tuple[int, int, int, int]] = []
    occupied = np.zeros((h, w), dtype=bool)
    for hh, ww in shapes:
        candidates = []
        for r0 in range(h - hh + 1):
            for c0 in range(w - ww + 1):
                r1 = r0 + hh - 1
                c1 = c0 + ww - 1
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(h - 1, r1 + 1)
                cc1 = min(w - 1, c1 + 1)
                if not occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                    candidates.append((r0, c0, r1, c1))
        if not candidates:
            return None
        box = random.choice(candidates)
        boxes.append(box)
        r0, c0, r1, c1 = box
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True

    top = full.copy()
    for r0, c0, r1, c1 in boxes:
        top[r0 : r1 + 1, c0 : c1 + 1] = BLACK

    if not _holes_obey_residue_budget(top, pr, pc):
        return None
    try:
        got_pr, got_pc, got_tile = _infer_periodic_tile(top)
    except AssertionError:
        return None
    if got_pr != pr or got_pc != pc or not np.array_equal(got_tile, _randomless_tile_from_full(full, pr, pc)):
        return None
    if len(_rectangles_from_mask(top == BLACK)) != len(boxes):
        return None
    return top, boxes


def _randomless_tile_from_full(full: np.ndarray, pr: int, pc: int) -> np.ndarray:
    return full[:pr, :pc].copy()


def _choose_defected_patch(
    predicted: np.ndarray,
    rivals: list[np.ndarray],
) -> np.ndarray | None:
    cells = [(r, c) for r in range(predicted.shape[0]) for c in range(predicted.shape[1])]
    random.shuffle(cells)
    for r, c in cells:
        colors = [color for color in COLOR_POOL if color != int(predicted[r, c])]
        random.shuffle(colors)
        for color in colors:
            patch = predicted.copy()
            patch[r, c] = int(color)
            if all(int(np.count_nonzero(patch != rival)) != 1 for rival in rivals):
                return patch
    return None


def _build_patches(
    full: np.ndarray,
    holes: list[tuple[int, int, int, int]],
) -> tuple[list[tuple[int, np.ndarray]], np.ndarray] | None:
    shapes_to_indices: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, box in enumerate(holes):
        r0, c0, r1, c1 = box
        shapes_to_indices[(r1 - r0 + 1, c1 - c0 + 1)].append(idx)

    duplicate_shapes = [shape for shape, idxs in shapes_to_indices.items() if len(idxs) >= 2]
    if not duplicate_shapes:
        return None
    forced_shape = random.choice(duplicate_shapes)
    forced_indices = shapes_to_indices[forced_shape][:2]

    hole_count = len(holes)
    if hole_count == 2:
        patch_count = random.choice([1, 2])
    else:
        patch_count = hole_count if random.random() < 0.45 else hole_count - 1

    patched = [forced_indices[0]]
    if patch_count >= 2 and random.random() < 0.7:
        patched.append(forced_indices[1])
    remaining = [idx for idx in range(hole_count) if idx not in patched]
    random.shuffle(remaining)
    patched.extend(remaining[: patch_count - len(patched)])
    patched = sorted(patched)

    expected = full.copy()
    patches: list[tuple[int, np.ndarray]] = []
    for idx in patched:
        box = holes[idx]
        predicted = _crop(full, box)
        shape = predicted.shape
        rivals = [_crop(full, holes[j]) for j in shapes_to_indices[shape] if j != idx]
        patch = _choose_defected_patch(predicted, rivals)
        if patch is None:
            return None
        r0, c0, r1, c1 = box
        expected[r0 : r1 + 1, c0 : c1 + 1] = patch
        patches.append((idx, patch))

    if patch_count < hole_count and len(patched) == hole_count:
        return None
    if not any(
        sum(
            1
            for j, other in enumerate(holes)
            if j != idx and (other[2] - other[0] + 1, other[3] - other[1] + 1) == patch.shape
        )
        > 0
        for idx, patch in patches
    ):
        return None
    return patches, expected


def _pack_patches(width: int, patches: list[np.ndarray], max_height: int) -> np.ndarray | None:
    order = list(range(len(patches)))
    random.shuffle(order)
    positions: dict[int, tuple[int, int]] = {}
    x = random.randint(0, 1)
    y = random.randint(0, 1)
    row_h = 0
    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        if pw > width:
            return None
        if x + pw > width and row_h > 0:
            y += row_h + random.randint(1, 2)
            x = random.randint(0, 1)
            row_h = 0
        if x + pw > width:
            return None
        positions[idx] = (y, x)
        x += pw + random.randint(1, 2)
        row_h = max(row_h, ph)

    tray_h = y + row_h + random.randint(0, 1)
    if tray_h <= 0 or tray_h > max_height:
        return None

    tray = np.zeros((tray_h, width), dtype=int)
    for idx, patch in enumerate(patches):
        r0, c0 = positions[idx]
        ph, pw = patch.shape
        if r0 + ph > tray_h or np.any(tray[r0 : r0 + ph, c0 : c0 + pw] != BLACK):
            return None
        tray[r0 : r0 + ph, c0 : c0 + pw] = patch
    if len(_rectangles_from_mask(tray != BLACK)) != len(patches):
        return None
    return tray


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> None:
    divider = _find_divider_row(inp)
    top = inp[:divider]
    tray = inp[divider + 1 :]
    pr, pc, tile = _infer_periodic_tile(top)
    periodic = _fill_by_tile(top.shape, pr, pc, tile)

    holes = [{"box": box, "shape": (box[2] - box[0] + 1, box[3] - box[1] + 1)} for box, _ in _rectangles_from_mask(top == BLACK)]
    patches = [_crop(tray, box) for box, _ in _rectangles_from_mask(tray != BLACK)]

    assert holes and patches
    same_shape_hole_exists = False
    matched_count = 0
    for patch in patches:
        matches = []
        for idx, hole in enumerate(holes):
            if hole["shape"] != patch.shape:
                continue
            if sum(1 for other in holes if other["shape"] == hole["shape"]) > 1:
                same_shape_hole_exists = True
            if int(np.count_nonzero(_crop(periodic, hole["box"]) != patch)) == 1:
                matches.append(idx)
        assert len(matches) == 1
        matched_count += 1
    assert same_shape_hole_exists
    assert matched_count == len(patches)
    assert np.array_equal(main(inp), expected)


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(400):
        full, pr, pc, _tile = _generate_periodic_output()
        holes_result = _place_holes(full, pr, pc)
        if holes_result is None:
            continue
        top, holes = holes_result
        patches_result = _build_patches(full, holes)
        if patches_result is None:
            continue
        indexed_patches, expected = patches_result
        tray = _pack_patches(full.shape[1], [patch for _idx, patch in indexed_patches], 30 - full.shape[0] - 1)
        if tray is None:
            continue
        divider = np.full((1, full.shape[1]), DIVIDER, dtype=int)
        inp = np.vstack([top, divider, tray])
        if max(inp.shape) > 30 or max(expected.shape) > 30:
            continue
        try:
            _semantic_checks(inp, expected)
        except AssertionError:
            continue
        return inp, expected
    raise RuntimeError("failed to generate a valid puzzle instance")


def generate_input() -> np.ndarray:
    return _generate_case()[0]
