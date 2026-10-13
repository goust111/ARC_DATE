from __future__ import annotations

import itertools
import random
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, crop_bbox, find_components, paste


BLACK = Color.BLACK
RED = Color.RED
GREY = Color.GREY
ORANGE = Color.ORANGE
CYAN = Color.TEAL

FRAME_SIZE_RANGE = (11, 13)
MARGIN_RANGE = (7, 8)
SIDE_GAP_RANGE = (4, 5)
HOLE_COUNT_OPTIONS = (3, 4)
GENERATION_RETRY_LIMIT = 500
MAX_LOCAL_TRIES = 200
SIDES = ("top", "bottom", "left", "right")
COUNT_PATTERNS = {
    3: [
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 1, 2),
        (1, 2, 2),
        (1, 2, 3),
    ],
    4: [
        (0, 0, 1, 2),
        (0, 1, 1, 2),
        (0, 1, 2, 2),
        (0, 1, 2, 3),
        (0, 1, 1, 3),
        (1, 1, 2, 2),
    ],
}


def _shape_key(coords: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    coords = np.asarray(coords, dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return (
        int(r1 - r0 + 1),
        int(c1 - c0 + 1),
        tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords)),
    )


def _mask_from_key(key: tuple[int, int, tuple[tuple[int, int], ...]]) -> np.ndarray:
    h, w, offsets = key
    mask = np.zeros((h, w), dtype=bool)
    for r, c in offsets:
        mask[int(r), int(c)] = True
    return mask


def _center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, box)
    return (r0 + r1) // 2, (c0 + c1) // 2


def _path_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = map(int, start)
    er, ec = map(int, end)
    cells: list[tuple[int, int]] = []

    step_c = 1 if ec >= sc else -1
    for c in range(sc, ec + step_c, step_c):
        cells.append((sr, c))

    if er != sr:
        step_r = 1 if er >= sr else -1
        for r in range(sr + step_r, er + step_r, step_r):
            cells.append((r, ec))

    return cells


def _coords_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in coords}


def _touches_crop_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
            return True
    return False


def _frame_and_holes(grid: np.ndarray) -> tuple[dict, list[dict]]:
    frames = find_components(grid, bg=BLACK, connectivity=4, monochromatic=True, colors=[RED])
    assert len(frames) == 1, "expected exactly one red frame"
    frame = frames[0]
    r0, c0, r1, c1 = map(int, frame["bbox"])
    crop = crop_bbox(grid, frame["bbox"])

    holes: list[dict] = []
    for coords in components_mask(crop == BLACK, connectivity=4):
        if _touches_crop_border(coords, crop.shape):
            continue
        global_coords = coords + np.array([r0, c0], dtype=int)
        holes.append(
            {
                "coords": global_coords,
                "bbox": bbox_coords(global_coords),
                "key": _shape_key(global_coords),
            }
        )

    assert 3 <= len(holes) <= 4, "expected 3-4 enclosed holes"
    return frame, holes


def _pieces(grid: np.ndarray) -> list[dict]:
    pieces = find_components(grid, bg=BLACK, connectivity=4, monochromatic=True, colors=[GREY])
    assert 3 <= len(pieces) <= 4, "expected 3-4 grey pieces"
    out = []
    for piece in pieces:
        out.append(
            {
                "coords": piece["coords"],
                "bbox": piece["bbox"],
                "key": _shape_key(piece["coords"]),
            }
        )
    return out


def _analyze_scene(grid: np.ndarray) -> list[dict]:
    frame, holes = _frame_and_holes(grid)
    pieces = _pieces(grid)
    assert len(pieces) == len(holes), "piece/hole count mismatch"

    hole_by_key: dict[tuple[int, int, tuple[tuple[int, int], ...]], dict] = {}
    for hole in holes:
        assert hole["key"] not in hole_by_key, "hole shapes must be unique"
        hole_by_key[hole["key"]] = hole

    placements: list[dict] = []
    used_holes: set[tuple[int, int, tuple[tuple[int, int], ...]]] = set()
    for piece in pieces:
        key = piece["key"]
        assert key in hole_by_key, "every piece must match one hole"
        assert key not in used_holes, "piece matching must be bijective"
        used_holes.add(key)

        hole = hole_by_key[key]
        start = _center(piece["bbox"])
        end = _center(hole["bbox"])
        path = _path_cells(start, end)
        obstacle_count = sum(int(grid[r, c]) == ORANGE for r, c in path)

        pr0, pc0, _pr1, _pc1 = map(int, piece["bbox"])
        hr0, hc0, _hr1, _hc1 = map(int, hole["bbox"])
        translated = piece["coords"] + np.array([hr0 - pr0, hc0 - pc0], dtype=int)
        assert _coords_set(translated) == _coords_set(hole["coords"]), "piece must translate exactly into its hole"

        placements.append(
            {
                "piece": piece,
                "hole": hole,
                "path": path,
                "obstacle_count": int(obstacle_count),
                "color": CYAN if obstacle_count % 2 else GREY,
                "translated": translated,
                "frame_bbox": frame["bbox"],
            }
        )

    return placements


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit each grey piece into its matching hole and recolor it by the parity of orange obstacles on its L-shaped travel path.

    Input structure:
    - The grid has a black background, one connected red frame-like object, and 3-4 enclosed black hole regions inside that red object.
    - Outside the red frame are 3-4 disconnected grey puzzle pieces whose normalized shapes uniquely match the holes.
    - Orange single-pixel obstacles lie outside the frame and away from the pieces.

    Rule:
    - Normalize every grey piece and every enclosed hole by cropping to its bounding box, then match them by exact shape equality.
    - For each matched pair, build the L path from the piece bbox center to the hole bbox center: move horizontally along the piece center row until the hole center column, then move vertically along that column to the hole center row.
    - Count how many orange pixels from the input lie on that path; odd counts recolor the moved piece cyan, even counts keep it grey.
    - Remove each piece from its original location and translate it into its matched hole. Keep the red frame and the orange obstacles unchanged.

    Key insight:
    - Shape matching chooses the destination hole, but the final color is determined separately by obstacle parity on a specific center-to-center L path, so neither shape alone nor position alone is sufficient.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    placements = _analyze_scene(g)

    for info in placements:
        coords = info["piece"]["coords"]
        out[coords[:, 0], coords[:, 1]] = BLACK

    for info in placements:
        coords = info["translated"]
        out[coords[:, 0], coords[:, 1]] = int(info["color"])

    return out


def _box(top: int, left: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = map(int, shape)
    return int(top), int(left), int(top + h - 1), int(left + w - 1)


def _boxes_too_close(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
    *,
    gap: int,
) -> bool:
    return not (
        box_a[2] + gap < box_b[0]
        or box_b[2] + gap < box_a[0]
        or box_a[3] + gap < box_b[1]
        or box_b[3] + gap < box_a[1]
    )


def _chebyshev_close(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1]))) <= 1


def _is_connected(mask: np.ndarray) -> bool:
    comps = components_mask(mask.astype(bool), connectivity=4)
    return len(comps) == 1


@lru_cache(None)
def _shape_catalog() -> tuple[tuple[int, np.ndarray, tuple[int, int, tuple[tuple[int, int], ...]]], ...]:
    catalog: list[tuple[int, np.ndarray, tuple[int, int, tuple[tuple[int, int], ...]]]] = []
    seen = set()
    for area in range(3, 7):
        for cells in itertools.combinations(range(9), area):
            raw = np.zeros((3, 3), dtype=bool)
            raw.flat[list(cells)] = True
            if not _is_connected(raw):
                continue
            coords = np.argwhere(raw)
            key = _shape_key(coords)
            h, w, _offsets = key
            if h % 2 == 0 or w % 2 == 0:
                continue
            if key in seen:
                continue
            seen.add(key)
            catalog.append((area, _mask_from_key(key), key))
    return tuple(catalog)


def _choose_shapes(n_shapes: int) -> list[tuple[int, np.ndarray, tuple[int, int, tuple[tuple[int, int], ...]]]] | None:
    catalog = list(_shape_catalog())
    by_area: dict[int, list[tuple[int, np.ndarray, tuple[int, int, tuple[tuple[int, int], ...]]]]] = {}
    for entry in catalog:
        by_area.setdefault(entry[0], []).append(entry)

    repeatable_areas = [area for area, entries in by_area.items() if len(entries) >= 2]
    if not repeatable_areas:
        return None

    repeat_area = random.choice(repeatable_areas)
    chosen = random.sample(by_area[repeat_area], 2)
    used_keys = {entry[2] for entry in chosen}

    other_pool = [entry for entry in catalog if entry[0] != repeat_area and entry[2] not in used_keys]
    if not other_pool:
        return None
    chosen.append(random.choice(other_pool))
    used_keys.add(chosen[-1][2])

    while len(chosen) < n_shapes:
        pool = [entry for entry in catalog if entry[2] not in used_keys]
        if not pool:
            return None
        pick = random.choice(pool)
        chosen.append(pick)
        used_keys.add(pick[2])

    random.shuffle(chosen)
    return chosen


def _build_frame_and_holes(
    frame_top: int,
    frame_left: int,
    frame_h: int,
    frame_w: int,
    shapes: list[tuple[int, np.ndarray, tuple[int, int, tuple[tuple[int, int], ...]]]],
) -> list[dict] | None:
    interior_h = frame_h - 2
    interior_w = frame_w - 2

    for _ in range(MAX_LOCAL_TRIES):
        order = list(range(len(shapes)))
        random.shuffle(order)
        placed: dict[int, dict] = {}
        used_boxes: list[tuple[int, int, int, int]] = []

        ok = True
        for idx in order:
            _area, mask, key = shapes[idx]
            h, w = mask.shape
            candidates = [
                (r0, c0)
                for r0 in range(1, interior_h - h + 2)
                for c0 in range(1, interior_w - w + 2)
            ]
            random.shuffle(candidates)
            for local_r0, local_c0 in candidates:
                local_box = _box(local_r0, local_c0, mask.shape)
                if any(_boxes_too_close(local_box, other, gap=1) for other in used_boxes):
                    continue

                global_r0 = frame_top + local_r0
                global_c0 = frame_left + local_c0
                coords = np.argwhere(mask) + np.array([global_r0, global_c0], dtype=int)
                placed[idx] = {
                    "area": _area,
                    "mask": mask,
                    "key": key,
                    "bbox": _box(global_r0, global_c0, mask.shape),
                    "coords": coords,
                }
                used_boxes.append(local_box)
                break
            else:
                ok = False
                break

        if ok and len(placed) == len(shapes):
            return [placed[i] for i in range(len(shapes))]

    return None


def _can_place_piece(blocked: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    h, w = mask.shape
    if top < 0 or left < 0 or top + h > blocked.shape[0] or left + w > blocked.shape[1]:
        return False

    coords = np.argwhere(mask) + np.array([top, left], dtype=int)
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if blocked[rr, cc]:
            return False
        r0 = max(0, rr - 1)
        r1 = min(blocked.shape[0], rr + 2)
        c0 = max(0, cc - 1)
        c1 = min(blocked.shape[1], cc + 2)
        if blocked[r0:r1, c0:c1].any():
            return False
    return True


def _side_candidates(
    side: str,
    frame_box: tuple[int, int, int, int],
    grid_shape: tuple[int, int],
    mask_shape: tuple[int, int],
    gap: int,
) -> list[tuple[int, int]]:
    h, w = map(int, mask_shape)
    r0, c0, r1, c1 = map(int, frame_box)
    H, W = map(int, grid_shape)
    candidates: list[tuple[int, int]] = []

    if side == "top":
        row = r0 - gap - h
        for col in range(0, W - w + 1):
            candidates.append((row, col))
    elif side == "bottom":
        row = r1 + gap + 1
        for col in range(0, W - w + 1):
            candidates.append((row, col))
    elif side == "left":
        col = c0 - gap - w
        for row in range(0, H - h + 1):
            candidates.append((row, col))
    elif side == "right":
        col = c1 + gap + 1
        for row in range(0, H - h + 1):
            candidates.append((row, col))
    else:
        raise AssertionError("unknown side")

    random.shuffle(candidates)
    return candidates


def _place_pieces(
    grid_shape: tuple[int, int],
    frame_box: tuple[int, int, int, int],
    holes: list[dict],
    chosen_sides: list[str],
) -> list[dict] | None:
    for _ in range(MAX_LOCAL_TRIES):
        hole_order = list(range(len(holes)))
        random.shuffle(hole_order)
        side_by_hole = {hole_idx: side for hole_idx, side in zip(hole_order, chosen_sides)}
        blocked = np.zeros(grid_shape, dtype=bool)
        fr0, fc0, fr1, fc1 = map(int, frame_box)
        blocked[fr0 : fr1 + 1, fc0 : fc1 + 1] = True

        pieces: dict[int, dict] = {}
        ok = True
        for hole_idx, hole in enumerate(holes):
            side = side_by_hole[hole_idx]
            gap = random.randint(*SIDE_GAP_RANGE)
            candidates = _side_candidates(side, frame_box, grid_shape, hole["mask"].shape, gap)
            for top, left in candidates:
                if not _can_place_piece(blocked, hole["mask"], top, left):
                    continue
                coords = np.argwhere(hole["mask"]) + np.array([top, left], dtype=int)
                pieces[hole_idx] = {
                    "mask": hole["mask"],
                    "bbox": _box(top, left, hole["mask"].shape),
                    "coords": coords,
                    "key": hole["key"],
                    "side": side,
                    "hole_index": hole_idx,
                }
                blocked[coords[:, 0], coords[:, 1]] = True
                break
            else:
                ok = False
                break

        if ok and len(pieces) == len(holes):
            return [pieces[i] for i in range(len(holes))]

    return None


def _coord_is_orange_safe(coord: tuple[int, int], blocked: np.ndarray, orange_coords: list[tuple[int, int]]) -> bool:
    r, c = map(int, coord)
    if blocked[r, c]:
        return False
    r0 = max(0, r - 1)
    r1 = min(blocked.shape[0], r + 2)
    c0 = max(0, c - 1)
    c1 = min(blocked.shape[1], c + 2)
    if blocked[r0:r1, c0:c1].any():
        return False
    return all(not _chebyshev_close(coord, other) for other in orange_coords)


def _piece_path_candidates(
    grid_shape: tuple[int, int],
    frame_box: tuple[int, int, int, int],
    holes: list[dict],
    pieces: list[dict],
) -> tuple[list[list[tuple[int, int]]], list[list[tuple[int, int]]]]:
    del frame_box  # Structural validation is already handled by blocked-mask checks.

    all_paths = []
    path_sets = []
    for piece in pieces:
        hole = holes[piece["hole_index"]]
        path = _path_cells(_center(piece["bbox"]), _center(hole["bbox"]))
        all_paths.append(path)
        path_sets.append(set(path))

    blocked = np.zeros(grid_shape, dtype=bool)
    for hole in holes:
        blocked[hole["coords"][:, 0], hole["coords"][:, 1]] = True
    for piece in pieces:
        blocked[piece["coords"][:, 0], piece["coords"][:, 1]] = True
    fr0, fc0, fr1, fc1 = map(int, pieces[0]["frame_bbox"])
    blocked[fr0 : fr1 + 1, fc0 : fc1 + 1] = True
    for hole in holes:
        blocked[hole["coords"][:, 0], hole["coords"][:, 1]] = False

    candidates: list[list[tuple[int, int]]] = []
    for idx, path in enumerate(all_paths):
        unique_cells = []
        for coord in path:
            if sum(coord in path_set for path_set in path_sets) != 1:
                continue
            if _coord_is_orange_safe(coord, blocked, []):
                unique_cells.append(coord)
        candidates.append(unique_cells)
    return all_paths, candidates


def _try_assign_path_oranges(
    pieces: list[dict],
    candidate_lists: list[list[tuple[int, int]]],
    count_pattern: tuple[int, ...],
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]]] | None:
    order = sorted(range(len(pieces)), key=lambda i: (-count_pattern[i], len(candidate_lists[i])))
    picked_by_piece: list[list[tuple[int, int]]] = [[] for _ in pieces]
    orange_coords: list[tuple[int, int]] = []

    for idx in order:
        need = int(count_pattern[idx])
        if need == 0:
            continue
        candidates = list(candidate_lists[idx])
        success = False
        for _ in range(40):
            random.shuffle(candidates)
            local = []
            for coord in candidates:
                if any(_chebyshev_close(coord, other) for other in orange_coords):
                    continue
                if any(_chebyshev_close(coord, other) for other in local):
                    continue
                local.append(coord)
                if len(local) == need:
                    break
            if len(local) == need:
                picked_by_piece[idx] = local
                orange_coords.extend(local)
                success = True
                break
        if not success:
            return None

    return picked_by_piece, orange_coords


def _choose_oranges(
    grid_shape: tuple[int, int],
    frame_box: tuple[int, int, int, int],
    holes: list[dict],
    pieces: list[dict],
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]]] | None:
    for piece in pieces:
        piece["frame_bbox"] = frame_box

    paths, candidate_lists = _piece_path_candidates(grid_shape, frame_box, holes, pieces)
    del paths

    count_patterns = []
    for pattern in COUNT_PATTERNS[len(pieces)]:
        count_patterns.extend(set(itertools.permutations(pattern)))
    random.shuffle(count_patterns)

    blocked = np.zeros(grid_shape, dtype=bool)
    fr0, fc0, fr1, fc1 = map(int, frame_box)
    blocked[fr0 : fr1 + 1, fc0 : fc1 + 1] = True
    for piece in pieces:
        blocked[piece["coords"][:, 0], piece["coords"][:, 1]] = True

    union_path = set()
    for piece in pieces:
        hole = holes[piece["hole_index"]]
        union_path.update(_path_cells(_center(piece["bbox"]), _center(hole["bbox"])))

    for pattern in count_patterns:
        assigned = _try_assign_path_oranges(pieces, candidate_lists, pattern)
        if assigned is None:
            continue
        picked_by_piece, orange_coords = assigned
        if not orange_coords:
            continue

        extra_candidates = []
        for r in range(grid_shape[0]):
            for c in range(grid_shape[1]):
                coord = (r, c)
                if coord in union_path:
                    continue
                if _coord_is_orange_safe(coord, blocked, orange_coords):
                    extra_candidates.append(coord)

        random.shuffle(extra_candidates)
        max_extra = min(2, 8 - len(orange_coords))
        extra_count = 0
        if max_extra > 0 and random.random() < 0.6:
            extra_count = random.randint(0, max_extra)

        extras = []
        for coord in extra_candidates:
            if any(_chebyshev_close(coord, other) for other in orange_coords):
                continue
            if any(_chebyshev_close(coord, other) for other in extras):
                continue
            extras.append(coord)
            if len(extras) == extra_count:
                break
        if len(extras) != extra_count:
            continue

        all_oranges = orange_coords + extras
        if not (3 <= len(all_oranges) <= 8):
            continue
        if not any(count % 2 == 0 for count in pattern):
            continue
        if not any(count % 2 == 1 for count in pattern):
            continue
        return picked_by_piece, all_oranges

    return None


def _render_case(
    grid_shape: tuple[int, int],
    frame_box: tuple[int, int, int, int],
    holes: list[dict],
    pieces: list[dict],
    oranges: list[tuple[int, int]],
) -> np.ndarray:
    H, W = map(int, grid_shape)
    grid = np.full((H, W), BLACK, dtype=int)
    fr0, fc0, fr1, fc1 = map(int, frame_box)
    grid[fr0 : fr1 + 1, fc0 : fc1 + 1] = RED

    for hole in holes:
        coords = hole["coords"]
        grid[coords[:, 0], coords[:, 1]] = BLACK

    for piece in pieces:
        patch = np.full(piece["mask"].shape, BLACK, dtype=int)
        patch[piece["mask"]] = GREY
        pr0, pc0, _pr1, _pc1 = map(int, piece["bbox"])
        paste(grid, patch, pr0, pc0, transparent=BLACK)

    for r, c in oranges:
        grid[int(r), int(c)] = ORANGE

    return grid


def _expected_output(grid: np.ndarray, pieces: list[dict], holes: list[dict]) -> np.ndarray:
    out = grid.copy()
    for piece in pieces:
        out[piece["coords"][:, 0], piece["coords"][:, 1]] = BLACK

    for piece in pieces:
        hole = holes[piece["hole_index"]]
        path = _path_cells(_center(piece["bbox"]), _center(hole["bbox"]))
        count = sum(int(grid[r, c]) == ORANGE for r, c in path)
        color = CYAN if count % 2 else GREY
        out[hole["coords"][:, 0], hole["coords"][:, 1]] = color

    return out


def _has_nonnearest_match(pieces: list[dict], holes: list[dict]) -> bool:
    hole_centers = [_center(hole["bbox"]) for hole in holes]
    for piece in pieces:
        piece_center = _center(piece["bbox"])
        distances = [
            abs(piece_center[0] - center[0]) + abs(piece_center[1] - center[1])
            for center in hole_centers
        ]
        matched = int(piece["hole_index"])
        if distances[matched] > min(distances):
            return True
    return False


def _validate_case(
    grid: np.ndarray,
    expected: np.ndarray,
    holes: list[dict],
    pieces: list[dict],
    oranges: list[tuple[int, int]],
) -> bool:
    if grid.ndim != 2 or expected.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if not (3 <= len(holes) == len(pieces) <= 4):
        return False
    if not (3 <= len(oranges) <= 8):
        return False

    areas = [int(piece["mask"].sum()) for piece in pieces]
    if len(set(areas)) == len(areas):
        return False
    if len(set(areas)) < 2:
        return False
    if not _has_nonnearest_match(pieces, holes):
        return False

    try:
        analyzed = _analyze_scene(grid)
    except Exception:
        return False

    parities = [info["obstacle_count"] % 2 for info in analyzed]
    if not (0 in parities and 1 in parities):
        return False
    if any(info["obstacle_count"] > 3 for info in analyzed):
        return False

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    return True


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    n_shapes = random.choice(HOLE_COUNT_OPTIONS)
    shapes = _choose_shapes(n_shapes)
    if shapes is None:
        return None

    frame_h = random.randint(*FRAME_SIZE_RANGE)
    frame_w = random.randint(*FRAME_SIZE_RANGE)
    top_margin = random.randint(*MARGIN_RANGE)
    bottom_margin = random.randint(*MARGIN_RANGE)
    left_margin = random.randint(*MARGIN_RANGE)
    right_margin = random.randint(*MARGIN_RANGE)
    H = top_margin + frame_h + bottom_margin
    W = left_margin + frame_w + right_margin
    if max(H, W) > 30:
        return None

    frame_box = (top_margin, left_margin, top_margin + frame_h - 1, left_margin + frame_w - 1)
    holes = _build_frame_and_holes(top_margin, left_margin, frame_h, frame_w, shapes)
    if holes is None:
        return None

    chosen_sides = random.sample(SIDES, n_shapes)
    pieces = _place_pieces((H, W), frame_box, holes, chosen_sides)
    if pieces is None:
        return None

    orange_assignment = _choose_oranges((H, W), frame_box, holes, pieces)
    if orange_assignment is None:
        return None
    _picked_by_piece, oranges = orange_assignment

    grid = _render_case((H, W), frame_box, holes, pieces, oranges)
    expected = _expected_output(grid, pieces, holes)
    if not _validate_case(grid, expected, holes, pieces, oranges):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRY_LIMIT):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid puzzle input")
