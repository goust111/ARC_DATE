from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4, 6)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
MAX_GEN_ATTEMPTS = 400


def _normalize_cells(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _cells_to_mask(cells: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = 1 + max(r for r, _c in cells)
    w = 1 + max(c for _r, c in cells)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _rotate_cells(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    arr = np.array(cells, dtype=int)
    arr = np.column_stack((arr[:, 1], -arr[:, 0]))
    return _normalize_cells({tuple(x) for x in arr.tolist()})


def _flip_cells(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    arr = np.array(cells, dtype=int)
    arr[:, 1] *= -1
    return _normalize_cells({tuple(x) for x in arr.tolist()})


def _random_orientation(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    out = cells
    for _ in range(random.randrange(4)):
        out = _rotate_cells(out)
    if random.random() < 0.5:
        out = _flip_cells(out)
    return out


def _random_polyomino(area: int) -> tuple[tuple[int, int], ...]:
    cells = {(0, 0)}
    frontier = {(0, 1), (1, 0), (0, -1), (-1, 0)}
    while len(cells) < area:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        r, c = cell
        for dr, dc in DIR4:
            nb = (r + dr, c + dc)
            if nb not in cells:
                frontier.add(nb)
    return _random_orientation(_normalize_cells(cells))


def _count_components_from_coords(coords: set[tuple[int, int]]) -> int:
    unseen = set(coords)
    count = 0
    while unseen:
        count += 1
        q = deque([unseen.pop()])
        while q:
            r, c = q.popleft()
            for dr, dc in DIR4:
                nb = (r + dr, c + dc)
                if nb in unseen:
                    unseen.remove(nb)
                    q.append(nb)
    return count


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    objects: list[dict] = []
    for r in range(H):
        for c in range(W):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            b = _bbox(arr)
            r0, c0, r1, c1 = b
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[arr[:, 0] - r0, arr[:, 1] - c0] = True
            objects.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": b,
                    "mask": mask,
                    "area": int(arr.shape[0]),
                    "height": int(mask.shape[0]),
                    "width": int(mask.shape[1]),
                }
            )
    return objects


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(mask.astype(bool), np.ones((factor, factor), dtype=bool))


def _window_match(big_mask: np.ndarray, small_mask: np.ndarray, factor: int) -> bool:
    scaled = _scale_mask(small_mask, factor)
    bh, bw = big_mask.shape
    sh, sw = scaled.shape
    if bh > sh or bw > sw:
        return False
    for r0 in range(sh - bh + 1):
        for c0 in range(sw - bw + 1):
            if np.array_equal(scaled[r0 : r0 + bh, c0 : c0 + bw], big_mask):
                return True
    return False


def _pair_match_records(obj_a: dict, obj_b: dict) -> list[dict]:
    records: list[dict] = []
    for small_idx, big_idx, small_obj, big_obj in (
        (0, 1, obj_a, obj_b),
        (1, 0, obj_b, obj_a),
    ):
        small_area = int(small_obj.get("area", int(np.sum(small_obj["mask"]))))
        big_area = int(big_obj.get("area", int(np.sum(big_obj["mask"]))))
        if small_area >= big_area:
            continue
        for factor in (2, 3):
            if _window_match(big_obj["mask"], small_obj["mask"], factor):
                records.append(
                    {
                        "small_local": small_idx,
                        "big_local": big_idx,
                        "factor": factor,
                        "small_height": int(small_obj["height"]),
                    }
                )
    return records


def _solve_metadata(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.all((g >= 0) & (g <= 9))
    objects = _find_objects(g)
    assert len(objects) == 3, "expected exactly three 4-connected foreground objects"

    pair_hits = []
    for i in range(3):
        for j in range(i + 1, 3):
            matches = _pair_match_records(objects[i], objects[j])
            for match in matches:
                pair_hits.append((i, j, match))

    assert len(pair_hits) == 1, f"expected exactly one scalable pair, found {len(pair_hits)}"
    i, j, match = pair_hits[0]
    small_idx = i if match["small_local"] == 0 else j
    big_idx = j if match["big_local"] == 1 else i
    target_idx = ({0, 1, 2} - {small_idx, big_idx}).pop()

    factor = int(match["factor"])
    small_height = int(objects[small_idx]["height"])
    shift = factor * small_height

    out = np.zeros_like(g)
    target = objects[target_idx]
    moved = target["coords"].copy()
    moved[:, 0] += shift
    keep = moved[:, 0] < g.shape[0]
    moved = moved[keep]
    out[moved[:, 0], moved[:, 1]] = int(target["color"])

    return {
        "objects": objects,
        "small_idx": small_idx,
        "big_idx": big_idx,
        "target_idx": target_idx,
        "factor": factor,
        "small_height": small_height,
        "shift": shift,
        "output": out,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the cross-scale matching pair, then slide the remaining object downward by the extracted scale-driven distance.

    Input structure:
    - The grid has black background and exactly three separated 4-connected monochromatic foreground objects.
    - Two objects share the same shape at different integer scales; the larger visible object may be clipped by the grid boundary.
    - The third object has a different shape and acts as the only object preserved into the output.

    Rule:
    - Test every object pair by scaling one candidate mask by 2 or 3 and checking whether the other mask matches some visible window of that scaled shape.
    - From the unique matching pair, take the scale factor K and the smaller matching object's bounding-box height H.
    - Clear both matching objects, then move the third object straight downward by K x H rows, keeping columns fixed and clipping pixels that pass below the bottom edge.

    Key insight:
    - The pair that disappears is still the information source: shape matching yields the scale factor, and that factor times the small shape's height determines how far the unrelated third object must slide.
    """
    return _solve_metadata(grid)["output"]


def _coords_from_mask(mask: np.ndarray, top: int, left: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    H, W = shape
    coords: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        rr = top + int(r)
        cc = left + int(c)
        if 0 <= rr < H and 0 <= cc < W:
            coords.append((rr, cc))
    return coords


def _candidate_visible_placements(
    mask: np.ndarray,
    grid_shape: tuple[int, int],
    *,
    clipped: bool | None = None,
    min_visible: int = 1,
    max_visible: int | None = None,
) -> list[dict]:
    H, W = grid_shape
    h, w = mask.shape
    candidates = []
    for top in range(-h + 1, H):
        for left in range(-w + 1, W):
            coords = _coords_from_mask(mask, top, left, grid_shape)
            if len(coords) < min_visible:
                continue
            if max_visible is not None and len(coords) > max_visible:
                continue
            is_clipped = top < 0 or left < 0 or top + h > H or left + w > W
            if clipped is not None and is_clipped != clipped:
                continue
            if _count_components_from_coords(set(coords)) != 1:
                continue
            candidates.append({"top": top, "left": left, "coords": coords, "clipped": is_clipped})
    return candidates


def _candidate_inbounds_placements(mask: np.ndarray, grid_shape: tuple[int, int]) -> list[dict]:
    H, W = grid_shape
    h, w = mask.shape
    candidates = []
    for top in range(0, H - h + 1):
        for left in range(0, W - w + 1):
            coords = _coords_from_mask(mask, top, left, grid_shape)
            candidates.append({"top": top, "left": left, "coords": coords, "clipped": False})
    return candidates


def _well_separated(existing: set[tuple[int, int]], candidate: list[tuple[int, int]]) -> bool:
    for r, c in candidate:
        for dr, dc in DIR8 + [(0, 0)]:
            if (r + dr, c + dc) in existing:
                return False
    return True


def _paint(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[r, c] = int(color)


def _target_output_coords(candidate: dict, shift: int, grid_shape: tuple[int, int]) -> list[tuple[int, int]]:
    H, _W = grid_shape
    moved = [(r + shift, c) for r, c in candidate["coords"] if r + shift < H]
    return moved


def _make_grid(
    grid_shape: tuple[int, int],
    small_place: dict,
    large_place: dict,
    target_place: dict,
    colors: tuple[int, int, int],
) -> np.ndarray:
    grid = np.zeros(grid_shape, dtype=int)
    _paint(grid, small_place["coords"], colors[0])
    _paint(grid, large_place["coords"], colors[1])
    _paint(grid, target_place["coords"], colors[2])
    return grid


def _shape_constraints_ok(cells: tuple[tuple[int, int], ...], factor: int) -> bool:
    mask = _cells_to_mask(cells)
    h, w = mask.shape
    area = int(mask.sum())
    shift = factor * h
    return 2 <= h <= 4 and 2 <= w <= 5 and h != w and area not in {h, w, shift}


def _sample_small_shape(factor: int, allow_unclipped_big: bool) -> np.ndarray:
    while True:
        area = random.randint(4, 5 if allow_unclipped_big else (7 if factor == 2 else 6))
        cells = _random_polyomino(area)
        if not _shape_constraints_ok(cells, factor):
            continue
        mask = _cells_to_mask(cells)
        scaled = _scale_mask(mask, factor)
        matches = _pair_match_records(
            {"mask": mask, "height": int(mask.shape[0]), "area": int(mask.sum())},
            {"mask": scaled, "height": int(scaled.shape[0]), "area": int(scaled.sum())},
        )
        if len(matches) != 1 or int(matches[0]["factor"]) != factor:
            continue
        return mask


def _sample_target_shape(small_mask: np.ndarray, factor: int, shift: int) -> np.ndarray:
    small_obj = {"mask": small_mask, "height": int(small_mask.shape[0])}
    while True:
        area = random.randint(4, 9)
        cells = _random_polyomino(area)
        mask = _cells_to_mask(cells)
        h, w = mask.shape
        if not (2 <= h <= 5 and 2 <= w <= 6):
            continue
        if area in {shift, h, w, factor, small_mask.shape[0], small_mask.shape[1]}:
            continue
        if _pair_match_records({"mask": mask, "height": h}, small_obj):
            continue
        if _pair_match_records(small_obj, {"mask": mask, "height": h}):
            continue
        return mask


def _shortcut_outputs(grid_shape: tuple[int, int], target_obj: dict, shifts: set[int]) -> set[tuple[bytes, tuple[int, int]]]:
    H, W = grid_shape
    outputs = set()
    for shift in shifts:
        out = np.zeros((H, W), dtype=int)
        for r, c in target_obj["coords"]:
            rr = int(r) + int(shift)
            if rr < H:
                out[rr, int(c)] = int(target_obj["color"])
        outputs.add((out.tobytes(), out.shape))
    return outputs


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        meta = _solve_metadata(grid)
    except AssertionError:
        return False

    objects = meta["objects"]
    areas = [obj["area"] for obj in objects]
    if not all(4 <= area <= 20 for area in areas):
        return False
    if len({obj["color"] for obj in objects}) != 3:
        return False

    small = objects[meta["small_idx"]]
    big = objects[meta["big_idx"]]
    target = objects[meta["target_idx"]]
    shift = meta["shift"]
    if shift in {meta["factor"], meta["small_height"], small["area"], target["area"], small["width"]}:
        return False
    if np.array_equal(grid, meta["output"]):
        return False

    wrong_shifts = {
        meta["factor"],
        meta["small_height"],
        small["area"],
        target["area"],
        small["width"],
        big["height"],
        big["area"],
    }
    wrong_shifts.discard(shift)
    actual_key = (meta["output"].tobytes(), meta["output"].shape)
    shortcut_outputs = _shortcut_outputs(grid.shape, target, {s for s in wrong_shifts if s >= 0})
    distinct_wrong = [key for key in shortcut_outputs if key != actual_key]
    if len(distinct_wrong) < 2:
        return False

    return True


def _separated_pair(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> bool:
    return _well_separated(set(a), b)


def _attempt_generate() -> np.ndarray | None:
    factor = 2 if random.random() < 0.8 else 3
    min_side = 17 if factor == 3 else 15
    H = random.randint(min_side, 19)
    W = random.randint(min_side, 19)
    want_big_clipped = True if factor == 3 else random.random() < 0.35
    want_target_clipped = random.random() < 0.25

    small_mask = _sample_small_shape(factor, allow_unclipped_big=not want_big_clipped)
    large_mask = _scale_mask(small_mask, factor)
    shift = factor * int(small_mask.shape[0])
    target_mask = _sample_target_shape(small_mask, factor, shift)

    small_h, small_w = small_mask.shape
    large_h, large_w = large_mask.shape
    target_h, target_w = target_mask.shape

    left_zone_max = max(0, W // 3 - target_w)
    mid_zone_min = max(2, W // 3)
    mid_zone_max = max(mid_zone_min, W // 2 - small_w)
    right_zone_min = max(W // 2 + 1, mid_zone_min + small_w + 2)
    if right_zone_min > W - 1:
        return None

    small_places = [
        {"top": top, "left": left, "coords": _coords_from_mask(small_mask, top, left, (H, W)), "clipped": False}
        for top in range(1, max(2, H // 3 + 1))
        for left in range(mid_zone_min, mid_zone_max + 1)
        if top + small_h <= H and left + small_w <= W
    ]

    target_places = []
    top_hi = H - shift - target_h
    if want_target_clipped:
        top_lo = max(0, H - shift - target_h + 1)
        top_hi_clip = min(H - target_h, H - shift - 1)
        for top in range(top_lo, top_hi_clip + 1):
            for left in range(0, left_zone_max + 1):
                coords = _coords_from_mask(target_mask, top, left, (H, W))
                moved = _target_output_coords({"coords": coords}, shift, (H, W))
                if len(moved) >= 2 and len(moved) < len(coords):
                    target_places.append({"top": top, "left": left, "coords": coords, "clipped": False})
    else:
        if top_hi < 0:
            return None
        for top in range(0, top_hi + 1):
            for left in range(0, left_zone_max + 1):
                coords = _coords_from_mask(target_mask, top, left, (H, W))
                if len(_target_output_coords({"coords": coords}, shift, (H, W))) == len(coords):
                    target_places.append({"top": top, "left": left, "coords": coords, "clipped": False})

    large_places = []
    if want_big_clipped:
        if large_w > W - right_zone_min:
            return None
        for left in range(right_zone_min, W - large_w + 1):
            for top in range(H - large_h + 1, H):
                coords = _coords_from_mask(large_mask, top, left, (H, W))
                if 4 <= len(coords) <= 20 and _count_components_from_coords(set(coords)) == 1:
                    large_places.append({"top": top, "left": left, "coords": coords, "clipped": True})
    else:
        if large_w > W - right_zone_min or large_h > H - (H // 2 + 1):
            return None
        for top in range(H // 2, H - large_h + 1):
            for left in range(right_zone_min, W - large_w + 1):
                coords = _coords_from_mask(large_mask, top, left, (H, W))
                if 4 <= len(coords) <= 20:
                    large_places.append({"top": top, "left": left, "coords": coords, "clipped": False})

    if not small_places or not target_places or not large_places:
        return None

    colors = tuple(random.sample(PALETTE, 3))
    random.shuffle(target_places)
    random.shuffle(small_places)
    random.shuffle(large_places)
    target_places = target_places[:20]
    small_places = small_places[:20]
    large_places = large_places[:20]

    for target_place in target_places:
        for small_place in small_places:
            if not _separated_pair(target_place["coords"], small_place["coords"]):
                continue
            for large_place in large_places:
                if not _separated_pair(target_place["coords"], large_place["coords"]):
                    continue
                if not _separated_pair(small_place["coords"], large_place["coords"]):
                    continue
                grid = _make_grid((H, W), small_place, large_place, target_place, colors)
                if _semantic_checks(grid):
                    return grid
    return None


def _fallback_generate() -> np.ndarray | None:
    factor = 2 if random.random() < 0.8 else 3
    H = random.randint(17 if factor == 3 else 15, 19)
    W = random.randint(17 if factor == 3 else 15, 19)
    want_big_clipped = True if factor == 3 else random.random() < 0.35
    want_target_clipped = random.random() < 0.2

    small_mask = _sample_small_shape(factor, allow_unclipped_big=not want_big_clipped)
    shift = factor * int(small_mask.shape[0])
    target_mask = _sample_target_shape(small_mask, factor, shift)
    large_mask = _scale_mask(small_mask, factor)
    colors = tuple(random.sample(PALETTE, 3))

    small_places = []
    for top in range(1, min(4, H - small_mask.shape[0] + 1)):
        for left in range(W // 3, max(W // 3 + 1, W // 2 - small_mask.shape[1] + 1)):
            coords = _coords_from_mask(small_mask, top, left, (H, W))
            small_places.append({"coords": coords})

    target_places = []
    target_left_max = max(0, W // 3 - target_mask.shape[1])
    if want_target_clipped:
        top_lo = max(0, H - shift - target_mask.shape[0] + 1)
        top_hi = min(H - target_mask.shape[0], H - shift - 1)
        for top in range(top_lo, top_hi + 1):
            for left in range(0, target_left_max + 1):
                coords = _coords_from_mask(target_mask, top, left, (H, W))
                moved = _target_output_coords({"coords": coords}, shift, (H, W))
                if len(moved) >= 2 and len(moved) < len(coords):
                    target_places.append({"coords": coords})
    else:
        top_hi = H - shift - target_mask.shape[0]
        for top in range(0, max(-1, top_hi) + 1):
            for left in range(0, target_left_max + 1):
                coords = _coords_from_mask(target_mask, top, left, (H, W))
                moved = _target_output_coords({"coords": coords}, shift, (H, W))
                if len(moved) == len(coords):
                    target_places.append({"coords": coords})

    large_places = []
    lw = large_mask.shape[1]
    if lw > W:
        return None
    for left in range(max(W // 2 + 1, W - lw - 1), W - lw + 1):
        if want_big_clipped:
            for top in range(H - large_mask.shape[0] + 1, H):
                coords = _coords_from_mask(large_mask, top, left, (H, W))
                if 4 <= len(coords) <= 20 and _count_components_from_coords(set(coords)) == 1:
                    large_places.append({"coords": coords})
        else:
            top = H - large_mask.shape[0]
            if top >= H // 2:
                coords = _coords_from_mask(large_mask, top, left, (H, W))
                if 4 <= len(coords) <= 20:
                    large_places.append({"coords": coords})

    random.shuffle(target_places)
    random.shuffle(small_places)
    random.shuffle(large_places)
    for target_place in target_places[:20]:
        for small_place in small_places[:20]:
            if not _separated_pair(target_place["coords"], small_place["coords"]):
                continue
            for large_place in large_places[:20]:
                if not _separated_pair(target_place["coords"], large_place["coords"]):
                    continue
                if not _separated_pair(small_place["coords"], large_place["coords"]):
                    continue
                grid = _make_grid((H, W), small_place, large_place, target_place, colors)
                if _semantic_checks(grid):
                    return grid
    return None


def generate_input() -> np.ndarray:
    primary_attempts = 40
    for _ in range(primary_attempts):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    for _ in range(MAX_GEN_ATTEMPTS - primary_attempts):
        grid = _fallback_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
