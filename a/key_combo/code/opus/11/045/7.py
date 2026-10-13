import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7
FILL_COLORS = (YELLOW, GREEN, ORANGE)

GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 200
BBOX_GAP = 2

ONE_HOLE_VERTICALS = [
    np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=bool,
    ),
]

THREE_HOLE_VERTICALS = [
    np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=bool,
    ),
]

PARTIAL_CACHE: dict[tuple[bytes, tuple[int, int], str, int], list[np.ndarray]] = {}


def neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def bbox_of(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_mask(mask: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = bbox_of(coords)
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return local, (r0, c0, r1, c1)


def mask_connected(mask: np.ndarray) -> bool:
    count = int(mask.sum())
    return count > 0 and len(components(mask)) == 1


def repair_embedded_red(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    to_repair = []
    for r, c in np.argwhere(g == RED):
        blue_neighbors = sum(
            1
            for nr, nc in neighbors4(int(r), int(c))
            if in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == BLUE
        )
        if blue_neighbors >= 3:
            to_repair.append((int(r), int(c)))
    for r, c in to_repair:
        out[r, c] = BLUE
    return out


def axis_candidates(local_mask: np.ndarray) -> list[dict]:
    h, w = local_mask.shape
    candidates: list[dict] = []

    if w >= 2:
        left_subset_right = True
        right_subset_left = True
        different = False
        for r in range(h):
            for c in range(w):
                mc = w - 1 - c
                if c >= mc:
                    continue
                left = bool(local_mask[r, c])
                right = bool(local_mask[r, mc])
                different |= left != right
                if left and not right:
                    left_subset_right = False
                if right and not left:
                    right_subset_left = False
        if different:
            if left_subset_right and not right_subset_left:
                candidates.append({"axis": "vertical", "complete_side": "right"})
            if right_subset_left and not left_subset_right:
                candidates.append({"axis": "vertical", "complete_side": "left"})

    if h >= 2:
        top_subset_bottom = True
        bottom_subset_top = True
        different = False
        for r in range(h):
            mr = h - 1 - r
            if r >= mr:
                continue
            for c in range(w):
                top = bool(local_mask[r, c])
                bottom = bool(local_mask[mr, c])
                different |= top != bottom
                if top and not bottom:
                    top_subset_bottom = False
                if bottom and not top:
                    bottom_subset_top = False
        if different:
            if top_subset_bottom and not bottom_subset_top:
                candidates.append({"axis": "horizontal", "complete_side": "bottom"})
            if bottom_subset_top and not top_subset_bottom:
                candidates.append({"axis": "horizontal", "complete_side": "top"})

    return candidates


def complete_local_mask(local_mask: np.ndarray, candidate: dict) -> np.ndarray:
    if candidate["axis"] == "vertical":
        return local_mask | np.fliplr(local_mask)
    return local_mask | np.flipud(local_mask)


def enclosed_black_holes(grid: np.ndarray) -> list[dict]:
    holes = []
    for coords in components(grid == BLACK):
        r0, c0, r1, c1 = bbox_of(coords)
        if r0 == 0 or c0 == 0 or r1 == grid.shape[0] - 1 or c1 == grid.shape[1] - 1:
            continue
        holes.append(
            {
                "coords": coords,
                "area": int(len(coords)),
                "top_left": (int(r0), int(c0)),
            }
        )
    holes.sort(key=lambda h: (h["area"], h["top_left"][1], h["top_left"][0]))
    return holes


def solve_core(
    grid: np.ndarray,
    *,
    do_repair: bool = True,
    do_complete: bool = True,
    do_fill: bool = True,
) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = repair_embedded_red(g) if do_repair else g.copy()

    if do_complete:
        blue_mask = out == BLUE
        for coords in components(blue_mask):
            local_mask, (r0, c0, r1, c1) = crop_mask(blue_mask, coords)
            candidates = axis_candidates(local_mask)
            assert len(candidates) == 1, "expected one unique symmetry axis"
            completed = complete_local_mask(local_mask, candidates[0])
            patch = out[r0 : r1 + 1, c0 : c1 + 1].copy()
            patch[completed] = BLUE
            out[r0 : r1 + 1, c0 : c1 + 1] = patch

    if do_fill:
        holes = enclosed_black_holes(out)
        assert 2 <= len(holes) <= 3, "expected two or three enclosed holes after completion"
        for hole, color in zip(holes, FILL_COLORS):
            out[hole["coords"][:, 0], hole["coords"][:, 1]] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair embedded red defects, mirror-complete each partial blue object, then color the new enclosed holes by area.

    Input structure:
    - Black is background; there are one or two large blue 4-connected objects plus a few red single-pixel markers.
    - Some red pixels are embedded defects inside blue structure, while one or two red pixels sit isolated on the background.
    - Each repaired blue object is centered on the middle row or middle column of its bbox, with one side more complete than its mirrored partner.

    Rule:
    - Recolor every red pixel with at least three blue 4-neighbors to blue, but keep isolated red pixels unchanged.
    - For each repaired blue object, find the unique bbox midline whose mirrored halves satisfy a strict subset relation, and mirror the fuller side to complete the missing side.
    - After all blue objects are completed, find enclosed black holes, sort them by area (breaking ties by leftmost then topmost hole), and fill them with yellow, then green, then orange.

    Key insight:
    Embedded red noise must be repaired before symmetry detection, and the hole colors can only be assigned after completion closes the final cavities whose areas distinguish the fill colors.
    """
    return solve_core(np.asarray(grid, dtype=int), do_repair=True, do_complete=True, do_fill=True)


def build_vertical_object(hole_count: int) -> np.ndarray:
    if hole_count == 1:
        return random.choice(ONE_HOLE_VERTICALS).copy()
    if hole_count == 3:
        return random.choice(THREE_HOLE_VERTICALS).copy()
    raise ValueError("unsupported hole count")


def search_vertical_partial(full_mask: np.ndarray, complete_side: str, min_safe: int) -> np.ndarray:
    key = (full_mask.astype(np.uint8).tobytes(), tuple(full_mask.shape), complete_side, int(min_safe))
    cached = PARTIAL_CACHE.get(key)
    if cached:
        return random.choice(cached).copy()

    h, w = full_mask.shape
    axis = w // 2
    if complete_side == "left":
        partial = full_mask.copy()
        partial[:, axis + 1 :] = False
        candidates = [(int(r), int(c)) for r, c in np.argwhere(full_mask[:, axis + 1 :]) + np.array([0, axis + 1])]
        edge_col = w - 1
    else:
        partial = full_mask.copy()
        partial[:, :axis] = False
        candidates = [(int(r), int(c)) for r, c in np.argwhere(full_mask[:, :axis])]
        edge_col = 0

    edge_cells = [cell for cell in candidates if cell[1] == edge_col]
    found: list[np.ndarray] = []

    def valid(part: np.ndarray) -> bool:
        if not (part[0].any() and part[-1].any() and part[:, 0].any() and part[:, -1].any()):
            return False
        if not mask_connected(part):
            return False
        if enclosed_black_holes(np.where(part, BLUE, BLACK)):
            return False
        cands = axis_candidates(part)
        if cands != [{"axis": "vertical", "complete_side": complete_side}]:
            return False
        if not np.array_equal(complete_local_mask(part, cands[0]), full_mask):
            return False
        if len(removable_sites(part)) < min_safe:
            return False
        return 20 <= int(part.sum()) <= 40

    for _ in range(4000):
        part = partial.copy()
        for r, c in edge_cells:
            if random.random() < 0.35:
                part[r, c] = True
        if not part[:, edge_col].any():
            er, ec = random.choice(edge_cells)
            part[er, ec] = True
        for r, c in candidates:
            if c == edge_col:
                continue
            if random.random() < 0.18:
                part[r, c] = True
        if valid(part):
            found.append(part.copy())
            if len(found) >= 12:
                break

    if not found:
        non_edge = [cell for cell in candidates if cell not in edge_cells]
        for edge_cell in edge_cells:
            for bits in range(1 << len(non_edge)):
                part = partial.copy()
                part[edge_cell] = True
                for i, (r, c) in enumerate(non_edge):
                    if bits & (1 << i):
                        part[r, c] = True
                if valid(part):
                    found.append(part.copy())
                    if len(found) >= 12:
                        break
            if found:
                break

    if not found:
        raise ValueError("failed to find a valid partial mask")

    PARTIAL_CACHE[key] = found
    return random.choice(found).copy()


def partialize_mask(full_mask: np.ndarray, axis: str, complete_side: str) -> np.ndarray:
    partial = full_mask.copy()
    h, w = partial.shape
    if axis == "vertical":
        side_width = w // 2
        choices = [s for s in range(1, side_width) if s < side_width]
        strip = random.choice(choices)
        mid = w // 2
        if complete_side == "left":
            partial[:, mid + strip + 1 :] = False
        else:
            partial[:, : mid - strip] = False
    else:
        side_height = h // 2
        choices = [s for s in range(1, side_height) if s < side_height]
        strip = random.choice(choices)
        mid = h // 2
        if complete_side == "top":
            partial[mid + strip + 1 :, :] = False
        else:
            partial[: mid - strip, :] = False
    return partial


def removable_sites(local_mask: np.ndarray) -> list[tuple[int, int]]:
    candidates = []
    for r, c in np.argwhere(local_mask):
        rr = int(r)
        cc = int(c)
        blue_neighbors = sum(
            1
            for nr, nc in neighbors4(rr, cc)
            if in_bounds(local_mask.shape, nr, nc) and local_mask[nr, nc]
        )
        if blue_neighbors >= 3:
            candidates.append((rr, cc))

    random.shuffle(candidates)
    picked: list[tuple[int, int]] = []
    removed = np.zeros_like(local_mask, dtype=bool)

    for r, c in candidates:
        if any(abs(r - pr) + abs(c - pc) <= 1 for pr, pc in picked):
            continue
        trial = removed.copy()
        trial[r, c] = True
        remain = local_mask & ~trial
        if not mask_connected(remain):
            continue
        picked.append((r, c))
        removed = trial

    return picked


def place_rectangles(shapes: list[np.ndarray], canvas_h: int, canvas_w: int) -> list[tuple[int, int]] | None:
    rects = []
    order = list(range(len(shapes)))
    order.sort(key=lambda i: int(shapes[i].shape[0] * shapes[i].shape[1]), reverse=True)

    placed: dict[int, tuple[int, int]] = {}
    for idx in order:
        shape = shapes[idx]
        h, w = shape.shape
        options = []
        for r in range(canvas_h - h + 1):
            for c in range(canvas_w - w + 1):
                ok = True
                for j, (rr, cc) in placed.items():
                    hh, ww = shapes[j].shape
                    if not (
                        r + h - 1 + BBOX_GAP < rr
                        or rr + hh - 1 + BBOX_GAP < r
                        or c + w - 1 + BBOX_GAP < cc
                        or cc + ww - 1 + BBOX_GAP < c
                    ):
                        ok = False
                        break
                if ok:
                    options.append((r, c))
        if not options:
            return None
        placed[idx] = random.choice(options)

    for idx in range(len(shapes)):
        rects.append(placed[idx])
    return rects


def compose_mask(canvas_shape: tuple[int, int], masks: list[np.ndarray], positions: list[tuple[int, int]]) -> np.ndarray:
    canvas = np.zeros(canvas_shape, dtype=bool)
    for mask, (r0, c0) in zip(masks, positions):
        h, w = mask.shape
        canvas[r0 : r0 + h, c0 : c0 + w] |= mask
    return canvas


def choose_isolated_reds(input_grid: np.ndarray, output_grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = input_grid.shape
    cells = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(input_grid[r, c]) != BLACK or int(output_grid[r, c]) != BLACK:
                continue
            ok = True
            for nr, nc in neighbors4(r, c):
                if int(input_grid[nr, nc]) != BLACK or int(output_grid[nr, nc]) != BLACK:
                    ok = False
                    break
            if ok:
                cells.append((r, c))
    random.shuffle(cells)
    picked: list[tuple[int, int]] = []
    target = random.randint(1, 2)
    for cell in cells:
        if any(abs(cell[0] - r) + abs(cell[1] - c) <= 2 for r, c in picked):
            continue
        picked.append(cell)
        if len(picked) == target:
            break
    if len(picked) != target:
        raise ValueError("failed to place isolated red markers")
    return picked


def build_example() -> tuple[np.ndarray, np.ndarray]:
    object_count = 1 if random.random() < 0.45 else 2
    if object_count == 1:
        object_specs = [3]
    else:
        object_specs = [1, 1]

    full_masks = []
    partial_masks = []
    safe_sites_per_object = []
    hole_areas: list[int] = []
    partial_areas: list[int] = []

    for hole_count in object_specs:
        for _ in range(80):
            axis = random.choice(["vertical", "horizontal"])
            complete_side = random.choice(["left", "right"] if axis == "vertical" else ["top", "bottom"])
            vertical_full = build_vertical_object(hole_count)
            min_safe = 2

            if axis == "vertical":
                full_mask = vertical_full
                partial_mask = search_vertical_partial(vertical_full, complete_side, min_safe)
            else:
                vertical_side = "left" if complete_side == "top" else "right"
                full_mask = vertical_full.T
                partial_mask = search_vertical_partial(vertical_full, vertical_side, min_safe).T

            full_holes = enclosed_black_holes(np.where(full_mask, BLUE, BLACK))
            if len(full_holes) != hole_count:
                continue
            areas = [hole["area"] for hole in full_holes]
            if len(set(areas)) != len(areas):
                continue
            if any(area in hole_areas for area in areas):
                continue

            safe_sites = removable_sites(partial_mask)
            axis_info = axis_candidates(partial_mask)
            if len(axis_info) != 1 or axis_info[0]["axis"] != axis or axis_info[0]["complete_side"] != complete_side:
                continue
            if not np.array_equal(complete_local_mask(partial_mask, axis_info[0]), full_mask):
                continue
            if len(safe_sites) < min_safe:
                continue
            full_masks.append(full_mask)
            partial_masks.append(partial_mask)
            safe_sites_per_object.append(safe_sites)
            hole_areas.extend(areas)
            partial_areas.append(int(partial_mask.sum()))
            break
        else:
            raise ValueError("failed to build a valid object")

    assert len(set(hole_areas)) == len(hole_areas)

    max_h = max(mask.shape[0] for mask in full_masks)
    max_w = max(mask.shape[1] for mask in full_masks)
    min_h = max(15, max_h + 4)
    min_w = max(15, max_w + 4)
    if object_count == 2:
        min_h = max(min_h, full_masks[0].shape[0] + full_masks[1].shape[0] + BBOX_GAP + 1)
        min_w = max(min_w, max(mask.shape[1] for mask in full_masks) + 5)

    for _ in range(PLACEMENT_ATTEMPTS):
        canvas_h = random.randint(min_h, 22)
        canvas_w = random.randint(min_w, 22)
        positions = place_rectangles(full_masks, canvas_h, canvas_w)
        if positions is not None:
            break
    else:
        raise ValueError("failed to place objects")

    full_blue = compose_mask((canvas_h, canvas_w), full_masks, positions)
    partial_blue = compose_mask((canvas_h, canvas_w), partial_masks, positions)
    if len(components(full_blue)) != object_count or len(components(partial_blue)) != object_count:
        raise ValueError("unexpected component count after placement")

    site_records = []
    for obj_idx, ((r0, c0), sites) in enumerate(zip(positions, safe_sites_per_object)):
        for rr, cc in sites:
            site_records.append((obj_idx, r0 + rr, c0 + cc))
    if len(site_records) < 2:
        raise ValueError("not enough removable embedded sites")

    possible_subsets = []
    for bits in range(1, 1 << len(site_records)):
        chosen = [site_records[i] for i in range(len(site_records)) if bits & (1 << i)]
        if not (2 <= len(chosen) <= 4):
            continue
        by_object = [0] * len(partial_masks)
        for obj_idx, _r, _c in chosen:
            by_object[obj_idx] += 1
        if any(partial_areas[i] - by_object[i] < 20 for i in range(len(partial_masks))):
            continue
        possible_subsets.append(chosen)
    if not possible_subsets:
        raise ValueError("no valid defect subset")
    chosen_records = random.choice(possible_subsets)
    defect_sites = [(r, c) for _obj_idx, r, c in chosen_records]

    input_grid = np.zeros((canvas_h, canvas_w), dtype=int)
    input_grid[partial_blue] = BLUE
    for r, c in defect_sites:
        input_grid[r, c] = RED

    full_blue_grid = np.zeros((canvas_h, canvas_w), dtype=int)
    full_blue_grid[full_blue] = BLUE
    output_grid = solve_core(full_blue_grid, do_repair=False, do_complete=False, do_fill=True)

    isolated_reds = choose_isolated_reds(input_grid, output_grid)
    for r, c in isolated_reds:
        input_grid[r, c] = RED
        output_grid[r, c] = RED

    return input_grid, output_grid


def validate_example(inp: np.ndarray, out: np.ndarray) -> None:
    assert inp.ndim == 2 and out.ndim == 2
    assert inp.shape == out.shape
    assert max(inp.shape) <= 30
    assert not np.array_equal(inp, out)

    red_cells = [(int(r), int(c)) for r, c in np.argwhere(inp == RED)]
    embedded = []
    isolated = []
    for r, c in red_cells:
        blue_neighbors = sum(
            1
            for nr, nc in neighbors4(r, c)
            if in_bounds(inp.shape, nr, nc) and int(inp[nr, nc]) == BLUE
        )
        if blue_neighbors >= 3:
            embedded.append((r, c))
            continue
        if all(int(inp[nr, nc]) == BLACK for nr, nc in neighbors4(r, c)):
            isolated.append((r, c))
            continue
        raise AssertionError("every red pixel must be embedded or isolated")
    assert 2 <= len(embedded) <= 4
    assert 1 <= len(isolated) <= 2

    repaired = repair_embedded_red(inp)
    comps = components(repaired == BLUE)
    assert 1 <= len(comps) <= 2

    completed = repaired.copy()
    changed = False
    for coords in comps:
        local_mask, (r0, c0, r1, c1) = crop_mask(repaired == BLUE, coords)
        cands = axis_candidates(local_mask)
        assert len(cands) == 1
        full_local = complete_local_mask(local_mask, cands[0])
        assert int(full_local.sum()) > int(local_mask.sum())
        changed |= not np.array_equal(full_local, local_mask)
        patch = completed[r0 : r1 + 1, c0 : c1 + 1].copy()
        patch[full_local] = BLUE
        completed[r0 : r1 + 1, c0 : c1 + 1] = patch
    assert changed

    holes = enclosed_black_holes(completed)
    assert 2 <= len(holes) <= 3
    assert len({hole["area"] for hole in holes}) == len(holes)

    expected = completed.copy()
    for hole, color in zip(holes, FILL_COLORS):
        expected[hole["coords"][:, 0], hole["coords"][:, 1]] = color
    assert np.array_equal(expected, out)

    try:
        no_repair = solve_core(inp, do_repair=False, do_complete=True, do_fill=True)
        assert not np.array_equal(no_repair, out)
    except AssertionError:
        pass

    try:
        no_complete = solve_core(inp, do_repair=True, do_complete=False, do_fill=True)
        assert not np.array_equal(no_complete, out)
    except AssertionError:
        pass


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            inp, out = build_example()
            assert np.array_equal(main(inp), out)
            validate_example(inp, out)
            return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input")
