from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
ANCHOR_COLOR = 7
TARGET_COLOR = 9
PALETTE = [1, 2, 3, 4, 5, 6, 8]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
MAX_GEN_ATTEMPTS = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r0, c0, _, _ = _bbox(list(coords))
    return sorted((r - r0, c - c0) for r, c in coords)


def _shape_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    norm = _normalize(set(coords))
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _double_symmetric(coords: list[tuple[int, int]]) -> tuple[bool, int, int]:
    mask = _shape_mask(coords)
    h, w = mask.shape
    if h % 2 == 0 or w % 2 == 0:
        return False, -1, -1
    if not np.array_equal(mask, np.flipud(mask)):
        return False, -1, -1
    if not np.array_equal(mask, np.fliplr(mask)):
        return False, -1, -1
    r0, c0, r1, c1 = _bbox(coords)
    return True, (r0 + r1) // 2, (c0 + c1) // 2


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BACKGROUND or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "coords": coords, "bbox": _bbox(coords), "area": len(coords)})
    return comps


def _contact_count(anchor: tuple[int, int], coords_set: set[tuple[int, int]]) -> int:
    ar, ac = anchor
    return sum((ar + dr, ac + dc) in coords_set for dr, dc in NEIGHBORS4)


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    comps = _find_components(g)

    markers = []
    anchors = []
    targets = []
    objects = []
    for comp in comps:
        if comp["area"] == 1:
            cell = comp["coords"][0]
            if comp["color"] == ANCHOR_COLOR:
                anchors.append(cell)
            elif comp["color"] == TARGET_COLOR:
                targets.append(cell)
            elif cell[0] == 0:
                markers.append((cell, comp["color"]))
            else:
                raise AssertionError("unexpected singleton outside top-row markers/controls")
        else:
            objects.append(comp)

    assert len(markers) in (2, 3), "expected 2 or 3 top-row markers"
    marker_colors = [color for _cell, color in markers]
    assert len(set(marker_colors)) == len(marker_colors), "marker colors must be distinct"
    assert len(anchors) == 1, "expected one anchor"
    assert len(targets) in (2, 3), "expected 2 or 3 targets"
    assert 4 <= len(objects) <= 6, "expected 4 to 6 foreground objects"

    survivors = [obj for obj in objects if obj["color"] in marker_colors]
    rejected = [obj for obj in objects if obj["color"] not in marker_colors]
    assert len(survivors) >= 2, "expected at least two surviving objects"
    assert len(rejected) >= 1, "expected rejected objects"

    survivor_masks = [_shape_mask(obj["coords"]) for obj in survivors]
    ref_mask = survivor_masks[0]
    for mask in survivor_masks[1:]:
        assert np.array_equal(mask, ref_mask), "surviving objects must share one shape"
    assert len({obj["color"] for obj in survivors}) == len(survivors), "survivor colors must differ"

    symmetric_rejected = []
    for obj in rejected:
        ok, axis_r, axis_c = _double_symmetric(obj["coords"])
        if ok:
            symmetric_rejected.append((obj, axis_r, axis_c))
    assert len(symmetric_rejected) == 1, "expected exactly one doubly symmetric rejected object"
    axis_object, axis_r, axis_c = symmetric_rejected[0]

    anchor = anchors[0]
    scored = []
    for obj in survivors:
        coords_set = set(obj["coords"])
        scored.append((_contact_count(anchor, coords_set), obj))
    best_contact = max(score for score, _obj in scored)
    assert best_contact > 0, "anchor must touch a surviving object"
    best = [obj for score, obj in scored if score == best_contact]
    assert len(best) == 1, "anchor must select one surviving object uniquely"
    source_object = best[0]

    return {
        "grid": g,
        "markers": markers,
        "marker_colors": marker_colors,
        "anchor": anchor,
        "targets": sorted(targets),
        "objects": objects,
        "survivors": survivors,
        "rejected": rejected,
        "source_object": source_object,
        "axis_object": axis_object,
        "axis_r": axis_r,
        "axis_c": axis_c,
    }


def _step4_stamp_only(info: dict) -> np.ndarray:
    g = info["grid"]
    anchor = info["anchor"]
    source = info["source_object"]
    offsets = [(r - anchor[0], c - anchor[1]) for r, c in source["coords"]]
    out = np.zeros_like(g)
    for tr, tc in info["targets"]:
        for dr, dc in offsets:
            rr = tr + dr
            cc = tc + dc
            assert _in_bounds(g.shape, rr, cc), "stamp must stay in bounds"
            out[rr, cc] = int(source["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the anchor-selected surviving sprite onto each target, then mirror the stamps across axes borrowed from the rejected symmetric object.

    Input structure:
    - A black background grid contains 4-6 monochromatic 4-connected objects, plus 2-3 single-pixel top-row color markers.
    - Some object colors match the top-row markers and therefore survive the color filter; the rest are rejected.
    - One orange anchor pixel touches one surviving object most strongly, and 2-3 maroon target pixels mark new anchor positions.
    - Among the rejected objects, exactly one has both horizontal and vertical mirror symmetry.

    Rule:
    - Keep only objects whose colors appear in the top-row markers.
    - Choose the surviving object that has the largest 4-neighbor contact count with the orange anchor, and record that object's pixel offsets relative to the anchor.
    - Clear the whole scene to black, then stamp those offsets in the chosen object's color at every maroon target pixel.
    - Use the rejected doubly symmetric object to read off one horizontal mirror axis and one vertical mirror axis through its center, and copy every stamped pixel to its horizontal, vertical, and central reflections without overwriting existing non-black pixels.

    Key insight:
    - The filter splits the scene into two useful roles at once: surviving objects provide the stamp sprite, while the discarded symmetric object contributes the global mirror axes.
    """
    info = _analyze(np.asarray(grid, dtype=int))
    stamped = _step4_stamp_only(info)
    out = stamped.copy()
    axis_r = info["axis_r"]
    axis_c = info["axis_c"]

    base_pixels = [(int(r), int(c)) for r, c in np.argwhere(stamped != BACKGROUND)]
    for r, c in base_pixels:
        color = int(stamped[r, c])
        mirrors = [
            (r, c),
            (2 * axis_r - r, c),
            (r, 2 * axis_c - c),
            (2 * axis_r - r, 2 * axis_c - c),
        ]
        for rr, cc in mirrors:
            assert _in_bounds(out.shape, rr, cc), "mirror copy must stay in bounds"
            if out[rr, cc] == BACKGROUND:
                out[rr, cc] = color
    return out


def _expand8(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    expanded = set(cells)
    for r, c in cells:
        for dr, dc in NEIGHBORS8:
            expanded.add((r + dr, c + dc))
    return expanded


def _orbit(coords: set[tuple[int, int]], axis_r: int, axis_c: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in coords:
        out.add((r, c))
        out.add((2 * axis_r - r, c))
        out.add((r, 2 * axis_c - c))
        out.add((2 * axis_r - r, 2 * axis_c - c))
    return out


def _random_connected_shape(area: int, max_h: int, max_w: int, forbid_double_symmetry: bool) -> set[tuple[int, int]] | None:
    for _ in range(100):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr = r + dr
                    nc = c + dc
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            cells.add(random.choice(frontier))
            norm = _normalize(cells)
            h = max(r for r, _ in norm) + 1
            w = max(c for _, c in norm) + 1
            if h > max_h or w > max_w:
                break
        norm_set = set(_normalize(cells))
        if len(norm_set) != area:
            continue
        h = max(r for r, _ in norm_set) + 1
        w = max(c for _, c in norm_set) + 1
        if h > max_h or w > max_w:
            continue
        if forbid_double_symmetry and _double_symmetric(list(norm_set))[0]:
            continue
        return norm_set
    return None


def _anchor_candidates(shape: set[tuple[int, int]]) -> list[tuple[int, int]]:
    candidates = []
    shape_set = set(shape)
    r0, c0, r1, c1 = _bbox(list(shape_set))
    for r in range(r0 - 1, r1 + 2):
        for c in range(c0 - 1, c1 + 2):
            if (r, c) in shape_set:
                continue
            count = _contact_count((r, c), shape_set)
            if count >= 2:
                candidates.append((r, c))
    return candidates


def _survivor_shape_and_anchor() -> tuple[set[tuple[int, int]], tuple[int, int]] | None:
    for _ in range(120):
        area = random.randint(4, 7)
        shape = _random_connected_shape(area, max_h=4, max_w=4, forbid_double_symmetry=False)
        if shape is None:
            continue
        candidates = _anchor_candidates(shape)
        if not candidates:
            continue
        anchor = random.choice(candidates)
        return shape, anchor
    return None


def _symmetric_shape() -> set[tuple[int, int]]:
    templates = [
        {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)},
        {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)},
        {(0, 2), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (3, 2), (4, 2)},
    ]
    return set(random.choice(templates))


def _place_centered(shape: set[tuple[int, int]], center: tuple[int, int]) -> set[tuple[int, int]]:
    mask = _shape_mask(list(shape))
    h, w = mask.shape
    cr = h // 2
    cc = w // 2
    return {(center[0] + r - cr, center[1] + c - cc) for r, c in shape}


def _can_place_entity(
    cells: set[tuple[int, int]],
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    protected: set[tuple[int, int]] | None = None,
) -> bool:
    protected = protected or set()
    if not cells:
        return False
    if any(not _in_bounds(shape, r, c) for r, c in cells):
        return False
    if cells & occupied:
        return False
    if cells & protected:
        return False
    expanded = _expand8(cells)
    if expanded & occupied:
        return False
    if expanded & protected:
        return False
    return True


def _try_place_shape_with_anchor(
    grid_shape: tuple[int, int],
    shape: set[tuple[int, int]],
    anchor_rel: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], tuple[int, int]] | None:
    sr0, sc0, sr1, sc1 = _bbox(list(shape))
    ar, ac = anchor_rel
    min_r = min(sr0, ar)
    max_r = max(sr1, ar)
    min_c = min(sc0, ac)
    max_c = max(sc1, ac)
    rows = list(range(2 - min_r, grid_shape[0] - max_r))
    cols = list(range(0 - min_c, grid_shape[1] - max_c))
    random.shuffle(rows)
    random.shuffle(cols)
    for base_r in rows:
        for base_c in cols:
            cells = {(base_r + r, base_c + c) for r, c in shape}
            anchor = (base_r + ar, base_c + ac)
            entity = set(cells)
            entity.add(anchor)
            if _can_place_entity(entity, grid_shape, occupied):
                return cells, anchor
    return None


def _candidate_targets(
    grid_shape: tuple[int, int],
    offsets: list[tuple[int, int]],
    axis_r: int,
    axis_c: int,
    occupied: set[tuple[int, int]],
    row_sign: int,
    col_sign: int,
    require_axis_touch: bool,
) -> list[tuple[tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]]]:
    out = []
    protected_target_cells = occupied
    for tr in range(2, grid_shape[0]):
        for tc in range(grid_shape[1]):
            target = {(tr, tc)}
            if not _can_place_entity(target, grid_shape, protected_target_cells):
                continue
            stamped = {(tr + dr, tc + dc) for dr, dc in offsets}
            if any(not _in_bounds(grid_shape, r, c) for r, c in stamped):
                continue
            if stamped & occupied:
                continue
            rows_ok = all((r - axis_r) * row_sign <= 0 for r, _c in stamped)
            cols_ok = all((c - axis_c) * col_sign <= 0 for _r, c in stamped)
            if not rows_ok or not cols_ok:
                continue
            mirrored = _orbit(stamped, axis_r, axis_c)
            if any(not _in_bounds(grid_shape, r, c) for r, c in mirrored):
                continue
            if mirrored & occupied:
                continue
            if len(mirrored) == len(stamped):
                continue
            if require_axis_touch and not any(r == axis_r or c == axis_c for r, c in stamped):
                continue
            out.append(((tr, tc), stamped, mirrored))
    random.shuffle(out)
    return out


def _pick_targets(
    grid_shape: tuple[int, int],
    offsets: list[tuple[int, int]],
    axis_r: int,
    axis_c: int,
    occupied: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]]] | None:
    target_count = random.randint(2, 3)
    row_sign = random.choice([-1, 1])
    col_sign = random.choice([-1, 1])
    require_axis_touch = random.random() < 0.4
    candidates = _candidate_targets(
        grid_shape,
        offsets,
        axis_r,
        axis_c,
        occupied,
        row_sign,
        col_sign,
        require_axis_touch,
    )
    if len(candidates) < target_count:
        return None
    chosen = []
    used_cells: set[tuple[int, int]] = set()
    used_output: set[tuple[int, int]] = set()
    axis_touch_seen = False
    for candidate in candidates:
        target, stamped, mirrored = candidate
        if _expand8({target}) & used_cells:
            continue
        if stamped & used_output:
            continue
        if mirrored & used_output:
            continue
        chosen.append(candidate)
        used_cells.add(target)
        used_output |= mirrored
        axis_touch_seen = axis_touch_seen or any(r == axis_r or c == axis_c for r, c in stamped)
        if len(chosen) == target_count:
            break
    if len(chosen) != target_count:
        return None
    if require_axis_touch and not axis_touch_seen:
        return None
    return chosen


def _random_asymmetric_shape() -> set[tuple[int, int]] | None:
    area = random.randint(4, 8)
    return _random_connected_shape(area, max_h=5, max_w=5, forbid_double_symmetry=True)


def _try_place_object(
    grid_shape: tuple[int, int],
    shape: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    protected: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    r0, c0, r1, c1 = _bbox(list(shape))
    rows = list(range(2 - r0, grid_shape[0] - r1))
    cols = list(range(0 - c0, grid_shape[1] - c1))
    random.shuffle(rows)
    random.shuffle(cols)
    for base_r in rows:
        for base_c in cols:
            cells = {(base_r + r, base_c + c) for r, c in shape}
            if _can_place_entity(cells, grid_shape, occupied, protected):
                return cells
    return None


def _build_grid(
    shape: tuple[int, int],
    markers: list[tuple[int, int]],
    source_cells: set[tuple[int, int]],
    source_color: int,
    anchor: tuple[int, int],
    target_data: list[tuple[tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]]],
    survivor_objects: list[tuple[set[tuple[int, int]], int]],
    rejected_objects: list[tuple[set[tuple[int, int]], int]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for c, color in markers:
        grid[0, c] = color
    for r, c in source_cells:
        grid[r, c] = source_color
    grid[anchor] = ANCHOR_COLOR
    for target, _stamped, _mirrored in target_data:
        grid[target] = TARGET_COLOR
    for cells, color in survivor_objects:
        for r, c in cells:
            grid[r, c] = color
    for cells, color in rejected_objects:
        for r, c in cells:
            grid[r, c] = color
    return grid


def _validate_generated(grid: np.ndarray) -> None:
    info = _analyze(grid)
    stamped = _step4_stamp_only(info)
    out = main(grid)
    assert np.count_nonzero(stamped) > 0
    assert np.count_nonzero(out) > np.count_nonzero(stamped), "mirror step must add pixels"
    assert np.all(stamped[out == BACKGROUND] == BACKGROUND)
    output_colors = sorted(int(x) for x in np.unique(out) if int(x) != BACKGROUND)
    assert output_colors == [int(info["source_object"]["color"])], "output must use only the chosen survivor color"


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        dims = (random.randint(15, 18), random.randint(15, 18))
        marker_count = random.randint(2, 3)
        total_objects = random.randint(4, 6)
        rejected_count = total_objects - marker_count
        if rejected_count < 1:
            continue

        color_pool = random.sample(PALETTE, total_objects)
        survivor_colors = color_pool[:marker_count]
        rejected_colors = color_pool[marker_count:]

        axis_r = random.randint(dims[0] // 2 - 1, dims[0] // 2 + 1)
        axis_c = random.randint(dims[1] // 2 - 1, dims[1] // 2 + 1)
        sym_shape = _symmetric_shape()
        sym_cells = _place_centered(sym_shape, (axis_r, axis_c))
        if any(not _in_bounds(dims, r, c) for r, c in sym_cells):
            continue

        occupied = set(sym_cells)

        survivor_pack = _survivor_shape_and_anchor()
        if survivor_pack is None:
            continue
        survivor_shape, anchor_rel = survivor_pack
        placed = _try_place_shape_with_anchor(dims, survivor_shape, anchor_rel, occupied)
        if placed is None:
            continue
        source_cells, anchor = placed
        occupied |= source_cells | {anchor}

        source_color = random.choice(survivor_colors)
        offsets = [(r - anchor_rel[0], c - anchor_rel[1]) for r, c in survivor_shape]
        target_data = _pick_targets(dims, offsets, axis_r, axis_c, occupied)
        if target_data is None:
            continue
        final_output = set()
        for _target, _stamped, mirrored in target_data:
            final_output |= mirrored

        other_survivors: list[tuple[set[tuple[int, int]], int]] = []
        protected = final_output
        failed = False
        for color in survivor_colors:
            if color == source_color:
                continue
            placed_cells = _try_place_object(dims, survivor_shape, occupied, protected)
            if placed_cells is None:
                failed = True
                break
            other_survivors.append((placed_cells, color))
            occupied |= placed_cells
        if failed:
            continue

        other_rejected: list[tuple[set[tuple[int, int]], int]] = [(sym_cells, rejected_colors[0])]
        for color in rejected_colors[1:]:
            shape = _random_asymmetric_shape()
            if shape is None:
                failed = True
                break
            placed_cells = _try_place_object(dims, shape, occupied, protected)
            if placed_cells is None:
                failed = True
                break
            if _double_symmetric(list(placed_cells))[0]:
                failed = True
                break
            other_rejected.append((placed_cells, color))
            occupied |= placed_cells
        if failed:
            continue

        marker_cols = random.sample(range(dims[1]), marker_count)
        marker_cols.sort()
        if any(abs(marker_cols[i] - marker_cols[i - 1]) <= 1 for i in range(1, len(marker_cols))):
            continue
        markers = list(zip(marker_cols, survivor_colors))

        grid = _build_grid(
            dims,
            markers,
            source_cells,
            source_color,
            anchor,
            target_data,
            other_survivors,
            other_rejected,
        )

        try:
            _validate_generated(grid)
            oracle = main(grid)
            if np.array_equal(grid, oracle):
                continue
        except Exception:
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
