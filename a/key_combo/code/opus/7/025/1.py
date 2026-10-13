from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GRAY = 5

SLOT_SIZE = 3
SLOT_GAP = 1
SLOT_MARGIN = 1
SLOT_STRIDE = SLOT_SIZE + SLOT_GAP
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _shape_key(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox_from_coords(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _shape_mask(shape_key: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = max(r for r, _ in shape_key) + 1
    w = max(c for _, c in shape_key) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape_key:
        mask[r, c] = True
    return mask


def _extract_components(grid: np.ndarray, color: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    objects: list[dict] = []

    for r in range(H):
        for c in range(W):
            if seen[r, c] or g[r, c] != color:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords_list: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords_list.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and g[nr, nc] == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            coords = np.array(coords_list, dtype=int)
            bbox = _bbox_from_coords(coords)
            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            objects.append(
                {
                    "coords": coords,
                    "bbox": bbox,
                    "center": center,
                    "shape_key": _shape_key(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return objects


def _marker_window(grid: np.ndarray) -> tuple[int, int, int, int]:
    marker_coords = np.argwhere(np.asarray(grid, dtype=int) == RED)
    assert len(marker_coords) >= 2, "expected at least two red markers"
    return _bbox_from_coords(marker_coords)


def _split_regions(objects: list[dict], window: tuple[int, int, int, int]) -> tuple[list[dict], list[dict]]:
    r0, c0, r1, c1 = window
    inside: list[dict] = []
    outside: list[dict] = []
    for obj in objects:
        cr, cc = obj["center"]
        if r0 <= cr <= r1 and c0 <= cc <= c1:
            inside.append(obj)
        else:
            outside.append(obj)
    return inside, outside


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor gray objects by their shape frequency inside versus outside the red-marker window.

    Input structure:
    - The grid uses black background, sparse red marker pixels, and several disjoint 4-connected gray objects.
    - The red pixels define one axis-aligned window via their minimum bounding box.
    - Each gray object is treated as a separate bbox-normalized shape and assigned to the inner or outer region by its bbox-center cell.

    Rule:
    - Compute the red-marker bounding box window.
    - Extract every 4-connected gray object and normalize its bbox mask to a translation-invariant shape key.
    - Split objects into the inner group and outer group depending on whether the floor-centered bbox center lies inside the window.
    - Within each group independently, recolor shapes seen once to yellow and shapes seen two or more times to blue, while keeping markers and background unchanged.

    Key insight:
    - The red window is not itself the output target; it creates two separate counting spaces, so the same shape can be unique in one region and repeated in the other.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    window = _marker_window(g)
    objects = _extract_components(g, GRAY)
    assert objects, "expected at least one gray object"

    inside, outside = _split_regions(objects, window)
    out = g.copy()
    for region in (inside, outside):
        counts = Counter(obj["shape_key"] for obj in region)
        for obj in region:
            color = YELLOW if counts[obj["shape_key"]] == 1 else BLUE
            out[obj["coords"][:, 0], obj["coords"][:, 1]] = color
    return out


def _enumerate_shape_library() -> list[tuple[tuple[int, int], ...]]:
    shapes: set[tuple[tuple[int, int], ...]] = set()
    for bits in range(1 << 9):
        area = bits.bit_count()
        if not 3 <= area <= 6:
            continue
        mask = np.zeros((3, 3), dtype=bool)
        for idx in range(9):
            if bits & (1 << idx):
                mask[idx // 3, idx % 3] = True
        coords = np.argwhere(mask)
        if coords.size == 0:
            continue

        seed = tuple(map(int, coords[0]))
        seen = {seed}
        queue = deque([seed])
        while queue:
            r, c = queue.popleft()
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < 3 and 0 <= nc < 3 and mask[nr, nc] and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    queue.append((nr, nc))
        if len(seen) != area:
            continue
        shapes.add(_shape_key(coords))

    return sorted(shapes)


SHAPE_LIBRARY = _enumerate_shape_library()


def _slot_top_left(slot_r: int, slot_c: int) -> tuple[int, int]:
    return SLOT_MARGIN + slot_r * SLOT_STRIDE, SLOT_MARGIN + slot_c * SLOT_STRIDE


def _grid_shape(slot_rows: int, slot_cols: int) -> tuple[int, int]:
    return (
        2 * SLOT_MARGIN + slot_rows * SLOT_SIZE + (slot_rows - 1) * SLOT_GAP,
        2 * SLOT_MARGIN + slot_cols * SLOT_SIZE + (slot_cols - 1) * SLOT_GAP,
    )


def _window_from_block(start_r: int, start_c: int, block_h: int, block_w: int) -> tuple[int, int, int, int]:
    top, left = _slot_top_left(start_r, start_c)
    bottom_slot_top, right_slot_left = _slot_top_left(start_r + block_h - 1, start_c + block_w - 1)
    return top - 1, left - 1, bottom_slot_top + SLOT_SIZE, right_slot_left + SLOT_SIZE


def _render_in_slot(grid: np.ndarray, slot_r: int, slot_c: int, shape_key: tuple[tuple[int, int], ...]) -> None:
    mask = _shape_mask(shape_key)
    h, w = mask.shape
    max_dr = SLOT_SIZE - h
    max_dc = SLOT_SIZE - w
    dr = random.randint(0, max_dr)
    dc = random.randint(0, max_dc)
    r0, c0 = _slot_top_left(slot_r, slot_c)
    rows, cols = np.nonzero(mask)
    grid[r0 + dr + rows, c0 + dc + cols] = GRAY


def _choose_marker_cells(window: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = window
    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    count = random.randint(2, 4)
    if count == 2:
        return random.choice([[(r0, c0), (r1, c1)], [(r0, c1), (r1, c0)]])
    if count == 3:
        missing = random.randrange(4)
        return [corner for idx, corner in enumerate(corners) if idx != missing]
    return corners


def _build_shape_plan() -> tuple[list[tuple[tuple[int, int], ...]], list[tuple[tuple[int, int], ...]]]:
    mandatory = random.sample(SHAPE_LIBRARY, 8)
    shared, in_rep, out_rep, in_uni, out_uni, extra_u1, extra_u2, extra_p1 = mandatory

    shared_inside = 1 if random.random() < 0.5 else 2
    shared_outside = 3 - shared_inside

    inside = [shared] * shared_inside + [in_rep, in_rep, in_uni]
    outside = [shared] * shared_outside + [out_rep, out_rep, out_uni]

    if random.random() < 0.65:
        if len(inside) < len(outside) and len(inside) < 6:
            inside.append(extra_u1)
        elif len(outside) < 6:
            outside.append(extra_u1)
    if random.random() < 0.45:
        if len(inside) <= 4 and (len(inside) <= len(outside) or len(outside) > 4):
            inside.extend([extra_p1, extra_p1])
        elif len(outside) <= 4:
            outside.extend([extra_p1, extra_p1])
    if random.random() < 0.25:
        if len(inside) < len(outside) and len(inside) < 6:
            inside.append(extra_u2)
        elif len(outside) < 6:
            outside.append(extra_u2)

    random.shuffle(inside)
    random.shuffle(outside)
    return inside, outside


def _choose_layout(
    inside_count: int, outside_count: int
) -> tuple[int, int, int, int, int, int] | None:
    candidates: list[tuple[int, int, int, int, int, int]] = []
    for slot_rows, slot_cols in ((4, 4), (4, 5), (5, 4), (5, 5)):
        for block_h, block_w in ((2, 2), (2, 3), (3, 2)):
            capacity = block_h * block_w
            if capacity < inside_count:
                continue
            if slot_rows * slot_cols - capacity < outside_count:
                continue
            for start_r in range(slot_rows - block_h + 1):
                for start_c in range(slot_cols - block_w + 1):
                    candidates.append((slot_rows, slot_cols, block_h, block_w, start_r, start_c))
    if not candidates:
        return None
    return random.choice(candidates)


def _collect_slots(
    slot_rows: int, slot_cols: int, block_h: int, block_w: int, start_r: int, start_c: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    inside_slots: list[tuple[int, int]] = []
    outside_slots: list[tuple[int, int]] = []
    for sr in range(slot_rows):
        for sc in range(slot_cols):
            if start_r <= sr < start_r + block_h and start_c <= sc < start_c + block_w:
                inside_slots.append((sr, sc))
            else:
                outside_slots.append((sr, sc))
    return inside_slots, outside_slots


def _chebyshev_separated(objects: list[dict]) -> bool:
    coord_sets = [set(map(tuple, obj["coords"])) for obj in objects]
    for i, cells_a in enumerate(coord_sets):
        for cells_b in coord_sets[i + 1 :]:
            for r0, c0 in cells_a:
                for r1, c1 in cells_b:
                    if max(abs(r0 - r1), abs(c0 - c1)) <= 1:
                        return False
    return True


def _semantic_summary(grid: np.ndarray) -> dict | None:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return None
    marker_coords = np.argwhere(g == RED)
    if not 2 <= len(marker_coords) <= 4:
        return None

    window = _marker_window(g)
    objects = _extract_components(g, GRAY)
    if not 8 <= len(objects) <= 12:
        return None
    if any(not 3 <= obj["area"] <= 6 for obj in objects):
        return None
    if not _chebyshev_separated(objects):
        return None

    inside, outside = _split_regions(objects, window)
    if len(inside) < 2 or len(outside) < 2:
        return None

    region_data = {}
    for name, region in (("inside", inside), ("outside", outside)):
        counts = Counter(obj["shape_key"] for obj in region)
        uniques = {key for key, count in counts.items() if count == 1}
        repeated = {key for key, count in counts.items() if count >= 2}
        if not uniques or not repeated:
            return None
        region_data[name] = {
            "objects": region,
            "counts": counts,
            "uniques": uniques,
            "repeated": repeated,
        }

    shared_shapes = set(region_data["inside"]["counts"]) & set(region_data["outside"]["counts"])
    cross_region_difference = False
    for key in shared_shapes:
        cin = region_data["inside"]["counts"][key]
        cout = region_data["outside"]["counts"][key]
        if {cin, cout} == {1, 2} or (cin == 1 and cout >= 2) or (cout == 1 and cin >= 2):
            cross_region_difference = True
            break
    if not cross_region_difference:
        return None

    return {
        "window": window,
        "objects": objects,
        "inside": region_data["inside"],
        "outside": region_data["outside"],
    }


def _passes_semantic_checks(grid: np.ndarray) -> bool:
    summary = _semantic_summary(grid)
    if summary is None:
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if not np.array_equal(out[grid == BLACK], grid[grid == BLACK]):
        return False
    if not np.array_equal(out[grid == RED], grid[grid == RED]):
        return False

    for region_name in ("inside", "outside"):
        region = summary[region_name]
        for obj in region["objects"]:
            expected = YELLOW if region["counts"][obj["shape_key"]] == 1 else BLUE
            rows = obj["coords"][:, 0]
            cols = obj["coords"][:, 1]
            values = set(map(int, out[rows, cols]))
            if values != {expected}:
                return False
        expected_colors = {
            YELLOW if region["counts"][obj["shape_key"]] == 1 else BLUE for obj in region["objects"]
        }
        if expected_colors != {BLUE, YELLOW}:
            return False

    return True


def _generate_attempt() -> np.ndarray | None:
    inside_shapes, outside_shapes = _build_shape_plan()
    layout = _choose_layout(len(inside_shapes), len(outside_shapes))
    if layout is None:
        return None
    slot_rows, slot_cols, block_h, block_w, start_r, start_c = layout
    H, W = _grid_shape(slot_rows, slot_cols)
    grid = np.zeros((H, W), dtype=int)

    inside_slots, outside_slots = _collect_slots(slot_rows, slot_cols, block_h, block_w, start_r, start_c)
    chosen_inside_slots = random.sample(inside_slots, len(inside_shapes))
    chosen_outside_slots = random.sample(outside_slots, len(outside_shapes))

    window = _window_from_block(start_r, start_c, block_h, block_w)
    for r, c in _choose_marker_cells(window):
        grid[r, c] = RED

    for (slot_r, slot_c), shape in zip(chosen_inside_slots, inside_shapes):
        _render_in_slot(grid, slot_r, slot_c, shape)
    for (slot_r, slot_c), shape in zip(chosen_outside_slots, outside_shapes):
        _render_in_slot(grid, slot_r, slot_c, shape)

    if not _passes_semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 1")
