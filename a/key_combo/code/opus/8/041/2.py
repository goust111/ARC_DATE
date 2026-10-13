import random
from collections import defaultdict, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

HOLE_TO_SIDE = {0: "up", 1: "right", 2: "down", 3: "left"}
SIDE_TO_DELTA = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}
SIDE_TO_COLOR = {"up": RED, "right": GREEN, "down": YELLOW, "left": ORANGE}

GRID_SIDE_RANGE = (23, 30)
OBJECT_COUNT_CHOICES = (3, 4)
GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 250


TEMPLATE_STRINGS = {
    0: [
        ("110", "111", "010"),
        ("010", "111", "110"),
        ("0110", "1110", "0111", "0010"),
    ],
    1: [
        ("111", "101", "111"),
        ("1111", "1001", "1011", "1111"),
        ("1111", "1001", "1001", "1011", "1111"),
    ],
    2: [
        ("11111", "10001", "11111", "10001", "11111"),
        ("111111", "100001", "111111", "100001", "111111"),
    ],
    3: [
        ("11111", "10001", "11111", "10001", "11111", "10001", "11111"),
        ("111111", "100001", "111111", "100001", "111111", "100001", "111111"),
    ],
}


def _template_to_mask(rows):
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _orientations(mask):
    out = []
    seen = set()
    for flip in (False, True):
        base = np.fliplr(mask) if flip else mask
        for k in range(4):
            oriented = np.rot90(base, k)
            key = oriented.tobytes() + bytes(oriented.shape)
            if key not in seen:
                seen.add(key)
                out.append(oriented.copy())
    return out


def _neighbors4(r, c, h, w):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _find_components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords):
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _hole_count(mask):
    bg = np.pad(~mask, 1, constant_values=True)
    h, w = bg.shape
    seen = np.zeros((h, w), dtype=bool)
    q = deque([(0, 0)])
    seen[0, 0] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if bg[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    holes = 0
    for r in range(h):
        for c in range(w):
            if bg[r, c] and not seen[r, c]:
                holes += 1
                q = deque([(r, c)])
                seen[r, c] = True
                while q:
                    rr, cc = q.popleft()
                    for nr, nc in _neighbors4(rr, cc, h, w):
                        if bg[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            q.append((nr, nc))
    return holes


def _marker_positions(shape):
    h, w = shape
    return {
        "up": (0, w // 2),
        "right": (h // 2, w - 1),
        "down": (h - 1, w // 2),
        "left": (h // 2, 0),
    }


def _analyze_objects(grid):
    blue_mask = grid == BLUE
    records = []
    for coords in _find_components(blue_mask):
        r0, c0, r1, c1 = _bbox(coords)
        local = blue_mask[r0 : r1 + 1, c0 : c1 + 1]
        holes = _hole_count(local)
        assert holes in HOLE_TO_SIDE, f"unsupported hole count: {holes}"
        side = HOLE_TO_SIDE[holes]
        step = local.shape[0] if side in ("up", "down") else local.shape[1]
        local_coords = np.argwhere(local)
        records.append(
            {
                "hole_count": holes,
                "side": side,
                "bbox": (r0, c0, r1, c1),
                "mask": local,
                "coords": coords,
                "local_coords": local_coords,
                "step": step,
            }
        )
    return records


def _copy_origins(spec, shape):
    h, w = spec["mask"].shape
    r0, c0, _, _ = spec["bbox"]
    dr, dc = SIDE_TO_DELTA[spec["side"]]
    step = spec["step"]
    origins = []
    k = 1
    while True:
        nr = r0 + dr * step * k
        nc = c0 + dc * step * k
        if nr < 0 or nc < 0 or nr + h > shape[0] or nc + w > shape[1]:
            break
        origins.append((nr, nc))
        k += 1
    return origins


def _render_from_specs(shape, specs):
    grid = np.zeros(shape, dtype=int)
    for side, (r, c) in _marker_positions(shape).items():
        grid[r, c] = SIDE_TO_COLOR[side]
    for spec in specs:
        r0 = spec["r0"]
        c0 = spec["c0"]
        h, w = spec["mask"].shape
        patch = grid[r0 : r0 + h, c0 : c0 + w]
        patch[spec["mask"]] = BLUE
    return grid


def _expected_output_from_specs(shape, specs):
    inp = _render_from_specs(shape, specs)
    out = inp.copy()
    for r, c in _marker_positions(shape).values():
        out[r, c] = BLACK

    original_blue = inp == BLUE
    for spec in sorted(specs, key=lambda item: item["hole_count"]):
        color = SIDE_TO_COLOR[spec["side"]]
        for nr, nc in _copy_origins(spec, shape):
            rr = nr + spec["cells"][:, 0]
            cc = nc + spec["cells"][:, 1]
            keep = ~original_blue[rr, cc]
            out[rr[keep], cc[keep]] = color
    return out


def _padded_bbox_intersects(box_a, box_b, pad=1):
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _copy_cells(spec, shape):
    cells = []
    for nr, nc in _copy_origins(spec, shape):
        rr = nr + spec["cells"][:, 0]
        cc = nc + spec["cells"][:, 1]
        cells.append({(int(r), int(c)) for r, c in zip(rr, cc)})
    return cells


def _copy_bboxes(spec, shape):
    h, w = spec["mask"].shape
    bboxes = []
    for nr, nc in _copy_origins(spec, shape):
        bboxes.append((nr, nc, nr + h - 1, nc + w - 1))
    return bboxes


def _range_with_bias(side, shape, mask_shape):
    h, w = shape
    oh, ow = mask_shape
    row_lo, row_hi = 1, h - oh - 2
    col_lo, col_hi = 1, w - ow - 2

    if side == "up":
        row_lo = max(row_lo, 2 * oh, h // 2)
    elif side == "down":
        row_hi = min(row_hi, h - 3 * oh, h // 2 - 1)
    elif side == "right":
        col_hi = min(col_hi, w - 3 * ow, w // 2 - 1)
    elif side == "left":
        col_lo = max(col_lo, 2 * ow, w // 2)

    return row_lo, row_hi, col_lo, col_hi


def _build_template_pool():
    pool = defaultdict(list)
    for holes, templates in TEMPLATE_STRINGS.items():
        for rows in templates:
            base = _template_to_mask(rows)
            for oriented in _orientations(base):
                if _hole_count(oriented) == holes and len(_find_components(oriented)) == 1:
                    pool[holes].append(oriented)
    for holes in range(4):
        assert pool[holes]
    return {k: list(v) for k, v in pool.items()}


TEMPLATE_POOL = _build_template_pool()


def _sample_specs(selected_holes):
    specs = []
    for holes in selected_holes:
        candidates = TEMPLATE_POOL[holes]
        side = HOLE_TO_SIDE[holes]
        if side in ("up", "down"):
            best = min(mask.shape[1] for mask in candidates)
            pool = [mask for mask in candidates if mask.shape[1] <= best + 1]
        else:
            best = min(mask.shape[0] for mask in candidates)
            pool = [mask for mask in candidates if mask.shape[0] <= best + 1]
        mask = random.choice(pool).copy()
        cells = np.argwhere(mask)
        specs.append(
            {
                "hole_count": holes,
                "side": side,
                "mask": mask,
                "cells": cells,
            }
        )
    return specs


def _choose_grid_shape(specs):
    min_side = GRID_SIDE_RANGE[0]
    max_h = max(spec["mask"].shape[0] for spec in specs)
    max_w = max(spec["mask"].shape[1] for spec in specs)
    low = max(min_side, 3 * max_h + 3, 3 * max_w + 3)
    high = GRID_SIDE_RANGE[1]
    assert low <= high
    return random.randint(low, high), random.randint(low, high)


def _sample_positions(shape, specs, mode):
    by_holes = {spec["hole_count"]: spec for spec in specs}
    ranges = {}
    for holes, spec in by_holes.items():
        ranges[holes] = _range_with_bias(spec["side"], shape, spec["mask"].shape)
        row_lo, row_hi, col_lo, col_hi = ranges[holes]
        if row_lo > row_hi or col_lo > col_hi:
            return None

    def near(preferred, lo, hi, radius=1):
        if lo > hi:
            return None
        preferred = min(max(preferred, lo), hi)
        start = max(lo, preferred - radius)
        stop = min(hi, preferred + radius)
        return random.randint(start, stop)

    pair_col = None
    pair_row = None

    if mode == "vertical":
        if 0 not in by_holes or 2 not in by_holes:
            return None
        col_lo = max(ranges[0][2], ranges[2][2])
        col_hi = min(ranges[0][3], ranges[2][3])
        if col_lo > col_hi:
            return None
        pair_col = near((col_lo + col_hi) // 2, col_lo, col_hi, radius=0)
    elif mode == "horizontal":
        if 1 not in by_holes or 3 not in by_holes:
            return None
        row_lo = max(ranges[1][0], ranges[3][0])
        row_hi = min(ranges[1][1], ranges[3][1])
        if row_lo > row_hi:
            return None
        pair_row = near((row_lo + row_hi) // 2, row_lo, row_hi, radius=0)

    def slot(lo, hi, name):
        span = hi - lo
        if name == "top":
            return lo
        if name == "upper":
            return lo + span // 3
        if name == "center":
            return (lo + hi) // 2
        if name == "lower":
            return hi - span // 3
        if name == "bottom":
            return hi
        if name == "left":
            return lo
        if name == "right":
            return hi
        raise ValueError(name)

    preferences = {}
    for holes in by_holes:
        row_lo, row_hi, col_lo, col_hi = ranges[holes]
        col_mid_left = col_lo + (col_hi - col_lo) // 3
        col_mid_right = col_hi - (col_hi - col_lo) // 3
        if mode == "vertical":
            if holes == 2:
                preferences[holes] = (slot(row_lo, row_hi, "top"), pair_col)
            elif holes == 0:
                preferences[holes] = (slot(row_lo, row_hi, "bottom"), pair_col)
            elif holes == 1:
                preferences[holes] = (slot(row_lo, row_hi, "upper"), slot(col_lo, col_hi, "left"))
            elif holes == 3:
                preferences[holes] = (slot(row_lo, row_hi, "lower"), slot(col_lo, col_hi, "right"))
        elif mode == "horizontal":
            if holes == 1:
                preferences[holes] = (pair_row, slot(col_lo, col_hi, "left"))
            elif holes == 3:
                preferences[holes] = (pair_row, slot(col_lo, col_hi, "right"))
            elif holes == 2:
                preferences[holes] = (slot(row_lo, row_hi, "top"), col_mid_right)
            elif holes == 0:
                preferences[holes] = (slot(row_lo, row_hi, "bottom"), col_mid_left)
        else:
            if holes == 2:
                preferences[holes] = (slot(row_lo, row_hi, "top"), col_mid_right)
            elif holes == 1:
                preferences[holes] = (slot(row_lo, row_hi, "upper"), slot(col_lo, col_hi, "left"))
            elif holes == 0:
                preferences[holes] = (slot(row_lo, row_hi, "bottom"), col_mid_left)
            elif holes == 3:
                preferences[holes] = (slot(row_lo, row_hi, "lower"), slot(col_lo, col_hi, "right"))
            else:
                preferences[holes] = (slot(row_lo, row_hi, "center"), slot(col_lo, col_hi, "center"))

    for spec in specs:
        holes = spec["hole_count"]
        row_lo, row_hi, col_lo, col_hi = ranges[holes]
        pref_r, pref_c = preferences[holes]
        r0 = near(pref_r, row_lo, row_hi)
        c0 = near(pref_c, col_lo, col_hi)
        if r0 is None or c0 is None:
            return None
        h, w = spec["mask"].shape
        spec["r0"] = r0
        spec["c0"] = c0
        spec["bbox"] = (r0, c0, r0 + h - 1, c0 + w - 1)
        spec["step"] = h if spec["side"] in ("up", "down") else w
    return specs


def _placement_is_clear(shape, specs, mode):
    for i, spec_a in enumerate(specs):
        for spec_b in specs[i + 1 :]:
            if _padded_bbox_intersects(spec_a["bbox"], spec_b["bbox"], pad=1):
                return False

    original_pixels = set()
    for spec in specs:
        rr = spec["r0"] + spec["cells"][:, 0]
        cc = spec["c0"] + spec["cells"][:, 1]
        original_pixels.update((int(r), int(c)) for r, c in zip(rr, cc))

    overlap_pairs = defaultdict(int)
    total_overlap = 0
    copy_pixel_total = 0
    all_seen = {}
    blocked_by_original = 0

    for spec in specs:
        origins = _copy_origins(spec, shape)
        if len(origins) < 2:
            return False
        for cell_set in _copy_cells(spec, shape):
            blocked_by_original += sum(cell in original_pixels for cell in cell_set)
            copy_pixel_total += len(cell_set)
            for cell in cell_set:
                if cell in all_seen:
                    other = all_seen[cell]
                    key = tuple(sorted((spec["hole_count"], other)))
                    overlap_pairs[key] += 1
                    total_overlap += 1
                else:
                    all_seen[cell] = spec["hole_count"]

    if copy_pixel_total > shape[0] * shape[1] * 0.42:
        return False

    if blocked_by_original > max(6, copy_pixel_total // 10):
        return False

    if mode == "vertical":
        if overlap_pairs.get((0, 2), 0) < 2:
            return False
    elif mode == "horizontal":
        if overlap_pairs.get((1, 3), 0) < 2:
            return False
    else:
        return False

    if total_overlap > copy_pixel_total * 0.2 + 3:
        return False
    return True


def _validate_generated_scene(inp, expected, specs, mode):
    marker_positions = _marker_positions(inp.shape)
    for side, (r, c) in marker_positions.items():
        if int(inp[r, c]) != SIDE_TO_COLOR[side]:
            return False

    records = _analyze_objects(inp)
    intended = sorted(spec["hole_count"] for spec in specs)
    observed = sorted(record["hole_count"] for record in records)
    if observed != intended or len(set(observed)) != len(observed):
        return False

    out = main(inp)
    if not np.array_equal(out, expected):
        return False

    colors = set(map(int, out.ravel()))
    if any(color in colors for color in SIDE_TO_COLOR.values()) and BLUE not in colors:
        return False

    if mode == "none":
        return True

    copy_hits = defaultdict(int)
    for spec in specs:
        for cell_set in _copy_cells(spec, inp.shape):
            for cell in cell_set:
                copy_hits[cell] += 1
    overlap = sum(1 for count in copy_hits.values() if count > 1)
    return overlap > 0


def _choose_mode(selected_holes):
    has_vertical = 0 in selected_holes and 2 in selected_holes
    has_horizontal = 1 in selected_holes and 3 in selected_holes
    if has_vertical and has_horizontal:
        return random.choice(["vertical", "horizontal"])
    if has_vertical:
        return "vertical"
    if has_horizontal:
        return "horizontal"
    return "none"


def _gen_attempt():
    object_count = random.choices(OBJECT_COUNT_CHOICES, weights=[2, 5], k=1)[0]
    selected_holes = sorted(random.sample([0, 1, 2, 3], k=object_count))
    specs = _sample_specs(selected_holes)
    shape = _choose_grid_shape(specs)
    mode = _choose_mode(selected_holes)

    for _ in range(PLACEMENT_ATTEMPTS):
        placed = _sample_positions(shape, [dict(spec) for spec in specs], mode)
        if placed is None:
            break
        if not _placement_is_clear(shape, placed, mode):
            continue
        inp = _render_from_specs(shape, placed)
        expected = _expected_output_from_specs(shape, placed)
        if _validate_generated_scene(inp, expected, placed, mode):
            return inp
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each blue object's enclosed holes, map that count to an edge direction, and stamp colored copies along that direction.

    Input structure:
    - The background is black.
    - There are 3 or 4 blue 4-connected objects, each with a distinct number of enclosed black holes.
    - The four border midpoints hold single-pixel direction markers: top, right, bottom, and left.

    Rule:
    - Count the number of 4-connected enclosed background regions inside each blue object.
    - Map hole counts to edge directions in order 0->top, 1->right, 2->bottom, 3->left, and read the stamp color from that edge marker.
    - Copy the object's exact blue shape repeatedly in that direction, stepping by its bbox height for vertical motion or bbox width for horizontal motion, until the next copy would leave the grid.
    - Keep the original objects blue, remove the four edge markers, and let later copies from larger hole counts overwrite earlier copies where copied pixels overlap.

    Key insight:
    - Hole count does not directly choose the output color; it first chooses a direction, and the edge marker at that direction supplies the color used for all copies.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    marker_colors = {}
    for side, (r, c) in _marker_positions(g.shape).items():
        marker_colors[side] = int(g[r, c])
        out[r, c] = BLACK

    records = _analyze_objects(g)
    holes = [record["hole_count"] for record in records]
    assert len(records) in (3, 4)
    assert len(set(holes)) == len(holes)

    original_blue = g == BLUE
    for record in sorted(records, key=lambda item: item["hole_count"]):
        color = marker_colors[record["side"]]
        for nr, nc in _copy_origins(record, g.shape):
            rr = nr + record["local_coords"][:, 0]
            cc = nc + record["local_coords"][:, 1]
            keep = ~original_blue[rr, cc]
            out[rr[keep], cc[keep]] = color

    return out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hole-count stamp task")
