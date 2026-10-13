from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

BG = 0
FILL = 5
BRIDGE = 8
INPUT_COLORS = [1, 2, 3, 4, 6, 7]
HOLE_TO_COLOR = {0: 1, 1: 2, 2: 3, 3: 4}

PATCH = 5
CANVAS = 26
GEN_ATTEMPTS = 400

HOLE_PATTERNS: dict[int, list[tuple[tuple[int, int], ...]]] = {
    0: [()],
    1: [((2, 2),)],
    2: [((1, 1), (3, 3)), ((1, 3), (3, 1))],
    3: [((1, 1), (1, 3), (3, 2))],
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    dirs = DIR4 if connectivity == 4 else DIR8
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for dr, dc in dirs:
                nr = cr + dr
                nc = cc + dc
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _complete_bbox_patch(sub: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
    black = sub == BG
    completed = sub.copy()
    hole_mask = np.zeros(sub.shape, dtype=bool)
    holes = 0

    for coords in _components(black, connectivity=4):
        touches_edge = bool(
            np.any(
                (coords[:, 0] == 0)
                | (coords[:, 1] == 0)
                | (coords[:, 0] == sub.shape[0] - 1)
                | (coords[:, 1] == sub.shape[1] - 1)
            )
        )
        if touches_edge:
            completed[coords[:, 0], coords[:, 1]] = FILL
        else:
            hole_mask[coords[:, 0], coords[:, 1]] = True
            holes += 1

    return completed, holes, hole_mask


def _analyze_objects(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    objects: list[dict] = []
    recolored = np.zeros_like(g)

    for coords in _components(g != BG, connectivity=8):
        r0, c0, r1, c1 = _bbox(coords)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        completed_sub, hole_count, hole_mask = _complete_bbox_patch(sub)
        if hole_count not in HOLE_TO_COLOR:
            raise AssertionError(f"unsupported hole count: {hole_count}")

        raw_mask = np.zeros_like(g, dtype=bool)
        raw_mask[coords[:, 0], coords[:, 1]] = True
        completed_mask = completed_sub != BG
        completed_global = np.zeros_like(g, dtype=bool)
        completed_global[r0 : r1 + 1, c0 : c1 + 1] = completed_mask

        final_color = HOLE_TO_COLOR[hole_count]
        patch_out = np.zeros_like(sub)
        patch_out[completed_mask] = final_color
        recolored[r0 : r1 + 1, c0 : c1 + 1] = patch_out

        colors = set(map(int, sub[sub != BG].tolist()))
        if len(colors) != 1:
            raise AssertionError("each input object must be monochromatic")

        missing_count = int(np.sum((sub == BG) & completed_mask))
        objects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "raw_mask": raw_mask,
                "completed_mask": completed_global,
                "hole_mask": hole_mask,
                "hole_count": hole_count,
                "final_color": final_color,
                "raw_color": next(iter(colors)),
                "missing_count": missing_count,
            }
        )

    return objects, recolored


def _bbox_bridge_candidate(grid: np.ndarray, a: dict, b: dict) -> tuple[int, tuple[int, int, int, int]] | None:
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]

    shared_r0 = max(ar0, br0)
    shared_r1 = min(ar1, br1)
    if shared_r1 - shared_r0 + 1 >= 3 and (ac1 < bc0 or bc1 < ac0):
        if ac1 < bc0:
            gap_c0, gap_c1 = ac1 + 1, bc0 - 1
        else:
            gap_c0, gap_c1 = bc1 + 1, ac0 - 1
        if gap_c0 <= gap_c1 and np.all(grid[shared_r0 : shared_r1 + 1, gap_c0 : gap_c1 + 1] == BG):
            return gap_c1 - gap_c0 + 1, (shared_r0 + 1, gap_c0, shared_r1 - 1, gap_c1)

    shared_c0 = max(ac0, bc0)
    shared_c1 = min(ac1, bc1)
    if shared_c1 - shared_c0 + 1 >= 3 and (ar1 < br0 or br1 < ar0):
        if ar1 < br0:
            gap_r0, gap_r1 = ar1 + 1, br0 - 1
        else:
            gap_r0, gap_r1 = br1 + 1, ar0 - 1
        if gap_r0 <= gap_r1 and np.all(grid[gap_r0 : gap_r1 + 1, shared_c0 : shared_c1 + 1] == BG):
            return gap_r1 - gap_r0 + 1, (gap_r0, shared_c0 + 1, gap_r1, shared_c1 - 1)

    return None


def _collect_bbox_bridges(grid: np.ndarray, objects: list[dict]) -> list[dict]:
    bridges: list[dict] = []
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if objects[i]["final_color"] != objects[j]["final_color"]:
                continue
            candidate = _bbox_bridge_candidate(grid, objects[i], objects[j])
            if candidate is None:
                continue
            distance, paint = candidate
            bridges.append({"pair": (i, j), "distance": distance, "paint": paint})
    bridges.sort(key=lambda item: item["distance"], reverse=True)
    return bridges


def _row_boundary_bridge(
    base: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
) -> tuple[int, tuple[int, int, int, int]] | None:
    rows: list[int] = []
    pairs: list[tuple[int, int]] = []
    for r in range(base.shape[0]):
        left_cols = np.flatnonzero(left_mask[r])
        right_cols = np.flatnonzero(right_mask[r])
        if left_cols.size == 0 or right_cols.size == 0:
            continue
        left_edge = int(left_cols.max())
        right_edge = int(right_cols.min())
        if right_edge - left_edge < 2:
            continue
        if np.all(base[r, left_edge + 1 : right_edge] == BG):
            rows.append(r)
            pairs.append((left_edge, right_edge))

    best: tuple[int, tuple[int, int, int, int]] | None = None
    i = 0
    while i < len(rows):
        j = i
        while j + 1 < len(rows) and rows[j + 1] == rows[j] + 1 and pairs[j + 1] == pairs[i]:
            j += 1
        run_len = rows[j] - rows[i] + 1
        if run_len >= 3:
            left_edge, right_edge = pairs[i]
            candidate = (right_edge - left_edge - 1, (rows[i] + 1, left_edge + 1, rows[j] - 1, right_edge - 1))
            if best is None or run_len > (best[1][2] - best[1][0] + 2):
                best = candidate
        i = j + 1
    return best


def _col_boundary_bridge(
    base: np.ndarray,
    top_mask: np.ndarray,
    bottom_mask: np.ndarray,
) -> tuple[int, tuple[int, int, int, int]] | None:
    cols: list[int] = []
    pairs: list[tuple[int, int]] = []
    for c in range(base.shape[1]):
        top_rows = np.flatnonzero(top_mask[:, c])
        bottom_rows = np.flatnonzero(bottom_mask[:, c])
        if top_rows.size == 0 or bottom_rows.size == 0:
            continue
        top_edge = int(top_rows.max())
        bottom_edge = int(bottom_rows.min())
        if bottom_edge - top_edge < 2:
            continue
        if np.all(base[top_edge + 1 : bottom_edge, c] == BG):
            cols.append(c)
            pairs.append((top_edge, bottom_edge))

    best: tuple[int, tuple[int, int, int, int]] | None = None
    i = 0
    while i < len(cols):
        j = i
        while j + 1 < len(cols) and cols[j + 1] == cols[j] + 1 and pairs[j + 1] == pairs[i]:
            j += 1
        run_len = cols[j] - cols[i] + 1
        if run_len >= 3:
            top_edge, bottom_edge = pairs[i]
            candidate = (bottom_edge - top_edge - 1, (top_edge + 1, cols[i] + 1, bottom_edge - 1, cols[j] - 1))
            if best is None or run_len > (best[1][3] - best[1][1] + 2):
                best = candidate
        i = j + 1
    return best


def _collect_silhouette_bridges(base: np.ndarray, objects: list[dict], mask_key: str) -> list[dict]:
    bridges: list[dict] = []
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if objects[i]["final_color"] != objects[j]["final_color"]:
                continue

            mask_i = objects[i][mask_key]
            mask_j = objects[j][mask_key]
            candidate = None
            for left, right in ((mask_i, mask_j), (mask_j, mask_i)):
                horizontal = _row_boundary_bridge(base, left, right)
                if horizontal is not None:
                    candidate = horizontal
            for top, bottom in ((mask_i, mask_j), (mask_j, mask_i)):
                vertical = _col_boundary_bridge(base, top, bottom)
                if vertical is not None:
                    candidate = vertical
            if candidate is not None:
                distance, paint = candidate
                bridges.append({"pair": (i, j), "distance": distance, "paint": paint})
    bridges.sort(key=lambda item: item["distance"], reverse=True)
    return bridges


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete broken 5x5 objects, recolor each by its completed hole count, then bridge same-color facing pairs.

    Input structure:
    - The grid has a black background and several separated 8-connected colored objects.
    - Each object sits inside a 5x5 bounding box and is missing 2-5 black edge-connected pixels.
    - Some black cells inside the 5x5 box are true enclosed holes that must stay black.

    Rule:
    - For each object's bounding box, find 4-connected black components inside the box.
    - Fill every black component that touches the box edge with gray to complete the object, but keep fully enclosed black components as holes.
    - Count the completed holes and recolor every non-black object pixel by the fixed map H=0->blue(1), H=1->red(2), H=2->green(3), H=3->yellow(4).
    - After recoloring, connect every same-color pair whose completed bounding boxes face each other horizontally or vertically across pure black background with overlap at least three cells, painting the gap cyan(8) after eroding the shared span by one cell at both ends.
    - If several bridges exist, draw all of them, ordering by gap length from long to short.

    Key insight:
    - The input colors are all different, so same-color pairs appear only after topology-based recoloring of the completed objects.
    """
    g = np.asarray(grid, dtype=int)
    objects, recolored = _analyze_objects(g)
    out = recolored.copy()
    for bridge in _collect_bbox_bridges(recolored, objects):
        r0, c0, r1, c1 = bridge["paint"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BRIDGE
    return out


def _completed_mask(hole_count: int) -> np.ndarray:
    mask = np.ones((PATCH, PATCH), dtype=bool)
    holes = random.choice(HOLE_PATTERNS[hole_count])
    for r, c in holes:
        mask[r, c] = False
    return mask


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask, connectivity=8)
    return len(comps) == 1


def _valid_removed(mask: np.ndarray, removed: set[tuple[int, int]]) -> bool:
    if not 2 <= len(removed) <= 5:
        return False

    for r, c in removed:
        if not mask[r, c]:
            return False
        if r not in (0, PATCH - 1) and c not in (0, PATCH - 1):
            return False
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < PATCH and 0 <= nc < PATCH and not mask[nr, nc]:
                return False

    remaining = mask.copy()
    for r, c in removed:
        remaining[r, c] = False
    if not _is_connected(remaining):
        return False

    coords = np.argwhere(remaining)
    r0, c0, r1, c1 = _bbox(coords)
    return (r1 - r0 + 1, c1 - c0 + 1) == (PATCH, PATCH)


def _choose_removed(mask: np.ndarray, required_cells: list[tuple[int, int]] | None = None) -> set[tuple[int, int]] | None:
    edge_cells = [(r, c) for r in range(PATCH) for c in range(PATCH) if mask[r, c] and (r in (0, PATCH - 1) or c in (0, PATCH - 1))]
    required = [cell for cell in (required_cells or []) if cell in edge_cells]
    if required_cells and not required:
        return None

    for _ in range(200):
        k = random.randint(2, 5)
        chosen: set[tuple[int, int]] = set()
        if required:
            chosen.add(random.choice(required))
        remaining = [cell for cell in edge_cells if cell not in chosen]
        random.shuffle(remaining)
        chosen.update(remaining[: k - len(chosen)])
        if _valid_removed(mask, chosen):
            return chosen
    return None


def _patch_from_mask(mask: np.ndarray, color: int, removed: set[tuple[int, int]]) -> np.ndarray:
    patch = np.zeros((PATCH, PATCH), dtype=int)
    for r in range(PATCH):
        for c in range(PATCH):
            if mask[r, c] and (r, c) not in removed:
                patch[r, c] = color
    return patch


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + padding < br0 or br1 + padding < ar0 or ac1 + padding < bc0 or bc1 + padding < ac0)


def _place_random_box(used: list[tuple[int, int, int, int]]) -> tuple[int, int] | None:
    for _ in range(200):
        r0 = random.randint(0, CANVAS - PATCH)
        c0 = random.randint(0, CANVAS - PATCH)
        box = (r0, c0, r0 + PATCH - 1, c0 + PATCH - 1)
        if all(not _boxes_overlap(box, other, padding=1) for other in used):
            return r0, c0
    return None


def _shared_mid_for_horizontal(a_row: int, b_row: int) -> int:
    rows = list(range(max(a_row, b_row), min(a_row + PATCH - 1, b_row + PATCH - 1) + 1))
    return rows[len(rows) // 2]


def _shared_mid_for_vertical(a_col: int, b_col: int) -> int:
    cols = list(range(max(a_col, b_col), min(a_col + PATCH - 1, b_col + PATCH - 1) + 1))
    return cols[len(cols) // 2]


def _crop_canvas(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != BG)
    r0, c0, r1, c1 = _bbox(coords)
    top = random.randint(0, min(2, r0))
    left = random.randint(0, min(2, c0))
    bottom = random.randint(0, min(2, grid.shape[0] - 1 - r1))
    right = random.randint(0, min(2, grid.shape[1] - 1 - c1))
    return grid[r0 - top : r1 + bottom + 1, c0 - left : c1 + right + 1]


def _validate_generated(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False

    objects, recolored = _analyze_objects(grid)
    if len(objects) not in (5, 6):
        return False

    input_colors = [obj["raw_color"] for obj in objects]
    if len(set(input_colors)) != len(input_colors):
        return False

    hole_counts = [obj["hole_count"] for obj in objects]
    groups = Counter(hole_counts)
    if sum(count >= 2 for count in groups.values()) < 2:
        return False

    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        if (r1 - r0 + 1, c1 - c0 + 1) != (PATCH, PATCH):
            return False
        if not 2 <= obj["missing_count"] <= 5:
            return False

    bbox_bridges = _collect_bbox_bridges(recolored, objects)
    if len(bbox_bridges) != 1:
        return False

    same_color_pairs = {
        (i, j)
        for i in range(len(objects))
        for j in range(i + 1, len(objects))
        if objects[i]["final_color"] == objects[j]["final_color"]
    }
    bridged_pairs = {tuple(bridge["pair"]) for bridge in bbox_bridges}
    if len(same_color_pairs - bridged_pairs) < 1:
        return False

    raw_base = np.zeros_like(grid)
    for obj in objects:
        raw_base[obj["raw_mask"]] = obj["final_color"]
    raw_silhouette_bridges = _collect_silhouette_bridges(raw_base, objects, "raw_mask")
    if len(raw_silhouette_bridges) >= len(bbox_bridges):
        return False

    expected = recolored.copy()
    for bridge in bbox_bridges:
        r0, c0, r1, c1 = bridge["paint"]
        expected[r0 : r1 + 1, c0 : c1 + 1] = BRIDGE
    return np.array_equal(main(grid), expected)


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.choice([5, 6])
    orientation = random.choice(["H", "V"])
    h_bridge, h_negative, *others = random.sample([0, 1, 2, 3], 4)
    hole_counts = [h_bridge, h_bridge, h_negative, h_negative]
    hole_counts.extend(others[: n_objects - 4])

    colors = random.sample(INPUT_COLORS, n_objects)
    canvas = np.zeros((CANVAS, CANVAS), dtype=int)
    used: list[tuple[int, int, int, int]] = []
    specs: list[dict] = []

    if orientation == "H":
        gap_pos = random.randint(2, 5)
        offset_pos = random.randint(0, 2)
        rp = random.randint(1, 5)
        cp = random.randint(1, 9)
        pos_left = (rp, cp)
        pos_right = (rp + offset_pos, cp + PATCH + gap_pos)

        gap_neg = random.randint(1, 4)
        rn = random.randint(13, 17)
        cn = random.randint(1, 9)
        neg_left = (rn, cn)
        neg_right = (rn + 3, cn + PATCH + gap_neg)

        pair_boxes = [
            (pos_left[0], pos_left[1], pos_left[0] + 4, pos_left[1] + 4),
            (pos_right[0], pos_right[1], pos_right[0] + 4, pos_right[1] + 4),
            (neg_left[0], neg_left[1], neg_left[0] + 4, neg_left[1] + 4),
            (neg_right[0], neg_right[1], neg_right[0] + 4, neg_right[1] + 4),
        ]
        if any(box[2] >= CANVAS or box[3] >= CANVAS for box in pair_boxes):
            return None
        if any(_boxes_overlap(pair_boxes[i], pair_boxes[j], padding=1) for i in range(4) for j in range(i + 1, 4)):
            return None

        mid = _shared_mid_for_horizontal(pos_left[0], pos_right[0])
        positive_required = [
            [(mid - pos_left[0], PATCH - 1)],
            [(mid - pos_right[0], 0)],
        ]
        pair_positions = [pos_left, pos_right, neg_left, neg_right]
    else:
        gap_pos = random.randint(2, 5)
        offset_pos = random.randint(0, 2)
        rp = random.randint(1, 9)
        cp = random.randint(1, 5)
        pos_top = (rp, cp)
        pos_bottom = (rp + PATCH + gap_pos, cp + offset_pos)

        gap_neg = random.randint(1, 4)
        rn = random.randint(1, 9)
        cn = random.randint(13, 17)
        neg_top = (rn, cn)
        neg_bottom = (rn + PATCH + gap_neg, cn + 3)

        pair_boxes = [
            (pos_top[0], pos_top[1], pos_top[0] + 4, pos_top[1] + 4),
            (pos_bottom[0], pos_bottom[1], pos_bottom[0] + 4, pos_bottom[1] + 4),
            (neg_top[0], neg_top[1], neg_top[0] + 4, neg_top[1] + 4),
            (neg_bottom[0], neg_bottom[1], neg_bottom[0] + 4, neg_bottom[1] + 4),
        ]
        if any(box[2] >= CANVAS or box[3] >= CANVAS for box in pair_boxes):
            return None
        if any(_boxes_overlap(pair_boxes[i], pair_boxes[j], padding=1) for i in range(4) for j in range(i + 1, 4)):
            return None

        mid = _shared_mid_for_vertical(pos_top[1], pos_bottom[1])
        positive_required = [
            [(PATCH - 1, mid - pos_top[1])],
            [(0, mid - pos_bottom[1])],
        ]
        pair_positions = [pos_top, pos_bottom, neg_top, neg_bottom]

    for idx in range(4):
        used.append(pair_boxes[idx])
        specs.append(
            {
                "hole_count": hole_counts[idx],
                "pos": pair_positions[idx],
                "color": colors[idx],
                "required": positive_required[idx] if idx < 2 else None,
            }
        )

    for idx in range(4, n_objects):
        pos = _place_random_box(used)
        if pos is None:
            return None
        box = (pos[0], pos[1], pos[0] + 4, pos[1] + 4)
        used.append(box)
        specs.append({"hole_count": hole_counts[idx], "pos": pos, "color": colors[idx], "required": None})

    random.shuffle(specs)
    for spec in specs:
        mask = _completed_mask(spec["hole_count"])
        removed = _choose_removed(mask, spec["required"])
        if removed is None:
            return None
        patch = _patch_from_mask(mask, spec["color"], removed)
        r0, c0 = spec["pos"]
        canvas[r0 : r0 + PATCH, c0 : c0 + PATCH] = patch

    cropped = _crop_canvas(canvas)
    if _validate_generated(cropped):
        return cropped
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
