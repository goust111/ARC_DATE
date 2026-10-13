from __future__ import annotations

from collections import Counter, defaultdict, deque
from itertools import combinations
import random

import numpy as np


BG = 0
SOURCE_COLORS = (1, 3, 5, 7, 8, 9)
MARKER_COLORS = (2, 4, 6)
CORNERS = ("TL", "TR", "BR", "BL")
NEXT_CORNER = {"TL": "TR", "TR": "BR", "BR": "BL", "BL": "TL"}


TL_TEMPLATES = [
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1], [1, 0], [1, 0]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 0]], dtype=bool),
    np.array([[1, 1, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=bool),
]


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox_from_coords(coords),
                    "area": len(coords),
                }
            )
    return out


def bbox_corner(bbox: tuple[int, int, int, int], h: int, w: int) -> str | None:
    r0, c0, r1, c1 = bbox
    hits = []
    if r0 == 0 and c0 == 0:
        hits.append("TL")
    if r0 == 0 and c1 == w - 1:
        hits.append("TR")
    if r1 == h - 1 and c1 == w - 1:
        hits.append("BR")
    if r1 == h - 1 and c0 == 0:
        hits.append("BL")
    return hits[0] if len(hits) == 1 else None


def corner_anchor(corner: str, h: int, w: int, obj_h: int, obj_w: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, w - obj_w
    if corner == "BR":
        return h - obj_h, w - obj_w
    return h - obj_h, 0


def shift_to_corner(coords: list[tuple[int, int]], bbox: tuple[int, int, int, int], dst_corner: str, h: int, w: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    obj_h = r1 - r0 + 1
    obj_w = c1 - c0 + 1
    dr0, dc0 = corner_anchor(dst_corner, h, w, obj_h, obj_w)
    return [(dr0 + (r - r0), dc0 + (c - c0)) for r, c in coords]


def rectangle_stamp(bbox: tuple[int, int, int, int], dst_corner: str, h: int, w: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    obj_h = r1 - r0 + 1
    obj_w = c1 - c0 + 1
    dr0, dc0 = corner_anchor(dst_corner, h, w, obj_h, obj_w)
    return [(r, c) for r in range(dr0, dr0 + obj_h) for c in range(dc0, dc0 + obj_w)]


def adjacency_counts(stamp: list[tuple[int, int]], markers_by_color: dict[int, set[tuple[int, int]]], shape: tuple[int, int]) -> dict[int, int]:
    counts = {}
    for color, cells in markers_by_color.items():
        total = 0
        for r, c in stamp:
            for nr, nc in neighbors4(r, c):
                if in_bounds(shape, nr, nc) and (nr, nc) in cells:
                    total += 1
        counts[int(color)] = total
    return counts


def is_singleton_marker(comp: dict, h: int, w: int) -> bool:
    if int(comp["area"]) != 1:
        return False
    r, c = comp["coords"][0]
    return 0 < r < h - 1 and 0 < c < w - 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each true corner object clockwise, then recolor each copy by the marker color with the strongest edge contact at its destination.

    Input structure:
    - The grid is mostly black background with 2 or 3 monochrome 4-connected objects anchored in true corners.
    - A relevant corner object is a single-color component whose bounding box touches exactly one grid corner, hence two adjacent borders.
    - The grid also contains isolated single-pixel colored markers away from the border, plus optional non-corner distractor blobs.

    Rule:
    - Keep the input unchanged as a base.
    - For each corner object, align its unrotated pixel mask to the next clockwise corner.
    - Count, for that destination mask, how many shared edges it has with marker pixels of each marker color.
    - Recolor the copied mask with the unique marker color having the largest total shared-edge count; source objects, markers, and distractors otherwise stay in place.
    - If a destination corner already contains another source object, the recolored copy overwrites those destination cells.

    Key insight:
    - The copy location must be determined before the color can be known, because the winning color depends on edge contacts around the destination footprint rather than on the source object's original color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = get_bg(g)

    comps = find_components(g, bg)
    corner_objects: dict[str, dict] = {}
    markers_by_color: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for comp in comps:
        corner = bbox_corner(comp["bbox"], h, w)
        if corner is not None:
            corner_objects[corner] = comp
            continue
        if is_singleton_marker(comp, h, w):
            markers_by_color[int(comp["color"])].add(comp["coords"][0])

    out = g.copy()
    for src_corner, comp in corner_objects.items():
        dst_corner = NEXT_CORNER[src_corner]
        stamp = shift_to_corner(comp["coords"], comp["bbox"], dst_corner, h, w)
        counts = adjacency_counts(stamp, markers_by_color, g.shape)
        positive = {color: count for color, count in counts.items() if count > 0}
        assert len(positive) >= 2, "destination must see at least two marker colors"
        best = max(positive.values())
        winners = [color for color, count in positive.items() if count == best]
        assert len(winners) == 1, "marker majority must be unique"
        fill = winners[0]
        for r, c in stamp:
            out[r, c] = fill
    return out


def orient_template(template: np.ndarray, corner: str) -> np.ndarray:
    if corner == "TL":
        return template.copy()
    if corner == "TR":
        return np.fliplr(template)
    if corner == "BR":
        return np.flipud(np.fliplr(template))
    return np.flipud(template)


def canonical_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def place_mask(grid: np.ndarray, mask: np.ndarray, corner: str, color: int) -> tuple[int, int, int, int]:
    h, w = grid.shape
    mh, mw = mask.shape
    r0, c0 = corner_anchor(corner, h, w, mh, mw)
    for rr in range(mh):
        for cc in range(mw):
            if mask[rr, cc]:
                grid[r0 + rr, c0 + cc] = color
    return r0, c0, r0 + mh - 1, c0 + mw - 1


def component_from_mask(mask: np.ndarray, corner: str, color: int, grid_shape: tuple[int, int]) -> dict:
    h, w = grid_shape
    mh, mw = mask.shape
    r0, c0 = corner_anchor(corner, h, w, mh, mw)
    coords = [(r0 + rr, c0 + cc) for rr in range(mh) for cc in range(mw) if mask[rr, cc]]
    return {"color": color, "coords": coords, "bbox": (r0, c0, r0 + mh - 1, c0 + mw - 1), "area": len(coords)}


def bbox_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def can_place_blob(grid: np.ndarray, cells: list[tuple[int, int]], blocked: set[tuple[int, int]]) -> bool:
    h, w = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in blocked or int(grid[r, c]) != BG:
            return False
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in blocked and (nr, nc) not in cell_set:
                return False
    return True


def random_blob_cells(grid_shape: tuple[int, int], occupied: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
    h, w = grid_shape
    for _ in range(80):
        bh = random.randint(1, 3)
        bw = random.randint(2, 4)
        r0 = random.randint(2, h - bh - 3)
        c0 = random.randint(2, w - bw - 3)
        seed = (r0 + random.randrange(bh), c0 + random.randrange(bw))
        cells = {seed}
        target = random.randint(2, min(4, bh * bw))
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if r0 <= nr < r0 + bh and c0 <= nc < c0 + bw and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        cells_list = sorted(cells)
        if len(cells_list) < 2:
            continue
        rr = [r for r, _ in cells_list]
        cc = [c for _, c in cells_list]
        if min(rr) in (0, h - 1) or max(rr) in (0, h - 1) or min(cc) in (0, w - 1) or max(cc) in (0, w - 1):
            continue
        if can_place_blob(np.full(grid_shape, BG, dtype=int), cells_list, occupied):
            return cells_list
    return None


def marker_candidates(
    grid: np.ndarray,
    stamp: list[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    selected_markers: set[tuple[int, int]],
    rect_stamp: list[tuple[int, int]],
) -> list[dict]:
    stamp_set = set(stamp)
    rect_set = set(rect_stamp)
    h, w = grid.shape
    out = []
    seen = set()
    for r, c in stamp:
        for nr, nc in neighbors4(r, c):
            if not in_bounds(grid.shape, nr, nc):
                continue
            if nr in (0, h - 1) or nc in (0, w - 1):
                continue
            if (nr, nc) in seen or (nr, nc) in forbidden or (nr, nc) in selected_markers:
                continue
            if int(grid[nr, nc]) != BG:
                continue
            if any(int(grid[ar, ac]) != BG for ar, ac in neighbors4(nr, nc) if in_bounds(grid.shape, ar, ac)):
                continue
            if any(abs(nr - mr) + abs(nc - mc) == 1 for mr, mc in selected_markers):
                continue
            actual = sum((ar, ac) in stamp_set for ar, ac in neighbors4(nr, nc))
            rect = sum((ar, ac) in rect_set for ar, ac in neighbors4(nr, nc))
            if actual == 0:
                continue
            seen.add((nr, nc))
            out.append({"pos": (nr, nc), "actual": actual, "rect": rect})
    random.shuffle(out)
    return out


def subsets_for_candidates(candidates: list[dict]) -> list[dict]:
    out = []
    n = len(candidates)
    for size in (2, 3):
        for ids in combinations(range(n), size):
            pts = [candidates[i]["pos"] for i in ids]
            ok = True
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    if abs(pts[i][0] - pts[j][0]) + abs(pts[i][1] - pts[j][1]) <= 1:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            actual = sum(candidates[i]["actual"] for i in ids)
            rect = sum(candidates[i]["rect"] for i in ids)
            out.append(
                {
                    "ids": ids,
                    "positions": pts,
                    "actual": actual,
                    "rect": rect,
                    "pixels": len(ids),
                }
            )
    random.shuffle(out)
    return out


def choose_marker_plan(
    candidates: list[dict],
    colors: tuple[int, int],
    prefer_bbox_difference: bool,
) -> tuple[dict[int, list[tuple[int, int]]], bool, bool] | None:
    subsets = subsets_for_candidates(candidates)
    if len(subsets) < 2:
        return None
    for a in subsets:
        set_a = set(a["ids"])
        for b in subsets:
            if set_a & set(b["ids"]):
                continue
            if a["actual"] == b["actual"]:
                continue
            winner_color = colors[0] if a["actual"] > b["actual"] else colors[1]
            rect_winner = None
            if a["rect"] != b["rect"]:
                rect_winner = colors[0] if a["rect"] > b["rect"] else colors[1]
            bbox_diff = rect_winner is not None and rect_winner != winner_color
            pixel_shortcut_fails = (
                (a["actual"] > b["actual"] and a["pixels"] < b["pixels"])
                or (b["actual"] > a["actual"] and b["pixels"] < a["pixels"])
            )
            if prefer_bbox_difference and not bbox_diff:
                continue
            return {colors[0]: a["positions"], colors[1]: b["positions"]}, bbox_diff, pixel_shortcut_fails
    return None


def validate_input(grid: np.ndarray) -> dict:
    h, w = grid.shape
    comps = find_components(grid, BG)
    corner_objects = {}
    markers_by_color: dict[int, set[tuple[int, int]]] = defaultdict(set)
    distractors = []
    for comp in comps:
        corner = bbox_corner(comp["bbox"], h, w)
        if corner is not None:
            corner_objects[corner] = comp
        elif is_singleton_marker(comp, h, w):
            markers_by_color[int(comp["color"])].add(comp["coords"][0])
        else:
            distractors.append(comp)
    assert 2 <= len(corner_objects) <= 3
    assert all(len(v) >= 2 for v in markers_by_color.values())
    assert len(markers_by_color) >= 2
    assert all(int(comp["area"]) >= 2 for comp in distractors)

    bbox_sensitive = False
    source_color_changes = False
    pixel_shortcut_fails = False
    for src_corner, comp in corner_objects.items():
        dst_corner = NEXT_CORNER[src_corner]
        stamp = shift_to_corner(comp["coords"], comp["bbox"], dst_corner, h, w)
        rect = rectangle_stamp(comp["bbox"], dst_corner, h, w)
        counts = adjacency_counts(stamp, markers_by_color, grid.shape)
        positive = {color: count for color, count in counts.items() if count > 0}
        assert len(positive) >= 2
        values = list(positive.values())
        assert len(set(values)) == len(values)
        winner = max(positive, key=positive.get)
        if winner != int(comp["color"]):
            source_color_changes = True
        rect_counts = adjacency_counts(rect, markers_by_color, grid.shape)
        rect_positive = {color: count for color, count in rect_counts.items() if count > 0}
        rect_winner = max(rect_positive, key=rect_positive.get)
        if rect_winner != winner:
            bbox_sensitive = True
        marker_pixels = {
            color: sum(1 for r, c in cells if any((nr, nc) in set(stamp) for nr, nc in neighbors4(r, c)))
            for color, cells in markers_by_color.items()
        }
        for color, count in positive.items():
            for other_color, other_count in positive.items():
                if count > other_count and marker_pixels[color] < marker_pixels[other_color]:
                    pixel_shortcut_fails = True

    out = main(grid)
    assert not np.array_equal(out, grid)
    assert source_color_changes
    assert bbox_sensitive or pixel_shortcut_fails
    return {
        "corner_count": len(corner_objects),
        "marker_color_count": len(markers_by_color),
        "bbox_sensitive": bbox_sensitive,
        "pixel_shortcut_fails": pixel_shortcut_fails,
    }


def generate_input() -> np.ndarray:
    for _ in range(500):
        h = random.randint(12, 16)
        w = random.randint(12, 16)
        grid = np.zeros((h, w), dtype=int)

        corner_count = 3 if random.random() < 0.7 else 2
        chosen_corners = random.sample(CORNERS, corner_count)
        chosen_templates = random.sample(TL_TEMPLATES, corner_count)
        random.shuffle(chosen_templates)
        source_colors = random.sample(SOURCE_COLORS, corner_count)

        corner_boxes = []
        corner_objects: dict[str, dict] = {}
        occupied: set[tuple[int, int]] = set()
        signatures = set()
        for corner, base_template, color in zip(chosen_corners, chosen_templates, source_colors):
            mask = orient_template(base_template, corner)
            sig = canonical_signature(mask)
            if sig in signatures:
                break
            signatures.add(sig)
            comp = component_from_mask(mask, corner, color, grid.shape)
            corner_objects[corner] = comp
            corner_boxes.append(comp["bbox"])
            place_mask(grid, mask, corner, color)
            occupied.update(comp["coords"])
        else:
            pass
        if len(corner_objects) != corner_count:
            continue

        bbox_forbidden = set()
        for box in corner_boxes:
            bbox_forbidden |= bbox_cells(box)

        distractor_count = random.choices([0, 1, 2], weights=[1, 2, 2])[0]
        distractors_ok = True
        for _d in range(distractor_count):
            cells = random_blob_cells(grid.shape, occupied | bbox_forbidden)
            if cells is None:
                distractors_ok = False
                break
            color = random.choice(tuple(c for c in SOURCE_COLORS if c not in source_colors))
            for r, c in cells:
                grid[r, c] = color
            occupied.update(cells)
        if not distractors_ok:
            continue

        selected_markers: set[tuple[int, int]] = set()
        plan_by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
        pair_cycle = [(2, 4), (4, 6), (6, 2)]
        random.shuffle(pair_cycle)
        pairs = pair_cycle[:corner_count]
        bbox_diff_seen = False
        pixel_shortcut_seen = False

        failed = False
        for idx, (src_corner, comp) in enumerate(sorted(corner_objects.items())):
            dst_corner = NEXT_CORNER[src_corner]
            stamp = shift_to_corner(comp["coords"], comp["bbox"], dst_corner, h, w)
            rect = rectangle_stamp(comp["bbox"], dst_corner, h, w)
            candidates = marker_candidates(grid, stamp, bbox_forbidden | occupied, selected_markers, rect)
            if len(candidates) < 4:
                failed = True
                break
            prefer_bbox_difference = (idx == 0)
            plan = choose_marker_plan(candidates, pairs[idx], prefer_bbox_difference=prefer_bbox_difference)
            if plan is None:
                if prefer_bbox_difference:
                    plan = choose_marker_plan(candidates, pairs[idx], prefer_bbox_difference=False)
                if plan is None:
                    failed = True
                    break
            colored_positions, bbox_diff, pixel_shortcut_fails = plan
            bbox_diff_seen |= bbox_diff
            pixel_shortcut_seen |= pixel_shortcut_fails
            for color, pts in colored_positions.items():
                for r, c in pts:
                    grid[r, c] = color
                    selected_markers.add((r, c))
                    plan_by_color[color].append((r, c))
        if failed:
            continue

        if any(len(plan_by_color[color]) < 2 for color in plan_by_color):
            continue
        if len(plan_by_color) < 2:
            continue

        try:
            info = validate_input(grid)
        except AssertionError:
            continue
        if not (bbox_diff_seen or info["bbox_sensitive"] or pixel_shortcut_seen or info["pixel_shortcut_fails"]):
            continue
        return grid

    raise RuntimeError("failed to generate a valid non-degenerate ARC input")
