import random
from collections import Counter, defaultdict

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
BASE_KEY_SHAPE = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 2),
    (2, 1),
    (2, 2),
)
GEN_RETRIES = 400
BOARD_SPAN = (9, 12)
TILE_COUNT_RANGE = (4, 7)
CANVAS_MARGIN_RANGE = (0, 3)
LEGEND_GAP = 2
LEGEND_SLOT_GAP = 1
MIN_ORIENTATION_COUNT = 3
ALL_FOUR_PROB = 0.7


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((int(r) - r0, int(c) - c0) for r, c in coords))


def _rotate_once(coords) -> tuple[tuple[int, int], ...]:
    rows = [int(r) for r, _ in coords]
    h = max(rows) + 1
    return tuple((int(c), h - 1 - int(r)) for r, c in coords)


def _all_rotations(coords) -> list[tuple[tuple[int, int], ...]]:
    out = []
    cur = tuple(coords)
    for _ in range(4):
        norm = _normalize(cur)
        if norm not in out:
            out.append(norm)
        cur = _rotate_once(cur)
    return out


ROTATIONS = _all_rotations(BASE_KEY_SHAPE)
TILE_AREA = len(BASE_KEY_SHAPE)


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(shape)
    return r1 - r0 + 1, c1 - c0 + 1


def _translate(shape: tuple[tuple[int, int], ...], r0: int, c0: int) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((r0 + r, c0 + c) for r, c in shape))


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _find_color_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for pts in _components_mask(grid == color):
            cells = tuple(sorted((int(r), int(c)) for r, c in pts.tolist()))
            comps.append(
                {
                    "color": color,
                    "coords": pts,
                    "cells": cells,
                    "bbox": _bbox(cells),
                    "area": len(cells),
                }
            )
    return comps


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _cover_solutions(
    mask_cells: set[tuple[int, int]],
    colored_shapes: list[tuple[int, tuple[tuple[int, int], ...]]],
    *,
    limit: int = 2,
) -> list[list[dict]]:
    cells = set(mask_cells)
    assert cells
    r0, c0, r1, c1 = _bbox(cells)

    candidates = []
    cell_to_ids: dict[tuple[int, int], list[int]] = defaultdict(list)
    for color, shape in colored_shapes:
        h, w = _shape_dims(shape)
        for ar in range(r0, r1 - h + 2):
            for ac in range(c0, c1 - w + 2):
                placed = _translate(shape, ar, ac)
                if all(cell in cells for cell in placed):
                    cand = {"color": int(color), "cells": placed}
                    idx = len(candidates)
                    candidates.append(cand)
                    for cell in placed:
                        cell_to_ids[cell].append(idx)

    if any(cell not in cell_to_ids for cell in cells):
        return []

    solutions: list[list[dict]] = []
    remaining = set(cells)
    chosen: list[int] = []

    def rec() -> None:
        if len(solutions) >= limit:
            return
        if not remaining:
            solutions.append([candidates[i] for i in chosen])
            return

        best_cell = None
        best_options = None
        for cell in remaining:
            options = [idx for idx in cell_to_ids[cell] if all(c in remaining for c in candidates[idx]["cells"])]
            if not options:
                return
            if best_options is None or len(options) < len(best_options):
                best_cell = cell
                best_options = options
                if len(best_options) == 1:
                    break

        assert best_cell is not None and best_options is not None
        for idx in best_options:
            chosen.append(idx)
            used_cells = candidates[idx]["cells"]
            remaining.difference_update(used_cells)
            rec()
            remaining.update(used_cells)
            chosen.pop()
            if len(solutions) >= limit:
                return

    rec()
    return solutions


def _parse_scene(grid: np.ndarray) -> tuple[dict, list[dict]]:
    bg = _get_bg(grid)
    comps = _find_color_components(grid, bg)
    assert len(comps) == 5, f"expected 5 non-background components, got {len(comps)}"

    areas = [comp["area"] for comp in comps]
    largest = max(range(len(comps)), key=lambda i: (areas[i], -comps[i]["bbox"][0], -comps[i]["bbox"][1]))
    mask_comp = comps[largest]
    legend_comps = [comp for i, comp in enumerate(comps) if i != largest]

    assert mask_comp["area"] > TILE_AREA, "mask must contain multiple key tiles"
    assert len(legend_comps) == 4
    assert all(comp["area"] == TILE_AREA for comp in legend_comps), "legend shapes must all match key area"

    legend_shapes = [_normalize(comp["cells"]) for comp in legend_comps]
    assert len(set(legend_shapes)) == 4, "legend must show four distinct rotations"
    assert set(legend_shapes) == set(ROTATIONS), "legend rotations do not match the key sprite"
    return mask_comp, legend_comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Partition the monochrome mask into rotated copies of one key sprite and recolor each copy from the legend.

    Input structure:
    - The grid has a dominant background color, one large monochrome foreground mask, and a separated legend in one corner.
    - The legend consists of four disconnected monochrome copies of the same asymmetric key shape, one for each 90-degree rotation, each painted a different color.

    Rule:
    - Ignore the legend positions and use only its shape-color associations.
    - Exactly cover the large mask with translated copies of the legend key shape, allowing any of the four rotations, with no overlap and no gaps.
    - Recolor each covered copy by the color of the legend entry with the same rotation, while leaving the legend and background unchanged.

    Key insight:
    - The mask is not segmented by color or outlines; only the unique global exact cover by the key polyomino reveals where each rotated copy starts and which legend color it should receive.
    """
    g = np.asarray(grid, dtype=int)
    mask_comp, legend_comps = _parse_scene(g)

    colored_shapes = [(comp["color"], _normalize(comp["cells"])) for comp in legend_comps]
    mask_cells = set(mask_comp["cells"])
    solutions = _cover_solutions(mask_cells, colored_shapes, limit=2)
    assert len(solutions) == 1, f"expected unique exact cover, found {len(solutions)}"

    out = g.copy()
    for placement in solutions[0]:
        rows = [r for r, _ in placement["cells"]]
        cols = [c for _, c in placement["cells"]]
        out[rows, cols] = placement["color"]
    return out


def _adjacent_to_occupied(cells: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    return any((r + dr, c + dc) in occupied for r, c in cells for dr, dc in DIR4)


def _edge_contacts(cells: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> int:
    return sum((r + dr, c + dc) in occupied for r, c in cells for dr, dc in DIR4)


def _orientation_multiset(tile_count: int) -> list[int]:
    all_four = tile_count >= 4 and random.random() < ALL_FOUR_PROB
    if all_four:
        picked = [0, 1, 2, 3]
    else:
        picked = random.sample(range(4), MIN_ORIENTATION_COUNT)
    while len(picked) < tile_count:
        picked.append(random.randrange(4))
    random.shuffle(picked)
    return picked


def _generate_cover() -> list[dict] | None:
    board_h = random.randrange(*BOARD_SPAN)
    board_w = random.randrange(*BOARD_SPAN)
    tile_count = random.randrange(*TILE_COUNT_RANGE)
    orientations = _orientation_multiset(tile_count)
    placements: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    bbox_overlap_pairs = 0

    def candidate_order(orient: int) -> list[tuple[int, int, set[tuple[int, int]], int, int]]:
        shape = ROTATIONS[orient]
        h, w = _shape_dims(shape)
        out = []
        for r0 in range(board_h - h + 1):
            for c0 in range(board_w - w + 1):
                cells = set(_translate(shape, r0, c0))
                if cells & occupied:
                    continue
                if occupied and not _adjacent_to_occupied(cells, occupied):
                    continue
                contacts = _edge_contacts(cells, occupied)
                overlaps = sum(_bbox_overlap(_bbox(cells), item["bbox"]) for item in placements)
                span_box = _bbox(tuple(occupied | cells)) if occupied else _bbox(cells)
                span_area = (span_box[2] - span_box[0] + 1) * (span_box[3] - span_box[1] + 1)
                out.append((contacts, overlaps, cells, span_area, r0 * board_w + c0))
        random.shuffle(out)
        out.sort(key=lambda item: (-item[1], -item[0], item[3], item[4]))
        return out

    def rec(idx: int) -> bool:
        nonlocal bbox_overlap_pairs
        if idx == tile_count:
            if bbox_overlap_pairs == 0:
                return False
            all_cells = occupied
            box = _bbox(all_cells)
            box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
            if box_area - len(all_cells) < tile_count:
                return False
            return True

        orient = orientations[idx]
        for contacts, overlaps, cells, _, _ in candidate_order(orient):
            if idx == tile_count - 1 and bbox_overlap_pairs + overlaps == 0:
                continue
            old_overlap = bbox_overlap_pairs
            bbox_overlap_pairs += overlaps
            occupied.update(cells)
            placements.append(
                {
                    "orientation": orient,
                    "cells": tuple(sorted(cells)),
                    "bbox": _bbox(cells),
                }
            )
            if rec(idx + 1):
                return True
            placements.pop()
            occupied.difference_update(cells)
            bbox_overlap_pairs = old_overlap
        return False

    first_orient = orientations[0]
    h, w = _shape_dims(ROTATIONS[first_orient])
    first_anchors = [
        (r0, c0)
        for r0 in range(max(1, board_h // 2 - 2), min(board_h - h + 1, board_h // 2 + 2))
        for c0 in range(max(1, board_w // 2 - 2), min(board_w - w + 1, board_w // 2 + 2))
    ]
    if not first_anchors:
        first_anchors = [(r0, c0) for r0 in range(board_h - h + 1) for c0 in range(board_w - w + 1)]
    random.shuffle(first_anchors)

    for r0, c0 in first_anchors:
        cells = set(_translate(ROTATIONS[first_orient], r0, c0))
        occupied = set(cells)
        bbox_overlap_pairs = 0
        placements[:] = [
            {
                "orientation": first_orient,
                "cells": tuple(sorted(cells)),
                "bbox": _bbox(cells),
            }
        ]
        if rec(1):
            return placements
    return None


def _crop_and_reanchor(placements: list[dict]) -> list[dict]:
    all_cells = [cell for item in placements for cell in item["cells"]]
    r0, c0, _, _ = _bbox(all_cells)
    out = []
    for item in placements:
        shifted = tuple(sorted((r - r0, c - c0) for r, c in item["cells"]))
        out.append({"orientation": item["orientation"], "cells": shifted, "bbox": _bbox(shifted)})
    return out


def _render_mask(placements: list[dict], color: int) -> np.ndarray:
    all_cells = [cell for item in placements for cell in item["cells"]]
    _, _, r1, c1 = _bbox(all_cells)
    grid = np.zeros((r1 + 1, c1 + 1), dtype=int)
    for item in placements:
        rows = [r for r, _ in item["cells"]]
        cols = [c for _, c in item["cells"]]
        grid[rows, cols] = color
    return grid


def _legend_layout_indices() -> list[tuple[int, int]]:
    slots = [(0, 0), (0, 1), (1, 0), (1, 1)]
    random.shuffle(slots)
    return slots


def _legend_bbox(corner: str, legend_h: int, legend_w: int, canvas_h: int, canvas_w: int) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, canvas_w - legend_w
    if corner == "bl":
        return canvas_h - legend_h, 0
    if corner == "br":
        return canvas_h - legend_h, canvas_w - legend_w
    raise ValueError(corner)


def _expanded_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _build_scene(placements: list[dict]) -> np.ndarray | None:
    bg = 0
    palette = list(range(1, 10))
    random.shuffle(palette)
    legend_colors = palette[:4]
    mask_color = palette[4]

    legend_specs = list(zip(legend_colors, ROTATIONS))
    random.shuffle(legend_specs)
    tile_h = max(_shape_dims(shape)[0] for _, shape in legend_specs)
    tile_w = max(_shape_dims(shape)[1] for _, shape in legend_specs)
    legend_h = 2 * tile_h + LEGEND_SLOT_GAP
    legend_w = 2 * tile_w + LEGEND_SLOT_GAP

    mask = _render_mask(placements, mask_color)
    mask_h, mask_w = mask.shape

    canvas_h = legend_h + mask_h + LEGEND_GAP + random.randrange(*CANVAS_MARGIN_RANGE) + random.randrange(*CANVAS_MARGIN_RANGE)
    canvas_w = legend_w + mask_w + LEGEND_GAP + random.randrange(*CANVAS_MARGIN_RANGE) + random.randrange(*CANVAS_MARGIN_RANGE)
    canvas_h = min(30, max(canvas_h, max(legend_h, mask_h) + LEGEND_GAP + 2))
    canvas_w = min(30, max(canvas_w, max(legend_w, mask_w) + LEGEND_GAP + 2))
    if canvas_h < max(legend_h, mask_h) or canvas_w < max(legend_w, mask_w):
        return None

    grid = np.full((canvas_h, canvas_w), bg, dtype=int)
    corner = random.choice(["tl", "tr", "bl", "br"])
    legend_r0, legend_c0 = _legend_bbox(corner, legend_h, legend_w, canvas_h, canvas_w)
    legend_box = (legend_r0, legend_c0, legend_r0 + legend_h - 1, legend_c0 + legend_w - 1)

    slots = _legend_layout_indices()
    for (color, shape), (sr, sc) in zip(legend_specs, slots):
        cell_r0 = legend_r0 + sr * (tile_h + LEGEND_SLOT_GAP)
        cell_c0 = legend_c0 + sc * (tile_w + LEGEND_SLOT_GAP)
        for r, c in shape:
            grid[cell_r0 + r, cell_c0 + c] = color

    possible = []
    for r0 in range(canvas_h - mask_h + 1):
        for c0 in range(canvas_w - mask_w + 1):
            box = (r0, c0, r0 + mask_h - 1, c0 + mask_w - 1)
            if _expanded_overlap(box, legend_box, LEGEND_GAP):
                continue
            possible.append((r0, c0))
    if not possible:
        return None

    mask_r0, mask_c0 = random.choice(possible)
    grid[mask_r0 : mask_r0 + mask_h, mask_c0 : mask_c0 + mask_w][mask != 0] = mask_color
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        mask_comp, legend_comps = _parse_scene(grid)
    except AssertionError:
        return False

    legend_shapes = [(comp["color"], _normalize(comp["cells"])) for comp in legend_comps]
    solutions = _cover_solutions(set(mask_comp["cells"]), legend_shapes, limit=2)
    if len(solutions) != 1:
        return False

    used_colors = {item["color"] for item in solutions[0]}
    if len(used_colors) < MIN_ORIENTATION_COUNT:
        return False

    legend_boxes = [comp["bbox"] for comp in legend_comps]
    if any(_bbox_overlap(a, b) for i, a in enumerate(legend_boxes) for b in legend_boxes[i + 1 :]):
        return False

    mask_box = mask_comp["bbox"]
    box_area = (mask_box[2] - mask_box[0] + 1) * (mask_box[3] - mask_box[1] + 1)
    if box_area - mask_comp["area"] < len(solutions[0]):
        return False

    expected = main(grid)
    out_colors = set(int(v) for v in np.unique(expected))
    if len(out_colors) < 5:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    placements = _generate_cover()
    if placements is None:
        return None
    placements = _crop_and_reanchor(placements)
    grid = _build_scene(placements)
    if grid is None:
        return None
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid unique-tiling key-cover input")
