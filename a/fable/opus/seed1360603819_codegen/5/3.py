import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_RETRIES = 400
PAIR_RETRIES = 180
GRID_H_RANGE = (19, 28)
GRID_W_RANGE = (19, 28)
PAIR_COUNT_CHOICES = [2, 3, 4]
ARROW_LENGTH_RANGE = (3, 7)
GAP_AFTER_MOVE_RANGE = (1, 5)
INPUT_PADDING = 1

RECT_DIMS = [
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (2, 2),
    (2, 3),
    (3, 2),
]

IRREGULAR_SHAPES = [
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (-1, 1), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (1, 2), (0, 2), (2, 2)),
]


def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIRECTIONS:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            cells.sort()
            comps.append(
                {
                    "color": color,
                    "cells": cells,
                    "area": len(cells),
                    "bbox": _bbox(cells),
                }
            )
    return comps


def _detect_arrow(cells):
    r0, c0, r1, c1 = _bbox(cells)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True

    if h == 3 and w >= 3 and bool(mask[1].all()):
        top = np.flatnonzero(mask[0])
        bottom = np.flatnonzero(mask[2])
        if len(top) == 1 and len(bottom) == 1 and int(top[0]) == int(bottom[0]):
            wing_col = int(top[0])
            if wing_col == 0:
                return {"direction": (0, -1), "tip": (r0 + 1, c0)}
            if wing_col == w - 1:
                return {"direction": (0, 1), "tip": (r0 + 1, c1)}

    if w == 3 and h >= 3 and bool(mask[:, 1].all()):
        left = np.flatnonzero(mask[:, 0])
        right = np.flatnonzero(mask[:, 2])
        if len(left) == 1 and len(right) == 1 and int(left[0]) == int(right[0]):
            wing_row = int(left[0])
            if wing_row == 0:
                return {"direction": (-1, 0), "tip": (r0, c0 + 1)}
            if wing_row == h - 1:
                return {"direction": (1, 0), "tip": (r1, c0 + 1)}

    return None


def _first_hit(component_grid, start, direction, ignore_id):
    h, w = component_grid.shape
    r = int(start[0] + direction[0])
    c = int(start[1] + direction[1])
    while 0 <= r < h and 0 <= c < w:
        cid = int(component_grid[r, c])
        if cid and cid != ignore_id:
            return cid
        r += direction[0]
        c += direction[1]
    return None


def _analyze_grid(grid):
    comps = _find_components(grid)
    if not comps:
        return None

    component_grid = np.zeros_like(grid, dtype=int)
    components = {}
    arrows = []
    blocks = []
    for idx, comp in enumerate(comps, start=1):
        for r, c in comp["cells"]:
            component_grid[r, c] = idx
        entry = dict(comp)
        entry["id"] = idx
        components[idx] = entry
        arrow = _detect_arrow(comp["cells"])
        if arrow is None:
            blocks.append(entry)
        else:
            arrows.append({**entry, **arrow})

    if not arrows or len(arrows) != len(blocks):
        return None

    block_ids = {block["id"] for block in blocks}
    pairs = {}
    used_targets = set()
    for arrow in arrows:
        target_id = _first_hit(component_grid, arrow["tip"], arrow["direction"], arrow["id"])
        if target_id is None or target_id not in block_ids or target_id in used_targets:
            return None
        used_targets.add(target_id)
        pairs[arrow["id"]] = target_id

    if used_targets != block_ids:
        return None

    arrows.sort(key=lambda item: item["id"])
    return {
        "components": components,
        "arrows": arrows,
        "blocks": blocks,
        "pairs": pairs,
    }


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    analysis = _analyze_grid(g)
    if analysis is None:
        return g.copy()

    out = g.copy()
    h, w = g.shape
    for arrow in analysis["arrows"]:
        for r, c in arrow["cells"]:
            out[r, c] = BG

    for arrow in analysis["arrows"]:
        target = analysis["components"][analysis["pairs"][arrow["id"]]]
        dr, dc = arrow["direction"]
        steps = int(target["area"])
        for r, c in arrow["cells"]:
            nr = r + dr * steps
            nc = c + dc * steps
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = int(arrow["color"])

    return out


def _arrow_offsets(direction, length):
    if direction == (0, 1):
        return [(-1, 0), (1, 0)] + [(0, -k) for k in range(length)]
    if direction == (0, -1):
        return [(-1, 0), (1, 0)] + [(0, k) for k in range(length)]
    if direction == (1, 0):
        return [(0, -1), (0, 1)] + [(-k, 0) for k in range(length)]
    return [(0, -1), (0, 1)] + [(k, 0) for k in range(length)]


def _transform_shape(canonical_shape, direction):
    out = []
    for cross, along in canonical_shape:
        if direction == (0, 1):
            out.append((cross, along))
        elif direction == (0, -1):
            out.append((cross, -along))
        elif direction == (1, 0):
            out.append((along, cross))
        else:
            out.append((-along, cross))
    return out


def _shape_is_arrow(offsets):
    return _detect_arrow([(int(r), int(c)) for r, c in offsets]) is not None


def _random_block_offsets(rng, direction):
    for _ in range(40):
        if bool(rng.integers(0, 2)):
            h, w = RECT_DIMS[int(rng.integers(0, len(RECT_DIMS)))]
            anchor_row = int(rng.integers(0, h))
            canonical = [(r - anchor_row, c) for r in range(h) for c in range(w)]
        else:
            shape = IRREGULAR_SHAPES[int(rng.integers(0, len(IRREGULAR_SHAPES)))]
            flip = -1 if bool(rng.integers(0, 2)) else 1
            canonical = [(flip * cross, along) for cross, along in shape]
        offsets = _transform_shape(canonical, direction)
        if not _shape_is_arrow(offsets):
            return offsets
    raise RuntimeError("failed to sample non-arrow block")


def _offset_bounds(offsets):
    rows = [r for r, _ in offsets]
    cols = [c for _, c in offsets]
    return min(rows), min(cols), max(rows), max(cols)


def _cells_from_tip(tip, offsets):
    tr, tc = tip
    return {(tr + dr, tc + dc) for dr, dc in offsets}


def _expand_cells(cells, h, w, pad):
    out = set()
    for r, c in cells:
        for rr in range(max(0, r - pad), min(h - 1, r + pad) + 1):
            for cc in range(max(0, c - pad), min(w - 1, c + pad) + 1):
                out.add((rr, cc))
    return out


def _build_grid(h, w, placements):
    grid = np.zeros((h, w), dtype=int)
    for placement in placements:
        for r, c in placement["block_cells"]:
            grid[r, c] = int(placement["block_color"])
    for placement in placements:
        for r, c in placement["arrow_cells"]:
            grid[r, c] = int(placement["arrow_color"])
    return grid


def _analysis_matches_pairs(analysis, expected_pairs):
    if analysis is None or len(analysis["arrows"]) != len(expected_pairs):
        return False
    arrows_by_color = {int(arrow["color"]): arrow for arrow in analysis["arrows"]}
    colors_by_id = {cid: int(comp["color"]) for cid, comp in analysis["components"].items()}
    for arrow_color, block_color in expected_pairs:
        arrow = arrows_by_color.get(int(arrow_color))
        if arrow is None:
            return False
        target_id = analysis["pairs"].get(arrow["id"])
        if target_id is None or colors_by_id[target_id] != int(block_color):
            return False
    return True


def _gen_attempt(rng):
    h = int(rng.integers(*GRID_H_RANGE))
    w = int(rng.integers(*GRID_W_RANGE))
    pair_count = int(rng.choice(PAIR_COUNT_CHOICES))
    if pair_count * 2 > len(PALETTE):
        return None

    colors = list(rng.choice(PALETTE, size=pair_count * 2, replace=False))
    directions = [DIRECTIONS[i] for i in rng.permutation(len(DIRECTIONS))[:pair_count]]

    placements = []
    expected_pairs = []
    input_cells = set()
    input_forbidden = set()
    output_occupied = set()

    for idx in range(pair_count):
        arrow_color = int(colors[idx])
        block_color = int(colors[pair_count + idx])
        direction = directions[idx]
        placed = False

        for _ in range(PAIR_RETRIES):
            length = int(rng.integers(*ARROW_LENGTH_RANGE))
            block_offsets_from_anchor = _random_block_offsets(rng, direction)
            area = len(block_offsets_from_anchor)
            gap_after_move = int(rng.integers(*GAP_AFTER_MOVE_RANGE))
            distance_to_block = area + gap_after_move

            arrow_offsets = _arrow_offsets(direction, length)
            block_offsets = [
                (direction[0] * distance_to_block + dr, direction[1] * distance_to_block + dc)
                for dr, dc in block_offsets_from_anchor
            ]
            moved_arrow_offsets = [
                (direction[0] * area + dr, direction[1] * area + dc)
                for dr, dc in arrow_offsets
            ]

            all_offsets = arrow_offsets + block_offsets + moved_arrow_offsets
            min_r, min_c, max_r, max_c = _offset_bounds(all_offsets)
            low_r = -min_r
            high_r = h - 1 - max_r
            low_c = -min_c
            high_c = w - 1 - max_c
            if low_r > high_r or low_c > high_c:
                continue

            tip = (
                int(rng.integers(low_r, high_r + 1)),
                int(rng.integers(low_c, high_c + 1)),
            )
            arrow_cells = _cells_from_tip(tip, arrow_offsets)
            block_cells = _cells_from_tip(tip, block_offsets)
            moved_arrow_cells = _cells_from_tip(tip, moved_arrow_offsets)

            if any(cell in input_forbidden for cell in arrow_cells | block_cells):
                continue
            if (block_cells | moved_arrow_cells) & output_occupied:
                continue

            candidate = {
                "arrow_color": arrow_color,
                "block_color": block_color,
                "direction": direction,
                "arrow_cells": arrow_cells,
                "block_cells": block_cells,
                "moved_arrow_cells": moved_arrow_cells,
            }

            test_placements = placements + [candidate]
            test_pairs = expected_pairs + [(arrow_color, block_color)]
            test_grid = _build_grid(h, w, test_placements)
            analysis = _analyze_grid(test_grid)
            if not _analysis_matches_pairs(analysis, test_pairs):
                continue

            placements.append(candidate)
            expected_pairs.append((arrow_color, block_color))
            input_cells |= arrow_cells | block_cells
            input_forbidden |= _expand_cells(arrow_cells | block_cells, h, w, INPUT_PADDING)
            output_occupied |= block_cells | moved_arrow_cells
            placed = True
            break

        if not placed:
            return None

    grid = _build_grid(h, w, placements)
    analysis = _analyze_grid(grid)
    if not _analysis_matches_pairs(analysis, expected_pairs):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
