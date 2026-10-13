import random
from collections import Counter, deque

import numpy as np


BG = 0
FRAME = 5
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_MAX_ATTEMPTS = 500
GEN_CANVAS_RANGE = (18, 26)
GEN_PIECE_COUNT_CHOICES = (3, 4)
GEN_INTERIOR_SHAPES = {
    3: ((4, 5), (5, 4), (5, 5), (4, 6), (6, 4)),
    4: ((4, 4), (4, 5), (5, 4), (4, 6), (6, 4)),
}
GEN_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    counts = np.bincount(values, minlength=12)
    return int(np.argmax(counts))


def _bbox_from_coords(coords) -> tuple[int, int, int, int]:
    pts = np.asarray(list(coords), dtype=int)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _orientation_variants(rel_cells) -> list[dict]:
    rel = np.asarray(rel_cells, dtype=int)
    seen: dict[tuple[tuple[int, int], ...], dict] = {}
    for reflected in (False, True):
        base = rel.copy()
        if reflected:
            base[:, 1] *= -1
        for rot in range(4):
            pts = base.copy()
            for _ in range(rot):
                pts = np.column_stack((-pts[:, 1], pts[:, 0]))
            key = tuple(sorted((int(r), int(c)) for r, c in pts.tolist()))
            if key not in seen:
                seen[key] = {
                    "cells": key,
                    "reflected": bool(reflected),
                    "rot": int(rot),
                }
    return list(seen.values())


def _enumerate_piece_placements(
    piece: dict,
    interior_bbox: tuple[int, int, int, int],
    fixed_anchor: tuple[int, int] | None,
) -> list[dict]:
    r0, c0, r1, c1 = interior_bbox
    placements: dict[tuple[tuple[tuple[int, int], ...], tuple[int, int]], dict] = {}
    for variant in _orientation_variants(piece["rel_cells"]):
        rel_cells = np.asarray(variant["cells"], dtype=int)
        min_r, min_c = rel_cells.min(axis=0)
        max_r, max_c = rel_cells.max(axis=0)
        if fixed_anchor is None:
            anchor_rows = range(r0 - min_r, r1 - max_r + 1)
            anchor_cols = range(c0 - min_c, c1 - max_c + 1)
            anchors = ((ar, ac) for ar in anchor_rows for ac in anchor_cols)
        else:
            anchors = [tuple(map(int, fixed_anchor))]
        for ar, ac in anchors:
            abs_cells = tuple(sorted((int(ar + dr), int(ac + dc)) for dr, dc in rel_cells.tolist()))
            rows = [cell[0] for cell in abs_cells]
            cols = [cell[1] for cell in abs_cells]
            if min(rows) < r0 or min(cols) < c0:
                continue
            if max(rows) > r1 or max(cols) > c1:
                continue
            if len(set(abs_cells)) != len(abs_cells):
                continue
            key = (abs_cells, (int(ar), int(ac)))
            placements[key] = {
                "cells": abs_cells,
                "cell_set": frozenset(abs_cells),
                "anchor": (int(ar), int(ac)),
                "reflected": variant["reflected"],
                "rot": variant["rot"],
            }
    return list(placements.values())


def _search_solutions(
    placements_by_piece: list[list[dict]],
    piece_areas: list[int],
    target_cells: frozenset[tuple[int, int]],
    limit: int = 2,
):
    order = sorted(range(len(placements_by_piece)), key=lambda i: (len(placements_by_piece[i]), -piece_areas[i], i))
    chosen = [None] * len(placements_by_piece)
    occupied: set[tuple[int, int]] = set()
    remaining_area = [0] * (len(order) + 1)
    for k in range(len(order) - 1, -1, -1):
        remaining_area[k] = remaining_area[k + 1] + piece_areas[order[k]]

    solutions = []

    def dfs(k: int) -> None:
        if len(solutions) >= limit:
            return
        if k == len(order):
            if frozenset(occupied) == target_cells:
                solutions.append(list(chosen))
            return
        idx = order[k]
        need_after = remaining_area[k + 1]
        for placement in placements_by_piece[idx]:
            cells = placement["cells"]
            cell_set = placement["cell_set"]
            if occupied & cell_set:
                continue
            covered = len(occupied) + len(cells)
            if covered + need_after < len(target_cells):
                continue
            occupied.update(cells)
            chosen[idx] = placement
            dfs(k + 1)
            for cell in cells:
                occupied.remove(cell)
            chosen[idx] = None

    dfs(0)
    return solutions


def _frame_and_anchors(grid: np.ndarray):
    frame_mask = grid == FRAME
    assert frame_mask.any(), "expected a grey frame"
    bbox = _bbox_from_coords(np.argwhere(frame_mask))
    r0, c0, r1, c1 = bbox
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose a non-empty interior"
    expected = np.zeros_like(frame_mask)
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    assert np.array_equal(frame_mask, expected), "grey pixels must form a single-cell rectangular frame"

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    anchor_coords = np.argwhere(interior != BG)
    assert len(anchor_coords) > 0, "expected anchor pixels inside the frame"
    anchor_map = {}
    for rr, cc in anchor_coords:
        color = int(interior[rr, cc])
        assert color != FRAME, "frame color cannot appear inside the interior"
        abs_pos = (int(r0 + 1 + rr), int(c0 + 1 + cc))
        assert color not in anchor_map, "anchor colors must be unique"
        anchor_map[color] = abs_pos
    return bbox, anchor_map


def _extract_pieces(grid: np.ndarray, frame_bbox, anchor_map) -> list[dict]:
    r0, c0, r1, c1 = frame_bbox
    outside_mask = grid != BG
    outside_mask[r0 : r1 + 1, c0 : c1 + 1] = False
    comps = _components(outside_mask)
    assert comps, "expected fragments outside the frame"
    pieces = []
    used_controls = set()
    for comp in comps:
        colors = Counter(int(grid[r, c]) for r, c in comp)
        assert len(colors) == 2, "each fragment must contain one body color plus one control color"
        control_candidates = [color for color, cnt in colors.items() if cnt == 1 and color in anchor_map]
        assert len(control_candidates) == 1, "fragment control color must match exactly one anchor color"
        control_color = int(control_candidates[0])
        body_colors = [color for color in colors if color != control_color]
        assert len(body_colors) == 1
        body_color = int(body_colors[0])
        control_cell = next((r, c) for r, c in comp if int(grid[r, c]) == control_color)
        rel_cells = tuple(sorted((r - control_cell[0], c - control_cell[1]) for r, c in comp))
        assert control_color not in used_controls, "control colors must be one-to-one with anchors"
        used_controls.add(control_color)
        pieces.append(
            {
                "body_color": body_color,
                "control_color": control_color,
                "rel_cells": rel_cells,
                "area": len(comp),
            }
        )
    assert used_controls == set(anchor_map), "anchor and fragment counts must match"
    return pieces


def _solve_from_parts(frame_bbox, anchor_map, pieces, require_unique: bool = False):
    r0, c0, r1, c1 = frame_bbox
    interior_bbox = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
    target_cells = frozenset((r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1))
    placements_by_piece = []
    for piece in pieces:
        anchor = anchor_map[piece["control_color"]]
        placements = _enumerate_piece_placements(piece, interior_bbox, anchor)
        assert placements, "every piece must have at least one anchor-respecting placement"
        placements_by_piece.append(placements)
    solutions = _search_solutions(placements_by_piece, [p["area"] for p in pieces], target_cells, limit=2)
    assert solutions, "expected at least one valid anchored assembly"
    if require_unique:
        assert len(solutions) == 1, "assembly must be unique"
    return placements_by_piece, solutions


def _build_output(shape, frame_bbox, pieces, solution) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    r0, c0, r1, c1 = frame_bbox
    out[r0, c0 : c1 + 1] = FRAME
    out[r1, c0 : c1 + 1] = FRAME
    out[r0 : r1 + 1, c0] = FRAME
    out[r0 : r1 + 1, c1] = FRAME
    for piece, placement in zip(pieces, solution):
        for r, c in placement["cells"]:
            out[r, c] = piece["body_color"]
        ar, ac = placement["anchor"]
        out[ar, ac] = piece["control_color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the outside fragments into the hollow frame so each control pixel lands on its same-color anchor.

    Input structure:
    - Black background contains one grey single-cell rectangular frame with a black interior.
    - Inside the frame are several distinct-colored single-pixel anchors.
    - Outside the frame are equally many 4-connected monochrome fragments, each containing exactly one differently colored control pixel.

    Rule:
    - Match each fragment to the anchor of the same control color.
    - Rotate and/or mirror each fragment so its control pixel coincides with that anchor.
    - Choose the unique non-overlapping anchored assembly that exactly fills the frame interior; keep the frame, keep fragment body colors and control colors, and clear the original outside fragments.

    Key insight:
    - The anchor colors do not just identify the pieces; they pin one exact cell of each piece, and the remaining orientation is forced by the exact-fill tiling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    frame_bbox, anchor_map = _frame_and_anchors(g)
    pieces = _extract_pieces(g, frame_bbox, anchor_map)
    _placements, solutions = _solve_from_parts(frame_bbox, anchor_map, pieces, require_unique=True)
    return _build_output(g.shape, frame_bbox, pieces, solutions[0])


def _sample_piece_sizes(total_area: int, piece_count: int, rng: random.Random) -> list[int] | None:
    min_area = 3
    for _ in range(80):
        sizes = [min_area] * piece_count
        remaining = total_area - min_area * piece_count
        for _ in range(remaining):
            sizes[rng.randrange(piece_count)] += 1
        if max(sizes) <= 10 and min(sizes) >= 3 and len(set(sizes)) >= 2:
            rng.shuffle(sizes)
            return sizes
    return None


def _grow_partition(height: int, width: int, sizes: list[int], rng: random.Random):
    cells = [(r, c) for r in range(height) for c in range(width)]
    piece_count = len(sizes)
    for _ in range(120):
        seeds = rng.sample(cells, piece_count)
        pieces = [set([seed]) for seed in seeds]
        unassigned = set(cells) - set(seeds)
        ok = True
        while unassigned:
            choices = []
            for idx, piece in enumerate(pieces):
                if len(piece) >= sizes[idx]:
                    continue
                frontier = set()
                for r, c in piece:
                    for dr, dc in DIR4:
                        nr, nc = r + dr, c + dc
                        if (nr, nc) in unassigned:
                            frontier.add((nr, nc))
                if frontier:
                    choices.append((idx, list(frontier)))
            if not choices:
                ok = False
                break
            weights = [sizes[idx] - len(pieces[idx]) for idx, _ in choices]
            idx, frontier = rng.choices(choices, weights=weights, k=1)[0]
            cell = rng.choice(frontier)
            pieces[idx].add(cell)
            unassigned.remove(cell)
        if ok and all(len(piece) == size for piece, size in zip(pieces, sizes)):
            return [tuple(sorted(piece)) for piece in pieces]
    return None


def _choose_anchor_cell(piece_cells, rng: random.Random) -> tuple[int, int]:
    piece_set = set(piece_cells)
    preferred = []
    for r, c in piece_cells:
        degree = sum((r + dr, c + dc) in piece_set for dr, dc in DIR4)
        if degree >= 2:
            preferred.append((r, c))
    pool = preferred if preferred else list(piece_cells)
    return tuple(map(int, rng.choice(pool)))


def _relax_solution_count(pieces, interior_shape) -> int:
    height, width = interior_shape
    bbox = (0, 0, height - 1, width - 1)
    placements_by_piece = [_enumerate_piece_placements(piece, bbox, fixed_anchor=None) for piece in pieces]
    target = frozenset((r, c) for r in range(height) for c in range(width))
    solutions = _search_solutions(placements_by_piece, [p["area"] for p in pieces], target, limit=2)
    return len(solutions)


def _scatter_variants(pieces, rng: random.Random):
    variants_by_piece = []
    changed = False
    has_reflection = False
    has_rotation = False
    for piece in pieces:
        variants = _orientation_variants(piece["rel_cells"])
        variants_by_piece.append(variants)

    choices = []
    for variants in variants_by_piece:
        current = None
        for variant in variants:
            if not variant["reflected"] and variant["rot"] == 0:
                current = variant
                break
        assert current is not None
        options = [variant for variant in variants if variant["cells"] != current["cells"]]
        if options:
            pick = rng.choice(options)
            changed = True
        else:
            pick = current
        if pick["reflected"]:
            has_reflection = True
        if pick["rot"] != 0:
            has_rotation = True
        choices.append(pick)

    if not changed:
        return None

    if not has_reflection:
        for idx, variants in enumerate(variants_by_piece):
            reflected = [v for v in variants if v["reflected"] and v["cells"] != choices[idx]["cells"]]
            if reflected:
                choices[idx] = rng.choice(reflected)
                has_reflection = True
                break
    if not has_rotation:
        for idx, variants in enumerate(variants_by_piece):
            rotated = [v for v in variants if v["rot"] != 0 and v["cells"] != choices[idx]["cells"]]
            if rotated:
                choices[idx] = rng.choice(rotated)
                has_rotation = True
                break
    return choices


def _place_scattered_pieces(
    canvas_shape,
    frame_bbox,
    pieces,
    chosen_variants,
    rng: random.Random,
):
    H, W = canvas_shape
    grid = np.full((H, W), BG, dtype=int)
    r0, c0, r1, c1 = frame_bbox
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME

    blocked = np.zeros((H, W), dtype=bool)
    br0 = max(0, r0 - 1)
    bc0 = max(0, c0 - 1)
    br1 = min(H - 1, r1 + 1)
    bc1 = min(W - 1, c1 + 1)
    blocked[br0 : br1 + 1, bc0 : bc1 + 1] = True

    for piece, variant in zip(pieces, chosen_variants):
        rel = np.asarray(variant["cells"], dtype=int)
        min_r, min_c = rel.min(axis=0)
        norm = rel - np.array([min_r, min_c], dtype=int)
        control = tuple(map(int, (-min_r, -min_c)))
        h = int(norm[:, 0].max()) + 1
        w = int(norm[:, 1].max()) + 1
        placed = False
        for _ in range(250):
            top = rng.randint(1, H - h - 2)
            left = rng.randint(1, W - w - 2)
            if np.any(blocked[max(0, top - 1) : min(H, top + h + 1), max(0, left - 1) : min(W, left + w + 1)]):
                continue
            abs_cells = [(top + int(rr), left + int(cc)) for rr, cc in norm.tolist()]
            if any(blocked[r, c] for r, c in abs_cells):
                continue
            for r, c in abs_cells:
                grid[r, c] = piece["body_color"]
            cr, cc = top + control[0], left + control[1]
            grid[cr, cc] = piece["control_color"]
            blocked[max(0, top - 1) : min(H, top + h + 1), max(0, left - 1) : min(W, left + w + 1)] = True
            placed = True
            break
        if not placed:
            return None
    return grid


def _gen_attempt(rng: random.Random):
    piece_count = rng.choice(GEN_PIECE_COUNT_CHOICES)
    interior_h, interior_w = rng.choice(GEN_INTERIOR_SHAPES[piece_count])
    total_area = interior_h * interior_w
    sizes = _sample_piece_sizes(total_area, piece_count, rng)
    if sizes is None:
        return None
    partition = _grow_partition(interior_h, interior_w, sizes, rng)
    if partition is None:
        return None

    palette = list(rng.sample(GEN_COLORS, 2 * piece_count))
    anchor_colors = palette[:piece_count]
    body_colors = palette[piece_count:]

    pieces = []
    anchor_map = {}
    solved_cells_by_color = {}
    for idx, piece_cells in enumerate(partition):
        anchor_cell = _choose_anchor_cell(piece_cells, rng)
        rel_cells = tuple(sorted((r - anchor_cell[0], c - anchor_cell[1]) for r, c in piece_cells))
        piece = {
            "body_color": int(body_colors[idx]),
            "control_color": int(anchor_colors[idx]),
            "rel_cells": rel_cells,
            "area": len(piece_cells),
        }
        pieces.append(piece)
        anchor_map[piece["control_color"]] = tuple(map(int, anchor_cell))
        solved_cells_by_color[piece["control_color"]] = tuple(sorted((int(r), int(c)) for r, c in piece_cells))

    interior_bbox = (0, 0, interior_h - 1, interior_w - 1)
    anchored_placements, anchored_solutions = _solve_from_parts(( -1, -1, interior_h, interior_w), anchor_map, pieces, require_unique=False)
    if len(anchored_solutions) != 1:
        return None
    if not any(len(options) > 1 for options in anchored_placements):
        return None
    for piece, placement in zip(pieces, anchored_solutions[0]):
        if tuple(sorted(placement["cells"])) != solved_cells_by_color[piece["control_color"]]:
            return None

    if _relax_solution_count(pieces, (interior_h, interior_w)) < 2:
        return None

    chosen_variants = _scatter_variants(pieces, rng)
    if chosen_variants is None:
        return None

    frame_h = interior_h + 2
    frame_w = interior_w + 2
    canvas_h = rng.randint(max(GEN_CANVAS_RANGE[0], frame_h + 6), GEN_CANVAS_RANGE[1])
    canvas_w = rng.randint(max(GEN_CANVAS_RANGE[0], frame_w + 6), GEN_CANVAS_RANGE[1])
    frame_top = rng.randint(3, canvas_h - frame_h - 3)
    frame_left = rng.randint(3, canvas_w - frame_w - 3)
    frame_bbox = (frame_top, frame_left, frame_top + frame_h - 1, frame_left + frame_w - 1)

    grid = _place_scattered_pieces((canvas_h, canvas_w), frame_bbox, pieces, chosen_variants, rng)
    if grid is None:
        return None

    expected = np.full_like(grid, BG)
    expected[frame_top, frame_left : frame_left + frame_w] = FRAME
    expected[frame_top + frame_h - 1, frame_left : frame_left + frame_w] = FRAME
    expected[frame_top : frame_top + frame_h, frame_left] = FRAME
    expected[frame_top : frame_top + frame_h, frame_left + frame_w - 1] = FRAME

    for color, (ar, ac) in anchor_map.items():
        grid[frame_top + 1 + ar, frame_left + 1 + ac] = color

    absolute_anchor_map = {color: (frame_top + 1 + ar, frame_left + 1 + ac) for color, (ar, ac) in anchor_map.items()}
    absolute_solution = []
    for piece in pieces:
        abs_cells = tuple(
            sorted((frame_top + 1 + r, frame_left + 1 + c) for r, c in solved_cells_by_color[piece["control_color"]])
        )
        absolute_solution.append({"cells": abs_cells, "anchor": absolute_anchor_map[piece["control_color"]]})
        for r, c in abs_cells:
            expected[r, c] = piece["body_color"]
        ar, ac = absolute_anchor_map[piece["control_color"]]
        expected[ar, ac] = piece["control_color"]

    produced = main(grid)
    if not np.array_equal(produced, expected):
        return None

    parsed_frame, parsed_anchors = _frame_and_anchors(grid)
    parsed_pieces = _extract_pieces(grid, parsed_frame, parsed_anchors)
    parsed_placements, parsed_solutions = _solve_from_parts(parsed_frame, parsed_anchors, parsed_pieces, require_unique=False)
    if len(parsed_solutions) != 1:
        return None
    if not any(len(options) > 1 for options in parsed_placements):
        return None
    if _relax_solution_count(
        [
            {
                "body_color": piece["body_color"],
                "control_color": piece["control_color"],
                "rel_cells": piece["rel_cells"],
                "area": piece["area"],
            }
            for piece in parsed_pieces
        ],
        (interior_h, interior_w),
    ) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid anchored-fragment assembly input")
