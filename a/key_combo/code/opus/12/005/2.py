from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5
TEAL = 8

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

SHAPE_TEMPLATES = {
    "square": [((0, 0), (0, 1), (1, 0), (1, 1))],
    "bar": [((0, 0), (0, 1), (0, 2)), ((0, 0), (1, 0), (2, 0))],
}

PAIR_SQUARE_COUNTS = {
    (3, 4): (3, 0),
    (4, 5): (4, 1),
    (5, 6): (5, 2),
}

GEN_RETRIES = 500
COMPONENT_RETRIES = 220
PLACEMENT_RETRIES = 180


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def bbox_dims(coords: set[tuple[int, int]]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox(coords)
    return r1 - r0 + 1, c1 - c0 + 1


def normalize_coords(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0, c0, _, _ = bbox(coords)
    return {(r - r0, c - c0) for r, c in coords}


def components_of_color(grid: np.ndarray, color: int) -> list[set[tuple[int, int]]]:
    h, w = grid.shape
    seen: set[tuple[int, int]] = set()
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != color or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            comp = set()
            while stack:
                cr, cc = stack.pop()
                comp.add((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == color and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def primitive_kind(cells: frozenset[tuple[int, int]]) -> str:
    if len(cells) == 4:
        rs = {r for r, _ in cells}
        cs = {c for _, c in cells}
        assert len(rs) == 2 and len(cs) == 2
        return "square"
    if len(cells) == 3:
        rs = {r for r, _ in cells}
        cs = {c for _, c in cells}
        assert len(rs) == 1 or len(cs) == 1
        return "bar"
    raise AssertionError("invalid primitive size")


def candidate_primitives(component: set[tuple[int, int]]) -> list[dict]:
    comp = set(component)
    candidates: list[dict] = []
    seen: set[frozenset[tuple[int, int]]] = set()
    for r, c in comp:
        sq = frozenset(((r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)))
        hb = frozenset(((r, c), (r, c + 1), (r, c + 2)))
        vb = frozenset(((r, c), (r + 1, c), (r + 2, c)))
        for cells in (sq, hb, vb):
            if cells in seen or not cells.issubset(comp):
                continue
            seen.add(cells)
            candidates.append({"cells": cells, "kind": primitive_kind(cells)})
    return candidates


def unique_decomposition(component: set[tuple[int, int]]) -> list[dict]:
    comp = set(component)
    candidates = candidate_primitives(comp)
    by_cell: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, primitive in enumerate(candidates):
        for cell in primitive["cells"]:
            by_cell[cell].append(idx)

    solutions: list[list[int]] = []

    def search(uncovered: set[tuple[int, int]], chosen: list[int]) -> None:
        if len(solutions) > 1:
            return
        if not uncovered:
            solutions.append(chosen.copy())
            return

        best_cell = None
        best_options = None
        for cell in uncovered:
            options = [idx for idx in by_cell[cell] if candidates[idx]["cells"].issubset(uncovered)]
            if best_options is None or len(options) < len(best_options):
                best_cell = cell
                best_options = options
            if best_options == []:
                break

        assert best_cell is not None
        if not best_options:
            return

        for idx in best_options:
            chosen.append(idx)
            search(uncovered - set(candidates[idx]["cells"]), chosen)
            chosen.pop()
            if len(solutions) > 1:
                return

    search(set(comp), [])
    assert len(solutions) == 1, "component must admit a unique primitive decomposition"
    return [candidates[idx] for idx in solutions[0]]


def rotate_point(point: tuple[int, int], pivot: tuple[int, int], quarter_turns: int) -> tuple[int, int]:
    pr, pc = pivot
    dr = point[0] - pr
    dc = point[1] - pc
    for _ in range(quarter_turns % 4):
        dr, dc = dc, -dr
    return pr + dr, pc + dc


def rotate_cells(cells: set[tuple[int, int]] | frozenset[tuple[int, int]], pivot: tuple[int, int], quarter_turns: int) -> set[tuple[int, int]]:
    return {rotate_point(cell, pivot, quarter_turns) for cell in cells}


def _pivot_assignments(components: list[set[tuple[int, int]]], pivots: set[tuple[int, int]]) -> list[tuple[set[tuple[int, int]], tuple[int, int]]]:
    assignments: dict[int, tuple[int, int]] = {}
    for pivot in pivots:
        touched = []
        for idx, comp in enumerate(components):
            if any(nb in comp for nb in neighbors4(*pivot)):
                touched.append(idx)
        assert len(touched) == 1, "each pivot must be adjacent to exactly one grey composite"
        comp_idx = touched[0]
        assert comp_idx not in assignments, "each grey composite must have exactly one pivot"
        assignments[comp_idx] = pivot

    assert len(assignments) == len(components), "every grey composite needs one pivot"
    return [(components[idx], assignments[idx]) for idx in range(len(components))]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decompose each gray composite into primitive squares and bars, rotate the primitives around the red pivot by a count-driven angle, then recolor by shape.

    Input structure:
    - The grid has a black background with 2-3 gray 4-connected composites.
    - Every composite is made from touching 2x2 squares and 1x3 or 3x1 bars, all initially gray.
    - Each composite has exactly one red pivot cell outside the composite and 4-neighbor-adjacent to it.

    Rule:
    - For each gray composite, recover the unique non-overlapping decomposition into 2x2 squares and length-3 bars.
    - Let C be the number of recovered primitives, and rotate every primitive clockwise by 90*C degrees around that composite's red pivot.
    - Remove the original gray cells, keep the red pivot unchanged, and redraw rotated squares in teal and rotated bars in green.

    Key insight:
    - The rotation angle is not visible from the whole blob alone; it is encoded by how many primitive pieces appear after the gray composite is correctly decomposed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)

    grey_components = components_of_color(g, GREY)
    assert grey_components, "expected at least one grey composite"
    pivots = {tuple(map(int, rc)) for rc in np.argwhere(g == RED)}
    assert pivots, "expected red pivots"

    out = np.full(g.shape, bg, dtype=int)
    for pivot in pivots:
        out[pivot] = RED

    for component, pivot in _pivot_assignments(grey_components, pivots):
        primitives = unique_decomposition(component)
        turns = len(primitives) % 4
        for primitive in primitives:
            rotated = rotate_cells(primitive["cells"], pivot, turns)
            color = TEAL if primitive["kind"] == "square" else GREEN
            for r, c in rotated:
                assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1]
                assert out[r, c] == bg, "rotated primitives must stay in bounds and not overlap"
                out[r, c] = color

    return out


def _translate(template: tuple[tuple[int, int], ...], dr: int, dc: int) -> frozenset[tuple[int, int]]:
    return frozenset((r + dr, c + dc) for r, c in template)


def _adjacent_to_component(cells: frozenset[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    return any((r + dr, c + dc) in occupied for r, c in cells for dr, dc in DIR4)


def _piece_candidates(occupied: set[tuple[int, int]], piece_class: str) -> list[frozenset[tuple[int, int]]]:
    templates = SHAPE_TEMPLATES[piece_class]
    if not occupied:
        return [_translate(random.choice(templates), 0, 0)]

    min_r, min_c, max_r, max_c = bbox(occupied)
    candidates: list[frozenset[tuple[int, int]]] = []
    seen: set[frozenset[tuple[int, int]]] = set()
    for template in templates:
        for dr in range(min_r - 3, max_r + 4):
            for dc in range(min_c - 3, max_c + 4):
                cells = _translate(template, dr, dc)
                if cells in seen or cells & occupied:
                    continue
                if not _adjacent_to_component(cells, occupied):
                    continue
                union = occupied | set(cells)
                h, w = bbox_dims(union)
                if h > 8 or w > 8:
                    continue
                seen.add(cells)
                candidates.append(cells)
    random.shuffle(candidates)
    return candidates


def _transform_component(pieces: list[dict], rot: int, flip: bool) -> list[dict]:
    transformed = []
    for piece in pieces:
        coords = np.array(list(piece["cells"]), dtype=int)
        if flip:
            coords[:, 1] *= -1
        for _ in range(rot % 4):
            coords = np.column_stack((coords[:, 1], -coords[:, 0]))
        cells = {(int(r), int(c)) for r, c in coords}
        transformed.append({"kind": piece["kind"], "cells": cells})

    all_cells = set().union(*(piece["cells"] for piece in transformed))
    r0, c0, _, _ = bbox(all_cells)
    normalized = []
    for piece in transformed:
        normalized.append(
            {
                "kind": piece["kind"],
                "cells": {(r - r0, c - c0) for r, c in piece["cells"]},
            }
        )
    return normalized


def _component_shape_key(cells: set[tuple[int, int]]) -> tuple[bytes, tuple[int, int]]:
    norm = normalize_coords(cells)
    best = None
    arr = np.array(list(norm), dtype=int)
    for flip in (False, True):
        work = arr.copy()
        if flip:
            work[:, 1] *= -1
        for rot in range(4):
            coords = work.copy()
            for _ in range(rot):
                coords = np.column_stack((coords[:, 1], -coords[:, 0]))
            cells2 = {(int(r), int(c)) for r, c in coords}
            cells2 = normalize_coords(cells2)
            h, w = bbox_dims(cells2)
            mask = np.zeros((h, w), dtype=np.uint8)
            for r, c in cells2:
                mask[r, c] = 1
            key = (mask.tobytes(), mask.shape)
            if best is None or key < best:
                best = key
    assert best is not None
    return best


def _build_component(piece_count: int, square_count: int) -> dict | None:
    piece_classes = ["square"] * square_count + ["bar"] * (piece_count - square_count)
    for _ in range(COMPONENT_RETRIES):
        random.shuffle(piece_classes)
        occupied: set[tuple[int, int]] = set()
        pieces: list[dict] = []
        ok = True
        for piece_class in piece_classes:
            candidates = _piece_candidates(occupied, piece_class)
            if not candidates:
                ok = False
                break
            cells = random.choice(candidates[: min(40, len(candidates))])
            occupied |= set(cells)
            pieces.append({"kind": piece_class, "cells": set(cells)})
        if not ok:
            continue

        pieces = _transform_component(pieces, random.randrange(4), bool(random.getrandbits(1)))
        component_cells = set().union(*(piece["cells"] for piece in pieces))
        h, w = bbox_dims(component_cells)
        area = len(component_cells)
        if not (9 <= area <= 20 and h <= 8 and w <= 8):
            continue

        try:
            decomposition = unique_decomposition(component_cells)
        except AssertionError:
            continue
        if len(decomposition) != piece_count:
            continue

        pivot_candidates = []
        for cell in component_cells:
            for dr, dc in DIR4:
                pivot = (cell[0] + dr, cell[1] + dc)
                if pivot in component_cells:
                    continue
                if sum((nr, nc) in component_cells for nr, nc in neighbors4(*pivot)) == 1:
                    pivot_candidates.append(pivot)
        pivot_candidates = sorted(set(pivot_candidates))
        if len(pivot_candidates) < 2:
            continue

        piece_mix = {piece["kind"] for piece in pieces}
        return {
            "pieces": pieces,
            "cells": component_cells,
            "piece_count": piece_count,
            "square_count": square_count,
            "area": area,
            "shape_key": _component_shape_key(component_cells),
            "pivot_candidates": pivot_candidates,
            "rotated_pieces_by_pivot": {},
            "turns": piece_count % 4,
        }
    return None


def _choose_component_specs(num_components: int) -> list[tuple[int, int]]:
    pair = random.choice(list(PAIR_SQUARE_COUNTS))
    specs = [(pair[0], PAIR_SQUARE_COUNTS[pair][0]), (pair[1], PAIR_SQUARE_COUNTS[pair][1])]
    if num_components == 3:
        extra_count = random.choice([c for c in (3, 4, 5, 6) if c not in pair] or list(pair))
        extra_sq_max = min(extra_count, 20 - 3 * extra_count)
        extra_sq = random.randint(0, extra_sq_max)
        specs.append((extra_count, extra_sq))
    random.shuffle(specs)
    return specs


def _dilate(points: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out = set()
    for r, c in points:
        for dr, dc in DIR8 + [(0, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out.add((nr, nc))
    return out


def _candidate_global_shifts(component: dict, pivot_local: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    source_points = set(component["cells"]) | {pivot_local}
    rotated_points = set()
    for piece in component["pieces"]:
        rotated_points |= rotate_cells(piece["cells"], pivot_local, component["turns"])
    all_points = source_points | rotated_points
    rows = [r for r, _ in all_points]
    cols = [c for _, c in all_points]
    dr_min = -min(rows)
    dr_max = h - 1 - max(rows)
    dc_min = -min(cols)
    dc_max = w - 1 - max(cols)
    shifts = [(dr, dc) for dr in range(dr_min, dr_max + 1) for dc in range(dc_min, dc_max + 1)]
    random.shuffle(shifts)
    return shifts


def _place_components(components: list[dict], h: int, w: int) -> list[dict] | None:
    placed: list[dict] = []
    input_blocked: set[tuple[int, int]] = set()
    output_blocked: set[tuple[int, int]] = set()

    order = sorted(range(len(components)), key=lambda i: (len(components[i]["cells"]), components[i]["piece_count"]), reverse=True)

    def backtrack(pos: int) -> bool:
        nonlocal input_blocked, output_blocked
        if pos == len(order):
            return True

        idx = order[pos]
        comp = components[idx]
        pivot_choices = comp["pivot_candidates"][:]
        random.shuffle(pivot_choices)
        for pivot_local in pivot_choices:
            shifts = _candidate_global_shifts(comp, pivot_local, h, w)
            for shift_r, shift_c in shifts[: min(len(shifts), 160)]:
                global_pivot = (pivot_local[0] + shift_r, pivot_local[1] + shift_c)
                global_cells = {(r + shift_r, c + shift_c) for r, c in comp["cells"]}
                input_points = global_cells | {global_pivot}
                if input_points & input_blocked:
                    continue

                rotated_pieces = []
                rotated_union = set()
                for piece in comp["pieces"]:
                    shifted = {(r + shift_r, c + shift_c) for r, c in piece["cells"]}
                    rotated = rotate_cells(shifted, global_pivot, comp["turns"])
                    rotated_pieces.append({"kind": piece["kind"], "cells": rotated})
                    rotated_union |= rotated

                output_points = rotated_union | {global_pivot}
                if output_points & output_blocked:
                    continue

                placed_entry = {
                    **comp,
                    "pivot": global_pivot,
                    "cells": global_cells,
                    "pieces": [{"kind": piece["kind"], "cells": {(r + shift_r, c + shift_c) for r, c in piece["cells"]}} for piece in comp["pieces"]],
                    "rotated_pieces": rotated_pieces,
                }
                placed.append(placed_entry)
                prev_input = input_blocked
                prev_output = output_blocked
                input_blocked = input_blocked | _dilate(input_points, h, w)
                output_blocked = output_blocked | _dilate(output_points, h, w)
                if backtrack(pos + 1):
                    return True
                placed.pop()
                input_blocked = prev_input
                output_blocked = prev_output
        return False

    return placed if backtrack(0) else None


def _render_input(h: int, w: int, placed: list[dict]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for comp in placed:
        for r, c in comp["cells"]:
            grid[r, c] = GREY
        pr, pc = comp["pivot"]
        grid[pr, pc] = RED
    return grid


def _render_output(h: int, w: int, placed: list[dict]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for comp in placed:
        pr, pc = comp["pivot"]
        grid[pr, pc] = RED
    for comp in placed:
        for piece in comp["rotated_pieces"]:
            color = TEAL if piece["kind"] == "square" else GREEN
            for r, c in piece["cells"]:
                assert grid[r, c] == BLACK
                grid[r, c] = color
    return grid


def _semantic_checks(inp: np.ndarray, placed: list[dict]) -> bool:
    if set(map(int, np.unique(inp))) - {BLACK, RED, GREY}:
        return False
    if not (2 <= len(placed) <= 3):
        return False
    counts = [comp["piece_count"] for comp in placed]
    if len(set(counts)) < 2:
        return False
    if all(comp["turns"] == 0 for comp in placed):
        return False

    kinds = {piece["kind"] for comp in placed for piece in comp["pieces"]}
    if kinds != {"square", "bar"}:
        return False

    has_same_area_diff_count = False
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            if placed[i]["area"] == placed[j]["area"] and placed[i]["piece_count"] != placed[j]["piece_count"]:
                has_same_area_diff_count = True
    if not has_same_area_diff_count:
        return False

    for comp in placed:
        try:
            solved = unique_decomposition(comp["cells"])
        except AssertionError:
            return False
        if len(solved) != comp["piece_count"]:
            return False
    return True


def _attempt_generate() -> np.ndarray | None:
    num_components = 3 if random.random() < 0.65 else 2
    h = random.randint(17, 22) if num_components == 3 else random.randint(15, 20)
    w = random.randint(17, 22) if num_components == 3 else random.randint(15, 20)

    specs = _choose_component_specs(num_components)
    components = []
    seen_shapes = set()
    for piece_count, square_count in specs:
        comp = _build_component(piece_count, square_count)
        if comp is None:
            return None
        if comp["shape_key"] in seen_shapes:
            return None
        seen_shapes.add(comp["shape_key"])
        components.append(comp)

    for _ in range(PLACEMENT_RETRIES):
        placed = _place_components(components, h, w)
        if placed is None:
            continue
        inp = _render_input(h, w, placed)
        expected = _render_output(h, w, placed)
        if not _semantic_checks(inp, placed):
            continue
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid for puzzle 2")
