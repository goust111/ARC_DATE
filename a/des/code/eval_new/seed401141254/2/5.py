import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


BG = 0
GREY = 5
GAP = 1
MAX_GEN_ATTEMPTS = 400
MIN_VISIBLE = 2

Coord = Tuple[int, int]
Box = Tuple[int, int, int, int]


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _template_variants() -> List[np.ndarray]:
    bases = [
        np.array(
            [
                [1, 1, 1],
                [0, 1, 0],
                [0, 1, 0],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [1, 0, 0],
                [1, 0, 0],
                [1, 1, 1],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [1, 1, 0],
                [0, 1, 0],
                [0, 1, 1],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [0, 1, 0, 0],
                [1, 1, 1, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [1, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 1, 1],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [1, 1, 1, 0],
                [0, 1, 0, 0],
                [0, 1, 1, 0],
            ],
            dtype=bool,
        ),
    ]

    variants: List[np.ndarray] = []
    seen: Set[bytes] = set()
    for base in bases:
        for rot in range(4):
            turned = np.rot90(base, rot)
            for flip in (False, True):
                cand = np.fliplr(turned) if flip else turned
                cand = _normalize_mask(cand)
                key = str(cand.shape).encode() + cand.astype(np.uint8).tobytes()
                if key not in seen:
                    seen.add(key)
                    variants.append(cand)
    return variants


TEMPLATE_LIBRARY = _template_variants()


def _mask_coords(mask: np.ndarray) -> List[Coord]:
    return [tuple(map(int, rc)) for rc in np.argwhere(mask)]


def _stamp_cells(template_coords: Sequence[Coord], offset: Coord) -> Set[Coord]:
    orow, ocol = offset
    return {(orow + r, ocol + c) for r, c in template_coords}


def _reference_patch(template: np.ndarray) -> np.ndarray:
    th, tw = template.shape
    patch = np.full((th + 4, tw + 4), BG, dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    patch[2 : 2 + th, 2 : 2 + tw][template] = GREY
    return patch


def _find_reference_box(grid: np.ndarray) -> Box:
    h, w = grid.shape
    best: Optional[Tuple[Tuple[int, int, int, int], Box]] = None
    for r0 in range(h - 4):
        for c0 in range(w - 4):
            for r1 in range(r0 + 4, min(h, r0 + 10)):
                for c1 in range(c0 + 4, min(w, c0 + 10)):
                    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
                    border = np.zeros_like(patch, dtype=bool)
                    border[0, :] = True
                    border[-1, :] = True
                    border[:, 0] = True
                    border[:, -1] = True
                    if not np.all(patch[border] == GREY):
                        continue
                    inner = patch[1:-1, 1:-1]
                    if inner.size == 0:
                        continue
                    if not np.all((inner == BG) | (inner == GREY)):
                        continue
                    if np.count_nonzero(inner == GREY) < 3:
                        continue
                    if not np.any(inner == BG):
                        continue
                    score = (
                        int(r1 == h - 1 or c1 == w - 1),
                        (r1 - r0 + 1) * (c1 - c0 + 1),
                        r1 + c1,
                    )
                    if best is None or score > best[0]:
                        best = (score, (r0, c0, r1, c1))
    if best is None:
        raise AssertionError("reference box not found")
    return best[1]


def _extract_template(grid: np.ndarray, box: Box) -> np.ndarray:
    r0, c0, r1, c1 = box
    inner = grid[r0 + 1 : r1, c0 + 1 : c1] == GREY
    if np.count_nonzero(inner) < 3:
        raise AssertionError("reference template must contain grey cells")
    return _normalize_mask(inner)


def _candidate_offsets(
    grid_shape: Tuple[int, int],
    template_coords: Sequence[Coord],
    visible_coords: Sequence[Coord],
    ref_mask: np.ndarray,
) -> List[Coord]:
    if not visible_coords:
        return []
    h, w = grid_shape
    visible_set = set(visible_coords)
    first_r, first_c = visible_coords[0]
    candidates: List[Coord] = []
    seen: Set[Coord] = set()
    for tr, tc in template_coords:
        offset = (first_r - tr, first_c - tc)
        if offset in seen:
            continue
        seen.add(offset)
        cells = _stamp_cells(template_coords, offset)
        if any(r < 0 or c < 0 or r >= h or c >= w for r, c in cells):
            continue
        if any(ref_mask[r, c] for r, c in cells):
            continue
        if not visible_set.issubset(cells):
            continue
        candidates.append(offset)
    return candidates


def _topological_order(colors: Sequence[int], edges: Dict[int, Set[int]]) -> Optional[List[int]]:
    indeg = {color: 0 for color in colors}
    for src in colors:
        for dst in edges[src]:
            indeg[dst] += 1

    first = [color for color in colors if indeg[color] == 0]
    if len(first) != 1:
        return None

    remaining = set(colors)
    ready = sorted(first)
    order: List[int] = []
    while ready:
        node = ready.pop(0)
        if node not in remaining:
            continue
        order.append(node)
        remaining.remove(node)
        for nxt in edges[node]:
            indeg[nxt] -= 1
        for cand in sorted(remaining):
            if indeg[cand] == 0 and cand not in ready:
                ready.append(cand)
    if remaining:
        return None
    return order


def _validate_configuration(
    grid: np.ndarray,
    colors: Sequence[int],
    ref_mask: np.ndarray,
    template_coords: Sequence[Coord],
    offsets: Dict[int, Coord],
) -> Optional[Tuple[Dict[int, Set[Coord]], List[int]]]:
    full_by_color: Dict[int, Set[Coord]] = {}
    covers: Dict[Coord, List[int]] = {}
    for color in colors:
        cells = _stamp_cells(template_coords, offsets[color])
        full_by_color[color] = cells
        for cell in cells:
            covers.setdefault(cell, []).append(color)

    edges = {color: set() for color in colors}
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if ref_mask[r, c]:
                continue
            observed = int(grid[r, c])
            covered = covers.get((r, c), [])
            if observed == BG:
                if covered:
                    return None
                continue
            if observed not in covered:
                return None
            for other in covered:
                if other != observed:
                    edges[observed].add(other)

    order = _topological_order(colors, edges)
    if order is None:
        return None
    return full_by_color, order


def _solve_layers(grid: np.ndarray) -> Tuple[np.ndarray, Dict[int, Set[Coord]], List[int]]:
    g = np.asarray(grid, dtype=int)
    ref_box = _find_reference_box(g)
    template = _extract_template(g, ref_box)
    template_coords = _mask_coords(template)

    ref_mask = np.zeros(g.shape, dtype=bool)
    r0, c0, r1, c1 = ref_box
    ref_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    colors = sorted(int(v) for v in np.unique(g) if v not in (BG, GREY))
    assert len(colors) >= 2, "need at least two colored stamps"

    visible_by_color: Dict[int, List[Coord]] = {}
    candidate_map: Dict[int, List[Coord]] = {}
    for color in colors:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        visible_by_color[color] = coords
        candidates = _candidate_offsets(g.shape, template_coords, coords, ref_mask)
        assert candidates, f"no placement candidates for color {color}"
        candidate_map[color] = candidates

    color_order = sorted(colors, key=lambda color: (len(candidate_map[color]), color))
    solution: Optional[Tuple[Dict[int, Set[Coord]], List[int]]] = None

    def search(idx: int, chosen: Dict[int, Coord]) -> bool:
        nonlocal solution
        if idx == len(color_order):
            checked = _validate_configuration(g, colors, ref_mask, template_coords, chosen)
            if checked is not None:
                solution = checked
                return True
            return False

        color = color_order[idx]
        for offset in candidate_map[color]:
            chosen[color] = offset
            if search(idx + 1, chosen):
                return True
            chosen.pop(color)
        return False

    ok = search(0, {})
    assert ok and solution is not None, "failed to reconstruct stamp layers"
    return ref_mask, solution[0], solution[1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the full lower stamp layers from a shared template and erase the topmost stamp.

    Input structure:
    - The background is black, and a small reference box with a grey frame sits on the right side or bottom.
    - Inside that reference box, grey cells show one template shape on black, separated from the outer frame.
    - In the main area, each non-grey color appears once as a visible full or partial copy of that same template after overlapping stamps hide lower cells.

    Rule:
    - Detect the template from the grey reference box and treat every colored object as one translated copy of that template.
    - Infer each stamp's full footprint and relative height order by checking which visible colors occupy cells that another color is missing.
    - Restore every non-top stamp to its full template shape, remove the unique topmost stamp entirely, and keep the grey reference box unchanged.

    Key insight:
    - The colored fragments are not different shapes: they are partial views of one shared stencil, so the missing cells can be reconstructed by combining template matching with overlap-order reasoning.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    ref_mask, full_by_color, top_to_bottom = _solve_layers(g)
    out = np.full(g.shape, BG, dtype=int)
    out[ref_mask] = g[ref_mask]
    for color in reversed(top_to_bottom[1:]):
        for r, c in full_by_color[color]:
            out[r, c] = color
    return out


def _overlap_shifts(template: np.ndarray) -> List[Coord]:
    coords = set(_mask_coords(template))
    h, w = template.shape
    shifts: List[Coord] = []
    for dr in range(-h + 1, h):
        for dc in range(-w + 1, w):
            if dr == 0 and dc == 0:
                continue
            overlap = sum((r + dr, c + dc) in coords for r, c in coords)
            if 1 <= overlap <= len(coords) - MIN_VISIBLE:
                shifts.append((dr, dc))
    return shifts


def _place_chain(template: np.ndarray, main_h: int, main_w: int, n_stamps: int) -> Optional[List[Coord]]:
    th, tw = template.shape
    template_coords = _mask_coords(template)
    shifts = _overlap_shifts(template)
    if not shifts:
        return None

    def cells(offset: Coord) -> Set[Coord]:
        return _stamp_cells(template_coords, offset)

    starts = [(r, c) for r in range(main_h - th + 1) for c in range(main_w - tw + 1)]
    random.shuffle(starts)

    def in_bounds(offset: Coord) -> bool:
        r0, c0 = offset
        return 0 <= r0 and 0 <= c0 and r0 + th <= main_h and c0 + tw <= main_w

    def search(positions: List[Coord], cell_sets: List[Set[Coord]]) -> Optional[List[Coord]]:
        if len(positions) == n_stamps:
            return list(positions)

        prev = positions[-1]
        prev_cells = cell_sets[-1]
        local_shifts = list(shifts)
        random.shuffle(local_shifts)
        for dr, dc in local_shifts:
            cand = (prev[0] + dr, prev[1] + dc)
            if not in_bounds(cand):
                continue
            cand_cells = cells(cand)
            if any(cand_cells & earlier for earlier in cell_sets[:-1]):
                continue
            if len(prev_cells - cand_cells) < MIN_VISIBLE:
                continue
            positions.append(cand)
            cell_sets.append(cand_cells)
            found = search(positions, cell_sets)
            if found is not None:
                return found
            positions.pop()
            cell_sets.pop()
        return None

    for start in starts:
        found = search([start], [cells(start)])
        if found is not None:
            return found
    return None


def _make_scene(template: np.ndarray, side: str, order_colors: Sequence[int]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    ref = _reference_patch(template)
    ref_h, ref_w = ref.shape

    if side == "right":
        main_h = random.randint(max(12, ref_h + 2), 22)
        main_w = random.randint(12, 18)
        grid_h = main_h
        grid_w = main_w + GAP + ref_w
        if grid_w > 30:
            return None
        ref_r = random.randint(0, grid_h - ref_h)
        ref_c = grid_w - ref_w
    else:
        main_h = random.randint(12, 18)
        main_w = random.randint(max(12, ref_w + 2), 22)
        grid_h = main_h + GAP + ref_h
        grid_w = main_w
        if grid_h > 30:
            return None
        ref_r = grid_h - ref_h
        ref_c = random.randint(0, grid_w - ref_w)

    positions = _place_chain(template, main_h, main_w, len(order_colors))
    if positions is None:
        return None

    template_coords = _mask_coords(template)
    full_by_color = {
        color: _stamp_cells(template_coords, offset)
        for color, offset in zip(order_colors, positions)
    }

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    inp[ref_r : ref_r + ref_h, ref_c : ref_c + ref_w] = ref
    for color in order_colors:
        for r, c in full_by_color[color]:
            inp[r, c] = color

    visible_counts = {color: int(np.count_nonzero(inp == color)) for color in order_colors}
    if any(count < MIN_VISIBLE for count in visible_counts.values()):
        return None

    out = np.full((grid_h, grid_w), BG, dtype=int)
    out[ref_r : ref_r + ref_h, ref_c : ref_c + ref_w] = ref
    for color in order_colors[:-1]:
        for r, c in full_by_color[color]:
            out[r, c] = color

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def _gen_attempt() -> Optional[np.ndarray]:
    template = random.choice(TEMPLATE_LIBRARY)
    side = random.choice(["right", "bottom"])
    n_stamps = random.randint(3, 5)
    colors = random.sample([c for c in range(1, 10) if c != GREY], n_stamps)
    scene = _make_scene(template, side, colors)
    if scene is None:
        return None
    return scene[0]


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
