from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
LEFT_SIZE = 9
RIGHT_SIZE = 9
GRID_SHAPE = (9, 18)

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SPIRAL_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
CORNER_NODES = [(2, 2), (2, 6), (6, 2), (6, 6)]
REFERENCE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
REFERENCE_SLOTS = [(0, 0), (0, 5), (5, 0), (5, 5)]

GEN_MAX_ATTEMPTS = 300
GEN_CANDIDATE_BUDGET = 220


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _shape_key(coords: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    assert coords.size > 0, "shape key requires nonempty coords"
    r0, c0, r1, c1 = _bbox_coords(coords)
    offsets = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    return (int(r1 - r0 + 1), int(c1 - c0 + 1)), offsets


def _normalize_patch(coords: np.ndarray, color: int) -> np.ndarray:
    (h, w), offsets = _shape_key(coords)
    patch = np.zeros((h, w), dtype=int)
    for r, c in offsets:
        patch[r, c] = int(color)
    return patch


def _spiral_midpoints(node_rc: tuple[int, int]) -> frozenset[tuple[int, int]]:
    node_r = node_rc[0] // 2
    node_c = node_rc[1] // 2
    r, c = node_r, node_c
    midpoints: list[tuple[int, int]] = []
    step = 1
    dir_idx = 0

    # A 5x5 node lattice is fully exhausted long before step 9.
    for _ in range(9):
        for _ in range(2):
            dr, dc = SPIRAL_DIRS[dir_idx % 4]
            for _ in range(step):
                nr, nc = r + dr, c + dc
                if 0 <= r < 5 and 0 <= c < 5 and 0 <= nr < 5 and 0 <= nc < 5:
                    midpoints.append((r + nr, c + nc))
                r, c = nr, nc
            dir_idx += 1
        step += 1

    return frozenset(midpoints)


SPIRAL_CACHE = {(r, c): _spiral_midpoints((r, c)) for r in range(0, 9, 2) for c in range(0, 9, 2)}

BASE_FOUR_HOLE_COORDS = [
    [(1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (3, 1), (4, 1)],
    [(1, 6), (2, 5), (2, 6), (3, 4), (3, 5), (4, 5)],
    [(4, 3), (5, 2), (5, 3), (5, 4), (6, 2)],
    [(4, 7), (5, 6), (5, 7), (6, 5), (6, 6), (7, 4), (7, 5)],
]


def _hole_components(left: np.ndarray) -> list[np.ndarray]:
    holes = []
    for coords in _components_from_mask(left == BLACK):
        if np.all((coords[:, 0] > 0) & (coords[:, 0] < LEFT_SIZE - 1) & (coords[:, 1] > 0) & (coords[:, 1] < LEFT_SIZE - 1)):
            holes.append(coords)
    return holes


def _hole_fragment(coords: np.ndarray) -> tuple[tuple[int, int], np.ndarray]:
    nodes = [(int(r), int(c)) for r, c in coords if r % 2 == 0 and c % 2 == 0]
    assert len(nodes) == 1, "each hole must contain exactly one lattice node"
    node = nodes[0]
    hole_set = {tuple(map(int, cell)) for cell in coords}
    fragment = sorted(hole_set & SPIRAL_CACHE[node])
    assert fragment, "each generated hole must keep a nonempty spiral fragment"
    return node, np.asarray(fragment, dtype=int)


def _reference_shapes(right: np.ndarray) -> dict[tuple[tuple[int, int], tuple[tuple[int, int], ...]], int]:
    refs: dict[tuple[tuple[int, int], tuple[tuple[int, int], ...]], int] = {}
    for color in sorted(int(v) for v in np.unique(right) if int(v) != BLACK):
        coords = np.argwhere(right == color)
        key = _shape_key(coords)
        assert key not in refs, "reference shapes must be unique"
        refs[key] = color
    return refs


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each left-template hole with the color of the right-side reference whose shape matches that hole's hidden spiral fragment.

    Input structure:
    - The 9x18 grid is split into a 9x9 left template and a 9x9 right reference panel.
    - The left panel is a grey template containing 3 or 4 enclosed black hole-regions; each hole contains exactly one implicit lattice node at an even row and even column.
    - The right panel contains one monochrome colored reference shape per hole on a black background.

    Rule:
    - For each hole, use its unique even-even cell as the start node of a square spiral with step lengths 1,1,2,2,3,3,... and directions up, right, down, left, repeating on the hidden 5x5 node lattice.
    - Keep only midpoint pixels of spiral edges whose two endpoint nodes both stay inside that 5x5 lattice, then keep only those midpoint pixels that lie inside the current hole.
    - Normalize that retained fragment to its bounding box and match it against the normalized shapes in the right panel.
    - Fill the entire hole with the uniquely matched reference color and return only the left 9x9 panel.

    Key insight:
    - The hole outline itself is not the matching cue; the true reference shape appears only after reconstructing the invisible lattice spiral and clipping its midpoint trace to the hole interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == GRID_SHAPE, "expected a 9x18 grid"
    assert _get_bg(g) == BLACK, "black must be the dominant background"

    left = g[:, :LEFT_SIZE]
    right = g[:, LEFT_SIZE:]
    assert set(int(v) for v in np.unique(left)) <= {BLACK, GREY}, "left panel must contain only template and hole colors in the input"
    refs = _reference_shapes(right)
    holes = _hole_components(left)
    assert 3 <= len(holes) <= 4, "expected 3 or 4 enclosed holes"
    assert len(refs) == len(holes), "reference count must match hole count"

    out = left.copy()
    for coords in holes:
        _node, fragment = _hole_fragment(coords)
        key = _shape_key(fragment)
        matches = [color for shape, color in refs.items() if shape == key]
        assert len(matches) == 1, "each hole fragment must match exactly one reference shape"
        color = matches[0]
        out[coords[:, 0], coords[:, 1]] = color

    return out


def _candidate_allowed_cells(node: tuple[int, int]) -> set[tuple[int, int]]:
    allowed: set[tuple[int, int]] = set()
    for r in range(1, LEFT_SIZE - 1):
        for c in range(1, LEFT_SIZE - 1):
            if r % 2 == 0 and c % 2 == 0 and (r, c) != node:
                continue
            allowed.add((r, c))
    return allowed


def _candidate_from_cells(cells: set[tuple[int, int]] | list[tuple[int, int]]) -> dict | None:
    coords = np.asarray(sorted(set(cells)), dtype=int)
    hole_mask = np.zeros((LEFT_SIZE, LEFT_SIZE), dtype=bool)
    hole_mask[coords[:, 0], coords[:, 1]] = True
    if len(_components_from_mask(hole_mask)) != 1:
        return None

    node_cells = [(int(r), int(c)) for r, c in coords if r % 2 == 0 and c % 2 == 0]
    if len(node_cells) != 1:
        return None
    node = node_cells[0]

    cell_set = frozenset((int(r), int(c)) for r, c in coords)
    fragment_cells = sorted(cell_set & SPIRAL_CACHE[node])
    if not (3 <= len(fragment_cells) <= 6):
        return None

    fragment = np.asarray(fragment_cells, dtype=int)
    frag_key = _shape_key(fragment)
    hole_key = _shape_key(coords)
    if frag_key[0][0] > 4 or frag_key[0][1] > 4:
        return None
    if hole_key == frag_key or len(coords) <= len(fragment):
        return None

    return {
        "node": node,
        "hole_coords": coords,
        "hole_set": cell_set,
        "hole_key": hole_key,
        "fragment": fragment,
        "fragment_key": frag_key,
    }


def _sample_hole_candidate(node: tuple[int, int], *, max_size: int) -> dict | None:
    allowed = _candidate_allowed_cells(node)
    cells = {node}
    target = random.randint(5, max_size)

    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            for dr, dc in DIRS4:
                nxt = (r + dr, c + dc)
                if nxt in allowed and nxt not in cells:
                    frontier.add(nxt)
        if not frontier:
            return None
        # Bias toward compact holes; this makes 4-hole layouts easier while keeping variety.
        best_dist = min(abs(r - node[0]) + abs(c - node[1]) for r, c in frontier)
        compact = [cell for cell in frontier if abs(cell[0] - node[0]) + abs(cell[1] - node[1]) <= best_dist + 1]
        cells.add(random.choice(compact))

    cand = _candidate_from_cells(cells)
    if cand is None or cand["node"] != node:
        return None
    return cand


def _holes_compatible(a: dict, b: dict) -> bool:
    aset = a["hole_set"]
    bset = b["hole_set"]
    if aset & bset:
        return False
    for r, c in aset:
        for dr, dc in DIRS4:
            if (r + dr, c + dc) in bset:
                return False
    return True


def _candidate_catalog(node: tuple[int, int], *, max_size: int) -> list[dict]:
    uniq: dict[tuple[frozenset[tuple[int, int]], tuple[tuple[int, int], tuple[tuple[int, int], ...]]], dict] = {}
    for _ in range(GEN_CANDIDATE_BUDGET):
        cand = _sample_hole_candidate(node, max_size=max_size)
        if cand is None:
            continue
        uniq[(cand["hole_set"], cand["fragment_key"])] = cand
    return list(uniq.values())


def _select_holes(nodes: list[tuple[int, int]], *, max_size: int) -> list[dict] | None:
    catalogs = {node: _candidate_catalog(node, max_size=max_size) for node in nodes}
    if any(not catalogs[node] for node in nodes):
        return None

    ordered = sorted(nodes, key=lambda node: len(catalogs[node]))
    chosen: list[dict] = []
    used_shapes: set[tuple[tuple[int, int], tuple[tuple[int, int], ...]]] = set()

    def rec(idx: int) -> bool:
        if idx == len(ordered):
            return True

        node = ordered[idx]
        candidates = catalogs[node][:]
        random.shuffle(candidates)
        for cand in candidates:
            if cand["fragment_key"] in used_shapes:
                continue
            if all(_holes_compatible(cand, prev) for prev in chosen):
                chosen.append(cand)
                used_shapes.add(cand["fragment_key"])
                if rec(idx + 1):
                    return True
                chosen.pop()
                used_shapes.remove(cand["fragment_key"])
        return False

    return chosen if rec(0) else None


def _build_four_hole_layout() -> list[dict]:
    layout = []
    for hole in BASE_FOUR_HOLE_COORDS:
        cand = _candidate_from_cells(hole)
        assert cand is not None, "base four-hole layout must stay valid"
        layout.append(cand)
    assert len({cand["fragment_key"] for cand in layout}) == 4, "four-hole layout must keep unique fragments"
    assert all(_holes_compatible(layout[i], layout[j]) for i in range(4) for j in range(i + 1, 4)), (
        "four-hole layout holes must stay separated"
    )
    return layout


FOUR_HOLE_LAYOUT = _build_four_hole_layout()


def _render_left(holes: list[dict]) -> np.ndarray:
    left = np.full((LEFT_SIZE, LEFT_SIZE), GREY, dtype=int)
    for hole in holes:
        coords = hole["hole_coords"]
        left[coords[:, 0], coords[:, 1]] = BLACK
    return left


def _render_right(holes: list[dict], colors: list[int]) -> np.ndarray:
    right = np.zeros((RIGHT_SIZE, RIGHT_SIZE), dtype=int)
    slots = random.sample(REFERENCE_SLOTS, len(holes))
    order = list(range(len(holes)))
    random.shuffle(order)

    for hole_idx, slot in zip(order, slots):
        hole = holes[hole_idx]
        color = colors[hole_idx]
        patch = _normalize_patch(hole["fragment"], color)
        h, w = patch.shape
        sr, sc = slot
        rr = sr + random.randint(0, 4 - h)
        cc = sc + random.randint(0, 4 - w)
        mask = patch != BLACK
        assert not np.any((right[rr : rr + h, cc : cc + w] != BLACK) & mask), "reference objects must not overlap"
        right[rr : rr + h, cc : cc + w][mask] = patch[mask]

    return right


def _expected_output(left: np.ndarray, holes: list[dict], colors: list[int]) -> np.ndarray:
    out = left.copy()
    for hole, color in zip(holes, colors):
        coords = hole["hole_coords"]
        out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def _validate_instance(inp: np.ndarray, expected: np.ndarray) -> bool:
    if inp.shape != GRID_SHAPE or expected.shape != (LEFT_SIZE, LEFT_SIZE):
        return False

    try:
        out = main(inp)
    except Exception:
        return False
    if not np.array_equal(out, expected):
        return False

    left = inp[:, :LEFT_SIZE]
    right = inp[:, LEFT_SIZE:]
    holes = _hole_components(left)
    refs = _reference_shapes(right)
    if len(holes) != len(refs):
        return False

    ref_shapes = set(refs)
    if len(ref_shapes) != len(refs):
        return False

    # Direct hole-shape matching must fail; the spiral step is required.
    for coords in holes:
        hole_key = _shape_key(coords)
        if hole_key in ref_shapes:
            return False
        _node, fragment = _hole_fragment(coords)
        if _shape_key(fragment) not in ref_shapes:
            return False

    return True


def _build_instance() -> tuple[np.ndarray, np.ndarray] | None:
    hole_count = random.choices([3, 4], weights=[5, 4], k=1)[0]
    if hole_count == 4:
        holes = list(FOUR_HOLE_LAYOUT)
    else:
        nodes = random.sample(CORNER_NODES, hole_count)
        holes = _select_holes(nodes, max_size=10)
        if holes is None:
            return None

    random.shuffle(holes)
    colors = random.sample(REFERENCE_COLORS, len(holes))
    left = _render_left(holes)
    right = _render_right(holes, colors)
    inp = np.concatenate([left, right], axis=1)
    out = _expected_output(left, holes, colors)
    return (inp, out) if _validate_instance(inp, out) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        built = _build_instance()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid spiral-hole matching instance")
