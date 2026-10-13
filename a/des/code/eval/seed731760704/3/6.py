import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

INPUT_COLORS = {BLACK, GREY, RED}

# --------------- generator config ---------------
GRID_H_RANGE = (22, 38)
GRID_W_RANGE = (22, 38)
ROOT_COUNT_CHOICES = (1, 2, 3)
INNER_SIZE_RANGE = (3, 7)
EXPAND_GAP_RANGE = (2, 4)
EXTRA_DEPTH_RANGE = (1, 3)
PLACEMENT_PADDING = 2
OUTER_MARGIN = 1
GENERATION_RETRIES = 500
MARKER_MODES = ("none", "single", "multi")

_RNG = random.Random()


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c1))
    if r1 > r0:
        for c in range(c1, c0 - 1, -1):
            cells.append((r1, c))
    if c1 > c0:
        for r in range(r1 - 1, r0, -1):
            cells.append((r, c0))
    return cells


def _extract_rings(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    values = set(map(int, np.unique(g)))
    assert values <= INPUT_COLORS, "input may only contain black, grey, and red"

    rings: list[dict[str, object]] = []
    for coords in _components(g != BLACK):
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0 = int(rows.min())
        c0 = int(cols.min())
        r1 = int(rows.max())
        c1 = int(cols.max())
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        assert h >= 3 and w >= 3, "every loop must have a non-empty interior"

        local = np.zeros((h, w), dtype=bool)
        local[rows - r0, cols - c0] = True

        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        assert np.array_equal(local, border), "each component must be a 1-pixel rectangular loop"

        colors = {int(g[int(r), int(c)]) for r, c in coords}
        assert colors <= {GREY, RED}, "loop pixels must be grey or red"
        marker_cells = [(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) == RED]
        assert len(marker_cells) <= 1, "each loop may contain at most one red marker"

        rings.append(
            {
                "bbox": (r0, c0, r1, c1),
                "coords": coords.copy(),
                "marker": marker_cells[0] if marker_cells else None,
            }
        )

    assert rings, "expected at least one loop"

    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            a = rings[i]["bbox"]
            b = rings[j]["bbox"]
            assert isinstance(a, tuple) and isinstance(b, tuple)
            if _boxes_intersect(a, b):
                assert _contains(a, b) or _contains(b, a), "loop bboxes must be disjoint or nested"

    return rings


def _nesting_forest(rings: list[dict[str, object]]) -> tuple[list[int | None], dict[int, list[int]], list[int]]:
    parents: list[int | None] = [None] * len(rings)
    for i, ring in enumerate(rings):
        bbox = ring["bbox"]
        assert isinstance(bbox, tuple)
        containers = [
            j
            for j, other in enumerate(rings)
            if i != j and _contains(other["bbox"], bbox)
        ]
        if containers:
            parents[i] = min(containers, key=lambda j: _bbox_area(rings[j]["bbox"]))

    children = {i: [] for i in range(len(rings))}
    roots: list[int] = []
    for i, parent in enumerate(parents):
        if parent is None:
            roots.append(i)
        else:
            children[parent].append(i)

    for child_list in children.values():
        child_list.sort(key=lambda idx: rings[idx]["bbox"])
    roots.sort(key=lambda idx: rings[idx]["bbox"])
    return parents, children, roots


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor nested rectangular loops by alternating depth, with red markers
    toggling the blue/yellow assignment for their own subtree.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rings = _extract_rings(g)
    _parents, children, roots = _nesting_forest(rings)
    out = np.full_like(g, BLACK)

    def paint(idx: int, depth: int, inverted: bool) -> None:
        ring = rings[idx]
        marker = ring["marker"]
        next_inverted = bool(inverted) ^ (marker is not None)
        base = BLUE if depth % 2 == 1 else YELLOW
        fill = YELLOW if (base == BLUE and next_inverted) else BLUE if (base == YELLOW and next_inverted) else base

        coords = ring["coords"]
        out[coords[:, 0], coords[:, 1]] = fill
        if marker is not None:
            mr, mc = marker
            out[int(mr), int(mc)] = RED

        for child in children[idx]:
            paint(child, depth + 1, next_inverted)

    for root in roots:
        paint(root, depth=1, inverted=False)

    return out


def _sample_inner_dims() -> tuple[int, int]:
    branch = _RNG.choice(("square", "wide", "tall", "free"))
    lo, hi = INNER_SIZE_RANGE
    if branch == "square":
        side = _RNG.randint(lo, hi)
        return side, side
    if branch == "wide":
        h = _RNG.randint(lo, hi - 1)
        w = _RNG.randint(h + 1, min(h + 4, hi + 2))
        return h, w
    if branch == "tall":
        w = _RNG.randint(lo, hi - 1)
        h = _RNG.randint(w + 1, min(w + 4, hi + 2))
        return h, w
    return _RNG.randint(lo, hi), _RNG.randint(lo, hi)


def _make_chain(depth: int) -> dict[str, object]:
    inner_h, inner_w = _sample_inner_dims()
    rects = [(0, 0, inner_h - 1, inner_w - 1)]
    height = inner_h
    width = inner_w

    for _ in range(depth - 1):
        top = _RNG.randint(*EXPAND_GAP_RANGE)
        bottom = _RNG.randint(*EXPAND_GAP_RANGE)
        left = _RNG.randint(*EXPAND_GAP_RANGE)
        right = _RNG.randint(*EXPAND_GAP_RANGE)
        rects = [(r0 + top, c0 + left, r1 + top, c1 + left) for (r0, c0, r1, c1) in rects]
        height += top + bottom
        width += left + right
        rects.insert(0, (0, 0, height - 1, width - 1))

    return {
        "rects": rects,
        "size": (height, width),
        "markers": {},
    }


def _choose_markers(mode: str, chains: list[dict[str, object]]) -> None:
    refs = [(chain_idx, rect_idx) for chain_idx, chain in enumerate(chains) for rect_idx in range(len(chain["rects"]))]
    assert refs

    chosen: list[tuple[int, int]] = []
    if mode == "single":
        chosen = [_RNG.choice(refs)]
    elif mode == "multi":
        main_refs = [(0, rect_idx) for rect_idx in range(len(chains[0]["rects"]))]
        marker_count = _RNG.randint(2, min(3, len(main_refs)))
        chosen = _RNG.sample(main_refs, marker_count)

    for chain_idx, rect_idx in chosen:
        rect = chains[chain_idx]["rects"][rect_idx]
        border = _border_cells(*rect)
        chains[chain_idx]["markers"][rect_idx] = _RNG.choice(border)


def _grid_lower_bounds(sizes: list[tuple[int, int]]) -> tuple[int, int]:
    max_h = max(h for h, _ in sizes)
    max_w = max(w for _, w in sizes)
    padded_area = sum((h + 2 * PLACEMENT_PADDING) * (w + 2 * PLACEMENT_PADDING) for h, w in sizes)
    side = int(np.ceil(np.sqrt(padded_area)))
    min_h = max(GRID_H_RANGE[0], max_h + 2 * OUTER_MARGIN, side)
    min_w = max(GRID_W_RANGE[0], max_w + 2 * OUTER_MARGIN, side)
    return min_h, min_w


def _candidate_positions(occupied: np.ndarray, box_h: int, box_w: int) -> list[tuple[int, int]]:
    h, w = occupied.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(OUTER_MARGIN, h - box_h - OUTER_MARGIN + 1):
        for c0 in range(OUTER_MARGIN, w - box_w - OUTER_MARGIN + 1):
            rr0 = max(0, r0 - PLACEMENT_PADDING)
            rr1 = min(h, r0 + box_h + PLACEMENT_PADDING)
            cc0 = max(0, c0 - PLACEMENT_PADDING)
            cc1 = min(w, c0 + box_w + PLACEMENT_PADDING)
            if not occupied[rr0:rr1, cc0:cc1].any():
                positions.append((r0, c0))
    return positions


def _place_chains(chains: list[dict[str, object]]) -> tuple[np.ndarray, dict[int, tuple[int, int]]] | None:
    sizes = [chain["size"] for chain in chains]
    min_h, min_w = _grid_lower_bounds(sizes)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    grid_h = _RNG.randint(min_h, GRID_H_RANGE[1])
    grid_w = _RNG.randint(min_w, GRID_W_RANGE[1])
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements: dict[int, tuple[int, int]] = {}
    order = sorted(range(len(chains)), key=lambda idx: chains[idx]["size"][0] * chains[idx]["size"][1], reverse=True)

    for idx in order:
        box_h, box_w = chains[idx]["size"]
        options = _candidate_positions(occupied, box_h, box_w)
        if not options:
            return None
        r0, c0 = _RNG.choice(options)
        occupied[r0 : r0 + box_h, c0 : c0 + box_w] = True
        placements[idx] = (r0, c0)

    return np.zeros((grid_h, grid_w), dtype=int), placements


def _draw_rect_border(grid: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = rect
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _render_input(chains: list[dict[str, object]], placements: dict[int, tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for idx, chain in enumerate(chains):
        top, left = placements[idx]
        for rect in chain["rects"]:
            r0, c0, r1, c1 = rect
            _draw_rect_border(grid, (top + r0, left + c0, top + r1, left + c1), GREY)
        for marker in chain["markers"].values():
            mr, mc = marker
            grid[top + mr, left + mc] = RED
    return grid


def _gen_attempt() -> np.ndarray | None:
    mode = _RNG.choice(MARKER_MODES)
    root_count = _RNG.choice(ROOT_COUNT_CHOICES)

    min_main_depth = 3 if mode == "multi" else 2
    main_depth = _RNG.randint(min_main_depth, 5)
    extra_depths = [_RNG.randint(*EXTRA_DEPTH_RANGE) for _ in range(root_count - 1)]

    chains = [_make_chain(main_depth)] + [_make_chain(depth) for depth in extra_depths]
    _choose_markers(mode, chains)

    placed = _place_chains(chains)
    if placed is None:
        return None
    canvas, placements = placed
    grid = _render_input(chains, placements, canvas.shape)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if int(np.count_nonzero(grid == RED)) == 0 and mode != "none":
        return None
    if int(np.count_nonzero(grid == RED)) != 0 and mode == "none":
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
