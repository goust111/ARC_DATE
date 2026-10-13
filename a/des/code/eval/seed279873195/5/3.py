import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 300
GEN_MAX_SIZE = 30
GEN_MIN_SIZE = 18

INNER_SIZE_RANGE = {
    1: (3, 6),
    2: (3, 5),
    3: (3, 4),
}
MARGIN_RANGE = (2, 3)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _component_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in neighbors:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _component_bbox(coords),
                }
            )
    return comps


def _is_rectangular_frame(comp: dict) -> bool:
    coords = comp["coords"]
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False

    coord_set = set(coords)
    if len(coord_set) != 2 * (h + w) - 4:
        return False

    for c in range(c0, c1 + 1):
        if (r0, c) not in coord_set or (r1, c) not in coord_set:
            return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in coord_set or (r, c1) not in coord_set:
            return False
    return True


def _contains(parent: dict, child: dict) -> bool:
    pr0, pc0, pr1, pc1 = parent["bbox"]
    cr0, cc0, cr1, cc1 = child["bbox"]
    return pr0 + 2 <= cr0 and pc0 + 2 <= cc0 and cr1 <= pr1 - 2 and cc1 <= pc1 - 2


def main(grid: np.ndarray) -> np.ndarray:
    """Group outer frames by nesting depth and repaint each group with its innermost color."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    if np.any(g == BG):
        bg = BG
    else:
        bg = int(np.bincount(g.ravel()).argmax())

    components = _find_components(g, bg)
    assert components, "expected at least one foreground component"

    frames = [comp for comp in components if _is_rectangular_frame(comp)]
    assert len(frames) == len(components), "all foreground components must be rectangular frames"

    n = len(frames)
    parents: list[int | None] = [None] * n
    children: dict[int, list[int]] = {i: [] for i in range(n)}

    for i, frame in enumerate(frames):
        containers = [j for j, other in enumerate(frames) if i != j and _contains(other, frame)]
        if not containers:
            continue
        parent = min(containers, key=lambda idx: _bbox_area(frames[idx]["bbox"]))
        parents[i] = parent
        children[parent].append(i)

    for idx, child_list in children.items():
        assert len(child_list) <= 1, f"frame {idx} has multiple direct children"

    roots = [idx for idx, parent in enumerate(parents) if parent is None]
    assert roots, "expected at least one outermost frame"

    depth_color: dict[int, int] = {}
    chains_by_depth: dict[int, list[list[int]]] = {}
    visited: set[int] = set()

    for root in roots:
        chain: list[int] = []
        cur = root
        while True:
            assert cur not in visited, "nesting relation must be acyclic"
            visited.add(cur)
            chain.append(cur)
            child_list = children[cur]
            if not child_list:
                break
            cur = child_list[0]

        depth = len(chain)
        assert 1 <= depth <= 3, "supported nesting depth is 1..3"
        deepest_color = int(frames[chain[-1]]["color"])
        if depth in depth_color:
            assert depth_color[depth] == deepest_color, "all frames in a depth group must share the innermost color"
        else:
            depth_color[depth] = deepest_color
        chains_by_depth.setdefault(depth, []).append(chain)

    assert len(visited) == n, "every frame must belong to exactly one nesting chain"

    out = g.copy()
    for depth, chains in chains_by_depth.items():
        target_color = depth_color[depth]
        for chain in chains:
            for idx in chain:
                for r, c in frames[idx]["coords"]:
                    out[r, c] = target_color
    return out


def _build_object(rng: np.random.Generator, depth: int, deepest_color: int, accent_pool: list[int]) -> dict:
    lo, hi = INNER_SIZE_RANGE[depth]
    inner_h = int(rng.integers(lo, hi + 1))
    inner_w = int(rng.integers(lo, hi + 1))

    bboxes = [(0, 0, inner_h - 1, inner_w - 1)]
    total_h, total_w = inner_h, inner_w

    for _ in range(depth - 1):
        top = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))
        bottom = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))
        left = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))
        right = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))

        shifted = []
        for r0, c0, r1, c1 in bboxes:
            shifted.append((r0 + top, c0 + left, r1 + top, c1 + left))
        total_h += top + bottom
        total_w += left + right
        bboxes = [(0, 0, total_h - 1, total_w - 1)] + shifted

    outer_colors = []
    if depth > 1:
        outer_colors = rng.choice(accent_pool, size=depth - 1, replace=False).tolist()
    colors = outer_colors + [deepest_color]

    patch = np.zeros((total_h, total_w), dtype=int)
    for bbox, color in zip(bboxes, colors):
        _draw_frame(patch, bbox, int(color))

    return {
        "depth": depth,
        "deepest_color": deepest_color,
        "patch": patch,
        "layer_bboxes": bboxes,
        "layer_colors": [int(color) for color in colors],
    }


def _random_composition(rng: np.random.Generator, total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(rng.choice(np.arange(1, total), size=parts - 1, replace=False).tolist())
    out = []
    prev = 0
    for cut in cuts + [total]:
        out.append(cut - prev)
        prev = cut
    return out


def _arrange_objects(
    rng: np.random.Generator, objects: list[dict]
) -> tuple[list[tuple[dict, int, int]], int, int] | None:
    n = len(objects)
    for _ in range(80):
        order = list(objects)
        rng.shuffle(order)

        rows = 2 if n <= 4 or rng.random() < 0.7 else 3
        rows = min(rows, n)
        row_sizes = _random_composition(rng, n, rows)

        split_rows: list[list[dict]] = []
        start = 0
        for size in row_sizes:
            split_rows.append(order[start : start + size])
            start += size

        row_heights: list[int] = []
        row_widths: list[int] = []
        horiz_gaps: list[list[int]] = []
        for row in split_rows:
            heights = [obj["patch"].shape[0] for obj in row]
            widths = [obj["patch"].shape[1] for obj in row]
            gaps = [int(rng.integers(1, 4)) for _ in range(max(0, len(row) - 1))]
            row_heights.append(max(heights))
            row_widths.append(sum(widths) + sum(gaps))
            horiz_gaps.append(gaps)

        vert_gaps = [int(rng.integers(1, 4)) for _ in range(max(0, rows - 1))]
        layout_h = sum(row_heights) + sum(vert_gaps)
        layout_w = max(row_widths)

        extra_top = int(rng.integers(1, 4))
        extra_bottom = int(rng.integers(1, 5))
        extra_left = int(rng.integers(1, 4))
        extra_right = int(rng.integers(1, 5))
        grid_h = layout_h + extra_top + extra_bottom
        grid_w = layout_w + extra_left + extra_right

        if not (GEN_MIN_SIZE <= grid_h <= GEN_MAX_SIZE and GEN_MIN_SIZE <= grid_w <= GEN_MAX_SIZE):
            continue

        placements = []
        row_top = extra_top
        for row_index, row in enumerate(split_rows):
            row_h = row_heights[row_index]
            row_w = row_widths[row_index]
            gaps = horiz_gaps[row_index]
            start_x = extra_left + int(rng.integers(0, layout_w - row_w + 1))
            x = start_x
            for idx, obj in enumerate(row):
                patch_h, patch_w = obj["patch"].shape
                y = row_top + int(rng.integers(0, row_h - patch_h + 1))
                placements.append((obj, y, x))
                x += patch_w
                if idx < len(gaps):
                    x += gaps[idx]
            row_top += row_h
            if row_index < len(vert_gaps):
                row_top += vert_gaps[row_index]

        return placements, grid_h, grid_w
    return None


def _make_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    deepest_colors = rng.choice(PALETTE, size=3, replace=False).tolist()
    depth_to_color = {1: int(deepest_colors[0]), 2: int(deepest_colors[1]), 3: int(deepest_colors[2])}
    accent_pool = [color for color in PALETTE if color not in depth_to_color.values()]

    counts = {1: 1, 2: 1, 3: 1}
    extra_count = int(rng.choice([1, 2], p=[0.65, 0.35]))
    extra_depths = rng.choice(
        [1, 2, 3], size=extra_count, replace=(extra_count == 2), p=[0.45, 0.35, 0.20]
    ).tolist()
    for depth in extra_depths:
        counts[int(depth)] += 1

    objects: list[dict] = []
    for depth in (1, 2, 3):
        for _ in range(counts[depth]):
            objects.append(_build_object(rng, depth, depth_to_color[depth], accent_pool))

    arranged = _arrange_objects(rng, objects)
    if arranged is None:
        return None
    placements, grid_h, grid_w = arranged

    grid = np.zeros((grid_h, grid_w), dtype=int)
    expected = np.zeros_like(grid)

    for obj, top, left in placements:
        patch = obj["patch"]
        h, w = patch.shape
        mask = patch != BG
        region = grid[top : top + h, left : left + w]
        region[mask] = patch[mask]

        for bbox in obj["layer_bboxes"]:
            r0, c0, r1, c1 = bbox
            abs_bbox = (top + r0, left + c0, top + r1, left + c1)
            _draw_frame(expected, abs_bbox, obj["deepest_color"])

    # Keep the background as black and validate that the independent expected output matches the solver.
    assert np.array_equal(main(grid), expected)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = np.random.default_rng()
        result = _make_case(rng)
        if result is not None:
            return result[0]
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
