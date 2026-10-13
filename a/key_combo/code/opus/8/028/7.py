import itertools

import numpy as np


BG = 0
FRAME_COLORS = (1, 2, 3, 4)
MASK_COLOR = 5
LEGEND_COLORS = (6, 7, 8, 9)

GEN_RETRIES = 1500
RECT_RETRIES = 6000
MASK_PICK_RETRIES = 80


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _interior_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}


def _safe_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {(r, c) for r in range(r0 + 2, r1 - 1) for c in range(c0 + 2, c1 - 1)}


def _neighbors4(cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = cell
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _is_connected(cells: tuple[tuple[int, int], ...] | list[tuple[int, int]] | set[tuple[int, int]]) -> bool:
    cell_set = set(cells)
    if not cell_set:
        return False
    stack = [next(iter(cell_set))]
    seen = set()
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        for nxt in _neighbors4(cell):
            if nxt in cell_set and nxt not in seen:
                stack.append(nxt)
    return seen == cell_set


def _normalized_shape(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _unique_topological_order(n: int, edges: set[tuple[int, int]]) -> list[int]:
    outgoing = {i: set() for i in range(n)}
    indeg = [0] * n
    for src, dst in edges:
        if dst not in outgoing[src]:
            outgoing[src].add(dst)
            indeg[dst] += 1

    order = []
    remaining = set(range(n))
    while remaining:
        roots = [node for node in remaining if indeg[node] == 0]
        assert len(roots) == 1, "depth constraints must define a unique total order"
        node = roots[0]
        remaining.remove(node)
        order.append(node)
        for dst in outgoing[node]:
            indeg[dst] -= 1
    return order


def _extract_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for color in FRAME_COLORS:
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        bbox = _bbox_from_coords(coords)
        r0, c0, r1, c1 = bbox
        assert min(r1 - r0 + 1, c1 - c0 + 1) >= 4, "frame must be at least 4x4"
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            assert rr in (r0, r1) or cc in (c0, c1), "frame color must lie on its rectangle border"
        assert np.any(coords[:, 0] == r0)
        assert np.any(coords[:, 0] == r1)
        assert np.any(coords[:, 1] == c0)
        assert np.any(coords[:, 1] == c1)
        frames.append(
            {
                "color": int(color),
                "bbox": bbox,
                "border": _border_cells(bbox),
            }
        )
    assert 3 <= len(frames) <= 4, "expected 3 or 4 frames"
    return frames


def _extract_legend(grid: np.ndarray, frame_count: int) -> dict[int, int]:
    rank_to_color: dict[int, int] = {}
    present = [color for color in LEGEND_COLORS if np.any(grid == color)]
    assert len(present) == frame_count, "legend size must match frame count"
    for color in present:
        coords = np.argwhere(grid == color)
        bbox = _bbox_from_coords(coords)
        r0, c0, r1, c1 = bbox
        assert c0 == c1, "legend entry must be one column wide"
        height = int(r1 - r0 + 1)
        assert int(coords.shape[0]) == height, "legend entry must be a solid vertical block"
        assert 1 <= height <= frame_count
        assert height not in rank_to_color, "legend heights must be unique"
        rank_to_color[height] = int(color)
    assert set(rank_to_color) == set(range(1, frame_count + 1))
    return rank_to_color


def _infer_depth_order(grid: np.ndarray, frames: list[dict]) -> list[int]:
    edges: set[tuple[int, int]] = set()
    for i, frame_a in enumerate(frames):
        for j in range(i + 1, len(frames)):
            frame_b = frames[j]
            overlap = frame_a["border"] & frame_b["border"]
            assert overlap, "every frame pair must intersect"
            visible = {
                int(grid[r, c])
                for r, c in overlap
                if int(grid[r, c]) in (frame_a["color"], frame_b["color"])
            }
            assert len(visible) == 1, "pairwise intersection evidence must be unambiguous"
            top_color = next(iter(visible))
            if top_color == frame_a["color"]:
                edges.add((i, j))
            else:
                edges.add((j, i))
    return _unique_topological_order(len(frames), edges)


def _assign_masks(grid: np.ndarray, frames: list[dict]) -> list[list[tuple[int, int]]]:
    owners = [[] for _ in frames]
    for r, c in np.argwhere(grid == MASK_COLOR):
        rr = int(r)
        cc = int(c)
        hits = []
        for idx, frame in enumerate(frames):
            r0, c0, r1, c1 = frame["bbox"]
            if r0 < rr < r1 and c0 < cc < c1:
                hits.append(idx)
        assert len(hits) == 1, "each mask pixel must belong to exactly one frame interior"
        owners[hits[0]].append((rr, cc))

    shapes = set()
    for cells in owners:
        assert 3 <= len(cells) <= 5, "each frame must have a 3-5 pixel mask"
        shape = _normalized_shape(cells)
        assert shape not in shapes, "mask shapes must be distinct"
        shapes.add(shape)
    return owners


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    frames = _extract_frames(g)
    legend = _extract_legend(g, len(frames))
    order = _infer_depth_order(g, frames)
    masks = _assign_masks(g, frames)

    out = g.copy()
    for rank, frame_idx in enumerate(order, start=1):
        color = legend[rank]
        for r, c in masks[frame_idx]:
            out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each framed grey mask from the legend color indexed by its inferred occlusion depth rank.

    Input structure:
    - The background is black, with 3 or 4 hollow colored rectangular frames drawn using distinct colors from {1,2,3,4}.
    - Frame borders are 1 pixel thick and may occlude each other at border-crossing pixels; each frame also contains one small grey interior mask.
    - A separate lower-right legend contains distinct one-column color blocks from {6,7,8,9}, where block height k encodes depth rank k.

    Rule:
    - Recover each frame's full rectangle from its visible same-color border pixels.
    - For every intersecting frame pair, inspect the shared border-crossing pixels: the visible color there is the front frame, and all such pairwise constraints combine into one strict shallow-to-deep order.
    - Use the legend block of height k as the target color for the frame with depth rank k, and recolor only that frame's grey interior mask; keep borders, legend, background, and empty interiors unchanged.

    Key insight:
    - The recoloring key is not any geometric size or position of a frame: it must be inferred globally from occlusion evidence, then converted indirectly through legend heights into colors.
    """
    return _solve(np.asarray(grid, dtype=int))


def _sample_rectangles(rng: np.random.Generator, n: int, area_h: int, area_w: int) -> list[tuple[int, int, int, int]] | None:
    for _ in range(RECT_RETRIES):
        rects = []
        for _frame in range(n):
            h = int(rng.integers(6, 8))
            w = int(rng.integers(6, 8))
            r0 = int(rng.integers(0, area_h - h + 1))
            c0 = int(rng.integers(0, area_w - w + 1))
            rects.append((r0, c0, r0 + h - 1, c0 + w - 1))
        if len(set(rects)) != n:
            continue

        tops = [b[0] for b in rects]
        bottoms = [b[2] for b in rects]
        lefts = [b[1] for b in rects]
        rights = [b[3] for b in rects]
        if len(set(tops)) < n or len(set(bottoms)) < n or len(set(lefts)) < n or len(set(rights)) < n:
            continue

        borders = [_border_cells(bbox) for bbox in rects]
        if any(not (borders[i] & borders[j]) for i in range(n) for j in range(i + 1, n)):
            continue

        all_border_cells = set().union(*borders)
        if any(sum(cell in border for border in borders) >= 3 for cell in all_border_cells):
            continue

        interiors = [_interior_cells(bbox) for bbox in rects]
        borders_union = [_border_cells(bbox) for bbox in rects]
        exclusive_counts = []
        for idx, bbox in enumerate(rects):
            exclusive = {
                cell
                for cell in _safe_cells(bbox)
                if all(cell not in interiors[j] for j in range(n) if j != idx)
                and all(cell not in borders_union[j] for j in range(n) if j != idx)
            }
            exclusive_counts.append(len(exclusive))
        if min(exclusive_counts) < 3:
            continue

        return rects
    return None


def _connected_subsets(cells: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    coords = sorted(cells)
    out = []
    for size in range(3, min(5, len(coords)) + 1):
        for subset in itertools.combinations(coords, size):
            if _is_connected(subset):
                out.append(list(subset))
    return out


def _pick_masks(
    rects: list[tuple[int, int, int, int]], rng: np.random.Generator
) -> list[list[tuple[int, int]]] | None:
    interiors = [_interior_cells(bbox) for bbox in rects]
    borders = [_border_cells(bbox) for bbox in rects]
    candidates = []
    for idx, bbox in enumerate(rects):
        exclusive = {
            cell
            for cell in _safe_cells(bbox)
            if all(cell not in interiors[j] for j in range(len(rects)) if j != idx)
            and all(cell not in borders[j] for j in range(len(rects)) if j != idx)
        }
        subsets = _connected_subsets(exclusive)
        if not subsets:
            return None
        rng.shuffle(subsets)
        candidates.append(subsets)

    order = sorted(range(len(rects)), key=lambda i: len(candidates[i]))
    chosen: list[list[tuple[int, int]] | None] = [None] * len(rects)
    used_shapes: set[tuple[tuple[int, int], ...]] = set()

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        frame_idx = order[pos]
        options = candidates[frame_idx][:]
        rng.shuffle(options)
        for subset in options[:MASK_PICK_RETRIES]:
            shape = _normalized_shape(subset)
            if shape in used_shapes:
                continue
            used_shapes.add(shape)
            chosen[frame_idx] = subset
            if backtrack(pos + 1):
                return True
            chosen[frame_idx] = None
            used_shapes.remove(shape)
        return False

    if not backtrack(0):
        return None
    return [list(cells) for cells in chosen]


def _is_trivial_order(
    rects: list[tuple[int, int, int, int]],
    frame_colors: list[int],
    order: list[int],
) -> bool:
    naive_orders = []
    naive_orders.append(tuple(sorted(range(len(rects)), key=lambda i: (rects[i][0], rects[i][1]))))
    naive_orders.append(tuple(sorted(range(len(rects)), key=lambda i: (frame_colors[i], i))))
    target = tuple(order)
    return any(target == naive or target == naive[::-1] for naive in naive_orders)


def _legend_height(n: int) -> int:
    return n * (n + 1) // 2 + (n - 1)


def _render_case(
    rects: list[tuple[int, int, int, int]],
    frame_colors: list[int],
    depth_order: list[int],
    masks: list[list[tuple[int, int]]],
    rank_colors: dict[int, int],
    rng: np.random.Generator,
    area_h: int,
    area_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(rects)
    legend_h = _legend_height(n)
    height = area_h + legend_h + 2
    width = area_w + 4 + int(rng.integers(0, 3))
    inp = np.zeros((height, width), dtype=int)
    out = np.zeros((height, width), dtype=int)

    legend_col = width - 2
    legend_top = height - legend_h - 1
    row = legend_top
    legend_order = list(range(1, n + 1))
    rng.shuffle(legend_order)
    for idx, rank in enumerate(legend_order):
        color = rank_colors[rank]
        inp[row : row + rank, legend_col] = color
        out[row : row + rank, legend_col] = color
        row += rank
        if idx != len(legend_order) - 1:
            row += 1

    for cells in masks:
        for r, c in cells:
            inp[r, c] = MASK_COLOR

    for frame_idx in depth_order[::-1]:
        color = frame_colors[frame_idx]
        for r, c in _border_cells(rects[frame_idx]):
            inp[r, c] = color
            out[r, c] = color

    for frame_idx, cells in enumerate(masks):
        rank = depth_order.index(frame_idx) + 1
        color = rank_colors[rank]
        for r, c in cells:
            out[r, c] = color

    return inp, out


def _visible_side_counts(grid: np.ndarray, color: int, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    top = int(np.sum(grid[r0, c0 : c1 + 1] == color))
    bottom = int(np.sum(grid[r1, c0 : c1 + 1] == color))
    left = int(np.sum(grid[r0 : r1 + 1, c0] == color))
    right = int(np.sum(grid[r0 : r1 + 1, c1] == color))
    return top, bottom, left, right


def _generate_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n = 3 if rng.random() < 0.55 else 4
    legend_h = _legend_height(n)
    area_h = int(rng.integers(12, 16 if n == 4 else 18))
    area_w = int(rng.integers(13, 20))
    if area_h + legend_h + 2 > 30:
        return None

    rects = _sample_rectangles(rng, n, area_h, area_w)
    if rects is None:
        return None

    frame_colors = [int(v) for v in rng.choice(np.array(FRAME_COLORS), size=n, replace=False)]
    legend_palette = [int(v) for v in rng.choice(np.array(LEGEND_COLORS), size=n, replace=False)]
    depth_order = [int(v) for v in rng.permutation(n)]
    if _is_trivial_order(rects, frame_colors, depth_order):
        return None

    rank_colors = {rank: legend_palette[rank - 1] for rank in range(1, n + 1)}
    masks = _pick_masks(rects, rng)
    if masks is None:
        return None

    inp, expected = _render_case(rects, frame_colors, depth_order, masks, rank_colors, rng, area_h, area_w)

    for bbox, color in zip(rects, frame_colors):
        coords = np.argwhere(inp == color)
        found = _bbox_from_coords(coords)
        if found != bbox:
            return None
        if min(_visible_side_counts(inp, color, bbox)) < 2:
            return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid occlusion-depth ARC input")
