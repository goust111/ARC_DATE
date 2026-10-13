from __future__ import annotations

import numpy as np


MIN_SIDE = 4
MAX_SIDE = 8
GRID_MIN = 12
GRID_MAX = 16
FRAME_COUNT_MIN = 3
FRAME_COUNT_MAX = 5
EDGE_MARGIN = 1
GEN_RETRY_LIMIT = 400
LAYER_PLACEMENT_TRIES = 220
PALETTE = np.arange(1, 10, dtype=int)
_GEN_RNG = np.random.default_rng()


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(int(lo), int(hi) + 1))


def _border_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _side_slices(bbox: tuple[int, int, int, int]):
    r0, c0, r1, c1 = map(int, bbox)
    return (
        (r0, slice(c0, c1 + 1)),
        (r1, slice(c0, c1 + 1)),
        (slice(r0, r1 + 1), c0),
        (slice(r0, r1 + 1), c1),
    )


def _has_full_visible_side(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> bool:
    for side in _side_slices(bbox):
        if bool(np.all(grid[side] == int(color))):
            return True
    return False


def _visible_side_count(blockers: np.ndarray, bbox: tuple[int, int, int, int]) -> int:
    visible = 0
    for side in _side_slices(bbox):
        if not bool(np.any(blockers[side])):
            visible += 1
    return visible


def _topological_order(
    nodes: list[int] | tuple[int, ...],
    edges: set[tuple[int, int]],
    *,
    require_unique: bool,
) -> list[int] | None:
    succ = {int(node): set() for node in nodes}
    indeg = {int(node): 0 for node in nodes}
    for a, b in edges:
        a = int(a)
        b = int(b)
        if a == b or b in succ[a]:
            if a == b:
                return None
            continue
        succ[a].add(b)
        indeg[b] += 1

    frontier = sorted(node for node in nodes if indeg[int(node)] == 0)
    order: list[int] = []
    while frontier:
        if require_unique and len(frontier) != 1:
            return None
        node = int(frontier.pop(0))
        order.append(node)
        for nxt in sorted(succ[node]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                frontier.append(nxt)
        frontier.sort()

    if len(order) != len(nodes):
        return None
    return order


def _enumerate_candidates(grid: np.ndarray, color: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    coords = np.argwhere(g == int(color))
    if coords.size == 0:
        return []

    support = g != 0
    rows = coords[:, 0]
    cols = coords[:, 1]
    min_r = int(rows.min())
    max_r = int(rows.max())
    min_c = int(cols.min())
    max_c = int(cols.max())

    candidates: list[dict] = []
    max_h = min(int(h), MAX_SIDE)
    max_w = min(int(w), MAX_SIDE)
    for height in range(MIN_SIDE, max_h + 1):
        r0_lo = max(0, max_r - height + 1)
        r0_hi = min(min_r, h - height)
        if r0_lo > r0_hi:
            continue
        for width in range(MIN_SIDE, max_w + 1):
            c0_lo = max(0, max_c - width + 1)
            c0_hi = min(min_c, w - width)
            if c0_lo > c0_hi:
                continue
            for r0 in range(r0_lo, r0_hi + 1):
                r1 = int(r0 + height - 1)
                inside_rows = (rows >= r0) & (rows <= r1)
                if not bool(np.all(inside_rows)):
                    continue
                for c0 in range(c0_lo, c0_hi + 1):
                    c1 = int(c0 + width - 1)
                    inside_cols = (cols >= c0) & (cols <= c1)
                    if not bool(np.all(inside_cols)):
                        continue

                    on_border = (rows == r0) | (rows == r1) | (cols == c0) | (cols == c1)
                    if not bool(np.all(on_border)):
                        continue

                    bbox = (int(r0), int(c0), int(r1), int(c1))
                    mask = _border_mask(g.shape, bbox)
                    if not bool(np.all(support[mask])):
                        continue
                    if not _has_full_visible_side(g, bbox, int(color)):
                        continue

                    visible_count = int(np.count_nonzero(g[mask] == int(color)))
                    candidates.append(
                        {
                            "bbox": bbox,
                            "mask": mask,
                            "visible_count": visible_count,
                            "hidden_count": int(mask.sum()) - visible_count,
                        }
                    )

    candidates.sort(key=lambda item: (item["hidden_count"], -item["visible_count"], item["bbox"]))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the front-to-back depth order of overlapping hollow rectangles and output their colors in one row.

    Input structure:
    - Black background with 3-5 distinct-colored, axis-aligned hollow rectangular frames.
    - Each frame is a 1-pixel-thick rectangle border whose hidden border cells may be replaced by colors from higher frames at shared cells.
    - Every frame retains at least one fully visible side, so each color still appears somewhere in the input.

    Rule:
    - For each visible color, enumerate every hollow rectangle whose border contains all visible pixels of that color and never passes through a black cell.
    - Choose the global rectangle set whose border overlaps imply a unique acyclic depth order: at any shared cell, the visible color must be above every other rectangle covering that cell.
    - Return a 1xK grid listing the frame colors from frontmost to backmost.

    Key insight:
    - Depth is not determined by position; it must be reconstructed from which color stays visible at border-overlap pixels after each full hollow frame is identified.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    colors = sorted(int(color) for color in np.unique(g) if int(color) != 0)
    assert FRAME_COUNT_MIN <= len(colors) <= FRAME_COUNT_MAX

    coords_by_color = {color: np.argwhere(g == int(color)) for color in colors}
    candidates_by_color = {color: _enumerate_candidates(g, int(color)) for color in colors}
    assert all(candidates_by_color[color] for color in colors)

    ordered_colors = sorted(colors, key=lambda color: len(candidates_by_color[color]))
    valid_orders: set[tuple[int, ...]] = set()

    def search(
        idx: int,
        chosen: dict[int, dict],
        edges: set[tuple[int, int]],
    ) -> None:
        if len(valid_orders) > 1:
            return
        if idx == len(ordered_colors):
            topo = _topological_order(colors, edges, require_unique=True)
            if topo is not None:
                valid_orders.add(tuple(int(color) for color in topo))
            return

        color = int(ordered_colors[idx])
        color_coords = coords_by_color[color]
        for cand in candidates_by_color[color]:
            next_edges = set(edges)
            ok = True
            for other, other_cand in chosen.items():
                if bool(other_cand["mask"][color_coords[:, 0], color_coords[:, 1]].any()):
                    next_edges.add((color, int(other)))

                other_coords = coords_by_color[int(other)]
                if bool(cand["mask"][other_coords[:, 0], other_coords[:, 1]].any()):
                    next_edges.add((int(other), color))

                if (color, int(other)) in next_edges and (int(other), color) in next_edges:
                    ok = False
                    break

            if not ok:
                continue
            partial_nodes = list(chosen) + [color]
            if _topological_order(partial_nodes, next_edges, require_unique=False) is None:
                continue

            chosen[color] = cand
            search(idx + 1, chosen, next_edges)
            del chosen[color]

    search(0, {}, set())
    assert len(valid_orders) == 1
    order = next(iter(valid_orders))
    return np.array([order], dtype=int)


def _sample_bbox(rng: np.random.Generator, shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    h, w = map(int, shape)
    max_h = min(MAX_SIDE, h - 2 * EDGE_MARGIN)
    max_w = min(MAX_SIDE, w - 2 * EDGE_MARGIN)
    if max_h < MIN_SIDE or max_w < MIN_SIDE:
        return None
    height = _randint(rng, MIN_SIDE, max_h)
    width = _randint(rng, MIN_SIDE, max_w)
    r0 = _randint(rng, EDGE_MARGIN, h - EDGE_MARGIN - height)
    c0 = _randint(rng, EDGE_MARGIN, w - EDGE_MARGIN - width)
    return (int(r0), int(c0), int(r0 + height - 1), int(c0 + width - 1))


def _sample_bbox_from_prev(
    rng: np.random.Generator,
    shape: tuple[int, int],
    prev_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    h, w = map(int, shape)
    max_h = min(MAX_SIDE, h - 2 * EDGE_MARGIN)
    max_w = min(MAX_SIDE, w - 2 * EDGE_MARGIN)
    if max_h < MIN_SIDE or max_w < MIN_SIDE:
        return None

    ph0, pw0, ph1, pw1 = map(int, prev_bbox)
    height = _randint(rng, MIN_SIDE, max_h)
    width = _randint(rng, MIN_SIDE, max_w)
    mode = int(rng.integers(0, 3))

    if mode == 0:
        shared_row = int(rng.choice([ph0, ph1]))
        use_top = bool(rng.integers(0, 2))
        r0 = shared_row if use_top else shared_row - height + 1
        r1 = r0 + height - 1
        c0_lo = max(EDGE_MARGIN, pw0 - width + 1)
        c0_hi = min(w - EDGE_MARGIN - width, pw1)
        if r0 < EDGE_MARGIN or r1 >= h - EDGE_MARGIN or c0_lo > c0_hi:
            return None
        c0 = _randint(rng, c0_lo, c0_hi)
        return (int(r0), int(c0), int(r1), int(c0 + width - 1))

    if mode == 1:
        shared_col = int(rng.choice([pw0, pw1]))
        use_left = bool(rng.integers(0, 2))
        c0 = shared_col if use_left else shared_col - width + 1
        c1 = c0 + width - 1
        r0_lo = max(EDGE_MARGIN, ph0 - height + 1)
        r0_hi = min(h - EDGE_MARGIN - height, ph1)
        if c0 < EDGE_MARGIN or c1 >= w - EDGE_MARGIN or r0_lo > r0_hi:
            return None
        r0 = _randint(rng, r0_lo, r0_hi)
        return (int(r0), int(c0), int(r0 + height - 1), int(c1))

    shared_row = int(rng.choice([ph0, ph1]))
    shared_col = int(rng.choice([pw0, pw1]))
    use_top = bool(rng.integers(0, 2))
    use_left = bool(rng.integers(0, 2))
    r0 = shared_row if use_top else shared_row - height + 1
    c0 = shared_col if use_left else shared_col - width + 1
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    if r0 < EDGE_MARGIN or c0 < EDGE_MARGIN or r1 >= h - EDGE_MARGIN or c1 >= w - EDGE_MARGIN:
        return None
    return (int(r0), int(c0), int(r1), int(c1))


def _render_scene(shape: tuple[int, int], layers: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for layer in reversed(layers):
        grid[layer["mask"]] = int(layer["color"])
    return grid


def _direct_visibility_edges(layers: list[dict], shape: tuple[int, int]) -> set[tuple[int, int]]:
    above_union = np.zeros(shape, dtype=bool)
    edges: set[tuple[int, int]] = set()
    for i, upper in enumerate(layers):
        for lower in layers[i + 1 :]:
            witness = upper["mask"] & lower["mask"] & ~above_union
            if bool(np.any(witness)):
                edges.add((int(upper["color"]), int(lower["color"])))
        above_union |= upper["mask"]
    return edges


def _matches_simple_order(layers: list[dict]) -> bool:
    order = tuple(int(layer["color"]) for layer in layers)

    def dims(layer: dict) -> tuple[int, int]:
        r0, c0, r1, c1 = map(int, layer["bbox"])
        return (r1 - r0 + 1, c1 - c0 + 1)

    metric_fns = (
        lambda layer: (layer["bbox"][0], layer["bbox"][1]),
        lambda layer: (layer["bbox"][1], layer["bbox"][0]),
        lambda layer: (layer["bbox"][2], layer["bbox"][0]),
        lambda layer: (layer["bbox"][3], layer["bbox"][1]),
        lambda layer: (dims(layer)[0] * dims(layer)[1], layer["bbox"][0], layer["bbox"][1]),
    )
    for metric in metric_fns:
        asc = tuple(int(layer["color"]) for layer in sorted(layers, key=metric))
        desc = tuple(reversed(asc))
        if order == asc or order == desc:
            return True
    return False


def _valid_scene(
    layers: list[dict],
    grid: np.ndarray,
    *,
    require_indirect: bool,
) -> bool:
    shape = grid.shape
    order = tuple(int(layer["color"]) for layer in layers)
    if len(set(order)) != len(order):
        return False
    if _matches_simple_order(layers):
        return False
    if np.count_nonzero(grid) >= int(grid.size * 0.42):
        return False

    higher_union = np.zeros(shape, dtype=bool)
    for i, layer in enumerate(layers):
        if np.count_nonzero(grid == int(layer["color"])) == 0:
            return False
        if not _has_full_visible_side(grid, layer["bbox"], int(layer["color"])):
            return False
        hidden = int(np.count_nonzero(layer["mask"] & higher_union))
        if i > 0 and hidden == 0:
            return False
        if i > 0 and _visible_side_count(higher_union, layer["bbox"]) == 0:
            return False
        higher_union |= layer["mask"]

    for i in range(1, len(layers)):
        blockers = np.zeros(shape, dtype=bool)
        for layer in layers[: i - 1]:
            blockers |= layer["mask"]
        witness = layers[i - 1]["mask"] & layers[i]["mask"] & ~blockers
        if not bool(np.any(witness)):
            return False

    edges = _direct_visibility_edges(layers, shape)
    topo = _topological_order(list(order), edges, require_unique=True)
    if topo is None or tuple(topo) != order:
        return False

    if require_indirect:
        needs_transitive = False
        for i in range(len(layers)):
            for j in range(i + 2, len(layers)):
                if (int(layers[i]["color"]), int(layers[j]["color"])) not in edges:
                    needs_transitive = True
                    break
            if needs_transitive:
                break
        if not needs_transitive:
            return False

    return True


def _place_next_layer(
    rng: np.random.Generator,
    shape: tuple[int, int],
    layers: list[dict],
    color: int,
) -> dict | None:
    higher_union = np.zeros(shape, dtype=bool)
    for layer in layers:
        higher_union |= layer["mask"]
    prev = layers[-1]
    older_union = np.zeros(shape, dtype=bool)
    for layer in layers[:-1]:
        older_union |= layer["mask"]

    for _ in range(LAYER_PLACEMENT_TRIES):
        if bool(rng.integers(0, 4)):
            bbox = _sample_bbox_from_prev(rng, shape, prev["bbox"])
        else:
            bbox = _sample_bbox(rng, shape)
        if bbox is None:
            continue
        mask = _border_mask(shape, bbox)
        hidden = int(np.count_nonzero(mask & higher_union))
        if hidden == 0:
            continue
        if _visible_side_count(higher_union, bbox) == 0:
            continue
        witness = mask & prev["mask"] & ~older_union
        if not bool(np.any(witness)):
            continue
        visible = int(mask.sum()) - hidden
        if visible < MIN_SIDE:
            continue
        return {"color": int(color), "bbox": bbox, "mask": mask}
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = _GEN_RNG
    height = _randint(rng, GRID_MIN, GRID_MAX)
    width = _randint(rng, GRID_MIN, GRID_MAX)
    shape = (int(height), int(width))

    frame_count = _randint(rng, FRAME_COUNT_MIN, FRAME_COUNT_MAX)
    colors = [int(color) for color in rng.choice(PALETTE, size=frame_count, replace=False)]
    require_indirect = bool(rng.integers(0, 2))

    first_bbox = _sample_bbox(rng, shape)
    if first_bbox is None:
        return None
    layers = [{"color": colors[0], "bbox": first_bbox, "mask": _border_mask(shape, first_bbox)}]

    for color in colors[1:]:
        layer = _place_next_layer(rng, shape, layers, int(color))
        if layer is None:
            return None
        layers.append(layer)

    grid = _render_scene(shape, layers)
    if not _valid_scene(layers, grid, require_indirect=require_indirect):
        return None

    expected = np.array([[int(layer["color"]) for layer in layers]], dtype=int)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid overlapping-frame scene")
