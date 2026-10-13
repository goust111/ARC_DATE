from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _color_coords(grid: np.ndarray, color: int) -> set[Coord]:
    return {tuple(map(int, cell)) for cell in np.argwhere(grid == color)}


def _normalize(coords: set[Coord]) -> tuple[tuple[int, int], ...]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _shape_bbox(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _shifted_cells(shape: tuple[tuple[int, int], ...], anchor: Coord) -> set[Coord]:
    r0, c0 = anchor
    return {(r0 + dr, c0 + dc) for dr, dc in shape}


def _candidate_anchors(
    visible: set[Coord], shape: tuple[tuple[int, int], ...], grid_shape: tuple[int, int]
) -> list[Coord]:
    if not visible:
        return []
    H, W = map(int, grid_shape)
    shape_set = set(shape)
    shape_h, shape_w = _shape_bbox(shape)
    seed = next(iter(visible))
    candidates: list[Coord] = []
    seen: set[Coord] = set()
    for dr, dc in shape:
        anchor = (seed[0] - dr, seed[1] - dc)
        if anchor in seen:
            continue
        seen.add(anchor)
        ar, ac = anchor
        if ar < 0 or ac < 0 or ar + shape_h > H or ac + shape_w > W:
            continue
        if all((vr - ar, vc - ac) in shape_set for vr, vc in visible):
            candidates.append(anchor)
    return candidates


def _topological_order(edges: dict[int, set[int]]) -> list[int]:
    indegree = {node: 0 for node in edges}
    for src, dsts in edges.items():
        for dst in dsts:
            if dst not in indegree:
                indegree[dst] = 0
            indegree[dst] += 1

    remaining = set(indegree)
    order: list[int] = []
    while remaining:
        layer = sorted(node for node in remaining if indegree[node] == 0)
        assert len(layer) == 1, "occlusion constraints must define one total order"
        node = layer[0]
        order.append(node)
        remaining.remove(node)
        for nxt in edges[node]:
            indegree[nxt] -= 1
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the shared sprite placements and redraw the same scene with reversed occlusion depth.

    Input structure:
    - The background is the most common color.
    - Every non-background color is one translated copy of the same sprite shape.
    - The top sprite is fully visible, while lower sprites are only visible as
      fragments because higher sprites overwrite some of their cells.

    Rule:
    - Use the unique fully visible sprite as the common shape template.
    - For each other color, find the unique template translation whose occupied
      cells contain that color's visible fragment.
    - Whenever a template cell expected for one color is occupied by another
      color instead, the occupying color is above it; topologically sort these
      relations from top to bottom, then repaint complete sprites on the same
      canvas with the layer order reversed.

    Key insight:
    - The visible fragment of each color is enough to pin down its hidden full
      sprite position, and the colors sitting inside its missing template cells
      directly encode the original occlusion order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    H, W = g.shape
    bg = _get_bg(g)

    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert len(colors) >= 2, "expected multiple sprite colors"

    coords_by_color = {color: _color_coords(g, color) for color in colors}
    counts = {color: len(coords) for color, coords in coords_by_color.items()}
    max_count = max(counts.values())
    template_colors = [color for color, count in counts.items() if count == max_count]
    assert len(template_colors) == 1, "expected one fully visible top sprite"
    template_color = template_colors[0]
    template_shape = _normalize(coords_by_color[template_color])

    anchors: dict[int, Coord] = {}
    for color in colors:
        visible = coords_by_color[color]
        candidates = _candidate_anchors(visible, template_shape, (H, W))
        assert len(candidates) == 1, "each fragment must determine one template placement"
        anchors[color] = candidates[0]

    edges = {color: set() for color in colors}
    for color in colors:
        for r, c in _shifted_cells(template_shape, anchors[color]):
            occupant = int(g[r, c])
            assert occupant != bg, "every template cell must belong to some sprite"
            if occupant != color:
                edges[occupant].add(color)

    top_to_bottom = _topological_order(edges)

    out = np.full_like(g, bg)
    for color in top_to_bottom:
        for r, c in _shifted_cells(template_shape, anchors[color]):
            out[r, c] = color
    return out


# --------------- generator ---------------

_GEN_MAX_ATTEMPTS = 400
_GEN_GRID_MIN = 12
_GEN_GRID_MAX = 22
_GEN_SPRITE_COUNT_RANGE = (2, 4)
_GEN_SHAPE_H_RANGE = (3, 6)
_GEN_SHAPE_W_RANGE = (3, 6)
_GEN_SHAPE_CELLS_RANGE = (6, 10)
_GEN_VISIBLE_MIN = 3
_GEN_SHIFT_RANGE = (-4, 4)
_GEN_PLACEMENT_TRIES = 180


def _neighbors4(cell: Coord) -> list[Coord]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _sample_shape(max_h: int, max_w: int) -> tuple[tuple[int, int], ...] | None:
    h = random.randint(_GEN_SHAPE_H_RANGE[0], min(_GEN_SHAPE_H_RANGE[1], max_h))
    w = random.randint(_GEN_SHAPE_W_RANGE[0], min(_GEN_SHAPE_W_RANGE[1], max_w))
    target = random.randint(_GEN_SHAPE_CELLS_RANGE[0], min(_GEN_SHAPE_CELLS_RANGE[1], h * w - 1))

    for _ in range(80):
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < target:
            frontier = []
            for cell in cells:
                for nxt in _neighbors4(cell):
                    r, c = nxt
                    if 0 <= r < h and 0 <= c < w and nxt not in cells:
                        frontier.append(nxt)
            if not frontier:
                break
            cells.add(random.choice(frontier))

        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if len(cells) != target or len(rows) < 2 or len(cols) < 2:
            continue
        if len(rows) == h and len(cols) == w and len(cells) == h * w:
            continue
        return _normalize(cells)
    return None


def _render(
    canvas_shape: tuple[int, int],
    bg: int,
    shape: tuple[tuple[int, int], ...],
    anchors: dict[int, Coord],
    draw_order: list[int],
) -> np.ndarray:
    out = np.full(canvas_shape, bg, dtype=int)
    for color in draw_order:
        for r, c in _shifted_cells(shape, anchors[color]):
            out[r, c] = color
    return out


def _expected_output(
    canvas_shape: tuple[int, int],
    bg: int,
    shape: tuple[tuple[int, int], ...],
    anchors: dict[int, Coord],
    top_to_bottom: list[int],
) -> np.ndarray:
    # Drawing original top-to-bottom in overwrite order flips the final depth order.
    return _render(canvas_shape, bg, shape, anchors, top_to_bottom)


def _build_scene() -> tuple[np.ndarray, np.ndarray] | None:
    H = random.randint(_GEN_GRID_MIN, _GEN_GRID_MAX)
    W = random.randint(_GEN_GRID_MIN, _GEN_GRID_MAX)
    bg = random.randint(0, 9)
    sprite_count = random.randint(*_GEN_SPRITE_COUNT_RANGE)
    palette = [color for color in range(10) if color != bg]
    colors = random.sample(palette, sprite_count)

    shape = _sample_shape(H - 2, W - 2)
    if shape is None:
        return None
    shape_h, shape_w = _shape_bbox(shape)

    if H <= shape_h or W <= shape_w:
        return None

    top_anchor = (
        random.randint(0, H - shape_h),
        random.randint(0, W - shape_w),
    )
    top_to_bottom = list(colors)
    anchors: dict[int, Coord] = {top_to_bottom[0]: top_anchor}
    occupied_above = _shifted_cells(shape, top_anchor)

    for idx in range(1, sprite_count):
        color = top_to_bottom[idx]
        prev_cells = _shifted_cells(shape, anchors[top_to_bottom[idx - 1]])
        placed = False
        for _ in range(_GEN_PLACEMENT_TRIES):
            dr = random.randint(*_GEN_SHIFT_RANGE)
            dc = random.randint(*_GEN_SHIFT_RANGE)
            if dr == 0 and dc == 0:
                continue
            anchor = (anchors[top_to_bottom[idx - 1]][0] + dr, anchors[top_to_bottom[idx - 1]][1] + dc)
            if anchor in anchors.values():
                continue
            ar, ac = anchor
            if ar < 0 or ac < 0 or ar + shape_h > H or ac + shape_w > W:
                continue

            cells = _shifted_cells(shape, anchor)
            if not (cells & prev_cells):
                continue

            visible = cells - occupied_above
            if not (_GEN_VISIBLE_MIN <= len(visible) < len(shape)):
                continue

            candidates = _candidate_anchors(visible, shape, (H, W))
            if candidates != [anchor]:
                continue

            anchors[color] = anchor
            occupied_above |= cells
            placed = True
            break
        if not placed:
            return None

    inp = _render((H, W), bg, shape, anchors, list(reversed(top_to_bottom)))
    out = _expected_output((H, W), bg, shape, anchors, top_to_bottom)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        built = _build_scene()
        if built is not None:
            inp, _out = built
            return inp
    raise RuntimeError(f"generate_input: failed after {_GEN_MAX_ATTEMPTS} attempts")
