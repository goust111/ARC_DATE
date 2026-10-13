from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


LineSpec = tuple[str, int]
Coord = tuple[int, int]

DIRS = ("H", "V", "D", "A")  # row, column, r-c, r+c
MIN_LINE_LENGTH = 7
GRID_SIZE_RANGE = (14, 30)
LINE_COUNT_RANGE = (3, 6)
GENERATION_ATTEMPTS = 600


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _line_coords(spec: LineSpec, shape: tuple[int, int]) -> list[Coord]:
    kind, key = spec
    H, W = shape
    coords: list[Coord] = []
    if kind == "H":
        if 0 <= key < H:
            coords = [(key, c) for c in range(W)]
    elif kind == "V":
        if 0 <= key < W:
            coords = [(r, key) for r in range(H)]
    elif kind == "D":
        coords = [(r, c) for r in range(H) for c in range(W) if r - c == key]
    elif kind == "A":
        coords = [(r, c) for r in range(H) for c in range(W) if r + c == key]
    else:
        raise ValueError(f"unknown line kind {kind!r}")
    return coords


def _infer_line(coords: list[Coord]) -> LineSpec:
    assert len(coords) >= 2, "each visible line must contain enough cells to define a unique line"
    r0, c0 = coords[0]
    candidates: list[LineSpec] = []
    if all(r == r0 for r, _c in coords):
        candidates.append(("H", int(r0)))
    if all(c == c0 for _r, c in coords):
        candidates.append(("V", int(c0)))
    d0 = r0 - c0
    if all(r - c == d0 for r, c in coords):
        candidates.append(("D", int(d0)))
    a0 = r0 + c0
    if all(r + c == a0 for r, c in coords):
        candidates.append(("A", int(a0)))
    assert len(candidates) == 1, "visible pixels must be collinear in exactly one allowed direction"
    return candidates[0]


def _recover_lines(grid: np.ndarray, bg: int) -> dict[int, LineSpec]:
    lines: dict[int, LineSpec] = {}
    for color in sorted(set(map(int, grid.reshape(-1))) - {int(bg)}):
        pts = [tuple(map(int, p)) for p in np.argwhere(grid == int(color))]
        spec = _infer_line(pts)
        full = _line_coords(spec, grid.shape)
        assert len(full) >= len(pts), "recovered line must fit the canvas"
        lines[int(color)] = spec

    assert len(lines) >= 2, "expected at least two foreground lines"
    assert len(set(lines.values())) == len(lines), "two colors may not describe the same full line"
    return lines


def _cell_members(lines: dict[int, LineSpec], shape: tuple[int, int]) -> dict[Coord, list[int]]:
    members: dict[Coord, list[int]] = defaultdict(list)
    for color, spec in lines.items():
        for rc in _line_coords(spec, shape):
            members[rc].append(int(color))
    return members


def _occlusion_edges(grid: np.ndarray, lines: dict[int, LineSpec]) -> dict[int, set[int]]:
    edges: dict[int, set[int]] = {int(c): set() for c in lines}
    members = _cell_members(lines, grid.shape)
    for (r, c), colors in members.items():
        if len(colors) < 2:
            continue
        visible = int(grid[r, c])
        assert visible in colors, "a crossing cell must show one of the crossing lines"
        for color in colors:
            if int(color) != visible:
                edges[int(color)].add(visible)
    return edges


def _unique_topological_order(colors: list[int], edges: dict[int, set[int]]) -> list[int]:
    indeg = {int(c): 0 for c in colors}
    for src in colors:
        for dst in edges.get(int(src), set()):
            indeg[int(dst)] += 1

    ready = [int(c) for c in colors if indeg[int(c)] == 0]
    order: list[int] = []
    while ready:
        assert len(ready) == 1, "occlusion evidence must determine one layer chain"
        cur = ready.pop()
        order.append(cur)
        for nxt in sorted(edges.get(cur, set())):
            indeg[int(nxt)] -= 1
            if indeg[int(nxt)] == 0:
                ready.append(int(nxt))

    assert len(order) == len(colors), "occlusion relations must be acyclic and include all lines"
    return order


def _layer_order(grid: np.ndarray, lines: dict[int, LineSpec]) -> list[int]:
    return _unique_topological_order(list(lines), _occlusion_edges(grid, lines))


def _render_lines(
    shape: tuple[int, int],
    bg: int,
    lines: dict[int, LineSpec],
    bottom_to_top: list[int],
    colors_after: dict[int, int] | None = None,
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for original_color in bottom_to_top:
        paint = int(colors_after.get(original_color, original_color) if colors_after else original_color)
        for r, c in _line_coords(lines[int(original_color)], shape):
            out[int(r), int(c)] = paint
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the hidden layer order of overlapping lines and rotate their colors once upward.

    Input structure:
    - The dominant color is the background.
    - Every non-background color is the visible part of one full one-cell-thick
      horizontal, vertical, or 45-degree diagonal line across the grid.
    - At crossings, the upper line hides the lower line, and the visible
      crossing colors determine one unique bottom-to-top layer chain.

    Rule:
    - Recover each full straight line from the collinear visible cells of its color.
    - At every crossing, add the occlusion relation that the visible line is above
      the other crossing lines, then topologically recover the unique layer chain.
    - Rotate colors along that chain: each line receives the color of the line one
      layer above it, while the top line receives the bottom line's color.
    - Keep the same geometry and original layer order, and re-render all full lines
      with the rotated colors.

    Key insight:
    - The color cycle is not ordered by position or direction; the usable order is
      the hidden occlusion chain reconstructed from crossing evidence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"

    bg = _background(g)
    lines = _recover_lines(g, bg)
    order = _layer_order(g, lines)

    rotated = {color: order[(i + 1) % len(order)] for i, color in enumerate(order)}
    return _render_lines(g.shape, bg, lines, order, rotated)


def _all_specs(H: int, W: int) -> list[LineSpec]:
    specs: list[LineSpec] = []
    for r in range(H):
        specs.append(("H", int(r)))
    for c in range(W):
        specs.append(("V", int(c)))
    for d in range(-(W - 1), H):
        specs.append(("D", int(d)))
    for a in range(H + W - 1):
        specs.append(("A", int(a)))
    return [s for s in specs if len(_line_coords(s, (H, W))) >= MIN_LINE_LENGTH]


def _intersects(a: LineSpec, b: LineSpec, shape: tuple[int, int]) -> bool:
    return bool(set(_line_coords(a, shape)) & set(_line_coords(b, shape)))


def _max_overlap(specs: list[LineSpec], shape: tuple[int, int]) -> int:
    counts: dict[Coord, int] = defaultdict(int)
    best = 0
    for spec in specs:
        for rc in _line_coords(spec, shape):
            counts[rc] += 1
            best = max(best, counts[rc])
    return best


def _spec_center(spec: LineSpec, shape: tuple[int, int]) -> tuple[float, float]:
    pts = _line_coords(spec, shape)
    return (
        sum(r for r, _c in pts) / len(pts),
        sum(c for _r, c in pts) / len(pts),
    )


def _select_line_chain(H: int, W: int, n: int) -> list[LineSpec] | None:
    shape = (H, W)
    universe = _all_specs(H, W)
    if not universe:
        return None

    for _ in range(120):
        selected = [random.choice(universe)]
        while len(selected) < n:
            prev = selected[-1]
            candidates = [
                s
                for s in universe
                if s not in selected
                and _intersects(prev, s, shape)
                and _max_overlap(selected + [s], shape) <= 2
            ]
            if not candidates:
                break

            used_dirs = {s[0] for s in selected}
            missing_diag = not ({"D", "A"} & used_dirs)
            if missing_diag:
                diag_candidates = [s for s in candidates if s[0] in {"D", "A"}]
                if diag_candidates:
                    candidates = diag_candidates
            elif len(used_dirs) < 3:
                new_dir_candidates = [s for s in candidates if s[0] not in used_dirs]
                if new_dir_candidates:
                    candidates = new_dir_candidates

            selected.append(random.choice(candidates))

        if len(selected) != n:
            continue
        dirs = {s[0] for s in selected}
        if len(dirs) < 3 or not ({"D", "A"} & dirs):
            continue
        if all(_intersects(a, b, shape) for a, b in zip(selected, selected[1:])):
            return selected
    return None


def _spatial_orders_are_decoys(lines: dict[int, LineSpec], layer_order: list[int], shape: tuple[int, int]) -> bool:
    colors = list(layer_order)
    by_row_col = sorted(colors, key=lambda c: (*_spec_center(lines[c], shape), lines[c][0], lines[c][1]))
    by_col_row = sorted(colors, key=lambda c: (_spec_center(lines[c], shape)[1], _spec_center(lines[c], shape)[0], lines[c][0], lines[c][1]))
    by_spec = sorted(colors, key=lambda c: (DIRS.index(lines[c][0]), lines[c][1]))
    return all(candidate != layer_order for candidate in (by_row_col, by_col_row, by_spec))


def _visible_counts_ok(grid: np.ndarray, bg: int) -> bool:
    for color in set(map(int, grid.reshape(-1))) - {int(bg)}:
        pts = np.argwhere(grid == int(color))
        if int(pts.shape[0]) < 3:
            return False
        try:
            _infer_line([tuple(map(int, p)) for p in pts])
        except AssertionError:
            return False
    return True


def _build_candidate() -> np.ndarray | None:
    H = random.randint(*GRID_SIZE_RANGE)
    W = random.randint(*GRID_SIZE_RANGE)
    n = random.randint(*LINE_COUNT_RANGE)
    if n >= 5 and min(H, W) < 17:
        return None

    specs = _select_line_chain(H, W, n)
    if specs is None:
        return None

    bg = random.randrange(10)
    colors = random.sample([c for c in range(10) if c != bg], n)
    if colors == sorted(colors) or colors == sorted(colors, reverse=True):
        return None

    lines = {int(color): spec for color, spec in zip(colors, specs)}
    grid = _render_lines((H, W), bg, lines, colors)

    if _background(grid) != bg:
        return None
    if not _visible_counts_ok(grid, bg):
        return None

    try:
        recovered = _recover_lines(grid, bg)
        inferred_order = _layer_order(grid, recovered)
        out = main(grid)
    except AssertionError:
        return None

    if inferred_order != colors:
        return None
    if not _spatial_orders_are_decoys(recovered, inferred_order, grid.shape):
        return None
    if np.array_equal(out, grid):
        return None

    # Adjacent inferred layers must have direct crossing evidence, not just transitive order.
    for lower, upper in zip(inferred_order, inferred_order[1:]):
        a = set(_line_coords(recovered[lower], grid.shape))
        b = set(_line_coords(recovered[upper], grid.shape))
        if not (a & b):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
