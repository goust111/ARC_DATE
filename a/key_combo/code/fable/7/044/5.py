import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
OPS = ("intersection", "difference")
MAX_GENERATION_ATTEMPTS = 400
PANEL_RENDER_ATTEMPTS = 80


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _normalized_shape(coords) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _extract_components(panel: np.ndarray) -> list[dict]:
    seen = np.zeros(panel.shape, dtype=bool)
    components = []
    h, w = panel.shape
    for r in range(h):
        for c in range(w):
            color = int(panel[r, c])
            if color == BLACK or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not _in_bounds(panel.shape, nr, nc):
                        continue
                    if seen[nr, nc] or int(panel[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "shape": _normalized_shape(coords),
                }
            )
    return components


def _apply_operation(left: np.ndarray, right: np.ndarray, op: str) -> np.ndarray:
    left_components = _extract_components(left)
    right_shapes = {comp["shape"] for comp in _extract_components(right)}
    out = np.zeros_like(left)
    for comp in left_components:
        in_right = comp["shape"] in right_shapes
        keep = in_right if op == "intersection" else not in_right
        if not keep:
            continue
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = int(comp["color"])
    return out


def _infer_operation(panel_a: np.ndarray, panel_b: np.ndarray, panel_r: np.ndarray) -> str:
    matches = [op for op in OPS if np.array_equal(_apply_operation(panel_a, panel_b, op), panel_r)]
    assert len(matches) == 1, f"expected one matching operation, got {matches}"
    return matches[0]


def _parse_panels(grid: np.ndarray) -> tuple[list[list[np.ndarray]], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_candidates = []
    for r in range(g.shape[0]):
        row = g[r, :]
        if np.all(row == row[0]) and int(row[0]) != BLACK:
            row_candidates.append((r, int(row[0])))

    col_candidates = []
    for c in range(g.shape[1]):
        col = g[:, c]
        if np.all(col == col[0]) and int(col[0]) != BLACK:
            col_candidates.append((c, int(col[0])))

    row_colors = {color for _, color in row_candidates}
    col_colors = {color for _, color in col_candidates}
    separator_colors = row_colors & col_colors
    assert separator_colors, "missing separator color"

    separator_color = None
    for color in separator_colors:
        rows = [idx for idx, c in row_candidates if c == color]
        cols = [idx for idx, c in col_candidates if c == color]
        if len(rows) == 1 and len(cols) == 2:
            separator_color = color
            break
    assert separator_color is not None, "failed to locate 2x3 panel separators"

    sep_row = [idx for idx, c in row_candidates if c == separator_color][0]
    sep_cols = sorted(idx for idx, c in col_candidates if c == separator_color)
    c0, c1 = sep_cols

    row_slices = [(0, sep_row), (sep_row + 1, g.shape[0])]
    col_slices = [(0, c0), (c0 + 1, c1), (c1 + 1, g.shape[1])]
    panel_h = row_slices[0][1] - row_slices[0][0]
    panel_w = col_slices[0][1] - col_slices[0][0]
    assert panel_h > 0 and panel_w > 0
    assert row_slices[1][1] - row_slices[1][0] == panel_h
    assert col_slices[1][1] - col_slices[1][0] == panel_w
    assert col_slices[2][1] - col_slices[2][0] == panel_w

    panels = []
    for r0, r1 in row_slices:
        row_panels = []
        for cc0, cc1 in col_slices:
            row_panels.append(g[r0:r1, cc0:cc1].copy())
        panels.append(row_panels)
    return panels, separator_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn an object-level shape-set operation from the completed top-row panels, then apply it to the bottom-row pair.

    Input structure:
    - The grid is split by uniform grey separators into a 2x3 arrangement of equal-size panels.
    - The top row contains panel A, panel B, and the completed result of applying a rule to A using B.
    - The bottom row contains panel C, panel D, and one empty black panel.
    - Each non-empty panel has a black background with several small disconnected monochrome objects.

    Rule:
    - Compare objects by translation-normalized shape rather than absolute position.
    - The top result keeps exactly those objects from A whose shapes either do appear in B (intersection) or do not appear in B (difference).
    - Infer which of those two object-level set operations is shown by the top row.
    - Apply the same operation to C using D, preserving the kept objects' original positions and colors from C.

    Key insight:
    - The analogy works on sets of normalized object shapes, not on aligned pixels: B acts only as a membership test for which shapes from the left panel survive.
    """
    panels, _separator_color = _parse_panels(np.asarray(grid, dtype=int))
    panel_a, panel_b, panel_r = panels[0]
    panel_c, panel_d, panel_blank = panels[1]
    assert np.all(panel_blank == BLACK), "expected the bottom-right panel to be blank"
    op = _infer_operation(panel_a, panel_b, panel_r)
    return _apply_operation(panel_c, panel_d, op)


def _transform_shape(shape: tuple[tuple[int, int], ...], rotation: int, reflect: bool) -> tuple[tuple[int, int], ...]:
    pts = np.array(shape, dtype=int)
    if reflect:
        pts = pts[:, [0, 1]]
        pts[:, 1] *= -1
    for _ in range(rotation % 4):
        pts = np.stack([pts[:, 1], -pts[:, 0]], axis=1)
    return _normalized_shape(pts.tolist())


def _random_shape(rng: random.Random) -> tuple[tuple[int, int], ...]:
    target_area = rng.randint(2, 5)
    for _ in range(80):
        cells = {(0, 0)}
        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            cells.add(rng.choice(tuple(frontier)))
        shape = _normalized_shape(cells)
        shape = _transform_shape(shape, rng.randrange(4), bool(rng.randrange(2)))
        rows = [r for r, _ in shape]
        cols = [c for _, c in shape]
        if max(rows) + 1 <= 4 and max(cols) + 1 <= 4:
            return shape
    raise RuntimeError("failed to generate a compact shape")


def _sample_distinct_shapes(
    rng: random.Random,
    count: int,
    forbidden: set[tuple[tuple[int, int], ...]] | None = None,
) -> list[tuple[tuple[int, int], ...]]:
    forbidden = set() if forbidden is None else set(forbidden)
    shapes = []
    used = set(forbidden)
    while len(shapes) < count:
        shape = _random_shape(rng)
        if shape in used:
            continue
        used.add(shape)
        shapes.append(shape)
    return shapes


def _pick_pair_shape_sets(
    rng: random.Random,
    op: str,
    forbidden: set[tuple[tuple[int, int], ...]] | None = None,
) -> tuple[list[tuple[tuple[int, int], ...]], list[tuple[tuple[int, int], ...]]]:
    forbidden = set() if forbidden is None else set(forbidden)
    left_count = rng.randint(4, 5)
    right_count = rng.randint(3, 5)
    overlap_max = min(left_count - 1, right_count - 1)
    overlap = rng.randint(1, overlap_max)
    left_shapes = _sample_distinct_shapes(rng, left_count, forbidden=forbidden)
    common = rng.sample(left_shapes, overlap)
    extra_right = _sample_distinct_shapes(rng, right_count - overlap, forbidden=forbidden | set(left_shapes))
    right_shapes = common + extra_right
    rng.shuffle(right_shapes)
    kept = set(common) if op == "intersection" else set(left_shapes) - set(common)
    assert 0 < len(kept) < len(left_shapes)
    return left_shapes, right_shapes


def _render_panel(
    rng: random.Random,
    panel_h: int,
    panel_w: int,
    shapes: list[tuple[tuple[int, int], ...]],
    exact_position_forbidden: dict[tuple[tuple[int, int], ...], frozenset[tuple[int, int]]] | None = None,
) -> tuple[np.ndarray, dict[tuple[tuple[int, int], ...], frozenset[tuple[int, int]]]] | None:
    exact_position_forbidden = {} if exact_position_forbidden is None else exact_position_forbidden
    pairs = list(zip(shapes, rng.sample(PALETTE, len(shapes))))
    pairs.sort(key=lambda item: (-len(item[0]), item[0]))

    for _ in range(PANEL_RENDER_ATTEMPTS):
        panel = np.zeros((panel_h, panel_w), dtype=int)
        blocked = np.zeros((panel_h, panel_w), dtype=bool)
        placed = {}
        ok = True

        for shape, color in pairs:
            shape_h = 1 + max(r for r, _ in shape)
            shape_w = 1 + max(c for _, c in shape)
            anchors = [
                (r0, c0)
                for r0 in range(panel_h - shape_h + 1)
                for c0 in range(panel_w - shape_w + 1)
            ]
            rng.shuffle(anchors)
            chosen_cells = None
            for r0, c0 in anchors:
                cells = [(r0 + dr, c0 + dc) for dr, dc in shape]
                if any(blocked[r, c] for r, c in cells):
                    continue
                if shape in exact_position_forbidden and frozenset(cells) == exact_position_forbidden[shape]:
                    continue
                chosen_cells = cells
                break
            if chosen_cells is None:
                ok = False
                break

            cell_set = frozenset(chosen_cells)
            placed[shape] = cell_set
            for r, c in chosen_cells:
                panel[r, c] = color
                blocked[r, c] = True
            for r, c in chosen_cells:
                for nr, nc in _neighbors8(r, c):
                    if _in_bounds(panel.shape, nr, nc):
                        blocked[nr, nc] = True

        if ok:
            return panel, placed
    return None


def _assemble_grid(
    top_left: np.ndarray,
    top_mid: np.ndarray,
    top_right: np.ndarray,
    bottom_left: np.ndarray,
    bottom_mid: np.ndarray,
    separator_color: int,
) -> np.ndarray:
    panel_h, panel_w = top_left.shape
    grid = np.full((panel_h * 2 + 1, panel_w * 3 + 2), separator_color, dtype=int)
    row0 = slice(0, panel_h)
    row1 = slice(panel_h + 1, panel_h * 2 + 1)
    col0 = slice(0, panel_w)
    col1 = slice(panel_w + 1, panel_w * 2 + 1)
    col2 = slice(panel_w * 2 + 2, panel_w * 3 + 2)
    grid[row0, col0] = top_left
    grid[row0, col1] = top_mid
    grid[row0, col2] = top_right
    grid[row1, col0] = bottom_left
    grid[row1, col1] = bottom_mid
    grid[row1, col2] = BLACK
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        op = rng.choice(OPS)
        panel_h = rng.randint(8, 12)
        panel_w = rng.randint(7, 9)

        top_left_shapes, top_mid_shapes = _pick_pair_shape_sets(rng, op)
        bottom_left_shapes, bottom_mid_shapes = _pick_pair_shape_sets(
            rng,
            op,
            forbidden=set(top_left_shapes) | set(top_mid_shapes),
        )

        top_left_render = _render_panel(rng, panel_h, panel_w, top_left_shapes)
        if top_left_render is None:
            continue
        top_left_panel, top_left_pos = top_left_render

        top_mid_render = _render_panel(
            rng,
            panel_h,
            panel_w,
            top_mid_shapes,
            exact_position_forbidden=top_left_pos,
        )
        if top_mid_render is None:
            continue
        top_mid_panel, top_mid_pos = top_mid_render

        bottom_left_render = _render_panel(rng, panel_h, panel_w, bottom_left_shapes)
        if bottom_left_render is None:
            continue
        bottom_left_panel, bottom_left_pos = bottom_left_render

        bottom_mid_render = _render_panel(
            rng,
            panel_h,
            panel_w,
            bottom_mid_shapes,
            exact_position_forbidden=bottom_left_pos,
        )
        if bottom_mid_render is None:
            continue
        bottom_mid_panel, _bottom_mid_pos = bottom_mid_render

        top_right_panel = _apply_operation(top_left_panel, top_mid_panel, op)
        bottom_result = _apply_operation(bottom_left_panel, bottom_mid_panel, op)

        if not np.any(top_right_panel) or np.array_equal(top_right_panel, top_left_panel):
            continue
        if not np.any(bottom_result) or np.array_equal(bottom_result, bottom_left_panel):
            continue

        grid = _assemble_grid(
            top_left_panel,
            top_mid_panel,
            top_right_panel,
            bottom_left_panel,
            bottom_mid_panel,
            separator_color=GREY,
        )

        try:
            solved = main(grid)
        except AssertionError:
            continue
        if np.array_equal(solved, bottom_result):
            return grid
    raise RuntimeError("failed to generate a valid input grid")
