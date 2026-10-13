import hashlib
import math
import random

import numpy as np


GREY = 5
BLACK = 0
SPRITE_COLORS = (1, 2, 3, 4, 7, 8)
SQUARE_AREAS = (4, 9, 16)
NON_SQUARE_AREAS = tuple(area for area in range(3, 17) if area not in SQUARE_AREAS)
MAX_GENERATION_ATTEMPTS = 600

_RECENT_OUTPUT_SIGNATURES: set[str] = set()


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _separator_indices(grid: np.ndarray, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]


def _spans_from_separators(seps: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for a, b in zip(seps, seps[1:]):
        if b - a <= 1:
            raise AssertionError("adjacent lattice separators are not allowed")
        spans.append((a + 1, b - 1))
    return spans


def _extract_cells(grid: np.ndarray) -> tuple[list[dict], int, int]:
    g = np.asarray(grid, dtype=int)
    row_seps = _separator_indices(g, axis=0)
    col_seps = _separator_indices(g, axis=1)
    assert row_seps and col_seps, "missing lattice separators"
    assert row_seps[0] == 0 and row_seps[-1] == g.shape[0] - 1, "lattice must close vertically"
    assert col_seps[0] == 0 and col_seps[-1] == g.shape[1] - 1, "lattice must close horizontally"

    row_spans = _spans_from_separators(row_seps)
    col_spans = _spans_from_separators(col_seps)
    assert len(row_spans) >= 4 and len(col_spans) >= 4, "expected at least a 4x4 cell array"

    cells: list[dict] = []
    for row_idx, (r0, r1) in enumerate(row_spans):
        for col_idx, (c0, c1) in enumerate(col_spans):
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            assert not np.any(patch == GREY), "cell interior may not contain lattice pixels"
            colors = [int(color) for color in np.unique(patch) if int(color) != BLACK]
            assert len(colors) == 1, "each cell must contain one monochrome sprite"
            color = colors[0]
            mask = patch == color
            comps = _components(mask)
            assert len(comps) == 1, "sprite must be 4-connected"
            area = int(mask.sum())
            assert 3 <= area <= 16
            cells.append(
                {
                    "row": row_idx,
                    "col": col_idx,
                    "color": color,
                    "area": area,
                }
            )
    return cells, len(row_spans), len(col_spans)


def _render_output(specs: list[dict]) -> np.ndarray:
    ordered = sorted(specs, key=lambda item: (item["side"], item["col"], item["row"]))
    return _paint_ordered_squares(ordered)


def _paint_ordered_squares(ordered: list[dict]) -> np.ndarray:
    height = max(item["side"] for item in ordered)
    width = sum(item["side"] for item in ordered) + len(ordered) - 1
    out = np.zeros((height, width), dtype=int)
    cursor = 0
    for idx, item in enumerate(ordered):
        side = item["side"]
        out[height - side : height, cursor : cursor + side] = item["color"]
        cursor += side
        if idx != len(ordered) - 1:
            cursor += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop perimeter-cell sprites from a grey lattice and repack interior sprites as sorted solid squares.

    Input structure:
    - Grey pixels form complete single-pixel lattice rows and columns that split the grid into a rectangular array of black-backed cells.
    - Every cell interior contains exactly one 4-connected monochrome sprite in a non-grey, non-black color.
    - Interior cells are the cells whose lattice row and lattice column are both strictly inside the array boundary.

    Rule:
    - Detect the lattice row and column separators and assign each cell its array row and column index.
    - Discard every sprite from the perimeter cells of that array.
    - For each remaining interior sprite, count its pixels, replace it by a filled square with side length equal to the square root of that count, and keep its color.
    - Place the output squares in one bottom-aligned row, ordered by increasing side length; break equal-size ties by original cell column first and original cell row second.

    Key insight:
    - The lattice position determines which sprites survive, while the surviving sprites' pixel counts independently determine both the square sizes and the final packing order.
    """
    cells, n_rows, n_cols = _extract_cells(grid)
    kept: list[dict] = []
    for cell in cells:
        if cell["row"] in (0, n_rows - 1) or cell["col"] in (0, n_cols - 1):
            continue
        side = math.isqrt(cell["area"])
        assert side * side == cell["area"], "interior sprite areas must be perfect squares"
        kept.append(
            {
                "row": cell["row"],
                "col": cell["col"],
                "color": cell["color"],
                "side": side,
            }
        )

    assert len(kept) >= 3, "expected at least three interior sprites"
    return _render_output(kept)


def _cell_origin(row: int, col: int, cell_h: int, cell_w: int) -> tuple[int, int]:
    return 1 + row * (cell_h + 1), 1 + col * (cell_w + 1)


def _shape_signature(mask: np.ndarray) -> bytes:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
    return cropped.astype(np.uint8).tobytes() + bytes(cropped.shape)


def _random_connected_shape(cell_h: int, cell_w: int, area: int) -> np.ndarray:
    if area == cell_h * cell_w and random.random() < 0.35:
        return np.ones((cell_h, cell_w), dtype=bool)

    for _ in range(240):
        seed = (random.randrange(cell_h), random.randrange(cell_w))
        cells = {seed}
        frontier = {
            nbr
            for nbr in _neighbors4(*seed)
            if 0 <= nbr[0] < cell_h and 0 <= nbr[1] < cell_w
        }
        while len(cells) < area:
            if not frontier:
                break
            choices = list(frontier)
            weights = []
            for r, c in choices:
                adj = sum((nr, nc) in cells for nr, nc in _neighbors4(r, c))
                center_bias = (cell_h - abs(2 * r - cell_h + 1)) + (cell_w - abs(2 * c - cell_w + 1))
                weights.append(max(1, 3 * adj + center_bias))
            pick = random.choices(choices, weights=weights, k=1)[0]
            frontier.remove(pick)
            cells.add(pick)
            for nbr in _neighbors4(*pick):
                if 0 <= nbr[0] < cell_h and 0 <= nbr[1] < cell_w and nbr not in cells:
                    frontier.add(nbr)
        if len(cells) != area:
            continue
        mask = np.zeros((cell_h, cell_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        coords = np.argwhere(mask)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if area > 1 and (r1 == r0 or c1 == c0):
            continue
        if len(_components(mask)) != 1:
            continue
        return mask

    if area == cell_h * cell_w:
        return np.ones((cell_h, cell_w), dtype=bool)

    mask = np.zeros((cell_h, cell_w), dtype=bool)
    r = max(0, (cell_h - 2) // 2)
    c = max(0, (cell_w - 2) // 2)
    for rr in range(cell_h):
        for cc in range(cell_w):
            if abs(rr - r) + abs(cc - c) <= cell_h + cell_w:
                mask[rr, cc] = True
                if int(mask.sum()) == area:
                    return mask
    raise RuntimeError("failed to build a sprite shape")


def _layout_options() -> list[tuple[int, int, int, int]]:
    options: list[tuple[int, int, int, int]] = []
    for n_rows in range(4, 6):
        for n_cols in range(4, 6):
            if (n_rows - 2) * (n_cols - 2) > 6:
                continue
            for cell_h in range(4, 7):
                for cell_w in range(4, 7):
                    height = n_rows * cell_h + (n_rows + 1)
                    width = n_cols * cell_w + (n_cols + 1)
                    if height <= 30 and width <= 30:
                        options.append((n_rows, n_cols, cell_h, cell_w))
    return options


LAYOUT_OPTIONS = _layout_options()


def _expected_from_specs(specs: list[dict], n_rows: int, n_cols: int) -> np.ndarray:
    kept = [
        {
            "row": spec["row"],
            "col": spec["col"],
            "color": spec["color"],
            "side": int(math.isqrt(spec["area"])),
        }
        for spec in specs
        if 0 < spec["row"] < n_rows - 1 and 0 < spec["col"] < n_cols - 1
    ]
    return _render_output(kept)


def _render_wrong_row_major(specs: list[dict], n_rows: int, n_cols: int) -> np.ndarray:
    kept = [
        {
            "row": spec["row"],
            "col": spec["col"],
            "color": spec["color"],
            "side": int(math.isqrt(spec["area"])),
        }
        for spec in specs
        if 0 < spec["row"] < n_rows - 1 and 0 < spec["col"] < n_cols - 1
    ]
    ordered = sorted(kept, key=lambda item: (item["side"], item["row"], item["col"]))
    return _paint_ordered_squares(ordered)


def _render_wrong_position_only(specs: list[dict], n_rows: int, n_cols: int) -> np.ndarray:
    kept = [
        {
            "row": spec["row"],
            "col": spec["col"],
            "color": spec["color"],
            "side": int(math.isqrt(spec["area"])),
        }
        for spec in specs
        if 0 < spec["row"] < n_rows - 1 and 0 < spec["col"] < n_cols - 1
    ]
    ordered = sorted(kept, key=lambda item: (item["col"], item["row"]))
    return _paint_ordered_squares(ordered)


def _render_wrong_keep_all_squares(specs: list[dict]) -> np.ndarray:
    kept = [
        {
            "row": spec["row"],
            "col": spec["col"],
            "color": spec["color"],
            "side": int(math.isqrt(spec["area"])),
        }
        for spec in specs
        if int(math.isqrt(spec["area"])) ** 2 == spec["area"]
    ]
    return _render_output(kept)


def _build_specs(n_rows: int, n_cols: int) -> tuple[list[dict], bool]:
    specs: list[dict] = []
    internal_rows = list(range(1, n_rows - 1))
    internal_cols = list(range(1, n_cols - 1))
    large_pos = (internal_rows[0], internal_cols[0])
    tie_a_pos = (internal_rows[0], internal_cols[-1])
    tie_b_pos = (internal_rows[-1], internal_cols[0])
    small_pos = (internal_rows[-1], internal_cols[-1])
    designated = {
        large_pos: 16,
        tie_a_pos: 9,
        tie_b_pos: 9,
        small_pos: 4,
    }
    designated_colors = random.sample(SPRITE_COLORS, 4)
    tie_colors = designated_colors[1], designated_colors[2]
    square_perimeter_branch = random.random() < 0.6
    perimeter_square_positions = set()
    perimeter_positions = [
        (r, c)
        for r in range(n_rows)
        for c in range(n_cols)
        if r in (0, n_rows - 1) or c in (0, n_cols - 1)
    ]
    if square_perimeter_branch:
        perimeter_square_positions.add(random.choice(perimeter_positions))

    for r in range(n_rows):
        for c in range(n_cols):
            if (r, c) in designated:
                idx = [large_pos, tie_a_pos, tie_b_pos, small_pos].index((r, c))
                color = designated_colors[idx]
                if (r, c) == tie_b_pos and color == tie_colors[0]:
                    color = tie_colors[1]
                specs.append({"row": r, "col": c, "area": designated[(r, c)], "color": color})
                continue
            if 0 < r < n_rows - 1 and 0 < c < n_cols - 1:
                area = random.choice(SQUARE_AREAS)
            elif (r, c) in perimeter_square_positions:
                area = random.choice((4, 9, 16))
            else:
                area = random.choice(NON_SQUARE_AREAS)
            color = random.choice(SPRITE_COLORS)
            specs.append({"row": r, "col": c, "area": area, "color": color})
    assert any(spec["color"] != tie_colors[0] for spec in specs if (spec["row"], spec["col"]) == tie_b_pos)
    return specs, square_perimeter_branch


def _place_specs_on_grid(
    specs: list[dict], n_rows: int, n_cols: int, cell_h: int, cell_w: int
) -> np.ndarray | None:
    height = n_rows * cell_h + (n_rows + 1)
    width = n_cols * cell_w + (n_cols + 1)
    grid = np.zeros((height, width), dtype=int)
    grid[0, :] = GREY
    grid[-1, :] = GREY
    grid[:, 0] = GREY
    grid[:, -1] = GREY
    for row in range(1, n_rows):
        grid[row * (cell_h + 1), :] = GREY
    for col in range(1, n_cols):
        grid[:, col * (cell_w + 1)] = GREY

    seen_shapes: dict[int, set[bytes]] = {area: set() for area in range(3, 17)}
    for spec in specs:
        shape = _random_connected_shape(cell_h, cell_w, spec["area"])
        sig = _shape_signature(shape)
        if len(seen_shapes[spec["area"]]) < 3 and sig in seen_shapes[spec["area"]]:
            alt_shape = _random_connected_shape(cell_h, cell_w, spec["area"])
            shape = alt_shape
            sig = _shape_signature(shape)
        seen_shapes[spec["area"]].add(sig)
        r0, c0 = _cell_origin(spec["row"], spec["col"], cell_h, cell_w)
        patch = grid[r0 : r0 + cell_h, c0 : c0 + cell_w]
        patch[shape] = spec["color"]
    return grid


def _semantic_checks(
    grid: np.ndarray,
    specs: list[dict],
    n_rows: int,
    n_cols: int,
    square_perimeter_branch: bool,
) -> bool:
    extracted, erows, ecols = _extract_cells(grid)
    if erows != n_rows or ecols != n_cols:
        return False

    spec_map = {(spec["row"], spec["col"]): spec for spec in specs}
    for cell in extracted:
        spec = spec_map[(cell["row"], cell["col"])]
        if cell["area"] != spec["area"] or cell["color"] != spec["color"]:
            return False

    interior = [spec for spec in specs if 0 < spec["row"] < n_rows - 1 and 0 < spec["col"] < n_cols - 1]
    if len(interior) < 3:
        return False
    if len({spec["area"] for spec in interior}) < 2:
        return False

    expected = _expected_from_specs(specs, n_rows, n_cols)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if max(actual.shape) > 30:
        return False

    if np.array_equal(actual, _render_wrong_row_major(specs, n_rows, n_cols)):
        return False
    if np.array_equal(actual, _render_wrong_position_only(specs, n_rows, n_cols)):
        return False
    if square_perimeter_branch and np.array_equal(actual, _render_wrong_keep_all_squares(specs)):
        return False

    output_sig = hashlib.md5(actual.tobytes() + str(actual.shape).encode()).hexdigest()
    if output_sig in _RECENT_OUTPUT_SIGNATURES:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_rows, n_cols, cell_h, cell_w = random.choice(LAYOUT_OPTIONS)
    specs, square_perimeter_branch = _build_specs(n_rows, n_cols)
    grid = _place_specs_on_grid(specs, n_rows, n_cols, cell_h, cell_w)
    if grid is None:
        return None
    if not _semantic_checks(grid, specs, n_rows, n_cols, square_perimeter_branch):
        return None

    out = main(grid)
    output_sig = hashlib.md5(out.tobytes() + str(out.shape).encode()).hexdigest()
    _RECENT_OUTPUT_SIGNATURES.add(output_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC opus/8/057/4")
