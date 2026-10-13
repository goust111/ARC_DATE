from __future__ import annotations

import numpy as np


GRID_SIZE = 20
FRAME_COLOR = 1
TEMPLATE_COLOR = 2
MARKER_COLORS = (3, 4, 5, 6, 7)
BG = 0
MAX_GENERATION_ATTEMPTS = 500


def _mirror_row(row: int, height: int) -> int:
    return height - 1 - int(row)


def _mirror_col(col: int, width: int) -> int:
    return width - 1 - int(col)


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _random_connected_mask(height: int, width: int, rng: np.random.Generator) -> np.ndarray | None:
    min_area = max(height + width - 2, 6)
    max_area = max(min_area, min(height * width - 1, int(np.ceil(height * width * 0.68))))
    if min_area > max_area:
        return None

    for _ in range(200):
        target_area = int(rng.integers(min_area, max_area + 1))
        mask = np.zeros((height, width), dtype=bool)
        seed = (int(rng.integers(height)), int(rng.integers(width)))
        mask[seed] = True

        while int(mask.sum()) < target_area:
            frontier: set[tuple[int, int]] = set()
            for r, c in np.argwhere(mask):
                rr = int(r)
                cc = int(c)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = list(frontier)[int(rng.integers(len(frontier)))]
            mask[nr, nc] = True

        if int(mask.sum()) != target_area:
            continue
        if not np.all(mask.any(axis=1)) or not np.all(mask.any(axis=0)):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if not np.any(mask[:, 0]) and not np.any(mask[:, -1]):
            continue
        return mask

    return None


def _template_cells_from_grid(grid: np.ndarray) -> tuple[list[tuple[int, int]], int]:
    coords = np.argwhere(grid == TEMPLATE_COLOR)
    assert len(coords) > 0, "expected one red template glyph"
    top = int(coords[:, 0].min())
    cells = [(int(r) - top, int(c)) for r, c in coords]
    return cells, top


def _find_marker_pairs(grid: np.ndarray) -> list[tuple[int, int]]:
    height, width = grid.shape
    pairs: list[tuple[int, int]] = []
    for row in range(height):
        left = int(grid[row, 1])
        right = int(grid[row, width - 2])
        if left == right and left not in (BG, FRAME_COLOR, TEMPLATE_COLOR):
            pairs.append((row, left))
    assert pairs, "expected at least one endpoint-marker pair"
    return pairs


def _render_reconstruction(
    base: np.ndarray,
    template_cells: list[tuple[int, int]],
    marker_pairs: list[tuple[int, int]],
    should_mirror,
) -> np.ndarray:
    out = base.copy()
    height, width = out.shape
    rows_by_color: dict[int, set[int]] = {}
    any_rows = set()
    for row, color in marker_pairs:
        rows_by_color.setdefault(int(color), set()).add(int(row))
        any_rows.add(int(row))

    claims: dict[tuple[int, int], int] = {}
    for row, color in marker_pairs:
        mirrored = bool(should_mirror(int(row), int(color), rows_by_color, any_rows, height))
        for dr, abs_col in template_cells:
            rr = int(row) + int(dr)
            cc = _mirror_col(abs_col, width) if mirrored else int(abs_col)
            assert 0 <= rr < height
            assert 1 <= cc <= width - 2
            prev = claims.get((rr, cc))
            assert prev in (None, int(color)), "reconstructed glyphs overlap with different colors"
            claims[(rr, cc)] = int(color)

    for (r, c), color in claims.items():
        out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Erase the learned red glyph and rebuild same-color marker rows in normal or mirrored orientation.

    Input structure:
    - The grid is 20x20 with black interior and blue full-height frame bars in the first and last columns.
    - Inside the frame there is one complete red glyph template and several same-row endpoint-marker pairs at columns 1 and 18.
    - Marker colors come from the non-red non-blue palette, and a color may appear on multiple rows.

    Rule:
    - Read the red template glyph, storing each red pixel by its row offset from the template's top row and by its absolute interior column.
    - Erase the red template to black, then collect every marker pair row and determine the horizontal middle of the usable rows.
    - For each marker pair, rebuild the glyph in that pair's color: use the original columns if the same color also appears on the row mirrored across the horizontal middle, otherwise reflect the glyph horizontally across the interior.
    - Preserve the blue frame bars; marker pixels are overwritten only if the rebuilt glyph lands on them.

    Key insight:
    - Orientation is controlled by a global symmetry test on all rows carrying the same marker color, not by any single marker pair in isolation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape
    assert (height, width) == (GRID_SIZE, GRID_SIZE)
    assert np.all(g[:, 0] == FRAME_COLOR)
    assert np.all(g[:, -1] == FRAME_COLOR)

    template_cells, _template_top = _template_cells_from_grid(g)
    marker_pairs = _find_marker_pairs(g)

    out = g.copy()
    out[g == TEMPLATE_COLOR] = BG

    def actual_rule(row: int, color: int, rows_by_color, _any_rows, h: int) -> bool:
        return _mirror_row(row, h) not in rows_by_color[int(color)]

    out = _render_reconstruction(out, template_cells, marker_pairs, actual_rule)
    out[:, 0] = FRAME_COLOR
    out[:, -1] = FRAME_COLOR
    return out


def _rows_fit_without_overlap(rows: list[int], glyph_height: int) -> bool:
    intervals = sorted((r, r + glyph_height - 1) for r in rows)
    return all(intervals[i][1] < intervals[i + 1][0] for i in range(len(intervals) - 1))


def _pick_symmetric_pair_rows(
    glyph_height: int,
    used_rows: list[int],
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    valid = list(range(0, GRID_SIZE - glyph_height + 1))
    candidates = [
        row
        for row in valid
        if row < _mirror_row(row, GRID_SIZE)
        and _mirror_row(row, GRID_SIZE) in valid
        and _rows_fit_without_overlap(used_rows + [row, _mirror_row(row, GRID_SIZE)], glyph_height)
    ]
    if not candidates:
        return None
    row = int(candidates[int(rng.integers(len(candidates)))])
    return row, _mirror_row(row, GRID_SIZE)


def _pick_single_row(
    glyph_height: int,
    used_rows: list[int],
    rng: np.random.Generator,
    require_mirror_present: bool = False,
) -> tuple[int, int | None] | None:
    valid = list(range(0, GRID_SIZE - glyph_height + 1))
    candidates = []
    for row in valid:
        mirror = _mirror_row(row, GRID_SIZE)
        if require_mirror_present and mirror not in valid:
            continue
        if not _rows_fit_without_overlap(used_rows + [row], glyph_height):
            continue
        if require_mirror_present and not _rows_fit_without_overlap(used_rows + [row, mirror], glyph_height):
            continue
        candidates.append((row, mirror if require_mirror_present else None))
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def _build_marker_layout(glyph_height: int, rng: np.random.Generator) -> list[tuple[str, int]] | None:
    scenarios = ["combo", "occupied_mirror_diff", "repeated_unsym", "simple"]
    rng.shuffle(scenarios)

    for scenario in scenarios:
        if glyph_height == 5 and scenario in {"combo", "occupied_mirror_diff"}:
            continue
        used: list[int] = []
        layout: list[tuple[str, int]] = []

        sym_rows = _pick_symmetric_pair_rows(glyph_height, used, rng)
        if sym_rows is None:
            continue
        upper, lower = sym_rows
        used.extend([upper, lower])
        layout.extend([("sym", upper), ("sym", lower)])

        if scenario == "simple":
            single = _pick_single_row(glyph_height, used, rng, require_mirror_present=False)
            if single is None:
                continue
            layout.append(("single", int(single[0])))
            return layout

        if scenario == "occupied_mirror_diff":
            pair = _pick_single_row(glyph_height, used, rng, require_mirror_present=True)
            if pair is None:
                continue
            row, mirror = pair
            used.extend([int(row), int(mirror)])
            layout.extend([("diff_a", int(row)), ("diff_b", int(mirror))])
            return layout

        if scenario == "repeated_unsym":
            first = _pick_single_row(glyph_height, used, rng, require_mirror_present=False)
            if first is None:
                continue
            used.append(int(first[0]))
            second_choices = []
            for row in range(0, GRID_SIZE - glyph_height + 1):
                if not _rows_fit_without_overlap(used + [row], glyph_height):
                    continue
                if int(row) == _mirror_row(int(first[0]), GRID_SIZE):
                    continue
                second_choices.append(int(row))
            if not second_choices:
                continue
            second = int(second_choices[int(rng.integers(len(second_choices)))])
            layout.extend([("dup", int(first[0])), ("dup", second)])
            return layout

        if scenario == "combo":
            pair = _pick_single_row(glyph_height, used, rng, require_mirror_present=True)
            if pair is None:
                continue
            row, mirror = pair
            used.extend([int(row), int(mirror)])
            layout.extend([("diff_a", int(row)), ("diff_b", int(mirror))])

            first = _pick_single_row(glyph_height, used, rng, require_mirror_present=False)
            if first is None:
                continue
            used.append(int(first[0]))
            second_choices = []
            for candidate in range(0, GRID_SIZE - glyph_height + 1):
                if not _rows_fit_without_overlap(used + [candidate], glyph_height):
                    continue
                if int(candidate) == _mirror_row(int(first[0]), GRID_SIZE):
                    continue
                second_choices.append(int(candidate))
            if not second_choices:
                continue
            second = int(second_choices[int(rng.integers(len(second_choices)))])
            layout.extend([("dup", int(first[0])), ("dup", second)])
            return layout

    return None


def _assign_colors(layout: list[tuple[str, int]], rng: np.random.Generator) -> list[tuple[int, int]]:
    roles = {role for role, _row in layout}
    colors = list(MARKER_COLORS)
    rng.shuffle(colors)

    mapping: dict[str, int] = {}
    if "sym" in roles:
        mapping["sym"] = colors.pop()
    if "dup" in roles:
        mapping["dup"] = colors.pop()
    for role in ("single", "diff_a", "diff_b"):
        if role in roles:
            mapping[role] = colors.pop()

    return [(row, mapping[role]) for role, row in layout]


def _place_template(
    mask: np.ndarray,
    marker_pairs: list[tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], int] | None:
    glyph_height, glyph_width = mask.shape
    modes = ["left", "right", "interior"]
    rng.shuffle(modes)

    marker_points = {(row, 1) for row, _color in marker_pairs} | {
        (row, GRID_SIZE - 2) for row, _color in marker_pairs
    }

    local_coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    for mode in modes:
        if mode == "left":
            left = 1
        elif mode == "right":
            left = GRID_SIZE - 1 - glyph_width
        else:
            low = 2
            high = GRID_SIZE - 1 - glyph_width
            if low > high:
                continue
            left = int(rng.integers(low, high + 1))

        row_candidates = list(range(0, GRID_SIZE - glyph_height + 1))
        rng.shuffle(row_candidates)
        for top in row_candidates:
            cells = [(top + r, left + c) for r, c in local_coords]
            if set(cells).isdisjoint(marker_points):
                return cells, top
    return None


def _build_input_grid(
    template_cells: list[tuple[int, int]],
    marker_pairs: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    grid[:, 0] = FRAME_COLOR
    grid[:, -1] = FRAME_COLOR
    for r, c in template_cells:
        grid[r, c] = TEMPLATE_COLOR
    for row, color in marker_pairs:
        grid[row, 1] = int(color)
        grid[row, GRID_SIZE - 2] = int(color)
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    if grid.shape != (GRID_SIZE, GRID_SIZE):
        return False
    if not np.all(grid[:, 0] == FRAME_COLOR) or not np.all(grid[:, -1] == FRAME_COLOR):
        return False

    template_coords = [(int(r), int(c)) for r, c in np.argwhere(grid == TEMPLATE_COLOR)]
    if not template_coords:
        return False
    r0, c0, r1, c1 = _bbox(template_coords)
    glyph_height = r1 - r0 + 1
    glyph_width = c1 - c0 + 1
    if not (3 <= glyph_height <= 5 and 4 <= glyph_width <= 8):
        return False

    local_mask = np.zeros((glyph_height, glyph_width), dtype=bool)
    for r, c in template_coords:
        local_mask[r - r0, c - c0] = True
    local_mask = _normalize_mask(local_mask)
    if np.array_equal(local_mask, np.fliplr(local_mask)):
        return False

    marker_pairs = _find_marker_pairs(grid)
    rows_by_color: dict[int, set[int]] = {}
    for row, color in marker_pairs:
        rows_by_color.setdefault(color, set()).add(row)

    has_symmetric = False
    has_asymmetric = False
    for row, color in marker_pairs:
        if _mirror_row(row, GRID_SIZE) in rows_by_color[color]:
            has_symmetric = True
        else:
            has_asymmetric = True
    if not (has_symmetric and has_asymmetric):
        return False

    top_cells, _template_top = _template_cells_from_grid(grid)
    erased = grid.copy()
    erased[erased == TEMPLATE_COLOR] = BG

    def actual_rule(row: int, color: int, rows_by_color, _any_rows, h: int) -> bool:
        return _mirror_row(row, h) not in rows_by_color[color]

    def all_normal(_row: int, _color: int, _rows_by_color, _any_rows, _h: int) -> bool:
        return False

    def all_mirror(_row: int, _color: int, _rows_by_color, _any_rows, _h: int) -> bool:
        return True

    def any_color_partner(row: int, _color: int, _rows_by_color, any_rows, h: int) -> bool:
        return _mirror_row(row, h) not in any_rows

    def duplicate_color_shortcut(row: int, color: int, rows_by_color, _any_rows, _h: int) -> bool:
        return len(rows_by_color[color]) == 1

    actual = _render_reconstruction(erased, top_cells, marker_pairs, actual_rule)
    normal_only = _render_reconstruction(erased, top_cells, marker_pairs, all_normal)
    mirror_only = _render_reconstruction(erased, top_cells, marker_pairs, all_mirror)
    any_partner = _render_reconstruction(erased, top_cells, marker_pairs, any_color_partner)
    dup_shortcut = _render_reconstruction(erased, top_cells, marker_pairs, duplicate_color_shortcut)

    if np.array_equal(actual, normal_only):
        return False
    if np.array_equal(actual, mirror_only):
        return False
    if (
        glyph_height != 5
        and np.array_equal(actual, any_partner)
        and np.array_equal(actual, dup_shortcut)
    ):
        return False

    for index, (row, color) in enumerate(marker_pairs):
        target_mirrored = actual_rule(row, color, rows_by_color, set(), GRID_SIZE)
        def toggled_rule(r: int, col: int, rows_by_color2, _any_rows, h: int, idx=index, desired=target_mirrored):
            if marker_pairs[idx] == (r, col):
                return not desired
            return actual_rule(r, col, rows_by_color2, set(), h)
        toggled = _render_reconstruction(erased, top_cells, marker_pairs, toggled_rule)
        if np.array_equal(actual, toggled):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = np.random.default_rng()
        glyph_height = int(rng.choice([3, 4, 5], p=[0.45, 0.4, 0.15]))
        glyph_width = int(rng.integers(4, 9))
        mask = _random_connected_mask(glyph_height, glyph_width, rng)
        if mask is None:
            continue

        layout = _build_marker_layout(glyph_height, rng)
        if layout is None:
            continue
        marker_pairs = _assign_colors(layout, rng)

        placement = _place_template(mask, marker_pairs, rng)
        if placement is None:
            continue
        template_cells, _template_top = placement

        grid = _build_input_grid(template_cells, marker_pairs)
        if not _semantic_checks(grid):
            continue
        return grid

    raise RuntimeError("failed to generate a valid mirror-conditional glyph puzzle")
