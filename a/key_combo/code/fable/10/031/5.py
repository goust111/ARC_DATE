from __future__ import annotations

import random

import numpy as np


MASK_POOL = (
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
)
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)
SPRITE_SIZE = 3
ORDER_OPTIONS = (3, 5)
GAP_OPTIONS = (1, 2)
MARGIN_OPTIONS = (1, 2)
GEN_RETRIES = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(x) for x in row) for row in np.asarray(mask, dtype=bool))


def _groups_from_projection(flags: np.ndarray) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    i = 0
    while i < len(flags):
        if not bool(flags[i]):
            i += 1
            continue
        start = i
        while i + 1 < len(flags) and bool(flags[i + 1]):
            i += 1
        groups.append((start, i))
        i += 1
    return groups


def _parse_slots(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]], list[list[dict | None]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    row_groups = _groups_from_projection(np.any(g != bg, axis=1))
    col_groups = _groups_from_projection(np.any(g != bg, axis=0))
    assert row_groups and col_groups, "expected foreground sprites"
    assert len(row_groups) == len(col_groups), "sprite lattice must be square"
    assert all(r1 - r0 + 1 == SPRITE_SIZE for r0, r1 in row_groups), "unexpected row band height"
    assert all(c1 - c0 + 1 == SPRITE_SIZE for c0, c1 in col_groups), "unexpected col band width"

    slots: list[list[dict | None]] = []
    for r0, r1 in row_groups:
        row: list[dict | None] = []
        for c0, c1 in col_groups:
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            mask = patch != bg
            if not mask.any():
                row.append(None)
                continue
            colors = sorted(set(map(int, patch[mask])))
            assert len(colors) == 1, "each sprite must be monochrome"
            assert mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any(), (
                "each sprite mask must touch every row and every column of its slot"
            )
            row.append({"mask": mask.copy(), "color": colors[0]})
        slots.append(row)
    return bg, len(row_groups), row_groups, col_groups, slots


def _render_mask(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = int(color)
    return patch


def _fill_slot(grid: np.ndarray, row_box: tuple[int, int], col_box: tuple[int, int], mask: np.ndarray, color: int, bg: int) -> None:
    r0, r1 = row_box
    c0, c1 = col_box
    grid[r0 : r1 + 1, c0 : c1 + 1] = _render_mask(mask, color, bg)


def _complete_symbols(
    entries: list[list[dict | None]],
    all_symbols: set,
    field: str,
) -> dict[tuple[int, int], object]:
    n = len(entries)
    blank_positions = [(r, c) for r in range(n) for c in range(n) if entries[r][c] is None]
    blanks_by_row = {}
    for r, c in blank_positions:
        blanks_by_row.setdefault(r, []).append(c)
    assert sorted(len(cols) for cols in blanks_by_row.values()) == [2], "expected exactly two blanks in one row"
    blank_row, blank_cols = next(iter(blanks_by_row.items()))
    assert len(blank_cols) == 2

    row_present = {entries[blank_row][c][field] for c in range(n) if entries[blank_row][c] is not None}
    row_missing = all_symbols - row_present
    assert len(row_missing) == 2, "blank row must miss two symbols"

    result = {}
    for c in blank_cols:
        col_present = {entries[r][c][field] for r in range(n) if entries[r][c] is not None}
        candidates = row_missing - col_present
        assert len(candidates) == 1, "column elimination must leave one candidate"
        result[(blank_row, c)] = next(iter(candidates))
    assert len(set(result.values())) == 2
    return result


def _latin_checks(entries: list[list[dict]], field: str) -> None:
    n = len(entries)
    symbols = {entries[r][c][field] for r in range(n) for c in range(n)}
    assert len(symbols) == n
    for r in range(n):
        assert {entries[r][c][field] for c in range(n)} == symbols
    for c in range(n):
        assert {entries[r][c][field] for r in range(n)} == symbols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the two blanks in a sprite array by solving independent Latin-square constraints for shape and color.

    Input structure:
    - The grid contains an n-by-n array of equally spaced sprite slots on a dominant background.
    - Every non-empty slot holds one monochrome 3x3 sprite mask; the mask encodes the shape and the fill color encodes the color.
    - Exactly two slots are blank, and those two blanks lie in the same row but in different columns.

    Rule:
    - Across the full array, shapes satisfy a Latin-square constraint: each row and each column contains every shape exactly once.
    - Colors independently satisfy the same Latin-square constraint, but shape-color pairings are not fixed.
    - In the blank row, find the two missing shapes and the two missing colors; for each blank column, eliminate whichever missing shape/color already appears in that column, then draw the uniquely remaining shape in the uniquely remaining color.

    Key insight:
    - The missing sprite is recovered by solving two separate Latin squares, one over shape masks and one over colors, and then recombining them; the needed shape-color pair need not already exist anywhere in the input.
    """
    g = np.asarray(grid, dtype=int)
    bg, n, row_groups, col_groups, slots = _parse_slots(g)

    shape_keys = {
        _mask_key(cell["mask"])
        for row in slots
        for cell in row
        if cell is not None
    }
    colors = {
        int(cell["color"])
        for row in slots
        for cell in row
        if cell is not None
    }
    assert len(shape_keys) == n, "all shape types must remain visible"
    assert len(colors) == n, "all color types must remain visible"

    shape_entries: list[list[dict | None]] = []
    for row in slots:
        shape_row: list[dict | None] = []
        for cell in row:
            if cell is None:
                shape_row.append(None)
            else:
                shape_row.append({"shape": _mask_key(cell["mask"]), "color": int(cell["color"])})
        shape_entries.append(shape_row)

    shape_solution = _complete_symbols(shape_entries, shape_keys, "shape")
    color_solution = _complete_symbols(shape_entries, colors, "color")

    out = g.copy()
    for (r, c), shape_key in shape_solution.items():
        mask = np.array(shape_key, dtype=bool)
        color = int(color_solution[(r, c)])
        _fill_slot(out, row_groups[r], col_groups[c], mask, color, bg)

    parsed = _parse_slots(out)[4]
    completed = [[cell for cell in row if cell is not None] for row in parsed]
    assert all(len(row) == n for row in completed), "output must have no blank slots"
    _latin_checks(
        [[{"shape": _mask_key(cell["mask"]), "color": int(cell["color"])} for cell in row] for row in parsed],
        "shape",
    )
    _latin_checks(
        [[{"shape": _mask_key(cell["mask"]), "color": int(cell["color"])} for cell in row] for row in parsed],
        "color",
    )
    return out


def _layout_size(n: int, gap: int, margin: int) -> tuple[int, int]:
    size = 2 * margin + n * SPRITE_SIZE + (n - 1) * gap
    return size, size


def _slot_origin(index: int, gap: int, margin: int) -> int:
    return margin + index * (SPRITE_SIZE + gap)


def _build_output(
    n: int,
    gap: int,
    margin: int,
    masks: list[np.ndarray],
    colors: list[int],
    row_perm: list[int],
    col_perm: list[int],
    shape_perm: list[int],
    color_perm: list[int],
    alpha: int,
    beta: int,
) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = _layout_size(n, gap, margin)
    out = np.zeros((h, w), dtype=int)
    pair_grid: list[list[tuple[int, int]]] = []
    for r in range(n):
        pair_row: list[tuple[int, int]] = []
        rr = row_perm[r]
        top = _slot_origin(r, gap, margin)
        for c in range(n):
            cc = col_perm[c]
            left = _slot_origin(c, gap, margin)
            shape_idx = shape_perm[(rr + cc) % n]
            color_idx = color_perm[(rr + alpha * cc + beta) % n]
            out[top : top + SPRITE_SIZE, left : left + SPRITE_SIZE] = _render_mask(
                masks[shape_idx], colors[color_idx], 0
            )
            pair_row.append((shape_idx, color_idx))
        pair_grid.append(pair_row)
    return out, pair_grid


def _blank_slot(grid: np.ndarray, r: int, c: int, gap: int, margin: int) -> None:
    top = _slot_origin(r, gap, margin)
    left = _slot_origin(c, gap, margin)
    grid[top : top + SPRITE_SIZE, left : left + SPRITE_SIZE] = 0


def _verify_generator_semantics(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    _, n, _row_groups, _col_groups, slots_in = _parse_slots(inp)
    _, _n2, _row_groups2, _col_groups2, slots_out = _parse_slots(expected)

    blank_positions = [(r, c) for r in range(n) for c in range(n) if slots_in[r][c] is None]
    if len(blank_positions) != 2:
        return False
    if len({r for r, _c in blank_positions}) != 1 or len({c for _r, c in blank_positions}) != 2:
        return False

    # The missing shape-color pairs must not already occur elsewhere in the input.
    present_pairs = {
        (_mask_key(cell["mask"]), int(cell["color"]))
        for row in slots_in
        for cell in row
        if cell is not None
    }
    token_grid = [
        [(_mask_key(cell["mask"]), int(cell["color"])) for cell in row]
        for row in slots_out
    ]
    missing_pairs = {token_grid[r][c] for r, c in blank_positions}
    if present_pairs & missing_pairs:
        return False

    # Shapes must appear with multiple colors, and colors with multiple shapes.
    shape_to_colors: dict[tuple[tuple[int, ...], ...], set[int]] = {}
    color_to_shapes: dict[int, set[tuple[tuple[int, ...], ...]]] = {}
    for row in slots_in:
        for cell in row:
            if cell is None:
                continue
            key = _mask_key(cell["mask"])
            color = int(cell["color"])
            shape_to_colors.setdefault(key, set()).add(color)
            color_to_shapes.setdefault(color, set()).add(key)
    if any(len(v) < 2 for v in shape_to_colors.values()):
        return False
    if any(len(v) < 2 for v in color_to_shapes.values()):
        return False

    # Wrongly swapping the two blank shapes or colors must break Latin constraints.
    br = blank_positions[0][0]
    c1, c2 = sorted(c for _r, c in blank_positions)
    s1, k1 = token_grid[br][c1]
    s2, k2 = token_grid[br][c2]
    if s1 == s2 or k1 == k2:
        return False

    def check_assignment(assign_shapes: tuple[int, int], assign_colors: tuple[int, int]) -> bool:
        entries: list[list[dict]] = []
        for r in range(n):
            entry_row: list[dict] = []
            for c in range(n):
                cell = slots_out[r][c]
                if r == br and c == c1:
                    entry_row.append({"shape": assign_shapes[0], "color": assign_colors[0]})
                elif r == br and c == c2:
                    entry_row.append({"shape": assign_shapes[1], "color": assign_colors[1]})
                else:
                    entry_row.append(
                        {
                            "shape": token_grid[r][c][0],
                            "color": token_grid[r][c][1],
                        }
                    )
            entries.append(entry_row)
        try:
            _latin_checks(entries, "shape")
            _latin_checks(entries, "color")
        except AssertionError:
            return False
        return True

    correct = check_assignment((s1, s2), (k1, k2))
    wrong_shape = check_assignment((s2, s1), (k1, k2))
    wrong_color = check_assignment((s1, s2), (k2, k1))
    wrong_both = check_assignment((s2, s1), (k2, k1))
    return correct and not wrong_shape and not wrong_color and not wrong_both


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(ORDER_OPTIONS)
    gap = random.choice(GAP_OPTIONS)
    margin = random.choice(MARGIN_OPTIONS)
    if max(_layout_size(n, gap, margin)) > 30:
        return None

    alpha_choices = [a for a in range(2, n) if np.gcd(a - 1, n) == 1 and np.gcd(a, n) == 1]
    if not alpha_choices:
        return None
    alpha = random.choice(alpha_choices)
    beta = random.randrange(n)

    masks = [mask.copy() for mask in random.sample(MASK_POOL, n)]
    colors = random.sample(PALETTE, n)
    row_perm = random.sample(range(n), n)
    col_perm = random.sample(range(n), n)
    shape_perm = random.sample(range(n), n)
    color_perm = random.sample(range(n), n)

    expected, pair_grid = _build_output(
        n,
        gap,
        margin,
        masks,
        colors,
        row_perm,
        col_perm,
        shape_perm,
        color_perm,
        alpha,
        beta,
    )

    blank_row = random.randrange(n)
    blank_cols = sorted(random.sample(range(n), 2))
    inp = expected.copy()
    for c in blank_cols:
        _blank_slot(inp, blank_row, c, gap, margin)

    return inp if _verify_generator_semantics(inp, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
