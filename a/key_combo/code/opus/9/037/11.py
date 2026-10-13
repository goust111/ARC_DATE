from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
TEAL = 8
CHAMBER_SIZE = 3
CELL_STRIDE = CHAMBER_SIZE + 1
PALETTE = (1, 3, 4, 5, 6, 7, 9)
MAX_GENERATION_ATTEMPTS = 400


def _find_chambers(grid: np.ndarray) -> list[tuple[int, int]]:
    """Return the top-left coordinate of every 3x3 chamber interior."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    chambers: list[tuple[int, int]] = []
    for r in range(1, h - CHAMBER_SIZE):
        for c in range(1, w - CHAMBER_SIZE):
            patch = g[r : r + CHAMBER_SIZE, c : c + CHAMBER_SIZE]
            if np.any(patch == TEAL):
                continue
            if not np.all(g[r - 1, c : c + CHAMBER_SIZE] == TEAL):
                continue
            if not np.all(g[r + CHAMBER_SIZE, c : c + CHAMBER_SIZE] == TEAL):
                continue
            if not np.all(g[r : r + CHAMBER_SIZE, c - 1] == TEAL):
                continue
            if not np.all(g[r : r + CHAMBER_SIZE, c + CHAMBER_SIZE] == TEAL):
                continue
            chambers.append((r, c))
    return chambers


def _marker_coord(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(np.asarray(grid, dtype=int) == RED)
    assert len(coords) == 1, "expected exactly one red marker pixel"
    return int(coords[0, 0]), int(coords[0, 1])


def _row_col_maps(chambers: list[tuple[int, int]]) -> tuple[dict[int, int], dict[int, int]]:
    rows = sorted({r for r, _ in chambers})
    cols = sorted({c for _, c in chambers})
    expected = {(r, c) for r in rows for c in cols}
    assert set(chambers) == expected, "chambers must form a regular grid"
    return {r: i for i, r in enumerate(rows)}, {c: i for i, c in enumerate(cols)}


def _transformed(tile: np.ndarray, dr: int, dc: int) -> np.ndarray:
    out = tile
    if dr < 0:
        out = np.flipud(out)
    if dc < 0:
        out = np.fliplr(out)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Use the marked chamber as a mirror-tiling template for every other chamber.

    Input structure:
    - A teal lattice partitions the grid into equal closed 3x3 chambers, with black corner gaps at lattice intersections.
    - Exactly one red pixel sits in one corner gap; the chamber immediately southeast of that marker is the template chamber.
    - The template chamber contains a non-symmetric multicolor pattern on black, while every other chamber interior is empty black.

    Rule:
    - Detect every 3x3 chamber enclosed by the teal lattice.
    - Read the red marker and extract the 3x3 template chamber directly southeast of it.
    - Erase the red marker to black, keep the template chamber unchanged, and fill every other empty chamber with the template.
    - If a target chamber is above the template, vertically flip the template; if it is left of the template, horizontally flip the template; combine both flips when both conditions hold.

    Key insight:
    - The red marker selects the source chamber, and each destination chamber receives a mirrored copy determined by its chamber-grid position relative to that source.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2-D grid"

    chambers = _find_chambers(g)
    assert chambers, "expected at least one lattice chamber"
    row_index, col_index = _row_col_maps(chambers)

    mr, mc = _marker_coord(g)
    template_top_left = (mr + 1, mc + 1)
    assert template_top_left in set(chambers), "marker must point to the chamber southeast of it"
    tr, tc = template_top_left
    template = g[tr : tr + CHAMBER_SIZE, tc : tc + CHAMBER_SIZE].copy()

    out = g.copy()
    out[mr, mc] = BLACK
    template_row = row_index[tr]
    template_col = col_index[tc]

    for r, c in chambers:
        if (r, c) == template_top_left:
            continue
        chamber = g[r : r + CHAMBER_SIZE, c : c + CHAMBER_SIZE]
        assert np.all(chamber == BLACK), "non-template chambers must start empty"
        dr = row_index[r] - template_row
        dc = col_index[c] - template_col
        out[r : r + CHAMBER_SIZE, c : c + CHAMBER_SIZE] = _transformed(template, dr, dc)

    return out


def _build_lattice(n_rows: int, n_cols: int) -> np.ndarray:
    h = n_rows * CELL_STRIDE + 1
    w = n_cols * CELL_STRIDE + 1
    grid = np.full((h, w), BLACK, dtype=int)
    for r in range(0, h, CELL_STRIDE):
        grid[r, :] = TEAL
    for c in range(0, w, CELL_STRIDE):
        grid[:, c] = TEAL
    for r in range(0, h, CELL_STRIDE):
        for c in range(0, w, CELL_STRIDE):
            grid[r, c] = BLACK
    return grid


def _template_ok(tile: np.ndarray) -> bool:
    colored = np.count_nonzero(tile != BLACK)
    colors = {int(v) for v in tile.flat if int(v) != BLACK}
    variants = [
        tile,
        np.fliplr(tile),
        np.flipud(tile),
        np.rot90(tile, 2),
    ]
    return (
        3 <= colored <= 5
        and 2 <= len(colors) <= 3
        and len({variant.tobytes() for variant in variants}) == 4
    )


def _make_template() -> np.ndarray:
    for _ in range(200):
        tile = np.zeros((CHAMBER_SIZE, CHAMBER_SIZE), dtype=int)
        palette = random.sample(PALETTE, k=random.randint(2, 3))
        coords = random.sample(
            [(r, c) for r in range(CHAMBER_SIZE) for c in range(CHAMBER_SIZE)],
            k=random.randint(3, 5),
        )
        forced = random.sample(coords, k=len(palette))
        for (r, c), color in zip(forced, palette):
            tile[r, c] = color
        for r, c in coords[len(palette) :]:
            tile[r, c] = random.choice(palette)
        if _template_ok(tile):
            return tile
    raise AssertionError("failed to build a non-degenerate template")


def _expected_output(n_rows: int, n_cols: int, template_rc: tuple[int, int], template: np.ndarray) -> np.ndarray:
    out = _build_lattice(n_rows, n_cols)
    tr, tc = template_rc
    for rr in range(n_rows):
        for cc in range(n_cols):
            r = rr * CELL_STRIDE + 1
            c = cc * CELL_STRIDE + 1
            out[r : r + CHAMBER_SIZE, c : c + CHAMBER_SIZE] = _transformed(
                template,
                rr - tr,
                cc - tc,
            )
    return out


def _validate_semantics(inp: np.ndarray, expected: np.ndarray, template_rc: tuple[int, int]) -> None:
    chambers = _find_chambers(inp)
    assert chambers, "generator must produce chambers"
    row_index, col_index = _row_col_maps(chambers)
    mr, mc = _marker_coord(inp)
    tr, tc = template_rc
    template_top_left = (tr * CELL_STRIDE + 1, tc * CELL_STRIDE + 1)
    assert template_top_left == (mr + 1, mc + 1), "marker must uniquely identify the template chamber"

    non_empty = []
    for r, c in chambers:
        patch = inp[r : r + CHAMBER_SIZE, c : c + CHAMBER_SIZE]
        if np.any(patch != BLACK):
            non_empty.append((r, c))
    assert non_empty == [template_top_left], "only the template chamber may be non-empty in the input"

    erows = {r for r, _ in chambers}
    ecols = {c for _, c in chambers}
    assert len(erows) in {3, 4} and len(ecols) in {3, 4}
    assert 0 < tr < len(erows) - 1 and 0 < tc < len(ecols) - 1, "template must be interior"

    template = inp[
        template_top_left[0] : template_top_left[0] + CHAMBER_SIZE,
        template_top_left[1] : template_top_left[1] + CHAMBER_SIZE,
    ]
    assert _template_ok(template), "template must visibly distinguish flips"

    expected_variants = set()
    for r, c in chambers:
        dr = row_index[r] - row_index[template_top_left[0]]
        dc = col_index[c] - col_index[template_top_left[1]]
        patch = expected[r : r + CHAMBER_SIZE, c : c + CHAMBER_SIZE]
        want = _transformed(template, dr, dc)
        assert np.array_equal(patch, want), "expected output must follow chamber-relative flips"
        expected_variants.add(patch.tobytes())
    assert len(expected_variants) == 4, "all four flip variants should be present"

    solved = main(inp)
    assert np.array_equal(solved, expected), "solver must match the generated ground truth"


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.randint(3, 4)
    n_cols = random.randint(3, 4)
    template_r = random.randint(1, n_rows - 2)
    template_c = random.randint(1, n_cols - 2)
    template = _make_template()

    inp = _build_lattice(n_rows, n_cols)
    r0 = template_r * CELL_STRIDE + 1
    c0 = template_c * CELL_STRIDE + 1
    inp[r0 : r0 + CHAMBER_SIZE, c0 : c0 + CHAMBER_SIZE] = template
    inp[template_r * CELL_STRIDE, template_c * CELL_STRIDE] = RED

    expected = _expected_output(n_rows, n_cols, (template_r, template_c), template)
    try:
        _validate_semantics(inp, expected, (template_r, template_c))
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
