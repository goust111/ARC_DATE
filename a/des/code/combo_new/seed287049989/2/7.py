from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
COLOR_POOL = (1, 2, 3, 6, 7, 8, 9)

TEMPLATE_SIZE = 3
TEMPLATE_STEP = 4
CELL_SIZE = 3
CELL_STEP = 4

GEN_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _shape(*rows: str) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(_components(mask)) == 1


def _build_template_library() -> tuple[np.ndarray, ...]:
    bases = (
        _shape("100", "100", "111"),
        _shape("111", "010", "010"),
        _shape("101", "101", "111"),
        _shape("111", "100", "111"),
        _shape("110", "010", "011"),
        _shape("100", "111", "001"),
    )

    seen: set[bytes] = set()
    library: list[np.ndarray] = []
    for base in bases:
        for variant in (base, np.fliplr(base)):
            for k in range(4):
                mask = np.rot90(variant, k)
                key = mask.tobytes()
                if key in seen:
                    continue
                assert mask.shape == (TEMPLATE_SIZE, TEMPLATE_SIZE)
                assert _connected(mask)
                assert mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()
                seen.add(key)
                library.append(mask.copy())
    return tuple(library)


TEMPLATE_LIBRARY = _build_template_library()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color each activated row/column lookup slot by stamping the matching grey template in the matching header color.

    Input structure:
    - The background is black.
    - Row 0 contains one contiguous run of distinct non-black header pixels; their left-to-right order defines the color columns.
    - Along the left side, several disconnected grey 3x3 template motifs are stacked top-to-bottom with one blank row between them; their order defines the template rows.
    - To the right of the templates, yellow single-pixel markers sit at the centers of some 3x3 matrix cells, with one matrix row per template and one matrix column per header color.

    Rule:
    - Pair the i-th grey template with the i-th matrix row and the j-th header color with the j-th matrix column.
    - For every yellow marker, clear its 3x3 matrix cell and repaint that cell with a copy of the corresponding template, using the corresponding header color.
    - Leave unmarked matrix cells black, and preserve the top color strip and left grey templates unchanged.

    Key insight:
    - The puzzle is a two-axis lookup table: template order selects the shape, header order selects the color, and each yellow center decides whether that row/column combination is rendered.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _most_common_color(g)

    header_cols = np.flatnonzero(g[0] != bg)
    assert header_cols.size >= 2, "expected a top-row color sequence"
    assert np.array_equal(header_cols, np.arange(header_cols[0], header_cols[-1] + 1)), "header colors must be contiguous"
    header_colors = [int(g[0, c]) for c in header_cols]
    assert len(set(header_colors)) == len(header_colors), "header colors must be distinct"
    assert all(color not in (bg, GREY, YELLOW) for color in header_colors), "header colors must exclude bg/grey/yellow"

    template_comps = _components(g == GREY)
    assert template_comps, "expected grey template motifs"
    templates: list[np.ndarray] = []
    template_lefts: list[int] = []
    prev_r1: int | None = None
    for comp in sorted(template_comps, key=lambda cells: _bbox(cells)[:2]):
        r0, c0, r1, c1 = _bbox(comp)
        assert (r1 - r0 + 1, c1 - c0 + 1) == (TEMPLATE_SIZE, TEMPLATE_SIZE), "templates must occupy 3x3 boxes"
        if prev_r1 is not None:
            assert r0 - prev_r1 >= 2, "templates must be vertically separated"
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        templates.append(sub == GREY)
        template_lefts.append(int(c0))
        prev_r1 = int(r1)
    assert len(set(template_lefts)) == 1, "templates must share one left column"

    markers = np.argwhere(g == YELLOW)
    assert markers.size > 0, "expected at least one activation marker"
    marker_rows = sorted({int(r) for r, _ in markers})
    marker_cols = sorted({int(c) for _, c in markers})
    assert len(marker_rows) == len(templates), "each template row must contain an activation marker"
    assert len(marker_cols) == len(header_colors), "each color column must contain an activation marker"
    if len(marker_rows) > 1:
        assert min(np.diff(marker_rows)) >= CELL_SIZE, "matrix rows must not overlap"
    if len(marker_cols) > 1:
        assert min(np.diff(marker_cols)) >= CELL_SIZE, "matrix columns must not overlap"

    row_to_template = {row: templates[i] for i, row in enumerate(marker_rows)}
    col_to_color = {col: header_colors[j] for j, col in enumerate(marker_cols)}

    out = g.copy()
    h, w = g.shape
    for r, c in markers:
        rr = int(r)
        cc = int(c)
        shape = row_to_template[rr]
        color = col_to_color[cc]
        pr0 = rr - 1
        pc0 = cc - 1
        pr1 = pr0 + CELL_SIZE
        pc1 = pc0 + CELL_SIZE
        assert 0 <= pr0 and pr1 <= h and 0 <= pc0 and pc1 <= w, "activation cell must stay in bounds"
        src_patch = g[pr0:pr1, pc0:pc1]
        assert np.all((src_patch == bg) | (src_patch == YELLOW)), "activation cell must be otherwise empty"
        patch = out[pr0:pr1, pc0:pc1]
        patch[:] = bg
        patch[shape] = color

    return out


def _random_activation(n_rows: int, n_cols: int, rng: random.Random) -> np.ndarray:
    density = rng.uniform(0.35, 0.7)
    active = np.array(
        [[rng.random() < density for _ in range(n_cols)] for _ in range(n_rows)],
        dtype=bool,
    )

    for r in range(n_rows):
        if not active[r].any():
            active[r, rng.randrange(n_cols)] = True
    for c in range(n_cols):
        if not active[:, c].any():
            active[rng.randrange(n_rows), c] = True

    if n_rows * n_cols > 1 and active.all():
        removable = [
            (r, c)
            for r in range(n_rows)
            for c in range(n_cols)
            if active[r].sum() > 1 and active[:, c].sum() > 1
        ]
        if removable:
            r, c = rng.choice(removable)
            active[r, c] = False

    return active


def _paint_cell(grid: np.ndarray, center_r: int, center_c: int, shape: np.ndarray, color: int) -> None:
    patch = grid[center_r - 1 : center_r + 2, center_c - 1 : center_c + 2]
    patch[:] = BG
    patch[shape] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_templates = rng.randint(2, 4)
    n_colors = rng.randint(2, 5)

    template_c0 = rng.randint(0, 2)
    template_r0 = rng.randint(2, 5)
    matrix_c0 = template_c0 + TEMPLATE_SIZE + rng.randint(2, 4)

    matrix_w = CELL_STEP * (n_colors - 1) + CELL_SIZE
    seq_c0 = matrix_c0 + rng.randint(0, matrix_w - n_colors)

    height = template_r0 + TEMPLATE_STEP * (n_templates - 1) + TEMPLATE_SIZE + rng.randint(1, 2)
    width = max(seq_c0 + n_colors, matrix_c0 + matrix_w) + rng.randint(1, 2)
    if height > 30 or width > 30:
        return None

    header_colors = rng.sample(COLOR_POOL, n_colors)
    templates = rng.sample(TEMPLATE_LIBRARY, n_templates)
    active = _random_activation(n_templates, n_colors, rng)

    grid = np.full((height, width), BG, dtype=int)
    grid[0, seq_c0 : seq_c0 + n_colors] = header_colors

    for i, shape in enumerate(templates):
        tr0 = template_r0 + TEMPLATE_STEP * i
        patch = grid[tr0 : tr0 + TEMPLATE_SIZE, template_c0 : template_c0 + TEMPLATE_SIZE]
        patch[shape] = GREY

    row_centers = [template_r0 + TEMPLATE_STEP * i + 1 for i in range(n_templates)]
    col_centers = [matrix_c0 + CELL_STEP * j + 1 for j in range(n_colors)]
    for i, rr in enumerate(row_centers):
        for j, cc in enumerate(col_centers):
            if active[i, j]:
                grid[rr, cc] = YELLOW

    expected = grid.copy()
    for i, rr in enumerate(row_centers):
        for j, cc in enumerate(col_centers):
            if active[i, j]:
                _paint_cell(expected, rr, cc, templates[i], header_colors[j])

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
