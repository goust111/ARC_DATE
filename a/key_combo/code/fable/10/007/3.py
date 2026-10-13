from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


SLOT = 3
BACKGROUND = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]

ROW_COUNT_RANGE = (3, 4)
COL_COUNT_RANGE = (3, 4)
GAP_RANGE = (4, 7)
MARGIN_RANGE = (1, 3)
MAX_DIM = 30
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _find_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors8(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _mask_id(mask: np.ndarray) -> int:
    acc = 0
    bit = 1
    for value in mask.reshape(-1):
        if bool(value):
            acc |= bit
        bit <<= 1
    return acc


def _render_mask(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros((SLOT, SLOT), dtype=int)
    patch[mask] = color
    return patch


def _sample_axis(count: int) -> tuple[int, list[int]] | None:
    for _ in range(100):
        gaps = [random.randint(*GAP_RANGE) for _ in range(count - 1)]
        if count >= 4 and len(set(gaps)) == 1:
            continue
        margin0 = random.randint(*MARGIN_RANGE)
        margin1 = random.randint(*MARGIN_RANGE)
        positions = [margin0]
        for gap in gaps:
            positions.append(positions[-1] + gap)
        size = positions[-1] + SLOT + margin1
        if size <= MAX_DIM:
            return size, positions
    return None


def _infer_instances(grid: np.ndarray) -> tuple[int, list[int], list[int], dict[tuple[int, int], np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    colors = sorted(set(map(int, g.reshape(-1))) - {bg})
    assert len(colors) == 1, "expected exactly one foreground color"
    color = colors[0]

    components = _find_components(g, color)
    assert components, "expected visible sprites"

    occupied: dict[tuple[int, int], np.ndarray] = {}
    for comp in components:
        r0 = int(comp[:, 0].min())
        c0 = int(comp[:, 1].min())
        assert r0 + SLOT <= g.shape[0] and c0 + SLOT <= g.shape[1], "sprite slot must fit inside grid"
        patch = g[r0 : r0 + SLOT, c0 : c0 + SLOT]
        mask = patch == color
        assert mask[0, 0], "every sprite must expose the slot anchor"
        assert _mask_connected(mask), "every visible sprite must be 8-connected"
        key = (r0, c0)
        assert key not in occupied, "slot anchors must be unique"
        visible_coords = {(int(r), int(c)) for r, c in comp}
        patch_coords = {
            (r0 + rr, c0 + cc)
            for rr in range(SLOT)
            for cc in range(SLOT)
            if mask[rr, cc]
        }
        assert visible_coords == patch_coords, "component must match its anchored 3x3 slot patch"
        occupied[key] = mask

    rows = sorted({r for r, _c in occupied})
    cols = sorted({c for _r, c in occupied})
    assert len(rows) >= 2 and len(cols) >= 2, "expected a row-column lattice"
    assert all(any((r, c) in occupied for c in cols) for r in rows), "every lattice row needs evidence"
    assert all(any((r, c) in occupied for r in rows) for c in cols), "every lattice column needs evidence"
    return color, rows, cols, occupied


def _recover_templates(
    rows: list[int],
    cols: list[int],
    occupied: dict[tuple[int, int], np.ndarray],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    row_templates: dict[int, np.ndarray] = {}
    col_templates: dict[int, np.ndarray] = {}

    for r in rows:
        row_masks = [occupied[(r, c)] for c in cols if (r, c) in occupied]
        assert len(row_masks) >= 2, "each row must show at least two different sprites"
        assert len({_mask_id(mask) for mask in row_masks}) == len(row_masks), "row sprites must differ"
        template = np.zeros((SLOT, SLOT), dtype=bool)
        for mask in row_masks:
            template |= mask
        row_templates[r] = template

    for c in cols:
        col_masks = [occupied[(r, c)] for r in rows if (r, c) in occupied]
        assert len(col_masks) >= 2, "each column must show at least two different sprites"
        assert len({_mask_id(mask) for mask in col_masks}) == len(col_masks), "column sprites must differ"
        template = np.zeros((SLOT, SLOT), dtype=bool)
        for mask in col_masks:
            template |= mask
        col_templates[c] = template

    for r in rows:
        for c in cols:
            if (r, c) in occupied:
                expected = row_templates[r] & col_templates[c]
                assert np.array_equal(occupied[(r, c)], expected), "visible sprite must be row/column template intersection"
    return row_templates, col_templates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover hidden row and column templates from the visible sprites, then fill every empty lattice slot with their intersection.

    Input structure
    - A black background contains several same-colored 3x3 sprites placed on some cells of a rectangular row/column lattice.
    - Each visible sprite is an anchored 8-connected mask inside its own 3x3 slot, and all empty slots are fully black.
    - The full row template and full column template are never drawn directly; only their pairwise intersections appear as the existing sprites.

    Rule
    - Read the visible sprite anchors to recover the lattice row starts and column starts.
    - For each lattice row, take the union of all visible sprite masks in that row to obtain the hidden row template; do the analogous union down each column to obtain the hidden column template.
    - Keep existing sprites unchanged, and for every empty slot paste the intersection of that slot's row template and column template in the common foreground color.

    Key insight
    The observed sprites are factorized pieces: each one is not the true template itself but the overlap of one hidden row mask and one hidden column mask, so the missing cells can only be solved by first restoring both template families via unions.
    """
    g = np.asarray(grid, dtype=int)
    color, rows, cols, occupied = _infer_instances(g)
    row_templates, col_templates = _recover_templates(rows, cols, occupied)

    out = g.copy()
    for r in rows:
        for c in cols:
            if (r, c) in occupied:
                continue
            mask = row_templates[r] & col_templates[c]
            assert mask.any(), "every synthesized sprite must be visible"
            assert _mask_connected(mask), "synthesized sprites must stay 8-connected"
            slot = out[r : r + SLOT, c : c + SLOT]
            assert slot.shape == (SLOT, SLOT) and np.all(slot == BACKGROUND), "empty slot must be blank"
            slot[mask] = color
    return out


def _build_candidates() -> list[np.ndarray]:
    out = []
    for bits in range(1 << (SLOT * SLOT)):
        if (bits & 1) == 0:
            continue
        mask = np.zeros((SLOT, SLOT), dtype=bool)
        for idx in range(SLOT * SLOT):
            if bits & (1 << idx):
                mask[idx // SLOT, idx % SLOT] = True
        area = int(mask.sum())
        if not (4 <= area <= 7):
            continue
        if not _mask_connected(mask):
            continue
        out.append(mask)
    return out


CANDIDATE_MASKS = _build_candidates()


def _pick_templates(
    n_rows: int,
    n_cols: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[np.ndarray]]] | None:
    for _ in range(300):
        row_masks = random.sample(CANDIDATE_MASKS, n_rows)
        col_masks = random.sample(CANDIDATE_MASKS, n_cols)
        if len({_mask_id(mask) for mask in row_masks}) < 2 or len({_mask_id(mask) for mask in col_masks}) < 2:
            continue

        full = [[row_masks[i] & col_masks[j] for j in range(n_cols)] for i in range(n_rows)]
        if any((not cell.any()) or (not _mask_connected(cell)) or int(cell.sum()) < 2 for row in full for cell in row):
            continue
        if any(len({_mask_id(cell) for cell in row}) != n_cols for row in full):
            continue
        if any(len({_mask_id(full[i][j]) for i in range(n_rows)}) != n_rows for j in range(n_cols)):
            continue
        if any(not np.array_equal(np.logical_or.reduce(row), row_masks[i]) for i, row in enumerate(full)):
            continue
        if any(
            not np.array_equal(
                np.logical_or.reduce([full[i][j] for i in range(n_rows)]),
                col_masks[j],
            )
            for j in range(n_cols)
        ):
            continue
        return row_masks, col_masks, full
    return None


def _sample_occupancy(full: list[list[np.ndarray]]) -> np.ndarray | None:
    n_rows = len(full)
    n_cols = len(full[0])
    total = n_rows * n_cols
    cells = [(i, j) for i in range(n_rows) for j in range(n_cols)]
    max_holes = min(total - 1, 5)
    for _ in range(300):
        holes = random.randint(2, max_holes)
        removed = set(random.sample(cells, holes))
        occ = np.ones((n_rows, n_cols), dtype=bool)
        for i, j in removed:
            occ[i, j] = False

        row_counts = occ.sum(axis=1)
        col_counts = occ.sum(axis=0)
        if np.any(row_counts < 2) or np.any(col_counts < 2):
            continue
        if len({i for i, _j in removed}) < 2 or len({j for _i, j in removed}) < 2:
            continue
        if any(len({_mask_id(full[i][j]) for j in range(n_cols) if occ[i, j]}) != int(row_counts[i]) for i in range(n_rows)):
            continue
        if any(len({_mask_id(full[i][j]) for i in range(n_rows) if occ[i, j]}) != int(col_counts[j]) for j in range(n_cols)):
            continue
        if any(
            not np.array_equal(
                np.logical_or.reduce([full[i][j] for j in range(n_cols) if occ[i, j]]),
                np.logical_or.reduce(full[i]),
            )
            for i in range(n_rows)
        ):
            continue
        if any(
            not np.array_equal(
                np.logical_or.reduce([full[i][j] for i in range(n_rows) if occ[i, j]]),
                np.logical_or.reduce([full[i][j] for i in range(n_rows)]),
            )
            for j in range(n_cols)
        ):
            continue

        missing_shapes = [_mask_id(full[i][j]) for i in range(n_rows) for j in range(n_cols) if not occ[i, j]]
        if len(set(missing_shapes)) < 2:
            continue
        row_missing_var = any(
            len({_mask_id(full[i][j]) for j in range(n_cols) if not occ[i, j]}) >= 2
            for i in range(n_rows)
        )
        col_missing_var = any(
            len({_mask_id(full[i][j]) for i in range(n_rows) if not occ[i, j]}) >= 2
            for j in range(n_cols)
        )
        if not (row_missing_var and col_missing_var):
            continue
        if not any(
            not np.array_equal(full[i][j], np.logical_or.reduce(full[i]))
            and not np.array_equal(full[i][j], np.logical_or.reduce([full[r][j] for r in range(n_rows)]))
            for i in range(n_rows)
            for j in range(n_cols)
            if not occ[i, j]
        ):
            continue
        return occ
    return None


def _render_case(
    row_masks: list[np.ndarray],
    col_masks: list[np.ndarray],
    full: list[list[np.ndarray]],
    occ: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    row_axis = _sample_axis(len(row_masks))
    col_axis = _sample_axis(len(col_masks))
    assert row_axis is not None and col_axis is not None
    h, rows = row_axis
    w, cols = col_axis
    color = random.choice(PALETTE)

    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)

    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            patch = _render_mask(full[i][j], color)
            out[r : r + SLOT, c : c + SLOT] = patch
            if occ[i, j]:
                inp[r : r + SLOT, c : c + SLOT] = patch
    return inp, out, rows, cols


def _validate_generated_case(
    inp: np.ndarray,
    out: np.ndarray,
    row_masks: list[np.ndarray],
    col_masks: list[np.ndarray],
    full: list[list[np.ndarray]],
    occ: np.ndarray,
    rows: list[int],
    cols: list[int],
) -> None:
    solved = main(inp)
    assert np.array_equal(solved, out), "solver must reproduce the planted output"

    color, inferred_rows, inferred_cols, occupied = _infer_instances(inp)
    assert inferred_rows == rows and inferred_cols == cols, "solver must recover the planted lattice"
    row_templates, col_templates = _recover_templates(inferred_rows, inferred_cols, occupied)
    for i, r in enumerate(rows):
        assert np.array_equal(row_templates[r], row_masks[i]), "row template recovery must match planted template"
    for j, c in enumerate(cols):
        assert np.array_equal(col_templates[c], col_masks[j]), "column template recovery must match planted template"

    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            patch = solved[r : r + SLOT, c : c + SLOT] == color
            assert np.array_equal(patch, full[i][j]), "every output slot must match the planted intersection"
            if occ[i, j]:
                input_patch = inp[r : r + SLOT, c : c + SLOT] == color
                assert np.array_equal(input_patch, full[i][j]), "visible sprites must stay unchanged"
            else:
                assert not np.any(inp[r : r + SLOT, c : c + SLOT]), "empty slots must be blank in input"

    assert occ.shape == (len(rows), len(cols))
    assert np.any(~occ), "there must be at least one missing slot"
    assert np.all(occ.sum(axis=1) >= 2) and np.all(occ.sum(axis=0) >= 2), "every row/column needs multiple clues"
    assert len({_mask_id(mask) for row in full for mask in row}) >= len(rows) + len(cols) - 1, "shape set should be varied"


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.randint(*ROW_COUNT_RANGE)
    n_cols = random.randint(*COL_COUNT_RANGE)
    picked = _pick_templates(n_rows, n_cols)
    if picked is None:
        return None
    row_masks, col_masks, full = picked

    occ = _sample_occupancy(full)
    if occ is None:
        return None

    inp, out, rows, cols = _render_case(row_masks, col_masks, full, occ)
    _validate_generated_case(inp, out, row_masks, col_masks, full, occ, rows, cols)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-template lattice input")
