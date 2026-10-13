import numpy as np


BG = 0
SEP = 5
PALETTE = (1, 2, 3, 4)
CELL_SIZE = 4
LATTICE_OPTIONS = (3, 4)
GEN_RETRIES = 400


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for pos in sorted(int(x) for x in separators):
        if start < pos:
            bands.append((start, pos))
        start = pos + 1
    if start < size:
        bands.append((start, size))
    return bands


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _component_count(mask: np.ndarray) -> int:
    return len(_component_coords(mask))


def _separator_structure(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    row_colors = {int(g[r, 0]) for r in range(h) if np.all(g[r, :] == g[r, 0])}
    col_colors = {int(g[0, c]) for c in range(w) if np.all(g[:, c] == g[0, c])}
    sep_colors = row_colors & col_colors
    assert len(sep_colors) == 1, "expected exactly one separator color"
    sep = next(iter(sep_colors))
    sep_rows = [r for r in range(h) if np.all(g[r, :] == sep)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == sep)]
    assert sep_rows and sep_cols, "expected full separator rows and columns"
    row_bands = _bands(h, sep_rows)
    col_bands = _bands(w, sep_cols)
    assert row_bands and col_bands, "separator lattice must define non-empty cells"
    row_sizes = {r1 - r0 for r0, r1 in row_bands}
    col_sizes = {c1 - c0 for c0, c1 in col_bands}
    assert len(row_sizes) == 1 and len(col_sizes) == 1, "lattice cells must be uniform"
    return sep, row_bands, col_bands


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use component counts inside separator-lattice cells to choose which mirror to apply.

    Input structure:
    - Full gray separator rows and columns divide the grid into equal rectangular cells.
    - Each cell interior has black background and one colored figure made of 1-3 distinct 4-connected components.
    - Separator pixels stay outside the per-cell figures and define the independent processing regions.

    Rule:
    - Detect the separator lattice from full-width and full-height lines.
    - For each cell interior, count 4-connected non-black, non-separator components.
    - If the count is 1, flip the entire cell interior top-to-bottom; if the count is 2, flip it left-to-right; if the count is 3 or more, leave it unchanged.
    - Write the transformed cell interiors back into the same lattice while preserving the separator lines.

    Key insight:
    - The component count does not become output content itself; it selects which geometric transformation each lattice cell receives.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep, row_bands, col_bands = _separator_structure(g)
    out = g.copy()
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            cell = g[r0:r1, c0:c1]
            fg = (cell != BG) & (cell != sep)
            ncomp = _component_count(fg)
            if ncomp == 1:
                out[r0:r1, c0:c1] = np.flipud(cell)
            elif ncomp == 2:
                out[r0:r1, c0:c1] = np.fliplr(cell)
            else:
                out[r0:r1, c0:c1] = cell
    return out


def _components_have_gap(comps: list[list[tuple[int, int]]]) -> bool:
    comp_sets = [set(comp) for comp in comps]
    for i in range(len(comp_sets)):
        for j in range(i + 1, len(comp_sets)):
            for r, c in comp_sets[i]:
                for rr in range(r - 1, r + 2):
                    for cc in range(c - 1, c + 2):
                        if (rr, cc) in comp_sets[j]:
                            return False
    return True


def _enumerate_candidate_masks() -> dict[int, dict[int, list[np.ndarray]]]:
    buckets: dict[int, dict[int, list[np.ndarray]]] = {1: {}, 2: {}, 3: {}}
    for bits in range(1, 1 << (CELL_SIZE * CELL_SIZE)):
        area = bits.bit_count()
        if area < 2 or area > 12:
            continue
        flat = np.array([(bits >> i) & 1 for i in range(CELL_SIZE * CELL_SIZE)], dtype=bool)
        mask = flat.reshape(CELL_SIZE, CELL_SIZE)
        comps = _component_coords(mask)
        ncomp = len(comps)
        if ncomp not in buckets:
            continue
        sizes = [len(comp) for comp in comps]
        if not all(2 <= size <= 4 for size in sizes):
            continue
        if not _components_have_gap(comps):
            continue
        if ncomp == 1 and np.array_equal(mask, np.flipud(mask)):
            continue
        if ncomp == 2 and np.array_equal(mask, np.fliplr(mask)):
            continue
        buckets[ncomp].setdefault(area, []).append(mask.copy())

    assert buckets[1].get(4), "need asymmetric 1-component area-4 masks"
    assert buckets[2].get(4), "need asymmetric 2-component area-4 masks"
    assert buckets[2].get(6), "need asymmetric 2-component area-6 masks"
    assert buckets[3].get(6), "need 3-component area-6 masks"
    return buckets


CANDIDATE_MASKS = _enumerate_candidate_masks()


def _cell_patch(mask: np.ndarray, color: int) -> np.ndarray:
    cell = np.zeros((CELL_SIZE, CELL_SIZE), dtype=int)
    cell[mask] = int(color)
    return cell


def _random_mask(rng: np.random.Generator, ncomp: int, area: int | None = None) -> np.ndarray:
    by_area = CANDIDATE_MASKS[ncomp]
    if area is None:
        area = int(rng.choice(np.array(sorted(by_area.keys()), dtype=int)))
    candidates = by_area[int(area)]
    return candidates[int(rng.integers(len(candidates)))]


def _expected_from_input(inp: np.ndarray) -> np.ndarray:
    sep, row_bands, col_bands = _separator_structure(inp)
    out = inp.copy()
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            cell = inp[r0:r1, c0:c1]
            fg = (cell != BG) & (cell != sep)
            ncomp = _component_count(fg)
            if ncomp == 1:
                out[r0:r1, c0:c1] = np.flipud(cell)
            elif ncomp == 2:
                out[r0:r1, c0:c1] = np.fliplr(cell)
            else:
                out[r0:r1, c0:c1] = cell
    return out


def _validate_semantics(inp: np.ndarray) -> bool:
    try:
        sep, row_bands, col_bands = _separator_structure(inp)
        if sep != SEP:
            return False
        if len(row_bands) not in LATTICE_OPTIONS or len(col_bands) != len(row_bands):
            return False
        if any((r1 - r0) != CELL_SIZE for r0, r1 in row_bands):
            return False
        if any((c1 - c0) != CELL_SIZE for c0, c1 in col_bands):
            return False

        counts_seen = {1: 0, 2: 0, 3: 0}
        areas_by_count: dict[int, set[int]] = {1: set(), 2: set(), 3: set()}
        color_to_counts: dict[int, set[int]] = {}
        out = main(inp)
        changed_1 = 0
        changed_2 = 0

        for r0, r1 in row_bands:
            for c0, c1 in col_bands:
                cell = inp[r0:r1, c0:c1]
                fg = (cell != BG) & (cell != sep)
                colors = sorted(int(v) for v in np.unique(cell[fg]))
                if len(colors) != 1:
                    return False
                color = colors[0]
                ncomp = _component_count(fg)
                if ncomp not in (1, 2, 3):
                    return False
                counts_seen[ncomp] += 1
                area = int(fg.sum())
                areas_by_count[ncomp].add(area)
                color_to_counts.setdefault(color, set()).add(ncomp)

                if ncomp == 1:
                    expected = np.flipud(cell)
                    if np.array_equal(expected, cell):
                        return False
                    changed_1 += int(not np.array_equal(expected, cell))
                elif ncomp == 2:
                    expected = np.fliplr(cell)
                    if np.array_equal(expected, cell):
                        return False
                    changed_2 += int(not np.array_equal(expected, cell))
                else:
                    expected = cell
                if not np.array_equal(out[r0:r1, c0:c1], expected):
                    return False

        if not all(counts_seen[k] >= 1 for k in counts_seen):
            return False
        if 4 not in areas_by_count[1] or 4 not in areas_by_count[2]:
            return False
        if 6 not in areas_by_count[2] or 6 not in areas_by_count[3]:
            return False
        if changed_1 == 0 or changed_2 == 0:
            return False
        if not any(len(v) >= 2 for v in color_to_counts.values()):
            return False
        if not np.array_equal(out, _expected_from_input(inp)):
            return False
        return True
    except AssertionError:
        return False


def _assemble_grid(lattice: int, cells: list[np.ndarray]) -> np.ndarray:
    size = lattice * CELL_SIZE + (lattice - 1)
    grid = np.zeros((size, size), dtype=int)
    stride = CELL_SIZE + 1
    for k in range(1, lattice):
        pos = k * CELL_SIZE + (k - 1)
        grid[pos, :] = SEP
        grid[:, pos] = SEP
    idx = 0
    for ri in range(lattice):
        for ci in range(lattice):
            r0 = ri * stride
            c0 = ci * stride
            grid[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE] = cells[idx]
            idx += 1
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    lattice = int(rng.choice(np.array(LATTICE_OPTIONS, dtype=int)))
    n_cells = lattice * lattice

    colors = np.array(PALETTE, dtype=int)
    rng.shuffle(colors)
    same_color_a = int(colors[0])
    same_color_b = int(colors[1])

    plans = [
        (1, 4, same_color_a),
        (2, 4, same_color_a),
        (2, 6, same_color_b),
        (3, 6, same_color_b),
    ]

    extra_counts: list[int] = []
    while len(plans) + len(extra_counts) < n_cells:
        perm = [1, 2, 3]
        rng.shuffle(perm)
        extra_counts.extend(perm)

    for ncomp in extra_counts[: n_cells - len(plans)]:
        areas = np.array(sorted(CANDIDATE_MASKS[ncomp].keys()), dtype=int)
        area = int(rng.choice(areas))
        color = int(rng.choice(np.array(PALETTE, dtype=int)))
        plans.append((ncomp, area, color))

    order = rng.permutation(n_cells)
    cell_patches = [np.zeros((CELL_SIZE, CELL_SIZE), dtype=int) for _ in range(n_cells)]
    for idx, (ncomp, area, color) in zip(order, plans):
        mask = _random_mask(rng, ncomp, area)
        cell_patches[int(idx)] = _cell_patch(mask, color)

    inp = _assemble_grid(lattice, cell_patches)
    if not _validate_semantics(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid selective-mirroring lattice input")
