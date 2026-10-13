import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _is_connected4(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(int(v) for v in coords[0])
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                nxt = (int(nr), int(nc))
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return len(seen) == int(mask.sum())


def _enumerate_valid_seeds() -> dict[int, list[np.ndarray]]:
    seeds_by_size = {k: [] for k in range(3, 7)}
    for bits in range(1 << 9):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        fg = int(mask.sum())
        if 3 <= fg <= 6 and _is_connected4(mask):
            seeds_by_size[fg].append(mask)
    return seeds_by_size


SEEDS_BY_SIZE = _enumerate_valid_seeds()


def _canonical_panels_from_seed(seed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = [tuple(int(v) for v in rc) for rc in np.argwhere(seed)]
    shared = {coords[0]}
    left_only = set(coords[1::2])
    right_only = set(coords[2::2])

    left = np.zeros((3, 3), dtype=bool)
    right = np.zeros((3, 3), dtype=bool)
    for r, c in shared | left_only:
        left[r, c] = True
    for r, c in shared | right_only:
        right[r, c] = True

    assert 2 <= int(left.sum()) <= 4
    assert 2 <= int(right.sum()) <= 4
    assert int((left & right).sum()) >= 1
    assert int((left & ~right).sum()) >= 1
    assert int((right & ~left).sum()) >= 1
    assert np.array_equal(left | right, seed)
    return left, right


def _build_input_from_seed(seed: np.ndarray) -> np.ndarray:
    left_mask, right_mask = _canonical_panels_from_seed(seed)
    grid = np.zeros((3, 7), dtype=int)
    grid[:, 3] = GREY
    grid[:, :3][left_mask] = YELLOW
    grid[:, 4:7][right_mask] = GREEN
    return grid


def _expected_output_from_seed(seed: np.ndarray) -> np.ndarray:
    color_selector = np.where(seed, RED, BLUE).astype(int)
    return np.kron(color_selector, seed.astype(int))


def _extract_panels(input_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grid = np.asarray(input_grid, dtype=int)
    grey_coords = np.argwhere(grid == GREY)
    assert grey_coords.shape[0] == 3, "expected exactly three grey separator cells"
    r0, c0, r1, c1 = _bbox_coords(grey_coords)
    assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 1), "separator must be a 3x1 vertical bar"
    assert c0 >= 3 and c0 + 4 <= grid.shape[1], "separator must have one 3x3 panel on each side"
    left = grid[r0 : r0 + 3, c0 - 3 : c0]
    right = grid[r0 : r0 + 3, c0 + 1 : c0 + 4]
    assert left.shape == (3, 3) and right.shape == (3, 3)
    return left, right


def _seed_from_input(input_grid: np.ndarray) -> np.ndarray:
    left, right = _extract_panels(input_grid)
    return (left != BLACK) | (right != BLACK)


def _validate_generated_input(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.shape == (3, 7)
    left, right = _extract_panels(grid)
    assert set(np.unique(grid)).issubset({BLACK, GREEN, YELLOW, GREY})
    assert np.all((left == BLACK) | (left == YELLOW))
    assert np.all((right == BLACK) | (right == GREEN))

    left_mask = left != BLACK
    right_mask = right != BLACK
    seed = left_mask | right_mask

    assert 2 <= int(left_mask.sum()) <= 4
    assert 2 <= int(right_mask.sum()) <= 4
    assert 3 <= int(seed.sum()) <= 6
    assert _is_connected4(seed)
    assert int((left_mask & right_mask).sum()) >= 1
    assert int((left_mask & ~right_mask).sum()) >= 1
    assert int((right_mask & ~left_mask).sum()) >= 1
    assert np.array_equal(grid, _build_input_from_seed(seed))
    assert np.array_equal(main(grid), _expected_output_from_seed(seed))


def _verify_catalog_uniqueness() -> None:
    total = 0
    input_sigs = set()
    output_sigs = set()
    for seed_list in SEEDS_BY_SIZE.values():
        for seed in seed_list:
            total += 1
            inp = _build_input_from_seed(seed)
            out = _expected_output_from_seed(seed)
            input_sigs.add(inp.tobytes())
            output_sigs.add(out.tobytes())
    assert len(input_sigs) == total
    assert len(output_sigs) == total


_verify_catalog_uniqueness()


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Build a 9x9 self-similar pattern from the boolean union of two side-by-side 3x3 panels.

    Input structure:
    - The input contains two 3x3 panels separated by a one-cell grey vertical divider.
    - Yellow pixels mark foreground in the left panel, green pixels mark foreground in the right panel, and black is background.
    - The two panels share at least one foreground coordinate and each panel also has at least one foreground cell that the other panel lacks.

    Rule:
    - Discard the grey divider and read the two adjacent 3x3 panels.
    - Compute a 3x3 seed mask by boolean union: a cell is foreground iff either panel has a non-black pixel there.
    - Replace each foreground seed cell by a red copy of the seed mask and each background seed cell by a blue copy of the same seed mask, keeping local seed-background cells black, to produce a 9x9 output.

    Key insight:
    - Neither panel is the recursive seed on its own; the seed only appears after unioning the two panel masks, and that derived mask drives the entire expansion.
    """
    seed = _seed_from_input(np.asarray(input_grid, dtype=int))
    return _expected_output_from_seed(seed)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    seed_size = int(rng.choice(np.array([3, 4, 5, 6], dtype=int)))
    seed_bank = SEEDS_BY_SIZE[seed_size]
    seed = seed_bank[int(rng.integers(len(seed_bank)))]
    grid = _build_input_from_seed(seed)
    _validate_generated_input(grid)
    return grid
