import hashlib
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
GREEN = 3
MAGENTA = 6
TEAL = 8

PANEL_RANGE = (3, 5)
SEED_COUNT_RANGE = (2, 4)
PANEL_FG_DENSITY_RANGE = (0.30, 0.80)
GEN_MAX_ATTEMPTS = 4000

_SEEN_OUTPUT_SIGNATURES: set[str] = set()


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height % 2 == 1, "expected two panels separated by one row"
    separator_row = height // 2
    assert np.all(g[separator_row] == MAGENTA), "expected a solid magenta separator row"

    top = g[:separator_row]
    bottom = g[separator_row + 1 :]
    assert top.shape == bottom.shape and top.size > 0, "expected two equal nonempty panels"
    assert PANEL_RANGE[0] <= top.shape[0] <= PANEL_RANGE[1], "unexpected panel height"
    assert PANEL_RANGE[0] <= top.shape[1] <= PANEL_RANGE[1], "unexpected panel width"
    assert set(map(int, np.unique(top))).issubset({BLACK, GREEN}), "top panel must use black and green only"
    assert set(map(int, np.unique(bottom))).issubset({BLACK, BLUE}), "bottom panel must use black and blue only"
    return top, bottom


def _seed_from_panels(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    seed = np.zeros_like(top, dtype=int)
    seed[(top == BLACK) & (bottom == BLACK)] = TEAL
    return seed


def _stamp_seed(seed: np.ndarray) -> np.ndarray:
    rows, cols = map(int, seed.shape)
    out = np.zeros((rows * rows, cols * cols), dtype=int)
    for r, c in np.argwhere(seed == TEAL):
        out[r * rows : (r + 1) * rows, c * cols : (c + 1) * cols] = seed
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a self-similar stamp expansion from the background intersection of two aligned panels.

    Input structure:
    - The input is a black grid split into two equal panels by a single solid magenta separator row.
    - The top panel uses only green foreground pixels on black background, and the bottom panel uses only blue foreground pixels on black background.
    - Both panels share the same shape N x M, with N and M between 3 and 5.

    Rule:
    - Remove the separator row and align the two panels by position.
    - Create an N x M seed pattern with teal at cells where both aligned panel cells are black, and black elsewhere.
    - Create an output grid of size (N*N) x (M*M), divided into N x M blocks of size N x M.
    - For every teal cell in the seed, copy the entire seed into the corresponding output block; blocks for black seed cells stay black.

    Key insight:
    - The visible seed is not drawn directly in the input: it is the intersection of the two panels' background masks, and then that inferred seed serves as both the stamp content and the placement mask.
    """
    top, bottom = _split_panels(grid)
    seed = _seed_from_panels(top, bottom)
    return _stamp_seed(seed)


def _component_count(mask: np.ndarray) -> int:
    visited = np.zeros_like(mask, dtype=bool)
    rows, cols = mask.shape
    count = 0
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or visited[r, c]:
                continue
            count += 1
            queue = deque([(r, c)])
            visited[r, c] = True
            while queue:
                rr, cc = queue.popleft()
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
    return count


def _choose_seed_mask(rows: int, cols: int) -> np.ndarray | None:
    want_connected = bool(np.random.randint(0, 2))
    valid_counts = [3, 4] if want_connected else [2, 3, 4]
    valid_counts = [k for k in valid_counts if SEED_COUNT_RANGE[0] <= k <= SEED_COUNT_RANGE[1] and k <= rows * cols]
    if not valid_counts:
        return None

    for _ in range(300):
        count = int(np.random.choice(valid_counts))
        mask = np.zeros((rows, cols), dtype=bool)
        chosen = np.random.choice(rows * cols, size=count, replace=False)
        mask.flat[chosen] = True
        coords = np.argwhere(mask)
        if len(set(coords[:, 0])) < 2 or len(set(coords[:, 1])) < 2:
            continue
        connected = _component_count(mask) == 1
        if connected != want_connected:
            continue
        return mask
    return None


def _density_bounds(total: int) -> tuple[int, int]:
    lo = int(np.ceil(total * PANEL_FG_DENSITY_RANGE[0]))
    hi = int(np.floor(total * PANEL_FG_DENSITY_RANGE[1]))
    return lo, hi


def _choose_categories(seed_mask: np.ndarray) -> np.ndarray | None:
    rows, cols = seed_mask.shape
    total = rows * cols
    nonseed = np.flatnonzero(~seed_mask.ravel())
    lo, hi = _density_bounds(total)

    for _ in range(300):
        cuts = np.sort(np.random.choice(np.arange(1, len(nonseed)), size=2, replace=False))
        counts = (
            int(cuts[0]),
            int(cuts[1] - cuts[0]),
            int(len(nonseed) - cuts[1]),
        )
        labels = np.array([1] * counts[0] + [2] * counts[1] + [3] * counts[2], dtype=int)
        np.random.shuffle(labels)

        categories = np.zeros((rows, cols), dtype=int)
        categories.flat[nonseed] = labels

        top_fg = int(np.count_nonzero(np.isin(categories, (2, 3))))
        bottom_fg = int(np.count_nonzero(np.isin(categories, (1, 3))))
        if not (lo <= top_fg <= hi and lo <= bottom_fg <= hi):
            continue
        return categories
    return None


def _build_panels(seed_mask: np.ndarray, categories: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    top = np.zeros(seed_mask.shape, dtype=int)
    bottom = np.zeros(seed_mask.shape, dtype=int)
    top[np.isin(categories, (2, 3))] = GREEN
    bottom[np.isin(categories, (1, 3))] = BLUE
    return top, bottom


def _render_input(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    separator = np.full((1, top.shape[1]), MAGENTA, dtype=int)
    return np.vstack([top, separator, bottom])


def _solid_scale(seed: np.ndarray) -> np.ndarray:
    return np.kron(seed, np.ones(seed.shape, dtype=int))


def _signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _semantic_check(inp: np.ndarray) -> bool:
    try:
        top, bottom = _split_panels(inp)
    except AssertionError:
        return False

    seed = _seed_from_panels(top, bottom)
    seed_mask = seed == TEAL
    seed_count = int(np.count_nonzero(seed_mask))
    if not (SEED_COUNT_RANGE[0] <= seed_count <= SEED_COUNT_RANGE[1]):
        return False

    coords = np.argwhere(seed_mask)
    if len(set(coords[:, 0])) < 2 or len(set(coords[:, 1])) < 2:
        return False

    top_only_black = int(np.count_nonzero((top == BLACK) & (bottom == BLUE)))
    bottom_only_black = int(np.count_nonzero((top == GREEN) & (bottom == BLACK)))
    both_fg = int(np.count_nonzero((top == GREEN) & (bottom == BLUE)))
    if min(top_only_black, bottom_only_black, both_fg) < 1:
        return False

    out = main(inp)
    if out.shape != (top.shape[0] * top.shape[0], top.shape[1] * top.shape[1]):
        return False

    if np.array_equal(out, _solid_scale(seed)):
        return False

    top_black_seed = np.where(top == BLACK, TEAL, BLACK)
    bottom_black_seed = np.where(bottom == BLACK, TEAL, BLACK)
    fg_intersection_seed = np.where((top == GREEN) & (bottom == BLUE), TEAL, BLACK)
    bg_union_seed = np.where((top == BLACK) | (bottom == BLACK), TEAL, BLACK)
    wrong_candidates = (
        _stamp_seed(top_black_seed),
        _stamp_seed(bottom_black_seed),
        _stamp_seed(fg_intersection_seed),
        _stamp_seed(bg_union_seed),
    )
    if any(np.array_equal(out, wrong) for wrong in wrong_candidates):
        return False

    return True


def _generate_case() -> np.ndarray | None:
    rows = int(np.random.randint(PANEL_RANGE[0], PANEL_RANGE[1] + 1))
    cols = int(np.random.randint(PANEL_RANGE[0], PANEL_RANGE[1] + 1))

    seed_mask = _choose_seed_mask(rows, cols)
    if seed_mask is None:
        return None

    categories = _choose_categories(seed_mask)
    if categories is None:
        return None

    top, bottom = _build_panels(seed_mask, categories)
    inp = _render_input(top, bottom)
    if not _semantic_check(inp):
        return None

    out_sig = _signature(main(inp))
    if out_sig in _SEEN_OUTPUT_SIGNATURES:
        return None
    _SEEN_OUTPUT_SIGNATURES.add(out_sig)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return case
    raise RuntimeError("failed to generate a valid non-degenerate input")
