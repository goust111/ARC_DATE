from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
ORANGE = 7

SCALE = 3

GREY_BLOCK = np.array(
    [
        [GREY, GREY, GREY],
        [GREY, RED, GREY],
        [GREY, GREY, GREY],
    ],
    dtype=int,
)

BLUE_BLOCK = np.array(
    [
        [YELLOW, BLUE, YELLOW],
        [BLUE, BLUE, BLUE],
        [YELLOW, BLUE, YELLOW],
    ],
    dtype=int,
)

BLACK_BLOCK = np.zeros((SCALE, SCALE), dtype=int)

# --------------- generator configuration ---------------

MIN_SIZE = 5
MAX_SIZE = 9
MIN_FOREGROUND = 4
MAX_FOREGROUND = 7
MAX_RETRIES = 400


def _pattern_for(color: int) -> np.ndarray:
    if color == BLACK:
        return BLACK_BLOCK
    if color == GREY:
        return GREY_BLOCK
    if color == BLUE:
        return BLUE_BLOCK
    raise AssertionError(f"unexpected color: {color}")


def _expand_grid(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    out = np.zeros((h * SCALE, w * SCALE), dtype=int)
    for r in range(h):
        for c in range(w):
            out[SCALE * r : SCALE * r + SCALE, SCALE * c : SCALE * c + SCALE] = _pattern_for(int(grid[r, c]))
    return out


def _apply_offset_shadow(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    # In-place row-major scanning lets newly painted orange pixels keep projecting.
    for r in range(h - 1):
        for c in range(w - 1):
            if out[r, c] != BLACK and out[r + 1, c + 1] == BLACK:
                out[r + 1, c + 1] = ORANGE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Expand grey and blue pixels into different 3x3 motifs, then project an in-place orange down-right shadow through black gaps.

    Input structure:
    - The input is a black background grid containing only black, grey, and blue.
    - Grey and blue are the only foreground colors.
    - Each foreground pixel is interpreted independently before any shadow is added.

    Rule:
    - Scale the grid by 3 in both dimensions.
    - A black cell becomes a 3x3 all-black block.
    - A grey cell becomes a 3x3 block with a red center and grey elsewhere.
    - A blue cell becomes a 3x3 block with yellow corners and blue cross positions.
    - After the scale-up, scan the enlarged grid in row-major order. Every non-black cell paints orange at its down-right neighbor when that target is black; already non-black cells are never overwritten, and newly painted orange cells continue projecting later in the same scan.

    Key insight:
    - The task is a composition of two deterministic stages: color-conditioned local 3x3 substitution, followed by a global in-place diagonal shadow that only occupies remaining black cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    assert set(map(int, np.unique(g))) <= {BLACK, BLUE, GREY}, "input may only use black, blue, and grey"

    return _apply_offset_shadow(_expand_grid(g))


def _single_step_shadow(base: np.ndarray) -> np.ndarray:
    """One non-cascading shadow pass used only for generator validation."""
    out = base.copy()
    h, w = base.shape
    shadow_targets: list[tuple[int, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            if base[r, c] != BLACK and base[r + 1, c + 1] == BLACK:
                shadow_targets.append((r + 1, c + 1))
    for r, c in shadow_targets:
        out[r, c] = ORANGE
    return out


def _foreground_coords(grid: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(grid != BLACK)]


def _orthogonally_separated(coords: list[tuple[int, int]]) -> bool:
    coord_set = set(coords)
    for r, c in coords:
        if (r - 1, c) in coord_set or (r + 1, c) in coord_set or (r, c - 1) in coord_set or (r, c + 1) in coord_set:
            return False
    return True


def _has_diagonal_pair(coords: list[tuple[int, int]]) -> bool:
    coord_set = set(coords)
    for r, c in coords:
        if (r + 1, c + 1) in coord_set or (r + 1, c - 1) in coord_set:
            return True
    return False


def _sample_positions(h: int, w: int, count: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    coords: list[tuple[int, int]] = []
    blocked = np.zeros((h, w), dtype=bool)

    if count >= 2 and h >= 2 and w >= 2 and rng.random() < 0.7:
        diagonal_starts = [(r, c) for r in range(h - 1) for c in range(w - 1)]
        rng.shuffle(diagonal_starts)
        for r, c in diagonal_starts:
            if blocked[r, c] or blocked[r + 1, c + 1]:
                continue
            coords.extend([(r, c), (r + 1, c + 1)])
            for rr, cc in ((r, c), (r + 1, c + 1)):
                blocked[rr, cc] = True
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w:
                        blocked[nr, nc] = True
            break

    while len(coords) < count:
        candidates = np.argwhere(~blocked)
        if candidates.size == 0:
            return None
        r, c = map(int, candidates[int(rng.integers(len(candidates)))])
        coords.append((r, c))
        blocked[r, c] = True
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True
    return coords


def _semantic_checks(grid: np.ndarray, base: np.ndarray, out: np.ndarray) -> bool:
    h, w = grid.shape
    if max(h, w) > 30 or max(out.shape) > 30:
        return False

    coords = _foreground_coords(grid)
    if not (MIN_FOREGROUND <= len(coords) <= MAX_FOREGROUND):
        return False
    if not _orthogonally_separated(coords):
        return False

    colors = set(map(int, grid[grid != BLACK]))
    if colors != {BLUE, GREY}:
        return False

    # Every sample should teach both substitution branches.
    if RED not in out or YELLOW not in out:
        return False

    # Keep the shadow visible and ensure the in-place cascading behavior matters.
    orange_count = int(np.count_nonzero(out == ORANGE))
    if orange_count < 4:
        return False
    one_step = _single_step_shadow(base)
    if orange_count <= int(np.count_nonzero(one_step == ORANGE)):
        return False

    # Avoid all samples collapsing into the easiest "totally isolated points" layout.
    if not _has_diagonal_pair(coords) and rng_bit_from_grid(grid) == 0:
        return False

    return True


def rng_bit_from_grid(grid: np.ndarray) -> int:
    """Deterministic bit used only to diversify rejection checks without external state."""
    return int((int(grid.shape[0]) + int(grid.shape[1]) + int(np.count_nonzero(grid == GREY))) % 2)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(MIN_SIZE, MAX_SIZE + 1))
    w = int(rng.integers(MIN_SIZE, MAX_SIZE + 1))
    count = int(rng.integers(MIN_FOREGROUND, MAX_FOREGROUND + 1))

    coords = _sample_positions(h, w, count, rng)
    if coords is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    grey_count = int(rng.integers(1, count))
    palette = np.array([GREY] * grey_count + [BLUE] * (count - grey_count), dtype=int)
    rng.shuffle(palette)
    for (r, c), color in zip(coords, palette, strict=False):
        grid[r, c] = int(color)

    base = _expand_grid(grid)
    expected = _apply_offset_shadow(base)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _semantic_checks(grid, base, actual):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
