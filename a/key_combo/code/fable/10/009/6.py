import random

import numpy as np


BACKGROUND = 0
GRID_SHAPE = (4, 4)
RETRY_LIMIT = 500
ORIGINAL = "original"
LR = "lr"
UD = "ud"
ROT180 = "rot180"
ALL_ORIENTATIONS = (ORIGINAL, LR, UD, ROT180)


def _variants(grid: np.ndarray) -> dict[str, np.ndarray]:
    return {
        ORIGINAL: grid,
        LR: np.fliplr(grid).copy(),
        UD: np.flipud(grid).copy(),
        ROT180: np.rot90(grid, 2).copy(),
    }


def _foreground_colors(grid: np.ndarray) -> list[int]:
    return [int(color) for color in np.unique(grid) if int(color) != BACKGROUND]


def _nontrivial_matches(grid: np.ndarray, r: int, c: int) -> list[str]:
    h, w = grid.shape
    color = int(grid[r, c])
    matches: list[str] = []

    if w - 1 - c != c and int(grid[r, w - 1 - c]) == color:
        matches.append(LR)
    if h - 1 - r != r and int(grid[h - 1 - r, c]) == color:
        matches.append(UD)
    if (h - 1 - r, w - 1 - c) != (r, c) and int(grid[h - 1 - r, w - 1 - c]) == color:
        matches.append(ROT180)
    return matches


def _choose_orientation(grid: np.ndarray, r: int, c: int) -> str:
    color = int(grid[r, c])
    assert color != BACKGROUND
    matches = _nontrivial_matches(grid, r, c)
    if len(matches) == 0:
        return ORIGINAL
    assert len(matches) == 1, f"ambiguous self-proof at {(r, c)}: {matches}"
    return matches[0]


def _orientation_map(grid: np.ndarray) -> dict[tuple[int, int], str]:
    mapping: dict[tuple[int, int], str] = {}
    for r, c in np.argwhere(grid != BACKGROUND):
        rr = int(r)
        cc = int(c)
        mapping[(rr, cc)] = _choose_orientation(grid, rr, cc)
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the seed into each marked block using the only nontrivial self-proofing orientation.

    Input structure:
    - The input is a small grid containing black plus exactly two foreground colors.
    - The same grid serves both as the seed sprite and as the mask of block locations.
    - The seed has no left-right, up-down, or 180-degree symmetry.

    Rule:
    - Partition the output into a block grid with the same number of rows and columns as the input,
      and give every block the same shape as the input itself.
    - For each non-black input cell (r, c), compare that host color against the three nontrivial
      orientations at the same local coordinate: left-right mirror, up-down mirror, and 180-degree
      rotation.
    - If exactly one of those nontrivial orientations shows the host color at (r, c), stamp that
      oriented seed into the corresponding output block; if none do, stamp the original seed.
    - Leave blocks for black input cells entirely black.

    Key insight:
    - Each block chooses its orientation by asking whether the host color can prove itself from a
      mirrored or half-turned copy at the very same slot; only cells with no nontrivial witness keep
      the original pose.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)
    assert h * h <= 30 and w * w <= 30, "output would exceed ARC size limit"

    foreground = _foreground_colors(g)
    assert len(foreground) == 2, "expected exactly two foreground colors"

    variants = _variants(g)
    assert not np.array_equal(g, variants[LR])
    assert not np.array_equal(g, variants[UD])
    assert not np.array_equal(g, variants[ROT180])

    out = np.full((h * h, w * w), BACKGROUND, dtype=int)
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) == BACKGROUND:
                continue
            orientation = _choose_orientation(g, r, c)
            out[r * h : (r + 1) * h, c * w : (c + 1) * w] = variants[orientation]
    return out


def _orbit_positions(rep_r: int, rep_c: int) -> list[tuple[int, int]]:
    h, w = GRID_SHAPE
    return [
        (rep_r, rep_c),
        (rep_r, w - 1 - rep_c),
        (h - 1 - rep_r, rep_c),
        (h - 1 - rep_r, w - 1 - rep_c),
    ]


def _paint_orbit(grid: np.ndarray, orbit: list[tuple[int, int]], category: str, colors: tuple[int, int]) -> None:
    color_a, color_b = colors
    p0, p1, p2, p3 = orbit

    if category == ORIGINAL:
        if random.random() < 0.5:
            chosen = random.choice([p0, p1, p2, p3])
            grid[chosen] = random.choice(colors)
        else:
            chosen = random.sample([p0, p1, p2, p3], k=2)
            grid[chosen[0]] = color_a
            grid[chosen[1]] = color_b
        return

    if category == LR:
        top_row = random.random() < 0.5
        pair = (p0, p1) if top_row else (p2, p3)
        rest = [p2, p3] if top_row else [p0, p1]
    elif category == UD:
        left_col = random.random() < 0.5
        pair = (p0, p2) if left_col else (p1, p3)
        rest = [p1, p3] if left_col else [p0, p2]
    else:
        main_diag = random.random() < 0.5
        pair = (p0, p3) if main_diag else (p1, p2)
        rest = [p1, p2] if main_diag else [p0, p3]

    pair_color = random.choice(colors)
    for cell in pair:
        grid[cell] = pair_color

    mode = random.choice(["pair_only", "pair_plus_singleton", "double_pair"])
    other_color = color_b if pair_color == color_a else color_a
    if mode == "pair_plus_singleton":
        grid[random.choice(rest)] = other_color
    elif mode == "double_pair":
        for cell in rest:
            grid[cell] = other_color


def _role_index(pos: tuple[int, int]) -> int:
    r, c = pos
    if r < GRID_SHAPE[0] // 2 and c < GRID_SHAPE[1] // 2:
        return 0
    if r < GRID_SHAPE[0] // 2 and c >= GRID_SHAPE[1] // 2:
        return 1
    if r >= GRID_SHAPE[0] // 2 and c < GRID_SHAPE[1] // 2:
        return 2
    return 3


def _validate_seed(grid: np.ndarray) -> bool:
    if tuple(grid.shape) != GRID_SHAPE:
        return False
    if len(_foreground_colors(grid)) != 2:
        return False
    if np.count_nonzero(grid != BACKGROUND) < 6:
        return False

    variants = _variants(grid)
    if any(np.array_equal(grid, variants[name]) for name in (LR, UD, ROT180)):
        return False

    try:
        orientations = _orientation_map(grid)
    except AssertionError:
        return False

    branch_set = set(orientations.values())
    if branch_set != set(ALL_ORIENTATIONS):
        return False

    branches_by_color: dict[int, set[str]] = {}
    for (r, c), branch in orientations.items():
        color = int(grid[r, c])
        branches_by_color.setdefault(color, set()).add(branch)
    if any(len(branches) < 2 for branches in branches_by_color.values()):
        return False

    branches_by_role: dict[int, set[str]] = {}
    for pos, branch in orientations.items():
        branches_by_role.setdefault(_role_index(pos), set()).add(branch)
    if sum(len(branches) >= 2 for branches in branches_by_role.values()) < 2:
        return False

    out = main(grid)
    if out.shape != (GRID_SHAPE[0] * GRID_SHAPE[0], GRID_SHAPE[1] * GRID_SHAPE[1]):
        return False

    for r in range(GRID_SHAPE[0]):
        for c in range(GRID_SHAPE[1]):
            block = out[r * GRID_SHAPE[0] : (r + 1) * GRID_SHAPE[0], c * GRID_SHAPE[1] : (c + 1) * GRID_SHAPE[1]]
            if int(grid[r, c]) == BACKGROUND:
                if np.any(block != BACKGROUND):
                    return False
            else:
                expected = variants[orientations[(r, c)]]
                if not np.array_equal(block, expected):
                    return False

    all_original = np.full_like(out, BACKGROUND)
    for r in range(GRID_SHAPE[0]):
        for c in range(GRID_SHAPE[1]):
            if int(grid[r, c]) != BACKGROUND:
                all_original[r * GRID_SHAPE[0] : (r + 1) * GRID_SHAPE[0], c * GRID_SHAPE[1] : (c + 1) * GRID_SHAPE[1]] = grid
    if np.array_equal(out, all_original):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    colors = tuple(int(x) for x in random.sample(range(1, 10), k=2))
    grid = np.full(GRID_SHAPE, BACKGROUND, dtype=int)

    categories = list(ALL_ORIENTATIONS)
    random.shuffle(categories)
    representatives = [(0, 0), (0, 1), (1, 0), (1, 1)]
    random.shuffle(representatives)

    for category, rep in zip(categories, representatives):
        orbit = _orbit_positions(*rep)
        _paint_orbit(grid, orbit, category, colors)

    if not _validate_seed(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
