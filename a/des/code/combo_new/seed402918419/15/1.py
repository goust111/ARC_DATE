import numpy as np


BG = 0
SEP = 9
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8)

A_BIT = 1
B_BIT = 2
C_BIT = 4

DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GENERATION_ATTEMPTS = 400
BLOB_ATTEMPTS = 60


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_panels(grid: np.ndarray, bg: int) -> tuple[list[np.ndarray], int]:
    h, w = grid.shape
    sep_cols = [c for c in range(w) if int(grid[0, c]) != bg and np.all(grid[:, c] == grid[0, c])]
    matches: list[tuple[int, int]] = []
    for i, c1 in enumerate(sep_cols):
        for c2 in sep_cols[i + 1 :]:
            left_w = c1
            mid_w = c2 - c1 - 1
            right_w = w - c2 - 1
            if left_w <= 0 or left_w != mid_w or left_w != right_w:
                continue
            if int(grid[0, c1]) != int(grid[0, c2]):
                continue
            matches.append((c1, c2))
    assert len(matches) == 1, "expected exactly two equal-width separator-delimited cuts"
    c1, c2 = matches[0]
    panels = [grid[:, :c1], grid[:, c1 + 1 : c2], grid[:, c2 + 1 :]]
    assert panels[0].shape == panels[1].shape == panels[2].shape
    return panels, int(grid[0, c1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cascade three separator-delimited masks so each layer filters the next one.

    Input structure:
    - The grid contains three equal-size panels arranged left to right.
    - Two full-height non-background separator columns split the panels.
    - Each panel has a black background and a single foreground color forming one or more 4-connected regions.

    Rule:
    - Use the foreground positions of the first panel as a mask on the second panel and erase every second-panel foreground pixel outside that mask.
    - Use the surviving second-panel foreground positions as a new mask on the third panel and erase every third-panel foreground pixel outside that new mask.
    - Return a single-panel grid containing only the remaining third-panel pixels in their original color on a black background.

    Key insight:
    - The three panels are not combined symmetrically; they form a left-to-right filtering pipeline where each residue becomes the next mask.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    panels, sep_color = _find_panels(g, bg)
    assert sep_color != bg

    first, second, third = panels
    second_after_first = np.where(first != bg, second, bg)
    out = np.full(third.shape, bg, dtype=int)
    keep = (second_after_first != bg) & (third != bg)
    out[keep] = third[keep]
    return out


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _blob_neighbors(cell: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _sample_blob(
    rng: np.random.Generator,
    shape: tuple[int, int],
    occupied: np.ndarray,
    min_size: int,
    max_size: int,
) -> np.ndarray | None:
    h, w = shape
    guard = _dilate4(occupied)
    free = np.argwhere(~guard)
    if free.size == 0:
        return None

    min_size = max(1, min_size)
    max_size = max(min_size, max_size)
    for _ in range(BLOB_ATTEMPTS):
        seed_r, seed_c = free[int(rng.integers(len(free)))]
        blob = {(int(seed_r), int(seed_c))}
        frontier = set(_blob_neighbors((int(seed_r), int(seed_c)), h, w))
        target = int(rng.integers(min_size, max_size + 1))

        while frontier and len(blob) < target:
            choice = list(frontier)[int(rng.integers(len(frontier)))]
            frontier.remove(choice)
            if guard[choice]:
                continue
            blob.add(choice)
            for nxt in _blob_neighbors(choice, h, w):
                if nxt not in blob:
                    frontier.add(nxt)
            if len(blob) >= min_size and rng.random() < 0.18:
                break

        if len(blob) >= min_size:
            coords = np.array(sorted(blob), dtype=int)
            return coords
    return None


def _combo_size_bounds(combo: int, panel_area: int) -> tuple[int, int]:
    if combo in (A_BIT | B_BIT | C_BIT, A_BIT | B_BIT):
        high = 6 if panel_area < 64 else 7
        return 3, high
    high = 4 if panel_area < 64 else 5
    return 2, high


def _sample_role_grid(rng: np.random.Generator, h: int, w: int) -> np.ndarray | None:
    roles = np.zeros((h, w), dtype=int)
    mandatory = [
        A_BIT | B_BIT | C_BIT,
        A_BIT | B_BIT,
        int(rng.choice((A_BIT, A_BIT | C_BIT))),
        int(rng.choice((B_BIT, B_BIT | C_BIT))),
        int(rng.choice((C_BIT, A_BIT | C_BIT, B_BIT | C_BIT))),
    ]
    extra_cap = 1 + int(h * w >= 56) + int(h * w >= 72)
    optional = list(
        map(
            int,
            rng.choice(
                (A_BIT, B_BIT, C_BIT, A_BIT | B_BIT, A_BIT | C_BIT, B_BIT | C_BIT, A_BIT | B_BIT | C_BIT),
                size=int(rng.integers(0, extra_cap + 1)),
                replace=True,
            ),
        )
    )
    combos = mandatory + optional
    rng.shuffle(combos)

    for combo in combos:
        low, high = _combo_size_bounds(combo, h * w)
        blob = _sample_blob(rng, (h, w), roles != 0, low, high)
        if blob is None:
            return None
        roles[blob[:, 0], blob[:, 1]] = combo
    return roles


def _masks_from_roles(roles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        (roles & A_BIT) != 0,
        (roles & B_BIT) != 0,
        (roles & C_BIT) != 0,
    )


def _valid_masks(masks: tuple[np.ndarray, np.ndarray, np.ndarray]) -> bool:
    m1, m2, m3 = masks
    panel_area = int(m1.size)
    areas = [int(m.sum()) for m in masks]
    if min(areas) == 0 or max(areas) > int(np.ceil(panel_area * 0.55)):
        return False

    inter12 = m1 & m2
    final = inter12 & m3
    if int(inter12.sum()) < 4 or int(final.sum()) < 2:
        return False
    if not (m1 & ~m2).any():
        return False
    if not (m2 & ~m1).any():
        return False
    if not (inter12 & ~m3).any():
        return False
    if not (m3 & ~inter12).any():
        return False
    return True


def _build_grid(
    masks: tuple[np.ndarray, np.ndarray, np.ndarray],
    colors: tuple[int, int, int],
) -> np.ndarray:
    h, w = masks[0].shape
    grid = np.full((h, 3 * w + 2), BG, dtype=int)
    grid[:, w] = SEP
    grid[:, 2 * w + 1] = SEP
    starts = (0, w + 1, 2 * w + 2)
    for start, mask, color in zip(starts, masks, colors):
        panel = grid[:, start : start + w]
        panel[mask] = int(color)
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(6, 13))
    w = int(rng.integers(6, 10))
    if 3 * w + 2 > 30:
        return None

    roles = _sample_role_grid(rng, h, w)
    if roles is None or len(set(map(int, roles[roles != 0].ravel()))) < 4:
        return None

    masks = _masks_from_roles(roles)
    if not _valid_masks(masks):
        return None

    colors = tuple(map(int, rng.choice(COLOR_POOL, size=3, replace=False)))
    grid = _build_grid(masks, colors)

    expected = np.full((h, w), BG, dtype=int)
    expected[masks[0] & masks[1] & masks[2]] = colors[2]
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
