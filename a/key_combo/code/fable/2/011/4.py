from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


def _load_arc_utils():
    here = Path(__file__).resolve()
    for parent in here.parents:
        new_root = parent / "new"
        utils_path = new_root / "solvers" / "arc_utils.py"
        if utils_path.exists():
            if str(new_root) not in sys.path:
                sys.path.insert(0, str(new_root))
            from solvers.arc_utils import Color, colors_present, paste  # type: ignore

            return Color, colors_present, paste
    raise ImportError("Could not locate new/solvers/arc_utils.py from this file")


Color, colors_present, paste = _load_arc_utils()

BG = Color.BLACK
SIDE_OPTIONS = (3, 5)
GEN_RETRIES = 400


def _foreground_color(grid: np.ndarray) -> int:
    fg_colors = colors_present(grid, exclude={BG})
    assert len(fg_colors) == 1, "expected exactly one foreground color on black"
    assert np.any(grid == BG), "expected black background cells"
    assert np.any(grid != BG), "expected at least one foreground cell"
    return int(fg_colors[0])


def _block_categories(mask: np.ndarray) -> dict[str, list[tuple[int, int]]]:
    n = mask.shape[0]
    mid = n // 2
    return {
        "unchanged": [(r, c) for r in range(mid + 1) for c in range(mid + 1) if bool(mask[r, c])],
        "hflip": [(r, c) for r in range(mid + 1) for c in range(mid + 1, n) if bool(mask[r, c])],
        "vflip": [(r, c) for r in range(mid + 1, n) for c in range(mid + 1) if bool(mask[r, c])],
        "rot180": [(r, c) for r in range(mid + 1, n) for c in range(mid + 1, n) if bool(mask[r, c])],
    }


def _independent_expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    n = g.shape[0]
    mid = n // 2
    out = np.full((n * n, n * n), BG, dtype=int)

    for r in range(n):
        for c in range(n):
            if int(g[r, c]) == BG:
                continue
            block = g.copy()
            if c > mid:
                block = block[:, ::-1]
            if r > mid:
                block = block[::-1, :]
            out[r * n : (r + 1) * n, c * n : (c + 1) * n] = block

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each foreground cell into a full input-sized block whose copy faces outward from the center.

    Input structure:
    - The input is a square odd-sized grid with black background and exactly one foreground color.
    - Foreground cells act as block selectors; background cells act as empty block slots.
    - The output is an n-by-n block array where every block has the same shape as the input.

    Rule:
    - Replace every background input cell with an all-black block.
    - Replace every foreground input cell with a copy of the whole input grid.
    - If the cell lies to the right of the center column, flip that copy horizontally; if it lies below the center row, flip it vertically; if both are true, apply both flips, which is a 180-degree rotation.

    Key insight:
    - This is a self-expansion like a Kronecker product, but the copied motif is reoriented by the source block's position so every visible copy points away from the center cross.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert g.shape[0] == g.shape[1], "expected a square grid"
    assert g.shape[0] % 2 == 1, "expected odd side length"
    _foreground_color(g)

    n = g.shape[0]
    mid = n // 2
    out = np.full((n * n, n * n), BG, dtype=int)
    oriented_blocks = {
        (False, False): g,
        (False, True): np.fliplr(g).copy(),
        (True, False): np.flipud(g).copy(),
        (True, True): np.flipud(np.fliplr(g)).copy(),
    }

    for r in range(n):
        for c in range(n):
            if int(g[r, c]) == BG:
                continue
            paste(out, oriented_blocks[(r > mid, c > mid)], r * n, c * n)

    return out


def _sample_mask(side: int) -> np.ndarray:
    mid = side // 2
    all_cells = [(r, c) for r in range(side) for c in range(side)]
    unchanged_all = [(r, c) for r in range(mid + 1) for c in range(mid + 1)]
    unchanged_special = list(dict.fromkeys(
        [(mid, mid)] + [(mid, c) for c in range(mid)] + [(r, mid) for r in range(mid)]
    ))
    h_center = [(mid, c) for c in range(mid + 1, side)]
    h_other = [(r, c) for r in range(mid) for c in range(mid + 1, side)]
    v_center = [(r, mid) for r in range(mid + 1, side)]
    v_other = [(r, c) for r in range(mid + 1, side) for c in range(mid)]
    rot = [(r, c) for r in range(mid + 1, side) for c in range(mid + 1, side)]

    used: set[tuple[int, int]] = set()

    def pick(pool: list[tuple[int, int]]) -> None:
        choices = [cell for cell in pool if cell not in used]
        assert choices, "generator pool unexpectedly empty"
        used.add(random.choice(choices))

    pick(unchanged_special if random.random() < 0.65 else unchanged_all)
    pick(h_center)
    pick(v_center)
    pick(rot)
    if h_other and random.random() < 0.8:
        pick(h_other)
    if v_other and random.random() < 0.8:
        pick(v_other)
    if unchanged_special and random.random() < 0.55:
        pick(unchanged_special)

    if side == 3:
        target = random.randint(max(4, len(used)), 7)
    else:
        target = random.randint(max(6, len(used)), 13)

    extras = [cell for cell in all_cells if cell not in used]
    random.shuffle(extras)
    for cell in extras:
        if len(used) >= target:
            break
        if random.random() < 0.45:
            used.add(cell)

    extras = [cell for cell in all_cells if cell not in used]
    random.shuffle(extras)
    for cell in extras:
        if len(used) >= target:
            break
        used.add(cell)

    mask = np.zeros((side, side), dtype=bool)
    for r, c in used:
        mask[r, c] = True
    return mask


def _standard_block_product(grid: np.ndarray) -> np.ndarray:
    return np.kron((grid != BG).astype(int), grid).astype(int)


def _valid_input(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] != g.shape[1] or g.shape[0] not in SIDE_OPTIONS:
        return False
    if g.shape[0] % 2 != 1:
        return False

    try:
        fg = _foreground_color(g)
    except AssertionError:
        return False

    mask = g == fg
    if int(mask.sum()) >= g.size:
        return False
    if np.array_equal(mask, np.fliplr(mask)):
        return False
    if np.array_equal(mask, np.flipud(mask)):
        return False
    if np.array_equal(mask, np.flipud(np.fliplr(mask))):
        return False

    mid = g.shape[0] // 2
    if not bool(mask[mid, mid + 1 :].any()):
        return False
    if not bool(mask[mid + 1 :, mid].any()):
        return False

    categories = _block_categories(mask)
    if not all(categories.values()):
        return False

    expected = _independent_expected_output(g)
    actual = main(g)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, _standard_block_product(g)):
        return False

    oriented_blocks = {
        "unchanged": g,
        "hflip": np.fliplr(g),
        "vflip": np.flipud(g),
        "rot180": np.flipud(np.fliplr(g)),
    }
    n = g.shape[0]
    for name, coords in categories.items():
        want = oriented_blocks[name]
        for r, c in coords:
            block = actual[r * n : (r + 1) * n, c * n : (c + 1) * n]
            if not np.array_equal(block, want):
                return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        side = random.choice(SIDE_OPTIONS)
        fg = int(random.choice(Color.NOT_BLACK))
        mask = _sample_mask(side)
        grid = np.full((side, side), BG, dtype=int)
        grid[mask] = fg
        if _valid_input(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for outward-oriented block expansion")
