import random
from collections import Counter, deque

import numpy as np


RED = 2
GREEN = 3
YELLOW = 4

INPUT_PALETTE = (0, 1, 5, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_RETRIES = 400
MIN_COMPONENTS = 8


def _components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            comps.append({"color": color, "coords": coords, "size": len(coords)})

    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each 4-connected island by its size parity, except every largest island becomes yellow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    comps = _components(g)
    max_size = max(comp["size"] for comp in comps)

    out = np.empty_like(g)
    for comp in comps:
        size = int(comp["size"])
        if size == max_size:
            fill = YELLOW
        else:
            fill = RED if size % 2 == 1 else GREEN
        for r, c in comp["coords"]:
            out[r, c] = fill

    return out


def _choose_mode(rng: random.Random) -> dict:
    if rng.random() < 0.55:
        return {
            "max_h": 2,
            "nonmax_h_choices": (1,),
            "max_w_choices": (4, 5),
        }
    return {
        "max_h": 3,
        "nonmax_h_choices": (1, 2),
        "max_w_choices": (3,),
    }


def _sample_band_sizes(
    rng: random.Random,
    count: int,
    max_value: int,
    nonmax_choices: tuple[int, ...],
    max_count: int,
) -> list[int]:
    max_indices = set(rng.sample(range(count), max_count))
    sizes = []
    for idx in range(count):
        if idx in max_indices:
            sizes.append(int(max_value))
        else:
            sizes.append(int(rng.choice(nonmax_choices)))
    return sizes


def _force_required_values(
    sizes: list[int],
    required: list[int],
    nonmax_value: int,
    max_value: int,
    rng: random.Random,
) -> None:
    candidate_idxs = [i for i, v in enumerate(sizes) if v != max_value]
    if not candidate_idxs:
        return
    rng.shuffle(candidate_idxs)
    for value in required:
        if value in sizes:
            continue
        sizes[candidate_idxs.pop()] = int(value)
        if not candidate_idxs:
            candidate_idxs = [i for i, v in enumerate(sizes) if v != max_value]
            rng.shuffle(candidate_idxs)
            if not candidate_idxs:
                break
    if nonmax_value not in sizes:
        idx = rng.choice([i for i, v in enumerate(sizes) if v != max_value])
        sizes[idx] = int(nonmax_value)


def _build_checkerboard(
    heights: list[int],
    widths: list[int],
    colors: tuple[int, int],
    start_parity: int,
) -> np.ndarray:
    h = sum(heights)
    w = sum(widths)
    grid = np.empty((h, w), dtype=int)

    r0 = 0
    for i, band_h in enumerate(heights):
        c0 = 0
        row_start = (start_parity + i) % 2
        for j, band_w in enumerate(widths):
            color = colors[(row_start + j) % 2]
            grid[r0 : r0 + band_h, c0 : c0 + band_w] = int(color)
            c0 += band_w
        r0 += band_h

    return grid


def _is_valid_example(grid: np.ndarray, tie_target: bool) -> bool:
    colors = np.unique(grid)
    if len(colors) != 2:
        return False

    comps = _components(grid)
    if len(comps) < MIN_COMPONENTS:
        return False

    sizes = [int(comp["size"]) for comp in comps]
    max_size = max(sizes)
    max_count = sum(size == max_size for size in sizes)
    nonmax_sizes = [size for size in sizes if size != max_size]

    if max_size > 10 or max_size < 4:
        return False
    if not nonmax_sizes:
        return False
    if not any(size % 2 == 1 for size in nonmax_sizes):
        return False
    if not any(size % 2 == 0 for size in nonmax_sizes):
        return False
    if (max_count > 1) != tie_target:
        return False
    if len(set(sizes)) < 3:
        return False

    by_color = Counter(int(comp["color"]) for comp in comps)
    if min(by_color.values()) < 2:
        return False

    out_colors = set(np.unique(main(grid)))
    if out_colors != {RED, GREEN, YELLOW}:
        return False

    return True


def _gen_attempt(rng: random.Random, tie_target: bool) -> np.ndarray | None:
    mode = _choose_mode(rng)
    max_h = int(mode["max_h"])
    max_w = int(rng.choice(mode["max_w_choices"]))
    nonmax_h_choices = tuple(int(v) for v in mode["nonmax_h_choices"])
    nonmax_w_choices = tuple(range(1, max_w))
    required_widths = [3 if max_w > 3 and rng.random() < 0.5 else 1]
    if max_h % 2 == 1:
        required_widths.append(2)

    n_rows = rng.randint(4, 7)
    n_cols = rng.randint(4, 7)

    tie_style = "unique"
    if tie_target:
        tie_style = rng.choice(("rows", "cols", "both"))

    row_max_count = 1
    col_max_count = 1
    if tie_style in ("rows", "both"):
        row_max_count = rng.randint(2, min(3, n_rows - 1))
    if tie_style in ("cols", "both"):
        max_tied_cols = min(3, n_cols - len(required_widths))
        if max_tied_cols < 2:
            return None
        col_max_count = rng.randint(2, max_tied_cols)

    heights = _sample_band_sizes(rng, n_rows, max_h, nonmax_h_choices, row_max_count)
    widths = _sample_band_sizes(rng, n_cols, max_w, nonmax_w_choices, col_max_count)

    _force_required_values(heights, [1], nonmax_h_choices[0], max_h, rng)
    _force_required_values(widths, required_widths, nonmax_w_choices[0], max_w, rng)

    colors = tuple(rng.sample(INPUT_PALETTE, 2))
    grid = _build_checkerboard(heights, widths, colors, rng.randrange(2))

    if rng.random() < 0.5:
        grid = grid.T
    if rng.random() < 0.75:
        grid = np.rot90(grid, k=rng.randrange(4)).copy()

    if not _is_valid_example(grid, tie_target):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    tie_target = rng.random() < 0.5

    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng, tie_target)
        if grid is not None:
            return grid

    raise RuntimeError("generate_input: exhausted retries")
