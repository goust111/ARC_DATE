from __future__ import annotations

import random
from collections import Counter

import numpy as np


GREY = 5
ORANGE = 7
DIR_STEPS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GEN_ATTEMPTS = 300
MIN_PATH_LEN = 2


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _dominant_non_bg(grid: np.ndarray, bg: int) -> int:
    values, counts = np.unique(grid[grid != bg], return_counts=True)
    assert len(values) > 0
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if visited[r, c]:
            continue
        visited[r, c] = True
        stack = [(r, c)]
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _away_step(seed_r: int, seed_c: int, center_r: float, center_c: float) -> tuple[int, int]:
    dr = float(seed_r) - center_r
    dc = float(seed_c) - center_c
    if abs(dr) >= abs(dc):
        if dr > 0:
            return 1, 0
        if dr < 0:
            return -1, 0
    if dc > 0:
        return 0, 1
    if dc < 0:
        return 0, -1
    raise AssertionError("seed lies exactly on the inferred center")


def _strict_dir_name(cell: tuple[int, int], center_r: float, center_c: float) -> str | None:
    dr = float(cell[0]) - center_r
    dc = float(cell[1]) - center_c
    if abs(dr) == abs(dc):
        return None
    if abs(dr) > abs(dc):
        if dr == 0:
            return None
        return "down" if dr > 0 else "up"
    if dc == 0:
        return None
    return "right" if dc > 0 else "left"


def _ray_cells(mask: np.ndarray, start: tuple[int, int], step: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    r, c = start
    dr, dc = step
    r += dr
    c += dc
    ray: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w and mask[r, c]:
        ray.append((r, c))
        r += dr
        c += dc
    return ray


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow alternating seed-color/orange bands outward inside each grey-walled region.

    Input structure:
    - The dominant color is the background.
    - One or more closed one-pixel-thick grey outlines enclose background interiors.
    - Each enclosed interior contains exactly two single-pixel seeds of different colors.

    Rule:
    - For each enclosed interior, compute a coarse center from the median row and median column
      of the interior background cells.
    - Each seed chooses the axis-aligned direction that points away from that center most strongly.
    - Starting from the cell after the seed, extend a straight band whose colors alternate between
      the seed color and orange, with the first grown cell using the seed color.
    - A band stops when its next cell is outside the grid, on a grey wall, on a seed, or on a
      previously grown band cell; the original seed pixels stay unchanged.

    Key insight:
    - The hidden control signal is the seed's position relative to the region's median-based center,
      not its distance to the nearest wall; the visible output is then just a straight alternating ray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _get_bg(g)
    wall = GREY if np.any(g == GREY) else _dominant_non_bg(g, bg)

    out = g.copy()
    for comp in _components(g != wall):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue

        bg_cells = np.array([(r, c) for r, c in comp if int(g[r, c]) == bg], dtype=int)
        if bg_cells.size == 0:
            continue

        seeds = [(int(r), int(c), int(g[r, c])) for r, c in comp if int(g[r, c]) != bg]
        assert len(seeds) == 2, "each enclosed region must contain exactly two seeds"

        center_r = float(np.median(bg_cells[:, 0]))
        center_c = float(np.median(bg_cells[:, 1]))
        states = []
        for r, c, color in seeds:
            dr, dc = _away_step(r, c, center_r, center_c)
            states.append([r, c, dr, dc, 0, color, True])

        while True:
            proposals: list[tuple[int, int, int]] = []
            for idx, state in enumerate(states):
                if not state[6]:
                    continue
                nr = state[0] + state[2]
                nc = state[1] + state[3]
                if not (0 <= nr < h and 0 <= nc < w) or int(out[nr, nc]) != bg:
                    state[6] = False
                    continue
                proposals.append((idx, nr, nc))

            if not proposals:
                break

            counts = Counter((nr, nc) for _idx, nr, nc in proposals)
            advanced = False
            for idx, nr, nc in proposals:
                state = states[idx]
                if counts[(nr, nc)] > 1:
                    state[6] = False
                    continue
                state[4] += 1
                out[nr, nc] = state[5] if state[4] % 2 == 1 else ORANGE
                state[0], state[1] = nr, nc
                advanced = True

            if not advanced:
                break

    return out


def _mask_connected(mask: np.ndarray) -> bool:
    cells = int(mask.sum())
    return cells > 0 and len(_components(mask)) == 1


def _shape_with_seed_pairs(rng: random.Random) -> tuple[np.ndarray, list[dict[str, object]]] | None:
    for _ in range(100):
        h = rng.randint(4, 7)
        w = rng.randint(4, 8)
        max_inset = max(1, min(2, w - 3))
        left = rng.randint(0, max_inset)
        right = rng.randint(0, max_inset)
        while w - left - right < 3:
            if left >= right and left > 0:
                left -= 1
            elif right > 0:
                right -= 1

        mask = np.zeros((h, w), dtype=bool)
        for r in range(h):
            if r > 0:
                left = max(0, min(max_inset, left + rng.choice((-1, 0, 1))))
                right = max(0, min(max_inset, right + rng.choice((-1, 0, 1))))
                while w - left - right < 3:
                    if left >= right and left > 0:
                        left -= 1
                    elif right > 0:
                        right -= 1
                    else:
                        break
            mask[r, left : w - right] = True

        if rng.random() < 0.75:
            notch_side = rng.choice(("left", "right"))
            span = rng.randint(1, max(1, h // 2))
            start = rng.randint(0, h - span)
            trial = mask.copy()
            for r in range(start, start + span):
                cols = np.flatnonzero(trial[r])
                if cols.size >= 4:
                    if notch_side == "left":
                        trial[r, int(cols[0])] = False
                    else:
                        trial[r, int(cols[-1])] = False
            if _mask_connected(trial):
                mask = trial

        if rng.random() < 0.5:
            mask = mask.T
        if rng.random() < 0.5:
            mask = mask[::-1, :]
        if rng.random() < 0.5:
            mask = mask[:, ::-1]

        if int(mask.sum()) < 10 or not _mask_connected(mask):
            continue

        cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]
        valid_pairs: list[dict[str, object]] = []
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                s1 = cells[i]
                s2 = cells[j]
                bg_cells = [cell for k, cell in enumerate(cells) if k not in (i, j)]
                rows = np.array([r for r, _c in bg_cells], dtype=float)
                cols = np.array([c for _r, c in bg_cells], dtype=float)
                center_r = float(np.median(rows))
                center_c = float(np.median(cols))
                d1 = _strict_dir_name(s1, center_r, center_c)
                d2 = _strict_dir_name(s2, center_r, center_c)
                if d1 is None or d2 is None:
                    continue
                ray1 = _ray_cells(mask, s1, DIR_STEPS[d1])
                ray2 = _ray_cells(mask, s2, DIR_STEPS[d2])
                if len(ray1) < MIN_PATH_LEN or len(ray2) < MIN_PATH_LEN:
                    continue
                set1 = set(ray1)
                set2 = set(ray2)
                if s2 in set1 or s1 in set2 or set1 & set2:
                    continue
                valid_pairs.append(
                    {
                        "seed1": s1,
                        "seed2": s2,
                        "dir1": d1,
                        "dir2": d2,
                        "len1": len(ray1),
                        "len2": len(ray2),
                    }
                )

        if valid_pairs:
            return mask, valid_pairs
    return None


def _choose_pair(
    valid_pairs: list[dict[str, object]],
    dir_counts: dict[str, int],
    rng: random.Random,
) -> dict[str, object]:
    rng.shuffle(valid_pairs)
    best: dict[str, object] | None = None
    best_key: tuple[int, int, int] | None = None
    for pair in valid_pairs:
        d1 = str(pair["dir1"])
        d2 = str(pair["dir2"])
        min_len = min(int(pair["len1"]), int(pair["len2"]))
        key = (
            dir_counts[d1] + dir_counts[d2],
            0 if d1 != d2 else 1,
            -min_len,
        )
        if best_key is None or key < best_key:
            best = pair
            best_key = key
    assert best is not None
    return best


def _render_patch(
    mask: np.ndarray,
    pair: dict[str, object],
    seed_colors: tuple[int, int],
    bg: int,
) -> np.ndarray:
    h, w = mask.shape
    patch = np.full((h + 2, w + 2), bg, dtype=int)
    interior = {(int(r) + 1, int(c) + 1) for r, c in np.argwhere(mask)}
    boundary: set[tuple[int, int]] = set()
    for r, c in interior:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) not in interior:
                boundary.add((nr, nc))
    for r, c in boundary:
        patch[r, c] = GREY

    s1r, s1c = pair["seed1"]
    s2r, s2c = pair["seed2"]
    patch[int(s1r) + 1, int(s1c) + 1] = seed_colors[0]
    patch[int(s2r) + 1, int(s2c) + 1] = seed_colors[1]
    return patch


def _place_patch(grid: np.ndarray, patch: np.ndarray, reserved: np.ndarray, rng: random.Random, bg: int) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if ph + 2 > h or pw + 2 > w:
        return False
    for _ in range(80):
        r0 = rng.randint(1, h - ph - 1)
        c0 = rng.randint(1, w - pw - 1)
        rr0 = r0 - 1
        cc0 = c0 - 1
        rr1 = r0 + ph + 1
        cc1 = c0 + pw + 1
        if reserved[rr0:rr1, cc0:cc1].any():
            continue
        mask = patch != bg
        grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
        reserved[rr0:rr1, cc0:cc1] = True
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    region_count = rng.randint(1, 3)
    if region_count == 1:
        h = rng.randint(14, 22)
        w = rng.randint(14, 22)
    elif region_count == 2:
        h = rng.randint(18, 26)
        w = rng.randint(18, 26)
    else:
        h = rng.randint(22, 28)
        w = rng.randint(22, 28)

    bg = rng.choice([color for color in range(10) if color not in {GREY, ORANGE}])
    color_pool = [color for color in range(10) if color not in {bg, GREY, ORANGE}]
    rng.shuffle(color_pool)
    grid = np.full((h, w), bg, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    dir_counts = {name: 0 for name in DIR_STEPS}

    for region_idx in range(region_count):
        built = _shape_with_seed_pairs(rng)
        if built is None:
            return None
        mask, valid_pairs = built
        pair = _choose_pair(valid_pairs, dir_counts, rng)
        colors = (color_pool[2 * region_idx], color_pool[2 * region_idx + 1])
        patch = _render_patch(mask, pair, colors, bg)
        if not _place_patch(grid, patch, reserved, rng, bg):
            return None
        dir_counts[str(pair["dir1"])] += 1
        dir_counts[str(pair["dir2"])] += 1

    if _get_bg(grid) != bg:
        return None

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid) or not np.any(out == ORANGE):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
