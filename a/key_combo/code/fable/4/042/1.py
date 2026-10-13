from __future__ import annotations

import random
from collections import Counter

import numpy as np


PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 500


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold asymmetric members of four-way mirror orbits into the upper-left quadrant to reveal the hidden pattern.

    Input structure:
    - The grid is an even-sized rectangle.
    - Ignoring a few modified cells, the whole picture is invariant under both horizontal and vertical reflection.
    - Each relevant orbit consists of the four cells at mirrored row/column positions.

    Rule:
    - For each orbit, compare its four colors.
    - If all four cells match, ignore that orbit.
    - If exactly one cell differs from the other three, fold that anomalous cell to the upper-left orbit coordinate and write its own color there.

    Key insight:
    - The anomaly color is not globally rare; the only reliable signal is the 3-to-1 vote inside each four-cell mirror orbit.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h % 2 == 0 and w % 2 == 0, "expected even grid dimensions"

    out = np.zeros((h // 2, w // 2), dtype=int)
    for r in range(h // 2):
        for c in range(w // 2):
            orbit = (
                (r, c),
                (r, w - 1 - c),
                (h - 1 - r, c),
                (h - 1 - r, w - 1 - c),
            )
            values = [int(g[rr, cc]) for rr, cc in orbit]
            counts = Counter(values)
            if len(counts) == 1:
                continue
            assert len(counts) == 2 and sorted(counts.values()) == [1, 3], "each broken orbit must be a 3:1 vote"
            out[r, c] = next(color for color, count in counts.items() if count == 1)
    return out


def _random_partition(total: int, parts: int, min_size: int) -> list[int]:
    sizes = [min_size] * parts
    for _ in range(total - parts * min_size):
        sizes[random.randrange(parts)] += 1
    random.shuffle(sizes)
    return sizes


def _build_base_quadrant(hh: int, ww: int) -> tuple[np.ndarray, list[int]]:
    min_row = 2 if hh >= 8 else 1
    min_col = 2 if ww >= 8 else 1
    row_parts = random.randint(2, min(5, hh // min_row))
    col_parts = random.randint(2, min(5, ww // min_col))
    row_sizes = _random_partition(hh, row_parts, min_row)
    col_sizes = _random_partition(ww, col_parts, min_col)

    cell_count = row_parts * col_parts
    palette_size = random.randint(3, min(5, cell_count, len(PALETTE)))
    palette = random.sample(PALETTE, palette_size)

    coarse_colors = palette.copy()
    while len(coarse_colors) < cell_count:
        coarse_colors.append(random.choice(palette))
    random.shuffle(coarse_colors)

    quadrant = np.zeros((hh, ww), dtype=int)
    idx = 0
    r0 = 0
    for rh in row_sizes:
        c0 = 0
        for cw in col_sizes:
            quadrant[r0 : r0 + rh, c0 : c0 + cw] = coarse_colors[idx]
            idx += 1
            c0 += cw
        r0 += rh
    return quadrant, palette


def _mirror4(quadrant: np.ndarray) -> np.ndarray:
    top = np.concatenate([quadrant, np.fliplr(quadrant)], axis=1)
    return np.concatenate([top, np.flipud(top)], axis=0)


def _component_masks(max_h: int, max_w: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []

    for h in range(1, min(3, max_h) + 1):
        for w in range(1, min(4, max_w) + 1):
            if 2 <= h * w <= 6:
                masks.append(np.ones((h, w), dtype=bool))

    for length in range(3, min(6, max_w) + 1):
        masks.append(np.ones((1, length), dtype=bool))
    for length in range(3, min(6, max_h) + 1):
        masks.append(np.ones((length, 1), dtype=bool))

    for arm in range(2, min(5, max_h, max_w) + 1):
        mask = np.zeros((arm, arm), dtype=bool)
        mask[0, :] = True
        mask[:, 0] = True
        masks.append(mask)

    if max_h >= 3 and max_w >= 3:
        plus = np.zeros((3, 3), dtype=bool)
        plus[1, :] = True
        plus[:, 1] = True
        masks.append(plus)

        tee = np.zeros((3, 3), dtype=bool)
        tee[0, :] = True
        tee[:, 1] = True
        masks.append(tee)

    for size in range(2, min(5, max_h, max_w) + 1):
        stair = np.zeros((size, size), dtype=bool)
        for i in range(size):
            stair[i, : i + 1] = True
        masks.append(stair)
        masks.append(np.fliplr(stair))

    return masks


def _can_place_component(occupied: set[tuple[int, int]], coords: list[tuple[int, int]]) -> bool:
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in occupied:
                    return False
    return True


def _build_target(base_quadrant: np.ndarray, usable_colors: list[int]) -> np.ndarray | None:
    hh, ww = map(int, base_quadrant.shape)
    masks = _component_masks(hh, ww)
    if not masks:
        return None

    min_area = 3
    max_area = max(4, min(18, (hh * ww) // 3))

    for _ in range(120):
        target = np.zeros((hh, ww), dtype=int)
        occupied: set[tuple[int, int]] = set()
        total_area = 0
        used_colors: set[int] = set()
        goal_components = random.randint(1, min(4, max(1, max_area // 2)))

        for _ in range(goal_components):
            placed = False
            for _ in range(80):
                mask = random.choice(masks)
                sh, sw = map(int, mask.shape)
                if sh > hh or sw > ww:
                    continue
                r0 = random.randint(0, hh - sh)
                c0 = random.randint(0, ww - sw)
                cells = np.argwhere(mask)
                coords = [(r0 + int(dr), c0 + int(dc)) for dr, dc in cells]
                if total_area + len(coords) > max_area:
                    continue
                if not _can_place_component(occupied, coords):
                    continue
                blocked = {int(base_quadrant[r, c]) for r, c in coords}
                color_choices = [color for color in usable_colors if color not in blocked]
                if not color_choices:
                    continue

                if used_colors and random.random() < 0.65:
                    favored = [color for color in color_choices if color not in used_colors]
                    if favored:
                        color_choices = favored
                color = random.choice(color_choices)

                for r, c in coords:
                    target[r, c] = color
                occupied.update(coords)
                total_area += len(coords)
                used_colors.add(color)
                placed = True
                break
            if not placed and total_area >= min_area:
                break

        fg = target[target != 0]
        if fg.size < min_area:
            continue
        if len(set(map(int, fg))) < 2 and fg.size >= 5 and random.random() < 0.7:
            continue
        return target

    return None


def _quadrant_assignments(count: int) -> list[int]:
    if count <= 0:
        return []
    if count == 1:
        return [random.randrange(4)]

    need = [0, 1, 2, 3] if count >= 4 else random.sample([0, 1, 2, 3], count)
    while len(need) < count:
        need.append(random.randrange(4))
    random.shuffle(need)
    return need


def _orbit_cell(hh: int, ww: int, r: int, c: int, quadrant_id: int) -> tuple[int, int]:
    if quadrant_id == 0:
        return r, c
    if quadrant_id == 1:
        return r, 2 * ww - 1 - c
    if quadrant_id == 2:
        return 2 * hh - 1 - r, c
    return 2 * hh - 1 - r, 2 * ww - 1 - c


def _gen_attempt() -> np.ndarray | None:
    hh = random.randint(4, 15)
    ww = random.randint(4, 15)

    base_quadrant, _palette = _build_base_quadrant(hh, ww)
    base_grid = _mirror4(base_quadrant)
    common_colors = [color for color in sorted(set(map(int, base_grid.flatten()))) if int(np.count_nonzero(base_grid == color)) >= 8]
    if len(common_colors) < 3:
        return None

    target = _build_target(base_quadrant, common_colors)
    if target is None:
        return None

    coords = [tuple(map(int, rc)) for rc in np.argwhere(target != 0)]
    assignments = _quadrant_assignments(len(coords))
    if len(set(assignments)) < min(2, len(coords)):
        return None

    inp = base_grid.copy()
    quadrant_counts = Counter(assignments)
    if max(quadrant_counts.values()) == len(coords):
        return None

    for (r, c), quadrant_id in zip(coords, assignments):
        color = int(target[r, c])
        if int(base_quadrant[r, c]) == color:
            return None
        rr, cc = _orbit_cell(hh, ww, r, c, quadrant_id)
        inp[rr, cc] = color

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, target):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
