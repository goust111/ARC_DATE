from __future__ import annotations

import random

import numpy as np


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400


def _legend_colors(row: np.ndarray) -> list[int]:
    row = np.asarray(row, dtype=int).reshape(-1)
    assert row.size in (4, 5), "legend row must be length 4 or 5"
    colors = [int(x) for x in row[:4]]
    assert len(set(colors)) == 4, "legend colors must be distinct"
    return colors


def _quadrant_patches(content: np.ndarray) -> list[np.ndarray]:
    return [
        np.rot90(content, 2),   # UL
        np.rot90(content, -1),  # UR
        np.rot90(content, 1),   # LL
        content.copy(),         # LR
    ]


def _repair_patch(patch: np.ndarray, repair_color: int) -> np.ndarray:
    fixed = patch.copy()
    fixed[fixed == 0] = int(repair_color)
    return fixed


def _compose_output(content: np.ndarray, legend: list[int]) -> np.ndarray:
    k = int(content.shape[0])
    out = np.zeros((2 * k, 2 * k), dtype=int)
    ul, ur, ll, lr = [
        _repair_patch(patch, color)
        for patch, color in zip(_quadrant_patches(content), legend, strict=True)
    ]
    out[:k, :k] = ul
    out[:k, k:] = ur
    out[k:, :k] = ll
    out[k:, k:] = lr
    return out


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, pts[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _local_majority_color(grid: np.ndarray, r: int, c: int) -> int | None:
    colors: dict[int, int] = {}
    for rr in range(max(0, r - 1), min(grid.shape[0], r + 2)):
        for cc in range(max(0, c - 1), min(grid.shape[1], c + 2)):
            if rr == r and cc == c:
                continue
            color = int(grid[rr, cc])
            if color == 0:
                continue
            colors[color] = colors.get(color, 0) + 1
    if len(colors) != 2:
        return None
    ordered = sorted(colors.items(), key=lambda item: (-item[1], item[0]))
    if ordered[0][1] == ordered[1][1]:
        return None
    if ordered[0][1] < 4:
        return None
    return int(ordered[0][0])


def _candidate_defects(full: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    k = int(full.shape[0])
    by_color: dict[int, list[tuple[int, int]]] = {}
    for r in range(1, k - 1):
        for c in range(1, k - 1):
            color = int(full[r, c])
            probe = full.copy()
            probe[r, c] = 0
            if _local_majority_color(probe, r, c) != color:
                continue
            mask = probe == color
            if not _is_connected(mask):
                continue
            by_color.setdefault(color, []).append((r, c))
    return by_color


def _far_enough(chosen: list[tuple[int, int]], cell: tuple[int, int]) -> bool:
    return all(max(abs(cell[0] - r), abs(cell[1] - c)) > 1 for r, c in chosen)


def _pick_defects(full: np.ndarray) -> list[tuple[int, int]] | None:
    by_color = _candidate_defects(full)
    all_cells = [cell for cells in by_color.values() for cell in cells]
    if not all_cells:
        return None

    max_defects = min(3, len(all_cells))
    counts = [1]
    if max_defects >= 2:
        counts.append(2)
    if max_defects >= 3:
        counts.append(3)

    colors = list(by_color)
    for _ in range(80):
        want = random.choice(counts)
        chosen: list[tuple[int, int]] = []

        if want >= 2 and len(colors) == 2 and all(by_color[color] for color in colors) and random.random() < 0.6:
            seeded = colors[:]
            random.shuffle(seeded)
            for color in seeded:
                picks = by_color[color][:]
                random.shuffle(picks)
                for cell in picks:
                    if _far_enough(chosen, cell):
                        chosen.append(cell)
                        break
            if len(chosen) > want:
                chosen = chosen[:want]

        pool = all_cells[:]
        random.shuffle(pool)
        for cell in pool:
            if len(chosen) >= want:
                break
            if cell not in chosen and _far_enough(chosen, cell):
                chosen.append(cell)
        if len(chosen) != want:
            continue

        defective = full.copy()
        for r, c in chosen:
            defective[r, c] = 0
        visible = [int(x) for x in np.unique(defective) if int(x) != 0]
        if len(visible) != 2:
            continue
        if not all(_is_connected(defective == color) for color in visible):
            continue
        if any(_local_majority_color(defective, r, c) != int(full[r, c]) for r, c in chosen):
            continue
        return chosen
    return None


def _make_content(k: int) -> np.ndarray | None:
    palette = list(range(1, 10))
    a, b = random.sample(palette, 2)
    for _ in range(120):
        cuts = [random.randint(1, k - 1) for _ in range(k)]
        if len(set(cuts)) == 1 and random.random() < 0.8:
            continue

        full = np.empty((k, k), dtype=int)
        for r, cut in enumerate(cuts):
            full[r, :cut] = a
            full[r, cut:] = b

        if random.random() < 0.5:
            full = full.T
        if random.random() < 0.5:
            full = np.flipud(full)
        if random.random() < 0.5:
            full = np.fliplr(full)

        if len({tuple(row.tolist()) for row in full}) == 1 and len({tuple(col.tolist()) for col in full.T}) == 1:
            continue
        if not _is_connected(full == a) or not _is_connected(full == b):
            continue
        return full
    return None


def _no_rotation_output(content: np.ndarray, legend: list[int]) -> np.ndarray:
    k = int(content.shape[0])
    out = np.zeros((2 * k, 2 * k), dtype=int)
    patches = [_repair_patch(content, color) for color in legend]
    out[:k, :k] = patches[0]
    out[:k, k:] = patches[1]
    out[k:, :k] = patches[2]
    out[k:, k:] = patches[3]
    return out


def _validate_instance(full: np.ndarray, defective: np.ndarray, legend_row: np.ndarray) -> None:
    k = int(full.shape[0])
    assert full.shape == (k, k)
    assert defective.shape == (k, k)
    colors = [int(x) for x in np.unique(full)]
    assert 0 not in colors and len(colors) == 2
    assert all(_is_connected(full == color) for color in colors)

    defects = [tuple(map(int, coord)) for coord in np.argwhere(defective == 0)]
    assert 1 <= len(defects) <= 3
    assert all(0 < r < k - 1 and 0 < c < k - 1 for r, c in defects)
    assert len({tuple(row.tolist()) for row in defective}) > 1 or len({tuple(col.tolist()) for col in defective.T}) > 1
    assert all(_local_majority_color(defective, r, c) == int(full[r, c]) for r, c in defects)

    visible = [int(x) for x in np.unique(defective) if int(x) != 0]
    assert len(visible) == 2
    assert all(_is_connected(defective == color) for color in visible)

    rotations = _quadrant_patches(defective)
    for i in range(4):
        for j in range(i + 1, 4):
            assert not np.array_equal(rotations[i], rotations[j])

    legend = _legend_colors(legend_row)
    assert set(legend).isdisjoint(colors)
    assert 0 not in legend

    grid = np.vstack([defective, legend_row[np.newaxis, :]])
    correct = _compose_output(defective, legend)
    assert np.array_equal(main(grid), correct)
    assert correct.shape == (2 * k, 2 * k)
    for color in legend:
        assert int(np.sum(correct == color)) == len(defects)
    assert not np.array_equal(correct, _no_rotation_output(defective, legend))
    swapped = legend[:]
    swapped[0], swapped[1] = swapped[1], swapped[0]
    assert not np.array_equal(correct, _compose_output(defective, swapped))
    assert not np.array_equal(correct, _compose_output(full, legend))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile a defective square into four rotated quadrants and recolor each defect by the quadrant legend.

    Input structure:
    - The input has k+1 rows and k columns, with k in {4, 5}; the top k×k block is the content square.
    - The content square contains exactly two non-black foreground colors and 1-3 black defect cells.
    - The last row is the color legend: its leftmost four cells encode the repair colors for UL, UR, LL, LR. If k=5, the fifth trailing cell is padding and is ignored.

    Rule:
    - Remove the last row to recover the k×k content square.
    - Build a 2k×2k output using four rotated copies: UL = 180 degrees, UR = 90 degrees clockwise, LL = 90 degrees counterclockwise, LR = unchanged.
    - In each quadrant, replace every black defect cell with that quadrant's legend color; leave all non-black cells unchanged.

    Key insight:
    - The same defect is copied into four differently rotated quadrants, but its repair color is determined only by quadrant position through the legend row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] == g.shape[1] + 1, "input must be a square content block plus one legend row"

    content = g[:-1, :]
    assert content.shape[0] == content.shape[1]
    legend = _legend_colors(g[-1, :])
    return _compose_output(content, legend)


def _gen_attempt() -> np.ndarray | None:
    k = random.choice((4, 5))
    full = _make_content(k)
    if full is None:
        return None

    defects = _pick_defects(full)
    if defects is None:
        return None

    defective = full.copy()
    for r, c in defects:
        defective[r, c] = 0

    available = [color for color in range(1, 10) if color not in set(int(x) for x in np.unique(full))]
    legend = random.sample(available, 4)
    legend_row = np.array(legend + ([legend[-1]] if k == 5 else []), dtype=int)

    try:
        _validate_instance(full, defective, legend_row)
    except AssertionError:
        return None

    return np.vstack([defective, legend_row[np.newaxis, :]])


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotated-quadrant repair input")
