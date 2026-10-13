import random
from collections import Counter

import numpy as np


BG = 0
DIVIDER = 5
LEGEND_KEYS = [1, 2, 3, 4]  # UL, UR, LR, LL
REPAIR_COLORS = [6, 7, 8, 9]
PATTERN_COLORS = [1, 2, 3, 4]
QUADRANT_ORDER = ["UL", "UR", "LR", "LL"]


def _find_divider(grid: np.ndarray) -> tuple[int, int]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == DIVIDER)]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == DIVIDER)]
    assert len(rows) == 1 and len(cols) == 1
    return int(rows[0]), int(cols[0])


def _extract_quadrants(grid: np.ndarray, mid_r: int, mid_c: int) -> dict[str, np.ndarray]:
    return {
        "UL": grid[:mid_r, :mid_c].copy(),
        "UR": grid[:mid_r, mid_c + 1 :].copy(),
        "LR": grid[mid_r + 1 :, mid_c + 1 :].copy(),
        "LL": grid[mid_r + 1 :, :mid_c].copy(),
    }


def _aligned_quadrants(quads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "UL": quads["UL"].copy(),
        "UR": np.rot90(quads["UR"], 1),
        "LR": np.rot90(quads["LR"], 2),
        "LL": np.rot90(quads["LL"], -1),
    }


def _aligned_valid_masks(size: int) -> dict[str, np.ndarray]:
    masks = {name: np.ones((size, size), dtype=bool) for name in QUADRANT_ORDER}
    lr_actual_mask = np.ones((size, size), dtype=bool)
    lr_actual_mask[size - 4 : size, size - 2 : size] = False
    masks["LR"] = np.rot90(lr_actual_mask, 2)
    return masks


def _majority_value(values: list[int]) -> int:
    counts = Counter(values)
    winners = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    assert len(winners) >= 1
    if len(winners) > 1:
        assert winners[0][1] > winners[1][1]
    return int(winners[0][0])


def _aligned_to_actual_local(name: str, r: int, c: int, size: int) -> tuple[int, int]:
    if name == "UL":
        return int(r), int(c)
    if name == "UR":
        return int(c), int(size - 1 - r)
    if name == "LR":
        return int(size - 1 - r), int(size - 1 - c)
    assert name == "LL"
    return int(size - 1 - c), int(r)


def _quadrant_origin(name: str, size: int) -> tuple[int, int]:
    if name == "UL":
        return 0, 0
    if name == "UR":
        return 0, size + 1
    if name == "LR":
        return size + 1, size + 1
    return size + 1, 0


def _read_repair_colors(grid: np.ndarray) -> dict[str, int]:
    legend = grid[-4:, -2:]
    mapping_by_key = {int(legend[i, 0]): int(legend[i, 1]) for i in range(4)}
    assert set(mapping_by_key) == set(LEGEND_KEYS)
    return {
        "UL": mapping_by_key[1],
        "UR": mapping_by_key[2],
        "LR": mapping_by_key[3],
        "LL": mapping_by_key[4],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect rotational-symmetry breaks across four quadrants and recolor the broken pixels using a palette legend.

    Input structure:
    - A square black grid is split into four equal quadrants by one grey divider row and one grey divider column.
    - The quadrants contain the same compact motif under 90-degree rotation, except that 2 or 3 quadrants each have one tampered pixel.
    - A 4x2 legend occupies the bottom-right corner: left column keys 1/2/3/4 identify UL/UR/LR/LL, and the right column gives that quadrant's repair color.

    Rule:
    - Read the quadrant-to-repair-color mapping from the legend.
    - Rotate UR, LR, and LL back into UL orientation and compare all corresponding logical positions across quadrants.
    - For each logical position, use the majority color among the aligned quadrants as the correct value.
    - Any quadrant cell that disagrees with the majority is a broken pixel and is replaced by that quadrant's repair color, while divider and legend cells stay unchanged.

    Key insight:
    - A broken pixel cannot be identified from one quadrant alone; it becomes visible only after all four rotated quadrants are aligned and majority-voted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1]

    mid_r, mid_c = _find_divider(g)
    assert mid_r == mid_c
    size = int(mid_r)

    quads = _extract_quadrants(g, mid_r, mid_c)
    aligned = _aligned_quadrants(quads)
    valid_masks = _aligned_valid_masks(size)
    repairs = _read_repair_colors(g)
    out = g.copy()

    for r in range(size):
        for c in range(size):
            values = [
                int(aligned[name][r, c])
                for name in QUADRANT_ORDER
                if valid_masks[name][r, c]
            ]
            correct = _majority_value(values)
            for name in QUADRANT_ORDER:
                if not valid_masks[name][r, c]:
                    continue
                if int(aligned[name][r, c]) == correct:
                    continue
                rr_local, cc_local = _aligned_to_actual_local(name, r, c, size)
                qr0, qc0 = _quadrant_origin(name, size)
                out[qr0 + rr_local, qc0 + cc_local] = repairs[name]

    return out


def _grow_connected_cells(height: int, width: int, count: int) -> list[tuple[int, int]]:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    frontier = set()
    while len(cells) < count:
        frontier.clear()
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return []
        cells.add(random.choice(list(frontier)))
    return list(cells)


def _build_ul_pattern(size: int) -> np.ndarray | None:
    allowed_width = size - 2
    if allowed_width < 2:
        return None

    height = random.randint(3, size)
    width = random.randint(max(2, (8 + height - 1) // height), allowed_width)
    max_area = height * width
    if max_area < 8:
        return None

    count = random.randint(8, min(15, max_area))
    cells = _grow_connected_cells(height, width, count)
    if not cells:
        return None

    top = random.randint(0, size - height)
    left = random.randint(2, size - width)
    ul = np.zeros((size, size), dtype=int)

    n_colors = 1 if random.random() < 0.45 else 2
    palette = random.sample(PATTERN_COLORS, n_colors)
    if n_colors == 1:
        color_map = {cell: palette[0] for cell in cells}
    else:
        second_count = random.randint(2, count - 2)
        second_cells = set(_grow_connected_cells(height, width, second_count))
        second_cells &= set(cells)
        if len(second_cells) < 2:
            second_cells = set(random.sample(cells, second_count))
        color_map = {}
        for cell in cells:
            color_map[cell] = palette[1] if cell in second_cells else palette[0]

    for r, c in cells:
        ul[top + r, left + c] = color_map[(r, c)]

    return ul


def _compose_grid(ul: np.ndarray, repair_order: list[int]) -> np.ndarray:
    size = ul.shape[0]
    grid = np.zeros((2 * size + 1, 2 * size + 1), dtype=int)
    grid[size, :] = DIVIDER
    grid[:, size] = DIVIDER

    ur = np.rot90(ul, -1)
    lr = np.rot90(ul, 2)
    ll = np.rot90(ul, 1)

    grid[:size, :size] = ul
    grid[:size, size + 1 :] = ur
    grid[size + 1 :, size + 1 :] = lr
    grid[size + 1 :, :size] = ll

    for i, key in enumerate(LEGEND_KEYS):
        grid[-4 + i, -2] = key
        grid[-4 + i, -1] = repair_order[i]

    return grid


def _tamper_grid(grid: np.ndarray, ul: np.ndarray) -> np.ndarray:
    size = ul.shape[0]
    out = grid.copy()
    nonzero_cells = [tuple(map(int, rc)) for rc in np.argwhere(ul != 0)]
    assert len(nonzero_cells) >= 8

    tampered_quadrants = random.sample(QUADRANT_ORDER, random.choice([2, 3]))
    chosen_cells = random.sample(nonzero_cells, len(tampered_quadrants))

    for name, (r, c) in zip(tampered_quadrants, chosen_cells, strict=False):
        rr_local, cc_local = _aligned_to_actual_local(name, r, c, size)
        qr0, qc0 = _quadrant_origin(name, size)
        rr = qr0 + rr_local
        cc = qc0 + cc_local
        correct = int(out[rr, cc])
        candidates = [col for col in PATTERN_COLORS if col != correct]
        out[rr, cc] = random.choice(candidates)

    return out


def _generate_attempt() -> np.ndarray | None:
    size = random.randint(5, 7)
    ul = _build_ul_pattern(size)
    if ul is None or np.count_nonzero(ul) < 8:
        return None

    repair_order = random.sample(REPAIR_COLORS, 4)
    grid = _compose_grid(ul, repair_order)
    grid = _tamper_grid(grid, ul)

    out = main(grid)
    if out.shape != grid.shape:
        return None

    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[-4:, -2:] = True
    diff = (out != grid) & (~legend_mask)
    if int(np.count_nonzero(diff)) not in (2, 3):
        return None
    if np.any(out[legend_mask] != grid[legend_mask]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
