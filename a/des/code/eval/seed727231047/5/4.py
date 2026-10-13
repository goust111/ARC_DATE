from __future__ import annotations

from collections import Counter
import numpy as np


BG = 0
ARROW = 5
GAP = 3
BLOCK_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
BLOCK_SIZE_OPTIONS = [3, 4, 5]
COMPONENT_OPTIONS = [
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (2, 2),
    (1, 4),
    (4, 1),
    (2, 3),
    (3, 2),
]
MAX_BLOCKS = len(BLOCK_COLORS)
MAX_SIDE = 30
GEN_ATTEMPTS = 400
DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]

ARROW_PATTERNS = {
    "R": {(1, 0), (1, 1), (1, 2), (0, 2), (2, 2)},
    "L": {(1, 0), (1, 1), (1, 2), (0, 0), (2, 0)},
    "D": {(0, 1), (1, 1), (2, 1), (2, 0), (2, 2)},
    "U": {(0, 1), (1, 1), (2, 1), (0, 0), (0, 2)},
}


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(counts.argmax())])


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "h": int(r1 - r0 + 1),
                    "w": int(c1 - c0 + 1),
                }
            )
    return comps


def _arrow_direction(comp: dict) -> str | None:
    if int(comp["area"]) != 5 or int(comp["h"]) != 3 or int(comp["w"]) != 3:
        return None
    rel = {
        (int(r - comp["r0"]), int(c - comp["c0"]))
        for r, c in comp["coords"].tolist()
    }
    for direction, pattern in ARROW_PATTERNS.items():
        if rel == pattern:
            return direction
    return None


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    components = _monochrome_components(g, bg)
    square_candidates: list[dict] = []
    arrows: list[dict] = []

    for comp in components:
        color = int(comp["color"])
        r0, c0, r1, c1 = int(comp["r0"]), int(comp["c0"]), int(comp["r1"]), int(comp["c1"])
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if int(comp["h"]) == int(comp["w"]) and int(comp["area"]) == int(comp["h"]) * int(comp["w"]) and bool((patch == color).all()):
            square_candidates.append(comp)
            continue
        direction = _arrow_direction(comp)
        if direction is not None:
            arrows.append({"dir": direction, **comp})

    assert square_candidates, "expected at least one solid square block"

    size_counts = Counter(int(comp["h"]) for comp in square_candidates)
    block_size = int(size_counts.most_common(1)[0][0])
    blocks = [comp for comp in square_candidates if int(comp["h"]) == block_size]
    assert blocks, "failed to isolate block squares"

    id_by_component = {id(comp): idx for idx, comp in enumerate(blocks)}
    preds: list[list[int]] = [[] for _ in blocks]

    for arrow in arrows:
        r0, c0, r1, c1 = int(arrow["r0"]), int(arrow["c0"]), int(arrow["r1"]), int(arrow["c1"])
        direction = str(arrow["dir"])

        if direction in ("L", "R"):
            left = [
                id_by_component[id(block)]
                for block in blocks
                if int(block["c1"]) == c0 - 1 and int(block["r0"]) <= r0 and int(block["r1"]) >= r1
            ]
            right = [
                id_by_component[id(block)]
                for block in blocks
                if int(block["c0"]) == c1 + 1 and int(block["r0"]) <= r0 and int(block["r1"]) >= r1
            ]
            assert len(left) == 1 and len(right) == 1, "horizontal arrow must sit between exactly two blocks"
            src, dst = (left[0], right[0]) if direction == "R" else (right[0], left[0])
        else:
            top = [
                id_by_component[id(block)]
                for block in blocks
                if int(block["r1"]) == r0 - 1 and int(block["c0"]) <= c0 and int(block["c1"]) >= c1
            ]
            bottom = [
                id_by_component[id(block)]
                for block in blocks
                if int(block["r0"]) == r1 + 1 and int(block["c0"]) <= c0 and int(block["c1"]) >= c1
            ]
            assert len(top) == 1 and len(bottom) == 1, "vertical arrow must sit between exactly two blocks"
            src, dst = (top[0], bottom[0]) if direction == "D" else (bottom[0], top[0])

        preds[dst].append(src)

    memo: dict[int, int] = {}

    def root_of(node: int) -> int:
        if node in memo:
            return memo[node]
        if not preds[node]:
            memo[node] = node
            return node
        roots = {root_of(parent) for parent in preds[node]}
        assert len(roots) == 1, "each block must trace back to a unique source color"
        memo[node] = next(iter(roots))
        return memo[node]

    out = g.copy()
    for idx, block in enumerate(blocks):
        root = blocks[root_of(idx)]
        out[int(block["r0"]) : int(block["r1"]) + 1, int(block["c0"]) : int(block["c1"]) + 1] = int(root["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each solid square block by following the grey-arrow infection chain back to its source block.
    """
    return _solve(grid)


def _paint_arrow(grid: np.ndarray, r0: int, c0: int, direction: str) -> None:
    for dr, dc in ARROW_PATTERNS[direction]:
        grid[int(r0 + dr), int(c0 + dc)] = ARROW


def _distribute_extra(rng: np.random.Generator, total: int, slots: int) -> list[int]:
    pieces = [0] * slots
    for _ in range(total):
        pieces[int(rng.integers(0, slots))] += 1
    return pieces


def _sample_component_rectangles(rng: np.random.Generator, n_components: int) -> list[tuple[int, int]] | None:
    for _ in range(200):
        dims: list[tuple[int, int]] = []
        total_blocks = 0
        for idx in range(n_components):
            remaining = n_components - idx - 1
            feasible = [
                dim
                for dim in COMPONENT_OPTIONS
                if total_blocks + int(dim[0]) * int(dim[1]) + 2 * remaining <= MAX_BLOCKS
            ]
            if not feasible:
                break
            choice = feasible[int(rng.integers(0, len(feasible)))]
            dims.append((int(choice[0]), int(choice[1])))
            total_blocks += int(choice[0]) * int(choice[1])
        if len(dims) != n_components:
            continue
        if total_blocks < 3:
            continue
        if max(int(r) * int(c) for r, c in dims) < 3 and n_components == 1:
            continue
        rng.shuffle(dims)
        return dims
    return None


def _footprint_size(rows: int, cols: int, block_size: int) -> tuple[int, int]:
    return (
        int(rows * block_size + (rows - 1) * GAP),
        int(cols * block_size + (cols - 1) * GAP),
    )


def _layout_diagonal(
    rng: np.random.Generator, component_dims: list[tuple[int, int]], block_size: int
) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    footprints = [_footprint_size(rows, cols, block_size) for rows, cols in component_dims]
    count = len(footprints)

    base_h = 2 + sum(h for h, _ in footprints) + 2 * (count - 1)
    base_w = 2 + sum(w for _, w in footprints) + 2 * (count - 1)
    if base_h > MAX_SIDE or base_w > MAX_SIDE:
        return None

    extra_h = int(rng.integers(0, min(6, MAX_SIDE - base_h) + 1))
    extra_w = int(rng.integers(0, min(6, MAX_SIDE - base_w) + 1))
    h_slots = _distribute_extra(rng, extra_h, count + 1)
    w_slots = _distribute_extra(rng, extra_w, count + 1)

    origins: list[tuple[int, int]] = []
    cur_r = 1 + h_slots[0]
    cur_c = 1 + w_slots[0]
    for idx, (h, w) in enumerate(footprints):
        origins.append((int(cur_r), int(cur_c)))
        cur_r += h
        cur_c += w
        if idx + 1 < count:
            cur_r += 2 + h_slots[idx + 1]
            cur_c += 2 + w_slots[idx + 1]

    height = int(cur_r + 1 + h_slots[-1])
    width = int(cur_c + 1 + w_slots[-1])
    if height > MAX_SIDE or width > MAX_SIDE:
        return None
    return (height, width), origins


def _build_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    n_components = int(rng.integers(1, 4))
    component_dims = _sample_component_rectangles(rng, n_components)
    if component_dims is None:
        return None

    feasible_sizes = [
        block_size
        for block_size in BLOCK_SIZE_OPTIONS
        if _layout_diagonal(rng, component_dims, block_size) is not None
    ]
    if not feasible_sizes:
        return None
    block_size = int(feasible_sizes[int(rng.integers(0, len(feasible_sizes)))])
    layout = _layout_diagonal(rng, component_dims, block_size)
    if layout is None:
        return None

    (height, width), origins = layout
    grid = np.full((height, width), BG, dtype=int)

    total_blocks = sum(int(rows) * int(cols) for rows, cols in component_dims)
    colors = rng.choice(np.array(BLOCK_COLORS, dtype=int), size=total_blocks, replace=False).tolist()
    color_iter = iter(int(color) for color in colors)

    block_records: list[dict] = []
    component_roots: list[int] = []
    component_cells: list[dict[tuple[int, int], int]] = []

    pitch = block_size + GAP
    for comp_idx, ((rows, cols), (base_r, base_c)) in enumerate(zip(component_dims, origins, strict=False)):
        cell_to_block: dict[tuple[int, int], int] = {}
        root_cell = (int(rng.integers(0, rows)), int(rng.integers(0, cols)))
        for rr in range(rows):
            for cc in range(cols):
                color = next(color_iter)
                r0 = int(base_r + rr * pitch)
                c0 = int(base_c + cc * pitch)
                grid[r0 : r0 + block_size, c0 : c0 + block_size] = color
                block_index = len(block_records)
                block_records.append(
                    {
                        "comp": comp_idx,
                        "cell": (rr, cc),
                        "color": color,
                        "r0": r0,
                        "c0": c0,
                        "r1": r0 + block_size - 1,
                        "c1": c0 + block_size - 1,
                    }
                )
                cell_to_block[(rr, cc)] = block_index
        component_cells.append(cell_to_block)
        component_roots.append(cell_to_block[root_cell])

    for comp_idx, (rows, cols) in enumerate(component_dims):
        root_index = component_roots[comp_idx]
        root_cell = block_records[root_index]["cell"]
        root_r = int(root_cell[0])
        root_c = int(root_cell[1])
        cell_to_block = component_cells[comp_idx]

        for rr in range(rows):
            for cc in range(cols - 1):
                left_index = cell_to_block[(rr, cc)]
                right_index = cell_to_block[(rr, cc + 1)]
                d_left = abs(rr - root_r) + abs(cc - root_c)
                d_right = abs(rr - root_r) + abs(cc + 1 - root_c)
                assert d_left != d_right
                direction = "R" if d_left < d_right else "L"
                left = block_records[left_index]
                offset = int(rng.integers(0, block_size - 2))
                _paint_arrow(grid, int(left["r0"]) + offset, int(left["c1"]) + 1, direction)

        for rr in range(rows - 1):
            for cc in range(cols):
                top_index = cell_to_block[(rr, cc)]
                bottom_index = cell_to_block[(rr + 1, cc)]
                d_top = abs(rr - root_r) + abs(cc - root_c)
                d_bottom = abs(rr + 1 - root_r) + abs(cc - root_c)
                assert d_top != d_bottom
                direction = "D" if d_top < d_bottom else "U"
                top = block_records[top_index]
                offset = int(rng.integers(0, block_size - 2))
                _paint_arrow(grid, int(top["r1"]) + 1, int(top["c0"]) + offset, direction)

    expected = grid.copy()
    for comp_idx, root_index in enumerate(component_roots):
        root_color = int(block_records[root_index]["color"])
        for block in block_records:
            if int(block["comp"]) != comp_idx:
                continue
            expected[int(block["r0"]) : int(block["r1"]) + 1, int(block["c0"]) : int(block["c1"]) + 1] = root_color

    solved = _solve(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    """
    Generate a grid of equal-sized colored square blocks separated by grey arrows.
    """
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        built = _build_case(rng)
        if built is None:
            continue
        grid, _expected = built
        return grid
    raise RuntimeError("failed to generate a valid infection-arrow ARC input")
