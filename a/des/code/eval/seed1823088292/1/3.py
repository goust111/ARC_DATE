import random
from typing import Dict, List, Sequence, Tuple

import numpy as np


BLACK = 0
PRISM = 9
PAIR_COLORS = [1, 2, 3, 4, 5, 6, 7, 8]

DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}

ARROW_SHAPES = {
    "right": {(0, 2), (1, 1), (1, 2), (2, 2)},
    "down": {(1, 1), (2, 0), (2, 1), (2, 2)},
    "left": {(0, 0), (1, 0), (1, 1), (2, 0)},
    "up": {(0, 0), (0, 1), (0, 2), (1, 1)},
}
PARSED_ARROW_SHAPES = {
    "right": {(0, 1), (1, 0), (1, 1), (2, 1)},
    "down": {(0, 1), (1, 0), (1, 1), (1, 2)},
    "left": {(0, 0), (1, 0), (1, 1), (2, 0)},
    "up": {(0, 0), (0, 1), (0, 2), (1, 1)},
}
SHAPE_TO_DIR = {frozenset(shape): direction for direction, shape in PARSED_ARROW_SHAPES.items()}

GEN_ATTEMPTS = 300
SCENARIOS = ("none", "single", "multi")


def _components_of_color(grid: np.ndarray, color: int) -> List[List[Tuple[int, int]]]:
    mask = grid == color
    visited = np.zeros_like(mask, dtype=bool)
    h, w = grid.shape
    components: List[List[Tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: List[Tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: Sequence[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _rel_coords(coords: Sequence[Tuple[int, int]]) -> frozenset[Tuple[int, int]]:
    r0, c0, _, _ = _bbox(coords)
    return frozenset((r - r0, c - c0) for r, c in coords)


def _find_prisms(grid: np.ndarray) -> Tuple[List[dict], Dict[Tuple[int, int], int]]:
    h, w = grid.shape
    prisms: List[dict] = []
    prism_by_cell: Dict[Tuple[int, int], int] = {}
    for coords in _components_of_color(grid, PRISM):
        assert len(coords) == 3, "each prism must contain exactly three maroon cells"
        r0, c0, r1, c1 = _bbox(coords)
        if r0 == r1 and c1 - c0 == 2:
            marker_pos = (r0, c1 + 1)
        elif c0 == c1 and r1 - r0 == 2:
            marker_pos = (r1 + 1, c0)
        else:
            raise AssertionError("prisms must be 1x3 or 3x1 bars")
        mr, mc = marker_pos
        assert 0 <= mr < h and 0 <= mc < w, "prism marker must stay inside the grid"
        marker_color = int(grid[mr, mc])
        assert marker_color not in (BLACK, PRISM), "prism marker must be a non-black non-maroon pixel"
        idx = len(prisms)
        prisms.append(
            {
                "bbox": (r0, c0, r1, c1),
                "marker_color": marker_color,
            }
        )
        for cell in coords:
            prism_by_cell[cell] = idx
    return prisms, prism_by_cell


def _find_pairs(grid: np.ndarray) -> List[dict]:
    arrows: Dict[int, List[dict]] = {}
    blocks: Dict[int, List[Tuple[int, int]]] = {}
    colors = sorted(set(map(int, grid.flat)) - {BLACK, PRISM})
    for color in colors:
        for coords in _components_of_color(grid, color):
            area = len(coords)
            r0, c0, r1, c1 = _bbox(coords)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            rel = _rel_coords(coords)
            if area == 4 and h == 2 and w == 2:
                blocks.setdefault(color, []).append((r0, c0))
            elif area == 4 and rel in SHAPE_TO_DIR:
                arrows.setdefault(color, []).append(
                    {
                        "top": r0,
                        "left": c0,
                        "direction": SHAPE_TO_DIR[rel],
                        "rows": {r for r, _ in coords},
                        "cols": {c for _, c in coords},
                    }
                )
            elif area == 1:
                continue
            else:
                raise AssertionError(f"unrecognized component for color {color}: area={area}, bbox={h}x{w}")

    assert set(arrows) == set(blocks), "arrow colors and block colors must match exactly"
    pairs: List[dict] = []
    for color in sorted(arrows):
        assert len(arrows[color]) == 1, "each color must have exactly one arrow"
        assert len(blocks[color]) == 1, "each color must have exactly one 2x2 block"
        arrow = arrows[color][0]
        br, bc = blocks[color][0]
        assert not (arrow["rows"] & {br, br + 1}), "arrow and block of same color must not share rows"
        assert not (arrow["cols"] & {bc, bc + 1}), "arrow and block of same color must not share cols"
        pairs.append(
            {
                "color": color,
                "direction": arrow["direction"],
                "arrow": (arrow["top"], arrow["left"]),
                "block": (br, bc),
            }
        )
    return pairs


def _cells_in_bounds(shape: Tuple[int, int], cells: Sequence[Tuple[int, int]]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _block_cells(top: int, left: int) -> List[Tuple[int, int]]:
    return [(top + dr, left + dc) for dr in (0, 1) for dc in (0, 1)]


def _paint_block(out: np.ndarray, cells: Sequence[Tuple[int, int]], color: int) -> None:
    for r, c in cells:
        assert int(out[r, c]) in (BLACK, color), "trail should not overwrite other objects"
    for r, c in cells:
        out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each 2x2 block in its arrow direction, leaving a trail and recoloring
    the trail whenever it passes through a maroon prism.
    """
    g = np.asarray(grid, dtype=int)
    prisms, prism_by_cell = _find_prisms(g)
    pairs = _find_pairs(g)

    out = g.copy()
    for pair in pairs:
        br, bc = pair["block"]
        out[br : br + 2, bc : bc + 2] = BLACK

    for pair in pairs:
        color = int(pair["color"])
        direction = pair["direction"]
        dr, dc = DIRS[direction]
        r, c = pair["block"]
        trail_color = color

        while True:
            nr, nc = r + dr, c + dc
            next_cells = _block_cells(nr, nc)
            if not _cells_in_bounds(g.shape, next_cells):
                break

            hit_prisms = {prism_by_cell[cell] for cell in next_cells if cell in prism_by_cell}
            if not hit_prisms:
                _paint_block(out, next_cells, trail_color)
                r, c = nr, nc
                continue

            assert len(hit_prisms) == 1, "a block front should hit at most one prism at a time"
            prism = prisms[next(iter(hit_prisms))]
            r0, c0, r1, c1 = prism["bbox"]
            if direction == "right":
                nr, nc = r, c1 + 1
            elif direction == "left":
                nr, nc = r, c0 - 2
            elif direction == "down":
                nr, nc = r1 + 1, c
            else:
                nr, nc = r0 - 2, c

            jump_cells = _block_cells(nr, nc)
            if not _cells_in_bounds(g.shape, jump_cells):
                break
            assert not any(cell in prism_by_cell for cell in jump_cells), "jump target must be beyond the prism"

            trail_color = int(prism["marker_color"])
            _paint_block(out, jump_cells, trail_color)
            r, c = nr, nc

    return out


def _pick_spaced(low: int, high: int, count: int, gap: int) -> List[int] | None:
    if count == 0:
        return []
    if high < low:
        return None
    values = list(range(low, high + 1))
    if len(values) < count:
        return None
    for _ in range(80):
        picks = sorted(random.sample(values, count))
        if all(b - a >= gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _sample_prism_counts(count: int) -> List[int]:
    scenario = random.choice(SCENARIOS)
    if scenario == "none":
        return [0] * count
    if scenario == "single":
        prism_counts = [0] * count
        active = random.sample(range(count), random.randint(1, count))
        for idx in active:
            prism_counts[idx] = 1
        return prism_counts
    prism_counts = [0] * count
    prism_counts[random.randrange(count)] = 2
    for idx in range(count):
        if prism_counts[idx] == 0 and random.random() < 0.5:
            prism_counts[idx] = 1
    return prism_counts


def _marker_colors(start_color: int, prism_count: int, forbidden_colors: Sequence[int]) -> List[int]:
    colors: List[int] = []
    current = start_color
    forbidden = set(forbidden_colors)
    for _ in range(prism_count):
        options = [color for color in PAIR_COLORS if color != current and color not in forbidden]
        assert options, "generator needs at least one marker color distinct from pair colors"
        new_color = random.choice(options)
        colors.append(new_color)
        current = new_color
    return colors


def _stamp_arrow(grid: np.ndarray, top: int, left: int, color: int, direction: str) -> None:
    for dr, dc in ARROW_SHAPES[direction]:
        assert int(grid[top + dr, left + dc]) == BLACK
        grid[top + dr, left + dc] = color


def _plan_horizontal_object(width: int, direction: str, prism_count: int) -> Tuple[int, List[int], int] | None:
    for _ in range(100):
        if direction == "right":
            start = random.randint(0, max(0, width // 3))
            prism_cols = _pick_spaced(start + 3, width - 4, prism_count, 4)
            arrow_candidates = [
                col
                for col in range(max(width // 2, 0), width - 2)
                if col > start + 1 or col + 2 < start
            ]
        else:
            start = random.randint(max(width // 2, 2), width - 2)
            prism_cols = _pick_spaced(3, start - 2, prism_count, 4)
            arrow_candidates = [
                col
                for col in range(0, max(0, width // 2 - 1))
                if col + 2 < start
            ]
        if prism_cols is None or not arrow_candidates:
            continue
        return start, prism_cols, random.choice(arrow_candidates)
    return None


def _plan_vertical_object(height: int, direction: str, prism_count: int) -> Tuple[int, List[int], int] | None:
    for _ in range(100):
        if direction == "down":
            start = random.randint(0, max(0, height // 3))
            prism_rows = _pick_spaced(start + 3, height - 4, prism_count, 4)
            arrow_candidates = [
                row
                for row in range(max(height // 2, 0), height - 2)
                if row > start + 1 or row + 2 < start
            ]
        else:
            start = random.randint(max(height // 2, 2), height - 2)
            prism_rows = _pick_spaced(3, start - 2, prism_count, 4)
            arrow_candidates = [
                row
                for row in range(0, max(0, height // 2 - 1))
                if row + 2 < start
            ]
        if prism_rows is None or not arrow_candidates:
            continue
        return start, prism_rows, random.choice(arrow_candidates)
    return None


def _gen_attempt() -> np.ndarray | None:
    object_count = random.randint(1, 4)
    axis = random.choice(["horizontal", "vertical"])
    prism_counts = _sample_prism_counts(object_count)
    pair_colors = random.sample(PAIR_COLORS, object_count)

    if axis == "horizontal":
        pad_top = random.randint(0, 1 if object_count == 4 else 2)
        pad_bottom = random.randint(0, 1 if object_count == 4 else 2)
        height = pad_top + 7 * object_count + pad_bottom
        width = random.randint(18, 30)
        if height > 30:
            return None
        grid = np.zeros((height, width), dtype=int)
        for idx, color in enumerate(pair_colors):
            base = pad_top + 7 * idx
            lane_r = base + 1
            arrow_top = base + 4
            direction = random.choice(["left", "right"])
            plan = _plan_horizontal_object(width, direction, prism_counts[idx])
            if plan is None:
                return None
            start_c, prism_cols, arrow_left = plan
            marker_colors = _marker_colors(color, prism_counts[idx], pair_colors)

            grid[lane_r : lane_r + 2, start_c : start_c + 2] = color
            _stamp_arrow(grid, arrow_top, arrow_left, color, direction)
            for prism_col, marker_color in zip(prism_cols, marker_colors):
                assert int(grid[lane_r - 1 : lane_r + 2, prism_col].sum()) == 0
                grid[lane_r - 1 : lane_r + 2, prism_col] = PRISM
                assert int(grid[lane_r + 2, prism_col]) == BLACK
                grid[lane_r + 2, prism_col] = marker_color
    else:
        pad_left = random.randint(0, 1 if object_count == 4 else 2)
        pad_right = random.randint(0, 1 if object_count == 4 else 2)
        width = pad_left + 7 * object_count + pad_right
        height = random.randint(18, 30)
        if width > 30:
            return None
        grid = np.zeros((height, width), dtype=int)
        for idx, color in enumerate(pair_colors):
            base = pad_left + 7 * idx
            lane_c = base + 1
            arrow_left = base + 4
            direction = random.choice(["up", "down"])
            plan = _plan_vertical_object(height, direction, prism_counts[idx])
            if plan is None:
                return None
            start_r, prism_rows, arrow_top = plan
            marker_colors = _marker_colors(color, prism_counts[idx], pair_colors)

            grid[start_r : start_r + 2, lane_c : lane_c + 2] = color
            _stamp_arrow(grid, arrow_top, arrow_left, color, direction)
            for prism_row, marker_color in zip(prism_rows, marker_colors):
                assert int(grid[prism_row, lane_c - 1 : lane_c + 2].sum()) == 0
                grid[prism_row, lane_c - 1 : lane_c + 2] = PRISM
                assert int(grid[prism_row, lane_c + 2]) == BLACK
                grid[prism_row, lane_c + 2] = marker_color

    try:
        _ = main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
