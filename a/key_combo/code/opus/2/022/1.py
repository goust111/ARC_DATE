from __future__ import annotations

import importlib.util
import random
from collections import deque
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")


def _load_arc_utils():
    spec = importlib.util.spec_from_file_location("_arc_utils_022_1", _ARC_UTILS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load arc_utils from {_ARC_UTILS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    _arc_utils = _load_arc_utils()
    Color = _arc_utils.Color
    find_components = _arc_utils.find_components
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        colors=None,
        min_area: int = 1,
        return_format: str = "dict",
    ):
        if return_format != "dict":
            raise ValueError("fallback find_components only supports return_format='dict'")
        if bg is None:
            values, counts = np.unique(grid, return_counts=True)
            bg = int(values[np.argmax(counts)])
        if not monochromatic:
            raise ValueError("fallback find_components only supports monochromatic=True")
        target_colors = sorted(set(int(c) for c in colors)) if colors is not None else [
            int(c) for c in np.unique(grid) if int(c) != int(bg)
        ]
        seen = np.zeros(grid.shape, dtype=bool)
        components = []
        for color in target_colors:
            for r, c in np.argwhere((grid == color) & ~seen):
                coords = _collect_component(grid == color, (int(r), int(c)), connectivity)
                for rr, cc in coords:
                    seen[rr, cc] = True
                if len(coords) < int(min_area):
                    continue
                arr = np.array(coords, dtype=int)
                r0, c0 = arr.min(axis=0)
                r1, c1 = arr.max(axis=0)
                components.append(
                    {
                        "color": color,
                        "coords": arr,
                        "bbox": (int(r0), int(c0), int(r1), int(c1)),
                        "area": int(arr.shape[0]),
                    }
                )
        return components


BG = Color.BLACK
WALL = Color.GREEN
BAR = Color.GREY

OUTER_MAP = {2: Color.BLUE, 3: Color.RED, 4: Color.YELLOW, 5: Color.ORANGE}
INNER_MAP = {2: Color.TEAL, 3: Color.PINK, 4: Color.MAROON, 5: Color.BLUE}

GEN_RETRIES = 400


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _collect_component(mask: np.ndarray, start: tuple[int, int], connectivity: int = 4) -> list[tuple[int, int]]:
    h, w = mask.shape
    sr, sc = start
    if not (0 <= sr < h and 0 <= sc < w) or not bool(mask[sr, sc]):
        return []
    seen = {start}
    q = deque([start])
    coords: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        coords.append((r, c))
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
        if connectivity == 8:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        q.append((nr, nc))
    return coords


def _component_masks(mask: np.ndarray) -> list[np.ndarray]:
    remaining = mask.copy()
    parts: list[np.ndarray] = []
    for r, c in np.argwhere(remaining):
        if not bool(remaining[int(r), int(c)]):
            continue
        coords = _collect_component(remaining, (int(r), int(c)), connectivity=4)
        part = np.zeros_like(mask, dtype=bool)
        for rr, cc in coords:
            part[rr, cc] = True
            remaining[rr, cc] = False
        parts.append(part)
    return parts


def _bbox_of_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _draw_rect_outline(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    bottom = top + height - 1
    right = left + width - 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color


def _exterior_background_mask(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    mask = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if grid[r, c] == BG and not mask[r, c]:
                mask[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grid[r, c] == BG and not mask[r, c]:
                mask[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == BG and not mask[nr, nc]:
                mask[nr, nc] = True
                q.append((nr, nc))
    return mask


def _bar_components(grid: np.ndarray) -> list[dict]:
    return find_components(
        grid,
        bg=BG,
        colors=[BAR],
        connectivity=4,
        monochromatic=True,
        min_area=1,
        return_format="dict",
    )


def _bar_length_and_cells(comp: dict) -> tuple[int, list[tuple[int, int]]]:
    coords = [(int(r), int(c)) for r, c in comp["coords"]]
    rows = {r for r, _ in coords}
    cols = sorted(c for _, c in coords)
    assert len(rows) == 1, "grey component must be horizontal"
    assert cols == list(range(cols[0], cols[0] + len(cols))), "grey component must be contiguous"
    return len(coords), coords


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor horizontal gray bars by length, using one table outside the fence and another inside it.

    Input structure:
    - A black background contains one 4-connected green orthogonal line object acting as a wall.
    - The wall encloses one or more holes, while other background remains border-connected exterior.
    - Several disconnected gray horizontal bar segments of length 2-5 lie entirely in either exterior or enclosed background.

    Rule:
    - Preserve the green wall and black background.
    - Flood-fill black cells from the grid border using 4-connectivity; reached cells are exterior and unreached black cells are interior.
    - For each gray bar, measure its length and select the recoloring table by whether the whole bar lies in exterior or interior.
    - Exterior bars map 2->1, 3->2, 4->4, 5->7; interior bars map 2->8, 3->6, 4->9, 5->1.

    Key insight:
    - The same visible bar length can map to different output colors, so topology from the flood-fill must gate which length-to-color table is used.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    region_grid = g.copy()
    region_grid[region_grid == BAR] = BG
    exterior = _exterior_background_mask(region_grid)
    out = g.copy()
    for comp in _bar_components(g):
        length, cells = _bar_length_and_cells(comp)
        assert 2 <= length <= 5
        in_exterior = [bool(exterior[r, c]) for r, c in cells]
        assert len(set(in_exterior)) == 1, "a bar must lie in exactly one region type"
        table = OUTER_MAP if in_exterior[0] else INNER_MAP
        for r, c in cells:
            out[r, c] = table[length]
    return out


def _can_place_bar(grid: np.ndarray, allowed: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if grid[r, c] != BG or not bool(allowed[r, c]):
            return False
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cell_set and grid[nr, nc] != BG:
                return False
    return True


def _candidate_bars(grid: np.ndarray, allowed: np.ndarray, length: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    candidates: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w - length + 1):
            cells = [(r, c + dc) for dc in range(length)]
            if _can_place_bar(grid, allowed, cells):
                candidates.append(cells)
    random.shuffle(candidates)
    return candidates


def _place_lengths(grid: np.ndarray, allowed: np.ndarray, lengths: list[int]) -> bool:
    lengths = sorted(lengths, reverse=True)

    def backtrack(i: int) -> bool:
        if i == len(lengths):
            return True
        for cells in _candidate_bars(grid, allowed, lengths[i]):
            for r, c in cells:
                grid[r, c] = BAR
            if backtrack(i + 1):
                return True
            for r, c in cells:
                grid[r, c] = BG
        return False

    return backtrack(0)


def _place_single_loop_inside(grid: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    top, left, height, width = rect
    safe_rows = list(range(top + 2, top + height - 2))
    safe_w = width - 4
    assert len(safe_rows) >= 4 and safe_w >= 7
    chosen_rows = sorted(random.sample(safe_rows, 4))
    lengths = [5, 2, 3, 4]
    if random.random() < 0.5:
        lengths = list(reversed(lengths))
    left_starts = {5: 0, 3: 0, 2: 0, 4: 0}
    right_starts = {5: safe_w - 5, 3: safe_w - 3, 2: safe_w - 2, 4: safe_w - 4}
    side_pattern = [0, 1, 0, 1] if random.random() < 0.5 else [1, 0, 1, 0]
    for row, length, side in zip(chosen_rows, lengths, side_pattern):
        start = left_starts[length] if side == 0 else right_starts[length]
        col = left + 2 + start
        for dc in range(length):
            grid[row, col + dc] = BAR


def _safe_hole_mask(rect: tuple[int, int, int, int], shape: tuple[int, int]) -> np.ndarray:
    top, left, height, width = rect
    mask = np.zeros(shape, dtype=bool)
    mask[top + 2 : top + height - 2, left + 2 : left + width - 2] = True
    return mask


def _build_single_loop() -> np.ndarray:
    h = random.randint(14, 18)
    w = random.randint(16, 21)
    grid = np.zeros((h, w), dtype=int)
    rect_h = random.randint(8, 9)
    rect_w = random.randint(11, 12)
    top = random.randint(2, h - rect_h - 2)
    left = random.randint(2, w - rect_w - 2)
    rect = (top, left, rect_h, rect_w)
    _draw_rect_outline(grid, top, left, rect_h, rect_w, WALL)
    _place_single_loop_inside(grid, rect)

    exterior = _exterior_background_mask(grid)
    if not _place_lengths(grid, exterior.copy(), [2, 3, 4, 5]):
        raise ValueError("failed to place single-loop exterior bars")
    return grid


def _build_double_loop() -> np.ndarray:
    h = random.randint(15, 19)
    gap = random.randint(5, 7)
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    top_margin = random.randint(2, 4)
    rect_h = 5
    rect_w1 = random.choice([7, 7, 8])
    rect_w2 = random.choice([7, 7, 8])
    w = left_margin + rect_w1 + gap + rect_w2 + right_margin
    grid = np.zeros((h, w), dtype=int)

    top = top_margin
    left1 = left_margin
    left2 = left1 + rect_w1 + gap
    rect1 = (top, left1, rect_h, rect_w1)
    rect2 = (top, left2, rect_h, rect_w2)
    _draw_rect_outline(grid, *rect1, WALL)
    _draw_rect_outline(grid, *rect2, WALL)

    connector_on_top = random.random() < 0.5
    conn_row = top if connector_on_top else top + rect_h - 1
    grid[conn_row, left1 + rect_w1 - 1 : left2 + 1] = WALL

    hole1 = _safe_hole_mask(rect1, grid.shape)
    hole2 = _safe_hole_mask(rect2, grid.shape)
    inside_lengths = [2, 3] if rect_w1 >= 7 and rect_w2 >= 7 else [2, 2]
    if not _place_lengths(grid, hole1, [inside_lengths[0]]):
        raise ValueError("failed to place left-hole bar")
    if not _place_lengths(grid, hole2, [inside_lengths[1]]):
        raise ValueError("failed to place right-hole bar")

    corridor = np.zeros_like(grid, dtype=bool)
    safe_row = top + 2 if connector_on_top else top + rect_h - 3
    right1 = left1 + rect_w1 - 1
    corridor_start = right1 + 2
    corridor_stop = left2 - 1
    corridor[safe_row : safe_row + 1, corridor_start:corridor_stop] = True
    forced_corridor_length = 2 if (corridor_stop - corridor_start) >= 2 else None
    if forced_corridor_length is None or not _place_lengths(grid, corridor, [forced_corridor_length]):
        raise ValueError("failed to place corridor counterexample bar")
    remaining = [3, 4, 5]

    exterior = _exterior_background_mask(grid)
    if not _place_lengths(grid, exterior.copy(), remaining):
        raise ValueError("failed to place double-loop exterior bars")
    return grid


def _validate_grid(grid: np.ndarray, layout: str) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2 and max(grid.shape) <= 30

    wall_components = find_components(
        grid,
        bg=BG,
        colors=[WALL],
        connectivity=4,
        monochromatic=True,
        min_area=1,
        return_format="dict",
    )
    assert len(wall_components) == 1, "green fence must be one 4-connected object"

    region_grid = grid.copy()
    region_grid[region_grid == BAR] = BG
    exterior = _exterior_background_mask(region_grid)
    interior = (region_grid == BG) & ~exterior
    holes = _component_masks(interior)
    assert len(holes) >= 1, "must have at least one enclosed region"

    bars = _bar_components(grid)
    assert 5 <= len(bars) <= 8, "bar count must stay in the specified range"

    inside_lengths: list[int] = []
    outside_lengths: list[int] = []
    hole_bar_counts = [0] * len(holes)
    for comp in bars:
        length, cells = _bar_length_and_cells(comp)
        assert 2 <= length <= 5
        is_outside = [bool(exterior[r, c]) for r, c in cells]
        assert len(set(is_outside)) == 1
        if is_outside[0]:
            outside_lengths.append(length)
        else:
            inside_lengths.append(length)
            owner = None
            for idx, hole in enumerate(holes):
                if all(bool(hole[r, c]) for r, c in cells):
                    owner = idx
                    break
            assert owner is not None, "inside bar must belong to a unique hole"
            hole_bar_counts[owner] += 1
        cell_set = set(cells)
        for r, c in cells:
            assert grid[r, c] == BAR
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and (nr, nc) not in cell_set:
                    assert grid[nr, nc] in (BG,), "bars may not 4-touch other objects"

    assert inside_lengths and outside_lengths
    assert set(inside_lengths) & set(outside_lengths), "same length must appear inside and outside"
    assert all(count >= 1 for count in hole_bar_counts), "each hole needs at least one bar"

    if layout == "single":
        assert sorted(inside_lengths) == [2, 3, 4, 5]
        assert sorted(outside_lengths) == [2, 3, 4, 5]
    else:
        assert len(holes) == 2
        assert set(outside_lengths) >= {2, 3, 4, 5}
        bbox = _bbox_of_mask(grid == WALL)
        r0, c0, r1, c1 = bbox
        assert any(
            all(r0 <= r <= r1 and c0 <= c <= c1 and bool(exterior[r, c]) for r, c in _bar_length_and_cells(comp)[1])
            for comp in bars
        ), "double-loop sample needs an exterior bar inside the overall fence bbox"

    out = main(grid)
    for comp in bars:
        length, cells = _bar_length_and_cells(comp)
        out_colors = {int(out[r, c]) for r, c in cells}
        assert len(out_colors) == 1
        region_is_outside = bool(exterior[cells[0][0], cells[0][1]])
        expected = OUTER_MAP[length] if region_is_outside else INNER_MAP[length]
        assert out_colors == {expected}

    shared_lengths = set(inside_lengths) & set(outside_lengths)
    for length in shared_lengths:
        assert OUTER_MAP[length] != INNER_MAP[length], "shared lengths must visibly distinguish the two tables"


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            if random.random() < 0.65:
                grid = _build_single_loop()
                _validate_grid(grid, "single")
            else:
                grid = _build_double_loop()
                _validate_grid(grid, "double")
            return grid
        except Exception:
            continue
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
