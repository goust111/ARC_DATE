from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "arc_utils_for_opus_8_024_1",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()
Color = _ARC_UTILS.Color
find_components = _ARC_UTILS.find_components


BACKGROUND = Color.BLACK
GREEN = Color.GREEN
MARKER_COLORS = (Color.BLUE, Color.RED, Color.YELLOW)
OBJECT_COLORS = (Color.BLUE, Color.RED, Color.YELLOW, Color.ORANGE)
MAX_GENERATION_ATTEMPTS = 300

SMALL_SHAPES: dict[int, tuple[tuple[tuple[int, int], ...], ...]] = {
    1: (((0, 0),),),
    2: (((0, 0), (0, 1)), ((0, 0), (1, 0))),
    3: (
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1)),
    ),
    4: (
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (1, 0), (2, 0), (3, 0)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 1), (1, 1), (2, 1), (2, 0)),
        ((0, 0), (0, 1), (0, 2), (1, 0)),
        ((0, 0), (0, 1), (0, 2), (1, 2)),
    ),
}


def _apply_row_fill(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for r in range(out.shape[0]):
        left = int(out[r, 0])
        right = int(out[r, -1])
        if left != BACKGROUND and left == right:
            row_mask = out[r] == BACKGROUND
            out[r, row_mask] = left
    return out


def _recolor_small_components(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for comp in find_components(out, bg=BACKGROUND, connectivity=4, monochromatic=True):
        if int(comp["area"]) <= 2:
            coords = comp["coords"]
            out[coords[:, 0], coords[:, 1]] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill same-endpoint rows, then recolor only the tiny same-color components that remain tiny after the fill.

    Input structure:
    - The background is black.
    - Some rows have left and right boundary marker pixels in colors blue, red, or yellow.
    - Interior foreground objects are small 4-connected monochrome shapes in blue, red, yellow, or orange.

    Rule:
    - Copy the grid.
    - For each row whose two boundary markers are present and have the same color, recolor every black cell in that row to that marker color while leaving existing foreground pixels unchanged.
    - On the filled grid, find all non-black 4-connected same-color components.
    - Recolor every component of area 1 or 2 to green; keep every component of area at least 3 unchanged.

    Key insight:
    - The row fill changes connectivity before size filtering, so a tiny object can survive only because the filled row bridges it into a larger same-color component.
    """
    filled = _apply_row_fill(np.asarray(grid, dtype=int))
    return _recolor_small_components(filled)


def _match_rows(grid: np.ndarray) -> list[int]:
    rows: list[int] = []
    for r in range(grid.shape[0]):
        left = int(grid[r, 0])
        right = int(grid[r, -1])
        if left != BACKGROUND and left == right:
            rows.append(r)
    return rows


def _translate(shape: tuple[tuple[int, int], ...], r0: int, c0: int) -> tuple[tuple[int, int], ...]:
    return tuple((int(r0) + int(r), int(c0) + int(c)) for r, c in shape)


def _neighbors8(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr != 0 or dc != 0
    )


def _can_place(
    grid: np.ndarray,
    cells: tuple[tuple[int, int], ...],
    *,
    gap8: bool = True,
    allow_existing: set[tuple[int, int]] | None = None,
) -> bool:
    H, W = grid.shape
    allow_existing = set() if allow_existing is None else set(allow_existing)
    cell_set = set(cells)
    for r, c in cell_set:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if c in (0, W - 1):
            return False
        if r in (0, H - 1):
            return False
        if grid[r, c] != BACKGROUND and (r, c) not in allow_existing:
            return False
    if not gap8:
        return True
    for r, c in cell_set:
        for nr, nc in _neighbors8(r, c):
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if (nr, nc) in cell_set:
                continue
            if grid[nr, nc] != BACKGROUND and (nr, nc) not in allow_existing:
                return False
    return True


def _paint(grid: np.ndarray, cells: tuple[tuple[int, int], ...], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _row_choices(H: int, exclude: set[int], count: int, *, min_gap: int = 0) -> list[int] | None:
    candidates = [r for r in range(1, H - 1) if r not in exclude]
    random.shuffle(candidates)
    chosen: list[int] = []
    for r in candidates:
        if all(abs(r - prev) > min_gap for prev in chosen):
            chosen.append(r)
            if len(chosen) == count:
                return chosen
    return None


def _bridge_cells(mode: str, area: int, bridge_row: int, col: int) -> tuple[tuple[int, int], ...] | None:
    if mode == "same":
        if area == 1:
            return ((bridge_row, col),)
        return ((bridge_row, col), (bridge_row, col + 1))
    if mode == "above":
        if area == 1:
            return ((bridge_row - 1, col),)
        if random.random() < 0.5:
            return ((bridge_row - 1, col), (bridge_row - 1, col + 1))
        return ((bridge_row - 2, col), (bridge_row - 1, col))
    if mode == "below":
        if area == 1:
            return ((bridge_row + 1, col),)
        if random.random() < 0.5:
            return ((bridge_row + 1, col), (bridge_row + 1, col + 1))
        return ((bridge_row + 1, col), (bridge_row + 2, col))
    return None


def _place_bridge_pair(grid: np.ndarray, bridge_row: int, color: int) -> bool:
    H, W = grid.shape
    combos = [
        ("above", "below"),
        ("same", "below"),
        ("above", "same"),
        ("same", "same"),
    ]
    for _ in range(60):
        mode_a, mode_b = random.choice(combos)
        area_a = random.choice((1, 2))
        area_b = random.choice((1, 2))
        if W < 13:
            return False
        c1 = random.randint(2, W - 8)
        c2 = random.randint(c1 + 3, W - 3)
        cells_a = _bridge_cells(mode_a, area_a, bridge_row, c1)
        cells_b = _bridge_cells(mode_b, area_b, bridge_row, c2)
        if cells_a is None or cells_b is None:
            continue
        if not _can_place(grid, cells_a, gap8=True):
            continue
        if not _can_place(grid, cells_b, gap8=True):
            continue
        if set(cells_a) & set(cells_b):
            continue
        _paint(grid, cells_a, color)
        _paint(grid, cells_b, color)
        return True
    return False


def _random_interior_shape(area: int) -> tuple[tuple[int, int], ...]:
    return random.choice(SMALL_SHAPES[int(area)])


def _place_random_object(
    grid: np.ndarray,
    color: int,
    area: int,
    *,
    allowed_rows: set[int],
    gap8: bool = True,
) -> bool:
    shape = _random_interior_shape(area)
    h = 1 + max(r for r, _ in shape)
    w = 1 + max(c for _, c in shape)
    H, W = grid.shape
    rows = list(allowed_rows)
    random.shuffle(rows)
    for _ in range(80):
        if not rows:
            break
        r0 = random.choice(rows)
        c0 = random.randint(1, W - w - 1)
        cells = _translate(shape, r0, c0)
        if {r for r, _ in cells} <= allowed_rows and _can_place(grid, cells, gap8=gap8):
            _paint(grid, cells, color)
            return True
    return False


def _place_endpoint_support(grid: np.ndarray, row: int, support_color: int, side: str) -> bool:
    W = grid.shape[1]
    if side == "left":
        cells = ((row, 1), (row, 2))
    else:
        cells = ((row, W - 3), (row, W - 2))
    if not _can_place(grid, cells, gap8=False):
        return False
    _paint(grid, cells, support_color)
    return True


def _build_candidate() -> np.ndarray | None:
    H = random.randint(11, 16)
    W = random.randint(13, 18)
    grid = np.zeros((H, W), dtype=int)

    bridge_row = random.randint(2, H - 3)
    bridge_color = random.choice(MARKER_COLORS)

    need_support = random.random() < 0.65
    extra_match_count = random.randint(0, 2)
    mismatch_count = random.randint(1, 2)
    total_needed = extra_match_count + mismatch_count + (1 if need_support else 0)
    selected = _row_choices(H, {bridge_row}, total_needed, min_gap=1)
    if selected is None:
        return None

    idx = 0
    support_row = None
    if need_support:
        support_row = selected[idx]
        idx += 1
    extra_match_rows = selected[idx : idx + extra_match_count]
    idx += extra_match_count
    mismatch_rows = selected[idx : idx + mismatch_count]

    matched_rows = [bridge_row, *extra_match_rows]
    row_markers: dict[int, tuple[int, int]] = {}
    row_markers[bridge_row] = (bridge_color, bridge_color)
    for r in extra_match_rows:
        color = random.choice(MARKER_COLORS)
        row_markers[r] = (color, color)

    if support_row is not None:
        side = random.choice(("left", "right"))
        support_color = random.choice(MARKER_COLORS)
        other = random.choice([c for c in MARKER_COLORS if c != support_color])
        row_markers[support_row] = (support_color, other) if side == "left" else (other, support_color)
        if not _place_endpoint_support(grid, support_row, support_color, side):
            return None
        mismatch_rows.append(support_row)

    for r in mismatch_rows:
        if r in row_markers:
            continue
        left, right = random.sample(MARKER_COLORS, 2)
        row_markers[r] = (left, right)

    for r, (left, right) in row_markers.items():
        grid[r, 0] = int(left)
        grid[r, W - 1] = int(right)

    if not _place_bridge_pair(grid, bridge_row, bridge_color):
        return None

    banned_near_fill = set()
    for r in matched_rows:
        banned_near_fill.update({r - 1, r, r + 1})
    safe_rows = {r for r in range(1, H - 1) if r not in banned_near_fill}
    if not safe_rows:
        return None

    far_small_area = random.choice((1, 2))
    far_small_color = random.choice(OBJECT_COLORS)
    if not _place_random_object(grid, far_small_color, far_small_area, allowed_rows=safe_rows):
        return None

    extra_object_count = random.randint(2, 5)
    for _ in range(extra_object_count):
        area = random.choice((1, 2, 3, 4, 3, 4))
        color = random.choice(OBJECT_COLORS)
        if not _place_random_object(grid, color, area, allowed_rows=safe_rows):
            continue

    return grid


def _semantic_valid(grid: np.ndarray) -> bool:
    H, W = grid.shape
    if grid.ndim != 2 or H > 30 or W > 30:
        return False
    if np.any(grid[:, 1:-1] == GREEN):
        return False
    if np.any(grid[:, 0] == GREEN) or np.any(grid[:, -1] == GREEN):
        return False
    for r in range(H):
        left = int(grid[r, 0])
        right = int(grid[r, -1])
        if (left == BACKGROUND) != (right == BACKGROUND):
            return False
    filled = _apply_row_fill(grid)
    out = _recolor_small_components(filled)
    if not np.array_equal(out, main(grid)):
        return False

    matched_rows = _match_rows(grid)
    if not matched_rows:
        return False

    filled_only = {
        (int(r), int(c))
        for r, c in np.argwhere((grid == BACKGROUND) & (filled != BACKGROUND))
    }
    after_components = find_components(filled, bg=BACKGROUND, connectivity=4, monochromatic=True)
    after_by_coord: dict[tuple[int, int], dict] = {}
    for comp in after_components:
        for r, c in comp["coords"]:
            after_by_coord[(int(r), int(c))] = comp

    bridge_survivors = 0
    before_small = find_components(grid, bg=BACKGROUND, connectivity=4, monochromatic=True)
    for comp in before_small:
        if int(comp["area"]) > 2:
            continue
        key = tuple(map(int, comp["coords"][0]))
        after_comp = after_by_coord.get(key)
        if after_comp is None:
            return False
        coords = comp["coords"]
        color = int(comp["color"])
        preserved = np.all(out[coords[:, 0], coords[:, 1]] == color)
        grew = int(after_comp["area"]) > int(comp["area"])
        uses_filled_bridge = any((int(r), int(c)) in filled_only for r, c in after_comp["coords"])
        if preserved and grew and uses_filled_bridge:
            bridge_survivors += 1

    if bridge_survivors < 2:
        return False

    far_green = False
    for comp in after_components:
        coords = comp["coords"]
        rr = coords[:, 0]
        if int(comp["area"]) <= 2 and np.all(out[rr, coords[:, 1]] == GREEN):
            if all(abs(int(r) - mr) > 1 for r in rr for mr in matched_rows):
                far_green = True
                break
    if not far_green:
        return False

    if np.array_equal(grid, out):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None and _semantic_valid(grid):
            return grid
    raise RuntimeError("failed to generate a valid row-bridge sample")
