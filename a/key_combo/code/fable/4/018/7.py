from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

if str(Path("/home/lixinyu/snap/new/solvers")) not in sys.path:
    sys.path.append(str(Path("/home/lixinyu/snap/new/solvers")))

from arc_utils import Color, find_components, get_bg  # type: ignore  # noqa: E402


MIN_GRID_SIDE = 10
MAX_GRID_SIDE = 24
MIN_POLE_COUNT = 2
MAX_POLE_COUNT = 7
MIN_POLE_HEIGHT = 2
MAX_POLE_HEIGHT = 8
GENERATION_RETRIES = 150

_POLE_COLORS = [
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
    Color.YELLOW,
]


def _segment_columns(foot_col: int, height: int, width: int, direction: int) -> list[int]:
    if direction == 1:
        end = min(width - 1, foot_col + height - 1)
        return list(range(foot_col, end + 1))
    start = max(0, foot_col - height + 1)
    return list(range(start, foot_col + 1))


def _extract_poles(grid: np.ndarray, bg: int) -> dict[int, dict[str, int | np.ndarray]]:
    components = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    poles: dict[int, dict[str, int | np.ndarray]] = {}

    for comp in components:
        color = int(comp["color"])
        if color == bg or color == Color.GREY:
            continue

        coords = np.asarray(comp["coords"], dtype=int)
        cols = np.unique(coords[:, 1])
        if cols.size != 1:
            continue

        rows = np.unique(coords[:, 0])
        if rows.size != rows.max() - rows.min() + 1:
            continue

        top = int(rows.min())
        foot = int(rows.max())
        if foot >= grid.shape[0] - 1:
            continue

        poles[color] = {
            "coords": coords,
            "col": int(cols[0]),
            "top_row": top,
            "foot_row": foot,
            "height": int(rows.size),
        }

    return poles


def _find_start(grid: np.ndarray, poles: dict[int, dict[str, int | np.ndarray]], bg: int) -> tuple[int, int]:
    gray_cells = np.argwhere(grid == Color.GREY)
    assert gray_cells.shape == (1, 2), "input must contain exactly one gray cap"

    hat_r, hat_c = map(int, gray_cells[0])
    candidates: list[tuple[int, int]] = []
    for color, pole in poles.items():
        if int(pole["top_row"]) != hat_r:
            continue
        pole_col = int(pole["col"])
        if abs(pole_col - hat_c) != 1:
            continue
        if grid[hat_r, pole_col] != color:
            continue
        direction = 1 if pole_col < hat_c else -1
        candidates.append((int(color), direction))

    assert len(candidates) == 1, "exactly one pole must have a gray top hat"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate height-based domino toppling of vertical poles.

    Input structure:
    - The background is the dominant color.
    - Several disjoint width-1 vertical monochrome poles stand above background.
    - Exactly one pole has one gray hat pixel touching the top on its left or right.
    - The hat position indicates the starting toppling direction.

    Rule:
    - Collapse the starting pole first and then every triggered pole in order.
    - A fallen pole vanishes and draws a horizontal segment on its foot row with length
      equal to its own height, clipped to the grid and including the foot cell.
    - If that segment touches the foot column of a standing pole, that pole topples
      immediately in the same direction.
    - Later fallen segments overwrite earlier cells when they overlap.

    Key insight:
    - The continuation is driven by the chain of heights: each new fallen pole sets
      a fresh reach length for the next reach test.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)

    poles = _extract_poles(g, bg=bg)
    assert poles, "no valid poles found"
    start_color, start_direction = _find_start(g, poles, bg=bg)

    out = g.copy()
    standing = set(poles.keys())
    queue: deque[tuple[int, int]] = deque([(start_color, start_direction)])

    while queue:
        color, direction = queue.popleft()
        if color not in standing:
            continue
        standing.remove(color)

        pole = poles[color]
        coords = np.asarray(pole["coords"], dtype=int)
        foot_row = int(pole["foot_row"])
        foot_col = int(pole["col"])
        height = int(pole["height"])

        out[coords[:, 0], coords[:, 1]] = bg

        cols = _segment_columns(foot_col, height, g.shape[1], direction)
        if cols:
            out[foot_row, cols] = color

        if not standing:
            continue

        touched: list[tuple[int, int]] = []
        for other_color, other in poles.items():
            if other_color not in standing:
                continue
            if int(other["col"]) in cols:
                dist = abs(int(other["col"]) - foot_col)
                touched.append((dist, other_color))

        if touched:
            touched.sort(key=lambda x: x[0])
            for _, other_color in touched:
                queue.append((other_color, direction))

    return out


def _random_height(H: int) -> int:
    max_h = max(MIN_POLE_HEIGHT, min(MAX_POLE_HEIGHT, H - 2))
    return random.randint(MIN_POLE_HEIGHT, max_h)


def _random_foot_row(H: int, height: int) -> int:
    return random.randint(height, H - 2)


def _can_draw_pole(grid: np.ndarray, col: int, top: int, foot: int, bg: int) -> bool:
    return bool(np.all(grid[top : foot + 1, col] == bg))


def _draw_pole(grid: np.ndarray, col: int, top: int, foot: int, color: int, bg: int) -> None:
    _ = bg
    grid[top : foot + 1, col] = int(color)


def _make_chain(
    start_col: int,
    start_foot: int,
    start_height: int,
    direction: int,
    chain_len: int,
    H: int,
    W: int,
    used_cols: set[int],
) -> list[tuple[int, int, int]]:
    chain = [(start_col, start_foot, start_height)]
    current_col = start_col
    current_height = start_height

    for _ in range(1, chain_len):
        max_offset = max(1, current_height - 1)
        placed = False
        for _ in range(40):
            offset = random.randint(1, max_offset)
            col = current_col + direction * offset
            if not (0 <= col < W) or col in used_cols:
                continue
            height = _random_height(H)
            foot = _random_foot_row(H, height)
            top = foot - height + 1
            if top < 0:
                continue
            chain.append((col, foot, height))
            used_cols.add(col)
            current_col = col
            current_height = height
            placed = True
            break
        if not placed:
            return []

    return chain


def _make_off_chain(col_count: int, H: int, W: int, used_cols: set[int]) -> list[tuple[int, int, int]]:
    candidates = [c for c in range(W) if c not in used_cols]
    random.shuffle(candidates)
    out = []
    for col in candidates[:col_count]:
        height = _random_height(H)
        foot = _random_foot_row(H, height)
        top = foot - height + 1
        out.append((col, foot, height))
    return out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        H = random.randint(MIN_GRID_SIDE, MAX_GRID_SIDE)
        W = random.randint(MIN_GRID_SIDE, MAX_GRID_SIDE)
        bg = Color.BLACK
        grid = np.full((H, W), bg, dtype=int)

        direction = random.choice((-1, 1))
        pole_count = random.randint(MIN_POLE_COUNT, MAX_POLE_COUNT)
        chain_mode = random.random()
        if chain_mode < 0.35:
            chain_len = 1
        elif chain_mode < 0.70:
            chain_len = random.randint(2, min(4, pole_count))
        elif chain_mode < 0.90:
            chain_len = random.randint(3, min(5, pole_count)) if pole_count >= 3 else 2
        else:
            chain_len = pole_count

        if direction == 1:
            start_col = random.randint(1, W - 2) if random.random() >= 0.25 else random.randint(max(0, W - 5), W - 2)
        else:
            start_col = random.randint(1, W - 2) if random.random() >= 0.25 else random.randint(1, min(4, W - 2))
        hat_col = start_col + direction
        if not (0 <= hat_col < W):
            continue

        start_height = _random_height(H)
        if random.random() < 0.25:
            start_height = max(start_height, _random_height(H))
        start_foot = _random_foot_row(H, start_height)
        start_top = start_foot - start_height + 1
        if start_top < 0:
            continue

        used_cols = {start_col, hat_col}
        chain = _make_chain(
            start_col=start_col,
            start_foot=start_foot,
            start_height=start_height,
            direction=direction,
            chain_len=chain_len,
            H=H,
            W=W,
            used_cols=used_cols,
        )
        if not chain:
            continue

        specs = chain + _make_off_chain(max(0, pole_count - len(chain)), H, W, used_cols=used_cols)
        if len(specs) != pole_count:
            continue

        specs_for_draw: list[tuple[int, int, int]] = []
        used_cols = set()
        for col, foot, height in specs:
            top = foot - height + 1
            if top < 0 or col in used_cols or not _can_draw_pole(grid, col=col, top=top, foot=foot, bg=bg):
                break
            used_cols.add(col)
            specs_for_draw.append((col, top, foot))
        if len(specs_for_draw) != len(specs):
            continue

        colors = random.sample(_POLE_COLORS, k=pole_count)
        random.shuffle(specs_for_draw)
        for (col, top, foot), color in zip(specs_for_draw, colors):
            _draw_pole(grid=grid, col=col, top=top, foot=foot, color=int(color), bg=bg)

        # Place the gray cap on the start pole top.
        if grid[start_top, hat_col] != bg:
            continue
        grid[start_top, hat_col] = Color.GREY

        # Rebuild to ensure our assumptions are respected.
        try:
            _ = main(grid)
            gray_cells = np.argwhere(grid == Color.GREY)
            if gray_cells.shape != (1, 2):
                continue
            poles = _extract_poles(grid, bg=bg)
            start_color, start_direction = _find_start(grid, poles, bg=bg)
            assert start_color in poles and start_direction in (-1, 1)
            assert start_direction == direction
        except Exception:
            continue

        return grid

    raise RuntimeError("failed to generate valid input for 7.py")
