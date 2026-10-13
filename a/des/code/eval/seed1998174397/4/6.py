from __future__ import annotations

from collections import deque
import random

import numpy as np


BORDER = 2
SIDE_ORDER = ("top", "right", "bottom", "left")
COLUMN_SIDES = {"top", "bottom"}
ROW_SIDES = {"left", "right"}
GEN_RETRIES = 500
PALETTE = tuple(range(10))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _border_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[:BORDER, :] = True
    mask[-BORDER:, :] = True
    mask[:, :BORDER] = True
    mask[:, -BORDER:] = True
    return mask


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _axis(side: str) -> str:
    return "col" if side in COLUMN_SIDES else "row"


def _detect_arrows(grid: np.ndarray, bg: int) -> list[dict[str, int | str]]:
    h, w = grid.shape
    mask = _border_mask(grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    arrows: list[dict[str, int | str]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or not mask[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            rows = {rr for rr, _ in cells}
            cols = {cc for _, cc in cells}
            if 0 in rows:
                side = "top"
                steps = sum(1 for rr, _ in cells if rr == 0)
                order_key = (0, min(cc for _, cc in cells))
            elif h - 1 in rows:
                side = "bottom"
                steps = sum(1 for rr, _ in cells if rr == h - 1)
                order_key = (2, -max(cc for _, cc in cells))
            elif 0 in cols:
                side = "left"
                steps = sum(1 for _, cc in cells if cc == 0)
                order_key = (3, -max(rr for rr, _ in cells))
            elif w - 1 in cols:
                side = "right"
                steps = sum(1 for _, cc in cells if cc == w - 1)
                order_key = (1, min(rr for rr, _ in cells))
            else:
                raise AssertionError("border marker does not touch an outer edge")

            arrows.append(
                {
                    "side": side,
                    "color": color,
                    "steps": int(steps),
                    "order_major": int(order_key[0]),
                    "order_minor": int(order_key[1]),
                }
            )

    arrows.sort(key=lambda arrow: (int(arrow["order_major"]), int(arrow["order_minor"])))
    return arrows


def _find_target_index(interior: np.ndarray, color: int, side: str) -> int:
    coords = np.argwhere(interior == color)
    if len(coords) == 0:
        raise AssertionError(f"missing target color {color}")
    if side in COLUMN_SIDES:
        cols = {int(c) for _, c in coords}
        if len(cols) != 1:
            raise AssertionError(f"color {color} must occupy exactly one target column")
        return next(iter(cols))
    rows = {int(r) for r, _ in coords}
    if len(rows) != 1:
        raise AssertionError(f"color {color} must occupy exactly one target row")
    return next(iter(rows))


def _apply_arrow(interior: np.ndarray, arrow: dict[str, int | str]) -> None:
    side = str(arrow["side"])
    color = int(arrow["color"])
    steps = int(arrow["steps"])
    target = _find_target_index(interior, color, side)
    if side == "top":
        interior[:, target] = np.roll(interior[:, target], steps)
    elif side == "bottom":
        interior[:, target] = np.roll(interior[:, target], -steps)
    elif side == "left":
        interior[target, :] = np.roll(interior[target, :], steps)
    elif side == "right":
        interior[target, :] = np.roll(interior[target, :], -steps)
    else:
        raise AssertionError(side)


def _transform(interior: np.ndarray, arrows: list[dict[str, int | str]]) -> np.ndarray:
    out = np.array(interior, dtype=int, copy=True)
    for arrow in arrows:
        _apply_arrow(out, arrow)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or min(g.shape) <= 2 * BORDER:
        raise AssertionError("expected a 2D grid with a 2-cell border")
    bg = _get_bg(g)
    arrows = _detect_arrows(g, bg)
    interior = g[BORDER:-BORDER, BORDER:-BORDER]
    return _transform(interior, arrows)


def _singleton_position(grid: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(grid == color)
    if len(coords) != 1:
        raise AssertionError(f"color {color} must be a singleton in generator state")
    r, c = coords[0]
    return int(r), int(c)


def _dependency_realized(interior: np.ndarray, arrows: list[dict[str, int | str]]) -> bool:
    current = np.array(interior, dtype=int, copy=True)
    initial = {int(arrow["color"]): _singleton_position(current, int(arrow["color"])) for arrow in arrows}
    changed_target = False
    for arrow in arrows:
        side = str(arrow["side"])
        color = int(arrow["color"])
        row, col = _singleton_position(current, color)
        initial_row, initial_col = initial[color]
        if side in COLUMN_SIDES and col != initial_col:
            changed_target = True
        if side in ROW_SIDES and row != initial_row:
            changed_target = True
        _apply_arrow(current, arrow)
    return changed_target


def _shuffled(values: list[int], rng: random.Random) -> list[int]:
    result = list(values)
    rng.shuffle(result)
    return result


def _choose_arrow_positions(sides: list[str], ih: int, iw: int, rng: random.Random) -> list[tuple[int, int]]:
    rows = list(range(1, ih - 1))
    cols = list(range(1, iw - 1))
    positions: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()

    first = (rng.choice(rows), rng.choice(cols))
    positions.append(first)
    used.add(first)

    for idx in range(1, len(sides)):
        side = sides[idx]
        axis = _axis(side)
        candidates: list[tuple[int, int]] = []

        prior_indices = list(range(idx))
        rng.shuffle(prior_indices)
        for prior_idx in prior_indices:
            prior_side = sides[prior_idx]
            if _axis(prior_side) == axis:
                continue
            prior_r, prior_c = positions[prior_idx]
            if _axis(prior_side) == "col":
                for row in _shuffled(rows, rng):
                    pos = (row, prior_c)
                    if pos not in used:
                        candidates.append(pos)
            else:
                for col in _shuffled(cols, rng):
                    pos = (prior_r, col)
                    if pos not in used:
                        candidates.append(pos)

        if not candidates:
            for row in _shuffled(rows, rng):
                for col in _shuffled(cols, rng):
                    pos = (row, col)
                    if pos not in used:
                        candidates.append(pos)

        if not candidates:
            raise AssertionError("unable to place arrow target colors")

        chosen = candidates[0]
        positions.append(chosen)
        used.add(chosen)

    return positions


def _place_fillers(interior: np.ndarray, colors: list[int], rng: random.Random) -> None:
    ih, iw = interior.shape
    rows = list(range(1, ih - 1))
    cols = list(range(1, iw - 1))
    bg = _get_bg(interior)

    for color in colors:
        for _ in range(20):
            orientation = rng.choice(("row", "col"))
            if orientation == "row":
                row = rng.choice(rows)
                free_cols = [col for col in cols if int(interior[row, col]) == bg]
                if len(free_cols) < 2:
                    continue
                count = rng.randint(2, min(3, len(free_cols)))
                chosen = rng.sample(free_cols, count)
                interior[row, chosen] = color
                break
            col = rng.choice(cols)
            free_rows = [row for row in rows if int(interior[row, col]) == bg]
            if len(free_rows) < 2:
                continue
            count = rng.randint(2, min(3, len(free_rows)))
            chosen = rng.sample(free_rows, count)
            interior[chosen, col] = color
            break


def _draw_arrow(
    grid: np.ndarray,
    side: str,
    color: int,
    steps: int,
    rng: random.Random,
) -> None:
    h, w = grid.shape
    if side in COLUMN_SIDES:
        start = rng.randint(0, (w - 2 * BORDER) - steps)
        cols = [BORDER + start + offset for offset in range(steps)]
        tip_col = cols[0] if rng.random() < 0.5 else cols[-1]
        if side == "top":
            grid[0, cols] = color
            grid[1, tip_col] = color
        else:
            grid[h - 1, cols] = color
            grid[h - 2, tip_col] = color
        return

    start = rng.randint(0, (h - 2 * BORDER) - steps)
    rows = [BORDER + start + offset for offset in range(steps)]
    tip_row = rows[0] if rng.random() < 0.5 else rows[-1]
    if side == "left":
        grid[rows, 0] = color
        grid[tip_row, 1] = color
    else:
        grid[rows, w - 1] = color
        grid[tip_row, w - 2] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    ih = rng.randint(6, 10)
    iw = rng.randint(6, 10)
    bg = rng.choice(PALETTE)
    palette = [color for color in PALETTE if color != bg]
    rng.shuffle(palette)

    side_count = rng.choice((3, 4))
    selected = set(rng.sample(SIDE_ORDER, side_count))
    sides = [side for side in SIDE_ORDER if side in selected]

    arrow_colors = palette[: len(sides)]
    filler_colors = palette[len(sides) : len(sides) + rng.randint(2, 4)]
    positions = _choose_arrow_positions(sides, ih, iw, rng)

    interior = np.full((ih, iw), bg, dtype=int)
    arrows: list[dict[str, int | str]] = []
    for side, color, (row, col) in zip(sides, arrow_colors, positions):
        interior[row, col] = color
        arrows.append({"side": side, "color": color, "steps": rng.randint(1, 4)})

    _place_fillers(interior, filler_colors, rng)

    full = np.full((ih + 2 * BORDER, iw + 2 * BORDER), bg, dtype=int)
    full[BORDER:-BORDER, BORDER:-BORDER] = interior
    for arrow in arrows:
        _draw_arrow(full, str(arrow["side"]), int(arrow["color"]), int(arrow["steps"]), rng)

    out = main(full)
    if out.shape != interior.shape:
        return None
    if np.count_nonzero(out != interior) < max(3, len(arrows)):
        return None
    if not _dependency_realized(interior, arrows):
        return None
    return full


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
