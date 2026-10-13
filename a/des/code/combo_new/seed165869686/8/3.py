import random
from collections import Counter

import numpy as np


RED = 2
GEN_RETRIES = 400
GRID_HEIGHT_RANGE = (7, 20)
GRID_WIDTH_RANGE = (8, 30)
MAX_ARROW_ROWS = 5
ROW_LAYOUT_ATTEMPTS = 120


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _arrow_specs(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    specs: list[dict[str, object]] = []
    rows_used: set[int] = set()
    for comp in _components(grid == RED):
        if len(comp) != 3:
            continue
        rows = sorted({r for r, _ in comp})
        cols = sorted({c for _, c in comp})
        if len(rows) != 3 or rows[2] - rows[0] != 2:
            continue
        if len(cols) != 2 or cols[1] - cols[0] != 1:
            continue
        if cols[0] != 0 and cols[1] != w - 1:
            continue

        row_to_col = {r: [] for r in rows}
        for r, c in comp:
            row_to_col[r].append(c)
        if any(len(row_to_col[r]) != 1 for r in rows):
            continue

        top_c = int(row_to_col[rows[0]][0])
        mid_c = int(row_to_col[rows[1]][0])
        bot_c = int(row_to_col[rows[2]][0])
        if top_c != bot_c or mid_c == top_c:
            continue

        side = "left" if cols[0] == 0 else "right"
        direction = "right" if mid_c < top_c else "left"
        center_row = int(rows[1])
        if center_row in rows_used:
            raise AssertionError("multiple arrows detected on one row")
        rows_used.add(center_row)
        specs.append(
            {
                "row": center_row,
                "side": side,
                "direction": direction,
            }
        )

    specs.sort(key=lambda item: int(item["row"]))
    return specs


def _row_groups(row: np.ndarray, bg: int) -> tuple[Counter[int], dict[int, list[int]]]:
    counts: Counter[int] = Counter()
    positions: dict[int, list[int]] = {}
    for c, value in enumerate(row):
        color = int(value)
        if color == bg or color == RED:
            continue
        counts[color] += 1
        positions.setdefault(color, []).append(int(c))
    return counts, positions


def _stack_row(row: np.ndarray, bg: int, side: str, direction: str) -> np.ndarray:
    counts, positions = _row_groups(row, bg)
    out = np.full(row.shape, int(bg), dtype=int)
    if not counts:
        return out

    if side == "left":
        ordered_colors = sorted(positions, key=lambda color: min(positions[color]))
    else:
        ordered_colors = sorted(positions, key=lambda color: max(positions[color]), reverse=True)

    if direction == "right":
        cursor = len(row) - 1
        for color in ordered_colors:
            count = int(counts[color])
            out[cursor - count + 1 : cursor + 1] = int(color)
            cursor -= count
    else:
        cursor = 0
        for color in ordered_colors:
            count = int(counts[color])
            out[cursor : cursor + count] = int(color)
            cursor += count

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repack each arrow-marked row into tight color blocks against the wall named by the arrow.

    Input structure:
    - The background color is the most common color.
    - Some rows are the middle row of a red 3-pixel border arrow: one shaft pixel on the border column and two diagonal head pixels one column inward, so the left-border pattern points right and the right-border pattern points left.
    - On each arrow row, the remaining non-background pixels are scattered single cells from several colors; rows without arrow centers contain no such payload pixels.

    Rule:
    - Detect every border arrow, record its center row, the border side where it sits, and whether it points left or right.
    - On that center row, collect all non-background, non-arrow pixels and group them by color while counting how many cells each color has.
    - Order the color groups by the first original occurrence seen from the arrow side: left-border arrows use the smallest column per color, right-border arrows use the largest column per color.
    - Clear the whole row and place the color groups contiguously against the wall named by the arrow: right-pointing arrows stack from the far right toward the left, and left-pointing arrows stack from the far left toward the right. Remove the entire arrow to background.

    Key insight:
    - The arrow carries two pieces of information at once: its border side defines the proximity-based sort order, while its pointing direction defines which wall receives the final tight stack.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    out = g.copy()
    out[out == RED] = int(bg)

    for spec in _arrow_specs(g):
        row = int(spec["row"])
        out[row, :] = _stack_row(g[row], bg, str(spec["side"]), str(spec["direction"]))

    return out


def _sample_center_rows(height: int, count: int, rng: random.Random) -> list[int] | None:
    candidates = list(range(1, height - 1))
    rng.shuffle(candidates)
    centers: list[int] = []
    for row in candidates:
        if all(abs(row - other) >= 3 for other in centers):
            centers.append(int(row))
        if len(centers) == count:
            break
    if len(centers) != count:
        return None
    centers.sort()
    return centers


def _sample_row_payload(
    width: int,
    bg: int,
    direction: str,
    rng: random.Random,
) -> np.ndarray | None:
    side = "left" if direction == "right" else "right"
    forbidden_col = 0 if side == "left" else width - 1
    usable_cols = [c for c in range(width) if c != forbidden_col]
    palette = [c for c in range(10) if c not in {bg, RED}]
    max_colors = min(5, len(palette), len(usable_cols) - 1)
    if max_colors < 2:
        return None

    for _ in range(ROW_LAYOUT_ATTEMPTS):
        num_colors = rng.randint(2, max_colors)
        counts = [rng.randint(1, 3) for _ in range(num_colors)]
        total = sum(counts)
        if total > len(usable_cols) - 1:
            continue

        colors = rng.sample(palette, num_colors)
        cols = sorted(rng.sample(usable_cols, total))
        bag: list[int] = []
        for color, count in zip(colors, counts):
            bag.extend([int(color)] * int(count))
        rng.shuffle(bag)

        row = np.full(width, int(bg), dtype=int)
        for col, color in zip(cols, bag):
            row[int(col)] = int(color)

        if np.array_equal(row, _stack_row(row, bg, side, direction)):
            continue
        return row

    return None


def _place_inward_arrow(grid: np.ndarray, row: int, direction: str) -> None:
    if direction == "right":
        grid[row, 0] = RED
        grid[row - 1, 1] = RED
        grid[row + 1, 1] = RED
    else:
        grid[row, -1] = RED
        grid[row - 1, -2] = RED
        grid[row + 1, -2] = RED


def _bg_is_unique_mode(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.reshape(-1)))
    return all(counts[bg] > count for color, count in counts.items() if color != bg)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_HEIGHT_RANGE)
    width = rng.randint(*GRID_WIDTH_RANGE)
    bg = rng.choice([c for c in range(10) if c != RED])
    grid = np.full((height, width), int(bg), dtype=int)

    max_arrow_rows = min(MAX_ARROW_ROWS, height // 3)
    if max_arrow_rows < 1:
        return None
    arrow_count = rng.randint(1, max_arrow_rows)
    rows = _sample_center_rows(height, arrow_count, rng)
    if rows is None:
        return None

    directions = [rng.choice(["left", "right"]) for _ in rows]
    for row, direction in zip(rows, directions):
        payload = _sample_row_payload(width, bg, direction, rng)
        if payload is None:
            return None
        grid[row, :] = payload
        _place_inward_arrow(grid, row, direction)

    if not _bg_is_unique_mode(grid, bg):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    for row in rows:
        cleared = grid[row].copy()
        cleared[cleared == RED] = int(bg)
        if np.array_equal(cleared, out[row]):
            return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
