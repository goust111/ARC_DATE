from __future__ import annotations

import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            seen[r, c] = True
            stack = [(r, c)]
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _border_mask(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _parse_legend_component(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    line_color: int,
    bg: int,
) -> dict[str, object] | None:
    r0, c0, r1, c1 = bbox
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]

    line_rows = [r for r in range(sub.shape[0]) if np.all(sub[r, :] == int(line_color))]
    line_cols = [c for c in range(sub.shape[1]) if np.all(sub[:, c] == int(line_color))]
    if len(line_rows) < 4 or len(line_cols) != 3:
        return None
    if line_rows[0] != 0 or line_rows[-1] != sub.shape[0] - 1:
        return None
    if line_cols[0] != 0 or line_cols[-1] != sub.shape[1] - 1:
        return None

    row_steps = [b - a for a, b in zip(line_rows, line_rows[1:])]
    col_steps = [b - a for a, b in zip(line_cols, line_cols[1:])]
    if len(set(row_steps)) != 1 or len(set(col_steps)) != 1:
        return None
    step = int(row_steps[0])
    if step != int(col_steps[0]) or step < 3:
        return None

    expected_lines = np.zeros_like(sub, dtype=bool)
    expected_lines[line_rows, :] = True
    expected_lines[:, line_cols] = True
    if not np.all(sub[expected_lines] == int(line_color)):
        return None
    if np.any(sub[~expected_lines] == int(line_color)):
        return None

    cell_size = step - 1
    mapping: dict[int, np.ndarray] = {}
    counts_in_order: list[int] = []
    for row_idx in range(len(line_rows) - 1):
        rr = slice(line_rows[row_idx] + 1, line_rows[row_idx + 1])
        left = sub[rr, line_cols[0] + 1 : line_cols[1]]
        right = sub[rr, line_cols[1] + 1 : line_cols[2]]
        key = int(np.count_nonzero(left != int(bg)))
        if key <= 0 or np.count_nonzero(right != int(bg)) == 0 or key in mapping:
            return None
        mapping[key] = right.copy()
        counts_in_order.append(key)

    expected_counts = list(range(1, len(counts_in_order) + 1))
    if counts_in_order != expected_counts:
        return None

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "line_color": int(line_color),
        "cell_size": int(cell_size),
        "mapping": mapping,
    }


def _find_legend(grid: np.ndarray, bg: int) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == int(bg):
            continue
        for comp in _color_components(grid, color):
            info = _parse_legend_component(grid, _bbox(comp), color, bg)
            if info is not None:
                candidates.append(info)
    assert candidates, "expected one top-left legend matrix"
    candidates.sort(
        key=lambda item: (
            int(item["bbox"][0]),
            int(item["bbox"][1]),
            -(
                (int(item["bbox"][2]) - int(item["bbox"][0]) + 1)
                * (int(item["bbox"][3]) - int(item["bbox"][1]) + 1)
            ),
        )
    )
    return candidates[0]


def _is_dice_face(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    line_color: int,
    bg: int,
    cell_size: int,
) -> bool:
    r0, c0, r1, c1 = bbox
    outer = cell_size + 2
    if r1 - r0 + 1 != outer or c1 - c0 + 1 != outer:
        return False
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = _border_mask(outer)
    if not np.all(sub[border] == int(line_color)):
        return False
    inner = sub[1:-1, 1:-1]
    if np.any(inner == int(line_color)):
        return False
    count = int(np.count_nonzero(inner != int(bg)))
    return count > 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each framed dot-count face with the pattern stored for that count in the top-left legend matrix.

    Input structure:
    - The background is the most common color.
    - A top-left legend matrix is drawn with one border/separator color and split into equal cells arranged in several rows and exactly two columns.
    - In each legend row, the left cell contains 1, 2, 3, ... foreground dots on background, while the right cell contains the replacement pattern for that count.
    - Elsewhere, scattered square faces use the same border color as a 1-pixel frame around a cell-sized interior that contains only foreground dots on background.

    Rule:
    - Detect the legend matrix, infer its cell size, and read the mapping from left-cell dot count to right-cell pattern.
    - For every scattered framed face with the matching border color and interior size, count its interior dots.
    - Replace that face's interior by the legend pattern for the same count, leaving the frame and all other cells unchanged.

    Key insight:
    - The left legend column is not output by itself; it is a count-based key that selects the pattern template stored in the same row's right legend cell.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    legend = _find_legend(g, bg)
    line_color = int(legend["line_color"])
    cell_size = int(legend["cell_size"])
    mapping = legend["mapping"]
    legend_bbox = legend["bbox"]

    out = g.copy()
    for comp in _color_components(g, line_color):
        box = _bbox(comp)
        if box == legend_bbox:
            continue
        if not _is_dice_face(g, box, line_color, bg, cell_size):
            continue
        r0, c0, r1, c1 = box
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        key = int(np.count_nonzero(inner != int(bg)))
        assert key in mapping, "every dice face count must appear in the legend"
        out[r0 + 1 : r1, c0 + 1 : c1] = np.asarray(mapping[key], dtype=int)

    return out


def _pip_pool(size: int) -> list[tuple[int, int]]:
    raw = [
        (0, 0),
        (0, size - 1),
        (size - 1, 0),
        (size - 1, size - 1),
        (size // 2, size // 2),
        (0, size // 2),
        (size // 2, 0),
        (size // 2, size - 1),
        (size - 1, size // 2),
        (1, 1),
        (1, size - 2),
        (size - 2, 1),
        (size - 2, size - 2),
    ]
    seen: set[tuple[int, int]] = set()
    pool: list[tuple[int, int]] = []
    for r, c in raw:
        if 0 <= r < size and 0 <= c < size and (r, c) not in seen:
            seen.add((r, c))
            pool.append((int(r), int(c)))
    return pool


def _random_dots(size: int, count: int) -> list[tuple[int, int]]:
    pool = _pip_pool(size)
    assert count <= len(pool)
    return random.sample(pool, count)


def _pattern_library(size: int) -> list[np.ndarray]:
    rr, cc = np.indices((size, size))
    mid = size // 2
    candidates = [
        np.ones((size, size), dtype=bool),
        (rr == mid) | (cc == mid),
        (rr == cc),
        (rr + cc == size - 1),
        (rr == cc) | (rr + cc == size - 1),
        (rr == size - 1) | (cc == 0),
        (rr == 0) | (cc == size - 1),
        (rr == 0) | (cc == 0) | (rr == size - 1),
        ((rr <= mid) & (cc == 0)) | (rr == mid),
        ((cc <= mid) & (rr == size - 1)) | (cc == mid),
    ]
    unique: list[np.ndarray] = []
    seen: set[bytes] = set()
    for mask in candidates:
        if not np.any(mask):
            continue
        key = mask.astype(np.uint8).tobytes()
        if key in seen:
            continue
        seen.add(key)
        unique.append(mask.astype(bool))
    return unique


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not bool(np.any(occupied[r0:r1, c0:c1]))


def _mark(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    occupied[top : top + height, left : left + width] = True


def _draw_legend(
    grid: np.ndarray,
    top: int,
    left: int,
    bg: int,
    line_color: int,
    dot_color: int,
    cell_size: int,
    counts: list[int],
    patterns: dict[int, np.ndarray],
) -> None:
    n_rows = len(counts)
    legend_h = n_rows * (cell_size + 1) + 1
    legend_w = 2 * (cell_size + 1) + 1
    grid[top : top + legend_h, left : left + legend_w] = int(line_color)
    for row_idx, count in enumerate(counts):
        inner_r = top + 1 + row_idx * (cell_size + 1)
        left_c = left + 1
        right_c = left + 1 + (cell_size + 1)
        grid[inner_r : inner_r + cell_size, left_c : left_c + cell_size] = int(bg)
        grid[inner_r : inner_r + cell_size, right_c : right_c + cell_size] = int(bg)
        for dr, dc in _random_dots(cell_size, count):
            grid[inner_r + dr, left_c + dc] = int(dot_color)
        right_cell = grid[inner_r : inner_r + cell_size, right_c : right_c + cell_size]
        pattern = patterns[count]
        mask = pattern != int(bg)
        right_cell[mask] = pattern[mask]


def _draw_face(
    grid: np.ndarray,
    top: int,
    left: int,
    bg: int,
    line_color: int,
    dot_color: int,
    cell_size: int,
    count: int,
) -> None:
    outer = cell_size + 2
    grid[top : top + outer, left : left + outer] = int(line_color)
    grid[top + 1 : top + 1 + cell_size, left + 1 : left + 1 + cell_size] = int(bg)
    for dr, dc in _random_dots(cell_size, count):
        grid[top + 1 + dr, left + 1 + dc] = int(dot_color)


def _choose_sizes() -> tuple[int, int]:
    options = [(3, 3), (3, 4), (3, 5), (4, 3), (4, 4), (4, 5), (5, 3), (5, 4)]
    return random.choice(options)


def _gen_attempt() -> np.ndarray | None:
    cell_size, n_rows = _choose_sizes()
    outer = cell_size + 2
    legend_h = n_rows * (cell_size + 1) + 1
    legend_w = 2 * (cell_size + 1) + 1
    counts = list(range(1, n_rows + 1))
    face_count = random.randint(n_rows, n_rows + 3)

    min_h = max(legend_h + 1, outer + 4, 22)
    min_w = max(legend_w + 1, outer + 4, 22)
    if min_h > 30 or min_w > 30:
        return None

    h = random.randint(min_h, 30)
    w = random.randint(min_w, 30)
    palette = random.sample(range(10), n_rows + 3)
    bg = palette[0]
    line_color = palette[1]
    dot_color = palette[2]
    pattern_colors = palette[3:]

    masks = random.sample(_pattern_library(cell_size), n_rows)
    patterns: dict[int, np.ndarray] = {}
    for count, mask, color in zip(counts, masks, pattern_colors):
        cell = np.full((cell_size, cell_size), int(bg), dtype=int)
        cell[mask] = int(color)
        patterns[count] = cell

    grid = np.full((h, w), int(bg), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    legend_top = random.randint(0, min(1, h - legend_h))
    legend_left = random.randint(0, min(1, w - legend_w))
    if not _can_place(occupied, legend_top, legend_left, legend_h, legend_w, pad=1):
        return None
    _draw_legend(
        grid,
        legend_top,
        legend_left,
        bg,
        line_color,
        dot_color,
        cell_size,
        counts,
        patterns,
    )
    _mark(occupied, legend_top, legend_left, legend_h, legend_w)

    face_counts = counts[:]
    face_counts.extend(random.choices(counts, k=face_count - len(counts)))
    random.shuffle(face_counts)
    face_specs: list[tuple[int, int, int]] = []

    for count in face_counts:
        positions = [
            (r, c)
            for r in range(h - outer + 1)
            for c in range(w - outer + 1)
            if _can_place(occupied, r, c, outer, outer, pad=1)
        ]
        if not positions:
            return None
        top, left = random.choice(positions)
        _draw_face(grid, top, left, bg, line_color, dot_color, cell_size, count)
        face_specs.append((top, left, count))
        _mark(occupied, top, left, outer, outer)

    if _most_common_color(grid) != int(bg):
        return None
    expected = grid.copy()
    for top, left, count in face_specs:
        expected[top + 1 : top + 1 + cell_size, left + 1 : left + 1 + cell_size] = np.asarray(patterns[count], dtype=int)
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
