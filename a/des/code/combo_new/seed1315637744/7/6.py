from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            components.append(np.array(cells, dtype=int))

    return components


def _runs(values: list[int]) -> list[tuple[int, int]]:
    assert values
    runs: list[tuple[int, int]] = []
    cur = int(values[0])
    length = 1
    for value in values[1:]:
        value = int(value)
        if value == cur:
            length += 1
        else:
            runs.append((cur, int(length)))
            cur = value
            length = 1
    runs.append((cur, int(length)))
    return runs


def _main_region(inner: np.ndarray) -> tuple[int, int] | None:
    n = inner.shape[0]
    assert inner.shape == (n, n)

    if all(bool(np.all(inner[r, :] == inner[r, 0])) for r in range(n)):
        row_colors = [int(inner[r, 0]) for r in range(n)]
        row_runs = _runs(row_colors)
        if len(row_runs) == 2 and int(row_runs[0][0]) != int(row_runs[1][0]):
            (color_a, len_a), (color_b, len_b) = row_runs
            if len_a != len_b:
                return (int(color_a), int(len_a)) if len_a > len_b else (int(color_b), int(len_b))

    if all(bool(np.all(inner[:, c] == inner[0, c])) for c in range(n)):
        col_colors = [int(inner[0, c]) for c in range(n)]
        col_runs = _runs(col_colors)
        if len(col_runs) == 2 and int(col_runs[0][0]) != int(col_runs[1][0]):
            (color_a, len_a), (color_b, len_b) = col_runs
            if len_a != len_b:
                return (int(color_a), int(len_a)) if len_a > len_b else (int(color_b), int(len_b))

    return None


def _extract_containers(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    components = _connected_components(g == GREY)
    containers: list[dict[str, int]] = []

    for cells in components:
        r0, c0 = map(int, cells.min(axis=0))
        r1, c1 = map(int, cells.max(axis=0))
        size = int(r1 - r0 + 1)
        if size < 5 or int(c1 - c0 + 1) != size:
            continue

        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        if not bool(np.all(sub[0, :] == GREY)):
            continue
        if not bool(np.all(sub[-1, :] == GREY)):
            continue
        if not bool(np.all(sub[:, 0] == GREY)):
            continue
        if not bool(np.all(sub[:, -1] == GREY)):
            continue

        inner = sub[1:-1, 1:-1]
        if inner.shape != (size - 2, size - 2):
            continue
        if inner.size == 0 or bool(np.any((inner == BLACK) | (inner == GREY))):
            continue

        colors = {int(v) for v in np.unique(inner)}
        if len(colors) != 2:
            continue

        info = _main_region(inner)
        if info is None:
            continue
        color, side = info
        containers.append({"left": int(c0), "top": int(r0), "main_color": int(color), "side": int(side)})

    assert containers, "expected at least one valid container"
    containers.sort(key=lambda item: (int(item["left"]), int(item["top"])))
    return containers


def _render_output(containers: list[dict[str, int]]) -> np.ndarray:
    height = max(int(item["side"]) for item in containers)
    width = sum(int(item["side"]) for item in containers) + max(0, len(containers) - 1)
    out = np.full((height, width), BLACK, dtype=int)

    c = 0
    for item in containers:
        side = int(item["side"])
        color = int(item["main_color"])
        out[height - side : height, c : c + side] = color
        c += side + 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn each partitioned grey container into a pure square whose size matches the larger color region's span.

    Input structure:
    - The background is black.
    - Several equal-sized square containers are outlined by a 1-pixel grey border.
    - Each container interior is filled by exactly two non-black, non-grey colors separated by one straight horizontal or vertical split.
    - One interior color occupies a strictly larger rectangular region than the other, and containers are spread from left to right.

    Rule:
    - Detect each grey square container and inspect its interior split.
    - Identify the larger color region; if the split is horizontal, its height in rows is K, and if the split is vertical, its width in columns is K.
    - Replace that container with a solid KxK square of the larger region's color.
    - Order the output squares by the original container positions from left to right, bottom-align them, and separate consecutive squares by one black column.

    Key insight:
    - The scale of the output is not the larger region's pixel area directly; it is the region's span along the split direction, which equals area divided by the common interior side length.
    """
    containers = _extract_containers(np.asarray(grid, dtype=int))
    return _render_output(containers)


def _distribute(total: int, slots: int) -> list[int]:
    parts = [0] * slots
    for _ in range(total):
        parts[random.randrange(slots)] += 1
    return parts


def _choose_layout() -> tuple[int, int]:
    count = random.choice((2, 3, 4))
    if count == 2:
        inner = random.randint(3, 9)
    elif count == 3:
        inner = random.randint(3, 6)
    else:
        inner = random.randint(3, 4)
    return count, inner


def _make_container(inner_size: int, orientation: str, major_side: int, main_first: bool, main_color: int, other_color: int) -> np.ndarray:
    size = inner_size + 2
    tile = np.full((size, size), GREY, dtype=int)
    inner = tile[1:-1, 1:-1]
    minor_side = inner_size - major_side
    assert 1 <= minor_side < major_side < inner_size

    if orientation == "horizontal":
        if main_first:
            inner[:major_side, :] = int(main_color)
            inner[major_side:, :] = int(other_color)
        else:
            inner[:minor_side, :] = int(other_color)
            inner[minor_side:, :] = int(main_color)
    else:
        if main_first:
            inner[:, :major_side] = int(main_color)
            inner[:, major_side:] = int(other_color)
        else:
            inner[:, :minor_side] = int(other_color)
            inner[:, minor_side:] = int(main_color)

    return tile


def _gen_attempt() -> np.ndarray | None:
    count, inner_size = _choose_layout()
    container_size = inner_size + 2

    min_width = count * container_size + (count - 1) + 2
    if min_width > 30:
        return None

    width_slack = 30 - min_width
    width_extra = random.randint(0, width_slack)
    width_parts = _distribute(width_extra, count + 1)
    left_margin = 1 + width_parts[0]
    gaps = [1 + width_parts[i + 1] for i in range(count - 1)]
    right_margin = 1 + width_parts[-1]
    width = count * container_size + sum(gaps) + left_margin + right_margin

    min_height = container_size + 2
    height_slack = min(6, 30 - min_height)
    height = min_height + random.randint(0, height_slack)

    max_top = height - container_size - 1
    if max_top < 1:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    specs: list[dict[str, int]] = []

    c0 = left_margin
    for idx in range(count):
        orientation = random.choice(("horizontal", "vertical"))
        major_side = random.randint(inner_size // 2 + 1, inner_size - 1)
        main_first = bool(random.getrandbits(1))
        main_color, other_color = random.sample(PALETTE, 2)
        top = random.randint(1, max_top)

        tile = _make_container(inner_size, orientation, major_side, main_first, int(main_color), int(other_color))
        grid[top : top + container_size, c0 : c0 + container_size] = tile
        specs.append({"left": int(c0), "top": int(top), "main_color": int(main_color), "side": int(major_side)})

        if idx < count - 1:
            c0 += container_size + gaps[idx]

    expected = _render_output(sorted(specs, key=lambda item: (int(item["left"]), int(item["top"]))))
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated attempts")
