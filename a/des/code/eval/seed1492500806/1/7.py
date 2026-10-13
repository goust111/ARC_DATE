from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
GREY = 5
MAROON = 9
ORANGE = 7
LIQUID_COLORS = (1, 2, 3, 4, 6, 8)
MAX_GENERATION_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_container(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict[str, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 2 or width < 3:
        return None

    shell = np.zeros((height, width), dtype=bool)
    shell[:, 0] = True
    shell[:, -1] = True
    shell[-1, :] = True

    patch = np.zeros((height, width), dtype=bool)
    for r, c in coords:
        patch[r - r0, c - c0] = True
    if not np.array_equal(shell, patch):
        return None

    interior = grid[r0:r1, c0 + 1 : c1]
    if interior.size == 0:
        return None

    liquid_mask = (interior != BG) & (interior != GREY)
    if not bool(liquid_mask.any()):
        return None

    liquid_values = interior[liquid_mask]
    liquid_color = int(Counter(map(int, liquid_values.reshape(-1))).most_common(1)[0][0])

    row_has_liquid = liquid_mask.any(axis=1)
    fill_height = 0
    for idx in range(row_has_liquid.shape[0] - 1, -1, -1):
        if bool(row_has_liquid[idx]):
            fill_height += 1
        else:
            break
    if fill_height == 0:
        return None
    if bool(row_has_liquid[:-fill_height].any()):
        return None

    filled_patch = interior[-fill_height:, :]
    if not bool(np.all(filled_patch == liquid_color)):
        return None
    if fill_height < interior.shape[0] and not bool(np.all(interior[:-fill_height, :] == BG)):
        return None

    marker_color = None
    marker_row = r1 + 1
    if marker_row < grid.shape[0]:
        for col in range(c0, c1 + 1):
            value = int(grid[marker_row, col])
            if value in (MAROON, ORANGE):
                marker_color = value
                break

    return {
        "top": int(r0),
        "left": int(c0),
        "height": int(height),
        "width": int(width),
        "fill_height": int(fill_height),
        "liquid_color": int(liquid_color),
        "marker_color": None if marker_color is None else int(marker_color),
    }


def _find_containers(grid: np.ndarray) -> list[dict[str, int]]:
    containers: list[dict[str, int]] = []
    for comp in _components(grid == GREY):
        container = _extract_container(grid, comp)
        if container is not None:
            containers.append(container)
    containers.sort(key=lambda item: item["left"])
    return containers


def _draw_container(
    canvas: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    fill_height: int,
    liquid_color: int,
    marker_color: int | None = None,
) -> None:
    canvas[top : top + height, left] = GREY
    canvas[top : top + height, left + width - 1] = GREY
    canvas[top + height - 1, left : left + width] = GREY
    if fill_height > 0:
        fill_top = top + (height - 1 - fill_height)
        canvas[fill_top : top + height - 1, left + 1 : left + width - 1] = liquid_color
    if marker_color is not None:
        marker_col = left + width // 2
        canvas[top + height, marker_col] = marker_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort U-shaped containers by their liquid heights, then repack them left-to-right.
    """
    g = np.asarray(grid, dtype=int)
    containers = _find_containers(g)
    assert containers, "expected at least one valid container"

    marker_colors = {item["marker_color"] for item in containers if item["marker_color"] is not None}
    assert marker_colors, "expected marker pixels below the containers"
    assert len(marker_colors) == 1, "all containers must share one sort-direction marker"
    descending = next(iter(marker_colors)) == ORANGE

    if descending:
        ordered = sorted(containers, key=lambda item: (-item["fill_height"], item["left"]))
    else:
        ordered = sorted(containers, key=lambda item: (item["fill_height"], item["left"]))

    out_h = max(item["height"] for item in ordered)
    out_w = sum(item["width"] for item in ordered) + max(0, len(ordered) - 1)
    out = np.full((out_h, out_w), BG, dtype=int)

    left = 0
    for item in ordered:
        top = out_h - item["height"]
        _draw_container(
            out,
            top=top,
            left=left,
            height=item["height"],
            width=item["width"],
            fill_height=item["fill_height"],
            liquid_color=item["liquid_color"],
        )
        left += item["width"] + 1

    return out


def _sample_fill_heights(count: int) -> list[int]:
    max_fill = random.randint(max(3, count), max(5, count + 3))
    return random.sample(range(1, max_fill + 1), count)


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(3, 5)
    inner_width = random.randint(2, 4)
    width = inner_width + 2
    marker_color = random.choice((MAROON, ORANGE))
    fill_heights = _sample_fill_heights(count)
    liquid_colors = random.sample(LIQUID_COLORS, count)

    containers: list[dict[str, int]] = []
    for idx in range(count):
        headroom = random.randint(0, 3)
        height = fill_heights[idx] + headroom + 1
        containers.append(
            {
                "height": int(height),
                "width": int(width),
                "fill_height": int(fill_heights[idx]),
                "liquid_color": int(liquid_colors[idx]),
            }
        )

    target_order = sorted(
        range(count),
        key=lambda idx: fill_heights[idx],
        reverse=marker_color == ORANGE,
    )

    input_order = list(range(count))
    random.shuffle(input_order)
    if input_order == target_order:
        return None

    gaps = [random.randint(1, 3) for _ in range(count - 1)]
    cluster_w = count * width + sum(gaps)
    max_height = max(item["height"] for item in containers)
    top_margin = random.randint(0, 3)
    left_margin = random.randint(0, 3)
    right_margin = random.randint(0, 3)
    bottom_margin = random.randint(1, 3)
    grid_h = top_margin + max_height + 1 + bottom_margin
    grid_w = left_margin + cluster_w + right_margin
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    baseline = top_margin + max_height - 1
    left = left_margin
    for pos, idx in enumerate(input_order):
        item = containers[idx]
        top = baseline - item["height"] + 1
        _draw_container(
            grid,
            top=top,
            left=left,
            height=item["height"],
            width=item["width"],
            fill_height=item["fill_height"],
            liquid_color=item["liquid_color"],
            marker_color=marker_color,
        )
        if pos < len(gaps):
            left += width + gaps[pos]

    detected = _find_containers(grid)
    if len(detected) != count:
        return None
    if {item["marker_color"] for item in detected} != {marker_color}:
        return None
    detected_fill_heights = sorted(item["fill_height"] for item in detected)
    if detected_fill_heights != sorted(fill_heights):
        return None

    output = main(grid)
    expected_w = count * width + (count - 1)
    if output.shape != (max_height, expected_w):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
