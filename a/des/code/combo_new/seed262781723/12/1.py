import random
from collections import Counter, deque

import numpy as np


BG = 0
MAX_RETRIES = 300
GRID_MIN = 18
GRID_MAX = 30
LEGEND_LEN_OPTIONS = (4, 5)
LEGEND_WIDTH_OPTIONS = (1, 2)
OBJECT_COUNT_RANGE = (2, 4)
OBJECT_PLACE_TRIES = 200


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _background_color(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    bg, count = Counter(corners).most_common(1)[0]
    if count >= 3:
        return int(bg)
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _find_legend(grid: np.ndarray, bg: int) -> tuple[list[int], np.ndarray]:
    h, w = grid.shape
    candidates = []
    for coords in _components(grid != bg):
        r0, c0, r1, c1 = _bbox(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        touches_corner = (
            (r0 == 0 and c0 == 0)
            or (r0 == 0 and c1 == w - 1)
            or (r1 == h - 1 and c0 == 0)
            or (r1 == h - 1 and c1 == w - 1)
        )
        if not touches_corner or height < 2 or height < width or width > 3:
            continue

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(patch == bg):
            continue
        if not all(np.all(patch[row] == patch[row, 0]) for row in range(height)):
            continue

        row_colors = [int(patch[row, 0]) for row in range(height)]
        if len(set(row_colors)) != height:
            continue

        candidates.append(((height, -width, -(height * width)), row_colors, coords))

    assert candidates, "expected a corner legend"
    _, row_colors, coords = max(candidates, key=lambda item: item[0])
    legend_mask = np.zeros_like(grid, dtype=bool)
    for r, c in coords:
        legend_mask[r, c] = True
    return row_colors, legend_mask


def _rectangle_object_info(
    grid: np.ndarray, bg: int, coords: list[tuple[int, int]]
) -> tuple[tuple[int, int, int, int], int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if int(np.count_nonzero(patch != bg)) != len(coords):
        return None

    border_mask = np.zeros_like(patch, dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    border_values = patch[border_mask]
    if np.any(border_values == bg):
        return None

    border_color = int(border_values[0])
    if np.any(border_values != border_color):
        return None

    interior = patch[1:-1, 1:-1]
    interior_colors = set(map(int, np.unique(interior))) - {bg}
    if len(interior_colors) > 1:
        return None

    hole_count = len(_components(interior == bg))
    return (r0, c0, r1, c1), hole_count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each closed rectangle border by indexing a corner legend with its hole count.

    Input structure:
    - The background is black and one corner contains a vertical color legend made of solid row blocks.
    - Elsewhere, several non-overlapping rectangular regions have a single-color one-pixel outline and a uniform interior fill.
    - Some interior cells are black holes; each hole is a separate enclosed background component inside the rectangle.

    Rule:
    - Read the legend from top to bottom so row 0 corresponds to rectangles with 0 holes, row 1 to 1 hole, and so on.
    - For each closed rectangle, count the number of black hole components inside its border.
    - Recolor only the rectangle's outline with the legend color at that hole-count index.
    - Keep the fill, the holes, and the legend itself unchanged.

    Key insight:
    The holes are the hidden lookup key: counting enclosed black components tells which legend row supplies the border color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    legend_colors, legend_mask = _find_legend(g, bg)

    out = g.copy()
    changed = 0
    for coords in _components(g != bg):
        if all(legend_mask[r, c] for r, c in coords):
            continue
        info = _rectangle_object_info(g, bg, coords)
        if info is None:
            continue
        (r0, c0, r1, c1), hole_count = info
        assert hole_count < len(legend_colors), "hole count exceeds legend"
        color = int(legend_colors[hole_count])
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
        changed += 1

    assert changed > 0, "expected at least one rectangle object"
    return out


def _hole_shape_choices(hole_count: int) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for inner_h in range(3, 8):
        for inner_w in range(3, 8):
            slot_count = ((inner_h + 1) // 2) * ((inner_w + 1) // 2)
            if slot_count < hole_count:
                continue
            if inner_h * inner_w <= hole_count:
                continue
            shapes.append((inner_h + 2, inner_w + 2))
    return shapes


def _hole_slots(inner_h: int, inner_w: int, hole_count: int, rng: random.Random) -> list[tuple[int, int]]:
    if hole_count == 0:
        return []
    offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]
    rng.shuffle(offsets)
    for row_offset, col_offset in offsets:
        slots = [(r, c) for r in range(row_offset, inner_h, 2) for c in range(col_offset, inner_w, 2)]
        if len(slots) >= hole_count:
            rng.shuffle(slots)
            return slots[:hole_count]
    raise RuntimeError("failed to find hole slots")


def _make_object_patch(
    height: int, width: int, border_color: int, fill_color: int, hole_count: int, rng: random.Random
) -> np.ndarray:
    patch = np.full((height, width), fill_color, dtype=int)
    patch[0, :] = border_color
    patch[-1, :] = border_color
    patch[:, 0] = border_color
    patch[:, -1] = border_color

    inner_h = height - 2
    inner_w = width - 2
    for r, c in _hole_slots(inner_h, inner_w, hole_count, rng):
        patch[r + 1, c + 1] = BG
    return patch


def _legend_position(corner: str, grid_h: int, grid_w: int, legend_h: int, legend_w: int) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, grid_w - legend_w
    if corner == "bl":
        return grid_h - legend_h, 0
    return grid_h - legend_h, grid_w - legend_w


def _reserve(mask: np.ndarray, r0: int, c0: int, height: int, width: int, pad: int = 1) -> None:
    h, w = mask.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r0 + height + pad)
    cc1 = min(w, c0 + width + pad)
    mask[rr0:rr1, cc0:cc1] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(GRID_MIN, GRID_MAX)
    grid_w = rng.randint(GRID_MIN, GRID_MAX)
    legend_len = rng.choice(LEGEND_LEN_OPTIONS)
    legend_w = rng.choice(LEGEND_WIDTH_OPTIONS)
    if grid_h < legend_len + 6 or grid_w < legend_w + 8:
        return None

    colors = list(range(1, 10))
    border_color = rng.choice(colors)
    colors.remove(border_color)
    fill_color = rng.choice(colors)
    colors.remove(fill_color)
    if len(colors) < legend_len:
        return None
    legend_colors = rng.sample(colors, legend_len)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    corner = rng.choice(["tl", "tr", "bl", "br"])
    legend_r0, legend_c0 = _legend_position(corner, grid_h, grid_w, legend_len, legend_w)
    for idx, color in enumerate(legend_colors):
        grid[legend_r0 + idx, legend_c0 : legend_c0 + legend_w] = int(color)
    _reserve(occupied, legend_r0, legend_c0, legend_len, legend_w, pad=1)

    max_objects = OBJECT_COUNT_RANGE[1]
    if grid_h * grid_w < 480:
        max_objects = 3
    object_count = rng.randint(OBJECT_COUNT_RANGE[0], max_objects)
    hole_counts = rng.sample(range(legend_len), object_count)

    object_specs = []
    for hole_count in hole_counts:
        height, width = rng.choice(_hole_shape_choices(hole_count))
        patch = _make_object_patch(height, width, border_color, fill_color, hole_count, rng)
        object_specs.append((height * width, hole_count, patch))
    object_specs.sort(reverse=True)

    for _, hole_count, patch in object_specs:
        height, width = patch.shape
        if grid_h - height - 1 < 1 or grid_w - width - 1 < 1:
            return None

        placed = False
        for _ in range(OBJECT_PLACE_TRIES):
            r0 = rng.randint(1, grid_h - height - 1)
            c0 = rng.randint(1, grid_w - width - 1)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(grid_h, r0 + height + 1)
            cc1 = min(grid_w, c0 + width + 1)
            if np.any(occupied[rr0:rr1, cc0:cc1]):
                continue
            grid[r0 : r0 + height, c0 : c0 + width] = patch
            _reserve(occupied, r0, c0, height, width, pad=1)
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_RETRIES} attempts")
