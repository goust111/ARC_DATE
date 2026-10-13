import math
import random
from collections import deque

import numpy as np


_GEN_RETRIES = 400
_GRID_MIN = 16
_GRID_MAX = 30
_TICK_SLOT_COUNT = 12


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _background_color(grid: np.ndarray) -> int:
    if grid.shape[0] == 1:
        border = grid[0]
    elif grid.shape[1] == 1:
        border = grid[:, 0]
    else:
        border = np.concatenate([grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]])
    values, counts = np.unique(border, return_counts=True)
    return int(values[np.argmax(counts)])


def _mode(values: np.ndarray) -> int:
    uniq, counts = np.unique(values, return_counts=True)
    return int(uniq[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(pts)
    return out


def _component_view(grid: np.ndarray, pts: list[tuple[int, int]]):
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        mask[r - r0, c - c0] = True
    return r0, c0, grid[r0 : r1 + 1, c0 : c1 + 1], mask


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    boundary = np.zeros_like(mask, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                boundary[r, c] = True
                continue
            for nr, nc in _neighbors4(r, c):
                if not mask[nr, nc]:
                    boundary[r, c] = True
                    break
    return boundary


def _angle(center: tuple[int, int], cell: tuple[int, int]) -> float:
    cr, cc = center
    r, c = cell
    return (math.atan2(r - cr, c - cc) + 2.0 * math.pi) % (2.0 * math.pi)


def _circular_distance(a: float, b: float) -> float:
    delta = abs(a - b) % (2.0 * math.pi)
    return min(delta, 2.0 * math.pi - delta)


def _mean_angle(angles: list[float]) -> float:
    s = sum(math.sin(a) for a in angles)
    c = sum(math.cos(a) for a in angles)
    return (math.atan2(s, c) + 2.0 * math.pi) % (2.0 * math.pi)


def _ordered_boundary_cells(mask: np.ndarray, center: tuple[int, int]) -> list[tuple[int, int]]:
    boundary = np.argwhere(_boundary_mask(mask))
    cells = [(int(r), int(c)) for r, c in boundary]
    return sorted(
        cells,
        key=lambda cell: (
            _angle(center, cell),
            (cell[0] - center[0]) ** 2 + (cell[1] - center[1]) ** 2,
            cell[0],
            cell[1],
        ),
    )


def _tick_anchor(boundary_cells: list[tuple[int, int]], center: tuple[int, int]) -> tuple[tuple[int, int], float]:
    angles = [_angle(center, cell) for cell in boundary_cells]
    mean = _mean_angle(angles)
    anchor = min(
        boundary_cells,
        key=lambda cell: (
            _circular_distance(_angle(center, cell), mean),
            (cell[0] - center[0]) ** 2 + (cell[1] - center[1]) ** 2,
            cell[0],
            cell[1],
        ),
    )
    return anchor, _angle(center, anchor)


def _bresenham_line(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    points: list[tuple[int, int]] = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr
    while True:
        points.append((r0, c0))
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c0 += sc
        if e2 < dc:
            err += dc
            r0 += sr
    return points


def _draw_pointer(patch: np.ndarray, mask: np.ndarray, center: tuple[int, int], anchor: tuple[int, int], color: int) -> None:
    for r, c in _bresenham_line(center, anchor):
        if mask[r, c]:
            patch[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each clock hand to the next clockwise border tick and recolor it to that tick.

    Input structure:
    - The background color surrounds several separated clock-like objects.
    - Each clock is a filled square or round dial whose outer boundary is mostly one border color.
    - The boundary also contains 2-4 differently colored tick marks, and the dial interior contains one straight pointer starting at the center.

    Rule:
    - For each clock, read the tick marks in clockwise order around the dial boundary.
    - The center pixel's color identifies which tick the current pointer is aimed at.
    - Remove the old pointer by restoring the dial's fill color, then draw a new straight pointer from the center to the next clockwise tick.
    - The new pointer uses the target tick's color, while the border and tick marks themselves stay unchanged.

    Key insight: The dial boundary gives a cyclic clockwise ordering of tick colors, and the hand color equals the currently selected tick color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    out = g.copy()

    for pts in _components(g != bg):
        r0, c0, sub, mask = _component_view(g, pts)
        center = (mask.shape[0] // 2, mask.shape[1] // 2)
        boundary = _boundary_mask(mask)
        interior = mask & ~boundary

        fill_color = _mode(sub[interior])
        border_color = _mode(sub[boundary])

        tick_colors = [int(color) for color in np.unique(sub[boundary]) if int(color) != border_color]
        assert 2 <= len(tick_colors) <= 4, "each dial must have 2-4 tick colors"

        tick_infos = []
        for color in tick_colors:
            cells = [(int(r), int(c)) for r, c in np.argwhere(boundary & (sub == color))]
            anchor, ang = _tick_anchor(cells, center)
            tick_infos.append({"color": color, "anchor": anchor, "angle": ang})
        tick_infos.sort(key=lambda item: item["angle"])

        current_color = int(sub[center])
        current_index = next(i for i, item in enumerate(tick_infos) if item["color"] == current_color)
        target = tick_infos[(current_index + 1) % len(tick_infos)]

        local_out = out[r0 : r0 + sub.shape[0], c0 : c0 + sub.shape[1]]
        local_out[interior] = fill_color
        _draw_pointer(local_out, mask, center, target["anchor"], target["color"])

    return out


def _square_mask(size: int) -> np.ndarray:
    return np.ones((size, size), dtype=bool)


def _circle_mask(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    center = radius
    mask = np.zeros((size, size), dtype=bool)
    limit = (radius + 0.35) ** 2
    for r in range(size):
        for c in range(size):
            if (r - center) ** 2 + (c - center) ** 2 <= limit:
                mask[r, c] = True
    return mask


def _slot_indices(boundary_len: int) -> list[int]:
    return [(i * boundary_len) // _TICK_SLOT_COUNT for i in range(_TICK_SLOT_COUNT)]


def _tick_segment(boundary_cells: list[tuple[int, int]], anchor_index: int, length: int) -> list[tuple[int, int]]:
    if length == 1:
        return [boundary_cells[anchor_index]]
    indices = [anchor_index - 1, anchor_index, anchor_index + 1]
    n = len(boundary_cells)
    return [boundary_cells[i % n] for i in indices]


def _build_clock_patch(rng: random.Random, bg: int):
    shape = rng.choice(["square", "circle"])
    if shape == "square":
        mask = _square_mask(rng.choice([7, 9, 11]))
    else:
        mask = _circle_mask(rng.choice([4, 5]))

    h, w = mask.shape
    center = (h // 2, w // 2)
    boundary = _boundary_mask(mask)
    boundary_cells = _ordered_boundary_cells(mask, center)
    if len(boundary_cells) < _TICK_SLOT_COUNT:
        return None

    colors = [color for color in range(10) if color != bg]
    rng.shuffle(colors)
    tick_count = rng.choice([2, 3, 4])
    fill_color = colors.pop()
    border_color = colors.pop()
    tick_colors = [colors.pop() for _ in range(tick_count)]
    segment_length = rng.choice([1, 3])

    base = np.full((h, w), bg, dtype=int)
    base[mask] = fill_color
    base[boundary] = border_color

    slot_ids = _slot_indices(len(boundary_cells))
    phase = rng.randrange(_TICK_SLOT_COUNT)
    step = _TICK_SLOT_COUNT // tick_count

    tick_infos = []
    for idx, color in enumerate(tick_colors):
        slot = (phase + idx * step) % _TICK_SLOT_COUNT
        anchor_index = slot_ids[slot]
        anchor = boundary_cells[anchor_index]
        for r, c in _tick_segment(boundary_cells, anchor_index, segment_length):
            base[r, c] = color
        tick_infos.append({"color": color, "anchor": anchor})

    current_index = rng.randrange(tick_count)
    next_index = (current_index + 1) % tick_count

    inp = base.copy()
    out = base.copy()
    _draw_pointer(inp, mask, center, tick_infos[current_index]["anchor"], tick_infos[current_index]["color"])
    _draw_pointer(out, mask, center, tick_infos[next_index]["anchor"], tick_infos[next_index]["color"])
    return inp, out


def _find_positions(occupied: np.ndarray, patch_shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = occupied.shape
    ph, pw = patch_shape
    choices: list[tuple[int, int]] = []
    for r0 in range(1, h - ph):
        for c0 in range(1, w - pw):
            if occupied[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1].any():
                continue
            choices.append((r0, c0))
    return choices


def _paste_non_bg(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    region = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
    mask = patch != bg
    region[mask] = patch[mask]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(_GRID_MIN, _GRID_MAX)
    w = rng.randint(_GRID_MIN, _GRID_MAX)
    bg = rng.randint(0, 9)
    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    max_clocks = 1
    if min(h, w) >= 18 and h * w >= 380:
        max_clocks = 2
    if min(h, w) >= 24 and h * w >= 560:
        max_clocks = 3
    num_clocks = rng.randint(1, max_clocks)

    for _ in range(num_clocks):
        built = _build_clock_patch(rng, bg)
        if built is None:
            return None
        patch_in, patch_out = built
        choices = _find_positions(occupied, patch_in.shape)
        if not choices:
            return None
        top, left = rng.choice(choices)
        _paste_non_bg(inp, patch_in, top, left, bg)
        _paste_non_bg(out, patch_out, top, left, bg)
        ph, pw = patch_in.shape
        occupied[top - 1 : top + ph + 1, left - 1 : left + pw + 1] = True

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid clock-rotation input")
