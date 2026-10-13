from __future__ import annotations

import random

import numpy as np


BG = 8
MARKER = 5
TEMPLATE = 9
FRAME_COLORS = [0, 1, 2, 3, 4, 6, 7]
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
FRAME_SIZE_CHOICES = (5, 7, 9)
CORNERS = ("tl", "tr", "bl", "br")
DIRECTIONS = ("up", "down", "left", "right")
SLOT_SIZE = 5
GEN_RETRIES = 400


_BASE_UP = np.array(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
    ],
    dtype=bool,
)

ARROW_MASKS = {
    "up": _BASE_UP,
    "right": np.rot90(_BASE_UP, 3),
    "down": np.rot90(_BASE_UP, 2),
    "left": np.rot90(_BASE_UP, 1),
}


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(pts, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _arrow_direction(mask: np.ndarray) -> str:
    tight = _tight_mask(mask)
    h, w = tight.shape
    if h > w:
        row_sums = tight.sum(axis=1)
        peak = int(np.argmax(row_sums))
        return "up" if peak < h // 2 else "down"
    col_sums = tight.sum(axis=0)
    peak = int(np.argmax(col_sums))
    return "left" if peak < w // 2 else "right"


def _extract_templates(grid: np.ndarray) -> dict[str, np.ndarray]:
    comps = _components(grid == TEMPLATE)
    templates: dict[str, np.ndarray] = {}
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            mask[int(r - r0), int(c - c0)] = True
        direction = _arrow_direction(mask)
        templates[direction] = mask
    assert set(templates) == set(DIRECTIONS), "expected exactly four arrow templates"
    return templates


def _frame_direction(local_marker: tuple[int, int], size: int) -> str | None:
    mid = size // 2
    mapping = {
        (1, mid): "up",
        (size - 2, mid): "down",
        (mid, 1): "left",
        (mid, size - 2): "right",
    }
    return mapping.get(local_marker)


def _find_frames(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int, str]]:
    frames: list[tuple[int, int, int, int, str]] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (bg, MARKER, TEMPLATE):
            continue
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            size = int(r1 - r0 + 1)
            if size != int(c1 - c0 + 1) or size < 5:
                continue
            if int(comp.shape[0]) != 4 * size - 4:
                continue

            local = np.zeros((size, size), dtype=bool)
            for r, c in comp:
                local[int(r - r0), int(c - c0)] = True
            border = np.zeros((size, size), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.array_equal(local, border):
                continue

            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            interior = sub[1:-1, 1:-1]
            marker_coords = np.argwhere(interior == MARKER)
            if marker_coords.shape[0] != 1:
                continue
            if int(np.count_nonzero(interior != bg)) != 1:
                continue

            mr, mc = map(int, marker_coords[0])
            direction = _frame_direction((mr + 1, mc + 1), size)
            if direction is None:
                continue
            frames.append((color, r0, c0, size, direction))

    assert frames, "expected at least one colored frame"
    return frames


def _paint_arrow(
    out: np.ndarray,
    r0: int,
    c0: int,
    size: int,
    color: int,
    mask: np.ndarray,
    bg: int,
) -> None:
    h, w = mask.shape
    out[r0 : r0 + size, c0 : c0 + size] = bg
    rr = r0 + (size - h) // 2
    cc = c0 + (size - w) // 2
    out[rr : rr + h, cc : cc + w][mask] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each hollow frame with the direction-matched corner arrow template in the frame's own color.

    Input structure:
    - The background is azure and the grid contains several same-sized hollow square frames with one-pixel borders.
    - Each frame uses a distinct border color and contains exactly one grey marker pixel on one interior edge midpoint.
    - One corner region contains four maroon arrow templates, separated by background, covering the four cardinal directions.

    Rule:
    - Locate the four maroon templates and infer which one points up, down, left, and right from its shape.
    - For each colored frame, read the grey marker position: top, bottom, left, or right interior midpoint selects the matching arrow direction.
    - Remove the corner templates, erase the frame region, and draw the selected arrow template centered inside that square using the frame's border color.

    Key insight:
    - The grey marker does not move anything directly; it names a direction, while the corner maroon arrows act as a reusable direction-to-shape dictionary.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    templates = _extract_templates(g)
    frames = _find_frames(g, bg)

    out = g.copy()
    out[out == TEMPLATE] = bg
    for color, r0, c0, size, direction in frames:
        _paint_arrow(out, r0, c0, size, color, templates[direction], bg)
    return out


def _template_region_bounds(h: int, w: int, corner: str, gap: int) -> tuple[int, int, int, int]:
    size = 2 * SLOT_SIZE + gap
    if corner == "tl":
        return 0, 0, size, size
    if corner == "tr":
        return 0, w - size, size, size
    if corner == "bl":
        return h - size, 0, size, size
    return h - size, w - size, size, size


def _render_templates(
    grid: np.ndarray,
    corner: str,
    gap: int,
    assignment: dict[tuple[int, int], str],
) -> None:
    r0, c0, region_h, region_w = _template_region_bounds(grid.shape[0], grid.shape[1], corner, gap)
    del region_h, region_w
    for sr in range(2):
        for sc in range(2):
            direction = assignment[(sr, sc)]
            mask = ARROW_MASKS[direction]
            ah, aw = mask.shape
            cell_r = r0 + sr * (SLOT_SIZE + gap)
            cell_c = c0 + sc * (SLOT_SIZE + gap)
            off_r = (SLOT_SIZE - ah) // 2
            off_c = (SLOT_SIZE - aw) // 2
            grid[cell_r + off_r : cell_r + off_r + ah, cell_c + off_c : cell_c + off_c + aw][mask] = TEMPLATE


def _blocked_ok(blocked: np.ndarray, r0: int, c0: int, size: int) -> bool:
    h, w = blocked.shape
    if r0 < 1 or c0 < 1 or r0 + size >= h - 1 or c0 + size >= w - 1:
        return False
    return not bool(blocked[r0 - 1 : r0 + size + 1, c0 - 1 : c0 + size + 1].any())


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, size: int) -> None:
    blocked[r0 - 1 : r0 + size + 1, c0 - 1 : c0 + size + 1] = True


def _draw_frame(grid: np.ndarray, r0: int, c0: int, size: int, color: int, direction: str) -> None:
    r1 = r0 + size
    c1 = c0 + size
    grid[r0:r1, c0] = color
    grid[r0:r1, c1 - 1] = color
    grid[r0, c0:c1] = color
    grid[r1 - 1, c0:c1] = color
    mid = size // 2
    marker_pos = {
        "up": (r0 + 1, c0 + mid),
        "down": (r0 + size - 2, c0 + mid),
        "left": (r0 + mid, c0 + 1),
        "right": (r0 + mid, c0 + size - 2),
    }[direction]
    grid[marker_pos] = MARKER


def _sample_case() -> tuple[np.ndarray, np.ndarray] | None:
    size = random.choices(FRAME_SIZE_CHOICES, weights=[4, 4, 2], k=1)[0]
    gap = random.choice([1, 2])
    corner = random.choice(CORNERS)
    frame_count_max = 6 if size == 5 else (5 if size == 7 else 4)
    frame_count = random.randint(4, frame_count_max)

    if size == 5:
        h = random.randint(18, 30)
        w = random.randint(18, 30)
    elif size == 7:
        h = random.randint(20, 30)
        w = random.randint(20, 30)
    else:
        h = random.randint(24, 30)
        w = random.randint(24, 30)

    region_r0, region_c0, region_h, region_w = _template_region_bounds(h, w, corner, gap)
    grid = np.full((h, w), BG, dtype=int)
    _render_templates(
        grid,
        corner,
        gap,
        {
            slot: direction
            for slot, direction in zip(
                [(0, 0), (0, 1), (1, 0), (1, 1)],
                random.sample(list(DIRECTIONS), k=4),
                strict=False,
            )
        },
    )

    blocked = np.zeros((h, w), dtype=bool)
    r_pad0 = max(0, region_r0 - 1)
    c_pad0 = max(0, region_c0 - 1)
    r_pad1 = min(h, region_r0 + region_h + 1)
    c_pad1 = min(w, region_c0 + region_w + 1)
    blocked[r_pad0:r_pad1, c_pad0:c_pad1] = True

    directions = list(DIRECTIONS)
    while len(directions) < frame_count:
        directions.append(random.choice(DIRECTIONS))
    random.shuffle(directions)
    colors = random.sample(FRAME_COLORS, k=frame_count)

    frames: list[tuple[int, int, int, int, str]] = []
    order = list(range(frame_count))
    random.shuffle(order)
    for idx in order:
        placed = False
        for _ in range(150):
            r0 = random.randint(1, h - size - 1)
            c0 = random.randint(1, w - size - 1)
            if not _blocked_ok(blocked, r0, c0, size):
                continue
            _mark_blocked(blocked, r0, c0, size)
            color = colors[idx]
            direction = directions[idx]
            _draw_frame(grid, r0, c0, size, color, direction)
            frames.append((color, r0, c0, size, direction))
            placed = True
            break
        if not placed:
            return None

    expected = np.array(grid, copy=True)
    expected[expected == TEMPLATE] = BG
    for color, r0, c0, frame_size, direction in frames:
        _paint_arrow(expected, r0, c0, frame_size, color, ARROW_MASKS[direction], BG)

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        case = _sample_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid input")
