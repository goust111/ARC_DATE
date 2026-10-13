from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

GEN_ATTEMPTS = 400
MASK_ATTEMPTS = 240
SIZE_CHOICES = (3, 4, 5)
COUNT_CHOICES = (3, 4, 5)
GAP_RANGE = (1, 3)
TOP_BOTTOM_MARGIN_RANGE = (0, 3)
WIDTH_EXTRA_MAX = 4
ROW_SEPARATION_RANGE = (2, 5)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_components(
    grid: np.ndarray,
    *,
    bg: int,
    connectivity: int = 8,
) -> list[dict]:
    neighbors = NEIGHBORS8 if connectivity == 8 else NEIGHBORS4
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    out: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or visited[r, c]:
                continue

            visited[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in neighbors:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            out.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )

    return out


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in component["coords"]:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _copy_starts(positions: list[int], anchor_index: int, anchor_start: int, sprite_len: int) -> list[int]:
    starts: list[int | None] = [None] * len(positions)
    starts[anchor_index] = int(anchor_start)

    for i in range(anchor_index + 1, len(positions)):
        gap = positions[i] - positions[i - 1] - 1
        assert gap >= 0
        assert starts[i - 1] is not None
        starts[i] = starts[i - 1] + sprite_len + gap

    for i in range(anchor_index - 1, -1, -1):
        gap = positions[i + 1] - positions[i] - 1
        assert gap >= 0
        assert starts[i + 1] is not None
        starts[i] = starts[i + 1] - sprite_len - gap

    return [int(start) for start in starts]


def _rotated_mask(mask: np.ndarray, signed_steps: int) -> np.ndarray:
    return np.rot90(mask, -int(signed_steps))


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _get_bg(g)
    components = _find_components(g, bg=bg, connectivity=8)
    sprite_components = [component for component in components if component["area"] > 1]
    assert len(sprite_components) == 1, "expected one non-singleton template sprite"

    sprite = sprite_components[0]
    template_color = int(sprite["color"])
    sr0, sc0, sr1, sc1 = sprite["bbox"]
    assert sr1 - sr0 == sc1 - sc0, "template bbox must be square"
    template = _component_mask(sprite)
    sprite_size = template.shape[0]
    assert template.shape == (sprite_size, sprite_size)

    controls: list[tuple[int, int, int]] = []
    for component in components:
        if component is sprite:
            continue
        assert component["area"] == 1, "all non-template components must be singleton controls"
        r, c = map(int, component["coords"][0])
        controls.append((r, c, int(component["color"])))

    assert len(controls) >= 2, "expected a control line"
    control_rows = {r for r, _, _ in controls}
    control_cols = {c for _, c, _ in controls}
    anchor_count = sum(color == template_color for _, _, color in controls)
    assert anchor_count == 1, "expected exactly one template-colored control pixel"

    placements: list[tuple[int, int, int, int]] = []
    if len(control_rows) == 1:
        controls.sort(key=lambda item: item[1])
        positions = [c for _, c, _ in controls]
        anchor_index = next(i for i, (_, _, color) in enumerate(controls) if color == template_color)
        starts = _copy_starts(positions, anchor_index, sc0, sprite_size)
        placements = [(sr0, start, color, i - anchor_index) for i, (start, (_, _, color)) in enumerate(zip(starts, controls))]
    elif len(control_cols) == 1:
        controls.sort(key=lambda item: item[0])
        positions = [r for r, _, _ in controls]
        anchor_index = next(i for i, (_, _, color) in enumerate(controls) if color == template_color)
        starts = _copy_starts(positions, anchor_index, sr0, sprite_size)
        placements = [(start, sc0, color, i - anchor_index) for i, (start, (_, _, color)) in enumerate(zip(starts, controls))]
    else:
        raise AssertionError("control pixels must share one row or one column")

    out = g.copy()
    h, w = g.shape
    for r0, c0, color, signed_steps in placements:
        rotated = _rotated_mask(template, signed_steps)
        assert 0 <= r0 and 0 <= c0 and r0 + sprite_size <= h and c0 + sprite_size <= w
        target = out[r0 : r0 + sprite_size, c0 : c0 + sprite_size]
        if signed_steps == 0:
            assert np.all(target[rotated] == template_color)
        else:
            assert np.all(target[rotated] == bg), "new copies must land on background"
        target[rotated] = int(color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the template along the control-pixel line and rotate each copy by its offset phase from the anchor.

    Input structure:
    - The grid has a black background, one monochrome template sprite, and a row of isolated control pixels.
    - The template sprite has a square bounding box and is not invariant under quarter-turn rotation.
    - Exactly one control pixel shares the template color; that control is the anchor, and the other control pixels may have different colors.

    Rule:
    - Read the control pixels in order along the line.
    - The anchor corresponds to the original template in place; every other control pixel creates one same-bbox copy colored with that control color.
    - The empty cells between adjacent control pixels become the empty columns or rows between adjacent copy bounding boxes.
    - A copy k positions to the right of the anchor rotates clockwise by 90 degrees times k, while a copy k positions to the left rotates counterclockwise by 90 degrees times k.

    Key insight:
    - The control line encodes two things at once: spacing between template slots and the rotation phase determined by each control pixel's signed index offset from the anchor.
    """
    return _solve(grid)


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _random_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    tr, tc = end
    path = [(r, c)]
    while (r, c) != (tr, tc):
        dr = _sign(tr - r)
        dc = _sign(tc - c)
        choices: list[tuple[int, int]] = []
        if dr != 0 and dc != 0:
            choices.extend([(r + dr, c + dc), (r + dr, c), (r, c + dc)])
        elif dr != 0:
            choices.append((r + dr, c))
        else:
            choices.append((r, c + dc))
        r, c = random.choice(choices)
        path.append((r, c))
    return path


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _non_symmetric(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _grow_connected_mask(size: int) -> np.ndarray:
    for _ in range(MASK_ATTEMPTS):
        center = (random.randrange(size), random.randrange(size))
        terminals = [
            (0, random.randrange(size)),
            (size - 1, random.randrange(size)),
            (random.randrange(size), 0),
            (random.randrange(size), size - 1),
        ]

        cells: set[tuple[int, int]] = {center}
        for terminal in random.sample(terminals, len(terminals)):
            cells.update(_random_path(center, terminal))

        target_min = max(len(cells), size + 2)
        target_cap = min(size * size - 1, max(target_min, int(round(size * size * 0.72))))
        if target_min > target_cap:
            continue
        target_area = random.randint(target_min, target_cap)

        while len(cells) < target_area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in NEIGHBORS8:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))

        mask = np.zeros((size, size), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if not _touches_all_sides(mask):
            continue
        if not _non_symmetric(mask):
            continue
        return np.rot90(mask, random.randrange(4))

    raise RuntimeError("failed to generate a non-symmetric square template")


def _choose_anchor_index(count: int) -> int:
    if count >= 3 and random.random() < 0.8:
        return random.randint(1, count - 2)
    return random.randrange(count)


def _starts_from_gaps(start: int, size: int, gaps: list[int]) -> list[int]:
    starts = [int(start)]
    for gap in gaps:
        starts.append(starts[-1] + size + int(gap))
    return starts


def _positions_from_gaps(start: int, gaps: list[int]) -> list[int]:
    positions = [int(start)]
    for gap in gaps:
        positions.append(positions[-1] + 1 + int(gap))
    return positions


def _render_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    size = random.choice(SIZE_CHOICES)
    count = random.choice(COUNT_CHOICES)
    gaps = [random.randint(*GAP_RANGE) for _ in range(count - 1)]
    copy_span = count * size + sum(gaps)
    if copy_span > 28:
        return None

    width = copy_span + random.randint(0, min(WIDTH_EXTRA_MAX, 30 - copy_span))
    copy_left = random.randint(0, width - copy_span)
    starts = _starts_from_gaps(copy_left, size, gaps)

    control_span = count + sum(gaps)
    control_left = random.randint(0, width - control_span)
    control_positions = _positions_from_gaps(control_left, gaps)

    top_margin = random.randint(*TOP_BOTTOM_MARGIN_RANGE)
    bottom_margin = random.randint(*TOP_BOTTOM_MARGIN_RANGE)
    blank_rows = random.randint(*ROW_SEPARATION_RANGE)
    control_above = random.random() < 0.5
    if control_above:
        control_row = top_margin
        sprite_top = control_row + blank_rows + 1
        height = sprite_top + size + bottom_margin
    else:
        sprite_top = top_margin
        control_row = sprite_top + size + blank_rows
        height = control_row + 1 + bottom_margin
    if height > 30:
        return None

    anchor_index = _choose_anchor_index(count)
    template_color = random.choice(PALETTE)
    other_colors = random.sample([c for c in PALETTE if c != template_color], count - 1)
    colors: list[int] = []
    other_i = 0
    for i in range(count):
        if i == anchor_index:
            colors.append(template_color)
        else:
            colors.append(other_colors[other_i])
            other_i += 1

    mask = _grow_connected_mask(size)
    out = np.zeros((height, width), dtype=int)
    for i, (start, color) in enumerate(zip(starts, colors)):
        rotated = _rotated_mask(mask, i - anchor_index)
        window = out[sprite_top : sprite_top + size, start : start + size]
        assert np.all(window[rotated] == BACKGROUND)
        window[rotated] = int(color)

    for col, color in zip(control_positions, colors):
        out[control_row, col] = int(color)

    inp = out.copy()
    for i, start in enumerate(starts):
        if i == anchor_index:
            continue
        rotated = _rotated_mask(mask, i - anchor_index)
        window = inp[sprite_top : sprite_top + size, start : start + size]
        window[rotated] = BACKGROUND

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rendered = _render_attempt()
        if rendered is None:
            continue
        inp, out = rendered
        try:
            solved = main(inp)
        except Exception:
            continue
        if np.array_equal(solved, out):
            return inp
    raise RuntimeError("failed to generate a valid phase-rotation control-line input")
