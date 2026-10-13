from __future__ import annotations

import random

import numpy as np


BG = 0
BRIDGE = 5
SHAPE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Restrict rectangle sizes to small, readable cases while still covering the rule.
RECT_DIMS = tuple(
    (height, width)
    for height in range(1, 5)
    for width in range(1, 5)
    if 4 <= height * width <= 12
)

GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 24)
GAP_RANGE = (1, 6)
GEN_ATTEMPTS = 400
PAIR_ATTEMPTS = 200


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _label_mask(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=int)
    components: list[np.ndarray] = []
    next_label = 0

    for row in range(height):
        for col in range(width):
            if not bool(mask[row, col]) or labels[row, col] != 0:
                continue

            next_label += 1
            stack = [(row, col)]
            labels[row, col] = next_label
            coords: list[tuple[int, int]] = []

            while stack:
                cur_row, cur_col = stack.pop()
                coords.append((cur_row, cur_col))
                for d_row, d_col in NEIGHBORS4:
                    nxt_row = cur_row + d_row
                    nxt_col = cur_col + d_col
                    if not (0 <= nxt_row < height and 0 <= nxt_col < width):
                        continue
                    if not bool(mask[nxt_row, nxt_col]) or labels[nxt_row, nxt_col] != 0:
                        continue
                    labels[nxt_row, nxt_col] = next_label
                    stack.append((nxt_row, nxt_col))

            components.append(np.array(coords, dtype=int))

    return labels, components


def _find_shape_components(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    shape_index = np.full(grid.shape, -1, dtype=int)
    components: list[dict] = []

    for color in np.unique(grid):
        color = int(color)
        if color in (BG, BRIDGE):
            continue

        _labels, color_components = _label_mask(grid == color)
        for coords in color_components:
            comp_id = len(components)
            r0, c0, r1, c1 = _bbox(coords)
            area = int(coords.shape[0])
            assert area == (r1 - r0 + 1) * (c1 - c0 + 1), "each shape must be a solid rectangle"
            shape_index[coords[:, 0], coords[:, 1]] = comp_id
            components.append(
                {
                    "id": comp_id,
                    "color": color,
                    "coords": coords,
                    "area": area,
                    "bbox": (r0, c0, r1, c1),
                }
            )

    return components, shape_index


def _bridge_contacts(coords: np.ndarray, shape_index: np.ndarray) -> dict[int, set[str]]:
    height, width = shape_index.shape
    contacts: dict[int, set[str]] = {}

    for row, col in coords:
        for d_row, d_col, side in (
            (-1, 0, "up"),
            (1, 0, "down"),
            (0, -1, "left"),
            (0, 1, "right"),
        ):
            nxt_row = int(row + d_row)
            nxt_col = int(col + d_col)
            if not (0 <= nxt_row < height and 0 <= nxt_col < width):
                continue
            comp_id = int(shape_index[nxt_row, nxt_col])
            if comp_id == -1:
                continue
            contacts.setdefault(comp_id, set()).add(side)

    return contacts


def _winner_loser(first: dict, second: dict, bridge_width: int) -> tuple[dict, dict]:
    area_first = int(first["area"])
    area_second = int(second["area"])
    assert area_first != area_second, "paired shapes must have different areas"

    if bridge_width % 2 == 1:
        return (first, second) if area_first > area_second else (second, first)
    return (first, second) if area_first < area_second else (second, first)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each bridged pair by the bridge-width-dependent area comparison rule.

    Odd-width bridges let the larger rectangle win; even-width bridges let the
    smaller rectangle win. The losing rectangle and the grey bridge both adopt
    the winner's color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    shape_components, shape_index = _find_shape_components(g)
    assert shape_components, "expected at least one colored rectangle"

    _bridge_labels, bridge_components = _label_mask(g == BRIDGE)
    assert bridge_components, "expected at least one grey bridge"

    out = g.copy()
    for coords in bridge_components:
        r0, c0, r1, c1 = _bbox(coords)
        contacts = _bridge_contacts(coords, shape_index)
        assert len(contacts) == 2, "each bridge must connect exactly two shapes"

        sides_by_component = {comp_id: frozenset(sides) for comp_id, sides in contacts.items()}
        assert all(len(sides) == 1 for sides in sides_by_component.values()), "bridge contacts must be unambiguous"
        side_set = {next(iter(sides)) for sides in sides_by_component.values()}

        if side_set == {"left", "right"}:
            bridge_width = r1 - r0 + 1
        elif side_set == {"up", "down"}:
            bridge_width = c1 - c0 + 1
        else:
            raise AssertionError("bridge must connect shapes across opposite sides")

        first_id, second_id = sorted(contacts)
        winner, loser = _winner_loser(shape_components[first_id], shape_components[second_id], bridge_width)

        loser_coords = loser["coords"]
        out[loser_coords[:, 0], loser_coords[:, 1]] = int(winner["color"])
        out[coords[:, 0], coords[:, 1]] = int(winner["color"])

    return out


def _trim_patch(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(inp != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return inp[r0 : r1 + 1, c0 : c1 + 1], out[r0 : r1 + 1, c0 : c1 + 1]


def _build_horizontal_pair(bridge_width: int) -> tuple[np.ndarray, np.ndarray] | None:
    candidates = [dims for dims in RECT_DIMS if dims[0] >= bridge_width]
    if not candidates:
        return None

    for _ in range(PAIR_ATTEMPTS):
        left_h, left_w = random.choice(candidates)
        right_h, right_w = random.choice(candidates)
        left_area = left_h * left_w
        right_area = right_h * right_w
        if left_area == right_area:
            continue

        colors = random.sample(SHAPE_COLORS, 2)
        gap = random.randint(*GAP_RANGE)
        offset = random.randint(-(right_h - bridge_width), left_h - bridge_width)

        top_left = max(0, -offset)
        top_right = max(0, offset)
        patch_h = max(top_left + left_h, top_right + right_h)
        patch_w = left_w + gap + right_w

        overlap_top = max(top_left, top_right)
        overlap_bottom = min(top_left + left_h, top_right + right_h)
        bridge_top = random.randint(overlap_top, overlap_bottom - bridge_width)

        inp = np.full((patch_h, patch_w), BG, dtype=int)
        inp[top_left : top_left + left_h, :left_w] = colors[0]
        inp[top_right : top_right + right_h, left_w + gap : left_w + gap + right_w] = colors[1]
        inp[bridge_top : bridge_top + bridge_width, left_w : left_w + gap] = BRIDGE

        out = inp.copy()
        winner_color = colors[0] if _winner_loser({"area": left_area}, {"area": right_area}, bridge_width)[0]["area"] == left_area else colors[1]
        if winner_color == colors[0]:
            out[top_right : top_right + right_h, left_w + gap : left_w + gap + right_w] = colors[0]
        else:
            out[top_left : top_left + left_h, :left_w] = colors[1]
        out[bridge_top : bridge_top + bridge_width, left_w : left_w + gap] = winner_color
        return _trim_patch(inp, out)

    return None


def _build_vertical_pair(bridge_width: int) -> tuple[np.ndarray, np.ndarray] | None:
    candidates = [dims for dims in RECT_DIMS if dims[1] >= bridge_width]
    if not candidates:
        return None

    for _ in range(PAIR_ATTEMPTS):
        top_h, top_w = random.choice(candidates)
        bottom_h, bottom_w = random.choice(candidates)
        top_area = top_h * top_w
        bottom_area = bottom_h * bottom_w
        if top_area == bottom_area:
            continue

        colors = random.sample(SHAPE_COLORS, 2)
        gap = random.randint(*GAP_RANGE)
        offset = random.randint(-(bottom_w - bridge_width), top_w - bridge_width)

        left_top = max(0, -offset)
        left_bottom = max(0, offset)
        patch_h = top_h + gap + bottom_h
        patch_w = max(left_top + top_w, left_bottom + bottom_w)

        overlap_left = max(left_top, left_bottom)
        overlap_right = min(left_top + top_w, left_bottom + bottom_w)
        bridge_left = random.randint(overlap_left, overlap_right - bridge_width)

        inp = np.full((patch_h, patch_w), BG, dtype=int)
        inp[:top_h, left_top : left_top + top_w] = colors[0]
        inp[top_h + gap : top_h + gap + bottom_h, left_bottom : left_bottom + bottom_w] = colors[1]
        inp[top_h : top_h + gap, bridge_left : bridge_left + bridge_width] = BRIDGE

        out = inp.copy()
        winner_color = colors[0] if _winner_loser({"area": top_area}, {"area": bottom_area}, bridge_width)[0]["area"] == top_area else colors[1]
        if winner_color == colors[0]:
            out[top_h + gap : top_h + gap + bottom_h, left_bottom : left_bottom + bottom_w] = colors[0]
        else:
            out[:top_h, left_top : left_top + top_w] = colors[1]
        out[top_h : top_h + gap, bridge_left : bridge_left + bridge_width] = winner_color
        return _trim_patch(inp, out)

    return None


def _build_pair(bridge_width: int) -> tuple[np.ndarray, np.ndarray] | None:
    if random.random() < 0.5:
        pair = _build_horizontal_pair(bridge_width)
        if pair is not None:
            return pair
        return _build_vertical_pair(bridge_width)

    pair = _build_vertical_pair(bridge_width)
    if pair is not None:
        return pair
    return _build_horizontal_pair(bridge_width)


def _parities(pair_count: int) -> list[int]:
    parities = [0, 1] * (pair_count // 2)
    if pair_count % 2 == 1:
        parities.append(random.randint(0, 1))
    random.shuffle(parities)
    return parities


def _sample_pair_count(height: int, width: int) -> int:
    area = height * width
    short_side = min(height, width)
    if short_side < 15 or area < 220:
        return random.randint(1, 2)
    if short_side < 18 or area < 320:
        return random.randint(1, 3)
    return random.randint(1, 4)


def _paste_patch(canvas: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    mask = patch != BG
    canvas_view = canvas[top : top + patch.shape[0], left : left + patch.shape[1]]
    canvas_view[mask] = patch[mask]


def _place_patch(
    canvas: np.ndarray,
    expected: np.ndarray,
    occupied: np.ndarray,
    patch_in: np.ndarray,
    patch_out: np.ndarray,
) -> bool:
    height, width = canvas.shape
    patch_h, patch_w = patch_in.shape
    positions = [
        (row, col)
        for row in range(height - patch_h + 1)
        for col in range(width - patch_w + 1)
    ]
    random.shuffle(positions)

    for row, col in positions:
        r0 = max(0, row - 1)
        c0 = max(0, col - 1)
        r1 = min(height, row + patch_h + 1)
        c1 = min(width, col + patch_w + 1)
        if bool(occupied[r0:r1, c0:c1].any()):
            continue

        _paste_patch(canvas, patch_in, row, col)
        _paste_patch(expected, patch_out, row, col)
        occupied[r0:r1, c0:c1] = True
        return True

    return False


def _attempt_generate_input() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    pair_count = _sample_pair_count(height, width)

    bridge_widths = [
        2 if parity == 0 else random.choice((1, 3))
        for parity in _parities(pair_count)
    ]

    patches: list[tuple[np.ndarray, np.ndarray]] = []
    for bridge_width in bridge_widths:
        pair = _build_pair(bridge_width)
        if pair is None:
            return None
        patches.append(pair)

    random.shuffle(patches)
    grid = np.full((height, width), BG, dtype=int)
    expected = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    for patch_in, patch_out in patches:
        if patch_in.shape[0] > height or patch_in.shape[1] > width:
            return None
        if not _place_patch(grid, expected, occupied, patch_in, patch_out):
            return None

    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate_input()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
