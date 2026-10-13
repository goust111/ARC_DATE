from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
MARKER_COLORS = (2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "right", "bottom", "left")
SIDE_TO_ROT_K = {"top": 0, "right": 3, "bottom": 2, "left": 1}
QUADRANT_CORNERS = {
    0: "upper-left",
    1: "upper-right",
    2: "lower-left",
    3: "lower-right",
}


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected at least one foreground pixel"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _quadrant_of_local(r: int, c: int, h: int, w: int) -> int | None:
    row_mid = h - 1
    col_mid = w - 1
    if 2 * r == row_mid or 2 * c == col_mid:
        return None
    top = 2 * r < row_mid
    left = 2 * c < col_mid
    if top and left:
        return 0
    if top and not left:
        return 1
    if not top and left:
        return 2
    return 3


def _rotate_local_coords(coords: np.ndarray, shape: tuple[int, int], rot_k: int) -> np.ndarray:
    coords = np.asarray(coords, dtype=int)
    if coords.size == 0 or rot_k % 4 == 0:
        return coords.copy()

    h, w = map(int, shape)
    assert rot_k in (1, 2, 3)
    if rot_k % 2 == 1:
        assert h % 2 == w % 2, "90-degree rotation needs matching row/column parity"

    row_mid = h - 1
    col_mid = w - 1
    dr2 = 2 * coords[:, 0] - row_mid
    dc2 = 2 * coords[:, 1] - col_mid

    if rot_k == 1:  # 90 degrees counterclockwise
        new_dr2 = dc2
        new_dc2 = -dr2
    elif rot_k == 2:
        new_dr2 = -dr2
        new_dc2 = -dc2
    else:  # rot_k == 3, 90 degrees clockwise
        new_dr2 = -dc2
        new_dc2 = dr2

    row_num = new_dr2 + row_mid
    col_num = new_dc2 + col_mid
    assert np.all(row_num % 2 == 0) and np.all(col_num % 2 == 0), "rotation left the lattice"

    out = np.stack([row_num // 2, col_num // 2], axis=1).astype(int)
    assert np.all((0 <= out[:, 0]) & (out[:, 0] < h))
    assert np.all((0 <= out[:, 1]) & (out[:, 1] < w))
    return out


def _frame_info(g: np.ndarray) -> tuple[tuple[int, int, int, int], int, str, dict[int, int]]:
    r0, c0, r1, c1 = _bbox(g != BLACK)

    border_vals = []
    border_vals.extend(map(int, g[r0, c0 : c1 + 1]))
    border_vals.extend(map(int, g[r1, c0 : c1 + 1]))
    border_vals.extend(map(int, g[r0 + 1 : r1, c0]))
    border_vals.extend(map(int, g[r0 + 1 : r1, c1]))
    counts: dict[int, int] = {}
    for color in border_vals:
        if color == BLACK:
            continue
        counts[color] = counts.get(color, 0) + 1
    assert counts, "expected a non-black frame"
    frame_color = max(counts.items(), key=lambda item: item[1])[0]

    gap_sides = []
    if np.any(g[r0, c0 + 1 : c1] == BLACK):
        gap_sides.append("top")
    if np.any(g[r0 + 1 : r1, c1] == BLACK):
        gap_sides.append("right")
    if np.any(g[r1, c0 + 1 : c1] == BLACK):
        gap_sides.append("bottom")
    if np.any(g[r0 + 1 : r1, c0] == BLACK):
        gap_sides.append("left")
    assert len(gap_sides) == 1, "expected exactly one border gap"

    corners = {
        0: int(g[r0, c0]),
        1: int(g[r0, c1]),
        2: int(g[r1, c0]),
        3: int(g[r1, c1]),
    }
    markers = {
        quadrant: color
        for quadrant, color in corners.items()
        if color not in (BLACK, frame_color)
    }
    assert 1 <= len(markers) <= 2, "expected one or two corner markers"
    return (r0, c0, r1, c1), int(frame_color), gap_sides[0], markers


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    (r0, c0, r1, c1), _frame_color, gap_side, markers = _frame_info(g)
    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    pattern_coords = np.argwhere(inner != BLACK)
    assert pattern_coords.size > 0, "expected an interior pattern"

    pattern_colors = np.unique(inner[inner != BLACK])
    assert pattern_colors.size == 1, "expected a single pattern color in the frame interior"
    base_color = int(pattern_colors[0])

    rotated = _rotate_local_coords(pattern_coords, inner.shape, SIDE_TO_ROT_K[gap_side])

    out = g.copy()
    out[r0 + 1 : r1, c0 + 1 : c1] = BLACK
    for rr, cc in rotated:
        quadrant = _quadrant_of_local(int(rr), int(cc), inner.shape[0], inner.shape[1])
        color = markers.get(quadrant, base_color) if quadrant is not None else base_color
        out[r0 + 1 + int(rr), c0 + 1 + int(cc)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate the inner sprite from the frame-gap direction, then recolor quadrants named by corner markers.

    Input structure:
    - Black background with one grey rectangular frame of thickness 1 around a black interior.
    - Exactly one frame side has a contiguous 2-3 cell gap cut out of the grey border.
    - The frame interior contains one small connected non-symmetric blue sprite.
    - One or two frame-corner pixels are replaced by non-grey marker colors.

    Rule:
    - Read the gap side as a rotation code for the interior sprite: top = none, right = 90 degrees clockwise,
      bottom = 180 degrees, left = 90 degrees counterclockwise.
    - Rotate the sprite around the frame center while leaving the frame, gap, and corner markers unchanged.
    - Split the interior into the four center-defined quadrants and recolor rotated sprite pixels that land in
      marked quadrants to that quadrant's corner color; pixels in unmarked quadrants stay blue.

    Key insight:
    - The border gap and the colored corners encode two independent controls: the gap chooses orientation first,
      and the corners then recolor the rotated sprite by quadrant membership.
    """
    return _transform(grid)


def _neighbors4(cell: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        cell = stack.pop()
        for nr, nc in _neighbors4(cell, h, w):
            if not mask[nr, nc] or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _is_rotationally_asymmetric(mask: np.ndarray) -> bool:
    for rot_k in (1, 2, 3):
        rotated = np.rot90(mask, rot_k)
        if rotated.shape == mask.shape and np.array_equal(rotated, mask):
            return False
    return True


def _random_sprite_mask() -> np.ndarray | None:
    for _ in range(200):
        h = random.choice((3, 4))
        w = random.choice((3, 4))
        target_area = random.randint(max(4, min(h, w) + 1), min(h * w - 1, 8))
        start = (random.randrange(h), random.randrange(w))
        chosen = {start}
        frontier = set(_neighbors4(start, h, w))
        while len(chosen) < target_area and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            if cell in chosen:
                continue
            chosen.add(cell)
            for nxt in _neighbors4(cell, h, w):
                if nxt not in chosen:
                    frontier.add(nxt)

        mask = np.zeros((h, w), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True
        mask = _tight_mask(mask)
        if mask.shape[0] not in (3, 4) or mask.shape[1] not in (3, 4):
            continue
        if not _is_connected(mask):
            continue
        if not _is_rotationally_asymmetric(mask):
            continue
        return mask
    return None


def _occupied_quadrants(coords: np.ndarray, inner_size: int) -> set[int]:
    quads = set()
    for rr, cc in coords:
        quad = _quadrant_of_local(int(rr), int(cc), inner_size, inner_size)
        if quad is not None:
            quads.add(quad)
    return quads


def _sample_output_coords(inner_size: int, marked_quads: set[int], rot_k: int) -> np.ndarray | None:
    for _ in range(300):
        mask = _random_sprite_mask()
        if mask is None:
            return None
        mh, mw = map(int, mask.shape)
        positions = [(r, c) for r in range(inner_size - mh + 1) for c in range(inner_size - mw + 1)]
        random.shuffle(positions)
        for top, left in positions:
            coords = np.argwhere(mask) + np.array([top, left])
            occupied = _occupied_quadrants(coords, inner_size)
            if not marked_quads.issubset(occupied):
                continue
            if not (occupied - marked_quads):
                continue

            input_coords = _rotate_local_coords(coords, (inner_size, inner_size), (-rot_k) % 4)
            if rot_k and set(map(tuple, coords)) == set(map(tuple, input_coords)):
                continue

            center = (inner_size - 1) / 2.0
            mean_r, mean_c = coords.mean(axis=0)
            if abs(float(mean_r) - center) < 0.35 and abs(float(mean_c) - center) < 0.35:
                continue
            return coords.astype(int)
    return None


def _centered_padding() -> tuple[int, int]:
    base = random.randint(1, 6)
    delta = random.choice((0, 0, 1))
    if random.random() < 0.5:
        return base, base + delta
    return base + delta, base


def _paint_frame(
    canvas: np.ndarray,
    top: int,
    left: int,
    inner_size: int,
    gap_side: str,
    gap_width: int,
    markers: dict[int, int],
) -> None:
    bottom = top + inner_size + 1
    right = left + inner_size + 1
    canvas[top, left : right + 1] = GREY
    canvas[bottom, left : right + 1] = GREY
    canvas[top : bottom + 1, left] = GREY
    canvas[top : bottom + 1, right] = GREY

    gap_start = random.randint(1, inner_size - gap_width + 1)
    if gap_side == "top":
        canvas[top, left + gap_start : left + gap_start + gap_width] = BLACK
    elif gap_side == "right":
        canvas[top + gap_start : top + gap_start + gap_width, right] = BLACK
    elif gap_side == "bottom":
        canvas[bottom, left + gap_start : left + gap_start + gap_width] = BLACK
    else:
        canvas[top + gap_start : top + gap_start + gap_width, left] = BLACK

    corner_cells = {
        0: (top, left),
        1: (top, right),
        2: (bottom, left),
        3: (bottom, right),
    }
    for quadrant, color in markers.items():
        rr, cc = corner_cells[quadrant]
        canvas[rr, cc] = int(color)


def _paint_pattern(
    canvas: np.ndarray,
    top: int,
    left: int,
    coords: np.ndarray,
    markers: dict[int, int] | None = None,
    inner_size: int | None = None,
) -> None:
    for rr, cc in coords:
        color = BLUE
        if markers is not None and inner_size is not None:
            quadrant = _quadrant_of_local(int(rr), int(cc), inner_size, inner_size)
            if quadrant is not None:
                color = markers.get(quadrant, BLUE)
        canvas[top + int(rr), left + int(cc)] = int(color)


def _gen_attempt() -> np.ndarray | None:
    inner_size = random.choice((6, 8, 10, 12))
    gap_side = random.choice(SIDES)
    rot_k = SIDE_TO_ROT_K[gap_side]

    marker_count = random.choice((1, 2))
    marked_quads = set(random.sample(range(4), marker_count))
    marker_colors = random.sample(MARKER_COLORS, marker_count)
    markers = {quad: color for quad, color in zip(marked_quads, marker_colors)}

    output_coords = _sample_output_coords(inner_size, marked_quads, rot_k)
    if output_coords is None:
        return None
    input_coords = _rotate_local_coords(output_coords, (inner_size, inner_size), (-rot_k) % 4)

    pad_top, pad_bottom = _centered_padding()
    pad_left, pad_right = _centered_padding()
    height = inner_size + 2 + pad_top + pad_bottom
    width = inner_size + 2 + pad_left + pad_right
    if height > 30 or width > 30:
        return None

    frame_top = pad_top
    frame_left = pad_left
    inner_top = frame_top + 1
    inner_left = frame_left + 1

    blank = np.full((height, width), BLACK, dtype=int)
    _paint_frame(blank, frame_top, frame_left, inner_size, gap_side, random.choice((2, 3)), markers)

    inp = blank.copy()
    _paint_pattern(inp, inner_top, inner_left, input_coords)

    expected = blank.copy()
    _paint_pattern(expected, inner_top, inner_left, output_coords, markers=markers, inner_size=inner_size)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
