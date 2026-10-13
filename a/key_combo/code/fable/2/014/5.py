from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
WHITE = 10

DIR_FROM_COLOR: dict[int, tuple[int, int]] = {
    RED: (0, 1),
    BLUE: (1, 0),
    GREEN: (0, -1),
    YELLOW: (-1, 0),
}
COLOR_FROM_DIR = {direction: color for color, direction in DIR_FROM_COLOR.items()}
TURN_ORDER = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up

MAX_ATTEMPTS = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _boundary_color(rect: np.ndarray) -> int:
    h, w = rect.shape
    assert h > 0 and w > 0
    parts = [rect[0, :], rect[-1, :]]
    if h > 2:
        parts.append(rect[1:-1, 0])
        if w > 1:
            parts.append(rect[1:-1, -1])
    elif w > 1:
        parts.extend([rect[:, 0], rect[:, -1]])
    boundary = np.concatenate([part.reshape(-1) for part in parts if part.size])
    colors = np.unique(boundary)
    assert colors.size == 1, "each layer boundary must be monochrome"
    return int(colors[0])


def _extract_layers(rect: np.ndarray) -> list[tuple[int, int]]:
    current = np.asarray(rect, dtype=int)
    layers: list[tuple[int, int]] = []

    while current.size:
        h, w = current.shape
        color = _boundary_color(current)
        assert color in DIR_FROM_COLOR, "layer colors must use the fixed direction palette"

        thickness = 0
        while h - 2 * thickness > 0 and w - 2 * thickness > 0:
            sub = current[thickness : h - thickness, thickness : w - thickness]
            if _boundary_color(sub) != color:
                break
            thickness += 1

        assert thickness >= 1
        layers.append((color, thickness))

        if h - 2 * thickness <= 0 or w - 2 * thickness <= 0:
            current = current[:0, :0]
        else:
            current = current[thickness : h - thickness, thickness : w - thickness]
            assert _boundary_color(current) != color, "adjacent layers may not share a color"

    return layers


def _trace_path(layers: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r = 0
    c = 0
    coords = [(r, c)]
    for color, steps in layers:
        dr, dc = DIR_FROM_COLOR[color]
        for _ in range(steps):
            r += dr
            c += dc
            coords.append((r, c))
    return coords


def _render_output(
    grid: np.ndarray,
    seed: tuple[int, int],
    layers: list[tuple[int, int]],
    onion_mask: np.ndarray,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    r, c = map(int, seed)
    out[r, c] = WHITE
    for color, steps in layers:
        dr, dc = DIR_FROM_COLOR[color]
        for _ in range(steps):
            r += dr
            c += dc
            assert 0 <= r < out.shape[0] and 0 <= c < out.shape[1], "path must stay in bounds"
            assert not onion_mask[r, c], "path may not overwrite the onion"
            out[r, c] = WHITE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the concentric rectangle layers as movement instructions and draw the seed's white walk.

    Input structure:
    - The background is black.
    - One solid concentric-rectangle "onion" appears as a filled rectangle made of
      nested monochrome rectangular layers.
    - One isolated white seed pixel lies in empty space.

    Rule:
    - Read the onion from outermost layer to innermost layer.
    - Each layer contributes one instruction: red=right, blue=down, green=left,
      yellow=up, and the step count is that layer's uniform thickness.
    - Starting from the white seed, execute the instructions in order and paint
      every visited cell white while leaving the onion unchanged.

    Key insight:
    - The nested layer order is the time order: a static colored onion acts like a
      program tape whose colors encode directions and whose thicknesses encode run lengths.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    allowed = set(DIR_FROM_COLOR) | {BLACK, WHITE}
    assert set(map(int, np.unique(g))).issubset(allowed), "unexpected colors present"

    seed_coords = np.argwhere(g == WHITE)
    assert seed_coords.shape == (1, 2), "expected exactly one white seed pixel"
    seed = tuple(map(int, seed_coords[0]))

    onion_mask = np.isin(g, list(DIR_FROM_COLOR))
    onion_coords = np.argwhere(onion_mask)
    assert onion_coords.size, "expected one colored onion"
    r0, c0, r1, c1 = _bbox_coords(onion_coords)
    onion = g[r0 : r1 + 1, c0 : c1 + 1]

    assert onion_mask.sum() == onion.size, "the onion must fill its bounding box"
    assert np.all(np.isin(onion, list(DIR_FROM_COLOR))), "only onion colors may fill the onion bbox"
    assert not onion_mask[seed], "seed must lie outside the onion"

    layers = _extract_layers(onion)
    return _render_output(g, seed, layers, onion_mask)


def _sample_directions(count: int) -> list[tuple[int, int]]:
    start = random.choice(TURN_ORDER)
    while True:
        turns = [random.choice((-1, 1)) for _ in range(count - 1)]
        if count < 4 or len(set(turns)) > 1:
            break

    directions = [start]
    idx = TURN_ORDER.index(start)
    for turn in turns:
        idx = (idx + turn) % 4
        directions.append(TURN_ORDER[idx])
    return directions


def _sample_thicknesses(count: int) -> list[int] | None:
    pool = [1, 1, 2, 2, 3]
    for _ in range(80):
        steps = [random.choice(pool) for _ in range(count)]
        if sum(steps) < max(5, count):
            continue
        if sum(steps) > 8:
            continue
        if len(set(steps)) < 2:
            continue
        return steps
    return None


def _build_onion(colors: list[int], thicknesses: list[int]) -> np.ndarray:
    final_thickness = thicknesses[-1]
    short_side = random.choice([2 * final_thickness - 1, 2 * final_thickness])
    long_side = random.randint(short_side, short_side + 3)
    if random.random() < 0.5:
        h, w = short_side, long_side
    else:
        h, w = long_side, short_side

    onion = np.full((h, w), colors[-1], dtype=int)
    for color, thickness in zip(reversed(colors[:-1]), reversed(thicknesses[:-1])):
        h, w = onion.shape
        wrapped = np.full((h + 2 * thickness, w + 2 * thickness), color, dtype=int)
        wrapped[thickness : thickness + h, thickness : thickness + w] = onion
        onion = wrapped
    return onion


def _path_bbox(path: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in path]
    cols = [c for _, c in path]
    return min(rows), min(cols), max(rows), max(cols)


def _sample_layout(
    onion_shape: tuple[int, int],
    path_box: tuple[int, int, int, int],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    oh, ow = map(int, onion_shape)
    pr0, pc0, pr1, pc1 = path_box
    ph = pr1 - pr0 + 1
    pw = pc1 - pc0 + 1

    side = random.choice(["left", "right", "top", "bottom"])
    top = random.randint(1, 3)
    bottom = random.randint(1, 3)
    left = random.randint(1, 3)
    right = random.randint(1, 3)
    gap = random.randint(3, 5)

    if side in {"left", "right"}:
        min_h = max(oh, ph) + top + bottom
        min_w = ow + pw + gap + left + right
        if min_h > 30 or min_w > 30:
            return None
        H = random.randint(min_h, min(30, min_h + 4))
        W = random.randint(min_w, min(30, min_w + 4))
        path_r = random.randint(top, H - bottom - ph)
        onion_r = random.randint(top, H - bottom - oh)
        if side == "left":
            path_c = random.randint(left, W - right - ow - gap - pw)
            onion_c = random.randint(path_c + pw + gap, W - right - ow)
        else:
            onion_c = random.randint(left, W - right - pw - gap - ow)
            path_c = random.randint(onion_c + ow + gap, W - right - pw)
    else:
        min_h = oh + ph + gap + top + bottom
        min_w = max(ow, pw) + left + right
        if min_h > 30 or min_w > 30:
            return None
        H = random.randint(min_h, min(30, min_h + 4))
        W = random.randint(min_w, min(30, min_w + 4))
        path_c = random.randint(left, W - right - pw)
        onion_c = random.randint(left, W - right - ow)
        if side == "top":
            path_r = random.randint(top, H - bottom - oh - gap - ph)
            onion_r = random.randint(path_r + ph + gap, H - bottom - oh)
        else:
            onion_r = random.randint(top, H - bottom - ph - gap - oh)
            path_r = random.randint(onion_r + oh + gap, H - bottom - ph)

    seed = (path_r - pr0, path_c - pc0)
    return (H, W), seed, (onion_r, onion_c)


def _render_input_and_output(
    onion: np.ndarray,
    path: list[tuple[int, int]],
    layers: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray] | None:
    layout = _sample_layout(onion.shape, _path_bbox(path))
    if layout is None:
        return None

    (H, W), seed, onion_pos = layout
    onion_r, onion_c = onion_pos
    grid = np.full((H, W), BLACK, dtype=int)
    grid[onion_r : onion_r + onion.shape[0], onion_c : onion_c + onion.shape[1]] = onion
    grid[seed] = WHITE

    onion_mask = np.isin(grid, list(DIR_FROM_COLOR))
    shifted_path = {(seed[0] + r, seed[1] + c) for r, c in path}
    if any(not (0 <= r < H and 0 <= c < W) for r, c in shifted_path):
        return None
    if any(onion_mask[r, c] for r, c in shifted_path):
        return None

    out = _render_output(grid, seed, layers, onion_mask)
    white_cells = {tuple(map(int, rc)) for rc in np.argwhere(out == WHITE)}
    if white_cells != shifted_path:
        return None
    return grid, out


def _semantic_ok(
    inp: np.ndarray,
    expected: np.ndarray,
    layers: list[tuple[int, int]],
    path: list[tuple[int, int]],
) -> bool:
    if inp.ndim != 2 or expected.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return False

    if np.argwhere(inp == WHITE).shape != (1, 2):
        return False

    onion_mask = np.isin(inp, list(DIR_FROM_COLOR))
    onion_coords = np.argwhere(onion_mask)
    if onion_coords.size == 0:
        return False
    r0, c0, r1, c1 = _bbox_coords(onion_coords)
    onion = inp[r0 : r1 + 1, c0 : c1 + 1]
    if onion_mask.sum() != onion.size:
        return False
    if _extract_layers(onion) != layers:
        return False

    seed = tuple(map(int, np.argwhere(inp == WHITE)[0]))
    expected_path = {(seed[0] + r, seed[1] + c) for r, c in path}
    white_cells = {tuple(map(int, rc)) for rc in np.argwhere(expected == WHITE)}
    if white_cells != expected_path:
        return False
    if int(((expected == WHITE) & (inp != WHITE)).sum()) != len(path) - 1:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    layer_count = random.randint(3, 6)
    thicknesses = _sample_thicknesses(layer_count)
    if thicknesses is None:
        return None

    directions = _sample_directions(layer_count)
    colors = [COLOR_FROM_DIR[direction] for direction in directions]
    if len(set(colors)) < min(3, layer_count):
        return None

    onion = _build_onion(colors, thicknesses)
    layers = list(zip(colors, thicknesses))
    if _extract_layers(onion) != layers:
        return None

    path = _trace_path(layers)
    if len(set(path)) != len(path):
        return None
    box = _path_bbox(path)
    if box[0] == box[2] or box[1] == box[3]:
        return None

    rendered = _render_input_and_output(onion, path, layers)
    if rendered is None:
        return None
    inp, expected = rendered

    if not _semantic_ok(inp, expected, layers, path):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
