from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

PATTERN_COLORS = (GREEN, YELLOW, GREY, PINK, ORANGE, TEAL, MAROON)
PATTERN_SCALE_CHOICES = (
    (2, 3),
    (2, 4),
    (3, 2),
    (3, 3),
    (3, 4),
    (4, 2),
    (4, 3),
)
MAX_GENERATION_ATTEMPTS = 800

_SEEN_INPUT_SIGNATURES: set[tuple[tuple[int, int], bytes]] = set()
_SEEN_OUTPUT_SIGNATURES: set[tuple[tuple[int, int], bytes]] = set()
_SEEN_CONFIG_SIGNATURES: set[tuple[int, int, str, tuple[int, ...], bytes]] = set()


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _signature(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(int(x) for x in grid.shape), grid.tobytes()


def _scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(sprite, factor, axis=0), factor, axis=1)


def _find_frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    red_components = _components(grid == RED)
    candidates: list[tuple[int, int, int, int]] = []
    for coords in red_components:
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        side_h = r1 - r0 + 1
        side_w = c1 - c0 + 1
        if side_h != side_w or side_h < 4:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(sub, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(sub[border] == RED):
            continue
        if np.any(sub[1:-1, 1:-1] == RED):
            continue
        candidates.append((r0, c0, r1, c1))
    assert len(candidates) == 1, "expected exactly one hollow red square frame"
    return candidates[0]


def _extract_pattern(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> np.ndarray:
    pattern_mask = ~np.isin(grid, (BLACK, BLUE, RED))
    components = _components(pattern_mask)
    assert len(components) == 1, "expected one separate multicolor pattern"
    bbox = _bbox_from_coords(components[0])
    assert not _bbox_intersects(bbox, frame_bbox), "pattern must be separate from the frame"
    r0, c0, r1, c1 = bbox
    sprite = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert sprite.shape[0] == sprite.shape[1], "pattern bbox must be square"
    assert np.all(sprite != BLACK), "pattern cells must all be colored"
    assert not np.any(np.isin(sprite, (BLUE, RED))), "pattern colors must exclude frame and marker colors"
    return sprite


def _marker_axis_positions(interior: np.ndarray) -> tuple[str, list[int]]:
    n = int(interior.shape[0])
    blue_mask = interior == BLUE
    assert blue_mask.any(), "expected blue marker bars inside the frame"

    row_positions = [r for r in range(n) if np.all(blue_mask[r, :])]
    col_positions = [c for c in range(n) if np.all(blue_mask[:, c])]

    row_mask = np.zeros_like(blue_mask, dtype=bool)
    col_mask = np.zeros_like(blue_mask, dtype=bool)
    if row_positions:
        row_mask[row_positions, :] = True
    if col_positions:
        col_mask[:, col_positions] = True

    row_ok = 2 <= len(row_positions) < n and np.array_equal(blue_mask, row_mask)
    col_ok = 2 <= len(col_positions) < n and np.array_equal(blue_mask, col_mask)
    assert row_ok ^ col_ok, "marker bars must be uniformly horizontal or vertical"

    positions = row_positions if row_ok else col_positions
    diffs = np.diff(np.array(positions, dtype=int))
    assert np.all(diffs == diffs[0]) and int(diffs[0]) > 0, "marker bars must be equally spaced"
    return ("rows" if row_ok else "cols"), [int(x) for x in positions]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale the external pattern into the red frame and overwrite the frame-selected periodic rows or columns in red.

    Input structure:
    - The grid has black background, one hollow red square frame, and one separate small square pattern made of non-black, non-red, non-blue colors.
    - Inside the frame, the red border surrounds an NxN black interior that also contains several equally spaced full blue marker rows or full blue marker columns.
    - The pattern's side length is a divisor of the frame's inner side length, so it can be scaled by an integer factor.

    Rule:
    - Read the red frame's inner side length and divide it by the small pattern's side length to get the nearest-neighbor scale factor.
    - Scale the small pattern to exactly fill the frame interior.
    - Use the blue full rows or full columns inside the input frame as local marker coordinates, then repaint those same rows or columns of the scaled interior red.
    - Return only the cropped red frame together with the scaled pattern interior after the red stripe overwrite.

    Key insight:
    - The red frame has a dual role: its inner size determines the scaling factor, while the blue bars embedded in that same frame specify the periodic stripe mask to apply after scaling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_r0, frame_c0, frame_r1, frame_c1 = _find_frame_bbox(g)
    frame_side = frame_r1 - frame_r0 + 1
    inner_size = frame_side - 2
    assert inner_size > 0

    interior = g[frame_r0 + 1 : frame_r1, frame_c0 + 1 : frame_c1]
    assert interior.shape == (inner_size, inner_size)
    assert set(map(int, np.unique(interior))) <= {BLACK, BLUE}, "frame interior may only contain black and blue"
    axis, positions = _marker_axis_positions(interior)

    pattern = _extract_pattern(g, (frame_r0, frame_c0, frame_r1, frame_c1))
    pattern_side = int(pattern.shape[0])
    assert inner_size % pattern_side == 0, "frame interior size must be an integer multiple of the pattern size"
    scale = inner_size // pattern_side
    scaled = _scale_sprite(pattern, scale)
    assert scaled.shape == (inner_size, inner_size)

    out = np.full((frame_side, frame_side), BLACK, dtype=int)
    out[0, :] = RED
    out[-1, :] = RED
    out[:, 0] = RED
    out[:, -1] = RED
    out[1:-1, 1:-1] = scaled
    if axis == "rows":
        out[1 + np.array(positions, dtype=int), 1:-1] = RED
    else:
        out[1:-1, 1 + np.array(positions, dtype=int)] = RED
    return out


def _random_pattern(side: int, rng: np.random.Generator) -> np.ndarray:
    max_colors = min(4, len(PATTERN_COLORS))
    for _ in range(300):
        color_count = int(rng.integers(2, max_colors + 1))
        palette = rng.choice(np.array(PATTERN_COLORS, dtype=int), size=color_count, replace=False)
        sprite = rng.choice(palette, size=(side, side), replace=True).astype(int)
        if len(set(map(int, sprite.flatten()))) < 2:
            continue
        if len({tuple(map(int, row)) for row in sprite}) < 2:
            continue
        if len({tuple(map(int, col)) for col in sprite.T}) < 2:
            continue
        return sprite
    raise RuntimeError("failed to sample a non-degenerate pattern")


def _sample_marker_positions(inner_size: int, rng: np.random.Generator) -> list[int]:
    max_period = min(5, inner_size - 1)
    for _ in range(300):
        period = int(rng.integers(2, max_period + 1))
        offset = int(rng.integers(0, period))
        positions = list(range(offset, inner_size, period))
        if 2 <= len(positions) < inner_size:
            return positions
    raise RuntimeError("failed to sample periodic marker positions")


def _pattern_checksum(pattern: np.ndarray, inner_size: int, axis: str, positions: list[int]) -> int:
    weights = np.arange(1, pattern.size + 1, dtype=int).reshape(pattern.shape)
    stripe_code = sum((idx + 1) * pos for idx, pos in enumerate(positions))
    return int(
        np.sum(pattern * weights)
        + 17 * inner_size
        + 19 * len(positions)
        + 23 * stripe_code
        + (29 if axis == "cols" else 0)
    )


def _layout(
    inner_size: int,
    pattern: np.ndarray,
    axis: str,
    positions: list[int],
) -> tuple[int, int, int, int, int, int]:
    pattern_side = int(pattern.shape[0])
    frame_side = inner_size + 2
    code = _pattern_checksum(pattern, inner_size, axis, positions)
    top_margin = 1 + (code % 3)
    left_margin = 1 + ((code // 3) % 3)
    gap = 2 + ((code // 9) % 3)
    variant = code % 4

    if variant == 0:
        frame_r0 = top_margin
        frame_c0 = left_margin
        pattern_r0 = top_margin
        pattern_c0 = frame_c0 + frame_side + gap
    elif variant == 1:
        pattern_r0 = top_margin
        pattern_c0 = left_margin
        frame_r0 = pattern_r0 + pattern_side + gap
        frame_c0 = left_margin
    elif variant == 2:
        pattern_r0 = top_margin
        pattern_c0 = left_margin
        frame_r0 = top_margin
        frame_c0 = pattern_c0 + pattern_side + gap
    else:
        frame_r0 = top_margin
        frame_c0 = left_margin
        pattern_r0 = frame_r0 + frame_side + gap
        pattern_c0 = left_margin

    height = max(frame_r0 + frame_side, pattern_r0 + pattern_side) + top_margin + 1
    width = max(frame_c0 + frame_side, pattern_c0 + pattern_side) + left_margin + 1
    assert height <= 30 and width <= 30
    return int(height), int(width), int(frame_r0), int(frame_c0), int(pattern_r0), int(pattern_c0)


def _build_input(
    pattern: np.ndarray,
    inner_size: int,
    axis: str,
    positions: list[int],
) -> np.ndarray:
    height, width, frame_r0, frame_c0, pattern_r0, pattern_c0 = _layout(inner_size, pattern, axis, positions)
    frame_side = inner_size + 2
    pattern_side = int(pattern.shape[0])

    grid = np.zeros((height, width), dtype=int)
    grid[frame_r0, frame_c0 : frame_c0 + frame_side] = RED
    grid[frame_r0 + frame_side - 1, frame_c0 : frame_c0 + frame_side] = RED
    grid[frame_r0 : frame_r0 + frame_side, frame_c0] = RED
    grid[frame_r0 : frame_r0 + frame_side, frame_c0 + frame_side - 1] = RED

    if axis == "rows":
        for row in positions:
            grid[frame_r0 + 1 + row, frame_c0 + 1 : frame_c0 + 1 + inner_size] = BLUE
    else:
        for col in positions:
            grid[frame_r0 + 1 : frame_r0 + 1 + inner_size, frame_c0 + 1 + col] = BLUE

    grid[pattern_r0 : pattern_r0 + pattern_side, pattern_c0 : pattern_c0 + pattern_side] = pattern
    return grid


def _semantic_checks(
    inp: np.ndarray,
    out: np.ndarray,
    pattern: np.ndarray,
    inner_size: int,
    axis: str,
    positions: list[int],
) -> bool:
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if out.shape != (inner_size + 2, inner_size + 2):
        return False

    expected = _scale_sprite(pattern, inner_size // int(pattern.shape[0]))
    inner_out = out[1:-1, 1:-1]
    if axis == "rows":
        if not np.all(inner_out[np.array(positions, dtype=int), :] == RED):
            return False
        unmarked = [idx for idx in range(inner_size) if idx not in positions]
        if len(unmarked) < 2:
            return False
        if len({tuple(map(int, inner_out[r, :])) for r in unmarked}) < 2:
            return False
        if np.array_equal(inner_out, expected):
            return False
    else:
        if not np.all(inner_out[:, np.array(positions, dtype=int)] == RED):
            return False
        unmarked = [idx for idx in range(inner_size) if idx not in positions]
        if len(unmarked) < 2:
            return False
        if len({tuple(map(int, inner_out[:, c])) for c in unmarked}) < 2:
            return False
        if np.array_equal(inner_out, expected):
            return False

    visible = inner_out[inner_out != RED]
    if visible.size == 0 or len(set(map(int, visible))) < 2:
        return False
    if not np.all(out[0, :] == RED):
        return False
    if not np.all(out[-1, :] == RED):
        return False
    if not np.all(out[:, 0] == RED):
        return False
    if not np.all(out[:, -1] == RED):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        pattern_side, scale = PATTERN_SCALE_CHOICES[int(rng.integers(0, len(PATTERN_SCALE_CHOICES)))]
        inner_size = pattern_side * scale
        axis = "rows" if int(rng.integers(0, 2)) == 0 else "cols"
        positions = _sample_marker_positions(inner_size, rng)
        pattern = _random_pattern(pattern_side, rng)

        config_signature = (
            int(pattern_side),
            int(scale),
            axis,
            tuple(int(x) for x in positions),
            pattern.tobytes(),
        )
        if config_signature in _SEEN_CONFIG_SIGNATURES:
            continue

        inp = _build_input(pattern, inner_size, axis, positions)
        try:
            out = main(inp)
        except AssertionError:
            continue

        if not _semantic_checks(inp, out, pattern, inner_size, axis, positions):
            continue

        input_signature = _signature(inp)
        output_signature = _signature(out)
        if input_signature in _SEEN_INPUT_SIGNATURES or output_signature in _SEEN_OUTPUT_SIGNATURES:
            continue

        _SEEN_CONFIG_SIGNATURES.add(config_signature)
        _SEEN_INPUT_SIGNATURES.add(input_signature)
        _SEEN_OUTPUT_SIGNATURES.add(output_signature)
        return inp

    raise RuntimeError("failed to generate a valid non-degenerate input")
