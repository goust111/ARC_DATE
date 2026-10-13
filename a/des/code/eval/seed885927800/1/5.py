import numpy as np


BG = 0
FRAME_LAYER_RANGE = (1, 4)
PATTERN_HEIGHT_RANGE = (2, 5)
PATTERN_WIDTH_RANGE = (2, 5)
INNER_PAD_RANGE = (1, 3)
FRAME_GAP_RANGE = (1, 3)
OUTER_MARGIN_RANGE = (1, 4)
PATTERN_COLOR_RANGE = (2, 4)
GENERATION_RETRIES = 400


def _frame_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _pad(arr: np.ndarray, top: int, bottom: int, left: int, right: int, value: int) -> np.ndarray:
    out = np.full((arr.shape[0] + top + bottom, arr.shape[1] + left + right), value, dtype=int)
    out[top : top + arr.shape[0], left : left + arr.shape[1]] = arr
    return out


def _wrap_with_frame(arr: np.ndarray, color: int) -> np.ndarray:
    out = np.full((arr.shape[0] + 2, arr.shape[1] + 2), BG, dtype=int)
    out[1:-1, 1:-1] = arr
    out[0, :] = color
    out[-1, :] = color
    out[:, 0] = color
    out[:, -1] = color
    return out


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    frames: list[tuple[int, int, int, int, int]] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != BG):
        mask = grid == color
        if not bool(mask.any()):
            continue
        coords = np.argwhere(mask)
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        box = mask[r0 : r1 + 1, c0 : c1 + 1]
        border = _frame_mask((h, w))
        if int(box.sum()) != int(border.sum()):
            continue
        if not bool(np.all(box[border])):
            continue
        if bool(np.any(box[~border])):
            continue
        frames.append((r0, c0, r1, c1, color))

    frames.sort(key=lambda item: ((item[2] - item[0] + 1) * (item[3] - item[1] + 1), -item[0], -item[1]), reverse=True)
    for outer, inner in zip(frames, frames[1:]):
        assert inner[0] >= outer[0] + 2 and inner[1] >= outer[1] + 2
        assert inner[2] <= outer[2] - 2 and inner[3] <= outer[3] - 2
    return frames


def _extract_core(grid: np.ndarray, inner_frame: tuple[int, int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1, _ = inner_frame
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    mask = interior != BG
    assert bool(mask.any()), "expected a non-empty inner pattern"
    coords = np.argwhere(mask)
    pr0, pc0 = map(int, coords.min(axis=0))
    pr1, pc1 = map(int, coords.max(axis=0))
    return interior[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()


def _apply_layer_flips(core: np.ndarray, n_layers: int) -> np.ndarray:
    out = np.array(core, dtype=int, copy=True)
    for layer in range(n_layers, 0, -1):
        if layer % 2 == 1:
            out = out[:, ::-1]
        else:
            out = out[::-1, :]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Remove nested frames and flip the core pattern according to layer parity."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _find_frames(g)
    assert 1 <= len(frames) <= 4, "expected 1 to 4 nested frames"

    core = _extract_core(g, frames[-1])
    return _apply_layer_flips(core, len(frames))


def _random_pattern(colors: list[int]) -> np.ndarray:
    max_colors = min(PATTERN_COLOR_RANGE[1], len(colors))
    min_colors = min(PATTERN_COLOR_RANGE[0], max_colors)
    for _ in range(200):
        h = int(np.random.randint(PATTERN_HEIGHT_RANGE[0], PATTERN_HEIGHT_RANGE[1] + 1))
        w = int(np.random.randint(PATTERN_WIDTH_RANGE[0], PATTERN_WIDTH_RANGE[1] + 1))
        k = int(np.random.randint(min_colors, max_colors + 1))
        chosen = list(int(c) for c in np.random.choice(colors, size=k, replace=False))
        pattern = np.random.choice(chosen, size=(h, w)).astype(int)
        if len(np.unique(pattern)) < 2:
            continue
        if any(int(c) not in pattern for c in chosen):
            continue
        if np.array_equal(pattern, pattern[:, ::-1]):
            continue
        if np.array_equal(pattern, pattern[::-1, :]):
            continue
        if np.array_equal(pattern, pattern[::-1, ::-1]):
            continue
        return pattern
    raise RuntimeError("failed to generate a non-symmetric pattern")


def _gen_attempt() -> np.ndarray | None:
    n_layers = int(np.random.randint(FRAME_LAYER_RANGE[0], FRAME_LAYER_RANGE[1] + 1))
    n_pattern_colors = int(np.random.randint(PATTERN_COLOR_RANGE[0], PATTERN_COLOR_RANGE[1] + 1))
    total_colors = n_layers + n_pattern_colors
    palette = list(int(c) for c in np.random.choice(np.arange(1, 10), size=total_colors, replace=False))
    frame_colors = palette[:n_layers]
    pattern_colors = palette[n_layers:]

    pattern = _random_pattern(pattern_colors)
    top = int(np.random.randint(INNER_PAD_RANGE[0], INNER_PAD_RANGE[1] + 1))
    bottom = int(np.random.randint(INNER_PAD_RANGE[0], INNER_PAD_RANGE[1] + 1))
    left = int(np.random.randint(INNER_PAD_RANGE[0], INNER_PAD_RANGE[1] + 1))
    right = int(np.random.randint(INNER_PAD_RANGE[0], INNER_PAD_RANGE[1] + 1))
    current = _pad(pattern, top, bottom, left, right, BG)

    for idx in range(n_layers - 1, -1, -1):
        current = _wrap_with_frame(current, frame_colors[idx])
        if idx > 0:
            gap = int(np.random.randint(FRAME_GAP_RANGE[0], FRAME_GAP_RANGE[1] + 1))
            current = _pad(current, gap, gap, gap, gap, BG)

    outer_top = int(np.random.randint(OUTER_MARGIN_RANGE[0], OUTER_MARGIN_RANGE[1] + 1))
    outer_bottom = int(np.random.randint(OUTER_MARGIN_RANGE[0], OUTER_MARGIN_RANGE[1] + 1))
    outer_left = int(np.random.randint(OUTER_MARGIN_RANGE[0], OUTER_MARGIN_RANGE[1] + 1))
    outer_right = int(np.random.randint(OUTER_MARGIN_RANGE[0], OUTER_MARGIN_RANGE[1] + 1))
    grid = _pad(current, outer_top, outer_bottom, outer_left, outer_right, BG)

    frames = _find_frames(grid)
    if len(frames) != n_layers:
        return None
    try:
        produced = main(grid)
    except AssertionError:
        return None
    expected = _apply_layer_flips(pattern, n_layers)
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame input")
