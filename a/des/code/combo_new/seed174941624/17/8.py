import random
from math import ceil

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
RATIO_OPTIONS = (
    (1, 1),
    (1, 3),
    (3, 1),
    (3, 5),
    (5, 3),
)
ARC_MAX = 30
GEN_RETRIES = 400


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _draw_frame(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    grid[top, left : left + width] = color
    grid[top + height - 1, left : left + width] = color
    grid[top : top + height, left] = color
    grid[top : top + height, left + width - 1] = color


def _extract_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, GREY)):
        coords = np.argwhere(grid == color)
        assert coords.size > 0, "foreground colors must appear"
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height >= 3 and width >= 3, "each frame must have an empty interior"
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = _border_mask(height, width)
        assert np.array_equal(patch == color, border), "each colored object must be a 1-pixel-thick hollow rectangle"
        frames.append(
            {
                "color": color,
                "height": height,
                "width": width,
                "area": height * width,
            }
        )

    assert len(frames) >= 2, "expected multiple hollow frames"
    return frames


def _build_nested_output(shape: tuple[int, int], target: tuple[int, int], frames: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    center_r, center_c = target

    ordered = sorted(frames, key=lambda item: (-item["area"], -item["height"], -item["width"]))
    for frame in ordered:
        height = int(frame["height"])
        width = int(frame["width"])
        top = center_r - height // 2
        left = center_c - width // 2
        assert 0 <= top and top + height <= shape[0], "nested frame would leave the grid vertically"
        assert 0 <= left and left + width <= shape[1], "nested frame would leave the grid horizontally"
        _draw_frame(out, top, left, height, width, int(frame["color"]))

    if out[center_r, center_c] == BG:
        out[center_r, center_c] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Gather scattered hollow frames and reassemble them as one concentric nest around the grey target point.

    Input structure:
    - The background is black.
    - Several non-overlapping colored hollow rectangles appear in different places; each one is a 1-pixel-thick rectangular ring with a black interior.
    - All frame colors are unique, all frames share the same aspect ratio, and a single grey pixel marks the assembly center.

    Rule:
    - Detect every colored frame and measure the area of its outer bounding rectangle.
    - Sort the frames from largest outer area to smallest.
    - Erase their original locations, then place the largest frame so the grey marker is at its center, and center every smaller frame on that same point so the frames nest concentrically.
    - Keep only the assembled nest at the target location; if the center cell is still uncovered after stamping the smallest frame, leave that cell grey.

    Key insight:
    - The separate rings are parts of one ordered size sequence: sorting them by outer size and aligning all their centers to the grey marker reveals the intended concentric construction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    targets = np.argwhere(g == GREY)
    assert targets.shape[0] == 1, "expected exactly one grey target pixel"
    target = tuple(map(int, targets[0]))

    frames = _extract_frames(g)
    return _build_nested_output(g.shape, target, frames)


def _choose_sizes() -> list[tuple[int, int]]:
    for _ in range(200):
        ratio_h, ratio_w = random.choice(RATIO_OPTIONS)
        min_scale = max(ceil(3 / ratio_h), ceil(3 / ratio_w))
        if min_scale % 2 == 0:
            min_scale += 1
        max_scale = min((ARC_MAX - 7) // ratio_h, (ARC_MAX - 7) // ratio_w)
        scales = [scale for scale in range(min_scale, max_scale + 1, 2)]
        if len(scales) < 2:
            continue

        max_count = min(5, len(scales))
        count = random.randint(2, max_count)
        chosen = sorted(random.sample(scales, count))
        sizes = [(ratio_h * scale, ratio_w * scale) for scale in chosen]

        largest_h, largest_w = sizes[-1]
        padded_area = sum((h + 2) * (w + 2) for h, w in sizes) + 9
        if largest_h > 23 or largest_w > 23:
            continue
        if padded_area > 650:
            continue
        return sizes
    raise RuntimeError("failed to sample compatible frame sizes")


def _sample_canvas(sizes: list[tuple[int, int]]) -> tuple[int, int]:
    largest_h, largest_w = sizes[-1]
    padded_area = sum((h + 2) * (w + 2) for h, w in sizes) + 9
    min_h = max(12, largest_h + 2)
    min_w = max(12, largest_w + 2)

    for _ in range(200):
        height = random.randint(min_h, ARC_MAX)
        width = random.randint(min_w, ARC_MAX)
        if height * width >= padded_area * 1.35:
            return height, width
    return ARC_MAX, ARC_MAX


def _free_positions(occupied: np.ndarray, height: int, width: int) -> list[tuple[int, int]]:
    grid_h, grid_w = occupied.shape
    positions: list[tuple[int, int]] = []
    for top in range(grid_h - height + 1):
        for left in range(grid_w - width + 1):
            r0 = max(0, top - 1)
            c0 = max(0, left - 1)
            r1 = min(grid_h, top + height + 1)
            c1 = min(grid_w, left + width + 1)
            if not occupied[r0:r1, c0:c1].any():
                positions.append((top, left))
    return positions


def _gen_attempt() -> np.ndarray | None:
    sizes = _choose_sizes()
    height, width = _sample_canvas(sizes)
    largest_h, largest_w = sizes[-1]

    center_r = random.randint(largest_h // 2, height - 1 - largest_h // 2)
    center_c = random.randint(largest_w // 2, width - 1 - largest_w // 2)

    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    occupied[max(0, center_r - 1) : min(height, center_r + 2), max(0, center_c - 1) : min(width, center_c + 2)] = True

    colors = random.sample(COLOR_POOL, len(sizes))
    color_order = list(zip(colors, sizes))
    random.shuffle(color_order)

    for color, (frame_h, frame_w) in sorted(color_order, key=lambda item: -(item[1][0] * item[1][1])):
        positions = _free_positions(occupied, frame_h, frame_w)
        if not positions:
            return None
        top, left = random.choice(positions)
        _draw_frame(grid, top, left, frame_h, frame_w, color)
        occupied[max(0, top - 1) : min(height, top + frame_h + 1), max(0, left - 1) : min(width, left + frame_w + 1)] = True

    grid[center_r, center_c] = GREY

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
