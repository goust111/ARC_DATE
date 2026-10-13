import random
from collections import deque

import numpy as np


BG_COLOR = 0
COLOR_POOL = tuple(range(1, 10))

PATTERN_HEIGHT_RANGE = (4, 7)
PATTERN_WIDTH_RANGE = (4, 7)
FRAME_COUNT_RANGE = (2, 4)
EXTRA_PATTERN_COLORS_MAX = 2
SLOT_GAP_OPTIONS = (0, 1)
GENERATION_RETRIES = 300

OUTER_SLOTS = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
)


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    components = []

    for row in range(height):
        for col in range(width):
            if seen[row, col] or int(grid[row, col]) != color:
                continue
            queue = deque([(row, col)])
            seen[row, col] = True
            coords = []
            while queue:
                r, c = queue.popleft()
                coords.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )

    return components


def _is_filter_frame(grid: np.ndarray, component: dict, bg: int) -> bool:
    height, width = grid.shape
    color = int(component["color"])
    r0, c0, r1, c1 = component["bbox"]
    box_h = r1 - r0 + 1
    box_w = c1 - c0 + 1
    if box_h < 3 or box_w < 3:
        return False
    if not (r0 <= 1 or c0 <= 1 or r1 >= height - 2 or c1 >= width - 2):
        return False
    if component["coords"].shape[0] != 2 * (box_h + box_w) - 4:
        return False

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((box_h, box_w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.all(patch[border] == color) and np.all(patch[~border] == bg))


def _mask_from_components(shape: tuple[int, int], components: list[dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for component in components:
        coords = component["coords"]
        mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def main(grid: np.ndarray) -> np.ndarray:
    """Project the central multicolor pattern into each edge frame as a color-selected layer.

    Input structure:
    - The dominant color is the background.
    - A large multicolor pattern occupies a central bounding box.
    - Several monochrome one-cell-thick hollow rectangular frames sit on or near the grid edges.
    - Every frame interior is background and has the same shape as the central pattern's bounding box.

    Rule:
    - Detect the hollow edge frames and keep their borders unchanged.
    - Take the bounding box of all non-background, non-frame pixels as the shared source window.
    - For each frame, align its interior with that source window and copy only the source pixels whose color matches the frame border color.
    - Leave the original pattern unchanged and keep every non-matching interior cell as background.

    Key insight:
    - Each frame color is a filter instruction: the frames do not receive independent drawings, they each extract one color layer from the same source pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)

    frame_candidates = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        for component in _components_of_color(g, color):
            if _is_filter_frame(g, component, bg):
                frame_candidates.append(component)

    assert frame_candidates, "expected at least one filter frame"

    candidate_mask = _mask_from_components(g.shape, frame_candidates)
    pattern_mask = (g != bg) & ~candidate_mask
    pr0, pc0, pr1, pc1 = _bbox_from_mask(pattern_mask)
    pattern = g[pr0 : pr1 + 1, pc0 : pc1 + 1]
    pattern_shape = pattern.shape

    frames = []
    for component in frame_candidates:
        r0, c0, r1, c1 = component["bbox"]
        if (r1 - r0 - 1, c1 - c0 - 1) == pattern_shape:
            frames.append(component)
    assert frames, "expected frames whose interiors match the pattern bbox"

    frame_mask = _mask_from_components(g.shape, frames)
    pattern_mask = (g != bg) & ~frame_mask
    pr0, pc0, pr1, pc1 = _bbox_from_mask(pattern_mask)
    pattern = g[pr0 : pr1 + 1, pc0 : pc1 + 1]
    assert pattern.shape == pattern_shape

    out = g.copy()
    for component in frames:
        color = int(component["color"])
        r0, c0, r1, c1 = component["bbox"]
        target = out[r0 + 1 : r1, c0 + 1 : c1]
        target[:] = np.where(pattern == color, color, bg)

    return out


def _slot_start(index: int, span: int, gap: int) -> int:
    return index * (span + gap)


def _draw_frame(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    grid[top, left : left + width] = color
    grid[top + height - 1, left : left + width] = color
    grid[top : top + height, left] = color
    grid[top : top + height, left + width - 1] = color


def _make_pattern(height: int, width: int, colors: list[int], frame_colors: list[int]) -> np.ndarray:
    rows, cols = np.indices((height, width))
    palette = np.array(colors, dtype=int)
    count = len(colors)
    mode = random.choice(("checker", "diagonal", "row_runs", "col_runs", "patchwork"))

    if mode == "checker":
        row_band = random.randint(1, min(2, height))
        col_band = random.randint(1, min(2, width))
        phase = random.randrange(count)
        pattern = palette[((rows // row_band) + (cols // col_band) + phase) % count]
    elif mode == "diagonal":
        band = random.randint(1, 2)
        phase = random.randrange(count)
        diagonal = rows + (cols if random.random() < 0.5 else (width - 1 - cols))
        pattern = palette[(diagonal // band + phase) % count]
    elif mode == "row_runs":
        pattern = np.empty((height, width), dtype=int)
        for row in range(height):
            cuts = (
                sorted(random.sample(range(1, width), random.randint(0, min(3, width - 1))))
                if width > 1
                else []
            )
            starts = [0] + cuts
            ends = cuts + [width]
            previous = None
            for c0, c1 in zip(starts, ends):
                choices = [color for color in colors if color != previous] or colors
                chosen = random.choice(choices)
                pattern[row, c0:c1] = chosen
                previous = chosen
    elif mode == "col_runs":
        pattern = np.empty((height, width), dtype=int)
        for col in range(width):
            cuts = (
                sorted(random.sample(range(1, height), random.randint(0, min(3, height - 1))))
                if height > 1
                else []
            )
            starts = [0] + cuts
            ends = cuts + [height]
            previous = None
            for r0, r1 in zip(starts, ends):
                choices = [color for color in colors if color != previous] or colors
                chosen = random.choice(choices)
                pattern[r0:r1, col] = chosen
                previous = chosen
    else:
        pattern = np.empty((height, width), dtype=int)
        row_cuts = (
            sorted(random.sample(range(1, height), random.randint(1, min(2, height - 1))))
            if height > 1
            else []
        )
        col_cuts = (
            sorted(random.sample(range(1, width), random.randint(1, min(2, width - 1))))
            if width > 1
            else []
        )
        row_edges = [0] + row_cuts + [height]
        col_edges = [0] + col_cuts + [width]
        last_color = None
        for r0, r1 in zip(row_edges, row_edges[1:]):
            for c0, c1 in zip(col_edges, col_edges[1:]):
                choices = [color for color in colors if color != last_color] or colors
                chosen = random.choice(choices)
                pattern[r0:r1, c0:c1] = chosen
                last_color = chosen

    required = {color: 1 for color in colors}
    for color in frame_colors:
        required[color] = 3
    positions = list(np.ndindex(pattern.shape))
    random.shuffle(positions)
    pos_index = 0
    for color, need in required.items():
        for _ in range(need):
            row, col = positions[pos_index]
            pos_index += 1
            pattern[row, col] = color

    return pattern


def _generate_case() -> np.ndarray | None:
    pattern_height = random.randint(*PATTERN_HEIGHT_RANGE)
    pattern_width = random.randint(*PATTERN_WIDTH_RANGE)
    frame_count = random.randint(*FRAME_COUNT_RANGE)
    extra_colors = random.randint(0, min(EXTRA_PATTERN_COLORS_MAX, len(COLOR_POOL) - frame_count))
    total_colors = frame_count + extra_colors
    pattern_colors = random.sample(COLOR_POOL, total_colors)
    frame_colors = random.sample(pattern_colors, frame_count)

    frame_height = pattern_height + 2
    frame_width = pattern_width + 2
    gap = random.choice(SLOT_GAP_OPTIONS)
    grid_height = 3 * frame_height + 2 * gap
    grid_width = 3 * frame_width + 2 * gap
    if max(grid_height, grid_width) > 30:
        return None

    grid = np.full((grid_height, grid_width), BG_COLOR, dtype=int)

    pattern = _make_pattern(pattern_height, pattern_width, pattern_colors, frame_colors)
    pattern_top = _slot_start(1, frame_height, gap) + 1
    pattern_left = _slot_start(1, frame_width, gap) + 1
    grid[pattern_top : pattern_top + pattern_height, pattern_left : pattern_left + pattern_width] = pattern

    chosen_slots = random.sample(OUTER_SLOTS, frame_count)
    for color, (slot_row, slot_col) in zip(frame_colors, chosen_slots):
        top = _slot_start(slot_row, frame_height, gap)
        left = _slot_start(slot_col, frame_width, gap)
        _draw_frame(grid, top, left, frame_height, frame_width, color)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid filter-frame input")
