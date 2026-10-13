import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ARC_ROOT = Path("/home/lixinyu/snap/new")
if str(ARC_ROOT) not in sys.path:
    sys.path.append(str(ARC_ROOT))

from solvers.arc_utils import Color, is_contiguous, paste, translate


BG_COLOR = Color.BLACK
DIVIDER_COLOR = Color.GREY
OBJECT_COLORS = [color for color in Color.NOT_BLACK if color != DIVIDER_COLOR]
MAX_GENERATION_ATTEMPTS = 500

# The solver searches this fixed family of simple frame-to-frame transforms.
# Each step is a quarter-turn rotation, an in-panel shift, or both.
TRANSFORM_CANDIDATES = [
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
    (1, 0, 0),
    (2, 0, 0),
    (3, 0, 0),
    (1, 1, 0),
    (1, -1, 0),
    (1, 0, 1),
    (1, 0, -1),
    (2, 1, 0),
    (2, -1, 0),
    (2, 0, 1),
    (2, 0, -1),
    (3, 1, 0),
    (3, -1, 0),
    (3, 0, 1),
    (3, 0, -1),
]


def _snake_positions(panel_rows: int, panel_cols: int) -> list[tuple[int, int]]:
    positions = []
    for pr in range(panel_rows):
        cols = range(panel_cols) if pr % 2 == 0 else range(panel_cols - 1, -1, -1)
        for pc in cols:
            positions.append((pr, pc))
    return positions


def _row_major_positions(panel_rows: int, panel_cols: int) -> list[tuple[int, int]]:
    return [(pr, pc) for pr in range(panel_rows) for pc in range(panel_cols)]


def _panel_origin(pr: int, pc: int, panel_size: int) -> tuple[int, int]:
    return 1 + pr * (panel_size + 1), 1 + pc * (panel_size + 1)


def _blank_table(panel_rows: int, panel_cols: int, panel_size: int) -> np.ndarray:
    height = panel_rows * panel_size + panel_rows + 1
    width = panel_cols * panel_size + panel_cols + 1
    grid = np.full((height, width), DIVIDER_COLOR, dtype=int)
    for pr in range(panel_rows):
        for pc in range(panel_cols):
            r0, c0 = _panel_origin(pr, pc, panel_size)
            grid[r0 : r0 + panel_size, c0 : c0 + panel_size] = BG_COLOR
    return grid


def _apply_transform(panel: np.ndarray, transform: tuple[int, int, int], bg: int) -> np.ndarray:
    rot, dr, dc = transform
    rotated = np.rot90(panel, k=int(rot))
    return translate(rotated, int(dr), int(dc), background=bg)


def _matching_transforms(frames: list[np.ndarray], bg: int) -> list[tuple[int, int, int]]:
    if len(frames) < 2:
        return []

    matches = []
    for transform in TRANSFORM_CANDIDATES:
        if transform[0] % 2 == 1 and frames[0].shape[0] != frames[0].shape[1]:
            continue
        if all(np.array_equal(_apply_transform(frames[i], transform, bg), frames[i + 1]) for i in range(len(frames) - 1)):
            matches.append(transform)
    return matches


def _detect_structure(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    candidates = []
    for color in map(int, np.unique(g)):
        divider_rows = [r for r in range(height) if np.all(g[r, :] == color)]
        divider_cols = [c for c in range(width) if np.all(g[:, c] == color)]
        if len(divider_rows) < 3 or len(divider_cols) < 3:
            continue
        if divider_rows[0] != 0 or divider_rows[-1] != height - 1:
            continue
        if divider_cols[0] != 0 or divider_cols[-1] != width - 1:
            continue
        row_steps = np.diff(divider_rows)
        col_steps = np.diff(divider_cols)
        if len(set(map(int, row_steps))) != 1 or len(set(map(int, col_steps))) != 1:
            continue
        if int(row_steps[0]) < 2 or int(col_steps[0]) < 2:
            continue
        candidates.append(
            {
                "divider_color": color,
                "divider_rows": divider_rows,
                "divider_cols": divider_cols,
                "panel_h": int(row_steps[0] - 1),
                "panel_w": int(col_steps[0] - 1),
            }
        )

    assert len(candidates) == 1, "expected exactly one regular panel grid"
    structure = candidates[0]
    panel_mask = np.ones_like(g, dtype=bool)
    panel_mask[structure["divider_rows"], :] = False
    panel_mask[:, structure["divider_cols"]] = False
    panel_values = g[panel_mask]
    bg = Counter(map(int, panel_values)).most_common(1)[0][0]

    structure["bg"] = int(bg)
    structure["panel_rows"] = len(structure["divider_rows"]) - 1
    structure["panel_cols"] = len(structure["divider_cols"]) - 1
    return structure


def _extract_panels(grid: np.ndarray, structure: dict) -> dict[tuple[int, int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    panels = {}
    for pr in range(structure["panel_rows"]):
        for pc in range(structure["panel_cols"]):
            r0 = structure["divider_rows"][pr] + 1
            c0 = structure["divider_cols"][pc] + 1
            h = structure["panel_h"]
            w = structure["panel_w"]
            panels[(pr, pc)] = g[r0 : r0 + h, c0 : c0 + w].copy()
    return panels


def _render_case(panel_rows: int, panel_cols: int, panel_size: int, frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    inp = _blank_table(panel_rows, panel_cols, panel_size)
    out = inp.copy()
    snake = _snake_positions(panel_rows, panel_cols)
    for index, pos in enumerate(snake):
        r0, c0 = _panel_origin(pos[0], pos[1], panel_size)
        paste(out, frames[index], r0, c0, transparent=BG_COLOR)
        if index < len(frames) - 1:
            paste(inp, frames[index], r0, c0, transparent=BG_COLOR)
    return inp, out


def _random_shape_patch(panel_size: int) -> np.ndarray | None:
    max_span = min(3, panel_size - 1)
    if max_span < 2:
        return None

    for _ in range(200):
        shape_h = random.randint(2, max_span)
        shape_w = random.randint(2, max_span)
        area = shape_h * shape_w
        size = random.randint(3, min(area, 5))
        mask = np.zeros((shape_h, shape_w), dtype=int)
        mask.ravel()[random.sample(range(area), size)] = 1
        if not is_contiguous(mask, background=0, connectivity=4):
            continue
        if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue

        palette_size = random.randint(2, min(3, size))
        palette = random.sample(OBJECT_COLORS, palette_size)
        colors = [random.choice(palette) for _ in range(size)]
        if len(set(colors)) < 2:
            continue

        patch = np.zeros((shape_h, shape_w), dtype=int)
        coords = np.argwhere(mask != 0)
        for (r, c), color in zip(coords, colors):
            patch[int(r), int(c)] = int(color)

        if any(np.array_equal(np.rot90(patch, k=k), patch) for k in (1, 2, 3)):
            continue
        return patch

    return None


def _random_initial_panel(panel_size: int) -> np.ndarray | None:
    patch = _random_shape_patch(panel_size)
    if patch is None:
        return None

    panel = np.zeros((panel_size, panel_size), dtype=int)
    max_r0 = panel_size - patch.shape[0]
    max_c0 = panel_size - patch.shape[1]
    r_low = 0
    c_low = 0
    r_high = max_r0
    c_high = max_c0
    if r_high < r_low or c_high < c_low:
        return None
    r0 = random.randint(r_low, r_high)
    c0 = random.randint(c_low, c_high)
    paste(panel, patch, r0, c0, transparent=0)
    return panel


def _simulate_frames(first_panel: np.ndarray, transform: tuple[int, int, int], count: int) -> list[np.ndarray] | None:
    frames = [np.asarray(first_panel, dtype=int).copy()]
    non_bg = int(np.count_nonzero(first_panel != BG_COLOR))
    for _ in range(count - 1):
        nxt = _apply_transform(frames[-1], transform, BG_COLOR)
        if int(np.count_nonzero(nxt != BG_COLOR)) != non_bg:
            return None
        if not np.any(nxt != BG_COLOR):
            return None
        frames.append(nxt)

    frame_sigs = {frame.tobytes() for frame in frames}
    if len(frame_sigs) != len(frames):
        return None
    return frames


def _row_major_is_plausible(panel_rows: int, panel_cols: int, panel_by_pos: dict[tuple[int, int], np.ndarray]) -> bool:
    snake_last = _snake_positions(panel_rows, panel_cols)[-1]
    row_order = _row_major_positions(panel_rows, panel_cols)
    if row_order[-1] != snake_last:
        return False
    row_frames = [panel_by_pos[pos] for pos in row_order[:-1]]
    return bool(_matching_transforms(row_frames, BG_COLOR))


def _generate_case() -> tuple[np.ndarray, np.ndarray, dict]:
    layouts = [(2, 2), (2, 3), (3, 2)]
    for _ in range(MAX_GENERATION_ATTEMPTS):
        panel_rows, panel_cols = random.choice(layouts)
        max_panel = min((30 - (panel_rows + 1)) // panel_rows, (30 - (panel_cols + 1)) // panel_cols)
        panel_size = random.randint(6, min(8, max_panel))
        panel_count = panel_rows * panel_cols
        transform = random.choice(TRANSFORM_CANDIDATES)

        first_panel = _random_initial_panel(panel_size)
        if first_panel is None:
            continue
        frames = _simulate_frames(first_panel, transform, panel_count)
        if frames is None:
            continue

        observed_frames = frames[:-1]
        matches = _matching_transforms(observed_frames, BG_COLOR)
        if matches != [transform]:
            continue

        inp, out = _render_case(panel_rows, panel_cols, panel_size, frames)
        structure = _detect_structure(inp)
        panels = _extract_panels(inp, structure)
        if _row_major_is_plausible(panel_rows, panel_cols, panels):
            continue
        if not np.array_equal(main(inp), out):
            continue

        meta = {
            "panel_rows": panel_rows,
            "panel_cols": panel_cols,
            "panel_size": panel_size,
            "transform": transform,
            "frames": frames,
        }
        return inp, out, meta
    raise RuntimeError("failed to generate valid snake-album case")


def generate_input() -> np.ndarray:
    return _generate_case()[0]


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Draw the next panel of a snake-read evolution sequence into the lone empty panel.

    Input structure:
    - One-cell divider lines split the grid into equal square panels with an outer border.
    - Every non-empty panel contains the same small colored sprite at a different frame of one evolution sequence.
    - Exactly one panel is empty, and that empty panel is the final position in boustrophedon order.

    Rule:
    - Read the panels in snake order: even panel rows left-to-right, odd panel rows right-to-left.
    - Infer the unique constant one-step transform that maps each observed frame to the next; the step is a quarter-turn rotation, an in-panel shift, or both.
    - Apply that same step once more to the last non-empty frame and draw the result in the empty panel, preserving all divider lines and other panels.

    Key insight:
    - The panel table is a snake-ordered film strip; ordinary row order breaks the frame-to-frame consistency, while snake order reveals one uniform step.
    """
    g = np.asarray(input_grid, dtype=int)
    structure = _detect_structure(g)
    bg = structure["bg"]
    panels = _extract_panels(g, structure)
    snake = _snake_positions(structure["panel_rows"], structure["panel_cols"])
    empty_pos = snake[-1]
    assert np.all(panels[empty_pos] == bg), "snake-terminal panel must be empty"

    frames = [panels[pos] for pos in snake[:-1]]
    matches = _matching_transforms(frames, bg)
    assert len(matches) == 1, "expected a unique frame-to-frame transform"
    predicted = _apply_transform(frames[-1], matches[0], bg)

    out = g.copy()
    r0 = structure["divider_rows"][empty_pos[0]] + 1
    c0 = structure["divider_cols"][empty_pos[1]] + 1
    paste(out, predicted, r0, c0, transparent=bg)
    return out
