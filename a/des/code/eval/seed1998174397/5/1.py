from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIRECTIONS = ("up", "down", "left", "right")
ARROW_COLOR = 4
FRAME_COLOR_POOL = [1, 2, 3, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 300

GRID_RANGES = {
    2: ((18, 24), (18, 24)),
    3: ((20, 27), (20, 27)),
    4: ((22, 30), (22, 30)),
}
FRAME_HEIGHT_RANGE = (5, 10)
FRAME_WIDTH_RANGE = (5, 10)
PLACEMENT_PADDING = 1

ARROW_MASKS = {
    "up": np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
    "down": np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    "left": np.array([[0, 1], [1, 1], [0, 1]], dtype=bool),
    "right": np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
}

ARROW_KEYS = {
    (2, 3, ((0, 1), (1, 0), (1, 1), (1, 2))): "up",
    (2, 3, ((0, 0), (0, 1), (0, 2), (1, 1))): "down",
    (3, 2, ((0, 1), (1, 0), (1, 1), (2, 1))): "left",
    (3, 2, ((0, 0), (1, 0), (1, 1), (2, 0))): "right",
}


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = map(int, grid.shape)
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or visited[r, c]:
                continue

            visited[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = map(int, arr.min(axis=0))
            r1, c1 = map(int, arr.max(axis=0))
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )

    return components


def _coords_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = component["coords"] - np.array([[r0, c0]], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _frame_opening(component: dict) -> str | None:
    r0, c0, r1, c1 = component["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 5 or w < 5:
        return None

    mask = _coords_mask(component)
    if np.any(mask[1:-1, 1:-1]):
        return None
    if not (mask[0, 0] and mask[0, -1] and mask[-1, 0] and mask[-1, -1]):
        return None

    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    expected_masks: dict[str, np.ndarray] = {}
    for direction in DIRECTIONS:
        expected = border.copy()
        if direction == "up":
            expected[0, 1:-1] = False
        elif direction == "down":
            expected[-1, 1:-1] = False
        elif direction == "left":
            expected[1:-1, 0] = False
        else:
            expected[1:-1, -1] = False
        expected_masks[direction] = expected

    matches = [direction for direction, expected in expected_masks.items() if np.array_equal(mask, expected)]
    if len(matches) != 1:
        return None
    return matches[0]


def _arrow_direction(component: dict) -> str | None:
    if int(component["area"]) != 4:
        return None
    r0, c0, r1, c1 = component["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    rel = component["coords"] - np.array([[r0, c0]], dtype=int)
    key = (h, w, tuple(sorted((int(r), int(c)) for r, c in rel)))
    return ARROW_KEYS.get(key)


def _arrow_patch(direction: str, color: int, bg: int) -> np.ndarray:
    mask = ARROW_MASKS[direction]
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = color
    return patch


def _frame_patch(direction: str, color: int, bg: int, h: int, w: int) -> np.ndarray:
    patch = np.full((h, w), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color

    if direction == "up":
        patch[0, 1:-1] = bg
    elif direction == "down":
        patch[-1, 1:-1] = bg
    elif direction == "left":
        patch[1:-1, 0] = bg
    else:
        patch[1:-1, -1] = bg

    return patch


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int) -> None:
    h, w = map(int, patch.shape)
    view = grid[r0 : r0 + h, c0 : c0 + w]
    mask = patch != transparent
    view[mask] = patch[mask]


def _can_place_bbox(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, *, padding: int) -> bool:
    grid_h, grid_w = map(int, occupied.shape)
    if r0 < 0 or c0 < 0 or r0 + h > grid_h or c0 + w > grid_w:
        return False

    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(grid_h, r0 + h + padding)
    cc1 = min(grid_w, c0 + w + padding)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _random_position(occupied: np.ndarray, h: int, w: int, *, padding: int) -> tuple[int, int] | None:
    positions = [(r, c) for r in range(occupied.shape[0] - h + 1) for c in range(occupied.shape[1] - w + 1)]
    random.shuffle(positions)
    for r0, c0 in positions:
        if _can_place_bbox(occupied, r0, c0, h, w, padding=padding):
            return r0, c0
    return None


def _draw_centered_arrow(grid: np.ndarray, frame: dict, *, bg: int) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    direction = frame["direction"]
    color = frame["color"]
    patch = _arrow_patch(direction, color, bg)
    inner_h = int(r1 - r0 - 1)
    inner_w = int(c1 - c0 - 1)
    ph, pw = map(int, patch.shape)
    assert inner_h >= ph and inner_w >= pw, "frame interior must fit its arrow"

    top = int(r0 + 1 + (inner_h - ph) // 2)
    left = int(c0 + 1 + (inner_w - pw) // 2)
    _paste(grid, patch, top, left, transparent=bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each external directional arrow to the frame whose missing side points
    the same way, move that arrow to the frame center, and recolor it to the
    frame color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    components = _find_components(g, bg)

    frames: list[dict] = []
    arrows: list[dict] = []
    for component in components:
        opening = _frame_opening(component)
        if opening is not None:
            frames.append(
                {
                    "bbox": component["bbox"],
                    "direction": opening,
                    "color": int(component["color"]),
                }
            )
            continue

        direction = _arrow_direction(component)
        if direction is not None:
            arrows.append(
                {
                    "coords": component["coords"],
                    "direction": direction,
                }
            )
            continue

        raise AssertionError("found an unrecognized non-background component")

    assert frames, "expected at least one frame"
    assert len(arrows) == len(frames), "arrow count must match frame count"
    assert Counter(frame["direction"] for frame in frames) == Counter(arrow["direction"] for arrow in arrows)

    out = g.copy()
    for arrow in arrows:
        rows = arrow["coords"][:, 0]
        cols = arrow["coords"][:, 1]
        out[rows, cols] = bg

    for frame in frames:
        _draw_centered_arrow(out, frame, bg=bg)

    return out


def _grid_shape(n_frames: int) -> tuple[int, int]:
    row_range, col_range = GRID_RANGES[n_frames]
    return random.randint(*row_range), random.randint(*col_range)


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.randint(2, 4)
    height, width = _grid_shape(n_frames)
    bg = 0

    directions = random.sample(list(DIRECTIONS), n_frames)
    frame_colors = random.sample(FRAME_COLOR_POOL, n_frames)

    inp = np.full((height, width), bg, dtype=int)
    expected = np.full_like(inp, bg)
    occupied = np.zeros((height, width), dtype=bool)
    frames: list[dict] = []

    frame_specs = list(zip(directions, frame_colors))
    random.shuffle(frame_specs)
    for direction, color in frame_specs:
        frame_h = random.randint(*FRAME_HEIGHT_RANGE)
        frame_w = random.randint(*FRAME_WIDTH_RANGE)
        pos = _random_position(occupied, frame_h, frame_w, padding=PLACEMENT_PADDING)
        if pos is None:
            return None
        r0, c0 = pos
        patch = _frame_patch(direction, color, bg, frame_h, frame_w)
        _paste(inp, patch, r0, c0, transparent=bg)
        _paste(expected, patch, r0, c0, transparent=bg)
        occupied[r0 : r0 + frame_h, c0 : c0 + frame_w] = True
        frames.append(
            {
                "bbox": (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1),
                "direction": direction,
                "color": color,
            }
        )

    arrow_directions = directions[:]
    random.shuffle(arrow_directions)
    for direction in arrow_directions:
        patch = _arrow_patch(direction, ARROW_COLOR, bg)
        arrow_h, arrow_w = map(int, patch.shape)
        pos = _random_position(occupied, arrow_h, arrow_w, padding=PLACEMENT_PADDING)
        if pos is None:
            return None
        r0, c0 = pos
        _paste(inp, patch, r0, c0, transparent=bg)
        occupied[r0 : r0 + arrow_h, c0 : c0 + arrow_w] = True

    for frame in frames:
        _draw_centered_arrow(expected, frame, bg=bg)

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
