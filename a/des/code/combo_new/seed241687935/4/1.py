import random
from typing import Iterable

import numpy as np


DIRS = ("up", "down", "left", "right")
DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
BG = 0
MAX_GENERATION_ATTEMPTS = 400
GRID_MIN = 20
GRID_MAX = 30
FRAME_COUNT_RANGE = (2, 4)
LEGEND_SLOT_HEIGHT = 4
LEGEND_TOTAL_HEIGHT = 15
LEGEND_BAND_WIDTH = 6
FRAME_PLACE_RETRIES = 200


def neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def normalize(coords: np.ndarray) -> np.ndarray:
    r0, c0, _, _ = bbox(coords)
    return coords - np.array([r0, c0], dtype=int)


def touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == w - 1)
    )


def is_rect_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = bbox(coords)
    hh = r1 - r0 + 1
    ww = c1 - c0 + 1
    if hh < 3 or ww < 3:
        return False
    if int(coords.shape[0]) != 2 * (hh + ww) - 4:
        return False
    mask = np.zeros((hh, ww), dtype=bool)
    shifted = coords - np.array([r0, c0], dtype=int)
    mask[shifted[:, 0], shifted[:, 1]] = True
    frame = np.zeros((hh, ww), dtype=bool)
    frame[0, :] = True
    frame[-1, :] = True
    frame[:, 0] = True
    frame[:, -1] = True
    return np.array_equal(mask, frame)


def pattern_direction(coords: np.ndarray) -> str:
    pts = normalize(coords)
    cells = {(int(r), int(c)) for r, c in pts}
    degree3: list[tuple[int, int]] = []
    for r, c in cells:
        deg = sum((r + dr, c + dc) in cells for dr, dc in DIR_VECS.values())
        if deg == 3:
            degree3.append((r, c))
    if len(degree3) == 1:
        jr, jc = degree3[0]
        missing = [
            name
            for name, (dr, dc) in DIR_VECS.items()
            if (jr + dr, jc + dc) not in cells
        ]
        if len(missing) == 1:
            return missing[0]

    h = int(pts[:, 0].max()) + 1
    w = int(pts[:, 1].max()) + 1
    side_counts = {
        "up": int(np.sum(pts[:, 0] == 0)),
        "down": int(np.sum(pts[:, 0] == h - 1)),
        "left": int(np.sum(pts[:, 1] == 0)),
        "right": int(np.sum(pts[:, 1] == w - 1)),
    }
    min_count = min(side_counts.values())
    best = [name for name, count in side_counts.items() if count == min_count]
    assert len(best) == 1, "pattern direction is ambiguous"
    return best[0]


def infer_frame_color(grid: np.ndarray) -> int:
    frame_scores: dict[int, tuple[int, int]] = {}
    for color in np.unique(grid):
        color = int(color)
        if color == BG:
            continue
        count = 0
        area = 0
        for comp in components_mask(grid == color):
            if is_rect_frame(comp):
                count += 1
                area += int(comp.shape[0])
        if count:
            frame_scores[color] = (count, area)
    assert frame_scores, "expected at least one hollow frame color"
    return max(frame_scores, key=lambda color: frame_scores[color])


def legend_mapping(grid: np.ndarray, frame_color: int) -> tuple[dict[str, int], set[tuple[int, int]]]:
    mapping: dict[str, int] = {}
    legend_cells: set[tuple[int, int]] = set()
    for comp in components_mask(grid != BG):
        if not touches_edge(comp, grid.shape):
            continue
        colors = {int(grid[r, c]) for r, c in comp}
        if colors == {frame_color} or frame_color not in colors or len(colors) != 2:
            continue
        target_colors = colors - {frame_color}
        if len(target_colors) != 1:
            continue
        target_color = next(iter(target_colors))
        target_cells = [(int(r), int(c)) for r, c in comp if int(grid[r, c]) == target_color]
        if len(target_cells) != 1:
            continue
        pattern = np.array(
            [(int(r), int(c)) for r, c in comp if int(grid[r, c]) == frame_color],
            dtype=int,
        )
        if pattern.size == 0:
            continue
        direction = pattern_direction(pattern)
        assert direction not in mapping, "legend directions must be unique"
        mapping[direction] = target_color
        legend_cells.update((int(r), int(c)) for r, c in comp)
    assert set(mapping) == set(DIRS), "legend must cover all four directions"
    return mapping, legend_cells


def find_frames(grid: np.ndarray, frame_color: int) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in components_mask(grid == frame_color):
        if is_rect_frame(comp):
            frames.append(bbox(comp))
    assert frames, "expected at least one rectangular frame"
    return frames


def interior_pattern_direction(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], frame_color: int) -> str:
    r0, c0, r1, c1 = frame_bbox
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    mask = (inner != BG) & (inner != frame_color)
    comps = components_mask(mask)
    assert len(comps) == 1, "each frame must contain exactly one inner pattern"
    return pattern_direction(comps[0])


def paint_frame_border(out: np.ndarray, frame_bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = frame_bbox
    out[r0, c0 : c1 + 1] = color
    out[r1, c0 : c1 + 1] = color
    out[r0 : r1 + 1, c0] = color
    out[r0 : r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each hollow frame border using the direction legend encoded by small interior patterns.

    Input structure:
    - The background is black.
    - Several same-colored hollow rectangular frames appear away from the grid edge.
    - Each frame contains one smaller directional pattern in another color, separated from the border by background.
    - The left edge holds four small legend items; each item is a gray directional thumbnail connected to one color pixel on its right.

    Rule:
    - Read the four edge legend items and infer a mapping from pattern opening direction to output color.
    - For each hollow frame, identify the opening direction of its inner pattern.
    - Recolor the entire frame border with the legend color for that direction, keeping the inner pattern unchanged.
    - Erase the legend area back to black in the output.

    Key insight:
    - The inner symbol is a lookup key, not decoration: its missing arm points to the legend entry that tells the frame which color to become.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_color = infer_frame_color(g)
    mapping, legend_cells = legend_mapping(g, frame_color)

    out = g.copy()
    for r, c in legend_cells:
        out[r, c] = BG

    for frame_bbox in find_frames(g, frame_color):
        direction = interior_pattern_direction(g, frame_bbox, frame_color)
        paint_frame_border(out, frame_bbox, mapping[direction])

    return out


def make_t_pattern(direction: str, arm_lengths: dict[str, int]) -> np.ndarray:
    coords = [(0, 0)]
    for name, (dr, dc) in DIR_VECS.items():
        length = 0 if name == direction else int(arm_lengths[name])
        for step in range(1, length + 1):
            coords.append((dr * step, dc * step))
    arr = np.array(coords, dtype=int)
    return normalize(arr)


def shift(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    return coords + np.array([dr, dc], dtype=int)


def rightmost_attachment_row(coords: np.ndarray) -> int:
    rightmost = int(coords[:, 1].max())
    rows = sorted(int(r) for r, c in coords if int(c) == rightmost)
    return rows[len(rows) // 2]


def frame_spec(rng: random.Random, direction: str) -> dict[str, object]:
    max_arm = 3 if rng.random() < 0.25 else 2
    arm_lengths = {
        name: (0 if name == direction else rng.randint(1, max_arm))
        for name in DIRS
    }
    pattern = make_t_pattern(direction, arm_lengths)
    ph = int(pattern[:, 0].max()) + 1
    pw = int(pattern[:, 1].max()) + 1
    pad_top = rng.randint(1, 2)
    pad_bottom = rng.randint(1, 2)
    pad_left = rng.randint(1, 2)
    pad_right = rng.randint(1, 2)
    outer_h = ph + pad_top + pad_bottom + 2
    outer_w = pw + pad_left + pad_right + 2
    placed_pattern = shift(pattern, 1 + pad_top, 1 + pad_left)
    return {
        "direction": direction,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "pattern": placed_pattern,
    }


def rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def draw_legend(
    grid: np.ndarray,
    rng: random.Random,
    frame_color: int,
    mapping: dict[str, int],
) -> None:
    h, _ = grid.shape
    top = rng.randint(0, h - LEGEND_TOTAL_HEIGHT)
    order = list(DIRS)
    rng.shuffle(order)
    for idx, direction in enumerate(order):
        pattern = make_t_pattern(direction, {name: 1 for name in DIRS})
        ph = int(pattern[:, 0].max()) + 1
        row0 = top + idx * LEGEND_SLOT_HEIGHT
        if ph == 2 and rng.random() < 0.5:
            row0 += 1
        grid[pattern[:, 0] + row0, pattern[:, 1]] = frame_color
        attach_row = row0 + rightmost_attachment_row(pattern)
        color_col = int(pattern[:, 1].max()) + 1
        grid[attach_row, color_col] = mapping[direction]


def place_frames(
    grid: np.ndarray,
    rng: random.Random,
    specs: list[dict[str, object]],
    frame_color: int,
    pattern_color: int,
) -> bool:
    h, w = grid.shape
    occupied: list[tuple[int, int, int, int]] = []
    for spec in sorted(specs, key=lambda item: int(item["outer_h"]) * int(item["outer_w"]), reverse=True):
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        if h - outer_h - 1 < 1 or w - outer_w - 1 < LEGEND_BAND_WIDTH:
            return False
        placed = False
        for _ in range(FRAME_PLACE_RETRIES):
            r0 = rng.randint(1, h - outer_h - 1)
            c0 = rng.randint(LEGEND_BAND_WIDTH, w - outer_w - 1)
            rect = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if any(rects_overlap(rect, other, gap=1) for other in occupied):
                continue
            occupied.append(rect)
            placed = True

            r1, c1 = rect[2], rect[3]
            grid[r0, c0 : c1 + 1] = frame_color
            grid[r1, c0 : c1 + 1] = frame_color
            grid[r0 : r1 + 1, c0] = frame_color
            grid[r0 : r1 + 1, c1] = frame_color

            pattern = shift(np.asarray(spec["pattern"], dtype=int), r0, c0)
            grid[pattern[:, 0], pattern[:, 1]] = pattern_color
            break
        if not placed:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    colors = rng.sample(list(range(1, 10)), 6)
    frame_color = colors[0]
    pattern_color = colors[1]
    mapping = dict(zip(DIRS, colors[2:], strict=True))

    frame_count = rng.randint(*FRAME_COUNT_RANGE)
    frame_dirs = rng.sample(list(DIRS), frame_count)
    specs = [frame_spec(rng, direction) for direction in frame_dirs]

    min_h = max(GRID_MIN, max(int(spec["outer_h"]) for spec in specs) + 3, LEGEND_TOTAL_HEIGHT)
    min_w = max(GRID_MIN, max(int(spec["outer_w"]) for spec in specs) + LEGEND_BAND_WIDTH + 2)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    h = rng.randint(min_h, GRID_MAX)
    w = rng.randint(min_w, GRID_MAX)
    grid = np.zeros((h, w), dtype=int)

    draw_legend(grid, rng, frame_color, mapping)
    if not place_frames(grid, rng, specs, frame_color, pattern_color):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
