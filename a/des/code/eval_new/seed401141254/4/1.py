import random

import numpy as np


BG = 0
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
ALL_DIRECTIONS = ("top", "bottom", "left", "right")

FRAME_PATTERN_MASKS = [
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
]

SCATTER_MASKS = [
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 1, 1]], dtype=bool),
]

GEN_RETRIES = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _find_direction_markers(grid: np.ndarray, bg: int) -> dict[str, int]:
    h, w = grid.shape
    markers: dict[str, int] = {}
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) == bg:
                continue
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            isolated = True
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != bg:
                    isolated = False
                    break
            if not isolated:
                continue
            if r == 0 and 0 < c < w - 1:
                markers["top"] = int(grid[r, c])
            elif r == h - 1 and 0 < c < w - 1:
                markers["bottom"] = int(grid[r, c])
            elif c == 0 and 0 < r < h - 1:
                markers["left"] = int(grid[r, c])
            elif c == w - 1 and 0 < r < h - 1:
                markers["right"] = int(grid[r, c])
    return markers


def _detect_frames(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in (int(v) for v in np.unique(grid) if int(v) != bg):
        for comp in _components(grid == color):
            if len(comp) < 16:
                continue
            r0, c0, r1, c1 = _bbox(comp)
            size = r1 - r0 + 1
            if size != c1 - c0 + 1 or size < 5:
                continue

            local = np.zeros((size, size), dtype=bool)
            local[comp[:, 0] - r0, comp[:, 1] - c0] = True
            border = np.zeros_like(local)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True

            if np.any(local & ~border):
                continue

            missing = np.argwhere(border & ~local)
            if len(missing) != 1:
                continue

            mr, mc = map(int, missing[0])
            if (mr in (0, size - 1)) and (mc in (0, size - 1)):
                continue

            if mr == 0:
                gap_dir = "top"
            elif mr == size - 1:
                gap_dir = "bottom"
            elif mc == 0:
                gap_dir = "left"
            else:
                gap_dir = "right"

            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            inner_colors = set(map(int, np.unique(interior))) - {bg, color}
            if len(inner_colors) != 1:
                continue

            frames.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "gap_dir": gap_dir,
                    "inner_color": inner_colors.pop(),
                }
            )
    return frames


def _inside_any_frame(coords: np.ndarray, frames: list[dict[str, object]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    for frame in frames:
        fr0, fc0, fr1, fc1 = frame["bbox"]  # type: ignore[index]
        if fr0 < r0 and r1 < fr1 and fc0 < c0 and c1 < fc1:
            return True
    return False


def _is_border_singleton(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    if len(coords) != 1:
        return False
    r, c = map(int, coords[0])
    h, w = shape
    return r in (0, h - 1) or c in (0, w - 1)


def main(grid: np.ndarray) -> np.ndarray:
    """Use frame-gap directions to build a color legend, then recolor the scattered monochrome sprites.

    Input structure:
    - The background is black. Several large hollow square frames appear in the grid; each frame is monochrome and has exactly one one-cell gap on one side.
    - Each frame encloses a small centered monochrome pattern whose color differs from the frame color.
    - Four isolated single-pixel markers sit on the top, bottom, left, and right borders, giving a direction-to-color reference.
    - Elsewhere in the grid are small scattered monochrome sprites whose colors may match the frame-interior colors.

    Rule:
    - Detect each frame and determine which side contains its missing border cell.
    - Read the border marker on that same direction to obtain a mapping from the frame's interior-pattern color to the marker color.
    - Recolor every scattered sprite whose color is a key in that mapping, keeping its shape and position unchanged.
    - Remove frames, interior guide patterns, and border markers; the output keeps only the recolored scattered sprites on black.

    Key insight:
    - The gap direction does not directly tell the recolor target; it indexes the border marker, while the enclosed color supplies the lookup key used on the distant scattered sprites.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    direction_markers = _find_direction_markers(g, bg)
    frames = _detect_frames(g, bg)

    recolor_map: dict[int, int] = {}
    for frame in frames:
        gap_dir = frame["gap_dir"]  # type: ignore[index]
        inner_color = int(frame["inner_color"])  # type: ignore[index]
        if gap_dir in direction_markers:
            recolor_map[inner_color] = int(direction_markers[gap_dir])

    out = np.full_like(g, bg)
    for color in (int(v) for v in np.unique(g) if int(v) != bg):
        for comp in _components(g == color):
            if _is_border_singleton(comp, g.shape):
                continue
            if _inside_any_frame(comp, frames):
                continue
            if color in recolor_map:
                out[comp[:, 0], comp[:, 1]] = recolor_map[color]
    return out


def _paint_mask(grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    patch = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    patch[mask] = color


def _draw_frame(grid: np.ndarray, top: int, left: int, size: int, color: int, gap_dir: str, rng: random.Random) -> None:
    grid[top, left : left + size] = color
    grid[top + size - 1, left : left + size] = color
    grid[top : top + size, left] = color
    grid[top : top + size, left + size - 1] = color

    if gap_dir == "top":
        gap = (top, rng.randint(left + 1, left + size - 2))
    elif gap_dir == "bottom":
        gap = (top + size - 1, rng.randint(left + 1, left + size - 2))
    elif gap_dir == "left":
        gap = (rng.randint(top + 1, top + size - 2), left)
    else:
        gap = (rng.randint(top + 1, top + size - 2), left + size - 1)
    grid[gap] = BG


def _free_with_margin(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = occupied.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + height + 1)
    c1 = min(w, left + width + 1)
    return not bool(occupied[r0:r1, c0:c1].any())


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(26, 30)
    w = rng.randint(26, 30)
    grid = np.full((h, w), BG, dtype=int)

    palette = list(range(1, 10))
    rng.shuffle(palette)
    marker_colors = palette[:4]
    inner_colors = palette[4:8]
    frame_color = palette[8]
    direction_colors = dict(zip(ALL_DIRECTIONS, marker_colors))

    grid[0, rng.randint(2, w - 3)] = direction_colors["top"]
    grid[h - 1, rng.randint(2, w - 3)] = direction_colors["bottom"]
    grid[rng.randint(2, h - 3), 0] = direction_colors["left"]
    grid[rng.randint(2, h - 3), w - 1] = direction_colors["right"]

    mid_r, mid_c = h // 2, w // 2
    quadrants = [
        (2, mid_r, 2, mid_c),
        (2, mid_r, mid_c + 1, w - 2),
        (mid_r + 1, h - 2, 2, mid_c),
        (mid_r + 1, h - 2, mid_c + 1, w - 2),
    ]

    gap_dirs = list(ALL_DIRECTIONS)
    rng.shuffle(gap_dirs)

    frame_boxes: list[tuple[int, int, int, int]] = []
    inner_to_marker: dict[int, int] = {}

    for idx, (rmin, rmax, cmin, cmax) in enumerate(quadrants):
        size = rng.randint(7, 9)
        top_lo = rmin
        top_hi = rmax - size - 1
        left_lo = cmin
        left_hi = cmax - size - 1
        if top_lo > top_hi or left_lo > left_hi:
            return None

        top = rng.randint(top_lo, top_hi)
        left = rng.randint(left_lo, left_hi)
        gap_dir = gap_dirs[idx]

        _draw_frame(grid, top, left, size, frame_color, gap_dir, rng)

        pattern_mask = rng.choice(FRAME_PATTERN_MASKS)
        pattern_top = top + 1 + (size - 5) // 2
        pattern_left = left + 1 + (size - 5) // 2
        inner_color = inner_colors[idx]
        _paint_mask(grid, pattern_mask, inner_color, pattern_top, pattern_left)

        frame_boxes.append((top, left, top + size - 1, left + size - 1))
        inner_to_marker[inner_color] = direction_colors[gap_dir]

    blocked = np.zeros_like(grid, dtype=bool)
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True
    for r0, c0, r1, c1 in frame_boxes:
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True

    occupied = grid != BG
    scatter_masks = SCATTER_MASKS.copy()
    rng.shuffle(scatter_masks)
    scatter_specs: list[tuple[int, np.ndarray, int, int]] = []

    for inner_color, mask in zip(inner_colors, scatter_masks[:4]):
        placed = False
        for _ in range(400):
            top = rng.randint(1, h - 4)
            left = rng.randint(1, w - 4)
            if bool(blocked[top : top + 3, left : left + 3].any()):
                continue
            if not _free_with_margin(occupied, top, left, 3, 3):
                continue
            _paint_mask(grid, mask, inner_color, top, left)
            occupied[top : top + 3, left : left + 3][mask] = True
            scatter_specs.append((inner_color, mask.copy(), top, left))
            placed = True
            break
        if not placed:
            return None

    if _get_bg(grid) != BG:
        return None

    expected = np.full_like(grid, BG)
    for inner_color, mask, top, left in scatter_specs:
        _paint_mask(expected, mask, inner_to_marker[inner_color], top, left)

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
