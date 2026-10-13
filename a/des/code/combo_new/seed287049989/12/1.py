from __future__ import annotations

import random

import numpy as np


_DIRECTIONS = ("up", "down", "left", "right")
_ARROW_MASKS: dict[str, np.ndarray] = {
    "up": np.array(
        [
            [0, 1, 0],
            [0, 1, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    "down": np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    "left": np.array(
        [
            [0, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
        ],
        dtype=bool,
    ),
    "right": np.array(
        [
            [1, 0, 0],
            [1, 1, 1],
            [1, 0, 0],
        ],
        dtype=bool,
    ),
}

_GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int], str]:
    candidates: list[tuple[int, tuple[int, int, int, int], str]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        pts = np.argwhere(grid == color)
        if pts.size == 0:
            continue
        r0, c0 = map(int, pts.min(axis=0))
        r1, c1 = map(int, pts.max(axis=0))
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 3 or width < 3:
            continue

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = _border_mask(height, width)
        if np.any(sub[~border] == color):
            continue
        if not np.all(sub[~border] == bg):
            continue

        gap_pts = np.argwhere(border & (sub == bg))
        if gap_pts.shape[0] != 1:
            continue
        if int(np.count_nonzero(sub == color)) != int(border.sum()) - 1:
            continue

        gr, gc = map(int, gap_pts[0])
        sides = [
            ("up", gr == 0),
            ("down", gr == height - 1),
            ("left", gc == 0),
            ("right", gc == width - 1),
        ]
        active = [name for name, present in sides if present]
        if len(active) != 1:
            continue
        candidates.append((color, (r0, c0, r1, c1), active[0]))

    assert len(candidates) == 1, f"expected one gapped rectangular frame, found {len(candidates)}"
    return candidates[0]


def _arrow_direction(mask: np.ndarray) -> str | None:
    for direction, ref in _ARROW_MASKS.items():
        if np.array_equal(mask, ref):
            return direction
    return None


def _find_arrows(
    grid: np.ndarray, bg: int, frame_color: int, frame_bbox: tuple[int, int, int, int]
) -> dict[str, tuple[int, np.ndarray]]:
    fr0, fc0, fr1, fc1 = frame_bbox
    arrow_map: dict[str, tuple[int, np.ndarray]] = {}
    for color in np.unique(grid):
        color = int(color)
        if color in (bg, frame_color):
            continue

        pts = np.argwhere(grid == color)
        r0, c0 = map(int, pts.min(axis=0))
        r1, c1 = map(int, pts.max(axis=0))
        if not (r1 < fr0 or r0 > fr1 or c1 < fc0 or c0 > fc1):
            raise AssertionError("arrow colors must stay outside the frame")
        if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
            continue

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any((sub != bg) & (sub != color)):
            continue
        if int(np.count_nonzero(sub == color)) != 5:
            continue

        direction = _arrow_direction(sub == color)
        if direction is None:
            continue
        assert direction not in arrow_map, f"duplicate arrow direction {direction}"
        arrow_map[direction] = (color, pts)

    assert set(arrow_map) == set(_DIRECTIONS), f"expected one arrow per direction, found {sorted(arrow_map)}"
    return arrow_map


def _paint_edge(out: np.ndarray, bbox: tuple[int, int, int, int], side: str, color: int) -> None:
    r0, c0, r1, c1 = bbox
    if side == "up":
        out[r0, c0 : c1 + 1] = color
    elif side == "down":
        out[r1, c0 : c1 + 1] = color
    elif side == "left":
        out[r0 : r1 + 1, c0] = color
    elif side == "right":
        out[r0 : r1 + 1, c1] = color
    else:
        raise ValueError(side)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the gapped side of a rectangular frame using the color of the outside arrow that points in the same direction.

    Input structure:
    - The background is the most common color.
    - Near the center there is one monochrome hollow rectangular frame with one
      background-colored one-pixel gap on exactly one side.
    - Outside the frame there are exactly four 3x3 monochrome T-shaped arrow
      sprites, one each for up/down/left/right, and all four arrow colors differ.

    Rule:
    - Detect which side of the frame contains the gap; that side determines a
      direction: top=up, bottom=down, left=left, right=right.
    - Find the outside arrow sprite whose orientation matches that direction.
    - Recolor the entire corresponding frame side, including the missing gap
      cell, to that arrow's color, keep the other three frame sides unchanged,
      and erase all outside arrows back to background.

    Key insight:
    - The arrow orientation is not a motion command; it is a direction label
      that must be matched to the frame side containing the unique gap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    frame_color, frame_bbox, gap_side = _find_frame(g, bg)
    arrows = _find_arrows(g, bg, frame_color, frame_bbox)

    out = g.copy()
    for _direction, (_color, pts) in arrows.items():
        out[pts[:, 0], pts[:, 1]] = bg
    _paint_edge(out, frame_bbox, gap_side, arrows[gap_side][0])
    return out


def _arrow_patch(bg: int, color: int, direction: str) -> np.ndarray:
    patch = np.full((3, 3), bg, dtype=int)
    patch[_ARROW_MASKS[direction]] = color
    return patch


def _frame_patch(
    bg: int,
    frame_color: int,
    height: int,
    width: int,
    gap_side: str,
    gap_index: int,
) -> np.ndarray:
    patch = np.full((height, width), bg, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    if gap_side == "up":
        patch[0, gap_index] = bg
    elif gap_side == "down":
        patch[-1, gap_index] = bg
    elif gap_side == "left":
        patch[gap_index, 0] = bg
    elif gap_side == "right":
        patch[gap_index, -1] = bg
    else:
        raise ValueError(gap_side)
    return patch


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _random_arrow_positions(
    grid_shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    height, width = grid_shape
    fr0, fc0, fr1, fc1 = frame_bbox
    expanded_frame = (max(0, fr0 - 1), max(0, fc0 - 1), min(height - 1, fr1 + 1), min(width - 1, fc1 + 1))

    candidates: list[tuple[int, int]] = []
    for r0 in range(1, height - 3):
        for c0 in range(1, width - 3):
            bbox = (r0, c0, r0 + 2, c0 + 2)
            if _rectangles_overlap(bbox, expanded_frame):
                continue
            candidates.append((r0, c0))

    random.shuffle(candidates)
    picked: list[tuple[int, int]] = []
    used: list[tuple[int, int, int, int]] = []
    for r0, c0 in candidates:
        bbox = (r0 - 1, c0 - 1, r0 + 3, c0 + 3)
        if any(_rectangles_overlap(bbox, other) for other in used):
            continue
        picked.append((r0, c0))
        used.append(bbox)
        if len(picked) == 4:
            return picked
    raise ValueError("failed to place four separated arrows")


def _generate_case() -> np.ndarray | None:
    height = random.randint(14, 30)
    width = random.randint(14, 30)

    palette = list(range(10))
    bg = random.choice(palette)
    others = [color for color in palette if color != bg]
    random.shuffle(others)
    frame_color = others[0]
    arrow_colors = others[1:5]

    max_frame_h = min(18, height - 8)
    max_frame_w = min(18, width - 8)
    if max_frame_h < 5 or max_frame_w < 5:
        return None
    frame_h = random.randint(5, max_frame_h)
    frame_w = random.randint(5, max_frame_w)

    frame_r0 = random.randint(4, height - frame_h - 4)
    frame_c0 = random.randint(4, width - frame_w - 4)
    frame_bbox = (frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1)

    gap_side = random.choice(_DIRECTIONS)
    if gap_side in ("up", "down"):
        gap_index = random.randint(1, frame_w - 2)
    else:
        gap_index = random.randint(1, frame_h - 2)

    grid = np.full((height, width), bg, dtype=int)
    frame = _frame_patch(bg, frame_color, frame_h, frame_w, gap_side, gap_index)
    grid[frame_r0 : frame_r0 + frame_h, frame_c0 : frame_c0 + frame_w] = frame

    try:
        positions = _random_arrow_positions((height, width), frame_bbox)
    except ValueError:
        return None
    directions = list(_DIRECTIONS)
    random.shuffle(directions)
    for (r0, c0), direction, color in zip(positions, directions, arrow_colors):
        patch = _arrow_patch(bg, color, direction)
        sub = grid[r0 : r0 + 3, c0 : c0 + 3]
        if np.any(sub != bg):
            return None
        grid[r0 : r0 + 3, c0 : c0 + 3] = patch

    if _get_bg(grid) != bg:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    expected = grid.copy()
    for color in arrow_colors:
        expected[expected == color] = bg
    _paint_edge(expected, frame_bbox, gap_side, arrow_colors[directions.index(gap_side)])

    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
