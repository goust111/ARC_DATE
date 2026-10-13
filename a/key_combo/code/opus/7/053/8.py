import random
from typing import Any

import numpy as np


BLACK = 0
RED = 2
ORANGE = 7
FRAME_COLOR_POOL = (1, 3, 4, 5, 6, 8, 9)
FRAME_SIZE_CHOICES = (5, 7)
MAX_GENERATION_ATTEMPTS = 400


def _frame_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = box
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _find_frames(grid: np.ndarray) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for color in sorted(int(v) for v in np.unique(grid)):
        if color in (BLACK, RED, ORANGE):
            continue
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 3 or w < 3:
            continue
        box = (int(r0), int(c0), int(r1), int(c1))
        perimeter = _frame_mask(grid.shape, box)
        if int(coords.shape[0]) != int(perimeter.sum()):
            continue
        if not np.all(grid[perimeter] == color):
            continue
        if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == color):
            continue
        frames.append(
            {
                "color": color,
                "box": box,
                "center_row2": int(r0 + r1),
                "center_col2": int(c0 + c1),
            }
        )
    frames.sort(key=lambda frame: frame["box"])
    return frames


def _swap_interior_and_majority(out: np.ndarray, box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    interior = out[r0 + 1 : r1, c0 + 1 : c1]
    fg_mask = (interior == RED) | (interior == ORANGE)
    if int(fg_mask.sum()) % 2 == 1:
        red_mask = interior == RED
        orange_mask = interior == ORANGE
        interior[red_mask] = ORANGE
        interior[orange_mask] = RED
    red_count = int(np.count_nonzero(interior == RED))
    orange_count = int(np.count_nonzero(interior == ORANGE))
    return RED if red_count >= orange_count else ORANGE


def _distance_to_topleft2(frame: dict[str, Any]) -> int:
    return int(frame["center_row2"] + frame["center_col2"])


def _draw_horizontal_bridge(
    out: np.ndarray,
    left: dict[str, Any],
    right: dict[str, Any],
    color: int,
) -> None:
    lr0, _lc0, lr1, lc1 = left["box"]
    _rr0, rc0, _rr1, _rc1 = right["box"]
    row = (left["center_row2"]) // 2
    for c in range(lc1 + 1, rc0):
        if out[row, c] == BLACK:
            out[row, c] = color


def _draw_vertical_bridge(
    out: np.ndarray,
    top: dict[str, Any],
    bottom: dict[str, Any],
    color: int,
) -> None:
    _tr0, tc0, tr1, _tc1 = top["box"]
    br0, _bc0, _br1, _bc1 = bottom["box"]
    col = (top["center_col2"]) // 2
    for r in range(tr1 + 1, br0):
        if out[r, col] == BLACK:
            out[r, col] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap red/orange interior pixels in odd-populated frames, then bridge aligned frame pairs using the nearer frame's post-swap majority color.

    Input structure:
    - The background is black.
    - The grid contains 4-6 one-cell-thick rectangular frames, each in a unique non-red/non-orange color.
    - Each frame interior contains black pixels plus a red/orange pattern, and some frame pairs share a center row or center column.

    Rule:
    - Detect every rectangular frame and count the non-black interior pixels in that frame.
    - If the count is odd, swap red and orange inside that frame; if it is even, leave the interior unchanged.
    - After that swap decision, compute the frame's majority interior color between red and orange, breaking ties in favor of red.
    - For every pair of frames with the same center row, draw a horizontal bridge across the black gap between their nearest vertical sides; for every pair with the same center column, draw the analogous vertical bridge.
    - The bridge color comes from the aligned pair member whose center is closer to the top-left corner by Manhattan distance, and bridges only occupy black background cells.

    Key insight:
    - The bridge color is not encoded by the frame border: it depends on a three-step chain of interior pixel counting, parity-gated color swapping, and post-swap majority-color selection.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = _find_frames(g)

    majority_by_box: dict[tuple[int, int, int, int], int] = {}
    for frame in frames:
        majority_by_box[frame["box"]] = _swap_interior_and_majority(out, frame["box"])

    for i, first in enumerate(frames):
        for second in frames[i + 1 :]:
            if first["center_row2"] == second["center_row2"]:
                left, right = sorted((first, second), key=lambda frame: frame["box"][1])
                anchor = min(
                    (first, second),
                    key=lambda frame: (_distance_to_topleft2(frame), frame["box"]),
                )
                _draw_horizontal_bridge(out, left, right, majority_by_box[anchor["box"]])
            elif first["center_col2"] == second["center_col2"]:
                top, bottom = sorted((first, second), key=lambda frame: frame["box"][0])
                anchor = min(
                    (first, second),
                    key=lambda frame: (_distance_to_topleft2(frame), frame["box"]),
                )
                _draw_vertical_bridge(out, top, bottom, majority_by_box[anchor["box"]])

    return out


def _boxes_too_close(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
    gap: int = 1,
) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _sample_frame_size() -> tuple[int, int]:
    return random.choice(FRAME_SIZE_CHOICES), random.choice(FRAME_SIZE_CHOICES)


def _build_box_from_center(center_r: int, center_c: int, h: int, w: int) -> tuple[int, int, int, int]:
    return (center_r - h // 2, center_c - w // 2, center_r + h // 2, center_c + w // 2)


def _can_place(
    box: tuple[int, int, int, int],
    existing: list[dict[str, Any]],
    shape: tuple[int, int],
    min_gap: int = 1,
) -> bool:
    r0, c0, r1, c1 = box
    h, w = shape
    if r0 < 1 or c0 < 1 or r1 >= h - 1 or c1 >= w - 1:
        return False
    return all(not _boxes_too_close(box, frame["box"], gap=min_gap) for frame in existing)


def _center_sets(frames: list[dict[str, Any]]) -> tuple[set[int], set[int]]:
    return (
        {frame["center_row2"] for frame in frames},
        {frame["center_col2"] for frame in frames},
    )


def _place_horizontal_pair(
    h: int,
    w: int,
    frames: list[dict[str, Any]],
    row_range: tuple[int, int],
    pair_right_edge_max: int | None = None,
) -> list[dict[str, Any]] | None:
    used_rows, used_cols = _center_sets(frames)
    for _ in range(200):
        h1, w1 = _sample_frame_size()
        h2, w2 = _sample_frame_size()
        row_min = max(row_range[0], 1 + max(h1, h2) // 2)
        row_max = min(row_range[1], h - 2 - max(h1, h2) // 2)
        if row_min > row_max:
            continue
        center_r = random.randint(row_min, row_max)
        if 2 * center_r in used_rows:
            continue

        gap = random.randint(2, 5)
        left_c0_min = 1
        left_c0_max = w - 2 - (w1 + gap + w2)
        if pair_right_edge_max is not None:
            left_c0_max = min(left_c0_max, pair_right_edge_max - (w1 + gap + w2 - 1))
        if left_c0_min > left_c0_max:
            continue
        left_c0 = random.randint(left_c0_min, left_c0_max)
        right_c0 = left_c0 + w1 + gap

        left = {
            "box": (center_r - h1 // 2, left_c0, center_r + h1 // 2, left_c0 + w1 - 1),
            "center_row2": 2 * center_r,
            "center_col2": 2 * left_c0 + w1 - 1,
        }
        right = {
            "box": (center_r - h2 // 2, right_c0, center_r + h2 // 2, right_c0 + w2 - 1),
            "center_row2": 2 * center_r,
            "center_col2": 2 * right_c0 + w2 - 1,
        }
        if left["center_col2"] in used_cols or right["center_col2"] in used_cols:
            continue
        if _distance_to_topleft2(left) == _distance_to_topleft2(right):
            continue
        if not _can_place(left["box"], frames, (h, w), min_gap=1):
            continue
        if not _can_place(right["box"], frames + [left], (h, w), min_gap=1):
            continue
        return [left, right]
    return None


def _place_vertical_pair(
    h: int,
    w: int,
    frames: list[dict[str, Any]],
    col_range: tuple[int, int],
    pair_top_edge_min: int | None = None,
) -> list[dict[str, Any]] | None:
    used_rows, used_cols = _center_sets(frames)
    for _ in range(200):
        h1, w1 = _sample_frame_size()
        h2, w2 = _sample_frame_size()
        col_min = max(col_range[0], 1 + max(w1, w2) // 2)
        col_max = min(col_range[1], w - 2 - max(w1, w2) // 2)
        if col_min > col_max:
            continue
        center_c = random.randint(col_min, col_max)
        if 2 * center_c in used_cols:
            continue

        gap = random.randint(2, 5)
        top_r0_min = 1
        top_r0_max = h - 2 - (h1 + gap + h2)
        if pair_top_edge_min is not None:
            top_r0_min = max(top_r0_min, pair_top_edge_min)
        if top_r0_min > top_r0_max:
            continue
        top_r0 = random.randint(top_r0_min, top_r0_max)
        bottom_r0 = top_r0 + h1 + gap

        top = {
            "box": (top_r0, center_c - w1 // 2, top_r0 + h1 - 1, center_c + w1 // 2),
            "center_row2": 2 * top_r0 + h1 - 1,
            "center_col2": 2 * center_c,
        }
        bottom = {
            "box": (bottom_r0, center_c - w2 // 2, bottom_r0 + h2 - 1, center_c + w2 // 2),
            "center_row2": 2 * bottom_r0 + h2 - 1,
            "center_col2": 2 * center_c,
        }
        if top["center_row2"] in used_rows or bottom["center_row2"] in used_rows:
            continue
        if _distance_to_topleft2(top) == _distance_to_topleft2(bottom):
            continue
        if not _can_place(top["box"], frames, (h, w), min_gap=1):
            continue
        if not _can_place(bottom["box"], frames + [top], (h, w), min_gap=1):
            continue
        return [top, bottom]
    return None


def _place_extra_frame(h: int, w: int, frames: list[dict[str, Any]]) -> dict[str, Any] | None:
    used_rows, used_cols = _center_sets(frames)
    for _ in range(300):
        fh, fw = _sample_frame_size()
        center_r = random.randint(1 + fh // 2, h - 2 - fh // 2)
        center_c = random.randint(1 + fw // 2, w - 2 - fw // 2)
        if 2 * center_r in used_rows or 2 * center_c in used_cols:
            continue
        box = _build_box_from_center(center_r, center_c, fh, fw)
        frame = {
            "box": box,
            "center_row2": 2 * center_r,
            "center_col2": 2 * center_c,
        }
        if _can_place(box, frames, (h, w), min_gap=1):
            return frame
    return None


def _fill_frame_interior(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    target_parity: int,
) -> None:
    r0, c0, r1, c1 = box
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    area = len(cells)
    min_fg = 3 if target_parity == 1 else 2
    possible_totals = [n for n in range(min_fg, area + 1) if n % 2 == target_parity]
    total_fg = random.choice(possible_totals)

    selected = random.sample(cells, total_fg)
    red_count = random.randint(1, total_fg - 1)
    orange_count = total_fg - red_count
    colors = [RED] * red_count + [ORANGE] * orange_count
    random.shuffle(colors)
    for (r, c), color in zip(selected, colors):
        grid[r, c] = color

    if area > total_fg:
        black_noise = random.sample([cell for cell in cells if cell not in set(selected)], random.randint(0, area - total_fg))
        for r, c in black_noise:
            grid[r, c] = BLACK

    assert inner_h >= 3 and inner_w >= 3
    assert any(grid[r, c] == RED for r, c in selected)
    assert any(grid[r, c] == ORANGE for r, c in selected)


def _paint_frame(
    grid: np.ndarray,
    frame: dict[str, Any],
    frame_color: int,
    target_parity: int,
) -> None:
    r0, c0, r1, c1 = frame["box"]
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color
    _fill_frame_interior(grid, frame["box"], target_parity)


def _aligned_pair_count(frames: list[dict[str, Any]]) -> int:
    count = 0
    for i, first in enumerate(frames):
        for second in frames[i + 1 :]:
            if first["center_row2"] == second["center_row2"] or first["center_col2"] == second["center_col2"]:
                count += 1
    return count


def _generate_layout() -> list[dict[str, Any]] | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    total_frames = random.randint(4, 6)
    extras_needed = total_frames - 4
    mode = random.choice(("horizontal_only", "vertical_only", "mixed"))

    frames: list[dict[str, Any]] = []
    if mode == "horizontal_only":
        pair_a = _place_horizontal_pair(h, w, frames, (3, h // 2 - 3))
        if pair_a is None:
            return None
        frames.extend(pair_a)
        pair_b = _place_horizontal_pair(h, w, frames, (h // 2 + 2, h - 4))
        if pair_b is None:
            return None
        frames.extend(pair_b)
    elif mode == "vertical_only":
        pair_a = _place_vertical_pair(h, w, frames, (3, w // 2 - 3))
        if pair_a is None:
            return None
        frames.extend(pair_a)
        pair_b = _place_vertical_pair(h, w, frames, (w // 2 + 2, w - 4))
        if pair_b is None:
            return None
        frames.extend(pair_b)
    else:
        pair_a = _place_horizontal_pair(h, w, frames, (3, h // 2 - 3), pair_right_edge_max=max(8, w // 2))
        if pair_a is None:
            return None
        frames.extend(pair_a)
        pair_b = _place_vertical_pair(h, w, frames, (w // 2 + 2, w - 4), pair_top_edge_min=max(1, h // 2 - 1))
        if pair_b is None:
            return None
        frames.extend(pair_b)

    for _ in range(extras_needed):
        extra = _place_extra_frame(h, w, frames)
        if extra is None:
            return None
        frames.append(extra)

    if _aligned_pair_count(frames) < 2:
        return None
    for frame in frames:
        frame["shape"] = (h, w)
    return frames


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        frames = _generate_layout()
        if frames is None:
            continue

        h, w = frames[0]["shape"]
        grid = np.zeros((h, w), dtype=int)
        frame_colors = random.sample(FRAME_COLOR_POOL, len(frames))
        parities = [0, 1] + [random.randint(0, 1) for _ in range(len(frames) - 2)]
        random.shuffle(parities)

        for frame, frame_color, parity in zip(frames, frame_colors, parities):
            frame["color"] = frame_color
            frame["parity"] = parity
            _paint_frame(grid, frame, frame_color, parity)

        detected = _find_frames(grid)
        if len(detected) != len(frames):
            continue

        odd_count = 0
        even_count = 0
        for frame in detected:
            r0, c0, r1, c1 = frame["box"]
            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            fg_count = int(np.count_nonzero((interior == RED) | (interior == ORANGE)))
            if fg_count < 2:
                break
            if fg_count % 2 == 1:
                odd_count += 1
            else:
                even_count += 1
            if not (np.any(interior == RED) and np.any(interior == ORANGE)):
                break
        else:
            if odd_count == 0 or even_count == 0:
                continue
            if _aligned_pair_count(detected) < 2:
                continue
            _ = main(grid)
            return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
