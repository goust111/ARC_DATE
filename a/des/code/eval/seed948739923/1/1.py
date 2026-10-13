from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
FRAME_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRECTIONS = ("up", "down", "left", "right")


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))

    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_area(h: int, w: int) -> int:
    return 2 * h + 2 * w - 4


def _has_rect_border(sub: np.ndarray, color: int) -> bool:
    return bool(
        np.all(sub[0, :] == color)
        and np.all(sub[-1, :] == color)
        and np.all(sub[:, 0] == color)
        and np.all(sub[:, -1] == color)
    )


def _find_colored_frames(grid: np.ndarray) -> dict[int, tuple[int, int, int, int]]:
    frames: dict[int, tuple[int, int, int, int]] = {}
    for color in FRAME_PALETTE:
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 5 or w < 5:
                continue
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not _has_rect_border(sub, color):
                continue
            if int(len(comp)) <= _border_area(h, w):
                continue
            if not np.any(sub[1:-1, 1:-1] == color):
                continue
            assert color not in frames, "frame colors must be unique"
            frames[color] = (r0, c0, r1, c1)
    assert frames, "expected colored partition frames"
    return frames


def _find_canvas_boxes(grid: np.ndarray) -> dict[int, tuple[tuple[int, int, int, int], tuple[int, int]]]:
    boxes: dict[int, tuple[tuple[int, int, int, int], tuple[int, int]]] = {}
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 5 or w < 5:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not _has_rect_border(sub, GREY):
            continue
        if int(len(comp)) != _border_area(h, w):
            continue

        interior = sub[1:-1, 1:-1]
        marker_coords = np.argwhere((interior != BG) & (interior != GREY))
        if marker_coords.shape[0] != 1:
            continue
        mr, mc = map(int, marker_coords[0])
        marker_color = int(interior[mr, mc])
        cleaned = interior.copy()
        cleaned[mr, mc] = BG
        if not np.all(cleaned == BG):
            continue
        assert marker_color not in boxes, "marker colors must be unique"
        boxes[marker_color] = ((r0, c0, r1, c1), (r0 + 1 + mr, c0 + 1 + mc))

    assert boxes, "expected grey canvas boxes"
    return boxes


def _gap_candidates(sub: np.ndarray, color: int) -> list[tuple[int, int, str]]:
    h, w = sub.shape
    candidates: list[tuple[int, int, str]] = []
    for c in range(2, w - 2):
        if sub[1, c] == BG and sub[0, c] == color and sub[2, c] == color:
            candidates.append((1, c, "up"))
        if sub[h - 2, c] == BG and sub[h - 3, c] == color and sub[h - 1, c] == color:
            candidates.append((h - 2, c, "down"))
    for r in range(2, h - 2):
        if sub[r, 1] == BG and sub[r, 0] == color and sub[r, 2] == color:
            candidates.append((r, 1, "left"))
        if sub[r, w - 2] == BG and sub[r, w - 3] == color and sub[r, w - 1] == color:
            candidates.append((r, w - 2, "right"))
    return candidates


def _count_rooms(repaired: np.ndarray) -> int:
    interior_bg = repaired[1:-1, 1:-1] == BG
    return len(_components(interior_bg))


def _analyze_frame(sub: np.ndarray, color: int) -> tuple[int, str]:
    candidates = _gap_candidates(sub, color)
    assert len(candidates) == 1, "expected one repairable wall gap"
    gr, gc, direction = candidates[0]
    repaired = sub.copy()
    repaired[gr, gc] = color
    rooms = _count_rooms(repaired)
    assert 2 <= rooms <= 5, "room count must be in range 2..5"
    return rooms, direction


def _square_origin(inner_h: int, inner_w: int, size: int, direction: str) -> tuple[int, int]:
    if direction == "up":
        return 0, (inner_w - size) // 2
    if direction == "down":
        return inner_h - size, (inner_w - size) // 2
    if direction == "left":
        return (inner_h - size) // 2, 0
    assert direction == "right"
    return (inner_h - size) // 2, inner_w - size


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a solid reduced copy in each marked canvas.

    The upper objects are same-color rectangular frames with internal wall
    segments and one one-cell gap. Repairing that gap yields a partition whose
    number of rooms gives the square size. The gap's side (up/down/left/right)
    gives the destination edge in the matching lower grey canvas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _find_colored_frames(g)
    boxes = _find_canvas_boxes(g)
    assert set(frames) == set(boxes), "frame colors and canvas markers must match"

    frame_info: dict[int, tuple[int, str]] = {}
    for color, (r0, c0, r1, c1) in frames.items():
        frame_info[color] = _analyze_frame(g[r0 : r1 + 1, c0 : c1 + 1], color)

    out = g.copy()
    for color, ((r0, c0, r1, c1), marker_pos) in boxes.items():
        rooms, direction = frame_info[color]
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        inner_h, inner_w = inner.shape
        assert rooms <= inner_h and rooms <= inner_w, "canvas too small for square"

        mr, mc = marker_pos
        out[mr, mc] = BG

        sr, sc = _square_origin(inner_h, inner_w, rooms, direction)
        inner[sr : sr + rooms, sc : sc + rooms] = color

    return out


def _shape_options(room_count: int, direction: str) -> list[tuple[int, int]]:
    if direction in ("up", "down"):
        options = [(1, room_count)]
        if room_count == 4:
            options.append((2, 2))
        return options
    options = [(room_count, 1)]
    if room_count == 4:
        options.append((2, 2))
    return options


def _frame_patch(color: int, room_count: int, direction: str) -> tuple[np.ndarray, dict[str, int | str]]:
    rows, cols = random.choice(_shape_options(room_count, direction))
    room_h = random.randint(2, 3)
    room_w = random.randint(2, 3)

    inner_h = rows * room_h + (rows - 1)
    inner_w = cols * room_w + (cols - 1)
    patch = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color

    horizontal_rows = [r * (room_h + 1) for r in range(1, rows)]
    vertical_cols = [c * (room_w + 1) for c in range(1, cols)]

    for wall_r in horizontal_rows:
        patch[wall_r, 1:-1] = color
    for wall_c in vertical_cols:
        patch[1:-1, wall_c] = color

    if direction in ("up", "down"):
        gap_c = random.choice(vertical_cols)
        gap_r = 1 if direction == "up" else inner_h
        patch[gap_r, gap_c] = BG
    else:
        gap_r = random.choice(horizontal_rows)
        gap_c = 1 if direction == "left" else inner_w
        patch[gap_r, gap_c] = BG

    meta = {
        "rooms": room_count,
        "direction": direction,
        "gap_r": gap_r,
        "gap_c": gap_c,
    }
    return patch, meta


def _box_patch(color: int, inner_size: int) -> np.ndarray:
    patch = np.full((inner_size + 2, inner_size + 2), BG, dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    patch[1, 1] = color
    return patch


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _render_expected(
    grid: np.ndarray,
    frame_meta: dict[int, dict[str, int | str]],
    box_positions: dict[int, tuple[int, int, int, int]],
) -> np.ndarray:
    out = grid.copy()
    for color, (r0, c0, r1, c1) in box_positions.items():
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        inner[0, 0] = BG
        rooms = int(frame_meta[color]["rooms"])
        direction = str(frame_meta[color]["direction"])
        sr, sc = _square_origin(inner.shape[0], inner.shape[1], rooms, direction)
        inner[sr : sr + rooms, sc : sc + rooms] = color
    return out


def _shuffled_box_order(colors: list[int]) -> list[int]:
    order = colors[:]
    random.shuffle(order)
    if len(order) > 1 and order == colors:
        order = order[1:] + order[:1]
    return order


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(2, 4)
    colors = random.sample(FRAME_PALETTE, frame_count)
    inner_box_size = random.choice((7, 8))

    frame_patches: list[np.ndarray] = []
    frame_meta: dict[int, dict[str, int | str]] = {}
    for color in colors:
        room_count = random.randint(2, 5)
        direction = random.choice(DIRECTIONS)
        patch, meta = _frame_patch(color, room_count, direction)
        frame_patches.append(patch)
        frame_meta[color] = meta

    box_order = _shuffled_box_order(colors)
    box_patches = {color: _box_patch(color, inner_box_size) for color in box_order}

    top_gaps = [random.randint(2, 4) for _ in range(frame_count - 1)]
    bottom_gaps = [random.randint(1, 3) for _ in range(frame_count - 1)]

    top_width = sum(p.shape[1] for p in frame_patches) + sum(top_gaps)
    bottom_width = sum(box_patches[c].shape[1] for c in box_order) + sum(bottom_gaps)
    top_height = max(p.shape[0] for p in frame_patches)
    bottom_height = max(box_patches[c].shape[0] for c in box_order)

    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    top_margin = random.randint(1, 2)
    middle_gap = random.randint(2, 4)
    bottom_margin = random.randint(1, 2)

    width = max(top_width, bottom_width) + left_margin + right_margin
    height = top_margin + top_height + middle_gap + bottom_height + bottom_margin
    if width > 52 or height > 36:
        return None

    grid = np.full((height, width), BG, dtype=int)

    top_x0 = random.randint(left_margin, width - right_margin - top_width)
    bottom_x0 = random.randint(left_margin, width - right_margin - bottom_width)
    top_y0 = top_margin
    bottom_y0 = top_margin + top_height + middle_gap

    x = top_x0
    for i, (color, patch) in enumerate(zip(colors, frame_patches)):
        _paste(grid, patch, top_y0, x)
        frame_meta[color]["bbox"] = (top_y0, x, top_y0 + patch.shape[0] - 1, x + patch.shape[1] - 1)
        if i < len(top_gaps):
            x += patch.shape[1] + top_gaps[i]

    box_positions: dict[int, tuple[int, int, int, int]] = {}
    x = bottom_x0
    for i, color in enumerate(box_order):
        patch = box_patches[color]
        _paste(grid, patch, bottom_y0, x)
        box_positions[color] = (bottom_y0, x, bottom_y0 + patch.shape[0] - 1, x + patch.shape[1] - 1)
        if i < len(bottom_gaps):
            x += patch.shape[1] + bottom_gaps[i]

    expected = _render_expected(grid, frame_meta, box_positions)
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
