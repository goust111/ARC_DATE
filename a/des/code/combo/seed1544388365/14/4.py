import random

import numpy as np


TOP = "top"
BOTTOM = "bottom"
LEFT = "left"
RIGHT = "right"
SIDES = (TOP, BOTTOM, LEFT, RIGHT)

OPPOSITE = {
    TOP: BOTTOM,
    BOTTOM: TOP,
    LEFT: RIGHT,
    RIGHT: LEFT,
}

CORNER_ALLOWED = {
    "tl": {TOP, LEFT},
    "tr": {TOP, RIGHT},
    "bl": {BOTTOM, LEFT},
    "br": {BOTTOM, RIGHT},
}

BG_COLOR = 0
MARKER_COLOR = 3
FRAME_COLOR_POOL = [1, 2, 4, 5, 6, 7, 8, 9]
FRAME_COUNT_RANGE = (3, 5)
INNER_SIZE_RANGE = (3, 7)
EXTRA_SIZE_CHOICES = [0, 0, 1, 2]
PLACEMENT_PADDING = 1
PLACEMENT_ATTEMPTS = 200
MAX_GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _corner_origins(height: int, width: int) -> dict[str, tuple[int, int]]:
    return {
        "tl": (0, 0),
        "tr": (0, width - 2),
        "bl": (height - 2, 0),
        "br": (height - 2, width - 2),
    }


def _find_marker(grid: np.ndarray, bg: int) -> tuple[str, int]:
    height, width = map(int, grid.shape)
    assert height >= 4 and width >= 4, "grid must have room for a 2x2 corner marker"

    candidates: list[tuple[str, int]] = []
    for corner, (r0, c0) in _corner_origins(height, width).items():
        block = grid[r0 : r0 + 2, c0 : c0 + 2]
        color = int(block[0, 0])
        if color == bg:
            continue
        if np.all(block == color) and int(np.count_nonzero(grid == color)) == 4:
            candidates.append((corner, color))

    assert len(candidates) == 1, "expected exactly one 2x2 corner marker"
    return candidates[0]


def _parse_frame(grid: np.ndarray, bg: int, color: int) -> dict[str, int | str]:
    coords = np.argwhere(grid == color)
    assert len(coords) > 0, "frame color must be present"

    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    height, width = map(int, sub.shape)
    assert height >= 3 and width >= 3, "each frame must have a nonempty interior"
    assert set(map(int, np.unique(sub))).issubset({bg, color}), "frame bbox must contain only bg and frame color"
    assert np.all(sub[1:-1, 1:-1] == bg), "frame interior must be background"

    border = np.zeros_like(sub, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    color_mask = sub == color
    missing = np.argwhere(border & (sub == bg))
    assert len(missing) == 1, "frame border must have exactly one missing pixel"
    gap_r, gap_c = map(int, missing[0])
    assert not (gap_r in (0, height - 1) and gap_c in (0, width - 1)), "gap must lie on a side, not a corner"
    assert int(color_mask.sum()) == int(border.sum()) - 1, "frame must be a 1-pixel border with one gap"

    if gap_r == 0:
        gap_side = TOP
        gap_offset = gap_c
    elif gap_r == height - 1:
        gap_side = BOTTOM
        gap_offset = gap_c
    elif gap_c == 0:
        gap_side = LEFT
        gap_offset = gap_r
    else:
        gap_side = RIGHT
        gap_offset = gap_r

    assert 1 <= gap_offset <= (width - 2 if gap_side in (TOP, BOTTOM) else height - 2)

    return {
        "color": int(color),
        "r0": r0,
        "c0": c0,
        "h": height,
        "w": width,
        "area": height * width,
        "gap_side": gap_side,
        "gap_offset": int(gap_offset),
    }


def _normalize_gap_side(gap_side: str, corner: str) -> str:
    allowed = CORNER_ALLOWED[corner]
    if gap_side in allowed:
        return gap_side

    normalized = OPPOSITE[gap_side]
    assert normalized in allowed, "gap side must be flippable toward the marker corner"
    return normalized


def _scale_gap_offset(offset: int, old_span: int, new_span: int) -> int:
    assert 3 <= new_span <= old_span, "scaled frame span must remain valid"
    old_slots = old_span - 2
    new_slots = new_span - 2
    assert 1 <= offset <= old_slots

    if old_slots == 1 or new_slots == 1:
        return 1
    if old_slots == new_slots:
        return int(offset)

    old_index = offset - 1
    numerator = old_index * (new_slots - 1)
    denominator = old_slots - 1
    mapped = (2 * numerator + denominator) // (2 * denominator)
    return 1 + int(min(max(mapped, 0), new_slots - 1))


def _solved_frames(grid: np.ndarray) -> tuple[int, list[dict[str, int | str]]]:
    bg = _background_color(grid)
    corner, marker_color = _find_marker(grid, bg)

    colors = [int(color) for color in np.unique(grid) if int(color) not in (bg, marker_color)]
    frames = [_parse_frame(grid, bg, color) for color in colors]
    assert len(frames) >= 2, "expected multiple frames"

    areas = [int(frame["area"]) for frame in frames]
    assert len(set(areas)) == len(areas), "frame areas must be unique"
    frames.sort(key=lambda frame: int(frame["area"]), reverse=True)

    solved: list[dict[str, int | str]] = []
    for i, frame in enumerate(frames):
        new_side = _normalize_gap_side(str(frame["gap_side"]), corner)
        if i == 0:
            new_h = int(frame["h"])
            new_w = int(frame["w"])
        else:
            new_h = int(solved[-1]["h"]) - 4
            new_w = int(solved[-1]["w"]) - 4
            assert new_h >= 3 and new_w >= 3, "nested frames must remain drawable"
            assert int(frame["h"]) >= new_h and int(frame["w"]) >= new_w, "input frame must be large enough to shrink"

        old_span = int(frame["w"]) if new_side in (TOP, BOTTOM) else int(frame["h"])
        new_span = new_w if new_side in (TOP, BOTTOM) else new_h
        solved.append(
            {
                "color": int(frame["color"]),
                "h": new_h,
                "w": new_w,
                "gap_side": new_side,
                "gap_offset": _scale_gap_offset(int(frame["gap_offset"]), old_span, new_span),
            }
        )

    return bg, solved


def _draw_frame(
    grid: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    color: int,
    gap_side: str,
    gap_offset: int,
    bg: int,
) -> None:
    bottom = top + height - 1
    right = left + width - 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color

    if gap_side == TOP:
        grid[top, left + gap_offset] = bg
    elif gap_side == BOTTOM:
        grid[bottom, left + gap_offset] = bg
    elif gap_side == LEFT:
        grid[top + gap_offset, left] = bg
    else:
        grid[top + gap_offset, right] = bg


def _render_output(shape: tuple[int, int], bg: int, frames: list[dict[str, int | str]]) -> np.ndarray:
    height, width = map(int, shape)
    out = np.full((height, width), bg, dtype=int)

    outer_h = int(frames[0]["h"])
    outer_w = int(frames[0]["w"])
    top0 = (height - outer_h) // 2
    left0 = (width - outer_w) // 2
    assert top0 >= 0 and left0 >= 0, "outer frame must fit inside the output grid"

    for i, frame in enumerate(frames):
        top = top0 + 2 * i
        left = left0 + 2 * i
        frame_h = int(frame["h"])
        frame_w = int(frame["w"])
        assert top + frame_h <= height and left + frame_w <= width
        _draw_frame(
            out,
            top,
            left,
            frame_h,
            frame_w,
            int(frame["color"]),
            str(frame["gap_side"]),
            int(frame["gap_offset"]),
            bg,
        )

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort gapped rectangular frames by area, orient their gaps toward the corner marker,
    then redraw them as a centered nested stack with 1-pixel spacing between layers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, frames = _solved_frames(g)
    return _render_output(g.shape, bg, frames)


def _frame_patch(height: int, width: int, color: int, gap_side: str, gap_offset: int, bg: int) -> np.ndarray:
    patch = np.full((height, width), bg, dtype=int)
    _draw_frame(patch, 0, 0, height, width, color, gap_side, gap_offset, bg)
    return patch


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    sub = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
    mask = patch != bg
    sub[mask] = patch[mask]


def _reserve(blocked: np.ndarray, top: int, left: int, height: int, width: int, padding: int) -> None:
    r0 = max(0, top - padding)
    c0 = max(0, left - padding)
    r1 = min(blocked.shape[0], top + height + padding)
    c1 = min(blocked.shape[1], left + width + padding)
    blocked[r0:r1, c0:c1] = True


def _can_place(blocked: np.ndarray, top: int, left: int, height: int, width: int, padding: int) -> bool:
    r0 = max(0, top - padding)
    c0 = max(0, left - padding)
    r1 = min(blocked.shape[0], top + height + padding)
    c1 = min(blocked.shape[1], left + width + padding)
    return not bool(blocked[r0:r1, c0:c1].any())


def _draw_marker(grid: np.ndarray, corner: str, color: int) -> None:
    height, width = map(int, grid.shape)
    r0, c0 = _corner_origins(height, width)[corner]
    grid[r0 : r0 + 2, c0 : c0 + 2] = color


def _planned_frames(
    corner: str, originals: list[dict[str, int | str]], targets: list[tuple[int, int]]
) -> list[dict[str, int | str]]:
    solved: list[dict[str, int | str]] = []
    for frame, (target_h, target_w) in zip(originals, targets):
        side = _normalize_gap_side(str(frame["gap_side"]), corner)
        old_span = int(frame["w"]) if side in (TOP, BOTTOM) else int(frame["h"])
        new_span = int(target_w) if side in (TOP, BOTTOM) else int(target_h)
        solved.append(
            {
                "color": int(frame["color"]),
                "h": int(target_h),
                "w": int(target_w),
                "gap_side": side,
                "gap_offset": _scale_gap_offset(int(frame["gap_offset"]), old_span, new_span),
            }
        )
    return solved


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(*FRAME_COUNT_RANGE)
    inner_h = random.randint(*INNER_SIZE_RANGE)
    inner_w = random.randint(*INNER_SIZE_RANGE)

    targets: list[tuple[int, int]] = []
    for depth in range(frame_count):
        inset = 4 * (frame_count - depth - 1)
        targets.append((inner_h + inset, inner_w + inset))

    originals: list[dict[str, int | str]] = []
    for i, (target_h, target_w) in enumerate(targets):
        extra_h = 0 if i == 0 else random.choice(EXTRA_SIZE_CHOICES)
        extra_w = 0 if i == 0 else random.choice(EXTRA_SIZE_CHOICES)
        originals.append(
            {
                "h": target_h + extra_h,
                "w": target_w + extra_w,
                "area": (target_h + extra_h) * (target_w + extra_w),
            }
        )

    if any(int(originals[i]["area"]) <= int(originals[i + 1]["area"]) for i in range(frame_count - 1)):
        return None

    colors = random.sample(FRAME_COLOR_POOL, frame_count)
    corner = random.choice(list(CORNER_ALLOWED))
    for frame, color in zip(originals, colors):
        side = random.choice(SIDES)
        span = int(frame["w"]) if side in (TOP, BOTTOM) else int(frame["h"])
        frame["color"] = color
        frame["gap_side"] = side
        frame["gap_offset"] = random.randint(1, span - 2)

    total_bbox_area = sum(int(frame["h"]) * int(frame["w"]) for frame in originals)
    outer_h = int(originals[0]["h"])
    outer_w = int(originals[0]["w"])
    base = max(outer_h + 8, outer_w + 8, int(np.ceil(np.sqrt(total_bbox_area))) + 8)
    height = base + random.randint(0, 4)
    width = base + random.randint(0, 4)

    grid = np.full((height, width), BG_COLOR, dtype=int)
    blocked = np.zeros_like(grid, dtype=bool)
    _draw_marker(grid, corner, MARKER_COLOR)
    _reserve(blocked, *_corner_origins(height, width)[corner], 2, 2, PLACEMENT_PADDING)

    for frame in originals:
        patch = _frame_patch(
            int(frame["h"]),
            int(frame["w"]),
            int(frame["color"]),
            str(frame["gap_side"]),
            int(frame["gap_offset"]),
            BG_COLOR,
        )
        placed = False
        max_top = height - patch.shape[0]
        max_left = width - patch.shape[1]
        if max_top < 0 or max_left < 0:
            return None

        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(0, max_top)
            left = random.randint(0, max_left)
            if not _can_place(blocked, top, left, patch.shape[0], patch.shape[1], PLACEMENT_PADDING):
                continue
            _place_patch(grid, patch, top, left, BG_COLOR)
            _reserve(blocked, top, left, patch.shape[0], patch.shape[1], PLACEMENT_PADDING)
            placed = True
            break

        if not placed:
            return None

    expected = _render_output(grid.shape, BG_COLOR, _planned_frames(corner, originals, targets))
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
