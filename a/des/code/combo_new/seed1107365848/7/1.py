import random

import numpy as np


BG_COLOR = 0
CROSS_COLOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
SIDES = ("top", "right", "bottom", "left")
ROTATE_TO_BOTTOM = {"bottom": 0, "left": 1, "top": 2, "right": 3}
FRAME_DIMENSIONS = [(5, 5), (5, 7), (5, 9), (7, 5), (7, 7), (7, 9), (9, 5), (9, 7), (9, 9)]
TIE_DIMENSIONS = ((5, 9), (9, 5))
GRID_RANGES = {2: (18, 30), 3: (22, 30), 4: (25, 30)}
SIZE_WEIGHTS = {2: [1, 3, 5], 3: [2, 3, 2], 4: [4, 3, 1]}
FRAME_COUNT_OPTIONS = [2, 3, 4]
FRAME_COUNT_WEIGHTS = [2, 3, 2]
CROSS_ARM_OPTIONS = [1, 2]
CROSS_CORNERS = ("tl", "tr", "bl", "br")
PLACEMENT_PAD = 1
GEN_RETRIES = 400


def _background_color(grid: np.ndarray) -> int:
    return int(np.bincount(grid.reshape(-1), minlength=10).argmax())


def _neighbors4(r: int, c: int):
    yield r + 1, c
    yield r - 1, c
    yield r, c + 1
    yield r, c - 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _cross_info(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict | None:
    colors = {int(grid[r, c]) for r, c in coords}
    if len(colors) != 1:
        return None

    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h != w or h < 3 or h % 2 == 0:
        return None

    mid_r = r0 + h // 2
    mid_c = c0 + w // 2
    expected = {(mid_r, c) for c in range(c0, c1 + 1)} | {(r, mid_c) for r in range(r0, r1 + 1)}
    cell_set = set(coords)
    if cell_set != expected:
        return None

    return {"color": int(next(iter(colors))), "cells": cell_set, "center": (mid_r, mid_c)}


def _marker_side(inside: np.ndarray, bg: int) -> tuple[str, int]:
    marker_vals = inside[inside != bg]
    assert marker_vals.size > 0, "frame interior must contain a marker line"
    marker_color = int(marker_vals[0])
    assert np.all(marker_vals == marker_color), "marker line must use one color"

    matches = []
    if np.all(inside[0, :] == marker_color) and np.all(inside[1:, :] == bg):
        matches.append("top")
    if np.all(inside[-1, :] == marker_color) and np.all(inside[:-1, :] == bg):
        matches.append("bottom")
    if np.all(inside[:, 0] == marker_color) and np.all(inside[:, 1:] == bg):
        matches.append("left")
    if np.all(inside[:, -1] == marker_color) and np.all(inside[:, :-1] == bg):
        matches.append("right")

    assert len(matches) == 1, "marker line must touch exactly one inner side"
    return matches[0], marker_color


def _frame_info(grid: np.ndarray, coords: list[tuple[int, int]], bg: int) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    h, w = patch.shape
    assert h >= 5 and w >= 5, "frames must be at least 5x5"
    assert int(np.count_nonzero(patch != bg)) == len(coords), "frame bbox may only contain one object"

    frame_color = int(patch[0, 0])
    assert frame_color != bg, "frame border must be colored"
    assert np.all(patch[0, :] == frame_color)
    assert np.all(patch[-1, :] == frame_color)
    assert np.all(patch[:, 0] == frame_color)
    assert np.all(patch[:, -1] == frame_color)

    inside = patch[1:-1, 1:-1]
    side, marker_color = _marker_side(inside, bg)
    assert marker_color != frame_color
    allowed = {bg, frame_color, marker_color}
    assert set(map(int, np.unique(patch))) <= allowed, "frame patch contains unexpected colors"

    rotated = np.rot90(patch, ROTATE_TO_BOTTOM[side])
    return {
        "area": h * w,
        "row": r0,
        "col": c0,
        "patch": patch,
        "rotated": rotated,
    }


def _stamp_non_bg(canvas: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    assert 0 <= top and 0 <= left and top + h <= canvas.shape[0] and left + w <= canvas.shape[1]
    region = canvas[top : top + h, left : left + w]
    mask = patch != bg
    region[mask] = patch[mask]


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)

    cross = None
    frames = []
    for coords in _components(g != bg):
        maybe_cross = _cross_info(g, coords)
        if maybe_cross is not None:
            assert cross is None, "expected exactly one stacking cross"
            cross = maybe_cross
        else:
            frames.append(_frame_info(g, coords, bg))

    assert cross is not None, "stacking cross not found"
    assert frames, "at least one frame is required"

    out = np.full_like(g, bg)
    center_r, center_c = cross["center"]
    ordered = sorted(frames, key=lambda item: (-item["area"], item["row"], item["col"]))
    for frame in ordered:
        patch = frame["rotated"]
        top = center_r - patch.shape[0] // 2
        left = center_c - patch.shape[1] // 2
        _stamp_non_bg(out, patch, top, left, bg)

    for r, c in cross["cells"]:
        out[r, c] = cross["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate marked hollow rectangles so their inner marker strips all sit on the bottom, then stack them at the gray cross center.

    Input structure:
    - The background is black, with several separated one-pixel-thick hollow rectangular frames.
    - Each frame contains one contiguous marker strip of a different color, attached to exactly one inner side of the frame.
    - A small monochrome cross in one corner marks the stacking center and remains fixed.

    Rule:
    - Detect every frame together with which inner side its marker strip touches: top, bottom, left, or right.
    - Rotate each frame so the marker strip ends up attached to the bottom inner side.
    - Sort the rotated frames by outer area (height x width), breaking ties by the original row-major position of the frame.
    - Clear the original frame locations, then stack the rotated frames centered on the cross intersection from largest to smallest, with later frames overwriting earlier ones only on their colored pixels.

    Key insight:
    - The marker strip is both part of the object and an orientation code: its touched side determines the rotation, while the frame bbox determines the stacking order.
    """
    return _solve(np.asarray(grid, dtype=int))


def _cross_cells(center: tuple[int, int], arm: int) -> set[tuple[int, int]]:
    r, c = center
    return {(r, cc) for cc in range(c - arm, c + arm + 1)} | {(rr, c) for rr in range(r - arm, r + arm + 1)}


def _frame_patch(height: int, width: int, frame_color: int, marker_color: int, side: str) -> np.ndarray:
    patch = np.full((height, width), BG_COLOR, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    if side == "top":
        patch[1, 1:-1] = marker_color
    elif side == "bottom":
        patch[-2, 1:-1] = marker_color
    elif side == "left":
        patch[1:-1, 1] = marker_color
    elif side == "right":
        patch[1:-1, -2] = marker_color
    else:
        raise ValueError(side)
    return patch


def _near_corner_range(lower: int, upper: int, toward_low: bool) -> list[int]:
    assert lower <= upper
    if toward_low:
        end = min(upper, lower + 3)
        return list(range(lower, end + 1))
    start = max(lower, upper - 3)
    return list(range(start, upper + 1))


def _choose_cross_center(shape: tuple[int, int], rotated_shapes: list[tuple[int, int]], corner: str, arm: int) -> tuple[int, int] | None:
    h, w = shape
    max_half_h = max(ph // 2 for ph, _ in rotated_shapes)
    max_half_w = max(pw // 2 for _, pw in rotated_shapes)

    low_r = max(arm, max_half_h)
    high_r = min(h - 1 - arm, h - 1 - max_half_h)
    low_c = max(arm, max_half_w)
    high_c = min(w - 1 - arm, w - 1 - max_half_w)
    if low_r > high_r or low_c > high_c:
        return None

    rows = _near_corner_range(low_r, high_r, toward_low=corner[0] == "t")
    cols = _near_corner_range(low_c, high_c, toward_low=corner[1] == "l")
    return random.choice(rows), random.choice(cols)


def _reserve(mask: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> None:
    h, w = mask.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    mask[r0:r1, c0:c1] = True


def _free_positions(mask: np.ndarray, height: int, width: int, pad: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    out = []
    for top in range(h - height + 1):
        for left in range(w - width + 1):
            r0 = max(0, top - pad)
            c0 = max(0, left - pad)
            r1 = min(h, top + height + pad)
            c1 = min(w, left + width + pad)
            if not mask[r0:r1, c0:c1].any():
                out.append((top, left))
    return out


def _sample_dimensions(frame_count: int, tie_slots: set[int], tie_side_group: tuple[str, str] | None) -> list[tuple[int, int, str]]:
    side_weighted_sizes = random.choices([5, 7, 9], weights=SIZE_WEIGHTS[frame_count], k=2 * frame_count)
    items: list[tuple[int, int, str]] = []
    tie_iter = iter(TIE_DIMENSIONS)
    for idx in range(frame_count):
        if idx in tie_slots:
            height, width = next(tie_iter)
            side = random.choice(tie_side_group)
        else:
            height = side_weighted_sizes[2 * idx]
            width = side_weighted_sizes[2 * idx + 1]
            side = random.choice(SIDES)
        items.append((height, width, side))
    return items


def _sample_specs(shape: tuple[int, int], frame_count: int) -> tuple[dict, list[dict]] | None:
    force_tie = frame_count >= 2 and random.random() < 0.45
    tie_slots: set[int] = set()
    tie_side_group = None
    if force_tie:
        tie_slots = set(random.sample(range(frame_count), 2))
        tie_side_group = random.choice((("top", "bottom"), ("left", "right")))

    dims = _sample_dimensions(frame_count, tie_slots, tie_side_group)
    color_pool = random.sample(PALETTE, 2 * frame_count)
    frame_colors = color_pool[:frame_count]
    marker_colors = color_pool[frame_count:]

    specs = []
    for idx, ((height, width, side), frame_color, marker_color) in enumerate(zip(dims, frame_colors, marker_colors)):
        patch = _frame_patch(height, width, frame_color, marker_color, side)
        rotated = np.rot90(patch, ROTATE_TO_BOTTOM[side])
        specs.append(
            {
                "id": idx,
                "patch": patch,
                "rotated": rotated,
                "area": height * width,
                "side": side,
            }
        )

    arm = random.choice(CROSS_ARM_OPTIONS)
    corner = random.choice(CROSS_CORNERS)
    center = _choose_cross_center(shape, [spec["rotated"].shape for spec in specs], corner, arm)
    if center is None:
        return None

    cross = {"arm": arm, "corner": corner, "center": center, "cells": _cross_cells(center, arm), "color": CROSS_COLOR}
    return cross, specs


def _place_specs(shape: tuple[int, int], cross: dict, specs: list[dict]) -> bool:
    occupied = np.zeros(shape, dtype=bool)
    cross_rows = [r for r, _ in cross["cells"]]
    cross_cols = [c for _, c in cross["cells"]]
    _reserve(
        occupied,
        min(cross_rows),
        min(cross_cols),
        max(cross_rows) - min(cross_rows) + 1,
        max(cross_cols) - min(cross_cols) + 1,
        PLACEMENT_PAD,
    )

    for spec in sorted(specs, key=lambda item: item["patch"].shape[0] * item["patch"].shape[1], reverse=True):
        ph, pw = spec["patch"].shape
        choices = _free_positions(occupied, ph, pw, PLACEMENT_PAD)
        if not choices:
            return False
        top, left = random.choice(choices)
        spec["top"] = top
        spec["left"] = left
        _reserve(occupied, top, left, ph, pw, PLACEMENT_PAD)
    return True


def _render_input(shape: tuple[int, int], cross: dict, specs: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG_COLOR, dtype=int)
    for spec in specs:
        _stamp_non_bg(grid, spec["patch"], spec["top"], spec["left"], BG_COLOR)
    for r, c in cross["cells"]:
        grid[r, c] = cross["color"]
    return grid


def _expected_output(shape: tuple[int, int], cross: dict, specs: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG_COLOR, dtype=int)
    center_r, center_c = cross["center"]
    ordered = sorted(specs, key=lambda item: (-item["area"], item["top"], item["left"]))
    for spec in ordered:
        patch = spec["rotated"]
        top = center_r - patch.shape[0] // 2
        left = center_c - patch.shape[1] // 2
        _stamp_non_bg(grid, patch, top, left, BG_COLOR)
    for r, c in cross["cells"]:
        grid[r, c] = cross["color"]
    return grid


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.choices(FRAME_COUNT_OPTIONS, weights=FRAME_COUNT_WEIGHTS)[0]
    low, high = GRID_RANGES[frame_count]
    shape = (random.randint(low, high), random.randint(low, high))

    sampled = _sample_specs(shape, frame_count)
    if sampled is None:
        return None
    cross, specs = sampled
    if not _place_specs(shape, cross, specs):
        return None

    inp = _render_input(shape, cross, specs)
    expected = _expected_output(shape, cross, specs)
    try:
        actual = main(inp)
    except Exception:
        return None
    return inp if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
