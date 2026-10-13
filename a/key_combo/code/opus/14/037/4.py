from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

FRAME_COLORS = (BLUE, RED)
CENTER_COLORS = (GREEN, YELLOW, ORANGE)
DISTRACTOR_COLORS = FRAME_COLORS + CENTER_COLORS

RING_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
FRAME_NEIGHBORHOOD_OFFSETS = tuple((dr, dc) for dr in range(-2, 3) for dc in range(-2, 3))

GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _frame_signature(frames: list[dict]) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(
        sorted(
            (
                int(frame["center"][0]),
                int(frame["center"][1]),
                int(frame["border_color"]),
                int(frame["center_color"]),
            )
            for frame in frames
        )
    )


def _frame_cells(frame: dict) -> list[tuple[int, int]]:
    r, c = frame["center"]
    return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]


def _frame_corners(frame: dict) -> list[tuple[int, int]]:
    r, c = frame["center"]
    return [(r - 2, c - 2), (r - 2, c + 2), (r + 2, c - 2), (r + 2, c + 2)]


def _ray_directions(border_color: int) -> tuple[tuple[int, int], ...]:
    if border_color == RED:
        return ((-1, 0), (1, 0))
    if border_color == BLUE:
        return ((0, -1), (0, 1))
    raise ValueError(f"unsupported border color: {border_color}")


def _sorted_frames(frames: list[dict]) -> list[dict]:
    return sorted(
        frames,
        key=lambda frame: (
            int(frame["center"][0]),
            int(frame["center"][1]),
            int(frame["border_color"]),
            int(frame["center_color"]),
        ),
    )


def _find_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    frames: list[dict] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            center_color = int(g[r, c])
            if center_color == BLACK:
                continue
            border = [int(g[r + dr, c + dc]) for dr, dc in RING_OFFSETS]
            if len(set(border)) != 1:
                continue
            border_color = border[0]
            if border_color == BLACK or border_color == center_color:
                continue
            if border_color not in FRAME_COLORS or center_color not in CENTER_COLORS:
                continue
            frames.append(
                {
                    "center": (int(r), int(c)),
                    "border_color": border_color,
                    "center_color": center_color,
                }
            )
    return _sorted_frames(frames)


def _paint_frame_only(out: np.ndarray, frame: dict) -> None:
    r, c = frame["center"]
    out[r - 1 : r + 2, c - 1 : c + 2] = int(frame["border_color"])
    out[r, c] = int(frame["center_color"])


def _render_frames(shape: tuple[int, int], frames: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for frame in _sorted_frames(frames):
        _paint_frame_only(out, frame)
    return out


def _trace_ray(frame: dict, direction: tuple[int, int], shape: tuple[int, int], blockers: set[tuple[int, int]]) -> tuple[list[tuple[int, int]], str]:
    r0, c0 = frame["center"]
    dr, dc = direction
    r = r0 + 2 * dr
    c = c0 + 2 * dc
    cells: list[tuple[int, int]] = []
    while _in_bounds(shape, r, c) and (r, c) not in blockers:
        cells.append((r, c))
        r += dr
        c += dc
    return cells, ("frame" if _in_bounds(shape, r, c) else "edge")


def _collect_ray_info(shape: tuple[int, int], frames: list[dict]) -> tuple[list[dict], dict[tuple[int, int], list[dict]]]:
    blockers = {cell for frame in frames for cell in _frame_cells(frame)}
    rays: list[dict] = []
    claims: dict[tuple[int, int], list[dict]] = {}
    for frame in _sorted_frames(frames):
        for direction in _ray_directions(int(frame["border_color"])):
            cells, stop_reason = _trace_ray(frame, direction, shape, blockers)
            ray = {
                "frame": frame,
                "direction": direction,
                "cells": cells,
                "stop_reason": stop_reason,
            }
            rays.append(ray)
            for cell in cells:
                claims.setdefault(cell, []).append(ray)
    return rays, claims


def _render_output(shape: tuple[int, int], frames: list[dict]) -> np.ndarray:
    out = _render_frames(shape, frames)
    rays, _ = _collect_ray_info(shape, frames)
    for ray in rays:
        color = int(ray["frame"]["center_color"])
        for r, c in ray["cells"]:
            out[r, c] = color
    for frame in _sorted_frames(frames):
        color = int(frame["center_color"])
        for r, c in _frame_corners(frame):
            if _in_bounds(shape, r, c):
                out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Keep only valid 3x3 frames, extend rays by frame-color direction, and color outside corners by center color.

    Input structure:
    - The grid is a black background with several valid 3x3 monochrome frames and scattered single-pixel distractors.
    - Each valid frame has one border color on its eight outer cells and a different non-black center color.
    - Border colors are red or blue, which determine whether the frame controls vertical or horizontal rays.

    Rule:
    - Discard every non-frame foreground pixel.
    - Preserve every valid frame, then for each red frame extend center-colored rays upward and downward, and for each blue frame extend center-colored rays leftward and rightward.
    - A ray starts just outside its own 3x3 frame and stops before the grid edge or before any pixel of another valid frame.
    - After all rays are drawn, place the frame's center color on the four diagonal outside-corner cells around that frame, overwriting any ray already there.

    Key insight:
    - The border color and center color control different output dimensions: border color chooses the ray axis, while center color chooses the color used for both rays and corner decorations.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    frames = _find_frames(g)
    return _render_output(g.shape, frames)


def _roles_for_count(frame_count: int) -> dict[str, tuple[int, int]]:
    mid_row = random.randint(5, 7)
    blue_row = mid_row + random.choice((-1, 0, 1))
    blue_col = random.randint(8, 10)
    roles = {
        "r1": (0, 0),
        "rmid": (mid_row, 2),
        "b1": (blue_row, blue_col),
    }
    if frame_count == 5 or (frame_count == 4 and random.choice((True, False))):
        roles["rblock"] = (mid_row + random.randint(5, 7), 0)
    if frame_count == 5 or (frame_count == 4 and "rblock" not in roles):
        roles["bblock"] = (blue_row, blue_col + random.randint(6, 8))
    return roles


def _transform_role_centers(role_centers: dict[str, tuple[int, int]]) -> dict[str, tuple[int, int]]:
    rows = [rc[0] for rc in role_centers.values()]
    cols = [rc[1] for rc in role_centers.values()]
    max_r = max(rows)
    max_c = max(cols)
    flip_v = random.choice((False, True))
    flip_h = random.choice((False, True))
    transformed = {}
    for role, (r, c) in role_centers.items():
        rr = max_r - r if flip_v else r
        cc = max_c - c if flip_h else c
        transformed[role] = (rr, cc)
    return transformed


def _build_frames(frame_count: int) -> tuple[tuple[int, int], list[dict]]:
    role_centers = _transform_role_centers(_roles_for_count(frame_count))
    max_r = max(r for r, _ in role_centers.values())
    max_c = max(c for _, c in role_centers.values())

    min_h = max(17, max_r + 5)
    min_w = max(17, max_c + 5)
    h = random.randint(min_h, min(26, min_h + 5))
    w = random.randint(min_w, min(26, min_w + 6))

    shift_r = random.randint(2, h - max_r - 3)
    shift_c = random.randint(2, w - max_c - 3)

    center_a, center_b, center_c = random.sample(CENTER_COLORS, 3)
    color_by_role = {
        "r1": center_a,
        "rblock": center_a,
        "b1": center_b,
        "bblock": center_b,
        "rmid": center_c,
    }

    frames: list[dict] = []
    for role, (r, c) in role_centers.items():
        frames.append(
            {
                "role": role,
                "center": (r + shift_r, c + shift_c),
                "border_color": RED if role.startswith("r") else BLUE,
                "center_color": color_by_role[role],
            }
        )
    return (h, w), _sorted_frames(frames)


def _well_separated(frames: list[dict]) -> bool:
    centers = [frame["center"] for frame in frames]
    for i, (r1, c1) in enumerate(centers):
        for r2, c2 in centers[i + 1 :]:
            if abs(r1 - r2) < 5 and abs(c1 - c2) < 5:
                return False
    return True


def _has_color_branch_coverage(frames: list[dict]) -> bool:
    border_colors = {int(frame["border_color"]) for frame in frames}
    center_colors = {int(frame["center_color"]) for frame in frames}
    return border_colors == {BLUE, RED} and center_colors == set(CENTER_COLORS)


def _is_candidate_distractor_cell(grid: np.ndarray, r: int, c: int) -> bool:
    if int(grid[r, c]) != BLACK:
        return False
    for rr, cc in _neighbors4(r, c):
        if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) != BLACK:
            return False
    return True


def _safe_distractor_colors(
    grid: np.ndarray,
    r: int,
    c: int,
    expected_signature: tuple[tuple[int, int, int, int], ...],
    preferred_color: int | None,
) -> list[int]:
    palette = list(DISTRACTOR_COLORS)
    random.shuffle(palette)
    if preferred_color in palette:
        palette.remove(int(preferred_color))
        palette.insert(0, int(preferred_color))
    safe: list[int] = []
    for color in palette:
        trial = grid.copy()
        trial[r, c] = int(color)
        if _frame_signature(_find_frames(trial)) == expected_signature:
            safe.append(int(color))
    return safe


def _place_distractors(base: np.ndarray, frames: list[dict], expected: np.ndarray) -> np.ndarray | None:
    grid = base.copy()
    expected_signature = _frame_signature(frames)

    frame_cells = {cell for frame in frames for cell in _frame_cells(frame)}
    interesting: list[tuple[int, int, int]] = []
    ordinary: list[tuple[int, int, int]] = []

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if not _is_candidate_distractor_cell(grid, r, c):
                continue
            preferred = int(expected[r, c]) if int(expected[r, c]) != BLACK else None
            item = (r, c, preferred if preferred is not None else -1)
            if (r, c) not in frame_cells and int(expected[r, c]) != BLACK:
                interesting.append(item)
            else:
                ordinary.append(item)

    random.shuffle(interesting)
    random.shuffle(ordinary)

    target = random.randint(7, 12)
    target_interesting = min(len(interesting), random.randint(2, 4))
    placed = 0

    def try_place(pool: list[tuple[int, int, int]], required: int = 0) -> int:
        nonlocal grid, placed
        added = 0
        for r, c, preferred in pool:
            if placed >= target:
                break
            if not _is_candidate_distractor_cell(grid, r, c):
                continue
            pref_color = None if preferred == -1 else preferred
            safe = _safe_distractor_colors(grid, r, c, expected_signature, pref_color)
            if not safe:
                continue
            grid[r, c] = int(safe[0])
            placed += 1
            added += 1
            if required and added >= required:
                break
        return added

    if try_place(interesting, target_interesting) < min(target_interesting, 2):
        return None
    try_place(ordinary)
    try_place(interesting)
    try_place(ordinary)

    if placed < 6:
        return None
    return grid


def _validate_layout(shape: tuple[int, int], frames: list[dict]) -> bool:
    if not _well_separated(frames):
        return False
    if not _has_color_branch_coverage(frames):
        return False

    rays, claims = _collect_ray_info(shape, frames)
    if not any(ray["stop_reason"] == "frame" and ray["cells"] for ray in rays):
        return False

    if not any(len({int(item["frame"]["center_color"]) for item in items}) > 1 for items in claims.values()):
        # Different-color ray overlaps are forbidden.
        pass
    for items in claims.values():
        if len({int(item["frame"]["center_color"]) for item in items}) > 1:
            return False

    ray_cells = set(claims)
    if not any(corner in ray_cells for frame in frames for corner in _frame_corners(frame)):
        return False

    return True


def _validate_instance(inp: np.ndarray, frames: list[dict]) -> bool:
    expected_signature = _frame_signature(frames)
    detected = _find_frames(inp)
    if _frame_signature(detected) != expected_signature:
        return False

    non_frame_cells = [
        (r, c)
        for r in range(inp.shape[0])
        for c in range(inp.shape[1])
        if int(inp[r, c]) != BLACK and (r, c) not in {cell for frame in frames for cell in _frame_cells(frame)}
    ]
    if len(non_frame_cells) < 6:
        return False
    for r, c in non_frame_cells:
        if sum(
            1
            for rr, cc in _neighbors4(r, c)
            if _in_bounds(inp.shape, rr, cc) and int(inp[rr, cc]) != BLACK
        ) != 0:
            return False

    expected = _render_output(inp.shape, frames)
    solved = main(inp)
    if not np.array_equal(expected, solved):
        return False

    frame_only = _render_frames(inp.shape, frames)
    if np.array_equal(expected, frame_only):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    frame_count = random.choices((3, 4, 5), weights=(1, 2, 3), k=1)[0]
    shape, frames = _build_frames(frame_count)
    if not _validate_layout(shape, frames):
        return None

    base = _render_frames(shape, frames)
    expected = _render_output(shape, frames)
    inp = _place_distractors(base, frames, expected)
    if inp is None:
        return None
    if not _validate_instance(inp, frames):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
