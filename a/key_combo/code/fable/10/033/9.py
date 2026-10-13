from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
RING_COORDS = (
    (0, 0), (0, 1), (0, 2),
    (1, 0),         (1, 2),
    (2, 0), (2, 1), (2, 2),
)

GEN_RETRIES = 800
SEEN_INPUT_SIGS: set[tuple[tuple[int, int], bytes]] = set()
SEEN_OUTPUT_SIGS: set[tuple[tuple[int, int], bytes]] = set()


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=16)
    return int(np.argmax(counts))


def _instruction_frames(grid: np.ndarray, bg: int) -> list[dict[str, int | tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    frames: list[dict[str, int | tuple[int, int]]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            center = int(g[r, c])
            if center == bg:
                continue
            ring = [int(g[r + dr - 1, c + dc - 1]) for dr, dc in RING_COORDS]
            ring_color = ring[0]
            if ring_color == bg or ring_color == center:
                continue
            if all(v == ring_color for v in ring):
                frames.append({"center": (r, c), "ring": ring_color, "color": center})
    frames.sort(key=lambda item: item["center"])  # type: ignore[arg-type]
    return frames


def _hollow_square_candidates(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, int, int, int]] = []
    max_side = min(h, w)
    for side in range(3, max_side + 1):
        for r0 in range(h - side + 1):
            for c0 in range(w - side + 1):
                patch = g[r0 : r0 + side, c0 : c0 + side]
                color = int(patch[0, 0])
                if color == bg:
                    continue
                if not (
                    np.all(patch[0, :] == color)
                    and np.all(patch[-1, :] == color)
                    and np.all(patch[:, 0] == color)
                    and np.all(patch[:, -1] == color)
                ):
                    continue
                if side > 2 and not np.all(patch[1:-1, 1:-1] == bg):
                    continue
                candidates.append((side, r0, c0, color))
    return candidates


def _pick_big_square(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    candidates = _hollow_square_candidates(grid, bg)
    assert candidates, "expected at least one hollow square"
    max_side = max(side for side, _r0, _c0, _color in candidates)
    biggest = [cand for cand in candidates if cand[0] == max_side]
    assert len(biggest) == 1, "expected a unique largest hollow square"
    return biggest[0]


def _program_output(side: int, frame_color: int, program: list[int]) -> np.ndarray:
    out = np.full((side, side), int(frame_color), dtype=int)
    for r in range(side):
        for c in range(side):
            depth = min(r, c, side - 1 - r, side - 1 - c)
            if depth > 0:
                out[r, c] = int(program[(depth - 1) % len(program)])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Read complete 3x3 instruction frames and use their center colors to fill a hollow square inward shell by shell.

    Input structure:
    - The grid has black background, exactly one large one-cell-thick hollow square frame, and several scattered 3x3 mini-frames.
    - A valid mini-frame is a complete 3x3 window whose eight border cells share one non-background color and whose center is a different non-background color.
    - Extra scattered pixels and broken near-frames may appear as distractors.

    Rule:
    - Identify the unique largest hollow square frame and crop the output to its bounding box.
    - Read only complete 3x3 mini-frames, ordering them by their center positions in row-major order.
    - Keep the outer border of the large square in its original color, then fill inner shells by border distance using the center-color sequence cyclically.

    Key insight:
    - The small framed colors are not isolated answers; they form an ordered repeating program that drives the concentric recoloring of the big square.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    side, r0, c0, frame_color = _pick_big_square(g, bg)
    frames = _instruction_frames(g, bg)
    assert frames, "expected at least one instruction frame"
    program = [int(frame["color"]) for frame in frames]

    return _program_output(side, frame_color, program)


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, margin: int = 1) -> None:
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(blocked.shape[0], r0 + h + margin)
    cc1 = min(blocked.shape[1], c0 + w + margin)
    blocked[rr0:rr1, cc0:cc1] = True


def _free_patch_positions(blocked: np.ndarray, patch_h: int, patch_w: int) -> list[tuple[int, int]]:
    h, w = blocked.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(h - patch_h + 1):
        for c0 in range(w - patch_w + 1):
            if not blocked[r0 : r0 + patch_h, c0 : c0 + patch_w].any():
                positions.append((r0, c0))
    return positions


def _draw_big_frame(grid: np.ndarray, r0: int, c0: int, side: int, color: int) -> None:
    grid[r0, c0 : c0 + side] = color
    grid[r0 + side - 1, c0 : c0 + side] = color
    grid[r0 : r0 + side, c0] = color
    grid[r0 : r0 + side, c0 + side - 1] = color


def _draw_valid_frame(grid: np.ndarray, r0: int, c0: int, ring_color: int, center_color: int) -> None:
    for dr, dc in RING_COORDS:
        grid[r0 + dr, c0 + dc] = ring_color
    grid[r0 + 1, c0 + 1] = center_color


def _draw_partial_frame(grid: np.ndarray, r0: int, c0: int, ring_color: int, center_color: int, style: str) -> None:
    patch = np.zeros((3, 3), dtype=int)
    for dr, dc in RING_COORDS:
        patch[dr, dc] = ring_color
    patch[1, 1] = center_color

    if style == "missing_one":
        dr, dc = random.choice(RING_COORDS)
        patch[dr, dc] = BG
    elif style == "missing_two":
        for dr, dc in random.sample(list(RING_COORDS), 2):
            patch[dr, dc] = BG
    else:
        dr, dc = random.choice(RING_COORDS)
        wrong_color = random.choice([c for c in PALETTE if c not in {ring_color, center_color}])
        patch[dr, dc] = wrong_color

    mask = patch != BG
    grid[r0 : r0 + 3, c0 : c0 + 3][mask] = patch[mask]


def _row_major_differs_from_column_major(centers: list[tuple[int, int]]) -> bool:
    row_major = sorted(centers)
    col_major = sorted(centers, key=lambda rc: (rc[1], rc[0]))
    return row_major != col_major


def _fill_scatter(grid: np.ndarray, blocked: np.ndarray, colors: list[int], count: int) -> bool:
    free = list(map(tuple, np.argwhere((grid == BG) & (~blocked))))
    if len(free) < count:
        return False
    random.shuffle(free)
    for (r, c), color in zip(free[:count], (random.choice(colors) for _ in range(count))):
        grid[int(r), int(c)] = int(color)
    return True


def _signature(arr: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, arr.shape)), arr.tobytes()


def _validate_case(
    grid: np.ndarray,
    big_spec: tuple[int, int, int, int],
    centers_row_major: list[tuple[int, int]],
    program: list[int],
    partial_count: int,
    scatter_count: int,
) -> bool:
    bg = _get_bg(grid)
    if bg != BG:
        return False

    side, r0, c0, frame_color = big_spec
    try:
        found_big = _pick_big_square(grid, bg)
    except AssertionError:
        return False
    if found_big != big_spec:
        return False

    frames = _instruction_frames(grid, bg)
    found_centers = [tuple(frame["center"]) for frame in frames]  # type: ignore[arg-type]
    found_program = [int(frame["color"]) for frame in frames]
    if found_centers != centers_row_major or found_program != program:
        return False

    if not _row_major_differs_from_column_major(centers_row_major):
        return False
    if partial_count < 1 or scatter_count < 1:
        return False
    if len(set(program)) != len(program):
        return False

    max_depth = (side - 1) // 2
    if max_depth <= len(program):
        return False

    expected = _program_output(side, frame_color, program)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    used_program_colors = set(int(v) for v in np.unique(actual)) - {frame_color}
    if used_program_colors != set(program):
        return False

    col_major_frames = sorted(frames, key=lambda item: item["center"][1::-1])  # type: ignore[index]
    col_program = [int(frame["color"]) for frame in col_major_frames]
    if col_program != program and np.array_equal(expected, _program_output(side, frame_color, col_program)):
        return False

    reversed_program = list(reversed(program))
    if reversed_program != program and np.array_equal(expected, _program_output(side, frame_color, reversed_program)):
        return False

    return True


def _generate_once() -> np.ndarray | None:
    h = random.randint(22, 30)
    w = random.randint(22, 30)
    program_len = random.randint(2, 4)

    min_side = 2 * program_len + 3
    max_side = min(h, w) - 8
    if max_side < min_side:
        return None

    side = random.randint(min_side, max_side)
    frame_color = random.choice(PALETTE)
    top = random.randint(2, h - side - 2)
    left = random.randint(2, w - side - 2)

    grid = np.zeros((h, w), dtype=int)
    _draw_big_frame(grid, top, left, side, frame_color)

    blocked = np.zeros((h, w), dtype=bool)
    _mark_blocked(blocked, top, left, side, side, margin=1)

    valid_positions: list[tuple[int, int]] = []
    for _ in range(program_len):
        options = _free_patch_positions(blocked, 3, 3)
        if not options:
            return None
        pos = random.choice(options)
        valid_positions.append(pos)
        _mark_blocked(blocked, pos[0], pos[1], 3, 3, margin=1)

    centers_row_major = sorted((r0 + 1, c0 + 1) for r0, c0 in valid_positions)
    if not _row_major_differs_from_column_major(centers_row_major):
        return None

    program = random.sample([c for c in PALETTE if c != frame_color], program_len)
    pos_by_center = {center: pos for center, pos in zip(((r0 + 1, c0 + 1) for r0, c0 in valid_positions), valid_positions)}
    for center, color in zip(centers_row_major, program):
        r0, c0 = pos_by_center[center]
        ring_color = random.choice([c for c in PALETTE if c != color])
        _draw_valid_frame(grid, r0, c0, ring_color, color)

    partial_count = random.randint(1, 3)
    for _ in range(partial_count):
        options = _free_patch_positions(blocked, 3, 3)
        if not options:
            return None
        r0, c0 = random.choice(options)
        blocked[r0 : r0 + 3, c0 : c0 + 3] = True
        _mark_blocked(blocked, r0, c0, 3, 3, margin=1)
        center_color = random.choice(program)
        ring_color = random.choice([c for c in PALETTE if c != center_color])
        style = random.choice(["missing_one", "missing_two", "wrong_one"])
        _draw_partial_frame(grid, r0, c0, ring_color, center_color, style)

    scatter_count = random.randint(4, 10)
    used_colors = [frame_color, *program]
    if not _fill_scatter(grid, blocked, used_colors, scatter_count):
        return None

    big_spec = (side, top, left, frame_color)
    if not _validate_case(grid, big_spec, centers_row_major, program, partial_count, scatter_count):
        return None

    out_sig = _signature(main(grid))
    in_sig = _signature(grid)
    if in_sig in SEEN_INPUT_SIGS or out_sig in SEEN_OUTPUT_SIGS:
        return None
    SEEN_INPUT_SIGS.add(in_sig)
    SEEN_OUTPUT_SIGS.add(out_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 环色程序 input")
