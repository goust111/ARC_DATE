from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

STAMP_COLORS = (RED, BLUE, GREEN, YELLOW)
FRAME_SIZE = 5
FRAME_INNER_SIZE = 3
MAX_RETRIES = 300

MISSING_TO_DIR = {
    (0, 0): (-1, -1),
    (0, 1): (-1, 1),
    (1, 0): (1, -1),
    (1, 1): (1, 1),
}

PATTERN_OFFSETS = {
    RED: [(r, c) for r in range(3) for c in range(3)],
    BLUE: [(1, 1)],
    GREEN: [(0, 0), (0, 2), (2, 0), (2, 2)],
    YELLOW: [(0, 1), (1, 0), (1, 2), (2, 1)],
}


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _background_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel())
    return int(counts.argmax())


def _find_frames(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    frames: list[tuple[int, int]] = []
    for r0 in range(h - FRAME_SIZE + 1):
        for c0 in range(w - FRAME_SIZE + 1):
            sub = grid[r0 : r0 + FRAME_SIZE, c0 : c0 + FRAME_SIZE]
            if not np.all(sub[1:-1, 1:-1] == bg):
                continue
            if not np.all(sub[0, :] == GREY):
                continue
            if not np.all(sub[-1, :] == GREY):
                continue
            if not np.all(sub[:, 0] == GREY):
                continue
            if not np.all(sub[:, -1] == GREY):
                continue
            frames.append((r0, c0))
    return frames


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _find_stamps(grid: np.ndarray) -> list[dict[str, object]]:
    stamps: list[dict[str, object]] = []
    for color in STAMP_COLORS:
        for pts in _components_of_color(grid, color):
            if len(pts) != 3:
                continue
            rows = [r for r, _ in pts]
            cols = [c for _, c in pts]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if r1 - r0 != 1 or c1 - c0 != 1:
                continue
            cells = set(pts)
            missing: list[tuple[int, int]] = []
            for dr in range(2):
                for dc in range(2):
                    if (r0 + dr, c0 + dc) not in cells:
                        missing.append((dr, dc))
            if len(missing) != 1:
                continue
            stamps.append(
                {
                    "color": color,
                    "top_left": (r0, c0),
                    "missing": missing[0],
                }
            )
    return stamps


def _select_target_frame(
    frames: list[tuple[int, int]], missing_cell: tuple[int, int], missing_offset: tuple[int, int]
) -> int | None:
    dir_r, dir_c = MISSING_TO_DIR[missing_offset]
    mr, mc = missing_cell
    best_idx: int | None = None
    best_dist2: int | None = None
    tied = False

    for idx, (fr, fc) in enumerate(frames):
        center_r, center_c = fr + 2, fc + 2
        dr = center_r - mr
        dc = center_c - mc
        if _sign(dr) != dir_r or _sign(dc) != dir_c:
            continue
        dist2 = dr * dr + dc * dc
        if best_dist2 is None or dist2 < best_dist2:
            best_dist2 = dist2
            best_idx = idx
            tied = False
        elif dist2 == best_dist2:
            tied = True

    return None if tied else best_idx


def main(grid: np.ndarray) -> np.ndarray:
    """
    Each L-shaped triomino is a stamp whose missing-corner direction picks a grey
    5x5 frame, and whose color selects the 3x3 pattern drawn inside that frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    frames = _find_frames(g, bg)
    stamps = _find_stamps(g)

    out = g.copy()
    for stamp in stamps:
        r0, c0 = stamp["top_left"]
        missing = stamp["missing"]
        color = int(stamp["color"])
        mr, mc = r0 + missing[0], c0 + missing[1]
        target_idx = _select_target_frame(frames, (mr, mc), missing)
        assert target_idx is not None, "each stamp must point to a unique frame"
        fr, fc = frames[target_idx]
        for dr, dc in PATTERN_OFFSETS[color]:
            out[fr + 1 + dr, fc + 1 + dc] = color
    return out


def _draw_frame(grid: np.ndarray, top_left: tuple[int, int]) -> None:
    r0, c0 = top_left
    r1, c1 = r0 + FRAME_SIZE - 1, c0 + FRAME_SIZE - 1
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _stamp_cells(top_left: tuple[int, int], missing: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = top_left
    cells: list[tuple[int, int]] = []
    for dr in range(2):
        for dc in range(2):
            if (dr, dc) != missing:
                cells.append((r0 + dr, c0 + dc))
    return cells


def _draw_stamp(grid: np.ndarray, top_left: tuple[int, int], missing: tuple[int, int], color: int) -> None:
    for r, c in _stamp_cells(top_left, missing):
        grid[r, c] = color


def _slot_positions(length: int) -> list[int]:
    positions: list[int] = []
    pos = random.randint(1, 3)
    while pos <= length - FRAME_SIZE - 1:
        positions.append(pos)
        pos += random.randint(7, 9)
    return positions


def _padded_clear(grid: np.ndarray, top_left: tuple[int, int]) -> bool:
    h, w = grid.shape
    r0, c0 = top_left
    if np.any(grid[r0 : r0 + 2, c0 : c0 + 2] != BLACK):
        return False
    rr0, rr1 = max(0, r0 - 1), min(h, r0 + 3)
    cc0, cc1 = max(0, c0 - 1), min(w, c0 + 3)
    return bool(np.all(grid[rr0:rr1, cc0:cc1] == BLACK))


def _candidate_stamp_positions(
    grid: np.ndarray,
    frames: list[tuple[int, int]],
    target_idx: int,
    missing: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = grid.shape
    fr, fc = frames[target_idx]
    center_r, center_c = fr + 2, fc + 2
    candidates: list[tuple[int, int]] = []
    for r0 in range(h - 1):
        for c0 in range(w - 1):
            if not _padded_clear(grid, (r0, c0)):
                continue
            mr, mc = r0 + missing[0], c0 + missing[1]
            if _select_target_frame(frames, (mr, mc), missing) != target_idx:
                continue
            if abs(center_r - mr) < 3 or abs(center_c - mc) < 3:
                continue
            candidates.append((r0, c0))
    return candidates


def _expected_output(
    grid: np.ndarray,
    frames: list[tuple[int, int]],
    placed_stamps: list[tuple[tuple[int, int], tuple[int, int], int, int]],
) -> np.ndarray:
    out = grid.copy()
    for _top_left, _missing, color, target_idx in placed_stamps:
        fr, fc = frames[target_idx]
        for dr, dc in PATTERN_OFFSETS[color]:
            out[fr + 1 + dr, fc + 1 + dc] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    row_slots = _slot_positions(h)
    col_slots = _slot_positions(w)
    if len(row_slots) < 2 or len(col_slots) < 2:
        return None

    slots = [(r, c) for r in row_slots for c in col_slots]
    if len(slots) < 2:
        return None

    n_frames = random.randint(2, min(6, len(slots)))
    frames = random.sample(slots, n_frames)
    grid = np.full((h, w), BLACK, dtype=int)
    for frame in frames:
        _draw_frame(grid, frame)

    max_stamps = min(4, n_frames)
    min_stamps = 1 if max_stamps == 1 else random.choice([1, 2])
    n_stamps = random.randint(min_stamps, max_stamps)
    target_indices = random.sample(range(n_frames), n_stamps)
    random.shuffle(target_indices)

    placed_stamps: list[tuple[tuple[int, int], tuple[int, int], int, int]] = []
    for target_idx in target_indices:
        orientation_order = random.sample(list(MISSING_TO_DIR.keys()), len(MISSING_TO_DIR))
        chosen_top_left: tuple[int, int] | None = None
        chosen_missing: tuple[int, int] | None = None
        for missing in orientation_order:
            candidates = _candidate_stamp_positions(grid, frames, target_idx, missing)
            if not candidates:
                continue
            chosen_top_left = random.choice(candidates)
            chosen_missing = missing
            break
        if chosen_top_left is None or chosen_missing is None:
            return None

        color = random.choice(STAMP_COLORS)
        _draw_stamp(grid, chosen_top_left, chosen_missing, color)
        placed_stamps.append((chosen_top_left, chosen_missing, color, target_idx))

    expected = _expected_output(grid, frames, placed_stamps)
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    if np.array_equal(expected, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate a valid input after {MAX_RETRIES} attempts")
