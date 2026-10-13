from __future__ import annotations

import random

import numpy as np


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    comps: list[np.ndarray] = []

    for start_r in range(height):
        for start_c in range(width):
            if not mask[start_r, start_c] or seen[start_r, start_c]:
                continue

            stack = [(start_r, start_c)]
            seen[start_r, start_c] = True
            cells: list[tuple[int, int]] = []

            while stack:
                r, c = stack.pop()
                cells.append((int(r), int(c)))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            comps.append(np.array(cells, dtype=int))

    return comps


def _most_common(values: np.ndarray) -> int:
    uniq, counts = np.unique(values, return_counts=True)
    return int(uniq[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _find_frames(grid: np.ndarray) -> tuple[int, int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    bg = _most_common(g.reshape(-1))

    non_bg = g[g != bg]
    assert non_bg.size > 0, "expected at least one frame"
    frame_color = _most_common(non_bg)

    frames: list[dict[str, object]] = []
    for comp in _components4(g == bg):
        ir0, ic0, ir1, ic1 = _bbox(comp)
        if ir0 < 1 or ic0 < 1 or ir1 > height - 2 or ic1 > width - 2:
            continue
        if comp.shape[0] != (ir1 - ir0 + 1) * (ic1 - ic0 + 1):
            continue

        fr0, fc0, fr1, fc1 = ir0 - 1, ic0 - 1, ir1 + 1, ic1 + 1
        top = g[fr0, fc0 : fc1 + 1]
        bottom = g[fr1, fc0 : fc1 + 1]
        left = g[fr0 : fr1 + 1, fc0]
        right = g[fr0 : fr1 + 1, fc1]

        if not (
            int(g[fr0, fc0]) == frame_color
            and int(g[fr0, fc1]) == frame_color
            and int(g[fr1, fc0]) == frame_color
            and int(g[fr1, fc1]) == frame_color
        ):
            continue
        if not np.all(bottom == frame_color) or not np.all(right == frame_color):
            continue
        if top[1:-1].size == 0 or left[1:-1].size == 0:
            continue
        if np.any(top[1:-1] == bg) or np.any(left[1:-1] == bg):
            continue

        top_markers = [
            (int(fc0 + 1 + offset), int(color))
            for offset, color in enumerate(top[1:-1])
            if int(color) != frame_color
        ]
        left_markers = [
            (int(fr0 + 1 + offset), int(color))
            for offset, color in enumerate(left[1:-1])
            if int(color) != frame_color
        ]

        frames.append(
            {
                "outer": (int(fr0), int(fc0), int(fr1), int(fc1)),
                "inner": (int(ir0), int(ic0), int(ir1), int(ic1)),
                "top_markers": top_markers,
                "left_markers": left_markers,
            }
        )

    frames.sort(key=lambda frame: tuple(frame["outer"]))  # type: ignore[arg-type]
    assert frames, "expected at least one valid frame"
    return bg, frame_color, frames


def _apply_projection(grid: np.ndarray, frames: list[dict[str, object]]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)

    for frame in frames:
        ir0, ic0, ir1, ic1 = frame["inner"]  # type: ignore[misc]
        for col, color in frame["top_markers"]:  # type: ignore[misc]
            out[ir0 : ir1 + 1, col] = color
        for row, color in frame["left_markers"]:  # type: ignore[misc]
            out[row, ic0 : ic1 + 1] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project top-edge and left-edge frame markers into the enclosed interior to make a crossed fill pattern.

    Input structure:
    - The background is black, and several non-overlapping grey rectangular frames sit on it.
    - Each frame has a one-pixel border, except that some non-corner cells on the top edge and left edge are replaced by colored marker pixels.
    - The inside of every frame is a solid black rectangle before the transformation.

    Rule:
    - Keep every frame border cell exactly as it is, including the colored markers on the top and left edges.
    - For each frame, every colored marker on the top edge fills its whole interior column with that marker's color.
    - After all column fills, every colored marker on the left edge fills its whole interior row with that marker's color, overwriting any earlier column color at crossings.

    Key insight:
    - The enclosed background rectangle identifies each frame's interior, while the top and left border markers are two independent projection instruction sets whose execution order is visible at their intersections.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    _bg, _frame_color, frames = _find_frames(g)
    return _apply_projection(g, frames)


# --------------- generator ---------------

BG = 0
FRAME = 5
MARKER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_SIZE_RANGE = (12, 30)
FRAME_COUNT_CHOICES = [1, 2, 3, 4]
FRAME_COUNT_WEIGHTS = [2, 4, 3, 1]
INNER_HEIGHT_RANGE = (2, 8)
INNER_WIDTH_RANGE = (2, 9)
PLACEMENT_PAD = 1
FRAME_PLACEMENT_RETRIES = 150
GEN_RETRIES = 400

COVERAGE_MODES = ("sparse", "top_full", "left_full", "both_full")


def _pick_marker_offsets(length: int, axis: str, mode: str) -> list[int]:
    assert length >= 1
    make_full = mode == "both_full" or (mode == "top_full" and axis == "top") or (mode == "left_full" and axis == "left")
    if make_full:
        return list(range(length))
    max_k = max(1, length - 1)
    k = random.randint(1, max_k)
    return sorted(random.sample(range(length), k))


def _pick_marker_colors(top_count: int, left_count: int) -> tuple[list[int], list[int]]:
    top_colors = [random.choice(MARKER_COLORS) for _ in range(top_count)]
    left_colors = [random.choice(MARKER_COLORS) for _ in range(left_count)]
    if len(set(top_colors + left_colors)) == 1:
        choices = [c for c in MARKER_COLORS if c != top_colors[0]]
        left_colors[-1] = random.choice(choices)
    return top_colors, left_colors


def _bbox_clear(reserved: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(reserved.shape[0] - 1, r1 + pad)
    cc1 = min(reserved.shape[1] - 1, c1 + pad)
    return not bool(reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _mark_reserved(reserved: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(reserved.shape[0] - 1, r1 + pad)
    cc1 = min(reserved.shape[1] - 1, c1 + pad)
    reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _add_frame(
    grid: np.ndarray,
    reserved: np.ndarray,
    outer_r0: int,
    outer_c0: int,
    inner_h: int,
    inner_w: int,
    mode: str,
) -> dict[str, object]:
    outer_r1 = outer_r0 + inner_h + 1
    outer_c1 = outer_c0 + inner_w + 1
    inner_r0 = outer_r0 + 1
    inner_c0 = outer_c0 + 1
    inner_r1 = outer_r1 - 1
    inner_c1 = outer_c1 - 1

    grid[outer_r0, outer_c0 : outer_c1 + 1] = FRAME
    grid[outer_r1, outer_c0 : outer_c1 + 1] = FRAME
    grid[outer_r0 : outer_r1 + 1, outer_c0] = FRAME
    grid[outer_r0 : outer_r1 + 1, outer_c1] = FRAME

    top_offsets = _pick_marker_offsets(inner_w, "top", mode)
    left_offsets = _pick_marker_offsets(inner_h, "left", mode)
    top_colors, left_colors = _pick_marker_colors(len(top_offsets), len(left_offsets))

    top_markers: list[tuple[int, int]] = []
    for offset, color in zip(top_offsets, top_colors, strict=False):
        col = inner_c0 + offset
        grid[outer_r0, col] = color
        top_markers.append((int(col), int(color)))

    left_markers: list[tuple[int, int]] = []
    for offset, color in zip(left_offsets, left_colors, strict=False):
        row = inner_r0 + offset
        grid[row, outer_c0] = color
        left_markers.append((int(row), int(color)))

    _mark_reserved(reserved, outer_r0, outer_c0, outer_r1, outer_c1, PLACEMENT_PAD)
    return {
        "outer": (int(outer_r0), int(outer_c0), int(outer_r1), int(outer_c1)),
        "inner": (int(inner_r0), int(inner_c0), int(inner_r1), int(inner_c1)),
        "top_markers": top_markers,
        "left_markers": left_markers,
    }


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIZE_RANGE)
    width = random.randint(*GRID_SIZE_RANGE)
    grid = np.full((height, width), BG, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    target_frames = random.choices(FRAME_COUNT_CHOICES, weights=FRAME_COUNT_WEIGHTS, k=1)[0]
    frames: list[dict[str, object]] = []

    for _ in range(target_frames):
        placed = False
        for _attempt in range(FRAME_PLACEMENT_RETRIES):
            inner_h = random.randint(*INNER_HEIGHT_RANGE)
            inner_w = random.randint(*INNER_WIDTH_RANGE)
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            if outer_h > height or outer_w > width:
                continue

            r0 = random.randint(0, height - outer_h)
            c0 = random.randint(0, width - outer_w)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            if not _bbox_clear(reserved, r0, c0, r1, c1, PLACEMENT_PAD):
                continue

            mode = random.choice(COVERAGE_MODES)
            frames.append(_add_frame(grid, reserved, r0, c0, inner_h, inner_w, mode))
            placed = True
            break

        if not placed:
            return None

    non_bg = grid[grid != BG]
    if non_bg.size == 0 or _most_common(non_bg) != FRAME:
        return None

    if not any(
        any(tcol != lcol for _tc, tcol in frame["top_markers"] for _lr, lcol in frame["left_markers"])  # type: ignore[misc]
        for frame in frames
    ):
        return None

    expected = _apply_projection(grid, frames)
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, grid):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
