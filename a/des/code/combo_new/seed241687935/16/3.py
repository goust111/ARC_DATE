from __future__ import annotations

from collections import Counter, deque

import numpy as np


_RNG = np.random.default_rng()
_MAX_RETRIES = 400
_COLORS = tuple(range(10))


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _find_frames(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for comp in _components4(grid != bg):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if (r1 - r0 + 1) < 5 or (c1 - c0 + 1) < 5:
            continue

        comp_set = set(comp)
        if comp_set != _perimeter_cells(r0, c0, r1, c1):
            continue

        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        if inner.shape[0] < 3 or inner.shape[1] < 3:
            continue
        if not (
            np.all(inner[0, :] == bg)
            and np.all(inner[-1, :] == bg)
            and np.all(inner[:, 0] == bg)
            and np.all(inner[:, -1] == bg)
        ):
            continue
        if not np.any(inner[1:-1, 1:-1] != bg):
            continue

        counts = Counter(int(grid[r, c]) for r, c in comp)
        if len(counts) != 2:
            continue
        border_color, _ = counts.most_common(1)[0]
        marker_cells = [(r, c) for r, c in comp if int(grid[r, c]) != border_color]
        if len(marker_cells) != 1:
            continue
        mr, mc = marker_cells[0]

        on_horizontal_edge = mr in (r0, r1) and c0 < mc < c1
        on_vertical_edge = mc in (c0, c1) and r0 < mr < r1
        if on_horizontal_edge == on_vertical_edge:
            continue

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "border_color": border_color,
                "flip_lr": on_horizontal_edge,
            }
        )
    frames.sort(key=lambda item: tuple(item["bbox"]))  # stable scan order
    return frames


def _apply_frames(grid: np.ndarray, frames: list[dict[str, object]]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        border_color = int(frame["border_color"])
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        flipped = np.fliplr(inner) if bool(frame["flip_lr"]) else np.flipud(inner)
        out[r0 + 1 : r1, c0 + 1 : c1] = flipped
        out[r0, c0 : c1 + 1] = border_color
        out[r1, c0 : c1 + 1] = border_color
        out[r0 : r1 + 1, c0] = border_color
        out[r0 : r1 + 1, c1] = border_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flip each enclosed multicolor pattern according to the marked side of its rectangular frame.

    Input structure:
    - The grid has one dominant background color.
    - Several non-overlapping one-pixel rectangular frames share a border color, and each frame contains a multicolor pattern separated from the border by at least one background cell.
    - Exactly one border pixel on each frame is replaced by a distinct marker color.

    Rule:
    - Detect every rectangular frame together with its single marker pixel.
    - If the marker lies on the top or bottom edge, mirror the full interior area left-to-right.
    - If the marker lies on the left or right edge, mirror the full interior area top-to-bottom.
    - Restore the marker pixel to the frame color and leave everything outside the frames unchanged.

    Key insight:
    - The marker's exact coordinate is irrelevant; only whether it sits on a horizontal edge or a vertical edge determines the reflection axis.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    frames = _find_frames(g, bg)
    return _apply_frames(g, frames)


def _branch_types(n_frames: int) -> list[bool]:
    if n_frames == 2:
        kinds = [True, False]
    elif n_frames == 3:
        kinds = [True, False, bool(_RNG.integers(0, 2))]
    else:
        kinds = [True, True, False, False]
    _RNG.shuffle(kinds)
    return kinds


def _sample_layout() -> tuple[int, int, int]:
    n_frames = int(_RNG.integers(2, 5))
    if n_frames == 2:
        rows, cols = ((1, 2) if bool(_RNG.integers(0, 2)) else (2, 1))
    else:
        rows, cols = 2, 2
    return n_frames, rows, cols


def _slot_sizes(count: int) -> list[int]:
    if count == 1:
        low, high = 10, 15
    else:
        low, high = 8, 13
    return [int(_RNG.integers(low, high)) for _ in range(count)]


def _choose_colors() -> tuple[int, int, int, list[int]]:
    palette = list(_RNG.choice(_COLORS, size=10, replace=False))
    bg = int(palette[0])
    border = int(palette[1])
    marker = int(palette[2])
    pattern_colors = [int(c) for c in palette[3:]]
    return bg, border, marker, pattern_colors


def _make_pattern(inner_h: int, inner_w: int, bg: int, palette: list[int], flip_lr: bool) -> np.ndarray:
    usable_h = inner_h - 2
    usable_w = inner_w - 2
    assert usable_h >= 1 and usable_w >= 1

    max_colors = min(4, len(palette))
    for _ in range(200):
        pattern = np.full((inner_h, inner_w), bg, dtype=int)
        n_colors = int(_RNG.integers(2, max_colors + 1))
        colors = [int(c) for c in _RNG.choice(palette, size=n_colors, replace=False)]
        part_count = int(_RNG.integers(n_colors, n_colors + 3))

        for idx in range(part_count):
            color = colors[idx] if idx < n_colors else int(colors[int(_RNG.integers(0, len(colors)))])
            ph = int(_RNG.integers(1, min(3, usable_h) + 1))
            pw = int(_RNG.integers(1, min(3, usable_w) + 1))
            rr = int(_RNG.integers(1, inner_h - ph))
            cc = int(_RNG.integers(1, inner_w - pw))
            pattern[rr : rr + ph, cc : cc + pw] = color

        extra = int(_RNG.integers(0, 3))
        for _ in range(extra):
            rr = int(_RNG.integers(1, inner_h - 1))
            cc = int(_RNG.integers(1, inner_w - 1))
            pattern[rr, cc] = int(colors[int(_RNG.integers(0, len(colors)))])

        colored = pattern != bg
        if int(colored.sum()) < 4:
            continue
        if len({int(v) for v in pattern[colored]}) < 2:
            continue
        transformed = np.fliplr(pattern) if flip_lr else np.flipud(pattern)
        if np.array_equal(pattern, transformed):
            continue
        return pattern

    raise RuntimeError("failed to create an asymmetric pattern")


def _build_example() -> tuple[np.ndarray, list[dict[str, object]], int]:
    n_frames, rows, cols = _sample_layout()
    row_sizes = _slot_sizes(rows)
    col_sizes = _slot_sizes(cols)
    top = int(_RNG.integers(1, 3))
    bottom = int(_RNG.integers(1, 3))
    left = int(_RNG.integers(1, 3))
    right = int(_RNG.integers(1, 3))
    row_gaps = [int(_RNG.integers(1, 3)) for _ in range(rows - 1)]
    col_gaps = [int(_RNG.integers(1, 3)) for _ in range(cols - 1)]

    height = top + bottom + sum(row_sizes) + sum(row_gaps)
    width = left + right + sum(col_sizes) + sum(col_gaps)
    if height > 30 or width > 30:
        raise RuntimeError("layout exceeds ARC bounds")

    bg, border, marker, pattern_colors = _choose_colors()
    grid = np.full((height, width), bg, dtype=int)

    row_starts = [top]
    for size, gap in zip(row_sizes, row_gaps):
        row_starts.append(row_starts[-1] + size + gap)
    col_starts = [left]
    for size, gap in zip(col_sizes, col_gaps):
        col_starts.append(col_starts[-1] + size + gap)

    slot_indices = [(rr, cc) for rr in range(rows) for cc in range(cols)]
    active_slots = list(_RNG.choice(len(slot_indices), size=n_frames, replace=False))
    branch_types = _branch_types(n_frames)

    frames: list[dict[str, object]] = []
    for slot_id, flip_lr in zip(active_slots, branch_types):
        sr, sc = slot_indices[int(slot_id)]
        slot_h = row_sizes[sr]
        slot_w = col_sizes[sc]
        frame_h = int(_RNG.integers(7, slot_h + 1))
        frame_w = int(_RNG.integers(7, slot_w + 1))
        top_pad = int(_RNG.integers(0, slot_h - frame_h + 1))
        left_pad = int(_RNG.integers(0, slot_w - frame_w + 1))

        r0 = row_starts[sr] + top_pad
        c0 = col_starts[sc] + left_pad
        r1 = r0 + frame_h - 1
        c1 = c0 + frame_w - 1

        grid[r0, c0 : c1 + 1] = border
        grid[r1, c0 : c1 + 1] = border
        grid[r0 : r1 + 1, c0] = border
        grid[r0 : r1 + 1, c1] = border

        if flip_lr:
            marker_row = r0 if bool(_RNG.integers(0, 2)) else r1
            marker_col = int(_RNG.integers(c0 + 1, c1))
        else:
            marker_row = int(_RNG.integers(r0 + 1, r1))
            marker_col = c0 if bool(_RNG.integers(0, 2)) else c1
        grid[marker_row, marker_col] = marker

        inner_h = frame_h - 2
        inner_w = frame_w - 2
        pattern = _make_pattern(inner_h, inner_w, bg, pattern_colors, flip_lr)
        grid[r0 + 1 : r1, c0 + 1 : c1] = pattern

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "border_color": border,
                "flip_lr": flip_lr,
            }
        )

    color_counts = Counter(int(v) for v in grid.flat)
    bg_count = color_counts[bg]
    if bg_count != max(color_counts.values()) or sum(v == bg_count for v in color_counts.values()) != 1:
        raise RuntimeError("background is not dominant")

    return grid, frames, bg


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        try:
            grid, frames, bg = _build_example()
        except RuntimeError:
            continue

        detected = _find_frames(grid, bg)
        expected = _apply_frames(grid, frames)
        if len(detected) != len(frames):
            continue
        expected_sig = {
            (tuple(frame["bbox"]), int(frame["border_color"]), bool(frame["flip_lr"])) for frame in frames
        }
        detected_sig = {
            (tuple(frame["bbox"]), int(frame["border_color"]), bool(frame["flip_lr"])) for frame in detected
        }
        if detected_sig != expected_sig:
            continue
        if not np.array_equal(main(grid), expected):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
