from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


GREY = 5
TEAL = 8
ALL_COLORS = tuple(range(10))
FRAME_HEIGHTS = (5, 7, 9)
FRAME_WIDTHS = (5, 7, 9, 11)
ROW_PATTERNS = {
    2: ((2,), (1, 1)),
    3: ((3,), (2, 1)),
    4: ((4,), (2, 2), (3, 1)),
    5: ((5,), (3, 2), (4, 1)),
}
MAX_GENERATION_ATTEMPTS = 400


@dataclass(frozen=True)
class Frame:
    color: int
    r0: int
    c0: int
    r1: int
    c1: int
    marker_count: int

    @property
    def capacity(self) -> int:
        return len(_dot_columns(self.c0, self.c1))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


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
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _label_components(mask: np.ndarray) -> np.ndarray:
    labels = np.zeros(mask.shape, dtype=int)
    for idx, comp in enumerate(_components(mask), start=1):
        for r, c in comp:
            labels[r, c] = idx
    return labels


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_mask(shape: tuple[int, int], r0: int, c0: int, r1: int, c1: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return mask


def _dot_columns(c0: int, c1: int) -> list[int]:
    return list(range(c1 - 2, c0, -2))


def _frame_interior_cells(frame: Frame) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(frame.r0 + 1, frame.r1)
        for c in range(frame.c0 + 1, frame.c1)
    ]


def _extract_frames(grid: np.ndarray, bg: int) -> list[Frame]:
    colors = sorted(set(map(int, grid.flatten())) - {bg, GREY, TEAL})
    frames: list[Frame] = []
    grey_mask = grid == GREY

    for color in colors:
        color_mask = grid == color
        if not np.any(color_mask):
            continue
        r0, c0, r1, c1 = _bbox(color_mask)
        if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
            continue
        expected = _border_mask(grid.shape, r0, c0, r1, c1)
        if not np.array_equal(color_mask, expected):
            continue
        marker_count = int(np.count_nonzero(grey_mask[r0 + 1 : r1, c0 + 1 : c1]))
        frames.append(Frame(color=color, r0=r0, c0=c0, r1=r1, c1=c1, marker_count=marker_count))

    return sorted(frames, key=lambda frame: (frame.r0, frame.c0, frame.color))


def _group_frames(grid: np.ndarray, frames: list[Frame]) -> list[list[Frame]]:
    structure = grid == TEAL
    for frame in frames:
        structure |= _border_mask(grid.shape, frame.r0, frame.c0, frame.r1, frame.c1)
    labels = _label_components(structure)

    groups: dict[int, list[Frame]] = {}
    for frame in frames:
        label = int(labels[frame.r0, frame.c0])
        groups.setdefault(label, []).append(frame)
    return list(groups.values())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge frame tallies through teal bridges, then refill each connected group by frame capacity order.

    Input structure:
    - The background is the most common color.
    - Several differently colored hollow rectangular frames appear, and grey marker pixels may sit inside them.
    - Teal bridge pixels outside the frames may 4-connect some frame borders into shared groups.

    Rule:
    - Build 4-connected components over frame-border pixels together with teal bridge pixels, ignoring grey markers.
    - For each connected frame group, count all grey markers in the group, then erase every grey marker and every teal bridge.
    - Each frame can display dots only on its center row, starting two cells left of the right border and then every other cell; fill group dots into frames from smaller capacity to larger until the group's total is exhausted.

    Key insight:
    - The input markers are not counted per frame: teal connectivity first creates a shared ledger, and only then is that total repacked into the ordered frame capacities.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames = _extract_frames(g, bg)
    out = g.copy()
    out[g == GREY] = bg
    out[g == TEAL] = bg

    for group in _group_frames(g, frames):
        total = sum(frame.marker_count for frame in group)
        remaining = total
        ordered = sorted(group, key=lambda frame: (frame.capacity, frame.r0, frame.c0))
        for frame in ordered:
            take = min(remaining, frame.capacity)
            mid_r = (frame.r0 + frame.r1) // 2
            for col in _dot_columns(frame.c0, frame.c1)[:take]:
                out[mid_r, col] = GREY
            remaining -= take
            if remaining <= 0:
                break

    return out


def _choose_layout(rng: np.random.Generator, n_frames: int) -> tuple[list[int], list[list[bool]]]:
    row_counts = list(ROW_PATTERNS[n_frames][int(rng.integers(0, len(ROW_PATTERNS[n_frames])) )])
    row_bridges: list[list[bool]] = []
    bridge_slots: list[tuple[int, int]] = []
    for row_idx, count in enumerate(row_counts):
        row = []
        for local_idx in range(count - 1):
            bridge = bool(rng.random() < 0.6)
            row.append(bridge)
            bridge_slots.append((row_idx, local_idx))
        row_bridges.append(row)

    if bridge_slots and not any(any(row) for row in row_bridges) and rng.random() < 0.8:
        row_idx, local_idx = bridge_slots[int(rng.integers(0, len(bridge_slots)))]
        row_bridges[row_idx][local_idx] = True
    if bridge_slots and all(flag for row in row_bridges for flag in row) and n_frames >= 4 and rng.random() < 0.35:
        row_idx, local_idx = bridge_slots[int(rng.integers(0, len(bridge_slots)))]
        row_bridges[row_idx][local_idx] = False

    return row_counts, row_bridges


def _run_lengths(row_counts: list[int], row_bridges: list[list[bool]]) -> list[list[int]]:
    runs: list[list[int]] = []
    start = 0
    for count, bridges in zip(row_counts, row_bridges, strict=False):
        run = [start]
        for offset, bridge in enumerate(bridges, start=1):
            if bridge:
                run.append(start + offset)
            else:
                runs.append(run)
                run = [start + offset]
        runs.append(run)
        start += count
    return runs


def _sample_specs(
    rng: np.random.Generator, row_counts: list[int], row_bridges: list[list[bool]]
) -> tuple[list[int], list[int]] | None:
    widths = [int(rng.choice(FRAME_WIDTHS)) for _ in range(sum(row_counts))]
    heights = [int(rng.choice(FRAME_HEIGHTS)) for _ in range(sum(row_counts))]
    caps = [(width - 3) // 2 for width in widths]

    for run in _run_lengths(row_counts, row_bridges):
        if len(run) > 1 and len({caps[idx] for idx in run}) == 1:
            return None
    return widths, heights


def _place_frames(
    rng: np.random.Generator,
    row_counts: list[int],
    row_bridges: list[list[bool]],
    widths: list[int],
    heights: list[int],
) -> tuple[int, int, list[Frame], list[list[int]]] | None:
    row_widths: list[int] = []
    row_max_heights: list[int] = []
    row_gaps: list[list[int]] = []
    cursor = 0
    for row_idx, count in enumerate(row_counts):
        bridges = row_bridges[row_idx]
        gaps = []
        for gap_idx in range(count - 1):
            if bridges[gap_idx]:
                gaps.append(int(rng.integers(2, 5)))
            else:
                gaps.append(int(rng.integers(3, 7)))
        row_gaps.append(gaps)
        row_widths.append(sum(widths[cursor : cursor + count]) + sum(gaps))
        row_max_heights.append(max(heights[cursor : cursor + count]))
        cursor += count

    if max(row_widths) + 2 > 30:
        return None

    vertical_gaps = [int(rng.integers(3, 7)) for _ in range(len(row_counts) - 1)]
    used_height = sum(row_max_heights) + sum(vertical_gaps)
    if used_height + 2 > 30:
        return None

    min_h = max(18, used_height + 2)
    min_w = max(18, max(row_widths) + 2)
    if min_h > 30 or min_w > 30:
        return None

    h = int(rng.integers(min_h, 31))
    w = int(rng.integers(min_w, 31))

    top_margin = int(rng.integers(1, h - used_height + 1))
    frames: list[Frame] = []
    row_frame_indices: list[list[int]] = []
    row_top = top_margin
    frame_idx = 0
    for row_idx, count in enumerate(row_counts):
        max_h = row_max_heights[row_idx]
        row_mid = row_top + max_h // 2
        row_width = row_widths[row_idx]
        left_margin = int(rng.integers(1, w - row_width + 1))
        col = left_margin
        indices: list[int] = []
        for local_idx in range(count):
            fh = heights[frame_idx]
            fw = widths[frame_idx]
            r0 = row_mid - fh // 2
            c0 = col
            frame = Frame(
                color=-1,
                r0=r0,
                c0=c0,
                r1=r0 + fh - 1,
                c1=c0 + fw - 1,
                marker_count=0,
            )
            frames.append(frame)
            indices.append(frame_idx)
            frame_idx += 1
            col += fw
            if local_idx < count - 1:
                col += row_gaps[row_idx][local_idx]
        row_frame_indices.append(indices)
        row_top += max_h
        if row_idx < len(vertical_gaps):
            row_top += vertical_gaps[row_idx]

    return h, w, frames, row_frame_indices


def _paint_frame(grid: np.ndarray, frame: Frame, color: int) -> Frame:
    grid[frame.r0 : frame.r1 + 1, frame.c0] = color
    grid[frame.r0 : frame.r1 + 1, frame.c1] = color
    grid[frame.r0, frame.c0 : frame.c1 + 1] = color
    grid[frame.r1, frame.c0 : frame.c1 + 1] = color
    return Frame(color=color, r0=frame.r0, c0=frame.c0, r1=frame.r1, c1=frame.c1, marker_count=0)


def _paint_bridges(
    grid: np.ndarray,
    frames: list[Frame],
    row_frame_indices: list[list[int]],
    row_bridges: list[list[bool]],
) -> None:
    for row_indices, bridges in zip(row_frame_indices, row_bridges, strict=False):
        for local_idx, bridge in enumerate(bridges):
            if not bridge:
                continue
            left = frames[row_indices[local_idx]]
            right = frames[row_indices[local_idx + 1]]
            mid_r = (max(left.r0, right.r0) + min(left.r1, right.r1)) // 2
            grid[mid_r, left.c1 + 1 : right.c0] = TEAL


def _group_indices(row_frame_indices: list[list[int]], row_bridges: list[list[bool]]) -> list[list[int]]:
    groups: list[list[int]] = []
    for row_indices, bridges in zip(row_frame_indices, row_bridges, strict=False):
        current = [row_indices[0]]
        for local_idx, bridge in enumerate(bridges):
            if bridge:
                current.append(row_indices[local_idx + 1])
            else:
                groups.append(current)
                current = [row_indices[local_idx + 1]]
        groups.append(current)
    return groups


def _add_markers(
    rng: np.random.Generator,
    grid: np.ndarray,
    frames: list[Frame],
    groups: list[list[int]],
) -> list[Frame] | None:
    updated = frames[:]
    for group in groups:
        capacities = [frames[idx].capacity for idx in group]
        total_capacity = sum(capacities)
        total_markers = int(rng.integers(1, total_capacity + 1))
        cells: list[tuple[int, int]] = []
        for idx in group:
            cells.extend(_frame_interior_cells(frames[idx]))
        if len(cells) < total_markers:
            return None
        chosen = rng.choice(len(cells), size=total_markers, replace=False)
        counts = {idx: 0 for idx in group}
        for raw_pos in np.atleast_1d(chosen).tolist():
            r, c = cells[int(raw_pos)]
            grid[r, c] = GREY
            for idx in group:
                frame = frames[idx]
                if frame.r0 < r < frame.r1 and frame.c0 < c < frame.c1:
                    counts[idx] += 1
                    break
        for idx in group:
            frame = frames[idx]
            updated[idx] = Frame(
                color=frame.color,
                r0=frame.r0,
                c0=frame.c0,
                r1=frame.r1,
                c1=frame.c1,
                marker_count=counts[idx],
            )
    return updated


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n_frames = int(rng.integers(2, 6))
    row_counts, row_bridges = _choose_layout(rng, n_frames)
    sampled = _sample_specs(rng, row_counts, row_bridges)
    if sampled is None:
        return None
    widths, heights = sampled
    placed = _place_frames(rng, row_counts, row_bridges, widths, heights)
    if placed is None:
        return None
    h, w, bare_frames, row_frame_indices = placed

    usable_colors = [color for color in ALL_COLORS if color not in (GREY, TEAL)]
    rng.shuffle(usable_colors)
    bg = usable_colors[0]
    frame_colors = usable_colors[1 : 1 + n_frames]
    if len(frame_colors) != n_frames:
        return None

    grid = np.full((h, w), bg, dtype=int)
    frames: list[Frame] = []
    for frame, color in zip(bare_frames, frame_colors, strict=False):
        frames.append(_paint_frame(grid, frame, int(color)))

    _paint_bridges(grid, frames, row_frame_indices, row_bridges)
    groups = _group_indices(row_frame_indices, row_bridges)
    frames_with_markers = _add_markers(rng, grid, frames, groups)
    if frames_with_markers is None:
        return None

    if np.array_equal(grid, main(grid)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
