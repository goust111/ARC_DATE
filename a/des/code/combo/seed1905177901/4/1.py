from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import permutations
import random

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
DIRECTIONS = ("up", "down", "left", "right")
GEN_ATTEMPTS = 300
MIN_SIDE = 18
MAX_SIDE = 30
GROUP_GAP = 3


@dataclass(frozen=True)
class FrameInfo:
    color: int
    bbox: tuple[int, int, int, int]
    direction: str
    gap_width: int
    area: int


@dataclass(frozen=True)
class LocalFrame:
    bbox: tuple[int, int, int, int]
    direction: str
    gap_width: int
    gap_start: int
    area: int


@dataclass(frozen=True)
class GroupLayout:
    direction: str
    frames: tuple[LocalFrame, ...]
    height: int
    width: int


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _is_contiguous(indices: np.ndarray) -> bool:
    return bool(indices.size) and np.all(indices == np.arange(indices[0], indices[0] + indices.size))


def _analyze_frame(comp: dict) -> FrameInfo:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        raise ValueError("bbox too small to be a frame")

    mask = np.zeros((h, w), dtype=bool)
    for rr, cc in comp["coords"]:
        mask[int(rr) - r0, int(cc) - c0] = True

    if np.any(mask[1:-1, 1:-1]):
        raise ValueError("frame interior must be empty")
    if not bool(mask[0, 0] and mask[0, -1] and mask[-1, 0] and mask[-1, -1]):
        raise ValueError("corners must be present")

    side_runs = {
        "up": np.flatnonzero(~mask[0, 1:-1]),
        "down": np.flatnonzero(~mask[-1, 1:-1]),
        "left": np.flatnonzero(~mask[1:-1, 0]),
        "right": np.flatnonzero(~mask[1:-1, -1]),
    }
    open_sides = [name for name, run in side_runs.items() if run.size > 0]
    if len(open_sides) != 1:
        raise ValueError("expected exactly one gapped side")

    direction = open_sides[0]
    gap_run = side_runs[direction]
    if not _is_contiguous(gap_run):
        raise ValueError("gap must be contiguous")
    gap_width = int(gap_run.size)

    if direction in ("up", "down"):
        if np.any(~mask[1:-1, 0]) or np.any(~mask[1:-1, -1]) or np.any(~mask[-1 if direction == "up" else 0, 1:-1]):
            raise ValueError("three non-gapped sides must be complete")
    else:
        if np.any(~mask[0, 1:-1]) or np.any(~mask[-1, 1:-1]) or np.any(~mask[1:-1, -1 if direction == "left" else 0]):
            raise ValueError("three non-gapped sides must be complete")

    expected_pixels = 2 * h + 2 * w - 4 - gap_width
    if int(mask.sum()) != expected_pixels:
        raise ValueError("pixel count does not match a 1px frame with one gap")

    return FrameInfo(
        color=int(comp["color"]),
        bbox=(r0, c0, r1, c1),
        direction=direction,
        gap_width=gap_width,
        area=h * w,
    )


def _extract_frames(grid: np.ndarray) -> list[FrameInfo]:
    bg = _get_bg(grid)
    frames = []
    for comp in _find_components(grid, bg):
        frames.append(_analyze_frame(comp))
    if not frames:
        raise ValueError("expected at least one frame")
    return frames


def _draw_frame(
    canvas: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
    direction: str,
    gap_width: int,
    bg: int,
    gap_start: int | None = None,
) -> None:
    r0, c0, r1, c1 = bbox
    canvas[r0, c0 : c1 + 1] = color
    canvas[r1, c0 : c1 + 1] = color
    canvas[r0 : r1 + 1, c0] = color
    canvas[r0 : r1 + 1, c1] = color

    inner_len = (c1 - c0 - 1) if direction in ("up", "down") else (r1 - r0 - 1)
    if not (1 <= gap_width <= inner_len):
        raise ValueError("gap does not fit the frame side")
    if gap_start is None:
        gap_start = (inner_len - gap_width) // 2
    if not (0 <= gap_start <= inner_len - gap_width):
        raise ValueError("gap start out of range")

    if direction == "up":
        start = c0 + 1 + gap_start
        canvas[r0, start : start + gap_width] = bg
    elif direction == "down":
        start = c0 + 1 + gap_start
        canvas[r1, start : start + gap_width] = bg
    elif direction == "left":
        start = r0 + 1 + gap_start
        canvas[start : start + gap_width, c0] = bg
    elif direction == "right":
        start = r0 + 1 + gap_start
        canvas[start : start + gap_width, c1] = bg
    else:
        raise ValueError(f"unknown direction: {direction}")


def _union_bbox(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge rectangular frames by the side that contains their gap.

    The merged frame uses the union bbox of its direction-group, keeps the same
    opening side, and uses a centered gap whose width is the sum of the group's
    original gap widths.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    bg = _get_bg(g)
    frames = _extract_frames(g)
    groups: dict[str, list[FrameInfo]] = defaultdict(list)
    for frame in frames:
        groups[frame.direction].append(frame)

    out = np.full_like(g, bg)
    for direction, items in groups.items():
        bbox = _union_bbox([item.bbox for item in items])
        total_gap = sum(item.gap_width for item in items)
        r0, c0, r1, c1 = bbox
        inner_len = (c1 - c0 - 1) if direction in ("up", "down") else (r1 - r0 - 1)
        if total_gap > inner_len:
            raise ValueError("merged gap does not fit the union bbox")
        winner = max(items, key=lambda item: item.area)
        _draw_frame(out, bbox, winner.color, direction, total_gap, bg)
    return out


def _build_group(direction: str, count: int) -> GroupLayout | None:
    frames: list[LocalFrame] = []
    used_areas: set[int] = set()

    if direction in ("up", "down"):
        x = 0
        max_bottom = 0
        for i in range(count):
            gap_width = random.randint(1, 3)
            for _ in range(30):
                height = random.randint(5, 7)
                width = random.randint(max(5, gap_width + 2), 6 if count >= 3 else 8)
                area = height * width
                if area not in used_areas:
                    break
            else:
                return None
            y = random.randint(0, 2)
            if i > 0:
                x += random.randint(1, 3)
            gap_start = random.randint(0, width - 2 - gap_width)
            bbox = (y, x, y + height - 1, x + width - 1)
            frames.append(LocalFrame(bbox, direction, gap_width, gap_start, area))
            used_areas.add(area)
            x += width
            max_bottom = max(max_bottom, y + height)
        group_width = x
        group_height = max_bottom
        if sum(frame.gap_width for frame in frames) > group_width - 2:
            return None
    else:
        y = 0
        max_right = 0
        for i in range(count):
            gap_width = random.randint(1, 3)
            for _ in range(30):
                width = random.randint(5, 7)
                height = random.randint(max(5, gap_width + 2), 6 if count >= 3 else 8)
                area = height * width
                if area not in used_areas:
                    break
            else:
                return None
            x = random.randint(0, 2)
            if i > 0:
                y += random.randint(1, 3)
            gap_start = random.randint(0, height - 2 - gap_width)
            bbox = (y, x, y + height - 1, x + width - 1)
            frames.append(LocalFrame(bbox, direction, gap_width, gap_start, area))
            used_areas.add(area)
            y += height
            max_right = max(max_right, x + width)
        group_height = y
        group_width = max_right
        if sum(frame.gap_width for frame in frames) > group_height - 2:
            return None

    return GroupLayout(direction, tuple(frames), group_height, group_width)


def _pack_groups(groups: list[GroupLayout]) -> tuple[list[tuple[int, int]], int, int] | None:
    best: tuple[tuple[int, int, int], list[tuple[int, int]], int, int] | None = None
    indices = list(range(len(groups)))

    for perm in permutations(indices):
        for split in range(len(groups) + 1):
            row0 = list(perm[:split])
            row1 = list(perm[split:])
            if not row0:
                continue

            width0 = sum(groups[i].width for i in row0) + GROUP_GAP * max(0, len(row0) - 1)
            width1 = sum(groups[i].width for i in row1) + GROUP_GAP * max(0, len(row1) - 1)
            height0 = max(groups[i].height for i in row0)
            height1 = max((groups[i].height for i in row1), default=0)
            total_w = max(width0, width1)
            total_h = height0 + (GROUP_GAP + height1 if row1 else 0)

            score = (max(total_h, total_w), total_h * total_w, total_w + total_h)
            if best is not None and score >= best[0]:
                continue

            positions = [(-1, -1)] * len(groups)
            x = 0
            for idx in row0:
                positions[idx] = (0, x)
                x += groups[idx].width + GROUP_GAP
            if row1:
                x = 0
                y = height0 + GROUP_GAP
                for idx in row1:
                    positions[idx] = (y, x)
                    x += groups[idx].width + GROUP_GAP
            best = (score, positions, total_h, total_w)

    if best is None:
        return None

    _, positions, height, width = best
    return positions, height, width


def _shift_bbox(bbox: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return (r0 + dr, c0 + dc, r1 + dr, c1 + dc)


def _expected_output_signature(frames: list[FrameInfo]) -> set[tuple[str, tuple[int, int, int, int], int, int]]:
    groups: dict[str, list[FrameInfo]] = defaultdict(list)
    for frame in frames:
        groups[frame.direction].append(frame)

    sig = set()
    for direction, items in groups.items():
        winner = max(items, key=lambda item: item.area)
        sig.add(
            (
                direction,
                _union_bbox([item.bbox for item in items]),
                sum(item.gap_width for item in items),
                winner.color,
            )
        )
    return sig


def _observed_signature(frames: list[FrameInfo]) -> set[tuple[str, tuple[int, int, int, int], int, int]]:
    return {(frame.direction, frame.bbox, frame.gap_width, frame.color) for frame in frames}


def _gen_attempt() -> np.ndarray | None:
    group_count = random.randint(2, 4)
    directions = random.sample(list(DIRECTIONS), group_count)
    max_per_group = 3 if group_count <= 2 else 2
    counts = [random.randint(1, max_per_group) for _ in directions]
    if all(count == 1 for count in counts):
        counts[random.randrange(group_count)] = max_per_group
    if sum(counts) > len(PALETTE):
        return None

    groups: list[GroupLayout] = []
    for direction, count in zip(directions, counts):
        layout = _build_group(direction, count)
        if layout is None:
            return None
        groups.append(layout)

    packed = _pack_groups(groups)
    if packed is None:
        return None
    positions, packed_h, packed_w = packed

    pad_top = random.randint(1, 3)
    pad_left = random.randint(1, 3)
    pad_bottom = random.randint(1, 3)
    pad_right = random.randint(1, 3)
    grid_h = packed_h + pad_top + pad_bottom
    grid_w = packed_w + pad_left + pad_right
    if not (MIN_SIDE <= grid_h <= MAX_SIDE and MIN_SIDE <= grid_w <= MAX_SIDE):
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    colors = random.sample(PALETTE, sum(counts))
    color_idx = 0
    placed_frames: list[FrameInfo] = []

    for group, (base_r, base_c) in zip(groups, positions):
        row_slack = max(0, group.height - max(frame.bbox[2] + 1 for frame in group.frames))
        col_slack = max(0, group.width - max(frame.bbox[3] + 1 for frame in group.frames))
        shift_r = pad_top + base_r + (random.randint(0, row_slack) if row_slack else 0)
        shift_c = pad_left + base_c + (random.randint(0, col_slack) if col_slack else 0)

        for frame in group.frames:
            color = colors[color_idx]
            color_idx += 1
            bbox = _shift_bbox(frame.bbox, shift_r, shift_c)
            _draw_frame(grid, bbox, color, frame.direction, frame.gap_width, BG, gap_start=frame.gap_start)
            placed_frames.append(
                FrameInfo(
                    color=color,
                    bbox=bbox,
                    direction=frame.direction,
                    gap_width=frame.gap_width,
                    area=frame.area,
                )
            )

    try:
        parsed_input = _extract_frames(grid)
    except ValueError:
        return None

    expected_input = {(frame.color, frame.bbox, frame.direction, frame.gap_width) for frame in placed_frames}
    observed_input = {(frame.color, frame.bbox, frame.direction, frame.gap_width) for frame in parsed_input}
    if expected_input != observed_input:
        return None

    out = main(grid)
    try:
        parsed_output = _extract_frames(out)
    except ValueError:
        return None

    if _expected_output_signature(placed_frames) != _observed_signature(parsed_output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
