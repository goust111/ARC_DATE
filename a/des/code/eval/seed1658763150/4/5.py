import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


BG = 0
FRAME_COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400
DIRS = ("up", "down", "left", "right")


def _bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _strictly_contains(outer: Tuple[int, int, int, int], inner: Tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _all_border_values_ok(g: np.ndarray, bbox: Tuple[int, int, int, int], color: int) -> bool:
    r0, c0, r1, c1 = bbox
    allowed = {BG, int(color)}
    top = set(map(int, g[r0, c0 : c1 + 1]))
    bottom = set(map(int, g[r1, c0 : c1 + 1]))
    left = set(map(int, g[r0 : r1 + 1, c0]))
    right = set(map(int, g[r0 : r1 + 1, c1]))
    return top <= allowed and bottom <= allowed and left <= allowed and right <= allowed


def _extract_gap_sets(g: np.ndarray, bbox: Tuple[int, int, int, int], color: int) -> Optional[Dict[str, set[int]]]:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return None
    if not _all_border_values_ok(g, bbox, color):
        return None

    top_colors = g[r0, c0 + 1 : c1]
    bottom_colors = g[r1, c0 + 1 : c1]
    left_colors = g[r0 + 1 : r1, c0]
    right_colors = g[r0 + 1 : r1, c1]
    if top_colors.size == 0 or bottom_colors.size == 0 or left_colors.size == 0 or right_colors.size == 0:
        return None
    if not np.any(top_colors == color):
        return None
    if not np.any(bottom_colors == color):
        return None
    if not np.any(left_colors == color):
        return None
    if not np.any(right_colors == color):
        return None
    if np.any(g[r0 + 1 : r1, c0 + 1 : c1] == color):
        return None

    return {
        "up": {int(c) for c in range(c0 + 1, c1) if int(g[r0, c]) == BG},
        "down": {int(c) for c in range(c0 + 1, c1) if int(g[r1, c]) == BG},
        "left": {int(r) for r in range(r0 + 1, r1) if int(g[r, c0]) == BG},
        "right": {int(r) for r in range(r0 + 1, r1) if int(g[r, c1]) == BG},
    }


def _find_frames(g: np.ndarray) -> List[Dict[str, object]]:
    frames: List[Dict[str, object]] = []
    for color in map(int, np.unique(g)):
        if color == BG:
            continue
        coords = np.argwhere(g == color)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        bbox = (int(r0), int(c0), int(r1), int(c1))
        if not np.all(
            (coords[:, 0] == r0) | (coords[:, 0] == r1) | (coords[:, 1] == c0) | (coords[:, 1] == c1)
        ):
            continue
        gaps = _extract_gap_sets(g, bbox, color)
        if gaps is None:
            continue
        frames.append({"color": int(color), "bbox": bbox, "gaps": gaps})
    return frames


def _group_towers(frames: Sequence[Dict[str, object]]) -> List[List[Dict[str, object]]]:
    by_center: Dict[Tuple[int, int], List[Dict[str, object]]] = {}
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[index]
        key = (int(r0 + r1), int(c0 + c1))
        by_center.setdefault(key, []).append(frame)

    towers: List[List[Dict[str, object]]] = []
    for group in by_center.values():
        ordered = sorted(group, key=lambda frame: _bbox_area(frame["bbox"]), reverse=True)  # type: ignore[index]
        if len(ordered) < 2:
            continue
        ok = True
        for outer, inner in zip(ordered, ordered[1:]):
            if not _strictly_contains(outer["bbox"], inner["bbox"]):  # type: ignore[index]
                ok = False
                break
        if ok:
            towers.append(ordered)
    return towers


def _core_pixels(g: np.ndarray, tower: Sequence[Dict[str, object]]) -> List[Tuple[int, int, int]]:
    inner_bbox = tower[-1]["bbox"]  # type: ignore[index]
    r0, c0, r1, c1 = inner_bbox
    frame_colors = {int(frame["color"]) for frame in tower}  # type: ignore[index]
    pixels: List[Tuple[int, int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            color = int(g[r, c])
            if color != BG and color not in frame_colors:
                pixels.append((int(r), int(c), color))
    return pixels


def _passes_all_layers(tower: Sequence[Dict[str, object]], r: int, c: int, direction: str) -> bool:
    for frame in tower:
        bbox = frame["bbox"]  # type: ignore[index]
        r0, c0, r1, c1 = bbox
        if direction == "up":
            if not (r > r0 and c in frame["gaps"]["up"]):  # type: ignore[index]
                return False
        elif direction == "down":
            if not (r < r1 and c in frame["gaps"]["down"]):  # type: ignore[index]
                return False
        elif direction == "left":
            if not (c > c0 and r in frame["gaps"]["left"]):  # type: ignore[index]
                return False
        else:
            if not (c < c1 and r in frame["gaps"]["right"]):  # type: ignore[index]
                return False
    return True


def _project_target(outer_bbox: Tuple[int, int, int, int], r: int, c: int, direction: str) -> Tuple[int, int]:
    r0, c0, r1, c1 = outer_bbox
    if direction == "up":
        return (int(r0 - 1), int(c))
    if direction == "down":
        return (int(r1 + 1), int(c))
    if direction == "left":
        return (int(r), int(c0 - 1))
    return (int(r), int(c1 + 1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    In nested concentric frame towers, aligned gaps form transparent channels that let core pixels appear outside.

    Input structure:
    - Background is black.
    - Each independent tower contains 2-4 nested rectangular 1-pixel frames with per-side gaps.
    - The innermost area contains a solid core pattern whose color(s) differ from the frame colors.

    Rule:
    - For every core pixel, test the four cardinal directions.
    - A direction succeeds only if every frame in that direction has a gap aligned with the pixel's row/column.
    - Successful rays paint exactly one pixel just outside the outermost frame, keeping the original frames unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    frames = _find_frames(g)
    towers = _group_towers(frames)
    assert towers, "expected at least one concentric frame tower"

    out = g.copy()
    for tower in towers:
        core = _core_pixels(g, tower)
        assert core, "expected visible core pixels inside each tower"
        outer_bbox = tower[0]["bbox"]  # type: ignore[index]
        for r, c, color in core:
            for direction in DIRS:
                if _passes_all_layers(tower, r, c, direction):
                    rr, cc = _project_target(outer_bbox, r, c, direction)
                    if 0 <= rr < h and 0 <= cc < w and int(out[rr, cc]) == BG:
                        out[rr, cc] = int(color)
    return out


def _sample_layer_counts(tower_count: int) -> List[int]:
    if tower_count == 1:
        return [random.randint(2, 4)]
    options = [(a, b) for a in range(2, 5) for b in range(2, 5) if a + b <= 7]
    return list(random.choice(options))


def _sample_core_shape() -> Tuple[int, int]:
    return random.choice(
        [
            (1, 2),
            (1, 3),
            (2, 1),
            (2, 2),
            (2, 3),
            (3, 1),
            (3, 2),
        ]
    )


def _open_intervals(
    vis_start: int,
    vis_end: int,
    side_ranges: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    layer_count = len(side_ranges)
    left_pin = random.randrange(layer_count)
    right_pin = random.randrange(layer_count)
    intervals: List[Tuple[int, int]] = []
    for layer, (side_start, side_end) in enumerate(side_ranges):
        start = vis_start if layer == left_pin else random.randint(side_start, vis_start)
        end = vis_end if layer == right_pin else random.randint(vis_end, side_end)
        if start == side_start and end == side_end:
            if random.random() < 0.5:
                start = random.randint(side_start + 1, vis_start)
            else:
                end = random.randint(vis_end, side_end - 1)
        intervals.append((int(start), int(end)))
    return intervals


def _random_side_interval(
    side_range: Tuple[int, int],
    allowed_ranges: Sequence[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    side_start, side_end = side_range
    candidates = [(lo, hi) for lo, hi in allowed_ranges if lo <= hi and side_start <= lo <= hi <= side_end]
    if not candidates:
        return None
    lo, hi = random.choice(candidates)
    a = int(random.randint(lo, hi))
    b = int(random.randint(lo, hi))
    return (min(a, b), max(a, b))


def _choose_side_modes() -> Dict[str, str]:
    while True:
        modes = {direction: random.choice(("open", "missing", "misaligned")) for direction in DIRS}
        open_count = sum(mode == "open" for mode in modes.values())
        if 1 <= open_count <= 3:
            return modes


def _assign_side_gaps(
    side_ranges: Sequence[Tuple[int, int]],
    core_start: int,
    core_end: int,
    mode: str,
) -> List[Optional[Tuple[int, int]]]:
    assert side_ranges
    assert all(side_start <= core_start <= core_end <= side_end for side_start, side_end in side_ranges)
    layer_count = len(side_ranges)

    if mode == "open":
        vis_start = random.randint(core_start, core_end)
        vis_end = random.randint(vis_start, core_end)
        return _open_intervals(vis_start, vis_end, side_ranges)

    if mode == "missing":
        blocker = random.randrange(layer_count)
        aligned = _open_intervals(core_start, core_end, side_ranges)
        out: List[Optional[Tuple[int, int]]] = []
        for layer in range(layer_count):
            out.append(None if layer == blocker else aligned[layer])
        return out

    # misaligned: every layer has a gap, but at least one layer avoids the core span.
    blocker = random.randrange(layer_count)
    side_start, side_end = side_ranges[blocker]
    outside_ranges: List[Tuple[int, int]] = []
    if side_start <= core_start - 1:
        outside_ranges.append((side_start, core_start - 1))
    if core_end + 1 <= side_end:
        outside_ranges.append((core_end + 1, side_end))
    if not outside_ranges:
        return _assign_side_gaps(side_ranges, core_start, core_end, "missing")

    aligned = _open_intervals(core_start, core_end, side_ranges)
    out = []
    for layer in range(layer_count):
        if layer == blocker:
            interval = _random_side_interval(side_ranges[layer], outside_ranges)
            assert interval is not None
            out.append(interval)
        else:
            out.append(aligned[layer])
    return out


def _draw_gap_on_frame(
    patch: np.ndarray,
    bbox: Tuple[int, int, int, int],
    direction: str,
    interval: Optional[Tuple[int, int]],
) -> None:
    if interval is None:
        return
    lo, hi = interval
    r0, c0, r1, c1 = bbox
    if direction == "up":
        patch[r0, lo : hi + 1] = BG
    elif direction == "down":
        patch[r1, lo : hi + 1] = BG
    elif direction == "left":
        patch[lo : hi + 1, c0] = BG
    else:
        patch[lo : hi + 1, c1] = BG


def _build_tower(frame_colors: Sequence[int], core_color: int) -> np.ndarray:
    layer_count = len(frame_colors)
    core_h, core_w = _sample_core_shape()
    inner_h = core_h + random.choice((2, 4))
    inner_w = core_w + random.choice((2, 4))
    inner_frame_h = inner_h + 2
    inner_frame_w = inner_w + 2
    outer_h = inner_frame_h + 4 * (layer_count - 1)
    outer_w = inner_frame_w + 4 * (layer_count - 1)
    patch = np.full((outer_h + 2, outer_w + 2), BG, dtype=int)

    frame_boxes: List[Tuple[int, int, int, int]] = []
    for layer in range(layer_count):
        inset = 1 + 2 * layer
        frame_h = outer_h - 4 * layer
        frame_w = outer_w - 4 * layer
        r0 = inset
        c0 = inset
        r1 = r0 + frame_h - 1
        c1 = c0 + frame_w - 1
        frame_boxes.append((int(r0), int(c0), int(r1), int(c1)))

    inner_r0, inner_c0, inner_r1, inner_c1 = frame_boxes[-1]
    interior_r0, interior_r1 = inner_r0 + 1, inner_r1 - 1
    interior_c0, interior_c1 = inner_c0 + 1, inner_c1 - 1

    top_margin = (inner_h - core_h) // 2
    left_margin = (inner_w - core_w) // 2
    core_r0 = interior_r0 + top_margin
    core_c0 = interior_c0 + left_margin
    core_r1 = core_r0 + core_h - 1
    core_c1 = core_c0 + core_w - 1

    modes = _choose_side_modes()
    side_gaps: Dict[str, List[Optional[Tuple[int, int]]]] = {}
    for direction in DIRS:
        if direction in ("up", "down"):
            side_ranges = [(bbox[1] + 1, bbox[3] - 1) for bbox in frame_boxes]
            core_start = core_c0
            core_end = core_c1
        else:
            side_ranges = [(bbox[0] + 1, bbox[2] - 1) for bbox in frame_boxes]
            core_start = core_r0
            core_end = core_r1
        side_gaps[direction] = _assign_side_gaps(
            side_ranges=side_ranges,
            core_start=int(core_start),
            core_end=int(core_end),
            mode=modes[direction],
        )

    for bbox, color in zip(frame_boxes, frame_colors):
        r0, c0, r1, c1 = bbox
        patch[r0, c0 : c1 + 1] = int(color)
        patch[r1, c0 : c1 + 1] = int(color)
        patch[r0 : r1 + 1, c0] = int(color)
        patch[r0 : r1 + 1, c1] = int(color)

    for layer, bbox in enumerate(frame_boxes):
        for direction in DIRS:
            _draw_gap_on_frame(patch, bbox, direction, side_gaps[direction][layer])

    patch[core_r0 : core_r1 + 1, core_c0 : core_c1 + 1] = int(core_color)
    return patch


def _place_horizontal(patches: Sequence[np.ndarray]) -> np.ndarray:
    heights = [patch.shape[0] for patch in patches]
    widths = [patch.shape[1] for patch in patches]
    gap = random.randint(2, 5)
    extra_top = random.randint(0, 3)
    extra_bottom = random.randint(0, 3)
    extra_left = random.randint(0, 3)
    extra_right = random.randint(0, 3)
    h = max(heights) + extra_top + extra_bottom
    w = sum(widths) + gap * (len(patches) - 1) + extra_left + extra_right
    grid = np.full((h, w), BG, dtype=int)

    cursor = extra_left
    for patch in patches:
        max_r = h - extra_bottom - patch.shape[0]
        min_r = extra_top
        r0 = random.randint(min_r, max_r)
        c0 = cursor
        grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch
        cursor += patch.shape[1] + gap
    return grid


def _place_vertical(patches: Sequence[np.ndarray]) -> np.ndarray:
    heights = [patch.shape[0] for patch in patches]
    widths = [patch.shape[1] for patch in patches]
    gap = random.randint(2, 5)
    extra_top = random.randint(0, 3)
    extra_bottom = random.randint(0, 3)
    extra_left = random.randint(0, 3)
    extra_right = random.randint(0, 3)
    h = sum(heights) + gap * (len(patches) - 1) + extra_top + extra_bottom
    w = max(widths) + extra_left + extra_right
    grid = np.full((h, w), BG, dtype=int)

    cursor = extra_top
    for patch in patches:
        max_c = w - extra_right - patch.shape[1]
        min_c = extra_left
        c0 = random.randint(min_c, max_c)
        r0 = cursor
        grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch
        cursor += patch.shape[0] + gap
    return grid


def _generate_once() -> Optional[np.ndarray]:
    tower_count = random.randint(1, 2)
    layer_counts = _sample_layer_counts(tower_count)
    total_layers = sum(layer_counts)
    if total_layers > len(FRAME_COLORS) - 1:
        return None

    frame_palette = list(FRAME_COLORS)
    random.shuffle(frame_palette)
    frame_colors = frame_palette[:total_layers]
    core_palette = [color for color in FRAME_COLORS if color not in frame_colors]
    if not core_palette:
        return None

    patches = []
    cursor = 0
    for layers in layer_counts:
        tower_frame_colors = frame_colors[cursor : cursor + layers]
        cursor += layers
        core_color = random.choice(core_palette)
        patch = _build_tower(tower_frame_colors, core_color)
        patches.append(patch)

    random.shuffle(patches)
    grid = _place_horizontal(patches) if tower_count == 1 or random.random() < 0.5 else _place_vertical(patches)
    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid puzzle input after {MAX_GENERATION_ATTEMPTS} attempts")
