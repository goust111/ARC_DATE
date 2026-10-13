from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, find_components, get_bg


_RNG = np.random.default_rng()
BG_COLOR = Color.BLACK
MARKER_COLOR = Color.GREY
FRAME_COLOR_POOL = [c for c in Color.NOT_BLACK if c != MARKER_COLOR]
MAX_GENERATION_ATTEMPTS = 500


def _edge_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _box_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _boxes_extent(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int]:
    r1 = max(box[2] for box in boxes)
    c1 = max(box[3] for box in boxes)
    return int(r1 + 1), int(c1 + 1)


def _shift_boxes(
    boxes: list[tuple[int, int, int, int]], dr: int, dc: int
) -> list[tuple[int, int, int, int]]:
    return [(r0 + dr, c0 + dc, r1 + dr, c1 + dc) for r0, c0, r1, c1 in boxes]


def _normalize_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    min_r = min(box[0] for box in boxes)
    min_c = min(box[1] for box in boxes)
    return _shift_boxes(boxes, -min_r, -min_c)


def _contains_point(box: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = box
    return r0 < r < r1 and c0 < c < c1


def _contains_box(
    outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]
) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _is_hollow_rect_component(component: dict) -> bool:
    r0, c0, r1, c1 = map(int, component["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False

    local = np.zeros((h, w), dtype=bool)
    coords = component["coords"]
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return np.array_equal(local, _edge_mask(h, w))


def _extract_frames_and_markers(
    grid: np.ndarray,
) -> tuple[int, list[dict], np.ndarray]:
    bg = int(get_bg(grid))
    components = find_components(grid, bg=bg, connectivity=4, monochromatic=True)

    frames: list[dict] = []
    frame_mask = np.zeros_like(grid, dtype=bool)
    for component in components:
        if not _is_hollow_rect_component(component):
            continue
        frames.append(
            {
                "bbox": tuple(map(int, component["bbox"])),
                "color": int(component["color"]),
            }
        )
        coords = component["coords"]
        frame_mask[coords[:, 0], coords[:, 1]] = True

    assert len(frames) >= 2
    assert len({frame["color"] for frame in frames}) == len(frames)

    marker_mask = (grid != bg) & ~frame_mask
    assert marker_mask.any()
    marker_colors = set(map(int, grid[marker_mask].ravel()))
    assert len(marker_colors) == 1
    markers = np.argwhere(marker_mask)
    return bg, frames, markers


def _ownership_counts(
    frames: list[dict], markers: np.ndarray, *, choose: str = "innermost"
) -> list[int]:
    counts = [0] * len(frames)
    areas = [_box_area(frame["bbox"]) for frame in frames]

    for r, c in markers:
        owners = [i for i, frame in enumerate(frames) if _contains_point(frame["bbox"], int(r), int(c))]
        assert owners
        if choose == "innermost":
            owner = min(owners, key=lambda i: areas[i])
        elif choose == "outermost":
            owner = max(owners, key=lambda i: areas[i])
        else:
            raise ValueError(f"unknown choose={choose}")
        counts[owner] += 1

    return counts


def _render_summary(
    bg: int, frames: list[dict], counts: list[int], *, shifted_colors: bool = True
) -> np.ndarray:
    assert len(frames) == len(counts)
    assert len(set(counts)) == len(counts)
    order = sorted(range(len(frames)), key=lambda i: counts[i])
    width = max(counts)
    out = np.full((len(frames), width), bg, dtype=int)

    for row, idx in enumerate(order):
        count = counts[idx]
        color_idx = order[(row + 1) % len(order)] if shifted_colors else idx
        out[row, :count] = frames[color_idx]["color"]

    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """Summarize each frame's direct marker count, but color each row with the next rank's frame color.

    Input structure:
    - The background is dominant, and several distinct-color hollow rectangular frames appear on it.
    - Some frames are nested inside others, while grey marker pixels lie in frame interiors.
    - A marker may lie inside multiple frames geometrically because of nesting.

    Rule:
    - Assign every marker to the unique innermost frame that contains it strictly inside the frame border.
    - Count markers per frame, sort frames by these direct counts from smallest to largest, and make one output row per rank.
    - Row i has length equal to the i-th smallest count, but its color is borrowed from the next rank cyclically; rows are left-aligned and padded with background to the maximum count.

    Key insight:
    - Two independent phases must both be respected: nested markers belong only to their innermost owner, and output row colors are shifted away from the row-length source frame.
    """
    grid = np.asarray(input_grid, dtype=int)
    bg, frames, markers = _extract_frames_and_markers(grid)
    counts = _ownership_counts(frames, markers, choose="innermost")
    assert min(counts) > 0
    assert len(set(counts)) == len(counts)
    return _render_summary(bg, frames, counts, shifted_colors=True)


def _standalone_region() -> list[tuple[int, int, int, int]]:
    h = int(_RNG.integers(5, 8))
    w = int(_RNG.integers(5, 8))
    return [(0, 0, h - 1, w - 1)]


def _chain_region(depth: int) -> list[tuple[int, int, int, int]]:
    leaf_h = int(_RNG.integers(5, 8))
    leaf_w = int(_RNG.integers(5, 8))
    boxes = [(0, 0, leaf_h - 1, leaf_w - 1)]
    for _ in range(depth - 1):
        top = int(_RNG.integers(2, 4))
        bottom = int(_RNG.integers(2, 4))
        left = int(_RNG.integers(2, 4))
        right = int(_RNG.integers(2, 4))
        r0, c0, r1, c1 = boxes[-1]
        boxes.append((r0 - top, c0 - left, r1 + bottom, c1 + right))
    return _normalize_boxes(boxes)


def _pack_regions(
    regions: list[list[tuple[int, int, int, int]]], gap_low: int, gap_high: int
) -> list[tuple[int, int, int, int]]:
    heights = []
    widths = []
    for region in regions:
        h, w = _boxes_extent(region)
        heights.append(h)
        widths.append(w)

    orientation = "horizontal" if _RNG.random() < 0.5 else "vertical"
    combined: list[tuple[int, int, int, int]] = []

    if orientation == "horizontal":
        patch_h = max(heights) + int(_RNG.integers(0, 3))
        x = 0
        for i, region in enumerate(regions):
            top = int(_RNG.integers(0, patch_h - heights[i] + 1))
            combined.extend(_shift_boxes(region, top, x))
            x += widths[i]
            if i + 1 < len(regions):
                x += int(_RNG.integers(gap_low, gap_high + 1))
    else:
        patch_w = max(widths) + int(_RNG.integers(0, 3))
        y = 0
        for i, region in enumerate(regions):
            left = int(_RNG.integers(0, patch_w - widths[i] + 1))
            combined.extend(_shift_boxes(region, y, left))
            y += heights[i]
            if i + 1 < len(regions):
                y += int(_RNG.integers(gap_low, gap_high + 1))

    return _normalize_boxes(combined)


def _outer_with_children_region(total_frames: int) -> list[tuple[int, int, int, int]]:
    assert total_frames in {3, 4}
    child_a = _chain_region(2) if total_frames == 4 else _standalone_region()
    child_b = _standalone_region()
    inner = _pack_regions([child_a, child_b], gap_low=2, gap_high=4)
    inner_h, inner_w = _boxes_extent(inner)
    pad_t = int(_RNG.integers(2, 4))
    pad_b = int(_RNG.integers(2, 4))
    pad_l = int(_RNG.integers(2, 4))
    pad_r = int(_RNG.integers(2, 4))
    shifted_inner = _shift_boxes(inner, pad_t, pad_l)
    outer = (0, 0, inner_h + pad_t + pad_b - 1, inner_w + pad_l + pad_r - 1)
    return [outer, *shifted_inner]


def _scene_boxes() -> list[tuple[int, int, int, int]]:
    scene_type = str(_RNG.choice(["chain", "siblings", "two_regions"]))
    if scene_type == "chain":
        boxes = _chain_region(int(_RNG.integers(3, 5)))
    elif scene_type == "siblings":
        boxes = _outer_with_children_region(int(_RNG.integers(3, 5)))
    else:
        total = int(_RNG.integers(3, 6))
        if total == 3:
            regions = [_chain_region(2), _standalone_region()]
        elif total == 4:
            regions = [_chain_region(2), _chain_region(2)]
        else:
            regions = [_chain_region(3), _chain_region(2)]
        boxes = _pack_regions(regions, gap_low=3, gap_high=5)
    return _normalize_boxes(boxes)


def _render_frames(
    shape: tuple[int, int], boxes: list[tuple[int, int, int, int]], colors: list[int]
) -> np.ndarray:
    grid = np.full(shape, BG_COLOR, dtype=int)
    for box, color in zip(boxes, colors):
        _draw_frame(grid, box, color)
    return grid


def _owner_candidates(
    frame_grid: np.ndarray, boxes: list[tuple[int, int, int, int]]
) -> list[list[tuple[int, int]]]:
    h, w = frame_grid.shape
    candidates = [[] for _ in boxes]
    areas = [_box_area(box) for box in boxes]

    for r in range(h):
        for c in range(w):
            if int(frame_grid[r, c]) != BG_COLOR:
                continue
            owners = [i for i, box in enumerate(boxes) if _contains_point(box, r, c)]
            if not owners:
                continue
            owner = min(owners, key=lambda i: areas[i])
            candidates[owner].append((r, c))

    return candidates


def _assign_counts(candidates: list[list[tuple[int, int]]]) -> list[int] | None:
    capacities = [len(cells) for cells in candidates]
    count = len(capacities)
    if min(capacities) < count:
        return None

    max_value = min(8, max(capacities))
    values = list(range(1, max_value + 1))
    if len(values) < count:
        return None

    sorted_caps = sorted(capacities)
    for _ in range(200):
        picked = sorted(map(int, _RNG.choice(values, size=count, replace=False)))
        if any(picked[i] > sorted_caps[i] for i in range(count)):
            continue
        for _ in range(50):
            perm = picked.copy()
            _RNG.shuffle(perm)
            if all(perm[i] <= capacities[i] for i in range(count)):
                return [int(v) for v in perm]
    return None


def _place_markers(
    candidates: list[list[tuple[int, int]]], counts: list[int]
) -> list[tuple[int, int]] | None:
    blocked: set[tuple[int, int]] = set()
    chosen_by_frame: list[list[tuple[int, int]]] = [[] for _ in counts]
    order = sorted(range(len(counts)), key=lambda i: (len(candidates[i]), counts[i]))

    for idx in order:
        pool = candidates[idx].copy()
        _RNG.shuffle(pool)
        for r, c in pool:
            if (r, c) in blocked:
                continue
            chosen_by_frame[idx].append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    blocked.add((r + dr, c + dc))
            if len(chosen_by_frame[idx]) == counts[idx]:
                break
        if len(chosen_by_frame[idx]) != counts[idx]:
            return None

    return [coord for coords in chosen_by_frame for coord in coords]


def _expected_from_meta(colors: list[int], counts: list[int]) -> np.ndarray:
    frames = [{"bbox": (0, 0, 0, 0), "color": int(color)} for color in colors]
    return _render_summary(BG_COLOR, frames, counts, shifted_colors=True)


def _validate_generated(
    grid: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    colors: list[int],
    counts: list[int],
    expected: np.ndarray,
) -> bool:
    if not any(
        _contains_box(boxes[i], boxes[j])
        for i in range(len(boxes))
        for j in range(len(boxes))
        if i != j
    ):
        return False

    if int(get_bg(grid)) != BG_COLOR:
        return False

    bg, frames, markers = _extract_frames_and_markers(grid)
    if bg != BG_COLOR:
        return False

    found = sorted((tuple(frame["bbox"]), int(frame["color"])) for frame in frames)
    target = sorted((tuple(box), int(color)) for box, color in zip(boxes, colors))
    if found != target:
        return False

    parsed_counts = _ownership_counts(frames, markers, choose="innermost")
    expected_count_map = {(tuple(box), int(color)): int(count) for box, color, count in zip(boxes, colors, counts)}
    for frame, parsed_count in zip(frames, parsed_counts):
        key = (tuple(frame["bbox"]), int(frame["color"]))
        if expected_count_map.get(key) != int(parsed_count):
            return False

    if min(parsed_counts) <= 0 or len(set(parsed_counts)) != len(parsed_counts):
        return False

    outer_counts = _ownership_counts(frames, markers, choose="outermost")
    if outer_counts == parsed_counts:
        return False

    unshifted = _render_summary(bg, frames, parsed_counts, shifted_colors=False)
    if np.array_equal(unshifted, expected):
        return False

    if not np.array_equal(main(grid), expected):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    boxes = _scene_boxes()
    patch_h, patch_w = _boxes_extent(boxes)
    if patch_h > 26 or patch_w > 26:
        return None

    extra_h = int(_RNG.integers(0, min(5, 30 - patch_h + 1)))
    extra_w = int(_RNG.integers(0, min(5, 30 - patch_w + 1)))
    h = patch_h + extra_h
    w = patch_w + extra_w
    boxes = _shift_boxes(boxes, int(_RNG.integers(0, extra_h + 1)), int(_RNG.integers(0, extra_w + 1)))

    colors = list(map(int, _RNG.choice(FRAME_COLOR_POOL, size=len(boxes), replace=False)))
    frame_grid = _render_frames((h, w), boxes, colors)
    candidates = _owner_candidates(frame_grid, boxes)
    counts = _assign_counts(candidates)
    if counts is None:
        return None

    markers = _place_markers(candidates, counts)
    if markers is None:
        return None

    grid = frame_grid.copy()
    for r, c in markers:
        grid[r, c] = MARKER_COLOR

    expected = _expected_from_meta(colors, counts)
    if not _validate_generated(grid, boxes, colors, counts, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
