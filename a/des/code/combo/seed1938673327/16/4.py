from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np


BG_COLOR = 0
FRAME_COLOR = 5
PATH_COLOR = 8
FILL_COLORS = [1, 2, 3, 4, 6, 7, 9]

GRID_MIN = 22
GRID_MAX = 30
FRAME_MIN = 4
FRAME_MAX = 7
GEN_RETRIES = 400

Point = Tuple[int, int]
Frame = Dict[str, int]
PathSpec = Dict[str, object]


def _components(mask: np.ndarray) -> List[List[Point]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: List[List[Point]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: List[Point] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: List[Point]) -> Tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_from_component(grid: np.ndarray, color: int, comp: List[Point], frame_id: int) -> Frame | None:
    r0, c0, r1, c1 = _bbox(comp)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None
    border: set[Point] = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    comp_set = set(comp)
    if comp_set != border:
        return None
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    if inner.size == 0:
        return None
    inner_colors = {int(v) for v in inner.ravel()}
    if len(inner_colors) != 1:
        return None
    fill = next(iter(inner_colors))
    if fill in (BG_COLOR, color):
        return None
    return {
        "id": frame_id,
        "top": r0,
        "left": c0,
        "bottom": r1,
        "right": c1,
        "height": h,
        "width": w,
        "area": h * w,
        "border_color": color,
        "fill_color": fill,
    }


def _detect_frames(grid: np.ndarray) -> List[Frame]:
    frames: List[Frame] = []
    next_id = 0
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BG_COLOR):
        comps = _components(grid == color)
        for comp in comps:
            frame = _frame_from_component(grid, color, comp, next_id)
            if frame is not None:
                frames.append(frame)
                next_id += 1
    frames.sort(key=lambda fr: (fr["top"], fr["left"], fr["area"]))
    for idx, frame in enumerate(frames):
        frame["id"] = idx
    return frames


def _frame_masks(grid: np.ndarray, frames: List[Frame]) -> Tuple[np.ndarray, Dict[Point, int], Dict[int, Frame]]:
    covered = np.zeros(grid.shape, dtype=bool)
    border_owner: Dict[Point, int] = {}
    frame_lookup: Dict[int, Frame] = {}
    for frame in frames:
        frame_lookup[frame["id"]] = frame
        r0, c0, r1, c1 = frame["top"], frame["left"], frame["bottom"], frame["right"]
        covered[r0 : r1 + 1, c0 : c1 + 1] = True
        for c in range(c0, c1 + 1):
            border_owner[(r0, c)] = frame["id"]
            border_owner[(r1, c)] = frame["id"]
        for r in range(r0, r1 + 1):
            border_owner[(r, c0)] = frame["id"]
            border_owner[(r, c1)] = frame["id"]
    return covered, border_owner, frame_lookup


def _path_sort_key(path: PathSpec, frame_lookup: Dict[int, Frame]) -> Tuple[int, int, int, int, int]:
    frame_a = frame_lookup[int(path["frame_a"])]
    frame_b = frame_lookup[int(path["frame_b"])]
    a1, a2 = sorted((frame_a["area"], frame_b["area"]))
    start_r, start_c = path["pixels"][0]
    orient_rank = 0 if path["orientation"] == "h" else 1
    return a1, a2, orient_rank, start_r, start_c


def _recolor_paths(base: np.ndarray, frame_lookup: Dict[int, Frame], paths: List[PathSpec]) -> np.ndarray:
    out = base.copy()
    for path in sorted(paths, key=lambda item: _path_sort_key(item, frame_lookup)):
        pixels: List[Point] = path["pixels"]  # type: ignore[assignment]
        color_a = frame_lookup[int(path["frame_a"])]["fill_color"]
        color_b = frame_lookup[int(path["frame_b"])]["fill_color"]
        split = (len(pixels) + 1) // 2
        for idx, (r, c) in enumerate(pixels):
            out[r, c] = color_a if idx < split else color_b
    return out


def _extract_paths(grid: np.ndarray, frames: List[Frame]) -> List[PathSpec]:
    covered, border_owner, _ = _frame_masks(grid, frames)
    path_mask = (grid != BG_COLOR) & (~covered)
    h, w = grid.shape
    paths: List[PathSpec] = []

    for r in range(h):
        c = 0
        while c < w:
            if not path_mask[r, c]:
                c += 1
                continue
            c0 = c
            while c + 1 < w and path_mask[r, c + 1]:
                c += 1
            c1 = c
            left_owner = border_owner.get((r, c0 - 1)) if c0 > 0 else None
            right_owner = border_owner.get((r, c1 + 1)) if c1 + 1 < w else None
            if left_owner is not None and right_owner is not None and left_owner != right_owner:
                pixels = [(r, cc) for cc in range(c0, c1 + 1)]
                paths.append(
                    {
                        "orientation": "h",
                        "frame_a": left_owner,
                        "frame_b": right_owner,
                        "pixels": pixels,
                    }
                )
            c += 1

    for c in range(w):
        r = 0
        while r < h:
            if not path_mask[r, c]:
                r += 1
                continue
            r0 = r
            while r + 1 < h and path_mask[r + 1, c]:
                r += 1
            r1 = r
            top_owner = border_owner.get((r0 - 1, c)) if r0 > 0 else None
            bottom_owner = border_owner.get((r1 + 1, c)) if r1 + 1 < h else None
            if top_owner is not None and bottom_owner is not None and top_owner != bottom_owner:
                pixels = [(rr, c) for rr in range(r0, r1 + 1)]
                paths.append(
                    {
                        "orientation": "v",
                        "frame_a": top_owner,
                        "frame_b": bottom_owner,
                        "pixels": pixels,
                    }
                )
            r += 1

    return paths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each white connector by the colors of the two framed regions it touches.

    Frames are 1-pixel gray rectangles with a solid colored interior. Horizontal paths
    are read left-to-right; vertical paths are read top-to-bottom. Each path's first
    half uses the first endpoint frame's fill color, and the remainder uses the second
    endpoint frame's fill color. Paths are painted in ascending endpoint-area order, so
    later paths overwrite earlier colors at crossings.
    """
    g = np.asarray(grid, dtype=int)
    frames = _detect_frames(g)
    if not frames:
        return g.copy()
    frame_lookup = {frame["id"]: frame for frame in frames}
    paths = _extract_paths(g, frames)
    if not paths:
        return g.copy()
    return _recolor_paths(g, frame_lookup, paths)


def _boxes_separate(a: Frame, b: Frame, pad: int = 1) -> bool:
    return (
        a["bottom"] + pad < b["top"]
        or b["bottom"] + pad < a["top"]
        or a["right"] + pad < b["left"]
        or b["right"] + pad < a["left"]
    )


def _sample_frame_specs(count: int, rng: random.Random) -> List[Tuple[int, int, int]]:
    colors = rng.sample(FILL_COLORS, count)
    specs: List[Tuple[int, int, int]] = []
    used_areas: set[int] = set()
    for fill in colors:
        for _ in range(200):
            h = rng.randint(FRAME_MIN, FRAME_MAX)
            w = rng.randint(FRAME_MIN, FRAME_MAX)
            area = h * w
            if area in used_areas:
                continue
            used_areas.add(area)
            specs.append((h, w, fill))
            break
        else:
            raise RuntimeError("failed to sample unique frame areas")
    return specs


def _sample_gap_pair(rng: random.Random, parity: int) -> Tuple[int, int]:
    choices = [(a, b) for a in range(1, 8) for b in range(1, 8) if (a + b + 1) % 2 == parity]
    return rng.choice(choices)


def _make_frame(frame_id: int, top: int, left: int, h: int, w: int, fill: int) -> Frame:
    return {
        "id": frame_id,
        "top": top,
        "left": left,
        "bottom": top + h - 1,
        "right": left + w - 1,
        "height": h,
        "width": w,
        "area": h * w,
        "border_color": FRAME_COLOR,
        "fill_color": fill,
    }


def _build_horizontal_pair(
    frame_id: int,
    row: int,
    center_col: int,
    left_spec: Tuple[int, int, int],
    right_spec: Tuple[int, int, int],
    h: int,
    w: int,
    rng: random.Random,
) -> Tuple[List[Frame], PathSpec] | None:
    lh, lw, lfill = left_spec
    rh, rw, rfill = right_spec
    parity = rng.randint(0, 1)
    for _ in range(80):
        gap_left, gap_right = _sample_gap_pair(rng, parity)
        left_offset = rng.randint(1, lh - 2)
        right_offset = rng.randint(1, rh - 2)
        top_left = row - left_offset
        top_right = row - right_offset
        left = center_col - gap_left - lw
        right = center_col + gap_right + 1
        frame_a = _make_frame(frame_id, top_left, left, lh, lw, lfill)
        frame_b = _make_frame(frame_id + 1, top_right, right, rh, rw, rfill)
        if not (1 <= frame_a["top"] and 1 <= frame_b["top"]):
            continue
        if not (frame_a["bottom"] <= h - 2 and frame_b["bottom"] <= h - 2):
            continue
        if not (1 <= frame_a["left"] and 1 <= frame_b["left"]):
            continue
        if not (frame_a["right"] <= w - 2 and frame_b["right"] <= w - 2):
            continue
        if not _boxes_separate(frame_a, frame_b, pad=1):
            continue
        pixels = [(row, c) for c in range(frame_a["right"] + 1, frame_b["left"])]
        if len(pixels) < 3:
            continue
        return [frame_a, frame_b], {"orientation": "h", "frame_a": frame_id, "frame_b": frame_id + 1, "pixels": pixels}
    return None


def _build_vertical_pair(
    frame_id: int,
    col: int,
    center_row: int,
    top_spec: Tuple[int, int, int],
    bottom_spec: Tuple[int, int, int],
    h: int,
    w: int,
    rng: random.Random,
) -> Tuple[List[Frame], PathSpec] | None:
    th, tw, tfill = top_spec
    bh, bw, bfill = bottom_spec
    parity = rng.randint(0, 1)
    for _ in range(80):
        gap_top, gap_bottom = _sample_gap_pair(rng, parity)
        top_offset = rng.randint(1, tw - 2)
        bottom_offset = rng.randint(1, bw - 2)
        left_top = col - top_offset
        left_bottom = col - bottom_offset
        top = center_row - gap_top - th
        bottom = center_row + gap_bottom + 1
        frame_a = _make_frame(frame_id, top, left_top, th, tw, tfill)
        frame_b = _make_frame(frame_id + 1, bottom, left_bottom, bh, bw, bfill)
        if not (1 <= frame_a["top"] and 1 <= frame_b["top"]):
            continue
        if not (frame_a["bottom"] <= h - 2 and frame_b["bottom"] <= h - 2):
            continue
        if not (1 <= frame_a["left"] and 1 <= frame_b["left"]):
            continue
        if not (frame_a["right"] <= w - 2 and frame_b["right"] <= w - 2):
            continue
        if not _boxes_separate(frame_a, frame_b, pad=1):
            continue
        pixels = [(r, col) for r in range(frame_a["bottom"] + 1, frame_b["top"])]
        if len(pixels) < 3:
            continue
        return [frame_a, frame_b], {"orientation": "v", "frame_a": frame_id, "frame_b": frame_id + 1, "pixels": pixels}
    return None


def _all_frames_separate(frames: List[Frame]) -> bool:
    for i, frame_a in enumerate(frames):
        for frame_b in frames[i + 1 :]:
            if not _boxes_separate(frame_a, frame_b, pad=1):
                return False
    return True


def _paint_frame(grid: np.ndarray, frame: Frame) -> None:
    r0, c0, r1, c1 = frame["top"], frame["left"], frame["bottom"], frame["right"]
    fill = frame["fill_color"]
    grid[r0 : r1 + 1, c0 : c1 + 1] = fill
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR


def _paint_input(h: int, w: int, frames: List[Frame], paths: List[PathSpec]) -> np.ndarray | None:
    grid = np.full((h, w), BG_COLOR, dtype=int)
    for frame in frames:
        _paint_frame(grid, frame)
    for path in paths:
        for r, c in path["pixels"]:
            if grid[r, c] not in (BG_COLOR, PATH_COLOR):
                return None
            grid[r, c] = PATH_COLOR
    return grid


def _expected_output(inp: np.ndarray, frames: List[Frame], paths: List[PathSpec]) -> np.ndarray:
    frame_lookup = {frame["id"]: frame for frame in frames}
    return _recolor_paths(inp, frame_lookup, paths)


def _make_cross_layout(h: int, w: int, rng: random.Random) -> Tuple[List[Frame], List[PathSpec]] | None:
    specs = _sample_frame_specs(4, rng)
    for _ in range(120):
        center_row = rng.randint(7, h - 8)
        center_col = rng.randint(7, w - 8)
        horiz = _build_horizontal_pair(0, center_row, center_col, specs[0], specs[1], h, w, rng)
        vert = _build_vertical_pair(2, center_col, center_row, specs[2], specs[3], h, w, rng)
        if horiz is None or vert is None:
            continue
        frames = horiz[0] + vert[0]
        if not _all_frames_separate(frames):
            continue
        return frames, [horiz[1], vert[1]]
    return None


def _lane_rows(h: int, count: int, rng: random.Random) -> List[int]:
    if count == 2:
        return [rng.randint(4, h // 2 - 4), rng.randint(h // 2 + 3, h - 5)]
    return [
        rng.randint(4, h // 3 + 1),
        rng.randint(h // 3 + 4, 2 * h // 3 - 3),
        rng.randint(2 * h // 3 + 2, h - 5),
    ]


def _lane_cols(w: int, count: int, rng: random.Random) -> List[int]:
    if count == 2:
        return [rng.randint(4, w // 2 - 4), rng.randint(w // 2 + 3, w - 5)]
    return [
        rng.randint(4, w // 3 + 1),
        rng.randint(w // 3 + 4, 2 * w // 3 - 3),
        rng.randint(2 * w // 3 + 2, w - 5),
    ]


def _make_multi_horizontal_layout(h: int, w: int, rng: random.Random) -> Tuple[List[Frame], List[PathSpec]] | None:
    path_count = 3 if rng.random() < 0.45 else 2
    specs = _sample_frame_specs(path_count * 2, rng)
    rows = _lane_rows(h, path_count, rng)
    frames: List[Frame] = []
    paths: List[PathSpec] = []
    next_id = 0
    for idx, row in enumerate(rows):
        center_col = rng.randint(7, w - 8)
        pair = _build_horizontal_pair(next_id, row, center_col, specs[2 * idx], specs[2 * idx + 1], h, w, rng)
        if pair is None:
            return None
        frames.extend(pair[0])
        paths.append(pair[1])
        next_id += 2
    if not _all_frames_separate(frames):
        return None
    return frames, paths


def _make_multi_vertical_layout(h: int, w: int, rng: random.Random) -> Tuple[List[Frame], List[PathSpec]] | None:
    path_count = 3 if rng.random() < 0.45 else 2
    specs = _sample_frame_specs(path_count * 2, rng)
    cols = _lane_cols(w, path_count, rng)
    frames: List[Frame] = []
    paths: List[PathSpec] = []
    next_id = 0
    for idx, col in enumerate(cols):
        center_row = rng.randint(7, h - 8)
        pair = _build_vertical_pair(next_id, col, center_row, specs[2 * idx], specs[2 * idx + 1], h, w, rng)
        if pair is None:
            return None
        frames.extend(pair[0])
        paths.append(pair[1])
        next_id += 2
    if not _all_frames_separate(frames):
        return None
    return frames, paths


def _generator_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    layout = rng.choice(["cross", "cross", "multi_h", "multi_v"])

    if layout == "cross":
        built = _make_cross_layout(h, w, rng)
    elif layout == "multi_h":
        built = _make_multi_horizontal_layout(h, w, rng)
    else:
        built = _make_multi_vertical_layout(h, w, rng)
    if built is None:
        return None
    frames, paths = built

    grid = _paint_input(h, w, frames, paths)
    if grid is None:
        return None

    detected_frames = _detect_frames(grid)
    if len(detected_frames) != len(frames):
        return None
    detected_paths = _extract_paths(grid, detected_frames)
    if len(detected_paths) != len(paths):
        return None
    if sorted(len(path["pixels"]) for path in detected_paths) != sorted(len(path["pixels"]) for path in paths):
        return None

    expected = _expected_output(grid, frames, paths)
    actual = main(grid)
    if np.array_equal(actual, grid):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _generator_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
