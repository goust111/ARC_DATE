from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

GRID_SIZE_RANGE = (16, 30)
LAYER_COUNT_RANGE = (2, 5)
CORE_HEIGHT_RANGE = (3, 7)
CORE_WIDTH_RANGE = (3, 7)
CORE_COLOR_COUNT_RANGE = (2, 4)
VERTICAL_GAP_RANGE = (0, 1)
HORIZONTAL_GAP_RANGE = (0, 2)
MAX_GENERATION_ATTEMPTS = 400


def _border_bg(g: np.ndarray) -> int:
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, counts = np.unique(border, return_counts=True)
    return int(vals[counts.argmax()])


def _components(g: np.ndarray, bg: int) -> list[dict]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue

            color = int(g[r, c])
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            mins = arr.min(axis=0)
            maxs = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])),
                }
            )
    return comps


def _is_rect_frame(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False

    sub = np.zeros((h, w), dtype=bool)
    coords = comp["coords"]
    sub[coords[:, 0] - r0, coords[:, 1] - c0] = True

    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return np.array_equal(sub, expected)


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and inner[2] < outer[2] and inner[3] < outer[3]


def _find_frames(g: np.ndarray, bg: int) -> list[dict]:
    frames = [comp for comp in _components(g, bg) if _is_rect_frame(comp)]
    assert frames, "expected at least one nested rectangular frame"

    frames.sort(
        key=lambda comp: (
            -((comp["bbox"][2] - comp["bbox"][0] + 1) * (comp["bbox"][3] - comp["bbox"][1] + 1)),
            comp["bbox"][0],
            comp["bbox"][1],
        )
    )

    frame_colors = [int(comp["color"]) for comp in frames]
    assert len(frame_colors) == len(set(frame_colors)), "frame colors must be distinct"
    for outer, inner in zip(frames, frames[1:]):
        assert _contains(outer["bbox"], inner["bbox"]), "frames must form one strict nesting chain"
    return frames


def _apply_nested_flips(g: np.ndarray, frames: list[dict]) -> np.ndarray:
    out = g.copy()
    for idx in range(len(frames) - 1, -1, -1):
        layer = idx + 1
        if layer % 2 == 1:
            continue
        r0, c0, r1, c1 = frames[idx]["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        out[r0 + 1 : r1, c0 + 1 : c1] = interior[:, ::-1].copy()
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    In a stack of nested rectangular frames, even-numbered layers mirror their interior horizontally.

    Input structure:
    - The background is black.
    - Distinct-color one-pixel rectangular frames are strictly nested.
    - The innermost frame contains an asymmetric multi-color core pattern.

    Rule:
    - Number frames from the outside inward starting at 1.
    - Process from the innermost frame outward.
    - For every even-numbered frame, mirror everything inside that frame left-right.
    - The frame border itself does not move.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _border_bg(g)
    frames = _find_frames(g, bg)
    return _apply_nested_flips(g, frames)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _count_frame_candidates(g: np.ndarray, bg: int) -> int:
    return sum(1 for comp in _components(g, bg) if _is_rect_frame(comp))


def _random_core_pattern(height: int, width: int, colors: list[int]) -> np.ndarray:
    for _ in range(200):
        core = np.empty((height, width), dtype=int)
        mode = random.randrange(4)

        if mode == 0:
            row_offsets = [random.randrange(len(colors)) for _ in range(height)]
            for r in range(height):
                for c in range(width):
                    core[r, c] = colors[(row_offsets[r] + c) % len(colors)]
        elif mode == 1:
            col_offsets = [random.randrange(len(colors)) for _ in range(width)]
            for r in range(height):
                for c in range(width):
                    core[r, c] = colors[(r + col_offsets[c]) % len(colors)]
        elif mode == 2:
            a = random.randint(1, len(colors))
            b = random.randint(1, len(colors))
            bias = random.randrange(len(colors))
            for r in range(height):
                for c in range(width):
                    core[r, c] = colors[(a * r + b * c + bias) % len(colors)]
        else:
            for r in range(height):
                for c in range(width):
                    core[r, c] = random.choice(colors)

        edits = random.randint(1, max(2, (height * width) // 3))
        for _ in range(edits):
            rr = random.randrange(height)
            cc = random.randrange(width)
            core[rr, cc] = random.choice(colors)

        if len(set(map(int, core.flat))) < 2:
            continue
        if np.array_equal(core, core[:, ::-1]):
            continue
        if _count_frame_candidates(core, bg=-1) != 0:
            continue
        return core

    raise RuntimeError("failed to generate asymmetric core pattern")


def _build_local_boxes(layer_count: int, core_h: int, core_w: int) -> tuple[list[tuple[int, int, int, int]], tuple[int, int, int, int]]:
    core_box = (0, 0, core_h - 1, core_w - 1)
    content_box = core_box
    inner_to_outer: list[tuple[int, int, int, int]] = []

    for _ in range(layer_count):
        top_gap = random.randint(*VERTICAL_GAP_RANGE)
        bottom_gap = random.randint(*VERTICAL_GAP_RANGE)
        left_gap = random.randint(*HORIZONTAL_GAP_RANGE)
        right_gap = random.randint(*HORIZONTAL_GAP_RANGE)
        r0, c0, r1, c1 = content_box
        frame_box = (
            int(r0 - top_gap - 1),
            int(c0 - left_gap - 1),
            int(r1 + bottom_gap + 1),
            int(c1 + right_gap + 1),
        )
        inner_to_outer.append(frame_box)
        content_box = frame_box

    outer = inner_to_outer[-1]
    row_shift = -outer[0]
    col_shift = -outer[1]

    def shift(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        return (box[0] + row_shift, box[1] + col_shift, box[2] + row_shift, box[3] + col_shift)

    frames = [shift(box) for box in reversed(inner_to_outer)]
    shifted_core = shift(core_box)
    return frames, shifted_core


def _shift_boxes(boxes: list[tuple[int, int, int, int]], dr: int, dc: int) -> list[tuple[int, int, int, int]]:
    return [(r0 + dr, c0 + dc, r1 + dr, c1 + dc) for r0, c0, r1, c1 in boxes]


def _generate_attempt() -> np.ndarray | None:
    layer_count = random.randint(*LAYER_COUNT_RANGE)
    core_h = random.randint(*CORE_HEIGHT_RANGE)
    core_w = random.randint(*CORE_WIDTH_RANGE)

    max_core_colors = min(CORE_COLOR_COUNT_RANGE[1], len(PALETTE) - layer_count)
    if max_core_colors < CORE_COLOR_COUNT_RANGE[0]:
        return None
    core_color_count = random.randint(CORE_COLOR_COUNT_RANGE[0], max_core_colors)
    colors = random.sample(PALETTE, layer_count + core_color_count)
    frame_colors = colors[:layer_count]
    core_colors = colors[layer_count:]

    core = _random_core_pattern(core_h, core_w, core_colors)
    frames, core_box = _build_local_boxes(layer_count, core_h, core_w)

    outer_h = frames[0][2] - frames[0][0] + 1
    outer_w = frames[0][3] - frames[0][1] + 1
    min_size, max_size = GRID_SIZE_RANGE
    grid_h_lo = max(min_size, outer_h + 2)
    grid_w_lo = max(min_size, outer_w + 2)
    if grid_h_lo > max_size or grid_w_lo > max_size:
        return None

    grid_h = random.randint(grid_h_lo, max_size)
    grid_w = random.randint(grid_w_lo, max_size)
    top = random.randint(1, grid_h - outer_h - 1)
    left = random.randint(1, grid_w - outer_w - 1)

    frames = _shift_boxes(frames, top, left)
    core_box = (core_box[0] + top, core_box[1] + left, core_box[2] + top, core_box[3] + left)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for bbox, color in zip(frames, frame_colors, strict=False):
        _draw_frame(grid, bbox, color)

    r0, c0, r1, c1 = core_box
    grid[r0 : r1 + 1, c0 : c1 + 1] = core

    try:
        detected = _find_frames(grid, BG)
        if len(detected) != layer_count:
            return None
        if [tuple(comp["bbox"]) for comp in detected] != frames:
            return None
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame sample")
