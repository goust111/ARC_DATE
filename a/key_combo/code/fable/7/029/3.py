from __future__ import annotations

from collections import Counter
import itertools

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


BLACK = 0
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400
FRAME_PADDING = 1
MIN_SCENE_SIDE = 10
MAX_SCENE_H = 22
MAX_SCENE_W = 28
INTERIOR_CANDIDATES = (
    (1, 1),
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (2, 2),
    (1, 4),
    (4, 1),
    (2, 3),
    (3, 2),
    (1, 5),
    (5, 1),
    (2, 4),
    (4, 2),
    (3, 3),
    (1, 6),
    (6, 1),
    (2, 5),
    (5, 2),
    (3, 4),
    (4, 3),
    (2, 6),
    (6, 2),
    (3, 5),
    (5, 3),
    (4, 4),
)


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    else:
        raise ValueError("connectivity must be 4 or 8")

    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in steps:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _frame_border_mask(height: int, width: int) -> np.ndarray:
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return border


def _panel_start(grid: np.ndarray) -> int:
    black_free_rows = [r for r in range(grid.shape[0]) if not np.any(grid[r] == BLACK)]
    assert black_free_rows, "expected a black-free lower scene panel"
    start = black_free_rows[0]
    assert start > 0, "expected an instruction strip above the panel"
    assert all(np.any(grid[r] == BLACK) for r in range(start)), "top strip rows must retain black background"
    assert all(not np.any(grid[r] == BLACK) for r in range(start, grid.shape[0])), "panel may not contain black"
    return start


def _parse_bars(top: np.ndarray) -> list[dict]:
    bars = []
    for coords in components_mask(top != BLACK, connectivity=4):
        bbox = bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        patch = crop_bbox(top, bbox)
        assert r0 == r1, "each indicator must be a one-cell-thick horizontal bar"
        color = int(patch[0, 0])
        assert color != BLACK
        assert np.all(patch == color), "each indicator bar must be monochrome"
        assert coords.shape[0] == patch.shape[1]
        bars.append({"bbox": bbox, "color": color, "length": int(c1 - c0 + 1)})

    assert bars, "expected at least one bar in the instruction strip"
    lengths = [bar["length"] for bar in bars]
    colors = [bar["color"] for bar in bars]
    assert len(lengths) == len(set(lengths)), "bar lengths must be unique"
    assert len(colors) == len(set(colors)), "bar colors must be unique"
    return bars


def _parse_frames(scene: np.ndarray, bg: int) -> list[dict]:
    frames = []
    for color in sorted(set(map(int, scene.ravel())) - {bg}):
        comps = components_mask(scene == color, connectivity=4)
        assert len(comps) == 1, "each frame color should form one component"
        coords = comps[0]
        bbox = bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height >= 3 and width >= 3, "frames must enclose a nonempty interior"

        patch = crop_bbox(scene, bbox)
        border = _frame_border_mask(height, width)
        assert np.all(patch[border] == color), "frame border must be solid"
        assert np.all(patch[~border] == bg), "frame interior must be empty in the input"
        assert coords.shape[0] == int(border.sum()), "foreground pixels must be exactly the border"

        inner_h = height - 2
        inner_w = width - 2
        frames.append(
            {
                "bbox": bbox,
                "color": color,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "interior_area": inner_h * inner_w,
                "perimeter": 2 * (inner_h + inner_w),
                "border_cells": 2 * (inner_h + inner_w) + 4,
            }
        )

    assert len(frames) >= 2, "expected multiple hollow frames"
    areas = [frame["interior_area"] for frame in frames]
    colors = [frame["color"] for frame in frames]
    assert len(areas) == len(set(areas)), "interior areas must be unique"
    assert len(colors) == len(set(colors)), "frame colors must be unique"
    return frames


def _render_output(scene: np.ndarray, bars: list[dict], frames: list[dict], bg: int) -> np.ndarray:
    out = scene.copy()
    by_length = {bar["length"]: bar for bar in bars}
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        if frame["interior_area"] in by_length:
            out[r0 + 1 : r1, c0 + 1 : c1] = by_length[frame["interior_area"]]["color"]
        else:
            out[r0 : r1 + 1, c0 : c1 + 1] = bg
    return out


def _parse_input(grid: np.ndarray) -> tuple[np.ndarray, list[dict], list[dict], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    start = _panel_start(g)
    top = g[:start]
    scene = g[start:]
    bg = get_bg(scene)
    assert bg != BLACK, "scene background must differ from black strip background"
    bars = _parse_bars(top)
    frames = _parse_frames(scene, bg)
    scene_colors = set(map(int, scene.ravel()))
    bar_colors = {bar["color"] for bar in bars}
    assert not (scene_colors & bar_colors), "bar colors must not appear in the lower scene"
    return scene, bars, frames, bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each top-bar length to the lower hollow frame with the same enclosed area, then keep only the scene panel.

    Input structure:
    - A black instruction strip occupies the top rows and contains several separated horizontal bars.
    - Below it sits a full-width rectangular scene panel with a single non-black background color.
    - The panel contains several one-cell-thick monochrome rectangular frames whose interiors are still background.

    Rule:
    - Measure each bar by its horizontal length.
    - For every lower frame, compute the area of its enclosed interior rectangle.
    - If a frame's interior area equals some bar length, fill that interior with the bar's color while preserving the frame wall.
    - If no bar length matches a frame's interior area, erase that whole frame to the scene background.
    - Remove the entire top instruction strip and return only the transformed lower panel.

    Key insight:
    The bars do not identify targets by color or by visible side lengths; their only meaningful attribute is the 1D bar length, which must be matched to each frame's 2D interior area.
    """
    scene, bars, frames, bg = _parse_input(grid)
    return _render_output(scene, bars, frames, bg)


def _choose_frame_specs(rng: np.random.Generator) -> tuple[list[dict], set[int]]:
    for _ in range(200):
        frame_count = int(rng.integers(3, 5))
        picks = rng.choice(len(INTERIOR_CANDIDATES), size=frame_count, replace=False)

        frames = []
        areas = set()
        for idx in picks:
            inner_h, inner_w = INTERIOR_CANDIDATES[int(idx)]
            area = inner_h * inner_w
            if area in areas:
                break
            areas.add(area)
            frames.append(
                {
                    "inner_h": inner_h,
                    "inner_w": inner_w,
                    "outer_h": inner_h + 2,
                    "outer_w": inner_w + 2,
                    "area": area,
                    "perimeter": 2 * (inner_h + inner_w),
                    "border_cells": 2 * (inner_h + inner_w) + 4,
                }
            )
        else:
            widths = [frame["inner_w"] for frame in frames]
            heights = [frame["inner_h"] for frame in frames]
            perimeters = [frame["perimeter"] for frame in frames]
            borders = [frame["border_cells"] for frame in frames]
            has_shape_variety = any(frame["inner_h"] == 1 or frame["inner_w"] == 1 for frame in frames) and any(
                frame["inner_h"] > 1 and frame["inner_w"] > 1 for frame in frames
            )
            if not has_shape_variety:
                continue

            valid_subsets: dict[int, list[set[int]]] = {}
            for matched_count in range(1, frame_count):
                for subset in itertools.combinations(range(frame_count), matched_count):
                    matched_indices = set(map(int, subset))
                    matched_areas = {frames[i]["area"] for i in matched_indices}
                    metric_failures = sum(
                        any(area not in values for area in matched_areas)
                        for values in (widths, heights, perimeters, borders)
                    )
                    has_decoy = any(
                        frame["area"] != area
                        and area in {frame["inner_h"], frame["inner_w"], frame["perimeter"], frame["border_cells"]}
                        for area in matched_areas
                        for frame in frames
                    )
                    if metric_failures >= 3 and has_decoy:
                        valid_subsets.setdefault(matched_count, []).append(matched_indices)

            if not valid_subsets:
                continue

            available_counts = sorted(valid_subsets)
            if 1 in valid_subsets and any(count >= 2 for count in available_counts) and rng.random() < 0.45:
                chosen_count = 1
            else:
                multi_counts = [count for count in available_counts if count >= 2]
                chosen_count = int(rng.choice(multi_counts if multi_counts else available_counts))
            choices = valid_subsets[chosen_count]
            return frames, choices[int(rng.integers(len(choices)))]
    raise RuntimeError("failed to sample frame specifications")


def _place_frames(
    rng: np.random.Generator, scene_h: int, scene_w: int, frames: list[dict]
) -> list[tuple[int, int]] | None:
    order = sorted(range(len(frames)), key=lambda idx: frames[idx]["outer_h"] * frames[idx]["outer_w"], reverse=True)
    occupied = np.zeros((scene_h, scene_w), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(frames)

    for idx in order:
        frame = frames[idx]
        fh = frame["outer_h"]
        fw = frame["outer_w"]
        candidates = [
            (r, c)
            for r in range(1, scene_h - fh)
            for c in range(1, scene_w - fw)
            if not occupied[
                r - FRAME_PADDING : r + fh + FRAME_PADDING,
                c - FRAME_PADDING : c + fw + FRAME_PADDING,
            ].any()
        ]
        if not candidates:
            return None
        r0, c0 = candidates[int(rng.integers(len(candidates)))]
        positions[idx] = (r0, c0)
        occupied[r0 : r0 + fh, c0 : c0 + fw] = True

    assert all(pos is not None for pos in positions)
    return [(int(r), int(c)) for r, c in positions]  # type: ignore[misc]


def _render_scene(
    scene_h: int,
    scene_w: int,
    bg: int,
    frames: list[dict],
    positions: list[tuple[int, int]],
) -> np.ndarray:
    scene = np.full((scene_h, scene_w), bg, dtype=int)
    for frame, (r0, c0) in zip(frames, positions):
        color = frame["color"]
        fh = frame["outer_h"]
        fw = frame["outer_w"]
        scene[r0 : r0 + fh, c0 : c0 + fw] = color
        scene[r0 + 1 : r0 + fh - 1, c0 + 1 : c0 + fw - 1] = bg
        frame["bbox"] = (r0, c0, r0 + fh - 1, c0 + fw - 1)
    return scene


def _render_top(
    width: int,
    bars: list[dict],
) -> np.ndarray:
    top_h = max(4, 2 * len(bars) + 1)
    top = np.zeros((top_h, width), dtype=int)
    for bar in bars:
        row = bar["row"]
        length = bar["length"]
        start = bar["start"]
        top[row, start : start + length] = bar["color"]
    return top


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    frames, matched_indices = _choose_frame_specs(rng)
    frame_count = len(frames)
    bar_count = len(matched_indices)

    colors = list(map(int, rng.choice(PALETTE, size=frame_count + bar_count + 1, replace=False)))
    bg = colors[0]
    frame_colors = colors[1 : 1 + frame_count]
    bar_colors = colors[1 + frame_count :]
    rng.shuffle(frame_colors)
    rng.shuffle(bar_colors)

    matched_areas = sorted(frames[idx]["area"] for idx in matched_indices)
    max_bar_length = max(matched_areas)
    max_frame_h = max(frame["outer_h"] for frame in frames)
    max_frame_w = max(frame["outer_w"] for frame in frames)
    top_h = max(4, 2 * bar_count + 1)

    min_scene_h = max(MIN_SCENE_SIDE, max_frame_h + 2)
    min_scene_w = max(MIN_SCENE_SIDE, max(max_frame_w + 2, max_bar_length + 2))
    max_scene_h = min(MAX_SCENE_H, 30 - top_h)
    max_scene_w = MAX_SCENE_W
    if min_scene_h > max_scene_h or min_scene_w > max_scene_w:
        return None

    for i, frame in enumerate(frames):
        frame["color"] = frame_colors[i]

    for _ in range(120):
        scene_h = int(rng.integers(min_scene_h, max_scene_h + 1))
        scene_w = int(rng.integers(min_scene_w, max_scene_w + 1))
        positions = _place_frames(rng, scene_h, scene_w, frames)
        if positions is None:
            continue

        bars = []
        row_slots = list(range(1, top_h, 2))
        rng.shuffle(row_slots)
        for color, frame_idx, row in zip(bar_colors, matched_indices, row_slots):
            length = frames[frame_idx]["area"]
            start_low = 1
            start_high = scene_w - length - 1
            if start_high < start_low:
                break
            bars.append(
                {
                    "color": int(color),
                    "length": int(length),
                    "row": int(row),
                    "start": int(rng.integers(start_low, start_high + 1)),
                }
            )
        else:
            scene = _render_scene(scene_h, scene_w, bg, frames, positions)
            top = _render_top(scene_w, bars)
            grid = np.vstack([top, scene])
            if max(grid.shape) <= 30:
                return grid
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid unnamed-ruler input")


def _validate_case(inp: np.ndarray, out: np.ndarray) -> dict:
    scene, bars, frames, bg = _parse_input(inp)
    expected = _render_output(scene, bars, frames, bg)
    assert np.array_equal(out, expected), "main output does not match parsed transformation"

    matched_areas = {bar["length"] for bar in bars}
    frame_areas = {frame["interior_area"] for frame in frames}
    unmatched = [frame for frame in frames if frame["interior_area"] not in matched_areas]
    matched = [frame for frame in frames if frame["interior_area"] in matched_areas]
    assert matched and unmatched, "generator must produce both filled and erased frames"
    assert matched_areas < frame_areas, "all bars should match frames, but not all frames should be matched"

    widths = {frame["inner_w"] for frame in frames}
    heights = {frame["inner_h"] for frame in frames}
    perimeters = {frame["perimeter"] for frame in frames}
    borders = {frame["border_cells"] for frame in frames}
    width_fail = any(area not in widths for area in matched_areas)
    height_fail = any(area not in heights for area in matched_areas)
    perimeter_fail = any(area not in perimeters for area in matched_areas)
    border_fail = any(area not in borders for area in matched_areas)
    assert sum((width_fail, height_fail, perimeter_fail, border_fail)) >= 3
    has_decoy = any(
        frame["interior_area"] != area
        and area in {frame["inner_h"], frame["inner_w"], frame["perimeter"], frame["border_cells"]}
        for area in matched_areas
        for frame in frames
    )
    assert has_decoy

    return {
        "width_fail": width_fail,
        "height_fail": height_fail,
        "perimeter_fail": perimeter_fail,
        "border_fail": border_fail,
        "has_decoy": has_decoy,
        "has_unit_interior": any(frame["inner_h"] == 1 and frame["inner_w"] == 1 for frame in frames),
        "has_slender_interior": any(frame["inner_h"] == 1 or frame["inner_w"] == 1 for frame in frames),
        "has_block_interior": any(frame["inner_h"] > 1 and frame["inner_w"] > 1 for frame in frames),
        "matched_count": len(matched),
        "frame_count": len(frames),
    }
