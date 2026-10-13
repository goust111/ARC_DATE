from __future__ import annotations

import random
from collections import Counter

import numpy as np


BBox = tuple[int, int, int, int]
Coord = tuple[int, int]


BACKGROUND = 0
ALL_COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)


def _component_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _find_monochromatic_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _bbox_height(bbox: BBox) -> int:
    return int(bbox[2] - bbox[0] + 1)


def _bbox_width(bbox: BBox) -> int:
    return int(bbox[3] - bbox[1] + 1)


def _rectangular_frame_bbox(shape: tuple[int, int], coords: np.ndarray) -> BBox | None:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    if int(coords.shape[0]) != 2 * h + 2 * w - 4:
        return None
    mask = _component_mask(shape, coords)[r0 : r1 + 1, c0 : c1 + 1]
    ring = np.zeros((h, w), dtype=bool)
    ring[0, :] = True
    ring[-1, :] = True
    ring[:, 0] = True
    ring[:, -1] = True
    if not np.array_equal(mask, ring):
        return None
    return (r0, c0, r1, c1)


def _contains(outer: BBox, inner: BBox) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and inner[2] < outer[2] and inner[3] < outer[3]


def _nested_gaps(outer: BBox, inner: BBox) -> tuple[int, int, int, int]:
    top_gap = int(inner[0] - outer[0] - 1)
    left_gap = int(inner[1] - outer[1] - 1)
    bottom_gap = int(outer[2] - inner[2] - 1)
    right_gap = int(outer[3] - inner[3] - 1)
    return top_gap, left_gap, bottom_gap, right_gap


def _sort_nested_frames(frames: list[dict[str, object]]) -> list[dict[str, object]]:
    ordered = sorted(
        frames,
        key=lambda frame: _bbox_height(frame["bbox"]) * _bbox_width(frame["bbox"]),
        reverse=True,
    )
    assert 2 <= len(ordered) <= 3, "expected 2-3 concentric rectangular frames"
    for i in range(len(ordered) - 1):
        outer = ordered[i]["bbox"]
        inner = ordered[i + 1]["bbox"]
        assert _contains(outer, inner), "frames must be strictly nested"
        top_gap, left_gap, bottom_gap, right_gap = _nested_gaps(outer, inner)
        assert min(top_gap, left_gap, bottom_gap, right_gap) >= 2, "adjacent frames need at least a 2-cell gap"
        assert top_gap == bottom_gap and left_gap == right_gap, "frames must be concentric"
    return ordered


def _region_specs(frames: list[dict[str, object]]) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for i, frame in enumerate(frames):
        outer = frame["bbox"]
        inner = frames[i + 1]["bbox"] if i + 1 < len(frames) else None
        anchor_r = int(outer[0] + 1)
        anchor_c = int(outer[1] + 1)
        cells: list[Coord] = []
        colored: list[Coord] = []
        for r in range(int(outer[0] + 1), int(outer[2])):
            for c in range(int(outer[1] + 1), int(outer[3])):
                if inner is not None and inner[0] <= r <= inner[2] and inner[1] <= c <= inner[3]:
                    continue
                cells.append((r, c))
                if (r - anchor_r + c - anchor_c) % 2 == 0:
                    colored.append((r, c))
        assert cells, "every ring region must contain at least one cell"
        specs.append(
            {
                "outer_bbox": outer,
                "inner_bbox": inner,
                "anchor": (anchor_r, anchor_c),
                "cells": cells,
                "colored": colored,
                "count": int(len(colored)),
            }
        )
    return specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match outside marker populations to concentric ring checkerboard counts and fill each ring accordingly.

    Input structure:
    - The background is black.
    - Near the center sit 2-3 one-pixel-thick concentric rectangular frames in distinct colors.
    - Outside the outermost frame are isolated single-pixel markers in several other colors; each marker color appears a unique number of times.

    Rule:
    - Detect the nested rectangular frames from outermost to innermost.
    - Treat each gap between consecutive frames, and the innermost frame interior, as one ring region.
    - In every region, start a checkerboard at that region's upper-left cell with the upper-left cell counted as colored, then count the colored checkerboard cells.
    - Match each region's count to the unique outside marker color with the same population.
    - Erase all outside markers and paint the matched color onto the region's colored checkerboard cells, leaving the other region cells as background and preserving the frame colors.

    Key insight:
    - The decisive numbers come from checkerboard counts on rectangular annuli rather than plain rectangles, so the geometry of each concentric gap determines which outside color belongs there.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    assert bg == BACKGROUND, "expected black background"

    frame_mask = np.zeros(g.shape, dtype=bool)
    frames: list[dict[str, object]] = []
    for comp in _find_monochromatic_components(g, bg):
        bbox = _rectangular_frame_bbox(g.shape, comp["coords"])
        if bbox is None:
            assert int(comp["area"]) == 1, "non-frame foreground components must be isolated marker pixels"
            continue
        comp["bbox"] = bbox
        frames.append(comp)
        frame_mask[comp["coords"][:, 0], comp["coords"][:, 1]] = True

    frames = _sort_nested_frames(frames)
    frame_colors = {int(frame["color"]) for frame in frames}
    regions = _region_specs(frames)
    counts = [int(region["count"]) for region in regions]
    assert len(set(counts)) == len(counts), "ring checkerboard counts must be unique"

    outer_bbox = frames[0]["bbox"]
    outside_mask = (g != bg) & ~frame_mask
    for r, c in np.argwhere(outside_mask):
        rr = int(r)
        cc = int(c)
        assert not (
            outer_bbox[0] <= rr <= outer_bbox[2] and outer_bbox[1] <= cc <= outer_bbox[3]
        ), "markers must lie outside the concentric structure"

    for region in regions:
        for r, c in region["cells"]:
            assert int(g[r, c]) == bg, "ring regions must be empty in the input"

    source_by_count: dict[int, int] = {}
    for color in sorted(set(map(int, g.ravel())) - {bg}):
        count = int(np.sum((g == color) & outside_mask))
        if count == 0:
            continue
        assert color not in frame_colors, "marker colors must differ from frame colors"
        assert count not in source_by_count, "outside marker counts must be unique"
        source_by_count[count] = int(color)

    assert len(source_by_count) == len(regions), "need one marker color per ring region"
    assert set(source_by_count) == set(counts), "marker populations must match ring checkerboard counts"

    out = g.copy()
    out[outside_mask] = bg
    for region in regions:
        fill_color = source_by_count[int(region["count"])]
        for r, c in region["colored"]:
            assert int(out[r, c]) == bg, "checkerboard fill cells must start empty"
            out[r, c] = fill_color
    return out


def _draw_frame(grid: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _build_bboxes(top: int, left: int, inner_h: int, inner_w: int, gaps: list[tuple[int, int]]) -> list[BBox]:
    sizes: list[tuple[int, int]] = [(inner_h + 2, inner_w + 2)]
    for gap_r, gap_c in gaps:
        prev_h, prev_w = sizes[-1]
        sizes.append((prev_h + 2 * (gap_r + 1), prev_w + 2 * (gap_c + 1)))

    outer_h, outer_w = sizes[-1]
    bboxes: list[BBox] = [(top, left, top + outer_h - 1, left + outer_w - 1)]
    for gap_r, gap_c in reversed(gaps):
        prev = bboxes[-1]
        bboxes.append(
            (
                int(prev[0] + gap_r + 1),
                int(prev[1] + gap_c + 1),
                int(prev[2] - gap_r - 1),
                int(prev[3] - gap_c - 1),
            )
        )
    return bboxes


def _sample_geometry() -> tuple[int, int, list[tuple[int, int]], list[int]] | None:
    n_frames = random.choice([2, 3])
    if n_frames == 2:
        inner_h = random.randint(2, 6)
        inner_w = random.randint(2, 6)
        gaps = [(random.randint(2, 4), random.randint(2, 4))]
    else:
        inner_h = random.randint(2, 4)
        inner_w = random.randint(2, 4)
        gaps = [(random.randint(2, 3), random.randint(2, 3)) for _ in range(2)]

    bboxes = _build_bboxes(0, 0, inner_h, inner_w, gaps)
    outer_h = _bbox_height(bboxes[0])
    outer_w = _bbox_width(bboxes[0])
    if outer_h > 18 or outer_w > 18:
        return None

    dummy_frames = [{"bbox": bbox} for bbox in bboxes]
    counts = [int(region["count"]) for region in _region_specs(dummy_frames)]
    if len(set(counts)) != len(counts):
        return None
    if min(counts) < 3:
        return None
    if sum(counts) > 42:
        return None
    return inner_h, inner_w, gaps, counts


def _centered_position(full: int, outer: int) -> int:
    slack = full - outer
    center = slack // 2
    wiggle = min(2, center, slack - center)
    if wiggle <= 0:
        return center
    return center + random.randint(-wiggle, wiggle)


def _marker_candidates(shape: tuple[int, int], outer_bbox: BBox) -> dict[tuple[int, int], list[Coord]]:
    h, w = shape
    r0, c0, r1, c1 = outer_bbox
    moat_r0 = max(0, r0 - 1)
    moat_c0 = max(0, c0 - 1)
    moat_r1 = min(h - 1, r1 + 1)
    moat_c1 = min(w - 1, c1 + 1)
    groups: dict[tuple[int, int], list[Coord]] = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}
    for r in range(h):
        for c in range(w):
            if moat_r0 <= r <= moat_r1 and moat_c0 <= c <= moat_c1:
                continue
            groups[(r % 2, c % 2)].append((r, c))
    return groups


def _render_output(shape: tuple[int, int], bboxes: list[BBox], frame_colors: list[int], fill_colors: list[int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for bbox, color in zip(bboxes, frame_colors):
        _draw_frame(out, bbox, color)
    for region, color in zip(_region_specs([{"bbox": bbox} for bbox in bboxes]), fill_colors):
        for r, c in region["colored"]:
            out[r, c] = int(color)
    return out


def _scatter_markers(inp: np.ndarray, outer_bbox: BBox, marker_counts: dict[int, int]) -> bool:
    total = sum(marker_counts.values())
    groups = _marker_candidates(inp.shape, outer_bbox)
    viable = [cells for cells in groups.values() if len(cells) >= total]
    if not viable:
        return False
    cells = random.choice(viable)[:]
    random.shuffle(cells)

    jobs: list[int] = []
    colors = list(marker_counts)
    random.shuffle(colors)
    for color in colors:
        jobs.extend([int(color)] * int(marker_counts[color]))
    random.shuffle(jobs)

    for (r, c), color in zip(cells, jobs):
        inp[r, c] = int(color)
    return True


def _gen_attempt() -> np.ndarray | None:
    sampled = _sample_geometry()
    if sampled is None:
        return None
    inner_h, inner_w, gaps, counts = sampled

    outer_h = inner_h + 2
    outer_w = inner_w + 2
    for gap_r, gap_c in gaps:
        outer_h += 2 * (gap_r + 1)
        outer_w += 2 * (gap_c + 1)

    min_h = max(15, outer_h + 6)
    max_h = min(26, outer_h + 12)
    min_w = max(15, outer_w + 6)
    max_w = min(26, outer_w + 12)
    if min_h > max_h or min_w > max_w:
        return None

    grid_h = random.randint(min_h, max_h)
    grid_w = random.randint(min_w, max_w)
    top = _centered_position(grid_h, outer_h)
    left = _centered_position(grid_w, outer_w)
    bboxes = _build_bboxes(top, left, inner_h, inner_w, gaps)

    frame_count = len(bboxes)
    palette = random.sample(ALL_COLORS, 2 * frame_count)
    frame_colors = palette[:frame_count]
    marker_colors = palette[frame_count:]
    fill_colors = marker_colors[:]
    random.shuffle(fill_colors)

    out = _render_output((grid_h, grid_w), bboxes, frame_colors, fill_colors)
    inp = out.copy()
    for region in _region_specs([{"bbox": bbox} for bbox in bboxes]):
        for r, c in region["colored"]:
            inp[r, c] = BACKGROUND

    marker_counts = {int(color): int(count) for color, count in zip(fill_colors, counts)}
    if not _scatter_markers(inp, bboxes[0], marker_counts):
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_ATTEMPTS} attempts")
