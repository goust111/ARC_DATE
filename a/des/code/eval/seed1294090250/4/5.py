from __future__ import annotations

import random
from typing import Iterable

import numpy as np


GREY = 5
GAP = 1
ALL_COLORS = tuple(range(10))
CORNERS = ("TL", "TR", "BL", "BR")
GEN_RETRIES = 800
MAX_GRID_SIZE = 30


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(pts)
    return out


def _bbox(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(points)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _sort_reading_order(items: list[dict]) -> list[dict]:
    return sorted(items, key=lambda item: (item["bbox"][0], item["bbox"][1]))


def _detect_frame(grid: np.ndarray, pts: list[tuple[int, int]], bg: int) -> dict | None:
    r0, c0, r1, c1 = _bbox(pts)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if np.any(sub[1:-1, 1:-1] != bg):
        return None

    perimeter_count = 2 * h + 2 * w - 4
    counts: dict[int, int] = {}
    marker_corner: str | None = None
    corner_map = {
        "TL": (0, 0),
        "TR": (0, w - 1),
        "BL": (h - 1, 0),
        "BR": (h - 1, w - 1),
    }

    for rr in range(h):
        for cc in range(w):
            on_perimeter = rr in {0, h - 1} or cc in {0, w - 1}
            val = int(sub[rr, cc])
            if on_perimeter:
                if val == bg:
                    return None
                counts[val] = counts.get(val, 0) + 1
            elif val != bg:
                return None

    if len(counts) != 2:
        return None

    border_candidates = [color for color, cnt in counts.items() if cnt == perimeter_count - 1]
    marker_candidates = [color for color, cnt in counts.items() if cnt == 1]
    if len(border_candidates) != 1 or len(marker_candidates) != 1:
        return None

    border_color = int(border_candidates[0])
    marker_color = int(marker_candidates[0])

    for corner, (rr, cc) in corner_map.items():
        val = int(sub[rr, cc])
        if val == marker_color:
            if marker_corner is not None:
                return None
            marker_corner = corner
        elif val != border_color:
            return None

    if marker_corner is None:
        return None

    for rr in range(h):
        for cc in range(w):
            if rr in {0, h - 1} or cc in {0, w - 1}:
                if (rr, cc) not in corner_map.values() and int(sub[rr, cc]) != border_color:
                    return None

    return {
        "bbox": (r0, c0, r1, c1),
        "border_color": border_color,
        "marker_color": marker_color,
        "anchor": marker_corner,
        "inner_h": h - 2,
        "inner_w": w - 2,
        "cells": pts,
    }


def _extract_fragment(grid: np.ndarray, pts: list[tuple[int, int]]) -> dict:
    colors = {int(grid[r, c]) for r, c in pts}
    assert len(colors) == 1, "fragments must be monochromatic"
    (color,) = tuple(colors)
    r0, c0, r1, c1 = _bbox(pts)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        mask[r - r0, c - c0] = True
    return {
        "bbox": (r0, c0, r1, c1),
        "color": int(color),
        "mask": mask,
        "cells": pts,
    }


def _pack_top_left(fragments: list[dict], inner_h: int, inner_w: int) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    row_top = 0
    row_height = 0
    used_width = 0
    row_empty = True

    for frag in fragments:
        h, w = map(int, frag["mask"].shape)
        assert h <= inner_h and w <= inner_w, "fragment bbox must fit inside the frame"

        if row_empty:
            col = 0
            needed_height = h
        elif used_width + GAP + w <= inner_w:
            col = used_width + GAP
            needed_height = max(row_height, h)
        else:
            row_top += row_height + GAP
            row_height = 0
            used_width = 0
            row_empty = True
            col = 0
            needed_height = h

        assert row_top + needed_height <= inner_h, "packed fragments overflow the frame height"

        positions.append((row_top, col))
        row_height = max(row_height, h)
        used_width = col + w
        row_empty = False

    return positions


def _anchor_position(
    norm_r: int,
    norm_c: int,
    frag_h: int,
    frag_w: int,
    inner_h: int,
    inner_w: int,
    anchor: str,
) -> tuple[int, int]:
    top = norm_r if anchor in {"TL", "TR"} else inner_h - (norm_r + frag_h)
    left = norm_c if anchor in {"TL", "BL"} else inner_w - (norm_c + frag_w)
    return int(top), int(left)


def _render_solution(
    grid: np.ndarray,
    bg: int,
    frames: list[dict],
    fragments: list[dict],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    grouped: dict[int, list[dict]] = {}
    for frag in fragments:
        for r, c in frag["cells"]:
            out[r, c] = bg
        grouped.setdefault(int(frag["color"]), []).append(frag)

    frame_colors = [int(frame["marker_color"]) for frame in frames]
    assert len(frame_colors) == len(set(frame_colors)), "locator colors must be unique"
    assert set(grouped) == set(frame_colors), "each fragment color must match exactly one frame"

    for frame in frames:
        marker_color = int(frame["marker_color"])
        owned = grouped.get(marker_color, [])
        assert owned, "each frame must own at least one fragment"
        placements = _pack_top_left(owned, int(frame["inner_h"]), int(frame["inner_w"]))
        r0, c0, _r1, _c1 = frame["bbox"]

        for frag, (norm_r, norm_c) in zip(owned, placements):
            mask = frag["mask"]
            frag_h, frag_w = map(int, mask.shape)
            top, left = _anchor_position(
                norm_r,
                norm_c,
                frag_h,
                frag_w,
                int(frame["inner_h"]),
                int(frame["inner_w"]),
                str(frame["anchor"]),
            )
            base_r = int(r0) + 1 + top
            base_c = int(c0) + 1 + left
            for rr in range(frag_h):
                for cc in range(frag_w):
                    if not mask[rr, cc]:
                        continue
                    assert out[base_r + rr, base_c + cc] == bg
                    out[base_r + rr, base_c + cc] = int(frag["color"])

    return out


def _extract_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    frames: list[dict] = []
    frame_mask = np.zeros_like(g, dtype=bool)
    for pts in _components(g != bg):
        frame = _detect_frame(g, pts, bg)
        if frame is None:
            continue
        frames.append(frame)
        for r, c in frame["cells"]:
            frame_mask[r, c] = True

    assert frames, "expected at least one frame"
    sizes = {(frame["inner_h"], frame["inner_w"]) for frame in frames}
    assert len(sizes) == 1, "all frames must share the same size"

    remainder = (g != bg) & (~frame_mask)
    fragments = [_extract_fragment(g, pts) for pts in _components(remainder)]
    assert fragments, "expected scattered fragments"
    fragments = _sort_reading_order(fragments)

    return bg, _sort_reading_order(frames), fragments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pack each frame's same-colored fragments into its interior, using the marked corner as the anchor.

    The unique corner marker on a frame identifies both which fragment color belongs to that frame
    and which corner of the interior should act as the packing origin.
    """
    bg, frames, fragments = _extract_scene(np.asarray(grid, dtype=int))
    return _render_solution(np.asarray(grid, dtype=int), bg, frames, fragments)


def _random_connected_shape(max_h: int, max_w: int, dense_bias: bool) -> np.ndarray:
    h = random.randint(1, max_h)
    w = random.randint(1, max_w)
    if h == 1 and w == 1:
        return np.ones((1, 1), dtype=bool)

    total = h * w
    min_area = max(1, total * (2 if dense_bias else 1) // 3)
    area = random.randint(min_area, total)
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = set()
    for nr, nc in _neighbors4(*start):
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))

    while len(cells) < area:
        if not frontier:
            break
        chosen = random.choice(tuple(frontier))
        frontier.remove(chosen)
        cells.add(chosen)
        for nr, nc in _neighbors4(*chosen):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    cropped = mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]

    if cropped.size >= 6 and random.random() < 0.35:
        for _ in range(3):
            rr = random.randrange(cropped.shape[0])
            cc = random.randrange(cropped.shape[1])
            if not cropped[rr, cc]:
                continue
            trial = cropped.copy()
            trial[rr, cc] = False
            if not trial.any():
                continue
            if len(_components(trial)) == 1:
                cropped = trial
                break

    return cropped


def _sample_frame_fragments(inner_h: int, inner_w: int, require_wrap: bool) -> list[np.ndarray] | None:
    for _ in range(300):
        if require_wrap:
            count = random.randint(3, 5)
            max_h = min(3, inner_h)
            max_w = min(max(2, inner_w - 1), 4)
            fragments = [_random_connected_shape(max_h, max_w, dense_bias=True) for _ in range(count)]
        else:
            count = random.randint(1, 4)
            max_h = min(3, inner_h)
            max_w = min(3, inner_w)
            fragments = [_random_connected_shape(max_h, max_w, dense_bias=False) for _ in range(count)]

        items = [{"mask": frag} for frag in fragments]
        try:
            placements = _pack_top_left(items, inner_h, inner_w)
        except AssertionError:
            continue

        wrapped = len({r for r, _ in placements}) > 1
        if wrapped != require_wrap:
            continue
        return fragments
    return None


def _mark_blocked(blocked: np.ndarray, r: int, c: int, h: int, w: int) -> None:
    r0 = max(0, r - 1)
    c0 = max(0, c - 1)
    r1 = min(blocked.shape[0], r + h + 1)
    c1 = min(blocked.shape[1], c + w + 1)
    blocked[r0:r1, c0:c1] = True


def _random_place_box(
    blocked: np.ndarray,
    h: int,
    w: int,
    after_key: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    grid_h, grid_w = blocked.shape
    if h + 2 > grid_h or w + 2 > grid_w:
        return None
    for _ in range(500):
        r = random.randint(1, grid_h - h - 1)
        c = random.randint(1, grid_w - w - 1)
        if after_key is not None and (r, c) <= after_key:
            continue
        r0 = r - 1
        c0 = c - 1
        r1 = r + h + 1
        c1 = c + w + 1
        if np.any(blocked[r0:r1, c0:c1]):
            continue
        return r, c
    return None


def _build_frame(
    grid: np.ndarray,
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    marker_color: int,
    anchor: str,
) -> dict:
    h = inner_h + 2
    w = inner_w + 2
    grid[top : top + h, left : left + w] = GREY
    grid[top + 1 : top + h - 1, left + 1 : left + w - 1] = _get_bg(grid)

    corner_offsets = {
        "TL": (0, 0),
        "TR": (0, w - 1),
        "BL": (h - 1, 0),
        "BR": (h - 1, w - 1),
    }
    rr, cc = corner_offsets[anchor]
    grid[top + rr, left + cc] = marker_color
    return {
        "bbox": (top, left, top + h - 1, left + w - 1),
        "border_color": GREY,
        "marker_color": marker_color,
        "anchor": anchor,
        "inner_h": inner_h,
        "inner_w": inner_w,
    }


def _bg_is_dominant(grid: np.ndarray, bg: int) -> bool:
    vals, counts = np.unique(grid, return_counts=True)
    counts_by_color = {int(v): int(c) for v, c in zip(vals, counts)}
    bg_count = counts_by_color.pop(bg, 0)
    return bool(counts_by_color) and bg_count > max(counts_by_color.values())


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([color for color in ALL_COLORS if color != GREY])
    frame_count = random.randint(1, 4)
    inner_h = random.randint(5, 7)
    inner_w = random.randint(6, 8)
    available_colors = [color for color in ALL_COLORS if color not in {bg, GREY}]
    if len(available_colors) < frame_count:
        return None

    marker_colors = random.sample(available_colors, frame_count)
    anchors = random.sample(list(CORNERS), frame_count)
    wrap_required = random.choice([False, True])
    wrap_frame = random.randrange(frame_count) if wrap_required else None

    fragment_masks_by_frame: list[list[np.ndarray]] = []
    for idx in range(frame_count):
        require_wrap = wrap_required and idx == wrap_frame
        masks = _sample_frame_fragments(inner_h, inner_w, require_wrap=require_wrap)
        if masks is None:
            return None
        fragment_masks_by_frame.append(masks)

    total_fragments = sum(len(masks) for masks in fragment_masks_by_frame)
    min_side = max(inner_h + 8, inner_w + 8, 18)
    side = min(MAX_GRID_SIZE, min_side + random.randint(4, 10))
    grid_h = side
    grid_w = side + random.randint(0, min(4, MAX_GRID_SIZE - side))
    grid = np.full((grid_h, grid_w), bg, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    frames: list[dict] = []
    frame_h = inner_h + 2
    frame_w = inner_w + 2
    for marker_color, anchor in zip(marker_colors, anchors):
        pos = _random_place_box(blocked, frame_h, frame_w)
        if pos is None:
            return None
        top, left = pos
        frame = _build_frame(grid, top, left, inner_h, inner_w, marker_color, anchor)
        frames.append(frame)
        _mark_blocked(blocked, top, left, frame_h, frame_w)

    fragment_specs: list[dict] = []
    for marker_color, masks in zip(marker_colors, fragment_masks_by_frame):
        after_key: tuple[int, int] | None = None
        for mask in masks:
            frag_h, frag_w = map(int, mask.shape)
            pos = _random_place_box(blocked, frag_h, frag_w, after_key=after_key)
            if pos is None:
                return None
            top, left = pos
            for rr in range(frag_h):
                for cc in range(frag_w):
                    if mask[rr, cc]:
                        grid[top + rr, left + cc] = marker_color
            fragment_specs.append(
                {
                    "bbox": (top, left, top + frag_h - 1, left + frag_w - 1),
                    "color": marker_color,
                    "mask": mask,
                    "cells": [
                        (top + rr, left + cc)
                        for rr in range(frag_h)
                        for cc in range(frag_w)
                        if mask[rr, cc]
                    ],
                }
            )
            _mark_blocked(blocked, top, left, frag_h, frag_w)
            after_key = (top, left)

    if len(fragment_specs) != total_fragments:
        return None
    if not _bg_is_dominant(grid, bg):
        return None

    fragments_sorted = _sort_reading_order(fragment_specs)
    expected = _render_solution(grid, bg, _sort_reading_order(frames), fragments_sorted)
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
