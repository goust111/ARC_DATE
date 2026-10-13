from __future__ import annotations

import random
from collections import Counter
from itertools import combinations

import numpy as np


BG = 0
GREY = 5
MARKER_COLORS = (1, 2, 3, 4)
TRANSFORMS = tuple((flip, rot) for flip in (False, True) for rot in range(4))


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def apply_transform(arr: np.ndarray, flip: bool, rot: int) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr.copy()
    return np.rot90(out, rot)


def marker_map(patch: np.ndarray) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere((patch != BG) & (patch != GREY)):
        out[(int(r), int(c))] = int(patch[r, c])
    return out


def _filled_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    rects = []
    for coords in components_mask(grid != BG):
        bbox = bbox_coords(coords)
        patch = crop_bbox(grid, bbox)
        if np.all(patch != BG) and GREY in patch:
            rects.append(bbox)
    assert len(rects) >= 2
    shapes = {(r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1 in rects}
    assert len(shapes) == 1
    return rects


def _frame_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames = []
    for coords in components_mask(grid == GREY):
        bbox = bbox_coords(coords)
        patch = crop_bbox(grid, bbox)
        h, w = patch.shape
        if h < 3 or w < 3:
            continue
        frame = np.zeros((h, w), dtype=bool)
        frame[0, :] = True
        frame[-1, :] = True
        frame[:, 0] = True
        frame[:, -1] = True
        if np.array_equal(patch == GREY, frame):
            frames.append(bbox)
    assert len(frames) == 1
    return frames


def _merge_sources(grid: np.ndarray, rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    h = rects[0][2] - rects[0][0] + 1
    w = rects[0][3] - rects[0][1] + 1
    merged = np.full((h, w), GREY, dtype=int)
    for bbox in rects:
        patch = crop_bbox(grid, bbox)
        for (r, c), color in marker_map(patch).items():
            assert int(merged[r, c]) in (GREY, color)
            merged[r, c] = color
    assert np.count_nonzero((merged != GREY) & (merged != BG)) >= 3
    return merged


def _anchors_in_frame(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> dict[tuple[int, int], int]:
    r0, c0, r1, c1 = frame_bbox
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    anchors = marker_map(inner)
    assert 1 <= len(anchors) <= 2
    return anchors


def _matching_transforms(merged: np.ndarray, frame_shape: tuple[int, int], anchors: dict[tuple[int, int], int]) -> list[tuple[bool, int]]:
    matches = []
    for flip, rot in TRANSFORMS:
        transformed = apply_transform(merged, flip, rot)
        if transformed.shape != frame_shape:
            continue
        transformed_markers = marker_map(transformed)
        if all(transformed_markers.get(pos) == color for pos, color in anchors.items()):
            matches.append((flip, rot))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """Merge marker layers from same-sized gray rectangles, rotate/reflect the merged sprite, and fill the matching target frame.

    Input structure:
    - Black background surrounds several same-sized gray filled rectangles whose interiors may contain non-gray marker pixels.
    - A separate gray hollow rectangle frame of matching interior size contains one or two colored anchor markers on black cells.
    - Source rectangles never overlap, and marker colors are drawn from a small visible palette distinct from black and gray.

    Rule:
    - Detect the filled gray source rectangles and overlay all their non-gray markers by local coordinates to form one merged rectangle on a gray base.
    - Enumerate the 8 rigid transforms of that merged rectangle and keep the unique transform whose marker pixels agree with the target frame's anchor markers at the same local coordinates.
    - Replace the black interior of the target frame with the entire transformed merged rectangle while preserving the gray frame border.

    Key insight:
    - The anchor markers do not identify an orientation from any single source layer; only the merged marker pattern yields a unique transformed match for the target frame.
    """
    g = np.asarray(grid, dtype=int)
    rects = _filled_rectangles(g)
    frame_bbox = _frame_rectangles(g)[0]
    merged = _merge_sources(g, rects)

    r0, c0, r1, c1 = frame_bbox
    frame_shape = (r1 - r0 - 1, c1 - c0 - 1)
    anchors = _anchors_in_frame(g, frame_bbox)
    matches = _matching_transforms(merged, frame_shape, anchors)
    assert len(matches) == 1

    fill = apply_transform(merged, *matches[0])
    out = g.copy()
    out[r0 + 1 : r1, c0 + 1 : c1] = fill
    return out


def _subset_matches(layers: list[dict[tuple[int, int], int]], subset: tuple[int, ...], frame_shape: tuple[int, int], anchors: dict[tuple[int, int], int], base_shape: tuple[int, int]) -> int:
    merged = np.full(base_shape, GREY, dtype=int)
    for idx in subset:
        for (r, c), color in layers[idx].items():
            assert int(merged[r, c]) in (GREY, color)
            merged[r, c] = color
    return len(_matching_transforms(merged, frame_shape, anchors))


def _unique_transform_count(merged: np.ndarray, frame_shape: tuple[int, int], anchors: dict[tuple[int, int], int]) -> int:
    return len(_matching_transforms(merged, frame_shape, anchors))


def _random_marker_cells(h: int, w: int, count: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(cells)
    return cells[:count]


def _split_layers(merged_markers: dict[tuple[int, int], int], layer_count: int) -> list[dict[tuple[int, int], int]]:
    items = list(merged_markers.items())
    random.shuffle(items)
    layers = [dict() for _ in range(layer_count)]
    for i, item in enumerate(items):
        layers[i % layer_count][item[0]] = item[1]
    if any(not layer for layer in layers):
        return []
    return layers


def _placements(shape: tuple[int, int], patch_shapes: list[tuple[int, int]], gap: int = 2) -> list[tuple[int, int]] | None:
    h, w = shape
    order = list(range(len(patch_shapes)))
    random.shuffle(order)
    occupied = np.zeros((h, w), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(patch_shapes)
    for idx in order:
        ph, pw = patch_shapes[idx]
        candidates = []
        for r in range(1, h - ph):
            for c in range(1, w - pw):
                r0 = max(0, r - gap)
                c0 = max(0, c - gap)
                r1 = min(h, r + ph + gap)
                c1 = min(w, c + pw + gap)
                if occupied[r0:r1, c0:c1].any():
                    continue
                candidates.append((r, c))
        if not candidates:
            return None
        pos = random.choice(candidates)
        positions[idx] = pos
        r, c = pos
        occupied[r : r + ph, c : c + pw] = True
    return [p for p in positions if p is not None]


def _paint_sources(grid: np.ndarray, positions: list[tuple[int, int]], shape: tuple[int, int], layers: list[dict[tuple[int, int], int]]) -> None:
    h, w = shape
    for (r0, c0), layer in zip(positions, layers):
        grid[r0 : r0 + h, c0 : c0 + w] = GREY
        for (r, c), color in layer.items():
            grid[r0 + r, c0 + c] = color


def _paint_frame(grid: np.ndarray, top_left: tuple[int, int], inner_shape: tuple[int, int], anchors: dict[tuple[int, int], int]) -> None:
    r0, c0 = top_left
    h, w = inner_shape
    grid[r0 : r0 + h + 2, c0 : c0 + w + 2] = BG
    grid[r0, c0 : c0 + w + 2] = GREY
    grid[r0 + h + 1, c0 : c0 + w + 2] = GREY
    grid[r0 : r0 + h + 2, c0] = GREY
    grid[r0 : r0 + h + 2, c0 + w + 1] = GREY
    for (r, c), color in anchors.items():
        grid[r0 + 1 + r, c0 + 1 + c] = color


def _build_candidate_case() -> tuple[np.ndarray, np.ndarray] | None:
    src_h, src_w = random.choice([(4, 5), (5, 4), (4, 6), (6, 4), (5, 6), (6, 5)])
    source_count = random.choice([2, 3])
    total_markers = random.randint(max(4, source_count + 1), min(src_h * src_w - 2, 7))
    merged_cells = _random_marker_cells(src_h, src_w, total_markers)

    color_pool = [random.choice(MARKER_COLORS) for _ in range(total_markers)]
    if len(set(color_pool)) == 1:
        color_pool[0] = random.choice(MARKER_COLORS)
        color_pool[-1] = random.choice([c for c in MARKER_COLORS if c != color_pool[0]])
    merged_markers = {cell: color for cell, color in zip(merged_cells, color_pool)}
    layers = _split_layers(merged_markers, source_count)
    if not layers:
        return None

    merged = np.full((src_h, src_w), GREY, dtype=int)
    for layer in layers:
        for (r, c), color in layer.items():
            merged[r, c] = color

    choices = [(flip, rot) for flip, rot in TRANSFORMS if rot != 0]
    target_tf = random.choice(choices)
    transformed = apply_transform(merged, *target_tf)
    th, tw = transformed.shape

    transformed_markers = list(marker_map(transformed).items())
    if len(transformed_markers) < 4:
        return None
    anchor_count = random.choice([1, 2])
    anchor_choices = list(combinations(range(len(transformed_markers)), anchor_count))
    random.shuffle(anchor_choices)

    random.shuffle(layers)
    layer_maps = [dict(layer) for layer in layers]
    for anchor_idx in anchor_choices:
        anchors = {transformed_markers[i][0]: transformed_markers[i][1] for i in anchor_idx}
        if _unique_transform_count(merged, (th, tw), anchors) != 1:
            continue
        if any(
            _subset_matches(layer_maps, subset, (th, tw), anchors, (src_h, src_w)) == 1
            for subset_size in range(1, source_count)
            for subset in combinations(range(source_count), subset_size)
        ):
            continue
        if len(marker_map(transformed)) <= len(anchors):
            continue

        grid_h = random.randint(max(src_h, th) + 8, 24)
        grid_w = random.randint(max(src_w, tw) + 8, 24)
        patch_shapes = [(src_h, src_w)] * source_count + [(th + 2, tw + 2)]
        positions = _placements((grid_h, grid_w), patch_shapes, gap=2)
        if positions is None:
            continue

        grid = np.zeros((grid_h, grid_w), dtype=int)
        _paint_sources(grid, positions[:source_count], (src_h, src_w), layer_maps)
        _paint_frame(grid, positions[-1], (th, tw), anchors)
        out = main(grid)
        expected = grid.copy()
        fr, fc = positions[-1]
        expected[fr + 1 : fr + 1 + th, fc + 1 : fc + 1 + tw] = transformed
        if not np.array_equal(out, expected):
            continue

        # Additional semantic checks: a wrong transform must disagree on at least one anchor.
        mismatch_count = 0
        for tf in TRANSFORMS:
            if tf == target_tf:
                continue
            candidate = apply_transform(merged, *tf)
            if candidate.shape != (th, tw):
                continue
            cmap = marker_map(candidate)
            if any(cmap.get(pos) != color for pos, color in anchors.items()):
                mismatch_count += 1
        if mismatch_count == 0:
            continue
        return grid, expected
    return None


def generate_input() -> np.ndarray:
    for _ in range(500):
        made = _build_candidate_case()
        if made is not None:
            return made[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
