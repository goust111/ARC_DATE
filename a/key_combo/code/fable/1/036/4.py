from __future__ import annotations

from collections import Counter

import numpy as np


FOG_COLOR = 5
FRAGMENT_COLOR = 7
GENERAL_COLORS = [0, 1, 2, 3, 4, 6, 8, 9]
GENERATION_RETRIES = 400

BASE_SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (1, 0), (2, 0), (2, 1)),                    # L tetromino
    ((0, 0), (0, 1), (0, 2), (1, 1)),                    # T tetromino
    ((0, 0), (0, 1), (1, 1), (1, 2)),                    # Z tetromino
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),            # P pentomino
    ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),            # U pentomino
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),            # W pentomino
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 0)),            # F-like pentomino
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),            # plus pentomino
    ((0, 1), (1, 1), (2, 0), (2, 1), (3, 1)),            # Y-like pentomino
)

_RNG = np.random.default_rng()


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
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


def _normalize_coords(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(coords, dtype=int)
    arr = arr - arr.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in arr.tolist()))


def _shape_dims(rel: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in rel]
    cols = [c for _, c in rel]
    return max(rows) + 1, max(cols) + 1


def _shape_mask(rel: tuple[tuple[int, int], ...]) -> np.ndarray:
    h, w = _shape_dims(rel)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in rel:
        mask[r, c] = True
    return mask


def _transform_shape(
    rel: tuple[tuple[int, int], ...],
    rot: int,
    flip_lr: bool,
) -> tuple[tuple[int, int], ...]:
    mask = _shape_mask(rel)
    if flip_lr:
        mask = np.fliplr(mask)
    mask = np.rot90(mask, int(rot))
    return _normalize_coords(np.argwhere(mask))


def _paint_shape(
    grid: np.ndarray,
    rel: tuple[tuple[int, int], ...],
    anchor: tuple[int, int],
    color: int,
) -> None:
    r0, c0 = anchor
    for dr, dc in rel:
        grid[r0 + dr, c0 + dc] = int(color)


def _find_separator_row(grid: np.ndarray, bg: int) -> int:
    rows = [
        r
        for r in range(grid.shape[0])
        if np.all(grid[r] == grid[r, 0]) and int(grid[r, 0]) != bg
    ]
    assert len(rows) == 1, "expected exactly one full non-background separator row"
    return int(rows[0])


def _extract_legend_shapes(legend: np.ndarray, bg: int) -> list[dict]:
    shapes: list[dict] = []
    for color in sorted(set(map(int, legend.flat)) - {int(bg)}):
        for coords in components_mask(legend == color):
            rel = _normalize_coords(coords)
            h, w = _shape_dims(rel)
            shapes.append(
                {
                    "color": int(color),
                    "rel": rel,
                    "h": h,
                    "w": w,
                }
            )
    assert shapes, "expected at least one legend shape"
    return shapes


def _find_matches(scene: np.ndarray, shapes: list[dict]) -> list[dict]:
    fragment_coords = np.argwhere(scene == FRAGMENT_COLOR)
    assert fragment_coords.size > 0, "expected orange fragments"

    fragments = frozenset((int(r), int(c)) for r, c in fragment_coords.tolist())
    fog = frozenset((int(r), int(c)) for r, c in np.argwhere(scene == FOG_COLOR).tolist())
    h, w = scene.shape
    matches: list[dict] = []

    for idx, shape in enumerate(shapes):
        rel = shape["rel"]
        for r0 in range(h - shape["h"] + 1):
            for c0 in range(w - shape["w"] + 1):
                cells = frozenset((r0 + dr, c0 + dc) for dr, dc in rel)
                if not fragments <= cells:
                    continue
                if not (cells - fragments) <= fog:
                    continue
                matches.append(
                    {
                        "shape_index": idx,
                        "color": int(shape["color"]),
                        "anchor": (int(r0), int(c0)),
                        "cells": cells,
                    }
                )
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the only legend shape that explains the orange fragments, restore it in its true color, and clear the fog.

    Input structure:
    - A single full separator row splits the grid into a top legend zone and a bottom scene zone.
    - The legend zone contains several separated monochrome shapes with distinct colors.
    - The scene zone contains only background, grey rectangular fog blocks, and orange fragment pixels.

    Rule:
    - Read each legend shape as a translation-only template.
    - In the scene, find the unique legend shape and placement whose cells cover every orange fragment, while every shape cell that is not an orange fragment lies inside grey fog.
    - Output only the scene zone: erase all fog, draw the recovered full shape in its legend color, and leave all other cells as background.

    Key insight:
    - The target is identified jointly by two constraints: every visible fragment must belong to the same translated legend mask, and every missing part of that mask must be hidden under fog.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    sep_r = _find_separator_row(g, bg)
    legend = g[:sep_r]
    scene = g[sep_r + 1 :]
    assert legend.size > 0 and scene.size > 0, "expected non-empty legend and scene zones"

    shapes = _extract_legend_shapes(legend, bg)
    matches = _find_matches(scene, shapes)
    assert len(matches) == 1, "legend shape and translation must be uniquely determined"

    match = matches[0]
    out = np.full(scene.shape, bg, dtype=int)
    for r, c in match["cells"]:
        out[r, c] = int(match["color"])
    return out


def _randint_inclusive(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(int(lo), int(hi) + 1))


def _bbox_from_cells(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(sorted(cells), dtype=int)
    return bbox_coords(coords)


def _intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _enumerate_target_rects(
    scene_shape: tuple[int, int],
    target_cells: set[tuple[int, int]],
) -> list[dict]:
    h, w = scene_shape
    r0, c0, r1, c1 = _bbox_from_cells(target_cells)
    row_lo = max(0, r0 - 1)
    row_hi = min(h - 1, r1 + 1)
    col_lo = max(0, c0 - 1)
    col_hi = min(w - 1, c1 + 1)

    candidates: list[dict] = []
    for rr0 in range(row_lo, row_hi + 1):
        for rr1 in range(rr0, row_hi + 1):
            for cc0 in range(col_lo, col_hi + 1):
                for cc1 in range(cc0, col_hi + 1):
                    area = (rr1 - rr0 + 1) * (cc1 - cc0 + 1)
                    if area > 12:
                        continue
                    covered = frozenset(
                        (r, c)
                        for r, c in target_cells
                        if rr0 <= r <= rr1 and cc0 <= c <= cc1
                    )
                    if not covered:
                        continue
                    candidates.append({"rect": (rr0, cc0, rr1, cc1), "cover": covered})
    return candidates


def _visible_count_ok(area: int, visible_count: int, mode: int) -> bool:
    if not (1 <= visible_count <= area - 1):
        return False
    if mode == 0:
        return visible_count <= min(2, area - 1)
    return visible_count >= max(2, area - 2)


def _choose_primary_rects(
    rng: np.random.Generator,
    scene_shape: tuple[int, int],
    target_cells: set[tuple[int, int]],
    block_count: int,
    visible_mode: int,
) -> tuple[list[tuple[int, int, int, int]], set[tuple[int, int]]] | None:
    candidates = _enumerate_target_rects(scene_shape, target_cells)
    if len(candidates) < block_count:
        return None

    target_area = len(target_cells)
    for _ in range(240):
        idxs = rng.choice(len(candidates), size=block_count, replace=False)
        covered: set[tuple[int, int]] = set()
        rects: list[tuple[int, int, int, int]] = []
        ok = True
        for idx in idxs.tolist():
            piece = set(candidates[idx]["cover"])
            if piece <= covered:
                ok = False
                break
            rects.append(candidates[idx]["rect"])
            covered |= piece
        if not ok:
            continue
        visible_count = target_area - len(covered)
        if not _visible_count_ok(target_area, visible_count, visible_mode):
            continue
        return rects, set(target_cells) - covered
    return None


def _choose_extra_rects(
    rng: np.random.Generator,
    scene_shape: tuple[int, int],
    target_bbox: tuple[int, int, int, int],
    existing: list[tuple[int, int, int, int]],
    extra_count: int,
) -> list[tuple[int, int, int, int]]:
    if extra_count == 0:
        return []

    h, w = scene_shape
    avoid = (
        max(0, target_bbox[0] - 1),
        max(0, target_bbox[1] - 1),
        min(h - 1, target_bbox[2] + 1),
        min(w - 1, target_bbox[3] + 1),
    )

    extras: list[tuple[int, int, int, int]] = []
    for _ in range(120):
        if len(extras) == extra_count:
            break
        rect_h = _randint_inclusive(rng, 1, min(4, h))
        rect_w = _randint_inclusive(rng, 1, min(5, max(1, w - 1)))
        if rect_w >= w:
            continue
        r0 = _randint_inclusive(rng, 0, h - rect_h)
        c0 = _randint_inclusive(rng, 0, w - rect_w)
        rect = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
        if _intersects(rect, avoid):
            continue
        if any(_intersects(rect, other) for other in [*existing, *extras]):
            continue
        extras.append(rect)
    return extras


def _sample_legend_shapes(
    rng: np.random.Generator,
    count: int,
) -> list[tuple[tuple[int, int], ...]] | None:
    for _ in range(80):
        base_ids = rng.choice(len(BASE_SHAPES), size=count, replace=False)
        shapes: list[tuple[tuple[int, int], ...]] = []
        seen: set[tuple[tuple[int, int], ...]] = set()
        ok = True
        for base_idx in base_ids.tolist():
            rel = _transform_shape(
                BASE_SHAPES[base_idx],
                rot=int(rng.integers(0, 4)),
                flip_lr=bool(rng.integers(0, 2)),
            )
            if rel in seen:
                ok = False
                break
            seen.add(rel)
            shapes.append(rel)
        if ok:
            return shapes
    return None


def _build_input_from_spec(
    bg: int,
    separator: int,
    width: int,
    top_h: int,
    scene_h: int,
    legend_shapes: list[tuple[tuple[int, int], ...]],
    legend_colors: list[int],
    gaps: list[int],
    target_index: int,
    target_anchor: tuple[int, int],
    fog_rects: list[tuple[int, int, int, int]],
    fragments: set[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    legend = np.full((top_h, width), bg, dtype=int)
    total_span = sum(_shape_dims(rel)[1] for rel in legend_shapes) + sum(gaps)
    start_c = 1 if width - total_span - 1 == 1 else _randint_inclusive(_RNG, 1, width - total_span - 1)
    cur_c = start_c
    for i, (rel, color) in enumerate(zip(legend_shapes, legend_colors)):
        shape_h, shape_w = _shape_dims(rel)
        max_r0 = top_h - shape_h - 1
        r0 = _randint_inclusive(_RNG, 1, max_r0)
        _paint_shape(legend, rel, (r0, cur_c), color)
        cur_c += shape_w + (gaps[i] if i < len(gaps) else 0)

    scene = np.full((scene_h, width), bg, dtype=int)
    for r0, c0, r1, c1 in fog_rects:
        scene[r0 : r1 + 1, c0 : c1 + 1] = FOG_COLOR
    for r, c in fragments:
        scene[r, c] = FRAGMENT_COLOR

    out = np.full((scene_h, width), bg, dtype=int)
    _paint_shape(out, legend_shapes[target_index], target_anchor, legend_colors[target_index])

    sep = np.full((1, width), separator, dtype=int)
    inp = np.vstack([legend, sep, scene])
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    rng = _RNG
    legend_count = int(rng.integers(3, 6))
    legend_shapes = _sample_legend_shapes(rng, legend_count)
    if legend_shapes is None:
        return None

    target_index = int(rng.integers(0, legend_count))
    target_rel = legend_shapes[target_index]
    target_h, target_w = _shape_dims(target_rel)

    gaps = [_randint_inclusive(rng, 1, 3) for _ in range(legend_count - 1)]
    legend_span = sum(_shape_dims(rel)[1] for rel in legend_shapes) + sum(gaps)

    width_lo = max(12, target_w + 6, legend_span + 2)
    width_hi = min(28, width_lo + 6)
    if width_lo > width_hi:
        return None
    width = _randint_inclusive(rng, width_lo, width_hi)

    top_h = max(_shape_dims(rel)[0] for rel in legend_shapes) + _randint_inclusive(rng, 2, 3)
    scene_h_lo = max(8, target_h + 5)
    scene_h_hi = min(15, scene_h_lo + 4)
    if top_h + 1 + scene_h_lo > 30:
        return None
    scene_h = _randint_inclusive(rng, scene_h_lo, scene_h_hi)

    colors = rng.permutation(np.array(GENERAL_COLORS, dtype=int)).tolist()
    bg = int(colors[0])
    separator = int(colors[1])
    legend_colors = [int(c) for c in colors[2 : 2 + legend_count]]

    max_r0 = scene_h - target_h - 1
    max_c0 = width - target_w - 1
    if max_r0 < 1 or max_c0 < 1:
        return None
    target_anchor = (_randint_inclusive(rng, 1, max_r0), _randint_inclusive(rng, 1, max_c0))
    target_cells = {
        (target_anchor[0] + dr, target_anchor[1] + dc)
        for dr, dc in target_rel
    }

    primary = _choose_primary_rects(
        rng,
        (scene_h, width),
        target_cells,
        block_count=int(rng.choice([1, 2, 3])),
        visible_mode=int(rng.integers(0, 2)),
    )
    if primary is None:
        return None
    primary_rects, fragments = primary
    if not fragments or len(fragments) == len(target_cells):
        return None

    target_bbox = _bbox_from_cells(target_cells)
    extra_rects = _choose_extra_rects(
        rng,
        (scene_h, width),
        target_bbox,
        primary_rects,
        extra_count=int(rng.integers(0, 3)),
    )

    inp, expected = _build_input_from_spec(
        bg=bg,
        separator=separator,
        width=width,
        top_h=top_h,
        scene_h=scene_h,
        legend_shapes=legend_shapes,
        legend_colors=legend_colors,
        gaps=gaps,
        target_index=target_index,
        target_anchor=target_anchor,
        fog_rects=[*primary_rects, *extra_rects],
        fragments=fragments,
    )

    if inp.shape[0] > 30 or inp.shape[1] > 30 or expected.shape[0] > 30 or expected.shape[1] > 30:
        return None
    if get_bg(inp) != bg:
        return None

    sep_r = _find_separator_row(inp, bg)
    legend = inp[:sep_r]
    scene = inp[sep_r + 1 :]
    shapes = _extract_legend_shapes(legend, bg)
    matches = _find_matches(scene, shapes)
    if len(matches) != 1:
        return None
    if matches[0]["anchor"] != target_anchor or matches[0]["color"] != legend_colors[target_index]:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fog-shadow ARC input")
