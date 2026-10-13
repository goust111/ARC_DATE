from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
NON_BLACK_NON_GREY = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _normalize_cells(cells: set[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    pts = list(cells)
    r0 = min(r for r, _c in pts)
    c0 = min(c for _r, c in pts)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in pts))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _c in shape) + 1, max(c for _r, c in shape) + 1


def _shape_mask(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    h, w = _shape_dims(shape)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return mask


def _is_orthogonally_convex(shape: tuple[tuple[int, int], ...]) -> bool:
    cells = set(shape)
    rows = sorted({r for r, _c in shape})
    cols = sorted({c for _r, c in shape})
    for r in rows:
        cs = sorted(c for rr, c in shape if rr == r)
        if cs != list(range(cs[0], cs[-1] + 1)):
            return False
    for c in cols:
        rs = sorted(r for r, cc in shape if cc == c)
        if rs != list(range(rs[0], rs[-1] + 1)):
            return False
    return len(cells) >= 3


def _dihedral_variants(shape: tuple[tuple[int, int], ...]) -> set[tuple[tuple[int, int], ...]]:
    mask = _shape_mask(shape)
    variants: set[tuple[tuple[int, int], ...]] = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        for arr in (rotated, np.fliplr(rotated)):
            coords = tuple(sorted((int(r), int(c)) for r, c in np.argwhere(arr)))
            variants.add(_normalize_cells(coords))
    return variants


def _enumerate_shapes(max_cells: int = 6, max_span: int = 4) -> list[tuple[tuple[int, int], ...]]:
    shapes_by_size: dict[int, set[tuple[tuple[int, int], ...]]] = {1: {((0, 0),)}}
    for size in range(2, max_cells + 1):
        prev = shapes_by_size[size - 1]
        cur: set[tuple[tuple[int, int], ...]] = set()
        for shape in prev:
            cells = set(shape)
            frontier = set()
            for r, c in shape:
                frontier.update(_neighbors4(r, c))
            for nr, nc in frontier - cells:
                new_shape = _normalize_cells(cells | {(nr, nc)})
                h, w = _shape_dims(new_shape)
                if max(h, w) <= max_span:
                    cur.add(new_shape)
        shapes_by_size[size] = cur
    out = []
    for size in range(3, max_cells + 1):
        for shape in sorted(shapes_by_size[size]):
            if _is_orthogonally_convex(shape):
                out.append(shape)
    return out


SHAPE_POOL = _enumerate_shapes()
ASYMMETRIC_TEMPLATE_POOL = [shape for shape in SHAPE_POOL if len(_dihedral_variants(shape)) > 1]


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_monochrome_components(grid: np.ndarray, exclude: set[int]) -> list[dict]:
    comps = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in exclude):
        for coords in _components_of_color(grid, color):
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "shape": _normalize_cells(tuple(map(tuple, coords.tolist()))),
                }
            )
    return comps


def _frame_bbox_and_template(grid: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[tuple[int, int], ...]]:
    grey_comps = _components_of_color(grid, GREY)
    assert len(grey_comps) == 1, "expected exactly one grey component"
    coords = grey_comps[0]
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    frame_cells = {(int(r), int(c)) for r, c in coords}
    expected = {(r1, c) for c in range(c0, c1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)}
    assert frame_cells == expected, "grey component must be a bottom-right L-frame"
    assert r1 > r0 and c1 > c0, "frame must enclose a non-empty interior"

    interior = grid[r0:r1, c0:c1]
    template_colors = [int(v) for v in np.unique(interior) if int(v) not in (BLACK, GREY)]
    assert len(template_colors) == 1, "frame interior must contain one template color"
    template_color = template_colors[0]
    template_coords = np.argwhere(interior == template_color)
    assert len(template_coords) >= 3, "template must contain 3-6 cells"
    return (int(r0), int(c0), int(r1), int(c1)), _normalize_cells(tuple(map(tuple, template_coords.tolist())))


def _selected_components(grid: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[tuple[int, int], ...], list[dict], list[dict]]:
    frame_bbox, template_shape = _frame_bbox_and_template(grid)
    r0, c0, r1, c1 = frame_bbox
    external = []
    for comp in _find_monochrome_components(grid, exclude={BLACK, GREY}):
        coords = comp["coords"]
        if np.any((r0 <= coords[:, 0]) & (coords[:, 0] <= r1) & (c0 <= coords[:, 1]) & (coords[:, 1] <= c1)):
            continue
        external.append(comp)
    matches = [comp for comp in external if comp["shape"] == template_shape]
    nonmatches = [comp for comp in external if comp["shape"] != template_shape]
    assert len(matches) >= 2, "expected at least two matching outside objects"
    assert len(nonmatches) >= 2, "expected at least two non-matching outside objects"
    return frame_bbox, template_shape, matches, nonmatches


def _fill_span_cells(base: np.ndarray, selected_coords: np.ndarray) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    row_adds: set[tuple[int, int]] = set()
    col_adds: set[tuple[int, int]] = set()
    selected = np.zeros(base.shape, dtype=bool)
    selected[selected_coords[:, 0], selected_coords[:, 1]] = True
    h, w = base.shape

    for r in range(h):
        cols = np.flatnonzero(selected[r])
        if len(cols) >= 2:
            for c in range(int(cols[0]), int(cols[-1]) + 1):
                if int(base[r, c]) == BLACK:
                    row_adds.add((int(r), int(c)))

    for c in range(w):
        rows = np.flatnonzero(selected[:, c])
        if len(rows) >= 2:
            for r in range(int(rows[0]), int(rows[-1]) + 1):
                if int(base[r, c]) == BLACK:
                    col_adds.add((int(r), int(c)))

    return row_adds, col_adds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the outside objects whose shape matches the framed template, then connect the kept pixels by row and column spans.

    Input structure:
    - A black background contains one grey L-frame in the bottom-right corner, with its horizontal leg on the bottom and its vertical leg on the right.
    - Inside the frame interior there is one small monochrome template shape in a non-black, non-grey color.
    - Elsewhere on the grid there are several small monochrome 4-connected objects that do not touch the frame.

    Rule:
    - Read the template shape inside the grey frame by taking its normalized binary mask inside the frame interior.
    - Compare every outside monochrome 4-connected object against that normalized mask, ignoring color and translation only.
    - Keep every matching outside object in its original color and erase every non-matching outside object to black.
    - Using only the kept outside-object pixels as markers, fill each row span and each column span between the outermost kept pixels with grey, but only on background cells.
    - Do not overwrite the kept colored markers, the grey frame, or the framed template.

    Key insight:
    - The grey L-frame does not mark an output color target; it selects which outside objects are allowed to participate in the later span-connection step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _frame_bbox, _template_shape, matches, nonmatches = _selected_components(g)
    out = g.copy()
    for comp in nonmatches:
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = BLACK

    selected_coords = np.concatenate([comp["coords"] for comp in matches], axis=0)
    row_adds, col_adds = _fill_span_cells(out, selected_coords)
    for r, c in sorted(row_adds | col_adds):
        if int(out[r, c]) == BLACK:
            out[r, c] = GREY
    return out


def _all_candidate_anchors(shape: tuple[tuple[int, int], ...], h: int, w: int, blocked: set[tuple[int, int]]) -> list[tuple[int, int]]:
    sh, sw = _shape_dims(shape)
    anchors = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            abs_cells = [(r0 + r, c0 + c) for r, c in shape]
            if all((rr, cc) not in blocked for rr, cc in abs_cells):
                anchors.append((r0, c0))
    return anchors


def _blocked_from_bbox(frame_bbox: tuple[int, int, int, int], h: int, w: int) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    blocked = set()
    for r in range(max(0, r0 - 1), min(h, r1 + 2)):
        for c in range(max(0, c0 - 1), min(w, c1 + 2)):
            blocked.add((r, c))
    return blocked


def _add_blocked(blocked: set[tuple[int, int]], cells: list[tuple[int, int]], h: int, w: int) -> None:
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked.add((nr, nc))


def _place_shape(
    placements: list[dict],
    blocked: set[tuple[int, int]],
    shape: tuple[tuple[int, int], ...],
    color: int,
    h: int,
    w: int,
    *,
    fixed_row: int | None = None,
    fixed_col: int | None = None,
) -> bool:
    anchors = _all_candidate_anchors(shape, h, w, blocked)
    if fixed_row is not None:
        anchors = [anchor for anchor in anchors if anchor[0] == int(fixed_row)]
    if fixed_col is not None:
        anchors = [anchor for anchor in anchors if anchor[1] == int(fixed_col)]
    if not anchors:
        return False
    r0, c0 = random.choice(anchors)
    cells = [(r0 + r, c0 + c) for r, c in shape]
    placements.append({"shape": shape, "color": int(color), "anchor": (int(r0), int(c0)), "cells": cells})
    _add_blocked(blocked, cells, h, w)
    return True


def _paint_scene(
    h: int,
    w: int,
    frame_bbox: tuple[int, int, int, int],
    template_shape: tuple[tuple[int, int], ...],
    template_color: int,
    template_offset: tuple[int, int],
    matches: list[dict],
    distractors: list[dict],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    fr0, fc0, fr1, fc1 = frame_bbox
    grid[fr1, fc0 : fc1 + 1] = GREY
    grid[fr0 : fr1 + 1, fc1] = GREY
    tr, tc = template_offset
    for r, c in template_shape:
        grid[fr0 + tr + r, fc0 + tc + c] = int(template_color)
    for obj in matches + distractors:
        for r, c in obj["cells"]:
            grid[r, c] = int(obj["color"])
    return grid


def _selected_coords(objects: list[dict]) -> np.ndarray:
    return np.array([cell for obj in objects for cell in obj["cells"]], dtype=int)


def _row_col_mode_ok(
    base_match_grid: np.ndarray, selected_match_coords: np.ndarray, mode: str
) -> tuple[bool, set[tuple[int, int]], set[tuple[int, int]]]:
    row_adds, col_adds = _fill_span_cells(base_match_grid, selected_match_coords)
    if mode == "both":
        ok = bool(row_adds) and bool(col_adds)
    elif mode == "row":
        ok = bool(row_adds) and not col_adds
    else:
        ok = bool(col_adds) and not row_adds
    return ok, row_adds, col_adds


def _random_template_shape() -> tuple[tuple[int, int], ...]:
    pool = ASYMMETRIC_TEMPLATE_POOL if random.random() < 0.7 else SHAPE_POOL
    return random.choice(pool)


def _nonmatching_shapes(template_shape: tuple[tuple[int, int], ...], count: int) -> list[tuple[tuple[int, int], ...]]:
    variants = [shape for shape in _dihedral_variants(template_shape) if shape != template_shape]
    pool = [shape for shape in SHAPE_POOL if shape != template_shape]
    chosen = []
    if variants:
        variant = random.choice(variants)
        if variant in pool:
            chosen.append(variant)
    while len(chosen) < count:
        candidate = random.choice(pool)
        if candidate not in chosen:
            chosen.append(candidate)
    return chosen


def _all_external_fill_cells(base_all_grid: np.ndarray, matches: list[dict], distractors: list[dict]) -> set[tuple[int, int]]:
    all_coords = _selected_coords(matches + distractors)
    row_adds, col_adds = _fill_span_cells(base_all_grid, all_coords)
    return row_adds | col_adds


def _base_grid_without_distractors(
    h: int,
    w: int,
    frame_bbox: tuple[int, int, int, int],
    template_shape: tuple[tuple[int, int], ...],
    template_color: int,
    template_offset: tuple[int, int],
    matches: list[dict],
) -> np.ndarray:
    return _paint_scene(h, w, frame_bbox, template_shape, template_color, template_offset, matches, [])


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(15, 19)
    w = random.randint(15, 19)
    template_shape = _random_template_shape()
    th, tw = _shape_dims(template_shape)

    inner_h = random.randint(max(3, th), min(5, max(3, th + 1)))
    inner_w = random.randint(max(3, tw), min(5, max(3, tw + 1)))
    frame_bbox = (h - (inner_h + 1), w - (inner_w + 1), h - 1, w - 1)
    tr = random.randint(0, inner_h - th)
    tc = random.randint(0, inner_w - tw)
    template_offset = (tr, tc)

    mode = random.choices(["both", "row", "col"], weights=[6, 2, 2], k=1)[0]
    total_objects = random.randint(5, 8)
    if mode == "both":
        match_count = random.randint(3, min(4, total_objects - 2))
    else:
        match_count = random.randint(2, min(3, total_objects - 2))
    distractor_count = total_objects - match_count
    if distractor_count < 2:
        return None

    external_colors = random.sample(NON_BLACK_NON_GREY, total_objects)
    template_color = random.choice(NON_BLACK_NON_GREY)
    blocked = _blocked_from_bbox(frame_bbox, h, w)
    matches: list[dict] = []

    if mode == "both":
        blocked_a = set(blocked)
        if not _place_shape(matches, blocked_a, template_shape, external_colors[0], h, w):
            return None
        r0, c0 = matches[0]["anchor"]
        if not _place_shape(matches, blocked_a, template_shape, external_colors[1], h, w, fixed_row=r0):
            return None
        if not _place_shape(matches, blocked_a, template_shape, external_colors[2], h, w, fixed_col=c0):
            return None
        while len(matches) < match_count:
            if not _place_shape(matches, blocked_a, template_shape, external_colors[len(matches)], h, w):
                return None
        blocked = blocked_a
    elif mode == "row":
        blocked_a = set(blocked)
        if not _place_shape(matches, blocked_a, template_shape, external_colors[0], h, w):
            return None
        r0, _c0 = matches[0]["anchor"]
        if not _place_shape(matches, blocked_a, template_shape, external_colors[1], h, w, fixed_row=r0):
            return None
        while len(matches) < match_count:
            if not _place_shape(matches, blocked_a, template_shape, external_colors[len(matches)], h, w):
                return None
        blocked = blocked_a
    else:
        blocked_a = set(blocked)
        if not _place_shape(matches, blocked_a, template_shape, external_colors[0], h, w):
            return None
        _r0, c0 = matches[0]["anchor"]
        if not _place_shape(matches, blocked_a, template_shape, external_colors[1], h, w, fixed_col=c0):
            return None
        while len(matches) < match_count:
            if not _place_shape(matches, blocked_a, template_shape, external_colors[len(matches)], h, w):
                return None
        blocked = blocked_a

    distractor_shapes = _nonmatching_shapes(template_shape, distractor_count)
    distractors: list[dict] = []

    # First distractor is intentionally aligned to perturb the "connect everything" shortcut.
    first_shape = distractor_shapes[0]
    aligned = False
    anchor_choices = []
    if random.random() < 0.5:
        anchor_choices.extend(("row", obj["anchor"][0]) for obj in matches)
    else:
        anchor_choices.extend(("col", obj["anchor"][1]) for obj in matches)
    random.shuffle(anchor_choices)
    for axis, value in anchor_choices:
        ok = _place_shape(
            distractors,
            blocked,
            first_shape,
            external_colors[match_count],
            h,
            w,
            fixed_row=value if axis == "row" else None,
            fixed_col=value if axis == "col" else None,
        )
        if ok:
            aligned = True
            break
    if not aligned and not _place_shape(distractors, blocked, first_shape, external_colors[match_count], h, w):
        return None

    while len(distractors) < distractor_count:
        idx = len(distractors)
        if not _place_shape(
            distractors,
            blocked,
            distractor_shapes[idx],
            external_colors[match_count + idx],
            h,
            w,
        ):
            return None

    base_match_grid = _base_grid_without_distractors(
        h, w, frame_bbox, template_shape, template_color, template_offset, matches
    )
    selected_match_coords = _selected_coords(matches)
    mode_ok, row_adds, col_adds = _row_col_mode_ok(base_match_grid, selected_match_coords, mode)
    if not mode_ok:
        return None

    full_input = _paint_scene(
        h, w, frame_bbox, template_shape, template_color, template_offset, matches, distractors
    )
    fill_match = row_adds | col_adds
    fill_all = _all_external_fill_cells(full_input, matches, distractors)
    if fill_match == fill_all:
        return None

    try:
        frame_bbox2, template_shape2, matches2, nonmatches2 = _selected_components(full_input)
    except AssertionError:
        return None
    if frame_bbox2 != frame_bbox or template_shape2 != template_shape:
        return None
    if len(matches2) != match_count or len(nonmatches2) != distractor_count:
        return None

    out = main(full_input)
    if np.array_equal(out, full_input):
        return None
    if not np.any(out == GREY):
        return None
    if np.count_nonzero(out == GREY) <= np.count_nonzero(full_input == GREY):
        return None

    # Semantic checks beyond simple self-consistency.
    if not all(int(out[r, c]) == BLACK for obj in nonmatches2 for r, c in obj["coords"]):
        return None
    if not fill_match:
        return None
    if np.array_equal(out, base_match_grid):
        return None
    return full_input


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
