import random
from typing import Iterable

import numpy as np


BG = 0
FRAME = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
TRANSFORMS = [(False, rot) for rot in range(4)] + [(True, rot) for rot in range(4)]
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
EDGE_CHOICES = ["top", "bottom", "left", "right"]
GEN_RETRIES = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            out.append(cells)
    return out


def _bbox(cells: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    cells = list(cells)
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _normalize(cells: Iterable[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    cells = list(cells)
    r0, c0, _, _ = _bbox(cells)
    norm = tuple(sorted((r - r0, c - c0) for r, c in cells))
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    return norm, int(h), int(w)


def _coords_to_mask(coords: Iterable[tuple[int, int]]) -> np.ndarray:
    norm, h, w = _normalize(coords)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _apply_transform(coords: Iterable[tuple[int, int]], flip: bool, rot: int) -> tuple[tuple[int, int], ...]:
    arr = _coords_to_mask(coords)
    if flip:
        arr = np.fliplr(arr)
    if rot:
        arr = np.rot90(arr, -rot)
    pts = [tuple(map(int, xy)) for xy in np.argwhere(arr)]
    norm, _, _ = _normalize(pts)
    return norm


def _transform_dir(direction: tuple[int, int], flip: bool, rot: int) -> tuple[int, int]:
    dr, dc = direction
    if flip:
        dc = -dc
    for _ in range(rot):
        dr, dc = dc, -dr
    return int(dr), int(dc)


def _canonical_signature(coords: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    norm, _, _ = _normalize(coords)
    return min(_apply_transform(norm, flip, rot) for flip, rot in TRANSFORMS)


def _find_frames(g: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(g == FRAME):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = {
            *((r0, c) for c in range(c0, c1 + 1)),
            *((r1, c) for c in range(c0, c1 + 1)),
            *((r, c0) for r in range(r0, r1 + 1)),
            *((r, c1) for r in range(r0, r1 + 1)),
        }
        if set(comp) != border:
            continue
        frames.append((int(r0), int(c0), int(r1), int(c1)))
    frames.sort()
    assert frames, "expected at least one grey frame"
    return frames


def _extract_fragment(g: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> dict:
    r0, c0, r1, c1 = frame_bbox
    sub = g[r0 + 1 : r1, c0 + 1 : c1]
    comps = _components(sub != BG)
    assert len(comps) == 1, "each frame interior must contain exactly one fragment"
    rel_cells = comps[0]
    colors = {int(sub[r, c]) for r, c in rel_cells}
    assert len(colors) == 1, "fragment must be monochrome"
    color = next(iter(colors))
    abs_cells = [(r0 + 1 + r, c0 + 1 + c) for r, c in rel_cells]
    norm, _, _ = _normalize(rel_cells)
    return {"bbox": frame_bbox, "color": int(color), "cells": abs_cells, "shape": norm}


def _marker_direction(shape_cells: list[tuple[int, int]], marker: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(shape_cells)
    mr, mc = marker
    if mr == r0 - 1 and c0 <= mc <= c1:
        return (-1, 0)
    if mr == r1 + 1 and c0 <= mc <= c1:
        return (1, 0)
    if mc == c0 - 1 and r0 <= mr <= r1:
        return (0, -1)
    if mc == c1 + 1 and r0 <= mr <= r1:
        return (0, 1)
    raise AssertionError("marker must lie immediately outside one side of the template shape")


def _find_templates(g: np.ndarray, frames: list[tuple[int, int, int, int]]) -> list[dict]:
    h, w = g.shape
    frame_mask = np.zeros((h, w), dtype=bool)
    for r0, c0, r1, c1 in frames:
        frame_mask[r0 : r1 + 1, c0 : c1 + 1] |= g[r0 : r1 + 1, c0 : c1 + 1] == FRAME

    templates: list[dict] = []
    for comp in _components((g != BG) & (~frame_mask)):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        colors = {int(g[r, c]) for r, c in comp}
        assert FRAME not in colors, "template components must not use grey"
        assert len(colors) == 2, "each template must contain one shape color and one marker color"
        counts = {color: 0 for color in colors}
        for r, c in comp:
            counts[int(g[r, c])] += 1
        marker_colors = [color for color, cnt in counts.items() if cnt == 1]
        assert len(marker_colors) == 1, "template marker must be a singleton color"
        marker_color = marker_colors[0]
        shape_color = next(color for color in colors if color != marker_color)
        marker = next((r, c) for r, c in comp if int(g[r, c]) == marker_color)
        shape_cells = [(r, c) for r, c in comp if int(g[r, c]) == shape_color]
        assert any(abs(marker[0] - r) + abs(marker[1] - c) == 1 for r, c in shape_cells), "marker must touch shape"
        direction = _marker_direction(shape_cells, marker)
        shape_norm, _, _ = _normalize(shape_cells)
        templates.append(
            {
                "shape": shape_norm,
                "shape_color": int(shape_color),
                "marker_color": int(marker_color),
                "marker_dir": direction,
            }
        )
    assert templates, "expected at least one edge template"
    return templates


def _match_template(fragment_shape: tuple[tuple[int, int], ...], templates: list[dict]) -> tuple[dict, tuple[int, int]]:
    candidates: list[tuple[dict, tuple[int, int]]] = []
    for template in templates:
        mapped_dirs = {
            _transform_dir(template["marker_dir"], flip, rot)
            for flip, rot in TRANSFORMS
            if _apply_transform(template["shape"], flip, rot) == fragment_shape
        }
        if mapped_dirs:
            assert len(mapped_dirs) == 1, "shape match must imply a unique border direction"
            candidates.append((template, next(iter(mapped_dirs))))
    assert len(candidates) == 1, "each fragment must match exactly one template"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each framed fragment to an edge template, then recolor the fragment and one frame side from the matched template orientation.

    Input structure:
    - The background is black.
    - Several grey rectangular 1-cell-thick frames are scattered in the grid; each frame encloses black space plus one monochrome irregular fragment.
    - Along one outer grid edge there is a strip of small template components; each template contains a fragment-shaped monochrome pattern and one adjacent singleton direction marker in a different color.

    Rule:
    - For each grey frame, extract the interior fragment shape and find the unique edge template with the same shape up to rotation or reflection.
    - Determine which rotation/reflection maps the template shape onto the framed fragment, and apply that same transform to the template marker's relative side.
    - Recolor the fragment to the template shape color.
    - Recolor the entire corresponding frame side (top, bottom, left, or right) from grey to that same template shape color, leaving the other three sides grey.

    Key insight:
    - The matched template contributes both the target color and an orientation-coded side, because the marker direction must be transformed together with the template shape.
    """
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g)
    templates = _find_templates(g, frames)

    out = g.copy()
    for frame in frames:
        fragment = _extract_fragment(g, frame)
        template, direction = _match_template(fragment["shape"], templates)
        color = int(template["shape_color"])
        for r, c in fragment["cells"]:
            out[r, c] = color
        r0, c0, r1, c1 = frame
        if direction == (-1, 0):
            out[r0, c0 : c1 + 1] = color
        elif direction == (1, 0):
            out[r1, c0 : c1 + 1] = color
        elif direction == (0, -1):
            out[r0 : r1 + 1, c0] = color
        else:
            assert direction == (0, 1)
            out[r0 : r1 + 1, c1] = color
    return out


def _random_shape() -> tuple[tuple[int, int], ...] | None:
    for _ in range(200):
        target = random.randint(4, 6)
        cells = {(0, 0)}
        frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
        while len(cells) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for dr, dc in DIRS4:
                nxt = (cell[0] + dr, cell[1] + dc)
                if nxt not in cells:
                    frontier.add(nxt)
        norm, h, w = _normalize(cells)
        if h < 2 or w < 2 or h > 4 or w > 4:
            continue
        if len({_apply_transform(norm, flip, rot) for flip, rot in TRANSFORMS}) != 8:
            continue
        return norm
    return None


def _shape_dims(coords: Iterable[tuple[int, int]]) -> tuple[int, int]:
    _, h, w = _normalize(coords)
    return h, w


def _make_template_patch(
    shape: tuple[tuple[int, int], ...], shape_color: int, marker_color: int, flip: bool, rot: int
) -> np.ndarray:
    _, h, w = _normalize(shape)
    right_rows = sorted(r for r, c in shape if c == w - 1)
    anchor_row = random.choice(right_rows)
    patch = np.zeros((h, w + 1), dtype=int)
    for r, c in shape:
        patch[r, c] = shape_color
    patch[anchor_row, w] = marker_color
    if flip:
        patch = np.fliplr(patch)
    if rot:
        patch = np.rot90(patch, -rot)
    return np.asarray(patch, dtype=int)


def _distribute_slots(total: int, widths: list[int]) -> list[int] | None:
    need = sum(widths) + (len(widths) - 1)
    if need > total:
        return None
    extra = total - need
    gaps = [0] * (len(widths) + 1)
    for _ in range(extra):
        gaps[random.randrange(len(gaps))] += 1
    pos = gaps[0]
    starts: list[int] = []
    for i, width in enumerate(widths):
        starts.append(pos)
        pos += width
        if i + 1 < len(widths):
            pos += 1 + gaps[i + 1]
    return starts


def _place_templates(
    grid: np.ndarray, edge: str, patches: list[np.ndarray]
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    order = list(range(len(patches)))
    random.shuffle(order)
    positions = [(-1, -1)] * len(patches)
    if edge in {"top", "bottom"}:
        widths = [int(patches[i].shape[1]) for i in order]
        starts = _distribute_slots(w, widths)
        if starts is None:
            return None
        for idx, c in zip(order, starts, strict=False):
            patch = patches[idx]
            r = 0 if edge == "top" else h - patch.shape[0]
            positions[idx] = (int(r), int(c))
    else:
        heights = [int(patches[i].shape[0]) for i in order]
        starts = _distribute_slots(h, heights)
        if starts is None:
            return None
        for idx, r in zip(order, starts, strict=False):
            patch = patches[idx]
            c = 0 if edge == "left" else w - patch.shape[1]
            positions[idx] = (int(r), int(c))
    return positions


def _body_bounds(shape: tuple[int, int], edge: str, strip: int) -> tuple[int, int, int, int]:
    h, w = shape
    if edge == "top":
        return strip + 1, 1, h - 2, w - 2
    if edge == "bottom":
        return 1, 1, h - strip - 2, w - 2
    if edge == "left":
        return 1, strip + 1, h - 2, w - 2
    return 1, 1, h - 2, w - strip - 2


def _place_frames(
    grid_shape: tuple[int, int], edge: str, strip: int, frame_sizes: list[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    bounds = _body_bounds(grid_shape, edge, strip)
    rmin, cmin, rmax, cmax = bounds
    if rmin > rmax or cmin > cmax:
        return None
    order = sorted(range(len(frame_sizes)), key=lambda i: frame_sizes[i][0] * frame_sizes[i][1], reverse=True)
    placed = [(-1, -1)] * len(frame_sizes)
    occupied = np.zeros(grid_shape, dtype=bool)
    if edge == "top":
        occupied[: strip + 1, :] = True
    elif edge == "bottom":
        occupied[grid_shape[0] - strip - 1 :, :] = True
    elif edge == "left":
        occupied[:, : strip + 1] = True
    else:
        occupied[:, grid_shape[1] - strip - 1 :] = True

    for idx in order:
        fh, fw = frame_sizes[idx]
        placed_ok = False
        rr_hi = rmax - fh + 1
        cc_hi = cmax - fw + 1
        if rr_hi < rmin or cc_hi < cmin:
            return None
        for _ in range(250):
            r0 = random.randint(rmin, rr_hi)
            c0 = random.randint(cmin, cc_hi)
            pr0 = max(0, r0 - 1)
            pc0 = max(0, c0 - 1)
            pr1 = min(grid_shape[0] - 1, r0 + fh)
            pc1 = min(grid_shape[1] - 1, c0 + fw)
            if np.any(occupied[pr0 : pr1 + 1, pc0 : pc1 + 1]):
                continue
            occupied[pr0 : pr1 + 1, pc0 : pc1 + 1] = True
            placed[idx] = (int(r0), int(c0))
            placed_ok = True
            break
        if not placed_ok:
            return None
    return placed


def _render_case(
    grid_shape: tuple[int, int],
    edge: str,
    templates: list[dict],
    template_positions: list[tuple[int, int]],
    frames: list[dict],
    frame_positions: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.zeros(grid_shape, dtype=int)
    for tpl, (r0, c0) in zip(templates, template_positions, strict=False):
        patch = tpl["patch"]
        ph, pw = patch.shape
        view = grid[r0 : r0 + ph, c0 : c0 + pw]
        mask = patch != BG
        assert not np.any(view[mask] != BG)
        view[mask] = patch[mask]

    expected = grid.copy()
    for frame, (r0, c0) in zip(frames, frame_positions, strict=False):
        fh, fw = frame["outer_size"]
        grid[r0, c0 : c0 + fw] = FRAME
        grid[r0 + fh - 1, c0 : c0 + fw] = FRAME
        grid[r0 : r0 + fh, c0] = FRAME
        grid[r0 : r0 + fh, c0 + fw - 1] = FRAME
        expected[r0, c0 : c0 + fw] = FRAME
        expected[r0 + fh - 1, c0 : c0 + fw] = FRAME
        expected[r0 : r0 + fh, c0] = FRAME
        expected[r0 : r0 + fh, c0 + fw - 1] = FRAME

        fr_off, fc_off = frame["fragment_offset"]
        for dr, dc in frame["fragment_shape"]:
            rr = r0 + fr_off + dr
            cc = c0 + fc_off + dc
            grid[rr, cc] = frame["fragment_color"]
            expected[rr, cc] = frame["template_color"]

        color = frame["template_color"]
        direction = frame["final_dir"]
        if direction == (-1, 0):
            expected[r0, c0 : c0 + fw] = color
        elif direction == (1, 0):
            expected[r0 + fh - 1, c0 : c0 + fw] = color
        elif direction == (0, -1):
            expected[r0 : r0 + fh, c0] = color
        else:
            expected[r0 : r0 + fh, c0 + fw - 1] = color
    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(2, 4)
    used_signatures: set[tuple[tuple[int, int], ...]] = set()
    base_shapes: list[tuple[tuple[int, int], ...]] = []
    while len(base_shapes) < n:
        shape = _random_shape()
        if shape is None:
            return None
        sig = _canonical_signature(shape)
        if sig in used_signatures:
            continue
        used_signatures.add(sig)
        base_shapes.append(shape)

    color_order = COLOR_POOL[:]
    random.shuffle(color_order)
    shape_colors = color_order[:n]
    marker_colors = color_order[n : 2 * n]

    templates: list[dict] = []
    frames: list[dict] = []
    patch_sizes: list[tuple[int, int]] = []
    frame_sizes: list[tuple[int, int]] = []
    for i, shape in enumerate(base_shapes):
        tpl_flip, tpl_rot = random.choice(TRANSFORMS)
        frag_flip, frag_rot = random.choice(TRANSFORMS)
        patch = _make_template_patch(shape, shape_colors[i], marker_colors[i], tpl_flip, tpl_rot)
        patch_sizes.append(tuple(map(int, patch.shape)))
        fragment_shape = _apply_transform(shape, frag_flip, frag_rot)
        frag_h, frag_w = _shape_dims(fragment_shape)
        pad_top = random.randint(1, 2)
        pad_bottom = random.randint(1, 2)
        pad_left = random.randint(1, 2)
        pad_right = random.randint(1, 2)
        outer_h = frag_h + pad_top + pad_bottom + 2
        outer_w = frag_w + pad_left + pad_right + 2
        frame_sizes.append((int(outer_h), int(outer_w)))
        fragment_color = random.choice([c for c in COLOR_POOL if c != shape_colors[i]])
        templates.append({"patch": patch})
        frames.append(
            {
                "fragment_shape": fragment_shape,
                "fragment_offset": (1 + pad_top, 1 + pad_left),
                "fragment_color": int(fragment_color),
                "template_color": int(shape_colors[i]),
                "final_dir": _transform_dir((0, 1), frag_flip, frag_rot),
                "outer_size": (int(outer_h), int(outer_w)),
            }
        )

    edge = random.choice(EDGE_CHOICES)
    strip = max(size[0] for size in patch_sizes) if edge in {"top", "bottom"} else max(size[1] for size in patch_sizes)
    min_h = max(18, max(fh for fh, _ in frame_sizes) + 4)
    min_w = max(18, max(fw for _, fw in frame_sizes) + 4)
    if edge in {"top", "bottom"}:
        min_h = max(min_h, strip + 6)
        min_w = max(min_w, sum(pw for _, pw in patch_sizes) + len(patch_sizes) - 1)
    else:
        min_w = max(min_w, strip + 6)
        min_h = max(min_h, sum(ph for ph, _ in patch_sizes) + len(patch_sizes) - 1)
    if min_h > 30 or min_w > 30:
        return None

    grid_h = random.randint(min_h, 30)
    grid_w = random.randint(min_w, 30)
    grid_shape = (grid_h, grid_w)
    template_positions = _place_templates(np.zeros(grid_shape, dtype=int), edge, [tpl["patch"] for tpl in templates])
    if template_positions is None:
        return None
    frame_positions = _place_frames(grid_shape, edge, strip, frame_sizes)
    if frame_positions is None:
        return None

    grid, expected = _render_case(grid_shape, edge, templates, template_positions, frames, frame_positions)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
