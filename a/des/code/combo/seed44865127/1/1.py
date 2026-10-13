from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(comp)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_border(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _coords_to_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _trim(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _shape_variants(mask: np.ndarray) -> list[np.ndarray]:
    base = _trim(mask.astype(bool))
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    for k in range(4):
        rot = np.rot90(base, k)
        for cand in (rot, np.fliplr(rot)):
            cand = _trim(cand)
            sig = (tuple(cand.shape), tuple(cand.astype(np.uint8).ravel()))
            if sig not in seen:
                seen.add(sig)
                variants.append(cand.copy())
    return variants


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    sigs = []
    for cand in _shape_variants(mask):
        sigs.append((tuple(cand.shape), tuple(cand.astype(np.uint8).ravel())))
    return min(sigs)


def _hole_count(mask: np.ndarray) -> int:
    bg = ~mask.astype(bool)
    count = 0
    h, w = bg.shape
    for comp in _components(bg):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        count += 1
    return count


def _holes_in_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> list[list[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    sub = grid[r0 : r1 + 1, c0 : c1 + 1] == bg
    holes: list[list[tuple[int, int]]] = []
    sh, sw = sub.shape
    for comp in _components(sub):
        if any(r == 0 or c == 0 or r == sh - 1 or c == sw - 1 for r, c in comp):
            continue
        holes.append([(r0 + r, c0 + c) for r, c in comp])
    return holes


def _find_legend(
    grid: np.ndarray, bg: int, comps: list[list[tuple[int, int]]]
) -> tuple[int, list[int]]:
    h, w = grid.shape
    candidates: list[int] = []
    for idx, comp in enumerate(comps):
        if not _touches_border(comp, h, w):
            continue
        if not 2 <= len(comp) <= 4:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if c0 != c1:
            continue
        if r1 - r0 + 1 != len(comp):
            continue
        comp_set = set(comp)
        if any((r, c0) not in comp_set for r in range(r0, r1 + 1)):
            continue
        colors = [int(grid[r, c0]) for r in range(r0, r1 + 1)]
        if bg in colors or len(set(colors)) != len(colors):
            continue
        candidates.append(idx)
    assert len(candidates) == 1, "legend must be the unique border-touching vertical strip"
    idx = candidates[0]
    ordered = sorted(comps[idx], key=lambda p: p[0])
    return idx, [int(grid[r, c]) for r, c in ordered]


def _find_frame(
    grid: np.ndarray, bg: int, comps: list[list[tuple[int, int]]], skip_idx: int
) -> tuple[int, list[list[tuple[int, int]]]]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int]] = []
    for idx, comp in enumerate(comps):
        if idx == skip_idx or _touches_border(comp, h, w):
            continue
        holes = _holes_in_bbox(grid, _bbox(comp), bg)
        if holes:
            bbox_area = (max(r for r, _ in comp) - min(r for r, _ in comp) + 1) * (
                max(c for _, c in comp) - min(c for _, c in comp) + 1
            )
            candidates.append((len(comp), bbox_area, idx))
    assert candidates, "must find a central frame with enclosed holes"
    _, _, frame_idx = max(candidates)
    frame_holes = _holes_in_bbox(grid, _bbox(comps[frame_idx]), bg)
    return frame_idx, frame_holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match exterior sprites to frame holes by shape, then recolor all fills using
    the legend entry indexed by the total number of holes.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    comps = _components(g != bg)
    assert comps, "grid must contain non-background objects"

    legend_idx, legend_colors = _find_legend(g, bg, comps)
    frame_idx, holes = _find_frame(g, bg, comps, legend_idx)

    hole_sigs = Counter(_shape_signature(_coords_to_mask(hole)) for hole in holes)
    sprite_sigs = Counter(
        _shape_signature(_coords_to_mask(comp))
        for idx, comp in enumerate(comps)
        if idx not in {legend_idx, frame_idx}
    )
    assert hole_sigs == sprite_sigs, "exterior sprites must match the frame holes"

    hole_count = len(holes)
    assert 1 <= hole_count <= len(legend_colors)
    fill_color = legend_colors[hole_count - 1]

    out = np.full_like(g, bg)
    for r, c in comps[frame_idx]:
        out[r, c] = g[r, c]
    for hole in holes:
        for r, c in hole:
            out[r, c] = fill_color
    return out


def _random_shape(rng: random.Random, area_min: int, area_max: int) -> np.ndarray:
    for _ in range(300):
        target = rng.randint(area_min, area_max)
        cells = {(0, 0)}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.append(nxt)
            cells.add(rng.choice(frontier))

        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        norm = sorted((r - min_r, c - min_c) for r, c in cells)
        h = max(r for r, _ in norm) + 1
        w = max(c for _, c in norm) + 1
        if h < 2 or w < 2 or h > 5 or w > 5:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in norm:
            mask[r, c] = True
        if int(mask.sum()) == h * w:
            continue
        if _hole_count(mask) != 0:
            continue
        if len(_shape_variants(mask)) < 2:
            continue
        return mask
    raise RuntimeError("failed to generate a non-degenerate hole shape")


def _generate_shapes(rng: random.Random, count: int) -> list[np.ndarray]:
    shapes: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    area_max = 7 if count >= 3 else 8
    while len(shapes) < count:
        shape = _random_shape(rng, 3, area_max)
        sig = _shape_signature(shape)
        if sig in seen:
            continue
        seen.add(sig)
        shapes.append(shape)
    return shapes


def _place_holes(
    rng: random.Random, shapes: list[np.ndarray]
) -> tuple[int, int, list[tuple[int, int]]] | None:
    coords_per_shape = [list(map(tuple, np.argwhere(shape))) for shape in shapes]
    max_h = max(shape.shape[0] for shape in shapes)
    max_w = max(shape.shape[1] for shape in shapes)
    footprint = sum((shape.shape[0] + 2) * (shape.shape[1] + 2) for shape in shapes)
    base = int(np.ceil(np.sqrt(footprint))) + 2
    order = sorted(range(len(shapes)), key=lambda i: len(coords_per_shape[i]), reverse=True)

    for _ in range(250):
        frame_h = rng.randint(max(max_h + 4, base), max(max_h + 8, base + 4))
        frame_w = rng.randint(max(max_w + 4, base), max(max_w + 10, base + 5))
        blocked = np.zeros((frame_h, frame_w), dtype=bool)
        placements: list[tuple[int, int] | None] = [None] * len(shapes)
        ok = True

        for idx in order:
            coords = coords_per_shape[idx]
            h, w = shapes[idx].shape
            candidates: list[tuple[int, int]] = []
            for r0 in range(1, frame_h - h):
                for c0 in range(1, frame_w - w):
                    if any(blocked[r0 + rr, c0 + cc] for rr, cc in coords):
                        continue
                    candidates.append((r0, c0))
            if not candidates:
                ok = False
                break

            r0, c0 = rng.choice(candidates)
            placements[idx] = (r0, c0)
            for rr, cc in coords:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = r0 + rr + dr, c0 + cc + dc
                        if 0 <= nr < frame_h and 0 <= nc < frame_w:
                            blocked[nr, nc] = True

        if ok:
            return frame_h, frame_w, [p for p in placements if p is not None]
    return None


def _mark_padded(blocked: np.ndarray, coords: list[tuple[int, int]], pad: int = 1) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _place_sprites(
    rng: random.Random,
    grid: np.ndarray,
    blocked: np.ndarray,
    sprite_specs: list[tuple[np.ndarray, int]],
) -> bool:
    h, w = grid.shape
    for mask, color in sprite_specs:
        coords = list(map(tuple, np.argwhere(mask)))
        sh, sw = mask.shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, h - sh):
            for c0 in range(1, w - sw):
                if any(blocked[r0 + rr, c0 + cc] for rr, cc in coords):
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return False

        r0, c0 = rng.choice(candidates)
        placed = []
        for rr, cc in coords:
            r, c = r0 + rr, c0 + cc
            grid[r, c] = color
            placed.append((r, c))
        _mark_padded(blocked, placed, pad=1)
    return True


def _build_scene(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    hole_count = rng.choice([1, 2, 3, 4])
    legend_len = rng.randint(max(2, hole_count), 4)
    shapes = _generate_shapes(rng, hole_count)

    frame_layout = _place_holes(rng, shapes)
    if frame_layout is None:
        return None
    frame_h, frame_w, placements = frame_layout

    colors = list(range(10))
    rng.shuffle(colors)
    bg = colors[0]
    frame_color = colors[1]
    legend_colors = colors[2 : 2 + legend_len]
    sprite_colors = colors[2 + legend_len : 2 + legend_len + hole_count]

    frame_patch = np.full((frame_h, frame_w), frame_color, dtype=int)
    hole_coords_local: list[list[tuple[int, int]]] = []
    for shape, (r0, c0) in zip(shapes, placements):
        coords = list(map(tuple, np.argwhere(shape)))
        local: list[tuple[int, int]] = []
        for rr, cc in coords:
            frame_patch[r0 + rr, c0 + cc] = bg
            local.append((r0 + rr, c0 + cc))
        hole_coords_local.append(local)

    actual_holes = _holes_in_bbox(frame_patch, (0, 0, frame_h - 1, frame_w - 1), bg)
    if Counter(_shape_signature(_coords_to_mask(hole)) for hole in actual_holes) != Counter(
        _shape_signature(shape) for shape in shapes
    ):
        return None

    grid_h = frame_h + rng.randint(8 + hole_count, 12 + hole_count)
    grid_w = frame_w + rng.randint(12 + 2 * hole_count, 18 + 2 * hole_count)
    legend_side = rng.choice(["left", "right"])
    left_gap = 3 if legend_side == "left" else 2
    right_gap = 3 if legend_side == "right" else 2
    top_gap = 2
    bottom_gap = 2

    if grid_h <= frame_h + top_gap + bottom_gap or grid_w <= frame_w + left_gap + right_gap:
        return None

    frame_r0 = rng.randint(top_gap, grid_h - frame_h - bottom_gap)
    frame_c0 = rng.randint(left_gap, grid_w - frame_w - right_gap)

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    grid[frame_r0 : frame_r0 + frame_h, frame_c0 : frame_c0 + frame_w] = frame_patch

    legend_col = 0 if legend_side == "left" else grid_w - 1
    legend_r0 = rng.randint(0, grid_h - legend_len)
    legend_cells = []
    for i, color in enumerate(legend_colors):
        cell = (legend_r0 + i, legend_col)
        grid[cell] = color
        legend_cells.append(cell)

    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    blocked[max(0, frame_r0 - 1) : min(grid_h, frame_r0 + frame_h + 1), max(0, frame_c0 - 1) : min(grid_w, frame_c0 + frame_w + 1)] = True
    _mark_padded(blocked, legend_cells, pad=1)

    sprite_specs: list[tuple[np.ndarray, int]] = []
    for shape, color in zip(shapes, sprite_colors):
        variant = rng.choice(_shape_variants(shape))
        sprite_specs.append((variant, color))
    rng.shuffle(sprite_specs)
    sprite_specs.sort(key=lambda item: int(item[0].sum()), reverse=True)

    if not _place_sprites(rng, grid, blocked, sprite_specs):
        return None

    if int(np.sum(grid == bg)) <= int(np.sum(grid == frame_color)):
        return None

    expected = np.full_like(grid, bg)
    expected[frame_r0 : frame_r0 + frame_h, frame_c0 : frame_c0 + frame_w] = frame_patch
    fill_color = legend_colors[hole_count - 1]
    for hole in hole_coords_local:
        for rr, cc in hole:
            expected[frame_r0 + rr, frame_c0 + cc] = fill_color

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None

    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        scene = _build_scene(rng)
        if scene is not None:
            return scene[0]
    raise RuntimeError("failed to generate a valid ARC input")
