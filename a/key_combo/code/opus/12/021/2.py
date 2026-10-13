from __future__ import annotations

import random
import numpy as np


Coord = tuple[int, int]


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize_shape(coords: tuple[Coord, ...]) -> tuple[Coord, ...]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _shape_variants(base: tuple[Coord, ...]) -> set[tuple[Coord, ...]]:
    variants: set[tuple[Coord, ...]] = set()
    current = tuple(base)
    for _ in range(4):
        rotated = tuple((c, -r) for r, c in current)
        current = rotated
        for reflected in (current, tuple((r, -c) for r, c in current)):
            norm = _normalize_shape(reflected)
            h = max(r for r, _ in norm) + 1
            w = max(c for _, c in norm) + 1
            if h <= 3 and w <= 3:
                variants.add(norm)
    return variants


def _build_shape_bank() -> list[tuple[Coord, ...]]:
    bases = [
        ((0, 0), (0, 1)),
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 1), (0, 2), (1, 0), (1, 1)),
    ]
    shapes: set[tuple[Coord, ...]] = set()
    for base in bases:
        shapes.update(_shape_variants(base))
    return sorted(shapes)


SHAPES = _build_shape_bank()


def _legend_pairs(grid: np.ndarray) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        left = int(grid[r, 0])
        right = int(grid[r, 1])
        if left == 0 or right == 0:
            break
        pairs.append((left, right))
    assert pairs, "expected a non-empty legend"
    assert len({src for src, _ in pairs}) == len(pairs), "legend source colors must be unique"
    return pairs


def _min_manhattan(point: Coord, coords: np.ndarray) -> int:
    pr, pc = point
    return min(abs(pr - int(r)) + abs(pc - int(c)) for r, c in coords)


def _match_markers_to_sprites(
    markers: list[Coord], sprites: list[np.ndarray]
) -> list[tuple[Coord, np.ndarray]]:
    if not markers and not sprites:
        return []
    assert len(markers) == len(sprites) > 0, "each used source color needs one marker per sprite"

    candidates: list[list[int]] = []
    for marker in markers:
        opts = [
            idx
            for idx, sprite in enumerate(sprites)
            if 1 <= _min_manhattan(marker, sprite) <= 2
        ]
        assert opts, "marker must be near at least one same-color sprite"
        candidates.append(opts)

    solutions: list[list[int]] = []
    used: set[int] = set()

    def dfs(i: int, assign: list[int]) -> None:
        if len(solutions) > 1:
            return
        if i == len(markers):
            solutions.append(assign[:])
            return
        for idx in candidates[i]:
            if idx in used:
                continue
            used.add(idx)
            assign.append(idx)
            dfs(i + 1, assign)
            assign.pop()
            used.remove(idx)

    dfs(0, [])
    assert len(solutions) == 1, "marker-sprite pairing must be unique"
    return [(markers[i], sprites[sprite_idx]) for i, sprite_idx in enumerate(solutions[0])]


def _paint_stamp(
    out: np.ndarray, sprite: np.ndarray, marker: Coord, scale: int, color: int
) -> None:
    h, w = out.shape
    r0, c0, _r1, _c1 = _bbox_coords(sprite)
    for r, c in sprite:
        rel_r = int(r) - r0
        rel_c = int(c) - c0
        for dr in range(scale):
            for dc in range(scale):
                rr = marker[0] + rel_r * scale + dr
                cc = marker[1] + rel_c * scale + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the two-column legend, scale and recolor each small sprite, then stamp the enlarged copies by legend priority.

    Input structure:
    - The grid has black background and a contiguous two-column legend in columns 0 and 1 starting at row 0.
    - Each legend row maps one source color to one target color; the number of legend rows is also the global scale factor.
    - Outside the legend are 2-4 monochrome source-color sprites plus detached same-color single-pixel markers near them.

    Rule:
    - Parse the legend to obtain source->target recoloring and the global scale factor k.
    - For each sprite, use its bounding box as the template, enlarge every occupied cell to a k x k block in the mapped target color, and place the enlarged sprite so the nearby marker cell is the enlarged bbox top-left.
    - Clear the legend, original sprites, and markers; if enlarged sprites overlap, earlier legend rows have higher priority and overwrite later ones.

    Key insight:
    - The legend carries three instructions at once: row count gives the scale, each row gives the recolor mapping, and row order gives the overlap priority.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    pairs = _legend_pairs(g)
    scale = len(pairs)
    legend_mask = np.zeros_like(g, dtype=bool)
    legend_mask[:scale, :2] = True

    outside = g.copy()
    outside[legend_mask] = 0
    outside_colors = {int(v) for v in np.unique(outside)} - {0}
    source_colors = {src for src, _ in pairs}
    assert outside_colors.issubset(source_colors), "outside the legend only source colors may appear"

    objects: list[tuple[int, int, Coord, np.ndarray]] = []
    for priority, (source, target) in enumerate(pairs):
        components = _components(outside == source)
        if not components:
            continue
        markers: list[Coord] = []
        sprites: list[np.ndarray] = []
        for comp in components:
            area = int(comp.shape[0])
            if area == 1:
                markers.append(tuple(map(int, comp[0])))
                continue
            assert 2 <= area <= 4, "sprite area must be 2..4"
            r0, c0, r1, c1 = _bbox_coords(comp)
            assert r1 - r0 + 1 <= 3 and c1 - c0 + 1 <= 3, "sprite bbox must fit within 3x3"
            sprites.append(comp)
        for marker, sprite in _match_markers_to_sprites(markers, sprites):
            objects.append((priority, target, marker, sprite))

    assert 2 <= len(objects) <= 4, "expected 2..4 sprites"
    out = np.zeros_like(g)
    for priority, target, marker, sprite in sorted(objects, key=lambda item: item[0], reverse=True):
        _paint_stamp(out, sprite, marker, scale, target)
    return out


def _cell_is_free(occupied: np.ndarray, cell: Coord) -> bool:
    r, c = cell
    h, w = occupied.shape
    if not (0 <= r < h and 0 <= c < w):
        return False
    if occupied[r, c]:
        return False
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
            return False
    return True


def _component_fits(occupied: np.ndarray, cells: set[Coord]) -> bool:
    if not cells:
        return False
    h, w = occupied.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if c < 3:
            return False
        if occupied[r, c]:
            return False
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc] and (nr, nc) not in cells:
                return False
    return True


def _candidate_markers(sprite_cells: set[Coord], h: int, w: int) -> list[Coord]:
    candidates: set[Coord] = set()
    for r, c in sprite_cells:
        for dr in range(-2, 3):
            dc = 2 - abs(dr)
            for sign in (-1, 1):
                rr = r + dr
                cc = c + sign * dc
                if dc == 0 and sign == 1:
                    continue
                candidates.add((rr, cc))
    out = []
    for marker in candidates:
        mr, mc = marker
        if 0 <= mr < h and 3 <= mc < w:
            dist = min(abs(mr - r) + abs(mc - c) for r, c in sprite_cells)
            if dist == 2:
                out.append(marker)
    return out


def _scaled_cells(shape: tuple[Coord, ...], top_left: Coord, scale: int) -> set[Coord]:
    cells: set[Coord] = set()
    for rel_r, rel_c in shape:
        for dr in range(scale):
            for dc in range(scale):
                cells.add((top_left[0] + rel_r * scale + dr, top_left[1] + rel_c * scale + dc))
    return cells


def _visible_cells(cells: set[Coord], h: int, w: int) -> set[Coord]:
    return {(r, c) for r, c in cells if 0 <= r < h and 0 <= c < w}


def _clipped(cells: set[Coord], h: int, w: int) -> bool:
    return any(not (0 <= r < h and 0 <= c < w) for r, c in cells)


def _shape_bounds(shape: tuple[Coord, ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _enumerate_candidates(
    h: int,
    w: int,
    shape: tuple[Coord, ...],
    scale: int,
    occupied: np.ndarray,
) -> list[dict]:
    sh, sw = _shape_bounds(shape)
    candidates: list[dict] = []
    for top_r in range(h - sh + 1):
        for top_c in range(3, w - sw + 1):
            sprite_cells = {(top_r + r, top_c + c) for r, c in shape}
            if not _component_fits(occupied, sprite_cells):
                continue
            occ_with_sprite = occupied.copy()
            for r, c in sprite_cells:
                occ_with_sprite[r, c] = True
            markers = _candidate_markers(sprite_cells, h, w)
            random.shuffle(markers)
            for marker in markers:
                if not _cell_is_free(occ_with_sprite, marker):
                    continue
                full_stamp = _scaled_cells(shape, marker, scale)
                visible = _visible_cells(full_stamp, h, w)
                if not visible:
                    continue
                candidates.append(
                    {
                        "sprite_cells": sprite_cells,
                        "marker": marker,
                        "full_stamp": full_stamp,
                        "visible_stamp": visible,
                        "clipped": _clipped(full_stamp, h, w),
                    }
                )
    random.shuffle(candidates)
    return candidates


def _paint_input_object(grid: np.ndarray, color: int, sprite_cells: set[Coord], marker: Coord) -> None:
    for r, c in sprite_cells:
        grid[r, c] = color
    grid[marker] = color


def _render_expected(
    shape_by_color: dict[int, tuple[Coord, ...]],
    placements: list[dict],
    pairs: list[tuple[int, int]],
    h: int,
    w: int,
) -> np.ndarray:
    scale = len(pairs)
    out = np.zeros((h, w), dtype=int)
    by_source = {source: target for source, target in pairs}
    priority = {source: idx for idx, (source, _target) in enumerate(pairs)}
    ordered = sorted(placements, key=lambda item: priority[item["source"]], reverse=True)
    for item in ordered:
        shape = shape_by_color[item["source"]]
        target = by_source[item["source"]]
        for rel_r, rel_c in shape:
            for dr in range(scale):
                for dc in range(scale):
                    rr = item["marker"][0] + rel_r * scale + dr
                    cc = item["marker"][1] + rel_c * scale + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        out[rr, cc] = target
    return out


def _overlap_pairs(placements: list[dict]) -> list[tuple[int, int, int]]:
    overlaps: list[tuple[int, int, int]] = []
    for i in range(len(placements)):
        for j in range(i + 1, len(placements)):
            count = len(placements[i]["visible_stamp"] & placements[j]["visible_stamp"])
            if count:
                overlaps.append((i, j, count))
    return overlaps


def _valid_overlap_size(a: dict, b: dict) -> bool:
    overlap = len(a["visible_stamp"] & b["visible_stamp"])
    if overlap == 0:
        return False
    smaller = min(len(a["visible_stamp"]), len(b["visible_stamp"]))
    return overlap <= max(6, smaller // 3)


def _generate_scene() -> tuple[np.ndarray, np.ndarray]:
    h = random.randint(18, 26)
    w = random.randint(18, 26)
    scale = random.randint(2, 4)

    colors = list(range(1, 10))
    random.shuffle(colors)
    source_colors = colors[:scale]
    target_colors = colors[scale : 2 * scale]
    pairs = list(zip(source_colors, target_colors))

    if scale == 2:
        n_objects = 2
    elif scale == 3:
        n_objects = random.choices([2, 3], weights=[3, 2])[0]
    else:
        n_objects = random.choices([2, 3, 4], weights=[3, 3, 1])[0]
    used_sources = random.sample(source_colors, n_objects)
    random.shuffle(used_sources)
    shape_by_color = {source: random.choice(SHAPES) for source in used_sources}

    want_overlap = n_objects >= 2 and random.random() < 0.6
    want_clip = random.random() < 0.4
    clip_index = random.randrange(n_objects) if want_clip else -1

    grid = np.zeros((h, w), dtype=int)
    for row, (source, target) in enumerate(pairs):
        grid[row, 0] = source
        grid[row, 1] = target

    occupied = np.zeros((h, w), dtype=bool)
    occupied[:scale, :3] = True
    placements: list[dict] = []

    for idx, source in enumerate(used_sources):
        candidates = _enumerate_candidates(h, w, shape_by_color[source], scale, occupied)
        if want_clip and idx == clip_index:
            candidates = [cand for cand in candidates if cand["clipped"]]
        if want_overlap and idx == 1:
            candidates = [
                cand
                for cand in candidates
                if any(_valid_overlap_size(cand, prev) for prev in placements)
            ]
        if not candidates:
            raise ValueError("no candidate placement")

        chosen = random.choice(candidates[: min(len(candidates), 80)])
        chosen = dict(chosen)
        chosen["source"] = source
        placements.append(chosen)
        _paint_input_object(grid, source, chosen["sprite_cells"], chosen["marker"])
        for r, c in chosen["sprite_cells"]:
            occupied[r, c] = True
        occupied[chosen["marker"]] = True

    overlaps = _overlap_pairs(placements)
    if want_overlap and not overlaps:
        raise ValueError("expected an overlap")
    if not want_overlap and overlaps and random.random() < 0.7:
        raise ValueError("wanted a cleaner non-overlap scene")
    if want_clip and not any(item["clipped"] for item in placements):
        raise ValueError("expected clipping")

    expected = _render_expected(shape_by_color, placements, pairs, h, w)
    if not np.array_equal(main(grid), expected):
        raise ValueError("oracle mismatch")

    # Semantic checks beyond self-consistency.
    used_set = set(used_sources)
    if scale > n_objects and not (set(source_colors) - used_set):
        raise ValueError("unused legend rows were expected")
    if len({item["source"] for item in placements}) != n_objects:
        raise ValueError("object colors must stay distinguishable")
    if len({item["marker"] for item in placements}) != n_objects:
        raise ValueError("markers must be unique")
    if expected.sum() == 0:
        raise ValueError("output must be non-empty")
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            grid, _expected = _generate_scene()
            return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid legend-scale stamping scene")
