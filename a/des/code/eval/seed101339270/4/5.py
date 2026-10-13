from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG = 0
RED = 2
PALETTE = [1, 3, 4, 5, 6, 7, 8, 9]
DIRECTIONS = ("U", "D", "L", "R")

GRID_H_RANGE = (18, 36)
GRID_W_RANGE = (18, 36)
GROUP_COUNT_CHOICES = (1, 2, 3)
LAYERS_CHOICES = (2, 3)
MAX_LINE_LEN = 4
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 300


@dataclass
class Ring:
    bbox: tuple[int, int, int, int]
    color: int
    marker: tuple[int, int]
    direction: str
    parent: int | None = None
    child: int | None = None


def _dominant_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    if values.size == 0:
        return BG
    if np.any(values == BG):
        return BG
    return int(np.bincount(values).argmax())


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
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
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _border_coords(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _marker_direction(bbox: tuple[int, int, int, int], marker: tuple[int, int]) -> str | None:
    r0, c0, r1, c1 = bbox
    mr, mc = marker
    if mr == r0 and c0 < mc < c1:
        return "U"
    if mr == r1 and c0 < mc < c1:
        return "D"
    if mc == c0 and r0 < mr < r1:
        return "L"
    if mc == c1 and r0 < mr < r1:
        return "R"
    return None


def _detect_rings(grid: np.ndarray, bg: int) -> list[Ring]:
    g = np.asarray(grid, dtype=int)
    rings: list[Ring] = []
    colors = [int(v) for v in np.unique(g) if int(v) not in (bg, RED)]
    for color in colors:
        for coords in _components(g == color):
            if coords.size == 0:
                continue
            bbox_ = _bbox(coords)
            r0, c0, r1, c1 = bbox_
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue

            border = set(_border_coords(bbox_))
            coord_set = {tuple(map(int, cell)) for cell in coords}
            if not coord_set.issubset(border):
                continue

            missing = border - coord_set
            if len(missing) != 1:
                continue

            marker = next(iter(missing))
            if g[marker] != RED:
                continue

            expected_area = len(border) - 1
            if len(coord_set) != expected_area:
                continue

            direction = _marker_direction(bbox_, marker)
            if direction is None:
                continue

            rings.append(
                Ring(
                    bbox=bbox_,
                    color=color,
                    marker=marker,
                    direction=direction,
                )
            )

    by_bbox = {ring.bbox: idx for idx, ring in enumerate(rings)}
    for idx, ring in enumerate(rings):
        r0, c0, r1, c1 = ring.bbox
        inner_bbox = (r0 + 2, c0 + 2, r1 - 2, c1 - 2)
        outer_bbox = (r0 - 2, c0 - 2, r1 + 2, c1 + 2)
        if inner_bbox in by_bbox:
            rings[idx].child = by_bbox[inner_bbox]
        if outer_bbox in by_bbox:
            rings[idx].parent = by_bbox[outer_bbox]
    return rings


def _extract_pattern(
    grid: np.ndarray, ring: Ring, bg: int
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = ring.bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    coords = np.argwhere(interior != bg)
    assert coords.size > 0, "innermost ring must contain a pattern"
    pr0 = int(coords[:, 0].min()) + r0 + 1
    pc0 = int(coords[:, 1].min()) + c0 + 1
    rel = [(int(rr) + r0 + 1 - pr0, int(cc) + c0 + 1 - pc0) for rr, cc in coords]
    ph = int(coords[:, 0].max() - coords[:, 0].min() + 1)
    pw = int(coords[:, 1].max() - coords[:, 1].min() + 1)
    return rel, (ph, pw), (pr0, pc0)


def _target_top_left(
    ring: Ring,
    shape: tuple[int, int],
) -> tuple[int, int]:
    ph, pw = shape
    r0, c0, r1, c1 = ring.bbox
    mr, mc = ring.marker
    if ring.direction == "U":
        return r0 - ph, mc
    if ring.direction == "D":
        return r1 + 1, mc
    if ring.direction == "L":
        return mr, c0 - pw
    if ring.direction == "R":
        return mr, c1 + 1
    raise AssertionError(f"unknown direction {ring.direction}")


def _stamp_shape(
    out: np.ndarray,
    rel: list[tuple[int, int]],
    top_left: tuple[int, int],
    color: int,
) -> None:
    base_r, base_c = top_left
    h, w = out.shape
    for dr, dc in rel:
        rr = base_r + dr
        cc = base_c + dc
        assert 0 <= rr < h and 0 <= cc < w
        out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the innermost motif through each concentric ring, following the red markers from inner to outer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _dominant_color(g)
    rings = _detect_rings(g, bg)
    assert rings, "expected at least one marked rectangular ring"

    out = g.copy()
    innermost_indices = [idx for idx, ring in enumerate(rings) if ring.child is None]
    assert innermost_indices, "expected at least one innermost ring"

    for inner_idx in innermost_indices:
        chain: list[Ring] = []
        cur = inner_idx
        while cur is not None:
            chain.append(rings[cur])
            cur = rings[cur].parent

        rel, shape, _initial_top_left = _extract_pattern(g, chain[0], bg)
        for layer_idx, ring in enumerate(chain):
            target_top_left = _target_top_left(ring, shape)
            if layer_idx + 1 < len(chain):
                target_color = chain[layer_idx + 1].color
            else:
                target_color = ring.color
            _stamp_shape(out, rel, target_top_left, target_color)

    return out


def _draw_ring(
    canvas: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
    marker: tuple[int, int],
) -> None:
    r0, c0, r1, c1 = bbox
    canvas[r0, c0 : c1 + 1] = color
    canvas[r1, c0 : c1 + 1] = color
    canvas[r0 : r1 + 1, c0] = color
    canvas[r0 : r1 + 1, c1] = color
    canvas[marker] = RED


def _shape_from_directions(directions: list[str]) -> tuple[int, int]:
    axes = {("h" if d in ("U", "D") else "v") for d in directions}
    if axes == {"h"}:
        return 1, random.randint(1, MAX_LINE_LEN)
    if axes == {"v"}:
        return random.randint(1, MAX_LINE_LEN), 1
    return 1, 1


def _ring_chain_bboxes(
    layers: int, inner_h: int, inner_w: int
) -> list[tuple[int, int, int, int]]:
    bboxes: list[tuple[int, int, int, int]] = []
    for depth in range(layers):
        offset = 1 + 2 * (layers - 1 - depth)
        h = inner_h + 4 * depth
        w = inner_w + 4 * depth
        bboxes.append((offset, offset, offset + h - 1, offset + w - 1))
    return bboxes


def _simulate_path(
    rings: list[tuple[int, int, int, int]],
    directions: list[str],
    start_top_left: tuple[int, int],
    shape: tuple[int, int],
    canvas_shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    ph, pw = shape
    h, w = canvas_shape
    cur_r, cur_c = start_top_left
    markers: list[tuple[int, int]] = []
    placements: list[tuple[int, int]] = []

    for bbox, direction in zip(rings, directions):
        r0, c0, r1, c1 = bbox
        if direction == "U":
            if not (c0 + 1 <= cur_c <= c1 - pw):
                return None
            marker = (r0, cur_c)
            new_r, new_c = r0 - ph, cur_c
        elif direction == "D":
            if not (c0 + 1 <= cur_c <= c1 - pw):
                return None
            marker = (r1, cur_c)
            new_r, new_c = r1 + 1, cur_c
        elif direction == "L":
            if not (r0 + 1 <= cur_r <= r1 - ph):
                return None
            marker = (cur_r, c0)
            new_r, new_c = cur_r, c0 - pw
        else:
            if not (r0 + 1 <= cur_r <= r1 - ph):
                return None
            marker = (cur_r, c1)
            new_r, new_c = cur_r, c1 + 1

        if not (0 <= new_r and new_r + ph <= h and 0 <= new_c and new_c + pw <= w):
            return None

        markers.append(marker)
        placements.append((new_r, new_c))
        cur_r, cur_c = new_r, new_c

    return markers, placements


def _build_expected_from_spec(
    inp: np.ndarray,
    rel: list[tuple[int, int]],
    rings: list[tuple[int, int, int, int]],
    ring_colors: list[int],
    directions: list[str],
    markers: list[tuple[int, int]],
) -> np.ndarray:
    out = inp.copy()
    detected = [Ring(bbox=b, color=c, marker=m, direction=d) for b, c, m, d in zip(rings, ring_colors, markers, directions)]
    for i, ring in enumerate(detected):
        top_left = _target_top_left(ring, (max(dr for dr, _ in rel) + 1, max(dc for _, dc in rel) + 1))
        target_color = ring_colors[i + 1] if i + 1 < len(ring_colors) else ring_colors[i]
        _stamp_shape(out, rel, top_left, target_color)
    return out


def _generate_group_patch() -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(200):
        layers = random.choice(LAYERS_CHOICES)
        directions = [random.choice(DIRECTIONS) for _ in range(layers)]
        ph, pw = _shape_from_directions(directions)

        inner_h = random.randint(max(ph + 2, 5), max(ph + 2, 7))
        inner_w = random.randint(max(pw + 2, 5), max(pw + 2, 7))
        canvas_h = inner_h + 4 * (layers - 1) + 2
        canvas_w = inner_w + 4 * (layers - 1) + 2
        rings = _ring_chain_bboxes(layers, inner_h, inner_w)
        inner_bbox = rings[0]

        starts: list[tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]]] = []
        for sr in range(inner_bbox[0] + 1, inner_bbox[2] - ph + 2):
            for sc in range(inner_bbox[1] + 1, inner_bbox[3] - pw + 2):
                simulated = _simulate_path(
                    rings,
                    directions,
                    (sr, sc),
                    (ph, pw),
                    (canvas_h, canvas_w),
                )
                if simulated is not None:
                    markers, placements = simulated
                    starts.append((sr, sc, markers, placements))
        if not starts:
            continue

        start_r, start_c, markers, _placements = random.choice(starts)
        group_colors = random.sample(PALETTE, layers + 1)
        pattern_color = group_colors[0]
        ring_colors = group_colors[1:]

        inp = np.full((canvas_h, canvas_w), BG, dtype=int)
        for bbox, color, marker in zip(rings, ring_colors, markers):
            _draw_ring(inp, bbox, color, marker)
        inp[start_r : start_r + ph, start_c : start_c + pw] = pattern_color

        rel = [(r, c) for r in range(ph) for c in range(pw)]
        expected = _build_expected_from_spec(inp, rel, rings, ring_colors, directions, markers)

        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp, expected
    return None


def _can_place(
    rect: tuple[int, int, int, int],
    placed: list[tuple[int, int, int, int]],
    padding: int = 1,
) -> bool:
    r0, c0, r1, c1 = rect
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + padding < pr0 or pr1 + padding < r0 or c1 + padding < pc0 or pc1 + padding < c0):
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        group_count = random.choice(GROUP_COUNT_CHOICES)
        group_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for _idx in range(group_count):
            pair = _generate_group_patch()
            if pair is None:
                break
            group_pairs.append(pair)
        if len(group_pairs) != group_count:
            continue

        patches = sorted(group_pairs, key=lambda pair: pair[0].shape[0] * pair[0].shape[1], reverse=True)
        grid_h = random.randint(*GRID_H_RANGE)
        grid_w = random.randint(*GRID_W_RANGE)

        inp = np.full((grid_h, grid_w), BG, dtype=int)
        expected = np.full((grid_h, grid_w), BG, dtype=int)
        placed: list[tuple[int, int, int, int]] = []
        ok = True

        for patch_inp, patch_out in patches:
            ph, pw = patch_inp.shape
            if ph > grid_h or pw > grid_w:
                ok = False
                break

            placed_here = False
            for _attempt in range(PLACEMENT_ATTEMPTS):
                r0 = random.randint(0, grid_h - ph)
                c0 = random.randint(0, grid_w - pw)
                rect = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                if not _can_place(rect, placed):
                    continue
                inp[r0 : r0 + ph, c0 : c0 + pw] = patch_inp
                expected[r0 : r0 + ph, c0 : c0 + pw] = patch_out
                placed.append(rect)
                placed_here = True
                break
            if not placed_here:
                ok = False
                break

        if not ok:
            continue
        if int(np.count_nonzero(inp == BG)) <= int(np.count_nonzero(inp != BG)):
            continue

        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp

    raise RuntimeError("generate_input: failed to build a valid sample")
