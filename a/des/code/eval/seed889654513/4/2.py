from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


GEN_BG = 0
GEN_FRAME = 5
GEN_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_GRID_H = (22, 30)
GEN_GRID_W = (22, 30)
GEN_FRAME_H = (9, 13)
GEN_FRAME_W = (10, 16)
GEN_HOLE_COUNTS = (2, 3, 4)

GEN_RETRIES = 400
GEN_SHAPE_RETRIES = 200
GEN_HOLE_PLACEMENT_RETRIES = 160
GEN_MARKER_RETRIES = 120
GEN_PIECE_RETRIES = 220


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


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
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(pts)
    return out


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _coords_to_mask(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> np.ndarray:
    norm = _normalize(coords)
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _rotate_shape(
    coords: list[tuple[int, int]] | tuple[tuple[int, int], ...], k: int
) -> tuple[tuple[int, int], ...]:
    mask = _coords_to_mask(coords)
    rotated = np.rot90(mask, int(k))
    pts = [tuple(map(int, rc)) for rc in np.argwhere(rotated)]
    return _normalize(pts)


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    best: tuple[tuple[int, int], int, tuple[int, int, int, int]] | None = None
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for pts in _components(grid == color):
            r0, c0, r1, c1 = _bbox(pts)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 5 or w < 5:
                continue
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not np.any(sub[1:-1, 1:-1] == bg):
                continue
            border = np.zeros((h, w), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.all(sub[border] == color):
                continue
            score = (h * w, len(pts))
            if best is None or score > best[0]:
                best = (score, color, (r0, c0, r1, c1))
    assert best is not None, "expected a grey template rectangle"
    return best[1], best[2]


def _extract_holes(grid: np.ndarray, bg: int, frame_bbox: tuple[int, int, int, int]) -> list[list[tuple[int, int]]]:
    r0, c0, r1, c1 = frame_bbox
    sub = grid[r0 : r1 + 1, c0 : c1 + 1] == bg
    holes: list[list[tuple[int, int]]] = []
    for pts in _components(sub):
        if any(rr == 0 or cc == 0 or rr == sub.shape[0] - 1 or cc == sub.shape[1] - 1 for rr, cc in pts):
            continue
        holes.append([(r0 + rr, c0 + cc) for rr, cc in pts])
    assert holes, "expected cut-out holes inside the template"
    return holes


def _color_objects(
    grid: np.ndarray, bg: int, frame_color: int, frame_bbox: tuple[int, int, int, int]
) -> dict[int, dict[str, object]]:
    h, w = grid.shape
    r0, c0, r1, c1 = frame_bbox
    out: dict[int, dict[str, object]] = {}
    for color in map(int, np.unique(grid)):
        if color in (bg, frame_color):
            continue
        marker: dict[str, tuple[int, int]] | None = None
        piece: list[tuple[int, int]] | None = None
        for pts in _components(grid == color):
            if len(pts) == 1:
                pr, pc = pts[0]
                border_neighbors = []
                if pr < r0 or pr > r1 or pc < c0 or pc > c1:
                    for dr, dc in DIR4:
                        nr, nc = pr + dr, pc + dc
                        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == frame_color:
                            if nr in (r0, r1) or nc in (c0, c1):
                                border_neighbors.append((nr, nc))
                if len(border_neighbors) == 1:
                    marker = {"pixel": (pr, pc), "border": border_neighbors[0]}
                    continue
            if piece is None:
                piece = pts
            else:
                assert False, "expected exactly one fragment for each marker color"
        if marker is not None and piece is not None:
            out[color] = {"marker": marker, "piece": piece}
    assert out, "expected colored markers and fragments"
    return out


def _match_holes(
    color_info: dict[int, dict[str, object]], holes: list[list[tuple[int, int]]]
) -> dict[int, list[tuple[int, int]]]:
    out: dict[int, list[tuple[int, int]]] = {}
    used: set[int] = set()
    for color, info in color_info.items():
        border_r, border_c = info["marker"]["border"]  # type: ignore[index]
        distances = [
            min(abs(border_r - hr) + abs(border_c - hc) for hr, hc in hole)
            for hole in holes
        ]
        best = min(distances)
        matches = [idx for idx, dist in enumerate(distances) if dist == best]
        assert len(matches) == 1, "marker-to-hole mapping must be unique"
        hole_idx = matches[0]
        assert hole_idx not in used, "each marker must point to a different hole"
        used.add(hole_idx)
        out[color] = holes[hole_idx]
    assert len(out) == len(holes) == len(color_info), "need a one-to-one color mapping"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each template hole with the same-color fragment after matching by border marker and rotation.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frame_color, frame_bbox = _find_frame(g, bg)
    holes = _extract_holes(g, bg, frame_bbox)
    color_info = _color_objects(g, bg, frame_color, frame_bbox)
    hole_by_color = _match_holes(color_info, holes)

    out = g.copy()
    for info in color_info.values():
        for r, c in info["piece"]:  # type: ignore[index]
            out[r, c] = bg

    for color, hole in hole_by_color.items():
        piece = color_info[color]["piece"]  # type: ignore[index]
        hole_shape = _normalize(hole)
        piece_shape = _normalize(piece)
        fits = [k for k in range(4) if _rotate_shape(piece_shape, k) == hole_shape]
        assert fits, "fragment must match the hole under rotation"
        for r, c in hole:
            out[r, c] = color

    return out


def _random_shape(rng: random.Random) -> np.ndarray | None:
    for _ in range(GEN_SHAPE_RETRIES):
        max_h = rng.randint(2, 4)
        max_w = rng.randint(2, 4)
        max_area = min(max_h * max_w - 1, 7)
        min_area = max(3, (max_h + max_w) // 2)
        if min_area > max_area:
            continue
        target_area = rng.randint(min_area, max_area)
        seed = (rng.randrange(max_h), rng.randrange(max_w))
        cells = {seed}
        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < max_h and 0 <= nc < max_w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(tuple(frontier)))
        if len(cells) != target_area:
            continue

        norm = _normalize(tuple(cells))
        h = max(r for r, _ in norm) + 1
        w = max(c for _, c in norm) + 1
        if h < 2 or w < 2:
            continue
        if any(_rotate_shape(norm, k) == norm for k in (1, 2, 3)):
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in norm:
            mask[r, c] = True
        return mask
    return None


def _place_holes(
    rng: random.Random, frame_h: int, frame_w: int, count: int
) -> list[dict[str, object]] | None:
    inner_h = frame_h - 2
    inner_w = frame_w - 2
    blocked = np.zeros((inner_h, inner_w), dtype=bool)
    holes: list[dict[str, object]] = []
    seen_shapes: set[tuple[tuple[int, int], ...]] = set()

    for _ in range(count):
        shape = _random_shape(rng)
        if shape is None:
            return None
        canonical = min(_rotate_shape(_normalize([tuple(map(int, rc)) for rc in np.argwhere(shape)]), k) for k in range(4))
        if canonical in seen_shapes:
            alt_shape = _random_shape(rng)
            if alt_shape is None:
                return None
            shape = alt_shape
            canonical = min(_rotate_shape(_normalize([tuple(map(int, rc)) for rc in np.argwhere(shape)]), k) for k in range(4))
        seen_shapes.add(canonical)

        sh, sw = shape.shape
        placed = False
        for _ in range(GEN_HOLE_PLACEMENT_RETRIES):
            top = rng.randint(0, inner_h - sh)
            left = rng.randint(0, inner_w - sw)
            cells = [(top + r, left + c) for r, c in np.argwhere(shape)]
            if any(blocked[r, c] for r, c in cells):
                continue
            for r, c in cells:
                for dr, dc in DIR8 + [(0, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < inner_h and 0 <= nc < inner_w:
                        blocked[nr, nc] = True
            holes.append({"shape": shape.copy(), "cells_inner": cells})
            placed = True
            break
        if not placed:
            return None
    return holes


def _marker_candidate(
    rng: random.Random,
    hole_cells_local: list[tuple[int, int]],
    frame_bbox: tuple[int, int, int, int],
    side: str,
) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    if side == "top":
        edge = min(r for r, _ in hole_cells_local)
        options = [c for r, c in hole_cells_local if r == edge]
        col = rng.choice(options)
        return (r0 - 1, c0 + col), (r0, c0 + col)
    if side == "bottom":
        edge = max(r for r, _ in hole_cells_local)
        options = [c for r, c in hole_cells_local if r == edge]
        col = rng.choice(options)
        return (r1 + 1, c0 + col), (r1, c0 + col)
    if side == "left":
        edge = min(c for _, c in hole_cells_local)
        options = [r for r, c in hole_cells_local if c == edge]
        row = rng.choice(options)
        return (r0 + row, c0 - 1), (r0 + row, c0)
    edge = max(c for _, c in hole_cells_local)
    options = [r for r, c in hole_cells_local if c == edge]
    row = rng.choice(options)
    return (r0 + row, c1 + 1), (r0 + row, c1)


def _assign_markers(
    rng: random.Random, holes: list[dict[str, object]], frame_bbox: tuple[int, int, int, int]
) -> bool:
    sides = ["top", "bottom", "left", "right"]
    all_cells = [hole["cells_global"] for hole in holes]

    for _ in range(GEN_MARKER_RETRIES):
        used_markers: set[tuple[int, int]] = set()
        used_borders: set[tuple[int, int]] = set()
        ok = True
        for hole in holes:
            order = sides[:]
            rng.shuffle(order)
            chosen = None
            for side in order:
                marker, border = _marker_candidate(
                    rng, hole["cells_local"], frame_bbox, side  # type: ignore[arg-type]
                )
                if marker in used_markers or border in used_borders:
                    continue
                if any(abs(marker[0] - mr) + abs(marker[1] - mc) <= 1 for mr, mc in used_markers):
                    continue
                chosen = (marker, border, side)
                break
            if chosen is None:
                ok = False
                break
            hole["marker"] = chosen[0]
            hole["border"] = chosen[1]
            hole["side"] = chosen[2]
            used_markers.add(chosen[0])
            used_borders.add(chosen[1])

        if not ok:
            continue

        for idx, hole in enumerate(holes):
            border_r, border_c = hole["border"]  # type: ignore[index]
            distances = [
                min(abs(border_r - r) + abs(border_c - c) for r, c in cells)  # type: ignore[misc]
                for cells in all_cells
            ]
            best = min(distances)
            if distances.count(best) != 1 or distances[idx] != best:
                ok = False
                break
        if ok:
            return True
    return False


def _place_piece(
    rng: random.Random, grid: np.ndarray, patch: np.ndarray, color: int
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    ph, pw = patch.shape
    for _ in range(GEN_PIECE_RETRIES):
        top = rng.randint(0, h - ph)
        left = rng.randint(0, w - pw)
        r0 = max(0, top - 1)
        c0 = max(0, left - 1)
        r1 = min(h, top + ph + 1)
        c1 = min(w, left + pw + 1)
        if np.any(grid[r0:r1, c0:c1] != GEN_BG):
            continue
        cells: list[tuple[int, int]] = []
        for r, c in np.argwhere(patch):
            rr = int(top + r)
            cc = int(left + c)
            grid[rr, cc] = color
            cells.append((rr, cc))
        return cells
    return None


def _expected_output(grid: np.ndarray, holes: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for hole in holes:
        for r, c in hole["piece_cells"]:  # type: ignore[index]
            out[r, c] = GEN_BG
        for r, c in hole["cells_global"]:  # type: ignore[index]
            out[r, c] = int(hole["color"])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GEN_GRID_H)
    w = rng.randint(*GEN_GRID_W)
    frame_h = rng.randint(*GEN_FRAME_H)
    frame_w = rng.randint(*GEN_FRAME_W)
    if frame_h > h - 4 or frame_w > w - 4:
        return None

    frame_r0 = rng.randint(2, h - frame_h - 2)
    frame_c0 = rng.randint(2, w - frame_w - 2)
    frame_bbox = (frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1)

    hole_count = rng.choice(GEN_HOLE_COUNTS)
    holes = _place_holes(rng, frame_h, frame_w, hole_count)
    if holes is None:
        return None

    colors = rng.sample(list(GEN_COLORS), hole_count)
    grid = np.full((h, w), GEN_BG, dtype=int)
    grid[frame_r0 : frame_r0 + frame_h, frame_c0 : frame_c0 + frame_w] = GEN_FRAME

    for hole, color in zip(holes, colors):
        cells_global = []
        cells_local = []
        for inner_r, inner_c in hole["cells_inner"]:  # type: ignore[index]
            rr = frame_r0 + 1 + int(inner_r)
            cc = frame_c0 + 1 + int(inner_c)
            grid[rr, cc] = GEN_BG
            cells_global.append((rr, cc))
            cells_local.append((1 + int(inner_r), 1 + int(inner_c)))
        hole["cells_global"] = cells_global
        hole["cells_local"] = cells_local
        hole["color"] = int(color)

    if not _assign_markers(rng, holes, frame_bbox):
        return None

    for hole in holes:
        mr, mc = hole["marker"]  # type: ignore[index]
        grid[mr, mc] = int(hole["color"])

    for hole in holes:
        rotation = rng.randrange(4)
        patch = np.rot90(hole["shape"], rotation)  # type: ignore[arg-type]
        piece_cells = _place_piece(rng, grid, patch, int(hole["color"]))
        if piece_cells is None:
            return None
        hole["rotation"] = rotation
        hole["piece_cells"] = piece_cells

    counts = Counter(int(v) for v in grid.ravel())
    bg_count = counts[GEN_BG]
    if any(color != GEN_BG and count >= bg_count for color, count in counts.items()):
        return None

    expected = _expected_output(grid, holes)
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
