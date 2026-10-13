from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
CONTROL_COLORS = (BLUE, GREEN, YELLOW, PINK)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _shape_signature(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox_from_coords(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _coords_by_row(coords: np.ndarray) -> dict[int, list[int]]:
    rows: dict[int, list[int]] = {}
    for r, c in coords:
        rows.setdefault(int(r), []).append(int(c))
    return rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit each grey puzzle piece into its matching hole in the red shape, then recolor the placed pixels row by row from the right-edge control markers.

    Input structure:
    - A black background contains one red shape and 2-3 separate grey puzzle pieces.
    - Black 4-connected regions inside the red shape's tight bounding box act as holes.
    - The rightmost column carries one control-color marker on every row covered by the red bbox.

    Rule:
    - Find the red shape and enumerate all black 4-connected hole regions inside its bbox.
    - Sort grey pieces by bbox top-left; for each piece, match its cropped shape to the unique hole with the same shape and clear the original grey piece.
    - Place the piece into that hole and recolor every placed pixel using the marker color from the same row in the rightmost column; if a row has no marker, that placed pixel stays black.

    Key insight:
    - Shape determines where each piece goes, but row position determines its final color, so one placed piece can split into multiple colors across different rows.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    red_components = _components(g == RED)
    assert len(red_components) == 1, "expected exactly one red object"
    red_coords = red_components[0]
    rr0, rc0, rr1, rc1 = _bbox_from_coords(red_coords)

    bbox_view = g[rr0 : rr1 + 1, rc0 : rc1 + 1]
    hole_components_local = _components(bbox_view == BLACK)
    assert 2 <= len(hole_components_local) <= 3, "expected 2-3 holes in red bbox"

    holes = []
    for local in hole_components_local:
        absolute = local.copy()
        absolute[:, 0] += rr0
        absolute[:, 1] += rc0
        holes.append(
            {
                "coords": absolute,
                "signature": _shape_signature(absolute),
            }
        )

    grey_components = _components(g == GREY)
    assert len(grey_components) == len(holes), "piece count must equal hole count"
    grey_components.sort(key=lambda coords: _bbox_from_coords(coords)[:2])

    out = g.copy()
    out[out == GREY] = BLACK
    unmatched = list(range(len(holes)))

    for piece in grey_components:
        signature = _shape_signature(piece)
        matches = [idx for idx in unmatched if holes[idx]["signature"] == signature]
        assert len(matches) == 1, "each grey piece must match exactly one hole"
        hole = holes[matches[0]]
        unmatched.remove(matches[0])

        for r, cols in _coords_by_row(hole["coords"]).items():
            row_color = int(out[r, -1])
            fill = row_color if row_color not in (BLACK, GREY) else BLACK
            out[r, cols] = fill

    return out


def _random_connected_shape(area: int) -> np.ndarray | None:
    for _ in range(80):
        h = random.randint(2, min(4, area))
        w = random.randint(2, min(4, area))
        if area > h * w:
            continue
        mask = np.zeros((h, w), dtype=bool)
        seed = (random.randrange(h), random.randrange(w))
        mask[seed] = True
        frontier = {seed}
        while int(mask.sum()) < area:
            candidates = set()
            for r, c in frontier:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        candidates.add((nr, nc))
            if not candidates:
                break
            nr, nc = random.choice(tuple(candidates))
            mask[nr, nc] = True
            frontier.add((nr, nc))
        if int(mask.sum()) != area:
            continue

        coords = np.argwhere(mask)
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
        if cropped.shape[0] == 1 and cropped.shape[1] == 1:
            continue
        if area == int(cropped.size) and random.random() < 0.65:
            continue
        return cropped
    return None


def _sample_layout() -> tuple[int, int, list[int]]:
    choice = random.random()
    if choice < 0.35:
        box_h, box_w, count = 8, 8, 2
        while True:
            areas = [random.randint(6, 8) for _ in range(count)]
            if 14 <= sum(areas) <= 18:
                return box_h, box_w, areas
    if choice < 0.75:
        box_h, box_w, count = 8, 8, 3
        while True:
            areas = [random.randint(4, 8) for _ in range(count)]
            if 16 <= sum(areas) <= 22:
                return box_h, box_w, areas

    box_h, box_w, count = random.choice(((8, 9, 3), (9, 8, 3)))
    while True:
        areas = [random.randint(6, 8) for _ in range(count)]
        if 22 <= sum(areas) <= 24:
            return box_h, box_w, areas


def _place_holes(box_h: int, box_w: int, shapes: list[np.ndarray]) -> list[np.ndarray] | None:
    occupancy = np.zeros((box_h, box_w), dtype=bool)
    positions: list[np.ndarray] = []
    for shape in sorted(shapes, key=lambda s: int(s.sum()), reverse=True):
        sh, sw = shape.shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, box_h - sh):
            for c0 in range(1, box_w - sw):
                ok = True
                for sr, sc in np.argwhere(shape):
                    ar = int(r0 + sr)
                    ac = int(c0 + sc)
                    if occupancy[ar, ac]:
                        ok = False
                        break
                    for nr, nc in _neighbors4(ar, ac):
                        if 0 <= nr < box_h and 0 <= nc < box_w and occupancy[nr, nc]:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        coords = np.argwhere(shape).astype(int)
        coords[:, 0] += r0
        coords[:, 1] += c0
        positions.append(coords)
        occupancy[coords[:, 0], coords[:, 1]] = True

    holes = positions
    holes.sort(key=lambda coords: _bbox_from_coords(coords)[:2])
    if len(_components(occupancy)) != len(holes):
        return None
    return holes


def _sample_row_colors(active_rows: list[int], holes_abs: list[np.ndarray]) -> dict[int, int] | None:
    for _ in range(80):
        colors: dict[int, int] = {}
        prev = random.choice(CONTROL_COLORS)
        for idx, row in enumerate(active_rows):
            if idx == 0:
                color = prev
            elif random.random() < 0.45:
                color = prev
            else:
                options = [c for c in CONTROL_COLORS if c != prev]
                color = random.choice(options)
            colors[row] = color
            prev = color

        if len(set(colors.values())) < 2:
            continue

        multi_row_multicolor = False
        for hole in holes_abs:
            rows = sorted({int(r) for r in hole[:, 0]})
            if len(rows) >= 2 and len({colors[r] for r in rows}) >= 2:
                multi_row_multicolor = True
                break
        if multi_row_multicolor:
            return colors
    return None


def _bbox_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _place_pieces(
    grid: np.ndarray,
    piece_shapes: list[np.ndarray],
    red_bbox: tuple[int, int, int, int],
    active_regions: list[str],
) -> bool:
    h, w = grid.shape
    piece_bboxes: list[tuple[int, int, int, int]] = []

    for shape, region in zip(piece_shapes, active_regions):
        sh, sw = shape.shape
        candidates: list[tuple[int, int]] = []
        if region == "top":
            row_range = range(0, red_bbox[0] - sh + 1)
            col_range = range(0, w - 1 - sw + 1)
        elif region == "bottom":
            row_range = range(red_bbox[2] + 2, h - sh + 1)
            col_range = range(0, w - 1 - sw + 1)
        else:
            row_range = range(0, h - sh + 1)
            col_range = range(0, red_bbox[1] - sw + 1)

        for r0 in row_range:
            for c0 in col_range:
                bbox = (r0, c0, r0 + sh - 1, c0 + sw - 1)
                if _bbox_intersects(bbox, red_bbox, pad=0):
                    continue
                if any(_bbox_intersects(bbox, other, pad=1) for other in piece_bboxes):
                    continue
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(h, r0 + sh + 1)
                cc1 = min(w - 1, c0 + sw + 1)
                if np.any(grid[rr0:rr1, cc0:cc1] != BLACK):
                    continue
                candidates.append((r0, c0))

        if not candidates:
            return False

        r0, c0 = random.choice(candidates)
        coords = np.argwhere(shape)
        for sr, sc in coords:
            grid[r0 + int(sr), c0 + int(sc)] = GREY
        piece_bboxes.append((r0, c0, r0 + sh - 1, c0 + sw - 1))

    return True


def _generate_attempt() -> np.ndarray | None:
    box_h, box_w, areas = _sample_layout()

    shapes: list[np.ndarray] = []
    signatures: set[tuple[tuple[int, int], ...]] = set()
    for area in sorted(areas, reverse=True):
        shape = _random_connected_shape(area)
        if shape is None:
            return None
        sig = _shape_signature(np.argwhere(shape))
        if sig in signatures:
            return None
        signatures.add(sig)
        shapes.append(shape)

    holes_local = _place_holes(box_h, box_w, shapes)
    if holes_local is None:
        return None

    red_area = box_h * box_w - sum(len(hole) for hole in holes_local)
    if not (30 <= red_area <= 50):
        return None

    top_margin = random.randint(3, 4)
    bottom_margin = random.randint(3, 4)
    left_margin = random.randint(4, 5)
    right_margin = random.randint(2, 3)
    h = top_margin + box_h + bottom_margin
    w = left_margin + box_w + right_margin + 1
    if max(h, w) > 30:
        return None

    grid = np.zeros((h, w), dtype=int)
    red_r0 = top_margin
    red_c0 = left_margin
    red_bbox = (red_r0, red_c0, red_r0 + box_h - 1, red_c0 + box_w - 1)

    grid[red_r0 : red_r0 + box_h, red_c0 : red_c0 + box_w] = RED
    holes_abs: list[np.ndarray] = []
    for hole in holes_local:
        placed = hole.copy()
        placed[:, 0] += red_r0
        placed[:, 1] += red_c0
        holes_abs.append(placed)
        grid[placed[:, 0], placed[:, 1]] = BLACK

    row_colors = _sample_row_colors(list(range(red_r0, red_r0 + box_h)), holes_abs)
    if row_colors is None:
        return None
    for row, color in row_colors.items():
        grid[row, -1] = color

    if len(holes_abs) == 2:
        regions = random.sample(["top", "bottom", "left"], 2)
    else:
        regions = ["top", "bottom", "left"]
        random.shuffle(regions)

    piece_shapes = [shape.astype(bool) for shape in shapes]
    if not _place_pieces(grid, piece_shapes, red_bbox, regions):
        return None

    if len(_components(grid == GREY)) != len(holes_abs):
        return None
    if len(_components(grid[red_r0 : red_r0 + box_h, red_c0 : red_c0 + box_w] == BLACK)) != len(holes_abs):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    inserted_colors = set(
        int(out[r, c])
        for hole in holes_abs
        for r, c in hole
        if int(out[r, c]) != BLACK
    )
    if len(inserted_colors) < 2:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 3")
