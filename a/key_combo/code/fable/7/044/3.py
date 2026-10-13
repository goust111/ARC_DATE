from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
TEAL = 8
PALETTE = (1, 2, 3, 4, 6, 7, 9)

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

GEN_MAX_ATTEMPTS = 300
GEN_OUTPUT_HEIGHT = (10, 18)
GEN_OUTPUT_WIDTH = (10, 18)
GEN_TEMPLATE_SHAPES = (1, 2)
GEN_MATCHES_PER_TEMPLATE = (1, 3)
GEN_DISTRACTORS = (1, 4)
GEN_TEMPLATE_AREA = (4, 7)
GEN_DISTRACTOR_AREA = (3, 7)
GEN_BBOX_LIMIT = 4
GEN_OBJECT_TRIES = 80


def _normalize_coords(coords: np.ndarray | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(coords, dtype=int)
    r0 = int(arr[:, 0].min())
    c0 = int(arr[:, 1].min())
    shifted = arr - np.array([r0, c0], dtype=int)
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def _component_dicts(grid: np.ndarray, colors: list[int] | None = None) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    if colors is None:
        colors = sorted(int(x) for x in np.unique(g) if int(x) != BLACK)

    components: list[dict] = []
    for color in colors:
        mask = g == int(color)
        seen = np.zeros_like(mask, dtype=bool)
        for r in range(height):
            for c in range(width):
                if not mask[r, c] or seen[r, c]:
                    continue
                q = deque([(r, c)])
                seen[r, c] = True
                cells: list[tuple[int, int]] = []
                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))
                    for dr, dc in DIR8:
                        nr = cr + dr
                        nc = cc + dc
                        if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            q.append((nr, nc))
                coords = np.asarray(cells, dtype=int)
                components.append({"color": int(color), "coords": coords})
    return components


def _compress_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    keep_rows = ~np.all(g == GREY, axis=1)
    keep_cols = ~np.all(g == GREY, axis=0)
    return g[keep_rows][:, keep_cols]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete full grey separator lines, then repaint every object whose compressed shape matches a teal template to teal.

    Input structure:
    - A black background contains several monochrome objects in assorted colors, with one or more teal objects acting as templates.
    - Some full grey rows and full grey columns have been inserted between original coordinates and may split objects into separated fragments.
    - After removing those grey full lines, the surviving non-black pixels form the true compressed scene.

    Rule:
    - Remove every row and every column that is entirely grey to obtain the compressed grid.
    - In that compressed grid, find all 8-connected monochrome teal components and treat their translation-normalized masks as template shapes.
    - Recolor every other 8-connected monochrome object whose normalized mask matches one of those teal templates to teal, leaving all remaining pixels unchanged.

    Key insight:
    - The grey full lines are inserted coordinate-expansion artifacts rather than walls, so shape matching only becomes visible after coordinate compression.
    """
    compressed = _compress_grid(np.asarray(grid, dtype=int))
    components = _component_dicts(compressed)

    template_shapes = {
        _normalize_coords(component["coords"])
        for component in components
        if int(component["color"]) == TEAL
    }
    assert template_shapes, "expected at least one teal template after compression"

    out = compressed.copy()
    for component in components:
        if int(component["color"]) == TEAL:
            continue
        if _normalize_coords(component["coords"]) not in template_shapes:
            continue
        coords = component["coords"]
        out[coords[:, 0], coords[:, 1]] = TEAL
    return out


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _shape_variants(shape: tuple[tuple[int, int], ...]) -> set[tuple[tuple[int, int], ...]]:
    h, w = _shape_dims(shape)
    canvas = np.zeros((h, w), dtype=int)
    for r, c in shape:
        canvas[r, c] = 1

    variants = set()
    for k in range(4):
        rot = np.rot90(canvas, k)
        variants.add(_normalize_coords(np.argwhere(rot == 1)))
        flip = np.fliplr(rot)
        variants.add(_normalize_coords(np.argwhere(flip == 1)))
    return variants


def _random_connected_shape(area_lo: int, area_hi: int, *, require_2d: bool) -> tuple[tuple[int, int], ...] | None:
    for _ in range(120):
        target_area = random.randint(area_lo, area_hi)
        cells = {(0, 0)}
        while len(cells) < target_area:
            base_r, base_c = random.choice(tuple(cells))
            options = [
                (base_r + dr, base_c + dc)
                for dr, dc in DIR8
                if (base_r + dr, base_c + dc) not in cells
            ]
            if not options:
                break
            new_cell = random.choice(options)
            cells.add(new_cell)
            norm = _normalize_coords(list(cells))
            height, width = _shape_dims(norm)
            if height > GEN_BBOX_LIMIT or width > GEN_BBOX_LIMIT:
                cells.remove(new_cell)

        if len(cells) != target_area:
            continue
        shape = _normalize_coords(list(cells))
        height, width = _shape_dims(shape)
        if height > GEN_BBOX_LIMIT or width > GEN_BBOX_LIMIT:
            continue
        if require_2d and (height < 2 or width < 2):
            continue
        return shape
    return None


def _can_place(shape: tuple[tuple[int, int], ...], top: int, left: int, blocked: np.ndarray) -> bool:
    height, width = blocked.shape
    for dr, dc in shape:
        r = top + dr
        c = left + dc
        if not (0 <= r < height and 0 <= c < width):
            return False
        if blocked[r, c]:
            return False
    return True


def _reserve(coords: list[tuple[int, int]], blocked: np.ndarray) -> None:
    height, width = blocked.shape
    for r, c in coords:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < height and 0 <= nc < width:
                    blocked[nr, nc] = True


def _place_object(
    grid: np.ndarray,
    blocked: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    color: int,
) -> dict | None:
    shape_h, shape_w = _shape_dims(shape)
    max_top = grid.shape[0] - shape_h
    max_left = grid.shape[1] - shape_w
    if max_top < 0 or max_left < 0:
        return None

    positions = [(r, c) for r in range(max_top + 1) for c in range(max_left + 1)]
    random.shuffle(positions)
    for top, left in positions[:GEN_OBJECT_TRIES]:
        if not _can_place(shape, top, left, blocked):
            continue
        coords = [(top + dr, left + dc) for dr, dc in shape]
        for r, c in coords:
            grid[r, c] = int(color)
        _reserve(coords, blocked)
        return {
            "shape": shape,
            "color": int(color),
            "coords": np.asarray(coords, dtype=int),
        }
    return None


def _object_cut_candidates(coords: np.ndarray) -> tuple[set[int], set[int]]:
    row_cuts = set()
    col_cuts = set()
    rows = sorted(set(int(r) for r in coords[:, 0]))
    cols = sorted(set(int(c) for c in coords[:, 1]))
    for r0, r1 in zip(rows, rows[1:]):
        if r1 == r0 + 1:
            row_cuts.add(r0)
    for c0, c1 in zip(cols, cols[1:]):
        if c1 == c0 + 1:
            col_cuts.add(c0)
    return row_cuts, col_cuts


def _select_spaced_cuts(candidates: set[int], count_hi: int) -> list[int]:
    if not candidates or count_hi <= 0:
        return []
    target = random.randint(1, min(len(candidates), count_hi))
    pool = list(candidates)
    random.shuffle(pool)
    chosen: list[int] = []
    for cut in pool:
        if all(abs(cut - prev) > 1 for prev in chosen):
            chosen.append(int(cut))
        if len(chosen) == target:
            break
    if not chosen:
        chosen.append(int(random.choice(pool)))
    return sorted(chosen)


def _expand_with_grey_lines(
    compressed: np.ndarray,
    row_cuts: list[int],
    col_cuts: list[int],
) -> np.ndarray:
    expanded = compressed.copy()
    for offset, cut in enumerate(sorted(row_cuts)):
        expanded = np.insert(expanded, cut + 1 + offset, GREY, axis=0)
    for offset, cut in enumerate(sorted(col_cuts)):
        expanded = np.insert(expanded, cut + 1 + offset, GREY, axis=1)
    return expanded


def _expected_output(compressed: np.ndarray) -> np.ndarray:
    return main(compressed)


def _attempt_case() -> np.ndarray | None:
    out_h = random.randint(*GEN_OUTPUT_HEIGHT)
    out_w = random.randint(*GEN_OUTPUT_WIDTH)
    compressed = np.zeros((out_h, out_w), dtype=int)
    blocked = np.zeros_like(compressed, dtype=bool)
    objects: list[dict] = []

    template_shape_count = random.randint(*GEN_TEMPLATE_SHAPES)
    template_shapes: list[tuple[tuple[int, int], ...]] = []
    template_shape_set: set[tuple[tuple[int, int], ...]] = set()

    for _ in range(template_shape_count):
        shape = None
        for _ in range(40):
            candidate = _random_connected_shape(*GEN_TEMPLATE_AREA, require_2d=True)
            if candidate is not None and candidate not in template_shape_set:
                shape = candidate
                break
        if shape is None:
            return None
        template_shapes.append(shape)
        template_shape_set.add(shape)

    for shape in template_shapes:
        template = _place_object(compressed, blocked, shape, TEAL)
        if template is None:
            return None
        objects.append(template)

        match_count = random.randint(*GEN_MATCHES_PER_TEMPLATE)
        for _ in range(match_count):
            color = random.choice(PALETTE)
            match = _place_object(compressed, blocked, shape, color)
            if match is None:
                return None
            objects.append(match)

    distractor_count = random.randint(*GEN_DISTRACTORS)
    for _ in range(distractor_count):
        shape = None
        for _ in range(40):
            if random.random() < 0.45 and template_shapes:
                base = random.choice(template_shapes)
                candidates = [variant for variant in _shape_variants(base) if variant not in template_shape_set]
                candidate = random.choice(candidates) if candidates else None
            else:
                candidate = _random_connected_shape(*GEN_DISTRACTOR_AREA, require_2d=False)
            if candidate is not None and candidate not in template_shape_set:
                shape = candidate
                break
        if shape is None:
            return None
        color = random.choice(PALETTE)
        distractor = _place_object(compressed, blocked, shape, color)
        if distractor is None:
            return None
        objects.append(distractor)

    row_candidates: set[int] = set()
    col_candidates: set[int] = set()
    for obj in objects:
        rows, cols = _object_cut_candidates(obj["coords"])
        row_candidates |= rows
        col_candidates |= cols

    row_room = 30 - out_h
    col_room = 30 - out_w
    row_possible = bool(row_candidates) and row_room > 0
    col_possible = bool(col_candidates) and col_room > 0
    if not row_possible and not col_possible:
        return None

    modes = []
    if row_possible:
        modes.append("row")
    if col_possible:
        modes.append("col")
    if row_possible and col_possible:
        modes.extend(["both", "both"])
    mode = random.choice(modes)

    row_cuts: list[int] = []
    col_cuts: list[int] = []
    if mode in {"row", "both"}:
        row_cuts = _select_spaced_cuts(row_candidates, min(3, row_room))
    if mode in {"col", "both"}:
        col_cuts = _select_spaced_cuts(col_candidates, min(3, col_room))
    if not row_cuts and not col_cuts:
        return None

    expected = _expected_output(compressed)
    inp = _expand_with_grey_lines(compressed, row_cuts, col_cuts)

    if inp.shape[0] > 30 or inp.shape[1] > 30:
        return None
    if not np.any(inp == GREY):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _attempt_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
