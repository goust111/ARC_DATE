from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BG = 0
DIR_NAMES = ("up", "down", "left", "right")
DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = {int(v) for v in np.unique(grid)}
    if exclude is not None:
        colors -= set(exclude)
    return sorted(colors)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def rect_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def shift_cells(cells: set[tuple[int, int]], dr: int, dc: int, steps: int) -> set[tuple[int, int]]:
    return {(r + dr * steps, c + dc * steps) for r, c in cells}


def neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _parse_body_and_stem(comp: list[tuple[int, int]]) -> dict[str, object]:
    comp_set = set(comp)
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    rmin, rmax = min(rows), max(rows)
    cmin, cmax = min(cols), max(cols)
    candidates: list[dict[str, object]] = []

    for body_h, body_w in ((2, 3), (3, 2)):
        for r0 in range(rmin - body_h + 1, rmax + 1):
            for c0 in range(cmin - body_w + 1, cmax + 1):
                body = rect_cells(r0, c0, body_h, body_w)
                if not body.issubset(comp_set):
                    continue
                stem = comp_set - body
                if not 2 <= len(stem) <= 5:
                    continue

                stem_rows = sorted({r for r, _ in stem})
                stem_cols = sorted({c for _, c in stem})
                direction: str | None = None

                if len(stem_cols) == 1 and body_w == 3:
                    stem_c = stem_cols[0]
                    stem_r_vals = sorted(r for r, _ in stem)
                    if stem_c != c0 + 1:
                        continue
                    if stem_r_vals != list(range(stem_r_vals[0], stem_r_vals[-1] + 1)):
                        continue
                    if stem_r_vals[-1] == r0 - 1:
                        direction = "up"
                    elif stem_r_vals[0] == r0 + body_h:
                        direction = "down"

                if len(stem_rows) == 1 and body_h == 3:
                    stem_r = stem_rows[0]
                    stem_c_vals = sorted(c for _, c in stem)
                    if stem_r != r0 + 1:
                        continue
                    if stem_c_vals != list(range(stem_c_vals[0], stem_c_vals[-1] + 1)):
                        continue
                    if stem_c_vals[-1] == c0 - 1:
                        direction = "left"
                    elif stem_c_vals[0] == c0 + body_w:
                        direction = "right"

                if direction is None:
                    continue

                candidates.append(
                    {
                        "body": body,
                        "body_bbox": (r0, c0, r0 + body_h - 1, c0 + body_w - 1),
                        "direction": direction,
                        "stem_len": len(stem),
                        "stem": stem,
                    }
                )

    assert len(candidates) == 1, "moving component must have a unique body+stem decomposition"
    return candidates[0]


def _front_edge_value(body: set[tuple[int, int]], direction: str) -> int:
    if direction == "up":
        return min(r for r, _ in body)
    if direction == "down":
        return max(r for r, _ in body)
    if direction == "left":
        return min(c for _, c in body)
    return max(c for _, c in body)


def _edge_targets(box: tuple[int, int, int, int], direction: str) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if direction == "up":
        return r0, r0 + 1
    if direction == "down":
        return r1, r1 - 1
    if direction == "left":
        return c0, c0 + 1
    return c1, c1 - 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a virtual rectangle from each color's three markers and slide the attached rectangle until it hits that invisible boundary.

    Input structure:
    - The grid has black background and 2-3 foreground colors, one independent group per color.
    - Each color contributes exactly three isolated marker pixels lying on the border of their shared bounding box.
    - The same color also forms one connected moving component: a solid 2x3 or 3x2 rectangular body plus a straight 1-cell-wide stem.

    Rule:
    - For each color separately, compute the bounding box of the three marker pixels; this box stays invisible.
    - Erase the moving component, keep only its rectangular body, and translate that body along the stem direction by exactly the stem length.
    - The translated body's front edge must land on the inferred box edge or one cell just inside it; markers stay fixed and the stem disappears.

    Key insight:
    - The stem gives the motion vector and distance, but the three sparse markers explain why that stopping point is correct: they define the invisible collision frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)

    out = g.copy()
    for color in colors_present(g, exclude={bg}):
        comps = components_mask(g == color)
        singletons = [comp for comp in comps if len(comp) == 1]
        movers = [comp for comp in comps if len(comp) > 1]
        assert len(singletons) == 3, "each color must have exactly three marker pixels"
        assert len(movers) == 1, "each color must have exactly one moving component"

        marker_coords = np.array([singletons[i][0] for i in range(3)], dtype=int)
        box = bbox_coords(marker_coords)
        on_border = (
            (marker_coords[:, 0] == box[0])
            | (marker_coords[:, 0] == box[2])
            | (marker_coords[:, 1] == box[1])
            | (marker_coords[:, 1] == box[3])
        )
        assert bool(np.all(on_border)), "markers must lie on their bounding-box border"

        parsed = _parse_body_and_stem(movers[0])
        direction = parsed["direction"]
        stem_len = int(parsed["stem_len"])
        dr, dc = DIR_VECS[direction]
        body = parsed["body"]
        moved_body = shift_cells(body, dr, dc, stem_len)

        target_values = _edge_targets(box, direction)
        assert _front_edge_value(moved_body, direction) in target_values, "body must stop at the inferred virtual edge"

        out[tuple(np.array(list(parsed["body"] | parsed["stem"])).T)] = bg
        rr, cc = zip(*sorted(moved_body))
        out[np.array(rr), np.array(cc)] = color

    return out


def _touches_full_bbox(markers: list[tuple[int, int]], h: int, w: int) -> bool:
    rows = [r for r, _ in markers]
    cols = [c for _, c in markers]
    return min(rows) == 0 and max(rows) == h - 1 and min(cols) == 0 and max(cols) == w - 1


def _sample_markers(
    h: int,
    w: int,
    forbidden_overlap: set[tuple[int, int]],
    forbidden_adj: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    perimeter = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if r in (0, h - 1) or c in (0, w - 1)
    ]
    candidates = [cell for cell in perimeter if cell not in forbidden_overlap and cell not in forbidden_adj]
    if len(candidates) < 3:
        return None

    for _ in range(250):
        markers = rng.sample(candidates, 3)
        if not _touches_full_bbox(markers, h, w):
            continue
        if len({r for r, _ in markers}) != 3 or len({c for _, c in markers}) != 3:
            continue
        if any(abs(r1 - r2) + abs(c1 - c2) == 1 for (r1, c1), (r2, c2) in combinations(markers, 2)):
            continue
        return markers
    return None


def _build_group_local(color: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(200):
        direction = rng.choice(DIR_NAMES)
        stem_len = rng.randint(2, 5)
        stop_offset = rng.randint(0, 1)

        if direction in ("up", "down"):
            body_h, body_w = 2, 3
            bbox_h_min = stem_len + body_h + stop_offset + 1
            bbox_h = rng.randint(max(6, bbox_h_min), max(6, bbox_h_min) + 3)
            bbox_w = rng.randint(5, 8)
            if bbox_w - body_w - 1 < 1:
                continue
            body_c0 = rng.randint(1, bbox_w - body_w - 1)
            final_r0 = stop_offset if direction == "up" else bbox_h - body_h - stop_offset
            input_r0 = final_r0 + stem_len if direction == "up" else final_r0 - stem_len
            if input_r0 < 0 or input_r0 + body_h > bbox_h:
                continue
            input_c0 = body_c0
            final_body = rect_cells(final_r0, body_c0, body_h, body_w)
            input_body = rect_cells(input_r0, input_c0, body_h, body_w)
            stem_c = body_c0 + 1
            if direction == "up":
                stem = {(r, stem_c) for r in range(input_r0 - stem_len, input_r0)}
            else:
                stem = {(r, stem_c) for r in range(input_r0 + body_h, input_r0 + body_h + stem_len)}
        else:
            body_h, body_w = 3, 2
            bbox_w_min = stem_len + body_w + stop_offset + 1
            bbox_w = rng.randint(max(6, bbox_w_min), max(6, bbox_w_min) + 3)
            bbox_h = rng.randint(5, 8)
            if bbox_h - body_h - 1 < 1:
                continue
            body_r0 = rng.randint(1, bbox_h - body_h - 1)
            final_c0 = stop_offset if direction == "left" else bbox_w - body_w - stop_offset
            input_c0 = final_c0 + stem_len if direction == "left" else final_c0 - stem_len
            if input_c0 < 0 or input_c0 + body_w > bbox_w:
                continue
            input_r0 = body_r0
            final_body = rect_cells(body_r0, final_c0, body_h, body_w)
            input_body = rect_cells(input_r0, input_c0, body_h, body_w)
            stem_r = body_r0 + 1
            if direction == "left":
                stem = {(stem_r, c) for c in range(input_c0 - stem_len, input_c0)}
            else:
                stem = {(stem_r, c) for c in range(input_c0 + body_w, input_c0 + body_w + stem_len)}

        composite = input_body | stem
        forbidden_overlap = composite | final_body
        forbidden_adj = set()
        for cell in composite | final_body:
            forbidden_adj.update(neighbors4(cell))

        markers = _sample_markers(bbox_h, bbox_w, forbidden_overlap, forbidden_adj, rng)
        if markers is None:
            continue

        grid = np.zeros((bbox_h, bbox_w), dtype=int)
        for r, c in composite:
            grid[r, c] = color
        for r, c in markers:
            grid[r, c] = color

        comps = components_mask(grid == color)
        if sorted(len(comp) for comp in comps) != [1, 1, 1, len(composite)]:
            continue

        moved = main(grid)
        if np.array_equal(moved, grid):
            continue
        return grid

    return None


def _boxes_separated(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int,
) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_groups = rng.choice((2, 2, 3))
    colors = rng.sample([1, 2, 3], n_groups)
    locals_: list[np.ndarray] = []
    for color in colors:
        group = _build_group_local(color, rng)
        if group is None:
            return None
        locals_.append(group)

    min_h = max(20, max(g.shape[0] for g in locals_) + 2)
    min_w = max(20, max(g.shape[1] for g in locals_) + 2)
    if min_h > 30 or min_w > 30:
        return None

    for _ in range(250):
        h = rng.randint(min_h, 30)
        w = rng.randint(min_w, 30)
        placements: list[tuple[int, int]] = []
        boxes: list[tuple[int, int, int, int]] = []
        ok = True

        for group in locals_:
            gh, gw = group.shape
            placed = False
            for _ in range(250):
                r0 = rng.randint(1, h - gh - 1)
                c0 = rng.randint(1, w - gw - 1)
                box = (r0, c0, r0 + gh - 1, c0 + gw - 1)
                if all(_boxes_separated(box, other, gap=2) for other in boxes):
                    placements.append((r0, c0))
                    boxes.append(box)
                    placed = True
                    break
            if not placed:
                ok = False
                break

        if not ok:
            continue

        grid = np.zeros((h, w), dtype=int)
        for group, (r0, c0) in zip(locals_, placements):
            gh, gw = group.shape
            patch = grid[r0 : r0 + gh, c0 : c0 + gw]
            if np.any((patch != 0) & (group != 0)):
                ok = False
                break
            patch[group != 0] = group[group != 0]
        if not ok:
            continue

        out = main(grid)
        assert out.shape == grid.shape
        if np.array_equal(out, grid):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
