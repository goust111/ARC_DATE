from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]  # (r0, c0, r1, c1), inclusive


GRID_SHAPE = (28, 28)
FRAME_TOP_RANGE = (8, 12)
FRAME_LEFT_RANGE = (8, 12)
PALETTE = tuple(range(10))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, grid.flatten()))
    if exclude:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    shifted = coords - np.array([[r0, c0]], dtype=int)
    mask[shifted[:, 0], shifted[:, 1]] = True
    return mask


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": len(cells),
                }
            )
    return comps


def _is_hollow_frame(comp: dict, grid: np.ndarray, bg: int) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    shifted = comp["coords"] - np.array([[r0, c0]], dtype=int)
    mask[shifted[:, 0], shifted[:, 1]] = True
    border = np.zeros_like(mask)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.array_equal(mask, border):
        return False
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    return interior.size > 0 and np.all(interior == int(bg))


def _find_frame(grid: np.ndarray, bg: int) -> dict:
    frames = [comp for comp in _find_components(grid, bg) if _is_hollow_frame(comp, grid, bg)]
    assert len(frames) == 1, "expected exactly one hollow rectangular frame"
    return frames[0]


def _rectangle_dims_from_pair(a_coords: np.ndarray, b_coords: np.ndarray) -> set[tuple[int, int]]:
    a_mask = _coords_to_mask(a_coords)
    b_mask = _coords_to_mask(b_coords)
    a_cells = np.argwhere(a_mask)
    b_cells = np.argwhere(b_mask)
    total_area = int(a_cells.shape[0] + b_cells.shape[0])
    dims: set[tuple[int, int]] = set()

    def _placements(mask_cells: np.ndarray, rect_h: int, rect_w: int, shape: tuple[int, int]) -> list[frozenset[int]]:
        mask_h, mask_w = shape
        if mask_h > rect_h or mask_w > rect_w:
            return []
        placed: list[frozenset[int]] = []
        for top in range(rect_h - mask_h + 1):
            for left in range(rect_w - mask_w + 1):
                idx = (mask_cells[:, 0] + top) * rect_w + (mask_cells[:, 1] + left)
                placed.append(frozenset(map(int, idx)))
        return placed

    for rect_h in range(1, total_area + 1):
        if total_area % rect_h != 0:
            continue
        rect_w = total_area // rect_h
        a_places = _placements(a_cells, rect_h, rect_w, a_mask.shape)
        b_places = _placements(b_cells, rect_h, rect_w, b_mask.shape)
        if not a_places or not b_places:
            continue
        if any(ap.isdisjoint(bp) for ap in a_places for bp in b_places):
            dims.add((int(rect_h), int(rect_w)))
    return dims


def _component_centroid(coords: np.ndarray) -> tuple[float, float]:
    return float(coords[:, 0].mean()), float(coords[:, 1].mean())


def _color_fragment_centers(by_color: dict[int, list[dict]]) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    for color, parts in by_color.items():
        rows = []
        cols = []
        for part in parts:
            rr, cc = _component_centroid(part["coords"])
            rows.append(rr)
            cols.append(cc)
        out[int(color)] = (float(sum(rows) / len(rows)), float(sum(cols) / len(cols)))
    return out


def _rectangle_center(board: np.ndarray, color: int, inner_origin: tuple[int, int]) -> tuple[float, float]:
    coords = np.argwhere(board == int(color))
    r0, c0, r1, c1 = _bbox_coords(coords)
    base_r, base_c = inner_origin
    return base_r + (r0 + r1) / 2.0, base_c + (c0 + c1) / 2.0


def _solve_tilings(
    height: int,
    width: int,
    candidate_dims: dict[int, set[tuple[int, int]]],
    *,
    limit: int = 64,
) -> list[np.ndarray]:
    colors = sorted(candidate_dims)
    board = np.full((height, width), -1, dtype=int)
    solutions: list[np.ndarray] = []

    def rec(used: set[int]) -> None:
        if len(solutions) >= limit:
            return
        empties = np.argwhere(board < 0)
        if empties.size == 0:
            if len(used) == len(colors):
                solutions.append(board.copy())
            return

        r = int(empties[0, 0])
        c = int(empties[0, 1])
        remaining = [color for color in colors if color not in used]
        remaining.sort(key=lambda color: (len(candidate_dims[color]), -max(h * w for h, w in candidate_dims[color])))

        for color in remaining:
            for rect_h, rect_w in sorted(candidate_dims[color]):
                top_lo = max(0, r - rect_h + 1)
                top_hi = min(r, height - rect_h)
                left_lo = max(0, c - rect_w + 1)
                left_hi = min(c, width - rect_w)
                for top in range(top_lo, top_hi + 1):
                    for left in range(left_lo, left_hi + 1):
                        window = board[top : top + rect_h, left : left + rect_w]
                        if np.any(window >= 0):
                            continue
                        window[:] = color
                        used.add(color)
                        rec(used)
                        used.remove(color)
                        window[:] = -1

    rec(set())
    return solutions


def _select_tiling(
    solutions: list[np.ndarray],
    by_color: dict[int, list[dict]],
    inner_origin: tuple[int, int],
) -> np.ndarray:
    assert solutions, "expected at least one tiling solution"
    if len(solutions) == 1:
        return solutions[0]

    fragment_centers = _color_fragment_centers(by_color)
    scored: list[tuple[float, int]] = []
    for idx, board in enumerate(solutions):
        score = 0.0
        for color, center in fragment_centers.items():
            rr, cc = _rectangle_center(board, color, inner_origin)
            score += (rr - center[0]) ** 2 + (cc - center[1]) ** 2
        scored.append((score, idx))
    scored.sort()
    assert len(scored) == 1 or scored[0][0] < scored[1][0], "expected a unique best global packing"
    return solutions[scored[0][1]]


def _parse_problem(grid: np.ndarray) -> tuple[int, dict, dict[int, list[dict]], dict[int, set[tuple[int, int]]]]:
    bg = _get_bg(grid)
    frame = _find_frame(grid, bg)
    frame_color = int(frame["color"])
    r0, c0, r1, c1 = frame["bbox"]

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    assert inner.size > 0 and np.all(inner == bg), "expected empty frame interior"

    frame_cells = {tuple(map(int, cell)) for cell in frame["coords"]}
    comps: list[dict] = []
    for comp in _find_components(grid, bg):
        if int(comp["color"]) != frame_color or comp["bbox"] != frame["bbox"]:
            comps.append(comp)
            continue
        comp_cells = {tuple(map(int, cell)) for cell in comp["coords"]}
        if comp_cells != frame_cells:
            comps.append(comp)

    by_color: dict[int, list[dict]] = {}
    for comp in comps:
        color = int(comp["color"])
        assert color != frame_color
        cr0, cc0, cr1, cc1 = comp["bbox"]
        assert cr1 < r0 or cr0 > r1 or cc1 < c0 or cc0 > c1, "fragments must stay outside the frame"
        by_color.setdefault(color, []).append(comp)

    assert by_color, "expected at least one fragment color"
    candidate_dims: dict[int, set[tuple[int, int]]] = {}
    for color, parts in by_color.items():
        assert len(parts) == 2, "each fragment color must appear in exactly two pieces"
        dims = _rectangle_dims_from_pair(parts[0]["coords"], parts[1]["coords"])
        assert dims, "each color pair must form at least one filled rectangle"
        candidate_dims[color] = dims

    return bg, frame, by_color, candidate_dims


def main(grid: np.ndarray) -> np.ndarray:
    """Pair each same-colored fragment pair into a solid rectangle, then tile the target frame exactly.

    Input structure:
    - The grid has a dominant background, one single-cell-thick hollow rectangular frame, and several outside fragments.
    - Each non-frame color appears in exactly two 4-connected fragments, and the frame interior is empty.
    - Translating the two fragments of one color can form a solid axis-aligned rectangle, but a color pair may admit multiple rectangle dimensions.

    Rule:
    - Detect the unique hollow frame and treat its interior as the target region.
    - For each non-frame color, enumerate every rectangle dimension obtainable by translating one same-colored fragment against the other without rotation or reflection.
    - Choose the unique global assignment in which one rectangle per color tiles the frame interior perfectly with no overlap, erase all outside fragments, and fill the frame interior with those solid color rectangles while keeping the frame.

    Key insight:
    - A fragment pair is not solved locally: the decisive information is the single global rectangle tiling that uses all colors and fills the frame interior exactly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    bg, frame, by_color, candidate_dims = _parse_problem(g)
    r0, c0, r1, c1 = frame["bbox"]
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1

    solutions = _solve_tilings(inner_h, inner_w, candidate_dims, limit=64)
    chosen = _select_tiling(solutions, by_color, (r0 + 1, c0 + 1))

    out = np.full_like(g, bg)
    out[frame["coords"][:, 0], frame["coords"][:, 1]] = int(frame["color"])
    out[r0 + 1 : r1, c0 + 1 : c1] = chosen
    return out


TEMPLATES = (
    {
        "inner_shape": (5, 6),
        "rects": (
            {"bbox": (0, 0, 1, 2), "kind": "amb"},
            {"bbox": (0, 3, 1, 5), "kind": "amb"},
            {"bbox": (2, 0, 4, 1), "kind": "amb"},
            {"bbox": (2, 2, 4, 5), "kind": "uniq"},
        ),
        "slot_pairs": (
            ("top_left", "left_top"),
            ("top_right", "right_top"),
            ("left_bottom", "bottom_left"),
            ("right_mid", "bottom_right"),
        ),
    },
    {
        "inner_shape": (6, 7),
        "rects": (
            {"bbox": (0, 0, 2, 3), "kind": "uniq"},
            {"bbox": (0, 4, 2, 6), "kind": "uniq"},
            {"bbox": (3, 0, 5, 1), "kind": "amb"},
            {"bbox": (3, 2, 5, 3), "kind": "amb"},
            {"bbox": (3, 4, 5, 6), "kind": "uniq"},
        ),
        "slot_pairs": (
            ("top_left", "left_top"),
            ("top_right", "right_top"),
            ("left_bottom", "bottom_left"),
            ("left_mid", "bottom_mid"),
            ("right_mid", "bottom_right"),
        ),
    },
)


def _ambiguous_masks(dim: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([[1, 1, 0], [1, 0, 0]], dtype=int)
    b = np.array([[0, 0, 1], [0, 1, 1]], dtype=int)
    if dim == (2, 3):
        return a.astype(bool), b.astype(bool)
    if dim == (3, 2):
        return a.T.astype(bool), b.T.astype(bool)
    raise ValueError("ambiguous family only supports 2x3 or 3x2 rectangles")


def _unique_masks(dim: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = dim
    assert h >= 2 and w >= 2
    a = np.zeros((h, w), dtype=bool)
    a[0, 0] = True
    a[0, min(1, w - 1)] = True
    a[min(1, h - 1), 0] = True
    b = ~a
    return a, b


def _pair_masks(dim: tuple[int, int], kind: str) -> tuple[np.ndarray, np.ndarray]:
    if kind == "amb":
        return _ambiguous_masks(dim)
    return _unique_masks(dim)


def _paint_patch(grid: np.ndarray, top: int, left: int, patch: np.ndarray, color: int) -> None:
    rows, cols = np.where(patch)
    grid[top + rows, left + cols] = int(color)


def _slot_topleft(frame_bbox: BBox, slot_name: str) -> Coord:
    r0, c0, r1, c1 = frame_bbox
    table = {
        "top_left": (r0 - 6, c0 - 2),
        "top_mid": (r0 - 6, c0 + 3),
        "top_right": (r0 - 6, c0 + 8),
        "bottom_left": (r1 + 4, c0 - 2),
        "bottom_mid": (r1 + 4, c0 + 3),
        "bottom_right": (r1 + 4, c0 + 8),
        "left_top": (r0 - 2, c0 - 6),
        "left_mid": (r0 + 3, c0 - 6),
        "left_bottom": (r0 + 8, c0 - 6),
        "right_top": (r0 - 2, c1 + 4),
        "right_mid": (r0 + 3, c1 + 4),
        "right_bottom": (r0 + 8, c1 + 4),
    }
    return table[slot_name]


def _render_template(template: dict, colors: list[int]) -> np.ndarray:
    inner_h, inner_w = template["inner_shape"]
    board = np.zeros((inner_h, inner_w), dtype=int)
    for rect, color in zip(template["rects"], colors):
        r0, c0, r1, c1 = rect["bbox"]
        board[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return board


def _build_case(template: dict, bg: int, frame_color: int, colors: list[int], frame_top: int, frame_left: int) -> tuple[np.ndarray, np.ndarray]:
    grid = np.full(GRID_SHAPE, bg, dtype=int)
    out = np.full(GRID_SHAPE, bg, dtype=int)

    inner_h, inner_w = template["inner_shape"]
    frame_bbox = (frame_top, frame_left, frame_top + inner_h + 1, frame_left + inner_w + 1)
    fr0, fc0, fr1, fc1 = frame_bbox
    grid[fr0, fc0 : fc1 + 1] = frame_color
    grid[fr1, fc0 : fc1 + 1] = frame_color
    grid[fr0 : fr1 + 1, fc0] = frame_color
    grid[fr0 : fr1 + 1, fc1] = frame_color
    out[:] = grid

    board = _render_template(template, colors)
    out[fr0 + 1 : fr1, fc0 + 1 : fc1] = board

    for rect, slots, color in zip(template["rects"], template["slot_pairs"], colors):
        r0, c0, r1, c1 = rect["bbox"]
        dim = (r1 - r0 + 1, c1 - c0 + 1)
        patch_a, patch_b = _pair_masks(dim, rect["kind"])
        top_a, left_a = _slot_topleft(frame_bbox, slots[0])
        top_b, left_b = _slot_topleft(frame_bbox, slots[1])
        _paint_patch(grid, top_a, left_a, patch_a, color)
        _paint_patch(grid, top_b, left_b, patch_b, color)

    return grid, out


def generate_input() -> np.ndarray:
    template = random.choice(TEMPLATES)
    bg = random.choice(PALETTE)
    frame_choices = [color for color in PALETTE if color != bg]
    frame_color = random.choice(frame_choices)
    rect_colors = random.sample([color for color in PALETTE if color not in {bg, frame_color}], len(template["rects"]))
    frame_top = random.randint(*FRAME_TOP_RANGE)
    frame_left = random.randint(*FRAME_LEFT_RANGE)
    inp, _out = _build_case(template, bg, frame_color, rect_colors, frame_top, frame_left)
    return inp
