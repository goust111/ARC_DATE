from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
RNG = np.random.default_rng()

MAX_GRID_SIDE = 30
MIN_PAIR_COUNT = 2
MAX_PAIR_COUNT = 5
MAX_GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 250
SPEC_ATTEMPTS = 300
MARGIN_CAP = 4

NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _normalize_piece(coords: np.ndarray) -> dict[str, object]:
    r0, c0, r1, c1 = _bbox(coords)
    rel = {(int(r - r0), int(c - c0)) for r, c in coords}
    return {
        "coords": rel,
        "bbox": (r0, c0, r1, c1),
        "h": int(r1 - r0 + 1),
        "w": int(c1 - c0 + 1),
        "area": int(coords.shape[0]),
    }


def _assembled_rectangle_dims(piece_a: dict[str, object], piece_b: dict[str, object]) -> tuple[int, int]:
    coords_a = piece_a["coords"]
    coords_b = piece_b["coords"]
    assert isinstance(coords_a, set) and isinstance(coords_b, set)
    h_a = int(piece_a["h"])
    w_a = int(piece_a["w"])
    h_b = int(piece_b["h"])
    w_b = int(piece_b["w"])

    solutions: set[tuple[int, int]] = set()
    for dr in range(-h_b, h_a + 1):
        for dc in range(-w_b, w_a + 1):
            shifted_b = {(r + dr, c + dc) for r, c in coords_b}
            if coords_a & shifted_b:
                continue
            union = coords_a | shifted_b
            rows = [r for r, _ in union]
            cols = [c for _, c in union]
            out_h = max(rows) - min(rows) + 1
            out_w = max(cols) - min(cols) + 1
            if out_h * out_w != len(union):
                continue
            if out_h == int(piece_a["h"]) and out_w == int(piece_a["w"]):
                continue
            if out_h == int(piece_b["h"]) and out_w == int(piece_b["w"]):
                continue
            solutions.add((int(out_h), int(out_w)))

    if len(solutions) != 1:
        raise AssertionError(f"expected a unique rectangle assembly, found {sorted(solutions)}")
    return next(iter(solutions))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair scattered same-color fragments, reassemble each pair into a full rectangle, and stack the rectangles by area.

    Input structure:
    - Black background with several scattered monochrome 4-connected objects.
    - Each non-black color appears in exactly two separate objects.
    - The two objects of a color are complementary fragments of one axis-aligned rectangle, separated by a horizontal or vertical cut and movable only by translation.

    Rule:
    - Group objects by color and extract the two fragments for each color.
    - Find the unique way to translate the two fragments so their union becomes one solid rectangle with no overlap and no gaps.
    - Compute each reconstructed rectangle's area, then sort rectangles from largest area to smallest.
    - Build a black output grid whose width is the maximum reconstructed width and whose height is the sum of reconstructed heights; place the sorted rectangles top-to-bottom, left-aligned, with no gaps between them.

    Key insight:
    - The important hidden structure is not the fragments' original positions but the unique rectangle each color pair can form after translation; the final layout depends only on those reconstructed rectangle sizes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    colors = sorted(int(color) for color in np.unique(g) if int(color) != BLACK)
    assert colors

    rectangles: list[tuple[int, int, int, int, int, int]] = []
    for color in colors:
        comps = _components_mask(g == color)
        if len(comps) != 2:
            raise AssertionError(f"color {color} must have exactly two fragments")
        pieces = [_normalize_piece(comp) for comp in comps]
        out_h, out_w = _assembled_rectangle_dims(pieces[0], pieces[1])
        area = out_h * out_w
        anchor_r = min(int(piece["bbox"][0]) for piece in pieces)
        anchor_c = min(int(piece["bbox"][1]) for piece in pieces)
        rectangles.append((area, out_h, out_w, color, anchor_r, anchor_c))

    rectangles.sort(key=lambda item: (-item[0], -item[1], -item[2], item[4], item[5]))
    out_h = sum(item[1] for item in rectangles)
    out_w = max(item[2] for item in rectangles)
    assert out_h <= MAX_GRID_SIDE and out_w <= MAX_GRID_SIDE

    out = np.full((out_h, out_w), BLACK, dtype=int)
    row = 0
    for _area, rect_h, rect_w, color, _anchor_r, _anchor_c in rectangles:
        out[row : row + rect_h, :rect_w] = color
        row += rect_h
    return out


def _orientation_schedule(pair_count: int) -> list[str]:
    if pair_count == 2:
        orientations = ["horizontal_cut", "vertical_cut"]
    else:
        horizontal = pair_count // 2
        vertical = pair_count // 2
        if pair_count % 2:
            if int(RNG.integers(0, 2)) == 0:
                horizontal += 1
            else:
                vertical += 1
        orientations = ["horizontal_cut"] * horizontal + ["vertical_cut"] * vertical
    RNG.shuffle(orientations)
    return orientations


def _sample_spec(orientation: str, used_areas: set[int], height_budget: int) -> dict[str, object] | None:
    for _ in range(SPEC_ATTEMPTS):
        if orientation == "horizontal_cut":
            full_h = int(RNG.integers(3, 9))
            full_w = int(RNG.integers(2, 9))
            if full_h > height_budget:
                continue
            cut = int(RNG.integers(1, full_h))
            if 2 * cut == full_h:
                continue
            parts = [(cut, full_w), (full_h - cut, full_w)]
        else:
            full_h = int(RNG.integers(2, 9))
            full_w = int(RNG.integers(3, 9))
            if full_h > height_budget:
                continue
            cut = int(RNG.integers(1, full_w))
            if 2 * cut == full_w:
                continue
            parts = [(full_h, cut), (full_h, full_w - cut)]

        area = full_h * full_w
        if area in used_areas:
            continue

        shuffled_parts = list(parts)
        RNG.shuffle(shuffled_parts)
        return {
            "orientation": orientation,
            "full_h": full_h,
            "full_w": full_w,
            "area": area,
            "parts": tuple((int(h), int(w)) for h, w in shuffled_parts),
        }
    return None


def _sample_specs(pair_count: int) -> list[dict[str, object]] | None:
    for _ in range(MAX_GEN_ATTEMPTS):
        orientations = _orientation_schedule(pair_count)
        specs: list[dict[str, object]] = []
        used_areas: set[int] = set()
        used_shapes: set[tuple[int, int]] = set()
        total_out_h = 0
        max_out_w = 0

        ok = True
        for orientation in orientations:
            remaining = MAX_GRID_SIDE - total_out_h - (len(orientations) - len(specs) - 1) * 2
            spec = _sample_spec(orientation, used_areas, remaining)
            if spec is None:
                ok = False
                break
            full_shape = (int(spec["full_h"]), int(spec["full_w"]))
            if full_shape in used_shapes:
                ok = False
                break
            specs.append(spec)
            used_areas.add(int(spec["area"]))
            used_shapes.add(full_shape)
            total_out_h += int(spec["full_h"])
            max_out_w = max(max_out_w, int(spec["full_w"]))

        if ok and total_out_h <= MAX_GRID_SIDE and max_out_w <= MAX_GRID_SIDE:
            return specs
    return None


def _can_place(blocked: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    return not blocked[top : top + height, left : left + width].any()


def _place_piece(canvas: np.ndarray, blocked: np.ndarray, color: int, height: int, width: int) -> bool:
    h, w = canvas.shape
    for _ in range(PLACEMENT_ATTEMPTS):
        top = int(RNG.integers(0, h - height + 1))
        left = int(RNG.integers(0, w - width + 1))
        if not _can_place(blocked, top, left, height, width):
            continue
        canvas[top : top + height, left : left + width] = color
        r0 = max(0, top - 1)
        c0 = max(0, left - 1)
        r1 = min(h, top + height + 1)
        c1 = min(w, left + width + 1)
        blocked[r0:r1, c0:c1] = True
        return True
    return False


def _crop_with_margins(canvas: np.ndarray) -> np.ndarray:
    coords = np.argwhere(canvas != BLACK)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)

    top_room = min(int(r0), MARGIN_CAP)
    left_room = min(int(c0), MARGIN_CAP)
    bottom_room = min(int(canvas.shape[0] - 1 - r1), MARGIN_CAP)
    right_room = min(int(canvas.shape[1] - 1 - c1), MARGIN_CAP)

    top = int(r0 - RNG.integers(0, top_room + 1))
    left = int(c0 - RNG.integers(0, left_room + 1))
    bottom = int(r1 + RNG.integers(0, bottom_room + 1))
    right = int(c1 + RNG.integers(0, right_room + 1))
    return canvas[top : bottom + 1, left : right + 1]


def _gen_attempt() -> np.ndarray | None:
    pair_count = int(RNG.integers(MIN_PAIR_COUNT, MAX_PAIR_COUNT + 1))
    specs = _sample_specs(pair_count)
    if specs is None:
        return None

    colors = list(map(int, RNG.choice(PALETTE, size=pair_count, replace=False)))
    pieces: list[tuple[int, int, int]] = []
    for color, spec in zip(colors, specs):
        for part_h, part_w in spec["parts"]:
            pieces.append((int(color), int(part_h), int(part_w)))

    order = list(range(len(pieces)))
    RNG.shuffle(order)
    canvas = np.zeros((MAX_GRID_SIDE, MAX_GRID_SIDE), dtype=int)
    blocked = np.zeros_like(canvas, dtype=bool)

    for idx in order:
        color, part_h, part_w = pieces[idx]
        if not _place_piece(canvas, blocked, color, part_h, part_w):
            return None

    grid = _crop_with_margins(canvas)
    if grid.ndim != 2 or max(grid.shape) > MAX_GRID_SIDE:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.ndim != 2 or max(out.shape) > MAX_GRID_SIDE:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {MAX_GEN_ATTEMPTS} attempts")
