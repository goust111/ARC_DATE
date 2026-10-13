from __future__ import annotations

import random
from collections import Counter
from typing import Iterable

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

ORTHO_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))
ALL_COLORS = list(range(1, 10))
GEN_ATTEMPTS = 600
CANVAS_RANGE = (16, 24)
RECT_H_RANGE = (8, 13)
RECT_W_RANGE = (8, 14)
HOLE_AREA_RANGE = (7, 11)
DISTRACTOR_COUNT_CHOICES = (2, 2, 3, 4)
MARGIN = 1


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox_from_coords(coords: Iterable[Coord]) -> BBox:
    pts = list(coords)
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _normalize_coords(coords: Iterable[Coord]) -> tuple[Coord, ...]:
    pts = sorted((int(r), int(c)) for r, c in coords)
    r0, c0, _, _ = _bbox_from_coords(pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _coords_to_patch(coords: Iterable[Coord], color: int) -> np.ndarray:
    norm = _normalize_coords(coords)
    h = 1 + max(r for r, _ in norm)
    w = 1 + max(c for _, c in norm)
    patch = np.zeros((h, w), dtype=int)
    for r, c in norm:
        patch[r, c] = int(color)
    return patch


def _mask_components(mask: np.ndarray) -> list[list[Coord]]:
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": tuple(sorted(coords)),
                    "bbox": _bbox_from_coords(coords),
                    "area": len(coords),
                }
            )
    return comps


def _orientations(coords: Iterable[Coord]) -> list[tuple[Coord, ...]]:
    pts = list(_normalize_coords(coords))
    transforms = (
        lambda r, c: (r, c),
        lambda r, c: (r, -c),
        lambda r, c: (-r, c),
        lambda r, c: (-r, -c),
        lambda r, c: (c, r),
        lambda r, c: (c, -r),
        lambda r, c: (-c, r),
        lambda r, c: (-c, -r),
    )
    out: list[tuple[Coord, ...]] = []
    seen: set[tuple[Coord, ...]] = set()
    for tf in transforms:
        variant = _normalize_coords(tf(r, c) for r, c in pts)
        if variant not in seen:
            seen.add(variant)
            out.append(variant)
    return out


def _placements_in_shape(coords: Iterable[Coord], shape: tuple[int, int], hole_set: set[Coord]) -> list[tuple[Coord, ...]]:
    out: list[tuple[Coord, ...]] = []
    seen: set[tuple[Coord, ...]] = set()
    H, W = shape
    for orient in _orientations(coords):
        h = 1 + max(r for r, _ in orient)
        w = 1 + max(c for _, c in orient)
        if h > H or w > W:
            continue
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                placed = tuple(sorted((r + r0, c + c0) for r, c in orient))
                if all(cell in hole_set for cell in placed) and placed not in seen:
                    seen.add(placed)
                    out.append(placed)
    return out


def _pair_solutions(
    hole_coords: Iterable[Coord],
    shape: tuple[int, int],
    piece_a: Iterable[Coord],
    piece_b: Iterable[Coord],
) -> list[tuple[tuple[Coord, ...], tuple[Coord, ...]]]:
    hole_set = set((int(r), int(c)) for r, c in hole_coords)
    placements_a = _placements_in_shape(piece_a, shape, hole_set)
    placements_b = _placements_in_shape(piece_b, shape, hole_set)
    b_map: dict[frozenset[Coord], list[tuple[Coord, ...]]] = {}
    for placed_b in placements_b:
        b_map.setdefault(frozenset(placed_b), []).append(placed_b)

    sols: list[tuple[tuple[Coord, ...], tuple[Coord, ...]]] = []
    for placed_a in placements_a:
        rem = hole_set - set(placed_a)
        if len(rem) != len(tuple(piece_b)):
            continue
        for placed_b in b_map.get(frozenset(rem), []):
            sols.append((placed_a, placed_b))
    return sols


def _solve_pair_fill(hole_coords: Iterable[Coord], shape: tuple[int, int], pieces: list[dict]) -> list[dict]:
    hole = tuple(sorted((int(r), int(c)) for r, c in hole_coords))
    target_area = len(hole)
    solutions: list[dict] = []
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            if pieces[i]["area"] + pieces[j]["area"] != target_area:
                continue
            pair_sols = _pair_solutions(hole, shape, pieces[i]["shape"], pieces[j]["shape"])
            for cells_a, cells_b in pair_sols:
                solutions.append(
                    {
                        "pair": (i, j),
                        "placements": (cells_a, cells_b),
                    }
                )
    return solutions


def _detect_bg(grid: np.ndarray) -> int:
    H, W = grid.shape
    border = np.concatenate([grid[0, :], grid[H - 1, :], grid[:, 0], grid[:, W - 1]])
    return Counter(map(int, border.tolist())).most_common(1)[0][0]


def _find_holed_rectangle(grid: np.ndarray, bg: int) -> dict:
    candidates: list[dict] = []
    for comp in _find_components(grid, bg):
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any((sub != color) & (sub != bg)):
            continue
        hole_mask = sub == bg
        if not hole_mask.any():
            continue
        if bool(hole_mask[0, :].any() or hole_mask[-1, :].any() or hole_mask[:, 0].any() or hole_mask[:, -1].any()):
            continue
        hole_comps = _mask_components(hole_mask)
        if len(hole_comps) != 1:
            continue
        box_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if comp["area"] + len(hole_comps[0]) != box_area:
            continue
        hole_coords = tuple(sorted((r0 + rr, c0 + cc) for rr, cc in hole_comps[0]))
        candidates.append(
            {
                "color": color,
                "bbox": comp["bbox"],
                "coords": comp["coords"],
                "hole_coords": hole_coords,
            }
        )
    assert len(candidates) == 1, "expected a unique holed rectangle"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pick the only two fragments that can interlock to fill the rectangle's hole, then erase the rest.

    Input structure:
    - One large monochromatic rectangle contains a single 4-connected background-colored hole.
    - Outside that rectangle lie several separated monochromatic polyomino fragments.
    - Exactly one pair of outside fragments is meant to fit the hole; the others are distractors.

    Rule:
    - Identify the unique pair of fragments whose areas sum to the hole area and which, after optional rotation and mirror, tile the hole with no gaps or overlaps.
    - Fill the hole with those two fragments using their original colors.
    - Clear every fragment outside the rectangle, leaving only the filled rectangle.

    Key insight:
    - Area narrows the candidates, but only one fragment pair has the complementary geometry needed to pack the irregular hole exactly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _detect_bg(g)
    rect = _find_holed_rectangle(g, bg)
    rect_cells = set(rect["coords"])
    hole_abs = tuple(sorted(rect["hole_coords"]))
    hr0, hc0, hr1, hc1 = _bbox_from_coords(hole_abs)
    hole_rel = tuple(sorted((r - hr0, c - hc0) for r, c in hole_abs))
    hole_shape = (hr1 - hr0 + 1, hc1 - hc0 + 1)

    pieces: list[dict] = []
    for comp in _find_components(g, bg):
        if set(comp["coords"]) == rect_cells:
            continue
        pieces.append(
            {
                "color": int(comp["color"]),
                "area": int(comp["area"]),
                "shape": _normalize_coords(comp["coords"]),
            }
        )

    assert len(pieces) >= 3, "expected distractor fragments outside the rectangle"
    solutions = _solve_pair_fill(hole_rel, hole_shape, pieces)
    assert len(solutions) == 1, "expected exactly one fitting fragment pair"
    solution = solutions[0]
    i, j = solution["pair"]
    placed_a, placed_b = solution["placements"]

    r0, c0, r1, c1 = rect["bbox"]
    out = np.full_like(g, bg)
    out[r0 : r1 + 1, c0 : c1 + 1] = int(rect["color"])
    for (rr, cc) in placed_a:
        out[hr0 + rr, hc0 + cc] = int(pieces[i]["color"])
    for (rr, cc) in placed_b:
        out[hr0 + rr, hc0 + cc] = int(pieces[j]["color"])
    return out


def _random_polyomino(area: int, *, max_tries: int = 80) -> tuple[Coord, ...] | None:
    for _ in range(max_tries):
        cells = {(0, 0)}
        frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
        while len(cells) < area and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            r, c = cell
            for dr, dc in ORTHO_STEPS:
                nb = (r + dr, c + dc)
                if nb not in cells:
                    frontier.add(nb)
        if len(cells) != area:
            continue
        norm = _normalize_coords(cells)
        h = 1 + max(r for r, _ in norm)
        w = 1 + max(c for _, c in norm)
        if area >= 5 and area == h * w:
            continue
        return norm
    return None


def _choose_irregular_hole() -> tuple[Coord, ...] | None:
    for _ in range(100):
        area = random.randint(*HOLE_AREA_RANGE)
        shape = _random_polyomino(area)
        if shape is None:
            continue
        h = 1 + max(r for r, _ in shape)
        w = 1 + max(c for _, c in shape)
        if h < 3 or w < 3 or h > 5 or w > 6:
            continue
        if area == h * w:
            continue
        if len({r for r, _ in shape}) < 3 or len({c for _, c in shape}) < 3:
            continue
        return shape
    return None


def _split_hole(shape: tuple[Coord, ...]) -> tuple[tuple[Coord, ...], tuple[Coord, ...]] | None:
    hole_set = set(shape)
    cells = list(shape)
    area = len(cells)
    sizes = list(range(2, area - 1))
    random.shuffle(sizes)
    for target_size in sizes:
        for _ in range(80):
            start = random.choice(cells)
            chosen = {start}
            frontier = {nb for nb in _neighbors4(*start) if nb in hole_set}
            while len(chosen) < target_size and frontier:
                nxt = random.choice(tuple(frontier))
                frontier.remove(nxt)
                chosen.add(nxt)
                for nb in _neighbors4(*nxt):
                    if nb in hole_set and nb not in chosen:
                        frontier.add(nb)
            if len(chosen) != target_size:
                continue
            rest = hole_set - chosen
            if len(_mask_components(_coords_to_mask(chosen))) != 1:
                continue
            if len(_mask_components(_coords_to_mask(rest))) != 1:
                continue
            return _normalize_coords(chosen), _normalize_coords(rest)
    return None


def _coords_to_mask(coords: Iterable[Coord]) -> np.ndarray:
    norm = _normalize_coords(coords)
    h = 1 + max(r for r, _ in norm)
    w = 1 + max(c for _, c in norm)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _random_oriented(coords: Iterable[Coord]) -> tuple[Coord, ...]:
    return random.choice(_orientations(coords))


def _pair_area_candidates(areas: list[int], target_area: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for i in range(len(areas)):
        for j in range(i + 1, len(areas)):
            if areas[i] + areas[j] == target_area:
                out.append((i, j))
    return out


def _false_pair_shapes(target_area: int, true_shapes: tuple[tuple[Coord, ...], tuple[Coord, ...]]) -> tuple[tuple[Coord, ...], tuple[Coord, ...]] | None:
    true_norm = {tuple(true_shapes[0]), tuple(true_shapes[1])}
    size_choices = list(range(2, target_area - 1))
    random.shuffle(size_choices)
    for size_a in size_choices:
        size_b = target_area - size_a
        if size_b < 2:
            continue
        for _ in range(80):
            shape_a = _random_polyomino(size_a)
            shape_b = _random_polyomino(size_b)
            if shape_a is None or shape_b is None:
                continue
            if tuple(shape_a) in true_norm or tuple(shape_b) in true_norm:
                continue
            return shape_a, shape_b
    return None


def _expanded_mask(cells: Iterable[Coord], shape: tuple[int, int]) -> np.ndarray:
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r) + dr
                cc = int(c) + dc
                if 0 <= rr < H and 0 <= cc < W:
                    mask[rr, cc] = True
    return mask


def _place_patch_random(
    grid: np.ndarray,
    forbidden: np.ndarray,
    patch: np.ndarray,
) -> tuple[int, int] | None:
    H, W = grid.shape
    h, w = patch.shape
    spots = [(r, c) for r in range(MARGIN, H - h - MARGIN + 1) for c in range(MARGIN, W - w - MARGIN + 1)]
    random.shuffle(spots)
    nz = np.argwhere(patch != 0)
    for r0, c0 in spots:
        abs_cells = [(r0 + int(rr), c0 + int(cc)) for rr, cc in nz]
        if any(forbidden[r, c] for r, c in abs_cells):
            continue
        return r0, c0
    return None


def _assemble_input(
    H: int,
    W: int,
    rect_box: BBox,
    rect_color: int,
    hole_rel: tuple[Coord, ...],
    hole_offset: Coord,
    fragment_shapes: list[tuple[Coord, ...]],
    fragment_colors: list[int],
) -> np.ndarray | None:
    grid = np.zeros((H, W), dtype=int)
    r0, c0, r1, c1 = rect_box
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(rect_color)
    hr0, hc0 = hole_offset
    for rr, cc in hole_rel:
        grid[hr0 + rr, hc0 + cc] = 0

    forbidden = np.zeros((H, W), dtype=bool)
    forbidden[max(0, r0 - 1) : min(H, r1 + 2), max(0, c0 - 1) : min(W, c1 + 2)] = True
    for shape, color in zip(fragment_shapes, fragment_colors):
        patch = _coords_to_patch(shape, color)
        loc = _place_patch_random(grid, forbidden, patch)
        if loc is None:
            return None
        pr, pc = loc
        nz = np.argwhere(patch != 0)
        abs_cells = [(pr + int(rr), pc + int(cc)) for rr, cc in nz]
        for rr, cc in abs_cells:
            grid[rr, cc] = int(color)
        forbidden |= _expanded_mask(abs_cells, (H, W))
    return grid


def _expected_output(
    H: int,
    W: int,
    rect_box: BBox,
    rect_color: int,
    hole_offset: Coord,
    placement_a: tuple[Coord, ...],
    placement_b: tuple[Coord, ...],
    color_a: int,
    color_b: int,
) -> np.ndarray:
    out = np.zeros((H, W), dtype=int)
    r0, c0, r1, c1 = rect_box
    out[r0 : r1 + 1, c0 : c1 + 1] = int(rect_color)
    hr0, hc0 = hole_offset
    for rr, cc in placement_a:
        out[hr0 + rr, hc0 + cc] = int(color_a)
    for rr, cc in placement_b:
        out[hr0 + rr, hc0 + cc] = int(color_b)
    return out


def _gen_attempt() -> np.ndarray | None:
    hole = _choose_irregular_hole()
    if hole is None:
        return None
    split = _split_hole(hole)
    if split is None:
        return None
    target_a, target_b = split
    hole_h = 1 + max(r for r, _ in hole)
    hole_w = 1 + max(c for _, c in hole)
    initial_solutions = _pair_solutions(hole, (hole_h, hole_w), target_a, target_b)
    if len(initial_solutions) != 1:
        return None

    H = random.randint(*CANVAS_RANGE)
    W = random.randint(*CANVAS_RANGE)
    rect_h = random.randint(max(RECT_H_RANGE[0], hole_h + 2), RECT_H_RANGE[1])
    rect_w = random.randint(max(RECT_W_RANGE[0], hole_w + 2), RECT_W_RANGE[1])
    if rect_h + 2 * MARGIN + 4 > H or rect_w + 2 * MARGIN + 4 > W:
        return None

    rect_r0 = random.randint(2, H - rect_h - 2)
    rect_c0 = random.randint(2, W - rect_w - 2)
    rect_box = (rect_r0, rect_c0, rect_r0 + rect_h - 1, rect_c0 + rect_w - 1)
    hole_r0 = random.randint(rect_r0 + 1, rect_r0 + rect_h - hole_h - 1)
    hole_c0 = random.randint(rect_c0 + 1, rect_c0 + rect_w - hole_w - 1)
    hole_offset = (hole_r0, hole_c0)

    false_pair = _false_pair_shapes(len(hole), (target_a, target_b))
    if false_pair is None:
        return None

    distractor_shapes = [false_pair[0], false_pair[1]]
    for _ in range(random.choice(DISTRACTOR_COUNT_CHOICES) - 2):
        size = random.randint(1, max(2, len(hole) - 1))
        shape = _random_polyomino(size)
        if shape is None:
            return None
        distractor_shapes.append(shape)

    target_display = [_random_oriented(target_a), _random_oriented(target_b)]
    distractor_display = [_random_oriented(shape) for shape in distractor_shapes]
    fragment_shapes = target_display + distractor_display
    n_colors = 1 + len(fragment_shapes)
    if n_colors > len(ALL_COLORS):
        return None
    colors = random.sample(ALL_COLORS, n_colors)
    rect_color = colors[0]
    target_colors = colors[1:3]
    distractor_colors = colors[3:]

    grid = _assemble_input(
        H,
        W,
        rect_box,
        rect_color,
        hole,
        hole_offset,
        fragment_shapes,
        target_colors + distractor_colors,
    )
    if grid is None:
        return None

    all_pieces = [
        {"color": target_colors[0], "area": len(target_a), "shape": _normalize_coords(target_display[0])},
        {"color": target_colors[1], "area": len(target_b), "shape": _normalize_coords(target_display[1])},
    ]
    for shape, color in zip(distractor_display, distractor_colors):
        all_pieces.append({"color": color, "area": len(shape), "shape": _normalize_coords(shape)})

    area_pairs = _pair_area_candidates([piece["area"] for piece in all_pieces], len(hole))
    if len(area_pairs) < 2:
        return None

    solutions = _solve_pair_fill(hole, (hole_h, hole_w), all_pieces)
    if len(solutions) != 1 or solutions[0]["pair"] != (0, 1):
        return None

    expected = _expected_output(
        H,
        W,
        rect_box,
        rect_color,
        hole_offset,
        solutions[0]["placements"][0],
        solutions[0]["placements"][1],
        target_colors[0],
        target_colors[1],
    )
    if np.array_equal(expected, grid):
        return None
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-piece hole-filling input")
