from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
CYAN = 8

PALETTE_TOP = (BLUE, RED, GREEN)
PALETTE_BOTTOM = (YELLOW, ORANGE, CYAN)
PALETTE_BLOCK = np.array([PALETTE_TOP, PALETTE_BOTTOM], dtype=int)


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def normalize_coords(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    pts = sorted((int(r), int(c)) for r, c in coords)
    r0, c0, _, _ = bbox(pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def coords_to_mask(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> np.ndarray:
    pts = normalize_coords(list(coords))
    h = max(r for r, _ in pts) + 1
    w = max(c for _, c in pts) + 1
    out = np.zeros((h, w), dtype=bool)
    for r, c in pts:
        out[r, c] = True
    return out


def mask_signature(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, ...], ...]:
    mask = coords_to_mask(coords)
    return tuple(tuple(int(v) for v in row) for row in mask)


def components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def locate_palette(grid: np.ndarray) -> tuple[dict[int, int], tuple[int, int, int, int]]:
    h, w = grid.shape
    matches: list[tuple[int, int, int, int]] = []
    for r in range(h - 1):
        for c in range(w - 2):
            if np.array_equal(grid[r : r + 2, c : c + 3], PALETTE_BLOCK):
                matches.append((r, c, r + 1, c + 2))
    assert len(matches) == 1, "expected exactly one 2x3 palette block"
    r0, c0, r1, c1 = matches[0]
    mapping = {int(grid[r0, c0 + k]): int(grid[r0 + 1, c0 + k]) for k in range(3)}
    return mapping, (r0, c0, r1, c1)


def analyze_grid(grid: np.ndarray):
    mapping, pal_box = locate_palette(grid)
    pr0, pc0, pr1, pc1 = pal_box
    palette_mask = np.zeros_like(grid, dtype=bool)
    palette_mask[pr0 : pr1 + 1, pc0 : pc1 + 1] = True

    pieces = components(grid == GREY)
    assert len(pieces) == 3, "expected three grey pieces"

    holes = []
    black_mask = (grid == BLACK) & ~palette_mask
    h, w = grid.shape
    for comp in components(black_mask):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        adj_colors = set()
        for r, c in comp:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and not palette_mask[nr, nc]:
                    color = int(grid[nr, nc])
                    if color in mapping:
                        adj_colors.add(color)
        if len(adj_colors) == 1:
            border_color = next(iter(adj_colors))
            holes.append(
                {
                    "coords": comp,
                    "shape": mask_signature(comp),
                    "border_color": border_color,
                }
            )

    assert len(holes) == 3, "expected three enclosed holes"
    for color in PALETTE_TOP:
        colored = components((grid == color) & ~palette_mask)
        assert len(colored) == 1, "expected one border component per palette key"

    piece_infos = [{"coords": comp, "shape": mask_signature(comp)} for comp in pieces]
    return mapping, pal_box, piece_infos, holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit each grey piece into the uniquely matching hole and recolor it via the host border's palette entry.

    Input structure:
    - Black is the background. Three colored hollow border sprites use blue, red, and green, and each encloses exactly one black 4-connected hole.
    - Three separate grey 4-connected pieces appear away from the sprites; each piece matches exactly one hole by translation only.
    - A 2x3 palette legend is present on the grid and pairs border colors in its top row with fill colors in its bottom row.

    Rule:
    - Read the palette legend as a border-color to fill-color mapping.
    - Find the three grey pieces and the three enclosed black holes, then match each piece to the unique hole with the same cropped shape without rotating or flipping.
    - Remove the piece from its original location and fill the matched hole with the palette color determined by that hole's surrounding border.
    - Leave the hollow borders and the palette unchanged.

    Key insight:
    - Geometry chooses which piece goes into which host hole, but the final inserted color is not intrinsic to the piece; it is looked up from the matched sprite's border color through the palette.
    """
    g = np.asarray(grid, dtype=int)
    mapping, _pal_box, pieces, holes = analyze_grid(g)

    match_counts = Counter()
    piece_to_hole: dict[int, int] = {}
    for pi, piece in enumerate(pieces):
        matches = [hi for hi, hole in enumerate(holes) if piece["shape"] == hole["shape"]]
        assert len(matches) == 1, "each piece must match exactly one hole"
        piece_to_hole[pi] = matches[0]
        match_counts[matches[0]] += 1
    assert all(match_counts[i] == 1 for i in range(3)), "hole matching must be bijective"

    out = g.copy()
    for piece in pieces:
        for r, c in piece["coords"]:
            out[r, c] = BLACK

    for pi, hi in piece_to_hole.items():
        hole = holes[hi]
        fill = mapping[int(hole["border_color"])]
        for r, c in hole["coords"]:
            out[r, c] = fill
    return out


def transform_shape(shape: tuple[tuple[int, int], ...], rot: int = 0, flip: bool = False) -> tuple[tuple[int, int], ...]:
    pts = [(r, -c) if flip else (r, c) for r, c in shape]
    for _ in range(rot % 4):
        pts = [(c, -r) for r, c in pts]
    return normalize_coords(pts)


def free_class_key(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    variants = [transform_shape(shape, rot=k, flip=f) for f in (False, True) for k in range(4)]
    return min(variants)


BASE_SHAPES = {
    "L4": ((0, 0), (1, 0), (2, 0), (2, 1)),
    "Z4": ((0, 0), (0, 1), (1, 1), (1, 2)),
    "T4": ((0, 1), (1, 0), (1, 1), (1, 2)),
}

ORIENTATIONS = {
    name: sorted({transform_shape(shape, rot=k, flip=f) for f in (False, True) for k in range(4)})
    for name, shape in BASE_SHAPES.items()
}


def make_ring(hole_shape: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    hole = set(hole_shape)
    ring = set()
    for r, c in hole:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if (nr, nc) not in hole:
                    ring.add((nr, nc))
    all_pts = normalize_coords(ring | hole)
    all_set = set(all_pts)
    hole_norm = normalize_coords(hole)
    # Shift hole into the same normalized frame as the ring.
    r_min = min(r for r, _ in ring | hole)
    c_min = min(c for _, c in ring | hole)
    hole_frame = tuple(sorted((r - r_min, c - c_min) for r, c in hole))
    ring_frame = tuple(sorted(all_set - set(hole_frame)))
    return ring_frame, hole_frame


def shift_coords(coords: tuple[tuple[int, int], ...], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in coords]


def can_place(occupied: np.ndarray, coords: list[tuple[int, int]], pad: int) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w) or occupied[r, c]:
            return False
    for r, c in coords:
        for rr in range(r - pad, r + pad + 1):
            for cc in range(c - pad, c + pad + 1):
                if 0 <= rr < h and 0 <= cc < w and occupied[rr, cc]:
                    return False
    return True


def reserve(occupied: np.ndarray, coords: list[tuple[int, int]], pad: int) -> None:
    h, w = occupied.shape
    for r, c in coords:
        for rr in range(r - pad, r + pad + 1):
            for cc in range(c - pad, c + pad + 1):
                if 0 <= rr < h and 0 <= cc < w:
                    occupied[rr, cc] = True


def place_object(
    rng: random.Random,
    occupied: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    row_max: int,
    pad: int,
) -> list[tuple[int, int]] | None:
    h, w = occupied.shape
    sh = max(r for r, _ in shape) + 1
    sw = max(c for _, c in shape) + 1
    if sh > row_max or sw > w - 2:
        return None
    for _ in range(300):
        r0 = rng.randint(1, row_max - sh)
        c0 = rng.randint(1, w - sw - 1)
        placed = shift_coords(shape, r0, c0)
        if can_place(occupied, placed, pad):
            reserve(occupied, placed, pad)
            return placed
    return None


def validate_generated(grid: np.ndarray) -> bool:
    try:
        mapping, _pal_box, pieces, holes = analyze_grid(grid)
        if set(mapping) != set(PALETTE_TOP):
            return False
        piece_shapes = [piece["shape"] for piece in pieces]
        hole_shapes = [hole["shape"] for hole in holes]
        if sorted(piece_shapes) != sorted(hole_shapes):
            return False
        exact_counts = [sum(shape == other for other in hole_shapes) for shape in piece_shapes]
        if exact_counts != [1, 1, 1]:
            return False
        areas = [sum(sum(row) for row in shape) for shape in hole_shapes]
        if max(Counter(areas).values()) < 2:
            return False
        free_counts = Counter(free_class_key(normalize_coords([(r, c) for r, row in enumerate(shape) for c, v in enumerate(row) if v])) for shape in hole_shapes)
        if max(free_counts.values()) < 2:
            return False
        out = main(grid)
        for piece in pieces:
            for r, c in piece["coords"]:
                if int(out[r, c]) != BLACK:
                    return False
        for hole in holes:
            fill = mapping[int(hole["border_color"])]
            if any(int(out[r, c]) != fill for r, c in hole["coords"]):
                return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        h = rng.randint(18, 24)
        w = rng.randint(18, 24)
        grid = np.zeros((h, w), dtype=int)

        palette_col = rng.randint(1, w - 4)
        grid[h - 2 : h, palette_col : palette_col + 3] = PALETTE_BLOCK

        occupied = np.zeros((h, w), dtype=bool)
        reserve(occupied, [(r, c) for r in range(h - 2, h) for c in range(palette_col, palette_col + 3)], 1)

        pair_family = rng.choice(["L4", "Z4"])
        pair_shapes = rng.sample(ORIENTATIONS[pair_family], 2)
        third_family = rng.choice([name for name in ORIENTATIONS if name != pair_family])
        third_shape = rng.choice(ORIENTATIONS[third_family])
        shapes = pair_shapes + [third_shape]
        rng.shuffle(shapes)

        sprite_specs = []
        ok = True
        row_limit = h - 5
        for border_color, shape in zip(rng.sample(list(PALETTE_TOP), 3), shapes):
            ring_shape, hole_shape = make_ring(shape)
            ring_coords = place_object(rng, occupied, ring_shape, row_limit, pad=1)
            if ring_coords is None:
                ok = False
                break
            rr0, cc0, _, _ = bbox(ring_coords)
            ring_norm = normalize_coords(ring_coords)
            hole_coords = shift_coords(hole_shape, rr0, cc0)
            sprite_specs.append((border_color, ring_coords, hole_coords, shape))
        if not ok:
            continue

        for border_color, ring_coords, _hole_coords, _shape in sprite_specs:
            for r, c in ring_coords:
                grid[r, c] = border_color

        for _border_color, _ring_coords, _hole_coords, shape in sprite_specs:
            piece_coords = place_object(rng, occupied, shape, row_limit, pad=2)
            if piece_coords is None:
                ok = False
                break
            for r, c in piece_coords:
                grid[r, c] = GREY
        if not ok:
            continue

        if validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
