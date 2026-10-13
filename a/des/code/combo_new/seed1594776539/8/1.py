from __future__ import annotations

from collections import deque
from itertools import combinations
import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
BG = 0
FRAME_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


Coord = tuple[int, int]
Shape = tuple[Coord, ...]


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(coords: Shape | tuple[Coord, ...] | list[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: Shape | tuple[Coord, ...] | list[Coord]) -> Shape:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _shape_dims(shape: Shape) -> tuple[int, int]:
    _, _, r1, c1 = _bbox(shape)
    return int(r1 + 1), int(c1 + 1)


def _components(mask: np.ndarray) -> list[Shape]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[Shape] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            queue = deque([(r, c)])
            visited[r, c] = True
            comp: list[Coord] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(tuple(sorted(comp)))
    return comps


def _frame_bbox(component: Shape) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(component)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    expected: set[Coord] = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0 + 1, r1):
        expected.add((r, c0))
        expected.add((r, c1))
    return (r0, c0, r1, c1) if set(component) == expected else None


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for comp in _components(grid == color):
            bbox = _frame_bbox(comp)
            if bbox is not None:
                candidates.append((color, bbox))
    assert len(candidates) == 1, f"expected one rectangular frame, got {candidates}"
    return candidates[0]


def _translated_subset_matches(
    fragment_coords: Shape,
    shape_coords: list[Shape],
    bounds: tuple[int, int, int, int] | None,
) -> list[tuple[int, int, int]]:
    matches: set[tuple[int, int, int]] = set()
    for idx, shape in enumerate(shape_coords):
        shape_set = set(shape)
        deltas = {(fr - sr, fc - sc) for fr, fc in fragment_coords for sr, sc in shape}
        for dr, dc in deltas:
            if not all((fr - dr, fc - dc) in shape_set for fr, fc in fragment_coords):
                continue
            if bounds is not None:
                r0, c0, r1, c1 = bounds
                if any(
                    rr < r0 or rr > r1 or cc < c0 or cc > c1
                    for rr, cc in ((sr + dr, sc + dc) for sr, sc in shape)
                ):
                    continue
            matches.add((int(idx), int(dr), int(dc)))
    return sorted(matches)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each colored fragment inside the grey frame with its matching outside shape, recolored by the fragment.

    Input structure:
    - The background is black.
    - One monochrome grey component forms a one-pixel-thick closed rectangular frame.
    - Outside the frame are several complete monochrome 4-connected shapes of distinct geometries.
    - Inside the frame are several monochrome 4-connected fragment clusters; each fragment is a proper translated subset of one outside shape.

    Rule:
    - Detect the unique rectangular frame and split the grid into outside shapes and inside fragments.
    - For each inside fragment, find the unique outside shape for which some translation makes every fragment pixel land on shape pixels.
    - Draw that full translated shape inside the frame using the fragment's color, replacing the original fragment cells.
    - Keep the outside shapes and the grey frame unchanged.

    Key insight:
    - The inside object is not matched by color; it is matched by translated subset geometry, while the fragment color only specifies the recoloring of the completed shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    bg = _background_color(g)
    frame_color, (fr0, fc0, fr1, fc1) = _find_frame(g, bg)
    assert fr1 - fr0 >= 2 and fc1 - fc0 >= 2
    inner_bounds = (fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1)
    assert inner_bounds[0] <= inner_bounds[2] and inner_bounds[1] <= inner_bounds[3]

    inside_mask = np.zeros_like(g, dtype=bool)
    inside_mask[fr0 + 1 : fr1, fc0 + 1 : fc1] = True
    outside_mask = np.ones_like(g, dtype=bool)
    outside_mask[fr0 : fr1 + 1, fc0 : fc1 + 1] = False

    shapes: list[Shape] = []
    fragments: list[tuple[int, Shape]] = []
    for color in map(int, np.unique(g)):
        if color in (bg, frame_color):
            continue
        for comp in _components((g == color) & outside_mask):
            shapes.append(_normalize(comp))
        for comp in _components((g == color) & inside_mask):
            fragments.append((color, comp))

    assert shapes, "no outside shapes found"
    assert fragments, "no inside fragments found"

    claims: dict[Coord, int] = {}
    clears: list[Coord] = []
    for frag_color, frag_coords in fragments:
        matches = _translated_subset_matches(frag_coords, shapes, inner_bounds)
        assert len(matches) == 1, f"ambiguous fragment match for color {frag_color}: {matches}"
        shape_idx, dr, dc = matches[0]
        clears.extend(frag_coords)
        for sr, sc in shapes[shape_idx]:
            rr, cc = sr + dr, sc + dc
            prev = claims.get((rr, cc))
            assert prev in (None, frag_color), "matched completed shapes must not overlap"
            claims[(rr, cc)] = frag_color

    out = g.copy()
    for r, c in clears:
        out[r, c] = bg
    for (r, c), color in claims.items():
        out[r, c] = color
    return out


def _is_connected(coords: tuple[Coord, ...]) -> bool:
    if not coords:
        return False
    coord_set = set(coords)
    queue = deque([coords[0]])
    seen = {coords[0]}
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt in coord_set and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen) == len(coord_set)


def _random_transform(shape: Shape) -> Shape:
    pts = list(shape)
    if random.random() < 0.5:
        pts = [(r, -c) for r, c in pts]
    turns = random.randrange(4)
    for _ in range(turns):
        pts = [(c, -r) for r, c in pts]
    return _normalize(tuple(pts))


def _random_polyomino(size: int) -> Shape | None:
    for _ in range(120):
        cells = {(0, 0)}
        while len(cells) < size:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.add(nxt)
            cells.add(random.choice(tuple(frontier)))
        shape = _random_transform(tuple(cells))
        h, w = _shape_dims(shape)
        if h < 2 or w < 2 or h > 4 or w > 4:
            continue
        if len(shape) == h * w:
            continue
        return shape
    return None


def _connected_proper_subsets(shape: Shape) -> list[Shape]:
    coords = list(shape)
    n = len(coords)
    subsets: list[Shape] = []
    for k in range(n - 1, 1, -1):
        for idxs in combinations(range(n), k):
            subset = tuple(sorted(coords[i] for i in idxs))
            if _is_connected(subset):
                subsets.append(subset)
    return subsets


def _choose_shapes_and_fragments(count: int) -> tuple[list[Shape], list[Shape]] | None:
    for _ in range(200):
        shapes: list[Shape] = []
        seen: set[Shape] = set()
        while len(shapes) < count:
            shape = _random_polyomino(random.randint(4, 6))
            if shape is None or shape in seen:
                continue
            seen.add(shape)
            shapes.append(shape)

        fragments: list[Shape] = []
        valid = True
        for idx, shape in enumerate(shapes):
            candidates = [
                subset
                for subset in _connected_proper_subsets(shape)
                if _translated_subset_matches(subset, shapes, bounds=None) == [(idx, 0, 0)]
            ]
            if not candidates:
                valid = False
                break
            best_size = max(len(subset) for subset in candidates)
            pool = [subset for subset in candidates if len(subset) >= best_size - 1]
            fragments.append(random.choice(pool))
        if valid:
            return shapes, fragments
    return None


def _can_place(
    shape: Shape,
    offset: Coord,
    grid_h: int,
    grid_w: int,
    occupied: np.ndarray,
    forbidden: np.ndarray,
) -> bool:
    dr, dc = offset
    for sr, sc in shape:
        rr, cc = sr + dr, sc + dc
        if rr < 0 or rr >= grid_h or cc < 0 or cc >= grid_w:
            return False
        if forbidden[rr, cc]:
            return False
        r0 = max(0, rr - 1)
        r1 = min(grid_h, rr + 2)
        c0 = max(0, cc - 1)
        c1 = min(grid_w, cc + 2)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _place_shapes(
    shapes: list[Shape],
    grid_h: int,
    grid_w: int,
    forbidden: np.ndarray | None = None,
    occupied_seed: np.ndarray | None = None,
    margin: int = 0,
) -> list[Coord] | None:
    forbidden_mask = forbidden if forbidden is not None else np.zeros((grid_h, grid_w), dtype=bool)
    occupied = occupied_seed.copy() if occupied_seed is not None else np.zeros((grid_h, grid_w), dtype=bool)
    positions: list[Coord | None] = [None] * len(shapes)

    order = list(range(len(shapes)))
    random.shuffle(order)
    order.sort(key=lambda idx: (-len(shapes[idx]), -_shape_dims(shapes[idx])[0] * _shape_dims(shapes[idx])[1]))

    for idx in order:
        shape = shapes[idx]
        sh, sw = _shape_dims(shape)
        if grid_h < sh + 2 * margin or grid_w < sw + 2 * margin:
            return None
        candidates: list[Coord] = []
        for r in range(margin, grid_h - sh - margin + 1):
            for c in range(margin, grid_w - sw - margin + 1):
                if _can_place(shape, (r, c), grid_h, grid_w, occupied, forbidden_mask):
                    candidates.append((r, c))
        if not candidates:
            return None
        pos = random.choice(candidates)
        positions[idx] = pos
        for sr, sc in shape:
            occupied[pos[0] + sr, pos[1] + sc] = True

    return [pos for pos in positions if pos is not None]


def _draw_shape(grid: np.ndarray, shape: Shape, offset: Coord, color: int) -> None:
    dr, dc = offset
    for r, c in shape:
        grid[dr + r, dc + c] = color


def _gen_attempt() -> np.ndarray | None:
    count = random.choice((2, 3, 4))
    picked = _choose_shapes_and_fragments(count)
    if picked is None:
        return None
    shapes, fragments = picked

    for _ in range(80):
        min_inner_h = max(max(_shape_dims(shape)[0] for shape in shapes) + 2, 9)
        min_inner_w = max(max(_shape_dims(shape)[1] for shape in shapes) + 2, 9)
        inner_h = random.randint(min_inner_h, 14)
        inner_w = random.randint(min_inner_w, 14)
        if inner_h * inner_w < sum(len(shape) for shape in shapes) * 3:
            continue
        inner_positions = _place_shapes(shapes, inner_h, inner_w, margin=1)
        if inner_positions is None:
            continue

        frame_h = inner_h + 2
        frame_w = inner_w + 2
        margins_ok = False
        for _ in range(80):
            top = random.randint(3, 7)
            bottom = random.randint(3, 7)
            left = random.randint(3, 7)
            right = random.randint(3, 7)
            grid_h = top + frame_h + bottom
            grid_w = left + frame_w + right
            if grid_h <= 30 and grid_w <= 30:
                margins_ok = True
                break
        if not margins_ok:
            continue

        fr0, fc0 = top, left
        fr1, fc1 = fr0 + frame_h - 1, fc0 + frame_w - 1
        grid = np.full((grid_h, grid_w), BG, dtype=int)
        grid[fr0, fc0 : fc1 + 1] = FRAME_COLOR
        grid[fr1, fc0 : fc1 + 1] = FRAME_COLOR
        grid[fr0 : fr1 + 1, fc0] = FRAME_COLOR
        grid[fr0 : fr1 + 1, fc1] = FRAME_COLOR

        frame_forbidden = np.zeros((grid_h, grid_w), dtype=bool)
        frame_forbidden[fr0 : fr1 + 1, fc0 : fc1 + 1] = True
        occupied_seed = np.zeros((grid_h, grid_w), dtype=bool)
        occupied_seed[fr0, fc0 : fc1 + 1] = True
        occupied_seed[fr1, fc0 : fc1 + 1] = True
        occupied_seed[fr0 : fr1 + 1, fc0] = True
        occupied_seed[fr0 : fr1 + 1, fc1] = True

        outer_positions = _place_shapes(
            shapes,
            grid_h,
            grid_w,
            forbidden=frame_forbidden,
            occupied_seed=occupied_seed,
            margin=1,
        )
        if outer_positions is None:
            continue

        colors = list(PALETTE)
        random.shuffle(colors)
        outer_colors = colors[:count]
        fragment_colors = colors[count : 2 * count]

        for shape, offset, color in zip(shapes, outer_positions, outer_colors):
            _draw_shape(grid, shape, offset, color)

        expected = grid.copy()
        inner_origin = (fr0 + 1, fc0 + 1)
        for shape, fragment, offset, color in zip(shapes, fragments, inner_positions, fragment_colors):
            full_offset = (inner_origin[0] + offset[0], inner_origin[1] + offset[1])
            _draw_shape(expected, shape, full_offset, color)
            _draw_shape(grid, fragment, full_offset, color)

        solved = main(grid)
        if np.array_equal(solved, expected):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
