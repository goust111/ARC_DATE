from __future__ import annotations

from collections import Counter, deque
import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
NAME_TO_DIR = {
    "E": (0, 1),
    "SE": (1, 1),
    "S": (1, 0),
    "SW": (1, -1),
    "W": (0, -1),
    "NW": (-1, -1),
    "N": (-1, 0),
    "NE": (-1, 1),
}


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _flood_from_border(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if _in_bounds(h, w, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(h, w, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _detect_gap_cells(grid: np.ndarray, color: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    fills: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BG:
                continue

            neighbors: list[tuple[int, int]] = []
            valid = True
            for dr, dc in DIRS8:
                nr, nc = r + dr, c + dc
                if not _in_bounds(h, w, nr, nc):
                    continue
                if int(grid[nr, nc]) != color:
                    continue
                neighbors.append((dr, dc))
                rr, cc = nr + dr, nc + dc
                if not (_in_bounds(h, w, rr, cc) and int(grid[rr, cc]) == color):
                    valid = False
                    break

            if valid and len(neighbors) == 2:
                fills.append((r, c))
    return fills


def _fill_enclosed_regions(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    bg_mask = out == BG
    outside = _flood_from_border(bg_mask)
    interior = bg_mask & ~outside

    for comp in _components4(interior):
        border_colors: Counter[int] = Counter()
        for r, c in comp:
            for dr, dc in DIRS8:
                nr, nc = r + dr, c + dc
                if _in_bounds(out.shape[0], out.shape[1], nr, nc):
                    col = int(out[nr, nc])
                    if col != BG:
                        border_colors[col] += 1
        if not border_colors:
            continue
        fill_color = border_colors.most_common(1)[0][0]
        for r, c in comp:
            out[r, c] = fill_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group same-colored line fragments, bridge 1-cell gaps to recover closed polygons,
    then flood-fill each polygon interior with that color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        for r, c in _detect_gap_cells(g, color):
            out[r, c] = color

    return _fill_enclosed_regions(out)


def _rotate_and_reflect(points: list[tuple[int, int]], rotation: int, mirror: bool) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r, c in points:
        rr, cc = r, c
        for _ in range(rotation):
            rr, cc = cc, -rr
        if mirror:
            cc = -cc
        out.append((rr, cc))
    return out


def _normalize(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in points)
    min_c = min(c for _, c in points)
    return [(r - min_r, c - min_c) for r, c in points]


def _outline_to_solid(outline: np.ndarray) -> np.ndarray:
    outside = _flood_from_border(~outline)
    return outline | (~outline & ~outside)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for dr, dc in DIRS8 + [(0, 0)]:
                nr, nc = r + dr, c + dc
                if _in_bounds(h, w, nr, nc):
                    out[nr, nc] = True
    return out


def _build_recipe(directions: list[str], frag_lengths: list[int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    pos_r, pos_c = 0, 0
    input_cells: list[tuple[int, int]] = []
    outline_cells: list[tuple[int, int]] = [(0, 0)]

    for name, frag_len in zip(directions, frag_lengths):
        dr, dc = NAME_TO_DIR[name]
        for step in range(1, frag_len + 1):
            cell = (pos_r + dr * step, pos_c + dc * step)
            input_cells.append(cell)
            outline_cells.append(cell)
        pos_r += dr * (frag_len + 1)
        pos_c += dc * (frag_len + 1)
        outline_cells.append((pos_r, pos_c))

    if (pos_r, pos_c) != (0, 0):
        raise ValueError("recipe does not close")
    if len(set(input_cells)) != len(input_cells):
        raise ValueError("input fragments overlap")
    if len(set(outline_cells)) != len(input_cells) + len(directions):
        raise ValueError("outline self-intersects")
    return input_cells, outline_cells


def _shape_from_recipe(directions: list[str], frag_lengths: list[int], rng: np.random.Generator) -> dict[str, np.ndarray] | None:
    try:
        raw_input, raw_outline = _build_recipe(directions, frag_lengths)
    except ValueError:
        return None

    rotation = int(rng.integers(0, 4))
    mirror = bool(rng.integers(0, 2))

    input_cells = _rotate_and_reflect(raw_input, rotation, mirror)
    outline_cells = _rotate_and_reflect(raw_outline, rotation, mirror)
    min_r = min(r for r, _ in outline_cells)
    min_c = min(c for _, c in outline_cells)
    input_cells = [(r - min_r, c - min_c) for r, c in input_cells]
    outline_cells = [(r - min_r, c - min_c) for r, c in outline_cells]

    h = max(r for r, _ in outline_cells) + 1
    w = max(c for _, c in outline_cells) + 1

    input_mask = np.zeros((h, w), dtype=bool)
    outline_mask = np.zeros((h, w), dtype=bool)
    for r, c in input_cells:
        input_mask[r, c] = True
    for r, c in outline_cells:
        outline_mask[r, c] = True

    solid_mask = _outline_to_solid(outline_mask)
    if int(solid_mask.sum()) <= int(outline_mask.sum()):
        return None

    input_mask = np.pad(input_mask, 1)
    solid_mask = np.pad(solid_mask, 1)
    halo_mask = _dilate8(solid_mask)
    return {"input": input_mask, "solid": solid_mask, "halo": halo_mask}


def _random_recipe(rng: np.random.Generator, segment_count: int) -> tuple[list[str], list[int]]:
    if segment_count == 3:
        t = int(rng.integers(2, 5))
        options = [
            (["E", "S", "NW"], [t, t, t]),
            (["E", "SW", "N"], [t, t, t]),
        ]
        return options[int(rng.integers(0, len(options)))]

    if segment_count == 4:
        family = int(rng.integers(0, 4))
        a = int(rng.integers(2, 5))
        b = int(rng.integers(2, 5))
        if family == 0:
            return ["E", "S", "W", "N"], [a, b, a, b]
        if family == 1:
            return ["NE", "SE", "SW", "NW"], [a, b, a, b]
        if family == 2:
            return ["E", "SE", "W", "NW"], [a, b, a, b]
        pairs = [(2, 2), (2, 4), (3, 3), (4, 2), (4, 4)]
        a, b = pairs[int(rng.integers(0, len(pairs)))]
        c = (a + b) // 2
        return ["E", "E", "SW", "NW"], [a, b, c, c]

    if segment_count == 5:
        while True:
            a = int(rng.integers(2, 5))
            b = int(rng.integers(2, 5))
            c = int(rng.integers(2, 5))
            d = a + b - c
            if 2 <= d <= 4:
                return ["E", "E", "S", "W", "NW"], [a, b, c, d, c]

    if segment_count == 6:
        family = int(rng.integers(0, 2))
        a = int(rng.integers(2, 5))
        b = int(rng.integers(2, 5))
        c = int(rng.integers(2, 5))
        if family == 0:
            return ["E", "SE", "SW", "W", "NW", "NE"], [a, b, c, a, b, c]
        return ["E", "E", "S", "W", "W", "N"], [a, b, c, a, b, c]

    raise ValueError("unsupported segment count")


def _random_shape(rng: np.random.Generator, segment_count: int) -> dict[str, np.ndarray] | None:
    for _ in range(40):
        directions, frag_lengths = _random_recipe(rng, segment_count)
        shape = _shape_from_recipe(directions, frag_lengths, rng)
        if shape is not None:
            return shape
    return None


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    window = grid[top : top + h, left : left + w]
    window[mask] = color


def _place_scene(
    shapes: list[dict[str, np.ndarray]],
    colors: list[int],
    h: int,
    w: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    order = sorted(range(len(shapes)), key=lambda i: int(shapes[i]["halo"].sum()), reverse=True)

    for idx in order:
        shape = shapes[idx]
        color = colors[idx]
        halo = shape["halo"]
        sh, sw = halo.shape
        if sh > h or sw > w:
            return None

        placed = False
        for _ in range(250):
            top = int(rng.integers(0, h - sh + 1))
            left = int(rng.integers(0, w - sw + 1))
            if np.any(reserved[top : top + sh, left : left + sw] & halo):
                continue
            reserved[top : top + sh, left : left + sw] |= halo
            _paint_mask(inp, shape["input"], top, left, color)
            _paint_mask(out, shape["solid"], top, left, color)
            placed = True
            break
        if not placed:
            return None

    return inp, out


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    num_shapes = int(rng.integers(3, 6))
    segment_counts = [int(rng.integers(3, 7)) for _ in range(num_shapes)]

    shapes: list[dict[str, np.ndarray]] = []
    for count in segment_counts:
        shape = _random_shape(rng, count)
        if shape is None:
            return None
        shapes.append(shape)

    colors = [int(v) for v in rng.choice(np.arange(1, 10), size=num_shapes, replace=False)]

    h = int(rng.integers(24, 33))
    w = int(rng.integers(24, 35))
    placed = _place_scene(shapes, colors, h, w, rng)
    if placed is None:
        return None
    inp, expected = placed

    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
