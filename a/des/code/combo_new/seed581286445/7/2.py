from __future__ import annotations

import numpy as np


_BG = 0
_PALETTE = tuple(range(1, 10))
_RNG = np.random.default_rng()
_MAX_GENERATION_ATTEMPTS = 400
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

_BASE_SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, -1), (0, 0), (0, 1)),
    ((0, -2), (0, -1), (0, 0), (0, 1), (0, 2)),
    ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)),
    ((-1, -1), (-1, 0), (0, 0), (1, 0), (1, 1)),
)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in _NEIGHBORS4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            components.append({"color": color, "coords": tuple(sorted(coords))})

    return components


def _mask_from_coords(coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    h = max(rows) - r0 + 1
    w = max(cols) - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _canonical_shape_key(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], bytes]:
    mask = _mask_from_coords(coords)
    best: tuple[tuple[int, int], bytes] | None = None
    for turns in range(4):
        rotated = np.rot90(mask, turns)
        for variant in (rotated, np.fliplr(rotated)):
            key = (tuple(map(int, variant.shape)), variant.tobytes())
            if best is None or key < best:
                best = key
    assert best is not None
    return best


def _anchor_from_coords(coords: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    size = len(coords)
    assert 3 <= size <= 5
    row_sum = sum(r for r, _ in coords)
    col_sum = sum(c for _, c in coords)
    assert row_sum % size == 0 and col_sum % size == 0
    anchor = (row_sum // size, col_sum // size)
    assert anchor in coords
    return anchor


def _paint_horizontal(
    base: np.ndarray,
    out: np.ndarray,
    row: int,
    c0: int,
    c1: int,
    color: int,
    bg: int,
) -> None:
    for col in range(c0, c1 + 1):
        if int(base[row, col]) == bg:
            out[row, col] = color


def _paint_vertical(
    base: np.ndarray,
    out: np.ndarray,
    col: int,
    r0: int,
    r1: int,
    color: int,
    bg: int,
) -> None:
    step = 1 if r1 >= r0 else -1
    for row in range(r0, r1 + step, step):
        if int(base[row, col]) == bg:
            out[row, col] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Pair same-shape sprites and connect each pair with a two-color L path.

    Input structure:
    - The background is black.
    - Several monochrome 4-connected sprites of size 3-5 are scattered apart.
    - Each sprite has a unique occupied geometric-center pixel.
    - Every shape appears exactly twice up to rotation and mirror reflection, and paired sprites use different colors.

    Rule:
    - Group sprites by canonical binary shape, allowing rotations and flips.
    - Within each pair, treat the left anchor as the start anchor; if both anchors share a column, use the upper one as the start.
    - Draw a horizontal segment from the start anchor to the partner column in the start sprite's color.
    - From that corner, draw a vertical segment to the partner anchor in the partner sprite's color.
    - Paint only background cells so the original sprites remain unchanged.

    Key insight:
    - The connector colors only make sense after matching sprites by rotation/flip-invariant shape, so the task requires shape pairing first and path coloring second.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    components = _find_components(g, bg)
    assert components and len(components) % 2 == 0

    groups: dict[tuple[tuple[int, int], bytes], list[dict[str, object]]] = {}
    for comp in components:
        coords = comp["coords"]
        assert isinstance(coords, tuple)
        key = _canonical_shape_key(coords)
        anchor = _anchor_from_coords(coords)
        groups.setdefault(key, []).append(
            {
                "color": int(comp["color"]),
                "anchor": anchor,
            }
        )

    out = g.copy()
    for pair in groups.values():
        assert len(pair) == 2
        pair.sort(key=lambda item: (int(item["anchor"][1]), int(item["anchor"][0])))
        start, end = pair
        r0, c0 = map(int, start["anchor"])
        r1, c1 = map(int, end["anchor"])

        if c0 <= c1:
            _paint_horizontal(base=g, out=out, row=r0, c0=c0, c1=c1, color=int(start["color"]), bg=bg)
            _paint_vertical(base=g, out=out, col=c1, r0=r0, r1=r1, color=int(end["color"]), bg=bg)
        else:
            _paint_horizontal(base=g, out=out, row=r1, c0=c1, c1=c0, color=int(end["color"]), bg=bg)
            _paint_vertical(base=g, out=out, col=c0, r0=r1, r1=r0, color=int(start["color"]), bg=bg)

    return out


def _transform_coords(
    coords: tuple[tuple[int, int], ...],
    turns: int,
    flip: bool,
) -> tuple[tuple[int, int], ...]:
    pts = list(coords)
    if flip:
        pts = [(r, -c) for r, c in pts]
    for _ in range(turns):
        pts = [(-c, r) for r, c in pts]
    return tuple(sorted((int(r), int(c)) for r, c in pts))


def _unique_transforms(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], ...]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[tuple[tuple[int, int], ...]] = []
    for flip in (False, True):
        for turns in range(4):
            transformed = _transform_coords(coords, turns=turns, flip=flip)
            if transformed not in seen:
                seen.add(transformed)
                out.append(transformed)
    return tuple(out)


_SHAPE_TRANSFORMS = tuple(_unique_transforms(shape) for shape in _BASE_SHAPES)


def _shape_extents(coords: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), max(rows), min(cols), max(cols)


def _translate(coords: tuple[tuple[int, int], ...], anchor: tuple[int, int]) -> set[tuple[int, int]]:
    ar, ac = anchor
    return {(int(ar + r), int(ac + c)) for r, c in coords}


def _choose_pair_layout(
    left_shape: tuple[tuple[int, int], ...],
    right_shape: tuple[tuple[int, int], ...],
    block_h: int,
    block_w: int,
) -> tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]] | None:
    minr0, maxr0, minc0, maxc0 = _shape_extents(left_shape)
    minr1, maxr1, minc1, maxc1 = _shape_extents(right_shape)

    row_candidates0 = range(-minr0, block_h - maxr0)
    row_candidates1 = range(-minr1, block_h - maxr1)
    col_candidates0 = [
        c for c in range(-minc0, block_w - maxc0)
        if c < block_w // 2
    ]
    col_candidates1 = [
        c for c in range(-minc1, block_w - maxc1)
        if c >= block_w // 2
    ]
    if not col_candidates0 or not col_candidates1:
        return None

    candidates: list[tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]]] = []
    for r0 in row_candidates0:
        for c0 in col_candidates0:
            anchor0 = (int(r0), int(c0))
            cells0 = _translate(left_shape, anchor0)
            for r1 in row_candidates1:
                if abs(r0 - r1) < 3:
                    continue
                for c1 in col_candidates1:
                    if c1 - c0 < 4:
                        continue
                    anchor1 = (int(r1), int(c1))
                    cells1 = _translate(right_shape, anchor1)
                    if cells0 & cells1:
                        continue

                    horizontal = {(int(r0), int(c)) for c in range(c0, c1 + 1)}
                    v_start = min(r0, r1)
                    v_end = max(r0, r1)
                    vertical = {(int(r), int(c1)) for r in range(v_start, v_end + 1)}

                    if any(cell in cells1 and cell != anchor1 for cell in horizontal):
                        continue
                    if any(cell in cells0 and cell != anchor0 for cell in vertical):
                        continue

                    candidates.append((anchor0, anchor1, cells0, cells1))

    if not candidates:
        return None
    return candidates[int(_RNG.integers(len(candidates)))]


def _sample_canvas(n_pairs: int) -> tuple[int, int, int, int, int, int, list[tuple[int, int]]]:
    if n_pairs == 2:
        layout_rows, layout_cols = ((1, 2), (2, 1), (2, 2))[int(_RNG.integers(3))]
    else:
        layout_rows, layout_cols = (2, 2)

    block_h = int(_RNG.integers(9, 13))
    block_w = int(_RNG.integers(10, 13))
    gap_h = int(_RNG.integers(1, 3))
    gap_w = int(_RNG.integers(1, 3))

    used_h = layout_rows * block_h + (layout_rows - 1) * gap_h
    used_w = layout_cols * block_w + (layout_cols - 1) * gap_w
    assert used_h <= 30 and used_w <= 30

    extra_h = int(_RNG.integers(0, 31 - used_h))
    extra_w = int(_RNG.integers(0, 31 - used_w))
    top_margin = int(_RNG.integers(0, extra_h + 1))
    left_margin = int(_RNG.integers(0, extra_w + 1))
    height = used_h + extra_h
    width = used_w + extra_w

    slots = [(r, c) for r in range(layout_rows) for c in range(layout_cols)]
    chosen = _RNG.choice(len(slots), size=n_pairs, replace=False)
    block_positions = [slots[int(i)] for i in chosen]
    return height, width, block_h, block_w, gap_h, gap_w, [(top_margin + br * (block_h + gap_h), left_margin + bc * (block_w + gap_w)) for br, bc in block_positions]


def _gen_attempt() -> np.ndarray | None:
    n_pairs = int(_RNG.integers(2, 5))
    shape_ids = _RNG.choice(len(_BASE_SHAPES), size=n_pairs, replace=False)
    colors = _RNG.choice(_PALETTE, size=2 * n_pairs, replace=False)
    height, width, block_h, block_w, _gap_h, _gap_w, block_tops = _sample_canvas(n_pairs)

    grid = np.full((height, width), _BG, dtype=int)

    for pair_idx, shape_id in enumerate(shape_ids):
        transforms = _SHAPE_TRANSFORMS[int(shape_id)]
        left_shape = transforms[int(_RNG.integers(len(transforms)))]
        right_shape = transforms[int(_RNG.integers(len(transforms)))]
        layout = _choose_pair_layout(left_shape, right_shape, block_h, block_w)
        if layout is None:
            return None

        left_anchor, right_anchor, left_cells, right_cells = layout
        base_r, base_c = block_tops[pair_idx]
        left_color = int(colors[2 * pair_idx])
        right_color = int(colors[2 * pair_idx + 1])
        if left_color == right_color:
            return None

        for r, c in left_cells:
            grid[base_r + r, base_c + c] = left_color
        for r, c in right_cells:
            grid[base_r + r, base_c + c] = right_color

        if abs(left_anchor[0] - right_anchor[0]) < 3 or right_anchor[1] - left_anchor[1] < 4:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
