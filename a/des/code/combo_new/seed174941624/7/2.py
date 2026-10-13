from __future__ import annotations

import random
from functools import lru_cache

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
ARC_MAX = 30

# Unique-area rectangle templates; non-squares may be transposed at generation time.
RECTANGLE_OPTIONS = (
    (2, 3),  # 6
    (2, 4),  # 8
    (3, 3),  # 9
    (2, 5),  # 10
    (2, 6),  # 12
    (3, 5),  # 15
    (4, 4),  # 16
    (3, 6),  # 18
    (4, 5),  # 20
)

COLOR_COUNT_RANGE = (2, 4)
FRAGMENT_COUNT_MAX = 4
INPUT_MIN_SIZE = 14
INPUT_MAX_SIZE = 28
PARTITION_TRIES = 80
PLACEMENT_TRIES = 120
GENERATION_TRIES = 400

_RECTANGLE_CACHE: dict[tuple, tuple[int, int]] = {}


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _extract_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    sprites: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            sprite = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                sprite[rr - r0, cc - c0] = True
            sprites.append(sprite)

    return sprites


def _piece_key(piece: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    return piece.shape[0], piece.shape[1], tuple(int(x) for x in np.flatnonzero(piece))


def _ordered_pieces(pieces: list[np.ndarray]) -> list[np.ndarray]:
    return sorted(
        pieces,
        key=lambda piece: (-int(piece.sum()), -piece.shape[0], -piece.shape[1], _piece_key(piece)),
    )


def _candidate_rectangles(area: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for h in range(1, int(area**0.5) + 1):
        if area % h != 0:
            continue
        w = area // h
        out.append((h, w))
        if h != w:
            out.append((w, h))
    return sorted(out)


def _can_tile_rectangle(pieces: list[np.ndarray], out_h: int, out_w: int) -> bool:
    pieces = _ordered_pieces(pieces)
    n = len(pieces)
    cell_count = out_h * out_w
    full_mask = (1 << cell_count) - 1
    all_used = (1 << n) - 1
    placements_by_cell: list[list[tuple[int, int]]] = [[] for _ in range(cell_count)]

    for idx, piece in enumerate(pieces):
        ph, pw = piece.shape
        if ph > out_h or pw > out_w:
            return False
        coords = [(int(r), int(c)) for r, c in np.argwhere(piece)]
        for r0 in range(out_h - ph + 1):
            for c0 in range(out_w - pw + 1):
                placement_mask = 0
                covered: list[int] = []
                for dr, dc in coords:
                    flat = (r0 + dr) * out_w + (c0 + dc)
                    placement_mask |= 1 << flat
                    covered.append(flat)
                for flat in covered:
                    placements_by_cell[flat].append((idx, placement_mask))

    @lru_cache(maxsize=None)
    def dfs(filled_mask: int, used_mask: int) -> bool:
        if used_mask == all_used:
            return filled_mask == full_mask

        remaining = full_mask ^ filled_mask
        best_options: list[tuple[int, int]] | None = None
        probe = remaining

        while probe:
            bit = probe & -probe
            flat = bit.bit_length() - 1
            options = []
            for piece_idx, placement_mask in placements_by_cell[flat]:
                if (used_mask >> piece_idx) & 1:
                    continue
                if filled_mask & placement_mask:
                    continue
                options.append((piece_idx, placement_mask))

            if not options:
                return False
            if best_options is None or len(options) < len(best_options):
                best_options = options
                if len(best_options) == 1:
                    break
            probe ^= bit

        assert best_options is not None
        for piece_idx, placement_mask in best_options:
            if dfs(filled_mask | placement_mask, used_mask | (1 << piece_idx)):
                return True
        return False

    return dfs(0, 0)


def _reconstruct_rectangle_dims(pieces: list[np.ndarray]) -> tuple[int, int]:
    key = tuple(sorted(_piece_key(piece) for piece in pieces))
    cached = _RECTANGLE_CACHE.get(key)
    if cached is not None:
        return cached

    pieces = _ordered_pieces(pieces)
    area = sum(int(piece.sum()) for piece in pieces)
    solutions = []
    for out_h, out_w in _candidate_rectangles(area):
        if _can_tile_rectangle(pieces, out_h, out_w):
            solutions.append((out_h, out_w))

    assert len(solutions) == 1
    _RECTANGLE_CACHE[key] = solutions[0]
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge each color's scattered fragments into one solid rectangle, then stack the rectangles by area.

    Input structure:
    - The background is black.
    - Each non-black color appears as several separated 4-connected fragments.
    - For every color, those fragments are translation-only pieces of one hidden filled rectangle.

    Rule:
    - Group fragments by color.
    - For each color, infer the unique rectangle whose cells can be tiled exactly by translating that color's fragments without rotating them.
    - Replace the whole color group by a filled rectangle of that color.
    - Sort the reconstructed rectangles by area from largest to smallest, then place them top-to-bottom, left-aligned, with no blank rows between rectangles.
    - The output width is the widest reconstructed rectangle, and the output height is the sum of all reconstructed rectangle heights.

    Key insight:
    - Same-colored fragments are complementary parts of one rectangle, so the scene must be solved color-by-color before the final area-based stacking step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    assert colors

    specs: list[tuple[int, int, int, int]] = []
    for color in colors:
        pieces = _extract_components(g == color)
        assert pieces
        out_h, out_w = _reconstruct_rectangle_dims(pieces)
        area = sum(int(piece.sum()) for piece in pieces)
        assert area == out_h * out_w
        specs.append((area, out_h, out_w, color))

    areas = [area for area, _h, _w, _color in specs]
    assert len(areas) == len(set(areas))

    ordered = sorted(specs, key=lambda item: item[0], reverse=True)
    out_h = sum(h for _area, h, _w, _color in ordered)
    out_w = max(w for _area, _h, w, _color in ordered)
    out = np.full((out_h, out_w), BG, dtype=int)

    row = 0
    for _area, h, w, color in ordered:
        out[row : row + h, :w] = color
        row += h
    return out


def _partition_rectangle(h: int, w: int, n_parts: int) -> list[np.ndarray] | None:
    cells = [(r, c) for r in range(h) for c in range(w)]
    seeds = random.sample(cells, n_parts)
    labels = np.full((h, w), -1, dtype=int)

    for label, (r, c) in enumerate(seeds):
        labels[r, c] = label

    frontier = []
    for label, (r, c) in enumerate(seeds):
        for nr, nc in _neighbors4(r, c, h, w):
            if labels[nr, nc] == -1:
                frontier.append((label, nr, nc))

    while (labels == -1).any():
        frontier = [(label, r, c) for label, r, c in frontier if labels[r, c] == -1]
        if not frontier:
            return None
        label, r, c = random.choice(frontier)
        labels[r, c] = label
        for nr, nc in _neighbors4(r, c, h, w):
            if labels[nr, nc] == -1:
                frontier.append((label, nr, nc))

    pieces: list[np.ndarray] = []
    irregular = False
    for label in range(n_parts):
        coords = np.argwhere(labels == label)
        if coords.shape[0] < 2:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        sprite = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        sprite[coords[:, 0] - r0, coords[:, 1] - c0] = True
        if int(sprite.sum()) != sprite.shape[0] * sprite.shape[1]:
            irregular = True
        pieces.append(sprite)

    if not irregular and random.random() < 0.7:
        return None
    return pieces


def _make_fragments_for_rectangle(h: int, w: int) -> list[np.ndarray] | None:
    area = h * w
    max_parts = min(FRAGMENT_COUNT_MAX, area // 2)
    counts = list(range(2, max_parts + 1))
    random.shuffle(counts)

    for n_parts in counts:
        for _ in range(PARTITION_TRIES):
            pieces = _partition_rectangle(h, w, n_parts)
            if pieces is None:
                continue
            try:
                dims = _reconstruct_rectangle_dims(pieces)
            except AssertionError:
                continue
            if dims == (h, w):
                return pieces
    return None


def _sample_group_specs() -> list[tuple[int, int, int]]:
    n_colors = random.randint(*COLOR_COUNT_RANGE)
    dims = random.sample(list(RECTANGLE_OPTIONS), n_colors)
    specs = []
    for h, w in dims:
        if h != w and random.random() < 0.5:
            h, w = w, h
        specs.append((h * w, h, w))
    return specs


def _free_positions(blocked: np.ndarray, sprite: np.ndarray) -> list[tuple[int, int]]:
    h, w = blocked.shape
    sh, sw = sprite.shape
    positions = []
    for r in range(h - sh + 1):
        for c in range(w - sw + 1):
            r0 = max(0, r - 1)
            r1 = min(h, r + sh + 1)
            c0 = max(0, c - 1)
            c1 = min(w, c + sw + 1)
            if not blocked[r0:r1, c0:c1].any():
                positions.append((r, c))
    return positions


def _scatter_fragments(fragments: list[tuple[int, np.ndarray]]) -> np.ndarray | None:
    reserve = sum((sprite.shape[0] + 2) * (sprite.shape[1] + 2) for _color, sprite in fragments)
    max_h = max(sprite.shape[0] for _color, sprite in fragments)
    max_w = max(sprite.shape[1] for _color, sprite in fragments)

    for _ in range(PLACEMENT_TRIES):
        min_h = max(INPUT_MIN_SIZE, max_h + 2)
        min_w = max(INPUT_MIN_SIZE, max_w + 2)
        inp_h = random.randint(min_h, INPUT_MAX_SIZE)
        inp_w = random.randint(min_w, INPUT_MAX_SIZE)
        if inp_h * inp_w < reserve:
            continue

        grid = np.zeros((inp_h, inp_w), dtype=int)
        blocked = np.zeros((inp_h, inp_w), dtype=bool)
        order = fragments[:]
        random.shuffle(order)
        order.sort(key=lambda item: item[1].shape[0] * item[1].shape[1], reverse=True)

        ok = True
        for color, sprite in order:
            choices = _free_positions(blocked, sprite)
            if not choices:
                ok = False
                break
            r, c = random.choice(choices)
            sh, sw = sprite.shape
            patch = grid[r : r + sh, c : c + sw]
            patch[sprite] = color
            blocked[max(0, r - 1) : min(inp_h, r + sh + 1), max(0, c - 1) : min(inp_w, c + sw + 1)] = True

        if ok:
            return grid
    return None


def _expected_output(specs: list[tuple[int, int, int, int]]) -> np.ndarray:
    ordered = sorted(specs, key=lambda item: item[0], reverse=True)
    out_h = sum(h for _area, h, _w, _color in ordered)
    out_w = max(w for _area, _h, w, _color in ordered)
    out = np.zeros((out_h, out_w), dtype=int)
    row = 0
    for _area, h, w, color in ordered:
        out[row : row + h, :w] = color
        row += h
    return out


def _gen_attempt() -> np.ndarray | None:
    rectangle_specs = _sample_group_specs()
    colors = random.sample(COLORS, len(rectangle_specs))

    groups: list[tuple[int, int, int, int, list[np.ndarray]]] = []
    for color, (area, h, w) in zip(colors, rectangle_specs):
        pieces = _make_fragments_for_rectangle(h, w)
        if pieces is None:
            return None
        groups.append((color, area, h, w, pieces))

    fragments = [(color, piece) for color, _area, _h, _w, pieces in groups for piece in pieces]
    grid = _scatter_fragments(fragments)
    if grid is None:
        return None

    for color, _area, _h, _w, pieces in groups:
        if len(_extract_components(grid == color)) != len(pieces):
            return None

    expected = _expected_output([(area, h, w, color) for color, area, h, w, _pieces in groups])
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
