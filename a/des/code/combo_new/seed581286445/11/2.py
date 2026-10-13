from __future__ import annotations

from collections import Counter, defaultdict, deque
import math

import numpy as np


_RNG = np.random.default_rng()

BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _normalize_coords(coords: list[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, c0 = min(rows), min(cols)
    norm = tuple(sorted((r - r0, c - c0) for r, c in coords))
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    return norm, h, w


def _find_components_by_color(grid: np.ndarray, bg: int) -> dict[int, list[tuple[tuple[int, int], ...]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    groups: dict[int, list[tuple[tuple[int, int], ...]]] = defaultdict(list)
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            norm, _ph, _pw = _normalize_coords(coords)
            groups[color].append(norm)
    return groups


def _tilings_for_dims(
    pieces: list[tuple[tuple[int, int], ...]],
    out_h: int,
    out_w: int,
    limit: int = 2,
) -> set[tuple[int, ...]]:
    board_area = out_h * out_w
    if sum(len(piece) for piece in pieces) != board_area:
        return set()

    piece_data = []
    for piece in pieces:
        ph = max(r for r, _ in piece) + 1
        pw = max(c for _, c in piece) + 1
        if ph > out_h or pw > out_w:
            return set()
        covers: list[list[int]] = [[] for _ in range(board_area)]
        for dr in range(out_h - ph + 1):
            for dc in range(out_w - pw + 1):
                mask = 0
                for r, c in piece:
                    idx = (r + dr) * out_w + (c + dc)
                    mask |= 1 << idx
                tmp = mask
                while tmp:
                    bit = tmp & -tmp
                    cell = bit.bit_length() - 1
                    covers[cell].append(mask)
                    tmp ^= bit
        piece_data.append((len(piece), covers))

    order = sorted(range(len(pieces)), key=lambda i: (-piece_data[i][0], pieces[i]))
    pieces = [pieces[i] for i in order]
    piece_data = [piece_data[i] for i in order]

    full_mask = (1 << board_area) - 1
    solutions: set[tuple[int, ...]] = set()
    dead_states: set[tuple[int, tuple[int, ...]]] = set()

    def dfs(occupied: int, remaining: tuple[int, ...], placed: list[int]) -> None:
        if len(solutions) >= limit:
            return
        if occupied == full_mask:
            solutions.add(tuple(sorted(placed)))
            return

        state = (occupied, remaining)
        if state in dead_states:
            return

        uncovered = full_mask ^ occupied
        bit = uncovered & -uncovered
        cell = bit.bit_length() - 1

        best_pos = -1
        best_options: list[int] | None = None
        for pos, idx in enumerate(remaining):
            options = [mask for mask in piece_data[idx][1][cell] if mask & occupied == 0]
            if not options:
                continue
            if best_options is None or len(options) < len(best_options):
                best_options = options
                best_pos = pos
                if len(best_options) == 1:
                    break

        if best_options is None:
            dead_states.add(state)
            return

        next_remaining = remaining[:best_pos] + remaining[best_pos + 1 :]
        before = len(solutions)
        for mask in best_options:
            placed.append(mask)
            dfs(occupied | mask, next_remaining, placed)
            placed.pop()
            if len(solutions) >= limit:
                return
        if len(solutions) == before:
            dead_states.add(state)

    dfs(0, tuple(range(len(pieces))), [])
    return solutions


def _solve_rectangle_dims(pieces: list[tuple[tuple[int, int], ...]]) -> tuple[int, int]:
    total = sum(len(piece) for piece in pieces)
    dims_with_solution: list[tuple[int, int]] = []
    min_h = max(max(r for r, _ in piece) + 1 for piece in pieces)
    min_w = max(max(c for _, c in piece) + 1 for piece in pieces)

    for h in range(1, math.isqrt(total) + 1):
        if total % h != 0:
            continue
        for out_h, out_w in ((h, total // h), (total // h, h)):
            if out_h < min_h or out_w < min_w:
                continue
            sols = _tilings_for_dims(pieces, out_h, out_w, limit=2)
            if len(sols) == 1:
                dims_with_solution.append((out_h, out_w))
            elif len(sols) > 1:
                raise AssertionError("ambiguous fragment assembly")

    dims_with_solution = list(dict.fromkeys(dims_with_solution))
    if len(dims_with_solution) != 1:
        raise AssertionError("rectangle dimensions are not uniquely determined")
    return dims_with_solution[0]


def _render_nested(rects: list[tuple[int, int, int]], bg: int) -> np.ndarray:
    rects = sorted(rects, key=lambda item: item[0] * item[1])
    max_h = max(h for h, _w, _color in rects)
    max_w = max(w for _h, w, _color in rects)
    out = np.full((max_h + 2, max_w + 2), bg, dtype=int)
    for h, w, color in sorted(rects, key=lambda item: item[0] * item[1], reverse=True):
        top = 1 + (max_h - h) // 2
        left = 1 + (max_w - w) // 2
        out[top : top + h, left : left + w] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Group same-color fragments into rectangles, then stack the rectangles concentrically by area.

    Input structure:
    - The background is black.
    - Several colors appear as separate monochrome 4-connected fragments scattered across the grid.
    - All fragments of the same color can be translated, without rotation or reflection, to tile one solid rectangle.
    - Different colors correspond to different rectangle areas.

    Rule:
    - Group the fragments by color and ignore their original absolute positions.
    - For each color, find the unique rectangle dimensions whose filled area can be tiled exactly by translated copies of that color's fragments.
    - Sort the reconstructed rectangles by area and draw them centered on a new canvas, with the largest rectangle first and progressively smaller rectangles painted on top.
    - The output canvas is the largest rectangle padded by a one-cell black border on every side.

    Key insight:
    - The visible fragment locations are decoys; the real structure is the translation-only rectangle assembly hidden inside each color group, and the final image depends on the reconstructed rectangle areas rather than on the input layout.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    groups = _find_components_by_color(g, bg)
    rects: list[tuple[int, int, int]] = []
    for color, pieces in groups.items():
        h, w = _solve_rectangle_dims(pieces)
        rects.append((h, w, int(color)))
    return _render_nested(rects, bg)


def _connected_partition(h: int, w: int, parts: int) -> list[tuple[tuple[int, int], ...]] | None:
    cells = [(r, c) for r in range(h) for c in range(w)]
    seed_ids = _RNG.choice(len(cells), size=parts, replace=False)
    groups = [{cells[int(idx)]} for idx in seed_ids]
    assigned = {cells[int(idx)] for idx in seed_ids}
    unassigned = set(cells) - assigned

    while unassigned:
        candidates: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
        for group_idx, group in enumerate(groups):
            for r, c in group:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    nbr = (nr, nc)
                    if 0 <= nr < h and 0 <= nc < w and nbr in unassigned:
                        candidates.append((group_idx, (r, c), nbr))
        if not candidates:
            return None
        group_idx, _src, nbr = candidates[int(_RNG.integers(len(candidates)))]
        groups[group_idx].add(nbr)
        unassigned.remove(nbr)

    pieces = []
    for group in groups:
        norm, _ph, _pw = _normalize_coords(list(group))
        pieces.append(norm)
    return pieces


def _sample_dims_chain() -> list[tuple[int, int]]:
    count = int(_RNG.integers(2, 5))
    h_parity = int(_RNG.integers(0, 2))
    w_parity = int(_RNG.integers(0, 2))
    height_choices = [v for v in range(2, 7) if v % 2 == h_parity]
    width_choices = [v for v in range(2, 7) if v % 2 == w_parity]
    base_h = int(_RNG.choice(height_choices))
    base_w = int(_RNG.choice(width_choices))
    if base_h * base_w < 6:
        return []

    dims = [(base_h, base_w)]
    cur_h, cur_w = base_h, base_w
    for _ in range(count - 1):
        cur_h += int(_RNG.choice([2, 4]))
        cur_w += int(_RNG.choice([2, 4]))
        if cur_h > 10 or cur_w > 10:
            return []
        dims.append((cur_h, cur_w))

    areas = [h * w for h, w in dims]
    if len(set(areas)) != len(areas):
        return []
    return dims


def _build_color_pieces(h: int, w: int) -> list[tuple[tuple[int, int], ...]] | None:
    area = h * w
    max_parts = min(4, area - 1)
    if max_parts < 2:
        return None
    for _ in range(120):
        parts = int(_RNG.integers(2, max_parts + 1))
        pieces = _connected_partition(h, w, parts)
        if pieces is None:
            continue
        if any(len(piece) == area for piece in pieces):
            continue
        try:
            solved_h, solved_w = _solve_rectangle_dims(list(pieces))
        except AssertionError:
            continue
        if (solved_h, solved_w) == (h, w):
            return list(pieces)
    return None


def _piece_bbox(piece: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in piece) + 1, max(c for _, c in piece) + 1


def _place_pieces_on_canvas(
    colored_pieces: list[tuple[int, tuple[tuple[int, int], ...]]],
    grid_h: int,
    grid_w: int,
) -> np.ndarray | None:
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    pieces = list(colored_pieces)
    _RNG.shuffle(pieces)

    for color, piece in pieces:
        ph, pw = _piece_bbox(piece)
        positions = [
            (r, c)
            for r in range(grid_h - ph + 1)
            for c in range(grid_w - pw + 1)
        ]
        _RNG.shuffle(positions)

        placed = False
        for top, left in positions:
            coords = [(top + r, left + c) for r, c in piece]
            if any(blocked[r, c] for r, c in coords):
                continue
            placed = True
            for r, c in coords:
                grid[r, c] = color
            for r, c in coords:
                blocked[r, c] = True
                for dr, dc in NEIGHBORS8:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid_h and 0 <= nc < grid_w:
                        blocked[nr, nc] = True
            break
        if not placed:
            return None
    return grid


def _generate_case() -> np.ndarray | None:
    dims = _sample_dims_chain()
    if not dims:
        return None

    color_count = len(dims)
    colors = list(_RNG.choice(PALETTE, size=color_count, replace=False))
    rects: list[tuple[int, int, int]] = []
    colored_pieces: list[tuple[int, tuple[tuple[int, int], ...]]] = []

    for (h, w), color in zip(dims, colors, strict=False):
        pieces = _build_color_pieces(h, w)
        if pieces is None:
            return None
        rects.append((h, w, int(color)))
        for piece in pieces:
            colored_pieces.append((int(color), piece))

    max_piece_h = max(_piece_bbox(piece)[0] for _color, piece in colored_pieces)
    max_piece_w = max(_piece_bbox(piece)[1] for _color, piece in colored_pieces)
    piece_count = len(colored_pieces)

    min_h = max(14, max_piece_h + 6, int(math.ceil(math.sqrt(piece_count * 18))))
    min_w = max(14, max_piece_w + 6, int(math.ceil(math.sqrt(piece_count * 18))))
    if min_h > 30 or min_w > 30:
        return None

    for _ in range(80):
        grid_h = int(_RNG.integers(min_h, 31))
        grid_w = int(_RNG.integers(min_w, 31))
        grid = _place_pieces_on_canvas(colored_pieces, grid_h, grid_w)
        if grid is None:
            continue
        expected = _render_nested(rects, BG)
        actual = main(grid)
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
