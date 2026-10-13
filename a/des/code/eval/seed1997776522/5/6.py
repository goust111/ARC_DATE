import random
from dataclasses import dataclass

import numpy as np


BLACK = 0
GREY = 5
GEN_RETRY_LIMIT = 300


@dataclass
class Piece:
    color: int
    mask: np.ndarray
    cells: np.ndarray
    h: int
    w: int
    area: int
    flat_top: bool
    flat_bottom: bool
    flat_left: bool
    flat_right: bool


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _is_frame_component(grid: np.ndarray, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if int(coords.shape[0]) != 2 * (h + w) - 4:
        return False
    coord_set = {tuple(rc) for rc in coords.tolist()}
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if ((r, c) in coord_set) != on_border:
                return False
    return bool(np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == BLACK))


def _build_piece(color: int, coords: np.ndarray) -> Piece:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    rel = coords - np.array([r0, c0], dtype=int)
    mask = np.zeros((h, w), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True

    lefts = []
    rights = []
    for r in range(h):
        cols = np.flatnonzero(mask[r])
        assert cols.size > 0, "piece rows must stay contiguous through the bbox"
        lefts.append(int(cols[0]))
        rights.append(int(cols[-1]))

    tops = []
    bottoms = []
    for c in range(w):
        rows = np.flatnonzero(mask[:, c])
        assert rows.size > 0, "piece columns must stay contiguous through the bbox"
        tops.append(int(rows[0]))
        bottoms.append(int(rows[-1]))

    cells = np.argwhere(mask).astype(int)
    return Piece(
        color=int(color),
        mask=mask,
        cells=cells,
        h=h,
        w=w,
        area=int(cells.shape[0]),
        flat_top=all(v == 0 for v in tops),
        flat_bottom=all(v == h - 1 for v in bottoms),
        flat_left=all(v == 0 for v in lefts),
        flat_right=all(v == w - 1 for v in rights),
    )


def _find_frame_and_pieces(grid: np.ndarray) -> tuple[tuple[int, int, int, int], int, list[Piece]]:
    frame = None
    frame_color = None
    pieces: list[Piece] = []
    for color in np.unique(grid):
        color = int(color)
        if color == BLACK:
            continue
        for coords in _components(grid == color):
            if _is_frame_component(grid, coords):
                assert frame is None, "expected exactly one reference frame"
                frame = _bbox(coords)
                frame_color = color
            else:
                pieces.append(_build_piece(color, coords))
    assert frame is not None and frame_color is not None, "reference frame not found"
    assert pieces, "expected scattered colored fragments"
    return frame, frame_color, pieces


def _axis_positions(length: int, piece_length: int, flat_lo: bool, flat_hi: bool) -> list[int]:
    if flat_lo and flat_hi:
        return [0] if piece_length == length else []
    if flat_lo:
        return [0]
    if flat_hi:
        return [length - piece_length]
    return list(range(length - piece_length + 1))


def _candidate_placements(piece: Piece, height: int, width: int) -> list[tuple[int, int, np.ndarray]]:
    rows = _axis_positions(height, piece.h, piece.flat_top, piece.flat_bottom)
    cols = _axis_positions(width, piece.w, piece.flat_left, piece.flat_right)
    candidates: list[tuple[int, int, np.ndarray]] = []
    for top in rows:
        for left in cols:
            indices = (piece.cells[:, 0] + top) * width + (piece.cells[:, 1] + left)
            candidates.append((top, left, indices.astype(int)))
    return candidates


def _solve_packing(
    pieces: list[Piece], height: int, width: int, max_solutions: int = 1
) -> list[list[tuple[int, int]]]:
    total_area = sum(piece.area for piece in pieces)
    if total_area != height * width:
        return []

    candidates = [_candidate_placements(piece, height, width) for piece in pieces]
    if any(not cand for cand in candidates):
        return []

    occupied = np.zeros(height * width, dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(pieces)
    solutions: list[list[tuple[int, int]]] = []

    def search(placed: int) -> None:
        if len(solutions) >= max_solutions:
            return
        if placed == len(pieces):
            if bool(np.all(occupied)):
                solutions.append([(int(r), int(c)) for r, c in placements if r is not None])
            return

        best_index = None
        best_options = None
        for idx, piece_cands in enumerate(candidates):
            if placements[idx] is not None:
                continue
            options = []
            for top, left, indices in piece_cands:
                if not bool(np.any(occupied[indices])):
                    options.append((top, left, indices))
            if not options:
                return
            if best_options is None or len(options) < len(best_options):
                best_index = idx
                best_options = options
                if len(best_options) == 1:
                    break

        assert best_index is not None and best_options is not None
        for top, left, indices in best_options:
            occupied[indices] = True
            placements[best_index] = (top, left)
            search(placed + 1)
            placements[best_index] = None
            occupied[indices] = False

    search(0)
    return solutions


def _render_solution(
    pieces: list[Piece],
    placements: list[tuple[int, int]],
    height: int,
    width: int,
    frame_color: int,
) -> np.ndarray:
    out = np.full((height + 2, width + 2), BLACK, dtype=int)
    out[0, :] = frame_color
    out[-1, :] = frame_color
    out[:, 0] = frame_color
    out[:, -1] = frame_color
    for piece, (top, left) in zip(pieces, placements):
        for r, c in piece.cells:
            out[top + 1 + int(r), left + 1 + int(c)] = piece.color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered monochrome fragments by matching their complementary
    jagged borders, then draw the completed rectangle inside the grey frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_bbox, frame_color, pieces = _find_frame_and_pieces(g)
    r0, c0, r1, c1 = frame_bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    assert inner_h > 0 and inner_w > 0, "frame interior must be non-empty"

    solutions = _solve_packing(pieces, inner_h, inner_w, max_solutions=1)
    assert solutions, "fragments could not be packed into the reference rectangle"
    return _render_solution(pieces, solutions[0], inner_h, inner_w, frame_color)


def _random_profile(length: int, used: set[tuple[int, ...]], rng: random.Random) -> np.ndarray:
    min_transitions = 2 if length >= 5 else 1
    for _ in range(200):
        profile = tuple(rng.randint(0, 1) for _ in range(length))
        if 0 not in profile or 1 not in profile:
            continue
        transitions = sum(profile[i] != profile[i - 1] for i in range(1, length))
        if transitions < min_transitions:
            continue
        if profile in used:
            continue
        used.add(profile)
        return np.array(profile, dtype=int)
    raise RuntimeError("failed to sample a distinctive jagged profile")


def _make_vertical_strips(rng: random.Random) -> tuple[list[np.ndarray], np.ndarray]:
    count = rng.randint(2, 4)
    height = rng.randint(4, 7)
    widths = []
    for idx in range(count):
        base = 2 if idx in (0, count - 1) else 3
        widths.append(base + rng.randint(0, 1))

    used: set[tuple[int, ...]] = set()
    cut_profiles = [_random_profile(height, used, rng) for _ in range(count - 1)]
    colors = rng.sample([1, 2, 3, 4, 6, 7, 8, 9], count)

    sprites: list[np.ndarray] = []
    final_width = sum(widths) - (count - 1)
    assembled = np.zeros((height, final_width), dtype=int)
    filled = np.zeros((height, final_width), dtype=bool)

    x = 0
    for idx in range(count):
        left_profile = np.zeros(height, dtype=int) if idx == 0 else cut_profiles[idx - 1]
        right_gap = np.zeros(height, dtype=int) if idx == count - 1 else 1 - cut_profiles[idx]
        sprite = np.zeros((height, widths[idx]), dtype=int)
        for r in range(height):
            start = int(left_profile[r])
            stop = int(widths[idx] - right_gap[r])
            assert stop > start
            sprite[r, start:stop] = colors[idx]
            for c in range(start, stop):
                rr = r
                cc = x + c
                assert not filled[rr, cc]
                filled[rr, cc] = True
                assembled[rr, cc] = colors[idx]
        sprites.append(sprite)
        x += widths[idx] - 1

    assert bool(np.all(filled))
    return sprites, assembled


def _make_horizontal_strips(rng: random.Random) -> tuple[list[np.ndarray], np.ndarray]:
    count = rng.randint(2, 4)
    width = rng.randint(4, 7)
    heights = []
    for idx in range(count):
        base = 2 if idx in (0, count - 1) else 3
        heights.append(base + rng.randint(0, 1))

    used: set[tuple[int, ...]] = set()
    cut_profiles = [_random_profile(width, used, rng) for _ in range(count - 1)]
    colors = rng.sample([1, 2, 3, 4, 6, 7, 8, 9], count)

    sprites: list[np.ndarray] = []
    final_height = sum(heights) - (count - 1)
    assembled = np.zeros((final_height, width), dtype=int)
    filled = np.zeros((final_height, width), dtype=bool)

    y = 0
    for idx in range(count):
        top_profile = np.zeros(width, dtype=int) if idx == 0 else cut_profiles[idx - 1]
        bottom_gap = np.zeros(width, dtype=int) if idx == count - 1 else 1 - cut_profiles[idx]
        sprite = np.zeros((heights[idx], width), dtype=int)
        for c in range(width):
            start = int(top_profile[c])
            stop = int(heights[idx] - bottom_gap[c])
            assert stop > start
            sprite[start:stop, c] = colors[idx]
            for r in range(start, stop):
                rr = y + r
                cc = c
                assert not filled[rr, cc]
                filled[rr, cc] = True
                assembled[rr, cc] = colors[idx]
        sprites.append(sprite)
        y += heights[idx] - 1

    assert bool(np.all(filled))
    return sprites, assembled


def _expected_output(assembled: np.ndarray, frame_color: int) -> np.ndarray:
    h, w = assembled.shape
    out = np.full((h + 2, w + 2), BLACK, dtype=int)
    out[0, :] = frame_color
    out[-1, :] = frame_color
    out[:, 0] = frame_color
    out[:, -1] = frame_color
    out[1:-1, 1:-1] = assembled
    return out


def _empty_frame(inner_h: int, inner_w: int, frame_color: int) -> np.ndarray:
    out = np.full((inner_h + 2, inner_w + 2), BLACK, dtype=int)
    out[0, :] = frame_color
    out[-1, :] = frame_color
    out[:, 0] = frame_color
    out[:, -1] = frame_color
    return out


def _mark_reserved(mask: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> None:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(mask.shape[0], top + h + pad)
    c1 = min(mask.shape[1], left + w + pad)
    mask[r0:r1, c0:c1] = True


def _paste_sprite(grid: np.ndarray, sprite: np.ndarray, top: int, left: int) -> None:
    h, w = sprite.shape
    window = grid[top : top + h, left : left + w]
    non_black = sprite != BLACK
    window[non_black] = sprite[non_black]


def _scatter_input(
    sprites: list[np.ndarray], frame: np.ndarray, rng: random.Random
) -> np.ndarray | None:
    frame_h, frame_w = frame.shape
    max_piece_h = max(sprite.shape[0] for sprite in sprites)
    max_piece_w = max(sprite.shape[1] for sprite in sprites)
    total_bbox_area = sum(int(sprite.shape[0] * sprite.shape[1]) for sprite in sprites)

    top_space = max_piece_h + rng.randint(2, 5)
    left_space = max_piece_w + rng.randint(2, 5)
    bottom_margin = rng.randint(0, 1)
    right_margin = rng.randint(0, 1)
    grid_h = top_space + frame_h + bottom_margin
    grid_w = left_space + frame_w + right_margin

    if top_space * grid_w + frame_h * left_space < total_bbox_area + 8 * len(sprites):
        return None

    frame_top = grid_h - bottom_margin - frame_h
    frame_left = grid_w - right_margin - frame_w

    grid = np.zeros((grid_h, grid_w), dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)
    _paste_sprite(grid, frame, frame_top, frame_left)
    _mark_reserved(reserved, frame_top, frame_left, frame_h, frame_w, pad=1)

    order = list(range(len(sprites)))
    rng.shuffle(order)
    for idx in order:
        sprite = sprites[idx]
        h, w = sprite.shape
        placed = False
        for _ in range(500):
            top = rng.randint(0, grid_h - h)
            left = rng.randint(0, grid_w - w)
            if not (top + h + 1 <= frame_top or left + w + 1 <= frame_left):
                continue
            r0 = max(0, top - 1)
            c0 = max(0, left - 1)
            r1 = min(grid_h, top + h + 1)
            c1 = min(grid_w, left + w + 1)
            if bool(np.any(reserved[r0:r1, c0:c1])):
                continue
            _paste_sprite(grid, sprite, top, left)
            _mark_reserved(reserved, top, left, h, w, pad=1)
            placed = True
            break
        if not placed:
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRY_LIMIT):
        if rng.random() < 0.5:
            sprites, assembled = _make_vertical_strips(rng)
        else:
            sprites, assembled = _make_horizontal_strips(rng)

        pieces = []
        for sprite in sprites:
            color = int(sprite[sprite != BLACK][0])
            coords = np.argwhere(sprite == color).astype(int)
            pieces.append(_build_piece(color, coords))

        solutions = _solve_packing(pieces, assembled.shape[0], assembled.shape[1], max_solutions=2)
        if len(solutions) != 1:
            continue

        expected = _expected_output(assembled, GREY)
        frame = _empty_frame(assembled.shape[0], assembled.shape[1], GREY)
        grid = _scatter_input(sprites, frame, rng)
        if grid is None:
            continue

        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid

    raise RuntimeError("failed to generate a valid jagged-fragment puzzle")
