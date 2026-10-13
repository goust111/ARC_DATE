import hashlib
import random
from collections import Counter

import numpy as np


BLACK = 0
RED = 2
INPUT_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
GRID_HEIGHT_RANGE = (11, 16)
GRID_WIDTH_RANGE = (11, 16)
TOTAL_PIXEL_RANGE = (8, 14)
GEN_ATTEMPTS = 600

_RECENT_OUTPUT_SIGNATURES: set[str] = set()


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _on_diagonal(sr: int, sc: int, r: int, c: int) -> bool:
    return abs(r - sr) == abs(c - sc)


def _diagonal_coords(shape: tuple[int, int], sr: int, sc: int, kind: str) -> list[tuple[int, int]]:
    h, w = shape
    coords: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) == (sr, sc):
                continue
            if kind == "main" and (r - sr == c - sc):
                coords.append((r, c))
            elif kind == "anti" and (r - sr == -(c - sc)):
                coords.append((r, c))
    return coords


def _draw_full_diagonals(out: np.ndarray, sr: int, sc: int, color: int) -> None:
    h, w = out.shape
    for r in range(h):
        dc = abs(r - sr)
        for c in (sc - dc, sc + dc):
            if 0 <= c < w:
                out[r, c] = color


def _paint_frame(out: np.ndarray, r: int, c: int) -> None:
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if _in_bounds(out.shape, rr, cc) and (rr, cc) != (r, c):
                out[rr, cc] = RED


def _solve_from_source(
    grid: np.ndarray, source: tuple[int, int], source_color: int
) -> np.ndarray:
    sr, sc = source
    out = np.zeros_like(grid)
    _draw_full_diagonals(out, sr, sc, source_color)

    framed_centers: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(grid != BLACK):
        rr = int(r)
        cc = int(c)
        if (rr, cc) == (sr, sc):
            continue
        if _on_diagonal(sr, sc, rr, cc):
            framed_centers.append((rr, cc, int(grid[rr, cc])))

    for r, c, _color in framed_centers:
        _paint_frame(out, r, c)

    for r, c, color in framed_centers:
        out[r, c] = color
    out[sr, sc] = source_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the uniquely frequent color to anchor two full diagonals, then frame only the other pixels lying on those diagonals.

    Input structure:
    - The grid has a black background with 8 to 14 isolated non-black pixels.
    - Exactly one non-black color appears once; its lone pixel is the source.
    - Other colors appear at least twice, and some non-source pixels lie on the source's diagonals while others are off them.

    Rule:
    - Count non-black color frequencies and locate the color that occurs exactly once; its pixel is the source.
    - Draw the two full diagonals through that source using the source color, extending to the grid edges.
    - For each other non-black pixel, keep it only if it lies on either source diagonal; around each kept pixel draw a clipped red 3x3 frame whose center stays the pixel's original color.
    - Erase every non-source pixel off the diagonals, and do not frame the source itself.

    Key insight:
    - Frequency identifies the diagonal origin, but geometric membership on the two diagonals decides which scattered pixels survive and receive red frames.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    non_black = g[g != BLACK]
    assert non_black.size > 0, "expected at least one non-black pixel"
    counts = Counter(map(int, non_black))
    unique_colors = [color for color, count in counts.items() if count == 1]
    assert len(unique_colors) == 1, "expected exactly one unique non-black color"

    source_color = int(unique_colors[0])
    coords = np.argwhere(g == source_color)
    assert coords.shape[0] == 1, "expected the unique color to occupy exactly one cell"
    source = tuple(map(int, coords[0]))
    return _solve_from_source(g, source, source_color)


def _sample_source(shape: tuple[int, int]) -> tuple[int, int] | None:
    h, w = shape
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in corners:
                continue
            if _diagonal_coords(shape, r, c, "main") and _diagonal_coords(shape, r, c, "anti"):
                candidates.append((r, c))
    return random.choice(candidates) if candidates else None


def _frame_would_clip(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _diag_order(
    coords: list[tuple[int, int]],
    source: tuple[int, int],
    shape: tuple[int, int],
    *,
    prefer_clipped: bool,
    prefer_near: bool,
) -> list[tuple[int, int]]:
    sr, sc = source
    h, w = shape

    def key(cell: tuple[int, int]) -> tuple[int, int, float]:
        r, c = cell
        dist = abs(r - sr)
        clipped_priority = 0 if prefer_clipped and _frame_would_clip((h, w), r, c) else 1
        near_priority = dist if prefer_near else -dist
        return (clipped_priority, near_priority, random.random())

    return sorted(coords, key=key)


def _pick_off_diagonal_points(
    shape: tuple[int, int],
    source: tuple[int, int],
    occupied: set[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]] | None:
    h, w = shape
    sr, sc = source
    candidates: list[tuple[tuple[int, int], tuple[int, int, int, float]]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in occupied or _on_diagonal(sr, sc, r, c):
                continue
            if any((nr, nc) in occupied for nr, nc in _neighbors4(r, c)):
                continue
            diag_gap = abs(abs(r - sr) - abs(c - sc))
            same_axis = 0 if (r == sr or c == sc) else 1
            edge = 0 if _frame_would_clip(shape, r, c) else 1
            candidates.append(((r, c), (diag_gap, same_axis, edge, random.random())))

    candidates.sort(key=lambda item: item[1])
    chosen: list[tuple[int, int]] = []
    used = set(occupied)
    for cell, _score in candidates:
        if any((nr, nc) in used for nr, nc in _neighbors4(*cell)):
            continue
        chosen.append(cell)
        used.add(cell)
        if len(chosen) == count:
            return chosen
    return None


def _distribute_counts(total: int, color_count: int) -> list[int]:
    counts = [2] * color_count
    remaining = total - 2 * color_count
    for _ in range(remaining):
        counts[random.randrange(color_count)] += 1
    random.shuffle(counts)
    return counts


def _assign_colors(
    survivors_main: list[tuple[int, int]],
    survivors_anti: list[tuple[int, int]],
    distractors: list[tuple[int, int]],
    other_colors: list[int],
) -> dict[tuple[int, int], int] | None:
    survivors = survivors_main + survivors_anti
    total = len(survivors) + len(distractors)
    counts = dict(zip(other_colors, _distribute_counts(total, len(other_colors))))

    assignment: dict[tuple[int, int], int] = {}
    remaining_survivors = survivors[:]
    remaining_main = survivors_main[:]
    remaining_anti = survivors_anti[:]
    remaining_distractors = distractors[:]

    mixed_candidates = [color for color, count in counts.items() if count >= 2]
    if not remaining_survivors or not remaining_distractors or not mixed_candidates:
        return None
    mixed_color = random.choice(mixed_candidates)
    s = random.choice(remaining_survivors)
    d = random.choice(remaining_distractors)
    assignment[s] = mixed_color
    assignment[d] = mixed_color
    counts[mixed_color] -= 2
    remaining_survivors.remove(s)
    remaining_distractors.remove(d)
    if s in remaining_main:
        remaining_main.remove(s)
    if s in remaining_anti:
        remaining_anti.remove(s)

    cross_candidates = [color for color, count in counts.items() if count >= 2]
    if remaining_main and remaining_anti and cross_candidates:
        cross_color = random.choice(cross_candidates)
        m = random.choice(remaining_main)
        a = random.choice(remaining_anti)
        assignment[m] = cross_color
        assignment[a] = cross_color
        counts[cross_color] -= 2
        remaining_survivors.remove(m)
        remaining_survivors.remove(a)
        remaining_main.remove(m)
        remaining_anti.remove(a)

    leftover_cells = remaining_survivors + remaining_distractors
    color_bag: list[int] = []
    for color, count in counts.items():
        color_bag.extend([color] * count)
    if len(color_bag) != len(leftover_cells):
        return None

    random.shuffle(leftover_cells)
    random.shuffle(color_bag)
    for cell, color in zip(leftover_cells, color_bag):
        assignment[cell] = color

    return assignment


def _build_grid(
    shape: tuple[int, int],
    source: tuple[int, int],
    source_color: int,
    assignment: dict[tuple[int, int], int],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    sr, sc = source
    grid[sr, sc] = source_color
    for (r, c), color in assignment.items():
        grid[r, c] = int(color)
    return grid


def _validate_input(grid: np.ndarray) -> tuple[tuple[int, int], int, list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid != BLACK)]
    assert TOTAL_PIXEL_RANGE[0] <= len(coords) <= TOTAL_PIXEL_RANGE[1]
    for r, c in coords:
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(grid.shape, nr, nc):
                assert grid[nr, nc] == BLACK

    colors = [int(v) for v in np.unique(grid) if int(v) != BLACK]
    assert 3 <= len(colors) <= 4

    counts = Counter(map(int, grid[grid != BLACK]))
    unique = [color for color, count in counts.items() if count == 1]
    assert len(unique) == 1
    source_color = int(unique[0])
    source = tuple(map(int, np.argwhere(grid == source_color)[0]))
    assert source not in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}

    sr, sc = source
    survivors: list[tuple[int, int]] = []
    distractors: list[tuple[int, int]] = []
    main_hits = 0
    anti_hits = 0
    for r, c in coords:
        if (r, c) == source:
            continue
        if _on_diagonal(sr, sc, r, c):
            survivors.append((r, c))
            if r - sr == c - sc:
                main_hits += 1
            if r - sr == -(c - sc):
                anti_hits += 1
        else:
            distractors.append((r, c))
    assert main_hits >= 1
    assert anti_hits >= 1
    assert distractors

    shared_color = False
    for color in colors:
        if color == source_color:
            continue
        on_count = sum(grid[r, c] == color for r, c in survivors)
        off_count = sum(grid[r, c] == color for r, c in distractors)
        if on_count and off_count:
            shared_color = True
            break
    assert shared_color, "need at least one repeated color on and off the diagonals"
    return source, source_color, survivors, distractors


def _validate_output_semantics(
    grid: np.ndarray,
    expected: np.ndarray,
    source: tuple[int, int],
    source_color: int,
    survivors: list[tuple[int, int]],
    distractors: list[tuple[int, int]],
) -> None:
    sr, sc = source
    actual = main(grid)
    assert np.array_equal(actual, expected)

    for r, c in distractors:
        assert actual[r, c] == BLACK
    for r, c in survivors:
        assert actual[r, c] == grid[r, c]
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if not _in_bounds(actual.shape, rr, cc):
                    continue
                if (rr, cc) == (r, c):
                    continue
                if (rr, cc) in survivors or (rr, cc) == source:
                    continue
                assert actual[rr, cc] == RED

    for r in range(actual.shape[0]):
        for c in range(actual.shape[1]):
            if (r, c) == source or (r, c) in survivors:
                continue
            if _on_diagonal(sr, sc, r, c):
                framed = any(abs(r - rr) <= 1 and abs(c - cc) <= 1 for rr, cc in survivors)
                assert actual[r, c] == (RED if framed else source_color)


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    shape = (h, w)
    source = _sample_source(shape)
    if source is None:
        return None

    total_pixels = random.randint(*TOTAL_PIXEL_RANGE)
    non_source_total = total_pixels - 1
    color_count = random.choice((3, 4))
    other_color_count = color_count - 1
    if non_source_total < 2 * other_color_count + 1:
        return None

    source_color = random.choice(INPUT_COLORS)
    other_colors = random.sample([c for c in INPUT_COLORS if c != source_color], other_color_count)

    main_candidates = _diagonal_coords(shape, source[0], source[1], "main")
    anti_candidates = _diagonal_coords(shape, source[0], source[1], "anti")
    if not main_candidates or not anti_candidates:
        return None

    max_survivors = min(5, non_source_total - 1, len(main_candidates) + len(anti_candidates))
    if max_survivors < 2:
        return None
    survivor_total = random.randint(2, max_survivors)
    main_count = random.randint(1, survivor_total - 1)
    anti_count = survivor_total - main_count
    if main_count > len(main_candidates) or anti_count > len(anti_candidates):
        return None

    prefer_clipped = random.random() < 0.45
    prefer_near = random.random() < 0.45
    main_order = _diag_order(
        main_candidates, source, shape, prefer_clipped=prefer_clipped, prefer_near=prefer_near
    )
    anti_order = _diag_order(
        anti_candidates, source, shape, prefer_clipped=prefer_clipped, prefer_near=prefer_near
    )

    survivors_main = main_order[:main_count]
    survivors_anti = anti_order[:anti_count]
    occupied = {source, *survivors_main, *survivors_anti}

    distractor_total = non_source_total - survivor_total
    distractors = _pick_off_diagonal_points(shape, source, occupied, distractor_total)
    if distractors is None:
        return None

    assignment = _assign_colors(survivors_main, survivors_anti, distractors, other_colors)
    if assignment is None:
        return None

    grid = _build_grid(shape, source, source_color, assignment)
    source2, source_color2, survivors, off = _validate_input(grid)
    expected = _solve_from_source(grid, source, source_color)
    _validate_output_semantics(grid, expected, source2, source_color2, survivors, off)

    out_sig = hashlib.md5(expected.tobytes() + str(expected.shape).encode()).hexdigest()
    if out_sig in _RECENT_OUTPUT_SIGNATURES:
        return None
    _RECENT_OUTPUT_SIGNATURES.add(out_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid non-degenerate example")
