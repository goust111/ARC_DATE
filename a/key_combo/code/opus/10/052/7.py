from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BACKGROUND = 0
ALL_COLORS = tuple(range(10))
GEN_ATTEMPTS = 400
GRID_SIZE_CHOICES = (6, 7, 8)
CORNERS = ("TL", "TR", "BL", "BR")


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bresenham_line(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr

    pts: list[tuple[int, int]] = []
    while True:
        pts.append((int(r0), int(c0)))
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c0 += sc
        if e2 < dc:
            err += dc
            r0 += sr
    return pts


def _corner_vertex(corner: str, size: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, size - 1
    if corner == "BL":
        return size - 1, 0
    assert corner == "BR"
    return size - 1, size - 1


def _oriented_offsets(corner: str, size: int, r: int, c: int) -> tuple[int, int]:
    vr, vc = _corner_vertex(corner, size)
    if corner == "TL":
        return r - vr, c - vc
    if corner == "TR":
        return r - vr, vc - c
    if corner == "BL":
        return vr - r, c - vc
    return vr - r, vc - c


def _triangle_outline(
    canvas_size: int, corner: str, horiz_len: int, vert_len: int
) -> tuple[set[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    side_h = 2 * horiz_len
    side_v = 2 * vert_len
    vr, vc = _corner_vertex(corner, canvas_size)

    if corner == "TL":
        h_end = (vr, vc + side_h)
        v_end = (vr + side_v, vc)
        leg_h = {(vr, c) for c in range(vc, vc + side_h + 1)}
        leg_v = {(r, vc) for r in range(vr, vr + side_v + 1)}
    elif corner == "TR":
        h_end = (vr, vc - side_h)
        v_end = (vr + side_v, vc)
        leg_h = {(vr, c) for c in range(vc - side_h, vc + 1)}
        leg_v = {(r, vc) for r in range(vr, vr + side_v + 1)}
    elif corner == "BL":
        h_end = (vr, vc + side_h)
        v_end = (vr - side_v, vc)
        leg_h = {(vr, c) for c in range(vc, vc + side_h + 1)}
        leg_v = {(r, vc) for r in range(vr - side_v, vr + 1)}
    else:
        h_end = (vr, vc - side_h)
        v_end = (vr - side_v, vc)
        leg_h = {(vr, c) for c in range(vc - side_h, vc + 1)}
        leg_v = {(r, vc) for r in range(vr - side_v, vr + 1)}

    diag = set(_bresenham_line(h_end, v_end))
    return leg_h | leg_v | diag, h_end, v_end


def _interior_mask(size: int, outline: set[tuple[int, int]]) -> np.ndarray:
    exterior = np.zeros((size, size), dtype=bool)
    blocked = np.zeros((size, size), dtype=bool)
    for r, c in outline:
        blocked[r, c] = True

    q: deque[tuple[int, int]] = deque()
    for r in range(size):
        for c in (0, size - 1):
            if not blocked[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(size):
        for r in (0, size - 1):
            if not blocked[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < size and 0 <= nc < size and not blocked[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))

    return (~blocked) & (~exterior)


def _fill_triangle(
    canvas_size: int,
    corner: str,
    bg: int,
    outline_color: int,
    motif: np.ndarray,
    horiz_len: int,
    vert_len: int,
) -> np.ndarray:
    outline, _h_end, _v_end = _triangle_outline(canvas_size, corner, horiz_len, vert_len)
    out = np.full((canvas_size, canvas_size), int(bg), dtype=int)
    for r, c in outline:
        out[r, c] = int(outline_color)

    interior = _interior_mask(canvas_size, outline)
    for r, c in np.argwhere(interior):
        lr, lc = _oriented_offsets(corner, canvas_size, int(r), int(c))
        out[r, c] = int(motif[lr % 2, lc % 2])
    return out


def _detect_corner_l(
    grid: np.ndarray, bg: int
) -> tuple[str, int, int, int]:
    h, w = grid.shape
    assert h == w, "expected a square grid"

    candidates: list[tuple[str, int, int, int]] = []

    for corner, (cr, cc) in {
        "TL": (0, 0),
        "TR": (0, w - 1),
        "BL": (h - 1, 0),
        "BR": (h - 1, w - 1),
    }.items():
        color = int(grid[cr, cc])
        if color == bg:
            continue

        if corner == "TL":
            horiz = 0
            while horiz < w and int(grid[0, horiz]) == color:
                horiz += 1
            vert = 0
            while vert < h and int(grid[vert, 0]) == color:
                vert += 1
            expected = np.zeros_like(grid, dtype=bool)
            expected[0, :horiz] = True
            expected[:vert, 0] = True
        elif corner == "TR":
            horiz = 0
            while horiz < w and int(grid[0, w - 1 - horiz]) == color:
                horiz += 1
            vert = 0
            while vert < h and int(grid[vert, w - 1]) == color:
                vert += 1
            expected = np.zeros_like(grid, dtype=bool)
            expected[0, w - horiz :] = True
            expected[:vert, w - 1] = True
        elif corner == "BL":
            horiz = 0
            while horiz < w and int(grid[h - 1, horiz]) == color:
                horiz += 1
            vert = 0
            while vert < h and int(grid[h - 1 - vert, 0]) == color:
                vert += 1
            expected = np.zeros_like(grid, dtype=bool)
            expected[h - 1, :horiz] = True
            expected[h - vert :, 0] = True
        else:
            horiz = 0
            while horiz < w and int(grid[h - 1, w - 1 - horiz]) == color:
                horiz += 1
            vert = 0
            while vert < h and int(grid[h - 1 - vert, w - 1]) == color:
                vert += 1
            expected = np.zeros_like(grid, dtype=bool)
            expected[h - 1, w - horiz :] = True
            expected[h - vert :, w - 1] = True

        if horiz >= 2 and vert >= 2 and horiz != vert and np.array_equal(grid == color, expected):
            candidates.append((corner, color, horiz, vert))

    assert len(candidates) == 1, "expected exactly one valid corner L"
    return candidates[0]


def _extract_motif(grid: np.ndarray, bg: int, l_color: int) -> np.ndarray:
    mask = (grid != bg) & (grid != l_color)
    coords = np.argwhere(mask)
    assert coords.shape == (4, 2), "expected exactly four motif pixels"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    assert int(r1 - r0) == 1 and int(c1 - c0) == 1, "motif must occupy a 2x2 box"
    motif = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()
    assert np.all(motif != bg) and np.all(motif != l_color)
    assert len(set(map(int, motif.ravel()))) == 4, "motif colors must all differ"
    return motif


def main(grid: np.ndarray) -> np.ndarray:
    """Infer a corner-anchored triangle from the L and fill it by tiling the opposite 2x2 motif.

    Input structure:
    - The input is a square grid on a black-dominant background.
    - One foreground color forms a one-pixel-thick corner L touching exactly one grid corner, with unequal horizontal and vertical arm lengths.
    - A separate 2x2 motif of four distinct non-background colors sits in the diagonally opposite region.

    Rule:
    - Read the L corner, color, and two arm lengths a and b.
    - Create a square output canvas of side max(2a, 2b) + 1, draw the corresponding right-triangle outline in the same corner, and connect the free arm endpoints with a Bresenham diagonal using the L color.
    - Use the triangle's right-angle vertex as the tiling origin, repeat the 2x2 motif periodically along that corner's row/column axes, and keep only motif pixels strictly inside the triangle while discarding motif pixels on or outside the outline.

    Key insight:
    - The L supplies only the triangle geometry, while the diagonally separated motif supplies only the interior texture, so both regions must be decoded and merged in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    corner, l_color, horiz_len, vert_len = _detect_corner_l(g, bg)
    motif = _extract_motif(g, bg, l_color)
    canvas_size = max(2 * horiz_len, 2 * vert_len) + 1
    return _fill_triangle(canvas_size, corner, bg, l_color, motif, horiz_len, vert_len)


def _motif_anchor_choices(size: int, corner: str) -> list[tuple[int, int]]:
    half = size // 2
    choices: list[tuple[int, int]] = []
    for r0 in range(size - 1):
        for c0 in range(size - 1):
            if corner == "TL" and r0 >= half and c0 >= half:
                choices.append((r0, c0))
            elif corner == "TR" and r0 >= half and c0 + 1 < size - half:
                choices.append((r0, c0))
            elif corner == "BL" and r0 + 1 < size - half and c0 >= half:
                choices.append((r0, c0))
            elif corner == "BR" and r0 + 1 < size - half and c0 + 1 < size - half:
                choices.append((r0, c0))
    return choices


def _build_input(
    size: int,
    corner: str,
    l_color: int,
    horiz_len: int,
    vert_len: int,
    motif: np.ndarray,
    motif_r0: int,
    motif_c0: int,
) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    if corner == "TL":
        grid[0, :horiz_len] = l_color
        grid[:vert_len, 0] = l_color
    elif corner == "TR":
        grid[0, size - horiz_len :] = l_color
        grid[:vert_len, size - 1] = l_color
    elif corner == "BL":
        grid[size - 1, :horiz_len] = l_color
        grid[size - vert_len :, 0] = l_color
    else:
        grid[size - 1, size - horiz_len :] = l_color
        grid[size - vert_len :, size - 1] = l_color
    grid[motif_r0 : motif_r0 + 2, motif_c0 : motif_c0 + 2] = motif
    return grid


def _non_bg_coords(grid: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(grid != BACKGROUND)}


def _touches_or_adjacent(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in b:
                    return True
    return False


def _sample_case() -> tuple[np.ndarray, dict[str, object]] | None:
    size = random.choice(GRID_SIZE_CHOICES)
    corner = random.choice(CORNERS)

    l_color = random.choice([c for c in ALL_COLORS if c != BACKGROUND])
    motif_colors = random.sample([c for c in ALL_COLORS if c not in (BACKGROUND, l_color)], 4)
    motif = np.array(motif_colors, dtype=int).reshape(2, 2)

    lengths = [(a, b) for a in range(2, size - 1) for b in range(2, size - 1) if a != b]
    random.shuffle(lengths)
    motif_choices = _motif_anchor_choices(size, corner)

    for horiz_len, vert_len in lengths:
        for motif_r0, motif_c0 in random.sample(motif_choices, k=len(motif_choices)):
            inp = _build_input(size, corner, l_color, horiz_len, vert_len, motif, motif_r0, motif_c0)
            l_cells = _non_bg_coords(np.where(inp == l_color, l_color, BACKGROUND))
            motif_cells = _non_bg_coords(np.where((inp != BACKGROUND) & (inp != l_color), 1, BACKGROUND))
            if _touches_or_adjacent(l_cells, motif_cells):
                continue
            out = main(inp)
            if max(out.shape) > 30:
                continue

            interior_colors = {int(v) for v in out.ravel() if int(v) not in (BACKGROUND, l_color)}
            if interior_colors != set(motif_colors):
                continue

            if np.count_nonzero(out == l_color) < 3:
                continue

            return inp, {
                "corner": corner,
                "l_color": l_color,
                "horiz_len": horiz_len,
                "vert_len": vert_len,
                "motif": motif.copy(),
                "motif_colors": tuple(motif_colors),
                "output_shape": out.shape,
            }
    return None


def _validate_case(inp: np.ndarray, meta: dict[str, object]) -> bool:
    try:
        out = main(inp)
    except Exception:
        return False

    corner = str(meta["corner"])
    l_color = int(meta["l_color"])
    horiz_len = int(meta["horiz_len"])
    vert_len = int(meta["vert_len"])
    motif = np.asarray(meta["motif"], dtype=int)

    if out.shape != tuple(meta["output_shape"]):
        return False

    expected_size = max(2 * horiz_len, 2 * vert_len) + 1
    if out.shape != (expected_size, expected_size):
        return False

    outline, h_end, v_end = _triangle_outline(expected_size, corner, horiz_len, vert_len)
    if any(int(out[r, c]) != l_color for r, c in outline):
        return False
    if set(_bresenham_line(h_end, v_end)) - outline:
        return False

    interior = _interior_mask(expected_size, outline)
    if not np.any(interior):
        return False
    if np.any(out[interior] == BACKGROUND) or np.any(out[interior] == l_color):
        return False

    seen = set()
    for r, c in np.argwhere(interior):
        lr, lc = _oriented_offsets(corner, expected_size, int(r), int(c))
        expected_color = int(motif[lr % 2, lc % 2])
        if int(out[r, c]) != expected_color:
            return False
        seen.add(expected_color)
    if seen != set(map(int, motif.ravel())):
        return False

    bg_cells = ~interior
    bg_cells[[r for r, _ in outline], [c for _, c in outline]] = False
    if np.any(out[bg_cells] != BACKGROUND):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _sample_case()
        if built is None:
            continue
        inp, meta = built
        if _validate_case(inp, meta):
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 7")
