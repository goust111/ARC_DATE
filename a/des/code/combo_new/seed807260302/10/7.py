from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
MARKER_COLORS = (1, 3, 4, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_nodes(g: np.ndarray) -> tuple[list[tuple[int, int, int, int]], dict[tuple[int, int], int]]:
    nodes: list[tuple[int, int, int, int]] = []
    red_to_node: dict[tuple[int, int], int] = {}
    for comp in _components(g == RED):
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert len(comp) == 4
        assert r1 - r0 == 1 and c1 - c0 == 1
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == RED)
        idx = len(nodes)
        nodes.append((r0, c0, r1, c1))
        for cell in comp:
            red_to_node[cell] = idx
    assert nodes
    return nodes, red_to_node


def _trace_to_node(
    g: np.ndarray,
    marker: tuple[int, int],
    direction: tuple[int, int],
    red_to_node: dict[tuple[int, int], int],
) -> int:
    h, w = g.shape
    dr, dc = direction
    r, c = marker[0] + dr, marker[1] + dc
    grey_count = 0
    while 0 <= r < h and 0 <= c < w:
        value = int(g[r, c])
        if value == GREY:
            grey_count += 1
            r += dr
            c += dc
            continue
        if value == RED:
            assert grey_count >= 1
            assert (r, c) in red_to_node
            return red_to_node[(r, c)]
        break
    raise AssertionError("each marker ray must end at a red node")


def _wing_cells(
    node: tuple[int, int, int, int],
    outward: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = node
    dr, dc = outward
    assert dr in (-1, 1) and dc in (-1, 1)

    r_edge = r0 if dr < 0 else r1
    c_edge = c0 if dc < 0 else c1
    r_out = r0 - 1 if dr < 0 else r1 + 1
    c_out = c0 - 1 if dc < 0 else c1 + 1
    return ((r_out, c_out), (r_out, c_edge), (r_edge, c_out))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Add colored L-shaped wings to the red nodes joined by each bent grey skeleton line.

    Input structure:
    - The background is black.
    - Foreground red pixels form solid 2x2 node squares.
    - Each node pair is joined by a one-cell-wide grey path made from one
      horizontal segment and one vertical segment.
    - The grey path's corner is a single non-black, non-red, non-grey marker
      pixel whose color may differ from the other markers.

    Rule:
    - For every marker pixel, follow its horizontal and vertical grey rays to
      the two red nodes connected by that bent skeleton line.
    - The two ray directions from the marker give the inner diagonal direction
      of the bend; use the opposite diagonal as the wing direction.
    - On each connected node, paint a three-pixel L shape in the marker color
      around that outside corner, with two length-2 arms sharing the corner.
    - Keep all original red nodes, grey skeleton pixels, and marker pixels.

    Key insight:
    - The geometry of the grey bend determines the wing direction, while the
      corner marker independently supplies the wing color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    nodes, red_to_node = _find_nodes(g)

    marker_mask = (g != BLACK) & (g != RED) & (g != GREY)
    markers = [tuple(map(int, rc)) for rc in np.argwhere(marker_mask)]
    assert markers

    out = g.copy()
    for marker in markers:
        color = int(g[marker])
        rays: list[tuple[tuple[int, int], int]] = []
        for direction in DIRS4:
            rr = marker[0] + direction[0]
            cc = marker[1] + direction[1]
            if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == GREY:
                node_idx = _trace_to_node(g, marker, direction, red_to_node)
                rays.append((direction, node_idx))

        assert len(rays) == 2
        assert len({idx for _direction, idx in rays}) == 2
        assert sum(abs(direction[0]) for direction, _idx in rays) == 1
        assert sum(abs(direction[1]) for direction, _idx in rays) == 1

        inner_dr = sum(direction[0] for direction, _idx in rays)
        inner_dc = sum(direction[1] for direction, _idx in rays)
        assert inner_dr in (-1, 1) and inner_dc in (-1, 1)
        outward = (-inner_dr, -inner_dc)

        for _direction, node_idx in rays:
            for wr, wc in _wing_cells(nodes[node_idx], outward):
                assert 0 <= wr < h and 0 <= wc < w
                assert int(out[wr, wc]) in (BLACK, color)
                out[wr, wc] = color

    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 500
PIECE_ATTEMPTS = 250
GRID_MIN = 12
GRID_MAX = 30
PIECE_COUNT_RANGE = (1, 4)
LEG_LENGTH_RANGE = (2, 6)
INNER_DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _piece_geometry(
    h: int,
    w: int,
    marker: tuple[int, int],
    inner: tuple[int, int],
    v_len: int,
    h_len: int,
    color: int,
) -> tuple[dict[tuple[int, int], int], set[tuple[int, int]]] | None:
    mr, mc = marker
    vr, hc = inner
    outward = (-vr, -hc)

    if vr < 0:
        v_r0 = mr - v_len - 2
    else:
        v_r0 = mr + v_len + 1
    v_c0 = mc if outward[1] > 0 else mc - 1
    vertical_node = (v_r0, v_c0, v_r0 + 1, v_c0 + 1)

    if hc < 0:
        h_c0 = mc - h_len - 2
    else:
        h_c0 = mc + h_len + 1
    h_r0 = mr if outward[0] > 0 else mr - 1
    horizontal_node = (h_r0, h_c0, h_r0 + 1, h_c0 + 1)

    cells: dict[tuple[int, int], int] = {(mr, mc): int(color)}
    for step in range(1, v_len + 1):
        cells[(mr + vr * step, mc)] = GREY
    for step in range(1, h_len + 1):
        cells[(mr, mc + hc * step)] = GREY

    for r0, c0, r1, c1 in (vertical_node, horizontal_node):
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if (r, c) in cells and cells[(r, c)] != RED:
                    return None
                cells[(r, c)] = RED

    output_cells = set(cells)
    for node in (vertical_node, horizontal_node):
        for cell in _wing_cells(node, outward):
            if cell in cells:
                return None
            output_cells.add(cell)

    if any(not (0 <= r < h and 0 <= c < w) for r, c in output_cells):
        return None
    return cells, output_cells


def _marker_bounds(
    h: int,
    w: int,
    inner: tuple[int, int],
    v_len: int,
    h_len: int,
) -> tuple[range, range] | None:
    vr, hc = inner
    out_r, out_c = -vr, -hc

    r_min, r_max = 0, h - 1
    if vr < 0:
        r_min = max(r_min, v_len + 2)
    else:
        r_max = min(r_max, h - v_len - 3)
    if out_r > 0:
        r_max = min(r_max, h - 3)
    else:
        r_min = max(r_min, 2)

    c_min, c_max = 0, w - 1
    if hc < 0:
        c_min = max(c_min, h_len + 2)
    else:
        c_max = min(c_max, w - h_len - 3)
    if out_c > 0:
        c_max = min(c_max, w - 3)
    else:
        c_min = max(c_min, 2)

    if r_min > r_max or c_min > c_max:
        return None
    return range(r_min, r_max + 1), range(c_min, c_max + 1)


def _clear_of_existing(cells: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in occupied:
                    return False
    return True


def _place_piece(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    color: int,
) -> bool:
    h, w = grid.shape
    for _ in range(PIECE_ATTEMPTS):
        inner = random.choice(INNER_DIRECTIONS)
        v_len = random.randint(*LEG_LENGTH_RANGE)
        h_len = random.randint(*LEG_LENGTH_RANGE)
        bounds = _marker_bounds(h, w, inner, v_len, h_len)
        if bounds is None:
            continue
        r_range, c_range = bounds
        marker = (random.choice(list(r_range)), random.choice(list(c_range)))
        piece = _piece_geometry(h, w, marker, inner, v_len, h_len, color)
        if piece is None:
            continue
        cells, output_cells = piece
        if not _clear_of_existing(output_cells, occupied):
            continue
        for (r, c), value in cells.items():
            if int(grid[r, c]) != BLACK:
                break
            grid[r, c] = value
        else:
            occupied.update(output_cells)
            return True

        for (r, c), value in cells.items():
            if int(grid[r, c]) == value:
                grid[r, c] = BLACK
    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid = np.full((h, w), BLACK, dtype=int)
    occupied: set[tuple[int, int]] = set()

    max_pieces = min(PIECE_COUNT_RANGE[1], max(1, (h * w) // 130))
    piece_count = random.randint(PIECE_COUNT_RANGE[0], max_pieces)
    colors = random.sample(MARKER_COLORS, piece_count)
    random.shuffle(colors)

    for color in colors:
        if not _place_piece(grid, occupied, color):
            return None

    try:
        solved = main(grid)
    except AssertionError:
        return None

    assert solved.shape == grid.shape
    assert max(solved.shape) <= 30
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bent-skeleton wing input")
