from __future__ import annotations

import math
import random
from collections import Counter, deque

import numpy as np


BLUE = 1
GREY = 5
ORANGE = 7
PALETTE = (2, 3, 4, 6, 8, 9)
EMPTY = -1
MAX_GEN_ATTEMPTS = 800

_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _ring_coords(h: int, w: int, offset: int) -> list[tuple[int, int]]:
    r0 = c0 = offset
    r1 = h - 1 - offset
    c1 = w - 1 - offset
    coords = [(r0, c) for c in range(c0, c1 + 1)]
    coords.extend((r, c1) for r in range(r0 + 1, r1 + 1))
    if r1 > r0:
        coords.extend((r1, c) for c in range(c1 - 1, c0 - 1, -1))
    if c1 > c0:
        coords.extend((r, c0) for r in range(r1 - 1, r0, -1))
    return coords


def _corner_top_left(missing: tuple[int, int], corner: str, h: int, w: int) -> tuple[int, int]:
    mr, mc = map(int, missing)
    if corner == "TL":
        return mr, mc
    if corner == "TR":
        return mr, mc - w + 1
    if corner == "BL":
        return mr - h + 1, mc
    if corner == "BR":
        return mr - h + 1, mc - w + 1
    raise ValueError(f"unknown corner {corner}")


def _marker_cells(missing: tuple[int, int], corner: str) -> list[tuple[int, int]]:
    mr, mc = map(int, missing)
    if corner == "TL":
        return [(mr - 1, mc - 1), (mr - 1, mc), (mr, mc - 1)]
    if corner == "TR":
        return [(mr - 1, mc), (mr - 1, mc + 1), (mr, mc + 1)]
    if corner == "BL":
        return [(mr, mc - 1), (mr + 1, mc - 1), (mr + 1, mc)]
    if corner == "BR":
        return [(mr, mc + 1), (mr + 1, mc), (mr + 1, mc + 1)]
    raise ValueError(f"unknown corner {corner}")


def _find_marker(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int], str]:
    def parse_color(color: int) -> tuple[int, tuple[int, int], str] | None:
        coords = np.argwhere(grid == color)
        if coords.shape[0] != 3:
            return None
        r0, c0, r1, c1 = _bbox_coords(coords)
        if r1 - r0 != 1 or c1 - c0 != 1:
            return None
        full = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        pts = {tuple(map(int, rc)) for rc in coords}
        missing = list(full - pts)
        if len(missing) != 1:
            return None
        miss = missing[0]
        if miss == (r0, c0):
            corner = "BR"
        elif miss == (r0, c1):
            corner = "BL"
        elif miss == (r1, c0):
            corner = "TR"
        else:
            corner = "TL"
        return int(color), (int(miss[0]), int(miss[1])), corner

    orange_marker = parse_color(ORANGE)
    if orange_marker is not None:
        return orange_marker

    candidates: list[tuple[int, tuple[int, int], str]] = []
    for color, count in Counter(map(int, grid.flat)).items():
        if color == bg or count != 3:
            continue
        parsed = parse_color(int(color))
        if parsed is not None:
            candidates.append(parsed)

    assert len(candidates) == 1, "expected exactly one L marker"
    return candidates[0]


def _extract_pieces(grid: np.ndarray, bg: int, marker_color: int) -> list[np.ndarray]:
    mask = (grid != bg) & (grid != marker_color)
    pieces: list[np.ndarray] = []
    for coords in _components(mask):
        r0, c0, r1, c1 = _bbox_coords(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        assert patch.size == coords.shape[0], "each piece must be a solid rectangle"
        pieces.append(patch)
    assert 4 <= len(pieces) <= 6, "expected 4-6 rectangular pieces"
    return pieces


def _compatible_line(a: np.ndarray, b: np.ndarray) -> bool:
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    mask = (a != BLUE) & (b != BLUE)
    return bool(np.all(a[mask] == b[mask]))


def _fill_and_validate_rings(rect: np.ndarray) -> tuple[np.ndarray, list[int]] | None:
    h, w = rect.shape
    out = rect.copy()
    ring_colors: list[int] = []
    for offset in range((min(h, w) + 1) // 2):
        coords = _ring_coords(h, w, offset)
        values = [int(rect[r, c]) for r, c in coords]
        non_blue = {value for value in values if value != BLUE}
        if len(non_blue) != 1:
            return None
        color = int(next(iter(non_blue)))
        ring_colors.append(color)
        for r, c in coords:
            if out[r, c] == BLUE:
                out[r, c] = color
            elif out[r, c] != color:
                return None
    return out, ring_colors


def _cycle_recolor(rect: np.ndarray, ring_colors: list[int]) -> np.ndarray:
    order: list[int] = []
    for color in ring_colors:
        if color not in order:
            order.append(color)
    assert len(order) >= 2, "expected at least two distinct ring colors"
    mapping = {color: order[i - 1] for i, color in enumerate(order)}
    out = rect.copy()
    for old_color, new_color in mapping.items():
        out[rect == old_color] = new_color
    return out


def _candidate_dims(area: int, canvas_shape: tuple[int, int], missing: tuple[int, int], corner: str) -> list[tuple[int, int]]:
    gh, gw = canvas_shape
    dims: list[tuple[int, int]] = []
    for h in range(2, int(math.isqrt(area)) + 1):
        if area % h != 0:
            continue
        for hh, ww in ((h, area // h), (area // h, h)):
            if hh < 2 or ww < 2:
                continue
            tr, tc = _corner_top_left(missing, corner, int(hh), int(ww))
            if 0 <= tr and 0 <= tc and tr + hh <= gh and tc + ww <= gw:
                dims.append((int(hh), int(ww)))
    seen: set[tuple[int, int]] = set()
    ordered: list[tuple[int, int]] = []
    for dim in dims:
        if dim not in seen:
            seen.add(dim)
            ordered.append(dim)
    ordered.sort(key=lambda x: (-x[0] * x[1], -x[0], -x[1]))
    return ordered


def _solve_stats_from_pieces(
    pieces: list[np.ndarray],
    canvas_shape: tuple[int, int],
    missing: tuple[int, int],
    corner: str,
) -> tuple[int, int, np.ndarray | None]:
    total_area = int(sum(piece.shape[0] * piece.shape[1] for piece in pieces))
    dims = _candidate_dims(total_area, canvas_shape, missing, corner)
    assert dims, "no feasible rectangle dimensions"

    piece_shapes = [tuple(map(int, piece.shape)) for piece in pieces]
    piece_sigs = [((int(piece.shape[0]), int(piece.shape[1])), piece.tobytes()) for piece in pieces]
    order = sorted(range(len(pieces)), key=lambda i: (-piece_shapes[i][0] * piece_shapes[i][1], -piece_shapes[i][0], -piece_shapes[i][1], i))

    edge_count = 0
    ring_count = 0
    first_output: np.ndarray | None = None

    for h, w in dims:
        board = np.full((h, w), EMPTY, dtype=int)
        used = [False] * len(pieces)

        def next_empty() -> tuple[int, int] | None:
            coords = np.argwhere(board == EMPTY)
            if coords.size == 0:
                return None
            rr, cc = coords[0]
            return int(rr), int(cc)

        def can_place(idx: int, r0: int, c0: int) -> bool:
            piece = pieces[idx]
            ph, pw = piece.shape
            if r0 + ph > h or c0 + pw > w:
                return False
            if np.any(board[r0 : r0 + ph, c0 : c0 + pw] != EMPTY):
                return False
            if c0 > 0:
                left = board[r0 : r0 + ph, c0 - 1]
                mask = left != EMPTY
                if mask.any() and not _compatible_line(left[mask], piece[:, 0][mask]):
                    return False
            if c0 + pw < w:
                right = board[r0 : r0 + ph, c0 + pw]
                mask = right != EMPTY
                if mask.any() and not _compatible_line(right[mask], piece[:, -1][mask]):
                    return False
            if r0 > 0:
                up = board[r0 - 1, c0 : c0 + pw]
                mask = up != EMPTY
                if mask.any() and not _compatible_line(up[mask], piece[0, :][mask]):
                    return False
            if r0 + ph < h:
                down = board[r0 + ph, c0 : c0 + pw]
                mask = down != EMPTY
                if mask.any() and not _compatible_line(down[mask], piece[-1, :][mask]):
                    return False
            return True

        def place(idx: int, r0: int, c0: int) -> None:
            piece = pieces[idx]
            ph, pw = piece.shape
            board[r0 : r0 + ph, c0 : c0 + pw] = piece
            used[idx] = True

        def unplace(idx: int, r0: int, c0: int) -> None:
            piece = pieces[idx]
            ph, pw = piece.shape
            board[r0 : r0 + ph, c0 : c0 + pw] = EMPTY
            used[idx] = False

        def bt() -> None:
            nonlocal edge_count, ring_count, first_output
            nxt = next_empty()
            if nxt is None:
                edge_count += 1
                ring_data = _fill_and_validate_rings(board)
                if ring_data is not None:
                    filled, ring_colors = ring_data
                    ring_count += 1
                    out = _cycle_recolor(filled, ring_colors)
                    if first_output is None:
                        first_output = out.copy()
                return

            r0, c0 = nxt
            seen_here: set[tuple[tuple[int, int], bytes]] = set()
            for idx in order:
                if used[idx]:
                    continue
                sig = piece_sigs[idx]
                if sig in seen_here:
                    continue
                seen_here.add(sig)
                if can_place(idx, r0, c0):
                    place(idx, r0, c0)
                    bt()
                    if ring_count > 1:
                        unplace(idx, r0, c0)
                        return
                    unplace(idx, r0, c0)

        bt()
        if ring_count > 1:
            break

    return edge_count, ring_count, first_output


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the scattered rectangles under the concentric-ring constraint, then cyclically recolor the recovered rings.

    Input structure:
    - A grey background contains 4-6 disconnected solid rectangular pieces and one orange 3-pixel L marker.
    - The L marker is a 2x2 box with one missing corner; that gap indicates a corner of the target assembled rectangle.
    - Piece edges may contain blue wildcard pixels that can match any color during assembly.

    Rule:
    - Translate the pieces without rotation so they tile one rectangle anchored at the marker gap.
    - A touching edge position is compatible when the two colors are equal or at least one side is blue.
    - Among all compatible rectangle assemblies, keep the unique one whose one-cell-thick concentric rectangular rings each have exactly one non-blue color.
    - Replace every blue pixel by its ring color, then read ring colors from outside to inside, deduplicate by first appearance, and recolor each color to the previous color in that cyclic order.

    Key insight:
    - Edge matching alone is intentionally ambiguous; the correct placement is the only assembly whose full rectangle resolves into valid monochrome concentric rings after blue wildcard filling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) > 0, "expected a non-empty 2D grid"

    bg = int(Counter(map(int, g.flat)).most_common(1)[0][0])
    marker_color, missing, corner = _find_marker(g, bg)
    pieces = _extract_pieces(g, bg, marker_color)
    _edge_count, ring_count, output = _solve_stats_from_pieces(pieces, g.shape, missing, corner)
    assert ring_count == 1 and output is not None, "expected exactly one ring-valid assembly"
    return output


def _paint_rings(h: int, w: int, ring_colors: list[int]) -> np.ndarray:
    rect = np.zeros((h, w), dtype=int)
    for offset, color in enumerate(ring_colors):
        r0 = c0 = offset
        r1 = h - 1 - offset
        c1 = w - 1 - offset
        rect[r0, c0 : c1 + 1] = color
        rect[r1, c0 : c1 + 1] = color
        rect[r0 : r1 + 1, c0] = color
        rect[r0 : r1 + 1, c1] = color
    return rect


def _random_parts(total: int, count: int, min_each: int = 2) -> list[int]:
    parts = [min_each] * count
    remaining = total - count * min_each
    for _ in range(remaining):
        parts[random.randrange(count)] += 1
    random.shuffle(parts)
    return parts


def _partition_rect(h: int, w: int, piece_count: int) -> list[tuple[int, int, int, int]]:
    if piece_count == 4:
        base_rows, base_cols = 2, 2
    elif piece_count == 6:
        base_rows, base_cols = random.choice([(2, 3), (3, 2)])
    else:
        base_rows, base_cols = random.choice([(2, 3), (3, 2)])

    row_parts = _random_parts(h, base_rows, min_each=2)
    col_parts = _random_parts(w, base_cols, min_each=2)
    row_starts = [0]
    col_starts = [0]
    for size in row_parts[:-1]:
        row_starts.append(row_starts[-1] + size)
    for size in col_parts[:-1]:
        col_starts.append(col_starts[-1] + size)

    cells: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for i, rh in enumerate(row_parts):
        for j, cw in enumerate(col_parts):
            cells[(i, j)] = (row_starts[i], col_starts[j], rh, cw)

    if piece_count in (4, 6):
        return list(cells.values())

    merged: list[tuple[tuple[int, int], tuple[int, int]]] = []
    if base_rows == 2 and base_cols == 3:
        if random.random() < 0.5:
            i = random.randrange(2)
            j = random.randrange(2)
            merged.append(((i, j), (i, j + 1)))
        else:
            j = random.randrange(3)
            merged.append(((0, j), (1, j)))
    else:
        if random.random() < 0.5:
            i = random.randrange(3)
            merged.append(((i, 0), (i, 1)))
        else:
            i = random.randrange(2)
            j = random.randrange(2)
            merged.append(((i, j), (i + 1, j)))

    consumed: set[tuple[int, int]] = set()
    pieces: list[tuple[int, int, int, int]] = []
    for a, b in merged:
        consumed.add(a)
        consumed.add(b)
        r0 = min(cells[a][0], cells[b][0])
        c0 = min(cells[a][1], cells[b][1])
        r1 = max(cells[a][0] + cells[a][2], cells[b][0] + cells[b][2])
        c1 = max(cells[a][1] + cells[a][3], cells[b][1] + cells[b][3])
        pieces.append((r0, c0, r1 - r0, c1 - c0))

    for key, rect in cells.items():
        if key not in consumed:
            pieces.append(rect)
    assert len(pieces) == 5
    return pieces


def _perimeter_coords(r0: int, c0: int, h: int, w: int) -> list[tuple[int, int]]:
    coords = [(r0, c) for c in range(c0, c0 + w)]
    coords.extend((r0 + h - 1, c) for c in range(c0, c0 + w))
    coords.extend((r, c0) for r in range(r0 + 1, r0 + h - 1))
    coords.extend((r, c0 + w - 1) for r in range(r0 + 1, r0 + h - 1))
    seen: set[tuple[int, int]] = set()
    unique: list[tuple[int, int]] = []
    for cell in coords:
        if cell not in seen:
            seen.add(cell)
            unique.append(cell)
    return unique


def _make_ring_colors(num_rings: int) -> list[int]:
    distinct_target = min(num_rings, random.choice([2, 3, 3, 4]))
    palette = list(random.sample(PALETTE, distinct_target))
    ring_colors = palette[:]
    while len(ring_colors) < num_rings:
        if random.random() < 0.6:
            ring_colors.append(random.choice(palette))
        else:
            ring_colors.append(palette[len(ring_colors) % distinct_target])
    if len(set(ring_colors)) < 2:
        ring_colors[-1] = random.choice([c for c in PALETTE if c != ring_colors[0]])
    return ring_colors


def _restore_visibility(masked: np.ndarray, original: np.ndarray, piece_rects: list[tuple[int, int, int, int]], ring_colors: list[int]) -> bool:
    h, w = masked.shape
    for offset, color in enumerate(ring_colors):
        coords = _ring_coords(h, w, offset)
        visible = sum(1 for r, c in coords if masked[r, c] != BLUE)
        target = 2 if len(coords) >= 8 else 1
        if visible >= target:
            continue
        candidates = coords[:]
        random.shuffle(candidates)
        for r, c in candidates:
            if masked[r, c] == BLUE:
                masked[r, c] = color
                visible += 1
                if visible >= target:
                    break
        if visible < target:
            return False

    for r0, c0, ph, pw in piece_rects:
        piece = masked[r0 : r0 + ph, c0 : c0 + pw]
        if np.all(piece == BLUE):
            rr = r0 + ph // 2
            cc = c0 + pw // 2
            masked[rr, cc] = original[rr, cc]

    for r0, c0, ph, pw in piece_rects:
        piece = masked[r0 : r0 + ph, c0 : c0 + pw]
        perimeter = _perimeter_coords(0, 0, ph, pw)
        blue_count = sum(1 for r, c in perimeter if int(piece[r, c]) == BLUE)
        if blue_count * 2 <= len(perimeter):
            return False
        if np.all(piece == BLUE):
            return False
    return True


def _piece_arrays_from_rect(rect: np.ndarray, piece_rects: list[tuple[int, int, int, int]]) -> list[np.ndarray]:
    return [rect[r0 : r0 + h, c0 : c0 + w].copy() for r0, c0, h, w in piece_rects]


def _random_canvas_dims(rect_h: int, rect_w: int) -> tuple[int, int]:
    gh_min = max(18, rect_h + 8)
    gw_min = max(18, rect_w + 8)
    gh_max = min(26, rect_h + 15)
    gw_max = min(26, rect_w + 15)
    return random.randint(gh_min, gh_max), random.randint(gw_min, gw_max)


def _random_missing(canvas_shape: tuple[int, int], rect_h: int, rect_w: int, corner: str) -> tuple[int, int]:
    gh, gw = canvas_shape
    if corner == "TL":
        return random.randint(1, gh - rect_h), random.randint(1, gw - rect_w)
    if corner == "TR":
        return random.randint(1, gh - rect_h), random.randint(rect_w - 1, gw - 2)
    if corner == "BL":
        return random.randint(rect_h - 1, gh - 2), random.randint(1, gw - rect_w)
    return random.randint(rect_h - 1, gh - 2), random.randint(rect_w - 1, gw - 2)


def _can_place_rect(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int, padding: int = 1) -> bool:
    h, w = patch.shape
    ch, cw = canvas.shape
    if r0 < 0 or c0 < 0 or r0 + h > ch or c0 + w > cw:
        return False
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(ch, r0 + h + padding)
    cc1 = min(cw, c0 + w + padding)
    return bool(np.all(canvas[rr0:rr1, cc0:cc1] == bg))


def _paste(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    canvas[r0 : r0 + h, c0 : c0 + w] = patch


def _scatter_pieces(canvas_shape: tuple[int, int], pieces: list[np.ndarray], missing: tuple[int, int], corner: str) -> np.ndarray | None:
    gh, gw = canvas_shape
    canvas = np.full((gh, gw), GREY, dtype=int)
    for r, c in _marker_cells(missing, corner):
        if not (0 <= r < gh and 0 <= c < gw):
            return None
        canvas[r, c] = ORANGE

    order = list(range(len(pieces)))
    random.shuffle(order)
    for idx in order:
        patch = pieces[idx]
        ph, pw = patch.shape
        positions = [
            (r0, c0)
            for r0 in range(gh - ph + 1)
            for c0 in range(gw - pw + 1)
            if _can_place_rect(canvas, patch, r0, c0, GREY, padding=1)
        ]
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        _paste(canvas, patch, r0, c0)
    return canvas


def _attempt() -> tuple[np.ndarray, np.ndarray] | None:
    rect_h = random.randint(6, 11)
    rect_w = random.randint(6, 11)
    num_rings = (min(rect_h, rect_w) + 1) // 2
    if num_rings < 3:
        return None

    ring_colors = _make_ring_colors(num_rings)
    original = _paint_rings(rect_h, rect_w, ring_colors)
    piece_count = random.choice([4, 5, 6])
    piece_rects = _partition_rect(rect_h, rect_w, piece_count)
    masked = original.copy()
    for r0, c0, ph, pw in piece_rects:
        for r, c in _perimeter_coords(r0, c0, ph, pw):
            masked[r, c] = BLUE

    if not _restore_visibility(masked, original, piece_rects, ring_colors):
        return None

    pieces = _piece_arrays_from_rect(masked, piece_rects)
    corner = random.choice(["TL", "TR", "BL", "BR"])
    canvas_shape = _random_canvas_dims(rect_h, rect_w)
    missing = _random_missing(canvas_shape, rect_h, rect_w, corner)
    scattered = _scatter_pieces(canvas_shape, pieces, missing, corner)
    if scattered is None:
        return None

    edge_count, ring_count, solved = _solve_stats_from_pieces(pieces, scattered.shape, missing, corner)
    if edge_count < 2 or ring_count != 1 or solved is None:
        return None

    expected = _cycle_recolor(_fill_and_validate_rings(masked)[0], ring_colors)
    if not np.array_equal(solved, expected):
        return None
    if not np.array_equal(main(scattered), expected):
        return None

    return scattered, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _attempt()
        if result is None:
            continue
        grid, output = result
        sig = (tuple(map(int, output.shape)), output.tobytes())
        if sig in _SEEN_OUTPUTS:
            continue
        _SEEN_OUTPUTS.add(sig)
        return grid
    raise RuntimeError("failed to generate a valid input")
