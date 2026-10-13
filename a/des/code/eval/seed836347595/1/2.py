from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np


BG = 0
FRAME = 5
MARKER = 9
PIECE_COLORS = (1, 2, 3, 4, 6, 7, 8)
CORNERS = ("tl", "tr", "bl", "br")
SIDE_CORNERS = {
    "left": {"tl", "bl"},
    "right": {"tr", "br"},
    "top": {"tl", "tr"},
    "bottom": {"bl", "br"},
}
OPPOSITE_SIDE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}

PIECE_COUNT_RANGE = (2, 4)
TOP_DIM_RANGE = (3, 5)
MAX_DIM = 7
FUTURE_SPAN_MAX = MAX_DIM - 2
CANVAS_EXTRA_RANGE = (0, 2)
LAYOUT_MARGIN = 1
GEN_RETRIES = 300


@dataclass
class Piece:
    color: int
    h: int
    w: int
    r: int = 0
    c: int = 0
    attached_side: str | None = None
    marker_corner: str = ""
    notch_side: str | None = None
    notch_offset: int = 0
    notch_span: int = 0
    notch_depth: int = 0


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_bboxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            r0 = r1 = r
            c0 = c1 = c
            while q:
                cr, cc = q.popleft()
                r0 = min(r0, cr)
                r1 = max(r1, cr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            boxes.append((r0, c0, r1, c1))
    return boxes


def _scan_colors(arr: np.ndarray, exclude: set[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for v in arr.ravel():
        vv = int(v)
        if vv in exclude or vv in seen:
            continue
        seen.add(vv)
        out.append(vv)
    return out


def _bbox(piece: Piece) -> tuple[int, int, int, int]:
    return piece.r, piece.c, piece.r + piece.h - 1, piece.c + piece.w - 1


def _bbox_intersection(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return r0, c0, r1, c1


def _corner_local(h: int, w: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - 1
    if corner == "bl":
        return h - 1, 0
    if corner == "br":
        return h - 1, w - 1
    raise ValueError(corner)


def _corner_name(r: int, c: int, h: int, w: int) -> str:
    mapping = {
        (0, 0): "tl",
        (0, w - 1): "tr",
        (h - 1, 0): "bl",
        (h - 1, w - 1): "br",
    }
    return mapping[(r, c)]


def _notch_local_bbox(piece: Piece) -> tuple[int, int, int, int] | None:
    if piece.notch_side is None:
        return None
    if piece.notch_side == "left":
        return (
            piece.notch_offset,
            0,
            piece.notch_offset + piece.notch_span - 1,
            piece.notch_depth - 1,
        )
    if piece.notch_side == "right":
        return (
            piece.notch_offset,
            piece.w - piece.notch_depth,
            piece.notch_offset + piece.notch_span - 1,
            piece.w - 1,
        )
    if piece.notch_side == "top":
        return (
            0,
            piece.notch_offset,
            piece.notch_depth - 1,
            piece.notch_offset + piece.notch_span - 1,
        )
    if piece.notch_side == "bottom":
        return (
            piece.h - piece.notch_depth,
            piece.notch_offset,
            piece.h - 1,
            piece.notch_offset + piece.notch_span - 1,
        )
    raise ValueError(piece.notch_side)


def _notch_global_bbox(piece: Piece) -> tuple[int, int, int, int] | None:
    box = _notch_local_bbox(piece)
    if box is None:
        return None
    r0, c0, r1, c1 = box
    return piece.r + r0, piece.c + c0, piece.r + r1, piece.c + c1


def _visible_mask(piece: Piece) -> np.ndarray:
    mask = np.ones((piece.h, piece.w), dtype=bool)
    notch = _notch_local_bbox(piece)
    if notch is not None:
        r0, c0, r1, c1 = notch
        mask[r0 : r1 + 1, c0 : c1 + 1] = False
    return mask


def _visible_cells(piece: Piece) -> list[tuple[int, int]]:
    mask = _visible_mask(piece)
    pts = []
    for rr, cc in np.argwhere(mask):
        pts.append((piece.r + int(rr), piece.c + int(cc)))
    return pts


def _find_canvas_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates = []
    for r0, c0, r1, c1 in _component_bboxes(grid == FRAME):
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(sub, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(sub[border] == FRAME):
            continue
        if not np.all(sub[~border] == BG):
            continue
        candidates.append((r0, c0, r1, c1))
    assert len(candidates) == 1
    return candidates[0]


def _extract_piece_data(grid: np.ndarray) -> dict[int, dict[str, object]]:
    piece_colors = _scan_colors(grid, {BG, FRAME, MARKER})
    assert piece_colors

    marker_for: dict[int, tuple[int, int]] = {}
    for mr, mc in np.argwhere(grid == MARKER):
        touching = {
            int(grid[nr, nc])
            for nr, nc in _neighbors4(int(mr), int(mc))
            if 0 <= nr < grid.shape[0]
            and 0 <= nc < grid.shape[1]
            and int(grid[nr, nc]) not in (BG, FRAME, MARKER)
        }
        assert len(touching) == 1
        color = next(iter(touching))
        assert color not in marker_for
        marker_for[color] = (int(mr), int(mc))

    assert set(marker_for) == set(piece_colors)

    pieces: dict[int, dict[str, object]] = {}
    for color in piece_colors:
        coords = np.argwhere(grid == color)
        assert coords.size
        mr, mc = marker_for[color]
        rows = [int(v) for v in coords[:, 0]] + [mr]
        cols = [int(v) for v in coords[:, 1]] + [mc]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        occ = np.zeros((h, w), dtype=bool)
        occ[coords[:, 0] - r0, coords[:, 1] - c0] = True
        occ[mr - r0, mc - c0] = True
        align = _corner_name(mr - r0, mc - c0, h, w)

        upper = None
        missing = np.argwhere(~occ)
        if missing.size:
            nr0, nc0 = missing.min(axis=0)
            nr1, nc1 = missing.max(axis=0)
            assert missing.shape[0] == (nr1 - nr0 + 1) * (nc1 - nc0 + 1)
            sides = []
            if nr0 == 0:
                sides.append("top")
            if nr1 == h - 1:
                sides.append("bottom")
            if nc0 == 0:
                sides.append("left")
            if nc1 == w - 1:
                sides.append("right")
            assert len(sides) == 1
            block = grid[r0 + nr0 : r0 + nr1 + 1, c0 + nc0 : c0 + nc1 + 1]
            filling = _scan_colors(block, {BG, FRAME, MARKER, color})
            assert len(filling) == 1
            upper = filling[0]

        pieces[color] = {"h": h, "w": w, "align": align, "upper": upper}
    return pieces


def _stack_order(piece_data: dict[int, dict[str, object]]) -> list[int]:
    incoming = {color: 0 for color in piece_data}
    edge: dict[int, int] = {}
    for color, info in piece_data.items():
        upper = info["upper"]
        if upper is None:
            continue
        edge[color] = int(upper)
        incoming[int(upper)] += 1

    roots = [color for color, deg in incoming.items() if deg == 0]
    assert len(roots) == 1
    order: list[int] = []
    cur = roots[0]
    seen: set[int] = set()
    while True:
        assert cur not in seen
        seen.add(cur)
        order.append(cur)
        if cur not in edge:
            break
        cur = edge[cur]
    assert len(order) == len(piece_data)
    return order


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1 = _find_canvas_bbox(g)
    piece_data = _extract_piece_data(g)
    order = _stack_order(piece_data)

    out = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    out[0, :] = FRAME
    out[-1, :] = FRAME
    out[:, 0] = FRAME
    out[:, -1] = FRAME

    inner_h = out.shape[0] - 2
    inner_w = out.shape[1] - 2
    for color in order:
        info = piece_data[color]
        h = int(info["h"])
        w = int(info["w"])
        align = str(info["align"])
        if align == "tl":
            pr, pc = 1, 1
        elif align == "tr":
            pr, pc = 1, 1 + inner_w - w
        elif align == "bl":
            pr, pc = 1 + inner_h - h, 1
        else:
            pr, pc = 1 + inner_h - h, 1 + inner_w - w
        out[pr : pr + h, pc : pc + w] = color

    return out


def _build_specs(rng: random.Random) -> list[Piece] | None:
    count = rng.randint(*PIECE_COUNT_RANGE)
    colors = rng.sample(list(PIECE_COLORS), count)
    pieces: list[Piece | None] = [None] * count
    pieces[-1] = Piece(
        color=colors[-1],
        h=rng.randint(*TOP_DIM_RANGE),
        w=rng.randint(*TOP_DIM_RANGE),
    )

    for idx in range(count - 2, -1, -1):
        upper = pieces[idx + 1]
        assert upper is not None
        feasible: list[str] = []
        if upper.h + 2 <= MAX_DIM:
            feasible.extend(["left", "right"])
        if upper.w + 2 <= MAX_DIM:
            feasible.extend(["top", "bottom"])
        if not feasible:
            return None

        side = rng.choice(feasible)
        if side in ("left", "right"):
            h = rng.randint(upper.h + 2, MAX_DIM)
            w_hi = MAX_DIM if idx == 0 else FUTURE_SPAN_MAX
            w = rng.randint(3, w_hi)
        else:
            h_hi = MAX_DIM if idx == 0 else FUTURE_SPAN_MAX
            h = rng.randint(3, h_hi)
            w = rng.randint(upper.w + 2, MAX_DIM)

        lower = Piece(color=colors[idx], h=h, w=w, notch_side=side)
        lower.notch_span = upper.h if side in ("left", "right") else upper.w
        pieces[idx] = lower
        upper.attached_side = OPPOSITE_SIDE[side]

    return [piece for piece in pieces if piece is not None]


def _place_chain(pieces: list[Piece], rng: random.Random) -> bool:
    pieces[0].r = 0
    pieces[0].c = 0
    for idx in range(len(pieces) - 1):
        lower = pieces[idx]
        upper = pieces[idx + 1]
        side = lower.notch_side
        assert side is not None
        ok = False
        for _ in range(50):
            if side in ("left", "right"):
                offset = rng.randint(1, lower.h - upper.h - 1)
                depth = rng.randint(1, min(lower.w - 1, upper.w - 1))
                upper.r = lower.r + offset
                if side == "right":
                    upper.c = lower.c + lower.w - depth
                else:
                    upper.c = lower.c - upper.w + depth
            else:
                offset = rng.randint(1, lower.w - upper.w - 1)
                depth = rng.randint(1, min(lower.h - 1, upper.h - 1))
                upper.c = lower.c + offset
                if side == "bottom":
                    upper.r = lower.r + lower.h - depth
                else:
                    upper.r = lower.r - upper.h + depth

            lower.notch_offset = offset
            lower.notch_depth = depth
            want = _notch_global_bbox(lower)
            assert want is not None
            hit_ok = True
            for prev in range(idx + 1):
                inter = _bbox_intersection(_bbox(upper), _bbox(pieces[prev]))
                if prev == idx:
                    if inter != want:
                        hit_ok = False
                        break
                elif inter is not None:
                    hit_ok = False
                    break
            if hit_ok:
                ok = True
                break
        if not ok:
            return False
    return True


def _normalize_pieces(pieces: list[Piece]) -> tuple[int, int]:
    min_r = min(piece.r for piece in pieces)
    min_c = min(piece.c for piece in pieces)
    for piece in pieces:
        piece.r -= min_r
        piece.c -= min_c
    max_r = max(piece.r + piece.h for piece in pieces)
    max_c = max(piece.c + piece.w for piece in pieces)
    return max_r, max_c


def _occupancy_without_markers(pieces: list[Piece]) -> dict[tuple[int, int], int] | None:
    occ: dict[tuple[int, int], int] = {}
    for piece in pieces:
        for cell in _visible_cells(piece):
            if cell in occ:
                return None
            occ[cell] = piece.color
    return occ


def _choose_markers(pieces: list[Piece], rng: random.Random) -> bool:
    occ = _occupancy_without_markers(pieces)
    if occ is None:
        return False

    order = pieces[:]
    rng.shuffle(order)
    for piece in order:
        valid: list[str] = []
        for corner in CORNERS:
            if piece.attached_side is not None and corner in SIDE_CORNERS[piece.attached_side]:
                continue
            lr, lc = _corner_local(piece.h, piece.w, corner)
            gr, gc = piece.r + lr, piece.c + lc
            if occ.get((gr, gc)) != piece.color:
                continue
            ok = True
            for nr, nc in _neighbors4(gr, gc):
                other = occ.get((nr, nc))
                if other is not None and other != piece.color:
                    ok = False
                    break
            if ok:
                valid.append(corner)
        if not valid:
            return False
        piece.marker_corner = rng.choice(valid)
    return True


def _expected_output(pieces: list[Piece], inner_h: int, inner_w: int) -> np.ndarray:
    out = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    out[0, :] = FRAME
    out[-1, :] = FRAME
    out[:, 0] = FRAME
    out[:, -1] = FRAME
    for piece in pieces:
        if piece.marker_corner == "tl":
            pr, pc = 1, 1
        elif piece.marker_corner == "tr":
            pr, pc = 1, 1 + inner_w - piece.w
        elif piece.marker_corner == "bl":
            pr, pc = 1 + inner_h - piece.h, 1
        else:
            pr, pc = 1 + inner_h - piece.h, 1 + inner_w - piece.w
        out[pr : pr + piece.h, pc : pc + piece.w] = piece.color
    return out


def _layout_scene(
    pieces: list[Piece], inner_h: int, inner_w: int, rng: random.Random
) -> tuple[int, int, tuple[int, int, int, int]]:
    cluster_h, cluster_w = _normalize_pieces(pieces)
    frame_h = inner_h + 2
    frame_w = inner_w + 2
    gap = rng.randint(2, 4)
    slack = rng.randint(0, 4)

    if rng.random() < 0.5:
        grid_h = max(cluster_h, frame_h) + 2 * LAYOUT_MARGIN + rng.randint(0, 4)
        grid_w = cluster_w + frame_w + gap + 2 * LAYOUT_MARGIN + slack
        cluster_r = LAYOUT_MARGIN + rng.randint(0, grid_h - 2 * LAYOUT_MARGIN - cluster_h)
        frame_r = LAYOUT_MARGIN + rng.randint(0, grid_h - 2 * LAYOUT_MARGIN - frame_h)
        extra = grid_w - 2 * LAYOUT_MARGIN - gap - cluster_w - frame_w
        left_pad = rng.randint(0, extra)
        if rng.random() < 0.5:
            cluster_c = LAYOUT_MARGIN + left_pad
            frame_c = cluster_c + cluster_w + gap
        else:
            frame_c = LAYOUT_MARGIN + left_pad
            cluster_c = frame_c + frame_w + gap
    else:
        grid_w = max(cluster_w, frame_w) + 2 * LAYOUT_MARGIN + rng.randint(0, 4)
        grid_h = cluster_h + frame_h + gap + 2 * LAYOUT_MARGIN + slack
        cluster_c = LAYOUT_MARGIN + rng.randint(0, grid_w - 2 * LAYOUT_MARGIN - cluster_w)
        frame_c = LAYOUT_MARGIN + rng.randint(0, grid_w - 2 * LAYOUT_MARGIN - frame_w)
        extra = grid_h - 2 * LAYOUT_MARGIN - gap - cluster_h - frame_h
        top_pad = rng.randint(0, extra)
        if rng.random() < 0.5:
            cluster_r = LAYOUT_MARGIN + top_pad
            frame_r = cluster_r + cluster_h + gap
        else:
            frame_r = LAYOUT_MARGIN + top_pad
            cluster_r = frame_r + frame_h + gap

    for piece in pieces:
        piece.r += cluster_r
        piece.c += cluster_c
    return grid_h, grid_w, (frame_r, frame_c, frame_r + frame_h - 1, frame_c + frame_w - 1)


def _render_input(
    pieces: list[Piece], grid_h: int, grid_w: int, frame_bbox: tuple[int, int, int, int]
) -> np.ndarray | None:
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    fr0, fc0, fr1, fc1 = frame_bbox
    grid[fr0, fc0 : fc1 + 1] = FRAME
    grid[fr1, fc0 : fc1 + 1] = FRAME
    grid[fr0 : fr1 + 1, fc0] = FRAME
    grid[fr0 : fr1 + 1, fc1] = FRAME

    for piece in pieces:
        mask = _visible_mask(piece)
        for rr, cc in np.argwhere(mask):
            gr = piece.r + int(rr)
            gc = piece.c + int(cc)
            if not (0 <= gr < grid_h and 0 <= gc < grid_w):
                return None
            if grid[gr, gc] != BG:
                return None
            grid[gr, gc] = piece.color

    for piece in pieces:
        mr, mc = _corner_local(piece.h, piece.w, piece.marker_corner)
        gr = piece.r + mr
        gc = piece.c + mc
        if grid[gr, gc] != piece.color:
            return None
        grid[gr, gc] = MARKER

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pieces = _build_specs(rng)
    if pieces is None:
        return None
    if not _place_chain(pieces, rng):
        return None
    if not _choose_markers(pieces, rng):
        return None

    inner_h = max(piece.h for piece in pieces) + rng.randint(*CANVAS_EXTRA_RANGE)
    inner_w = max(piece.w for piece in pieces) + rng.randint(*CANVAS_EXTRA_RANGE)
    grid_h, grid_w, frame_bbox = _layout_scene(pieces, inner_h, inner_w, rng)
    grid = _render_input(pieces, grid_h, grid_w, frame_bbox)
    if grid is None:
        return None

    expected = _expected_output(pieces, inner_h, inner_w)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
