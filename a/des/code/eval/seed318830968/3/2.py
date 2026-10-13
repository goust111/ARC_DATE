from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


BG = 8
PALETTE = [1, 2, 3, 4, 5, 6, 7, 9]
DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 200


def _bbox(coords: np.ndarray) -> BBox:
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
            q: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            pts: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _normalize(coords: np.ndarray) -> np.ndarray:
    pts = coords.copy()
    pts[:, 0] -= int(pts[:, 0].min())
    pts[:, 1] -= int(pts[:, 1].min())
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    return pts[order]


def _transform_shape(coords: np.ndarray, rot: int, flip: bool) -> np.ndarray:
    pts = coords.copy()
    for _ in range(rot):
        pts = np.stack([pts[:, 1], -pts[:, 0]], axis=1)
    if flip:
        pts[:, 1] *= -1
    return _normalize(pts)


def _random_polyomino(size: int) -> np.ndarray:
    assert 2 <= size <= 5
    for _ in range(100):
        cells = {(0, 0)}
        frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
        while len(cells) < size:
            cell = random.choice(tuple(frontier))
            frontier.discard(cell)
            if cell in cells:
                continue
            cells.add(cell)
            r, c = cell
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (r + dr, c + dc)
                if nxt not in cells:
                    frontier.add(nxt)

        pts = _normalize(np.array(sorted(cells), dtype=int))
        pts = _transform_shape(pts, random.randrange(4), random.choice([False, True]))
        r0, c0, r1, c1 = _bbox(pts)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if size == 4 and h == 2 and w == 2:
            continue
        return pts
    raise RuntimeError("failed to sample polyomino")


def _frame_from_component(grid: np.ndarray, color: int, coords: np.ndarray, bg: int) -> dict | None:
    if coords.shape[0] < 7:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    pts = {tuple(map(int, p)) for p in coords}
    perim: list[Coord] = []
    for c in range(c0, c1 + 1):
        perim.append((r0, c))
        perim.append((r1, c))
    for r in range(r0 + 1, r1):
        perim.append((r, c0))
        perim.append((r, c1))
    perim_set = set(perim)
    if pts - perim_set:
        return None
    missing = sorted(perim_set - pts)
    if len(missing) != 1:
        return None
    gap = missing[0]
    gr, gc = gap
    if gr in (r0, r1) and gc in (c0, c1):
        return None
    if not np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == bg):
        return None
    if gr == r0:
        move = (1, 0)
    elif gr == r1:
        move = (-1, 0)
    elif gc == c0:
        move = (0, 1)
    else:
        move = (0, -1)
    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "move": move,
    }


def _fragment_distance(frame_bbox: BBox, move: Coord, frag_bbox: BBox) -> int:
    r0, c0, r1, c1 = frame_bbox
    fr0, fc0, fr1, fc1 = frag_bbox
    dr, dc = move
    if dr == 1:
        return r0 - fr1
    if dr == -1:
        return fr0 - r1
    if dc == 1:
        return c0 - fc1
    return fc0 - c1


def _inside_bbox(coords: np.ndarray, bbox: BBox) -> bool:
    r0, c0, r1, c1 = bbox
    return (
        int(coords[:, 0].min()) >= r0
        and int(coords[:, 0].max()) <= r1
        and int(coords[:, 1].min()) >= c0
        and int(coords[:, 1].max()) <= c1
    )


def _entry_steps(coords: np.ndarray, move: Coord, interior_bbox: BBox) -> int:
    r0, c0, r1, c1 = interior_bbox
    dr, dc = move
    if dr == 1:
        return r0 - int(coords[:, 0].min())
    if dr == -1:
        return int(coords[:, 0].max()) - r1
    if dc == 1:
        return c0 - int(coords[:, 1].min())
    return int(coords[:, 1].max()) - c1


def _place_fragment(
    coords: np.ndarray,
    move: Coord,
    interior_bbox: BBox,
    occupied: np.ndarray,
    frame_color: int,
    out: np.ndarray,
) -> None:
    dr, dc = move
    t = _entry_steps(coords, move, interior_bbox)
    shifted = coords + np.array([dr * t, dc * t], dtype=int)
    assert _inside_bbox(shifted, interior_bbox)
    assert not np.any(occupied[shifted[:, 0], shifted[:, 1]])
    while True:
        nxt = shifted + np.array([dr, dc], dtype=int)
        if not _inside_bbox(nxt, interior_bbox):
            break
        if np.any(occupied[nxt[:, 0], nxt[:, 1]]):
            break
        shifted = nxt
    occupied[shifted[:, 0], shifted[:, 1]] = True
    out[shifted[:, 0], shifted[:, 1]] = frame_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide same-colored fragments along their frame's dotted guide direction, then
    stack them inside the hollow frame from the far wall back toward the gap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    counts = Counter(map(int, g.ravel()))
    bg = max(counts, key=counts.get)
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert colors

    out = np.full_like(g, bg)

    for color in colors:
        comps = _components(g == color)
        frame_infos = []
        fragments: list[np.ndarray] = []
        for comp in comps:
            info = _frame_from_component(g, color, comp, bg)
            if info is not None:
                frame_infos.append(info)
            elif comp.shape[0] > 1:
                fragments.append(comp)
        assert len(frame_infos) == 1, "each color must define exactly one gapped frame"
        assert fragments, "each frame color must have at least one fragment"
        frame = frame_infos[0]
        r0, c0, r1, c1 = frame["bbox"]
        gap = frame["gap"]
        move = frame["move"]
        frags_with_keys = []
        for frag in fragments:
            fb = _bbox(frag)
            dist = _fragment_distance(frame["bbox"], move, fb)
            assert dist > 0
            frags_with_keys.append((dist, fb[0], fb[1], _normalize(frag.copy()), frag))
        frags_with_keys.sort(key=lambda item: (item[0], item[1], item[2]))

        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
        out[gap] = color

        interior_bbox = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
        occupied = np.zeros_like(g, dtype=bool)
        for _dist, _row, _col, _norm, frag in frags_with_keys:
            _place_fragment(frag, move, interior_bbox, occupied, color, out)

    return out


def _shape_spans(coords: np.ndarray, move: Coord) -> tuple[int, int]:
    dr, dc = move
    if dr != 0:
        axial = int(coords[:, 0].max()) + 1
        cross = int(coords[:, 1].max()) + 1
    else:
        axial = int(coords[:, 1].max()) + 1
        cross = int(coords[:, 0].max()) + 1
    return axial, cross


def _translate(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    return coords + np.array([dr, dc], dtype=int)


def _simulate_interior(
    pieces: list[np.ndarray],
    move: Coord,
    interior_shape: tuple[int, int],
    cross_origin: int,
) -> list[np.ndarray] | None:
    ih, iw = interior_shape
    occupied = np.zeros((ih, iw), dtype=bool)
    placed: list[np.ndarray] = []
    dr, dc = move
    for piece in pieces:
        if dr == 1:
            shifted = piece + np.array([0, cross_origin], dtype=int)
        elif dr == -1:
            shifted = piece + np.array([ih - 1 - int(piece[:, 0].max()), cross_origin], dtype=int)
        elif dc == 1:
            shifted = piece + np.array([cross_origin, 0], dtype=int)
        else:
            shifted = piece + np.array([cross_origin, iw - 1 - int(piece[:, 1].max())], dtype=int)

        if not _inside_bbox(shifted, (0, 0, ih - 1, iw - 1)):
            return None
        if np.any(occupied[shifted[:, 0], shifted[:, 1]]):
            return None
        while True:
            nxt = _translate(shifted, dr, dc)
            if not _inside_bbox(nxt, (0, 0, ih - 1, iw - 1)):
                break
            if np.any(occupied[nxt[:, 0], nxt[:, 1]]):
                break
            shifted = nxt
        occupied[shifted[:, 0], shifted[:, 1]] = True
        placed.append(shifted)
    return placed


def _build_spec_patch(color: int, move: Coord, pieces: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    spans = [_shape_spans(piece, move) for piece in pieces]
    max_cross = max(cross for _axial, cross in spans)
    total_axial = sum(axial for axial, _cross in spans)
    cross_dim = max_cross + random.randint(0, 2)
    along_dim = total_axial + random.randint(1, 3)

    dr, dc = move
    if dr != 0:
        interior_shape = (along_dim, cross_dim)
        cross_limit = cross_dim - max_cross
    else:
        interior_shape = (cross_dim, along_dim)
        cross_limit = cross_dim - max_cross
    if cross_limit < 0:
        return None
    cross_origin = random.randint(0, cross_limit)
    placed_inside = _simulate_interior(pieces, move, interior_shape, cross_origin)
    if placed_inside is None:
        return None

    ih, iw = interior_shape
    fh, fw = ih + 2, iw + 2
    line_len = random.randint(1, 3)
    extra_gap_choices = [2, 3]

    outside_positions: list[np.ndarray] = []
    if dr == 1:
        cursor_bottom = -(2 * line_len + 1)
        for piece in pieces:
            shift = np.array([cursor_bottom - int(piece[:, 0].max()), 1 + cross_origin], dtype=int)
            placed = piece + shift
            outside_positions.append(placed)
            cursor_bottom = int(placed[:, 0].min()) - random.choice(extra_gap_choices)
        gap_cell = (0, 1 + cross_origin)
        dots = [(-1 - 2 * k, 1 + cross_origin) for k in range(line_len)]
    elif dr == -1:
        cursor_top = fh + 2 * line_len
        for piece in pieces:
            shift = np.array([cursor_top, 1 + cross_origin], dtype=int)
            placed = piece + shift
            outside_positions.append(placed)
            cursor_top = int(placed[:, 0].max()) + random.choice(extra_gap_choices)
        gap_cell = (fh - 1, 1 + cross_origin)
        dots = [(fh + 2 * k, 1 + cross_origin) for k in range(line_len)]
    elif dc == 1:
        cursor_right = -(2 * line_len + 1)
        for piece in pieces:
            shift = np.array([1 + cross_origin, cursor_right - int(piece[:, 1].max())], dtype=int)
            placed = piece + shift
            outside_positions.append(placed)
            cursor_right = int(placed[:, 1].min()) - random.choice(extra_gap_choices)
        gap_cell = (1 + cross_origin, 0)
        dots = [(1 + cross_origin, -1 - 2 * k) for k in range(line_len)]
    else:
        cursor_left = fw + 2 * line_len
        for piece in pieces:
            shift = np.array([1 + cross_origin, cursor_left], dtype=int)
            placed = piece + shift
            outside_positions.append(placed)
            cursor_left = int(placed[:, 1].max()) + random.choice(extra_gap_choices)
        gap_cell = (1 + cross_origin, fw - 1)
        dots = [(1 + cross_origin, fw + 2 * k) for k in range(line_len)]

    all_pts: list[Coord] = []
    frame_pts: list[Coord] = []
    for c in range(fw):
        frame_pts.append((0, c))
        frame_pts.append((fh - 1, c))
    for r in range(1, fh - 1):
        frame_pts.append((r, 0))
        frame_pts.append((r, fw - 1))
    frame_pts = [pt for pt in frame_pts if pt != gap_cell]
    all_pts.extend(frame_pts)
    all_pts.extend(dots)
    for placed in outside_positions:
        all_pts.extend((int(r), int(c)) for r, c in placed)

    min_r = min(r for r, _c in all_pts)
    min_c = min(c for _r, c in all_pts)
    max_r = max(r for r, _c in all_pts)
    max_c = max(c for _r, c in all_pts)
    h = max_r - min_r + 1
    w = max_c - min_c + 1

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    sr = -min_r
    sc = -min_c

    for r, c in frame_pts:
        inp[r + sr, c + sc] = color
    for r, c in dots:
        inp[r + sr, c + sc] = color
    for placed in outside_positions:
        inp[placed[:, 0] + sr, placed[:, 1] + sc] = color

    out[sr : sr + fh, sc : sc + fw][0, :] = color
    out[sr : sr + fh, sc : sc + fw][fh - 1, :] = color
    out[sr : sr + fh, sc : sc + fw][:, 0] = color
    out[sr : sr + fh, sc : sc + fw][:, fw - 1] = color
    for placed in placed_inside:
        out[placed[:, 0] + sr + 1, placed[:, 1] + sc + 1] = color

    return inp, out


def _overlay(dst: np.ndarray, src: np.ndarray, top: int, left: int) -> None:
    h, w = src.shape
    mask = src != BG
    view = dst[top : top + h, left : left + w]
    view[mask] = src[mask]


def _boxes_overlap(a: BBox, b: BBox, pad: int = 1) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _assemble_canvas(patches: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray] | None:
    dims = [(inp.shape[0], inp.shape[1]) for inp, _out in patches]
    max_h = max(h for h, _w in dims)
    max_w = max(w for _h, w in dims)
    total_h = sum(h for h, _w in dims)
    total_w = sum(w for _h, w in dims)
    min_h = max(18, max_h + 2, total_h // 2)
    min_w = max(18, max_w + 2, total_w // 2)
    if min_h > 30 or min_w > 30:
        return None

    for _ in range(80):
        H = random.randint(min_h, 30)
        W = random.randint(min_w, 30)
        order = sorted(range(len(patches)), key=lambda i: patches[i][0].shape[0] * patches[i][0].shape[1], reverse=True)
        boxes: list[BBox] = []
        positions: dict[int, tuple[int, int]] = {}
        ok = True
        for idx in order:
            ph, pw = patches[idx][0].shape
            r_lo = 1 if H - ph >= 2 else 0
            c_lo = 1 if W - pw >= 2 else 0
            r_hi = H - ph - (1 if H - ph >= 2 else 0)
            c_hi = W - pw - (1 if W - pw >= 2 else 0)
            if r_hi < r_lo or c_hi < c_lo:
                ok = False
                break
            placed = False
            for _ in range(PLACEMENT_ATTEMPTS):
                top = random.randint(r_lo, r_hi)
                left = random.randint(c_lo, c_hi)
                box = (top, left, top + ph - 1, left + pw - 1)
                if any(_boxes_overlap(box, prev) for prev in boxes):
                    continue
                boxes.append(box)
                positions[idx] = (top, left)
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        inp = np.full((H, W), BG, dtype=int)
        out = np.full((H, W), BG, dtype=int)
        for idx, (patch_inp, patch_out) in enumerate(patches):
            top, left = positions[idx]
            _overlay(inp, patch_inp, top, left)
            _overlay(out, patch_out, top, left)
        return inp, out
    return None


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(1, 3)
    colors = random.sample(PALETTE, frame_count)
    directions = random.sample(DIRS, frame_count) if frame_count <= len(DIRS) else [random.choice(DIRS) for _ in range(frame_count)]

    patches: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(frame_count):
        piece_count = random.randint(1, 3)
        pieces = [_random_polyomino(random.randint(2, 5)) for _ in range(piece_count)]
        spec = _build_spec_patch(colors[i], directions[i], pieces)
        if spec is None:
            return None
        patches.append(spec)

    assembled = _assemble_canvas(patches)
    if assembled is None:
        return None
    inp, out = assembled
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
