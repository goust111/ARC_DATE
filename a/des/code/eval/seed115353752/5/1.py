from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
FRAME = 5
SIDES = ("top", "bottom", "left", "right")
PALETTE = tuple(c for c in range(10) if c not in (BG, FRAME))
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _coords_to_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _transform(mask: np.ndarray, code: int) -> np.ndarray:
    out = np.rot90(mask, code % 4)
    if code >= 4:
        out = np.fliplr(out)
    return out.copy()


def _variants(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[bytes] = set()
    for code in range(8):
        variant = _transform(mask, code)
        key = bytes([variant.shape[0], variant.shape[1]]) + variant.tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append(variant)
    return out


def _canonical_key(mask: np.ndarray) -> bytes:
    return min(bytes([v.shape[0], v.shape[1]]) + v.tobytes() for v in _variants(mask))


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[cnts.argmax()])


def _scan_hole_from_anchor(
    grid: np.ndarray,
    hole_by_cell: dict[tuple[int, int], int],
    frame_bbox: tuple[int, int, int, int],
    anchor_pos: tuple[int, int],
    side: str,
    bg: int,
) -> int:
    fr0, fc0, fr1, fc1 = frame_bbox
    r, c = anchor_pos
    if side == "top":
        seq = ((rr, c) for rr in range(fr0, fr1 + 1))
    elif side == "bottom":
        seq = ((rr, c) for rr in range(fr1, fr0 - 1, -1))
    elif side == "left":
        seq = ((r, cc) for cc in range(fc0, fc1 + 1))
    else:
        seq = ((r, cc) for cc in range(fc1, fc0 - 1, -1))
    for rr, cc in seq:
        if int(grid[rr, cc]) == bg:
            return hole_by_cell[(int(rr), int(cc))]
    raise AssertionError("anchor ray must hit a hole")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each outside grey fragment to the uniquely shaped hole inside the grey frame,
    then fill that hole using the color of the line joining the fragment anchor and the hole anchor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    non_bg = g[g != bg]
    assert non_bg.size > 0, "expected foreground content"
    frame_color = int(Counter(map(int, non_bg)).most_common(1)[0][0])

    grey_comps = _components(g == frame_color)
    assert grey_comps, "expected grey components"
    frame_index = max(range(len(grey_comps)), key=lambda i: int(len(grey_comps[i])))
    frame_coords = grey_comps[frame_index]
    frame_bbox = _bbox(frame_coords)
    fr0, fc0, fr1, fc1 = frame_bbox

    local_bg = g[fr0 : fr1 + 1, fc0 : fc1 + 1] == bg
    holes: list[dict[str, object]] = []
    hole_by_cell: dict[tuple[int, int], int] = {}
    for local_coords in _components(local_bg):
        if np.any(local_coords[:, 0] == 0) or np.any(local_coords[:, 1] == 0):
            continue
        if np.any(local_coords[:, 0] == local_bg.shape[0] - 1) or np.any(local_coords[:, 1] == local_bg.shape[1] - 1):
            continue
        global_coords = local_coords + np.array([fr0, fc0], dtype=int)
        mask, bbox = _coords_to_mask(global_coords)
        hole_id = len(holes)
        for rr, cc in global_coords:
            hole_by_cell[(int(rr), int(cc))] = hole_id
        holes.append({"coords": global_coords, "mask": mask, "bbox": bbox})
    assert holes, "expected enclosed holes inside the frame"

    vals, cnts = np.unique(g, return_counts=True)
    singleton_colors = {
        int(v)
        for v, cnt in zip(vals, cnts, strict=False)
        if int(v) not in (bg, frame_color) and int(cnt) == 1
    }
    anchors = {
        color: tuple(map(int, np.argwhere(g == color)[0]))
        for color in singleton_colors
    }
    assert anchors, "expected anchor pixels"

    hole_anchor_by_hole: dict[int, dict[str, object]] = {}
    hole_anchor_colors: set[int] = set()
    for color, pos in anchors.items():
        r, c = pos
        side = None
        if r == fr0 - 1 and fc0 <= c <= fc1:
            side = "top"
        elif r == fr1 + 1 and fc0 <= c <= fc1:
            side = "bottom"
        elif c == fc0 - 1 and fr0 <= r <= fr1:
            side = "left"
        elif c == fc1 + 1 and fr0 <= r <= fr1:
            side = "right"
        if side is None:
            continue
        hole_id = _scan_hole_from_anchor(g, hole_by_cell, frame_bbox, pos, side, bg)
        assert hole_id not in hole_anchor_by_hole, "each hole needs one anchor"
        hole_anchor_by_hole[hole_id] = {"color": color, "pos": pos}
        hole_anchor_colors.add(color)
    assert len(hole_anchor_by_hole) == len(holes), "every hole needs a frame-side anchor"

    piece_comps: dict[int, dict[str, object]] = {}
    piece_cell_to_id: dict[tuple[int, int], int] = {}
    next_piece_id = 0
    for i, coords in enumerate(grey_comps):
        if i == frame_index:
            continue
        mask, bbox = _coords_to_mask(coords)
        piece_comps[next_piece_id] = {"coords": coords, "mask": mask, "bbox": bbox}
        for rr, cc in coords:
            piece_cell_to_id[(int(rr), int(cc))] = next_piece_id
        next_piece_id += 1
    assert piece_comps, "expected outside grey pieces"

    piece_anchor_by_piece: dict[int, dict[str, object]] = {}
    for color, pos in anchors.items():
        if color in hole_anchor_colors:
            continue
        r, c = pos
        touching = {
            piece_cell_to_id[(nr, nc)]
            for nr, nc in _neighbors4(r, c)
            if (nr, nc) in piece_cell_to_id
        }
        assert len(touching) == 1, "piece anchor must touch exactly one grey piece"
        piece_id = next(iter(touching))
        assert piece_id not in piece_anchor_by_piece, "each piece needs one anchor"
        piece_anchor_by_piece[piece_id] = {"color": color, "pos": pos}
    assert len(piece_anchor_by_piece) == len(piece_comps), "every piece needs one anchor"

    line_pair_to_color: dict[frozenset[int], int] = {}
    anchor_colors = set(anchors)
    for color in set(map(int, g.ravel())) - {bg, frame_color} - singleton_colors:
        for coords in _components(g == color):
            adjacent = set()
            for r, c in coords:
                for nr, nc in _neighbors4(int(r), int(c)):
                    if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                        neighbor_color = int(g[nr, nc])
                        if neighbor_color in anchor_colors:
                            adjacent.add(neighbor_color)
            if len(adjacent) == 2:
                key = frozenset(adjacent)
                assert key not in line_pair_to_color, "anchor pair must map to one line color"
                line_pair_to_color[key] = int(color)
    assert line_pair_to_color, "expected color-encoding lines"

    hole_by_key: dict[bytes, int] = {}
    for hole_id, hole in enumerate(holes):
        key = _canonical_key(hole["mask"])
        assert key not in hole_by_key, "hole shapes must be unique up to rotation/flip"
        hole_by_key[key] = hole_id

    fills: dict[int, int] = {}
    for piece_id, piece in piece_comps.items():
        key = _canonical_key(piece["mask"])
        assert key in hole_by_key, "piece shape must match a hole shape"
        hole_id = hole_by_key[key]
        assert hole_id not in fills, "holes must match one piece each"
        piece_anchor = int(piece_anchor_by_piece[piece_id]["color"])
        hole_anchor = int(hole_anchor_by_hole[hole_id]["color"])
        pair = frozenset((piece_anchor, hole_anchor))
        assert pair in line_pair_to_color, "matched anchors need a line color"
        fills[hole_id] = int(line_pair_to_color[pair])
    assert len(fills) == len(holes), "every hole must be filled"

    out = np.full_like(g, bg)
    out[frame_coords[:, 0], frame_coords[:, 1]] = frame_color
    for info in hole_anchor_by_hole.values():
        r, c = info["pos"]
        out[int(r), int(c)] = int(info["color"])
    for hole_id, color in fills.items():
        coords = holes[hole_id]["coords"]
        out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def _clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def _grow_shape() -> np.ndarray:
    area = random.randint(4, 7)
    chosen = {(0, 0)}
    frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
    while len(chosen) < area:
        cands = [p for p in frontier if p not in chosen]
        nr, nc = random.choice(cands)
        chosen.add((nr, nc))
        frontier.update({(nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)})

    rows = [r for r, _ in chosen]
    cols = [c for _, c in chosen]
    r0, c0 = min(rows), min(cols)
    norm = {(r - r0, c - c0) for r, c in chosen}
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    if h < 2 or w < 2 or h > 4 or w > 4:
        raise ValueError("shape outside allowed size")
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    if len(_variants(mask)) < 2:
        raise ValueError("shape is too symmetric")
    return mask


def _sample_unique_shapes(n: int) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[bytes] = set()
    attempts = 0
    while len(out) < n and attempts < 200:
        attempts += 1
        try:
            mask = _grow_shape()
        except ValueError:
            continue
        key = _canonical_key(mask)
        if key in seen:
            continue
        seen.add(key)
        out.append(mask)
    if len(out) != n:
        raise ValueError("failed to sample distinct shapes")
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[:-1, 1:] |= mask[1:, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    return out


def _choose_hole_position(
    side: str,
    mask: np.ndarray,
    frame_shape: tuple[int, int],
    forbidden: np.ndarray,
) -> tuple[int, int]:
    fh, fw = frame_shape
    h, w = mask.shape
    options: list[tuple[int, int]] = []
    if side == "top":
        r_candidates = range(2, min(5, fh - h - 1))
        c_center = fw // 2 - w // 2
        c_candidates = range(_clamp(c_center - 3, 2, fw - w - 2), _clamp(c_center + 4, 2, fw - w - 2) + 1)
    elif side == "bottom":
        lo = max(2, fh - h - 4)
        hi = fh - h - 2
        r_candidates = range(lo, hi + 1)
        c_center = fw // 2 - w // 2
        c_candidates = range(_clamp(c_center - 3, 2, fw - w - 2), _clamp(c_center + 4, 2, fw - w - 2) + 1)
    elif side == "left":
        c_candidates = range(2, min(5, fw - w - 1))
        r_center = fh // 2 - h // 2
        r_candidates = range(_clamp(r_center - 3, 2, fh - h - 2), _clamp(r_center + 4, 2, fh - h - 2) + 1)
    else:
        lo = max(2, fw - w - 4)
        hi = fw - w - 2
        c_candidates = range(lo, hi + 1)
        r_center = fh // 2 - h // 2
        r_candidates = range(_clamp(r_center - 3, 2, fh - h - 2), _clamp(r_center + 4, 2, fh - h - 2) + 1)

    for r0 in r_candidates:
        for c0 in c_candidates:
            if np.any(forbidden[r0 : r0 + h, c0 : c0 + w] & mask):
                continue
            options.append((int(r0), int(c0)))
    if not options:
        raise ValueError("no hole placement fits")
    return random.choice(options)


def _boundary_offsets(mask: np.ndarray, side: str) -> list[tuple[int, int]]:
    coords = np.argwhere(mask)
    if side == "top":
        edge = int(coords[:, 0].min())
        subset = coords[coords[:, 0] == edge]
    elif side == "bottom":
        edge = int(coords[:, 0].max())
        subset = coords[coords[:, 0] == edge]
    elif side == "left":
        edge = int(coords[:, 1].min())
        subset = coords[coords[:, 1] == edge]
    else:
        edge = int(coords[:, 1].max())
        subset = coords[coords[:, 1] == edge]
    return [(int(r), int(c)) for r, c in subset]


def _place_piece(
    grid: np.ndarray,
    side: str,
    mask: np.ndarray,
    piece_anchor_color: int,
    hole_anchor_pos: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
) -> tuple[tuple[int, int], list[tuple[int, int]], tuple[int, int]]:
    h, w = mask.shape
    H, W = grid.shape
    fr0, fc0, fr1, fc1 = frame_bbox
    edge_offsets = _boundary_offsets(mask, {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}[side])

    for _ in range(80):
        ar_off_r, ar_off_c = random.choice(edge_offsets)
        if side == "top":
            r0_max = fr0 - h - 2
            if r0_max < 0:
                continue
            r0 = random.randint(0, r0_max)
            base_c = hole_anchor_pos[1] - ar_off_c
            c0 = _clamp(base_c + random.randint(-4, 4), 1, W - w - 2)
            anchor = (r0 + h, c0 + ar_off_c)
        elif side == "bottom":
            r0_min = fr1 + 3
            if r0_min > H - h - 1:
                continue
            r0 = random.randint(r0_min, H - h - 1)
            base_c = hole_anchor_pos[1] - ar_off_c
            c0 = _clamp(base_c + random.randint(-4, 4), 1, W - w - 2)
            anchor = (r0 - 1, c0 + ar_off_c)
        elif side == "left":
            c0_max = fc0 - w - 2
            if c0_max < 0:
                continue
            c0 = random.randint(0, c0_max)
            base_r = hole_anchor_pos[0] - ar_off_r
            r0 = _clamp(base_r + random.randint(-4, 4), 1, H - h - 2)
            anchor = (r0 + ar_off_r, c0 + w)
        else:
            c0_min = fc1 + 3
            if c0_min > W - w - 1:
                continue
            c0 = random.randint(c0_min, W - w - 1)
            base_r = hole_anchor_pos[0] - ar_off_r
            r0 = _clamp(base_r + random.randint(-4, 4), 1, H - h - 2)
            anchor = (r0 + ar_off_r, c0 - 1)

        body_cells = [(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(mask)]
        if any(grid[r, c] != BG for r, c in body_cells):
            continue
        if grid[anchor] != BG:
            continue

        for r, c in body_cells:
            grid[r, c] = FRAME
        grid[anchor] = piece_anchor_color
        return (r0, c0), body_cells, anchor
    raise ValueError("failed to place piece")


def _path_points(start: tuple[int, int], end: tuple[int, int], side: str) -> list[tuple[int, int]]:
    sr, sc = start
    er, ec = end
    if side in ("top", "bottom"):
        mid = (sr, ec)
    else:
        mid = (er, sc)
    return [start, mid, end]


def _expand_path(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    path = [points[0]]
    for (r0, c0), (r1, c1) in zip(points, points[1:], strict=False):
        if r0 == r1:
            step = 1 if c1 > c0 else -1
            for c in range(c0 + step, c1 + step, step):
                path.append((int(r0), int(c)))
        else:
            step = 1 if r1 > r0 else -1
            for r in range(r0 + step, r1 + step, step):
                path.append((int(r), int(c0)))
    return path


def _draw_line_component(
    grid: np.ndarray,
    side: str,
    start: tuple[int, int],
    end: tuple[int, int],
    color: int,
) -> list[tuple[int, int]]:
    path = _expand_path(_path_points(start, end, side))
    cells = path[1:-1]
    if len(cells) < 2:
        raise ValueError("line must occupy at least two cells")
    for r, c in cells:
        if grid[r, c] != BG:
            raise ValueError("line path collides with existing content")
    for r, c in cells:
        grid[r, c] = color
    return cells


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    n = random.choice((2, 3))
    sides = random.sample(SIDES, n)
    anchor_colors = random.sample(PALETTE, 2 * n)
    remaining = [c for c in PALETTE if c not in anchor_colors]
    if not remaining:
        raise ValueError("need colors for line encoding")
    fill_palette = random.sample(remaining, random.randint(1, min(2, len(remaining))))

    fh = random.randint(14, 18)
    fw = random.randint(14, 18)
    mt = random.randint(6, 9)
    mb = random.randint(6, 9)
    ml = random.randint(6, 9)
    mr = random.randint(6, 9)
    H = fh + mt + mb
    W = fw + ml + mr
    frame_bbox = (mt, ml, mt + fh - 1, ml + fw - 1)

    shapes = _sample_unique_shapes(n)
    frame_local = np.full((fh, fw), FRAME, dtype=int)
    forbidden = np.zeros((fh, fw), dtype=bool)
    hole_specs: list[dict[str, object]] = []

    for idx, side in enumerate(sides):
        hole_mask = _transform(shapes[idx], random.randrange(8))
        hr, hc = _choose_hole_position(side, hole_mask, (fh, fw), forbidden)
        coords_local = [(hr + int(r), hc + int(c)) for r, c in np.argwhere(hole_mask)]
        for r, c in coords_local:
            frame_local[r, c] = BG
        stamp = np.zeros_like(forbidden)
        for r, c in coords_local:
            stamp[r, c] = True
        forbidden |= _dilate8(stamp)

        boundary = _boundary_offsets(hole_mask, side)
        br, bc = random.choice(boundary)
        if side == "top":
            anchor_local = (-1, hc + bc)
        elif side == "bottom":
            anchor_local = (fh, hc + bc)
        elif side == "left":
            anchor_local = (hr + br, -1)
        else:
            anchor_local = (hr + br, fw)

        hole_specs.append(
            {
                "side": side,
                "mask": hole_mask,
                "coords_local": coords_local,
                "hole_anchor_color": anchor_colors[2 * idx],
                "piece_anchor_color": anchor_colors[2 * idx + 1],
                "fill_color": random.choice(fill_palette),
                "anchor_local": anchor_local,
            }
        )

    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    inp[mt : mt + fh, ml : ml + fw] = frame_local
    out[mt : mt + fh, ml : ml + fw] = frame_local

    for spec in hole_specs:
        ar, ac = spec["anchor_local"]
        gr = mt + int(ar)
        gc = ml + int(ac)
        inp[gr, gc] = int(spec["hole_anchor_color"])
        out[gr, gc] = int(spec["hole_anchor_color"])
        for lr, lc in spec["coords_local"]:
            out[mt + int(lr), ml + int(lc)] = int(spec["fill_color"])

    for spec in hole_specs:
        piece_mask = _transform(spec["mask"], random.randrange(8))
        hole_anchor_pos = (mt + int(spec["anchor_local"][0]), ml + int(spec["anchor_local"][1]))
        _place_piece(inp, str(spec["side"]), piece_mask, int(spec["piece_anchor_color"]), hole_anchor_pos, frame_bbox)

    for spec in hole_specs:
        piece_anchor_pos = tuple(map(int, np.argwhere(inp == int(spec["piece_anchor_color"]))[0]))
        hole_anchor_pos = (mt + int(spec["anchor_local"][0]), ml + int(spec["anchor_local"][1]))
        _draw_line_component(inp, str(spec["side"]), piece_anchor_pos, hole_anchor_pos, int(spec["fill_color"]))

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            inp, expected = _build_case()
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid ARC instance")
