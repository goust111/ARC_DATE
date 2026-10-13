from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 5

DIR_STEP = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIR_STEP.values():
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_coords(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _grid_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps = []
    for coords in _components(grid != bg):
        colors = {int(grid[r, c]) for r, c in coords}
        mask, bbox = _mask_from_coords(coords)
        comps.append(
            {
                "coords": coords,
                "bbox": bbox,
                "mask": mask,
                "color": int(next(iter(colors))) if len(colors) == 1 else None,
            }
        )
    return comps


def _find_frame(grid: np.ndarray, bg: int) -> dict:
    best = None
    best_score = None
    for comp in _grid_components(grid, bg):
        color = comp["color"]
        if color is None:
            continue
        r0, c0, r1, c1 = comp["bbox"]
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        h, w = patch.shape
        rows = [r for r in range(h) if bool(np.all(patch[r] == color))]
        cols = [c for c in range(w) if bool(np.all(patch[:, c] == color))]
        if len(rows) < 2 or len(cols) < 2:
            continue
        if rows[0] != 0 or rows[-1] != h - 1 or cols[0] != 0 or cols[-1] != w - 1:
            continue
        row_gaps = [rows[i + 1] - rows[i] - 1 for i in range(len(rows) - 1)]
        col_gaps = [cols[i + 1] - cols[i] - 1 for i in range(len(cols) - 1)]
        if min(row_gaps) <= 0 or min(col_gaps) <= 0:
            continue
        if len(set(row_gaps)) != 1 or len(set(col_gaps)) != 1:
            continue
        line_mask = np.zeros((h, w), dtype=bool)
        line_mask[rows, :] = True
        line_mask[:, cols] = True
        if not bool(np.all((patch == color) == line_mask)):
            continue
        score = ((len(rows) - 1) * (len(cols) - 1), h * w)
        if best_score is None or score > best_score:
            best_score = score
            best = {
                "color": color,
                "bbox": comp["bbox"],
                "rows": [r0 + r for r in rows],
                "cols": [c0 + c for c in cols],
                "mask_coords": comp["coords"],
            }
    assert best is not None, "expected one rectangular grey target frame"
    return best


def _slots_from_frame(frame: dict) -> tuple[list[dict], int, int, int, int]:
    rows = frame["rows"]
    cols = frame["cols"]
    cell_h = rows[1] - rows[0] - 1
    cell_w = cols[1] - cols[0] - 1
    slots = []
    for rr in range(len(rows) - 1):
        for cc in range(len(cols) - 1):
            r0 = rows[rr] + 1
            r1 = rows[rr + 1] - 1
            c0 = cols[cc] + 1
            c1 = cols[cc + 1] - 1
            slots.append({"coord": (rr, cc), "bbox": (r0, c0, r1, c1)})
    return slots, len(rows) - 1, len(cols) - 1, cell_h, cell_w


def _rect_descriptor(
    delta: np.ndarray,
    base_r0: int,
    base_c0: int,
    cell_h: int,
    cell_w: int,
    is_extra: bool,
):
    pts = np.argwhere(delta)
    if pts.size == 0:
        return None
    r0, c0 = map(int, pts.min(axis=0))
    r1, c1 = map(int, pts.max(axis=0))
    if not bool(np.all(delta[r0 : r1 + 1, c0 : c1 + 1])):
        return False

    if is_extra:
        if r1 == base_r0 - 1 and base_c0 < c0 and c1 < base_c0 + cell_w - 1:
            return ("up", c0 - base_c0, c1 - c0 + 1, base_r0 - r0)
        if r0 == base_r0 + cell_h and base_c0 < c0 and c1 < base_c0 + cell_w - 1:
            return ("down", c0 - base_c0, c1 - c0 + 1, r1 - (base_r0 + cell_h) + 1)
        if c1 == base_c0 - 1 and base_r0 < r0 and r1 < base_r0 + cell_h - 1:
            return ("left", r0 - base_r0, r1 - r0 + 1, base_c0 - c0)
        if c0 == base_c0 + cell_w and base_r0 < r0 and r1 < base_r0 + cell_h - 1:
            return ("right", r0 - base_r0, r1 - r0 + 1, c1 - (base_c0 + cell_w) + 1)
        return False

    if r0 == base_r0 and base_c0 < c0 and c1 < base_c0 + cell_w - 1:
        return ("up", c0 - base_c0, c1 - c0 + 1, r1 - r0 + 1)
    if r1 == base_r0 + cell_h - 1 and base_c0 < c0 and c1 < base_c0 + cell_w - 1:
        return ("down", c0 - base_c0, c1 - c0 + 1, r1 - r0 + 1)
    if c0 == base_c0 and base_r0 < r0 and r1 < base_r0 + cell_h - 1:
        return ("left", r0 - base_r0, r1 - r0 + 1, c1 - c0 + 1)
    if c1 == base_c0 + cell_w - 1 and base_r0 < r0 and r1 < base_r0 + cell_h - 1:
        return ("right", r0 - base_r0, r1 - r0 + 1, c1 - c0 + 1)
    return False


def _analyze_piece(mask: np.ndarray, cell_h: int, cell_w: int) -> tuple[tuple | None, tuple | None]:
    h, w = mask.shape
    assert h >= cell_h and w >= cell_w
    fits = set()
    for base_r0 in range(h - cell_h + 1):
        for base_c0 in range(w - cell_w + 1):
            base = np.zeros((h, w), dtype=bool)
            base[base_r0 : base_r0 + cell_h, base_c0 : base_c0 + cell_w] = True
            extra = _rect_descriptor(mask & ~base, base_r0, base_c0, cell_h, cell_w, True)
            notch = _rect_descriptor(base & ~mask, base_r0, base_c0, cell_h, cell_w, False)
            if extra is False or notch is False:
                continue
            if extra is None and notch is None:
                continue
            if extra is not None and notch is not None and extra[0] == notch[0]:
                continue
            fits.add((notch, extra))
    assert len(fits) == 1, "piece geometry must decode uniquely"
    return next(iter(fits))


def _pair_key(desc: tuple) -> tuple:
    side, offset, span, depth = desc
    axis = "h" if side in ("left", "right") else "v"
    return axis, int(offset), int(span), int(depth)


def _piece_records(grid: np.ndarray, bg: int, frame: dict, cell_h: int, cell_w: int) -> list[dict]:
    frame_mask = np.zeros_like(grid, dtype=bool)
    for r, c in frame["mask_coords"]:
        frame_mask[r, c] = True

    pieces = []
    for comp in _grid_components(grid, bg):
        if comp["color"] is None:
            continue
        if all(frame_mask[r, c] for r, c in comp["coords"]):
            continue
        color = comp["color"]
        patch = grid[
            comp["bbox"][0] : comp["bbox"][2] + 1,
            comp["bbox"][1] : comp["bbox"][3] + 1,
        ] == color
        notch, tab = _analyze_piece(patch, cell_h, cell_w)
        pieces.append(
            {
                "color": color,
                "bbox": comp["bbox"],
                "notch": notch,
                "tab": tab,
            }
        )
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered notch/tab fragments into the grey target rectangle.

    Pieces are monochrome rectangles of a common base size. Each piece may have
    one inset notch (incoming neighbor) and one outward tab (outgoing neighbor).
    Matching notch/tab signatures determine adjacency; the tab direction gives
    the neighbor's relative position. The grey frame on the right provides the
    final rectangular slot grid.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(np.bincount(g.ravel()).argmax())

    frame = _find_frame(g, bg)
    slots, n_rows, n_cols, cell_h, cell_w = _slots_from_frame(frame)
    pieces = _piece_records(g, bg, frame, cell_h, cell_w)
    assert len(pieces) == len(slots), "piece count must match target slots"

    notch_map: dict[tuple, int] = {}
    for idx, piece in enumerate(pieces):
        if piece["notch"] is None:
            continue
        desc = piece["notch"]
        notch_map[(desc[0],) + _pair_key(desc)] = idx

    out_edge: dict[int, tuple[int, str]] = {}
    in_edge: dict[int, int] = {}
    for idx, piece in enumerate(pieces):
        if piece["tab"] is None:
            continue
        desc = piece["tab"]
        key = (OPPOSITE[desc[0]],) + _pair_key(desc)
        assert key in notch_map, "every tab must have one matching notch"
        nxt = notch_map[key]
        assert idx not in out_edge and nxt not in in_edge
        out_edge[idx] = (nxt, desc[0])
        in_edge[nxt] = idx

    starts = [idx for idx in range(len(pieces)) if idx not in in_edge]
    assert len(starts) == 1, "expected one directed chain start"
    start = starts[0]

    coords = {start: (0, 0)}
    cur = start
    seen = {start}
    while cur in out_edge:
        nxt, side = out_edge[cur]
        assert nxt not in seen
        dr, dc = DIR_STEP[side]
        pr, pc = coords[cur]
        coords[nxt] = (pr + dr, pc + dc)
        seen.add(nxt)
        cur = nxt

    assert len(coords) == len(pieces), "chain must cover every piece"

    min_r = min(r for r, _ in coords.values())
    min_c = min(c for _, c in coords.values())
    norm = {idx: (r - min_r, c - min_c) for idx, (r, c) in coords.items()}

    occupied = set(norm.values())
    assert occupied == {(r, c) for r in range(n_rows) for c in range(n_cols)}

    slot_map = {slot["coord"]: slot["bbox"] for slot in slots}
    out = np.full_like(g, bg)
    for r, c in frame["mask_coords"]:
        out[r, c] = frame["color"]
    for idx, piece in enumerate(pieces):
        r0, c0, r1, c1 = slot_map[norm[idx]]
        out[r0 : r1 + 1, c0 : c1 + 1] = piece["color"]
    return out


LAYOUTS = [(2, 2), (2, 3), (3, 2), (2, 4), (4, 2)]
CELL_H_RANGE = (5, 8)
CELL_W_RANGE = (5, 8)
FRAME_RIGHT_MARGIN = (1, 3)
FRAME_GAP_RANGE = (3, 6)
TOP_BOTTOM_MARGIN = (1, 4)
GRID_EXTRA_HEIGHT = (0, 6)
GEN_TRIES = 400
PIECE_PADDING = 1


def _direction(a: tuple[int, int], b: tuple[int, int]) -> str:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    for side, step in DIR_STEP.items():
        if step == (dr, dc):
            return side
    raise AssertionError("non-cardinal adjacency")


def _snake_path(rows: int, cols: int, by_rows: bool) -> list[tuple[int, int]]:
    path = []
    if by_rows:
        for r in range(rows):
            col_iter = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
            for c in col_iter:
                path.append((r, c))
    else:
        for c in range(cols):
            row_iter = range(rows) if c % 2 == 0 else range(rows - 1, -1, -1)
            for r in row_iter:
                path.append((r, c))
    return path


def _transform_path(path: list[tuple[int, int]], rows: int, cols: int) -> list[tuple[int, int]]:
    out = path[:]
    if random.random() < 0.5:
        out = [(rows - 1 - r, c) for r, c in out]
    if random.random() < 0.5:
        out = [(r, cols - 1 - c) for r, c in out]
    if random.random() < 0.5:
        out = list(reversed(out))
    return out


def _signature_choices(side_len: int, max_depth: int) -> list[tuple[int, int, int]]:
    choices = []
    for offset in range(1, side_len - 1):
        for span in range(1, side_len - offset):
            if offset + span >= side_len:
                continue
            for depth in range(1, max_depth + 1):
                choices.append((offset, span, depth))
    return choices


def _connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    assert pts.size > 0
    start = tuple(map(int, pts[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR_STEP.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _apply_notch(mask: np.ndarray, desc: tuple | None) -> None:
    if desc is None:
        return
    side, offset, span, depth = desc
    if side == "up":
        mask[0:depth, offset : offset + span] = False
    elif side == "down":
        mask[-depth:, offset : offset + span] = False
    elif side == "left":
        mask[offset : offset + span, 0:depth] = False
    else:
        mask[offset : offset + span, -depth:] = False


def _piece_mask(cell_h: int, cell_w: int, notch: tuple | None, tab: tuple | None) -> np.ndarray:
    cells = {(r, c) for r in range(cell_h) for c in range(cell_w)}
    if notch is not None:
        side, offset, span, depth = notch
        if side == "up":
            for r in range(depth):
                for c in range(offset, offset + span):
                    cells.discard((r, c))
        elif side == "down":
            for r in range(cell_h - depth, cell_h):
                for c in range(offset, offset + span):
                    cells.discard((r, c))
        elif side == "left":
            for r in range(offset, offset + span):
                for c in range(depth):
                    cells.discard((r, c))
        else:
            for r in range(offset, offset + span):
                for c in range(cell_w - depth, cell_w):
                    cells.discard((r, c))

    if tab is not None:
        side, offset, span, depth = tab
        if side == "up":
            for r in range(-depth, 0):
                for c in range(offset, offset + span):
                    cells.add((r, c))
        elif side == "down":
            for r in range(cell_h, cell_h + depth):
                for c in range(offset, offset + span):
                    cells.add((r, c))
        elif side == "left":
            for r in range(offset, offset + span):
                for c in range(-depth, 0):
                    cells.add((r, c))
        else:
            for r in range(offset, offset + span):
                for c in range(cell_w, cell_w + depth):
                    cells.add((r, c))

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    assert _connected(mask)
    return mask


def _draw_frame(grid: np.ndarray, top: int, left: int, rows: int, cols: int, cell_h: int, cell_w: int) -> None:
    h = rows * cell_h + rows + 1
    w = cols * cell_w + cols + 1
    for i in range(rows + 1):
        rr = top + i * (cell_h + 1)
        grid[rr, left : left + w] = FRAME
    for j in range(cols + 1):
        cc = left + j * (cell_w + 1)
        grid[top : top + h, cc] = FRAME


def _slot_bbox(top: int, left: int, cell_h: int, cell_w: int, coord: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = coord
    r0 = top + 1 + r * (cell_h + 1)
    c0 = left + 1 + c * (cell_w + 1)
    return r0, c0, r0 + cell_h - 1, c0 + cell_w - 1


def _try_place_patch(
    grid: np.ndarray,
    occ: np.ndarray,
    patch: np.ndarray,
    max_col_exclusive: int,
) -> tuple[int, int] | None:
    h, w = grid.shape
    ph, pw = patch.shape
    cells = np.argwhere(patch != BG)
    if pw >= max_col_exclusive:
        return None
    positions = []
    for r0 in range(h - ph + 1):
        for c0 in range(max_col_exclusive - pw):
            positions.append((r0, c0))
    random.shuffle(positions)
    for r0, c0 in positions:
        ok = True
        for rr, cc in cells:
            ar, ac = int(r0 + rr), int(c0 + cc)
            for nr in range(max(0, ar - PIECE_PADDING), min(h, ar + PIECE_PADDING + 1)):
                for nc in range(max(0, ac - PIECE_PADDING), min(w, ac + PIECE_PADDING + 1)):
                    if occ[nr, nc]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            continue
        grid[r0 : r0 + ph, c0 : c0 + pw][patch != BG] = patch[patch != BG]
        occ[r0 : r0 + ph, c0 : c0 + pw] |= patch != BG
        return r0, c0
    return None


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    rows, cols = random.choice(LAYOUTS)
    by_rows = random.random() < 0.5
    cell_h = random.randrange(*CELL_H_RANGE)
    cell_w = random.randrange(*CELL_W_RANGE)

    path = _transform_path(_snake_path(rows, cols, by_rows), rows, cols)
    horiz_edges = sum(_direction(a, b) in ("left", "right") for a, b in zip(path, path[1:]))
    vert_edges = (len(path) - 1) - horiz_edges

    horiz_choices = _signature_choices(cell_h, min(2, cell_w - 2))
    vert_choices = _signature_choices(cell_w, min(2, cell_h - 2))
    if horiz_edges > len(horiz_choices) or vert_edges > len(vert_choices):
        return None
    random.shuffle(horiz_choices)
    random.shuffle(vert_choices)

    notch_map: dict[tuple[int, int], tuple | None] = {coord: None for coord in path}
    tab_map: dict[tuple[int, int], tuple | None] = {coord: None for coord in path}
    h_idx = 0
    v_idx = 0
    for a, b in zip(path, path[1:]):
        side = _direction(a, b)
        if side in ("left", "right"):
            offset, span, depth = horiz_choices[h_idx]
            h_idx += 1
        else:
            offset, span, depth = vert_choices[v_idx]
            v_idx += 1
        tab_desc = (side, offset, span, depth)
        notch_desc = (OPPOSITE[side], offset, span, depth)
        tab_map[a] = tab_desc
        notch_map[b] = notch_desc

    color_pool = [c for c in range(1, 10) if c != FRAME]
    if len(path) > len(color_pool):
        return None
    random.shuffle(color_pool)
    colors = color_pool[: len(path)]
    coord_to_color = {coord: colors[i] for i, coord in enumerate(path)}

    piece_patches = []
    for coord in path:
        mask = _piece_mask(cell_h, cell_w, notch_map[coord], tab_map[coord])
        if _analyze_piece(mask, cell_h, cell_w) != (notch_map[coord], tab_map[coord]):
            return None
        patch = np.full(mask.shape, BG, dtype=int)
        patch[mask] = coord_to_color[coord]
        piece_patches.append((coord, patch))

    frame_h = rows * cell_h + rows + 1
    frame_w = cols * cell_w + cols + 1
    piece_max_h = max(p.shape[0] for _, p in piece_patches)
    piece_max_w = max(p.shape[1] for _, p in piece_patches)
    total_piece_area = sum(int(np.count_nonzero(p)) for _, p in piece_patches)

    grid_h_min = max(frame_h + 2, piece_max_h + 2, 16)
    grid_h = grid_h_min + random.randint(*GRID_EXTRA_HEIGHT)
    left_band_min = max(piece_max_w + 2, total_piece_area // max(4, cell_h) + 8)
    gap = random.randint(*FRAME_GAP_RANGE)
    right_margin = random.randint(*FRAME_RIGHT_MARGIN)
    grid_w = left_band_min + gap + frame_w + right_margin

    top_margin = random.randint(*TOP_BOTTOM_MARGIN)
    if frame_h + top_margin > grid_h:
        grid_h = frame_h + top_margin
    frame_top = random.randint(0, grid_h - frame_h)
    frame_left = grid_w - frame_w - right_margin
    if frame_left <= piece_max_w + 2:
        return None

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    out = np.full((grid_h, grid_w), BG, dtype=int)
    _draw_frame(inp, frame_top, frame_left, rows, cols, cell_h, cell_w)
    _draw_frame(out, frame_top, frame_left, rows, cols, cell_h, cell_w)

    for coord in path:
        r0, c0, r1, c1 = _slot_bbox(frame_top, frame_left, cell_h, cell_w, coord)
        out[r0 : r1 + 1, c0 : c1 + 1] = coord_to_color[coord]

    occ = inp != BG
    random.shuffle(piece_patches)
    for _coord, patch in piece_patches:
        if _try_place_patch(inp, occ, patch, frame_left - 1) is None:
            return None

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_TRIES):
        built = _build_attempt()
        if built is not None:
            return built[0]
    raise RuntimeError(f"failed to generate a valid puzzle after {GEN_TRIES} attempts")
