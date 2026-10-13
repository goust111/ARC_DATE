import random
from collections import Counter, deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5
PIECE_COLORS = (1, 4, 6, 7, 8, 9)
GRID_SHAPES = ((2, 2), (2, 3), (3, 2))
CELL_H_RANGE = (4, 6)
CELL_W_RANGE = (4, 6)
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                comp.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _edge_signature(arr: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> tuple[tuple[int, ...], ...]:
    def row_offsets(row: int, start: int, stop: int, color: int) -> tuple[int, ...]:
        if row < 0 or row >= arr.shape[0] or stop <= start:
            return ()
        return tuple(int(i) + 1 for i, v in enumerate(arr[row, start:stop]) if int(v) == color)

    def col_offsets(col: int, start: int, stop: int, color: int) -> tuple[int, ...]:
        if col < 0 or col >= arr.shape[1] or stop <= start:
            return ()
        return tuple(int(i) + 1 for i, v in enumerate(arr[start:stop, col]) if int(v) == color)

    return (
        row_offsets(r0 - 1, c0 + 1, c1, GREEN),
        row_offsets(r0, c0 + 1, c1, RED),
        col_offsets(c1 + 1, r0 + 1, r1, GREEN),
        col_offsets(c1, r0 + 1, r1, RED),
        row_offsets(r1 + 1, c0 + 1, c1, GREEN),
        row_offsets(r1, c0 + 1, c1, RED),
        col_offsets(c0 - 1, r0 + 1, r1, GREEN),
        col_offsets(c0, r0 + 1, r1, RED),
    )


def _complement_signature(sig: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    return (sig[1], sig[0], sig[3], sig[2], sig[5], sig[4], sig[7], sig[6])


def _frame_data(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int], list[int], list[int]]:
    comps = _components(grid != bg)
    assert comps, "expected non-background objects"
    frame_comp = max(comps, key=len)
    fr0, fc0, fr1, fc1 = _bbox(frame_comp)

    frame_colors = [int(grid[r, c]) for r, c in frame_comp if int(grid[r, c]) not in (RED, GREEN)]
    assert frame_colors, "frame must contain a base color"
    frame_color = Counter(frame_colors).most_common(1)[0][0]

    hlines = [
        r
        for r in range(fr0, fr1 + 1)
        if np.all(np.isin(grid[r, fc0 : fc1 + 1], (frame_color, GREEN)))
    ]
    vlines = [
        c
        for c in range(fc0, fc1 + 1)
        if np.all(np.isin(grid[fr0 : fr1 + 1, c], (frame_color, GREEN)))
    ]
    assert len(hlines) >= 2 and len(vlines) >= 2, "frame must define a slot grid"
    assert len(set(np.diff(hlines))) == 1 and len(set(np.diff(vlines))) == 1, "slots must be equal-sized"
    return frame_color, (fr0, fc0, fr1, fc1), hlines, vlines


def _extract_slots(grid: np.ndarray, hlines: list[int], vlines: list[int]) -> list[dict]:
    slots = []
    for i in range(len(hlines) - 1):
        for j in range(len(vlines) - 1):
            r0 = hlines[i] + 1
            r1 = hlines[i + 1] - 1
            c0 = vlines[j] + 1
            c1 = vlines[j + 1] - 1
            slots.append(
                {
                    "r0": int(r0),
                    "r1": int(r1),
                    "c0": int(c0),
                    "c1": int(c1),
                    "h": int(r1 - r0 + 1),
                    "w": int(c1 - c0 + 1),
                    "sig": _edge_signature(grid, r0, c0, r1, c1),
                }
            )
    return slots


def _extract_pieces(grid: np.ndarray, bg: int, frame_bbox: tuple[int, int, int, int]) -> list[dict]:
    fr0, fc0, fr1, fc1 = frame_bbox
    pieces = []
    for comp in _components(grid != bg):
        r0, c0, r1, c1 = _bbox(comp)
        if (r0, c0, r1, c1) == frame_bbox:
            continue
        if fr0 <= r0 and r1 <= fr1 and fc0 <= c0 and c1 <= fc1:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        body_mask = (patch != bg) & (patch != RED) & (patch != GREEN)
        body_vals = [int(v) for v in patch[body_mask].tolist()]
        assert body_vals, "piece must contain a colored body"
        counts = Counter(body_vals)
        assert len(counts) == 1, "piece bodies must be monochrome"
        color = next(iter(counts))
        body_coords = np.argwhere(patch == color)
        br0, bc0 = body_coords.min(axis=0)
        br1, bc1 = body_coords.max(axis=0)
        pieces.append(
            {
                "color": int(color),
                "h": int(br1 - br0 + 1),
                "w": int(bc1 - bc0 + 1),
                "sig": _edge_signature(patch, int(br0), int(bc0), int(br1), int(bc1)),
            }
        )
    return pieces


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frame_color, frame_bbox, hlines, vlines = _frame_data(g, bg)
    slots = _extract_slots(g, hlines, vlines)
    pieces = _extract_pieces(g, bg, frame_bbox)
    assert len(pieces) == len(slots), "piece count must equal slot count"

    piece_map: dict[tuple[int, int, tuple[tuple[int, ...], ...]], int] = {}
    for piece in pieces:
        key = (piece["h"], piece["w"], piece["sig"])
        assert key not in piece_map, "piece signatures must be unique"
        piece_map[key] = int(piece["color"])

    out = np.full_like(g, bg)
    fr0, fc0, fr1, fc1 = frame_bbox
    for r in hlines:
        out[r, fc0 : fc1 + 1] = frame_color
    for c in vlines:
        out[fr0 : fr1 + 1, c] = frame_color

    for slot in slots:
        key = (slot["h"], slot["w"], _complement_signature(slot["sig"]))
        assert key in piece_map, "every slot must have a complementary piece"
        out[slot["r0"] : slot["r1"] + 1, slot["c0"] : slot["c1"] + 1] = piece_map[key]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered puzzle pieces into the matching framed slots by complementary tabs and dents.
    """
    return _solve(grid)


def _random_edge(positions: list[int], rng: random.Random) -> tuple[tuple[int, ...], tuple[int, ...]]:
    max_count = min(2, len(positions))
    count = rng.randint(0, max_count)
    if count == 0:
        return (), ()
    chosen = sorted(rng.sample(positions, count))
    outs = tuple(p for p in chosen if rng.random() < 0.5)
    ins = tuple(p for p in chosen if p not in outs)
    return outs, ins


def _random_slot_signature(cell_h: int, cell_w: int, rng: random.Random) -> tuple[tuple[int, ...], ...]:
    top_out, top_in = _random_edge(list(range(1, cell_w - 1)), rng)
    right_out, right_in = _random_edge(list(range(1, cell_h - 1)), rng)
    bottom_out, bottom_in = _random_edge(list(range(1, cell_w - 1)), rng)
    left_out, left_in = _random_edge(list(range(1, cell_h - 1)), rng)
    return (top_out, top_in, right_out, right_in, bottom_out, bottom_in, left_out, left_in)


def _signature_marker_count(sig: tuple[tuple[int, ...], ...]) -> int:
    return sum(len(part) for part in sig)


def _draw_frame_patch(
    rows: int,
    cols: int,
    cell_h: int,
    cell_w: int,
    slot_sigs: list[tuple[tuple[int, ...], ...]],
) -> tuple[np.ndarray, list[int], list[int]]:
    board_h = rows * cell_h + rows + 1
    board_w = cols * cell_w + cols + 1
    patch = np.full((board_h, board_w), BLACK, dtype=int)
    hlines = [i * (cell_h + 1) for i in range(rows + 1)]
    vlines = [j * (cell_w + 1) for j in range(cols + 1)]

    patch[hlines, :] = GREY
    patch[:, vlines] = GREY

    idx = 0
    for i in range(rows):
        for j in range(cols):
            sig = slot_sigs[idx]
            idx += 1
            r0 = hlines[i] + 1
            r1 = hlines[i + 1] - 1
            c0 = vlines[j] + 1
            c1 = vlines[j + 1] - 1
            for off in sig[0]:
                patch[r0 - 1, c0 + off] = GREEN
            for off in sig[1]:
                patch[r0, c0 + off] = RED
            for off in sig[2]:
                patch[r0 + off, c1 + 1] = GREEN
            for off in sig[3]:
                patch[r0 + off, c1] = RED
            for off in sig[4]:
                patch[r1 + 1, c0 + off] = GREEN
            for off in sig[5]:
                patch[r1, c0 + off] = RED
            for off in sig[6]:
                patch[r0 + off, c0 - 1] = GREEN
            for off in sig[7]:
                patch[r0 + off, c0] = RED

    return patch, hlines, vlines


def _draw_piece_patch(color: int, cell_h: int, cell_w: int, sig: tuple[tuple[int, ...], ...]) -> np.ndarray:
    top_pad = 1 if sig[0] else 0
    right_pad = 1 if sig[2] else 0
    bottom_pad = 1 if sig[4] else 0
    left_pad = 1 if sig[6] else 0
    patch = np.full((cell_h + top_pad + bottom_pad, cell_w + left_pad + right_pad), BLACK, dtype=int)
    r0 = top_pad
    r1 = r0 + cell_h - 1
    c0 = left_pad
    c1 = c0 + cell_w - 1
    patch[r0 : r1 + 1, c0 : c1 + 1] = color

    for off in sig[0]:
        patch[r0 - 1, c0 + off] = GREEN
    for off in sig[1]:
        patch[r0, c0 + off] = RED
    for off in sig[2]:
        patch[r0 + off, c1 + 1] = GREEN
    for off in sig[3]:
        patch[r0 + off, c1] = RED
    for off in sig[4]:
        patch[r1 + 1, c0 + off] = GREEN
    for off in sig[5]:
        patch[r1, c0 + off] = RED
    for off in sig[6]:
        patch[r0 + off, c0 - 1] = GREEN
    for off in sig[7]:
        patch[r0 + off, c0] = RED

    return patch


def _mark_halo(occ: np.ndarray, coords: np.ndarray) -> None:
    h, w = occ.shape
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < h and 0 <= nc < w:
                    occ[nr, nc] = True


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    rr, cc = np.where(patch != BLACK)
    grid[rr + r0, cc + c0] = patch[rr, cc]


def _expected_output(
    shape: tuple[int, int],
    frame_origin: tuple[int, int],
    hlines: list[int],
    vlines: list[int],
    cell_h: int,
    cell_w: int,
    slot_colors: list[int],
) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    fr0, fc0 = frame_origin
    for r in hlines:
        out[fr0 + r, fc0 + vlines[0] : fc0 + vlines[-1] + 1] = GREY
    for c in vlines:
        out[fr0 + hlines[0] : fr0 + hlines[-1] + 1, fc0 + c] = GREY

    idx = 0
    for i in range(len(hlines) - 1):
        for j in range(len(vlines) - 1):
            r0 = fr0 + hlines[i] + 1
            c0 = fc0 + vlines[j] + 1
            out[r0 : r0 + cell_h, c0 : c0 + cell_w] = slot_colors[idx]
            idx += 1
    return out


def _place_pieces(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    piece_patches: list[np.ndarray],
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    occ = np.zeros((h, w), dtype=bool)
    occ[[0, -1], :] = True
    occ[:, [0, -1]] = True

    fr0, fc0, fr1, fc1 = frame_bbox
    board_coords = np.array([(r, c) for r in range(fr0, fr1 + 1) for c in range(fc0, fc1 + 1)], dtype=int)
    _mark_halo(occ, board_coords)

    ordering = sorted(piece_patches, key=lambda p: int(np.count_nonzero(p != BLACK)), reverse=True)

    for patch in ordering:
        coords = np.argwhere(patch != BLACK)
        candidates = []
        ph, pw = patch.shape
        for r0 in range(0, h - ph + 1):
            for c0 in range(0, w - pw + 1):
                abs_coords = coords + np.array([r0, c0], dtype=int)
                if np.any(occ[abs_coords[:, 0], abs_coords[:, 1]]):
                    continue
                candidates.append((int(r0), int(c0)))
        if not candidates:
            return False
        r0, c0 = rng.choice(candidates)
        _paste_patch(grid, patch, r0, c0)
        _mark_halo(occ, coords + np.array([r0, c0], dtype=int))
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rows, cols = rng.choice(GRID_SHAPES)
    slots_n = rows * cols
    cell_h = rng.randint(*CELL_H_RANGE)
    cell_w = rng.randint(*CELL_W_RANGE)

    raw_slot_sigs: list[tuple[tuple[int, ...], ...]] = []
    seen = set()
    while len(raw_slot_sigs) < slots_n:
        sig = _random_slot_signature(cell_h, cell_w, rng)
        if _signature_marker_count(sig) == 0:
            continue
        if sig in seen:
            continue
        seen.add(sig)
        raw_slot_sigs.append(sig)

    board_patch, hlines, vlines = _draw_frame_patch(rows, cols, cell_h, cell_w, raw_slot_sigs)
    visible_slots = _extract_slots(board_patch, hlines, vlines)
    visible_sigs = [slot["sig"] for slot in visible_slots]
    if len(set(visible_sigs)) != slots_n:
        return None

    total_out = sum(len(sig[0]) + len(sig[2]) + len(sig[4]) + len(sig[6]) for sig in visible_sigs)
    total_in = sum(len(sig[1]) + len(sig[3]) + len(sig[5]) + len(sig[7]) for sig in visible_sigs)
    if total_out == 0 or total_in == 0 or any(_signature_marker_count(sig) == 0 for sig in visible_sigs):
        return None

    colors = rng.sample(list(PIECE_COLORS), slots_n)
    piece_patches = [
        _draw_piece_patch(color, cell_h, cell_w, _complement_signature(sig))
        for color, sig in zip(colors, visible_sigs)
    ]

    board_h, board_w = board_patch.shape
    min_h = max(board_h + 6, 20)
    min_w = max(board_w + 6, 20)
    if min_h > 30 or min_w > 30:
        return None
    height = rng.randint(min_h, 30)
    width = rng.randint(min_w, 30)
    if height - board_h < 4 or width - board_w < 4:
        return None

    fr0 = rng.randint(2, height - board_h - 2)
    fc0 = rng.randint(2, width - board_w - 2)
    grid = np.full((height, width), BLACK, dtype=int)
    grid[fr0 : fr0 + board_h, fc0 : fc0 + board_w] = board_patch

    frame_bbox = (fr0, fc0, fr0 + board_h - 1, fc0 + board_w - 1)
    if not _place_pieces(grid, frame_bbox, piece_patches, rng):
        return None

    expected = _expected_output(grid.shape, (fr0, fc0), hlines, vlines, cell_h, cell_w, colors)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after retries")
