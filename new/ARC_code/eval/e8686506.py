import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location
import random


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            h = n // w
            out.append((h, w))
    return out


def _build_requirements(
    H: int,
    W: int,
    comps_by_color: dict[int, list[list[tuple[int, int]]]],
) -> np.ndarray:
    """
    Build per-cell required color constraints (req[r,c] == -1 means unconstrained).

    Heuristics supported by training examples:
      - If there is a color with exactly 4 singleton components, those are the four corners.
      - If there are full horizontal bars of length W-2 in the input, use them as the top/bottom edges
        between the corners (ordered by their vertical position in the input).
      - If there is a color with exactly two full vertical line components (width=1) in the input,
        place them on the left/right edges centered vertically.
    """
    req = np.full((H, W), -1, dtype=int)

    # 4 singleton components -> corners
    corner_colors = [
        col
        for col, comps in comps_by_color.items()
        if len(comps) == 4 and all(len(p) == 1 for p in comps)
    ]
    assert len(corner_colors) <= 1, "corner singleton color must be unique when present"
    corner_color = corner_colors[0] if corner_colors else None
    if corner_color is not None:
        req[0, 0] = corner_color
        req[0, W - 1] = corner_color
        req[H - 1, 0] = corner_color
        req[H - 1, W - 1] = corner_color

    # Horizontal bar(s) of length W-2 (filled, height 1)
    bar_colors: list[tuple[int, int]] = []  # (min_input_row, color)
    if W >= 3 and corner_color is not None:
        for col, comps in comps_by_color.items():
            for pts in comps:
                r0, c0, r1, c1 = bbox_coords(pts)
                hh, ww = r1 - r0 + 1, c1 - c0 + 1
                if hh == 1 and ww == (W - 2) and len(pts) == ww:
                    bar_colors.append((r0, col))
                    break
        if bar_colors:
            assert len(bar_colors) <= 2, "horizontal edge bar colors must be top/bottom only"
            bar_colors.sort()
            top_col = bar_colors[0][1]
            bot_col = bar_colors[-1][1]
            req[0, 1 : W - 1] = top_col
            req[H - 1, 1 : W - 1] = bot_col

    # Two full vertical line components -> left/right edges, centered vertically
    vertical_line_candidates = []
    for col, comps in comps_by_color.items():
        if len(comps) != 2:
            continue
        sigs = [_piece_sig(pts) for pts in comps]
        (h0, w0, coords0), (h1, w1, coords1) = sigs[0], sigs[1]
        if w0 == 1 and w1 == 1 and h0 == h1 and coords0 == coords1 and h0 > 1:
            vertical_line_candidates.append((col, h0))
    assert len(vertical_line_candidates) <= 1, "paired vertical edge-line color must be unique when present"
    if vertical_line_candidates:
        col, L = vertical_line_candidates[0]
        top = (H - L) // 2
        assert 0 <= top and top + L <= H, "paired vertical edge lines must fit in the output height"
        req[top : top + L, 0] = col
        req[top : top + L, W - 1] = col

    return req


def _assemble(
    types: list[dict],
    H: int,
    W: int,
    req: np.ndarray,
    disallow_touch: set[int] | None = None,
) -> np.ndarray:
    board = np.full((H, W), -1, dtype=int)
    counts = [int(t["cnt"]) for t in types]

    def mirror_ok(r: int, c: int, col: int) -> bool:
        mc = W - 1 - c
        v = board[r, mc]
        return (v == -1) or (v == col)

    def can_place(ti: int, r0: int, c0: int) -> bool:
        t = types[ti]
        col = int(t["col"])
        is_singleton = len(t["coords"]) == 1
        for dr, dc in t["coords"]:
            r, c = r0 + dr, c0 + dc
            if board[r, c] != -1:
                return False
            if req[r, c] != -1 and int(req[r, c]) != col:
                return False
            if not mirror_ok(r, c, col):
                return False
            if is_singleton and not (r == 0 or r == H - 1 or c == 0 or c == W - 1):
                return False
            if disallow_touch and col in disallow_touch:
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < H and 0 <= cc < W and board[rr, cc] == col:
                        return False
        return True

    def place(ti: int, r0: int, c0: int) -> list[tuple[int, int]]:
        t = types[ti]
        col = int(t["col"])
        changed = []
        for dr, dc in t["coords"]:
            r, c = r0 + dr, c0 + dc
            board[r, c] = col
            changed.append((r, c))
        return changed

    def unplace(changed: list[tuple[int, int]]) -> None:
        for r, c in changed:
            board[r, c] = -1

    def placements_for(ti: int) -> list[tuple[int, int]]:
        t = types[ti]
        return [(r0, c0)
                for r0 in range(H - int(t["h"]) + 1)
                for c0 in range(W - int(t["w"]) + 1)
                if can_place(ti, r0, c0)]

    # Pure constraint propagation: each step, every type must be forced (1 position)
    # or batch-forced (positions == cnt). No backtracking. Asserts if stuck.
    changed = True
    while changed:
        changed = False
        for ti in range(len(types)):
            if counts[ti] <= 0:
                continue
            places = placements_for(ti)
            assert len(places) > 0, "assembly dead end: a piece has no valid placement"
            if len(places) == 1:
                counts[ti] -= 1
                place(ti, places[0][0], places[0][1])
                changed = True
            elif len(places) == counts[ti]:
                for r0, c0 in places:
                    counts[ti] -= 1
                    place(ti, r0, c0)
                changed = True
    assert all(c == 0 for c in counts), "assembly not purely propagation-solvable"
    assert np.array_equal(board, board[:, ::-1]), "assembled board must be LR-symmetric"
    return board


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered colored fragments into a left-right symmetric rectangle, jigsaw-style.

    Input structure:
    - bg = most common color; remaining colors form small sprite fragments.
    - Optional roles: 4 singleton pixels of one color mark corners; horizontal bars mark top/bottom edges; paired vertical lines mark left/right edges.

    Rule:
    - Collect all non-bg connected components as jigsaw pieces.
    - Determine output rectangle dimensions from total piece area.
    - Pack pieces into the rectangle so it is exactly mirror-symmetric (left-right) and same-color pieces never 4-touch.

    Key insight: Left-right symmetry plus the no-same-color-touching constraint make the assembly unique; corner/edge markers anchor it when present.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    # Extract pieces (connected components per color)
    comps_by_color: dict[int, list[list[tuple[int, int]]]] = {}
    pieces = []
    color_piece_count: dict[int, int] = {}

    for col in sorted(set(map(int, np.unique(g)))):
        if col == bg:
            continue
        comps = components_mask(g == col, connectivity=4)
        comps_by_color[col] = comps
        color_piece_count[col] = len(comps)
        for pts in comps:
            _, _, coords = _piece_sig(pts)
            pieces.append((col, coords))

    # Group identical pieces into types
    from collections import Counter

    ctr = Counter(pieces)
    types = []
    for (col, coords), cnt in sorted(ctr.items(), key=lambda kv: (-len(kv[0][1]), kv[0][0], kv[0][1])):
        arr = np.array(coords, dtype=int)
        h = int(arr[:, 0].max() + 1)
        w = int(arr[:, 1].max() + 1)
        types.append({"col": int(col), "coords": coords, "h": h, "w": w, "cnt": int(cnt)})

    total_area = sum(int(t["cnt"]) * len(t["coords"]) for t in types)
    assert total_area > 0, "total_area must be positive: grid must contain non-background pieces"

    # Disallow 4-neigh touching between different pieces of same color (preserve component counts)
    # Eval inputs use the unique closest-to-square portrait factor pair as the output shape.
    dims = [(h, w) for (h, w) in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factor pair must exist for the total piece area"
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    H, W = best_dims[0]

    # Frame constraint: under 8-connectivity, 1-2 pieces must exist whose union bbox
    # equals the output rectangle (H x W). Their relative positions are fixed in output.
    comps8_all = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=8):
            comps8_all.append((int(col), pts))
    def _is_lr_symmetric(parts, h, w):
        """Check if the union of frame parts forms an LR-symmetric bitmap."""
        all_pts = np.concatenate([p for _, p in parts])
        r0, c0 = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
        bmp = np.zeros((h, w), dtype=bool)
        for _, pts in parts:
            for r, c in pts:
                bmp[int(r) - r0, int(c) - c0] = True
        return np.array_equal(bmp, bmp[:, ::-1])

    frame_parts: list[tuple[int, np.ndarray]] = []
    for i, (c1, p1) in enumerate(comps8_all):
        r0, c0, r1, c1b = bbox_coords(p1)
        if r1 - r0 + 1 == H and c1b - c0 + 1 == W:
            if _is_lr_symmetric([(c1, p1)], H, W):
                frame_parts = [(c1, p1)]
                break
        for j in range(i + 1, len(comps8_all)):
            c2, p2 = comps8_all[j]
            union = np.concatenate([p1, p2])
            ur0, uc0, ur1, uc1 = bbox_coords(union)
            if ur1 - ur0 + 1 == H and uc1 - uc0 + 1 == W:
                if _is_lr_symmetric([(c1, p1), (c2, p2)], H, W):
                    frame_parts = [(c1, p1), (c2, p2)]
                    break
        if frame_parts:
            break
    assert frame_parts, "input must contain 1-2 LR-symmetric pieces (8-conn) whose union bbox equals output dims"

    req = _build_requirements(H, W, comps_by_color)

    # Lock frame cells into req: their output positions are determined by input-relative positions.
    all_frame_pts = np.concatenate([p for _, p in frame_parts])
    fr0, fc0 = int(all_frame_pts[:, 0].min()), int(all_frame_pts[:, 1].min())
    for col, pts in frame_parts:
        for r, c in pts:
            or_, oc = int(r) - fr0, int(c) - fc0
            if 0 <= or_ < H and 0 <= oc < W:
                req[or_, oc] = col

    disallow_touch = {col for col, n in color_piece_count.items() if n > 1}
    sol = _assemble(types, H, W, req, disallow_touch)
    return sol.astype(int)

# --------------- generator (o5x) ---------------

# === CFG: Output dimensions ===
_AREA_MIN = 25               # minimum total piece area
_AREA_MAX = 50               # maximum total piece area (exclusive upper in range)
_WIDTH_REQUIRED = 5           # all valid outputs have width 5
_HEIGHT_MAX = 10              # maximum output height

# === CFG: Color and palette ===
_N_COLORS_MIN = 4             # min distinct colors in abstract output
_N_COLORS_MAX = 6             # max distinct colors in abstract output
_TWO_FRAME_MIN_H = 6          # min height to allow 2-color frame
_TWO_FRAME_MIN_COLORS = 5     # min palette size to allow 2-color frame
_TWO_FRAME_PROB = 0.35         # probability of splitting frame into 2 colors
_CENTER_CAP_PROB = 0.5         # probability of capping center col at top/bottom

# === CFG: Piece constraints ===
_MAX_PIECE_AREA = 6           # maximum area of any single 4-conn piece
_MAX_CHUNK_SIZE = 6           # max cells per hole chunk when splitting large holes
_PIECE_COUNT_MIN = 9          # min total 4-conn components in output
_PIECE_COUNT_MAX = 17         # max total 4-conn components in output

# === CFG: Canvas / scatter ===
_CANVAS_MIN = 13              # min canvas dimension (rows and cols)
_CANVAS_MAX = 15              # max canvas dimension (rows and cols)
_SCATTER_PADDING = 1          # Chebyshev padding between scattered pieces

# === CFG: Retry ===
_GEN_MAX_RETRIES = 300        # max attempts in generate_input()


def _valid_dims():
    """Precompute valid (H, W) pairs: unique closest-to-square portrait factorization with W=5."""
    results = []
    for area in range(_AREA_MIN, _AREA_MAX + 1):
        dims = [(area // w, w) for w in range(1, area + 1) if area % w == 0 and area // w >= w]
        best_gap = min(abs(h - w) for h, w in dims)
        best = [(h, w) for h, w in dims if abs(h - w) == best_gap]
        if len(best) == 1:
            h, w = best[0]
            if w == _WIDTH_REQUIRED and h <= _HEIGHT_MAX:
                results.append((h, w))
    return results


_VALID_DIMS = _valid_dims()
_DIM_WEIGHTS = [h * w for h, w in _VALID_DIMS]


def _zigzag_frame(H):
    """Generate frame bitmap (Hx2 left-half) via zigzag path from top to bottom.
    Returns (cells, start_c, end_c) where start_c/end_c are the column at top/bottom."""
    cells = set()
    start_c = random.randint(0, 1)
    c = start_c
    for r in range(H):
        cells.add((r, c))
        if r == H - 1:
            break
        move = random.choice(['down', 'horiz', 'diag'])
        if move == 'down':
            pass
        elif move == 'horiz':
            cells.add((r, 1 - c))
            c = 1 - c
        else:
            c = 1 - c
    return cells, start_c, c


def _fill_holes(grid, H, W, half, frame1, frame2, split, fill_pool):
    """Fill non-frame cells by assigning colors to 4-conn hole regions.
    Large holes are split into chunks of <= _MAX_CHUNK_SIZE cells."""
    holes = components_mask(grid == -1, connectivity=4)
    prev_col = None
    for hole in holes:
        cells = sorted((int(r), int(c)) for r, c in hole)
        # Split into chunks of bounded size
        chunks = [cells[i:i + _MAX_CHUNK_SIZE] for i in range(0, len(cells), _MAX_CHUNK_SIZE)]
        for chunk in chunks:
            # Determine which frame colors are adjacent to this chunk
            fc_set = {frame1 if r < split else frame2 for r, _ in chunk}
            # Pick a fill color avoiding frame and previous neighbor
            avoid = fc_set | ({prev_col} if prev_col else set())
            opts = [f for f in fill_pool if f not in avoid]
            if not opts:
                opts = [f for f in fill_pool if f not in fc_set] or fill_pool
            col = random.choice(opts)
            for r, c in chunk:
                grid[r, c] = col
                grid[r, W - 1 - c] = col
            prev_col = col


def _build_output(H, W):
    """Build an abstract LR-symmetric output grid with zigzag frame and filled holes.
    Returns (grid, frame_colors_set) or (None, set()) on failure."""
    half = W // 2
    n_colors = random.randint(_N_COLORS_MIN, _N_COLORS_MAX)
    palette = list(range(n_colors))
    random.shuffle(palette)

    use_two_frames = (H >= _TWO_FRAME_MIN_H and n_colors >= _TWO_FRAME_MIN_COLORS
                      and random.random() < _TWO_FRAME_PROB)
    ci = 0
    frame1 = palette[ci]; ci += 1
    frame2 = frame1
    if use_two_frames:
        frame2 = palette[ci]; ci += 1
    fill_pool = palette[ci:]
    if not fill_pool:
        return None, set()

    frame_half, start_c, end_c = _zigzag_frame(H)
    split = H // 2 if frame1 != frame2 else H

    grid = np.full((H, W), -1, dtype=int)

    # Paint frame cells (mirror left half to right)
    for r, c in frame_half:
        fc = frame1 if r < split else frame2
        grid[r, c] = fc
        grid[r, W - 1 - c] = fc

    # Cap center column at top/bottom when zigzag endpoint is adjacent
    if start_c == 1 and random.random() < _CENTER_CAP_PROB:
        grid[0, half] = frame1
    if end_c == 1 and random.random() < _CENTER_CAP_PROB:
        grid[H - 1, half] = frame2

    _fill_holes(grid, H, W, half, frame1, frame2, split, fill_pool)

    if np.any(grid < 0):
        return None, set()
    frame_set = {frame1}
    if frame2 != frame1:
        frame_set.add(frame2)
    return grid, frame_set


def _make_output() -> tuple[np.ndarray | None, int, set[int]]:
    """Select dims, build abstract output, then remap to real ARC colors."""
    H, W = random.choices(_VALID_DIMS, weights=_DIM_WEIGHTS, k=1)[0]
    out, abstract_fcs = _build_output(H, W)
    if out is None:
        return None, 0, set()
    bg = random.randint(0, 9)
    abstract = sorted(np.unique(out))
    available = [c for c in range(10) if c != bg]
    random.shuffle(available)
    if len(available) < len(abstract):
        return None, 0, set()
    cmap = {int(a): available[i] for i, a in enumerate(abstract)}
    real_fcs = {cmap[a] for a in abstract_fcs}
    return np.vectorize(lambda x: cmap[int(x)])(out), bg, real_fcs


def _extract_non_frame_patches(out, bg, fcs):
    """Extract non-frame piece patches from the output grid, sorted largest-first."""
    patches = []
    for col in sorted(map(int, np.unique(out))):
        if col in fcs:
            continue
        for pts in components_mask(out == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for r, c in pts:
                patch[int(r) - r0, int(c) - c0] = col
            patches.append(patch)
    random.shuffle(patches)
    patches.sort(key=lambda p: (-(p != bg).sum(), random.random()))
    return patches


def _scatter(out: np.ndarray, bg: int, fcs: set[int]) -> np.ndarray:
    """Place pieces onto a canvas: frame as a group, non-frame scattered with padding."""
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    oh, ow = out.shape

    max_r = rows - oh - 2
    max_c = cols - ow - 2
    if max_r < 1 or max_c < 1:
        raise ValueError
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)

    # Block the entire frame bbox so fill pieces stay outside the outline
    canvas[off_r:off_r + oh, off_c:off_c + ow] = -1

    # Scatter non-frame pieces into the area OUTSIDE the blocked bbox
    for patch in _extract_non_frame_patches(out, bg, fcs):
        r, c = random_free_location(canvas, patch, transparent=bg,
                                    padding=_SCATTER_PADDING, padding_connectivity=8)
        paste(canvas, patch, r, c, transparent=bg)

    # Replace blocked bbox with frame cells (non-frame positions revert to bg)
    frame_mask = np.zeros((oh, ow), dtype=bool)
    for col in fcs:
        frame_mask |= (out == col)
    bbox_region = canvas[off_r:off_r + oh, off_c:off_c + ow]
    bbox_region[:] = bg
    bbox_region[frame_mask] = out[frame_mask]
    return canvas


def _piece_count(out: np.ndarray) -> int:
    return sum(len(components_mask(out == col, connectivity=4)) for col in np.unique(out))


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build output, validate constraints, scatter, oracle check."""
    out, bg, fcs = _make_output()
    if out is None:
        return None
    if not np.array_equal(out, out[:, ::-1]):
        return None
    if not (_N_COLORS_MIN <= len(np.unique(out)) <= _N_COLORS_MAX):
        return None
    pc = _piece_count(out)
    if not (_PIECE_COUNT_MIN <= pc <= _PIECE_COUNT_MAX):
        return None
    if any(len(pts) > _MAX_PIECE_AREA for col in np.unique(out)
           for pts in components_mask(out == col, connectivity=4)):
        return None
    try:
        inp = _scatter(out, bg, fcs)
        got = main(inp)
    except Exception:
        return None
    if np.array_equal(got, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 input")
