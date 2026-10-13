import random

import numpy as np


def _mode(values) -> int:
    vals, cnts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _background_color(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    return _mode(corners)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected at least one foreground cell"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bands(lo: int, hi: int, cuts: list[int]) -> list[tuple[int, int]]:
    starts = [int(lo)] + [int(cut) + 1 for cut in cuts]
    ends = [int(cut) - 1 for cut in cuts] + [int(hi)]
    return [(a, b) for a, b in zip(starts, ends) if a <= b]


def _separator_color(grid: np.ndarray, bg: int, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    box = grid[r0 : r1 + 1, c0 : c1 + 1]
    best = None
    best_score = -1
    for color in np.unique(box):
        color = int(color)
        if color == bg:
            continue
        row_hits = int(np.sum(np.all(box == color, axis=1)))
        col_hits = int(np.sum(np.all(box == color, axis=0)))
        score = row_hits + col_hits
        if score > best_score:
            best_score = score
            best = color
    assert best is not None and best_score > 0, "expected a separator color"
    return int(best)


def _negate_signature(sig: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(0 if v == 0 else -int(v) for v in side) for side in sig)


def _edge_fill_color(patch: np.ndarray, bg: int, sep: int) -> tuple[int, np.ndarray]:
    content = [int(v) for v in np.unique(patch) if int(v) not in {bg, sep}]
    assert content, "expected at least one non-background color in a cell"

    edge_mask = np.zeros_like(patch, dtype=bool)
    edge_mask[1, 1:-1] = True
    edge_mask[-2, 1:-1] = True
    edge_mask[1:-1, 1] = True
    edge_mask[1:-1, -2] = True

    fill_candidates = [color for color in content if np.any((patch == color) & edge_mask)]
    assert len(fill_candidates) == 1, "expected a unique visible fill color"
    fill = int(fill_candidates[0])

    cover_mask = np.zeros_like(patch, dtype=bool)
    for color in content:
        if color != fill:
            cover_mask |= patch == color
    return fill, cover_mask


def _edge_signature(patch: np.ndarray, bg: int, sep: int) -> tuple[tuple[int, ...], ...]:
    occ = (patch != bg) & (patch != sep)
    size = int(patch.shape[0])
    assert patch.shape[0] == patch.shape[1] and size >= 5, "expected square cells with a 1-cell jigsaw margin"
    positions = list(range(2, size - 2))
    assert positions, "expected at least one inner edge position"

    top = []
    bottom = []
    left = []
    right = []
    for c in positions:
        if occ[0, c]:
            top.append(1)
        elif not occ[1, c]:
            top.append(-1)
        else:
            top.append(0)

        if occ[-1, c]:
            bottom.append(1)
        elif not occ[-2, c]:
            bottom.append(-1)
        else:
            bottom.append(0)

    for r in positions:
        if occ[r, 0]:
            left.append(1)
        elif not occ[r, 1]:
            left.append(-1)
        else:
            left.append(0)

        if occ[r, -1]:
            right.append(1)
        elif not occ[r, -2]:
            right.append(-1)
        else:
            right.append(0)

    return tuple(tuple(side) for side in (tuple(top), tuple(right), tuple(bottom), tuple(left)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore maroon-covered jigsaw pieces by matching complementary edge-tooth patterns.

    Input structure:
    - The background is black.
    - Same-size colored jigsaw-like squares are arranged in a row or small grid, with full grey separator lines between neighboring cells.
    - Each piece has one-pixel protrusions or indentations on its four sides, and every fill color appears exactly twice.
    - Some pieces have an interior maroon rectangle covering part of their fill while leaving the edge shape visible.

    Rule:
    - Segment the puzzle cells using the full grey separator rows and columns.
    - For each cell, read its four side-tooth patterns from the visible edge geometry.
    - Any maroon-covered cell must be paired with the uncovered cell whose same-direction side patterns are the exact protrusion/indentation complement.
    - Replace the maroon pixels with that partner cell's fill color and leave all other pixels unchanged.

    Key insight:
    - The hidden color is recovered indirectly: complementary jigsaw edge signatures identify the partner piece, and the partner's visible fill color resolves the occlusion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    bbox = _bbox(g != bg)
    sep = _separator_color(g, bg, bbox)
    r0, c0, r1, c1 = bbox
    box = g[r0 : r1 + 1, c0 : c1 + 1]

    sep_rows = [r0 + int(i) for i in np.flatnonzero(np.all(box == sep, axis=1))]
    sep_cols = [c0 + int(i) for i in np.flatnonzero(np.all(box == sep, axis=0))]

    row_bands = _bands(r0, r1, sep_rows)
    col_bands = _bands(c0, c1, sep_cols)
    assert row_bands and col_bands, "expected at least one cell band"

    heights = {b - a + 1 for a, b in row_bands}
    widths = {b - a + 1 for a, b in col_bands}
    assert len(heights) == 1 and len(widths) == 1, "expected equal-sized cell bands"
    cell_size = int(next(iter(heights)))
    assert cell_size == int(next(iter(widths))), "expected square cells"

    cells = []
    for rb in row_bands:
        for cb in col_bands:
            patch = g[rb[0] : rb[1] + 1, cb[0] : cb[1] + 1]
            fill_color, cover_mask = _edge_fill_color(patch, bg, sep)
            cells.append(
                {
                    "bbox": (rb[0], cb[0], rb[1], cb[1]),
                    "fill": int(fill_color),
                    "cover": cover_mask,
                    "covered": bool(np.any(cover_mask)),
                    "signature": _edge_signature(patch, bg, sep),
                }
            )

    visible_by_signature = {}
    for idx, cell in enumerate(cells):
        if cell["covered"]:
            continue
        sig = cell["signature"]
        assert sig not in visible_by_signature, "visible jigsaw signature must be unique"
        visible_by_signature[sig] = idx

    out = g.copy()
    for cell in cells:
        if not cell["covered"]:
            continue
        partner_sig = _negate_signature(cell["signature"])
        assert partner_sig in visible_by_signature, "every covered cell needs a visible complementary partner"
        partner = cells[visible_by_signature[partner_sig]]
        rr0, cc0, rr1, cc1 = cell["bbox"]
        patch = out[rr0 : rr1 + 1, cc0 : cc1 + 1]
        patch[cell["cover"]] = int(partner["fill"])

    return out


BG = 0
SEP = 5
OCC = 9
FILL_POOL = (1, 2, 3, 4, 6, 7, 8)
BASE_SIZE_CHOICES = (4, 5, 6, 7)
LAYOUT_CANDIDATES = ((1, 2), (2, 1), (1, 4), (4, 1), (2, 2), (2, 3), (3, 2))
MAX_GRID_SIZE = 30
MAX_ATTEMPTS = 400


def _side_patterns(length: int) -> list[tuple[int, ...]]:
    patterns = []
    for bits in range(1 << length):
        side = tuple(1 if (bits >> i) & 1 else -1 for i in range(length))
        if len(set(side)) > 1:
            patterns.append(side)
    return patterns or [tuple([1] * length), tuple([-1] * length)]


def _choose_layout(cell_size: int) -> tuple[int, int]:
    by_kind = {"line": [], "grid": []}
    for rows, cols in LAYOUT_CANDIDATES:
        h = rows * cell_size + (rows - 1)
        w = cols * cell_size + (cols - 1)
        if h <= MAX_GRID_SIZE and w <= MAX_GRID_SIZE:
            kind = "line" if rows == 1 or cols == 1 else "grid"
            by_kind[kind].append((rows, cols))

    preferred = random.choice(("line", "grid"))
    options = by_kind[preferred] or by_kind["grid" if preferred == "line" else "line"]
    assert options, "expected at least one valid layout"
    return random.choice(options)


def _piece_patch(base_size: int, color: int, signature: tuple[tuple[int, ...], ...]) -> np.ndarray:
    size = base_size + 2
    patch = np.full((size, size), BG, dtype=int)
    patch[1 : base_size + 1, 1 : base_size + 1] = int(color)

    top, right, bottom, left = signature
    for offset, state in enumerate(top, start=2):
        if state > 0:
            patch[0, offset] = int(color)
        elif state < 0:
            patch[1, offset] = BG
    for offset, state in enumerate(bottom, start=2):
        if state > 0:
            patch[-1, offset] = int(color)
        elif state < 0:
            patch[-2, offset] = BG
    for offset, state in enumerate(left, start=2):
        if state > 0:
            patch[offset, 0] = int(color)
        elif state < 0:
            patch[offset, 1] = BG
    for offset, state in enumerate(right, start=2):
        if state > 0:
            patch[offset, -1] = int(color)
        elif state < 0:
            patch[offset, -2] = BG
    return patch


def _apply_occluder(patch: np.ndarray, base_size: int) -> np.ndarray:
    out = patch.copy()
    lo = 2
    hi = base_size - 1
    r0 = random.randint(lo, hi)
    r1 = random.randint(r0, hi)
    c0 = random.randint(lo, hi)
    c1 = random.randint(c0, hi)
    out[r0 : r1 + 1, c0 : c1 + 1] = OCC
    return out


def _assemble_grid(rows: int, cols: int, patches: list[np.ndarray]) -> np.ndarray:
    cell_size = int(patches[0].shape[0])
    height = rows * cell_size + (rows - 1)
    width = cols * cell_size + (cols - 1)
    grid = np.full((height, width), BG, dtype=int)

    for r in range(1, rows):
        sep_row = r * (cell_size + 1) - 1
        grid[sep_row, :] = SEP
    for c in range(1, cols):
        sep_col = c * (cell_size + 1) - 1
        grid[:, sep_col] = SEP

    it = iter(patches)
    for r in range(rows):
        for c in range(cols):
            sr = r * (cell_size + 1)
            sc = c * (cell_size + 1)
            grid[sr : sr + cell_size, sc : sc + cell_size] = next(it)
    return grid


def _canonical_signature(sig: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    neg = _negate_signature(sig)
    return min(sig, neg)


def _generate_case() -> np.ndarray | None:
    base_size = random.choice(BASE_SIZE_CHOICES)
    cell_size = base_size + 2
    rows, cols = _choose_layout(cell_size)
    piece_count = rows * cols
    pair_count = piece_count // 2

    colors = random.sample(FILL_POOL, pair_count)
    side_options = _side_patterns(base_size - 2)
    signatures = []
    used = set()
    for _ in range(pair_count):
        for _attempt in range(100):
            sig = tuple(random.choice(side_options) for _ in range(4))
            canon = _canonical_signature(sig)
            if canon not in used:
                used.add(canon)
                signatures.append(sig)
                break
        else:
            return None

    occluded_pairs = set(random.sample(range(pair_count), random.randint(1, pair_count)))
    pieces = []
    for pair_idx, (color, sig) in enumerate(zip(colors, signatures)):
        hidden_member = random.randrange(2) if pair_idx in occluded_pairs else None
        variants = (sig, _negate_signature(sig))
        for member_idx, variant in enumerate(variants):
            clean = _piece_patch(base_size, color, variant)
            shown = _apply_occluder(clean, base_size) if member_idx == hidden_member else clean.copy()
            pieces.append({"inp": shown, "out": clean})

    random.shuffle(pieces)
    inp = _assemble_grid(rows, cols, [piece["inp"] for piece in pieces])
    out = _assemble_grid(rows, cols, [piece["out"] for piece in pieces])

    try:
        if np.array_equal(main(inp), out):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        candidate = _generate_case()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid jigsaw-occlusion puzzle")
