from __future__ import annotations

from collections import deque
import random

import numpy as np


# --------------- shared geometry helpers ---------------

ALL_COLORS = tuple(range(10))
GRID_SIZE_RANGE = (24, 30)
TEMPLATE_SIZE_RANGE = (9, 15)
OUTER_MARGIN = 6
HOLE_DIM_MAX = 4
HOLE_COUNT_CHOICES = (2, 3, 4)
PLACEMENT_PAD = 1
GEN_RETRIES = 400


def _border_bg(g: np.ndarray) -> int:
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    values, counts = np.unique(border, return_counts=True)
    return int(values[counts.argmax()])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _shape_from_bbox(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return r1 - r0 + 1, c1 - c0 + 1


def _is_filled_rectangle(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    return len(points) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _find_template(
    g: np.ndarray,
) -> tuple[int, int, np.ndarray, tuple[int, int, int, int]]:
    bg = _border_bg(g)
    best: tuple[int, int, list[tuple[int, int]]] | None = None
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        for pts in _components(g == color):
            if best is None or len(pts) > best[0]:
                best = (len(pts), color, pts)

    assert best is not None, "expected a non-background template component"
    _area, template_color, template_pts = best
    mask = np.zeros(g.shape, dtype=bool)
    for r, c in template_pts:
        mask[r, c] = True
    return bg, template_color, mask, _bbox(template_pts)


def _find_holes(
    g: np.ndarray,
    bg: int,
    template_color: int,
    template_bbox: tuple[int, int, int, int],
) -> list[dict[str, int]]:
    r0, c0, r1, c1 = template_bbox
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    allowed = {int(bg), int(template_color)}
    assert set(map(int, np.unique(sub))) <= allowed, "template bbox must contain only template and holes"

    holes: list[dict[str, int]] = []
    for pts in _components(sub == bg):
        if any(rr == 0 or cc == 0 or rr == sub.shape[0] - 1 or cc == sub.shape[1] - 1 for rr, cc in pts):
            continue
        global_pts = [(r0 + rr, c0 + cc) for rr, cc in pts]
        assert _is_filled_rectangle(global_pts), "each hole must be a rectangle"
        hr0, hc0, hr1, hc1 = _bbox(global_pts)
        holes.append(
            {
                "r0": int(hr0),
                "c0": int(hc0),
                "r1": int(hr1),
                "c1": int(hc1),
                "h": int(hr1 - hr0 + 1),
                "w": int(hc1 - hc0 + 1),
            }
        )
    assert holes, "expected at least one hole inside the template"
    return holes


def _find_pieces(
    g: np.ndarray,
    bg: int,
    template_mask: np.ndarray,
    template_bbox: tuple[int, int, int, int],
) -> list[dict[str, int]]:
    tr0, tc0, tr1, tc1 = template_bbox
    comps = _components((g != bg) & (~template_mask))
    pieces: list[dict[str, int]] = []
    for pts in comps:
        colors = {int(g[r, c]) for r, c in pts}
        assert len(colors) == 1, "each outside fragment must be monochrome"
        assert _is_filled_rectangle(pts), "each outside fragment must be a rectangle"
        pr0, pc0, pr1, pc1 = _bbox(pts)
        assert pr1 < tr0 or pr0 > tr1 or pc1 < tc0 or pc0 > tc1, "pieces must stay outside the template bbox"
        pieces.append(
            {
                "color": int(next(iter(colors))),
                "r0": int(pr0),
                "c0": int(pc0),
                "r1": int(pr1),
                "c1": int(pc1),
                "h": int(pr1 - pr0 + 1),
                "w": int(pc1 - pc0 + 1),
            }
        )
    assert pieces, "expected at least one outside fragment"
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    """Slide scattered fragments into matching holes of a gray template and recolor them to the template color.

    Input structure:
    - The background is the border-dominant color.
    - A large rectangular template is the largest monochrome component away from the border and uses one uniform color.
    - Inside that template bbox are one or more rectangular background holes.
    - Outside the template lie several smaller 4-connected monochrome rectangular fragments, one per hole.

    Rule:
    - Detect the largest template component and enumerate its internal rectangular holes.
    - Match each outside fragment to the unique hole with the same height and width.
    - Verify the fragment is aligned with its hole on exactly one axis: either the row span matches for a horizontal slide or the column span matches for a vertical slide.
    - Clear the fragment's original location to background, then fill the matched hole with the template color.

    Key insight:
    - Size identifies which hole a fragment belongs to, but row-versus-column alignment reveals the slide direction; after entering the template every fragment loses its original color and becomes part of the gray template.
    """
    g = np.asarray(grid, dtype=int)
    bg, template_color, template_mask, template_bbox = _find_template(g)
    holes = _find_holes(g, bg, template_color, template_bbox)
    pieces = _find_pieces(g, bg, template_mask, template_bbox)

    holes_by_size: dict[tuple[int, int], dict[str, int]] = {}
    for hole in holes:
        size = (hole["h"], hole["w"])
        assert size not in holes_by_size, "hole sizes must be unique"
        holes_by_size[size] = hole

    assert len(pieces) == len(holes_by_size), "piece and hole counts must match"

    out = g.copy()
    matched_sizes: set[tuple[int, int]] = set()
    for piece in pieces:
        size = (piece["h"], piece["w"])
        assert size in holes_by_size, "every piece must match a hole by size"
        hole = holes_by_size[size]
        row_aligned = piece["r0"] == hole["r0"] and piece["r1"] == hole["r1"]
        col_aligned = piece["c0"] == hole["c0"] and piece["c1"] == hole["c1"]
        assert row_aligned ^ col_aligned, "matching must be horizontal or vertical, but not both"

        out[piece["r0"] : piece["r1"] + 1, piece["c0"] : piece["c1"] + 1] = bg
        out[hole["r0"] : hole["r1"] + 1, hole["c0"] : hole["c1"] + 1] = template_color
        matched_sizes.add(size)

    assert matched_sizes == set(holes_by_size), "every hole must receive one piece"
    return out


# --------------- generator ---------------

def _rect_with_padding_clear(
    blocked: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = PLACEMENT_PAD
) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(blocked.shape[0], r0 + h + pad)
    cc1 = min(blocked.shape[1], c0 + w + pad)
    return not bool(blocked[rr0:rr1, cc0:cc1].any())


def _sample_colors(n_pieces: int) -> tuple[int, int, list[int]]:
    colors = list(ALL_COLORS)
    random.shuffle(colors)
    bg = int(colors[0])
    template_color = int(colors[1])
    piece_colors = [int(c) for c in colors[2 : 2 + n_pieces]]
    return bg, template_color, piece_colors


def _sample_template() -> tuple[int, int, int, int, int, int]:
    rows = random.randint(*GRID_SIZE_RANGE)
    cols = random.randint(*GRID_SIZE_RANGE)
    max_h = min(TEMPLATE_SIZE_RANGE[1], rows - 2 * OUTER_MARGIN)
    max_w = min(TEMPLATE_SIZE_RANGE[1], cols - 2 * OUTER_MARGIN)
    if max_h < TEMPLATE_SIZE_RANGE[0] or max_w < TEMPLATE_SIZE_RANGE[0]:
        raise AssertionError("grid too small for template margins")

    h = random.randint(TEMPLATE_SIZE_RANGE[0], max_h)
    w = random.randint(TEMPLATE_SIZE_RANGE[0], max_w)
    r0 = random.randint(OUTER_MARGIN, rows - h - OUTER_MARGIN)
    c0 = random.randint(OUTER_MARGIN, cols - w - OUTER_MARGIN)
    return rows, cols, r0, c0, h, w


def _sample_hole_sizes(inner_h: int, inner_w: int, n_holes: int) -> list[tuple[int, int]]:
    sizes = [
        (h, w)
        for h in range(1, min(HOLE_DIM_MAX, inner_h) + 1)
        for w in range(1, min(HOLE_DIM_MAX, inner_w) + 1)
        if h * w >= 2
    ]
    assert len(sizes) >= n_holes
    weights = [1.0 / (h * w) for h, w in sizes]
    chosen: list[tuple[int, int]] = []
    pool = sizes.copy()
    pool_weights = weights.copy()
    for _ in range(n_holes):
        size = random.choices(pool, weights=pool_weights, k=1)[0]
        idx = pool.index(size)
        chosen.append(size)
        pool.pop(idx)
        pool_weights.pop(idx)
    return chosen


def _place_holes(
    inner_h: int, inner_w: int, sizes: list[tuple[int, int]]
) -> list[tuple[int, int, int, int]] | None:
    holes: list[tuple[int, int, int, int]] = []
    blocked = np.zeros((inner_h, inner_w), dtype=bool)

    for h, w in sorted(sizes, key=lambda item: item[0] * item[1], reverse=True):
        candidates: list[tuple[int, int]] = []
        for r0 in range(inner_h - h + 1):
            for c0 in range(inner_w - w + 1):
                if _rect_with_padding_clear(blocked, r0, c0, h, w):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        blocked[r0 : r0 + h, c0 : c0 + w] = True
        holes.append((int(r0), int(c0), int(h), int(w)))
    return holes


def _balanced_orientations(n_holes: int) -> list[str]:
    first = random.choice(["row", "col"])
    second = "col" if first == "row" else "row"
    return [first if i % 2 == 0 else second for i in range(n_holes)]


def _piece_positions_for_hole(
    rows: int,
    cols: int,
    template_bbox: tuple[int, int, int, int],
    hole_box: tuple[int, int, int, int],
    orientation: str,
) -> list[tuple[int, int]]:
    tr0, tc0, tr1, tc1 = template_bbox
    hr0, hc0, hr1, hc1 = hole_box
    h = hr1 - hr0 + 1
    w = hc1 - hc0 + 1
    positions: list[tuple[int, int]] = []

    if orientation == "row":
        left_positions = [(hr0, c0) for c0 in range(0, tc0 - w)]
        right_positions = [(hr0, c0) for c0 in range(tc1 + 2, cols - w + 1)]
        side_groups = [left_positions, right_positions]
    else:
        top_positions = [(r0, hc0) for r0 in range(0, tr0 - h)]
        bottom_positions = [(r0, hc0) for r0 in range(tr1 + 2, rows - h + 1)]
        side_groups = [top_positions, bottom_positions]

    random.shuffle(side_groups)
    for group in side_groups:
        shuffled = group.copy()
        random.shuffle(shuffled)
        positions.extend(shuffled)
    return positions


def _place_pieces(
    rows: int,
    cols: int,
    template_bbox: tuple[int, int, int, int],
    hole_boxes: list[tuple[int, int, int, int]],
    orientations: list[str],
) -> list[tuple[int, int, int, int]] | None:
    blocked = np.zeros((rows, cols), dtype=bool)
    tr0, tc0, tr1, tc1 = template_bbox
    blocked[tr0 : tr1 + 1, tc0 : tc1 + 1] = True

    placed: list[tuple[int, int, int, int]] = []
    order = list(range(len(hole_boxes)))
    random.shuffle(order)
    pending: dict[int, tuple[int, int, int, int]] = {}

    for idx in order:
        hole_box = hole_boxes[idx]
        hr0, hc0, hr1, hc1 = hole_box
        h = hr1 - hr0 + 1
        w = hc1 - hc0 + 1
        candidates = _piece_positions_for_hole(rows, cols, template_bbox, hole_box, orientations[idx])
        chosen: tuple[int, int] | None = None
        for r0, c0 in candidates:
            if _rect_with_padding_clear(blocked, r0, c0, h, w):
                chosen = (r0, c0)
                break
        if chosen is None:
            return None
        r0, c0 = chosen
        blocked[r0 : r0 + h, c0 : c0 + w] = True
        pending[idx] = (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))

    for idx in range(len(hole_boxes)):
        placed.append(pending[idx])
    return placed


def _expected_output(
    grid: np.ndarray,
    template_color: int,
    holes: list[tuple[int, int, int, int]],
    pieces: list[tuple[int, int, int, int]],
) -> np.ndarray:
    out = grid.copy()
    bg = _border_bg(grid)
    for pr0, pc0, pr1, pc1 in pieces:
        out[pr0 : pr1 + 1, pc0 : pc1 + 1] = bg
    for hr0, hc0, hr1, hc1 in holes:
        out[hr0 : hr1 + 1, hc0 : hc1 + 1] = int(template_color)
    return out


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    rows, cols, tr0, tc0, th, tw = _sample_template()
    inner_h = th - 2
    inner_w = tw - 2
    if min(inner_h, inner_w) < 3:
        return None

    max_holes = 4 if min(th, tw) >= 12 else 3
    n_holes = random.choice([n for n in HOLE_COUNT_CHOICES if n <= max_holes])
    sizes = _sample_hole_sizes(inner_h, inner_w, n_holes)
    holes_local = _place_holes(inner_h, inner_w, sizes)
    if holes_local is None:
        return None

    hole_boxes = [
        (tr0 + 1 + r0, tc0 + 1 + c0, tr0 + r0 + h, tc0 + c0 + w)
        for r0, c0, h, w in holes_local
    ]
    orientations = _balanced_orientations(n_holes)
    piece_boxes = _place_pieces(rows, cols, (tr0, tc0, tr0 + th - 1, tc0 + tw - 1), hole_boxes, orientations)
    if piece_boxes is None:
        return None

    bg, template_color, piece_colors = _sample_colors(n_holes)
    grid = np.full((rows, cols), bg, dtype=int)
    grid[tr0 : tr0 + th, tc0 : tc0 + tw] = template_color

    for hr0, hc0, hr1, hc1 in hole_boxes:
        grid[hr0 : hr1 + 1, hc0 : hc1 + 1] = bg
    for color, (pr0, pc0, pr1, pc1) in zip(piece_colors, piece_boxes, strict=False):
        grid[pr0 : pr1 + 1, pc0 : pc1 + 1] = int(color)

    expected = _expected_output(grid, template_color, hole_boxes, piece_boxes)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        case = _generate_case()
        if case is None:
            continue
        inp, expected = case
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid ARC sample")
