import numpy as np
from collections import deque


BLACK = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
RNG = np.random.default_rng()
GEN_ATTEMPTS = 300
PAIR_ATTEMPTS = 200
PLACE_ATTEMPTS = 200
COLOR_POOL = tuple(range(1, 10))
HOLE_TEMPLATES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
)


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    outside = np.zeros((h, w), dtype=bool)
    q = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    return mask | (~mask & ~outside)


def _mask_coords(mask: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(mask)]


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                cell = (int(nr), int(nc))
                if cell not in seen:
                    seen.add(cell)
                    q.append(cell)
    return len(seen) == int(mask.sum())


def _pattern_from_holes(holes: set[tuple[int, int]], color: int) -> np.ndarray:
    if not holes:
        return np.zeros((1, 1), dtype=int)
    rows = [r for r, _ in holes]
    cols = [c for _, c in holes]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    pat = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in holes:
        pat[r - r0, c - c0] = int(color)
    return pat


def _adjacency_score(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> int:
    score = 0
    for r, c in a:
        for dr, dc in DIR4:
            if (r + dr, c + dc) in b:
                score += 1
    return score


def _assemble_pair(piece_a: dict, piece_b: dict) -> np.ndarray:
    fill_a = piece_a["filled_mask"]
    fill_b = piece_b["filled_mask"]
    color_a = piece_a["color_mask"]
    color_b = piece_b["color_mask"]
    h1, w1 = fill_a.shape
    h2, w2 = fill_b.shape
    fill_a_coords = _mask_coords(fill_a)
    fill_b_coords = _mask_coords(fill_b)
    color_a_coords = _mask_coords(color_a)
    color_b_coords = _mask_coords(color_b)
    best = None

    for dr in range(-h2, h1 + 1):
        for dc in range(-w2, w1 + 1):
            top = min(0, dr)
            left = min(0, dc)
            bottom = max(h1, dr + h2)
            right = max(w1, dc + w2)
            out_h = bottom - top
            out_w = right - left
            off_a = (-top, -left)
            off_b = (dr - top, dc - left)

            set_a = {(r + off_a[0], c + off_a[1]) for r, c in fill_a_coords}
            set_b = {(r + off_b[0], c + off_b[1]) for r, c in fill_b_coords}
            if set_a & set_b:
                continue
            if len(set_a) + len(set_b) != out_h * out_w:
                continue
            if _adjacency_score(set_a, set_b) == 0:
                continue

            color_set = {(r + off_a[0], c + off_a[1]) for r, c in color_a_coords}
            color_set |= {(r + off_b[0], c + off_b[1]) for r, c in color_b_coords}
            holes = {(r, c) for r in range(out_h) for c in range(out_w)} - color_set
            if any(r in (0, out_h - 1) or c in (0, out_w - 1) for r, c in holes):
                continue

            score = (
                _adjacency_score(set_a, set_b),
                -out_h * out_w,
                -abs(out_h - out_w),
                -abs(dr),
                -abs(dc),
            )
            candidate = (score, _pattern_from_holes(holes, piece_a["color"]))
            if best is None or candidate[0] > best[0]:
                best = candidate

    if best is None:
        raise ValueError("failed to assemble a same-color pair into a rectangle")
    return best[1]


def _extract_pieces(grid: np.ndarray) -> dict[int, list[dict]]:
    pieces_by_color: dict[int, list[dict]] = {}
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != BLACK):
        comps = _mask_components(grid == int(color))
        items = []
        for comp in comps:
            r0, c0, r1, c1 = _bbox_from_coords(comp)
            color_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for r, c in comp:
                color_mask[int(r) - r0, int(c) - c0] = True
            filled_mask = _fill_holes(color_mask)
            items.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "color_mask": color_mask,
                    "filled_mask": filled_mask,
                }
            )
        pieces_by_color[int(color)] = items
    return pieces_by_color


def _stack_patterns(items: list[tuple[int, int, np.ndarray]]) -> np.ndarray:
    ordered = sorted(items, key=lambda x: (int(x[0]), int(x[1])))
    patterns = [pat for _row, _col, pat in ordered]
    out_h = sum(int(p.shape[0]) for p in patterns) + max(0, len(patterns) - 1)
    out_w = max(int(p.shape[1]) for p in patterns)
    out = np.zeros((out_h, out_w), dtype=int)
    r = 0
    for i, pat in enumerate(patterns):
        h, w = pat.shape
        out[r : r + h, :w] = pat
        r += h
        if i != len(patterns) - 1:
            r += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-color jigsaw fragments, rebuild each hidden rectangle, and output its internal black-space pattern.

    Input structure:
    - Black background.
    - Several non-touching monochrome connected components are scattered across the grid.
    - Each non-black color appears on exactly two fragments that belong together.
    - A fragment is a near-rectangular half-piece with one tab or socket edge, and it may contain a few enclosed black hole cells.

    Rule:
    - Group fragments by color.
    - For each fragment, fill enclosed holes to recover its outer silhouette while keeping open sockets untouched.
    - Translate the two same-color silhouettes, without rotation or reflection, until their union becomes one full rectangle with complementary tab-socket matching.
    - Inside that rebuilt rectangle, collect the black cells from the original pieces, crop their bounding box, recolor that pattern with the pair color, and stack all extracted patterns from top to bottom by the input row of each pair's upper fragment, with one black separator row between patterns.

    Key insight:
    - The visible fragments are only an intermediate assembly puzzle: the real output is the negative-space hole pattern revealed after reassembling each same-color rectangle.
    """
    g = np.asarray(grid, dtype=int)
    pieces_by_color = _extract_pieces(g)
    items = []

    for color, pieces in pieces_by_color.items():
        if len(pieces) != 2:
            raise ValueError(f"color {color} must form exactly two fragments")
        upper_bbox = min((piece["bbox"] for piece in pieces), key=lambda b: (b[0], b[1]))
        pattern = _assemble_pair(pieces[0], pieces[1])
        items.append((int(upper_bbox[0]), int(upper_bbox[1]), pattern))

    return _stack_patterns(items)


def _build_pair(color: int, max_dim: int) -> dict | None:
    for _ in range(PAIR_ATTEMPTS):
        rect_h = int(RNG.integers(5, max_dim + 1))
        rect_w = int(RNG.integers(5, max_dim + 1))
        orientation = "vertical" if float(RNG.random()) < 0.5 else "horizontal"
        sil_a = np.zeros((rect_h, rect_w), dtype=bool)
        sil_b = np.zeros((rect_h, rect_w), dtype=bool)

        if orientation == "vertical":
            split = int(RNG.integers(2, rect_w - 1))
            tab_len = int(RNG.integers(1, 3))
            start = int(RNG.integers(1, rect_h - tab_len))
            tab_rows = slice(start, start + tab_len)
            if float(RNG.random()) < 0.5:
                sil_a[:, :split] = True
                sil_b[:, split:] = True
                sil_a[tab_rows, split] = True
                sil_b[tab_rows, split] = False
            else:
                sil_a[:, :split] = True
                sil_b[:, split:] = True
                sil_a[tab_rows, split - 1] = False
                sil_b[tab_rows, split - 1] = True
        else:
            split = int(RNG.integers(2, rect_h - 1))
            tab_len = int(RNG.integers(1, 3))
            start = int(RNG.integers(1, rect_w - tab_len))
            tab_cols = slice(start, start + tab_len)
            if float(RNG.random()) < 0.5:
                sil_a[:split, :] = True
                sil_b[split:, :] = True
                sil_a[split, tab_cols] = True
                sil_b[split, tab_cols] = False
            else:
                sil_a[:split, :] = True
                sil_b[split:, :] = True
                sil_a[split - 1, tab_cols] = False
                sil_b[split - 1, tab_cols] = True

        safe = np.zeros((rect_h, rect_w), dtype=bool)
        for r in range(1, rect_h - 1):
            for c in range(1, rect_w - 1):
                if sil_a[r, c] and all(sil_a[r + dr, c + dc] for dr, dc in DIR4):
                    safe[r, c] = True
                if sil_b[r, c] and all(sil_b[r + dr, c + dc] for dr, dc in DIR4):
                    safe[r, c] = True

        candidates = []
        for template in HOLE_TEMPLATES:
            max_r = max(r for r, _ in template)
            max_c = max(c for _, c in template)
            for r0 in range(rect_h - max_r):
                for c0 in range(rect_w - max_c):
                    cells = {(r0 + dr, c0 + dc) for dr, dc in template}
                    if not all(safe[r, c] for r, c in cells):
                        continue

                    holes_a = np.zeros_like(sil_a)
                    holes_b = np.zeros_like(sil_b)
                    for r, c in cells:
                        if sil_a[r, c]:
                            holes_a[r, c] = True
                        elif sil_b[r, c]:
                            holes_b[r, c] = True

                    color_a = sil_a & ~holes_a
                    color_b = sil_b & ~holes_b
                    if not _is_connected(color_a) or not _is_connected(color_b):
                        continue
                    candidates.append((color_a, color_b, cells))

        if not candidates:
            continue

        color_a, color_b, holes = candidates[int(RNG.integers(len(candidates)))]
        r0a, c0a, r1a, c1a = _bbox_from_coords(np.argwhere(sil_a))
        r0b, c0b, r1b, c1b = _bbox_from_coords(np.argwhere(sil_b))
        patch_a = np.zeros((r1a - r0a + 1, c1a - c0a + 1), dtype=int)
        patch_b = np.zeros((r1b - r0b + 1, c1b - c0b + 1), dtype=int)
        patch_a[color_a[r0a : r1a + 1, c0a : c1a + 1]] = int(color)
        patch_b[color_b[r0b : r1b + 1, c0b : c1b + 1]] = int(color)
        sil_crop_a = sil_a[r0a : r1a + 1, c0a : c1a + 1].copy()
        sil_crop_b = sil_b[r0b : r1b + 1, c0b : c1b + 1].copy()

        expected = _pattern_from_holes(holes, color)
        return {
            "color": int(color),
            "pieces": [
                {"patch": patch_a, "silhouette": sil_crop_a},
                {"patch": patch_b, "silhouette": sil_crop_b},
            ],
            "pattern": expected,
        }
    return None


def _mark_forbidden(forbidden: np.ndarray, sil: np.ndarray, top: int, left: int) -> None:
    h, w = sil.shape
    for r, c in np.argwhere(sil):
        rr = int(top + r)
        cc = int(left + c)
        for dr, dc in DIR8 + [(0, 0)]:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < forbidden.shape[0] and 0 <= nc < forbidden.shape[1]:
                forbidden[nr, nc] = True


def _can_place(forbidden: np.ndarray, sil: np.ndarray, top: int, left: int) -> bool:
    h, w = sil.shape
    if top < 0 or left < 0 or top + h > forbidden.shape[0] or left + w > forbidden.shape[1]:
        return False
    for r, c in np.argwhere(sil):
        if forbidden[int(top + r), int(left + c)]:
            return False
    return True


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    sub = grid[top : top + h, left : left + w]
    sub[patch != BLACK] = patch[patch != BLACK]


def _gen_attempt() -> np.ndarray | None:
    num_pairs = int(RNG.integers(2, 5))
    max_dim = 6 if num_pairs >= 4 else 7
    colors = [int(c) for c in RNG.choice(COLOR_POOL, size=num_pairs, replace=False)]
    pairs = []
    for color in colors:
        pair = _build_pair(color, max_dim)
        if pair is None:
            return None
        pairs.append(pair)

    max_piece_width = max(max(piece["patch"].shape[1] for piece in pair["pieces"]) for pair in pairs)
    min_width = max(18, max_piece_width * 2 + 4)
    if min_width > 30:
        return None
    grid_w = int(RNG.integers(min_width, 31))

    band_heights = []
    for pair in pairs:
        band_heights.append(max(piece["patch"].shape[0] for piece in pair["pieces"]) + int(RNG.integers(2, 5)))

    top_margin = int(RNG.integers(0, 3))
    bottom_margin = int(RNG.integers(0, 3))
    gaps = [int(RNG.integers(1, 3)) for _ in range(max(0, num_pairs - 1))]
    grid_h = top_margin + bottom_margin + sum(band_heights) + sum(gaps)
    if grid_h > 30:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    forbidden = np.zeros((grid_h, grid_w), dtype=bool)
    expected_items = []
    cursor = top_margin

    for idx, pair in enumerate(pairs):
        band_top = cursor
        band_height = band_heights[idx]
        band_bottom = band_top + band_height
        pieces = list(pair["pieces"])
        RNG.shuffle(pieces)
        tops = []

        for piece in pieces:
            ph, pw = piece["patch"].shape
            placed = False
            row_lo = band_top
            row_hi = band_bottom - ph
            for _ in range(PLACE_ATTEMPTS):
                top = int(RNG.integers(row_lo, row_hi + 1))
                left = int(RNG.integers(0, grid_w - pw + 1))
                if not _can_place(forbidden, piece["silhouette"], top, left):
                    continue
                _paste_patch(grid, piece["patch"], top, left)
                _mark_forbidden(forbidden, piece["silhouette"], top, left)
                tops.append((top, left))
                placed = True
                break
            if not placed:
                return None

        upper = min(tops, key=lambda t: (t[0], t[1]))
        expected_items.append((int(upper[0]), int(upper[1]), pair["pattern"]))
        cursor = band_bottom
        if idx != num_pairs - 1:
            cursor += gaps[idx]

    expected = _stack_patterns(expected_items)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for seed581286445/7")
