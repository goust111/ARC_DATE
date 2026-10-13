import random
from collections import Counter, deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
CORNERS = ("TL", "TR", "BL", "BR")
MARKER_REL = {
    "TL": (1, 1),
    "TR": (1, -2),
    "BL": (-2, 1),
    "BR": (-2, -2),
}
CORNER_DEST = {
    "TL": lambda h, w: (0, 0),
    "TR": lambda h, w: (0, w - 1),
    "BL": lambda h, w: (h - 1, 0),
    "BR": lambda h, w: (h - 1, w - 1),
}

OUTPUT_AREAS = (64, 72, 80, 81, 90, 96, 99, 100, 108)
PARTITION_ATTEMPTS = 400
GENERATION_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 500


def _closest_factor_pair(area: int) -> tuple[int, int]:
    best = None
    for h in range(1, int(area**0.5) + 1):
        if area % h != 0:
            continue
        w = area // h
        cand = (abs(w - h), h, w)
        if best is None or cand < best:
            best = cand
    assert best is not None
    return int(best[1]), int(best[2])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _partition_labels(h: int, w: int, row_step: int, col_step: int, x_top: int, x_bottom: int, y_left: int, y_right: int) -> np.ndarray:
    labels = np.empty((h, w), dtype="<U2")
    for r in range(h):
        x_cut = x_top if r < row_step else x_bottom
        for c in range(w):
            y_cut = y_left if c < col_step else y_right
            top = r < y_cut
            left = c < x_cut
            labels[r, c] = ("T" if top else "B") + ("L" if left else "R")
    return labels


def _valid_partition(labels: np.ndarray) -> bool:
    h, w = labels.shape
    corner_blocks = {
        "TL": ((0, 0), (0, 1), (1, 0), (1, 1)),
        "TR": ((0, w - 2), (0, w - 1), (1, w - 2), (1, w - 1)),
        "BL": ((h - 2, 0), (h - 2, 1), (h - 1, 0), (h - 1, 1)),
        "BR": ((h - 2, w - 2), (h - 2, w - 1), (h - 1, w - 2), (h - 1, w - 1)),
    }
    for key, cells in corner_blocks.items():
        if any(labels[r, c] != key for r, c in cells):
            return False

    for key in CORNERS:
        comps = _components(labels == key)
        if len(comps) != 1:
            return False
        pts = comps[0]
        r0, c0, r1, c1 = _bbox(pts)
        bh = r1 - r0 + 1
        bw = c1 - c0 + 1
        if bh < 4 or bw < 4:
            return False
        if len(pts) == bh * bw:
            return False
    return True


def _random_partition(h: int, w: int) -> np.ndarray | None:
    for _ in range(PARTITION_ATTEMPTS):
        row_step = random.randint(2, h - 2)
        col_step = random.randint(2, w - 2)
        x_top = random.randint(2, w - 2)
        x_bottom = random.randint(2, w - 2)
        y_left = random.randint(2, h - 2)
        y_right = random.randint(2, h - 2)
        if x_top == x_bottom or y_left == y_right:
            continue
        labels = _partition_labels(h, w, row_step, col_step, x_top, x_bottom, y_left, y_right)
        if _valid_partition(labels):
            return labels
    return None


def _output_marker_cell(key: str, h: int, w: int) -> tuple[int, int]:
    return {
        "TL": (1, 1),
        "TR": (1, w - 2),
        "BL": (h - 2, 1),
        "BR": (h - 2, w - 2),
    }[key]


def _assemble(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    comps = _components(g != BG)
    assert len(comps) == 4, "expected exactly four multicolor fragments"

    pieces = []
    total_area = 0

    for pts in comps:
        vals = [int(g[r, c]) for r, c in pts]
        body_color = Counter(vals).most_common(1)[0][0]
        marker_cells = [(r, c) for r, c in pts if int(g[r, c]) != body_color]
        assert len(marker_cells) == 1, "each fragment must contain exactly one marker pixel"
        mr, mc = marker_cells[0]

        r0, c0, r1, c1 = _bbox(pts)
        ph = r1 - r0 + 1
        pw = c1 - c0 + 1
        rel = (mr - r0, mc - c0)

        candidates = []
        specs = {
            "TL": ((1, 1), (r0, c0)),
            "TR": ((1, pw - 2), (r0, c1)),
            "BL": ((ph - 2, 1), (r1, c0)),
            "BR": ((ph - 2, pw - 2), (r1, c1)),
        }
        for key, (marker_rel, anchor) in specs.items():
            if rel == marker_rel:
                candidates.append((key, anchor))
        assert len(candidates) == 1, "marker position must identify a unique target corner"

        target_corner, anchor = candidates[0]
        cells = []
        for r, c in pts:
            color = body_color if (r, c) == (mr, mc) else int(g[r, c])
            cells.append((r, c, color))

        pieces.append((target_corner, anchor, cells))
        total_area += len(pts)

    out_h, out_w = _closest_factor_pair(total_area)
    out = np.full((out_h, out_w), BG, dtype=int)
    used = set()

    for target_corner, anchor, cells in pieces:
        assert target_corner not in used, "two fragments claim the same output corner"
        used.add(target_corner)
        ar, ac = anchor
        dest_r, dest_c = CORNER_DEST[target_corner](out_h, out_w)
        for r, c, color in cells:
            rr = dest_r + (r - ar)
            cc = dest_c + (c - ac)
            assert 0 <= rr < out_h and 0 <= cc < out_w, "piece does not fit the inferred rectangle"
            assert out[rr, cc] in (BG, color), "pieces overlap with conflicting colors"
            out[rr, cc] = color

    assert np.all(out != BG), "assembled rectangle must be gap-free"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble four scattered multicolor L-corner fragments into one rectangle.

    Each fragment is a 4-connected component on a black background. The dominant
    color is the fragment body color and exactly one off-color pixel acts as an
    internal direction marker. The marker sits one cell inward from the fragment
    corner that should align to a rectangle corner; its diagonal position encodes
    which rectangle corner the fragment belongs to. Recolor the marker back to
    the fragment body color and translate all four fragments so their marked
    corners align to the four corners of the output rectangle whose area equals
    the total foreground area and whose factor pair is closest to a square.
    """
    return _assemble(grid)


def _place_piece(grid: np.ndarray, blocked: np.ndarray, piece: np.ndarray) -> bool:
    gh, gw = grid.shape
    ph, pw = piece.shape
    mask = piece != BG
    for _ in range(PLACEMENT_ATTEMPTS):
        r0 = random.randint(0, gh - ph)
        c0 = random.randint(0, gw - pw)
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(gh, r0 + ph + 1)
        cc1 = min(gw, c0 + pw + 1)
        if blocked[rr0:rr1, cc0:cc1].any():
            continue
        patch = grid[r0 : r0 + ph, c0 : c0 + pw]
        patch[mask] = piece[mask]
        grid[r0 : r0 + ph, c0 : c0 + pw] = patch
        blocked[rr0:rr1, cc0:cc1] = True
        return True
    return False


def _build_sample() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(GENERATION_ATTEMPTS):
        out_h, out_w = _closest_factor_pair(random.choice(OUTPUT_AREAS))
        labels = _random_partition(out_h, out_w)
        if labels is None:
            continue

        colors = random.sample(PALETTE, 8)
        body_colors = {key: colors[i] for i, key in enumerate(CORNERS)}
        marker_colors = {key: colors[i + 4] for i, key in enumerate(CORNERS)}

        expected = np.full((out_h, out_w), BG, dtype=int)
        pieces = []

        for key in CORNERS:
            piece_full = np.full((out_h, out_w), BG, dtype=int)
            piece_full[labels == key] = body_colors[key]
            mr, mc = _output_marker_cell(key, out_h, out_w)
            assert piece_full[mr, mc] == body_colors[key]
            piece_full[mr, mc] = marker_colors[key]
            expected[labels == key] = body_colors[key]

            ys, xs = np.where(piece_full != BG)
            r0, r1 = int(ys.min()), int(ys.max())
            c0, c1 = int(xs.min()), int(xs.max())
            pieces.append(piece_full[r0 : r1 + 1, c0 : c1 + 1].copy())

        random.shuffle(pieces)
        max_h = max(piece.shape[0] for piece in pieces)
        max_w = max(piece.shape[1] for piece in pieces)
        grid_h = max(20, 2 * max_h + 8, out_h + 8)
        grid_w = max(20, 2 * max_w + 8, out_w + 8)
        grid = np.full((grid_h, grid_w), BG, dtype=int)
        blocked = np.zeros((grid_h, grid_w), dtype=bool)

        ok = True
        for piece in pieces:
            if not _place_piece(grid, blocked, piece):
                ok = False
                break
        if not ok:
            continue

        if np.array_equal(_assemble(grid), expected):
            return grid, expected

    raise RuntimeError("failed to generate a valid input")


def generate_input() -> np.ndarray:
    grid, _ = _build_sample()
    return grid
