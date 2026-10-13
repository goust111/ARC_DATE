import random
from math import isqrt

import numpy as np


BG = 0
MARKER = 9
PIECE_COLORS = [1, 2, 3, 4, 5, 6, 7, 8]

LAYOUT_CHOICES = [(2, 2), (2, 3), (3, 2)]
SEGMENT_SIZE_CHOICES = [3, 4]
GEN_RETRIES = 300
SCATTER_RETRIES = 80


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_component(grid: np.ndarray, coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _normalize_piece(patch: np.ndarray) -> np.ndarray:
    assert int(np.count_nonzero(patch == MARKER)) == 1
    best_patch = None
    best_score = None
    for k in range(4):
        rot = np.rot90(patch, k)
        pos = np.argwhere(rot == MARKER)
        assert len(pos) == 1
        mr, mc = map(int, pos[0])
        h, w = rot.shape
        in_top_left = (2 * mr < h) and (2 * mc < w)
        score = (0 if in_top_left else 1, mr + mc, mr, mc, h + w)
        if best_score is None or score < best_score:
            best_score = score
            best_patch = rot
    assert best_patch is not None and best_score is not None and best_score[0] == 0
    return best_patch


def _piece_sort_key(patch: np.ndarray) -> tuple[int, int, int, bytes]:
    mask = patch != BG
    return (
        -int(mask.sum()),
        -int(patch.shape[0]),
        -int(patch.shape[1]),
        mask.astype(np.uint8).tobytes(),
    )


def _extract_normalized_pieces(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    comps = _components(g != BG)
    assert comps, "expected at least one foreground piece"
    pieces = [_normalize_piece(_crop_component(g, comp)) for comp in comps]
    pieces.sort(key=_piece_sort_key)
    return pieces


def _candidate_rectangles(area: int, pieces: list[dict]) -> list[tuple[int, int]]:
    max_h = max(piece["h"] for piece in pieces)
    max_w = max(piece["w"] for piece in pieces)
    dims = []
    for h in range(1, isqrt(area) + 1):
        if area % h:
            continue
        w = area // h
        for hh, ww in ((h, w),) if h == w else ((h, w), (w, h)):
            if hh >= max_h and ww >= max_w:
                dims.append((int(hh), int(ww)))
    dims.sort(key=lambda hw: (abs(hw[0] - hw[1]), hw[0], hw[1]))
    return dims


def _pack_solutions(patches: list[np.ndarray], limit: int = 2) -> list[np.ndarray]:
    pieces = []
    total_area = 0
    for patch in patches:
        coords = np.argwhere(patch != BG)
        rows = coords[:, 0].astype(int)
        cols = coords[:, 1].astype(int)
        vals = patch[rows, cols].astype(int)
        area = int(len(coords))
        total_area += area
        pieces.append(
            {
                "patch": patch,
                "rows": rows,
                "cols": cols,
                "vals": vals,
                "h": int(patch.shape[0]),
                "w": int(patch.shape[1]),
                "area": area,
            }
        )

    solutions: list[np.ndarray] = []
    seen_keys: set[tuple[tuple[int, int], bytes]] = set()

    def record(board: np.ndarray) -> None:
        key = (tuple(map(int, board.shape)), board.tobytes())
        if key in seen_keys:
            return
        seen_keys.add(key)
        solutions.append(board.copy())

    def search(
        H: int,
        W: int,
        occ: np.ndarray,
        board: np.ndarray,
        remaining: tuple[int, ...],
    ) -> None:
        if len(solutions) >= limit:
            return
        if not remaining:
            record(board)
            return

        first = int(np.flatnonzero(~occ)[0])
        fr, fc = divmod(first, W)

        for piece_id in remaining:
            piece = pieces[piece_id]
            tried_positions = set()
            for pr, pc in zip(piece["rows"], piece["cols"]):
                top = int(fr - pr)
                left = int(fc - pc)
                if (top, left) in tried_positions:
                    continue
                tried_positions.add((top, left))
                if top < 0 or left < 0 or top + piece["h"] > H or left + piece["w"] > W:
                    continue
                rs = top + piece["rows"]
                cs = left + piece["cols"]
                if bool(occ[rs, cs].any()):
                    continue

                occ[rs, cs] = True
                board[rs, cs] = piece["vals"]
                next_remaining = tuple(idx for idx in remaining if idx != piece_id)
                search(H, W, occ, board, next_remaining)
                board[rs, cs] = BG
                occ[rs, cs] = False

                if len(solutions) >= limit:
                    return

    for H, W in _candidate_rectangles(total_area, pieces):
        occ = np.zeros((H, W), dtype=bool)
        board = np.full((H, W), BG, dtype=int)
        search(H, W, occ, board, tuple(range(len(pieces))))
        if len(solutions) >= limit:
            break

    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    pieces = _extract_normalized_pieces(np.asarray(grid, dtype=int))
    solutions = _pack_solutions(pieces, limit=2)
    assert len(solutions) == 1, "assembly must be unique"
    return solutions[0]


def _choose_layout(rng: random.Random) -> tuple[int, int]:
    if rng.random() < 0.5:
        return 2, 2
    return rng.choice([(2, 3), (3, 2)])


def _sample_segments(count: int, rng: random.Random) -> list[int]:
    while True:
        vals = [rng.choice(SEGMENT_SIZE_CHOICES) for _ in range(count)]
        if len(set(vals)) > 1 or count == 1:
            return vals


def _marker_cell(mask: np.ndarray, rng: random.Random) -> tuple[int, int]:
    h, w = mask.shape
    r_lim = max(2, (h + 2) // 3)
    c_lim = max(2, (w + 2) // 3)
    coords = np.argwhere(mask)
    candidates = [(int(r), int(c)) for r, c in coords if int(r) < r_lim and int(c) < c_lim]
    if not candidates:
        candidates = [(int(r), int(c)) for r, c in coords if 2 * int(r) < h and 2 * int(c) < w]
    assert candidates
    candidates.sort(key=lambda rc: (rc[0] + rc[1], rc[0], rc[1]))
    return rng.choice(candidates[: min(4, len(candidates))])


def _build_owner_grid(rng: random.Random) -> np.ndarray:
    for _ in range(100):
        rows, cols = _choose_layout(rng)
        heights = _sample_segments(rows, rng)
        widths = _sample_segments(cols, rng)

        H = sum(heights)
        W = sum(widths)
        owner = np.full((H, W), -1, dtype=int)

        row_starts = [0]
        for size in heights[:-1]:
            row_starts.append(row_starts[-1] + size)
        col_starts = [0]
        for size in widths[:-1]:
            col_starts.append(col_starts[-1] + size)

        for i in range(rows):
            r0 = row_starts[i]
            r1 = r0 + heights[i]
            for j in range(cols):
                c0 = col_starts[j]
                c1 = c0 + widths[j]
                owner[r0:r1, c0:c1] = i * cols + j

        for i in range(rows):
            r0 = row_starts[i]
            r1 = r0 + heights[i]
            edge_len = heights[i]
            for j in range(cols - 1):
                split = col_starts[j + 1]
                left_id = i * cols + j
                right_id = left_id + 1
                max_len = min(2, edge_len - 2)
                tooth_len = 1 if max_len == 1 else rng.choice([1, 2])
                start = rng.randint(r0 + 1, r1 - tooth_len - 1)
                if rng.random() < 0.5:
                    owner[start : start + tooth_len, split] = left_id
                else:
                    owner[start : start + tooth_len, split - 1] = right_id

        for i in range(rows - 1):
            split = row_starts[i + 1]
            for j in range(cols):
                c0 = col_starts[j]
                c1 = c0 + widths[j]
                edge_len = widths[j]
                top_id = i * cols + j
                bottom_id = top_id + cols
                max_len = min(2, edge_len - 2)
                tooth_len = 1 if max_len == 1 else rng.choice([1, 2])
                start = rng.randint(c0 + 1, c1 - tooth_len - 1)
                if rng.random() < 0.5:
                    owner[split, start : start + tooth_len] = top_id
                else:
                    owner[split - 1, start : start + tooth_len] = bottom_id

        good = True
        for pid in range(int(owner.max()) + 1):
            if len(_components(owner == pid)) != 1:
                good = False
                break
        if good:
            return owner

    raise AssertionError("failed to build connected jigsaw pieces")


def _owner_to_case(owner: np.ndarray, rng: random.Random) -> tuple[np.ndarray, list[np.ndarray]]:
    piece_count = int(owner.max()) + 1
    colors = rng.sample(PIECE_COLORS, piece_count)

    output = np.zeros(owner.shape, dtype=int)
    for pid, color in enumerate(colors):
        output[owner == pid] = color

    patches: list[np.ndarray] = []
    for pid, color in enumerate(colors):
        coords = np.argwhere(owner == pid)
        r0, c0, r1, c1 = _bbox(coords)
        mask = owner[r0 : r1 + 1, c0 : c1 + 1] == pid
        patch = np.zeros(mask.shape, dtype=int)
        patch[mask] = color
        mr, mc = _marker_cell(mask, rng)
        patch[mr, mc] = MARKER
        output[r0 + mr, c0 + mc] = MARKER
        patches.append(patch)

    return output, patches


def _can_scatter(occ: np.ndarray, cells: np.ndarray, top: int, left: int) -> bool:
    H, W = occ.shape
    for rr, cc in cells:
        r = int(top + rr)
        c = int(left + cc)
        if bool(occ[max(0, r - 1) : min(H, r + 2), max(0, c - 1) : min(W, c + 2)].any()):
            return False
    return True


def _scatter_pieces(
    patches: list[np.ndarray],
    output_shape: tuple[int, int],
    rng: random.Random,
) -> np.ndarray | None:
    count = len(patches)
    rotations = [rng.randrange(4) for _ in range(count)]
    if all(rot == 0 for rot in rotations):
        rotations[rng.randrange(count)] = rng.choice([1, 2, 3])

    rotated = [np.rot90(patch, rot) for patch, rot in zip(patches, rotations)]
    total_area = sum(int(np.count_nonzero(patch)) for patch in rotated)
    max_h = max(int(patch.shape[0]) for patch in rotated)
    max_w = max(int(patch.shape[1]) for patch in rotated)

    base_side = int(np.ceil(np.sqrt(total_area * 2.6)))
    min_h = max(base_side, output_shape[0] + 4, max_h + 2)
    min_w = max(base_side, output_shape[1] + 4, max_w + 2)

    for _ in range(SCATTER_RETRIES):
        H = min_h + rng.randint(0, 4 + count)
        W = min_w + rng.randint(0, 4 + count)
        grid = np.zeros((H, W), dtype=int)
        occ = np.zeros((H, W), dtype=bool)

        order = list(range(count))
        rng.shuffle(order)
        ok = True

        for idx in order:
            patch = rotated[idx]
            cells = np.argwhere(patch != BG)
            ph, pw = map(int, patch.shape)
            positions = []

            for _ in range(40):
                top = rng.randint(0, H - ph)
                left = rng.randint(0, W - pw)
                if _can_scatter(occ, cells, top, left):
                    positions.append((top, left))
                    if len(positions) >= 4:
                        break

            if not positions:
                for top in range(H - ph + 1):
                    for left in range(W - pw + 1):
                        if _can_scatter(occ, cells, top, left):
                            positions.append((top, left))
                if not positions:
                    ok = False
                    break

            top, left = rng.choice(positions)
            rs = top + cells[:, 0]
            cs = left + cells[:, 1]
            grid[rs, cs] = patch[cells[:, 0], cells[:, 1]]
            occ[rs, cs] = True

        if ok:
            return grid

    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    owner = _build_owner_grid(rng)
    output, patches = _owner_to_case(owner, rng)
    inp = _scatter_pieces(patches, output.shape, rng)
    if inp is None:
        return None
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, output):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid puzzle instance")
