import numpy as np
from collections import deque


_RNG = np.random.default_rng()
_GEN_RETRIES = 400

_BG = 0
_FRAME = 5
_PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
_DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in _DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(coords)
    return parts


def _find_frame(grid: np.ndarray) -> dict:
    bg = _most_common_color(grid)
    best = None
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        coords_arr = np.argwhere(grid == color)
        if coords_arr.size == 0:
            continue
        r0, c0 = coords_arr.min(axis=0)
        r1, c1 = coords_arr.max(axis=0)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1] == color
        full_rows = np.flatnonzero(np.all(sub, axis=1))
        full_cols = np.flatnonzero(np.all(sub, axis=0))
        if full_rows.size < 4 or full_cols.size < 4:
            continue
        if (
            full_rows[0] != 0
            or full_rows[-1] != sub.shape[0] - 1
            or full_cols[0] != 0
            or full_cols[-1] != sub.shape[1] - 1
        ):
            continue
        row_steps = np.diff(full_rows)
        col_steps = np.diff(full_cols)
        if len(set(map(int, row_steps))) != 1 or len(set(map(int, col_steps))) != 1:
            continue
        if int(row_steps[0]) < 2 or int(col_steps[0]) < 2:
            continue
        expected = np.zeros_like(sub, dtype=bool)
        expected[full_rows, :] = True
        expected[:, full_cols] = True
        if not np.array_equal(sub, expected):
            continue
        n_rows = int(full_rows.size - 1)
        n_cols = int(full_cols.size - 1)
        if n_rows < 3 or n_cols < 3:
            continue
        score = int(sub.size)
        candidate = {
            "bg": bg,
            "frame": color,
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "row_lines": [int(r0 + x) for x in full_rows],
            "col_lines": [int(c0 + x) for x in full_cols],
            "cell_h": int(row_steps[0] - 1),
            "cell_w": int(col_steps[0] - 1),
            "n_rows": n_rows,
            "n_cols": n_cols,
        }
        if best is None or score > best[0]:
            best = (score, candidate)
    assert best is not None, "frame not found"
    return best[1]


def _extract_cells(grid: np.ndarray, info: dict) -> tuple[list[list[np.ndarray]], list[tuple[int, int]]]:
    cells: list[list[np.ndarray]] = []
    missing: list[tuple[int, int]] = []
    bg = int(info["bg"])
    row_lines = info["row_lines"]
    col_lines = info["col_lines"]
    for rr in range(info["n_rows"]):
        row_cells: list[np.ndarray] = []
        for cc in range(info["n_cols"]):
            r0 = row_lines[rr] + 1
            r1 = row_lines[rr + 1]
            c0 = col_lines[cc] + 1
            c1 = col_lines[cc + 1]
            patch = grid[r0:r1, c0:c1].copy()
            row_cells.append(patch)
            if np.all(patch == bg):
                missing.append((rr, cc))
        cells.append(row_cells)
    return cells, missing


def _extract_candidates(grid: np.ndarray, info: dict) -> list[np.ndarray]:
    bg = int(info["bg"])
    frame = int(info["frame"])
    cell_shape = (int(info["cell_h"]), int(info["cell_w"]))
    r0, c0, r1, c1 = info["bbox"]
    outside = np.ones_like(grid, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    mask = outside & (grid != bg) & (grid != frame)
    pieces: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for coords in _components(mask):
        br0, bc0, br1, bc1 = _bbox(coords)
        patch = grid[br0 : br1 + 1, bc0 : bc1 + 1].copy()
        if patch.shape != cell_shape:
            continue
        pieces.append(((br0, bc0, br1, bc1), patch))
    pieces.sort(key=lambda item: item[0])
    return [patch for _, patch in pieces]


def _fits_against_fixed(
    pos: tuple[int, int],
    patch: np.ndarray,
    cells: list[list[np.ndarray]],
    missing_set: set[tuple[int, int]],
) -> bool:
    r, c = pos
    if r > 0 and (r - 1, c) not in missing_set and not np.array_equal(patch[0, :], cells[r - 1][c][-1, :]):
        return False
    if r + 1 < len(cells) and (r + 1, c) not in missing_set and not np.array_equal(patch[-1, :], cells[r + 1][c][0, :]):
        return False
    if c > 0 and (r, c - 1) not in missing_set and not np.array_equal(patch[:, 0], cells[r][c - 1][:, -1]):
        return False
    if c + 1 < len(cells[0]) and (r, c + 1) not in missing_set and not np.array_equal(patch[:, -1], cells[r][c + 1][:, 0]):
        return False
    return True


def _fits_against_placed(
    pos: tuple[int, int],
    patch: np.ndarray,
    placed: dict[tuple[int, int], np.ndarray],
) -> bool:
    r, c = pos
    if (r - 1, c) in placed and not np.array_equal(patch[0, :], placed[(r - 1, c)][-1, :]):
        return False
    if (r + 1, c) in placed and not np.array_equal(patch[-1, :], placed[(r + 1, c)][0, :]):
        return False
    if (r, c - 1) in placed and not np.array_equal(patch[:, 0], placed[(r, c - 1)][:, -1]):
        return False
    if (r, c + 1) in placed and not np.array_equal(patch[:, -1], placed[(r, c + 1)][:, 0]):
        return False
    return True


def _solve_assignment(
    cells: list[list[np.ndarray]],
    missing: list[tuple[int, int]],
    candidates: list[np.ndarray],
) -> dict[tuple[int, int], tuple[int, int, np.ndarray]]:
    missing_set = set(missing)
    if not missing:
        return {}

    options: dict[tuple[int, int], list[tuple[int, int, np.ndarray]]] = {}
    for pos in missing:
        opts: list[tuple[int, int, np.ndarray]] = []
        for idx, candidate in enumerate(candidates):
            for rot in range(4):
                turned = np.rot90(candidate, rot)
                if _fits_against_fixed(pos, turned, cells, missing_set):
                    opts.append((idx, rot, turned))
        assert opts, "each missing slot must have a candidate"
        options[pos] = opts

    order = sorted(missing, key=lambda pos: len(options[pos]))
    placed: dict[tuple[int, int], np.ndarray] = {}
    used: set[int] = set()
    solutions: list[dict[tuple[int, int], tuple[int, int, np.ndarray]]] = []

    def dfs(depth: int, chosen: dict[tuple[int, int], tuple[int, int, np.ndarray]]) -> None:
        if len(solutions) > 1:
            return
        if depth == len(order):
            solutions.append(chosen.copy())
            return
        pos = order[depth]
        for idx, rot, turned in options[pos]:
            if idx in used:
                continue
            if not _fits_against_placed(pos, turned, placed):
                continue
            used.add(idx)
            placed[pos] = turned
            chosen[pos] = (idx, rot, turned)
            dfs(depth + 1, chosen)
            chosen.pop(pos)
            placed.pop(pos)
            used.remove(idx)

    dfs(0, {})
    assert len(solutions) == 1, "candidate assignment must be unique"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the missing sub-tiles inside a framed grid by matching edge color sequences,
    rotating outside candidate patches as needed, and discarding all unused candidates.
    """
    g = np.asarray(grid, dtype=int)
    info = _find_frame(g)
    cells, missing = _extract_cells(g, info)
    assert 1 <= len(missing) <= 2, "expected one or two missing cells"
    candidates = _extract_candidates(g, info)
    assert len(candidates) > len(missing), "expected distractor candidates"
    assignment = _solve_assignment(cells, missing, candidates)

    out = np.full_like(g, int(info["bg"]))
    r0, c0, r1, c1 = info["bbox"]
    out[r0 : r1 + 1, c0 : c1 + 1] = g[r0 : r1 + 1, c0 : c1 + 1]
    for (rr, cc), (_, _, patch) in assignment.items():
        sr = info["row_lines"][rr] + 1
        er = info["row_lines"][rr + 1]
        sc = info["col_lines"][cc] + 1
        ec = info["col_lines"][cc + 1]
        out[sr:er, sc:ec] = patch
    return out


def _pick_colors() -> np.ndarray:
    return _RNG.choice(_PALETTE, size=3, replace=False)


def _different_color(color: int, colors: np.ndarray) -> int:
    choices = [int(x) for x in colors if int(x) != int(color)]
    return int(_RNG.choice(choices))


def _edge_sequence(length: int, colors: np.ndarray, start: int, end: int) -> np.ndarray:
    seq = _RNG.choice(colors, size=length, replace=True)
    seq[0] = int(start)
    seq[-1] = int(end)
    if np.unique(seq).size == 1:
        seq[length // 2] = _different_color(int(seq[length // 2]), colors)
    return seq.astype(int)


def _paint_motif(patch: np.ndarray, colors: np.ndarray) -> None:
    h, w = map(int, patch.shape)
    base = int(_RNG.choice(colors))
    accent = _different_color(base, colors)
    patch[:, :] = base
    if h <= 2 or w <= 2:
        return
    inner = patch[1:-1, 1:-1]
    mode = int(_RNG.integers(0, 5))
    if mode == 0:
        inner[inner.shape[0] // 2, :] = accent
    elif mode == 1:
        inner[:, inner.shape[1] // 2] = accent
    elif mode == 2:
        inner[:, inner.shape[1] // 2] = accent
        inner[inner.shape[0] // 2, :] = accent
    elif mode == 3:
        inner[: max(1, inner.shape[0] // 2), : max(1, inner.shape[1] // 2)] = accent
    else:
        for i in range(min(inner.shape)):
            inner[i, i] = accent
    if colors.size >= 3 and h >= 5 and w >= 5 and _RNG.random() < 0.45:
        accent2_choices = [int(x) for x in colors if int(x) not in (base, accent)]
        accent2 = int(_RNG.choice(accent2_choices))
        inner[-1, :] = accent2


def _build_patch(
    cell_h: int,
    cell_w: int,
    top: np.ndarray,
    bottom: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    colors: np.ndarray,
) -> np.ndarray | None:
    patch = np.empty((cell_h, cell_w), dtype=int)
    _paint_motif(patch, colors)
    patch[0, :] = top
    patch[-1, :] = bottom
    patch[:, 0] = left
    patch[:, -1] = right
    unique = np.unique(patch)
    if unique.size < 2 or unique.size > 3:
        return None
    return patch


def _build_complete_tiles(
    n_rows: int, n_cols: int, cell_h: int, cell_w: int, colors: np.ndarray
) -> list[list[np.ndarray]] | None:
    corners = _RNG.choice(colors, size=(n_rows + 1, n_cols + 1), replace=True)
    h_edges = [
        [
            _edge_sequence(cell_w, colors, int(corners[r, c]), int(corners[r, c + 1]))
            for c in range(n_cols)
        ]
        for r in range(n_rows + 1)
    ]
    v_edges = [
        [
            _edge_sequence(cell_h, colors, int(corners[r, c]), int(corners[r + 1, c]))
            for c in range(n_cols + 1)
        ]
        for r in range(n_rows)
    ]
    tiles: list[list[np.ndarray]] = []
    seen_signatures: set[bytes] = set()
    for r in range(n_rows):
        row_tiles: list[np.ndarray] = []
        for c in range(n_cols):
            patch = _build_patch(
                cell_h,
                cell_w,
                h_edges[r][c],
                h_edges[r + 1][c],
                v_edges[r][c],
                v_edges[r][c + 1],
                colors,
            )
            if patch is None:
                return None
            sig = patch.tobytes()
            if sig in seen_signatures:
                return None
            seen_signatures.add(sig)
            row_tiles.append(patch)
        tiles.append(row_tiles)
    return tiles


def _eligible_missing_positions(n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    eligible: list[tuple[int, int]] = []
    for r in range(n_rows):
        for c in range(n_cols):
            degree = 4 - (r == 0) - (r == n_rows - 1) - (c == 0) - (c == n_cols - 1)
            if degree >= 3:
                eligible.append((r, c))
    return eligible


def _choose_missing_positions(n_rows: int, n_cols: int, count: int) -> list[tuple[int, int]] | None:
    eligible = _eligible_missing_positions(n_rows, n_cols)
    if len(eligible) < count:
        return None
    if count == 1:
        return [eligible[int(_RNG.integers(0, len(eligible)))]]
    order = _RNG.permutation(len(eligible))
    for i in order:
        a = eligible[int(i)]
        for j in order:
            b = eligible[int(j)]
            if a == b:
                continue
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) > 1:
                return [a, b]
    return None


def _mutated_distractor(source: np.ndarray, colors: np.ndarray) -> np.ndarray:
    patch = source.copy()
    h, w = map(int, patch.shape)
    side = int(_RNG.integers(0, 4))
    if side == 0:
        idx = int(_RNG.integers(0, w))
        patch[0, idx] = _different_color(int(patch[0, idx]), colors)
    elif side == 1:
        idx = int(_RNG.integers(0, w))
        patch[-1, idx] = _different_color(int(patch[-1, idx]), colors)
    elif side == 2:
        idx = int(_RNG.integers(0, h))
        patch[idx, 0] = _different_color(int(patch[idx, 0]), colors)
    else:
        idx = int(_RNG.integers(0, h))
        patch[idx, -1] = _different_color(int(patch[idx, -1]), colors)
    return np.rot90(patch, int(_RNG.integers(0, 4)))


def _render_frame(
    tiles: list[list[np.ndarray]],
    frame_r0: int,
    frame_c0: int,
    frame_color: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    n_rows = len(tiles)
    n_cols = len(tiles[0])
    cell_h, cell_w = map(int, tiles[0][0].shape)
    frame_h = n_rows * cell_h + n_rows + 1
    frame_w = n_cols * cell_w + n_cols + 1
    region = np.full((frame_h, frame_w), frame_color, dtype=int)
    row_lines = [r * (cell_h + 1) for r in range(n_rows + 1)]
    col_lines = [c * (cell_w + 1) for c in range(n_cols + 1)]
    for r in range(n_rows):
        for c in range(n_cols):
            sr = row_lines[r] + 1
            sc = col_lines[c] + 1
            region[sr : sr + cell_h, sc : sc + cell_w] = tiles[r][c]
    return region, (frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1)


def _place_candidates(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    patches: list[np.ndarray],
) -> bool:
    h, w = map(int, grid.shape)
    ph, pw = map(int, patches[0].shape)
    occupied = np.zeros((h, w), dtype=bool)
    fr0, fc0, fr1, fc1 = frame_bbox
    occupied[max(0, fr0 - 1) : min(h, fr1 + 2), max(0, fc0 - 1) : min(w, fc1 + 2)] = True
    order = _RNG.permutation(len(patches))
    placements: list[tuple[int, int, np.ndarray]] = []
    for idx in order:
        patch = patches[int(idx)]
        positions = [
            (r, c)
            for r in range(1, h - ph)
            for c in range(1, w - pw)
            if not occupied[r - 1 : r + ph + 1, c - 1 : c + pw + 1].any()
        ]
        if not positions:
            return False
        r, c = positions[int(_RNG.integers(0, len(positions)))]
        placements.append((r, c, patch))
        occupied[r - 1 : r + ph + 1, c - 1 : c + pw + 1] = True
    for r, c, patch in placements:
        grid[r : r + ph, c : c + pw] = patch
    return True


def _gen_attempt() -> np.ndarray | None:
    n_rows = int(_RNG.choice([3, 4]))
    n_cols = int(_RNG.choice([3, 4]))
    cell_size = int(_RNG.choice([3, 4, 5]))
    cell_h = cell_size
    cell_w = cell_size
    colors = _pick_colors()
    tiles = _build_complete_tiles(n_rows, n_cols, cell_h, cell_w, colors)
    if tiles is None:
        return None

    missing_count = int(_RNG.choice([1, 2]))
    missing = _choose_missing_positions(n_rows, n_cols, missing_count)
    if missing is None:
        return None

    frame_h = n_rows * cell_h + n_rows + 1
    frame_w = n_cols * cell_w + n_cols + 1
    canvas_h = frame_h + 4 * cell_h + 12
    canvas_w = frame_w + 4 * cell_w + 12
    grid = np.full((canvas_h, canvas_w), _BG, dtype=int)

    frame_r0 = (canvas_h - frame_h) // 2 + int(_RNG.integers(-1, 2))
    frame_c0 = (canvas_w - frame_w) // 2 + int(_RNG.integers(-1, 2))
    frame_r0 = max(cell_h + 3, min(frame_r0, canvas_h - frame_h - cell_h - 3))
    frame_c0 = max(cell_w + 3, min(frame_c0, canvas_w - frame_w - cell_w - 3))

    frame_region, frame_bbox = _render_frame(tiles, frame_r0, frame_c0, _FRAME)
    fr0, fc0, fr1, fc1 = frame_bbox
    solved = np.full_like(grid, _BG)
    solved[fr0 : fr1 + 1, fc0 : fc1 + 1] = frame_region
    grid[fr0 : fr1 + 1, fc0 : fc1 + 1] = frame_region

    for r, c in missing:
        sr = fr0 + 1 + r * (cell_h + 1)
        sc = fc0 + 1 + c * (cell_w + 1)
        grid[sr : sr + cell_h, sc : sc + cell_w] = _BG

    candidate_patches: list[np.ndarray] = []
    for r, c in missing:
        rot = int(_RNG.integers(0, 4))
        candidate_patches.append(np.rot90(tiles[r][c], rot))

    extra_count = int(_RNG.choice([1, 2, 3]))
    non_missing_sources = [tiles[r][c] for r in range(n_rows) for c in range(n_cols) if (r, c) not in set(missing)]
    for _ in range(extra_count):
        source = non_missing_sources[int(_RNG.integers(0, len(non_missing_sources)))]
        candidate_patches.append(_mutated_distractor(source, colors))
    _RNG.shuffle(candidate_patches)
    if not _place_candidates(grid, frame_bbox, candidate_patches):
        return None

    try:
        produced = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(produced, solved):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
