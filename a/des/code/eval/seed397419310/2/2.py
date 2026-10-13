import numpy as np


BG = 0
PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
SIDES = ("top", "bottom", "left", "right")
GEN_MAX_ATTEMPTS = 400
MIN_GRID_SIDE = 14
MAX_GRID_SIDE = 30


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _component_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _connected_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _normalize_coords(coords: np.ndarray) -> frozenset[tuple[int, int]]:
    if len(coords) == 0:
        return frozenset()
    rr = coords[:, 0] - int(coords[:, 0].min())
    cc = coords[:, 1] - int(coords[:, 1].min())
    return frozenset((int(r), int(c)) for r, c in np.column_stack((rr, cc)))


def _normalize_mask(mask: np.ndarray) -> frozenset[tuple[int, int]]:
    return _normalize_coords(np.argwhere(mask))


def _infer_ring(coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 3 or width < 3:
        return None
    mask = _component_mask(coords, (r0, c0, r1, c1))
    border = _border_mask(height, width)
    if np.any(mask & ~border):
        return None

    missing = np.argwhere(border & ~mask)
    if missing.size == 0:
        return None

    side: str | None = None
    axis: np.ndarray
    side_len: int
    if np.all(missing[:, 0] == 0):
        side = "top"
        axis = missing[:, 1]
        side_len = width
    elif np.all(missing[:, 0] == height - 1):
        side = "bottom"
        axis = missing[:, 1]
        side_len = width
    elif np.all(missing[:, 1] == 0):
        side = "left"
        axis = missing[:, 0]
        side_len = height
    elif np.all(missing[:, 1] == width - 1):
        side = "right"
        axis = missing[:, 0]
        side_len = height
    else:
        return None

    axis = np.sort(axis.astype(int))
    if not np.array_equal(axis, np.arange(int(axis[0]), int(axis[0]) + len(axis))):
        return None
    if int(axis[0]) <= 0 or int(axis[-1]) >= side_len - 1:
        return None

    gap_abs = missing + np.array([[r0, c0]], dtype=int)
    return {
        "bbox": (r0, c0, r1, c1),
        "side": side,
        "gap_coords": gap_abs,
        "gap_shape": _normalize_coords(gap_abs),
    }


def _piece_matches_gap(piece_coords: np.ndarray, gap_shape: frozenset[tuple[int, int]]) -> bool:
    bbox = _bbox(piece_coords)
    mask = _component_mask(piece_coords, bbox)
    seen: set[frozenset[tuple[int, int]]] = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        shape = _normalize_mask(rotated)
        if shape in seen:
            continue
        seen.add(shape)
        if shape == gap_shape:
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()

    colors = [int(c) for c in np.unique(g) if int(c) != BG]
    for color in colors:
        comps = _connected_components(g, color)
        assert len(comps) == 2, f"color {color} should have one ring and one fragment"

        ring_candidates: list[tuple[np.ndarray, dict]] = []
        piece_candidates: list[np.ndarray] = []
        for comp in comps:
            ring = _infer_ring(comp)
            if ring is not None:
                ring_candidates.append((comp, ring))
            else:
                piece_candidates.append(comp)

        assert len(ring_candidates) == 1, f"color {color} should have exactly one rectangular ring"
        assert len(piece_candidates) == 1, f"color {color} should have exactly one fragment"

        piece_coords = piece_candidates[0]
        ring_info = ring_candidates[0][1]
        gap_coords = np.asarray(ring_info["gap_coords"], dtype=int)
        assert int(piece_coords.shape[0]) == int(gap_coords.shape[0]), "fragment area must equal gap length"
        assert _piece_matches_gap(piece_coords, ring_info["gap_shape"]), "fragment must match the gap after rotation"

        out[piece_coords[:, 0], piece_coords[:, 1]] = BG
        out[gap_coords[:, 0], gap_coords[:, 1]] = color

    return out


def _build_ring_patch(color: int, height: int, width: int, side: str, gap_len: int) -> np.ndarray:
    patch = np.zeros((height, width), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color

    if side in ("top", "bottom"):
        start = (width - gap_len) // 2
        row = 0 if side == "top" else height - 1
        patch[row, start : start + gap_len] = BG
    else:
        start = (height - gap_len) // 2
        col = 0 if side == "left" else width - 1
        patch[start : start + gap_len, col] = BG
    return patch


def _choose_near(rng: np.random.Generator, allowed: list[int], preferred: int) -> int:
    assert allowed
    close = [v for v in allowed if abs(v - preferred) <= 2]
    pool = close if close else allowed
    idx = int(rng.integers(0, len(pool)))
    return int(pool[idx])


def _crop_nonzero(patch: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch != BG)
    r0, c0, r1, c1 = _bbox(coords)
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy()


def _build_pair_patch(rng: np.random.Generator, color: int) -> np.ndarray:
    side = SIDES[int(rng.integers(0, len(SIDES)))]
    gap_len = int(rng.choice(np.array([2, 3], dtype=int)))
    height = int(rng.integers(5, 10))
    width = int(rng.integers(5, 10))
    distance = int(rng.integers(2, 5))
    piece_orient = "h" if int(rng.integers(0, 2)) == 0 else "v"

    ring = _build_ring_patch(color, height, width, side, gap_len)
    if piece_orient == "h":
        piece = np.full((1, gap_len), color, dtype=int)
    else:
        piece = np.full((gap_len, 1), color, dtype=int)
    piece_h, piece_w = piece.shape

    if side in ("top", "bottom"):
        canvas = np.zeros((height + distance + piece_h, width), dtype=int)
        if piece_orient == "h":
            allowed = list(range(0, width - gap_len + 1))
            preferred = (width - gap_len) // 2
        else:
            allowed = list(range(1, width - 1))
            preferred = width // 2
        piece_c = _choose_near(rng, allowed, preferred)
        if side == "top":
            piece_r = 0
            ring_r = piece_h + distance
        else:
            ring_r = 0
            piece_r = height + distance
        canvas[ring_r : ring_r + height, :] = ring
        canvas[piece_r : piece_r + piece_h, piece_c : piece_c + piece_w] = piece
    else:
        canvas = np.zeros((height, width + distance + piece_w), dtype=int)
        if piece_orient == "v":
            allowed = list(range(0, height - gap_len + 1))
            preferred = (height - gap_len) // 2
        else:
            allowed = list(range(1, height - 1))
            preferred = height // 2
        piece_r = _choose_near(rng, allowed, preferred)
        if side == "left":
            piece_c = 0
            ring_c = piece_w + distance
        else:
            ring_c = 0
            piece_c = width + distance
        canvas[:, ring_c : ring_c + width] = ring
        canvas[piece_r : piece_r + piece_h, piece_c : piece_c + piece_w] = piece

    return _crop_nonzero(canvas)


def _row_partition(rng: np.random.Generator, count: int) -> list[int]:
    if count == 1:
        return [1]
    max_rows = min(3, count)
    min_rows = 1 if count == 2 else 2
    rows = int(rng.integers(min_rows, max_rows + 1))
    cuts = sorted(int(x) for x in rng.choice(np.arange(1, count), size=rows - 1, replace=False))
    parts: list[int] = []
    prev = 0
    for cut in cuts + [count]:
        parts.append(int(cut - prev))
        prev = cut
    return parts


def _assemble_grid(rng: np.random.Generator, patches: list[np.ndarray]) -> np.ndarray | None:
    order = list(range(len(patches)))
    rng.shuffle(order)
    ordered = [patches[i] for i in order]
    row_sizes = _row_partition(rng, len(ordered))

    rows: list[list[np.ndarray]] = []
    pos = 0
    for row_size in row_sizes:
        rows.append(ordered[pos : pos + row_size])
        pos += row_size

    h_gaps: list[list[int]] = []
    row_widths: list[int] = []
    row_heights: list[int] = []
    for row in rows:
        gaps = [int(rng.integers(2, 5)) for _ in range(max(0, len(row) - 1))]
        widths = [int(p.shape[1]) for p in row]
        heights = [int(p.shape[0]) for p in row]
        h_gaps.append(gaps)
        row_widths.append(int(sum(widths) + sum(gaps)))
        row_heights.append(int(max(heights)))

    v_gaps = [int(rng.integers(2, 5)) for _ in range(max(0, len(rows) - 1))]
    margin_top = int(rng.integers(1, 4))
    margin_bottom = int(rng.integers(1, 4))
    margin_left = int(rng.integers(1, 4))
    margin_right = int(rng.integers(1, 4))

    base_h = margin_top + margin_bottom + sum(row_heights) + sum(v_gaps)
    base_w = margin_left + margin_right + max(row_widths)
    if base_h > MAX_GRID_SIDE or base_w > MAX_GRID_SIDE:
        return None

    need_h = max(0, MIN_GRID_SIDE - base_h)
    need_w = max(0, MIN_GRID_SIDE - base_w)
    extra_h = int(rng.integers(need_h, max(need_h, min(5, MAX_GRID_SIDE - base_h)) + 1))
    extra_w = int(rng.integers(need_w, max(need_w, min(5, MAX_GRID_SIDE - base_w)) + 1))
    grid_h = base_h + extra_h
    grid_w = base_w + extra_w
    if grid_h > MAX_GRID_SIDE or grid_w > MAX_GRID_SIDE:
        return None

    top_shift = int(rng.integers(0, extra_h + 1))
    row_r = margin_top + top_shift
    placements: list[tuple[np.ndarray, int, int]] = []
    for row_idx, row in enumerate(rows):
        row_h = row_heights[row_idx]
        row_w = row_widths[row_idx]
        free_x = grid_w - margin_left - margin_right - row_w
        row_c = margin_left + int(rng.integers(0, free_x + 1))
        for patch_idx, patch in enumerate(row):
            patch_h, patch_w = patch.shape
            offset_r = int(rng.integers(0, row_h - patch_h + 1))
            placements.append((patch, row_r + offset_r, row_c))
            row_c += patch_w
            if patch_idx < len(row) - 1:
                row_c += h_gaps[row_idx][patch_idx]
        row_r += row_h
        if row_idx < len(rows) - 1:
            row_r += v_gaps[row_idx]

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for patch, r0, c0 in placements:
        h, w = patch.shape
        region = grid[r0 : r0 + h, c0 : c0 + w]
        if np.any((region != BG) & (patch != BG)):
            return None
        region[patch != BG] = patch[patch != BG]
    return grid


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    object_count = int(rng.integers(1, 5))
    colors = [int(c) for c in rng.choice(PALETTE, size=object_count, replace=False)]
    patches = [_build_pair_patch(rng, color) for color in colors]
    grid = _assemble_grid(rng, patches)
    if grid is None:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ring-gap puzzle")
