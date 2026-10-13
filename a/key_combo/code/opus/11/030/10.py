import numpy as np


BG = 0
BLUE = 1
RED = 2
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _frame_local_mask(height: int, width: int) -> np.ndarray:
    out = np.zeros((height, width), dtype=bool)
    out[0, :] = True
    out[-1, :] = True
    out[:, 0] = True
    out[:, -1] = True
    return out


def _frame_coords(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _is_rectangular_frame(comp: np.ndarray) -> bool:
    if len(comp) < 8:
        return False
    rows = comp[:, 0]
    cols = comp[:, 1]
    r0, c0 = rows.min(), cols.min()
    r1, c1 = rows.max(), cols.max()
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False
    local = np.zeros((h, w), dtype=bool)
    local[rows - r0, cols - c0] = True
    return np.array_equal(local, _frame_local_mask(h, w))


def _extract_frame_boxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    comps = _components(grid == RED, connectivity=4)
    boxes = []
    for comp in comps:
        if not _is_rectangular_frame(comp):
            raise AssertionError("all red components must be 1-pixel rectangular frames")
        rows = comp[:, 0]
        cols = comp[:, 1]
        boxes.append((int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())))
    boxes.sort(key=lambda b: (b[2] - b[0] + 1) * (b[3] - b[1] + 1), reverse=True)
    if not 2 <= len(boxes) <= 4:
        raise AssertionError("expected 2-4 nested frames")
    for outer, inner in zip(boxes, boxes[1:]):
        if inner != (outer[0] + 2, outer[1] + 2, outer[2] - 2, outer[3] - 2):
            raise AssertionError("frames must be concentric with a 1-cell black gap")
    return boxes


def _diag_only_relative_to_box(cell: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    r, c = cell
    frame = _frame_coords(box)
    has_diag = any((r + dr, c + dc) in frame for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)))
    has_ortho = any((r + dr, c + dc) in frame for dr, dc in DIR4)
    return has_diag and not has_ortho


def _diagonal_validation_rounds(grid: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> int:
    h, w = grid.shape
    rounds = 0
    prev_box = None
    for box in boxes:
        r0, c0, r1, c1 = box
        corners = [(r0 - 1, c0 - 1), (r0 - 1, c1 + 1), (r1 + 1, c0 - 1), (r1 + 1, c1 + 1)]
        for rr, cc in corners:
            if not _in_bounds((h, w), rr, cc):
                raise AssertionError("outer frame must have padding for diagonal validation")
            if grid[rr, cc] != BG:
                raise AssertionError("diagonal validation cells must stay black")
            if not _diag_only_relative_to_box((rr, cc), box):
                raise AssertionError("validation cell must be diagonal-only relative to its frame")
            if prev_box is not None:
                pr0, pc0, pr1, pc1 = prev_box
                if not (pr0 < rr < pr1 and pc0 < cc < pc1):
                    raise AssertionError("inner validation cells must lie inside the previous frame")
        prev_box = box
        rounds += 1
    return rounds


def _nearest_resize(grid: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    h, w = grid.shape
    row_idx = (np.arange(out_h) * h) // out_h
    col_idx = (np.arange(out_w) * w) // out_w
    return grid[row_idx[:, None], col_idx[None, :]]


def _extract_motif(grid: np.ndarray, inner_box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = inner_box
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if np.any((interior != BG) & (interior != BLUE)):
        raise AssertionError("the innermost frame interior may only contain the blue motif")
    motif_box = _bbox(interior == BLUE)
    if motif_box is None:
        raise AssertionError("expected a blue motif inside the innermost frame")
    motif = _crop_bbox(np.where(interior == BLUE, BLUE, BG), motif_box)
    if len(_components(motif == BLUE, connectivity=4)) != 1:
        raise AssertionError("blue motif must be 4-connected")
    return motif


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the nested-frame depth and resize the innermost blue motif to an N-by-N icon.

    Input structure:
    - The grid has black background, 2-4 concentric red rectangular frames, and a blue motif only inside the innermost frame.
    - Every frame is 1 pixel thick, all frames share the same red color, and consecutive frames are separated by exactly 1 black cell.
    - The blue motif is a small 4-connected pattern placed somewhere inside the innermost frame interior.

    Rule:
    - Detect the red nested frames and count their depth N.
    - Confirm that each frame contributes one diagonal-only validation layer, so the diagonal propagation stabilizes after N rounds.
    - Crop the blue motif to its tight bounding box, then resize that cropped pattern to N by N with nearest-neighbor scaling.
    - Output only the resized blue motif on a black N-by-N canvas.

    Key insight:
    - The same nesting depth appears twice: geometrically as the number of frames and numerically as the output resolution needed to normalize the innermost pattern.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")
    boxes = _extract_frame_boxes(g)
    n = len(boxes)

    red_mask = g == RED
    blue_mask = g == BLUE
    if np.any(~((g == BG) | red_mask | blue_mask)):
        raise AssertionError("unexpected colors present")
    inner_box = boxes[-1]
    ir0, ic0, ir1, ic1 = inner_box
    allowed_blue = np.zeros_like(blue_mask, dtype=bool)
    allowed_blue[ir0 + 1 : ir1, ic0 + 1 : ic1] = True
    if np.any(blue_mask & ~allowed_blue):
        raise AssertionError("blue pixels must lie strictly inside the innermost frame")

    rounds = _diagonal_validation_rounds(g, boxes)
    if rounds != n:
        raise AssertionError("diagonal validation rounds must equal nesting depth")

    motif = _extract_motif(g, inner_box)
    out = _nearest_resize(motif, n, n)
    return np.where(out == BLUE, BLUE, BG).astype(int)


def _normalize_binary(mask: np.ndarray) -> np.ndarray:
    return np.where(mask != 0, BLUE, BG).astype(int)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    seen = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        for variant in (rotated, np.fliplr(rotated)):
            box = _bbox(variant == BLUE)
            cropped = _crop_bbox(variant, box)
            key = (cropped.shape, cropped.tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(cropped.copy())
    return variants


def _is_asymmetric(mask: np.ndarray) -> bool:
    original = _normalize_binary(mask)
    for k in range(4):
        rotated = np.rot90(original, k)
        for variant in (rotated, np.fliplr(rotated)):
            if k == 0 and np.array_equal(variant, original):
                continue
            if variant.shape == original.shape and np.array_equal(variant, original):
                return False
    return True


def _pattern_library() -> list[np.ndarray]:
    bases = []
    for coords in (
        [(0, 0), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1), (2, 1)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
        [(0, 0), (1, 0), (1, 1), (1, 2)],
        [(0, 1), (1, 1), (2, 0), (2, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 0)],
    ):
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        arr = np.zeros((max(rows) + 1, max(cols) + 1), dtype=int)
        for r, c in coords:
            arr[r, c] = BLUE
        bases.append(arr)

    library = []
    seen = set()
    for base in bases:
        for variant in _dihedral_variants(base):
            if len(_components(variant == BLUE, connectivity=4)) != 1:
                continue
            if not _is_asymmetric(variant):
                continue
            key = (variant.shape, variant.tobytes())
            if key not in seen:
                seen.add(key)
                library.append(variant.copy())
    return library


PATTERNS = _pattern_library()


def _draw_frames(grid: np.ndarray, outer_top: int, outer_left: int, n: int, inner_h: int, inner_w: int) -> list[tuple[int, int, int, int]]:
    boxes = []
    frame_h = inner_h + 2 + 4 * (n - 1)
    frame_w = inner_w + 2 + 4 * (n - 1)
    for layer in range(n):
        r0 = outer_top + 2 * layer
        c0 = outer_left + 2 * layer
        r1 = r0 + frame_h - 1 - 4 * layer
        c1 = c0 + frame_w - 1 - 4 * layer
        boxes.append((r0, c0, r1, c1))
        grid[r0, c0 : c1 + 1] = RED
        grid[r1, c0 : c1 + 1] = RED
        grid[r0 : r1 + 1, c0] = RED
        grid[r0 : r1 + 1, c1] = RED
    return boxes


def _choose_pattern(rng: np.random.Generator, n: int, inner_h: int, inner_w: int) -> np.ndarray | None:
    candidates = []
    for pattern in PATTERNS:
        ph, pw = pattern.shape
        if ph > inner_h or pw > inner_w:
            continue
        if (ph, pw) == (n, n):
            continue
        if ph == inner_h and pw == inner_w:
            continue
        candidates.append(pattern)
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))].copy()


def _validate_generated_input(grid: np.ndarray) -> tuple[int, np.ndarray]:
    if grid.ndim != 2 or max(grid.shape) > 30:
        raise AssertionError("generated input must be a 2D ARC-sized grid")
    boxes = _extract_frame_boxes(grid)
    n = len(boxes)
    if _diagonal_validation_rounds(grid, boxes) != n:
        raise AssertionError("generated input must satisfy the diagonal depth check")
    motif = _extract_motif(grid, boxes[-1])
    mh, mw = motif.shape
    inner_box = boxes[-1]
    inner_h = inner_box[2] - inner_box[0] - 1
    inner_w = inner_box[3] - inner_box[1] - 1
    if (mh, mw) == (n, n):
        raise AssertionError("scaling must be non-identity")
    if mh == inner_h and mw == inner_w:
        raise AssertionError("cropping must discard at least one empty row or column")
    if not _is_asymmetric(motif):
        raise AssertionError("motif must stay asymmetric after cropping")
    expected = _nearest_resize(motif, n, n)
    actual = main(grid)
    if actual.shape != (n, n):
        raise AssertionError("output shape must be N by N")
    if not np.array_equal(actual, expected):
        raise AssertionError("main() must match the independently computed resize result")
    if np.count_nonzero(actual == BLUE) == 0 or np.count_nonzero(actual == BLUE) == actual.size:
        raise AssertionError("output must remain non-degenerate")
    return n, motif


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        n = int(rng.integers(2, 5))
        inner_h = int(rng.integers(3, 6))
        inner_w = int(rng.integers(3, 6))
        pattern = _choose_pattern(rng, n, inner_h, inner_w)
        if pattern is None:
            continue
        ph, pw = pattern.shape
        if inner_h == ph and inner_w == pw:
            continue

        outer_h = inner_h + 2 + 4 * (n - 1)
        outer_w = inner_w + 2 + 4 * (n - 1)
        max_h = min(30, outer_h + 8)
        max_w = min(30, outer_w + 8)
        if max_h < outer_h + 2 or max_w < outer_w + 2:
            continue
        grid_h = int(rng.integers(outer_h + 2, max_h + 1))
        grid_w = int(rng.integers(outer_w + 2, max_w + 1))
        outer_top = int(rng.integers(1, grid_h - outer_h))
        outer_left = int(rng.integers(1, grid_w - outer_w))

        grid = np.full((grid_h, grid_w), BG, dtype=int)
        boxes = _draw_frames(grid, outer_top, outer_left, n, inner_h, inner_w)
        ir0, ic0, _, _ = boxes[-1]
        motif_top = int(rng.integers(ir0 + 1, ir0 + inner_h - ph + 2))
        motif_left = int(rng.integers(ic0 + 1, ic0 + inner_w - pw + 2))
        grid[motif_top : motif_top + ph, motif_left : motif_left + pw][pattern == BLUE] = BLUE

        try:
            _validate_generated_input(grid)
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid nested-frame ARC input")
