import numpy as np


BACKGROUND = 0
MARKER = 1
BASE = 4
NOISE_COLORS = (2, 3, 5, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GEN_ATTEMPTS = 400
MARKER_ATTEMPTS = 300
NOISE_ATTEMPTS = 500


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _rectangle_candidates(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    candidates: list[tuple[int, int, int, int, int]] = []
    for coords in _components(grid != BACKGROUND):
        r0, c0, r1, c1 = _bbox_coords(coords)
        crop = grid[r0 : r1 + 1, c0 : c1 + 1]
        if int(crop.size) != int(coords.shape[0]):
            continue
        if not np.isin(crop, (BASE, MARKER)).all():
            continue
        if not (np.all(crop[0, :] == BASE) and np.all(crop[-1, :] == BASE)):
            continue
        if not (np.all(crop[:, 0] == BASE) and np.all(crop[:, -1] == BASE)):
            continue
        marker_coords = np.argwhere(crop == MARKER)
        if marker_coords.shape[0] < 1:
            continue
        if not (
            np.all(marker_coords[:, 0] > 0)
            and np.all(marker_coords[:, 0] < crop.shape[0] - 1)
            and np.all(marker_coords[:, 1] > 0)
            and np.all(marker_coords[:, 1] < crop.shape[1] - 1)
        ):
            continue
        area = int(crop.shape[0] * crop.shape[1])
        candidates.append((area, r0, c0, r1, c1))
    return candidates


def _find_target_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates = _rectangle_candidates(grid)
    assert candidates, "expected a marked rectangle candidate"
    max_area = max(area for area, *_ in candidates)
    winners = [cand for cand in candidates if cand[0] == max_area]
    assert len(winners) == 1, "expected a unique largest marked rectangle"
    _area, r0, c0, r1, c1 = winners[0]
    return int(r0), int(c0), int(r1), int(c1)


def _mirrored_marker_set(marker_coords: np.ndarray, width: int) -> set[tuple[int, int]]:
    full = {tuple(map(int, pos)) for pos in marker_coords}
    for r, c in marker_coords:
        full.add((int(r), int(width - 1 - c)))
    return full


def _expand_crosses(shape: tuple[int, int], marker_positions: set[tuple[int, int]]) -> np.ndarray:
    h, w = map(int, shape)
    out = np.full((h, w), BASE, dtype=int)
    rows = {r for r, _ in marker_positions}
    cols = {c for _, c in marker_positions}
    if rows:
        out[list(rows), :] = MARKER
    if cols:
        out[:, list(cols)] = MARKER
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror the sparse marker pixels inside the rectangle and expand all original+mirrored markers into full cross lines.

    Input structure:
    - The grid has a black background with isolated non-black noise pixels outside the main object.
    - One large filled rectangle uses yellow as its base color and contains 2-3 interior blue marker pixels.
    - Each marker lies away from the rectangle's vertical midline, so horizontal mirroring creates new marker locations.

    Rule:
    - Find the unique largest rectangular non-background component whose bounding box is fully filled and whose border is yellow.
    - Crop to that rectangle, keep the yellow base, and horizontally mirror every blue marker within the cropped rectangle's own width.
    - Take the union of original and mirrored marker positions, then paint every row and every column containing any such marker blue.

    Key insight:
    - The output is not determined by the original blue pixels alone: the horizontal mirror creates extra controlling columns, and only the combined marker set yields the symmetric row-column lattice.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    r0, c0, r1, c1 = _find_target_bbox(g)
    crop = g[r0 : r1 + 1, c0 : c1 + 1]
    marker_coords = np.argwhere(crop == MARKER)
    assert marker_coords.size > 0, "expected at least one marker"

    full_markers = _mirrored_marker_set(marker_coords, crop.shape[1])
    return _expand_crosses(crop.shape, full_markers)


def _sample_marker_positions(rng: np.random.Generator, h: int, w: int) -> list[tuple[int, int]] | None:
    rows_all = np.arange(1, h - 1)
    cols_all = [c for c in range(1, w - 1) if 2 * c != w - 1]
    if not cols_all:
        return None

    for _ in range(MARKER_ATTEMPTS):
        n_markers = int(rng.choice((2, 2, 2, 3)))
        pattern = str(rng.choice(("distinct", "shared_row", "shared_col")))

        if pattern == "shared_row":
            shared = int(rng.choice(rows_all))
            rows = [shared, shared]
            while len(rows) < n_markers:
                rows.append(int(rng.choice(rows_all)))
        elif pattern == "shared_col":
            rows = [int(rng.choice(rows_all)) for _ in range(n_markers)]
        else:
            replace = n_markers > len(rows_all)
            rows = list(map(int, rng.choice(rows_all, size=n_markers, replace=replace)))

        if pattern == "shared_col":
            shared = int(rng.choice(cols_all))
            cols = [shared, shared]
            while len(cols) < n_markers:
                cols.append(int(rng.choice(cols_all)))
        else:
            replace = n_markers > len(cols_all)
            cols = list(map(int, rng.choice(cols_all, size=n_markers, replace=replace)))

        markers = list(zip(rows, cols))
        if len(set(markers)) != n_markers:
            continue

        mirrored = {(r, w - 1 - c) for r, c in markers}
        if set(markers) & mirrored:
            continue

        orig_cols = {c for _, c in markers}
        mirror_cols = {w - 1 - c for _, c in markers}
        if mirror_cols <= orig_cols:
            continue

        if pattern == "shared_row" and len({r for r, _ in markers}) == n_markers:
            continue
        if pattern == "shared_col" and len({c for _, c in markers}) == n_markers:
            continue
        if pattern == "distinct" and (len({r for r, _ in markers}) < n_markers or len(orig_cols) < n_markers):
            continue

        return [(int(r), int(c)) for r, c in markers]

    return None


def _place_noise(
    rng: np.random.Generator,
    grid: np.ndarray,
    rect_bbox: tuple[int, int, int, int],
) -> bool:
    h, w = grid.shape
    r0, c0, r1, c1 = rect_bbox
    occupied = {tuple(map(int, pos)) for pos in np.argwhere(grid != BACKGROUND)}
    noise_count = int(rng.integers(4, 11))

    for _ in range(noise_count):
        placed = False
        for _ in range(NOISE_ATTEMPTS):
            r = int(rng.integers(0, h))
            c = int(rng.integers(0, w))
            if (r, c) in occupied:
                continue
            if r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1:
                continue
            too_close = False
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    if (rr, cc) in occupied:
                        too_close = True
                        break
                if too_close:
                    break
            if too_close:
                continue
            grid[r, c] = int(rng.choice(NOISE_COLORS))
            occupied.add((r, c))
            placed = True
            break
        if not placed:
            return False
    return True


def _build_input(
    grid_shape: tuple[int, int],
    rect_bbox: tuple[int, int, int, int],
    markers: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros(grid_shape, dtype=int)
    r0, c0, r1, c1 = rect_bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = BASE
    for r, c in markers:
        grid[r0 + r, c0 + c] = MARKER
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        r0, c0, r1, c1 = _find_target_bbox(grid)
        crop = grid[r0 : r1 + 1, c0 : c1 + 1]
        marker_coords = np.argwhere(crop == MARKER)
        if marker_coords.shape[0] not in (2, 3):
            return False
        full_markers = _mirrored_marker_set(marker_coords, crop.shape[1])
        orig_cols = {int(c) for _, c in marker_coords}
        mirror_cols = {crop.shape[1] - 1 - int(c) for _, c in marker_coords}
        if mirror_cols <= orig_cols:
            return False

        out = main(grid)
        expected = _expand_crosses(crop.shape, full_markers)
        if not np.array_equal(out, expected):
            return False
        if not np.array_equal(out, out[:, ::-1]):
            return False

        rows = {r for r, _ in full_markers}
        cols = {c for _, c in full_markers}
        blue_mask = out == MARKER
        for rr in range(out.shape[0]):
            for cc in range(out.shape[1]):
                if bool(blue_mask[rr, cc]) != (rr in rows or cc in cols):
                    return False

        winners = [cand for cand in _rectangle_candidates(grid) if cand[0] == max(a for a, *_ in _rectangle_candidates(grid))]
        if len(winners) != 1:
            return False
    except Exception:
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    rect_h = int(rng.integers(6, 11))
    rect_w = int(rng.integers(8, 13))

    grid_h = int(rng.integers(rect_h + 3, min(19, rect_h + 8)))
    grid_w = int(rng.integers(rect_w + 3, min(21, rect_w + 8)))
    if grid_h > 30 or grid_w > 30:
        return None

    top = int(rng.integers(1, grid_h - rect_h))
    left = int(rng.integers(1, grid_w - rect_w))
    bbox = (top, left, top + rect_h - 1, left + rect_w - 1)

    markers = _sample_marker_positions(rng, rect_h, rect_w)
    if markers is None:
        return None

    grid = _build_input((grid_h, grid_w), bbox, markers)
    if not _place_noise(rng, grid, bbox):
        return None
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
