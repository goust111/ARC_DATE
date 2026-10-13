import random
from collections import Counter, deque

import numpy as np


BG = 5
BLUE = 1
RED = 2
GREEN = 3
WINDOW = 7
INTERNAL_BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_RNG = random.Random()


def _get_bg(grid: np.ndarray) -> int:
    return Counter(map(int, grid.ravel())).most_common(1)[0][0]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    parts = []
    h, w = mask.shape
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
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _shape_bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _render_coords(coords: set[tuple[int, int]], color: int) -> np.ndarray:
    r0, c0, r1, c1 = _shape_bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in coords:
        out[r - r0, c - c0] = color
    return out


def _is_connected(coords: set[tuple[int, int]]) -> bool:
    if not coords:
        return False
    start = next(iter(coords))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nb = (r + dr, c + dc)
            if nb in coords and nb not in seen:
                seen.add(nb)
                q.append(nb)
    return len(seen) == len(coords)


def _is_irregular(coords: set[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _shape_bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return h >= 2 and w >= 2 and len(coords) < h * w


def _mask_contacts_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            neighbors = 0
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                    neighbors += 1
            if neighbors >= 2:
                return True
    return False


def _piece_patch(coords: set[tuple[int, int]], color: int, *, marker: tuple[int, int] | None = None) -> np.ndarray:
    patch = np.full(_render_coords(coords, color).shape, BG, dtype=int)
    r0, c0, _, _ = _shape_bbox(coords)
    for r, c in coords:
        patch[r - r0, c - c0] = color
    if marker is not None:
        patch[marker[0] - r0, marker[1] - c0] = GREEN
    return patch


def _to_internal(patch: np.ndarray) -> np.ndarray:
    out = patch.copy()
    out[out == BG] = INTERNAL_BG
    return out


def _pack_pieces(
    blue: np.ndarray, red: np.ndarray
) -> tuple[np.ndarray, dict]:
    blue_occ = [tuple(map(int, rc)) for rc in np.argwhere(blue != INTERNAL_BG)]
    red_occ = [tuple(map(int, rc)) for rc in np.argwhere(red != INTERNAL_BG)]
    blue_set = set(blue_occ)
    candidates = []

    for dr in range(-red.shape[0], blue.shape[0] + 1):
        for dc in range(-red.shape[1], blue.shape[1] + 1):
            shifted_red = [(r + dr, c + dc) for r, c in red_occ]
            if any(cell in blue_set for cell in shifted_red):
                continue

            all_rows = [r for r, _ in blue_occ] + [r for r, _ in shifted_red]
            all_cols = [c for _, c in blue_occ] + [c for _, c in shifted_red]
            min_r, max_r = min(all_rows), max(all_rows)
            min_c, max_c = min(all_cols), max(all_cols)
            h = max_r - min_r + 1
            w = max_c - min_c + 1
            blue_anchor = (-min_r, -min_c)
            red_anchor = (dr - min_r, dc - min_c)

            packed = np.zeros((h, w), dtype=int)
            for r, c in blue_occ:
                packed[r - min_r, c - min_c] = int(blue[r, c])
            for r, c in red_occ:
                packed[r + dr - min_r, c + dc - min_c] = int(red[r, c])

            candidates.append(
                {
                    "key": (h * w, w, blue_anchor[0], blue_anchor[1]),
                    "blue_anchor": blue_anchor,
                    "red_anchor": red_anchor,
                    "dr": dr,
                    "dc": dc,
                    "packed": packed,
                }
            )

    assert candidates, "expected at least one non-overlapping packing"
    best_key = min(item["key"] for item in candidates)
    best = [item for item in candidates if item["key"] == best_key]

    packed_signatures = {
        (tuple(item["packed"].shape), item["packed"].tobytes()): item for item in best
    }
    assert packed_signatures, "best packing set must be non-empty"
    chosen = min(best, key=lambda item: (item["dr"], item["dc"]))
    return chosen["packed"], chosen


def _extract_regions(
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    comps = _components(grid != BG)
    assert len(comps) == 3, "expected exactly three separated non-background regions"

    blue_patch = red_patch = window_patch = None
    for comp in comps:
        rows = comp[:, 0]
        cols = comp[:, 1]
        box = (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))
        patch = _crop_bbox(grid, box)
        colors = set(map(int, grid[rows, cols]))
        if colors <= {BLUE, GREEN}:
            assert blue_patch is None, "expected one blue region"
            assert np.count_nonzero(patch == GREEN) == 1
            blue_patch = patch
        elif colors == {RED}:
            assert red_patch is None, "expected one red region"
            red_patch = patch
        elif colors <= {WINDOW, GREEN}:
            assert window_patch is None, "expected one window region"
            assert np.count_nonzero(patch == GREEN) == 1
            assert np.all(np.isin(patch, [BG, WINDOW, GREEN]))
            assert patch.size == np.count_nonzero(patch == WINDOW) + np.count_nonzero(patch == GREEN)
            window_patch = patch
        else:
            raise AssertionError("unexpected region colors")

    assert blue_patch is not None and red_patch is not None and window_patch is not None
    return blue_patch, red_patch, window_patch


def main(grid: np.ndarray) -> np.ndarray:
    """Pack the marked blue piece with the red piece, then paste that packed sprite into the marked orange window by aligning the shared green marker.

    Input structure:
    - The grid has a dominant gray background and three separated regions: a blue piece with one embedded green marker, a red piece, and an orange rectangular window with one embedded green marker.
    - The green marker belongs to the blue piece geometrically, while the window is otherwise solid orange.
    - The blue and red pieces are translation-only puzzle pieces: they may move but may not rotate or reflect.

    Rule:
    - Crop the blue+green piece and the red piece to their own bounding boxes.
    - Try every relative translation that keeps the two cropped pieces non-overlapping, and choose the placement whose combined bounding box area is smallest; break ties by smaller width, then by the placement where the blue piece sits farther toward the upper-left within that combined box.
    - Find the green marker inside that packed result and align it to the green marker inside the orange window.
    - Paste the packed result into the window so blue, red, and green overwrite orange, and leave all untouched window cells orange.
    - Output only the cropped window.

    Key insight:
    - The pieces must first be packed to determine their relative geometry, and that packing changes the packed marker offset; only then can the shared green marker determine the absolute paste position inside the target window.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert _get_bg(g) == BG

    blue_patch, red_patch, window = _extract_regions(g)
    assert np.count_nonzero(blue_patch == GREEN) == 1
    assert np.all(np.isin(blue_patch, [BG, BLUE, GREEN]))
    assert np.all(np.isin(red_patch, [BG, RED]))
    assert len(_components(np.isin(blue_patch, [BLUE, GREEN]))) == 1
    assert len(_components(red_patch == RED)) == 1

    packed, _meta = _pack_pieces(_to_internal(blue_patch), _to_internal(red_patch))
    src_marker = tuple(map(int, np.argwhere(packed == GREEN)[0]))
    dst_marker = tuple(map(int, np.argwhere(window == GREEN)[0]))

    out = window.copy()
    dr = dst_marker[0] - src_marker[0]
    dc = dst_marker[1] - src_marker[1]
    for r, c in np.argwhere(packed != INTERNAL_BG):
        rr = int(r) + dr
        cc = int(c) + dc
        assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1], "packed sprite must fit inside the window"
        out[rr, cc] = int(packed[r, c])

    return out


def _random_connected_cells(size: int, max_h: int, max_w: int) -> set[tuple[int, int]] | None:
    cells = {(0, 0)}
    for _ in range(size - 1):
        frontier = set()
        for r, c in cells:
            for dr, dc in DIRS4:
                nb = (r + dr, c + dc)
                if nb not in cells:
                    frontier.add(nb)
        if not frontier:
            return None
        cells.add(_RNG.choice(tuple(frontier)))
        r0, c0, r1, c1 = _shape_bbox(cells)
        if r1 - r0 + 1 > max_h or c1 - c0 + 1 > max_w:
            return None
    r0, c0, _, _ = _shape_bbox(cells)
    return {(r - r0, c - c0) for r, c in cells}


def _partition_union(
    union_cells: set[tuple[int, int]], blue_size: int
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    cells = list(union_cells)
    for _ in range(300):
        blue = {_RNG.choice(cells)}
        while len(blue) < blue_size:
            frontier = []
            for r, c in blue:
                for dr, dc in DIRS4:
                    nb = (r + dr, c + dc)
                    if nb in union_cells and nb not in blue:
                        frontier.append(nb)
            if not frontier:
                break
            blue.add(_RNG.choice(frontier))
        if len(blue) != blue_size:
            continue
        red = union_cells - blue
        if not _is_connected(blue) or not _is_connected(red):
            continue
        if not _is_irregular(blue) or not _is_irregular(red):
            continue
        if not (_mask_contacts_hole(_render_coords(blue, 1) == 1) or _mask_contacts_hole(_render_coords(red, 1) == 1)):
            continue
        return blue, red
    return None


def _generate_piece_pair() -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(500):
        total = _RNG.randint(9, 14)
        blue_size = _RNG.randint(4, min(8, total - 4))
        red_size = total - blue_size
        if red_size > 8:
            continue
        union = _random_connected_cells(total, max_h=6, max_w=6)
        if union is None:
            continue
        parts = _partition_union(union, blue_size)
        if parts is None:
            continue
        blue_cells, red_cells = parts
        marker = _RNG.choice(tuple(blue_cells))
        blue_patch = _piece_patch(blue_cells, BLUE, marker=marker)
        red_patch = _piece_patch(red_cells, RED)

        packed, meta = _pack_pieces(_to_internal(blue_patch), _to_internal(red_patch))
        if packed.shape[0] > 9 or packed.shape[1] > 9:
            continue
        if meta["blue_anchor"] == (0, 0):
            continue
        src_marker = tuple(map(int, np.argwhere(_to_internal(blue_patch) == GREEN)[0]))
        packed_marker = tuple(map(int, np.argwhere(packed == GREEN)[0]))
        if src_marker == packed_marker:
            continue
        if not _packing_beats_bbox_touch(_to_internal(blue_patch), _to_internal(red_patch), meta["key"][0]):
            continue
        return blue_patch, red_patch
    return None


def _packing_beats_bbox_touch(blue: np.ndarray, red: np.ndarray, best_area: int) -> bool:
    naive_area = None
    hb, wb = blue.shape
    hr, wr = red.shape
    candidates = []
    for dc in range(-wr + 1, wb):
        candidates.append((-hr, dc))
        candidates.append((hb, dc))
    for dr in range(-hr + 1, hb):
        candidates.append((dr, -wr))
        candidates.append((dr, wb))

    blue_set = {tuple(map(int, rc)) for rc in np.argwhere(blue != INTERNAL_BG)}
    red_occ = [tuple(map(int, rc)) for rc in np.argwhere(red != INTERNAL_BG)]
    for dr, dc in candidates:
        shifted_red = [(r + dr, c + dc) for r, c in red_occ]
        if any(cell in blue_set for cell in shifted_red):
            continue
        rows = [r for r, _ in blue_set] + [r for r, _ in shifted_red]
        cols = [c for _, c in blue_set] + [c for _, c in shifted_red]
        area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
        naive_area = area if naive_area is None else min(naive_area, area)
    return naive_area is not None and best_area < naive_area


def _place_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    placed_boxes: list[tuple[int, int, int, int]],
    padding: int,
) -> tuple[int, int] | None:
    h, w = patch.shape
    candidates = []
    for r0 in range(grid.shape[0] - h + 1):
        for c0 in range(grid.shape[1] - w + 1):
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            ok = True
            for ar0, ac0, ar1, ac1 in placed_boxes:
                if not (
                    box[2] + padding < ar0
                    or ar1 + padding < box[0]
                    or box[3] + padding < ac0
                    or ac1 + padding < box[1]
                ):
                    ok = False
                    break
            if ok:
                candidates.append((r0, c0))
    if not candidates:
        return None
    return _RNG.choice(candidates)


def _draw_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    view = grid[top : top + h, left : left + w]
    mask = patch != BG
    view[mask] = patch[mask]


def _build_case() -> np.ndarray | None:
    pair = _generate_piece_pair()
    if pair is None:
        return None
    blue_patch, red_patch = pair
    packed, _meta = _pack_pieces(_to_internal(blue_patch), _to_internal(red_patch))
    packed_marker = tuple(map(int, np.argwhere(packed == GREEN)[0]))

    win_h = _RNG.randint(max(7, packed.shape[0]), 9)
    win_w = _RNG.randint(max(7, packed.shape[1]), 9)
    top_r = _RNG.randint(0, win_h - packed.shape[0])
    top_c = _RNG.randint(0, win_w - packed.shape[1])
    dst_marker = (top_r + packed_marker[0], top_c + packed_marker[1])
    window_patch = np.full((win_h, win_w), WINDOW, dtype=int)
    window_patch[dst_marker] = GREEN

    grid_h = _RNG.randint(18, 24)
    grid_w = _RNG.randint(18, 24)
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    placed = []
    for patch in _RNG.sample([blue_patch, red_patch, window_patch], 3):
        loc = _place_patch(grid, patch, placed, padding=1)
        if loc is None:
            return None
        _draw_patch(grid, patch, loc[0], loc[1])
        placed.append((loc[0], loc[1], loc[0] + patch.shape[0] - 1, loc[1] + patch.shape[1] - 1))

    if not _semantic_checks(grid):
        return None
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        if g.ndim != 2 or max(g.shape) > 30:
            return False
        if _get_bg(g) != BG:
            return False
        if np.count_nonzero(g == GREEN) != 2:
            return False

        blue_patch, red_patch, window = _extract_regions(g)
        if len(_components(np.isin(blue_patch, [BLUE, GREEN]))) != 1:
            return False
        if len(_components(red_patch == RED)) != 1:
            return False

        packed, meta = _pack_pieces(_to_internal(blue_patch), _to_internal(red_patch))
        if packed.shape[0] > window.shape[0] or packed.shape[1] > window.shape[1]:
            return False
        if meta["blue_anchor"] == (0, 0):
            return False

        blue_marker = tuple(map(int, np.argwhere(_to_internal(blue_patch) == GREEN)[0]))
        packed_marker = tuple(map(int, np.argwhere(packed == GREEN)[0]))
        if blue_marker == packed_marker:
            return False
        if not _packing_beats_bbox_touch(_to_internal(blue_patch), _to_internal(red_patch), meta["key"][0]):
            return False

        expected = main(g)
        if expected.shape != window.shape:
            return False
        if not np.any(expected == BLUE) or not np.any(expected == RED):
            return False
        if np.count_nonzero(expected == GREEN) != 1:
            return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(800):
        case = _build_case()
        if case is not None:
            return case
    raise RuntimeError("failed to generate a valid puzzle instance")
