import itertools
import random
from functools import lru_cache

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9
BG = 0

PATTERN_COLORS = (RED, GREEN, YELLOW, TEAL, MAROON)
GRID_SIZES = (13, 15, 17, 19)
OCCLUDER_SIZES = (3, 4)
MAX_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mirror(shape: tuple[int, int], cell: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    r, c = cell
    return h - 1 - int(r), w - 1 - int(c)


def _triangle_mask(shape: tuple[int, int], corner: str, side: int) -> np.ndarray:
    h, w = shape
    rr, cc = np.indices((h, w))
    if corner == "TL":
        return rr + cc < side
    if corner == "TR":
        return rr + (w - 1 - cc) < side
    if corner == "BL":
        return (h - 1 - rr) + cc < side
    assert corner == "BR"
    return (h - 1 - rr) + (w - 1 - cc) < side


def _nearest_corner_for_bbox(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    h, w = shape
    r0, c0, r1, c1 = bbox
    cr = (r0 + r1) / 2.0
    cc = (c0 + c1) / 2.0
    corners = {
        "TL": (0.0, 0.0),
        "TR": (0.0, w - 1.0),
        "BL": (h - 1.0, 0.0),
        "BR": (h - 1.0, w - 1.0),
    }
    scores = sorted(
        ((abs(cr - rr) + abs(cc - cc0), name) for name, (rr, cc0) in corners.items())
    )
    assert scores[0][0] < scores[1][0], "occluder corner must be unique"
    return scores[0][1]


def _rotated_bbox(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return h - 1 - r1, w - 1 - c1, h - 1 - r0, w - 1 - c0


def _paint_triangle(out: np.ndarray, corner: str, side: int, color: int) -> None:
    out[_triangle_mask(out.shape, corner, side)] = int(color)


def _normalize_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    norm = sorted((r - r0, c - c0) for r, c in cells)
    return tuple(norm)


def _rotate_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    h = max(r for r, _ in cells) + 1
    rotated = [(c, h - 1 - r) for r, c in cells]
    return _normalize_shape(tuple(rotated))


def _flip_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    w = max(c for _, c in cells) + 1
    flipped = [(r, w - 1 - c) for r, c in cells]
    return _normalize_shape(tuple(flipped))


def _shape_variants(cells: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    seen = set()
    cur = _normalize_shape(cells)
    variants = []
    for _ in range(4):
        for variant in (cur, _flip_shape(cur)):
            if variant not in seen:
                seen.add(variant)
                variants.append(variant)
        cur = _rotate_shape(cur)
    return variants


BASE_SHAPES = []
for _shape in (
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
):
    BASE_SHAPES.extend(_shape_variants(_shape))
BASE_SHAPES = list(dict.fromkeys(BASE_SHAPES))


def _neighbors8(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


@lru_cache(maxsize=None)
def _connected_local_shapes(size: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    cells = [(r, c) for r in range(size) for c in range(size)]
    out = []
    for k in (2, 3, 4):
        for subset in itertools.combinations(cells, k):
            pending = [subset[0]]
            seen = {subset[0]}
            subset_set = set(subset)
            while pending:
                r, c = pending.pop()
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (nr, nc) in subset_set and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        pending.append((nr, nc))
            if len(seen) == len(subset):
                out.append(tuple(sorted(subset)))
    return tuple(out)


def _apply_symmetric_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)
        mr, mc = _mirror(grid.shape, (r, c))
        grid[mr, mc] = int(color)


def _cells_clear_with_padding(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    require_isolated: bool,
) -> bool:
    h, w = grid.shape
    new_cells = set(cells)
    for r, c in new_cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in forbidden:
            return False
        if int(grid[r, c]) != BG:
            return False
    if not require_isolated:
        return True
    for r, c in new_cells:
        for nr, nc in _neighbors8((r, c)):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in new_cells and int(grid[nr, nc]) != BG:
                return False
    return True


def _place_shape_pair(
    grid: np.ndarray,
    shape_cells: tuple[tuple[int, int], ...],
    color: int,
    forbidden: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    max_r = max(r for r, _ in shape_cells)
    max_c = max(c for _, c in shape_cells)
    anchors = [(r0, c0) for r0 in range(h - max_r) for c0 in range(w - max_c)]
    random.shuffle(anchors)
    for r0, c0 in anchors:
        base = [(r0 + r, c0 + c) for r, c in shape_cells]
        mirrored = [_mirror(grid.shape, cell) for cell in base]
        all_cells = list(dict.fromkeys(base + mirrored))
        if any(cell == _mirror(grid.shape, cell) for cell in base):
            continue
        if _cells_clear_with_padding(grid, all_cells, forbidden, require_isolated=True):
            for r, c in all_cells:
                grid[r, c] = int(color)
            return True
    return False


def _place_marker_pair(
    grid: np.ndarray,
    forbidden: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    candidates = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(candidates)
    for cell in candidates:
        mirror = _mirror(grid.shape, cell)
        if cell >= mirror or cell == mirror:
            continue
        all_cells = [cell, mirror]
        if _chebyshev(cell, mirror) <= 1:
            continue
        if _cells_clear_with_padding(grid, all_cells, forbidden, require_isolated=True):
            grid[cell] = PINK
            grid[mirror] = PINK
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the 180-degree-symmetric pattern hidden by the blue block, count all magenta markers, and paint an orange corner triangle.

    Input structure:
    - The background is black.
    - A single solid blue rectangle occludes part of an otherwise 180-degree rotationally symmetric multicolor pattern.
    - Magenta singleton markers are part of that symmetric pattern, and some of them may be hidden under the blue rectangle.
    - The blue rectangle sits clearly in one quadrant, so its nearest grid corner is uniquely determined.

    Rule:
    - Find the blue rectangle and record both its bounding box and its nearest corner.
    - Rotate the full grid by 180 degrees, then copy the rotated counterpart patch back into the blue rectangle's bounding box to restore the hidden content.
    - Count all magenta cells in the restored grid.
    - Overwrite the solid right-triangle region at the recorded corner with orange, filling exactly the cells whose two edge distances sum to less than the magenta count.

    Key insight:
    - The blue block is doing three jobs at once: it hides pattern pixels, hides part of the magenta count, and encodes which corner receives the final triangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    blue_mask = g == BLUE
    blue_bbox = _bbox(blue_mask)
    assert blue_bbox is not None, "expected one blue occluder"
    r0, c0, r1, c1 = blue_bbox
    assert int(blue_mask.sum()) == (r1 - r0 + 1) * (c1 - c0 + 1), "blue cells must form a solid rectangle"
    assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == BLUE), "blue rectangle must be solid"

    corner = _nearest_corner_for_bbox(g.shape, blue_bbox)
    restored = g.copy()
    rotated = np.rot90(g, 2)
    restored[r0 : r1 + 1, c0 : c1 + 1] = rotated[r0 : r1 + 1, c0 : c1 + 1]

    side = int(np.count_nonzero(restored == PINK))
    assert side > 0
    out = restored.copy()
    _paint_triangle(out, corner, side, ORANGE)
    return out


def _choose_total_magenta() -> tuple[int, int, int]:
    total = random.randint(3, 6)
    center_flag = total % 2
    pair_count = total // 2
    return total, center_flag, pair_count


def _choose_occluder_bbox(
    shape: tuple[int, int], corner: str, size: int, side: int
) -> tuple[int, int, int, int]:
    h, w = shape
    tri = _triangle_mask(shape, corner, side)
    candidates = []
    for r0 in range(h - size + 1):
        for c0 in range(w - size + 1):
            r1 = r0 + size - 1
            c1 = c0 + size - 1
            if (h // 2) in range(r0, r1 + 1) and (w // 2) in range(c0, c1 + 1):
                continue
            if corner == "TL" and not (r1 < h // 2 and c1 < w // 2):
                continue
            if corner == "TR" and not (r1 < h // 2 and c0 > w // 2):
                continue
            if corner == "BL" and not (r0 > h // 2 and c1 < w // 2):
                continue
            if corner == "BR" and not (r0 > h // 2 and c0 > w // 2):
                continue
            bbox = (r0, c0, r1, c1)
            if _nearest_corner_for_bbox(shape, bbox) != corner:
                continue
            patch_tri = tri[r0 : r1 + 1, c0 : c1 + 1]
            overlap = int(patch_tri.sum())
            if not (1 <= overlap < size * size):
                continue
            candidates.append(bbox)
    assert candidates, "expected at least one valid occluder placement"
    return random.choice(candidates)


def _local_triangle_mask(
    shape: tuple[int, int], bbox: tuple[int, int, int, int], corner: str, side: int
) -> np.ndarray:
    tri = _triangle_mask(shape, corner, side)
    r0, c0, r1, c1 = bbox
    return tri[r0 : r1 + 1, c0 : c1 + 1].copy()


def _choose_hidden_layout(
    size: int, hidden_pairs: int, local_tri: np.ndarray
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    cells = [(r, c) for r in range(size) for c in range(size)]
    marker_combos = [
        combo
        for combo in itertools.combinations(cells, hidden_pairs)
        if all(_chebyshev(a, b) > 1 for a, b in itertools.combinations(combo, 2))
    ]
    random.shuffle(marker_combos)
    shapes = list(_connected_local_shapes(size))
    random.shuffle(shapes)
    for marker_combo in marker_combos:
        forbidden = set(marker_combo)
        for marker in marker_combo:
            forbidden.update(_neighbors8(marker))
        for shape in shapes:
            shape_set = set(shape)
            if shape_set & set(marker_combo):
                continue
            if shape_set & forbidden:
                continue
            tri_hits = [bool(local_tri[r, c]) for r, c in shape]
            if not any(tri_hits) or all(tri_hits):
                continue
            return tuple(sorted(marker_combo)), tuple(sorted(shape))
    raise RuntimeError("failed to find a hidden layout inside the occluder")


def _forbidden_from_bboxes(bboxes: list[tuple[int, int, int, int]]) -> set[tuple[int, int]]:
    cells = set()
    for r0, c0, r1, c1 in bboxes:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                cells.add((r, c))
    return cells


def _paint_hidden_content(
    full: np.ndarray,
    bbox: tuple[int, int, int, int],
    markers_local: tuple[tuple[int, int], ...],
    shape_local: tuple[tuple[int, int], ...],
) -> None:
    r0, c0, _r1, _c1 = bbox
    color = random.choice(PATTERN_COLORS)
    for dr, dc in shape_local:
        rr, cc = r0 + dr, c0 + dc
        full[rr, cc] = color
        mr, mc = _mirror(full.shape, (rr, cc))
        full[mr, mc] = color
    for dr, dc in markers_local:
        rr, cc = r0 + dr, c0 + dc
        full[rr, cc] = PINK
        mr, mc = _mirror(full.shape, (rr, cc))
        full[mr, mc] = PINK


def _extra_shape_count() -> int:
    return random.randint(2, 5)


def _build_full_pattern() -> tuple[np.ndarray, tuple[int, int, int, int], str, int]:
    h = random.choice(GRID_SIZES)
    w = random.choice(GRID_SIZES)
    side, center_flag, pair_count = _choose_total_magenta()
    corner = random.choice(("TL", "TR", "BL", "BR"))
    size = random.choice(OCCLUDER_SIZES)
    hidden_cap = 2 if size == 3 else 3
    hidden_pairs = random.randint(1, min(pair_count, hidden_cap))
    blue_bbox = _choose_occluder_bbox((h, w), corner, size, side)
    sym_bbox = _rotated_bbox((h, w), blue_bbox)
    local_tri = _local_triangle_mask((h, w), blue_bbox, corner, side)
    markers_local, shape_local = _choose_hidden_layout(size, hidden_pairs, local_tri)

    full = np.zeros((h, w), dtype=int)
    _paint_hidden_content(full, blue_bbox, markers_local, shape_local)

    forbidden = _forbidden_from_bboxes([blue_bbox, sym_bbox])
    remaining_pairs = pair_count - hidden_pairs
    for _ in range(remaining_pairs):
        if not _place_marker_pair(full, forbidden):
            raise RuntimeError("failed to place a visible marker pair")

    center = (h // 2, w // 2)
    if center_flag:
        if int(full[center]) != BG:
            raise RuntimeError("center marker collided with existing pattern")
        if any(
            0 <= nr < h and 0 <= nc < w and int(full[nr, nc]) != BG
            for nr, nc in _neighbors8(center)
        ):
            raise RuntimeError("center marker must remain isolated")
        full[center] = PINK

    for _ in range(_extra_shape_count()):
        shape = random.choice(BASE_SHAPES)
        color = random.choice(PATTERN_COLORS)
        if not _place_shape_pair(full, shape, color, forbidden):
            raise RuntimeError("failed to place an extra symmetric shape")

    assert np.array_equal(full, np.rot90(full, 2)), "full pattern must be 180-degree symmetric"
    return full, blue_bbox, corner, side


def _manual_output(full: np.ndarray, corner: str, side: int) -> np.ndarray:
    out = full.copy()
    _paint_triangle(out, corner, side, ORANGE)
    return out


def _semantic_checks(
    inp: np.ndarray,
    full: np.ndarray,
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    corner: str,
    side: int,
) -> None:
    r0, c0, r1, c1 = bbox
    center = (inp.shape[0] // 2, inp.shape[1] // 2)
    assert center[0] not in range(r0, r1 + 1) or center[1] not in range(c0, c1 + 1)
    assert _nearest_corner_for_bbox(inp.shape, bbox) == corner
    assert np.array_equal(main(inp), out)

    visible_magenta = int(np.count_nonzero(inp == PINK))
    hidden_magenta = side - visible_magenta
    assert 3 <= side <= 6
    assert hidden_magenta > 0, "count must really depend on restoration"

    tri = _triangle_mask(inp.shape, corner, side)
    bbox_tri = tri[r0 : r1 + 1, c0 : c1 + 1]
    restored_patch = full[r0 : r1 + 1, c0 : c1 + 1]
    assert np.any((restored_patch != BG) & bbox_tri), "triangle must overwrite some restored content"
    assert np.any((restored_patch != BG) & (~bbox_tri)), "some restored content must remain visible"
    assert np.any((restored_patch != BG) & (restored_patch != PINK)), "occluder must hide non-marker pattern too"

    rotated = np.rot90(inp, 2)
    assert np.array_equal(restored_patch, rotated[r0 : r1 + 1, c0 : c1 + 1])

    wrong_visible_count = _manual_output(full, corner, visible_magenta)
    assert not np.array_equal(wrong_visible_count, out), "visible-only counting must fail"

    wrong_no_restore = inp.copy()
    _paint_triangle(wrong_no_restore, corner, side, ORANGE)
    assert not np.array_equal(wrong_no_restore, out), "restoring non-blue pattern must matter"

    assert np.any(inp == BLUE)
    assert np.any(out == ORANGE)
    assert not np.array_equal(inp, out)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            full, blue_bbox, corner, side = _build_full_pattern()
        except RuntimeError:
            continue

        inp = full.copy()
        r0, c0, r1, c1 = blue_bbox
        inp[r0 : r1 + 1, c0 : c1 + 1] = BLUE
        out = _manual_output(full, corner, side)

        try:
            _semantic_checks(inp, full, out, blue_bbox, corner, side)
        except AssertionError:
            continue
        return inp
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
