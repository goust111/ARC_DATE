from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
BRACKET_ORIENTATIONS = ("tl", "tr", "bl", "br")
GEN_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).flatten())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(np.asarray(grid, dtype=int))))
    if exclude:
        colors -= set(exclude)
    return sorted(colors)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    if connectivity == 4:
        nbrs = DIRS4
    else:
        nbrs = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))

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
                for dr, dc in nbrs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _rect_cells(rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, h, w = rect
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _rect_intersection(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int]:
    ar0, ac0, ah, aw = a
    br0, bc0, bh, bw = b
    r0 = max(ar0, br0)
    c0 = max(ac0, bc0)
    r1 = min(ar0 + ah, br0 + bh)
    c1 = min(ac0 + aw, bc0 + bw)
    return max(0, r1 - r0), max(0, c1 - c0)


def _place_related_rect(
    base: tuple[int, int, int, int],
    new_h: int,
    new_w: int,
    corner: str,
    overlap_h: int,
    overlap_w: int,
) -> tuple[int, int, int, int]:
    r0, c0, h, w = base
    if corner == "tl":
        nr = r0 - new_h + overlap_h
        nc = c0 - new_w + overlap_w
    elif corner == "tr":
        nr = r0 - new_h + overlap_h
        nc = c0 + w - overlap_w
    elif corner == "bl":
        nr = r0 + h - overlap_h
        nc = c0 - new_w + overlap_w
    elif corner == "br":
        nr = r0 + h - overlap_h
        nc = c0 + w - overlap_w
    else:
        raise ValueError(f"unknown corner {corner}")
    return nr, nc, new_h, new_w


def _bracket_cells(
    top: int,
    left: int,
    height: int,
    width: int,
    orientation: str,
) -> set[tuple[int, int]]:
    if orientation == "tl":
        return {(top, c) for c in range(left, left + width)} | {(r, left) for r in range(top, top + height)}
    if orientation == "tr":
        return {(top, c) for c in range(left, left + width)} | {(r, left + width - 1) for r in range(top, top + height)}
    if orientation == "bl":
        return {(top + height - 1, c) for c in range(left, left + width)} | {(r, left) for r in range(top, top + height)}
    if orientation == "br":
        return {(top + height - 1, c) for c in range(left, left + width)} | {
            (r, left + width - 1) for r in range(top, top + height)
        }
    raise ValueError(f"unknown orientation {orientation}")


def _is_corner_bracket(coords: np.ndarray) -> bool:
    if coords.shape[0] < 3:
        return False
    r0, c0, r1, c1 = bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 2 or w < 2 or coords.shape[0] != h + w - 1:
        return False
    coord_set = {tuple(map(int, p)) for p in coords}
    for orientation in BRACKET_ORIENTATIONS:
        if coord_set == _bracket_cells(r0, c0, h, w, orientation):
            return True
    return False


def _find_bracket(grid: np.ndarray) -> tuple[set[tuple[int, int]], tuple[int, int, int, int], tuple[int, int], int]:
    grey_components = components_mask(grid == GREY, connectivity=4)
    matches = []
    for comp in grey_components:
        if not _is_corner_bracket(comp):
            continue
        r0, c0, r1, c1 = bbox_coords(comp)
        comp_set = {tuple(map(int, p)) for p in comp}
        markers = [
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if (r, c) not in comp_set and grid[r, c] not in (BLACK, GREY)
        ]
        if len(markers) != 1:
            continue
        matches.append((comp_set, (r0, c0, r1, c1), markers[0], int(grid[markers[0]])))
    assert len(matches) == 1, "expected exactly one valid grey corner bracket"
    return matches[0]


def _color_bboxes(
    grid: np.ndarray,
    ignored_coords: set[tuple[int, int]] | None = None,
) -> dict[int, tuple[int, int, int, int]]:
    boxes: dict[int, tuple[int, int, int, int]] = {}
    for color in colors_present(grid, exclude={BLACK, GREY}):
        coords = np.argwhere(grid == color)
        if ignored_coords:
            coords = np.array(
                [p for p in coords.tolist() if tuple(map(int, p)) not in ignored_coords],
                dtype=int,
            )
        assert coords.size > 0
        boxes[color] = bbox_coords(coords)
    return boxes


def _complete_shapes(
    grid: np.ndarray,
    ignored_coords: set[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, dict[int, tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    boxes = _color_bboxes(g, ignored_coords=ignored_coords)
    out = g.copy()
    for color in sorted(boxes):
        r0, c0, r1, c1 = boxes[color]
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(region == BLACK) or np.any(region == GREY):
            continue
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out, boxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the occluded rectangles, use the bracketed marker color to choose one source rectangle, and stamp it onto each grey target pixel.

    Input structure:
    - Black background contains 3-4 colored foreground shapes; each color belongs to one axis-aligned solid rectangle, though overlaps make some colors look incomplete.
    - A single grey 4-connected corner bracket sits near a grid corner, and exactly one non-black non-grey marker pixel lies inside its bbox.
    - Additional grey pixels appear as isolated single-cell markers away from the bracket.

    Rule:
    - For each non-black, non-grey color, compute the bbox of its visible pixels; when that bbox contains no black or grey cells, complete the implied solid rectangle by filling the whole bbox with that color, processing colors in ascending numeric order.
    - Read the color of the unique marker inside the grey bracket, then crop that color's completed bbox as the stamp source; its bbox dimensions are odd, so it has a unique center.
    - For every grey marker pixel not belonging to the bracket component, paste the solid source rectangle centered on that grey cell; keep the bracket and its internal marker unchanged.

    Key insight:
    - The chosen stamp is not directly visible in the input: first reconstruct the hidden rectangle footprints from foreground-only bboxes, then let the bracket's enclosed marker select which reconstructed rectangle to copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bracket_cells, _bbox, inner_marker, selected_color = _find_bracket(g)
    completed, boxes = _complete_shapes(g, ignored_coords={inner_marker})
    assert selected_color in boxes, "bracket marker must name an existing foreground color"
    r0, c0, r1, c1 = boxes[selected_color]
    patch = completed[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert patch.shape[0] % 2 == 1 and patch.shape[1] % 2 == 1
    assert np.all(patch == selected_color), "selected completed rectangle must be solid"

    out = completed.copy()
    grey_components = components_mask(g == GREY, connectivity=4)
    for comp in grey_components:
        comp_set = {tuple(map(int, p)) for p in comp}
        if comp_set == bracket_cells:
            continue
        assert comp.shape[0] == 1, "non-bracket grey markers must be isolated pixels"
        cr, cc = map(int, comp[0])
        pr = cr - patch.shape[0] // 2
        pc = cc - patch.shape[1] // 2
        assert 0 <= pr and 0 <= pc
        assert pr + patch.shape[0] <= out.shape[0] and pc + patch.shape[1] <= out.shape[1]
        paste(out, patch, pr, pc)

    for r, c in bracket_cells:
        out[r, c] = GREY
    out[inner_marker] = selected_color
    return out


def _paint_rectangles(
    shape: tuple[int, int],
    rectangles: list[tuple[int, int, int, int]],
    colors_in_paint_order: list[int],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for rect, color in zip(rectangles, colors_in_paint_order, strict=True):
        r0, c0, h, w = rect
        grid[r0 : r0 + h, c0 : c0 + w] = int(color)
    return grid


def _choose_bracket(
    h: int,
    w: int,
) -> tuple[set[tuple[int, int]], tuple[int, int], tuple[int, int, int, int], str]:
    corner = random.choice(BRACKET_ORIENTATIONS)
    bh = random.choice((2, 3))
    bw = random.choice((2, 3))
    if corner == "tl":
        top = random.randint(0, 1)
        left = random.randint(0, 1)
    elif corner == "tr":
        top = random.randint(0, 1)
        left = w - bw - random.randint(0, 1)
    elif corner == "bl":
        top = h - bh - random.randint(0, 1)
        left = random.randint(0, 1)
    else:
        top = h - bh - random.randint(0, 1)
        left = w - bw - random.randint(0, 1)
    cells = _bracket_cells(top, left, bh, bw, corner)
    interior = [(r, c) for r in range(top, top + bh) for c in range(left, left + bw) if (r, c) not in cells]
    assert interior, "corner bracket must enclose at least one cell"
    marker = random.choice(interior)
    return cells, marker, (top, left, top + bh - 1, left + bw - 1), corner


def _sample_chain_rectangles(
    grid_h: int,
    grid_w: int,
    count: int,
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int, int, int]] | None:
    selected_h = random.choice((3, 5, 7))
    selected_w = random.choice((3, 5, 7))
    if selected_h >= grid_h - 4 or selected_w >= grid_w - 4:
        return None

    base_r_min = max(2, grid_h // 3 - 2)
    base_r_max = min(grid_h - selected_h - 3, grid_h // 2 + 1)
    base_c_min = max(2, grid_w // 3 - 3)
    base_c_max = min(grid_w - selected_w - 3, grid_w // 2 + 2)
    if base_r_min > base_r_max or base_c_min > base_c_max:
        return None

    for _ in range(80):
        base = (
            random.randint(base_r_min, base_r_max),
            random.randint(base_c_min, base_c_max),
            selected_h,
            selected_w,
        )
        rects = [base]
        if _rect_cells(base) & forbidden:
            continue

        ok = True
        for _idx in range(1, count):
            prev = rects[-1]
            new_h = random.randint(3, 7)
            new_w = random.randint(3, 7)
            max_oh = min(3, prev[2] - 1, new_h - 1)
            max_ow = min(3, prev[3] - 1, new_w - 1)
            if max_oh < 1 or max_ow < 1:
                ok = False
                break

            placed = None
            for _ in range(40):
                overlap_h = random.randint(1, max_oh)
                overlap_w = random.randint(1, max_ow)
                corner = random.choice(BRACKET_ORIENTATIONS)
                candidate = _place_related_rect(prev, new_h, new_w, corner, overlap_h, overlap_w)
                r0, c0, h0, w0 = candidate
                if r0 < 0 or c0 < 0 or r0 + h0 > grid_h or c0 + w0 > grid_w:
                    continue
                cells = _rect_cells(candidate)
                if cells & forbidden:
                    continue
                if not (_rect_intersection(prev, candidate)[0] > 0 and _rect_intersection(prev, candidate)[1] > 0):
                    continue
                bad = False
                for earlier in rects[:-1]:
                    if _rect_intersection(earlier, candidate) != (0, 0):
                        bad = True
                        break
                if bad:
                    continue
                placed = candidate
                break
            if placed is None:
                ok = False
                break
            rects.append(placed)
        if ok:
            return rects
    return None


def _choose_colors(count: int) -> tuple[list[int], int] | None:
    colors = random.sample(PALETTE, count)
    selected = random.choice(colors)
    lower = [c for c in colors if c < selected]
    if not lower:
        return None
    first_occluder = random.choice(lower)
    remaining = [c for c in colors if c not in {selected, first_occluder}]
    random.shuffle(remaining)
    return [selected, first_occluder, *remaining], selected


def _choose_marker_centers(
    base_grid: np.ndarray,
    patch_shape: tuple[int, int],
    count: int,
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    h, w = base_grid.shape
    ph, pw = patch_shape
    candidates = [
        (r, c)
        for r in range(ph // 2, h - ph // 2)
        for c in range(pw // 2, w - pw // 2)
    ]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    windows: list[tuple[int, int, int, int]] = []
    for r, c in candidates:
        r0 = r - ph // 2
        c0 = c - pw // 2
        r1 = r0 + ph
        c1 = c0 + pw
        if np.any(base_grid[r0:r1, c0:c1] != BLACK):
            continue
        window_cells = {(rr, cc) for rr in range(r0, r1) for cc in range(c0, c1)}
        if window_cells & forbidden:
            continue
        touches_existing = False
        for wr0, wc0, wr1, wc1 in windows:
            if not (r1 <= wr0 or wr1 <= r0 or c1 <= wc0 or wc1 <= c0):
                touches_existing = True
                break
        if touches_existing:
            continue
        chosen.append((r, c))
        windows.append((r0, c0, r1, c1))
        if len(chosen) == count:
            return chosen
    return None


def _state_expected_output(
    inp: np.ndarray,
    rectangles: list[tuple[int, int, int, int]],
    colors_in_paint_order: list[int],
    selected_color: int,
    bracket_cells: set[tuple[int, int]],
    bracket_marker: tuple[int, int],
    targets: list[tuple[int, int]],
) -> np.ndarray:
    out = inp.copy()
    for color in sorted(colors_in_paint_order):
        rect = rectangles[colors_in_paint_order.index(color)]
        r0, c0, h, w = rect
        out[r0 : r0 + h, c0 : c0 + w] = int(color)

    sel_rect = rectangles[colors_in_paint_order.index(selected_color)]
    patch = np.full((sel_rect[2], sel_rect[3]), selected_color, dtype=int)
    for cr, cc in targets:
        paste(out, patch, cr - patch.shape[0] // 2, cc - patch.shape[1] // 2)

    for r, c in bracket_cells:
        out[r, c] = GREY
    out[bracket_marker] = selected_color
    return out


def _validate_instance(
    inp: np.ndarray,
    rectangles: list[tuple[int, int, int, int]],
    colors_in_paint_order: list[int],
    selected_color: int,
    bracket_cells: set[tuple[int, int]],
    bracket_marker: tuple[int, int],
    targets: list[tuple[int, int]],
) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if len(colors_in_paint_order) not in (3, 4):
        return False
    if len(targets) not in (2, 3):
        return False
    if len(set(colors_in_paint_order)) != len(colors_in_paint_order):
        return False
    if selected_color not in colors_in_paint_order:
        return False

    boxes = _color_bboxes(inp, ignored_coords={bracket_marker})
    for rect, color in zip(rectangles, colors_in_paint_order, strict=True):
        r0, c0, h, w = rect
        if boxes.get(color) != (r0, c0, r0 + h - 1, c0 + w - 1):
            return False
        region = inp[r0 : r0 + h, c0 : c0 + w]
        if np.any(region == BLACK) or np.any(region == GREY):
            return False

    selected_idx = colors_in_paint_order.index(selected_color)
    if selected_idx != 0:
        return False
    if rectangles[0][2] % 2 != 1 or rectangles[0][3] % 2 != 1:
        return False

    grey_components = components_mask(inp == GREY, connectivity=4)
    if len(grey_components) != len(targets) + 1:
        return False
    try:
        found_bracket, _, found_marker, found_color = _find_bracket(inp)
    except AssertionError:
        return False
    if found_bracket != bracket_cells or found_marker != bracket_marker or found_color != selected_color:
        return False

    expected = _state_expected_output(
        inp,
        rectangles,
        colors_in_paint_order,
        selected_color,
        bracket_cells,
        bracket_marker,
        targets,
    )
    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _generate_attempt() -> np.ndarray | None:
    grid_h = random.randint(18, 24)
    grid_w = random.randint(18, 24)
    shape_count = random.choice((3, 4))
    marker_count = random.choice((2, 3))

    color_choice = _choose_colors(shape_count)
    if color_choice is None:
        return None
    colors_in_paint_order, selected_color = color_choice

    bracket_cells, bracket_marker, bracket_bbox, _corner = _choose_bracket(grid_h, grid_w)
    br0, bc0, br1, bc1 = bracket_bbox
    forbidden = {
        (r, c)
        for r in range(max(0, br0 - 1), min(grid_h, br1 + 2))
        for c in range(max(0, bc0 - 1), min(grid_w, bc1 + 2))
    }

    rectangles = _sample_chain_rectangles(grid_h, grid_w, shape_count, forbidden)
    if rectangles is None:
        return None

    base = _paint_rectangles((grid_h, grid_w), rectangles, colors_in_paint_order)
    if bracket_cells:
        brs, bcs = zip(*sorted(bracket_cells), strict=True)
        if np.any(base[np.array(brs), np.array(bcs)] != BLACK):
            return None
    inp = base.copy()
    for r, c in bracket_cells:
        inp[r, c] = GREY
    inp[bracket_marker] = selected_color

    sel_rect = rectangles[0]
    targets = _choose_marker_centers(inp, (sel_rect[2], sel_rect[3]), marker_count, forbidden)
    if targets is None:
        return None
    for r, c in targets:
        inp[r, c] = GREY

    if not _validate_instance(inp, rectangles, colors_in_paint_order, selected_color, bracket_cells, bracket_marker, targets):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
