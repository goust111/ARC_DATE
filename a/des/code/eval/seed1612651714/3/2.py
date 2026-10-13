from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _translate(coords: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in coords]


def _components_of_color(
    grid: np.ndarray,
    color: int,
    ignored: set[tuple[int, int]] | None = None,
) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    ignored = ignored or set()
    visited: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if (r, c) in visited or (r, c) in ignored or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            visited.add((r, c))
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and (nr, nc) not in visited
                        and (nr, nc) not in ignored
                        and int(grid[nr, nc]) == int(color)
                    ):
                        visited.add((nr, nc))
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _find_palette(
    grid: np.ndarray,
    bg: int,
) -> tuple[int, list[int], set[tuple[int, int]]]:
    singleton_cells: list[tuple[int, int, int]] = []
    for color in sorted(set(map(int, grid.ravel())) - {int(bg)}):
        for comp in _components_of_color(grid, color):
            if len(comp) == 1:
                r, c = comp[0]
                singleton_cells.append((r, c, int(color)))

    rows: dict[int, list[tuple[int, int]]] = {}
    for r, c, color in singleton_cells:
        rows.setdefault(r, []).append((c, color))

    candidates = [(len(cells), r) for r, cells in rows.items() if len(cells) >= 2]
    assert candidates, "expected a palette row made of singleton colored pixels"
    best = max(count for count, _r in candidates)
    best_rows = [r for count, r in candidates if count == best]
    assert len(best_rows) == 1, "expected a unique palette row"

    palette_row = int(best_rows[0])
    palette_cells = sorted(rows[palette_row])
    palette_colors = [color for _c, color in palette_cells]
    assert len(palette_colors) == len(set(palette_colors)), "palette colors must be distinct"

    palette_cols = {c for c, _color in palette_cells}
    row_non_bg_cols = {c for c in range(grid.shape[1]) if int(grid[palette_row, c]) != int(bg)}
    assert palette_cols == row_non_bg_cols, "all non-background cells on the palette row must belong to the palette"
    palette_coords = {(palette_row, c) for c in palette_cols}
    return palette_row, palette_colors, palette_coords


def _moved_pair(
    comp_a: list[tuple[int, int]],
    comp_b: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    a_box = _bbox(comp_a)
    b_box = _bbox(comp_b)

    same_row_center = (a_box[0] + a_box[2]) == (b_box[0] + b_box[2])
    same_col_center = (a_box[1] + a_box[3]) == (b_box[1] + b_box[3])
    assert same_row_center ^ same_col_center, "pair centers must align on exactly one axis"

    if same_row_center:
        left, right = (comp_a, comp_b) if _bbox(comp_a)[1] <= _bbox(comp_b)[1] else (comp_b, comp_a)
        left_box = _bbox(left)
        right_box = _bbox(right)
        gap = right_box[1] - left_box[3] - 1
        assert gap >= 0, "horizontal pair fragments must not overlap"
        left_shift = gap // 2
        right_shift = gap - left_shift
        return _translate(left, 0, left_shift) + _translate(right, 0, -right_shift)

    top, bottom = (comp_a, comp_b) if _bbox(comp_a)[0] <= _bbox(comp_b)[0] else (comp_b, comp_a)
    top_box = _bbox(top)
    bottom_box = _bbox(bottom)
    gap = bottom_box[0] - top_box[2] - 1
    assert gap >= 0, "vertical pair fragments must not overlap"
    top_shift = gap // 2
    bottom_shift = gap - top_shift
    return _translate(top, top_shift, 0) + _translate(bottom, -bottom_shift, 0)


def _hole_cells(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True

    visited = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not visited[r, c]:
                visited[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not visited[r, c]:
                visited[r, c] = True
                stack.append((r, c))

    while stack:
        cr, cc = stack.pop()
        for nr, nc in _neighbors4(cr, cc):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not visited[nr, nc]:
                visited[nr, nc] = True
                stack.append((nr, nc))

    holes: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] and not visited[r, c]:
                holes.append((r0 + r, c0 + c))
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair equal-colored fragments, slide each pair toward each other until their
    bounding boxes touch, then fill the holes created by the merged shape using
    the palette color indexed by total hole area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _bg_color(g)
    _palette_row, palette_colors, palette_coords = _find_palette(g, bg)

    scene = g.copy()
    for r, c in palette_coords:
        scene[r, c] = bg

    pair_colors = sorted(set(map(int, scene.ravel())) - {int(bg)})
    assert pair_colors, "expected at least one fragment pair"

    out = g.copy()
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if (r, c) not in palette_coords and int(out[r, c]) != int(bg):
                out[r, c] = bg

    occupied = set(palette_coords)
    for color in pair_colors:
        components = _components_of_color(scene, color)
        assert len(components) == 2, "each fragment color must form exactly two components"
        moved = _moved_pair(components[0], components[1])
        holes = _hole_cells(moved)
        hole_area = len(holes)
        assert 1 <= hole_area <= len(palette_colors), "hole area must index a palette color"
        fill_color = int(palette_colors[hole_area - 1])

        for r, c in moved:
            assert (r, c) not in occupied, "moved fragments must not collide"
            occupied.add((r, c))
            out[r, c] = int(color)
        for r, c in holes:
            assert (r, c) not in occupied, "hole fill must stay inside empty enclosed cells"
            occupied.add((r, c))
            out[r, c] = fill_color

    return out


GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 200
PAIR_COUNT_OPTIONS = (1, 2, 3)
GAP_OPTIONS = (2, 4, 6)
GRID_LIMIT = 30


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for h in range(1, area + 1):
        if area % h == 0:
            pairs.append((h, area // h))
    return pairs


def _is_connected(coords: set[tuple[int, int]]) -> bool:
    if not coords:
        return False
    start = next(iter(coords))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in coords and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _build_pair_patch(
    bg: int,
    fragment_color: int,
    fill_color: int,
    area: int,
) -> tuple[np.ndarray, np.ndarray]:
    inner_h, inner_w = random.choice(_factor_pairs(area))
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    gap = random.choice(GAP_OPTIONS)
    shift = gap // 2
    orientation = random.choice(("horizontal", "vertical"))

    if orientation == "horizontal":
        patch_h, patch_w = outer_h, outer_w + gap
        frame_top, frame_left = 0, shift
        seam = random.randint(1, outer_w - 1)
    else:
        patch_h, patch_w = outer_h + gap, outer_w
        frame_top, frame_left = shift, 0
        seam = random.randint(1, outer_h - 1)

    border: set[tuple[int, int]] = set()
    for r in range(frame_top, frame_top + outer_h):
        for c in range(frame_left, frame_left + outer_w):
            if r in (frame_top, frame_top + outer_h - 1) or c in (frame_left, frame_left + outer_w - 1):
                border.add((r, c))

    holes = {
        (r, c)
        for r in range(frame_top + 1, frame_top + outer_h - 1)
        for c in range(frame_left + 1, frame_left + outer_w - 1)
    }
    assert len(holes) == area

    if orientation == "horizontal":
        split_c = frame_left + seam
        left = {cell for cell in border if cell[1] < split_c}
        right = border - left
        assert left and right and _is_connected(left) and _is_connected(right)
        input_border = {(r, c - shift) for r, c in left} | {(r, c + shift) for r, c in right}
    else:
        split_r = frame_top + seam
        top = {cell for cell in border if cell[0] < split_r}
        bottom = border - top
        assert top and bottom and _is_connected(top) and _is_connected(bottom)
        input_border = {(r - shift, c) for r, c in top} | {(r + shift, c) for r, c in bottom}

    inp = np.full((patch_h, patch_w), bg, dtype=int)
    out = np.full((patch_h, patch_w), bg, dtype=int)
    for r, c in input_border:
        inp[r, c] = fragment_color
    for r, c in border:
        out[r, c] = fragment_color
    for r, c in holes:
        out[r, c] = fill_color
    return inp, out


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    ph, pw = patch.shape
    view = grid[top : top + ph, left : left + pw]
    mask = patch != bg
    view[mask] = patch[mask]


def _candidate_positions(blocked: np.ndarray, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = blocked.shape
    ph, pw = shape
    positions: list[tuple[int, int]] = []
    for r in range(h - ph + 1):
        for c in range(w - pw + 1):
            r0 = max(0, r - 1)
            r1 = min(h, r + ph + 1)
            c0 = max(0, c - 1)
            c1 = min(w, c + pw + 1)
            if not blocked[r0:r1, c0:c1].any():
                positions.append((r, c))
    return positions


def _mark_blocked(blocked: np.ndarray, top: int, left: int, shape: tuple[int, int]) -> None:
    h, w = blocked.shape
    ph, pw = shape
    r0 = max(0, top - 1)
    r1 = min(h, top + ph + 1)
    c0 = max(0, left - 1)
    c1 = min(w, left + pw + 1)
    blocked[r0:r1, c0:c1] = True


def _choose_grid_shape(patches: list[np.ndarray], palette_len: int) -> tuple[int, int]:
    max_h = max(int(patch.shape[0]) for patch in patches)
    max_w = max(int(patch.shape[1]) for patch in patches)
    padded_area = sum((int(patch.shape[0]) + 2) * (int(patch.shape[1]) + 2) for patch in patches)
    base_side = int(np.ceil(np.sqrt(padded_area + 2 * (palette_len + 2))))

    h_lo = max(10, max_h + 3, base_side - 2)
    w_lo = max(10, max_w + 3, palette_len + 2, base_side - 2)
    h_hi = min(GRID_LIMIT, max(h_lo, base_side + 4))
    w_hi = min(GRID_LIMIT, max(w_lo, base_side + 4))

    return random.randint(h_lo, h_hi), random.randint(w_lo, w_hi)


def _build_candidate() -> tuple[np.ndarray, np.ndarray] | None:
    num_pairs = random.choice(PAIR_COUNT_OPTIONS)
    bg = random.randrange(10)
    available = [color for color in range(10) if color != bg]
    random.shuffle(available)

    fragment_colors = available[:num_pairs]
    remaining = available[num_pairs:]
    max_palette_len = len(remaining)
    if max_palette_len < 3:
        return None
    palette_len = random.randint(3, max_palette_len)
    palette_colors = remaining[:palette_len]

    patches: list[tuple[np.ndarray, np.ndarray]] = []
    for fragment_color in fragment_colors:
        area = random.randint(1, palette_len)
        fill_color = palette_colors[area - 1]
        patch_in, patch_out = _build_pair_patch(bg, fragment_color, fill_color, area)
        patches.append((patch_in, patch_out))

    height, width = _choose_grid_shape([patch_in for patch_in, _patch_out in patches], palette_len)
    inp = np.full((height, width), bg, dtype=int)
    out = np.full((height, width), bg, dtype=int)

    palette_row = 0 if random.random() < 0.5 else height - 1
    blocked = np.zeros((height, width), dtype=bool)
    blocked[palette_row, :] = True
    blocked[1 if palette_row == 0 else height - 2, :] = True

    palette_start = random.randint(0, width - palette_len)
    for i, color in enumerate(palette_colors):
        inp[palette_row, palette_start + i] = color
        out[palette_row, palette_start + i] = color

    placement_order = list(range(num_pairs))
    placement_order.sort(key=lambda idx: patches[idx][0].shape[0] * patches[idx][0].shape[1], reverse=True)

    for idx in placement_order:
        patch_in, patch_out = patches[idx]
        positions = _candidate_positions(blocked, patch_in.shape)
        if not positions:
            return None
        top, left = random.choice(positions)
        _paint_patch(inp, patch_in, top, left, bg)
        _paint_patch(out, patch_out, top, left, bg)
        _mark_blocked(blocked, top, left, patch_in.shape)

    counts = Counter(int(v) for v in inp.ravel())
    bg_count = counts[int(bg)]
    if any(bg_count <= count for color, count in counts.items() if color != int(bg)):
        return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        candidate = _build_candidate()
        if candidate is None:
            continue
        inp, expected = candidate
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
