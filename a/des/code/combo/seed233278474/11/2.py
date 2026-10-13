from __future__ import annotations

import random

import numpy as np


BG = 0
MARKER = 7
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize_cells(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    min_r = min(rows)
    min_c = min(cols)
    return frozenset((r - min_r, c - min_c) for r, c in cells)


def _cells_to_patch(cells: frozenset[tuple[int, int]], color: int) -> np.ndarray:
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    patch = np.full((h, w), BG, dtype=int)
    for r, c in cells:
        patch[r, c] = int(color)
    return patch


def _find_marker_side(g: np.ndarray) -> str:
    marker = np.argwhere(g == MARKER)
    assert marker.size > 0, "marker line must exist"
    if np.all(marker[:, 0] == 0):
        return "top"
    if np.all(marker[:, 1] == 0):
        return "left"
    raise AssertionError("marker must lie on the top edge or left edge")


def _assembled_dim_set(
    a: frozenset[tuple[int, int]],
    b: frozenset[tuple[int, int]],
) -> set[tuple[int, int]]:
    area = len(a) + len(b)
    limit = area
    dims: set[tuple[int, int]] = set()
    rows_a = [r for r, _ in a]
    cols_a = [c for _, c in a]
    min_ra = min(rows_a)
    max_ra = max(rows_a)
    min_ca = min(cols_a)
    max_ca = max(cols_a)

    list_b = list(b)
    for dr in range(-limit, limit + 1):
        for dc in range(-limit, limit + 1):
            shifted = {(r + dr, c + dc) for r, c in list_b}
            if shifted & a:
                continue
            rows_b = [r for r, _ in shifted]
            cols_b = [c for _, c in shifted]
            min_r = min(min_ra, min(rows_b))
            max_r = max(max_ra, max(rows_b))
            min_c = min(min_ca, min(cols_b))
            max_c = max(max_ca, max(cols_b))
            h = max_r - min_r + 1
            w = max_c - min_c + 1
            if h * w == area:
                dims.add((h, w))
    return dims


def _assembled_dims(
    a: frozenset[tuple[int, int]],
    b: frozenset[tuple[int, int]],
) -> tuple[int, int]:
    dims = _assembled_dim_set(a, b)
    assert len(dims) == 1, "each color pair must assemble into one rectangle size"
    return next(iter(dims))


def _line_layout(rects: list[np.ndarray], vertical: bool) -> np.ndarray:
    if not rects:
        return np.full((0, 0), BG, dtype=int)
    if vertical:
        h = sum(rect.shape[0] for rect in rects) + len(rects) - 1
        w = max(rect.shape[1] for rect in rects)
        out = np.full((h, w), BG, dtype=int)
        r = 0
        for rect in rects:
            hh, ww = rect.shape
            out[r : r + hh, :ww] = rect
            r += hh + 1
        return out
    h = max(rect.shape[0] for rect in rects)
    w = sum(rect.shape[1] for rect in rects) + len(rects) - 1
    out = np.full((h, w), BG, dtype=int)
    c = 0
    for rect in rects:
        hh, ww = rect.shape
        out[:hh, c : c + ww] = rect
        c += ww + 1
    return out


def _combine_groups(first: np.ndarray, second: np.ndarray, stack_vertical: bool) -> np.ndarray:
    if first.size == 0:
        return second.copy()
    if second.size == 0:
        return first.copy()
    if stack_vertical:
        h = first.shape[0] + 1 + second.shape[0]
        w = max(first.shape[1], second.shape[1])
        out = np.full((h, w), BG, dtype=int)
        out[: first.shape[0], : first.shape[1]] = first
        r0 = first.shape[0] + 1
        out[r0 : r0 + second.shape[0], : second.shape[1]] = second
        return out
    h = max(first.shape[0], second.shape[0])
    w = first.shape[1] + 1 + second.shape[1]
    out = np.full((h, w), BG, dtype=int)
    out[: first.shape[0], : first.shape[1]] = first
    c0 = first.shape[1] + 1
    out[: second.shape[0], c0 : c0 + second.shape[1]] = second
    return out


def _render_output(rect_specs: list[tuple[int, int, int]], marker_side: str) -> np.ndarray:
    horiz: list[np.ndarray] = []
    vert: list[np.ndarray] = []
    for color, h, w in rect_specs:
        rect = np.full((h, w), int(color), dtype=int)
        if w >= h:
            horiz.append(rect)
        else:
            vert.append(rect)
    if marker_side == "top":
        top = _line_layout(horiz, vertical=False)
        bottom = _line_layout(vert, vertical=False)
        return _combine_groups(top, bottom, stack_vertical=True)
    left = _line_layout(vert, vertical=True)
    right = _line_layout(horiz, vertical=True)
    return _combine_groups(left, right, stack_vertical=False)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored fragments into rectangles, then arrange the rectangles by
    aspect group and by the orange edge marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    marker_side = _find_marker_side(g)
    colors = [int(c) for c in np.unique(g) if int(c) not in (BG, MARKER)]
    assert colors, "there must be at least one fragment pair"

    first_index = {
        color: int(np.flatnonzero(g.ravel() == color)[0])
        for color in colors
    }

    rect_specs: list[tuple[int, int, int]] = []
    for color in sorted(colors, key=lambda c: first_index[c]):
        comps = _components(g == color)
        assert len(comps) == 2, "each color must appear in exactly two fragments"
        a = _normalize_cells(comps[0])
        b = _normalize_cells(comps[1])
        h, w = _assembled_dims(a, b)
        rect_specs.append((color, h, w))

    return _render_output(rect_specs, marker_side)


def _random_profile(length: int, low: int, high: int) -> list[int]:
    assert low <= high
    cur = random.randint(low, high)
    profile = [cur]
    for _ in range(length - 1):
        cur = min(high, max(low, cur + random.choice([-1, 0, 1])))
        profile.append(cur)
    if high > low and len(set(profile)) == 1 and random.random() < 0.7:
        idx = random.randrange(length)
        delta = random.choice([-1, 1])
        profile[idx] = min(high, max(low, profile[idx] + delta))
    return profile


def _split_rectangle(h: int, w: int) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]:
    if w >= 2 and (h == 1 or random.random() < 0.5):
        cut = _random_profile(h, 1, w - 1)
        a = {(r, c) for r in range(h) for c in range(cut[r])}
        b = {(r, c) for r in range(h) for c in range(cut[r], w)}
    else:
        cut = _random_profile(w, 1, h - 1)
        a = {(r, c) for c in range(w) for r in range(cut[c])}
        b = {(r, c) for c in range(w) for r in range(cut[c], h)}
    if random.random() < 0.5:
        a, b = b, a
    return _normalize_cells(a), _normalize_cells(b)


def _choose_rect_kind() -> str:
    return random.choice(["h", "v", "s"])


def _rect_dims(kind: str) -> tuple[int, int]:
    if kind == "h":
        h = random.randint(2, 4)
        w = random.randint(h + 1, min(7, h + 3))
        return h, w
    if kind == "v":
        w = random.randint(2, 4)
        h = random.randint(w + 1, min(7, w + 3))
        return h, w
    side = random.randint(2, 5)
    return side, side


def _rect_kinds(count: int) -> list[str]:
    if count == 1:
        return [_choose_rect_kind()]
    if random.random() < 0.75:
        kinds = ["h", "v"]
        while len(kinds) < count:
            kinds.append(_choose_rect_kind())
        random.shuffle(kinds)
        return kinds
    return [_choose_rect_kind() for _ in range(count)]


def _can_place(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = patch.shape
    gh, gw = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > gh or c0 + w > gw:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(gh, r0 + h + 1)
    cc1 = min(gw, c0 + w + 1)
    return bool(np.all(grid[rr0:rr1, cc0:cc1] == BG))


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    mask = patch != BG
    target = grid[r0 : r0 + h, c0 : c0 + w]
    target[mask] = patch[mask]


def _first_index(g: np.ndarray, color: int) -> int:
    return int(np.flatnonzero(g.ravel() == color)[0])


def _expected_from_specs(
    grid: np.ndarray,
    marker_side: str,
    specs: list[tuple[int, int, int]],
) -> np.ndarray:
    ordered = sorted(specs, key=lambda item: _first_index(grid, item[0]))
    return _render_output(ordered, marker_side)


def _gen_attempt() -> np.ndarray | None:
    marker_side = random.choice(["top", "left"])
    pair_count = random.randint(2, 5)
    colors = random.sample([1, 2, 3, 4, 5, 6, 8, 9], pair_count)
    kinds = _rect_kinds(pair_count)

    specs: list[tuple[int, int, int]] = []
    patches: list[tuple[int, np.ndarray]] = []
    for color, kind in zip(colors, kinds, strict=False):
        h, w = _rect_dims(kind)
        a, b = _split_rectangle(h, w)
        if _assembled_dim_set(a, b) != {(h, w)}:
            return None
        specs.append((color, h, w))
        patches.append((color, _cells_to_patch(a, color)))
        patches.append((color, _cells_to_patch(b, color)))

    random.shuffle(patches)

    grid_h = random.randint(18, 24)
    grid_w = random.randint(18, 24)
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    if marker_side == "top":
        grid[0, :] = MARKER
        row_min = 2
        col_min = 1
    else:
        grid[:, 0] = MARKER
        row_min = 1
        col_min = 2

    for _color, patch in patches:
        placed = False
        h, w = patch.shape
        if row_min > grid_h - h or col_min > grid_w - w:
            return None
        for _ in range(200):
            r0 = random.randint(row_min, grid_h - h)
            c0 = random.randint(col_min, grid_w - w)
            if _can_place(grid, patch, r0, c0):
                _place_patch(grid, patch, r0, c0)
                placed = True
                break
        if not placed:
            return None

    try:
        out = main(grid)
    except Exception:
        return None

    expected = _expected_from_specs(grid, marker_side, specs)
    if np.array_equal(out, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
