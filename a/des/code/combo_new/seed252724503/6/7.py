import random
from collections import deque

import numpy as np


BG = 0
GEN_MAIN_COLOR = 5
GEN_FRAGMENT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_MIN_SIDE = 18
GRID_MAX_SIDE = 30
HOST_MIN_SIDE = 8
HOST_MAX_SIDE = 16
HOLE_COUNT_RANGE = (1, 4)
HOLE_SIDE_MIN = 1
HOLE_SIDE_MAX = 4
TAB_EXT_RANGE = (1, 3)
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _filled_rectangle(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc), h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _monochrome_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == int(bg) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc), h, w):
                    if bool(seen[nr, nc]) or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            box = _bbox(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": box,
                    "area": len(coords),
                }
            )
    return comps


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    comps = _monochrome_components(g, bg=BG)
    assert comps, "expected at least one non-background component"
    main_comp = max(comps, key=lambda comp: int(comp["area"]))
    main_color = int(main_comp["color"])

    holes = []
    for coords in _components_from_mask(g == int(BG)):
        if any(r == 0 or c == 0 or r == g.shape[0] - 1 or c == g.shape[1] - 1 for r, c in coords):
            continue
        if not _filled_rectangle(coords):
            continue
        coord_set = set(coords)
        enclosed = True
        for r, c in coords:
            for nr, nc in _neighbors4(r, c, g.shape[0], g.shape[1]):
                if (nr, nc) not in coord_set and int(g[nr, nc]) != main_color:
                    enclosed = False
                    break
            if not enclosed:
                break
        if not enclosed:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        holes.append(
            {
                "bbox": (r0, c0, r1, c1),
                "shape": (r1 - r0 + 1, c1 - c0 + 1),
            }
        )

    fragments = []
    for comp in comps:
        if comp is main_comp:
            continue
        if int(comp["color"]) == main_color:
            continue
        if not _filled_rectangle(comp["coords"]):
            continue
        r0, c0, r1, c1 = comp["bbox"]
        fragments.append(
            {
                "color": int(comp["color"]),
                "bbox": (r0, c0, r1, c1),
                "shape": (r1 - r0 + 1, c1 - c0 + 1),
                "coords": comp["coords"],
            }
        )

    hole_by_shape: dict[tuple[int, int], dict] = {}
    for hole in holes:
        shape = tuple(int(x) for x in hole["shape"])
        assert shape not in hole_by_shape, "hole sizes must be unique"
        hole_by_shape[shape] = hole

    fragment_by_shape: dict[tuple[int, int], dict] = {}
    for fragment in fragments:
        shape = tuple(int(x) for x in fragment["shape"])
        if shape not in hole_by_shape:
            continue
        assert shape not in fragment_by_shape, "fragment sizes must be unique"
        fragment_by_shape[shape] = fragment

    assert set(fragment_by_shape) == set(hole_by_shape), "holes and fragments must match by size"

    out = g.copy()
    for fragment in fragment_by_shape.values():
        for r, c in fragment["coords"]:
            out[r, c] = int(BG)
    for shape, fragment in fragment_by_shape.items():
        hole = hole_by_shape[shape]
        r0, c0, r1, c1 = hole["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(fragment["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reinsert the scattered colored rectangles into the matching holes of the large central pattern.

    Input structure:
    - The background is black.
    - One large connected monochrome pattern occupies the center of the grid.
    - Inside that pattern are one or more fully enclosed black rectangular holes.
    - Separate solid-color rectangular fragments sit on the grid edges or corners, with one fragment per hole.

    Rule:
    - Identify the large central pattern as the unique largest non-background component.
    - Find every enclosed black rectangular hole surrounded by that pattern.
    - Find each scattered solid rectangular fragment and match it to the unique hole with the same height and width.
    - Erase the fragment from its original location and fill the matched hole with the fragment's color, leaving the rest unchanged.

    Key insight:
    The fragments do not move by spatial proximity or color relation; the only reliable pairing cue is exact rectangle size matching between edge fragments and internal holes.
    """
    return _solve(grid)


def _paint_rect(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int):
    grid[r0 : r0 + h, c0 : c0 + w] = int(color)


def _hole_size_candidates(host_h: int, host_w: int) -> list[tuple[int, int]]:
    out = []
    for h in range(HOLE_SIDE_MIN, min(HOLE_SIDE_MAX, host_h - 2) + 1):
        for w in range(HOLE_SIDE_MIN, min(HOLE_SIDE_MAX, host_w - 2) + 1):
            if h * w >= 2:
                out.append((int(h), int(w)))
    return out


def _place_holes(host_h: int, host_w: int, sizes: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    order = sorted(sizes, key=lambda shape: (shape[0] * shape[1], shape[0], shape[1]), reverse=True)
    for _ in range(80):
        blocked = np.zeros((host_h, host_w), dtype=bool)
        placed: dict[tuple[int, int], tuple[int, int, int, int]] = {}
        ok = True
        for h, w in order:
            candidates: list[tuple[int, int]] = []
            for r in range(1, host_h - h):
                for c in range(1, host_w - w):
                    rr0 = max(0, r - 1)
                    rr1 = min(host_h, r + h + 1)
                    cc0 = max(0, c - 1)
                    cc1 = min(host_w, c + w + 1)
                    if bool(blocked[rr0:rr1, cc0:cc1].any()):
                        continue
                    candidates.append((int(r), int(c)))
            if not candidates:
                ok = False
                break
            r, c = random.choice(candidates)
            placed[(h, w)] = (r, c, r + h - 1, c + w - 1)
            blocked[max(0, r - 1) : min(host_h, r + h + 1), max(0, c - 1) : min(host_w, c + w + 1)] = True
        if ok:
            return [placed[size] for size in sizes]
    return None


def _tab_sides() -> tuple[str, ...]:
    styles = (
        (),
        ("top",),
        ("bottom",),
        ("left",),
        ("right",),
        ("top", "bottom"),
        ("left", "right"),
        ("top", "right"),
        ("bottom", "left"),
    )
    return random.choice(styles)


def _add_tabs(grid: np.ndarray, r0: int, c0: int, host_h: int, host_w: int):
    h, w = grid.shape
    for side in _tab_sides():
        if side == "top":
            max_ext = min(TAB_EXT_RANGE[1], r0 - 2)
            if max_ext < TAB_EXT_RANGE[0]:
                continue
            ext = random.randint(TAB_EXT_RANGE[0], max_ext)
            span = random.randint(max(2, host_w // 3), host_w)
            start = random.randint(c0, c0 + host_w - span)
            _paint_rect(grid, r0 - ext, start, ext, span, GEN_MAIN_COLOR)
        elif side == "bottom":
            max_ext = min(TAB_EXT_RANGE[1], h - (r0 + host_h) - 2)
            if max_ext < TAB_EXT_RANGE[0]:
                continue
            ext = random.randint(TAB_EXT_RANGE[0], max_ext)
            span = random.randint(max(2, host_w // 3), host_w)
            start = random.randint(c0, c0 + host_w - span)
            _paint_rect(grid, r0 + host_h, start, ext, span, GEN_MAIN_COLOR)
        elif side == "left":
            max_ext = min(TAB_EXT_RANGE[1], c0 - 2)
            if max_ext < TAB_EXT_RANGE[0]:
                continue
            ext = random.randint(TAB_EXT_RANGE[0], max_ext)
            span = random.randint(max(2, host_h // 3), host_h)
            start = random.randint(r0, r0 + host_h - span)
            _paint_rect(grid, start, c0 - ext, span, ext, GEN_MAIN_COLOR)
        else:
            max_ext = min(TAB_EXT_RANGE[1], w - (c0 + host_w) - 2)
            if max_ext < TAB_EXT_RANGE[0]:
                continue
            ext = random.randint(TAB_EXT_RANGE[0], max_ext)
            span = random.randint(max(2, host_h // 3), host_h)
            start = random.randint(r0, r0 + host_h - span)
            _paint_rect(grid, start, c0 + host_w, span, ext, GEN_MAIN_COLOR)


def _edge_positions(grid_h: int, grid_w: int, h: int, w: int) -> list[tuple[int, int]]:
    pos = set()
    for c in range(grid_w - w + 1):
        pos.add((0, c))
        pos.add((grid_h - h, c))
    for r in range(grid_h - h + 1):
        pos.add((r, 0))
        pos.add((r, grid_w - w))
    positions = list(pos)
    random.shuffle(positions)
    return positions


def _can_place_fragment(grid: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    if bool((grid[r0 : r0 + h, c0 : c0 + w] != int(BG)).any()):
        return False
    rr0 = max(0, r0 - 1)
    rr1 = min(grid.shape[0], r0 + h + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(grid.shape[1], c0 + w + 1)
    return not bool((grid[rr0:rr1, cc0:cc1] != int(BG)).any())


def _place_fragments(
    grid: np.ndarray,
    sizes: list[tuple[int, int]],
    colors: list[int],
) -> list[tuple[int, int, int, int, int]] | None:
    order = sorted(zip(sizes, colors), key=lambda item: (item[0][0] * item[0][1], item[0][0], item[0][1]), reverse=True)
    placements: list[tuple[int, int, int, int, int]] = []
    for (h, w), color in order:
        placed = None
        for r0, c0 in _edge_positions(grid.shape[0], grid.shape[1], h, w):
            if _can_place_fragment(grid, r0, c0, h, w):
                _paint_rect(grid, r0, c0, h, w, color)
                placed = (r0, c0, h, w, color)
                break
        if placed is None:
            return None
        placements.append(placed)
    return placements


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN_SIDE, GRID_MAX_SIDE)
    w = random.randint(GRID_MIN_SIDE, GRID_MAX_SIDE)
    max_host_h = min(HOST_MAX_SIDE, h - 8)
    max_host_w = min(HOST_MAX_SIDE, w - 8)
    if max_host_h < HOST_MIN_SIDE or max_host_w < HOST_MIN_SIDE:
        return None

    host_h = random.randint(HOST_MIN_SIDE, max_host_h)
    host_w = random.randint(HOST_MIN_SIDE, max_host_w)
    r0 = random.randint(4, h - host_h - 4)
    c0 = random.randint(4, w - host_w - 4)

    sizes_all = _hole_size_candidates(host_h, host_w)
    hole_count = random.randint(HOLE_COUNT_RANGE[0], min(HOLE_COUNT_RANGE[1], len(sizes_all)))
    hole_sizes = random.sample(sizes_all, hole_count)
    hole_boxes = _place_holes(host_h, host_w, hole_sizes)
    if hole_boxes is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    _paint_rect(grid, r0, c0, host_h, host_w, GEN_MAIN_COLOR)
    _add_tabs(grid, r0, c0, host_h, host_w)

    absolute_holes: list[tuple[int, int, int, int]] = []
    for (hole_h, hole_w), (hr0, hc0, hr1, hc1) in zip(hole_sizes, hole_boxes):
        abs_r0 = r0 + hr0
        abs_c0 = c0 + hc0
        absolute_holes.append((abs_r0, abs_c0, abs_r0 + hole_h - 1, abs_c0 + hole_w - 1))
        grid[abs_r0 : abs_r0 + hole_h, abs_c0 : abs_c0 + hole_w] = int(BG)

    colors = random.sample(GEN_FRAGMENT_COLORS, hole_count)
    fragment_grid = grid.copy()
    fragment_placements = _place_fragments(fragment_grid, hole_sizes, colors)
    if fragment_placements is None:
        return None

    expected = fragment_grid.copy()
    for fr0, fc0, fh, fw, _color in fragment_placements:
        expected[fr0 : fr0 + fh, fc0 : fc0 + fw] = int(BG)
    for (_hr0, _hc0, _hr1, _hc1), color in zip(absolute_holes, colors):
        hole_h, hole_w = _hr1 - _hr0 + 1, _hc1 - _hc0 + 1
        expected[_hr0 : _hr0 + hole_h, _hc0 : _hc0 + hole_w] = int(color)

    try:
        solved = main(fragment_grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return fragment_grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
