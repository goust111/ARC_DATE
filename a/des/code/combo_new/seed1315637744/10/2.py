import numpy as np


AZURE = 8
GREY = 5
PALETTE = (0, 1, 2, 3, 4, 6, 7, 9)
MAX_GRID = 30
GENERATION_ATTEMPTS = 500

_RNG = np.random.default_rng()


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(int(lo), int(hi) + 1))


def _choice(seq):
    return seq[_randint(0, len(seq) - 1)]


def _sample(seq, k: int) -> list[int]:
    idx = _RNG.choice(len(seq), size=int(k), replace=False)
    return [int(seq[int(i)]) for i in idx]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    rows, cols = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(rows):
        for c in range(cols):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _cross_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    return [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_cross_center(grid: np.ndarray) -> tuple[int, int]:
    valid: list[tuple[int, int]] = []
    for comp in _components(grid == GREY):
        if int(comp.shape[0]) != 5:
            continue
        coords = {(int(r), int(c)) for r, c in comp}
        for r, c in coords:
            if coords == set(_cross_cells((int(r), int(c)))):
                valid.append((int(r), int(c)))
                break
    assert len(valid) == 1, "expected one grey plus marker"
    return valid[0]


def _rectangle_groups(grid: np.ndarray) -> list[dict]:
    groups: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color in (AZURE, GREY):
            continue
        comps = _components(grid == color)
        if not comps:
            continue

        rects: list[dict] = []
        first = None
        for comp in comps:
            r0, c0, r1, c1 = _bbox(comp)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            area = int(comp.shape[0])
            assert area == h * w, "foreground objects must be solid rectangles"
            rects.append({"color": color, "h": h, "w": w, "area": area})
            pos = (int(r0), int(c0))
            first = pos if first is None else min(first, pos)

        assert len(rects) == 3, "each color must form exactly three rectangles"
        rects.sort(key=lambda item: int(item["area"]), reverse=True)
        assert rects[0]["area"] > rects[1]["area"] > rects[2]["area"], "areas must be distinct within a color group"
        groups.append({"color": color, "first": first, "rects": rects})

    assert groups, "expected at least one color group"
    groups.sort(key=lambda item: item["first"])
    return groups


def _render_nested(layers: list[dict], center: tuple[int, int]) -> np.ndarray:
    placements = []
    for layer in layers:
        h = int(layer["h"])
        w = int(layer["w"])
        r0 = int(center[0] - h // 2)
        c0 = int(center[1] - w // 2)
        placements.append((r0, c0, h, w, int(layer["color"])))

    min_r = min(r0 for r0, _, _, _, _ in placements)
    min_c = min(c0 for _, c0, _, _, _ in placements)
    max_r = max(r0 + h for r0, _, h, _, _ in placements)
    max_c = max(c0 + w for _, c0, _, w, _ in placements)

    out = np.full((max_r - min_r, max_c - min_c), AZURE, dtype=int)
    for r0, c0, h, w, color in placements:
        out[r0 - min_r : r0 - min_r + h, c0 - min_c : c0 - min_c + w] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered same-color rectangles, sort each group by area, and rebuild them as one centered nesting.

    Input structure:
    - The background is azure.
    - Several foreground colors each form exactly three separate 4-connected solid rectangles.
    - A small grey plus marker in one corner indicates the common center used by the reconstructed nesting.

    Rule:
    - Ignore the grey plus after locating its center.
    - For each foreground color, collect its three rectangles and sort them by area from largest to smallest as L, M, S.
    - Order the color groups by the row-major first appearance of that color in the input, then place every rectangle centered on the grey plus:
      first group's L, then M, then S, then the next group's L, M, S, and so on.
    - Crop the result to the smallest rectangle containing the full centered stack.

    Key insight:
    - The puzzle combines two hidden orderings: rectangles must first be grouped by color and ranked by area, then the color groups themselves are ranked by spatial first appearance before everything is center-stacked on the marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    center = _find_cross_center(g)
    groups = _rectangle_groups(g)
    layers = [rect for group in groups for rect in group["rects"]]
    return _render_nested(layers, center)


def _lex_gt(pos: tuple[int, int], lower: tuple[int, int] | None) -> bool:
    if lower is None:
        return True
    return pos[0] > lower[0] or (pos[0] == lower[0] and pos[1] > lower[1])


def _free_positions(
    occupied: np.ndarray,
    h: int,
    w: int,
    *,
    lower: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    rows, cols = occupied.shape
    positions: list[tuple[int, int]] = []
    for r in range(rows - h + 1):
        for c in range(cols - w + 1):
            pos = (int(r), int(c))
            if not _lex_gt(pos, lower):
                continue
            r0 = max(0, r - 1)
            c0 = max(0, c - 1)
            r1 = min(rows, r + h + 1)
            c1 = min(cols, c + w + 1)
            if bool(occupied[r0:r1, c0:c1].any()):
                continue
            positions.append(pos)
    return positions


def _paint_rect(grid: np.ndarray, occupied: np.ndarray, rect: dict, pos: tuple[int, int]) -> None:
    r, c = pos
    h = int(rect["h"])
    w = int(rect["w"])
    grid[r : r + h, c : c + w] = int(rect["color"])
    occupied[r : r + h, c : c + w] = True


def _choose_chain(group_count: int) -> list[tuple[int, int]] | None:
    total = 3 * int(group_count)
    for _ in range(200):
        h = _choice((1, 3, 5))
        w = _choice((1, 3, 5))
        dims = [(int(h), int(w))]
        grew_h = False
        grew_w = False
        for _ in range(total - 1):
            roll = float(_RNG.random())
            if roll < 0.4:
                dh, dw = 2, 0
            elif roll < 0.8:
                dh, dw = 0, 2
            else:
                dh, dw = 2, 2
            h += dh
            w += dw
            grew_h = grew_h or dh > 0
            grew_w = grew_w or dw > 0
            dims.append((int(h), int(w)))

        dims = list(reversed(dims))
        outer_h, outer_w = dims[0]
        total_area = sum(hh * ww for hh, ww in dims)
        if not grew_h or not grew_w:
            continue
        if min(outer_h, outer_w) < 3 or max(outer_h, outer_w) > MAX_GRID:
            continue
        if total_area > 260:
            continue
        return dims
    return None


def _corner_center(rows: int, cols: int) -> tuple[int, int]:
    return _choice(((1, 1), (1, cols - 2), (rows - 2, 1), (rows - 2, cols - 2)))


def _pick_leader_position(choices: list[tuple[int, int]], groups_left: int) -> tuple[int, int]:
    window = max(1, len(choices) // max(1, groups_left + 1))
    return choices[_randint(0, window - 1)]


def _build_groups() -> tuple[list[dict], list[dict]] | None:
    group_count = _randint(2, 4)
    dims = _choose_chain(group_count)
    if dims is None:
        return None

    colors = _sample(PALETTE, group_count)
    groups: list[dict] = []
    layers: list[dict] = []
    idx = 0
    for color in colors:
        rects = []
        for _ in range(3):
            h, w = dims[idx]
            idx += 1
            rect = {"color": int(color), "h": int(h), "w": int(w), "area": int(h * w)}
            rects.append(rect)
            layers.append(rect)
        groups.append({"color": int(color), "rects": rects})
    return groups, layers


def _place_groups(groups: list[dict], rows: int, cols: int, center: tuple[int, int]) -> np.ndarray | None:
    grid = np.full((rows, cols), AZURE, dtype=int)
    occupied = np.zeros((rows, cols), dtype=bool)
    for r, c in _cross_cells(center):
        grid[r, c] = GREY
        occupied[r, c] = True

    prev_first = None
    total_groups = len(groups)
    for gi, group in enumerate(groups):
        rects = group["rects"]
        leader_idx = _randint(0, 2)
        leader = rects[leader_idx]
        leader_choices = _free_positions(occupied, int(leader["h"]), int(leader["w"]), lower=prev_first)
        if not leader_choices:
            return None
        leader_pos = _pick_leader_position(leader_choices, total_groups - gi)
        _paint_rect(grid, occupied, leader, leader_pos)

        other_indices = [idx for idx in range(3) if idx != leader_idx]
        other_indices.sort(key=lambda idx: int(rects[idx]["area"]), reverse=True)
        for idx in other_indices:
            rect = rects[idx]
            choices = _free_positions(occupied, int(rect["h"]), int(rect["w"]), lower=leader_pos)
            if not choices:
                return None
            pos = _choice(choices)
            _paint_rect(grid, occupied, rect, pos)

        prev_first = leader_pos

    return grid


def _gen_attempt() -> np.ndarray | None:
    built = _build_groups()
    if built is None:
        return None
    groups, layers = built

    total_area = sum(int(layer["area"]) for layer in layers)
    min_side = max(18, int(np.ceil(np.sqrt(total_area * 2.2))))
    rows = _randint(min(min_side, MAX_GRID), MAX_GRID)
    cols = _randint(min(min_side, MAX_GRID), MAX_GRID)
    center = _corner_center(rows, cols)

    grid = _place_groups(groups, rows, cols, center)
    if grid is None:
        return None

    expected = _render_nested(layers, center)
    if max(expected.shape) > MAX_GRID:
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
