import random

import numpy as np


BG = 0
GREY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

GEN_ATTEMPTS = 400
GRID_HEIGHT_RANGE = (16, 24)
GRID_WIDTH_RANGE = (28, 48)
COLOR_GROUP_RANGE = (2, 4)
FRAGMENT_COUNT_CHOICES = [2, 3, 4]
ANCHOR_GAP_RANGE = (2, 4)
SCATTER_ROW_MIN = 4

TOP_HEAVY_SHAPES = {
    2: [
        [(0, 0), (0, 1)],
    ],
    3: [
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (0, 1), (1, 1)],
    ],
    4: [
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(0, 0), (0, 1), (0, 2), (1, 0)],
        [(0, 0), (0, 1), (0, 2), (1, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 2)],
    ],
    5: [
        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
        [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)],
        [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1)],
        [(0, 0), (0, 1), (0, 2), (0, 3), (1, 2)],
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)],
    ],
}

REFERENCE_PATTERNS = {
    2: [
        [(0, 0), (1, 1)],
        [(0, 1), (1, 0)],
        [(0, 0), (1, 2)],
        [(0, 2), (1, 0)],
    ],
    3: [
        [(0, 0), (0, 3), (1, 1)],
        [(0, 1), (0, 4), (1, 2)],
        [(0, 0), (0, 4), (1, 1)],
        [(0, 1), (0, 5), (1, 3)],
    ],
    4: [
        [(0, 0), (0, 3), (1, 1), (1, 4)],
        [(0, 1), (0, 4), (1, 0), (1, 3)],
        [(0, 0), (0, 4), (1, 2), (1, 5)],
        [(0, 1), (0, 5), (1, 2), (1, 4)],
    ],
}


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


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
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> list[tuple[int, int]]:
    r0, c0, _, _ = _bbox(cells)
    return sorted((r - r0, c - c0) for r, c in cells)


def _round_half_up(total: int, count: int) -> int:
    return int((2 * total + count) // (2 * count))


def _center_of_cells(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int]:
    pts = list(cells)
    return _round_half_up(sum(r for r, _ in pts), len(pts)), _round_half_up(
        sum(c for _, c in pts), len(pts)
    )


def _is_border_cell(r: int, c: int, h: int, w: int) -> bool:
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _reference_points(grid: np.ndarray) -> list[tuple[int, int]]:
    points = []
    for comp in _components(grid == GREY):
        r0, c0, _, _ = _bbox(comp)
        points.append((r0, c0))
    points.sort()
    return points


def _shape_signature(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(_normalize(cells))


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    anchors: dict[int, tuple[int, int]] = {}
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color in (bg, GREY) or not _is_border_cell(r, c, h, w):
                continue
            isolated = True
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) != bg:
                    isolated = False
                    break
            if isolated:
                anchors[color] = (r, c)

    ref_points = _reference_points(g)
    assert 2 <= len(ref_points) <= 4
    ref_base_r = min(r for r, _ in ref_points)
    ref_base_c = min(c for _, c in ref_points)

    anchor_cells = set(anchors.values())
    fragments_by_color: dict[int, list[list[tuple[int, int]]]] = {}
    to_clear: list[tuple[int, int]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in (bg, GREY)):
        for comp in _components(g == color):
            if len(comp) == 1 and comp[0] in anchor_cells:
                continue
            fragments_by_color.setdefault(color, []).append(comp)
            to_clear.extend(comp)

    out = g.copy()
    for r, c in to_clear:
        out[r, c] = bg

    for color, anchor in anchors.items():
        fragments = fragments_by_color.get(color, [])
        assert len(fragments) == len(ref_points)
        fragments.sort(key=lambda comp: (-len(comp), _bbox(comp)[0], _bbox(comp)[1]))

        assembled: set[tuple[int, int]] = set()
        for frag, ref_pt in zip(fragments, ref_points):
            frag_r0, frag_c0, _, _ = _bbox(frag)
            dr = ref_pt[0] - ref_base_r
            dc = ref_pt[1] - ref_base_c
            for r, c in frag:
                assembled.add((r - frag_r0 + dr, c - frag_c0 + dc))

        center_r, center_c = _center_of_cells(assembled)
        shift_r = anchor[0] - center_r
        shift_c = anchor[1] - center_c
        for r, c in assembled:
            rr = r + shift_r
            cc = c + shift_c
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = color

    return out


def _mirror_h(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    _, _, _, c1 = _bbox(cells)
    return _normalize([(r, c1 - c) for r, c in cells])


def _random_fragment(area: int) -> list[tuple[int, int]]:
    cells = random.choice(TOP_HEAVY_SHAPES[area])
    if random.random() < 0.5:
        cells = _mirror_h(cells)
    return _normalize(cells)


def _assembled_from_parts(
    fragments: list[list[tuple[int, int]]], ref_points: list[tuple[int, int]]
) -> set[tuple[int, int]]:
    ref_base_r = min(r for r, _ in ref_points)
    ref_base_c = min(c for _, c in ref_points)
    assembled: set[tuple[int, int]] = set()
    total_area = 0
    for frag, ref_pt in zip(fragments, ref_points):
        total_area += len(frag)
        for r, c in frag:
            assembled.add((r + ref_pt[0] - ref_base_r, c + ref_pt[1] - ref_base_c))
    if len(assembled) != total_area:
        return set()
    return assembled


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]] | set[tuple[int, int]], pad: int = 1) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _scatter_fragment(
    grid: np.ndarray, blocked: np.ndarray, color: int, frag: list[tuple[int, int]], max_row: int
) -> bool:
    h, w = grid.shape
    frag_h = 1 + max(r for r, _ in frag)
    frag_w = 1 + max(c for _, c in frag)
    if max_row < frag_h - 1 or w < frag_w:
        return False
    for _ in range(250):
        r0 = random.randint(SCATTER_ROW_MIN, max_row - frag_h + 1)
        c0 = random.randint(0, w - frag_w)
        abs_cells = [(r0 + r, c0 + c) for r, c in frag]
        if any(blocked[r, c] for r, c in abs_cells):
            continue
        for r, c in abs_cells:
            grid[r, c] = color
        _mark_blocked(blocked, abs_cells, pad=1)
        return True
    return False


def _sample_reference_pattern(k: int) -> list[tuple[int, int]]:
    pattern = list(random.choice(REFERENCE_PATTERNS[k]))
    shift = random.randint(0, 1)
    return [(r, c + shift) for r, c in pattern]


def _sample_group_specs(
    colors: list[int], ref_points: list[tuple[int, int]]
) -> list[dict] | None:
    specs = []
    for color in colors:
        for _ in range(80):
            areas = sorted(random.sample([2, 3, 4, 5], len(ref_points)), reverse=True)
            fragments = [_random_fragment(area) for area in areas]
            assembled = _assembled_from_parts(fragments, ref_points)
            if not assembled:
                continue
            if not _is_connected(assembled):
                continue
            center_r, center_c = _center_of_cells(assembled)
            if center_r != 0:
                continue
            specs.append(
                {
                    "color": color,
                    "fragments": fragments,
                    "assembled": assembled,
                    "center": (center_r, center_c),
                    "bbox": _bbox(assembled),
                }
            )
            break
        else:
            return None

    signatures = {_shape_signature(spec["assembled"]) for spec in specs}
    if len(signatures) < len(specs):
        return None
    return specs


def _anchor_layout(specs: list[dict]) -> tuple[int, list[dict]] | None:
    ordered = specs[:]
    random.shuffle(ordered)
    cursor = random.randint(1, 3)
    width_need = cursor
    for idx, spec in enumerate(ordered):
        _r0, c0, _r1, c1 = spec["bbox"]
        width = c1 - c0 + 1
        spec["anchor_col"] = cursor + spec["center"][1]
        cursor += width
        width_need = cursor
        if idx != len(ordered) - 1:
            cursor += random.randint(*ANCHOR_GAP_RANGE)
            width_need = cursor
    width_need += random.randint(1, 3)
    if width_need > GRID_WIDTH_RANGE[1]:
        return None
    grid_w = random.randint(max(width_need, GRID_WIDTH_RANGE[0]), GRID_WIDTH_RANGE[1])
    return grid_w, ordered


def _render_expected(
    h: int, w: int, ref_origin: tuple[int, int], ref_points: list[tuple[int, int]], specs: list[dict]
) -> np.ndarray:
    out = np.full((h, w), BG, dtype=int)
    for rr, cc in ref_points:
        out[ref_origin[0] + rr, ref_origin[1] + cc] = GREY
    for spec in specs:
        color = spec["color"]
        anchor_col = spec["anchor_col"]
        out[0, anchor_col] = color
        center_r, center_c = spec["center"]
        shift_r = -center_r
        shift_c = anchor_col - center_c
        for r, c in spec["assembled"]:
            out[r + shift_r, c + shift_c] = color
    return out


def _render_input(
    h: int, w: int, ref_origin: tuple[int, int], ref_points: list[tuple[int, int]], specs: list[dict]
) -> np.ndarray | None:
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    ref_cells = [(ref_origin[0] + rr, ref_origin[1] + cc) for rr, cc in ref_points]
    for r, c in ref_cells:
        grid[r, c] = GREY
    _mark_blocked(blocked, ref_cells, pad=1)

    for spec in specs:
        anchor = (0, spec["anchor_col"])
        grid[anchor] = spec["color"]
        _mark_blocked(blocked, [anchor], pad=1)

    max_scatter_row = ref_origin[0] - 2
    if max_scatter_row < SCATTER_ROW_MIN + 1:
        return None

    fragments: list[tuple[int, list[tuple[int, int]]]] = []
    for spec in specs:
        for frag in spec["fragments"]:
            fragments.append((spec["color"], frag))
    random.shuffle(fragments)
    fragments.sort(key=lambda item: -len(item[1]))

    for color, frag in fragments:
        if not _scatter_fragment(grid, blocked, color, frag, max_scatter_row):
            return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    group_count = random.randint(*COLOR_GROUP_RANGE)
    colors = random.sample(PALETTE, group_count)
    fragment_count = random.choice(FRAGMENT_COUNT_CHOICES)
    ref_points = _sample_reference_pattern(fragment_count)

    specs = _sample_group_specs(colors, ref_points)
    if specs is None:
        return None

    layout = _anchor_layout(specs)
    if layout is None:
        return None
    grid_w, ordered_specs = layout

    ref_width = 1 + max(c for _, c in ref_points)
    grid_h = random.randint(*GRID_HEIGHT_RANGE)
    ref_r0 = random.randint(max(8, grid_h - 5), grid_h - 3)
    ref_c0_max = grid_w - ref_width - 2
    if ref_c0_max < 1:
        return None
    ref_origin = (ref_r0, random.randint(1, ref_c0_max))

    expected = _render_expected(grid_h, grid_w, ref_origin, ref_points, ordered_specs)
    if np.any(expected[:, :] == GREY) and np.any(expected[0, :] == GREY):
        return None

    inp = _render_input(grid_h, grid_w, ref_origin, ref_points, ordered_specs)
    if inp is None:
        return None

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
