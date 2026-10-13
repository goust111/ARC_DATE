import numpy as np

from collections import deque


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _normalize(coords) -> tuple[frozenset[tuple[int, int]], int, int]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    norm = frozenset((r - min_r, c - min_c) for r, c in coords)
    max_r = max(r for r, _ in norm)
    max_c = max(c for _, c in norm)
    return norm, max_r + 1, max_c + 1


def _rotate_once(coords: frozenset[tuple[int, int]], h: int, w: int):
    rotated = frozenset((c, h - 1 - r) for r, c in coords)
    return rotated, w, h


def _rotate_k(coords: frozenset[tuple[int, int]], h: int, w: int, k: int):
    out_coords, out_h, out_w = coords, h, w
    for _ in range(k % 4):
        out_coords, out_h, out_w = _rotate_once(out_coords, out_h, out_w)
    return out_coords, out_h, out_w


def _unique_rotations(coords: frozenset[tuple[int, int]], h: int, w: int):
    rotations = []
    seen = set()
    cur_coords, cur_h, cur_w = coords, h, w
    for _ in range(4):
        if cur_coords not in seen:
            seen.add(cur_coords)
            rotations.append((cur_coords, cur_h, cur_w))
        cur_coords, cur_h, cur_w = _rotate_once(cur_coords, cur_h, cur_w)
    return rotations


def _edge_components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if not seen[nr, nc] and mask[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_legend(grid: np.ndarray, bg: int):
    h, w = grid.shape
    border = np.zeros_like(grid, dtype=bool)
    border[0, :] = True
    border[h - 1, :] = True
    border[:, 0] = True
    border[:, w - 1] = True
    edge_mask = border & (grid != bg)

    candidates = []
    for comp in _edge_components(edge_mask):
        rows = {r for r, _ in comp}
        cols = {c for _, c in comp}
        if len(rows) == 1:
            row = next(iter(rows))
            ordered = sorted(c for _, c in comp)
            if ordered == list(range(ordered[0], ordered[-1] + 1)):
                candidates.append(("h", row, ordered))
        elif len(cols) == 1:
            col = next(iter(cols))
            ordered = sorted(r for r, _ in comp)
            if ordered == list(range(ordered[0], ordered[-1] + 1)):
                candidates.append(("v", col, ordered))

    assert candidates, "expected a border legend"
    candidates.sort(key=lambda item: (-len(item[2]), item[0], item[1], item[2][0]))
    orientation, fixed, ordered = candidates[0]

    legend_mask = np.zeros_like(grid, dtype=bool)
    legend = []
    if orientation == "h":
        for c in ordered:
            legend_mask[fixed, c] = True
            legend.append(int(grid[fixed, c]))
    else:
        for r in ordered:
            legend_mask[r, fixed] = True
            legend.append(int(grid[r, fixed]))

    assert len(legend) >= 2, "legend must contain at least two colors"
    assert len(set(legend)) == len(legend), "legend colors must be unique"
    return legend, legend_mask


def _find_fragments(grid: np.ndarray, bg: int, ignore_mask: np.ndarray):
    h, w = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    comps = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or ignore_mask[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if seen[nr, nc] or ignore_mask[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            norm, sh, sw = _normalize(coords)
            rs = [rr for rr, _ in coords]
            cs = [cc for _, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rs), min(cs), max(rs), max(cs)),
                    "shape": norm,
                    "shape_h": sh,
                    "shape_w": sw,
                    "area": len(coords),
                    "rotations": _unique_rotations(norm, sh, sw),
                }
            )

    return comps


def _rectangle_dims_for_pair(
    shape_a: frozenset[tuple[int, int]],
    h_a: int,
    w_a: int,
    shape_b: frozenset[tuple[int, int]],
    h_b: int,
    w_b: int,
):
    total = len(shape_a) + len(shape_b)
    dims = set()

    for factor in range(1, int(total ** 0.5) + 1):
        if total % factor != 0:
            continue
        for rect_h, rect_w in {(factor, total // factor), (total // factor, factor)}:
            if rect_h < h_a or rect_w < w_a or rect_h < h_b or rect_w < w_b:
                continue
            rect_cells = {(r, c) for r in range(rect_h) for c in range(rect_w)}
            for off_r in range(rect_h - h_a + 1):
                for off_c in range(rect_w - w_a + 1):
                    placed_a = {(r + off_r, c + off_c) for r, c in shape_a}
                    complement = rect_cells - placed_a
                    if len(complement) != len(shape_b):
                        continue
                    min_r = min(r for r, _ in complement)
                    min_c = min(c for _, c in complement)
                    max_r = max(r for r, _ in complement)
                    max_c = max(c for _, c in complement)
                    if max_r - min_r + 1 != h_b or max_c - min_c + 1 != w_b:
                        continue
                    norm_comp = frozenset((r - min_r, c - min_c) for r, c in complement)
                    if norm_comp == shape_b:
                        dims.add((rect_h, rect_w))

    return dims


def _pair_info(comp_a, comp_b):
    dims = set()
    for rot_shape, rot_h, rot_w in comp_b["rotations"]:
        dims |= _rectangle_dims_for_pair(
            comp_a["shape"],
            comp_a["shape_h"],
            comp_a["shape_w"],
            rot_shape,
            rot_h,
            rot_w,
        )
    if len(dims) != 1:
        return None
    rect_h, rect_w = next(iter(dims))
    return {"height": rect_h, "width": rect_w, "area": rect_h * rect_w}


def _unique_matching(components):
    count = len(components)
    assert count > 0 and count % 2 == 0, "fragment count must be positive and even"

    pair_map = {}
    neighbors = [[] for _ in range(count)]
    for i in range(count):
        for j in range(i + 1, count):
            info = _pair_info(components[i], components[j])
            if info is None:
                continue
            pair_map[(i, j)] = info
            neighbors[i].append(j)
            neighbors[j].append(i)

    solutions = []
    current = []
    used = [False] * count

    def search():
        if len(solutions) >= 2:
            return
        unused = [idx for idx in range(count) if not used[idx]]
        if not unused:
            solutions.append(current.copy())
            return
        pivot = min(unused, key=lambda idx: sum(not used[j] for j in neighbors[idx]))
        options = [j for j in sorted(neighbors[pivot]) if not used[j]]
        if not options:
            return

        used[pivot] = True
        for j in options:
            used[j] = True
            key = (pivot, j) if pivot < j else (j, pivot)
            i0, j0 = key
            current.append((i0, j0, pair_map[key]))
            search()
            current.pop()
            used[j] = False
        used[pivot] = False

    search()
    assert len(solutions) == 1, "expected exactly one valid fragment pairing"
    return solutions[0]


def _pair_output_color(color_a: int, color_b: int, legend: list[int]) -> int:
    pos = {color: idx for idx, color in enumerate(legend)}
    assert color_a in pos and color_b in pos, "fragment colors must appear in legend"
    return int(legend[(pos[color_a] + pos[color_b]) // 2])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair scattered fragments into complete rectangles, then recolor each rectangle by midpoint lookup on an edge legend.

    Input structure:
    - bg = the dominant background color.
    - Away from the border legend, the grid contains an even number of small monochrome 4-connected fragments.
    - Fragments come in pairs whose two shapes are complementary parts of a solid rectangle, after optionally rotating one fragment by 0/90/180/270 degrees.
    - One grid edge contains a contiguous horizontal or vertical color-band legend whose visible order defines the color sequence.

    Rule:
    - Detect the edge legend and remove it from fragment parsing.
    - Pair the remaining fragments so that each pair can be translated, with one member optionally rotated, to tile a gap-free solid rectangle.
    - For each pair, locate both fragment colors in the legend and choose the color at the floor midpoint of their legend positions; this matches the same-color case, the left/upper color for adjacent entries, and the exact middle color for wider even gaps.
    - Sort the reconstructed rectangles by decreasing area and then by their source top/left position, and stack them tightly from top to bottom, left-aligned, filling each rectangle uniformly with its derived output color.

    Key insight:
    - The puzzle composes three independent cues: geometric complementarity determines the pairing, the legend order determines the output color, and the final arrangement depends only on the completed rectangle sizes rather than on the fragments' original layout.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    legend, legend_mask = _extract_legend(g, bg)
    components = _find_fragments(g, bg, legend_mask)
    matching = _unique_matching(components)

    rectangles = []
    for i, j, info in matching:
        comp_a = components[i]
        comp_b = components[j]
        out_color = _pair_output_color(comp_a["color"], comp_b["color"], legend)
        top = min(comp_a["bbox"][0], comp_b["bbox"][0])
        left = min(comp_a["bbox"][1], comp_b["bbox"][1])
        rectangles.append(
            {
                "height": info["height"],
                "width": info["width"],
                "area": info["area"],
                "top": top,
                "left": left,
                "color": out_color,
            }
        )

    rectangles.sort(key=lambda item: (-item["area"], item["top"], item["left"], item["height"], item["width"]))
    out_h = sum(item["height"] for item in rectangles)
    out_w = max(item["width"] for item in rectangles)
    out = np.full((out_h, out_w), bg, dtype=int)

    row = 0
    for item in rectangles:
        out[row : row + item["height"], : item["width"]] = item["color"]
        row += item["height"]
    return out


# --------------- generator ---------------

_RNG = np.random.default_rng()
_MAX_RETRIES = 400
_BRANCHES = ("same", "adjacent", "middle")
_GRID_MIN = 16
_GRID_MAX = 30
_PAIR_COUNT_CHOICES = (2, 3, 4)
_LEGEND_LEN_CHOICES = (5, 6, 7)


def _is_connected(coords) -> bool:
    coords = set(coords)
    start = next(iter(coords))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in coords and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(coords)


def _random_walk_sequence(length: int, max_cut: int):
    cur = int(_RNG.integers(1, max_cut + 1))
    seq = [cur]
    for _ in range(length - 1):
        step = int(_RNG.choice([-1, 0, 1]))
        cur = int(np.clip(cur + step, 1, max_cut))
        seq.append(cur)
    if len(set(seq)) == 1 and max_cut >= 2:
        idx = int(_RNG.integers(length))
        if seq[idx] == 1:
            seq[idx] = 2
        elif seq[idx] == max_cut:
            seq[idx] = max_cut - 1
        else:
            seq[idx] += int(_RNG.choice([-1, 1]))
    return seq


def _make_fragment_pair(rect_h: int, rect_w: int):
    for _ in range(80):
        if _RNG.random() < 0.5:
            cuts = _random_walk_sequence(rect_h, rect_w - 1)
            shape_a = {(r, c) for r in range(rect_h) for c in range(cuts[r])}
        else:
            cuts = _random_walk_sequence(rect_w, rect_h - 1)
            shape_a = {(r, c) for c in range(rect_w) for r in range(cuts[c])}
        rect_cells = {(r, c) for r in range(rect_h) for c in range(rect_w)}
        shape_b = rect_cells - shape_a
        if min(len(shape_a), len(shape_b)) < 2:
            continue
        if not _is_connected(shape_a) or not _is_connected(shape_b):
            continue
        norm_a, h_a, w_a = _normalize(shape_a)
        norm_b, h_b, w_b = _normalize(shape_b)
        return (norm_a, h_a, w_a), (norm_b, h_b, w_b)
    raise RuntimeError("failed to construct complementary connected fragments")


def _sample_pair_colors(legend: list[int], branch: str):
    n = len(legend)
    if branch == "same":
        idx = int(_RNG.integers(n))
        return int(legend[idx]), int(legend[idx])
    if branch == "adjacent":
        idx = int(_RNG.integers(n - 1))
        colors = [int(legend[idx]), int(legend[idx + 1])]
    else:
        gaps = [gap for gap in range(2, n) if gap % 2 == 0]
        if not gaps:
            idx = int(_RNG.integers(n - 1))
            colors = [int(legend[idx]), int(legend[idx + 1])]
        else:
            gap = int(_RNG.choice(gaps))
            idx = int(_RNG.integers(0, n - gap))
            colors = [int(legend[idx]), int(legend[idx + gap])]
    if _RNG.random() < 0.5:
        colors.reverse()
    return colors[0], colors[1]


def _mark_blocked(blocked: np.ndarray, coords):
    h, w = blocked.shape
    for r, c in coords:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                blocked[rr, cc] = True


def _try_place_shape(grid: np.ndarray, blocked: np.ndarray, shape, sh: int, sw: int, color: int) -> bool:
    h, w = grid.shape
    if h - sh <= 1 or w - sw <= 1:
        return False
    for _ in range(200):
        top = int(_RNG.integers(1, h - sh))
        left = int(_RNG.integers(1, w - sw))
        coords = [(top + r, left + c) for r, c in shape]
        if any(blocked[r, c] for r, c in coords):
            continue
        for r, c in coords:
            grid[r, c] = color
        _mark_blocked(blocked, coords)
        return True
    return False


def _place_legend(grid: np.ndarray, blocked: np.ndarray, legend: list[int]):
    h, w = grid.shape
    edge = str(_RNG.choice(["top", "bottom", "left", "right"]))
    coords = []
    if edge in ("top", "bottom"):
        row = 0 if edge == "top" else h - 1
        start = int(_RNG.integers(0, w - len(legend) + 1))
        for offset, color in enumerate(legend):
            coord = (row, start + offset)
            coords.append(coord)
            grid[coord] = color
    else:
        col = 0 if edge == "left" else w - 1
        start = int(_RNG.integers(0, h - len(legend) + 1))
        for offset, color in enumerate(legend):
            coord = (start + offset, col)
            coords.append(coord)
            grid[coord] = color
    _mark_blocked(blocked, coords)


def _planned_branches(pair_count: int):
    branches = []
    if pair_count >= 3:
        perm = list(_BRANCHES)
        _RNG.shuffle(perm)
        branches.extend(perm)
    while len(branches) < pair_count:
        branches.append(str(_RNG.choice(_BRANCHES)))
    _RNG.shuffle(branches)
    return branches[:pair_count]


def _gen_attempt():
    bg = int(_RNG.integers(10))
    available = [c for c in range(10) if c != bg]
    legend_len = int(_RNG.choice(_LEGEND_LEN_CHOICES))
    legend = [int(c) for c in _RNG.choice(available, size=legend_len, replace=False)]

    pair_count = int(_RNG.choice(_PAIR_COUNT_CHOICES, p=[0.2, 0.45, 0.35]))
    branches = _planned_branches(pair_count)

    pair_specs = []
    for branch in branches:
        for _ in range(60):
            rect_h = int(_RNG.integers(2, 7))
            rect_w = int(_RNG.integers(2, 7))
            if rect_h * rect_w < 6:
                continue
            if rect_h + rect_w > 10 and _RNG.random() < 0.5:
                continue
            try:
                frag_a, frag_b = _make_fragment_pair(rect_h, rect_w)
            except RuntimeError:
                continue
            color_a, color_b = _sample_pair_colors(legend, branch)
            rot_a = int(_RNG.integers(4))
            rot_b = int(_RNG.integers(4))
            piece_a = _rotate_k(*frag_a, rot_a)
            piece_b = _rotate_k(*frag_b, rot_b)
            pair_specs.append(
                {
                    "branch": branch,
                    "rect": (rect_h, rect_w),
                    "pieces": [
                        {"shape": piece_a[0], "h": piece_a[1], "w": piece_a[2], "color": color_a},
                        {"shape": piece_b[0], "h": piece_b[1], "w": piece_b[2], "color": color_b},
                    ],
                }
            )
            break
        else:
            return None

    grid_h = int(_RNG.integers(_GRID_MIN, _GRID_MAX + 1))
    grid_w = int(_RNG.integers(_GRID_MIN, _GRID_MAX + 1))
    grid = np.full((grid_h, grid_w), bg, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    _place_legend(grid, blocked, legend)

    pieces = []
    for spec in pair_specs:
        pieces.extend(spec["pieces"])
    pieces.sort(key=lambda item: (-len(item["shape"]), -item["h"] * item["w"]))

    for piece in pieces:
        if not _try_place_shape(grid, blocked, piece["shape"], piece["h"], piece["w"], piece["color"]):
            return None

    try:
        legend_out, legend_mask = _extract_legend(grid, bg)
        if legend_out != legend:
            return None
        components = _find_fragments(grid, bg, legend_mask)
        if len(components) != 2 * pair_count:
            return None
        _unique_matching(components)
        out = main(grid)
        if out.ndim != 2 or max(out.shape) > 30:
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_MAX_RETRIES} attempts")
