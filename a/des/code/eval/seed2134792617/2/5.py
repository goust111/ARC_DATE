import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BG = 0
FRAME_COLOR = 5
REGION_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
RNG = np.random.default_rng()
GEN_RETRIES = 500


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    comps: list[list[tuple[int, int]]] = []
    cid = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            cid += 1
            stack = [(r, c)]
            labels[r, c] = cid
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = cid
                        stack.append((nr, nc))
            comps.append(coords)
    return labels, comps


def _normalize_coords(coords: list[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], tuple[int, int], tuple[int, int]]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    norm = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    return norm, (int(r0), int(c0)), (int(r1 - r0 + 1), int(c1 - c0 + 1))


def _adjacent_colors(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, frame_color: int) -> set[int]:
    h, w = grid.shape
    out: set[int] = set()
    for r, c in coords:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            val = int(grid[nr, nc])
            if val != bg and val != frame_color:
                out.add(val)
    return out


def _coords_has_hole(coords: list[tuple[int, int]]) -> bool:
    norm, _, (h, w) = _normalize_coords(coords)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    outside = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []
    for r in range(h):
        for c in [0, w - 1]:
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in [0, h - 1]:
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                stack.append((nr, nc))
    return bool(np.any((~mask) & (~outside)))


def _find_frame_bbox(grid: np.ndarray, bg: int) -> tuple[int, int, int, int, int]:
    candidates: list[tuple[int, int, int, int, int, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        pts = np.argwhere(grid == color)
        if len(pts) < 8:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 4 or w < 4:
            continue
        sub = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        border = np.zeros_like(sub, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if np.all(sub[border] == color) and not np.any(sub[1:-1, 1:-1] == color):
            perimeter = 2 * h + 2 * w - 4
            if int(np.count_nonzero(sub == color)) == perimeter:
                candidates.append((perimeter, h * w, int(r0), int(c0), int(r1), int(c1), color))
    if not candidates:
        raise AssertionError("frame not found")
    _, _, r0, c0, r1, c1, color = max(candidates)
    return r0, c0, r1, c1, color


def _extract_regions(frame: np.ndarray, bg: int, frame_color: int) -> list[dict]:
    labels, comps = _connected_components(frame == bg)
    regions = []
    for idx, coords in enumerate(comps, start=1):
        shape, _, _ = _normalize_coords(coords)
        regions.append(
            {
                "index": idx,
                "coords": coords,
                "shape": shape,
                "adj_colors": _adjacent_colors(frame, coords, bg, frame_color),
            }
        )
    return regions


def _extract_fragments(grid: np.ndarray, bg: int, frame_bbox: tuple[int, int, int, int]) -> list[dict]:
    r0, c0, r1, c1 = frame_bbox
    mask = grid != bg
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    _, comps = _connected_components(mask)
    fragments = []
    for coords in comps:
        colors = {int(grid[r, c]) for r, c in coords}
        if len(colors) != 1:
            raise AssertionError("fragment must be monochrome")
        color = next(iter(colors))
        shape, _, _ = _normalize_coords(coords)
        fragments.append({"coords": coords, "shape": shape, "color": int(color)})
    return fragments


def _search_assignment(regions: list[dict], fragments: list[dict], candidate_lists: list[list[int]]) -> dict[int, int] | None:
    order = sorted(range(len(regions)), key=lambda idx: (len(candidate_lists[idx]), len(regions[idx]["coords"])))
    used: set[int] = set()
    chosen: dict[int, int] = {}

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        ri = order[pos]
        cands = sorted(
            candidate_lists[ri],
            key=lambda fi: (
                fragments[fi]["color"] not in regions[ri]["adj_colors"],
                len(fragments[fi]["coords"]),
            ),
        )
        for fi in cands:
            if fi in used:
                continue
            used.add(fi)
            chosen[ri] = fi
            if dfs(pos + 1):
                return True
            used.remove(fi)
            del chosen[ri]
        return False

    return chosen if dfs(0) else None


def _match_regions(regions: list[dict], fragments: list[dict]) -> dict[int, int]:
    if len(regions) != len(fragments):
        raise AssertionError("region count must equal fragment count")
    exact_lists: list[list[int]] = []
    pref_lists: list[list[int]] = []
    for region in regions:
        exact = [i for i, frag in enumerate(fragments) if frag["shape"] == region["shape"]]
        if not exact:
            raise AssertionError("every region must have a matching fragment shape")
        preferred = [i for i in exact if fragments[i]["color"] in region["adj_colors"]]
        exact_lists.append(exact)
        pref_lists.append(preferred if preferred else exact)
    mapping = _search_assignment(regions, fragments, pref_lists)
    if mapping is None:
        mapping = _search_assignment(regions, fragments, exact_lists)
    if mapping is None:
        raise AssertionError("failed to assign fragments to regions")
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)

    fr0, fc0, fr1, fc1, frame_color = _find_frame_bbox(g, bg)
    frame = g[fr0 : fr1 + 1, fc0 : fc1 + 1].copy()

    regions = _extract_regions(frame, bg, frame_color)
    fragments = _extract_fragments(g, bg, (fr0, fc0, fr1, fc1))
    mapping = _match_regions(regions, fragments)

    out = frame.copy()
    for ri, region in enumerate(regions):
        frag = fragments[mapping[ri]]
        for r, c in region["coords"]:
            out[r, c] = int(frag["color"])
    return out


def _split_rectangles(cell_h: int, cell_w: int, leaf_target: int) -> list[tuple[int, int, int, int]] | None:
    leaves = [(0, 0, cell_h, cell_w)]
    while len(leaves) < leaf_target:
        candidates = [i for i, (_r0, _c0, h, w) in enumerate(leaves) if h > 1 or w > 1]
        if not candidates:
            return None
        weights = np.array([leaves[i][2] * leaves[i][3] for i in candidates], dtype=float)
        pick_idx = int(candidates[int(RNG.choice(len(candidates), p=weights / weights.sum()))])
        r0, c0, h, w = leaves.pop(pick_idx)
        options = []
        if h > 1:
            for cut in range(1, h):
                options.append(("h", cut))
        if w > 1:
            for cut in range(1, w):
                options.append(("v", cut))
        if not options:
            return None
        orient, cut = options[int(RNG.integers(len(options)))]
        if orient == "h":
            leaves.append((r0, c0, cut, w))
            leaves.append((r0 + cut, c0, h - cut, w))
        else:
            leaves.append((r0, c0, h, cut))
            leaves.append((r0, c0 + cut, h, w - cut))
    return leaves


def _partition_labels(cell_h: int, cell_w: int, region_count: int, leaf_target: int) -> np.ndarray | None:
    leaves = _split_rectangles(cell_h, cell_w, leaf_target)
    if leaves is None:
        return None
    leaf_grid = np.full((cell_h, cell_w), -1, dtype=int)
    for leaf_id, (r0, c0, h, w) in enumerate(leaves):
        leaf_grid[r0 : r0 + h, c0 : c0 + w] = leaf_id

    parent = list(range(len(leaves)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    groups = len(leaves)
    while groups > region_count:
        pairs = set()
        for r in range(cell_h):
            for c in range(cell_w - 1):
                a = find(int(leaf_grid[r, c]))
                b = find(int(leaf_grid[r, c + 1]))
                if a != b:
                    pairs.add(tuple(sorted((a, b))))
        for r in range(cell_h - 1):
            for c in range(cell_w):
                a = find(int(leaf_grid[r, c]))
                b = find(int(leaf_grid[r + 1, c]))
                if a != b:
                    pairs.add(tuple(sorted((a, b))))
        if not pairs:
            return None
        pair_list = list(pairs)
        a, b = pair_list[int(RNG.integers(len(pair_list)))]
        if union(a, b):
            groups -= 1

    roots = sorted({find(i) for i in range(len(leaves))})
    perm = list(RNG.permutation(len(roots)))
    root_to_region = {root: perm[idx] for idx, root in enumerate(roots)}
    labels = np.zeros((cell_h, cell_w), dtype=int)
    for r in range(cell_h):
        for c in range(cell_w):
            labels[r, c] = root_to_region[find(int(leaf_grid[r, c]))]
    return labels


def _build_wall_mask(labels: np.ndarray) -> np.ndarray:
    cell_h, cell_w = labels.shape
    inner_h = 2 * cell_h - 1
    inner_w = 2 * cell_w - 1
    wall = np.zeros((inner_h, inner_w), dtype=bool)

    for r in range(cell_h):
        for c in range(cell_w - 1):
            if int(labels[r, c]) != int(labels[r, c + 1]):
                wall[2 * r, 2 * c + 1] = True
    for r in range(cell_h - 1):
        for c in range(cell_w):
            if int(labels[r, c]) != int(labels[r + 1, c]):
                wall[2 * r + 1, 2 * c] = True
    for r in range(cell_h - 1):
        for c in range(cell_w - 1):
            tl = int(labels[r, c])
            tr = int(labels[r, c + 1])
            bl = int(labels[r + 1, c])
            br = int(labels[r + 1, c + 1])
            has_vertical = (tl != tr) or (bl != br)
            has_horizontal = (tl != bl) or (tr != br)
            if has_vertical or has_horizontal:
                wall[2 * r + 1, 2 * c + 1] = True
    return wall


def _render_template(labels: np.ndarray, region_colors: list[int]) -> tuple[np.ndarray, dict[int, list[tuple[int, int]]], dict[int, int]] | None:
    cell_h, cell_w = labels.shape
    wall = _build_wall_mask(labels)
    comp_labels, comps = _connected_components(~wall)
    region_count = len(region_colors)
    if len(comps) != region_count:
        return None

    region_coords: dict[int, list[tuple[int, int]]] = {}
    region_areas: dict[int, int] = {}
    for region_id in range(region_count):
        seed = np.argwhere(labels == region_id)
        if len(seed) == 0:
            return None
        sr, sc = map(int, seed[0])
        cid = int(comp_labels[2 * sr, 2 * sc])
        if cid == 0:
            return None
        coords = [(r + 1, c + 1) for r, c in comps[cid - 1]]
        region_coords[region_id] = coords
        region_areas[region_id] = len(coords)

    inner_h, inner_w = wall.shape
    inner = np.full((inner_h, inner_w), BG, dtype=int)
    for r in range(cell_h):
        for c in range(cell_w - 1):
            a = int(labels[r, c])
            b = int(labels[r, c + 1])
            if a == b:
                continue
            owner = a if region_areas[a] > region_areas[b] else b
            inner[2 * r, 2 * c + 1] = int(region_colors[owner])
    for r in range(cell_h - 1):
        for c in range(cell_w):
            a = int(labels[r, c])
            b = int(labels[r + 1, c])
            if a == b:
                continue
            owner = a if region_areas[a] > region_areas[b] else b
            inner[2 * r + 1, 2 * c] = int(region_colors[owner])
    for r in range(cell_h - 1):
        for c in range(cell_w - 1):
            if not wall[2 * r + 1, 2 * c + 1]:
                continue
            ids = {int(labels[r, c]), int(labels[r, c + 1]), int(labels[r + 1, c]), int(labels[r + 1, c + 1])}
            owner = max(ids, key=lambda rid: region_areas[rid])
            inner[2 * r + 1, 2 * c + 1] = int(region_colors[owner])

    crop = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    crop[0, :] = FRAME_COLOR
    crop[-1, :] = FRAME_COLOR
    crop[:, 0] = FRAME_COLOR
    crop[:, -1] = FRAME_COLOR
    crop[1:-1, 1:-1] = inner
    return crop, region_coords, region_areas


def _can_place_shape(coords: tuple[tuple[int, int], ...], top: int, left: int, occ: np.ndarray) -> bool:
    h, w = occ.shape
    for r, c in coords:
        rr = top + r
        cc = left + c
        if rr < 0 or cc < 0 or rr >= h or cc >= w:
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and occ[nr, nc]:
                    return False
    return True


def _try_pack_fragments(specs: list[dict], area_h: int, area_w: int) -> list[tuple[int, int]] | None:
    occ = np.zeros((area_h, area_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(specs)
    order = sorted(range(len(specs)), key=lambda i: (-len(specs[i]["shape"]), -(specs[i]["dims"][0] * specs[i]["dims"][1])))
    for idx in order:
        spec = specs[idx]
        sh, sw = spec["dims"]
        candidates = [(r, c) for r in range(area_h - sh + 1) for c in range(area_w - sw + 1)]
        if not candidates:
            return None
        for pick in RNG.permutation(len(candidates)):
            top, left = candidates[int(pick)]
            if not _can_place_shape(spec["shape"], int(top), int(left), occ):
                continue
            placements[idx] = (int(top), int(left))
            for r, c in spec["shape"]:
                occ[int(top + r), int(left + c)] = True
            break
        if placements[idx] is None:
            return None
    return [(int(r), int(c)) for r, c in placements]


def _pack_fragments(specs: list[dict], frame_shape: tuple[int, int]) -> tuple[int, int, list[tuple[int, int]]] | None:
    frame_h, frame_w = frame_shape
    max_h = max(spec["dims"][0] for spec in specs)
    max_w = max(spec["dims"][1] for spec in specs)
    sum_h = sum(spec["dims"][0] for spec in specs)
    sum_w = sum(spec["dims"][1] for spec in specs)
    mode = int(RNG.integers(3))
    for extra in range(12):
        if mode == 0:
            area_h = max(frame_h + 1, sum_h + len(specs) + extra)
            area_w = max(frame_w - 1, max_w + 6 + extra)
        elif mode == 1:
            area_h = max(frame_h - 1, max_h + 6 + extra)
            area_w = max(frame_w + 1, sum_w + len(specs) + extra)
        else:
            area_h = max(frame_h + 1, max_h * 2 + 4 + extra)
            area_w = max(frame_w + 1, max_w * 2 + 4 + extra)
        placements = _try_pack_fragments(specs, int(area_h), int(area_w))
        if placements is not None:
            return int(area_h), int(area_w), placements
    return None


def _gen_attempt() -> np.ndarray | None:
    region_count = int(RNG.integers(3, 6))
    cell_h = int(RNG.integers(4, 7))
    cell_w = int(RNG.integers(5, 9))
    extra_leaves = int(RNG.integers(0, 4))
    leaf_target = min(cell_h * cell_w, region_count + extra_leaves)
    labels = _partition_labels(cell_h, cell_w, region_count, leaf_target)
    if labels is None:
        return None

    region_colors = list(RNG.choice(REGION_COLORS, size=region_count, replace=False))
    rendered = _render_template(labels, region_colors)
    if rendered is None:
        return None
    crop, region_coords, region_areas = rendered

    if len(set(region_areas.values())) != region_count:
        return None

    shape_keys = []
    fragment_specs = []
    for region_id in range(region_count):
        coords = region_coords[region_id]
        if len(coords) < 3 or _coords_has_hole(coords):
            return None
        shape, _, dims = _normalize_coords(coords)
        shape_keys.append(shape)
        fragment_specs.append({"shape": shape, "dims": dims, "color": int(region_colors[region_id])})
    if len(set(shape_keys)) != region_count:
        return None

    packed = _pack_fragments(fragment_specs, tuple(map(int, crop.shape)))
    if packed is None:
        return None
    area_h, area_w, placements = packed

    left_half_w = int(crop.shape[1] + RNG.integers(2, 5))
    right_half_w = int(area_w + RNG.integers(2, 5))
    grid_w = left_half_w + right_half_w
    grid_h = int(max(crop.shape[0], area_h) + RNG.integers(2, 6))
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    frame_r0 = int(RNG.integers(1, grid_h - crop.shape[0] + 1))
    frame_c0 = int(RNG.integers(1, left_half_w - crop.shape[1] + 1))
    grid[frame_r0 : frame_r0 + crop.shape[0], frame_c0 : frame_c0 + crop.shape[1]] = crop

    frag_r0 = int(RNG.integers(1, grid_h - area_h + 1))
    frag_c0 = int(left_half_w + RNG.integers(1, right_half_w - area_w + 1))
    for spec_idx in RNG.permutation(region_count):
        top, left = placements[spec_idx]
        color = fragment_specs[spec_idx]["color"]
        for r, c in fragment_specs[spec_idx]["shape"]:
            grid[frag_r0 + top + r, frag_c0 + left + c] = int(color)

    if _most_common_color(grid) != BG:
        return None

    expected = crop.copy()
    for region_id in range(region_count):
        for r, c in region_coords[region_id]:
            expected[r, c] = int(region_colors[region_id])

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
