import random
import numpy as np


BG = 8
GEN_RETRIES = 400
GRID_H_RANGE = (17, 24)
GRID_W_MAX = 34
HOLE_COUNT_RANGE = (2, 4)
HOLE_H_RANGE = (2, 5)
HOLE_W_RANGE = (2, 6)
TEMPLATE_H_RANGE = (11, 16)
TEMPLATE_W_RANGE = (10, 15)
COLOR_POOL = [0, 1, 2, 3, 4, 5, 6, 7, 9]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            comp: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_edge(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _border_mode(grid: np.ndarray) -> int:
    border = np.concatenate([grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if BG in vals:
        return BG
    return int(vals[int(cnts.argmax())])


def _rect_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _hole_signature(grid: np.ndarray, box: tuple[int, int, int, int]) -> tuple[tuple[int, ...], ...]:
    r0, c0, r1, c1 = box
    top = tuple(int(v) for v in grid[r0 - 1, c0 : c1 + 1])
    right = tuple(int(v) for v in grid[r0 : r1 + 1, c1 + 1])
    bottom = tuple(int(v) for v in grid[r1 + 1, c0 : c1 + 1])
    left = tuple(int(v) for v in grid[r0 : r1 + 1, c0 - 1])
    return top, right, bottom, left


def _patch_signature(patch: np.ndarray) -> tuple[tuple[int, ...], ...]:
    top = tuple(int(v) for v in patch[0, :])
    right = tuple(int(v) for v in patch[:, -1])
    bottom = tuple(int(v) for v in patch[-1, :])
    left = tuple(int(v) for v in patch[:, 0])
    return top, right, bottom, left


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _border_mode(g)

    non_bg = _components(g != bg)
    assert non_bg, "expected a template and fragments"
    infos = []
    for coords in non_bg:
        box = _bbox(coords)
        infos.append({"coords": coords, "bbox": box, "area": len(coords)})

    template = max(infos, key=lambda info: (info["area"], -info["bbox"][1], -info["bbox"][0]))
    template_set = set(template["coords"])

    holes = []
    for coords in _components(g == bg):
        if _touches_edge(coords, h, w):
            continue
        box = _bbox(coords)
        if _rect_area(box) != len(coords):
            continue
        r0, c0, r1, c1 = box
        ring = (
            [(r0 - 1, c) for c in range(c0, c1 + 1)]
            + [(r, c1 + 1) for r in range(r0, r1 + 1)]
            + [(r1 + 1, c) for c in range(c0, c1 + 1)]
            + [(r, c0 - 1) for r in range(r0, r1 + 1)]
        )
        if any(pos not in template_set for pos in ring):
            continue
        holes.append({"bbox": box, "sig": _hole_signature(g, box)})

    assert holes, "expected rectangular holes inside the template"
    hole_sizes = {
        (box[2] - box[0] + 1, box[3] - box[1] + 1)
        for box in (hole["bbox"] for hole in holes)
    }

    fragments = []
    for info in infos:
        if info is template:
            continue
        box = info["bbox"]
        hh = box[2] - box[0] + 1
        ww = box[3] - box[1] + 1
        if info["area"] != hh * ww:
            continue
        if (hh, ww) not in hole_sizes:
            continue
        r0, c0, r1, c1 = box
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        fragments.append({"bbox": box, "patch": patch, "sig": _patch_signature(patch)})

    assert len(fragments) == len(holes), "expected one fragment per hole"
    frag_by_sig: dict[tuple[tuple[int, ...], ...], list[dict[str, object]]] = {}
    for frag in fragments:
        frag_by_sig.setdefault(frag["sig"], []).append(frag)

    out = g.copy()
    for hole in holes:
        matches = frag_by_sig.get(hole["sig"], [])
        assert len(matches) == 1, "each hole must match exactly one fragment"
        frag = matches[0]
        r0, c0, r1, c1 = hole["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = frag["patch"]
        fr0, fc0, fr1, fc1 = frag["bbox"]
        out[fr0 : fr1 + 1, fc0 : fc1 + 1] = bg

    return out


def _pattern_patch(h: int, w: int, palette: list[int]) -> np.ndarray:
    k = len(palette)
    shift = random.randrange(k)
    mode = random.choice(["diag", "bands", "checker", "mix"])
    patch = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            if mode == "diag":
                idx = (r + 2 * c + shift) % k
            elif mode == "bands":
                idx = ((r // 2) + c + shift) % k
            elif mode == "checker":
                idx = ((r // 2) + (c // 2) + shift) % k
            else:
                idx = (2 * r + 3 * c + shift) % k
            patch[r, c] = palette[idx]
    return patch


def _make_fragment_patch(h: int, w: int) -> np.ndarray:
    palette = random.sample(COLOR_POOL, random.randint(3, min(6, len(COLOR_POOL))))
    patch = _pattern_patch(h, w, palette)

    tl = random.choice(palette)
    tr = random.choice(palette)
    bl = random.choice(palette)
    br = random.choice(palette)
    top = [tl] + [random.choice(palette) for _ in range(max(0, w - 2))] + [tr]
    bottom = [bl] + [random.choice(palette) for _ in range(max(0, w - 2))] + [br]
    left = [tl] + [random.choice(palette) for _ in range(max(0, h - 2))] + [bl]
    right = [tr] + [random.choice(palette) for _ in range(max(0, h - 2))] + [br]

    patch[0, :] = np.array(top, dtype=int)
    patch[-1, :] = np.array(bottom, dtype=int)
    patch[:, 0] = np.array(left, dtype=int)
    patch[:, -1] = np.array(right, dtype=int)
    return patch


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _sample_hole_sizes(n_holes: int) -> list[tuple[int, int]]:
    choices = [
        (h, w)
        for h in range(HOLE_H_RANGE[0], HOLE_H_RANGE[1] + 1)
        for w in range(HOLE_W_RANGE[0], HOLE_W_RANGE[1] + 1)
    ]
    return random.sample(choices, n_holes)


def _place_holes(template_h: int, template_w: int, sizes: list[tuple[int, int]]) -> list[dict[str, tuple[int, int, int, int]]] | None:
    placements: list[dict[str, tuple[int, int, int, int]]] = []
    order = sizes[:]
    random.shuffle(order)
    for h, w in order:
        if template_h < h + 4 or template_w < w + 4:
            return None
        placed = False
        for _ in range(200):
            r0 = random.randint(1, template_h - h - 2)
            c0 = random.randint(1, template_w - w - 2)
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            expanded = (r0 - 1, c0 - 1, r0 + h, c0 + w)
            if any(_boxes_overlap(expanded, item["expanded"]) for item in placements):
                continue
            placements.append({"hole": box, "expanded": expanded})
            placed = True
            break
        if not placed:
            return None
    return placements


def _place_fragments(
    grid_h: int,
    grid_w: int,
    min_col: int,
    patches: list[np.ndarray],
) -> list[dict[str, object]] | None:
    placements: list[dict[str, object]] = []
    blocked: list[tuple[int, int, int, int]] = []
    order = patches[:]
    random.shuffle(order)
    for patch in order:
        h, w = patch.shape
        if min_col > grid_w - w - 1:
            return None
        placed = False
        for _ in range(200):
            r0 = random.randint(1, grid_h - h - 1)
            c0 = random.randint(min_col, grid_w - w - 1)
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            expanded = (r0 - 1, c0 - 1, r0 + h, c0 + w)
            if any(_boxes_overlap(expanded, other) for other in blocked):
                continue
            blocked.append(expanded)
            placements.append({"bbox": box, "patch": patch})
            placed = True
            break
        if not placed:
            return None
    return placements


def _build_expected(
    grid: np.ndarray,
    holes_global: list[tuple[int, int, int, int]],
    fragment_places: list[dict[str, object]],
) -> np.ndarray:
    out = grid.copy()
    patch_by_sig = {
        _patch_signature(item["patch"]): item
        for item in fragment_places
    }
    for box in holes_global:
        sig = _hole_signature(grid, box)
        item = patch_by_sig[sig]
        r0, c0, r1, c1 = box
        out[r0 : r1 + 1, c0 : c1 + 1] = item["patch"]
        fr0, fc0, fr1, fc1 = item["bbox"]
        out[fr0 : fr1 + 1, fc0 : fc1 + 1] = BG
    return out


def _attempt_generate() -> np.ndarray | None:
    n_holes = random.randint(*HOLE_COUNT_RANGE)
    sizes = _sample_hole_sizes(n_holes)
    max_h = max(h for h, _ in sizes)
    max_w = max(w for _, w in sizes)

    template_h_min = max(TEMPLATE_H_RANGE[0], max_h + 4)
    template_w_min = max(TEMPLATE_W_RANGE[0], max_w + 4)
    if template_h_min > TEMPLATE_H_RANGE[1] or template_w_min > TEMPLATE_W_RANGE[1]:
        return None

    template_h = random.randint(template_h_min, TEMPLATE_H_RANGE[1])
    template_w = random.randint(template_w_min, TEMPLATE_W_RANGE[1])
    grid_h_min = max(GRID_H_RANGE[0], template_h + 2)
    if grid_h_min > GRID_H_RANGE[1]:
        return None
    grid_h = random.randint(grid_h_min, GRID_H_RANGE[1])
    grid_w_min = max(2 * template_w + 4, 24)
    if grid_w_min > GRID_W_MAX:
        return None
    grid_w = random.randint(grid_w_min, GRID_W_MAX)

    mid = grid_w // 2
    max_template_c0 = mid - template_w - 1
    if max_template_c0 < 1:
        return None
    template_r0 = random.randint(1, grid_h - template_h - 1)
    template_c0 = random.randint(1, max_template_c0)

    hole_places = _place_holes(template_h, template_w, sizes)
    if hole_places is None:
        return None

    template_colors = random.sample(COLOR_POOL, random.randint(3, 6))
    template = _pattern_patch(template_h, template_w, template_colors)

    patches: list[np.ndarray] = []
    holes_global: list[tuple[int, int, int, int]] = []
    for item in hole_places:
        r0, c0, r1, c1 = item["hole"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        patch = _make_fragment_patch(h, w)
        patches.append(patch)

        template[r0 - 1, c0 : c1 + 1] = patch[0, :]
        template[r1 + 1, c0 : c1 + 1] = patch[-1, :]
        template[r0 : r1 + 1, c0 - 1] = patch[:, 0]
        template[r0 : r1 + 1, c1 + 1] = patch[:, -1]
        template[r0 : r1 + 1, c0 : c1 + 1] = BG

        holes_global.append((template_r0 + r0, template_c0 + c0, template_r0 + r1, template_c0 + c1))

    fragment_places = _place_fragments(grid_h, grid_w, mid + 1, patches)
    if fragment_places is None:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[template_r0 : template_r0 + template_h, template_c0 : template_c0 + template_w] = template
    for item in fragment_places:
        r0, c0, r1, c1 = item["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = item["patch"]

    hole_sigs = [_hole_signature(grid, box) for box in holes_global]
    frag_sigs = [_patch_signature(item["patch"]) for item in fragment_places]
    if len(set(hole_sigs)) != n_holes:
        return None
    if sorted(hole_sigs) != sorted(frag_sigs):
        return None

    expected = _build_expected(grid, holes_global, fragment_places)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
