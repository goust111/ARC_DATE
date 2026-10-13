import random
from collections import deque

import numpy as np


BG = 0
TEMPLATE_COLORS = (1, 2, 3, 4, 6, 9)
DIVIDER = 5
QUERY = 7
OUTLINE = 8

ROT_OPS = ("r0", "r90", "r180", "r270")
REF_OPS = ("fh", "fv", "fd", "fa")
ALL_OPS = ROT_OPS + REF_OPS
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

LEGEND_HEIGHT = 5
TEMPLATE_GAP = 1
GRID_MIN = 22
GRID_MAX = 28
QUERY_COUNT_RANGE = (5, 8)
QUERY_BBOX_GAP = 2
GEN_ATTEMPTS = 400


def _normalize(coords):
    pts = [tuple(map(int, p)) for p in coords]
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _transform(coords, op):
    out = []
    for r, c in coords:
        r = int(r)
        c = int(c)
        if op == "r0":
            out.append((r, c))
        elif op == "r90":
            out.append((c, -r))
        elif op == "r180":
            out.append((-r, -c))
        elif op == "r270":
            out.append((-c, r))
        elif op == "fh":
            out.append((-r, c))
        elif op == "fv":
            out.append((r, -c))
        elif op == "fd":
            out.append((c, r))
        elif op == "fa":
            out.append((-c, -r))
        else:
            raise ValueError(f"unknown transform {op}")
    return _normalize(out)


def _variants(coords, ops):
    return {_transform(coords, op) for op in ops}


def _shape_dims(coords):
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return max(rs) + 1, max(cs) + 1


def _patch_from_shape(coords, color):
    h, w = _shape_dims(coords)
    patch = np.full((h, w), BG, dtype=int)
    rr, cc = np.array(coords, dtype=int).T
    patch[rr, cc] = int(color)
    return patch


def _bbox_from_coords(coords):
    rr = coords[:, 0]
    cc = coords[:, 1]
    return int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max())


def _components_of_color(grid, color):
    g = np.asarray(grid, dtype=int)
    mask = g == int(color)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _legend_templates(grid):
    divider_r, divider_c = _divider_corner(grid)
    legend = grid[:divider_r, :divider_c]
    templates = []
    for color in sorted(set(map(int, legend.flatten())) - {BG, DIVIDER}):
        for coords in _components_of_color(legend, color):
            templates.append(
                {
                    "color": int(color),
                    "shape": _normalize(coords),
                }
            )
    return templates


def _divider_corner(grid):
    coords = np.argwhere(grid == DIVIDER)
    if coords.size == 0:
        raise AssertionError("missing divider")
    r = int(coords[:, 0].max())
    c = int(coords[:, 1].max())
    if r < 1 or c < 1:
        raise AssertionError("invalid divider")
    if not np.all(grid[r, : c + 1] == DIVIDER):
        raise AssertionError("divider row must be solid")
    if not np.all(grid[: r + 1, c] == DIVIDER):
        raise AssertionError("divider column must be solid")
    return r, c


def _query_category(coords):
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    if r0 + r1 < c0 + c1:
        return "upper"
    if r0 + r1 > c0 + c1:
        return "lower"
    return "diagonal"


def _ops_for_category(category):
    if category == "upper":
        return ROT_OPS
    if category == "lower":
        return REF_OPS
    if category == "diagonal":
        return ALL_OPS
    raise ValueError(f"unknown category {category}")


def _match_template(shape, templates, category):
    matches = []
    allowed = _ops_for_category(category)
    for template in templates:
        if shape in _variants(template["shape"], allowed):
            matches.append(template)
    if len(matches) != 1:
        raise AssertionError(f"expected unique match for {category}, found {len(matches)}")
    return matches[0]


def _draw_outline(out, coords):
    h, w = out.shape
    sprite = {tuple(map(int, p)) for p in coords}
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    for r in range(max(0, r0 - 1), min(h - 1, r1 + 1) + 1):
        for c in range(max(0, c0 - 1), min(w - 1, c1 + 1) + 1):
            if int(out[r, c]) != BG:
                continue
            for dr, dc in NEIGHBORS4:
                if (r + dr, c + dc) in sprite:
                    out[r, c] = OUTLINE
                    break


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each orange query sprite by shape-matching against the legend under a diagonal-dependent transform set.

    Input structure:
    - A square black grid contains a grey L-shaped divider that isolates a top-left legend region.
    - The legend region holds several colored monochrome template sprites.
    - Outside the legend, all query sprites are orange 4-connected polyominoes.

    Rule:
    - Read the legend templates as shape/color entries.
    - For each orange query sprite, use its bbox center relative to the main diagonal to choose the allowed transforms:
      rotations only above the diagonal, reflections only below it, and all eight rigid transforms on the diagonal.
    - Match the query shape to exactly one legend template inside that restricted transform set and recolor the query to
      the template color.
    - If the bbox center lies exactly on the main diagonal, then after recoloring add a cyan one-pixel 4-neighbor
      outline on nearby black cells within the bbox expanded by one.

    Key insight:
    - The query position is not cosmetic: the main diagonal decides which symmetries are legal, and that can change
      which legend entry the same visible shape matches.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        raise AssertionError("expected a square 2D grid")
    _divider_corner(g)
    templates = _legend_templates(g)
    if not templates:
        raise AssertionError("legend must contain templates")

    out = g.copy()
    for coords in _components_of_color(g, QUERY):
        shape = _normalize(coords)
        category = _query_category(coords)
        template = _match_template(shape, templates, category)
        out[coords[:, 0], coords[:, 1]] = int(template["color"])
        if category == "diagonal":
            _draw_outline(out, coords)
    return out


def _polyomino_catalog():
    fixed = {1: {((0, 0),)}}
    for size in range(2, 6):
        shapes = set()
        for shape in fixed[size - 1]:
            cells = set(shape)
            for r, c in shape:
                for dr, dc in NEIGHBORS4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        shapes.add(_normalize(cells | {nxt}))
        fixed[size] = shapes

    classes = {}
    for size in range(3, 6):
        for shape in fixed[size]:
            orbit = tuple(sorted(_variants(shape, ALL_OPS)))
            key = orbit[0]
            if key in classes:
                continue
            classes[key] = {
                "key": key,
                "size": size,
                "orbit": orbit,
            }
    out = []
    for info in classes.values():
        orbit = [v for v in info["orbit"] if _shape_dims(v)[0] <= 4 and _shape_dims(v)[1] <= 4]
        if not orbit:
            continue
        info = dict(info)
        info["orbit"] = tuple(orbit)
        out.append(info)
    return out


_CLASS_INFOS = _polyomino_catalog()
_CHIRAL_CLASSES = [info for info in _CLASS_INFOS if _variants(info["key"], ROT_OPS).isdisjoint(_variants(info["key"], REF_OPS))]


def _bbox_gap_ok(bbox, existing, gap):
    r0, c0, r1, c1 = bbox
    for er0, ec0, er1, ec1 in existing:
        if r1 < er0:
            dr = er0 - r1 - 1
        elif er1 < r0:
            dr = r0 - er1 - 1
        else:
            dr = 0
        if c1 < ec0:
            dc = ec0 - c1 - 1
        elif ec1 < c0:
            dc = c0 - ec1 - 1
        else:
            dc = 0
        if max(dr, dc) < gap:
            return False
    return True


def _cells_clear(grid, r0, c0, h, w):
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(grid.shape[0], r0 + h + 1)
    cc1 = min(grid.shape[1], c0 + w + 1)
    return bool(np.all(grid[rr0:rr1, cc0:cc1] == BG))


def _category_position_ok(shape, r0, c0, category):
    h, w = _shape_dims(shape)
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    if category == "upper":
        return r1 < c0
    if category == "lower":
        return r0 > c1
    if r0 + r1 != c0 + c1:
        return False
    global_pts = [(r0 + r, c0 + c) for r, c in shape]
    return any(r < c for r, c in global_pts) and any(r > c for r, c in global_pts)


def _place_query(grid, shape, color, category, bboxes):
    h, w = _shape_dims(shape)
    size = grid.shape[0]
    candidates = []
    for r0 in range(1, size - h):
        for c0 in range(1, size - w):
            bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
            if not _category_position_ok(shape, r0, c0, category):
                continue
            if not _bbox_gap_ok(bbox, bboxes, QUERY_BBOX_GAP):
                continue
            if not _cells_clear(grid, r0, c0, h, w):
                continue
            candidates.append((r0, c0, bbox))
    if not candidates:
        return None
    r0, c0, bbox = random.choice(candidates)
    patch = _patch_from_shape(shape, color)
    grid[r0 : r0 + h, c0 : c0 + w][patch != BG] = int(color)
    coords = np.argwhere(patch != BG) + np.array([[r0, c0]])
    bboxes.append(bbox)
    return coords


def _pick_template_set():
    pair_class = random.choice(_CHIRAL_CLASSES)
    others = [info for info in _CHIRAL_CLASSES if info["key"] != pair_class["key"]]
    unique_c = random.choice(others)
    remaining = [info for info in _CLASS_INFOS if info["key"] not in {pair_class["key"], unique_c["key"]}]
    template_count = random.randint(3, 4)

    pair_a = random.choice(pair_class["orbit"])
    pair_b = _transform(pair_a, "fv")

    c_shape = random.choice(unique_c["orbit"])
    templates = [
        {"role": "pair_a", "shape": pair_a},
        {"role": "pair_b", "shape": pair_b},
        {"role": "unique_c", "shape": c_shape},
    ]
    if template_count == 4:
        d_class = random.choice(remaining)
        d_shape = random.choice(d_class["orbit"])
        templates.append({"role": "unique_d", "shape": d_shape})
    return templates


def _draw_legend(grid, templates):
    ordered = templates[:]
    random.shuffle(ordered)
    colors = random.sample(TEMPLATE_COLORS, len(ordered))
    cursor = 0
    by_role = {}
    for template, color in zip(ordered, colors):
        patch = _patch_from_shape(template["shape"], color)
        h, w = patch.shape
        row = random.randint(0, LEGEND_HEIGHT - h)
        grid[row : row + h, cursor : cursor + w][patch != BG] = int(color)
        by_role[template["role"]] = {
            "shape": template["shape"],
            "color": int(color),
        }
        cursor += w + TEMPLATE_GAP
    divider_c = cursor - TEMPLATE_GAP
    grid[LEGEND_HEIGHT, : divider_c + 1] = DIVIDER
    grid[: LEGEND_HEIGHT + 1, divider_c] = DIVIDER
    return by_role, divider_c


def _query_shape_options(template_shape):
    rot_only = sorted(_variants(template_shape, ROT_OPS) - _variants(template_shape, REF_OPS))
    ref_only = sorted(_variants(template_shape, REF_OPS) - _variants(template_shape, ROT_OPS))
    rot_any = sorted(_variants(template_shape, ROT_OPS))
    ref_any = sorted(_variants(template_shape, REF_OPS))
    return {
        "rot_only": rot_only or rot_any,
        "ref_only": ref_only or ref_any,
        "rot_any": rot_any,
        "ref_any": ref_any,
    }


def _query_plan(role_map):
    pair_a = role_map["pair_a"]
    pair_b = role_map["pair_b"]
    unique_c = role_map["unique_c"]

    pair_a_opts = _query_shape_options(pair_a["shape"])
    c_opts = _query_shape_options(unique_c["shape"])

    queries = [
        {"shape": pair_a["shape"], "category": "upper", "color": pair_a["color"], "tag": "contrast_upper"},
        {"shape": pair_a["shape"], "category": "lower", "color": pair_b["color"], "tag": "contrast_lower"},
        {"shape": random.choice(c_opts["ref_only"]), "category": "diagonal", "color": unique_c["color"], "tag": "diag"},
        {"shape": random.choice(c_opts["rot_only"]), "category": "upper", "color": unique_c["color"], "tag": "upper_c"},
        {"shape": random.choice(c_opts["ref_only"]), "category": "lower", "color": unique_c["color"], "tag": "lower_c"},
    ]

    extra_total = random.randint(*QUERY_COUNT_RANGE) - len(queries)
    optional_roles = [role for role in role_map if role not in {"pair_a", "pair_b"}]
    for _ in range(extra_total):
        role = random.choice(optional_roles)
        template = role_map[role]
        opts = _query_shape_options(template["shape"])
        category = random.choice(("upper", "lower", "diagonal"))
        if category == "upper":
            shape = random.choice(opts["rot_only"])
        elif category == "lower":
            shape = random.choice(opts["ref_only"])
        else:
            shape = random.choice(opts["rot_any"] + opts["ref_any"])
        queries.append({"shape": shape, "category": category, "color": template["color"], "tag": f"extra_{role}"})
    return queries


def _build_instance():
    templates = _pick_template_set()
    legend_width = sum(_shape_dims(t["shape"])[1] for t in templates) + TEMPLATE_GAP * (len(templates) - 1)
    size_min = max(GRID_MIN, legend_width + 6, 2 * LEGEND_HEIGHT + 8)
    if size_min > GRID_MAX:
        return None
    size = random.randint(size_min, GRID_MAX)
    base = np.full((size, size), BG, dtype=int)
    role_map, divider_c = _draw_legend(base, templates)
    if divider_c >= size - 3:
        return None

    recolored = base.copy()
    bboxes = []
    query_infos = []
    for query in _query_plan(role_map):
        coords = _place_query(recolored, query["shape"], query["color"], query["category"], bboxes)
        if coords is None:
            return None
        query_infos.append(
            {
                "shape": _normalize(query["shape"]),
                "category": query["category"],
                "coords": coords,
                "color": int(query["color"]),
                "tag": query["tag"],
            }
        )

    output = recolored.copy()
    for info in query_infos:
        if info["category"] == "diagonal":
            _draw_outline(output, info["coords"])

    inp = recolored.copy()
    for info in query_infos:
        coords = info["coords"]
        inp[coords[:, 0], coords[:, 1]] = QUERY
    return inp, output, query_infos


def _validate_generated(inp, expected, query_infos):
    got = main(inp)
    if not np.array_equal(got, expected):
        return False

    categories = {info["category"] for info in query_infos}
    if categories != {"upper", "lower", "diagonal"}:
        return False

    contrast_upper = next(info for info in query_infos if info["tag"] == "contrast_upper")
    contrast_lower = next(info for info in query_infos if info["tag"] == "contrast_lower")
    if contrast_upper["shape"] != contrast_lower["shape"]:
        return False
    if contrast_upper["color"] == contrast_lower["color"]:
        return False

    templates = _legend_templates(inp)
    for info in query_infos:
        if _query_category(info["coords"]) != info["category"]:
            return False
        match = _match_template(info["shape"], templates, info["category"])
        if int(match["color"]) != info["color"]:
            return False

    diag_coords = [info["coords"] for info in query_infos if info["category"] == "diagonal"]
    if not diag_coords:
        return False
    outline_cells = np.argwhere(got == OUTLINE)
    if outline_cells.size == 0:
        return False
    for coords in diag_coords:
        sprite = {tuple(map(int, p)) for p in coords}
        if not any(any((r + dr, c + dc) in sprite for dr, dc in NEIGHBORS4) for r, c in outline_cells):
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_instance()
        if built is None:
            continue
        inp, expected, query_infos = built
        if _validate_generated(inp, expected, query_infos):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
