import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste
import random


def _canonical_shape(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    variants = []
    for k in range(8):
        transformed = []
        for r, c in pts:
            if k == 0:
                rr, cc = r, c
            elif k == 1:
                rr, cc = r, -c
            elif k == 2:
                rr, cc = -r, c
            elif k == 3:
                rr, cc = -r, -c
            elif k == 4:
                rr, cc = c, r
            elif k == 5:
                rr, cc = c, -r
            elif k == 6:
                rr, cc = -c, r
            else:
                rr, cc = -c, -r
            transformed.append((rr, cc))

        min_r = min(r for r, _ in transformed)
        min_c = min(c for _, c in transformed)
        variants.append(tuple(sorted((r - min_r, c - min_c) for r, c in transformed)))
    return min(variants)


def _find_legend_divider(g: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = g.shape
    candidates = []
    for color in colors_present(g, exclude={bg}):
        mask = g == color
        for r in range(h):
            c = -1
            while c + 1 < w and mask[r, c + 1]:
                c += 1
            if r >= 2 and c >= 3 and mask[: r + 1, c].all():
                candidates.append((int(color), int(r), int(c)))

    assert len(candidates) == 1, "expected one L-shaped divider touching the top and left edges"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each query sprite by matching its shape, up to rotation/reflection, to a colored legend sprite.

    Input structure
    A background grid contains a one-color L-shaped divider from the top edge to the left edge.  The upper-left
    region inside that divider holds monochrome colored template sprites.  Elsewhere, all query sprites share one
    non-template color.

    Rule
    Read the upper-left templates as a shape-to-color legend.  For each query sprite, compare its connected-cell
    shape to the legend shapes under all rotations and reflections, then recolor that sprite to the matched
    template color.  Leave the divider, background, and legend pixels unchanged.

    Key insight
    The divider marks a dictionary of sprite shapes; the query color is only a placeholder, and orientation does
    not matter for the lookup.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    divider_color, divider_r, divider_c = _find_legend_divider(g, bg)

    legend = g[:divider_r, :divider_c]
    template_by_shape: dict[tuple[tuple[int, int], ...], int] = {}
    template_colors: set[int] = set()
    for comp in find_components(legend, bg=bg, connectivity=4, monochromatic=True):
        color = int(comp["color"])
        assert color != divider_color, "divider color must not appear inside the legend interior"
        key = _canonical_shape(comp["coords"])
        assert key not in template_by_shape, "each legend shape must map to exactly one color"
        template_by_shape[key] = color
        template_colors.add(color)
    assert template_by_shape, "legend must contain at least one template sprite"

    role_colors = set(colors_present(g, exclude={bg, divider_color}))
    query_colors = role_colors - template_colors
    assert len(query_colors) == 1, "expected exactly one non-template query color"
    query_color = next(iter(query_colors))

    out = g.copy()
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[query_color]):
        key = _canonical_shape(comp["coords"])
        assert key in template_by_shape, "query sprite shape must occur in the legend"
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = template_by_shape[key]

    return out

# --------------- generator (o5x) ---------------

# Canvas and role colors.
GRID_HEIGHT_RANGE = (10, 18)
GRID_WIDTH_RANGE = (10, 17)
BG_COLOR = 0
QUERY_COLOR = 3
DIVIDER_COLOR = 5
TEMPLATE_COLORS = (1, 2, 4, 6, 7, 8, 9)

# Legend and divider layout.
DIVIDER_ROW = 3
DIVIDER_COL_RANGE = (7, 11)
LEGEND_HEIGHT = 3
TEMPLATE_GAP = 1

# Template shape catalog.
TEMPLATE_COUNT_RANGE = (2, 4)
TEMPLATE_SIZE_RANGE = (2, 5)
TEMPLATE_WIDTH_RANGE = (1, 5)
TEMPLATE_MAX_HEIGHT = 3
NEIGHBOR_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIHEDRAL_TRANSFORMS = (
    (1, 0, 0, 1), (1, 0, 0, -1), (-1, 0, 0, 1), (-1, 0, 0, -1),
    (0, 1, 1, 0), (0, 1, -1, 0), (0, -1, 1, 0), (0, -1, -1, 0),
)

# Query placement.
QUERY_COUNT_RANGE = (3, 12)
QUERY_MIN_BBOX_GAP = 2
TOP_RIGHT_QUERY_PROB = 0.6

# Generation retries.
GEN_ATTEMPT_LIMIT = 300


def _gen_normalize(coords):
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in coords))


def _gen_variants(coords):
    pts = [(int(r), int(c)) for r, c in coords]
    variants = []
    for rr_r, rr_c, cc_r, cc_c in DIHEDRAL_TRANSFORMS:
        transformed = []
        for r, c in pts:
            transformed.append((rr_r * r + rr_c * c, cc_r * r + cc_c * c))
        variants.append(_gen_normalize(transformed))
    return tuple(sorted(set(variants)))


def _gen_canonical_shape(coords):
    return min(_gen_variants(coords))


def _gen_polyomino_catalog():
    """Build orientation-expanded shape choices keyed by bbox width."""
    min_size, max_size = TEMPLATE_SIZE_RANGE
    min_width, max_width = TEMPLATE_WIDTH_RANGE
    fixed_by_size = {min_size - 1: {((0, 0),)}}
    for size in range(min_size, max_size + 1):
        shapes = set()
        for shape in fixed_by_size[size - 1]:
            cells = set(shape)
            for r, c in shape:
                for dr, dc in NEIGHBOR_STEPS:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        shapes.add(_gen_normalize(cells | {nxt}))
        fixed_by_size[size] = shapes

    by_width = {w: [] for w in range(min_width, max_width + 1)}
    seen = set()
    for size in range(min_size, max_size + 1):
        for shape in fixed_by_size[size]:
            key = _gen_canonical_shape(shape)
            if key in seen:
                continue
            seen.add(key)
            for coords in _gen_variants(key):
                h = max(r for r, _ in coords) + 1
                w = max(c for _, c in coords) + 1
                if h <= TEMPLATE_MAX_HEIGHT and min_width <= w <= max_width:
                    by_width[w].append(
                        {
                            "key": key,
                            "coords": coords,
                            "size": len(coords),
                            "h": h,
                            "w": w,
                        }
                    )
    return by_width


def _gen_width_compositions(total, count):
    if count == 0:
        return [()] if total == 0 else []
    out = []
    min_width, max_width = TEMPLATE_WIDTH_RANGE
    lo = max(min_width, total - max_width * (count - 1))
    hi = min(max_width, total - min_width * (count - 1))
    for width in range(lo, hi + 1):
        for rest in _gen_width_compositions(total - width, count - 1):
            out.append((width,) + rest)
    return out


def _gen_pick_templates(widths, catalog):
    choices = []
    used = set()
    for width in widths:
        candidates = [item for item in catalog[width] if item["key"] not in used]
        if not candidates:
            return None
        item = random.choice(candidates)
        used.add(item["key"])
        choices.append(item)
    return choices


def _gen_patch(coords, color):
    h = max(r for r, _ in coords) + 1
    w = max(c for _, c in coords) + 1
    patch = np.full((h, w), BG_COLOR, dtype=int)
    rr, cc = np.array(coords, dtype=int).T
    patch[rr, cc] = int(color)
    return patch


def _gen_bbox_gap_ok(bbox, existing):
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

        if max(dr, dc) < QUERY_MIN_BBOX_GAP:
            return False
    return True


def _gen_collect_query_candidates(out, mask, regions, query_bboxes):
    h, w = mask.shape
    candidates = []
    for r_min, r_max, c_min, c_max in regions:
        for r0 in range(r_min, r_max - h + 2):
            for c0 in range(c_min, c_max - w + 2):
                bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
                if not _gen_bbox_gap_ok(bbox, query_bboxes):
                    continue
                if np.any(out[r0 : r0 + h, c0 : c0 + w][mask] != BG_COLOR):
                    continue
                candidates.append((r0, c0, bbox))
    return candidates


def _gen_place_query(out, query_mask, coords, color, regions, query_bboxes):
    patch = _gen_patch(coords, color)
    mask = patch != BG_COLOR
    candidates = _gen_collect_query_candidates(out, mask, regions, query_bboxes)
    if not candidates:
        return False

    r0, c0, bbox = random.choice(candidates)
    h, w = patch.shape
    paste(out, patch, r0, c0, transparent=BG_COLOR)
    query_mask[r0 : r0 + h, c0 : c0 + w] |= mask
    query_bboxes.append(bbox)
    return True


def _gen_choose_templates(num_templates, grid_w, catalog):
    min_col, max_col = DIVIDER_COL_RANGE
    c_options = list(range(min_col, min(max_col, grid_w - 1) + 1))
    random.shuffle(c_options)
    for divider_c in c_options:
        total_width = divider_c - TEMPLATE_GAP * (num_templates - 1)
        comps = _gen_width_compositions(total_width, num_templates)
        random.shuffle(comps)
        for widths in comps:
            templates = _gen_pick_templates(widths, catalog)
            if templates is not None:
                return divider_c, templates
    return None, None


def _gen_draw_legend(out, templates, colors):
    """Render legend templates left-to-right and return their shape/color lookup info."""
    c = 0
    template_infos = []
    for item, color in zip(templates, colors):
        patch = _gen_patch(item["coords"], color)
        r0 = random.randint(0, LEGEND_HEIGHT - patch.shape[0])
        paste(out, patch, r0, c, transparent=BG_COLOR)
        template_infos.append({"key": item["key"], "color": color})
        c += patch.shape[1] + TEMPLATE_GAP
    return template_infos


def _gen_place_all_queries(out, template_infos, assignments, divider_c):
    """Place every query sprite and return a mask of cells to recolor to query color."""
    _, w = out.shape
    query_mask = np.zeros_like(out, dtype=bool)
    query_bboxes = []
    top_right_index = None
    if random.random() < TOP_RIGHT_QUERY_PROB and divider_c + 1 < w:
        top_right_index = random.randrange(len(assignments))

    for i, template_idx in enumerate(assignments):
        info = template_infos[template_idx]
        variants = list(_gen_variants(info["key"]))
        random.shuffle(variants)
        use_top = i == top_right_index
        if _gen_try_place_query_variants(out, query_mask, variants, info["color"], use_top, divider_c, query_bboxes):
            continue
        if not use_top or not _gen_try_place_query_variants(
            out, query_mask, variants, info["color"], False, divider_c, query_bboxes
        ):
            return None
    return query_mask


def _gen_try_place_query_variants(out, query_mask, variants, color, use_top, divider_c, query_bboxes):
    h, w = out.shape
    if use_top:
        regions = [(0, LEGEND_HEIGHT - 1, divider_c + 1, w - 1)]
    else:
        regions = [(DIVIDER_ROW + 1, h - 1, 0, w - 1)]
    for coords in variants:
        if _gen_place_query(out, query_mask, coords, color, regions, query_bboxes):
            return True
    return False


_GEN_POLYOMINO_CATALOG = _gen_polyomino_catalog()


def _gen_attempt():
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    num_templates = random.randint(*TEMPLATE_COUNT_RANGE)
    catalog = _GEN_POLYOMINO_CATALOG
    divider_c, templates = _gen_choose_templates(num_templates, w, catalog)
    if templates is None:
        return None

    colors = random.sample(TEMPLATE_COLORS, num_templates)
    out = np.full((h, w), BG_COLOR, dtype=int)
    template_infos = _gen_draw_legend(out, templates, colors)
    out[DIVIDER_ROW, : divider_c + 1] = DIVIDER_COLOR
    out[: DIVIDER_ROW + 1, divider_c] = DIVIDER_COLOR

    min_queries, max_queries = QUERY_COUNT_RANGE
    query_count = random.randint(max(min_queries, num_templates), max_queries)
    assignments = list(range(num_templates))
    assignments.extend(random.randrange(num_templates) for _ in range(query_count - num_templates))
    random.shuffle(assignments)
    query_mask = _gen_place_all_queries(out, template_infos, assignments, divider_c)
    if query_mask is None:
        return None

    inp = out.copy()
    inp[query_mask] = QUERY_COLOR

    values, counts = np.unique(inp, return_counts=True)
    if int(values[np.argmax(counts)]) != BG_COLOR:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input():
    for _ in range(GEN_ATTEMPT_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
