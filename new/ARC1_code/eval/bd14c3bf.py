import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, get_bg, paste, random_free_location
import itertools
import random


def _component_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _compress_runs(mask: np.ndarray) -> np.ndarray:
    assert mask.any()

    row_keep = [0]
    for r in range(1, mask.shape[0]):
        if not np.array_equal(mask[r], mask[r - 1]):
            row_keep.append(r)
    mask = mask[row_keep, :]

    col_keep = [0]
    for c in range(1, mask.shape[1]):
        if not np.array_equal(mask[:, c], mask[:, c - 1]):
            col_keep.append(c)
    return mask[:, col_keep]


def _signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    compressed = _compress_runs(mask)
    return tuple(tuple(int(v) for v in row) for row in compressed)


def _prototype_signatures(mask: np.ndarray) -> set[tuple[tuple[int, ...], ...]]:
    signatures = set()
    for turns in range(4):
        rotated = np.rot90(mask, turns)
        signatures.add(_signature(rotated))
        signatures.add(_signature(np.fliplr(rotated)))
    return signatures


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor every candidate object whose run-compressed shape matches the single prototype object, allowing rotation and reflection.
    Input structure
    A black-background grid contains exactly two foreground colors: one color forms a single prototype object, and the other forms multiple candidate objects.
    Rule
    Crop each object to its bbox, collapse adjacent identical rows and columns to remove stretch, and recolor every candidate whose collapsed mask is a rotation or reflection of the prototype's collapsed mask.
    Key insight
    The prototype is a line-drawing symbol; candidate symbols can be stretched in row/column runs, so matching must compare the compressed topology rather than raw size.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 2

    comps_by_color = {
        color: components_mask(g == color, connectivity=4) for color in fg_colors
    }
    prototype_colors = [
        color for color, comps in comps_by_color.items() if len(comps) == 1
    ]
    assert len(prototype_colors) == 1

    prototype_color = prototype_colors[0]
    candidate_colors = [color for color in fg_colors if color != prototype_color]
    assert len(candidate_colors) == 1
    candidate_color = candidate_colors[0]
    assert len(comps_by_color[candidate_color]) > 1

    prototype_mask = _component_mask(comps_by_color[prototype_color][0])
    prototype_signatures = _prototype_signatures(prototype_mask)

    out = g.copy()
    matched = 0
    for coords in comps_by_color[candidate_color]:
        candidate_mask = _component_mask(coords)
        if _signature(candidate_mask) in prototype_signatures:
            out[coords[:, 0], coords[:, 1]] = prototype_color
            matched += 1

    assert matched > 0
    return out

# --------------- generator (o5x) ---------------

# Canvas, palette, and object counts.
_GEN_CANVAS_MIN, _GEN_CANVAS_MAX = 15, 19
_GEN_FOREGROUND_COLORS = range(1, 10)
_GEN_COUNT_CHOICES = [(3, 3), (3, 4), (3, 5), (4, 3), (4, 4)]

# Line-symbol shape, density, and compression bounds.
_GEN_SYMBOL_SIZE_MIN, _GEN_SYMBOL_SIZE_MAX = 3, 5
_GEN_SYMBOL_DENSITY_MIN, _GEN_SYMBOL_DENSITY_MAX = 0.55, 0.89
_GEN_PROTO_DENSITY_MIN = 0.78
_GEN_COMPRESSED_SIZE_MIN, _GEN_COMPRESSED_SIZE_MAX = 2, 5

# Stretch, area-budget, placement, and retry limits.
_GEN_STRETCH_FACTORS = (1, 2, 3)
_GEN_MAX_STRETCH_EXTRA = 2
_GEN_MAX_STRETCH_RATIO = 2.3
_GEN_MAX_SYMBOL_AREA = 25
_GEN_MIN_AREA_BUDGET, _GEN_PROTO_BUDGET_PAD = 8, 4
_GEN_CANDIDATE_RESERVED_CELLS, _GEN_CANDIDATE_BUDGET_PAD = 1, 2
_GEN_MAX_AREA_PAD = 2
_GEN_PLACEMENT_PADDING, _GEN_PLACEMENT_CONNECTIVITY = 1, 8
_GEN_MAX_ATTEMPTS = 300


def _gen_mask_key(mask):
    return tuple(tuple(int(v) for v in row) for row in mask)


def _gen_compress_runs(mask):
    row_keep = [0]
    for r in range(1, mask.shape[0]):
        if not np.array_equal(mask[r], mask[r - 1]):
            row_keep.append(r)
    mask = mask[row_keep, :]

    col_keep = [0]
    for c in range(1, mask.shape[1]):
        if not np.array_equal(mask[:, c], mask[:, c - 1]):
            col_keep.append(c)
    return mask[:, col_keep]


def _gen_signature(mask):
    return _gen_mask_key(_gen_compress_runs(mask))


def _gen_orientations(mask):
    variants = []
    seen = set()
    for turns in range(4):
        rotated = np.rot90(mask, turns)
        for variant in (rotated, np.fliplr(rotated)):
            key = _gen_mask_key(variant)
            if key not in seen:
                seen.add(key)
                variants.append(variant.copy())
    return variants


def _gen_signature_set(mask):
    return {_gen_signature(variant) for variant in _gen_orientations(mask)}


def _gen_touches_all_edges(mask):
    return mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()


def _gen_line_symbol_pool():
    masks = []
    seen = set()
    for h in range(_GEN_SYMBOL_SIZE_MIN, _GEN_SYMBOL_SIZE_MAX + 1):
        for w in range(_GEN_SYMBOL_SIZE_MIN, _GEN_SYMBOL_SIZE_MAX + 1):
            rows_all = range(h)
            cols_all = range(w)
            for row_count in range(1, h + 1):
                for rows in itertools.combinations(rows_all, row_count):
                    for col_count in range(1, w + 1):
                        for cols in itertools.combinations(cols_all, col_count):
                            mask = np.zeros((h, w), dtype=bool)
                            mask[list(rows), :] = True
                            mask[:, list(cols)] = True
                            density = float(mask.mean())
                            if not _GEN_SYMBOL_DENSITY_MIN <= density <= _GEN_SYMBOL_DENSITY_MAX:
                                continue
                            if not _gen_touches_all_edges(mask):
                                continue
                            key = _gen_mask_key(mask)
                            if key in seen:
                                continue
                            seen.add(key)
                            masks.append(mask)
    return masks


_GEN_LINE_SYMBOLS = _gen_line_symbol_pool()
_GEN_PROTO_SYMBOLS = [
    m
    for m in _GEN_LINE_SYMBOLS
    if _GEN_PROTO_DENSITY_MIN <= float(m.mean()) <= _GEN_SYMBOL_DENSITY_MAX
    and _GEN_COMPRESSED_SIZE_MIN <= _gen_compress_runs(m).shape[0] <= _GEN_COMPRESSED_SIZE_MAX
    and _GEN_COMPRESSED_SIZE_MIN <= _gen_compress_runs(m).shape[1] <= _GEN_COMPRESSED_SIZE_MAX
]


def _gen_factor_tuples(length):
    length = int(length)
    opts = []
    for factors in itertools.product(_GEN_STRETCH_FACTORS, repeat=length):
        total = sum(factors)
        extra = total - length
        if 0 <= extra <= _GEN_MAX_STRETCH_EXTRA and total <= _GEN_SYMBOL_SIZE_MAX and total / length <= _GEN_MAX_STRETCH_RATIO:
            opts.append(factors)
    return opts


def _gen_stretch_options(mask, *, min_density, max_density, max_area):
    row_opts = _gen_factor_tuples(mask.shape[0])
    col_opts = _gen_factor_tuples(mask.shape[1])
    out = []
    seen = set()
    for row_factors in row_opts:
        for col_factors in col_opts:
            stretched = np.repeat(np.repeat(mask, row_factors, axis=0), col_factors, axis=1)
            if stretched.shape[0] > _GEN_SYMBOL_SIZE_MAX or stretched.shape[1] > _GEN_SYMBOL_SIZE_MAX:
                continue
            area = int(stretched.sum())
            density = area / float(stretched.size)
            if area > int(max_area) or not min_density <= density <= max_density:
                continue
            key = _gen_mask_key(stretched)
            if key in seen:
                continue
            seen.add(key)
            out.append(stretched)
    return out


_GEN_STRETCHED_SYMBOLS = [
    (_gen_signature(stretched), stretched, int(stretched.sum()))
    for base in _GEN_LINE_SYMBOLS
    for stretched in _gen_stretch_options(base, min_density=_GEN_SYMBOL_DENSITY_MIN, max_density=_GEN_SYMBOL_DENSITY_MAX, max_area=_GEN_MAX_SYMBOL_AREA)
]


def _gen_place_objects(out, objects):
    random.shuffle(objects)
    objects.sort(key=lambda obj: obj["mask"].size, reverse=True)
    placements = []
    for obj in objects:
        patch = np.zeros(obj["mask"].shape, dtype=int)
        patch[obj["mask"]] = obj["out_color"]
        try:
            r0, c0 = random_free_location(
                out,
                patch,
                transparent=0,
                border=0,
                padding=_GEN_PLACEMENT_PADDING,
                padding_connectivity=_GEN_PLACEMENT_CONNECTIVITY,
            )
        except ValueError:
            return None
        paste(out, patch, r0, c0, transparent=0)
        coords = np.argwhere(obj["mask"]) + np.array([r0, c0])
        placements.append((obj, coords))
    return placements


def _gen_scene_params():
    n = random.randint(_GEN_CANVAS_MIN, _GEN_CANVAS_MAX)
    proto_color, cand_color = random.sample(_GEN_FOREGROUND_COLORS, 2)
    match_count, distractor_count = random.choice(_GEN_COUNT_CHOICES)
    return n, proto_color, cand_color, match_count, distractor_count


def _gen_select_prototype(n, total_candidates):
    proto_budget = max(_GEN_MIN_AREA_BUDGET, ((n * n) // 2) // (total_candidates + 1) + _GEN_PROTO_BUDGET_PAD)
    proto_choices = [m for m in _GEN_PROTO_SYMBOLS if int(m.sum()) <= proto_budget]
    if not proto_choices:
        return None
    return random.choice(proto_choices).copy()


def _gen_match_masks(proto_core, match_count, max_area):
    match_masks = []
    for _ in range(match_count):
        oriented = random.choice(_gen_orientations(proto_core))
        options = _gen_stretch_options(
            oriented,
            min_density=_GEN_SYMBOL_DENSITY_MIN,
            max_density=_GEN_SYMBOL_DENSITY_MAX,
            max_area=max_area,
        )
        if not options:
            return None
        match_masks.append(random.choice(options).copy())
    return match_masks


def _gen_distractor_masks(proto_sigs, distractor_count, max_area):
    distractor_pool = [mask for sig, mask, area in _GEN_STRETCHED_SYMBOLS if sig not in proto_sigs and area <= max_area]
    if len(distractor_pool) < distractor_count:
        return None
    return [random.choice(distractor_pool).copy() for _ in range(distractor_count)]


def _gen_input_from_output(out, placements):
    inp = out.copy()
    for obj, coords in placements:
        if obj["input_color"] != obj["out_color"]:
            inp[coords[:, 0], coords[:, 1]] = obj["input_color"]
    return inp


def _gen_solver_accepts(inp, expected):
    try:
        solved = main(inp)
    except Exception:
        return False
    return np.array_equal(solved, expected)


def _gen_attempt():
    n, proto_color, cand_color, match_count, distractor_count = _gen_scene_params()
    total_candidates = match_count + distractor_count

    proto_mask = _gen_select_prototype(n, total_candidates)
    if proto_mask is None:
        return None
    proto_sigs = _gen_signature_set(proto_mask)
    proto_core = _gen_compress_runs(proto_mask)
    candidate_budget = (n * n - int(proto_mask.sum()) - _GEN_CANDIDATE_RESERVED_CELLS) // 2
    avg_budget = max(_GEN_MIN_AREA_BUDGET, candidate_budget // total_candidates + _GEN_CANDIDATE_BUDGET_PAD)
    max_candidate_area = avg_budget + _GEN_MAX_AREA_PAD

    match_masks = _gen_match_masks(proto_core, match_count, max_candidate_area)
    if match_masks is None:
        return None

    distractor_masks = _gen_distractor_masks(proto_sigs, distractor_count, max_candidate_area)
    if distractor_masks is None:
        return None

    candidate_masks = match_masks + distractor_masks
    candidate_area = sum(int(m.sum()) for m in candidate_masks)
    if candidate_area >= n * n - int(proto_mask.sum()) - candidate_area:
        return None

    out = np.zeros((n, n), dtype=int)
    objects = [{"mask": proto_mask, "out_color": proto_color, "input_color": proto_color}]
    objects.extend({"mask": mask, "out_color": proto_color, "input_color": cand_color} for mask in match_masks)
    objects.extend({"mask": mask, "out_color": cand_color, "input_color": cand_color} for mask in distractor_masks)
    placements = _gen_place_objects(out, objects)
    if placements is None:
        return None

    inp = _gen_input_from_output(out, placements)
    if not _gen_solver_accepts(inp, out):
        return None
    return inp


def generate_input():
    for _ in range(_GEN_MAX_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate valid bd14c3bf input")
