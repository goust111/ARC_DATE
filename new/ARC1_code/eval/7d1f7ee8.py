import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg, paste, random_free_location
import random


def _is_rectangular_perimeter(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = bbox_h(comp["bbox"])
    w = bbox_w(comp["bbox"])
    coords = comp["coords"]
    if h == 1 or w == 1:
        return int(comp["area"]) == h * w
    on_border = (
        (coords[:, 0] == r0) | (coords[:, 0] == r1) | (coords[:, 1] == c0) | (coords[:, 1] == c1)
    )
    return int(comp["area"]) == 2 * h + 2 * w - 4 and bool(np.all(on_border))


def _strictly_contains(outer: dict, inner: dict) -> bool:
    r0, c0, r1, c1 = outer["bbox"]
    rr0, cc0, rr1, cc1 = inner["bbox"]
    return r0 < rr0 and c0 < cc0 and rr1 < r1 and cc1 < c1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every nested foreground rectangle to the color of the outermost rectangle that contains it.

    Input structure
    - The background is the dominant color.
    - Every foreground object is a 4-connected monochromatic rectangular perimeter, possibly a degenerate rectangle with no interior.
    - Some rectangles are strictly nested inside the interior of larger rectangles.

    Rule
    - Find the foreground rectangles that are not strictly inside any other foreground rectangle.
    - For each such outermost rectangle, recolor every non-background pixel strictly inside its bbox to that rectangle's color.
    - Preserve the outermost rectangle borders, the background, and any top-level rectangle that has no foreground pixels inside it.

    Key insight
    - The visible containment hierarchy, not the inner objects' own colors or shapes, determines the replacement color.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert comps
    assert all(_is_rectangular_perimeter(comp) for comp in comps)

    outermost = [
        comp
        for comp in comps
        if not any(_strictly_contains(parent, comp) for parent in comps if parent is not comp)
    ]
    assert outermost
    assert any(
        _strictly_contains(parent, child)
        for parent in outermost
        for child in comps
        if child is not parent
    )

    out = g.copy()
    for comp in outermost:
        r0, c0, r1, c1 = comp["bbox"]
        if r1 <= r0 + 1 or c1 <= c0 + 1:
            continue
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        foreground = g[r0 + 1 : r1, c0 + 1 : c1] != bg
        interior[foreground] = int(comp["color"])
    return out

# --------------- generator (o5x) ---------------

# Palette and canvas bounds.
_GEN_BG, _GEN_COLORS = 0, list(range(1, 10))
_GEN_HEIGHT_RANGE, _GEN_WIDTH_RANGE, _GEN_PALETTE_SIZE_RANGE = (12, 29), (18, 28), (4, 6)

# Forest shape bounds.
_GEN_TOTAL_RECTS_RANGE, _GEN_ROOT_RANGE, _GEN_MAX_TREE_DEPTH = (4, 10), (2, 4), 3
_GEN_CHILDLESS_ROOT_NODES, _GEN_CHILD_ROOT_MIN_NODES, _GEN_CHILD_COUNT_OPTIONS = 1, 2, (1, 2)

# Rectangle sizing and nested layout.
_GEN_MAX_PATCH_HEIGHT, _GEN_MAX_PATCH_WIDTH = 16, 21
_GEN_LEAF_DEGENERATE_PROB, _GEN_LEAF_DEGENERATE_AXIS_PROB = 0.28, 0.5
_GEN_LEAF_LINE_LENGTH_RANGE, _GEN_LEAF_HEIGHT_RANGE, _GEN_LEAF_WIDTH_RANGE = (2, 5), (2, 5), (2, 7)
_GEN_MARGIN_VALUES, _GEN_MARGIN_WEIGHTS = [1, 2, 3], [2, 5, 2]
_GEN_SIBLING_GAP_RANGE, _GEN_HORIZONTAL_LAYOUT_PROB = (1, 3), 0.5

# Top-level placement and retry behavior.
_GEN_EDGE_ROOT_PROB, _GEN_EDGE_BORDER, _GEN_DEFAULT_BORDER = 0.12, 0, 1
_GEN_PLACEMENT_PADDING, _GEN_RETRY_LIMIT = 1, 300


def _gen_randint(bounds: tuple[int, int]) -> int:
    return random.randint(bounds[0], bounds[1])


def _gen_max_nodes(depth: int) -> int:
    return (1 << (int(depth) + 1)) - 1


def _gen_count_options(total: int, roots: int, cap: int) -> list[int]:
    opts = []
    if _GEN_CHILD_ROOT_MIN_NODES * roots <= total <= roots * cap:
        opts.append(0)
    min_total = _GEN_CHILDLESS_ROOT_NODES + (roots - 1) * _GEN_CHILD_ROOT_MIN_NODES
    max_total = _GEN_CHILDLESS_ROOT_NODES + (roots - 1) * cap
    if min_total <= total <= max_total:
        opts.append(1)
    return opts


def _gen_distribute_counts(total: int, roots: int, cap: int) -> list[int]:
    childless = random.choice(_gen_count_options(total, roots, cap))
    counts = [_GEN_CHILDLESS_ROOT_NODES] * childless + [_GEN_CHILD_ROOT_MIN_NODES] * (roots - childless)
    remaining = total - sum(counts)
    growable = list(range(childless, roots))
    while remaining > 0:
        choices = [i for i in growable if counts[i] < cap]
        i = random.choice(choices)
        counts[i] += 1
        remaining -= 1
    random.shuffle(counts)
    return counts


def _gen_choose_forest_params() -> tuple[int, int, int, list[int]]:
    total = _gen_randint(_GEN_TOTAL_RECTS_RANGE)
    root_choices = []
    for roots in range(_GEN_ROOT_RANGE[0], _GEN_ROOT_RANGE[1] + 1):
        min_total = _GEN_CHILDLESS_ROOT_NODES + (roots - 1) * _GEN_CHILD_ROOT_MIN_NODES
        if roots > total - 1 or total < min_total:
            continue
        k_choices = [
            k
            for k in range(_GEN_PALETTE_SIZE_RANGE[0], min(_GEN_PALETTE_SIZE_RANGE[1], total) + 1)
            if _gen_count_options(total, roots, k)
        ]
        if k_choices:
            root_choices.append((roots, k_choices))
    roots, k_choices = random.choice(root_choices)
    palette_size = random.choice(k_choices)
    counts = _gen_distribute_counts(total, roots, palette_size)
    return total, roots, palette_size, counts


def _gen_tree_shape(count: int, depth: int, next_id: int) -> tuple[dict, int]:
    node = {"id": next_id, "children": []}
    next_id += 1
    if count == 1:
        return node, next_id

    rem = count - 1
    child_cap = _gen_max_nodes(depth - 1)
    child_counts = [k for k in _GEN_CHILD_COUNT_OPTIONS if rem >= k and rem <= k * child_cap]
    n_children = random.choice(child_counts)

    if n_children == 1:
        parts = [rem]
    else:
        lo = max(1, rem - child_cap)
        hi = min(child_cap, rem - 1)
        first = random.randint(lo, hi)
        parts = [first, rem - first]
        random.shuffle(parts)

    for part in parts:
        child, next_id = _gen_tree_shape(part, depth - 1, next_id)
        node["children"].append(child)
    return node, next_id


def _gen_make_forest(counts: list[int]) -> list[dict]:
    roots = []
    next_id = 0
    for count in counts:
        if count == 1:
            depth = 0
        else:
            depth = random.choice([d for d in range(1, _GEN_MAX_TREE_DEPTH + 1) if _gen_max_nodes(d) >= count])
        root, next_id = _gen_tree_shape(count, depth, next_id)
        roots.append(root)
    return roots


def _gen_collect_ids(node: dict) -> list[int]:
    ids = [int(node["id"])]
    for child in node["children"]:
        ids.extend(_gen_collect_ids(child))
    return ids


def _gen_assign_colors(roots: list[dict], palette: list[int]) -> dict[int, int] | None:
    color_by_id = {}
    tree_used = []
    root_colors = random.sample(palette, len(roots))
    for root, color in zip(roots, root_colors):
        color_by_id[int(root["id"])] = int(color)
        tree_used.append({int(color)})

    slots = []
    for ti, root in enumerate(roots):
        for node_id in _gen_collect_ids(root)[1:]:
            slots.append((ti, int(node_id)))
    random.shuffle(slots)

    for color in [c for c in palette if c not in root_colors]:
        choices = [i for i, (ti, _) in enumerate(slots) if color not in tree_used[ti]]
        if not choices:
            return None
        slot_i = random.choice(choices)
        ti, node_id = slots.pop(slot_i)
        color_by_id[node_id] = int(color)
        tree_used[ti].add(int(color))

    while slots:
        ti, node_id = slots.pop()
        choices = [c for c in palette if c not in tree_used[ti]]
        if not choices:
            return None
        color = int(random.choice(choices))
        color_by_id[node_id] = color
        tree_used[ti].add(color)

    return color_by_id


def _gen_draw_rect(h: int, w: int, color: int) -> np.ndarray:
    arr = np.zeros((int(h), int(w)), dtype=int)
    arr[0, :] = int(color)
    arr[-1, :] = int(color)
    arr[:, 0] = int(color)
    arr[:, -1] = int(color)
    return arr


def _gen_leaf_size() -> tuple[int, int]:
    if random.random() < _GEN_LEAF_DEGENERATE_PROB:
        if random.random() < _GEN_LEAF_DEGENERATE_AXIS_PROB:
            return 1, _gen_randint(_GEN_LEAF_LINE_LENGTH_RANGE)
        return _gen_randint(_GEN_LEAF_LINE_LENGTH_RANGE), 1
    return _gen_randint(_GEN_LEAF_HEIGHT_RANGE), _gen_randint(_GEN_LEAF_WIDTH_RANGE)


def _gen_margin() -> int:
    return random.choices(_GEN_MARGIN_VALUES, weights=_GEN_MARGIN_WEIGHTS)[0]


def _gen_two_child_layout(a: np.ndarray, b: np.ndarray):
    gap = _gen_randint(_GEN_SIBLING_GAP_RANGE)
    if random.random() < _GEN_HORIZONTAL_LAYOUT_PROB:
        union_h = max(a.shape[0], b.shape[0])
        union_w = a.shape[1] + gap + b.shape[1]
        a_r = random.randint(0, union_h - a.shape[0])
        b_r = random.randint(0, union_h - b.shape[0])
        a_c = 0
        b_c = a.shape[1] + gap
    else:
        union_h = a.shape[0] + gap + b.shape[0]
        union_w = max(a.shape[1], b.shape[1])
        a_r = 0
        b_r = a.shape[0] + gap
        a_c = random.randint(0, union_w - a.shape[1])
        b_c = random.randint(0, union_w - b.shape[1])
    return union_h, union_w, [(a, a_r, a_c), (b, b_r, b_c)]


def _gen_wrap_children(color: int, child_arrays: list[np.ndarray]) -> np.ndarray | None:
    if len(child_arrays) == 1:
        child = child_arrays[0]
        union_h, union_w = child.shape
        placements = [(child, 0, 0)]
    else:
        union_h, union_w, placements = _gen_two_child_layout(child_arrays[0], child_arrays[1])

    top, bottom, left, right = (_gen_margin(), _gen_margin(), _gen_margin(), _gen_margin())
    h = union_h + top + bottom
    w = union_w + left + right
    if h > _GEN_MAX_PATCH_HEIGHT or w > _GEN_MAX_PATCH_WIDTH:
        return None

    out = _gen_draw_rect(h, w, color)
    for child, r, c in placements:
        paste(out, child, top + r, left + c, transparent=_GEN_BG)
    return out


def _gen_build_tree_array(node: dict, color_by_id: dict[int, int]) -> np.ndarray | None:
    color = color_by_id[int(node["id"])]
    children = node["children"]
    if not children:
        h, w = _gen_leaf_size()
        return _gen_draw_rect(h, w, color)

    child_arrays = []
    for child in children:
        arr = _gen_build_tree_array(child, color_by_id)
        if arr is None:
            return None
        child_arrays.append(arr)

    return _gen_wrap_children(color, child_arrays)


def _gen_plan_forest() -> tuple[list[dict], dict[int, int]] | None:
    _, _, palette_size, counts = _gen_choose_forest_params()
    roots = _gen_make_forest(counts)
    palette = random.sample(_GEN_COLORS, palette_size)
    color_by_id = _gen_assign_colors(roots, palette)
    if color_by_id is None:
        return None
    return roots, color_by_id


def _gen_render_roots(roots: list[dict], color_by_id: dict[int, int]):
    patches = []
    root_colors = []
    has_children = []
    for root in roots:
        patch = _gen_build_tree_array(root, color_by_id)
        if patch is None:
            return None
        patches.append(patch)
        root_colors.append(color_by_id[int(root["id"])])
        has_children.append(bool(root["children"]))
    return patches, root_colors, has_children


def _gen_place_patches(height: int, width: int, patches: list[np.ndarray]):
    inp = np.zeros((height, width), dtype=int)
    reserved = np.zeros_like(inp)
    order = sorted(range(len(patches)), key=lambda i: patches[i].size, reverse=True)
    edge_root = random.choice(order) if random.random() < _GEN_EDGE_ROOT_PROB else None
    placements = {}
    for i in order:
        patch = patches[i]
        full_bbox = np.ones(patch.shape, dtype=int)
        border = _GEN_EDGE_BORDER if i == edge_root else _GEN_DEFAULT_BORDER
        try:
            r, c = random_free_location(
                reserved,
                full_bbox,
                transparent=_GEN_BG,
                border=border,
                padding=_GEN_PLACEMENT_PADDING,
            )
        except ValueError:
            return None
        paste(reserved, full_bbox, r, c, transparent=_GEN_BG)
        paste(inp, patch, r, c, transparent=_GEN_BG)
        placements[i] = (r, c)
    return inp, placements


def _gen_expected_output(inp, patches, root_colors, has_children, placements) -> np.ndarray:
    expected = inp.copy()
    for i, (r, c) in placements.items():
        if not has_children[i]:
            continue
        h, w = patches[i].shape
        foreground = inp[r + 1 : r + h - 1, c + 1 : c + w - 1] != _GEN_BG
        interior = expected[r + 1 : r + h - 1, c + 1 : c + w - 1]
        interior[foreground] = root_colors[i]
    return expected


def _gen_matches_oracle(inp: np.ndarray, expected: np.ndarray) -> bool:
    if get_bg(inp) != _GEN_BG:
        return False
    try:
        actual = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    height = _gen_randint(_GEN_HEIGHT_RANGE)
    width = _gen_randint(_GEN_WIDTH_RANGE)

    planned = _gen_plan_forest()
    if planned is None:
        return None
    roots, color_by_id = planned

    rendered = _gen_render_roots(roots, color_by_id)
    if rendered is None:
        return None
    patches, root_colors, has_children = rendered

    placed = _gen_place_patches(height, width, patches)
    if placed is None:
        return None
    inp, placements = placed

    expected = _gen_expected_output(inp, patches, root_colors, has_children, placements)
    if not _gen_matches_oracle(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 7d1f7ee8 input")
