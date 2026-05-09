# Combo variant of 64efde09 (index 13)
# combo_spec: D11/2+D16/0
# Name: 交叉标记+链式反应

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg


# Design analysis for the fused variant
# ------------------------------------
# Concrete behavior:
# - The scene contains four horizontal sprites stacked vertically:
#   top carrier (active edge faces up), upper inner emitter (faces down),
#   lower inner emitter (faces up), bottom carrier (faces down).
# - A boundary marker aligned with a carrier port does NOT color that carrier
#   port directly. Instead it cross-selects the opposite inner emitter that
#   faces into the stack. That inner emitter sends a probe ray first.
# - The color is assigned to the first active port hit by that probe on the
#   other inner emitter. Once that hit template port is known, the color
#   propagates globally from every sprite's corresponding port. Any further
#   active-edge hit activates another template port of the same color, so the
#   process is a BFS over (template-port, color) states.
#
# Why this is not a simplification:
# - If we let "hit sprite emits from the same port" only on that sprite, the
#   second step would merely bounce back on the incoming line and become almost
#   invisible. Promoting each activated hit to a template-wide emission keeps
#   the original task's shared-template reasoning while making D16/0 visible.
# - The inner pair use opposite id orders, so a seed column on one inner sprite
#   maps to a different template id on the first hit. This prevents regression
#   to the original "marker directly labels its own aligned port" behavior.
#
# Assumption audit:
# - Original assumption "marker is on the same sprite it labels" is false.
#   Here markers live on boundary-facing carriers and label the opposite inner
#   emitter only after that emitter's probe reaches another active port.
# - Original assumption "all colors are resolved in one pass" is false.
#   Here a queue is required because one activated template id can hit another.
# - New generator guarantees exactly one up-facing inner sprite and one
#   down-facing inner sprite without markers, plus one marked carrier for each
#   boundary direction. That makes the cross-mapping rule observable and
#   unambiguous for both humans and the solver.
#
# Placement / visual constraints:
# - All four sprites share one horizontal span, so vertical ray interactions
#   are clean and easy to read.
# - Marker columns are chosen from distinct mirror-pairs and avoid the middle
#   fixed point, so each color activates a two-port chain rather than a
#   degenerate self-map.


UP = (-1, 0)
DOWN = (1, 0)
BG = 8
NON_BG_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]

GRID_H_RANGE = (21, 29)
GRID_W_RANGE = (27, 29)
TEMPLATE_N_RANGE = (7, 9)
TEMPLATE_COLOR_COUNT = (3, 4)
ACTIVE_MAJORITY_RANGE = (5, 9)
NUM_MARKERS = 3
GEN_RETRIES = 400


def _dihedral_variants(template: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    ids = np.arange(template.size, dtype=int).reshape(template.shape)
    variants: list[tuple[str, np.ndarray, np.ndarray]] = []
    for k in range(4):
        rotated = np.rot90(template, k)
        rotated_ids = np.rot90(ids, k)
        variants.append((f"rot{k}", rotated, rotated_ids))
        variants.append((f"rot{k}_fliplr", np.fliplr(rotated), np.fliplr(rotated_ids)))
    return variants


def _active_side(sub: np.ndarray) -> tuple[tuple[int, int], set[tuple[int, int]]]:
    h, w = sub.shape
    assert h != w, "sprites must be non-square rectangles"

    if h > w:
        sides = [
            ((0, -1), [(r, 0) for r in range(h)]),
            ((0, 1), [(r, w - 1) for r in range(h)]),
        ]
    else:
        sides = [
            ((-1, 0), [(0, c) for c in range(w)]),
            ((1, 0), [(h - 1, c) for c in range(w)]),
        ]

    scored = []
    for direction, coords in sides:
        vals = [int(sub[r, c]) for r, c in coords]
        scored.append((max(Counter(vals).values()), direction, set(coords)))

    scored.sort(key=lambda item: item[0], reverse=True)
    assert scored[0][0] > scored[1][0], "active side must be structurally unique"
    return scored[0][1], scored[0][2]


def _first_marker_on_ray(
    g: np.ndarray,
    bg: int,
    markers: dict[tuple[int, int], int],
    r: int,
    c: int,
    dr: int,
    dc: int,
) -> int | None:
    H, W = g.shape
    r += dr
    c += dc
    while 0 <= r < H and 0 <= c < W:
        if int(g[r, c]) != bg:
            return markers.get((r, c))
        r += dr
        c += dc
    return None


def _sorted_active_coords(active: set[tuple[int, int]]) -> list[tuple[int, int]]:
    sample = next(iter(active))
    key_idx = 1 if len({r for r, _c in active}) == 1 else 0
    return sorted(active, key=lambda rc: rc[key_idx])


def _extract_scene(g: np.ndarray) -> tuple[int, list[dict], dict[tuple[int, int], int], dict[tuple[int, int], tuple[int, int, int]]]:
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False)

    sprites_raw: list[dict] = []
    markers: dict[tuple[int, int], int] = {}
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        area = int(comp["area"])
        if area == 1:
            r, c = map(int, comp["coords"][0])
            markers[(r, c)] = int(g[r, c])
        else:
            assert area == h * w, "sprite components must be dense rectangles"
            sprites_raw.append({"bbox": (r0, c0, r1, c1), "sub": crop_bbox(g, (r0, c0, r1, c1))})

    assert len(set(markers.values())) == len(markers), "marker colors must be unique"
    assert len(sprites_raw) == 4, "generator builds exactly four sprites"
    assert len(markers) == NUM_MARKERS, "expected three cross-markers"

    variants = _dihedral_variants(sprites_raw[0]["sub"])
    sprites: list[dict] = []
    cell_to_sprite: dict[tuple[int, int], tuple[int, int, int]] = {}
    for sprite in sprites_raw:
        matches = [
            (ids, name)
            for name, sub, ids in variants
            if sub.shape == sprite["sub"].shape and np.array_equal(sub, sprite["sub"])
        ]
        assert len(matches) == 1, "each sprite must match the template in exactly one orientation"
        ids, name = matches[0]
        direction, active = _active_side(sprite["sub"])
        active_list = _sorted_active_coords(active)
        port_markers: dict[int, int] = {}
        r0, c0, _r1, _c1 = sprite["bbox"]
        dr, dc = direction
        for lr, lc in active_list:
            color = _first_marker_on_ray(g, bg, markers, r0 + lr, c0 + lc, dr, dc)
            if color is not None:
                port_markers[int(ids[lr, lc])] = int(color)
        sprite_info = {
            **sprite,
            "ids": ids,
            "name": name,
            "direction": direction,
            "active": active,
            "active_list": active_list,
            "port_markers": port_markers,
        }
        si = len(sprites)
        sprites.append(sprite_info)
        for lr in range(sprite["sub"].shape[0]):
            for lc in range(sprite["sub"].shape[1]):
                cell_to_sprite[(r0 + lr, c0 + lc)] = (si, lr, lc)

    return bg, sprites, markers, cell_to_sprite


def _locate_template_port(sprite: dict, template_id: int) -> tuple[int, int]:
    locs = np.argwhere(sprite["ids"] == int(template_id))
    assert locs.shape[0] == 1
    lr, lc = map(int, locs[0])
    assert (lr, lc) in sprite["active"], "template id must land on an active-edge port"
    return lr, lc


def _trace_first_hit(
    g: np.ndarray,
    bg: int,
    out: np.ndarray | None,
    cell_to_sprite: dict[tuple[int, int], tuple[int, int, int]],
    start_r: int,
    start_c: int,
    dr: int,
    dc: int,
    color: int | None,
) -> tuple[int | None, int | None, int | None, list[tuple[int, int]]]:
    H, W = g.shape
    path: list[tuple[int, int]] = []
    r = int(start_r + dr)
    c = int(start_c + dc)
    while 0 <= r < H and 0 <= c < W:
        if int(g[r, c]) != bg:
            hit = cell_to_sprite.get((r, c))
            if hit is None:
                return None, None, None, path
            si, lr, lc = hit
            return int(si), int(lr), int(lc), path
        if out is not None and color is not None:
            assert int(out[r, c]) in (bg, color), "rays must not conflict"
            out[r, c] = int(color)
        path.append((r, c))
        r += dr
        c += dc
    return None, None, None, path


def _identify_roles(sprites: list[dict]) -> dict[str, int]:
    up_marked = [i for i, sp in enumerate(sprites) if sp["direction"] == UP and sp["port_markers"]]
    down_marked = [i for i, sp in enumerate(sprites) if sp["direction"] == DOWN and sp["port_markers"]]
    up_plain = [i for i, sp in enumerate(sprites) if sp["direction"] == UP and not sp["port_markers"]]
    down_plain = [i for i, sp in enumerate(sprites) if sp["direction"] == DOWN and not sp["port_markers"]]
    assert len(up_marked) == 1 and len(down_marked) == 1
    assert len(up_plain) == 1 and len(down_plain) == 1
    return {
        "top_carrier": up_marked[0],
        "bottom_carrier": down_marked[0],
        "upper_inner": down_plain[0],
        "lower_inner": up_plain[0],
    }


def _initial_activations(
    g: np.ndarray,
    bg: int,
    sprites: list[dict],
    roles: dict[str, int],
    cell_to_sprite: dict[tuple[int, int], tuple[int, int, int]],
) -> list[tuple[int, int]]:
    activations: list[tuple[int, int]] = []

    def cross_activate(carrier_idx: int, partner_idx: int) -> None:
        carrier = sprites[carrier_idx]
        partner = sprites[partner_idx]
        r0, c0, _r1, _c1 = partner["bbox"]
        dr, dc = partner["direction"]
        for seed_tid, color in carrier["port_markers"].items():
            lr, lc = _locate_template_port(partner, seed_tid)
            hit_si, target_lr, target_lc, _path = _trace_first_hit(
                g,
                bg,
                None,
                cell_to_sprite,
                r0 + lr,
                c0 + lc,
                dr,
                dc,
                None,
            )
            assert hit_si is not None, "cross-selected inner emitter must hit another sprite"
            hit_sprite = sprites[hit_si]
            assert (target_lr, target_lc) in hit_sprite["active"], "probe must hit an active port"
            target_tid = int(hit_sprite["ids"][target_lr, target_lc])
            activations.append((int(color), target_tid))

    cross_activate(roles["top_carrier"], roles["upper_inner"])
    cross_activate(roles["bottom_carrier"], roles["lower_inner"])
    return activations


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve boundary markers through opposite inner sprites, then spread colors by hit-triggered template activations.

    Input structure:
    - bg is the dominant color.
    - Four dense horizontal sprite instances are dihedral transforms of one thin rectangular template.
    - Three isolated boundary markers sit above or below two boundary-facing carrier sprites, while two unmarked inner sprites face each other inside the stack.

    Rule:
    - A marker above an up-facing carrier cross-selects the unique down-facing inner sprite; a marker below a down-facing carrier cross-selects the unique up-facing inner sprite.
    - The selected inner sprite first emits from the corresponding template port. The color is assigned to the first active-edge port hit on the opposite inner sprite, not to the carrier-aligned port itself.
    - Once a template port is activated with a color, every sprite emits that color from its corresponding active-edge port. Any ray that reaches another sprite's active port activates that hit template port too, and the process repeats until no new (color, template-port) states appear.

    Key insight:
    - Boundary markers are only indirect clues: they tell you which inner sprite launches the probe, and the real colored template id is determined one hit later. The rest of the task is a queue-driven propagation over shared template ports.
    """
    g = np.asarray(grid, dtype=int)
    bg, sprites, _markers, cell_to_sprite = _extract_scene(g)
    roles = _identify_roles(sprites)

    out = g.copy()
    queue = deque(_initial_activations(g, bg, sprites, roles, cell_to_sprite))
    seen: set[tuple[int, int]] = set(queue)

    while queue:
        color, template_id = queue.popleft()
        for sprite in sprites:
            lr, lc = _locate_template_port(sprite, template_id)
            r0, c0, _r1, _c1 = sprite["bbox"]
            dr, dc = sprite["direction"]
            hit_si, target_lr, target_lc, _path = _trace_first_hit(
                g,
                bg,
                out,
                cell_to_sprite,
                r0 + lr,
                c0 + lc,
                dr,
                dc,
                color,
            )
            if hit_si is None:
                continue
            hit_sprite = sprites[hit_si]
            if (target_lr, target_lc) not in hit_sprite["active"]:
                continue
            target_tid = int(hit_sprite["ids"][target_lr, target_lc])
            state = (color, target_tid)
            if state not in seen:
                seen.add(state)
                queue.append(state)

    return out


def _build_column(n: int, colors: list[int], majority: int, dominant: int) -> list[int]:
    col = [dominant] * majority
    others = [c for c in colors if c != dominant]
    fill = others if others else colors
    for _ in range(n - majority):
        col.append(random.choice(fill))
    random.shuffle(col)
    return col


def _design_template(n: int) -> tuple[np.ndarray, np.ndarray, list[int]] | None:
    num_colors = random.randint(*TEMPLATE_COLOR_COUNT)
    template_colors = random.sample(NON_BG_COLORS, num_colors)

    active_majority = random.randint(ACTIVE_MAJORITY_RANGE[0], min(n, ACTIVE_MAJORITY_RANGE[1]))
    left_col = _build_column(n, template_colors, active_majority, random.choice(template_colors))

    max_right_majority = active_majority - 1
    right_majority = random.randint(max(2, n // num_colors), max_right_majority)
    right_dominant = random.choice(template_colors)
    if len(template_colors) < 2:
        return None
    right_col = _build_column(n, template_colors, right_majority, right_dominant)

    if max(Counter(left_col).values()) <= max(Counter(right_col).values()):
        return None

    template = np.column_stack([left_col, right_col]).astype(int)
    ids_base = np.arange(n * 2, dtype=int).reshape(n, 2)
    return template, ids_base, template_colors


def _deduplicated_variants(
    template: np.ndarray, ids_base: np.ndarray
) -> list[dict]:
    raw = _dihedral_variants(template)
    raw_ids = _dihedral_variants(ids_base)
    seen: list[np.ndarray] = []
    result: list[dict] = []
    for (name, sub, _ids1), (_name2, _sub2, ids_arr) in zip(raw, raw_ids):
        if any(s.shape == sub.shape and np.array_equal(s, sub) for s in seen):
            continue
        try:
            direction, active_set = _active_side(sub)
        except AssertionError:
            continue
        active_list = _sorted_active_coords(active_set)
        result.append(
            {
                "name": name,
                "sub": sub,
                "ids": ids_arr,
                "direction": direction,
                "active": set(active_set),
                "active_list": active_list,
                "active_ids": tuple(int(ids_arr[r, c]) for r, c in active_list),
            }
        )
        seen.append(sub)
    return result


def _choose_layout_variants(template: np.ndarray, ids_base: np.ndarray) -> dict[str, dict] | None:
    variants = _deduplicated_variants(template, ids_base)
    by_dir_order: dict[tuple[tuple[int, int], str], dict] = {}
    for variant in variants:
        sub = variant["sub"]
        if sub.shape[0] >= sub.shape[1]:
            continue
        active_ids = variant["active_ids"]
        order = "asc" if active_ids[0] < active_ids[-1] else "desc"
        by_dir_order.setdefault((variant["direction"], order), variant)

    needed = [(UP, "asc"), (UP, "desc"), (DOWN, "asc"), (DOWN, "desc")]
    if any(key not in by_dir_order for key in needed):
        return None

    return {
        "top_carrier": by_dir_order[(UP, "asc")],
        "lower_inner": by_dir_order[(UP, "desc")],
        "upper_inner": by_dir_order[(DOWN, "asc")],
        "bottom_carrier": by_dir_order[(DOWN, "desc")],
    }


def _place_fixed_stack(H: int, W: int, layout: dict[str, dict]) -> dict[str, dict] | None:
    width = layout["top_carrier"]["sub"].shape[1]
    height = layout["top_carrier"]["sub"].shape[0]
    assert height == 2

    if width + 2 > W:
        return None

    top_gap = random.randint(1, 3)
    gap1 = random.randint(3, 5)
    gap2 = random.randint(3, 5)
    gap3 = random.randint(3, 5)

    top_r = top_gap
    upper_r = top_r + height + gap1
    lower_r = upper_r + height + gap2
    bottom_r = lower_r + height + gap3
    if bottom_r + height > H - 1:
        return None

    left = random.randint(1, W - width - 1)
    placed: dict[str, dict] = {}
    for name, row in (
        ("top_carrier", top_r),
        ("upper_inner", upper_r),
        ("lower_inner", lower_r),
        ("bottom_carrier", bottom_r),
    ):
        placed[name] = {
            **layout[name],
            "bbox": (row, left, row + height - 1, left + width - 1),
        }
    return placed


def _partner_hit_tid(source_sprite: dict, target_sprite: dict, seed_tid: int) -> int:
    src_locs = np.argwhere(source_sprite["ids"] == int(seed_tid))
    assert src_locs.shape[0] == 1
    _lr, src_lc = map(int, src_locs[0])
    target_active_row = target_sprite["active_list"][0][0]
    target_tid = int(target_sprite["ids"][target_active_row, src_lc])
    return target_tid


def _marker_plan(placed: dict[str, dict]) -> list[dict] | None:
    width = placed["top_carrier"]["sub"].shape[1]
    pair_indices = list(range(width // 2))
    if len(pair_indices) < NUM_MARKERS:
        return None
    cols = random.sample(pair_indices, NUM_MARKERS)
    # Keep both legal top/bottom splits (1+2 and 2+1) equally likely so the
    # two carrier-activation branches are covered without a structural bias.
    top_count = random.choice([1, 2])
    top_cols = cols[:top_count]
    bottom_cols = cols[top_count:]
    if not bottom_cols:
        top_cols, bottom_cols = cols[:2], cols[2:]

    available = [c for c in range(10) if c != BG and c not in set(int(x) for x in placed["top_carrier"]["sub"].flat)]
    if len(available) < NUM_MARKERS:
        return None
    colors = random.sample(available, NUM_MARKERS)

    plan: list[dict] = []
    for color, col in zip(colors[: len(top_cols)], top_cols):
        tid = int(placed["top_carrier"]["ids"][placed["top_carrier"]["active_list"][col][0], placed["top_carrier"]["active_list"][col][1]])
        hit_tid = _partner_hit_tid(placed["upper_inner"], placed["lower_inner"], tid)
        if hit_tid == tid:
            return None
        plan.append({"side": "top", "color": color, "seed_tid": tid, "hit_tid": hit_tid, "col": col})
    for color, col in zip(colors[len(top_cols) :], bottom_cols):
        tid = int(placed["bottom_carrier"]["ids"][placed["bottom_carrier"]["active_list"][col][0], placed["bottom_carrier"]["active_list"][col][1]])
        hit_tid = _partner_hit_tid(placed["lower_inner"], placed["upper_inner"], tid)
        if hit_tid == tid:
            return None
        plan.append({"side": "bottom", "color": color, "seed_tid": tid, "hit_tid": hit_tid, "col": col})

    seen_pairs: set[frozenset[int]] = set()
    for item in plan:
        pair = frozenset({int(item["seed_tid"]), int(item["hit_tid"])})
        if pair in seen_pairs:
            return None
        seen_pairs.add(pair)
    return plan


def _paint_sprite(grid: np.ndarray, sprite: dict) -> None:
    r0, c0, r1, c1 = sprite["bbox"]
    grid[r0 : r1 + 1, c0 : c1 + 1] = sprite["sub"]


def _marker_cell(sprite: dict, template_id: int, H: int) -> tuple[int, int]:
    lr, lc = _locate_template_port(sprite, template_id)
    r0, c0, _r1, _c1 = sprite["bbox"]
    if sprite["direction"] == UP:
        return 0, c0 + lc
    return H - 1, c0 + lc


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*TEMPLATE_N_RANGE)
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    design = _design_template(n)
    if design is None:
        return None
    template, ids_base, _template_colors = design

    layout = _choose_layout_variants(template, ids_base)
    if layout is None:
        return None
    placed = _place_fixed_stack(H, W, layout)
    if placed is None:
        return None

    marker_plan = _marker_plan(placed)
    if marker_plan is None:
        return None

    grid = np.full((H, W), BG, dtype=int)
    for sprite in placed.values():
        _paint_sprite(grid, sprite)

    used_marker_cells: set[tuple[int, int]] = set()
    for item in marker_plan:
        sprite = placed["top_carrier"] if item["side"] == "top" else placed["bottom_carrier"]
        mr, mc = _marker_cell(sprite, int(item["seed_tid"]), H)
        if (mr, mc) in used_marker_cells or int(grid[mr, mc]) != BG:
            return None
        used_marker_cells.add((mr, mc))
        grid[mr, mc] = int(item["color"])

    try:
        metrics = _metrics_for_instance(grid)
        assert metrics["cross_changed_colors"] == NUM_MARKERS, "every marker must cross-remap"
        assert metrics["colors_with_chain_expansion"] == NUM_MARKERS, "every color must trigger a chain"
        assert metrics["total_activated_states"] >= 2 * NUM_MARKERS, "chain reaction must add states"
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")


def _metrics_for_instance(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg, sprites, _markers, cell_to_sprite = _extract_scene(g)
    roles = _identify_roles(sprites)
    initial = _initial_activations(g, bg, sprites, roles, cell_to_sprite)

    initial_map = {int(color): int(tid) for color, tid in initial}
    top_carrier = sprites[roles["top_carrier"]]
    bottom_carrier = sprites[roles["bottom_carrier"]]
    direct_map = {int(color): int(tid) for tid, color in top_carrier["port_markers"].items()}
    direct_map.update({int(color): int(tid) for tid, color in bottom_carrier["port_markers"].items()})
    cross_changed = sum(int(initial_map[color] != direct_map[color]) for color in initial_map)

    queue = deque(initial)
    seen: set[tuple[int, int]] = set(queue)
    activated_by_color: dict[int, set[int]] = {}
    while queue:
        color, template_id = queue.popleft()
        activated_by_color.setdefault(int(color), set()).add(int(template_id))
        for sprite in sprites:
            lr, lc = _locate_template_port(sprite, template_id)
            r0, c0, _r1, _c1 = sprite["bbox"]
            dr, dc = sprite["direction"]
            hit_si, target_lr, target_lc, _path = _trace_first_hit(
                g,
                bg,
                None,
                cell_to_sprite,
                r0 + lr,
                c0 + lc,
                dr,
                dc,
                None,
            )
            if hit_si is None:
                continue
            hit_sprite = sprites[hit_si]
            if (target_lr, target_lc) not in hit_sprite["active"]:
                continue
            target_tid = int(hit_sprite["ids"][target_lr, target_lc])
            state = (int(color), target_tid)
            if state not in seen:
                seen.add(state)
                queue.append(state)

    chain_expanded = sum(int(len(tids) > 1) for tids in activated_by_color.values())
    return {
        "cross_changed_colors": int(cross_changed),
        "colors_with_chain_expansion": int(chain_expanded),
        "total_activated_states": int(sum(len(v) for v in activated_by_color.values())),
    }
