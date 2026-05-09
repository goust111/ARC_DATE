# Combo variant of 64efde09 (index 6)
# combo_spec: D6/1+D16/0
# Name: 重着色链式传播

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg

# Design analysis for the combo variant
# -------------------------------------
# concrete_behavior:
#   input_family:
#     - Keep the original object grammar from 64efde09: 3-4 dense thin rectangles
#       that are dihedral transforms of one shared Nx2 template, plus exactly
#       3 isolated boundary markers with unique colors outside the template palette.
#     - The active side of every sprite is still the unique long side whose cells
#       have the larger repeated-color majority.
#   fused_output_rule:
#     - Original task: each marker color labels one template port, then every
#       transformed sprite emits that color outward from the corresponding port in
#       one global step, while sprite bodies stay unchanged.
#     - New combo:
#       1. The initial boundary markers still label three template ports exactly as
#          before.
#       2. Each labeled port immediately behaves like the old task: every sprite
#          emits that color from its corresponding port.
#       3. If any emitted ray first hits another sprite exactly on an active-edge
#          port, that hit template port becomes newly activated with the same color.
#       4. Newly activated ports trigger another global broadcast from every sprite's
#          corresponding copy of that template port.
#       5. Every activated port cell on every sprite is recolored to the propagating
#          color, so chain growth becomes visible inside the sprites as well as in
#          the outside rays.
#   traced_example:
#     - Suppose color 4 initially labels template port A.
#     - Old behavior: A emits from all sprites once; if one ray reaches another
#       sprite, it merely stops there.
#     - New behavior: if color-4 port A on a right-facing sprite hits port B on a
#       left-facing sprite, then B itself turns color 4 on every sprite and B emits
#       in a second round. If a copy of B later hits port C, then C also turns 4 and
#       emits in a third round.
#     - Visible difference: the output now shows a multi-stage same-color network:
#       boundary marker -> first family of rays -> recolored ports A -> second family
#       of rays from recolored ports B -> possibly more recolored ports and rays.
# complexity_calibration:
#   - The core freedom is kept small and repeated: exactly 3 marker colors, all
#     disjoint from template colors, and all propagation still lives on the active
#     edge of one shared thin template.
#   - Chain growth happens on template-port identities, not on arbitrary hit cells,
#     so the rule stays structural rather than numeric.
#   - The generator always forces at least one extra activated template port beyond
#     the 3 seed ports; otherwise the sample would collapse back to the simpler D6
#     recolor-only behavior.
# implicit_assumption_audit:
#   - Component parsing still assumes markers are area-1 components and sprites are
#     dense rectangles. The combo keeps the input unchanged, so that remains valid.
#   - Orientation matching still depends on the input sprites retaining their
#     original colors; therefore all recoloring is applied only to the output copy.
#   - The original broadcast loop assumed a fixed color<->template-port bijection
#     known up front. That breaks under chain reaction, so the solver now maintains a
#     queue of newly activated template ports.
#   - Ray obstruction must still be checked against the original input grid `g`,
#     not the progressively recolored output, otherwise an early recolored port would
#     incorrectly block later broadcasts.
#   - Hitting a non-active-edge sprite cell must not trigger propagation. The solver
#     therefore only chains through the first non-bg cell when it is one of the
#     precomputed active-edge ports.
# placement_legality:
#   - The generator uses one structured four-sprite layout made of two left/right
#     pairs placed on separate rows. This makes a readable horizontal chain possible
#     while keeping all sprites in-bounds and padded apart.
#   - Marker corridors are checked explicitly so every marker sees exactly its
#     intended seed port and no other object.
#   - Candidate inputs are accepted only after running the combo solver and the
#     no-chain baseline, rejecting collisions, ambiguous hits, and degenerate cases.
# visual_observability:
#   - Human solvers can infer the combo from 2-3 examples because every active color
#     appears in three synchronized places:
#       1. its seed boundary marker,
#       2. its external rays,
#       3. every activated homologous port embedded inside the sprites.
#   - Chain expansion is visible because some same-colored embedded ports and rays
#     appear even though no boundary marker was aligned to those later template ports.
# cleanliness:
#   - The generator keeps the chain sparse and axis-aligned. Only the two intended
#     left/right corridors are used for secondary hits, so the grid does not collapse
#     into a dense pile of overlapping beams.
# anti_degeneracy_metric:
#   - For every accepted sample, `extra_template_count = activated_template_count - 3`
#     must be at least 1, and the combo output must differ from the seed-only recolor
#     baseline. This quantifies that chain propagation, not just static recoloring,
#     is genuinely present.


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


def _extract_scene(grid: np.ndarray) -> tuple[np.ndarray, int, list[dict], dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
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

    assert len(sprites_raw) >= 2, "expected multiple transformed rectangular sprites"
    assert len(markers) == 3, "expected exactly three control markers"
    assert len(set(markers.values())) == len(markers), "control marker colors must be unique"

    variants = _dihedral_variants(sprites_raw[0]["sub"])
    sprites: list[dict] = []
    active_port_lookup: dict[tuple[int, int], int] = {}
    for sprite in sprites_raw:
        matches = [
            ids
            for _name, sub, ids in variants
            if sub.shape == sprite["sub"].shape and np.array_equal(sub, sprite["sub"])
        ]
        assert len(matches) == 1, "each sprite must match the template in exactly one orientation"
        ids = matches[0]
        direction, active = _active_side(sprite["sub"])
        port_by_tid = {int(ids[lr, lc]): (int(lr), int(lc)) for lr, lc in active}
        sprite_idx = len(sprites)
        sprites.append({**sprite, "ids": ids, "direction": direction, "active": active, "port_by_tid": port_by_tid})
        r0, c0, _r1, _c1 = sprite["bbox"]
        for tid, (lr, lc) in port_by_tid.items():
            active_port_lookup[(r0 + lr, c0 + lc)] = int(tid)

    return g, bg, sprites, markers, active_port_lookup


def _simulate(
    grid: np.ndarray,
    *,
    chain_reaction: bool,
    recolor_ports: bool,
) -> tuple[np.ndarray, dict]:
    g, bg, sprites, markers, active_port_lookup = _extract_scene(grid)

    seed_color_to_tid: dict[int, int] = {}
    template_id_to_color: dict[int, int] = {}
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        dr, dc = sprite["direction"]
        for lr, lc in sprite["active"]:
            marker_color = _first_marker_on_ray(g, bg, markers, r0 + lr, c0 + lc, dr, dc)
            if marker_color is None:
                continue
            template_id = int(sprite["ids"][lr, lc])
            if marker_color in seed_color_to_tid:
                assert seed_color_to_tid[marker_color] == template_id
            if template_id in template_id_to_color:
                assert template_id_to_color[template_id] == marker_color
            seed_color_to_tid[marker_color] = template_id
            template_id_to_color[template_id] = marker_color

    assert set(seed_color_to_tid) == set(markers.values()), "every marker must label one active-edge cell"

    out = g.copy()
    H, W = g.shape
    queue: deque[tuple[int, int]] = deque()
    processed: set[tuple[int, int]] = set()
    depth_by_pair: dict[tuple[int, int], int] = {}
    activation_edges: list[tuple[int, int, int]] = []

    for color, template_id in seed_color_to_tid.items():
        pair = (int(color), int(template_id))
        queue.append(pair)
        depth_by_pair[pair] = 0

    while queue:
        color, template_id = queue.popleft()
        pair = (int(color), int(template_id))
        if pair in processed:
            continue
        processed.add(pair)

        for sprite in sprites:
            assert template_id in sprite["port_by_tid"], "broadcast template id must stay on the active edge"
            lr, lc = sprite["port_by_tid"][template_id]
            r0, c0, _r1, _c1 = sprite["bbox"]
            rr = int(r0 + lr)
            cc = int(c0 + lc)
            if recolor_ports:
                out[rr, cc] = int(color)

            dr, dc = sprite["direction"]
            r = int(rr + dr)
            c = int(cc + dc)
            while 0 <= r < H and 0 <= c < W:
                if int(g[r, c]) != bg:
                    if chain_reaction:
                        hit_tid = active_port_lookup.get((r, c))
                        if hit_tid is not None:
                            prev = template_id_to_color.get(hit_tid)
                            if prev is None:
                                template_id_to_color[hit_tid] = int(color)
                                new_pair = (int(color), int(hit_tid))
                                queue.append(new_pair)
                                depth_by_pair[new_pair] = depth_by_pair[pair] + 1
                                activation_edges.append((int(template_id), int(hit_tid), int(color)))
                            else:
                                assert prev == color, "each activated template port must keep one color"
                    break
                assert int(out[r, c]) in (bg, color), "propagated rays must not conflict"
                out[r, c] = int(color)
                r += dr
                c += dc

    info = {
        "seed_template_ids": sorted(int(tid) for tid in seed_color_to_tid.values()),
        "activated_template_ids": sorted(int(tid) for tid in template_id_to_color),
        "extra_template_ids": sorted(int(tid) for tid in set(template_id_to_color) - set(seed_color_to_tid.values())),
        "template_id_to_color": {int(k): int(v) for k, v in template_id_to_color.items()},
        "max_depth": max(depth_by_pair.values(), default=0),
        "activation_edges": activation_edges,
    }
    return out, info


def _seed_only_output(grid: np.ndarray) -> np.ndarray:
    return _simulate(grid, chain_reaction=False, recolor_ports=True)[0]


def _combo_metrics(grid: np.ndarray) -> dict[str, int]:
    combo_out, info = _simulate(grid, chain_reaction=True, recolor_ports=True)
    seed_only = _seed_only_output(grid)
    g, _bg, sprites, _markers, _lookup = _extract_scene(grid)

    sprite_mask = np.zeros(g.shape, dtype=bool)
    for sprite in sprites:
        r0, c0, r1, c1 = sprite["bbox"]
        sprite_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    return {
        "seed_count": int(len(info["seed_template_ids"])),
        "activated_count": int(len(info["activated_template_ids"])),
        "extra_template_count": int(len(info["extra_template_ids"])),
        "max_depth": int(info["max_depth"]),
        "delta_vs_seed_only": int(np.count_nonzero(combo_out != seed_only)),
        "extra_recolored_ports": int(np.count_nonzero((combo_out != seed_only) & sprite_mask)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Diffuse boundary marker colors through template-matched port broadcasts, allowing ray hits to unlock later broadcasts.

    Input structure:
    - The background is the most frequent color.
    - Several dense 2-by-N or N-by-2 rectangles are dihedral copies of one thin template.
    - Three isolated boundary markers sit on outward sightlines from active-edge ports of some sprites.

    Rule:
    - First recover the seed template ports exactly as in the base task: a boundary marker seen first along an active-port ray gives that template port its color.
    - Each seed port then emits its color outward from the corresponding port on every transformed sprite; if a ray first hits another sprite exactly on one of its active-edge ports, that newly hit template port adopts the same color and becomes another broadcaster in the next BFS round.
    - Every template port that ever becomes active also recolors the corresponding embedded port cell on every sprite to that propagated color.

    Key insight:
    - The control colors no longer stop at one global replay. They spread over a network whose nodes are shared template ports: markers seed a few ports, ray hits activate more homologous ports, and the growing set of recolored embedded ports reveals the chain reaction directly.
    """
    return _simulate(grid, chain_reaction=True, recolor_ports=True)[0]


# --------------- generator (combo variant) ---------------

_BG = 8
_NON_BG_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]
_GRID_H_RANGE = (23, 29)
_GRID_W_RANGE = (27, 29)
_TEMPLATE_N_RANGE = (7, 9)
_TEMPLATE_COLOR_COUNT = (3, 4)
_ACTIVE_MAJORITY_RANGE = (5, 9)
_NUM_MARKERS = 3
_LAYOUT_RETRIES = 200
_GEN_RETRIES = 400
_PAD = 2


def _build_column(n: int, colors: list[int], majority: int, dominant: int) -> list[int]:
    col = [dominant] * majority
    others = [c for c in colors if c != dominant]
    fill = others if others else colors
    for _ in range(n - majority):
        col.append(random.choice(fill))
    random.shuffle(col)
    return col


def _design_template(N: int) -> tuple[np.ndarray, np.ndarray, list[int]] | None:
    num_colors = random.randint(*_TEMPLATE_COLOR_COUNT)
    template_colors = random.sample(_NON_BG_COLORS, num_colors)

    active_majority = random.randint(_ACTIVE_MAJORITY_RANGE[0], min(N, _ACTIVE_MAJORITY_RANGE[1]))
    left_col = _build_column(N, template_colors, active_majority, random.choice(template_colors))

    max_right_majority = active_majority - 1
    right_majority = random.randint(max(2, N // num_colors), max_right_majority)
    right_dominant = random.choice(template_colors)
    if all(c == right_dominant for c in template_colors):
        return None
    right_col = _build_column(N, template_colors, right_majority, right_dominant)

    if max(Counter(left_col).values()) <= max(Counter(right_col).values()):
        return None

    template = np.column_stack([left_col, right_col]).astype(int)
    if len(set(int(v) for v in template.flat)) < num_colors:
        return None

    ids_base = np.arange(N * 2, dtype=int).reshape(N, 2)
    return template, ids_base, template_colors


def _deduplicated_variants(
    template: np.ndarray,
    ids_base: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, tuple[int, int], dict[int, tuple[int, int]]]]:
    raw = _dihedral_variants(template)
    raw_ids = _dihedral_variants(ids_base)

    seen: list[np.ndarray] = []
    result = []
    for (_name, sub, _), (_id_name, _id_sub, ids_arr) in zip(raw, raw_ids):
        if any(s.shape == sub.shape and np.array_equal(s, sub) for s in seen):
            continue
        try:
            _direction, active_set = _active_side(sub)
        except AssertionError:
            continue
        direction, active_set = _active_side(sub)
        seen.append(sub)
        port_by_tid = {int(ids_arr[lr, lc]): (int(lr), int(lc)) for lr, lc in active_set}
        result.append((sub, ids_arr, direction, port_by_tid))
    return result


def _boundary_position(r: int, c: int, dr: int, dc: int, H: int, W: int) -> tuple[int, int]:
    if dr == 0 and dc == -1:
        return r, 0
    if dr == 0 and dc == 1:
        return r, W - 1
    if dr == -1 and dc == 0:
        return 0, c
    return H - 1, c


def _corridor_clear(
    grid: np.ndarray,
    start_r: int,
    start_c: int,
    dr: int,
    dc: int,
    end_r: int,
    end_c: int,
) -> bool:
    r = int(start_r + dr)
    c = int(start_c + dc)
    while (r, c) != (end_r, end_c):
        if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
            return False
        if int(grid[r, c]) != _BG:
            return False
        r += dr
        c += dc
    return True


def _paint_sprite(grid: np.ndarray, placement: dict) -> None:
    r0 = placement["r0"]
    c0 = placement["c0"]
    sub = placement["sub"]
    h, w = sub.shape
    grid[r0 : r0 + h, c0 : c0 + w] = sub


def _placements_overlap(a: dict, b: dict, pad: int) -> bool:
    ar0, ac0 = a["r0"], a["c0"]
    ah, aw = a["sub"].shape
    br0, bc0 = b["r0"], b["c0"]
    bh, bw = b["sub"].shape
    return not (
        ar0 + ah - 1 + pad < br0
        or br0 + bh - 1 + pad < ar0
        or ac0 + aw - 1 + pad < bc0
        or bc0 + bw - 1 + pad < ac0
    )


def _structured_input_attempt() -> np.ndarray | None:
    N = random.randint(*_TEMPLATE_N_RANGE)
    H = random.randint(*_GRID_H_RANGE)
    W = random.randint(*_GRID_W_RANGE)

    design = _design_template(N)
    if design is None:
        return None
    template, ids_base, template_colors = design

    variants = _deduplicated_variants(template, ids_base)
    by_dir: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray, tuple[int, int], dict[int, tuple[int, int]]]]] = {}
    for item in variants:
        by_dir.setdefault(item[2], []).append(item)
    if len(by_dir.get((0, 1), [])) < 2 or len(by_dir.get((0, -1), [])) < 2:
        return None

    active_template_ids = [int(ids_base[r, 0]) for r in range(N)]
    if len(active_template_ids) < 5:
        return None

    available_marker_colors = [c for c in range(10) if c != _BG and c not in template_colors]
    if len(available_marker_colors) < _NUM_MARKERS:
        return None

    for _ in range(_LAYOUT_RETRIES):
        color1, color2, color3 = random.sample(available_marker_colors, 3)

        right_choices = random.sample(by_dir[(0, 1)], 2)
        left_choices = random.sample(by_dir[(0, -1)], 2)
        (r1_sub, r1_ids, r1_dir, r1_ports), (r2_sub, r2_ids, r2_dir, r2_ports) = right_choices
        (l1_sub, l1_ids, l1_dir, l1_ports), (l2_sub, l2_ids, l2_dir, l2_ports) = left_choices

        h_vert = r1_sub.shape[0]
        w_vert = r1_sub.shape[1]
        cR1 = random.randint(3, 4)
        cL1 = cR1 + w_vert + random.randint(4, 5)
        cR2 = cL1 + w_vert + random.randint(4, 5)
        cL2 = cR2 + w_vert + random.randint(4, 5)
        if cL2 + w_vert >= W - 1:
            continue

        candidate_triples: list[tuple[int, int, int]] = []
        for t1 in active_template_ids:
            for t2 in active_template_ids:
                if t2 == t1 or abs(r1_ports[t1][0] - l1_ports[t2][0]) < N - 2:
                    continue
                for t3 in active_template_ids:
                    if t3 in (t1, t2):
                        continue
                    if abs(r2_ports[t2][0] - l2_ports[t3][0]) < N - 2:
                        continue
                    candidate_triples.append((t1, t2, t3))
        random.shuffle(candidate_triples)
        if not candidate_triples:
            continue

        accepted_layout = None
        for t1, t2, t3 in candidate_triples:
            shift1 = r1_ports[t1][0] - l1_ports[t2][0]
            shift2 = r2_ports[t2][0] - l2_ports[t3][0]

            low_r1 = max(2, 2 - shift1)
            high_r1 = min(H - h_vert - 3, H - h_vert - 3 - shift1)
            low_r2 = max(2, 2 - shift2)
            high_r2 = min(H - h_vert - 3, H - h_vert - 3 - shift2)
            if low_r1 > high_r1 or low_r2 > high_r2:
                continue

            rR1 = low_r1
            rL1 = rR1 + shift1
            rR2 = high_r2
            rL2 = rR2 + shift2

            def _r1_hits_l1(tid: int) -> bool:
                row = rR1 + r1_ports[tid][0]
                return rL1 <= row <= rL1 + h_vert - 1

            def _l1_hits_r1(tid: int) -> bool:
                row = rL1 + l1_ports[tid][0]
                return rR1 <= row <= rR1 + h_vert - 1

            def _r2_hits_l2(tid: int) -> bool:
                row = rR2 + r2_ports[tid][0]
                return rL2 <= row <= rL2 + h_vert - 1

            def _l2_hits_r2(tid: int) -> bool:
                row = rL2 + l2_ports[tid][0]
                return rR2 <= row <= rR2 + h_vert - 1

            if _l1_hits_r1(t1):
                continue

            safe_ids = [
                tid
                for tid in active_template_ids
                if tid not in (t1, t2, t3)
                and not _r1_hits_l1(tid)
                and not _l1_hits_r1(tid)
                and not _r2_hits_l2(tid)
                and not _l2_hits_r2(tid)
            ]
            if len(safe_ids) < 2:
                continue
            t4, t5 = random.sample(safe_ids, 2)

            placements = [
                {"r0": rR1, "c0": cR1, "sub": r1_sub, "ids": r1_ids, "direction": r1_dir, "port_by_tid": r1_ports},
                {"r0": rL1, "c0": cL1, "sub": l1_sub, "ids": l1_ids, "direction": l1_dir, "port_by_tid": l1_ports},
                {"r0": rR2, "c0": cR2, "sub": r2_sub, "ids": r2_ids, "direction": r2_dir, "port_by_tid": r2_ports},
                {"r0": rL2, "c0": cL2, "sub": l2_sub, "ids": l2_ids, "direction": l2_dir, "port_by_tid": l2_ports},
            ]
            if any(_placements_overlap(placements[i], placements[j], _PAD) for i in range(4) for j in range(i + 1, 4)):
                continue

            grid = np.full((H, W), _BG, dtype=int)
            for placement in placements:
                _paint_sprite(grid, placement)

            marker_specs = [
                (placements[1], t1, color1),
                (placements[1], t4, color2),
                (placements[1], t5, color3),
            ]
            marker_cells: set[tuple[int, int]] = set()
            ok = True
            for placement, tid, color in marker_specs:
                lr, lc = placement["port_by_tid"][tid]
                pr = int(placement["r0"] + lr)
                pc = int(placement["c0"] + lc)
                dr, dc = placement["direction"]
                mr, mc = _boundary_position(pr, pc, dr, dc, H, W)
                if (mr, mc) in marker_cells or grid[mr, mc] != _BG:
                    ok = False
                    break
                if mr in (0, H - 1) and mc in (0, W - 1):
                    ok = False
                    break
                if not _corridor_clear(grid, pr, pc, dr, dc, mr, mc):
                    ok = False
                    break
                marker_cells.add((mr, mc))
                grid[mr, mc] = int(color)
            if not ok:
                continue

            accepted_layout = grid
            break

        if accepted_layout is None:
            continue

        try:
            combo_out = main(accepted_layout)
            seed_only = _seed_only_output(accepted_layout)
            metrics = _combo_metrics(accepted_layout)
        except Exception:
            continue

        if metrics["seed_count"] != _NUM_MARKERS:
            continue
        if metrics["extra_template_count"] < 1:
            continue
        if metrics["max_depth"] < 1:
            continue
        if metrics["delta_vs_seed_only"] <= 0:
            continue
        if metrics["extra_recolored_ports"] < len(placements):
            continue
        if np.array_equal(combo_out, seed_only):
            continue
        return accepted_layout

    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _structured_input_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid combo sample after {_GEN_RETRIES} attempts")
