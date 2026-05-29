# Combo variant of 409aa875 (index 20)
# combo_spec: D4/3+D8/1+D14/0
# Name: 连锁反应+密集精灵+紧密排列

import random
from collections import Counter, defaultdict, deque

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8


# Design analysis for the combo variant
# a) Concrete behavior:
#    The original task had isolated 3-cell triangles, so each sprite could be read
#    as one 8-connected component and only one 5-step shot mattered. Here 12-16
#    same-color triangles are packed tightly enough that many of them touch by an
#    edge or corner, so the visible input can contain larger same-color blobs. A
#    concrete trace is:
#      root R base-shot hits sprite A at apex+5*dir;
#      A turns maroon and then fires its continuation shot to apex+10*dir, hitting B;
#      B does the same and hits C; meanwhile untouched dense neighbors still fire
#      their own normal 5-step shots to empty background or collision cells.
#    Compared with the original logic, two things changed at once: touching blobs
#    must first be decomposed back into legal triangles, and struck sprites are now
#    active participants in later rounds instead of terminal hit targets.
# b) Complexity calibration:
#    Allowing arbitrary touching plus arbitrary cascades would create unreadable
#    tangles. The generator therefore keeps only one sprite color, uses the same
#    two canonical 3-cell triangle vocabularies, and forces the cascade to follow a
#    small number of directed hit chains while the remaining dense sprites mostly
#    contribute clean background impacts. This keeps the rule visibly richer than
#    the base task without making inference depend on hidden bookkeeping.
# c) Hidden assumptions broken in the original solver:
#    - "One 8-connected component equals one sprite" is false once same-color
#      sprites may touch. We must exact-cover each blob with legal 3-cell L/V
#      placements before reading directions.
#    - "All effects are known after one 5-step lookup" is false once hit sprites
#      emit continuation shots. We need queue-based propagation over newly
#      activated sprites.
#    - "Sparse placement prevents accidental ambiguity" is false with 12-16
#      sprites. The generator must validate unique decomposition and reserve all
#      intended empty landing cells so later sprite placements do not silently turn
#      background impacts into sprite hits.
# d) Placement legality:
#    The generator forbids only cell overlap, not Chebyshev-1 contact. It builds
#    explicit hit chains backward from terminal targets, reserves every intended
#    background landing cell, and finally verifies that the full dense scene has a
#    unique decomposition into legal 3-cell triangles.
# e) Visual observability:
#    Accepted samples must differ from the original one-pass transform on the same
#    input, must contain multi-round propagation, and must contain several touching
#    sprite pairs. That way the viewer can see both the dense touching arrangement
#    and the extra recoloring caused by the chain reaction.
# f) Visual cleanliness:
#    Although sprites are dense, only whole sprites recolor and empty-cell impacts
#    stay single-cell. The generator keeps the deliberate sprite-hit structure
#    narrow, so the output reads as maroon spread through many triangles plus a few
#    isolated maroon/blue landing markers, rather than as an overpainted mess.
# g) Color inference:
#    Untouched sprites always share one non-special color; activated sprites and
#    lone background impacts become 9; shared background landing cells become 1.
#    No rule depends on numeric color ordering or any other non-visual cue.


H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 2, 3, 4, 5, 6, 8]

PROJECTILE_DIST = 5
CHAIN_DIST = 10
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

SPRITE_COUNT_RANGE = (12, 16)
MIN_TOUCHING_PAIRS = 6
MIN_TOUCH_CLUSTER = 3
MIN_ACTIVATED = 6
MIN_CHAIN_DEPTH = 3
MAX_ATTEMPTS = 1400


def _triangle_apex_and_dir(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if len(comp) != 3:
        return None
    s = set(comp)

    deg4: dict[tuple[int, int], int] = {}
    deg8: dict[tuple[int, int], int] = {}
    for r, c in comp:
        deg4[(r, c)] = sum(1 for n in neighbors4(r, c) if n in s)
        deg8[(r, c)] = sum(1 for n in neighbors8(r, c) if n in s)

    corners = [p for p, d in deg4.items() if d == 2]
    if corners:
        corner = corners[0]
        cr, cc = corner
        arms = []
        for r, c in comp:
            if (r, c) == corner:
                continue
            dr, dc = r - cr, c - cc
            if abs(dr) + abs(dc) == 1:
                arms.append((dr, dc))
        if len(arms) != 2:
            return None
        dr_sum = arms[0][0] + arms[1][0]
        dc_sum = arms[0][1] + arms[1][1]
        dr = 0 if dr_sum == 0 else (-1 if dr_sum > 0 else 1)
        dc = 0 if dc_sum == 0 else (-1 if dc_sum > 0 else 1)
        if abs(dr) + abs(dc) != 2:
            return None
        return corner, (dr, dc)

    apexes = [p for p, d in deg8.items() if d == 2]
    if len(apexes) != 1:
        return None
    apex = apexes[0]
    ar, ac = apex
    others = [p for p in comp if p != apex]
    d1 = (others[0][0] - ar, others[0][1] - ac)
    d2 = (others[1][0] - ar, others[1][1] - ac)
    dr = -((d1[0] + d2[0]) // 2)
    dc = -((d1[1] + d2[1]) // 2)
    if abs(dr) + abs(dc) != 1:
        return None
    return apex, (dr, dc)


def _sprite_cells(apex: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    ar, ac = apex
    dr, dc = direction
    if abs(dr) + abs(dc) == 2:
        return ((ar, ac), (ar - dr, ac), (ar, ac - dc))
    if dr != 0:
        return ((ar, ac), (ar - dr, ac - 1), (ar - dr, ac + 1))
    return ((ar, ac), (ar - 1, ac - dc), (ar + 1, ac - dc))


def _in_bounds_cells(cells: tuple[tuple[int, int], ...]) -> bool:
    return all(0 <= r < H and 0 <= c < W for r, c in cells)


def _touch_kind(cells_a: tuple[tuple[int, int], ...], cells_b: tuple[tuple[int, int], ...]) -> str | None:
    kind = None
    for r1, c1 in cells_a:
        for r2, c2 in cells_b:
            dr = abs(r1 - r2)
            dc = abs(c1 - c2)
            if max(dr, dc) != 1:
                continue
            if dr + dc == 1:
                return "edge"
            kind = "diag"
    return kind


def _placement_sort_key(placement: dict) -> tuple:
    return placement["apex"], placement["dir"], placement["cells"]


def _make_placement(apex: tuple[int, int], direction: tuple[int, int], pid: int) -> dict | None:
    dest = (apex[0] + PROJECTILE_DIST * direction[0], apex[1] + PROJECTILE_DIST * direction[1])
    if not (0 <= dest[0] < H and 0 <= dest[1] < W):
        return None
    chain_dest = (apex[0] + CHAIN_DIST * direction[0], apex[1] + CHAIN_DIST * direction[1])
    cells = tuple(sorted(_sprite_cells(apex, direction)))
    if not _in_bounds_cells(cells):
        return None
    ad = _triangle_apex_and_dir(list(cells))
    if ad != (apex, direction):
        return None
    return {
        "id": pid,
        "apex": apex,
        "dir": direction,
        "dest": dest,
        "chain_dest": chain_dest if 0 <= chain_dest[0] < H and 0 <= chain_dest[1] < W else None,
        "cells": cells,
        "cell_set": frozenset(cells),
    }


def _build_placements() -> tuple[
    list[dict],
    dict[int, dict],
    dict[tuple[int, int], list[dict]],
    dict[tuple[int, int], list[dict]],
    dict[tuple[int, int], list[dict]],
    dict[int, list[int]],
]:
    placements: list[dict] = []
    by_id: dict[int, dict] = {}
    by_dest: dict[tuple[int, int], list[dict]] = defaultdict(list)
    by_chain_dest: dict[tuple[int, int], list[dict]] = defaultdict(list)
    by_cell: dict[tuple[int, int], list[dict]] = defaultdict(list)
    pid = 0
    for ar in range(H):
        for ac in range(W):
            for direction in ALL_DIRS:
                placement = _make_placement((ar, ac), direction, pid)
                if placement is None:
                    continue
                placements.append(placement)
                by_id[pid] = placement
                by_dest[placement["dest"]].append(placement)
                if placement["chain_dest"] is not None:
                    by_chain_dest[placement["chain_dest"]].append(placement)
                for cell in placement["cells"]:
                    by_cell[cell].append(placement)
                pid += 1

    touching: dict[int, list[int]] = defaultdict(list)
    for i, first in enumerate(placements):
        for second in placements[i + 1 :]:
            if first["cell_set"] & second["cell_set"]:
                continue
            if _touch_kind(first["cells"], second["cells"]) is None:
                continue
            touching[first["id"]].append(second["id"])
            touching[second["id"]].append(first["id"])

    for cell in by_cell:
        by_cell[cell].sort(key=_placement_sort_key)
    return placements, by_id, by_dest, by_chain_dest, by_cell, touching


PLACEMENTS, PLACEMENT_BY_ID, PLACEMENTS_BY_DEST, PLACEMENTS_BY_CHAIN_DEST, PLACEMENTS_BY_CELL, TOUCHING_BY_ID = (
    _build_placements()
)


def _build_predecessor_maps(
    index: dict[tuple[int, int], list[dict]],
) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for target in PLACEMENTS:
        pred_ids: set[int] = set()
        for cell in target["cells"]:
            for placement in index.get(cell, []):
                if placement["cell_set"] & target["cell_set"]:
                    continue
                pred_ids.add(placement["id"])
        out[target["id"]] = sorted((PLACEMENT_BY_ID[pid] for pid in pred_ids), key=_placement_sort_key)
    return out


BASE_PREDS_BY_TARGET_ID = _build_predecessor_maps(PLACEMENTS_BY_DEST)
CHAIN_PREDS_BY_TARGET_ID = _build_predecessor_maps(PLACEMENTS_BY_CHAIN_DEST)


def _cluster_candidates(cells: list[tuple[int, int]]) -> dict[tuple[int, int], list[dict]]:
    cell_set = frozenset(cells)
    candidate_ids: set[int] = set()
    for cell in cell_set:
        for placement in PLACEMENTS_BY_CELL.get(cell, []):
            if placement["cell_set"] <= cell_set:
                candidate_ids.add(placement["id"])

    by_cell: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for pid in candidate_ids:
        placement = PLACEMENT_BY_ID[pid]
        for cell in placement["cells"]:
            if cell in cell_set:
                by_cell[cell].append(placement)
    for cell in by_cell:
        by_cell[cell].sort(key=_placement_sort_key)
    return by_cell


def _solve_cluster_cover(cells: list[tuple[int, int]], limit: int = 1) -> tuple[int, list[dict] | None]:
    if len(cells) % 3 != 0:
        return 0, None

    remaining0 = frozenset(cells)
    by_cell = _cluster_candidates(cells)
    for cell in remaining0:
        if cell not in by_cell:
            return 0, None

    count = 0
    first_solution: list[dict] | None = None
    stack: list[dict] = []

    def dfs(remaining: frozenset[tuple[int, int]]) -> None:
        nonlocal count, first_solution
        if count >= limit:
            return
        if not remaining:
            count += 1
            if first_solution is None:
                first_solution = stack.copy()
            return

        pivot = None
        pivot_options = None
        best_len = None
        for cell in sorted(remaining):
            options = [placement for placement in by_cell[cell] if placement["cell_set"] <= remaining]
            if not options:
                return
            option_len = len(options)
            if best_len is None or option_len < best_len:
                best_len = option_len
                pivot = cell
                pivot_options = options
                if best_len == 1:
                    break

        assert pivot is not None and pivot_options is not None
        for placement in pivot_options:
            stack.append(placement)
            dfs(remaining - placement["cell_set"])
            stack.pop()
            if count >= limit:
                return

    dfs(remaining0)
    return count, first_solution


def _extract_sprites(grid: np.ndarray, require_unique: bool = False) -> tuple[list[dict], dict[tuple[int, int], int]] | None:
    bg = _bg_color(grid)
    sprites: list[dict] = []
    cell_to_comp: dict[tuple[int, int], int] = {}

    for color in sorted(set(int(v) for v in np.unique(grid)) - {bg}):
        for cluster in components_mask(grid == color, connectivity=8):
            cells = [(int(r), int(c)) for r, c in cluster]
            count, solution = _solve_cluster_cover(cells, limit=2 if require_unique else 1)
            if solution is None or count == 0:
                return None
            if require_unique and count != 1:
                return None
            assert solution is not None
            for placement in sorted(solution, key=_placement_sort_key):
                cid = len(sprites)
                sprites.append(placement)
                for rc in placement["cells"]:
                    cell_to_comp[rc] = cid

    return sprites, cell_to_comp


def _collect_scene(grid: np.ndarray, require_unique: bool = False) -> dict:
    g = np.asarray(grid, dtype=int)
    extracted = _extract_sprites(g, require_unique=require_unique)
    if extracted is None:
        raise ValueError("Grid cannot be decomposed into legal triangle sprites")
    sprites, cell_to_comp = extracted

    shots: dict[int, dict] = {}
    for cid, sprite in enumerate(sprites):
        shots[cid] = {
            "dir": sprite["dir"],
            "base_dest": sprite["dest"],
            "base_target_cid": cell_to_comp.get(sprite["dest"]),
            "chain_dest": sprite["chain_dest"],
            "chain_target_cid": cell_to_comp.get(sprite["chain_dest"]) if sprite["chain_dest"] is not None else None,
        }

    return {
        "grid": g,
        "bg": _bg_color(g),
        "sprites": sprites,
        "cell_to_comp": cell_to_comp,
        "shots": shots,
    }


def _touch_graph_stats(sprites: list[dict]) -> tuple[int, int]:
    if not sprites:
        return 0, 0
    ids = [sprite["id"] for sprite in sprites]
    id_set = set(ids)
    neighbors: dict[int, set[int]] = {pid: set() for pid in ids}
    touch_pairs = 0
    for i, first in enumerate(sprites):
        for second in sprites[i + 1 :]:
            if _touch_kind(first["cells"], second["cells"]) is None:
                continue
            touch_pairs += 1
            neighbors[first["id"]].add(second["id"])
            neighbors[second["id"]].add(first["id"])

    seen: set[int] = set()
    largest = 0
    for pid in ids:
        if pid in seen:
            continue
        stack = [pid]
        seen.add(pid)
        size = 0
        while stack:
            cur = stack.pop()
            size += 1
            for nxt in neighbors[cur]:
                if nxt in id_set and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        largest = max(largest, size)
    return touch_pairs, largest


def _simulate_cascade(grid: np.ndarray, require_unique: bool = False) -> dict:
    scene = _collect_scene(grid, require_unique=require_unique)
    shots = scene["shots"]
    activated: set[int] = set()
    activation_depth: dict[int, int] = {}
    chain_fired: set[int] = set()
    bg_counts: Counter[tuple[int, int]] = Counter()
    queue: deque[tuple[int, int]] = deque()

    def activate(target_cid: int, depth: int) -> None:
        if target_cid in activated:
            return
        activated.add(target_cid)
        activation_depth[target_cid] = depth
        if shots[target_cid]["chain_dest"] is not None:
            queue.append((target_cid, depth))

    for cid in sorted(shots):
        info = shots[cid]
        target_cid = info["base_target_cid"]
        if target_cid is None:
            bg_counts[info["base_dest"]] += 1
            continue
        activate(target_cid, 1)

    while queue:
        cid, depth = queue.popleft()
        if cid in chain_fired:
            continue
        chain_fired.add(cid)
        info = shots[cid]
        dest = info["chain_dest"]
        if dest is None:
            continue
        target_cid = info["chain_target_cid"]
        if target_cid is None:
            bg_counts[dest] += 1
            continue
        activate(target_cid, depth + 1)

    touch_pairs, largest_cluster = _touch_graph_stats(scene["sprites"])
    scene["activated"] = activated
    scene["activation_depth"] = activation_depth
    scene["chain_fired"] = chain_fired
    scene["bg_counts"] = bg_counts
    scene["max_depth"] = max(activation_depth.values(), default=0)
    scene["touching_pairs"] = touch_pairs
    scene["largest_touch_cluster"] = largest_cluster
    return scene


def _render_variant_output(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    scene = _simulate_cascade(grid, require_unique=False)
    out = np.array(grid, dtype=int, copy=True)
    for cid in scene["activated"]:
        for r, c in scene["sprites"][cid]["cells"]:
            out[r, c] = HIT_COLOR
    for (r, c), count in scene["bg_counts"].items():
        out[r, c] = COLLISION_COLOR if count >= 2 else HIT_COLOR
    return out, scene


def _render_original_output(grid: np.ndarray) -> np.ndarray:
    scene = _collect_scene(grid, require_unique=False)
    out = np.array(grid, dtype=int, copy=True)
    hit_components = {
        info["base_target_cid"]
        for info in scene["shots"].values()
        if info["base_target_cid"] is not None
    }
    bg_counts = Counter(
        info["base_dest"]
        for info in scene["shots"].values()
        if info["base_target_cid"] is None
    )
    for cid in hit_components:
        for r, c in scene["sprites"][cid]["cells"]:
            out[r, c] = HIT_COLOR
    for (r, c), count in bg_counts.items():
        out[r, c] = COLLISION_COLOR if count >= 2 else HIT_COLOR
    return out


def variant_metrics(grid: np.ndarray) -> dict:
    out, scene = _render_variant_output(grid)
    old_out = _render_original_output(grid)
    return {
        "sprite_count": len(scene["sprites"]),
        "activated_count": len(scene["activated"]),
        "max_chain_depth": int(scene["max_depth"]),
        "chain_shooter_count": len(scene["chain_fired"]),
        "blue_bg_cells": sum(1 for count in scene["bg_counts"].values() if count >= 2),
        "bg_hit_cells": len(scene["bg_counts"]),
        "touching_pairs": int(scene["touching_pairs"]),
        "largest_touch_cluster": int(scene["largest_touch_cluster"]),
        "delta_vs_original": int(np.count_nonzero(out != old_out)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decompose dense touching triangle blobs, then propagate chained projectile hits.

    Input structure:
    - The background is the most common color. The non-background area contains 12-16 monochrome 3-cell triangle sprites, but many same-color sprites may touch by an edge or corner and merge into larger blobs.
    - Each hidden sprite is still one of the original two triangle types: an axis-aligned L that fires diagonally away from its missing 2x2 corner, or a diagonal V that fires along the axis opposite its opening.

    Rule:
    - First split every same-color blob into legal 3-cell triangle sprites, then fire every sprite's normal projectile to the cell 5 steps from its apex.
    - A projectile that lands on another sprite recolors that entire struck sprite to maroon (9). Unlike the base task, a struck sprite also emits one continuation projectile of its own, continuing along its stored direction to the cell 10 steps from that sprite's apex. Newly struck sprites keep doing this until no new sprite is activated.
    - Any projectile that lands on empty background paints that cell maroon (9), except that an empty cell reached by two or more total projectiles over the whole cascade becomes blue (1).

    Key insight:
    - Solving the puzzle now requires two stages: recover the hidden individual triangles from dense touching blobs, then run a multi-round activation queue because sprite-hit events can create further sprite-hit events downstream.
    """
    out, _scene = _render_variant_output(grid)
    return out


def _fits(placement: dict, occupied: set[tuple[int, int]], reserved_bg: set[tuple[int, int]]) -> bool:
    return not (placement["cell_set"] & occupied) and not (placement["cell_set"] & reserved_bg)


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _touch_score(placement: dict, occupied: set[tuple[int, int]]) -> int:
    seen: set[tuple[int, int]] = set()
    for r, c in placement["cells"]:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) == (r, c):
                    continue
                if (rr, cc) in occupied:
                    seen.add((rr, cc))
    return len(seen)


def _center_score(placement: dict) -> float:
    ar, ac = placement["apex"]
    return -(abs(ar - (H - 1) / 2) + abs(ac - (W - 1) / 2))


def _choose_scored(rng: random.Random, candidates: list[dict], occupied: set[tuple[int, int]]) -> dict | None:
    if not candidates:
        return None
    rng.shuffle(candidates)
    candidates.sort(
        key=lambda p: (_touch_score(p, occupied), len(TOUCHING_BY_ID[p["id"]]), _center_score(p)),
        reverse=True,
    )
    return rng.choice(candidates[: min(8, len(candidates))])


def _ranked_candidates(
    rng: random.Random,
    candidates: list[dict],
    occupied: set[tuple[int, int]],
    limit: int = 12,
) -> list[dict]:
    if not candidates:
        return []
    rng.shuffle(candidates)
    candidates.sort(
        key=lambda p: (_touch_score(p, occupied), len(TOUCHING_BY_ID[p["id"]]), _center_score(p)),
        reverse=True,
    )
    return candidates[: min(limit, len(candidates))]


def _candidate_chain_predecessors(
    target_sprite: dict,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> list[dict]:
    return [
        placement
        for placement in CHAIN_PREDS_BY_TARGET_ID[target_sprite["id"]]
        if placement["dest"] not in occupied
        and placement["dest"] not in reserved_bg
        and _fits(placement, occupied, reserved_bg)
    ]


def _candidate_base_predecessors(
    target_sprite: dict,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> list[dict]:
    return [
        placement
        for placement in BASE_PREDS_BY_TARGET_ID[target_sprite["id"]]
        if _fits(placement, occupied, reserved_bg)
    ]


def _search_chain_backward(
    rng: random.Random,
    target_sprite: dict,
    chain_steps_remaining: int,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> list[dict] | None:
    if chain_steps_remaining == 0:
        roots = _ranked_candidates(rng, _candidate_base_predecessors(target_sprite, occupied, reserved_bg), occupied)
        for root in roots:
            return [root, target_sprite]
        return None

    preds = _ranked_candidates(rng, _candidate_chain_predecessors(target_sprite, occupied, reserved_bg), occupied)
    for predecessor in preds:
        occ2 = occupied | set(predecessor["cells"])
        res2 = reserved_bg | {predecessor["dest"]}
        path = _search_chain_backward(rng, predecessor, chain_steps_remaining - 1, occ2, res2)
        if path is not None:
            return path + [target_sprite]
    return None


def _build_chain_path(
    rng: random.Random,
    chain_len: int,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> list[dict] | None:
    terminals = [
        placement
        for placement in PLACEMENTS
        if placement["chain_dest"] is not None
        and placement["dest"] not in occupied
        and placement["dest"] not in reserved_bg
        and placement["chain_dest"] not in occupied
        and placement["chain_dest"] not in reserved_bg
        and CHAIN_PREDS_BY_TARGET_ID[placement["id"]]
        and _fits(placement, occupied, reserved_bg)
    ]
    rng.shuffle(terminals)
    terminals.sort(
        key=lambda p: (_touch_score(p, occupied), len(TOUCHING_BY_ID[p["id"]]), _center_score(p)),
        reverse=True,
    )
    for terminal in terminals:
        occ1 = occupied | set(terminal["cells"])
        res1 = reserved_bg | {terminal["dest"], terminal["chain_dest"]}
        path = _search_chain_backward(rng, terminal, chain_len - 1, occ1, res1)
        if path is not None:
            return path
    return None


def _choose_dense_bg_shooter(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    require_touch: bool,
) -> dict | None:
    candidates = [
        placement
        for placement in PLACEMENTS
        if placement["dest"] not in occupied
        and placement["dest"] not in reserved_bg
        and _fits(placement, occupied, reserved_bg)
        and (not require_touch or _touch_score(placement, occupied) > 0)
    ]
    return _choose_scored(rng, candidates, occupied)


def _choose_collision_pair(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[dict]] | None:
    dests = list(PLACEMENTS_BY_DEST)
    rng.shuffle(dests)
    best: tuple[tuple[int, int], list[dict], tuple[int, int, float]] | None = None
    for dest in dests:
        if dest in occupied or dest in reserved_bg:
            continue
        candidates = [
            placement
            for placement in PLACEMENTS_BY_DEST[dest]
            if _fits(placement, occupied, reserved_bg)
        ]
        if len(candidates) < 2:
            continue
        rng.shuffle(candidates)
        for i, first in enumerate(candidates):
            occ2 = occupied | set(first["cells"])
            for second in candidates[i + 1 :]:
                if second["cell_set"] & first["cell_set"]:
                    continue
                if second["cell_set"] & reserved_bg:
                    continue
                score = (
                    _touch_score(first, occupied) + _touch_score(second, occ2),
                    len(TOUCHING_BY_ID[first["id"]]) + len(TOUCHING_BY_ID[second["id"]]),
                    _center_score(first) + _center_score(second),
                )
                if best is None or score > best[2]:
                    best = (dest, [first, second], score)
        if best is not None and best[2][0] >= 2:
            break
    if best is None:
        return None
    return best[0], best[1]


def _place_chain(
    rng: random.Random,
    chain_len: int,
    sprites: list[dict],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> bool:
    path = _build_chain_path(rng, chain_len, occupied, reserved_bg)
    if path is None:
        return False
    for i, placement in enumerate(path):
        sprites.append(placement)
        occupied.update(placement["cells"])
        if i > 0:
            reserved_bg.add(placement["dest"])
    reserved_bg.add(path[-1]["chain_dest"])
    return True


def _place_collision(
    rng: random.Random,
    sprites: list[dict],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> bool:
    choice = _choose_collision_pair(rng, occupied, reserved_bg)
    if choice is None:
        return False
    dest, pair = choice
    reserved_bg.add(dest)
    for placement in pair:
        sprites.append(placement)
        occupied.update(placement["cells"])
    return True


def _place_dense_fillers(
    rng: random.Random,
    count: int,
    sprites: list[dict],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> bool:
    for _ in range(count):
        placement = _choose_dense_bg_shooter(rng, occupied, reserved_bg, require_touch=True)
        if placement is None:
            placement = _choose_dense_bg_shooter(rng, occupied, reserved_bg, require_touch=False)
        if placement is None:
            return False
        reserved_bg.add(placement["dest"])
        sprites.append(placement)
        occupied.update(placement["cells"])
    return True


def _validate_input(grid: np.ndarray, n_sprites: int) -> bool:
    if grid.shape != (H, W):
        return False
    if _bg_color(grid) != BG:
        return False
    colors = sorted(set(int(v) for v in np.unique(grid)) - {BG})
    if len(colors) != 1 or colors[0] not in SPRITE_COLORS:
        return False
    extracted = _extract_sprites(grid, require_unique=True)
    if extracted is None:
        return False
    sprites, _cell_to_comp = extracted
    if len(sprites) != n_sprites:
        return False
    touch_pairs, largest_cluster = _touch_graph_stats(sprites)
    return touch_pairs >= MIN_TOUCHING_PAIRS and largest_cluster >= MIN_TOUCH_CLUSTER


def _plan_counts(rng: random.Random, n_sprites: int) -> tuple[list[int], bool, int]:
    if n_sprites <= 12:
        chain_lengths = [3, 3]
    elif n_sprites == 13:
        chain_lengths = [4, 3]
    else:
        chain_lengths = [4, 3] if rng.random() < 0.7 else [4, 4]
    want_collision = True
    used = sum(length + 1 for length in chain_lengths) + (2 if want_collision else 0)
    fillers = n_sprites - used
    if fillers < 0:
        chain_lengths = [3, 3]
        used = sum(length + 1 for length in chain_lengths) + 2
        fillers = n_sprites - used
    return chain_lengths, want_collision, fillers


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    sprite_color = rng.choice(SPRITE_COLORS)
    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)
    chain_lengths, want_collision, filler_count = _plan_counts(rng, n_sprites)
    if filler_count < 0:
        return None

    sprites: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    reserved_bg: set[tuple[int, int]] = set()

    for chain_len in chain_lengths:
        if not _place_chain(rng, chain_len, sprites, occupied, reserved_bg):
            return None

    if want_collision and not _place_collision(rng, sprites, occupied, reserved_bg):
        return None

    if not _place_dense_fillers(rng, filler_count, sprites, occupied, reserved_bg):
        return None

    inp = np.full((H, W), BG, dtype=int)
    for placement in sprites:
        _paint_sprite(inp, placement, sprite_color)

    if not _validate_input(inp, n_sprites):
        return None

    out, scene = _render_variant_output(inp)
    old_out = _render_original_output(inp)
    delta_vs_original = int(np.count_nonzero(out != old_out))
    activated_count = len(scene["activated"])

    if activated_count < MIN_ACTIVATED:
        return None
    if scene["max_depth"] < MIN_CHAIN_DEPTH:
        return None
    if scene["touching_pairs"] < MIN_TOUCHING_PAIRS:
        return None
    if scene["largest_touch_cluster"] < MIN_TOUCH_CLUSTER:
        return None
    if delta_vs_original < 9:
        return None
    if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
        return None
    if not np.any(out == HIT_COLOR) or not np.any(out == COLLISION_COLOR):
        return None
    if np.array_equal(inp, out):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_ATTEMPTS} attempts")
