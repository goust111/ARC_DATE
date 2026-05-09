# Combo variant of 409aa875 (index 3)
# combo_spec: D4/3+D11/2
# Name: 连锁反应+传播直到稳定

import random
from collections import Counter, defaultdict, deque

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8

# Design analysis for combo D4/3 + D11/2
# a) Concrete behavior:
#    Keep the original geometry: every 3-pixel triangle still has one apex, one encoded direction, and one normal
#    landing cell at apex + 5*dir. The fusion changes what happens after a sprite is struck. Example: sprite A's normal
#    shot lands on sprite B. Under the original task, B just turns maroon. Under the combo, B turns maroon and becomes
#    newly active, so it emits a continuation shot that advances one more 5-cell segment along B's own direction, i.e.
#    toward B.apex + 10*dir. If that continuation hits C, then C also activates and continues, and so on until no new
#    sprite is reached. Background cells still accumulate all shots across the whole cascade; one shot paints 9, two or
#    more shots paint 1.
# b) Complexity calibration:
#    Letting an activated sprite re-fire only to apex + 5*dir would be visually indistinguishable from the original
#    simultaneous first wave, because that shot already existed. Using the continued 10-step landing is the minimal
#    non-degenerate interpretation of "沿原方向继续", and requiring generator samples with chain depth >= 2 makes the
#    fusion visible in a few examples without adding extra object types or hidden state.
# c) Broken assumptions in the original solver:
#    - A single `dest_to_shooters` table is no longer enough; hit results can create new shots later.
#    - A flat `hit_components` set is no longer enough; newly hit sprites must be queued exactly once for stable BFS.
#    - Output is no longer determined only by first-wave 5-step landings; continuation shots may add both sprite hits and
#      background marks after the initial simultaneous pass.
#    - Generator samples that merely contain first-wave hits would silently regress to the original task, so the new
#      generator must reject inputs where combo and original outputs match.
# d) Placement legality:
#    The generator keeps the original 16x16 canvas, 3-pixel L/V sprites, and Chebyshev>=2 spacing by using placement
#    halos. For chain construction it works backward: terminal sprite first, then sprites whose continuation shots can hit
#    the next node, then a root whose normal 5-step shot starts the cascade. Normal 5-step landing cells of internal
#    chain nodes are reserved as background so the cascade is not accidentally solved in round 1.
# e) Visual observability:
#    Accepted samples must show at least two activated sprites and chain depth >= 2, and the combo output must differ
#    from the original single-round output on the same input. The generator also rejects one-family shape sets so both
#    axis-fired V sprites and diagonal-fired L sprites stay visible across training samples. This makes the extra
#    propagation leg the only plausible explanation for the additional recolored sprites / markers.
# f) Visual cleanliness:
#    The fusion still uses only three visible states: untouched sprite color, maroon activations / lone landings, and
#    blue multi-shot background collisions. No extra guide marks are introduced; spacing carries the chain structure.
# g) Inferability of change:
#    Color semantics stay sparse and repeated across samples. 9 always means either a struck sprite or a singly reached
#    background cell, and 1 always means repeated background impact. The new inferential burden is purely structural:
#    some maroon sprites create downstream effects after being hit, which is visible because additional remote changes
#    appear beyond the first-wave 5-step targets.
#
# Review of new solver assumptions:
# - `_triangle_apex_and_dir` assumes every non-bg component is a valid 3-pixel L or V. `_validate_input` enforces that.
# - `_collect_scene` assumes each sprite's base landing cell is in bounds. The generator precomputes placements so this is
#   always true for constructed samples.
# - `_simulate_cascade` assumes each activated sprite should emit at most one continuation shot, so `chain_fired` guards
#   against repeated processing when several shots hit the same sprite.
# - Background collision counting now spans both base and continuation shots. The generator deliberately allows optional
#   extra background shots, but keeps them sparse so the scene remains readable.


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


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


def _collect_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    comps: list[list[tuple[int, int]]] = []
    cell_to_comp: dict[tuple[int, int], int] = {}
    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        for comp in _cc8(g == col):
            cid = len(comps)
            comps.append(comp)
            for rc in comp:
                cell_to_comp[rc] = cid

    shots: dict[int, dict] = {}
    for cid, comp in enumerate(comps):
        ad = _triangle_apex_and_dir(comp)
        if ad is None:
            continue
        (ar, ac), direction = ad
        dr, dc = direction
        base_dest = (ar + 5 * dr, ac + 5 * dc)
        if not (0 <= base_dest[0] < h and 0 <= base_dest[1] < w):
            continue
        chain_dest = (ar + 10 * dr, ac + 10 * dc)
        chain_in_bounds = 0 <= chain_dest[0] < h and 0 <= chain_dest[1] < w
        shots[cid] = {
            "apex": (ar, ac),
            "dir": direction,
            "base_dest": base_dest,
            "base_target_cid": cell_to_comp.get(base_dest),
            "chain_dest": chain_dest if chain_in_bounds else None,
            "chain_target_cid": cell_to_comp.get(chain_dest) if chain_in_bounds else None,
        }

    return {
        "grid": g,
        "bg": bg,
        "comps": comps,
        "cell_to_comp": cell_to_comp,
        "shots": shots,
    }


def _simulate_cascade(grid: np.ndarray) -> dict:
    scene = _collect_scene(grid)
    shots = scene["shots"]
    activated: set[int] = set()
    activation_depth: dict[int, int] = {}
    chain_fired: set[int] = set()
    bg_counts: Counter[tuple[int, int]] = Counter()
    queue: deque[tuple[int, int]] = deque()

    def _activate(cid: int, depth: int) -> None:
        if cid in activated:
            return
        activated.add(cid)
        activation_depth[cid] = depth
        if cid in shots and shots[cid]["chain_dest"] is not None:
            queue.append((cid, depth))

    for cid in sorted(shots):
        info = shots[cid]
        target_cid = info["base_target_cid"]
        if target_cid is None:
            bg_counts[info["base_dest"]] += 1
        else:
            _activate(target_cid, 1)

    while queue:
        cid, depth = queue.popleft()
        if cid in chain_fired or cid not in shots:
            continue
        chain_fired.add(cid)
        info = shots[cid]
        dest = info["chain_dest"]
        if dest is None:
            continue
        target_cid = info["chain_target_cid"]
        if target_cid is None:
            bg_counts[dest] += 1
        else:
            _activate(target_cid, depth + 1)

    scene["activated"] = activated
    scene["activation_depth"] = activation_depth
    scene["chain_fired"] = chain_fired
    scene["bg_counts"] = bg_counts
    scene["base_fired"] = set(shots)
    scene["max_depth"] = max(activation_depth.values(), default=0)
    return scene


def _render_variant_output(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    scene = _simulate_cascade(grid)
    out = np.array(grid, dtype=int, copy=True)
    for cid in scene["activated"]:
        for r, c in scene["comps"][cid]:
            out[r, c] = 9
    for (r, c), count in scene["bg_counts"].items():
        out[r, c] = 1 if count >= 2 else 9
    return out, scene


def _render_original_output(grid: np.ndarray) -> np.ndarray:
    scene = _collect_scene(grid)
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
        for r, c in scene["comps"][cid]:
            out[r, c] = 9
    for (r, c), count in bg_counts.items():
        out[r, c] = 1 if count >= 2 else 9
    return out


def variant_metrics(grid: np.ndarray) -> dict:
    scene = _simulate_cascade(grid)
    return {
        "base_shooter_count": len(scene["base_fired"]),
        "chain_shooter_count": len(scene["chain_fired"]),
        "activated_count": len(scene["activated"]),
        "max_chain_depth": int(scene["max_depth"]),
        "blue_bg_cells": sum(1 for count in scene["bg_counts"].values() if count >= 2),
        "bg_hit_cells": len(scene["bg_counts"]),
    }


def _shape_family_counts(scene: dict) -> tuple[int, int]:
    axis = 0
    diag = 0
    for info in scene["shots"].values():
        dr, dc = info["dir"]
        if abs(dr) + abs(dc) == 1:
            axis += 1
        else:
            diag += 1
    return axis, diag


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire triangle projectiles, then let struck sprites continue the cascade until no new hits remain.

    Input structure:
    - The background is the most common color. Every non-background 3-cell component is a monochrome triangle sprite,
      either an axis-aligned L or a diagonal V, and that shape determines one direction.
    - Each sprite still has the same apex-based first shot as in the original task: its normal landing cell is
      `apex + 5*dir`, provided that cell lies inside the 16x16 board.

    Rule:
    - Compute all normal 5-step landings simultaneously. A landing on empty background contributes one shot to that cell;
      a landing on another sprite activates that whole sprite and recolors it maroon (9).
    - An activated sprite emits exactly one continuation shot of its own, still following its stored direction and
      continuing one more 5-cell segment, so the continuation landing is `apex + 10*dir`. If that continuation reaches a
      fresh sprite, the new sprite activates and continues as well. Process these activations until the queue is empty.
    - Across the entire cascade, any background cell reached by one total shot becomes maroon (9), while a background
      cell reached by two or more total shots becomes blue (1).

    Key insight:
    - The first-wave geometry is unchanged, but the visible output is not a one-pass lookup anymore. You must recover each
      sprite's encoded direction, simulate the initial simultaneous hits, and then run a fixpoint BFS over newly struck
      sprites whose continuation shots can cause additional remote effects.
    """
    out, _scene = _render_variant_output(grid)
    return out


H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 2, 3, 4, 5, 6, 8]

PROJECTILE_DIST = 5
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

SPRITE_COUNT_RANGE = (4, 9)
MAX_CHAIN_HITS = 4
MAX_COLL_EVENTS = 1
MAX_ATTEMPTS = 600


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


def _halo(cells: tuple[tuple[int, int], ...]) -> frozenset[tuple[int, int]]:
    return frozenset(
        (rr, cc)
        for r, c in cells
        for rr in range(max(0, r - 1), min(H, r + 2))
        for cc in range(max(0, c - 1), min(W, c + 2))
    )


def _build_placements() -> tuple[list[dict], dict[tuple[int, int], list[dict]], dict[tuple[int, int], list[dict]]]:
    placements: list[dict] = []
    by_dest: dict[tuple[int, int], list[dict]] = defaultdict(list)
    by_chain_dest: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for ar in range(H):
        for ac in range(W):
            for dr, dc in ALL_DIRS:
                dest = (ar + PROJECTILE_DIST * dr, ac + PROJECTILE_DIST * dc)
                if not (0 <= dest[0] < H and 0 <= dest[1] < W):
                    continue
                chain_dest = (ar + 2 * PROJECTILE_DIST * dr, ac + 2 * PROJECTILE_DIST * dc)
                cells = _sprite_cells((ar, ac), (dr, dc))
                if not _in_bounds_cells(cells):
                    continue
                placement = {
                    "apex": (ar, ac),
                    "dir": (dr, dc),
                    "dest": dest,
                    "chain_dest": chain_dest if 0 <= chain_dest[0] < H and 0 <= chain_dest[1] < W else None,
                    "cells": cells,
                    "cell_set": frozenset(cells),
                    "halo": _halo(cells),
                }
                placements.append(placement)
                by_dest[dest].append(placement)
                if placement["chain_dest"] is not None:
                    by_chain_dest[placement["chain_dest"]].append(placement)
    return placements, by_dest, by_chain_dest


PLACEMENTS, PLACEMENTS_BY_DEST, PLACEMENTS_BY_CHAIN_DEST = _build_placements()


def _fits(placement: dict, occupied: set[tuple[int, int]], reserved_bg: set[tuple[int, int]]) -> bool:
    return not (placement["halo"] & occupied) and not (placement["cell_set"] & reserved_bg)


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _choose_unique(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> dict | None:
    candidates = PLACEMENTS[:]
    rng.shuffle(candidates)
    for placement in candidates:
        if placement["dest"] in reserved_bg or placement["dest"] in occupied:
            continue
        if _fits(placement, occupied, reserved_bg):
            return placement
    return None


def _choose_collision_pair(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[dict]] | None:
    dests = list(PLACEMENTS_BY_DEST)
    rng.shuffle(dests)
    for dest in dests:
        if dest in reserved_bg or dest in occupied:
            continue
        candidates = [p for p in PLACEMENTS_BY_DEST[dest] if _fits(p, occupied, reserved_bg)]
        rng.shuffle(candidates)
        for i, first in enumerate(candidates):
            occ2 = occupied | set(first["cells"])
            for second in candidates[i + 1 :]:
                if second["cell_set"] & first["cell_set"]:
                    continue
                if second["halo"] & occ2:
                    continue
                return dest, [first, second]
    return None


def _choose_hit(
    rng: random.Random,
    target_sprite: dict,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> tuple[dict, tuple[int, int]] | None:
    cells = list(target_sprite["placement"]["cells"])
    rng.shuffle(cells)
    for cell in cells:
        candidates = [p for p in PLACEMENTS_BY_DEST[cell] if _fits(p, occupied, reserved_bg)]
        rng.shuffle(candidates)
        if candidates:
            return candidates[0], cell
    return None


def _choose_chain_hit(
    rng: random.Random,
    target_sprite: dict,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> tuple[dict, tuple[int, int]] | None:
    cells = list(target_sprite["placement"]["cells"])
    rng.shuffle(cells)
    for cell in cells:
        candidates = []
        for placement in PLACEMENTS_BY_CHAIN_DEST[cell]:
            if placement["dest"] in reserved_bg or placement["dest"] in occupied:
                continue
            if _fits(placement, occupied, reserved_bg):
                candidates.append(placement)
        rng.shuffle(candidates)
        if candidates:
            return candidates[0], cell
    return None


def _validate_input(grid: np.ndarray, n_sprites: int) -> bool:
    if grid.shape != (H, W):
        return False
    if _bg_color(grid) != BG:
        return False
    colors = sorted(set(int(v) for v in np.unique(grid)) - {BG})
    if len(colors) != 1 or colors[0] not in SPRITE_COLORS:
        return False
    comps = _cc8(grid != BG)
    if len(comps) != n_sprites:
        return False
    for comp in comps:
        if len(comp) != 3:
            return False
        ad = _triangle_apex_and_dir(comp)
        if ad is None:
            return False
        (ar, ac), (dr, dc) = ad
        rr = ar + PROJECTILE_DIST * dr
        cc = ac + PROJECTILE_DIST * dc
        if not (0 <= rr < H and 0 <= cc < W):
            return False
    return True


def _counts_for_n(rng: random.Random, n_sprites: int) -> tuple[int, int, int] | None:
    options = []
    for chain_hits in range(2, min(MAX_CHAIN_HITS, n_sprites - 1) + 1):
        remaining = n_sprites - (chain_hits + 1)
        for e_coll in range(0, min(MAX_COLL_EVENTS, remaining // 2) + 1):
            unique = remaining - 2 * e_coll
            if unique < 0:
                continue
            options.append((chain_hits, e_coll, unique))
    if not options:
        return None

    with_collision = [opt for opt in options if opt[1] > 0]
    without_collision = [opt for opt in options if opt[1] == 0]
    if with_collision and without_collision:
        bucket = with_collision if rng.random() < 0.5 else without_collision
        return rng.choice(bucket)
    return rng.choice(options)


def _place_chain(
    rng: random.Random,
    chain_hits: int,
    sprites: list[dict],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> bool:
    terminals = [p for p in PLACEMENTS if p["chain_dest"] is not None]
    rng.shuffle(terminals)
    terminal = None
    for placement in terminals:
        if placement["dest"] in reserved_bg or placement["dest"] in occupied:
            continue
        if placement["chain_dest"] in reserved_bg or placement["chain_dest"] in occupied:
            continue
        if _fits(placement, occupied, reserved_bg):
            terminal = placement
            break
    if terminal is None:
        return False

    chain = [{"placement": terminal}]
    occupied.update(terminal["cells"])
    reserved_bg.add(terminal["dest"])
    reserved_bg.add(terminal["chain_dest"])
    current = chain[0]

    for _ in range(chain_hits - 1):
        choice = _choose_chain_hit(rng, current, occupied, reserved_bg)
        if choice is None:
            return False
        placement, _cell = choice
        chain.append({"placement": placement})
        occupied.update(placement["cells"])
        reserved_bg.add(placement["dest"])
        current = chain[-1]

    root_choice = _choose_hit(rng, current, occupied, reserved_bg)
    if root_choice is None:
        return False
    root, _cell = root_choice
    chain.append({"placement": root})
    occupied.update(root["cells"])

    chain.reverse()
    sprites.extend(chain)
    return True


def _place_collisions(
    rng: random.Random,
    e_coll: int,
    sprites: list[dict],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> bool:
    for _ in range(e_coll):
        choice = _choose_collision_pair(rng, occupied, reserved_bg)
        if choice is None:
            return False
        dest, pair = choice
        reserved_bg.add(dest)
        for placement in pair:
            sprites.append({"placement": placement})
            occupied.update(placement["cells"])
    return True


def _place_unique_landings(
    rng: random.Random,
    n: int,
    sprites: list[dict],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> bool:
    for _ in range(n):
        placement = _choose_unique(rng, occupied, reserved_bg)
        if placement is None:
            return False
        reserved_bg.add(placement["dest"])
        sprites.append({"placement": placement})
        occupied.update(placement["cells"])
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    sprite_color = rng.choice(SPRITE_COLORS)
    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)
    counts = _counts_for_n(rng, n_sprites)
    if counts is None:
        return None
    chain_hits, e_coll, unique_roots = counts

    sprites: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    reserved_bg: set[tuple[int, int]] = set()

    if not _place_chain(rng, chain_hits, sprites, occupied, reserved_bg):
        return None
    if not _place_collisions(rng, e_coll, sprites, occupied, reserved_bg):
        return None
    if not _place_unique_landings(rng, unique_roots, sprites, occupied, reserved_bg):
        return None

    inp = np.full((H, W), BG, dtype=int)
    for sprite in sprites:
        _paint_sprite(inp, sprite["placement"], sprite_color)

    if not _validate_input(inp, n_sprites):
        return None

    out, _scene = _render_variant_output(inp)
    old_out = _render_original_output(inp)
    metrics = variant_metrics(inp)
    axis_count, diag_count = _shape_family_counts(_scene)

    if metrics["max_chain_depth"] < 2:
        return None
    if metrics["activated_count"] < 2:
        return None
    if metrics["chain_shooter_count"] < 2:
        return None
    if axis_count == 0 or diag_count == 0:
        return None
    if np.array_equal(out, old_out):
        return None
    if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
        return None
    if not np.any(out == HIT_COLOR):
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
