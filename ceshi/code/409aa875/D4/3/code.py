# Variant of 409aa875
# Dimension: D4 — 碰撞规则（投射物-精灵） (transform)
# Change: 投射物命中精灵时，整个被命中精灵重着色为栗色(9)... → 命中精灵产生连锁反应（继续发射）

import random
from collections import Counter, defaultdict, deque

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8

# Design analysis for the D4 variant
# a) Concrete behavior:
#    Triangles still encode one fixed direction and a normal 5-cell projectile landing point. The D4 change happens only
#    after a sprite is struck: it turns maroon and emits one extra projectile that continues along the same direction for
#    one more 5-cell segment, i.e. toward apex + 10*dir. Example: if A's normal shot hits B, and B's continuation shot
#    reaches C, the original task only recolors B; the variant recolors both B and C, and C may continue the chain.
# b) Complexity calibration:
#    A naive "hit sprite fires again to the same 5-cell destination" would collapse to the original behavior, because the
#    second shot would duplicate information already present in round 1. Using the continued 10-cell landing preserves the
#    original geometry while making multi-round propagation genuinely observable.
# c) Old assumptions that break:
#    - The original `dest_to_shooters` pass assumed every relevant effect was already determined by one 5-cell lookup.
#      Now we need a second, conditional 10-cell continuation lookup for activated sprites.
#    - `hit_components` as a flat set was sufficient before; now newly hit sprites must be queued so their continuation
#      shots can trigger further hits.
#    - The original generator could allow sprite color 9 because recoloring stayed visible enough; here activation is the
#      key cue, so generator colors exclude 9 and also exclude 1 to keep blue bg-collisions visually distinct.
# d) Placement legality:
#    The generator still uses the original 16x16 grid, 3-pixel L/V sprites, and Chebyshev>=2 separation. It constructs a
#    backward chain where the root uses a normal 5-step hit, later links use 10-step continuation hits, and all ordinary
#    5-step landings of chain nodes are reserved as background so the cascade is not pre-solved by round 1.
# e) Visual observability:
#    Every accepted sample must contain a continuation depth >= 2 and must differ from the original one-pass transform on
#    the exact same input. This guarantees that the chain reaction, not just the base shots, changes the output.
# f) Visual cleanliness:
#    Only whole sprites recolor and bg markers stay single-cell, so outputs remain readable. Collision points are sparse,
#    and chain structure is carried by spacing, not by overdrawn guide marks.
# g) Color inference:
#    The palette stays simple: untouched sprites keep one shared non-special color, activated sprites and lone bg hits
#    become 9, and multi-hit bg cells become 1. The key observable difference is the extra downstream hit from a struck
#    sprite, not any hidden color-id ordering.


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _triangle_apex_and_dir(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if len(comp) != 3:
        return None
    s = set(comp)

    deg4 = {}
    deg8 = {}
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
        rr, cc = base_dest
        if not (0 <= rr < h and 0 <= cc < w):
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
    activation_depth: dict[int, int] = {}
    activated: set[int] = set()
    chain_fired: set[int] = set()
    bg_counts: Counter[tuple[int, int]] = Counter()
    queue: deque[tuple[int, int]] = deque()

    def _activate(target_cid: int, depth: int) -> None:
        if target_cid in activated:
            return
        activated.add(target_cid)
        activation_depth[target_cid] = depth
        if target_cid in shots and shots[target_cid]["chain_dest"] is not None:
            queue.append((target_cid, depth))

    for cid in sorted(shots):
        info = shots[cid]
        target_cid = info["base_target_cid"]
        if target_cid is None:
            bg_counts[info["base_dest"]] += 1
            continue
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
            continue
        _activate(target_cid, depth + 1)

    scene["base_fired"] = set(shots)
    scene["chain_fired"] = chain_fired
    scene["activated"] = activated
    scene["bg_counts"] = bg_counts
    scene["activation_depth"] = activation_depth
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire triangle shots, then continue from struck sprites to build a chain reaction.

    Input structure:
    - The background is the most common color. Every non-background 3-cell component is a monochrome triangle sprite:
      either an axis-aligned L or a diagonal V, and its orientation encodes one firing direction.
    - The normal projectile for any sprite is still defined by a 5-cell step from the sprite's apex along that direction.

    Rule:
    - First recover each sprite's apex, direction, and normal 5-step landing cell, and fire all of those normal shots.
    - If a normal or continuation shot first lands on another sprite, the whole struck sprite turns maroon (9) and emits
      one continuation shot of its own, still along its stored direction, but continuing one more 5-cell segment to the
      10-step landing cell from that sprite's apex. Newly struck sprites keep propagating the same way.
    - Any shot landing on empty background paints that cell maroon (9), except that a background cell reached by two or
      more total shots over the whole cascade becomes blue (1).

    Key insight:
    - The puzzle is no longer solved by a single 5-step lookup. You must simulate the first wave, then keep a queue of
      struck sprites whose continuation shots can create further hits at 10-step distance.
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
    return rng.choice(options) if options else None


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

    out, scene = _render_variant_output(inp)
    old_out = _render_original_output(inp)
    metrics = variant_metrics(inp)

    if metrics["max_chain_depth"] < 2:
        return None
    if metrics["activated_count"] < 2:
        return None
    if np.array_equal(out, old_out):
        return None
    if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
        return None
    if not np.any(out == HIT_COLOR):
        return None
    if np.array_equal(inp, out):
        return None
    if metrics["chain_shooter_count"] < 2:
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
