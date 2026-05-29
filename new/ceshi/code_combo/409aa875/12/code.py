# Combo variant of 409aa875 (index 12)
# combo_spec: D4/2+D8/1
# Name: 命中推移+密集精灵

# Design analysis
# - Concrete behavior and trace:
#   The base task still reads every 3-cell triangle as one shooter, with the landing cell at apex + 5*dir.
#   The D4 change replaces the old "recolor hit sprite in place" rule with "recolor the directly hit sprite to
#   the shooter's color and move it 1 cell along the shot direction". The dense D8 change then makes that motion
#   spatially consequential: if the moved sprite's destination cells are already occupied by another sprite, that
#   blocker is pushed 1 cell in the same direction, which can push another blocker, producing a domino chain.
#   Small trace: three adjacent triangles A,B,C sit in a tight row of repeated copies. A separate shooter S fires
#   into A. Original task: A would just become maroon at its old location. D4-only: A would recolor to S's color
#   and shift into B. Fusion rule: A recolors and shifts into B's old cells, B shifts into C's old cells, and C
#   shifts into the next free cells; only A changes color, while B and C merely move.
# - Complexity calibration:
#   A dense 12-16 sprite canvas needs compactness without turning into unreadable blobs. The generator therefore
#   uses a small repeat palette that excludes 1, 7, 9, with the push lane cycling through three colors and the
#   deliberate shooter taking a fourth contrast color when needed. Different colors may touch, so the packed layout
#   is visible, while same-color sprites are kept 8-disconnected so each triangle remains an identifiable monochrome
#   component. A single deliberate hit chain per grid keeps the push rule legible instead of mixing multiple
#   interacting domino directions.
# - Hidden assumptions broken from the old main():
#   1. The old solver assumed sprite hits are local recolors. That no longer holds; moved sprites must be cleared
#      at their old cells and repainted at translated cells.
#   2. The old solver only needed hit-component ids. The fused rule needs the shooter's direction and color, plus
#      recursive blocker discovery, so the hit table stores structured shooter records.
#   3. The old dense-separation assumption is invalid. To allow packed domino lanes, the solver extracts 8-connected
#      components separately for each color, so different-color sprites may touch without merging.
#   4. The old code never had to reason about secondary motion. The new push resolver assumes each generated hit
#      chain is direction-consistent and acyclic; the generator enforces exactly one planned hit chain and reserves
#      its final cells away from stationary sprites and bg markers.
# - Placement legality:
#   The generator is still output-first in spirit: it first chooses a push chain whose translated tail stays
#   in-bounds, then reserves every final moved cell and every bg impact cell before adding fillers. Later sprites
#   may touch other colors for density, but they may neither overlap existing cells nor create same-color 8-links.
# - Visual observability:
#   Every sample has 12-16 sprites, exactly one direct hit, and a domino chain of length at least 3. That guarantees
#   the fused behavior is not reducible to the original task: multiple sprites move, and the front sprite adopts the
#   shooter's color. Because the dense lane reuses a small recurring palette, the recolor cue is visible across
#   several objects instead of looking like a one-off palette accident.
# - Visual cleanliness:
#   The generator forbids bg markers from appearing under moved sprites, keeps same-color components separate in both
#   input and output, and packs sprites by maximizing local contact rather than random scattering. The result is dense
#   but still decomposable by eye into small triangles and one obvious push lane.
# - New solver assumption review:
#   The recursive push resolver assumes that a direct-hit target chooses one shooter and one direction. The generator
#   guarantees a single direct hit event, no competing hits on the same target, and no filler sprite whose own shot
#   accidentally hits another sprite. The bg-marker loop also assumes that reserved bg destinations remain empty in
#   the final state; generation reserves them away from all translated chain cells.

import random
from collections import Counter, defaultdict
from itertools import permutations

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8


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
        dr = -dr_sum
        dc = -dc_sum
        dr = 0 if dr == 0 else (1 if dr > 0 else -1)
        dc = 0 if dc == 0 else (1 if dc > 0 else -1)
        return corner, (dr, dc)

    apexes = [p for p, d in deg8.items() if d == 2]
    if len(apexes) != 1:
        return None
    apex = apexes[0]
    ar, ac = apex
    others = [p for p in comp if p != apex]
    d1 = (others[0][0] - ar, others[0][1] - ac)
    d2 = (others[1][0] - ar, others[1][1] - ac)
    dr_sum = d1[0] + d2[0]
    dc_sum = d1[1] + d2[1]
    dr = -(dr_sum // 2)
    dc = -(dc_sum // 2)
    if abs(dr) + abs(dc) != 1:
        return None
    return apex, (dr, dc)


def _shift_cells(
    cells: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    dr: int,
    dc: int,
) -> tuple[tuple[int, int], ...]:
    return tuple((r + dr, c + dc) for r, c in cells)


def _cells_in_bounds(shape: tuple[int, int], cells: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _halo(cells: tuple[tuple[int, int], ...] | list[tuple[int, int]] | set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    return frozenset(
        (rr, cc)
        for r, c in cells
        for rr in range(max(0, r - 1), min(H, r + 2))
        for cc in range(max(0, c - 1), min(W, c + 2))
    )


def _extract_sprites(grid: np.ndarray, bg: int) -> tuple[list[dict], dict[tuple[int, int], int]]:
    sprites: list[dict] = []
    cell_to_sprite: dict[tuple[int, int], int] = {}
    for color in sorted(set(int(v) for v in np.unique(grid)) - {bg}):
        for comp in _cc8(grid == color):
            sid = len(sprites)
            cells = tuple(comp)
            sprites.append({"cells": cells, "color": color})
            for rc in cells:
                cell_to_sprite[rc] = sid
    return sprites, cell_to_sprite


def _projectiles(
    sprites: list[dict],
    shape: tuple[int, int],
) -> dict[tuple[int, int], list[dict]]:
    h, w = shape
    dest_to_shooters: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for sid, sprite in enumerate(sprites):
        ad = _triangle_apex_and_dir(list(sprite["cells"]))
        if ad is None:
            continue
        (ar, ac), (dr, dc) = ad
        dest = (ar + 5 * dr, ac + 5 * dc)
        if 0 <= dest[0] < h and 0 <= dest[1] < w:
            dest_to_shooters[dest].append({"sid": sid, "dir": (dr, dc), "color": sprite["color"]})
    return dest_to_shooters


def _collect_push_chain(
    sid: int,
    direction: tuple[int, int],
    sprites: list[dict],
    cell_to_sprite: dict[tuple[int, int], int],
    shape: tuple[int, int],
    seen: set[int],
) -> set[int] | None:
    if sid in seen:
        return None
    shifted = _shift_cells(sprites[sid]["cells"], *direction)
    if not _cells_in_bounds(shape, shifted):
        return None
    blockers = sorted(
        {
            cell_to_sprite[rc]
            for rc in shifted
            if rc in cell_to_sprite and cell_to_sprite[rc] != sid
        }
    )
    members = {sid}
    for blocker in blockers:
        sub = _collect_push_chain(blocker, direction, sprites, cell_to_sprite, shape, seen | {sid})
        if sub is None:
            return None
        members |= sub
    return members


def _resolve_pushes(
    sprites: list[dict],
    cell_to_sprite: dict[tuple[int, int], int],
    shape: tuple[int, int],
    dest_to_shooters: dict[tuple[int, int], list[dict]],
) -> tuple[dict[int, int], dict[int, tuple[int, int]], dict[int, tuple[tuple[int, int], ...]], dict[int, int]]:
    direct_hits: dict[int, dict] = {}
    for dest, shooters in dest_to_shooters.items():
        target_sid = cell_to_sprite.get(dest)
        if target_sid is None or target_sid in direct_hits:
            continue
        direct_hits[target_sid] = sorted(shooters, key=lambda item: item["sid"])[0]

    recolors: dict[int, int] = {}
    move_dirs: dict[int, tuple[int, int]] = {}
    final_cells_by_sid: dict[int, tuple[tuple[int, int], ...]] = {}
    chain_lengths: dict[int, int] = {}

    for target_sid, shooter in sorted(direct_hits.items()):
        if target_sid in move_dirs:
            continue
        members = _collect_push_chain(target_sid, shooter["dir"], sprites, cell_to_sprite, shape, set())
        if members is None:
            continue
        if any((sid in move_dirs and move_dirs[sid] != shooter["dir"]) for sid in members):
            continue
        if any(sid in recolors for sid in members if sid != target_sid):
            continue
        for sid in members:
            move_dirs[sid] = shooter["dir"]
            final_cells_by_sid[sid] = _shift_cells(sprites[sid]["cells"], *shooter["dir"])
        recolors[target_sid] = shooter["color"]
        chain_lengths[target_sid] = len(members)

    return recolors, move_dirs, final_cells_by_sid, chain_lengths


def _baseline_main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    sprites, cell_to_sprite = _extract_sprites(g, bg)
    dest_to_shooters = _projectiles(sprites, g.shape)

    out = g.copy()
    hit_sprites = {cell_to_sprite[dest] for dest in dest_to_shooters if dest in cell_to_sprite}
    for sid in hit_sprites:
        for r, c in sprites[sid]["cells"]:
            out[r, c] = 9
    for dest, shooters in dest_to_shooters.items():
        if dest in cell_to_sprite:
            continue
        out[dest] = 1 if len(shooters) >= 2 else 9
    return out


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    sprites, cell_to_sprite = _extract_sprites(g, bg)
    dest_to_shooters = _projectiles(sprites, g.shape)
    recolors, move_dirs, final_cells_by_sid, chain_lengths = _resolve_pushes(sprites, cell_to_sprite, g.shape, dest_to_shooters)

    contact_pairs = 0
    for i, sprite_i in enumerate(sprites):
        halo_i = _halo(sprite_i["cells"])
        cells_i = set(sprite_i["cells"])
        for j in range(i + 1, len(sprites)):
            if sprites[j]["cells"][0] in cells_i:
                continue
            if set(sprites[j]["cells"]) & halo_i:
                contact_pairs += 1

    bg_multi = 0
    for dest, shooters in dest_to_shooters.items():
        if dest not in cell_to_sprite and len(shooters) >= 2:
            bg_multi += 1

    return {
        "sprite_count": len(sprites),
        "direct_hits": len(recolors),
        "moved_sprites": len(move_dirs),
        "max_chain": max(chain_lengths.values(), default=0),
        "bg_multi": bg_multi,
        "contact_pairs": contact_pairs,
        "final_occupied": sum(len(cells) for cells in final_cells_by_sid.values()),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow every triangle shot; empty impacts still mark the board, but a sprite hit now starts a one-step push chain.

    Input structure:
    - The majority color is background 7.
    - Every foreground object is a monochrome 3-cell triangle, either an axis-aligned L or a diagonal V.
    - Different colors may touch, so sprites must be recognized as 8-connected components within each color separately.

    Rule:
    - Each triangle's geometry determines its apex and firing direction.
    - The projectile lands exactly 5 cells from that apex.
    - If the landing cell is empty background, paint that cell maroon (9), or blue (1) when multiple projectiles share it.
    - If the landing cell belongs to a sprite, recolor that directly hit sprite to the shooter's color and move it 1 cell
      in the shooting direction. If that translated position is occupied by another sprite, push the blocker 1 cell in the
      same direction too, repeating as needed like dominoes. Secondary pushed sprites keep their own colors.

    Key insight: The triangle still encodes the projectile destination, but the important new effect is not just "what gets
    hit" — it is the whole translation chain caused by that hit. The front sprite adopts the shooter's color, while every
    blocker in the same lane shifts forward one cell.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    sprites, cell_to_sprite = _extract_sprites(g, bg)
    dest_to_shooters = _projectiles(sprites, g.shape)
    recolors, move_dirs, final_cells_by_sid, _ = _resolve_pushes(sprites, cell_to_sprite, g.shape, dest_to_shooters)

    out = g.copy()

    for sid in move_dirs:
        for r, c in sprites[sid]["cells"]:
            out[r, c] = bg

    moved_cells: set[tuple[int, int]] = set()
    for sid, final_cells in final_cells_by_sid.items():
        color = recolors.get(sid, sprites[sid]["color"])
        for r, c in final_cells:
            out[r, c] = color
            moved_cells.add((r, c))

    for dest, shooters in dest_to_shooters.items():
        if dest in cell_to_sprite or dest in moved_cells:
            continue
        out[dest] = 1 if len(shooters) >= 2 else 9

    return out


H = W = 16
BG = 7
HIT_MARK_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [2, 3, 4, 5, 6, 8]

PROJECTILE_DIST = 5
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

SPRITE_COUNT_RANGE = (12, 16)
CHAIN_LENGTHS = (3, 4)
MAX_ATTEMPTS = 1200
_GEN_CALL_COUNT = 0
_COLORING_PLANS_BY_LEN: dict[int, list[tuple[list[int], int]]] = {}


def _sprite_cells(apex: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    ar, ac = apex
    dr, dc = direction
    if abs(dr) + abs(dc) == 2:
        return ((ar, ac), (ar - dr, ac), (ar, ac - dc))
    if dr != 0:
        return ((ar, ac), (ar - dr, ac - 1), (ar - dr, ac + 1))
    return ((ar, ac), (ar - 1, ac - dc), (ar + 1, ac - dc))


def _in_bounds_cells(cells: tuple[tuple[int, int], ...]) -> bool:
    return _cells_in_bounds((H, W), cells)


def _shift_placement(placement: dict, dr: int, dc: int) -> dict:
    cells = _shift_cells(placement["cells"], dr, dc)
    dest = (placement["dest"][0] + dr, placement["dest"][1] + dc)
    return {
        "apex": (placement["apex"][0] + dr, placement["apex"][1] + dc),
        "dir": placement["dir"],
        "dest": dest,
        "cells": cells,
        "cell_set": frozenset(cells),
        "halo": _halo(cells),
    }


def _build_placements() -> tuple[list[dict], dict[tuple[tuple[int, int], tuple[int, int]], list[dict]]]:
    placements: list[dict] = []
    by_dest_dir: dict[tuple[tuple[int, int], tuple[int, int]], list[dict]] = defaultdict(list)
    for ar in range(H):
        for ac in range(W):
            for dr, dc in ALL_DIRS:
                dest = (ar + PROJECTILE_DIST * dr, ac + PROJECTILE_DIST * dc)
                if not (0 <= dest[0] < H and 0 <= dest[1] < W):
                    continue
                cells = _sprite_cells((ar, ac), (dr, dc))
                if not _in_bounds_cells(cells):
                    continue
                placement = {
                    "apex": (ar, ac),
                    "dir": (dr, dc),
                    "dest": dest,
                    "cells": cells,
                    "cell_set": frozenset(cells),
                    "halo": _halo(cells),
                }
                placements.append(placement)
                by_dest_dir[(dest, (dr, dc))].append(placement)
    return placements, by_dest_dir


PLACEMENTS, PLACEMENTS_BY_DEST_DIR = _build_placements()


def _same_color_ok(layout: list[tuple[int, tuple[tuple[int, int], ...]]]) -> bool:
    halos: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for color, cells in layout:
        if set(cells) & halos[color]:
            return False
        halos[color].update(_halo(cells))
    return True


def _paint_cells(grid: np.ndarray, cells: tuple[tuple[int, int], ...], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _dir_family(direction: tuple[int, int]) -> str:
    return "diag" if abs(direction[0]) + abs(direction[1]) == 2 else "axis"


def _bundle_colorings(rng: random.Random, chain_len: int) -> list[tuple[list[int], int]]:
    if chain_len not in _COLORING_PLANS_BY_LEN:
        plans: list[tuple[list[int], int]] = []
        for shooter_color, color_a, color_b, color_c in permutations(SPRITE_COLORS, 4):
            cycle = [color_a, color_b, color_c]
            for offset in range(3):
                chain_colors = [cycle[(offset + i) % 3] for i in range(chain_len)]
                plans.append((chain_colors, shooter_color))
        _COLORING_PLANS_BY_LEN[chain_len] = plans
    plans = _COLORING_PLANS_BY_LEN[chain_len]
    start = rng.randrange(len(plans))
    return plans[start:] + plans[:start]


def _choose_chain_bundle(
    rng: random.Random,
    preferred_family: str | None = None,
    preferred_chain_len: int | None = None,
) -> dict | None:
    dirs = [d for d in ALL_DIRS if preferred_family is None or _dir_family(d) == preferred_family]
    rng.shuffle(dirs)
    for direction in dirs:
        dir_placements = [p for p in PLACEMENTS if p["dir"] == direction]
        rng.shuffle(dir_placements)
        lengths = list(CHAIN_LENGTHS)
        rng.shuffle(lengths)
        if preferred_chain_len in lengths:
            lengths.remove(preferred_chain_len)
            lengths.insert(0, preferred_chain_len)
        for chain_len in lengths:
            for base in dir_placements:
                chain = [_shift_placement(base, i * direction[0], i * direction[1]) for i in range(chain_len)]
                chain_cells = [placement["cells"] for placement in chain]
                all_chain_cells = set().union(*(set(cells) for cells in chain_cells))
                if len(all_chain_cells) != 3 * chain_len:
                    continue
                tail_cells = _shift_cells(chain[-1]["cells"], *direction)
                if not _in_bounds_cells(tail_cells):
                    continue
                final_chain_cells = set(tail_cells)
                for placement in chain[1:]:
                    final_chain_cells.update(placement["cells"])
                if len(final_chain_cells) != 3 * chain_len:
                    continue

                bg_dests = [placement["dest"] for placement in chain]
                if len(set(bg_dests)) != chain_len:
                    continue
                if any(dest in all_chain_cells or dest in final_chain_cells for dest in bg_dests):
                    continue

                head_cells = list(chain[0]["cells"])
                rng.shuffle(head_cells)
                for hit_cell in head_cells:
                    shooter_candidates = PLACEMENTS_BY_DEST_DIR[(hit_cell, direction)][:]
                    rng.shuffle(shooter_candidates)
                    for shooter in shooter_candidates:
                        if shooter["cell_set"] & all_chain_cells:
                            continue
                        if shooter["cell_set"] & final_chain_cells:
                            continue
                        if any(dest in shooter["cell_set"] for dest in bg_dests):
                            continue
                        for chain_colors, shooter_color in _bundle_colorings(rng, chain_len):
                            input_layout = [
                                (chain_colors[i], chain[i]["cells"])
                                for i in range(chain_len)
                            ] + [(shooter_color, shooter["cells"])]
                            if not _same_color_ok(input_layout):
                                continue
                            output_layout = [(shooter_color, _shift_cells(chain[0]["cells"], *direction))]
                            output_layout.extend(
                                (chain_colors[i], _shift_cells(chain[i]["cells"], *direction))
                                for i in range(1, chain_len)
                            )
                            output_layout.append((shooter_color, shooter["cells"]))
                            if not _same_color_ok(output_layout):
                                continue
                            return {
                                "chain": chain,
                                "chain_colors": chain_colors,
                                "shooter": shooter,
                                "shooter_color": shooter_color,
                                "direction": direction,
                            }
    return None


def _bbox_area(cells: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> int:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)


def _fits_filler(
    placement: dict,
    color: int,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    final_reserved: set[tuple[int, int]],
    input_color_halo: dict[int, set[tuple[int, int]]],
    final_color_halo: dict[int, set[tuple[int, int]]],
) -> bool:
    if placement["cell_set"] & occupied:
        return False
    if placement["cell_set"] & reserved_bg:
        return False
    if placement["cell_set"] & final_reserved:
        return False
    if placement["dest"] in occupied or placement["dest"] in reserved_bg or placement["dest"] in final_reserved:
        return False
    if placement["cell_set"] & input_color_halo[color]:
        return False
    if placement["cell_set"] & final_color_halo[color]:
        return False
    return True


def _choose_dense_filler(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    final_reserved: set[tuple[int, int]],
    input_color_halo: dict[int, set[tuple[int, int]]],
    final_color_halo: dict[int, set[tuple[int, int]]],
    color_counts: Counter,
) -> tuple[dict, int] | None:
    candidates = []
    for placement in PLACEMENTS:
        colors = [
            color
            for color in SPRITE_COLORS
            if _fits_filler(
                placement,
                color,
                occupied,
                reserved_bg,
                final_reserved,
                input_color_halo,
                final_color_halo,
            )
        ]
        if not colors:
            continue
        chosen_color = min(colors, key=lambda color: (color_counts[color], color))
        contact = len((placement["halo"] - placement["cell_set"]) & occupied)
        area = _bbox_area(occupied | set(placement["cells"])) if occupied else _bbox_area(set(placement["cells"]))
        score = (contact, -area, -abs(placement["apex"][0] - 7.5) - abs(placement["apex"][1] - 7.5))
        candidates.append((score, placement, chosen_color))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    top = candidates[: min(24, len(candidates))]
    return rng.choice(top)[1:]


def _validate_input(grid: np.ndarray, n_sprites: int) -> bool:
    if grid.shape != (H, W):
        return False
    if _bg_color(grid) != BG:
        return False
    sprites, _ = _extract_sprites(grid, BG)
    if len(sprites) != n_sprites:
        return False
    for sprite in sprites:
        if len(sprite["cells"]) != 3:
            return False
        if _triangle_apex_and_dir(list(sprite["cells"])) is None:
            return False
    return True


def _gen_attempt(preferred_family: str | None = None, preferred_chain_len: int | None = None) -> np.ndarray | None:
    rng = random.Random()
    bundle = _choose_chain_bundle(rng, preferred_family=preferred_family, preferred_chain_len=preferred_chain_len)
    if bundle is None:
        return None

    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)
    chain = bundle["chain"]
    chain_colors = bundle["chain_colors"]
    shooter = bundle["shooter"]
    shooter_color = bundle["shooter_color"]
    direction = bundle["direction"]

    sprites: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    reserved_bg: set[tuple[int, int]] = set()
    final_reserved: set[tuple[int, int]] = set()
    input_color_halo: dict[int, set[tuple[int, int]]] = defaultdict(set)
    final_color_halo: dict[int, set[tuple[int, int]]] = defaultdict(set)
    color_counts: Counter = Counter()

    for idx, placement in enumerate(chain):
        color = chain_colors[idx]
        sprites.append({"placement": placement, "color": color, "target_kind": "bg"})
        occupied.update(placement["cells"])
        reserved_bg.add(placement["dest"])
        input_color_halo[color].update(placement["halo"])
        color_counts[color] += 1

    head_final = _shift_cells(chain[0]["cells"], *direction)
    final_reserved.update(head_final)
    final_color_halo[shooter_color].update(_halo(head_final))
    for idx in range(1, len(chain)):
        moved_cells = _shift_cells(chain[idx]["cells"], *direction)
        final_reserved.update(moved_cells)
        final_color_halo[chain_colors[idx]].update(_halo(moved_cells))

    sprites.append(
        {
            "placement": shooter,
            "color": shooter_color,
            "target_kind": "hit",
            "target_sprite": 0,
        }
    )
    occupied.update(shooter["cells"])
    input_color_halo[shooter_color].update(shooter["halo"])
    color_counts[shooter_color] += 1

    remaining = n_sprites - len(sprites)
    if remaining < 0:
        return None
    for _ in range(remaining):
        choice = _choose_dense_filler(
            rng,
            occupied,
            reserved_bg,
            final_reserved,
            input_color_halo,
            final_color_halo,
            color_counts,
        )
        if choice is None:
            return None
        placement, color = choice
        sprites.append({"placement": placement, "color": color, "target_kind": "bg"})
        occupied.update(placement["cells"])
        reserved_bg.add(placement["dest"])
        input_color_halo[color].update(placement["halo"])
        color_counts[color] += 1

    inp = np.full((H, W), BG, dtype=int)
    for sprite in sprites:
        _paint_cells(inp, sprite["placement"]["cells"], sprite["color"])

    if not _validate_input(inp, n_sprites):
        return None

    out = main(inp)
    baseline = _baseline_main(inp)
    metrics = _behavior_metrics(inp)

    if np.array_equal(inp, out):
        return None
    if np.array_equal(out, baseline):
        return None
    if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
        return None
    if metrics["sprite_count"] != n_sprites:
        return None
    if metrics["direct_hits"] != 1:
        return None
    if metrics["moved_sprites"] != len(chain):
        return None
    if metrics["max_chain"] < len(chain):
        return None
    if metrics["contact_pairs"] < max(6, n_sprites - 4):
        return None
    if not np.any(out == HIT_MARK_COLOR):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    preferred_family = "diag" if _GEN_CALL_COUNT % 2 == 0 else "axis"
    preferred_chain_len = CHAIN_LENGTHS[(_GEN_CALL_COUNT // 2) % len(CHAIN_LENGTHS)]
    _GEN_CALL_COUNT += 1

    other_family = "axis" if preferred_family == "diag" else "diag"
    attempt_plan = [
        (preferred_family, preferred_chain_len),
        (preferred_family, None),
        (other_family, preferred_chain_len),
        (other_family, None),
        (None, None),
    ]

    for attempt_idx in range(MAX_ATTEMPTS):
        family, chain_len = attempt_plan[attempt_idx % len(attempt_plan)]
        grid = _gen_attempt(preferred_family=family, preferred_chain_len=chain_len)
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_ATTEMPTS} attempts")
