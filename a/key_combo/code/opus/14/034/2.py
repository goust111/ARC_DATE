import random
from collections import Counter

import numpy as np


BACKGROUND = 0
GREY = 5
GEN_TILE_SIZE = 8
GEN_MAX_ATTEMPTS = 500
GEN_PALETTE = (1, 2, 3, 4)
GEN_TILE_OFFSETS = ((0, 0), (0, 1), (1, 0), (1, 1))


def _rot90_local(r: int, c: int, side: int) -> tuple[int, int]:
    return int(c), int(side - 1 - r)


def _rotation_orbit_local(r: int, c: int, side: int) -> tuple[tuple[int, int], ...]:
    orbit = []
    seen = set()
    cur = (int(r), int(c))
    for _ in range(4):
        if cur not in seen:
            seen.add(cur)
            orbit.append(cur)
        cur = _rot90_local(cur[0], cur[1], side)
    return tuple(orbit)


def _all_local_orbits(side: int) -> list[tuple[tuple[int, int], ...]]:
    orbits = []
    seen = set()
    for r in range(side):
        for c in range(side):
            orbit = tuple(sorted(_rotation_orbit_local(r, c, side)))
            if orbit in seen:
                continue
            seen.add(orbit)
            orbits.append(orbit)
    return orbits


def _tile_size(grid: np.ndarray) -> int:
    h, w = map(int, grid.shape)
    assert h == w and h % 2 == 0, "expected a square 2x2 tiling"
    side = h // 2
    assert w == 2 * side and side >= 2, "expected four equal square tiles"
    return side


def _translation_positions(r: int, c: int, side: int) -> list[tuple[int, int]]:
    lr, lc = int(r) % side, int(c) % side
    return [
        (tr * side + lr, tc * side + lc)
        for tr in range(2)
        for tc in range(2)
    ]


def _rotation_positions(r: int, c: int, side: int) -> list[tuple[int, int]]:
    base_r = (int(r) // side) * side
    base_c = (int(c) // side) * side
    orbit = _rotation_orbit_local(int(r) % side, int(c) % side, side)
    return [(base_r + rr, base_c + cc) for rr, cc in orbit]


def _visible_colors(grid: np.ndarray, coords: list[tuple[int, int]]) -> set[int]:
    return {
        int(grid[r, c])
        for r, c in coords
        if int(grid[r, c]) != GREY
    }


def _translation_phase(grid: np.ndarray, side: int) -> tuple[np.ndarray, int]:
    out = np.array(grid, dtype=int, copy=True)
    changed = 0
    for r, c in np.argwhere(grid == GREY):
        colors = _visible_colors(grid, _translation_positions(int(r), int(c), side))
        assert len(colors) <= 1, f"translation conflict at {(int(r), int(c))}: {colors}"
        if colors:
            out[int(r), int(c)] = next(iter(colors))
            changed += 1
    return out, changed


def _rotation_phase(grid: np.ndarray, side: int) -> tuple[np.ndarray, int]:
    out = np.array(grid, dtype=int, copy=True)
    changed = 0
    for r, c in np.argwhere(grid == GREY):
        colors = _visible_colors(grid, _rotation_positions(int(r), int(c), side))
        assert len(colors) <= 1, f"rotation conflict at {(int(r), int(c))}: {colors}"
        if colors:
            out[int(r), int(c)] = next(iter(colors))
            changed += 1
    return out, changed


def _solve_iteratively(grid: np.ndarray, side: int, max_rounds: int = 6) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    cur = np.array(grid, dtype=int, copy=True)
    trace = []
    for _ in range(max_rounds):
        after_t, trans_changed = _translation_phase(cur, side)
        after_r, rot_changed = _rotation_phase(after_t, side)
        trace.append((trans_changed, rot_changed, int(np.count_nonzero(after_r == GREY))))
        if np.array_equal(after_r, cur):
            cur = after_r
            break
        cur = after_r
        if not np.any(cur == GREY):
            break
    return cur, trace


def _fixed_point_single_constraint(grid: np.ndarray, side: int, phase: str) -> np.ndarray:
    cur = np.array(grid, dtype=int, copy=True)
    for _ in range(6):
        nxt, changed = (_translation_phase if phase == "T" else _rotation_phase)(cur, side)
        cur = nxt
        if changed == 0:
            break
    return cur


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover grey occlusions by jointly enforcing tile translation periodicity and within-tile quarter-turn symmetry.

    Input structure:
    - The grid is a 2x2 assembly of four equal square tiles; the four tiles are identical copies of one base tile.
    - Inside each tile, the foreground pattern is invariant under 90-degree rotation about the tile center.
    - Some cells are replaced by grey (5), which marks missing data rather than a true color.

    Rule:
    - Use the 2x2 translational orbit of each grey cell across the four tiles; if any counterpart is non-grey, copy that unique color.
    - Then use the 90/180/270-degree rotation orbit of each still-grey cell inside its own tile; if any orbit mate is non-grey, copy that unique color.
    - Alternate these two repairs until no grey cells remain, keeping every non-grey input cell unchanged.

    Key insight:
    - Some missing cells are revealed only by tile copies, some only by rotational orbit mates, and the hardest ones unlock only after a translation repair creates a new rotational clue inside the same tile.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    side = _tile_size(g)
    solved, _trace = _solve_iteratively(g, side)
    assert not np.any(solved == GREY), "joint symmetry repair left unresolved grey cells"
    return solved


def _local_orbit_layers(side: int) -> list[dict]:
    center = (side - 1) / 2.0
    out = []
    for orbit in _all_local_orbits(side):
        maxd = max(max(abs(r - center), abs(c - center)) for r, c in orbit)
        on_edge = any(r in (0, side - 1) or c in (0, side - 1) for r, c in orbit)
        out.append({"orbit": orbit, "maxd": maxd, "on_edge": on_edge})
    return out


def _build_symmetric_tile(side: int) -> tuple[np.ndarray, list[tuple[tuple[int, int], ...]]]:
    orbit_infos = _local_orbit_layers(side)
    edge_ids = [i for i, info in enumerate(orbit_infos) if info["on_edge"]]
    inner_ids = [i for i, info in enumerate(orbit_infos) if not info["on_edge"]]

    for _ in range(200):
        num_colors = random.choice((3, 4))
        palette = random.sample(list(GEN_PALETTE), num_colors)
        num_fg_orbits = random.choice((7, 8, 9))

        chosen = set()
        chosen.add(random.choice(edge_ids))
        chosen.add(random.choice(inner_ids))
        remaining_ids = [i for i in range(len(orbit_infos)) if i not in chosen]
        chosen.update(random.sample(remaining_ids, num_fg_orbits - len(chosen)))
        chosen = list(chosen)
        random.shuffle(chosen)

        assignments = {}
        for orbit_id, color in zip(chosen, palette):
            assignments[orbit_id] = color
        for orbit_id in chosen[len(palette) :]:
            assignments[orbit_id] = random.choice(palette)

        tile = np.full((side, side), BACKGROUND, dtype=int)
        for orbit_id, color in assignments.items():
            for r, c in orbit_infos[orbit_id]["orbit"]:
                tile[r, c] = int(color)

        if not np.array_equal(tile, np.rot90(tile)):
            continue
        if any(np.array_equal(tile, np.roll(tile, shift=shift, axis=(0, 1))) for shift in ((0, 2), (2, 0), (0, 4), (4, 0), (4, 4))):
            continue
        return tile, [orbit_infos[i]["orbit"] for i in chosen]

    raise RuntimeError("failed to build a non-degenerate symmetric tile")


def _global_coord(tile_index: int, local: tuple[int, int], side: int) -> tuple[int, int]:
    tr, tc = GEN_TILE_OFFSETS[tile_index]
    return int(tr * side + local[0]), int(tc * side + local[1])


def _apply_chain_occlusion(mask: set[tuple[int, int]], orbit: tuple[tuple[int, int], ...], side: int) -> None:
    bridge = random.choice(list(orbit))
    others = [cell for cell in orbit if cell != bridge]
    fully_hidden_tiles = set(random.sample(range(4), 2))
    for tile_index in range(4):
        if tile_index in fully_hidden_tiles:
            for local in orbit:
                mask.add(_global_coord(tile_index, local, side))
        else:
            for local in others:
                mask.add(_global_coord(tile_index, local, side))


def _apply_a_occlusion(mask: set[tuple[int, int]], orbit: tuple[tuple[int, int], ...], side: int) -> None:
    local = random.choice(list(orbit))
    grey_tiles = random.sample(range(4), random.choice((2, 3)))
    for tile_index in grey_tiles:
        mask.add(_global_coord(tile_index, local, side))


def _apply_b_occlusion(mask: set[tuple[int, int]], orbit: tuple[tuple[int, int], ...], side: int) -> None:
    local = random.choice(list(orbit))
    for tile_index in range(4):
        mask.add(_global_coord(tile_index, local, side))


def _classify_missing(grid: np.ndarray, side: int) -> dict[tuple[int, int], str]:
    classes = {}
    for r, c in map(tuple, np.argwhere(grid == GREY)):
        trans_colors = _visible_colors(grid, _translation_positions(r, c, side))
        rot_positions = _rotation_positions(r, c, side)
        rot_colors = _visible_colors(grid, rot_positions)
        if trans_colors:
            classes[(r, c)] = "A"
            continue
        if rot_colors:
            classes[(r, c)] = "B"
            continue
        unlocked = False
        for rr, cc in rot_positions:
            if (rr, cc) == (r, c) or int(grid[rr, cc]) != GREY:
                continue
            mate_trans = _visible_colors(grid, _translation_positions(rr, cc, side))
            if mate_trans:
                unlocked = True
                break
        classes[(r, c)] = "C" if unlocked else "?"
    return classes


def _validate_attempt(inp: np.ndarray, out: np.ndarray, side: int) -> bool:
    if inp.shape != (2 * side, 2 * side):
        return False
    if not np.array_equal(main(inp), out):
        return False

    grey_coords = np.argwhere(inp == GREY)
    if not (15 <= len(grey_coords) <= 25):
        return False

    classes = _classify_missing(inp, side)
    counts = Counter(classes.values())
    if counts.get("?", 0) != 0:
        return False
    if min(counts.get(label, 0) for label in ("A", "B", "C")) <= 0:
        return False

    grey_colors = {int(out[r, c]) for r, c in map(tuple, grey_coords)}
    if len(grey_colors) < 2:
        return False

    t_only = _fixed_point_single_constraint(inp, side, "T")
    r_only = _fixed_point_single_constraint(inp, side, "R")
    if np.array_equal(t_only, out) or np.array_equal(r_only, out):
        return False

    after_t, t_changed = _translation_phase(inp, side)
    if t_changed <= 0 or not np.any(after_t == GREY):
        return False
    remaining_after_t = list(map(tuple, np.argwhere(after_t == GREY)))
    if not remaining_after_t:
        return False
    if not all(_visible_colors(after_t, _rotation_positions(r, c, side)) for r, c in remaining_after_t):
        return False

    after_tr, r_changed = _rotation_phase(after_t, side)
    if r_changed <= 0 or np.any(after_tr == GREY):
        return False

    _solved, trace = _solve_iteratively(inp, side)
    if len(trace) > 2:
        return False
    if trace[0][0] <= 0 or trace[0][1] <= 0:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    side = GEN_TILE_SIZE
    tile, colored_orbits = _build_symmetric_tile(side)
    if len(colored_orbits) < 4:
        return None

    out = np.tile(tile, (2, 2))
    available = list(colored_orbits)
    random.shuffle(available)

    chain_orbit = available.pop()
    mask: set[tuple[int, int]] = set()
    _apply_chain_occlusion(mask, chain_orbit, side)

    extra_a_specs = random.choice((1, 2))
    for _ in range(extra_a_specs):
        if not available:
            return None
        _apply_a_occlusion(mask, available.pop(), side)

    if available and random.random() < 0.65:
        _apply_b_occlusion(mask, available.pop(), side)

    inp = np.array(out, copy=True)
    for r, c in mask:
        inp[r, c] = GREY

    if not _validate_attempt(inp, out, side):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid joint-symmetry repair input")
