import random
from functools import lru_cache
from time import perf_counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

FOREGROUND_COLOR = YELLOW

# (name, relative cells, output color)
TILE_PATTERNS = (
    ("square", ((0, 0), (0, 1), (1, 0), (1, 1)), BLUE),
    ("missing_ul", ((0, 1), (1, 0), (1, 1)), RED),
    ("missing_ll", ((0, 0), (0, 1), (1, 1)), GREEN),
    ("missing_ur", ((0, 0), (1, 0), (1, 1)), YELLOW),
)

GEN_MAX_ATTEMPTS = 300
GEN_HEIGHT_RANGE = (11, 14)
GEN_WIDTH_RANGE = (12, 15)
GEN_TILE_COUNT_RANGE = (13, 15)
GEN_NOISE_COUNT_RANGE = (3, 6)
GEN_CLEAN_LOCAL_ATTEMPTS = 40
GEN_NOISE_LOCAL_ATTEMPTS = 30
GEN_MAX_FINAL_SOLVE_SECONDS = 0.75


def _neighbors4(shape: tuple[int, int], r: int, c: int):
    h, w = shape
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(mask.shape, rr, cc):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _count_fg_neighbors(mask: np.ndarray, cell: tuple[int, int]) -> int:
    r, c = cell
    return sum(int(mask[nr, nc]) for nr, nc in _neighbors4(mask.shape, r, c))


def _tile_cells(tile_type: int, r0: int, c0: int) -> tuple[tuple[int, int], ...]:
    return tuple((r0 + dr, c0 + dc) for dr, dc in TILE_PATTERNS[tile_type][1])


def _touches4(cells: tuple[tuple[int, int], ...], occupied: set[tuple[int, int]]) -> bool:
    if not occupied:
        return True
    for r, c in cells:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in occupied:
                return True
    return False


def _contact_score(cells: tuple[tuple[int, int], ...], occupied: set[tuple[int, int]]) -> int:
    score = 0
    for r, c in cells:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            score += int((nr, nc) in occupied)
    return score


def _render_input(shape: tuple[int, int], placements: list[tuple[int, tuple[tuple[int, int], ...]]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for _tile_type, cells in placements:
        for r, c in cells:
            out[r, c] = FOREGROUND_COLOR
    return out


def _render_output(shape: tuple[int, int], placements: list[tuple[int, tuple[tuple[int, int], ...]]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for tile_type, cells in placements:
        color = TILE_PATTERNS[tile_type][2]
        for r, c in cells:
            out[r, c] = color
    return out


def _analyze_mask(mask: np.ndarray) -> dict:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not coords:
        raise AssertionError("expected non-empty foreground mask")

    cell_to_idx = {cell: idx for idx, cell in enumerate(coords)}
    full_state = (1 << len(coords)) - 1

    tile_masks: list[int] = []
    tile_sizes: list[int] = []
    tile_infos: list[dict] = []
    seen_masks: set[int] = set()
    h, w = mask.shape

    for r0 in range(h - 1):
        for c0 in range(w - 1):
            for tile_type, (_name, rels, color) in enumerate(TILE_PATTERNS):
                bits = 0
                cells: list[tuple[int, int]] = []
                ok = True
                for dr, dc in rels:
                    cell = (r0 + dr, c0 + dc)
                    if not mask[cell]:
                        ok = False
                        break
                    bits |= 1 << cell_to_idx[cell]
                    cells.append(cell)
                if not ok or bits in seen_masks:
                    continue
                seen_masks.add(bits)
                tile_masks.append(bits)
                tile_sizes.append(len(cells))
                tile_infos.append(
                    {
                        "tile_type": tile_type,
                        "color": color,
                        "cells": tuple(cells),
                        "mask": bits,
                    }
                )

    by_cell: list[list[int]] = [[] for _ in coords]
    for idx, bits in enumerate(tile_masks):
        cur = bits
        while cur:
            lsb = cur & -cur
            by_cell[lsb.bit_length() - 1].append(idx)
            cur ^= lsb

    @lru_cache(maxsize=None)
    def solve(state: int) -> tuple[int, int, int, int]:
        if state == 0:
            return 0, 1, -1, -1

        pivot_idx = -1
        pivot_bit = 0
        options: list[int] = []
        min_option_count = None
        cur = state
        while cur:
            lsb = cur & -cur
            cell_idx = lsb.bit_length() - 1
            avail = [idx for idx in by_cell[cell_idx] if (tile_masks[idx] & state) == tile_masks[idx]]
            count = len(avail)
            if min_option_count is None or count < min_option_count:
                min_option_count = count
                pivot_idx = cell_idx
                pivot_bit = lsb
                options = avail
                if count == 0:
                    break
            cur ^= lsb

        if min_option_count == 0:
            best_score, best_count, _kind, _value = solve(state ^ pivot_bit)
            return best_score, best_count, 0, pivot_idx

        best_score = -1
        best_count = 0
        best_kind = -1
        best_value = -1

        options.sort(key=lambda idx: (-tile_sizes[idx], tile_infos[idx]["cells"]))
        for tile_idx in options:
            sub_score, sub_count, _kind, _value = solve(state ^ tile_masks[tile_idx])
            score = sub_score + tile_sizes[tile_idx]
            if score > best_score:
                best_score = score
                best_count = sub_count
                best_kind = 1
                best_value = tile_idx
            elif score == best_score:
                best_count = min(2, best_count + sub_count)

        sub_score, sub_count, _kind, _value = solve(state ^ pivot_bit)
        if sub_score > best_score:
            best_score = sub_score
            best_count = sub_count
            best_kind = 0
            best_value = pivot_idx
        elif sub_score == best_score:
            best_count = min(2, best_count + sub_count)

        return best_score, best_count, best_kind, best_value

    chosen_tiles: list[dict] = []
    noise_cells: list[tuple[int, int]] = []
    state = full_state
    while state:
        _score, _count, kind, value = solve(state)
        if kind == 1:
            chosen_tiles.append(tile_infos[value])
            state ^= tile_masks[value]
        elif kind == 0:
            noise_cells.append(coords[value])
            state ^= 1 << value
        else:
            raise AssertionError("invalid reconstruction state")

    covered = np.zeros_like(mask, dtype=bool)
    for tile in chosen_tiles:
        for r, c in tile["cells"]:
            covered[r, c] = True
    out = np.zeros(mask.shape, dtype=int)
    for tile in chosen_tiles:
        for r, c in tile["cells"]:
            out[r, c] = tile["color"]

    return {
        "best_score": solve(full_state)[0],
        "optimal_count": solve(full_state)[1],
        "tiles": chosen_tiles,
        "noise_cells": tuple(noise_cells),
        "covered_mask": covered,
        "output": out,
        "all_candidate_tiles": tuple(tile_infos),
        "foreground_area": len(coords),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove foreground noise by keeping only the unique largest tile-coverable subset, then recolor each kept tile by shape.

    Input structure:
    - The grid is a black background with exactly one non-black foreground color forming a roughly connected mask.
    - The foreground mask contains a clean tileable core plus a few extra same-color noise pixels.
    - The clean core can be covered exactly by translated 2x2 squares and three 2x2 L-tromino orientations.

    Rule:
    - Consider only tile placements whose cells are already foreground.
    - Among all disjoint tile selections, keep the unique optimal one that covers the maximum number of foreground cells.
    - Erase every uncovered foreground pixel as noise, and recolor each kept tile by shape: square -> blue, missing upper-left -> red, missing lower-left -> green, missing upper-right -> yellow.

    Key insight:
    - Noise cannot be identified reliably from local support alone; the correct foreground subset is the one admitted by the unique global exact cover with maximum covered area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colors = sorted(int(x) for x in np.unique(g) if int(x) != BLACK)
    assert len(colors) == 1, "expected exactly one foreground color"

    analysis = _analyze_mask(g != BLACK)
    assert analysis["optimal_count"] == 1, "expected a unique optimal exact cover"
    return analysis["output"]


def _sample_tile_types() -> list[int]:
    tile_count = random.randint(*GEN_TILE_COUNT_RANGE)
    tile_types = [0, 1, 2, 3]
    weights = [0.08, 0.31, 0.31, 0.30]
    while len(tile_types) < tile_count:
        tile_types.append(random.choices([0, 1, 2, 3], weights=weights, k=1)[0])
    random.shuffle(tile_types)
    return tile_types


def _place_clean_tiles(height: int, width: int, tile_types: list[int]) -> list[tuple[int, tuple[tuple[int, int], ...]]] | None:
    anchors = [(r, c) for r in range(height - 1) for c in range(width - 1)]
    placements: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    occupied: set[tuple[int, int]] = set()

    for index, tile_type in enumerate(tile_types):
        candidates: list[tuple[int, float, tuple[tuple[int, int], ...]]] = []
        for r0, c0 in anchors:
            cells = _tile_cells(tile_type, r0, c0)
            if any(cell in occupied for cell in cells):
                continue
            if index > 0 and not _touches4(cells, occupied):
                continue
            contacts = _contact_score(cells, occupied)
            candidates.append((contacts, random.random(), cells))

        if not candidates:
            return None

        candidates.sort()
        pool = candidates
        if index == 0:
            center_r = (height - 2) / 2.0
            center_c = (width - 2) / 2.0
            pool = sorted(
                candidates,
                key=lambda item: (
                    abs(sum(r for r, _c in item[2]) / len(item[2]) - center_r)
                    + abs(sum(c for _r, c in item[2]) / len(item[2]) - center_c),
                    item[1],
                ),
            )[: max(6, len(candidates) // 3)]
        else:
            min_contacts = min(item[0] for item in candidates)
            pool = [item for item in candidates if item[0] <= min_contacts + 1]
            if len(pool) > 18:
                pool = pool[:18]

        _score, _rand, cells = random.choice(pool)
        placements.append((tile_type, cells))
        occupied.update(cells)

    return placements


def _boundary_noise_pool(mask: np.ndarray) -> list[tuple[int, int]]:
    pool: list[tuple[int, int]] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            if any(mask[nr, nc] for nr, nc in _neighbors4(mask.shape, r, c)):
                pool.append((r, c))
    random.shuffle(pool)
    return pool


def _local_tile_count_with_added(mask: np.ndarray, cell: tuple[int, int]) -> int:
    r, c = cell
    h, w = mask.shape
    total = 0
    seen: set[frozenset[tuple[int, int]]] = set()
    for r0 in range(max(0, r - 1), min(h - 2, r) + 1):
        for c0 in range(max(0, c - 1), min(w - 2, c) + 1):
            for _name, rels, _color in TILE_PATTERNS:
                cells = frozenset((r0 + dr, c0 + dc) for dr, dc in rels)
                if cell not in cells:
                    continue
                if all(mask[rr, cc] or (rr, cc) == cell for rr, cc in cells):
                    seen.add(cells)
    total = len(seen)
    return total


def _noise_semantics(
    noisy_mask: np.ndarray,
    clean_mask: np.ndarray,
    analysis: dict,
) -> tuple[int, int]:
    noise_cells = [cell for cell in map(tuple, np.argwhere(noisy_mask & ~clean_mask))]
    supported = 0
    tricky = 0
    candidate_tiles = analysis["all_candidate_tiles"]

    for cell in noise_cells:
        if _count_fg_neighbors(noisy_mask, cell) >= 2:
            supported += 1
        if any(cell in tile["cells"] and sum(int(clean_mask[rc]) for rc in tile["cells"]) >= 2 for tile in candidate_tiles):
            tricky += 1
    return supported, tricky


def _analysis_matches_clean(
    noisy_mask: np.ndarray,
    clean_mask: np.ndarray,
    expected_output: np.ndarray,
) -> tuple[bool, dict | None]:
    try:
        analysis = _analyze_mask(noisy_mask)
    except AssertionError:
        return False, None
    if analysis["optimal_count"] != 1:
        return False, None
    if analysis["best_score"] != int(clean_mask.sum()):
        return False, None
    if not np.array_equal(analysis["covered_mask"], clean_mask):
        return False, None
    if not np.array_equal(analysis["output"], expected_output):
        return False, None
    return True, analysis


def _add_noise(
    clean_mask: np.ndarray,
    expected_output: np.ndarray,
) -> np.ndarray | None:
    for _ in range(GEN_NOISE_LOCAL_ATTEMPTS):
        target_noise = random.randint(*GEN_NOISE_COUNT_RANGE)
        pool = _boundary_noise_pool(clean_mask)
        noisy_mask = clean_mask.copy()
        added: list[tuple[int, int]] = []
        used_tricky_seed = False

        while len(added) < target_noise:
            remaining = [cell for cell in pool if not noisy_mask[cell]]
            if not remaining:
                break

            scored = []
            for cell in remaining:
                local_count = _local_tile_count_with_added(noisy_mask, cell)
                scored.append((cell, _count_fg_neighbors(clean_mask, cell), local_count))

            if not used_tricky_seed:
                tricky_seed = [cell for cell, support, local_count in scored if support >= 2 and 1 <= local_count <= 2]
                if not tricky_seed:
                    break
                choice_pool = tricky_seed
            else:
                easy = [cell for cell, _support, local_count in scored if local_count == 0]
                mild = [cell for cell, _support, local_count in scored if local_count <= 1]
                choice_pool = easy or mild or []
                if not choice_pool:
                    break
            random.shuffle(choice_pool)

            accepted = False
            for cell in choice_pool:
                trial = noisy_mask.copy()
                trial[cell] = True
                ok, _analysis = _analysis_matches_clean(trial, clean_mask, expected_output)
                if ok:
                    noisy_mask = trial
                    added.append(cell)
                    used_tricky_seed = True
                    accepted = True
                    break
            if not accepted:
                break

        if len(added) < target_noise:
            continue

        ok, analysis = _analysis_matches_clean(noisy_mask, clean_mask, expected_output)
        if not ok or analysis is None:
            continue

        supported, tricky = _noise_semantics(noisy_mask, clean_mask, analysis)
        if supported >= 1 and tricky >= 1:
            return noisy_mask
    return None


def _generate_clean_instance() -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(GEN_CLEAN_LOCAL_ATTEMPTS):
        height = random.randint(*GEN_HEIGHT_RANGE)
        width = random.randint(*GEN_WIDTH_RANGE)
        tile_types = _sample_tile_types()
        placements = _place_clean_tiles(height, width, tile_types)
        if placements is None:
            continue

        clean_input = _render_input((height, width), placements)
        clean_mask = clean_input == FOREGROUND_COLOR
        if len(_components4(clean_mask)) != 1:
            continue

        area = int(clean_mask.sum())
        if not (40 <= area <= 60):
            continue

        bbox = np.argwhere(clean_mask)
        h_span = int(bbox[:, 0].max() - bbox[:, 0].min() + 1)
        w_span = int(bbox[:, 1].max() - bbox[:, 1].min() + 1)
        if h_span < 6 or w_span < 6:
            continue

        expected_output = _render_output((height, width), placements)
        ok, analysis = _analysis_matches_clean(clean_mask, clean_mask, expected_output)
        if not ok or analysis is None:
            continue
        if len({tile["tile_type"] for tile in analysis["tiles"]}) < 4:
            continue
        return clean_input, expected_output
    return None


def _gen_attempt() -> np.ndarray | None:
    clean = _generate_clean_instance()
    if clean is None:
        return None
    clean_input, expected_output = clean
    clean_mask = clean_input == FOREGROUND_COLOR

    noisy_mask = _add_noise(clean_mask, expected_output)
    if noisy_mask is None:
        return None

    inp = np.where(noisy_mask, FOREGROUND_COLOR, BLACK).astype(int)
    try:
        start = perf_counter()
        out = main(inp)
        elapsed = perf_counter() - start
    except AssertionError:
        return None
    if elapsed > GEN_MAX_FINAL_SOLVE_SECONDS:
        return None
    if not np.array_equal(out, expected_output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid noisy exact-cover tiling input")
