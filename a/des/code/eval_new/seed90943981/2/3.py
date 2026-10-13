from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIR_ORDER = ("up", "right", "down", "left")
DIR_VECS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
CW_DIR = {DIR_ORDER[i]: DIR_ORDER[(i + 1) % 4] for i in range(4)}

ARROW_FROM_HEAD = {
    "up": ((0, 0), (1, -1), (1, 1)),
    "down": ((0, 0), (-1, -1), (-1, 1)),
    "left": ((0, 0), (-1, 1), (1, 1)),
    "right": ((0, 0), (-1, -1), (1, -1)),
}
ARROW_BY_NORM = {
    frozenset({(0, 1), (1, 0), (1, 2)}): ("up", (0, 1)),
    frozenset({(0, 0), (0, 2), (1, 1)}): ("down", (1, 1)),
    frozenset({(0, 1), (1, 0), (2, 1)}): ("left", (1, 0)),
    frozenset({(0, 0), (1, 1), (2, 0)}): ("right", (1, 1)),
}
PRISM_NORMS = {
    frozenset({(0, 0), (0, 1), (1, 0)}),
    frozenset({(0, 0), (0, 1), (1, 1)}),
    frozenset({(0, 0), (1, 0), (1, 1)}),
    frozenset({(0, 1), (1, 0), (1, 1)}),
}


def _all_prism_relative_shapes() -> list[frozenset[tuple[int, int]]]:
    seen: set[frozenset[tuple[int, int]]] = set()
    rels: list[frozenset[tuple[int, int]]] = []
    for base in PRISM_NORMS:
        for orow, ocol in base:
            rel = frozenset((r - orow, c - ocol) for r, c in base)
            if rel not in seen:
                seen.add(rel)
                rels.append(rel)
    return rels


ALL_PRISM_RELS = _all_prism_relative_shapes()

# --------------- solver ---------------


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(int(v) for v in grid.ravel()).most_common(1)[0][0])


def _components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = rr + dr
                        nc = cc + dc
                        if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                            seen[nr, nc] = True
                            stack.append((int(nr), int(nc)))
            comps.append(coords)
    return comps


def _classify_component(coords: list[tuple[int, int]]) -> tuple[str, str | None, tuple[int, int] | None]:
    assert len(coords) == 3, "every foreground object must contain 3 pixels"
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    norm = frozenset((r - r0, c - c0) for r, c in coords)
    if norm in ARROW_BY_NORM:
        direction, head_rel = ARROW_BY_NORM[norm]
        head = (int(r0 + head_rel[0]), int(c0 + head_rel[1]))
        return "arrow", direction, head
    assert norm in PRISM_NORMS, "expected a V-arrow or an L-prism"
    return "prism", None, None


def _parse_objects(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    bg = _get_bg(grid)
    arrows: list[dict] = []
    prisms: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for comp in _components8(grid == color):
            kind, direction, head = _classify_component(comp)
            coords = {(int(r), int(c)) for r, c in comp}
            if kind == "arrow":
                arrows.append({"color": int(color), "dir": direction, "head": head, "coords": coords})
            else:
                prisms.append({"color": int(color), "coords": coords})
    assert arrows, "expected at least one arrow"
    return bg, arrows, prisms


def _line_cells(origin: tuple[int, int], direction: str, shape: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = DIR_VECS[direction]
    h, w = map(int, shape)
    r, c = map(int, origin)
    out: list[tuple[int, int]] = []
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return out
        out.append((int(r), int(c)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shoot rays from V-shaped arrows, splitting them at L-prisms into a straight beam plus a clockwise side beam.

    Input structure:
    - The dominant color is the background.
    - Every non-background object is a monochrome 3-pixel component.
    - Arrow objects are 3-pixel V shapes whose geometry reveals one cardinal head direction.
    - Prism objects are 3-pixel L shapes; their colors differ from the arrow colors.

    Rule:
    - Keep all arrows and prisms in place.
    - From each arrow head, extend a 1-cell-wide ray in the arrow's direction until the grid edge.
    - When a primary ray first reaches a prism, it still continues straight, and an extra ray of the prism's color starts from that hit point in the clockwise-rotated direction.
    - Extra rays extend to the edge but do not split again if they later cross prisms.
    - Rays never overwrite existing non-background pixels.

    Key insight:
    The 3-pixel V encodes direction, while each L-prism contributes a new perpendicular beam whose color comes from the prism and whose turn is always clockwise.
    """
    g = np.asarray(grid, dtype=int)
    bg, arrows, prisms = _parse_objects(g)

    prism_at: dict[tuple[int, int], int] = {}
    prism_color: dict[int, int] = {}
    for pid, prism in enumerate(prisms):
        prism_color[int(pid)] = int(prism["color"])
        for cell in prism["coords"]:
            prism_at[cell] = int(pid)

    out = g.copy()

    def trace(origin: tuple[int, int], direction: str, color: int, can_split: bool) -> None:
        seen_prisms: set[int] = set()
        for cell in _line_cells(origin, direction, g.shape):
            pid = prism_at.get(cell)
            if can_split and pid is not None and pid not in seen_prisms:
                seen_prisms.add(pid)
                trace(cell, CW_DIR[direction], prism_color[pid], False)
            rr, cc = cell
            if int(out[rr, cc]) == int(bg):
                out[rr, cc] = int(color)

    for arrow in sorted(arrows, key=lambda item: (item["head"][0], item["head"][1], item["color"])):
        trace(arrow["head"], str(arrow["dir"]), int(arrow["color"]), True)

    return out


# --------------- generator ---------------

GRID_SIDE_MIN = 15
GRID_SIDE_MAX = 28
GEN_RETRIES = 300
HEAD_CANDIDATE_LIMIT = 140
COMBO_CANDIDATE_LIMIT = 80

SCENARIOS = ("clear", "split1", "split2", "branch_hit")
PRIMARY_HIT_COUNT = {"clear": 0, "split1": 1, "split2": 2, "branch_hit": 1}
MIN_PRIMARY_LEN = {"clear": 3, "split1": 5, "split2": 8, "branch_hit": 6}
MIN_BRANCH_LEN = {"clear": 0, "split1": 3, "split2": 3, "branch_hit": 6}
SCENARIO_ORDER = {"clear": 0, "split1": 1, "branch_hit": 2, "split2": 3}


def _translate(shape: tuple[tuple[int, int], ...] | frozenset[tuple[int, int]], origin: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = origin
    return {(int(r0 + dr), int(c0 + dc)) for dr, dc in shape}


def _in_bounds(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _dilate(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((int(rr), int(cc)))
    return out


def _groups_separated(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r1, c1 in a:
        for r2, c2 in b:
            if max(abs(int(r1) - int(r2)), abs(int(c1) - int(c2))) <= 1:
                return False
    return True


def _prism_rel_options(ray_dir: str, clear_dir: str | None) -> list[frozenset[tuple[int, int]]]:
    along = DIR_VECS[ray_dir]
    back = (-along[0], -along[1])
    clear_vec = DIR_VECS[clear_dir] if clear_dir is not None else None
    preferred: list[frozenset[tuple[int, int]]] = []
    fallback: list[frozenset[tuple[int, int]]] = []
    for rel in ALL_PRISM_RELS:
        if along in rel or back in rel:
            continue
        if clear_vec is not None and clear_vec not in rel:
            preferred.append(rel)
        else:
            fallback.append(rel)
    return preferred + fallback


def _index_combos(line_len: int, count: int, start_idx: int = 2, gap: int = 3) -> list[tuple[int, ...]]:
    if count == 0:
        return [()]
    results: list[tuple[int, ...]] = []

    def rec(next_idx: int, need: int, acc: list[int]) -> None:
        if need == 0:
            results.append(tuple(acc))
            return
        max_idx = line_len - 2 - (need - 1) * gap
        for idx in range(next_idx, max_idx + 1):
            acc.append(int(idx))
            rec(idx + gap, need - 1, acc)
            acc.pop()

    rec(start_idx, count, [])
    return results


def _candidate_heads(h: int, w: int, direction: str, scenario: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    branch_dir = CW_DIR[direction]
    rel = ARROW_FROM_HEAD[direction]
    for r in range(h):
        for c in range(w):
            head = (int(r), int(c))
            cells = _translate(rel, head)
            if not _in_bounds(cells, h, w):
                continue
            if len(_line_cells(head, direction, (h, w))) < MIN_PRIMARY_LEN[scenario]:
                continue
            if len(_line_cells(head, branch_dir, (h, w))) < MIN_BRANCH_LEN[scenario]:
                continue
            out.append(head)
    random.shuffle(out)
    if len(out) > HEAD_CANDIDATE_LIMIT:
        out = random.sample(out, HEAD_CANDIDATE_LIMIT)
    return out


def _object_cells_ok(
    cells: set[tuple[int, int]],
    occupied_margin: set[tuple[int, int]],
    reserved_paint: set[tuple[int, int]],
) -> bool:
    return cells.isdisjoint(occupied_margin) and cells.isdisjoint(reserved_paint)


def _reserve_trajectory(
    path: list[tuple[int, int]],
    allowed_object_cells: set[tuple[int, int]],
    occupied_cells: set[tuple[int, int]],
    new_object_cells: set[tuple[int, int]],
    reserved_paint: set[tuple[int, int]],
    new_reserved: set[tuple[int, int]],
) -> bool:
    for cell in path:
        if cell in occupied_cells:
            return False
        if cell in new_object_cells:
            if cell not in allowed_object_cells:
                return False
            continue
        if cell in reserved_paint or cell in new_reserved:
            return False
        new_reserved.add(cell)
    return True


def _commit_groups(
    grid: np.ndarray,
    groups: list[set[tuple[int, int]]],
    color_pool: list[int],
    occupied_cells: set[tuple[int, int]],
    occupied_margin: set[tuple[int, int]],
    reserved_paint: set[tuple[int, int]],
    new_reserved: set[tuple[int, int]],
) -> None:
    colors = random.sample(color_pool, len(groups))
    for cells, color in zip(groups, colors):
        for r, c in cells:
            grid[r, c] = int(color)
        color_pool.remove(int(color))
    all_new = set().union(*groups)
    occupied_cells |= all_new
    occupied_margin |= _dilate(all_new, *grid.shape)
    reserved_paint |= new_reserved


def _try_place_scenario(
    grid: np.ndarray,
    scenario: str,
    color_pool: list[int],
    occupied_cells: set[tuple[int, int]],
    occupied_margin: set[tuple[int, int]],
    reserved_paint: set[tuple[int, int]],
) -> bool:
    h, w = map(int, grid.shape)
    primary_count = PRIMARY_HIT_COUNT[scenario]

    dirs = list(DIR_ORDER)
    random.shuffle(dirs)
    for direction in dirs:
        branch_dir = CW_DIR[direction]
        for head in _candidate_heads(h, w, direction, scenario):
            arrow_cells = _translate(ARROW_FROM_HEAD[direction], head)
            if not _object_cells_ok(arrow_cells, occupied_margin, reserved_paint):
                continue

            primary_line = _line_cells(head, direction, (h, w))
            combos = _index_combos(len(primary_line), primary_count)
            random.shuffle(combos)
            if len(combos) > COMBO_CANDIDATE_LIMIT:
                combos = random.sample(combos, COMBO_CANDIDATE_LIMIT)

            for combo in combos:
                primary_hits = [primary_line[idx] for idx in combo]
                prism_groups: list[set[tuple[int, int]]] = []
                placed_groups = [arrow_cells]
                options = _prism_rel_options(direction, branch_dir)

                ok = True
                for hit in primary_hits:
                    local_options = options[:]
                    random.shuffle(local_options)
                    chosen = None
                    for rel in local_options:
                        cells = _translate(rel, hit)
                        if not _in_bounds(cells, h, w):
                            continue
                        if not _object_cells_ok(cells, occupied_margin, reserved_paint):
                            continue
                        if all(_groups_separated(cells, other) for other in placed_groups):
                            chosen = cells
                            break
                    if chosen is None:
                        ok = False
                        break
                    prism_groups.append(chosen)
                    placed_groups.append(chosen)
                if not ok:
                    continue

                extra_groups: list[set[tuple[int, int]]] = []
                extra_hit: tuple[int, int] | None = None
                if scenario == "branch_hit":
                    branch_line = _line_cells(primary_hits[0], branch_dir, (h, w))
                    branch_combos = _index_combos(len(branch_line), 1)
                    random.shuffle(branch_combos)
                    if len(branch_combos) > COMBO_CANDIDATE_LIMIT:
                        branch_combos = random.sample(branch_combos, COMBO_CANDIDATE_LIMIT)

                    chosen = None
                    for branch_combo in branch_combos:
                        hit = branch_line[branch_combo[0]]
                        local_options = _prism_rel_options(branch_dir, CW_DIR[branch_dir])
                        random.shuffle(local_options)
                        for rel in local_options:
                            cells = _translate(rel, hit)
                            if not _in_bounds(cells, h, w):
                                continue
                            if not _object_cells_ok(cells, occupied_margin, reserved_paint):
                                continue
                            if all(_groups_separated(cells, other) for other in placed_groups):
                                chosen = cells
                                extra_hit = hit
                                break
                        if chosen is not None:
                            break
                    if chosen is None:
                        continue
                    extra_groups.append(chosen)
                    placed_groups.append(chosen)

                groups = [arrow_cells] + prism_groups + extra_groups
                if len(groups) > len(color_pool):
                    continue

                new_object_cells = set().union(*groups)
                new_reserved: set[tuple[int, int]] = set()
                if not _reserve_trajectory(
                    primary_line,
                    set(primary_hits),
                    occupied_cells,
                    new_object_cells,
                    reserved_paint,
                    new_reserved,
                ):
                    continue

                branch_ok = True
                for idx, hit in enumerate(primary_hits):
                    allowed = {extra_hit} if scenario == "branch_hit" and idx == 0 and extra_hit is not None else set()
                    if not _reserve_trajectory(
                        _line_cells(hit, branch_dir, (h, w)),
                        allowed,
                        occupied_cells,
                        new_object_cells,
                        reserved_paint,
                        new_reserved,
                    ):
                        branch_ok = False
                        break
                if not branch_ok:
                    continue

                _commit_groups(
                    grid,
                    groups,
                    color_pool,
                    occupied_cells,
                    occupied_margin,
                    reserved_paint,
                    new_reserved,
                )
                return True

    return False


def _try_place_decoy_prism(
    grid: np.ndarray,
    color_pool: list[int],
    occupied_cells: set[tuple[int, int]],
    occupied_margin: set[tuple[int, int]],
    reserved_paint: set[tuple[int, int]],
) -> bool:
    if not color_pool:
        return False

    h, w = map(int, grid.shape)
    shapes = list(PRISM_NORMS)
    random.shuffle(shapes)
    for shape in shapes:
        rows = [r for r, _ in shape]
        cols = [c for _, c in shape]
        sh = max(rows) + 1
        sw = max(cols) + 1
        positions = [(r, c) for r in range(h - sh + 1) for c in range(w - sw + 1)]
        random.shuffle(positions)
        for r0, c0 in positions[:HEAD_CANDIDATE_LIMIT]:
            cells = {(int(r0 + r), int(c0 + c)) for r, c in shape}
            if not _object_cells_ok(cells, occupied_margin, reserved_paint):
                continue
            if not cells.isdisjoint(occupied_cells):
                continue
            color = random.choice(color_pool)
            for r, c in cells:
                grid[r, c] = int(color)
            color_pool.remove(int(color))
            occupied_cells |= cells
            occupied_margin |= _dilate(cells, h, w)
            return True
    return False


def _scenario_types(n_arrows: int) -> list[str]:
    types = [random.choice(SCENARIOS) for _ in range(n_arrows)]
    if all(t == "clear" for t in types):
        types[random.randrange(n_arrows)] = random.choice(SCENARIOS[1:])
    types.sort(key=lambda item: SCENARIO_ORDER[item], reverse=True)
    return types


def _valid_generated(grid: np.ndarray, expected_arrows: int) -> bool:
    try:
        bg, arrows, prisms = _parse_objects(grid)
        out = main(grid)
    except Exception:
        return False
    if len(arrows) != expected_arrows:
        return False
    if len(prisms) == 0:
        return False
    if np.array_equal(out, grid):
        return False
    counts = np.bincount(grid.ravel(), minlength=10)
    if int(np.argmax(counts)) != bg:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_arrows = random.choice((1, 2, 2, 3))
    min_side = GRID_SIDE_MIN + 2 * (n_arrows - 1)
    h = random.randint(min_side, GRID_SIDE_MAX)
    w = random.randint(min_side, GRID_SIDE_MAX)
    bg = random.randrange(10)

    grid = np.full((h, w), int(bg), dtype=int)
    color_pool = [color for color in range(10) if color != bg]
    occupied_cells: set[tuple[int, int]] = set()
    occupied_margin: set[tuple[int, int]] = set()
    reserved_paint: set[tuple[int, int]] = set()

    for scenario in _scenario_types(n_arrows):
        if not _try_place_scenario(
            grid,
            scenario,
            color_pool,
            occupied_cells,
            occupied_margin,
            reserved_paint,
        ):
            return None

    if color_pool and random.random() < 0.5:
        _try_place_decoy_prism(grid, color_pool, occupied_cells, occupied_margin, reserved_paint)

    return grid if _valid_generated(grid, n_arrows) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid prism-ray puzzle input")
