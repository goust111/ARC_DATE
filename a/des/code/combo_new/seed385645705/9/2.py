from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


GREY = 5
MAROON = 9
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_TO_NAME = {(-1, 0): "up", (1, 0): "down", (0, -1): "left", (0, 1): "right"}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        nbrs = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    else:
        raise ValueError("connectivity must be 4 or 8")

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in nbrs:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _reflect(direction: tuple[int, int], orientation: str) -> tuple[int, int]:
    dr, dc = direction
    if dr == 0:
        return (1, 0) if orientation == "\\" else (-1, 0)
    return (0, 1) if orientation == "\\" else (0, -1)


def _turn_orientation(direction: tuple[int, int], next_direction: tuple[int, int]) -> str:
    dr, dc = direction
    ndr, ndc = next_direction
    if dr == 0:
        if ndr == 1 and ndc == 0:
            return "\\"
        if ndr == -1 and ndc == 0:
            return "/"
    else:
        if ndr == 0 and ndc == 1:
            return "\\"
        if ndr == 0 and ndc == -1:
            return "/"
    raise ValueError("invalid 90-degree turn")


def _mirror_map(grid: np.ndarray) -> dict[tuple[int, int], str]:
    mirrors = grid == MAROON
    orientations: dict[tuple[int, int], str] = {}
    for comp in _components(mirrors, connectivity=8):
        if len(comp) < 2:
            raise AssertionError("mirror component must have length at least two")
        diffs = {r - c for r, c in comp}
        sums = {r + c for r, c in comp}
        if len(diffs) == 1 and len(sums) > 1:
            orient = "\\"
        elif len(sums) == 1 and len(diffs) > 1:
            orient = "/"
        else:
            raise AssertionError("maroon component must be a clean diagonal segment")
        for cell in comp:
            orientations[cell] = orient
    return orientations


def _find_towers(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    towers: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (bg, GREY, MAROON):
                continue
            grey_dirs: list[tuple[int, int]] = []
            for dr, dc in DIRS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == GREY:
                    grey_dirs.append((int(dr), int(dc)))
            if len(grey_dirs) != 1:
                continue
            towers.append({"pos": (int(r), int(c)), "color": color, "dir": grey_dirs[0]})
    towers.sort(key=lambda item: item["pos"])
    return towers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace colored signal rays through diagonal maroon mirrors and paint the traversed background.

    Input structure:
    - The most common color is the background.
    - Several single-pixel signal towers use distinct non-background colors; each tower has exactly one adjacent grey pixel whose side encodes the launch direction.
    - Maroon pixels form separate 45-degree diagonal segments of length 2-4, acting as mirrors with either `\\` or `/` orientation.

    Rule:
    - Remove every grey direction marker back to background.
    - For each tower, start a ray in the direction indicated by its grey neighbor, skipping over that grey cell.
    - When the ray hits a maroon mirror pixel, reflect it by 90 degrees: horizontal rays turn down on `\\` and up on `/`, while vertical rays turn right on `\\` and left on `/`.
    - Stop the ray when it leaves the grid or reaches any signal tower, and paint only the original background pixels along the ray with the tower's color.

    Key insight:
    - The task is a multi-step trace: infer each launch direction from grey adjacency, infer each mirror orientation from the diagonal segment geometry, then simulate the reflected ray path cell by cell.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    mirrors = _mirror_map(g)
    towers = _find_towers(g, bg=bg)
    tower_cells = {tuple(item["pos"]): int(item["color"]) for item in towers}

    out = g.copy()
    out[out == GREY] = bg

    for tower in towers:
        color = int(tower["color"])
        r0, c0 = tower["pos"]
        dr, dc = tower["dir"]
        r = int(r0 + 2 * dr)
        c = int(c0 + 2 * dc)
        seen_states: set[tuple[int, int, int, int]] = set()

        while 0 <= r < h and 0 <= c < w:
            state = (int(r), int(c), int(dr), int(dc))
            if state in seen_states:
                break
            seen_states.add(state)

            if (int(r), int(c)) in tower_cells:
                break

            if int(g[r, c]) == MAROON:
                dr, dc = _reflect((int(dr), int(dc)), mirrors[(int(r), int(c))])
                r += dr
                c += dc
                continue

            if int(g[r, c]) == bg:
                out[r, c] = color

            r += dr
            c += dc

    return out


PALETTE = [0, 1, 2, 3, 4, 6, 7, 8]
H_RANGE = (14, 25)
W_RANGE = (14, 25)
TOWER_COUNT_RANGE = (2, 5)
REFLECTION_CHOICES = [0, 1, 2]
MIN_TOWER_GAP = 4
GEN_RETRIES = 400


def _inside(h: int, w: int, cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < h and 0 <= c < w


def _add(cell: tuple[int, int], direction: tuple[int, int], k: int = 1) -> tuple[int, int]:
    return (int(cell[0] + direction[0] * k), int(cell[1] + direction[1] * k))


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _perpendicular_choices(direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    if dr == 0:
        return [(-1, 0), (1, 0)]
    return [(0, -1), (0, 1)]


def _mirror_segment_candidates(
    hit: tuple[int, int],
    orientation: str,
    h: int,
    w: int,
    rng: np.random.Generator,
) -> list[list[tuple[int, int]]]:
    diag = (1, 1) if orientation == "\\" else (1, -1)
    candidates: list[list[tuple[int, int]]] = []
    lengths = [2, 3, 4]
    rng.shuffle(lengths)
    for length in lengths:
        hit_indices = list(range(length))
        rng.shuffle(hit_indices)
        for hit_idx in hit_indices:
            cells = [_add(hit, diag, offset - hit_idx) for offset in range(length)]
            if all(_inside(h, w, cell) for cell in cells):
                candidates.append(cells)
    return candidates


def _mirror_isolated(candidate: Iterable[tuple[int, int]], mirror_cells: set[tuple[int, int]]) -> bool:
    cand = set(candidate)
    for r, c in cand:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in mirror_cells and (rr, cc) not in cand:
                    return False
    return True


def _can_use_bg(
    cell: tuple[int, int],
    occupied_input: set[tuple[int, int]],
    reserved_path: set[tuple[int, int]],
    local_input: set[tuple[int, int]],
    local_path: set[tuple[int, int]],
) -> bool:
    return (
        cell not in occupied_input
        and cell not in reserved_path
        and cell not in local_input
        and cell not in local_path
    )


def _target_candidates(
    start: tuple[int, int],
    direction: tuple[int, int],
    source: tuple[int, int],
    tower_positions: set[tuple[int, int]],
    occupied_input: set[tuple[int, int]],
    reserved_path: set[tuple[int, int]],
    local_input: set[tuple[int, int]],
    local_path: set[tuple[int, int]],
    rng: np.random.Generator,
) -> list[tuple[list[tuple[int, int]], tuple[int, int]]]:
    sr, sc = start
    dr, dc = direction
    candidates: list[tuple[list[tuple[int, int]], tuple[int, int]]] = []

    for tower in tower_positions:
        if tower == source:
            continue
        tr, tc = tower
        if dr != 0:
            if tc != sc or (tr - sr) * dr <= 0:
                continue
            distance = abs(tr - sr)
        else:
            if tr != sr or (tc - sc) * dc <= 0:
                continue
            distance = abs(tc - sc)
        if distance < 2:
            continue

        cells = [_add(start, direction, step) for step in range(distance - 1)]
        if not cells:
            continue
        if all(_can_use_bg(cell, occupied_input, reserved_path, local_input, local_path) for cell in cells):
            candidates.append((cells, tower))

    rng.shuffle(candidates)
    if dr != 0:
        candidates.sort(key=lambda item: abs(item[1][0] - sr))
    else:
        candidates.sort(key=lambda item: abs(item[1][1] - sc))
    return candidates


def _boundary_path(
    start: tuple[int, int],
    direction: tuple[int, int],
    h: int,
    w: int,
    occupied_input: set[tuple[int, int]],
    reserved_path: set[tuple[int, int]],
    local_input: set[tuple[int, int]],
    local_path: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    cells: list[tuple[int, int]] = []
    cur = start
    while _inside(h, w, cur):
        if not _can_use_bg(cur, occupied_input, reserved_path, local_input, local_path):
            return None
        cells.append(cur)
        cur = _add(cur, direction)
    return cells if cells else None


def _place_one_ray(
    source: tuple[int, int],
    reflections: int,
    tower_positions: set[tuple[int, int]],
    occupied_input: set[tuple[int, int]],
    reserved_path: set[tuple[int, int]],
    mirror_cells: set[tuple[int, int]],
    h: int,
    w: int,
    rng: np.random.Generator,
    require_target: bool,
    preferred_orients: list[str] | None = None,
) -> dict[str, object] | None:
    directions = DIRS4[:]
    rng.shuffle(directions)

    for initial_dir in directions:
        grey = _add(source, initial_dir)
        first = _add(source, initial_dir, 2)
        if not _inside(h, w, grey) or not _inside(h, w, first):
            continue
        if grey in occupied_input or grey in reserved_path or first in occupied_input or first in reserved_path:
            continue

        local_input = {grey}
        local_path: set[tuple[int, int]] = set()
        local_mirrors: list[list[tuple[int, int]]] = []
        used_orients: list[str] = []
        current = first
        direction = initial_dir
        ok = True

        for turn_idx in range(reflections):
            next_dirs = _perpendicular_choices(direction)
            rng.shuffle(next_dirs)
            if preferred_orients is not None and turn_idx < len(preferred_orients):
                want = preferred_orients[turn_idx]
                next_dirs.sort(key=lambda d: _turn_orientation(direction, d) != want)

            found_turn = False
            path_prefix: list[tuple[int, int]] = []
            probe = current

            while _inside(h, w, probe) and _can_use_bg(probe, occupied_input, reserved_path, local_input, local_path):
                path_prefix.append(probe)
                hit = _add(probe, direction)
                if not _inside(h, w, hit):
                    break
                if hit in occupied_input or hit in reserved_path or hit in local_input or hit in local_path:
                    break

                for next_dir in next_dirs:
                    orientation = _turn_orientation(direction, next_dir)
                    next_start = _add(hit, next_dir)
                    if not _inside(h, w, next_start):
                        continue
                    if not _can_use_bg(next_start, occupied_input, reserved_path, local_input, local_path):
                        continue

                    for segment in _mirror_segment_candidates(hit, orientation, h, w, rng):
                        if any(
                            cell in occupied_input
                            or cell in reserved_path
                            or cell in local_input
                            or cell in local_path
                            for cell in segment
                        ):
                            continue
                        future_mirrors = mirror_cells | local_input
                        if not _mirror_isolated(segment, future_mirrors):
                            continue

                        local_path.update(path_prefix)
                        for cell in segment:
                            local_input.add(cell)
                        local_mirrors.append(segment)
                        used_orients.append(orientation)
                        current = next_start
                        direction = next_dir
                        found_turn = True
                        break
                    if found_turn:
                        break
                if found_turn:
                    break
                probe = _add(probe, direction)

            if not found_turn:
                ok = False
                break

        if not ok:
            continue

        choices: list[tuple[str, object]] = []
        target_opts = _target_candidates(
            start=current,
            direction=direction,
            source=source,
            tower_positions=tower_positions,
            occupied_input=occupied_input,
            reserved_path=reserved_path,
            local_input=local_input,
            local_path=local_path,
            rng=rng,
        )
        if target_opts:
            choices.append(("tower", target_opts[0]))
        boundary_cells = _boundary_path(
            start=current,
            direction=direction,
            h=h,
            w=w,
            occupied_input=occupied_input,
            reserved_path=reserved_path,
            local_input=local_input,
            local_path=local_path,
        )
        if boundary_cells is not None:
            choices.append(("edge", boundary_cells))

        if require_target:
            choices = [choice for choice in choices if choice[0] == "tower"]
        if not choices:
            continue

        if not require_target and len(choices) == 2 and rng.random() < 0.5:
            choices = choices[::-1]

        mode, payload = choices[0]
        if mode == "tower":
            final_cells, target = payload
            local_path.update(final_cells)
            stop = target
        else:
            local_path.update(payload)
            stop = None

        return {
            "dir": initial_dir,
            "grey": grey,
            "path": set(local_path),
            "mirrors": local_mirrors,
            "used_orients": used_orients,
            "stop": stop,
        }

    return None


def _tower_positions(rng: np.random.Generator, h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    rng.shuffle(interior)
    towers: list[tuple[int, int]] = []
    for cell in interior:
        if all(_chebyshev(cell, other) >= MIN_TOWER_GAP for other in towers):
            towers.append(cell)
        if len(towers) == count:
            return towers
    return None


def _reflection_plan(rng: np.random.Generator, count: int) -> list[int]:
    plan = [REFLECTION_CHOICES[i % len(REFLECTION_CHOICES)] for i in range(count)]
    while len(plan) < count:
        plan.append(int(rng.choice(REFLECTION_CHOICES)))
    rng.shuffle(plan)
    if count >= 2 and 0 not in plan:
        plan[0] = 0
    if count >= 2 and max(plan) == 0:
        plan[1] = 1
    return plan[:count]


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(H_RANGE[0], H_RANGE[1] + 1))
    w = int(rng.integers(W_RANGE[0], W_RANGE[1] + 1))
    tower_count = int(rng.integers(TOWER_COUNT_RANGE[0], TOWER_COUNT_RANGE[1] + 1))
    towers = _tower_positions(rng, h, w, tower_count)
    if towers is None:
        return None

    bg = int(rng.choice(PALETTE))
    tower_palette = [c for c in PALETTE if c != bg]
    rng.shuffle(tower_palette)
    tower_colors = tower_palette[:tower_count]
    reflection_counts = _reflection_plan(rng, tower_count)
    order = list(range(tower_count))
    order.sort(key=lambda idx: (-reflection_counts[idx], towers[idx]))
    occupied_input = set(towers)
    reserved_path: set[tuple[int, int]] = set()
    mirror_cells: set[tuple[int, int]] = set()
    ray_specs: dict[int, dict[str, object]] = {}
    used_tower_stop = False
    need_tower_stop = tower_count >= 3 and bool(rng.integers(0, 2))
    forced_orients = ["\\", "/"] if sum(reflection_counts) >= 2 else []
    orient_cursor = 0

    for idx in order:
        reflections = reflection_counts[idx]
        preferred = None
        if reflections > 0 and orient_cursor < len(forced_orients):
            preferred = [forced_orients[orient_cursor]]
            orient_cursor += 1
        require_target = need_tower_stop and not used_tower_stop

        spec = _place_one_ray(
            source=towers[idx],
            reflections=reflections,
            tower_positions=set(towers),
            occupied_input=occupied_input,
            reserved_path=reserved_path,
            mirror_cells=mirror_cells,
            h=h,
            w=w,
            rng=rng,
            require_target=require_target,
            preferred_orients=preferred,
        )
        if spec is None:
            return None

        occupied_input.add(spec["grey"])
        reserved_path.update(spec["path"])
        for mirror in spec["mirrors"]:
            mirror_cells.update(mirror)
            occupied_input.update(mirror)
        ray_specs[idx] = spec
        if spec["stop"] is not None:
            used_tower_stop = True

    if need_tower_stop and not used_tower_stop:
        return None

    used_orients = {orient for spec in ray_specs.values() for orient in spec["used_orients"]}
    if sum(reflection_counts) >= 2 and used_orients != {"\\", "/"}:
        return None

    grid = np.full((h, w), bg, dtype=int)
    for idx, pos in enumerate(towers):
        r, c = pos
        grid[r, c] = tower_colors[idx]
    for spec in ray_specs.values():
        gr, gc = spec["grey"]
        grid[gr, gc] = GREY
        for mirror in spec["mirrors"]:
            for mr, mc in mirror:
                grid[mr, mc] = MAROON

    tower_detected = _find_towers(grid, bg=bg)
    if len(tower_detected) != tower_count:
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if not np.any((grid == bg) & (out != bg)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after retries")
