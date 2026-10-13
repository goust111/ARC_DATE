import numpy as np


BG_COLOR = 0
SPRITE_COLOR = 2
WALL_COLOR = 5
MARKER_COLOR = 8

DIRS = {
    "U": (-1, 0),
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
}

OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}

GRID_H_RANGE = (12, 19)
GRID_W_RANGE = (12, 19)
STEP_COUNT_RANGE = (3, 7)
STEP_DISTANCE_RANGE = (2, 7)
MAX_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _parse_objects(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = [int(x) for x in np.unique(g) if int(x) != bg]
    assert colors, "expected foreground objects"

    sprite_candidates = []
    marker_candidates = []
    wall_candidates = []
    for color in colors:
        comps = _components(g == color)
        sizes = [len(comp) for comp in comps]
        if len(comps) == 1 and sizes == [4]:
            r0, c0, r1, c1 = _bbox(comps[0])
            if (r1 - r0 + 1, c1 - c0 + 1) == (2, 2):
                sprite_candidates.append((color, r0, c0))
                continue
        if comps and all(size == 1 for size in sizes):
            marker_candidates.append(color)
            continue
        wall_candidates.append(color)

    assert len(sprite_candidates) == 1, "expected exactly one 2x2 sprite"
    assert len(marker_candidates) == 1, "expected exactly one marker color"
    sprite_color, sr, sc = sprite_candidates[0]
    marker_color = marker_candidates[0]
    wall_color = wall_candidates[0] if wall_candidates else WALL_COLOR

    marker_coords = {tuple(map(int, xy)) for xy in np.argwhere(g == marker_color)}
    walls = g == wall_color if wall_candidates else np.zeros_like(g, dtype=bool)

    return {
        "bg": bg,
        "sprite_color": sprite_color,
        "marker_color": marker_color,
        "wall_color": wall_color,
        "sprite_pos": (int(sr), int(sc)),
        "markers": marker_coords,
        "walls": walls,
    }


def _sprite_mask(shape: tuple[int, int], pos: tuple[int, int]) -> np.ndarray:
    r, c = pos
    mask = np.zeros(shape, dtype=bool)
    mask[r : r + 2, c : c + 2] = True
    return mask


def _sweep_mask(shape: tuple[int, int], pos0: tuple[int, int], pos1: tuple[int, int]) -> np.ndarray:
    r0, c0 = pos0
    r1, c1 = pos1
    mask = np.zeros(shape, dtype=bool)
    if r0 == r1:
        lo = min(c0, c1)
        hi = max(c0, c1)
        mask[r0 : r0 + 2, lo : hi + 2] = True
    else:
        lo = min(r0, r1)
        hi = max(r0, r1)
        mask[lo : hi + 2, c0 : c0 + 2] = True
    return mask


def _landing_pos(sprite_pos: tuple[int, int], direction: str, marker: tuple[int, int]) -> tuple[int, int]:
    r, c = sprite_pos
    mr, mc = marker
    if direction == "U":
        return mr, c
    if direction == "D":
        return mr - 1, c
    if direction == "L":
        return r, mc
    return r, mc - 1


def _visible_candidates(
    shape: tuple[int, int],
    sprite_pos: tuple[int, int],
    walls: np.ndarray,
    trail: np.ndarray,
    markers: set[tuple[int, int]],
    use_walls: bool,
    use_trail: bool,
) -> list[dict]:
    h, w = shape
    r, c = sprite_pos
    blockers = np.zeros(shape, dtype=bool)
    if use_walls:
        blockers |= walls
    if use_trail:
        blockers |= trail

    candidates: list[dict] = []

    for cc in (c, c + 1):
        rr = r - 1
        while rr >= 0:
            cell = (rr, cc)
            if blockers[cell]:
                break
            if cell in markers:
                candidates.append({"marker": cell, "direction": "U", "distance": r - rr})
                break
            rr -= 1

    for cc in (c, c + 1):
        rr = r + 2
        while rr < h:
            cell = (rr, cc)
            if blockers[cell]:
                break
            if cell in markers:
                candidates.append({"marker": cell, "direction": "D", "distance": rr - (r + 1)})
                break
            rr += 1

    for rr in (r, r + 1):
        cc = c - 1
        while cc >= 0:
            cell = (rr, cc)
            if blockers[cell]:
                break
            if cell in markers:
                candidates.append({"marker": cell, "direction": "L", "distance": c - cc})
                break
            cc -= 1

    for rr in (r, r + 1):
        cc = c + 2
        while cc < w:
            cell = (rr, cc)
            if blockers[cell]:
                break
            if cell in markers:
                candidates.append({"marker": cell, "direction": "R", "distance": cc - (c + 1)})
                break
            cc += 1

    dedup: dict[tuple[int, int], dict] = {}
    for candidate in candidates:
        dedup[candidate["marker"]] = candidate
    return list(dedup.values())


def _simulate(grid: np.ndarray, *, use_trail: bool = True, use_walls: bool = True) -> dict:
    g = np.asarray(grid, dtype=int)
    info = _parse_objects(g)
    h, w = g.shape
    sprite_pos = info["sprite_pos"]
    walls = info["walls"]
    markers = set(info["markers"])
    trail = np.zeros((h, w), dtype=bool)

    states = [sprite_pos]
    consumed: list[tuple[int, int]] = []
    steps: list[dict] = []

    while True:
        candidates = _visible_candidates((h, w), sprite_pos, walls, trail, markers, use_walls, use_trail)
        if not candidates:
            break
        min_dist = min(item["distance"] for item in candidates)
        best = [item for item in candidates if item["distance"] == min_dist]
        if len(best) != 1:
            return {"valid": False, "reason": "nearest_visible_not_unique"}
        chosen = best[0]
        marker = chosen["marker"]
        next_pos = _landing_pos(sprite_pos, chosen["direction"], marker)
        nr, nc = next_pos
        if not (0 <= nr and nr + 1 < h and 0 <= nc and nc + 1 < w):
            return {"valid": False, "reason": "sprite_leaves_grid"}

        current_mask = _sprite_mask((h, w), sprite_pos)
        final_mask = _sprite_mask((h, w), next_pos)
        sweep = _sweep_mask((h, w), sprite_pos, next_pos)
        path_mask = sweep & ~current_mask

        blocked = np.zeros((h, w), dtype=bool)
        if use_walls:
            blocked |= walls
        if use_trail:
            blocked |= trail
        if np.any(blocked[path_mask]):
            return {"valid": False, "reason": "path_hits_obstacle"}

        for other in markers:
            if other == marker:
                continue
            if path_mask[other]:
                return {"valid": False, "reason": "path_hits_other_marker"}

        if use_trail:
            trail |= sweep
            trail[final_mask] = False

        markers.remove(marker)
        sprite_pos = next_pos
        states.append(sprite_pos)
        consumed.append(marker)
        steps.append(
            {
                "direction": chosen["direction"],
                "distance": int(chosen["distance"]),
                "marker": marker,
                "candidate_count": len(candidates),
                "candidates": tuple(sorted((item["marker"], item["direction"], item["distance"]) for item in candidates)),
            }
        )

    out = np.full_like(g, info["bg"])
    if use_walls:
        out[walls] = info["wall_color"]
    if use_trail:
        out[trail] = info["wall_color"]
    for mr, mc in markers:
        out[mr, mc] = info["marker_color"]
    sr, sc = sprite_pos
    out[sr : sr + 2, sc : sc + 2] = info["sprite_color"]

    return {
        "valid": True,
        "output": out,
        "states": states,
        "steps": steps,
        "consumed": consumed,
        "remaining": markers,
        "trail": trail,
        "walls": walls,
        "sprite_color": info["sprite_color"],
        "marker_color": info["marker_color"],
        "wall_color": info["wall_color"],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the 2x2 sprite toward the nearest visible marker, turning its swept path into blocking walls.

    Input structure:
    - A black background contains one red 2x2 sprite, several cyan single-pixel markers, and a few grey wall segments.
    - Markers are isolated pixels; walls are fixed obstacles; the sprite is the only solid 2x2 block.

    Rule:
    - From the sprite's current 2x2 footprint, look outward along the four cardinal directions using the sprite's two rows or two columns as the sight lines.
    - A marker is visible if it lies on one of those lines and every intervening cell is empty; grey walls, earlier trail, and nearer markers all stop the sight line. Among all visible markers, move to the unique nearest one so the leading edge of the 2x2 sprite lands on it.
    - Every traversed sprite cell except the final resting 2x2 becomes grey trail, and that trail blocks later sight and movement exactly like the original walls.
    - Repeat until no marker is visible. Output the final sprite, all original walls plus the grey trail, and any cyan markers that remain sealed off.

    Key insight:
    - The path is an active obstacle: earlier moves rewrite the grid into new walls, so the nearest visible target now determines which later markers stay reachable and which become permanently trapped.
    """
    result = _simulate(np.asarray(grid, dtype=int), use_trail=True, use_walls=True)
    assert result["valid"], result.get("reason", "invalid grid")
    return result["output"]


def _move_spec(pos: tuple[int, int], direction: str, lane: int, distance: int) -> tuple[tuple[int, int], tuple[int, int]]:
    r, c = pos
    if direction == "U":
        return (r - distance, c), (r - distance, c + lane)
    if direction == "D":
        return (r + distance, c), (r + distance + 1, c + lane)
    if direction == "L":
        return (r, c - distance), (r + lane, c - distance)
    return (r, c + distance), (r + lane, c + distance + 1)


def _sample_base_case(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*GRID_H_RANGE))
    w = int(rng.integers(*GRID_W_RANGE))
    start = (int(rng.integers(2, h - 3)), int(rng.integers(2, w - 3)))
    n_steps = int(rng.integers(*STEP_COUNT_RANGE))

    trail = np.zeros((h, w), dtype=bool)
    pos = start
    prev_dir = None
    used_dirs: list[str] = []
    used_dists: list[int] = []
    markers: list[tuple[int, int]] = []

    for step_idx in range(n_steps):
        choices = []
        for direction in DIRS:
            for lane in (0, 1):
                for distance in range(*STEP_DISTANCE_RANGE):
                    next_pos, marker = _move_spec(pos, direction, lane, distance)
                    nr, nc = next_pos
                    if not (0 <= nr and nr + 1 < h and 0 <= nc and nc + 1 < w):
                        continue
                    if marker in markers:
                        continue
                    if not _marker_isolated_from_set(markers, marker):
                        continue
                    current_mask = _sprite_mask((h, w), pos)
                    sweep = _sweep_mask((h, w), pos, next_pos)
                    path_mask = sweep & ~current_mask
                    if np.any(trail[path_mask]):
                        continue
                    choices.append((direction, lane, distance, next_pos, marker))

        if not choices:
            return None

        weights = []
        for direction, _lane, distance, next_pos, _marker in choices:
            weight = 1.0
            if prev_dir is not None and direction != prev_dir:
                weight *= 2.5
            if direction not in used_dirs:
                weight *= 1.7
            if distance not in used_dists:
                weight *= 1.2
            if prev_dir is not None and direction == OPPOSITE[prev_dir]:
                weight *= 0.6
            nr, nc = next_pos
            margin = min(nr, nc, h - (nr + 2), w - (nc + 2))
            if margin <= 1:
                weight *= 0.75
            weights.append(weight)

        choice = choices[int(rng.choice(len(choices), p=np.array(weights) / np.sum(weights)))]
        direction, _lane, distance, next_pos, marker = choice
        sweep = _sweep_mask((h, w), pos, next_pos)
        trail |= sweep
        trail[_sprite_mask((h, w), next_pos)] = False
        markers.append(marker)
        used_dirs.append(direction)
        used_dists.append(distance)
        prev_dir = direction
        pos = next_pos

    grid = np.full((h, w), BG_COLOR, dtype=int)
    sr, sc = start
    grid[sr : sr + 2, sc : sc + 2] = SPRITE_COLOR
    for mr, mc in markers:
        if grid[mr, mc] != BG_COLOR:
            return None
        grid[mr, mc] = MARKER_COLOR

    analysis = _simulate(grid)
    if not analysis["valid"]:
        return None
    if analysis["consumed"] != markers:
        return None
    dirs = {step["direction"] for step in analysis["steps"]}
    dists = {step["distance"] for step in analysis["steps"]}
    if len(dirs) < 2 or len(dists) < 2:
        return None
    return grid


def _same_base_order(analysis: dict, base_sequence: list[tuple[int, int]]) -> bool:
    filtered = [cell for cell in analysis["consumed"] if cell in set(base_sequence)]
    return filtered == base_sequence


def _search_trail_leftover(
    rng: np.random.Generator,
    grid: np.ndarray,
    base_sequence: list[tuple[int, int]],
    *,
    require_multi_visible: bool,
) -> np.ndarray | None:
    h, w = grid.shape
    empties = [(r, c) for r in range(h) for c in range(w) if grid[r, c] == BG_COLOR]
    rng.shuffle(empties)
    for r, c in empties:
        if not _marker_isolated(grid, (r, c)):
            continue
        trial = grid.copy()
        trial[r, c] = MARKER_COLOR
        analysis = _simulate(trial)
        if not analysis["valid"]:
            continue
        if not _same_base_order(analysis, base_sequence):
            continue
        if (r, c) in analysis["consumed"]:
            continue
        if require_multi_visible and max(step["candidate_count"] for step in analysis["steps"]) < 2:
            continue
        no_trail = _simulate(trial, use_trail=False, use_walls=True)
        if not no_trail["valid"]:
            continue
        if (r, c) not in no_trail["consumed"]:
            continue
        return trial
    return None


def _segment_cells(shape: tuple[int, int], anchor: tuple[int, int], orientation: str, length: int) -> list[tuple[int, int]] | None:
    h, w = shape
    r, c = anchor
    if orientation == "H":
        for start in range(c - length + 1, c + 1):
            cells = [(r, start + k) for k in range(length)]
            if cells[0][1] <= c <= cells[-1][1] and all(0 <= rr < h and 0 <= cc < w for rr, cc in cells):
                return cells
    else:
        for start in range(r - length + 1, r + 1):
            cells = [(start + k, c) for k in range(length)]
            if cells[0][0] <= r <= cells[-1][0] and all(0 <= rr < h and 0 <= cc < w for rr, cc in cells):
                return cells
    return None


def _marker_isolated(grid: np.ndarray, cell: tuple[int, int]) -> bool:
    h, w = grid.shape
    r, c = cell
    for rr in range(max(0, r - 1), min(h, r + 2)):
        for cc in range(max(0, c - 1), min(w, c + 2)):
            if (rr, cc) != cell and grid[rr, cc] == MARKER_COLOR:
                return False
    return True


def _marker_isolated_from_set(markers: list[tuple[int, int]], cell: tuple[int, int]) -> bool:
    r, c = cell
    for mr, mc in markers:
        if max(abs(r - mr), abs(c - mc)) <= 1:
            return False
    return True


def _search_wall_hidden_marker(
    rng: np.random.Generator,
    grid: np.ndarray,
    base_sequence: list[tuple[int, int]],
) -> np.ndarray | None:
    base_analysis = _simulate(grid)
    if not base_analysis["valid"]:
        return None
    h, w = grid.shape
    for sprite_pos in base_analysis["states"]:
        r, c = sprite_pos
        rays = [
            ("U", (c, c + 1), range(r - 1, -1, -1)),
            ("D", (c, c + 1), range(r + 2, h)),
            ("L", (r, r + 1), range(c - 1, -1, -1)),
            ("R", (r, r + 1), range(c + 2, w)),
        ]
        rng.shuffle(rays)
        for direction, lines, steps in rays:
            line_values = list(lines)
            step_values = list(steps)
            rng.shuffle(line_values)
            for line in line_values:
                for wall_dist in range(1, min(5, len(step_values))):
                    wall_coord = step_values[wall_dist - 1]
                    for marker_dist in range(wall_dist + 2, min(len(step_values) + 1, wall_dist + 7)):
                        marker_coord = step_values[marker_dist - 1]
                        if direction in ("U", "D"):
                            block_cell = (wall_coord, line)
                            marker_cell = (marker_coord, line)
                            orientations = ("H", "V")
                        else:
                            block_cell = (line, wall_coord)
                            marker_cell = (line, marker_coord)
                            orientations = ("V", "H")
                        if grid[marker_cell] != BG_COLOR or grid[block_cell] != BG_COLOR:
                            continue
                        if not _marker_isolated(grid, marker_cell):
                            continue
                        for orientation in orientations:
                            seg_len = int(rng.integers(2, 4))
                            segment = _segment_cells((h, w), block_cell, orientation, seg_len)
                            if segment is None:
                                continue
                            if any(grid[cell] != BG_COLOR for cell in segment):
                                continue
                            trial = grid.copy()
                            trial[marker_cell] = MARKER_COLOR
                            for cell in segment:
                                trial[cell] = WALL_COLOR
                            analysis = _simulate(trial)
                            if not analysis["valid"]:
                                continue
                            if not _same_base_order(analysis, base_sequence):
                                continue
                            if marker_cell in analysis["consumed"]:
                                continue
                            without_walls = trial.copy()
                            for cell in segment:
                                without_walls[cell] = BG_COLOR
                            changed = _simulate(without_walls)
                            if not changed["valid"]:
                                continue
                            if marker_cell not in changed["consumed"] and np.array_equal(changed["output"], analysis["output"]):
                                continue
                            return trial
    return None


def _search_filler_wall(rng: np.random.Generator, grid: np.ndarray) -> np.ndarray | None:
    h, w = grid.shape
    empties = [(r, c) for r in range(h) for c in range(w) if grid[r, c] == BG_COLOR]
    rng.shuffle(empties)
    for anchor in empties:
        for orientation in ("H", "V"):
            length = int(rng.integers(2, 5))
            segment = _segment_cells((h, w), anchor, orientation, length)
            if segment is None:
                continue
            if any(grid[cell] != BG_COLOR for cell in segment):
                continue
            trial = grid.copy()
            for cell in segment:
                trial[cell] = WALL_COLOR
            analysis = _simulate(trial)
            if analysis["valid"]:
                return trial
    return None


def _semantic_ok(grid: np.ndarray) -> bool:
    analysis = _simulate(grid)
    if not analysis["valid"]:
        return False
    if not analysis["remaining"]:
        return False
    if len(analysis["steps"]) < 3:
        return False
    if len({step["direction"] for step in analysis["steps"]}) < 2:
        return False
    if len({step["distance"] for step in analysis["steps"]}) < 2:
        return False
    if max(step["candidate_count"] for step in analysis["steps"]) < 2:
        return False
    no_trail = _simulate(grid, use_trail=False, use_walls=True)
    if not no_trail["valid"]:
        return False
    if np.array_equal(no_trail["output"], analysis["output"]):
        return False
    wallless = _simulate(np.where(grid == WALL_COLOR, BG_COLOR, grid), use_trail=True, use_walls=True)
    if not wallless["valid"]:
        return False
    if np.array_equal(wallless["output"], analysis["output"]):
        return False
    return True


def _build_case(rng: np.random.Generator) -> np.ndarray | None:
    base = _sample_base_case(rng)
    if base is None:
        return None
    base_analysis = _simulate(base)
    base_sequence = list(base_analysis["consumed"])

    grid = base
    need_multi = max(step["candidate_count"] for step in base_analysis["steps"]) < 2
    grid = _search_trail_leftover(rng, grid, base_sequence, require_multi_visible=need_multi)
    if grid is None:
        return None

    extra_leftover = _search_trail_leftover(rng, grid, base_sequence, require_multi_visible=False)
    if extra_leftover is not None:
        grid = extra_leftover

    walled = _search_wall_hidden_marker(rng, grid, base_sequence)
    if walled is None:
        return None
    grid = walled

    if np.count_nonzero(grid == WALL_COLOR) < 2:
        filled = _search_filler_wall(rng, grid)
        if filled is None:
            return None
        grid = filled

    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _build_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
