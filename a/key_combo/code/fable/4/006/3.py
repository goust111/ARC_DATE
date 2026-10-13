from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
RED = 2
GREEN = 3
TRACK_COLORS = {RED, GREEN}
SQUARE_COLORS = [1, 4, 5, 6, 7, 8, 9]
DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))
GEN_RETRIES = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                yield r + dr, c + dc


def _components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        comp: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for nr, nc in _neighbors8(r, c):
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_squares(g: np.ndarray) -> list[dict]:
    obj_mask = (g != BG) & (g != RED) & (g != GREEN)
    squares = []
    for comp in _components8(obj_mask):
        r0, c0, r1, c1 = _bbox(comp)
        if len(comp) != 4 or (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
            continue
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        if np.all((patch != BG) & (patch != RED) & (patch != GREEN)):
            squares.append({"bbox": (r0, c0, r1, c1), "patch": patch})
    squares.sort(key=lambda s: s["bbox"])
    return squares


def _ray_track_candidate(
    g: np.ndarray,
    anchor: tuple[int, int],
    direction: tuple[int, int],
) -> dict | None:
    ar, ac = anchor
    dr, dc = direction
    visible_dists: list[int] = []
    for r, c in np.argwhere((g == GREEN) | (g == RED)):
        r, c = int(r), int(c)
        rr, cc = r - ar, c - ac
        if dr == 0:
            if rr != 0 or cc * dc <= 0:
                continue
            visible_dists.append(abs(cc))
        else:
            if cc != 0 or rr * dr <= 0:
                continue
            visible_dists.append(abs(rr))
    if len(visible_dists) < 3:
        return None

    step = min(visible_dists)
    if step < 3:
        return None

    before = (ar - dr * step, ac - dc * step)
    if _in_bounds(g.shape, *before) and int(g[before]) in TRACK_COLORS:
        return None

    slots = [(ar, ac)]
    statuses = [GREEN]
    k = 1
    while True:
        r, c = ar + dr * step * k, ac + dc * step * k
        if not _in_bounds(g.shape, r, c):
            break
        v = int(g[r, c])
        if v not in TRACK_COLORS:
            break
        slots.append((r, c))
        statuses.append(v)
        k += 1

    if len(slots) < 4:
        return None

    max_dist = step * (len(slots) - 1)
    for dist in range(1, max_dist + 1):
        r, c = ar + dr * dist, ac + dc * dist
        if int(g[r, c]) in TRACK_COLORS and dist % step != 0:
            return None

    return {
        "anchor": anchor,
        "direction": direction,
        "step": step,
        "slots": slots,
        "statuses": statuses,
    }


def _candidate_tracks_for_square(g: np.ndarray, square: dict) -> list[dict]:
    r0, c0, r1, c1 = square["bbox"]
    candidates = []
    for ar in range(r0, r1 + 1):
        for ac in range(c0, c1 + 1):
            for direction in DIRS:
                cand = _ray_track_candidate(g, (ar, ac), direction)
                if cand is None:
                    continue
                cand = dict(cand)
                cand["square"] = square
                cand["anchor_offset"] = (ar - r0, ac - c0)
                candidates.append(cand)
    return candidates


def _assign_tracks(g: np.ndarray, squares: list[dict]) -> list[dict]:
    all_candidates = [_candidate_tracks_for_square(g, sq) for sq in squares]
    assert all(all_candidates), "every square must identify one dotted track"

    order = sorted(range(len(squares)), key=lambda i: len(all_candidates[i]))
    solutions: list[list[dict | None]] = []
    current: list[dict | None] = [None] * len(squares)

    def dfs(pos: int, used: set[tuple[int, int]]) -> None:
        if len(solutions) > 1:
            return
        if pos == len(order):
            solutions.append(current.copy())
            return
        idx = order[pos]
        for cand in all_candidates[idx]:
            pts = set(cand["slots"])
            if pts & used:
                continue
            current[idx] = cand
            dfs(pos + 1, used | pts)
            current[idx] = None

    dfs(0, set())
    assert len(solutions) == 1, "track ownership must be uniquely determined"
    return [cand for cand in solutions[0] if cand is not None]


def _color_count(patch: np.ndarray) -> int:
    colors = {int(v) for v in patch.ravel() if int(v) != BG}
    return len(colors)


def _simulate(statuses: list[int], steps: int) -> tuple[int, bool, bool]:
    assert len(statuses) >= 2, "track needs at least two slots"
    assert any(v == GREEN for v in statuses), "track needs a usable green slot"
    idx = 0
    direction = 1
    remaining = int(steps)
    bounced = False
    skipped = False
    guard = 10 + len(statuses) * max(1, remaining) * 6
    while remaining > 0:
        guard -= 1
        assert guard > 0, "track simulation did not terminate"
        nxt = idx + direction
        if nxt < 0 or nxt >= len(statuses):
            direction *= -1
            bounced = True
            nxt = idx + direction
        idx = nxt
        if statuses[idx] == RED:
            skipped = True
            continue
        assert statuses[idx] == GREEN
        remaining -= 1
    return idx, bounced, skipped


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each 2x2 color-diversity block along its dotted track, skipping blocked slots and bouncing at the far end.

    Input structure:
    - The black background contains several straight horizontal or vertical tracks made of evenly spaced green dots.
    - Some track dots are red blocked slots, and each track's first green slot is hidden under a 2x2 non-track-colored square.
    - Every square contains one to four distinct non-black colors; red and green are reserved for the track marks.

    Rule:
    - Count the distinct colors inside each square to get its step count.
    - Starting from the hidden first slot, advance along the track direction over that many green slots; red slots are crossed but not counted.
    - If counting would run past the far endpoint, reverse direction at the endpoint and continue counting with the same red-slot rule.
    - Clear the old square, restore the hidden green start dot, redraw the track dots, and place the square with the same anchor on the counted green slot.

    Key insight:
    - The motion distance is encoded by color diversity inside each square, while the visible dotted track defines the discrete slots, red exemptions, and endpoint reflection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30 by 30"

    squares = _find_squares(g)
    assert squares, "expected at least one 2x2 multicolor square"
    tracks = _assign_tracks(g, squares)

    out = g.copy()
    for sq in squares:
        r0, c0, r1, c1 = sq["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BG

    for tr in tracks:
        for (r, c), status in zip(tr["slots"], tr["statuses"]):
            out[r, c] = int(status)

    for tr in tracks:
        sq = tr["square"]
        patch = sq["patch"]
        steps = _color_count(patch)
        target_i, _bounced, _skipped = _simulate(tr["statuses"], steps)
        target_r, target_c = tr["slots"][target_i]
        off_r, off_c = tr["anchor_offset"]
        paste_r, paste_c = target_r - off_r, target_c - off_c
        ph, pw = patch.shape
        assert 0 <= paste_r and 0 <= paste_c
        assert paste_r + ph <= out.shape[0] and paste_c + pw <= out.shape[1]
        out[paste_r : paste_r + ph, paste_c : paste_c + pw] = patch

    return out


def _make_square(distinct: int) -> np.ndarray:
    colors = random.sample(SQUARE_COLORS, distinct)
    cells = colors[:] + [random.choice(colors) for _ in range(4 - distinct)]
    random.shuffle(cells)
    return np.array(cells, dtype=int).reshape(2, 2)


def _status_sample(category: str, steps: int) -> list[int] | None:
    if category == "regular":
        length = random.randint(max(5, steps + 2), 7)
        return [GREEN] * length

    if category == "bounce":
        if steps != 4:
            return None
        statuses = [GREEN, GREEN, GREEN, GREEN]
        if random.random() < 0.35:
            statuses[-1] = RED
        idx, bounced, _skipped = _simulate(statuses, steps)
        if bounced and idx != 0:
            return statuses
        return None

    if category == "skip_bounce":
        if steps != 4:
            return None
        statuses = [GREEN, GREEN, GREEN, RED]
        idx, bounced, skipped = _simulate(statuses, steps)
        if bounced and skipped and idx != 0:
            return statuses
        return None

    if category == "skip":
        for _ in range(80):
            length = random.randint(max(5, steps + 2), 7)
            statuses = [GREEN] * length
            max_red = min(2, length - 2)
            red_positions = random.sample(range(1, length - 1), random.randint(1, max_red))
            for p in red_positions:
                statuses[p] = RED
            idx, bounced, skipped = _simulate(statuses, steps)
            no_skip_idx, _, _ = _simulate([GREEN] * length, steps)
            if skipped and not bounced and idx != no_skip_idx:
                return statuses
        return None

    raise AssertionError(f"unknown category: {category}")


def _slot_coords(start: tuple[int, int], direction: tuple[int, int], step: int, length: int) -> list[tuple[int, int]]:
    sr, sc = start
    dr, dc = direction
    return [(sr + dr * step * i, sc + dc * step * i) for i in range(length)]


def _patch_cells(r: int, c: int) -> list[tuple[int, int]]:
    return [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]


def _mark_reserved(reserved: np.ndarray, cells: list[tuple[int, int]], pad: int = 1) -> None:
    h, w = reserved.shape
    for r, c in cells:
        r0, r1 = max(0, r - pad), min(h, r + pad + 2)
        c0, c1 = max(0, c - pad), min(w, c + pad + 2)
        reserved[r0:r1, c0:c1] = True


def _random_start(shape: tuple[int, int], direction: tuple[int, int], step: int, length: int) -> tuple[int, int] | None:
    h, w = shape
    dr, dc = direction
    span = step * (length - 1)
    r_min, r_max = 1, h - 3
    c_min, c_max = 1, w - 3
    if dr == 1:
        r_max -= span
    elif dr == -1:
        r_min += span
    if dc == 1:
        c_max -= span
    elif dc == -1:
        c_min += span
    if r_min > r_max or c_min > c_max:
        return None
    return random.randint(r_min, r_max), random.randint(c_min, c_max)


def _try_place_track(
    grid: np.ndarray,
    reserved: np.ndarray,
    category: str,
) -> dict | None:
    if category in ("bounce", "skip_bounce"):
        steps = 4
    elif category == "regular":
        steps = random.choice([1, 2, 3, 4])
    else:
        steps = random.choice([2, 3, 4])

    statuses = _status_sample(category, steps)
    if statuses is None:
        return None

    patch = _make_square(steps)
    target_idx, bounced, skipped = _simulate(statuses, steps)
    if target_idx == 0:
        return None

    direction = random.choice(DIRS)
    step = random.choice([3, 4])
    start = _random_start(grid.shape, direction, step, len(statuses))
    if start is None:
        return None
    slots = _slot_coords(start, direction, step, len(statuses))

    start_cells = _patch_cells(*start)
    target_cells = _patch_cells(*slots[target_idx])
    visible_slots = slots[1:]
    local_input = start_cells + visible_slots
    local_output = target_cells

    for r, c in local_input + local_output:
        if not _in_bounds(grid.shape, r, c) or reserved[r, c]:
            return None

    # The only intentional local overlap is the destination square covering its landing dot.
    target_set = set(target_cells)
    for i, p in enumerate(visible_slots, start=1):
        if p in target_set and i != target_idx:
            return None
        if p in set(start_cells):
            return None

    for (r, c), status in zip(slots[1:], statuses[1:]):
        grid[r, c] = int(status)
    r0, c0 = start
    grid[r0 : r0 + 2, c0 : c0 + 2] = patch

    reserve_cells = local_input + local_output
    _mark_reserved(reserved, reserve_cells, pad=1)
    return {
        "category": category,
        "start": start,
        "direction": direction,
        "step": step,
        "slots": slots,
        "statuses": statuses,
        "patch": patch,
        "steps": steps,
        "target_idx": target_idx,
        "bounced": bounced,
        "skipped": skipped,
    }


def _expected_from_specs(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        r0, c0 = spec["start"]
        out[r0 : r0 + 2, c0 : c0 + 2] = BG
    for spec in specs:
        for (r, c), status in zip(spec["slots"], spec["statuses"]):
            out[r, c] = int(status)
    for spec in specs:
        r, c = spec["slots"][spec["target_idx"]]
        out[r : r + 2, c : c + 2] = spec["patch"]
    return out


def _semantic_checks(specs: list[dict]) -> bool:
    if not any(spec["category"] == "regular" and not spec["bounced"] and not spec["skipped"] for spec in specs):
        return False
    if not any(spec["bounced"] for spec in specs):
        return False
    if not any(spec["skipped"] for spec in specs):
        return False

    for spec in specs:
        if _color_count(spec["patch"]) != spec["steps"]:
            return False
        idx, bounced, skipped = _simulate(spec["statuses"], spec["steps"])
        if idx != spec["target_idx"] or bounced != spec["bounced"] or skipped != spec["skipped"]:
            return False
        if spec["skipped"]:
            no_skip_idx, _, _ = _simulate([GREEN] * len(spec["statuses"]), spec["steps"])
            if no_skip_idx == idx:
                return False
        if spec["bounced"]:
            forward_green = sum(1 for v in spec["statuses"][1:] if v == GREEN)
            if forward_green >= spec["steps"]:
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(20, 30)
    w = random.randint(20, 30)
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    categories = ["regular", "skip", "bounce"]
    if random.random() < 0.45:
        categories.append(random.choice(["regular", "skip", "skip_bounce"]))
    random.shuffle(categories)

    specs = []
    for category in categories:
        placed = None
        for _ in range(250):
            trial_grid = grid.copy()
            trial_reserved = reserved.copy()
            placed = _try_place_track(trial_grid, trial_reserved, category)
            if placed is not None:
                grid = trial_grid
                reserved = trial_reserved
                break
        if placed is None:
            return None
        specs.append(placed)

    if not _semantic_checks(specs):
        return None

    expected = _expected_from_specs(grid, specs)
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fable/4/006/3 input")
