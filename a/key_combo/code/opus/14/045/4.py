from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
WALL = 5
MARKER_COLORS = (1, 2, 3)
MAX_GENERATION_ATTEMPTS = 400


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    result: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            result.append(np.array(cells, dtype=int))
    return result


def _majority_color(colors: list[int]) -> int:
    counts = Counter(colors)
    [(top_color, top_count), *rest] = counts.most_common()
    if rest and top_count == rest[0][1]:
        raise AssertionError("marker vote must have a unique strict majority")
    return int(top_color)


def fill_seed(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()
    h, w = g.shape
    for chamber in components(g == BG):
        markers: dict[tuple[int, int], int] = {}
        for r, c in chamber:
            for nr, nc in neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w:
                    color = int(g[nr, nc])
                    if color not in (BG, WALL):
                        markers[(nr, nc)] = color
        assert len(markers) >= 2, "each chamber must touch at least two markers"
        fill = _majority_color(list(markers.values()))
        out[chamber[:, 0], chamber[:, 1]] = fill
    return out


def reflect_tile(seed: np.ndarray) -> np.ndarray:
    top = np.concatenate((seed, np.fliplr(seed)), axis=1)
    return np.concatenate((top, np.flipud(top)), axis=0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each wall-bounded black chamber by boundary-marker majority, then mirror-tile the filled seed across both axes.

    Input structure:
    - The grid contains black background cells inside enclosed chambers, grey wall cells, and red/blue/green marker pixels embedded in wall positions.
    - Black cells form 4-connected chambers fully separated by the wall-and-marker barrier.

    Rule:
    - For each 4-connected black chamber, collect all orthogonally adjacent marker pixels and find the unique majority color among them.
    - Recolor every black cell in that chamber with the majority color while leaving walls and markers unchanged; this produces the hidden seed tile.
    - Return a 2x2 biaxial reflection tiling of that seed tile: upper-left = seed, upper-right = left-right mirror, lower-left = up-down mirror, lower-right = 180-degree rotation.

    Key insight:
    - The visible input is only a voting scaffold for reconstructing an unseen filled tile, and only that reconstructed tile should be used for the geometric reflection step.
    """
    seed = fill_seed(np.asarray(grid, dtype=int))
    return reflect_tile(seed)


def _bbox_area(coords: np.ndarray) -> int:
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _has_irregular_chamber(chambers: list[np.ndarray]) -> bool:
    return any(len(chamber) < _bbox_area(chamber) for chamber in chambers)


def _enumerate_splits(room: tuple[int, int, int, int], min_area: int):
    r0, r1, c0, c1 = room
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    choices = []
    for split_r in range(r0 + 1, r1 + 1):
        top_h = split_r - r0
        bottom_h = r1 - split_r
        if top_h * width >= min_area and bottom_h * width >= min_area:
            choices.append(("h", split_r))
    for split_c in range(c0 + 1, c1 + 1):
        left_w = split_c - c0
        right_w = c1 - split_c
        if height * left_w >= min_area and height * right_w >= min_area:
            choices.append(("v", split_c))
    return choices


def _split_rooms(height: int, width: int, target_chambers: int) -> np.ndarray | None:
    grid = np.full((height, width), WALL, dtype=int)
    grid[1:-1, 1:-1] = BG
    rooms = [(1, height - 2, 1, width - 2)]
    min_area = 2
    while len(rooms) < target_chambers:
        candidates = []
        for idx, room in enumerate(rooms):
            for orientation, split_at in _enumerate_splits(room, min_area):
                candidates.append((idx, room, orientation, split_at))
        if not candidates:
            return None
        idx, room, orientation, split_at = random.choice(candidates)
        r0, r1, c0, c1 = room
        rooms.pop(idx)
        if orientation == "h":
            grid[split_at, c0 : c1 + 1] = WALL
            rooms.append((r0, split_at - 1, c0, c1))
            rooms.append((split_at + 1, r1, c0, c1))
        else:
            grid[r0 : r1 + 1, split_at] = WALL
            rooms.append((r0, r1, c0, split_at - 1))
            rooms.append((r0, r1, split_at + 1, c1))
    return grid


def _nibble_one_wall_cell(grid: np.ndarray) -> np.ndarray | None:
    chambers = components(grid == BG)
    chamber_count = len(chambers)
    choices = []
    h, w = grid.shape
    for chamber in chambers:
        chamber_cells = {tuple(cell) for cell in chamber.tolist()}
        for r, c in chamber_cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                wr, wc = r - dr, c - dc
                if not (0 <= wr < h and 0 <= wc < w and grid[wr, wc] == WALL):
                    continue
                trial = grid.copy()
                trial[r, c] = WALL
                trial_chambers = components(trial == BG)
                if len(trial_chambers) == chamber_count and _has_irregular_chamber(trial_chambers):
                    choices.append(trial)
    if not choices:
        return None
    return random.choice(choices)


def _wall_adjacency(grid: np.ndarray, chambers: list[np.ndarray]) -> dict[tuple[int, int], tuple[int, ...]]:
    chamber_index: dict[tuple[int, int], int] = {}
    for idx, chamber in enumerate(chambers):
        for cell in chamber.tolist():
            chamber_index[tuple(cell)] = idx
    adjacency: dict[tuple[int, int], set[int]] = {}
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if grid[r, c] != WALL:
                continue
            ids = set()
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) in chamber_index:
                    ids.add(chamber_index[(nr, nc)])
            if ids:
                adjacency[(r, c)] = ids
    return {pos: tuple(sorted(ids)) for pos, ids in adjacency.items()}


def _assign_chamber_colors(count: int) -> list[int]:
    used = random.sample(MARKER_COLORS, k=3 if count >= 4 and random.random() < 0.6 else 2)
    colors = [random.choice(used) for _ in range(count)]
    if len(set(colors)) == 1:
        colors[random.randrange(count)] = used[1]
    return colors


def _paint_markers(base_grid: np.ndarray, chambers: list[np.ndarray]) -> np.ndarray | None:
    adjacency = _wall_adjacency(base_grid, chambers)
    private_slots = {
        idx: [pos for pos, ids in adjacency.items() if ids == (idx,)]
        for idx in range(len(chambers))
    }
    if any(len(slots) < 2 for slots in private_slots.values()):
        return None

    chamber_colors = _assign_chamber_colors(len(chambers))
    mixed_chamber = random.randrange(len(chambers))
    grid = base_grid.copy()
    used_positions: set[tuple[int, int]] = set()
    had_mixed = False

    for idx, target in enumerate(chamber_colors):
        slots = private_slots[idx][:]
        random.shuffle(slots)
        max_total = min(4, len(slots))
        if max_total < 2:
            return None
        if idx == mixed_chamber:
            if max_total < 3:
                return None
            total = random.randint(3, max_total)
            minority = random.randint(1, min(2, total - 2))
            had_mixed = True
        else:
            total = random.randint(2, max_total)
            minority = 0
            if total >= 3 and random.random() < 0.55:
                minority = 1
                had_mixed = True
        majority = total - minority
        if majority <= minority:
            return None
        chosen = slots[:total]
        for pos in chosen[:majority]:
            grid[pos] = target
            used_positions.add(pos)
        for pos in chosen[majority:]:
            other_choices = [color for color in MARKER_COLORS if color != target]
            grid[pos] = random.choice(other_choices)
            used_positions.add(pos)

    if not had_mixed:
        return None

    # Optionally add a shared-wall marker as a distractor when it keeps every chamber majority strict.
    shared_positions = [pos for pos, ids in adjacency.items() if len(ids) >= 2 and pos not in used_positions]
    random.shuffle(shared_positions)
    for pos in shared_positions[:2]:
        ids = adjacency[pos]
        candidate_colors = list(MARKER_COLORS)
        random.shuffle(candidate_colors)
        original = grid[pos]
        for color in candidate_colors:
            grid[pos] = color
            try:
                fill_seed(grid)
                break
            except AssertionError:
                grid[pos] = original

    if len(set(chamber_colors)) < 2:
        return None
    return grid


def _build_candidate() -> np.ndarray | None:
    target_chambers = random.randint(3, 5)
    size_pool = [(6, 6), (6, 7), (7, 6), (7, 7)]
    if target_chambers == 5:
        size_pool = [(7, 7), (7, 6), (6, 7)]
    height, width = random.choice(size_pool)
    base = _split_rooms(height, width, target_chambers)
    if base is None:
        return None
    nibbled = _nibble_one_wall_cell(base)
    if nibbled is None:
        return None
    chambers = components(nibbled == BG)
    if not (3 <= len(chambers) <= 5):
        return None
    if not _has_irregular_chamber(chambers):
        return None
    return _paint_markers(nibbled, chambers)


def _semantic_checks(inp: np.ndarray) -> bool:
    if not isinstance(inp, np.ndarray) or inp.ndim != 2:
        return False
    if max(inp.shape) > 30:
        return False
    if not set(np.unique(inp)).issubset({0, 1, 2, 3, 5}):
        return False
    chambers = components(inp == BG)
    if not (3 <= len(chambers) <= 5):
        return False
    if not _has_irregular_chamber(chambers):
        return False

    h, w = inp.shape
    mixed_chambers = 0
    fills = []
    for chamber in chambers:
        marker_positions = {}
        for r, c in chamber:
            for nr, nc in neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w:
                    color = int(inp[nr, nc])
                    if color not in (BG, WALL):
                        marker_positions[(nr, nc)] = color
        if len(marker_positions) < 2:
            return False
        colors = list(marker_positions.values())
        fill = _majority_color(colors)
        if len(set(colors)) >= 2:
            mixed_chambers += 1
        fills.append(fill)
    if mixed_chambers == 0:
        return False
    if len(set(fills)) < 2:
        return False

    seed = fill_seed(inp)
    if np.array_equal(seed, np.fliplr(seed)) or np.array_equal(seed, np.flipud(seed)):
        return False
    out = main(inp)
    if out.shape != (inp.shape[0] * 2, inp.shape[1] * 2):
        return False
    if not np.array_equal(out, reflect_tile(seed)):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _build_candidate()
        if candidate is not None and _semantic_checks(candidate):
            return candidate.astype(int)
    raise RuntimeError("failed to generate a valid input grid")
