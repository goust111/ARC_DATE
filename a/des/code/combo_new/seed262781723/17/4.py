from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
GRAY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _degree_map(mask: np.ndarray) -> np.ndarray:
    deg = np.zeros(mask.shape, dtype=int)
    deg[1:, :] += mask[:-1, :]
    deg[:-1, :] += mask[1:, :]
    deg[:, 1:] += mask[:, :-1]
    deg[:, :-1] += mask[:, 1:]
    return deg


def _neighbors(mask: np.ndarray, cell: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]):
            out.append((nr, nc))
    return out


def _trace_seed(
    grid: np.ndarray,
    walkable: np.ndarray,
    seed: tuple[int, int],
    seed_set: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], int]]:
    color = int(grid[seed])
    prev: tuple[int, int] | None = None
    current = seed
    distance = 0
    seen_states: set[tuple[tuple[int, int] | None, tuple[int, int]]] = set()
    covered: list[tuple[tuple[int, int], int]] = []

    while True:
        if current not in seed_set:
            covered.append((current, distance))

        state = (prev, current)
        if state in seen_states:
            break
        seen_states.add(state)

        candidates = [nb for nb in _neighbors(walkable, current) if prev is None or nb != prev]
        if not candidates:
            break

        if prev is None or len(candidates) == 1:
            nxt = candidates[0]
        else:
            in_dir = (current[0] - prev[0], current[1] - prev[1])
            matching = [nb for nb in candidates if int(grid[nb]) == color and nb not in seed_set]
            if len(matching) == 1:
                nxt = matching[0]
            else:
                straight = [nb for nb in candidates if (nb[0] - current[0], nb[1] - current[1]) == in_dir]
                if len(straight) == 1:
                    nxt = straight[0]
                else:
                    break

        prev, current = current, nxt
        distance += 1

    return covered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each seed color through the grey corridor network, using same-colored junction markers to choose turns.

    Input structure:
    - The background is black.
    - Grey pixels form an orthogonal corridor network with dead ends and right-angle intersections.
    - Colored seed pixels sit on some dead ends, and colored marker pixels sit on corridor cells adjacent to junction centers.

    Rule:
    - Start one colored ribbon from every seed and let it advance through the connected corridor cells.
    - Along straight corridor segments the ribbon keeps moving forward; at a junction it prefers the outgoing corridor cell whose marker matches the seed color, otherwise it continues straight if that is the only unambiguous continuation.
    - Every grey corridor pixel reached by a ribbon is recolored to the seed color, and traversed marker pixels are covered as part of the same route.
    - If two ribbons would color the same corridor cell, keep the color of the ribbon that reaches that cell in fewer steps from its seed.

    Key insight:
    - The seed color does not only identify a ribbon; it also matches the colored marker that selects left, right, or straight at each junction, so solving requires tracing the whole route and then resolving shortest-path overlaps.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    non_bg = [int(color) for color in np.unique(g) if int(color) != int(bg)]
    assert non_bg, "expected a non-background corridor color"
    gray = max(non_bg, key=lambda color: int(np.sum(g == int(color))))

    walkable = g != int(bg)
    degree = _degree_map(walkable)
    seed_mask = (g != int(bg)) & (g != int(gray)) & (degree == 1)
    seed_coords = [tuple(map(int, pt)) for pt in np.argwhere(seed_mask)]
    seed_set = set(seed_coords)
    assert seed_coords, "expected at least one colored seed endpoint"

    best: dict[tuple[int, int], tuple[int, int]] = {}
    for seed in seed_coords:
        color = int(g[seed])
        for cell, dist in _trace_seed(g, walkable, seed, seed_set):
            current = best.get(cell)
            if current is None or dist < current[0]:
                best[cell] = (int(dist), color)

    out = g.copy()
    for (r, c), (_dist, color) in best.items():
        out[r, c] = int(color)
    return out


def _sample_positions(length: int, count: int, min_gap: int) -> list[int] | None:
    choices = list(range(2, length - 2))
    for _ in range(100):
        pos = sorted(random.sample(choices, count))
        if all(b - a >= min_gap for a, b in zip(pos, pos[1:])):
            return pos
    return None


def _rotate_patch(patch: np.ndarray) -> np.ndarray:
    return np.rot90(patch, random.randint(0, 3)).copy()


def _build_route_component(color: int) -> np.ndarray | None:
    length = random.randint(11, 17)
    max_cross = min(3, max(1, (length - 5) // 4))
    cross_count = random.randint(1, max_cross)
    xs = _sample_positions(length, cross_count, min_gap=4)
    if xs is None:
        return None

    up_lengths = {x: random.randint(2, 4) for x in xs}
    down_lengths = {x: random.randint(2, 4) for x in xs}
    top = max(up_lengths.values())
    bottom = max(down_lengths.values())
    mid = top
    height = top + 1 + bottom

    walkable: set[tuple[int, int]] = {(mid, c) for c in range(length)}
    for x in xs:
        for delta in range(1, up_lengths[x] + 1):
            walkable.add((mid - delta, x))
        for delta in range(1, down_lengths[x] + 1):
            walkable.add((mid + delta, x))

    adj = {cell: [] for cell in walkable}
    for r, c in walkable:
        for dr, dc in DIRS:
            nb = (r + dr, c + dc)
            if nb in walkable:
                adj[(r, c)].append(nb)

    endpoints = [cell for cell, nbs in adj.items() if len(nbs) == 1]
    if not endpoints:
        return None
    start = random.choice(endpoints)

    markers: set[tuple[int, int]] = set()
    relation_counts = Counter()
    prev: tuple[int, int] | None = None
    current = start
    seen_states: set[tuple[tuple[int, int] | None, tuple[int, int]]] = set()

    while True:
        state = (prev, current)
        if state in seen_states:
            return None
        seen_states.add(state)

        candidates = [nb for nb in adj[current] if prev is None or nb != prev]
        if not candidates:
            break
        if prev is None or len(candidates) == 1:
            nxt = candidates[0]
        else:
            in_dir = (current[0] - prev[0], current[1] - prev[1])
            options: list[tuple[str, tuple[int, int]]] = []
            for nb in candidates:
                out_dir = (nb[0] - current[0], nb[1] - current[1])
                if out_dir == in_dir:
                    rel = "straight"
                elif out_dir == (-in_dir[1], in_dir[0]):
                    rel = "left"
                elif out_dir == (in_dir[1], -in_dir[0]):
                    rel = "right"
                else:
                    continue
                options.append((rel, nb))
            if not options:
                return None
            rel, nxt = random.choice(options)
            markers.add(nxt)
            relation_counts[rel] += 1
        prev, current = current, nxt

    if sum(relation_counts.values()) == 0:
        return None

    patch = np.full((height, length), BG, dtype=int)
    for r, c in walkable:
        patch[r, c] = GRAY
    patch[start] = color
    for marker in markers:
        patch[marker] = color
    return _rotate_patch(patch)


def _build_conflict_component(color_a: int, color_b: int) -> np.ndarray:
    inner = random.choice([4, 6, 8, 10])
    length = inner + 2
    patch = np.full((3, length), BG, dtype=int)
    row = 1
    patch[row, 1 : length - 1] = GRAY
    patch[row, 0] = color_a
    patch[row, length - 1] = color_b
    return _rotate_patch(patch)


def _can_place(occupied: np.ndarray, coords: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        r = int(r)
        c = int(c)
        if not (0 <= r < h and 0 <= c < w):
            return False
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        if bool(np.any(occupied[r0:r1, c0:c1])):
            return False
    return True


def _place_patch(canvas: np.ndarray, occupied: np.ndarray, patch: np.ndarray) -> bool:
    coords = np.argwhere(patch != BG)
    ph, pw = patch.shape
    h, w = canvas.shape
    if ph > h or pw > w:
        return False

    for _ in range(200):
        r0 = random.randint(0, h - ph)
        c0 = random.randint(0, w - pw)
        shifted = coords + np.array([r0, c0])
        if not _can_place(occupied, shifted):
            continue
        for rr, cc in shifted:
            rr = int(rr)
            cc = int(cc)
            canvas[rr, cc] = int(patch[rr - r0, cc - c0])
            occupied[rr, cc] = True
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    route_count = random.randint(1, 2)
    need_conflict = random.choice([False, True])
    color_count = route_count + 2 if need_conflict else route_count
    colors = random.sample(PALETTE, color_count)

    patches: list[np.ndarray] = []
    for color in colors[:route_count]:
        patch = _build_route_component(color)
        if patch is None:
            return None
        patches.append(patch)

    if need_conflict:
        patches.append(_build_conflict_component(colors[-2], colors[-1]))

    max_h = max(int(p.shape[0]) for p in patches)
    max_w = max(int(p.shape[1]) for p in patches)
    h_lo = max(16, max_h + 2)
    w_lo = max(16, max_w + 2)
    h = random.randint(h_lo, 30)
    w = random.randint(w_lo, 30)

    canvas = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    random.shuffle(patches)
    patches.sort(key=lambda p: int(np.sum(p != BG)), reverse=True)
    for patch in patches:
        if not _place_patch(canvas, occupied, patch):
            return None

    try:
        out = main(canvas)
    except Exception:
        return None

    if not np.any((canvas == GRAY) & (out != GRAY)):
        return None

    seed_mask = (canvas != BG) & (canvas != GRAY) & (_degree_map(canvas != BG) == 1)
    if int(np.sum(seed_mask)) < route_count + (2 if need_conflict else 0):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
