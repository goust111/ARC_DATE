import itertools
import math
import random
from collections import Counter

import numpy as np


RING_SHAPE = 4
RING_CELLS = tuple(
    (r, c)
    for r in range(RING_SHAPE)
    for c in range(RING_SHAPE)
    if r in (0, RING_SHAPE - 1) or c in (0, RING_SHAPE - 1)
)
RING_SET = frozenset(RING_CELLS)
COLOR_POOL = tuple(range(1, 10))

GEN_H_RANGE = (18, 25)
GEN_W_RANGE = (23, 27)
GEN_RING_COUNT_RANGE = (4, 9)
GEN_GROUP_GAP = 1
GEN_COL_WINDOW_WIDTH = 7
GEN_ROW_WINDOW_HEIGHT = 4
GEN_MAX_GROUP_SIZE = 4
GEN_CLIP_PROB = 0.35
GEN_MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _translated_ring(offset: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = offset
    return {(r0 + dr, c0 + dc) for dr, dc in RING_CELLS}


def _matching_offsets(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    if not coords:
        return []
    matches: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr, dc in RING_CELLS:
            offset = (r - dr, c - dc)
            if all((rr - offset[0], cc - offset[1]) in RING_SET for rr, cc in coords):
                matches.add(offset)
    return sorted(matches)


def _detect_sprites(grid: np.ndarray) -> dict[int, dict]:
    sprites: dict[int, dict] = {}
    for color in sorted(int(x) for x in np.unique(grid) if int(x) != 0):
        coords = {tuple(map(int, rc)) for rc in np.argwhere(grid == color)}
        if len(coords) < 4:
            continue
        offsets = _matching_offsets(coords)
        if len(offsets) != 1:
            continue
        offset = offsets[0]
        sprites[color] = {
            "color": color,
            "coords": coords,
            "offset": offset,
        }
    return sprites


def _occlusion_graph(grid: np.ndarray, sprites: dict[int, dict]) -> dict[int, set[int]]:
    h, w = grid.shape
    colors = set(sprites)
    edges = {color: set() for color in colors}
    for below, info in sprites.items():
        expected = _translated_ring(info["offset"])
        for r, c in expected:
            if not _in_bounds((h, w), r, c):
                continue
            seen = int(grid[r, c])
            if seen == below:
                continue
            if seen not in colors:
                raise AssertionError("missing ring pixel must be covered by another legal sprite")
            edges[seen].add(below)
    return edges


def _topological_order(sprites: dict[int, dict], edges: dict[int, set[int]]) -> list[int]:
    indegree = {color: 0 for color in sprites}
    for above, belows in edges.items():
        for below in belows:
            indegree[below] += 1

    remaining = set(sprites)
    order: list[int] = []
    while remaining:
        ready = [color for color in remaining if indegree[color] == 0]
        if not ready:
            raise AssertionError("occlusion relation must be acyclic")
        ready.sort(key=lambda color: sprites[color]["offset"])
        color = ready[0]
        order.append(color)
        remaining.remove(color)
        for below in edges[color]:
            indegree[below] -= 1
    return order


def _snake_output(colors: list[int]) -> np.ndarray:
    rows = int(math.ceil(len(colors) / 3))
    out = np.zeros((rows, 3), dtype=int)
    for idx, color in enumerate(colors):
        r = idx // 3
        k = idx % 3
        c = k if r % 2 == 0 else 2 - k
        out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer front-to-back ring occlusion order and encode it as a compact snake-filled color grid.

    Input structure:
    - The background is black.
    - Several distinct non-black colors are visible fragments of identical 4x4 hollow rectangular rings.
    - Rings partially overlap, so upper rings cover expected ring pixels of lower rings; a few isolated off-template noise pixels may also exist.

    Rule:
    - For each substantial color, keep it only if all of its visible pixels fit one translated 4x4 ring border template.
    - Use every missing in-bounds template pixel to identify which other legal ring color is covering it, creating an above->below occlusion graph.
    - Topologically sort colors from top to bottom, breaking ties by ring top-left row then column, and write that sequence into a ceil(N/3)-by-3 black canvas in boustrophedon order.

    Key insight:
    - The spatial overlap does not need full sprite reconstruction in the output; it only supplies a deterministic layer-order sequence that is then re-encoded by snake filling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sprites = _detect_sprites(g)
    assert sprites, "expected at least one legal ring sprite"
    edges = _occlusion_graph(g, sprites)
    order = _topological_order(sprites, edges)
    return _snake_output(order)


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    vals = []
    prev = 0
    for cut in cuts + [total]:
        vals.append(cut - prev)
        prev = cut
    return vals


def _group_sizes(n: int) -> list[int]:
    min_groups = 2
    max_groups = min(3, n)
    choices = []
    for parts in range(min_groups, max_groups + 1):
        for cuts in itertools.combinations(range(1, n), parts - 1):
            vals = []
            prev = 0
            for cut in cuts + (n,):
                vals.append(cut - prev)
                prev = cut
            if max(vals) <= GEN_MAX_GROUP_SIZE:
                choices.append(vals)
    if not choices:
        raise AssertionError("no valid group composition")
    return list(random.choice(choices))


def _window_starts(total: int, count: int, span: int, gap: int) -> list[int]:
    need = count * span + (count - 1) * gap
    if need > total:
        raise AssertionError("layout does not fit")
    slack = total - need
    extras = _random_composition(slack + count + 1, count + 1)
    extras = [x - 1 for x in extras]
    starts = []
    cur = extras[0]
    for i in range(count):
        starts.append(cur)
        cur += span
        if i < count - 1:
            cur += gap + extras[i + 1]
    return starts


def _sequence_for_group(size: int, allowed_cols: list[int], require_nonmonotone: bool) -> list[int] | None:
    if size == 1:
        return [random.choice(allowed_cols)]
    candidates = []
    for seq in itertools.permutations(allowed_cols, size):
        if require_nonmonotone and list(seq) == sorted(seq):
            continue
        candidates.append(seq)
    if not candidates:
        return None
    return list(random.choice(candidates))


def _render_scene(shape: tuple[int, int], order: list[int], offsets: dict[int, tuple[int, int]]) -> np.ndarray:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    for color in reversed(order):
        r0, c0 = offsets[color]
        for dr, dc in RING_CELLS:
            rr, cc = r0 + dr, c0 + dc
            if _in_bounds((h, w), rr, cc):
                grid[rr, cc] = color
    return grid


def _transitive_closure(order: list[int], group_ids: dict[int, int]) -> set[tuple[int, int]]:
    closure = set()
    for i, above in enumerate(order):
        for below in order[i + 1 :]:
            if group_ids[above] == group_ids[below]:
                closure.add((above, below))
    return closure


def _semantic_validate(
    grid: np.ndarray,
    order: list[int],
    offsets: dict[int, tuple[int, int]],
    group_ids: dict[int, int],
    noise_cells: list[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    rank = {color: i for i, color in enumerate(order)}
    template_union: set[tuple[int, int]] = set()
    seen_cross_group_overlap = False
    actual_edges: set[tuple[int, int]] = set()

    per_color_sets = {color: _translated_ring(offsets[color]) for color in order}
    colors = set(order)
    for i, color_a in enumerate(order):
        for color_b in order[i + 1 :]:
            if group_ids[color_a] != group_ids[color_b]:
                inter = {
                    cell
                    for cell in per_color_sets[color_a] & per_color_sets[color_b]
                    if _in_bounds((h, w), cell[0], cell[1])
                }
                seen_cross_group_overlap |= bool(inter)

    if seen_cross_group_overlap:
        return False

    for color in order:
        visible = 0
        for r, c in per_color_sets[color]:
            if not _in_bounds((h, w), r, c):
                continue
            template_union.add((r, c))
            seen = int(grid[r, c])
            if seen == color:
                visible += 1
                continue
            if seen not in colors or rank[seen] >= rank[color]:
                return False
            actual_edges.add((seen, color))
        if visible < 4:
            return False

    if not actual_edges:
        return False

    topo_by_position = [color for color, _ in sorted(offsets.items(), key=lambda kv: kv[1])]
    if topo_by_position == order:
        return False

    if len(set(group_ids.values())) < 2:
        return False

    for r, c in noise_cells:
        if (r, c) in template_union:
            return False
        if not _in_bounds((h, w), r, c):
            return False
        if int(grid[r, c]) == 0 or int(grid[r, c]) in colors:
            return False

    sprites = _detect_sprites(grid)
    if set(sprites) != colors:
        return False
    if any(sprites[color]["offset"] != offsets[color] for color in colors):
        return False

    expected = _snake_output(order)
    try:
        solved = main(grid)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    return True


def _build_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_H_RANGE)
    w = random.randint(*GEN_W_RANGE)
    n = random.randint(*GEN_RING_COUNT_RANGE)
    order = random.sample(COLOR_POOL, n)
    sizes = _group_sizes(n)
    axis = random.choice(("row", "col"))
    clip = random.random() < GEN_CLIP_PROB

    offsets: dict[int, tuple[int, int]] = {}
    group_ids: dict[int, int] = {}
    force_inversion_done = False
    index = 0

    if axis == "row":
        starts = _window_starts(h, len(sizes), GEN_ROW_WINDOW_HEIGHT, GEN_GROUP_GAP)
        for gid, (group_size, row_start) in enumerate(zip(sizes, starts)):
            clip_mode = None
            if clip and gid == random.randrange(len(sizes)):
                clip_mode = random.choice(("left", "right"))
                clip = False
            if clip_mode == "left":
                allowed_cols = [-1, 0, 1, 2]
            elif clip_mode == "right":
                allowed_cols = [w - 6, w - 5, w - 4, w - 3]
            else:
                c0 = random.randint(0, w - 7)
                allowed_cols = [c0 + d for d in range(4)]

            require_nonmonotone = (group_size > 1) and (not force_inversion_done)
            seq = _sequence_for_group(group_size, allowed_cols, require_nonmonotone)
            if seq is None:
                return None
            if seq != sorted(seq):
                force_inversion_done = True

            for col in seq:
                color = order[index]
                offsets[color] = (row_start, col)
                group_ids[color] = gid
                index += 1
    else:
        starts = _window_starts(w, len(sizes), GEN_COL_WINDOW_WIDTH, GEN_GROUP_GAP)
        row = random.randint(0, h - 4)
        clip_gid = random.randrange(len(sizes)) if clip else None
        for gid, (group_size, col_start) in enumerate(zip(sizes, starts)):
            if gid == clip_gid and gid == 0 and random.random() < 0.5:
                allowed_cols = [-1, 0, 1, 2]
            elif gid == clip_gid and gid == len(sizes) - 1:
                allowed_cols = [w - 6, w - 5, w - 4, w - 3]
            else:
                allowed_cols = [col_start + d for d in range(4)]

            require_nonmonotone = (group_size > 1) and (not force_inversion_done)
            seq = _sequence_for_group(group_size, allowed_cols, require_nonmonotone)
            if seq is None:
                return None
            if seq != sorted(seq):
                force_inversion_done = True

            for col in seq:
                color = order[index]
                offsets[color] = (row, col)
                group_ids[color] = gid
                index += 1

    if not force_inversion_done:
        return None

    grid = _render_scene((h, w), order, offsets)

    used_colors = set(order)
    template_union = {
        cell
        for color in order
        for cell in _translated_ring(offsets[color])
        if _in_bounds((h, w), cell[0], cell[1])
    }
    noise_palette = [color for color in COLOR_POOL if color not in used_colors]
    max_noise = min(2, len(noise_palette))
    noise_count = random.randint(0, max_noise)
    noise_cells: list[tuple[int, int]] = []
    if noise_count:
        free_cells = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if (r, c) not in template_union and grid[r, c] == 0
        ]
        if len(free_cells) < noise_count:
            return None
        for color, (r, c) in zip(random.sample(noise_palette, noise_count), random.sample(free_cells, noise_count)):
            grid[r, c] = color
            noise_cells.append((r, c))

    if not _semantic_validate(grid, order, offsets, group_ids, noise_cells):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid occlusion-order snake input")
