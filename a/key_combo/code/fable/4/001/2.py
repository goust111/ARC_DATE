from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
TRAIL = 5
SPRITE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS:
        yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[list[tuple[int, int]]] = []
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
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        out.append(comp)
    return out


def _order_path(path_set: set[tuple[int, int]], start: tuple[int, int], end: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    ordered: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        ordered.append(cur)
        if cur == end:
            break
        nxt = [p for p in _neighbors4(*cur) if p in path_set and p != prev]
        assert len(nxt) == 1, "tail must be a single unbranched path"
        prev, cur = cur, nxt[0]
    assert len(ordered) == len(path_set), "tail path must visit every grey cell once"
    return tuple(ordered)


def _parse_items(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be two-dimensional"
    assert np.all((0 <= g) & (g <= 9)), "ARC colors must be in 0..9"

    sprites: list[dict] = []
    for color in sorted(int(x) for x in np.unique(g) if int(x) not in (BG, TRAIL)):
        for comp in _components(g == color):
            cells = tuple(sorted(comp))
            sprites.append({"color": color, "coords": cells, "set": set(cells)})

    trails = _components(g == TRAIL)
    assert sprites and len(trails) == len(sprites), "each sprite must have exactly one grey tail"

    sprite_at: dict[tuple[int, int], int] = {}
    for idx, sprite in enumerate(sprites):
        for cell in sprite["coords"]:
            assert cell not in sprite_at, "sprite cells may not overlap"
            sprite_at[cell] = idx

    for cell, idx in sprite_at.items():
        for nb in _neighbors4(*cell):
            other = sprite_at.get(nb)
            assert other is None or other == idx, "different sprites must not touch"

    items: list[dict] = []
    used_sprites: set[int] = set()
    for trail in trails:
        path_set = set(trail)
        assert len(path_set) >= 2, "tail needs at least two cells to define inertia"
        degrees = {p: sum((nb in path_set) for nb in _neighbors4(*p)) for p in path_set}
        endpoints = [p for p, deg in degrees.items() if deg == 1]
        assert len(endpoints) == 2 and all(deg in (1, 2) for deg in degrees.values()), "tail must be a simple path"

        touches: list[tuple[tuple[int, int], int, tuple[int, int]]] = []
        for p in path_set:
            for nb in _neighbors4(*p):
                sprite_idx = sprite_at.get(nb)
                if sprite_idx is not None:
                    touches.append((p, sprite_idx, nb))
        assert len(touches) == 1, "tail must touch exactly one sprite cell at one end"

        near, sprite_idx, contact = touches[0]
        assert near in endpoints, "sprite must attach to a tail endpoint"
        assert sprite_idx not in used_sprites, "a sprite may only be assigned one tail"
        used_sprites.add(sprite_idx)
        far = endpoints[0] if endpoints[1] == near else endpoints[1]
        path = _order_path(path_set, near, far)
        last_dir = (path[-1][0] - path[-2][0], path[-1][1] - path[-2][1])
        assert last_dir in DIRS, "last tail segment must be cardinal"

        dirs = [(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) for i in range(len(path) - 1)]
        items.append(
            {
                "color": sprites[sprite_idx]["color"],
                "coords": sprites[sprite_idx]["coords"],
                "contact": contact,
                "path": path,
                "length": len(path),
                "last_dir": last_dir,
                "bent": any(dirs[i] != dirs[i - 1] for i in range(1, len(dirs))),
                "first_dir": (path[0][0] - contact[0], path[0][1] - contact[1]),
            }
        )

    assert len(used_sprites) == len(sprites), "every sprite must be matched to a tail"
    lengths = [item["length"] for item in items]
    assert len(set(lengths)) == len(lengths), "tail lengths must be unique"
    return items


def _solve(grid: np.ndarray, *, descending: bool = True, collect: bool = False):
    g = np.asarray(grid, dtype=int)
    items = _parse_items(g)
    out = np.full(g.shape, BG, dtype=int)
    occupied: set[tuple[int, int]] = set()
    meta: list[dict] = []

    for item in sorted(items, key=lambda x: x["length"], reverse=descending):
        contact = item["contact"]
        sprite = item["coords"]

        for target in item["path"]:
            dr = target[0] - contact[0]
            dc = target[1] - contact[1]
            pose = {(r + dr, c + dc) for r, c in sprite}
            assert all(_in_bounds(g.shape, p) for p in pose), "sprite must fit while walking its tail"

        far = item["path"][-1]
        dr = far[0] - contact[0]
        dc = far[1] - contact[1]
        cur = {(r + dr, c + dc) for r, c in sprite}
        assert not (cur & occupied), "sprite may not overlap a settled sprite at the tail end"

        slide_dir = item["last_dir"]
        slide_steps = 0
        while True:
            nxt = {(r + slide_dir[0], c + slide_dir[1]) for r, c in cur}
            hits_boundary = not all(_in_bounds(g.shape, p) for p in nxt)
            hits_sprite = bool(nxt & occupied)
            if hits_boundary or hits_sprite:
                stop = "collision" if hits_sprite else "boundary"
                break
            cur = nxt
            slide_steps += 1

        for r, c in cur:
            out[r, c] = int(item["color"])
        occupied.update(cur)
        meta.append(
            {
                "color": item["color"],
                "length": item["length"],
                "stop": stop,
                "slide_steps": slide_steps,
                "bent": item["bent"],
                "first_dir": item["first_dir"],
                "last_dir": item["last_dir"],
            }
        )

    return (out, meta) if collect else out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each sprite along its grey trail, then slide it by the trail's final direction.

    Input structure:
    - The background is black. Grey 4-connected one-cell-wide paths are trails.
    - Every non-black, non-grey monochrome connected component is a small sprite.
    - Each trail is a simple path whose near endpoint touches exactly one sprite
      cell, and all trail lengths are distinct.

    Rule:
    - Sort sprites by their own trail length from longest to shortest.
    - For each sprite, translate the whole sprite so its touching cell walks
      along every grey trail cell to the far endpoint.
    - Continue translating in the direction of the final trail segment until the
      next step would leave the grid or overlap an already settled sprite.
    - Remove all grey trails and keep each sprite's original shape and color.

    Key insight:
    - The grey trail simultaneously gives the route prefix, the final inertia
      direction, and the priority order; collisions are resolved only against
      sprites that have already stopped.
    """
    return _solve(grid)


# ---------------- generator ----------------

BASE_SHAPES_RIGHT = (
    ((0, 0),),
    ((-1, 0), (0, 0), (1, 0)),
    ((0, 0), (1, 0), (1, -1)),
    ((0, -1), (0, 0), (1, -1), (1, 0)),
    ((-1, 0), (0, -1), (0, 0), (1, 0)),
)


def _neg(d: tuple[int, int]) -> tuple[int, int]:
    return -d[0], -d[1]


def _perps(d: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    dr, dc = d
    return (dc, -dr), (-dc, dr)


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def _sub(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] - b[0], a[1] - b[1]


def _shape_offsets(direction: tuple[int, int], shape_id: int) -> tuple[tuple[int, int], ...]:
    offsets = BASE_SHAPES_RIGHT[shape_id]
    if direction == (0, 1):
        return offsets
    if direction == (0, -1):
        return tuple((r, -c) for r, c in offsets)
    if direction == (1, 0):
        return tuple((c, -r) for r, c in offsets)
    if direction == (-1, 0):
        return tuple((-c, r) for r, c in offsets)
    raise ValueError("invalid direction")


def _has_existing_neighbor(cells: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    return any(nb in occupied for cell in cells for nb in _neighbors4(*cell))


def _composition(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(1, total), parts - 1)) if parts > 1 else []
    vals = []
    prev = 0
    for cut in cuts + [total]:
        vals.append(cut - prev)
        prev = cut
    return vals


def _try_path_to_far(
    shape: tuple[int, int],
    far: tuple[int, int],
    last_dir: tuple[int, int],
    length: int,
    shape_id: int,
    occupied: set[tuple[int, int]],
    *,
    force_bend: bool = False,
):
    steps = length - 1
    if steps < 1:
        return None

    for _ in range(120):
        max_segments = min(3, steps)
        min_segments = 2 if force_bend and max_segments >= 2 else 1
        segment_count = random.randint(min_segments, max_segments)
        lens = _composition(steps, segment_count)
        reverse_dirs = [_neg(last_dir)]
        for _seg in range(1, segment_count):
            reverse_dirs.append(random.choice(_perps(reverse_dirs[-1])))

        rev = [far]
        used = {far}
        cur = far
        ok = True
        for direction, seg_len in zip(reverse_dirs, lens):
            for _step in range(seg_len):
                nxt = _add(cur, direction)
                if not _in_bounds(shape, nxt) or nxt in used or nxt in occupied:
                    ok = False
                    break
                for nb in _neighbors4(*nxt):
                    if nb in used and nb != cur:
                        ok = False
                        break
                if not ok:
                    break
                rev.append(nxt)
                used.add(nxt)
                cur = nxt
            if not ok:
                break
        if not ok:
            continue

        path = tuple(reversed(rev))
        path_set = set(path)
        if _has_existing_neighbor(path_set, occupied):
            continue

        first_step = _sub(path[1], path[0])
        if force_bend and first_step == last_dir:
            continue
        contact = _sub(path[0], first_step)
        if not _in_bounds(shape, contact) or contact in path_set or contact in occupied:
            continue

        sprite = {(contact[0] + r, contact[1] + c) for r, c in _shape_offsets(first_step, shape_id)}
        if len(sprite) != len(BASE_SHAPES_RIGHT[shape_id]):
            continue
        if not all(_in_bounds(shape, cell) for cell in sprite):
            continue
        if sprite & occupied or sprite & path_set or _has_existing_neighbor(sprite, occupied):
            continue

        touches = [(s, nb) for s in sprite for nb in _neighbors4(*s) if nb in path_set]
        if touches != [(contact, path[0])]:
            continue

        return path, contact, tuple(sorted(sprite))
    return None


def _draw_spec(grid: np.ndarray, spec: dict) -> None:
    for r, c in spec["path"]:
        grid[r, c] = TRAIL
    for r, c in spec["sprite"]:
        grid[r, c] = int(spec["color"])


def _choose_spaced(values: list[int], count: int, min_gap: int) -> list[int] | None:
    random.shuffle(values)
    for _ in range(80):
        chosen: list[int] = []
        for value in values:
            if all(abs(value - old) >= min_gap for old in chosen):
                chosen.append(value)
                if len(chosen) == count:
                    return chosen
    return None


def _lane_far_points(h: int, w: int, direction: tuple[int, int], count: int) -> list[tuple[int, int]] | None:
    if direction in ((0, 1), (0, -1)):
        row = random.randint(4, h - 5)
        if direction == (0, 1):
            vals = list(range(4, w - count - 2))
        else:
            vals = list(range(count + 2, w - 4))
        cols = _choose_spaced(vals, count, 3)
        if cols is None:
            return None
        return [(row, c) for c in cols]

    col = random.randint(4, w - 5)
    if direction == (1, 0):
        vals = list(range(4, h - count - 2))
    else:
        vals = list(range(count + 2, h - 4))
    rows = _choose_spaced(vals, count, 3)
    if rows is None:
        return None
    return [(r, col) for r in rows]


def _random_far(shape: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    if direction == (0, 1):
        return random.randint(3, h - 4), random.randint(3, w - 5)
    if direction == (0, -1):
        return random.randint(3, h - 4), random.randint(4, w - 4)
    if direction == (1, 0):
        return random.randint(3, h - 5), random.randint(3, w - 4)
    return random.randint(4, h - 4), random.randint(3, w - 4)


def _make_spec(
    shape: tuple[int, int],
    far: tuple[int, int],
    last_dir: tuple[int, int],
    length: int,
    color: int,
    shape_id: int,
    occupied: set[tuple[int, int]],
    *,
    force_bend: bool = False,
) -> dict | None:
    built = _try_path_to_far(shape, far, last_dir, length, shape_id, occupied, force_bend=force_bend)
    if built is None:
        return None
    path, contact, sprite = built
    return {"path": path, "contact": contact, "sprite": sprite, "color": color}


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        out, meta = _solve(grid, collect=True)
        asc = _solve(grid, descending=False)
    except AssertionError:
        return False

    if np.array_equal(out, asc):
        return False
    if not any(m["stop"] == "collision" for m in meta):
        return False
    if not any(m["stop"] == "boundary" for m in meta):
        return False
    if not any(m["bent"] and m["first_dir"] != m["last_dir"] for m in meta):
        return False
    if np.count_nonzero(out != BG) < len(meta):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 30)
    w = random.randint(16, 30)
    shape = (h, w)
    count = random.choice([2, 3, 3, 4])
    stack_count = 2 if count < 4 else random.choice([2, 3])
    stack_dir = random.choice(DIRS)
    stack_fars = _lane_far_points(h, w, stack_dir, stack_count)
    if stack_fars is None:
        return None
    random.shuffle(stack_fars)

    lengths = random.sample(range(5, 14), count)
    colors = random.sample(SPRITE_COLORS, count)
    grid = np.full(shape, BG, dtype=int)
    occupied: set[tuple[int, int]] = set()
    specs: list[dict] = []

    for i, far in enumerate(stack_fars):
        spec = _make_spec(
            shape,
            far,
            stack_dir,
            lengths[i],
            colors[i],
            0,
            occupied,
            force_bend=(i == 0),
        )
        if spec is None:
            return None
        specs.append(spec)
        occupied.update(spec["path"])
        occupied.update(spec["sprite"])

    for i in range(stack_count, count):
        spec = None
        for _ in range(100):
            direction = random.choice(DIRS)
            far = _random_far(shape, direction)
            shape_id = random.randint(1, len(BASE_SHAPES_RIGHT) - 1)
            spec = _make_spec(shape, far, direction, lengths[i], colors[i], shape_id, occupied)
            if spec is not None:
                break
        if spec is None:
            return None
        specs.append(spec)
        occupied.update(spec["path"])
        occupied.update(spec["sprite"])

    for spec in specs:
        _draw_spec(grid, spec)

    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tail-route inertia puzzle input")
