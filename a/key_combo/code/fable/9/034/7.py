import random
from collections import Counter

import numpy as np


ORIENTATION_TO_DIR = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

SPRITE_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
)

GENERATION_RETRIES = 600


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 8:
        deltas = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    else:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in deltas:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _wall_geometry(
    r0: int,
    c0: int,
    height: int,
    width: int,
    orientation: str,
) -> dict | None:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    if height < 3 or width < 3:
        return None

    wall = set()
    if orientation == "up":
        if width % 2 == 0:
            return None
        for c in range(c0, c1 + 1):
            wall.add((r1, c))
        for r in range(r0, r1 + 1):
            wall.add((r, c0))
            wall.add((r, c1))
        start = (r0, c0 + width // 2)
    elif orientation == "down":
        if width % 2 == 0:
            return None
        for c in range(c0, c1 + 1):
            wall.add((r0, c))
        for r in range(r0, r1 + 1):
            wall.add((r, c0))
            wall.add((r, c1))
        start = (r1, c0 + width // 2)
    elif orientation == "left":
        if height % 2 == 0:
            return None
        for r in range(r0, r1 + 1):
            wall.add((r, c1))
        for c in range(c0, c1 + 1):
            wall.add((r0, c))
            wall.add((r1, c))
        start = (r0 + height // 2, c0)
    elif orientation == "right":
        if height % 2 == 0:
            return None
        for r in range(r0, r1 + 1):
            wall.add((r, c0))
        for c in range(c0, c1 + 1):
            wall.add((r0, c))
            wall.add((r1, c))
        start = (r0 + height // 2, c1)
    else:
        return None

    cavity = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
    ]
    return {
        "bbox": (r0, c0, r1, c1),
        "wall": sorted(wall),
        "cavity": cavity,
        "start": start,
        "direction": ORIENTATION_TO_DIR[orientation],
        "orientation": orientation,
    }


def _infer_wall(component: list[tuple[int, int]], wall_mask: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox(component)
    crop = wall_mask[r0:r1 + 1, c0:c1 + 1]
    h, w = crop.shape
    if h < 3 or w < 3:
        raise AssertionError("wall component is too small")
    if crop[1:-1, 1:-1].any():
        raise AssertionError("wall interior must be empty")

    open_top = bool(crop[0, 0] and crop[0, -1] and not crop[0, 1:-1].any())
    open_bottom = bool(crop[-1, 0] and crop[-1, -1] and not crop[-1, 1:-1].any())
    open_left = bool(crop[0, 0] and crop[-1, 0] and not crop[1:-1, 0].any())
    open_right = bool(crop[0, -1] and crop[-1, -1] and not crop[1:-1, -1].any())
    full_top = bool(crop[0, :].all())
    full_bottom = bool(crop[-1, :].all())
    full_left = bool(crop[:, 0].all())
    full_right = bool(crop[:, -1].all())

    candidates = []
    if open_top and full_bottom and full_left and full_right and w % 2 == 1:
        candidates.append("up")
    if open_bottom and full_top and full_left and full_right and w % 2 == 1:
        candidates.append("down")
    if open_left and full_top and full_bottom and full_right and h % 2 == 1:
        candidates.append("left")
    if open_right and full_top and full_bottom and full_left and h % 2 == 1:
        candidates.append("right")
    if len(candidates) != 1:
        raise AssertionError("wall orientation is ambiguous")

    wall = _wall_geometry(r0, c0, h, w, candidates[0])
    assert wall is not None
    if set(wall["wall"]) != set(component):
        raise AssertionError("wall shape does not match supported U template")
    return wall


def _first_hit(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
    wall_owner: np.ndarray,
    sprite_owner: np.ndarray,
) -> tuple[str, int] | None:
    h, w = shape
    dr, dc = direction
    r, c = start
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return None
        wid = int(wall_owner[r, c])
        if wid != -1:
            return ("wall", wid)
        sid = int(sprite_owner[r, c])
        if sid != -1:
            return ("sprite", sid)


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    if not colors:
        raise AssertionError("scene needs foreground objects")

    wall_color = max(colors, key=lambda color: int((g == color).sum()))
    wall_mask = g == wall_color
    wall_components = _components(wall_mask, connectivity=4)
    walls = [_infer_wall(comp, wall_mask) for comp in wall_components]
    walls.sort(key=lambda wall: wall["bbox"])

    wall_owner = np.full(g.shape, -1, dtype=int)
    for idx, wall in enumerate(walls):
        for r, c in wall["wall"]:
            wall_owner[r, c] = idx

    sprites = []
    sprite_owner = np.full(g.shape, -1, dtype=int)
    for color in colors:
        if color == wall_color:
            continue
        for comp in _components(g == color, connectivity=8):
            item = {
                "color": color,
                "coords": sorted(comp),
                "bbox": _bbox(comp),
            }
            sprites.append(item)
    sprites.sort(key=lambda sprite: sprite["bbox"])
    for idx, sprite in enumerate(sprites):
        for r, c in sprite["coords"]:
            sprite_owner[r, c] = idx

    targets: list[tuple[str, int]] = []
    for wall in walls:
        hit = _first_hit(g.shape, wall["start"], wall["direction"], wall_owner, sprite_owner)
        if hit is None:
            raise AssertionError("every wall must see a foreground object")
        targets.append(hit)
        wall["target"] = hit

    memo: dict[int, tuple[int, int]] = {}
    visiting: set[int] = set()

    def resolve(wid: int) -> tuple[int, int]:
        if wid in memo:
            return memo[wid]
        if wid in visiting:
            raise AssertionError("wall sight graph must be acyclic")
        visiting.add(wid)
        kind, obj_id = walls[wid]["target"]
        if kind == "sprite":
            result = (sprites[obj_id]["color"], obj_id)
        else:
            result = resolve(obj_id)
        visiting.remove(wid)
        memo[wid] = result
        return result

    resolved = [resolve(wid) for wid in range(len(walls))]
    return {
        "grid": g,
        "bg": bg,
        "wall_color": wall_color,
        "walls": walls,
        "sprites": sprites,
        "resolved": resolved,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each U-wall cavity by tracing its opening sightline through relay walls to a terminal sprite.

    Input structure:
    - Black background with several same-colored axis-aligned U-shaped wall components whose interiors are empty.
    - Other foreground components are small solid sprites, each in its own color.
    - Every wall opening has a single central sightline aimed either at a sprite or at another wall.

    Rule:
    - Identify each wall's opening direction from the missing side of its bounding box.
    - From the opening center, march outward along that direction and take the first non-background object hit.
    - If the hit object is a sprite, fill the wall's cavity with that sprite color and erase the whole sprite.
    - If the hit object is another wall, recursively inherit that wall's final fill color; keep all wall pixels unchanged and preserve untouched sprites.

    Key insight:
    - The first seen object is only the next relay node; color flows backward along an acyclic wall-to-wall sight chain until it reaches a terminal sprite, and only that tail sprite is consumed.
    """
    scene = _analyze_scene(grid)
    out = np.array(scene["grid"], copy=True)
    used_sprites: set[int] = set()

    for wall, (color, sprite_id) in zip(scene["walls"], scene["resolved"]):
        for r, c in wall["cavity"]:
            out[r, c] = color
        used_sprites.add(sprite_id)

    for sprite_id in used_sprites:
        for r, c in scene["sprites"][sprite_id]["coords"]:
            out[r, c] = scene["bg"]

    return out


def _expanded_clear(owner: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> bool:
    h, w = owner.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return bool(np.all(owner[rr0:rr1 + 1, cc0:cc1 + 1] == -1))


def _place_sprite(
    grid: np.ndarray,
    owner: np.ndarray,
    obj_id: int,
    color: int,
    rng: random.Random,
    *,
    spacious: bool,
    forbidden: set[tuple[int, int]] | None = None,
) -> dict | None:
    h, w = grid.shape
    shapes = list(SPRITE_SHAPES)
    rng.shuffle(shapes)
    attempts = 180 if spacious else 240
    margin = 5 if spacious else 1
    pad = 4 if spacious else 1
    forbidden = forbidden or set()

    for _ in range(attempts):
        shape = rng.choice(shapes)
        sr = rng.randint(margin, h - margin - 1)
        sc = rng.randint(margin, w - margin - 1)
        cells = [(sr + dr, sc + dc) for dr, dc in shape]
        if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
            continue
        bbox = _bbox(cells)
        if not _expanded_clear(owner, bbox, pad):
            continue
        if any((r, c) in forbidden for r, c in cells):
            continue
        for r, c in cells:
            grid[r, c] = color
            owner[r, c] = obj_id
        return {"kind": "sprite", "color": color, "coords": cells, "bbox": bbox}
    return None


def _wall_size_options(orientation: str) -> list[tuple[int, int]]:
    if orientation in ("up", "down"):
        return [(height, width) for height in range(4, 8) for width in (5, 7)]
    return [(height, width) for height in (5, 7) for width in range(4, 8)]


def _wall_from_start(
    start: tuple[int, int],
    orientation: str,
    height: int,
    width: int,
) -> dict | None:
    sr, sc = start
    if orientation == "up":
        r0 = sr
        c0 = sc - width // 2
    elif orientation == "down":
        r0 = sr - height + 1
        c0 = sc - width // 2
    elif orientation == "left":
        r0 = sr - height // 2
        c0 = sc
    elif orientation == "right":
        r0 = sr - height // 2
        c0 = sc - width + 1
    else:
        return None
    return _wall_geometry(r0, c0, height, width, orientation)


def _ray_hit_owner(
    owner: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[int | None, int]:
    h, w = owner.shape
    dr, dc = direction
    r, c = start
    steps = 0
    while True:
        r += dr
        c += dc
        steps += 1
        if not (0 <= r < h and 0 <= c < w):
            return None, steps
        hit = int(owner[r, c])
        if hit != -1:
            return hit, steps


def _find_wall_candidate(
    owner: np.ndarray,
    target: dict,
    rng: random.Random,
) -> dict | None:
    orientations = list(ORIENTATION_TO_DIR)
    rng.shuffle(orientations)
    target_cells = list(target["coords"])
    rng.shuffle(target_cells)

    for orientation in orientations:
        direction = ORIENTATION_TO_DIR[orientation]
        size_options = _wall_size_options(orientation)
        rng.shuffle(size_options)
        gaps = list(range(3, 9))
        rng.shuffle(gaps)
        for tr, tc in target_cells:
            for gap in gaps:
                sr = tr - direction[0] * gap
                sc = tc - direction[1] * gap
                for height, width in size_options:
                    wall = _wall_from_start((sr, sc), orientation, height, width)
                    if wall is None:
                        continue
                    r0, c0, r1, c1 = wall["bbox"]
                    h, w = owner.shape
                    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
                        continue
                    if not _expanded_clear(owner, wall["bbox"], 1):
                        continue
                    hit, steps = _ray_hit_owner(owner, wall["start"], wall["direction"])
                    if hit != target["id"] or steps < 2:
                        continue
                    return wall
    return None


def _place_wall(
    grid: np.ndarray,
    owner: np.ndarray,
    obj_id: int,
    wall_color: int,
    wall: dict,
) -> dict:
    for r, c in wall["wall"]:
        grid[r, c] = wall_color
        owner[r, c] = obj_id
    placed = dict(wall)
    placed["kind"] = "wall"
    placed["coords"] = list(placed["wall"])
    return placed


def _trace_wall_rays(objects: list[dict], owner: np.ndarray) -> set[tuple[int, int]]:
    forbidden: set[tuple[int, int]] = set()
    for obj in objects:
        if obj["kind"] != "wall":
            continue
        r, c = obj["start"]
        dr, dc = obj["direction"]
        h, w = owner.shape
        while True:
            if 0 <= r < h and 0 <= c < w:
                forbidden.add((r, c))
            r += dr
            c += dc
            if not (0 <= r < h and 0 <= c < w):
                break
            forbidden.add((r, c))
            if owner[r, c] != -1:
                break
    return forbidden


def _chain_depths(scene: dict) -> list[int]:
    walls = scene["walls"]
    memo: dict[int, int] = {}

    def depth(wid: int) -> int:
        if wid in memo:
            return memo[wid]
        kind, obj_id = walls[wid]["target"]
        if kind == "sprite":
            memo[wid] = 1
        else:
            memo[wid] = 1 + depth(obj_id)
        return memo[wid]

    return [depth(i) for i in range(len(walls))]


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        scene = _analyze_scene(grid)
    except Exception:
        return False

    walls = scene["walls"]
    sprites = scene["sprites"]
    if len(walls) < 3:
        return False
    if len(sprites) < 3:
        return False

    direct = sum(1 for wall in walls if wall["target"][0] == "sprite")
    relay = sum(1 for wall in walls if wall["target"][0] == "wall")
    if direct < 2 or relay < 1:
        return False

    if len({wall["orientation"] for wall in walls}) < 2:
        return False

    depths = _chain_depths(scene)
    if max(depths) < 2:
        return False

    resolved_colors = [color for color, _ in scene["resolved"]]
    if len(set(resolved_colors)) < 2:
        return False

    used_sprites = {sprite_id for _, sprite_id in scene["resolved"]}
    if len(sprites) - len(used_sprites) < 1:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False

    bg = scene["bg"]
    for wall, (color, _) in zip(walls, scene["resolved"]):
        if not wall["cavity"]:
            return False
        if any(int(out[r, c]) != color for r, c in wall["cavity"]):
            return False

    for sid, sprite in enumerate(sprites):
        vals = [int(out[r, c]) for r, c in sprite["coords"]]
        if sid in used_sprites:
            if any(v != bg for v in vals):
                return False
        else:
            original = [int(grid[r, c]) for r, c in sprite["coords"]]
            if vals != original:
                return False

    return True


def _generate_once(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(22, 29)
    w = rng.randint(22, 29)
    grid = np.zeros((h, w), dtype=int)
    owner = np.full((h, w), -1, dtype=int)

    palette = list(range(1, 10))
    rng.shuffle(palette)
    wall_color = palette[0]
    sprite_colors = palette[1:]

    n_walls = rng.randint(3, 5)
    n_unused = 1 if rng.random() < 0.7 else 2
    needed_sprites = 2 + n_unused
    if len(sprite_colors) < needed_sprites:
        return None

    objects: list[dict] = []

    for color in sprite_colors[:2]:
        obj_id = len(objects)
        sprite = _place_sprite(grid, owner, obj_id, color, rng, spacious=True)
        if sprite is None:
            return None
        sprite["id"] = obj_id
        objects.append(sprite)

    direct_target_ids = [0, 1]
    for target_id in direct_target_ids:
        target = objects[target_id]
        obj_id = len(objects)
        wall = _find_wall_candidate(owner, target, rng)
        if wall is None:
            return None
        placed = _place_wall(grid, owner, obj_id, wall_color, wall)
        placed["id"] = obj_id
        objects.append(placed)

    relay_choices = [obj for obj in objects if obj["kind"] == "wall"]
    relay_target = rng.choice(relay_choices)
    obj_id = len(objects)
    wall = _find_wall_candidate(owner, relay_target, rng)
    if wall is None:
        return None
    placed = _place_wall(grid, owner, obj_id, wall_color, wall)
    placed["id"] = obj_id
    objects.append(placed)

    while len([obj for obj in objects if obj["kind"] == "wall"]) < n_walls:
        candidates = [obj for obj in objects if obj["kind"] == "wall"]
        if rng.random() < 0.35:
            candidates.extend(obj for obj in objects[:2] if obj["kind"] == "sprite")
        target = rng.choice(candidates)
        obj_id = len(objects)
        wall = _find_wall_candidate(owner, target, rng)
        if wall is None:
            return None
        placed = _place_wall(grid, owner, obj_id, wall_color, wall)
        placed["id"] = obj_id
        objects.append(placed)

    forbidden = _trace_wall_rays(objects, owner)
    for color in sprite_colors[2:2 + n_unused]:
        obj_id = len(objects)
        sprite = _place_sprite(
            grid,
            owner,
            obj_id,
            color,
            rng,
            spacious=False,
            forbidden=forbidden,
        )
        if sprite is None:
            return None
        sprite["id"] = obj_id
        objects.append(sprite)

    counts = Counter(int(v) for v in grid.ravel())
    if counts[0] <= max(v for k, v in counts.items() if k != 0):
        return None

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    seed = random.randrange(1 << 30)
    for offset in range(GENERATION_RETRIES):
        rng = random.Random(seed + offset)
        grid = _generate_once(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
