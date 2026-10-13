import random
from collections import deque

import numpy as np

BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
MARKER = 9
SPRITE_COLORS = (1, 2, 3, 4, 8)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
MAX_GEN_ATTEMPTS = 400


def _mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


SHAPE_LIBRARY = [
    _mask(["111"]),
    _mask(["10", "11"]),
    _mask(["11", "11"]),
    _mask(["1111"]),
    _mask(["100", "111"]),
    _mask(["111", "010"]),
    _mask(["110", "011"]),
    _mask(["010", "111", "010"]),
    _mask(["100", "100", "111"]),
    _mask(["111", "110"]),
    _mask(["111", "101"]),
    _mask(["110", "011", "001"]),
]


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    dirs = DIRS4 if connectivity == 4 else DIRS8
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0])
    out[rel[:, 0], rel[:, 1]] = True
    return out


def _transform_mask(mask: np.ndarray, rot: int = 0, flip: bool = False) -> np.ndarray:
    m = np.fliplr(mask) if flip else mask
    return np.rot90(m, rot)


def _all_transforms(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for flip in (False, True):
        base = np.fliplr(mask) if flip else mask
        for rot in range(4):
            out.append(np.rot90(base, rot))
    return out


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    keys = []
    for transformed in _all_transforms(mask):
        trimmed = transformed.astype(bool)
        keys.append((trimmed.shape, trimmed.tobytes()))
    return min(keys)


def _rect_border_mask(h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[0, :] = True
    m[-1, :] = True
    m[:, 0] = True
    m[:, -1] = True
    return m


def _shift_box(box: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return (r0 + dr, c0 + dc, r1 + dr, c1 + dc)


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _direction_from_side(side: str) -> tuple[int, int]:
    return {
        "left": (0, 1),
        "right": (0, -1),
        "top": (1, 0),
        "bottom": (-1, 0),
    }[side]


def _head_for_box(box: tuple[int, int, int, int], side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if side == "left":
        return ((r0 + r1) // 2, c0 - 1)
    if side == "right":
        return ((r0 + r1) // 2, c1 + 1)
    if side == "top":
        return (r0 - 1, (c0 + c1) // 2)
    return (r1 + 1, (c0 + c1) // 2)


def _target_for_box(box: tuple[int, int, int, int], side: str, gap: int) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if side == "left":
        return ((r0 + r1) // 2, c1 + gap + 1)
    if side == "right":
        return ((r0 + r1) // 2, c0 - gap - 1)
    if side == "top":
        return (r1 + gap + 1, (c0 + c1) // 2)
    return (r0 - gap - 1, (c0 + c1) // 2)


def _scan_target(grid: np.ndarray, head: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    h, w = grid.shape
    r, c = head
    dr, dc = direction
    while True:
        r += dr
        c += dc
        assert 0 <= r < h and 0 <= c < w, "missing target marker on movement axis"
        if int(grid[r, c]) == MARKER:
            return (r, c)


def _render_container_patch(container: dict, fill_holes: bool, charge_steps: int = 0) -> tuple[np.ndarray, int]:
    oh, ow = container["outer"]
    ih, iw = oh - 2, ow - 2
    patch = np.full((oh, ow), BG, dtype=int)
    patch[0, :] = BORDER
    patch[-1, :] = BORDER
    patch[:, 0] = BORDER
    patch[:, -1] = BORDER
    patch[1:-1, 1:-1] = GREY

    matched = 0
    for hole in container["holes"]:
        hr, hc = hole["pos"]
        rr, cc = np.nonzero(hole["mask"])
        hole_color = hole.get("color")
        color = hole_color if fill_holes and hole_color is not None else BG
        if hole_color is not None:
            matched += 1
        patch[1 + hr + rr, 1 + hc + cc] = color

    steps = min(charge_steps, iw if container["dir"][1] else ih)
    if steps > 0:
        if container["dir"] == (0, 1):
            cols = [iw - 1 - k for k in range(steps)]
            for col in cols:
                cells = patch[1:-1, 1 + col]
                cells[cells == GREY] = ORANGE
        elif container["dir"] == (0, -1):
            cols = [k for k in range(steps)]
            for col in cols:
                cells = patch[1:-1, 1 + col]
                cells[cells == GREY] = ORANGE
        elif container["dir"] == (1, 0):
            rows = [ih - 1 - k for k in range(steps)]
            for row in rows:
                cells = patch[1 + row, 1:-1]
                cells[cells == GREY] = ORANGE
        else:
            rows = [k for k in range(steps)]
            for row in rows:
                cells = patch[1 + row, 1:-1]
                cells[cells == GREY] = ORANGE
    return patch, matched


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill container holes by shape-matching outside sprites, then move each container by its match count.

    Input structure:
    - The background is magenta (6).
    - There are 2-3 black rectangular containers with grey interiors and magenta hole components inside.
    - Each container has one maroon head pixel touching one outside side, plus another maroon target pixel farther ahead on the movement axis.
    - Outside all containers, small monochrome 4-connected sprites encode colors for hole shapes.

    Rule:
    - Detect each container, its internal magenta hole components, and the outside sprites.
    - Match every hole against an outside sprite by 4-connected shape, allowing 0/90/180/270 rotations and horizontal flips.
    - Fill matched holes with the matched sprite color; unmatched holes stay magenta; erase all outside sprites.
    - Let E be the number of matched holes and D the straight-line distance from head to target; move the container together with its head by min(E, D) cells along the head-indicated direction.
    - During that movement, convert grey interior cells to orange from the movement-front side inward, one interior row or column per step.

    Key insight:
    - Hole-shape matching has two simultaneous effects: it colors the holes and it also determines the container's movement energy.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape

    containers = []
    for coords in _components(g == BORDER, connectivity=4):
        r0, c0, r1, c1 = _bbox(coords)
        box_h, box_w = r1 - r0 + 1, c1 - c0 + 1
        if box_h < 3 or box_w < 3:
            continue
        mask = np.zeros((box_h, box_w), dtype=bool)
        rel = coords - np.array([r0, c0])
        mask[rel[:, 0], rel[:, 1]] = True
        if not np.array_equal(mask, _rect_border_mask(box_h, box_w)):
            continue
        containers.append({"box": (r0, c0, r1, c1), "outer": (box_h, box_w)})

    assert containers, "no valid containers found"

    marker_mask = g == MARKER
    sprite_block = np.zeros_like(g, dtype=bool)
    for container in containers:
        r0, c0, r1, c1 = container["box"]
        side_hits = []
        if c0 - 1 >= 0:
            hits = [(r, c0 - 1) for r in range(r0, r1 + 1) if marker_mask[r, c0 - 1]]
            if hits:
                side_hits.append(("left", hits))
        if c1 + 1 < w:
            hits = [(r, c1 + 1) for r in range(r0, r1 + 1) if marker_mask[r, c1 + 1]]
            if hits:
                side_hits.append(("right", hits))
        if r0 - 1 >= 0:
            hits = [(r0 - 1, c) for c in range(c0, c1 + 1) if marker_mask[r0 - 1, c]]
            if hits:
                side_hits.append(("top", hits))
        if r1 + 1 < h:
            hits = [(r1 + 1, c) for c in range(c0, c1 + 1) if marker_mask[r1 + 1, c]]
            if hits:
                side_hits.append(("bottom", hits))

        assert len(side_hits) == 1 and len(side_hits[0][1]) == 1, "head marker must be unique"
        side = side_hits[0][0]
        head = side_hits[0][1][0]
        direction = _direction_from_side(side)
        target = _scan_target(g, head, direction)
        container["side"] = side
        container["head"] = head
        container["dir"] = direction
        container["target"] = target
        sprite_block[r0 : r1 + 1, c0 : c1 + 1] = True

    sprite_block |= marker_mask
    sprite_map: dict[tuple[tuple[int, int], bytes], int] = {}
    outside_mask = (g != BG) & (~sprite_block)
    for coords in _components(outside_mask, connectivity=4):
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1, "sprites must be monochrome"
        color = next(iter(colors))
        key = _canonical_shape_key(_coords_to_mask(coords))
        assert key not in sprite_map, "sprite shapes must be unique"
        sprite_map[key] = color

    out = np.full_like(g, BG)
    for container in containers:
        r0, c0, r1, c1 = container["box"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        holes = []
        for coords in _components(inner == BG, connectivity=4):
            hole_mask = _coords_to_mask(coords)
            holes.append(
                {
                    "pos": (int(coords[:, 0].min()), int(coords[:, 1].min())),
                    "mask": hole_mask,
                    "color": sprite_map.get(_canonical_shape_key(hole_mask)),
                }
            )
        container["holes"] = holes
        patch, energy = _render_container_patch(container, fill_holes=True, charge_steps=0)
        head = container["head"]
        target = container["target"]
        direction = container["dir"]
        distance = abs(target[0] - head[0]) + abs(target[1] - head[1])
        steps = min(energy, distance)
        patch, _ = _render_container_patch(container, fill_holes=True, charge_steps=steps)
        dr, dc = direction
        nr0, nc0 = r0 + dr * steps, c0 + dc * steps
        _paste(out, patch, nr0, nc0)
        out[target] = MARKER
        out[head[0] + dr * steps, head[1] + dc * steps] = MARKER

    return out


def _nonoverlap_bbox(
    occupied: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    pad: int = 1,
) -> bool:
    h, w = occupied.shape
    pr0 = max(0, r0 - pad)
    pc0 = max(0, c0 - pad)
    pr1 = min(h - 1, r1 + pad)
    pc1 = min(w - 1, c1 + pad)
    return not bool(occupied[pr0 : pr1 + 1, pc0 : pc1 + 1].any())


def _mark_bbox(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    h, w = occupied.shape
    pr0 = max(0, r0 - pad)
    pc0 = max(0, c0 - pad)
    pr1 = min(h - 1, r1 + pad)
    pc1 = min(w - 1, c1 + pad)
    occupied[pr0 : pr1 + 1, pc0 : pc1 + 1] = True


def _sample_holes(rng: random.Random, ih: int, iw: int, shape_ids: list[int]) -> list[dict] | None:
    occupied: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()
    holes = []
    for shape_id in shape_ids:
        base = SHAPE_LIBRARY[shape_id]
        placed = False
        transforms = _all_transforms(base)
        rng.shuffle(transforms)
        for transformed in transforms:
            mh, mw = transformed.shape
            row_choices = list(range(1, ih - mh))
            col_choices = list(range(1, iw - mw))
            rng.shuffle(row_choices)
            rng.shuffle(col_choices)
            for r0 in row_choices:
                for c0 in col_choices:
                    cells = {(r0 + rr, c0 + cc) for rr, cc in zip(*np.nonzero(transformed))}
                    if any(cell in blocked for cell in cells):
                        continue
                    holes.append({"pos": (r0, c0), "mask": transformed.copy(), "shape_id": shape_id})
                    occupied |= cells
                    for rr, cc in cells:
                        for dr, dc in DIRS4 + [(0, 0)]:
                            nr, nc = rr + dr, cc + dc
                            if 0 <= nr < ih and 0 <= nc < iw:
                                blocked.add((nr, nc))
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None
    return holes


def _build_container_specs(rng: random.Random, count: int) -> list[dict] | None:
    all_shape_ids = list(range(len(SHAPE_LIBRARY)))
    rng.shuffle(all_shape_ids)
    specs = []
    used = 0
    directions = ["left", "right", "top", "bottom"]
    rng.shuffle(directions)
    chosen_dirs = directions[: max(2, count)]
    while len(chosen_dirs) < count:
        chosen_dirs.append(rng.choice(directions))
    for idx in range(count):
        side = chosen_dirs[idx]
        direction = _direction_from_side(side)
        ih = rng.randint(7, 9)
        iw = rng.randint(7, 9)
        outer = (ih + 2, iw + 2)
        if idx == 0:
            hole_count = 2 if rng.random() < 0.7 else 3
        else:
            hole_count = rng.choices([1, 2, 3], weights=[3, 4, 1])[0]
        if used + hole_count > len(all_shape_ids):
            return None
        hole_shape_ids = all_shape_ids[used : used + hole_count]
        used += hole_count
        holes = _sample_holes(rng, ih, iw, hole_shape_ids)
        if holes is None:
            return None
        if idx == 0:
            matched_count = rng.randint(1, hole_count - 1)
        else:
            matched_count = rng.randint(1, hole_count)
        steps = matched_count
        gap = steps + rng.randint(1, 4)
        spec = {
            "side": side,
            "dir": direction,
            "inner": (ih, iw),
            "outer": outer,
            "holes": holes,
            "matched_count": matched_count,
            "gap": gap,
        }
        specs.append(spec)
    return specs


def _place_containers(rng: random.Random, h: int, w: int, specs: list[dict]) -> list[dict] | None:
    occupied = np.zeros((h, w), dtype=bool)
    placed: list[dict] = []
    for spec in specs:
        oh, ow = spec["outer"]
        side = spec["side"]
        gap = spec["gap"]
        steps = spec["matched_count"]
        for _ in range(120):
            if side == "left":
                r0 = rng.randint(1, h - oh - 1)
                c0 = rng.randint(1, w - ow - gap - 2)
                box = (r0, c0, r0 + oh - 1, c0 + ow - 1)
                corridor = (r0, c0 - 1, r0 + oh - 1, c0 + ow + gap)
            elif side == "right":
                r0 = rng.randint(1, h - oh - 1)
                c0 = rng.randint(gap + 2, w - ow - 1)
                box = (r0, c0, r0 + oh - 1, c0 + ow - 1)
                corridor = (r0, c0 - gap - 1, r0 + oh - 1, c0 + ow)
            elif side == "top":
                r0 = rng.randint(1, h - oh - gap - 2)
                c0 = rng.randint(1, w - ow - 1)
                box = (r0, c0, r0 + oh - 1, c0 + ow - 1)
                corridor = (r0 - 1, c0, r0 + oh + gap, c0 + ow - 1)
            else:
                r0 = rng.randint(gap + 2, h - oh - 1)
                c0 = rng.randint(1, w - ow - 1)
                box = (r0, c0, r0 + oh - 1, c0 + ow - 1)
                corridor = (r0 - gap - 1, c0, r0 + oh, c0 + ow - 1)
            head = _head_for_box(box, side)
            target = _target_for_box(box, side, gap)
            final_box = _shift_box(box, spec["dir"][0] * steps, spec["dir"][1] * steps)
            if not (0 <= target[0] < h and 0 <= target[1] < w):
                continue
            if final_box[0] < 0 or final_box[1] < 0 or final_box[2] >= h or final_box[3] >= w:
                continue
            if not _nonoverlap_bbox(occupied, *corridor, pad=0):
                continue
            _mark_bbox(occupied, *corridor, pad=0)
            placed.append(
                {
                    **spec,
                    "box": box,
                    "head": head,
                    "target": target,
                }
            )
            break
        else:
            return None
    return placed


def _render_input_grid(h: int, w: int, containers: list[dict], sprites: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for container in containers:
        r0, c0, _r1, _c1 = container["box"]
        patch, _ = _render_container_patch(container, fill_holes=False, charge_steps=0)
        _paste(grid, patch, r0, c0)
        grid[container["head"]] = MARKER
        grid[container["target"]] = MARKER
    for sprite in sprites:
        r0, c0 = sprite["pos"]
        rr, cc = np.nonzero(sprite["mask"])
        grid[r0 + rr, c0 + cc] = sprite["color"]
    return grid


def _render_expected_output(h: int, w: int, containers: list[dict]) -> np.ndarray:
    out = np.full((h, w), BG, dtype=int)
    for container in containers:
        steps = container["matched_count"]
        patch, _ = _render_container_patch(container, fill_holes=True, charge_steps=steps)
        dr, dc = container["dir"]
        r0, c0, _r1, _c1 = container["box"]
        _paste(out, patch, r0 + dr * steps, c0 + dc * steps)
        out[container["target"]] = MARKER
        out[container["head"][0] + dr * steps, container["head"][1] + dc * steps] = MARKER
    return out


def _place_sprites(rng: random.Random, base_grid: np.ndarray, containers: list[dict]) -> list[dict] | None:
    h, w = base_grid.shape
    occupied = base_grid != BG
    hole_keys = {_canonical_shape_key(hole["mask"]) for container in containers for hole in container["holes"]}
    sprites = []

    matched_holes = []
    unmatched_holes = []
    for container in containers:
        for idx, hole in enumerate(container["holes"]):
            if idx < container["matched_count"]:
                matched_holes.append(hole)
            else:
                unmatched_holes.append(hole)

    color_pool = [rng.choice(SPRITE_COLORS) for _ in range(len(matched_holes))]
    for hole, color in zip(matched_holes, color_pool):
        hole["color"] = color
    for hole in unmatched_holes:
        hole["color"] = None

    for hole in matched_holes:
        transformed = rng.choice(_all_transforms(hole["mask"]))
        sh, sw = transformed.shape
        placed = False
        for _ in range(160):
            r0 = rng.randint(0, h - sh)
            c0 = rng.randint(0, w - sw)
            rr, cc = np.nonzero(transformed)
            cells = [(r0 + int(r), c0 + int(c)) for r, c in zip(rr, cc)]
            if any(occupied[r, c] for r, c in cells):
                continue
            valid = True
            for r, c in cells:
                for dr, dc in DIRS8 + [(0, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            for r, c in cells:
                occupied[r, c] = True
            sprites.append({"pos": (r0, c0), "mask": transformed.copy(), "color": hole["color"]})
            placed = True
            break
        if not placed:
            return None

    distractor_ids = [i for i, shape in enumerate(SHAPE_LIBRARY) if _canonical_shape_key(shape) not in hole_keys]
    rng.shuffle(distractor_ids)
    distractor_count = min(rng.randint(0, 2), len(distractor_ids))
    for shape_id in distractor_ids[:distractor_count]:
        transformed = rng.choice(_all_transforms(SHAPE_LIBRARY[shape_id]))
        sh, sw = transformed.shape
        color = rng.choice(SPRITE_COLORS)
        for _ in range(120):
            r0 = rng.randint(0, h - sh)
            c0 = rng.randint(0, w - sw)
            rr, cc = np.nonzero(transformed)
            cells = [(r0 + int(r), c0 + int(c)) for r, c in zip(rr, cc)]
            if any(occupied[r, c] for r, c in cells):
                continue
            valid = True
            for r, c in cells:
                for dr, dc in DIRS8 + [(0, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            for r, c in cells:
                occupied[r, c] = True
            sprites.append({"pos": (r0, c0), "mask": transformed.copy(), "color": color})
            break
    return sprites


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.choices([2, 3], weights=[3, 1])[0]
    specs = _build_container_specs(rng, count)
    if specs is None:
        return None
    h = rng.randint(25, 30)
    w = rng.randint(25, 30)
    containers = _place_containers(rng, h, w, specs)
    if containers is None:
        return None
    base = _render_input_grid(h, w, containers, [])
    sprites = _place_sprites(rng, base, containers)
    if sprites is None:
        return None
    inp = _render_input_grid(h, w, containers, sprites)
    expected = _render_expected_output(h, w, containers)

    all_holes = [hole for container in containers for hole in container["holes"]]
    if not any(hole["color"] is None for hole in all_holes):
        return None
    if not any(container["matched_count"] < len(container["holes"]) for container in containers):
        return None
    if len({container["side"] for container in containers}) < 2:
        return None
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
