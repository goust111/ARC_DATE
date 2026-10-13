from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
BODY = 1
OUTLINE = 2
GREEN = 3
YELLOW = 4
PINK = 6
ORANGE = 7
TEAL = 8
PAIR_ORDER = (2, 3, 4, 7)


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def marker_direction(sprite: np.ndarray, r: int, c: int) -> tuple[int, int]:
    h, w = sprite.shape
    dirs = []
    if r == 0:
        dirs.append((-1, 0))
    if r == h - 1:
        dirs.append((1, 0))
    if c == 0:
        dirs.append((0, -1))
    if c == w - 1:
        dirs.append((0, 1))
    if len(dirs) != 1:
        raise AssertionError(f"marker at {(r, c)} must lie on exactly one side")
    return dirs[0]


def parse_objects(grid: np.ndarray):
    bg = get_bg(grid)
    fg = grid[grid != bg]
    assert fg.size > 0
    vals, counts = np.unique(fg, return_counts=True)
    body = int(vals[np.argmax(counts)])
    objects = []
    marker_occurrences: dict[int, list[int]] = {}

    for obj_id, coords in enumerate(find_components(grid != bg)):
        r0, c0, r1, c1 = bbox_from_coords(coords)
        sprite = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        for rr, cc in coords:
            sprite[int(rr) - r0, int(cc) - c0] = grid[int(rr), int(cc)]
        marker_map = {}
        for rr, cc in np.argwhere((sprite != bg) & (sprite != body)):
            color = int(sprite[rr, cc])
            direction = marker_direction(sprite, int(rr), int(cc))
            marker_map[color] = {"pos": (int(rr), int(cc)), "dir": direction}
            marker_occurrences.setdefault(color, []).append(obj_id)
        objects.append(
            {
                "id": obj_id,
                "origin": (r0, c0),
                "sprite": sprite,
                "markers": marker_map,
                "shape": sprite.shape,
            }
        )
    return bg, body, objects, marker_occurrences


def dock_objects(objects, marker_occurrences):
    placements = {obj["id"]: obj["origin"] for obj in objects}
    groups = {obj["id"]: {obj["id"]} for obj in objects}
    group_of = {obj["id"]: obj["id"] for obj in objects}
    orig_anchor = {obj["id"]: obj["origin"] for obj in objects}
    by_id = {obj["id"]: obj for obj in objects}

    def abs_pos(obj_id: int, local: tuple[int, int]) -> tuple[int, int]:
        orr, occ = placements[obj_id]
        return orr + local[0], occ + local[1]

    def group_key(gid: int):
        return min(orig_anchor[obj_id] for obj_id in groups[gid])

    for color in sorted(marker_occurrences):
        owners = marker_occurrences[color]
        if len(owners) != 2:
            continue
        a, b = owners
        ga, gb = group_of[a], group_of[b]
        if ga == gb:
            continue
        if len(groups[ga]) > 1 and len(groups[gb]) == 1:
            fixed_obj, moving_obj = a, b
        elif len(groups[gb]) > 1 and len(groups[ga]) == 1:
            fixed_obj, moving_obj = b, a
        else:
            if group_key(ga) <= group_key(gb):
                fixed_obj, moving_obj = a, b
            else:
                fixed_obj, moving_obj = b, a
        fixed_gid = group_of[fixed_obj]
        moving_gid = group_of[moving_obj]

        fixed_marker = by_id[fixed_obj]["markers"][color]
        moving_marker = by_id[moving_obj]["markers"][color]
        target_r, target_c = abs_pos(fixed_obj, fixed_marker["pos"])
        dr, dc = fixed_marker["dir"]
        target_r += dr
        target_c += dc
        cur_r, cur_c = abs_pos(moving_obj, moving_marker["pos"])
        shift = (target_r - cur_r, target_c - cur_c)

        for obj_id in groups[moving_gid]:
            r0, c0 = placements[obj_id]
            placements[obj_id] = (r0 + shift[0], c0 + shift[1])

        moved_members = groups[moving_gid]
        groups[fixed_gid].update(moved_members)
        for obj_id in moved_members:
            group_of[obj_id] = fixed_gid
        del groups[moving_gid]

    return placements


def render_scene(objects, placements, shape: tuple[int, int], bg: int):
    grid = np.full(shape, bg, dtype=int)
    owner = np.full(shape, -1, dtype=int)
    for obj in objects:
        r0, c0 = placements[obj["id"]]
        sprite = obj["sprite"]
        h, w = sprite.shape
        assert 0 <= r0 and 0 <= c0 and r0 + h <= shape[0] and c0 + w <= shape[1]
        mask = sprite != bg
        target = grid[r0 : r0 + h, c0 : c0 + w]
        own = owner[r0 : r0 + h, c0 : c0 + w]
        assert not np.any(mask & (target != bg))
        target[mask] = sprite[mask]
        own[mask] = obj["id"]
    return grid, owner


def fill_holes_and_recolor(grid: np.ndarray, owner: np.ndarray, body: int, bg: int) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    visited = np.zeros((h, w), dtype=bool)
    participating: set[int] = set()

    for r in range(h):
        for c in range(w):
            if out[r, c] != bg or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            region = []
            touches_edge = False
            while q:
                cr, cc = q.popleft()
                region.append((cr, cc))
                if cr in (0, h - 1) or cc in (0, w - 1):
                    touches_edge = True
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and out[nr, nc] == bg and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            if touches_edge:
                continue
            for rr, cc in region:
                out[rr, cc] = PINK
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and owner[nr, nc] >= 0:
                        participating.add(int(owner[nr, nc]))

    for obj_id in participating:
        mask = (owner == obj_id) & (out == body)
        out[mask] = TEAL
    return out


def add_outline(grid: np.ndarray, bg: int) -> np.ndarray:
    base = grid.copy()
    out = grid.copy()
    mask = base != bg
    for comp in find_components(mask):
        comp_set = {tuple(p) for p in comp.tolist()}
        for r, c in comp_set:
            for nr, nc in neighbors8(r, c):
                if 0 <= nr < base.shape[0] and 0 <= nc < base.shape[1] and base[nr, nc] == bg:
                    out[nr, nc] = OUTLINE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Assemble matched marker objects, recolor hole-forming pieces, fill the emergent holes, and outline the result.

    Input structure:
    - A black background contains 4-connected blue foreground objects with 1-2 colored connector markers on their outer sides.
    - Marker colors are pairwise docking labels: a color appears either once as a free end or twice on two different objects.
    - Each object is hole-free by itself, but docking some objects can create one or two enclosed background holes.

    Rule:
    - Process marker colors in ascending order and dock equal-colored marker pairs so their exposed sides face each other edge-to-edge, moving only the still-undocked object/group.
    - After all paired docks, detect every 4-connected background hole created inside the assembled shape.
    - Recolor blue pixels of every object that borders a hole to teal, keep connector markers unchanged, fill the holes with magenta, and finally draw a red 8-neighborhood outline around every resulting foreground component.

    Key insight:
    - The trigger is topological and only appears after assembly: no single input object has a hole, but docking creates enclosed voids that identify which assembled objects must be recolored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, body, objects, marker_occurrences = parse_objects(g)
    placements = dock_objects(objects, marker_occurrences)
    assembled, owner = render_scene(objects, placements, g.shape, bg)
    recolored = fill_holes_and_recolor(assembled, owner, body, bg)
    return add_outline(recolored, bg)


def make_left_bottom_object(outer_height: int, top_width: int, two_holes: bool, marker_color: int = OUTLINE) -> np.ndarray:
    height = outer_height - 3
    sprite = np.zeros((height, 3 + top_width), dtype=int)
    sprite[:, :3] = BODY
    sprite[height - 2 :, 3:] = BODY
    if two_holes:
        pillar_col = 3 + top_width // 2
        sprite[:, pillar_col : pillar_col + 1] = BODY
    sprite[0, 1] = marker_color
    return sprite


def make_top_object(top_width: int) -> np.ndarray:
    sprite = np.full((3, top_width + 3), BODY, dtype=int)
    sprite[2, 1] = OUTLINE
    sprite[1, -1] = GREEN
    return sprite


def make_right_wall_object(height: int, with_tail: bool) -> np.ndarray:
    sprite = np.full((height, 3), BODY, dtype=int)
    sprite[1, 0] = GREEN
    if with_tail:
        sprite[-1, 1] = YELLOW
    return sprite


def make_tail_object(height: int) -> np.ndarray:
    sprite = np.full((height, 3), BODY, dtype=int)
    sprite[0, 1] = YELLOW
    if height >= 4 and random.random() < 0.5:
        sprite[-1, 2] = BODY
    return sprite


def make_free_object() -> np.ndarray:
    h = random.randint(4, 5)
    w = random.randint(4, 5)
    sprite = np.full((h, w), BODY, dtype=int)
    notch = random.choice(["tl", "tr", "bl", "br", "none"])
    if notch == "tl":
        sprite[0:2, 0] = BG
    elif notch == "tr":
        sprite[0:2, -1] = BG
    elif notch == "bl":
        sprite[-2:, 0] = BG
    elif notch == "br":
        sprite[-2:, -1] = BG
    sprite[0, w // 2] = ORANGE
    return sprite


def can_place(occupied: np.ndarray, sprite: np.ndarray, origin: tuple[int, int], pad: int) -> bool:
    r0, c0 = origin
    h, w = sprite.shape
    if r0 < 0 or c0 < 0 or r0 + h > occupied.shape[0] or c0 + w > occupied.shape[1]:
        return False
    cells = np.argwhere(sprite != BG)
    for rr, cc in cells:
        gr = r0 + int(rr)
        gc = c0 + int(cc)
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr = gr + dr
                nc = gc + dc
                if 0 <= nr < occupied.shape[0] and 0 <= nc < occupied.shape[1] and occupied[nr, nc]:
                    return False
    return True


def stamp_occupied(occupied: np.ndarray, sprite: np.ndarray, origin: tuple[int, int]) -> None:
    r0, c0 = origin
    for rr, cc in np.argwhere(sprite != BG):
        occupied[r0 + int(rr), c0 + int(cc)] = True


def random_place(
    occupied: np.ndarray,
    sprite: np.ndarray,
    row_range: tuple[int, int],
    col_range: tuple[int, int],
    pad: int,
    predicate=None,
) -> tuple[int, int] | None:
    choices = [
        (r, c)
        for r in range(row_range[0], row_range[1] + 1)
        for c in range(col_range[0], col_range[1] + 1)
    ]
    random.shuffle(choices)
    for origin in choices:
        if predicate is not None and not predicate(origin):
            continue
        if can_place(occupied, sprite, origin, pad):
            return origin
    return None


def count_holes(grid: np.ndarray, bg: int = BG) -> int:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if grid[r, c] != bg or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            touches_edge = False
            while q:
                cr, cc = q.popleft()
                if cr in (0, h - 1) or cc in (0, w - 1):
                    touches_edge = True
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == bg and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            if not touches_edge:
                holes += 1
    return holes


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(22, 26)
    width = random.randint(22, 26)
    two_holes = random.random() < 0.45
    include_tail = random.random() < 0.7
    include_free = random.random() < 0.7
    if not include_tail and not include_free:
        include_free = True

    ring_h = random.randint(8, 10)
    top_w = random.randint(4, 6)
    tail_h = random.randint(3, 5)

    sprites: dict[str, np.ndarray] = {
        "B": make_left_bottom_object(ring_h, top_w, two_holes),
        "A": make_top_object(top_w),
        "C": make_right_wall_object(ring_h, include_tail),
    }
    if include_tail:
        sprites["D"] = make_tail_object(tail_h)
    if include_free:
        sprites["E"] = make_free_object()

    assembly_h = ring_h + (tail_h if include_tail else 0)
    assembly_w = top_w + 6
    if assembly_h + 2 > height or assembly_w + 2 > width:
        return None

    assembly_origin = (
        random.randint(1, height - assembly_h - 1),
        random.randint(1, width - assembly_w - 1),
    )
    out_pos = {
        "A": assembly_origin,
        "B": (assembly_origin[0] + 3, assembly_origin[1]),
        "C": (assembly_origin[0], assembly_origin[1] + 3 + top_w),
    }
    if include_tail:
        out_pos["D"] = (assembly_origin[0] + ring_h, assembly_origin[1] + 3 + top_w)

    output_occupied = np.zeros((height, width), dtype=bool)
    for name in ("B", "A", "C", "D"):
        if name in sprites:
            if not can_place(output_occupied, sprites[name], out_pos[name], pad=0):
                return None
            stamp_occupied(output_occupied, sprites[name], out_pos[name])
    if include_free:
        free_origin = random_place(
            output_occupied,
            sprites["E"],
            (1, height - sprites["E"].shape[0] - 1),
            (1, width - sprites["E"].shape[1] - 1),
            pad=2,
        )
        if free_origin is None:
            return None
        out_pos["E"] = free_origin
        stamp_occupied(output_occupied, sprites["E"], free_origin)

    grid = np.zeros((height, width), dtype=int)
    in_pos = {"B": out_pos["B"]}
    occupied = np.zeros((height, width), dtype=bool)
    stamp_occupied(occupied, sprites["B"], in_pos["B"])
    if include_free:
        in_pos["E"] = out_pos["E"]
        stamp_occupied(occupied, sprites["E"], in_pos["E"])

    for name in ("A", "C", "D"):
        if name not in sprites:
            continue
        predicate = None
        if name == "A":
            b_origin = in_pos["B"]
            predicate = lambda origin, b=b_origin: origin > b
        origin = random_place(
            occupied,
            sprites[name],
            (1, height - sprites[name].shape[0] - 1),
            (1, width - sprites[name].shape[1] - 1),
            pad=2,
            predicate=predicate,
        )
        if origin is None:
            return None
        in_pos[name] = origin
        stamp_occupied(occupied, sprites[name], origin)

    for name, sprite in sprites.items():
        r0, c0 = in_pos[name]
        mask = sprite != BG
        grid[r0 : r0 + sprite.shape[0], c0 : c0 + sprite.shape[1]][mask] = sprite[mask]

    comp_count = len(find_components(grid != BG))
    expected_count = 3 + int(include_tail) + int(include_free)
    if comp_count != expected_count:
        return None

    if any(count_holes(sprite) for sprite in sprites.values()):
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(out == PINK) == 0:
        return None
    if np.count_nonzero(out == TEAL) == 0:
        return None
    if np.count_nonzero(out == BODY) == 0:
        return None
    if count_holes((out != BG).astype(int), bg=0) != 0:
        return None
    hole_components = len(find_components(out == PINK))
    if hole_components != (2 if two_holes else 1):
        return None
    return grid
