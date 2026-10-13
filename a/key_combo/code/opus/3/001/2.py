from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 5
BODY = 7
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

# Same seven-body directional sprite family used in 13f06aa5: the orange body
# sits behind the marker, so the ray direction is opposite the centroid bias.
BODY_REL = {
    (-1, 0): [(0, -2), (0, -1), (0, 1), (0, 2), (1, -1), (1, 0), (1, 1)],
    (1, 0): [(-1, -1), (-1, 0), (-1, 1), (0, -2), (0, -1), (0, 1), (0, 2)],
    (0, -1): [(-2, 0), (-1, 0), (-1, 1), (0, 1), (1, 0), (1, 1), (2, 0)],
    (0, 1): [(-2, 0), (-1, -1), (-1, 0), (0, -1), (1, -1), (1, 0), (2, 0)],
}

_CLOSED_TEMPLATES = [
    np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 1, 1]], dtype=bool),
    np.array(
        [[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]], dtype=bool
    ),
]

GEN_ATTEMPTS = 600
PALETTE = [1, 2, 3, 4, 6, 8, 9]


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_from_mask(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
    out: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _mask_from_coords(coords: np.ndarray, bbox: tuple[int, int, int, int] | None = None) -> np.ndarray:
    if bbox is None:
        bbox = _bbox_coords(coords)
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _has_enclosed_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    exterior = np.zeros_like(mask, dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                stack.append((nr, nc))

    return bool(np.any((~mask) & (~exterior)))


def _object_patch(coords: np.ndarray, color: int, bg: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    bbox = _bbox_coords(coords)
    mask = _mask_from_coords(coords, bbox)
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = int(color)
    rel = coords - np.array([bbox[0], bbox[1]], dtype=int)
    return patch, rel, bbox


def _sprite_direction(body_coords: np.ndarray, marker: tuple[int, int]) -> tuple[int, int]:
    rel = body_coords - np.array(marker, dtype=int)
    row_bias = int(rel[:, 0].sum())
    col_bias = int(rel[:, 1].sum())
    if abs(row_bias) == abs(col_bias):
        raise AssertionError("sprite bias must quantize to one cardinal direction")
    if abs(row_bias) > abs(col_bias):
        return (-1, 0) if row_bias > 0 else (1, 0)
    return (0, -1) if col_bias > 0 else (0, 1)


def _sprite_pixels(marker: tuple[int, int], direction: tuple[int, int], marker_color: int) -> list[tuple[int, int, int]]:
    mr, mc = marker
    pixels = [(mr, mc, int(marker_color))]
    for dr, dc in BODY_REL[direction]:
        pixels.append((mr + dr, mc + dc, BODY))
    return pixels


def _sprite_coords(marker: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    return [(r, c) for r, c, _color in _sprite_pixels(marker, direction, marker_color=1)]


def _stamp_pixels(grid: np.ndarray, pixels: list[tuple[int, int, int]]) -> None:
    for r, c, color in pixels:
        grid[r, c] = int(color)


def _can_place_coords(coords: list[tuple[int, int]], occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                    return False
    return True


def _occupy(coords: list[tuple[int, int]], occupied: np.ndarray) -> None:
    for r, c in coords:
        occupied[r, c] = True


def _extract_entities(
    grid: np.ndarray,
) -> tuple[int, list[dict], list[dict], np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)

    non_bg_components = _components_from_mask(g != bg, connectivity=8)
    sprite_mask = np.zeros_like(g, dtype=bool)
    sprites: list[dict] = []

    for coords in non_bg_components:
        colors = [int(g[r, c]) for r, c in coords]
        counts = Counter(colors)
        if (
            len(coords) == 8
            and counts.get(BODY, 0) == 7
            and len(counts) == 2
            and 1 in counts.values()
        ):
            marker_colors = [color for color, count in counts.items() if color != BODY and count == 1]
            if len(marker_colors) != 1:
                continue
            marker_color = int(marker_colors[0])
            marker_rc = tuple(map(int, np.argwhere(g == marker_color)[0]))
            marker = None
            for r, c in coords:
                if int(g[r, c]) == marker_color:
                    marker = (int(r), int(c))
                    break
            assert marker is not None
            body_coords = np.array(
                [(int(r), int(c)) for r, c in coords if int(g[r, c]) == BODY], dtype=int
            )
            direction = _sprite_direction(body_coords, marker)
            pixels = [(int(r), int(c), int(g[r, c])) for r, c in coords]
            bbox = _bbox_coords(coords)
            sprites.append(
                {
                    "marker": marker,
                    "marker_color": marker_color,
                    "body_coords": body_coords,
                    "direction": direction,
                    "pixels": pixels,
                    "bbox": bbox,
                }
            )
            sprite_mask[coords[:, 0], coords[:, 1]] = True

    remaining = g.copy()
    remaining[sprite_mask] = bg
    objects: list[dict] = []
    for color in sorted(set(int(v) for v in np.unique(remaining)) - {bg}):
        for coords in _components_from_mask(remaining == color, connectivity=4):
            patch, rel, bbox = _object_patch(coords, color, bg)
            mask = patch != bg
            objects.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox,
                    "patch": patch,
                    "rel": rel,
                    "closed": _has_enclosed_hole(mask),
                    "height": int(patch.shape[0]),
                    "width": int(patch.shape[1]),
                }
            )

    object_mask = (remaining != bg)
    return bg, sprites, objects, sprite_mask, object_mask


def _can_place_object(canvas: np.ndarray, obj: dict, row: int, bg: int) -> bool:
    c0 = int(obj["bbox"][1])
    for dr, dc in obj["rel"]:
        if canvas[row + int(dr), c0 + int(dc)] != bg:
            return False
    return True


def _place_object(canvas: np.ndarray, object_mask: np.ndarray, obj: dict, row: int, bg: int) -> dict:
    c0 = int(obj["bbox"][1])
    for dr, dc in obj["rel"]:
        rr = row + int(dr)
        cc = c0 + int(dc)
        canvas[rr, cc] = int(obj["color"])
        object_mask[rr, cc] = True
    placed = dict(obj)
    placed["final_row"] = int(row)
    placed["final_bbox"] = (int(row), c0, int(row + obj["height"] - 1), int(c0 + obj["width"] - 1))
    return placed


def _settle_entities(
    grid: np.ndarray, sprites: list[dict], objects: list[dict], bg: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    settled = np.full((h, w), bg, dtype=int)
    sprite_mask = np.zeros_like(settled, dtype=bool)
    for sprite in sprites:
        for r, c, color in sprite["pixels"]:
            settled[r, c] = int(color)
            sprite_mask[r, c] = True

    object_mask = np.zeros_like(settled, dtype=bool)
    placed_objects: list[dict] = []

    closed = sorted(objects, key=lambda obj: (obj["bbox"][0], obj["bbox"][1], obj["bbox"][2], obj["bbox"][3]))
    open_objects = sorted(
        [obj for obj in objects if not obj["closed"]],
        key=lambda obj: (-obj["bbox"][2], obj["bbox"][1], -obj["bbox"][0], obj["bbox"][3]),
    )
    closed = [obj for obj in closed if obj["closed"]]

    for obj in closed:
        for row in range(0, h - obj["height"] + 1):
            if _can_place_object(settled, obj, row, bg):
                placed_objects.append(_place_object(settled, object_mask, obj, row, bg))
                break
        else:
            raise AssertionError("closed object could not be placed")

    for obj in open_objects:
        for row in range(h - obj["height"], -1, -1):
            if _can_place_object(settled, obj, row, bg):
                placed_objects.append(_place_object(settled, object_mask, obj, row, bg))
                break
        else:
            raise AssertionError("open object could not be placed")

    return settled, object_mask, sprite_mask, placed_objects


def _trace_rays(
    obstacle: np.ndarray,
    sprites: list[dict],
    object_mask: np.ndarray,
    sprite_mask: np.ndarray,
    bg: int,
    *,
    paint: bool,
) -> tuple[np.ndarray, list[dict]]:
    out = obstacle.copy()
    ordered = sorted(sprites, key=lambda sprite: (sprite["bbox"][1], sprite["bbox"][0], sprite["marker"][1], sprite["marker"][0]))
    seen_touches: dict[tuple[int, int], tuple[int, int]] = {}
    conflict_coords: set[tuple[int, int]] = set()
    ray_meta: list[dict] = []

    def touch(coord: tuple[int, int], color: int, ray_idx: int) -> None:
        prev = seen_touches.get(coord)
        if prev is not None and prev[1] != color:
            conflict_coords.add(coord)
        seen_touches[coord] = (ray_idx, color)

    for ray_idx, sprite in enumerate(ordered):
        mr, mc = sprite["marker"]
        dr, dc = sprite["direction"]
        color = int(sprite["marker_color"])
        step = 1
        touched: list[tuple[int, int]] = []
        while True:
            rr = mr + step * dr
            cc = mc + step * dc
            if not (0 <= rr < obstacle.shape[0] and 0 <= cc < obstacle.shape[1]):
                raise AssertionError("ray left the grid without reaching a boundary cell")
            if obstacle[rr, cc] != bg:
                hit_kind = "hit_object" if object_mask[rr, cc] else "hit_sprite"
                if paint:
                    out[rr, cc] = color
                touch((rr, cc), color, ray_idx)
                touched.append((rr, cc))
                ray_meta.append(
                    {
                        "marker": sprite["marker"],
                        "direction": sprite["direction"],
                        "outcome": hit_kind,
                        "travel": int(step),
                        "stop": ("hit", hit_kind, int(rr), int(cc)),
                        "touched": touched,
                    }
                )
                break

            if rr == 0 or rr == obstacle.shape[0] - 1 or cc == 0 or cc == obstacle.shape[1] - 1:
                if paint:
                    if dr != 0:
                        out[rr, :] = color
                    else:
                        out[:, cc] = color
                if dr != 0:
                    side_cells = [(int(rr), int(col)) for col in range(obstacle.shape[1])]
                    stop = ("edge", "row", int(rr))
                else:
                    side_cells = [(int(row), int(cc)) for row in range(obstacle.shape[0])]
                    stop = ("edge", "col", int(cc))
                for coord in side_cells:
                    touch(coord, color, ray_idx)
                touched.extend(side_cells)
                ray_meta.append(
                    {
                        "marker": sprite["marker"],
                        "direction": sprite["direction"],
                        "outcome": "edge",
                        "travel": int(step),
                        "stop": stop,
                        "touched": touched,
                    }
                )
                break

            if step % 2 == 1:
                if paint:
                    out[rr, cc] = color
                touch((rr, cc), color, ray_idx)
                touched.append((rr, cc))
            step += 1

    for ray in ray_meta:
        ray["conflict_coords"] = sorted(coord for coord in ray["touched"] if coord in conflict_coords)
    return out, ray_meta


def _solve(grid: np.ndarray, *, return_meta: bool = False) -> np.ndarray | tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    bg, sprites, objects, sprite_mask, object_mask_initial = _extract_entities(g)
    assert len(sprites) >= 2, "expected at least two directional sprites"
    assert objects, "expected at least one slideable object"
    assert any(obj["closed"] for obj in objects), "expected a closed object"
    assert any(not obj["closed"] for obj in objects), "expected an open object"

    settled, object_mask_settled, sprite_mask_settled, placed_objects = _settle_entities(g, sprites, objects, bg)
    out, rays = _trace_rays(settled, sprites, object_mask_settled, sprite_mask_settled, bg, paint=True)

    if not return_meta:
        return out

    _unused_out, rays_on_input = _trace_rays(g, sprites, object_mask_initial, sprite_mask, bg, paint=False)
    meta = {
        "bg": bg,
        "sprites": sprites,
        "objects": objects,
        "settled": settled,
        "placed_objects": placed_objects,
        "rays": rays,
        "rays_on_input": rays_on_input,
        "sprite_mask": sprite_mask_settled,
        "object_mask_initial": object_mask_initial,
        "object_mask_settled": object_mask_settled,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide hole-bearing objects upward and holeless objects downward, then fire dotted rays that stop on the settled obstacle layout.

    Input structure:
    - The background is gray, with 2-3 monochrome 4-connected non-sprite objects in other colors.
    - At least one object is a closed frame whose bbox encloses a gray hole, and at least one object is open with no enclosed hole.
    - There are 2-3 eight-connected directional sprites: seven orange body pixels plus one non-orange marker pixel.

    Rule:
    - Separate the sprites from the slideable objects. Classify each object by whether its bbox contains an enclosed gray hole.
    - Slide closed objects straight upward and open objects straight downward, preserving each object's shape, color, columns, and within-direction vertical order; sprites stay fixed.
    - For each sprite, infer the ray direction as the direction opposite the orange centroid bias relative to its marker. Starting at the marker, color step 1, skip step 2, color step 3, and so on.
    - If the ray first meets a non-background settled obstacle, recolor that hit pixel to the marker color and stop. If it instead reaches a grid side, color the entire reached boundary row or column. Later sprites in left-to-right, then top-to-bottom order overwrite earlier ray paint on conflicts.

    Key insight:
    - The rays must be traced against the post-slide topology: moving hole objects upward and holeless objects downward changes which obstacle each sprite reaches, so ray length, stopping point, and recoloring all depend on the slide stage.
    """
    return _solve(grid, return_meta=False)  # type: ignore[return-value]


def _random_open_mask() -> np.ndarray:
    for _ in range(200):
        target = random.randint(5, 8)
        cells = {(0, 0)}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.append(nxt)
            frontier.sort(key=lambda rc: (abs(rc[0]) + abs(rc[1]), rc[0], rc[1]))
            best = frontier[: max(6, len(frontier) // 2)]
            cells.add(random.choice(best))

        coords = np.array(sorted(cells), dtype=int)
        coords -= coords.min(axis=0)
        h = int(coords[:, 0].max()) + 1
        w = int(coords[:, 1].max()) + 1
        if h > 5 or w > 5 or h < 2 or w < 2:
            continue
        mask = np.zeros((h, w), dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        if int(mask.sum()) != target:
            continue
        if _has_enclosed_hole(mask):
            continue
        if int(mask.sum()) == h * w:
            continue
        if np.count_nonzero(mask.sum(axis=0) > 1) == 0 and np.count_nonzero(mask.sum(axis=1) > 1) == 0:
            continue
        return mask
    raise RuntimeError("failed to generate an open shape")


def _random_closed_mask() -> np.ndarray:
    mask = random.choice(_CLOSED_TEMPLATES).copy()
    k = random.randrange(4)
    if k:
        mask = np.rot90(mask, k)
    return mask


def _mask_cells(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    rows, cols = np.where(mask)
    return [(int(top + r), int(left + c)) for r, c in zip(rows.tolist(), cols.tolist())]


def _place_object_in_lane(
    occupied: np.ndarray,
    mask: np.ndarray,
    color: int,
    ray_col: int,
    row_candidates: list[int],
) -> tuple[int, int, list[tuple[int, int]]] | None:
    h, w = occupied.shape
    mh, mw = mask.shape
    c_min = max(1, ray_col - mw + 1)
    c_max = min(w - mw - 1, ray_col)
    if c_min > c_max:
        return None

    candidates: list[tuple[int, int, list[tuple[int, int]]]] = []
    rows = row_candidates[:]
    cols = list(range(c_min, c_max + 1))
    random.shuffle(rows)
    random.shuffle(cols)
    for top in rows:
        if top < 1 or top + mh > h - 1:
            continue
        for left in cols:
            cells = _mask_cells(mask, top, left)
            if _can_place_coords(cells, occupied):
                candidates.append((top, left, cells))
    if not candidates:
        return None
    top, left, cells = random.choice(candidates)
    return int(top), int(left), cells


def _place_sprite(
    occupied: np.ndarray,
    marker: tuple[int, int],
    direction: tuple[int, int],
    marker_color: int,
) -> dict | None:
    cells = _sprite_coords(marker, direction)
    if not _can_place_coords(cells, occupied):
        return None
    _occupy(cells, occupied)
    return {"marker": marker, "direction": direction, "marker_color": int(marker_color)}


def _stamp_generated_object(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rows, cols = np.where(mask)
    grid[top + rows, left + cols] = int(color)


def _generate_candidate() -> np.ndarray | None:
    h = random.randint(19, 23)
    w = random.randint(19, 23)
    if w < 19:
        return None

    ray_col_open = random.randint(4, w // 2 - 3)
    ray_col_closed = random.randint(w // 2 + 2, w - 5)
    if ray_col_closed - ray_col_open < 8:
        return None

    row_down = random.randint(3, 4)
    row_up = h - random.randint(4, 5)

    target_lane = random.choice(["open", "closed"])
    if target_lane == "open":
        row_choices = [row_down + d for d in (5, 7, 9) if row_down + d < row_up - 3]
        if not row_choices:
            return None
        row_h = random.choice(row_choices)
        dir_h = (0, -1)
        c_lo = ray_col_open + 5
        c_hi = ray_col_closed - 3
        cols_h = [c for c in range(c_lo, c_hi + 1) if (c - ray_col_open) % 2 == 1]
    else:
        row_choices = [row_up - d for d in (5, 7, 9) if row_up - d > row_down + 3]
        if not row_choices:
            return None
        row_h = random.choice(row_choices)
        dir_h = (0, 1)
        c_lo = ray_col_open + 3
        c_hi = ray_col_closed - 5
        cols_h = [c for c in range(c_lo, c_hi + 1) if (ray_col_closed - c) % 2 == 1]
    if not cols_h:
        return None
    col_h = random.choice(cols_h)

    n_objects = 2 + random.randint(0, 1)
    open_count = 1
    closed_count = 1
    if n_objects == 3:
        if random.random() < 0.5:
            open_count += 1
        else:
            closed_count += 1

    total_colors = n_objects + 3
    colors = random.sample(PALETTE, total_colors)
    object_colors = colors[:n_objects]
    marker_colors = colors[n_objects:]

    occupied = np.zeros((h, w), dtype=bool)
    vertical_down = _place_sprite(occupied, (row_down, ray_col_open), (1, 0), marker_colors[0])
    if vertical_down is None:
        return None
    vertical_up = _place_sprite(occupied, (row_up, ray_col_closed), (-1, 0), marker_colors[1])
    if vertical_up is None:
        return None
    horizontal = _place_sprite(occupied, (row_h, col_h), dir_h, marker_colors[2])
    if horizontal is None:
        return None

    open_rows = list(range(row_down + 3, h - 7))
    closed_rows = list(range(4, row_up - 5))
    if not open_rows or not closed_rows:
        return None

    placements: list[tuple[np.ndarray, int, int, int]] = []
    colors_iter = iter(object_colors)

    open_shapes = [_random_open_mask() for _ in range(open_count)]
    closed_shapes = [_random_closed_mask() for _ in range(closed_count)]

    for mask in open_shapes:
        color = next(colors_iter)
        placed = _place_object_in_lane(occupied, mask, color, ray_col_open, open_rows)
        if placed is None:
            return None
        top, left, cells = placed
        _occupy(cells, occupied)
        placements.append((mask, top, left, color))

    for mask in closed_shapes:
        color = next(colors_iter)
        placed = _place_object_in_lane(occupied, mask, color, ray_col_closed, closed_rows)
        if placed is None:
            return None
        top, left, cells = placed
        _occupy(cells, occupied)
        placements.append((mask, top, left, color))

    grid = np.full((h, w), BG, dtype=int)
    for sprite in (vertical_down, vertical_up, horizontal):
        _stamp_pixels(grid, _sprite_pixels(sprite["marker"], sprite["direction"], sprite["marker_color"]))
    for mask, top, left, color in placements:
        _stamp_generated_object(grid, mask, top, left, color)
    return grid


def _validate_candidate(grid: np.ndarray) -> bool:
    try:
        out, meta = _solve(grid, return_meta=True)
    except Exception:
        return False

    objects = meta["objects"]
    rays = meta["rays"]
    rays_on_input_by_marker = {tuple(ray["marker"]): ray for ray in meta["rays_on_input"]}

    if len(meta["sprites"]) != 3:
        return False
    if len(objects) not in (2, 3):
        return False
    if sum(obj["closed"] for obj in objects) < 1 or sum(not obj["closed"] for obj in objects) < 1:
        return False

    if not any(obj["final_row"] != obj["bbox"][0] for obj in meta["placed_objects"]):
        return False

    if sum(ray["outcome"] == "hit_object" for ray in rays) < 2:
        return False
    if sum(ray["outcome"] == "edge" for ray in rays) < 1:
        return False
    if not any(ray["outcome"] == "edge" and ray["travel"] >= 4 for ray in rays):
        return False
    if not any(ray["outcome"] == "hit_object" and ray["travel"] >= 4 for ray in rays):
        return False

    changed_rays = 0
    for ray in rays:
        src = rays_on_input_by_marker[tuple(ray["marker"])]
        if ray["stop"] != src["stop"]:
            changed_rays += 1
    if changed_rays < 2:
        return False

    conflict_coords = set()
    for ray in rays:
        conflict_coords.update(tuple(coord) for coord in ray["conflict_coords"])
    if not conflict_coords:
        return False

    if np.array_equal(grid, out):
        return False
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False
    if np.count_nonzero(out != meta["settled"]) < 8:
        return False
    if np.count_nonzero(out != meta["settled"]) > grid.size // 3:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_candidate()
        if grid is None:
            continue
        if _validate_candidate(grid):
            return grid
    raise RuntimeError("failed to generate a valid topological obstacle ray input")
