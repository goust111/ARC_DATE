from __future__ import annotations

from collections import Counter, deque
import hashlib

import numpy as np


BLUE = 1
PINK = 6

ALL_COLORS = tuple(range(10))
BG_CHOICES = (0, 2, 3, 4, 5, 7, 8, 9)
GEN_ATTEMPTS = 1200

_RNG = np.random.default_rng()
_SEEN_INPUT_SIGS: set[str] = set()
_SEEN_OUTPUT_SIGS: set[str] = set()


Coord = tuple[int, int]


def _signature(grid: np.ndarray) -> str:
    arr = np.asarray(grid, dtype=int)
    return hashlib.md5(arr.tobytes() + str(tuple(arr.shape)).encode()).hexdigest()


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components4(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[Coord] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _corner_positions(shape: tuple[int, int]) -> list[Coord]:
    h, w = map(int, shape)
    return [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]


def _find_marker(grid: np.ndarray, bg: int) -> tuple[Coord, int]:
    g = np.asarray(grid, dtype=int)
    counts = Counter(map(int, g.ravel()))

    pink_corners = [(r, c) for r, c in _corner_positions(g.shape) if int(g[r, c]) == PINK]
    if len(pink_corners) == 1:
        return pink_corners[0], PINK

    unique_corner_markers: list[tuple[Coord, int]] = []
    for r, c in _corner_positions(g.shape):
        color = int(g[r, c])
        if color != bg and counts[color] == 1:
            unique_corner_markers.append(((r, c), color))
    assert len(unique_corner_markers) == 1, "expected one unique corner marker"
    return unique_corner_markers[0]


def _flood_reachable_background(grid: np.ndarray, bg: int, marker: Coord) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    reach = np.zeros_like(g, dtype=bool)
    q = deque([marker])
    reach[marker] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(g.shape, nr, nc) or bool(reach[nr, nc]):
                continue
            if int(g[nr, nc]) == bg:
                reach[nr, nc] = True
                q.append((nr, nc))
    return reach


def _find_sprites(grid: np.ndarray, bg: int, marker_color: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    sprite_mask = (g != bg) & (g != BLUE) & (g != marker_color)
    comps = _components4(sprite_mask)
    sprites = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprites.append(
            {
                "coords": comp,
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
            }
        )
    sprites.sort(key=lambda item: item["bbox"])
    return sprites


def _sprite_reachable(sprite: dict, reach: np.ndarray) -> bool:
    for r, c in sprite["coords"]:
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(reach.shape, nr, nc) and bool(reach[nr, nc]):
                return True
    return False


def _clear_sprites(grid: np.ndarray, sprites: list[dict], bg: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for sprite in sprites:
        for r, c in sprite["coords"]:
            out[r, c] = bg
    return out


def _mirror_all_sprites(grid: np.ndarray, sprites: list[dict], bg: int) -> np.ndarray:
    out = _clear_sprites(grid, sprites, bg)
    width = int(grid.shape[1])
    for sprite in sprites:
        r0, c0, r1, c1 = sprite["bbox"]
        patch = sprite["patch"]
        h, w = patch.shape
        dst_c0 = width - w - c0
        mask = patch != bg
        target = out[r0 : r0 + h, dst_c0 : dst_c0 + w]
        target[mask] = patch[mask]
    return out


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    bg = _get_bg(g)
    marker, marker_color = _find_marker(g, bg)
    reach = _flood_reachable_background(g, bg, marker)
    sprites = _find_sprites(g, bg, marker_color)
    assert sprites, "expected at least one sprite"

    out = _clear_sprites(g, sprites, bg)
    width = int(g.shape[1])

    for sprite in sprites:
        if not _sprite_reachable(sprite, reach):
            continue
        r0, c0, _r1, _c1 = sprite["bbox"]
        patch = sprite["patch"]
        h, w = patch.shape
        dst_c0 = width - w - c0
        assert 0 <= dst_c0 <= width - w, "mirrored sprite must stay in bounds"

        mask = patch != bg
        target = out[r0 : r0 + h, dst_c0 : dst_c0 + w]
        assert not np.any(target[mask] != bg), "mirrored sprite landing cells must be empty"
        target[mask] = patch[mask]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood from the corner marker to decide which sprites survive, then mirror only the reachable ones horizontally.

    Input structure:
    - The background is the most common color, with blue wall pixels partitioning the canvas into regions.
    - A single magenta marker pixel occupies one corner of the grid.
    - Several small multicolor sprites sit on background cells, and their mirrored landing boxes are empty.

    Rule:
    - Flood-fill from the magenta corner marker using 4-connectivity through background cells only; blue walls block the flood.
    - Treat a sprite as reachable iff at least one of its pixels is 4-neighbor adjacent to the flooded background.
    - Move each reachable sprite without flipping it to the horizontally mirrored column position `new_c0 = grid_width - sprite_width - old_c0`, and erase every unreachable sprite to background.
    - Keep the walls and the corner marker unchanged.

    Key insight:
    - The output is not obtained by mirroring every sprite: flood-fill first decides which sprites are connected to the marker-side background, and the mirror changes position only, not sprite shape.
    """
    return _transform(grid)


def _sample_sprite_patch(bg: int) -> np.ndarray:
    sprite_palette = [c for c in ALL_COLORS if c not in {bg, BLUE, PINK}]
    for _ in range(200):
        h = int(_RNG.integers(2, 5))
        w = int(_RNG.integers(2, 5))
        target_area = int(_RNG.integers(3, min(7, h * w) + 1))

        coords = {(int(_RNG.integers(h)), int(_RNG.integers(w)))}
        while len(coords) < target_area:
            frontier: set[Coord] = set()
            for r, c in coords:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in coords:
                        frontier.add((nr, nc))
            if not frontier:
                break
            coords.add(frontier.pop() if len(frontier) == 1 else list(frontier)[int(_RNG.integers(len(frontier)))])

        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, c0, r1, c1 = min(rows), min(cols), max(rows), max(cols)
        if r1 == r0 or c1 == c0:
            continue

        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        local_coords = [(r - r0, c - c0) for r, c in coords]
        color_count = min(len(local_coords), int(_RNG.integers(2, 4)))
        colors = list(map(int, _RNG.choice(sprite_palette, size=color_count, replace=False).tolist()))
        assignments = colors[:]
        while len(assignments) < len(local_coords):
            assignments.append(int(_RNG.choice(colors)))
        _RNG.shuffle(assignments)
        for (r, c), color in zip(local_coords, assignments):
            patch[r, c] = color
        return patch
    raise RuntimeError("failed to sample sprite patch")


def _mark_reserved(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(reserved.shape[0], r0 + h + pad)
    cc1 = min(reserved.shape[1], c0 + w + pad)
    reserved[rr0:rr1, cc0:cc1] = True


def _reserve_conflict(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(reserved.shape[0], r0 + h + pad)
    cc1 = min(reserved.shape[1], c0 + w + pad)
    return bool(np.any(reserved[rr0:rr1, cc0:cc1]))


def _room_cells(shape: tuple[int, int], side: str, inner_h: int, inner_w: int, offset: int) -> tuple[set[Coord], set[Coord]]:
    h, w = shape
    walls: set[Coord] = set()
    interior: set[Coord] = set()

    if side == "top":
        c0 = offset
        c1 = c0 + inner_w + 1
        bottom = inner_h
        for r in range(0, bottom + 1):
            walls.add((r, c0))
            walls.add((r, c1))
        for c in range(c0, c1 + 1):
            walls.add((bottom, c))
        for r in range(0, inner_h):
            for c in range(c0 + 1, c1):
                interior.add((r, c))
    elif side == "bottom":
        c0 = offset
        c1 = c0 + inner_w + 1
        top = h - inner_h - 1
        for r in range(top, h):
            walls.add((r, c0))
            walls.add((r, c1))
        for c in range(c0, c1 + 1):
            walls.add((top, c))
        for r in range(top + 1, h):
            for c in range(c0 + 1, c1):
                interior.add((r, c))
    elif side == "left":
        r0 = offset
        r1 = r0 + inner_h + 1
        right = inner_w
        for c in range(0, right + 1):
            walls.add((r0, c))
            walls.add((r1, c))
        for r in range(r0, r1 + 1):
            walls.add((r, right))
        for r in range(r0 + 1, r1):
            for c in range(0, inner_w):
                interior.add((r, c))
    elif side == "right":
        r0 = offset
        r1 = r0 + inner_h + 1
        left = w - inner_w - 1
        for c in range(left, w):
            walls.add((r0, c))
            walls.add((r1, c))
        for r in range(r0, r1 + 1):
            walls.add((r, left))
        for r in range(r0 + 1, r1):
            for c in range(left + 1, w):
                interior.add((r, c))
    else:
        raise AssertionError("unknown room side")

    assert all(_in_bounds(shape, r, c) for r, c in walls | interior)
    return walls, interior


def _cells_with_padding(cells: set[Coord], shape: tuple[int, int], pad: int = 1) -> set[Coord]:
    expanded: set[Coord] = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr, nc = r + dr, c + dc
                if _in_bounds(shape, nr, nc):
                    expanded.add((nr, nc))
    return expanded


def _place_border_room(grid: np.ndarray, wall_mask: np.ndarray, marker: Coord) -> set[Coord] | None:
    h, w = grid.shape
    sides = ["top", "bottom", "left", "right"]
    _RNG.shuffle(sides)

    for side in sides:
        for _ in range(60):
            if side in {"top", "bottom"}:
                inner_h = int(_RNG.integers(3, min(7, h - 2)))
                inner_w = int(_RNG.integers(3, min(8, w - 2)))
                max_offset = w - inner_w - 2
                if max_offset < 0:
                    continue
                offset = int(_RNG.integers(max_offset + 1))
            else:
                inner_h = int(_RNG.integers(3, min(8, h - 2)))
                inner_w = int(_RNG.integers(3, min(7, w - 2)))
                max_offset = h - inner_h - 2
                if max_offset < 0:
                    continue
                offset = int(_RNG.integers(max_offset + 1))

            walls, interior = _room_cells(grid.shape, side, inner_h, inner_w, offset)
            if marker in walls or marker in interior:
                continue
            if len(interior) < 9:
                continue
            expanded = _cells_with_padding(walls, grid.shape, pad=1)
            if any(bool(wall_mask[r, c]) for r, c in expanded):
                continue
            for r, c in walls:
                wall_mask[r, c] = True
                grid[r, c] = BLUE
            return interior
    return None


def _place_wall_stub(grid: np.ndarray, wall_mask: np.ndarray, marker: Coord) -> bool:
    h, w = grid.shape
    kinds = ["top_v", "bottom_v", "left_h", "right_h"]
    _RNG.shuffle(kinds)
    for kind in kinds:
        for _ in range(40):
            if kind == "top_v":
                length = int(_RNG.integers(2, min(7, h)))
                c = int(_RNG.integers(1, w - 1))
                cells = {(r, c) for r in range(length)}
            elif kind == "bottom_v":
                length = int(_RNG.integers(2, min(7, h)))
                c = int(_RNG.integers(1, w - 1))
                cells = {(h - 1 - r, c) for r in range(length)}
            elif kind == "left_h":
                length = int(_RNG.integers(2, min(7, w)))
                r = int(_RNG.integers(1, h - 1))
                cells = {(r, c) for c in range(length)}
            else:
                length = int(_RNG.integers(2, min(7, w)))
                r = int(_RNG.integers(1, h - 1))
                cells = {(r, w - 1 - c) for c in range(length)}

            if marker in cells:
                continue
            expanded = _cells_with_padding(cells, grid.shape, pad=1)
            if any(bool(wall_mask[r, c]) for r, c in expanded):
                continue
            for r, c in cells:
                wall_mask[r, c] = True
                grid[r, c] = BLUE
            return True
    return False


def _component_masks(grid: np.ndarray, bg: int, marker: Coord) -> tuple[np.ndarray, list[np.ndarray]]:
    bg_mask = grid == bg
    reach = _flood_reachable_background(grid, bg, marker)
    unreachable_masks: list[np.ndarray] = []
    unreached_bg = bg_mask & ~reach
    for comp in _components4(unreached_bg):
        comp_mask = np.zeros_like(grid, dtype=bool)
        for r, c in comp:
            comp_mask[r, c] = True
        unreachable_masks.append(comp_mask)
    return reach & bg_mask, unreachable_masks


def _candidate_positions(
    grid: np.ndarray,
    reserved: np.ndarray,
    region_mask: np.ndarray,
    patch: np.ndarray,
    bg: int,
) -> list[Coord]:
    h, w = patch.shape
    height, width = grid.shape
    valid: list[Coord] = []
    for r0 in range(height - h + 1):
        for c0 in range(width - w + 1):
            dst_c0 = width - w - c0
            if dst_c0 == c0:
                continue
            if not np.all(region_mask[r0 : r0 + h, c0 : c0 + w]):
                continue
            if np.any(grid[r0 : r0 + h, c0 : c0 + w] != bg):
                continue
            if np.any(grid[r0 : r0 + h, dst_c0 : dst_c0 + w] != bg):
                continue
            if _reserve_conflict(reserved, r0, c0, h, w) or _reserve_conflict(reserved, r0, dst_c0, h, w):
                continue
            valid.append((r0, c0))
    return valid


def _place_sprite(
    grid: np.ndarray,
    reserved: np.ndarray,
    region_mask: np.ndarray,
    patch: np.ndarray,
    bg: int,
) -> bool:
    choices = _candidate_positions(grid, reserved, region_mask, patch, bg)
    if not choices:
        return False
    r0, c0 = choices[int(_RNG.integers(len(choices)))]
    h, w = patch.shape
    dst_c0 = grid.shape[1] - w - c0

    mask = patch != bg
    target = grid[r0 : r0 + h, c0 : c0 + w]
    target[mask] = patch[mask]
    _mark_reserved(reserved, r0, c0, h, w, pad=1)
    _mark_reserved(reserved, r0, dst_c0, h, w, pad=1)
    return True


def _side_label(bbox: tuple[int, int, int, int], width: int) -> str:
    _r0, c0, _r1, c1 = bbox
    center = (c0 + c1) / 2.0
    return "L" if center < (width - 1) / 2.0 else "R"


def _semantic_example_ok(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        bg = _get_bg(g)
        marker, marker_color = _find_marker(g, bg)
        reach = _flood_reachable_background(g, bg, marker)
        sprites = _find_sprites(g, bg, marker_color)
        if len(sprites) < 3:
            return False

        reachables = [sprite for sprite in sprites if _sprite_reachable(sprite, reach)]
        unreachables = [sprite for sprite in sprites if not _sprite_reachable(sprite, reach)]
        if not reachables or not unreachables:
            return False

        for sprite in sprites:
            r0, c0, _r1, _c1 = sprite["bbox"]
            patch = sprite["patch"]
            h, w = patch.shape
            dst_c0 = g.shape[1] - w - c0
            if dst_c0 == c0:
                return False
            if np.any(g[r0 : r0 + h, dst_c0 : dst_c0 + w] != bg):
                return False

        reachable_sides = {_side_label(sprite["bbox"], g.shape[1]) for sprite in reachables}
        unreachable_sides = {_side_label(sprite["bbox"], g.shape[1]) for sprite in unreachables}
        if not (reachable_sides & unreachable_sides):
            return False

        out = main(g)
        oracle = _transform(g)
        if not np.array_equal(out, oracle):
            return False

        all_mirrored = _mirror_all_sprites(g, sprites, bg)
        if np.array_equal(out, all_mirrored):
            return False

        if int(np.sum(out != bg)) >= int(np.sum(g != bg)):
            return False

        if not np.array_equal(out[g == BLUE], g[g == BLUE]):
            return False
        if int(out[marker]) != int(g[marker]):
            return False

        for sprite in unreachables:
            for r, c in sprite["coords"]:
                if int(out[r, c]) != bg:
                    return False
        return True
    except Exception:
        return False


def _generate_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(16, 25))
    width = int(_RNG.integers(16, 25))
    bg = int(_RNG.choice(BG_CHOICES))

    grid = np.full((height, width), bg, dtype=int)
    marker = _corner_positions(grid.shape)[int(_RNG.integers(4))]
    grid[marker] = PINK

    wall_mask = np.zeros_like(grid, dtype=bool)
    room_count = 1 + int(_RNG.integers(2))
    placed_rooms = 0
    for _ in range(room_count + 2):
        if placed_rooms >= room_count:
            break
        interior = _place_border_room(grid, wall_mask, marker)
        if interior is not None:
            placed_rooms += 1
    if placed_rooms == 0:
        return None

    if int(_RNG.integers(3)) > 0:
        _place_wall_stub(grid, wall_mask, marker)

    reachable_mask, unreachable_masks = _component_masks(grid, bg, marker)
    usable_unreachable = [mask for mask in unreachable_masks if int(mask.sum()) >= 9]
    if not usable_unreachable or int(reachable_mask.sum()) < 24:
        return None

    reserved = wall_mask.copy()
    reserved[marker] = True

    reachable_target = int(_RNG.integers(2, 5))
    unreachable_target = int(_RNG.integers(1, min(4, len(usable_unreachable) + 2)))

    reachable_placed = 0
    for _ in range(reachable_target * 6):
        patch = _sample_sprite_patch(bg)
        if _place_sprite(grid, reserved, reachable_mask, patch, bg):
            reachable_placed += 1
            if reachable_placed >= reachable_target:
                break
    if reachable_placed < 2:
        return None

    unreachable_placed = 0
    choices = list(range(len(usable_unreachable)))
    while len(choices) < unreachable_target:
        choices.append(int(_RNG.integers(len(usable_unreachable))))
    _RNG.shuffle(choices)
    for idx in choices:
        patch = _sample_sprite_patch(bg)
        if _place_sprite(grid, reserved, usable_unreachable[idx], patch, bg):
            unreachable_placed += 1
    if unreachable_placed < 1:
        return None

    if not _semantic_example_ok(grid):
        return None

    inp_sig = _signature(grid)
    out_sig = _signature(main(grid))
    if inp_sig in _SEEN_INPUT_SIGS:
        return None
    if out_sig in _SEEN_OUTPUT_SIGS:
        return None

    _SEEN_INPUT_SIGS.add(inp_sig)
    _SEEN_OUTPUT_SIGS.add(out_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 可达性镜像 input")
