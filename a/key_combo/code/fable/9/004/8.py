from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
GREY = 5
WALL = 4
SPRITE_COLORS = (1, 2, 3, 6, 7, 8, 9)
DIM_OPTIONS = ((1, 3), (1, 5), (1, 7), (3, 3), (3, 5), (3, 7), (5, 5))
GEN_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = map(int, patch.shape)
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def extract_rooms(grid: np.ndarray, bg: int) -> list[dict]:
    passable = (grid == bg) | (grid == GREY)
    rooms: list[dict] = []
    for coords in components_from_mask(passable):
        cells = {tuple(map(int, rc)) for rc in coords}
        if any(r == 0 or c == 0 or r == grid.shape[0] - 1 or c == grid.shape[1] - 1 for r, c in cells):
            continue
        markers = [(r, c) for r, c in cells if int(grid[r, c]) == GREY]
        if len(markers) != 1:
            continue
        boundary: set[tuple[int, int]] = set()
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) not in cells:
                    boundary.add((nr, nc))
        if not boundary or any(int(grid[r, c]) in (bg, GREY) for r, c in boundary):
            continue
        marker = markers[0]
        rooms.append({"cells": cells, "marker": marker, "area": len(cells)})
    return rooms


def extract_sprites(grid: np.ndarray, bg: int, rooms: list[dict]) -> list[dict]:
    sprite_mask = np.zeros_like(grid, dtype=bool)
    room_cells = set().union(*(room["cells"] for room in rooms))
    sprites = []
    foreground_mask = (grid != bg) & (grid != GREY)
    for coords in components_from_mask(foreground_mask):
        is_wall = False
        for r, c in coords:
            for nr, nc in neighbors4(int(r), int(c)):
                if (nr, nc) in room_cells:
                    is_wall = True
                    break
            if is_wall:
                break
        if is_wall:
            continue
        for r, c in coords:
            sprite_mask[int(r), int(c)] = True

    for coords in components_from_mask(sprite_mask):
        r0, c0, r1, c1 = bbox_coords(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprites.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "area": int(coords.shape[0]),
                "patch": patch,
            }
        )
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each sprite into the unique closed room whose cavity area matches the sprite's pixel count.
    
    Input structure:
    - A black background contains several foreground sprites with odd-height and odd-width bounding boxes.
    - Several closed rooms are outlined by walls; each room's empty cavity contains exactly one gray marker pixel at its center.
    - The original sprites sit away from the rooms, and the room walls remain visible in the scene.
    
    Rule:
    - For each sprite, count its foreground pixels and find the unique room whose cavity size equals that count, including the gray marker cell.
    - Copy the sprite's bounding-box patch so its center lands on that room's gray marker, treating background cells inside the patch as transparent.
    - Keep the original sprites and all room walls unchanged while replacing each gray marker with the stamped copy.
    
    Key insight:
    - The gray marker only gives the landing center; the actual address of each destination room is its cavity area, not its distance, color, or shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    rooms = extract_rooms(g, bg)
    assert rooms, "expected at least one enclosed room"
    room_by_area = {int(room["area"]): room for room in rooms}
    assert len(room_by_area) == len(rooms), "room cavity areas must be unique"

    sprites = extract_sprites(g, bg, rooms)
    assert sprites, "expected at least one sprite"

    out = g.copy()
    for sprite in sprites:
        area = int(sprite["area"])
        assert area in room_by_area, "every sprite must match a room by area"
        room = room_by_area[area]
        marker_r, marker_c = room["marker"]
        r0, c0, r1, c1 = sprite["bbox"]
        patch = sprite["patch"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h % 2 == 1 and w % 2 == 1
        paste_r = marker_r - h // 2
        paste_c = marker_c - w // 2
        out[marker_r, marker_c] = bg
        paste(out, patch, paste_r, paste_c, transparent=bg)
    return out


def _make_sprite(h: int, w: int, primary: int, extras: list[int]) -> np.ndarray:
    patch = np.full((h, w), primary, dtype=int)
    if not extras:
        return patch
    style = random.choice(("rows", "cols", "checker", "corners", "random"))
    colors = [primary] + extras
    if style == "rows":
        for r in range(h):
            patch[r, :] = colors[r % len(colors)]
    elif style == "cols":
        for c in range(w):
            patch[:, c] = colors[c % len(colors)]
    elif style == "checker":
        for r in range(h):
            for c in range(w):
                patch[r, c] = colors[(r + c) % len(colors)]
    elif style == "corners":
        corner_cells = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
        for idx, cell in enumerate(corner_cells):
            patch[cell] = colors[idx % len(colors)]
    else:
        for color in extras:
            rr = random.randrange(h)
            cc = random.randrange(w)
            patch[rr, cc] = color
        noise_cells = max(0, (h * w) // 4 - len(extras))
        for _ in range(noise_cells):
            rr = random.randrange(h)
            cc = random.randrange(w)
            patch[rr, cc] = random.choice(colors)
    patch[h // 2, w // 2] = primary
    return patch


def _choose_sprite_specs() -> list[dict]:
    count = random.randint(2, 4)
    dims = random.sample(DIM_OPTIONS, count)
    primary_colors = random.sample(SPRITE_COLORS, count)
    specs = []
    for idx, ((h, w), primary) in enumerate(zip(dims, primary_colors)):
        remaining = [c for c in SPRITE_COLORS if c != primary]
        extra_count = random.randint(0, min(2, len(remaining), h * w - 1))
        extras = random.sample(remaining, extra_count)
        patch = _make_sprite(h, w, primary, extras)
        specs.append({"id": idx, "dims": (h, w), "patch": patch, "area": h * w})
    return specs


def _place_rectangles(
    shape: tuple[int, int], items: list[tuple[str, int, int, int]], margin: int = 1
) -> dict[tuple[str, int], tuple[int, int]] | None:
    h, w = shape
    occ = np.zeros((h, w), dtype=bool)
    placed: dict[tuple[str, int], tuple[int, int]] = {}
    for kind, idx, ih, iw in sorted(items, key=lambda item: item[2] * item[3], reverse=True):
        candidates = []
        for r0 in range(h - ih + 1):
            for c0 in range(w - iw + 1):
                rr0 = max(0, r0 - margin)
                cc0 = max(0, c0 - margin)
                rr1 = min(h, r0 + ih + margin)
                cc1 = min(w, c0 + iw + margin)
                if not occ[rr0:rr1, cc0:cc1].any():
                    candidates.append((r0, c0))
        if not candidates:
            return None
        pos = random.choice(candidates)
        r0, c0 = pos
        rr0 = max(0, r0 - margin)
        cc0 = max(0, c0 - margin)
        rr1 = min(h, r0 + ih + margin)
        cc1 = min(w, c0 + iw + margin)
        occ[rr0:rr1, cc0:cc1] = True
        placed[(kind, idx)] = pos
    return placed


def _nearest_room_profile(specs: list[dict], placed: dict[tuple[str, int], tuple[int, int]]) -> tuple[bool, bool]:
    room_centers = {}
    for spec in specs:
        h, w = spec["dims"]
        r0, c0 = placed[("room", spec["id"])]
        room_centers[spec["id"]] = (r0 + 1 + h // 2, c0 + 1 + w // 2)

    any_mismatch = False
    any_match = False
    for spec in specs:
        h, w = spec["dims"]
        r0, c0 = placed[("sprite", spec["id"])]
        center = (r0 + h // 2, c0 + w // 2)
        distances = []
        for other in specs:
            mr, mc = room_centers[other["id"]]
            dist = abs(center[0] - mr) + abs(center[1] - mc)
            distances.append((dist, other["id"]))
        distances.sort()
        if len(distances) > 1 and distances[0][0] == distances[1][0]:
            return False, False
        nearest = distances[0][1]
        if nearest == spec["id"]:
            any_match = True
        else:
            any_mismatch = True
    return any_mismatch, any_match


def _render_output(shape: tuple[int, int], specs: list[dict], placed: dict[tuple[str, int], tuple[int, int]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for spec in specs:
        h, w = spec["dims"]
        patch = spec["patch"]
        sr, sc = placed[("sprite", spec["id"])]
        paste(grid, patch, sr, sc)

        rr, rc = placed[("room", spec["id"])]
        grid[rr : rr + h + 2, rc : rc + w + 2] = WALL
        grid[rr + 1 : rr + 1 + h, rc + 1 : rc + 1 + w] = BG
        paste(grid, patch, rr + 1, rc + 1)
    return grid


def _derive_input(output: np.ndarray, specs: list[dict], placed: dict[tuple[str, int], tuple[int, int]]) -> np.ndarray:
    inp = output.copy()
    for spec in specs:
        h, w = spec["dims"]
        rr, rc = placed[("room", spec["id"])]
        inp[rr + 1 : rr + 1 + h, rc + 1 : rc + 1 + w] = BG
        inp[rr + 1 + h // 2, rc + 1 + w // 2] = GREY
    return inp


def _gen_attempt() -> np.ndarray | None:
    specs = _choose_sprite_specs()
    grid_shape = (random.randint(18, 30), random.randint(18, 30))
    items = []
    for spec in specs:
        h, w = spec["dims"]
        items.append(("sprite", spec["id"], h, w))
        items.append(("room", spec["id"], h + 2, w + 2))
    placed = _place_rectangles(grid_shape, items, margin=1)
    if placed is None:
        return None

    any_mismatch, any_match = _nearest_room_profile(specs, placed)
    if not any_mismatch:
        return None
    if len(specs) >= 3 and not any_match:
        return None

    output = _render_output(grid_shape, specs, placed)
    inp = _derive_input(output, specs, placed)
    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
