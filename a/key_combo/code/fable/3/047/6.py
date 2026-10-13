from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GRAY = 5
DIRS4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
ARC_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    h, w = mask.shape
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q = deque([(sr, sc)])
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


def _rotate_coord(coord: tuple[int, int], turns: int) -> tuple[int, int]:
    r, c = coord
    for _ in range(turns % 4):
        r, c = c, -r
    return int(r), int(c)


def _rotation_steps(src: tuple[int, int], dst: tuple[int, int]) -> int:
    for turns in range(4):
        if _rotate_coord(src, turns) == dst:
            return turns
    raise AssertionError(f"cannot rotate heading {src} to {dst}")


def _find_frame(g: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for color in sorted(set(map(int, g.ravel())) - {BG, GRAY}):
        for comp in _components(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            h, w = r1 - r0 + 1, c1 - c0 + 1
            if h < 4 or w < 4:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros((h, w), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.all(sub[border] == color):
                continue
            if len(comp) != int(np.count_nonzero(border)):
                continue
            inner = sub[1:-1, 1:-1]
            inner_colors = set(map(int, inner.ravel())) - {BG}
            if GRAY not in inner_colors:
                continue
            candidates.append((h * w, int(color), (r0, c0, r1, c1)))

    assert candidates, "expected one hollow rectangular frame"
    candidates.sort(reverse=True)
    if len(candidates) > 1:
        assert candidates[0][0] > candidates[1][0], "ambiguous rectangular frame"
    _, frame_color, box = candidates[0]
    return frame_color, box


def _find_placeholders(
    g: np.ndarray, frame_color: int, frame_box: tuple[int, int, int, int]
) -> list[dict]:
    fr0, fc0, fr1, fc1 = frame_box
    slots: list[dict] = []
    used_colors: set[int] = set()
    used_cells: set[tuple[int, int]] = set()

    for hr in range(fr0 + 1, fr1):
        for hc in range(fc0 + 1, fc1):
            if int(g[hr, hc]) != GRAY:
                continue
            colored_neighbors: list[tuple[int, int, int]] = []
            for nr, nc in _neighbors4(hr, hc):
                if not (fr0 < nr < fr1 and fc0 < nc < fc1):
                    continue
                color = int(g[nr, nc])
                if color not in (BG, GRAY):
                    colored_neighbors.append((nr, nc, color))
            assert len(colored_neighbors) == 1, "each slot gray cell must touch one colored main cell"

            mr, mc, color = colored_neighbors[0]
            gray_neighbors = [
                (nr, nc)
                for nr, nc in _neighbors4(mr, mc)
                if fr0 < nr < fr1 and fc0 < nc < fc1 and int(g[nr, nc]) == GRAY
            ]
            assert gray_neighbors == [(hr, hc)], "each slot main cell must touch only its gray head"
            assert color not in used_colors, "placeholder main colors must be unique"
            assert (mr, mc) not in used_cells and (hr, hc) not in used_cells
            used_colors.add(color)
            used_cells.update({(mr, mc), (hr, hc)})
            slots.append(
                {
                    "color": color,
                    "main": (mr, mc),
                    "head": (hr, hc),
                    "heading": (hr - mr, hc - mc),
                }
            )

    assert slots, "expected at least one pose slot"
    return slots


def _find_sprites(
    g: np.ndarray, frame_box: tuple[int, int, int, int]
) -> dict[int, dict]:
    fr0, fc0, fr1, fc1 = frame_box
    outside = g != BG
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    sprites: dict[int, dict] = {}

    for comp in _components(outside):
        values = [int(g[r, c]) for r, c in comp]
        colors = sorted(set(values) - {GRAY})
        if not colors:
            continue
        assert len(colors) == 1, "each external sprite must have one body color"
        color = colors[0]
        heads = [(r, c) for r, c in comp if int(g[r, c]) == GRAY]
        assert len(heads) == 1, "each external sprite must contain one gray head"
        head = heads[0]
        comp_set = set(comp)
        necks = [
            (nr, nc)
            for nr, nc in _neighbors4(*head)
            if (nr, nc) in comp_set and int(g[nr, nc]) == color
        ]
        assert len(necks) == 1, "sprite gray head must touch exactly one same-color cell"
        neck = necks[0]
        assert color not in sprites, "duplicate external sprite color"
        sprites[color] = {
            "heading": (head[0] - neck[0], head[1] - neck[1]),
            "coords": tuple((r - head[0], c - head[1]) for r, c in comp),
        }

    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each external colored sprite to the pose slot of the same color and stamp it into the frame.

    Input structure:
    - The grid has a black background and one single-color hollow rectangular frame.
    - Inside the frame are two-cell domino pose slots: one colored main cell and one adjacent gray direction cell; main colors are unique.
    - Outside the frame, each slot color has one connected sprite with one embedded gray head pixel.
    - The gray head in a sprite touches exactly one same-color body pixel, defining the sprite's current heading from that body pixel to the head.

    Rule:
    - For every inside slot, find the outside sprite of the same color.
    - Rotate the sprite by 0, 90, 180, or 270 degrees so the sprite heading matches the slot's gray-cell direction from its colored main cell.
    - Clear all slot domino cells to black, then translate the rotated sprite so its gray head lands on the slot gray cell.
    - Stamp every rotated sprite cell into the frame using the sprite color, including the gray head cell; keep the frame and all outside sprites unchanged.

    Key insight:
    - The domino is a pose slot: its color selects a sprite, its gray cell gives both the target orientation and the alignment point.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frame_color, frame_box = _find_frame(g)
    fr0, fc0, fr1, fc1 = frame_box
    slots = _find_placeholders(g, frame_color, frame_box)
    sprites = _find_sprites(g, frame_box)

    out = g.copy()
    for slot in slots:
        out[slot["main"]] = BG
        out[slot["head"]] = BG

    occupied: set[tuple[int, int]] = set()
    for slot in slots:
        color = int(slot["color"])
        assert color in sprites, f"missing external sprite for color {color}"
        sprite = sprites[color]
        turns = _rotation_steps(sprite["heading"], slot["heading"])
        hr, hc = slot["head"]

        rotated = [_rotate_coord(coord, turns) for coord in sprite["coords"]]
        targets = [(hr + dr, hc + dc) for dr, dc in rotated]
        for rr, cc in targets:
            assert fr0 < rr < fr1 and fc0 < cc < fc1, "stamped sprite must remain inside the frame"
            assert (rr, cc) not in occupied, "stamped sprites must not overlap"
            assert int(out[rr, cc]) == BG, "stamped sprite must land on cleared/background interior"
        for rr, cc in targets:
            out[rr, cc] = color
            occupied.add((rr, cc))

    return out


# ---------------------------- generator ----------------------------


def _make_sprite_cells(heading: tuple[int, int]) -> set[tuple[int, int]]:
    neck = (-heading[0], -heading[1])
    cells = {neck, (2 * neck[0], 2 * neck[1])}
    target_size = random.randint(6, 9)

    for _ in range(120):
        if len(cells) >= target_size:
            break
        base = random.choice(tuple(cells))
        random_dirs = list(DIRS4)
        random.shuffle(random_dirs)
        for dr, dc in random_dirs:
            nr, nc = base[0] + dr, base[1] + dc
            cand = (nr, nc)
            if cand == (0, 0) or cand in cells:
                continue
            if abs(nr) > 3 or abs(nc) > 3:
                continue
            if abs(nr) + abs(nc) == 1 and cand != neck:
                continue
            cells.add(cand)
            break

    if len(cells) < 6:
        side_a = (-heading[1], heading[0])
        side_b = (heading[1], -heading[0])
        cells.add((neck[0] + side_a[0], neck[1] + side_a[1]))
        cells.add((2 * neck[0] + side_b[0], 2 * neck[1] + side_b[1]))

    head_neighbors = [p for p in cells if abs(p[0]) + abs(p[1]) == 1]
    assert head_neighbors == [neck], "generated sprite must have one neck"
    return cells


def _sprite_patch(cells: set[tuple[int, int]], color: int) -> np.ndarray:
    all_cells = set(cells) | {(0, 0)}
    r0, c0, r1, c1 = _bbox(list(all_cells))
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in cells:
        patch[r - r0, c - c0] = color
    patch[-r0, -c0] = GRAY
    return patch


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _pad_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out.add((r + dr, c + dc))
    return out


def _place_slots(
    frame_box: tuple[int, int, int, int],
    colors: list[int],
    target_headings: list[tuple[int, int]],
    current_headings: list[tuple[int, int]],
    sprite_cells: list[set[tuple[int, int]]],
) -> list[dict] | None:
    fr0, fc0, fr1, fc1 = frame_box
    ir0, ic0, ir1, ic1 = fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1
    occupied: set[tuple[int, int]] = set()
    slots: list[dict] = []

    order = list(range(len(colors)))
    random.shuffle(order)
    for i in order:
        turns = _rotation_steps(current_headings[i], target_headings[i])
        rel = {_rotate_coord(p, turns) for p in (sprite_cells[i] | {(0, 0)})}
        min_r, min_c, max_r, max_c = _bbox(list(rel))
        low_r, high_r = ir0 - min_r, ir1 - max_r
        low_c, high_c = ic0 - min_c, ic1 - max_c
        if low_r > high_r or low_c > high_c:
            return None

        placed = False
        for _ in range(180):
            hr = random.randint(low_r, high_r)
            hc = random.randint(low_c, high_c)
            stamp_cells = {(hr + dr, hc + dc) for dr, dc in rel}
            padded = _pad_cells(stamp_cells)
            if any(cell in occupied for cell in padded):
                continue
            main = (hr - target_headings[i][0], hc - target_headings[i][1])
            if main not in stamp_cells:
                return None
            slots.append(
                {
                    "index": i,
                    "color": colors[i],
                    "main": main,
                    "head": (hr, hc),
                    "heading": target_headings[i],
                    "stamp_cells": stamp_cells,
                    "turns": turns,
                }
            )
            occupied.update(padded)
            placed = True
            break
        if not placed:
            return None

    slots.sort(key=lambda s: s["index"])
    return slots


def _can_place_bbox(
    reserve: np.ndarray, r0: int, c0: int, h: int, w: int, padding: int = 1
) -> bool:
    h_grid, w_grid = reserve.shape
    if r0 < 0 or c0 < 0 or r0 + h > h_grid or c0 + w > w_grid:
        return False
    pr0 = max(0, r0 - padding)
    pc0 = max(0, c0 - padding)
    pr1 = min(h_grid, r0 + h + padding)
    pc1 = min(w_grid, c0 + w + padding)
    return not bool(np.any(reserve[pr0:pr1, pc0:pc1]))


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    for rr in range(patch.shape[0]):
        for cc in range(patch.shape[1]):
            val = int(patch[rr, cc])
            if val != BG:
                grid[r0 + rr, c0 + cc] = val


def _place_external_sprites(
    grid: np.ndarray,
    reserve: np.ndarray,
    patches: list[np.ndarray],
) -> bool:
    order = sorted(range(len(patches)), key=lambda i: patches[i].size, reverse=True)
    for i in order:
        patch = patches[i]
        ph, pw = patch.shape
        choices = [(r, c) for r in range(grid.shape[0] - ph + 1) for c in range(grid.shape[1] - pw + 1)]
        random.shuffle(choices)
        placed = False
        for r, c in choices:
            if not _can_place_bbox(reserve, r, c, ph, pw, padding=1):
                continue
            _paste_patch(grid, patch, r, c)
            reserve[max(0, r - 1) : min(grid.shape[0], r + ph + 1), max(0, c - 1) : min(grid.shape[1], c + pw + 1)] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _expected_from_metadata(grid: np.ndarray, slots: list[dict]) -> np.ndarray:
    out = grid.copy()
    for slot in slots:
        out[slot["main"]] = BG
        out[slot["head"]] = BG
    for slot in slots:
        for r, c in slot["stamp_cells"]:
            out[r, c] = int(slot["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(28, 30)
    width = random.randint(28, 30)
    frame_h = random.randint(14, 17)
    frame_w = random.randint(14, 17)
    if height - frame_h < 10 or width - frame_w < 10:
        return None

    fr0 = random.randint(5, height - frame_h - 5)
    fc0 = random.randint(5, width - frame_w - 5)
    frame_box = (fr0, fc0, fr0 + frame_h - 1, fc0 + frame_w - 1)

    frame_color = random.choice(ARC_COLORS)
    usable = [c for c in ARC_COLORS if c != frame_color]
    count = random.choice([3, 4, 4, 5])
    colors = random.sample(usable, count)

    target_headings = list(DIRS4)
    random.shuffle(target_headings)
    while len(target_headings) < count:
        target_headings.append(random.choice(DIRS4))
    target_headings = target_headings[:count]

    if count >= 4:
        turns_needed = [0, 1, 2, 3] + [random.randrange(4) for _ in range(count - 4)]
    else:
        turns_needed = [0] + random.sample([1, 2, 3], count - 1)
    random.shuffle(turns_needed)

    current_headings = [
        _rotate_coord(target_headings[i], -turns_needed[i]) for i in range(count)
    ]
    sprite_cells = [_make_sprite_cells(current_headings[i]) for i in range(count)]

    slots = _place_slots(frame_box, colors, target_headings, current_headings, sprite_cells)
    if slots is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    _draw_frame(grid, frame_box, frame_color)
    for slot in slots:
        grid[slot["main"]] = int(slot["color"])
        grid[slot["head"]] = GRAY

    reserve = np.zeros((height, width), dtype=bool)
    fr0, fc0, fr1, fc1 = frame_box
    reserve[max(0, fr0 - 1) : min(height, fr1 + 2), max(0, fc0 - 1) : min(width, fc1 + 2)] = True
    patches = [_sprite_patch(sprite_cells[i], colors[i]) for i in range(count)]
    if not _place_external_sprites(grid, reserve, patches):
        return None

    expected = _expected_from_metadata(grid, slots)
    if np.count_nonzero(expected != grid) < count * 4:
        return None
    if len(set(target_headings)) < min(3, count):
        return None
    if not any(t == 0 for t in turns_needed) or not any(t != 0 for t in turns_needed):
        return None

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate pose-slot stamping input")
