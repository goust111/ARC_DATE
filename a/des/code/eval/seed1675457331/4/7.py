from __future__ import annotations

import random
from collections import deque

import numpy as np


def _majority_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
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
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _room_spec_from_component(
    grid: np.ndarray, color: int, coords: np.ndarray, bg: int
) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if h < 5 or w < 5:
        return None

    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if np.any(local & ~border):
        return None

    missing = np.argwhere(border & ~local)
    if missing.shape[0] != 1:
        return None

    mr, mc = map(int, missing[0])
    if (mr in (0, h - 1)) and (mc in (0, w - 1)):
        return None

    if mr == 0:
        side = "top"
    elif mr == h - 1:
        side = "bottom"
    elif mc == 0:
        side = "left"
    else:
        side = "right"

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    if inner.size == 0:
        return None

    inner_nonbg = inner != bg
    if not inner_nonbg.any():
        return None

    # Real rooms have a sprite floating in background; reject fully filled interiors.
    if not np.any(inner == bg):
        return None

    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "side": side,
    }


def _detect_rooms(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    colors = [int(v) for v in np.unique(grid) if int(v) != bg]
    candidates: list[tuple[int, int, list[dict[str, object]]]] = []

    for color in colors:
        mask = grid == color
        comps = _components(mask)
        specs: list[dict[str, object]] = []
        covered = 0
        for comp in comps:
            spec = _room_spec_from_component(grid, color, comp, bg)
            if spec is None:
                continue
            specs.append(spec)
            covered += int(comp.shape[0])
        if specs and covered == int(mask.sum()):
            candidates.append((len(specs), covered, specs))

    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return candidates[0][2]

    rooms: list[dict[str, object]] = []
    for color in colors:
        for comp in _components(grid == color):
            spec = _room_spec_from_component(grid, color, comp, bg)
            if spec is not None:
                rooms.append(spec)
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove each room's interior sprite, flip it across the wall-with-gap side,
    and place the flipped sprite just outside that side while keeping the
    perpendicular alignment unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _majority_color(g)
    rooms = _detect_rooms(g, bg)
    assert rooms, "expected at least one room"

    out = g.copy()
    moves: list[tuple[np.ndarray, str, int, int, int, int, int, int]] = []

    for room in rooms:
        wall_color = int(room["color"])
        r0, c0, r1, c1 = room["bbox"]  # inclusive
        side = str(room["side"])

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        sprite_mask = (inner != bg) & (inner != wall_color)
        assert sprite_mask.any(), "room interior must contain a sprite"

        sprite_comps = _components(sprite_mask)
        assert len(sprite_comps) == 1, "sprite must be a single 4-connected object"
        sr0, sc0, sr1, sc1 = _bbox(sprite_comps[0])
        sprite_patch = inner[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()

        inner_out = out[r0 + 1 : r1, c0 + 1 : c1]
        inner_out[sprite_mask] = bg

        global_sr0 = r0 + 1 + sr0
        global_sc0 = c0 + 1 + sc0
        moves.append((sprite_patch, side, r0, c0, r1, c1, global_sr0, global_sc0))

    for sprite_patch, side, r0, c0, r1, c1, global_sr0, global_sc0 in moves:
        sh, sw = sprite_patch.shape
        if side == "top":
            flipped = np.flipud(sprite_patch)
            tr, tc = r0 - sh, global_sc0
        elif side == "bottom":
            flipped = np.flipud(sprite_patch)
            tr, tc = r1 + 1, global_sc0
        elif side == "left":
            flipped = np.fliplr(sprite_patch)
            tr, tc = global_sr0, c0 - sw
        else:
            flipped = np.fliplr(sprite_patch)
            tr, tc = global_sr0, c1 + 1

        assert 0 <= tr and 0 <= tc
        assert tr + sh <= out.shape[0] and tc + sw <= out.shape[1]
        mask = flipped != bg
        dest = out[tr : tr + sh, tc : tc + sw]
        assert not np.any(mask & (dest != bg)), "reflected sprite must land on background"
        dest[mask] = flipped[mask]

    return out


_BG = 0
_WALL = 1
_SPRITE_COLORS = (2, 3, 4, 5, 6, 7, 8, 9)
_SIDES = ("top", "bottom", "left", "right")
_GEN_RETRIES = 400


def _random_connected_mask(max_h: int, max_w: int, rng: random.Random) -> np.ndarray:
    for _ in range(80):
        h = rng.randint(1, max_h)
        w = rng.randint(1, max_w)
        area = h * w
        if area == 1:
            target = 1
        else:
            low = 2
            high = area if area <= 4 else area - 1
            target = rng.randint(low, high)

        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(tuple(frontier)))

        if len(cells) != target:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        pts = np.argwhere(mask)
        r0, c0, r1, c1 = _bbox(pts)
        cropped = mask[r0 : r1 + 1, c0 : c1 + 1]

        if cropped.size >= 4 and np.all(cropped):
            continue
        return cropped

    raise RuntimeError("failed to generate a connected sprite mask")


def _colorize_mask(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    arr = np.full(mask.shape, _BG, dtype=int)
    cells = [tuple(map(int, pt)) for pt in np.argwhere(mask)]
    color_count = min(len(cells), rng.randint(1, 3))
    chosen_colors = rng.sample(_SPRITE_COLORS, color_count)
    seeds = rng.sample(cells, color_count)
    for r, c in cells:
        distances = [abs(r - sr) + abs(c - sc) for sr, sc in seeds]
        best = min(range(color_count), key=lambda idx: (distances[idx], idx))
        arr[r, c] = chosen_colors[best]
    return arr


def _build_room_patch(rng: random.Random) -> np.ndarray:
    side = rng.choice(_SIDES)

    while True:
        room_h = rng.randint(6, 11)
        room_w = rng.randint(6, 11)
        inner_h = room_h - 2
        inner_w = room_w - 2
        max_sprite_h = min(5, inner_h - 2)
        max_sprite_w = min(5, inner_w - 2)
        if max_sprite_h < 1 or max_sprite_w < 1:
            continue

        sprite_mask = _random_connected_mask(max_sprite_h, max_sprite_w, rng)
        sh, sw = sprite_mask.shape
        if sh > inner_h - 2 or sw > inner_w - 2:
            continue

        sprite = _colorize_mask(sprite_mask, rng)
        sprite_top = rng.randint(1, inner_h - sh - 1)
        sprite_left = rng.randint(1, inner_w - sw - 1)

        pad = 1
        if side == "top":
            patch_h = sh + room_h + 2 * pad
            patch_w = room_w + 2 * pad
            room_top = sh + pad
            room_left = pad
        elif side == "bottom":
            patch_h = room_h + sh + 2 * pad
            patch_w = room_w + 2 * pad
            room_top = pad
            room_left = pad
        elif side == "left":
            patch_h = room_h + 2 * pad
            patch_w = sw + room_w + 2 * pad
            room_top = pad
            room_left = sw + pad
        else:
            patch_h = room_h + 2 * pad
            patch_w = room_w + sw + 2 * pad
            room_top = pad
            room_left = pad

        patch = np.full((patch_h, patch_w), _BG, dtype=int)
        patch[room_top, room_left : room_left + room_w] = _WALL
        patch[room_top + room_h - 1, room_left : room_left + room_w] = _WALL
        patch[room_top : room_top + room_h, room_left] = _WALL
        patch[room_top : room_top + room_h, room_left + room_w - 1] = _WALL

        if side in ("top", "bottom"):
            gap_col = rng.randint(1, room_w - 2)
            gap_row = room_top if side == "top" else room_top + room_h - 1
            patch[gap_row, room_left + gap_col] = _BG
        else:
            gap_row = rng.randint(1, room_h - 2)
            gap_col = room_left if side == "left" else room_left + room_w - 1
            patch[room_top + gap_row, gap_col] = _BG

        rr = room_top + 1 + sprite_top
        cc = room_left + 1 + sprite_left
        sub = patch[rr : rr + sh, cc : cc + sw]
        sub[sprite_mask] = sprite[sprite_mask]
        return patch


def _split_rows(items: list[np.ndarray], row_count: int, rng: random.Random) -> list[list[np.ndarray]]:
    if row_count == 1:
        return [items]
    cuts = sorted(rng.sample(range(1, len(items)), row_count - 1))
    rows = []
    start = 0
    for cut in cuts + [len(items)]:
        rows.append(items[start:cut])
        start = cut
    return rows


def _pack_patches(patches: list[np.ndarray], rng: random.Random) -> np.ndarray | None:
    items = patches[:]
    rng.shuffle(items)
    row_count = rng.randint(1, min(2, len(items)))
    rows = _split_rows(items, row_count, rng)

    outer_top = rng.randint(1, 3)
    outer_bottom = rng.randint(1, 3)
    outer_left = rng.randint(1, 3)
    outer_right = rng.randint(1, 3)
    row_gaps = [rng.randint(1, 3) for _ in range(max(0, len(rows) - 1))]

    row_heights = [max(p.shape[0] for p in row) for row in rows]
    row_widths = []
    per_row_xgaps = []
    for row in rows:
        gaps = [rng.randint(1, 3) for _ in range(max(0, len(row) - 1))]
        per_row_xgaps.append(gaps)
        row_widths.append(sum(p.shape[1] for p in row) + sum(gaps))

    total_h = outer_top + outer_bottom + sum(row_heights) + sum(row_gaps)
    total_w = outer_left + outer_right + max(row_widths)
    if total_h > 30 or total_w > 30:
        return None

    grid = np.full((total_h, total_w), _BG, dtype=int)
    y = outer_top
    for row_idx, row in enumerate(rows):
        row_h = row_heights[row_idx]
        row_w = row_widths[row_idx]
        x = outer_left + rng.randint(0, total_w - outer_left - outer_right - row_w)
        for idx, patch in enumerate(row):
            ph, pw = patch.shape
            py = y + rng.randint(0, row_h - ph)
            grid[py : py + ph, x : x + pw] = patch
            if idx < len(per_row_xgaps[row_idx]):
                x += pw + per_row_xgaps[row_idx][idx]
        y += row_h
        if row_idx < len(row_gaps):
            y += row_gaps[row_idx]

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    patch_count = rng.randint(1, 3)
    patches = [_build_room_patch(rng) for _ in range(patch_count)]
    grid = _pack_patches(patches, rng)
    if grid is None:
        return None
    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
