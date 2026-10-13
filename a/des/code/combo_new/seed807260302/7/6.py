from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 500
REGION_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 120


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_border(comp: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)


def _parse_palette(g: np.ndarray) -> list[int]:
    top = g[0]
    non_bg = np.flatnonzero(top != BG)
    assert non_bg.size >= 2, "expected a top-edge palette strip"
    start = int(non_bg[0])
    stop = int(non_bg[-1]) + 1
    assert np.all(top[start:stop] != BG), "palette strip must be contiguous"
    assert np.all(top[:start] == BG) and np.all(top[stop:] == BG), "top row may only contain palette plus black margins"
    palette = [int(v) for v in top[start:stop]]
    assert len(palette) == len(set(palette)), "palette entries must be distinct"
    return palette


def _grey_component_index(g: np.ndarray) -> tuple[list[set[tuple[int, int]]], np.ndarray]:
    comps = _components(g == GREY)
    comp_sets = [set(comp) for comp in comps]
    ids = -np.ones(g.shape, dtype=int)
    for comp_id, comp in enumerate(comps):
        for r, c in comp:
            ids[r, c] = int(comp_id)
    return comp_sets, ids


def _hole_count_for_interior(
    interior: list[tuple[int, int]],
    grey_sets: list[set[tuple[int, int]]],
    grey_ids: np.ndarray,
    h: int,
    w: int,
) -> int:
    interior_set = set(interior)
    adjacent_ids: set[int] = set()
    for r, c in interior:
        for dr, dc in DIR4:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and int(grey_ids[nr, nc]) >= 0:
                adjacent_ids.add(int(grey_ids[nr, nc]))

    count = 0
    for comp_id in adjacent_ids:
        comp = grey_sets[int(comp_id)]
        enclosed = True
        for r, c in comp:
            for dr, dc in DIR4:
                nr = int(r + dr)
                nc = int(c + dc)
                if (nr, nc) in comp:
                    continue
                if not (0 <= nr < h and 0 <= nc < w):
                    enclosed = False
                    break
                if (nr, nc) not in interior_set:
                    enclosed = False
                    break
            if not enclosed:
                break
        if enclosed:
            count += 1
    return int(count)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each sealed grey room with the palette color indexed by its interior hole count.

    Input structure:
    - Black background.
    - The top row contains one contiguous strip of distinct non-black sample colors,
      optionally with black margins on the left or right.
    - Below the strip are one or more 1-pixel-thick closed grey outlines; each outline
      encloses a black interior and may contain small solid grey 1x1 or 2x2 blocks.

    Rule:
    - Find every 4-connected black component that does not touch the grid border; each
      such component is the interior of one closed region.
    - For each interior, count the adjacent grey connected components that are fully
      surrounded by that interior; those enclosed grey islands are the hole blocks.
    - Use the hole count N as a 0-based index into the top-row sample strip, and fill
      that interior's black cells with the selected sample color.
    - Leave the sample strip, the grey outlines, and the grey hole blocks unchanged.

    Key insight:
    - The small grey islands inside a room are not the regions to fill; their count is
      the lookup index that selects the room's fill color from the top-edge palette.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    palette = _parse_palette(g)
    grey_sets, grey_ids = _grey_component_index(g)

    out = g.copy()
    black_components = _components(g == BG)
    for comp in black_components:
        if _touches_border(comp, h, w):
            continue
        holes = _hole_count_for_interior(comp, grey_sets, grey_ids, h, w)
        assert 0 <= holes < len(palette), "hole count must index the palette"
        color = int(palette[holes])
        for r, c in comp:
            out[r, c] = color
    return out


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _boundary_from_room(room: np.ndarray) -> np.ndarray:
    boundary = np.zeros_like(room, dtype=bool)
    h, w = room.shape
    for r, c in np.argwhere(room):
        rr = int(r)
        cc = int(c)
        for dr, dc in DIR4:
            nr = int(rr + dr)
            nc = int(cc + dc)
            if 0 <= nr < h and 0 <= nc < w and not bool(room[nr, nc]):
                boundary[nr, nc] = True
    return boundary


def _room_base_shape(hole_count: int) -> np.ndarray | None:
    side_bias = 5 + (1 if hole_count >= 2 else 0) + (1 if hole_count >= 4 else 0)
    h = random.randint(side_bias, min(8, side_bias + 2))
    w = random.randint(side_bias, min(8, side_bias + 2))
    mask = np.ones((h, w), dtype=bool)

    max_notches = 1 if hole_count >= 3 else 3
    for _ in range(random.randint(0, max_notches)):
        side = random.choice(("top", "bottom", "left", "right"))
        cand = mask.copy()
        if side in ("top", "bottom"):
            if h < 5 or w < 4:
                continue
            depth = random.randint(1, min(2, h - 3))
            width = random.randint(1, max(1, w - 2))
            start = random.randint(0, w - width)
            rows = slice(0, depth) if side == "top" else slice(h - depth, h)
            cand[rows, start : start + width] = False
        else:
            if w < 5 or h < 4:
                continue
            depth = random.randint(1, min(2, w - 3))
            height = random.randint(1, max(1, h - 2))
            start = random.randint(0, h - height)
            cols = slice(0, depth) if side == "left" else slice(w - depth, w)
            cand[start : start + height, cols] = False
        if cand.sum() < max(12, hole_count * 6):
            continue
        if _is_connected(cand):
            mask = cand

    return mask if _is_connected(mask) else None


def _place_holes(room: np.ndarray, hole_count: int) -> np.ndarray | None:
    holes = np.zeros_like(room, dtype=bool)
    reserved = np.zeros_like(room, dtype=bool)
    h, w = room.shape

    for _ in range(hole_count):
        preferred = random.choice((1, 2))
        placed = False
        for size in (preferred, 1 if preferred == 2 else 2):
            candidates: list[tuple[int, int]] = []
            for r in range(1, h - size - 1):
                for c in range(1, w - size - 1):
                    window = room[r - 1 : r + size + 1, c - 1 : c + size + 1]
                    if not bool(window.all()):
                        continue
                    if bool(reserved[r - 1 : r + size + 1, c - 1 : c + size + 1].any()):
                        continue
                    candidates.append((int(r), int(c)))
            if not candidates:
                continue
            r, c = random.choice(candidates)
            holes[r : r + size, c : c + size] = True
            reserved[r - 1 : r + size + 1, c - 1 : c + size + 1] = True
            placed = True
            break
        if not placed:
            return None

    fill_mask = room & ~holes
    return holes if _is_connected(fill_mask) else None


def _generate_region_patch(hole_count: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(REGION_ATTEMPTS):
        inner = _room_base_shape(hole_count)
        if inner is None:
            continue

        room = np.zeros((inner.shape[0] + 2, inner.shape[1] + 2), dtype=bool)
        room[1:-1, 1:-1] = inner

        holes = _place_holes(room, hole_count)
        if holes is None:
            continue

        fill_mask = room & ~holes
        boundary = _boundary_from_room(room)
        if not bool(boundary.any()):
            continue

        patch = np.full(room.shape, BG, dtype=int)
        patch[boundary | holes] = GREY

        black_inside = [comp for comp in _components(patch == BG) if not _touches_border(comp, patch.shape[0], patch.shape[1])]
        if len(black_inside) != 1:
            continue
        if len(black_inside[0]) != int(fill_mask.sum()):
            continue
        return patch, fill_mask
    return None


def _choose_hole_counts(palette_len: int, region_count: int) -> list[int]:
    if region_count == 1:
        return [random.randrange(palette_len)]
    choices = list(range(palette_len))
    random.shuffle(choices)
    counts = choices[:region_count]
    random.shuffle(counts)
    return [int(v) for v in counts]


def _scene_canvas_sizes(
    patches: list[np.ndarray],
    palette_len: int,
) -> list[tuple[int, int]]:
    max_h = max(int(p.shape[0]) for p in patches)
    max_w = max(int(p.shape[1]) for p in patches)
    padded_area = sum((int(p.shape[0]) + 2) * (int(p.shape[1]) + 2) for p in patches)
    min_h = max(10, max_h + 4)
    min_w = max(10, max(max_w + 2, palette_len + 2))
    candidates = []
    for h in range(min_h, 31):
        for w in range(min_w, 31):
            if h * w >= padded_area + 18:
                candidates.append((int(h), int(w)))
    return candidates


def _place_scene(
    patches: list[np.ndarray],
    fill_masks: list[np.ndarray],
    hole_counts: list[int],
    palette: list[int],
) -> tuple[np.ndarray, np.ndarray] | None:
    candidates = _scene_canvas_sizes(patches, len(palette))
    if not candidates:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        h, w = random.choice(candidates)
        grid = np.full((h, w), BG, dtype=int)
        expected = grid.copy()

        start = random.randint(0, w - len(palette))
        grid[0, start : start + len(palette)] = np.array(palette, dtype=int)
        expected[0, start : start + len(palette)] = np.array(palette, dtype=int)

        reserved = np.zeros((h, w), dtype=bool)
        reserved[:2, :] = True
        placements: list[tuple[int, int, int]] = []

        order = list(range(len(patches)))
        order.sort(key=lambda idx: int(patches[idx].shape[0] * patches[idx].shape[1]), reverse=True)

        ok = True
        for idx in order:
            patch = patches[idx]
            ph, pw = map(int, patch.shape)
            positions: list[tuple[int, int]] = []
            for r0 in range(2, h - ph):
                for c0 in range(1, w - pw):
                    if bool(reserved[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1].any()):
                        continue
                    positions.append((int(r0), int(c0)))
            if not positions:
                ok = False
                break
            r0, c0 = random.choice(positions)
            reserved[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1] = True
            placements.append((idx, r0, c0))

        if not ok:
            continue

        for idx, r0, c0 in placements:
            patch = patches[idx]
            fill_mask = fill_masks[idx]
            ph, pw = map(int, patch.shape)
            region = patch != BG
            grid[r0 : r0 + ph, c0 : c0 + pw][region] = patch[region]
            expected[r0 : r0 + ph, c0 : c0 + pw][region] = patch[region]
            expected[r0 : r0 + ph, c0 : c0 + pw][fill_mask] = int(palette[hole_counts[idx]])

        black_inside = [comp for comp in _components(grid == BG) if not _touches_border(comp, h, w)]
        if len(black_inside) != len(patches):
            continue
        if not np.array_equal(main(grid), expected):
            continue
        return grid, expected
    return None


def _gen_attempt() -> np.ndarray | None:
    palette_len = random.randint(3, 5)
    palette = random.sample(PALETTE_COLORS, palette_len)
    region_count = random.randint(1, 3)
    hole_counts = _choose_hole_counts(palette_len, region_count)

    patches: list[np.ndarray] = []
    fill_masks: list[np.ndarray] = []
    for hole_count in hole_counts:
        generated = _generate_region_patch(hole_count)
        if generated is None:
            return None
        patch, fill_mask = generated
        patches.append(patch)
        fill_masks.append(fill_mask)

    placed = _place_scene(patches, fill_masks, hole_counts, palette)
    if placed is None:
        return None
    grid, _expected = placed
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hole-count palette input")
