from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
WALL = 5
WATER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise AssertionError("expected non-empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _flood_mask(passable: np.ndarray, starts: list[tuple[int, int]]) -> np.ndarray:
    h, w = passable.shape
    out = np.zeros((h, w), dtype=bool)
    q = deque()
    for r, c in starts:
        if 0 <= r < h and 0 <= c < w and passable[r, c] and not out[r, c]:
            out[r, c] = True
            q.append((r, c))
    while q:
        rr, cc = q.popleft()
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not out[nr, nc]:
                out[nr, nc] = True
                q.append((nr, nc))
    return out


def _parse_scene(grid: np.ndarray, *, require_droplet: bool = True) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    counts = Counter(map(int, g.ravel()))
    bg = int(counts.most_common(1)[0][0])
    fg_colors = [int(c) for c in counts if int(c) != bg]
    if require_droplet:
        assert len(fg_colors) == 2, f"expected exactly two non-background colors, got {fg_colors}"
        fg_colors.sort(key=lambda c: counts[c])
        water_color = int(fg_colors[0])
        wall_color = int(fg_colors[1])
    else:
        assert fg_colors == [WALL], f"expected wall-only construction grid, got {fg_colors}"
        water_color = None
        wall_color = WALL

    wall_mask = g == wall_color
    wall_comps = _components(wall_mask)
    assert wall_comps, "expected wall component"
    wall_comp = max(wall_comps, key=len)
    comp_mask = np.zeros_like(wall_mask, dtype=bool)
    for r, c in wall_comp:
        comp_mask[r, c] = True
    r0, c0, r1, c1 = _bbox_from_mask(comp_mask)
    assert r1 - r0 >= 4 and c1 - c0 >= 6, "container too small"

    opening_cols = [
        c
        for c in range(c0 + 1, c1)
        if int(g[r0, c]) == bg and int(g[r0 + 1, c]) == bg
    ]
    assert opening_cols, "expected at least one chamber mouth"

    passable = np.zeros_like(g, dtype=bool)
    passable[r0 + 1 : r1, c0 + 1 : c1] = g[r0 + 1 : r1, c0 + 1 : c1] == bg
    starts = [(r0 + 1, c) for c in opening_cols]
    cavity_mask = _flood_mask(passable, starts)
    assert int(cavity_mask.sum()) > 0, "expected non-empty cavity"

    if water_color is None:
        droplet_mask = np.zeros_like(g, dtype=bool)
    else:
        droplet_mask = g == water_color
        droplet_comps = _components(droplet_mask)
        assert len(droplet_comps) == 1, "expected one droplet component"

    opening_groups: list[tuple[int, int]] = []
    start = opening_cols[0]
    prev = start
    for c in opening_cols[1:]:
        if c == prev + 1:
            prev = c
            continue
        opening_groups.append((start, prev))
        start = prev = c
    opening_groups.append((start, prev))

    return {
        "grid": g,
        "bg": bg,
        "wall_color": wall_color,
        "water_color": water_color,
        "wall_bbox": (r0, c0, r1, c1),
        "cavity_mask": cavity_mask,
        "droplet_mask": droplet_mask,
        "opening_groups": opening_groups,
        "opening_cols": opening_cols,
    }


def _fill_mask_for_volume(cavity_mask: np.ndarray, volume: int) -> np.ndarray:
    rows = np.where(cavity_mask.any(axis=1))[0]
    assert rows.size > 0
    row_ids = np.arange(cavity_mask.shape[0])[:, None]
    matches: dict[bytes, np.ndarray] = {}
    for level in rows.tolist():
        fill = cavity_mask & (row_ids >= int(level))
        if int(fill.sum()) == int(volume):
            matches[fill.tobytes()] = fill.copy()
    assert len(matches) == 1, f"expected a unique horizontal fill for volume {volume}, got {len(matches)}"
    return next(iter(matches.values()))


def _chamber_index_for_col(opening_groups: list[tuple[int, int]], col: int) -> int | None:
    for idx, (c0, c1) in enumerate(opening_groups):
        if c0 <= col <= c1:
            return idx
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pour the hanging droplet into a bottom-connected multi-chamber vessel and fill all cavity cells below the common waterline.

    Input structure:
    - A grey wall structure sits on a black background and forms one container with several upward-open vertical chambers.
    - The chambers are connected through lower internal passages, and some chambers contain bottom protrusions or steps.
    - Exactly one connected colored droplet hangs above one chamber mouth; its pixel count is the water volume.

    Rule:
    - Count the droplet pixels to get volume n.
    - Identify the connected inner cavity reachable from the chamber mouths inside the wall bounding box.
    - Choose the unique horizontal level whose strictly-below cavity cells total exactly n, erase the droplet, and recolor those cavity cells with the droplet color.

    Key insight:
    - The droplet does not stay in its source chamber: bottom connectivity forces one shared waterline, so the answer is determined jointly by volume conservation and a global equal-height fill across all chambers.
    """
    info = _parse_scene(np.asarray(grid, dtype=int))
    g = info["grid"]
    bg = int(info["bg"])
    water_color = int(info["water_color"])
    volume = int(info["droplet_mask"].sum())
    fill_mask = _fill_mask_for_volume(info["cavity_mask"], volume)

    out = g.copy()
    out[info["droplet_mask"]] = bg
    out[fill_mask] = water_color
    return out


def _grow_connected_blob(height: int, width: int, area: int, rng: random.Random) -> np.ndarray | None:
    if area <= 0 or area > height * width:
        return None
    for _ in range(80):
        mask = np.zeros((height, width), dtype=bool)
        seed_r = rng.randrange(height)
        seed_c = rng.randrange(width)
        mask[seed_r, seed_c] = True
        frontier = [(seed_r, seed_c)]
        while int(mask.sum()) < area and frontier:
            rr, cc = frontier[rng.randrange(len(frontier))]
            candidates = []
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    candidates.append((nr, nc))
            if not candidates:
                frontier.remove((rr, cc))
                continue
            nr, nc = rng.choice(candidates)
            mask[nr, nc] = True
            frontier.append((nr, nc))
        if int(mask.sum()) == area:
            return mask
    return None


def _build_attempt(rng: random.Random) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    chamber_count = rng.randint(3, 5)
    chamber_widths = [rng.randint(2, 5) for _ in range(chamber_count)]
    inner_h = rng.randint(9, 14)
    outer_w = 2 + sum(chamber_widths) + (chamber_count - 1)
    outer_h = inner_h + 2

    top_margin = rng.randint(4, 7)
    bottom_margin = rng.randint(1, 3)
    left_margin = rng.randint(1, 3)
    right_margin = rng.randint(1, 3)
    h = top_margin + outer_h + bottom_margin
    w = left_margin + outer_w + right_margin
    if h > 30 or w > 30:
        return None

    grid = np.full((h, w), BG, dtype=int)
    top = top_margin
    left = left_margin
    bottom = top + outer_h - 1
    right = left + outer_w - 1

    grid[top:bottom + 1, left] = WALL
    grid[top:bottom + 1, right] = WALL
    grid[bottom, left:right + 1] = WALL
    grid[top, left:right + 1] = WALL

    chamber_cols: list[tuple[int, int]] = []
    partition_cols: list[int] = []
    cur = left + 1
    for idx, width_i in enumerate(chamber_widths):
        c0 = cur
        c1 = cur + width_i - 1
        chamber_cols.append((c0, c1))
        grid[top, c0:c1 + 1] = BG
        cur = c1 + 1
        if idx != chamber_count - 1:
            partition_cols.append(cur)
            cur += 1

    max_gap = min(3, inner_h - 2)
    for col in partition_cols:
        gap_h = rng.randint(1, max_gap)
        part_end = bottom - gap_h - 1
        grid[top:part_end + 1, col] = WALL

    obstacle_count = 0
    for c0, c1 in chamber_cols:
        width_i = c1 - c0 + 1
        if width_i < 2 or rng.random() < 0.2:
            continue
        anchor_left = rng.random() < 0.5
        block_w = rng.randint(1, width_i - 1)
        block_h = rng.randint(1, min(4, inner_h - 3))
        if anchor_left:
            bc0 = c0
            bc1 = c0 + block_w - 1
        else:
            bc1 = c1
            bc0 = c1 - block_w + 1
        br0 = bottom - block_h
        grid[br0:bottom, bc0:bc1 + 1] = WALL
        obstacle_count += 1

        if width_i >= 4 and block_w >= 2 and block_h >= 2 and rng.random() < 0.45:
            step_w = rng.randint(1, block_w - 1)
            step_h = rng.randint(1, block_h - 1)
            if anchor_left:
                sc0 = bc0
                sc1 = sc0 + step_w - 1
            else:
                sc1 = bc1
                sc0 = sc1 - step_w + 1
            sr0 = br0 - step_h
            if sr0 > top + 1:
                grid[sr0:br0, sc0:sc1 + 1] = WALL

    if obstacle_count == 0:
        return None
    return grid, chamber_cols


def _pick_level_and_source(
    grid: np.ndarray,
    chamber_cols: list[tuple[int, int]],
    rng: random.Random,
) -> tuple[int, int, np.ndarray] | None:
    info = _parse_scene(grid, require_droplet=False)
    cavity_mask = info["cavity_mask"]
    r0, _c0, r1, _c1 = info["wall_bbox"]
    rows = np.where(cavity_mask.any(axis=1))[0].tolist()
    if any(int(cavity_mask[r].sum()) == 0 for r in range(r0 + 1, r1)):
        return None

    chamber_masks = []
    for c0, c1 in chamber_cols:
        mask = np.zeros_like(cavity_mask, dtype=bool)
        mask[:, c0:c1 + 1] = cavity_mask[:, c0:c1 + 1]
        chamber_masks.append(mask)

    candidates: list[tuple[int, int, np.ndarray]] = []
    for level in rows:
        fill = cavity_mask & (np.arange(grid.shape[0])[:, None] >= level)
        volume = int(fill.sum())
        if volume < 4 or volume > 16:
            continue
        chamber_fill_counts = [int((fill & mask).sum()) for mask in chamber_masks]
        touched = [idx for idx, cnt in enumerate(chamber_fill_counts) if cnt > 0]
        if len(touched) < 2:
            continue
        for src_idx, (c0, c1) in enumerate(chamber_cols):
            if volume > (r0 * (c1 - c0 + 1)):
                continue
            only_source = chamber_masks[src_idx] & (np.arange(grid.shape[0])[:, None] >= level)
            if np.array_equal(only_source, fill):
                continue
            naive = np.zeros_like(fill, dtype=bool)
            src_cells = [(r, c) for r, c in np.argwhere(chamber_masks[src_idx])]
            src_cells.sort(key=lambda rc: (-rc[0], rc[1]))
            for rr, cc in src_cells[:volume]:
                naive[rr, cc] = True
            if np.array_equal(naive, fill):
                continue
            candidates.append((level, src_idx, fill.copy()))

    if not candidates:
        return None
    return rng.choice(candidates)


def _place_droplet(
    grid: np.ndarray,
    chamber_cols: list[tuple[int, int]],
    src_idx: int,
    volume: int,
    water_color: int,
    rng: random.Random,
) -> np.ndarray | None:
    out = grid.copy()
    r0 = _bbox_from_mask(out == WALL)[0]
    c0, c1 = chamber_cols[src_idx]
    box_h = r0
    box_w = c1 - c0 + 1
    blob = _grow_connected_blob(box_h, box_w, volume, rng)
    if blob is None:
        return None
    out[:r0, c0:c1 + 1][blob] = water_color
    return out


def _validate_generated(grid: np.ndarray, chamber_cols: list[tuple[int, int]]) -> bool:
    info = _parse_scene(grid)
    out = main(grid)
    water_color = int(info["water_color"])
    volume = int(info["droplet_mask"].sum())
    fill_mask = out == water_color
    fill_mask[info["droplet_mask"]] = False
    cavity_fill = fill_mask & info["cavity_mask"]
    if int(cavity_fill.sum()) != volume:
        return False
    if np.any(fill_mask & ~info["cavity_mask"] & ~info["droplet_mask"]):
        return False

    chamber_hits = 0
    source_idx = None
    droplet_cols = np.where(info["droplet_mask"].any(axis=0))[0].tolist()
    for idx, (c0, c1) in enumerate(chamber_cols):
        if any(c0 <= c <= c1 for c in droplet_cols):
            source_idx = idx
        if int(cavity_fill[:, c0:c1 + 1].sum()) > 0:
            chamber_hits += 1
    if chamber_hits < 2 or source_idx is None:
        return False

    source_only = np.zeros_like(info["cavity_mask"], dtype=bool)
    c0, c1 = chamber_cols[source_idx]
    source_only[:, c0:c1 + 1] = info["cavity_mask"][:, c0:c1 + 1]
    if np.array_equal(cavity_fill, source_only & info["cavity_mask"]):
        return False

    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_attempt(rng)
        if built is None:
            continue
        base_grid, chamber_cols = built
        picked = _pick_level_and_source(base_grid, chamber_cols, rng)
        if picked is None:
            continue
        _level, src_idx, fill = picked
        water_color = rng.choice(WATER_COLORS)
        volume = int(fill.sum())
        grid = _place_droplet(base_grid, chamber_cols, src_idx, volume, water_color, rng)
        if grid is None:
            continue
        if not _validate_generated(grid, chamber_cols):
            continue
        return grid
    raise RuntimeError("failed to generate a valid communicating-vessel sample")
