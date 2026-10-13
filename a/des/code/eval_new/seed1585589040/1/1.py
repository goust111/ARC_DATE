from __future__ import annotations

from collections import deque

import numpy as np


DIR_STEPS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIR_STEPS.values():
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _component_from_seed(mask: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    h, w = mask.shape
    sr, sc = seed
    assert 0 <= sr < h and 0 <= sc < w and mask[sr, sc]
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque([seed])
    seen[sr, sc] = True
    cells: list[tuple[int, int]] = []
    while q:
        cr, cc = q.popleft()
        cells.append((cr, cc))
        for dr, dc in DIR_STEPS.values():
            nr = cr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return np.array(cells, dtype=int)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = {(r, c0) for r in range(r0, r1 + 1)}
    cells.update((r, c1) for r in range(r0, r1 + 1))
    cells.update((r0, c) for c in range(c0, c1 + 1))
    cells.update((r1, c) for c in range(c0, c1 + 1))
    return cells


def _open_rectangle_info(coords: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[int, int], str] | None:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    border = _border_cells(r0, c0, r1, c1)
    comp = {tuple(map(int, cell)) for cell in coords}
    if not comp <= border:
        return None

    missing = list(border - comp)
    if len(missing) != 1:
        return None

    gap = missing[0]
    if gap in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
        return None
    if gap[0] == r0:
        side = "up"
    elif gap[0] == r1:
        side = "down"
    elif gap[1] == c0:
        side = "left"
    else:
        side = "right"
    return (r0, c0, r1, c1), gap, side


def _closed_rectangle_info(
    grid: np.ndarray,
    bg: int,
    coords: np.ndarray,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    border = _border_cells(r0, c0, r1, c1)
    comp = {tuple(map(int, cell)) for cell in coords}
    if comp != border:
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0 or not np.all(interior == bg):
        return None
    return (r0, c0, r1, c1), (r0 + 1, c0 + 1, r1 - 1, c1 - 1)


def _find_source_patch(
    grid: np.ndarray,
    bg: int,
    frame_color: int,
    gap: tuple[int, int],
    direction: str,
) -> np.ndarray:
    h, w = grid.shape
    dr, dc = DIR_STEPS[direction]
    r = gap[0] + dr
    c = gap[1] + dc
    sprite_mask = (grid != bg) & (grid != frame_color)

    while 0 <= r < h and 0 <= c < w:
        value = int(grid[r, c])
        if value == bg:
            r += dr
            c += dc
            continue
        assert value != frame_color, "gap ray should point to a sprite, not another frame"
        coords = _component_from_seed(sprite_mask, (r, c))
        sr0, sc0, sr1, sc1 = _bbox(coords)
        return grid[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()

    raise AssertionError("missing sprite along the gap direction")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each inner hole with the external sprite pointed to by the missing side of its outer frame.

    Input structure:
    - The dominant color is background.
    - Several same-colored outer rectangular frames appear on the grid; each one
      is a one-pixel-thick rectangle with exactly one missing border cell on one side.
    - Inside every outer frame there is a smaller complete rectangular frame of the
      same color, leaving a background hole in one corner.
    - Outside each outer frame, along the straight ray that exits through the missing
      border cell, there is a separated sprite whose bounding box matches the hole size.

    Rule:
    - Identify every outer frame and locate its unique missing border cell to get a
      direction: up, down, left, or right.
    - Search outward from that gap along the corresponding ray until reaching the
      associated external sprite.
    - Copy the sprite pattern into the background hole enclosed by the inner frame,
      aligning the sprite's top-left corner with the hole's top-left corner.
    - Leave all other cells unchanged.

    Key insight:
    - The missing border cell acts as a directional pointer: it is the only visible
      cue that pairs each internal hole with the correct external sprite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors, "expected at least one foreground color"

    outer_by_color: dict[int, list[tuple[tuple[int, int, int, int], tuple[int, int], str]]] = {}
    comps_by_color: dict[int, list[np.ndarray]] = {}
    for color in colors:
        comps = _components(g == color)
        comps_by_color[color] = comps
        outer = []
        for coords in comps:
            info = _open_rectangle_info(coords)
            if info is not None:
                outer.append(info)
        if outer:
            outer_by_color[color] = outer

    assert len(outer_by_color) == 1, "expected a unique frame color"
    frame_color = next(iter(outer_by_color))
    outer_infos = sorted(outer_by_color[frame_color], key=lambda item: item[0])

    closed_frames = []
    for coords in comps_by_color[frame_color]:
        info = _closed_rectangle_info(g, bg, coords)
        if info is not None:
            closed_frames.append(info)

    out = g.copy()
    for outer_bbox, gap, direction in outer_infos:
        r0, c0, r1, c1 = outer_bbox
        inner_candidates = [
            hole_bbox
            for frame_bbox, hole_bbox in closed_frames
            if r0 < frame_bbox[0] and c0 < frame_bbox[1] and frame_bbox[2] < r1 and frame_bbox[3] < c1
        ]
        assert len(inner_candidates) == 1, "each outer frame must contain exactly one inner hole frame"
        hr0, hc0, hr1, hc1 = inner_candidates[0]

        source = _find_source_patch(g, bg, frame_color, gap, direction)
        hole_shape = (hr1 - hr0 + 1, hc1 - hc0 + 1)
        assert source.shape == hole_shape, "sprite size must match the hole size"
        out[hr0 : hr1 + 1, hc0 : hc1 + 1] = source

    return out


def _paste_non_bg(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    ph, pw = patch.shape
    region = grid[top : top + ph, left : left + pw]
    mask = patch != bg
    region[mask] = patch[mask]


def _dilate_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    blocked: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked.add((nr, nc))
    return blocked


def _make_sprite_pattern(rng: np.random.Generator, shape: tuple[int, int], bg: int, frame_color: int) -> np.ndarray:
    h, w = shape
    colors = [color for color in range(10) if color not in {bg, frame_color}]
    rng.shuffle(colors)
    palette = [int(color) for color in colors[:3]]
    mode = str(rng.choice(["solid", "vstripe", "hstripe", "checker", "split"]))
    patch = np.empty((h, w), dtype=int)

    if mode == "solid":
        patch[:, :] = palette[0]
    elif mode == "vstripe":
        width = min(len(palette), max(1, w))
        use = palette[:width]
        for c in range(w):
            patch[:, c] = use[c % len(use)]
    elif mode == "hstripe":
        height = min(len(palette), max(1, h))
        use = palette[:height]
        for r in range(h):
            patch[r, :] = use[r % len(use)]
    elif mode == "checker":
        use = palette[:2]
        if len(use) == 1:
            use = [palette[0], palette[0]]
        for r in range(h):
            for c in range(w):
                patch[r, c] = use[(r + c) % 2]
    else:
        use = palette[:2]
        if len(use) == 1:
            use = [palette[0], palette[0]]
        split = int(rng.integers(1, max(2, w))) if w > 1 else 1
        patch[:, :split] = use[0]
        patch[:, split:] = use[1]

    return patch


def _build_outer_patch(
    outer_h: int,
    outer_w: int,
    hole_h: int,
    hole_w: int,
    corner: str,
    direction: str,
    gap_offset: int,
    bg: int,
    frame_color: int,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int, int, int]]:
    patch = np.full((outer_h, outer_w), bg, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color

    if direction == "up":
        gap = (0, gap_offset)
    elif direction == "down":
        gap = (outer_h - 1, gap_offset)
    elif direction == "left":
        gap = (gap_offset, 0)
    else:
        gap = (gap_offset, outer_w - 1)
    patch[gap] = bg

    frame_h = hole_h + 2
    frame_w = hole_w + 2
    if corner == "tl":
        fr0 = 2
        fc0 = 2
    elif corner == "tr":
        fr0 = 2
        fc0 = outer_w - frame_w - 2
    elif corner == "bl":
        fr0 = outer_h - frame_h - 2
        fc0 = 2
    else:
        fr0 = outer_h - frame_h - 2
        fc0 = outer_w - frame_w - 2

    patch[fr0, fc0 : fc0 + frame_w] = frame_color
    patch[fr0 + frame_h - 1, fc0 : fc0 + frame_w] = frame_color
    patch[fr0 : fr0 + frame_h, fc0] = frame_color
    patch[fr0 : fr0 + frame_h, fc0 + frame_w - 1] = frame_color

    hole_bbox = (fr0 + 1, fc0 + 1, fr0 + hole_h, fc0 + hole_w)
    return patch, gap, hole_bbox


def _try_place_object(
    rng: np.random.Generator,
    shape: tuple[int, int],
    bg: int,
    frame_color: int,
    spec: dict[str, object],
    blocked_actual: set[tuple[int, int]],
    occupied_actual: set[tuple[int, int]],
    reserved_corridors: set[tuple[int, int]],
) -> tuple[dict[str, object], set[tuple[int, int]], set[tuple[int, int]]] | None:
    h, w = shape
    direction = str(spec["direction"])
    corner = str(spec["corner"])
    hole_h = int(spec["hole_h"])
    hole_w = int(spec["hole_w"])
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    sprite = np.asarray(spec["sprite"], dtype=int)
    sprite_h, sprite_w = sprite.shape

    for _ in range(300):
        dist = int(rng.integers(2, 6))
        if direction in {"up", "down"}:
            gap_offset = int(rng.integers(1, outer_w - 1))
        else:
            gap_offset = int(rng.integers(1, outer_h - 1))
        outer_patch, local_gap, local_hole = _build_outer_patch(
            outer_h,
            outer_w,
            hole_h,
            hole_w,
            corner,
            direction,
            gap_offset,
            bg,
            frame_color,
        )

        if direction == "up":
            min_r0 = dist + sprite_h - 1
            max_r0 = h - outer_h
            if min_r0 > max_r0:
                continue
            r0 = int(rng.integers(min_r0, max_r0 + 1))
            c0 = int(rng.integers(0, w - outer_w + 1))
            ray_offset = int(rng.integers(0, sprite_w))
            sprite_r0 = r0 - dist - sprite_h + 1
            sprite_c0 = c0 + gap_offset - ray_offset
            gap_global = (r0, c0 + gap_offset)
            corridor = {(rr, gap_global[1]) for rr in range(gap_global[0] - 1, gap_global[0] - dist, -1)}
        elif direction == "down":
            max_r0 = h - outer_h - dist - sprite_h + 1
            if max_r0 < 0:
                continue
            r0 = int(rng.integers(0, max_r0 + 1))
            c0 = int(rng.integers(0, w - outer_w + 1))
            ray_offset = int(rng.integers(0, sprite_w))
            sprite_r0 = r0 + outer_h - 1 + dist
            sprite_c0 = c0 + gap_offset - ray_offset
            gap_global = (r0 + outer_h - 1, c0 + gap_offset)
            corridor = {(rr, gap_global[1]) for rr in range(gap_global[0] + 1, sprite_r0)}
        elif direction == "left":
            min_c0 = dist + sprite_w - 1
            max_c0 = w - outer_w
            if min_c0 > max_c0:
                continue
            r0 = int(rng.integers(0, h - outer_h + 1))
            c0 = int(rng.integers(min_c0, max_c0 + 1))
            ray_offset = int(rng.integers(0, sprite_h))
            sprite_r0 = r0 + gap_offset - ray_offset
            sprite_c0 = c0 - dist - sprite_w + 1
            gap_global = (r0 + gap_offset, c0)
            corridor = {(gap_global[0], cc) for cc in range(gap_global[1] - 1, gap_global[1] - dist, -1)}
        else:
            max_c0 = w - outer_w - dist - sprite_w + 1
            if max_c0 < 0:
                continue
            r0 = int(rng.integers(0, h - outer_h + 1))
            c0 = int(rng.integers(0, max_c0 + 1))
            ray_offset = int(rng.integers(0, sprite_h))
            sprite_r0 = r0 + gap_offset - ray_offset
            sprite_c0 = c0 + outer_w - 1 + dist
            gap_global = (r0 + gap_offset, c0 + outer_w - 1)
            corridor = {(gap_global[0], cc) for cc in range(gap_global[1] + 1, sprite_c0)}

        if not (0 <= sprite_r0 and 0 <= sprite_c0 and sprite_r0 + sprite_h <= h and sprite_c0 + sprite_w <= w):
            continue

        actual_cells = {
            (r0 + int(rr), c0 + int(cc))
            for rr, cc in np.argwhere(outer_patch != bg)
        }
        actual_cells.update(
            (sprite_r0 + int(rr), sprite_c0 + int(cc))
            for rr, cc in np.argwhere(sprite != bg)
        )
        if actual_cells & blocked_actual:
            continue
        if actual_cells & reserved_corridors:
            continue
        if corridor & occupied_actual:
            continue
        if corridor & blocked_actual:
            continue

        hr0, hc0, hr1, hc1 = local_hole
        placement = {
            "outer_patch": outer_patch,
            "outer_top": r0,
            "outer_left": c0,
            "sprite_patch": sprite,
            "sprite_top": sprite_r0,
            "sprite_left": sprite_c0,
            "hole_bbox": (r0 + hr0, c0 + hc0, r0 + hr1, c0 + hc1),
        }
        return placement, actual_cells, corridor

    return None


def _sample_specs(rng: np.random.Generator, bg: int, frame_color: int, count: int) -> list[dict[str, object]]:
    directions = [str(direction) for direction in rng.permutation(["up", "down", "left", "right"])[:count]]
    specs: list[dict[str, object]] = []
    for direction in directions:
        hole_h = int(rng.integers(2, 5))
        hole_w = int(rng.integers(2, 5))
        outer_h = hole_h + int(rng.integers(6, 9))
        outer_w = hole_w + int(rng.integers(6, 9))
        sprite = _make_sprite_pattern(rng, (hole_h, hole_w), bg, frame_color)
        specs.append(
            {
                "direction": direction,
                "corner": str(rng.choice(["tl", "tr", "bl", "br"])),
                "hole_h": hole_h,
                "hole_w": hole_w,
                "outer_h": outer_h,
                "outer_w": outer_w,
                "sprite": sprite,
                "footprint": outer_h * outer_w + hole_h * hole_w,
            }
        )
    specs.sort(key=lambda spec: int(spec["footprint"]), reverse=True)
    return specs


def _generate_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    count = int(rng.choice([2, 3, 4], p=[0.4, 0.4, 0.2]))
    if count == 2:
        h = int(rng.integers(18, 27))
        w = int(rng.integers(18, 27))
    elif count == 3:
        h = int(rng.integers(22, 31))
        w = int(rng.integers(22, 31))
    else:
        h = int(rng.integers(26, 31))
        w = int(rng.integers(26, 31))

    colors = np.arange(10, dtype=int)
    rng.shuffle(colors)
    bg = int(colors[0])
    frame_color = int(colors[1])

    specs = _sample_specs(rng, bg, frame_color, count)
    blocked_actual: set[tuple[int, int]] = set()
    occupied_actual: set[tuple[int, int]] = set()
    reserved_corridors: set[tuple[int, int]] = set()
    placements: list[dict[str, object]] = []

    for spec in specs:
        result = _try_place_object(
            rng,
            (h, w),
            bg,
            frame_color,
            spec,
            blocked_actual,
            occupied_actual,
            reserved_corridors,
        )
        if result is None:
            return None
        placement, actual_cells, corridor = result
        placements.append(placement)
        occupied_actual |= actual_cells
        blocked_actual |= _dilate_cells(actual_cells, (h, w))
        reserved_corridors |= corridor

    inp = np.full((h, w), bg, dtype=int)
    out = inp.copy()
    for placement in placements:
        outer_patch = np.asarray(placement["outer_patch"], dtype=int)
        outer_top = int(placement["outer_top"])
        outer_left = int(placement["outer_left"])
        sprite_patch = np.asarray(placement["sprite_patch"], dtype=int)
        sprite_top = int(placement["sprite_top"])
        sprite_left = int(placement["sprite_left"])
        hr0, hc0, hr1, hc1 = placement["hole_bbox"]

        _paste_non_bg(inp, outer_patch, outer_top, outer_left, bg)
        _paste_non_bg(inp, sprite_patch, sprite_top, sprite_left, bg)
        _paste_non_bg(out, outer_patch, outer_top, outer_left, bg)
        _paste_non_bg(out, sprite_patch, sprite_top, sprite_left, bg)
        out[hr0 : hr1 + 1, hc0 : hc1 + 1] = sprite_patch

    if _get_bg(inp) != bg:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(500):
        rng = np.random.default_rng()
        generated = _generate_case(rng)
        if generated is not None:
            return generated[0]
    raise RuntimeError("failed to generate a valid ARC sample")
