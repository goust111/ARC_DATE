from __future__ import annotations

import random
from collections import deque

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden mirror axis from the only true twin-sprite pair, then complete the missing half of the multicolor pattern.

    Input structure:
    - A dominant background contains several separated monochrome sprites of one shared color.
    - Exactly one unordered sprite pair is related by reflection across an unseen horizontal, vertical,
      main-diagonal, or anti-diagonal axis; the other sprites are only translated or rotated decoys.
    - A connected multicolor pattern lies entirely on one side of that same axis, while the reflected side is empty.

    Rule:
    - Identify the unique sprite color and split it into connected components.
    - Among those components, find the only pair for which one component reflects exactly onto the other under one
      latent straight axis, and recover that axis.
    - Reflect every non-background, non-sprite pixel across the recovered axis, preserving all original pixels.

    Key insight:
    - The axis is not drawn anywhere in the grid and cannot be inferred from the half-pattern alone; only the unique
      mirror relation among the same-colored sprites determines where the completion must happen.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _get_bg(g)
    sprite_color, sprite_components = _find_sprite_color_and_components(g, bg)
    axis = _find_unique_twin_axis(sprite_components)

    pattern_mask = (g != bg) & (g != sprite_color)
    pattern_coords = np.argwhere(pattern_mask)
    assert len(pattern_coords) > 0, "expected a multicolor half-pattern"

    signs = {_axis_side(axis, int(r), int(c)) for r, c in pattern_coords}
    assert 0 not in signs and len({np.sign(s) for s in signs}) == 1, "pattern must lie strictly on one side"

    out = g.copy()
    for r_raw, c_raw in pattern_coords:
        r = int(r_raw)
        c = int(c_raw)
        rr, cc = _reflect_point(axis, r, c)
        assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]
        assert int(out[rr, cc]) == bg, "reflected half must land on empty background"
        out[rr, cc] = int(g[r, c])

    return out


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))
AXIS_KINDS = ("row", "col", "main", "anti")
MAX_ATTEMPTS = 500
AXIS_BAG: list[str] = []


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _component_coords(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
    comps: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coords_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in coords}


def _find_sprite_color_and_components(grid: np.ndarray, bg: int) -> tuple[int, list[np.ndarray]]:
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    counts_and_components = []
    for color in colors:
        comps = _component_coords(grid == color, connectivity=4)
        counts_and_components.append((len(comps), color, comps))
    counts_and_components.sort(reverse=True)
    count, color, comps = counts_and_components[0]
    assert count >= 4, "sprite color must have many components"
    assert len(counts_and_components) == 1 or count > counts_and_components[1][0], "sprite color must be unique"
    return color, comps


def _axis_side(axis: tuple[str, int], r: int, c: int) -> int:
    kind, param = axis
    if kind == "row":
        return 2 * r - param
    if kind == "col":
        return 2 * c - param
    if kind == "main":
        return c - r - param
    return r + c - param


def _reflect_point(axis: tuple[str, int], r: int, c: int) -> tuple[int, int]:
    kind, param = axis
    if kind == "row":
        return param - r, c
    if kind == "col":
        return r, param - c
    if kind == "main":
        return c - param, r + param
    return param - c, param - r


def _candidate_axes_for_pair(a: np.ndarray, b: np.ndarray) -> list[tuple[str, int]]:
    if len(a) != len(b):
        return []

    set_b = _coords_set(b)
    axes: list[tuple[str, int]] = []
    for kind in AXIS_KINDS:
        params = set()
        for ra, ca in a:
            ra = int(ra)
            ca = int(ca)
            for rb, cb in b:
                rb = int(rb)
                cb = int(cb)
                if kind == "row":
                    params.add(ra + rb)
                elif kind == "col":
                    params.add(ca + cb)
                elif kind == "main":
                    params.add(ca - rb)
                else:
                    params.add(ra + cb)
        for param in params:
            axis = (kind, int(param))
            reflected = {_reflect_point(axis, int(r), int(c)) for r, c in a}
            if reflected == set_b:
                axes.append(axis)
    return axes


def _find_unique_twin_axis(components: list[np.ndarray]) -> tuple[str, int]:
    matches: list[tuple[int, int, tuple[str, int]]] = []
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            axes = _candidate_axes_for_pair(components[i], components[j])
            for axis in axes:
                matches.append((i, j, axis))
    assert len(matches) == 1, "expected exactly one mirror-related sprite pair"
    return matches[0][2]


def _mask_to_local_coords(mask: np.ndarray) -> tuple[list[tuple[int, int]], int, int]:
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    return coords, mask.shape[0], mask.shape[1]


def _local_coords_to_mask(coords: list[tuple[int, int]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _rot90_local(coords: list[tuple[int, int]], h: int, w: int) -> tuple[list[tuple[int, int]], int, int]:
    rotated = [(c, h - 1 - r) for r, c in coords]
    return rotated, w, h


def _rotate_local(coords: list[tuple[int, int]], h: int, w: int, k: int) -> tuple[list[tuple[int, int]], int, int]:
    out_coords = list(coords)
    out_h, out_w = h, w
    for _ in range(k % 4):
        out_coords, out_h, out_w = _rot90_local(out_coords, out_h, out_w)
    return out_coords, out_h, out_w


def _translate_local(coords: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(r0 + r, c0 + c) for r, c in coords]


def _shape_has_nontrivial_symmetry(mask: np.ndarray) -> bool:
    variants = [
        np.rot90(mask, k) for k in range(4)
    ] + [
        np.fliplr(mask),
        np.flipud(mask),
        np.transpose(mask),
        np.fliplr(np.transpose(mask)),
    ]
    original = mask
    for idx, variant in enumerate(variants):
        if idx == 0:
            continue
        if variant.shape == original.shape and np.array_equal(variant, original):
            return True
    return False


def _generate_sprite_mask() -> np.ndarray:
    for _ in range(200):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        area = random.randint(4, min(7, h * w - 1))
        cells = {(random.randrange(h), random.randrange(w))}
        frontier = set(cells)
        while len(cells) < area and frontier:
            rr, cc = random.choice(tuple(frontier))
            options = []
            for dr, dc in DIR4:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    options.append((nr, nc))
            if not options:
                frontier.remove((rr, cc))
                continue
            new_cell = random.choice(options)
            cells.add(new_cell)
            frontier.add(new_cell)
        if len(cells) != area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not _component_coords(mask, connectivity=4):
            continue
        r0, c0, r1, c1 = _bbox(np.argwhere(mask))
        mask = mask[r0 : r1 + 1, c0 : c1 + 1]
        if _shape_has_nontrivial_symmetry(mask):
            continue
        return mask
    raise RuntimeError("failed to generate asymmetric sprite")


def _axis_param_candidates(kind: str, H: int, W: int) -> list[int]:
    if kind == "row":
        values = list(range(4, 2 * H - 5))
        center = H - 1
    elif kind == "col":
        values = list(range(4, 2 * W - 5))
        center = W - 1
    elif kind == "main":
        values = list(range(-(H - 4), W - 3))
        center = 0
    else:
        values = list(range(3, H + W - 4))
        center = H - 1
    offcenter = [v for v in values if abs(v - center) >= 2]
    centerish = [v for v in values if v not in offcenter]
    if offcenter and random.random() < 0.8:
        return offcenter
    return offcenter + centerish


def _next_axis_kind() -> str:
    global AXIS_BAG
    if not AXIS_BAG:
        AXIS_BAG = list(AXIS_KINDS)
        random.shuffle(AXIS_BAG)
    return AXIS_BAG.pop()


def _in_bounds(H: int, W: int, coords: list[tuple[int, int]]) -> bool:
    return all(0 <= r < H and 0 <= c < W for r, c in coords)


def _can_place(coords: list[tuple[int, int]], occupied: np.ndarray, padding: int = 1) -> bool:
    H, W = occupied.shape
    for r, c in coords:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if occupied[r, c]:
            return False
        for dr in range(-padding, padding + 1):
            for dc in range(-padding, padding + 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < H and 0 <= nc < W and occupied[nr, nc]:
                    return False
    return True


def _paint(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[r, c] = color


def _generate_pattern_local(colors: list[int]) -> tuple[list[tuple[int, int, int]], int, int]:
    for _ in range(200):
        h = random.randint(4, 6)
        w = random.randint(4, 7)
        area = random.randint(max(8, len(colors) * 2), min(16, h * w - 1))
        cells = {(random.randrange(h), random.randrange(w))}
        frontier = set(cells)
        while len(cells) < area and frontier:
            rr, cc = random.choice(tuple(frontier))
            options = []
            for dr, dc in DIR4:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    options.append((nr, nc))
            if not options:
                frontier.remove((rr, cc))
                continue
            cell = random.choice(options)
            cells.add(cell)
            frontier.add(cell)
        if len(cells) != area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        r0, c0, r1, c1 = _bbox(np.argwhere(mask))
        mask = mask[r0 : r1 + 1, c0 : c1 + 1]
        coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
        if len(coords) < len(colors) * 2:
            continue

        seeds = random.sample(coords, len(colors))
        labels: dict[tuple[int, int], int] = {seed: idx for idx, seed in enumerate(seeds)}
        q = deque(seeds)
        while q:
            rr, cc = q.popleft()
            for dr, dc in DIR4:
                nr = rr + dr
                nc = cc + dc
                if (nr, nc) not in labels and 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                    labels[(nr, nc)] = labels[(rr, cc)]
                    q.append((nr, nc))

        groups = [[] for _ in colors]
        for cell, idx in labels.items():
            groups[idx].append(cell)
        if any(len(group) < 2 for group in groups):
            continue

        colored = []
        for idx, color in enumerate(colors):
            for r, c in groups[idx]:
                colored.append((r, c, color))
        return colored, mask.shape[0], mask.shape[1]
    raise RuntimeError("failed to generate pattern")


def _choose_twin_pair(
    axis: tuple[str, int],
    sprite_local: list[tuple[int, int]],
    sh: int,
    sw: int,
    H: int,
    W: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    candidates = []
    for r0 in range(H - sh + 1):
        for c0 in range(W - sw + 1):
            a = _translate_local(sprite_local, r0, c0)
            sides = {_axis_side(axis, r, c) for r, c in a}
            if 0 in sides or len({np.sign(s) for s in sides}) != 1:
                continue
            b = [_reflect_point(axis, r, c) for r, c in a]
            if not _in_bounds(H, W, b):
                continue
            if set(a) & set(b):
                continue
            if not _can_place(a, np.zeros((H, W), dtype=bool), padding=1):
                continue
            occ = np.zeros((H, W), dtype=bool)
            for r, c in a:
                occ[r, c] = True
            if not _can_place(b, occ, padding=1):
                continue
            candidates.append((a, b))
    if not candidates:
        return None
    return random.choice(candidates)


def _place_decoys(
    occupied: np.ndarray,
    sprite_variants: list[tuple[list[tuple[int, int]], int, int]],
    H: int,
    W: int,
    count: int,
) -> list[list[tuple[int, int]]] | None:
    placed: list[list[tuple[int, int]]] = []
    for _ in range(count):
        order = list(range(len(sprite_variants)))
        random.shuffle(order)
        locations = []
        for idx in order:
            local, h, w = sprite_variants[idx]
            for r0 in range(H - h + 1):
                for c0 in range(W - w + 1):
                    coords = _translate_local(local, r0, c0)
                    if _can_place(coords, occupied, padding=1):
                        locations.append(coords)
            if locations:
                break
        if not locations:
            return None
        coords = random.choice(locations)
        for r, c in coords:
            occupied[r, c] = True
        placed.append(coords)
    return placed


def _place_pattern(
    axis: tuple[str, int],
    occupied: np.ndarray,
    pattern_local: list[tuple[int, int, int]],
    ph: int,
    pw: int,
    H: int,
    W: int,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]] | None:
    target_sign = random.choice((-1, 1))
    candidates = []
    source_coords_2d = [(r, c) for r, c, _color in pattern_local]
    color_map = {(r, c): color for r, c, color in pattern_local}
    for r0 in range(H - ph + 1):
        for c0 in range(W - pw + 1):
            src = [(r0 + r, c0 + c, color_map[(r, c)]) for r, c in source_coords_2d]
            sides = [_axis_side(axis, r, c) for r, c, _ in src]
            if any(s == 0 or np.sign(s) != target_sign for s in sides):
                continue
            gap = min(abs(s) for s in sides)
            if gap < 2:
                continue
            src_xy = [(r, c) for r, c, _ in src]
            if not _can_place(src_xy, occupied, padding=1):
                continue
            dst = [(*_reflect_point(axis, r, c), color) for r, c, color in src]
            dst_xy = [(r, c) for r, c, _ in dst]
            if not _in_bounds(H, W, dst_xy):
                continue
            if any(occupied[r, c] for r, c in dst_xy):
                continue
            if set(src_xy) & set(dst_xy):
                continue
            # Avoid the easy shortcut "mirror across the pattern bbox edge".
            r_vals = [r for r, c in src_xy]
            c_vals = [c for r, c in src_xy]
            bbox_mid_row_twice = min(r_vals) + max(r_vals)
            bbox_mid_col_twice = min(c_vals) + max(c_vals)
            if axis[0] == "row" and axis[1] in {bbox_mid_row_twice - 1, bbox_mid_row_twice + 1}:
                continue
            if axis[0] == "col" and axis[1] in {bbox_mid_col_twice - 1, bbox_mid_col_twice + 1}:
                continue
            candidates.append((src, dst))
    if not candidates:
        return None
    return random.choice(candidates)


def _assemble_input(
    H: int,
    W: int,
    bg: int,
    sprite_color: int,
    twins: tuple[list[tuple[int, int]], list[tuple[int, int]]],
    decoys: list[list[tuple[int, int]]],
    src_pattern: list[tuple[int, int, int]],
) -> np.ndarray:
    grid = np.full((H, W), bg, dtype=int)
    for coords in [twins[0], twins[1], *decoys]:
        _paint(grid, coords, sprite_color)
    for r, c, color in src_pattern:
        grid[r, c] = color
    return grid


def _assemble_output(inp: np.ndarray, dst_pattern: list[tuple[int, int, int]]) -> np.ndarray:
    out = inp.copy()
    for r, c, color in dst_pattern:
        out[r, c] = color
    return out


def _semantic_checks(
    inp: np.ndarray,
    out: np.ndarray,
    axis: tuple[str, int],
    sprite_color: int,
    src_pattern: list[tuple[int, int, int]],
    dst_pattern: list[tuple[int, int, int]],
) -> bool:
    bg = _get_bg(inp)
    found_color, comps = _find_sprite_color_and_components(inp, bg)
    if found_color != sprite_color:
        return False
    try:
        found_axis = _find_unique_twin_axis(comps)
    except AssertionError:
        return False
    if found_axis != axis:
        return False

    src_xy = [(r, c) for r, c, _ in src_pattern]
    dst_xy = [(r, c) for r, c, _ in dst_pattern]
    if set(dst_xy) & set(src_xy):
        return False
    if any(int(inp[r, c]) != bg for r, c in dst_xy):
        return False
    if len({color for _r, _c, color in src_pattern}) < 2:
        return False
    if len(_component_coords((inp != bg) & (inp != sprite_color), connectivity=4)) != 1:
        return False

    src_signs = {_axis_side(axis, r, c) for r, c in src_xy}
    dst_signs = {_axis_side(axis, r, c) for r, c in dst_xy}
    if 0 in src_signs or 0 in dst_signs:
        return False
    if len({np.sign(s) for s in src_signs}) != 1 or len({np.sign(s) for s in dst_signs}) != 1:
        return False
    if next(iter({np.sign(s) for s in src_signs})) == next(iter({np.sign(s) for s in dst_signs})):
        return False

    try:
        solved = main(inp)
    except AssertionError:
        return False
    return np.array_equal(solved, out)


def _attempt_generate() -> np.ndarray | None:
    H = random.randint(18, 28)
    W = random.randint(18, 28)
    kind = _next_axis_kind()
    axis_candidates = _axis_param_candidates(kind, H, W)
    if not axis_candidates:
        return None
    axis = (kind, random.choice(axis_candidates))

    palette = list(range(10))
    bg = random.choice(palette)
    sprite_color = random.choice([c for c in palette if c != bg])
    pattern_color_count = random.randint(2, 4)
    pattern_colors = random.sample([c for c in palette if c not in {bg, sprite_color}], pattern_color_count)

    sprite_mask = _generate_sprite_mask()
    sprite_local, sh, sw = _mask_to_local_coords(sprite_mask)
    twins = _choose_twin_pair(axis, sprite_local, sh, sw, H, W)
    if twins is None:
        return None

    occupied = np.zeros((H, W), dtype=bool)
    for coords in twins:
        for r, c in coords:
            occupied[r, c] = True

    variant_indices = [0, 0, 0, random.randrange(4)]
    sprite_variants = []
    for k in variant_indices:
        rotated, rh, rw = _rotate_local(sprite_local, sh, sw, k)
        sprite_variants.append((rotated, rh, rw))

    decoy_count = random.randint(2, 4)
    decoys = _place_decoys(occupied, sprite_variants, H, W, decoy_count)
    if decoys is None:
        return None

    pattern_local, ph, pw = _generate_pattern_local(pattern_colors)
    placed_pattern = _place_pattern(axis, occupied, pattern_local, ph, pw, H, W)
    if placed_pattern is None:
        return None
    src_pattern, dst_pattern = placed_pattern

    inp = _assemble_input(H, W, bg, sprite_color, twins, decoys, src_pattern)
    out = _assemble_output(inp, dst_pattern)
    if not _semantic_checks(inp, out, axis, sprite_color, src_pattern, dst_pattern):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-axis twin puzzle")
