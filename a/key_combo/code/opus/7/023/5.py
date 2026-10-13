from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]

BG = 0
TEMPLATE = 1
RED = 2
GREEN = 3
PATH = 4
MARKER = 5
SPRITE_COLORS = (RED, GREEN)

_GROUP_GAP = 2
_GROUP_RETRIES = 300
_GRID_RETRIES = 300

_TEMPLATE_SHAPES = [
    {(0, 0), (1, 0), (1, 1)},
    {(0, 0), (1, 0), (2, 0), (2, 1)},
    {(0, 1), (1, 0), (1, 1), (2, 0)},
    {(0, 1), (1, 0), (1, 1), (1, 2)},
    {(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)},
    {(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)},
]

_SPRITE_SHAPES = [
    {(0, 0), (1, 0), (1, 1)},
    {(0, 0), (1, 0), (2, 0), (2, 1)},
    {(0, 1), (1, 0), (1, 1), (2, 0)},
    {(0, 1), (1, 0), (1, 1), (1, 2)},
    {(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)},
    {(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)},
]


def neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: list[Coord] | np.ndarray) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_from_mask(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize_shape(cells: set[Coord]) -> frozenset[Coord]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _transform_shape(cells: set[Coord], rot: int, flip: bool) -> set[Coord]:
    pts = list(cells)
    out: list[Coord] = []
    for r, c in pts:
        rr, cc = r, c
        for _ in range(rot % 4):
            rr, cc = cc, -rr
        if flip:
            cc = -cc
        out.append((rr, cc))
    norm = _normalize_shape(set(out))
    return set(norm)


def _random_shape(shape_bank: list[set[Coord]]) -> set[Coord]:
    base = random.choice(shape_bank)
    return _transform_shape(base, random.randrange(4), bool(random.randrange(2)))


def _shape_bbox(cells: set[Coord]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox_coords(list(cells))
    return r1 - r0 + 1, c1 - c0 + 1


def _place_cells(cells: set[Coord], top: int, left: int) -> set[Coord]:
    return {(top + r, left + c) for r, c in cells}


def _repair_sites(shape: set[Coord]) -> list[Coord]:
    h, w = _shape_bbox(shape)
    sites: list[Coord] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in shape:
                continue
            count = sum((nr, nc) in shape for nr, nc in neighbors4(r, c))
            if count >= 3:
                sites.append((r, c))
    return sites


def _boundary_cells(shape: set[Coord], side: str) -> list[Coord]:
    if side == "left":
        return [(r, c) for r, c in shape if (r, c - 1) not in shape]
    if side == "right":
        return [(r, c) for r, c in shape if (r, c + 1) not in shape]
    raise ValueError("side must be left or right")


def _dfs_path(
    free: np.ndarray,
    start: Coord,
    end: Coord,
    length: int,
    forbidden_adj: set[Coord],
) -> list[Coord] | None:
    target_steps = int(length) - 1
    used = {start}
    path = [start]

    def rec(cur: Coord, steps: int) -> list[Coord] | None:
        dist = abs(cur[0] - end[0]) + abs(cur[1] - end[1])
        remaining = target_steps - steps
        if dist > remaining or ((remaining - dist) % 2 != 0):
            return None
        if steps == target_steps:
            return path.copy() if cur == end else None

        nbrs = list(neighbors4(*cur))
        random.shuffle(nbrs)
        for nr, nc in nbrs:
            nxt = (nr, nc)
            if not in_bounds(free.shape, nr, nc) or not free[nr, nc] or nxt in used:
                continue
            if nxt != end and nxt in forbidden_adj:
                continue
            used.add(nxt)
            path.append(nxt)
            found = rec(nxt, steps + 1)
            if found is not None:
                return found
            path.pop()
            used.remove(nxt)
        return None

    return rec(start, 0)


def _sprite_patch_from_coords(coords: list[Coord], color: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = bbox_coords(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = color
    return patch, (r0, c0, r1, c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each connected sprite above or below its blue template according to the parity of the yellow trail, then repair embedded gray markers.

    Input structure:
    - The background is black.
    - There are 2 or 3 separated structures made of a blue template, a red or green sprite, and a yellow 4-connected trail joining them.
    - Gray single-pixel markers sit near the template, typically in the rows where the stamped copy may appear.

    Rule:
    - For each structure, count only the yellow trail pixels.
    - Crop the sprite by its bounding box and keep its exact orientation and color.
    - If the trail length is odd, paste the sprite flush below the template; if it is even, paste it flush above. Align the sprite bbox left edge with the template bbox left edge, and paste only non-background sprite pixels.
    - Erase all yellow trail pixels to black.
    - After stamping, recolor any gray marker whose 4-neighborhood contains at least three neighbors of the same sprite color; leave all other gray markers unchanged.

    Key insight:
    The trail is not a motion path here: its pixel-count parity is an implicit orientation code that decides whether the copied sprite belongs above or below the template.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    structural = np.isin(g, [TEMPLATE, RED, GREEN, PATH])
    group_components = _components_from_mask(structural)
    assert 1 <= len(group_components) <= 3

    for comp in group_components:
        colors = {int(g[r, c]) for r, c in comp}
        assert TEMPLATE in colors and PATH in colors
        sprite_colors = sorted(colors & set(SPRITE_COLORS))
        assert len(sprite_colors) == 1
        sprite_color = sprite_colors[0]

        template_cells = [(r, c) for r, c in comp if int(g[r, c]) == TEMPLATE]
        path_cells = [(r, c) for r, c in comp if int(g[r, c]) == PATH]
        sprite_cells = [(r, c) for r, c in comp if int(g[r, c]) == sprite_color]
        assert template_cells and path_cells and sprite_cells

        sprite_patch, sprite_bbox = _sprite_patch_from_coords(sprite_cells, sprite_color)
        tr0, tc0, tr1, tc1 = bbox_coords(template_cells)
        sprite_h, sprite_w = sprite_patch.shape

        paste_r = tr1 + 1 if len(path_cells) % 2 == 1 else tr0 - sprite_h
        paste_c = tc0
        assert 0 <= paste_r and paste_r + sprite_h <= out.shape[0]
        assert 0 <= paste_c and paste_c + sprite_w <= out.shape[1]

        mask = sprite_patch != BG
        for rr in range(sprite_h):
            for cc in range(sprite_w):
                if mask[rr, cc]:
                    out[paste_r + rr, paste_c + cc] = sprite_color

        for r, c in path_cells:
            out[r, c] = BG

    for r, c in np.argwhere(out == MARKER):
        counts = {}
        for color in SPRITE_COLORS:
            counts[color] = sum(
                1
                for nr, nc in neighbors4(int(r), int(c))
                if in_bounds(out.shape, nr, nc) and int(out[nr, nc]) == color
            )
        candidates = [color for color, count in counts.items() if count >= 3]
        assert len(candidates) <= 1
        if candidates:
            out[int(r), int(c)] = candidates[0]

    return out


def _build_group(parity: int, sprite_side: str, need_repair: bool) -> tuple[np.ndarray, np.ndarray, dict] | None:
    for _ in range(_GROUP_RETRIES):
        template_shape = _random_shape(_TEMPLATE_SHAPES)
        sprite_shape = _random_shape(_SPRITE_SHAPES)
        if _normalize_shape(template_shape) == _normalize_shape(sprite_shape):
            continue

        repair_sites = _repair_sites(sprite_shape)
        if need_repair and not repair_sites:
            continue

        sprite_color = random.choice(SPRITE_COLORS)
        template_h, template_w = _shape_bbox(template_shape)
        sprite_h, sprite_w = _shape_bbox(sprite_shape)
        local_h, local_w = 13, 15
        template_top = 4
        template_left = 6 if sprite_side == "left" else 4

        template_cells = _place_cells(template_shape, template_top, template_left)
        tr0, tc0, tr1, tc1 = bbox_coords(list(template_cells))

        stamp_top = tr1 + 1 if parity == 1 else tr0 - sprite_h
        stamp_left = tc0
        if stamp_top < 0 or stamp_top + sprite_h > local_h:
            continue
        stamp_cells = _place_cells(sprite_shape, stamp_top, stamp_left)

        candidate_rows = list(range(max(0, template_top - 2), min(local_h - sprite_h + 1, template_top + 3)))
        random.shuffle(candidate_rows)
        candidate_cols = [0, 1] if sprite_side == "left" else [local_w - sprite_w - 1, local_w - sprite_w - 2]

        path_cells: list[Coord] | None = None
        sprite_cells: set[Coord] | None = None
        chosen_length = None

        template_edge = _boundary_cells(template_shape, "left" if sprite_side == "left" else "right")

        for sprite_top in candidate_rows:
            for sprite_left in candidate_cols:
                if sprite_left < 0 or sprite_left + sprite_w > local_w:
                    continue
                sprite_cells_candidate = _place_cells(sprite_shape, sprite_top, sprite_left)
                if sprite_cells_candidate & template_cells:
                    continue
                if sprite_cells_candidate & stamp_cells:
                    continue

                sprite_edge = _boundary_cells(sprite_shape, "right" if sprite_side == "left" else "left")
                template_edge_cells = [
                    (template_top + r, template_left + c)
                    for r, c in template_edge
                ]
                sprite_edge_cells = [
                    (sprite_top + r, sprite_left + c)
                    for r, c in sprite_edge
                ]
                pairs = [
                    (tcell, scell)
                    for tcell in template_edge_cells
                    for scell in sprite_edge_cells
                ]
                random.shuffle(pairs)

                object_cells = template_cells | sprite_cells_candidate
                forbidden_adj: set[Coord] = set()
                for r, c in object_cells:
                    for nr, nc in neighbors4(r, c):
                        if (nr, nc) not in object_cells:
                            forbidden_adj.add((nr, nc))

                for tcell, scell in pairs:
                    if sprite_side == "left":
                        start = (tcell[0], tcell[1] - 1)
                        end = (scell[0], scell[1] + 1)
                    else:
                        start = (tcell[0], tcell[1] + 1)
                        end = (scell[0], scell[1] - 1)

                    if not in_bounds((local_h, local_w), *start) or not in_bounds((local_h, local_w), *end):
                        continue
                    if start in object_cells or end in object_cells:
                        continue

                    free = np.ones((local_h, local_w), dtype=bool)
                    for r, c in object_cells | stamp_cells:
                        free[r, c] = False
                    free[start] = True
                    free[end] = True

                    lengths = [length for length in range(3, 9) if length % 2 == parity]
                    random.shuffle(lengths)
                    for length in lengths:
                        path = _dfs_path(free, start, end, length, forbidden_adj - {start, end})
                        if path is None:
                            continue

                        if any(cell in stamp_cells for cell in path):
                            continue
                        if sum(1 for nr, nc in neighbors4(*start) if (nr, nc) in template_cells) != 1:
                            continue
                        if sum(1 for nr, nc in neighbors4(*end) if (nr, nc) in sprite_cells_candidate) != 1:
                            continue

                        ok = True
                        for idx, cell in enumerate(path):
                            adj_template = sum((nr, nc) in template_cells for nr, nc in neighbors4(*cell))
                            adj_sprite = sum((nr, nc) in sprite_cells_candidate for nr, nc in neighbors4(*cell))
                            if idx == 0:
                                ok &= adj_template == 1 and adj_sprite == 0
                            elif idx == len(path) - 1:
                                ok &= adj_template == 0 and adj_sprite == 1
                            else:
                                ok &= adj_template == 0 and adj_sprite == 0
                            if not ok:
                                break
                        if not ok:
                            continue

                        path_cells = path
                        sprite_cells = sprite_cells_candidate
                        chosen_length = length
                        break
                    if path_cells is not None:
                        break
                if path_cells is not None:
                    break
            if path_cells is not None:
                break

        if path_cells is None or sprite_cells is None or chosen_length is None:
            continue

        repair_markers: list[Coord] = []
        if need_repair:
            random.shuffle(repair_sites)
            placed = False
            for rr, cc in repair_sites:
                marker = (stamp_top + rr, stamp_left + cc)
                if marker in template_cells or marker in sprite_cells or marker in path_cells:
                    continue
                repair_markers.append(marker)
                placed = True
                break
            if not placed:
                continue

        isolated_markers: list[Coord] = []
        marker_side = "above" if parity == 1 else "below"
        row_offsets = [1, 2, 3]
        random.shuffle(row_offsets)
        cols = list(range(tc0 - 1, tc1 + 2))
        random.shuffle(cols)
        for dist in row_offsets:
            row = tr0 - dist if marker_side == "above" else tr1 + dist
            if not (0 <= row < local_h):
                continue
            for col in cols:
                cell = (row, col)
                if not in_bounds((local_h, local_w), row, col):
                    continue
                occupied = template_cells | sprite_cells | set(path_cells) | stamp_cells | set(repair_markers) | set(isolated_markers)
                if cell in occupied:
                    continue
                future_color_neighbors = sum((nr, nc) in stamp_cells for nr, nc in neighbors4(*cell))
                if future_color_neighbors >= 3:
                    continue
                if abs(col - tc0) > 3:
                    continue
                isolated_markers.append(cell)
                break
            if isolated_markers:
                break
        if not isolated_markers:
            continue

        all_markers = repair_markers + isolated_markers
        if len(all_markers) != len(set(all_markers)):
            continue

        input_patch = np.zeros((local_h, local_w), dtype=int)
        output_patch = np.zeros((local_h, local_w), dtype=int)

        for r, c in template_cells:
            input_patch[r, c] = TEMPLATE
            output_patch[r, c] = TEMPLATE
        for r, c in sprite_cells:
            input_patch[r, c] = sprite_color
            output_patch[r, c] = sprite_color
        for r, c in path_cells:
            input_patch[r, c] = PATH
        for r, c in stamp_cells:
            output_patch[r, c] = sprite_color
        for r, c in isolated_markers:
            input_patch[r, c] = MARKER
            output_patch[r, c] = MARKER
        for r, c in repair_markers:
            input_patch[r, c] = MARKER
            output_patch[r, c] = sprite_color

        mask = (input_patch != BG) | (output_patch != BG)
        r0, c0, r1, c1 = bbox_coords(list(map(tuple, np.argwhere(mask))))
        input_patch = input_patch[r0 : r1 + 1, c0 : c1 + 1].copy()
        output_patch = output_patch[r0 : r1 + 1, c0 : c1 + 1].copy()

        if not np.array_equal(main(input_patch), output_patch):
            continue

        return (
            input_patch,
            output_patch,
            {
                "parity": parity,
                "sprite_side": sprite_side,
                "repair_count": len(repair_markers),
                "isolated_count": len(isolated_markers),
                "path_length": chosen_length,
            },
        )
    return None


def _pack_patches(patches: list[np.ndarray], max_width: int) -> tuple[list[tuple[int, int]], int, int] | None:
    placements: list[tuple[int, int]] = []
    x = 1
    y = 1
    row_h = 0
    used_w = 0
    for patch in patches:
        h, w = patch.shape
        if x > 1 and x + w > max_width - 1:
            y += row_h + _GROUP_GAP
            x = 1
            row_h = 0
        placements.append((y, x))
        x += w + _GROUP_GAP
        row_h = max(row_h, h)
        used_w = max(used_w, x - _GROUP_GAP)
    total_h = y + row_h + 1
    total_w = used_w + 1
    if total_h > 30 or total_w > 30:
        return None
    return placements, total_h, total_w


def _grid_signature(arr: np.ndarray) -> tuple[int, ...]:
    return tuple(map(int, arr.ravel()))


def _semantic_check(inp: np.ndarray, expected: np.ndarray, metas: list[dict]) -> None:
    out = main(inp)
    assert np.array_equal(out, expected)
    assert np.any(inp == PATH)
    assert not np.any(out == PATH)

    structural_groups = _components_from_mask(np.isin(inp, [TEMPLATE, RED, GREEN, PATH]))
    assert 2 <= len(structural_groups) <= 3
    parities = [meta["parity"] for meta in metas]
    assert len(set(parities)) > 1
    assert sum(meta["repair_count"] for meta in metas) >= 1
    assert sum(meta["isolated_count"] for meta in metas) >= 1

    side_buckets: dict[str, set[int]] = {"left": set(), "right": set()}
    for meta in metas:
        side_buckets[meta["sprite_side"]].add(meta["parity"])
    assert any(len(v) >= 2 for v in side_buckets.values())

    assert _grid_signature(inp) != _grid_signature(out)


def generate_input() -> np.ndarray:
    for _ in range(_GRID_RETRIES):
        group_count = 2 if random.random() < 0.45 else 3
        parities = [1, 0]
        if group_count == 3:
            parities.append(random.choice([0, 1]))
        random.shuffle(parities)

        shared_side = random.choice(["left", "right"])
        other_side = "right" if shared_side == "left" else "left"
        sides = [shared_side, shared_side]
        if group_count == 3:
            sides.append(other_side if random.random() < 0.7 else shared_side)
        random.shuffle(sides)

        repair_flags = [True, False]
        while len(repair_flags) < group_count:
            repair_flags.append(bool(random.randrange(2)))
        random.shuffle(repair_flags)

        built: list[tuple[np.ndarray, np.ndarray, dict]] = []
        failed = False
        for parity, side, need_repair in zip(parities, sides, repair_flags):
            group = _build_group(parity, side, need_repair)
            if group is None:
                failed = True
                break
            built.append(group)
        if failed:
            continue

        inputs = [item[0] for item in built]
        outputs = [item[1] for item in built]
        metas = [item[2] for item in built]

        width_choices = list(range(max(18, max(p.shape[1] for p in inputs) + 2), 29))
        random.shuffle(width_choices)
        packed = None
        for width in width_choices:
            packed = _pack_patches(inputs, width)
            if packed is not None:
                break
        if packed is None:
            continue
        placements, h, w = packed

        extra_bottom = random.randint(0, max(0, 26 - h))
        extra_right = random.randint(0, max(0, 26 - w))
        h += extra_bottom
        w += extra_right
        if h > 30 or w > 30:
            continue

        inp = np.zeros((h, w), dtype=int)
        expected = np.zeros((h, w), dtype=int)
        ok = True
        for (top, left), patch_in, patch_out in zip(placements, inputs, outputs):
            ph, pw = patch_in.shape
            if top + ph > h or left + pw > w:
                ok = False
                break
            if np.any(inp[top : top + ph, left : left + pw] != BG):
                ok = False
                break
            inp[top : top + ph, left : left + pw] = patch_in
            expected[top : top + ph, left : left + pw] = patch_out
        if not ok:
            continue

        try:
            _semantic_check(inp, expected, metas)
        except AssertionError:
            continue
        return inp

    raise RuntimeError("failed to generate valid puzzle input")
