from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
ANCHOR = 2
SKELETON = 5
ANNOTATION_POOL = (1, 3, 4, 6, 7, 8, 9)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr != 0 or dc != 0
    ]


def _normalize_coords(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _c in coords)
    min_c = min(c for _r, c in coords)
    return {(r - min_r, c - min_c) for r, c in coords}


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors8(cr, cc):
                    if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(sorted(cells), dtype=int))
    return components


def _render_patch(cells: dict[tuple[int, int], int]) -> np.ndarray:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    patch = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
    for (r, c), color in cells.items():
        patch[r - min_r, c - min_c] = int(color)
    return patch


def _dihedral_variants(patch: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rotated = np.rot90(patch, k)
        for variant in (rotated, np.fliplr(rotated)):
            key = (variant.shape, variant.tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(variant.copy())
    return variants


def _component_patch(grid: np.ndarray, coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in coords:
        patch[int(r) - r0, int(c) - c0] = int(grid[int(r), int(c)])
    return patch


def _component_roles(
    grid: np.ndarray, coords: np.ndarray
) -> tuple[set[tuple[int, int]], tuple[int, int], dict[tuple[int, int], int]]:
    skeleton = {
        (int(r), int(c))
        for r, c in coords
        if int(grid[int(r), int(c)]) == SKELETON
    }
    anchors = [
        (int(r), int(c))
        for r, c in coords
        if int(grid[int(r), int(c)]) == ANCHOR
    ]
    ann = {
        (int(r), int(c)): int(grid[int(r), int(c)])
        for r, c in coords
        if int(grid[int(r), int(c)]) not in (BG, ANCHOR, SKELETON)
    }
    assert 6 <= len(skeleton) <= 10, "each motif must contain a 6-10 pixel skeleton"
    assert len(anchors) == 1, "each motif must contain exactly one anchor"
    return skeleton, anchors[0], ann


def _candidate_fill(
    variant: np.ndarray,
    target_skeleton: set[tuple[int, int]],
    target_anchor: tuple[int, int],
    target_ann: dict[tuple[int, int], int],
    template_colors: set[int],
) -> tuple[tuple[int, int, int], ...] | None:
    anchor_coords = np.argwhere(variant == ANCHOR)
    if anchor_coords.shape != (1, 2):
        return None

    ar, ac = map(int, anchor_coords[0])
    off_r = int(target_anchor[0]) - ar
    off_c = int(target_anchor[1]) - ac

    shifted_skeleton = {
        (int(r) + off_r, int(c) + off_c)
        for r, c in np.argwhere(variant == SKELETON)
    }
    if shifted_skeleton != target_skeleton:
        return None

    variant_ann = {
        (int(r) + off_r, int(c) + off_c): int(variant[int(r), int(c)])
        for r, c in np.argwhere(
            (variant != BG) & (variant != ANCHOR) & (variant != SKELETON)
        )
    }
    if not set(target_ann).issubset(set(variant_ann)):
        return None

    if not set(target_ann.values()).issubset(template_colors):
        return None

    mapping: dict[int, int] = {}
    inverse: dict[int, int] = {}
    for pos, seen_color in target_ann.items():
        tpl_color = int(variant_ann[pos])
        if tpl_color in mapping and mapping[tpl_color] != seen_color:
            return None
        if seen_color in inverse and inverse[seen_color] != tpl_color:
            return None
        mapping[tpl_color] = int(seen_color)
        inverse[int(seen_color)] = tpl_color

    missing_tpl = template_colors - set(mapping)
    missing_target = template_colors - set(mapping.values())
    if len(missing_tpl) != len(missing_target):
        return None
    if len(missing_tpl) > 1:
        return None
    if len(missing_tpl) == 1:
        mapping[next(iter(missing_tpl))] = next(iter(missing_target))

    filled = []
    for (r, c), tpl_color in variant_ann.items():
        filled.append((int(r), int(c), int(mapping[int(tpl_color)])))
    return tuple(sorted(filled))


def main(grid: np.ndarray) -> np.ndarray:
    """Complete each rotated or reflected skeleton copy by transferring the template's annotation layout through a per-copy color permutation.

    Input structure:
    - The background is black/0 and contains 4-5 separated 8-connected motifs.
    - Every motif has a gray/5 4-connected skeleton with 6-10 pixels, one red/2 anchor pixel, and optionally colored annotation pixels.
    - Exactly one motif is the template with the most annotation pixels; every other motif shows only a subset of the template's annotation positions.

    Rule:
    - Identify the unique template as the motif with the largest annotation count.
    - For each incomplete motif, enumerate the 8 dihedral transforms of the template and keep the unique one whose transformed skeleton and anchor align exactly after translation.
    - Use the aligned observed annotation pixels to infer that motif's one-to-one color mapping from template colors, then stamp the mapped colors onto every template annotation position in the target orientation.

    Key insight:
    - Skeleton-plus-anchor alignment tells you which local positions correspond, and those same aligned observed pixels reveal the color permutation needed to finish the rest of the copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.all((g >= 0) & (g <= 9)), "ARC grids must use color ids 0-9"

    components = _find_components(g != BG)
    assert 4 <= len(components) <= 5, "expected 4-5 separated motifs"

    roles = [_component_roles(g, comp) for comp in components]
    ann_counts = [len(ann) for _skel, _anchor, ann in roles]
    max_ann = max(ann_counts)
    template_ids = [i for i, count in enumerate(ann_counts) if count == max_ann]
    assert len(template_ids) == 1, "template must be uniquely determined by annotation count"
    template_id = template_ids[0]

    template_patch = _component_patch(g, components[template_id])
    template_colors = {
        int(v)
        for v in np.unique(template_patch)
        if int(v) not in (BG, ANCHOR, SKELETON)
    }
    assert 2 <= len(template_colors) <= 3, "template must use 2-3 annotation colors"

    variants = _dihedral_variants(template_patch)
    out = g.copy()

    for i, (_target_skel, target_anchor, target_ann) in enumerate(roles):
        if i == template_id:
            continue
        assert len(target_ann) == len(template_colors) - 1, "each incomplete copy must show one color fewer than the template set"

        candidates = []
        for variant in variants:
            filled = _candidate_fill(
                variant,
                roles[i][0],
                target_anchor,
                target_ann,
                template_colors,
            )
            if filled is not None:
                candidates.append(filled)

        unique = sorted(set(candidates))
        assert len(unique) == 1, "each incomplete motif must admit a unique aligned color completion"
        for r, c, color in unique[0]:
            if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
                assert int(out[r, c]) in (BG, int(color)), "completion conflicts with existing content"
                out[r, c] = int(color)

    return out


# --------------- generator ---------------

GEN_MAX_ATTEMPTS = 400
GEN_GRID_RANGE = (19, 29)
GEN_MOTIF_COUNT_RANGE = (4, 5)
GEN_SKELETON_SIZE_RANGE = (6, 10)
GEN_SKELETON_BBOX_MAX = 6
GEN_FULL_BBOX_MAX = 8
GEN_PLACEMENT_PADDING = 1


def _random_polyomino(size: int) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < size:
        frontier = set()
        for r, c in cells:
            frontier.update(_neighbors4(r, c))
        frontier -= cells
        cells.add(random.choice(tuple(frontier)))
    return _normalize_coords(cells)


def _has_turn_or_branch(cells: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        neighbors = [p for p in _neighbors4(r, c) if p in cells]
        if len(neighbors) >= 3:
            return True
        if (
            ((r - 1, c) in cells or (r + 1, c) in cells)
            and ((r, c - 1) in cells or (r, c + 1) in cells)
        ):
            return True
    return False


def _coord_bbox(coords: set[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _adjacent4_open(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    open_cells = set()
    for r, c in cells:
        for p in _neighbors4(r, c):
            if p not in cells:
                open_cells.add(p)
    return sorted(open_cells)


def _annotation_candidates(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    candidates = set()
    for r, c in cells:
        for p in _neighbors8(r, c):
            if p not in cells:
                candidates.add(p)
    return sorted(candidates)


def _pattern_spread_ok(positions: tuple[tuple[int, int], ...]) -> bool:
    rows = {r for r, _c in positions}
    cols = {c for _r, c in positions}
    return len(rows) >= 2 or len(cols) >= 2


def _choose_template_patch() -> tuple[np.ndarray, tuple[int, ...]]:
    for _ in range(300):
        color_count = random.choice((2, 3))
        skeleton = _random_polyomino(random.randint(*GEN_SKELETON_SIZE_RANGE))
        if not _has_turn_or_branch(skeleton):
            continue
        if max(_coord_bbox(skeleton)) > GEN_SKELETON_BBOX_MAX:
            continue

        anchor_options = _adjacent4_open(skeleton)
        random.shuffle(anchor_options)
        for anchor in anchor_options:
            role_cells = dict.fromkeys(skeleton, SKELETON)
            role_cells[anchor] = ANCHOR
            role_patch = _render_patch(role_cells)
            if len(_dihedral_variants(role_patch)) != 8:
                continue

            occupied = set(skeleton) | {anchor}
            ann_candidates = _annotation_candidates(occupied)
            random.shuffle(ann_candidates)
            if len(ann_candidates) < color_count:
                continue

            combos = list(itertools.combinations(ann_candidates, color_count))
            random.shuffle(combos)
            for ann_positions in combos[:50]:
                if not _pattern_spread_ok(ann_positions):
                    continue
                colors = tuple(random.sample(ANNOTATION_POOL, color_count))
                cells = dict(role_cells)
                for pos, color in zip(ann_positions, colors):
                    cells[pos] = int(color)
                patch = _render_patch(cells)
                if max(patch.shape) > GEN_FULL_BBOX_MAX:
                    continue
                if len(_dihedral_variants(patch)) != 8:
                    continue
                return patch, colors
    raise RuntimeError("failed to build a non-degenerate template patch")


def _permutation_maps(colors: tuple[int, ...], count: int) -> list[dict[int, int]]:
    k = len(colors)
    maps: list[dict[int, int]] = []
    if k == 2:
        perms = [(0, 1), (1, 0)]
    else:
        perms = [(0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (1, 0, 2), (2, 1, 0)]
    random.shuffle(perms)
    while len(maps) < count:
        perm = perms[len(maps) % len(perms)]
        maps.append({int(colors[i]): int(colors[j]) for i, j in enumerate(perm)})
    return maps


def _missing_template_colors(colors: tuple[int, ...], count: int) -> list[int]:
    start = random.randrange(len(colors))
    return [int(colors[(start + i) % len(colors)]) for i in range(count)]


def _orientation_indices(count: int) -> list[int]:
    template_idx = random.randrange(8)
    remaining = [i for i in range(8) if i != template_idx]
    even = [i for i in remaining if i % 2 == 0]
    odd = [i for i in remaining if i % 2 == 1]
    chosen = []
    if count - 1 >= 1:
        chosen.append(random.choice(even))
    if count - 1 >= 2:
        odd_choice = random.choice([i for i in odd if i not in chosen])
        chosen.append(odd_choice)
    pool = [i for i in remaining if i not in chosen]
    random.shuffle(pool)
    while len(chosen) < count - 1:
        chosen.append(pool.pop())
    random.shuffle(chosen)
    return [template_idx] + chosen


def _patch_with_mapping(
    variant: np.ndarray,
    template_colors: set[int],
    color_map: dict[int, int],
    missing_color: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    partial = np.zeros_like(variant)
    complete = np.zeros_like(variant)

    role_mask = (variant == SKELETON) | (variant == ANCHOR)
    partial[role_mask] = variant[role_mask]
    complete[role_mask] = variant[role_mask]

    for r, c in np.argwhere(np.isin(variant, list(template_colors))):
        tpl_color = int(variant[int(r), int(c)])
        mapped = int(color_map[tpl_color])
        complete[int(r), int(c)] = mapped
        if missing_color is None or tpl_color != int(missing_color):
            partial[int(r), int(c)] = mapped
    return partial, complete


def _place_specs(
    height: int,
    width: int,
    specs: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]] | None:
    grid = np.zeros((height, width), dtype=int)
    expected = np.zeros_like(grid)
    blocked = np.zeros_like(grid, dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(specs)

    order = sorted(
        range(len(specs)),
        key=lambda i: -int(np.count_nonzero(np.asarray(specs[i]["complete"]))),
    )
    for i in order:
        partial = np.asarray(specs[i]["partial"], dtype=int)
        complete = np.asarray(specs[i]["complete"], dtype=int)
        h, w = map(int, complete.shape)
        reserve = np.argwhere(complete != BG)
        options = [(r, c) for r in range(height - h + 1) for c in range(width - w + 1)]
        random.shuffle(options)

        placed = False
        for r0, c0 in options:
            if any(bool(blocked[r0 + int(dr), c0 + int(dc)]) for dr, dc in reserve):
                continue
            for dr, dc in np.argwhere(partial != BG):
                grid[r0 + int(dr), c0 + int(dc)] = int(partial[int(dr), int(dc)])
            for dr, dc in reserve:
                expected[r0 + int(dr), c0 + int(dc)] = int(complete[int(dr), int(dc)])
                rr = r0 + int(dr)
                cc = c0 + int(dc)
                for nr in range(max(0, rr - GEN_PLACEMENT_PADDING), min(height, rr + GEN_PLACEMENT_PADDING + 1)):
                    for nc in range(max(0, cc - GEN_PLACEMENT_PADDING), min(width, cc + GEN_PLACEMENT_PADDING + 1)):
                        blocked[nr, nc] = True
            positions[i] = (r0, c0)
            placed = True
            break

        if not placed:
            return None

    return grid, expected, [pos for pos in positions if pos is not None]


def _alignment_count_against_template(grid: np.ndarray, components: list[np.ndarray], template_id: int) -> list[int]:
    template_patch = _component_patch(grid, components[template_id])
    template_colors = {
        int(v) for v in np.unique(template_patch) if int(v) not in (BG, ANCHOR, SKELETON)
    }
    variants = _dihedral_variants(template_patch)
    counts = []
    for i, comp in enumerate(components):
        if i == template_id:
            counts.append(0)
            continue
        skeleton, anchor, ann = _component_roles(grid, comp)
        match_count = 0
        for variant in variants:
            if _candidate_fill(variant, skeleton, anchor, ann, template_colors) is not None:
                match_count += 1
        counts.append(match_count)
    return counts


def _semantic_instance_ok(
    grid: np.ndarray,
    expected: np.ndarray,
    specs: list[dict[str, object]],
) -> bool:
    if grid.ndim != 2 or expected.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False

    components = _find_components(grid != BG)
    if len(components) != len(specs):
        return False

    roles = [_component_roles(grid, comp) for comp in components]
    ann_counts = [len(ann) for _s, _a, ann in roles]
    template_ids = [i for i, count in enumerate(ann_counts) if count == max(ann_counts)]
    if len(template_ids) != 1:
        return False
    template_id = template_ids[0]

    template_patch = _component_patch(grid, components[template_id])
    template_colors = {
        int(v) for v in np.unique(template_patch) if int(v) not in (BG, ANCHOR, SKELETON)
    }
    if len(template_colors) not in (2, 3):
        return False
    if ann_counts[template_id] != len(template_colors):
        return False
    if any(
        ann_counts[i] != len(template_colors) - 1
        for i in range(len(components))
        if i != template_id
    ):
        return False

    align_counts = _alignment_count_against_template(grid, components, template_id)
    if any(
        align_counts[i] != 1
        for i in range(len(components))
        if i != template_id
    ):
        return False

    try:
        produced = main(grid)
    except Exception:
        return False
    return bool(np.array_equal(produced, expected))


def _attempt_generation() -> tuple[np.ndarray, np.ndarray] | None:
    template_patch, template_colors = _choose_template_patch()
    variants = _dihedral_variants(template_patch)
    motif_count = random.randint(*GEN_MOTIF_COUNT_RANGE)
    orientation_ids = _orientation_indices(motif_count)

    specs: list[dict[str, object]] = []
    template_idx = orientation_ids[0]
    template_variant = variants[template_idx]
    specs.append(
        {
            "partial": template_variant.copy(),
            "complete": template_variant.copy(),
        }
    )

    incomplete_count = motif_count - 1
    perm_maps = _permutation_maps(template_colors, incomplete_count)
    missing_colors = _missing_template_colors(template_colors, incomplete_count)
    order = list(range(incomplete_count))
    random.shuffle(order)

    for j, orient in enumerate(orientation_ids[1:]):
        variant = variants[orient]
        perm_map = perm_maps[order[j]]
        missing = missing_colors[j]
        partial, complete = _patch_with_mapping(
            variant,
            set(template_colors),
            perm_map,
            missing,
        )
        if len({int(v) for v in np.unique(partial) if int(v) not in (BG, ANCHOR, SKELETON)}) != len(template_colors) - 1:
            return None
        specs.append({"partial": partial, "complete": complete})

    height = random.randint(*GEN_GRID_RANGE)
    width = random.randint(*GEN_GRID_RANGE)
    placed = _place_specs(height, width, specs)
    if placed is None:
        return None
    grid, expected, _positions = placed

    if not _semantic_instance_ok(grid, expected, specs):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        generated = _attempt_generation()
        if generated is not None:
            grid, _expected = generated
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
