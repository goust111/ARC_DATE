from __future__ import annotations

from collections import Counter
import random

import numpy as np


GEN_BG = 0
GEN_MARKER_COLORS = (1, 3)
GEN_ANCHOR_COLOR = 2
GEN_SKELETON_COLOR = 4

GEN_GRID_ROWS = (17, 24)
GEN_GRID_COLS = (17, 24)
GEN_TOTAL_MOTIFS = (3, 4)  # one full reference + 2-3 incomplete copies
GEN_SKELETON_SIZE = (5, 6)
GEN_MARKER_COUNT = (4, 6)
GEN_BBOX_MARGIN = 2
GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> np.ndarray:
    r0 = int(r0)
    c0 = int(c0)
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > dst.shape[0] or c0 + w > dst.shape[1]:
        raise ValueError("paste out of bounds")
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]
    return dst


def find_components(grid: np.ndarray, *, bg: int, connectivity: int = 8) -> list[dict]:
    assert connectivity == 8
    mask = grid != int(bg)
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(sorted(cells), dtype=int)
            comps.append({"coords": coords, "bbox": bbox_coords(coords), "area": int(coords.shape[0])})
    return comps


def _component_patch(grid: np.ndarray, comp: dict) -> np.ndarray:
    return crop_bbox(grid, comp["bbox"])


def _cell_key(patch: np.ndarray, colors: set[int]) -> tuple[tuple[int, int, int], ...]:
    coords = np.argwhere(np.isin(patch, list(colors)))
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    return tuple(sorted((int(r - r0), int(c - c0), int(patch[r, c])) for r, c in coords))


def _dihedral_variants(patch: np.ndarray) -> list[dict]:
    variants = []
    seen = set()
    for k in range(4):
        rot = np.rot90(patch, k)
        for mirrored, arr in ((False, rot), (True, np.fliplr(rot))):
            key = (arr.shape, arr.tobytes(), mirrored)
            if key in seen:
                continue
            seen.add(key)
            variants.append({"patch": arr.copy(), "mirrored": mirrored})
    return variants


def _infer_role_colors(grid: np.ndarray, comps: list[dict], bg: int) -> tuple[int, int, tuple[int, int]]:
    color_sets = []
    count_maps = []
    for comp in comps:
        values = [int(grid[int(r), int(c)]) for r, c in comp["coords"]]
        counts = Counter(values)
        color_sets.append(set(counts))
        count_maps.append(counts)

    shared = set.intersection(*color_sets)
    anchor_candidates = [color for color in shared if all(counts[color] == 1 for counts in count_maps)]
    assert len(anchor_candidates) == 1, "expected exactly one shared one-pixel anchor color"
    anchor = int(anchor_candidates[0])

    skeleton_candidates = [color for color in shared if color != anchor]
    assert len(skeleton_candidates) == 1, "expected exactly one shared skeleton color"
    skeleton = int(skeleton_candidates[0])

    extras = sorted(set(map(int, grid.flatten())) - {int(bg), skeleton, anchor})
    assert len(extras) == 2, "expected exactly two marker colors"
    return skeleton, anchor, (int(extras[0]), int(extras[1]))


def _aligned_marker_cells(
    variant: np.ndarray,
    *,
    mirrored: bool,
    target_skeleton: set[tuple[int, int]],
    target_anchor: tuple[int, int],
    skeleton: int,
    anchor: int,
    marker_colors: tuple[int, int],
) -> tuple[tuple[int, int, int], ...] | None:
    anchor_coords = np.argwhere(variant == int(anchor))
    if len(anchor_coords) != 1:
        return None
    ar, ac = map(int, anchor_coords[0])
    tr, tc = map(int, target_anchor)
    dr, dc = tr - ar, tc - ac

    variant_skeleton = {(int(r) + dr, int(c) + dc) for r, c in np.argwhere(variant == int(skeleton))}
    if variant_skeleton != target_skeleton:
        return None

    swap = {marker_colors[0]: marker_colors[1], marker_colors[1]: marker_colors[0]} if mirrored else {}
    marker_cells = []
    for r, c in np.argwhere(np.isin(variant, list(marker_colors))):
        color = int(variant[int(r), int(c)])
        marker_cells.append((int(r) + dr, int(c) + dc, int(swap.get(color, color))))
    return tuple(sorted(marker_cells))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the template's full annotation onto each aligned sprite, but swap the two marker colors on mirrored copies.

    Input structure:
    - A dominant background contains several 8-connected sprites.
    - Every sprite shares one skeleton color and exactly one anchor-color pixel.
    - Exactly one sprite also carries two additional marker colors, while the other sprites contain only skeleton and anchor.

    Rule:
    - Treat the fully marked sprite as the template.
    - For every incomplete sprite, find the unique rotation/reflection plus translation that aligns the template's skeleton and anchor onto the target skeleton and anchor.
    - Stamp all transformed marker pixels onto the target, preserving the skeleton and anchor.
    - If that alignment uses a mirror, swap the two marker colors during stamping; if it is a pure rotation, keep the original marker colors.

    Key insight:
    - Orientation matching happens in two layers: first align the skeleton+anchor, then use the sprite's chirality to decide whether the marker palette must be exchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    comps = find_components(g, bg=bg, connectivity=8)
    assert len(comps) >= 3, "expected one template and multiple target sprites"

    skeleton, anchor, marker_colors = _infer_role_colors(g, comps, bg)
    reference_indices = [
        i
        for i, comp in enumerate(comps)
        if any(int(g[int(r), int(c)]) in marker_colors for r, c in comp["coords"])
    ]
    assert len(reference_indices) == 1, "expected exactly one fully annotated template"
    ref_i = reference_indices[0]
    reference_patch = _component_patch(g, comps[ref_i])
    variants = _dihedral_variants(reference_patch)

    out = g.copy()
    for i, comp in enumerate(comps):
        if i == ref_i:
            continue
        colors = {int(g[int(r), int(c)]) for r, c in comp["coords"]}
        assert colors <= {skeleton, anchor}, "target sprites must be incomplete"

        target_skeleton = {
            (int(r), int(c))
            for r, c in comp["coords"]
            if int(g[int(r), int(c)]) == int(skeleton)
        }
        target_anchor_cells = [
            (int(r), int(c))
            for r, c in comp["coords"]
            if int(g[int(r), int(c)]) == int(anchor)
        ]
        assert len(target_anchor_cells) == 1, "each target sprite must contain one anchor"
        target_anchor = target_anchor_cells[0]

        candidates = []
        for item in variants:
            aligned = _aligned_marker_cells(
                item["patch"],
                mirrored=bool(item["mirrored"]),
                target_skeleton=target_skeleton,
                target_anchor=target_anchor,
                skeleton=skeleton,
                anchor=anchor,
                marker_colors=marker_colors,
            )
            if aligned is not None:
                candidates.append(aligned)
        unique = sorted(set(candidates))
        assert len(unique) == 1, "each target must have a unique aligned transform"

        for r, c, color in unique[0]:
            assert 0 <= r < out.shape[0] and 0 <= c < out.shape[1], "marker left the grid"
            assert int(out[r, c]) in (int(bg), int(color)), "marker conflicts with existing content"
            out[r, c] = int(color)
    return out


def _normalize_shape(shape: set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0 = min(r for r, _ in shape)
    c0 = min(c for _, c in shape)
    return frozenset((r - r0, c - c0) for r, c in shape)


_SKELETON_POOL_CACHE: dict[int, list[tuple[tuple[int, int], ...]]] = {}


def _skeleton_pool(size: int) -> list[tuple[tuple[int, int], ...]]:
    if size in _SKELETON_POOL_CACHE:
        return _SKELETON_POOL_CACHE[size]

    shapes: set[frozenset[tuple[int, int]]] = {frozenset({(0, 0)})}
    for _ in range(size - 1):
        nxt: set[frozenset[tuple[int, int]]] = set()
        for shape in shapes:
            frontier = set()
            for cell in shape:
                frontier.update(neighbors4(*cell))
            for cell in frontier - set(shape):
                grown = set(shape)
                grown.add(cell)
                nxt.add(_normalize_shape(grown))
        shapes = nxt

    pool = []
    for shape in shapes:
        rows = [r for r, _ in shape]
        cols = [c for _, c in shape]
        if max(rows) - min(rows) + 1 <= 5 and max(cols) - min(cols) + 1 <= 5:
            pool.append(tuple(sorted(shape)))
    _SKELETON_POOL_CACHE[size] = pool
    return pool


def _build_patch(cells: list[tuple[int, int, int]]) -> np.ndarray:
    coords = np.array([(r, c) for r, c, _ in cells], dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c, color in cells:
        patch[int(r - r0), int(c - c0)] = int(color)
    return patch


def _combo_variant_count(skeleton: tuple[tuple[int, int], ...], anchor: tuple[int, int]) -> int:
    patch = _build_patch(
        [(r, c, GEN_SKELETON_COLOR) for r, c in skeleton] + [(anchor[0], anchor[1], GEN_ANCHOR_COLOR)]
    )
    keys = {_cell_key(item["patch"], {GEN_SKELETON_COLOR, GEN_ANCHOR_COLOR}) for item in _dihedral_variants(patch)}
    return len(keys)


def _adjacent_open_cells(
    skeleton: tuple[tuple[int, int], ...],
    *,
    blocked: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    blocked = set() if blocked is None else set(blocked)
    skel = set(skeleton)
    out = set()
    for r, c in skel:
        for nr, nc in neighbors8(r, c):
            if (nr, nc) not in skel and (nr, nc) not in blocked:
                out.add((nr, nc))
    return sorted(out)


def _swap_marker_colors(patch: np.ndarray, marker_colors: tuple[int, int]) -> np.ndarray:
    a, b = marker_colors
    out = patch.copy()
    out[patch == a] = b
    out[patch == b] = a
    return out


def _erase_markers(patch: np.ndarray, marker_colors: tuple[int, int]) -> np.ndarray:
    out = patch.copy()
    out[np.isin(out, list(marker_colors))] = GEN_BG
    return out


def _patch_bbox(patch: np.ndarray, bg: int = GEN_BG) -> tuple[int, int, int, int]:
    coords = np.argwhere(patch != int(bg))
    return bbox_coords(coords)


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + margin < br0 or br1 + margin < ar0 or ac1 + margin < bc0 or bc1 + margin < ac0)


def _place_patches(patches: list[np.ndarray], rows: int, cols: int) -> list[tuple[int, int]] | None:
    order = sorted(range(len(patches)), key=lambda i: np.count_nonzero(patches[i]), reverse=True)
    placements: list[tuple[int, int] | None] = [None] * len(patches)
    placed_boxes: list[tuple[int, int, int, int]] = []

    for idx in order:
        patch = patches[idx]
        h, w = patch.shape
        candidates = [(r, c) for r in range(rows - h + 1) for c in range(cols - w + 1)]
        random.shuffle(candidates)
        local_box = _patch_bbox(patch)
        for r0, c0 in candidates:
            abs_box = (
                r0 + local_box[0],
                c0 + local_box[1],
                r0 + local_box[2],
                c0 + local_box[3],
            )
            if any(_boxes_too_close(abs_box, other, GEN_BBOX_MARGIN) for other in placed_boxes):
                continue
            placements[idx] = (r0, c0)
            placed_boxes.append(abs_box)
            break
        else:
            return None
    return [tuple(p) for p in placements if p is not None]


def _render_scene(
    rows: int,
    cols: int,
    input_patches: list[np.ndarray],
    output_patches: list[np.ndarray],
    placements: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros((rows, cols), dtype=int)
    out = np.zeros((rows, cols), dtype=int)
    for patch, (r, c) in zip(input_patches, placements):
        paste(inp, patch, r, c, transparent=GEN_BG)
    for patch, (r, c) in zip(output_patches, placements):
        paste(out, patch, r, c, transparent=GEN_BG)
    return inp, out


def _choose_reference_patch() -> np.ndarray | None:
    size = random.randint(*GEN_SKELETON_SIZE)
    skeleton = random.choice(_skeleton_pool(size))
    anchors = _adjacent_open_cells(skeleton)
    random.shuffle(anchors)
    for anchor in anchors:
        if _combo_variant_count(skeleton, anchor) != 8:
            continue
        marker_candidates = _adjacent_open_cells(skeleton, blocked={anchor})
        if len(marker_candidates) < GEN_MARKER_COUNT[0]:
            continue
        for _ in range(40):
            count = random.randint(*GEN_MARKER_COUNT)
            if count > len(marker_candidates):
                continue
            cells = random.sample(marker_candidates, count)
            split = random.randint(1, count - 1)
            first = set(cells[:split])
            colored = []
            for r, c in skeleton:
                colored.append((r, c, GEN_SKELETON_COLOR))
            colored.append((anchor[0], anchor[1], GEN_ANCHOR_COLOR))
            for r, c in cells:
                color = GEN_MARKER_COLORS[0] if (r, c) in first else GEN_MARKER_COLORS[1]
                colored.append((r, c, color))
            patch = _build_patch(colored)
            if len(np.unique(patch)) < 4:
                continue
            variants = _dihedral_variants(patch)
            if len(variants) < 8:
                continue
            return random.choice(variants)["patch"]
    return None


def _build_target_set(reference_patch: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], list[bool]] | None:
    variants = _dihedral_variants(reference_patch)
    pure = [item for item in variants if not item["mirrored"] and not np.array_equal(item["patch"], reference_patch)]
    mirrored = [item for item in variants if item["mirrored"]]
    if not pure or not mirrored:
        return None

    target_total = random.randint(GEN_TOTAL_MOTIFS[0] - 1, GEN_TOTAL_MOTIFS[1] - 1)
    first_pure = random.choice(pure)
    first_mirror = random.choice(mirrored)
    chosen = [first_pure, first_mirror]

    remaining = [
        item
        for item in variants
        if not np.array_equal(item["patch"], first_pure["patch"])
        and not np.array_equal(item["patch"], first_mirror["patch"])
        and not np.array_equal(item["patch"], reference_patch)
    ]
    random.shuffle(remaining)
    chosen.extend(remaining[: max(0, target_total - 2)])
    if len(chosen) != target_total:
        return None

    input_patches = [reference_patch]
    output_patches = [reference_patch]
    mirrored_flags = []
    for item in chosen:
        target_patch = item["patch"]
        expected = _swap_marker_colors(target_patch, GEN_MARKER_COLORS) if item["mirrored"] else target_patch
        output_patches.append(expected)
        input_patches.append(_erase_markers(expected, GEN_MARKER_COLORS))
        mirrored_flags.append(bool(item["mirrored"]))
    return input_patches, output_patches, mirrored_flags


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    output_patches: list[np.ndarray],
    placements: list[tuple[int, int]],
    mirrored_flags: list[bool],
) -> bool:
    if not mirrored_flags or all(mirrored_flags) or not any(mirrored_flags):
        return False
    if np.array_equal(inp, expected):
        return False
    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    naive_no_swap = np.zeros_like(expected)
    naive_all_swap = np.zeros_like(expected)
    for i, (patch, (r, c)) in enumerate(zip(output_patches, placements)):
        if i == 0:
            paste(naive_no_swap, patch, r, c, transparent=GEN_BG)
            paste(naive_all_swap, patch, r, c, transparent=GEN_BG)
            continue
        orig = _swap_marker_colors(patch, GEN_MARKER_COLORS) if mirrored_flags[i - 1] else patch
        swapped = _swap_marker_colors(orig, GEN_MARKER_COLORS)
        paste(naive_no_swap, orig, r, c, transparent=GEN_BG)
        paste(naive_all_swap, swapped, r, c, transparent=GEN_BG)
    if np.array_equal(naive_no_swap, expected):
        return False
    if np.array_equal(naive_all_swap, expected):
        return False

    bg = get_bg(inp)
    comps = find_components(inp, bg=bg, connectivity=8)
    if len(comps) != len(output_patches):
        return False

    outputs = {patch.tobytes() + str(patch.shape).encode() for patch in output_patches[1:]}
    return len(outputs) == len(output_patches) - 1


def _gen_attempt() -> np.ndarray | None:
    reference_patch = _choose_reference_patch()
    if reference_patch is None:
        return None

    target_set = _build_target_set(reference_patch)
    if target_set is None:
        return None
    input_patches, output_patches, mirrored_flags = target_set

    rows = random.randint(*GEN_GRID_ROWS)
    cols = random.randint(*GEN_GRID_COLS)
    placements = _place_patches(output_patches, rows, cols)
    if placements is None:
        return None

    inp, expected = _render_scene(rows, cols, input_patches, output_patches, placements)
    if not _semantic_checks(inp, expected, output_patches, placements, mirrored_flags):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chirality-color-transfer input")
