from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
STAGE_SIZE = 30
GEN_ATTEMPTS = 500


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _nested_patch(colors_outer_to_inner: list[int], core_size: int) -> np.ndarray:
    n = len(colors_outer_to_inner)
    size = int(core_size + 2 * (n - 1))
    patch = np.full((size, size), int(colors_outer_to_inner[0]), dtype=int)
    for depth in range(1, n):
        inner_size = int(size - 2 * depth)
        patch[depth : depth + inner_size, depth : depth + inner_size] = int(colors_outer_to_inner[depth])
    return patch


def _parse_seed(grid: np.ndarray, coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h != w or h < 3 or h % 2 == 0:
        return None
    if int(coords.shape[0]) != h * w:
        return None

    sub = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    if bool((sub == BLACK).any()):
        return None

    layers = (h + 1) // 2
    colors: list[int] = []
    for depth in range(layers):
        inner = sub[depth : h - depth, depth : w - depth]
        if inner.shape == (1, 1):
            ring_vals = np.array([int(inner[0, 0])], dtype=int)
        else:
            ring_mask = np.zeros(inner.shape, dtype=bool)
            ring_mask[0, :] = True
            ring_mask[-1, :] = True
            ring_mask[:, 0] = True
            ring_mask[:, -1] = True
            ring_vals = inner[ring_mask]
        uniq = np.unique(ring_vals)
        if len(uniq) != 1:
            return None
        colors.append(int(uniq[0]))

    if len(set(colors)) != len(colors):
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "colors": colors,
        "layers": int(layers),
    }


def _extract_objects(grid: np.ndarray) -> tuple[list[dict], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    comps = _find_components(g != BLACK)
    seeds: list[dict] = []
    markers: list[tuple[int, int]] = []

    for coords in comps:
        if int(coords.shape[0]) == 1:
            r, c = map(int, coords[0])
            if int(g[r, c]) == GREY:
                markers.append((int(r), int(c)))
            continue
        seed = _parse_seed(g, coords)
        if seed is not None:
            seeds.append(seed)

    seeds.sort(key=lambda obj: (int(obj["bbox"][0]), int(obj["bbox"][1])))
    markers.sort()
    return seeds, markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a larger concentric square at each grey marker using the matched seed's layer count as the core scale.

    Input structure:
    - The background is black.
    - Several disconnected seed objects are filled odd-sized squares made of 1-pixel concentric color layers, ending in a 1x1 core.
    - Separate grey 1x1 singleton markers appear elsewhere, with the same count as the seeds.

    Rule:
    - Sort the seed objects by their top-left coordinates and sort the grey singleton markers the same way; pair them in order.
    - For a seed with n layers, enlarge only its innermost core color into an n x n solid square.
    - Re-wrap that enlarged core with 1-pixel square borders using the seed's original colors from inner-adjacent layer out to the outermost layer.
    - Draw the rebuilt concentric square centered on the paired grey marker, overwriting the marker while leaving the original seeds in place.

    Key insight:
    - The seed's nesting depth is itself the scale factor, and the enlargement applies only to the core before the original layer order is reconstructed around it.
    """
    g = np.asarray(grid, dtype=int)
    seeds, markers = _extract_objects(g)
    assert seeds and len(seeds) == len(markers)

    out = g.copy()
    for seed, marker in zip(seeds, markers):
        patch = _nested_patch(seed["colors"], int(seed["layers"]))
        size = int(patch.shape[0])
        offset = (size - 1) // 2
        r0 = int(marker[0] - offset)
        c0 = int(marker[1] - offset)
        r1 = int(r0 + size)
        c1 = int(c0 + size)
        assert 0 <= r0 and 0 <= c0 and r1 <= g.shape[0] and c1 <= g.shape[1]
        out[r0:r1, c0:c1] = patch

    return out


def _rect_conflict(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _place_rect(h: int, w: int, blocked: list[tuple[int, int, int, int]], gap: int) -> tuple[int, int] | None:
    candidates = [
        (r0, c0)
        for r0 in range(STAGE_SIZE - h + 1)
        for c0 in range(STAGE_SIZE - w + 1)
        if all(not _rect_conflict((r0, c0, r0 + h - 1, c0 + w - 1), other, gap) for other in blocked)
    ]
    if not candidates:
        return None
    return random.choice(candidates)


def _crop_bounds(rects: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    min_r = min(r0 for r0, _c0, _r1, _c1 in rects)
    min_c = min(c0 for _r0, c0, _r1, _c1 in rects)
    max_r = max(r1 for _r0, _c0, r1, _c1 in rects)
    max_c = max(c1 for _r0, _c0, _r1, c1 in rects)

    top = random.randint(0, min_r)
    left = random.randint(0, min_c)
    bottom = random.randint(max_r, STAGE_SIZE - 1)
    right = random.randint(max_c, STAGE_SIZE - 1)
    return int(top), int(left), int(bottom), int(right)


def _gen_attempt() -> np.ndarray | None:
    stage = np.full((STAGE_SIZE, STAGE_SIZE), BLACK, dtype=int)
    num_seeds = random.randint(1, 3)

    seed_records: list[dict] = []
    seed_rects: list[tuple[int, int, int, int]] = []
    blocked_for_markers: list[tuple[int, int, int, int]] = []

    for _ in range(num_seeds):
        layers = random.randint(2, 4)
        colors = random.sample(COLOR_POOL, layers)
        patch = _nested_patch(colors, 1)
        pos = _place_rect(int(patch.shape[0]), int(patch.shape[1]), seed_rects, gap=2)
        if pos is None:
            return None
        r0, c0 = pos
        h, w = map(int, patch.shape)
        stage[r0 : r0 + h, c0 : c0 + w] = patch
        bbox = (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))
        seed_records.append({"bbox": bbox, "colors": colors, "layers": layers})
        seed_rects.append(bbox)
        blocked_for_markers.append(bbox)

    marker_positions: list[tuple[int, int]] = []
    marker_rects: list[tuple[int, int, int, int]] = []
    for _ in range(num_seeds):
        pos = _place_rect(1, 1, blocked_for_markers + marker_rects, gap=1)
        if pos is None:
            return None
        r0, c0 = pos
        stage[r0, c0] = GREY
        marker_positions.append((int(r0), int(c0)))
        marker_rects.append((int(r0), int(c0), int(r0), int(c0)))

    seeds_sorted = sorted(seed_records, key=lambda obj: (int(obj["bbox"][0]), int(obj["bbox"][1])))
    markers_sorted = sorted(marker_positions)

    expansion_rects: list[tuple[int, int, int, int]] = []
    patches: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for seed, marker in zip(seeds_sorted, markers_sorted):
        patch = _nested_patch(seed["colors"], int(seed["layers"]))
        size = int(patch.shape[0])
        offset = (size - 1) // 2
        r0 = int(marker[0] - offset)
        c0 = int(marker[1] - offset)
        r1 = int(r0 + size - 1)
        c1 = int(c0 + size - 1)
        if r0 < 0 or c0 < 0 or r1 >= STAGE_SIZE or c1 >= STAGE_SIZE:
            return None
        bbox = (r0, c0, r1, c1)
        if any(_rect_conflict(bbox, other, gap=1) for other in seed_rects):
            return None
        if any(_rect_conflict(bbox, other, gap=1) for other in expansion_rects):
            return None
        expansion_rects.append(bbox)
        patches.append((bbox, patch))

    crop = _crop_bounds(seed_rects + expansion_rects)
    top, left, bottom, right = crop
    inp = stage[top : bottom + 1, left : right + 1].copy()
    expected = inp.copy()

    for (r0, c0, _r1, _c1), patch in patches:
        rr = int(r0 - top)
        cc = int(c0 - left)
        h, w = map(int, patch.shape)
        expected[rr : rr + h, cc : cc + w] = patch

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after %d attempts" % GEN_ATTEMPTS)
