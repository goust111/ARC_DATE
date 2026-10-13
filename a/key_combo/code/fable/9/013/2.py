from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


MARKER_COLOR = 5
RING_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
BG_COLOR = 0

GEN_RETRY_LIMIT = 500
GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
GROUP_DEPTH_SINGLE = [3, 4, 5]
GROUP_DEPTH_DOUBLE = [(3, 3), (3, 4), (4, 3)]
GROUP_SEP_RANGE = (2, 5)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_color_components(grid: np.ndarray, color: int) -> list[dict]:
    mask = grid == int(color)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "area": int(len(cells)),
                    "bbox": _bbox_coords(coords),
                }
            )
    return comps


def _find_ring_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg), int(MARKER_COLOR)}):
        comps.extend(_find_color_components(grid, color))
    return comps


def _is_rectangular_perimeter(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False
    coords = comp["coords"]
    on_border = (
        (coords[:, 0] == r0)
        | (coords[:, 0] == r1)
        | (coords[:, 1] == c0)
        | (coords[:, 1] == c1)
    )
    return int(comp["area"]) == 2 * h + 2 * w - 4 and bool(np.all(on_border))


def _strictly_contains(outer: dict, inner: dict) -> bool:
    r0, c0, r1, c1 = outer["bbox"]
    rr0, cc0, rr1, cc1 = inner["bbox"]
    return r0 < rr0 and c0 < cc0 and rr1 < r1 and cc1 < c1


def _center_key(comp: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = comp["bbox"]
    return int(r0 + r1), int(c0 + c1)


def _bbox_area(comp: dict) -> int:
    r0, c0, r1, c1 = comp["bbox"]
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _build_chains(comps: list[dict]) -> list[list[dict]]:
    assert comps, "expected ring components"
    centers = {_center_key(comp) for comp in comps}
    assert len(centers) <= len(comps)

    parent: list[int | None] = [None] * len(comps)
    children: list[list[int]] = [[] for _ in comps]
    for i, comp in enumerate(comps):
        candidates = [
            ( _bbox_area(other), j)
            for j, other in enumerate(comps)
            if i != j and _center_key(other) == _center_key(comp) and _strictly_contains(other, comp)
        ]
        if candidates:
            parent_idx = min(candidates)[1]
            parent[i] = int(parent_idx)
            children[parent_idx].append(i)

    roots = [i for i, p in enumerate(parent) if p is None]
    assert roots, "expected at least one nested group"

    seen = set()
    chains: list[list[dict]] = []
    for root in roots:
        cur = int(root)
        chain: list[dict] = []
        while True:
            assert cur not in seen, "containment graph must be a forest of chains"
            seen.add(cur)
            chain.append(comps[cur])
            assert len(children[cur]) <= 1, "each group must be a single nesting chain"
            if not children[cur]:
                break
            cur = int(children[cur][0])
        chains.append(chain)

    assert len(seen) == len(comps), "every ring must belong to exactly one chain"
    assert all(len(chain) >= 3 for chain in chains), "each group must have depth at least three"
    return chains


def _cyclic_shift_outer_to_inner(colors: list[int], k: int) -> list[int]:
    n = len(colors)
    shift = int(k) % int(n)
    if shift == 0:
        return list(colors)
    return list(colors[-shift:]) + list(colors[:-shift])


def _draw_perimeter(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cycle the outer-to-inner colors of each nested ring group by the number of grey markers in its innermost cavity.

    Input structure:
    - The background is the dominant color.
    - Foreground consists of several separated groups of concentric monochrome rectangular perimeters.
    - Each group is a single strict-containment chain of depth at least three, and every ring color differs within that group.
    - Grey marker pixels appear only inside the cavity enclosed by the innermost ring of each group.

    Rule:
    - For each nested group, count the grey marker pixels in the innermost cavity; call this count k.
    - Read that group's ring colors from outermost to innermost and cyclically shift the sequence by k positions, so each ring passes its color to the k-th ring inward with wraparound.
    - Keep every ring's shape and position fixed, repaint only the ring pixels, and erase all grey markers back to background.

    Key insight:
    - The nested rectangles act like a hidden one-dimensional color cycle, and the number of deepest markers specifies how far that cycle rotates.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    assert bg != MARKER_COLOR, "marker color must be distinct from background"

    comps = _find_ring_components(g, bg)
    assert comps, "expected rectangular ring components"
    assert all(_is_rectangular_perimeter(comp) for comp in comps), "all foreground components must be 1-cell rectangular perimeters"

    chains = _build_chains(comps)

    marker_cover = np.zeros(g.shape, dtype=int)
    shifts: list[int] = []
    for chain in chains:
        innermost = chain[-1]
        r0, c0, r1, c1 = innermost["bbox"]
        cavity = g[r0 + 1 : r1, c0 + 1 : c1]
        assert cavity.size > 0, "innermost ring must have a non-empty cavity"
        assert np.all((cavity == bg) | (cavity == MARKER_COLOR)), "markers may only occupy the deepest cavity"
        marker_mask = cavity == MARKER_COLOR
        count = int(marker_mask.sum())
        assert count > 0, "each deepest cavity must contain at least one marker"
        marker_cover[r0 + 1 : r1, c0 + 1 : c1] += marker_mask.astype(int)
        shifts.append(count)

    assert np.any(g == MARKER_COLOR), "expected grey markers"
    assert np.all(marker_cover[g == MARKER_COLOR] == 1), "every grey marker must belong to exactly one deepest cavity"

    out = g.copy()
    out[g == MARKER_COLOR] = bg
    for chain, k in zip(chains, shifts):
        new_colors = _cyclic_shift_outer_to_inner([int(comp["color"]) for comp in chain], k)
        for comp, color in zip(chain, new_colors):
            rr, cc = comp["coords"][:, 0], comp["coords"][:, 1]
            out[rr, cc] = int(color)
    return out


def _sample_group_count() -> tuple[int, tuple[int, ...]]:
    if random.random() < 0.72:
        return 2, random.choice(GROUP_DEPTH_DOUBLE)
    return 1, (random.choice(GROUP_DEPTH_SINGLE),)


def _group_sizes(cavity_h: int, cavity_w: int, gaps: list[int]) -> list[tuple[int, int]]:
    sizes = [(int(cavity_h) + 2, int(cavity_w) + 2)]
    for gap in gaps:
        prev_h, prev_w = sizes[-1]
        grow = 2 * (int(gap) + 1)
        sizes.append((prev_h + grow, prev_w + grow))
    return sizes


def _sample_group_geometry(depth: int) -> dict | None:
    for _ in range(120):
        cavity_h = random.randint(2, 5 if depth <= 4 else 6)
        cavity_w = random.randint(2, 6)
        gaps = [random.randint(1, 2) for _ in range(depth - 1)]
        sizes = _group_sizes(cavity_h, cavity_w, gaps)
        outer_h, outer_w = sizes[-1]
        if outer_h > 26 or outer_w > 26:
            continue
        return {
            "depth": int(depth),
            "cavity_h": int(cavity_h),
            "cavity_w": int(cavity_w),
            "gaps": [int(x) for x in gaps],
            "sizes_inner_to_outer": sizes,
            "outer_h": int(outer_h),
            "outer_w": int(outer_w),
        }
    return None


def _count_candidates(depth: int, cavity_area: int) -> tuple[list[int], list[int]]:
    low = [k for k in range(1, min(cavity_area, depth) + 1) if k % depth != 0]
    high = [k for k in range(depth + 1, min(cavity_area, 2 * depth + 3) + 1) if k % depth != 0]
    if not low and not high:
        return [], []
    if not low:
        low = [high[0]]
    return low, high


def _choose_marker_counts(groups: list[dict]) -> list[int] | None:
    high_indices = [i for i, g in enumerate(groups) if g["high_counts"]]
    forced_high = random.choice(high_indices) if high_indices and random.random() < 0.65 else None

    counts: list[int] = []
    used_mods: set[int] = set()
    for i, group in enumerate(groups):
        depth = int(group["depth"])
        choices = list(group["low_counts"])
        if i == forced_high:
            choices = list(group["high_counts"])
        elif group["high_counts"] and random.random() < 0.35:
            choices += list(group["high_counts"])
        choices = sorted(set(int(x) for x in choices))
        if not choices:
            return None
        mod_fresh = [k for k in choices if (k % depth) not in used_mods]
        chosen = random.choice(mod_fresh if mod_fresh else choices)
        counts.append(int(chosen))
        used_mods.add(int(chosen) % depth)
    return counts


def _deterministic_marker_cells(h: int, w: int, count: int, salt: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(int(h)) for c in range(int(w))]
    rng = random.Random(int(salt))
    rng.shuffle(cells)
    return [(int(r), int(c)) for r, c in cells[: int(count)]]


def _build_group_patch(group: dict, colors: list[int], marker_count: int, salt: int) -> tuple[np.ndarray, dict]:
    outer_h = int(group["outer_h"])
    outer_w = int(group["outer_w"])
    patch = np.full((outer_h, outer_w), BG_COLOR, dtype=int)

    ring_bboxes: list[tuple[int, int, int, int]] = []
    for (h, w), color in zip(reversed(group["sizes_inner_to_outer"]), colors):
        top = (outer_h - int(h)) // 2
        left = (outer_w - int(w)) // 2
        bbox = (top, left, top + int(h) - 1, left + int(w) - 1)
        ring_bboxes.append(bbox)
        _draw_perimeter(patch, bbox, int(color))

    r0, c0, r1, c1 = ring_bboxes[-1]
    cavity_h = int(r1 - r0 - 1)
    cavity_w = int(c1 - c0 - 1)
    cells = _deterministic_marker_cells(cavity_h, cavity_w, marker_count, salt)
    for rr, cc in cells:
        patch[r0 + 1 + rr, c0 + 1 + cc] = MARKER_COLOR

    meta = {
        "depth": int(group["depth"]),
        "colors": [int(c) for c in colors],
        "marker_count": int(marker_count),
        "local_ring_bboxes": ring_bboxes,
    }
    return patch, meta


def _feasible_orientations(sizes: list[tuple[int, int]]) -> list[str]:
    if len(sizes) == 1:
        return ["single"]
    (h1, w1), (h2, w2) = sizes
    out = []
    if w1 + w2 + GROUP_SEP_RANGE[0] <= 30 and max(h1, h2) <= 30:
        out.append("horizontal")
    if h1 + h2 + GROUP_SEP_RANGE[0] <= 30 and max(w1, w2) <= 30:
        out.append("vertical")
    return out


def _place_groups(patches: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    sizes = [(int(p.shape[0]), int(p.shape[1])) for p in patches]
    orientations = _feasible_orientations(sizes)
    if not orientations:
        return None

    if len(patches) == 1:
        ph, pw = sizes[0]
        h = random.randint(max(GRID_H_RANGE[0], ph), GRID_H_RANGE[1])
        w = random.randint(max(GRID_W_RANGE[0], pw), GRID_W_RANGE[1])
        grid = np.full((h, w), BG_COLOR, dtype=int)
        r = random.randint(0, h - ph)
        c = random.randint(0, w - pw)
        grid[r : r + ph, c : c + pw] = np.where(patches[0] != BG_COLOR, patches[0], grid[r : r + ph, c : c + pw])
        return grid, [(r, c)]

    order = list(range(len(patches)))
    random.shuffle(order)
    patches = [patches[i] for i in order]
    sizes = [sizes[i] for i in order]

    orientation = random.choice(orientations)
    sep = random.randint(*GROUP_SEP_RANGE)
    (h1, w1), (h2, w2) = sizes

    if orientation == "horizontal":
        min_h = max(h1, h2)
        min_w = w1 + w2 + sep
        if min_h > 30 or min_w > 30:
            return None
        h = random.randint(max(GRID_H_RANGE[0], min_h), GRID_H_RANGE[1])
        w = random.randint(max(GRID_W_RANGE[0], min_w), GRID_W_RANGE[1])
        extra_w = w - min_w
        left = random.randint(0, extra_w)
        remaining = extra_w - left
        middle_extra = random.randint(0, remaining)
        c1 = left
        c2 = left + w1 + sep + middle_extra
        r1 = random.randint(0, h - h1)
        r2 = random.randint(0, h - h2)
        placements = [(r1, c1), (r2, c2)]
    else:
        min_h = h1 + h2 + sep
        min_w = max(w1, w2)
        if min_h > 30 or min_w > 30:
            return None
        h = random.randint(max(GRID_H_RANGE[0], min_h), GRID_H_RANGE[1])
        w = random.randint(max(GRID_W_RANGE[0], min_w), GRID_W_RANGE[1])
        extra_h = h - min_h
        top = random.randint(0, extra_h)
        remaining = extra_h - top
        middle_extra = random.randint(0, remaining)
        r1 = top
        r2 = top + h1 + sep + middle_extra
        c1 = random.randint(0, w - w1)
        c2 = random.randint(0, w - w2)
        placements = [(r1, c1), (r2, c2)]

    grid = np.full((h, w), BG_COLOR, dtype=int)
    for patch, (r, c) in zip(patches, placements):
        ph, pw = patch.shape
        grid[r : r + ph, c : c + pw] = np.where(patch != BG_COLOR, patch, grid[r : r + ph, c : c + pw])

    inverse = [0] * len(order)
    for new_i, old_i in enumerate(order):
        inverse[old_i] = new_i
    restored = [placements[inverse[i]] for i in range(len(order))]
    return grid, restored


def _expected_from_meta(inp: np.ndarray, metas: list[dict]) -> np.ndarray:
    out = inp.copy()
    out[out == MARKER_COLOR] = BG_COLOR
    for meta in metas:
        shifted = _cyclic_shift_outer_to_inner(meta["colors"], meta["marker_count"])
        for bbox, color in zip(meta["global_ring_bboxes"], shifted):
            _draw_perimeter(out, bbox, color)
    return out


def _augment_meta(metas: list[dict], placements: list[tuple[int, int]]) -> None:
    for meta, (dr, dc) in zip(metas, placements):
        global_bboxes = []
        for r0, c0, r1, c1 in meta["local_ring_bboxes"]:
            global_bboxes.append((r0 + dr, c0 + dc, r1 + dr, c1 + dc))
        meta["global_ring_bboxes"] = global_bboxes


def _semantic_checks(inp: np.ndarray, out: np.ndarray, metas: list[dict]) -> bool:
    if not isinstance(inp, np.ndarray) or not isinstance(out, np.ndarray):
        return False
    if inp.ndim != 2 or out.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if np.array_equal(inp, out):
        return False
    if _get_bg(inp) != BG_COLOR:
        return False
    if not np.any(inp == MARKER_COLOR):
        return False
    if int(np.count_nonzero(inp != BG_COLOR)) >= (inp.size // 2):
        return False
    if any(meta["marker_count"] % meta["depth"] == 0 for meta in metas):
        return False
    if not np.array_equal(out, _expected_from_meta(inp, metas)):
        return False
    try:
        solved = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(solved, out))


def _gen_attempt() -> np.ndarray | None:
    group_count, depth_tuple = _sample_group_count()
    assert group_count == len(depth_tuple)

    groups = []
    for depth in depth_tuple:
        geom = _sample_group_geometry(int(depth))
        if geom is None:
            return None
        cavity_area = int(geom["cavity_h"]) * int(geom["cavity_w"])
        low_counts, high_counts = _count_candidates(int(depth), cavity_area)
        if not low_counts and not high_counts:
            return None
        geom["low_counts"] = low_counts
        geom["high_counts"] = high_counts
        groups.append(geom)

    if not _feasible_orientations([(g["outer_h"], g["outer_w"]) for g in groups]):
        return None

    counts = _choose_marker_counts(groups)
    if counts is None:
        return None

    total_rings = sum(int(g["depth"]) for g in groups)
    if total_rings > len(RING_COLORS):
        return None
    color_pool = random.sample(RING_COLORS, total_rings)

    patches = []
    metas = []
    cursor = 0
    for i, (group, marker_count) in enumerate(zip(groups, counts)):
        depth = int(group["depth"])
        colors = color_pool[cursor : cursor + depth]
        cursor += depth
        patch, meta = _build_group_patch(group, colors, int(marker_count), salt=1009 + 37 * i + 17 * marker_count)
        patches.append(patch)
        metas.append(meta)

    placed = _place_groups(patches)
    if placed is None:
        return None
    inp, placements = placed
    _augment_meta(metas, placements)
    out = _expected_from_meta(inp, metas)
    if not _semantic_checks(inp, out, metas):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-ring cyclic-shift input")
