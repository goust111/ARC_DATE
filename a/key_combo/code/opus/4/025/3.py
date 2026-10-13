from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7

REF_COLORS = (BLUE, RED, GREEN, YELLOW, ORANGE)
OCCLUDER_SIZES = (2, 3)
MAX_GEN_ATTEMPTS = 400

# Curated compact irregular 4-connected sprites (normalized to top-left origin).
SHAPE_LIBRARY: tuple[frozenset[tuple[int, int]], ...] = (
    frozenset({(0, 0), (0, 1), (1, 0), (2, 0)}),
    frozenset({(0, 1), (1, 0), (1, 1), (1, 2)}),
    frozenset({(0, 0), (0, 1), (1, 1), (1, 2), (2, 1)}),
    frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)}),
    frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 0)}),
    frozenset({(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)}),
    frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)}),
    frozenset({(0, 0), (0, 1), (1, 1), (1, 2), (2, 1), (2, 2)}),
    frozenset({(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 1)}),
    frozenset({(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 1)}),
    frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 2)}),
    frozenset({(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)}),
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: list[tuple[int, int]] | set[tuple[int, int]] | np.ndarray) -> tuple[int, int, int, int]:
    pts = list(coords)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_coords(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[frozenset[tuple[int, int]], tuple[int, int]]:
    r0, c0, _r1, _c1 = _bbox_coords(coords)
    return frozenset((r - r0, c - c0) for r, c in coords), (r0, c0)


def _shape_dims(shape: frozenset[tuple[int, int]]) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox_coords(shape)
    return r1 - r0 + 1, c1 - c0 + 1


def _components_of_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _is_connected(coords: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> bool:
    pts = set(coords)
    if not pts:
        return False
    q = deque([next(iter(pts))])
    seen = {q[0]}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in pts and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(pts)


def _find_separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if bool(np.all(grid[:, c] == PINK))]
    assert len(cols) == 1, "expected exactly one full-height magenta separator"
    return cols[0]


def _extract_references(grid: np.ndarray, sep_c: int) -> list[dict]:
    left = grid[:, :sep_c]
    comps = _components_of_mask((left != BG) & (left != PINK))
    refs = []
    for comp in comps:
        colors = {int(left[r, c]) for r, c in comp}
        assert len(colors) == 1, "reference sprite must be monochrome"
        color = colors.pop()
        assert color in REF_COLORS, "unexpected reference color"
        shape, origin = _normalize_coords(comp)
        refs.append({"shape": shape, "color": color, "origin": origin})
    assert 3 <= len(refs) <= 5, "expected 3 to 5 reference sprites"
    assert len({ref["shape"] for ref in refs}) == len(refs), "reference shapes must be distinct"
    assert len({ref["color"] for ref in refs}) == len(refs), "reference colors must be distinct"
    return refs


def _extract_visible_right(grid: np.ndarray, sep_c: int) -> list[dict]:
    right = grid[:, sep_c + 1 :]
    comps = _components_of_mask(right == GREY)
    visibles = []
    for comp in comps:
        shape, origin = _normalize_coords(comp)
        visibles.append({"shape": shape, "origin": origin, "coords": comp})
    return visibles


def _candidate_offsets(ref_shape: frozenset[tuple[int, int]], vis_shape: frozenset[tuple[int, int]]) -> list[tuple[int, int]]:
    offsets = set()
    ref_pts = tuple(ref_shape)
    vis_pts = tuple(vis_shape)
    for rr, cc in ref_pts:
        for vr, vc in vis_pts:
            dr = rr - vr
            dc = cc - vc
            if all((pr + dr, pc + dc) in ref_shape for pr, pc in vis_pts):
                offsets.add((dr, dc))
    return sorted(offsets)


def _all_matches(
    refs: list[dict], vis_shape: frozenset[tuple[int, int]]
) -> list[tuple[int, tuple[int, int]]]:
    matches: list[tuple[int, tuple[int, int]]] = []
    for idx, ref in enumerate(refs):
        for offset in _candidate_offsets(ref["shape"], vis_shape):
            matches.append((idx, offset))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore each occluded gray sprite on the right by matching its visible fragment to a complete reference sprite on the left.

    Input structure:
    - Black background contains one full-height magenta separator column that splits the grid into left and right regions.
    - The left region holds 3 to 5 complete monochrome 4-connected sprites in distinct non-gray colors.
    - The right region holds the same number of gray 4-connected visible fragments; each comes from one matching left sprite after partial black rectangular occlusion.

    Rule:
    - Detect the separator and collect normalized reference masks plus their colors from the left region.
    - For each gray component on the right, normalize its visible mask and find the unique reference mask and unique internal offset for which the visible mask is a translated subset.
    - Reconstruct the full matched reference shape at the corresponding right-side location and recolor both visible and hidden pixels with the reference color.
    - Erase the entire left region and the separator so the output keeps only the restored right-side sprites on black.

    Key insight:
    - The right-side gray fragments are insufficient by themselves; only subset matching against the left reference dictionary reveals which hidden pixels must be restored and which color to paint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_c = _find_separator_col(g)
    refs = _extract_references(g, sep_c)
    visibles = _extract_visible_right(g, sep_c)
    assert len(visibles) == len(refs), "right side must contain one visible fragment per reference"

    out = np.zeros_like(g)
    used_ref_indices: set[int] = set()
    for vis in visibles:
        matches = _all_matches(refs, vis["shape"])
        assert len(matches) == 1, "each visible fragment must match exactly one reference with one offset"
        ref_idx, (dr, dc) = matches[0]
        assert ref_idx not in used_ref_indices, "reference matching must be one-to-one"
        used_ref_indices.add(ref_idx)

        ref = refs[ref_idx]
        full_r0 = vis["origin"][0] - dr
        full_c0 = vis["origin"][1] - dc
        for rr, cc in ref["shape"]:
            ar = full_r0 + rr
            ac = sep_c + 1 + full_c0 + cc
            assert 0 <= ar < g.shape[0] and 0 <= ac < g.shape[1], "restored sprite must stay in bounds"
            existing = int(out[ar, ac])
            assert existing in (BG, ref["color"]), "restored sprites must not overlap"
            out[ar, ac] = ref["color"]

    assert len(used_ref_indices) == len(refs), "every reference must be used exactly once"
    return out


def _paint_shape(grid: np.ndarray, shape: frozenset[tuple[int, int]], top: tuple[int, int], color: int) -> None:
    r0, c0 = top
    for dr, dc in shape:
        grid[r0 + dr, c0 + dc] = color


def _sample_positions(
    zone_h: int,
    zone_w: int,
    dims: list[tuple[int, int]],
    *,
    margin: int,
    padding: int,
) -> list[tuple[int, int]] | None:
    occupied = np.zeros((zone_h, zone_w), dtype=bool)
    order = sorted(range(len(dims)), key=lambda i: dims[i][0] * dims[i][1], reverse=True)
    positions: list[tuple[int, int] | None] = [None] * len(dims)
    for idx in order:
        h, w = dims[idx]
        candidates: list[tuple[int, int]] = []
        for r in range(margin, zone_h - h - margin + 1):
            for c in range(margin, zone_w - w - margin + 1):
                rr0 = max(0, r - padding)
                cc0 = max(0, c - padding)
                rr1 = min(zone_h, r + h + padding)
                cc1 = min(zone_w, c + w + padding)
                if not occupied[rr0:rr1, cc0:cc1].any():
                    candidates.append((r, c))
        if not candidates:
            return None
        pick = random.choice(candidates)
        positions[idx] = pick
        r, c = pick
        rr0 = max(0, r - padding)
        cc0 = max(0, c - padding)
        rr1 = min(zone_h, r + h + padding)
        cc1 = min(zone_w, c + w + padding)
        occupied[rr0:rr1, cc0:cc1] = True
    return [pos for pos in positions if pos is not None]


def _occlusion_candidates(shape: frozenset[tuple[int, int]], refs: list[dict], ref_idx: int) -> list[dict]:
    h, w = _shape_dims(shape)
    cells = set(shape)
    candidates: list[dict] = []
    for rect_h in OCCLUDER_SIZES:
        for rect_w in OCCLUDER_SIZES:
            for r0 in range(-rect_h + 1, h):
                for c0 in range(-rect_w + 1, w):
                    hidden = {
                        (r, c)
                        for r, c in cells
                        if r0 <= r < r0 + rect_h and c0 <= c < c0 + rect_w
                    }
                    if not hidden or len(hidden) == len(cells):
                        continue
                    visible = cells - hidden
                    if len(visible) < 2 or not _is_connected(visible):
                        continue
                    visible_norm, _ = _normalize_coords(visible)
                    matches = _all_matches(refs, visible_norm)
                    if len(matches) != 1 or matches[0][0] != ref_idx:
                        continue
                    candidates.append(
                        {
                            "rect": (r0, c0, rect_h, rect_w),
                            "visible_norm": visible_norm,
                            "hidden_count": len(hidden),
                            "visible_count": len(visible),
                        }
                    )
    return candidates


def _choose_shapes_and_occlusions() -> tuple[list[dict], list[dict]]:
    for _ in range(MAX_GEN_ATTEMPTS):
        n = random.randint(3, 5)
        chosen_shapes = random.sample(list(SHAPE_LIBRARY), n)
        refs = [{"shape": shape, "color": -1} for shape in chosen_shapes]
        full_areas = [len(shape) for shape in chosen_shapes]
        if len(set(full_areas)) == len(full_areas):
            continue

        options = [_occlusion_candidates(shape, refs, idx) for idx, shape in enumerate(chosen_shapes)]
        if any(not opts for opts in options):
            continue

        for _attempt in range(120):
            picks = [random.choice(opts) for opts in options]
            visible_areas = [pick["visible_count"] for pick in picks]
            if len(set(visible_areas)) == len(visible_areas):
                continue
            if max(pick["hidden_count"] for pick in picks) < 2:
                continue
            return refs, picks
    raise RuntimeError("failed to choose shape/occlusion bundle")


def _build_expected(grid_shape: tuple[int, int], sep_c: int, right_specs: list[dict]) -> np.ndarray:
    out = np.zeros(grid_shape, dtype=int)
    for spec in right_specs:
        _paint_shape(out, spec["shape"], (spec["top"][0], sep_c + 1 + spec["top"][1]), spec["color"])
    return out


def _validate_instance(inp: np.ndarray, expected: np.ndarray) -> None:
    solved = main(inp)
    assert np.array_equal(solved, expected), "solver must match generator oracle"

    sep_c = _find_separator_col(inp)
    refs = _extract_references(inp, sep_c)
    visibles = _extract_visible_right(inp, sep_c)
    assert len(refs) == len(visibles)

    ref_areas = Counter(len(ref["shape"]) for ref in refs)
    vis_areas = Counter(len(vis["shape"]) for vis in visibles)
    assert max(ref_areas.values()) >= 2, "at least two reference sprites should share an area"
    assert max(vis_areas.values()) >= 2, "at least two visible fragments should share an area"

    for vis in visibles:
        matches = _all_matches(refs, vis["shape"])
        assert len(matches) == 1, "visible fragment must have a unique semantic match"

    assert bool(np.any(inp[:, : sep_c + 1] != 0)), "input must contain a non-empty legend side"
    assert not bool(np.any(expected[:, : sep_c + 1] != 0)), "output must erase the legend side"
    assert not bool(np.any(expected[:, sep_c] != 0)), "separator must be erased"


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        refs, occlusions = _choose_shapes_and_occlusions()
        n = len(refs)

        colors = random.sample(list(REF_COLORS), n)
        perm = list(range(n))
        random.shuffle(perm)

        left_dims = [_shape_dims(refs[i]["shape"]) for i in range(n)]
        right_dims = [_shape_dims(refs[i]["shape"]) for i in perm]

        height = random.randint(16, 24)
        left_w = random.randint(8, 11)
        right_w = random.randint(10, 14)
        if left_w + 1 + right_w > 30:
            continue

        left_positions = _sample_positions(height, left_w, left_dims, margin=1, padding=1)
        right_positions = _sample_positions(height, right_w, right_dims, margin=2, padding=2)
        if left_positions is None or right_positions is None:
            continue

        left_order = sorted(range(n), key=lambda i: (left_positions[i][0], left_positions[i][1]))
        right_shape_order = [perm[i] for i in sorted(range(n), key=lambda i: (right_positions[i][0], right_positions[i][1]))]
        if n > 1 and right_shape_order == left_order:
            continue

        grid = np.zeros((height, left_w + 1 + right_w), dtype=int)
        sep_c = left_w
        grid[:, sep_c] = PINK

        left_specs: list[dict] = []
        for i, ref in enumerate(refs):
            color = colors[i]
            refs[i]["color"] = color
            left_specs.append({"shape": ref["shape"], "top": left_positions[i], "color": color})
            _paint_shape(grid, ref["shape"], left_positions[i], color)

        right_specs: list[dict] = []
        for right_slot, ref_idx in enumerate(perm):
            ref = refs[ref_idx]
            occ = occlusions[ref_idx]
            top = right_positions[right_slot]
            color = ref["color"]
            full_cells = {(top[0] + r, top[1] + c) for r, c in ref["shape"]}
            rect_r, rect_c, rect_h, rect_w = occ["rect"]
            visible_cells = {
                (r, c)
                for r, c in full_cells
                if not (
                    top[0] + rect_r <= r < top[0] + rect_r + rect_h
                    and top[1] + rect_c <= c < top[1] + rect_c + rect_w
                )
            }
            visible_shape, _ = _normalize_coords(visible_cells)
            matches = _all_matches(refs, visible_shape)
            if len(matches) != 1 or matches[0][0] != ref_idx:
                break
            for r, c in visible_cells:
                grid[r, sep_c + 1 + c] = GREY
            right_specs.append({"shape": ref["shape"], "top": top, "color": color})
        else:
            expected = _build_expected(grid.shape, sep_c, right_specs)
            try:
                _validate_instance(grid, expected)
            except AssertionError:
                continue
            return grid

    raise RuntimeError("failed to generate a valid ARC instance")
