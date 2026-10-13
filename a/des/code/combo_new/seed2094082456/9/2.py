from __future__ import annotations

from collections import Counter

import numpy as np


_RNG = np.random.default_rng()
_NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_BG = 0
_SEP = 5
_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
_MAX_GEN_ATTEMPTS = 400


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit each scattered fragment into the unique template hole with the same shape after rotation.

    Input structure:
    - A dominant background fills the grid, and one solid non-background separator column splits it into left and right panels.
    - The left panel contains several monochrome template objects, each with exactly one internal background hole.
    - The right panel contains the same number of monochrome 4-connected fragments, each in a distinct color.

    Rule:
    - Detect the separator column and treat the columns on its left and right as separate panels.
    - For each right-panel fragment, test the four rotations 0°, 90°, 180°, and 270°.
    - Match each rotated fragment to the unique left-panel hole whose binary shape matches exactly.
    - Produce the output on the original canvas: clear the fragment from the right panel and fill the matched hole with the fragment's color.

    Key insight:
    - The mapping is determined by exact geometric fit under rotation, not by color, order, or area alone.
    """
    return _solve(np.asarray(grid, dtype=int))


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = _get_bg(g)
    sep_c = _find_separator_col(g, bg)

    holes = _extract_template_holes(g[:, :sep_c], bg, col_offset=0)
    frags = _extract_fragments(g[:, sep_c + 1 :], bg, col_offset=sep_c + 1)

    assert holes, "expected at least one left-panel template hole"
    assert len(holes) == len(frags), "fragment and hole counts must match"

    out = g.copy()
    for frag in frags:
        out[frag["coords"][:, 0], frag["coords"][:, 1]] = bg

    used_holes: set[int] = set()
    for frag in frags:
        matches: list[tuple[int, int]] = []
        for hole_idx, hole in enumerate(holes):
            for rot in range(4):
                rotated = np.rot90(frag["mask"], rot)
                if rotated.shape == hole["mask"].shape and np.array_equal(rotated, hole["mask"]):
                    matches.append((hole_idx, rot))
        assert len(matches) == 1, "each fragment must match exactly one hole and one rotation"
        hole_idx, _rot = matches[0]
        assert hole_idx not in used_holes, "each hole must be used once"
        used_holes.add(hole_idx)

        hole = holes[hole_idx]
        r0, c0 = hole["top"]
        hh, ww = hole["mask"].shape
        patch = out[r0 : r0 + hh, c0 : c0 + ww]
        patch[hole["mask"]] = frag["color"]

    assert len(used_holes) == len(holes)
    return out


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_separator_col(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    candidates = []
    for c in range(1, w - 1):
        col = grid[:, c]
        if col[0] == bg:
            continue
        if not np.all(col == col[0]):
            continue
        if not np.any(grid[:, :c] != bg):
            continue
        if not np.any(grid[:, c + 1 :] != bg):
            continue
        candidates.append(c)
    assert len(candidates) == 1, "expected exactly one full-height separator column"
    return int(candidates[0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _coords_to_mask(coords: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return (r0, c0, r1, c1), mask


def _extract_template_holes(panel: np.ndarray, bg: int, col_offset: int) -> list[dict]:
    holes: list[dict] = []
    for comp in _components(panel != bg):
        colors = np.unique(panel[comp[:, 0], comp[:, 1]])
        colors = colors[colors != bg]
        assert len(colors) == 1, "each template must be monochrome"

        (r0, c0, r1, c1), _mask = _coords_to_mask(comp)
        patch = panel[r0 : r1 + 1, c0 : c1 + 1]
        inner = []
        for bg_comp in _components(patch == bg):
            if np.any(bg_comp[:, 0] == 0) or np.any(bg_comp[:, 0] == patch.shape[0] - 1):
                continue
            if np.any(bg_comp[:, 1] == 0) or np.any(bg_comp[:, 1] == patch.shape[1] - 1):
                continue
            (hr0, hc0, hr1, hc1), hmask = _coords_to_mask(bg_comp)
            inner.append(
                {
                    "top": (r0 + hr0, col_offset + c0 + hc0),
                    "mask": hmask,
                }
            )
        assert len(inner) == 1, "each template must contain exactly one internal hole"
        holes.append(inner[0])
    return holes


def _extract_fragments(panel: np.ndarray, bg: int, col_offset: int) -> list[dict]:
    frags: list[dict] = []
    for comp in _components(panel != bg):
        colors = np.unique(panel[comp[:, 0], comp[:, 1]])
        colors = colors[colors != bg]
        assert len(colors) == 1, "each fragment must be monochrome"
        (r0, c0, _r1, _c1), mask = _coords_to_mask(comp)
        global_coords = comp.copy()
        global_coords[:, 1] += int(col_offset)
        frags.append(
            {
                "color": int(colors[0]),
                "coords": global_coords,
                "mask": mask,
                "top": (r0, col_offset + c0),
            }
        )
    return frags


def _rotation_keys(mask: np.ndarray) -> list[tuple[tuple[int, int], bytes]]:
    return [(tuple(np.rot90(mask, k).shape), np.rot90(mask, k).tobytes()) for k in range(4)]


def _canonical_rot_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(_rotation_keys(mask))


def _has_distinct_rotations(mask: np.ndarray) -> bool:
    keys = _rotation_keys(mask)
    return len(set(keys)) == 4


def _random_connected_shape() -> np.ndarray | None:
    bbox_h = int(_RNG.integers(2, 5))
    bbox_w = int(_RNG.integers(2, 5))
    max_area = min(bbox_h * bbox_w - 1, 6)
    if max_area < 3:
        return None
    target_area = int(_RNG.integers(3, max_area + 1))

    start = (int(_RNG.integers(0, bbox_h)), int(_RNG.integers(0, bbox_w)))
    cells = {start}
    frontier = set()

    def add_frontier(cell: tuple[int, int]) -> None:
        r, c = cell
        for dr, dc in _NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < bbox_h and 0 <= nc < bbox_w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    add_frontier(start)
    while len(cells) < target_area and frontier:
        choices = tuple(frontier)
        nxt = choices[int(_RNG.integers(len(choices)))]
        frontier.remove(nxt)
        cells.add(nxt)
        add_frontier(nxt)

    if len(cells) != target_area:
        return None

    coords = np.array(sorted(cells), dtype=int)
    (_r0, _c0, _r1, _c1), mask = _coords_to_mask(coords)
    if not _has_distinct_rotations(mask):
        return None
    return mask


def _sample_hole_shapes(count: int) -> list[np.ndarray] | None:
    want_same_area = count >= 2 and bool(_RNG.integers(0, 2))
    shapes: list[np.ndarray] = []
    canonical: set[tuple[tuple[int, int], bytes]] = set()

    for _ in range(200):
        if len(shapes) == count:
            break
        mask = _random_connected_shape()
        if mask is None:
            continue
        key = _canonical_rot_key(mask)
        if key in canonical:
            continue
        shapes.append(mask)
        canonical.add(key)

    if len(shapes) != count:
        return None

    if want_same_area:
        areas = [int(mask.sum()) for mask in shapes]
        if len(set(areas)) == len(areas):
            return None
    return shapes


def _make_template_patch(hole_mask: np.ndarray, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    hole_h, hole_w = hole_mask.shape
    top = int(_RNG.integers(1, 3))
    bottom = int(_RNG.integers(1, 3))
    left = int(_RNG.integers(1, 3))
    right = int(_RNG.integers(1, 3))
    patch = np.full((hole_h + top + bottom, hole_w + left + right), color, dtype=int)
    patch[top : top + hole_h, left : left + hole_w][hole_mask] = _BG
    return patch, (top, left)


def _make_fragment_patch(hole_mask: np.ndarray, color: int, rotation: int) -> np.ndarray:
    frag_mask = np.rot90(hole_mask, rotation)
    patch = np.full(frag_mask.shape, _BG, dtype=int)
    patch[frag_mask] = color
    return patch


def _placement_candidates(panel_h: int, panel_w: int, patch_h: int, patch_w: int) -> list[tuple[int, int]]:
    inner = [
        (r, c)
        for r in range(1, panel_h - patch_h)
        for c in range(1, panel_w - patch_w)
    ]
    if inner:
        return inner
    return [
        (r, c)
        for r in range(0, panel_h - patch_h + 1)
        for c in range(0, panel_w - patch_w + 1)
    ]


def _place_patch(panel: np.ndarray, occupied: np.ndarray, patch: np.ndarray, pad: int = 1) -> tuple[int, int] | None:
    panel_h, panel_w = panel.shape
    patch_h, patch_w = patch.shape
    candidates = _placement_candidates(panel_h, panel_w, patch_h, patch_w)
    _RNG.shuffle(candidates)

    for r, c in candidates:
        rr0 = max(0, r - pad)
        cc0 = max(0, c - pad)
        rr1 = min(panel_h, r + patch_h + pad)
        cc1 = min(panel_w, c + patch_w + pad)
        if occupied[rr0:rr1, cc0:cc1].any():
            continue
        sub = panel[r : r + patch_h, c : c + patch_w]
        sub[patch != _BG] = patch[patch != _BG]
        occupied[rr0:rr1, cc0:cc1] = True
        return int(r), int(c)
    return None


def _gen_attempt() -> np.ndarray | None:
    n = int(_RNG.integers(2, 5))
    shapes = _sample_hole_shapes(n)
    if shapes is None:
        return None

    colors = list(_RNG.permutation(_COLOR_POOL))
    template_colors = colors[:n]
    fragment_colors = colors[n : 2 * n]

    rotations = [int(_RNG.integers(0, 4)) for _ in range(n)]
    if all(rot == 0 for rot in rotations):
        rotations[int(_RNG.integers(0, n))] = int(_RNG.integers(1, 4))

    templates = []
    fragments = []
    for hole_mask, tcol, fcol, rot in zip(shapes, template_colors, fragment_colors, rotations):
        template_patch, hole_offset = _make_template_patch(hole_mask, int(tcol))
        fragment_patch = _make_fragment_patch(hole_mask, int(fcol), int(rot))
        templates.append({"patch": template_patch, "hole_mask": hole_mask, "hole_offset": hole_offset})
        fragments.append({"patch": fragment_patch, "color": int(fcol)})

    temp_heights = sorted((item["patch"].shape[0] for item in templates), reverse=True)
    temp_widths = sorted((item["patch"].shape[1] for item in templates), reverse=True)
    frag_heights = sorted((item["patch"].shape[0] for item in fragments), reverse=True)
    frag_widths = sorted((item["patch"].shape[1] for item in fragments), reverse=True)

    h = max(
        14,
        (temp_heights[0] if temp_heights else 0) + (temp_heights[1] if len(temp_heights) > 1 else 0) + 4,
        (frag_heights[0] if frag_heights else 0) + (frag_heights[1] if len(frag_heights) > 1 else 0) + 5,
    )
    h = min(24, h + int(_RNG.integers(0, 3)))

    left_w = max(
        10,
        (temp_widths[0] if temp_widths else 0) + (temp_widths[1] if len(temp_widths) > 1 else 0) + 3,
    )
    left_w = min(18, left_w + int(_RNG.integers(0, 2)))

    right_w = max(
        8,
        (frag_widths[0] if frag_widths else 0) + (frag_widths[1] if len(frag_widths) > 1 else 0) + 4,
    )
    right_w = min(11, right_w + int(_RNG.integers(0, 2)))

    if h > 30 or left_w + 1 + right_w > 30:
        return None

    left_panel = np.full((h, left_w), _BG, dtype=int)
    right_panel = np.full((h, right_w), _BG, dtype=int)
    left_occ = np.zeros((h, left_w), dtype=bool)
    right_occ = np.zeros((h, right_w), dtype=bool)

    left_order = sorted(range(n), key=lambda i: templates[i]["patch"].shape[0] * templates[i]["patch"].shape[1], reverse=True)
    right_order = sorted(range(n), key=lambda i: fragments[i]["patch"].shape[0] * fragments[i]["patch"].shape[1], reverse=True)

    for idx in left_order:
        placed = _place_patch(left_panel, left_occ, templates[idx]["patch"], pad=1)
        if placed is None:
            return None
        templates[idx]["panel_top_left"] = placed

    for idx in right_order:
        placed = _place_patch(right_panel, right_occ, fragments[idx]["patch"], pad=1)
        if placed is None:
            return None
        fragments[idx]["panel_top_left"] = placed

    grid = np.full((h, left_w + 1 + right_w), _BG, dtype=int)
    grid[:, left_w] = _SEP
    grid[:, :left_w] = left_panel
    grid[:, left_w + 1 :] = right_panel

    if len([c for c in range(1, grid.shape[1] - 1) if np.all(grid[:, c] == grid[0, c]) and grid[0, c] != _BG]) != 1:
        return None

    try:
        _ = _solve(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
