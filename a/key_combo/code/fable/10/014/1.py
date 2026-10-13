from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
ARROW_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


ARROW_PATTERNS = {
    frozenset({(0, 0), (2, 0), (1, 1)}): ((1, 1), (0, 1)),   # >
    frozenset({(0, 1), (2, 1), (1, 0)}): ((1, 0), (0, -1)),  # <
    frozenset({(0, 0), (0, 2), (1, 1)}): ((1, 1), (1, 0)),   # v
    frozenset({(1, 0), (1, 2), (0, 1)}): ((0, 1), (-1, 0)),  # ^
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    if connectivity == 4:
        nbrs = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        nbrs = (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),            (0, 1),
            (1, -1),  (1, 0),   (1, 1),
        )
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in nbrs:
                    nr = rr + dr
                    nc = cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _border_reachable_black(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if grid[r, c] == BLACK and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grid[r, c] == BLACK and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] == BLACK and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _parse_arrow(coords: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    assert coords.shape == (3, 2), "each arrow must be a 3-pixel sprite"
    r0, c0 = coords.min(axis=0)
    rel = frozenset((int(r - r0), int(c - c0)) for r, c in coords)
    assert rel in ARROW_PATTERNS, "unexpected arrow sprite shape"
    (tr, tc), direction = ARROW_PATTERNS[rel]
    return (int(r0 + tr), int(c0 + tc)), direction


def _find_arrows(grid: np.ndarray) -> list[dict]:
    colors = sorted(set(int(v) for v in np.unique(grid)) - {BLACK, GREY})
    arrows: list[dict] = []
    for color in colors:
        comps = _components(grid == color, connectivity=8)
        assert len(comps) == 1, "each arrow color must form exactly one sprite"
        tip, direction = _parse_arrow(comps[0])
        arrows.append({"color": int(color), "tip": tip, "dir": direction})
    assert arrows, "expected at least one arrow"
    return arrows


def _draw_rays(grid: np.ndarray, arrows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = np.array(grid, dtype=int, copy=True)
    ray_mask = np.zeros_like(out, dtype=bool)
    ray_color = np.zeros_like(out, dtype=int)
    for arrow in arrows:
        color = int(arrow["color"])
        (r, c) = arrow["tip"]
        dr, dc = arrow["dir"]
        rr = r + dr
        cc = c + dc
        steps = 0
        while _in_bounds(out.shape, rr, cc) and grid[rr, cc] != GREY:
            assert grid[rr, cc] == BLACK, "ray path must pass only through empty background"
            out[rr, cc] = color
            ray_mask[rr, cc] = True
            ray_color[rr, cc] = color
            steps += 1
            rr += dr
            cc += dc
        assert steps >= 1, "each arrow must emit a non-empty ray"
    return out, ray_mask, ray_color


def _solve_debug(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert np.count_nonzero(g == BLACK) > 0
    assert np.count_nonzero(g == GREY) > 0

    arrows = _find_arrows(g)
    with_rays, ray_mask, ray_color = _draw_rays(g, arrows)

    exterior_before = _border_reachable_black(g)
    exterior_after = _border_reachable_black(with_rays)
    preclosed_mask = (g == BLACK) & ~exterior_before
    newly_enclosed = (with_rays == BLACK) & exterior_before & ~exterior_after

    out = with_rays.copy()
    fill_colors: list[int] = []
    fill_areas: list[int] = []
    for comp in _components(newly_enclosed, connectivity=4):
        neighbor_colors = set()
        for r, c in comp:
            for nr, nc in _neighbors4(int(r), int(c)):
                if _in_bounds(g.shape, nr, nc) and ray_mask[nr, nc]:
                    neighbor_colors.add(int(ray_color[nr, nc]))
        assert len(neighbor_colors) == 1, "each new cavity must be sealed by one ray color"
        color = next(iter(neighbor_colors))
        out[comp[:, 0], comp[:, 1]] = color
        fill_colors.append(color)
        fill_areas.append(int(comp.shape[0]))

    debug = {
        "arrows": arrows,
        "with_rays": with_rays,
        "ray_mask": ray_mask,
        "ray_color": ray_color,
        "newly_enclosed": newly_enclosed,
        "fill_colors": fill_colors,
        "fill_areas": fill_areas,
        "preclosed_areas": [int(comp.shape[0]) for comp in _components(preclosed_mask, connectivity=4)],
    }
    return out, debug


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire each arrow to seal a wall gap, then fill only the background cavities that become newly enclosed.

    Input structure:
    - The grid has a black background, one-cell-thick grey wall segments, and several 3-pixel colored arrow sprites.
    - Each arrow is an axis-pointing V-shape: two same-color tail pixels and one same-color tip pixel, with the tip giving the firing direction.
    - Some grey wall shapes are already closed, while others have a straight missing side segment that one arrow can seal.

    Rule:
    - From each arrow tip, extend a same-color straight ray in the indicated horizontal or vertical direction until the cell just before a grey wall or the canvas edge.
    - Treat grey walls and the newly drawn ray cells as blocking walls, then find black regions that are no longer 4-connected to the canvas boundary.
    - Fill only those regions that were boundary-reachable before the rays but enclosed afterward, using the color of the unique ray that sealed them.
    - Keep the original arrows, the grey walls, and any region that was already closed before firing unchanged.

    Key insight:
    - The colored rays are not the final answer by themselves; they act as new wall segments, and only the cavities that become enclosed because of those new walls should be flood-filled.
    """
    out, _ = _solve_debug(grid)
    return out


def _make_closer_patch(color: int) -> np.ndarray:
    interior_h = random.randint(2, 4)
    interior_w = random.randint(3, 6)
    patch = np.zeros((interior_h + 3, interior_w + 3), dtype=int)
    patch[0, 0] = color
    patch[2, 0] = color
    patch[1, 1] = color
    patch[2:, 1] = GREY
    patch[1:, interior_w + 2] = GREY
    patch[interior_h + 2, 1:] = GREY
    return np.rot90(patch, random.randrange(4))


def _make_decoy_patch() -> np.ndarray:
    interior_h = random.randint(2, 4)
    interior_w = random.randint(2, 5)
    patch = np.zeros((interior_h + 2, interior_w + 2), dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    return patch


def _make_distractor_patch(color: int) -> np.ndarray:
    ray_len = random.randint(3, 6)
    patch = np.zeros((3, ray_len + 3), dtype=int)
    patch[0, 0] = color
    patch[2, 0] = color
    patch[1, 1] = color
    patch[1, ray_len + 2] = GREY
    if random.random() < 0.5:
        patch[2, ray_len + 2] = GREY
    return np.rot90(patch, random.randrange(4))


def _place_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    max_h = max(p.shape[0] for p in patches)
    max_w = max(p.shape[1] for p in patches)
    reserve_area = sum((p.shape[0] + 4) * (p.shape[1] + 4) for p in patches)
    min_side = max(max_h + 4, max_w + 4, int(np.ceil(np.sqrt(reserve_area))) - 1)
    if min_side > 30:
        return None

    h = random.randint(max(18, min_side), 30)
    w = random.randint(max(18, min_side), 30)
    canvas = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    for patch in sorted(patches, key=lambda p: p.shape[0] * p.shape[1], reverse=True):
        ph, pw = patch.shape
        placed = False
        for _ in range(300):
            r0 = random.randint(0, h - ph)
            c0 = random.randint(0, w - pw)
            r1 = max(0, r0 - 2)
            c1 = max(0, c0 - 2)
            r2 = min(h, r0 + ph + 2)
            c2 = min(w, c0 + pw + 2)
            if reserved[r1:r2, c1:c2].any():
                continue
            mask = patch != BLACK
            canvas[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
            reserved[r1:r2, c1:c2] = True
            placed = True
            break
        if not placed:
            return None
    return canvas


def _gen_candidate() -> np.ndarray | None:
    n_closers = random.randint(1, 3)
    n_decoys = random.randint(1, 2)
    n_distractors = 1 if random.random() < 0.6 else 0

    colors = random.sample(ARROW_COLORS, n_closers + n_distractors)
    closer_colors = colors[:n_closers]
    distractor_colors = colors[n_closers:]

    patches: list[np.ndarray] = []
    for color in closer_colors:
        patches.append(_make_closer_patch(color))
    for _ in range(n_decoys):
        patches.append(_make_decoy_patch())
    for color in distractor_colors:
        patches.append(_make_distractor_patch(color))
    random.shuffle(patches)

    grid = _place_patches(patches)
    if grid is None:
        return None

    out, debug = _solve_debug(grid)
    fill_colors = sorted(debug["fill_colors"])
    if fill_colors != sorted(closer_colors):
        return None
    if any(area < 4 for area in debug["fill_areas"]):
        return None
    if len(debug["preclosed_areas"]) < n_decoys:
        return None
    if any(area < 4 for area in debug["preclosed_areas"]):
        return None
    for color in closer_colors + distractor_colors:
        if np.count_nonzero(debug["ray_color"] == color) < 2:
            return None
    for color in distractor_colors:
        if color in debug["fill_colors"]:
            return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-wall ARC instance")
