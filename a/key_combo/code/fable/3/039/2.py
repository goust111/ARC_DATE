from __future__ import annotations

from collections import Counter
import random

import numpy as np


PALETTE = list(range(10))
GEN_ATTEMPTS = 300
CHAIN_SEPARATION = 1


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray, *, conn8: bool = False) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if conn8:
        steps.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
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
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.asarray(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_boundary(filled: np.ndarray) -> np.ndarray:
    h, w = filled.shape
    boundary = np.zeros((h, w), dtype=bool)
    coords = np.argwhere(filled)
    for r, c in coords:
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            boundary[r, c] = True
            continue
        if not (filled[r - 1, c] and filled[r + 1, c] and filled[r, c - 1] and filled[r, c + 1]):
            boundary[r, c] = True
    return boundary


def _frame_inside_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = _bbox(coords)
    local = np.zeros((r1 - r0 + 3, c1 - c0 + 3), dtype=bool)
    for r, c in coords:
        local[r - r0 + 1, c - c0 + 1] = True

    visited = np.zeros_like(local, dtype=bool)
    stack = [(0, 0)]
    visited[0, 0] = True
    while stack:
        cr, cc = stack.pop()
        for nr, nc in _neighbors4(cr, cc):
            if 0 <= nr < local.shape[0] and 0 <= nc < local.shape[1]:
                if not local[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))

    inside_local = (~visited) & (~local)
    inside = np.zeros((h, w), dtype=bool)
    inside[r0 : r1 + 1, c0 : c1 + 1] = inside_local[1:-1, 1:-1]
    return inside


def _frame_color(grid: np.ndarray, bg: int) -> int:
    counts = Counter(int(v) for v in grid.ravel() if int(v) != bg)
    assert counts, "expected non-background colors"
    return max(counts.items(), key=lambda item: item[1])[0]


def _ordered_code(grid: np.ndarray, bg: int, frame_color: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    coords = np.argwhere((grid != bg) & (grid != frame_color))
    assert len(coords) > 0, "expected a corner code"

    transformed_specs = (
        lambda r, c: (r, c),
        lambda r, c: (r, w - 1 - c),
        lambda r, c: (h - 1 - r, c),
        lambda r, c: (h - 1 - r, w - 1 - c),
    )
    best: tuple[tuple[int, int, int], list[tuple[int, int, int]]] | None = None
    for transform in transformed_specs:
        diag_coords: list[tuple[int, int, int]] = []
        valid = True
        for r, c in coords:
            rr, cc = transform(int(r), int(c))
            if rr != cc:
                valid = False
                break
            diag_coords.append((rr, int(r), int(c)))
        if not valid:
            continue
        diag_coords.sort()
        dists = [d for d, _, _ in diag_coords]
        if len(set(dists)) != len(dists):
            continue
        score = (min(dists), sum(dists), len(dists))
        if best is None or score < best[0]:
            best = (score, diag_coords)

    assert best is not None, "corner code must lie on one diagonal from a corner"
    return [(r, c) for _, r, c in best[1]]


def _structure_info(grid: np.ndarray) -> tuple[int, int, list[int], list[tuple[np.ndarray, int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frame_color = _frame_color(g, bg)
    code_order = _ordered_code(g, bg, frame_color)
    code_colors = [int(g[r, c]) for r, c in code_order]

    interiors: list[np.ndarray] = []
    interior_sizes: list[int] = []
    for coords in _components(g == frame_color, conn8=True):
        inside = _frame_inside_mask(g.shape, coords)
        if inside.any():
            interiors.append(inside)
            interior_sizes.append(int(inside.sum()))
    assert interiors, "expected at least one closed frame"

    regions: list[tuple[np.ndarray, int, int]] = []
    for coords in _components(g == bg):
        sample = tuple(map(int, coords[0]))
        enclosers = [i for i, inside in enumerate(interiors) if bool(inside[sample])]
        if not enclosers:
            continue
        root = max(enclosers, key=lambda idx: interior_sizes[idx])
        regions.append((coords, len(enclosers), int(root)))
    assert regions, "expected enclosed background regions"
    return bg, frame_color, code_colors, regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill every nested-chain cavity with the corner-diagonal code color indexed from the leaf region outward.

    Input structure:
    - The grid has one dominant background color, one shared frame color, and a small diagonal code of single pixels in one corner.
    - The frame color forms several disjoint closed loops, arranged as non-branching nested chains where each frame directly contains at most one child.
    - The code colors differ from both the background and the frame color.

    Rule:
    - For each enclosed background region, count how many frame layers contain it.
    - Within one nested chain, the innermost region uses the first code pixel color, the next outer region uses the second code pixel color, and so on moving outward.
    - Flood-fill every enclosed region with its assigned code color while keeping the frames, the code pixels, and the exterior background unchanged.

    Key insight:
    - The palette index is anchored at the leaf cavity of each chain rather than at the outermost frame, so the same outward depth can map to different colors when chain depths differ.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, _frame_color_value, code_colors, regions = _structure_info(g)
    max_depth_by_root: dict[int, int] = {}
    for _coords, depth, root in regions:
        max_depth_by_root[root] = max(max_depth_by_root.get(root, 0), int(depth))

    assert max(max_depth_by_root.values()) <= len(code_colors), "corner code is too short"
    out = g.copy()
    for coords, depth, root in regions:
        code_index = max_depth_by_root[root] - int(depth)
        color = code_colors[code_index]
        out[coords[:, 0], coords[:, 1]] = color

    assert np.any(out != g), "expected at least one filled region"
    assert np.array_equal(out[g != bg], g[g != bg]), "frames and code must stay unchanged"
    return out


def _make_irregular_frame(core_h: int, core_w: int, rng: random.Random) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    mt = 1
    mb = 1
    ml = 1
    mr = 1
    h = core_h + mt + mb
    w = core_w + ml + mr
    r0, c0 = mt, ml
    r1, c1 = mt + core_h - 1, ml + core_w - 1

    filled = np.zeros((h, w), dtype=bool)
    filled[r0 : r1 + 1, c0 : c1 + 1] = True

    sides = []
    if mt > 0:
        sides.append("top")
    if mb > 0:
        sides.append("bottom")
    if ml > 0:
        sides.append("left")
    if mr > 0:
        sides.append("right")
    rng.shuffle(sides)
    tab_count = 1 + int(len(sides) > 1 and rng.random() < 0.55)

    for side in sides[:tab_count]:
        if side in ("top", "bottom"):
            max_span = max(2, core_w - 1)
            span = rng.randint(2, max_span)
            start = rng.randint(c0, c1 - span + 1)
            if side == "top":
                filled[: r0 + 1, start : start + span] = True
            else:
                filled[r1:, start : start + span] = True
        else:
            max_span = max(2, core_h - 1)
            span = rng.randint(2, max_span)
            start = rng.randint(r0, r1 - span + 1)
            if side == "left":
                filled[start : start + span, : c0 + 1] = True
            else:
                filled[start : start + span, c1:] = True

    boundary = _extract_boundary(filled)
    return boundary, (r0, c0, r1, c1)


def _paint_patch(dst: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    view = dst[top : top + h, left : left + w]
    mask = patch != bg
    view[mask] = patch[mask]


def _make_leaf_chain(bg: int, frame_color: int, rng: random.Random) -> np.ndarray:
    cavity_h = rng.randint(1, 3)
    cavity_w = rng.randint(1, 4)
    boundary, _core_box = _make_irregular_frame(cavity_h + 2, cavity_w + 2, rng)
    grid = np.full(boundary.shape, bg, dtype=int)
    grid[boundary] = frame_color
    return grid


def _wrap_with_parent(child: np.ndarray, bg: int, frame_color: int, rng: random.Random) -> np.ndarray:
    child_h, child_w = child.shape
    top_gap = 1 + int(rng.random() < 0.25)
    bottom_gap = 1 + int(rng.random() < 0.25)
    left_gap = 1 + int(rng.random() < 0.25)
    right_gap = 1 + int(rng.random() < 0.25)

    core_h = child_h + top_gap + bottom_gap + 2
    core_w = child_w + left_gap + right_gap + 2
    boundary, (r0, c0, _r1, _c1) = _make_irregular_frame(core_h, core_w, rng)
    grid = np.full(boundary.shape, bg, dtype=int)
    grid[boundary] = frame_color
    child_top = r0 + 1 + top_gap
    child_left = c0 + 1 + left_gap
    _paint_patch(grid, child, child_top, child_left, bg)
    return grid


def _build_chain(depth: int, bg: int, frame_color: int, rng: random.Random) -> np.ndarray | None:
    max_dim_by_depth = {1: 9, 2: 14, 3: 18, 4: 22}
    for _ in range(50):
        chain = _make_leaf_chain(bg, frame_color, rng)
        for _level in range(depth - 1):
            chain = _wrap_with_parent(chain, bg, frame_color, rng)
        if max(chain.shape) <= max_dim_by_depth[depth]:
            return chain
    return None


def _sample_depths(chain_count: int, rng: random.Random) -> list[int]:
    if chain_count == 1:
        return [rng.choice([2, 3, 4])]
    if chain_count == 2:
        max_depth = rng.choice([2, 3, 3, 4])
        other = rng.choice([d for d in range(1, max_depth + 1) if d != max_depth])
        depths = [max_depth, other]
        rng.shuffle(depths)
        return depths
    patterns = ([1, 2, 3], [1, 1, 2], [1, 2, 2])
    depths = list(rng.choice(patterns))
    rng.shuffle(depths)
    return depths


def _base_canvas_dims(patches: list[np.ndarray], rng: random.Random) -> tuple[int, int]:
    hs = [patch.shape[0] for patch in patches]
    ws = [patch.shape[1] for patch in patches]
    row = (max(hs), sum(ws) + CHAIN_SEPARATION * (len(patches) - 1))
    col = (sum(hs) + CHAIN_SEPARATION * (len(patches) - 1), max(ws))
    options = [row, col]
    if len(patches) == 3:
        order = sorted(range(3), key=lambda i: ws[i], reverse=True)
        first = order[:2]
        second = order[2:]
        two_row = (
            max(hs[i] for i in first) + CHAIN_SEPARATION + max(hs[i] for i in second),
            max(sum(ws[i] for i in first) + CHAIN_SEPARATION, max(ws[i] for i in second)),
        )
        options.append(two_row)
        first = order[:1]
        second = order[1:]
        two_col = (
            max(sum(hs[i] for i in first) + CHAIN_SEPARATION, max(hs[i] for i in second)),
            max(ws[i] for i in first) + CHAIN_SEPARATION + max(ws[i] for i in second),
        )
        options.append(two_col)
    options.sort(key=lambda hw: (max(hw), hw[0] * hw[1], hw[0] + hw[1]))
    best = options[0]
    nearly_best = [opt for opt in options if max(opt) <= max(best) + 1]
    base_h, base_w = rng.choice(nearly_best)
    return int(base_h), int(base_w)


def _forbidden_mask(shape: tuple[int, int], corner: str, code_len: int) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    size = code_len
    for dr in range(size):
        for dc in range(size):
            if corner == "tl":
                r, c = dr, dc
            elif corner == "tr":
                r, c = dr, w - 1 - dc
            elif corner == "bl":
                r, c = h - 1 - dr, dc
            else:
                r, c = h - 1 - dr, w - 1 - dc
            if 0 <= r < h and 0 <= c < w:
                mask[r, c] = True
    return mask


def _place_chains(shape: tuple[int, int], patches: list[np.ndarray], corner: str, code_len: int, rng: random.Random) -> list[tuple[int, int]] | None:
    h, w = shape
    occupancy = _forbidden_mask(shape, corner, code_len)
    order = sorted(range(len(patches)), key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)
    positions: list[tuple[int, int] | None] = [None] * len(patches)

    for idx in order:
        ph, pw = patches[idx].shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, h - ph):
            for c0 in range(1, w - pw):
                rr0 = max(0, r0 - CHAIN_SEPARATION)
                rr1 = min(h, r0 + ph + CHAIN_SEPARATION)
                cc0 = max(0, c0 - CHAIN_SEPARATION)
                cc1 = min(w, c0 + pw + CHAIN_SEPARATION)
                if occupancy[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = rng.choice(candidates)
        positions[idx] = (r0, c0)
        rr0 = max(0, r0 - CHAIN_SEPARATION)
        rr1 = min(h, r0 + ph + CHAIN_SEPARATION)
        cc0 = max(0, c0 - CHAIN_SEPARATION)
        cc1 = min(w, c0 + pw + CHAIN_SEPARATION)
        occupancy[rr0:rr1, cc0:cc1] = True

    return [pos for pos in positions if pos is not None]


def _paint_code(grid: np.ndarray, corner: str, colors: list[int]) -> None:
    h, w = grid.shape
    for i, color in enumerate(colors):
        if corner == "tl":
            r, c = i, i
        elif corner == "tr":
            r, c = i, w - 1 - i
        elif corner == "bl":
            r, c = h - 1 - i, i
        else:
            r, c = h - 1 - i, w - 1 - i
        grid[r, c] = color


def _analyze_depths(grid: np.ndarray) -> list[int]:
    _bg, _frame_color_value, code_colors, regions = _structure_info(grid)
    max_depth_by_root: dict[int, int] = {}
    for _coords, depth, root in regions:
        max_depth_by_root[root] = max(max_depth_by_root.get(root, 0), int(depth))
    assert max(max_depth_by_root.values()) <= len(code_colors)
    return sorted(max_depth_by_root.values())


def _valid_generated(grid: np.ndarray, depths: list[int]) -> bool:
    bg = _get_bg(grid)
    counts = Counter(map(int, grid.ravel()))
    other_max = max((count for color, count in counts.items() if color != bg), default=-1)
    if counts[bg] <= other_max:
        return False

    try:
        inferred_depths = _analyze_depths(grid)
        out = main(grid)
    except Exception:
        return False

    if inferred_depths != sorted(depths):
        return False
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    chain_count = rng.choices([1, 2, 3], weights=[1, 4, 2], k=1)[0]
    depths = _sample_depths(chain_count, rng)

    bg = rng.choice(PALETTE)
    frame_color = rng.choice([c for c in PALETTE if c != bg])
    code_len = max(depths)
    code_colors = rng.sample([c for c in PALETTE if c not in {bg, frame_color}], code_len)

    patches: list[np.ndarray] = []
    for depth in depths:
        patch = _build_chain(depth, bg, frame_color, rng)
        if patch is None:
            return None
        patches.append(patch)

    base_h, base_w = _base_canvas_dims(patches, rng)
    h_lo = max(max(p.shape[0] for p in patches) + 2, base_h + 2)
    w_lo = max(max(p.shape[1] for p in patches) + 2, base_w + 2)
    if h_lo > 30 or w_lo > 30:
        return None

    h = rng.randint(h_lo, min(30, h_lo + 4))
    w = rng.randint(w_lo, min(30, w_lo + 4))
    corner = rng.choice(["tl", "tr", "bl", "br"])
    positions = _place_chains((h, w), patches, corner, code_len, rng)
    if positions is None:
        return None

    grid = np.full((h, w), bg, dtype=int)
    for patch, (r0, c0) in zip(patches, positions):
        _paint_patch(grid, patch, r0, c0, bg)
    _paint_code(grid, corner, code_colors)

    if not _valid_generated(grid, depths):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
