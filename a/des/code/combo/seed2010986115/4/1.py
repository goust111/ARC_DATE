from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
SPRITE_COLORS = [1, 2, 3, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_ROT_CW = {"up": 0, "right": 1, "down": 2, "left": 3}

GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 200
PLACE_ATTEMPTS = 80


def _rotate_cw(arr: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(arr, -(k % 4))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _component_from_seed(mask: np.ndarray, seed: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    sr, sc = seed
    assert 0 <= sr < h and 0 <= sc < w and mask[sr, sc]
    seen = {seed}
    stack = [seed]
    coords: list[tuple[int, int]] = []
    while stack:
        r, c = stack.pop()
        coords.append((r, c))
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return coords


def _touches_edge(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        out[r - r0, c - c0] = True
    return out


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols]


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _has_internal_void(mask: np.ndarray) -> bool:
    for comp in _components(~mask):
        if not _touches_edge(comp, mask.shape[0], mask.shape[1]):
            return True
    return False


def _matching_rotations(sprite_mask: np.ndarray, hole_mask: np.ndarray) -> list[int]:
    matches: list[int] = []
    for k in range(4):
        if np.array_equal(_rotate_cw(sprite_mask, k), hole_mask):
            matches.append(k)
    return matches


def _side_from_marker(marker: tuple[int, int], sprite_coords: set[tuple[int, int]]) -> str | None:
    r, c = marker
    if (r + 1, c) in sprite_coords:
        return "up"
    if (r - 1, c) in sprite_coords:
        return "down"
    if (r, c + 1) in sprite_coords:
        return "left"
    if (r, c - 1) in sprite_coords:
        return "right"
    return None


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    internal_black = [comp for comp in _components(g == BLACK) if not _touches_edge(comp, h, w)]
    assert len(internal_black) == 1, "expected exactly one enclosed black hole"
    hole_coords = internal_black[0]
    hole_mask = _normalize_mask(hole_coords)
    hr0, hc0, hr1, hc1 = _bbox(hole_coords)

    frame_adjacent: list[tuple[int, int]] = []
    frame_colors = set()
    for r, c in hole_coords:
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and g[nr, nc] != BLACK:
                frame_adjacent.append((nr, nc))
                frame_colors.add(int(g[nr, nc]))
    assert len(frame_colors) == 1, "hole should be surrounded by a single frame color"
    frame_color = next(iter(frame_colors))

    frame_coords = _component_from_seed(g == frame_color, frame_adjacent[0])
    frame_set = set(frame_coords)
    fr0, fc0, fr1, fc1 = _bbox(frame_coords)

    match_comp: list[tuple[int, int]] | None = None
    match_color: int | None = None
    fill_rot: int | None = None
    for color in np.unique(g):
        color = int(color)
        if color in (BLACK, YELLOW):
            continue
        for comp in _components(g == color):
            comp_set = set(comp)
            if comp_set.issubset(frame_set):
                continue
            rots = _matching_rotations(_normalize_mask(comp), hole_mask)
            if rots:
                assert match_comp is None, "expected a unique matching sprite"
                match_comp = comp
                match_color = color
                fill_rot = rots[0]

    assert match_comp is not None and match_color is not None and fill_rot is not None
    sprite_coords = set(match_comp)

    marker_side: str | None = None
    for marker in map(tuple, np.argwhere(g == YELLOW).tolist()):
        side = _side_from_marker(marker, sprite_coords)
        if side is not None:
            marker_side = side
            break
    assert marker_side is not None, "expected a yellow marker adjacent to the matching sprite"

    completed = g[fr0 : fr1 + 1, fc0 : fc1 + 1].copy()
    rotated_sprite = _rotate_cw(_normalize_mask(match_comp), fill_rot)
    hole_view = completed[hr0 - fr0 : hr1 - fr0 + 1, hc0 - fc0 : hc1 - fc0 + 1]
    hole_view[rotated_sprite] = match_color

    return _rotate_cw(completed, SIDE_TO_ROT_CW[marker_side]).astype(int)


def _random_connected_mask(rng: random.Random) -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        h = rng.randint(2, 5)
        w = rng.randint(2, 5)
        target = rng.randint(3, h * w - 1)

        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        frontier = set()
        while len(cells) < target:
            frontier.clear()
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(tuple(frontier)))

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _trim_mask(mask)

        if mask.sum() < 3 or mask.sum() == mask.size or min(mask.shape) < 2:
            continue
        if _has_internal_void(mask):
            continue
        if len({_mask_key(_rotate_cw(mask, k)) for k in range(4)}) != 4:
            continue
        return mask
    raise RuntimeError("failed to build an asymmetric connected mask")


def _random_distractor_mask(rng: random.Random, hole_mask: np.ndarray) -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        mask = _random_connected_mask(rng)
        if any(np.array_equal(_rotate_cw(mask, k), hole_mask) for k in range(4)):
            continue
        return mask
    raise RuntimeError("failed to build a distractor mask")


def _make_frame_patch(
    hole_mask: np.ndarray, top: int, left: int, bottom: int, right: int
) -> tuple[np.ndarray, tuple[int, int]]:
    hh, hw = hole_mask.shape
    frame = np.full((top + hh + bottom, left + hw + right), GREY, dtype=int)
    hole_view = frame[top : top + hh, left : left + hw]
    hole_view[hole_mask] = BLACK
    return frame, (top, left)


def _make_sprite_patch(
    mask: np.ndarray, color: int, rng: random.Random, side: str | None = None
) -> np.ndarray:
    h, w = mask.shape
    if side is None:
        patch = np.zeros((h, w), dtype=int)
        patch[mask] = color
        return patch

    if side == "up":
        patch = np.zeros((h + 1, w), dtype=int)
        patch[1:, :][mask] = color
        cols = [c for c in range(w) if mask[0, c]]
        patch[0, rng.choice(cols)] = YELLOW
        return patch
    if side == "down":
        patch = np.zeros((h + 1, w), dtype=int)
        patch[:h, :][mask] = color
        cols = [c for c in range(w) if mask[h - 1, c]]
        patch[h, rng.choice(cols)] = YELLOW
        return patch
    if side == "left":
        patch = np.zeros((h, w + 1), dtype=int)
        patch[:, 1:][mask] = color
        rows = [r for r in range(h) if mask[r, 0]]
        patch[rng.choice(rows), 0] = YELLOW
        return patch

    patch = np.zeros((h, w + 1), dtype=int)
    patch[:, :w][mask] = color
    rows = [r for r in range(h) if mask[r, w - 1]]
    patch[rng.choice(rows), w] = YELLOW
    return patch


def _can_place_bbox(occ: np.ndarray, r0: int, c0: int, ph: int, pw: int, pad: int = 1) -> bool:
    h, w = occ.shape
    r1 = max(0, r0 - pad)
    c1 = max(0, c0 - pad)
    r2 = min(h, r0 + ph + pad)
    c2 = min(w, c0 + pw + pad)
    return not occ[r1:r2, c1:c2].any()


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    ph, pw = patch.shape
    view = grid[r0 : r0 + ph, c0 : c0 + pw]
    mask = patch != BLACK
    view[mask] = patch[mask]


def _place_patches(
    patches: list[tuple[str, np.ndarray]], rng: random.Random
) -> np.ndarray | None:
    max_h = max(p.shape[0] for _name, p in patches)
    max_w = max(p.shape[1] for _name, p in patches)
    padded_area = sum((p.shape[0] + 2) * (p.shape[1] + 2) for _name, p in patches)
    min_side = max(max_h + 2, max_w + 2, int(np.ceil(np.sqrt(padded_area))) + 1)
    if min_side > 30:
        return None

    for _ in range(PLACE_ATTEMPTS):
        h = rng.randint(min_side, 30)
        w = rng.randint(min_side, 30)
        grid = np.zeros((h, w), dtype=int)
        occ = np.zeros((h, w), dtype=bool)
        ok = True
        for _name, patch in sorted(patches, key=lambda item: item[1].shape[0] * item[1].shape[1], reverse=True):
            ph, pw = patch.shape
            positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
            rng.shuffle(positions)
            placed = False
            for r0, c0 in positions:
                if not _can_place_bbox(occ, r0, c0, ph, pw, pad=1):
                    continue
                _paste_patch(grid, patch, r0, c0)
                occ[r0 : r0 + ph, c0 : c0 + pw] = True
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return grid
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    hole_mask = _random_connected_mask(rng)
    top = rng.randint(1, 3)
    left = rng.randint(1, 3)
    bottom = rng.randint(1, 3)
    right = rng.randint(1, 3)
    frame_patch, (hole_r0, hole_c0) = _make_frame_patch(hole_mask, top, left, bottom, right)

    n_sprites = rng.randint(3, 6)
    colors = rng.sample(SPRITE_COLORS, n_sprites)
    target_color = colors[0]
    distractor_colors = colors[1:]

    fill_rot = rng.randrange(4)
    target_mask = _rotate_cw(hole_mask, (4 - fill_rot) % 4)
    marker_side = rng.choice(["up", "right", "down", "left"])
    target_patch = _make_sprite_patch(target_mask, target_color, rng, side=marker_side)

    distractor_patches: list[np.ndarray] = []
    for color in distractor_colors:
        mask = _random_distractor_mask(rng, hole_mask)
        mask = _rotate_cw(mask, rng.randrange(4))
        distractor_patches.append(_make_sprite_patch(mask, color, rng))

    patches = [("frame", frame_patch), ("target", target_patch)]
    for idx, patch in enumerate(distractor_patches):
        patches.append((f"distractor_{idx}", patch))

    grid = _place_patches(patches, rng)
    if grid is None:
        return None

    expected = frame_patch.copy()
    hh, hw = hole_mask.shape
    filled = expected[hole_r0 : hole_r0 + hh, hole_c0 : hole_c0 + hw]
    filled[hole_mask] = target_color
    expected = _rotate_cw(expected, SIDE_TO_ROT_CW[marker_side])

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
