import random
from collections import deque

import numpy as np


CONFLICT_COLOR = 9
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
CORNER_ORDER = ["tl", "tr", "br", "bl"]
MAX_GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _corner_coords(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    return [(0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, 0)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIRS8:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _tight_crop(arr: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(arr != bg)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _find_instruction_corner(patch: np.ndarray, bg: int) -> int:
    values, counts = np.unique(patch, return_counts=True)
    freq = {int(v): int(n) for v, n in zip(values, counts)}
    candidates = []
    for idx, (r, c) in enumerate(_corner_coords(patch.shape)):
        color = int(patch[r, c])
        if color != bg and freq.get(color, 0) == 1:
            candidates.append(idx)
    assert len(candidates) == 1
    return int(candidates[0])


def _overlay(patches: list[np.ndarray], bg: int) -> np.ndarray:
    assert patches
    shape = patches[0].shape
    assert all(p.shape == shape for p in patches)
    out = np.full(shape, bg, dtype=int)
    occupied = np.zeros(shape, dtype=int)
    for patch in patches:
        mask = patch != bg
        fresh = mask & (occupied == 0)
        clash = mask & (occupied > 0)
        out[fresh] = patch[fresh]
        out[clash] = CONFLICT_COLOR
        occupied[mask] += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify each sprite's unique corner instruction pixel, rotate the sprite accordingly,
    then overlay all rotated sprites on a shared canvas; conflicts become color 9.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    comps = _components(g != bg)
    assert comps

    rotated: list[np.ndarray] = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        rot = _find_instruction_corner(patch, bg)
        mr, mc = _corner_coords(patch.shape)[rot]
        patch[mr, mc] = bg
        restored = np.rot90(patch, k=-rot)
        rotated.append(_tight_crop(restored, bg))

    return _overlay(rotated, bg)


def _path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int], rng: random.Random) -> None:
    r, c = start
    er, ec = end
    mask[r, c] = True
    while (r, c) != (er, ec):
        moves = []
        if r < er:
            moves.append((1, 0))
        elif r > er:
            moves.append((-1, 0))
        if c < ec:
            moves.append((0, 1))
        elif c > ec:
            moves.append((0, -1))
        dr, dc = rng.choice(moves)
        nr, nc = r + dr, c + dc
        if (nr, nc) in ((0, 0), (0, mask.shape[1] - 1), (mask.shape[0] - 1, mask.shape[1] - 1), (mask.shape[0] - 1, 0)):
            alt = [step for step in moves if step != (dr, dc)]
            if alt:
                dr, dc = rng.choice(alt)
                nr, nc = r + dr, c + dc
        r, c = nr, nc
        mask[r, c] = True


def _valid_body(mask: np.ndarray) -> bool:
    s = mask.shape[0]
    if mask.shape[0] != mask.shape[1]:
        return False
    if any(mask[r, c] for r, c in _corner_coords(mask.shape)):
        return False
    if not np.any(mask[0, 1 : s - 1]):
        return False
    if not np.any(mask[s - 1, 1 : s - 1]):
        return False
    if not np.any(mask[1 : s - 1, 0]):
        return False
    if not np.any(mask[1 : s - 1, s - 1]):
        return False

    neighborhoods = [
        [(0, 1), (1, 0), (1, 1)],
        [(0, s - 2), (1, s - 1), (1, s - 2)],
        [(s - 2, s - 1), (s - 1, s - 2), (s - 2, s - 2)],
        [(s - 2, 0), (s - 1, 1), (s - 2, 1)],
    ]
    if not all(any(mask[r, c] for r, c in pts) for pts in neighborhoods):
        return False

    coords = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    if not coords:
        return False
    queue = deque([next(iter(coords))])
    seen = {queue[0]}
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if (nr, nc) in coords and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(coords)


def _generate_body_mask(size: int, rng: random.Random, target_area: int) -> np.ndarray:
    anchors = [
        (0, rng.randint(1, size - 2)),
        (rng.randint(1, size - 2), size - 1),
        (size - 1, rng.randint(1, size - 2)),
        (rng.randint(1, size - 2), 0),
    ]
    corner_neighbors = [
        rng.choice([(0, 1), (1, 0), (1, 1)]),
        rng.choice([(0, size - 2), (1, size - 1), (1, size - 2)]),
        rng.choice([(size - 2, size - 1), (size - 1, size - 2), (size - 2, size - 2)]),
        rng.choice([(size - 2, 0), (size - 1, 1), (size - 2, 1)]),
    ]

    mask = np.zeros((size, size), dtype=bool)
    required = anchors + corner_neighbors
    rng.shuffle(required)
    current = required[0]
    mask[current] = True
    for nxt in required[1:]:
        _path(mask, current, nxt, rng)
        current = nxt

    target_area = max(target_area, int(mask.sum()))
    while int(mask.sum()) < target_area:
        frontier = []
        for r, c in np.argwhere(mask):
            for dr, dc in DIRS4:
                nr, nc = int(r) + dr, int(c) + dc
                if not (0 <= nr < size and 0 <= nc < size):
                    continue
                if mask[nr, nc]:
                    continue
                if (nr, nc) in _corner_coords(mask.shape):
                    continue
                frontier.append((nr, nc))
        if not frontier:
            break
        mask[rng.choice(frontier)] = True

    return mask


def _colorize(mask: np.ndarray, bg: int, rng: random.Random) -> np.ndarray:
    occupied = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    n_cells = len(occupied)
    n_colors = 2 if n_cells < 10 else rng.randint(2, 4)
    palette = [c for c in range(10) if c not in {bg, CONFLICT_COLOR}]
    colors = rng.sample(palette, n_colors)
    seeds = rng.sample(occupied, n_colors)

    patch = np.full(mask.shape, bg, dtype=int)
    groups = [[] for _ in range(n_colors)]
    for r, c in occupied:
        best = min(
            range(n_colors),
            key=lambda idx: (
                abs(r - seeds[idx][0]) + abs(c - seeds[idx][1]),
                idx,
            ),
        )
        groups[best].append((r, c))

    for idx, cells in enumerate(groups):
        if not cells:
            cell = occupied[idx % n_cells]
            groups[idx].append(cell)
            donor = next(j for j, pts in enumerate(groups) if len(pts) > 1 and j != idx)
            groups[donor].remove(cell)

    for idx, cells in enumerate(groups):
        for r, c in cells:
            patch[r, c] = int(colors[idx])

    return patch


def _make_sprite_body(size: int, bg: int, rng: random.Random, dense: bool) -> np.ndarray:
    min_area = max(8, size + 2)
    max_area = size * size - 4
    if dense:
        lo = max(min_area, (size * size) // 2)
        hi = max(lo, int(0.72 * size * size))
    else:
        lo = min_area
        hi = max(lo, int(0.5 * size * size))

    for _ in range(120):
        target_area = rng.randint(lo, hi)
        mask = _generate_body_mask(size, rng, target_area)
        if _valid_body(mask):
            return _colorize(mask, bg, rng)
    raise RuntimeError("failed to generate valid sprite body")


def _encode_sprite(body: np.ndarray, rot: int, bg: int, rng: random.Random) -> np.ndarray:
    patch = np.rot90(body, k=rot).copy()
    marker_color_choices = [c for c in range(10) if c not in {bg, CONFLICT_COLOR} and c not in set(map(int, np.unique(patch)))]
    assert marker_color_choices
    marker_color = rng.choice(marker_color_choices)
    mr, mc = _corner_coords(patch.shape)[rot]
    patch[mr, mc] = int(marker_color)
    return patch


def _sample_rotations(n: int, rng: random.Random) -> list[int]:
    if n == 1:
        return [rng.randint(0, 3)]
    rots = list(range(4))
    rng.shuffle(rots)
    return rots[:n]


def _compositions(total: int, parts: int, rng: random.Random) -> list[int]:
    assert total >= parts
    remaining = total - parts
    cuts = sorted(rng.sample(range(remaining + parts - 1), parts - 1))
    out = []
    prev = -1
    for cut in cuts + [remaining + parts - 1]:
        out.append(cut - prev)
        prev = cut
    return out


def _place_sprites(patches: list[np.ndarray], bg: int, rng: random.Random) -> np.ndarray:
    n = len(patches)
    size = patches[0].shape[0]
    for _ in range(120):
        slot_rows = rng.randint(1, n)
        slot_cols = (n + slot_rows - 1) // slot_rows
        min_h = slot_rows * size + (slot_rows + 1)
        min_w = slot_cols * size + (slot_cols + 1)
        if min_h > 30 or min_w > 30:
            continue
        h = rng.randint(min_h, min(30, min_h + 8))
        w = rng.randint(min_w, min(30, min_w + 8))
        row_gaps = _compositions(h - slot_rows * size, slot_rows + 1, rng)
        col_gaps = _compositions(w - slot_cols * size, slot_cols + 1, rng)

        row_starts = []
        cur = row_gaps[0]
        for idx in range(slot_rows):
            row_starts.append(cur)
            cur += size + row_gaps[idx + 1]

        col_starts = []
        cur = col_gaps[0]
        for idx in range(slot_cols):
            col_starts.append(cur)
            cur += size + col_gaps[idx + 1]

        slots = [(r, c) for r in row_starts for c in col_starts]
        rng.shuffle(slots)
        chosen = slots[:n]
        grid = np.full((h, w), bg, dtype=int)
        for patch, (r0, c0) in zip(patches, chosen):
            grid[r0 : r0 + size, c0 : c0 + size] = patch
        return grid
    raise RuntimeError("failed to place sprite patches")


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    bg = rng.randint(0, 8)
    multi_sprite = rng.random() < 0.5
    n_sprites = 1 if not multi_sprite else rng.randint(2, 4)
    size = rng.randint(5, 8)
    dense = multi_sprite and rng.random() < 0.5
    rotations = _sample_rotations(n_sprites, rng)

    bodies = [_make_sprite_body(size, bg, rng, dense=dense) for _ in range(n_sprites)]
    expected = _overlay(bodies, bg)
    if n_sprites > 1 and not np.any(expected == CONFLICT_COLOR):
        raise RuntimeError("multi-sprite sample without overlap conflict")

    patches = [_encode_sprite(body, rot, bg, rng) for body, rot in zip(bodies, rotations)]
    inp = _place_sprites(patches, bg, rng)
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = random.Random()
        try:
            inp, expected = _build_case(rng)
        except (AssertionError, RuntimeError, ValueError):
            continue
        try:
            solved = main(inp)
        except AssertionError:
            continue
        if np.array_equal(solved, expected):
            return inp
    raise RuntimeError("failed to generate valid input")
