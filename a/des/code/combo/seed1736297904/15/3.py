import numpy as np
from collections import Counter


GREY = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
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
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _cropped_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _rotation_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        key = (tuple(map(int, rot.shape)), rot.tobytes())
        if key not in seen:
            seen.add(key)
            variants.append(rot)
    return variants


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min((tuple(map(int, rot.shape)), rot.tobytes()) for rot in _rotation_variants(mask))


def _has_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []
    for r in range(h):
        for c in (0, w - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return bool(np.any(bg & ~seen))


def _find_separator_row(grid: np.ndarray, bg: int) -> int:
    candidates: list[int] = []
    h, _ = grid.shape
    for r in range(1, h - 1):
        row = grid[r]
        if np.all(row == row[0]) and int(row[0]) != int(bg):
            if np.any(grid[:r] != bg) and np.any(grid[r + 1 :] != bg):
                candidates.append(r)
    grey_rows = [r for r in candidates if int(grid[r, 0]) == GREY]
    if grey_rows:
        return int(grey_rows[0])
    assert candidates, "expected a separator row"
    return int(candidates[0])


def _largest_enclosed_hole(grid: np.ndarray, bg: int, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r0, c0, r1, c1 = _bbox(coords)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    comp = np.zeros(sub.shape, dtype=bool)
    comp[coords[:, 0] - r0, coords[:, 1] - c0] = True
    bg_mask = sub == int(bg)
    seen = np.zeros(sub.shape, dtype=bool)
    stack: list[tuple[int, int]] = []
    h, w = sub.shape
    for r in range(h):
        for c in (0, w - 1):
            if bg_mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg_mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bg_mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    enclosed = bg_mask & ~seen
    holes = _components(enclosed)
    assert holes, "expected each lock to contain an enclosed hole"
    hole = max(holes, key=lambda arr: int(arr.shape[0]))
    hole_abs = hole + np.array([r0, c0], dtype=int)
    return hole_abs, _cropped_mask(hole)


def _assign_pairs(keys: list[dict], locks: list[dict]) -> dict[int, int]:
    compat: dict[int, list[int]] = {}
    for li, lock in enumerate(locks):
        matches: list[int] = []
        for ki, key in enumerate(keys):
            if any(np.array_equal(rot, lock["hole_mask"]) for rot in _rotation_variants(key["mask"])):
                matches.append(ki)
        compat[li] = matches
    order = sorted(range(len(locks)), key=lambda idx: len(compat[idx]))
    used: set[int] = set()
    out: dict[int, int] = {}

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        li = order[pos]
        for ki in compat[li]:
            if ki in used:
                continue
            used.add(ki)
            out[li] = ki
            if dfs(pos + 1):
                return True
            used.remove(ki)
            del out[li]
        return False

    assert dfs(0), "keys and locks must admit a unique full matching"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase top-panel keys and fill each bottom-panel lock hole with its border color.
    Matching is by exact key shape under 90-degree rotations only.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    sep = _find_separator_row(g, bg)

    top_mask = g[:sep] != int(bg)
    key_comps = _components(top_mask)
    keys = []
    for comp in key_comps:
        abs_coords = comp.copy()
        key_mask = _cropped_mask(abs_coords)
        keys.append({"coords": abs_coords, "mask": key_mask})

    bottom_mask = g[sep + 1 :] != int(bg)
    lock_comps = _components(bottom_mask)
    locks = []
    for comp in lock_comps:
        abs_coords = comp + np.array([sep + 1, 0], dtype=int)
        color = int(Counter(int(g[r, c]) for r, c in abs_coords).most_common(1)[0][0])
        hole_coords, hole_mask = _largest_enclosed_hole(g, bg, abs_coords)
        locks.append({"coords": abs_coords, "color": color, "hole_coords": hole_coords, "hole_mask": hole_mask})

    assert len(keys) == len(locks) and keys, "expected a one-to-one key/lock pairing"
    _assign_pairs(keys, locks)

    out = g.copy()
    for key in keys:
        out[key["coords"][:, 0], key["coords"][:, 1]] = int(bg)
    for lock in locks:
        out[lock["hole_coords"][:, 0], lock["hole_coords"][:, 1]] = int(lock["color"])
    return out


def _random_composition(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(int(x) for x in rng.integers(0, total + 1, size=parts - 1))
    vals: list[int] = []
    prev = 0
    for cut in cuts:
        vals.append(int(cut - prev))
        prev = cut
    vals.append(int(total - prev))
    return vals


def _random_key_shape(rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        h = int(rng.integers(2, 5))
        w = int(rng.integers(2, 5))
        max_area = min(h * w - 1, 7)
        if max_area < 4:
            continue
        target = int(rng.integers(4, max_area + 1))
        start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        cells = {start}
        frontier = set()
        for dr, dc in DIRS4:
            nr, nc = start[0] + dr, start[1] + dc
            if 0 <= nr < h and 0 <= nc < w:
                frontier.add((nr, nc))
        while len(cells) < target and frontier:
            idx = int(rng.integers(0, len(frontier)))
            cell = list(frontier)[idx]
            frontier.remove(cell)
            cells.add(cell)
            for dr, dc in DIRS4:
                nr, nc = cell[0] + dr, cell[1] + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if len(cells) != target:
            continue
        raw = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            raw[r, c] = True
        coords = np.argwhere(raw)
        mask = _cropped_mask(coords)
        if min(mask.shape) < 2:
            continue
        if int(mask.sum()) == int(mask.size):
            continue
        if _has_hole(mask):
            continue
        if len(_rotation_variants(mask)) < 2:
            continue
        return mask
    raise RuntimeError("failed to generate a valid key shape")


def _paint_patch(target: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    sub = target[top : top + h, left : left + w]
    mask = patch != int(bg)
    sub[mask] = patch[mask]


def _build_key_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(color)
    return patch


def _build_lock_patch(hole_mask: np.ndarray, color: int, bg: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = hole_mask.shape
    patch = np.full((h + 2, w + 2), int(bg), dtype=int)
    patch[0, :] = int(color)
    patch[-1, :] = int(color)
    patch[:, 0] = int(color)
    patch[:, -1] = int(color)
    inner = patch[1:-1, 1:-1]
    inner[~hole_mask] = int(color)
    hole_coords = np.argwhere(hole_mask) + np.array([1, 1], dtype=int)
    return patch, hole_coords


def _required_row_width(patches: list[np.ndarray], base_gap: int = 1) -> int:
    return int(sum(int(p.shape[1]) for p in patches) + base_gap * (len(patches) + 1))


def _pack_row(
    patches: list[np.ndarray], panel_h: int, panel_w: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    order = list(rng.permutation(len(patches)))
    base_gap = 1
    widths = [int(patches[i].shape[1]) for i in order]
    required = sum(widths) + base_gap * (len(patches) + 1)
    assert required <= panel_w
    extras = _random_composition(panel_w - required, len(patches) + 1, rng)
    gaps = [base_gap + extra for extra in extras]
    placements: list[tuple[int, int] | None] = [None] * len(patches)
    x = gaps[0]
    for j, idx in enumerate(order):
        patch = patches[idx]
        h, w = map(int, patch.shape)
        if panel_h - h >= 2:
            y = int(rng.integers(1, panel_h - h))
        else:
            y = int(max(0, (panel_h - h) // 2))
        placements[idx] = (y, x)
        x += w + gaps[j + 1]
    return [(int(y), int(x)) for y, x in placements]


def _generate_example(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    bg_choices = [0, 1, 2, 3, 4, 6, 7, 8, 9]
    bg = int(rng.choice(bg_choices))
    n_pairs = int(rng.choice([1, 2, 2, 3, 3]))

    key_shapes: list[np.ndarray] = []
    seen_shapes: set[tuple[tuple[int, int], bytes]] = set()
    while len(key_shapes) < n_pairs:
        shape = _random_key_shape(rng)
        key = _shape_key(shape)
        if key in seen_shapes:
            continue
        seen_shapes.add(key)
        key_shapes.append(shape)

    rotations = [int(rng.integers(0, 4)) for _ in range(n_pairs)]
    if all(rot == 0 for rot in rotations):
        rotations[int(rng.integers(0, n_pairs))] = int(rng.integers(1, 4))

    palette = [c for c in range(10) if c not in (bg, GREY)]
    rng.shuffle(palette)
    lock_colors = [int(c) for c in palette[:n_pairs]]

    same_flags = [bool(rng.random() < 0.45) for _ in range(n_pairs)]
    if n_pairs >= 2 and (all(same_flags) or not any(same_flags)):
        flip_idx = int(rng.integers(0, n_pairs))
        same_flags[flip_idx] = not same_flags[flip_idx]
    key_colors: list[int] = []
    for i in range(n_pairs):
        if same_flags[i]:
            key_colors.append(int(lock_colors[i]))
        else:
            choices = [c for c in palette if int(c) != int(lock_colors[i])]
            key_colors.append(int(rng.choice(choices)))

    key_patches = [_build_key_patch(mask, color, bg) for mask, color in zip(key_shapes, key_colors)]
    lock_patches: list[np.ndarray] = []
    hole_offsets: list[np.ndarray] = []
    rotated_holes: list[np.ndarray] = []
    for mask, color, rot in zip(key_shapes, lock_colors, rotations):
        hole = np.rot90(mask, rot)
        rotated_holes.append(hole)
        patch, hole_coords = _build_lock_patch(hole, color, bg)
        lock_patches.append(patch)
        hole_offsets.append(hole_coords)

    top_req = _required_row_width(key_patches)
    bottom_req = _required_row_width(lock_patches)
    width_min = max(14, top_req, bottom_req)
    if width_min > 30:
        raise RuntimeError("width budget exceeded")
    width = int(width_min + rng.integers(0, min(7, 31 - width_min)))

    top_h_min = max(int(p.shape[0]) for p in key_patches) + 2
    bottom_h_min = max(int(p.shape[0]) for p in lock_patches) + 2
    top_h = int(top_h_min + rng.integers(0, 4))
    bottom_h = int(bottom_h_min + rng.integers(0, 5))
    if top_h + 1 + bottom_h > 30:
        raise RuntimeError("height budget exceeded")

    top_panel = np.full((top_h, width), int(bg), dtype=int)
    bottom_panel = np.full((bottom_h, width), int(bg), dtype=int)
    top_places = _pack_row(key_patches, top_h, width, rng)
    bottom_places = _pack_row(lock_patches, bottom_h, width, rng)

    inp = np.full((top_h + 1 + bottom_h, width), int(bg), dtype=int)
    inp[top_h, :] = GREY

    for patch, (r, c) in zip(key_patches, top_places):
        _paint_patch(inp[:top_h], patch, r, c, bg)
    for patch, (r, c), hole_coords, color in zip(lock_patches, bottom_places, hole_offsets, lock_colors):
        _paint_patch(inp[top_h + 1 :], patch, r, c, bg)
    expected = inp.copy()
    expected[:top_h, :] = int(bg)
    expected[top_h, :] = GREY
    for (r, c), patch in zip(bottom_places, lock_patches):
        _paint_patch(expected[top_h + 1 :], patch, r, c, bg)
    for hole_coords, (r, c), color in zip(hole_offsets, bottom_places, lock_colors):
        abs_holes = hole_coords + np.array([top_h + 1 + r, c], dtype=int)
        expected[abs_holes[:, 0], abs_holes[:, 1]] = int(color)

    if _most_common_color(inp) != bg:
        raise RuntimeError("background lost dominance")

    return inp, expected


def generate_input() -> np.ndarray:
    """
    Randomly generate a two-panel key/lock puzzle.
    The lock hole matches a rotated key, while the fill color comes from the lock.
    """
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            inp, expected = _generate_example(rng)
        except RuntimeError:
            continue
        if not np.array_equal(main(inp), expected):
            continue
        return inp
    raise RuntimeError("failed to generate a valid input")
