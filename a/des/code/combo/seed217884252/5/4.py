import numpy as np


BG_COLOR = 7
FRAME_COLOR = 5
PATTERN_COLORS = (1, 2, 3, 4, 6, 8, 9)
MAX_GENERATION_ATTEMPTS = 400
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_RNG = np.random.default_rng()


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop_to_mask(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _is_frame_component(grid: np.ndarray, color: int, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.array_equal(sub == color, border)


def _find_frames(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    groups: dict[tuple[int, int, int], list[tuple[int, int, int, int]]] = {}
    for color in np.unique(grid):
        if int(color) == bg:
            continue
        for comp in _components(grid == int(color)):
            if not _is_frame_component(grid, int(color), comp):
                continue
            box = _bbox_from_coords(comp)
            h = box[2] - box[0] + 1
            w = box[3] - box[1] + 1
            groups.setdefault((int(color), h, w), []).append(box)

    candidates = [(key, boxes) for key, boxes in groups.items() if len(boxes) >= 2]
    assert candidates, "expected at least two equal hollow frames"
    key, boxes = max(candidates, key=lambda item: (len(item[1]), item[0][1] * item[0][2]))
    _ = key
    return sorted(boxes, key=lambda box: (box[1], box[0]))


def _extract_reference(grid: np.ndarray, bg: int, frames: list[tuple[int, int, int, int]]) -> np.ndarray:
    outside = grid != bg
    for r0, c0, r1, c1 in frames:
        outside[r0 : r1 + 1, c0 : c1 + 1] = False
    return _crop_to_mask(grid, outside)


def main(grid: np.ndarray) -> np.ndarray:
    """Normalize each rotated framed motif to the reference orientation and concatenate them."""
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames = _find_frames(g, bg)
    reference = _extract_reference(g, bg, frames)

    normalized: list[np.ndarray] = []
    for r0, c0, r1, c1 in frames:
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        motif = _crop_to_mask(interior, interior != bg)
        match = None
        for k in range(4):
            rotated = np.rot90(motif, k)
            if rotated.shape == reference.shape and np.array_equal(rotated, reference):
                match = rotated
                break
        assert match is not None, "frame motif must match the reference up to rotation"
        normalized.append(match)

    return np.concatenate(normalized, axis=1)


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _grow_connected_mask(h: int, w: int, target: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (int(rng.integers(h)), int(rng.integers(w)))
    mask[start] = True
    frontier: set[tuple[int, int]] = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                frontier.add((nr, nc))

    add_frontier(*start)
    while int(mask.sum()) < target and frontier:
        idx = int(rng.integers(len(frontier)))
        cell = tuple(list(frontier)[idx])
        frontier.remove(cell)
        mask[cell] = True
        add_frontier(*cell)
    return mask


def _pattern_signature(arr: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return (tuple(map(int, arr.shape)), arr.tobytes())


def _make_pattern(inner_side: int, rng: np.random.Generator) -> np.ndarray | None:
    ph = int(rng.integers(3, inner_side + 1))
    pw = int(rng.integers(3, inner_side + 1))
    area = ph * pw
    lo = max(5, int(np.ceil(area * 0.35)))
    hi = min(area, int(np.ceil(area * 0.78)))
    if lo > hi:
        return None
    target = int(rng.integers(lo, hi + 1))

    mask = _grow_connected_mask(ph, pw, target, rng)
    if int(mask.sum()) != target or not _touches_all_sides(mask):
        return None

    n_colors = int(rng.integers(1, 4))
    colors = rng.choice(PATTERN_COLORS, size=n_colors, replace=False)
    coords = np.argwhere(mask)
    rng.shuffle(coords)
    labels = rng.choice(colors, size=len(coords))
    for i, color in enumerate(colors):
        labels[i] = color

    pattern = np.full((ph, pw), BG_COLOR, dtype=int)
    for (r, c), color in zip(coords, labels):
        pattern[int(r), int(c)] = int(color)

    variants = {_pattern_signature(np.rot90(pattern, k)) for k in range(4)}
    if len(variants) != 4:
        return None
    return pattern


def _frame_patch(pattern: np.ndarray, inner_side: int, rotation: int, rng: np.random.Generator) -> np.ndarray:
    patch = np.full((inner_side + 2, inner_side + 2), BG_COLOR, dtype=int)
    patch[0, :] = FRAME_COLOR
    patch[-1, :] = FRAME_COLOR
    patch[:, 0] = FRAME_COLOR
    patch[:, -1] = FRAME_COLOR

    rotated = np.rot90(pattern, rotation)
    rh, rw = rotated.shape
    off_r = int(rng.integers(0, inner_side - rh + 1))
    off_c = int(rng.integers(0, inner_side - rw + 1))
    patch[1 + off_r : 1 + off_r + rh, 1 + off_c : 1 + off_c + rw] = rotated
    return patch


def _assemble_grid(
    pattern: np.ndarray,
    frame_patches: list[np.ndarray],
    frame_gap: int,
    ref_side: str,
    ref_gap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    outer = int(frame_patches[0].shape[0])
    total_w = len(frame_patches) * outer + (len(frame_patches) - 1) * frame_gap
    ref_h, ref_w = map(int, pattern.shape)

    if ref_side in ("top", "bottom"):
        width = max(total_w, ref_w) + int(rng.integers(0, 4))
        height = ref_h + ref_gap + outer
        grid = np.full((height, width), BG_COLOR, dtype=int)
        frames_c = int(rng.integers(0, width - total_w + 1))
        ref_c = int(rng.integers(0, width - ref_w + 1))
        if ref_side == "top":
            ref_r, frames_r = 0, ref_h + ref_gap
        else:
            frames_r, ref_r = 0, outer + ref_gap
    else:
        height = max(outer, ref_h) + int(rng.integers(0, 4))
        width = ref_w + ref_gap + total_w
        grid = np.full((height, width), BG_COLOR, dtype=int)
        frames_r = int(rng.integers(0, height - outer + 1))
        ref_r = int(rng.integers(0, height - ref_h + 1))
        if ref_side == "left":
            ref_c, frames_c = 0, ref_w + ref_gap
        else:
            frames_c, ref_c = 0, total_w + ref_gap

    for i, patch in enumerate(frame_patches):
        c0 = frames_c + i * (outer + frame_gap)
        grid[frames_r : frames_r + outer, c0 : c0 + outer] = patch

    ref_mask = pattern != BG_COLOR
    target = grid[ref_r : ref_r + ref_h, ref_c : ref_c + ref_w]
    target[ref_mask] = pattern[ref_mask]

    pads = [(int(rng.integers(1, 4)), int(rng.integers(1, 4))) for _ in range(2)]
    return np.pad(grid, pads, constant_values=BG_COLOR)


def _expected_output(pattern: np.ndarray, count: int) -> np.ndarray:
    return np.concatenate([pattern.copy() for _ in range(count)], axis=1)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    inner_side = int(rng.integers(5, 10))
    pattern = _make_pattern(inner_side, rng)
    if pattern is None:
        return None

    frame_count = int(rng.integers(2, 6))
    rotations = rng.integers(0, 4, size=frame_count)
    frame_gap = int(rng.integers(1, 4))
    ref_gap = int(rng.integers(2, 5))
    ref_side = str(rng.choice(["top", "bottom", "left", "right"]))

    patches = [_frame_patch(pattern, inner_side, int(k), rng) for k in rotations]
    grid = _assemble_grid(pattern, patches, frame_gap, ref_side, ref_gap, rng)
    if np.array_equal(main(grid), _expected_output(pattern, frame_count)):
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = _RNG
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
