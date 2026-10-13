from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
KEY_COLORS = (2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            out.append(comp)
    return out


def _coords_to_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    arr = np.array(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    mask[arr[:, 0] - r0, arr[:, 1] - c0] = True
    return mask


def _shape_signature(mask: np.ndarray) -> tuple[int, int, bytes]:
    mask = _tight_crop(mask.astype(bool))
    best: tuple[int, int, bytes] | None = None
    for k in range(4):
        rot = _tight_crop(np.rot90(mask, int(k)))
        sig = (int(rot.shape[0]), int(rot.shape[1]), rot.astype(np.uint8).tobytes())
        if best is None or sig < best:
            best = sig
    assert best is not None
    return best


def _find_divider_row(grid: np.ndarray) -> int:
    candidates = [r for r in range(1, grid.shape[0] - 1) if np.all(grid[r, :] == GREY)]
    assert len(candidates) == 1
    return int(candidates[0])


def _extract_key_colors(top: np.ndarray) -> dict[tuple[int, int, bytes], int]:
    mapping: dict[tuple[int, int, bytes], int] = {}
    for comp in _components(top != BLACK):
        colors = {int(top[r, c]) for r, c in comp}
        assert len(colors) == 1
        color = colors.pop()
        assert color not in {BLACK, BLUE, GREY}
        sig = _shape_signature(_coords_to_mask(comp))
        assert sig not in mapping
        mapping[sig] = int(color)
    assert mapping
    return mapping


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each blue frame interior with the color of the dictionary key whose shape matches the frame's border gap.

    Input structure:
    - Black background.
    - A full-width grey horizontal divider splits the grid into an upper dictionary region and a lower puzzle region.
    - The upper region contains several disconnected monochrome key polyominoes in distinct non-black colors.
    - The lower region contains several blue hollow rectangular frames; each frame border has a black gap whose missing-pixel shape is a connected polyomino matching one dictionary key up to rotation.

    Rule:
    - Find the grey divider row and extract every key shape from the dictionary region.
    - For each blue frame, recover its outer rectangle from the frame component's bounding box and read the gap as the set of missing perimeter cells.
    - Match the gap shape to the dictionary under 0/90/180/270 degree rotation and take the matched key's color.
    - Fill the black pixels strictly inside that frame rectangle with the matched color, leaving the blue border and the gap unchanged.

    Key insight:
    - The missing perimeter cells form a standalone lookup shape; once the gap is normalized and matched against the rotated key shapes, the frame's interior color is determined.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    divider = _find_divider_row(g)
    key_colors = _extract_key_colors(g[:divider, :])

    out = g.copy()
    lower = g[divider + 1 :, :]
    for comp in _components(lower == BLUE):
        arr = np.array(comp, dtype=int)
        r0, c0 = arr.min(axis=0)
        r1, c1 = arr.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        assert r1 - r0 >= 2 and c1 - c0 >= 2

        gap = [(r, c) for r, c in _perimeter_cells(r0, c0, r1, c1) if int(lower[r, c]) != BLUE]
        assert gap
        gap_sig = _shape_signature(_coords_to_mask(gap))
        fill_color = key_colors[gap_sig]

        rr0 = divider + 1 + r0 + 1
        rr1 = divider + 1 + r1
        cc0 = c0 + 1
        cc1 = c1
        interior = out[rr0:rr1, cc0:cc1]
        interior[interior == BLACK] = fill_color

    return out


SHAPE_LIBRARY = (
    {"name": "line3", "mask": np.array([[1, 1, 1]], dtype=bool), "family": "line"},
    {"name": "line4", "mask": np.array([[1, 1, 1, 1]], dtype=bool), "family": "line"},
    {"name": "line5", "mask": np.array([[1, 1, 1, 1, 1]], dtype=bool), "family": "line"},
    {"name": "el3", "mask": np.array([[1, 1], [1, 0]], dtype=bool), "family": "corner"},
    {"name": "el4", "mask": np.array([[1, 1, 1], [1, 0, 0]], dtype=bool), "family": "corner"},
    {"name": "el5a", "mask": np.array([[1, 1, 1, 1], [1, 0, 0, 0]], dtype=bool), "family": "corner"},
    {"name": "el5b", "mask": np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool), "family": "corner"},
)

GEN_MAX_ATTEMPTS = 400
CANVAS_TRIES = 80
KEY_PAD = 1
FRAME_PAD = 1


def _paint_mask(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BLACK, dtype=int)
    patch[mask] = int(color)
    return patch


def _place_patches(
    patches: list[np.ndarray], height: int, width: int, pad: int, rng: np.random.Generator
) -> list[tuple[int, int]] | None:
    if height < 3 or width < 3:
        return None
    blocked = np.zeros((height, width), dtype=bool)
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True

    order = sorted(
        range(len(patches)),
        key=lambda idx: int(patches[idx].shape[0] * patches[idx].shape[1]),
        reverse=True,
    )
    placements: list[tuple[int, int] | None] = [None] * len(patches)
    for idx in order:
        ph, pw = map(int, patches[idx].shape)
        candidates: list[tuple[int, int]] = []
        for r in range(1, height - ph):
            for c in range(1, width - pw):
                if not bool(blocked[r : r + ph, c : c + pw].any()):
                    candidates.append((r, c))
        if not candidates:
            return None
        pos = candidates[int(rng.integers(0, len(candidates)))]
        placements[idx] = pos
        r, c = pos
        rr0 = max(0, r - pad)
        rr1 = min(height, r + ph + pad)
        cc0 = max(0, c - pad)
        cc1 = min(width, c + pw + pad)
        blocked[rr0:rr1, cc0:cc1] = True

    assert all(pos is not None for pos in placements)
    return [(int(pos[0]), int(pos[1])) for pos in placements]


def _frame_gap_from_corner(mask: np.ndarray, outer_h: int, outer_w: int, rot_k: int) -> list[tuple[int, int]]:
    rot = _tight_crop(np.rot90(mask, int(rot_k)))
    gh, gw = map(int, rot.shape)
    if rot_k % 4 == 0:
        base_r, base_c = 0, 0
    elif rot_k % 4 == 1:
        base_r, base_c = 0, outer_w - gw
    elif rot_k % 4 == 2:
        base_r, base_c = outer_h - gh, outer_w - gw
    else:
        base_r, base_c = outer_h - gh, 0
    coords: list[tuple[int, int]] = []
    for r in range(gh):
        for c in range(gw):
            if bool(rot[r, c]):
                coords.append((int(base_r + r), int(base_c + c)))
    return coords


def _make_frame_patch(shape: dict, rng: np.random.Generator) -> np.ndarray:
    base_mask = shape["mask"]
    rot_k = int(rng.integers(0, 4))

    if shape["family"] == "line":
        gap_mask = _tight_crop(np.rot90(base_mask, rot_k))
        gh, gw = map(int, gap_mask.shape)
        if gh == 1:
            outer_h = int(rng.integers(5, 10))
            outer_w = int(rng.integers(max(5, gw + 2), 11))
            side = "top" if bool(rng.integers(0, 2)) else "bottom"
            start = int(rng.integers(1, outer_w - gw))
            gap = [
                (0 if side == "top" else outer_h - 1, start + dc)
                for dc in range(gw)
            ]
        else:
            outer_h = int(rng.integers(max(5, gh + 2), 11))
            outer_w = int(rng.integers(5, 10))
            side = "left" if bool(rng.integers(0, 2)) else "right"
            start = int(rng.integers(1, outer_h - gh))
            gap = [
                (start + dr, 0 if side == "left" else outer_w - 1)
                for dr in range(gh)
            ]
    else:
        gap_mask = _tight_crop(np.rot90(base_mask, rot_k))
        gh, gw = map(int, gap_mask.shape)
        outer_h = int(rng.integers(max(5, gh + 1), 11))
        outer_w = int(rng.integers(max(5, gw + 1), 11))
        gap = _frame_gap_from_corner(base_mask, outer_h, outer_w, rot_k)

    patch = np.full((outer_h, outer_w), BLACK, dtype=int)
    patch[0, :] = BLUE
    patch[-1, :] = BLUE
    patch[:, 0] = BLUE
    patch[:, -1] = BLUE
    for r, c in gap:
        patch[int(r), int(c)] = BLACK
    return patch


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_keys = int(rng.integers(3, 6))
    shape_ids = rng.choice(len(SHAPE_LIBRARY), size=n_keys, replace=False)
    colors = rng.choice(np.array(KEY_COLORS, dtype=int), size=n_keys, replace=False)

    key_specs = []
    key_patches: list[np.ndarray] = []
    for shape_id, color in zip(shape_ids.tolist(), colors.tolist(), strict=False):
        shape = SHAPE_LIBRARY[int(shape_id)]
        rot_k = int(rng.integers(0, 4))
        key_mask = _tight_crop(np.rot90(shape["mask"], rot_k))
        key_patches.append(_paint_mask(key_mask, int(color)))
        key_specs.append({"shape": shape, "color": int(color)})

    max_frames = min(4, n_keys)
    n_frames = int(rng.integers(2, max_frames + 1))
    used_keys = rng.choice(n_keys, size=n_frames, replace=False).tolist()

    frame_specs = []
    frame_patches: list[np.ndarray] = []
    for key_idx in used_keys:
        shape = key_specs[int(key_idx)]["shape"]
        frame_patches.append(_make_frame_patch(shape, rng))
        frame_specs.append({"fill_color": int(key_specs[int(key_idx)]["color"])})

    max_key_h = max(int(p.shape[0]) for p in key_patches)
    max_frame_h = max(int(p.shape[0]) for p in frame_patches)

    top_reserved = sum((int(p.shape[0]) + 2 * KEY_PAD) * (int(p.shape[1]) + 2 * KEY_PAD) for p in key_patches)
    bottom_reserved = sum((int(p.shape[0]) + 2 * FRAME_PAD) * (int(p.shape[1]) + 2 * FRAME_PAD) for p in frame_patches)

    for _ in range(CANVAS_TRIES):
        width = int(rng.integers(16, 31))
        usable_w = max(1, width - 2)
        top_h = max(6, max_key_h + 2, (top_reserved + usable_w - 1) // usable_w + 2)
        bottom_h = max(8, max_frame_h + 2, (bottom_reserved + usable_w - 1) // usable_w + 2)
        total_min = top_h + 1 + bottom_h
        if total_min > 30:
            continue

        extra = int(rng.integers(0, 31 - total_min))
        extra_top = int(rng.integers(0, extra + 1))
        top_h += extra_top
        bottom_h += extra - extra_top

        key_positions = _place_patches(key_patches, top_h, width, KEY_PAD, rng)
        if key_positions is None:
            continue
        frame_positions = _place_patches(frame_patches, bottom_h, width, FRAME_PAD, rng)
        if frame_positions is None:
            continue

        grid = np.full((top_h + 1 + bottom_h, width), BLACK, dtype=int)
        grid[top_h, :] = GREY
        for patch, (r0, c0) in zip(key_patches, key_positions, strict=False):
            ph, pw = patch.shape
            target = grid[r0 : r0 + ph, c0 : c0 + pw]
            target[patch != BLACK] = patch[patch != BLACK]

        for patch, (r0, c0) in zip(frame_patches, frame_positions, strict=False):
            ph, pw = patch.shape
            grid[top_h + 1 + r0 : top_h + 1 + r0 + ph, c0 : c0 + pw] = patch

        expected = grid.copy()
        for patch, spec, (r0, c0) in zip(frame_patches, frame_specs, frame_positions, strict=False):
            ph, pw = patch.shape
            rr0 = top_h + 1 + r0 + 1
            rr1 = top_h + 1 + r0 + ph - 1
            cc0 = c0 + 1
            cc1 = c0 + pw - 1
            expected[rr0:rr1, cc0:cc1] = spec["fill_color"]

        try:
            actual = main(grid)
        except Exception:
            continue
        if not np.array_equal(actual, expected):
            continue
        if np.array_equal(grid, expected):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            assert grid.ndim == 2 and max(grid.shape) <= 30
            return grid
    raise RuntimeError("failed to generate a valid input")
