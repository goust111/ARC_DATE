from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
GREY = 5
KEY_COLORS = (2, 3, 4, 6, 7, 8, 9)

GEN_RETRIES = 400
SHAPE_RETRIES = 200
PLACE_RETRIES = 120
PAIR_COUNT_RANGE = (2, 4)
SHAPE_SIDE_RANGE = (1, 4)
SHAPE_AREA_MAX = 6
FRAME_MARGIN_RANGE = (1, 2)
GRID_SIDE_RANGE = (22, 34)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    deltas = _neighbors4(0, 0) if connectivity == 4 else _neighbors8(0, 0)

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in deltas:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _shape_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    shape = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    shape[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return shape


def _shape_sig(shape: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return (tuple(int(x) for x in shape.shape), shape.astype(np.uint8).tobytes())


def _touches_border(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _boundary_colors(grid: np.ndarray, coords: np.ndarray) -> set[int]:
    h, w = grid.shape
    pts = {(int(r), int(c)) for r, c in coords}
    colors: set[int] = set()
    for r, c in pts:
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in pts:
                colors.add(int(grid[nr, nc]))
    return colors


def _is_rect_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False
    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(local, border))


def _extract_keys(grid: np.ndarray) -> list[dict]:
    keys: list[dict] = []
    for comp in _components(grid == GREY, connectivity=4):
        if not _is_rect_frame(comp):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        key_mask = (interior != BG) & (interior != GREY)
        comps = _components(key_mask, connectivity=4)
        assert len(comps) == 1, "every frame must contain exactly one key component"
        coords = comps[0]
        ih, iw = interior.shape
        assert not _touches_border(coords, int(ih), int(iw)), "key must not touch its frame"
        colors = {int(v) for v in interior[key_mask]}
        assert len(colors) == 1, "key must be monochrome"
        color = colors.pop()
        keys.append({"shape": _shape_from_coords(coords), "color": int(color)})
    assert keys, "expected at least one grey frame"
    return keys


def _extract_lock_cavities(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    cavities: list[dict] = []
    for comp in _components(grid == BG, connectivity=4):
        if _touches_border(comp, int(h), int(w)):
            continue
        bcols = _boundary_colors(grid, comp)
        if bcols == {BLUE}:
            cavities.append({"coords": comp, "shape": _shape_from_coords(comp)})
    return cavities


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each framed key shape to the unique blue lockhole cavity with the same
    translated shape, then fill that cavity with the key's color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    keys = _extract_keys(g)
    cavities = _extract_lock_cavities(g)

    key_by_shape: dict[tuple[tuple[int, int], bytes], int] = {}
    for key in keys:
        sig = _shape_sig(key["shape"])
        assert sig not in key_by_shape, "key shapes must be unique"
        key_by_shape[sig] = int(key["color"])

    assert len(cavities) == len(keys), "keys and lockholes must pair one-to-one"

    seen: set[tuple[tuple[int, int], bytes]] = set()
    for cavity in cavities:
        sig = _shape_sig(cavity["shape"])
        assert sig in key_by_shape, "every lockhole cavity must match a key shape"
        seen.add(sig)
        coords = cavity["coords"]
        out[coords[:, 0], coords[:, 1]] = int(key_by_shape[sig])

    assert seen == set(key_by_shape), "every key must be used exactly once"
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    pts = np.argwhere(mask)
    for r, c in pts:
        for nr, nc in _neighbors8(int(r), int(c)) + [(int(r), int(c))]:
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = True
    return out


def _shape_has_hole(shape: np.ndarray) -> bool:
    bg_comps = _components(~shape, connectivity=4)
    h, w = shape.shape
    for comp in bg_comps:
        if not _touches_border(comp, int(h), int(w)):
            return True
    return False


def _make_random_shape() -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        raw_h = random.randint(*SHAPE_SIDE_RANGE)
        raw_w = random.randint(*SHAPE_SIDE_RANGE)
        max_area = min(raw_h * raw_w, SHAPE_AREA_MAX)
        area = random.randint(1, max_area)

        start = (random.randrange(raw_h), random.randrange(raw_w))
        cells = {start}
        frontier = {
            (nr, nc)
            for nr, nc in _neighbors4(*start)
            if 0 <= nr < raw_h and 0 <= nc < raw_w
        }

        while len(cells) < area and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nr, nc in _neighbors4(*cell):
                if 0 <= nr < raw_h and 0 <= nc < raw_w and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        if len(cells) != area:
            continue

        shape = np.zeros((raw_h, raw_w), dtype=bool)
        for r, c in cells:
            shape[int(r), int(c)] = True
        shape = _shape_from_coords(np.argwhere(shape))
        shape = np.rot90(shape, random.randrange(4))
        if _shape_has_hole(shape):
            continue
        return shape
    raise RuntimeError("failed to sample a valid key shape")


def _make_lock_patch(shape: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    cavity = np.pad(shape.astype(bool), 1, constant_values=False)
    outline = _dilate8(cavity) & ~cavity
    patch = np.full(cavity.shape, BG, dtype=int)
    patch[outline] = BLUE

    blue_comps = _components(patch == BLUE, connectivity=8)
    if len(blue_comps) != 1:
        return None

    cavities = _extract_lock_cavities(patch)
    if len(cavities) != 1:
        return None
    if not np.array_equal(cavities[0]["shape"], shape):
        return None
    return patch, cavity


def _make_frame_patch(shape: np.ndarray, color: int) -> np.ndarray:
    top = random.randint(*FRAME_MARGIN_RANGE)
    bottom = random.randint(*FRAME_MARGIN_RANGE)
    left = random.randint(*FRAME_MARGIN_RANGE)
    right = random.randint(*FRAME_MARGIN_RANGE)
    sh, sw = shape.shape
    h = sh + top + bottom + 2
    w = sw + left + right + 2
    patch = np.full((h, w), BG, dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    r0 = 1 + top
    c0 = 1 + left
    patch[r0 : r0 + sh, c0 : c0 + sw][shape] = int(color)
    return patch


def _choose_colors(count: int) -> list[int]:
    if random.random() < 0.5:
        return random.sample(list(KEY_COLORS), count)
    return [random.choice(KEY_COLORS) for _ in range(count)]


def _can_place(
    reserved: np.ndarray, patch_shape: tuple[int, int], top: int, left: int, margin: int = 1
) -> bool:
    h, w = reserved.shape
    ph, pw = patch_shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    r0 = max(0, top - margin)
    c0 = max(0, left - margin)
    r1 = min(h, top + ph + margin)
    c1 = min(w, left + pw + margin)
    return not bool(reserved[r0:r1, c0:c1].any())


def _paste_object(
    grid: np.ndarray, reserved: np.ndarray, patch: np.ndarray, top: int, left: int
) -> None:
    ph, pw = patch.shape
    sub = grid[top : top + ph, left : left + pw]
    mask = patch != BG
    sub[mask] = patch[mask]
    reserved[top : top + ph, left : left + pw] = True


def _sample_grid_shape(patches: list[np.ndarray]) -> tuple[int, int]:
    max_h = max(int(p.shape[0]) for p in patches)
    max_w = max(int(p.shape[1]) for p in patches)
    total_area = sum(int((p.shape[0] + 1) * (p.shape[1] + 1)) for p in patches)
    base = max(
        max_h + 2,
        max_w + 2,
        int(np.ceil(np.sqrt(float(total_area) * 1.55))),
    )
    lo = max(GRID_SIDE_RANGE[0], base)
    hi = max(lo, GRID_SIDE_RANGE[1])
    return random.randint(lo, hi), random.randint(lo, hi)


def _place_patches(
    patches: list[np.ndarray],
) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    h, w = _sample_grid_shape(patches)
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(patches)
    order = sorted(range(len(patches)), key=lambda i: int(patches[i].shape[0] * patches[i].shape[1]), reverse=True)

    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        candidates = [
            (r, c)
            for r in range(h - ph + 1)
            for c in range(w - pw + 1)
            if _can_place(reserved, (int(ph), int(pw)), int(r), int(c))
        ]
        if not candidates:
            return None
        top, left = random.choice(candidates)
        _paste_object(grid, reserved, patch, int(top), int(left))
        placements[idx] = (int(top), int(left))

    return grid, [(int(r), int(c)) for r, c in placements if r is not None and c is not None]


def _expected_output(
    grid: np.ndarray,
    lock_specs: list[dict],
    lock_positions: list[tuple[int, int]],
) -> np.ndarray:
    out = grid.copy()
    for spec, (top, left) in zip(lock_specs, lock_positions, strict=True):
        cavity = spec["cavity_mask"]
        color = int(spec["color"])
        ph, pw = cavity.shape
        region = out[top : top + ph, left : left + pw]
        region[cavity] = color
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        pair_count = random.randint(*PAIR_COUNT_RANGE)
        shapes: list[np.ndarray] = []
        used_shapes: set[tuple[tuple[int, int], bytes]] = set()

        for _pair_idx in range(pair_count):
            for _ in range(SHAPE_RETRIES):
                shape = _make_random_shape()
                lock = _make_lock_patch(shape)
                if lock is None:
                    continue
                sig = _shape_sig(shape)
                if sig in used_shapes:
                    continue
                used_shapes.add(sig)
                shapes.append(shape)
                break
            else:
                shapes = []
                break
        if len(shapes) != pair_count:
            continue

        colors = _choose_colors(pair_count)
        frame_patches: list[np.ndarray] = []
        lock_patches: list[np.ndarray] = []
        lock_specs: list[dict] = []
        all_patches: list[np.ndarray] = []

        for shape, color in zip(shapes, colors, strict=True):
            frame_patch = _make_frame_patch(shape, int(color))
            lock = _make_lock_patch(shape)
            if lock is None:
                break
            lock_patch, cavity_mask = lock
            frame_patches.append(frame_patch)
            lock_patches.append(lock_patch)
            lock_specs.append({"cavity_mask": cavity_mask, "color": int(color)})
            all_patches.append(frame_patch)
            all_patches.append(lock_patch)
        else:
            placement = None
            for _ in range(PLACE_RETRIES):
                placement = _place_patches(all_patches)
                if placement is not None:
                    break
            if placement is None:
                continue

            grid, positions = placement
            frame_positions = positions[0::2]
            lock_positions = positions[1::2]
            assert len(frame_positions) == pair_count and len(lock_positions) == pair_count

            expected = _expected_output(grid, lock_specs, lock_positions)
            actual = main(grid)
            if not np.array_equal(actual, expected):
                continue
            if len(_extract_keys(grid)) != pair_count:
                continue
            if len(_extract_lock_cavities(grid)) != pair_count:
                continue
            return grid

    raise RuntimeError("failed to generate a valid input grid")
