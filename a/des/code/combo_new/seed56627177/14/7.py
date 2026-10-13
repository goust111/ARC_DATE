import numpy as np


BG = 0
CHEST_COLOR = 5
LOCK_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
LOCK_SIDES = ("top", "bottom", "left", "right")

GEN_RETRIES = 300
KEY_SHAPE_RETRIES = 200
CANVAS_RETRIES = 80

_RNG = np.random.default_rng()


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
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rs = coords[:, 0]
    cs = coords[:, 1]
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _tight_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _find_chests(grid: np.ndarray) -> tuple[int, list[dict]]:
    best = None
    for color in np.unique(grid):
        color = int(color)
        if color == BG:
            continue
        rects = []
        total_area = 0
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            area = int(coords.shape[0])
            if h >= 3 and w >= 3 and area == h * w:
                rects.append(
                    {
                        "coords": coords,
                        "bbox": (r0, c0, r1, c1),
                        "height": h,
                        "width": w,
                    }
                )
                total_area += area
        if rects:
            score = (len(rects), total_area)
            if best is None or score > best[0]:
                best = (score, color, rects)
    assert best is not None, "expected at least one solid rectangle chest"
    _, chest_color, chests = best
    chests.sort(key=lambda chest: (chest["bbox"][0], chest["bbox"][1]))
    return int(chest_color), chests


def _adjacent_chest_indices(coords: np.ndarray, chests: list[dict]) -> list[int]:
    hits: list[int] = []
    for idx, chest in enumerate(chests):
        r0, c0, r1, c1 = chest["bbox"]
        found = False
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            for dr, dc in DIR4:
                nr = rr + dr
                nc = cc + dc
                if r0 <= nr <= r1 and c0 <= nc <= c1:
                    found = True
                    break
            if found:
                break
        if found:
            hits.append(idx)
    return hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unlock each solid gray chest by matching its colored lock to the same-colored key and reveal the key shape inside.

    Input structure:
    - The background is black.
    - Several solid gray rectangles act as chests; each chest is at least 3x3.
    - Each chest has one adjacent 2-cell monochrome lock placed flush against one outer side.
    - Elsewhere, one 4-connected monochrome key shape of the same color appears for each lock.

    Rule:
    - Pair every lock with the key of the same color.
    - Remove the original key from the background.
    - Stamp that key's shape into the interior of the corresponding chest, centered within the inner area and biased to the top-left when exact centering is impossible.
    - Keep the chest border gray, keep the lock unchanged, and leave all other chest-interior cells gray.

    Key insight:
    - Color equality, not spatial proximity, decides which scattered key belongs to which chest; the transferred shape is then centered inside the matched gray rectangle.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    chest_color, chests = _find_chests(g)

    locks: dict[int, int] = {}
    keys: dict[int, dict] = {}
    for color in np.unique(g):
        color = int(color)
        if color in (BG, chest_color):
            continue
        for coords in _components(g == color):
            mask, bbox = _tight_mask(coords)
            h, w = mask.shape
            area = int(coords.shape[0])
            touching = _adjacent_chest_indices(coords, chests)
            is_lock = area == 2 and (h, w) in {(1, 2), (2, 1)} and len(touching) == 1
            if is_lock:
                locks[color] = touching[0]
            else:
                keys[color] = {"coords": coords, "mask": mask, "bbox": bbox}

    assert locks, "expected colored locks"
    assert set(locks) == set(keys), "every lock color must have exactly one key"

    for key in keys.values():
        for r, c in key["coords"]:
            out[int(r), int(c)] = BG

    for color, chest_idx in locks.items():
        chest = chests[chest_idx]
        key_mask = keys[color]["mask"]
        kh, kw = key_mask.shape
        r0, c0, r1, c1 = chest["bbox"]
        inner_h = int(r1 - r0 - 1)
        inner_w = int(c1 - c0 - 1)
        assert kh <= inner_h and kw <= inner_w, "key must fit chest interior"
        top = int(r0 + 1 + (inner_h - kh) // 2)
        left = int(c0 + 1 + (inner_w - kw) // 2)
        rows, cols = np.where(key_mask)
        out[top + rows, left + cols] = int(color)

    return out


def _random_key_mask() -> np.ndarray:
    area = int(_RNG.integers(2, 7))
    mode = int(_RNG.integers(0, 3))
    if mode == 0:
        if _RNG.random() < 0.5:
            return np.ones((1, area), dtype=bool)
        return np.ones((area, 1), dtype=bool)
    if mode == 1:
        factor_pairs = [(h, w) for h in range(1, area + 1) for w in range(1, area + 1) if h * w == area]
        h, w = factor_pairs[int(_RNG.integers(0, len(factor_pairs)))]
        return np.ones((h, w), dtype=bool)

    for _ in range(KEY_SHAPE_RETRIES):
        box_h = int(_RNG.integers(1, 7))
        box_w = int(_RNG.integers(1, 7))
        if box_h * box_w < area:
            continue
        cells = {(int(_RNG.integers(0, box_h)), int(_RNG.integers(0, box_w)))}
        while len(cells) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = frontier[int(_RNG.integers(0, len(frontier)))]
            cells.add((nr, nc))
        if len(cells) != area:
            continue
        coords = np.array(sorted(cells), dtype=int)
        mask, _ = _tight_mask(coords)
        return mask
    raise RuntimeError("failed to generate a connected key mask")


def _build_chest_patch(chest_h: int, chest_w: int, side: str, color: int) -> np.ndarray:
    extra_h = 1 if side in ("top", "bottom") else 0
    extra_w = 1 if side in ("left", "right") else 0
    patch = np.zeros((chest_h + extra_h, chest_w + extra_w), dtype=int)
    chest_r0 = 1 if side == "top" else 0
    chest_c0 = 1 if side == "left" else 0
    patch[chest_r0 : chest_r0 + chest_h, chest_c0 : chest_c0 + chest_w] = CHEST_COLOR

    if side == "top":
        start = int(_RNG.integers(0, chest_w - 1))
        patch[chest_r0 - 1, chest_c0 + start : chest_c0 + start + 2] = color
    elif side == "bottom":
        start = int(_RNG.integers(0, chest_w - 1))
        patch[chest_r0 + chest_h, chest_c0 + start : chest_c0 + start + 2] = color
    elif side == "left":
        start = int(_RNG.integers(0, chest_h - 1))
        patch[chest_r0 + start : chest_r0 + start + 2, chest_c0 - 1] = color
    else:
        start = int(_RNG.integers(0, chest_h - 1))
        patch[chest_r0 + start : chest_r0 + start + 2, chest_c0 + chest_w] = color

    return patch


def _make_key_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _all_positions(limit: int) -> range:
    return range(1, limit) if limit > 1 else range(0)


def _place_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    padded_area = sum((patch.shape[0] + 2) * (patch.shape[1] + 2) for patch in patches)
    max_h = max(patch.shape[0] for patch in patches)
    max_w = max(patch.shape[1] for patch in patches)
    side_floor = int(np.ceil(np.sqrt(padded_area)))
    min_h = max(max_h + 2, side_floor - 2, 14)
    min_w = max(max_w + 2, side_floor - 2, 14)
    if min_h > 30 or min_w > 30:
        return None

    order = sorted(range(len(patches)), key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)
    for _ in range(CANVAS_RETRIES):
        h = int(_RNG.integers(min_h, 31))
        w = int(_RNG.integers(min_w, 31))
        grid = np.zeros((h, w), dtype=int)
        blocked = np.zeros((h, w), dtype=bool)
        placed = True
        for idx in order:
            patch = patches[idx]
            ph, pw = patch.shape
            rows = list(_all_positions(h - ph))
            cols = list(_all_positions(w - pw))
            if not rows or not cols:
                placed = False
                break
            positions = [(r, c) for r in rows for c in cols]
            _RNG.shuffle(positions)
            found = False
            for r, c in positions:
                rr0 = max(0, r - 1)
                cc0 = max(0, c - 1)
                rr1 = min(h, r + ph + 1)
                cc1 = min(w, c + pw + 1)
                if blocked[rr0:rr1, cc0:cc1].any():
                    continue
                mask = patch != 0
                grid[r : r + ph, c : c + pw][mask] = patch[mask]
                blocked[rr0:rr1, cc0:cc1] = True
                found = True
                break
            if not found:
                placed = False
                break
        if placed:
            return grid
    return None


def _gen_attempt() -> np.ndarray | None:
    n_chests = int(_RNG.integers(2, 5))
    colors = list(_RNG.choice(LOCK_COLOR_POOL, size=n_chests, replace=False))
    patches: list[np.ndarray] = []

    for color in colors:
        key_mask = _random_key_mask()
        kh, kw = key_mask.shape
        inner_h = kh + int(_RNG.integers(0, 4))
        inner_w = kw + int(_RNG.integers(0, 4))
        chest_h = inner_h + 2
        chest_w = inner_w + 2
        side = LOCK_SIDES[int(_RNG.integers(0, len(LOCK_SIDES)))]
        chest_patch = _build_chest_patch(chest_h, chest_w, side, int(color))
        key_patch = _make_key_patch(key_mask, int(color))
        patches.extend([chest_patch, key_patch])

    _RNG.shuffle(patches)
    grid = _place_patches(patches)
    if grid is None:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
