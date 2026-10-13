from __future__ import annotations

import random

import numpy as np


BG = 0
LOCK = 5
KEY_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_ROT = {"top": 0, "left": 1, "bottom": 2, "right": 3}
SIDES = tuple(SIDE_TO_ROT)
GEN_MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


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
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.tobytes()


def _rotation_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        key = _mask_key(rot)
        if key not in seen:
            seen.add(key)
            variants.append(rot)
    return variants


def _foreground_bbox_positions(grid: np.ndarray, patch: np.ndarray, padding: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    ph, pw = patch.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            rr0 = max(0, r0 - padding)
            rr1 = min(h, r0 + ph + padding)
            cc0 = max(0, c0 - padding)
            cc1 = min(w, c0 + pw + padding)
            if np.any(grid[rr0:rr1, cc0:cc1] != BG):
                continue
            positions.append((r0, c0))
    return positions


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    sub = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    mask = patch != BG
    sub[mask] = patch[mask]


def _grow_connected_mask(height: int, width: int, area: int) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=bool)
    start = (0, random.randrange(width))
    mask[start] = True
    frontier = set()
    frontier.update(
        (nr, nc)
        for nr, nc in _neighbors4(*start)
        if 0 <= nr < height and 0 <= nc < width
    )

    while int(mask.sum()) < area:
        choices = [(r, c) for r, c in frontier if not mask[r, c]]
        if not choices:
            return None
        r, c = random.choice(choices)
        mask[r, c] = True
        frontier.discard((r, c))
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                frontier.add((nr, nc))

    tight = _tight_mask(np.argwhere(mask))
    if not tight[0].any():
        return None
    return tight


def _random_slot_mask() -> np.ndarray:
    for _ in range(200):
        height = random.randint(2, 4)
        width = random.randint(2, 5)
        max_area = min(7, height * width - 1)
        if max_area < 4:
            continue
        area = random.randint(4, max_area)
        mask = _grow_connected_mask(height, width, area)
        if mask is None:
            continue
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) >= mask.size:
            continue
        if len(_rotation_variants(mask)) != 4:
            continue
        return mask
    raise RuntimeError("failed to generate a non-degenerate slot mask")


def _random_nonmatching_mask(slot_mask: np.ndarray) -> np.ndarray:
    slot_keys = {_mask_key(rot) for rot in _rotation_variants(slot_mask)}
    slot_area = int(slot_mask.sum())

    for _ in range(200):
        if random.random() < 0.5:
            area = slot_area
        else:
            delta = random.choice([-2, -1, 1, 2])
            area = max(3, min(8, slot_area + delta))
        height = random.randint(2, 5)
        width = random.randint(2, 5)
        if area >= height * width or area > height * width:
            continue
        mask = _grow_connected_mask(height, width, area)
        if mask is None:
            continue
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if any(_mask_key(rot) in slot_keys for rot in _rotation_variants(mask)):
            continue
        return mask
    raise RuntimeError("failed to generate a mismatching key mask")


def _build_lock_patches(color: int, side: str, slot_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sh, sw = map(int, slot_mask.shape)
    if side in {"top", "bottom"}:
        rect_h = random.randint(sh + 2, min(10, sh + 5))
        rect_w = random.randint(sw + 2, min(12, sw + 6))
        slot_offset = random.randint(1, rect_w - sw - 1)
        body = np.full((rect_h, rect_w), LOCK, dtype=int)
        if side == "top":
            slot_view = body[:sh, slot_offset : slot_offset + sw]
        else:
            slot_view = body[rect_h - sh :, slot_offset : slot_offset + sw]
        slot_view[slot_mask] = BG
        shoulder_col = slot_offset - 1 if random.random() < 0.5 else slot_offset + sw

        patch = np.full((rect_h + 1, rect_w), BG, dtype=int)
        if side == "top":
            patch[1:, :] = body
            patch[0, shoulder_col] = color
            filled_body = body.copy()
            filled_body[:sh, slot_offset : slot_offset + sw][slot_mask] = color
            filled = patch.copy()
            filled[1:, :] = filled_body
        else:
            patch[:rect_h, :] = body
            patch[rect_h, shoulder_col] = color
            filled_body = body.copy()
            filled_body[rect_h - sh :, slot_offset : slot_offset + sw][slot_mask] = color
            filled = patch.copy()
            filled[:rect_h, :] = filled_body
        return patch, filled

    rect_h = random.randint(sh + 2, min(12, sh + 6))
    rect_w = random.randint(sw + 2, min(10, sw + 5))
    slot_offset = random.randint(1, rect_h - sh - 1)
    body = np.full((rect_h, rect_w), LOCK, dtype=int)
    if side == "left":
        slot_view = body[slot_offset : slot_offset + sh, :sw]
    else:
        slot_view = body[slot_offset : slot_offset + sh, rect_w - sw :]
    slot_view[slot_mask] = BG
    shoulder_row = slot_offset - 1 if random.random() < 0.5 else slot_offset + sh

    patch = np.full((rect_h, rect_w + 1), BG, dtype=int)
    if side == "left":
        patch[:, 1:] = body
        patch[shoulder_row, 0] = color
        filled_body = body.copy()
        filled_body[slot_offset : slot_offset + sh, :sw][slot_mask] = color
        filled = patch.copy()
        filled[:, 1:] = filled_body
    else:
        patch[:, :rect_w] = body
        patch[shoulder_row, rect_w] = color
        filled_body = body.copy()
        filled_body[slot_offset : slot_offset + sh, rect_w - sw :][slot_mask] = color
        filled = patch.copy()
        filled[:, :rect_w] = filled_body
    return patch, filled


def _key_patch(color: int, mask: np.ndarray) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = color
    return patch


def _expected_from_pairs(
    shape: tuple[int, int],
    pairs: list[dict],
    lock_positions: list[tuple[int, int]],
    key_positions: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, BG, dtype=int)
    out = np.full(shape, BG, dtype=int)

    for pair, (lr, lc), (kr, kc) in zip(pairs, lock_positions, key_positions, strict=True):
        _paste_patch(inp, pair["lock_in"], lr, lc)
        _paste_patch(out, pair["lock_out"], lr, lc)
        _paste_patch(inp, pair["key_patch"], kr, kc)
        if not pair["matched"]:
            _paste_patch(out, pair["key_patch"], kr, kc)

    return inp, out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert each colored key into the matching grey lock notch after rotating it to the complementary orientation.

    Input structure:
    - The background is black.
    - Large grey components are rectangular lock bodies with one black notch carved into one side.
    - Each lock has a single colored marker pixel on the notch entrance edge, and another separate component of the same color is the movable key.

    Rule:
    - Pair each key with the singleton marker of the same color.
    - The marker identifies which grey lock the key belongs to; the black cells inside that lock's grey bbox are the notch to fill.
    - Try the four right-angle rotations of the key. If one rotated key shape matches the notch exactly, erase the key from its original location and fill the notch with the key color. If none match, leave both key and lock unchanged.

    Key insight:
    Color matches keys to locks, but the actual move is decided by geometric shape matching under rotation between the free key component and the lock's carved black notch.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape

    lock_mask = g == LOCK
    lock_components = _components(lock_mask)
    lock_labels = np.full((h, w), -1, dtype=int)
    for idx, coords in enumerate(lock_components):
        lock_labels[coords[:, 0], coords[:, 1]] = idx

    markers: dict[int, tuple[int, int]] = {}
    keys: dict[int, np.ndarray] = {}
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in {BG, LOCK}):
        comps = _components(g == color)
        for comp in comps:
            if len(comp) == 1:
                markers[color] = tuple(map(int, comp[0]))
            else:
                keys[color] = comp

    for color, marker in markers.items():
        key_coords = keys.get(color)
        if key_coords is None:
            continue

        adjacent_locks = set()
        for nr, nc in _neighbors4(*marker):
            if 0 <= nr < h and 0 <= nc < w and lock_labels[nr, nc] >= 0:
                adjacent_locks.add(int(lock_labels[nr, nc]))
        if len(adjacent_locks) != 1:
            continue

        lock_coords = lock_components[next(iter(adjacent_locks))]
        r0, c0, r1, c1 = _bbox(lock_coords)
        lock_patch = g[r0 : r1 + 1, c0 : c1 + 1]
        hole_mask = lock_patch == BG
        if not hole_mask.any():
            continue

        notch_mask = _tight_mask(np.argwhere(hole_mask))
        key_mask = _tight_mask(key_coords)
        matched = any(np.array_equal(np.rot90(key_mask, k), notch_mask) for k in range(4))
        if not matched:
            continue

        out[key_coords[:, 0], key_coords[:, 1]] = BG
        target = out[r0 : r1 + 1, c0 : c1 + 1]
        target[hole_mask] = color

    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        n_pairs = random.randint(1, 3)
        colors = random.sample(KEY_COLORS, n_pairs)
        height = random.randint(16 + 3 * (n_pairs - 1), 30)
        width = random.randint(16 + 3 * (n_pairs - 1), 30)

        pairs: list[dict] = []
        try:
            for color in colors:
                canonical_slot = _random_slot_mask()
                side = random.choice(SIDES)
                slot_mask = np.rot90(canonical_slot, SIDE_TO_ROT[side])
                lock_in, lock_out = _build_lock_patches(color, side, slot_mask)
                matched = random.random() < 0.5
                if matched:
                    key_mask = np.rot90(slot_mask, random.randrange(4))
                else:
                    key_mask = _random_nonmatching_mask(slot_mask)
                pairs.append(
                    {
                        "color": color,
                        "matched": matched,
                        "lock_in": lock_in,
                        "lock_out": lock_out,
                        "key_patch": _key_patch(color, key_mask),
                    }
                )
        except RuntimeError:
            continue

        pairs = sorted(pairs, key=lambda p: p["lock_in"].shape[0] * p["lock_in"].shape[1], reverse=True)
        grid = np.full((height, width), BG, dtype=int)
        lock_positions: list[tuple[int, int]] = []
        okay = True
        for pair in pairs:
            positions = _foreground_bbox_positions(grid, pair["lock_in"], padding=1)
            if not positions:
                okay = False
                break
            pos = random.choice(positions)
            lock_positions.append(pos)
            _paste_patch(grid, pair["lock_in"], *pos)
        if not okay:
            continue

        key_positions: list[tuple[int, int]] = []
        for pair in pairs:
            positions = _foreground_bbox_positions(grid, pair["key_patch"], padding=1)
            if not positions:
                okay = False
                break
            pos = random.choice(positions)
            key_positions.append(pos)
            _paste_patch(grid, pair["key_patch"], *pos)
        if not okay:
            continue

        inp, expected = _expected_from_pairs((height, width), pairs, lock_positions, key_positions)
        if not np.array_equal(inp, grid):
            continue
        if not np.array_equal(main(inp), expected):
            continue
        return inp

    raise RuntimeError("generate_input failed to find a valid sample")
