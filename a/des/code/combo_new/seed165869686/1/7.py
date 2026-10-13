from __future__ import annotations

import random
import numpy as np


BG = 0
RED = 2
GREY = 5
MAROON = 9

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

KEY_COLORS = (1, 3, 4, 6, 7, 8)
PAYLOAD_COLORS = (1, 3, 4, 6, 7, 8)
SHAPE_LIBRARY = (
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
)

GEN_RETRIES = 500


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
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
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS.values():
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _trim_patch(patch: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch != 0)
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=int)
    sub = patch[r0 : r1 + 1, c0 : c1 + 1]
    out[sub != 0] = sub[sub != 0]
    return out


def _border_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _shape_cells(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    cells = np.argwhere(mask)
    return [(int(top + r), int(left + c)) for r, c in cells]


def _patch_cells(patch: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    cells = np.argwhere(patch != 0)
    return [(int(top + r), int(left + c)) for r, c in cells]


def _rect_cells(top: int, left: int, height: int, width: int) -> list[tuple[int, int]]:
    return [
        (int(r), int(c))
        for r in range(int(top), int(top + height))
        for c in range(int(left), int(left + width))
    ]


def _fits_cells(blocked: np.ndarray, cells: list[tuple[int, int]], *, margin: int = 0) -> bool:
    h, w = blocked.shape
    for r, c in cells:
        if not (int(margin) <= r < h - int(margin) and int(margin) <= c < w - int(margin)):
            return False
        if blocked[r, c]:
            return False
    return True


def _reserve(blocked: np.ndarray, cells: list[tuple[int, int]], *, pad: int) -> None:
    h, w = blocked.shape
    for r, c in cells:
        r0 = max(0, int(r - pad))
        r1 = min(h, int(r + pad + 1))
        c0 = max(0, int(c - pad))
        c1 = min(w, int(c + pad + 1))
        blocked[r0:r1, c0:c1] = True


def _rot_asym(arr: np.ndarray) -> bool:
    for k in (1, 2, 3):
        rot = np.rot90(arr, k)
        if arr.shape == rot.shape and np.array_equal(arr, rot):
            return False
    return True


def _find_chests(grid: np.ndarray) -> list[dict]:
    chests: list[dict] = []
    for coords in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(patch == BG):
            continue
        if np.any((patch == RED) | (patch == MAROON)):
            continue
        border = _border_mask(patch.shape)
        if not bool(np.all(patch[border] == GREY)):
            continue
        content_mask = patch != GREY
        area = int(content_mask.sum())
        if not (2 <= area <= 6):
            continue
        content = np.zeros_like(patch)
        content[content_mask] = patch[content_mask]
        chests.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "content": _trim_patch(content),
                "lock_mask": None,
            }
        )
    assert chests, "expected at least one chest"
    return chests


def _attach_lockholes(grid: np.ndarray, chests: list[dict]) -> None:
    maroon_comps = [coords for coords in _components(grid == MAROON) if 2 <= len(coords) <= 4]
    assert len(maroon_comps) == len(chests), "expected one lockhole per chest"

    for coords in maroon_comps:
        rr0, cc0, rr1, cc1 = _bbox(coords)
        matches: list[int] = []
        for idx, chest in enumerate(chests):
            r0, c0, r1, c1 = chest["bbox"]
            side_match = (
                (cc1 == c0 - 1 and not (rr1 < r0 or rr0 > r1))
                or (cc0 == c1 + 1 and not (rr1 < r0 or rr0 > r1))
                or (rr1 == r0 - 1 and not (cc1 < c0 or cc0 > c1))
                or (rr0 == r1 + 1 and not (cc1 < c0 or cc0 > c1))
            )
            if side_match:
                matches.append(int(idx))
        assert len(matches) == 1, "each lockhole must belong to exactly one chest"
        idx = matches[0]
        assert chests[idx]["lock_mask"] is None, "each chest should have a single lockhole"
        mask = np.zeros((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), dtype=bool)
        for r, c in coords:
            mask[int(r - rr0), int(c - cc0)] = True
        chests[idx]["lock_mask"] = _trim_mask(mask)

    for chest in chests:
        assert chest["lock_mask"] is not None, "missing lockhole"


def _find_keys(grid: np.ndarray, chests: list[dict]) -> list[dict]:
    inside_chest = np.zeros(grid.shape, dtype=bool)
    for chest in chests:
        r0, c0, r1, c1 = chest["bbox"]
        inside_chest[r0 : r1 + 1, c0 : c1 + 1] = True

    payload_mask = (
        (grid != BG)
        & (grid != RED)
        & (grid != GREY)
        & (grid != MAROON)
        & (~inside_chest)
    )
    red_pixels = {tuple(map(int, p)) for p in np.argwhere(grid == RED)}
    used_markers: set[tuple[int, int]] = set()
    keys: list[dict] = []

    for coords in _components(payload_mask):
        area = int(len(coords))
        if not (2 <= area <= 4):
            continue
        r0, c0, r1, c1 = _bbox(coords)
        patch = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=int)
        for r, c in coords:
            patch[int(r - r0), int(c - c0)] = int(grid[r, c])

        side_hits = []
        probes = {
            "up": [(int(r0 - 1), int(c)) for c in range(int(c0), int(c1 + 1))],
            "down": [(int(r1 + 1), int(c)) for c in range(int(c0), int(c1 + 1))],
            "left": [(int(r), int(c0 - 1)) for r in range(int(r0), int(r1 + 1))],
            "right": [(int(r), int(c1 + 1)) for r in range(int(r0), int(r1 + 1))],
        }
        for side, cells in probes.items():
            hits = [cell for cell in cells if cell in red_pixels]
            if hits:
                side_hits.append((side, hits))

        assert len(side_hits) == 1, "each key must have one directional red marker"
        direction, hits = side_hits[0]
        assert len(hits) == 1, "marker must be a single red pixel"
        marker = hits[0]
        assert marker not in used_markers, "markers must be unique"
        used_markers.add(marker)

        keys.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "area": int(area),
                "patch": patch,
                "shape": patch != 0,
                "marker": marker,
                "direction": direction,
                "coords": [(int(r), int(c)) for r, c in coords],
            }
        )

    assert len(keys) == len(chests), "expected one key for each chest"
    return keys


def _match_keys(keys: list[dict], chests: list[dict]) -> list[tuple[dict, dict, int]]:
    matches: list[tuple[dict, dict, int]] = []
    used_chests: set[int] = set()

    for key in keys:
        candidates: list[tuple[int, int]] = []
        for idx, chest in enumerate(chests):
            for rot_k in range(4):
                key_shape = _trim_mask(np.rot90(key["shape"], rot_k))
                if key_shape.shape == chest["lock_mask"].shape and np.array_equal(key_shape, chest["lock_mask"]):
                    candidates.append((int(idx), int(rot_k)))
        assert len(candidates) == 1, "each key must match exactly one lockhole rotation"
        chest_idx, rot_k = candidates[0]
        assert chest_idx not in used_chests, "lockhole matches must be one-to-one"
        used_chests.add(int(chest_idx))
        matches.append((key, chests[chest_idx], int(rot_k)))

    return matches


def _paste_nonzero(out: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    sub = out[int(top) : int(top + h), int(left) : int(left + w)]
    assert sub.shape == patch.shape, "payload placement must stay inside the grid"
    mask = patch != 0
    assert not bool(np.any((sub != BG) & mask)), "released content must not overlap other objects"
    sub[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each key shape to a chest lockhole, then release the chest payload near that key.

    Input structure:
    - The background is black.
    - Grey filled rectangles are chests; each chest contains a small colored payload inside.
    - A maroon lockhole sprite of 2-4 pixels is attached to one outside side of each chest.
    - Elsewhere, isolated small colored key sprites of 2-4 pixels each have a nearby single red marker on one side.

    Rule:
    - Pair every key with the unique lockhole whose shape matches after a rotation of the key.
    - Copy the matching chest's payload, rotate that payload by the same amount used in the key-lock match, and place it with its top-left aligned to the key bbox.
    - Then translate the placed payload in the direction indicated by the red marker by a distance equal to the key's pixel count.
    - Remove the original key pixels and red marker; keep every chest, payload-in-chest, and lockhole unchanged.

    Key insight:
    Shape matching chooses both the correct chest and the payload rotation, while the red marker encodes the release direction and the key size supplies the translation distance.
    """
    g = np.asarray(grid, dtype=int)
    chests = _find_chests(g)
    _attach_lockholes(g, chests)
    keys = _find_keys(g, chests)
    matched = _match_keys(keys, chests)

    out = g.copy()
    for key in keys:
        for r, c in key["coords"]:
            out[int(r), int(c)] = BG
        mr, mc = key["marker"]
        out[int(mr), int(mc)] = BG

    for key, chest, rot_k in matched:
        payload = np.rot90(chest["content"], rot_k)
        dr, dc = DIRS[key["direction"]]
        r0, c0, _r1, _c1 = key["bbox"]
        top = int(r0 + dr * key["area"])
        left = int(c0 + dc * key["area"])
        _paste_nonzero(out, payload, top, left)

    return out


def _random_connected_mask(max_h: int, max_w: int) -> np.ndarray | None:
    max_h = int(min(max_h, 4))
    max_w = int(min(max_w, 4))
    if max_h < 2 or max_w < 2:
        return None

    for _ in range(100):
        h = random.randint(2, max_h)
        w = random.randint(2, max_w)
        max_area = min(6, h * w)
        if max_area < 3:
            continue
        target = random.randint(3, max_area)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier = {
            (start[0] - 1, start[1]),
            (start[0] + 1, start[1]),
            (start[0], start[1] - 1),
            (start[0], start[1] + 1),
        }
        while len(cells) < target and frontier:
            candidates = [
                p
                for p in frontier
                if 0 <= p[0] < h and 0 <= p[1] < w and p not in cells
            ]
            if not candidates:
                break
            cell = random.choice(candidates)
            frontier.discard(cell)
            cells.add(cell)
            frontier.update(
                {
                    (cell[0] - 1, cell[1]),
                    (cell[0] + 1, cell[1]),
                    (cell[0], cell[1] - 1),
                    (cell[0], cell[1] + 1),
                }
            )
        if len(cells) != target:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[int(r), int(c)] = True
        mask = _trim_mask(mask)
        if mask.shape[0] > max_h or mask.shape[1] > max_w:
            continue
        if not _rot_asym(mask):
            continue
        return mask
    return None


def _random_payload(max_h: int, max_w: int) -> np.ndarray | None:
    for _ in range(100):
        mask = _random_connected_mask(int(max_h), int(max_w))
        if mask is None:
            continue
        coords = [tuple(map(int, p)) for p in np.argwhere(mask)]
        random.shuffle(coords)
        n_colors = random.randint(1, min(3, len(coords)))
        chosen = random.sample(PAYLOAD_COLORS, n_colors)
        patch = np.zeros(mask.shape, dtype=int)
        for idx, (r, c) in enumerate(coords):
            if idx < len(chosen):
                color = chosen[idx]
            else:
                color = random.choice(chosen)
            patch[int(r), int(c)] = int(color)
        if not _rot_asym(patch):
            continue
        return patch
    return None


def _place_chest(grid: np.ndarray, blocked: np.ndarray, base_shape: np.ndarray) -> dict | None:
    h, w = grid.shape
    for _ in range(160):
        lock_rot = random.randrange(4)
        lock_mask = np.rot90(base_shape, lock_rot)
        lock_h, lock_w = map(int, lock_mask.shape)
        side = random.choice(tuple(DIRS))

        chest_h_min = max(5, lock_h if side in ("left", "right") else 5)
        chest_w_min = max(5, lock_w if side in ("up", "down") else 5)
        if chest_h_min > 8 or chest_w_min > 8:
            continue
        chest_h = random.randint(chest_h_min, 8)
        chest_w = random.randint(chest_w_min, 8)

        payload = _random_payload(chest_h - 2, chest_w - 2)
        if payload is None:
            continue
        pay_h, pay_w = map(int, payload.shape)
        if pay_h > chest_h - 2 or pay_w > chest_w - 2:
            continue

        chest_r = random.randint(1, h - chest_h - 2)
        chest_c = random.randint(1, w - chest_w - 2)
        pay_r = random.randint(1, chest_h - pay_h - 1)
        pay_c = random.randint(1, chest_w - pay_w - 1)

        if side in ("left", "right"):
            lock_r = chest_r + random.randint(0, chest_h - lock_h)
            lock_c = chest_c - lock_w if side == "left" else chest_c + chest_w
        else:
            lock_r = chest_r - lock_h if side == "up" else chest_r + chest_h
            lock_c = chest_c + random.randint(0, chest_w - lock_w)

        chest_cells = _rect_cells(chest_r, chest_c, chest_h, chest_w)
        lock_cells = _shape_cells(lock_mask, lock_r, lock_c)
        entity_cells = chest_cells + lock_cells
        if not _fits_cells(blocked, entity_cells, margin=1):
            continue

        grid[chest_r : chest_r + chest_h, chest_c : chest_c + chest_w] = GREY
        sub = grid[chest_r + pay_r : chest_r + pay_r + pay_h, chest_c + pay_c : chest_c + pay_c + pay_w]
        mask = payload != 0
        sub[mask] = payload[mask]
        for r, c in lock_cells:
            grid[int(r), int(c)] = MAROON
        _reserve(blocked, entity_cells, pad=1)

        return {
            "shape": np.asarray(base_shape, dtype=bool),
            "lock_rot": int(lock_rot),
            "payload": payload,
        }
    return None


def _edge_marker_for_key(key_mask: np.ndarray, direction: str, top: int, left: int) -> tuple[int, int]:
    if direction == "up":
        cols = [int(c) for c in np.flatnonzero(key_mask[0])]
        return int(top - 1), int(left + random.choice(cols))
    if direction == "down":
        cols = [int(c) for c in np.flatnonzero(key_mask[-1])]
        return int(top + key_mask.shape[0]), int(left + random.choice(cols))
    if direction == "left":
        rows = [int(r) for r in np.flatnonzero(key_mask[:, 0])]
        return int(top + random.choice(rows)), int(left - 1)
    rows = [int(r) for r in np.flatnonzero(key_mask[:, -1])]
    return int(top + random.choice(rows)), int(left + key_mask.shape[1])


def _place_key(
    grid: np.ndarray,
    blocked: np.ndarray,
    pair: dict,
    key_color: int,
) -> bool:
    h, w = grid.shape
    for _ in range(220):
        key_rot = random.randrange(4)
        key_mask = np.rot90(pair["shape"], key_rot)
        direction = random.choice(tuple(DIRS))
        marker = None

        key_h, key_w = map(int, key_mask.shape)
        key_r = random.randint(1, h - key_h - 2)
        key_c = random.randint(1, w - key_w - 2)
        marker = _edge_marker_for_key(key_mask, direction, key_r, key_c)

        rot_k = (int(pair["lock_rot"]) - int(key_rot)) % 4
        released = np.rot90(pair["payload"], rot_k)
        distance = int(key_mask.sum())
        dr, dc = DIRS[direction]
        release_r = int(key_r + dr * distance)
        release_c = int(key_c + dc * distance)

        key_cells = _shape_cells(key_mask, key_r, key_c)
        release_cells = _patch_cells(released, release_r, release_c)
        if marker in set(key_cells) or marker in set(release_cells):
            continue
        candidate_cells = key_cells + [marker] + release_cells
        if len(set(candidate_cells)) != len(candidate_cells):
            continue
        if not _fits_cells(blocked, candidate_cells, margin=1):
            continue

        for r, c in key_cells:
            grid[int(r), int(c)] = int(key_color)
        mr, mc = marker
        grid[int(mr), int(mc)] = RED
        _reserve(blocked, candidate_cells, pad=1)
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.randint(1, len(SHAPE_LIBRARY))
    min_side = 14 + 4 * (n_pairs - 1)
    h = random.randint(min_side, 30)
    w = random.randint(min_side, 30)

    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    pairs: list[dict] = []

    for base_shape in random.sample(SHAPE_LIBRARY, n_pairs):
        pair = _place_chest(grid, blocked, base_shape)
        if pair is None:
            return None
        pairs.append(pair)

    for pair, key_color in zip(pairs, random.sample(KEY_COLORS, n_pairs)):
        if not _place_key(grid, blocked, pair, key_color):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    assert out.shape == grid.shape
    assert max(grid.shape) <= 30 and max(out.shape) <= 30
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
