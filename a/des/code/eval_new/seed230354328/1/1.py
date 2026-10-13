from __future__ import annotations

import random
from collections import deque

import numpy as np


GEN_BG = 0
GEN_WALL = 5
KEY_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_MAX_RETRIES = 400


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_true(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    cropped = _crop_true(mask)
    return tuple(map(int, cropped.shape)), cropped.tobytes()


def _extract_room(
    grid: np.ndarray, coords: np.ndarray, wall_color: int, bg: int
) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    local = grid[r0 : r1 + 1, c0 : c1 + 1]
    if int(np.count_nonzero(local == wall_color)) != int(coords.shape[0]):
        return None

    h, w = map(int, local.shape)
    candidates = []
    for thickness in range(1, min(h, w) // 2 + 1):
        if h <= 2 * thickness or w <= 2 * thickness:
            continue

        border = np.ones((h, w), dtype=bool)
        border[thickness : h - thickness, thickness : w - thickness] = False

        if np.any(local[~border] != bg):
            continue

        border_vals = local[border]
        if np.any((border_vals != wall_color) & (border_vals != bg)):
            continue

        gap_mask = border & (local == bg)
        if not gap_mask.any():
            continue
        if len(_components(gap_mask)) != 1:
            continue

        gap_coords = np.argwhere(gap_mask)
        sides = []
        if np.all(gap_coords[:, 0] < thickness):
            sides.append("top")
        if np.all(gap_coords[:, 0] >= h - thickness):
            sides.append("bottom")
        if np.all(gap_coords[:, 1] < thickness):
            sides.append("left")
        if np.all(gap_coords[:, 1] >= w - thickness):
            sides.append("right")
        if len(sides) != 1:
            continue

        side = sides[0]
        if side == "top":
            if not (np.any(gap_coords[:, 0] == 0) and np.any(gap_coords[:, 0] == thickness - 1)):
                continue
        elif side == "bottom":
            if not (
                np.any(gap_coords[:, 0] == h - 1)
                and np.any(gap_coords[:, 0] == h - thickness)
            ):
                continue
        elif side == "left":
            if not (np.any(gap_coords[:, 1] == 0) and np.any(gap_coords[:, 1] == thickness - 1)):
                continue
        else:
            if not (
                np.any(gap_coords[:, 1] == w - 1)
                and np.any(gap_coords[:, 1] == w - thickness)
            ):
                continue

        candidates.append(
            {
                "bbox": (r0, c0, r1, c1),
                "thickness": int(thickness),
                "side": side,
                "gap_signature": _shape_signature(gap_mask),
            }
        )

    if len(candidates) != 1:
        return None
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Match each toothed key sprite to its wall gap and flood the room behind it with the key color.

    Input structure:
    - The grid has a dominant black background.
    - Several separate grey rectangular room frames contain black rectangular interiors.
    - Each room has one connected black lockhole carved through a single wall band.
    - Scattered monochrome key sprites elsewhere on the grid have the exact same shape as one room's lockhole, without any rotation.

    Rule:
    - Identify the grey room components, recover each room's wall thickness, and extract the connected lockhole mask from the missing wall cells.
    - Identify every non-grey key sprite and match it to the unique room whose lockhole has the same translated shape.
    - Erase the scattered keys from their original positions.
    - Fill the full interior rectangle of each matched room with the color of its key, while leaving the grey walls and the black lockhole gap unchanged.

    Key insight:
    - The colors only say what paint to use after unlocking; the actual pairing comes from exact geometric equality between each key mask and each missing wall mask.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _mode_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors, "expected at least one foreground color"
    wall_color = max(colors, key=lambda color: int(np.count_nonzero(g == color)))

    room_components = _components(g == wall_color)
    rooms = []
    for coords in room_components:
        room = _extract_room(g, coords, wall_color, bg)
        assert room is not None, "every wall component must be a valid room"
        rooms.append(room)
    assert rooms, "expected at least one room"

    key_mask = (g != bg) & (g != wall_color)
    key_components = _components(key_mask)
    assert len(key_components) == len(rooms), "room/key counts must match"

    key_colors_by_signature: dict[tuple[tuple[int, int], bytes], int] = {}
    for coords in key_components:
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        color = int(g[int(coords[0, 0]), int(coords[0, 1])])
        assert np.all(g[coords[:, 0], coords[:, 1]] == color), "each key must be monochrome"

        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        signature = _shape_signature(mask)
        assert signature not in key_colors_by_signature, "key shapes must be unique"
        key_colors_by_signature[signature] = color

    out = g.copy()
    out[key_mask] = bg

    used_signatures: set[tuple[tuple[int, int], bytes]] = set()
    for room in rooms:
        signature = room["gap_signature"]
        assert signature in key_colors_by_signature, "every lockhole must have a matching key"
        assert signature not in used_signatures, "lockhole shapes must be unique"
        used_signatures.add(signature)

        color = key_colors_by_signature[signature]
        r0, c0, r1, c1 = room["bbox"]
        thickness = int(room["thickness"])
        out[r0 + thickness : r1 - thickness + 1, c0 + thickness : c1 - thickness + 1] = color

    assert len(used_signatures) == len(key_colors_by_signature)
    return out


def _partition(total: int, parts: int) -> list[int]:
    pieces = [1] * parts
    for _ in range(total - parts):
        pieces[random.randrange(parts)] += 1
    random.shuffle(pieces)
    return pieces


def _random_profile(thickness: int, length: int) -> tuple[int, ...] | None:
    if thickness < 2 or length < 3:
        return None

    run_count = random.randint(2, min(4, length))
    run_lengths = _partition(length, run_count)
    depths = []
    for _ in range(run_count):
        choices = list(range(1, thickness + 1))
        if depths:
            choices = [value for value in choices if value != depths[-1]]
        depths.append(random.choice(choices))

    if max(depths) < thickness:
        depths[random.randrange(run_count)] = thickness
    if min(depths) == thickness:
        depths[random.randrange(run_count)] = random.randint(1, thickness - 1)
    if len(set(depths)) == 1:
        return None

    profile: list[int] = []
    for run_len, depth in zip(run_lengths, depths):
        profile.extend([depth] * run_len)
    if max(profile) < thickness or min(profile) == thickness:
        return None
    return tuple(profile)


def _build_room_patch(
    inner_h: int, inner_w: int, thickness: int, side: str, profile: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    h = inner_h + 2 * thickness
    w = inner_w + 2 * thickness
    patch = np.full((h, w), GEN_WALL, dtype=int)
    patch[thickness : h - thickness, thickness : w - thickness] = GEN_BG
    gap = np.zeros((h, w), dtype=bool)

    span = len(profile)
    side_len = w if side in ("top", "bottom") else h
    start = random.randint(thickness, side_len - span - thickness)

    if side == "top":
        for i, depth in enumerate(profile):
            patch[:depth, start + i] = GEN_BG
            gap[:depth, start + i] = True
    elif side == "bottom":
        for i, depth in enumerate(profile):
            patch[h - depth :, start + i] = GEN_BG
            gap[h - depth :, start + i] = True
    elif side == "left":
        for i, depth in enumerate(profile):
            patch[start + i, :depth] = GEN_BG
            gap[start + i, :depth] = True
    else:
        for i, depth in enumerate(profile):
            patch[start + i, w - depth :] = GEN_BG
            gap[start + i, w - depth :] = True

    return patch, gap


def _build_key_patch(gap_mask: np.ndarray, color: int) -> np.ndarray:
    key = np.full(gap_mask.shape, GEN_BG, dtype=int)
    key[gap_mask] = color
    return key


def _find_positions(occupied: np.ndarray, shape: tuple[int, int], padding: int) -> list[tuple[int, int]]:
    h, w = shape
    H, W = map(int, occupied.shape)
    positions = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            rr0 = max(0, r0 - padding)
            cc0 = max(0, c0 - padding)
            rr1 = min(H, r0 + h + padding)
            cc1 = min(W, c0 + w + padding)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            positions.append((r0, c0))
    return positions


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, shape: tuple[int, int]) -> None:
    h, w = shape
    occupied[r0 : r0 + h, c0 : c0 + w] = True


def _paste_non_bg(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    region = grid[r0 : r0 + h, c0 : c0 + w]
    mask = patch != GEN_BG
    region[mask] = patch[mask]


def _design_room_specs(room_count: int) -> list[dict[str, object]] | None:
    specs = []
    seen_shapes: set[tuple[tuple[int, int], bytes]] = set()
    for _ in range(room_count):
        for _attempt in range(40):
            thickness = random.choice([2, 3])
            inner_h = random.randint(3, 7)
            inner_w = random.randint(3, 7)
            side = random.choice(["top", "bottom", "left", "right"])
            side_len = inner_w + 2 * thickness if side in ("top", "bottom") else inner_h + 2 * thickness
            max_span = min(7, side_len - 2 * thickness)
            if max_span < 3:
                continue
            span = random.randint(3, max_span)
            profile = _random_profile(thickness, span)
            if profile is None:
                continue

            room_patch, gap_mask = _build_room_patch(inner_h, inner_w, thickness, side, profile)
            cropped_gap = _crop_true(gap_mask)
            signature = _shape_signature(cropped_gap)
            if signature in seen_shapes:
                continue

            specs.append(
                {
                    "room_patch": room_patch,
                    "gap_shape": cropped_gap,
                    "signature": signature,
                    "thickness": int(thickness),
                }
            )
            seen_shapes.add(signature)
            break
        else:
            return None
    return specs


def _expected_from_layout(
    grid: np.ndarray,
    placed_rooms: list[dict[str, object]],
    placed_keys: list[dict[str, object]],
) -> np.ndarray:
    expected = grid.copy()
    for key in placed_keys:
        r0, c0 = key["pos"]
        patch = key["patch"]
        mask = patch != GEN_BG
        expected[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = GEN_BG

    for room in placed_rooms:
        r0, c0 = room["pos"]
        patch = room["patch"]
        thickness = int(room["thickness"])
        color = int(room["color"])
        expected[
            r0 + thickness : r0 + patch.shape[0] - thickness,
            c0 + thickness : c0 + patch.shape[1] - thickness,
        ] = color
    return expected


def _gen_attempt() -> np.ndarray | None:
    room_count = random.randint(1, 3)
    room_specs = _design_room_specs(room_count)
    if room_specs is None:
        return None

    colors = random.sample(KEY_COLOR_POOL, room_count)
    for spec, color in zip(room_specs, colors):
        spec["color"] = int(color)
        spec["key_patch"] = _build_key_patch(spec["gap_shape"], int(color))

    all_patches = [spec["room_patch"] for spec in room_specs] + [spec["key_patch"] for spec in room_specs]
    max_h = max(int(patch.shape[0]) for patch in all_patches)
    max_w = max(int(patch.shape[1]) for patch in all_patches)
    reserve_area = sum(int((patch.shape[0] + 2) * (patch.shape[1] + 2)) for patch in all_patches)

    min_dim = max(
        12,
        max_h + 2,
        max_w + 2,
        int(np.ceil(np.sqrt(reserve_area))),
        10 + 4 * room_count,
    )
    if min_dim > 30:
        return None

    H = random.randint(min_dim, 30)
    W = random.randint(min_dim, 30)
    grid = np.full((H, W), GEN_BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    placed_rooms = []
    room_order = sorted(room_specs, key=lambda spec: -int(spec["room_patch"].shape[0] * spec["room_patch"].shape[1]))
    for spec in room_order:
        patch = spec["room_patch"]
        positions = _find_positions(occupied, patch.shape, padding=1)
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        _paste_non_bg(grid, patch, r0, c0)
        _mark_occupied(occupied, r0, c0, patch.shape)
        placed_rooms.append(
            {
                "patch": patch,
                "pos": (int(r0), int(c0)),
                "thickness": int(spec["thickness"]),
                "signature": spec["signature"],
                "color": int(spec["color"]),
            }
        )

    placed_keys = []
    key_specs = room_specs[:]
    random.shuffle(key_specs)
    for spec in key_specs:
        patch = spec["key_patch"]
        positions = _find_positions(occupied, patch.shape, padding=1)
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        _paste_non_bg(grid, patch, r0, c0)
        _mark_occupied(occupied, r0, c0, patch.shape)
        placed_keys.append(
            {
                "patch": patch,
                "pos": (int(r0), int(c0)),
                "signature": spec["signature"],
                "color": int(spec["color"]),
            }
        )

    expected = _expected_from_layout(grid, placed_rooms, placed_keys)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid key-lock ARC input")
