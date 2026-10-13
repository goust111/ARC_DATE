from __future__ import annotations

import random

import numpy as np


BG = 0
LOCK_COLOR = 5
KEY_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

PROFILE_LEN_RANGE = (4, 7)
PROFILE_DEPTH_RANGE = (0, 4)
LOCK_BODY_RANGE = (2, 4)
PAIR_COUNT_RANGE = (2, 4)
GRID_SIDE_RANGE = (20, 30)
PLACEMENT_PADDING = 1
MAX_GENERATION_ATTEMPTS = 400

_RNG = random.Random()


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r0 in range(h):
        for c0 in range(w):
            color = int(g[r0, c0])
            if color == bg or seen[r0, c0]:
                continue

            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in DIR4:
                    nr = r + dr
                    nc = c + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            r1 = max(rows)
            c1 = max(cols)
            r0b = min(rows)
            c0b = min(cols)
            mask = np.zeros((r1 - r0b + 1, c1 - c0b + 1), dtype=bool)
            for r, c in coords:
                mask[r - r0b, c - c0b] = True
            comps.append(
                {
                    "color": color,
                    "bbox": (r0b, c0b, r1, c1),
                    "mask": mask,
                }
            )
    return comps


def _top_lock_profile(mask: np.ndarray) -> tuple[int, ...] | None:
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    if h < 2 or w < 3:
        return None

    depths: list[int] = []
    for c in range(w):
        col = m[:, c]
        filled = np.flatnonzero(col)
        if len(filled) == 0:
            return None
        first = int(filled[0])
        if np.any(~col[first:]):
            return None
        depths.append(first)

    if not any(depth > 0 for depth in depths):
        return None
    if not any(depth == 0 for depth in depths):
        return None
    return tuple(depths)


def _top_key_profile(mask: np.ndarray) -> tuple[int, ...] | None:
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    if h < 2 or w < 3:
        return None

    depths: list[int] = []
    for c in range(w):
        col = m[:, c]
        if not bool(col[0]):
            return None
        filled = np.flatnonzero(col)
        last = int(filled[-1])
        if np.any(~col[: last + 1]):
            return None
        if np.any(col[last + 1 :]):
            return None
        depths.append(last)

    if not any(depth > 0 for depth in depths):
        return None
    return tuple(depths)


def _canonical_profile(
    mask: np.ndarray,
    extractor,
) -> tuple[int, ...] | None:
    hits: list[tuple[int, ...]] = []
    for turns in range(4):
        profile = extractor(np.rot90(mask, turns))
        if profile is not None:
            hits.append(profile)
    if len(hits) != 1:
        return None
    return hits[0]


def _parse_scene(grid: np.ndarray) -> tuple[int, list[dict[str, object]], list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    locks: list[dict[str, object]] = []
    keys: list[dict[str, object]] = []

    for comp in _find_components(g, bg):
        mask = np.asarray(comp["mask"], dtype=bool)
        color = int(comp["color"])
        if color == LOCK_COLOR:
            lock_profile = _canonical_profile(mask, _top_lock_profile)
            if lock_profile is None:
                raise AssertionError("every gray component must be a valid lock")
            locks.append(
                {
                    "bbox": comp["bbox"],
                    "profile": lock_profile,
                }
            )
        else:
            key_profile = _canonical_profile(mask, _top_key_profile)
            if key_profile is None:
                raise AssertionError("every non-gray component must be a valid key")
            keys.append(
                {
                    "color": color,
                    "profile": key_profile,
                }
            )

    if not locks or len(locks) != len(keys):
        raise AssertionError("scene must contain the same number of locks and keys")
    if len({lock["profile"] for lock in locks}) != len(locks):
        raise AssertionError("lock profiles must be unique")
    if len({key["profile"] for key in keys}) != len(keys):
        raise AssertionError("key profiles must be unique")

    return bg, locks, keys


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each key to the lock whose notch profile is its rotation-equivalent complement,
    erase the keys, and recolor each lock's full rectangle with the matched key color.
    """
    g = np.asarray(grid, dtype=int)
    bg, locks, keys = _parse_scene(g)
    color_by_profile = {key["profile"]: int(key["color"]) for key in keys}

    out = np.full_like(g, bg)
    for lock in locks:
        color = color_by_profile.get(lock["profile"])
        if color is None:
            raise AssertionError("every lock must have a matching key")
        r0, c0, r1, c1 = map(int, lock["bbox"])
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _is_monotone(seq: tuple[int, ...]) -> bool:
    nondecreasing = all(a <= b for a, b in zip(seq, seq[1:], strict=False))
    nonincreasing = all(a >= b for a, b in zip(seq, seq[1:], strict=False))
    return nondecreasing or nonincreasing


def _rotate_patch(patch: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(np.asarray(patch, dtype=int), int(turns))


def _build_lock_patch(profile: tuple[int, ...], body: int) -> np.ndarray:
    height = max(profile) + int(body)
    width = len(profile)
    patch = np.full((height, width), LOCK_COLOR, dtype=int)
    for c, depth in enumerate(profile):
        if depth > 0:
            patch[:depth, c] = BG
    return patch


def _build_key_patch(profile: tuple[int, ...], color: int) -> np.ndarray:
    height = max(profile) + 1
    width = len(profile)
    patch = np.full((height, width), BG, dtype=int)
    patch[0, :] = int(color)
    for c, depth in enumerate(profile):
        if depth > 0:
            patch[1 : depth + 1, c] = int(color)
    return patch


def _sample_profile(used: set[tuple[int, ...]]) -> tuple[int, ...] | None:
    for _ in range(600):
        width = _RNG.randint(*PROFILE_LEN_RANGE)
        inner = [_RNG.randint(*PROFILE_DEPTH_RANGE) for _ in range(width - 2)]
        profile = (0, *inner, 0)
        if sum(depth > 0 for depth in profile) < 2:
            continue
        if max(profile) < 2:
            continue
        if _is_monotone(profile):
            continue
        if profile in used or profile[::-1] in used:
            continue

        lock_mask = _build_lock_patch(profile, body=2) == LOCK_COLOR
        key_mask = _build_key_patch(profile, color=1) == 1
        if _canonical_profile(lock_mask, _top_lock_profile) != profile:
            continue
        if _canonical_profile(key_mask, _top_key_profile) != profile:
            continue
        return profile
    return None


def _paste_patch(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    region = canvas[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BG
    region[mask] = patch[mask]


def _can_place(reserved: np.ndarray, shape: tuple[int, int], r0: int, c0: int) -> bool:
    h, w = shape
    H, W = reserved.shape
    if r0 < 1 or c0 < 1 or r0 + h >= H or c0 + w >= W:
        return False
    rr0 = max(0, r0 - PLACEMENT_PADDING)
    rr1 = min(H, r0 + h + PLACEMENT_PADDING)
    cc0 = max(0, c0 - PLACEMENT_PADDING)
    cc1 = min(W, c0 + w + PLACEMENT_PADDING)
    return not bool(reserved[rr0:rr1, cc0:cc1].any())


def _reserve(reserved: np.ndarray, shape: tuple[int, int], r0: int, c0: int) -> None:
    h, w = shape
    H, W = reserved.shape
    rr0 = max(0, r0 - PLACEMENT_PADDING)
    rr1 = min(H, r0 + h + PLACEMENT_PADDING)
    cc0 = max(0, c0 - PLACEMENT_PADDING)
    cc1 = min(W, c0 + w + PLACEMENT_PADDING)
    reserved[rr0:rr1, cc0:cc1] = True


def _sample_canvas(objects: list[dict[str, object]]) -> tuple[int, int] | None:
    packed_area = sum((obj["patch"].shape[0] + 2) * (obj["patch"].shape[1] + 2) for obj in objects)
    tallest = max(int(obj["patch"].shape[0]) for obj in objects)
    widest = max(int(obj["patch"].shape[1]) for obj in objects)
    side_min = max(
        GRID_SIDE_RANGE[0],
        tallest + 3,
        widest + 3,
        int(np.ceil(np.sqrt(1.6 * packed_area))),
    )
    side_max = GRID_SIDE_RANGE[1]
    if side_min > side_max:
        return None
    return _RNG.randint(side_min, side_max), _RNG.randint(side_min, side_max)


def _place_objects(objects: list[dict[str, object]], height: int, width: int) -> np.ndarray | None:
    grid = np.full((height, width), BG, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    order = list(range(len(objects)))
    _RNG.shuffle(order)
    order.sort(
        key=lambda idx: int(objects[idx]["patch"].shape[0]) * int(objects[idx]["patch"].shape[1]),
        reverse=True,
    )

    for idx in order:
        patch = np.asarray(objects[idx]["patch"], dtype=int)
        h, w = patch.shape
        positions = [(r, c) for r in range(1, height - h) for c in range(1, width - w)]
        _RNG.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            if not _can_place(reserved, patch.shape, r0, c0):
                continue
            _paste_patch(grid, patch, r0, c0)
            _reserve(reserved, patch.shape, r0, c0)
            objects[idx]["pos"] = (r0, c0)
            placed = True
            break
        if not placed:
            return None
    return grid


def _expected_output(objects: list[dict[str, object]], shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for obj in objects:
        if obj["kind"] != "lock":
            continue
        patch = np.asarray(obj["patch"], dtype=int)
        r0, c0 = map(int, obj["pos"])
        h, w = patch.shape
        out[r0 : r0 + h, c0 : c0 + w] = int(obj["fill_color"])
    return out


def _generate_candidate() -> np.ndarray | None:
    pair_count = _RNG.randint(*PAIR_COUNT_RANGE)
    profiles: list[tuple[int, ...]] = []
    used: set[tuple[int, ...]] = set()
    for _ in range(pair_count):
        profile = _sample_profile(used)
        if profile is None:
            return None
        profiles.append(profile)
        used.add(profile)
        used.add(profile[::-1])

    colors = _RNG.sample(KEY_COLORS, pair_count)
    objects: list[dict[str, object]] = []
    for profile, color in zip(profiles, colors, strict=False):
        lock_patch = _build_lock_patch(profile, _RNG.randint(*LOCK_BODY_RANGE))
        key_patch = _build_key_patch(profile, color)
        objects.append(
            {
                "kind": "lock",
                "profile": profile,
                "fill_color": color,
                "patch": _rotate_patch(lock_patch, _RNG.randint(0, 3)),
            }
        )
        objects.append(
            {
                "kind": "key",
                "profile": profile,
                "patch": _rotate_patch(key_patch, _RNG.randint(0, 3)),
            }
        )

    canvas = _sample_canvas(objects)
    if canvas is None:
        return None
    height, width = canvas
    grid = _place_objects(objects, height, width)
    if grid is None:
        return None

    if _most_common_color(grid) != BG:
        return None

    expected = _expected_output(objects, grid.shape)
    try:
        parsed_bg, locks, keys = _parse_scene(grid)
    except AssertionError:
        return None
    if parsed_bg != BG or len(locks) != pair_count or len(keys) != pair_count:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid key-lock puzzle instance")
