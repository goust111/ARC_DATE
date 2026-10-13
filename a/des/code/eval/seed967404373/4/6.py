from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
LOCK = 5
KEY_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("left", "right", "top", "bottom")
OPPOSITE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = np.min(coords, axis=0)
    r1, c1 = np.max(coords, axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_patch(grid: np.ndarray, coords: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in coords:
        patch[int(r) - r0, int(c) - c0] = int(grid[int(r), int(c)])
    return (r0, c0, r1, c1), patch


def _detect_signature(mask: np.ndarray, is_lock: bool) -> tuple[str, int, tuple[int, ...]]:
    h, w = mask.shape
    candidates: list[tuple[str, int, tuple[int, ...]]] = []

    for side in ("left", "right"):
        for depth in range(1, w):
            body = mask[:, depth:] if side == "left" else mask[:, : w - depth]
            if body.size == 0 or not bool(np.all(body)):
                continue
            strip = mask[:, :depth] if side == "left" else mask[:, w - depth :]
            row_bits = strip[:, 0]
            if not bool(np.all(strip == row_bits[:, None])):
                continue
            if bool(np.all(row_bits)) or not bool(np.any(row_bits)):
                continue
            profile = row_bits.astype(int)
            if is_lock:
                profile = 1 - profile
            candidates.append((side, depth, tuple(int(x) for x in profile)))
            break

    for side in ("top", "bottom"):
        for depth in range(1, h):
            body = mask[depth:, :] if side == "top" else mask[: h - depth, :]
            if body.size == 0 or not bool(np.all(body)):
                continue
            strip = mask[:depth, :] if side == "top" else mask[h - depth :, :]
            col_bits = strip[0, :]
            if not bool(np.all(strip == col_bits[None, :])):
                continue
            if bool(np.all(col_bits)) or not bool(np.any(col_bits)):
                continue
            profile = col_bits.astype(int)
            if is_lock:
                profile = 1 - profile
            candidates.append((side, depth, tuple(int(x) for x in profile)))
            break

    assert len(candidates) == 1, f"expected exactly one toothed side, got {candidates}"
    return candidates[0]


def _analyze_component(
    grid: np.ndarray,
    coords: np.ndarray,
    is_lock: bool,
) -> dict[str, object]:
    bbox, patch = _extract_patch(grid, coords)
    mask = patch != BG
    side, depth, profile = _detect_signature(mask, is_lock=is_lock)
    return {
        "bbox": bbox,
        "patch": patch,
        "coords": coords,
        "side": side,
        "depth": int(depth),
        "profile": profile,
        "shape": patch.shape,
        "color": int(patch[mask][0]),
    }


def _target_top_left(key: dict[str, object], lock: dict[str, object]) -> tuple[int, int]:
    r0, c0, r1, c1 = lock["bbox"]  # type: ignore[misc]
    kh, kw = key["shape"]  # type: ignore[misc]
    depth = int(key["depth"])
    side = str(lock["side"])
    if side == "right":
        return int(r0), int(c1) - depth + 1
    if side == "left":
        return int(r0), int(c0) - (int(kw) - depth)
    if side == "bottom":
        return int(r1) - depth + 1, int(c0)
    return int(r0) - (int(kh) - depth), int(c0)


def _paste_non_bg(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    sub = grid[top : top + h, left : left + w]
    mask = patch != BG
    sub[mask] = patch[mask]
    grid[top : top + h, left : left + w] = sub


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored key to the grey lock whose notch profile is the exact complement,
    then translate the key onto that lock so the teeth fill the black cavities.
    """
    g = np.asarray(grid, dtype=int)

    locks = [_analyze_component(g, comp, is_lock=True) for comp in _components(g == LOCK)]
    key_colors = [int(c) for c in np.unique(g) if int(c) not in (BG, LOCK)]
    keys: list[dict[str, object]] = []
    for color in key_colors:
        comps = _components(g == int(color))
        assert len(comps) == 1, "each key color must form exactly one component"
        keys.append(_analyze_component(g, comps[0], is_lock=False))

    assert len(locks) == len(keys) and locks

    lock_map: dict[tuple[str, int, tuple[int, ...]], dict[str, object]] = {}
    for lock in locks:
        sig = (str(lock["side"]), int(lock["depth"]), lock["profile"])  # type: ignore[misc]
        assert sig not in lock_map, "lock signatures must be unique"
        lock_map[sig] = lock

    out = g.copy()
    for key in keys:
        for r, c in key["coords"]:  # type: ignore[misc]
            out[int(r), int(c)] = BG

    for key in keys:
        want = (OPPOSITE[str(key["side"])], int(key["depth"]), key["profile"])  # type: ignore[misc]
        assert want in lock_map, f"missing matching lock for {want}"
        lock = lock_map[want]
        top, left = _target_top_left(key, lock)
        _paste_non_bg(out, key["patch"], top, left)  # type: ignore[arg-type]

    return out


def _sample_profile() -> tuple[int, ...]:
    teeth = random.randint(2, 4)
    profile: list[int] = []
    for idx in range(teeth):
        profile.extend([0] * random.randint(1, 2))
        profile.extend([1] * random.randint(1, 3))
        if idx == teeth - 1:
            profile.extend([0] * random.randint(1, 2))
    return tuple(profile)


def _build_lock(side: str, profile: tuple[int, ...], depth: int, body: int) -> np.ndarray:
    if side in ("left", "right"):
        patch = np.full((len(profile), body + depth), LOCK, dtype=int)
        for r, bit in enumerate(profile):
            if bit:
                if side == "left":
                    patch[r, :depth] = BG
                else:
                    patch[r, body:] = BG
        return patch

    patch = np.full((body + depth, len(profile)), LOCK, dtype=int)
    for c, bit in enumerate(profile):
        if bit:
            if side == "top":
                patch[:depth, c] = BG
            else:
                patch[body:, c] = BG
    return patch


def _build_key(side: str, profile: tuple[int, ...], depth: int, body: int, color: int) -> np.ndarray:
    if side in ("left", "right"):
        patch = np.zeros((len(profile), body + depth), dtype=int)
        if side == "left":
            patch[:, depth:] = int(color)
            for r, bit in enumerate(profile):
                if bit:
                    patch[r, :depth] = int(color)
        else:
            patch[:, :body] = int(color)
            for r, bit in enumerate(profile):
                if bit:
                    patch[r, body:] = int(color)
        return patch

    patch = np.zeros((body + depth, len(profile)), dtype=int)
    if side == "top":
        patch[depth:, :] = int(color)
        for c, bit in enumerate(profile):
            if bit:
                patch[:depth, c] = int(color)
    else:
        patch[:body, :] = int(color)
        for c, bit in enumerate(profile):
            if bit:
                patch[body:, c] = int(color)
    return patch


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    grid[top : top + h, left : left + w] = np.where(patch != BG, patch, grid[top : top + h, left : left + w])


def _rect_free(blocked: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> bool:
    h, w = blocked.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, int(r0) - pad)
    cc0 = max(0, int(c0) - pad)
    rr1 = min(h - 1, int(r1) + pad)
    cc1 = min(w - 1, int(c1) + pad)
    return not bool(np.any(blocked[rr0 : rr1 + 1, cc0 : cc1 + 1]))


def _mark_rect(blocked: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> None:
    h, w = blocked.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, int(r0) - pad)
    cc0 = max(0, int(c0) - pad)
    rr1 = min(h - 1, int(r1) + pad)
    cc1 = min(w - 1, int(c1) + pad)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _make_pair(side: str, color: int) -> dict[str, object]:
    profile = _sample_profile()
    depth = random.randint(1, 3)
    lock_body = random.randint(3, 6)
    key_body = random.randint(2, 5)
    travel = depth + random.randint(1, 5)
    key_side = OPPOSITE[side]

    lock_patch = _build_lock(side, profile, depth, lock_body)
    key_patch = _build_key(key_side, profile, depth, key_body, color)

    lh, lw = lock_patch.shape
    kh, kw = key_patch.shape
    if side == "right":
        zone_h, zone_w = lh, lw + key_body + travel
        lock_pos = (0, 0)
        key_pos = (0, lw - depth + travel)
        target_pos = (0, lw - depth)
    elif side == "left":
        zone_h, zone_w = lh, lw + key_body + travel
        lock_pos = (0, key_body + travel)
        key_pos = (0, 0)
        target_pos = (0, travel)
    elif side == "bottom":
        zone_h, zone_w = lh + key_body + travel, lw
        lock_pos = (0, 0)
        key_pos = (lh - depth + travel, 0)
        target_pos = (lh - depth, 0)
    else:
        zone_h, zone_w = lh + key_body + travel, lw
        lock_pos = (key_body + travel, 0)
        key_pos = (0, 0)
        target_pos = (travel, 0)

    key_bbox = (
        int(key_pos[0]),
        int(key_pos[1]),
        int(key_pos[0]) + kh - 1,
        int(key_pos[1]) + kw - 1,
    )
    lock_bbox = (
        int(lock_pos[0]),
        int(lock_pos[1]),
        int(lock_pos[0]) + lh - 1,
        int(lock_pos[1]) + lw - 1,
    )
    target_bbox = (
        int(target_pos[0]),
        int(target_pos[1]),
        int(target_pos[0]) + kh - 1,
        int(target_pos[1]) + kw - 1,
    )

    return {
        "side": side,
        "depth": depth,
        "profile": profile,
        "color": int(color),
        "lock_patch": lock_patch,
        "key_patch": key_patch,
        "zone_shape": (int(zone_h), int(zone_w)),
        "lock_pos": (int(lock_pos[0]), int(lock_pos[1])),
        "key_pos": (int(key_pos[0]), int(key_pos[1])),
        "target_pos": (int(target_pos[0]), int(target_pos[1])),
        "lock_bbox": lock_bbox,
        "key_bbox": key_bbox,
        "target_bbox": target_bbox,
    }


def _build_expected(grid: np.ndarray, placements: list[dict[str, object]]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for info in placements:
        kr, kc = info["key_abs"]  # type: ignore[misc]
        kh, kw = info["key_patch"].shape  # type: ignore[misc]
        key_patch = info["key_patch"]  # type: ignore[assignment]
        mask = key_patch != BG
        out[kr : kr + kh, kc : kc + kw][mask] = BG
    for info in placements:
        tr, tc = info["target_abs"]  # type: ignore[misc]
        _place_patch(out, info["key_patch"], int(tr), int(tc))  # type: ignore[arg-type]
    return out


def generate_input() -> np.ndarray:
    for _ in range(400):
        pair_count = random.randint(2, 4)
        colors = random.sample(KEY_COLORS, pair_count)

        pairs: list[dict[str, object]] = []
        used_signatures: set[tuple[str, int, tuple[int, ...]]] = set()
        ok = True
        for color in colors:
            for _inner in range(80):
                side = random.choice(SIDES)
                pair = _make_pair(side, color)
                sig = (OPPOSITE[side], int(pair["depth"]), pair["profile"])  # key signature
                if sig not in used_signatures:
                    used_signatures.add(sig)
                    pairs.append(pair)
                    break
            else:
                ok = False
                break
        if not ok:
            continue

        zone_h_max = max(int(pair["zone_shape"][0]) for pair in pairs)
        zone_w_max = max(int(pair["zone_shape"][1]) for pair in pairs)
        total_zone_area = sum(int(pair["zone_shape"][0]) * int(pair["zone_shape"][1]) for pair in pairs)
        min_h = max(20, zone_h_max + 2)
        min_w = max(20, zone_w_max + 2)
        side_hint = int(total_zone_area ** 0.5) + 6
        min_h = max(min_h, side_hint - 2)
        min_w = max(min_w, side_hint - 2)
        if min_h > 40 or min_w > 40:
            continue
        h = random.randint(min_h, min(40, min_h + 10))
        w = random.randint(min_w, min(40, min_w + 10))

        grid = np.zeros((h, w), dtype=int)
        blocked = np.zeros((h, w), dtype=bool)
        placements: list[dict[str, object]] = []

        zone_order = sorted(
            pairs,
            key=lambda pair: int(pair["zone_shape"][0]) * int(pair["zone_shape"][1]),
            reverse=True,
        )

        for pair in zone_order:
            zh, zw = pair["zone_shape"]  # type: ignore[misc]
            if zh > h or zw > w:
                ok = False
                break
            placed = False
            for _inner in range(200):
                top = random.randint(1, h - int(zh) - 1)
                left = random.randint(1, w - int(zw) - 1)
                zone_bbox = (top, left, top + int(zh) - 1, left + int(zw) - 1)
                if not _rect_free(blocked, zone_bbox, pad=1):
                    continue
                _mark_rect(blocked, zone_bbox, pad=1)

                lr, lc = pair["lock_pos"]  # type: ignore[misc]
                kr, kc = pair["key_pos"]  # type: ignore[misc]
                tr, tc = pair["target_pos"]  # type: ignore[misc]
                lock_abs = (top + int(lr), left + int(lc))
                key_abs = (top + int(kr), left + int(kc))
                target_abs = (top + int(tr), left + int(tc))

                _place_patch(grid, pair["lock_patch"], lock_abs[0], lock_abs[1])  # type: ignore[arg-type]
                _place_patch(grid, pair["key_patch"], key_abs[0], key_abs[1])  # type: ignore[arg-type]

                placements.append(
                    {
                        "lock_patch": pair["lock_patch"],
                        "key_patch": pair["key_patch"],
                        "lock_abs": lock_abs,
                        "key_abs": key_abs,
                        "target_abs": target_abs,
                    }
                )
                placed = True
                break
            if not placed:
                ok = False
                break

        if not ok:
            continue

        expected = _build_expected(grid, placements)
        actual = main(grid)
        if np.array_equal(actual, expected):
            return grid

    raise RuntimeError("failed to generate a valid key-lock puzzle")
