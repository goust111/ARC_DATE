from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG_COLOR = 0
LOCK_COLOR = 5
KEY_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GRID_SIDE_MIN = 20
GRID_SIDE_MAX = 30
LOCK_COUNT_OPTIONS = (1, 2, 2, 3)
EXTRA_KEY_OPTIONS = (0, 1, 1, 2)
GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 300
SIGNATURE_ATTEMPTS = 400
ORTHOGONAL_PAD = 1


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[cnts.argmax()])


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            queue = deque([(r, c)])
            while queue:
                rr, cc = queue.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _component_records(grid: np.ndarray, color: int) -> list[dict]:
    mask = grid == int(color)
    comps = _components_from_mask(mask)
    records = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox_from_coords(comp)
        rel = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            rel[r - r0, c - c0] = True
        records.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": rel,
                "coords": comp,
                "color": int(color),
            }
        )
    return records


def _row_spans(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    h, _w = mask.shape
    lefts: list[int] = []
    rights: list[int] = []
    for r in range(h):
        cols = np.flatnonzero(mask[r])
        if cols.size == 0:
            return None
        if int(cols[-1]) - int(cols[0]) + 1 != int(cols.size):
            return None
        lefts.append(int(cols[0]))
        rights.append(int(cols[-1]))
    return np.array(lefts, dtype=int), np.array(rights, dtype=int)


def _col_spans(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    _h, w = mask.shape
    tops: list[int] = []
    bottoms: list[int] = []
    for c in range(w):
        rows = np.flatnonzero(mask[:, c])
        if rows.size == 0:
            return None
        if int(rows[-1]) - int(rows[0]) + 1 != int(rows.size):
            return None
        tops.append(int(rows[0]))
        bottoms.append(int(rows[-1]))
    return np.array(tops, dtype=int), np.array(bottoms, dtype=int)


def _positive_band(amounts: np.ndarray) -> tuple[int, tuple[int, ...]] | None:
    idx = np.flatnonzero(amounts > 0)
    if idx.size == 0 or idx.size == amounts.size:
        return None
    if int(idx[-1]) - int(idx[0]) + 1 != int(idx.size):
        return None
    return int(idx[0]), tuple(int(amounts[i]) for i in idx)


def _extract_key_signature(mask: np.ndarray) -> tuple[str, tuple[int, ...]] | None:
    h, w = mask.shape
    candidates: list[tuple[str, tuple[int, ...]]] = []

    row_spans = _row_spans(mask)
    if row_spans is not None:
        lefts, rights = row_spans
        if np.all(lefts == 0):
            base_right = int(rights.min())
            if base_right < w - 1:
                band = _positive_band(rights - base_right)
                if band is not None:
                    _start, profile = band
                    candidates.append(("right", profile))
        if np.all(rights == w - 1):
            base_left = int(lefts.max())
            if base_left > 0:
                band = _positive_band(base_left - lefts)
                if band is not None:
                    _start, profile = band
                    candidates.append(("left", profile))

    col_spans = _col_spans(mask)
    if col_spans is not None:
        tops, bottoms = col_spans
        if np.all(tops == 0):
            base_bottom = int(bottoms.min())
            if base_bottom < h - 1:
                band = _positive_band(bottoms - base_bottom)
                if band is not None:
                    _start, profile = band
                    candidates.append(("bottom", profile))
        if np.all(bottoms == h - 1):
            base_top = int(tops.max())
            if base_top > 0:
                band = _positive_band(base_top - tops)
                if band is not None:
                    _start, profile = band
                    candidates.append(("top", profile))

    unique = list(dict.fromkeys(candidates))
    if len(unique) != 1:
        return None
    return unique[0]


def _fill_lock_shape(lock_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = lock_mask.shape
    outside = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not lock_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not lock_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if not lock_mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                queue.append((nr, nc))

    holes = (~lock_mask) & (~outside)
    filled = lock_mask | holes
    return filled, holes


def _extract_lock_signature_and_notch(
    filled_mask: np.ndarray,
) -> tuple[tuple[str, tuple[int, ...]], np.ndarray] | None:
    h, w = filled_mask.shape
    candidates: list[tuple[tuple[str, tuple[int, ...]], np.ndarray]] = []

    row_spans = _row_spans(filled_mask)
    if row_spans is not None:
        lefts, rights = row_spans
        if np.all(lefts == 0):
            miss = (w - 1) - rights
            band = _positive_band(miss)
            if band is not None:
                _start, profile = band
                notch = np.zeros_like(filled_mask, dtype=bool)
                rows = np.flatnonzero(miss > 0)
                for r in rows:
                    notch[r, int(rights[r]) + 1 : w] = True
                candidates.append((("right", profile), notch))
        if np.all(rights == w - 1):
            miss = lefts
            band = _positive_band(miss)
            if band is not None:
                _start, profile = band
                notch = np.zeros_like(filled_mask, dtype=bool)
                rows = np.flatnonzero(miss > 0)
                for r in rows:
                    notch[r, : int(lefts[r])] = True
                candidates.append((("left", profile), notch))

    col_spans = _col_spans(filled_mask)
    if col_spans is not None:
        tops, bottoms = col_spans
        if np.all(tops == 0):
            miss = (h - 1) - bottoms
            band = _positive_band(miss)
            if band is not None:
                _start, profile = band
                notch = np.zeros_like(filled_mask, dtype=bool)
                cols = np.flatnonzero(miss > 0)
                for c in cols:
                    notch[int(bottoms[c]) + 1 : h, c] = True
                candidates.append((("bottom", profile), notch))
        if np.all(bottoms == h - 1):
            miss = tops
            band = _positive_band(miss)
            if band is not None:
                _start, profile = band
                notch = np.zeros_like(filled_mask, dtype=bool)
                cols = np.flatnonzero(miss > 0)
                for c in cols:
                    notch[: int(tops[c]), c] = True
                candidates.append((("top", profile), notch))

    if len(candidates) != 1:
        return None
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match small key tabs to complementary lock notches, then fill matched lock interiors with the key color.

    Input structure:
    - A black background contains large grey hollow lock outlines and several smaller solid colored keys.
    - Each lock is a single hollow outline whose filled outer silhouette is a rectangle with exactly one inward notch on one side.
    - Each key is a solid rectangle-like body with exactly one outward tab on one side; key colors are non-grey.

    Rule:
    - For every lock, recover its filled outer silhouette, extract the notch profile, and compare it to every key's tab profile.
    - If a key tab matches a lock notch exactly, erase that key, fill the lock's interior hole with the key color, and paint the notch area grey to seal it.
    - Erase all keys from the output; locks without a matching key keep their original hollow grey outline and open notch.

    Key insight:
    - The deciding feature is the local contour profile on one side of each object: the key's protruding tab transfers only its color, while the lock keeps its own grey outline geometry.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    out = g.copy()

    lock_records = []
    for record in _component_records(g, LOCK_COLOR):
        filled, holes = _fill_lock_shape(record["mask"])
        if not holes.any():
            continue
        extracted = _extract_lock_signature_and_notch(filled)
        if extracted is None:
            continue
        signature, notch = extracted
        record = dict(record)
        record["signature"] = signature
        record["hole_mask"] = holes
        record["notch_mask"] = notch
        lock_records.append(record)

    key_records = []
    next_key_id = 0
    for color in sorted(int(c) for c in np.unique(g) if int(c) not in (bg, LOCK_COLOR)):
        for record in _component_records(g, color):
            signature = _extract_key_signature(record["mask"])
            if signature is None:
                continue
            record = dict(record)
            record["signature"] = signature
            record["key_id"] = int(next_key_id)
            next_key_id += 1
            key_records.append(record)

    for key in key_records:
        for r, c in key["coords"]:
            out[r, c] = bg

    keys_by_signature: dict[tuple[str, tuple[int, ...]], list[dict]] = {}
    for key in key_records:
        keys_by_signature.setdefault(key["signature"], []).append(key)

    used_keys: set[int] = set()
    for lock in lock_records:
        candidates = keys_by_signature.get(lock["signature"], [])
        chosen = None
        for key in candidates:
            key_id = int(key["key_id"])
            if key_id not in used_keys:
                chosen = key
                break
        if chosen is None:
            continue
        key = chosen
        used_keys.add(int(key["key_id"]))
        r0, c0, r1, c1 = lock["bbox"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[lock["hole_mask"]] = int(key["color"])
        patch[lock["notch_mask"]] = LOCK_COLOR
        out[r0 : r1 + 1, c0 : c1 + 1] = patch

    return out


def _boundary_mask(filled: np.ndarray) -> np.ndarray:
    h, w = filled.shape
    boundary = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not filled[r, c]:
                continue
            if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                boundary[r, c] = True
                continue
            for nr, nc in _neighbors4(r, c, h, w):
                if not filled[nr, nc]:
                    boundary[r, c] = True
                    break
    return boundary


def _sample_signature(existing: set[tuple[str, tuple[int, ...]]]) -> tuple[str, tuple[int, ...]] | None:
    sides = ("left", "right", "top", "bottom")
    for _ in range(SIGNATURE_ATTEMPTS):
        side = random.choice(sides)
        band_len = random.randint(1, 3)
        profile = tuple(random.randint(1, 3) for _ in range(band_len))
        signature = (side, profile)
        if signature not in existing:
            return signature
    return None


def _carve_side(mask: np.ndarray, side: str, profile: tuple[int, ...], start: int) -> None:
    h, w = mask.shape
    if side == "right":
        for i, depth in enumerate(profile):
            mask[start + i, w - depth : w] = False
    elif side == "left":
        for i, depth in enumerate(profile):
            mask[start + i, :depth] = False
    elif side == "bottom":
        for i, depth in enumerate(profile):
            mask[h - depth : h, start + i] = False
    else:
        for i, depth in enumerate(profile):
            mask[:depth, start + i] = False


def _build_lock_object(signature: tuple[str, tuple[int, ...]]) -> dict | None:
    side, profile = signature
    band_len = len(profile)
    max_depth = max(profile)

    if side in ("left", "right"):
        outer_h = random.randint(max(6, band_len + 4), 10)
        outer_w = random.randint(max(6, max_depth + 4), 10)
        start = random.randint(1, outer_h - band_len - 1)
    else:
        outer_h = random.randint(max(6, max_depth + 4), 10)
        outer_w = random.randint(max(6, band_len + 4), 10)
        start = random.randint(1, outer_w - band_len - 1)

    filled = np.ones((outer_h, outer_w), dtype=bool)
    _carve_side(filled, side, profile, start)
    boundary = _boundary_mask(filled)
    hole = filled & (~boundary)
    if not hole.any():
        return None
    if len(_components_from_mask(boundary)) != 1:
        return None
    extracted = _extract_lock_signature_and_notch(filled)
    if extracted is None or extracted[0] != signature:
        return None

    sprite = np.full((outer_h, outer_w), BG_COLOR, dtype=int)
    sprite[boundary] = LOCK_COLOR
    return {
        "kind": "lock",
        "signature": signature,
        "sprite": sprite,
        "hole_mask": hole,
        "notch_mask": extracted[1],
    }


def _build_key_object(signature: tuple[str, tuple[int, ...]], color: int) -> dict | None:
    side, profile = signature
    band_len = len(profile)
    max_depth = max(profile)

    if side in ("left", "right"):
        body_h = random.randint(max(4, band_len + 2), 7)
        body_w = random.randint(3, 7)
        start = random.randint(1, body_h - band_len - 1)
        total_h = body_h
        total_w = body_w + max_depth
        mask = np.zeros((total_h, total_w), dtype=bool)
        if side == "right":
            mask[:, :body_w] = True
            for i, depth in enumerate(profile):
                mask[start + i, body_w : body_w + depth] = True
        else:
            mask[:, max_depth : max_depth + body_w] = True
            for i, depth in enumerate(profile):
                mask[start + i, max_depth - depth : max_depth] = True
    else:
        body_h = random.randint(3, 7)
        body_w = random.randint(max(4, band_len + 2), 7)
        start = random.randint(1, body_w - band_len - 1)
        total_h = body_h + max_depth
        total_w = body_w
        mask = np.zeros((total_h, total_w), dtype=bool)
        if side == "bottom":
            mask[:body_h, :] = True
            for i, depth in enumerate(profile):
                mask[body_h : body_h + depth, start + i] = True
        else:
            mask[max_depth : max_depth + body_h, :] = True
            for i, depth in enumerate(profile):
                mask[max_depth - depth : max_depth, start + i] = True

    if _extract_key_signature(mask) != signature:
        return None

    sprite = np.full(mask.shape, BG_COLOR, dtype=int)
    sprite[mask] = int(color)
    return {
        "kind": "key",
        "signature": signature,
        "sprite": sprite,
        "mask": mask,
        "color": int(color),
    }


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    hh, ww = occupied.shape
    r1 = r0 + h
    c1 = c0 + w
    rr0 = max(0, r0 - ORTHOGONAL_PAD)
    cc0 = max(0, c0 - ORTHOGONAL_PAD)
    rr1 = min(hh, r1 + ORTHOGONAL_PAD)
    cc1 = min(ww, c1 + ORTHOGONAL_PAD)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _paint_sprite(grid: np.ndarray, sprite: np.ndarray, top: int, left: int) -> None:
    h, w = sprite.shape
    block = grid[top : top + h, left : left + w]
    mask = sprite != BG_COLOR
    block[mask] = sprite[mask]
    grid[top : top + h, left : left + w] = block


def _place_objects(objects: list[dict], grid_h: int, grid_w: int) -> list[dict] | None:
    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placed: list[dict] = []

    order = sorted(objects, key=lambda obj: obj["sprite"].shape[0] * obj["sprite"].shape[1], reverse=True)
    for obj in order:
        sprite = obj["sprite"]
        h, w = sprite.shape
        success = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(0, grid_h - h)
            left = random.randint(0, grid_w - w)
            if not _can_place(occupied, top, left, h, w):
                continue
            _paint_sprite(grid, sprite, top, left)
            occupied[top : top + h, left : left + w] |= sprite != BG_COLOR
            placed_obj = dict(obj)
            placed_obj["top"] = int(top)
            placed_obj["left"] = int(left)
            placed_obj["grid_shape"] = (int(grid_h), int(grid_w))
            placed.append(placed_obj)
            success = True
            break
        if not success:
            return None

    return placed


def _expected_output(grid: np.ndarray, placed: list[dict], matched: dict[tuple[str, tuple[int, ...]], int]) -> np.ndarray:
    out = grid.copy()
    for obj in placed:
        top = obj["top"]
        left = obj["left"]
        if obj["kind"] == "key":
            h, w = obj["sprite"].shape
            patch = out[top : top + h, left : left + w]
            patch[obj["mask"]] = BG_COLOR
            out[top : top + h, left : left + w] = patch
        else:
            color = matched.get(obj["signature"])
            if color is None:
                continue
            h, w = obj["sprite"].shape
            patch = out[top : top + h, left : left + w]
            patch[obj["hole_mask"]] = int(color)
            patch[obj["notch_mask"]] = LOCK_COLOR
            out[top : top + h, left : left + w] = patch
    return out


def _grid_dimensions(objects: list[dict]) -> tuple[int, int] | None:
    padded_area = sum((obj["sprite"].shape[0] + 2) * (obj["sprite"].shape[1] + 2) for obj in objects)
    min_side = max(max(obj["sprite"].shape) for obj in objects)
    need_side = max(GRID_SIDE_MIN, min_side + 2, int(np.ceil(np.sqrt(padded_area * 1.5))))
    if need_side > GRID_SIDE_MAX:
        return None
    grid_h = random.randint(need_side, GRID_SIDE_MAX)
    grid_w = random.randint(need_side, GRID_SIDE_MAX)
    return int(grid_h), int(grid_w)


def _generate_scene() -> np.ndarray | None:
    num_locks = random.choice(LOCK_COUNT_OPTIONS)
    matched_count = random.randint(0, num_locks)
    extra_keys = random.choice(EXTRA_KEY_OPTIONS)
    if matched_count == 0:
        extra_keys = max(1, extra_keys)

    used_signatures: set[tuple[str, tuple[int, ...]]] = set()
    lock_objects: list[dict] = []
    for _ in range(num_locks):
        signature = _sample_signature(used_signatures)
        if signature is None:
            return None
        used_signatures.add(signature)
        lock = _build_lock_object(signature)
        if lock is None:
            return None
        lock_objects.append(lock)

    colors = random.sample(KEY_COLOR_POOL, matched_count + extra_keys)
    key_objects: list[dict] = []
    matched_signatures = random.sample([lock["signature"] for lock in lock_objects], matched_count)
    matched_map: dict[tuple[str, tuple[int, ...]], int] = {}

    for signature, color in zip(matched_signatures, colors[:matched_count], strict=False):
        key = _build_key_object(signature, color)
        if key is None:
            return None
        key_objects.append(key)
        matched_map[signature] = int(color)

    extra_used = set(used_signatures)
    for color in colors[matched_count:]:
        signature = _sample_signature(extra_used)
        if signature is None:
            return None
        extra_used.add(signature)
        key = _build_key_object(signature, color)
        if key is None:
            return None
        key_objects.append(key)

    objects = lock_objects + key_objects
    dims = _grid_dimensions(objects)
    if dims is None:
        return None
    grid_h, grid_w = dims
    placed = _place_objects(objects, grid_h, grid_w)
    if placed is None:
        return None

    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    for obj in placed:
        _paint_sprite(grid, obj["sprite"], obj["top"], obj["left"])

    expected = _expected_output(grid, placed, matched_map)
    actual = main(grid)
    if not np.array_equal(expected, actual):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lock-and-key scene")
