from __future__ import annotations

import random

import numpy as np


BG = 0
SEP = 5
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for dr, dc in NEIGHBORS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rs = [int(r) for r, _ in coords]
    cs = [int(c) for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _normalize_coords(
    coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]
) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _coords_dims(coords: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return max(rs) + 1, max(cs) + 1


def _rotate_coords(
    coords: list[tuple[int, int]] | tuple[tuple[int, int], ...], k: int
) -> tuple[tuple[int, int], ...]:
    out = _normalize_coords(coords)
    for _ in range(k % 4):
        h, _w = _coords_dims(out)
        out = _normalize_coords([(c, h - 1 - r) for r, c in out])
    return out


def _flip_coords(
    coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]
) -> tuple[tuple[int, int], ...]:
    out = _normalize_coords(coords)
    _h, w = _coords_dims(out)
    return _normalize_coords([(r, w - 1 - c) for r, c in out])


def _coords_to_mask(coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    h, w = _coords_dims(coords)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    return mask


def _rotate_square_coords(
    coords: list[tuple[int, int]] | tuple[tuple[int, int], ...], size: int, k: int
) -> tuple[tuple[int, int], ...]:
    out = [(int(r), int(c)) for r, c in coords]
    for _ in range(k % 4):
        out = [(size - 1 - c, r) for r, c in out]
    return tuple(sorted(out))


def _touching_sides(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...], size: int) -> set[str]:
    sides = set()
    for r, c in coords:
        if r == 0:
            sides.add("top")
        if r == size - 1:
            sides.add("bottom")
        if c == 0:
            sides.add("left")
        if c == size - 1:
            sides.add("right")
    return sides


def _find_unique_rotation(
    key_shape: tuple[tuple[int, int], ...], notch_shape: tuple[tuple[int, int], ...]
) -> int:
    matches: set[int] = set()
    for flipped in (False, True):
        variant = _flip_coords(key_shape) if flipped else key_shape
        for k in range(4):
            if _rotate_coords(variant, k) == notch_shape:
                matches.add(int(k))
    assert len(matches) == 1, "key orientation must determine a unique rotation angle"
    return next(iter(matches))


def _parse_locks(top: np.ndarray) -> dict[int, dict]:
    locks: dict[int, dict] = {}
    for comp in _components(top != BG):
        color_values = {int(top[r, c]) for r, c in comp}
        assert len(color_values) == 1, "each lock must be monochrome"
        color = color_values.pop()
        assert color not in {BG, SEP}
        r0, c0, r1, c1 = _bbox(comp)
        patch = top[r0 : r1 + 1, c0 : c1 + 1]
        h, w = map(int, patch.shape)
        assert h == w, "locks are expected to use square bounding boxes"
        assert set(map(int, np.unique(patch))).issubset({BG, color})
        lock_mask = patch == color
        notch_cells = [(r, c) for r in range(h) for c in range(w) if not bool(lock_mask[r, c])]
        assert notch_cells, "each lock must contain a notch"
        assert len(_components(_coords_to_mask(_normalize_coords(notch_cells)))) == 1
        assert len(_touching_sides(notch_cells, h)) == 1, "a notch must open on exactly one side"
        assert color not in locks, "lock colors must be unique"
        locks[color] = {
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "size": int(h),
            "mask": lock_mask.copy(),
            "notch_cells": tuple((int(r), int(c)) for r, c in notch_cells),
            "notch_shape": _normalize_coords(notch_cells),
        }
    assert locks, "expected at least one lock"
    return locks


def _parse_keys(bottom: np.ndarray) -> dict[int, dict]:
    keys: dict[int, dict] = {}
    for comp in _components(bottom != BG):
        counts: dict[int, list[tuple[int, int]]] = {}
        for r, c in comp:
            counts.setdefault(int(bottom[r, c]), []).append((int(r), int(c)))
        assert len(counts) == 2, "each key must use exactly two colors"
        handle_candidates = [color for color, cells in counts.items() if len(cells) == 1]
        assert len(handle_candidates) == 1, "each key must have one singleton handle pixel"
        handle_color = int(handle_candidates[0])
        body_color = next(color for color in counts if color != handle_color)
        body_cells = counts[body_color]
        handle_cell = counts[handle_color][0]
        assert len(body_cells) >= 2
        assert any(abs(handle_cell[0] - r) + abs(handle_cell[1] - c) == 1 for r, c in body_cells)
        assert handle_color not in keys, "key handles must map one-to-one to locks"
        keys[handle_color] = {
            "body_color": int(body_color),
            "body_shape": _normalize_coords(body_cells),
        }
    assert keys, "expected at least one key"
    return keys


def main(grid: np.ndarray) -> np.ndarray:
    """Pair each colored key with its matching lock, infer the insertion rotation from the notch fit, and rotate the completed lock by that same angle.

    Input structure:
    - The background is black, and a single full-width grey row separates an upper lock region from a lower key region.
    - The upper region contains several large single-color locks; each lock is a nearly solid square with one connected notch cut from exactly one side.
    - The lower region contains several small two-color keys; each key has one protrusion shape plus a single-pixel handle whose color matches one lock.

    Rule:
    - Use the handle color to pair each key with the lock of the same color.
    - Compare the key's protrusion with the lock's missing notch and find the unique rotation that makes the protrusion fit the notch.
    - Fill the notch with the protrusion color, discard the handle, and rotate the entire completed lock by that same angle.
    - Return only the rotated locks on a black canvas that keeps their original relative positions from the upper region.

    Key insight:
    - The key is not copied into the lower area; instead, the rotation needed to insert the protrusion is reused as the final rotation instruction for the whole lock.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    sep_rows = np.flatnonzero(np.all(g == SEP, axis=1))
    assert len(sep_rows) == 1, "expected one grey separator row"
    sep = int(sep_rows[0])
    assert 0 < sep < g.shape[0] - 1

    top = g[:sep, :]
    bottom = g[sep + 1 :, :]

    locks = _parse_locks(top)
    keys = _parse_keys(bottom)
    assert set(locks) == set(keys), "key handles and lock colors must match exactly"

    out = np.full(top.shape, BG, dtype=int)
    for lock_color, lock in locks.items():
        key = keys[int(lock_color)]
        rotation_k = _find_unique_rotation(key["body_shape"], lock["notch_shape"])

        patch = np.full((lock["size"], lock["size"]), BG, dtype=int)
        patch[lock["mask"]] = int(lock_color)
        for r, c in lock["notch_cells"]:
            patch[int(r), int(c)] = int(key["body_color"])
        rotated = np.rot90(patch, int(rotation_k))

        r0, c0, r1, c1 = lock["bbox"]
        target = out[r0 : r1 + 1, c0 : c1 + 1]
        assert target.shape == rotated.shape and not bool(np.any(target != BG))
        target[:, :] = rotated

    return out


# --------------- generator ---------------

LOCK_COUNT_CHOICES = (2, 3, 4)
LOCK_SIZE_RANGE = (6, 8)
LOCK_CANVAS_W_LIMIT = 26
LOCK_CANVAS_H_LIMIT = 21
KEY_CANVAS_H_LIMIT = 8
LOCK_GAP_RANGE = (1, 3)
KEY_GAP_RANGE = (2, 4)
ROW_GAP_RANGE = (1, 3)
MARGIN_MAX = 2
GEN_RETRIES = 400

RAW_NOTCH_LIBRARY = [
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0)),
    ((0, 0), (0, 1), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
]


def _has_four_rotations(shape: tuple[tuple[int, int], ...]) -> bool:
    return len({_rotate_coords(shape, k) for k in range(4)}) == 4


NOTCH_LIBRARY = [shape for shape in map(_normalize_coords, RAW_NOTCH_LIBRARY) if _has_four_rotations(shape)]


def _row_counts(n_items: int, row_count: int) -> list[int]:
    counts = [n_items // row_count] * row_count
    for i in range(n_items % row_count):
        counts[i] += 1
    return counts


def _layout_rectangles(
    sizes: list[tuple[int, int]],
    row_count: int,
    width_limit: int,
    height_limit: int,
    gap_range: tuple[int, int],
    row_gap_range: tuple[int, int],
    exact_width: int | None = None,
) -> tuple[dict[int, tuple[int, int]], int, int] | None:
    indices = list(range(len(sizes)))
    random.shuffle(indices)
    counts = _row_counts(len(sizes), row_count)
    rows: list[list[int]] = []
    start = 0
    for count in counts:
        row_ids = indices[start : start + count]
        start += count
        random.shuffle(row_ids)
        rows.append(row_ids)

    row_specs = []
    max_content_w = 0
    for row_ids in rows:
        widths = [int(sizes[i][1]) for i in row_ids]
        heights = [int(sizes[i][0]) for i in row_ids]
        gaps = [random.randint(*gap_range) for _ in range(max(0, len(row_ids) - 1))]
        content_w = sum(widths) + sum(gaps)
        row_h = max(heights)
        row_specs.append((row_ids, gaps, row_h, content_w))
        max_content_w = max(max_content_w, int(content_w))

    if exact_width is None:
        if max_content_w > width_limit:
            return None
        canvas_w = max_content_w + random.randint(0, min(MARGIN_MAX * 2, width_limit - max_content_w))
    else:
        canvas_w = int(exact_width)
        if max_content_w > canvas_w:
            return None

    positions: dict[int, tuple[int, int]] = {}
    y = random.randint(0, MARGIN_MAX)
    for row_index, (row_ids, gaps, row_h, content_w) in enumerate(row_specs):
        x = random.randint(0, canvas_w - content_w)
        cursor = int(x)
        for i, rect_index in enumerate(row_ids):
            rect_h, rect_w = map(int, sizes[rect_index])
            yoff = random.randint(0, row_h - rect_h)
            positions[int(rect_index)] = (int(y + yoff), int(cursor))
            cursor += rect_w
            if i < len(gaps):
                cursor += int(gaps[i])
        y += int(row_h)
        if row_index < len(row_specs) - 1:
            y += random.randint(*row_gap_range)
    canvas_h = int(y + random.randint(0, MARGIN_MAX))
    if canvas_h > height_limit:
        return None
    return positions, int(canvas_h), int(canvas_w)


def _embed_notch(shape: tuple[tuple[int, int], ...], size: int, side_k: int) -> tuple[tuple[int, int], ...] | None:
    _h, w = _coords_dims(shape)
    max_offset = size - w - 1
    if max_offset < 1:
        return None
    offset = random.randint(1, max_offset)
    top_coords = [(int(r), int(c + offset)) for r, c in shape]
    return _rotate_square_coords(top_coords, size, side_k)


def _render_key_patch(
    body_shape: tuple[tuple[int, int], ...], body_color: int, handle_color: int
) -> np.ndarray:
    h, w = _coords_dims(body_shape)
    patch = np.zeros((h + 2, w + 2), dtype=int)
    for r, c in body_shape:
        patch[int(r) + 1, int(c) + 1] = int(body_color)

    candidates = set()
    for r, c in np.argwhere(patch == body_color):
        for dr, dc in NEIGHBORS4:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < patch.shape[0] and 0 <= nc < patch.shape[1] and int(patch[nr, nc]) == BG:
                candidates.add((nr, nc))
    handle_r, handle_c = random.choice(sorted(candidates))
    patch[handle_r, handle_c] = int(handle_color)

    ys, xs = np.where(patch != BG)
    return patch[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1].copy()


def _build_lock_pair(lock_color: int, key_color: int) -> dict | None:
    size = random.randint(*LOCK_SIZE_RANGE)
    base_shape = random.choice(NOTCH_LIBRARY)
    side_k = random.randrange(4)
    rotation_k = random.randrange(4)
    notch_cells = _embed_notch(base_shape, size, side_k)
    if notch_cells is None:
        return None

    lock_patch = np.full((size, size), int(lock_color), dtype=int)
    for r, c in notch_cells:
        lock_patch[int(r), int(c)] = BG

    notch_shape = _normalize_coords(notch_cells)
    key_body_shape = _rotate_coords(notch_shape, (-rotation_k) % 4)
    key_patch = _render_key_patch(key_body_shape, int(key_color), int(lock_color))

    filled_patch = lock_patch.copy()
    for r, c in notch_cells:
        filled_patch[int(r), int(c)] = int(key_color)
    output_patch = np.rot90(filled_patch, int(rotation_k))

    return {
        "lock_patch": lock_patch,
        "output_patch": output_patch,
        "key_patch": key_patch,
    }


def _choose_palette(n_locks: int) -> tuple[list[int], list[int]]:
    colors = [c for c in range(10) if c not in {BG, SEP}]
    sampled = random.sample(colors, 2 * n_locks)
    return sampled[:n_locks], sampled[n_locks:]


def _gen_attempt() -> np.ndarray | None:
    n_locks = random.choice(LOCK_COUNT_CHOICES)
    lock_colors, key_colors = _choose_palette(n_locks)

    pairs = []
    for lock_color, key_color in zip(lock_colors, key_colors):
        pair = _build_lock_pair(lock_color, key_color)
        if pair is None:
            return None
        pairs.append(pair)

    lock_sizes = [pair["lock_patch"].shape for pair in pairs]
    lock_rows = 1 if n_locks == 2 and random.random() < 0.5 else 2
    lock_layout = _layout_rectangles(
        lock_sizes,
        row_count=lock_rows,
        width_limit=LOCK_CANVAS_W_LIMIT,
        height_limit=LOCK_CANVAS_H_LIMIT,
        gap_range=LOCK_GAP_RANGE,
        row_gap_range=ROW_GAP_RANGE,
    )
    if lock_layout is None:
        return None
    lock_positions, upper_h, width = lock_layout

    key_sizes = [pair["key_patch"].shape for pair in pairs]
    key_rows = 1 if n_locks == 2 and random.random() < 0.5 else 2
    key_layout = _layout_rectangles(
        key_sizes,
        row_count=key_rows,
        width_limit=width,
        height_limit=KEY_CANVAS_H_LIMIT,
        gap_range=KEY_GAP_RANGE,
        row_gap_range=ROW_GAP_RANGE,
        exact_width=width,
    )
    if key_layout is None:
        return None
    key_positions, lower_h, _ = key_layout

    input_h = upper_h + 1 + lower_h
    if input_h > 30 or width > 30:
        return None

    inp = np.full((input_h, width), BG, dtype=int)
    expected = np.full((upper_h, width), BG, dtype=int)
    inp[upper_h, :] = SEP

    for idx, pair in enumerate(pairs):
        lock_r, lock_c = lock_positions[idx]
        lock_patch = pair["lock_patch"]
        out_patch = pair["output_patch"]
        size = int(lock_patch.shape[0])
        assert lock_patch.shape == out_patch.shape == (size, size)
        inp[lock_r : lock_r + size, lock_c : lock_c + size] = lock_patch
        expected[lock_r : lock_r + size, lock_c : lock_c + size] = out_patch

        key_patch = pair["key_patch"]
        key_r, key_c = key_positions[idx]
        kh, kw = map(int, key_patch.shape)
        target = inp[upper_h + 1 + key_r : upper_h + 1 + key_r + kh, key_c : key_c + kw]
        assert target.shape == key_patch.shape and not bool(np.any(target != BG))
        target[:, :] = key_patch

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        sample = _gen_attempt()
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid lock-and-key ARC input")
