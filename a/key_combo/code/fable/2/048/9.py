from __future__ import annotations

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid), return_counts=True)
    return int(values[counts.argmax()])


def _grid_size(k: int) -> int:
    return k * k + (k - 1)


def _infer_k(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    assert h == w, "expected a square slot lattice"
    candidates = [k for k in range(2, 10) if _grid_size(k) == h]
    assert len(candidates) == 1, "grid size does not match a k-by-k lattice of k-by-k slots"
    k = candidates[0]

    for idx in range(k - 1):
        sep = idx * (k + 1) + k
        assert np.all(grid[sep, :] == bg), "separator rows must stay background"
        assert np.all(grid[:, sep] == bg), "separator columns must stay background"
    return k


def _slot_origin(k: int, slot_r: int, slot_c: int) -> tuple[int, int]:
    return slot_r * (k + 1), slot_c * (k + 1)


def _slot_patch(grid: np.ndarray, k: int, slot_r: int, slot_c: int) -> np.ndarray:
    r0, c0 = _slot_origin(k, slot_r, slot_c)
    return grid[r0 : r0 + k, c0 : c0 + k].copy()


def _iter_slots(k: int):
    for slot_r in range(k):
        for slot_c in range(k):
            yield slot_r, slot_c


def _parse_sprite(slot: np.ndarray, bg: int) -> tuple[int, int, tuple[int, int]]:
    colors = [int(color) for color in np.unique(slot) if int(color) != bg]
    assert len(colors) == 2, "each occupied slot must contain one sprite color and one marker color"

    counts = {color: int(np.count_nonzero(slot == color)) for color in colors}
    marker_candidates = [color for color, count in counts.items() if count == 1]
    assert len(marker_candidates) == 1, "the marker color must appear exactly once per sprite"
    marker = marker_candidates[0]
    body = next(color for color in colors if color != marker)

    marker_pos = tuple(map(int, np.argwhere(slot == marker)[0]))
    return body, marker, marker_pos


def _parse_records(grid: np.ndarray) -> tuple[int, int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    k = _infer_k(g, bg)
    records: list[dict[str, object]] = []
    for slot_r, slot_c in _iter_slots(k):
        patch = _slot_patch(g, k, slot_r, slot_c)
        if np.all(patch == bg):
            continue
        body, marker, marker_pos = _parse_sprite(patch, bg)
        records.append(
            {
                "origin": (slot_r, slot_c),
                "dest": marker_pos,
                "body": body,
                "marker": marker,
                "patch": patch,
            }
        )

    assert records, "expected at least one occupied slot"
    destinations = [tuple(record["dest"]) for record in records]
    assert len(destinations) == len(set(destinations)), "destination slots must be unique"
    return bg, k, records


def _rewritten_patch(record: dict[str, object]) -> np.ndarray:
    patch = np.asarray(record["patch"], dtype=int).copy()
    marker_r, marker_c = map(int, record["dest"])
    patch[marker_r, marker_c] = int(record["body"])
    origin_r, origin_c = map(int, record["origin"])
    patch[origin_r, origin_c] = int(record["marker"])
    return patch


def _render_records(
    k: int,
    bg: int,
    records: list[dict[str, object]],
    *,
    place_by: str,
    rewrite_marker: bool,
) -> np.ndarray:
    out = np.full((_grid_size(k), _grid_size(k)), int(bg), dtype=int)
    for record in records:
        slot_r, slot_c = map(int, record[place_by])
        patch = _rewritten_patch(record) if rewrite_marker else np.asarray(record["patch"], dtype=int)
        r0, c0 = _slot_origin(k, slot_r, slot_c)
        out[r0 : r0 + k, c0 : c0 + k] = patch
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each sprite to the slot addressed by its own marker cell, then rewrite that marker to the source slot address.

    Input structure:
    - The full canvas is a k by k lattice of k by k slots, with one-cell background separator lines between neighboring slots.
    - Some slots contain a k by k sprite patch; inside each occupied slot, exactly one non-background cell uses a special marker color and all other foreground cells use one sprite color.
    - The marker cell's local coordinates inside the sprite patch are also a lattice address for one destination slot.

    Rule:
    - Read every occupied slot independently.
    - Move the whole k by k sprite patch into the slot whose lattice coordinates equal the marker cell's local coordinates.
    - After moving, recolor the old marker cell back to the sprite color, then paint the cell at the source slot coordinates with the marker color.
    - Leave empty slots and separator lines unchanged.

    Key insight:
    - The sprite carries its own delivery address: the marker's local row and column are interpreted in the same coordinate system as the slot lattice, and the output is only correct after the marker is rewritten from destination address to origin address.
    """
    bg, k, records = _parse_records(grid)
    return _render_records(k, bg, records, place_by="dest", rewrite_marker=True)


GEN_BG = 0
GEN_PALETTE = tuple(range(1, 10))
GEN_K_VALUES = (2, 3, 4, 5)
GEN_MAX_SPRITES = 5
GEN_MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int, k: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < k:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < k:
        yield r, c + 1


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    k = int(mask.shape[0])
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c, k):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _path_cells(a: tuple[int, int], b: tuple[int, int], rng: np.random.Generator) -> list[tuple[int, int]]:
    r, c = a
    rr, cc = b
    cells = [(r, c)]
    if int(rng.integers(2)) == 0:
        while c != cc:
            c += 1 if cc > c else -1
            cells.append((r, c))
        while r != rr:
            r += 1 if rr > r else -1
            cells.append((r, c))
    else:
        while r != rr:
            r += 1 if rr > r else -1
            cells.append((r, c))
        while c != cc:
            c += 1 if cc > c else -1
            cells.append((r, c))
    return cells


def _grow_mask(
    k: int,
    required_a: tuple[int, int],
    required_b: tuple[int, int],
    target_area: int,
    rng: np.random.Generator,
) -> np.ndarray:
    mask = np.zeros((k, k), dtype=bool)
    for r, c in _path_cells(required_a, required_b, rng):
        mask[r, c] = True

    while int(mask.sum()) < target_area:
        frontier = []
        for r, c in np.argwhere(mask):
            for nr, nc in _neighbors4(int(r), int(c), k):
                if not mask[nr, nc]:
                    frontier.append((nr, nc))
        assert frontier, "connected growth frontier unexpectedly empty"
        nr, nc = frontier[int(rng.integers(len(frontier)))]
        mask[nr, nc] = True
    return mask


def _make_patch(
    k: int,
    origin: tuple[int, int],
    dest: tuple[int, int],
    body_color: int,
    marker_color: int,
    rng: np.random.Generator,
) -> np.ndarray:
    base_area = len(_path_cells(origin, dest, rng))
    min_area = max(3, base_area)
    max_area = k * k
    for _ in range(80):
        target_area = int(rng.integers(min_area, max_area + 1))
        mask = _grow_mask(k, origin, dest, target_area, rng)
        if not mask[origin] or not mask[dest]:
            continue
        if not _is_connected(mask):
            continue

        patch = np.full((k, k), GEN_BG, dtype=int)
        patch[mask] = int(body_color)
        patch[dest] = int(marker_color)
        if np.count_nonzero(patch == body_color) < 2:
            continue
        return patch
    raise RuntimeError("failed to build a connected sprite patch")


def _sample_without_replacement(items: list[tuple[int, int]], n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    indices = rng.choice(len(items), size=n, replace=False)
    return [items[int(index)] for index in indices]


def _derangement(items: list[tuple[int, int]], rng: np.random.Generator) -> list[tuple[int, int]]:
    if len(items) == 2:
        return [items[1], items[0]]
    perm = items.copy()
    for _ in range(100):
        rng.shuffle(perm)
        if all(a != b for a, b in zip(items, perm)):
            return perm.copy()
    raise RuntimeError("failed to sample a derangement")


def _assign_targets(
    origins: list[tuple[int, int]],
    target_pool: list[tuple[int, int]],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    for _ in range(200):
        perm = target_pool.copy()
        rng.shuffle(perm)
        if all(origin != dest for origin, dest in zip(origins, perm)):
            return perm
    raise RuntimeError("failed to assign distinct target slots")


def _sample_layout(k: int, rng: np.random.Generator) -> tuple[str, list[tuple[int, int]], list[tuple[int, int]]]:
    all_slots = [(r, c) for r in range(k) for c in range(k)]
    n_slots = len(all_slots)

    feasible_modes = ["cycle"]
    if n_slots // 2 >= 2:
        feasible_modes.append("empty_only")
    if n_slots >= 3:
        feasible_modes.append("mixed")
    mode = str(rng.choice(feasible_modes))

    if mode == "cycle":
        n = int(rng.integers(2, min(GEN_MAX_SPRITES, n_slots) + 1))
        origins = _sample_without_replacement(all_slots, n, rng)
        dests = _derangement(origins, rng)
        return mode, origins, dests

    if mode == "empty_only":
        n = int(rng.integers(2, min(GEN_MAX_SPRITES, n_slots // 2) + 1))
        origins = _sample_without_replacement(all_slots, n, rng)
        empty_slots = [slot for slot in all_slots if slot not in set(origins)]
        dests = _sample_without_replacement(empty_slots, n, rng)
        rng.shuffle(dests)
        return mode, origins, dests

    n = int(rng.integers(2, min(GEN_MAX_SPRITES, n_slots - 1) + 1))
    origins = _sample_without_replacement(all_slots, n, rng)
    origin_set = set(origins)
    empty_slots = [slot for slot in all_slots if slot not in origin_set]
    empty_target_count = int(rng.integers(1, min(len(empty_slots), n - 1) + 1))
    occupied_target_count = n - empty_target_count
    target_pool = _sample_without_replacement(origins, occupied_target_count, rng)
    target_pool += _sample_without_replacement(empty_slots, empty_target_count, rng)
    dests = _assign_targets(origins, target_pool, rng)
    return mode, origins, dests


def _build_records(k: int, rng: np.random.Generator) -> tuple[str, int, list[dict[str, object]]]:
    mode, origins, dests = _sample_layout(k, rng)
    marker_color = int(rng.choice(GEN_PALETTE))
    body_palette = [color for color in GEN_PALETTE if color != marker_color]
    body_colors = rng.choice(body_palette, size=len(origins), replace=False)

    records: list[dict[str, object]] = []
    for origin, dest, body_color in zip(origins, dests, body_colors):
        patch = _make_patch(k, origin, dest, int(body_color), marker_color, rng)
        records.append(
            {
                "origin": origin,
                "dest": dest,
                "body": int(body_color),
                "marker": marker_color,
                "patch": patch,
            }
        )
    return mode, marker_color, records


def _record_signature(record: dict[str, object]) -> tuple[object, ...]:
    patch = np.asarray(record["patch"], dtype=int)
    return (
        tuple(record["origin"]),
        tuple(record["dest"]),
        int(record["body"]),
        int(record["marker"]),
        tuple(map(int, patch.ravel())),
    )


def _validate_instance(
    inp: np.ndarray,
    expected: np.ndarray,
    k: int,
    mode: str,
    records: list[dict[str, object]],
) -> bool:
    if np.array_equal(inp, expected):
        return False
    if not np.array_equal(main(inp), expected):
        return False

    bg, parsed_k, parsed_records = _parse_records(inp)
    if parsed_k != k:
        return False
    if sorted(_record_signature(record) for record in parsed_records) != sorted(
        _record_signature(record) for record in records
    ):
        return False

    move_only = _render_records(k, bg, records, place_by="dest", rewrite_marker=False)
    rewrite_only = _render_records(k, bg, records, place_by="origin", rewrite_marker=True)
    if np.array_equal(move_only, expected) or np.array_equal(rewrite_only, expected):
        return False

    origins = [tuple(record["origin"]) for record in records]
    dests = [tuple(record["dest"]) for record in records]
    if any(origin == dest for origin, dest in zip(origins, dests)):
        return False
    if len(dests) != len(set(dests)):
        return False

    for record in records:
        patch = np.asarray(record["patch"], dtype=int)
        if patch.shape != (k, k):
            return False
        if not _is_connected(patch != GEN_BG):
            return False
        if patch[tuple(record["origin"])] == GEN_BG:
            return False
        if patch[tuple(record["dest"])] != int(record["marker"]):
            return False

        delivered = _rewritten_patch(record)
        if delivered[tuple(record["origin"])] != int(record["marker"]):
            return False
        if delivered[tuple(record["dest"])] != int(record["body"]):
            return False

    origin_set = set(origins)
    dest_set = set(dests)
    if mode == "cycle" and dest_set != origin_set:
        return False
    if mode == "empty_only" and dest_set & origin_set:
        return False
    if mode == "mixed":
        if not (dest_set & origin_set):
            return False
        if not (dest_set - origin_set):
            return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_MAX_ATTEMPTS):
        k = int(rng.choice(GEN_K_VALUES))
        mode, _marker_color, records = _build_records(k, rng)
        inp = _render_records(k, GEN_BG, records, place_by="origin", rewrite_marker=False)
        expected = _render_records(k, GEN_BG, records, place_by="dest", rewrite_marker=True)
        if _validate_instance(inp, expected, k, mode, records):
            return inp
    raise RuntimeError("failed to generate a valid self-addressed delivery puzzle")
