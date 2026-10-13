from __future__ import annotations

from collections import Counter, deque

import numpy as np


BG_COLOR = 0
MARKER_COLOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
PROBE_CANON = np.array([[-1, -1], [-1, 0], [0, -1]], dtype=int)
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < h and 0 <= int(c) < w


def _sorted_coords(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=int)
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[order]


def _normalize_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    arr = np.asarray(coords, dtype=int)
    mins = arr.min(axis=0)
    shifted = _sorted_coords(arr - mins)
    return shifted, (int(mins[0]), int(mins[1]))


def _coord_key(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    norm, _ = _normalize_coords(coords)
    return tuple(map(tuple, norm.tolist()))


def _apply_transform(coords: np.ndarray, transform_id: int) -> np.ndarray:
    arr = np.asarray(coords, dtype=int)
    r = arr[:, 0]
    c = arr[:, 1]
    tid = int(transform_id)
    if tid == 0:
        out = np.column_stack([r, c])
    elif tid == 1:
        out = np.column_stack([c, -r])
    elif tid == 2:
        out = np.column_stack([-r, -c])
    elif tid == 3:
        out = np.column_stack([-c, r])
    elif tid == 4:
        out = np.column_stack([r, -c])
    elif tid == 5:
        out = np.column_stack([-c, -r])
    elif tid == 6:
        out = np.column_stack([-r, c])
    elif tid == 7:
        out = np.column_stack([c, r])
    else:
        raise ValueError("invalid transform id")
    return out.astype(int)


def _shift_coords(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    return np.asarray(coords, dtype=int) + np.array([int(dr), int(dc)], dtype=int)


def _cells(coords: np.ndarray) -> set[tuple[int, int]]:
    return set(map(tuple, np.asarray(coords, dtype=int).tolist()))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    return int(arr[:, 0].min()), int(arr[:, 1].min()), int(arr[:, 0].max()), int(arr[:, 1].max())


def _is_l_probe(coords: np.ndarray) -> bool:
    arr = np.asarray(coords, dtype=int)
    if arr.shape != (3, 2):
        return False
    r0, c0, r1, c1 = _bbox(arr)
    if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
        return False
    return len(_cells(arr)) == 3


def _find_components(
    grid: np.ndarray,
    *,
    bg: int,
    connectivity: int = 4,
    monochromatic: bool = False,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        offsets = NEIGHBORS4
    else:
        offsets = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(g[r, c]) == int(bg):
                continue
            seed_color = int(g[r, c])
            q = deque([(r, c)])
            visited[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in offsets:
                    nr = rr + dr
                    nc = cc + dc
                    if not _in_bounds(g.shape, nr, nc) or visited[nr, nc]:
                        continue
                    if int(g[nr, nc]) == int(bg):
                        continue
                    if monochromatic and int(g[nr, nc]) != seed_color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            vals = g[arr[:, 0], arr[:, 1]]
            out.append(
                {
                    "coords": arr,
                    "bbox": _bbox(arr),
                    "area": int(arr.shape[0]),
                    "colors": set(map(int, vals.tolist())),
                }
            )
    return out


def _match_translation(source_probe_rel: np.ndarray, target_probe_abs: np.ndarray, transform_id: int) -> tuple[int, int] | None:
    probe_tf = _apply_transform(source_probe_rel, transform_id)
    norm_probe, probe_min = _normalize_coords(probe_tf)
    norm_target, target_min = _normalize_coords(target_probe_abs)
    if not np.array_equal(norm_probe, norm_target):
        return None
    return int(target_min[0] - probe_min[0]), int(target_min[1] - probe_min[1])


def _paint_cells(grid: np.ndarray, coords: np.ndarray, values: np.ndarray) -> None:
    for (r, c), color in zip(np.asarray(coords, dtype=int), np.asarray(values, dtype=int)):
        grid[int(r), int(c)] = int(color)


def _colored_pattern_key(coords: np.ndarray, colors: np.ndarray) -> tuple[tuple[int, int, int], ...]:
    arr = np.asarray(coords, dtype=int)
    mins = arr.min(axis=0)
    shifted = arr - mins
    triples = sorted((int(r), int(c), int(color)) for (r, c), color in zip(shifted, colors))
    return tuple(triples)


def _random_connected_mask(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(3, 7))
    w = int(rng.integers(3, 7))
    lo = max(4, (h * w) // 3)
    hi = min(h * w, 10)
    if lo > hi:
        return None
    total = int(rng.integers(lo, hi + 1))

    required = {(0, 0), (h - 1, int(rng.integers(0, w))), (int(rng.integers(0, h)), w - 1)}
    cells = {(0, 0)}
    frontier = {(0, 1), (1, 0)}
    while len(cells) < total:
        options = []
        for rr, cc in list(frontier):
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                options.append((rr, cc))
        if not options:
            return None
        pick = options[int(rng.integers(len(options)))]
        cells.add(pick)
        frontier.discard(pick)
        for dr, dc in NEIGHBORS4:
            frontier.add((pick[0] + dr, pick[1] + dc))

    if not required.issubset(cells):
        return None

    mask = np.zeros((h, w), dtype=bool)
    for rr, cc in cells:
        mask[rr, cc] = True
    return mask


def _build_source_sprite(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(80):
        mask = _random_connected_mask(rng)
        if mask is None:
            continue
        coords = np.argwhere(mask)
        colors = rng.choice(PALETTE, size=len(coords), replace=True).astype(int)
        if len(set(map(int, colors.tolist()))) < 2:
            continue
        keys = set()
        for transform_id in range(8):
            coords_tf = _apply_transform(coords, transform_id)
            keys.add(_colored_pattern_key(coords_tf, colors))
        if len(keys) != 8:
            continue
        return coords.astype(int), colors.astype(int)
    return None


def _choose_translation(coords: np.ndarray, shape: tuple[int, int], rng: np.random.Generator) -> tuple[int, int] | None:
    arr = np.asarray(coords, dtype=int)
    min_r, min_c = arr.min(axis=0)
    max_r, max_c = arr.max(axis=0)
    h, w = shape
    rows = list(range(-int(min_r), h - int(max_r)))
    cols = list(range(-int(min_c), w - int(max_c)))
    if not rows or not cols:
        return None
    rng.shuffle(rows)
    rng.shuffle(cols)
    for rr in rows:
        for cc in cols:
            return int(rr), int(cc)
    return None


def _valid_sprite_landing(
    sprite_abs: np.ndarray,
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> bool:
    for r, c in np.asarray(sprite_abs, dtype=int):
        rr = int(r)
        cc = int(c)
        if not _in_bounds(shape, rr, cc) or (rr, cc) in blocked:
            return False
    return True


def _candidate_target_positions(
    source_probe_rel: np.ndarray,
    source_sprite_rel: np.ndarray,
    transform_id: int,
    shape: tuple[int, int],
    occupied_input: set[tuple[int, int]],
    occupied_output: set[tuple[int, int]],
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
    probe_tf = _apply_transform(source_probe_rel, transform_id)
    sprite_tf = _apply_transform(source_sprite_rel, transform_id)
    all_tf = np.vstack([probe_tf, sprite_tf])
    arr = np.asarray(all_tf, dtype=int)
    min_r, min_c = arr.min(axis=0)
    max_r, max_c = arr.max(axis=0)
    h, w = shape

    translations = [
        (tr, tc)
        for tr in range(-int(min_r), h - int(max_r))
        for tc in range(-int(min_c), w - int(max_c))
    ]
    rng.shuffle(translations)

    good: list[tuple[np.ndarray, np.ndarray]] = []
    for tr, tc in translations:
        probe_abs = _shift_coords(probe_tf, tr, tc)
        sprite_abs = _shift_coords(sprite_tf, tr, tc)
        probe_cells = _cells(probe_abs)
        sprite_cells = _cells(sprite_abs)
        if probe_cells & occupied_input:
            continue
        if sprite_cells & occupied_output:
            continue
        if not _valid_sprite_landing(sprite_abs, shape, occupied_input | probe_cells):
            continue

        valid_ids = []
        for other_id in range(8):
            shift = _match_translation(source_probe_rel, probe_abs, other_id)
            if shift is None:
                continue
            other_sprite = _shift_coords(_apply_transform(source_sprite_rel, other_id), shift[0], shift[1])
            if _valid_sprite_landing(other_sprite, shape, occupied_input | probe_cells):
                valid_ids.append(other_id)
        if valid_ids == [int(transform_id)]:
            good.append((probe_abs, sprite_abs))
    return good


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the multicolor source sprite to each gray L marker, using each marker's pose to choose the sprite's transformed pose.

    Input structure:
    - The background contains one multicolor source sprite that includes a gray three-cell L probe attached at one corner.
    - Additional isolated gray three-cell L markers are scattered elsewhere on the background.

    Rule:
    - Separate the unique mixed-color source component from the gray-only target markers.
    - For each isolated marker, test the eight dihedral transforms of the source probe and keep the unique one whose transformed sprite lands fully on background cells.
    - Erase the isolated marker and paint the source sprite without gray cells at that transformed placement, while leaving the original source sprite and source probe unchanged.

    Key insight:
    - The gray L is not only a destination anchor: its local pose determines the same rotation or reflection applied to the whole source sprite, and the sprite-to-probe offset transforms with it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    assert bg != MARKER_COLOR, "gray marker must differ from the background"

    components = _find_components(g, bg=bg, connectivity=4, monochromatic=False)
    source_component = None
    target_markers = []
    for comp in components:
        coords = comp["coords"]
        vals = g[coords[:, 0], coords[:, 1]]
        has_marker = np.any(vals == MARKER_COLOR)
        has_sprite = np.any(vals != MARKER_COLOR)
        if has_marker and has_sprite:
            assert source_component is None, "expected exactly one source component"
            source_component = comp
        elif has_marker:
            target_markers.append(coords)
        else:
            raise AssertionError("unexpected foreground component without a gray probe")

    assert source_component is not None, "missing source component"
    assert target_markers, "expected at least one isolated gray target marker"

    source_coords = source_component["coords"]
    source_vals = g[source_coords[:, 0], source_coords[:, 1]]
    source_probe_abs = source_coords[source_vals == MARKER_COLOR]
    source_sprite_abs = source_coords[source_vals != MARKER_COLOR]
    source_sprite_colors = source_vals[source_vals != MARKER_COLOR]

    assert _is_l_probe(source_probe_abs), "source marker must be a three-cell L"
    for marker in target_markers:
        vals = g[marker[:, 0], marker[:, 1]]
        assert np.all(vals == MARKER_COLOR), "target markers must be gray-only"
        assert _is_l_probe(marker), "target markers must be three-cell L probes"

    source_ref = _sorted_coords(source_probe_abs)[0]
    source_probe_rel = source_probe_abs - source_ref
    source_sprite_rel = source_sprite_abs - source_ref

    placements: list[tuple[np.ndarray, np.ndarray]] = []
    for target_probe_abs in target_markers:
        matches = []
        for transform_id in range(8):
            shift = _match_translation(source_probe_rel, target_probe_abs, transform_id)
            if shift is None:
                continue
            sprite_abs = _shift_coords(_apply_transform(source_sprite_rel, transform_id), shift[0], shift[1])
            if not _valid_sprite_landing(sprite_abs, g.shape, _cells(target_probe_abs) | (_cells(source_coords)) | set().union(*(_cells(m) for m in target_markers if not np.array_equal(m, target_probe_abs)))):
                continue
            matches.append((transform_id, sprite_abs))

        assert len(matches) == 1, "each target probe must admit exactly one valid transformed sprite"
        placements.append((target_probe_abs, matches[0][1]))

    out = g.copy()
    occupied = _cells(source_coords)
    for target_probe_abs, sprite_abs in placements:
        for r, c in target_probe_abs:
            out[int(r), int(c)] = bg
        if occupied & _cells(sprite_abs):
            raise AssertionError("copied sprites must not overlap the source or each other")
        _paint_cells(out, sprite_abs, source_sprite_colors)
        occupied |= _cells(sprite_abs)
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    built = _build_source_sprite(rng)
    if built is None:
        return None
    sprite_coords_canon, sprite_colors = built

    source_transform = int(rng.integers(8))
    source_probe_local = _apply_transform(PROBE_CANON, source_transform)
    source_sprite_local = _apply_transform(sprite_coords_canon, source_transform)

    source_ref_local = _sorted_coords(source_probe_local)[0]
    source_probe_rel = source_probe_local - source_ref_local
    source_sprite_rel = source_sprite_local - source_ref_local

    grid_h = int(rng.integers(17, 30))
    grid_w = int(rng.integers(17, 30))
    grid_shape = (grid_h, grid_w)

    source_all_local = np.vstack([source_probe_local, source_sprite_local])
    source_shift = _choose_translation(source_all_local, grid_shape, rng)
    if source_shift is None:
        return None
    source_probe_abs = _shift_coords(source_probe_local, source_shift[0], source_shift[1])
    source_sprite_abs = _shift_coords(source_sprite_local, source_shift[0], source_shift[1])
    source_component_abs = np.vstack([source_probe_abs, source_sprite_abs])

    occupied_input = _cells(source_component_abs)
    occupied_output = set(occupied_input)

    target_count = int(rng.integers(2, 5))
    transform_ids = []
    rotation_ids = [0, 1, 2, 3]
    mirror_ids = [4, 5, 6, 7]
    if target_count >= 2:
        transform_ids.append(int(rotation_ids[int(rng.integers(len(rotation_ids)))]))
        transform_ids.append(int(mirror_ids[int(rng.integers(len(mirror_ids)))]))
    pool = list(range(8))
    rng.shuffle(pool)
    for transform_id in pool:
        if len(transform_ids) >= target_count:
            break
        if transform_id not in transform_ids:
            transform_ids.append(int(transform_id))
    rng.shuffle(transform_ids)

    target_probes: list[np.ndarray] = []
    target_sprites: list[np.ndarray] = []
    for transform_id in transform_ids:
        candidates = _candidate_target_positions(
            source_probe_rel,
            source_sprite_rel,
            transform_id,
            grid_shape,
            occupied_input,
            occupied_output,
            rng,
        )
        if not candidates:
            return None
        pick = candidates[int(rng.integers(len(candidates)))]
        probe_abs, sprite_abs = pick
        target_probes.append(probe_abs)
        target_sprites.append(sprite_abs)
        occupied_input |= _cells(probe_abs)
        occupied_output |= _cells(sprite_abs)

    inp = np.full(grid_shape, BG_COLOR, dtype=int)
    _paint_cells(inp, source_sprite_abs, sprite_colors)
    _paint_cells(inp, source_probe_abs, np.full(len(source_probe_abs), MARKER_COLOR, dtype=int))
    for probe_abs in target_probes:
        _paint_cells(inp, probe_abs, np.full(len(probe_abs), MARKER_COLOR, dtype=int))

    out = inp.copy()
    for probe_abs, sprite_abs in zip(target_probes, target_sprites):
        for r, c in probe_abs:
            out[int(r), int(c)] = BG_COLOR
        _paint_cells(out, sprite_abs, sprite_colors)

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid posture-probe input")
