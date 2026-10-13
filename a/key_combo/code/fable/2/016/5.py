from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def _tight_crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "mask must be non-empty"
    return _crop_bbox(mask.astype(bool), _bbox_coords(coords)).astype(bool)


def _mask_bbox(mask: np.ndarray) -> BBox:
    coords = np.argwhere(mask)
    assert coords.size > 0, "mask must be non-empty"
    return _bbox_coords(coords)


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:, :] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r0 in range(h):
        for c0 in range(w):
            color = int(g[r0, c0])
            if color == bg or seen[r0, c0]:
                continue
            queue = deque([(r0, c0)])
            seen[r0, c0] = True
            cells: list[Coord] = []
            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            coords = np.array(cells, dtype=int)
            bbox = _bbox_coords(coords)
            bh = bbox[2] - bbox[0] + 1
            bw = bbox[3] - bbox[1] + 1
            mask = np.zeros((bh, bw), dtype=bool)
            mask[coords[:, 0] - bbox[0], coords[:, 1] - bbox[1]] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox,
                    "area": int(coords.shape[0]),
                    "mask": mask,
                }
            )
    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["area"], comp["color"]))
    return comps


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in np.asarray(mask, dtype=np.uint8))


def _all_orientations(mask: np.ndarray) -> list[tuple[str, np.ndarray]]:
    base = np.asarray(mask, dtype=bool)
    transforms = [
        ("rot0", base),
        ("rot90", np.rot90(base, 1)),
        ("rot180", np.rot90(base, 2)),
        ("rot270", np.rot90(base, 3)),
        ("flip", np.fliplr(base)),
        ("flip_rot90", np.rot90(np.fliplr(base), 1)),
        ("flip_rot180", np.rot90(np.fliplr(base), 2)),
        ("flip_rot270", np.rot90(np.fliplr(base), 3)),
    ]
    out: list[tuple[str, np.ndarray]] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for name, arr in transforms:
        key = _mask_key(arr)
        if key in seen:
            continue
        seen.add(key)
        out.append((name, arr.copy()))
    return out


def _canonical_shape(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return min(_mask_key(arr) for _, arr in _all_orientations(mask))


def _find_embeddings(template_mask: np.ndarray, residue_mask: np.ndarray) -> list[dict]:
    residue = np.asarray(residue_mask, dtype=bool)
    rh, rw = residue.shape
    placements: list[dict] = []
    for name, oriented in _all_orientations(template_mask):
        oh, ow = oriented.shape
        if rh > oh or rw > ow:
            continue
        for dr in range(oh - rh + 1):
            for dc in range(ow - rw + 1):
                window = oriented[dr : dr + rh, dc : dc + rw]
                if np.all(window[residue]):
                    placements.append({"orientation": name, "mask": oriented, "offset": (dr, dc)})
    return placements


def _solve_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _get_bg(g)
    comps = _find_components(g, bg)
    assert len(comps) >= 2 and len(comps) % 2 == 0, "expected an even number of foreground objects"

    by_residue: dict[int, list[dict]] = {idx: [] for idx in range(len(comps))}
    for template_idx, template in enumerate(comps):
        for residue_idx, residue in enumerate(comps):
            if template_idx == residue_idx or int(template["area"]) <= int(residue["area"]):
                continue
            placements = _find_embeddings(template["mask"], residue["mask"])
            if len(placements) == 1:
                match = dict(placements[0])
                match["template_idx"] = template_idx
                match["residue_idx"] = residue_idx
                by_residue[residue_idx].append(match)

    solutions: list[list[dict]] = []

    def backtrack(used: set[int], picked: list[dict]) -> None:
        if len(used) == len(comps):
            solutions.append([dict(item) for item in picked])
            return

        residue_options = []
        for residue_idx in range(len(comps)):
            if residue_idx in used:
                continue
            options = [cand for cand in by_residue[residue_idx] if cand["template_idx"] not in used]
            if options:
                residue_options.append((len(options), int(comps[residue_idx]["area"]), residue_idx, options))

        if not residue_options:
            return

        _, _area, residue_idx, options = min(residue_options, key=lambda item: (item[0], item[1], item[2]))
        used.add(residue_idx)
        for cand in sorted(options, key=lambda item: (item["template_idx"], item["offset"])):
            template_idx = cand["template_idx"]
            if template_idx in used:
                continue
            used.add(template_idx)
            picked.append(cand)
            backtrack(used, picked)
            picked.pop()
            used.remove(template_idx)
            if len(solutions) > 1:
                break
        used.remove(residue_idx)

    backtrack(set(), [])
    assert len(solutions) == 1, "scene must admit exactly one residue/template pairing"
    matches = sorted(solutions[0], key=lambda item: item["residue_idx"])
    assert len(matches) * 2 == len(comps), "every object must belong to exactly one pair"
    return bg, comps, matches


def _render_completion(grid: np.ndarray, bg: int, comps: list[dict], matches: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for match in matches:
        residue = comps[match["residue_idx"]]
        color = int(residue["color"])
        bbox = residue["bbox"]
        dr, dc = match["offset"]
        full_mask = np.asarray(match["mask"], dtype=bool)
        fh, fw = full_mask.shape
        full_r0 = int(bbox[0]) - int(dr)
        full_c0 = int(bbox[1]) - int(dc)
        assert 0 <= full_r0 <= out.shape[0] - fh, "completed residue must stay in bounds"
        assert 0 <= full_c0 <= out.shape[1] - fw, "completed residue must stay in bounds"
        view = out[full_r0 : full_r0 + fh, full_c0 : full_c0 + fw]
        blocked = full_mask & (view != bg) & (view != color)
        assert not np.any(blocked), "repaired pixels must not overlap unrelated objects"
        view[full_mask] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Repair each residue by completing it with the uniquely matching template orientation and placement.

    Input structure
    A background grid contains an even number of separated monochrome objects. For each hidden pair, one object is
    a complete template, while the other is a connected residue obtained by rotating or reflecting that same shape
    and then erasing some of its pixels. Colors are arbitrary and do not identify the pairs.

    Rule
    Identify every pair where one object's visible cells are a strict subset of some rotation or reflection of
    another object's full shape. For each residue, choose the only template variant and in-bounds placement whose
    occupied cells fully cover the residue's surviving pixels, then fill the missing cells in the residue's own
    color. Leave the complete templates and the background unchanged.

    Key insight
    Pairing and completion depend only on geometric containment under the eight dihedral orientations. The template's
    drawn pose is irrelevant; the residue's surviving pixels uniquely determine which transformed template must be
    reconstructed in place.
    """
    g = np.asarray(grid, dtype=int)
    bg, comps, matches = _solve_scene(g)
    return _render_completion(g, bg, comps, matches)


GEN_BG = 0
GEN_COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_PAIR_COUNT_RANGE = (2, 3)
GEN_GRID_RANGES = {2: ((18, 24), (18, 24)), 3: ((23, 30), (23, 30))}
GEN_SHAPE_BBOX_RANGE = (3, 6)
GEN_SHAPE_AREA_RANGE = (7, 11)
GEN_PLACEMENT_BORDER = 1
GEN_PLACEMENT_PADDING = 2
GEN_ATTEMPTS = 400
GEN_SHAPE_TRIES = 200
GEN_RESIDUE_TRIES = 300


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _random_shape() -> np.ndarray:
    min_bbox, max_bbox = GEN_SHAPE_BBOX_RANGE
    min_area, max_area = GEN_SHAPE_AREA_RANGE
    for _ in range(GEN_SHAPE_TRIES):
        h = random.randint(min_bbox, max_bbox)
        w = random.randint(min_bbox, max_bbox)
        max_fill = min(max_area, h * w - 2)
        if max_fill < min_area:
            continue
        target_area = random.randint(min_area, max_fill)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier = set()
        for nr, nc in _neighbors4(*start):
            if 0 <= nr < h and 0 <= nc < w:
                frontier.add((nr, nc))
        while len(cells) < target_area and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nr, nc in _neighbors4(*cell):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if len(cells) != target_area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _tight_crop_mask(mask)
        if min(mask.shape) < 3:
            continue
        if not _is_connected(mask):
            continue
        if len(_all_orientations(mask)) != 8:
            continue
        return mask
    raise RuntimeError("failed to sample an asymmetric connected shape")


def _choose_color_pair(force_same: bool | None) -> tuple[int, int]:
    if force_same is True:
        color = random.choice(GEN_COLORS)
        return color, color
    template_color = random.choice(GEN_COLORS)
    residue_color = random.choice(GEN_COLORS)
    if force_same is False:
        while residue_color == template_color:
            residue_color = random.choice(GEN_COLORS)
    return template_color, residue_color


def _pick_residue(
    template_display_mask: np.ndarray,
    residue_full_mask: np.ndarray,
    *,
    require_shift: bool,
) -> tuple[np.ndarray, np.ndarray]:
    full = np.asarray(residue_full_mask, dtype=bool)
    fh, fw = full.shape
    coords = [tuple(map(int, cell)) for cell in np.argwhere(full)]
    area = len(coords)
    max_remove = min(5, area - 3)
    assert max_remove >= 1, "shape must leave room for a strict connected residue"

    side_groups = []
    top_cells = {(r, c) for r, c in coords if r == 0}
    left_cells = {(r, c) for r, c in coords if c == 0}
    if 1 <= len(top_cells) <= max_remove:
        side_groups.append(top_cells)
    if 1 <= len(left_cells) <= max_remove:
        side_groups.append(left_cells)

    for attempt in range(GEN_RESIDUE_TRIES):
        seed = set()
        if require_shift and side_groups and attempt < GEN_RESIDUE_TRIES // 2:
            seed = set(random.choice(side_groups))
        remove_count = random.randint(max(1, len(seed)), max_remove)
        removable = [cell for cell in coords if cell not in seed]
        if remove_count - len(seed) > len(removable):
            continue
        removed = set(seed)
        if remove_count > len(seed):
            removed.update(random.sample(removable, remove_count - len(seed)))

        survivor_full = full.copy()
        for r, c in removed:
            survivor_full[r, c] = False
        if survivor_full.sum() < 3 or not _is_connected(survivor_full):
            continue

        sr0, sc0, sr1, sc1 = _mask_bbox(survivor_full)
        shifted = sr0 > 0 or sc0 > 0
        if require_shift != shifted:
            continue
        residue_crop = survivor_full[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
        placements = _find_embeddings(template_display_mask, residue_crop)
        if len(placements) != 1:
            continue
        placement = placements[0]
        if placement["offset"] != (sr0, sc0):
            continue
        if not np.array_equal(placement["mask"], full):
            continue
        if residue_crop.shape == (fh, fw) and np.array_equal(residue_crop, full):
            continue
        return survivor_full, residue_crop
    raise RuntimeError("failed to build an unambiguous residue")


def _find_location(occupied: np.ndarray, shape: tuple[int, int]) -> Coord | None:
    h, w = shape
    H, W = occupied.shape
    candidates: list[Coord] = []
    for r0 in range(GEN_PLACEMENT_BORDER, H - h - GEN_PLACEMENT_BORDER + 1):
        for c0 in range(GEN_PLACEMENT_BORDER, W - w - GEN_PLACEMENT_BORDER + 1):
            rr0 = max(0, r0 - GEN_PLACEMENT_PADDING)
            cc0 = max(0, c0 - GEN_PLACEMENT_PADDING)
            rr1 = min(H, r0 + h + GEN_PLACEMENT_PADDING)
            cc1 = min(W, c0 + w + GEN_PLACEMENT_PADDING)
            if not np.any(occupied[rr0:rr1, cc0:cc1]):
                candidates.append((r0, c0))
    if not candidates:
        return None
    return random.choice(candidates)


def _mask_patch(mask: np.ndarray, color: int, bg: int = GEN_BG) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[np.asarray(mask, dtype=bool)] = int(color)
    return patch


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*GEN_PAIR_COUNT_RANGE)
    row_range, col_range = GEN_GRID_RANGES[pair_count]
    H = random.randint(*row_range)
    W = random.randint(*col_range)

    canonical_seen: set[tuple[tuple[int, ...], ...]] = set()
    used_areas: set[int] = set()
    pair_specs = []

    force_same_flags = [True, False] + [None] * (pair_count - 2)
    residue_modes = ["rotation", "reflection"] + [random.choice(["rotation", "reflection"]) for _ in range(pair_count - 2)]
    shift_flags = [True, False] + [random.choice([True, False]) for _ in range(pair_count - 2)]

    for idx in range(pair_count):
        for _ in range(GEN_SHAPE_TRIES):
            base_mask = _random_shape()
            canon = _canonical_shape(base_mask)
            area = int(base_mask.sum())
            if canon in canonical_seen or area in used_areas:
                continue

            orientations = _all_orientations(base_mask)
            rotation_orients = [(name, arr) for name, arr in orientations if not name.startswith("flip")]
            reflection_orients = [(name, arr) for name, arr in orientations if name.startswith("flip")]
            template_name, template_mask = random.choice(orientations)
            residue_pool = rotation_orients if residue_modes[idx] == "rotation" else reflection_orients
            residue_pool = [item for item in residue_pool if item[0] != template_name]
            if not residue_pool:
                residue_pool = [item for item in orientations if item[0] != template_name]
            if not residue_pool:
                continue
            residue_name, residue_full_mask = random.choice(residue_pool)

            try:
                residue_visible_full, residue_crop = _pick_residue(
                    template_mask,
                    residue_full_mask,
                    require_shift=shift_flags[idx],
                )
            except RuntimeError:
                continue

            template_color, residue_color = _choose_color_pair(force_same_flags[idx])
            canonical_seen.add(canon)
            used_areas.add(area)
            pair_specs.append(
                {
                    "base_mask": base_mask,
                    "template_name": template_name,
                    "template_mask": template_mask,
                    "template_color": template_color,
                    "residue_name": residue_name,
                    "residue_full_mask": residue_full_mask,
                    "residue_visible_full": residue_visible_full,
                    "residue_crop": residue_crop,
                    "residue_color": residue_color,
                    "needs_shift": shift_flags[idx],
                }
            )
            break
        else:
            return None

    occupied = np.zeros((H, W), dtype=bool)
    objects = []
    for spec in sorted(pair_specs, key=lambda item: max(item["template_mask"].size, item["residue_full_mask"].size), reverse=True):
        template_pos = _find_location(occupied, spec["template_mask"].shape)
        if template_pos is None:
            return None
        tr, tc = template_pos
        occupied[tr : tr + spec["template_mask"].shape[0], tc : tc + spec["template_mask"].shape[1]] = True

        residue_pos = _find_location(occupied, spec["residue_full_mask"].shape)
        if residue_pos is None:
            return None
        rr, rc = residue_pos
        occupied[rr : rr + spec["residue_full_mask"].shape[0], rc : rc + spec["residue_full_mask"].shape[1]] = True

        objects.append((spec, template_pos, residue_pos))

    inp = np.full((H, W), GEN_BG, dtype=int)
    expected = np.full_like(inp, GEN_BG)

    for spec, template_pos, residue_pos in objects:
        template_patch = _mask_patch(spec["template_mask"], spec["template_color"])
        residue_input_patch = _mask_patch(spec["residue_visible_full"], spec["residue_color"])
        residue_output_patch = _mask_patch(spec["residue_full_mask"], spec["residue_color"])
        _paste(inp, template_patch, template_pos[0], template_pos[1], transparent=GEN_BG)
        _paste(inp, residue_input_patch, residue_pos[0], residue_pos[1], transparent=GEN_BG)
        _paste(expected, template_patch, template_pos[0], template_pos[1], transparent=GEN_BG)
        _paste(expected, residue_output_patch, residue_pos[0], residue_pos[1], transparent=GEN_BG)

    if np.array_equal(inp, expected):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    # Semantic branch coverage inside each sample: at least one shifted residue and one reflected residue.
    if not any(spec["needs_shift"] for spec, _, _ in objects):
        return None
    if not any(spec["residue_name"].startswith("flip") for spec, _, _ in objects):
        return None
    if not any(spec["template_color"] == spec["residue_color"] for spec, _, _ in objects):
        return None
    if not any(spec["template_color"] != spec["residue_color"] for spec, _, _ in objects):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid residue-repair scene")
