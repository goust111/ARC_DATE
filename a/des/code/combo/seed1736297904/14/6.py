from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
SEP = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GEN_ATTEMPTS = 400
_RNG = np.random.default_rng()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _exterior_background(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    ext = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not ext[r, c]:
                ext[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not ext[r, c]:
                ext[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not ext[nr, nc]:
                ext[nr, nc] = True
                q.append((nr, nc))
    return ext


def _filled_silhouette(mask: np.ndarray) -> np.ndarray:
    return ~_exterior_background(mask)


def _hole_records(mask: np.ndarray) -> list[dict[str, object]]:
    holes_mask = (~mask) & (~_exterior_background(mask))
    holes: list[dict[str, object]] = []
    for coords in _components(holes_mask):
        r0, c0, r1, c1 = _bbox(coords)
        rel = coords - np.array([r0, c0], dtype=int)
        hole = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        hole[rel[:, 0], rel[:, 1]] = True
        holes.append({"bbox": (int(r0), int(c0), int(r1), int(c1)), "mask": hole})
    holes.sort(key=lambda rec: tuple(rec["bbox"]))  # type: ignore[arg-type]
    return holes


def _all_dihedral(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for flipped in (False, True):
        base = np.fliplr(mask) if flipped else mask
        for k in range(4):
            variants.append(_crop_bool(np.rot90(base, k)))
    return variants


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min((tuple(arr.shape), arr.tobytes()) for arr in _all_dihedral(mask))


def _is_chiral(mask: np.ndarray) -> bool:
    mirror = _crop_bool(np.fliplr(mask))
    for k in range(4):
        if np.array_equal(_crop_bool(np.rot90(mask, k)), mirror):
            return False
    return True


def _find_separator(grid: np.ndarray) -> tuple[int, int]:
    cols = [c for c in range(grid.shape[1]) if bool((grid[:, c] == SEP).all())]
    assert cols, "expected a full-height grey separator"
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(cols):
        j = i
        while j + 1 < len(cols) and cols[j + 1] == cols[j] + 1:
            j += 1
        runs.append((cols[i], cols[j]))
        i = j + 1
    best = None
    for s, e in runs:
        if s == 0 or e == grid.shape[1] - 1:
            continue
        score = (e - s + 1, -abs((s + e) / 2.0 - (grid.shape[1] - 1) / 2.0))
        if best is None or score > best[0]:
            best = (score, (s, e))
    assert best is not None
    return best[1]


def _extract_monochrome_components(grid: np.ndarray, skip_colors: set[int] | None = None) -> list[dict[str, object]]:
    skip = {BG}
    if skip_colors is not None:
        skip |= {int(x) for x in skip_colors}
    parts: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in skip):
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            rel = coords - np.array([r0, c0], dtype=int)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[rel[:, 0], rel[:, 1]] = True
            parts.append({"color": color, "bbox": (r0, c0, r1, c1), "mask": mask})
    parts.sort(key=lambda rec: tuple(rec["bbox"]))  # type: ignore[arg-type]
    return parts


def _template_matches(mask: np.ndarray, hole: np.ndarray, use_mirror: bool) -> bool:
    for k in range(4):
        candidate = np.rot90(mask, k)
        if use_mirror:
            candidate = np.fliplr(candidate)
        if candidate.shape == hole.shape and np.array_equal(candidate, hole):
            return True
    return False


def _match_template(hole_mask: np.ndarray, templates: list[dict[str, object]], prefer_mirror: bool) -> dict[str, object]:
    modes = (prefer_mirror, not prefer_mirror)
    for use_mirror in modes:
        matches = [tmpl for tmpl in templates if _template_matches(tmpl["mask"], hole_mask, use_mirror)]  # type: ignore[arg-type]
        if len(matches) == 1:
            return matches[0]
    raise AssertionError("hole-template match must be unique")


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    sep0, sep1 = _find_separator(g)
    left = g[:, :sep0].copy()
    right = g[:, sep1 + 1 :]
    templates = _extract_monochrome_components(right)

    out = left.copy()
    objects = _extract_monochrome_components(left)
    for obj in objects:
        mask = obj["mask"]  # type: ignore[assignment]
        bbox = obj["bbox"]  # type: ignore[assignment]
        silhouette = _filled_silhouette(mask)  # type: ignore[arg-type]
        horizontal_axis_symmetric = np.array_equal(silhouette, np.flipud(silhouette))
        for hole in _hole_records(mask):  # type: ignore[arg-type]
            hole_mask = hole["mask"]  # type: ignore[assignment]
            tmpl = _match_template(hole_mask, templates, prefer_mirror=not horizontal_axis_symmetric)
            hr0, hc0, _hr1, _hc1 = hole["bbox"]  # type: ignore[misc]
            r0, c0, _r1, _c1 = bbox
            view = out[r0 + hr0 : r0 + hr0 + hole_mask.shape[0], c0 + hc0 : c0 + hc0 + hole_mask.shape[1]]
            view[hole_mask] = int(tmpl["color"])
    return out


def _expand8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    coords = np.argwhere(mask)
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def _random_connected_shape(rng: np.random.Generator, used_keys: set[tuple[tuple[int, int], bytes]], require_chiral: bool) -> np.ndarray:
    for _ in range(300):
        box_h = int(rng.integers(2, 5))
        box_w = int(rng.integers(2, 5))
        target = int(rng.integers(3, min(box_h * box_w, 7) + 1))
        start = (int(rng.integers(box_h)), int(rng.integers(box_w)))
        cells = {start}
        while len(cells) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = frontier[int(rng.integers(len(frontier)))]
            cells.add((nr, nc))
        if len(cells) != target:
            continue
        mask = np.zeros((box_h, box_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_bool(mask)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) >= mask.size:
            continue
        key = _canonical_key(mask)
        if key in used_keys:
            continue
        if require_chiral and not _is_chiral(mask):
            continue
        return mask
    raise RuntimeError("failed to sample a unique template shape")


def _build_object(
    rng: np.random.Generator,
    hole_specs: list[dict[str, object]],
    horizontal_axis_symmetric: bool,
    color: int,
) -> dict[str, object]:
    hole_masks = [spec["hole_mask"] for spec in hole_specs]
    max_h = max(int(mask.shape[0]) for mask in hole_masks)  # type: ignore[index]
    max_w = max(int(mask.shape[1]) for mask in hole_masks)  # type: ignore[index]

    placements = None
    core_shape = None
    for _ in range(120):
        core_h = int(rng.integers(max(6, max_h + 4), max(8, max_h + 7) + 1))
        core_w = int(rng.integers(max(6, max_w + 4), max(8, max_w + 7) + 1))
        used = np.zeros((core_h, core_w), dtype=bool)
        blocked = np.zeros((core_h, core_w), dtype=bool)
        placed: list[tuple[int, int]] = []
        ok = True
        order = sorted(range(len(hole_masks)), key=lambda i: int(hole_masks[i].sum()), reverse=True)
        pos = [(-1, -1)] * len(hole_masks)
        for idx in order:
            hole = hole_masks[idx]
            hh, ww = hole.shape  # type: ignore[assignment]
            candidates: list[tuple[int, int]] = []
            for r in range(1, core_h - hh):
                for c in range(1, core_w - ww):
                    trial = np.zeros((core_h, core_w), dtype=bool)
                    trial[r : r + hh, c : c + ww][hole] = True  # type: ignore[index]
                    if bool((trial & blocked).any()):
                        continue
                    candidates.append((r, c))
            if not candidates:
                ok = False
                break
            r, c = candidates[int(rng.integers(len(candidates)))]
            placed.append((r, c))
            pos[idx] = (r, c)
            used[r : r + hh, c : c + ww][hole] = True  # type: ignore[index]
            blocked = _expand8(used)
        if not ok:
            continue
        core = np.ones((core_h, core_w), dtype=bool)
        for spec, (r, c) in zip(hole_specs, pos):
            hole = spec["hole_mask"]  # type: ignore[assignment]
            hh, ww = hole.shape  # type: ignore[assignment]
            core[r : r + hh, c : c + ww][hole] = False
        placements = pos
        core_shape = core
        break

    if placements is None or core_shape is None:
        raise RuntimeError("failed to place holes inside object core")

    ext = 2
    core_h, core_w = core_shape.shape
    canvas = np.zeros((core_h + 2 * ext + 2, core_w + 2 * ext + 2), dtype=bool)
    canvas[ext : ext + core_h, ext : ext + core_w] = core_shape

    def add_top_tab(width: int, height: int, col: int) -> None:
        canvas[ext - height : ext, ext + col : ext + col + width] = True

    def add_bottom_tab(width: int, height: int, col: int) -> None:
        canvas[ext + core_h : ext + core_h + height, ext + col : ext + col + width] = True

    if horizontal_axis_symmetric:
        tab_count = int(rng.integers(1, 3))
        for _ in range(tab_count):
            width = int(rng.integers(2, min(core_w, 5) + 1))
            height = int(rng.integers(1, 3))
            col = int(rng.integers(0, core_w - width + 1))
            add_top_tab(width, height, col)
            add_bottom_tab(width, height, col)
    else:
        width = int(rng.integers(2, min(core_w, 5) + 1))
        height = int(rng.integers(1, 3))
        col = int(rng.integers(0, core_w - width + 1))
        add_top_tab(width, height, col)
        if rng.random() < 0.5:
            width2 = int(rng.integers(2, min(core_w, 5) + 1))
            height2 = int(rng.integers(1, 3))
            col2 = int(rng.integers(0, core_w - width2 + 1))
            if width2 != width or height2 != height or col2 != col:
                add_bottom_tab(width2, height2, col2)

    obj_mask = _crop_bool(canvas)
    if horizontal_axis_symmetric != np.array_equal(_filled_silhouette(obj_mask), np.flipud(_filled_silhouette(obj_mask))):
        if horizontal_axis_symmetric:
            raise RuntimeError("symmetric object lost its intended symmetry")
        # Add a one-cell nub on the top to force asymmetry.
        padded = np.pad(obj_mask, ((1, 0), (0, 0)), constant_values=False)
        cols = np.argwhere(padded[1]).ravel()
        if cols.size == 0:
            raise RuntimeError("invalid object mask")
        padded[0, int(cols[0])] = True
        obj_mask = _crop_bool(padded)

    coords = np.argwhere(obj_mask)
    r0, c0, _r1, _c1 = _bbox(coords)
    del r0, c0  # crop already normalized

    patch = np.full(obj_mask.shape, BG, dtype=int)
    patch[obj_mask] = int(color)

    holes: list[dict[str, object]] = []
    for spec, (hr, hc) in zip(hole_specs, placements):
        hole = spec["hole_mask"]  # type: ignore[assignment]
        hh, ww = hole.shape  # type: ignore[assignment]
        # Core begins at (ext, ext) before cropping; recover the crop offset from obj_mask.
        canvas_hole_r = ext + hr
        canvas_hole_c = ext + hc
        full_coords = np.argwhere(canvas)
        cr0, cc0, _cr1, _cc1 = _bbox(full_coords)
        local_r = canvas_hole_r - cr0
        local_c = canvas_hole_c - cc0
        holes.append({"r": int(local_r), "c": int(local_c), "mask": hole, "fill_color": int(spec["color"])})

    for hole in holes:
        r = int(hole["r"])
        c = int(hole["c"])
        hole_mask = hole["mask"]  # type: ignore[assignment]
        patch[r : r + hole_mask.shape[0], c : c + hole_mask.shape[1]][hole_mask] = BG

    return {"patch": patch, "holes": holes, "symmetric": horizontal_axis_symmetric}


def _place_bboxes(
    rng: np.random.Generator,
    canvas_shape: tuple[int, int],
    patches: list[np.ndarray],
    pad: int = 1,
) -> list[tuple[int, int]] | None:
    h, w = canvas_shape
    occ = np.zeros((h, w), dtype=bool)
    order = sorted(range(len(patches)), key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)
    pos = [(-1, -1)] * len(patches)
    for idx in order:
        ph, pw = patches[idx].shape
        candidates: list[tuple[int, int]] = []
        for r in range(0, h - ph + 1):
            for c in range(0, w - pw + 1):
                r0 = max(0, r - pad)
                c0 = max(0, c - pad)
                r1 = min(h, r + ph + pad)
                c1 = min(w, c + pw + pad)
                if bool(occ[r0:r1, c0:c1].any()):
                    continue
                candidates.append((r, c))
        if not candidates:
            return None
        pos[idx] = candidates[int(rng.integers(len(candidates)))]
        r, c = pos[idx]
        occ[r : r + ph, c : c + pw] = True
    return pos


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    object_count = int(rng.integers(2, 4))
    if object_count == 2:
        hole_counts = [1, int(rng.integers(1, 3))]
    else:
        hole_counts = [1, 1, int(rng.integers(1, 3))]
    rng.shuffle(hole_counts)
    total_holes = int(sum(hole_counts))

    if object_count == 2:
        sym_flags = [True, False]
    else:
        sym_flags = [True, True, False] if rng.random() < 0.5 else [True, False, False]
    rng.shuffle(sym_flags)

    color_pool = rng.permutation(PALETTE).tolist()
    if object_count + total_holes > len(color_pool):
        return None
    object_colors = [int(color_pool.pop()) for _ in range(object_count)]
    fill_colors = [int(color_pool.pop()) for _ in range(total_holes)]

    used_shape_keys: set[tuple[tuple[int, int], bytes]] = set()
    objects: list[dict[str, object]] = []
    templates: list[np.ndarray] = []
    template_arrays: list[np.ndarray] = []
    template_colors: list[int] = []
    fill_idx = 0

    for obj_idx in range(object_count):
        hole_specs: list[dict[str, object]] = []
        for _ in range(hole_counts[obj_idx]):
            display_shape = _random_connected_shape(rng, used_shape_keys, require_chiral=not sym_flags[obj_idx])
            used_shape_keys.add(_canonical_key(display_shape))
            display_shape = np.rot90(display_shape, int(rng.integers(4)))
            hole_shape = np.rot90(display_shape, int(rng.integers(4)))
            if not sym_flags[obj_idx]:
                hole_shape = np.fliplr(hole_shape)
            fill_color = fill_colors[fill_idx]
            fill_idx += 1
            hole_specs.append({"hole_mask": hole_shape, "color": fill_color})
            template_patch = np.full(display_shape.shape, BG, dtype=int)
            template_patch[display_shape] = fill_color
            template_arrays.append(template_patch)
            templates.append(display_shape)
            template_colors.append(fill_color)
        obj = _build_object(rng, hole_specs, sym_flags[obj_idx], object_colors[obj_idx])
        objects.append(obj)

    grid_h = int(rng.integers(18, 25))
    left_w = int(rng.integers(15, 19))
    right_w = int(rng.integers(8, 11))
    if left_w + 1 + right_w > 30:
        return None

    object_patches = [obj["patch"] for obj in objects]  # type: ignore[list-item]
    obj_positions = _place_bboxes(rng, (grid_h, left_w), object_patches, pad=1)
    if obj_positions is None:
        return None
    tmpl_positions = _place_bboxes(rng, (grid_h, right_w), template_arrays, pad=1)
    if tmpl_positions is None:
        return None

    left = np.full((grid_h, left_w), BG, dtype=int)
    expected = np.full((grid_h, left_w), BG, dtype=int)
    for obj, (r, c) in zip(objects, obj_positions):
        patch = obj["patch"]  # type: ignore[assignment]
        ph, pw = patch.shape  # type: ignore[assignment]
        left[r : r + ph, c : c + pw][patch != BG] = patch[patch != BG]
        expected[r : r + ph, c : c + pw][patch != BG] = patch[patch != BG]
        for hole in obj["holes"]:  # type: ignore[index]
            hr = int(hole["r"])
            hc = int(hole["c"])
            hole_mask = hole["mask"]  # type: ignore[assignment]
            fill_color = int(hole["fill_color"])
            view = expected[r + hr : r + hr + hole_mask.shape[0], c + hc : c + hc + hole_mask.shape[1]]
            view[hole_mask] = fill_color

    right = np.full((grid_h, right_w), BG, dtype=int)
    for patch, (r, c) in zip(template_arrays, tmpl_positions):
        ph, pw = patch.shape
        right[r : r + ph, c : c + pw][patch != BG] = patch[patch != BG]

    full = np.concatenate([left, np.full((grid_h, 1), SEP, dtype=int), right], axis=1)

    if len(np.unique(full.reshape(-1, full.shape[1]), axis=0)) == 1:
        return None
    if not np.array_equal(main(full), expected):
        return None
    return full


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
