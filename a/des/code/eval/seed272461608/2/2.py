from __future__ import annotations

import numpy as np


BG = 8
SHADOW = 5
ANCHOR = 9

_RNG = np.random.default_rng()
_DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
_DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_TRANSFORMS = ("identity", "hflip", "vflip", "rot180")

# --------------- solver helpers ---------------


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in _DIRS8:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) != int(bg):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _apply_transform(arr: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return np.array(arr, copy=True)
    if transform == "hflip":
        return np.fliplr(arr)
    if transform == "vflip":
        return np.flipud(arr)
    assert transform == "rot180"
    return np.rot90(arr, 2)


def _transform_coord(coord: tuple[int, int], shape: tuple[int, int], transform: str) -> tuple[int, int]:
    r, c = map(int, coord)
    h, w = map(int, shape)
    if transform == "identity":
        return r, c
    if transform == "hflip":
        return r, w - 1 - c
    if transform == "vflip":
        return h - 1 - r, c
    assert transform == "rot180"
    return h - 1 - r, w - 1 - c


def _signature(mask: np.ndarray, anchor: tuple[int, int], transform: str) -> tuple[int, int, bytes, int, int]:
    tm = _apply_transform(mask.astype(np.uint8), transform)
    ar, ac = _transform_coord(anchor, mask.shape, transform)
    return int(tm.shape[0]), int(tm.shape[1]), tm.tobytes(), int(ar), int(ac)


def _extract_object(grid: np.ndarray, coords: list[tuple[int, int]], bg: int) -> dict[str, object]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = int(grid[r, c])

    anchor_pts = np.argwhere(patch == ANCHOR)
    assert len(anchor_pts) == 1, "each object must contain exactly one maroon anchor"
    anchor = tuple(map(int, anchor_pts[0]))
    mask = patch != bg
    colors = set(map(int, np.unique(patch[mask])))

    is_shadow = colors <= {SHADOW, ANCHOR} and SHADOW in colors
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "coords": [(int(r), int(c)) for r, c in coords],
        "patch": patch,
        "mask": mask,
        "anchor": anchor,
        "is_shadow": bool(is_shadow),
    }


def _candidate_transforms(stamp: dict[str, object], shadow: dict[str, object]) -> list[str]:
    stamp_mask = np.asarray(stamp["mask"], dtype=bool)
    shadow_mask = np.asarray(shadow["mask"], dtype=bool)
    if stamp_mask.shape != shadow_mask.shape:
        return []

    stamp_anchor = tuple(stamp["anchor"])
    shadow_anchor = tuple(shadow["anchor"])
    out: list[str] = []
    for transform in _TRANSFORMS:
        if not np.array_equal(_apply_transform(stamp_mask.astype(np.uint8), transform).astype(bool), shadow_mask):
            continue
        if _transform_coord(stamp_anchor, stamp_mask.shape, transform) != shadow_anchor:
            continue
        out.append(transform)
    return out


def _match_pairs(stamps: list[dict[str, object]], shadows: list[dict[str, object]]) -> list[tuple[int, int, str]]:
    candidates: list[list[tuple[int, str]]] = []
    for stamp in stamps:
        options: list[tuple[int, str]] = []
        for shadow_idx, shadow in enumerate(shadows):
            for transform in _candidate_transforms(stamp, shadow):
                options.append((int(shadow_idx), transform))
        assert options, "every stamp must match at least one shadow"
        candidates.append(options)

    order = sorted(range(len(stamps)), key=lambda idx: len(candidates[idx]))
    assignment: list[tuple[int, int, str]] = []
    used_shadows: set[int] = set()

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        stamp_idx = order[pos]
        for shadow_idx, transform in candidates[stamp_idx]:
            if shadow_idx in used_shadows:
                continue
            used_shadows.add(shadow_idx)
            assignment.append((int(stamp_idx), int(shadow_idx), transform))
            if dfs(pos + 1):
                return True
            assignment.pop()
            used_shadows.remove(shadow_idx)
        return False

    assert dfs(0), "stamp-shadow matching must have a perfect assignment"
    assignment.sort(key=lambda item: item[0])
    return assignment


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored stamp onto the matching grey shadow with the shadow's orientation.

    Input structure:
    - Background is azure.
    - Each foreground object is either a multicolor stamp or a grey shadow.
    - Every object has exactly one maroon anchor pixel.
    - A shadow has the same shape as its paired stamp, but may be horizontally flipped,
      vertically flipped, or rotated 180 degrees.

    Rule:
    - Match each stamp to a shadow by shape plus transformed anchor position.
    - Apply the same geometric transform to the stamp's full color pattern.
    - Erase the original stamp and redraw the transformed stamp at the shadow location.
    - Keep the maroon anchor maroon.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)

    objects = [_extract_object(g, comp, bg) for comp in _find_components(g, bg)]
    stamps = [obj for obj in objects if not bool(obj["is_shadow"])]
    shadows = [obj for obj in objects if bool(obj["is_shadow"])]

    assert stamps and shadows, "expected both stamps and shadows"
    assert len(stamps) == len(shadows), "stamp and shadow counts must match"

    out = np.array(g, copy=True)
    for stamp in stamps:
        for r, c in stamp["coords"]:
            out[r, c] = int(bg)

    for stamp_idx, shadow_idx, transform in _match_pairs(stamps, shadows):
        stamp = stamps[stamp_idx]
        shadow = shadows[shadow_idx]
        for r, c in shadow["coords"]:
            out[r, c] = int(bg)

        patch = _apply_transform(np.asarray(stamp["patch"], dtype=int), transform)
        mask = patch != bg
        r0, c0, r1, c1 = shadow["bbox"]
        assert patch.shape == (r1 - r0 + 1, c1 - c0 + 1)
        out_slice = out[r0 : r1 + 1, c0 : c1 + 1]
        out_slice[mask] = patch[mask]
        out[r0 : r1 + 1, c0 : c1 + 1] = out_slice

    return out


# --------------- generator helpers ---------------

GRID_H_RANGE = (24, 33)
GRID_W_RANGE = (24, 33)
PAIR_COUNT_RANGE = (4, 7)
MAX_RETRIES = 400
PLACEMENT_ATTEMPTS = 300
MASK_ATTEMPTS = 300
PAIR_ATTEMPTS = 300
COLOR_POOL = [0, 1, 2, 3, 4, 6, 7]
DUPLICATE_SHAPE_PROB = 0.45


def _mask_symmetry_signatures(mask: np.ndarray) -> set[tuple[int, int, bytes]]:
    out = set()
    for transform in _TRANSFORMS:
        tm = _apply_transform(mask.astype(np.uint8), transform)
        out.add((int(tm.shape[0]), int(tm.shape[1]), tm.tobytes()))
    return out


def _generate_mask() -> np.ndarray:
    for _ in range(MASK_ATTEMPTS):
        h = int(_RNG.integers(3, 6))
        w = int(_RNG.integers(3, 6))
        max_area = h * w - 1
        min_area = max(4, int(np.ceil(0.45 * h * w)))
        if min_area > max_area:
            continue
        target = int(_RNG.integers(min_area, max_area + 1))

        start = (int(_RNG.integers(h)), int(_RNG.integers(w)))
        cells = {start}
        while len(cells) < target:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in _DIRS4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((int(nr), int(nc)))
            if not frontier:
                break
            pick = list(frontier)[int(_RNG.integers(len(frontier)))]
            cells.add(pick)

        if len(cells) != target:
            continue
        if not any(r == 0 for r, _ in cells):
            continue
        if not any(r == h - 1 for r, _ in cells):
            continue
        if not any(c == 0 for _, c in cells):
            continue
        if not any(c == w - 1 for _, c in cells):
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if len(_mask_symmetry_signatures(mask)) != 4:
            continue
        return mask
    raise RuntimeError("failed to generate a non-symmetric mask")


def _canonical_mask_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    return min(_mask_symmetry_signatures(mask))


def _canonical_anchor_key(mask: np.ndarray, anchor: tuple[int, int]) -> tuple[int, int, bytes, int, int]:
    return min(_signature(mask, anchor, transform) for transform in _TRANSFORMS)


def _anchor_options(mask: np.ndarray) -> list[tuple[int, int]]:
    keys_seen: set[tuple[int, int, bytes, int, int]] = set()
    anchors: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        anchor = (int(r), int(c))
        key = _canonical_anchor_key(mask, anchor)
        if key in keys_seen:
            continue
        keys_seen.add(key)
        anchors.append(anchor)
    return anchors


def _make_stamp(mask: np.ndarray, anchor: tuple[int, int]) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    cells = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
    non_anchor = [cell for cell in cells if cell != anchor]
    fill_count = min(len(non_anchor), int(_RNG.integers(1, 3)))
    fill_colors = list(map(int, _RNG.choice(COLOR_POOL, size=fill_count, replace=False)))
    order = [non_anchor[idx] for idx in _RNG.permutation(len(non_anchor))]

    for idx, (r, c) in enumerate(order):
        color = fill_colors[idx] if idx < fill_count else fill_colors[int(_RNG.integers(fill_count))]
        patch[r, c] = int(color)
    patch[anchor] = ANCHOR
    return patch


def _make_shadow(mask: np.ndarray, anchor: tuple[int, int]) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = SHADOW
    patch[anchor] = ANCHOR
    return patch


def _place_patch(
    grid: np.ndarray,
    blocked: np.ndarray,
    patch: np.ndarray,
) -> tuple[int, int] | None:
    h, w = grid.shape
    ph, pw = patch.shape
    coords = [
        (int(r0), int(c0))
        for r0 in range(h - ph + 1)
        for c0 in range(w - pw + 1)
    ]
    if not coords:
        return None
    order = _RNG.permutation(len(coords))
    tries = min(len(coords), PLACEMENT_ATTEMPTS)

    for idx in order[:tries]:
        r0, c0 = coords[int(idx)]
        r1 = r0 + ph - 1
        c1 = c0 + pw - 1
        br0 = max(0, r0 - 1)
        bc0 = max(0, c0 - 1)
        br1 = min(h - 1, r1 + 1)
        bc1 = min(w - 1, c1 + 1)
        if np.any(blocked[br0 : br1 + 1, bc0 : bc1 + 1]):
            continue
        mask = patch != BG
        grid_slice = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(grid_slice[mask] != BG):
            continue

        grid_slice[mask] = patch[mask]
        grid[r0 : r1 + 1, c0 : c1 + 1] = grid_slice
        blocked[br0 : br1 + 1, bc0 : bc1 + 1] = True
        return int(r0), int(c0)
    return None


def _overlay(out: np.ndarray, patch: np.ndarray, top: tuple[int, int]) -> None:
    r0, c0 = map(int, top)
    ph, pw = patch.shape
    mask = patch != BG
    sub = out[r0 : r0 + ph, c0 : c0 + pw]
    sub[mask] = patch[mask]
    out[r0 : r0 + ph, c0 : c0 + pw] = sub


def _clear_patch(out: np.ndarray, patch: np.ndarray, top: tuple[int, int]) -> None:
    r0, c0 = map(int, top)
    ph, pw = patch.shape
    mask = patch != BG
    sub = out[r0 : r0 + ph, c0 : c0 + pw]
    sub[mask] = BG
    out[r0 : r0 + ph, c0 : c0 + pw] = sub


def _build_pair_specs(pair_count: int) -> list[dict[str, object]]:
    transforms = list(_TRANSFORMS)
    while len(transforms) < pair_count:
        transforms.append(_TRANSFORMS[int(_RNG.integers(len(_TRANSFORMS)))])
    transforms = [transforms[idx] for idx in _RNG.permutation(len(transforms))]

    pair_specs: list[dict[str, object]] = []
    used_anchor_keys: set[tuple[int, int, bytes, int, int]] = set()
    reusable_masks: list[np.ndarray] = []

    for pair_idx in range(pair_count):
        for _ in range(PAIR_ATTEMPTS):
            use_duplicate = bool(reusable_masks) and bool(_RNG.random() < DUPLICATE_SHAPE_PROB)
            mask = reusable_masks[int(_RNG.integers(len(reusable_masks)))] if use_duplicate else _generate_mask()
            anchors = _anchor_options(mask)
            if not anchors:
                continue
            anchor = anchors[int(_RNG.integers(len(anchors)))]
            anchor_key = _canonical_anchor_key(mask, anchor)
            if anchor_key in used_anchor_keys:
                continue

            used_anchor_keys.add(anchor_key)
            pair_specs.append(
                {
                    "mask": mask,
                    "anchor": anchor,
                    "transform": transforms[pair_idx],
                }
            )
            if not use_duplicate:
                reusable_masks.append(mask)
            break
        else:
            raise RuntimeError("failed to build unique pair specs")

    return pair_specs


def _expected_output(grid: np.ndarray, placements: list[dict[str, object]]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for item in placements:
        _clear_patch(out, item["stamp"], item["stamp_top"])
    for item in placements:
        _clear_patch(out, item["shadow"], item["shadow_top"])
        moved = _apply_transform(np.asarray(item["stamp"], dtype=int), str(item["transform"]))
        _overlay(out, moved, item["shadow_top"])
    return out


def _gen_attempt() -> np.ndarray | None:
    h = int(_RNG.integers(*GRID_H_RANGE))
    w = int(_RNG.integers(*GRID_W_RANGE))
    pair_count = int(_RNG.integers(*PAIR_COUNT_RANGE))

    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    pair_specs = _build_pair_specs(pair_count)
    placements: list[dict[str, object]] = []

    objects: list[tuple[str, int, np.ndarray]] = []
    for idx, spec in enumerate(pair_specs):
        mask = np.asarray(spec["mask"], dtype=bool)
        anchor = tuple(spec["anchor"])
        stamp = _make_stamp(mask, anchor)
        shadow = _apply_transform(_make_shadow(mask, anchor), str(spec["transform"]))
        objects.append(("stamp", idx, stamp))
        objects.append(("shadow", idx, shadow))

    objects = [objects[idx] for idx in _RNG.permutation(len(objects))]
    staged: dict[int, dict[str, object]] = {
        idx: {
            "transform": spec["transform"],
        }
        for idx, spec in enumerate(pair_specs)
    }

    for kind, idx, patch in objects:
        top = _place_patch(grid, blocked, patch)
        if top is None:
            return None
        staged[idx][kind] = patch
        staged[idx][f"{kind}_top"] = top

    for idx in range(pair_count):
        item = staged[idx]
        if "stamp" not in item or "shadow" not in item:
            return None
        placements.append(item)

    expected = _expected_output(grid, placements)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
