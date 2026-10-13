from __future__ import annotations

import random

import numpy as np


BG = 0
AXIS_COLOR = 5
FOREGROUND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
QUADRANTS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
MODE_BY_COUNT = {1: "H", 2: "V", 3: "HV"}
GEN_ATTEMPTS = 500


def _mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


MARKER_PATTERNS = [
    _mask(["00100", "01110", "11111", "01110", "00100"]),
    _mask(["00100", "00100", "11111", "00100", "00100"]),
    _mask(["01110", "01010", "11111", "01010", "01110"]),
    _mask(["0001000", "0011100", "0111110", "1111111", "0111110", "0011100", "0001000"]),
    _mask(["0001000", "0011100", "1111111", "0011100", "0001000"]),
    _mask(["00100", "01110", "01110", "11111", "01110", "01110", "00100"]),
]

COMPONENT_PATTERNS = [
    _mask(["11"]),
    _mask(["1", "1"]),
    _mask(["10", "11"]),
    _mask(["01", "11"]),
    _mask(["11", "10"]),
    _mask(["11", "01"]),
    _mask(["111"]),
    _mask(["1", "1", "1"]),
    _mask(["100", "111"]),
    _mask(["001", "111"]),
    _mask(["110", "011"]),
    _mask(["011", "110"]),
]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_coords(coords)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(comp, dtype=int))
    return out


def _paste_mask(dst: np.ndarray, patch: np.ndarray, top: int, left: int, color: int) -> None:
    coords = np.argwhere(patch)
    for r, c in coords:
        dst[int(top + r), int(left + c)] = int(color)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for arr in (rot, np.fliplr(rot)):
            key = (arr.shape, tuple(map(int, arr.ravel())))
            if key not in seen:
                seen.add(key)
                variants.append(np.array(arr, dtype=bool, copy=True))
    return variants


def _find_axis_rows_cols(grid: np.ndarray) -> tuple[int, int]:
    grey_components = _components_mask(grid == AXIS_COLOR)
    candidates: list[tuple[int, int, int]] = []
    for comp in grey_components:
        r0, c0, r1, c1 = _bbox_coords(comp)
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local = comp - np.array([[r0, c0]])
        patch[local[:, 0], local[:, 1]] = True
        if (
            patch.shape[0] % 2 == 1
            and patch.shape[1] % 2 == 1
            and np.array_equal(patch, np.flipud(patch))
            and np.array_equal(patch, np.fliplr(patch))
        ):
            candidates.append((int(comp.shape[0]), int((r0 + r1) // 2), int((c0 + c1) // 2)))
    assert candidates, "expected a grey object with both horizontal and vertical symmetry"
    candidates.sort(reverse=True)
    _area, r_h, c_v = candidates[0]
    return r_h, c_v


def _reflect_coords(
    coords: np.ndarray,
    *,
    r_h: int | None = None,
    c_v: int | None = None,
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r, c in coords:
        rr = int(r if r_h is None else 2 * int(r_h) - int(r))
        cc = int(c if c_v is None else 2 * int(c_v) - int(c))
        out.append((rr, cc))
    return out


def _reflected_targets(
    coords: np.ndarray,
    mode: str,
    r_h: int,
    c_v: int,
    shape: tuple[int, int],
) -> tuple[set[tuple[int, int]], bool]:
    raw_groups: list[list[tuple[int, int]]] = []
    if mode in {"H", "HV"}:
        raw_groups.append(_reflect_coords(coords, r_h=r_h))
    if mode in {"V", "HV"}:
        raw_groups.append(_reflect_coords(coords, c_v=c_v))
    if mode == "HV":
        raw_groups.append(_reflect_coords(coords, r_h=r_h, c_v=c_v))

    in_bounds_targets: set[tuple[int, int]] = set()
    had_clip = False
    for group in raw_groups:
        for rr, cc in group:
            if _in_bounds(shape, rr, cc):
                in_bounds_targets.add((int(rr), int(cc)))
            else:
                had_clip = True
    return in_bounds_targets, had_clip


def _closure_coords(
    coords: np.ndarray,
    mode: str,
    r_h: int,
    c_v: int,
    shape: tuple[int, int],
) -> tuple[set[tuple[int, int]], bool]:
    source = {tuple(map(int, rc)) for rc in coords}
    mirrored, had_clip = _reflected_targets(coords, mode, r_h, c_v, shape)
    return source | mirrored, had_clip


def main(grid: np.ndarray) -> np.ndarray:
    """
    Choose a borrowed symmetry axis by component count, then complete each colored object by reflection.

    Input structure:
    - Black background contains one grey 4-connected marker object with both
      horizontal and vertical mirror symmetry.
    - The grey object's center row and center column act as the borrowed mirror axes.
    - Two or three other colors each form one multi-pixel object, and each such
      color may split into one, two, or three 4-connected components inside one quadrant.

    Rule:
    - Find the grey symmetric object and read its horizontal axis row `rH` and
      vertical axis column `cV`.
    - For each non-grey foreground color, count its 4-connected components.
    - If the count is 1, mirror that color only across row `rH`; if it is 2,
      mirror only across column `cV`; if it is 3 or more, add all three borrowed
      reflections: horizontal, vertical, and the doubly reflected copy.
    - Keep the original colored pixels and keep the grey axis object unchanged;
      discard reflected pixels that fall outside the grid.

    Key insight:
    - The grey object supplies two candidate axes, but the object's own component
      count decides which borrowed reflection family must be applied to that color.
    """
    g = np.asarray(grid, dtype=int)
    r_h, c_v = _find_axis_rows_cols(g)
    out = g.copy()

    for color in sorted(int(c) for c in np.unique(g) if int(c) not in (BG, AXIS_COLOR)):
        coords = np.argwhere(g == int(color))
        components = _components_mask(g == int(color))
        k = len(components)
        mode = "H" if k == 1 else "V" if k == 2 else "HV"
        reflected, _had_clip = _reflected_targets(coords, mode, r_h, c_v, g.shape)
        for rr, cc in reflected:
            out[rr, cc] = int(color)

    return out


def _expanded8(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    expanded = set(cells)
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if _in_bounds(shape, rr, cc):
                    expanded.add((rr, cc))
    return expanded


def _touches4(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    if a & b:
        return True
    for r, c in a:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in b:
                return True
    return False


def _quadrant_bounds(shape: tuple[int, int], r_h: int, c_v: int, quad: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = map(int, shape)
    row_lo = 0 if quad[0] < 0 else r_h + 1
    row_hi = r_h - 1 if quad[0] < 0 else h - 1
    col_lo = 0 if quad[1] < 0 else c_v + 1
    col_hi = c_v - 1 if quad[1] < 0 else w - 1
    return int(row_lo), int(row_hi), int(col_lo), int(col_hi)


def _random_component_patch() -> np.ndarray:
    return np.array(random.choice(_dihedral_variants(random.choice(COMPONENT_PATTERNS))), dtype=bool, copy=True)


def _random_marker_patch() -> np.ndarray:
    return np.array(random.choice(MARKER_PATTERNS), dtype=bool, copy=True)


def _source_sign(coords: np.ndarray, r_h: int, c_v: int) -> tuple[int, int]:
    row_side = {(-1 if int(r) < int(r_h) else 1) for r, _c in coords}
    col_side = {(-1 if int(c) < int(c_v) else 1) for _r, c in coords}
    assert len(row_side) == 1 and len(col_side) == 1
    return next(iter(row_side)), next(iter(col_side))


def _mode_is_distinguishable(coords: np.ndarray, mode: str, r_h: int, c_v: int, shape: tuple[int, int]) -> bool:
    correct, _ = _closure_coords(coords, mode, r_h, c_v, shape)
    for alt in ("H", "V", "HV"):
        if alt == mode:
            continue
        alt_closure, _ = _closure_coords(coords, alt, r_h, c_v, shape)
        if alt_closure == correct:
            return False
    return True


def _try_object(
    shape: tuple[int, int],
    r_h: int,
    c_v: int,
    quad: tuple[int, int],
    k: int,
    color: int,
    blocked: set[tuple[int, int]],
) -> dict | None:
    row_lo, row_hi, col_lo, col_hi = _quadrant_bounds(shape, r_h, c_v, quad)
    if row_hi < row_lo or col_hi < col_lo:
        return None

    mode = MODE_BY_COUNT[min(k, 3)]
    for _ in range(180):
        source_cells: set[tuple[int, int]] = set()
        ok = True
        component_areas: list[int] = []
        for _comp_idx in range(k):
            patch = _random_component_patch()
            ph, pw = map(int, patch.shape)
            if row_lo + ph - 1 > row_hi or col_lo + pw - 1 > col_hi:
                ok = False
                break

            placements: list[set[tuple[int, int]]] = []
            for top in range(row_lo, row_hi - ph + 2):
                for left in range(col_lo, col_hi - pw + 2):
                    cells = {
                        (int(top + r), int(left + c))
                        for r, c in np.argwhere(patch)
                    }
                    if cells & source_cells:
                        continue
                    if _touches4(cells, source_cells):
                        continue
                    placements.append(cells)
            if not placements:
                ok = False
                break

            random.shuffle(placements)
            chosen = placements[0]
            source_cells |= chosen
            component_areas.append(len(chosen))

        if not ok or not source_cells:
            continue

        coords = np.array(sorted(source_cells), dtype=int)
        if len(_components_mask(_coords_to_local_mask(coords))) != k:
            continue
        if len(source_cells) < max(2, k + 1):
            continue

        full_closure, had_clip = _closure_coords(coords, mode, r_h, c_v, shape)
        if len(full_closure) <= len(source_cells):
            continue
        if full_closure & blocked:
            continue
        if not _mode_is_distinguishable(coords, mode, r_h, c_v, shape):
            continue

        row_side, col_side = _source_sign(coords, r_h, c_v)
        bbox = _bbox_coords(coords)
        if bbox[2] - bbox[0] == 0 and bbox[3] - bbox[1] == 0:
            continue

        return {
            "color": int(color),
            "k": int(k),
            "mode": mode,
            "quad": (int(row_side), int(col_side)),
            "coords": coords,
            "closure": full_closure,
            "had_clip": bool(had_clip),
            "component_areas": component_areas,
        }
    return None


def _coords_to_local_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local = coords - np.array([[r0, c0]])
    mask[local[:, 0], local[:, 1]] = True
    return mask


def _build_expected(inp: np.ndarray, objects: list[dict], r_h: int, c_v: int) -> np.ndarray:
    out = inp.copy()
    for obj in objects:
        reflected, _had_clip = _reflected_targets(obj["coords"], obj["mode"], r_h, c_v, inp.shape)
        for rr, cc in reflected:
            out[rr, cc] = int(obj["color"])
    return out


def _validate_generated(inp: np.ndarray, objects: list[dict], marker_patch: np.ndarray, marker_top: int, marker_left: int, need_clip: bool) -> None:
    h, w = inp.shape
    assert inp.ndim == 2 and h <= 30 and w <= 30
    r_h, c_v = _find_axis_rows_cols(inp)

    marker_bbox = _bbox(inp == AXIS_COLOR)
    assert marker_bbox is not None
    mr0, mc0, mr1, mc1 = marker_bbox
    expected_bbox = (marker_top, marker_left, marker_top + marker_patch.shape[0] - 1, marker_left + marker_patch.shape[1] - 1)
    assert marker_bbox == expected_bbox
    assert r_h == marker_top + marker_patch.shape[0] // 2
    assert c_v == marker_left + marker_patch.shape[1] // 2

    colors = sorted(int(c) for c in np.unique(inp) if int(c) not in (BG, AXIS_COLOR))
    assert len(colors) == len(objects)
    assert len({obj["k"] for obj in objects}) == len(objects)
    if need_clip:
        assert any(obj["had_clip"] for obj in objects)

    for obj in objects:
        color = int(obj["color"])
        mask = inp == color
        comps = _components_mask(mask)
        assert len(comps) == int(obj["k"])
        assert mask.sum() == obj["coords"].shape[0]
        assert _source_sign(obj["coords"], r_h, c_v) == tuple(obj["quad"])
        assert _mode_is_distinguishable(obj["coords"], obj["mode"], r_h, c_v, inp.shape)
        correct_closure, _ = _closure_coords(obj["coords"], obj["mode"], r_h, c_v, inp.shape)
        assert correct_closure == obj["closure"]
        assert len(correct_closure) > obj["coords"].shape[0]

    expected = _build_expected(inp, objects, r_h, c_v)
    solved = main(inp)
    assert np.array_equal(solved, expected)
    assert not np.array_equal(solved, inp)


def _gen_attempt() -> tuple[np.ndarray, list[dict], np.ndarray, int, int, bool] | None:
    h = random.randint(15, 23)
    w = random.randint(15, 23)
    marker_patch = _random_marker_patch()
    mh, mw = map(int, marker_patch.shape)

    base_r = h // 2
    base_c = w // 2
    r_h = random.randint(max(mh // 2 + 2, base_r - 2), min(h - mh // 2 - 3, base_r + 2))
    c_v = random.randint(max(mw // 2 + 2, base_c - 2), min(w - mw // 2 - 3, base_c + 2))
    marker_top = r_h - mh // 2
    marker_left = c_v - mw // 2

    inp = np.full((h, w), BG, dtype=int)
    _paste_mask(inp, marker_patch, marker_top, marker_left, AXIS_COLOR)

    grey_cells = {tuple(map(int, rc)) for rc in np.argwhere(inp == AXIS_COLOR)}
    blocked = _expanded8(grey_cells, inp.shape)

    object_count = 3 if random.random() < 0.7 else 2
    ks = [1, 2, 3] if object_count == 3 else list(random.choice([(1, 2), (1, 3), (2, 3)]))
    random.shuffle(ks)
    quadrants = random.sample(QUADRANTS, object_count)
    colors = random.sample(FOREGROUND_COLORS, object_count)
    need_clip = random.random() < 0.25

    objects: list[dict] = []
    for quad, k, color in zip(quadrants, ks, colors):
        obj = _try_object(inp.shape, r_h, c_v, quad, k, color, blocked)
        if obj is None:
            return None
        objects.append(obj)
        blocked |= _expanded8(obj["closure"], inp.shape)

    if need_clip and not any(obj["had_clip"] for obj in objects):
        return None

    for obj in objects:
        for r, c in obj["coords"]:
            inp[int(r), int(c)] = int(obj["color"])

    _validate_generated(inp, objects, marker_patch, marker_top, marker_left, need_clip)
    return inp, objects, marker_patch, marker_top, marker_left, need_clip


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid input")
