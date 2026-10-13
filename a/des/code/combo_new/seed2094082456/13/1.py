from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
OBJECT_COLOR = 5
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GEN_ATTEMPTS = 400

HOLE_TEMPLATES: dict[str, np.ndarray] = {
    "dot": np.array([[True]], dtype=bool),
    "domino": np.array([[True], [True]], dtype=bool),
    "bar3": np.array([[True], [True], [True]], dtype=bool),
}


def _background_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[cnts.argmax()])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert len(pts) > 0, "expected a non-empty mask"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return np.asarray(mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=bool)


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    trimmed = _trim_mask(np.asarray(mask, dtype=bool))
    best: tuple[tuple[int, int], bytes] | None = None
    for k in range(4):
        rot = np.rot90(trimmed, k)
        for variant in (rot, np.fliplr(rot)):
            key = (tuple(map(int, variant.shape)), variant.tobytes())
            if best is None or key < best:
                best = key
    assert best is not None
    return best


def _foreground_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    mask = g != int(bg)
    comps = []
    for coords in _components_from_mask(mask):
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        comps.append(
            {
                "coords": coords,
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
                "colors": {int(g[r, c]) for r, c in coords},
            }
        )
    return comps


def _component_local_mask(coords: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    for r, c in coords:
        mask[int(r - r0), int(c - c0)] = True
    return mask


def _enclosed_holes(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    bg_mask = ~np.asarray(mask, dtype=bool)
    seen = np.zeros((h, w), dtype=bool)
    holes: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(bg_mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            touches_boundary = False
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    touches_boundary = True
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(bg_mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            if touches_boundary:
                continue
            hole = np.zeros((h, w), dtype=bool)
            for rr, cc in coords:
                hole[int(rr), int(cc)] = True
            holes.append(hole)
    return holes


def _paste_non_bg(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    region = dst[int(r0) : int(r0) + h, int(c0) : int(c0) + w]
    mask = patch != BG
    region[mask] = patch[mask]


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(np.asarray(mask, dtype=bool), random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey hollow object by matching its hole shape to a left-edge legend cutout.

    Input structure:
    - The background is black.
    - The left edge contains several width-2 monochrome legend bands, stacked vertically and separated by black rows.
    - Each legend band contains one black cutout shape, and each non-legend foreground component is a grey 4-connected object with exactly one enclosed black hole.

    Rule:
    - Extract the black cutout shape from every left-edge legend band and canonicalize it up to rotation and reflection.
    - For each non-legend foreground object, find its unique enclosed black hole and canonicalize that hole shape with the same invariance.
    - Use the matching legend band color to recolor all pixels of that object, while leaving the legend unchanged and keeping the hole black.

    Key insight:
    - The grey shell is only a carrier: the decisive feature is the object's enclosed hole shape, which must be matched to the left-edge legend modulo rotation and flip.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    comps = _foreground_components(g, bg)

    legend_by_shape: dict[tuple[tuple[int, int], bytes], int] = {}
    legend_cells: set[tuple[int, int]] = set()

    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        if c0 != 0 or c1 != 1 or len(comp["colors"]) != 1:
            continue
        patch = g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        hole_mask = patch == int(bg)
        if not bool(hole_mask.any()):
            continue
        hole_comps = _components_from_mask(hole_mask)
        assert len(hole_comps) == 1, "each legend band must contain exactly one cutout"
        key = _canonical_shape_key(hole_mask)
        color = next(iter(comp["colors"]))
        assert key not in legend_by_shape, "legend hole shapes must be unique"
        legend_by_shape[key] = int(color)
        legend_cells.update((int(r), int(c)) for r, c in comp["coords"])

    assert legend_by_shape, "expected at least one left-edge legend band"

    out = g.copy()
    for comp in comps:
        if all((int(r), int(c)) in legend_cells for r, c in comp["coords"]):
            continue
        local = _component_local_mask(comp["coords"], comp["bbox"])
        holes = _enclosed_holes(local)
        assert len(holes) == 1, "each object must contain exactly one enclosed hole"
        key = _canonical_shape_key(holes[0])
        assert key in legend_by_shape, "every object hole must match a legend cutout"
        recolor = int(legend_by_shape[key])
        for r, c in comp["coords"]:
            out[int(r), int(c)] = recolor

    return out


def _make_legend_segment(template: np.ndarray, color: int) -> np.ndarray:
    hole = np.asarray(template, dtype=bool)
    hole_h, _hole_w = hole.shape
    seg_h = hole_h + random.randint(1, 4)
    row0 = random.randint(0, seg_h - hole_h)
    col0 = random.randint(0, 1)

    seg = np.full((int(seg_h), 2), int(color), dtype=int)
    for rr in range(hole_h):
        seg[int(row0 + rr), int(col0)] = BG
    return seg


def _object_coords_from_hole(hole_mask: np.ndarray) -> set[tuple[int, int]]:
    hole = np.asarray(hole_mask, dtype=bool)
    hole_h, hole_w = map(int, hole.shape)
    top = random.randint(1, 2)
    bottom = random.randint(1, 2)
    left = random.randint(1, 2)
    right = random.randint(1, 2)
    outer_h = hole_h + top + bottom
    outer_w = hole_w + left + right

    coords = {
        (r, c)
        for r in range(outer_h)
        for c in range(outer_w)
        if not (top <= r < top + hole_h and left <= c < left + hole_w and bool(hole[r - top, c - left]))
    }

    for _ in range(random.randint(0, 2)):
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        side = random.choice(["top", "bottom", "left", "right"])
        depth = random.randint(1, 2)

        if side in {"top", "bottom"}:
            span = random.randint(1, width)
            start = random.randint(0, width - span)
            for dd in range(1, depth + 1):
                rr = min_r - dd if side == "top" else max_r + dd
                for cc in range(min_c + start, min_c + start + span):
                    coords.add((int(rr), int(cc)))
        else:
            span = random.randint(1, height)
            start = random.randint(0, height - span)
            for dd in range(1, depth + 1):
                cc = min_c - dd if side == "left" else max_c + dd
                for rr in range(min_r + start, min_r + start + span):
                    coords.add((int(rr), int(cc)))

    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    shift_r = -min(rows)
    shift_c = -min(cols)
    return {(int(r + shift_r), int(c + shift_c)) for r, c in coords}


def _coords_to_patch(coords: set[tuple[int, int]], color: int) -> np.ndarray:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    h = max(rows) + 1
    w = max(cols) + 1
    patch = np.full((int(h), int(w)), BG, dtype=int)
    for r, c in coords:
        patch[int(r), int(c)] = int(color)
    return patch


def _make_object_patch(template: np.ndarray) -> np.ndarray:
    hole = _random_transform(np.asarray(template, dtype=bool))
    coords = _object_coords_from_hole(hole)
    return _coords_to_patch(coords, OBJECT_COLOR)


def _can_place(occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = patch.shape
    H, W = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    cells = np.argwhere(patch != BG)
    for dr, dc in cells:
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        r1 = max(0, rr - 1)
        r2 = min(H, rr + 2)
        c1 = max(0, cc - 1)
        c2 = min(W, cc + 2)
        if bool(occupied[r1:r2, c1:c2].any()):
            return False
    return True


def _build_sample() -> tuple[np.ndarray, np.ndarray] | None:
    n_legend = random.choice([2, 3])
    chosen = random.sample(list(HOLE_TEMPLATES.items()), n_legend)
    random.shuffle(chosen)
    colors = random.sample(LEGEND_COLORS, n_legend)

    segments = []
    assignments: list[tuple[str, int]] = []
    for (name, template), color in zip(chosen, colors):
        segments.append({"name": name, "template": template, "color": int(color), "patch": _make_legend_segment(template, int(color))})
        assignments.append((name, int(color)))

    extra_objects = random.randint(0, 2)
    for _ in range(extra_objects):
        name, color = random.choice(assignments)
        assignments.append((name, int(color)))
    random.shuffle(assignments)

    objects = []
    for name, color in assignments:
        template = HOLE_TEMPLATES[name]
        patch = _make_object_patch(template)
        objects.append({"patch": patch, "color": int(color), "area": int(np.count_nonzero(patch))})

    gap_rows = [random.randint(1, 2) for _ in range(max(0, n_legend - 1))]
    legend_h = sum(seg["patch"].shape[0] for seg in segments) + sum(gap_rows)
    max_obj_h = max(obj["patch"].shape[0] for obj in objects)
    max_obj_w = max(obj["patch"].shape[1] for obj in objects)

    min_h = max(legend_h, max_obj_h + 2, 14)
    min_w = max(2 + 2 + max_obj_w, 14)
    if min_h > 30 or min_w > 30:
        return None

    H = random.randint(min_h, 30)
    W = random.randint(min_w, 30)
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    top_margin = random.randint(0, H - legend_h)
    rr = top_margin
    for i, seg in enumerate(segments):
        patch = seg["patch"]
        _paste_non_bg(inp, patch, rr, 0)
        _paste_non_bg(out, patch, rr, 0)
        occupied[rr : rr + patch.shape[0], 0:2] |= patch != BG
        rr += patch.shape[0]
        if i < len(gap_rows):
            rr += gap_rows[i]

    objects_sorted = sorted(objects, key=lambda item: (-item["area"], -item["patch"].shape[0], -item["patch"].shape[1]))
    for obj in objects_sorted:
        patch = obj["patch"]
        h, w = patch.shape
        placed = False
        for _ in range(250):
            r0 = random.randint(0, H - h)
            c0 = random.randint(4, W - w)
            if not _can_place(occupied, patch, int(r0), int(c0)):
                continue
            _paste_non_bg(inp, patch, int(r0), int(c0))
            recolored = np.where(patch == OBJECT_COLOR, int(obj["color"]), BG)
            _paste_non_bg(out, recolored, int(r0), int(c0))
            occupied[r0 : r0 + h, c0 : c0 + w] |= patch != BG
            placed = True
            break
        if not placed:
            return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        sample = _build_sample()
        if sample is None:
            continue
        inp, expected = sample
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input grid")
