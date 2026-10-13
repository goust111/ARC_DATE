from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.asarray(pts, dtype=int)
            r0, c0, r1, c1 = _bbox(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(len(coords)),
                }
            )
    return comps


def _normalize(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    pts = [(int(r - r0), int(c - c0)) for r, c in coords]
    return tuple(sorted(pts))


def _component_mask_shape(comp: dict) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = comp["bbox"]
    rel = comp["coords"] - np.array([r0, c0], dtype=int)
    return _normalize(rel)


def _hole_data(patch: np.ndarray, bg: int) -> list[dict]:
    holes: list[dict] = []
    for coords in _components_mask(patch == bg):
        touches_edge = np.any(
            (coords[:, 0] == 0)
            | (coords[:, 1] == 0)
            | (coords[:, 0] == patch.shape[0] - 1)
            | (coords[:, 1] == patch.shape[1] - 1)
        )
        if touches_edge:
            continue
        holes.append({"coords": coords, "shape": _normalize(coords)})
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the foreground object whose bbox contains background holes, and recolor
    each enclosed hole using the color of the standalone marker with the same
    normalized shape.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _find_components(g, bg)
    assert len(comps) >= 3, "need one holed object and marker objects"

    candidates: list[tuple[dict, list[dict], dict]] = []
    for idx, comp in enumerate(comps):
        r0, c0, r1, c1 = comp["bbox"]
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        holes = _hole_data(patch, bg)
        if not holes:
            continue

        shape_to_colors: dict[tuple[tuple[int, int], ...], set[int]] = {}
        for j, other in enumerate(comps):
            if j == idx:
                continue
            shape = _component_mask_shape(other)
            shape_to_colors.setdefault(shape, set()).add(int(other["color"]))

        mapping: dict[tuple[tuple[int, int], ...], int] = {}
        ok = True
        for hole in holes:
            colors = shape_to_colors.get(hole["shape"], set())
            if len(colors) != 1:
                ok = False
                break
            mapping[hole["shape"]] = next(iter(colors))
        if ok:
            candidates.append((comp, holes, mapping))

    assert len(candidates) == 1, "target object must be unique"
    comp, holes, mapping = candidates[0]
    r0, c0, r1, c1 = comp["bbox"]
    out = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    for hole in holes:
        color = mapping[hole["shape"]]
        rows, cols = hole["coords"][:, 0], hole["coords"][:, 1]
        out[rows, cols] = int(color)
    return out


SHAPES = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
]


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    return _normalize(coords)


def _paste(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int | None = None) -> None:
    h, w = patch.shape
    if bg is None:
        grid[top : top + h, left : left + w] = patch
    else:
        sub = grid[top : top + h, left : left + w]
        mask = patch != bg
        sub[mask] = patch[mask]


def _random_blob(h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    while int(mask.sum()) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in np.argwhere(mask):
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.append((int(nr), int(nc)))
        if not frontier:
            break
        rr, cc = random.choice(frontier)
        mask[rr, cc] = True
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _place_holes(obj_h: int, obj_w: int, shapes: list[np.ndarray]) -> list[tuple[np.ndarray, int, int]] | None:
    occ = np.zeros((obj_h, obj_w), dtype=bool)
    placed: list[tuple[np.ndarray, int, int]] = []
    for shape in shapes:
        sh, sw = shape.shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, obj_h - sh):
            for c0 in range(1, obj_w - sw):
                ok = True
                for dr in range(sh):
                    for dc in range(sw):
                        if not shape[dr, dc]:
                            continue
                        rr = r0 + dr
                        cc = c0 + dc
                        if occ[rr, cc]:
                            ok = False
                            break
                        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                            if occ[nr, nc]:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if ok:
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        for dr in range(sh):
            for dc in range(sw):
                if shape[dr, dc]:
                    occ[r0 + dr, c0 + dc] = True
        placed.append((shape, r0, c0))
    return placed


def _make_target_and_expected(
    bg: int, obj_color: int, marker_colors: list[int]
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], set[tuple[tuple[int, int], ...]]] | None:
    obj_h = random.randint(8, 14)
    obj_w = random.randint(8, 14)
    hole_count = random.randint(2, min(4, len(marker_colors)))

    shape_keys = {_shape_key(s) for s in SHAPES}
    chosen_keys = random.sample(list(shape_keys), hole_count)
    chosen_shapes = []
    used = set()
    for shape in SHAPES:
        key = _shape_key(shape)
        if key in chosen_keys and key not in used:
            chosen_shapes.append(shape)
            used.add(key)
    random.shuffle(chosen_shapes)

    placed = _place_holes(obj_h, obj_w, chosen_shapes)
    if placed is None or len(placed) != hole_count:
        return None

    target = np.full((obj_h, obj_w), int(obj_color), dtype=int)
    expected = target.copy()
    markers: list[np.ndarray] = []
    matched_keys: set[tuple[tuple[int, int], ...]] = set()
    for shape, r0, c0 in placed:
        sh, sw = shape.shape
        color = marker_colors[len(markers)]
        target[r0 : r0 + sh, c0 : c0 + sw][shape] = int(bg)
        expected[r0 : r0 + sh, c0 : c0 + sw][shape] = int(color)
        marker = np.full((sh, sw), int(bg), dtype=int)
        marker[shape] = int(color)
        markers.append(marker)
        matched_keys.add(_shape_key(shape))
    return target, expected, markers, matched_keys


def _make_distractor(bg: int, color: int, forbid: set[tuple[tuple[int, int], ...]], large: bool) -> np.ndarray:
    for _ in range(60):
        if large:
            h = random.randint(10, 15)
            w = random.randint(10, 15)
            area = random.randint((h * w) // 2, min(h * w - 1, (3 * h * w) // 4))
        else:
            h = random.randint(2, 6)
            w = random.randint(2, 6)
            area = random.randint(max(2, (h * w) // 3), h * w)
        mask = _random_blob(h, w, area)
        if _shape_key(mask) not in forbid:
            patch = np.full(mask.shape, int(bg), dtype=int)
            patch[mask] = int(color)
            return patch
    patch = np.full((2, 3), int(bg), dtype=int)
    patch[:, :] = int(color)
    return patch


def _can_place(used: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> bool:
    h, w = patch.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(used.shape[0], top + h + 1)
    c1 = min(used.shape[1], left + w + 1)
    if used[r0:r1, c0:c1].any():
        return False
    return True


def _place_objects(grid: np.ndarray, patches: list[np.ndarray], bg: int) -> bool:
    used = np.zeros(grid.shape, dtype=bool)
    order = list(range(len(patches)))
    random.shuffle(order)
    for idx in order:
        patch = patches[idx]
        h, w = patch.shape
        positions = [
            (r, c)
            for r in range(0, grid.shape[0] - h + 1)
            for c in range(0, grid.shape[1] - w + 1)
            if _can_place(used, patch, r, c, bg)
        ]
        if not positions:
            return False
        r0, c0 = random.choice(positions)
        _paste(grid, patch, r0, c0, bg=bg)
        used[r0 : r0 + h, c0 : c0 + w] = True
    return True


def _gen_attempt() -> np.ndarray | None:
    palette = random.sample(range(10), random.randint(6, 9))
    bg = palette[0]
    obj_color = palette[1]
    marker_count = random.randint(2, 4)
    marker_colors = palette[2 : 2 + marker_count]

    made = _make_target_and_expected(bg, obj_color, marker_colors)
    if made is None:
        return None
    target, expected, markers, matched_keys = made

    patches: list[np.ndarray] = [target] + markers
    spare_colors = [c for c in range(10) if c not in {bg, obj_color, *marker_colors}]
    random.shuffle(spare_colors)

    large_branch = bool(random.randrange(2))
    if large_branch:
        patches.append(_make_distractor(bg, spare_colors.pop(), matched_keys, large=True))
    for _ in range(random.randint(0, 2)):
        if not spare_colors:
            break
        patches.append(_make_distractor(bg, spare_colors.pop(), matched_keys, large=False))

    grid_h = random.randint(max(22, target.shape[0] + 6), 30)
    grid_w = random.randint(max(22, target.shape[1] + 6), 30)
    grid = np.full((grid_h, grid_w), int(bg), dtype=int)
    if not _place_objects(grid, patches, bg):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
