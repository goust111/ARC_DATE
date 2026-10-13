from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


ORANGE = 7
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _flood_from_boundary(passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _shape_key(mask: np.ndarray) -> tuple:
    reps = []
    for k in range(4):
        rot = np.rot90(mask, k)
        reps.append((tuple(int(x) for x in rot.shape), tuple(int(x) for x in rot.astype(np.uint8).ravel())))
    return min(reps)


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return {(int(r - r0), int(c - c0)) for r, c in cells}


def _random_polyomino(size: int) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < size:
        frontier = set()
        for r, c in cells:
            for dr, dc in DIR4:
                pos = (r + dr, c + dc)
                if pos not in cells:
                    frontier.add(pos)
        cells.add(random.choice(list(frontier)))
    return _normalize_cells(cells)


def _polyomino_to_masks(cavity_cells: set[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    norm = _normalize_cells(cavity_cells)
    cavity = {(r + 1, c + 1) for r, c in norm}
    border = set()
    for r, c in cavity:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                pos = (r + dr, c + dc)
                if pos not in cavity:
                    border.add(pos)
    h = 1 + max(r for r, _ in border | cavity)
    w = 1 + max(c for _, c in border | cavity)
    border_mask = np.zeros((h, w), dtype=bool)
    cavity_mask = np.zeros((h, w), dtype=bool)
    for r, c in border:
        border_mask[r, c] = True
    for r, c in cavity:
        cavity_mask[r, c] = True
    return border_mask, cavity_mask


def _render_object(
    cavity_cells: set[tuple[int, int]],
    border_color: int,
    bg: int,
    marker_count: int,
    rotation_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    border_mask, cavity_mask = _polyomino_to_masks(cavity_cells)
    border_mask = np.rot90(border_mask, int(rotation_k))
    cavity_mask = np.rot90(cavity_mask, int(rotation_k))
    patch = np.full(border_mask.shape, int(bg), dtype=int)
    patch[border_mask] = int(border_color)
    if marker_count:
        spots = [(int(r), int(c)) for r, c in np.argwhere(cavity_mask)]
        for r, c in random.sample(spots, int(marker_count)):
            patch[r, c] = ORANGE
    return patch, cavity_mask


def _extract_objects(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    objects: list[dict] = []
    colors = sorted(set(int(x) for x in np.unique(grid)) - {int(bg), ORANGE})
    for color in colors:
        for comp in _components(grid == int(color)):
            r0, c0, r1, c1 = _bbox(comp)
            comp_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            local_border = np.zeros((r1 - r0 + 3, c1 - c0 + 3), dtype=bool)
            for r, c in comp:
                comp_mask[r - r0, c - c0] = True
                local_border[r - r0 + 1, c - c0 + 1] = True

            exterior = _flood_from_boundary(~local_border)
            cavity_local = (~local_border) & (~exterior)
            cavity_comps = _components(cavity_local)
            if len(cavity_comps) != 1:
                continue

            cavity_global: list[tuple[int, int]] = []
            ok = True
            marker_count = 0
            for lr, lc in cavity_comps[0]:
                gr = int(r0 + lr - 1)
                gc = int(c0 + lc - 1)
                if not (0 <= gr < h and 0 <= gc < w):
                    ok = False
                    break
                val = int(grid[gr, gc])
                if val == ORANGE:
                    marker_count += 1
                elif val != int(bg):
                    ok = False
                    break
                cavity_global.append((gr, gc))
            if not ok:
                continue

            objects.append(
                {
                    "color": int(color),
                    "border_coords": [(int(r), int(c)) for r, c in comp],
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "border_mask": comp_mask,
                    "shape_key": _shape_key(comp_mask),
                    "cavity_coords": cavity_global,
                    "marker_count": int(marker_count),
                }
            )
    return objects


def _pair_objects(objects: list[dict]) -> list[tuple[dict, dict]]:
    groups: dict[tuple, list[dict]] = {}
    for obj in objects:
        groups.setdefault(obj["shape_key"], []).append(obj)

    pairs: list[tuple[dict, dict]] = []
    for group in groups.values():
        masters = [obj for obj in group if int(obj["marker_count"]) > 0]
        children = [obj for obj in group if int(obj["marker_count"]) == 0]
        assert len(masters) == 1 and len(children) == 1, "expected one master and one child per shape"
        pairs.append((masters[0], children[0]))
    return pairs


def _shell_distances(
    grid: np.ndarray,
    bg: int,
    border_coords: list[tuple[int, int]],
    layers: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = grid.shape
    r0, c0, r1, c1 = _bbox(border_coords)
    pad = int(layers) + 1
    sr0 = max(0, int(r0 - pad))
    sc0 = max(0, int(c0 - pad))
    sr1 = min(int(h - 1), int(r1 + pad))
    sc1 = min(int(w - 1), int(c1 + pad))
    sub = grid[sr0 : sr1 + 1, sc0 : sc1 + 1]
    passable = sub == int(bg)
    exterior = _flood_from_boundary(passable)
    dist = np.full(sub.shape, -1, dtype=int)
    q: deque[tuple[int, int]] = deque()

    for gr, gc in border_coords:
        lr = int(gr - sr0)
        lc = int(gc - sc0)
        for dr, dc in DIR4:
            nr, nc = lr + dr, lc + dc
            if 0 <= nr < sub.shape[0] and 0 <= nc < sub.shape[1] and exterior[nr, nc] and dist[nr, nc] == -1:
                dist[nr, nc] = 1
                q.append((nr, nc))

    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < sub.shape[0] and 0 <= nc < sub.shape[1] and exterior[nr, nc] and dist[nr, nc] == -1:
                dist[nr, nc] = int(dist[r, c] + 1)
                q.append((nr, nc))

    return dist, (int(sr0), int(sc0))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair master/child frame shapes by rotation-invariant border shape.
    The number of orange marker pixels inside each master decides how many
    external outline layers are added around the matching child.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    objects = _extract_objects(g, bg)
    pairs = _pair_objects(objects)

    out = g.copy()
    for master, child in pairs:
        layers = int(master["marker_count"])
        dist, (sr0, sc0) = _shell_distances(g, bg, child["border_coords"], layers)
        for lr, lc in np.argwhere((dist >= 1) & (dist <= layers)):
            layer = int(dist[lr, lc])
            color = int(master["color"]) if layer % 2 == 1 else int(child["color"])
            gr = int(sr0 + lr)
            gc = int(sc0 + lc)
            assert int(out[gr, gc]) == int(bg), "expansion layers must only paint background"
            out[gr, gc] = color
    return out


def _place_patch(
    grid: np.ndarray,
    reserved: np.ndarray,
    patch: np.ndarray,
    pad: int,
) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    rows = range(int(pad), int(h - ph - pad + 1))
    cols = range(int(pad), int(w - pw - pad + 1))
    candidates = [(r, c) for r in rows for c in cols]
    random.shuffle(candidates)
    for r, c in candidates:
        rr0 = int(r - pad)
        cc0 = int(c - pad)
        rr1 = int(r + ph + pad)
        cc1 = int(c + pw + pad)
        if reserved[rr0:rr1, cc0:cc1].any():
            continue
        sub = grid[r : r + ph, c : c + pw]
        if np.any((patch != sub) & (patch != _get_bg(grid)) & (sub != _get_bg(grid))):
            continue
        grid[r : r + ph, c : c + pw][patch != _get_bg(grid)] = patch[patch != _get_bg(grid)]
        reserved[rr0:rr1, cc0:cc1] = True
        return True
    return False


def _sample_unique_shapes(count: int) -> list[set[tuple[int, int]]]:
    shapes: list[set[tuple[int, int]]] = []
    seen: set[tuple] = set()
    attempts = 0
    while len(shapes) < count and attempts < 200:
        attempts += 1
        cells = _random_polyomino(random.randint(4, 5))
        border_mask, _ = _polyomino_to_masks(cells)
        key = _shape_key(border_mask)
        if key in seen:
            continue
        seen.add(key)
        shapes.append(cells)
    return shapes


def _build_candidate() -> np.ndarray | None:
    pair_count = random.choice([2, 3, 3, 4])
    h = random.randint(22, 30)
    w = random.randint(22, 30)
    bg = random.choice([0, 1, 2, 3, 4, 5, 6, 8, 9])
    palette = [c for c in range(10) if c not in {bg, ORANGE}]
    if len(palette) < 2 * pair_count:
        return None

    shapes = _sample_unique_shapes(pair_count)
    if len(shapes) != pair_count:
        return None

    random.shuffle(palette)
    master_colors = palette[:pair_count]
    child_colors = palette[pair_count : 2 * pair_count]
    layers_by_pair = [random.randint(1, 3) for _ in range(pair_count)]

    specs: list[dict] = []
    for i, shape in enumerate(shapes):
        master_patch, _ = _render_object(
            shape,
            master_colors[i],
            bg,
            layers_by_pair[i],
            random.randint(0, 3),
        )
        child_patch, _ = _render_object(
            shape,
            child_colors[i],
            bg,
            0,
            random.randint(0, 3),
        )
        specs.append({"patch": child_patch, "pad": layers_by_pair[i] + 1})
        specs.append({"patch": master_patch, "pad": 1})

    random.shuffle(specs)
    specs.sort(key=lambda item: (-int(item["pad"]), -int(item["patch"].shape[0] * item["patch"].shape[1])))

    grid = np.full((h, w), int(bg), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    for spec in specs:
        if not _place_patch(grid, reserved, spec["patch"], int(spec["pad"])):
            return None

    if _get_bg(grid) != int(bg):
        return None

    objects = _extract_objects(grid, bg)
    if len(objects) != 2 * pair_count:
        return None
    try:
        pairs = _pair_objects(objects)
        if len(pairs) != pair_count:
            return None
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _build_candidate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid puzzle instance")
