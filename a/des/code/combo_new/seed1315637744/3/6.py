from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
KINDS = ("lr", "ud", "rot")
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _transform_coord(kind: str, shape: tuple[int, int], pos: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    r, c = pos
    if kind == "lr":
        return r, w - 1 - c
    if kind == "ud":
        return h - 1 - r, c
    if kind == "rot":
        return h - 1 - r, w - 1 - c
    raise AssertionError(kind)


def _transform_array(arr: np.ndarray, kind: str) -> np.ndarray:
    if kind == "lr":
        return np.fliplr(arr)
    if kind == "ud":
        return np.flipud(arr)
    if kind == "rot":
        return np.rot90(arr, 2)
    raise AssertionError(kind)


def _symmetry_mismatch(mask: np.ndarray, kind: str) -> int:
    return int(np.count_nonzero(mask != _transform_array(mask, kind)))


def _bbox_of_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_of_mask(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    q: deque[tuple[int, int]] = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                cell = (nr, nc)
                if cell not in seen:
                    seen.add(cell)
                    q.append(cell)
    return len(seen) == int(mask.sum())


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _majority_color(patch: np.ndarray) -> int:
    vals = [int(v) for v in patch.ravel() if int(v) != BG]
    counts = Counter(vals)
    assert counts
    return counts.most_common(1)[0][0]


def _unique_anomaly(patch: np.ndarray, main_color: int) -> tuple[int, int]:
    coords = np.argwhere((patch != BG) & (patch != main_color))
    assert coords.shape[0] == 1
    return tuple(map(int, coords[0]))


def _repair_signature(
    patch: np.ndarray,
    main_color: int,
    anomaly_pos: tuple[int, int],
    kind: str,
) -> tuple[int, int, int, int]:
    support = patch != BG
    q = _transform_coord(kind, patch.shape, anomaly_pos)
    q_val = int(patch[q])
    repaired = support.copy()
    repaired[anomaly_pos] = True

    penalty = 0
    branch = 0
    if q == anomaly_pos:
        penalty += 4
    elif q_val == BG:
        repaired[q] = True
        branch = 1
    elif q_val != main_color:
        penalty += 8

    residual = _symmetry_mismatch(repaired, kind)
    original = _symmetry_mismatch(support, kind)
    return residual, penalty, original, branch


def _infer_kind(patch: np.ndarray) -> str:
    main_color = _majority_color(patch)
    anomaly_pos = _unique_anomaly(patch, main_color)
    scored = sorted((_repair_signature(patch, main_color, anomaly_pos, kind), kind) for kind in KINDS)
    return scored[0][1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each almost-symmetric object and read out the anomaly colors from left to right.

    Input structure:
    - The background is black.
    - Several disconnected foreground objects are scattered on the canvas.
    - Each object is mostly one color and nearly matches one symmetry type:
      left-right reflection, up-down reflection, or 180-degree rotation.
    - Exactly one non-black pixel in each object has a different color and is the
      anomaly whose pre-repair color must be reported.

    Rule:
    - Split the grid into connected foreground objects and crop each object's bbox.
    - For each object, find its majority color and the unique differently colored pixel.
    - Test the three candidate symmetries; the intended one is the transform that
      becomes most consistent after restoring that anomaly orbit with the majority color.
    - Collect the anomaly colors, ordered by object bbox left edge from left to right,
      and output them as a single-row grid.

    Key insight:
    - The output is not the repaired drawing: symmetry is only an intermediate cue
      used to identify which unique odd-colored pixel belongs to each object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    comps = _components(g != BG)
    assert comps

    extracted: list[tuple[int, int, int]] = []
    for comp in comps:
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        main_color = _majority_color(patch)
        anomaly_pos = _unique_anomaly(patch, main_color)
        _infer_kind(patch)
        extracted.append((c0, r0, int(patch[anomaly_pos])))

    extracted.sort()
    return np.array([[color for _c0, _r0, color in extracted]], dtype=int)


def _orbit_list(kind: str, h: int, w: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[tuple[int, int], ...]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            q = _transform_coord(kind, (h, w), (r, c))
            orbit = tuple(sorted({(r, c), q}))
            seen.update(orbit)
            out.append(orbit)
    return out


def _orbit_is_bridge(orbit: tuple[tuple[int, int], ...]) -> bool:
    if len(orbit) == 1:
        return True
    (r0, c0), (r1, c1) = orbit
    return abs(r0 - r1) + abs(c0 - c1) == 1


def _orbit_touches(cells: set[tuple[int, int]], orbit: tuple[tuple[int, int], ...]) -> bool:
    for r, c in orbit:
        for dr, dc in NEIGHBORS4:
            if (r + dr, c + dc) in cells:
                return True
    return False


def _shape_dims(kind: str) -> tuple[int, int]:
    if kind == "rot":
        choices = [5, 5, 7]
        return random.choice(choices), random.choice(choices)
    height = random.randint(4, 6)
    width = random.randint(4, 6)
    if height < 5 and width < 5:
        if random.random() < 0.5:
            height = 5
        else:
            width = 5
    return height, width


def _generate_base_mask(kind: str) -> np.ndarray | None:
    for _ in range(200):
        h, w = _shape_dims(kind)
        orbits = _orbit_list(kind, h, w)
        bridges = [orbit for orbit in orbits if _orbit_is_bridge(orbit)]
        if not bridges:
            continue

        cells = set(random.choice(bridges))
        target_area = random.randint(max(7, len(cells) + 4), min(16, h * w - 1))
        while len(cells) < target_area:
            candidates = [
                orbit
                for orbit in orbits
                if not set(orbit).issubset(cells) and _orbit_touches(cells, orbit)
            ]
            if not candidates:
                break
            cells.update(random.choice(candidates))
            if len(cells) >= target_area and random.random() < 0.35:
                break

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not _is_connected(mask):
            continue
        mask = _crop_mask(mask)
        if int(mask.sum()) < 7 or min(mask.shape) < 3:
            continue
        if not np.array_equal(mask, _transform_array(mask, kind)):
            continue
        if any(np.array_equal(mask, _transform_array(mask, other)) for other in KINDS if other != kind):
            continue
        if min(_symmetry_mismatch(mask, other) for other in KINDS if other != kind) < 4:
            continue
        return mask
    return None


def _candidate_pairs(mask: np.ndarray, kind: str, mode: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    random.shuffle(coords)
    out: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for p in coords:
        q = _transform_coord(kind, mask.shape, p)
        if q == p or not mask[q]:
            continue
        if mode == "fill":
            shrunk = mask.copy()
            shrunk[q] = False
            if not _is_connected(shrunk):
                continue
            if _crop_mask(shrunk).shape != mask.shape:
                continue
        out.append((p, q))
    return out


def _corrupt_patch(mask: np.ndarray, kind: str, mode: str, main_color: int, anomaly_color: int) -> np.ndarray | None:
    pairs = _candidate_pairs(mask, kind, mode)
    if not pairs:
        return None
    p, q = random.choice(pairs)
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = main_color
    patch[p] = anomaly_color
    if mode == "fill":
        patch[q] = BG

    if len(_components(patch != BG)) != 1:
        return None
    if _majority_color(patch) != main_color:
        return None
    if _unique_anomaly(patch, main_color) != p:
        return None
    scored = sorted((_repair_signature(patch, main_color, p, candidate), candidate) for candidate in KINDS)
    if scored[0][1] != kind or scored[0][0] == scored[1][0]:
        return None
    return patch


def _allocate_columns(widths: list[int]) -> tuple[int, list[int]] | None:
    n = len(widths)
    min_total = 1 + sum(widths) + 2 * (n - 1) + 1
    if min_total > 30:
        return None
    canvas_w = random.randint(min_total, 30)
    slack = canvas_w - min_total
    extras = [0] * (n + 1)
    for _ in range(slack):
        extras[random.randrange(n + 1)] += 1

    cols: list[int] = []
    cur = 1 + extras[0]
    for i, width in enumerate(widths):
        cols.append(cur)
        cur += width
        if i < n - 1:
            cur += 2 + extras[i + 1]
    return canvas_w, cols


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    n_objects = random.choice([3, 4])

    kinds = list(KINDS)
    while len(kinds) < n_objects:
        kinds.append(random.choice(KINDS))
    random.shuffle(kinds)

    modes = ["replace", "fill"] * 2
    random.shuffle(modes)
    modes = modes[:n_objects]

    anomaly_colors = random.sample(range(1, 10), n_objects)
    main_pool = [c for c in range(1, 10) if c not in anomaly_colors]
    objects: list[tuple[np.ndarray, int]] = []

    for i, (kind, mode) in enumerate(zip(kinds, modes)):
        mask = _generate_base_mask(kind)
        if mask is None:
            return None
        main_color = random.choice(main_pool)
        patch = _corrupt_patch(mask, kind, mode, main_color, anomaly_colors[i])
        if patch is None:
            return None
        objects.append((patch, anomaly_colors[i]))

    widths = [patch.shape[1] for patch, _ in objects]
    layout = _allocate_columns(widths)
    if layout is None:
        return None
    canvas_w, cols = layout

    max_h = max(patch.shape[0] for patch, _ in objects)
    min_h = max(max_h + 1, 10)
    if min_h > 30:
        return None
    canvas_h = random.randint(min_h, 30)

    grid = np.zeros((canvas_h, canvas_w), dtype=int)
    for (patch, _anomaly), c0 in zip(objects, cols):
        h, w = patch.shape
        r0 = random.randint(0, canvas_h - h)
        window = grid[r0 : r0 + h, c0 : c0 + w]
        if np.any((window != BG) & (patch != BG)):
            return None
        window[patch != BG] = patch[patch != BG]

    expected = np.array([[anomaly for _patch, anomaly in objects]], dtype=int)
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        built = _build_attempt()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid symmetry-anomaly puzzle")
