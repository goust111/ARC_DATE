from __future__ import annotations

import numpy as np
from scipy.ndimage import label


ORANGE = 7
GREY = 5
YELLOW = 4
TEAL = 8

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

_RNG = np.random.default_rng()
_GEN_COUNTER = 0

GRID_MIN = 20
GRID_MAX = 30
MAX_GEN_ATTEMPTS = 400
OBJECT_ATTEMPTS = 80


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_from_mask(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    structure = np.ones((3, 3), dtype=int) if connectivity == 8 else np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    labeled, n = label(mask.astype(bool), structure=structure)
    return [np.argwhere(labeled == idx) for idx in range(1, int(n) + 1)]


def _grey_components(grid: np.ndarray) -> list[np.ndarray]:
    return _components_from_mask(grid == GREY, connectivity=8)


def _adjacent_background_cells(solid: np.ndarray) -> list[tuple[int, int]]:
    h, w = solid.shape
    cells = set()
    for r, c in np.argwhere(solid):
        rr, cc = int(r), int(c)
        for dr, dc in DIR8:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and not solid[nr, nc]:
                cells.add((nr, nc))
    return sorted(cells)


def _analyze_component(shape: tuple[int, int], coords: np.ndarray) -> dict[str, object]:
    solid = np.zeros(shape, dtype=bool)
    solid[coords[:, 0], coords[:, 1]] = True

    bg_components = _components_from_mask(~solid, connectivity=8)
    if len(bg_components) == 2:
        return {"kind": "separator"}
    if len(bg_components) != 1:
        return {"kind": "none"}

    repairs: list[tuple[tuple[int, int], np.ndarray]] = []
    for cell in _adjacent_background_cells(solid):
        solid[cell] = True
        split_components = _components_from_mask(~solid, connectivity=8)
        solid[cell] = False
        if len(split_components) != 2:
            continue
        size0 = int(split_components[0].shape[0])
        size1 = int(split_components[1].shape[0])
        if size0 == size1:
            continue
        smaller = split_components[0] if size0 < size1 else split_components[1]
        repairs.append((cell, smaller))

    if len(repairs) == 1:
        fill, smaller = repairs[0]
        return {"kind": "repairable", "fill": fill, "smaller": smaller}
    return {"kind": "none"}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the uniquely leaky grey separator, then paint the smaller isolated orange side yellow.

    Input structure:
    - An orange background contains several grey 8-connected components.
    - Some grey components already disconnect the background by themselves.
    - Some grey components have exactly one adjacent orange cell whose addition would make them disconnect the background.
    - Other grey components are distractors and stay irrelevant.

    Rule:
    - Inspect each grey 8-connected component in isolation, with every other grey component removed.
    - If that isolated component already splits the orange background into two regions, leave it unchanged.
    - Otherwise test each adjacent orange cell; when exactly one such cell makes the isolated component become a separator, recolor that cell teal.
    - For that repaired component, recolor the smaller of the two newly separated background regions yellow, while preserving all grey pixels from every component.

    Key insight:
    - The missing leak cannot be recognized from local appearance alone; it is the unique one-cell repair that changes whole-grid background connectivity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for coords in _grey_components(g):
        analysis = _analyze_component(g.shape, coords)
        if analysis["kind"] != "repairable":
            continue
        fill_r, fill_c = analysis["fill"]
        out[fill_r, fill_c] = TEAL
        smaller = analysis["smaller"]
        for r, c in smaller:
            rr, cc = int(r), int(c)
            if out[rr, cc] == ORANGE:
                out[rr, cc] = YELLOW
    return out


def _normalize_coords(coords: set[tuple[int, int]]) -> np.ndarray:
    arr = np.asarray(sorted(coords), dtype=int)
    r0, c0, _, _ = _bbox_coords(arr)
    arr[:, 0] -= r0
    arr[:, 1] -= c0
    return arr


def _ring_local(h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)}


def _gap_cell(h: int, w: int, side: str, offset: int) -> tuple[int, int]:
    if side == "top":
        return 0, offset
    if side == "bottom":
        return h - 1, offset
    if side == "left":
        return offset, 0
    if side == "right":
        return offset, w - 1
    raise ValueError("invalid side")


def _make_ring_kind(kind: str) -> np.ndarray:
    if kind == "double_gap_ring":
        h = int(_RNG.integers(5, 9))
        w = int(_RNG.integers(5, 9))
    else:
        h = int(_RNG.integers(4, 9))
        w = int(_RNG.integers(4, 9))
    pts = _ring_local(h, w)

    sides = ["top", "bottom", "left", "right"]
    if kind == "repair_ring":
        side = str(_RNG.choice(sides))
        offset = int(_RNG.integers(1, w - 1)) if side in ("top", "bottom") else int(_RNG.integers(1, h - 1))
        pts.remove(_gap_cell(h, w, side, offset))
    elif kind == "double_gap_ring":
        side = str(_RNG.choice(sides))
        if side in ("top", "bottom"):
            start = int(_RNG.integers(1, w - 2))
        else:
            start = int(_RNG.integers(1, h - 2))
        pts.remove(_gap_cell(h, w, side, start))
        pts.remove(_gap_cell(h, w, side, start + 1))
    elif kind != "separator_ring":
        raise ValueError("invalid ring kind")

    return np.asarray(sorted(pts), dtype=int)


def _make_blob_local() -> np.ndarray:
    presets = [
        {(0, 0), (0, 1), (1, 0), (1, 1)},
        {(0, 0), (1, 0), (2, 0), (2, 1)},
        {(0, 0), (0, 1), (0, 2), (1, 1)},
        {(0, 1), (1, 0), (1, 1), (2, 1)},
        {(0, 0), (1, 1), (2, 0), (2, 1)},
        {(0, 0), (0, 1), (1, 1), (1, 2)},
    ]
    pts = presets[int(_RNG.integers(0, len(presets)))]
    return _normalize_coords(set(pts))


def _place_local_object(
    blocked: np.ndarray,
    allowed_box: tuple[int, int, int, int],
    local_coords: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    r0, c0, r1, c1 = allowed_box
    h = int(local_coords[:, 0].max()) + 1
    w = int(local_coords[:, 1].max()) + 1
    if r1 - r0 + 1 < h or c1 - c0 + 1 < w:
        return None

    for _ in range(OBJECT_ATTEMPTS):
        top = int(_RNG.integers(r0, r1 - h + 2))
        left = int(_RNG.integers(c0, c1 - w + 2))
        placed = local_coords + np.array([top, left], dtype=int)
        bbox = (top, left, top + h - 1, left + w - 1)
        er0 = max(0, bbox[0] - 1)
        ec0 = max(0, bbox[1] - 1)
        er1 = min(blocked.shape[0] - 1, bbox[2] + 1)
        ec1 = min(blocked.shape[1] - 1, bbox[3] + 1)
        if blocked[er0 : er1 + 1, ec0 : ec1 + 1].any():
            continue
        return placed, bbox
    return None


def _mark_rect(blocked: np.ndarray, bbox: tuple[int, int, int, int], margin: int = 1) -> None:
    r0 = max(0, bbox[0] - margin)
    c0 = max(0, bbox[1] - margin)
    r1 = min(blocked.shape[0] - 1, bbox[2] + margin)
    c1 = min(blocked.shape[1] - 1, bbox[3] + margin)
    blocked[r0 : r1 + 1, c0 : c1 + 1] = True


def _build_wall(
    shape: tuple[int, int],
    repairable: bool,
) -> tuple[np.ndarray, tuple[int, int, int, int], np.ndarray]:
    h, w = shape
    blocked = np.zeros(shape, dtype=bool)
    orient = str(_RNG.choice(["v", "h"]))
    small_span = int(_RNG.integers(2, 6))
    side = str(_RNG.choice(["near_start", "near_end"]))

    if orient == "v":
        c = small_span if side == "near_start" else w - 1 - small_span
        gap_r = int(_RNG.integers(2, h - 2))
        pts = {(r, c) for r in range(h)}
        if repairable:
            pts.remove((gap_r, c))
            pts.add((gap_r, c + 1 if side == "near_start" else c - 1))
        coords = np.asarray(sorted(pts), dtype=int)
        if side == "near_start":
            blocked[:, : c + 3] = True
            allowed = (1, c + 3, h - 2, w - 2)
        else:
            blocked[:, c - 2 :] = True
            allowed = (1, 1, h - 2, c - 3)
    else:
        r = small_span if side == "near_start" else h - 1 - small_span
        gap_c = int(_RNG.integers(2, w - 2))
        pts = {(r, c) for c in range(w)}
        if repairable:
            pts.remove((r, gap_c))
            pts.add((r + 1 if side == "near_start" else r - 1, gap_c))
        coords = np.asarray(sorted(pts), dtype=int)
        if side == "near_start":
            blocked[: r + 3, :] = True
            allowed = (r + 3, 1, h - 2, w - 2)
        else:
            blocked[r - 2 :, :] = True
            allowed = (1, 1, r - 3, w - 2)

    return coords, allowed, blocked


def _specs_for_family(family: int) -> tuple[list[str], str | None]:
    if family == 0:
        specs = ["repair_ring", "separator_ring", "double_gap_ring"]
        extra = "blob" if _RNG.random() < 0.7 else None
        return specs, extra
    if family == 1:
        specs = ["separator_ring", "double_gap_ring"]
        extra = "blob" if _RNG.random() < 0.5 else None
        return specs, extra
    if family == 2:
        specs = ["repair_ring", "repair_ring", "double_gap_ring"]
        extra = "separator_ring" if _RNG.random() < 0.6 else "blob"
        return specs, extra
    specs = ["repair_ring", "double_gap_ring"]
    extra = "blob" if _RNG.random() < 0.8 else "separator_ring"
    return specs, extra


def _local_shape(kind: str) -> np.ndarray:
    if kind == "blob":
        return _make_blob_local()
    return _make_ring_kind(kind)


def _expected_kind(kind: str) -> str:
    if kind == "repair_ring":
        return "repairable"
    if kind == "separator_ring":
        return "separator"
    return "none"


def _place_local_kind(
    grid_shape: tuple[int, int],
    blocked: np.ndarray,
    allowed_box: tuple[int, int, int, int],
    kind: str,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    for _ in range(OBJECT_ATTEMPTS):
        local = _local_shape(kind)
        placed = _place_local_object(blocked, allowed_box, local)
        if placed is None:
            continue
        coords, bbox = placed
        if kind != "blob" or _analyze_component(grid_shape, coords)["kind"] == "none":
            return coords, bbox
    return None


def _validate_scene(grid: np.ndarray, objects: list[dict[str, object]]) -> bool:
    comps = _grey_components(grid)
    if len(comps) != len(objects):
        return False

    comp_at: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(comps):
        for r, c in comp:
            comp_at[(int(r), int(c))] = idx

    analyses: list[dict[str, object]] = []
    used = set()
    for obj in objects:
        rep = obj["rep"]
        idx = comp_at.get(rep)
        if idx is None or idx in used:
            return False
        used.add(idx)
        analysis = _analyze_component(grid.shape, comps[idx])
        if analysis["kind"] != obj["kind"]:
            return False
        analyses.append(analysis)

    out = main(grid)
    yellow_allowed = np.zeros_like(grid, dtype=bool)
    teal_cells = set()
    for analysis in analyses:
        if analysis["kind"] != "repairable":
            continue
        fill = tuple(map(int, analysis["fill"]))
        teal_cells.add(fill)
        smaller = analysis["smaller"]
        yellow_allowed[smaller[:, 0], smaller[:, 1]] = True

    if len(teal_cells) == 0:
        return False
    if int(np.sum(out == TEAL)) != len(teal_cells):
        return False
    for cell in teal_cells:
        if int(out[cell]) != TEAL:
            return False
    yellow_mask = out == YELLOW
    if not yellow_mask.any():
        return False
    if np.any(yellow_mask & ~yellow_allowed):
        return False
    if np.any((grid == GREY) & yellow_mask):
        return False
    return True


def _gen_attempt(family: int) -> np.ndarray | None:
    h = int(_RNG.integers(GRID_MIN, GRID_MAX + 1))
    w = int(_RNG.integers(GRID_MIN, GRID_MAX + 1))
    grid = np.full((h, w), ORANGE, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    allowed_box = (1, 1, h - 2, w - 2)
    objects: list[dict[str, object]] = []

    if family in (1, 3):
        wall_coords, allowed_box, wall_blocked = _build_wall((h, w), repairable=(family == 1))
        grid[wall_coords[:, 0], wall_coords[:, 1]] = GREY
        blocked |= wall_blocked
        objects.append({"coords": wall_coords, "rep": tuple(map(int, wall_coords[0])), "kind": "repairable" if family == 1 else "separator"})

    specs, extra = _specs_for_family(family)
    if extra is not None:
        specs.append(extra)

    for kind in specs:
        placed = _place_local_kind((h, w), blocked, allowed_box, kind)
        if placed is None:
            return None
        coords, bbox = placed
        grid[coords[:, 0], coords[:, 1]] = GREY
        _mark_rect(blocked, bbox, margin=1)
        objects.append({"coords": coords, "rep": tuple(map(int, coords[0])), "kind": _expected_kind(kind)})

    if len({tuple(obj["rep"]) for obj in objects}) != len(objects):
        return None
    if not _validate_scene(grid, objects):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    for attempt in range(MAX_GEN_ATTEMPTS):
        family = (_GEN_COUNTER + attempt) % 4
        grid = _gen_attempt(family)
        if grid is not None:
            _GEN_COUNTER += 1
            return grid
    raise RuntimeError("failed to generate a valid leaky-wall sample")
