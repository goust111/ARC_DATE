from __future__ import annotations

from collections import deque
import random

import numpy as np


TILE_COLORS = {1, 3}
OBJECT_COLORS = [2, 4, 5, 6, 7, 8, 9]
MAX_GEN_ATTEMPTS = 400


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _find_monochromatic_components(grid: np.ndarray, colors: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color not in colors:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(pts, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": int(len(pts)),
                }
            )
    return comps


def _period_values(
    grid: np.ndarray,
    visible: np.ndarray,
    row_period: int,
    col_period: int,
) -> dict[tuple[int, int], int] | None:
    values: dict[tuple[int, int], int] = {}
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if not visible[r, c]:
                continue
            key = (r % row_period, c % col_period)
            value = int(grid[r, c])
            prev = values.get(key)
            if prev is not None and prev != value:
                return None
            values[key] = value
    if len(values) != row_period * col_period:
        return None
    return values


def _reconstruct_periodic_grid(
    shape: tuple[int, int],
    values: dict[tuple[int, int], int],
    row_period: int,
    col_period: int,
) -> np.ndarray:
    h, w = shape
    out = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            out[r, c] = int(values[(r % row_period, c % col_period)])
    return out


def _infer_periodic_grid(grid: np.ndarray, visible: np.ndarray) -> tuple[np.ndarray, int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int, np.ndarray]] = []
    min_area: int | None = None
    visible_count = int(visible.sum())
    for row_period in range(1, h + 1):
        for col_period in range(1, w + 1):
            period_area = row_period * col_period
            if period_area > visible_count:
                continue
            values = _period_values(grid, visible, row_period, col_period)
            if values is None:
                continue
            filled = _reconstruct_periodic_grid(grid.shape, values, row_period, col_period)
            if not np.array_equal(filled[visible], grid[visible]):
                continue
            if min_area is None or period_area < min_area:
                min_area = period_area
                candidates = [(row_period, col_period, filled)]
            elif period_area == min_area:
                candidates.append((row_period, col_period, filled))
    assert candidates, "visible pixels must determine at least one full periodic reconstruction"
    first = candidates[0][2]
    assert all(np.array_equal(first, filled) for _, _, filled in candidates), "minimal-period reconstructions disagree"
    return first, int(candidates[0][0]), int(candidates[0][1])


def _extract_objects(grid: np.ndarray) -> list[dict]:
    colors = set(map(int, np.unique(grid))) - TILE_COLORS - {0}
    comps = _find_monochromatic_components(grid, colors)
    assert comps, "expected non-pattern rectangle objects"
    out: list[dict] = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        assert h >= 3 and w >= 3, "each rectangle must be at least 3x3"
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        color = int(comp["color"])
        mask = sub == color
        full = np.ones((h, w), dtype=bool)
        border = _border_mask(h, w)
        is_solid = np.array_equal(mask, full)
        is_frame_shape = np.array_equal(mask, border)
        assert is_solid or is_frame_shape, "non-pattern object must be a solid rectangle or a 1-cell frame"
        out.append(comp)
    return out


def _classify_object(grid: np.ndarray, pattern: np.ndarray, obj: dict) -> str:
    r0, c0, r1, c1 = obj["bbox"]
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    pat = pattern[r0 : r1 + 1, c0 : c1 + 1]
    h, w = sub.shape
    color = int(obj["color"])
    mask = sub == color
    if np.all(mask):
        return "solid"
    border = _border_mask(h, w)
    assert np.array_equal(mask, border), "frame classification expects exact border occupancy"
    assert np.array_equal(sub[1:-1, 1:-1], pat[1:-1, 1:-1]), "frame interior must reveal the periodic tile"
    return "frame"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify rectangular occluders, restore tiles under solid blocks, and erase hollow-frame borders.

    Input structure:
    - The whole grid is a blue/green periodic tile pattern using colors 1 and 3.
    - Several non-pattern monochrome rectangles sit on top of that tile.
    - Each rectangle is either a fully filled solid block or a one-cell-thick hollow frame whose interior still shows the tile.

    Rule:
    - Find every 4-connected non-pattern object and treat its bounding box as one rectangle.
    - Infer the smallest row/column periodic background pattern from pixels outside all rectangle bounding boxes.
    - If a rectangle is fully filled by its object color, replace its whole area with the inferred tile; if it is a hollow frame whose interior already matches that tile, erase only its one-cell border to black 0.

    Key insight:
    - The rectangle type is a branch decision, not a ranking cue: solid rectangles hide the tile and must be reconstructed, while hollow rectangles already expose the tile inside and only contribute removable borders.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects = _extract_objects(g)
    blocked = np.zeros(g.shape, dtype=bool)
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True

    visible = ~blocked
    assert visible.any(), "need visible tile pixels outside the rectangles"
    assert set(map(int, np.unique(g[visible]))) <= TILE_COLORS, "only tile colors may remain outside rectangle boxes"

    pattern, _rp, _cp = _infer_periodic_grid(g, visible)

    out = g.copy()
    for obj in objects:
        kind = _classify_object(g, pattern, obj)
        r0, c0, r1, c1 = obj["bbox"]
        if kind == "solid":
            out[r0 : r1 + 1, c0 : c1 + 1] = pattern[r0 : r1 + 1, c0 : c1 + 1]
        else:
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            border = _border_mask(h, w)
            sub = out[r0 : r1 + 1, c0 : c1 + 1]
            sub[border] = 0
    return out


def _tile_min_period(tile: np.ndarray) -> tuple[int, int]:
    h, w = tile.shape
    best = (h, w)
    best_area = h * w
    for row_period in range(1, h + 1):
        for col_period in range(1, w + 1):
            if row_period == h and col_period == w:
                continue
            ok = True
            for r in range(h):
                for c in range(w):
                    if int(tile[r, c]) != int(tile[r % row_period, c % col_period]):
                        ok = False
                        break
                if not ok:
                    break
            if ok and row_period * col_period < best_area:
                best = (row_period, col_period)
                best_area = row_period * col_period
    return best


def _random_tile() -> np.ndarray:
    for _ in range(200):
        h = random.choice([2, 3])
        w = random.choice([2, 3])
        tile = np.random.choice([1, 3], size=(h, w)).astype(int)
        if len(set(map(int, tile.ravel()))) < 2:
            continue
        if _tile_min_period(tile) == (h, w):
            return tile
    raise RuntimeError("failed to sample a non-degenerate tile")


def _tile_fill(shape: tuple[int, int], tile: np.ndarray, offset: tuple[int, int]) -> np.ndarray:
    h, w = shape
    th, tw = tile.shape
    roff, coff = offset
    out = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            out[r, c] = int(tile[(r + roff) % th, (c + coff) % tw])
    return out


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _all_residues_visible(blocked: np.ndarray, row_period: int, col_period: int) -> bool:
    seen = np.zeros((row_period, col_period), dtype=bool)
    for r in range(blocked.shape[0]):
        for c in range(blocked.shape[1]):
            if not blocked[r, c]:
                seen[r % row_period, c % col_period] = True
    return bool(seen.all())


def _apply_objects(base: np.ndarray, objects: list[dict]) -> np.ndarray:
    grid = base.copy()
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        color = int(obj["color"])
        if obj["kind"] == "solid":
            grid[r0 : r1 + 1, c0 : c1 + 1] = color
        else:
            grid[r0, c0 : c1 + 1] = color
            grid[r1, c0 : c1 + 1] = color
            grid[r0 : r1 + 1, c0] = color
            grid[r0 : r1 + 1, c1] = color
    return grid


def _expected_output(base: np.ndarray, objects: list[dict]) -> np.ndarray:
    out = base.copy()
    for obj in objects:
        if obj["kind"] != "frame":
            continue
        r0, c0, r1, c1 = obj["bbox"]
        out[r0, c0 : c1 + 1] = 0
        out[r1, c0 : c1 + 1] = 0
        out[r0 : r1 + 1, c0] = 0
        out[r0 : r1 + 1, c1] = 0
    return out


def _shortcut_outputs(inp: np.ndarray, base: np.ndarray, objects: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    all_as_solid = base.copy()
    all_as_frame = inp.copy()
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        all_as_frame[r0, c0 : c1 + 1] = 0
        all_as_frame[r1, c0 : c1 + 1] = 0
        all_as_frame[r0 : r1 + 1, c0] = 0
        all_as_frame[r0 : r1 + 1, c1] = 0
    return all_as_solid, all_as_frame


def _semantic_checks(
    inp: np.ndarray,
    base: np.ndarray,
    objects: list[dict],
    expected: np.ndarray,
    row_period: int,
    col_period: int,
) -> bool:
    solids = [obj for obj in objects if obj["kind"] == "solid"]
    frames = [obj for obj in objects if obj["kind"] == "frame"]
    if not solids or not frames:
        return False

    blocked = np.zeros(inp.shape, dtype=bool)
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True
    if not _all_residues_visible(blocked, row_period, col_period):
        return False

    outside = inp[~blocked]
    if len(set(map(int, outside.tolist()))) < 2:
        return False

    all_as_solid, all_as_frame = _shortcut_outputs(inp, base, objects)
    if np.array_equal(expected, all_as_solid):
        return False
    if np.array_equal(expected, all_as_frame):
        return False

    changed_to_zero = int(np.sum((inp != 0) & (expected == 0)))
    changed_from_object = int(np.sum((inp != expected) & np.isin(inp, OBJECT_COLORS)))
    if changed_to_zero == 0 or changed_from_object == 0:
        return False

    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    reconstructed, _, _ = _infer_periodic_grid(inp, ~blocked)
    if not np.array_equal(reconstructed, base):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(16, 22)
    width = random.randint(16, 22)
    tile = _random_tile()
    base = _tile_fill((height, width), tile, offset=(random.randrange(tile.shape[0]), random.randrange(tile.shape[1])))

    total = random.randint(4, 6)
    solid_count = random.randint(1, total - 1)
    frame_count = total - solid_count
    kinds = ["solid"] * solid_count + ["frame"] * frame_count
    random.shuffle(kinds)
    colors = random.sample(OBJECT_COLORS, total)

    objects: list[dict] = []
    blocked = np.zeros((height, width), dtype=bool)
    target_cover = random.uniform(0.18, 0.34)
    max_cover = int(target_cover * height * width)
    cover = 0

    for kind, color in zip(kinds, colors, strict=True):
        placed = False
        for _ in range(80):
            obj_h = random.randint(3, min(7, height - 2))
            obj_w = random.randint(3, min(7, width - 2))
            if cover + obj_h * obj_w > max_cover:
                continue
            r0 = random.randint(0, height - obj_h)
            c0 = random.randint(0, width - obj_w)
            box = (r0, c0, r0 + obj_h - 1, c0 + obj_w - 1)
            if any(not _boxes_separated(box, other["bbox"], gap=1) for other in objects):
                continue
            objects.append({"kind": kind, "color": color, "bbox": box})
            blocked[r0 : r0 + obj_h, c0 : c0 + obj_w] = True
            cover += obj_h * obj_w
            placed = True
            break
        if not placed:
            return None

    true_rp, true_cp = tile.shape
    if not _all_residues_visible(blocked, true_rp, true_cp):
        return None

    inp = _apply_objects(base, objects)
    expected = _expected_output(base, objects)

    if not _semantic_checks(inp, base, objects, expected, true_rp, true_cp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GEN_ATTEMPTS} attempts")
