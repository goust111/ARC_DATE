from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import find_components, get_bg
except Exception:  # pragma: no cover
    _ROOT = Path(__file__).resolve()
    _UTILS_DIR = None
    for parent in _ROOT.parents:
        candidate = parent / "new" / "solvers"
        if (candidate / "arc_utils.py").exists():
            _UTILS_DIR = candidate
            break
    if _UTILS_DIR is None:
        _UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
    if str(_UTILS_DIR) not in sys.path:
        sys.path.append(str(_UTILS_DIR))
    from arc_utils import find_components, get_bg  # type: ignore


# Mapping for the 4 possible corner-bracket openings.
# UL -> top-left corner cells, opening to the South-East (SE)
# UR -> top-right corner cells, opening to the South-West (SW)
# LL -> bottom-left corner cells, opening to the North-East (NE)
# LR -> bottom-right corner cells, opening to the North-West (NW)


def _bracket_cells(r0: int, c0: int, r1: int, c1: int, orientation: str) -> set[tuple[int, int]]:
    """All cells of an axis-aligned unit-thickness corner bracket.

    orientation: UL/UR/LL/LR
    """
    if orientation == "UL":
        return {(r0, c) for c in range(c0, c1 + 1)} | {(r, c0) for r in range(r0, r1 + 1)}
    if orientation == "UR":
        return {(r0, c) for c in range(c0, c1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)}
    if orientation == "LL":
        return {(r1, c) for c in range(c0, c1 + 1)} | {(r, c0) for r in range(r0, r1 + 1)}
    if orientation == "LR":
        return {(r1, c) for c in range(c0, c1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)}
    raise ValueError(f"invalid bracket orientation: {orientation}")


def _bracket_dir(orientation: str) -> tuple[int, int]:
    if orientation == "UL":
        return 1, 1
    if orientation == "UR":
        return 1, -1
    if orientation == "LL":
        return -1, 1
    if orientation == "LR":
        return -1, -1
    raise ValueError(f"invalid bracket orientation: {orientation}")


def _bracket_anchor(orientation: str, bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    """Inner corner touching the missing diagonal inside the bracket."""
    r0, c0, r1, c1 = map(int, bbox)
    if orientation == "UL":
        return r0 + 1, c0 + 1
    if orientation == "UR":
        return r0 + 1, c1 - 1
    if orientation == "LL":
        return r1 - 1, c0 + 1
    if orientation == "LR":
        return r1 - 1, c1 - 1
    raise ValueError(f"invalid bracket orientation: {orientation}")


def _bracket_inner_bbox(orientation: str, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Inner area that can hold inserted objects for this bracket."""
    r0, c0, r1, c1 = map(int, bbox)
    if orientation == "UL":
        return r0 + 1, c0 + 1, r1, c1
    if orientation == "UR":
        return r0 + 1, c0, r1, c1 - 1
    if orientation == "LL":
        return r0, c0 + 1, r1 - 1, c1
    if orientation == "LR":
        return r0, c0, r1 - 1, c1 - 1
    raise ValueError(f"invalid bracket orientation: {orientation}")


def _ray_cells(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    """Return all cells from start in a diagonal line until leaving the grid."""
    h, w = map(int, shape)
    r, c = map(int, start)
    dr, dc = map(int, direction)
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((int(r), int(c)))
        r += dr
        c += dc
    return cells


def _inside(r: int, c: int, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = map(int, bbox)
    return int(r0) <= int(r) <= int(r1) and int(c0) <= int(c) <= int(c1)


def _shape_fits(obj_bbox: tuple[int, int, int, int], inner_bbox: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = map(int, obj_bbox)
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    return (or1 - or0 + 1) <= (ir1 - ir0 + 1) and (oc1 - oc0 + 1) <= (ic1 - ic0 + 1)


def _bracket_component_info(component: dict) -> dict | None:
    """Return normalized bracket info when `component` is exactly a 1-thick corner."""
    r0, c0, r1, c1 = map(int, component["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 2 or w < 2:
        return None
    coords = {tuple(map(int, p)) for p in component["coords"]}
    if len(coords) != h + w - 1:
        return None

    for orientation in ("UL", "UR", "LL", "LR"):
        if coords == _bracket_cells(r0, c0, r1, c1, orientation):
            dr, dc = _bracket_dir(orientation)
            anchor = _bracket_anchor(orientation, (r0, c0, r1, c1))
            inner_bbox = _bracket_inner_bbox(orientation, (r0, c0, r1, c1))
            return {
                "bbox": (r0, c0, r1, c1),
                "orientation": orientation,
                "dr": dr,
                "dc": dc,
                "anchor": anchor,
                "inner_bbox": inner_bbox,
            }
    return None


def _align_shift(
    obj_bbox: tuple[int, int, int, int],
    orientation: str,
    inner_anchor: tuple[int, int],
) -> tuple[int, int]:
    or0, oc0, or1, oc1 = map(int, obj_bbox)
    ar, ac = map(int, inner_anchor)
    if orientation == "UL":
        return ar - or0, ac - oc0
    if orientation == "UR":
        return ar - or0, ac - oc1
    if orientation == "LL":
        return ar - or1, ac - oc0
    if orientation == "LR":
        return ar - or1, ac - oc1
    raise ValueError(f"invalid bracket orientation: {orientation}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the first size-fitting object on each corner-bracket diagonal sightline into that bracket.

    Input structure:
    - Background is the most frequent color.
    - Several monochromatic 4-connected corner brackets (single-pixel L shapes) of any size.
    - One or more disconnected foreground objects, each a 4-connected component not belonging to brackets.

    Rule:
    - For each bracket, build a diagonal sightline from its inner opening corner along its open direction.
    - Scan objects intersecting the line in order of distance; skip those whose bbox is larger than the bracket interior.
    - Capture the first object whose bbox fits, move the whole object so its bbox corner aligns with the bracket opening corner, and erase its original pixels.
    - Objects not captured remain unchanged.

    Key insight:
    - Each bracket is a directional, size-constrained selector: only the first object on the opening diagonal that fits the bracket's inner area is inserted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "main expects a 2D grid"

    bg = get_bg(g)

    bracket_components = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    bracket_cells: set[tuple[int, int]] = set()
    brackets: list[dict] = []
    for component in bracket_components:
        info = _bracket_component_info(component)
        if info is None:
            continue
        for p in component["coords"]:
            bracket_cells.add((int(p[0]), int(p[1])))
        brackets.append(info)

    assert brackets, "expected at least one corner bracket"

    objects: list[dict] = []
    cell_to_obj: dict[tuple[int, int], int] = {}
    for component in find_components(g, bg=bg, connectivity=4, monochromatic=False, min_area=1):
        coords = [tuple(map(int, p)) for p in component["coords"]]
        if any(coord in bracket_cells for coord in coords):
            continue
        if not coords:
            continue
        obj_id = len(objects)
        objs = {
            "bbox": tuple(map(int, component["bbox"])),
            "coords": coords,
        }
        objects.append(objs)
        for coord in coords:
            cell_to_obj[coord] = obj_id

    captures: dict[int, tuple[int, int]] = {}
    captured_by_obj: set[int] = set()
    for bracket in sorted(brackets, key=lambda item: item["bbox"]):
        orientation = str(bracket["orientation"])
        ar, ac = bracket["anchor"]
        dr = int(bracket["dr"])
        dc = int(bracket["dc"])
        inner_bbox = tuple(map(int, bracket["inner_bbox"]))
        inner_h = inner_bbox[2] - inner_bbox[0] + 1
        inner_w = inner_bbox[3] - inner_bbox[1] + 1

        seen: set[int] = set()
        for r, c in _ray_cells(g.shape, (ar, ac), (dr, dc)):
            obj_id = cell_to_obj.get((int(r), int(c)))
            if obj_id is None or obj_id in seen or obj_id in captured_by_obj:
                continue
            seen.add(obj_id)

            obj = objects[obj_id]
            ob = obj["bbox"]
            if not _shape_fits(ob, inner_bbox):
                continue

            shift_r, shift_c = _align_shift(ob, orientation, (ar, ac))
            for sr, sc in obj["coords"]:
                nr = int(sr + shift_r)
                nc = int(sc + shift_c)
                assert _inside(nr, nc, inner_bbox), "captured object must stay inside bracket interior"

            captures[obj_id] = (shift_r, shift_c)
            captured_by_obj.add(obj_id)
            break

    if not captures:
        return g.copy()

    out = g.copy()
    clear_cells: set[tuple[int, int]] = set()
    paint: dict[tuple[int, int], int] = {}

    for obj_id, (shift_r, shift_c) in captures.items():
        obj = objects[obj_id]
        for sr, sc in obj["coords"]:
            clear_cells.add((int(sr), int(sc)))

    for r, c in clear_cells:
        out[r, c] = int(bg)

    for obj_id, (shift_r, shift_c) in captures.items():
        obj = objects[obj_id]
        for sr, sc in obj["coords"]:
            nr = int(sr + shift_r)
            nc = int(sc + shift_c)
            assert 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1], "captured object must remain in bounds"
            if (nr, nc) not in clear_cells and out[nr, nc] != bg:
                raise AssertionError("destination overlaps a different unchanged object")

            prev = paint.get((nr, nc))
            color = int(g[sr, sc])
            if prev is not None and prev != color:
                raise AssertionError("multiple captures try to write different colors to one pixel")
            paint[(nr, nc)] = color

    for (nr, nc), color in paint.items():
        out[nr, nc] = color

    return out


# -------------------------- generator --------------------------

_GRID_MIN, _GRID_MAX = 12, 28
_ORIENTATIONS = ("UL", "UR", "LL", "LR")


def _occupied_ok(
    cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
    margin: int,
) -> bool:
    h, w = shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        for rr in range(r - margin, r + margin + 1):
            for cc in range(c - margin, c + margin + 1):
                if (rr, cc) in occupied:
                    return False
    return True


def _rect_cells(top: int, left: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(top, top + h) for c in range(left, left + w)}


def _ray_align_top_left(
    ray_cell: tuple[int, int],
    orientation: str,
    obj_h: int,
    obj_w: int,
) -> tuple[int, int]:
    r, c = ray_cell
    if orientation == "UL":
        return r, c
    if orientation == "UR":
        return r, c - obj_w + 1
    if orientation == "LL":
        return r - obj_h + 1, c
    if orientation == "LR":
        return r - obj_h + 1, c - obj_w + 1
    raise ValueError(f"invalid bracket orientation: {orientation}")


def _place_rect_on_ray(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    path: list[tuple[int, int]],
    orientation: str,
    color: int,
    obj_h: int,
    obj_w: int,
    *,
    start_index: int = 0,
    avoid: set[tuple[int, int]] | None = None,
) -> tuple[int, set[tuple[int, int]]] | None:
    if avoid is None:
        avoid = set()
    H, W = grid.shape
    for idx in range(start_index, len(path)):
        r, c = path[idx]
        tl_r, tl_c = _ray_align_top_left((r, c), orientation, obj_h, obj_w)
        cells = _rect_cells(tl_r, tl_c, obj_h, obj_w)
        if not cells:
            continue
        if any(cell in avoid for cell in cells):
            continue
        if not _occupied_ok(cells, occupied, (H, W), margin=1):
            continue
        for rr, cc in cells:
            grid[rr, cc] = int(color)
            occupied.add((rr, cc))
        return idx, cells
    return None


def _build_instance() -> np.ndarray | None:
    H = random.randint(_GRID_MIN, _GRID_MAX)
    W = random.randint(_GRID_MIN, _GRID_MAX)
    bg = random.randint(0, 9)
    colors = [c for c in range(10) if c != bg]
    bracket_color = random.choice(colors)
    object_colors = [c for c in colors if c != bracket_color]

    grid = np.full((H, W), int(bg), dtype=int)
    occupied: set[tuple[int, int]] = set()

    # Build brackets.
    n_brackets = random.choices([1, 2, 3], weights=[5, 3, 2], k=1)[0]
    bracket_infos: list[dict] = []
    ray_cells_all: list[tuple[int, int]] = []

    for _ in range(n_brackets):
        placed = False
        for _ in range(120):
            orientation = random.choice(_ORIENTATIONS)
            bh = random.randint(3, min(8, H - 1))
            bw = random.randint(3, min(8, W - 1))
            if bh >= H or bw >= W:
                continue

            r0_max = max(0, H - bh)
            c0_max = max(0, W - bw)
            if r0_max == 0 or c0_max == 0:
                continue

            r0 = random.randint(0, r0_max - 1)
            c0 = random.randint(0, c0_max - 1)

            cells = _bracket_cells(r0, c0, r0 + bh - 1, c0 + bw - 1, orientation)
            if not _occupied_ok(cells, occupied, (H, W), margin=1):
                continue

            for rr, cc in cells:
                grid[rr, cc] = int(bracket_color)
                occupied.add((rr, cc))

            dr, dc = _bracket_dir(orientation)
            inner = _bracket_inner_bbox(orientation, (r0, c0, r0 + bh - 1, c0 + bw - 1))
            ih = inner[2] - inner[0] + 1
            iw = inner[3] - inner[1] + 1
            if ih < 2 or iw < 2:
                for rr, cc in cells:
                    grid[rr, cc] = int(bg)
                    occupied.discard((rr, cc))
                break

            path = _ray_cells((H, W), _bracket_anchor(orientation, (r0, c0, r0 + bh - 1, c0 + bw - 1)), (dr, dc))
            if len(path) < 4:
                for rr, cc in cells:
                    grid[rr, cc] = int(bg)
                    occupied.discard((rr, cc))
                continue

            anchor = _bracket_anchor(orientation, (r0, c0, r0 + bh - 1, c0 + bw - 1))
            bracket_infos.append(
                {
                    "orientation": orientation,
                    "r0": r0,
                    "c0": c0,
                    "r1": r0 + bh - 1,
                    "c1": c0 + bw - 1,
                    "dr": dr,
                    "dc": dc,
                    "anchor": anchor,
                    "inner_h": ih,
                    "inner_w": iw,
                    "path": path,
                    "bbox": (r0, c0, r0 + bh - 1, c0 + bw - 1),
                    "inner_bbox": inner,
                }
            )
            ray_cells_all.extend(path)
            placed = True
            break

        if not placed:
            return None

    if not bracket_infos:
        return None

    # Force a mixed set of branch patterns, including both capture and non-capture cases.
    modes: list[str] = []
    capture_modes = 0
    for _ in bracket_infos:
        mode = random.choices(
            ["capture", "skip_then_capture", "skip_only", "empty"],
            weights=[0.40, 0.35, 0.15, 0.10],
            k=1,
        )[0]
        modes.append(mode)
        if mode in {"capture", "skip_then_capture"}:
            capture_modes += 1

    if capture_modes == 0:
        # Guarantee at least one expected capture for semantic visibility.
        modes[0] = random.choice(("capture", "skip_then_capture"))
        if modes[0] in {"capture", "skip_then_capture"}:
            capture_modes = 1

    # A small set to avoid accidental extra captures from background noise.
    ray_block = {cell for cell in ray_cells_all}

    for info, mode in zip(bracket_infos, modes):
        orientation = info["orientation"]
        path = info["path"]
        ih = int(info["inner_h"])
        iw = int(info["inner_w"])

        if mode == "capture":
            placed = False
            for _ in range(70):
                obj_h = random.randint(1, ih)
                obj_w = random.randint(1, iw)
                if obj_h == 1 and obj_w == 1:
                    obj_h = 1
                color = random.choice(object_colors)
                res = _place_rect_on_ray(
                    grid,
                    occupied,
                    path,
                    orientation,
                    color,
                    obj_h,
                    obj_w,
                    start_index=0,
                )
                if res is None:
                    continue
                placed = True
                break
            if not placed:
                return None

        elif mode == "skip_then_capture":
            # One non-fit object first (must be larger than bracket interior in one dimension),
            # then a fit object further on the same ray.
            nonfit = None
            for _ in range(90):
                if random.random() < 0.5:
                    obj_h = random.randint(ih + 1, ih + 2)
                    obj_w = random.randint(1, iw)
                else:
                    obj_w = random.randint(iw + 1, iw + 2)
                    obj_h = random.randint(1, ih)

                color = random.choice(object_colors)
                nonfit = _place_rect_on_ray(
                    grid,
                    occupied,
                    path,
                    orientation,
                    color,
                    obj_h,
                    obj_w,
                    start_index=0,
                )
                if nonfit is not None:
                    break

            if nonfit is None:
                return None

            start = nonfit[0] + 1
            placed_fit = False
            for _ in range(70):
                obj_h = random.randint(1, ih)
                obj_w = random.randint(1, iw)
                color = random.choice(object_colors)
                fit = _place_rect_on_ray(
                    grid,
                    occupied,
                    path,
                    orientation,
                    color,
                    obj_h,
                    obj_w,
                    start_index=start,
                )
                if fit is None:
                    continue
                placed_fit = True
                break
            if not placed_fit:
                return None

        elif mode == "skip_only":
            placed = False
            for _ in range(70):
                if random.random() < 0.5:
                    obj_h = random.randint(ih + 1, ih + 2)
                    obj_w = random.randint(1, iw)
                else:
                    obj_w = random.randint(iw + 1, iw + 2)
                    obj_h = random.randint(1, ih)

                color = random.choice(object_colors)
                skip_only = _place_rect_on_ray(
                    grid,
                    occupied,
                    path,
                    orientation,
                    color,
                    obj_h,
                    obj_w,
                    start_index=0,
                )
                if skip_only is None:
                    continue
                placed = True
                break
            if not placed:
                return None

    # Extra scattered objects to make the scene less artificial.
    n_noise = random.randint(2, 6)
    for _ in range(n_noise):
        for _ in range(80):
            obj_h = random.randint(1, 4)
            obj_w = random.randint(1, 4)
            cells: set[tuple[int, int]] = set()
            top = random.randint(0, H - obj_h)
            left = random.randint(0, W - obj_w)
            cells = _rect_cells(top, left, obj_h, obj_w)
            if any(cell in ray_block for cell in cells):
                continue
            if not _occupied_ok(cells, occupied, (H, W), margin=1):
                continue
            color = random.choice([c for c in object_colors if c != bg])
            for r, c in cells:
                grid[r, c] = int(color)
                occupied.add((r, c))
            break

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _build_instance()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for bracket-capture task")
