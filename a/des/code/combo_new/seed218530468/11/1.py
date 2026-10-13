from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FRAME_COLOR = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400
_RNG = random.Random()
_N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _background_color(grid: np.ndarray) -> int:
    border = np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))
    colors, counts = np.unique(border, return_counts=True)
    return int(colors[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in _N4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _shape_key(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _mask_from_key(key: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = max(r for r, _c in key) + 1
    w = max(c for _r, c in key) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in key:
        mask[r, c] = True
    return mask


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    comps: list[dict[str, object]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(grid == color):
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _find_frame_and_holes(
    grid: np.ndarray, comps: list[dict[str, object]], bg: int
) -> tuple[dict[str, object], list[np.ndarray]]:
    best_comp: dict[str, object] | None = None
    best_holes: list[np.ndarray] = []
    best_score: tuple[int, int] | None = None

    for comp in comps:
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all((patch == bg) | (patch == color)):
            continue
        if (
            np.any(patch[0, :] != color)
            or np.any(patch[-1, :] != color)
            or np.any(patch[:, 0] != color)
            or np.any(patch[:, -1] != color)
        ):
            continue

        holes_here: list[np.ndarray] = []
        for rel in _components(patch == bg):
            if (
                np.any(rel[:, 0] == 0)
                or np.any(rel[:, 1] == 0)
                or np.any(rel[:, 0] == patch.shape[0] - 1)
                or np.any(rel[:, 1] == patch.shape[1] - 1)
            ):
                continue
            holes_here.append(rel + np.array([r0, c0], dtype=int))

        if not holes_here:
            continue

        score = ((r1 - r0 + 1) * (c1 - c0 + 1), int(comp["area"]))
        if best_score is None or score > best_score:
            best_score = score
            best_comp = comp
            best_holes = holes_here

    assert best_comp is not None, "main: expected one grey frame with enclosed holes"
    return best_comp, best_holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each internal hole to an exterior sprite of the same shape and recolor the frame rows it spans.

    Input structure:
    - The background is black.
    - One large grey rectangular frame-like slab occupies a rectangular bounding box; inside it, several black 4-connected hole regions are cut out and remain strictly enclosed by grey pixels.
    - Outside the grey slab sit several small monochrome non-grey sprites, one per hole, and each sprite has exactly the same shape as its matching hole up to translation.

    Rule:
    - Identify the large grey rectangle, extract every enclosed black hole, and normalize each hole shape by translating it to the origin.
    - Identify each exterior monochrome sprite, normalize its shape the same way, and match holes to sprites by exact shape equality to recover the sprite color for each hole.
    - For every row touched by a hole, recolor the grey frame pixels on that row with the matched color; if multiple holes touch the same row, the hole whose topmost row is smaller has priority.
    - Erase the exterior sprites to black and leave the holes themselves black.

    Key insight:
    - The exterior colored sprites are a shape-to-color legend for the holes, and the recovered color paints the frame rows containing that hole instead of filling the hole itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    comps = _components_by_color(g, bg)
    frame_comp, holes = _find_frame_and_holes(g, comps, bg)

    shape_to_color: dict[tuple[tuple[int, int], ...], int] = {}
    exterior_comps: list[dict[str, object]] = []
    for comp in comps:
        if comp is frame_comp:
            continue
        key = _shape_key(comp["coords"])
        assert key not in shape_to_color, "main: exterior object shapes must be unique"
        shape_to_color[key] = int(comp["color"])
        exterior_comps.append(comp)

    assert len(holes) == len(exterior_comps), "main: expected one exterior object per hole"

    hole_info: list[dict[str, object]] = []
    for hole in holes:
        key = _shape_key(hole)
        assert key in shape_to_color, "main: every hole must have a matching exterior sprite"
        rows = sorted(int(r) for r in np.unique(hole[:, 0]))
        hole_info.append(
            {
                "color": int(shape_to_color[key]),
                "rows": rows,
                "top": int(rows[0]),
                "left": int(hole[:, 1].min()),
            }
        )

    hole_info.sort(key=lambda item: (int(item["top"]), int(item["left"])))

    row_to_color: dict[int, int] = {}
    for info in hole_info:
        for row in info["rows"]:
            row_to_color.setdefault(int(row), int(info["color"]))

    out = g.copy()
    for comp in exterior_comps:
        rr = comp["coords"][:, 0]
        cc = comp["coords"][:, 1]
        out[rr, cc] = bg

    frame_mask = np.zeros_like(g, dtype=bool)
    rr = frame_comp["coords"][:, 0]
    cc = frame_comp["coords"][:, 1]
    frame_mask[rr, cc] = True

    for row, color in row_to_color.items():
        out[row, frame_mask[row]] = int(color)

    return out


def _normalize_cell_set(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _random_shape(
    rng: random.Random,
    min_area: int = 2,
    max_area: int = 6,
    max_h: int = 4,
    max_w: int = 4,
    min_h: int = 1,
) -> tuple[tuple[tuple[int, int], ...], np.ndarray]:
    for _ in range(200):
        target = rng.randint(min_area, max_area)
        cells = {(0, 0)}
        while len(cells) < target:
            frontier = list(
                {
                    (r + dr, c + dc)
                    for r, c in cells
                    for dr, dc in _N4
                    if (r + dr, c + dc) not in cells
                }
            )
            rng.shuffle(frontier)
            added = False
            for cell in frontier:
                trial = set(cells)
                trial.add(cell)
                key = _normalize_cell_set(trial)
                h = max(r for r, _c in key) + 1
                w = max(c for _r, c in key) + 1
                if h <= max_h and w <= max_w:
                    cells = set(key)
                    added = True
                    break
            if not added:
                break

        key = _normalize_cell_set(cells)
        h = max(r for r, _c in key) + 1
        if len(key) == target and h >= min_h:
            return key, _mask_from_key(key)
    raise RuntimeError("shape generation failed")


def _build_hole_shapes(
    rng: random.Random, hole_count: int, mode: str
) -> list[dict[str, object]] | None:
    shapes: list[dict[str, object]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    min_h0 = 2 if mode == "overlap" else 1
    for idx in range(hole_count):
        min_h = min_h0 if idx == 0 else 1
        for _ in range(200):
            key, mask = _random_shape(rng, min_h=min_h)
            if key in seen:
                continue
            seen.add(key)
            shapes.append({"key": key, "mask": mask, "h": int(mask.shape[0]), "w": int(mask.shape[1])})
            break
        else:
            return None
    tail = shapes[1:]
    rng.shuffle(tail)
    shapes[1:] = tail
    return shapes


def _layout_holes(
    rng: random.Random, shapes: list[dict[str, object]], mode: str
) -> tuple[list[dict[str, object]], int, int] | None:
    count = len(shapes)
    placed: list[dict[str, object]] = []

    if mode == "overlap":
        x = 0
        tops = list(range(count))
        for idx, shape in enumerate(shapes):
            placed.append(
                {
                    "key": shape["key"],
                    "mask": shape["mask"],
                    "h": shape["h"],
                    "w": shape["w"],
                    "top": int(tops[idx]),
                    "left": int(x),
                }
            )
            x += int(shape["w"]) + rng.randint(1, 2)

        rows0 = set(range(placed[0]["top"], placed[0]["top"] + placed[0]["h"]))
        rows1 = set(range(placed[1]["top"], placed[1]["top"] + placed[1]["h"]))
        if not rows0 & rows1:
            return None

        pad_top = rng.randint(0, 2)
        pad_left = rng.randint(0, 2)
        pad_bottom = rng.randint(0, 2)
        pad_right = rng.randint(0, 2)
        for item in placed:
            item["top"] += pad_top
            item["left"] += pad_left
        inner_h = max(int(item["top"]) + int(item["h"]) for item in placed) + pad_bottom
        inner_w = max(int(item["left"]) + int(item["w"]) for item in placed) + pad_right
    else:
        inner_w = max(int(shape["w"]) for shape in shapes) + rng.randint(2, 5)
        y = 0
        for shape in shapes:
            max_left = inner_w - int(shape["w"])
            left = rng.randint(0, max_left)
            placed.append(
                {
                    "key": shape["key"],
                    "mask": shape["mask"],
                    "h": shape["h"],
                    "w": shape["w"],
                    "top": int(y),
                    "left": int(left),
                }
            )
            y += int(shape["h"]) + rng.randint(1, 2)
        inner_h = max(int(item["top"]) + int(item["h"]) for item in placed) + rng.randint(0, 2)

        row_owners: dict[int, int] = {}
        for idx, item in enumerate(placed):
            for row in range(int(item["top"]), int(item["top"]) + int(item["h"])):
                if row in row_owners:
                    return None
                row_owners[row] = idx

    if inner_h > 18 or inner_w > 20:
        return None
    return placed, inner_h, inner_w


def _draw_frame_with_holes(
    grid: np.ndarray, frame_top: int, frame_left: int, inner_h: int, inner_w: int, holes: list[dict[str, object]]
) -> list[dict[str, object]]:
    frame_h = inner_h + 2
    frame_w = inner_w + 2
    grid[frame_top : frame_top + frame_h, frame_left : frame_left + frame_w] = FRAME_COLOR

    hole_specs: list[dict[str, object]] = []
    for item in holes:
        mask = item["mask"]
        top = frame_top + 1 + int(item["top"])
        left = frame_left + 1 + int(item["left"])
        rr, cc = np.where(mask)
        grid[top + rr, left + cc] = BG
        abs_coords = np.column_stack((top + rr, left + cc)).astype(int)
        hole_specs.append(
            {
                "key": item["key"],
                "coords": abs_coords,
                "top": int(abs_coords[:, 0].min()),
                "rows": sorted(int(r) for r in np.unique(abs_coords[:, 0])),
                "h": int(mask.shape[0]),
                "w": int(mask.shape[1]),
                "mask": mask,
            }
        )
    return hole_specs


def _can_place_object(grid: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    h, w = mask.shape
    gh, gw = grid.shape
    if top < 1 or left < 1 or top + h >= gh or left + w >= gw:
        return False
    region = grid[top - 1 : top + h + 1, left - 1 : left + w + 1]
    return not bool(np.any(region != BG))


def _place_objects(
    rng: random.Random, grid: np.ndarray, holes: list[dict[str, object]], colors: list[int]
) -> bool:
    order = list(range(len(holes)))
    rng.shuffle(order)
    gh, gw = grid.shape

    for idx in order:
        hole = holes[idx]
        mask = hole["mask"]
        color = colors[idx]
        h, w = mask.shape
        for _ in range(300):
            top = rng.randint(1, gh - h - 1)
            left = rng.randint(1, gw - w - 1)
            if not _can_place_object(grid, mask, top, left):
                continue
            rr, cc = np.where(mask)
            grid[top + rr, left + cc] = color
            break
        else:
            return False
    return True


def _row_overlap_exists(holes: list[dict[str, object]]) -> bool:
    owners: dict[int, list[int]] = {}
    for hole in holes:
        top = int(hole["top"])
        for row in hole["rows"]:
            owners.setdefault(int(row), []).append(top)
    return any(len(vals) >= 2 and len(set(vals)) >= 2 for vals in owners.values())


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    mode = rng.choice(("overlap", "separate"))
    hole_count = rng.randint(2, 4)

    shapes = _build_hole_shapes(rng, hole_count, mode)
    if shapes is None:
        return None

    layout = _layout_holes(rng, shapes, mode)
    if layout is None:
        return None
    holes_rel, inner_h, inner_w = layout

    frame_h = inner_h + 2
    frame_w = inner_w + 2
    min_h = max(16, frame_h + 4)
    min_w = max(18, frame_w + 6)
    if min_h > 30 or min_w > 30:
        return None

    grid_h = rng.randint(min_h, 30)
    grid_w = rng.randint(min_w, 30)
    frame_top = rng.randint(1, grid_h - frame_h - 1)
    frame_left = rng.randint(1, grid_w - frame_w - 1)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    holes = _draw_frame_with_holes(grid, frame_top, frame_left, inner_h, inner_w, holes_rel)

    if mode == "overlap":
        if not _row_overlap_exists(holes):
            return None
    else:
        if _row_overlap_exists(holes):
            return None

    colors = list(rng.sample(OBJECT_COLORS, hole_count))
    if not _place_objects(rng, grid, holes, colors):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
