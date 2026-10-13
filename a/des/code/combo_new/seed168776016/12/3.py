import random

import numpy as np


BLACK = 0
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8, 9)
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 250
MASK_MUTATION_ATTEMPTS = 12
FOOTPRINT_AXIS_CAP = 24


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _color_components(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BLACK or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            rows = [p[0] for p in pts]
            cols = [p[1] for p in pts]
            comps.append(
                {
                    "color": color,
                    "coords": pts,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _hole_components(object_mask: np.ndarray, bg_mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = object_mask.shape
    holes: list[list[tuple[int, int]]] = []
    for pts in _components(bg_mask):
        if any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in pts):
            continue
        holes.append(pts)
    return holes


def _count_holes_and_fill(object_mask: np.ndarray, bg_mask: np.ndarray) -> tuple[int, np.ndarray]:
    holes = _hole_components(object_mask, bg_mask)
    filled = object_mask.copy()
    for pts in holes:
        for r, c in pts:
            filled[r, c] = True
    return len(holes), filled


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2D")

    objects = _color_components(g)
    placements: list[dict[str, object]] = []
    out_h, out_w = map(int, g.shape)

    for comp in objects:
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        subgrid = g[r0 : r1 + 1, c0 : c1 + 1]
        object_mask = subgrid == color
        bg_mask = subgrid == BLACK
        holes, solid_mask = _count_holes_and_fill(object_mask, bg_mask)
        mask_h, mask_w = map(int, solid_mask.shape)
        horizontal = mask_w >= mask_h
        copies = holes + 1
        if horizontal:
            out_h = max(out_h, r0 + mask_h)
            out_w = max(out_w, c0 + copies * mask_w + holes)
        else:
            out_h = max(out_h, r0 + copies * mask_h + holes)
            out_w = max(out_w, c0 + mask_w)
        placements.append(
            {
                "color": color,
                "top": r0,
                "left": c0,
                "mask": solid_mask,
                "copies": copies,
                "horizontal": horizontal,
            }
        )

    out = np.zeros((out_h, out_w), dtype=int)
    for item in placements:
        color = int(item["color"])
        top = int(item["top"])
        left = int(item["left"])
        mask = np.asarray(item["mask"], dtype=bool)
        copies = int(item["copies"])
        mask_h, mask_w = map(int, mask.shape)
        step_r = 0
        step_c = mask_w + 1
        if not bool(item["horizontal"]):
            step_r = mask_h + 1
            step_c = 0
        for k in range(copies):
            rr = top + k * step_r
            cc = left + k * step_c
            coords = np.argwhere(mask)
            rows = rr + coords[:, 0]
            cols = cc + coords[:, 1]
            if np.any((out[rows, cols] != BLACK) & (out[rows, cols] != color)):
                raise AssertionError("replicated objects must not overlap")
            out[rows, cols] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each object's holes and repeat the solid object according to that hole count.

    Input structure:
    - The background is black.
    - Several non-touching colored 4-connected objects appear, and each object uses a distinct color.
    - Every object is visibly longer in either the horizontal or vertical direction.
    - Some objects contain 0 to 3 enclosed black holes fully surrounded by that object's pixels.

    Rule:
    - Process each colored object independently.
    - Count the number of enclosed black holes inside the object, then fill those holes with the object's own color to make a solid version.
    - Keep one solid copy at the object's original position, and add one more copy for each hole.
    - Arrange the copies to the right for horizontally longer objects, or downward for vertically longer objects, leaving exactly one black cell gap between consecutive copies.

    Key insight:
    - The holes are not missing data: their count encodes how many solid copies must appear, while the object's longer axis determines the replication direction.
    """
    return _transform(grid)


def _independent_capacity(h: int, w: int) -> int:
    if h <= 0 or w <= 0:
        return 0
    return (h * w + 1) // 2


def _candidate_dims(orientation: str, holes: int) -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    for h in range(3, 7):
        for w in range(3, 8):
            if orientation == "h":
                if w <= h:
                    continue
                axis = w
            else:
                if h <= w:
                    continue
                axis = h
            if _independent_capacity(h - 2, w - 2) < holes:
                continue
            if (holes + 1) * axis + holes > FOOTPRINT_AXIS_CAP:
                continue
            dims.append((h, w))
    return dims


def _select_holes(h: int, w: int, holes: int, rng: random.Random) -> list[tuple[int, int]] | None:
    if holes == 0:
        return []
    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    for _ in range(200):
        order = interior[:]
        rng.shuffle(order)
        chosen: list[tuple[int, int]] = []
        for cell in order:
            if all(abs(cell[0] - r) + abs(cell[1] - c) > 1 for r, c in chosen):
                chosen.append(cell)
                if len(chosen) == holes:
                    return chosen
    return None


def _mask_spans_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _count_holes(mask: np.ndarray) -> int:
    return len(_hole_components(mask, ~mask))


def _is_connected(mask: np.ndarray) -> bool:
    return int(mask.sum()) > 0 and len(_components(mask)) == 1


def _mutate_outline(mask: np.ndarray, holes: int, rng: random.Random) -> np.ndarray:
    out = mask.copy()
    goal = rng.randint(0, 2)
    removed = 0
    candidates = [(r, c) for r in range(out.shape[0]) for c in range(out.shape[1]) if bool(out[r, c]) and (r in (0, out.shape[0] - 1) or c in (0, out.shape[1] - 1))]
    rng.shuffle(candidates)
    for r, c in candidates:
        if removed >= goal:
            break
        trial = out.copy()
        trial[r, c] = False
        if not _mask_spans_bbox(trial):
            continue
        if not _is_connected(trial):
            continue
        if _count_holes(trial) != holes:
            continue
        out = trial
        removed += 1
    return out


def _make_object_mask(orientation: str, holes: int, rng: random.Random) -> np.ndarray | None:
    dims = _candidate_dims(orientation, holes)
    if not dims:
        return None
    for _ in range(MASK_MUTATION_ATTEMPTS):
        h, w = rng.choice(dims)
        hole_cells = _select_holes(h, w, holes, rng)
        if hole_cells is None:
            continue
        mask = np.ones((h, w), dtype=bool)
        for r, c in hole_cells:
            mask[r, c] = False
        mask = _mutate_outline(mask, holes, rng)
        axis_ok = (mask.shape[1] > mask.shape[0]) if orientation == "h" else (mask.shape[0] > mask.shape[1])
        if not axis_ok:
            continue
        if _count_holes(mask) != holes or not _is_connected(mask):
            continue
        return mask
    return None


def _footprint(mask: np.ndarray, orientation: str, holes: int) -> tuple[int, int]:
    h, w = map(int, mask.shape)
    copies = holes + 1
    if orientation == "h":
        return h, copies * w + holes
    return copies * h + holes, w


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_specs(specs: list[dict[str, object]], full_h: int, full_w: int, rng: random.Random) -> bool:
    padded_boxes: list[tuple[int, int, int, int]] = []
    for spec in sorted(specs, key=lambda item: int(item["foot_h"]) * int(item["foot_w"]), reverse=True):
        foot_h = int(spec["foot_h"])
        foot_w = int(spec["foot_w"])
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = rng.randint(0, full_h - foot_h)
            left = rng.randint(0, full_w - foot_w)
            box = (top, left, top + foot_h - 1, left + foot_w - 1)
            padded = (box[0] - 1, box[1] - 1, box[2] + 1, box[3] + 1)
            if any(_rectangles_overlap(padded, other) for other in padded_boxes):
                continue
            spec["top"] = top
            spec["left"] = left
            padded_boxes.append(padded)
            placed = True
            break
        if not placed:
            return False
    return True


def _build_specs(rng: random.Random) -> list[dict[str, object]] | None:
    count = rng.choice((1, 2, 3))
    colors = rng.sample(COLOR_POOL, count)
    specs: list[dict[str, object]] = []
    for color in colors:
        holes = rng.choice((0, 1, 2, 3))
        orientation = rng.choice(("h", "v"))
        mask = _make_object_mask(orientation, holes, rng)
        if mask is None:
            return None
        foot_h, foot_w = _footprint(mask, orientation, holes)
        specs.append(
            {
                "color": color,
                "holes": holes,
                "orientation": orientation,
                "mask": mask,
                "foot_h": foot_h,
                "foot_w": foot_w,
            }
        )
    return specs


def _draw_input(specs: list[dict[str, object]], input_h: int, input_w: int) -> np.ndarray:
    grid = np.zeros((input_h, input_w), dtype=int)
    for spec in specs:
        color = int(spec["color"])
        top = int(spec["top"])
        left = int(spec["left"])
        mask = np.asarray(spec["mask"], dtype=bool)
        coords = np.argwhere(mask)
        rows = top + coords[:, 0]
        cols = left + coords[:, 1]
        grid[rows, cols] = color
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    specs = _build_specs(rng)
    if specs is None:
        return None

    max_foot_h = max(int(spec["foot_h"]) for spec in specs)
    max_foot_w = max(int(spec["foot_w"]) for spec in specs)
    area_hint = sum(int(spec["foot_h"]) * int(spec["foot_w"]) for spec in specs)
    side_hint = min(30, max(8, int(np.ceil(np.sqrt(area_hint))) + rng.randint(2, 6)))

    full_h_low = max(max_foot_h, side_hint - rng.randint(0, 3))
    full_w_low = max(max_foot_w, side_hint - rng.randint(0, 3))
    if full_h_low > 30 or full_w_low > 30:
        return None

    full_h = rng.randint(full_h_low, 30)
    full_w = rng.randint(full_w_low, 30)
    if not _place_specs(specs, full_h, full_w, rng):
        return None

    max_input_h = max(int(spec["top"]) + spec["mask"].shape[0] for spec in specs)
    max_input_w = max(int(spec["left"]) + spec["mask"].shape[1] for spec in specs)
    input_h = rng.randint(max_input_h, full_h)
    input_w = rng.randint(max_input_w, full_w)
    inp = _draw_input(specs, input_h, input_w)

    if len(_color_components(inp)) != len(specs):
        return None

    out = _transform(inp)
    if inp.ndim != 2 or out.ndim != 2:
        return None
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
