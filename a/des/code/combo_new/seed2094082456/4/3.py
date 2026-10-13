from __future__ import annotations

from collections import Counter
import random
import numpy as np


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as coordinate arrays."""
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            parts.append(np.asarray(coords, dtype=int))
    return parts


def _tight_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Convert absolute or relative coordinates into a tight boolean mask and bbox."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    rel = coords - np.array([r0, c0], dtype=int)
    mask = np.zeros((h, w), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask, (int(r0), int(c0), int(r1), int(c1))


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    tight, _ = _tight_mask(np.argwhere(mask))
    return tuple(map(int, tight.shape)), tight.astype(np.uint8).tobytes()


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == int(bg):
            continue
        for coords in _components(grid == int(color)):
            mask, bbox = _tight_mask(coords)
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "mask": mask,
                    "bbox": bbox,
                    "area": int(len(coords)),
                }
            )
    return comps


def _find_holes(grid: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> list[dict]:
    """Find black hole components inside a component bbox."""
    r0, c0, r1, c1 = map(int, bbox)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    holes: list[dict] = []

    for rel_coords in _components(sub == int(bg)):
        if np.any(rel_coords[:, 0] == 0) or np.any(rel_coords[:, 0] == sub.shape[0] - 1):
            continue
        if np.any(rel_coords[:, 1] == 0) or np.any(rel_coords[:, 1] == sub.shape[1] - 1):
            continue
        mask, rel_bbox = _tight_mask(rel_coords)
        abs_coords = rel_coords + np.array([r0, c0], dtype=int)
        holes.append(
            {
                "coords": abs_coords,
                "mask": mask,
                "bbox": (
                    int(r0 + rel_bbox[0]),
                    int(c0 + rel_bbox[1]),
                    int(r0 + rel_bbox[2]),
                    int(c0 + rel_bbox[3]),
                ),
                "area": int(len(rel_coords)),
            }
        )
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reinsert scattered colored fragments into the matching holes of a large gray template.

    Input structure:
    - The background is black.
    - One large connected gray template sits near the grid center; inside its bbox are one or more black holes.
    - Outside the template are several monochrome colored fragments, each a separate 4-connected component.

    Rule:
    - Identify the template and extract every internal black hole as a tight binary shape.
    - For each hole, find the outside fragment with the exact same tight shape, using translation only.
    - Remove all outside fragments, then fill each hole with the color of its matched fragment while keeping the gray template unchanged.

    Key insight:
    The colors of the loose pieces are only fill labels; matching is determined entirely by exact shape equality between a hole and a fragment.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])

    comps = _monochrome_components(g, bg)
    assert comps, "expected a template and loose fragments"

    template_idx = None
    template_holes: list[dict] = []
    template_score: tuple[int, int] | None = None
    for idx, comp in enumerate(comps):
        holes = _find_holes(g, comp["bbox"], bg)
        if not holes:
            continue
        score = (int(comp["area"]), sum(int(h["area"]) for h in holes))
        if template_score is None or score > template_score:
            template_idx = int(idx)
            template_holes = holes
            template_score = score

    assert template_idx is not None and template_holes, "expected a large template with holes"

    fragments = [comp for idx, comp in enumerate(comps) if idx != template_idx]
    assert len(fragments) == len(template_holes), "hole and fragment counts must match"

    fragments_by_shape: dict[tuple[tuple[int, int], bytes], dict] = {}
    for frag in fragments:
        key = _shape_key(frag["mask"])
        assert key not in fragments_by_shape, "fragment shapes must be unique"
        fragments_by_shape[key] = frag

    out = g.copy()
    for frag in fragments:
        out[frag["coords"][:, 0], frag["coords"][:, 1]] = int(bg)

    for hole in template_holes:
        frag = fragments_by_shape.get(_shape_key(hole["mask"]))
        assert frag is not None, "every hole must have a matching fragment"
        out[hole["coords"][:, 0], hole["coords"][:, 1]] = int(frag["color"])

    return out


# --------------- generator ---------------

_BG = 0
_TEMPLATE = 5
_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

_GRID_MIN = 18
_GRID_MAX = 30
_TEMPLATE_H_RANGE = (9, 16)
_TEMPLATE_W_RANGE = (9, 16)
_TEMPLATE_MARGIN = 3

_HOLE_COUNT_RANGE = (2, 5)
_HOLE_AREA_RANGE = (3, 7)
_HOLE_SIDE_MAX = 5
_PLACE_RETRIES = 100
_SHAPE_RETRIES = 200
_GEN_RETRIES = 500


def _centered_origin(limit: int, span: int, margin: int) -> int | None:
    low = int(margin)
    high = int(limit - span - margin)
    if high < low:
        return None
    center = int((limit - span) // 2)
    lo = max(low, center - 2)
    hi = min(high, center + 2)
    return random.randint(lo, hi)


def _shape_coords(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask).astype(int)


def _mark_blocked(blocked: np.ndarray, coords: np.ndarray, r0: int, c0: int, pad: int) -> None:
    h, w = map(int, blocked.shape)
    for dr, dc in coords:
        r = int(r0 + dr)
        c = int(c0 + dc)
        rr0 = max(0, r - int(pad))
        rr1 = min(h, r + int(pad) + 1)
        cc0 = max(0, c - int(pad))
        cc1 = min(w, c + int(pad) + 1)
        blocked[rr0:rr1, cc0:cc1] = True


def _can_place(blocked: np.ndarray, coords: np.ndarray, r0: int, c0: int) -> bool:
    for dr, dc in coords:
        if blocked[int(r0 + dr), int(c0 + dc)]:
            return False
    return True


def _interesting_shape(mask: np.ndarray) -> bool:
    h, w = map(int, mask.shape)
    area = int(mask.sum())
    if area < _HOLE_AREA_RANGE[0]:
        return False
    if h < 2 or w < 2:
        return False
    if area == h * w:
        return False
    return True


def _random_shape() -> np.ndarray | None:
    raw_h = random.randint(2, _HOLE_SIDE_MAX)
    raw_w = random.randint(2, _HOLE_SIDE_MAX)
    max_area = min(_HOLE_AREA_RANGE[1], raw_h * raw_w - 1)
    if max_area < _HOLE_AREA_RANGE[0]:
        return None
    target = random.randint(_HOLE_AREA_RANGE[0], max_area)

    mask = np.zeros((raw_h, raw_w), dtype=bool)
    start = (random.randrange(raw_h), random.randrange(raw_w))
    mask[start] = True
    frontier = set(_neighbors4(start[0], start[1], raw_h, raw_w))

    while int(mask.sum()) < target:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        if mask[cell]:
            continue
        mask[cell] = True
        for nb in _neighbors4(cell[0], cell[1], raw_h, raw_w):
            if not mask[nb]:
                frontier.add(nb)

    tight, _ = _tight_mask(np.argwhere(mask))
    if not _interesting_shape(tight):
        return None
    return tight


def _sample_shapes(count: int) -> list[dict] | None:
    shapes: list[dict] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for _ in range(_SHAPE_RETRIES):
        if len(shapes) == count:
            return shapes
        mask = _random_shape()
        if mask is None:
            continue
        key = _shape_key(mask)
        if key in seen:
            continue
        coords = _shape_coords(mask)
        seen.add(key)
        shapes.append({"mask": mask, "coords": coords, "area": int(mask.sum())})
    return None


def _place_holes(template_h: int, template_w: int, shapes: list[dict]) -> list[dict] | None:
    for _ in range(_PLACE_RETRIES):
        blocked = np.zeros((template_h, template_w), dtype=bool)
        order = sorted(random.sample(shapes, len(shapes)), key=lambda item: -int(item["area"]))
        placed: list[dict] = []
        ok = True

        for shape in order:
            mask = shape["mask"]
            coords = shape["coords"]
            h, w = map(int, mask.shape)
            candidates = []
            for r0 in range(1, template_h - h):
                for c0 in range(1, template_w - w):
                    if _can_place(blocked, coords, r0, c0):
                        candidates.append((int(r0), int(c0)))
            if not candidates:
                ok = False
                break
            r0, c0 = random.choice(candidates)
            placed.append(
                {
                    "mask": mask,
                    "coords": coords,
                    "area": int(shape["area"]),
                    "r0": int(r0),
                    "c0": int(c0),
                }
            )
            _mark_blocked(blocked, coords, r0, c0, pad=1)

        if ok:
            return placed
    return None


def _place_fragments(grid: np.ndarray, fragments: list[dict], template_bbox: tuple[int, int, int, int]) -> bool:
    h, w = map(int, grid.shape)
    blocked = np.zeros((h, w), dtype=bool)
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True

    tr0, tc0, tr1, tc1 = map(int, template_bbox)
    rr0 = max(0, tr0 - 1)
    rr1 = min(h, tr1 + 2)
    cc0 = max(0, tc0 - 1)
    cc1 = min(w, tc1 + 2)
    blocked[rr0:rr1, cc0:cc1] = True

    order = sorted(random.sample(fragments, len(fragments)), key=lambda item: -int(item["area"]))
    for frag in order:
        mask = frag["mask"]
        coords = frag["coords"]
        ph, pw = map(int, mask.shape)
        candidates = []
        for r0 in range(1, h - ph):
            for c0 in range(1, w - pw):
                if _can_place(blocked, coords, r0, c0):
                    candidates.append((int(r0), int(c0)))
        if not candidates:
            return False
        r0, c0 = random.choice(candidates)
        for dr, dc in coords:
            grid[int(r0 + dr), int(c0 + dc)] = int(frag["color"])
        _mark_blocked(blocked, coords, r0, c0, pad=1)
    return True


def _gen_attempt() -> np.ndarray | None:
    template_h = random.randint(*_TEMPLATE_H_RANGE)
    template_w = random.randint(*_TEMPLATE_W_RANGE)

    min_h = max(_GRID_MIN, template_h + 2 * _TEMPLATE_MARGIN)
    min_w = max(_GRID_MIN, template_w + 2 * _TEMPLATE_MARGIN)
    if min_h > _GRID_MAX or min_w > _GRID_MAX:
        return None

    grid_h = random.randint(min_h, _GRID_MAX)
    grid_w = random.randint(min_w, _GRID_MAX)
    top = _centered_origin(grid_h, template_h, _TEMPLATE_MARGIN)
    left = _centered_origin(grid_w, template_w, _TEMPLATE_MARGIN)
    if top is None or left is None:
        return None

    interior_area = (template_h - 2) * (template_w - 2)
    max_holes = min(_HOLE_COUNT_RANGE[1], len(_COLOR_POOL), max(2, interior_area // 12))
    if max_holes < _HOLE_COUNT_RANGE[0]:
        return None
    hole_count = random.randint(_HOLE_COUNT_RANGE[0], max_holes)

    shapes = _sample_shapes(hole_count)
    if shapes is None:
        return None
    holes = _place_holes(template_h, template_w, shapes)
    if holes is None:
        return None

    colors = random.sample(_COLOR_POOL, hole_count)
    output = np.full((grid_h, grid_w), _BG, dtype=int)
    output[top : top + template_h, left : left + template_w] = int(_TEMPLATE)
    inp = output.copy()

    fragment_specs: list[dict] = []
    for hole, color in zip(holes, colors):
        for dr, dc in hole["coords"]:
            rr = int(top + hole["r0"] + dr)
            cc = int(left + hole["c0"] + dc)
            output[rr, cc] = int(color)
            inp[rr, cc] = int(_BG)
        fragment_specs.append(
            {
                "mask": hole["mask"],
                "coords": hole["coords"],
                "area": int(hole["area"]),
                "color": int(color),
            }
        )

    template_bbox = (int(top), int(left), int(top + template_h - 1), int(left + template_w - 1))
    if not _place_fragments(inp, fragment_specs, template_bbox):
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
