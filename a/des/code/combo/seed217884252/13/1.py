import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

SIDE_TO_COLOR = {
    "top": RED,
    "right": BLUE,
    "bottom": GREEN,
    "left": YELLOW,
}

SIDES = ("top", "right", "bottom", "left")
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_ATTEMPTS = 400


def _mode(values: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return int(vals[counts.argmax()])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _infer_side_colors(grid: np.ndarray) -> dict[str, int]:
    h, w = grid.shape
    assert h >= 3 and w >= 3
    return {
        "top": _mode(grid[0, 1:-1]),
        "right": _mode(grid[1:-1, w - 1]),
        "bottom": _mode(grid[h - 1, 1:-1]),
        "left": _mode(grid[1:-1, 0]),
    }


def _infer_object_color(grid: np.ndarray, bg: int, side_colors: dict[str, int]) -> int:
    forbidden = {bg, *side_colors.values()}
    interior = grid[1:-1, 1:-1]
    colors = [int(color) for color in np.unique(interior) if int(color) not in forbidden]
    assert colors, "expected a foreground object color inside the border strips"
    if len(colors) == 1:
        return int(colors[0])
    counts = {int(color): int(np.count_nonzero(interior == int(color))) for color in colors}
    return int(max(counts, key=counts.get))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey bbox-edge gap using the color of the matching global border strip.

    Input structure:
    - black background
    - four single-pixel border strips on the grid edges
    - several grey 4-connected foreground objects in the interior
    - some bbox-edge cells of each object are missing (black single-pixel gaps)

    Rule:
    - For every grey connected component, compute its bbox.
    - Any black cell on the top/right/bottom/left side of that bbox is a gap.
    - Fill the gap with the color of the corresponding global border strip.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _mode(g)
    side_colors = _infer_side_colors(g)
    object_color = _infer_object_color(g, bg, side_colors)

    out = g.copy()
    for comp in _components4(g == object_color):
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0 = int(min(rows))
        r1 = int(max(rows))
        c0 = int(min(cols))
        c1 = int(max(cols))

        for c in range(c0 + 1, c1):
            if int(out[r0, c]) == bg:
                out[r0, c] = side_colors["top"]
            if int(out[r1, c]) == bg:
                out[r1, c] = side_colors["bottom"]

        for r in range(r0 + 1, r1):
            if int(out[r, c0]) == bg:
                out[r, c0] = side_colors["left"]
            if int(out[r, c1]) == bg:
                out[r, c1] = side_colors["right"]

    return out


def _pick_spaced(candidates: list[int], count: int, min_spacing: int = 2) -> list[int] | None:
    order = candidates[:]
    random.shuffle(order)
    picked: list[int] = []
    for value in order:
        if all(abs(value - prior) >= min_spacing for prior in picked):
            picked.append(int(value))
            if len(picked) == count:
                return sorted(picked)
    return None


def _local_gap_cells(h: int, w: int, sides: set[str]) -> dict[str, list[tuple[int, int]]] | None:
    for _ in range(200):
        mask = np.ones((h, w), dtype=bool)
        gaps: dict[str, list[tuple[int, int]]] = {side: [] for side in SIDES}

        ok = True
        for side in sides:
            span = w - 2 if side in ("top", "bottom") else h - 2
            if span <= 0:
                ok = False
                break

            count = 1
            if span >= 4 and random.random() < 0.35:
                count = 2
            positions = _pick_spaced(list(range(1, span + 1)), count)
            if positions is None:
                positions = [random.randint(1, span)]

            if side == "top":
                cells = [(0, pos) for pos in positions]
            elif side == "bottom":
                cells = [(h - 1, pos) for pos in positions]
            elif side == "left":
                cells = [(pos, 0) for pos in positions]
            else:
                cells = [(pos, w - 1) for pos in positions]

            for r, c in cells:
                mask[r, c] = False
            gaps[side] = [(int(r), int(c)) for r, c in cells]

        if not ok:
            continue
        if len(_components4(mask)) != 1:
            continue
        return gaps
    return None


def _sample_object_shape(sides: set[str]) -> tuple[np.ndarray, dict[str, list[tuple[int, int]]]] | None:
    min_size = 4 if len(sides) >= 3 else 3
    for _ in range(200):
        h = random.randint(min_size, 7)
        w = random.randint(min_size, 8)
        gaps = _local_gap_cells(h, w, sides)
        if gaps is None:
            continue
        mask = np.ones((h, w), dtype=bool)
        for cells in gaps.values():
            for r, c in cells:
                mask[r, c] = False
        if len(_components4(mask)) == 1:
            return mask, gaps
    return None


def _bbox_separated(
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
    clearance: int = 1,
) -> bool:
    ar0, c0a, ar1, c1a = bbox_a
    br0, c0b, br1, c1b = bbox_b
    return (
        ar1 + clearance < br0
        or br1 + clearance < ar0
        or c1a + clearance < c0b
        or c1b + clearance < c0a
    )


def _draw_border_strips(grid: np.ndarray) -> None:
    h, w = grid.shape
    grid[0, :] = RED
    grid[h - 1, :] = GREEN
    grid[1:-1, 0] = YELLOW
    grid[1:-1, w - 1] = BLUE


def _build_input_and_expected() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(12, 22)
    w = random.randint(12, 22)
    n_objects = random.randint(1, 4)

    side_sets = [set() for _ in range(n_objects)]
    for idx in range(n_objects):
        side_sets[idx].add(random.choice(SIDES))
    for side in SIDES:
        side_sets[random.randrange(n_objects)].add(side)
    for idx in range(n_objects):
        for side in SIDES:
            if side not in side_sets[idx] and random.random() < 0.30:
                side_sets[idx].add(side)

    objects = []
    for idx, sides in enumerate(side_sets):
        built = _sample_object_shape(sides)
        if built is None:
            return None
        mask, gaps = built
        objects.append(
            {
                "index": idx,
                "mask": mask,
                "gaps": gaps,
                "sides": sides,
                "h": int(mask.shape[0]),
                "w": int(mask.shape[1]),
            }
        )

    objects.sort(key=lambda obj: (-(obj["h"] * obj["w"]), -len(obj["sides"])))

    placed: list[dict[str, object]] = []
    for obj in objects:
        max_r0 = h - int(obj["h"]) - 2
        max_c0 = w - int(obj["w"]) - 2
        if max_r0 < 2 or max_c0 < 2:
            return None

        candidates = [(r0, c0) for r0 in range(2, max_r0 + 1) for c0 in range(2, max_c0 + 1)]
        random.shuffle(candidates)
        chosen = None
        for r0, c0 in candidates:
            bbox = (r0, c0, r0 + int(obj["h"]) - 1, c0 + int(obj["w"]) - 1)
            if all(_bbox_separated(bbox, other["bbox"]) for other in placed):
                chosen = (r0, c0, bbox)
                break
        if chosen is None:
            return None
        r0, c0, bbox = chosen
        placed.append({**obj, "r0": r0, "c0": c0, "bbox": bbox})

    inp = np.full((h, w), BG, dtype=int)
    _draw_border_strips(inp)
    expected = inp.copy()

    for obj in placed:
        mask = obj["mask"]
        r0 = int(obj["r0"])
        c0 = int(obj["c0"])
        oh = int(obj["h"])
        ow = int(obj["w"])
        inp[r0 : r0 + oh, c0 : c0 + ow][mask] = GREY
        expected[r0 : r0 + oh, c0 : c0 + ow][mask] = GREY

        for side, color in SIDE_TO_COLOR.items():
            for lr, lc in obj["gaps"][side]:
                rr = int(r0 + lr)
                cc = int(c0 + lc)
                inp[rr, cc] = BG
                expected[rr, cc] = int(color)

    if _mode(inp) != BG:
        return None

    side_colors = _infer_side_colors(inp)
    if side_colors != SIDE_TO_COLOR:
        return None

    comps = _components4(inp == GREY)
    if len(comps) != n_objects:
        return None

    if not all(any(obj["gaps"][side] for obj in placed) for side in SIDES):
        return None

    if not np.array_equal(main(inp), expected):
        return None

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_input_and_expected()
        if built is not None:
            inp, _expected = built
            return inp
    raise RuntimeError("failed to generate a valid ARC sample")
