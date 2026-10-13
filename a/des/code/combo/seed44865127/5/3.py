from __future__ import annotations

import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BG = 0
L_SHAPES = (
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
)


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _foreground_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not bool(seen[nr, nc]) and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))

            arr = np.asarray(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "color": int(color),
                    "coords": arr,
                    "area": int(len(coords)),
                    "bbox": (r0, c0, r1, c1),
                }
            )
    return comps


def _normalize_coords(coords: np.ndarray | list[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    arr = np.asarray(coords, dtype=int)
    r0 = int(arr[:, 0].min())
    c0 = int(arr[:, 1].min())
    norm = tuple(sorted((int(r - r0), int(c - c0)) for r, c in arr))
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    return norm, int(h), int(w)


def _is_l_shape(coords: np.ndarray | list[tuple[int, int]]) -> bool:
    norm, h, w = _normalize_coords(coords)
    return len(norm) == 3 and h == 2 and w == 2


def _l_missing_corner(coords: np.ndarray | list[tuple[int, int]]) -> tuple[int, int]:
    norm, h, w = _normalize_coords(coords)
    assert len(norm) == 3 and h == 2 and w == 2
    missing = {(0, 0), (0, 1), (1, 0), (1, 1)} - set(norm)
    assert len(missing) == 1
    return next(iter(missing))


def _extract_template_holes(grid: np.ndarray, comp: dict, bg: int) -> list[dict] | None:
    color = int(comp["color"])
    r0, c0, r1, c1 = comp["bbox"]
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = sub.shape

    if h < 4 or w < 4:
        return None
    if not np.all((sub == color) | (sub == bg)):
        return None
    if not (
        np.all(sub[0, :] == color)
        and np.all(sub[-1, :] == color)
        and np.all(sub[:, 0] == color)
        and np.all(sub[:, -1] == color)
    ):
        return None

    hole_mask = sub == bg
    if not bool(hole_mask.any()):
        return None

    holes: list[dict] = []
    total_hole_cells = 0
    for hole in _components_mask(hole_mask):
        if np.any(hole[:, 0] == 0) or np.any(hole[:, 0] == h - 1) or np.any(hole[:, 1] == 0) or np.any(hole[:, 1] == w - 1):
            return None
        if not _is_l_shape(hole):
            return None
        total_hole_cells += int(len(hole))
        holes.append(
            {
                "color": int(color),
                "orientation": _l_missing_corner(hole),
                "coords": [(r0 + int(r), c0 + int(c)) for r, c in hole],
            }
        )

    if int(comp["area"]) + total_hole_cells != int(sub.size):
        return None
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each small monochrome L stamp into an L-shaped hole of the same-color template rectangle.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    out = g.copy()

    stamps_by_color: dict[int, list[dict]] = {}
    holes_by_color: dict[int, list[dict]] = {}

    for comp in _foreground_components(g, bg):
        color = int(comp["color"])
        coords = comp["coords"]
        if int(comp["area"]) == 3 and _is_l_shape(coords):
            stamps_by_color.setdefault(color, []).append(comp)
            continue

        holes = _extract_template_holes(g, comp, bg)
        assert holes is not None, "foreground objects must be either L stamps or rectangular templates"
        holes_by_color.setdefault(color, []).extend(holes)

    assert holes_by_color and stamps_by_color
    assert set(holes_by_color) == set(stamps_by_color), "template colors and stamp colors must match"

    for color, holes in holes_by_color.items():
        stamps = stamps_by_color.get(int(color), [])
        assert len(stamps) == len(holes), "each hole needs exactly one same-color stamp"

    for stamps in stamps_by_color.values():
        for stamp in stamps:
            for r, c in stamp["coords"]:
                out[int(r), int(c)] = bg

    for color, holes in holes_by_color.items():
        for hole in holes:
            for r, c in hole["coords"]:
                out[int(r), int(c)] = int(color)

    return out


def _hole_slots(height: int, width: int) -> list[tuple[int, int]]:
    rows = list(range(1, height - 2, 3))
    cols = list(range(1, width - 2, 3))
    return [(r, c) for r in rows for c in cols]


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    rr0 = max(0, int(r0) - pad)
    cc0 = max(0, int(c0) - pad)
    rr1 = min(blocked.shape[0], int(r0) + int(h) + pad)
    cc1 = min(blocked.shape[1], int(c0) + int(w) + pad)
    blocked[rr0:rr1, cc0:cc1] = True


def _can_place_bbox(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    if r0 < 0 or c0 < 0 or r0 + h > blocked.shape[0] or c0 + w > blocked.shape[1]:
        return False
    return not bool(blocked[r0 : r0 + h, c0 : c0 + w].any())


def _random_bbox_position(blocked: np.ndarray, h: int, w: int, attempts: int = 200) -> tuple[int, int] | None:
    for _ in range(attempts):
        r0 = random.randint(0, blocked.shape[0] - h)
        c0 = random.randint(0, blocked.shape[1] - w)
        if _can_place_bbox(blocked, r0, c0, h, w):
            return int(r0), int(c0)
    return None


def _make_templates() -> list[dict]:
    template_count = random.randint(1, 3)
    colors = random.sample(list(range(1, 10)), template_count)
    specs: list[dict] = []

    for color in colors:
        height = random.randint(5, 10)
        width = random.randint(5, 10)
        slots = _hole_slots(height, width)
        max_holes = min(4, len(slots))
        hole_count = random.randint(1, max_holes)
        chosen_slots = random.sample(slots, hole_count)
        holes = [{"anchor": (int(r), int(c)), "shape": random.randrange(4)} for r, c in chosen_slots]
        specs.append(
            {
                "color": int(color),
                "height": int(height),
                "width": int(width),
                "holes": holes,
            }
        )
    return specs


def _choose_canvas(specs: list[dict]) -> tuple[int, int]:
    max_h = max(int(spec["height"]) for spec in specs)
    max_w = max(int(spec["width"]) for spec in specs)
    colored_budget = sum(int(spec["height"]) * int(spec["width"]) for spec in specs)
    colored_budget += 4 * sum(len(spec["holes"]) for spec in specs)

    for _ in range(200):
        height = random.randint(max(16, max_h + 6), 28)
        width = random.randint(max(16, max_w + 6), 28)
        if height * width >= colored_budget * 3:
            return int(height), int(width)
    return int(max(18, max_h + 8)), int(max(18, max_w + 8))


def _build_example() -> tuple[np.ndarray, np.ndarray]:
    specs = _make_templates()
    height, width = _choose_canvas(specs)

    expected = np.full((height, width), BG, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    placed_specs: list[dict] = []

    for spec in sorted(specs, key=lambda item: item["height"] * item["width"], reverse=True):
        pos = _random_bbox_position(blocked, int(spec["height"]), int(spec["width"]))
        if pos is None:
            raise AssertionError("template placement failed")
        r0, c0 = pos
        expected[r0 : r0 + int(spec["height"]), c0 : c0 + int(spec["width"])] = int(spec["color"])
        _mark_blocked(blocked, r0, c0, int(spec["height"]), int(spec["width"]), pad=1)
        placed_specs.append({**spec, "top": int(r0), "left": int(c0)})

    inp = expected.copy()
    stamp_specs: list[dict] = []

    for spec in placed_specs:
        color = int(spec["color"])
        top = int(spec["top"])
        left = int(spec["left"])
        for hole in spec["holes"]:
            hr, hc = hole["anchor"]
            shape = int(hole["shape"])
            for dr, dc in L_SHAPES[shape]:
                inp[top + hr + dr, left + hc + dc] = BG
            stamp_specs.append({"color": color, "shape": random.randrange(4)})

    random.shuffle(stamp_specs)

    for stamp in stamp_specs:
        pos = _random_bbox_position(blocked, 2, 2)
        if pos is None:
            raise AssertionError("stamp placement failed")
        r0, c0 = pos
        for dr, dc in L_SHAPES[int(stamp["shape"])]:
            inp[r0 + dr, c0 + dc] = int(stamp["color"])
        _mark_blocked(blocked, r0, c0, 2, 2, pad=1)

    if int(np.count_nonzero(inp == BG)) <= inp.size // 2:
        raise AssertionError("background should remain dominant")

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            inp, expected = _build_example()
            if np.array_equal(main(inp), expected):
                return inp
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid sample")
