import random
from collections import Counter, deque

import numpy as np


BG_COLOR = 0
GEN_FRAME_COLOR = 5
GEN_GRID_H_RANGE = (14, 29)
GEN_GRID_W_RANGE = (14, 29)
GEN_OBJECT_COUNT_RANGE = (1, 5)
GEN_INNER_H_RANGE = (2, 8)
GEN_INNER_W_RANGE = (2, 8)
GEN_ATTEMPTS = 400
GEN_PLACE_TRIES = 80
GEN_MARGIN = 1
GEN_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SIDES = ("top", "bottom", "left", "right")


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def border_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def gap_side(gap: tuple[int, int], r0: int, c0: int, r1: int, c1: int) -> str | None:
    gr, gc = gap
    if gr == r0 and c0 < gc < c1:
        return "top"
    if gr == r1 and c0 < gc < c1:
        return "bottom"
    if gc == c0 and r0 < gr < r1:
        return "left"
    if gc == c1 and r0 < gr < r1:
        return "right"
    return None


def _frame_specs(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    specs: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        for comp in components_mask(g == color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            border = border_cells(r0, c0, r1, c1)
            border_set = set(border)
            comp_set = set(comp)
            if not comp_set <= border_set:
                continue
            if len(comp_set) != len(border_set) - 1:
                continue
            gaps = [cell for cell in border if cell not in comp_set]
            if len(gaps) != 1:
                continue
            gap = gaps[0]
            side = gap_side(gap, r0, c0, r1, c1)
            if side is None:
                continue
            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            seed_coords = np.argwhere((interior != bg) & (interior != color))
            if seed_coords.shape != (1, 2):
                continue
            sr = int(seed_coords[0, 0] + r0 + 1)
            sc = int(seed_coords[0, 1] + c0 + 1)
            specs.append(
                {
                    "frame_color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "gap": (int(gap[0]), int(gap[1])),
                    "side": side,
                    "seed": (sr, sc),
                    "seed_color": int(g[sr, sc]),
                }
            )
    return specs


def _apply_spec(out: np.ndarray, spec: dict) -> None:
    r0, c0, r1, c1 = spec["bbox"]
    sr, sc = spec["seed"]
    color = int(spec["seed_color"])
    side = spec["side"]
    gr, gc = spec["gap"]
    if side == "top":
        out[sr:r1, c0 + 1 : c1] = color
        rr, cc = r0 - 1, gc
    elif side == "bottom":
        out[r0 + 1 : sr + 1, c0 + 1 : c1] = color
        rr, cc = r1 + 1, gc
    elif side == "left":
        out[r0 + 1 : r1, sc:c1] = color
        rr, cc = gr, c0 - 1
    else:
        out[r0 + 1 : r1, c0 + 1 : sc + 1] = color
        rr, cc = gr, c1 + 1
    if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
        out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a seed-colored slab inside each gapped rectangular frame.

    The frame side containing the gap determines the opposite growth direction.
    Starting from the seed pixel's row or column, fill complete interior rows or
    columns away from the gap, leave the gap-side half empty, and place a
    single overflow pixel just outside the gap.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for spec in _frame_specs(g):
        _apply_spec(out, spec)
    return out


def _reserve_box(r0: int, c0: int, r1: int, c1: int) -> tuple[int, int, int, int]:
    return r0 - GEN_MARGIN, c0 - GEN_MARGIN, r1 + GEN_MARGIN, c1 + GEN_MARGIN


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _sample_gap(r0: int, c0: int, r1: int, c1: int, side: str, rng: random.Random) -> tuple[int, int]:
    if side == "top":
        return (r0, rng.randint(c0 + 1, c1 - 1))
    if side == "bottom":
        return (r1, rng.randint(c0 + 1, c1 - 1))
    if side == "left":
        return (rng.randint(r0 + 1, r1 - 1), c0)
    return (rng.randint(r0 + 1, r1 - 1), c1)


def _draw_input_from_specs(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG_COLOR, dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        gap = spec["gap"]
        for cell in border_cells(r0, c0, r1, c1):
            if cell != gap:
                grid[cell] = GEN_FRAME_COLOR
        sr, sc = spec["seed"]
        grid[sr, sc] = int(spec["seed_color"])
    return grid


def _expected_from_specs(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    out = _draw_input_from_specs(shape, specs)
    for spec in specs:
        _apply_spec(out, spec)
    return out


def _sample_spec(
    h: int,
    w: int,
    used_boxes: list[tuple[int, int, int, int]],
    used_colors: set[int],
    rng: random.Random,
) -> dict | None:
    inner_h = rng.randint(*GEN_INNER_H_RANGE)
    inner_w = rng.randint(*GEN_INNER_W_RANGE)
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    if outer_h + 2 > h or outer_w + 2 > w:
        return None
    side = rng.choice(SIDES)
    seed_color_choices = [c for c in GEN_COLORS if c not in used_colors]
    if not seed_color_choices:
        return None
    for _ in range(GEN_PLACE_TRIES):
        r0 = rng.randint(1, h - outer_h - 1)
        c0 = rng.randint(1, w - outer_w - 1)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        reserved = _reserve_box(r0, c0, r1, c1)
        if any(_boxes_overlap(reserved, other) for other in used_boxes):
            continue
        sr = rng.randint(r0 + 1, r1 - 1)
        sc = rng.randint(c0 + 1, c1 - 1)
        spec = {
            "frame_color": GEN_FRAME_COLOR,
            "bbox": (r0, c0, r1, c1),
            "gap": _sample_gap(r0, c0, r1, c1, side, rng),
            "side": side,
            "seed": (sr, sc),
            "seed_color": rng.choice(seed_color_choices),
        }
        used_boxes.append(reserved)
        used_colors.add(int(spec["seed_color"]))
        return spec
    return None


def _bg_is_mode(grid: np.ndarray) -> bool:
    return get_bg(grid) == BG_COLOR


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GEN_GRID_H_RANGE)
    w = rng.randint(*GEN_GRID_W_RANGE)
    max_objects = min(GEN_OBJECT_COUNT_RANGE[1] - 1, len(GEN_COLORS), max(1, (h * w) // 90 + 1))
    min_objects = min(GEN_OBJECT_COUNT_RANGE[0], max_objects)
    if max_objects < 1:
        return None
    target_objects = rng.randint(min_objects, max_objects)
    used_boxes: list[tuple[int, int, int, int]] = []
    used_colors: set[int] = set()
    specs: list[dict] = []
    for _ in range(target_objects):
        spec = _sample_spec(h, w, used_boxes, used_colors, rng)
        if spec is None:
            return None
        specs.append(spec)

    inp = _draw_input_from_specs((h, w), specs)
    expected = _expected_from_specs((h, w), specs)

    if not _bg_is_mode(inp):
        return None
    if len(_frame_specs(inp)) != len(specs):
        return None
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None

    side_counts = Counter(spec["side"] for spec in specs)
    if sum(side_counts.values()) != len(specs):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
