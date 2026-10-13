from __future__ import annotations

import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_BY_SIDE = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

WALL_COLOR = 5
BG_CHOICES = (0, 1, 2, 8)
GRID_SIDE_RANGE = (18, 30)
OBJECT_COUNT_RANGE = (1, 4)
SEED_COUNT_RANGE = (1, 8)
INNER_H_RANGE = (2, 5)
INNER_W_RANGE = (2, 6)
GEN_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 200
CLIPPED_PROB = 0.35


def _get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(colors[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _perimeter_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        if r1 != r0:
            coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        if c1 != c0:
            coords.append((r, c1))
    return coords


def _gap_side(gap: tuple[int, int], bbox: tuple[int, int, int, int]) -> str | None:
    r, c = gap
    r0, c0, r1, c1 = bbox
    if r == r0 and c0 < c < c1:
        return "up"
    if r == r1 and c0 < c < c1:
        return "down"
    if c == c0 and r0 < r < r1:
        return "left"
    if c == c1 and r0 < r < r1:
        return "right"
    return None


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames: list[dict[str, object]] = []
    for wall_color in sorted(set(map(int, np.unique(g))) - {bg}):
        for comp in _components(g == wall_color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue

            if any(r not in (r0, r1) and c not in (c0, c1) for r, c in comp):
                continue

            perimeter = _perimeter_coords(r0, c0, r1, c1)
            if len(comp) != len(perimeter) - 1:
                continue

            missing = [(r, c) for r, c in perimeter if int(g[r, c]) != wall_color]
            if len(missing) != 1:
                continue
            gap = missing[0]
            if int(g[gap]) != bg:
                continue

            side = _gap_side(gap, (r0, c0, r1, c1))
            if side is None:
                continue

            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            if interior.size == 0:
                continue
            seed_mask = (interior != bg) & (interior != wall_color)
            if not np.any(seed_mask):
                continue
            seed_colors = set(map(int, np.unique(interior[seed_mask])))
            if len(seed_colors) != 1:
                continue
            seed_color = seed_colors.pop()
            seed_count = int(np.count_nonzero(interior == seed_color))
            frames.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "gap": gap,
                    "side": side,
                    "wall_color": wall_color,
                    "seed_color": seed_color,
                    "seed_count": seed_count,
                }
            )
    frames.sort(key=lambda frame: tuple(frame["bbox"]))  # type: ignore[arg-type]
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow an outward branch from each rectangular frame gap.

    Each frame's interior seeds encode the branch color and length:
    - gap side -> growth direction
    - seed color -> branch color
    - seed count -> branch length
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape

    for frame in _find_frames(g):
        gap_r, gap_c = frame["gap"]  # type: ignore[misc]
        seed_color = int(frame["seed_color"])
        seed_count = int(frame["seed_count"])
        dr, dc = DIR_BY_SIDE[str(frame["side"])]
        for step in range(1, seed_count + 1):
            rr = gap_r + dr * step
            cc = gap_c + dc * step
            if not (0 <= rr < h and 0 <= cc < w):
                break
            out[rr, cc] = seed_color

    return out


def _sample_inner_shape(seed_count: int) -> tuple[int, int] | None:
    for _ in range(40):
        inner_h = random.randint(*INNER_H_RANGE)
        inner_w = random.randint(*INNER_W_RANGE)
        if inner_h * inner_w > seed_count:
            return inner_h, inner_w
    return None


def _reserve_footprint(
    h: int,
    w: int,
    bbox: tuple[int, int, int, int],
    gap: tuple[int, int],
    side: str,
    branch_len: int,
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: set[tuple[int, int]] = set()
    for r in range(max(0, r0 - 1), min(h - 1, r1 + 1) + 1):
        for c in range(max(0, c0 - 1), min(w - 1, c1 + 1) + 1):
            cells.add((r, c))

    dr, dc = DIR_BY_SIDE[side]
    gap_r, gap_c = gap
    for step in range(1, branch_len + 1):
        rr = gap_r + dr * step
        cc = gap_c + dc * step
        if not (0 <= rr < h and 0 <= cc < w):
            break
        for pr in range(rr - 1, rr + 2):
            for pc in range(cc - 1, cc + 2):
                if 0 <= pr < h and 0 <= pc < w:
                    cells.add((pr, pc))
    return cells


def _sample_spec(
    h: int,
    w: int,
    seed_color: int,
    side: str,
    reserved: set[tuple[int, int]],
) -> tuple[dict[str, object], set[tuple[int, int]]] | None:
    for _ in range(PLACEMENT_ATTEMPTS):
        seed_count = random.randint(*SEED_COUNT_RANGE)
        inner_shape = _sample_inner_shape(seed_count)
        if inner_shape is None:
            continue
        inner_h, inner_w = inner_shape
        box_h, box_w = inner_h + 2, inner_w + 2

        if side in ("up", "down"):
            if w - box_w - 1 < 1:
                continue
            available_max = h - box_h - 1
        else:
            if h - box_h - 1 < 1:
                continue
            available_max = w - box_w - 1
        if available_max < 1:
            continue

        clipped = random.random() < CLIPPED_PROB
        if clipped:
            clipped_hi = min(seed_count - 1, available_max)
            if clipped_hi < 1:
                clipped = False
        if not clipped and available_max < seed_count:
            continue

        if clipped:
            available = random.randint(1, clipped_hi)
        else:
            available = random.randint(seed_count, available_max)

        if side == "up":
            r0 = available
            c0 = random.randint(1, w - box_w - 1)
            gap = (r0, random.randint(c0 + 1, c0 + box_w - 2))
        elif side == "down":
            r0 = h - box_h - available
            c0 = random.randint(1, w - box_w - 1)
            gap = (r0 + box_h - 1, random.randint(c0 + 1, c0 + box_w - 2))
        elif side == "left":
            r0 = random.randint(1, h - box_h - 1)
            c0 = available
            gap = (random.randint(r0 + 1, r0 + box_h - 2), c0)
        else:
            r0 = random.randint(1, h - box_h - 1)
            c0 = w - box_w - available
            gap = (random.randint(r0 + 1, r0 + box_h - 2), c0 + box_w - 1)

        bbox = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
        branch_len = min(seed_count, available)
        footprint = _reserve_footprint(h, w, bbox, gap, side, branch_len)
        if footprint & reserved:
            continue

        interior_cells = [(r, c) for r in range(r0 + 1, r0 + box_h - 1) for c in range(c0 + 1, c0 + box_w - 1)]
        seed_positions = random.sample(interior_cells, seed_count)
        spec = {
            "bbox": bbox,
            "gap": gap,
            "side": side,
            "wall_color": WALL_COLOR,
            "seed_color": seed_color,
            "seed_count": seed_count,
            "seed_positions": seed_positions,
        }
        return spec, footprint
    return None


def _render_input(h: int, w: int, bg: int, specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]  # type: ignore[misc]
        gap = spec["gap"]  # type: ignore[misc]
        wall_color = int(spec["wall_color"])
        seed_color = int(spec["seed_color"])
        for r, c in _perimeter_coords(r0, c0, r1, c1):
            if (r, c) != gap:
                grid[r, c] = wall_color
        for r, c in spec["seed_positions"]:  # type: ignore[index]
            grid[r, c] = seed_color
    return grid


def _expected_output(inp: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    h, w = out.shape
    for spec in specs:
        gap_r, gap_c = spec["gap"]  # type: ignore[misc]
        seed_color = int(spec["seed_color"])
        seed_count = int(spec["seed_count"])
        dr, dc = DIR_BY_SIDE[str(spec["side"])]
        for step in range(1, seed_count + 1):
            rr = gap_r + dr * step
            cc = gap_c + dc * step
            if not (0 <= rr < h and 0 <= cc < w):
                break
            out[rr, cc] = seed_color
    return out


def _unique_bg_is_dominant(grid: np.ndarray, bg: int) -> bool:
    colors, counts = np.unique(grid, return_counts=True)
    max_count = int(np.max(counts))
    winners = {int(color) for color, count in zip(colors, counts) if int(count) == max_count}
    return winners == {bg}


def _spec_signature(spec: dict[str, object]) -> tuple[tuple[int, int, int, int], tuple[int, int], str, int, int]:
    return (
        tuple(spec["bbox"]),  # type: ignore[arg-type]
        tuple(spec["gap"]),  # type: ignore[arg-type]
        str(spec["side"]),
        int(spec["seed_color"]),
        int(spec["seed_count"]),
    )


def _gen_attempt() -> np.ndarray | None:
    object_count = random.randint(*OBJECT_COUNT_RANGE)
    min_side = max(GRID_SIDE_RANGE[0], 10 + 4 * object_count)
    if min_side > GRID_SIDE_RANGE[1]:
        return None

    h = random.randint(min_side, GRID_SIDE_RANGE[1])
    w = random.randint(min_side, GRID_SIDE_RANGE[1])
    bg = random.choice(BG_CHOICES)
    seed_palette = [color for color in range(10) if color not in {bg, WALL_COLOR}]
    seed_colors = random.sample(seed_palette, k=object_count)
    sides = random.sample(list(DIR_BY_SIDE.keys()), k=object_count)

    specs: list[dict[str, object]] = []
    reserved: set[tuple[int, int]] = set()
    for seed_color, side in zip(seed_colors, sides):
        sampled = _sample_spec(h, w, seed_color, side, reserved)
        if sampled is None:
            return None
        spec, footprint = sampled
        specs.append(spec)
        reserved |= footprint

    inp = _render_input(h, w, bg, specs)
    expected = _expected_output(inp, specs)

    if not _unique_bg_is_dominant(inp, bg):
        return None
    if np.array_equal(inp, expected):
        return None

    found = _find_frames(inp)
    if len(found) != len(specs):
        return None
    if sorted(_spec_signature(spec) for spec in specs) != sorted(_spec_signature(spec) for spec in found):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
