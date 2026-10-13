from __future__ import annotations

import random

import numpy as np


GREEN = 3
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
SIDES = ("top", "bottom", "left", "right")


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: set[tuple[int, int]] = set()
            while stack:
                cr, cc = stack.pop()
                comp.add((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((int(r0), int(c)))
        cells.add((int(r1), int(c)))
    for r in range(r0 + 1, r1):
        cells.add((int(r), int(c0)))
        cells.add((int(r), int(c1)))
    return cells


def _inner_ring_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    inner_r0 = r0 + 1
    inner_c0 = c0 + 1
    inner_r1 = r1 - 1
    inner_c1 = c1 - 1
    assert inner_r0 <= inner_r1 and inner_c0 <= inner_c1, "wall must enclose at least one interior cell"

    cells: set[tuple[int, int]] = set()
    for r in range(inner_r0, inner_r1 + 1):
        for c in range(inner_c0, inner_c1 + 1):
            if r in (inner_r0, inner_r1) or c in (inner_c0, inner_c1):
                cells.add((int(r), int(c)))
    return cells


def _gap_from_spec(spec: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = spec["bbox"]
    side = spec["gap_side"]
    offset = spec["gap_offset"]
    if side == "top":
        return int(r0), int(c0 + offset)
    if side == "bottom":
        return int(r1), int(c0 + offset)
    if side == "left":
        return int(r0 + offset), int(c0)
    if side == "right":
        return int(r0 + offset), int(c1)
    raise ValueError(side)


def _marker_from_spec(spec: dict) -> tuple[int, int]:
    gap_r, gap_c = _gap_from_spec(spec)
    side = spec["gap_side"]
    if side == "top":
        return int(gap_r - 1), int(gap_c)
    if side == "bottom":
        return int(gap_r + 1), int(gap_c)
    if side == "left":
        return int(gap_r), int(gap_c - 1)
    if side == "right":
        return int(gap_r), int(gap_c + 1)
    raise ValueError(side)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Lay a colored pipe around the inside of each green rectangular wall.

    Input structure:
    - The background color is the most frequent value.
    - Green pixels form one-pixel-thick rectangular walls with exactly one missing border pixel.
    - A single colored marker pixel sits directly outside each gap.

    Rule:
    - Find the missing wall pixel and read the color of its outside marker.
    - Remove the marker.
    - Fill the wall's inner ring with the marker color while keeping the green wall and the gap unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    walls = _components(g == GREEN)
    assert walls, "expected at least one green wall"

    for comp in walls:
        coords = np.array(sorted(comp), dtype=int)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        border = _border_cells(int(r0), int(c0), int(r1), int(c1))
        missing = list(border - comp)
        extra = comp - border
        assert not extra, "green component must lie on its bounding rectangle border"
        assert len(missing) == 1, "each wall must have exactly one gap"
        gap_r, gap_c = missing[0]

        marker_candidates: list[tuple[int, int, int]] = []
        for dr, dc in DIRS4:
            nr = gap_r + dr
            nc = gap_c + dc
            if not (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]):
                continue
            if int(r0) <= nr <= int(r1) and int(c0) <= nc <= int(c1):
                continue
            value = int(g[nr, nc])
            if value not in {bg, GREEN}:
                marker_candidates.append((int(nr), int(nc), value))
        assert len(marker_candidates) == 1, "each gap must have one outside marker"
        mr, mc, pipe_color = marker_candidates[0]
        out[mr, mc] = bg

        for pr, pc in _inner_ring_cells(int(r0), int(c0), int(r1), int(c1)):
            out[pr, pc] = pipe_color

    return out


# --------------- generator ---------------

MIN_SIDE = 14
MAX_SIDE = 26
MAX_WALLS = 4
PLACEMENT_PAD = 2
MAX_GENERATION_ATTEMPTS = 400

ROOMY_MAX_INNER = {1: 8, 2: 7, 3: 6, 4: 5}
THIN_MAX_LONG = {1: 9, 2: 8, 3: 7, 4: 6}


def _sample_roomy_flags(n_walls: int) -> list[bool]:
    if n_walls == 1:
        return [bool(random.getrandbits(1))]
    roomy = [True] * (n_walls // 2)
    thin = [False] * (n_walls // 2)
    flags = roomy + thin
    if n_walls % 2:
        flags.append(bool(random.getrandbits(1)))
    random.shuffle(flags)
    return flags


def _sample_wall_spec(pipe_color: int, roomy: bool, n_walls: int) -> dict:
    if roomy:
        max_inner = ROOMY_MAX_INNER[n_walls]
        inner_h = random.randint(3, max_inner)
        inner_w = random.randint(3, max_inner)
    else:
        max_long = THIN_MAX_LONG[n_walls]
        if bool(random.getrandbits(1)):
            inner_h = 2
            inner_w = random.randint(2, max_long)
        else:
            inner_h = random.randint(2, max_long)
            inner_w = 2

    outer_h = inner_h + 2
    outer_w = inner_w + 2
    gap_side = random.choice(SIDES)
    if gap_side in {"top", "bottom"}:
        gap_offset = random.randint(1, outer_w - 2)
    else:
        gap_offset = random.randint(1, outer_h - 2)

    return {
        "pipe_color": int(pipe_color),
        "inner_h": int(inner_h),
        "inner_w": int(inner_w),
        "outer_h": int(outer_h),
        "outer_w": int(outer_w),
        "gap_side": gap_side,
        "gap_offset": int(gap_offset),
    }


def _can_place(occupied: np.ndarray, r0: int, c0: int, outer_h: int, outer_w: int) -> bool:
    h, w = occupied.shape
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    pr0 = max(0, r0 - PLACEMENT_PAD)
    pc0 = max(0, c0 - PLACEMENT_PAD)
    pr1 = min(h - 1, r1 + PLACEMENT_PAD)
    pc1 = min(w - 1, c1 + PLACEMENT_PAD)
    return not bool(np.any(occupied[pr0 : pr1 + 1, pc0 : pc1 + 1]))


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, outer_h: int, outer_w: int) -> None:
    h, w = occupied.shape
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    pr0 = max(0, r0 - PLACEMENT_PAD)
    pc0 = max(0, c0 - PLACEMENT_PAD)
    pr1 = min(h - 1, r1 + PLACEMENT_PAD)
    pc1 = min(w - 1, c1 + PLACEMENT_PAD)
    occupied[pr0 : pr1 + 1, pc0 : pc1 + 1] = True


def _place_specs(height: int, width: int, specs: list[dict]) -> list[dict] | None:
    occupied = np.zeros((height, width), dtype=bool)
    placed: list[dict] = []

    for spec in sorted(specs, key=lambda s: (s["outer_h"] * s["outer_w"], s["outer_h"], s["outer_w"]), reverse=True):
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, height - spec["outer_h"]):
            for c0 in range(1, width - spec["outer_w"]):
                if _can_place(occupied, int(r0), int(c0), int(spec["outer_h"]), int(spec["outer_w"])):
                    candidates.append((int(r0), int(c0)))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        spec = dict(spec)
        spec["bbox"] = (
            int(r0),
            int(c0),
            int(r0 + spec["outer_h"] - 1),
            int(c0 + spec["outer_w"] - 1),
        )
        mr, mc = _marker_from_spec(spec)
        if not (0 <= mr < height and 0 <= mc < width):
            return None
        placed.append(spec)
        _mark_occupied(occupied, int(r0), int(c0), int(spec["outer_h"]), int(spec["outer_w"]))

    return placed


def _render_pair(height: int, width: int, bg: int, specs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((height, width), bg, dtype=int)
    out = np.full((height, width), bg, dtype=int)

    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        gap_r, gap_c = _gap_from_spec(spec)
        mr, mc = _marker_from_spec(spec)
        pipe_color = spec["pipe_color"]

        for canvas in (inp, out):
            canvas[r0 : r1 + 1, c0] = GREEN
            canvas[r0 : r1 + 1, c1] = GREEN
            canvas[r0, c0 : c1 + 1] = GREEN
            canvas[r1, c0 : c1 + 1] = GREEN
            canvas[gap_r, gap_c] = bg

        inp[mr, mc] = pipe_color
        for pr, pc in _inner_ring_cells(int(r0), int(c0), int(r1), int(c1)):
            out[pr, pc] = pipe_color

    return inp, out


def _bg_is_dominant(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    count_map = {int(v): int(n) for v, n in zip(values, counts)}
    bg_count = count_map[int(bg)]
    return all(bg_count > n for color, n in count_map.items() if color != int(bg))


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(MIN_SIDE, MAX_SIDE)
    width = random.randint(MIN_SIDE, MAX_SIDE)
    max_walls = min(MAX_WALLS, max(1, (height * width) // 120))
    n_walls = random.randint(1, max_walls)

    bg_choices = [color for color in range(10) if color != GREEN]
    bg = random.choice(bg_choices)
    pipe_choices = [color for color in range(10) if color not in {bg, GREEN}]
    pipe_colors = random.sample(pipe_choices, n_walls)

    roomy_flags = _sample_roomy_flags(n_walls)
    specs = [_sample_wall_spec(pipe_color, roomy, n_walls) for pipe_color, roomy in zip(pipe_colors, roomy_flags)]
    placed = _place_specs(height, width, specs)
    if placed is None:
        return None

    inp, expected = _render_pair(height, width, int(bg), placed)
    if not _bg_is_dominant(inp, int(bg)):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input")
