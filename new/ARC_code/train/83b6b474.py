from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg, paste, random_free_location
from functools import lru_cache
import random


def _border(shape: tuple[int, int]) -> frozenset[tuple[int, int]]:
    h, w = shape
    return frozenset(
        (r, c)
        for r in range(h)
        for c in range(w)
        if r == 0 or r == h - 1 or c == 0 or c == w - 1
    )


def _candidate_outputs(
    bg: int, objects: list[dict], total_area: int
) -> list[np.ndarray]:
    outputs: list[tuple[int, np.ndarray]] = []

    for h in range(2, total_area + 1):
        for w in range(2, total_area + 1):
            if 2 * h + 2 * w - 4 != total_area:
                continue

            border = _border((h, w))
            placements_by_object: list[list[frozenset[tuple[int, int]]]] = []
            for obj in objects:
                placements: list[frozenset[tuple[int, int]]] = []
                for r0 in range(h - obj["h"] + 1):
                    for c0 in range(w - obj["w"] + 1):
                        cells = frozenset(
                            (r0 + r, c0 + c) for r, c in obj["rel"]
                        )
                        if cells <= border:
                            placements.append(cells)
                if not placements:
                    break
                placements_by_object.append(placements)
            else:
                order = sorted(
                    range(len(objects)),
                    key=lambda i: (
                        len(placements_by_object[i]),
                        -objects[i]["area"],
                        objects[i]["bbox"][0],
                        objects[i]["bbox"][1],
                    ),
                )
                solved: list[np.ndarray] = []

                def search(
                    k: int,
                    used: frozenset[tuple[int, int]],
                    chosen: list[tuple[int, frozenset[tuple[int, int]]]],
                ) -> None:
                    if k == len(order):
                        if used == border:
                            out = np.full((h, w), bg, dtype=int)
                            for obj_idx, cells in chosen:
                                for r, c in cells:
                                    out[r, c] = objects[obj_idx]["color"]
                            solved.append(out)
                        return

                    obj_idx = order[k]
                    for cells in placements_by_object[obj_idx]:
                        if cells & used:
                            continue
                        search(k + 1, used | cells, chosen + [(obj_idx, cells)])

                search(0, frozenset(), [])

                seen: set[tuple[int, ...]] = set()
                for out in solved:
                    key = tuple(map(int, out.ravel()))
                    if key in seen:
                        continue
                    seen.add(key)
                    outputs.append((abs(h - w), out))

    assert outputs, "the objects must assemble into at least one rectangular frame"
    best_balance = min(balance for balance, _out in outputs)
    best = [out for balance, out in outputs if balance == best_balance]
    unique = {(out.shape, tuple(map(int, out.ravel()))) for out in best}
    assert len(unique) == 1, "the most balanced frame assembly must be unique"
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """Translate the colored objects into the most balanced complete rectangular frame.

    Input structure
    A grid with one dominant background color and several monochromatic
    4-connected foreground objects that are straight bars or corner-like line
    sprites.

    Rule
    Preserve every foreground object exactly, allowing translation only, and
    place the objects so they cover every pixel of a one-cell-thick rectangular
    frame exactly once; the frame interior is the background color.

    Key insight
    The output shape is not copied from the input: the foreground pixel count
    determines possible frame sizes, and the intended output is the valid frame
    assembly with the least difference between height and width.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(
        g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"
    )
    assert components, "at least one foreground object is required"

    objects: list[dict] = []
    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        rel = tuple(
            sorted((int(r - r0), int(c - c0)) for r, c in comp["coords"])
        )
        objects.append(
            {
                "color": int(comp["color"]),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "rel": rel,
                "h": int(r1 - r0 + 1),
                "w": int(c1 - c0 + 1),
                "area": int(comp["area"]),
            }
        )

    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1], -obj["area"]))
    total_area = sum(obj["area"] for obj in objects)
    best = _candidate_outputs(bg, objects, total_area)
    return best[0]

# --------------- generator (att2) ---------------

# Palette and object mix.
COLOR_COUNT = 10
OBJECT_COUNT_CHOICES = (4, 5)
L_COUNTS_BY_OBJECT_COUNT = {4: (2, 3), 5: (3,)}

# Border arc and sprite shape bounds.
ARC_MIN_LEN, ARC_MAX_LEN = 1, 7
BAR_MIN_AREA, BAR_MAX_AREA = 1, 7
L_MIN_AREA, L_MAX_AREA = 3, 7
L_MIN_HEIGHT, L_MAX_HEIGHT = 2, 6
L_MIN_WIDTH, L_MAX_WIDTH = 2, 5
MIN_LENGTH_VARIETY_BY_COUNT = {4: 2, 5: 3}

# Frame, input canvas, and placement bounds.
FRAME_MIN_SIDE, FRAME_MAX_SIDE = 4, 8
FRAME_SIZE_OFFSETS = ((0, 0), (0, 1), (1, 0))
INPUT_MIN_SIDE, INPUT_MAX_SIDE, INPUT_FRAME_MARGIN = 6, 16, 2
PLACEMENT_BORDER, PLACEMENT_PADDING = 0, 0
GENERATION_ATTEMPTS = 300


def _border_coords(h: int, w: int) -> tuple[tuple[int, int], ...]:
    coords = [(0, c) for c in range(w)]
    coords += [(r, w - 1) for r in range(1, h)]
    coords += [(h - 1, c) for c in range(w - 2, -1, -1)]
    coords += [(r, 0) for r in range(h - 2, 0, -1)]
    return tuple(coords)


def _compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []

    def rec(remaining: int, n: int, prefix: list[int]) -> None:
        if n == 1:
            if ARC_MIN_LEN <= remaining <= ARC_MAX_LEN:
                out.append(tuple(prefix + [remaining]))
            return
        lo = max(ARC_MIN_LEN, remaining - ARC_MAX_LEN * (n - 1))
        hi = min(ARC_MAX_LEN, remaining - ARC_MIN_LEN * (n - 1))
        for x in range(lo, hi + 1):
            rec(remaining - x, n - 1, prefix + [x])

    rec(total, parts, [])
    return out


def _coord_bounds(coords: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows, cols = zip(*coords)
    return min(rows), min(cols), max(rows), max(cols)


def _arc_kind(coords: tuple[tuple[int, int], ...]) -> str | None:
    r0, c0, r1, c1 = _coord_bounds(coords)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    area = len(coords)

    if h == 1 or w == 1:
        return "bar" if BAR_MIN_AREA <= area <= BAR_MAX_AREA else None

    if not (
        L_MIN_AREA <= area <= L_MAX_AREA
        and L_MIN_HEIGHT <= h <= L_MAX_HEIGHT
        and L_MIN_WIDTH <= w <= L_MAX_WIDTH
    ):
        return None
    if area != h + w - 1:
        return None

    actual = {(r - r0, c - c0) for r, c in coords}
    for elbow_r, elbow_c in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        expected = {
            (r, c)
            for r in range(h)
            for c in range(w)
            if r == elbow_r or c == elbow_c
        }
        if actual == expected:
            return "L"
    return None


@lru_cache(maxsize=None)
def _valid_plans(h: int, w: int, count: int, l_count: int):
    border = _border_coords(h, w)
    perimeter = len(border)
    plans: list[tuple[tuple[tuple[int, int], ...], ...]] = []

    for start in range(perimeter):
        for lengths in _compositions(perimeter, count):
            pos = start
            arcs: list[tuple[tuple[int, int], ...]] = []
            kinds: list[str] = []
            for length in lengths:
                coords = tuple(border[(pos + i) % perimeter] for i in range(length))
                kind = _arc_kind(coords)
                if kind is None:
                    break
                arcs.append(coords)
                kinds.append(kind)
                pos = (pos + length) % perimeter
            else:
                if pos != start:
                    continue
                if kinds.count("L") != l_count or kinds.count("bar") != count - l_count:
                    continue
                if len(set(lengths)) < MIN_LENGTH_VARIETY_BY_COUNT[count]:
                    continue
                plans.append(tuple(arcs))

    return tuple(plans)


def _patch_from_arc(coords: tuple[tuple[int, int], ...], bg: int, color: int) -> np.ndarray:
    r0, c0, r1, c1 = _coord_bounds(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = color
    return patch


_FRAME_SIZES = tuple(
    dict.fromkeys(
        (n + dr, n + dc)
        for n in range(FRAME_MIN_SIDE, FRAME_MAX_SIDE + 1)
        for dr, dc in FRAME_SIZE_OFFSETS
        if n + dr <= FRAME_MAX_SIDE and n + dc <= FRAME_MAX_SIDE
    )
)


def _render_plan(h: int, w: int, bg: int, plans, colors: list[int]) -> tuple[np.ndarray, list[np.ndarray]]:
    arcs = list(random.choice(plans))
    random.shuffle(colors)
    out = np.full((h, w), bg, dtype=int)
    patches: list[np.ndarray] = []
    for coords, color in zip(arcs, colors):
        for r, c in coords:
            out[r, c] = color
        patches.append(_patch_from_arc(coords, bg, color))
    return out, patches


def _gen_attempt() -> np.ndarray | None:
    count = random.choice(OBJECT_COUNT_CHOICES)
    l_choices = list(L_COUNTS_BY_OBJECT_COUNT[count])
    random.shuffle(l_choices)

    sizes = list(_FRAME_SIZES)
    random.shuffle(sizes)
    bg = random.randrange(COLOR_COUNT)
    colors = random.sample([c for c in range(COLOR_COUNT) if c != bg], count)

    for h, w in sizes:
        for l_count in l_choices:
            plans = _valid_plans(h, w, count, l_count)
            if not plans:
                continue

            out, patches = _render_plan(h, w, bg, plans, colors)
            H = random.randint(max(INPUT_MIN_SIDE, h + INPUT_FRAME_MARGIN), INPUT_MAX_SIDE)
            W = random.randint(max(INPUT_MIN_SIDE, w + INPUT_FRAME_MARGIN), INPUT_MAX_SIDE)
            inp = np.full((H, W), bg, dtype=int)
            try:
                for patch in random.sample(patches, len(patches)):
                    r, c = random_free_location(
                        inp,
                        patch,
                        transparent=bg,
                        border=PLACEMENT_BORDER,
                        padding=PLACEMENT_PADDING,
                    )
                    paste(inp, patch, r, c, transparent=bg)
            except ValueError:
                continue

            try:
                solved = main(inp)
            except Exception:
                continue
            if np.array_equal(solved, out):
                return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 83b6b474 input")
