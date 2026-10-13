from __future__ import annotations

from collections import Counter
import numpy as np


GREY = 5
DIRS = ("up", "down", "left", "right")
MAX_ATTEMPTS = 400
PLACEMENT_TRIES = 80


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _ring_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    if not coords:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    expected_area = 2 * (h + w) - 4
    coord_set = set(coords)
    if len(coord_set) != expected_area:
        return None
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            on_border = rr in (r0, r1) or cc in (c0, c1)
            if on_border != ((rr, cc) in coord_set):
                return None
    return r0, c0, r1, c1


def _find_fences(grid: np.ndarray, bg: int) -> list[dict]:
    fences: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == int(bg):
            continue
        for comp in _color_components(grid, color):
            bbox = _ring_bbox(comp)
            if bbox is None:
                continue
            fences.append(
                {
                    "color": color,
                    "bbox": bbox,
                    "coords": [(int(r), int(c)) for r, c in comp],
                }
            )
    fences.sort(key=lambda fence: tuple(int(v) for v in fence["bbox"]))
    return fences


def _expel_destination(
    r: int, c: int, bbox: tuple[int, int, int, int]
) -> tuple[str, tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    distances = [
        ("up", int(r - r0)),
        ("down", int(r1 - r)),
        ("left", int(c - c0)),
        ("right", int(c1 - c)),
    ]
    best = min(dist for _, dist in distances)
    for direction, dist in distances:
        if dist != int(best):
            continue
        if direction == "up":
            return direction, (int(r0 - 1), int(c))
        if direction == "down":
            return direction, (int(r1 + 1), int(c))
        if direction == "left":
            return direction, (int(r), int(c0 - 1))
        return direction, (int(r), int(c1 + 1))
    raise AssertionError("unreachable")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each rectangular fence by the majority flower color and expel minorities.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = _most_common_color(g)

    for fence in _find_fences(g, bg):
        r0, c0, r1, c1 = map(int, fence["bbox"])
        fence_color = int(fence["color"])

        flowers: list[tuple[int, int, int]] = []
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                val = int(g[r, c])
                if val != int(bg) and val != int(fence_color):
                    flowers.append((int(r), int(c), val))
        if not flowers:
            continue

        counts = Counter(val for _, _, val in flowers)
        max_count = max(int(v) for v in counts.values())
        majority_color = next(val for _r, _c, val in flowers if int(counts[val]) == int(max_count))

        for r, c in fence["coords"]:
            out[int(r), int(c)] = int(majority_color)

        moves: list[tuple[int, int, int]] = []
        for r, c, color in flowers:
            if int(color) == int(majority_color):
                continue
            out[int(r), int(c)] = int(bg)
            _direction, (rr, cc) = _expel_destination(int(r), int(c), fence["bbox"])
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                moves.append((int(rr), int(cc), int(color)))

        for rr, cc, color in moves:
            out[int(rr), int(cc)] = int(color)

    return out


def _choice_inclusive(rng: np.random.Generator, low: int, high: int) -> int:
    return int(rng.integers(int(low), int(high) + 1))


def _direction_candidates(
    bbox: tuple[int, int, int, int]
) -> dict[str, list[tuple[tuple[int, int], tuple[int, int]]]]:
    r0, c0, r1, c1 = bbox
    out: dict[str, list[tuple[tuple[int, int], tuple[int, int]]]] = {d: [] for d in DIRS}
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            distances = {
                "up": int(r - r0),
                "down": int(r1 - r),
                "left": int(c - c0),
                "right": int(c1 - c),
            }
            best = min(distances.values())
            winners = [d for d, dist in distances.items() if int(dist) == int(best)]
            if len(winners) != 1:
                continue
            direction = winners[0]
            if direction == "up":
                dest = (int(r0 - 1), int(c))
            elif direction == "down":
                dest = (int(r1 + 1), int(c))
            elif direction == "left":
                dest = (int(r), int(c0 - 1))
            else:
                dest = (int(r), int(c1 + 1))
            out[direction].append(((int(r), int(c)), dest))
    return out


def _sample_fence_spec(
    rng: np.random.Generator,
    bg: int,
    bbox: tuple[int, int, int, int],
) -> dict | None:
    r0, c0, r1, c1 = bbox
    interior_h = r1 - r0 - 1
    interior_w = c1 - c0 - 1
    interior_area = interior_h * interior_w
    max_flowers = max(3, interior_area // 2)
    max_minor = min(4, max_flowers - 2)
    if max_minor < 1:
        return None

    candidates = _direction_candidates(bbox)
    available_dirs = [d for d in DIRS if candidates[d]]
    if not available_dirs:
        return None

    n_minor = _choice_inclusive(rng, 1, min(len(available_dirs), max_minor))
    chosen_dirs = [str(x) for x in rng.choice(np.array(available_dirs, dtype=object), size=n_minor, replace=False)]

    minority_positions: list[tuple[int, int]] = []
    minority_dests: list[tuple[int, int]] = []
    for direction in chosen_dirs:
        options = candidates[direction]
        pos, dest = options[int(rng.integers(0, len(options)))]
        minority_positions.append((int(pos[0]), int(pos[1])))
        minority_dests.append((int(dest[0]), int(dest[1])))

    majority_max = min(5, max_flowers - n_minor, interior_area - n_minor)
    if majority_max < 2:
        return None
    majority_count = _choice_inclusive(rng, 2, majority_max)

    palette = [color for color in range(10) if color not in (int(bg), GREY)]
    if len(palette) < n_minor + 1:
        return None
    chosen_colors = [int(x) for x in rng.choice(np.array(palette, dtype=int), size=n_minor + 1, replace=False)]
    majority_color = int(chosen_colors[0])
    minority_colors = [int(x) for x in chosen_colors[1:]]

    all_cells = [(int(r), int(c)) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    used = set(minority_positions)
    free_cells = [cell for cell in all_cells if cell not in used]
    if len(free_cells) < majority_count:
        return None
    chosen_idx = rng.choice(len(free_cells), size=majority_count, replace=False)
    majority_positions = [free_cells[int(i)] for i in np.atleast_1d(chosen_idx)]

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "majority_color": int(majority_color),
        "majority_positions": [(int(r), int(c)) for r, c in majority_positions],
        "minority": [
            {
                "pos": (int(pos[0]), int(pos[1])),
                "color": int(color),
                "dest": (int(dest[0]), int(dest[1])),
            }
            for pos, color, dest in zip(minority_positions, minority_colors, minority_dests)
        ],
    }


def _draw_ring(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[int(r0), int(c0) : int(c1) + 1] = int(color)
    grid[int(r1), int(c0) : int(c1) + 1] = int(color)
    grid[int(r0) : int(r1) + 1, int(c0)] = int(color)
    grid[int(r0) : int(r1) + 1, int(c1)] = int(color)


def _render_input(h: int, w: int, bg: int, specs: list[dict]) -> np.ndarray:
    grid = np.full((h, w), int(bg), dtype=int)
    for spec in specs:
        _draw_ring(grid, spec["bbox"], GREY)
        for r, c in spec["majority_positions"]:
            grid[int(r), int(c)] = int(spec["majority_color"])
        for flower in spec["minority"]:
            r, c = flower["pos"]
            grid[int(r), int(c)] = int(flower["color"])
    return grid


def _render_output(h: int, w: int, bg: int, specs: list[dict]) -> np.ndarray:
    grid = np.full((h, w), int(bg), dtype=int)
    for spec in specs:
        _draw_ring(grid, spec["bbox"], int(spec["majority_color"]))
        for r, c in spec["majority_positions"]:
            grid[int(r), int(c)] = int(spec["majority_color"])
        for flower in spec["minority"]:
            rr, cc = flower["dest"]
            grid[int(rr), int(cc)] = int(flower["color"])
    return grid


def _background_is_dominant(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(int(v) for v in grid.ravel())
    bg_count = int(counts[int(bg)])
    return all(int(bg_count) > int(count) for color, count in counts.items() if int(color) != int(bg))


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = _choice_inclusive(rng, 14, 24)
    w = _choice_inclusive(rng, 14, 24)
    bg_choices = [color for color in range(10) if color != GREY]
    bg = int(rng.choice(np.array(bg_choices, dtype=int)))

    max_fences = 4 if h * w >= 320 else 3
    n_fences = _choice_inclusive(rng, 1, max_fences)

    occupied = np.zeros((h, w), dtype=bool)
    specs: list[dict] = []
    for _ in range(n_fences):
        placed = False
        for _ in range(PLACEMENT_TRIES):
            outer_h = _choice_inclusive(rng, 5, min(10, h - 2))
            outer_w = _choice_inclusive(rng, 5, min(10, w - 2))
            if h - outer_h < 2 or w - outer_w < 2:
                continue
            r0 = int(rng.integers(1, h - outer_h))
            c0 = int(rng.integers(1, w - outer_w))
            r1 = int(r0 + outer_h - 1)
            c1 = int(c0 + outer_w - 1)
            if occupied[int(r0 - 1) : int(r1 + 2), int(c0 - 1) : int(c1 + 2)].any():
                continue
            spec = _sample_fence_spec(rng, bg, (int(r0), int(c0), int(r1), int(c1)))
            if spec is None:
                continue
            occupied[int(r0 - 1) : int(r1 + 2), int(c0 - 1) : int(c1 + 2)] = True
            specs.append(spec)
            placed = True
            break
        if not placed:
            return None

    inp = _render_input(h, w, bg, specs)
    expected = _render_output(h, w, bg, specs)

    if _most_common_color(inp) != int(bg):
        return None
    if not _background_is_dominant(inp, bg):
        return None
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
