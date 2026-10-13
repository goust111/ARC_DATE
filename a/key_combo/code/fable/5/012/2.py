from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
COUNTER_COLOR = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if (dr, dc) != (0, 0)
)
GROWTH_STEPS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATION_ATTEMPTS = 500


def _components8(mask: np.ndarray) -> list[np.ndarray]:
    """Return 8-connected components as arrays of canonical (row, column) coordinates."""
    assert mask.ndim == 2
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for start_r, start_c in np.argwhere(mask):
        start = (int(start_r), int(start_c))
        if seen[start]:
            continue
        seen[start] = True
        stack = [start]
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for dr, dc in NEIGHBORS8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        components.append(np.asarray(coords, dtype=int))
    return components


def _extract_objects(g: np.ndarray) -> tuple[list[dict], int]:
    """Parse and validate the visible objects and isolated gray counter pixels."""
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    assert np.all((0 <= g) & (g <= 9)), "expected ARC palette colors"

    counter_components = _components8(g == COUNTER_COLOR)
    assert counter_components, "expected at least one gray counter pixel"
    assert all(len(comp) == 1 for comp in counter_components), "gray counter pixels must be mutually nonadjacent"

    # A counter pixel must be the sole foreground cell in its 3x3 neighborhood.
    h, w = g.shape
    for comp in counter_components:
        r, c = map(int, comp[0])
        window = g[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]
        assert int(np.count_nonzero(window)) == 1, "gray counters must not touch colored objects"

    objects: list[dict] = []
    colors = [int(color) for color in np.unique(g) if int(color) not in (BACKGROUND, COUNTER_COLOR)]
    for color in colors:
        components = _components8(g == color)
        assert len(components) == 1, "each object color must identify exactly one 8-connected object"
        coords = components[0]
        assert len(coords) >= 2, "colored objects must have area at least two"
        objects.append({"color": color, "area": int(len(coords)), "coords": coords})

    assert len(objects) >= 2, "expected several colored objects"
    areas = [obj["area"] for obj in objects]
    assert len(set(areas)) == len(areas), "object areas must be pairwise distinct"
    return objects, len(counter_components)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate object colors around the ascending-area cycle by the number of isolated gray pixels.

    Input structure:
    - Black is the background. Each non-gray foreground color forms one 8-connected
      monochrome object of area at least two; object colors and areas are pairwise distinct.
    - Gray cells are isolated singleton counter pixels, mutually nonadjacent and not
      adjacent to any colored object; gray is not an object color.

    Rule:
    - Sort the colored objects by area from smallest to largest and regard that ranking
      as a cycle. If there are k gray pixels, send the original color of rank i to
      rank i+k, wrapping cyclically. Preserve every object mask and erase all gray pixels.

    Key insight:
    - The gray singleton count is the rotation distance on the area-ranked object cycle,
      rather than noise or a fixed adjacent swap.
    """
    g = np.asarray(grid, dtype=int)
    objects, shift = _extract_objects(g)
    ranked = sorted(objects, key=lambda obj: obj["area"])

    out = g.copy()
    out[out == COUNTER_COLOR] = BACKGROUND
    count = len(ranked)
    for source_rank, source in enumerate(ranked):
        destination = ranked[(source_rank + shift) % count]
        coords = destination["coords"]
        out[coords[:, 0], coords[:, 1]] = source["color"]
    return out


# ------------------------------ generator ------------------------------

_generation_count = 0


def _random_polyomino(area: int) -> np.ndarray:
    """Grow a compact 4-connected mask with an exact requested area."""
    for _ in range(80):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in GROWTH_STEPS4:
                    candidate = (r + dr, c + dc)
                    if candidate not in cells:
                        frontier.add(candidate)
            if not frontier:
                break

            # Prefer compact growth most of the time, while still allowing arms and bends.
            choices = list(frontier)
            if random.random() < 0.72:
                min_radius = min(max(abs(r), abs(c)) for r, c in choices)
                compact = [(r, c) for r, c in choices if max(abs(r), abs(c)) <= min_radius + 1]
                choices = compact or choices
            cells.add(random.choice(choices))

        rows = [r for r, _c in cells]
        cols = [c for _r, c in cells]
        h = max(rows) - min(rows) + 1
        w = max(cols) - min(cols) + 1
        if len(cells) != area or (area >= 3 and min(h, w) == 1):
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r - min(rows), c - min(cols)] = True
        turns = random.randrange(4)
        mask = np.rot90(mask, turns)
        if random.choice((False, True)):
            mask = np.fliplr(mask)
        return mask.copy()
    raise RuntimeError("failed to grow a suitable object mask")


def _place_mask(canvas: np.ndarray, mask: np.ndarray, color: int) -> np.ndarray | None:
    """Place a mask with an empty one-cell moat around its bounding box."""
    h, w = canvas.shape
    mh, mw = mask.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(1, h - mh):
        for c0 in range(1, w - mw):
            moat = canvas[r0 - 1 : r0 + mh + 1, c0 - 1 : c0 + mw + 1]
            if np.all(moat == BACKGROUND):
                candidates.append((r0, c0))
    if not candidates:
        return None

    r0, c0 = random.choice(candidates)
    coords = np.argwhere(mask) + np.array([r0, c0], dtype=int)
    canvas[coords[:, 0], coords[:, 1]] = int(color)
    return coords


def _place_counters(canvas: np.ndarray, count: int) -> list[tuple[int, int]] | None:
    """Scatter singleton counters with Chebyshev distance at least two from foreground."""
    h, w = canvas.shape
    candidates = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []

    for r, c in candidates:
        if canvas[r, c] != BACKGROUND:
            continue
        if np.any(canvas[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] != BACKGROUND):
            continue
        if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in chosen):
            continue
        chosen.append((r, c))
        if len(chosen) == count:
            for rr, cc in chosen:
                canvas[rr, cc] = COUNTER_COLOR
            return chosen
    return None


def _oracle_output(grid: np.ndarray, objects: list[dict], shift: int, *, order: str = "area") -> np.ndarray:
    """Construct the intended result directly from generator records, independently of parsing."""
    if order == "area":
        ranked = sorted(objects, key=lambda obj: obj["area"])
    elif order == "space":
        ranked = sorted(
            objects,
            key=lambda obj: (
                int(obj["coords"][:, 0].min()),
                int(obj["coords"][:, 1].min()),
            ),
        )
    else:
        raise ValueError("unknown oracle ordering")

    out = grid.copy()
    out[out == COUNTER_COLOR] = BACKGROUND
    n = len(ranked)
    for i, source in enumerate(ranked):
        destination = ranked[(i + shift) % n]
        coords = destination["coords"]
        out[coords[:, 0], coords[:, 1]] = source["color"]
    return out


def _semantic_validation(grid: np.ndarray, objects: list[dict], shift: int, expected: np.ndarray) -> bool:
    """Check the rule's visible contract and reject several plausible shortcut explanations."""
    try:
        parsed, parsed_shift = _extract_objects(grid)
        if parsed_shift != shift or len(parsed) != len(objects):
            return False
        if not (grid.shape == expected.shape and max(grid.shape) <= 30):
            return False

        # Generator records must match the independently parsed colored masks exactly.
        recorded = {
            obj["color"]: {tuple(map(int, rc)) for rc in obj["coords"]}
            for obj in objects
        }
        observed = {
            obj["color"]: {tuple(map(int, rc)) for rc in obj["coords"]}
            for obj in parsed
        }
        if recorded != observed:
            return False

        n = len(objects)
        if shift % n in (0, 1) or (2 * shift) % n == 0:
            return False
        if not np.array_equal(expected, _oracle_output(grid, objects, shift, order="area")):
            return False

        object_support = (grid != BACKGROUND) & (grid != COUNTER_COLOR)
        if not np.array_equal(expected != BACKGROUND, object_support):
            return False
        if np.any(expected == COUNTER_COLOR):
            return False

        # Fixed one-step rotation, reverse direction, no rotation, and spatial ranking
        # must all produce visibly different results for this individual example.
        alternatives = (
            _oracle_output(grid, objects, 1, order="area"),
            _oracle_output(grid, objects, -shift, order="area"),
            _oracle_output(grid, objects, 0, order="area"),
            _oracle_output(grid, objects, shift, order="space"),
        )
        if any(np.array_equal(expected, candidate) for candidate in alternatives):
            return False

        # Avoid accidental numeric color/area correlation: color IDs are labels only.
        area_rank_colors = [obj["color"] for obj in sorted(objects, key=lambda obj: obj["area"])]
        if area_rank_colors in (sorted(area_rank_colors), sorted(area_rank_colors, reverse=True)):
            return False

        return bool(np.array_equal(main(grid), expected))
    except (AssertionError, RuntimeError, ValueError):
        return False


def _choose_shift(n_objects: int, coverage_mode: int) -> int:
    """Cover small, near-full-circle, and middle shifts across each three generated examples."""
    allowed = [k for k in range(2, n_objects) if (2 * k) % n_objects != 0]
    if coverage_mode == 0:
        return min(allowed)
    if coverage_mode == 1:
        return max(allowed)
    middle = [k for k in allowed if k not in (min(allowed), max(allowed))]
    return random.choice(middle or allowed)


def _generation_attempt(coverage_mode: int) -> np.ndarray | None:
    h = random.randint(20, 30)
    w = random.randint(20, 30)
    n_objects = random.randint(5, 7)
    shift = _choose_shift(n_objects, coverage_mode)

    colors = random.sample(list(OBJECT_COLORS), n_objects)
    areas = random.sample(range(2, 16), n_objects)
    random.shuffle(areas)

    canvas = np.zeros((h, w), dtype=int)
    records: list[dict] = []
    placement_specs = list(zip(colors, areas))
    random.shuffle(placement_specs)
    for color, area in placement_specs:
        try:
            mask = _random_polyomino(area)
        except RuntimeError:
            return None
        coords = _place_mask(canvas, mask, color)
        if coords is None:
            return None
        records.append({"color": int(color), "area": int(area), "coords": coords})

    if _place_counters(canvas, shift) is None:
        return None

    expected = _oracle_output(canvas, records, shift, order="area")
    return canvas if _semantic_validation(canvas, records, shift, expected) else None


def generate_input() -> np.ndarray:
    """Generate a varied, oracle-checked instance with anti-shortcut coverage."""
    global _generation_count
    coverage_mode = _generation_count % 3
    _generation_count += 1

    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _generation_attempt(coverage_mode)
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid gray-count area-cycle instance")
