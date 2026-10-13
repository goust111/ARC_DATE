from __future__ import annotations

from collections import Counter, deque
from itertools import combinations

import numpy as np


Coord = tuple[int, int]


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[set[Coord]]:
    """Return 4-connected components using the common ARC (row, column) convention."""
    h, w = mask.shape
    unseen = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result: list[set[Coord]] = []
    while unseen:
        start = unseen.pop()
        comp = {start}
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) in unseen:
                    unseen.remove((nr, nc))
                    comp.add((nr, nc))
                    queue.append((nr, nc))
        result.append(comp)
    return result


def _reflect(coords: set[Coord], orientation: str, axis2: int) -> set[Coord]:
    """Reflect across 2*c=axis2 or 2*r=axis2, allowing half-cell axes."""
    if orientation == "v":
        return {(r, axis2 - c) for r, c in coords}
    return {(axis2 - r, c) for r, c in coords}


def _strictly_one_side(coords: set[Coord], orientation: str, axis2: int) -> bool:
    values = [2 * (c if orientation == "v" else r) for r, c in coords]
    return max(values) < axis2 or min(values) > axis2


def _candidate_matches(
    g: np.ndarray,
    bg: int,
    sprites: list[dict],
    all_sprite_cells: set[Coord],
    color: int,
    marks: set[Coord],
) -> list[dict]:
    h, w = g.shape
    candidates: dict[tuple, dict] = {}
    for sprite_index, sprite in enumerate(sprites):
        source = sprite["coords"]
        for mr, mc in marks:
            for sr, sc in source:
                proposals: list[tuple[str, int]] = []
                if mr == sr:
                    proposals.append(("v", mc + sc))
                if mc == sc:
                    proposals.append(("h", mr + sr))
                for orientation, axis2 in proposals:
                    if not _strictly_one_side(source, orientation, axis2):
                        continue
                    target = _reflect(source, orientation, axis2)
                    if target & all_sprite_cells:
                        continue
                    if any(not (0 <= r < h and 0 <= c < w) for r, c in target):
                        continue
                    if any(int(g[r, c]) not in (bg, color) for r, c in target):
                        continue
                    inliers = target & marks
                    if len(inliers) < 2:
                        continue
                    key = (sprite_index, orientation, axis2, tuple(sorted(target)))
                    candidates[key] = {
                        "sprite": sprite_index,
                        "orientation": orientation,
                        "axis2": axis2,
                        "target": target,
                        "inliers": inliers,
                        "score": len(inliers),
                    }
    return list(candidates.values())


def _resolve(grid: np.ndarray) -> tuple[np.ndarray, dict[int, dict], dict[int, set[Coord]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size, "expected a nonempty 2D grid"
    bg = _background(g)

    by_color: dict[int, list[set[Coord]]] = {}
    for color in sorted(set(map(int, g.ravel())) - {bg}):
        by_color[color] = _components(g == color)

    sprites: list[dict] = []
    residue: dict[int, set[Coord]] = {}
    for color, comps in by_color.items():
        large = [comp for comp in comps if len(comp) >= 4]
        singles = [next(iter(comp)) for comp in comps if len(comp) == 1]
        if large:
            assert not singles, "sprite colors must not also encode residue pixels"
            sprites.extend({"color": color, "coords": comp} for comp in large)
        else:
            assert len(singles) == len(comps), "residue pixels must be isolated"
            residue[color] = set(singles)

    assert len(sprites) >= 2 and residue, "expected multiple sprites and residue colors"
    all_sprite_cells = set().union(*(sprite["coords"] for sprite in sprites))
    options: dict[int, list[dict]] = {}
    for color, marks in residue.items():
        choices = _candidate_matches(g, bg, sprites, all_sprite_cells, color, marks)
        assert choices, "each residue color must support a reflection"
        options[color] = choices

    colors = sorted(options)
    best_score = -1
    best: list[dict[int, dict]] = []

    def search(i: int, used_sprites: set[int], score: int, chosen: dict[int, dict]) -> None:
        nonlocal best_score, best
        if i == len(colors):
            if score > best_score:
                best_score, best = score, [chosen.copy()]
            elif score == best_score:
                best.append(chosen.copy())
            return
        color = colors[i]
        for candidate in options[color]:
            sprite_index = int(candidate["sprite"])
            if sprite_index in used_sprites:
                continue
            chosen[color] = candidate
            search(i + 1, used_sprites | {sprite_index}, score + int(candidate["score"]), chosen)
            chosen.pop(color)

    search(0, set(), 0, {})
    assert len(best) == 1, "the residue evidence must determine one global assignment"
    matches = best[0]
    assert len(matches) == len(sprites), "every sprite must receive exactly one reflected copy"

    out = g.copy()
    for color, marks in residue.items():
        for r, c in marks:
            out[r, c] = bg
        for r, c in matches[color]["target"]:
            assert out[r, c] == bg, "reflected copies must not overlap"
            out[r, c] = color
    return out, matches, residue


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer each sprite's hidden mirror axis from fragment pixels, complete its reflection, and erase false fragments.

    Input structure:
    - The dominant color is background; two or three separated monochrome 4-connected sprites each contain at least four pixels.
    - Every other foreground color occurs only as isolated pixels and differs from all sprite colors.
    - For each sprite, two to four pixels of a uniquely colored reflected copy are visible, while extra isolated pixels of that same color may be distractors.

    Rule:
    - For every residue color, test horizontal and vertical axes in the full-grid row/column coordinate system.
    - Select the unique global one-to-one assignment of residue colors to sprites that maximizes how many same-colored pixels lie in complete reflected footprints.
    - Erase all isolated residues, then paint every winning reflected footprint with its residue color while preserving the original sprites.

    Key insight:
    - A single dot is ambiguous, but the largest jointly consistent subset fixes both the sprite and its invisible axis; same-colored dots outside that reflection are false traces.
    """
    return _resolve(grid)[0]


# ------------------------------ generator ------------------------------


_BASE_SHAPES: tuple[set[Coord], ...] = (
    {(0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3)},
    {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (3, 1), (3, 2)},
    {(0, 0), (1, 0), (1, 1), (1, 2), (2, 2), (3, 2), (3, 3), (2, 3)},
    {(0, 1), (1, 1), (1, 2), (2, 0), (2, 1), (3, 0), (3, 1)},
)


def _normalize(coords: set[Coord]) -> set[Coord]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return {(r - r0, c - c0) for r, c in coords}


def _random_shape(rng: np.random.Generator) -> set[Coord]:
    coords = set(_BASE_SHAPES[int(rng.integers(len(_BASE_SHAPES)))])
    for _ in range(int(rng.integers(4))):
        coords = {(c, -r) for r, c in coords}
    if bool(rng.integers(2)):
        coords = {(r, -c) for r, c in coords}
    return _normalize(coords)


def _separated_subsets(coords: set[Coord], count: int) -> list[tuple[Coord, ...]]:
    return [
        combo
        for combo in combinations(sorted(coords), count)
        if all(abs(a[0] - b[0]) + abs(a[1] - b[1]) > 1 for a, b in combinations(combo, 2))
    ]


def _expanded(coords: set[Coord], h: int, w: int, radius: int = 1) -> set[Coord]:
    return {
        (r + dr, c + dc)
        for r, c in coords
        for dr in range(-radius, radius + 1)
        for dc in range(-radius, radius + 1)
        if 0 <= r + dr < h and 0 <= c + dc < w
    }


def _make_pair(
    shape: set[Coord], orientation: str, gap: int, swap: bool
) -> tuple[set[Coord], set[Coord], int]:
    if orientation == "v":
        far = max(c for _, c in shape)
        axis2 = 2 * far + gap + 1
    else:
        far = max(r for r, _ in shape)
        axis2 = 2 * far + gap + 1
    reflected = _reflect(shape, orientation, axis2)
    source, target = (reflected, shape) if swap else (shape, reflected)
    union = source | target
    r0, c0 = min(r for r, _ in union), min(c for _, c in union)
    source = {(r - r0, c - c0) for r, c in source}
    target = {(r - r0, c - c0) for r, c in target}
    if orientation == "v":
        axis2 -= 2 * c0
    else:
        axis2 -= 2 * r0
    return source, target, axis2


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(22, 29))
    w = int(rng.integers(22, 29))
    object_count = int(rng.integers(2, 4))
    palette = list(map(int, rng.choice(np.arange(1, 10), size=2 * object_count, replace=False)))
    sprite_colors = palette[:object_count]
    fragment_colors = palette[object_count:]
    grid = np.zeros((h, w), dtype=int)
    reserved: set[Coord] = set()
    records: list[dict] = []

    orientations = ["v", "h"] + (["v" if bool(rng.integers(2)) else "h"] if object_count == 3 else [])
    rng.shuffle(orientations)
    for index in range(object_count):
        shape = _random_shape(rng)
        orientation = orientations[index]
        source0, target0, axis20 = _make_pair(
            shape, orientation, int(rng.integers(2, 6)), bool(rng.integers(2))
        )
        unit = source0 | target0
        uh = max(r for r, _ in unit) + 1
        uw = max(c for _, c in unit) + 1
        positions = [(r, c) for r in range(1, h - uh) for c in range(1, w - uw)]
        rng.shuffle(positions)
        placed = None
        for r0, c0 in positions:
            translated_unit = {(r + r0, c + c0) for r, c in unit}
            if translated_unit & _expanded(reserved, h, w, 2):
                continue
            source = {(r + r0, c + c0) for r, c in source0}
            target = {(r + r0, c + c0) for r, c in target0}
            axis2 = axis20 + (2 * c0 if orientation == "v" else 2 * r0)
            placed = (source, target, axis2)
            break
        if placed is None:
            return None
        source, target, axis2 = placed
        for r, c in source:
            grid[r, c] = sprite_colors[index]
        reserved |= source | target
        records.append(
            {
                "source": source,
                "target": target,
                "axis2": axis2,
                "orientation": orientation,
                "sprite_color": sprite_colors[index],
                "fragment_color": fragment_colors[index],
            }
        )

    visible: set[Coord] = set().union(*(record["source"] for record in records))
    for index, record in enumerate(records):
        requested = 2 + index  # Three-object cases cover the upper boundary of four traces.
        subsets = _separated_subsets(record["target"], requested)
        if not subsets:
            return None
        true_marks = set(subsets[int(rng.integers(len(subsets)))])
        for r, c in true_marks:
            grid[r, c] = record["fragment_color"]
        record["true_marks"] = true_marks
        visible |= true_marks

    # One same-colored false trace per object makes the filtering operation necessary.
    free = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1) if (r, c) not in reserved]
    rng.shuffle(free)
    for record in records:
        decoy = None
        for pos in free:
            if pos in reserved or pos in _expanded(visible, h, w, 1):
                continue
            decoy = pos
            break
        if decoy is None:
            return None
        grid[decoy] = record["fragment_color"]
        record["decoy"] = decoy
        visible.add(decoy)
        free.remove(decoy)

    expected = grid.copy()
    for record in records:
        expected[record["decoy"]] = 0
        for r, c in record["target"]:
            expected[r, c] = record["fragment_color"]

    try:
        actual, matches, residue = _resolve(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if {match["orientation"] for match in matches.values()} != {"h", "v"}:
        return None
    if any(record["decoy"] in matches[record["fragment_color"]]["target"] for record in records):
        return None
    if any(len(residue[record["fragment_color"]] - matches[record["fragment_color"]]["target"]) != 1 for record in records):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, oracle-checked instance with both axis directions and false traces."""
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a uniquely solvable hidden-axis reflection puzzle")
