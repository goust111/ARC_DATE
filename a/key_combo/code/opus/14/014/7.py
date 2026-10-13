from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
COMPACT_SIZE = 10
SPARSE_SIZE = COMPACT_SIZE * 2 - 1
MAX_GENERATION_ATTEMPTS = 800

NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _neighbors(r: int, c: int, h: int, w: int, connectivity: int):
    deltas = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    for dr, dc in deltas:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(
    grid: np.ndarray,
    *,
    bg: int = BG,
    connectivity: int = 8,
    monochromatic: bool = False,
) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            queue = [(r, c)]
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, h, w, connectivity):
                    if seen[nr, nc] or int(grid[nr, nc]) == bg:
                        continue
                    if monochromatic and int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))
            coords.sort()
            colors = {int(grid[rr, cc]) for rr, cc in coords}
            comps.append(
                {
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                    "colors": colors,
                    "color": color if monochromatic else None,
                }
            )
    return comps


def _crop_component(grid: np.ndarray, comp: dict[str, object]) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _relative_cells(grid: np.ndarray, comp: dict[str, object]) -> list[tuple[int, int, int]]:
    r0, c0, _r1, _c1 = comp["bbox"]
    return [(r - r0, c - c0, int(grid[r, c])) for r, c in comp["coords"]]


def _stamp_preserving(canvas: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    ph, pw = patch.shape
    for r in range(ph):
        for c in range(pw):
            color = int(patch[r, c])
            if color == BG:
                continue
            rr, cc = top + r, left + c
            if int(canvas[rr, cc]) == BG:
                canvas[rr, cc] = color


def _downsample(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert g.shape[0] % 2 == 1 and g.shape[1] % 2 == 1, "expected odd sparse-grid dimensions"
    assert np.all(g[1::2, :] == BG), "odd rows must be black separators"
    assert np.all(g[:, 1::2] == BG), "odd columns must be black separators"
    return g[::2, ::2].copy()


def _find_sprite_component(compact: np.ndarray) -> dict[str, object]:
    comps = _find_components(compact, bg=BG, connectivity=8, monochromatic=False)
    multicolor = [comp for comp in comps if len(comp["colors"]) > 1]
    assert multicolor, "expected one multicolor source sprite"
    max_area = max(int(comp["area"]) for comp in multicolor)
    best = [comp for comp in multicolor if int(comp["area"]) == max_area]
    assert len(best) == 1, "largest multicolor source sprite must be unique"
    return best[0]


def _fragment_components(compact: np.ndarray, sprite_comp: dict[str, object]) -> list[dict[str, object]]:
    masked = compact.copy()
    for r, c in sprite_comp["coords"]:
        masked[r, c] = BG
    return _find_components(masked, bg=BG, connectivity=8, monochromatic=True)


def _translation_scores(
    sprite_patch: np.ndarray,
    group_coords: list[tuple[int, int]],
    group_color: int,
    canvas_shape: tuple[int, int],
) -> list[tuple[int, int, int]]:
    ph, pw = sprite_patch.shape
    h, w = canvas_shape
    scores: list[tuple[int, int, int]] = []
    for top in range(h - ph + 1):
        for left in range(w - pw + 1):
            score = 0
            for r, c in group_coords:
                sr = r - top
                sc = c - left
                if 0 <= sr < ph and 0 <= sc < pw and int(sprite_patch[sr, sc]) == group_color:
                    score += 1
            scores.append((score, top, left))
    return scores


def main(grid: np.ndarray) -> np.ndarray:
    """
    Downsample the sparse lattice, locate the revealed sprite fragments, and complete each hidden sprite copy.

    Input structure:
    - The input is an odd-sized sparse grid whose odd rows and odd columns are all black separators.
    - The even-row/even-column intersections form a compact logical payload with black background.
    - That payload contains one largest multicolor 8-connected sprite plus one or two smaller monochromatic fragment groups.

    Rule:
    - Remove all separator rows and columns by keeping only even-index intersections.
    - In the compact grid, identify the unique largest multicolor 8-connected component as the source sprite.
    - For each monochromatic fragment group, test every in-bounds translation of the full sprite and score exact same-color overlaps with the group's pixels.
    - Choose the unique translation with maximal overlap and stamp the full sprite there, preserving any existing non-background pixels.

    Key insight:
    - The separator lattice hides the fragment geometry; only after downsampling do the fragment pixels collapse into a pattern whose maximal same-color overlap reveals where the full sprite must be stamped.
    """
    compact = _downsample(grid)
    sprite_comp = _find_sprite_component(compact)
    sprite_patch = _crop_component(compact, sprite_comp)
    groups = _fragment_components(compact, sprite_comp)
    assert 1 <= len(groups) <= 2, "expected one or two fragment groups"

    out = compact.copy()
    groups.sort(key=lambda comp: comp["coords"][0])
    for group in groups:
        group_coords = list(group["coords"])
        group_color = int(group["color"])
        scores = _translation_scores(sprite_patch, group_coords, group_color, compact.shape)
        best_score = max(score for score, _top, _left in scores)
        best_positions = [(top, left) for score, top, left in scores if score == best_score]
        assert best_score == len(group_coords), "fragment must fully match some translated same-color sprite subset"
        assert len(best_positions) == 1, "best-overlap translation must be unique"
        top, left = best_positions[0]
        _stamp_preserving(out, sprite_patch, top, left)

    return out


def _normalize_coords(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, c0 = min(rows), min(cols)
    return sorted((r - r0, c - c0) for r, c in coords)


def _grow_shape(height: int, width: int, cells: int) -> list[tuple[int, int]] | None:
    shape = {(random.randrange(height), random.randrange(width))}
    while len(shape) < cells:
        frontier = set()
        for r, c in shape:
            for nr, nc in _neighbors(r, c, height, width, 4):
                if (nr, nc) not in shape:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        shape.add(random.choice(tuple(frontier)))

    norm = _normalize_coords(shape)
    bbox_h = max(r for r, _ in norm) + 1
    bbox_w = max(c for _, c in norm) + 1
    if bbox_h != height or bbox_w != width:
        return None
    return norm


def _connected_subset(
    available: set[tuple[int, int]],
    size: int,
    *,
    connectivity: int = 8,
) -> set[tuple[int, int]] | None:
    if len(available) < size:
        return None
    seed = random.choice(tuple(available))
    subset = {seed}
    while len(subset) < size:
        frontier = set()
        for r, c in subset:
            for dr, dc in (NEIGHBORS4 if connectivity == 4 else NEIGHBORS8):
                nxt = (r + dr, c + dc)
                if nxt in available and nxt not in subset:
                    frontier.add(nxt)
        if not frontier:
            return None
        subset.add(random.choice(tuple(frontier)))
    return subset


def _build_sprite() -> dict[str, object] | None:
    color_count = 2 if random.random() < 0.45 else 3
    height = random.randint(3, 4)
    width = random.randint(3, 4)
    area = random.randint(6, min(10, height * width))
    shape = _grow_shape(height, width, area)
    if shape is None:
        return None

    shape_set = set(shape)
    if color_count == 2:
        minority_size = random.randint(2, min(4, area - 2))
        minority = _connected_subset(shape_set, minority_size, connectivity=8)
        if minority is None:
            return None
        groups = [minority]
    else:
        size_options = [
            (a, b)
            for a in range(2, min(4, area - 4) + 1)
            for b in range(2, min(4, area - a - 2) + 1)
        ]
        if not size_options:
            return None
        size_a, size_b = random.choice(size_options)
        group_a = _connected_subset(shape_set, size_a, connectivity=8)
        if group_a is None:
            return None
        group_b = _connected_subset(shape_set - group_a, size_b, connectivity=8)
        if group_b is None:
            return None
        groups = [group_a, group_b]

    colors = random.sample(range(1, 10), color_count)
    dominant = colors[0]
    patch = np.zeros((height, width), dtype=int)
    coord_color = {coord: dominant for coord in shape}
    fragment_specs: list[dict[str, object]] = []
    for color, subset in zip(colors[1:], groups):
        for coord in subset:
            coord_color[coord] = color
        fragment_specs.append({"color": color, "rel_cells": sorted(subset)})

    for (r, c), color in coord_color.items():
        patch[r, c] = color

    return {
        "patch": patch,
        "shape": shape,
        "fragment_specs": fragment_specs,
    }


def _boxes_far(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
    gap: int = 1,
) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return bool(
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _choose_positions(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    all_positions = [(r, c) for r in range(COMPACT_SIZE - height + 1) for c in range(COMPACT_SIZE - width + 1)]
    for _ in range(250):
        shuffled = all_positions[:]
        random.shuffle(shuffled)
        chosen: list[tuple[int, int]] = []
        chosen_boxes: list[tuple[int, int, int, int]] = []
        for top, left in shuffled:
            box = (top, left, top + height - 1, left + width - 1)
            if all(_boxes_far(box, other, gap=1) for other in chosen_boxes):
                chosen.append((top, left))
                chosen_boxes.append(box)
                if len(chosen) == count:
                    return chosen
    return None


def _place_patch(canvas: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    ph, pw = patch.shape
    for r in range(ph):
        for c in range(pw):
            color = int(patch[r, c])
            if color != BG:
                canvas[top + r, left + c] = color


def _add_fragment(canvas: np.ndarray, fragment_spec: dict[str, object], top: int, left: int) -> list[tuple[int, int]]:
    color = int(fragment_spec["color"])
    abs_cells = []
    for r, c in fragment_spec["rel_cells"]:
        canvas[top + r, left + c] = color
        abs_cells.append((top + r, left + c))
    abs_cells.sort()
    return abs_cells


def _compact_to_sparse(compact: np.ndarray) -> np.ndarray:
    sparse = np.zeros((SPARSE_SIZE, SPARSE_SIZE), dtype=int)
    sparse[::2, ::2] = compact
    return sparse


def _make_expected(sprite_patch: np.ndarray, placements: list[tuple[int, int]]) -> np.ndarray:
    expected = np.zeros((COMPACT_SIZE, COMPACT_SIZE), dtype=int)
    for top, left in placements:
        _stamp_preserving(expected, sprite_patch, top, left)
    return expected


def _validate_instance(
    sparse_input: np.ndarray,
    compact_input: np.ndarray,
    compact_output: np.ndarray,
    sprite_patch: np.ndarray,
    source_pos: tuple[int, int],
    fragment_records: list[dict[str, object]],
) -> bool:
    if sparse_input.shape != (SPARSE_SIZE, SPARSE_SIZE):
        return False
    if not np.array_equal(_downsample(sparse_input), compact_input):
        return False
    if not np.array_equal(main(sparse_input), compact_output):
        return False

    sprite_comp = _find_sprite_component(compact_input)
    if not np.array_equal(_crop_component(compact_input, sprite_comp), sprite_patch):
        return False
    if tuple(sprite_comp["bbox"][:2]) != source_pos:
        return False

    groups = _fragment_components(compact_input, sprite_comp)
    if len(groups) != len(fragment_records):
        return False

    groups_by_signature = {}
    for group in groups:
        signature = (int(group["color"]), tuple(group["coords"]))
        groups_by_signature[signature] = group

    for record in fragment_records:
        signature = (int(record["color"]), tuple(record["abs_cells"]))
        if signature not in groups_by_signature:
            return False
        scores = _translation_scores(
            sprite_patch,
            list(record["abs_cells"]),
            int(record["color"]),
            (COMPACT_SIZE, COMPACT_SIZE),
        )
        best_score = max(score for score, _top, _left in scores)
        best_positions = [(top, left) for score, top, left in scores if score == best_score]
        if best_score != len(record["abs_cells"]):
            return False
        if best_positions != [tuple(record["stamp_pos"])]:
            return False
        alt_positive = sum(1 for score, top, left in scores if 0 < score < best_score)
        if alt_positive == 0:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    sprite = _build_sprite()
    if sprite is None:
        return None

    sprite_patch = sprite["patch"]
    fragment_specs = list(sprite["fragment_specs"])
    ph, pw = sprite_patch.shape
    group_count = 1 if random.random() < 0.5 else 2

    if group_count == 1:
        chosen_specs = [random.choice(fragment_specs)]
    else:
        if len(fragment_specs) == 1 or random.random() < 0.35:
            chosen_specs = [random.choice(fragment_specs), random.choice(fragment_specs)]
        else:
            chosen_specs = random.sample(fragment_specs, 2)

    placements = _choose_positions(ph, pw, group_count + 1)
    if placements is None:
        return None

    source_pos = placements[0]
    stamp_positions = placements[1:]
    compact_input = np.zeros((COMPACT_SIZE, COMPACT_SIZE), dtype=int)
    _place_patch(compact_input, sprite_patch, *source_pos)

    fragment_records = []
    for spec, stamp_pos in zip(chosen_specs, stamp_positions):
        abs_cells = _add_fragment(compact_input, spec, *stamp_pos)
        fragment_records.append(
            {
                "color": int(spec["color"]),
                "abs_cells": tuple(abs_cells),
                "stamp_pos": tuple(stamp_pos),
            }
        )

    compact_output = _make_expected(sprite_patch, placements)
    sparse_input = _compact_to_sparse(compact_input)

    if not _validate_instance(
        sparse_input,
        compact_input,
        compact_output,
        sprite_patch,
        source_pos,
        fragment_records,
    ):
        return None
    return sparse_input


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sparse sprite-fragment puzzle")
