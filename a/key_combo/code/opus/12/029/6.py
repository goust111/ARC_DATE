import random
from collections import Counter

import numpy as np


BG_COLOR = 0
SOURCE_COLOR = 1
FILL_COLOR = 8
GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _rot180(mask: np.ndarray) -> np.ndarray:
    return np.rot90(mask, 2)


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _component_masks(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            if visited[r0, c0] or int(grid[r0, c0]) != color:
                continue
            stack = [(r0, c0)]
            visited[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            rs = [r for r, _ in coords]
            cs = [c for _, c in coords]
            br0, br1 = min(rs), max(rs)
            bc0, bc1 = min(cs), max(cs)
            mask = np.zeros((br1 - br0 + 1, bc1 - bc0 + 1), dtype=bool)
            for r, c in coords:
                mask[r - br0, c - bc0] = True
            components.append(
                {
                    "bbox": (br0, bc0, br1, bc1),
                    "top_left": (br0, bc0),
                    "coords": tuple(coords),
                    "mask": mask,
                }
            )
    return components


def _constant_step(values: list[int]) -> int:
    assert len(values) >= 2, "expected at least two lattice coordinates"
    diffs = [b - a for a, b in zip(values, values[1:])]
    assert all(d > 0 for d in diffs), "lattice coordinates must be distinct and sorted"
    assert len(set(diffs)) == 1, "expected uniform lattice spacing"
    return diffs[0]


def _paste_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    assert patch.shape == (h, w), "sprite placement must fit in bounds"
    patch[mask] = int(color)


def _infer_template(sprites: list[dict], rows: list[int], cols: list[int]) -> np.ndarray:
    masks_by_key: dict[tuple[tuple[int, ...], ...], np.ndarray] = {}
    counts: Counter[tuple[tuple[int, ...], ...]] = Counter()
    for sprite in sprites:
        key = _mask_key(sprite["mask"])
        masks_by_key[key] = sprite["mask"]
        counts[key] += 1

    assert len(counts) in {1, 2}, "expected one or two sprite orientations"
    ranked = counts.most_common()
    template = masks_by_key[ranked[0][0]]
    rotated = _rot180(template)
    assert not np.array_equal(template, rotated), "template must differ from its 180-degree rotation"

    row_index = {r: i for i, r in enumerate(rows)}
    col_index = {c: j for j, c in enumerate(cols)}

    def matches(candidate: np.ndarray) -> bool:
        candidate_rot = _rot180(candidate)
        for sprite in sprites:
            i = row_index[sprite["top_left"][0]]
            j = col_index[sprite["top_left"][1]]
            expected = candidate if (i + j) % 2 == 0 else candidate_rot
            if not np.array_equal(sprite["mask"], expected):
                return False
        return True

    if matches(template):
        return template
    if matches(rotated):
        return rotated
    raise AssertionError("could not align sprite orientations with position parity")


def _analyze(grid: np.ndarray, color: int = SOURCE_COLOR) -> dict:
    sprites = _component_masks(grid, color)
    assert sprites, "expected at least one foreground sprite"

    rows = sorted({int(sprite["top_left"][0]) for sprite in sprites})
    cols = sorted({int(sprite["top_left"][1]) for sprite in sprites})
    assert len(rows) >= 3 and len(cols) >= 3, "expected at least a 3x3 lattice"

    row_pitch = _constant_step(rows)
    col_pitch = _constant_step(cols)
    assert row_pitch == col_pitch, "row and column pitch must match"

    occupied = {(int(sprite["top_left"][0]), int(sprite["top_left"][1])) for sprite in sprites}
    assert any(all((r, c) in occupied for c in cols) for r in rows), "expected a complete occupied row"
    assert any(all((r, c) in occupied for r in rows) for c in cols), "expected a complete occupied column"

    template = _infer_template(sprites, rows, cols)
    rotated = _rot180(template)
    th, tw = template.shape

    row_index = {r: i for i, r in enumerate(rows)}
    col_index = {c: j for j, c in enumerate(cols)}
    for sprite in sprites:
        assert sprite["mask"].shape == template.shape, "all sprites must share one bbox size"
        i = row_index[sprite["top_left"][0]]
        j = col_index[sprite["top_left"][1]]
        expected = template if (i + j) % 2 == 0 else rotated
        assert np.array_equal(sprite["mask"], expected), "occupied slots must follow parity-driven rotation"

    return {
        "sprites": sprites,
        "rows": rows,
        "cols": cols,
        "pitch": row_pitch,
        "occupied": occupied,
        "template": template,
        "rotated": rotated,
        "shape": (th, tw),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the missing lattice sprites, rotating each filled copy by slot parity.

    Input structure:
    - A black background contains several blue 4-connected sprite instances arranged on a rectangular lattice.
    - Every sprite is the same asymmetric 3-5-cell shape up to a 180-degree rotation, and all instances share one bbox size.
    - The occupied lattice slots include at least one complete row and one complete column, while some row-column intersections are empty.

    Rule:
    - Find all blue sprite components and normalize them by their bbox top-left corners.
    - Infer the lattice row starts and column starts from the component top-left coordinates.
    - Determine the base sprite template and enforce the rule that slots with even row-plus-column index use the template, while odd slots use its 180-degree rotation.
    - Fill every empty lattice slot with a teal copy of the orientation required by that slot parity, leaving existing blue sprites unchanged.

    Key insight:
    - The missing cells are not filled with one fixed mask: the sprite orientation alternates like a checkerboard based on lattice position parity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    info = _analyze(g, SOURCE_COLOR)
    rows = info["rows"]
    cols = info["cols"]
    occupied = info["occupied"]
    template = info["template"]
    rotated = info["rotated"]

    out = g.copy()
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            if (r, c) in occupied:
                continue
            mask = template if (i + j) % 2 == 0 else rotated
            patch = out[r : r + mask.shape[0], c : c + mask.shape[1]]
            assert np.all(patch == BG_COLOR), "missing lattice slot must be empty"
            _paste_mask(out, mask, r, c, FILL_COLOR)
    return out


def _random_sprite_mask() -> np.ndarray:
    for _ in range(200):
        area = random.randint(3, 5)
        seed_h = random.randint(2, 4)
        seed_w = random.randint(2, 4)
        cells = {(random.randrange(seed_h), random.randrange(seed_w))}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < seed_h and 0 <= nc < seed_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue

        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in cells:
            mask[r - r0, c - c0] = True

        if min(mask.shape) < 2:
            continue
        if np.array_equal(mask, _rot180(mask)):
            continue
        return mask
    raise RuntimeError("failed to sample a valid asymmetric sprite")


def _feasible_pitches(n_rows: int, n_cols: int, sprite_shape: tuple[int, int]) -> list[int]:
    h, w = sprite_shape
    min_pitch = max(h, w) + 1
    feasible = []
    for pitch in range(min_pitch, 9):
        max_h = 3 + (n_rows - 1) * pitch + h + 3
        max_w = 3 + (n_cols - 1) * pitch + w + 3
        if max_h <= 30 and max_w <= 30:
            feasible.append(pitch)
    return feasible


def _choose_missing(
    n_rows: int, n_cols: int, full_row: int, full_col: int
) -> set[tuple[int, int]] | None:
    mandatory = {(full_row, j) for j in range(n_cols)} | {(i, full_col) for i in range(n_rows)}
    extras = [(i, j) for i in range(n_rows) for j in range(n_cols) if (i, j) not in mandatory]
    if len(extras) < 2:
        return None

    for _ in range(200):
        missing_count = random.randint(2, min(4, len(extras)))
        missing = set(random.sample(extras, missing_count))
        parities = {(i + j) % 2 for i, j in missing}
        if parities != {0, 1}:
            continue

        occupied = {(i, j) for i in range(n_rows) for j in range(n_cols)} - missing
        even_count = sum((i + j) % 2 == 0 for i, j in occupied)
        odd_count = len(occupied) - even_count
        if even_count <= odd_count:
            continue
        if even_count == 0 or odd_count == 0:
            continue
        if not any(all((i, j) in occupied for j in range(n_cols)) for i in range(n_rows)):
            continue
        if not any(all((i, j) in occupied for i in range(n_rows)) for j in range(n_cols)):
            continue
        return missing
    return None


def _build_input() -> np.ndarray | None:
    mask = _random_sprite_mask()
    rotated = _rot180(mask)

    n_rows = random.randint(3, 4)
    n_cols = random.randint(3, 4)
    pitches = _feasible_pitches(n_rows, n_cols, mask.shape)
    if not pitches:
        return None
    pitch = random.choice(pitches)

    full_row = random.randrange(n_rows)
    full_col = random.randrange(n_cols)
    missing = _choose_missing(n_rows, n_cols, full_row, full_col)
    if missing is None:
        return None

    top = random.randint(1, 3)
    left = random.randint(1, 3)
    bottom = random.randint(1, 3)
    right = random.randint(1, 3)

    h, w = mask.shape
    height = top + (n_rows - 1) * pitch + h + bottom
    width = left + (n_cols - 1) * pitch + w + right
    if height > 30 or width > 30:
        return None

    row_starts = [top + i * pitch for i in range(n_rows)]
    col_starts = [left + j * pitch for j in range(n_cols)]

    grid = np.zeros((height, width), dtype=int)
    for i, r in enumerate(row_starts):
        for j, c in enumerate(col_starts):
            if (i, j) in missing:
                continue
            sprite = mask if (i + j) % 2 == 0 else rotated
            _paste_mask(grid, sprite, r, c, SOURCE_COLOR)

    return grid


def _semantic_check(grid: np.ndarray) -> None:
    info = _analyze(grid, SOURCE_COLOR)
    rows = info["rows"]
    cols = info["cols"]
    template = info["template"]
    rotated = info["rotated"]

    row_index = {r: i for i, r in enumerate(rows)}
    col_index = {c: j for j, c in enumerate(cols)}
    occupied_slots = {(row_index[r], col_index[c]) for r, c in info["occupied"]}
    all_slots = {(i, j) for i in range(len(rows)) for j in range(len(cols))}
    missing = all_slots - occupied_slots

    assert 2 <= len(missing) <= 4, "generator must leave 2-4 holes"
    assert {(i + j) % 2 for i, j in missing} == {0, 1}, "missing slots must cover both parities"

    even_occupied = sum((i + j) % 2 == 0 for i, j in occupied_slots)
    odd_occupied = len(occupied_slots) - even_occupied
    assert even_occupied > odd_occupied, "0-degree template must be the most frequent orientation"

    input_keys = Counter(_mask_key(sprite["mask"]) for sprite in info["sprites"])
    assert len(input_keys) == 2, "occupied slots must expose both orientations"
    counts = sorted(input_keys.values(), reverse=True)
    assert counts[0] > counts[1], "template frequency must strictly dominate"

    out = main(grid)
    assert np.array_equal(out[grid == SOURCE_COLOR], grid[grid == SOURCE_COLOR]), "existing blue sprites must be preserved"

    blue_out = _component_masks(out, SOURCE_COLOR)
    teal_out = _component_masks(out, FILL_COLOR)
    assert len(blue_out) == len(info["sprites"]), "output must not add or remove blue sprites"
    assert {(d["top_left"]) for d in blue_out} == {(d["top_left"]) for d in info["sprites"]}, "blue sprite positions must stay fixed"
    assert len(teal_out) == len(missing), "one teal sprite should be added per missing slot"

    teal_by_pos = {d["top_left"]: d["mask"] for d in teal_out}
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            if (i, j) not in missing:
                continue
            expected = template if (i + j) % 2 == 0 else rotated
            assert (r, c) in teal_by_pos, "every missing slot must be filled"
            assert np.array_equal(teal_by_pos[(r, c)], expected), "filled orientation must follow slot parity"


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_input()
        if grid is None:
            continue
        try:
            _semantic_check(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
