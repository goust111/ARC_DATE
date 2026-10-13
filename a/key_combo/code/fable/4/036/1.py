from __future__ import annotations

import numpy as np


BACKGROUND = 0
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as arrays of (row, column) coordinates."""
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    result: list[np.ndarray] = []
    height, width = mask.shape
    for start_r, start_c in np.argwhere(mask):
        start_r, start_c = int(start_r), int(start_c)
        if seen[start_r, start_c]:
            continue
        seen[start_r, start_c] = True
        stack = [(start_r, start_c)]
        cells: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    """Inclusive bbox, following arc_utils' (r0, c0, r1, c1) convention."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(coords: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    crop = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    crop[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return crop


def _pair_record(grid: np.ndarray, color: int) -> dict:
    """Identify the uniquely aligned complete/damaged pair of one color."""
    objects = _components(grid == color)
    assert len(objects) == 2, f"color {color} must form exactly two objects"

    entries = []
    for coords in objects:
        box = _bbox(coords)
        mask = _crop_mask(coords, box)
        assert mask.shape[0] == mask.shape[1], "pair bboxes must be square"
        entries.append((box, mask))
    assert entries[0][1].shape == entries[1][1].shape, "paired bboxes must have equal size"

    candidates = []
    for complete_i, damaged_i in ((0, 1), (1, 0)):
        complete = entries[complete_i][1]
        damaged = entries[damaged_i][1]
        missing = complete & ~damaged
        if missing.any() and np.all(~damaged | complete):
            candidates.append((complete_i, damaged_i, missing))
    assert len(candidates) == 1, "the complete-to-damaged subset relation must be unique"

    complete_i, damaged_i, missing = candidates[0]
    k = len(_components(missing))
    assert k >= 1, "the damaged copy must omit at least one connected region"
    return {
        "color": int(color),
        "complete_box": entries[complete_i][0],
        "damaged_box": entries[damaged_i][0],
        "complete": entries[complete_i][1],
        "damaged": entries[damaged_i][1],
        "missing": missing,
        "k": k,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Move each complete motif onto its damaged copy, rotating it by the number of missing components.

    Input structure:
    - A black background contains several pairs of monochrome motifs; every pair has
      its own color and consists of two separated 4-connected objects.
    - The two objects in a pair have equal square bboxes. After their bbox top-left
      corners are aligned, one mask is a strict subset of the other.

    Rule:
    - For each color, identify the complete mask and its damaged subset, then count
      the 4-connected components k in complete AND NOT damaged.
    - Clear both original bboxes and draw the complete mask, rotated clockwise by
      k quarter-turns, at the damaged object's bbox.

    Key insight:
    - The damage is an instruction rather than noise: only its aligned 4-connected
      component count determines the rotation, while the complete object supplies
      the motif that is moved.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    colors = set(map(int, np.unique(g))) - {BACKGROUND}
    assert colors, "expected at least one foreground pair"

    # Analyze every pair before painting, so object recognition always uses the input.
    records = [_pair_record(g, color) for color in colors]
    out = g.copy()
    for rec in records:
        for r0, c0, r1, c1 in (rec["complete_box"], rec["damaged_box"]):
            out[r0 : r1 + 1, c0 : c1 + 1] = BACKGROUND

    for rec in records:
        r0, c0, r1, c1 = rec["damaged_box"]
        # np.rot90 is counterclockwise for positive k; a negative k is clockwise.
        rotated = np.rot90(rec["complete"], k=-(rec["k"] % 4))
        destination = out[r0 : r1 + 1, c0 : c1 + 1]
        destination[rotated] = rec["color"]
    return out


# ------------------------------ generator ------------------------------

GRID_SIDE_MIN = 28
GRID_SIDE_MAX = 30
PAIR_COUNT_CHOICES = (4, 5)
MOTIF_SIZE_CHOICES = (5, 6)
MISSING_AREA = 4
PREEXISTING_HOLE_COUNT = 2
SLOT_PITCH = 7
GENERATION_ATTEMPTS = 100
MASK_ATTEMPTS = 7000


def _count_components(mask: np.ndarray) -> int:
    return len(_components(mask))


def _rotationally_asymmetric(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _sample_pair_masks(
    rng: np.random.Generator, size: int, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Sample masks with aligned damage semantics and several anti-shortcut controls."""
    interior = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
    for _ in range(MASK_ATTEMPTS):
        pre_indices = rng.choice(len(interior), size=PREEXISTING_HOLE_COUNT, replace=False)
        pre_holes = {interior[int(i)] for i in pre_indices}
        available = [cell for cell in interior if cell not in pre_holes]
        damage_indices = rng.choice(len(available), size=MISSING_AREA, replace=False)
        damage_cells = {available[int(i)] for i in damage_indices}

        complete = np.ones((size, size), dtype=bool)
        for cell in pre_holes:
            complete[cell] = False
        missing = np.zeros((size, size), dtype=bool)
        for cell in damage_cells:
            missing[cell] = True
        damaged = complete & ~missing

        if _count_components(missing) != k:
            continue
        if not _rotationally_asymmetric(complete):
            continue
        if _count_components(complete) != 1 or _count_components(damaged) != 1:
            continue

        # All damaged copies have six absent cells forming one region. Thus neither
        # raw hole area nor the damaged copy's hole count can replace aligned subtraction.
        if _count_components(~damaged) != 1:
            continue
        return complete, damaged, missing
    return None


def _render_rule(meta: list[dict], shape: tuple[int, int], step_key: str) -> np.ndarray:
    """Independent renderer used only for semantic generator checks."""
    out = np.zeros(shape, dtype=int)
    for item in meta:
        r0, c0 = item["damaged_pos"]
        steps = int(item[step_key])
        mask = np.rot90(item["complete"], k=-(steps % 4))
        patch = out[r0 : r0 + item["size"], c0 : c0 + item["size"]]
        patch[mask] = item["color"]
    return out


def _semantic_validation(inp: np.ndarray, expected: np.ndarray, meta: list[dict]) -> bool:
    """Check rule meaning, uniqueness, non-degeneracy, and explicit shortcut counterexamples."""
    if inp.ndim != 2 or expected.shape != inp.shape or max(inp.shape) > 30:
        return False
    if {item["k"] for item in meta} != {1, 2, 3, 4}:
        return False
    if {int(item["missing"].sum()) for item in meta} != {MISSING_AREA}:
        return False

    try:
        records = {rec["color"]: rec for rec in (_pair_record(inp, item["color"]) for item in meta)}
        actual = main(inp)
    except (AssertionError, ValueError, IndexError):
        return False
    if not np.array_equal(actual, expected) or np.array_equal(actual, inp):
        return False

    for item in meta:
        rec = records.get(item["color"])
        if rec is None or rec["k"] != item["k"]:
            return False
        if rec["complete_box"][:2] != item["complete_pos"]:
            return False
        if rec["damaged_box"][:2] != item["damaged_pos"]:
            return False
        if not np.array_equal(rec["missing"], item["missing"]):
            return False
        if not _rotationally_asymmetric(rec["complete"]):
            return False
        if _count_components(~rec["damaged"]) != 1:
            return False

    # These deliberately wrong rules must fail on every generated example.
    shortcut_meta = []
    for item in meta:
        copy = dict(item)
        copy["identity_steps"] = 0
        copy["area_steps"] = int(item["missing"].sum())
        copy["damaged_hole_steps"] = _count_components(~item["damaged"])
        shortcut_meta.append(copy)
    for wrong_key in ("identity_steps", "area_steps", "damaged_hole_steps"):
        if np.array_equal(_render_rule(shortcut_meta, inp.shape, wrong_key), expected):
            return False
    return True


def _generation_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(GRID_SIDE_MIN, GRID_SIDE_MAX + 1))
    width = int(rng.integers(GRID_SIDE_MIN, GRID_SIDE_MAX + 1))
    pair_count = int(rng.choice(PAIR_COUNT_CHOICES))

    # Four rows by four columns provide sixteen visibly separated square slots.
    base_r = int(rng.integers(0, height - (3 * SLOT_PITCH + max(MOTIF_SIZE_CHOICES)) + 1))
    base_c = int(rng.integers(0, width - (3 * SLOT_PITCH + max(MOTIF_SIZE_CHOICES)) + 1))
    slots = [(base_r + r * SLOT_PITCH, base_c + c * SLOT_PITCH) for r in range(4) for c in range(4)]
    chosen = rng.choice(len(slots), size=2 * pair_count, replace=False)
    positions = [slots[int(i)] for i in chosen]

    colors = [int(c) for c in rng.choice(np.arange(1, 10), size=pair_count, replace=False)]
    ks = [1, 2, 3, 4]
    if pair_count == 5:
        ks.append(int(rng.integers(1, 5)))
    rng.shuffle(ks)

    inp = np.zeros((height, width), dtype=int)
    expected = np.zeros_like(inp)
    meta: list[dict] = []
    for pair_index, (color, k) in enumerate(zip(colors, ks)):
        size = int(rng.choice(MOTIF_SIZE_CHOICES))
        masks = _sample_pair_masks(rng, size, k)
        if masks is None:
            return None
        complete, damaged, missing = masks
        first_pos = positions[2 * pair_index]
        second_pos = positions[2 * pair_index + 1]
        if rng.random() < 0.5:
            complete_pos, damaged_pos = first_pos, second_pos
        else:
            complete_pos, damaged_pos = second_pos, first_pos

        cr, cc = complete_pos
        dr, dc = damaged_pos
        inp[cr : cr + size, cc : cc + size][complete] = color
        inp[dr : dr + size, dc : dc + size][damaged] = color
        rotated = np.rot90(complete, k=-(k % 4))
        expected[dr : dr + size, dc : dc + size][rotated] = color
        meta.append(
            {
                "color": color,
                "k": k,
                "size": size,
                "complete": complete,
                "damaged": damaged,
                "missing": missing,
                "complete_pos": complete_pos,
                "damaged_pos": damaged_pos,
            }
        )

    if not _semantic_validation(inp, expected, meta):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        result = _generation_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATION_ATTEMPTS} attempts")
