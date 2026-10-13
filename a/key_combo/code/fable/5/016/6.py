from __future__ import annotations

from collections import Counter, deque

import numpy as np


_RNG = np.random.default_rng()
_MAX_GENERATION_ATTEMPTS = 400


def _background(grid: np.ndarray) -> int:
    """Return the unique most frequent color."""
    counts = Counter(map(int, grid.ravel()))
    largest = max(counts.values())
    winners = [color for color, count in counts.items() if count == largest]
    assert len(winners) == 1, "expected a unique modal background color"
    return int(winners[0])


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    """Return foreground components as (N, 2) arrays of canonical (row, col) coordinates."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        steps = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        steps = tuple(
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0)
        )

    result: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        queue = deque([(r, c)])
        cells: list[tuple[int, int]] = []
        while queue:
            rr, cc = queue.popleft()
            cells.append((rr, cc))
            for dr, dc in steps:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    """Inclusive bounding box in (r0, c0, r1, c1) order."""
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])


def _local_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    box = _bbox(coords)
    r0, c0, r1, c1 = box
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, box


def _transform(array: np.ndarray, quarter_turns: int, reflected: bool) -> np.ndarray:
    """Apply a D4 transform: CCW quarter-turns, then a left-right reflection."""
    result = np.rot90(array, int(quarter_turns) % 4)
    if reflected:
        result = np.fliplr(result)
    return result.copy()


def _dihedral(array: np.ndarray):
    for quarter_turns in range(4):
        for reflected in (False, True):
            yield quarter_turns, reflected, _transform(array, quarter_turns, reflected)


def _array_key(array: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, array.shape)), np.ascontiguousarray(array).tobytes()


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp the posed multicolor master into each monochrome imprint and punch out its named color.

    Input structure:
    - The unique modal color is the background. One separated 8-connected object is multicolored.
    - Several other separated objects are monochrome copies of the master's tight binary mask
      under rotations and reflections; their distinct colors all occur in the master.

    Rule:
    - For each monochrome imprint, uniquely match its tight mask to one of the eight dihedral
      transforms of the master mask, stamp the equally transformed full color pattern there,
      and replace every stamped pixel equal to the imprint's original color with background.
      The original multicolor master is left unchanged.

    Key insight:
    - Each imprint is a two-part instruction: its shape determines the pose of the stamp, while
      its monochrome color names the transformed master layer that must be punched away.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _background(g)

    objects = _components(g != bg, connectivity=8)
    assert len(objects) >= 2, "expected a master and at least one imprint"

    records: list[dict] = []
    for coords in objects:
        mask, box = _local_mask(coords)
        colors = set(map(int, g[coords[:, 0], coords[:, 1]]))
        records.append({"coords": coords, "mask": mask, "box": box, "colors": colors})

    masters = [record for record in records if len(record["colors"]) >= 2]
    assert len(masters) == 1, "expected exactly one multicolor master object"
    master = masters[0]
    imprints = [record for record in records if record is not master]
    assert all(len(record["colors"]) == 1 for record in imprints)

    mr0, mc0, mr1, mc1 = master["box"]
    master_mask = master["mask"]
    master_patch = np.full(master_mask.shape, bg, dtype=int)
    source = g[mr0 : mr1 + 1, mc0 : mc1 + 1]
    master_patch[master_mask] = source[master_mask]
    master_palette = set(map(int, master_patch[master_mask]))

    # A symmetric master would make the shape-to-pose instruction ambiguous.
    mask_variants = list(_dihedral(master_mask))
    assert len({_array_key(variant) for _, _, variant in mask_variants}) == 8

    cue_colors: list[int] = []
    out = g.copy()
    for imprint in imprints:
        cue_color = int(next(iter(imprint["colors"])))
        cue_colors.append(cue_color)
        assert cue_color in master_palette, "imprint color must name a master layer"

        matches = [
            (quarter_turns, reflected)
            for quarter_turns, reflected, variant in mask_variants
            if variant.shape == imprint["mask"].shape
            and np.array_equal(variant, imprint["mask"])
        ]
        assert len(matches) == 1, "imprint shape must determine one unique pose"

        quarter_turns, reflected = matches[0]
        posed_patch = _transform(master_patch, quarter_turns, reflected)
        posed_mask = posed_patch != bg
        assert np.array_equal(posed_mask, imprint["mask"])

        r0, c0, r1, c1 = imprint["box"]
        target = out[r0 : r1 + 1, c0 : c1 + 1]
        target[posed_mask] = posed_patch[posed_mask]
        target[posed_mask & (posed_patch == cue_color)] = bg

    assert len(set(cue_colors)) == len(cue_colors), "imprint colors must be distinct"
    return out


def _random_asymmetric_mask(rng: np.random.Generator, minimum_area: int) -> np.ndarray | None:
    """Grow an irregular connected polyomino with eight distinct D4 poses."""
    for _ in range(160):
        canvas_h = int(rng.integers(5, 7))
        canvas_w = int(rng.integers(5, 7))
        target_area = int(rng.integers(max(minimum_area, 13), min(21, canvas_h * canvas_w - 2) + 1))
        start = (int(rng.integers(canvas_h)), int(rng.integers(canvas_w)))
        cells = {start}
        frontier: set[tuple[int, int]] = set()

        while len(cells) < target_area:
            frontier.clear()
            for r, c in cells:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < canvas_h and 0 <= nc < canvas_w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            choices = tuple(frontier)
            cells.add(choices[int(rng.integers(len(choices)))])

        coords = np.asarray(tuple(cells), dtype=int)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        h, w = int(r1 - r0 + 1), int(c1 - c0 + 1)
        if min(h, w) < 4 or h == w:
            continue
        mask = np.zeros((h, w), dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        if len({_array_key(variant) for _, _, variant in _dihedral(mask)}) != 8:
            continue
        return mask
    return None


def _color_master(
    mask: np.ndarray, palette: list[int], rng: np.random.Generator
) -> np.ndarray:
    """Color every layer at least twice so every punch instruction is visibly consequential."""
    coords = np.argwhere(mask)
    assignments = palette * 2
    assignments.extend(
        int(palette[int(rng.integers(len(palette)))])
        for _ in range(len(coords) - len(assignments))
    )
    rng.shuffle(assignments)
    patch = np.zeros(mask.shape, dtype=int)
    patch[coords[:, 0], coords[:, 1]] = np.asarray(assignments, dtype=int)
    return patch


def _boxes_have_gap(
    candidate: tuple[int, int, int, int],
    existing: tuple[int, int, int, int],
    gap: int = 1,
) -> bool:
    """Whether two half-open boxes have at least `gap` blank rows or columns between them."""
    r0, c0, r1, c1 = candidate
    er0, ec0, er1, ec1 = existing
    return (
        r1 + gap <= er0
        or er1 + gap <= r0
        or c1 + gap <= ec0
        or ec1 + gap <= c0
    )


def _place_patches(
    shapes: list[tuple[int, int]], h: int, w: int, rng: np.random.Generator
) -> list[tuple[int, int]] | None:
    """Randomly place tight bboxes with a clear one-cell separator and outer margin."""
    positions: list[tuple[int, int] | None] = [None] * len(shapes)
    boxes: list[tuple[int, int, int, int]] = []
    order = list(map(int, rng.permutation(len(shapes))))
    for index in order:
        ph, pw = shapes[index]
        candidates = [
            (r, c)
            for r in range(1, h - ph)
            for c in range(1, w - pw)
        ]
        rng.shuffle(candidates)
        for r, c in candidates:
            box = (r, c, r + ph, c + pw)
            if all(_boxes_have_gap(box, old_box) for old_box in boxes):
                positions[index] = (r, c)
                boxes.append(box)
                break
        if positions[index] is None:
            return None
    return [position for position in positions if position is not None]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    cue_count = int(rng.integers(3, 6))
    bg = int(rng.integers(10))
    available = [color for color in range(10) if color != bg]
    rng.shuffle(available)
    cue_colors = list(map(int, available[:cue_count]))
    extra_color = int(available[cue_count])
    palette = cue_colors + [extra_color]

    master_mask = _random_asymmetric_mask(rng, minimum_area=2 * len(palette))
    if master_mask is None:
        return None
    raw_master = _color_master(master_mask, palette, rng)
    master_patch = np.full(master_mask.shape, bg, dtype=int)
    master_patch[master_mask] = raw_master[master_mask]

    # Every example visibly exercises an odd rotation and a reflection; remaining poses vary.
    required_rotation = (int(rng.choice((1, 3))), False)
    required_reflection = (int(rng.integers(4)), True)
    all_poses = [(k, reflected) for k in range(4) for reflected in (False, True)]
    poses = [required_rotation, required_reflection]
    remaining = [pose for pose in all_poses if pose not in poses]
    rng.shuffle(remaining)
    poses.extend(remaining[: cue_count - len(poses)])
    rng.shuffle(poses)

    # Independent semantic guards: both shape and color instructions must affect the answer.
    for cue_color in cue_colors:
        punched = []
        for k, reflected in all_poses:
            posed = _transform(master_patch, k, reflected)
            posed = posed.copy()
            posed[posed == cue_color] = bg
            punched.append(_array_key(posed))
        if len(set(punched)) != 8:
            return None

    imprint_masks = [_transform(master_mask, k, reflected) for k, reflected in poses]
    shapes = [master_patch.shape] + [mask.shape for mask in imprint_masks]
    low = min(27, 20 + cue_count)
    h = int(rng.integers(low, 31))
    w = int(rng.integers(low, 31))
    positions = _place_patches(shapes, h, w, rng)
    if positions is None:
        return None

    inp = np.full((h, w), bg, dtype=int)
    mr, mc = positions[0]
    master_view = inp[mr : mr + master_patch.shape[0], mc : mc + master_patch.shape[1]]
    master_view[master_mask] = master_patch[master_mask]

    output_keys: list[tuple[tuple[int, int], bytes]] = []
    for index, (cue_color, pose, imprint_mask) in enumerate(
        zip(cue_colors, poses, imprint_masks), start=1
    ):
        r, c = positions[index]
        input_view = inp[r : r + imprint_mask.shape[0], c : c + imprint_mask.shape[1]]
        input_view[imprint_mask] = cue_color

        k, reflected = pose
        posed = _transform(master_patch, k, reflected)
        punched = posed.copy()
        punched[punched == cue_color] = bg
        output_keys.append(_array_key(punched))

    # Build an independent expected output from the rendered input and recorded instructions.
    expected = inp.copy()
    for index, (cue_color, pose, imprint_mask) in enumerate(
        zip(cue_colors, poses, imprint_masks), start=1
    ):
        r, c = positions[index]
        k, reflected = pose
        posed = _transform(master_patch, k, reflected)
        target = expected[r : r + posed.shape[0], c : c + posed.shape[1]]
        target[imprint_mask] = posed[imprint_mask]
        target[imprint_mask & (posed == cue_color)] = bg

    if len(set(output_keys)) != len(output_keys):
        return None
    if _background(inp) != bg:
        return None
    objects = _components(inp != bg, connectivity=8)
    if len(objects) != cue_count + 1:
        return None
    multicolor_count = sum(
        len(set(map(int, inp[coords[:, 0], coords[:, 1]]))) >= 2 for coords in objects
    )
    if multicolor_count != 1:
        return None
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a varied, nondegenerate instance satisfying all solver assumptions."""
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid monochrome-imprint instance")
