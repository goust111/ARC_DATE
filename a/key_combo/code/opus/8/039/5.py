from __future__ import annotations

from itertools import combinations, product
import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5

GEN_MAX_ATTEMPTS = 400
GEN_SLOT_EXTRA_PAD = (0, 2)
GEN_ELBOW_WEIGHTS = ((2, 4), (3, 4), (4, 2))
GEN_VERTICAL_OPTIONS = {
    2: (1, 2),
    3: (0, 1),
    4: (0,),
}


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _strip_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    top = np.asarray(grid[:2], dtype=int)
    b = _bbox(top == RED)
    assert b is not None, "expected a red glyph strip in the top two rows"
    r0, c0, r1, c1 = b
    assert (r0, r1) == (0, 1), "glyph strip must span exactly the top two rows"
    return b


def _parse_glyphs(grid: np.ndarray) -> tuple[list[int | None], int]:
    """Return the decoded step list and the trimmed strip width."""
    _r0, c0, _r1, c1 = _strip_bbox(grid)
    strip = np.asarray(grid[:2, c0 : c1 + 1], dtype=int)
    width = int(strip.shape[1])
    steps: list[int | None] = []
    c = 0
    while c < width:
        if not strip[:, c].any():
            c += 1
            continue
        start = c
        while c < width and strip[:, c].any():
            c += 1
        glyph = strip[:, start:c]
        assert glyph.shape[1] in (1, 2), "glyphs must be 1 or 2 columns wide"
        if glyph.shape[1] == 1:
            assert glyph[0, 0] == RED and glyph[1, 0] == RED
            steps.append(None)
        else:
            top_cols = np.flatnonzero(glyph[0] == RED)
            bottom_cols = np.flatnonzero(glyph[1] == RED)
            assert len(top_cols) == 1 and len(bottom_cols) == 2
            assert set(map(int, bottom_cols)) == {0, 1}
            entry = int(top_cols[0])
            steps.append(1 if entry == 0 else -1)
    assert steps, "expected at least one glyph"
    return steps, width


def _render_uncropped_from_steps(steps: list[int | None], strip_width: int) -> np.ndarray:
    out_h = 1 + sum(2 if step is None else 1 for step in steps)
    out_w = strip_width // 2 + 1
    out = np.full((out_h, out_w), BLACK, dtype=int)
    r = 0
    c = out_w // 2
    out[r, c] = GREEN
    r += 1
    for step in steps:
        if step is None:
            out[r : r + 2, c] = RED
            r += 2
        else:
            next_c = c + step
            assert 0 <= next_c < out_w
            out[r, c] = RED
            out[r, next_c] = RED
            c = next_c
            r += 1
    assert r == out_h
    return out


def _decoded_template(grid: np.ndarray) -> np.ndarray:
    steps, strip_width = _parse_glyphs(grid)
    uncropped = _render_uncropped_from_steps(steps, strip_width)
    b = _bbox(uncropped != BLACK)
    assert b is not None
    return _crop_bbox(uncropped, b)


def _slot_regions(grid: np.ndarray) -> tuple[int, list[tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    full_grey_rows = [r for r in range(g.shape[0]) if np.all(g[r] == GREY)]
    assert len(full_grey_rows) == 1, "expected exactly one grey horizontal divider row"
    divider_r = int(full_grey_rows[0])

    slot_band = g[divider_r + 1 :]
    assert slot_band.size > 0, "expected a slot band below the divider"
    div_cols = [c for c in range(g.shape[1]) if np.all(slot_band[:, c] == GREY)]
    cuts = [-1, *div_cols, g.shape[1]]
    regions: list[tuple[int, int, int, int]] = []
    for left, right in zip(cuts, cuts[1:]):
        c0 = left + 1
        c1 = right - 1
        if c0 <= c1:
            regions.append((divider_r + 1, c0, g.shape[0] - 1, c1))

    assert regions, "expected at least one slot"
    widths = {c1 - c0 + 1 for _r0, c0, _r1, c1 in regions}
    heights = {r1 - r0 + 1 for r0, _c0, r1, _c1 in regions}
    assert len(widths) == 1 and len(heights) == 1, "slots must be equal sized"
    return divider_r, regions


def _place_patch(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _rotations(template: np.ndarray, count: int) -> list[np.ndarray]:
    rots = []
    cur = template.copy()
    for idx in range(count):
        if idx > 0:
            cur = np.rot90(cur, k=-1)
        rots.append(cur.copy())
    return rots


def main(grid: np.ndarray) -> np.ndarray:
    """Decode the top glyph strip into a traced template, then fill the lower slots with successive clockwise rotations.

    Input structure:
    - The top two rows contain a black-backed red glyph strip read left to right; each glyph is either a vertical 2x1 bar or a 2x2 elbow whose missing top pixel indicates a left or right turn.
    - A full grey horizontal divider separates the strip from a lower band of equal black slots split by grey vertical dividers.
    - The number of lower slots matches the number of elbow glyphs in the strip.

    Rule:
    - Decode the glyph strip into a black path canvas with a green start marker on the top row and red traced segments descending from it: vertical glyphs draw two red cells in the current column, while elbow glyphs draw a horizontal two-cell segment and shift the current column left or right.
    - Count the elbow glyphs, crop the decoded path image to its tight non-black bounding box, and use that crop as the template.
    - Keep the input grid size and dividers unchanged, place the template in the first slot, then place successive 90-degree clockwise rotations of that template into each later slot, aligned to each slot's top-left corner.

    Key insight:
    - The red strip is not copied directly: it must be decoded into a traced image, and the elbows simultaneously determine both the traced shape and how many rotated slot copies must be filled below.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    steps, _strip_width = _parse_glyphs(g)
    template = _decoded_template(g)
    elbow_count = sum(step is not None for step in steps)
    _divider_r, regions = _slot_regions(g)
    assert elbow_count == len(regions), "slot count must equal elbow count"

    for region, patch in zip(regions, _rotations(template, elbow_count)):
        r0, c0, r1, c1 = region
        slot_h = r1 - r0 + 1
        slot_w = c1 - c0 + 1
        ph, pw = patch.shape
        assert ph <= slot_h and pw <= slot_w, "rotated template must fit inside each slot"
        _place_patch(out, patch, r0, c0)
    return out


def _encode_strip(steps: list[int | None]) -> np.ndarray:
    width = sum(1 if step is None else 2 for step in steps) + len(steps) - 1
    strip = np.full((2, width), BLACK, dtype=int)
    c = 0
    for idx, step in enumerate(steps):
        if step is None:
            strip[:, c] = RED
            c += 1
        else:
            strip[1, c : c + 2] = RED
            strip[0, c if step == 1 else c + 1] = RED
            c += 2
        if idx + 1 < len(steps):
            c += 1
    return strip


def _steps_stay_in_bounds(steps: list[int | None], strip_width: int) -> bool:
    out_w = strip_width // 2 + 1
    c = out_w // 2
    for step in steps:
        if step is None:
            continue
        c += step
        if not (0 <= c < out_w):
            return False
    return True


def _left_crop_margin(uncropped: np.ndarray) -> int:
    b = _bbox(uncropped != BLACK)
    assert b is not None
    return int(b[1])


def _is_interesting_steps(steps: list[int | None]) -> bool:
    strip_width = _encode_strip(steps).shape[1]
    if not _steps_stay_in_bounds(steps, strip_width):
        return False

    uncropped = _render_uncropped_from_steps(steps, strip_width)
    if _left_crop_margin(uncropped) <= 0:
        return False

    template = _crop_bbox(uncropped, _bbox(uncropped != BLACK))
    elbow_count = sum(step is not None for step in steps)
    rots = _rotations(template, elbow_count)
    for i in range(len(rots)):
        for j in range(i + 1, len(rots)):
            if np.array_equal(rots[i], rots[j]):
                return False

    directions = [step for step in steps if step is not None]
    if elbow_count >= 2 and len(set(directions)) == 1 and all(step is not None for step in steps):
        return False
    return True


def _candidate_steps(elbow_count: int, vertical_count: int) -> list[list[int | None]]:
    total = elbow_count + vertical_count
    candidates: list[list[int | None]] = []
    for vertical_pos in combinations(range(total), vertical_count):
        vertical_set = set(vertical_pos)
        for dirs in product((-1, 1), repeat=elbow_count):
            steps: list[int | None] = []
            di = 0
            for idx in range(total):
                if idx in vertical_set:
                    steps.append(None)
                else:
                    steps.append(int(dirs[di]))
                    di += 1
            if _is_interesting_steps(steps):
                candidates.append(steps)
    return candidates


def _build_input_from_steps(steps: list[int | None]) -> np.ndarray | None:
    strip = _encode_strip(steps)
    template = _decoded_template(np.vstack([strip, np.full((1, strip.shape[1]), GREY, dtype=int)]))
    elbow_count = sum(step is not None for step in steps)
    base_slot = max(template.shape)
    max_slot = min(
        base_slot + GEN_SLOT_EXTRA_PAD[1],
        (30 - (elbow_count - 1)) // elbow_count,
        27,
    )
    min_slot = base_slot + GEN_SLOT_EXTRA_PAD[0]
    if min_slot > max_slot:
        return None
    slot_size = random.randint(min_slot, max_slot)

    full_w = elbow_count * slot_size + (elbow_count - 1)
    full_h = 3 + slot_size
    assert full_w <= 30 and full_h <= 30

    grid = np.full((full_h, full_w), BLACK, dtype=int)
    divider_r = 2
    grid[divider_r, :] = GREY
    for slot_idx in range(1, elbow_count):
        divider_c = slot_idx * slot_size + (slot_idx - 1)
        grid[divider_r + 1 :, divider_c] = GREY

    offset = random.randint(0, full_w - strip.shape[1])
    grid[:2, offset : offset + strip.shape[1]] = strip
    return grid


def _expected_output_for_steps(grid: np.ndarray, steps: list[int | None]) -> np.ndarray:
    out = grid.copy()
    template = _decoded_template(grid)
    elbow_count = sum(step is not None for step in steps)
    _divider_r, regions = _slot_regions(grid)
    for region, patch in zip(regions, _rotations(template, elbow_count)):
        r0, c0, _r1, _c1 = region
        _place_patch(out, patch, r0, c0)
    return out


def _output_without_crop(grid: np.ndarray, steps: list[int | None]) -> np.ndarray:
    out = grid.copy()
    strip_width = _encode_strip(steps).shape[1]
    uncropped = _render_uncropped_from_steps(steps, strip_width)
    _divider_r, regions = _slot_regions(grid)
    for region, patch in zip(regions, _rotations(uncropped, len(regions))):
        r0, c0, _r1, _c1 = region
        _place_patch(out, patch, r0, c0)
    return out


def _output_without_rotation(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    template = _decoded_template(grid)
    _divider_r, regions = _slot_regions(grid)
    for region in regions:
        r0, c0, _r1, _c1 = region
        _place_patch(out, template, r0, c0)
    return out


def _validate_generated(grid: np.ndarray, steps: list[int | None]) -> bool:
    try:
        parsed_steps, _strip_width = _parse_glyphs(grid)
        if parsed_steps != steps:
            return False

        out = main(grid)
        expected = _expected_output_for_steps(grid, steps)
        if not np.array_equal(out, expected):
            return False

        elbow_count = sum(step is not None for step in steps)
        if elbow_count != len(_slot_regions(grid)[1]):
            return False

        template = _decoded_template(grid)
        rots = _rotations(template, elbow_count)
        for i in range(len(rots)):
            for j in range(i + 1, len(rots)):
                if np.array_equal(rots[i], rots[j]):
                    return False

        if np.array_equal(out, _output_without_crop(grid, steps)):
            return False
        if np.array_equal(out, _output_without_rotation(grid)):
            return False
        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    elbow_count = random.choices(
        [pair[0] for pair in GEN_ELBOW_WEIGHTS],
        weights=[pair[1] for pair in GEN_ELBOW_WEIGHTS],
        k=1,
    )[0]
    vertical_count = random.choice(GEN_VERTICAL_OPTIONS[elbow_count])
    candidates = _candidate_steps(elbow_count, vertical_count)
    if not candidates:
        return None

    random.shuffle(candidates)
    for steps in candidates:
        grid = _build_input_from_steps(steps)
        if grid is None:
            continue
        if _validate_generated(grid, steps):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid glyph-rotation ARC input")
