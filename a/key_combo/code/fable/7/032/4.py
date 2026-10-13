from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
OCCLUDER = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the inclusive (r0, c0, r1, c1) bbox used by arc_utils."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(mask: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _scale(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(mask, np.ones((factor, factor), dtype=bool))


def _is_connected(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if len(points) == 0:
        return False
    start = tuple(map(int, points[0]))
    seen = {start}
    todo = deque([start])
    h, w = mask.shape
    while todo:
        r, c = todo.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                todo.append((rr, cc))
    return len(seen) == int(mask.sum())


def _color_objects(grid: np.ndarray) -> list[dict]:
    """Treat each non-black, non-grey color as one monochrome object."""
    objects = []
    for color in sorted(set(map(int, grid.ravel())) - {BACKGROUND, OCCLUDER}):
        mask = grid == color
        box = _bbox(mask)
        assert box is not None
        local = _crop(mask, box)
        objects.append(
            {
                "color": color,
                "bbox": box,
                "mask": local,
                "bbox_area": int(local.size),
                "area": int(local.sum()),
            }
        )
    return objects


def _matching_candidates(grid: np.ndarray) -> tuple[dict, list[tuple[dict, int, np.ndarray]]]:
    objects = _color_objects(grid)
    assert len(objects) >= 3, "expected one large object and at least two candidates"

    largest_area = max(obj["bbox_area"] for obj in objects)
    large_hits = [obj for obj in objects if obj["bbox_area"] == largest_area]
    assert len(large_hits) == 1, "large object must have the unique largest bbox"
    large = large_hits[0]
    lr0, lc0, lr1, lc1 = large["bbox"]
    large_h, large_w = lr1 - lr0 + 1, lc1 - lc0 + 1

    local_grid = grid[lr0 : lr1 + 1, lc0 : lc1 + 1]
    visible = local_grid != OCCLUDER
    assert np.any(~visible), "occluder must overlap the large object's bbox"
    assert np.all(np.isin(local_grid, (BACKGROUND, OCCLUDER, large["color"])))
    observed = local_grid == large["color"]

    matches = []
    for candidate in objects:
        if candidate is large:
            continue
        ch, cw = candidate["mask"].shape
        if large_h % ch or large_w % cw:
            continue
        kr, kc = large_h // ch, large_w // cw
        if kr != kc or kr < 2:
            continue
        enlarged = _scale(candidate["mask"], kr)
        if np.array_equal(enlarged[visible], observed[visible]):
            matches.append((candidate, kr, enlarged))
    return large, matches


def main(grid: np.ndarray) -> np.ndarray:
    """Restore a grey-occluded large object from its uniquely matching integer-scaled twin.

    Input structure:
    - A black grid contains one large monochrome object partly covered by one
      solid grey rectangle, plus several complete, separated monochrome candidates.
    - Exactly one candidate has bbox dimensions that scale by the same integer
      factor in both axes to the large object's bbox.

    Rule:
    - Enlarge every candidate by nearest-neighbour cell replication and compare
      the whole large-object bbox outside the grey rectangle, including visible
      foreground and visible black cells.  The unique full match is the twin.
    - Erase the complete grey rectangle to black, then use the enlarged twin to
      repaint the hidden large-object cells in the large object's color.  Leave
      every candidate and all other cells unchanged.

    Key insight:
    - Grey cells are ignored only during cross-scale matching; after the unique
      twin is selected, its scaled mask determines both foreground and background
      behind the occluder.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30

    grey_box = _bbox(g == OCCLUDER)
    assert grey_box is not None, "expected a grey occluder"
    gr0, gc0, gr1, gc1 = grey_box
    assert np.all(g[gr0 : gr1 + 1, gc0 : gc1 + 1] == OCCLUDER), "occluder must be solid"

    large, matches = _matching_candidates(g)
    assert len(matches) == 1, "visible evidence must identify exactly one scaled twin"
    _, _, enlarged = matches[0]

    out = g.copy()
    out[out == OCCLUDER] = BACKGROUND
    lr0, lc0, lr1, lc1 = large["bbox"]
    region = out[lr0 : lr1 + 1, lc0 : lc1 + 1]
    region[enlarged] = int(large["color"])
    region[~enlarged] = BACKGROUND
    return out


def _random_polyomino(h: int, w: int) -> np.ndarray:
    """Grow a connected, non-solid shape touching every side of its bbox."""
    target = random.randint(max(h + w - 1, 6), h * w - 2)
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    while int(mask.sum()) < target or not (
        mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()
    ):
        frontier = []
        for r, c in np.argwhere(mask):
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                    frontier.append((rr, cc))
        if not frontier or int(mask.sum()) >= h * w - 1:
            break
        mask[random.choice(frontier)] = True
    return mask


def _distractors(base: np.ndarray, count: int, hidden: np.ndarray) -> list[np.ndarray] | None:
    """Add different visible cells, making near-matches that the occluder cannot hide."""
    choices = []
    h, w = base.shape
    for r, c in np.argwhere(~base):
        r, c = int(r), int(c)
        adjacent = any(
            0 <= r + dr < h and 0 <= c + dc < w and base[r + dr, c + dc]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
        if adjacent and not bool(hidden[r, c]):
            choices.append((r, c))
    random.shuffle(choices)
    if len(choices) < count:
        return None
    result = []
    for r, c in choices[:count]:
        altered = base.copy()
        altered[r, c] = True
        assert _is_connected(altered)
        result.append(altered)
    return result


def _free_location(canvas: np.ndarray, patch: np.ndarray, padding: int = 1) -> tuple[int, int] | None:
    """Small generator counterpart of arc_utils.random_free_location."""
    h, w = patch.shape
    locations = []
    for r in range(1, canvas.shape[0] - h):
        for c in range(1, canvas.shape[1] - w):
            r0, r1 = max(0, r - padding), min(canvas.shape[0], r + h + padding)
            c0, c1 = max(0, c - padding), min(canvas.shape[1], c + w + padding)
            if not np.any(canvas[r0:r1, c0:c1] != BACKGROUND):
                locations.append((r, c))
    return random.choice(locations) if locations else None


def _semantic_valid(
    inp: np.ndarray,
    expected: np.ndarray,
    base: np.ndarray,
    enlarged: np.ndarray,
    large_box: tuple[int, int, int, int],
    occ_box: tuple[int, int, int, int],
    candidate_masks: list[np.ndarray],
) -> bool:
    """Check puzzle meaning independently of merely calling main(generate_input())."""
    lr0, lc0, lr1, lc1 = large_box
    or0, oc0, or1, oc1 = occ_box
    occ_local = np.zeros_like(enlarged)
    occ_local[or0 - lr0 : or1 - lr0 + 1, oc0 - lc0 : oc1 - lc0 + 1] = True
    visible = ~occ_local

    # The curtain must hide both kinds of content, so both restore and erase matter.
    if not np.any(enlarged & occ_local) or not np.any((~enlarged) & occ_local):
        return False
    if float((enlarged & occ_local).sum()) / float(enlarged.sum()) > 0.45:
        return False
    if not (np.any(enlarged & visible) and np.any((~enlarged) & visible)):
        return False

    scaled = [_scale(mask, enlarged.shape[0] // mask.shape[0]) for mask in candidate_masks]
    matches = [np.array_equal(mask[visible], enlarged[visible]) for mask in scaled]
    if matches.count(True) != 1 or not matches[0]:
        return False
    # Every distractor shares most of the silhouette but has visible contrary evidence.
    for mask in scaled[1:]:
        agreement = float((mask[visible] == enlarged[visible]).mean())
        if not (0.72 <= agreement < 1.0) or not np.any((mask != enlarged) & visible):
            return False

    if not _is_connected(base) or base.all() or not np.array_equal(inp[inp != OCCLUDER], expected[inp != OCCLUDER]):
        # Outside grey, output must preserve all input content exactly.
        return False
    if np.array_equal(inp, expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(3, 5), random.randint(3, 5)
    factor = random.choice((2, 3))
    base = _random_polyomino(h, w)
    if not (_is_connected(base) and not base.all() and (~base).sum() >= 3):
        return None
    enlarged = _scale(base, factor)
    big_h, big_w = enlarged.shape

    grid_h = random.randint(max(22, big_h + 5), 30)
    grid_w = random.randint(max(22, big_w + 5), 30)
    layout = random.choice(("left", "top"))
    if layout == "left":
        split = big_w + random.randint(3, 5)
        if split + w + 2 > grid_w:
            return None
        lr0 = random.randint(1, grid_h - big_h - 1)
        lc0 = random.randint(1, split - big_w - 1)
        candidate_view = (slice(0, grid_h), slice(split, grid_w))
    else:
        split = big_h + random.randint(3, 5)
        if split + h + 2 > grid_h:
            return None
        lr0 = random.randint(1, split - big_h - 1)
        lc0 = random.randint(1, grid_w - big_w - 1)
        candidate_view = (slice(split, grid_h), slice(0, grid_w))
    lr1, lc1 = lr0 + big_h - 1, lc0 + big_w - 1

    # Choose an interior rectangle that crosses scaled foreground and background.
    occ_choice = None
    for _ in range(100):
        oh = random.randint(2, min(6, big_h - 2))
        ow = random.randint(2, min(6, big_w - 2))
        rr = random.randint(1, big_h - oh - 1)
        cc = random.randint(1, big_w - ow - 1)
        hidden = enlarged[rr : rr + oh, cc : cc + ow]
        if hidden.any() and (~hidden).any():
            occ_choice = (rr, cc, oh, ow)
            break
    if occ_choice is None:
        return None
    rr, cc, oh, ow = occ_choice
    occ_local_base = np.zeros(base.shape, dtype=bool)
    for br in range(h):
        for bc in range(w):
            block = np.zeros(enlarged.shape, dtype=bool)
            block[br * factor : (br + 1) * factor, bc * factor : (bc + 1) * factor] = True
            if np.all(block <= np.pad(np.ones((oh, ow), dtype=bool), ((rr, big_h-rr-oh), (cc, big_w-cc-ow)))):
                occ_local_base[br, bc] = True

    count = random.randint(2, 4)
    distractors = _distractors(base, count - 1, occ_local_base)
    if distractors is None:
        return None
    candidate_masks = [base] + distractors

    colors = random.sample(PALETTE, count + 1)
    large_color, candidate_colors = colors[0], colors[1:]
    inp = np.zeros((grid_h, grid_w), dtype=int)
    expected = inp.copy()
    big_in = inp[lr0 : lr1 + 1, lc0 : lc1 + 1]
    big_out = expected[lr0 : lr1 + 1, lc0 : lc1 + 1]
    big_in[enlarged] = large_color
    big_out[enlarged] = large_color

    # Place complete candidates with a blank moat so colors never merge semantically.
    view = inp[candidate_view]
    out_view = expected[candidate_view]
    placement_order = list(zip(candidate_masks, candidate_colors))
    random.shuffle(placement_order)
    for mask, color in placement_order:
        patch = np.where(mask, color, BACKGROUND)
        pos = _free_location(view, patch, padding=1)
        if pos is None:
            return None
        pr, pc = pos
        view[pr : pr + h, pc : pc + w][mask] = color
        out_view[pr : pr + h, pc : pc + w][mask] = color

    or0, oc0 = lr0 + rr, lc0 + cc
    or1, oc1 = or0 + oh - 1, oc0 + ow - 1
    inp[or0 : or1 + 1, oc0 : oc1 + 1] = OCCLUDER

    if not _semantic_valid(
        inp, expected, base, enlarged, (lr0, lc0, lr1, lc1),
        (or0, oc0, or1, oc1), candidate_masks,
    ):
        return None
    try:
        solved = main(inp)
    except (AssertionError, ValueError):
        return None
    return inp if np.array_equal(solved, expected) else None


def generate_input() -> np.ndarray:
    """Generate a diverse, uniquely solvable occluded cross-scale matching grid."""
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {MAX_GENERATION_ATTEMPTS} attempts")
