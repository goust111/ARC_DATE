from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

ANCHOR_RELS = {(0, 0), (0, 1)}
BODY_COLORS = (BLUE, GREEN, GREY, PINK, ORANGE, TEAL, MAROON)


def _rot_rel(cell: tuple[int, int], turns: int) -> tuple[int, int]:
    """Rotate a relative (r, c) coordinate clockwise around (0, 0)."""
    r, c = map(int, cell)
    turns %= 4
    if turns == 0:
        return r, c
    if turns == 1:
        return c, -r
    if turns == 2:
        return -r, -c
    return -c, r


def _neighbors(cell: tuple[int, int], connectivity: int = 4):
    r, c = cell
    if connectivity == 4:
        yield r - 1, c
        yield r + 1, c
        yield r, c - 1
        yield r, c + 1
        return

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                yield r + dr, c + dc


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    H, W = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[np.ndarray] = []
    for r0, c0 in np.argwhere(mask):
        r0 = int(r0)
        c0 = int(c0)
        if seen[r0, c0]:
            continue
        cells: list[tuple[int, int]] = []
        queue: deque[tuple[int, int]] = deque([(r0, c0)])
        seen[r0, c0] = True
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for rr, cc in _neighbors((r, c), connectivity):
                if 0 <= rr < H and 0 <= cc < W and bool(mask[rr, cc]) and not seen[rr, cc]:
                    seen[rr, cc] = True
                    queue.append((rr, cc))
        out.append(np.array(cells, dtype=int))
    return out


def _is_connected(cells: set[tuple[int, int]], connectivity: int = 8) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        cell = stack.pop()
        for nb in _neighbors(cell, connectivity):
            if nb in cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells)


def _direction_to_rotation(delta: tuple[int, int]) -> int:
    mapping = {
        (0, 1): 0,
        (1, 0): 1,
        (0, -1): 2,
        (-1, 0): 3,
    }
    assert delta in mapping, "yellow anchor must be 4-adjacent to the red anchor"
    return mapping[delta]


def _fragment_from_component(g: np.ndarray, coords: np.ndarray) -> dict:
    red_cells = [(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) == RED]
    yellow_cells = [(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) == YELLOW]
    assert len(red_cells) == 1, "each fragment must contain exactly one red anchor pixel"
    assert len(yellow_cells) == 1, "each fragment must contain exactly one yellow anchor pixel"

    red = red_cells[0]
    yellow = yellow_cells[0]
    delta = (yellow[0] - red[0], yellow[1] - red[1])
    rotation = _direction_to_rotation(delta)

    normalized: dict[tuple[int, int], int] = {}
    inverse = (-rotation) % 4
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        rel = (r - red[0], c - red[1])
        norm_rel = _rot_rel(rel, inverse)
        color = int(g[r, c])
        if norm_rel in normalized:
            assert normalized[norm_rel] == color, "one normalized cell cannot carry two colors"
        normalized[norm_rel] = color

    assert normalized.get((0, 0)) == RED
    assert normalized.get((0, 1)) == YELLOW
    return {"red": red, "rotation": rotation, "normalized": normalized}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore every rotated damaged copy of one hidden sprite from the aligned union of its fragments.

    Input structure:
    - A black background contains several separated multicolor sprite fragments.
    - Every fragment is an incomplete copy of the same sprite, rotated by 0, 90,
      180, or 270 degrees.
    - Each fragment keeps exactly one red pixel and one adjacent yellow pixel;
      the red-to-yellow direction is the fragment's rotation anchor.

    Rule:
    - Use the red-yellow direction to rotate every fragment back to the canonical
      orientation where yellow is immediately right of red.
    - Align normalized fragments by their red pixel and take the union of all
      visible non-black cells to recover the complete sprite.
    - Rotate that recovered sprite back to each fragment's original orientation
      and stamp it at the fragment's red-pixel position, preserving all colors.

    Key insight:
    - No complete sprite is visible in the input; only rotation-normalizing the
      fragments around the red anchor makes their complementary missing cells
      line up into the full multicolor sprite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components = _components(g != BLACK, connectivity=8)
    assert len(components) >= 2, "expected multiple sprite fragments"
    fragments = [_fragment_from_component(g, coords) for coords in components]

    complete: dict[tuple[int, int], int] = {}
    for fragment in fragments:
        for rel, color in fragment["normalized"].items():
            if rel in complete:
                assert complete[rel] == color, "aligned fragments must agree on color"
            else:
                complete[rel] = color

    assert complete.get((0, 0)) == RED and complete.get((0, 1)) == YELLOW

    H, W = g.shape
    out = g.copy()
    for fragment in fragments:
        rr, cc = fragment["red"]
        rotation = int(fragment["rotation"])
        for rel, color in complete.items():
            dr, dc = _rot_rel(rel, rotation)
            r = rr + dr
            c = cc + dc
            assert 0 <= r < H and 0 <= c < W, "completed sprite must stay in bounds"
            existing = int(out[r, c])
            assert existing in (BLACK, int(color)), "completed fragments must not conflict"
            out[r, c] = int(color)

    return out


# --------------- generator ---------------

GEN_RETRY_LIMIT = 400
GEN_SPRITE_H_RANGE = (4, 7)
GEN_SPRITE_W_RANGE = (4, 7)
GEN_SPRITE_AREA_RANGE = (12, 20)
GEN_FRAGMENT_COUNT_RANGE = (3, 5)
GEN_GRID_SIZE_RANGE = (20, 30)
GEN_PLACEMENT_BORDER = 1
GEN_PADDING = 1
GEN_MASK_RETRY_LIMIT = 160
GEN_PLACEMENT_RETRY_LIMIT = 120


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _build_sprite() -> dict[tuple[int, int], int] | None:
    h = random.randint(*GEN_SPRITE_H_RANGE)
    w = random.randint(*GEN_SPRITE_W_RANGE)
    ar = random.randint(1, h - 2)
    ac = random.randint(1, w - 2)

    target_hi = min(GEN_SPRITE_AREA_RANGE[1], h * w - 2)
    target_lo = min(GEN_SPRITE_AREA_RANGE[0], target_hi)
    target = random.randint(target_lo, target_hi)

    cells = {(ar, ac), (ar, ac + 1)}
    while len(cells) < target:
        candidates = []
        for cell in cells:
            for nb in _neighbors(cell, 4):
                r, c = nb
                if 0 <= r < h and 0 <= c < w and nb not in cells:
                    candidates.append(nb)
        if not candidates:
            return None
        cells.add(random.choice(candidates))

    rel_cells = {(r - ar, c - ac) for r, c in cells}
    r0, c0, r1, c1 = _bbox(rel_cells)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 4:
        return None
    if not (r0 < 0 and r1 > 0 and c0 < 0 and c1 > 1):
        return None

    body = sorted(rel_cells - ANCHOR_RELS)
    if len(body) < 10:
        return None

    palette_size = random.randint(3, min(5, len(BODY_COLORS)))
    palette = random.sample(BODY_COLORS, palette_size)
    sprite: dict[tuple[int, int], int] = {(0, 0): RED, (0, 1): YELLOW}
    forced = palette[:]
    random.shuffle(forced)
    for i, rel in enumerate(body):
        sprite[rel] = forced[i] if i < len(forced) else random.choice(palette)

    if len(set(sprite.values()) - {RED, YELLOW}) < 3:
        return None
    return sprite


def _choose_rotations(count: int) -> list[int]:
    base = [0, 1, 2, 3]
    random.shuffle(base)
    if count <= 4:
        rotations = base[:count]
    else:
        rotations = base + [random.randrange(4)]
    random.shuffle(rotations)
    return rotations


def _make_one_visible(all_rels: set[tuple[int, int]], min_missing: int) -> set[tuple[int, int]] | None:
    visible = set(all_rels)
    body = list(all_rels - ANCHOR_RELS)
    random.shuffle(body)
    max_missing = max(min_missing, int(round(len(body) * random.uniform(0.35, 0.55))))
    missing = 0
    min_visible = max(7, int(round(len(all_rels) * 0.45)))

    for rel in body:
        if missing >= max_missing:
            break
        candidate = visible - {rel}
        if len(candidate) < min_visible:
            continue
        if _is_connected(candidate, connectivity=8):
            visible = candidate
            missing += 1

    if len(all_rels - visible) < min_missing:
        return None
    return visible


def _make_masks(sprite: dict[tuple[int, int], int], count: int) -> list[set[tuple[int, int]]] | None:
    all_rels = set(sprite)
    body = all_rels - ANCHOR_RELS
    min_missing = 2 if len(body) < 14 else 3

    for _ in range(GEN_MASK_RETRY_LIMIT):
        masks = []
        for _idx in range(count):
            visible = _make_one_visible(all_rels, min_missing)
            if visible is None:
                break
            masks.append(visible)
        if len(masks) != count:
            continue

        for rel in all_rels:
            if not any(rel in mask for mask in masks):
                masks[random.randrange(count)].add(rel)

        if not _masks_are_semantic(all_rels, masks, min_missing):
            continue
        return masks
    return None


def _masks_are_semantic(
    all_rels: set[tuple[int, int]], masks: list[set[tuple[int, int]]], min_missing: int
) -> bool:
    body = all_rels - ANCHOR_RELS
    for mask in masks:
        if not ANCHOR_RELS <= mask:
            return False
        if not mask < all_rels:
            return False
        if len(all_rels - mask) < min_missing:
            return False
        if not _is_connected(mask, connectivity=8):
            return False

    coverage = {rel: sum(rel in mask for mask in masks) for rel in all_rels}
    if any(v == 0 for v in coverage.values()):
        return False
    if min(coverage[rel] for rel in body) != 1:
        return False
    if not any(coverage[rel] >= 2 for rel in body):
        return False
    return True


def _cells_for(red: tuple[int, int], rels: set[tuple[int, int]], rotation: int) -> set[tuple[int, int]]:
    rr, cc = red
    return {(rr + dr, cc + dc) for dr, dc in (_rot_rel(rel, rotation) for rel in rels)}


def _dilate(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    H, W = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-GEN_PADDING, GEN_PADDING + 1):
            for dc in range(-GEN_PADDING, GEN_PADDING + 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    out.add((rr, cc))
    return out


def _candidate_reds(
    shape: tuple[int, int],
    rels: set[tuple[int, int]],
    rotation: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    H, W = shape
    rotated = {_rot_rel(rel, rotation) for rel in rels}
    r0, c0, r1, c1 = _bbox(rotated)
    rows = range(GEN_PLACEMENT_BORDER - r0, H - GEN_PLACEMENT_BORDER - r1)
    cols = range(GEN_PLACEMENT_BORDER - c0, W - GEN_PLACEMENT_BORDER - c1)
    candidates = []
    for rr in rows:
        for cc in cols:
            cells = _cells_for((rr, cc), rels, rotation)
            if not (cells & blocked):
                candidates.append((rr, cc))
    random.shuffle(candidates)
    return candidates


def _place_fragments(
    sprite: dict[tuple[int, int], int],
    masks: list[set[tuple[int, int]]],
    rotations: list[int],
) -> tuple[np.ndarray, np.ndarray] | None:
    rels = set(sprite)
    for _ in range(GEN_PLACEMENT_RETRY_LIMIT):
        H = random.randint(*GEN_GRID_SIZE_RANGE)
        W = random.randint(*GEN_GRID_SIZE_RANGE)
        shape = (H, W)
        inp = np.zeros(shape, dtype=int)
        expected = np.zeros(shape, dtype=int)
        blocked: set[tuple[int, int]] = set()
        placements: list[tuple[int, int]] = []

        ok = True
        for rotation in rotations:
            candidates = _candidate_reds(shape, rels, rotation, blocked)
            if not candidates:
                ok = False
                break
            red = candidates[0]
            full_cells = _cells_for(red, rels, rotation)
            blocked.update(_dilate(full_cells, shape))
            placements.append(red)

        if not ok:
            continue

        for red, rotation, visible in zip(placements, rotations, masks):
            for rel in rels:
                color = int(sprite[rel])
                dr, dc = _rot_rel(rel, rotation)
                r = red[0] + dr
                c = red[1] + dc
                expected[r, c] = color
                if rel in visible:
                    inp[r, c] = color

        return inp, expected
    return None


def _generator_semantic_check(
    inp: np.ndarray,
    expected: np.ndarray,
    sprite: dict[tuple[int, int], int],
    masks: list[set[tuple[int, int]]],
    rotations: list[int],
) -> bool:
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if len(set(rotations)) < min(3, len(rotations)):
        return False
    all_rels = set(sprite)
    if any(mask == all_rels for mask in masks):
        return False
    if not any(rotation != 0 for rotation in rotations):
        return False
    if not np.any(expected != inp):
        return False

    additions_per_fragment = [len(all_rels - mask) for mask in masks]
    if min(additions_per_fragment) < 2:
        return False
    try:
        actual = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    sprite = _build_sprite()
    if sprite is None:
        return None

    count = random.randint(*GEN_FRAGMENT_COUNT_RANGE)
    rotations = _choose_rotations(count)
    masks = _make_masks(sprite, count)
    if masks is None:
        return None

    placed = _place_fragments(sprite, masks, rotations)
    if placed is None:
        return None
    inp, expected = placed

    if not _generator_semantic_check(inp, expected, sprite, masks, rotations):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid residual-union sprite restoration input")
