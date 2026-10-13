from __future__ import annotations

from collections import Counter
from itertools import combinations, permutations
import random

import numpy as np


# ARC uses (row, column) coordinates and inclusive bboxes.  These small local
# helpers follow the conventions of the shared arc_utils.py toolbox while
# keeping this single task directly loadable by SourceFileLoader.
Coord = tuple[int, int]
GRAY = 5
PALETTE = tuple(range(10))
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (N, 2) arrays of (r, c)."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    result: list[np.ndarray] = []
    for r0, c0 in np.argwhere(mask):
        r0, c0 = int(r0), int(c0)
        if seen[r0, c0]:
            continue
        seen[r0, c0] = True
        stack = [(r0, c0)]
        cells: list[Coord] = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in NEIGHBORS4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _rotational_match(a: np.ndarray, b: np.ndarray) -> bool:
    return any(a.shape == rotated.shape and np.array_equal(a, rotated)
               for rotated in (np.rot90(b, k) for k in range(4)))


def _downscale_exactly_two(mask: np.ndarray) -> np.ndarray | None:
    """Undo nearest-neighbor 2x scaling, rejecting non-uniform 2x2 blocks."""
    h, w = mask.shape
    if h % 2 or w % 2:
        return None
    small = mask[::2, ::2]
    enlarged = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)
    return small.copy() if np.array_equal(mask, enlarged) else None


def _frame_cavities(g: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    """Find cavities strictly in each gray frame's own interior coordinates."""
    frames: list[dict] = []
    cavities: list[dict] = []
    for gray_coords in _components(g == GRAY):
        r0, c0, r1, c1 = _bbox(gray_coords)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        patch = g[r0:r1 + 1, c0:c1 + 1]
        perimeter_is_gray = (
            np.all(patch[0, :] == GRAY)
            and np.all(patch[-1, :] == GRAY)
            and np.all(patch[:, 0] == GRAY)
            and np.all(patch[:, -1] == GRAY)
        )
        if not perimeter_is_gray:
            continue

        frame_index = len(frames)
        frame = {"bbox": (r0, c0, r1, c1), "coords": gray_coords}
        frames.append(frame)
        interior = g[r0 + 1:r1, c0 + 1:c1]
        local_cavities = _components(interior == bg)
        assert local_cavities, "each gray frame must contain at least one background cavity"
        for local in local_cavities:
            absolute = local + np.array([r0 + 1, c0 + 1], dtype=int)
            cavities.append({
                "frame": frame_index,
                "coords": absolute,
                "shape": _tight_mask(absolute),
            })

    assert len(frames) >= 2, "expected multiple separated gray hollow frames"
    return frames, cavities


def _colored_sprites(g: np.ndarray, bg: int, frames: list[dict]) -> list[dict]:
    sprites: list[dict] = []
    for color in sorted(set(map(int, np.unique(g))) - {bg, GRAY}):
        for coords in _components(g == color):
            for r, c in coords:
                assert not any(
                    r0 <= int(r) <= r1 and c0 <= int(c) <= c1
                    for r0, c0, r1, c1 in (frame["bbox"] for frame in frames)
                ), "colored sprites must lie outside every frame"
            sprites.append({
                "color": int(color),
                "coords": coords,
                "shape": _tight_mask(coords),
            })
    assert sprites, "expected colored sprites outside the frames"
    return sprites


def _analyze(g: np.ndarray) -> dict:
    bg = _background(g)
    assert bg != GRAY, "gray is reserved for the frames"
    frames, cavities = _frame_cavities(g, bg)
    sprites = _colored_sprites(g, bg, frames)

    matches: list[int | None] = []
    for sprite in sprites:
        normalized = _downscale_exactly_two(sprite["shape"])
        assert normalized is not None, "every colored sprite must be an exact 2x enlargement"
        targets = {
            index for index, cavity in enumerate(cavities)
            if _rotational_match(normalized, cavity["shape"])
        }
        assert len(targets) <= 1, "a sprite must not match more than one cavity"
        matches.append(next(iter(targets)) if targets else None)

    unmatched = sum(target is None for target in matches)
    assert unmatched == 1, "expected exactly one unmatched distractor sprite"
    matched = [target for target in matches if target is not None]
    assert len(matched) == len(cavities), "every cavity must have exactly one key sprite"
    assert len(set(matched)) == len(matched), "two sprites must not claim the same cavity"
    return {
        "bg": bg,
        "frames": frames,
        "cavities": cavities,
        "sprites": sprites,
        "matches": matches,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gray-frame cavity from its rotated, half-scale colored key and erase all keys.

    Input structure:
    - A dominant background contains multiple separated gray rectangular frames whose
      gray-filled interiors leave uniquely shaped 4-connected background cavities.
    - Outside the frame bboxes are monochrome colored sprites; each is an exact 2x
      enlargement of one cavity after a possible quarter-turn, plus one distractor.

    Rule:
    - In each frame's own interior coordinate system, crop every background cavity.
    - Reduce each colored sprite by exactly 1/2, try its four rotations, and require a
      unique one-to-one cavity match. Fill that cavity with the sprite color.
    - Erase every colored sprite, including the unmatched distractor, to background;
      preserve every gray cell.

    Key insight:
    - Key and lock cannot be compared at their visible sizes: normalize the 2x scale
      first, then use rotation-equivalent shape topology rather than color, distance,
      area, dimensions, or ordering.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    analysis = _analyze(g)
    bg = int(analysis["bg"])

    out = g.copy()
    for sprite in analysis["sprites"]:
        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = bg
    for sprite, target in zip(analysis["sprites"], analysis["matches"]):
        if target is None:
            continue
        coords = analysis["cavities"][target]["coords"]
        out[coords[:, 0], coords[:, 1]] = int(sprite["color"])

    assert np.array_equal(out[g == GRAY], g[g == GRAY]), "gray frame cells must remain unchanged"
    return out


# ------------------------------- generator -------------------------------

SHAPE_SIZE = 4
SHAPE_AREAS = (7, 8, 9)
FRAME_COUNT_RANGE = (2, 3)
MAX_GENERATION_ATTEMPTS = 400
_SHAPE_CACHE: dict[int, list[np.ndarray]] = {}


def _is_connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(_components(mask)) == 1


def _has_hole(mask: np.ndarray) -> bool:
    for coords in _components(~mask):
        if not any(
            int(r) in (0, mask.shape[0] - 1) or int(c) in (0, mask.shape[1] - 1)
            for r, c in coords
        ):
            return True
    return False


def _rotation_key(mask: np.ndarray) -> bytes:
    return min(np.packbits(np.rot90(mask, k).ravel()).tobytes() for k in range(4))


def _shape_bank(area: int) -> list[np.ndarray]:
    if area in _SHAPE_CACHE:
        return _SHAPE_CACHE[area]

    bank: list[np.ndarray] = []
    seen: set[bytes] = set()
    for chosen in combinations(range(SHAPE_SIZE * SHAPE_SIZE), area):
        mask = np.zeros((SHAPE_SIZE, SHAPE_SIZE), dtype=bool)
        mask.ravel()[list(chosen)] = True
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if not _is_connected(mask) or _has_hole(mask):
            continue
        rotations = [np.rot90(mask, k) for k in range(4)]
        if len({rot.tobytes() for rot in rotations}) != 4:
            continue
        key = _rotation_key(mask)
        if key not in seen:
            seen.add(key)
            bank.append(mask)
    _SHAPE_CACHE[area] = bank
    return bank


def _free_bbox_positions(reserved: np.ndarray, h: int, w: int) -> list[Coord]:
    H, W = reserved.shape
    positions: list[Coord] = []
    for r in range(1, H - h):
        for c in range(1, W - w):
            rr0, cc0 = max(0, r - 1), max(0, c - 1)
            rr1, cc1 = min(H, r + h + 1), min(W, c + w + 1)
            if not reserved[rr0:rr1, cc0:cc1].any():
                positions.append((r, c))
    return positions


def _place_footprint(reserved: np.ndarray, h: int, w: int) -> Coord | None:
    positions = _free_bbox_positions(reserved, h, w)
    if not positions:
        return None
    r, c = random.choice(positions)
    reserved[r:r + h, c:c + w] = True
    return r, c


def _center(pos: Coord, h: int, w: int) -> tuple[float, float]:
    return pos[0] + (h - 1) / 2, pos[1] + (w - 1) / 2


def _distance_counterexample(
    frame_positions: list[Coord], sprite_positions: list[Coord], assignment: tuple[int, ...]
) -> bool:
    frame_centers = [_center(pos, 6, 6) for pos in frame_positions]
    for sprite_pos, shape_index in zip(sprite_positions, assignment):
        sr, sc = _center(sprite_pos, 8, 8)
        distances = [(sr - fr) ** 2 + (sc - fc) ** 2 for fr, fc in frame_centers]
        nearest = int(np.argmin(distances))
        if distances.count(min(distances)) == 1 and nearest != shape_index:
            return True
    return False


def _choose_assignment(frame_positions: list[Coord], matched_positions: list[Coord]) -> tuple[int, ...] | None:
    n = len(frame_positions)
    frame_order = tuple(sorted(range(n), key=lambda i: frame_positions[i]))
    options = []
    for assignment in permutations(range(n)):
        sprite_order = tuple(
            assignment[i] for i in sorted(range(n), key=lambda j: matched_positions[j])
        )
        if sprite_order == frame_order:
            continue
        if _distance_counterexample(frame_positions, matched_positions, assignment):
            options.append(assignment)
    return random.choice(options) if options else None


def _render_frame(grid: np.ndarray, pos: Coord, cavity: np.ndarray, bg: int) -> np.ndarray:
    r, c = pos
    patch = np.full((6, 6), GRAY, dtype=int)
    patch[1:5, 1:5][cavity] = bg
    grid[r:r + 6, c:c + 6] = patch
    coords = np.argwhere(cavity) + np.array([r + 1, c + 1])
    return coords


def _render_sprite(grid: np.ndarray, pos: Coord, shape: np.ndarray, color: int) -> np.ndarray:
    r, c = pos
    enlarged = np.repeat(np.repeat(shape, 2, axis=0), 2, axis=1)
    patch = grid[r:r + 8, c:c + 8]
    patch[enlarged] = int(color)
    return np.argwhere(enlarged) + np.array([r, c])


def _independent_oracle(
    inp: np.ndarray,
    bg: int,
    cavity_coords: list[np.ndarray],
    sprite_coords: list[np.ndarray],
    sprite_assignment: list[int | None],
    sprite_colors: list[int],
) -> np.ndarray:
    out = inp.copy()
    for coords in sprite_coords:
        out[coords[:, 0], coords[:, 1]] = bg
    for coords, target, color in zip(sprite_coords, sprite_assignment, sprite_colors):
        if target is not None:
            cavity = cavity_coords[target]
            out[cavity[:, 0], cavity[:, 1]] = int(color)
    return out


def _semantic_validation(
    inp: np.ndarray,
    expected: np.ndarray,
    area: int,
    rotations: list[int],
    frame_positions: list[Coord],
    matched_positions: list[Coord],
    assignment: tuple[int, ...],
) -> bool:
    """Check construction semantics and anti-shortcut witnesses, not only closure."""
    try:
        analysis = _analyze(inp)
        normalized = [_downscale_exactly_two(s["shape"]) for s in analysis["sprites"]]
        if any(mask is None for mask in normalized):
            return False
        # Area and 4x4 bbox deliberately cannot identify a target.
        if any(mask.shape != (4, 4) or int(mask.sum()) != area for mask in normalized):
            return False
        if any(cavity["shape"].shape != (4, 4) or int(cavity["shape"].sum()) != area
               for cavity in analysis["cavities"]):
            return False
        # Both the no-turn and genuine quarter-turn branches occur in every sample.
        if 0 not in rotations or not any(k in (1, 3) for k in rotations):
            return False
        if not _distance_counterexample(frame_positions, matched_positions, assignment):
            return False
        if np.array_equal(inp, expected) or not np.array_equal(expected[inp == GRAY], inp[inp == GRAY]):
            return False
        return np.array_equal(main(inp), expected)
    except (AssertionError, ValueError):
        return False


def _generation_attempt() -> np.ndarray | None:
    n = random.randint(*FRAME_COUNT_RANGE)
    H = random.randint(24, 28) if n == 2 else random.randint(28, 30)
    W = random.randint(24, 28) if n == 2 else random.randint(28, 30)
    bg = random.choice([color for color in PALETTE if color != GRAY])
    sprite_colors = random.sample([color for color in PALETTE if color not in (bg, GRAY)], n + 1)

    area = random.choice(SHAPE_AREAS)
    bank = _shape_bank(area)
    if len(bank) < n + 1:
        return None
    shapes = [shape.copy() for shape in random.sample(bank, n + 1)]

    reserved = np.zeros((H, W), dtype=bool)
    footprints = [("frame", i, 6, 6) for i in range(n)] + [
        ("sprite", i, 8, 8) for i in range(n + 1)
    ]
    random.shuffle(footprints)
    frame_positions: list[Coord | None] = [None] * n
    sprite_positions: list[Coord | None] = [None] * (n + 1)
    for kind, index, h, w in footprints:
        pos = _place_footprint(reserved, h, w)
        if pos is None:
            return None
        if kind == "frame":
            frame_positions[index] = pos
        else:
            sprite_positions[index] = pos
    assert all(pos is not None for pos in frame_positions + sprite_positions)
    frames = [pos for pos in frame_positions if pos is not None]
    sprites = [pos for pos in sprite_positions if pos is not None]

    distractor_slot = random.randrange(n + 1)
    matched_slots = [slot for slot in range(n + 1) if slot != distractor_slot]
    matched_positions = [sprites[slot] for slot in matched_slots]
    assignment = _choose_assignment(frames, matched_positions)
    if assignment is None:
        return None

    rotations = [0, random.choice((1, 3))]
    if n == 3:
        rotations.append(random.choice((1, 2, 3)))
    random.shuffle(rotations)

    grid = np.full((H, W), bg, dtype=int)
    cavity_coords = [_render_frame(grid, frames[i], shapes[i], bg) for i in range(n)]
    sprite_coords: list[np.ndarray] = [np.empty((0, 2), dtype=int) for _ in range(n + 1)]
    sprite_assignment: list[int | None] = [None] * (n + 1)
    for matched_i, (slot, shape_index) in enumerate(zip(matched_slots, assignment)):
        rotated = np.rot90(shapes[shape_index], rotations[matched_i])
        sprite_coords[slot] = _render_sprite(grid, sprites[slot], rotated, sprite_colors[slot])
        sprite_assignment[slot] = shape_index
    distractor_rotation = random.randrange(4)
    sprite_coords[distractor_slot] = _render_sprite(
        grid, sprites[distractor_slot], np.rot90(shapes[-1], distractor_rotation),
        sprite_colors[distractor_slot]
    )

    expected = _independent_oracle(
        grid, bg, cavity_coords, sprite_coords, sprite_assignment, sprite_colors
    )
    if not _semantic_validation(
        grid, expected, area, rotations, frames, matched_positions, assignment
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a diverse, semantically checked input grid of at most 30x30."""
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid enlarged-key cavity puzzle")
