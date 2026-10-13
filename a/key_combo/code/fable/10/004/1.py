from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

DIRS = ("left", "right", "above", "below")
GRID_MIN = 12
GRID_MAX = 30
GAP_RANGE = (2, 5)
STEP_CHOICES = (2, 3, 4)
STEP_WEIGHTS = (2, 3, 2)
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: list[Coord] | tuple[Coord, ...] | set[Coord]) -> BBox:
    rs = [int(r) for r, _ in coords]
    cs = [int(c) for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _neighbors(r: int, c: int, connectivity: int) -> list[Coord]:
    if connectivity == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _find_components(grid: np.ndarray, bg: int, connectivity: int = 8) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(
                {
                    "color": color,
                    "coords": tuple(sorted(coords)),
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    return components


def _is_connected_mask(mask: np.ndarray) -> bool:
    pts = [tuple(map(int, p)) for p in np.argwhere(mask)]
    if not pts:
        return False
    seen = {pts[0]}
    stack = [pts[0]]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors(r, c, 8):
            if (nr, nc) in seen:
                continue
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]):
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(pts)


def _normalize(coords: list[Coord] | tuple[Coord, ...] | set[Coord]) -> frozenset[Coord]:
    r0, c0, _, _ = _bbox(coords)
    return frozenset((int(r) - r0, int(c) - c0) for r, c in coords)


def _anchor_cells(box: BBox) -> set[Coord]:
    r0, c0, r1, c1 = map(int, box)
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _paint_coords(grid: np.ndarray, coords: list[Coord] | tuple[Coord, ...] | set[Coord], color: int) -> None:
    for r, c in coords:
        grid[int(r), int(c)] = int(color)


def _paint_anchor(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = map(int, box)
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _is_solid_2x2(comp: dict, grid: np.ndarray) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    return (
        int(comp["area"]) == 4
        and r1 - r0 + 1 == 2
        and c1 - c0 + 1 == 2
        and bool((grid[r0 : r1 + 1, c0 : c1 + 1] == int(comp["color"])).all())
    )


def _relation(actor: list[Coord] | tuple[Coord, ...] | set[Coord], anchor_box: BBox) -> tuple[str, int, int] | None:
    sr0, sc0, sr1, sc1 = _bbox(actor)
    ar0, ac0, ar1, ac1 = map(int, anchor_box)
    matches: list[tuple[str, int, int]] = []
    if sc1 < ac0 and _intervals_overlap(sr0, sr1, ar0, ar1):
        matches.append(("left", 0, ac0 - sc1 - 1))
    if sc0 > ac1 and _intervals_overlap(sr0, sr1, ar0, ar1):
        matches.append(("right", 0, ac1 + 1 - sc0))
    if sr1 < ar0 and _intervals_overlap(sc0, sc1, ac0, ac1):
        matches.append(("above", ar0 - sr1 - 1, 0))
    if sr0 > ar1 and _intervals_overlap(sc0, sc1, ac0, ac1):
        matches.append(("below", ar1 + 1 - sr0, 0))
    if len(matches) != 1:
        return None
    return matches[0]


def _mirror_step(
    actor: list[Coord] | tuple[Coord, ...] | set[Coord], anchor_box: BBox, side: str, dr: int, dc: int
) -> tuple[tuple[Coord, ...], tuple[Coord, ...]]:
    ar0, ac0, ar1, ac1 = map(int, anchor_box)
    moved: list[Coord] = []
    mirrored: list[Coord] = []
    for r, c in actor:
        rr = int(r) + int(dr)
        cc = int(c) + int(dc)
        moved.append((rr, cc))
        if side == "left":
            mirrored.append((rr, 2 * ac0 - 1 - cc))
        elif side == "right":
            mirrored.append((rr, 2 * ac1 + 1 - cc))
        elif side == "above":
            mirrored.append((2 * ar0 - 1 - rr, cc))
        else:
            mirrored.append((2 * ar1 + 1 - rr, cc))
    return tuple(sorted(moved)), tuple(sorted(mirrored))


def _simulate_chain(initial_actor: tuple[Coord, ...], anchors: list[dict]) -> tuple[list[dict], tuple[Coord, ...]]:
    current = tuple(sorted(initial_actor))
    used: set[int] = set()
    copies: list[dict] = []
    while True:
        matches: list[tuple[int, dict, str, int, int]] = []
        for idx, anchor in enumerate(anchors):
            if idx in used:
                continue
            rel = _relation(current, anchor["bbox"])
            if rel is not None:
                side, dr, dc = rel
                matches.append((idx, anchor, side, dr, dc))
        if not matches:
            break
        assert len(matches) == 1, "current actor must face exactly one unused anchor"
        idx, anchor, side, dr, dc = matches[0]
        moved, mirrored = _mirror_step(current, anchor["bbox"], side, dr, dc)
        copies.append(
            {
                "anchor_index": idx,
                "anchor_color": int(anchor["color"]),
                "side": side,
                "moved": moved,
                "coords": mirrored,
            }
        )
        used.add(idx)
        current = mirrored
        assert len(used) <= len(anchors)
    return copies, current


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the current sprite to each uniquely faced anchor and keep every mirrored relay copy.

    Input structure:
    - The background is the most common color.
    - There is one non-square monochrome sprite and several 2x2 solid square anchors in distinct colors.
    - At any moment, the current acting sprite faces exactly one still-unused anchor along one row-aligned or column-aligned direction.

    Rule:
    - Slide the current sprite straight toward that unique anchor until it touches the anchor's facing side.
    - Reflect the slid sprite across that contact side into the anchor's side and recolor the reflected copy with the anchor color.
    - The pre-mirror sprite and the used anchor disappear from the working state; the reflected copy becomes the next acting sprite.
    - Repeat until no unused anchor faces the current sprite. The output keeps every reflected relay copy, while the initial sprite and all anchors are absent.

    Key insight:
    - The anchor order is not read directly from the input layout: each reflected copy changes the next actor position, so the chain is determined causally step by step.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _find_components(g, bg=bg, connectivity=8)
    anchors = [comp for comp in comps if _is_solid_2x2(comp, g)]
    sprites = [comp for comp in comps if comp not in anchors]
    assert len(sprites) == 1, "expected exactly one non-anchor sprite"

    copies, _ = _simulate_chain(tuple(sprites[0]["coords"]), anchors)
    out = np.full_like(g, bg)
    for copy in copies:
        _paint_coords(out, copy["coords"], copy["anchor_color"])
    return out


def _enumerate_sprite_masks() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for h, w in ((2, 3), (3, 2), (3, 3), (3, 4), (4, 3)):
        for area in range(4, min(7, h * w + 1)):
            for cells in itertools.combinations(range(h * w), area):
                mask = np.zeros((h, w), dtype=bool)
                for cell in cells:
                    mask[cell // w, cell % w] = True
                if not _is_connected_mask(mask):
                    continue
                if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
                    continue
                if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)):
                    continue
                masks.append(mask)
    assert masks
    return masks


SPRITE_MASKS = _enumerate_sprite_masks()


def _canonical_anchor_box(actor: tuple[Coord, ...], side: str, gap: int) -> BBox:
    r0, c0, r1, c1 = _bbox(actor)
    if side == "left":
        return (r0, c1 + gap + 1, r0 + 1, c1 + gap + 2)
    if side == "right":
        return (r0, c0 - gap - 2, r0 + 1, c0 - gap - 1)
    if side == "above":
        return (r1 + gap + 1, c0, r1 + gap + 2, c0 + 1)
    return (r0 - gap - 2, c0, r0 - gap - 1, c0 + 1)


def _prefix_unique(states: list[tuple[Coord, ...]], anchors: list[BBox]) -> bool:
    for step_idx in range(len(anchors)):
        candidates = [idx for idx in range(step_idx, len(anchors)) if _relation(states[step_idx], anchors[idx]) is not None]
        if candidates != [step_idx]:
            return False
    return True


def _sample_direction(prev_side: str | None) -> str:
    if prev_side is None:
        return random.choice(DIRS)
    prev_axis_horizontal = prev_side in {"left", "right"}
    turn_dirs = [d for d in DIRS if (d in {"left", "right"}) != prev_axis_horizontal]
    straight_dirs = [d for d in DIRS if d not in turn_dirs]
    if random.random() < 0.65:
        return random.choice(turn_dirs)
    return random.choice(straight_dirs)


def _build_chain() -> tuple[tuple[Coord, ...], list[BBox], list[tuple[Coord, ...]]]:
    mask = random.choice(SPRITE_MASKS)
    initial_actor = tuple(sorted((int(r), int(c)) for r, c in np.argwhere(mask)))
    initial_set = set(initial_actor)

    steps = random.choices(STEP_CHOICES, weights=STEP_WEIGHTS, k=1)[0]
    states = [initial_actor]
    anchors: list[BBox] = []
    copies: list[tuple[Coord, ...]] = []
    occupied_copies: set[Coord] = set()
    current = initial_actor
    prev_side: str | None = None

    for _step in range(steps):
        placed = False
        for _ in range(500):
            side = _sample_direction(prev_side)
            gap = random.randint(*GAP_RANGE)
            box = _canonical_anchor_box(current, side, gap)
            anchor_pts = _anchor_cells(box)
            if anchor_pts & initial_set:
                continue
            if anchor_pts & occupied_copies:
                continue
            if any(anchor_pts & _anchor_cells(old_box) for old_box in anchors):
                continue
            rel = _relation(current, box)
            if rel is None or rel[0] != side:
                continue
            _, dr, dc = rel
            _moved, mirrored = _mirror_step(current, box, side, dr, dc)
            mirrored_set = set(mirrored)
            if _normalize(current) == _normalize(mirrored):
                continue
            if mirrored_set & initial_set:
                continue
            if mirrored_set & occupied_copies:
                continue
            if any(mirrored_set & _anchor_cells(old_box) for old_box in anchors):
                continue
            trial_states = states + [mirrored]
            trial_anchors = anchors + [box]
            if not _prefix_unique(trial_states, trial_anchors):
                continue
            states = trial_states
            anchors = trial_anchors
            copies.append(mirrored)
            occupied_copies |= mirrored_set
            current = mirrored
            prev_side = side
            placed = True
            break
        if not placed:
            raise RuntimeError("failed to place a unique next anchor")
    return initial_actor, anchors, copies


def _translate(coords: tuple[Coord, ...], dr: int, dc: int) -> tuple[Coord, ...]:
    return tuple(sorted((int(r) + int(dr), int(c) + int(dc)) for r, c in coords))


def _translate_box(box: BBox, dr: int, dc: int) -> BBox:
    r0, c0, r1, c1 = map(int, box)
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def _render_pair(initial_actor: tuple[Coord, ...], anchors: list[BBox], copies: list[tuple[Coord, ...]]) -> tuple[np.ndarray, np.ndarray]:
    all_points = set(initial_actor)
    for box in anchors:
        all_points |= _anchor_cells(box)
    for copy in copies:
        all_points |= set(copy)

    r0, c0, r1, c1 = _bbox(all_points)
    span_h = r1 - r0 + 1
    span_w = c1 - c0 + 1
    if span_h > GRID_MAX or span_w > GRID_MAX:
        raise RuntimeError("chain exceeds ARC grid bounds")

    extra_h = random.randint(0, min(6, GRID_MAX - span_h))
    extra_w = random.randint(0, min(6, GRID_MAX - span_w))
    top = random.randint(0, extra_h)
    left = random.randint(0, extra_w)
    height = span_h + extra_h
    width = span_w + extra_w
    if height < GRID_MIN:
        top += GRID_MIN - height
        height = GRID_MIN
    if width < GRID_MIN:
        left += GRID_MIN - width
        width = GRID_MIN
    if height > GRID_MAX or width > GRID_MAX:
        raise RuntimeError("translated grid exceeds ARC bounds")

    shift_r = top - r0
    shift_c = left - c0
    initial_actor = _translate(initial_actor, shift_r, shift_c)
    anchors = [_translate_box(box, shift_r, shift_c) for box in anchors]
    copies = [_translate(copy, shift_r, shift_c) for copy in copies]

    bg = random.randrange(10)
    palette = [color for color in range(10) if color != bg]
    chosen = random.sample(palette, len(anchors) + 1)
    sprite_color = chosen[0]
    anchor_colors = chosen[1:]

    inp = np.full((height, width), bg, dtype=int)
    out = np.full_like(inp, bg)
    _paint_coords(inp, initial_actor, sprite_color)
    for box, color in zip(anchors, anchor_colors):
        _paint_anchor(inp, box, color)
    for copy, color in zip(copies, anchor_colors):
        _paint_coords(out, copy, color)
    return inp, out


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    bg = _get_bg(inp)
    comps = _find_components(inp, bg=bg, connectivity=8)
    anchors = [comp for comp in comps if _is_solid_2x2(comp, inp)]
    sprites = [comp for comp in comps if comp not in anchors]
    if len(sprites) != 1 or len(anchors) < 2:
        return False

    initial_actor = tuple(sprites[0]["coords"])
    copies, _ = _simulate_chain(initial_actor, anchors)
    if len(copies) != len(anchors):
        return False
    if {int(copy["anchor_index"]) for copy in copies} != set(range(len(anchors))):
        return False

    states = [initial_actor] + [tuple(copy["coords"]) for copy in copies]
    anchor_boxes = [tuple(anchor["bbox"]) for anchor in anchors]

    seen_copy_cells: set[Coord] = set()
    initial_set = set(initial_actor)
    rendered = np.full_like(inp, bg)
    for step_idx, copy in enumerate(copies):
        coords = tuple(copy["coords"])
        if _normalize(states[step_idx]) == _normalize(coords):
            return False
        coord_set = set(coords)
        if coord_set & initial_set:
            return False
        if coord_set & seen_copy_cells:
            return False
        for other_idx, other_box in enumerate(anchor_boxes):
            if other_idx == copy["anchor_index"]:
                continue
            if coord_set & _anchor_cells(other_box):
                return False
        seen_copy_cells |= coord_set
        _paint_coords(rendered, coords, int(copy["anchor_color"]))

    if not np.array_equal(rendered, expected):
        return False
    if len(copies) == 1:
        return False
    first_only = np.full_like(inp, bg)
    _paint_coords(first_only, copies[0]["coords"], int(copies[0]["anchor_color"]))
    if np.array_equal(first_only, expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    try:
        initial_actor, anchor_boxes, copies = _build_chain()
        inp, expected = _render_pair(initial_actor, anchor_boxes, copies)
    except Exception:
        return None
    if not _semantic_checks(inp, expected):
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chain-mirror relay input")
