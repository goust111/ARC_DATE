from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7

BODY_COLORS = (BLUE, GREEN, GREY, ORANGE)
SIDE_STEPS = {
    "top": (-1, 0),
    "right": (0, 1),
    "bottom": (1, 0),
    "left": (0, -1),
}
OPPOSITE_SIDE = {
    "top": "bottom",
    "right": "left",
    "bottom": "top",
    "left": "right",
}
DIR_TO_STEP = {
    0: (-1, 0),  # up
    1: (0, 1),   # right
    2: (1, 0),   # down
    3: (0, -1),  # left
}
SIDE_TO_CONNECTOR = {
    "top": (3, 5),
    "right": (5, 7),
    "bottom": (7, 5),
    "left": (5, 3),
}
BODY_ZONE_ORIGIN = (4, 4)
BODY_SIDE_SUPPORT = {
    "top": (0, 1),
    "right": (1, 2),
    "bottom": (2, 1),
    "left": (1, 0),
}
BODY_TEMPLATES = (
    {
        "mask": np.array(
            [
                [1, 1, 0],
                [0, 1, 1],
                [0, 1, 0],
            ],
            dtype=bool,
        ),
        "supports": {"top", "right", "bottom"},
    },
    {
        "mask": np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [1, 0, 0],
            ],
            dtype=bool,
        ),
        "supports": {"top", "right", "left"},
    },
    {
        "mask": np.array(
            [
                [1, 0, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=bool,
        ),
        "supports": {"right", "bottom", "left"},
    },
    {
        "mask": np.array(
            [
                [0, 1, 0],
                [1, 1, 0],
                [0, 1, 1],
            ],
            dtype=bool,
        ),
        "supports": {"top", "bottom", "left"},
    },
)


def _neighbors(r: int, c: int, connectivity: int) -> Iterable[tuple[int, int]]:
    if connectivity == 4:
        yield r - 1, c
        yield r + 1, c
        yield r, c - 1
        yield r, c + 1
        return
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_crop_with_offset(arr: np.ndarray, bg: int = BLACK) -> tuple[np.ndarray, int, int]:
    coords = np.argwhere(arr != bg)
    if coords.size == 0:
        return arr[:0, :0].copy(), 0, 0
    r0, c0, r1, c1 = _bbox_coords(coords)
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy(), r0, c0


def _tight_crop(arr: np.ndarray, bg: int = BLACK) -> np.ndarray:
    crop, _r0, _c0 = _tight_crop_with_offset(arr, bg=bg)
    return crop


def _rotate_clockwise(arr: np.ndarray, steps: int) -> np.ndarray:
    return np.rot90(arr, (-steps) % 4).copy()


def _dilate_once(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for r, c in np.argwhere(mask):
        r0 = max(0, int(r) - 1)
        r1 = min(h, int(r) + 2)
        c0 = max(0, int(c) - 1)
        c1 = min(w, int(c) + 2)
        out[r0:r1, c0:c1] = True
    return out


def _find_indicator_angle(crop: np.ndarray) -> int:
    h, w = crop.shape
    matches: list[tuple[int, int, int]] = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = crop[r0 : r0 + 3, c0 : c0 + 3]
            if int(sub[1, 1]) != BLACK:
                continue
            cardinal_hits = []
            for angle, (dr, dc) in DIR_TO_STEP.items():
                if int(sub[1 + dr, 1 + dc]) == PINK:
                    cardinal_hits.append(angle)
            if len(cardinal_hits) != 1:
                continue
            other = sub.copy()
            other[1, 1] = BLACK
            for dr, dc in DIR_TO_STEP.values():
                other[1 + dr, 1 + dc] = BLACK
            if np.any(other != BLACK):
                continue
            matches.append((r0, c0, cardinal_hits[0]))
    assert matches, "expected at least one 3x3 rotation-indicator patch"
    matches.sort()
    return int(matches[0][2])


def _has_yellow_anchor(crop: np.ndarray) -> bool:
    h, w = crop.shape
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            if np.all(crop[r0 : r0 + 3, c0 : c0 + 3] == YELLOW):
                return True
    return False


def _red_coords(arr: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(arr == RED)]


def _extract_sprites(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    mask = g != BLACK
    expanded = _dilate_once(mask)
    groups = _components(expanded, connectivity=8)
    sprites = []
    for comp in groups:
        r0, c0, r1, c1 = _bbox_coords(comp)
        submask = mask[r0 : r1 + 1, c0 : c1 + 1]
        coords = np.argwhere(submask)
        assert coords.size > 0
        sr0, sc0, sr1, sc1 = _bbox_coords(coords)
        rr0, cc0 = r0 + sr0, c0 + sc0
        rr1, cc1 = r0 + sr1, c0 + sc1
        crop = g[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()
        sprites.append(
            {
                "bbox": (int(rr0), int(cc0), int(rr1), int(cc1)),
                "crop": crop,
                "angle": _find_indicator_angle(crop),
                "is_anchor": _has_yellow_anchor(crop),
            }
        )
    sprites.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return sprites


def _overlay_with_counts(
    canvas: np.ndarray,
    piece: np.ndarray,
    top: int,
    left: int,
    red_counts: dict[tuple[int, int], int],
    *,
    require_overlap: bool,
) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    h, w = canvas.shape
    ph, pw = piece.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return None

    overlaps: list[tuple[int, int]] = []
    for pr, pc in np.argwhere(piece != BLACK):
        gr = top + int(pr)
        gc = left + int(pc)
        src = int(piece[pr, pc])
        dst = int(canvas[gr, gc])
        if dst == BLACK:
            continue
        if src == RED and dst == RED and red_counts.get((gr, gc), 0) == 1:
            overlaps.append((gr, gc))
            continue
        return None

    if require_overlap and not overlaps:
        return None

    out = canvas.copy()
    for pr, pc in np.argwhere(piece != BLACK):
        gr = top + int(pr)
        gc = left + int(pc)
        out[gr, gc] = int(piece[pr, pc])
    return out, overlaps


def _assemble_from_rotated_pieces(pieces: list[dict], grid_shape: tuple[int, int]) -> np.ndarray:
    anchors = [idx for idx, piece in enumerate(pieces) if piece["is_anchor"]]
    assert len(anchors) == 1, "expected exactly one anchor sprite"
    anchor_idx = anchors[0]
    h, w = grid_shape
    canvas = np.full((h, w), BLACK, dtype=int)
    anchor = pieces[anchor_idx]
    ar0, ac0, _ar1, _ac1 = anchor["bbox"]
    placed = _overlay_with_counts(canvas, anchor["rotated"], ar0, ac0, {}, require_overlap=False)
    assert placed is not None
    canvas, _ = placed

    red_counts: dict[tuple[int, int], int] = {}
    for rr, cc in anchor["reds"]:
        red_counts[(ar0 + rr, ac0 + cc)] = 1

    remaining = [idx for idx in range(len(pieces)) if idx != anchor_idx]
    solutions: list[np.ndarray] = []

    def feasible_placements(piece_idx: int) -> list[tuple[int, int, tuple[tuple[int, int], ...]]]:
        piece = pieces[piece_idx]
        unpaired = [coord for coord, count in red_counts.items() if count == 1]
        cand: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
        for ur, uc in unpaired:
            for pr, pc in piece["reds"]:
                top = ur - pr
                left = uc - pc
                placed_res = _overlay_with_counts(canvas, piece["rotated"], top, left, red_counts, require_overlap=True)
                if placed_res is None:
                    continue
                _next_canvas, overlaps = placed_res
                key = (int(top), int(left))
                cur = tuple(sorted(overlaps))
                if key not in cand or len(cur) > len(cand[key]):
                    cand[key] = cur
        out = [(top, left, overlaps) for (top, left), overlaps in cand.items()]
        out.sort(key=lambda item: (-len(item[2]), item[0], item[1]))
        return out

    def dfs(cur_canvas: np.ndarray, cur_counts: dict[tuple[int, int], int], rem: list[int]) -> None:
        nonlocal canvas, red_counts
        if len(solutions) >= 2:
            return
        if not rem:
            if all(count == 2 for count in cur_counts.values()):
                solutions.append(cur_canvas.copy())
            return

        prev_canvas = canvas
        prev_counts = red_counts
        canvas = cur_canvas
        red_counts = cur_counts

        choices: list[tuple[int, tuple[int, int], list[tuple[int, int, tuple[tuple[int, int], ...]]]]] = []
        for piece_idx in rem:
            placements = feasible_placements(piece_idx)
            if placements:
                choices.append((piece_idx, pieces[piece_idx]["bbox"][:2], placements))
        if not choices:
            canvas = prev_canvas
            red_counts = prev_counts
            return

        choices.sort(key=lambda item: item[1])
        piece_idx, _bbox_key, placements = choices[0]
        next_rem = [idx for idx in rem if idx != piece_idx]
        piece = pieces[piece_idx]

        for top, left, _overlaps in placements:
            placed_res = _overlay_with_counts(cur_canvas, piece["rotated"], top, left, cur_counts, require_overlap=True)
            if placed_res is None:
                continue
            next_canvas, overlaps = placed_res
            next_counts = dict(cur_counts)
            ok = True
            for rr, cc in piece["reds"]:
                coord = (top + rr, left + cc)
                next_counts[coord] = next_counts.get(coord, 0) + 1
                if next_counts[coord] > 2:
                    ok = False
                    break
            if not ok:
                continue
            if tuple(sorted(overlaps)) != tuple(sorted(_overlaps)):
                continue
            dfs(next_canvas, next_counts, next_rem)
            if len(solutions) >= 2:
                break

        canvas = prev_canvas
        red_counts = prev_counts

    dfs(canvas, red_counts, remaining)
    assert len(solutions) == 1, "assembly must have exactly one deterministic output"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each sprite by its clock-hand marker, then assemble all sprites onto the yellow-anchor sprite by overlapping matching red connector pixels.

    Input structure:
    - Black (0) background contains 4-6 separated sprites.
    - Each sprite has a colored 4-connected body, 1-3 red connector pixels on its boundary, and one isolated 3x3 rotation-indicator patch whose single pink pixel points up/right/down/left.
    - Exactly one sprite also contains a solid 3x3 yellow block; that sprite is the fixed anchor and its indicator points up.

    Rule:
    - Read each non-anchor sprite's indicator as a clockwise rotation: up=0, right=90, down=180, left=270.
    - Rotate the entire sprite around its own tight bounding-box center, which is equivalent to rotating its tight crop by quarter turns.
    - Keep the anchor in place, then repeatedly attach one unplaced rotated sprite by translating it so one or more of its red connector pixels overlap currently unpaired red connector pixels in the growing assembly.
    - When several sprites are attachable at the same time, choose the one whose original input bbox has the smallest top row, then the smallest left column; paired red pixels remain red in the output.

    Key insight:
    - The connector geometry that determines which sprite can attach only exists after rotating the whole sprite, so rotation must be resolved before connector matching and incremental assembly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sprites = _extract_sprites(g)
    assert 4 <= len(sprites) <= 6, "expected 4-6 sprites"
    assert sum(1 for sprite in sprites if sprite["is_anchor"]) == 1, "expected exactly one anchor"

    rotated_pieces = []
    for sprite in sprites:
        angle = 0 if sprite["is_anchor"] else int(sprite["angle"])
        if sprite["is_anchor"]:
            assert int(sprite["angle"]) == 0, "anchor indicator must point up"
        rotated = _rotate_clockwise(sprite["crop"], angle)
        reds = _red_coords(rotated)
        assert 1 <= len(reds) <= 3, "each sprite must expose 1-3 red connectors after rotation"
        rotated_pieces.append(
            {
                "bbox": sprite["bbox"],
                "is_anchor": sprite["is_anchor"],
                "rotated": rotated,
                "reds": reds,
            }
        )

    return _assemble_from_rotated_pieces(rotated_pieces, g.shape)


def _allowed_side_sets(count: int) -> list[tuple[str, ...]]:
    if count == 1:
        return [("top",), ("right",), ("bottom",), ("left",)]
    if count == 2:
        return [
            ("top", "right"),
            ("top", "bottom"),
            ("top", "left"),
            ("right", "bottom"),
            ("right", "left"),
            ("bottom", "left"),
        ]
    if count == 3:
        return [
            ("top", "right", "bottom"),
            ("right", "bottom", "left"),
            ("bottom", "left", "top"),
            ("left", "top", "right"),
        ]
    raise ValueError("connector count must be 1..3")


def _choose_tree(piece_count: int, rng: random.Random) -> list[int]:
    options = {
        4: [
            [-1, 0, 0, 1],
            [-1, 0, 1, 1],
        ],
        5: [
            [-1, 0, 0, 1, 1],
            [-1, 0, 0, 1, 2],
            [-1, 0, 1, 1, 2],
        ],
        6: [
            [-1, 0, 0, 1, 1, 2],
            [-1, 0, 0, 1, 2, 2],
            [-1, 0, 1, 1, 2, 2],
        ],
    }
    return list(rng.choice(options[piece_count]))


def _children_of(parents: list[int]) -> list[list[int]]:
    children = [[] for _ in parents]
    for idx, parent in enumerate(parents):
        if parent >= 0:
            children[parent].append(idx)
    return children


def _choose_body_template(sides: set[str], rng: random.Random) -> np.ndarray:
    candidates = [template["mask"] for template in BODY_TEMPLATES if sides <= template["supports"]]
    assert candidates, "no body template supports chosen connector sides"
    return np.array(rng.choice(candidates), copy=True)


def _paint_body(arr: np.ndarray, mask: np.ndarray, rng: random.Random) -> None:
    colors = list(rng.sample(BODY_COLORS, 2 if rng.random() < 0.6 else 1))
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if len(colors) == 1:
        for rr, cc in cells:
            arr[rr, cc] = colors[0]
        return
    primary, secondary = colors
    secondary_count = max(1, len(cells) // 3)
    secondary_cells = set(rng.sample(cells, secondary_count))
    for rr, cc in cells:
        arr[rr, cc] = secondary if (rr, cc) in secondary_cells else primary


def _build_sprite(
    *,
    is_anchor: bool,
    angle: int,
    connector_count: int,
    rng: random.Random,
    forced_sides: tuple[str, ...] | None = None,
) -> dict:
    side_set = set(forced_sides) if forced_sides is not None else set(rng.choice(_allowed_side_sets(connector_count)))
    body_mask = _choose_body_template(side_set, rng)

    full = np.zeros((8, 8), dtype=int)
    if is_anchor:
        full[4:7, 0:3] = YELLOW

    br0, bc0 = BODY_ZONE_ORIGIN
    for rr, cc in np.argwhere(body_mask):
        full[br0 + int(rr), bc0 + int(cc)] = 1
    _paint_body(full, body_mask, rng)

    for side in side_set:
        rr, cc = SIDE_TO_CONNECTOR[side]
        full[rr, cc] = RED

    center = (1, 4)
    dr, dc = DIR_TO_STEP[angle]
    full[center[0] + dr, center[1] + dc] = PINK
    full[center] = BLACK

    crop, r0, c0 = _tight_crop_with_offset(full)
    side_to_red = {
        side: (SIDE_TO_CONNECTOR[side][0] - r0, SIDE_TO_CONNECTOR[side][1] - c0)
        for side in side_set
    }
    return {
        "input_crop": crop,
        "side_set": tuple(sorted(side_set)),
        "angle": int(angle),
        "rotated": _rotate_clockwise(crop, angle),
        "side_to_red": side_to_red,
    }


def _side_before_clockwise_rotation(output_side: str, angle: int) -> str:
    sides = ["top", "right", "bottom", "left"]
    return sides[(sides.index(output_side) - int(angle)) % 4]


def _global_reds(piece: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    return [(top + int(r), left + int(c)) for r, c in np.argwhere(piece == RED)]


def _delta_to_side(dr: int, dc: int) -> str:
    for side, (sr, sc) in SIDE_STEPS.items():
        if (dr, dc) == (sr, sc):
            return side
    raise ValueError(f"invalid side delta {(dr, dc)}")


def _ordered_side_tuple(sides: set[str]) -> tuple[str, ...]:
    return tuple(side for side in ("top", "right", "bottom", "left") if side in sides)


def _bbox_gap_ok(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _rotate_point_clockwise(
    shape: tuple[int, int],
    coord: tuple[int, int],
    steps: int,
) -> tuple[int, int]:
    r, c = coord
    h, w = shape
    steps %= 4
    if steps == 0:
        return int(r), int(c)
    if steps == 1:
        return int(c), int(h - 1 - r)
    if steps == 2:
        return int(h - 1 - r), int(w - 1 - c)
    return int(w - 1 - c), int(r)


def _build_oriented_sprite(
    *,
    is_anchor: bool,
    input_angle: int,
    output_sides: tuple[str, ...],
    rng: random.Random,
) -> dict:
    input_sides = tuple(
        _side_before_clockwise_rotation(side, input_angle) for side in output_sides
    )
    base = _build_sprite(
        is_anchor=is_anchor,
        angle=input_angle,
        connector_count=len(input_sides),
        rng=rng,
        forced_sides=input_sides,
    )
    input_crop = base["input_crop"]
    rotated = base["rotated"]
    assert _find_indicator_angle(input_crop) == input_angle
    side_to_red = {}
    for output_side in output_sides:
        input_side = _side_before_clockwise_rotation(output_side, input_angle)
        side_to_red[output_side] = _rotate_point_clockwise(
            input_crop.shape,
            base["side_to_red"][input_side],
            input_angle,
        )
    return {
        "input_crop": input_crop,
        "rotated": rotated.copy(),
        "reds": _red_coords(rotated),
        "side_to_red": side_to_red,
        "angle": int(input_angle),
        "is_anchor": bool(is_anchor),
    }


def _choose_embedded_tree(
    piece_count: int,
    rng: random.Random,
) -> tuple[list[int], dict[int, tuple[int, int]]] | None:
    parents = [-1] + [0] * (piece_count - 1)
    positions: dict[int, tuple[int, int]] = {0: (0, 0)}
    degrees = [0] * piece_count

    for idx in range(1, piece_count):
        candidates: list[tuple[int, str, tuple[int, int]]] = []
        occupied = set(positions.values())
        for parent, (pr, pc) in positions.items():
            if degrees[parent] >= 3:
                continue
            used_sides = set()
            for other, (orow, ocol) in positions.items():
                if other == parent:
                    continue
                dr = orow - pr
                dc = ocol - pc
                if abs(dr) + abs(dc) == 1:
                    used_sides.add(_delta_to_side(dr, dc))
            for side, (dr, dc) in SIDE_STEPS.items():
                if side in used_sides:
                    continue
                pos = (pr + dr, pc + dc)
                if pos in occupied:
                    continue
                if any(
                    abs(pos[0] - rr) + abs(pos[1] - cc) == 1 and (rr, cc) != (pr, pc)
                    for rr, cc in occupied
                ):
                    continue
                candidates.append((parent, side, pos))
        if not candidates:
            return None
        rng.shuffle(candidates)
        parent, _side, pos = candidates[0]
        parents[idx] = parent
        positions[idx] = pos
        degrees[parent] += 1
        degrees[idx] = 1

    return parents, positions


def _tree_connector_data(
    parents: list[int],
    positions: dict[int, tuple[int, int]],
) -> tuple[list[tuple[str, ...]], dict[int, tuple[str, str]]]:
    side_sets = [set() for _ in parents]
    edge_sides: dict[int, tuple[str, str]] = {}
    for child in range(1, len(parents)):
        parent = parents[child]
        dr = positions[child][0] - positions[parent][0]
        dc = positions[child][1] - positions[parent][1]
        parent_side = _delta_to_side(dr, dc)
        child_side = OPPOSITE_SIDE[parent_side]
        side_sets[parent].add(parent_side)
        side_sets[child].add(child_side)
        edge_sides[child] = (parent_side, child_side)
    return [_ordered_side_tuple(sides) for sides in side_sets], edge_sides


def _assemble_tree_output(
    sprites: list[dict],
    parents: list[int],
    edge_sides: dict[int, tuple[str, str]],
    grid_shape: tuple[int, int],
    rng: random.Random,
) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    rel_positions = [(0, 0) for _ in sprites]
    for idx in range(1, len(sprites)):
        parent = parents[idx]
        parent_side, child_side = edge_sides[idx]
        pr, pc = sprites[parent]["side_to_red"][parent_side]
        cr, cc = sprites[idx]["side_to_red"][child_side]
        parent_top, parent_left = rel_positions[parent]
        rel_positions[idx] = (parent_top + pr - cr, parent_left + pc - cc)

    min_top = min(top for top, _left in rel_positions)
    min_left = min(left for _top, left in rel_positions)
    max_bottom = max(top + sprites[idx]["rotated"].shape[0] - 1 for idx, (top, _left) in enumerate(rel_positions))
    max_right = max(left + sprites[idx]["rotated"].shape[1] - 1 for idx, (_top, left) in enumerate(rel_positions))

    margin = 2
    shift_top_min = margin - min_top
    shift_left_min = margin - min_left
    shift_top_max = grid_shape[0] - margin - 1 - max_bottom
    shift_left_max = grid_shape[1] - margin - 1 - max_right
    if shift_top_min > shift_top_max or shift_left_min > shift_left_max:
        return None

    shift_top = rng.randint(shift_top_min, shift_top_max)
    shift_left = rng.randint(shift_left_min, shift_left_max)
    positions = [(top + shift_top, left + shift_left) for top, left in rel_positions]

    canvas = np.zeros(grid_shape, dtype=int)
    red_counts: dict[tuple[int, int], int] = {}
    for idx, (top, left) in enumerate(positions):
        piece = sprites[idx]["rotated"]
        require_overlap = idx != 0
        placed = _overlay_with_counts(canvas, piece, top, left, red_counts, require_overlap=require_overlap)
        if placed is None:
            return None
        canvas, overlaps = placed
        if idx == 0:
            if overlaps:
                return None
        else:
            parent = parents[idx]
            parent_side, _child_side = edge_sides[idx]
            pr, pc = sprites[parent]["side_to_red"][parent_side]
            expected = (positions[parent][0] + pr, positions[parent][1] + pc)
            if overlaps != [expected]:
                return None
        for rr, cc in sprites[idx]["reds"]:
            coord = (top + rr, left + cc)
            red_counts[coord] = red_counts.get(coord, 0) + 1
            if red_counts[coord] > 2:
                return None

    if not red_counts or not all(count == 2 for count in red_counts.values()):
        return None
    return canvas, positions


def _place_assembled_output(
    sprites: list[dict],
    parents: list[int],
    grid_shape: tuple[int, int],
    anchor_top: int,
    anchor_left: int,
    rng: random.Random,
) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    h, w = grid_shape
    canvas = np.zeros((h, w), dtype=int)
    placed_meta: dict[int, dict] = {}

    anchor_piece = sprites[0]["rotated"]
    if anchor_top < 0 or anchor_left < 0 or anchor_top + anchor_piece.shape[0] > h or anchor_left + anchor_piece.shape[1] > w:
        return None
    anchor_res = _overlay_with_counts(canvas, anchor_piece, anchor_top, anchor_left, {}, require_overlap=False)
    if anchor_res is None:
        return None
    canvas, _ = anchor_res
    anchor_reds = set(_global_reds(anchor_piece, anchor_top, anchor_left))
    placed_meta[0] = {
        "top": anchor_top,
        "left": anchor_left,
        "free_reds": anchor_reds,
    }

    order = list(range(1, len(sprites)))

    def rec(
        idx: int,
        cur_canvas: np.ndarray,
        cur_meta: dict[int, dict],
        cur_positions: list[tuple[int, int]],
    ) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
        if idx == len(order):
            if any(value["free_reds"] for value in cur_meta.values()):
                return None
            return cur_canvas, list(cur_positions)
        node = order[idx]
        parent = parents[node]
        piece = sprites[node]["rotated"]
        red_locals = _red_coords(piece)
        targets = list(cur_meta[parent]["free_reds"])
        rng.shuffle(targets)
        rng.shuffle(red_locals)
        candidates: list[tuple[int, int, tuple[int, int]]] = []
        for target in targets:
            for rr, cc in red_locals:
                top = target[0] - rr
                left = target[1] - cc
                placed = _overlay_with_counts(cur_canvas, piece, top, left, {coord: 1 for coord in targets}, require_overlap=True)
                if placed is None:
                    continue
                _next_canvas, overlaps = placed
                if overlaps != [target]:
                    continue
                candidates.append((top, left, target))
        rng.shuffle(candidates)
        candidates.sort(key=lambda item: (abs(item[0] - anchor_top) + abs(item[1] - anchor_left), item[0], item[1]))

        for top, left, target in candidates:
            placed = _overlay_with_counts(cur_canvas, piece, top, left, {target: 1}, require_overlap=True)
            if placed is None:
                continue
            next_canvas, overlaps = placed
            if overlaps != [target]:
                continue
            next_meta = {key: {"top": value["top"], "left": value["left"], "free_reds": set(value["free_reds"])} for key, value in cur_meta.items()}
            next_meta[parent]["free_reds"].remove(target)
            child_reds = set(_global_reds(piece, top, left))
            child_reds.remove(target)
            next_meta[node] = {"top": top, "left": left, "free_reds": child_reds}
            next_positions = list(cur_positions)
            next_positions[node] = (top, left)
            result = rec(idx + 1, next_canvas, next_meta, next_positions)
            if result is not None:
                return result
        return None

    start_positions = [(-1, -1) for _ in sprites]
    start_positions[0] = (anchor_top, anchor_left)
    return rec(0, canvas, placed_meta, start_positions)


def _scatter_input(
    sprites: list[dict],
    assembled_positions: list[tuple[int, int]],
    grid_shape: tuple[int, int],
    rng: random.Random,
) -> np.ndarray | None:
    h, w = grid_shape
    grid = np.zeros((h, w), dtype=int)

    anchor = sprites[0]["input_crop"]
    atop, aleft = assembled_positions[0]
    anchor_res = _overlay_with_counts(grid, anchor, atop, aleft, {}, require_overlap=False)
    if anchor_res is None:
        return None
    grid, _ = anchor_res
    placed_boxes = [(atop, aleft, atop + anchor.shape[0] - 1, aleft + anchor.shape[1] - 1)]

    order = sorted(
        range(1, len(sprites)),
        key=lambda idx: sprites[idx]["input_crop"].shape[0] * sprites[idx]["input_crop"].shape[1],
        reverse=True,
    )

    def rec(
        cur_grid: np.ndarray,
        cur_boxes: list[tuple[int, int, int, int]],
        pos: int,
    ) -> np.ndarray | None:
        if pos == len(order):
            return cur_grid

        idx = order[pos]
        piece = sprites[idx]["input_crop"]
        ph, pw = piece.shape
        out_top, out_left = assembled_positions[idx]
        candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
        for top in range(h - ph + 1):
            for left in range(w - pw + 1):
                if abs(top - out_top) + abs(left - out_left) < 6:
                    continue
                box = (top, left, top + ph - 1, left + pw - 1)
                if not all(_bbox_gap_ok(box, old_box, gap=3) for old_box in cur_boxes):
                    continue
                if _overlay_with_counts(cur_grid, piece, top, left, {}, require_overlap=False) is None:
                    continue
                candidates.append((top, left, box))
        if not candidates:
            return None

        rng.shuffle(candidates)
        candidates.sort(
            key=lambda item: (
                -(abs(item[0] - out_top) + abs(item[1] - out_left)),
                -(abs(item[0] - atop) + abs(item[1] - aleft)),
            )
        )
        candidates = candidates[:40]

        for top, left, box in candidates:
            placed = _overlay_with_counts(cur_grid, piece, top, left, {}, require_overlap=False)
            if placed is None:
                continue
            next_grid, _ = placed
            result = rec(next_grid, cur_boxes + [box], pos + 1)
            if result is not None:
                return result
        return None

    return rec(grid, placed_boxes, 0)


def _main_without_rotation(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    sprites = _extract_sprites(g)
    pieces = []
    for sprite in sprites:
        rotated = sprite["crop"] if not sprite["is_anchor"] else sprite["crop"]
        pieces.append(
            {
                "bbox": sprite["bbox"],
                "is_anchor": sprite["is_anchor"],
                "rotated": rotated,
                "reds": _red_coords(rotated),
            }
        )
    return _assemble_from_rotated_pieces(pieces, g.shape)


def _random_chain_sprite(
    *,
    is_anchor: bool,
    connector_count: int,
    rng: random.Random,
) -> dict:
    angle = 0 if is_anchor else rng.choice((1, 2, 3))
    base = _build_sprite(
        is_anchor=is_anchor,
        angle=angle,
        connector_count=connector_count,
        rng=rng,
    )
    return {
        "input_crop": base["input_crop"],
        "rotated": base["rotated"],
        "reds": _red_coords(base["rotated"]),
        "angle": int(angle),
    }


def _extend_chain_once(
    *,
    canvas: np.ndarray,
    red_counts: dict[tuple[int, int], int],
    connector_count: int,
    rng: random.Random,
) -> tuple[dict, int, int, np.ndarray, dict[tuple[int, int], int]] | None:
    target_unpaired = 0 if connector_count == 1 else 1
    unpaired = [coord for coord, count in red_counts.items() if count == 1]
    if len(unpaired) != 1:
        return None
    target = unpaired[0]

    for _ in range(180):
        piece = _random_chain_sprite(is_anchor=False, connector_count=connector_count, rng=rng)
        candidates: list[tuple[int, int, np.ndarray, dict[tuple[int, int], int]]] = []
        for rr, cc in piece["reds"]:
            top = target[0] - rr
            left = target[1] - cc
            placed = _overlay_with_counts(canvas, piece["rotated"], top, left, red_counts, require_overlap=True)
            if placed is None:
                continue
            next_canvas, overlaps = placed
            if overlaps != [target]:
                continue
            next_counts = dict(red_counts)
            ok = True
            for pr, pc in piece["reds"]:
                coord = (top + pr, left + pc)
                next_counts[coord] = next_counts.get(coord, 0) + 1
                if next_counts[coord] > 2:
                    ok = False
                    break
            if not ok:
                continue
            if sum(1 for count in next_counts.values() if count == 1) != target_unpaired:
                continue
            candidates.append((top, left, next_canvas, next_counts))
        if candidates:
            rng.shuffle(candidates)
            top, left, next_canvas, next_counts = candidates[0]
            return piece, top, left, next_canvas, next_counts
    return None


def _make_instance(rng: random.Random) -> np.ndarray | None:
    piece_count = 4
    grid_side = rng.randint(24, 28)
    grid_shape = (grid_side, grid_side)

    sprites: list[dict] = [_random_chain_sprite(is_anchor=True, connector_count=1, rng=rng)]
    anchor_piece = sprites[0]["rotated"]
    anchor_top = rng.randint(6, grid_shape[0] - anchor_piece.shape[0] - 6)
    anchor_left = rng.randint(6, grid_shape[1] - anchor_piece.shape[1] - 6)

    output_grid = np.zeros(grid_shape, dtype=int)
    anchor_res = _overlay_with_counts(output_grid, anchor_piece, anchor_top, anchor_left, {}, require_overlap=False)
    if anchor_res is None:
        return None
    output_grid, _ = anchor_res

    red_counts = {(anchor_top + rr, anchor_left + cc): 1 for rr, cc in sprites[0]["reds"]}
    assembled_positions = [(anchor_top, anchor_left)]

    for idx in range(1, piece_count):
        connector_count = 1 if idx == piece_count - 1 else 2
        extension = _extend_chain_once(
            canvas=output_grid,
            red_counts=red_counts,
            connector_count=connector_count,
            rng=rng,
        )
        if extension is None:
            return None
        piece, top, left, output_grid, red_counts = extension
        sprites.append(piece)
        assembled_positions.append((top, left))

    if not red_counts or not all(count == 2 for count in red_counts.values()):
        return None

    input_grid = _scatter_input(sprites, assembled_positions, grid_shape, rng)
    if input_grid is None:
        return None

    parsed = _extract_sprites(input_grid)
    if len(parsed) != piece_count:
        return None

    solved = main(input_grid)
    if not np.array_equal(solved, output_grid):
        return None

    try:
        shortcut = _main_without_rotation(input_grid)
    except AssertionError:
        shortcut = None
    if shortcut is not None and np.array_equal(shortcut, output_grid):
        return None

    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        seed = random.randrange(1 << 30)
        rng = random.Random(seed)
        try:
            candidate = _make_instance(rng)
        except AssertionError:
            candidate = None
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid clock-rotation connector-assembly instance")
