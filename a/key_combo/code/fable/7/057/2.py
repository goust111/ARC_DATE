from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, get_bg, in_bounds, paste


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR_TO_SIDE = {v: k for k, v in SIDE_TO_DIR.items()}
WALL_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.GREY, Color.ORANGE, Color.TEAL, Color.MAROON]

MIN_CHAMBER_SIDE = 4
MAX_CHAMBER_SIDE = 5
MIN_LINK_GAP = 2
MAX_LINK_GAP = 4
MIN_OUTER_MARGIN = 1
MAX_OUTER_MARGIN = 3
GENERATION_RETRIES = 500


@dataclass(frozen=True)
class Opening:
    pos: tuple[int, int]
    direction: tuple[int, int]
    side: str


@dataclass(frozen=True)
class Chamber:
    bbox: tuple[int, int, int, int]
    openings: tuple[Opening, Opening]
    seed: bool

    @property
    def interior(self) -> tuple[int, int, int, int]:
        r0, c0, r1, c1 = self.bbox
        return (r0 + 1, c0 + 1, r1 - 1, c1 - 1)


@dataclass
class ChamberSpec:
    h: int
    w: int
    openings: tuple[tuple[str, int], tuple[str, int]]
    r0: int = 0
    c0: int = 0
    seed: bool = False

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.r0, self.c0, self.r0 + self.h - 1, self.c0 + self.w - 1)


def _opposite(direction: tuple[int, int]) -> tuple[int, int]:
    return (-direction[0], -direction[1])


def _opening_coord(spec: ChamberSpec, side: str, offset: int) -> tuple[int, int]:
    if side == "top":
        return (spec.r0, spec.c0 + offset)
    if side == "bottom":
        return (spec.r0 + spec.h - 1, spec.c0 + offset)
    if side == "left":
        return (spec.r0 + offset, spec.c0)
    if side == "right":
        return (spec.r0 + offset, spec.c0 + spec.w - 1)
    raise ValueError(f"unknown side {side}")


def _side_index(spec: ChamberSpec, side: str) -> int:
    for i, (curr_side, _offset) in enumerate(spec.openings):
        if curr_side == side:
            return i
    raise ValueError(f"side {side} not present")


def _sample_offset(side: str, h: int, w: int) -> int:
    if side in {"top", "bottom"}:
        return random.randint(1, w - 2)
    return random.randint(1, h - 2)


def _make_spec(sides: tuple[str, str], *, seed: bool = False) -> ChamberSpec:
    h = random.randint(MIN_CHAMBER_SIDE, MAX_CHAMBER_SIDE)
    w = random.randint(MIN_CHAMBER_SIDE, MAX_CHAMBER_SIDE)
    openings = tuple((side, _sample_offset(side, h, w)) for side in sides)
    return ChamberSpec(h=h, w=w, openings=openings, seed=seed)


def _render_spec(spec: ChamberSpec, wall_color: int, bg: int) -> np.ndarray:
    patch = np.full((spec.h, spec.w), bg, dtype=int)
    patch[0, :] = wall_color
    patch[-1, :] = wall_color
    patch[:, 0] = wall_color
    patch[:, -1] = wall_color
    for side, offset in spec.openings:
        rr, cc = _opening_coord(ChamberSpec(spec.h, spec.w, spec.openings), side, offset)
        patch[rr, cc] = bg
    return patch


def _wall_color(grid: np.ndarray, bg: int) -> int:
    colors = sorted(set(map(int, np.unique(grid))) - {int(bg)})
    wall_colors = [color for color in colors if color != Color.YELLOW]
    assert len(wall_colors) == 1, "expected exactly one wall color"
    return int(wall_colors[0])


def _extract_chambers(grid: np.ndarray) -> tuple[list[Chamber], int, int, int]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    wall = _wall_color(g, bg)
    H, W = g.shape
    wall_mask = (g == wall)
    candidates: list[Chamber] = []

    for r0 in range(H - (MIN_CHAMBER_SIDE - 1)):
        top_cols = np.where(wall_mask[r0])[0]
        for i, c0 in enumerate(top_cols):
            for c1 in top_cols[i + 1 :]:
                if c1 - c0 + 1 < MIN_CHAMBER_SIDE:
                    continue
                if not wall_mask[r0, c0] or not wall_mask[r0, c1]:
                    continue
                for r1 in range(r0 + MIN_CHAMBER_SIDE - 1, H):
                    if not (wall_mask[r1, c0] and wall_mask[r1, c1]):
                        continue

                    openings: list[Opening] = []
                    valid = True
                    for c in range(c0 + 1, c1):
                        top_val = int(g[r0, c])
                        bot_val = int(g[r1, c])
                        if top_val == wall:
                            pass
                        elif top_val == bg:
                            openings.append(Opening((r0, c), SIDE_TO_DIR["top"], "top"))
                        else:
                            valid = False
                            break
                        if bot_val == wall:
                            pass
                        elif bot_val == bg:
                            openings.append(Opening((r1, c), SIDE_TO_DIR["bottom"], "bottom"))
                        else:
                            valid = False
                            break
                    if not valid:
                        continue
                    for r in range(r0 + 1, r1):
                        left_val = int(g[r, c0])
                        right_val = int(g[r, c1])
                        if left_val == wall:
                            pass
                        elif left_val == bg:
                            openings.append(Opening((r, c0), SIDE_TO_DIR["left"], "left"))
                        else:
                            valid = False
                            break
                        if right_val == wall:
                            pass
                        elif right_val == bg:
                            openings.append(Opening((r, c1), SIDE_TO_DIR["right"], "right"))
                        else:
                            valid = False
                            break
                    if not valid or len(openings) != 2:
                        continue

                    interior = g[r0 + 1 : r1, c0 + 1 : c1]
                    if interior.size == 0 or np.any(interior == wall):
                        continue
                    if not np.all((interior == bg) | (interior == Color.YELLOW)):
                        continue

                    seed = bool((interior == Color.YELLOW).any())
                    if seed and not np.all(interior == Color.YELLOW):
                        continue
                    if not seed and not np.all(interior == bg):
                        continue
                    candidates.append(Chamber(bbox=(r0, c0, r1, c1), openings=(openings[0], openings[1]), seed=seed))

    assert candidates, "expected at least one rectangular chamber candidate"
    candidates.sort(
        key=lambda chamber: (
            -((chamber.bbox[2] - chamber.bbox[0] + 1) * (chamber.bbox[3] - chamber.bbox[1] + 1)),
            chamber.bbox,
        )
    )

    selected: list[Chamber] = []
    for chamber in candidates:
        r0, c0, r1, c1 = chamber.bbox
        overlaps = False
        for other in selected:
            or0, oc0, or1, oc1 = other.bbox
            if not (r1 < or0 or or1 < r0 or c1 < oc0 or oc1 < c0):
                overlaps = True
                break
        if not overlaps:
            selected.append(chamber)

    assert selected, "expected parsed chambers"
    coverage = np.zeros_like(g, dtype=int)
    for chamber in selected:
        r0, c0, r1, c1 = chamber.bbox
        coverage[r0, c0 : c1 + 1] += 1
        coverage[r1, c0 : c1 + 1] += 1
        coverage[r0 : r1 + 1, c0] += 1
        coverage[r0 : r1 + 1, c1] += 1
        for opening in chamber.openings:
            coverage[opening.pos] -= 1
    assert np.array_equal(coverage > 0, wall_mask), "selected rectangles must explain all wall pixels"

    seed_indices = [idx for idx, chamber in enumerate(selected) if chamber.seed]
    assert len(seed_indices) == 1, "expected exactly one seed chamber"
    selected.sort(key=lambda chamber: chamber.bbox)
    seed_indices = [idx for idx, chamber in enumerate(selected) if chamber.seed]
    return selected, wall, bg, seed_indices[0]


def _fill_interior(out: np.ndarray, chamber: Chamber, color: int) -> None:
    r0, c0, r1, c1 = chamber.interior
    out[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _opening_map(chambers: list[Chamber]) -> dict[tuple[int, int], tuple[int, int]]:
    mapping: dict[tuple[int, int], tuple[int, int]] = {}
    for ch_idx, chamber in enumerate(chambers):
        for op_idx, opening in enumerate(chamber.openings):
            assert opening.pos not in mapping, "openings must be unique grid cells"
            mapping[opening.pos] = (ch_idx, op_idx)
    return mapping


def _trace_target(
    grid: np.ndarray,
    wall: int,
    chambers: list[Chamber],
    mapping: dict[tuple[int, int], tuple[int, int]],
    chamber_idx: int,
    opening_idx: int,
) -> tuple[int, int] | None:
    r, c = chambers[chamber_idx].openings[opening_idx].pos
    dr, dc = chambers[chamber_idx].openings[opening_idx].direction
    while True:
        nr, nc = r + dr, c + dc
        if not in_bounds(grid.shape, nr, nc):
            return None
        if int(grid[nr, nc]) == wall:
            return None
        r, c = nr, nc
        hit = mapping.get((r, c))
        if hit is None or hit[0] == chamber_idx:
            continue
        target = chambers[hit[0]].openings[hit[1]]
        if (dr, dc) == _opposite(target.direction):
            return hit


def _cast_ray(
    out: np.ndarray,
    grid: np.ndarray,
    wall: int,
    chambers: list[Chamber],
    mapping: dict[tuple[int, int], tuple[int, int]],
    chamber_idx: int,
    opening_idx: int,
) -> tuple[int, int] | None:
    r, c = chambers[chamber_idx].openings[opening_idx].pos
    dr, dc = chambers[chamber_idx].openings[opening_idx].direction
    out[r, c] = Color.YELLOW
    while True:
        nr, nc = r + dr, c + dc
        if not in_bounds(grid.shape, nr, nc):
            return None
        if int(grid[nr, nc]) == wall:
            return None
        r, c = nr, nc
        out[r, c] = Color.YELLOW
        hit = mapping.get((r, c))
        if hit is None or hit[0] == chamber_idx:
            continue
        target = chambers[hit[0]].openings[hit[1]]
        if (dr, dc) == _opposite(target.direction):
            return hit


def _propagate(
    grid: np.ndarray,
    chambers: list[Chamber],
    wall: int,
    seed_idx: int,
    *,
    all_aligned: bool = False,
    keep_direction: bool = False,
) -> tuple[np.ndarray, set[int], dict[int, dict[str, int | tuple[int, int]]]]:
    out = np.asarray(grid, dtype=int).copy()
    mapping = _opening_map(chambers)
    lit: set[int] = {seed_idx}
    entry_info: dict[int, dict[str, int | tuple[int, int]]] = {}
    _fill_interior(out, chambers[seed_idx], Color.YELLOW)

    queue = deque((seed_idx, op_idx, None) for op_idx in range(2))
    if all_aligned:
        for chamber_idx in range(len(chambers)):
            if chamber_idx == seed_idx:
                continue
            targets = [_trace_target(grid, wall, chambers, mapping, chamber_idx, op_idx) for op_idx in range(2)]
            if any(target is not None for target in targets):
                lit.add(chamber_idx)
                _fill_interior(out, chambers[chamber_idx], Color.YELLOW)
                queue.append((chamber_idx, 0, None))
                queue.append((chamber_idx, 1, None))

    while queue:
        chamber_idx, opening_idx, incoming_dir = queue.popleft()
        hit = _cast_ray(out, grid, wall, chambers, mapping, chamber_idx, opening_idx)
        if hit is None:
            continue

        target_chamber, target_opening = hit
        if target_chamber in lit:
            continue

        lit.add(target_chamber)
        _fill_interior(out, chambers[target_chamber], Color.YELLOW)
        next_idx = 1 - target_opening
        if keep_direction and incoming_dir is not None:
            matching = [
                idx
                for idx, opening in enumerate(chambers[target_chamber].openings)
                if idx != target_opening and opening.direction == incoming_dir
            ]
            if matching:
                next_idx = matching[0]
            else:
                continue

        entry_info[target_chamber] = {
            "entry_idx": target_opening,
            "incoming_dir": chambers[chamber_idx].openings[opening_idx].direction,
        }
        queue.append((target_chamber, next_idx, chambers[chamber_idx].openings[opening_idx].direction))

    return out, lit, entry_info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Light every chamber reachable from the unique yellow seed by following front-facing doorway alignments.

    Input structure:
    - The grid has a black background and several same-colored rectangular outline chambers.
    - Each chamber boundary has exactly two one-cell openings, and exactly one chamber interior is fully filled yellow.
    - Other chamber interiors are empty, and some non-seed-aligned chamber pairs may exist as distractors.

    Rule:
    - Identify each chamber from its rectangular wall outline and detect the outward normal of each opening from its boundary side.
    - The seed chamber emits a yellow ray from each opening; a newly reached chamber fills its interior yellow and then emits only from its other opening.
    - Each ray travels straight through background cells, stops before a wall, reaches the edge if unobstructed, and ignites only when it enters another chamber frontally through an aligned opening.

    Key insight:
    - Reachability is governed by a directed geometry graph on openings: only seed-reachable opening-to-opening alignments propagate the chain, while isolated aligned distractors stay dark.
    """
    chambers, wall, _bg, seed_idx = _extract_chambers(np.asarray(grid, dtype=int))
    out, _lit, _entry = _propagate(np.asarray(grid, dtype=int), chambers, wall, seed_idx)
    return out


def _random_chain_directions() -> tuple[int, int, list[tuple[int, int]]]:
    while True:
        n_chain = 4
        seed_idx = random.randint(1, n_chain - 2)
        dirs = [random.choice(DIRS4)]
        for _ in range(n_chain - 2):
            choices = [d for d in DIRS4 if d != _opposite(dirs[-1])]
            dirs.append(random.choice(choices))
        if not any(dirs[i - 1] != dirs[i] for i in range(1, len(dirs))):
            continue
        if not any(i != seed_idx and dirs[i - 1] != dirs[i] for i in range(1, n_chain - 1)):
            continue
        return n_chain, seed_idx, dirs


def _build_chain_specs() -> list[ChamberSpec]:
    n_chain, seed_idx, dirs = _random_chain_directions()
    specs: list[ChamberSpec] = []
    for i in range(n_chain):
        if i == 0:
            sides = (DIR_TO_SIDE[dirs[0]], DIR_TO_SIDE[_opposite(dirs[0])])
        elif i == n_chain - 1:
            sides = (DIR_TO_SIDE[_opposite(dirs[-1])], DIR_TO_SIDE[dirs[-1]])
        else:
            sides = (DIR_TO_SIDE[_opposite(dirs[i - 1])], DIR_TO_SIDE[dirs[i]])
        specs.append(_make_spec(sides, seed=(i == seed_idx)))

    specs[0].r0 = 0
    specs[0].c0 = 0
    for i, direction in enumerate(dirs):
        curr = specs[i]
        nxt = specs[i + 1]
        curr_side = DIR_TO_SIDE[direction]
        next_side = DIR_TO_SIDE[_opposite(direction)]
        curr_offset = curr.openings[_side_index(curr, curr_side)][1]
        next_offset = nxt.openings[_side_index(nxt, next_side)][1]
        curr_r, curr_c = _opening_coord(curr, curr_side, curr_offset)
        gap = random.randint(MIN_LINK_GAP, MAX_LINK_GAP)
        if direction == (0, 1):
            nxt.r0 = curr_r - next_offset
            nxt.c0 = curr_c + gap + 1
        elif direction == (0, -1):
            nxt.r0 = curr_r - next_offset
            nxt.c0 = curr_c - gap - nxt.w
        elif direction == (1, 0):
            nxt.r0 = curr_r + gap + 1
            nxt.c0 = curr_c - next_offset
        else:
            nxt.r0 = curr_r - gap - nxt.h
            nxt.c0 = curr_c - next_offset
    return specs


def _build_distractor_pair(anchor_bbox: tuple[int, int, int, int]) -> list[ChamberSpec]:
    direction = random.choice(DIRS4)
    first = _make_spec((DIR_TO_SIDE[direction], DIR_TO_SIDE[_opposite(direction)]))
    second = _make_spec((DIR_TO_SIDE[_opposite(direction)], DIR_TO_SIDE[direction]))
    margin = random.randint(2, 4)

    if direction in {(0, 1), (0, -1)}:
        if random.random() < 0.5:
            first.r0 = anchor_bbox[2] + margin
        else:
            first.r0 = anchor_bbox[0] - margin - first.h
        first.c0 = anchor_bbox[1] + random.randint(-1, 1)
    else:
        first.r0 = anchor_bbox[0] + random.randint(-1, 1)
        if random.random() < 0.5:
            first.c0 = anchor_bbox[3] + margin
        else:
            first.c0 = anchor_bbox[1] - margin - first.w

    first_side = DIR_TO_SIDE[direction]
    second_side = DIR_TO_SIDE[_opposite(direction)]
    first_offset = first.openings[_side_index(first, first_side)][1]
    second_offset = second.openings[_side_index(second, second_side)][1]
    first_r, first_c = _opening_coord(first, first_side, first_offset)
    gap = random.randint(MIN_LINK_GAP, MAX_LINK_GAP)
    if direction == (0, 1):
        second.r0 = first_r - second_offset
        second.c0 = first_c + gap + 1
    elif direction == (0, -1):
        second.r0 = first_r - second_offset
        second.c0 = first_c - gap - second.w
    elif direction == (1, 0):
        second.r0 = first_r + gap + 1
        second.c0 = first_c - second_offset
    else:
        second.r0 = first_r - gap - second.h
        second.c0 = first_c - second_offset
    return [first, second]


def _bbox_union(specs: list[ChamberSpec]) -> tuple[int, int, int, int]:
    r0 = min(spec.r0 for spec in specs)
    c0 = min(spec.c0 for spec in specs)
    r1 = max(spec.r0 + spec.h - 1 for spec in specs)
    c1 = max(spec.c0 + spec.w - 1 for spec in specs)
    return (r0, c0, r1, c1)


def _boxes_separate(specs: list[ChamberSpec], padding: int = 1) -> bool:
    for i, a in enumerate(specs):
        ar0, ac0, ar1, ac1 = a.bbox
        for b in specs[i + 1 :]:
            br0, bc0, br1, bc1 = b.bbox
            if not (ar1 + padding < br0 or br1 + padding < ar0 or ac1 + padding < bc0 or bc1 + padding < ac0):
                return False
    return True


def _normalized_specs(specs: list[ChamberSpec]) -> tuple[list[ChamberSpec], tuple[int, int]]:
    r0, c0, r1, c1 = _bbox_union(specs)
    top = random.randint(MIN_OUTER_MARGIN, MAX_OUTER_MARGIN)
    left = random.randint(MIN_OUTER_MARGIN, MAX_OUTER_MARGIN)
    bottom = random.randint(MIN_OUTER_MARGIN, MAX_OUTER_MARGIN)
    right = random.randint(MIN_OUTER_MARGIN, MAX_OUTER_MARGIN)
    height = (r1 - r0 + 1) + top + bottom
    width = (c1 - c0 + 1) + left + right
    if height > 30 or width > 30:
        raise ValueError("grid would exceed ARC bounds")

    out_specs: list[ChamberSpec] = []
    for spec in specs:
        out_specs.append(
            ChamberSpec(
                h=spec.h,
                w=spec.w,
                openings=spec.openings,
                r0=spec.r0 - r0 + top,
                c0=spec.c0 - c0 + left,
                seed=spec.seed,
            )
        )
    return out_specs, (height, width)


def _render_input(specs: list[ChamberSpec], shape: tuple[int, int], wall_color: int) -> np.ndarray:
    grid = np.full(shape, Color.BLACK, dtype=int)
    for spec in specs:
        patch = _render_spec(spec, wall_color, Color.BLACK)
        paste(grid, patch, spec.r0, spec.c0, transparent=Color.BLACK)
    seed_specs = [spec for spec in specs if spec.seed]
    assert len(seed_specs) == 1
    seed = seed_specs[0]
    grid[seed.r0 + 1 : seed.r0 + seed.h - 1, seed.c0 + 1 : seed.c0 + seed.w - 1] = Color.YELLOW
    return grid


def _wrong_fill_all_aligned(grid: np.ndarray) -> np.ndarray:
    chambers, wall, _bg, seed_idx = _extract_chambers(grid)
    out, _lit, _info = _propagate(grid, chambers, wall, seed_idx, all_aligned=True)
    return out


def _wrong_continue_straight(grid: np.ndarray) -> np.ndarray:
    chambers, wall, _bg, seed_idx = _extract_chambers(grid)
    out, _lit, _info = _propagate(grid, chambers, wall, seed_idx, keep_direction=True)
    return out


def _validate_semantics(grid: np.ndarray) -> bool:
    chambers, wall, _bg, seed_idx = _extract_chambers(grid)
    assert len(chambers) >= 6, "need a nontrivial chain and distractors"
    mapping = _opening_map(chambers)
    graph = {
        (ch_idx, op_idx): _trace_target(grid, wall, chambers, mapping, ch_idx, op_idx)
        for ch_idx in range(len(chambers))
        for op_idx in range(2)
    }

    seed_targets = [graph[(seed_idx, 0)], graph[(seed_idx, 1)]]
    assert seed_targets[0] is not None and seed_targets[1] is not None
    assert seed_targets[0][0] != seed_targets[1][0], "seed must launch along two distinct branches"

    true_out, lit, entry_info = _propagate(grid, chambers, wall, seed_idx)
    assert len(lit) >= 4
    assert len(lit) < len(chambers), "some distractor chambers must remain dark"
    assert any(
        target is not None and src_ch not in lit and target[0] not in lit
        for (src_ch, _src_op), target in graph.items()
    ), "need aligned distractor chambers outside the seed component"

    assert any(
        ch_idx != seed_idx and info["incoming_dir"] != chambers[ch_idx].openings[1 - int(info["entry_idx"])].direction
        for ch_idx, info in entry_info.items()
    ), "need at least one reachable turning chamber"

    assert not np.array_equal(true_out, _wrong_fill_all_aligned(grid)), "reachability shortcut must fail"
    assert not np.array_equal(true_out, _wrong_continue_straight(grid)), "turn-handling shortcut must fail"
    assert np.array_equal(main(grid), true_out)
    return True


def _gen_attempt() -> np.ndarray | None:
    try:
        chain = _build_chain_specs()
        distractors = _build_distractor_pair(_bbox_union(chain))
        specs = chain + distractors
        if not _boxes_separate(specs, padding=1):
            return None
        specs, shape = _normalized_specs(specs)
        wall_color = random.choice(WALL_COLORS)
        grid = _render_input(specs, shape, wall_color)
        if _validate_semantics(grid):
            return grid
    except (AssertionError, ValueError):
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid opening-chain ignition puzzle")
