from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, in_bounds, paste


PALETTE = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 500
PLACEMENT_TRIES = 240
BOX_GAP = 1


def _border_values(arr: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> list[int]:
    vals = [int(v) for v in arr[r0, c0 : c1 + 1]]
    if r1 > r0:
        vals.extend(int(v) for v in arr[r1, c0 : c1 + 1])
    if c1 > c0:
        vals.extend(int(arr[r, c0]) for r in range(r0 + 1, r1))
        vals.extend(int(arr[r, c1]) for r in range(r0 + 1, r1))
    return vals


def _parse_nest(arr: np.ndarray, bg: int) -> list[int] | None:
    if arr.size == 0 or np.any(arr == int(bg)):
        return None

    r0, c0 = 0, 0
    r1, c1 = arr.shape[0] - 1, arr.shape[1] - 1
    outer_to_inner: list[int] = []
    while r0 <= r1 and c0 <= c1:
        colors = set(_border_values(arr, r0, c0, r1, c1))
        if len(colors) != 1:
            return None
        color = colors.pop()
        if color == int(bg):
            return None
        outer_to_inner.append(int(color))
        r0 += 1
        c0 += 1
        r1 -= 1
        c1 -= 1
    return outer_to_inner[::-1]


def _render_nest(sequence: list[int], inner_h: int, inner_w: int) -> np.ndarray:
    depth = len(sequence) - 1
    h = int(inner_h + 2 * depth)
    w = int(inner_w + 2 * depth)
    arr = np.empty((h, w), dtype=int)
    outer_to_inner = list(reversed(sequence))
    for offset, color in enumerate(outer_to_inner):
        arr[offset : h - offset, offset : w - offset] = int(color)
    return arr


def _paint_ring(out: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    nr0, nc0 = r0 - 1, c0 - 1
    nr1, nc1 = r1 + 1, c1 + 1
    for c in range(nc0, nc1 + 1):
        if in_bounds(out.shape, nr0, c):
            out[nr0, c] = int(color)
        if in_bounds(out.shape, nr1, c):
            out[nr1, c] = int(color)
    for r in range(nr0 + 1, nr1):
        if in_bounds(out.shape, r, nc0):
            out[r, nc0] = int(color)
        if in_bounds(out.shape, r, nc1):
            out[r, nc1] = int(color)
    return nr0, nc0, nr1, nc1


def _intersect_bbox(
    bbox: tuple[int, int, int, int], h: int, w: int
) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = map(int, bbox)
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(h - 1, r1)
    cc1 = min(w - 1, c1)
    if rr0 > rr1 or cc0 > cc1:
        return None
    return rr0, cc0, rr1, cc1


def _boxes_conflict(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = BOX_GAP
) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _sequence_match_indices(chain: list[int], seq: list[int]) -> list[int]:
    matches = []
    for start in range(len(chain) - len(seq) + 1):
        if chain[start : start + len(seq)] == seq:
            matches.append(start)
    return matches


def _extract_objects(grid: np.ndarray) -> list[dict]:
    bg = int(get_bg(grid))
    objects = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=False):
        sub = crop_bbox(grid, comp["bbox"])
        seq = _parse_nest(sub, bg)
        assert seq is not None, "every foreground component must be a full concentric rectangle nest"
        objects.append({"bbox": tuple(map(int, comp["bbox"])), "seq": seq})
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue each seed concentric rectangle outward by following the teacher nest's adjacent-layer color chain.

    Input structure:
    - The background contains several separate 4-connected multicolor objects, each filling its own bbox as nested axis-aligned rectangular layers.
    - Exactly one object is a complete teacher nest whose outermost two layers have the same color.
    - Every other object is a smaller seed nest whose layer-color sequence is a contiguous subsequence of the teacher sequence.

    Rule:
    - Read the teacher nest from inner layer to outer layer.
    - Treat each adjacent color pair in that sequence as a deterministic current-color -> next-color recurrence.
    - For every seed nest, match its existing layer sequence inside the teacher sequence, then add one-cell-thick outer rectangular rings by repeatedly applying the recurrence from the current outermost color.
    - After adding the final repeated-color ring produced by the unique self-map, stop; clipped rings are allowed at the grid boundary. The teacher nest stays unchanged.

    Key insight:
    - The hidden color-transition table is encoded geometrically by the teacher nest's layer order, and the repeated outer color reveals the only fixed point where growth must terminate.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    objects = _extract_objects(g)

    teacher_candidates = [obj for obj in objects if len(obj["seq"]) >= 2 and obj["seq"][-1] == obj["seq"][-2]]
    assert len(teacher_candidates) == 1, "expected exactly one teacher nest"
    teacher = teacher_candidates[0]
    teacher_seq = list(teacher["seq"])

    transition: dict[int, int] = {}
    for cur, nxt in zip(teacher_seq, teacher_seq[1:]):
        if int(cur) in transition:
            assert transition[int(cur)] == int(nxt), "teacher chain must define a unique recurrence"
        transition[int(cur)] = int(nxt)
    fixed_points = [color for color, nxt in transition.items() if int(color) == int(nxt)]
    assert fixed_points == [teacher_seq[-1]], "teacher chain must end in a unique self-map"

    out = g.copy()
    for obj in objects:
        if obj is teacher:
            continue

        seq = list(obj["seq"])
        matches = _sequence_match_indices(teacher_seq, seq)
        assert len(matches) == 1, "each seed sequence must align uniquely inside the teacher chain"
        start = matches[0]
        end = start + len(seq) - 1

        growth_colors: list[int] = []
        cur = int(seq[-1])
        while True:
            nxt = int(transition[cur])
            growth_colors.append(nxt)
            if nxt == cur:
                break
            cur = nxt
        assert growth_colors == teacher_seq[end + 1 :]

        bbox = tuple(map(int, obj["bbox"]))
        for color in growth_colors:
            bbox = _paint_ring(out, bbox, color)

    return out


def _sample_teacher_sequence(bg: int) -> list[int]:
    layer_count = random.randint(4, 6)
    colors = random.sample([c for c in PALETTE if c != int(bg)], layer_count - 1)
    return [*colors, colors[-1]]


def _sample_seed_specs(teacher_seq: list[int]) -> list[dict] | None:
    layer_count = len(teacher_seq)
    starts = list(range(layer_count - 2))
    seed_count = random.randint(2, min(3, len(starts)))
    chosen_starts = random.sample(starts, seed_count)

    terminal_candidates = [s for s in chosen_starts if s >= layer_count - 4]
    terminal_start = random.choice(terminal_candidates) if terminal_candidates and random.random() < 0.6 else None

    specs: list[dict] = []
    for start in chosen_starts:
        if start == terminal_start:
            end = layer_count - 2
        else:
            end = random.randint(start, min(layer_count - 2, start + 2))

        specs.append(
            {
                "start": int(start),
                "end": int(end),
                "input_seq": list(teacher_seq[start : end + 1]),
                "grow_colors": list(teacher_seq[end + 1 :]),
                "inner_h": random.randint(1, 2),
                "inner_w": random.randint(1, 2),
            }
        )

    if not any(len(spec["grow_colors"]) >= 2 for spec in specs):
        return None

    for spec in specs:
        spec["input_patch"] = _render_nest(spec["input_seq"], spec["inner_h"], spec["inner_w"])
        spec["input_h"], spec["input_w"] = map(int, spec["input_patch"].shape)
        final_len = len(teacher_seq) - int(spec["start"])
        spec["final_h"] = int(spec["inner_h"] + 2 * (final_len - 1))
        spec["final_w"] = int(spec["inner_w"] + 2 * (final_len - 1))
        spec["grow_count"] = len(spec["grow_colors"])

    return specs


def _place_full_bbox(
    box_h: int,
    box_w: int,
    grid_h: int,
    grid_w: int,
    placed: list[tuple[int, int, int, int]],
    *,
    prefer_interior: bool,
) -> tuple[int, int, int, int] | None:
    if box_h > grid_h or box_w > grid_w:
        return None

    r_lo, r_hi = 0, grid_h - box_h
    c_lo, c_hi = 0, grid_w - box_w
    if prefer_interior and grid_h - box_h >= 2:
        r_lo, r_hi = 1, grid_h - box_h - 1
    if prefer_interior and grid_w - box_w >= 2:
        c_lo, c_hi = 1, grid_w - box_w - 1

    if r_hi < r_lo or c_hi < c_lo:
        return None

    for _ in range(PLACEMENT_TRIES):
        r0 = random.randint(r_lo, r_hi)
        c0 = random.randint(c_lo, c_hi)
        bbox = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
        if any(_boxes_conflict(bbox, other) for other in placed):
            continue
        return bbox
    return None


def _place_seed(
    spec: dict,
    grid_h: int,
    grid_w: int,
    placed: list[tuple[int, int, int, int]],
    *,
    clipped: bool,
) -> bool:
    final_h = int(spec["final_h"])
    final_w = int(spec["final_w"])
    grow_count = int(spec["grow_count"])
    input_h = int(spec["input_h"])
    input_w = int(spec["input_w"])
    if final_h > grid_h or final_w > grid_w:
        return False

    for _ in range(PLACEMENT_TRIES):
        if clipped:
            side = random.choice(("top", "bottom", "left", "right"))
            clip_amt = random.randint(1, grow_count)
            if side == "top":
                final_r0 = -clip_amt
                final_c0 = random.randint(0, grid_w - final_w)
            elif side == "bottom":
                final_r0 = grid_h - final_h + clip_amt
                final_c0 = random.randint(0, grid_w - final_w)
            elif side == "left":
                final_r0 = random.randint(0, grid_h - final_h)
                final_c0 = -clip_amt
            else:
                final_r0 = random.randint(0, grid_h - final_h)
                final_c0 = grid_w - final_w + clip_amt
        else:
            final_r0 = random.randint(0, grid_h - final_h)
            final_c0 = random.randint(0, grid_w - final_w)

        final_bbox = (
            int(final_r0),
            int(final_c0),
            int(final_r0 + final_h - 1),
            int(final_c0 + final_w - 1),
        )
        visible_bbox = _intersect_bbox(final_bbox, grid_h, grid_w)
        if visible_bbox is None:
            continue
        if clipped and visible_bbox == final_bbox:
            continue
        if any(_boxes_conflict(visible_bbox, other) for other in placed):
            continue

        input_bbox = (
            int(final_r0 + grow_count),
            int(final_c0 + grow_count),
            int(final_r0 + grow_count + input_h - 1),
            int(final_c0 + grow_count + input_w - 1),
        )
        if not (
            0 <= input_bbox[0] <= input_bbox[2] < grid_h
            and 0 <= input_bbox[1] <= input_bbox[3] < grid_w
        ):
            continue

        spec["final_bbox"] = final_bbox
        spec["visible_bbox"] = visible_bbox
        spec["input_bbox"] = input_bbox
        spec["clipped"] = clipped
        placed.append(visible_bbox)
        return True
    return False


def _construct_expected(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = np.array(inp, copy=True)
    for spec in specs:
        bbox = tuple(map(int, spec["input_bbox"]))
        for color in spec["grow_colors"]:
            bbox = _paint_ring(out, bbox, int(color))
    return out


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    bg: int,
    teacher_seq: list[int],
    seed_specs: list[dict],
    clipped_idx: int | None,
) -> bool:
    if int(get_bg(inp)) != int(bg):
        return False
    if np.array_equal(inp, expected):
        return False

    objects = _extract_objects(inp)
    if len(objects) != len(seed_specs) + 1:
        return False

    teacher_candidates = [obj for obj in objects if len(obj["seq"]) >= 2 and obj["seq"][-1] == obj["seq"][-2]]
    if len(teacher_candidates) != 1 or teacher_candidates[0]["seq"] != teacher_seq:
        return False

    parsed_seeds = sorted(tuple(obj["seq"]) for obj in objects if obj not in teacher_candidates)
    target_seeds = sorted(tuple(spec["input_seq"]) for spec in seed_specs)
    if parsed_seeds != target_seeds:
        return False

    for spec in seed_specs:
        matches = _sequence_match_indices(teacher_seq, spec["input_seq"])
        if len(matches) != 1 or matches[0] != int(spec["start"]):
            return False

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    if clipped_idx is not None:
        delta = actual != inp
        if not (
            np.any(delta[0, :])
            or np.any(delta[-1, :])
            or np.any(delta[:, 0])
            or np.any(delta[:, -1])
        ):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(PALETTE)
    teacher_seq = _sample_teacher_sequence(bg)
    seed_specs = _sample_seed_specs(teacher_seq)
    if seed_specs is None:
        return None

    teacher_patch = _render_nest(teacher_seq, random.randint(1, 3), random.randint(1, 3))
    teacher_h, teacher_w = map(int, teacher_patch.shape)

    min_h = max([teacher_h, *[int(spec["final_h"]) for spec in seed_specs]]) + 2
    min_w = max([teacher_w, *[int(spec["final_w"]) for spec in seed_specs]]) + 2
    if min_h > 30 or min_w > 30:
        return None

    grid_h = random.randint(max(20, min_h), 30)
    grid_w = random.randint(max(20, min_w), 30)

    clipped_idx = random.randrange(len(seed_specs)) if random.random() < 0.45 else None
    placed: list[tuple[int, int, int, int]] = []

    teacher_bbox = _place_full_bbox(teacher_h, teacher_w, grid_h, grid_w, placed, prefer_interior=True)
    if teacher_bbox is None:
        return None
    placed.append(teacher_bbox)

    order = sorted(
        range(len(seed_specs)),
        key=lambda idx: (
            0 if idx == clipped_idx else 1,
            -(int(seed_specs[idx]["final_h"]) * int(seed_specs[idx]["final_w"])),
        ),
    )
    for idx in order:
        if not _place_seed(seed_specs[idx], grid_h, grid_w, placed, clipped=(idx == clipped_idx)):
            return None

    inp = np.full((grid_h, grid_w), int(bg), dtype=int)
    paste(inp, teacher_patch, int(teacher_bbox[0]), int(teacher_bbox[1]))
    for spec in seed_specs:
        r0, c0, _r1, _c1 = map(int, spec["input_bbox"])
        paste(inp, spec["input_patch"], r0, c0)

    expected = _construct_expected(inp, seed_specs)
    if not _semantic_checks(inp, expected, bg, teacher_seq, seed_specs, clipped_idx):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid concentric-chain growth puzzle")
