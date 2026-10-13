from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


_TOOLBOX_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_TOOLBOX_DIR) not in sys.path:
    sys.path.append(str(_TOOLBOX_DIR))

from arc_utils import bbox_coords, crop_bbox, find_components, get_bg, paste


ROTATION_STEPS = (1, -1)
SPRITE_SIZES = (4, 5)
MAX_GEN_ATTEMPTS = 400


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "sprite masks must be non-empty"
    return crop_bbox(mask.astype(int), bbox_coords(coords)).astype(bool)


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(mask, dtype=np.uint8)
    return arr.shape, arr.tobytes()


def _canonical_mask(mask: np.ndarray) -> np.ndarray:
    variants = [_normalize_mask(np.rot90(mask, k)) for k in range(4)]
    unique = {_mask_signature(variant) for variant in variants}
    assert len(unique) == 4, "sprite must have four distinct quarter-turn rotations"
    return min(variants, key=_mask_signature)


def _orientation_index(mask: np.ndarray, canonical: np.ndarray) -> int | None:
    matches = [k for k in range(4) if np.array_equal(mask, np.rot90(canonical, k))]
    if len(matches) != 1:
        return None
    return int(matches[0])


def _fits(grid_shape: tuple[int, int], top_left: tuple[int, int], patch_shape: tuple[int, int]) -> bool:
    h, w = map(int, patch_shape)
    r0, c0 = map(int, top_left)
    H, W = map(int, grid_shape)
    return 0 <= r0 and 0 <= c0 and r0 + h <= H and c0 + w <= W


def _indices_for_step(
    positions: list[tuple[int, int]],
    step: tuple[int, int],
) -> tuple[dict[tuple[int, int], int], tuple[int, int]] | None:
    dr, dc = map(int, step)
    if dr == 0 and dc == 0:
        return None

    ref = positions[0]
    index_by_pos: dict[tuple[int, int], int] = {}
    for pos in positions:
        rr = int(pos[0] - ref[0])
        cc = int(pos[1] - ref[1])
        if dr == 0:
            if rr != 0 or cc % dc != 0:
                return None
            t = cc // dc
        elif dc == 0:
            if cc != 0 or rr % dr != 0:
                return None
            t = rr // dr
        else:
            if rr % dr != 0 or cc % dc != 0:
                return None
            tr = rr // dr
            tc = cc // dc
            if tr != tc:
                return None
            t = tr

        if pos in index_by_pos or t in index_by_pos.values():
            return None
        index_by_pos[pos] = int(t)
    return index_by_pos, ref


def _solution_key(frames: list[tuple[int, int, int]]) -> tuple[tuple[int, int, int], ...]:
    return tuple(sorted((int(r), int(c), int(o)) for r, c, o in frames))


def _enumerate_solutions(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, min_area=1)
    if len(components) < 3:
        return []

    colors = {int(comp["color"]) for comp in components}
    if len(colors) != 1:
        return []
    color = next(iter(colors))

    box_shapes = set()
    positions: list[tuple[int, int]] = []
    masks: list[np.ndarray] = []
    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        patch = crop_bbox(g, comp["bbox"])
        mask = patch == color
        if not np.any(mask):
            return []
        box_shapes.add(tuple(mask.shape))
        positions.append((int(r0), int(c0)))
        masks.append(mask)

    if len(box_shapes) != 1:
        return []
    patch_shape = next(iter(box_shapes))
    if patch_shape[0] != patch_shape[1]:
        return []

    canonical = _canonical_mask(masks[0])
    orientations: dict[tuple[int, int], int] = {}
    for pos, mask in zip(positions, masks):
        ori = _orientation_index(mask, canonical)
        if ori is None:
            return []
        orientations[pos] = ori

    candidate_steps: set[tuple[int, int]] = set()
    visible = len(positions)
    for i, a in enumerate(positions):
        for j, b in enumerate(positions):
            if i == j:
                continue
            diff_r = int(b[0] - a[0])
            diff_c = int(b[1] - a[1])
            for scale in range(1, visible + 1):
                if diff_r % scale == 0 and diff_c % scale == 0:
                    step = (diff_r // scale, diff_c // scale)
                    if step != (0, 0):
                        candidate_steps.add(step)

    explanations: dict[tuple[tuple[int, int, int], ...], dict] = {}
    for step in candidate_steps:
        indexed = _indices_for_step(positions, step)
        if indexed is None:
            continue
        index_by_pos, ref = indexed
        present = set(index_by_pos.values())
        t_min = min(present)
        t_max = max(present)

        for rot_step in ROTATION_STEPS:
            phase = None
            consistent = True
            for pos, idx in index_by_pos.items():
                implied = (orientations[pos] - rot_step * idx) % 4
                if phase is None:
                    phase = implied
                elif phase != implied:
                    consistent = False
                    break
            if not consistent or phase is None:
                continue

            missing_indices: list[int] = []
            if t_max - t_min == visible:
                missing_inside = [t for t in range(t_min, t_max + 1) if t not in present]
                if len(missing_inside) == 1:
                    missing_indices = missing_inside
            elif t_max - t_min == visible - 1:
                endpoint_candidates = []
                for miss_idx in (t_min - 1, t_max + 1):
                    pos = (ref[0] + miss_idx * step[0], ref[1] + miss_idx * step[1])
                    if _fits(g.shape, pos, patch_shape):
                        endpoint_candidates.append(int(miss_idx))
                if len(endpoint_candidates) == 1:
                    missing_indices = endpoint_candidates

            for miss_idx in missing_indices:
                miss_pos = (ref[0] + miss_idx * step[0], ref[1] + miss_idx * step[1])
                if not _fits(g.shape, miss_pos, patch_shape):
                    continue

                window = g[
                    miss_pos[0] : miss_pos[0] + patch_shape[0],
                    miss_pos[1] : miss_pos[1] + patch_shape[1],
                ]
                if np.any(window != bg):
                    continue

                t0 = min(t_min, miss_idx)
                t1 = max(t_max, miss_idx)
                frames = [
                    (
                        int(ref[0] + idx * step[0]),
                        int(ref[1] + idx * step[1]),
                        int((phase + rot_step * idx) % 4),
                    )
                    for idx in range(t0, t1 + 1)
                ]
                key = _solution_key(frames)
                explanations[key] = {
                    "bg": int(bg),
                    "color": int(color),
                    "canonical": canonical,
                    "patch_shape": patch_shape,
                    "missing_pos": (int(miss_pos[0]), int(miss_pos[1])),
                    "missing_orientation": int((phase + rot_step * miss_idx) % 4),
                    "step": (int(step[0]), int(step[1])),
                    "frames": frames,
                }

    return list(explanations.values())


def _render_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(color)
    return patch


def _unique_solution(grid: np.ndarray) -> dict:
    solutions = _enumerate_solutions(grid)
    assert len(solutions) == 1, f"expected a unique trajectory explanation, found {len(solutions)}"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the single erased snapshot in a straight-line sequence of a rotating moving sprite.

    Input structure:
    - The background is the dominant color and all foreground cells belong to several disconnected snapshots of one monochrome sprite.
    - Every snapshot is the same non-symmetric sprite seen in one of its four quarter-turn orientations, and all snapshot bounding boxes have the same square shape.
    - The snapshot positions lie on one arithmetic progression of bbox top-left coordinates, but exactly one frame of that progression is missing entirely.

    Rule:
    - Recover the common translation step from the visible snapshot positions and the fixed quarter-turn rotation step from their orientations.
    - Determine whether the missing frame is an internal gap or a unique in-bounds endpoint extension consistent with the grid boundary.
    - Paste the sprite back at that missing position with the required orientation, leaving every existing pixel unchanged.

    Key insight:
    - The clues are two synchronized arithmetic sequences: snapshot locations progress by one constant translation vector, while orientations progress by one constant signed quarter-turn per step.
    """
    g = np.asarray(grid, dtype=int)
    solution = _unique_solution(g)

    out = g.copy()
    rotated = np.rot90(solution["canonical"], solution["missing_orientation"])
    patch = _render_patch(rotated, solution["color"], solution["bg"])
    paste(out, patch, solution["missing_pos"][0], solution["missing_pos"][1], transparent=solution["bg"])
    return out


def _neighbors4(r: int, c: int, size: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < size and 0 <= cc < size:
            out.append((rr, cc))
    return out


def _random_connected_mask(size: int) -> np.ndarray | None:
    area = random.randint(size + 3, size * size - 4)
    cells = {(
        random.randrange(size),
        random.randrange(size),
    )}
    while len(cells) < area:
        frontier = {
            nbr
            for r, c in cells
            for nbr in _neighbors4(r, c, size)
            if nbr not in cells
        }
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))

    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    if not ({0, size - 1} <= rows and {0, size - 1} <= cols):
        return None

    mask = np.zeros((size, size), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    if len(find_components(mask.astype(int), bg=0, colors=[1], connectivity=4)) != 1:
        return None
    if len({_mask_signature(np.rot90(mask, k)) for k in range(4)}) != 4:
        return None
    return mask


def _generate_sprite() -> np.ndarray:
    for _ in range(200):
        size = random.choice(SPRITE_SIZES)
        mask = _random_connected_mask(size)
        if mask is not None:
            return mask
    raise RuntimeError("failed to generate a non-symmetric sprite")


def _step_choices(size: int) -> list[tuple[int, int]]:
    major = size + 1
    minor_values = (-1, 1)
    steps: set[tuple[int, int]] = set()

    for sign in (-1, 1):
        steps.add((0, sign * major))
        steps.add((sign * major, 0))
    for sr in (-1, 1):
        for sc in (-1, 1):
            steps.add((sr * major, sc * major))
            for minor in minor_values:
                steps.add((sr * major, sc * minor))
                steps.add((sr * minor, sc * major))
    return list(steps)


def _sequence_positions(
    size: int,
    step: tuple[int, int],
    frame_count: int,
) -> tuple[tuple[int, int], list[tuple[int, int]], tuple[int, int]] | None:
    dr, dc = map(int, step)
    span_h = size + (frame_count - 1) * abs(dr)
    span_w = size + (frame_count - 1) * abs(dc)
    if span_h > 30 or span_w > 30:
        return None

    before_r = random.randint(0, max(abs(dr) - 1, 2 if dr == 0 else 0))
    after_r = random.randint(0, max(abs(dr) - 1, 2 if dr == 0 else 0))
    before_c = random.randint(0, max(abs(dc) - 1, 2 if dc == 0 else 0))
    after_c = random.randint(0, max(abs(dc) - 1, 2 if dc == 0 else 0))

    H = span_h + before_r + after_r
    W = span_w + before_c + after_c
    if H > 30 or W > 30:
        return None

    rel_rows = [i * dr for i in range(frame_count)]
    rel_cols = [i * dc for i in range(frame_count)]
    anchor_r = before_r - min(rel_rows)
    anchor_c = before_c - min(rel_cols)
    positions = [(anchor_r + i * dr, anchor_c + i * dc) for i in range(frame_count)]
    return (H, W), positions, (before_r, before_c)


def _maximal_at_boundaries(
    grid_shape: tuple[int, int],
    positions: list[tuple[int, int]],
    step: tuple[int, int],
    size: int,
) -> bool:
    prev_pos = (positions[0][0] - step[0], positions[0][1] - step[1])
    next_pos = (positions[-1][0] + step[0], positions[-1][1] + step[1])
    return not _fits(grid_shape, prev_pos, (size, size)) and not _fits(grid_shape, next_pos, (size, size))


def _render_case(
    grid_shape: tuple[int, int],
    sprite: np.ndarray,
    color: int,
    bg: int,
    positions: list[tuple[int, int]],
    orientations: list[int],
    missing_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(grid_shape, int(bg), dtype=int)
    out = np.full(grid_shape, int(bg), dtype=int)

    for i, (pos, ori) in enumerate(zip(positions, orientations)):
        patch = _render_patch(np.rot90(sprite, ori), color, bg)
        paste(out, patch, pos[0], pos[1], transparent=bg)
        if i != missing_index:
            paste(inp, patch, pos[0], pos[1], transparent=bg)
    return inp, out


def _case_metadata(
    positions: list[tuple[int, int]],
    orientations: list[int],
    missing_index: int,
    step: tuple[int, int],
) -> dict:
    return {
        "missing_index": int(missing_index),
        "branch": "endpoint" if missing_index in (0, len(positions) - 1) else "internal",
        "step": (int(step[0]), int(step[1])),
        "positions": tuple((int(r), int(c)) for r, c in positions),
        "orientations": tuple(int(o) for o in orientations),
    }


def _validate_case(inp: np.ndarray, expected: np.ndarray, meta: dict, sprite: np.ndarray, color: int, bg: int) -> bool:
    size = int(sprite.shape[0])
    if not _maximal_at_boundaries(inp.shape, list(meta["positions"]), meta["step"], size):
        return False

    visible = find_components(inp, bg=bg, colors=[color], connectivity=4)
    if len(visible) != len(meta["positions"]) - 1:
        return False

    solutions = _enumerate_solutions(inp)
    if len(solutions) != 1:
        return False
    solution = solutions[0]
    if solution["missing_pos"] != meta["positions"][meta["missing_index"]]:
        return False
    if not np.array_equal(main(inp), expected):
        return False
    return True


def _generate_case() -> tuple[np.ndarray, np.ndarray, dict]:
    sprite = _generate_sprite()
    size = int(sprite.shape[0])
    frame_count = random.randint(4, 5 if size == 5 else 6)
    step = random.choice(_step_choices(size))
    layout = _sequence_positions(size, step, frame_count)
    if layout is None:
        raise RuntimeError("layout did not fit")
    grid_shape, positions, _margins = layout

    if not _maximal_at_boundaries(grid_shape, positions, step, size):
        raise RuntimeError("layout is not boundary-maximal")

    bg = random.randrange(10)
    color_choices = [value for value in range(10) if value != bg]
    color = random.choice(color_choices)

    start_orientation = random.randrange(4)
    rot_step = random.choice(ROTATION_STEPS)
    orientations = [int((start_orientation + rot_step * i) % 4) for i in range(frame_count)]

    if random.random() < 0.5:
        missing_index = random.choice((0, frame_count - 1))
    else:
        missing_index = random.randint(1, frame_count - 2)

    inp, expected = _render_case(grid_shape, sprite, color, bg, positions, orientations, missing_index)
    meta = _case_metadata(positions, orientations, missing_index, step)
    if not _validate_case(inp, expected, meta, sprite, color, bg):
        raise RuntimeError("generated case failed semantic validation")
    return inp, expected, meta


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            inp, _expected, _meta = _generate_case()
            return inp
        except RuntimeError:
            continue
    raise RuntimeError("failed to generate a valid input for puzzle 7")
