from __future__ import annotations

import itertools
import random

import numpy as np


UNKNOWN = -1
WORKSPACE_SIDE = 30
MASTER_SIZE = 3
MAX_GENERATION_ATTEMPTS = 400
FRAME_COUNT_CHOICES = (4, 5)


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def rotate_coord(r: int, c: int, n: int, k: int) -> tuple[int, int]:
    k %= 4
    if k == 0:
        return int(r), int(c)
    if k == 1:
        return int(n - 1 - c), int(r)
    if k == 2:
        return int(n - 1 - r), int(n - 1 - c)
    return int(c), int(n - 1 - r)


def unrotate_coord(r: int, c: int, n: int, k: int) -> tuple[int, int]:
    return rotate_coord(int(r), int(c), int(n), (-int(k)) % 4)


def _serialize_grid(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(grid, dtype=int)
    return tuple(map(int, arr.shape)), arr.astype(np.int16, copy=False).tobytes()


def _assign_or_conflict(table: np.ndarray, r: int, c: int, color: int) -> bool:
    cur = int(table[r, c])
    if cur != UNKNOWN and cur != int(color):
        return False
    table[r, c] = int(color)
    return True


def _extract_frame(
    grid: np.ndarray,
    bg: int,
    frame_color: int,
    coords: np.ndarray,
) -> dict | None:
    r0, c0, r1, c1 = bbox_coords(coords)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = sub == int(frame_color)
    h, w = mask.shape
    if h != w:
        return None

    row_lines = [r for r in range(h) if bool(mask[r, :].all())]
    col_lines = [c for c in range(w) if bool(mask[:, c].all())]
    if len(row_lines) < 3 or len(row_lines) != len(col_lines):
        return None
    if row_lines[0] != 0 or row_lines[-1] != h - 1 or col_lines[0] != 0 or col_lines[-1] != w - 1:
        return None
    if any(a + 1 >= b for a, b in zip(row_lines, row_lines[1:])):
        return None
    if any(a + 1 >= b for a, b in zip(col_lines, col_lines[1:])):
        return None

    expected = np.zeros_like(mask, dtype=bool)
    expected[row_lines, :] = True
    expected[:, col_lines] = True
    if not np.array_equal(mask, expected):
        return None

    n = len(row_lines) - 1
    if n < 2:
        return None

    observed = np.full((n, n), UNKNOWN, dtype=int)
    clues: list[tuple[int, int, int]] = []
    for sr in range(n):
        rr0, rr1 = row_lines[sr] + 1, row_lines[sr + 1]
        for sc in range(n):
            cc0, cc1 = col_lines[sc] + 1, col_lines[sc + 1]
            region = sub[rr0:rr1, cc0:cc1]
            vals = sorted(set(map(int, region.ravel())) - {int(bg), int(frame_color)})
            if len(vals) > 1:
                return None
            if vals:
                color = int(vals[0])
                if not np.all((region == int(bg)) | (region == int(color))):
                    return None
                observed[sr, sc] = color
                clues.append((int(sr), int(sc), int(color)))

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "row_lines": [int(r0) + r for r in row_lines],
        "col_lines": [int(c0) + c for c in col_lines],
        "n": int(n),
        "frame_pixels": int(mask.sum()),
        "observed": observed,
        "clues": clues,
    }


def _parse_frames(grid: np.ndarray) -> tuple[int, int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    colors = [int(color) for color in np.unique(g) if int(color) != int(bg)]
    candidates: list[tuple[int, int, int, list[dict]]] = []

    for color in colors:
        comps = components_mask(g == int(color))
        if len(comps) < 2:
            continue
        frames: list[dict] = []
        ok = True
        for comp in comps:
            frame = _extract_frame(g, bg, int(color), comp)
            if frame is None:
                ok = False
                break
            frames.append(frame)
        if not ok:
            continue
        ns = {frame["n"] for frame in frames}
        if len(ns) != 1:
            continue
        total_frame_pixels = sum(int(frame["frame_pixels"]) for frame in frames)
        candidates.append((len(frames), total_frame_pixels, int(color), frames))

    assert candidates, "expected lattice frames"
    candidates.sort(key=lambda item: (int(item[0]), int(item[1]), -int(item[2])), reverse=True)
    _, _, frame_color, frames = candidates[0]
    frames = sorted(frames, key=lambda frame: (frame["bbox"][0], frame["bbox"][1]))
    n = int(frames[0]["n"])
    assert all(int(frame["n"]) == n for frame in frames)
    return int(bg), int(frame_color), frames


def _materialize_output(grid: np.ndarray, frames: list[dict], master: np.ndarray, rotations: tuple[int, ...]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    n = int(master.shape[0])
    for frame, rot in zip(frames, rotations, strict=True):
        filled = np.rot90(master, int(rot))
        rows = frame["row_lines"]
        cols = frame["col_lines"]
        for sr in range(n):
            for sc in range(n):
                out[rows[sr] + 1 : rows[sr + 1], cols[sc] + 1 : cols[sc + 1]] = int(filled[sr, sc])
    return out


def _unique_completed_outputs(grid: np.ndarray, frames: list[dict]) -> list[np.ndarray]:
    n = int(frames[0]["n"])
    master_seed = np.full((n, n), UNKNOWN, dtype=int)
    for r, c, color in frames[0]["clues"]:
        assert _assign_or_conflict(master_seed, int(r), int(c), int(color))

    unique: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for rest in itertools.product(range(4), repeat=max(0, len(frames) - 1)):
        rotations = (0, *rest)
        master = master_seed.copy()
        ok = True
        for frame_index in range(1, len(frames)):
            rot = int(rotations[frame_index])
            for lr, lc, color in frames[frame_index]["clues"]:
                mr, mc = unrotate_coord(int(lr), int(lc), n, rot)
                if not _assign_or_conflict(master, mr, mc, int(color)):
                    ok = False
                    break
            if not ok:
                break
        if not ok or np.any(master == UNKNOWN):
            continue
        out = _materialize_output(grid, frames, master, rotations)
        unique[_serialize_grid(out)] = out
        if len(unique) > 1:
            break
    return list(unique.values())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cross-check partial rotated lattice tables to recover the hidden master table and fill every blank slot.

    Input structure:
    - The background contains several separated square lattice frames drawn in one shared frame color.
    - Each frame is split into the same N x N set of slots; some slot interiors are colored and the others stay as background.
    - All visible slot colors come from one hidden N x N master table, but each frame shows that table under its own quarter-turn rotation.

    Rule:
    - Treat the first frame as the canonical reference orientation, and test every 0/90/180/270 rotation choice for the other frames.
    - For any candidate rotation assignment, map every visible slot back into master-table coordinates; reject assignments with color conflicts or missing master slots.
    - The input is valid only when these constraints imply one unique completed output grid; repaint every slot interior in every frame from that recovered table while leaving frame lines and outside background unchanged.

    Key insight:
    - Matching slots by absolute position is a trap: only by undoing each frame's own rotation can the partial clues from different frames agree on one shared master table.
    """
    g = np.asarray(grid, dtype=int)
    _, _, frames = _parse_frames(g)
    outputs = _unique_completed_outputs(g, frames)
    assert len(outputs) == 1, "input must imply a unique completed output"
    return outputs[0]


def _sample_palette() -> tuple[int, int, list[int]]:
    colors = random.sample(list(range(10)), 10)
    bg = int(colors[0])
    frame_color = int(colors[1])
    return bg, frame_color, [int(color) for color in colors[2:]]


def _sample_master(slot_pool: list[int]) -> np.ndarray | None:
    distinct = random.randint(5, len(slot_pool))
    chosen = random.sample(slot_pool, distinct)
    vals = list(chosen)
    while len(vals) < MASTER_SIZE * MASTER_SIZE:
        vals.append(int(random.choice(chosen)))
    random.shuffle(vals)
    master = np.array(vals, dtype=int).reshape(MASTER_SIZE, MASTER_SIZE)
    if len(set(map(int, master.ravel()))) < 4:
        return None
    if any(np.array_equal(master, np.rot90(master, k)) for k in range(1, 4)):
        return None
    return master


def _sample_orientations(frame_count: int) -> list[int]:
    if frame_count == 4:
        base = random.sample(range(4), 3)
        rots = base + [int(random.choice(base))]
    else:
        rots = [0, 1, 2, 3, int(random.randrange(4))]
    random.shuffle(rots)
    return [int(rot) for rot in rots]


def _sample_cell_sizes(frame_count: int) -> list[int]:
    sizes = [1] * frame_count
    upgrades = random.randint(0, min(2, frame_count - 2))
    for idx in random.sample(range(frame_count), upgrades):
        sizes[int(idx)] = 2
    return [int(size) for size in sizes]


def _sample_revealed_master_slots(frame_count: int) -> list[set[tuple[int, int]]] | None:
    all_slots = [(r, c) for r in range(MASTER_SIZE) for c in range(MASTER_SIZE)]
    min_clues = 3
    max_clues = 5 if frame_count == 5 else 6
    targets = [random.randint(min_clues, max_clues) for _ in range(frame_count)]
    if sum(targets) < len(all_slots) + frame_count:
        return None

    revealed = [set() for _ in range(frame_count)]
    shuffled_slots = all_slots[:]
    random.shuffle(shuffled_slots)

    for slot in shuffled_slots:
        eligible = [i for i in range(frame_count) if len(revealed[i]) < targets[i]]
        if not eligible:
            return None
        random.shuffle(eligible)
        eligible.sort(key=lambda idx: (len(revealed[idx]), targets[idx] - len(revealed[idx])))
        pick_count = 2 if len(eligible) > 1 and random.random() < 0.35 else 1
        chosen = eligible[:pick_count]
        for idx in chosen:
            revealed[idx].add(slot)

    frame_order = list(range(frame_count))
    random.shuffle(frame_order)
    for idx in frame_order:
        while len(revealed[idx]) < targets[idx]:
            extras = [slot for slot in all_slots if slot not in revealed[idx]]
            if not extras:
                return None
            overlap_first = [slot for slot in extras if any(slot in revealed[j] for j in range(frame_count) if j != idx)]
            if overlap_first and random.random() < 0.75:
                revealed[idx].add(random.choice(overlap_first))
            else:
                revealed[idx].add(random.choice(extras))

    return revealed


def _render_frame_patch(local_values: np.ndarray, cell_size: int, frame_color: int, bg: int, mask: np.ndarray | None) -> np.ndarray:
    n = int(local_values.shape[0])
    step = int(cell_size) + 1
    size = n * int(cell_size) + n + 1
    lines = [i * step for i in range(n + 1)]
    patch = np.full((size, size), int(bg), dtype=int)
    patch[lines, :] = int(frame_color)
    patch[:, lines] = int(frame_color)

    for r in range(n):
        for c in range(n):
            if mask is not None and not bool(mask[r, c]):
                continue
            patch[lines[r] + 1 : lines[r + 1], lines[c] + 1 : lines[c + 1]] = int(local_values[r, c])
    return patch


def _place_patch_rectangles(sizes: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    occ = np.zeros((WORKSPACE_SIDE, WORKSPACE_SIDE), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(sizes)
    order = sorted(range(len(sizes)), key=lambda idx: sizes[idx][0] * sizes[idx][1], reverse=True)

    for idx in order:
        h, w = sizes[idx]
        r_max = WORKSPACE_SIDE - h - 1
        c_max = WORKSPACE_SIDE - w - 1
        if r_max < 1 or c_max < 1:
            return None
        candidates: list[tuple[int, int]] = []
        for r in range(1, r_max + 1):
            for c in range(1, c_max + 1):
                rr0 = max(0, r - 1)
                rr1 = min(WORKSPACE_SIDE, r + h + 1)
                cc0 = max(0, c - 1)
                cc1 = min(WORKSPACE_SIDE, c + w + 1)
                if not bool(occ[rr0:rr1, cc0:cc1].any()):
                    candidates.append((int(r), int(c)))
        if not candidates:
            return None
        positions[idx] = random.choice(candidates)
        r0, c0 = positions[idx]
        occ[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1] = True

    return [pos for pos in positions if pos is not None]


def _crop_workspace(inp: np.ndarray, out: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(inp != int(bg))
    r0, c0, r1, c1 = bbox_coords(coords)

    def _margin(low: int, high: int) -> int:
        if high <= 0:
            return 0
        return random.randint(low, high)

    top = _margin(0 if r0 == 0 else 1, min(3, r0))
    left = _margin(0 if c0 == 0 else 1, min(3, c0))
    bottom = _margin(0 if r1 == WORKSPACE_SIDE - 1 else 1, min(3, WORKSPACE_SIDE - 1 - r1))
    right = _margin(0 if c1 == WORKSPACE_SIDE - 1 else 1, min(3, WORKSPACE_SIDE - 1 - c1))

    sl_r0 = int(r0 - top)
    sl_c0 = int(c0 - left)
    sl_r1 = int(r1 + bottom + 1)
    sl_c1 = int(c1 + right + 1)
    return inp[sl_r0:sl_r1, sl_c0:sl_c1].copy(), out[sl_r0:sl_r1, sl_c0:sl_c1].copy()


def _same_position_conflict(frames: list[dict]) -> bool:
    n = int(frames[0]["n"])
    for r in range(n):
        for c in range(n):
            colors = {
                int(color)
                for frame in frames
                for rr, cc, color in frame["clues"]
                if int(rr) == r and int(cc) == c
            }
            if len(colors) > 1:
                return True
    return False


def _validate_instance(
    inp: np.ndarray,
    expected_out: np.ndarray,
    orientations: list[int],
    revealed_slots: list[set[tuple[int, int]]],
) -> bool:
    if np.array_equal(inp, expected_out):
        return False

    all_slots = {(r, c) for r in range(MASTER_SIZE) for c in range(MASTER_SIZE)}
    if set().union(*revealed_slots) != all_slots:
        return False
    if len(set(map(int, orientations))) < 3:
        return False
    if len(set(map(int, orientations))) == len(orientations):
        return False

    for slots in revealed_slots:
        if not (2 <= len(slots) < MASTER_SIZE * MASTER_SIZE):
            return False
        others = set().union(*(revealed_slots[j] for j in range(len(revealed_slots)) if revealed_slots[j] is not slots))
        if not (slots & others):
            return False

    grouped: dict[int, set[tuple[int, int]]] = {}
    counts: dict[int, int] = {}
    for rot, slots in zip(orientations, revealed_slots, strict=True):
        grouped.setdefault(int(rot), set()).update(slots)
        counts[int(rot)] = counts.get(int(rot), 0) + 1
    for rot, merged in grouped.items():
        if counts[int(rot)] >= 2 and len(merged) >= MASTER_SIZE * MASTER_SIZE:
            return False

    try:
        _, _, frames = _parse_frames(inp)
    except AssertionError:
        return False
    if len(frames) != len(revealed_slots) or any(int(frame["n"]) != MASTER_SIZE for frame in frames):
        return False
    if not _same_position_conflict(frames):
        return False

    outputs = _unique_completed_outputs(inp, frames)
    return len(outputs) == 1 and np.array_equal(outputs[0], expected_out)


def _generate_attempt() -> np.ndarray | None:
    bg, frame_color, slot_pool = _sample_palette()
    master = _sample_master(slot_pool)
    if master is None:
        return None

    frame_count = int(random.choice(FRAME_COUNT_CHOICES))
    orientations = _sample_orientations(frame_count)
    cell_sizes = _sample_cell_sizes(frame_count)
    revealed_slots = _sample_revealed_master_slots(frame_count)
    if revealed_slots is None:
        return None

    input_patches: list[np.ndarray] = []
    output_patches: list[np.ndarray] = []
    for rot, cell_size, frame_slots in zip(orientations, cell_sizes, revealed_slots, strict=True):
        local_full = np.rot90(master, int(rot))
        local_mask = np.zeros((MASTER_SIZE, MASTER_SIZE), dtype=bool)
        for mr, mc in frame_slots:
            lr, lc = rotate_coord(int(mr), int(mc), MASTER_SIZE, int(rot))
            local_mask[lr, lc] = True
        input_patches.append(
            _render_frame_patch(local_full, int(cell_size), int(frame_color), int(bg), local_mask)
        )
        output_patches.append(
            _render_frame_patch(local_full, int(cell_size), int(frame_color), int(bg), None)
        )

    placements = _place_patch_rectangles([tuple(map(int, patch.shape)) for patch in input_patches])
    if placements is None:
        return None

    inp = np.full((WORKSPACE_SIDE, WORKSPACE_SIDE), int(bg), dtype=int)
    out = np.full_like(inp, int(bg))
    for patch_in, patch_out, (r0, c0) in zip(input_patches, output_patches, placements, strict=True):
        h, w = patch_in.shape
        inp[r0 : r0 + h, c0 : c0 + w] = patch_in
        out[r0 : r0 + h, c0 : c0 + w] = patch_out

    inp, expected_out = _crop_workspace(inp, out, int(bg))
    if not _validate_instance(inp, expected_out, orientations, revealed_slots):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
