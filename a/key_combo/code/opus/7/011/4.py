from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
ORIGINAL = 1
FILL = 8

FULL_H = 5
HALF_W = 3
FULL_W = HALF_W * 2 - 1

NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    mask = g == int(color)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            pts = []
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(pts, dtype=int)
            comps.append(
                {
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _paste_mask(dst: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    rows, cols = np.where(mask)
    dst[rows + int(r0), cols + int(c0)] = int(color)


def _orientation_from_half(mask: np.ndarray) -> str:
    coords = np.argwhere(mask)
    mean_c = float(coords[:, 1].mean())
    mid = (mask.shape[1] - 1) / 2.0
    return "left" if mean_c < mid else "right"


def _complete_half(mask: np.ndarray, orientation: str) -> np.ndarray:
    h, w = mask.shape
    axis = w - 1
    full = np.zeros((h, 2 * w - 1), dtype=bool)

    rows, cols = np.where(mask)
    if orientation == "left":
        full[rows, cols] = True
        full[rows, 2 * axis - cols] = True
    else:
        full_cols = cols + axis
        full[rows, full_cols] = True
        full[rows, 2 * axis - full_cols] = True
    return full


def _slot_from_component(comp: dict, local_mask: np.ndarray, orientation: str) -> tuple[int, int]:
    r0, c0, _r1, c1 = comp["bbox"]
    axis_col = int(c1) if orientation == "left" else int(c0)
    return int(r0), int(axis_col - (local_mask.shape[1] - 1))


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the half-sprites by mirror symmetry, then use the recovered template to fill the empty lattice slots.

    Input structure:
    - A black background contains blue connected sprites placed on a regular row/column lattice.
    - Every visible sprite is only the left half or right half of the same horizontally symmetric figure.
    - Some lattice slots are empty, and no complete figure is shown directly in the input.

    Rule:
    - Detect each blue component, decide whether it is a left half or right half from how its pixels lean inside its own footprint, and mirror-complete it into the full symmetric figure.
    - Among the completed figures, take the common full template and recover the lattice from the figure top rows and axis-aligned columns.
    - Replace every occupied slot with the completed blue template and fill every empty row-column intersection with the same template in teal.

    Key insight:
    - The lattice cannot be completed from the raw input alone because the shared template only becomes visible after each half-sprite is mirrored into a full symmetric figure.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    comps = _components_of_color(g, ORIGINAL)
    assert comps, "expected blue half-sprites"

    instances = []
    template_votes: Counter = Counter()
    template_by_key: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}

    for comp in comps:
        local = _crop_bbox(g == ORIGINAL, comp["bbox"])
        orientation = _orientation_from_half(local)
        full_mask = _complete_half(local, orientation)
        slot = _slot_from_component(comp, local, orientation)
        key = _mask_key(full_mask)
        template_votes[key] += 1
        template_by_key[key] = full_mask
        instances.append({"slot": slot, "mask": full_mask, "key": key})

    template_key, _count = template_votes.most_common(1)[0]
    template = template_by_key[template_key]
    th, tw = template.shape

    slots = [tuple(map(int, item["slot"])) for item in instances if item["key"] == template_key]
    rows = sorted({r for r, _c in slots})
    cols = sorted({c for _r, c in slots})
    assert len(rows) >= 2 and len(cols) >= 2

    occupied = set(slots)
    out = np.zeros_like(g)
    for r in rows:
        for c in cols:
            color = ORIGINAL if (r, c) in occupied else FILL
            _paste_mask(out, template, r, c, color)
    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = {tuple(map(int, coords[0]))}
    stack = [tuple(map(int, coords[0]))]
    mask_set = {tuple(map(int, pt)) for pt in coords}
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nxt = (r + dr, c + dc)
            if nxt in mask_set and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(mask_set)


def _make_left_half() -> np.ndarray | None:
    for _ in range(400):
        mask = np.zeros((FULL_H, HALF_W), dtype=bool)
        size = random.randint(7, 10)
        seed = (random.randrange(FULL_H), random.randrange(HALF_W))
        active = {seed}
        frontier = {seed}
        while len(active) < size and frontier:
            r, c = random.choice(tuple(frontier))
            nbrs = []
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < FULL_H and 0 <= nc < HALF_W and (nr, nc) not in active:
                    nbrs.append((nr, nc))
            if not nbrs:
                frontier.remove((r, c))
                continue
            nxt = random.choice(nbrs)
            active.add(nxt)
            frontier.add(nxt)

        for r, c in active:
            mask[r, c] = True

        if not mask[0].any() or not mask[-1].any():
            continue
        if not mask[:, 0].any() or np.count_nonzero(mask[:, -1]) < 2:
            continue
        if not _is_connected(mask):
            continue

        coords = np.argwhere(mask)
        if _bbox_coords(coords) != (0, 0, FULL_H - 1, HALF_W - 1):
            continue
        mean_c = float(coords[:, 1].mean())
        if not mean_c < 0.95:
            continue

        full = _complete_half(mask, "left")
        if not _is_connected(full):
            continue
        if _bbox_coords(np.argwhere(full)) != (0, 0, FULL_H - 1, FULL_W - 1):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        return mask
    return None


def _choose_slots(n_rows: int, n_cols: int) -> set[tuple[int, int]] | None:
    total = n_rows * n_cols
    min_occ = max(4, int(np.ceil(total * 0.60)))
    max_occ = min(total - 2, int(np.floor(total * 0.75)))
    counts = [k for k in range(min_occ, max_occ + 1) if total - k >= 2]
    if not counts:
        return None

    all_slots = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    for _ in range(300):
        n_occ = random.choice(counts)
        occupied = set(random.sample(all_slots, n_occ))
        if all(any((r, c) in occupied for c in range(n_cols)) for r in range(n_rows)) and all(
            any((r, c) in occupied for r in range(n_rows)) for c in range(n_cols)
        ):
            return occupied
    return None


def _render_instance(
    left_half: np.ndarray,
    occupied: set[tuple[int, int]],
    orientations: dict[tuple[int, int], str],
    row_starts: list[int],
    col_starts: list[int],
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    full = _complete_half(left_half, "left")
    right_half = np.fliplr(left_half)

    inp = np.zeros((height, width), dtype=int)
    out = np.zeros((height, width), dtype=int)
    for r_idx, r0 in enumerate(row_starts):
        for c_idx, c0 in enumerate(col_starts):
            out_color = ORIGINAL if (r_idx, c_idx) in occupied else FILL
            _paste_mask(out, full, r0, c0, out_color)
            if (r_idx, c_idx) not in occupied:
                continue
            patch = left_half if orientations[(r_idx, c_idx)] == "left" else right_half
            offset_c = c0 if orientations[(r_idx, c_idx)] == "left" else c0 + (FULL_W - HALF_W)
            _paste_mask(inp, patch, r0, offset_c, ORIGINAL)
    return inp, out


def _validate_instance(inp: np.ndarray, expected: np.ndarray) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False

    comps = _components_of_color(inp, ORIGINAL)
    if len(comps) < 4:
        return False
    if any(_crop_bbox(inp == ORIGINAL, comp["bbox"]).shape != (FULL_H, HALF_W) for comp in comps):
        return False

    orientations = []
    completed = []
    slots = []
    for comp in comps:
        local = _crop_bbox(inp == ORIGINAL, comp["bbox"])
        orient = _orientation_from_half(local)
        orientations.append(orient)
        if orient == "left":
            if not float(np.argwhere(local)[:, 1].mean()) < 0.95:
                return False
        else:
            if not float(np.argwhere(local)[:, 1].mean()) > 1.05:
                return False
        completed.append(_complete_half(local, orient))
        slots.append(_slot_from_component(comp, local, orient))

    if orientations.count("left") < 2 or orientations.count("right") < 2:
        return False
    if len({_mask_key(mask) for mask in completed}) != 1:
        return False

    rows = sorted({r for r, _c in slots})
    cols = sorted({c for _r, c in slots})
    if len(rows) not in {3, 4} or len(cols) != 3:
        return False
    if len(rows) != len(set(rows)) or len(cols) != len(set(cols)):
        return False
    if len(rows) >= 3 and len(set(np.diff(rows))) != 1:
        return False
    if len(cols) >= 2 and len(set(np.diff(cols))) != 1:
        return False

    total = len(rows) * len(cols)
    occupied = len(slots)
    if total - occupied < 2:
        return False
    ratio = occupied / total
    if not (0.60 <= ratio <= 0.75):
        return False

    if np.array_equal(inp, expected):
        return False
    if not np.array_equal(main(inp), expected):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        left_half = _make_left_half()
        if left_half is None:
            continue

        n_rows = random.choice([3, 4])
        n_cols = 3
        occupied = _choose_slots(n_rows, n_cols)
        if occupied is None:
            continue

        occ_list = list(occupied)
        random.shuffle(occ_list)
        left_count = random.randint(2, len(occ_list) - 2)
        orientations = {}
        for idx, slot in enumerate(occ_list):
            orientations[slot] = "left" if idx < left_count else "right"
        if list(orientations.values()).count("left") < 2 or list(orientations.values()).count("right") < 2:
            continue

        step = random.choice([6, 7])
        top = random.randint(0, 30 - (FULL_H + (n_rows - 1) * step))
        left = random.randint(0, 30 - (FULL_W + (n_cols - 1) * step))
        height = top + FULL_H + (n_rows - 1) * step + random.randint(0, 30 - (top + FULL_H + (n_rows - 1) * step))
        width = left + FULL_W + (n_cols - 1) * step + random.randint(0, 30 - (left + FULL_W + (n_cols - 1) * step))

        row_starts = [top + i * step for i in range(n_rows)]
        col_starts = [left + i * step for i in range(n_cols)]
        inp, out = _render_instance(left_half, occupied, orientations, row_starts, col_starts, height, width)
        if _validate_instance(inp, out):
            return inp

    raise RuntimeError("failed to generate a valid puzzle-4 input")
