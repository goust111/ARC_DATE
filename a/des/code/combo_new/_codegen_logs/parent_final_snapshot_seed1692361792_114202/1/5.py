import random
from collections import deque

import numpy as np


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            parts.append(cells)
    return parts


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_component_mask(cells: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(cells)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            touches_border = r in (0, h - 1) or c in (0, w - 1)
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] or seen[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    if nr in (0, h - 1) or nc in (0, w - 1):
                        touches_border = True
                    q.append((int(nr), int(nc)))
            if not touches_border:
                holes += 1
    return holes


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((int(r), int(c)))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((int(r), int(c)))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if mask[nr, nc] or exterior[nr, nc]:
                continue
            exterior[nr, nc] = True
            q.append((int(nr), int(nc)))

    return mask | (~mask & ~exterior)


def _rotate_clockwise(mask: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(mask, -(int(turns) % 4))


def _anchor_and_object(grid: np.ndarray, color: int) -> tuple[tuple[int, int], np.ndarray]:
    comps = _components(grid == int(color))
    anchors = [comp for comp in comps if len(comp) == 1]
    objects = [comp for comp in comps if len(comp) > 1]
    assert len(anchors) == 1, "each color must have one anchor pixel"
    assert len(objects) == 1, "each color must have one nontrivial object"
    return anchors[0][0], _crop_component_mask(objects[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each stamp by its hole count and re-stamp it at the matching anchor pixel.

    Input structure:
    - The background is black.
    - Each non-black color appears in exactly two 4-connected components: one medium-sized stamp object and one isolated single-pixel anchor.
    - Stamp objects may contain 0, 1, 2, or 3 enclosed black holes, and different colors correspond to different stamp-anchor pairs.

    Rule:
    - For each color, separate the isolated anchor from the multi-pixel stamp object.
    - Count the number of enclosed holes inside the stamp, rotate the stamp clockwise by 90 degrees per hole, and fill those holes before drawing.
    - Translate the rotated stamp so that the top-left corner of its bounding box lands on the same-color anchor pixel.
    - Clear the original objects; only the relocated filled stamps remain in the output, and they do not overlap.

    Key insight:
    The stamp's topology, not its color or size, determines the rotation angle, while the lone same-color pixel is the placement anchor for the final filled imprint.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.zeros((h, w), dtype=int)

    for color in np.unique(g):
        color = int(color)
        if color == 0:
            continue
        anchor, obj_mask = _anchor_and_object(g, color)
        hole_count = _count_holes(obj_mask)
        stamp = _rotate_clockwise(_fill_holes(obj_mask), hole_count)
        ar, ac = anchor
        sh, sw = stamp.shape
        assert ar + sh <= h and ac + sw <= w, "stamped object must stay inside the grid"
        target = out[ar : ar + sh, ac : ac + sw]
        assert not np.any((target != 0) & stamp), "stamped objects must not overlap"
        target[stamp] = color
    return out


_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
_OBJECT_COUNT_OPTIONS = [2, 3, 4]
_GRID_MAX = 30
_GEN_ATTEMPTS = 400


def _sample_hole_positions(h: int, w: int, hole_count: int) -> list[tuple[int, int]] | None:
    if hole_count == 0:
        return []
    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    for _ in range(200):
        random.shuffle(interior)
        chosen: list[tuple[int, int]] = []
        for r, c in interior:
            if all(abs(r - rr) + abs(c - cc) != 1 and (r, c) != (rr, cc) for rr, cc in chosen):
                chosen.append((r, c))
                if len(chosen) == hole_count:
                    return chosen
    return None


def _is_connected(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _random_stamp_mask(hole_count: int) -> np.ndarray | None:
    for _ in range(200):
        h = random.randint(4, 7)
        w = random.randint(4, 7)
        holes = _sample_hole_positions(h, w, hole_count)
        if holes is None:
            continue

        mask = np.ones((h, w), dtype=bool)
        for r, c in holes:
            mask[r, c] = False

        if _count_holes(mask) != hole_count:
            continue

        candidates = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if mask[r, c]
            and (r in (0, h - 1) or c in (0, w - 1))
            and (r, c) not in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
        ]
        random.shuffle(candidates)
        notch_goal = random.randint(0, min(5, len(candidates)))
        removed = 0
        for r, c in candidates:
            if removed >= notch_goal:
                break
            trial = mask.copy()
            trial[r, c] = False
            if not (trial[0, 0] and trial[0, -1] and trial[-1, 0] and trial[-1, -1]):
                continue
            if not _is_connected(trial):
                continue
            if _count_holes(trial) != hole_count:
                continue
            mask = trial
            removed += 1

        if _is_connected(mask) and _count_holes(mask) == hole_count:
            return mask
    return None


def _can_place(used: np.ndarray, patch: np.ndarray, top: int, left: int, pad: int = 1) -> bool:
    h, w = used.shape
    ph, pw = patch.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    for dr, dc in np.argwhere(patch):
        r = int(top + dr)
        c = int(left + dc)
        r0 = max(0, r - pad)
        r1 = min(h, r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(w, c + pad + 1)
        if np.any(used[r0:r1, c0:c1]):
            return False
    return True


def _mark_used(used: np.ndarray, patch: np.ndarray, top: int, left: int, pad: int = 1) -> None:
    h, w = used.shape
    for dr, dc in np.argwhere(patch):
        r = int(top + dr)
        c = int(left + dc)
        r0 = max(0, r - pad)
        r1 = min(h, r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(w, c + pad + 1)
        used[r0:r1, c0:c1] = True


def _all_positions(h: int, w: int, ph: int, pw: int) -> list[tuple[int, int]]:
    positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
    random.shuffle(positions)
    return positions


def _paste_color(grid: np.ndarray, patch: np.ndarray, top: int, left: int, color: int) -> None:
    ph, pw = patch.shape
    view = grid[top : top + ph, left : left + pw]
    view[patch] = int(color)


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    n_objects = random.choice(_OBJECT_COUNT_OPTIONS)
    min_side = 16 + 2 * n_objects
    h = random.randint(min_side, _GRID_MAX)
    w = random.randint(min_side, _GRID_MAX)
    colors = random.sample(_COLOR_POOL, n_objects)
    hole_counts = random.sample([0, 1, 2, 3], n_objects)

    records = []
    for color, hole_count in zip(colors, hole_counts, strict=False):
        base = _random_stamp_mask(hole_count)
        if base is None:
            return None
        input_rot = random.randrange(4)
        input_mask = _rotate_clockwise(base, input_rot)
        output_mask = _rotate_clockwise(_fill_holes(input_mask), hole_count)
        assert input_mask[0, 0] and input_mask[0, -1] and input_mask[-1, 0] and input_mask[-1, -1]
        assert output_mask[0, 0], "anchors should be covered by stamped objects"
        records.append(
            {
                "color": int(color),
                "hole_count": int(hole_count),
                "input_mask": input_mask,
                "output_mask": output_mask,
                "input_area": int(np.sum(input_mask)),
                "output_area": int(np.sum(output_mask)),
            }
        )

    used = np.zeros((h, w), dtype=bool)

    for rec in sorted(records, key=lambda item: (-item["output_area"], -item["output_mask"].shape[0] * item["output_mask"].shape[1])):
        patch = rec["output_mask"]
        ph, pw = patch.shape
        placed = False
        for top, left in _all_positions(h, w, ph, pw):
            if _can_place(used, patch, top, left, pad=1):
                rec["anchor"] = (int(top), int(left))
                _mark_used(used, patch, top, left, pad=1)
                placed = True
                break
        if not placed:
            return None

    for rec in sorted(records, key=lambda item: (-item["input_area"], -item["input_mask"].shape[0] * item["input_mask"].shape[1])):
        patch = rec["input_mask"]
        ph, pw = patch.shape
        placed = False
        for top, left in _all_positions(h, w, ph, pw):
            if _can_place(used, patch, top, left, pad=1):
                rec["input_top"] = (int(top), int(left))
                _mark_used(used, patch, top, left, pad=1)
                placed = True
                break
        if not placed:
            return None

    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)
    for rec in records:
        color = rec["color"]
        anchor_r, anchor_c = rec["anchor"]
        input_r, input_c = rec["input_top"]
        input_mask = rec["input_mask"]
        output_mask = rec["output_mask"]

        _paste_color(inp, input_mask, input_r, input_c, color)
        assert inp[anchor_r, anchor_c] == 0, "anchor must stay isolated"
        inp[anchor_r, anchor_c] = color
        _paste_color(out, output_mask, anchor_r, anchor_c, color)

        for rr, cc in np.argwhere(input_mask):
            r = int(input_r + rr)
            c = int(input_c + cc)
            assert abs(r - anchor_r) + abs(c - anchor_c) > 1, "anchor must not touch its object"

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        candidate = _generate_case()
        if candidate is not None:
            return candidate[0]
    raise RuntimeError(f"failed to generate a valid input after {_GEN_ATTEMPTS} attempts")
