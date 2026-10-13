from __future__ import annotations

import random

import numpy as np


_MAX_PERIOD = 6
_GEN_MAX_ATTEMPTS = 400
_GEN_CALL_COUNT = 0
_DISTRACTOR_SHAPES = (
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 0)),
)


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _band_of_col(c: int, b1: int, b2: int) -> int:
    if c < b1:
        return 0
    if c < b2:
        return 1
    return 2


def _find_occluder(grid: np.ndarray) -> dict[str, object]:
    best: tuple[int, int, int, int, int, int] | None = None
    for color in np.unique(grid):
        coords = np.argwhere(grid == int(color))
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = _bbox_coords(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        area = h * w
        if h < 2 or w < 2:
            continue
        if area != len(coords):
            continue
        cand = (area, h, w, r0, c0, int(color))
        if best is None or cand > best:
            best = cand
    assert best is not None, "grid must contain a unique-color occluder rectangle"
    area, h, w, r0, c0, color = best
    mask = np.zeros_like(grid, dtype=bool)
    mask[r0 : r0 + h, c0 : c0 + w] = True
    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)),
        "mask": mask,
        "area": int(area),
    }


def _period_candidates(grid: np.ndarray, blocked: np.ndarray, axis: int) -> list[int]:
    dim = grid.shape[axis]
    max_p = min(_MAX_PERIOD, dim - 1)
    if max_p < 2:
        return []
    items: list[tuple[float, int, int]] = []
    for p in range(2, max_p + 1):
        if axis == 0:
            mask = ~blocked[:-p, :] & ~blocked[p:, :]
            total = int(mask.sum())
            if total == 0:
                continue
            matches = int(np.count_nonzero((grid[:-p, :] == grid[p:, :]) & mask))
        else:
            mask = ~blocked[:, :-p] & ~blocked[:, p:]
            total = int(mask.sum())
            if total == 0:
                continue
            matches = int(np.count_nonzero((grid[:, :-p] == grid[:, p:]) & mask))
        ratio = matches / total
        items.append((ratio, matches, p))
    items.sort(key=lambda x: (x[0], x[1], -x[2]), reverse=True)
    if not items:
        return list(range(2, max_p + 1))
    keep: list[int] = []
    best_ratio = items[0][0]
    for ratio, matches, p in items:
        if ratio + 1e-9 >= best_ratio - 0.06 or len(keep) < 2:
            keep.append(int(p))
        if len(keep) >= 3:
            break
    for p in range(2, max_p + 1):
        if p not in keep and len(keep) < 3:
            keep.append(p)
    return sorted(set(keep))


def _build_prefix(grid: np.ndarray, visible: np.ndarray, pr: int, pc: int) -> np.ndarray:
    h, w = grid.shape
    col_counts = np.zeros((w, pr, 10), dtype=np.int16)
    coords = np.argwhere(visible)
    for r, c in coords:
        col_counts[int(c), int(r % pr), int(grid[r, c])] += 1

    prefix = np.zeros((pr, w + 1, pr, pc, 10), dtype=np.int16)
    for off in range(pr):
        shifted = np.roll(col_counts, shift=off, axis=1)
        for c in range(w):
            prefix[off, c + 1] = prefix[off, c]
            prefix[off, c + 1, :, c % pc, :] += shifted[c]
    return prefix


def _phase_score(phase_counts: np.ndarray) -> int:
    return int(phase_counts.max(axis=-1).sum())


def _infer_model(grid: np.ndarray, blocked: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visible = ~blocked

    pr_choices = _period_candidates(g, blocked, axis=0)
    pc_choices = _period_candidates(g, blocked, axis=1)
    if not pr_choices:
        pr_choices = list(range(2, min(_MAX_PERIOD, h - 1) + 1))
    if not pc_choices:
        pc_choices = list(range(2, min(_MAX_PERIOD, w - 1) + 1))

    best_three: dict[str, object] | None = None
    second_three = -1
    best_two = -1
    best_one = -1
    min_band = 2

    for pr in pr_choices:
        for pc in pc_choices:
            prefix = _build_prefix(g, visible, pr, pc)
            full0 = prefix[0, w]
            one_score = _phase_score(full0)
            if one_score > best_one:
                best_one = one_score

            for split in range(min_band, w - min_band + 1):
                left0 = prefix[0, split]
                for off in range(1, pr):
                    phase_counts = left0 + (prefix[off, w] - prefix[off, split])
                    two_score = _phase_score(phase_counts)
                    if two_score > best_two:
                        best_two = two_score

            for b1 in range(min_band, w - 2 * min_band + 1):
                left0 = prefix[0, b1]
                for b2 in range(b1 + min_band, w - min_band + 1):
                    right_start = b2
                    for o1 in range(1, pr):
                        mid = prefix[o1, b2] - prefix[o1, b1]
                        for o2 in range(pr):
                            if o2 == o1:
                                continue
                            phase_counts = left0 + mid + (prefix[o2, w] - prefix[o2, right_start])
                            score = _phase_score(phase_counts)
                            if best_three is None or score > int(best_three["score"]):
                                if best_three is not None:
                                    second_three = max(second_three, int(best_three["score"]))
                                best_three = {
                                    "score": int(score),
                                    "pr": int(pr),
                                    "pc": int(pc),
                                    "b1": int(b1),
                                    "b2": int(b2),
                                    "o1": int(o1),
                                    "o2": int(o2),
                                    "phase_counts": phase_counts.copy(),
                                }
                            elif score > second_three:
                                second_three = int(score)

    assert best_three is not None, "failed to infer a three-band periodic model"
    best_three["motif"] = np.argmax(best_three["phase_counts"], axis=-1).astype(int)
    best_three["runner_up"] = int(second_three)
    best_three["best_two"] = int(best_two)
    best_three["best_one"] = int(best_one)
    return best_three


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the occluded cells of a three-band faulted translational tiling without straightening the faults.

    Input structure:
    - The whole canvas is filled by a multicolor 2D translational motif.
    - Two hidden vertical fault lines split the grid into three column bands; each band keeps the same horizontal phase but has its own vertical phase offset.
    - One unique-color rectangle occludes part of the tiling and may cross one or both faults; small non-rectangular distractors may also be present.

    Rule:
    - Detect the occluder rectangle and ignore it while inferring the underlying motif.
    - Recover the motif period, the two fault columns, and the vertical phase offset of each band from the still-visible cells.
    - Repaint only the occluded rectangle so each covered cell receives the motif color for its own band phase; keep the faults, offsets, and every non-occluder cell unchanged.

    Key insight:
    - The grid is not globally periodic: the same motif repeats with different vertical phase shifts in three vertical bands, so completion must be done band-by-band rather than with one global tile.
    """
    g = np.asarray(grid, dtype=int)
    occ = _find_occluder(g)
    model = _infer_model(g, occ["mask"])
    out = g.copy()
    pr = int(model["pr"])
    pc = int(model["pc"])
    b1 = int(model["b1"])
    b2 = int(model["b2"])
    o1 = int(model["o1"])
    o2 = int(model["o2"])
    motif = np.asarray(model["motif"], dtype=int)

    for r, c in np.argwhere(occ["mask"]):
        rr = int(r)
        cc = int(c)
        band = _band_of_col(cc, b1, b2)
        off = 0 if band == 0 else o1 if band == 1 else o2
        out[rr, cc] = int(motif[(rr + off) % pr, cc % pc])
    return out


def _minimal_row_period(motif: np.ndarray) -> int:
    pr = motif.shape[0]
    for p in range(1, pr + 1):
        if pr % p != 0:
            continue
        if np.array_equal(motif, np.take(motif, np.arange(pr) % p, axis=0)):
            return p
    return pr


def _minimal_col_period(motif: np.ndarray) -> int:
    pc = motif.shape[1]
    for p in range(1, pc + 1):
        if pc % p != 0:
            continue
        if np.array_equal(motif, np.take(motif, np.arange(pc) % p, axis=1)):
            return p
    return pc


def _build_clean_grid(
    motif: np.ndarray,
    height: int,
    widths: tuple[int, int, int],
    offsets: tuple[int, int, int],
) -> np.ndarray:
    pr, pc = motif.shape
    w0, w1, w2 = widths
    width = w0 + w1 + w2
    grid = np.zeros((height, width), dtype=int)
    b1 = w0
    b2 = w0 + w1
    for r in range(height):
        for c in range(width):
            band = _band_of_col(c, b1, b2)
            grid[r, c] = int(motif[(r + offsets[band]) % pr, c % pc])
    return grid


def _shape_mask(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    arr = np.asarray(shape, dtype=int)
    r0, c0, r1, c1 = _bbox_coords(arr)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in arr:
        mask[int(r - r0), int(c - c0)] = True
    return mask


def _try_place_distractors(
    grid: np.ndarray,
    reserved: np.ndarray,
    colors: list[int],
    rng: random.Random,
    count: int,
) -> bool:
    h, w = grid.shape
    shapes = list(_DISTRACTOR_SHAPES)
    rng.shuffle(shapes)
    for idx in range(count):
        color = colors[idx]
        placed = False
        for shape in shapes:
            mask = _shape_mask(shape)
            sh, sw = mask.shape
            cells = np.argwhere(mask)
            tries = 0
            while tries < 40:
                tries += 1
                r0 = rng.randint(0, h - sh)
                c0 = rng.randint(0, w - sw)
                if reserved[r0 : r0 + sh, c0 : c0 + sw].any():
                    continue
                for rr, cc in cells:
                    grid[r0 + rr, c0 + cc] = int(color)
                    reserved[r0 + rr, c0 + cc] = True
                placed = True
                break
            if placed:
                break
        if not placed:
            return False
    return True


def _phase_coverage(
    shape: tuple[int, int],
    blocked: np.ndarray,
    pr: int,
    pc: int,
    b1: int,
    b2: int,
    offsets: tuple[int, int, int],
) -> np.ndarray:
    h, w = shape
    seen = np.zeros((pr, pc), dtype=bool)
    for r in range(h):
        for c in range(w):
            if blocked[r, c]:
                continue
            band = _band_of_col(c, b1, b2)
            seen[(r + offsets[band]) % pr, c % pc] = True
    return seen


def _choose_occluder(
    h: int,
    widths: tuple[int, int, int],
    pr: int,
    branch: int,
    rng: random.Random,
) -> tuple[int, int, int, int]:
    w0, w1, w2 = widths
    b1 = w0
    b2 = w0 + w1
    width = w0 + w1 + w2
    if branch == 0:
        left_min = max(0, b1 - max(2, w0 // 2) - 1)
        right_max = min(width - 2, b2 + max(1, w2 // 2))
        c0 = rng.randint(left_min, b1 - 1)
        c1 = rng.randint(b2, right_max)
    elif branch == 1:
        c0 = rng.randint(max(0, b1 - max(2, w0 // 2)), b1 - 1)
        c1 = rng.randint(b1, b2 - 1)
    else:
        c0 = rng.randint(b1, max(b1, b2 - 2))
        c1 = rng.randint(b2, width - 1)

    min_h = max(3, pr + 1)
    max_h = max(min_h, h - 2)
    if min_h > max_h:
        min_h = max_h
    occ_h = rng.randint(min_h, max_h)
    r0 = rng.randint(1, h - occ_h - 1)
    return int(r0), int(c0), int(r0 + occ_h - 1), int(c1)


def _normalize_offsets(offsets: tuple[int, int, int], pr: int) -> tuple[int, int, int]:
    base = offsets[0] % pr
    return tuple(int((x - base) % pr) for x in offsets)


def _same_model(
    inferred: dict[str, object],
    motif: np.ndarray,
    b1: int,
    b2: int,
    offsets: tuple[int, int, int],
) -> bool:
    pr, pc = motif.shape
    if int(inferred["pr"]) != pr or int(inferred["pc"]) != pc:
        return False
    if int(inferred["b1"]) != b1 or int(inferred["b2"]) != b2:
        return False
    inferred_offsets = (0, int(inferred["o1"]), int(inferred["o2"]))
    if _normalize_offsets(inferred_offsets, pr) != _normalize_offsets(offsets, pr):
        return False
    return np.array_equal(np.asarray(inferred["motif"], dtype=int), motif)


def _gen_attempt(branch: int) -> np.ndarray | None:
    rng = random.Random(random.randrange(1 << 30))

    pr = rng.randint(2, 5)
    pc = rng.randint(2, 5)
    palette = list(range(10))
    rng.shuffle(palette)
    motif_colors = palette[: rng.randint(3, 5)]
    spare_colors = palette[len(motif_colors) :]
    if len(spare_colors) < 3:
        return None

    motif = rng.choices(motif_colors, k=pr * pc)
    motif = np.asarray(motif, dtype=int).reshape(pr, pc)
    if len(set(int(x) for x in motif.ravel())) < 3:
        return None
    if _minimal_row_period(motif) != pr or _minimal_col_period(motif) != pc:
        return None

    min_band = max(4, pc + 2)
    widths = (
        rng.randint(min_band, min_band + 3),
        rng.randint(min_band, min_band + 3),
        rng.randint(min_band, min_band + 3),
    )
    width = sum(widths)
    if width > 28:
        return None

    height = rng.randint(max(8, 2 * pr + 2), min(18, 5 * pr + 4))
    if height > 30:
        return None

    d1 = rng.randint(1, pr - 1)
    d2 = rng.randint(1, pr - 1)
    offsets = (0, d1, (d1 + d2) % pr)
    clean = _build_clean_grid(motif, height, widths, offsets)

    occ_color = spare_colors[0]
    distractor_colors = spare_colors[1:]
    occ_r0, occ_c0, occ_r1, occ_c1 = _choose_occluder(height, widths, pr, branch, rng)
    occ_mask = np.zeros_like(clean, dtype=bool)
    occ_mask[occ_r0 : occ_r1 + 1, occ_c0 : occ_c1 + 1] = True

    reserved = occ_mask.copy()
    distractor_count = 1 + ((branch + _GEN_CALL_COUNT) % 2)
    if not _try_place_distractors(clean, reserved, distractor_colors, rng, distractor_count):
        return None

    inp = clean.copy()
    inp[occ_mask] = int(occ_color)
    out = clean.copy()

    b1 = widths[0]
    b2 = widths[0] + widths[1]
    if not _phase_coverage(inp.shape, occ_mask, pr, pc, b1, b2, offsets).all():
        return None

    try:
        inferred = _infer_model(inp, occ_mask)
    except AssertionError:
        return None
    if not _same_model(inferred, motif, b1, b2, offsets):
        return None
    if inferred["runner_up"] >= inferred["score"]:
        return None
    if int(inferred["best_two"]) >= int(inferred["score"]):
        return None
    if int(inferred["best_one"]) >= int(inferred["score"]):
        return None

    if branch == 0 and not (occ_c0 < b1 and occ_c1 >= b2):
        return None
    if branch == 1 and not (occ_c0 < b1 <= occ_c1 < b2):
        return None
    if branch == 2 and not (b1 <= occ_c0 < b2 <= occ_c1):
        return None

    if np.array_equal(inp, out):
        return None
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    branch = _GEN_CALL_COUNT % 3
    _GEN_CALL_COUNT += 1
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(branch)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid multi-fault repair input")
