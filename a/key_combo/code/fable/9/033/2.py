from __future__ import annotations

import random

import numpy as np


TEAL = 8
BG = 0
FRAME_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
CORNERS = ("tl", "tr", "bl", "br")
ROTATION_ORDER = ("tl", "bl", "br", "tr")
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _border_mask(n: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _corner_local(corner: str, n: int) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, n - 1
    if corner == "bl":
        return n - 1, 0
    if corner == "br":
        return n - 1, n - 1
    raise AssertionError(f"unknown corner {corner}")


def _rotation_steps(src_corner: str, dst_corner: str) -> int:
    return (ROTATION_ORDER.index(dst_corner) - ROTATION_ORDER.index(src_corner)) % 4


def _count_run(grid: np.ndarray, r: int, c: int, dr: int, dc: int, color: int) -> int:
    count = 0
    h, w = grid.shape
    while 0 <= r < h and 0 <= c < w and int(grid[r, c]) == int(color):
        count += 1
        r += dr
        c += dc
    return count


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _frame_from_teal(grid: np.ndarray, r: int, c: int, bg: int) -> dict[str, object]:
    h, w = grid.shape
    candidates: list[dict[str, object]] = []
    specs = {
        "tl": ((0, 1), (1, 0)),
        "tr": ((0, -1), (1, 0)),
        "bl": ((0, 1), (-1, 0)),
        "br": ((0, -1), (-1, 0)),
    }

    for corner, (hdir, vdir) in specs.items():
        rr1, cc1 = r + hdir[0], c + hdir[1]
        rr2, cc2 = r + vdir[0], c + vdir[1]
        if not (0 <= rr1 < h and 0 <= cc1 < w and 0 <= rr2 < h and 0 <= cc2 < w):
            continue

        color = int(grid[rr1, cc1])
        if color in (bg, TEAL) or int(grid[rr2, cc2]) != color:
            continue

        arm_h = _count_run(grid, rr1, cc1, hdir[0], hdir[1], color)
        arm_v = _count_run(grid, rr2, cc2, vdir[0], vdir[1], color)
        if arm_h != arm_v or arm_h < 3:
            continue
        n = arm_h + 1

        if corner == "tl":
            r0, c0 = r, c
        elif corner == "tr":
            r0, c0 = r, c - n + 1
        elif corner == "bl":
            r0, c0 = r - n + 1, c
        else:
            r0, c0 = r - n + 1, c - n + 1
        r1, c1 = r0 + n - 1, c0 + n - 1
        if not (0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w):
            continue

        window = grid[r0 : r1 + 1, c0 : c1 + 1]
        if window.shape != (n, n):
            continue

        local_r, local_c = _corner_local(corner, n)
        border = _border_mask(n)
        allowed = {bg, color, TEAL}
        if any(int(v) not in allowed for v in np.unique(window)):
            continue
        if int(window[local_r, local_c]) != TEAL:
            continue
        other_teals = np.argwhere(window == TEAL)
        if other_teals.shape != (1, 2):
            continue

        border_without_corner = border.copy()
        border_without_corner[local_r, local_c] = False
        if not np.all(window[border_without_corner] == color):
            continue
        interior = window[1:-1, 1:-1]
        if interior.size == 0 or not np.all((interior == bg) | (interior == color)):
            continue

        motif = (window == color) & ~border
        candidates.append(
            {
                "bbox": (r0, c0, r1, c1),
                "corner": corner,
                "color": color,
                "n": n,
                "motif": motif,
            }
        )

    assert len(candidates) == 1, "each teal pixel must anchor exactly one square frame"
    return candidates[0]


def _extract_frames(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    teal_coords = np.argwhere(g == TEAL)
    assert len(teal_coords) >= 2, "expected multiple teal-cornered frames"

    frames = [_frame_from_teal(g, int(r), int(c), bg) for r, c in teal_coords]
    bboxes = [frame["bbox"] for frame in frames]
    assert len(set(bboxes)) == len(bboxes), "two teal pixels cannot belong to the same frame"
    for i, bbox_a in enumerate(bboxes):
        for bbox_b in bboxes[i + 1 :]:
            assert not _bboxes_overlap(bbox_a, bbox_b), "frames must not overlap"

    covered = np.zeros_like(g, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        covered[r0 : r1 + 1, c0 : c1 + 1] |= g[r0 : r1 + 1, c0 : c1 + 1] != bg
    assert np.array_equal(covered, g != bg), "all non-background pixels must belong to frames"

    sizes = {int(frame["n"]) for frame in frames}
    assert len(sizes) == 1, "all frames must share one square size"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the unique frame motif by teal-corner direction and stamp it into every other same-sized frame.

    Input structure:
    - A black background contains several one-cell-thick hollow square frames of equal size.
    - Each frame uses one non-teal color, except exactly one corner border cell is replaced by a teal control pixel.
    - Exactly one frame also contains same-colored interior motif pixels; all other frame interiors are empty.

    Rule:
    - Treat the patterned frame as the template and keep its current motif orientation relative to its teal corner.
    - For every other frame, rotate the template motif around the square center by the amount that moves the template teal corner to that frame's teal corner.
    - Paint the rotated motif inside the target frame using that frame's own color, while leaving all borders and teal corners unchanged.
    - The template frame itself stays unchanged.

    Key insight:
    - The teal corner is an orientation pointer, so the interior motif must be copied by rotation relative to corner position rather than by plain translation.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    assert bg == BG, "generator expects black background"

    frames = _extract_frames(g)
    patterned = [frame for frame in frames if np.any(frame["motif"])]
    assert len(patterned) == 1, "expected exactly one patterned template frame"
    template = patterned[0]

    for frame in frames:
        if frame is template:
            continue
        assert not np.any(frame["motif"]), "non-template frames must start hollow"

    out = g.copy()
    template_motif = np.asarray(template["motif"], dtype=bool)
    for frame in frames:
        if frame is template:
            continue
        k = _rotation_steps(str(template["corner"]), str(frame["corner"]))
        rotated = np.rot90(template_motif, k)
        r0, c0, r1, c1 = frame["bbox"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[rotated] = int(frame["color"])
    return out


def _frame_patch(n: int, color: int, corner: str, motif: np.ndarray | None = None) -> np.ndarray:
    patch = np.full((n, n), BG, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    rr, cc = _corner_local(corner, n)
    patch[rr, cc] = TEAL
    if motif is not None:
        patch[motif] = color
        patch[rr, cc] = TEAL
    return patch


def _build_wrong_unrotated_output(inp: np.ndarray) -> np.ndarray:
    frames = _extract_frames(inp)
    template = next(frame for frame in frames if np.any(frame["motif"]))
    out = inp.copy()
    template_motif = np.asarray(template["motif"], dtype=bool)
    for frame in frames:
        if frame is template:
            continue
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1][template_motif] = int(frame["color"])
    return out


def _random_motif(n: int) -> np.ndarray:
    inner_n = n - 2
    for _ in range(300):
        inner = np.zeros((inner_n, inner_n), dtype=bool)

        row = random.randrange(inner_n)
        c0 = random.randrange(inner_n - 1)
        c1 = random.randrange(c0 + 1, inner_n)
        inner[row, c0 : c1 + 1] = True

        col = random.randrange(inner_n)
        r0 = random.randrange(inner_n - 1)
        r1 = random.randrange(r0 + 1, inner_n)
        inner[r0 : r1 + 1, col] = True

        if random.random() < 0.8:
            if random.random() < 0.5:
                row2 = random.randrange(inner_n)
                start = random.randrange(inner_n)
                length = random.randint(1, max(1, inner_n - start))
                inner[row2, start : start + length] = True
            else:
                col2 = random.randrange(inner_n)
                start = random.randrange(inner_n)
                length = random.randint(1, max(1, inner_n - start))
                inner[start : start + length, col2] = True

        for _ in range(random.randint(1, 3)):
            inner[random.randrange(inner_n), random.randrange(inner_n)] = True

        if inner.sum() < 4 or inner.sum() > inner_n * inner_n - 2:
            continue
        coords = np.argwhere(inner)
        if len(np.unique(coords[:, 0])) < 2 or len(np.unique(coords[:, 1])) < 2:
            continue

        full = np.zeros((n, n), dtype=bool)
        full[1:-1, 1:-1] = inner
        if any(np.array_equal(full, np.rot90(full, k)) for k in (1, 2, 3)):
            continue
        return full
    raise RuntimeError("failed to sample a non-symmetric motif")


def _can_place(occupied: np.ndarray, r: int, c: int, n: int, gap: int) -> bool:
    h, w = occupied.shape
    r0 = max(0, r - gap)
    c0 = max(0, c - gap)
    r1 = min(h, r + n + gap)
    c1 = min(w, c + n + gap)
    return not occupied[r0:r1, c0:c1].any()


def _place_boxes(h: int, w: int, n: int, count: int, gap: int = 1) -> list[tuple[int, int]] | None:
    occupied = np.zeros((h, w), dtype=bool)
    cells = [(r, c) for r in range(h - n + 1) for c in range(w - n + 1)]
    random.shuffle(cells)
    placed: list[tuple[int, int]] = []
    for r, c in cells:
        if not _can_place(occupied, r, c, n, gap):
            continue
        occupied[r : r + n, c : c + n] = True
        placed.append((r, c))
        if len(placed) == count:
            return placed
    return None


def _sample_corners(count: int) -> list[str]:
    assert count >= 3
    template_corner = random.choice(CORNERS)
    unique_goal = min(4, 3 if count <= 4 else 4)
    pool = [template_corner]
    others = [corner for corner in CORNERS if corner != template_corner]
    random.shuffle(others)
    pool.extend(others[: unique_goal - 1])

    rest = pool[1:].copy()
    while len(rest) < count - 1:
        rest.append(random.choice(pool))
    random.shuffle(rest)
    return [template_corner, *rest[: count - 1]]


def _generate_pair() -> tuple[np.ndarray, np.ndarray] | None:
    frame_count = random.randint(3, 5)
    n = random.randint(5, 8)
    min_side = max(15, 2 * n + 3)
    h = random.randint(min_side, 30)
    w = random.randint(min_side, 30)
    positions = _place_boxes(h, w, n, frame_count, gap=1)
    if positions is None:
        return None

    colors = random.sample(FRAME_COLORS, k=frame_count)
    corners = _sample_corners(frame_count)
    template_idx = random.randrange(frame_count)
    motif = _random_motif(n)

    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)
    template_corner = corners[0]

    frame_specs = []
    for i in range(frame_count):
        if i == template_idx:
            corner = template_corner
        else:
            non_template_index = 1 + len(frame_specs) - (1 if template_idx < i else 0)
            corner = corners[non_template_index]
        frame_specs.append((colors[i], corner))

    for i, ((r, c), (color, corner)) in enumerate(zip(positions, frame_specs)):
        k = _rotation_steps(template_corner, corner)
        rotated = np.rot90(motif, k)
        out[r : r + n, c : c + n] = _frame_patch(n, color, corner, rotated)
        inp[r : r + n, c : c + n] = _frame_patch(n, color, corner, rotated if i == template_idx else None)

    return inp, out


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        parsed = _extract_frames(inp)
    except AssertionError:
        return False

    patterned = [frame for frame in parsed if np.any(frame["motif"])]
    if len(patterned) != 1:
        return False
    template = patterned[0]
    distinct_corners = {str(frame["corner"]) for frame in parsed}
    if len(distinct_corners) < 2:
        return False
    if not any(str(frame["corner"]) != str(template["corner"]) for frame in parsed if frame is not template):
        return False

    wrong = _build_wrong_unrotated_output(inp)
    if np.array_equal(wrong, out):
        return False

    if not np.array_equal(main(inp), out):
        return False

    produced = _extract_frames(out)
    template_out = next(frame for frame in produced if frame["bbox"] == template["bbox"])
    if not np.array_equal(np.asarray(template["motif"]), np.asarray(template_out["motif"])):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        pair = _generate_pair()
        if pair is None:
            continue
        inp, out = pair
        if _semantic_checks(inp, out):
            return inp
    raise RuntimeError("failed to generate a valid oriented-stamp sample")
