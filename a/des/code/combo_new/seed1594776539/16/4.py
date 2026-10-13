from __future__ import annotations

import itertools
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(arr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return arr[r0 : r1 + 1, c0 : c1 + 1]


def _tight_crop_mask(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    if box is None:
        return mask[:0, :0]
    return _crop_bbox(mask, box)


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _neighbors8(r: int, c: int, h: int, w: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w:
                yield nr, nc


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    stepper = _neighbors4 if connectivity == 4 else _neighbors8
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        pts: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            pts.append((rr, cc))
            for nr, nc in stepper(rr, cc, h, w):
                if mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.asarray(pts, dtype=int))
    return comps


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    pts = np.argwhere(mask)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def _canonical_template(mask: np.ndarray) -> np.ndarray:
    base = _tight_crop_mask(mask)
    candidates: list[tuple[tuple[object, ...], np.ndarray]] = []
    for k in range(4):
        rotated = _tight_crop_mask(np.rot90(base, k))
        h, w = map(int, rotated.shape)
        cent_r, cent_c = _centroid(rotated)
        score = (
            0 if w >= h else 1,
            cent_r,
            cent_c,
            rotated.shape,
            rotated.tobytes(),
        )
        candidates.append((score, rotated))
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    canon = _canonical_template(mask)
    return tuple(map(int, canon.shape)), canon.tobytes()


def _find_indicator(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = grid.shape
    non_bg = grid != bg
    corners = (
        lambda r, c: r < 2 and c < 2,
        lambda r, c: r < 2 and c >= w - 2,
        lambda r, c: r >= h - 2 and c < 2,
        lambda r, c: r >= h - 2 and c >= w - 2,
    )
    candidates: list[tuple[int, int, int]] = []
    for comp in _components(non_bg, connectivity=8):
        if int(comp.shape[0]) != 1:
            continue
        r, c = map(int, comp[0])
        if any(check(r, c) for check in corners):
            candidates.append((r, c, int(grid[r, c])))
    assert len(candidates) == 1, "expected a unique corner indicator pixel"
    return candidates[0]


def _candidate_supports(comp: np.ndarray, bg: int) -> dict[tuple[tuple[int, int], bytes], np.ndarray]:
    colors = [int(c) for c in np.unique(comp) if int(c) != int(bg)]
    assert 2 <= len(colors) <= 4, "expected 2-4 copy colors"
    visible = {color: comp == int(color) for color in colors}
    canonical: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}

    for rotations in itertools.product(range(4), repeat=len(colors)):
        support = np.zeros(comp.shape, dtype=bool)
        for color, rot in zip(colors, rotations):
            support |= np.rot90(visible[color], -int(rot))
        full_masks = {
            color: np.rot90(support, int(rot))
            for color, rot in zip(colors, rotations)
        }

        for order in itertools.permutations(colors):
            covered_above = np.zeros(comp.shape, dtype=bool)
            ok = True
            for color in reversed(order):
                visible_now = full_masks[color] & ~covered_above
                if not np.array_equal(visible_now, visible[color]):
                    ok = False
                    break
                covered_above |= full_masks[color]
            if ok:
                canonical[_canonical_key(support)] = _canonical_template(support)
                break

    return canonical


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a shared template from overlapping rotated copies and recolor it with the indicator color.

    Input structure:
    - The background is black.
    - Away from one corner marker, the foreground is made of 2-4 colors that belong to overlapping copies of one connected irregular template.
    - Each copy is a 0/90/180/270-degree rotation of that same template, and later copies overwrite earlier ones on overlaps.
    - A single isolated non-black indicator pixel sits inside one 2x2 corner region and repeats one copy's color.

    Rule:
    - Remove the corner indicator from consideration and crop the remaining overlapped stack.
    - For each foreground color, treat the visible cells as a partially occluded rotated copy of one hidden common template.
    - Find a template plus a layering order whose rotated full copies exactly reproduce the observed visible color masks.
    - Rotate the recovered template to its canonical orientation: prefer a tight crop whose width is at least its height, then break remaining ties toward the orientation whose centroid sits most upper-left.
    - Output the tight crop of that canonical template, recolored with the indicator pixel's color on a black background.

    Key insight:
    The colored fragments are visibility masks from several rotated instances of the same hidden shape; the lone corner pixel selects only the output color, not which visible fragment to copy.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    ind_r, ind_c, out_color = _find_indicator(g, bg)

    composite_mask = g != bg
    composite_mask[ind_r, ind_c] = False
    comp_box = _bbox(composite_mask)
    assert comp_box is not None, "missing overlapped copy region"
    comp = _crop_bbox(g, comp_box)
    assert comp.shape[0] == comp.shape[1], "overlap region must admit a square support"

    canonical = _candidate_supports(comp, bg)
    assert canonical, "failed to recover a consistent shared template"
    assert len(canonical) == 1, "template reconstruction is ambiguous"
    template = next(iter(canonical.values()))

    out = np.zeros(template.shape, dtype=int)
    out[template] = int(out_color)
    return out


GRID_SIDE_MIN = 14
GRID_SIDE_MAX = 18
TEMPLATE_H_RANGE = (3, 6)
TEMPLATE_W_MAX = 8
TEMPLATE_DENSITY_MIN = 0.45
TEMPLATE_DENSITY_MAX = 0.80
N_COPIES_RANGE = (2, 4)
MIN_VISIBLE_PIXELS = 2
GEN_RETRIES = 500
TEMPLATE_RETRIES = 200


def _connected_mask(height: int, width: int, target_area: int) -> np.ndarray | None:
    mask = np.zeros((int(height), int(width)), dtype=bool)
    steps = ["D"] * (int(height) - 1) + ["R"] * (int(width) - 1)
    random.shuffle(steps)

    r = c = 0
    mask[r, c] = True
    for step in steps:
        if step == "D":
            r += 1
        else:
            c += 1
        mask[r, c] = True

    if int(mask.sum()) > int(target_area):
        return None

    frontier: set[tuple[int, int]] = set()

    def add_frontier(rr: int, cc: int) -> None:
        for nr, nc in _neighbors4(rr, cc, int(height), int(width)):
            if not mask[nr, nc]:
                frontier.add((int(nr), int(nc)))

    for rr, cc in np.argwhere(mask):
        add_frontier(int(rr), int(cc))

    while int(mask.sum()) < int(target_area):
        if not frontier:
            return None
        rr, cc = random.choice(tuple(frontier))
        frontier.remove((rr, cc))
        mask[rr, cc] = True
        add_frontier(rr, cc)

    return mask


def _rotation_count(mask: np.ndarray) -> int:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    base = _tight_crop_mask(mask)
    for k in range(4):
        rotated = _tight_crop_mask(np.rot90(base, k))
        seen.add((tuple(map(int, rotated.shape)), rotated.tobytes()))
    return len(seen)


def _sample_template() -> np.ndarray | None:
    for _ in range(TEMPLATE_RETRIES):
        height = random.randint(*TEMPLATE_H_RANGE)
        width = random.randint(height, TEMPLATE_W_MAX)
        area = int(height * width)
        min_area = max(height + width - 1, int(np.ceil(TEMPLATE_DENSITY_MIN * area)))
        max_area = min(area - 1, int(np.floor(TEMPLATE_DENSITY_MAX * area)))
        if min_area > max_area:
            continue
        target_area = random.randint(min_area, max_area)
        mask = _connected_mask(height, width, target_area)
        if mask is None:
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if _rotation_count(mask) != 4:
            continue
        return mask
    return None


def _support_from_template(template: np.ndarray) -> np.ndarray:
    side = int(template.shape[1])
    support = np.zeros((side, side), dtype=bool)
    support[: template.shape[0], : template.shape[1]] = template
    return support


def _render_stack(
    support: np.ndarray,
    colors: list[int],
    rotations: list[int],
    order: list[int],
) -> np.ndarray:
    full_masks = {
        color: np.rot90(support, int(rot))
        for color, rot in zip(colors, rotations)
    }
    comp = np.zeros(support.shape, dtype=int)
    for color in order:
        comp[full_masks[color]] = int(color)
    return comp


def _place_indicator(
    grid: np.ndarray,
    color: int,
    top: int,
    left: int,
    side: int,
) -> bool:
    h, w = grid.shape
    corner_specs = [
        ((0, 1), (0, 1)),
        ((0, 1), (w - 2, w - 1)),
        ((h - 2, h - 1), (0, 1)),
        ((h - 2, h - 1), (w - 2, w - 1)),
    ]
    random.shuffle(corner_specs)
    for rows, cols in corner_specs:
        rr = random.randint(rows[0], rows[1])
        cc = random.randint(cols[0], cols[1])
        if grid[rr, cc] != 0:
            continue
        if top - 2 <= rr <= top + side + 1 and left - 2 <= cc <= left + side + 1:
            continue
        for nr in range(max(0, rr - 1), min(h, rr + 2)):
            for nc in range(max(0, cc - 1), min(w, cc + 2)):
                if (nr != rr or nc != cc) and grid[nr, nc] != 0:
                    break
            else:
                continue
            break
        else:
            grid[rr, cc] = int(color)
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    template = _sample_template()
    if template is None:
        return None

    n_copies = random.randint(*N_COPIES_RANGE)
    rotations = random.sample(range(4), n_copies)
    colors = random.sample(range(1, 10), n_copies)
    order = random.sample(colors, n_copies)

    support = _support_from_template(template)
    comp = _render_stack(support, colors, rotations, order)
    side = int(support.shape[0])
    comp_box = _bbox(comp != 0)
    if comp_box != (0, 0, side - 1, side - 1):
        return None

    visible = {color: comp == int(color) for color in colors}
    recovered = np.zeros_like(support)
    for color, rot in zip(colors, rotations):
        recovered |= np.rot90(visible[color], -int(rot))
    if not np.array_equal(recovered, support):
        return None
    if min(int(visible[color].sum()) for color in colors) < MIN_VISIBLE_PIXELS:
        return None

    min_side = max(GRID_SIDE_MIN, side + 7)
    if min_side > GRID_SIDE_MAX:
        return None
    grid_h = random.randint(min_side, GRID_SIDE_MAX)
    grid_w = random.randint(min_side, GRID_SIDE_MAX)
    top = random.randint(3, grid_h - side - 4)
    left = random.randint(3, grid_w - side - 4)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[top : top + side, left : left + side] = comp

    indicator_color = random.choice(colors)
    if not _place_indicator(grid, indicator_color, top, left, side):
        return None

    expected_mask = _canonical_template(support)
    expected = np.zeros(expected_mask.shape, dtype=int)
    expected[expected_mask] = int(indicator_color)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid overlapping-rotation input")
