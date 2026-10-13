from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as arrays of (r, c) coordinates."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or not bool(mask[sr, sc]):
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            pts: list[Coord] = []
            while stack:
                r, c = stack.pop()
                pts.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _dihedral_variants(template: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    ids = np.arange(template.size, dtype=int).reshape(template.shape)
    variants: list[tuple[str, np.ndarray, np.ndarray]] = []
    for k in range(4):
        rot = np.rot90(template, k)
        rot_ids = np.rot90(ids, k)
        variants.append((f"r{k}", rot.copy(), rot_ids.copy()))
        variants.append((f"r{k}f", np.fliplr(rot).copy(), np.fliplr(rot_ids).copy()))
    return variants


def _unique_dihedral(template: np.ndarray) -> bool:
    variants = _dihedral_variants(template)
    keys = {(sub.shape, sub.tobytes()) for _name, sub, _ids in variants}
    return len(keys) == len(variants)


def _solve_for_clean_and_scar(
    patches: list[np.ndarray],
    clean_index: int,
    scar_color: int,
    bg: int,
) -> np.ndarray | None:
    template = patches[clean_index]
    if int(scar_color) in set(map(int, template.ravel())):
        return None
    if not _unique_dihedral(template):
        return None

    rows, cols = template.shape
    variants = _dihedral_variants(template)
    folded: set[Coord] = set()
    piece_sets: list[set[Coord]] = []
    matched_names: list[str] = []
    nontrivial_fold = False

    for i, patch in enumerate(patches):
        if i == clean_index:
            if not np.array_equal(patch, template):
                return None
            continue

        scar_mask = patch == int(scar_color)
        if not bool(scar_mask.any()):
            return None

        matches: list[tuple[str, np.ndarray]] = []
        for name, sub, ids in variants:
            if sub.shape != patch.shape:
                continue
            if np.all(patch[~scar_mask] == sub[~scar_mask]):
                matches.append((name, ids))
        if len(matches) != 1:
            return None

        name, ids = matches[0]
        matched_names.append(name)
        piece: set[Coord] = set()
        for lr, lc in np.argwhere(scar_mask):
            local_r, local_c = int(lr), int(lc)
            tid = int(ids[local_r, local_c])
            tr, tc = divmod(tid, cols)
            coord = (int(tr), int(tc))
            if coord in folded:
                return None
            folded.add(coord)
            piece.add(coord)
            if patch.shape != template.shape or (local_r, local_c) != coord:
                nontrivial_fold = True
        piece_sets.append(piece)

    if len(piece_sets) < 2:
        return None
    if not folded or any(piece == folded for piece in piece_sets):
        return None
    if not nontrivial_fold or not any(name != "r0" for name in matched_names):
        return None

    out = np.full(template.shape, int(bg), dtype=int)
    for r, c in folded:
        out[r, c] = int(scar_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold scar pixels from transformed rectangular sprite copies back into the clean template frame.

    Input structure:
    - The background is the dominant color.
    - The foreground consists of separated dense rectangular sprite components, all dihedral
      transforms of one multicolor R x C template.
    - Exactly one component is the clean template in the output orientation; every other
      component has one or more cells replaced by one shared scar color not used by the template.
    - The template has no rotational or mirror symmetry, so each scarred component has one
      unique dihedral correspondence to the clean copy.

    Rule:
    - Identify the clean copy and the shared scar color by the unique assignment that lets
      every other dense rectangle match one transformed template after hiding scar cells.
    - For each scarred rectangle, use that rectangle's dihedral id map to fold every scar
      cell back to its coordinate in the clean template.
    - Return an R x C grid filled with the input background color, drawing the scar color
      exactly on the union of all folded scar coordinates.

    Key insight:
    - The scar pixels are not noise to repair; they are disjoint fragments of a target shape
      that never appears whole until the transformed host sprites are aligned to the clean copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    comps = _components(g != int(bg))
    assert len(comps) >= 3, "expected one clean copy and at least two scarred copies"

    patches: list[np.ndarray] = []
    for coords in comps:
        box = _bbox(coords)
        patch = _crop(g, box)
        assert int(coords.shape[0]) == int(patch.size), "sprite components must be dense rectangles"
        assert np.all(patch != int(bg)), "sprite rectangles must not contain background holes"
        patches.append(patch)

    areas = {int(patch.size) for patch in patches}
    assert len(areas) == 1, "all transformed sprites must have the same cell count"

    all_colors = set(map(int, np.unique(g))) - {int(bg)}
    solutions: list[np.ndarray] = []
    for clean_index, template in enumerate(patches):
        template_colors = set(map(int, template.ravel()))
        for scar_color in sorted(all_colors - template_colors):
            solved = _solve_for_clean_and_scar(patches, clean_index, int(scar_color), int(bg))
            if solved is not None:
                solutions.append(solved)

    assert len(solutions) == 1, "expected a unique clean-copy/scar-color interpretation"
    return solutions[0]


# --------------- generator ---------------

_GEN_CALL_COUNT = 0
_MAX_ATTEMPTS = 600
_PALETTE = tuple(range(10))


def _make_template(rows: int, cols: int, colors: list[int]) -> np.ndarray | None:
    for _ in range(400):
        arr = np.array(
            [[random.choice(colors) for _ in range(cols)] for _ in range(rows)],
            dtype=int,
        )
        forced = random.sample(range(rows * cols), len(colors))
        for idx, color in zip(forced, colors):
            arr.flat[idx] = int(color)
        if _unique_dihedral(arr):
            return arr
    return None


def _connected_target(rows: int, cols: int, count: int) -> list[Coord] | None:
    for _ in range(200):
        start = (random.randrange(rows), random.randrange(cols))
        cells = {start}
        frontier = {start}
        while len(cells) < count and frontier:
            base = random.choice(tuple(frontier))
            nbrs = [
                (base[0] + dr, base[1] + dc)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= base[0] + dr < rows and 0 <= base[1] + dc < cols
            ]
            random.shuffle(nbrs)
            added = False
            for nb in nbrs:
                if nb not in cells:
                    cells.add(nb)
                    frontier.add(nb)
                    added = True
                    break
            if not added:
                frontier.discard(base)
        if len(cells) == count:
            vals = list(cells)
            random.shuffle(vals)
            return vals
    return None


def _split_cells(cells: list[Coord], groups: int) -> list[list[Coord]]:
    shuffled = list(cells)
    random.shuffle(shuffled)
    out = [[shuffled[i]] for i in range(groups)]
    for cell in shuffled[groups:]:
        out[random.randrange(groups)].append(cell)
    random.shuffle(out)
    return out


def _orientation_plan(variants: list[tuple[str, np.ndarray, np.ndarray]], scarred_count: int, case: int):
    by_name = {name: (name, sub, ids) for name, sub, ids in variants}
    non_identity = [item for item in variants if item[0] != "r0"]

    if case == 1 and scarred_count >= 2:
        preferred_names = ["r0f", "r1"]
        chosen = [by_name[name] for name in preferred_names if name in by_name]
    elif case == 2 and scarred_count >= 3:
        chosen = [by_name["r0"], random.choice(non_identity)]
    else:
        chosen = [random.choice(non_identity)]

    used = {item[0] for item in chosen}
    pool = [item for item in variants if item[0] not in used]
    random.shuffle(pool)
    while len(chosen) < scarred_count and pool:
        chosen.append(pool.pop())
    if len(chosen) != scarred_count:
        return None
    random.shuffle(chosen)
    return chosen


def _boxes_too_close(a: BBox, b: BBox, pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar0 - pad <= br1 and ar1 + pad >= br0 and ac0 - pad <= bc1 and ac1 + pad >= bc0


def _place_patches(patches: list[np.ndarray], bg: int) -> np.ndarray | None:
    for _ in range(200):
        h = random.randint(18, 30)
        w = random.randint(18, 30)
        grid = np.full((h, w), int(bg), dtype=int)
        boxes: list[BBox] = []
        order = list(patches)
        random.shuffle(order)

        ok = True
        for patch in order:
            ph, pw = patch.shape
            placed = False
            if ph + 2 > h or pw + 2 > w:
                ok = False
                break
            for _try in range(250):
                r0 = random.randint(1, h - ph - 1)
                c0 = random.randint(1, w - pw - 1)
                box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                if any(_boxes_too_close(box, old, 1) for old in boxes):
                    continue
                grid[r0 : r0 + ph, c0 : c0 + pw] = patch
                boxes.append(box)
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return grid
    return None


def _dims_for_case(case: int) -> tuple[int, int]:
    if case == 1:
        side = random.randint(4, 6)
        return side, side
    if case == 0:
        rows = random.randint(3, 5)
        choices = [c for c in range(4, 7) if c != rows]
        return rows, random.choice(choices)
    rows = random.randint(3, 5)
    cols = random.randint(3, 6)
    if random.random() < 0.35:
        cols = rows
    return rows, cols


def _gen_attempt(case: int) -> np.ndarray | None:
    rows, cols = _dims_for_case(case)
    area = rows * cols
    max_scarred = min(5, area - 2)
    if max_scarred < 2:
        return None
    scarred_count = random.randint(2, max_scarred)
    if case == 2:
        scarred_count = max(3, scarred_count)

    bg, scar = random.sample(_PALETTE, 2)
    template_color_count = random.randint(3, min(5, len(_PALETTE) - 2, area))
    template_colors = random.sample([c for c in _PALETTE if c not in (bg, scar)], template_color_count)
    template = _make_template(rows, cols, template_colors)
    if template is None:
        return None

    variants = _dihedral_variants(template)
    chosen = _orientation_plan(variants, scarred_count, case)
    if chosen is None:
        return None

    target_max = min(area - 2, max(scarred_count, area // 2), scarred_count * 3, 10)
    if target_max < scarred_count:
        return None
    target_count = random.randint(scarred_count, target_max)
    target_cells = _connected_target(rows, cols, target_count)
    if target_cells is None:
        return None
    groups = _split_cells(target_cells, scarred_count)

    patches = [template.copy()]
    folded_union = set(target_cells)
    nontrivial_fold = False
    for (name, sub, ids), group in zip(chosen, groups):
        scarred = sub.copy()
        for tr, tc in group:
            tid = int(tr) * cols + int(tc)
            locs = np.argwhere(ids == tid)
            if locs.shape[0] != 1:
                return None
            lr, lc = map(int, locs[0])
            scarred[lr, lc] = int(scar)
            if scarred.shape != template.shape or (lr, lc) != (int(tr), int(tc)):
                nontrivial_fold = True
        patches.append(scarred)

    if not nontrivial_fold:
        return None
    if any(set(group) == folded_union for group in groups):
        return None

    expected = np.full((rows, cols), int(bg), dtype=int)
    for r, c in folded_union:
        expected[r, c] = int(scar)

    grid = _place_patches(patches, int(bg))
    if grid is None or _get_bg(grid) != int(bg):
        return None
    if len(_components(grid != int(bg))) != len(patches):
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if int(np.count_nonzero(actual == int(scar))) != len(folded_union):
        return None
    if not np.all((actual == int(bg)) | (actual == int(scar))):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    case = _GEN_CALL_COUNT % 3
    _GEN_CALL_COUNT += 1
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt(case)
        if grid is not None:
            return grid
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt(random.randrange(3))
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid shattered-scar sprite input")
