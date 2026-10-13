from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
HOLE_SIZE = 4
MAX_GENERATION_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in neighbors:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_components(grid: np.ndarray, bg: int, connectivity: int = 4) -> list[dict]:
    comps = []
    for coords in _mask_components(np.asarray(grid) != int(bg), connectivity=connectivity):
        comps.append(
            {
                "coords": coords,
                "bbox": bbox_coords(coords),
                "area": int(coords.shape[0]),
            }
        )
    return comps


def _find_holes(main_crop: np.ndarray, bg: int) -> list[dict]:
    h, w = main_crop.shape
    holes: list[dict] = []
    for coords in _mask_components(main_crop == int(bg), connectivity=4):
        bbox = bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        side_h = r1 - r0 + 1
        side_w = c1 - c0 + 1
        if side_h != side_w:
            continue
        if int(coords.shape[0]) != side_h * side_w:
            continue
        holes.append({"coords": coords, "bbox": bbox, "size": int(side_h)})
    holes.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return holes


def _hole_target(main_crop: np.ndarray, hole_bbox: tuple[int, int, int, int]) -> dict[str, np.ndarray]:
    r0, c0, r1, c1 = hole_bbox
    return {
        "top": main_crop[r0 - 1, c0 : c1 + 1].copy(),
        "bottom": main_crop[r1 + 1, c0 : c1 + 1].copy(),
        "left": main_crop[r0 : r1 + 1, c0 - 1].copy(),
        "right": main_crop[r0 : r1 + 1, c1 + 1].copy(),
    }


def _seam_match_count(patch: np.ndarray, target: dict[str, np.ndarray]) -> int:
    count = 0
    count += int(np.array_equal(patch[0, :], target["top"]))
    count += int(np.array_equal(patch[-1, :], target["bottom"]))
    count += int(np.array_equal(patch[:, 0], target["left"]))
    count += int(np.array_equal(patch[:, -1], target["right"]))
    return count


def _match_entries(main_crop: np.ndarray, holes: list[dict], patches: list[np.ndarray]) -> list[list[dict]]:
    all_entries: list[list[dict]] = []
    for hole in holes:
        target = _hole_target(main_crop, hole["bbox"])
        entries: list[dict] = []
        for patch_index, patch in enumerate(patches):
            for rot in range(4):
                rotated = np.rot90(patch, rot)
                entries.append(
                    {
                        "patch_index": int(patch_index),
                        "rot": int(rot),
                        "match_count": int(_seam_match_count(rotated, target)),
                    }
                )
        all_entries.append(entries)
    return all_entries


def _extract_problem(grid: np.ndarray) -> tuple[int, np.ndarray, list[dict], list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = _find_components(g, bg=bg, connectivity=4)
    assert comps, "expected foreground components"

    main_comp = max(comps, key=lambda item: int(item["area"]))
    main_crop = crop_bbox(g, main_comp["bbox"])

    holes = _find_holes(main_crop, bg)
    assert holes, "expected at least one interior square hole"
    hole_size = holes[0]["size"]
    assert all(int(hole["size"]) == int(hole_size) for hole in holes), "all holes must share one size"

    patches: list[np.ndarray] = []
    for comp in comps:
        if comp is main_comp:
            continue
        patch = crop_bbox(g, comp["bbox"])
        assert patch.shape == (hole_size, hole_size), "all detached components must be square patches"
        assert not np.any(patch == int(bg)), "patches should be fully filled"
        patches.append(patch)

    assert len(patches) == len(holes) + 1, "expected exactly one extra patch"
    return bg, main_crop, holes, patches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate square patches until their four seams continue the surrounding pattern, then refill the holes.

    Input structure:
    - One side of the grid contains a large multicolor picture with one or more same-sized
      square holes cut out as background.
    - The other side contains the same number of square patch blocks plus one extra distractor patch.
    - Each real patch may be rotated by a multiple of ninety degrees relative to its correct pose.

    Rule:
    - Treat the largest foreground component as the damaged main picture and detect all interior
      square background holes inside its bounding box.
    - For each hole, test every detached square patch in every quarter-turn rotation.
    - A patch fits only when its top, bottom, left, and right edges all exactly match the pixels
      immediately outside the hole on the corresponding seam.
    - Fill each hole with its unique fitting rotated patch, discard the leftover distractor patch,
      and return only the repaired main picture.

    Key insight:
    - The identity and orientation of a patch are determined jointly by the four seam signatures;
      local color counts or single-edge similarity are not sufficient.
    """
    _bg, main_crop, holes, patches = _extract_problem(np.asarray(grid, dtype=int))
    entries_by_hole = _match_entries(main_crop, holes, patches)

    out = main_crop.copy()
    used_patches: set[int] = set()
    for hole, entries in zip(holes, entries_by_hole):
        exact = [entry for entry in entries if int(entry["match_count"]) == 4]
        assert len(exact) == 1, "each hole must have one unique patch-orientation match"
        patch_index = int(exact[0]["patch_index"])
        rot = int(exact[0]["rot"])
        assert patch_index not in used_patches, "one patch cannot fill two holes"
        used_patches.add(patch_index)

        rotated = np.rot90(patches[patch_index], rot)
        r0, c0, r1, c1 = hole["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = rotated

    return out


def _make_coarse_pattern(coarse_h: int, coarse_w: int, palette: list[int]) -> np.ndarray:
    coarse = np.full((coarse_h, coarse_w), random.choice(palette), dtype=int)
    for _ in range(random.randint(6, 10)):
        color = random.choice(palette)
        kind = random.choice(["rect", "row", "col", "corner"])
        if kind == "rect":
            r0 = random.randrange(coarse_h)
            r1 = random.randrange(r0, coarse_h)
            c0 = random.randrange(coarse_w)
            c1 = random.randrange(c0, coarse_w)
            coarse[r0 : r1 + 1, c0 : c1 + 1] = color
        elif kind == "row":
            r0 = random.randrange(coarse_h)
            r1 = random.randrange(r0, coarse_h)
            coarse[r0 : r1 + 1, :] = color
        elif kind == "col":
            c0 = random.randrange(coarse_w)
            c1 = random.randrange(c0, coarse_w)
            coarse[:, c0 : c1 + 1] = color
        else:
            r0 = random.randrange(coarse_h)
            c0 = random.randrange(coarse_w)
            coarse[r0:, c0:] = color

    for _ in range(random.randint(2, 4)):
        coarse[random.randrange(coarse_h), random.randrange(coarse_w)] = random.choice(palette)

    if len(np.unique(coarse)) < 3:
        return _make_coarse_pattern(coarse_h, coarse_w, palette)
    return coarse


def _make_full_pattern(main_h: int, main_w: int, palette: list[int]) -> np.ndarray:
    coarse = _make_coarse_pattern(main_h // 2, main_w // 2, palette)
    full = np.kron(coarse, np.ones((2, 2), dtype=int))
    if random.random() < 0.5:
        full = np.flipud(full)
    if random.random() < 0.5:
        full = np.fliplr(full)
    full = np.rot90(full, random.randrange(4))
    return np.asarray(full, dtype=int)


def _patch_is_expressive(patch: np.ndarray) -> bool:
    if len(np.unique(patch)) < 3:
        return False
    edges = [
        tuple(map(int, patch[0, :])),
        tuple(map(int, patch[-1, :])),
        tuple(map(int, patch[:, 0])),
        tuple(map(int, patch[:, -1])),
    ]
    return len(set(edges)) >= 3


def _bbox_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + padding < br0 or br1 + padding < ar0 or ac1 + padding < bc0 or bc1 + padding < ac0


def _select_holes(full: np.ndarray, hole_count: int) -> list[tuple[int, int, int, int]] | None:
    h, w = full.shape
    candidates: list[tuple[int, int, int, int]] = []
    for r0 in range(1, h - HOLE_SIZE, 2):
        for c0 in range(1, w - HOLE_SIZE, 2):
            bbox = (r0, c0, r0 + HOLE_SIZE - 1, c0 + HOLE_SIZE - 1)
            patch = crop_bbox(full, bbox)
            if _patch_is_expressive(patch):
                candidates.append(bbox)
    random.shuffle(candidates)

    chosen: list[tuple[int, int, int, int]] = []
    for bbox in candidates:
        if all(_bbox_separated(bbox, other, padding=2) for other in chosen):
            chosen.append(bbox)
            if len(chosen) == hole_count:
                return chosen
    return None


def _sample_distractor(full: np.ndarray, holes: list[tuple[int, int, int, int]]) -> np.ndarray | None:
    h, w = full.shape
    starts = [(r, c) for r in range(h - HOLE_SIZE + 1) for c in range(w - HOLE_SIZE + 1)]
    random.shuffle(starts)
    for r0, c0 in starts[:80]:
        bbox = (r0, c0, r0 + HOLE_SIZE - 1, c0 + HOLE_SIZE - 1)
        if any(not _bbox_separated(bbox, hole, padding=0) for hole in holes):
            continue
        patch = crop_bbox(full, bbox)
        if _patch_is_expressive(patch):
            return patch
    return None


def _place_patches(count: int, side: str) -> tuple[list[tuple[int, int]], int, int]:
    if side == "right":
        rows = random.choice([count, max(2, (count + 1) // 2)])
        cols = (count + rows - 1) // rows
    else:
        cols = random.choice([count, max(2, (count + 1) // 2)])
        rows = (count + cols - 1) // cols

    cell_h = HOLE_SIZE + 1
    cell_w = HOLE_SIZE + 1
    strip_h = rows * cell_h - 1
    strip_w = cols * cell_w - 1

    slots = [(rr * cell_h, cc * cell_w) for rr in range(rows) for cc in range(cols)]
    random.shuffle(slots)
    return slots[:count], strip_h, strip_w


def _build_input(
    damaged: np.ndarray,
    patches: list[np.ndarray],
) -> np.ndarray:
    main_h, main_w = damaged.shape
    patch_positions: list[tuple[int, int]]
    side = random.choice(["right", "bottom"])
    patch_positions, strip_h, strip_w = _place_patches(len(patches), side)
    outer = random.randint(0, 1)
    gap = 2

    if side == "right":
        total_h = max(main_h, strip_h) + 2 * outer
        total_w = main_w + gap + strip_w + 2 * outer
        main_r0 = random.randint(outer, total_h - outer - main_h)
        patch_r0 = random.randint(outer, total_h - outer - strip_h)
        main_c0 = outer
        patch_c0 = outer + main_w + gap
    else:
        total_h = main_h + gap + strip_h + 2 * outer
        total_w = max(main_w, strip_w) + 2 * outer
        main_r0 = outer
        patch_r0 = outer + main_h + gap
        main_c0 = random.randint(outer, total_w - outer - main_w)
        patch_c0 = random.randint(outer, total_w - outer - strip_w)

    grid = np.full((total_h, total_w), BG, dtype=int)
    grid[main_r0 : main_r0 + main_h, main_c0 : main_c0 + main_w] = damaged
    for patch, (dr, dc) in zip(patches, patch_positions):
        r0 = patch_r0 + dr
        c0 = patch_c0 + dc
        grid[r0 : r0 + HOLE_SIZE, c0 : c0 + HOLE_SIZE] = patch
    return grid


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        _bg, main_crop, holes, patches = _extract_problem(inp)
        entries_by_hole = _match_entries(main_crop, holes, patches)
    except AssertionError:
        return False

    if len(holes) < 2:
        return False

    solutions: list[tuple[int, int]] = []
    for entries in entries_by_hole:
        exact = [(int(item["patch_index"]), int(item["rot"])) for item in entries if int(item["match_count"]) == 4]
        if len(exact) != 1:
            return False
        solutions.append(exact[0])

    if len({patch_index for patch_index, _rot in solutions}) != len(holes):
        return False

    if not any(int(rot) != 0 for _patch_index, rot in solutions):
        return False

    # Each hole should have at least one tempting but incomplete seam match.
    for solution, entries in zip(solutions, entries_by_hole):
        if not any(
            0 < int(item["match_count"]) < 4
            and (int(item["patch_index"]), int(item["rot"])) != solution
            for item in entries
        ):
            return False

    # At least one solved patch should have a wrong rotation that still matches some seams.
    if not any(
        any(
            int(item["patch_index"]) == int(solution[0])
            and int(item["rot"]) != int(solution[1])
            and 0 < int(item["match_count"]) < 4
            for item in entries
        )
        for solution, entries in zip(solutions, entries_by_hole)
        if int(solution[1]) != 0
    ):
        return False

    # At least one wrong patch should partially match some hole, so patch identity matters too.
    if not any(
        any(
            int(item["patch_index"]) != int(solution[0]) and 0 < int(item["match_count"]) < 4
            for item in entries
        )
        for solution, entries in zip(solutions, entries_by_hole)
    ):
        return False

    try:
        solved = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(solved, expected))


def _generate_attempt() -> np.ndarray | None:
    hole_count = random.choice([2, 3])
    main_h = random.choice([14, 16, 18] if hole_count == 3 else [12, 14, 16, 18])
    main_w = random.choice([14, 16, 18] if hole_count == 3 else [12, 14, 16, 18])
    palette = random.sample(list(range(1, 10)), random.choice([4, 5]))

    full = _make_full_pattern(main_h, main_w, palette)
    holes = _select_holes(full, hole_count)
    if holes is None:
        return None

    hole_patches = [crop_bbox(full, hole) for hole in holes]
    distractor = _sample_distractor(full, holes)
    if distractor is None:
        return None

    damaged = full.copy()
    for r0, c0, r1, c1 in holes:
        damaged[r0 : r1 + 1, c0 : c1 + 1] = BG

    stored_patches = [np.rot90(patch, random.randrange(4)) for patch in hole_patches]
    stored_patches.append(np.rot90(distractor, random.randrange(4)))
    random.shuffle(stored_patches)

    inp = _build_input(damaged, stored_patches)
    if max(inp.shape) > 30:
        return None
    if not _semantic_checks(inp, full):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotated-patch docking instance")
