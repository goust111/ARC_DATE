from __future__ import annotations

from collections import deque
import random

import numpy as np


_BG = 0
_COLOR_POOL = tuple(range(1, 10))
_CORNERS = ("tl", "tr", "bl", "br")
_OPPOSITE = {"tl": "br", "tr": "bl", "bl": "tr", "br": "tl"}
_MAX_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _tight_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (int(r0), int(c0), int(r1), int(c1))


def _rotation_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        key = (tuple(map(int, rotated.shape)), rotated.tobytes())
        if key in seen:
            continue
        seen.add(key)
        variants.append(rotated)
    return variants


def _internal_holes(patch: np.ndarray, bg: int) -> list[dict]:
    holes: list[dict] = []
    ph, pw = patch.shape
    for coords in _components(patch == int(bg)):
        if any(
            int(r) == 0 or int(c) == 0 or int(r) == ph - 1 or int(c) == pw - 1
            for r, c in coords
        ):
            continue
        mask, (r0, c0, r1, c1) = _tight_mask(coords)
        holes.append(
            {
                "coords": coords,
                "mask": mask,
                "area": int(coords.shape[0]),
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
            }
        )
    return holes


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    target = grid[int(r0) : int(r0) + h, int(c0) : int(c0) + w]
    mask = patch != int(bg)
    assert target.shape == patch.shape
    assert not bool(np.any((target != int(bg)) & mask))
    target[mask] = patch[mask]


def _rotation_key_count(mask: np.ndarray) -> int:
    keys = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        keys.add((tuple(map(int, rotated.shape)), rotated.tobytes()))
    return len(keys)


def _random_piece_mask(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        box_h = rng.randint(2, 5)
        box_w = rng.randint(2, 5)
        max_area = min(box_h * box_w - 1, 7)
        if max_area < 3:
            continue
        area = rng.randint(3, max_area)
        mask = np.zeros((box_h, box_w), dtype=bool)
        start = (rng.randrange(box_h), rng.randrange(box_w))
        mask[start] = True
        while int(mask.sum()) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in np.argwhere(mask):
                rr = int(r)
                cc = int(c)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < box_h and 0 <= nc < box_w and not mask[nr, nc]:
                        frontier.append((int(nr), int(nc)))
            if not frontier:
                break
            nr, nc = rng.choice(frontier)
            mask[nr, nc] = True
        if int(mask.sum()) != area:
            continue
        coords = np.argwhere(mask)
        r0, c0, r1, c1 = _bbox_coords(coords)
        tight = mask[r0 : r1 + 1, c0 : c1 + 1]
        if tight.shape[0] < 2 or tight.shape[1] < 2:
            continue
        if _rotation_key_count(tight) != 4:
            continue
        return tight
    raise RuntimeError("failed to generate a rotation-asymmetric piece")


def _corner_offset(corner: str, cell_h: int, cell_w: int, patch_h: int, patch_w: int) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, int(cell_w - patch_w)
    if corner == "bl":
        return int(cell_h - patch_h), 0
    return int(cell_h - patch_h), int(cell_w - patch_w)


def _build_pair(rng: random.Random, color: int) -> dict:
    base_mask = _random_piece_mask(rng)
    hole_mask = np.rot90(base_mask, rng.randrange(4))
    small_mask = np.rot90(hole_mask, rng.randrange(4))

    hole_h, hole_w = map(int, hole_mask.shape)
    hole_area = int(hole_mask.sum())

    top = rng.randint(1, 3)
    bottom = rng.randint(1, 3)
    left = rng.randint(1, 3)
    right = rng.randint(1, 3)
    outer_h = int(hole_h + top + bottom)
    outer_w = int(hole_w + left + right)
    while outer_h * outer_w - hole_area <= hole_area:
        side = rng.choice(("top", "bottom", "left", "right"))
        if side == "top":
            top += 1
        elif side == "bottom":
            bottom += 1
        elif side == "left":
            left += 1
        else:
            right += 1
        outer_h = int(hole_h + top + bottom)
        outer_w = int(hole_w + left + right)

    large_input = np.full((outer_h, outer_w), int(color), dtype=int)
    large_input[top : top + hole_h, left : left + hole_w][hole_mask] = int(_BG)
    large_output = np.full((outer_h, outer_w), int(color), dtype=int)

    small_patch = np.full(small_mask.shape, int(_BG), dtype=int)
    small_patch[small_mask] = int(color)

    gap_r = rng.randint(1, 3)
    gap_c = rng.randint(1, 3)
    cell_h = int(outer_h + small_patch.shape[0] + gap_r)
    cell_w = int(outer_w + small_patch.shape[1] + gap_c)
    large_corner = rng.choice(_CORNERS)
    small_corner = _OPPOSITE[large_corner]
    large_off = _corner_offset(large_corner, cell_h, cell_w, outer_h, outer_w)
    small_off = _corner_offset(
        small_corner,
        cell_h,
        cell_w,
        int(small_patch.shape[0]),
        int(small_patch.shape[1]),
    )

    return {
        "color": int(color),
        "large_input": large_input,
        "large_output": large_output,
        "small_input": small_patch,
        "cell_h": int(cell_h),
        "cell_w": int(cell_w),
        "cell_area": int(cell_h * cell_w),
        "outer_area": int(outer_h * outer_w),
        "large_off": (int(large_off[0]), int(large_off[1])),
        "small_off": (int(small_off[0]), int(small_off[1])),
    }


def _feasible_dims(pairs: list[dict]) -> list[tuple[int, int]]:
    min_h = max(int(pair["cell_h"]) + 2 for pair in pairs)
    min_w = max(int(pair["cell_w"]) + 2 for pair in pairs)
    total_cell_area = sum(int(pair["cell_area"]) for pair in pairs)
    total_fg = sum(int(pair["outer_area"]) for pair in pairs)

    dims: list[tuple[int, int]] = []
    for h in range(max(12, min_h), 31):
        for w in range(max(12, min_w), 31):
            if total_fg * 2 >= h * w:
                continue
            if total_cell_area + 6 * len(pairs) > h * w:
                continue
            dims.append((int(h), int(w)))
    return dims


def _place_cells(pairs: list[dict], h: int, w: int, rng: random.Random) -> list[tuple[int, int]] | None:
    occupied = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(pairs)
    order = list(range(len(pairs)))
    rng.shuffle(order)
    order.sort(key=lambda idx: pairs[idx]["cell_area"], reverse=True)

    for idx in order:
        cell_h = int(pairs[idx]["cell_h"])
        cell_w = int(pairs[idx]["cell_w"])
        valid: list[tuple[int, int]] = []
        for r0 in range(1, h - cell_h):
            for c0 in range(1, w - cell_w):
                r1 = int(r0 - 1)
                c1 = int(c0 - 1)
                r2 = int(r0 + cell_h + 1)
                c2 = int(c0 + cell_w + 1)
                if occupied[r1:r2, c1:c2].any():
                    continue
                valid.append((int(r0), int(c0)))
        if not valid:
            return None
        placements[idx] = rng.choice(valid)
        pr0, pc0 = placements[idx]
        occupied[pr0 : pr0 + cell_h, pc0 : pc0 + cell_w] = True

    return [(int(r), int(c)) for r, c in placements if r is not None and c is not None]


def _assemble_grids(pairs: list[dict], placements: list[tuple[int, int]], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, int(_BG), dtype=int)
    out = np.full(shape, int(_BG), dtype=int)

    for pair, (base_r, base_c) in zip(pairs, placements):
        large_r = int(base_r + pair["large_off"][0])
        large_c = int(base_c + pair["large_off"][1])
        small_r = int(base_r + pair["small_off"][0])
        small_c = int(base_c + pair["small_off"][1])
        _paint_patch(inp, pair["large_input"], large_r, large_c, _BG)
        _paint_patch(inp, pair["small_input"], small_r, small_c, _BG)
        _paint_patch(out, pair["large_output"], large_r, large_c, _BG)

    return inp, out


def _valid_color_pairing(grid: np.ndarray, bg: int) -> bool:
    for color in np.unique(grid):
        color = int(color)
        if color == int(bg):
            continue
        if len(_components(grid == color)) != 2:
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert each small same-color fragment into the matching hole inside the larger fragment.

    Input structure:
    - The background is the most common color.
    - Each non-background color appears in exactly two 4-connected monochrome objects.
    - In each same-color pair, the larger object contains a background-colored interior hole, and the smaller object has the same shape up to a 90-degree rotation.

    Rule:
    - For each non-background color, find its two connected components and identify the larger and smaller one by area.
    - Locate the interior background hole inside the larger object's bbox.
    - Rotate the smaller object's shape by 0/90/180/270 degrees until it matches that hole, then move it into the hole.
    - Clear the smaller object from its original position and fill the hole with the larger object's color.

    Key insight:
    - Color reveals which two objects belong together, and the decisive second step is matching the smaller object's shape to the larger object's enclosed hole under rotation.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(_get_bg(g))
    out = g.copy()

    for color in sorted(int(v) for v in np.unique(g) if int(v) != int(bg)):
        comps = _components(g == int(color))
        assert len(comps) == 2, "each non-background color must form exactly one pair"

        infos = []
        for coords in comps:
            mask, (r0, c0, r1, c1) = _tight_mask(coords)
            infos.append(
                {
                    "coords": coords,
                    "mask": mask,
                    "area": int(coords.shape[0]),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                }
            )
        infos.sort(key=lambda item: (int(item["area"]), int(item["mask"].shape[0] * item["mask"].shape[1])))
        small, large = infos[0], infos[1]
        assert int(large["area"]) > int(small["area"]), "the larger fragment must be strictly larger"

        large_patch = g[large["r0"] : large["r1"] + 1, large["c0"] : large["c1"] + 1]
        holes = _internal_holes(large_patch, bg)
        assert holes, "the larger fragment must contain an interior hole"

        matches = []
        for hole in holes:
            if int(hole["area"]) != int(small["area"]):
                continue
            if any(
                rotated.shape == hole["mask"].shape and np.array_equal(rotated, hole["mask"])
                for rotated in _rotation_variants(small["mask"])
            ):
                matches.append(hole)
        assert len(matches) == 1, "expected a unique hole matching the smaller fragment"
        hole = matches[0]

        out[small["coords"][:, 0], small["coords"][:, 1]] = int(bg)
        fill = out[large["r0"] + hole["r0"] : large["r0"] + hole["r1"] + 1, large["c0"] + hole["c0"] : large["c0"] + hole["c1"] + 1]
        fill[hole["mask"]] = int(color)

    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    pair_count = rng.randint(1, 3)
    colors = rng.sample(_COLOR_POOL, pair_count)
    pairs = [_build_pair(rng, int(color)) for color in colors]

    dims = _feasible_dims(pairs)
    if not dims:
        return None
    h, w = rng.choice(dims)
    placements = _place_cells(pairs, h, w, rng)
    if placements is None or len(placements) != len(pairs):
        return None

    inp, expected = _assemble_grids(pairs, placements, (int(h), int(w)))
    if _get_bg(inp) != int(_BG):
        return None
    if not _valid_color_pairing(inp, _BG):
        return None
    if int(np.count_nonzero(inp == int(_BG))) <= inp.size // 2:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
