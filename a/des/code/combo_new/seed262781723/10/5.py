import random
from collections import deque

import numpy as np


BG = 0
AZURE = 8
MAROON = 9
BUILDING_COLORS = (1, 2, 3, 4, 5, 6, 7)
ORTHO_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

GEN_RETRIES = 300
STAMP_RETRIES = 120
BUILDING_RETRIES = 120
PLACEMENT_RETRIES = 300


def _component_dicts_from_mask(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ORTHO_DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            r1 = int(arr[:, 0].max())
            c0 = int(arr[:, 1].min())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    return _component_dicts_from_mask(np.asarray(grid, dtype=int) == int(color))


def _dist_to_bbox(r: int, c: int, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    dr = 0 if r0 <= r <= r1 else min(abs(r - r0), abs(r - r1))
    dc = 0 if c0 <= c <= c1 else min(abs(c - c0), abs(c - c1))
    return int(dr + dc)


def _center_from_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return (r0 + r1) // 2, (c0 + c1) // 2


def _decode_legend(grid: np.ndarray) -> dict[str, dict]:
    stamp_components = _components_of_color(grid, MAROON)
    assert len(stamp_components) == 2, "expected exactly two maroon stamp components"

    azure_coords = np.argwhere(grid == AZURE)
    assert azure_coords.shape[0] >= 4, "expected azure cue dots next to the stamps"

    row_groups = []
    for row in sorted(set(map(int, azure_coords[:, 0]))):
        cells = [(int(r), int(c)) for r, c in azure_coords if int(r) == row]
        if len(cells) >= 2:
            row_groups.append(cells)
    col_groups = []
    for col in sorted(set(map(int, azure_coords[:, 1]))):
        cells = [(int(r), int(c)) for r, c in azure_coords if int(c) == col]
        if len(cells) >= 2:
            col_groups.append(cells)

    assert len(row_groups) == 1 and len(col_groups) == 1, "azure dots must form one cue row and one cue column"
    cue_groups = [("h", row_groups[0]), ("v", col_groups[0])]

    best_perm = None
    best_cost = None
    for idx0 in range(2):
        idx1 = 1 - idx0
        perm = (idx0, idx1)
        cost = 0
        for group_idx, (_, cells) in enumerate(cue_groups):
            bbox = stamp_components[perm[group_idx]]["bbox"]
            cost += sum(_dist_to_bbox(r, c, bbox) for r, c in cells)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_perm = perm

    assert best_perm is not None
    decoded: dict[str, dict] = {}
    for (orientation, _cells), comp_idx in zip(cue_groups, best_perm):
        comp = stamp_components[comp_idx]
        r0, c0, r1, c1 = comp["bbox"]
        stamp_mask = grid[r0 : r1 + 1, c0 : c1 + 1] == MAROON
        anchor = (stamp_mask.shape[0] // 2, stamp_mask.shape[1] // 2)
        assert stamp_mask[anchor], "stamp center pixel must be maroon"
        decoded[orientation] = {"mask": stamp_mask, "anchor": anchor}

    assert set(decoded) == {"h", "v"}, "legend must encode horizontal and vertical cases"
    return decoded


def _window_orientation(subgrid: np.ndarray) -> str:
    bg_comps = _component_dicts_from_mask(subgrid == BG)
    assert bg_comps, "building must contain background windows"

    centers2_r = []
    centers2_c = []
    for comp in bg_comps:
        r0, c0, r1, c1 = comp["bbox"]
        assert r0 > 0 and c0 > 0 and r1 < subgrid.shape[0] - 1 and c1 < subgrid.shape[1] - 1, (
            "windows must stay strictly inside the building rectangle"
        )
        centers2_r.append(r0 + r1)
        centers2_c.append(c0 + c1)

    if len(set(centers2_r)) == 1 and len(set(centers2_c)) > 1:
        return "h"
    if len(set(centers2_c)) == 1 and len(set(centers2_r)) > 1:
        return "v"
    raise AssertionError("window centers must form one horizontal row or one vertical column")


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    stamps = _decode_legend(g)

    building_count = 0
    for color in sorted(set(map(int, np.unique(g))) - {BG, AZURE, MAROON}):
        for comp in _components_of_color(g, color):
            r0, c0, r1, c1 = comp["bbox"]
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            assert np.all((sub == BG) | (sub == color)), "building bbox must contain only its color and windows"
            assert np.all(sub[0, :] == color)
            assert np.all(sub[-1, :] == color)
            assert np.all(sub[:, 0] == color)
            assert np.all(sub[:, -1] == color)

            orientation = _window_orientation(sub)
            stamp = stamps[orientation]
            center_r = (r0 + r1) // 2
            center_c = (c0 + c1) // 2
            building_count += 1

            for sr, sc in np.argwhere(stamp["mask"]):
                rr = center_r + int(sr) - stamp["anchor"][0]
                cc = center_c + int(sc) - stamp["anchor"][1]
                if r0 <= rr <= r1 and c0 <= cc <= c1 and g[rr, cc] == color:
                    out[rr, cc] = MAROON

    assert building_count > 0, "expected at least one building"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Decode the corner legend and stamp each building center with the template selected by its window direction.

    Input structure:
    - The background is black.
    - A corner legend contains exactly two maroon stamp shapes, each paired with nearby azure cue dots arranged either in one row or in one column.
    - Elsewhere, several large monochrome rectangular buildings of distinct colors contain small black square windows aligned as a single horizontal row or a single vertical column.

    Rule:
    - Read the legend by matching each maroon stamp to its adjacent azure cue: a horizontal cue row labels the stamp for horizontally arranged windows, and a vertical cue column labels the stamp for vertically arranged windows.
    - For every colored building rectangle, inspect the internal black windows to decide whether they form a horizontal row or a vertical column.
    - Place the corresponding maroon stamp so that its center pixel aligns with the building bbox center, clipping any out-of-bounds stamp cells, and only paint stamp pixels onto the building's own colored cells.

    Key insight:
    - The corner sprites are a two-step legend: azure dot orientation decodes which maroon stamp belongs to which case, and the buildings' window orientation chooses which decoded stamp to apply.
    """
    return _solve(grid)


def _draw_path(mask: np.ndarray, start: tuple[int, int], target: tuple[int, int]) -> None:
    r, c = start
    tr, tc = target
    if random.random() < 0.5:
        while r != tr:
            mask[r, c] = True
            r += 1 if tr > r else -1
        while c != tc:
            mask[r, c] = True
            c += 1 if tc > c else -1
    else:
        while c != tc:
            mask[r, c] = True
            c += 1 if tc > c else -1
        while r != tr:
            mask[r, c] = True
            r += 1 if tr > r else -1
    mask[tr, tc] = True


def _random_stamp() -> np.ndarray:
    for _ in range(STAMP_RETRIES):
        h = random.choice((3, 5))
        w = random.choice((3, 5))
        center = (h // 2, w // 2)
        mask = np.zeros((h, w), dtype=bool)
        mask[center] = True

        targets = [
            (0, random.randint(0, w - 1)),
            (h - 1, random.randint(0, w - 1)),
            (random.randint(0, h - 1), 0),
            (random.randint(0, h - 1), w - 1),
        ]
        for target in targets:
            _draw_path(mask, center, target)

        for _ in range(random.randint(0, max(2, h + w - 4))):
            occupied = np.argwhere(mask)
            rr, cc = occupied[random.randrange(len(occupied))]
            dr, dc = random.choice(ORTHO_DIRS)
            nr, nc = int(rr + dr), int(cc + dc)
            if 0 <= nr < h and 0 <= nc < w:
                mask[nr, nc] = True

        if mask.sum() == h * w:
            continue
        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        return mask
    raise RuntimeError("failed to generate a stamp mask")


def _unit_patch(stamp_mask: np.ndarray, orientation: str, cue_count: int) -> np.ndarray:
    sh, sw = stamp_mask.shape
    if orientation == "h":
        unit_h = sh
        unit_w = sw + 2 * cue_count
        patch = np.full((unit_h, unit_w), BG, dtype=int)
        patch[:sh, :sw][stamp_mask] = MAROON
        cue_cols = [sw + 1 + 2 * i for i in range(cue_count)]
        patch[unit_h // 2, cue_cols] = AZURE
        return patch

    cue_h = 2 * cue_count - 1
    unit_h = max(sh, cue_h)
    unit_w = sw + 2
    patch = np.full((unit_h, unit_w), BG, dtype=int)
    stamp_top = (unit_h - sh) // 2
    patch[stamp_top : stamp_top + sh, :sw][stamp_mask] = MAROON
    cue_rows = [2 * i for i in range(cue_count)]
    patch[cue_rows, sw + 1] = AZURE
    return patch


def _legend_patch(h_stamp: np.ndarray, v_stamp: np.ndarray) -> np.ndarray:
    units = [
        _unit_patch(h_stamp, "h", random.randint(3, 4)),
        _unit_patch(v_stamp, "v", random.randint(3, 4)),
    ]
    random.shuffle(units)

    gap = random.randint(2, 3)
    patch_h = sum(unit.shape[0] for unit in units) + gap
    patch_w = max(unit.shape[1] for unit in units)
    patch = np.full((patch_h, patch_w), BG, dtype=int)

    top = 0
    for unit in units:
        left = 0
        mask = unit != BG
        block = patch[top : top + unit.shape[0], left : left + unit.shape[1]]
        block[mask] = unit[mask]
        top += unit.shape[0] + gap
    return patch


def _paste_in_corner(grid: np.ndarray, patch: np.ndarray, corner: str) -> tuple[int, int, int, int]:
    placed = patch
    if corner in {"tr", "br"}:
        placed = np.fliplr(placed)
    if corner in {"bl", "br"}:
        placed = np.flipud(placed)

    margin_r = random.randint(0, 1)
    margin_c = random.randint(0, 1)
    if corner in {"tl", "tr"}:
        r0 = margin_r
    else:
        r0 = grid.shape[0] - placed.shape[0] - margin_r
    if corner in {"tl", "bl"}:
        c0 = margin_c
    else:
        c0 = grid.shape[1] - placed.shape[1] - margin_c

    mask = placed != BG
    block = grid[r0 : r0 + placed.shape[0], c0 : c0 + placed.shape[1]]
    assert np.all(block[mask] == BG)
    block[mask] = placed[mask]
    return r0, c0, placed.shape[0], placed.shape[1]


def _random_building_patch(color: int, orientation: str) -> np.ndarray:
    for _ in range(BUILDING_RETRIES):
        window_size = random.choice((1, 1, 2))
        window_count = random.randint(2, 4)
        gap = random.randint(1, 2)
        span = window_count * window_size + (window_count - 1) * gap

        if orientation == "h":
            min_h = window_size + 2
            min_w = span + 2
            max_h = 10
            max_w = 13
            if min_h > max_h or min_w > max_w:
                continue
            h = random.randint(min_h, max_h)
            w = random.randint(min_w, max_w)
            row0 = random.randint(1, h - window_size - 1)
            col0 = random.randint(1, w - span - 1)
            patch = np.full((h, w), color, dtype=int)
            for idx in range(window_count):
                cc = col0 + idx * (window_size + gap)
                patch[row0 : row0 + window_size, cc : cc + window_size] = BG
        else:
            min_w = window_size + 2
            min_h = span + 2
            max_w = 10
            max_h = 13
            if min_h > max_h or min_w > max_w:
                continue
            h = random.randint(min_h, max_h)
            w = random.randint(min_w, max_w)
            row0 = random.randint(1, h - span - 1)
            col0 = random.randint(1, w - window_size - 1)
            patch = np.full((h, w), color, dtype=int)
            for idx in range(window_count):
                rr = row0 + idx * (window_size + gap)
                patch[rr : rr + window_size, col0 : col0 + window_size] = BG

        comps = _components_of_color(patch, color)
        if len(comps) != 1:
            continue
        if _window_orientation(patch) != orientation:
            continue
        return patch
    raise RuntimeError("failed to generate a building patch")


def _place_building(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray) -> bool:
    gh, gw = grid.shape
    ph, pw = patch.shape
    for _ in range(PLACEMENT_RETRIES):
        r0 = random.randint(0, gh - ph)
        c0 = random.randint(0, gw - pw)
        r1 = r0 + ph - 1
        c1 = c0 + pw - 1
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(gh - 1, r1 + 1)
        cc1 = min(gw - 1, c1 + 1)
        if occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
            continue
        block = grid[r0 : r0 + ph, c0 : c0 + pw]
        assert np.all(block == BG)
        block[:, :] = patch
        occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    stamp_a = _random_stamp()
    stamp_b = _random_stamp()
    for _ in range(STAMP_RETRIES):
        if stamp_a.shape != stamp_b.shape or not np.array_equal(stamp_a, stamp_b):
            break
        stamp_b = _random_stamp()
    else:
        return None

    if random.random() < 0.5:
        h_stamp, v_stamp = stamp_a, stamp_b
    else:
        h_stamp, v_stamp = stamp_b, stamp_a

    legend = _legend_patch(h_stamp, v_stamp)
    min_h = max(18, legend.shape[0] + 6)
    min_w = max(18, legend.shape[1] + 8)
    if min_h > 30 or min_w > 30:
        return None

    grid_h = random.randint(min_h, 30)
    grid_w = random.randint(min_w, 30)
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    corner = random.choice(("tl", "tr", "bl", "br"))
    lr0, lc0, lh, lw = _paste_in_corner(grid, legend, corner)
    rr0 = max(0, lr0 - 1)
    cc0 = max(0, lc0 - 1)
    rr1 = min(grid_h - 1, lr0 + lh)
    cc1 = min(grid_w - 1, lc0 + lw)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True

    building_count = random.choice((2, 3, 4))
    orientations = ["h", "v"] + [random.choice(("h", "v")) for _ in range(building_count - 2)]
    random.shuffle(orientations)
    colors = random.sample(BUILDING_COLORS, building_count)

    for color, orientation in zip(colors, orientations):
        patch = _random_building_patch(color, orientation)
        if not _place_building(grid, occupied, patch):
            return None

    out = _solve(grid)
    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
