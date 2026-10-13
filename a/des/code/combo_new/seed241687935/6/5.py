from __future__ import annotations

import random
from collections import Counter

import numpy as np


_DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_PALETTE = list(range(1, 10))
_MAX_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in _DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not bool(bg[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            touches_boundary = False
            while stack:
                rr, cc = stack.pop()
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_boundary = True
                for dr, dc in _DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(bg[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_boundary:
                holes += 1
    return holes


def _is_connected(mask: np.ndarray) -> bool:
    return int(mask.sum()) > 0 and len(_components(mask)) == 1


def _mask_bbox_coords(mask: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    coords = []
    for r, c in np.argwhere(mask):
        coords.append((int(r0 + r), int(c0 + c)))
    return coords


def _inplace_transform(mask: np.ndarray, hole_count: int) -> tuple[np.ndarray | None, int, int]:
    if hole_count == 0:
        return np.fliplr(mask), 0, 0
    if hole_count == 1:
        rotated = np.rot90(mask, -1)
        dr = (mask.shape[0] - rotated.shape[0]) // 2
        dc = (mask.shape[1] - rotated.shape[1]) // 2
        return rotated, int(dr), int(dc)
    return None, 0, 0


def _paint(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[int(r), int(c)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each colored object by its hole count, then let the survivors fall and stack at the bottom.

    Input structure:
    - The background is black.
    - The foreground contains several single-color 4-connected objects separated by background.
    - Each object fits inside a bbox no larger than 6x6 and may contain 0, 1, or 2 enclosed black holes.

    Rule:
    - Count the number of enclosed black 4-connected regions inside each object's input mask.
    - 0 holes -> horizontally mirror the object inside its bbox; 1 hole -> rotate it 90 degrees clockwise around its bbox center; 2 holes -> erase it.
    - After these in-place transforms, keep each surviving object's horizontal position fixed and let it fall downward until it hits the bottom or an already settled object.

    Key insight:
    - The hidden classifier is the number of internal holes, and only after applying the class-dependent geometry do all remaining objects share the same gravity step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = 0

    objects = []
    for comp in _components(g != bg):
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1
        color = int(next(iter(colors)))

        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            mask[r - r0, c - c0] = True

        holes = _hole_count(mask)
        transformed, dr, dc = _inplace_transform(mask, holes)
        if transformed is None:
            continue

        top = int(r0 + dr)
        left = int(c0 + dc)
        coords = _mask_bbox_coords(transformed, top, left)
        objects.append(
            {
                "color": color,
                "coords": coords,
                "bottom": max(r for r, _ in coords),
                "top": min(r for r, _ in coords),
                "left": min(c for _, c in coords),
            }
        )

    objects.sort(key=lambda obj: (-obj["bottom"], -obj["top"], obj["left"]))

    out = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    for obj in objects:
        coords = obj["coords"]
        drop = 0
        while True:
            next_coords = [(r + drop + 1, c) for r, c in coords]
            if any(r >= h for r, _ in next_coords):
                break
            if any(occupied[r, c] for r, c in next_coords):
                break
            drop += 1
        final_coords = [(r + drop, c) for r, c in coords]
        for r, c in final_coords:
            if 0 <= r < h and 0 <= c < w:
                occupied[r, c] = True
        _paint(out, final_coords, obj["color"])

    return out


def _random_d4(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.array(out, dtype=bool, copy=True)


def _trim_mask(mask: np.ndarray, target_holes: int, min_area: int, keep_bbox: bool) -> np.ndarray:
    out = mask.copy()
    attempts = random.randint(1, 8)
    for _ in range(attempts):
        boundary = []
        hh, ww = out.shape
        for r, c in np.argwhere(out):
            rr = int(r)
            cc = int(c)
            if rr in (0, hh - 1) or cc in (0, ww - 1):
                boundary.append((rr, cc))
                continue
            if any(not bool(out[rr + dr, cc + dc]) for dr, dc in _DIR4):
                boundary.append((rr, cc))
        random.shuffle(boundary)
        changed = False
        for r, c in boundary:
            trial = out.copy()
            trial[r, c] = False
            if int(trial.sum()) < int(min_area):
                continue
            if keep_bbox:
                if not trial[0].any() or not trial[-1].any() or not trial[:, 0].any() or not trial[:, -1].any():
                    continue
            if not _is_connected(trial):
                continue
            if _hole_count(trial) != target_holes:
                continue
            out = trial
            changed = True
            break
        if not changed:
            break
    return _crop_mask(out)


def _make_zero_hole_mask(require_change: bool) -> np.ndarray:
    for _ in range(200):
        hh = random.randint(2, 6)
        ww = random.randint(2, 6)
        base = np.ones((hh, ww), dtype=bool)
        mask = _trim_mask(base, target_holes=0, min_area=max(4, (hh * ww + 1) // 2), keep_bbox=False)
        mask = _random_d4(mask)
        mask = _crop_mask(mask)
        if mask.shape[0] > 6 or mask.shape[1] > 6:
            continue
        if min(mask.shape) < 2 or int(mask.sum()) < 4:
            continue
        if require_change and np.array_equal(mask, np.fliplr(mask)):
            continue
        return mask
    raise RuntimeError("failed to build 0-hole mask")


def _make_one_hole_mask(require_change: bool) -> np.ndarray:
    for _ in range(300):
        size = random.randint(4, 6)
        hole_h = random.randint(1, size - 2)
        hole_w = random.randint(1, size - 2)
        hole_r = random.randint(1, size - hole_h - 1)
        hole_c = random.randint(1, size - hole_w - 1)

        base = np.ones((size, size), dtype=bool)
        base[hole_r : hole_r + hole_h, hole_c : hole_c + hole_w] = False
        if _hole_count(base) != 1:
            continue

        mask = _trim_mask(base, target_holes=1, min_area=max(8, size * size // 2), keep_bbox=True)
        mask = _random_d4(mask)
        if mask.shape[0] != mask.shape[1] or mask.shape[0] > 6:
            continue
        if require_change and np.array_equal(mask, np.rot90(mask, -1)):
            continue
        return mask
    raise RuntimeError("failed to build 1-hole mask")


def _make_two_hole_mask() -> np.ndarray:
    for _ in range(400):
        hh = random.randint(4, 6)
        ww = random.randint(4, 6)
        base = np.ones((hh, ww), dtype=bool)
        for _ in range(2):
            hole_h = random.randint(1, min(2, hh - 2))
            hole_w = random.randint(1, min(2, ww - 2))
            hole_r = random.randint(1, hh - hole_h - 1)
            hole_c = random.randint(1, ww - hole_w - 1)
            base[hole_r : hole_r + hole_h, hole_c : hole_c + hole_w] = False
        if _hole_count(base) != 2:
            continue

        mask = _trim_mask(base, target_holes=2, min_area=max(8, hh * ww // 2), keep_bbox=False)
        mask = _random_d4(mask)
        mask = _crop_mask(mask)
        if mask.shape[0] > 6 or mask.shape[1] > 6:
            continue
        if _hole_count(mask) != 2:
            continue
        return mask
    raise RuntimeError("failed to build 2-hole mask")


def _build_mask(branch: int, require_change: bool) -> np.ndarray:
    if branch == 0:
        return _make_zero_hole_mask(require_change=require_change)
    if branch == 1:
        return _make_one_hole_mask(require_change=require_change)
    return _make_two_hole_mask()


def _coords_clear(occupied: np.ndarray, coords: list[tuple[int, int]], pad: int = 1) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        r0 = max(0, r - pad)
        r1 = min(h, r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(w, c + pad + 1)
        if bool(occupied[r0:r1, c0:c1].any()):
            return False
    return True


def _mark(occupied: np.ndarray, coords: list[tuple[int, int]]) -> None:
    for r, c in coords:
        occupied[int(r), int(c)] = True


def _preferred_rows(min_r: int, max_r: int) -> list[int]:
    if max_r < min_r:
        return []
    rows = list(range(min_r, max_r + 1))
    split = min(max_r, min_r + max(2, (max_r - min_r) * 2 // 3))
    head = [r for r in rows if r <= split]
    tail = [r for r in rows if r > split]
    random.shuffle(head)
    random.shuffle(tail)
    return head + tail


def _candidate_positions(spec: dict, height: int, width: int, preferred_cols: list[int]) -> list[tuple[int, int]]:
    ih, iw = spec["mask"].shape
    th, tw = spec["trans_shape"]
    dr, dc = spec["delta"]

    min_r = max(0, -dr)
    max_r = min(height - ih, height - th - dr)
    min_c = max(0, -dc)
    max_c = min(width - iw, width - tw - dc)
    if spec["branch"] != 2:
        max_r = min(max_r, height - ih - 2)
    else:
        max_r = min(max_r, height - ih - 1)
    if max_r < min_r or max_c < min_c:
        return []

    rows = _preferred_rows(min_r, max_r)
    cols = list(range(min_c, max_c + 1))
    random.shuffle(cols)
    cols.sort(key=lambda c: min(abs(c - pref) for pref in preferred_cols))

    return [(r, c) for r in rows for c in cols]


def _after_local_grid(height: int, width: int, placements: list[dict]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for item in placements:
        if item["branch"] == 2:
            continue
        coords = _mask_bbox_coords(item["transformed"], item["r0"] + item["delta"][0], item["c0"] + item["delta"][1])
        _paint(grid, coords, item["color"])
    return grid


def _generate_attempt() -> np.ndarray | None:
    branches = [0, 1, 2]
    if random.random() < 0.6:
        branches.append(random.choice([0, 1, 2]))
    random.shuffle(branches)

    colors = random.sample(_PALETTE, len(branches))
    specs: list[dict] = []
    for idx, branch in enumerate(branches):
        require_change = branch in (0, 1)
        mask = _build_mask(branch, require_change=require_change)
        holes = _hole_count(mask)
        if holes != branch:
            return None
        transformed, dr, dc = _inplace_transform(mask, holes)
        specs.append(
            {
                "branch": branch,
                "color": colors[idx],
                "mask": mask,
                "transformed": transformed,
                "delta": (dr, dc),
                "trans_shape": (0, 0) if transformed is None else tuple(map(int, transformed.shape)),
                "area": int(mask.sum()),
            }
        )

    min_height = sum(spec["mask"].shape[0] for spec in specs) + len(specs) + 2
    if min_height > 30:
        return None
    height = random.randint(max(14, min_height), min(30, max(18, min_height + 5)))

    max_width = max(spec["mask"].shape[1] for spec in specs)
    width = random.randint(max(12, max_width + 4), min(30, max(18, max_width + 10)))

    survivors = [i for i, spec in enumerate(specs) if spec["branch"] != 2]
    if len(survivors) < 2:
        return None

    max_survivor_width = max(specs[i]["trans_shape"][1] for i in survivors)
    stack_col = random.randint(0, width - max_survivor_width)
    alt_col = random.randint(0, width - max_survivor_width)

    first_zero = next(i for i, spec in enumerate(specs) if spec["branch"] == 0)
    first_one = next(i for i, spec in enumerate(specs) if spec["branch"] == 1)
    stack_pair = {first_zero, first_one}

    preferences: dict[int, list[int]] = {}
    for i, spec in enumerate(specs):
        if i in stack_pair:
            preferences[i] = [stack_col]
        elif spec["branch"] != 2:
            preferences[i] = [stack_col] if random.random() < 0.5 else [alt_col]
        else:
            preferences[i] = [random.randint(0, width - spec["mask"].shape[1])]

    order = sorted(
        range(len(specs)),
        key=lambda i: (specs[i]["branch"] != 2, specs[i]["area"], specs[i]["mask"].shape[0] * specs[i]["mask"].shape[1]),
        reverse=True,
    )

    occupied_input = np.zeros((height, width), dtype=bool)
    occupied_transformed = np.zeros((height, width), dtype=bool)
    placements: list[dict] = []

    for idx in order:
        spec = specs[idx]
        placed = False
        for r0, c0 in _candidate_positions(spec, height, width, preferences[idx]):
            input_coords = _mask_bbox_coords(spec["mask"], r0, c0)
            if not _coords_clear(occupied_input, input_coords, pad=1):
                continue

            transformed_coords: list[tuple[int, int]] = []
            if spec["branch"] != 2:
                transformed_coords = _mask_bbox_coords(
                    spec["transformed"],
                    r0 + spec["delta"][0],
                    c0 + spec["delta"][1],
                )
                if not _coords_clear(occupied_transformed, transformed_coords, pad=1):
                    continue

            _mark(occupied_input, input_coords)
            if transformed_coords:
                _mark(occupied_transformed, transformed_coords)
            placement = dict(spec)
            placement["r0"] = int(r0)
            placement["c0"] = int(c0)
            placements.append(placement)
            placed = True
            break
        if not placed:
            return None

    inp = np.zeros((height, width), dtype=int)
    for item in placements:
        coords = _mask_bbox_coords(item["mask"], item["r0"], item["c0"])
        _paint(inp, coords, item["color"])

    if len(_components(inp != 0)) != len(specs):
        return None

    if Counter(_hole_count(item["mask"]) for item in placements) != Counter(branches):
        return None

    after_local = _after_local_grid(height, width, placements)
    out = main(inp)
    if np.array_equal(inp, out):
        return None

    moved_down = False
    for item in placements:
        if item["branch"] == 2:
            if int(np.sum(out == item["color"])) != 0:
                return None
            continue
        before_rows = np.where(after_local == item["color"])[0]
        after_rows = np.where(out == item["color"])[0]
        if before_rows.size == 0 or after_rows.size == 0:
            return None
        if int(after_rows.min()) > int(before_rows.min()):
            moved_down = True
    if not moved_down:
        return None

    if int(np.sum(out == specs[first_zero]["color"])) == 0 or int(np.sum(out == specs[first_one]["color"])) == 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
