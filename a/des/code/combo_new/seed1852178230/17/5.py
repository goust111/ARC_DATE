import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400
SHAPE_ATTEMPTS = 200
MARKER_ATTEMPTS = 120
PLACEMENT_GAP = 1
GRID_MARGIN = 1
MODES = ("distinct", "tie_pair", "tie_plus_leader")


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
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _interior_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not mask[r, c]:
                continue
            if (
                mask[r - 1, c]
                and mask[r + 1, c]
                and mask[r, c - 1]
                and mask[r, c + 1]
            ):
                cells.append((r, c))
    return cells


def _color_mask_connected(mask: np.ndarray, grey_cells: set[tuple[int, int]]) -> bool:
    color_mask = mask.copy()
    for r, c in grey_cells:
        color_mask[r, c] = False
    comps = _components(color_mask)
    return len(comps) == 1


def _pick_markers(mask: np.ndarray, count: int) -> set[tuple[int, int]] | None:
    if count == 0:
        return set()

    interior = _interior_cells(mask)
    if len(interior) < count:
        return None
    h, w = mask.shape
    interior_set = set(interior)

    for _ in range(MARKER_ATTEMPTS):
        if count == 1 or random.random() < 0.5:
            markers = set(random.sample(interior, count))
        else:
            start = random.choice(interior)
            markers = {start}
            while len(markers) < count:
                frontier = []
                for r, c in markers:
                    for nr, nc in _neighbors4(r, c, h, w):
                        if (nr, nc) in interior_set and (nr, nc) not in markers:
                            frontier.append((nr, nc))
                if not frontier:
                    remaining = [cell for cell in interior if cell not in markers]
                    if not remaining:
                        break
                    markers.add(random.choice(remaining))
                else:
                    markers.add(random.choice(frontier))

        if len(markers) == count and _color_mask_connected(mask, markers):
            return markers
    return None


def _make_region_patch(priority: int, color: int) -> np.ndarray | None:
    required_interior = max(1, priority)
    core_options = [
        (h, w)
        for h in range(3, 7)
        for w in range(3, 7)
        if (h - 2) * (w - 2) >= required_interior
    ]
    if not core_options:
        return None

    for _ in range(SHAPE_ATTEMPTS):
        core_h, core_w = random.choice(core_options)
        canvas = np.zeros((core_h + 4, core_w + 4), dtype=bool)
        r0 = c0 = 2
        canvas[r0 : r0 + core_h, c0 : c0 + core_w] = True

        for side in ("top", "bottom", "left", "right"):
            if random.random() >= 0.7:
                continue
            thickness = random.randint(1, 2)
            if side in ("top", "bottom"):
                span = random.randint(max(2, core_w // 2), core_w)
                start = random.randint(0, core_w - span)
                cc0 = c0 + start
                cc1 = cc0 + span
                if side == "top":
                    canvas[r0 - thickness : r0, cc0:cc1] = True
                else:
                    canvas[r0 + core_h : r0 + core_h + thickness, cc0:cc1] = True
            else:
                span = random.randint(max(2, core_h // 2), core_h)
                start = random.randint(0, core_h - span)
                rr0 = r0 + start
                rr1 = rr0 + span
                if side == "left":
                    canvas[rr0:rr1, c0 - thickness : c0] = True
                else:
                    canvas[rr0:rr1, c0 + core_w : c0 + core_w + thickness] = True

        mask = _crop_mask(canvas)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            mask = np.flipud(mask)

        for _ in range(random.randint(0, 2)):
            h, w = mask.shape
            corner_cells = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
            candidates = [cell for cell in corner_cells if mask[cell]]
            if not candidates:
                break
            rr, cc = random.choice(candidates)
            test = mask.copy()
            test[rr, cc] = False
            if len(_components(test)) == 1 and len(_interior_cells(test)) >= priority:
                mask = test

        if len(_components(mask)) != 1:
            continue
        if len(_interior_cells(mask)) < priority:
            continue

        markers = _pick_markers(mask, priority)
        if markers is None:
            continue

        patch = np.zeros(mask.shape, dtype=int)
        patch[mask] = color
        for rr, cc in markers:
            patch[rr, cc] = GREY
        return patch
    return None


def _dilate8(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    out = mask.copy()
    for _ in range(radius):
        prev = out.copy()
        out[1:, :] |= prev[:-1, :]
        out[:-1, :] |= prev[1:, :]
        out[:, 1:] |= prev[:, :-1]
        out[:, :-1] |= prev[:, 1:]
        out[1:, 1:] |= prev[:-1, :-1]
        out[1:, :-1] |= prev[:-1, 1:]
        out[:-1, 1:] |= prev[1:, :-1]
        out[:-1, :-1] |= prev[1:, 1:]
    return out


def _candidate_positions(h: int, w: int, ph: int, pw: int, slot: str) -> list[tuple[int, int]]:
    r_lo = GRID_MARGIN
    r_hi = h - ph - GRID_MARGIN
    c_lo = GRID_MARGIN
    c_hi = w - pw - GRID_MARGIN
    if r_lo > r_hi or c_lo > c_hi:
        return []

    if slot == "left":
        c_hi = min(c_hi, max(c_lo, w // 2 - pw))
    elif slot == "right":
        c_lo = max(c_lo, min(c_hi, w // 2))
    elif slot == "top":
        r_hi = min(r_hi, max(r_lo, h // 2 - ph))
    elif slot == "bottom":
        r_lo = max(r_lo, min(r_hi, h // 2))

    if r_lo > r_hi or c_lo > c_hi:
        if slot == "any":
            return []
        return _candidate_positions(h, w, ph, pw, "any")

    positions = [(r, c) for r in range(r_lo, r_hi + 1) for c in range(c_lo, c_hi + 1)]
    random.shuffle(positions)
    return positions


def _place_patch(
    grid: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
    slot: str,
) -> bool:
    ph, pw = patch.shape
    patch_mask = patch != BLACK
    guard = _dilate8(occupied, radius=PLACEMENT_GAP)

    for r0, c0 in _candidate_positions(grid.shape[0], grid.shape[1], ph, pw, slot):
        if np.any(guard[r0 : r0 + ph, c0 : c0 + pw] & patch_mask):
            continue
        window = grid[r0 : r0 + ph, c0 : c0 + pw]
        window[patch_mask] = patch[patch_mask]
        occupied[r0 : r0 + ph, c0 : c0 + pw] |= patch_mask
        return True
    return False


def _extract_regions(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    base_mask = (g != BLACK) & (g != GREY)
    base_comps = _components(base_mask)
    assert base_comps, "expected at least one colored region"

    comp_id = np.full((h, w), -1, dtype=int)
    regions: list[dict] = []

    for rid, comp in enumerate(base_comps):
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "each region must be monochrome apart from grey markers"
        color = colors.pop()
        for r, c in comp:
            comp_id[r, c] = rid
        regions.append(
            {
                "color": int(color),
                "cells": set((int(r), int(c)) for r, c in comp),
                "priority": 0,
            }
        )

    for comp in _components(g == GREY):
        owners = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c, h, w):
                if comp_id[nr, nc] != -1:
                    owners.add(int(comp_id[nr, nc]))
        assert len(owners) == 1, "each grey marker cluster must belong to exactly one region"
        rid = owners.pop()
        regions[rid]["priority"] += len(comp)
        regions[rid]["cells"].update((int(r), int(c)) for r, c in comp)

    for region in regions:
        region["seed_size"] = len(region["cells"])
    return regions


def _solve(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    regions = _extract_regions(g)
    priorities = [int(region["priority"]) for region in regions]
    owner = np.full((h, w), -1, dtype=int)

    for rid, region in enumerate(regions):
        for r, c in region["cells"]:
            owner[r, c] = rid

    while True:
        claims: dict[tuple[int, int], int] = {}
        for r in range(h):
            for c in range(w):
                if g[r, c] != BLACK or owner[r, c] != -1:
                    continue
                contenders = set()
                for nr, nc in _neighbors4(r, c, h, w):
                    rid = int(owner[nr, nc])
                    if rid != -1:
                        contenders.add(rid)
                if not contenders:
                    continue
                top_priority = max(priorities[rid] for rid in contenders)
                winners = [rid for rid in contenders if priorities[rid] == top_priority]
                if len(winners) == 1:
                    claims[(r, c)] = winners[0]
        if not claims:
            break
        for (r, c), rid in claims.items():
            owner[r, c] = rid

    out = np.full((h, w), BLACK, dtype=int)
    for rid, region in enumerate(regions):
        out[owner == rid] = int(region["color"])
    return out, owner, regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand all colored regions outward across the black background, breaking conflicts by the number of enclosed grey markers.

    Input structure:
    - The background is black.
    - Several 4-connected colored regions sit on that background; each region has a single visible body color.
    - Grey marker pixels appear strictly inside a region, possibly as isolated pixels or small grey clusters, and every grey pixel is 4-neighbor enclosed by that same region's body.

    Rule:
    - Count how many grey pixels belong to each colored region; that count is the region's priority.
    - Treat each region as starting from all of its own colored pixels plus its enclosed grey pixels, then let all regions expand simultaneously in 4-neighbor steps into black cells.
    - When multiple regions reach the same black cell in the same round, the highest-priority region claims it; if the best priority is tied, the cell stays black as a boundary.
    - Continue rounds until no more black cells can be claimed, then recolor every original grey marker pixel back to its surrounding region color.

    Key insight:
    - The visible expansion pattern is controlled by a hidden weight: the count of grey markers inside each seed region determines which region wins asymmetric frontier conflicts, while equal counts leave persistent black separator lines.
    """
    out, _owner, _regions = _solve(np.asarray(grid, dtype=int))
    return out


def _choose_mode() -> str:
    return random.choices(MODES, weights=(2, 2, 1), k=1)[0]


def _choose_priorities(mode: str) -> list[int]:
    if mode == "distinct":
        n_regions = random.randint(2, 4)
        return random.sample(range(6), n_regions)

    if mode == "tie_pair":
        value = random.randint(0, 5)
        return [value, value]

    shared = random.randint(0, 5)
    lower = [v for v in range(shared) if v != shared]
    higher = [v for v in range(shared + 1, 6)]
    if lower and higher:
        third = random.choice(random.choice((lower, higher)))
    else:
        third = random.choice([v for v in range(6) if v != shared])
    return [shared, shared, third]


def _mode_slots(mode: str, count: int) -> list[str]:
    if mode == "tie_pair":
        return ["left", "right"]
    if mode == "tie_plus_leader":
        return ["left", "right", random.choice(("top", "bottom", "any"))]
    slots = ["any"] * count
    random.shuffle(slots)
    return slots


def _build_input(mode: str, priorities: list[int]) -> np.ndarray | None:
    colors = random.sample(PALETTE, len(priorities))
    patches = []
    for priority, color in zip(priorities, colors):
        patch = _make_region_patch(priority, color)
        if patch is None:
            return None
        patches.append(patch)

    min_dim = {2: 14, 3: 18, 4: 22}[len(priorities)]
    h = random.randint(min_dim, 30)
    w = random.randint(min_dim, 30)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    slots = _mode_slots(mode, len(priorities))

    for patch, slot in zip(patches, slots):
        if not _place_patch(grid, occupied, patch, slot):
            return None

    if np.count_nonzero(grid) >= (h * w) // 2:
        return None
    return grid


def _valid_output(mode: str, grid: np.ndarray, out: np.ndarray, owner: np.ndarray, regions: list[dict]) -> bool:
    if np.array_equal(grid, out):
        return False

    residual_black = int(np.count_nonzero((grid == BLACK) & (out == BLACK)))
    if mode == "distinct" and residual_black != 0:
        return False
    if mode != "distinct" and residual_black == 0:
        return False

    for rid, region in enumerate(regions):
        if int(np.count_nonzero(owner == rid)) <= int(region["seed_size"]):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    mode = _choose_mode()
    priorities = _choose_priorities(mode)
    grid = _build_input(mode, priorities)
    if grid is None:
        return None

    try:
        out, owner, regions = _solve(grid)
    except AssertionError:
        return None

    extracted = sorted(region["priority"] for region in regions)
    if sorted(priorities) != extracted:
        return None
    if not _valid_output(mode, grid, out, owner, regions):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
