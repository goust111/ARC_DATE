import random
from typing import Iterable

import numpy as np


BG = 0
YELLOW = 4
AZURE = 8
FRAME_COLORS = (1, 2, 3, 5, 6, 7, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_RETRIES = 400
SHAPE_RETRIES = 200
ASSEMBLY_RETRIES = 120


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _bbox(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(points)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0, "expected a non-empty mask"
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask.astype(bool))


def _divider_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == AZURE)]
    assert len(cols) == 1, f"expected exactly one full-height azure divider, got {cols}"
    return cols[0]


def _is_rect_frame(left: np.ndarray, pts: list[tuple[int, int]], color: int) -> bool:
    r0, c0, r1, c1 = _bbox(pts)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    pset = set(pts)
    if pset != border:
        return False
    box = left[r0 : r1 + 1, c0 : c1 + 1]
    if np.any(box[1:-1, 1:-1] == color):
        return False
    return True


def _extract_templates(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    divider = _divider_col(grid)
    left = grid[:, :divider]
    templates: list[dict[str, object]] = []
    for color in sorted(int(c) for c in np.unique(left)):
        if color in (BG, YELLOW, AZURE):
            continue
        for comp in _components(left == color):
            if not _is_rect_frame(left, comp, color):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            inner = left[r0 + 1 : r1, c0 + 1 : c1] == YELLOW
            assert inner.any(), f"frame color {color} has no yellow content"
            mask = _trim_mask(inner)
            templates.append({"color": color, "mask": mask})
    assert 2 <= len(templates) <= 4, f"expected 2-4 templates, got {len(templates)}"
    return divider, templates


def _candidate_placements(region_mask: np.ndarray, templates: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[list[int]]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(region_mask)]
    assert coords, "expected a non-empty composite region"
    index = {cell: i for i, cell in enumerate(coords)}
    placements: list[dict[str, object]] = []
    by_cell: list[list[int]] = [[] for _ in coords]
    h, w = region_mask.shape

    for template_idx, template in enumerate(templates):
        mask = np.asarray(template["mask"], dtype=bool)
        offsets = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        th, tw = mask.shape
        for top in range(h - th + 1):
            for left in range(w - tw + 1):
                cells: list[tuple[int, int]] = []
                valid = True
                for dr, dc in offsets:
                    rr = top + dr
                    cc = left + dc
                    if not region_mask[rr, cc]:
                        valid = False
                        break
                    cells.append((rr, cc))
                if not valid:
                    continue
                bits = 0
                cell_ids: list[int] = []
                for cell in cells:
                    cid = index[cell]
                    cell_ids.append(cid)
                    bits |= 1 << cid
                pid = len(placements)
                placement = {
                    "template_idx": template_idx,
                    "color": int(template["color"]),
                    "bits": bits,
                    "cells": cells,
                    "cell_ids": cell_ids,
                }
                placements.append(placement)
                for cid in cell_ids:
                    by_cell[cid].append(pid)
    return placements, by_cell


def _solve_exact_cover(region_mask: np.ndarray, templates: list[dict[str, object]], max_solutions: int = 2) -> list[list[dict[str, object]]]:
    placements, by_cell = _candidate_placements(region_mask, templates)
    full = (1 << int(region_mask.sum())) - 1
    solutions: list[list[int]] = []

    def dfs(covered: int, chosen: list[int]) -> None:
        if len(solutions) >= max_solutions:
            return
        if covered == full:
            solutions.append(chosen.copy())
            return

        remaining = full & ~covered
        best_opts: list[int] | None = None
        while remaining:
            lsb = remaining & -remaining
            cid = lsb.bit_length() - 1
            remaining ^= lsb
            opts = [pid for pid in by_cell[cid] if placements[pid]["bits"] & covered == 0]
            if not opts:
                return
            if best_opts is None or len(opts) < len(best_opts):
                best_opts = opts
                if len(best_opts) == 1:
                    break
        assert best_opts is not None

        for pid in best_opts:
            chosen.append(pid)
            dfs(covered | int(placements[pid]["bits"]), chosen)
            chosen.pop()

    dfs(0, [])
    decoded: list[list[dict[str, object]]] = []
    for sol in solutions:
        decoded.append([placements[pid] for pid in sol])
    return decoded


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split the right yellow composite into translated copies of the left dictionary shapes
    and recolor each copy with the matching frame color.
    """
    g = np.asarray(grid, dtype=int)
    divider, templates = _extract_templates(g)
    right = g[:, divider + 1 :]
    region_mask = right == YELLOW
    solutions = _solve_exact_cover(region_mask, templates, max_solutions=2)
    assert len(solutions) == 1, f"expected a unique decomposition, got {len(solutions)} solutions"

    out = g.copy()
    for placement in solutions[0]:
        color = int(placement["color"])
        for r, c in placement["cells"]:
            out[r, divider + 1 + c] = color
    return out


def _random_shape(rng: random.Random, area: int) -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        cells = {(rng.randrange(5), rng.randrange(5))}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < 5 and 0 <= nc < 5 and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(sorted(frontier)))
        if len(cells) != area:
            continue
        mask = np.zeros((5, 5), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _trim_mask(mask)
        if min(mask.shape) < 2:
            continue
        if mask.all():
            continue
        return mask
    raise RuntimeError(f"failed to build a shape of area {area}")


def _make_entry(mask: np.ndarray, color: int, rng: random.Random) -> np.ndarray:
    th, tw = mask.shape
    extra_top = rng.randint(0, 2)
    extra_bottom = rng.randint(0, 2)
    extra_left = rng.randint(0, 2)
    extra_right = rng.randint(0, 2)
    inner_h = th + extra_top + extra_bottom
    inner_w = tw + extra_left + extra_right
    out = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    out[0, :] = color
    out[-1, :] = color
    out[:, 0] = color
    out[:, -1] = color
    patch = out[1 + extra_top : 1 + extra_top + th, 1 + extra_left : 1 + extra_left + tw]
    patch[mask] = YELLOW
    return out


def _compose_union(masks: list[np.ndarray], rng: random.Random) -> tuple[np.ndarray, list[set[tuple[int, int]]]] | None:
    order = list(range(len(masks)))
    rng.shuffle(order)

    placed: dict[int, set[tuple[int, int]]] = {}
    first_cells = {tuple(map(int, rc)) for rc in np.argwhere(masks[order[0]])}
    placed[order[0]] = first_cells
    union = set(first_cells)

    for idx in order[1:]:
        offsets = [tuple(map(int, rc)) for rc in np.argwhere(masks[idx])]
        candidates: list[tuple[int, int, set[tuple[int, int]], int]] = []
        tried = set()
        for ur, uc in union:
            for sr, sc in offsets:
                for dr, dc in DIR4:
                    top = ur + dr - sr
                    left = uc + dc - sc
                    if (top, left) in tried:
                        continue
                    tried.add((top, left))
                    shifted = {(top + r, left + c) for r, c in offsets}
                    if shifted & union:
                        continue
                    if not any((r + ddr, c + ddc) in union for r, c in shifted for ddr, ddc in DIR4):
                        continue
                    all_cells = union | shifted
                    r0, c0, r1, c1 = _bbox(all_cells)
                    height = r1 - r0 + 1
                    width = c1 - c0 + 1
                    candidates.append((top, left, shifted, height * width))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[3])
        top, left, shifted, _ = rng.choice(candidates[: min(12, len(candidates))])
        placed[idx] = shifted
        union |= shifted

    r0, c0, r1, c1 = _bbox(union)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    composite = np.zeros((h, w), dtype=bool)
    placement_sets: list[set[tuple[int, int]]] = []
    for idx in range(len(masks)):
        shifted = {(r - r0, c - c0) for r, c in placed[idx]}
        placement_sets.append(shifted)
        for r, c in shifted:
            composite[r, c] = True
    return composite, placement_sets


def _build_expected(grid: np.ndarray, placements: list[set[tuple[int, int]]], colors: list[int], top: int, left: int) -> np.ndarray:
    out = grid.copy()
    for cells, color in zip(placements, colors):
        for r, c in cells:
            out[top + r, left + c] = color
    return out


def _generate_case(rng: random.Random) -> np.ndarray | None:
    n_templates = rng.randint(2, 4)
    areas = rng.sample((4, 5, 6, 7, 8), n_templates)
    colors = rng.sample(FRAME_COLORS, n_templates)

    masks: list[np.ndarray] = []
    seen = set()
    for area in areas:
        for _ in range(SHAPE_RETRIES):
            mask = _random_shape(rng, area)
            key = _mask_key(mask)
            if key not in seen:
                seen.add(key)
                masks.append(mask)
                break
        else:
            return None

    assembly = None
    for _ in range(ASSEMBLY_RETRIES):
        assembly = _compose_union(masks, rng)
        if assembly is not None:
            break
    if assembly is None:
        return None
    composite_mask, placements = assembly

    entries = [_make_entry(mask, color, rng) for mask, color in zip(masks, colors)]
    order = list(range(n_templates))
    rng.shuffle(order)
    entries = [entries[i] for i in order]

    max_entry_w = max(entry.shape[1] for entry in entries)
    between = [rng.randint(1, 3) for _ in range(max(0, n_templates - 1))]
    content_h = sum(entry.shape[0] for entry in entries) + sum(between)
    left_w = max_entry_w + rng.randint(4, 7)
    min_h = max(content_h + 2, composite_mask.shape[0] + 2)
    height = min_h + rng.randint(0, 6)
    right_w = composite_mask.shape[1] + rng.randint(4, 8)
    width = left_w + 1 + right_w

    grid = np.full((height, width), BG, dtype=int)
    divider = left_w
    grid[:, divider] = AZURE

    top_margin = rng.randint(1, height - content_h - 1)
    cur = top_margin
    for idx, entry in enumerate(entries):
        eh, ew = entry.shape
        left = rng.randint(1, left_w - ew - 1)
        grid[cur : cur + eh, left : left + ew] = entry
        cur += eh
        if idx < len(entries) - 1:
            cur += between[idx]

    comp_h, comp_w = composite_mask.shape
    comp_top = rng.randint(1, height - comp_h - 1)
    comp_left = divider + 1 + rng.randint(1, right_w - comp_w - 1)
    for r, c in np.argwhere(composite_mask):
        grid[comp_top + int(r), comp_left + int(c)] = YELLOW

    if len(_components(grid[:, divider + 1 :] == YELLOW)) != 1:
        return None

    solved = _solve_exact_cover(grid[:, divider + 1 :] == YELLOW, [{"color": c, "mask": m} for c, m in zip(colors, masks)], max_solutions=2)
    if len(solved) != 1:
        return None

    # The generator should only emit instances that the actual solver interprets identically.
    expected = _build_expected(grid, placements, colors, comp_top, comp_left)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _generate_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
