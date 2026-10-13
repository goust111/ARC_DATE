from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FENCE = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_fences(g: np.ndarray) -> list[dict]:
    fences: list[dict] = []
    for comp in _components(g == FENCE):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 4 or w < 4:
            continue
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        grey_mask = sub == FENCE
        if np.any(grey_mask & ~border):
            continue
        gap = np.argwhere(border & ~grey_mask)
        if gap.shape != (1, 2):
            continue
        gr, gc = map(int, gap[0])
        if gr == 0:
            direction = "up"
        elif gr == h - 1:
            direction = "down"
        elif gc == 0:
            direction = "left"
        else:
            direction = "right"
        fences.append(
            {
                "bbox": (r0, c0, r1, c1),
                "direction": direction,
                "gap": (r0 + gr, c0 + gc),
            }
        )
    assert fences, "expected at least one valid fence"
    return fences


def _color_patterns(g: np.ndarray, fence: dict) -> list[dict]:
    r0, c0, r1, c1 = fence["bbox"]
    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    colors = [int(c) for c in np.unique(interior) if int(c) not in (BG, FENCE)]
    patterns: list[dict] = []
    for color in colors:
        pts_local = [tuple(map(int, x)) for x in np.argwhere(interior == color)]
        assert pts_local, "seed color must have pixels inside fence"
        pr0, pc0, pr1, pc1 = _bbox(pts_local)
        patch = np.full((pr1 - pr0 + 1, pc1 - pc0 + 1), BG, dtype=int)
        for rr, cc in pts_local:
            patch[rr - pr0, cc - pc0] = color
        patterns.append(
            {
                "color": color,
                "count": len(pts_local),
                "patch": patch,
                "top": r0 + 1 + pr0,
                "left": c0 + 1 + pc0,
            }
        )
    assert patterns, "each fence must contain at least one colored seed"
    return patterns


def _stamp(out: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    ph, pw = patch.shape
    for rr in range(ph):
        for cc in range(pw):
            val = int(patch[rr, cc])
            if val != BG:
                out[top + rr, left + cc] = val


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each monochrome seed pattern out of the unique fence gap.

    For a fence with a gap on one side, every non-background, non-grey color
    inside that fence is cropped to its tight bounding box. If the color has N
    seed pixels, that cropped patch is copied N times outside the fence in the
    gap direction. Copies are contiguous along the growth axis, preserving their
    perpendicular alignment. Smaller-N colors are painted first, so larger-N
    colors overwrite them on overlap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()

    for fence in _find_fences(g):
        patterns = _color_patterns(g, fence)
        patterns.sort(key=lambda item: (item["count"], item["top"], item["left"]))
        r0, c0, r1, c1 = fence["bbox"]
        for item in patterns:
            patch = item["patch"]
            ph, pw = patch.shape
            count = int(item["count"])
            top = int(item["top"])
            left = int(item["left"])
            direction = fence["direction"]

            if direction == "up":
                start_r, start_c = r0 - ph, left
                dr, dc = -ph, 0
            elif direction == "down":
                start_r, start_c = r1 + 1, left
                dr, dc = ph, 0
            elif direction == "left":
                start_r, start_c = top, c0 - pw
                dr, dc = 0, -pw
            else:
                start_r, start_c = top, c1 + 1
                dr, dc = 0, pw

            for step in range(count):
                rr = start_r + dr * step
                cc = start_c + dc * step
                assert 0 <= rr and rr + ph <= g.shape[0]
                assert 0 <= cc and cc + pw <= g.shape[1]
                _stamp(out, patch, rr, cc)
    return out


def _grow_connected_mask(height: int, width: int, area: int) -> np.ndarray:
    cells = {(random.randrange(height), random.randrange(width))}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0 = min(rows), min(cols)
    patch = np.zeros((max(rows) - r0 + 1, max(cols) - c0 + 1), dtype=bool)
    for r, c in cells:
        patch[r - r0, c - c0] = True
    return patch


def _side_anchor_positions(
    direction: str,
    interior_h: int,
    interior_w: int,
    patch_h: int,
    patch_w: int,
    gap_offset: int,
    overlap_mode: bool,
    taken: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    if direction in ("up", "down"):
        row = 0 if direction == "up" else interior_h - patch_h
        lo = max(0, gap_offset - patch_w + 1)
        hi = min(gap_offset, interior_w - patch_w)
        for col in range(lo, hi + 1):
            candidates.append((row, col))
    else:
        col = 0 if direction == "left" else interior_w - patch_w
        lo = max(0, gap_offset - patch_h + 1)
        hi = min(gap_offset, interior_h - patch_h)
        for row in range(lo, hi + 1):
            candidates.append((row, col))

    random.shuffle(candidates)
    good: list[tuple[int, int]] = []
    for row, col in candidates:
        box = (row, col, row + patch_h - 1, col + patch_w - 1)
        ok = True
        for tr0, tc0, tr1, tc1 in taken:
            if not (box[2] + 1 < tr0 or tr1 + 1 < box[0] or box[3] + 1 < tc0 or tc1 + 1 < box[1]):
                ok = False
                break
        if ok:
            good.append((row, col))
    if good:
        return good
    if overlap_mode:
        return candidates
    return []


def _growth_span(direction: str, patch_h: int, patch_w: int, count: int) -> int:
    return count * (patch_h if direction in ("up", "down") else patch_w)


def _reserve_box(
    top: int,
    left: int,
    box_h: int,
    box_w: int,
    direction: str,
    span: int,
) -> tuple[int, int, int, int]:
    if direction == "up":
        return top - span, left, top + box_h - 1, left + box_w - 1
    if direction == "down":
        return top, left, top + box_h - 1 + span, left + box_w - 1
    if direction == "left":
        return top, left - span, top + box_h - 1, left + box_w - 1
    return top, left, top + box_h - 1, left + box_w - 1 + span


def _boxes_separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    return a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1]


def _build_fence_spec() -> dict | None:
    interior_h = random.randint(5, 8)
    interior_w = random.randint(5, 8)
    box_h = interior_h + 2
    box_w = interior_w + 2
    direction = random.choice(["up", "down", "left", "right"])
    color_count = random.randint(1, 3)
    counts = random.sample([1, 2, 3, 4], color_count)
    colors = random.sample(PALETTE, color_count)
    overlap_mode = color_count >= 2 and random.random() < 0.5

    if direction in ("up", "down"):
        gap_offset = random.randint(1, interior_w - 2)
    else:
        gap_offset = random.randint(1, interior_h - 2)

    seed_specs: list[dict] = []
    taken: list[tuple[int, int, int, int]] = []
    max_span = 0
    for color, count in sorted(zip(colors, counts), key=lambda x: x[1]):
        shape_options = [(h, w) for h in (1, 2, 3) for w in (1, 2, 3) if h * w >= count]
        random.shuffle(shape_options)
        placed = False
        for patch_h, patch_w in shape_options:
            positions = _side_anchor_positions(
                direction,
                interior_h,
                interior_w,
                patch_h,
                patch_w,
                gap_offset,
                overlap_mode,
                taken,
            )
            if not positions:
                continue
            for row, col in positions:
                mask = _grow_connected_mask(patch_h, patch_w, count)
                if direction == "up" and not mask[0].any():
                    continue
                if direction == "down" and not mask[-1].any():
                    continue
                if direction == "left" and not mask[:, 0].any():
                    continue
                if direction == "right" and not mask[:, -1].any():
                    continue
                seed_specs.append(
                    {
                        "color": color,
                        "count": count,
                        "mask": mask,
                        "row": row,
                        "col": col,
                    }
                )
                taken.append((row, col, row + patch_h - 1, col + patch_w - 1))
                max_span = max(max_span, _growth_span(direction, patch_h, patch_w, count))
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    patch = np.full((box_h, box_w), BG, dtype=int)
    patch[0, :] = FENCE
    patch[-1, :] = FENCE
    patch[:, 0] = FENCE
    patch[:, -1] = FENCE
    if direction == "up":
        patch[0, gap_offset + 1] = BG
    elif direction == "down":
        patch[-1, gap_offset + 1] = BG
    elif direction == "left":
        patch[gap_offset + 1, 0] = BG
    else:
        patch[gap_offset + 1, -1] = BG

    for spec in seed_specs:
        mask = spec["mask"]
        pr, pc = spec["row"] + 1, spec["col"] + 1
        for rr in range(mask.shape[0]):
            for cc in range(mask.shape[1]):
                if mask[rr, cc]:
                    patch[pr + rr, pc + cc] = spec["color"]

    return {
        "patch": patch,
        "box_h": box_h,
        "box_w": box_w,
        "direction": direction,
        "max_span": max_span,
    }


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(24, 32)
    w = random.randint(24, 32)
    grid = np.full((h, w), BG, dtype=int)
    specs: list[tuple[dict, int, int, tuple[int, int, int, int]]] = []
    target_fences = random.randint(1, 4)

    for _ in range(target_fences):
        placed = False
        for _ in range(80):
            spec = _build_fence_spec()
            if spec is None:
                continue
            box_h = spec["box_h"]
            box_w = spec["box_w"]
            direction = spec["direction"]
            span = spec["max_span"]

            if direction == "up":
                min_r, max_r = span, h - box_h
                min_c, max_c = 0, w - box_w
            elif direction == "down":
                min_r, max_r = 0, h - box_h - span
                min_c, max_c = 0, w - box_w
            elif direction == "left":
                min_r, max_r = 0, h - box_h
                min_c, max_c = span, w - box_w
            else:
                min_r, max_r = 0, h - box_h
                min_c, max_c = 0, w - box_w - span
            if min_r > max_r or min_c > max_c:
                continue

            top = random.randint(min_r, max_r)
            left = random.randint(min_c, max_c)
            reserve = _reserve_box(top, left, box_h, box_w, direction, span)
            if any(not _boxes_separate(reserve, prev, pad=1) for *_ignore, prev in specs):
                continue
            specs.append((spec, top, left, reserve))
            placed = True
            break
        if not placed:
            break

    if not specs:
        return None

    for spec, top, left, _reserve in specs:
        patch = spec["patch"]
        box_h, box_w = patch.shape
        if np.any(grid[top : top + box_h, left : left + box_w] != BG):
            return None
        grid[top : top + box_h, left : left + box_w] = patch

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid fence-growth input")
