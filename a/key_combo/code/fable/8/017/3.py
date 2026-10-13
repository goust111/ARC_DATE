from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]


def main(grid: np.ndarray) -> np.ndarray:
    """Draw a colored rectangle from the four mirror-vacancy anchors of the broken glyphs.

    Input structure:
    - A dominant background contains several isolated monochrome glyph components.
    - Each glyph is almost mirror-symmetric inside its own bounding box about either
      a horizontal or a vertical axis.
    - Exactly four glyphs have one unmatched solid pixel whose mirrored position is
      a background vacancy; fully symmetric glyphs are distractors.

    Rule:
    - For each glyph, test horizontal and vertical reflection inside its own bbox.
    - When exactly one axis has a two-cell mismatch, treat the occupied mismatch as
      the defect pixel and its mirrored empty partner as the anchor vacancy.
    - Use the topmost, bottommost, leftmost, and rightmost anchor vacancies as the
      four rectangle extremes; paint the top and bottom edges across the full span
      and paint the left and right edges only between those horizontal edges.
    - Paint only onto background cells and leave all original glyph pixels unchanged.

    Key insight:
    - The rectangle is determined by the missing mirrored cells, not by the visible
      defect pixels, and each glyph must choose its own symmetry axis locally.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _background_color(g)
    broken: list[dict[str, int]] = []

    for comp in _find_components(g, bg):
        r0, c0, r1, c1 = comp["bbox"]
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        coords = comp["coords"]
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

        candidates = []
        for axis in ("h", "v"):
            report = _symmetry_report(mask, axis)
            if report["mismatch"] == 2 and report.get("valid", False):
                vr, vc = report["vacancy"]
                candidates.append((axis, r0 + int(vr), c0 + int(vc)))

        if len(candidates) == 1:
            axis, vr, vc = candidates[0]
            broken.append({"axis": 0 if axis == "h" else 1, "row": vr, "col": vc, "color": comp["color"]})

    assert len(broken) == 4, f"expected 4 broken glyphs, got {len(broken)}"

    top = min(broken, key=lambda item: item["row"])
    bottom = max(broken, key=lambda item: item["row"])
    left = min(broken, key=lambda item: item["col"])
    right = max(broken, key=lambda item: item["col"])

    assert sum(item["row"] == top["row"] for item in broken) == 1
    assert sum(item["row"] == bottom["row"] for item in broken) == 1
    assert sum(item["col"] == left["col"] for item in broken) == 1
    assert sum(item["col"] == right["col"] for item in broken) == 1

    r_top = int(top["row"])
    r_bottom = int(bottom["row"])
    c_left = int(left["col"])
    c_right = int(right["col"])
    assert r_top < r_bottom and c_left < c_right

    out = g.copy()
    for c in range(c_left, c_right + 1):
        if out[r_top, c] == bg:
            out[r_top, c] = int(top["color"])
        if out[r_bottom, c] == bg:
            out[r_bottom, c] = int(bottom["color"])
    for r in range(r_top + 1, r_bottom):
        if out[r, c_left] == bg:
            out[r, c_left] = int(left["color"])
        if out[r, c_right] == bg:
            out[r, c_right] = int(right["color"])
    return out


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            out.append({"color": color, "coords": arr, "bbox": (int(r0), int(c0), int(r1), int(c1))})
    return out


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _mirror_coord(axis: str, shape: tuple[int, int], cell: Coord) -> Coord:
    r, c = cell
    h, w = shape
    if axis == "h":
        return h - 1 - r, c
    return r, w - 1 - c


def _symmetry_report(mask: np.ndarray, axis: str) -> dict[str, object]:
    refl = np.flipud(mask) if axis == "h" else np.fliplr(mask)
    diff = mask ^ refl
    mismatch = int(diff.sum())
    report: dict[str, object] = {"mismatch": mismatch}
    if mismatch != 2:
        return report

    coords = [tuple(map(int, cell)) for cell in np.argwhere(diff)]
    a, b = coords
    a_val = bool(mask[a])
    b_val = bool(mask[b])
    if a_val == b_val:
        return report

    vacancy = a if not a_val else b
    extra = b if not a_val else a
    if _mirror_coord(axis, mask.shape, extra) != vacancy:
        return report

    report["valid"] = True
    report["vacancy"] = vacancy
    report["extra"] = extra
    return report


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _bbox_is_full(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    h, w = mask.shape
    return int(r0) == 0 and int(c0) == 0 and int(r1) == h - 1 and int(c1) == w - 1


def _all_orbits(axis: str, h: int, w: int) -> list[tuple[Coord, ...]]:
    seen: set[tuple[Coord, ...]] = set()
    out: list[tuple[Coord, ...]] = []
    for r in range(h):
        for c in range(w):
            orb = tuple(sorted({(r, c), _mirror_coord(axis, (h, w), (r, c))}))
            if orb not in seen:
                seen.add(orb)
                out.append(orb)
    return out


def _gen_symmetric_base(axis: str, *, require_other_asym: bool) -> np.ndarray:
    other = "v" if axis == "h" else "h"
    for _ in range(200):
        h = random.randint(4, 7)
        w = random.randint(4, 7)
        if axis == "h" and h < 3:
            continue
        if axis == "v" and w < 3:
            continue

        mask = np.ones((h, w), dtype=bool)
        orbits = _all_orbits(axis, h, w)
        random.shuffle(orbits)
        remove_target = random.randint(1, max(1, len(orbits) // 2))
        removed = 0

        for orbit in orbits:
            if removed >= remove_target:
                break
            cand = mask.copy()
            for r, c in orbit:
                cand[r, c] = False
            if int(cand.sum()) < max(h, w) + 2:
                continue
            if not _bbox_is_full(cand) or not _is_connected(cand):
                continue
            mask = cand
            removed += 1

        if removed == 0:
            continue
        if _symmetry_report(mask, axis)["mismatch"] != 0:
            continue
        if require_other_asym and _symmetry_report(mask, other)["mismatch"] == 0:
            continue
        return mask
    raise RuntimeError("failed to generate symmetric base")


def _gen_broken_patch(axis: str) -> tuple[np.ndarray, Coord]:
    other = "v" if axis == "h" else "h"
    for _ in range(300):
        base = _gen_symmetric_base(axis, require_other_asym=True)
        cells = [tuple(map(int, cell)) for cell in np.argwhere(base)]
        random.shuffle(cells)
        for cell in cells:
            mate = _mirror_coord(axis, base.shape, cell)
            if mate == cell or not base[mate]:
                continue
            broken = base.copy()
            broken[cell] = False
            if not _bbox_is_full(broken) or not _is_connected(broken):
                continue
            intended = _symmetry_report(broken, axis)
            orth = _symmetry_report(broken, other)
            if intended.get("valid", False) and intended["vacancy"] == cell and orth["mismatch"] not in (0, 2):
                return broken, cell
    raise RuntimeError("failed to generate broken patch")


def _gen_distractor_patch() -> np.ndarray:
    for _ in range(200):
        axis = random.choice(("h", "v"))
        mask = _gen_symmetric_base(axis, require_other_asym=False)
        other = "v" if axis == "h" else "h"
        if _symmetry_report(mask, axis)["mismatch"] == 0 and _symmetry_report(mask, other)["mismatch"] != 2:
            return mask
    raise RuntimeError("failed to generate distractor patch")


def _can_place(occupied: np.ndarray, patch: np.ndarray, top: int, left: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    ph, pw = patch.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    for pr, pc in np.argwhere(patch):
        gr = top + int(pr)
        gc = left + int(pc)
        r0 = max(0, gr - pad)
        r1 = min(h, gr + pad + 1)
        c0 = max(0, gc - pad)
        c1 = min(w, gc + pad + 1)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, top: int, left: int, color: int) -> None:
    for pr, pc in np.argwhere(patch):
        gr = top + int(pr)
        gc = left + int(pc)
        grid[gr, gc] = int(color)
        occupied[gr, gc] = True


def _try_place_random(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, color: int) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
    random.shuffle(positions)
    for top, left in positions[:200]:
        if _can_place(occupied, patch, top, left):
            _place_patch(grid, occupied, patch, top, left, color)
            return True
    return False


def _attempt_input() -> np.ndarray | None:
    h = random.randint(20, 28)
    w = random.randint(20, 28)
    bg = random.randint(0, 9)
    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    top = random.randint(5, h // 2 - 3)
    bottom = random.randint(h // 2 + 2, h - 6)
    left = random.randint(5, w // 2 - 3)
    right = random.randint(w // 2 + 2, w - 6)
    if bottom - top < 6 or right - left < 6:
        return None

    top_col = random.randint(left + 1, right - 1)
    bottom_col = random.randint(left + 1, right - 1)
    left_row = random.randint(top + 1, bottom - 1)
    right_row = random.randint(top + 1, bottom - 1)

    color_pool = [c for c in range(10) if c != bg]
    distractor_count = random.randint(1, 4)
    need = 4 + distractor_count
    if len(color_pool) < need:
        return None
    colors = random.sample(color_pool, need)

    broken_specs = [
        ("top", "h", (top, top_col)),
        ("bottom", "h", (bottom, bottom_col)),
        ("left", "v", (left_row, left)),
        ("right", "v", (right_row, right)),
    ]
    random.shuffle(broken_specs)

    for idx, (_name, axis, anchor) in enumerate(broken_specs):
        patch, vacancy = _gen_broken_patch(axis)
        top_left = anchor[0] - vacancy[0]
        left_left = anchor[1] - vacancy[1]
        if not _can_place(occupied, patch, top_left, left_left):
            return None
        _place_patch(grid, occupied, patch, top_left, left_left, colors[idx])

    for color in colors[4:]:
        patch = _gen_distractor_patch()
        if not _try_place_random(grid, occupied, patch, color):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
