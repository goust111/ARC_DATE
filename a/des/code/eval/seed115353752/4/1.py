from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
YELLOW = 4
GREY = 5

FRAME_COLOR_POOL = [1, 2, 3, 6, 7, 8, 9]
GENERATE_RETRIES = 400
DIR_TO_CORNER = {
    (-1, 0): "tr",  # up
    (0, -1): "tl",  # left
    (0, 1): "br",   # right
    (1, 0): "bl",   # down
}
CORNER_TO_DIR = {v: k for k, v in DIR_TO_CORNER.items()}


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _coords_to_mask(coords: Iterable[tuple[int, int]]) -> np.ndarray:
    pts = np.array(list(coords), dtype=int)
    r0, c0, r1, c1 = _bbox(pts)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[pts[:, 0] - r0, pts[:, 1] - c0] = True
    return mask


def _mask_variants(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[int, int, bytes]] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        key = (int(rot.shape[0]), int(rot.shape[1]), rot.tobytes())
        if key not in seen:
            seen.add(key)
            out.append(rot.copy())
    return out


def _canonical_mask_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    return min((int(v.shape[0]), int(v.shape[1]), v.tobytes()) for v in _mask_variants(mask))


def _find_containers(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    colors = [int(c) for c in np.unique(g) if int(c) not in (BG, YELLOW, GREY)]
    containers: list[dict] = []
    for color in colors:
        coords = np.argwhere(g == color)
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
            continue
        if not np.all(
            (coords[:, 0] == r0) | (coords[:, 0] == r1) | (coords[:, 1] == c0) | (coords[:, 1] == c1)
        ):
            continue
        inner_corner_map = {
            "tl": (r0 + 1, c0 + 1),
            "tr": (r0 + 1, c1 - 1),
            "bl": (r1 - 1, c0 + 1),
            "br": (r1 - 1, c1 - 1),
        }
        marker_corners = [corner for corner, pos in inner_corner_map.items() if int(g[pos]) == YELLOW]
        if len(marker_corners) != 1:
            continue

        perimeter = _perimeter_cells(r0, c0, r1, c1)
        missing = [(r, c) for r, c in perimeter if int(g[r, c]) != color]
        if not missing:
            continue

        gap_mask = np.zeros_like(g, dtype=bool)
        for r, c in missing:
            gap_mask[r, c] = True
        gaps: list[dict] = []
        for comp in _components_from_mask(gap_mask):
            mask = _coords_to_mask((tuple(map(int, pt)) for pt in comp))
            gaps.append(
                {
                    "cells": [tuple(map(int, pt)) for pt in comp],
                    "mask": mask,
                    "color": color,
                }
            )
        containers.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "corner": marker_corners[0],
                "gaps": gaps,
            }
        )
    containers.sort(key=lambda d: (d["bbox"][0], d["bbox"][1]))
    return containers


def _find_pieces(grid: np.ndarray) -> list[dict]:
    pieces: list[dict] = []
    for comp in _components_from_mask(np.asarray(grid) == GREY):
        mask = _coords_to_mask((tuple(map(int, pt)) for pt in comp))
        pieces.append({"mask": mask, "variants": _mask_variants(mask)})
    return pieces


def _match_gaps_to_pieces(gaps: list[dict], pieces: list[dict]) -> dict[int, int]:
    candidates: list[list[int]] = []
    for gap in gaps:
        gap_cands: list[int] = []
        for piece_idx, piece in enumerate(pieces):
            if any(v.shape == gap["mask"].shape and np.array_equal(v, gap["mask"]) for v in piece["variants"]):
                gap_cands.append(piece_idx)
        if not gap_cands:
            raise AssertionError("every gap must have a matching grey piece")
        candidates.append(gap_cands)

    order = sorted(range(len(gaps)), key=lambda idx: len(candidates[idx]))

    def rec(pos: int, used: set[int], chosen: dict[int, int]) -> dict[int, int] | None:
        if pos == len(order):
            return chosen.copy()
        gap_idx = order[pos]
        for piece_idx in candidates[gap_idx]:
            if piece_idx in used:
                continue
            used.add(piece_idx)
            chosen[gap_idx] = piece_idx
            solved = rec(pos + 1, used, chosen)
            if solved is not None:
                return solved
            used.remove(piece_idx)
            del chosen[gap_idx]
        return None

    solution = rec(0, set(), {})
    if solution is None or len(solution) != len(gaps):
        raise AssertionError("gap-piece matching must be one-to-one")
    return solution


def _settle_inner(inner: np.ndarray, frame_color: int) -> np.ndarray:
    work = np.array(inner, dtype=int, copy=True)
    h, w = work.shape
    marker_map = {
        (0, 0): CORNER_TO_DIR["tl"],
        (0, w - 1): CORNER_TO_DIR["tr"],
        (h - 1, 0): CORNER_TO_DIR["bl"],
        (h - 1, w - 1): CORNER_TO_DIR["br"],
    }
    found = [marker_map[pos] for pos in marker_map if int(work[pos]) == YELLOW]
    if len(found) != 1:
        raise AssertionError("each container must have one yellow corner marker")
    dr, dc = found[0]

    movable = (work != BG) & (work != frame_color) & (work != YELLOW)
    coords = np.argwhere(movable)
    values = work[movable].copy()
    work[movable] = BG
    work[work == YELLOW] = BG
    if coords.size == 0:
        return work

    shift = 0
    while True:
        nxt = coords + np.array([dr, dc], dtype=int) * (shift + 1)
        if np.any(nxt[:, 0] < 0) or np.any(nxt[:, 0] >= h) or np.any(nxt[:, 1] < 0) or np.any(nxt[:, 1] >= w):
            break
        if np.any(work[nxt[:, 0], nxt[:, 1]] != BG):
            break
        shift += 1

    final = coords + np.array([dr, dc], dtype=int) * shift
    work[final[:, 0], final[:, 1]] = values
    return work


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    containers = _find_containers(g)
    if not containers:
        raise AssertionError("expected at least one container")
    pieces = _find_pieces(g)
    gaps = [gap for container in containers for gap in container["gaps"]]
    if len(pieces) != len(gaps):
        raise AssertionError("each gap must be represented by one grey piece")
    _match_gaps_to_pieces(gaps, pieces)

    out[out == GREY] = BG
    for gap in gaps:
        for r, c in gap["cells"]:
            out[r, c] = gap["color"]

    for container in containers:
        r0, c0, r1, c1 = container["bbox"]
        inner = out[r0 + 1 : r1, c0 + 1 : c1].copy()
        out[r0 + 1 : r1, c0 + 1 : c1] = _settle_inner(inner, int(container["color"]))

    out[out == YELLOW] = BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Fill frame gaps with matching rotated grey pieces, recolor them, then settle interior contents by the yellow corner marker."""
    return _solve(np.asarray(grid, dtype=int))


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _random_connected_mask(rng: random.Random, h: int, w: int, area: int) -> np.ndarray:
    cells = {(rng.randrange(h), rng.randrange(w))}
    frontier = set(cells)
    while len(cells) < area and frontier:
        cr, cc = rng.choice(tuple(frontier))
        nbrs = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                nbrs.append((nr, nc))
        if not nbrs:
            frontier.remove((cr, cc))
            continue
        nxt = rng.choice(nbrs)
        cells.add(nxt)
        frontier.add(nxt)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _enumerate_gap_candidates(h: int, w: int) -> list[dict]:
    candidates: list[dict] = []

    def add(cells: list[tuple[int, int]]) -> None:
        mask = _coords_to_mask(cells)
        candidates.append(
            {
                "cells_local": tuple(sorted(cells)),
                "mask": mask,
                "canon": _canonical_mask_key(mask),
            }
        )

    max_h_len = min(5, w - 2)
    max_v_len = min(5, h - 2)
    for length in range(2, max_h_len + 1):
        for start in range(1, w - length):
            add([(0, start + dc) for dc in range(length)])
            add([(h - 1, start + dc) for dc in range(length)])
    for length in range(2, max_v_len + 1):
        for start in range(1, h - length):
            add([(start + dr, 0) for dr in range(length)])
            add([(start + dr, w - 1) for dr in range(length)])

    max_arm_h = min(4, w - 2)
    max_arm_v = min(4, h - 2)
    for arm_h in range(2, max_arm_h + 1):
        for arm_v in range(2, max_arm_v + 1):
            add([(0, c) for c in range(arm_h)] + [(r, 0) for r in range(1, arm_v)])
            add([(0, w - 1 - c) for c in range(arm_h)] + [(r, w - 1) for r in range(1, arm_v)])
            add([(h - 1, c) for c in range(arm_h)] + [(h - 1 - r, 0) for r in range(1, arm_v)])
            add([(h - 1, w - 1 - c) for c in range(arm_h)] + [(h - 1 - r, w - 1) for r in range(1, arm_v)])

    return candidates


def _touches_existing(cells: Iterable[tuple[int, int]], existing: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        if (r, c) in existing:
            return True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (r + dr, c + dc) in existing:
                return True
    return False


def _choose_gap_set(
    rng: random.Random,
    h: int,
    w: int,
    gap_count: int,
    used_canons: set[tuple[int, int, bytes]],
) -> list[dict] | None:
    base = _enumerate_gap_candidates(h, w)
    rng.shuffle(base)
    chosen: list[dict] = []
    occupied: set[tuple[int, int]] = set()

    def rec(start: int) -> bool:
        if len(chosen) == gap_count:
            return True
        for idx in range(start, len(base)):
            cand = base[idx]
            if cand["canon"] in used_canons:
                continue
            if any(prev["canon"] == cand["canon"] for prev in chosen):
                continue
            if _touches_existing(cand["cells_local"], occupied):
                continue
            chosen.append(cand)
            for cell in cand["cells_local"]:
                occupied.add(cell)
            if rec(idx + 1):
                return True
            for cell in cand["cells_local"]:
                occupied.remove(cell)
            chosen.pop()
        return False

    if not rec(0):
        return None
    return chosen


def _generate_inner_patch(
    rng: random.Random,
    inner_h: int,
    inner_w: int,
    direction: tuple[int, int],
    palette: list[int],
) -> np.ndarray | None:
    corner = DIR_TO_CORNER[direction]
    marker_pos = {
        "tl": (0, 0),
        "tr": (0, inner_w - 1),
        "bl": (inner_h - 1, 0),
        "br": (inner_h - 1, inner_w - 1),
    }[corner]

    max_box_h = min(4, inner_h - 1)
    max_box_w = min(4, inner_w - 1)
    for _ in range(40):
        box_h = rng.randint(2, max_box_h)
        box_w = rng.randint(2, max_box_w)
        area = rng.randint(2, min(box_h * box_w, max(3, min(6, box_h * box_w))))
        mask = _random_connected_mask(rng, box_h, box_w, area)
        pts = np.argwhere(mask)
        placements: list[tuple[int, int]] = []
        for top in range(inner_h - box_h + 1):
            for left in range(inner_w - box_w + 1):
                placed = pts + np.array([top, left], dtype=int)
                cells = {(int(r), int(c)) for r, c in placed}
                if marker_pos in cells:
                    continue
                if direction == (0, -1) and placed[:, 1].min() == 0:
                    continue
                if direction == (0, 1) and placed[:, 1].max() == inner_w - 1:
                    continue
                if direction == (-1, 0) and placed[:, 0].min() == 0:
                    continue
                if direction == (1, 0) and placed[:, 0].max() == inner_h - 1:
                    continue
                placements.append((top, left))
        if not placements:
            continue
        top, left = rng.choice(placements)
        patch = np.full((inner_h, inner_w), BG, dtype=int)
        patch[marker_pos] = YELLOW
        n_colors = rng.randint(1, min(3, len(palette)))
        colors = rng.sample(palette, n_colors)
        for dr, dc in pts:
            patch[top + int(dr), left + int(dc)] = rng.choice(colors)
        return patch
    return None


def _can_place_patch(
    grid: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
    pad: int = 1,
    forbidden: np.ndarray | None = None,
) -> bool:
    h, w = mask.shape
    gh, gw = grid.shape
    if top < 0 or left < 0 or top + h > gh or left + w > gw:
        return False
    if np.any(grid[top : top + h, left : left + w][mask] != BG):
        return False
    if forbidden is not None and np.any(forbidden[top : top + h, left : left + w][mask]):
        return False
    rr0 = max(0, top - pad)
    cc0 = max(0, left - pad)
    rr1 = min(gh, top + h + pad)
    cc1 = min(gw, left + w + pad)
    return bool(np.all(grid[rr0:rr1, cc0:cc1] == BG))


def _place_patch_random(
    rng: random.Random,
    grid: np.ndarray,
    mask: np.ndarray,
    color: int,
    forbidden: np.ndarray | None = None,
    attempts: int = 250,
) -> bool:
    gh, gw = grid.shape
    h, w = mask.shape
    for _ in range(attempts):
        top = rng.randint(0, gh - h)
        left = rng.randint(0, gw - w)
        if not _can_place_patch(grid, mask, top, left, pad=1, forbidden=forbidden):
            continue
        grid[top : top + h, left : left + w][mask] = color
        return True
    return False


def _render_frame(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
    gaps_local: list[dict] | None = None,
) -> None:
    r0, c0, r1, c1 = bbox
    gap_cells = set()
    if gaps_local is not None:
        for gap in gaps_local:
            gap_cells.update((r0 + gr, c0 + gc) for gr, gc in gap["cells_local"])
    for r, c in _perimeter_cells(r0, c0, r1, c1):
        if (r, c) not in gap_cells:
            grid[r, c] = color


def _render_expected(
    shape: tuple[int, int],
    containers: list[dict],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for spec in containers:
        _render_frame(out, spec["bbox"], spec["color"], gaps_local=None)
        r0, c0, r1, c1 = spec["bbox"]
        settled = _settle_inner(spec["inner_patch"], spec["color"])
        out[r0 + 1 : r1, c0 + 1 : c1] = settled
    return out


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    n_containers = rng.choices([1, 2, 3], weights=[2, 3, 2])[0]
    h = rng.randint(24 + 2 * n_containers, 32 + 2 * n_containers)
    w = rng.randint(24 + 2 * n_containers, 32 + 2 * n_containers)
    input_grid = np.full((h, w), BG, dtype=int)

    frame_colors = rng.sample(FRAME_COLOR_POOL, n_containers)
    content_palette = [c for c in FRAME_COLOR_POOL if c not in frame_colors]
    if len(content_palette) < 3:
        return None

    containers: list[dict] = []
    used_canons: set[tuple[int, int, bytes]] = set()
    blocked: list[tuple[int, int, int, int]] = []

    for frame_color in frame_colors:
        placed = False
        for _ in range(200):
            frame_h = rng.randint(7, 11)
            frame_w = rng.randint(7, 11)
            if frame_h >= h - 2 or frame_w >= w - 2:
                continue
            r0 = rng.randint(1, h - frame_h - 1)
            c0 = rng.randint(1, w - frame_w - 1)
            bbox = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
            padded = (max(0, r0 - 2), max(0, c0 - 2), min(h - 1, bbox[2] + 2), min(w - 1, bbox[3] + 2))
            if any(_bbox_overlap(padded, other) for other in blocked):
                continue

            gap_count = rng.choice([1, 2])
            gaps = _choose_gap_set(rng, frame_h, frame_w, gap_count, used_canons)
            if gaps is None:
                continue

            direction = rng.choice(list(DIR_TO_CORNER.keys()))
            inner_patch = _generate_inner_patch(rng, frame_h - 2, frame_w - 2, direction, content_palette)
            if inner_patch is None:
                continue

            if not np.any((inner_patch != BG) & (inner_patch != YELLOW)):
                continue

            spec = {
                "bbox": bbox,
                "color": frame_color,
                "direction": direction,
                "inner_patch": inner_patch,
                "gaps": gaps,
            }
            containers.append(spec)
            blocked.append(padded)
            used_canons.update(gap["canon"] for gap in gaps)
            placed = True
            break
        if not placed:
            return None

    for spec in containers:
        _render_frame(input_grid, spec["bbox"], spec["color"], spec["gaps"])
        r0, c0, r1, c1 = spec["bbox"]
        input_grid[r0 + 1 : r1, c0 + 1 : c1] = spec["inner_patch"]

    forbidden = np.zeros_like(input_grid, dtype=bool)
    for spec in containers:
        r0, c0, r1, c1 = spec["bbox"]
        forbidden[r0 : r1 + 1, c0 : c1 + 1] = True

    piece_masks: list[np.ndarray] = []
    for spec in containers:
        for gap in spec["gaps"]:
            piece_masks.append(np.rot90(gap["mask"], rng.randrange(4)))
    rng.shuffle(piece_masks)
    for mask in piece_masks:
        if not _place_patch_random(rng, input_grid, mask, GREY, forbidden=forbidden):
            return None

    expected = _render_expected(input_grid.shape, containers)
    try:
        solved = _solve(input_grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(input_grid, expected):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATE_RETRIES):
        sample = _generate_attempt(rng)
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid ARC sample")
