from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

GEN_ATTEMPTS = 400
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _mask(rows: list[str]) -> np.ndarray:
    return np.array([[1 if ch == "#" else 0 for ch in row] for row in rows], dtype=int)


H_ONLY_MASKS = {
    4: [_mask([".#", "##", ".#"])],
    5: [_mask(["#..", "###", "#.."])],
    6: [_mask(["#...", "####", "#..."])],
    7: [_mask(["##.", "###", "##."])],
}

V_ONLY_MASKS = {
    4: [_mask([".#.", "###"])],
    5: [_mask(["###", ".#.", ".#."])],
    6: [_mask(["###", ".#.", ".#.", ".#."])],
    7: [_mask(["###", "###", ".#."])],
}

BOTH_MASKS = {
    3: [_mask(["###"]), _mask(["#", "#", "#"])],
    4: [_mask(["##", "##"])],
    5: [_mask([".#.", "###", ".#."])],
    6: [_mask(["###", "###"])],
}

ASYM_MASKS = {
    3: [_mask(["#.", "##"])],
    4: [_mask(["#..", "###"])],
    5: [_mask(["##.", "###"])],
    6: [_mask(["###", "##.", ".#."])],
    7: [_mask(["###", "#.#", "##."])],
    8: [_mask(["###.", "###.", "..##"])],
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    h, w = grid.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    out = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    out[coords[:, 0] - int(r0), coords[:, 1] - int(c0)] = True
    return out


def _symmetry_flags(mask: np.ndarray) -> tuple[bool, bool]:
    return bool(np.array_equal(mask, np.flipud(mask))), bool(np.array_equal(mask, np.fliplr(mask)))


def _grey_threshold(grid: np.ndarray) -> int:
    greys = np.argwhere(grid == GREY)
    assert len(greys) >= 2, "expected at least two grey blocker pixels"
    best = None
    for i in range(len(greys)):
        for j in range(i + 1, len(greys)):
            dist = int(abs(int(greys[i, 0]) - int(greys[j, 0])) + abs(int(greys[i, 1]) - int(greys[j, 1])))
            best = dist if best is None else min(best, dist)
    assert best is not None
    return int(best)


def _classify_component(coords: np.ndarray, threshold: int) -> dict:
    mask = _tight_mask(coords)
    horiz, vert = _symmetry_flags(mask)
    area = int(len(coords))
    if area <= threshold:
        if horiz:
            outcome = BLUE
        elif vert:
            outcome = GREEN
        elif not horiz and not vert:
            outcome = YELLOW
        else:
            raise AssertionError("unreachable small-component branch")
    else:
        outcome = ORANGE if (horiz or vert) else BLACK
    return {
        "area": area,
        "horizontal": horiz,
        "vertical": vert,
        "outcome": int(outcome),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor or erase each red sprite according to the blocker-defined size threshold and its reflection symmetry.

    Input structure:
    - A black background contains red 4-connected sprite objects, exactly two grey blocker pixels, and optional
      isolated single red noise pixels.
    - Each non-noise sprite has between 3 and 8 red pixels and may be horizontally symmetric, vertically symmetric,
      both, or neither inside its tight bounding box.

    Rule:
    - Erase every singleton red 4-connected component.
    - Compute threshold T as the minimum pairwise Manhattan distance between the grey blocker pixels.
    - For each remaining red component, test horizontal and vertical reflection symmetry on its tight bbox mask and
      test whether its area is <= T.
    - Recolor horizontally symmetric small sprites blue, vertically-only symmetric small sprites green, any symmetric
      large sprite orange, asymmetric small sprites yellow, and erase asymmetric large sprites; keep grey blockers.

    Key insight:
    - The grey pixels do not act locally on shapes; instead their spacing encodes one global size cutoff that must be
      crossed with an independent symmetry classification to determine each sprite's fate.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    out = g.copy()
    out[out == RED] = bg

    threshold = _grey_threshold(g)
    for coords in _components_of_color(g, RED):
        if len(coords) == 1:
            continue
        info = _classify_component(coords, threshold)
        color = int(info["outcome"])
        if color == bg:
            continue
        out[coords[:, 0], coords[:, 1]] = color

    return out


def _random_dihedral(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _random_preserving_variant(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _sample_mask(kind: str, area: int) -> np.ndarray:
    if kind == "H":
        return _random_preserving_variant(random.choice(H_ONLY_MASKS[area]))
    if kind == "V":
        return _random_preserving_variant(random.choice(V_ONLY_MASKS[area]))
    if kind == "B":
        return _random_dihedral(random.choice(BOTH_MASKS[area]))
    if kind == "A":
        return _random_dihedral(random.choice(ASYM_MASKS[area]))
    raise ValueError(f"unknown kind {kind}")


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask > 0):
        rr, cc = int(top + r), int(left + c)
        grid[rr, cc] = int(color)
        coords.append((rr, cc))
    return coords


def _mark_blocked(blocked: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _place_mask(grid: np.ndarray, blocked: np.ndarray, mask: np.ndarray) -> bool:
    h, w = grid.shape
    mh, mw = mask.shape
    positions = [(r, c) for r in range(h - mh + 1) for c in range(w - mw + 1)]
    random.shuffle(positions)
    for top, left in positions:
        abs_coords = [(int(top + r), int(left + c)) for r, c in np.argwhere(mask > 0)]
        if all(grid[r, c] == BLACK and not blocked[r, c] for r, c in abs_coords):
            coords = _paint_mask(grid, mask, top, left, RED)
            _mark_blocked(blocked, coords)
            return True
    return False


def _choose_grey_pair(grid: np.ndarray, blocked: np.ndarray, threshold: int) -> bool:
    h, w = grid.shape
    free = [(r, c) for r in range(h) for c in range(w) if grid[r, c] == BLACK and not blocked[r, c]]
    random.shuffle(free)
    for r1, c1 in free:
        targets = [(r2, c2) for r2, c2 in free if (r2, c2) != (r1, c1) and abs(r1 - r2) + abs(c1 - c2) == threshold]
        random.shuffle(targets)
        for r2, c2 in targets:
            grid[r1, c1] = GREY
            grid[r2, c2] = GREY
            return True
    return False


def _place_noise(grid: np.ndarray, blocked: np.ndarray, count: int) -> bool:
    h, w = grid.shape
    for _ in range(count):
        candidates = [(r, c) for r in range(h) for c in range(w) if grid[r, c] == BLACK and not blocked[r, c]]
        if not candidates:
            return False
        r, c = random.choice(candidates)
        grid[r, c] = RED
        _mark_blocked(blocked, [(r, c)])
    return True


def _analyze_input(grid: np.ndarray) -> tuple[int, list[dict], int]:
    threshold = _grey_threshold(grid)
    comps = _components_of_color(grid, RED)
    infos = []
    noise_count = 0
    for coords in comps:
        if len(coords) == 1:
            noise_count += 1
            continue
        info = _classify_component(coords, threshold)
        info["coords"] = coords
        infos.append(info)
    return threshold, infos, noise_count


def _validate_input(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if int((grid == GREY).sum()) != 2:
        return False

    threshold, infos, noise_count = _analyze_input(grid)
    if threshold not in {4, 5, 6}:
        return False
    if not (5 <= len(infos) <= 7):
        return False
    if not (0 <= noise_count <= 2):
        return False

    outcomes = {info["outcome"] for info in infos}
    if outcomes != {BLUE, GREEN, YELLOW, ORANGE, BLACK}:
        return False

    if not any(info["horizontal"] and info["vertical"] and info["outcome"] == BLUE for info in infos):
        return False
    if not any(info["horizontal"] and not info["vertical"] and info["area"] == threshold and info["outcome"] == BLUE for info in infos):
        return False
    if not any(info["vertical"] and not info["horizontal"] and info["area"] == threshold and info["outcome"] == GREEN for info in infos):
        return False
    if not any(info["horizontal"] and not info["vertical"] and info["area"] == threshold + 1 and info["outcome"] == ORANGE for info in infos):
        return False
    if not any((not info["horizontal"]) and (not info["vertical"]) and info["area"] < threshold and info["outcome"] == YELLOW for info in infos):
        return False
    if not any((not info["horizontal"]) and (not info["vertical"]) and info["area"] > threshold and info["outcome"] == BLACK for info in infos):
        return False

    area_to_outcomes: dict[int, set[int]] = {}
    for info in infos:
        area_to_outcomes.setdefault(info["area"], set()).add(info["outcome"])
    if not any(len(colors) >= 2 for colors in area_to_outcomes.values()):
        return False

    red_cells = np.argwhere(grid == RED)
    grey_cells = np.argwhere(grid == GREY)
    for gr, gc in grey_cells:
        if len(red_cells) and np.min(np.max(np.abs(red_cells - np.array([gr, gc])), axis=1)) <= 1:
            return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if not np.array_equal(out[grid == GREY], np.full(int((grid == GREY).sum()), GREY)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 22)
    w = random.randint(18, 22)
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    threshold = random.randint(4, 6)

    specs = [
        ("H", threshold),
        ("V", threshold),
        ("H", threshold + 1),
        ("A", max(3, threshold - 1)),
        ("A", threshold + 2),
        ("B", 4),
    ]
    if random.random() < 0.55:
        extra_choices = [
            ("A", threshold),
            ("B", min(6, threshold)),
            ("V", min(7, threshold + 1)),
            ("H", min(7, threshold + 1)),
        ]
        specs.append(random.choice(extra_choices))

    masks = [_sample_mask(kind, area) for kind, area in specs]
    masks.sort(key=lambda m: (-int(m.sum()), -int(m.shape[0] * m.shape[1])))
    for mask in masks:
        if not _place_mask(grid, blocked, mask):
            return None

    if not _choose_grey_pair(grid, blocked, threshold):
        return None

    noise_count = random.choices([0, 1, 2], weights=[1, 3, 2])[0]
    if not _place_noise(grid, blocked, noise_count):
        return None

    return grid if _validate_input(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
