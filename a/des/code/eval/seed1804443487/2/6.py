from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
MAROON = 9
TOWER_COLORS = (1, 2, 3, 6, 7, 8)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
TURN_MAP = {
    (1, 1): (1, 0),
    (1, -1): (0, -1),
    (-1, 1): (0, 1),
    (-1, -1): (-1, 0),
}
RAY_SPECS = {
    "NW": {"corner_offset": (1, 1), "diag": (1, 1), "turn": (1, 0)},
    "NE": {"corner_offset": (1, 0), "diag": (1, -1), "turn": (0, -1)},
    "SW": {"corner_offset": (0, 1), "diag": (-1, 1), "turn": (0, 1)},
    "SE": {"corner_offset": (0, 0), "diag": (-1, -1), "turn": (-1, 0)},
}
MODE_CHOICES = (
    "single_direct",
    "single_prism",
    "multi_direct",
    "multi_mixed",
)

# Generator knobs.
GRID_MIN = 18
GRID_MAX = 24
TARGET_MARGIN = 7
DIRECT_LEN_RANGE = (2, 5)
PRISM_DIAG_LEN_RANGE = (2, 4)
ORTH_LEN_RANGE = (2, 4)
MAX_BASE_ATTEMPTS = 120
MAX_EXTRA_ATTEMPTS = 80
MAX_GENERATION_ATTEMPTS = 400
EXTRA_RECEIVER_PROB = 0.45
EXTRA_PRISM_PROB = 0.35


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _tower_cells(top_left: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = top_left
    return [(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)]


def _find_towers(g: np.ndarray) -> list[dict]:
    towers: list[dict] = []
    seen_colors: set[int] = set()
    specials = {BG, YELLOW, GREY, MAROON}
    for color in map(int, np.unique(g)):
        if color in specials:
            continue
        comps = _components(g == color)
        assert len(comps) == 1, "each tower color must belong to one component"
        comp = comps[0]
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert len(comp) == 4 and (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "tower must be a solid 2x2 block"
        assert all(int(g[r, c]) == color for r, c in _tower_cells((r0, c0)))
        assert color not in seen_colors, "tower colors must be distinct"
        seen_colors.add(color)
        towers.append({"color": color, "top_left": (r0, c0)})
    assert towers, "expected at least one tower"
    return towers


def _singleton_cells(g: np.ndarray, color: int) -> set[tuple[int, int]]:
    cells = {(int(r), int(c)) for r, c in np.argwhere(g == color)}
    for comp in _components(g == color):
        assert len(comp) == 1, "markers must be single pixels"
    return cells


def _tower_rays(tower: dict) -> list[tuple[tuple[int, int], tuple[int, int], int]]:
    r0, c0 = tower["top_left"]
    color = int(tower["color"])
    return [
        ((r0, c0), (-1, -1), color),
        ((r0, c0 + 1), (-1, 1), color),
        ((r0 + 1, c0), (1, -1), color),
        ((r0 + 1, c0 + 1), (1, 1), color),
    ]


def _simulate(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert int(np.min(g)) >= 0 and int(np.max(g)) <= 9, "ARC colors must stay in 0..9"
    towers = _find_towers(g)
    prisms = _singleton_cells(g, YELLOW)
    receivers = _singleton_cells(g, GREY)

    out = g.copy()
    painted: set[tuple[int, int]] = set()
    prism_hits: list[tuple[int, int]] = []
    receiver_hits: dict[tuple[int, int], list[int]] = {cell: [] for cell in receivers}
    active = [
        {"pos": pos, "dir": direction, "color": color}
        for tower in towers
        for pos, direction, color in _tower_rays(tower)
    ]

    limit = g.shape[0] * g.shape[1] * 8
    for _ in range(limit):
        if not active:
            break

        proposals: list[tuple[str, int, tuple[int, int]]] = []
        paint_targets: dict[tuple[int, int], list[int]] = {}
        for idx, ray in enumerate(active):
            r, c = ray["pos"]
            dr, dc = ray["dir"]
            nr, nc = r + dr, c + dc
            if not _in_bounds(g.shape, nr, nc):
                continue
            cell = (nr, nc)
            if cell in prisms:
                assert (dr, dc) in TURN_MAP, "prisms must be hit by diagonal rays"
                proposals.append(("turn", idx, cell))
                continue
            if cell in receivers:
                proposals.append(("receiver", idx, cell))
                continue
            if int(g[nr, nc]) != BG or cell in painted:
                proposals.append(("stop", idx, cell))
                continue
            proposals.append(("paint", idx, cell))
            paint_targets.setdefault(cell, []).append(idx)

        for ids in paint_targets.values():
            assert len(ids) == 1, "simultaneous ray-cell collisions are not expected"

        next_active: list[dict] = []
        for kind, idx, cell in proposals:
            ray = active[idx]
            if kind == "paint":
                out[cell] = int(ray["color"])
                painted.add(cell)
                next_active.append({"pos": cell, "dir": ray["dir"], "color": int(ray["color"])})
            elif kind == "turn":
                prism_hits.append(cell)
                next_active.append(
                    {
                        "pos": cell,
                        "dir": TURN_MAP[tuple(ray["dir"])],
                        "color": int(ray["color"]),
                    }
                )
            elif kind == "receiver":
                receiver_hits[cell].append(int(ray["color"]))
        active = next_active
    else:
        raise AssertionError("ray simulation failed to terminate")

    for cell, hits in receiver_hits.items():
        if len(hits) == 1:
            out[cell] = int(hits[0])
        elif len(hits) > 1:
            out[cell] = MAROON
    return {
        "out": out,
        "towers": towers,
        "prisms": prisms,
        "receivers": receivers,
        "receiver_hits": receiver_hits,
        "prism_hits": prism_hits,
        "painted": painted,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit diagonal signals from each 2x2 tower, turn them at yellow prisms, and recolor grey receivers.

    Input structure:
    - Black background.
    - Towers are solid 2x2 monochrome blocks in distinct non-special colors.
    - Prisms are single yellow pixels.
    - Receivers are single grey pixels.

    Rule:
    - Each tower fires from its four corner pixels along the four diagonal directions.
    - Rays paint only background cells.
    - A diagonal ray that reaches a yellow prism turns to the mapped orthogonal direction and continues.
    - Rays stop at any non-background non-yellow cell or at an already painted ray trace.
    - A receiver hit by one ray takes that ray color; a receiver hit by multiple rays becomes maroon.
    """
    return _simulate(grid)["out"]


def _can_place_tower(scene: np.ndarray, top_left: tuple[int, int]) -> bool:
    r0, c0 = top_left
    h, w = scene.shape
    if not _in_bounds(scene.shape, r0, c0) or not _in_bounds(scene.shape, r0 + 1, c0 + 1):
        return False
    if np.any(scene[r0 : r0 + 2, c0 : c0 + 2] != BG):
        return False
    rr0 = max(0, r0 - 1)
    rr1 = min(h, r0 + 3)
    cc0 = max(0, c0 - 1)
    cc1 = min(w, c0 + 3)
    return not bool(np.any(scene[rr0:rr1, cc0:cc1] != BG))


def _can_place_marker(scene: np.ndarray, cell: tuple[int, int]) -> bool:
    r, c = cell
    if not _in_bounds(scene.shape, r, c) or int(scene[r, c]) != BG:
        return False
    for nr in range(r - 1, r + 2):
        for nc in range(c - 1, c + 2):
            if _in_bounds(scene.shape, nr, nc) and int(scene[nr, nc]) != BG:
                return False
    return True


def _paint_tower(scene: np.ndarray, top_left: tuple[int, int], color: int) -> None:
    r0, c0 = top_left
    scene[r0 : r0 + 2, c0 : c0 + 2] = int(color)


def _target_coord(limit: int) -> int:
    lo = TARGET_MARGIN
    hi = limit - TARGET_MARGIN - 1
    assert lo <= hi, "grid is too small for the planned central receiver"
    return random.randint(lo, hi)


def _plan_signal(
    scene: np.ndarray,
    reserved_path: set[tuple[int, int]],
    target: tuple[int, int],
    spec_name: str,
    use_prism: bool,
    color: int,
) -> bool:
    spec = RAY_SPECS[spec_name]
    dr, dc = spec["diag"]
    off_r, off_c = spec["corner_offset"]
    turn_r, turn_c = spec["turn"]

    for _ in range(MAX_BASE_ATTEMPTS):
        if use_prism:
            diag_len = random.randint(*PRISM_DIAG_LEN_RANGE)
            orth_len = random.randint(*ORTH_LEN_RANGE)
            prism = (target[0] - orth_len * turn_r, target[1] - orth_len * turn_c)
            emitter = (prism[0] - diag_len * dr, prism[1] - diag_len * dc)
            path_cells = [
                (emitter[0] + step * dr, emitter[1] + step * dc)
                for step in range(1, diag_len)
            ] + [
                (prism[0] + step * turn_r, prism[1] + step * turn_c)
                for step in range(1, orth_len)
            ]
        else:
            diag_len = random.randint(*DIRECT_LEN_RANGE)
            prism = None
            emitter = (target[0] - diag_len * dr, target[1] - diag_len * dc)
            path_cells = [
                (emitter[0] + step * dr, emitter[1] + step * dc)
                for step in range(1, diag_len)
            ]

        top_left = (emitter[0] - off_r, emitter[1] - off_c)
        tower_cells = _tower_cells(top_left)
        if not _can_place_tower(scene, top_left):
            continue
        if any(not _in_bounds(scene.shape, r, c) for r, c in path_cells):
            continue
        if any(cell in reserved_path for cell in path_cells):
            continue
        if any(int(scene[r, c]) != BG for r, c in path_cells):
            continue
        if prism is not None and not _can_place_marker(scene, prism):
            continue

        if int(scene[target]) == BG:
            if not _can_place_marker(scene, target):
                continue
        elif int(scene[target]) != GREY:
            continue

        if any(cell == target for cell in tower_cells):
            continue

        _paint_tower(scene, top_left, color)
        if prism is not None:
            scene[prism] = YELLOW
        if int(scene[target]) == BG:
            scene[target] = GREY
        reserved_path.update(path_cells)
        return True
    return False


def _summary(scene: np.ndarray) -> dict | None:
    try:
        sim = _simulate(scene)
    except Exception:
        return None
    hit_counts = {cell: len(colors) for cell, colors in sim["receiver_hits"].items()}
    return {
        "sim": sim,
        "tower_count": len(sim["towers"]),
        "prism_count": len(sim["prisms"]),
        "receiver_count": len(sim["receivers"]),
        "hit_counts": hit_counts,
        "prism_hit_count": len(sim["prism_hits"]),
    }


def _mode_ok(summary: dict, mode: str, *, require_prism_present: bool) -> bool:
    if summary["tower_count"] < 2 or summary["receiver_count"] < 1:
        return False
    if require_prism_present and summary["prism_count"] < 1:
        return False

    hit_counts = list(summary["hit_counts"].values())
    if any(count > 2 for count in hit_counts):
        return False

    has_single = any(count == 1 for count in hit_counts)
    has_double = any(count == 2 for count in hit_counts)
    want_prism = mode in {"single_prism", "multi_mixed"}
    want_multi = mode in {"multi_direct", "multi_mixed"}

    if want_prism:
        if summary["prism_hit_count"] < 1:
            return False
    elif summary["prism_hit_count"] != 0:
        return False

    if want_multi:
        return has_double
    return has_single and not has_double


def _decoy_tower_positions(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    return [(0, 0), (0, w - 2), (h - 2, 0), (h - 2, w - 2)]


def _try_add_decoy_tower(
    scene: np.ndarray,
    used_colors: set[int],
    mode: str,
    require_prism_present: bool,
) -> bool:
    positions = _decoy_tower_positions(scene.shape)
    random.shuffle(positions)
    colors = [color for color in TOWER_COLORS if color not in used_colors]
    random.shuffle(colors)
    for top_left in positions:
        if not _can_place_tower(scene, top_left):
            continue
        for color in colors:
            trial = scene.copy()
            _paint_tower(trial, top_left, color)
            summary = _summary(trial)
            if summary is None:
                continue
            if _mode_ok(summary, mode, require_prism_present=require_prism_present):
                scene[:, :] = trial
                used_colors.add(color)
                return True
    return False


def _add_safe_marker(
    scene: np.ndarray,
    color: int,
    forbidden_path: set[tuple[int, int]],
) -> bool:
    h, w = scene.shape
    candidates = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if (r, c) not in forbidden_path and _can_place_marker(scene, (r, c))
    ]
    random.shuffle(candidates)
    if not candidates:
        return False
    scene[candidates[0]] = int(color)
    return True


def _build_scene() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    scene = np.full((h, w), BG, dtype=int)
    reserved_path: set[tuple[int, int]] = set()
    mode = random.choice(MODE_CHOICES)

    if mode == "single_direct":
        plans = [(random.choice(tuple(RAY_SPECS)), False)]
    elif mode == "single_prism":
        plans = [(random.choice(tuple(RAY_SPECS)), True)]
    elif mode == "multi_direct":
        names = random.sample(tuple(RAY_SPECS), 2)
        plans = [(names[0], False), (names[1], False)]
    else:
        names = random.sample(tuple(RAY_SPECS), 2)
        prism_idx = random.randrange(2)
        plans = [(names[i], i == prism_idx) for i in range(2)]

    used_colors = set(random.sample(TOWER_COLORS, len(plans)))
    tower_colors = list(used_colors)
    random.shuffle(tower_colors)

    placed = False
    for _ in range(MAX_BASE_ATTEMPTS):
        trial = np.full((h, w), BG, dtype=int)
        path_cells: set[tuple[int, int]] = set()
        target = (_target_coord(h), _target_coord(w))
        ok = True
        for color, (spec_name, use_prism) in zip(tower_colors, plans):
            if not _plan_signal(trial, path_cells, target, spec_name, use_prism, color):
                ok = False
                break
        if not ok:
            continue
        summary = _summary(trial)
        if summary is None:
            continue
        if _mode_ok(summary, mode, require_prism_present=(mode in {"single_prism", "multi_mixed"})):
            scene = trial
            reserved_path = path_cells
            placed = True
            break
    if not placed:
        return None

    target_tower_count = random.randint(2, 4)
    require_prism_present = mode in {"single_prism", "multi_mixed"}
    while len(_find_towers(scene)) < target_tower_count:
        if not _try_add_decoy_tower(scene, used_colors, mode, require_prism_present):
            break

    summary = _summary(scene)
    if summary is None or not _mode_ok(summary, mode, require_prism_present=require_prism_present):
        return None

    if mode in {"single_direct", "multi_direct"}:
        if not _add_safe_marker(scene, YELLOW, summary["sim"]["painted"]):
            return None
        require_prism_present = True
        summary = _summary(scene)
        if summary is None or not _mode_ok(summary, mode, require_prism_present=True):
            return None
    elif random.random() < EXTRA_PRISM_PROB:
        trial = scene.copy()
        if _add_safe_marker(trial, YELLOW, summary["sim"]["painted"]):
            extra_summary = _summary(trial)
            if extra_summary is not None and _mode_ok(extra_summary, mode, require_prism_present=True):
                scene = trial
                summary = extra_summary

    if random.random() < EXTRA_RECEIVER_PROB:
        trial = scene.copy()
        if _add_safe_marker(trial, GREY, summary["sim"]["painted"]):
            extra_summary = _summary(trial)
            if extra_summary is not None and _mode_ok(extra_summary, mode, require_prism_present=True):
                scene = trial
                summary = extra_summary

    return scene if _mode_ok(summary, mode, require_prism_present=True) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-tower scene")
