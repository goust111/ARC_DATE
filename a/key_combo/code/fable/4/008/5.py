from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
GREY = 5
DRAW_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 700

L_SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
)

NON_L_SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 1)),
)


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []

    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue

        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        cells: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for rr, cc in _neighbors4(r, c, h, w):
                if bool(mask[rr, cc]) and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(np.array(cells, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _shape_signature(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox_coords(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _is_l_triomino(coords: np.ndarray) -> bool:
    if len(coords) != 3:
        return False
    r0, c0, r1, c1 = _bbox_coords(coords)
    if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
        return False
    return _shape_signature(coords) in L_SHAPES


def _grey_component_labels(g: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    comps = _components(g == GREY)
    labels = np.full(g.shape, -1, dtype=int)
    for idx, coords in enumerate(comps):
        labels[coords[:, 0], coords[:, 1]] = int(idx)
    return labels, comps


def _find_cavities(
    g: np.ndarray, grey_id: np.ndarray
) -> dict[int, dict[str, object]]:
    h, w = g.shape
    cavities: dict[int, list[dict[str, object]]] = {}

    for coords in _components(g == BLACK):
        if not _is_l_triomino(coords):
            continue

        cset = {(int(r), int(c)) for r, c in coords}
        grey_neighbors: set[int] = set()
        enclosed = True
        for r, c in cset:
            for rr, cc in _neighbors4(r, c, h, w):
                if (rr, cc) in cset:
                    continue
                gid = int(grey_id[rr, cc])
                if gid < 0:
                    enclosed = False
                    break
                grey_neighbors.add(gid)
            if not enclosed:
                break

        if enclosed and len(grey_neighbors) == 1:
            sid = next(iter(grey_neighbors))
            cavities.setdefault(sid, []).append(
                {
                    "sid": sid,
                    "coords": coords,
                    "shape": _shape_signature(coords),
                }
            )

    assert cavities, "expected at least one enclosed L-shaped black cavity"
    out: dict[int, dict[str, object]] = {}
    for sid, records in cavities.items():
        assert len(records) == 1, "each grey lock structure must have one L cavity"
        out[int(sid)] = records[0]
    return out


def _find_inlays(
    g: np.ndarray, grey_id: np.ndarray, lock_sids: set[int]
) -> dict[int, tuple[tuple[int, int], int]]:
    h, w = g.shape
    found: dict[int, list[tuple[tuple[int, int], int]]] = {sid: [] for sid in lock_sids}

    for r, c in np.argwhere((g != BLACK) & (g != GREY)):
        r = int(r)
        c = int(c)
        touching = {
            int(grey_id[rr, cc])
            for rr, cc in _neighbors4(r, c, h, w)
            if int(grey_id[rr, cc]) in lock_sids
        }
        if len(touching) == 1:
            sid = next(iter(touching))
            found[sid].append(((r, c), int(g[r, c])))

    out: dict[int, tuple[tuple[int, int], int]] = {}
    for sid in lock_sids:
        assert len(found[sid]) == 1, "each lock must have exactly one colored inlay pixel"
        out[sid] = found[sid][0]
    return out


def _parse_locks(g: np.ndarray) -> list[dict[str, object]]:
    grey_id, _grey_comps = _grey_component_labels(g)
    cavities = _find_cavities(g, grey_id)
    inlays = _find_inlays(g, grey_id, set(cavities))

    locks: list[dict[str, object]] = []
    for sid in sorted(cavities):
        inlay_pos, inlay_color = inlays[sid]
        locks.append(
            {
                "sid": sid,
                "cavity": np.asarray(cavities[sid]["coords"], dtype=int),
                "shape": cavities[sid]["shape"],
                "inlay": inlay_pos,
                "color": int(inlay_color),
            }
        )
    return locks


def _external_color_components(
    g: np.ndarray, inlay_positions: set[tuple[int, int]]
) -> list[dict[str, object]]:
    comps: list[dict[str, object]] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) not in (BLACK, GREY)):
        mask = g == int(color)
        for r, c in inlay_positions:
            if int(g[r, c]) == color:
                mask[r, c] = False
        for coords in _components(mask):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "is_l": _is_l_triomino(coords),
                    "shape": _shape_signature(coords),
                }
            )
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock each matching colored L triomino into the grey lock whose cavity and inlay jointly identify it.

    Input structure:
    - The background is black, with one or more detached grey structures.
    - Each grey structure contains exactly one enclosed three-cell black L cavity and has exactly one colored single-pixel inlay on its surface.
    - Outside the grey structures are detached colored components: some are L triomino keys and others are non-L distractors.

    Rule:
    - A key matches a lock only when its color equals that lock's inlay color and its normalized L orientation equals the black cavity orientation under translation alone.
    - All outside colored components disappear from their original positions.
    - For every matched lock, fill the black cavity with the key color and turn the inlay pixel grey; keep all other grey structure cells and background unchanged.

    Key insight:
    - The lock has two separated teeth: the cavity gives the unrotated shape, while the inlay gives the color, so either cue alone selects visible distractors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30 by 30"

    locks = _parse_locks(g)
    assert locks, "expected at least one lock"

    inlay_positions = {lock["inlay"] for lock in locks}
    external = _external_color_components(g, inlay_positions)

    lock_counts = Counter((int(lock["color"]), lock["shape"]) for lock in locks)
    key_counts = Counter(
        (int(comp["color"]), comp["shape"])
        for comp in external
        if bool(comp["is_l"])
    )

    for key, count in lock_counts.items():
        assert key_counts[key] == count, "each lock must have exactly one matching external key"
    for key, count in key_counts.items():
        if key in lock_counts:
            assert count == lock_counts[key], "matching keys must not be ambiguous extras"

    out = g.copy()
    for comp in external:
        coords = np.asarray(comp["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = BLACK

    for lock in locks:
        coords = np.asarray(lock["cavity"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = int(lock["color"])
        ir, ic = lock["inlay"]
        out[int(ir), int(ic)] = GREY

    return out


# ---------------- generator ----------------


def _shape_size(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _c in shape) + 1, max(c for _r, c in shape) + 1


def _paint_shape(
    grid: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    origin: tuple[int, int],
    color: int,
) -> list[tuple[int, int]]:
    r0, c0 = origin
    cells: list[tuple[int, int]] = []
    for dr, dc in shape:
        r = int(r0 + dr)
        c = int(c0 + dc)
        grid[r, c] = int(color)
        cells.append((r, c))
    return cells


def _place_box(
    blocked: np.ndarray,
    height: int,
    width: int,
    *,
    border: int = 1,
    padding: int = 1,
) -> tuple[int, int] | None:
    h, w = blocked.shape
    candidates: list[tuple[int, int]] = []
    for r in range(border, h - border - height + 1):
        for c in range(border, w - border - width + 1):
            rr0 = max(0, r - padding)
            cc0 = max(0, c - padding)
            rr1 = min(h, r + height + padding)
            cc1 = min(w, c + width + padding)
            if not bool(blocked[rr0:rr1, cc0:cc1].any()):
                candidates.append((r, c))

    if not candidates:
        return None

    r, c = random.choice(candidates)
    rr0 = max(0, r - padding)
    cc0 = max(0, c - padding)
    rr1 = min(h, r + height + padding)
    cc1 = min(w, c + width + padding)
    blocked[rr0:rr1, cc0:cc1] = True
    return int(r), int(c)


def _make_lock_patch(
    shape: tuple[tuple[int, int], ...], color: int
) -> tuple[np.ndarray, list[tuple[int, int]], tuple[int, int]]:
    height = random.randint(6, 9)
    width = random.randint(6, 9)
    patch = np.full((height, width), GREY, dtype=int)

    cr = random.randint(1, height - 3)
    cc = random.randint(1, width - 3)
    cavity = [(cr + r, cc + c) for r, c in shape]
    cavity_set = set(cavity)
    for r, c in cavity:
        patch[r, c] = BLACK

    border_cells: list[tuple[int, int]] = []
    for r in range(height):
        for c in range(width):
            if r not in (0, height - 1) and c not in (0, width - 1):
                continue
            if (r in (0, height - 1)) and (c in (0, width - 1)):
                continue
            if (r, c) in cavity_set:
                continue
            if any(abs(r - rr) + abs(c - cc2) == 1 for rr, cc2 in cavity_set):
                continue
            border_cells.append((r, c))

    assert border_cells, "lock patch must have room for an inlay"
    inlay = random.choice(border_cells)
    patch[inlay[0], inlay[1]] = int(color)
    return patch, cavity, inlay


def _render_expected(
    inp: np.ndarray,
    locks: list[dict[str, object]],
    external_cells: list[tuple[int, int]],
) -> np.ndarray:
    out = inp.copy()
    for r, c in external_cells:
        out[int(r), int(c)] = BLACK
    for lock in locks:
        for r, c in lock["cavity"]:
            out[int(r), int(c)] = int(lock["color"])
        ir, ic = lock["inlay"]
        out[int(ir), int(ic)] = GREY
    return out


def _place_lock(
    grid: np.ndarray,
    blocked: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    color: int,
) -> dict[str, object] | None:
    patch, rel_cavity, rel_inlay = _make_lock_patch(shape, color)
    loc = _place_box(blocked, patch.shape[0], patch.shape[1], border=1, padding=1)
    if loc is None:
        return None

    r0, c0 = loc
    grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch
    cavity = [(r0 + r, c0 + c) for r, c in rel_cavity]
    inlay = (r0 + rel_inlay[0], c0 + rel_inlay[1])
    return {"color": int(color), "shape": shape, "cavity": cavity, "inlay": inlay}


def _place_external(
    grid: np.ndarray,
    blocked: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    color: int,
) -> list[tuple[int, int]] | None:
    height, width = _shape_size(shape)
    loc = _place_box(blocked, height, width, border=1, padding=1)
    if loc is None:
        return None
    return _paint_shape(grid, shape, loc, int(color))


def _distractor_specs(
    locks: list[dict[str, object]],
    spare_colors: list[int],
) -> list[tuple[tuple[tuple[int, int], ...], int]]:
    specs: list[tuple[tuple[tuple[int, int], ...], int]] = []

    first = locks[0]
    wrong_shapes = [shape for shape in L_SHAPES if shape != first["shape"]]
    specs.append((random.choice(wrong_shapes), int(first["color"])))

    second = locks[1]
    specs.append((second["shape"], int(random.choice(spare_colors))))

    for _ in range(random.randint(2, 4)):
        specs.append((random.choice(NON_L_SHAPES), int(random.choice(DRAW_COLORS))))

    for _ in range(random.randint(1, 3)):
        lock = random.choice(locks)
        if random.random() < 0.5:
            wrong = [shape for shape in L_SHAPES if shape != lock["shape"]]
            specs.append((random.choice(wrong), int(lock["color"])))
        else:
            specs.append((lock["shape"], int(random.choice(spare_colors))))

    random.shuffle(specs)
    return specs


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    if inp.ndim != 2 or expected.ndim != 2 or inp.shape != expected.shape:
        return False
    if max(inp.shape) > 30:
        return False

    try:
        locks = _parse_locks(inp)
        solved = main(inp)
    except Exception:
        return False

    if not np.array_equal(solved, expected):
        return False
    if len(locks) < 2:
        return False

    inlays = {lock["inlay"] for lock in locks}
    external = _external_color_components(inp, inlays)
    if not any(not bool(comp["is_l"]) for comp in external):
        return False

    same_color_wrong_shape = False
    same_shape_wrong_color = False
    for comp in external:
        if not bool(comp["is_l"]):
            continue
        comp_color = int(comp["color"])
        comp_shape = comp["shape"]
        for lock in locks:
            if comp_color == int(lock["color"]) and comp_shape != lock["shape"]:
                same_color_wrong_shape = True
            if comp_shape == lock["shape"] and comp_color != int(lock["color"]):
                same_shape_wrong_color = True

    if not same_color_wrong_shape or not same_shape_wrong_color:
        return False

    changed = np.argwhere(inp != expected)
    if len(changed) < 8:
        return False

    for lock in locks:
        if any(int(inp[r, c]) != BLACK for r, c in lock["cavity"]):
            return False
        if any(int(expected[r, c]) != int(lock["color"]) for r, c in lock["cavity"]):
            return False
        ir, ic = lock["inlay"]
        if int(expected[ir, ic]) != GREY:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(22, 30)
    width = random.randint(22, 30)
    grid = np.zeros((height, width), dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    n_locks = random.choice([2, 2, 3, 3, 4])
    lock_colors = random.sample(DRAW_COLORS, n_locks)
    lock_shapes = random.sample(L_SHAPES, n_locks)
    spare_colors = [c for c in DRAW_COLORS if c not in lock_colors]

    locks: list[dict[str, object]] = []
    for color, shape in zip(lock_colors, lock_shapes):
        lock = _place_lock(grid, blocked, shape, int(color))
        if lock is None:
            return None
        locks.append(lock)

    external_specs: list[tuple[tuple[tuple[int, int], ...], int]] = [
        (lock["shape"], int(lock["color"])) for lock in locks
    ]
    external_specs.extend(_distractor_specs(locks, spare_colors))
    random.shuffle(external_specs)

    external_cells: list[tuple[int, int]] = []
    for shape, color in external_specs:
        cells = _place_external(grid, blocked, shape, int(color))
        if cells is None:
            return None
        external_cells.extend(cells)

    expected = _render_expected(grid, locks, external_cells)
    if not _semantic_checks(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid split double-tooth lock input")
