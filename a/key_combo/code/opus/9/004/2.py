from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, find_components, get_bg


PALETTE = [Color.BLUE, Color.GREEN, Color.YELLOW]
RETRY_LIMIT = 800

SHAPES = [
    np.array([(0, 0), (1, 0), (1, 1)], dtype=int),
    np.array([(0, 1), (1, 0), (1, 1)], dtype=int),
    np.array([(0, 0), (0, 1), (1, 1), (2, 1)], dtype=int),
    np.array([(0, 0), (1, 0), (1, 1), (2, 0)], dtype=int),
    np.array([(0, 1), (1, 0), (1, 1), (1, 2)], dtype=int),
    np.array([(0, 0), (0, 1), (1, 1), (1, 2)], dtype=int),
    np.array([(0, 0), (1, 0), (1, 1), (2, 1)], dtype=int),
    np.array([(0, 1), (1, 0), (1, 1), (2, 1), (2, 2)], dtype=int),
    np.array([(0, 0), (0, 1), (1, 1), (2, 0), (2, 1)], dtype=int),
    np.array([(0, 0), (0, 1), (1, 0), (1, 2), (2, 0), (2, 1)], dtype=int),
]


def _normalize(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=int)
    mins = coords.min(axis=0)
    return coords - mins


def _shape_variants(coords: np.ndarray) -> list[np.ndarray]:
    pts = np.asarray(coords, dtype=int)
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for k in range(4):
        rotated = pts.copy()
        for _ in range(k):
            rotated = np.stack([rotated[:, 1], -rotated[:, 0]], axis=1)
        for flip in (False, True):
            cur = rotated.copy()
            if flip:
                cur[:, 1] *= -1
            cur = _normalize(cur)
            key = tuple(sorted(map(tuple, cur.tolist())))
            if key not in seen:
                seen.add(key)
                variants.append(cur)
    return variants


ALL_SHAPES = [variant for shape in SHAPES for variant in _shape_variants(shape)]


def _diamond_geometry(grid: np.ndarray) -> tuple[tuple[int, int], int, np.ndarray]:
    red = np.argwhere(grid == Color.RED)
    assert len(red) >= 8, "expected a complete red diamond frame"
    r0, c0 = red.min(axis=0)
    r1, c1 = red.max(axis=0)
    assert (r1 - r0) == (c1 - c0) and (r1 - r0) % 2 == 0
    center = ((int(r0) + int(r1)) // 2, (int(c0) + int(c1)) // 2)
    radius = (int(r1) - int(r0)) // 2
    rr, cc = np.indices(grid.shape)
    dist = np.abs(rr - center[0]) + np.abs(cc - center[1])
    assert np.array_equal(grid == Color.RED, dist == radius), "red pixels must form the diamond frame"
    return center, radius, dist < radius


def _object_cells(coords: np.ndarray, shift: int = 0) -> np.ndarray:
    out = np.asarray(coords, dtype=int).copy()
    out[:, 0] += int(shift)
    return out


def _paint(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _sorted_movers(components: list[dict]) -> list[dict]:
    return sorted(
        components,
        key=lambda comp: (
            -int(comp["coords"][:, 0].max()),
            int(comp["coords"][:, 0].min()),
            int(comp["coords"][:, 1].min()),
        ),
    )


def _simulate(
    grid_shape: tuple[int, int],
    movers: list[dict],
    blockers: np.ndarray,
    interior: np.ndarray,
    *,
    use_diamond_boundary: bool,
    use_blockers: bool,
    use_collisions: bool,
) -> tuple[list[dict], np.ndarray]:
    h, w = map(int, grid_shape)
    occupied_blockers = blockers.copy() if use_blockers else np.zeros((h, w), dtype=bool)
    settled = np.zeros((h, w), dtype=bool)
    placements: list[dict] = []

    def valid(next_coords: np.ndarray) -> tuple[bool, set[str]]:
        reasons: set[str] = set()
        in_bounds = (
            (next_coords[:, 0] >= 0)
            & (next_coords[:, 0] < h)
            & (next_coords[:, 1] >= 0)
            & (next_coords[:, 1] < w)
        )
        if not np.all(in_bounds):
            reasons.add("boundary")
            return False, reasons
        if use_diamond_boundary:
            if np.any(~interior[next_coords[:, 0], next_coords[:, 1]]):
                reasons.add("boundary")
        if use_blockers and np.any(occupied_blockers[next_coords[:, 0], next_coords[:, 1]]):
            reasons.add("blocker")
        if use_collisions and np.any(settled[next_coords[:, 0], next_coords[:, 1]]):
            reasons.add("collision")
        return not reasons, reasons

    for mover in movers:
        base = np.asarray(mover["coords"], dtype=int)
        shift = 0
        stop_reasons: set[str] = set()
        while True:
            next_coords = _object_cells(base, shift + 1)
            okay, reasons = valid(next_coords)
            if not okay:
                stop_reasons = reasons
                break
            shift += 1
        final_coords = _object_cells(base, shift)
        if use_collisions:
            settled[final_coords[:, 0], final_coords[:, 1]] = True
        placements.append(
            {
                "color": int(mover["color"]),
                "start": base,
                "final": final_coords,
                "shift": int(shift),
                "reasons": stop_reasons,
            }
        )

    return placements, settled | occupied_blockers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the colored objects inside the red diamond while grey single-pixel blockers stay fixed.

    Input structure:
    - A black background contains one complete one-pixel-thick red diamond frame.
    - Inside the diamond are 2-3 irregular 4-connected colored objects in blue, green, or yellow.
    - Grey single pixels inside the diamond act as stationary blockers; all non-frame items start strictly inside the frame and separated.

    Rule:
    - Treat each grey single pixel as a fixed blocker and each colored component of area at least three as a falling object.
    - Process falling objects from the one with the greatest lowest row upward, translating each object straight downward.
    - An object stops when its next downward step would place any pixel on or outside the diamond interior, onto a grey blocker, or onto a previously settled falling object.
    - Keep the red frame, the grey blockers, and every settled object in their final colors; all other cells remain black.

    Key insight:
    - The red diamond is the gravity container, so the effective floor is a V shape whose height depends on column, and falling order matters because higher objects can land on lower ones.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == Color.BLACK

    _center, _radius, interior = _diamond_geometry(g)
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    blockers: list[dict] = []
    movers: list[dict] = []
    for comp in comps:
        color = int(comp["color"])
        area = int(comp["area"])
        if color == Color.RED:
            continue
        if color == Color.GREY:
            assert area == 1, "grey components must be single-pixel blockers"
            blockers.append(comp)
        else:
            assert color in PALETTE, "unexpected mover color"
            assert area >= 3, "colored components must be movers"
            movers.append(comp)

    assert len(movers) >= 2
    assert blockers

    blocker_mask = np.zeros_like(g, dtype=bool)
    for blocker in blockers:
        coords = np.asarray(blocker["coords"], dtype=int)
        assert interior[coords[0, 0], coords[0, 1]]
        blocker_mask[coords[:, 0], coords[:, 1]] = True

    movers = _sorted_movers(movers)
    placements, _ = _simulate(
        g.shape,
        movers,
        blocker_mask,
        interior,
        use_diamond_boundary=True,
        use_blockers=True,
        use_collisions=True,
    )

    out = g.copy()
    for mover in movers:
        coords = np.asarray(mover["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = bg
    for placement in placements:
        _paint(out, placement["final"], placement["color"])
    return out


def _entity_neighbors(coords: np.ndarray) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in map(tuple, np.asarray(coords, dtype=int)):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out.add((int(r) + dr, int(c) + dc))
    return out


def _place_shape(
    interior: np.ndarray,
    forbidden: set[tuple[int, int]],
    shape: np.ndarray,
    row_choices: list[int],
    col_choices: list[int],
) -> np.ndarray | None:
    for r0 in row_choices:
        for c0 in col_choices:
            coords = shape + np.array([r0, c0], dtype=int)
            if np.any(~interior[coords[:, 0], coords[:, 1]]):
                continue
            if any((int(r), int(c)) in forbidden for r, c in coords):
                continue
            return coords
    return None


def _random_shape(min_width: int = 1) -> np.ndarray:
    candidates = [shape for shape in ALL_SHAPES if (shape[:, 1].max() - shape[:, 1].min() + 1) >= min_width]
    return np.asarray(random.choice(candidates), dtype=int)


def _reserve(forbidden: set[tuple[int, int]], coords: np.ndarray) -> None:
    forbidden.update(_entity_neighbors(coords))


def _column_bottoms(interior: np.ndarray) -> dict[int, int]:
    bottoms: dict[int, int] = {}
    for c in range(interior.shape[1]):
        rows = np.flatnonzero(interior[:, c])
        if len(rows):
            bottoms[int(c)] = int(rows.max())
    return bottoms


def _candidate_placements(
    interior: np.ndarray,
    forbidden: set[tuple[int, int]],
    shape: np.ndarray,
    *,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
) -> list[np.ndarray]:
    h, w = interior.shape
    sh = int(shape[:, 0].max()) + 1
    sw = int(shape[:, 1].max()) + 1
    row_max = min(int(row_max), h - sh)
    col_max = min(int(col_max), w - sw)
    row_min = max(0, int(row_min))
    col_min = max(0, int(col_min))
    if row_min > row_max or col_min > col_max:
        return []

    out: list[np.ndarray] = []
    for r0 in range(row_min, row_max + 1):
        for c0 in range(col_min, col_max + 1):
            coords = shape + np.array([r0, c0], dtype=int)
            if np.any(~interior[coords[:, 0], coords[:, 1]]):
                continue
            if any((int(r), int(c)) in forbidden for r, c in coords):
                continue
            out.append(coords)
    random.shuffle(out)
    return out


def _generate_case() -> np.ndarray | None:
    size = random.choice([15, 17, 19, 21])
    margin = random.choice([1, 2])
    radius = size // 2 - margin
    if radius < 5:
        return None

    center = (size // 2, size // 2)
    rr, cc = np.indices((size, size))
    dist = np.abs(rr - center[0]) + np.abs(cc - center[1])
    interior = dist < radius
    grid = np.zeros((size, size), dtype=int)
    grid[dist == radius] = Color.RED

    colors = random.sample(PALETTE, 3)
    forbidden = set(zip(*np.where(~interior)))
    forbidden.update(zip(*np.where(dist == radius)))
    bottoms = _column_bottoms(interior)
    upper_row_max = center[0] - 2
    side = random.choice(["left", "right"])

    a_shape = _random_shape(min_width=2)
    a_h = int(a_shape[:, 0].max()) + 1
    a_w = int(a_shape[:, 1].max()) + 1
    a_col_min = 1 if side == "left" else center[1] + 1
    a_col_max = center[1] - a_w - 1 if side == "left" else size - a_w - 2
    a_candidates = _candidate_placements(
        interior,
        forbidden,
        a_shape,
        row_min=max(1, center[0] - radius // 2 - 1),
        row_max=upper_row_max - a_h + 1,
        col_min=a_col_min,
        col_max=a_col_max,
    )
    a_filtered: list[np.ndarray] = []
    for coords in a_candidates:
        mover = [{"color": colors[0], "coords": coords}]
        actual, _ = _simulate(
            grid.shape,
            mover,
            np.zeros_like(grid, dtype=bool),
            interior,
            use_diamond_boundary=True,
            use_blockers=False,
            use_collisions=True,
        )
        rect, _ = _simulate(
            grid.shape,
            mover,
            np.zeros_like(grid, dtype=bool),
            interior,
            use_diamond_boundary=False,
            use_blockers=False,
            use_collisions=True,
        )
        if actual[0]["shift"] >= 3 and actual[0]["shift"] < rect[0]["shift"] and "boundary" in actual[0]["reasons"]:
            a_filtered.append(coords)
    if not a_filtered:
        return None

    for a_coords in a_filtered[:20]:
        forbidden_ab = set(forbidden)
        _reserve(forbidden_ab, a_coords)
        a_cols = {int(c) for _r, c in a_coords}

        b_shape = _random_shape(min_width=2)
        b_h = int(b_shape[:, 0].max()) + 1
        b_w = int(b_shape[:, 1].max()) + 1
        b_candidates = _candidate_placements(
            interior,
            forbidden_ab,
            b_shape,
            row_min=1,
            row_max=int(a_coords[:, 0].min()) - b_h - 2,
            col_min=max(1, min(a_cols) - b_w + 1),
            col_max=min(size - b_w - 1, max(a_cols)),
        )
        b_filtered: list[np.ndarray] = []
        for b_coords in b_candidates:
            if not a_cols & {int(c) for _r, c in b_coords}:
                continue
            pair = _sorted_movers(
                [
                    {"color": colors[0], "coords": a_coords},
                    {"color": colors[1], "coords": b_coords},
                ]
            )
            sim, _ = _simulate(
                grid.shape,
                pair,
                np.zeros_like(grid, dtype=bool),
                interior,
                use_diamond_boundary=True,
                use_blockers=False,
                use_collisions=True,
            )
            if len(sim) == 2 and "collision" in sim[1]["reasons"] and sim[1]["shift"] >= 2:
                b_filtered.append(b_coords)
        if not b_filtered:
            continue

        for b_coords in b_filtered[:20]:
            forbidden_abc = set(forbidden_ab)
            _reserve(forbidden_abc, b_coords)
            c_shape = _random_shape(min_width=2)
            c_h = int(c_shape[:, 0].max()) + 1
            c_w = int(c_shape[:, 1].max()) + 1
            c_sides = [("right" if side == "left" else "left"), "center"]
            c_candidates: list[np.ndarray] = []
            for c_mode in c_sides:
                if c_mode == "left":
                    c_col_min, c_col_max = 1, center[1] - c_w - 1
                elif c_mode == "right":
                    c_col_min, c_col_max = center[1] + 1, size - c_w - 2
                else:
                    c_col_min = max(1, center[1] - c_w)
                    c_col_max = min(size - c_w - 1, center[1])
                c_candidates.extend(
                    _candidate_placements(
                        interior,
                        forbidden_abc,
                        c_shape,
                        row_min=1,
                        row_max=max(1, upper_row_max - c_h + 1),
                        col_min=c_col_min,
                        col_max=c_col_max,
                    )
                )
            if not c_candidates:
                continue

            random.shuffle(c_candidates)
            for c_coords in c_candidates[:30]:
                col_to_rows: dict[int, list[int]] = {}
                for r, c in map(tuple, c_coords):
                    col_to_rows.setdefault(int(c), []).append(int(r))
                blocker_options: list[tuple[int, int]] = []
                for col, rows in col_to_rows.items():
                    floor = bottoms[col]
                    for br in range(max(rows) + 3, floor - 1):
                        cell = (int(br), int(col))
                        if cell not in forbidden_abc:
                            blocker_options.append(cell)
                random.shuffle(blocker_options)
                for blocker in blocker_options[:20]:
                    movers = _sorted_movers(
                        [
                            {"color": colors[0], "coords": a_coords},
                            {"color": colors[1], "coords": b_coords},
                            {"color": colors[2], "coords": c_coords},
                        ]
                    )
                    blocker_mask = np.zeros_like(grid, dtype=bool)
                    blocker_mask[blocker[0], blocker[1]] = True
                    actual, _ = _simulate(
                        grid.shape,
                        movers,
                        blocker_mask,
                        interior,
                        use_diamond_boundary=True,
                        use_blockers=True,
                        use_collisions=True,
                    )
                    rect, _ = _simulate(
                        grid.shape,
                        movers,
                        blocker_mask,
                        interior,
                        use_diamond_boundary=False,
                        use_blockers=True,
                        use_collisions=True,
                    )
                    noblock, _ = _simulate(
                        grid.shape,
                        movers,
                        blocker_mask,
                        interior,
                        use_diamond_boundary=True,
                        use_blockers=False,
                        use_collisions=True,
                    )
                    nocoll, _ = _simulate(
                        grid.shape,
                        movers,
                        blocker_mask,
                        interior,
                        use_diamond_boundary=True,
                        use_blockers=True,
                        use_collisions=False,
                    )
                    if not any("boundary" in info["reasons"] for info in actual):
                        continue
                    if not any("collision" in info["reasons"] for info in actual):
                        continue
                    if not any("blocker" in info["reasons"] for info in actual):
                        continue
                    if all(_same_shifts(actual, rect)):
                        continue
                    if all(_same_shifts(actual, noblock)):
                        continue
                    if all(_same_shifts(actual, nocoll)):
                        continue

                    blocker_cells = [blocker]
                    if random.random() < 0.5:
                        local_forbidden = set(forbidden_abc)
                        _reserve(local_forbidden, c_coords)
                        local_forbidden.update(_entity_neighbors(np.array([blocker], dtype=int)))
                        spare_options = [tuple(map(int, cell)) for cell in np.argwhere(interior) if tuple(map(int, cell)) not in local_forbidden]
                        random.shuffle(spare_options)
                        if spare_options:
                            blocker_cells.append(spare_options[0])

                    out = grid.copy()
                    for mover in movers:
                        _paint(out, mover["coords"], mover["color"])
                    for r, c in blocker_cells:
                        out[r, c] = Color.GREY
                    if _valid_case(out):
                        return out
    return None


def _same_shifts(a: list[dict], b: list[dict]) -> list[bool]:
    return [int(x["shift"]) == int(y["shift"]) for x, y in zip(a, b)]


def _valid_case(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
        center, radius, interior = _diamond_geometry(grid)
    except Exception:
        return False

    del center, radius
    bg = int(get_bg(grid))
    if bg != Color.BLACK:
        return False

    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    movers = []
    blocker_mask = np.zeros_like(grid, dtype=bool)
    for comp in comps:
        color = int(comp["color"])
        if color == Color.RED:
            continue
        if color == Color.GREY:
            if int(comp["area"]) != 1:
                return False
            blocker_mask[comp["coords"][:, 0], comp["coords"][:, 1]] = True
        else:
            if color not in PALETTE or not (3 <= int(comp["area"]) <= 6):
                return False
            movers.append(comp)

    if len(movers) != 3 or blocker_mask.sum() not in (1, 2):
        return False

    movers = _sorted_movers(movers)
    actual, _ = _simulate(
        grid.shape,
        movers,
        blocker_mask,
        interior,
        use_diamond_boundary=True,
        use_blockers=True,
        use_collisions=True,
    )
    rectangular, _ = _simulate(
        grid.shape,
        movers,
        blocker_mask,
        interior,
        use_diamond_boundary=False,
        use_blockers=True,
        use_collisions=True,
    )
    no_blockers, _ = _simulate(
        grid.shape,
        movers,
        blocker_mask,
        interior,
        use_diamond_boundary=True,
        use_blockers=False,
        use_collisions=True,
    )
    no_collisions, _ = _simulate(
        grid.shape,
        movers,
        blocker_mask,
        interior,
        use_diamond_boundary=True,
        use_blockers=True,
        use_collisions=False,
    )

    if all(_same_shifts(actual, rectangular)):
        return False
    if all(_same_shifts(actual, no_blockers)):
        return False
    if all(_same_shifts(actual, no_collisions)):
        return False
    if not any("boundary" in info["reasons"] for info in actual):
        return False
    if not any("blocker" in info["reasons"] for info in actual):
        return False
    if not any("collision" in info["reasons"] for info in actual):
        return False
    if not all(int(info["shift"]) > 0 for info in actual):
        return False

    expected = grid.copy()
    for mover in movers:
        expected[mover["coords"][:, 0], mover["coords"][:, 1]] = Color.BLACK
    for info in actual:
        expected[info["final"][:, 0], info["final"][:, 1]] = int(info["color"])
    if not np.array_equal(out, expected):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diamond gravity input")
