from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
Component = dict[str, object]


ORTHO_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
PAD_DIRS = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: set[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: set[Coord]) -> set[Coord]:
    r0, c0, _, _ = _bbox(coords)
    return {(r - r0, c - c0) for r, c in coords}


def _shape_dims(coords: set[Coord]) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    return r1 - r0 + 1, c1 - c0 + 1


def _shape_signature(coords: set[Coord]) -> tuple[Coord, ...]:
    return tuple(sorted(_normalize(coords)))


def _reflect_local(coords: set[Coord], axis_kind: str) -> set[Coord]:
    h, w = _shape_dims(coords)
    if axis_kind == "vertical":
        return {(r, w - 1 - c) for r, c in coords}
    if axis_kind == "horizontal":
        return {(h - 1 - r, c) for r, c in coords}
    raise ValueError(axis_kind)


def _is_asymmetric(coords: set[Coord]) -> bool:
    norm = _normalize(coords)
    return norm != _normalize(_reflect_local(norm, "vertical")) and norm != _normalize(
        _reflect_local(norm, "horizontal")
    )


def _translate(coords: set[Coord], dr: int, dc: int) -> set[Coord]:
    return {(r + dr, c + dc) for r, c in coords}


def _reflect_global(coords: set[Coord], axis_kind: str, axis2: int) -> set[Coord]:
    if axis_kind == "vertical":
        return {(r, axis2 - c) for r, c in coords}
    if axis_kind == "horizontal":
        return {(axis2 - r, c) for r, c in coords}
    raise ValueError(axis_kind)


def _find_components(grid: np.ndarray, bg: int) -> list[Component]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[Component] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: set[Coord] = set()
            while q:
                rr, cc = q.popleft()
                coords.add((rr, cc))
                for dr, dc in ORTHO_DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                    "sig": _shape_signature(coords),
                }
            )
    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))  # type: ignore[index]
    return comps


def _pair_reflections(comp_a: Component, comp_b: Component) -> list[tuple[str, int]]:
    if int(comp_a["color"]) != int(comp_b["color"]) or int(comp_a["area"]) != int(comp_b["area"]):
        return []

    coords_a = comp_a["coords"]  # type: ignore[assignment]
    coords_b = comp_b["coords"]  # type: ignore[assignment]
    bba = comp_a["bbox"]  # type: ignore[assignment]
    bbb = comp_b["bbox"]  # type: ignore[assignment]

    matches: list[tuple[str, int]] = []
    axis2 = int(bba[1] + bbb[3])
    if axis2 == int(bba[3] + bbb[1]) and _reflect_global(coords_a, "vertical", axis2) == coords_b:
        matches.append(("vertical", axis2))

    axis2 = int(bba[0] + bbb[2])
    if axis2 == int(bba[2] + bbb[0]) and _reflect_global(coords_a, "horizontal", axis2) == coords_b:
        matches.append(("horizontal", axis2))
    return matches


def _find_unique_mirror_pair(components: list[Component]) -> tuple[int, int, str, int]:
    hits: list[tuple[int, int, str, int]] = []
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            for axis_kind, axis2 in _pair_reflections(components[i], components[j]):
                hits.append((i, j, axis_kind, axis2))
    assert len(hits) == 1, "expected exactly one mirrored twin pair"
    return hits[0]


def _paint(grid: np.ndarray, coords: set[Coord], color: int) -> None:
    for r, c in coords:
        grid[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the unique mirrored twin pair, infer its hidden axis, and mirror every other sprite across that axis.

    Input structure:
    - A black background contains several separated monochrome 4-connected sprites.
    - Exactly one same-color pair are true mirror twins across one hidden vertical or horizontal axis.
    - At least one other same-color pair are only translated copies, plus optional additional asymmetric sprites.

    Rule:
    - Detect the unique pair whose two components are exact reflections of each other and derive the shared axis from them.
    - Keep that generating twin pair fixed in place.
    - Clear every other sprite from its original cells and redraw the whole sprite at its reflected position across the inferred axis.

    Key insight:
    - The axis is never drawn; only the genuine mirror twins encode it, while translated lookalikes are decoys because they do not match under reflection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    components = _find_components(g, bg)
    assert len(components) >= 4, "expected multiple sprites"

    twin_i, twin_j, axis_kind, axis2 = _find_unique_mirror_pair(components)
    twin_ids = {twin_i, twin_j}

    out = np.full_like(g, bg)
    occupied: set[Coord] = set()
    for idx, comp in enumerate(components):
        coords = comp["coords"] if idx in twin_ids else _reflect_global(comp["coords"], axis_kind, axis2)  # type: ignore[arg-type]
        for cell in coords:
            assert cell not in occupied, "reflected sprites must not overlap"
            occupied.add(cell)
        _paint(out, coords, int(comp["color"]))
    return out


def _random_shape() -> set[Coord]:
    for _ in range(300):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        target = random.randint(3, min(6, h * w - 1))
        cells = {(random.randrange(h), random.randrange(w))}
        frontier = list(cells)
        while len(cells) < target and frontier:
            base_r, base_c = random.choice(frontier)
            dr, dc = random.choice(ORTHO_DIRS)
            nr, nc = base_r + dr, base_c + dc
            if 0 <= nr < h and 0 <= nc < w:
                cells.add((nr, nc))
                frontier.append((nr, nc))
            if len(frontier) > 40:
                break
        norm = _normalize(cells)
        hh, ww = _shape_dims(norm)
        if len(norm) >= 3 and hh >= 2 and ww >= 2 and _is_asymmetric(norm):
            return norm
    raise RuntimeError("failed to sample an asymmetric sprite")


def _fits(cells: set[Coord], shape: tuple[int, int]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _side_ok(cells: set[Coord], axis_kind: str, axis2: int, side: str) -> bool:
    if side == "any":
        return True
    if axis_kind == "vertical":
        cols = [c for _, c in cells]
        if side == "left":
            return 2 * max(cols) < axis2
        return 2 * min(cols) > axis2
    rows = [r for r, _ in cells]
    if side == "top":
        return 2 * max(rows) < axis2
    return 2 * min(rows) > axis2


def _blocked_ok(blocked: np.ndarray, cells: set[Coord]) -> bool:
    h, w = blocked.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w) or blocked[r, c]:
            return False
    return True


def _mark_blocked(blocked: np.ndarray, cells: set[Coord]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for dr, dc in PAD_DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _all_positions(
    local: set[Coord],
    grid_shape: tuple[int, int],
    axis_kind: str,
    axis2: int,
    side: str,
) -> list[set[Coord]]:
    h, w = grid_shape
    sh, sw = _shape_dims(local)
    out: list[set[Coord]] = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            placed = _translate(local, r0, c0)
            if not _side_ok(placed, axis_kind, axis2, side):
                continue
            mirrored = _reflect_global(placed, axis_kind, axis2)
            if _fits(mirrored, grid_shape):
                out.append(placed)
    random.shuffle(out)
    return out


def _place_twin_pair(
    local: set[Coord],
    color: int,
    grid_shape: tuple[int, int],
    axis_kind: str,
    axis2: int,
    blocked_in: np.ndarray,
    blocked_out: np.ndarray,
) -> list[dict[str, object]] | None:
    if axis_kind == "vertical":
        primary_side = random.choice(["left", "right"])
    else:
        primary_side = random.choice(["top", "bottom"])

    for placed_a in _all_positions(local, grid_shape, axis_kind, axis2, primary_side):
        placed_b = _reflect_global(placed_a, axis_kind, axis2)
        if placed_a == placed_b or placed_a & placed_b:
            continue
        if not _blocked_ok(blocked_in, placed_a | placed_b):
            continue
        if not _blocked_ok(blocked_out, placed_a | placed_b):
            continue
        _mark_blocked(blocked_in, placed_a)
        _mark_blocked(blocked_in, placed_b)
        _mark_blocked(blocked_out, placed_a)
        _mark_blocked(blocked_out, placed_b)
        return [
            {"color": color, "coords": placed_a, "role": "twin"},
            {"color": color, "coords": placed_b, "role": "twin"},
        ]
    return None


def _place_single(
    local: set[Coord],
    color: int,
    role: str,
    grid_shape: tuple[int, int],
    axis_kind: str,
    axis2: int,
    blocked_in: np.ndarray,
    blocked_out: np.ndarray,
    side: str = "any",
) -> dict[str, object] | None:
    for placed in _all_positions(local, grid_shape, axis_kind, axis2, side):
        moved = placed if role == "twin" else _reflect_global(placed, axis_kind, axis2)
        if placed == moved:
            continue
        if not _blocked_ok(blocked_in, placed) or not _blocked_ok(blocked_out, moved):
            continue
        _mark_blocked(blocked_in, placed)
        _mark_blocked(blocked_out, moved)
        return {"color": color, "coords": placed, "role": role}
    return None


def _build_grid(shape: tuple[int, int], components: list[dict[str, object]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for comp in components:
        _paint(grid, comp["coords"], int(comp["color"]))  # type: ignore[arg-type]
    return grid


def _expected_output(
    shape: tuple[int, int],
    components: list[dict[str, object]],
    axis_kind: str,
    axis2: int,
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for comp in components:
        coords = comp["coords"] if comp["role"] == "twin" else _reflect_global(comp["coords"], axis_kind, axis2)  # type: ignore[arg-type]
        _paint(out, coords, int(comp["color"]))
    return out


def _translation_pairs(components: list[Component]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            if int(components[i]["color"]) != int(components[j]["color"]):
                continue
            if components[i]["sig"] != components[j]["sig"]:
                continue
            if _pair_reflections(components[i], components[j]):
                continue
            pairs.append((i, j))
    return pairs


def _validate_generated(
    grid: np.ndarray,
    expected: np.ndarray,
    axis_kind: str,
    axis2: int,
    twin_color: int,
    decoy_color: int,
) -> bool:
    if np.array_equal(grid, expected):
        return False

    bg = _get_bg(grid)
    comps = _find_components(grid, bg)
    if len(comps) < 4:
        return False
    if any(not _is_asymmetric(comp["coords"]) for comp in comps):  # type: ignore[arg-type]
        return False

    try:
        i, j, found_kind, found_axis2 = _find_unique_mirror_pair(comps)
    except AssertionError:
        return False
    if found_kind != axis_kind or found_axis2 != axis2:
        return False
    if int(comps[i]["color"]) != twin_color or int(comps[j]["color"]) != twin_color:
        return False

    decoy_pairs = _translation_pairs(comps)
    if len(decoy_pairs) != 1:
        return False
    di, dj = decoy_pairs[0]
    if int(comps[di]["color"]) != decoy_color or int(comps[dj]["color"]) != decoy_color:
        return False

    try:
        solved = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(solved, expected):
        return False

    moved = 0
    for comp in comps:
        reflected = _reflect_global(comp["coords"], axis_kind, axis2)  # type: ignore[arg-type]
        if reflected != comp["coords"]:
            moved += 1
    return moved >= len(comps) - 2


def _attempt_generate() -> np.ndarray | None:
    axis_kind = random.choice(["vertical", "horizontal"])
    if axis_kind == "vertical":
        height = random.randint(14, 24)
        width = random.randint(18, 30)
        center2 = width - 1
    else:
        height = random.randint(18, 30)
        width = random.randint(14, 24)
        center2 = height - 1

    axis2_choices = [v for v in range(4, 2 * (width - 1) + 1)] if axis_kind == "vertical" else [
        v for v in range(4, 2 * (height - 1) + 1)
    ]
    axis2_choices = [v for v in axis2_choices if abs(v - center2) >= 2]
    if not axis2_choices:
        return None
    axis2 = random.choice(axis2_choices)

    grid_shape = (height, width)
    blocked_in = np.zeros(grid_shape, dtype=bool)
    blocked_out = np.zeros(grid_shape, dtype=bool)

    colors = random.sample(PALETTE, k=random.randint(3, 5))
    twin_color = colors[0]
    decoy_color = colors[1]
    extra_colors = colors[2:]

    components: list[dict[str, object]] = []

    twin_shape = _random_shape()
    twin_pair = _place_twin_pair(
        twin_shape, twin_color, grid_shape, axis_kind, axis2, blocked_in, blocked_out
    )
    if twin_pair is None:
        return None
    components.extend(twin_pair)

    decoy_shape = _random_shape()
    decoy_side = (
        random.choice(["left", "right"]) if axis_kind == "vertical" else random.choice(["top", "bottom"])
    )
    for _ in range(2):
        placed = _place_single(
            decoy_shape,
            decoy_color,
            "decoy",
            grid_shape,
            axis_kind,
            axis2,
            blocked_in,
            blocked_out,
            side=decoy_side,
        )
        if placed is None:
            return None
        components.append(placed)

    extra_count = random.randint(1, len(extra_colors))
    used_sigs = {tuple(sorted(twin_shape)), tuple(sorted(decoy_shape))}
    for color in extra_colors[:extra_count]:
        for _ in range(60):
            shape = _random_shape()
            sig = tuple(sorted(shape))
            if sig in used_sigs:
                continue
            used_sigs.add(sig)
            placed = _place_single(
                shape,
                color,
                "extra",
                grid_shape,
                axis_kind,
                axis2,
                blocked_in,
                blocked_out,
                side=random.choice(["any", "any", "left" if axis_kind == "vertical" else "top"]),
            )
            if placed is not None:
                components.append(placed)
                break
        else:
            return None

    if len(components) < 5:
        return None

    grid = _build_grid(grid_shape, components)
    expected = _expected_output(grid_shape, components, axis_kind, axis2)
    if not _validate_generated(grid, expected, axis_kind, axis2, twin_color, decoy_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-twins puzzle")
