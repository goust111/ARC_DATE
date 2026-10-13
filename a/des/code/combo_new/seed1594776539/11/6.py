from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 0
FOREGROUND_COLORS = tuple(range(1, 10))
ORTH_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

PAIR_COUNT_RANGE = (3, 5)
GRID_LIMIT = 30
GAP_RANGE = (2, 7)
GENERATION_RETRIES = 500
PLACEMENT_RETRIES = 250


def _find_components(grid: np.ndarray) -> list[dict[str, object]]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ORTH_DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            components.append(
                {
                    "color": color,
                    "coords": cells,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )

    return components


def _mask_from_component(component: dict[str, object]) -> np.ndarray:
    coords = component["coords"]
    r0, c0, r1, c1 = component["bbox"]
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    mask = np.zeros((height, width), dtype=bool)
    for r, c in coords:
        mask[int(r - r0), int(c - c0)] = True
    return mask


def _classify_component(component: dict[str, object]) -> str:
    mask = _mask_from_component(component)
    if mask.all():
        return "solid"

    if mask.shape[0] >= 3 and mask.shape[1] >= 3:
        border = np.zeros_like(mask, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if np.all(mask[border]) and not np.any(mask[1:-1, 1:-1]):
            return "hollow"

    raise AssertionError("component must be either solid-filled or hollow-ring shaped")


def _pair_geometry(a: dict[str, object], b: dict[str, object]) -> dict[str, object]:
    ar0, ac0, ar1, ac1 = map(int, a["bbox"])
    br0, bc0, br1, bc1 = map(int, b["bbox"])

    row_start = max(ar0, br0)
    row_stop = min(ar1, br1)
    col_start = max(ac0, bc0)
    col_stop = min(ac1, bc1)

    horizontal = row_start <= row_stop and (ac1 < bc0 or bc1 < ac0)
    vertical = col_start <= col_stop and (ar1 < br0 or br1 < ar0)
    assert horizontal ^ vertical, "paired motifs must align horizontally or vertically"

    if horizontal:
        left, right = (a, b) if ac0 <= bc0 else (b, a)
        lr0, lc0, lr1, lc1 = map(int, left["bbox"])
        rr0, rc0, rr1, rc1 = map(int, right["bbox"])
        row_start = max(lr0, rr0)
        row_stop = min(lr1, rr1)
        start = lc1 + 1
        stop = rc0 - 1
        assert start <= stop, "horizontal pair must have a positive gap"
        return {
            "orientation": "H",
            "first": left,
            "second": right,
            "row_start": row_start,
            "row_stop": row_stop,
            "start": start,
            "stop": stop,
        }

    top, bottom = (a, b) if ar0 <= br0 else (b, a)
    tr0, tc0, tr1, tc1 = map(int, top["bbox"])
    br0, bc0, br1, bc1 = map(int, bottom["bbox"])
    col_start = max(tc0, bc0)
    col_stop = min(tc1, bc1)
    start = tr1 + 1
    stop = br0 - 1
    assert start <= stop, "vertical pair must have a positive gap"
    return {
        "orientation": "V",
        "first": top,
        "second": bottom,
        "col_start": col_start,
        "col_stop": col_stop,
        "start": start,
        "stop": stop,
    }


def _fill_corridor(
    out: np.ndarray,
    geometry: dict[str, object],
    color: int,
    fill_from: str | None,
    dashed: bool,
) -> None:
    start = int(geometry["start"])
    stop = int(geometry["stop"])
    gap = stop - start + 1
    orientation = geometry["orientation"]

    if dashed:
        if orientation == "H":
            rs = int(geometry["row_start"])
            re = int(geometry["row_stop"])
            for c in range(start, stop + 1):
                if (c - start) % 2 == 0:
                    out[rs : re + 1, c] = color
            return

        cs = int(geometry["col_start"])
        ce = int(geometry["col_stop"])
        for r in range(start, stop + 1):
            if (r - start) % 2 == 0:
                out[r, cs : ce + 1] = color
        return

    if fill_from is None:
        fill_start = start
        fill_stop = stop
    else:
        count = (gap + 1) // 2
        if fill_from == "first":
            fill_start = start
            fill_stop = start + count - 1
        else:
            fill_start = stop - count + 1
            fill_stop = stop

    if orientation == "H":
        rs = int(geometry["row_start"])
        re = int(geometry["row_stop"])
        out[rs : re + 1, fill_start : fill_stop + 1] = color
    else:
        cs = int(geometry["col_start"])
        ce = int(geometry["col_stop"])
        out[fill_start : fill_stop + 1, cs : ce + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-colored motif pair with a line pattern determined by whether the motifs are solid or hollow.

    Input structure:
    - The background is black, and each non-black color appears in exactly two disconnected small motifs.
    - Every motif is either a solid filled rectangle or a hollow ring whose bounding-box border is filled and interior stays black.
    - Each same-colored pair is aligned horizontally or vertically with an unobstructed black corridor between the two motifs.

    Rule:
    - Keep every motif unchanged and process each color pair independently.
    - If both motifs are solid, fill the whole corridor between them with that color.
    - If both motifs are hollow, fill every other corridor slice starting from the left side for horizontal pairs or the top side for vertical pairs.
    - If one motif is solid and the other is hollow, fill only the half of the corridor adjacent to the solid motif, stopping at the midpoint.

    Key insight:
    - Color identifies which motifs pair together, but the connector style comes from the pair's internal shape combination: solid-solid, hollow-hollow, or mixed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components = _find_components(g)
    by_color: dict[int, list[dict[str, object]]] = defaultdict(list)
    for component in components:
        by_color[int(component["color"])].append(component)

    out = g.copy()
    for color, pair in by_color.items():
        assert color != BG
        assert len(pair) == 2, "each non-black color must appear in exactly two motifs"

        first_kind = _classify_component(pair[0])
        second_kind = _classify_component(pair[1])
        geometry = _pair_geometry(pair[0], pair[1])

        if first_kind == "solid" and second_kind == "solid":
            _fill_corridor(out, geometry, color, fill_from=None, dashed=False)
        elif first_kind == "hollow" and second_kind == "hollow":
            _fill_corridor(out, geometry, color, fill_from=None, dashed=True)
        elif first_kind == "solid":
            fill_from = "first" if geometry["first"] is pair[0] else "second"
            _fill_corridor(out, geometry, color, fill_from=fill_from, dashed=False)
        else:
            fill_from = "first" if geometry["first"] is pair[1] else "second"
            _fill_corridor(out, geometry, color, fill_from=fill_from, dashed=False)

    return out


def _build_sprite(kind: str, dims: tuple[int, int], color: int) -> np.ndarray:
    height, width = dims
    if kind == "solid":
        return np.full((height, width), color, dtype=int)

    assert kind == "hollow"
    assert dims == (3, 3)
    sprite = np.full((3, 3), color, dtype=int)
    sprite[1, 1] = BG
    return sprite


def _choose_pair_kinds() -> list[tuple[str, str]]:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    kinds: list[tuple[str, str]] = [
        ("solid", "solid"),
        ("hollow", "hollow"),
        random.choice((("solid", "hollow"), ("hollow", "solid"))),
    ]
    while len(kinds) < pair_count:
        branch = random.choice(("solid", "hollow", "mixed"))
        if branch == "solid":
            kinds.append(("solid", "solid"))
        elif branch == "hollow":
            kinds.append(("hollow", "hollow"))
        else:
            kinds.append(random.choice((("solid", "hollow"), ("hollow", "solid"))))
    random.shuffle(kinds)
    return kinds


def _sample_dims(orientation: str, kinds: tuple[str, str]) -> tuple[tuple[int, int], tuple[int, int]]:
    a_kind, b_kind = kinds
    if a_kind == "hollow" and b_kind == "hollow":
        return (3, 3), (3, 3)

    if orientation == "H":
        common_height = 3 if "hollow" in kinds else random.choice((2, 3))
        a_width = 3 if a_kind == "hollow" else random.choice((2, 3))
        b_width = 3 if b_kind == "hollow" else random.choice((2, 3))
        return (common_height, a_width), (common_height, b_width)

    common_width = 3 if "hollow" in kinds else random.choice((2, 3))
    a_height = 3 if a_kind == "hollow" else random.choice((2, 3))
    b_height = 3 if b_kind == "hollow" else random.choice((2, 3))
    return (a_height, common_width), (b_height, common_width)


def _clamped_box(
    top: int,
    left: int,
    bottom: int,
    right: int,
    height: int,
    width: int,
    pad: int = 1,
) -> tuple[int, int, int, int]:
    return (
        max(0, top - pad),
        max(0, left - pad),
        min(height - 1, bottom + pad),
        min(width - 1, right + pad),
    )


def _paint_sprite(grid: np.ndarray, sprite: np.ndarray, top: int, left: int) -> None:
    height, width = sprite.shape
    patch = grid[top : top + height, left : left + width]
    mask = sprite != BG
    patch[mask] = sprite[mask]


def _place_pair(
    grid: np.ndarray,
    blocked: np.ndarray,
    color: int,
    orientation: str,
    kinds: tuple[str, str],
    dims_a: tuple[int, int],
    dims_b: tuple[int, int],
    gap: int,
) -> bool:
    height, width = grid.shape
    sprite_a = _build_sprite(kinds[0], dims_a, color)
    sprite_b = _build_sprite(kinds[1], dims_b, color)

    if orientation == "H":
        pair_height = dims_a[0]
        pair_width = dims_a[1] + gap + dims_b[1]
        if pair_height > height or pair_width > width:
            return False

        for _ in range(PLACEMENT_RETRIES):
            top = random.randint(0, height - pair_height)
            left = random.randint(0, width - pair_width)
            bottom = top + pair_height - 1
            right = left + pair_width - 1
            br0, bc0, br1, bc1 = _clamped_box(top, left, bottom, right, height, width)
            if np.any(blocked[br0 : br1 + 1, bc0 : bc1 + 1]):
                continue

            _paint_sprite(grid, sprite_a, top, left)
            _paint_sprite(grid, sprite_b, top, left + dims_a[1] + gap)
            blocked[br0 : br1 + 1, bc0 : bc1 + 1] = True
            return True

        return False

    pair_height = dims_a[0] + gap + dims_b[0]
    pair_width = dims_a[1]
    if pair_height > height or pair_width > width:
        return False

    for _ in range(PLACEMENT_RETRIES):
        top = random.randint(0, height - pair_height)
        left = random.randint(0, width - pair_width)
        bottom = top + pair_height - 1
        right = left + pair_width - 1
        br0, bc0, br1, bc1 = _clamped_box(top, left, bottom, right, height, width)
        if np.any(blocked[br0 : br1 + 1, bc0 : bc1 + 1]):
            continue

        _paint_sprite(grid, sprite_a, top, left)
        _paint_sprite(grid, sprite_b, top + dims_a[0] + gap, left)
        blocked[br0 : br1 + 1, bc0 : bc1 + 1] = True
        return True

    return False


def _generate_attempt() -> np.ndarray | None:
    pair_kinds = _choose_pair_kinds()
    pair_count = len(pair_kinds)

    min_side = max(16, 10 + 3 * pair_count)
    height = random.randint(min_side, GRID_LIMIT)
    width = random.randint(min_side, GRID_LIMIT)
    colors = random.sample(FOREGROUND_COLORS, pair_count)

    specs: list[dict[str, object]] = []
    for color, kinds in zip(colors, pair_kinds, strict=False):
        orientation = random.choice(("H", "V"))
        dims_a, dims_b = _sample_dims(orientation, kinds)
        gap = random.randint(*GAP_RANGE)
        if orientation == "H":
            extent = dims_a[0] * (dims_a[1] + gap + dims_b[1])
        else:
            extent = (dims_a[0] + gap + dims_b[0]) * dims_a[1]
        specs.append(
            {
                "color": color,
                "orientation": orientation,
                "kinds": kinds,
                "dims_a": dims_a,
                "dims_b": dims_b,
                "gap": gap,
                "extent": extent,
            }
        )

    random.shuffle(specs)
    specs.sort(key=lambda spec: int(spec["extent"]), reverse=True)

    grid = np.zeros((height, width), dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    for spec in specs:
        placed = _place_pair(
            grid,
            blocked,
            int(spec["color"]),
            str(spec["orientation"]),
            tuple(spec["kinds"]),
            tuple(spec["dims_a"]),
            tuple(spec["dims_b"]),
            int(spec["gap"]),
        )
        if not placed:
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _generate_attempt()
        if grid is not None and max(grid.shape) <= GRID_LIMIT:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
