from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
STAMP_COLORS = [1, 2, 3, 4, 6, 7, 9]
SIDES = ["top", "right", "bottom", "left"]
ROT_K_BY_SIDE = {"top": 0, "right": 3, "bottom": 2, "left": 1}
GEN_ATTEMPTS = 400
PLACEMENT_PADDING = 1


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(points[:, 0].min())
    c0 = int(points[:, 1].min())
    r1 = int(points[:, 0].max())
    c1 = int(points[:, 1].max())
    return r0, c0, r1, c1


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _is_rotationally_asymmetric(mask: np.ndarray) -> bool:
    return all(not bool(np.array_equal(mask, np.rot90(mask, k))) for k in (1, 2, 3))


def _canvas_from_component(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict | None:
    pts = np.array(comp, dtype=int)
    r0, c0, r1, c1 = _bbox(pts)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h != w or h % 2 == 0 or h < 3 or len(comp) != h * w:
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if bool(np.any(patch == BLACK)):
        return None

    mid = h // 2
    marker_pos = {
        "top": (0, mid),
        "right": (mid, h - 1),
        "bottom": (h - 1, mid),
        "left": (mid, 0),
    }
    markers = {side: int(patch[r, c]) for side, (r, c) in marker_pos.items()}
    if any(color in (BLACK, GREY) for color in markers.values()):
        return None
    if list(markers.values()).count(AZURE) != 3:
        return None

    matched = [(side, color) for side, color in markers.items() if color != AZURE]
    if len(matched) != 1:
        return None
    match_side, match_color = matched[0]

    normal = np.ones((h, h), dtype=bool)
    for r, c in marker_pos.values():
        normal[r, c] = False
    if not bool(np.all(patch[normal] == GREY)):
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "size": h,
        "inner_size": h - 2,
        "match_color": int(match_color),
        "match_side": match_side,
    }


def _detect_canvases(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    canvases: list[dict] = []
    canvas_mask = np.zeros_like(grid, dtype=bool)
    for comp in _components4(grid != BLACK):
        canvas = _canvas_from_component(grid, comp)
        if canvas is None:
            continue
        r0, c0, r1, c1 = canvas["bbox"]
        canvas_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        canvases.append(canvas)

    if not canvases:
        raise AssertionError("expected at least one canvas")

    colors = [canvas["match_color"] for canvas in canvases]
    if len(colors) != len(set(colors)):
        raise AssertionError("canvas match colors must be unique")

    return canvases, canvas_mask


def _detect_stamps(grid: np.ndarray, canvas_mask: np.ndarray, canvases: list[dict]) -> dict[int, dict]:
    stamps: dict[int, dict] = {}
    for canvas in canvases:
        color = int(canvas["match_color"])
        pts = np.argwhere((grid == color) & (~canvas_mask))
        if pts.size == 0:
            raise AssertionError("missing stamp pixels for canvas color")
        r0, c0, r1, c1 = _bbox(pts)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        expected = int(canvas["inner_size"])
        if h != w or h != expected:
            raise AssertionError("stamp bbox must match canvas interior size")
        patch = np.where(grid[r0 : r1 + 1, c0 : c1 + 1] == color, color, BLACK)
        if not _touches_all_sides(patch == color):
            raise AssertionError("stamp must expose its full square extent")
        stamps[color] = {
            "bbox": (r0, c0, r1, c1),
            "patch": patch.astype(int),
        }
    return stamps


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    canvases, canvas_mask = _detect_canvases(g)
    stamps = _detect_stamps(g, canvas_mask, canvases)

    out = g.copy()
    for stamp in stamps.values():
        r0, c0, r1, c1 = stamp["bbox"]
        patch = stamp["patch"]
        stamp_view = out[r0 : r1 + 1, c0 : c1 + 1]
        stamp_view[patch != BLACK] = BLACK

    for canvas in canvases:
        color = int(canvas["match_color"])
        stamp_patch = stamps[color]["patch"]
        rotated = np.rot90(stamp_patch, ROT_K_BY_SIDE[canvas["match_side"]])
        r0, c0, r1, c1 = canvas["bbox"]
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        mask = rotated != BLACK
        inner[mask] = rotated[mask]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each colored stamp to the side marked on its matching grey canvas and print it into the canvas interior.

    Input structure:
    - The background is Black.
    - The grid contains several separated stamp-canvas pairs.
    - Each canvas is a solid odd-sized Grey square whose four edge midpoints are colored:
      three are Azure and one is a unique non-Grey non-Black match color.
    - Each stamp uses that same match color inside a square Black field whose side length equals
      the paired canvas interior size.

    Rule:
    - Detect every canvas and read which side carries the unique non-Azure midpoint color.
    - Pair that canvas with the stamp that uses the same color.
    - Rotate the stamp by 0/90/180/270 degrees for top/right/bottom/left respectively.
    - Overlay the rotated non-Black stamp pixels onto the inside of the canvas, and clear the
      original stamp pixels back to Black.

    Key insight:
    - The unique non-Azure edge marker serves two roles at once: it identifies which stamp belongs
      to the canvas and it encodes the stamp rotation by its side position.
    """
    return _solve(np.asarray(grid, dtype=int))


def _make_canvas(size: int, color: int, side: str) -> np.ndarray:
    patch = np.full((size, size), GREY, dtype=int)
    mid = size // 2
    patch[0, mid] = AZURE
    patch[mid, size - 1] = AZURE
    patch[size - 1, mid] = AZURE
    patch[mid, 0] = AZURE
    if side == "top":
        patch[0, mid] = color
    elif side == "right":
        patch[mid, size - 1] = color
    elif side == "bottom":
        patch[size - 1, mid] = color
    else:
        patch[mid, 0] = color
    return patch


def _path_to(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int], rng: random.Random) -> None:
    r, c = start
    er, ec = end
    mask[r, c] = True
    while (r, c) != (er, ec):
        steps: list[tuple[int, int]] = []
        if r < er:
            steps.append((r + 1, c))
        elif r > er:
            steps.append((r - 1, c))
        if c < ec:
            steps.append((r, c + 1))
        elif c > ec:
            steps.append((r, c - 1))
        r, c = rng.choice(steps)
        mask[r, c] = True


def _stamp_templates_3() -> list[np.ndarray]:
    return [
        np.array([[1, 1, 1], [1, 0, 1], [0, 0, 1]], dtype=bool),
        np.array([[1, 1, 0], [1, 0, 1], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 0, 1], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=bool),
    ]


def _random_stamp_mask(size: int, rng: random.Random) -> np.ndarray:
    if size == 3:
        mask = rng.choice(_stamp_templates_3()).copy()
        return np.rot90(mask, rng.randrange(4))

    for _ in range(200):
        mask = np.zeros((size, size), dtype=bool)
        center = (size // 2, size // 2)
        anchors = [
            (0, rng.randrange(size)),
            (rng.randrange(size), size - 1),
            (size - 1, rng.randrange(size)),
            (rng.randrange(size), 0),
        ]
        for anchor in anchors:
            _path_to(mask, center, anchor, rng)

        target_min = max(size + 3, (size * size) // 3)
        target_max = min(size * size - 1, (2 * size * size) // 3 + 1)
        target = rng.randint(target_min, target_max)

        while int(mask.sum()) < target:
            frontier: set[tuple[int, int]] = set()
            for r, c in np.argwhere(mask):
                for nr, nc in _neighbors4(int(r), int(c)):
                    if 0 <= nr < size and 0 <= nc < size and not bool(mask[nr, nc]):
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = rng.choice(list(frontier))
            mask[nr, nc] = True

        if not _touches_all_sides(mask):
            continue
        if not _is_rotationally_asymmetric(mask):
            continue
        if int(mask.sum()) >= size * size:
            continue
        return np.rot90(mask, rng.randrange(4))

    raise AssertionError("failed to generate a valid stamp mask")


def _stamp_patch(size: int, color: int, rng: random.Random) -> np.ndarray:
    mask = _random_stamp_mask(size, rng)
    patch = np.zeros((size, size), dtype=int)
    patch[mask] = color
    return patch


def _can_place(occupied: np.ndarray, r0: int, c0: int, size: int, pad: int = PLACEMENT_PADDING) -> bool:
    h, w = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + size > h or c0 + size > w:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r0 + size + pad)
    cc1 = min(w, c0 + size + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, size: int) -> None:
    occupied[r0 : r0 + size, c0 : c0 + size] = True


def _place_square(occupied: np.ndarray, size: int, rng: random.Random) -> tuple[int, int] | None:
    h, w = occupied.shape
    positions = [(r, c) for r in range(h - size + 1) for c in range(w - size + 1)]
    rng.shuffle(positions)
    for r0, c0 in positions:
        if _can_place(occupied, r0, c0, size):
            _mark_occupied(occupied, r0, c0, size)
            return r0, c0
    return None


def _expected_from_specs(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        sr, sc = spec["stamp_pos"]
        stamp = spec["stamp_patch"]
        out[sr : sr + stamp.shape[0], sc : sc + stamp.shape[1]][stamp != BLACK] = BLACK
    for spec in specs:
        cr, cc = spec["canvas_pos"]
        canvas = spec["canvas_patch"]
        stamp = np.rot90(spec["stamp_patch"], ROT_K_BY_SIDE[spec["side"]])
        inner = out[cr + 1 : cr + canvas.shape[0] - 1, cc + 1 : cc + canvas.shape[1] - 1]
        inner[stamp != BLACK] = stamp[stamp != BLACK]
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(GEN_ATTEMPTS):
        pair_count = rng.randint(1, 4)
        min_side = 12 if pair_count == 1 else 14 + 3 * (pair_count - 2)
        h = rng.randint(min_side, 30)
        w = rng.randint(min_side, 30)
        grid = np.full((h, w), BLACK, dtype=int)
        occupied = np.zeros((h, w), dtype=bool)

        colors = rng.sample(STAMP_COLORS, pair_count)
        size_choices = [5, 7] + ([9] if pair_count <= 2 else [])
        specs: list[dict] = []
        ok = True

        for color in colors:
            canvas_size = rng.choice(size_choices)
            side = rng.choice(SIDES)
            canvas_patch = _make_canvas(canvas_size, color, side)
            stamp_patch = _stamp_patch(canvas_size - 2, color, rng)
            specs.append(
                {
                    "color": color,
                    "side": side,
                    "canvas_patch": canvas_patch,
                    "stamp_patch": stamp_patch,
                }
            )

        order = sorted(
            [(idx, spec["canvas_patch"].shape[0]) for idx, spec in enumerate(specs) for _ in (0, 1)],
            key=lambda item: item[1],
            reverse=True,
        )

        assigned: dict[tuple[int, str], tuple[int, int]] = {}
        for idx, _size in order:
            spec = specs[idx]
            if (idx, "canvas") in assigned:
                kind = "stamp"
                patch = spec["stamp_patch"]
            else:
                kind = "canvas"
                patch = spec["canvas_patch"]
            pos = _place_square(occupied, patch.shape[0], rng)
            if pos is None:
                ok = False
                break
            assigned[(idx, kind)] = pos

        if not ok:
            continue

        for idx, spec in enumerate(specs):
            cr, cc = assigned[(idx, "canvas")]
            sr, sc = assigned[(idx, "stamp")]
            spec["canvas_pos"] = (cr, cc)
            spec["stamp_pos"] = (sr, sc)
            canvas = spec["canvas_patch"]
            stamp = spec["stamp_patch"]
            grid[cr : cr + canvas.shape[0], cc : cc + canvas.shape[1]] = canvas
            stamp_view = grid[sr : sr + stamp.shape[0], sc : sc + stamp.shape[1]]
            mask = stamp != BLACK
            stamp_view[mask] = stamp[mask]

        expected = _expected_from_specs(grid, specs)
        solved = _solve(grid)
        if not bool(np.array_equal(solved, expected)):
            continue
        if len(_detect_canvases(grid)[0]) != pair_count:
            continue

        return grid

    raise AssertionError("generate_input failed to build a valid puzzle")
