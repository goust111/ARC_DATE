from __future__ import annotations

from collections import Counter, defaultdict
import random
import numpy as np


_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_NEIGHBORS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in _NEIGHBORS4]


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def find_components(grid: np.ndarray, bg: int, connectivity: int = 8) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    steps = _NEIGHBORS4 if connectivity == 4 else _NEIGHBORS8
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in steps:
                    nr, nc = int(cr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) != int(bg):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return comps


def _unique_marker(grid: np.ndarray, coords: np.ndarray) -> tuple[int, tuple[int, int]] | None:
    colors = [int(grid[int(r), int(c)]) for r, c in coords]
    counts = Counter(colors)
    if len(counts) < 2:
        return None
    least = min(counts.values())
    marker_colors = [color for color, count in counts.items() if int(count) == int(least)]
    if len(marker_colors) != 1:
        return None
    marker_color = int(marker_colors[0])
    marker_pixels = [(int(r), int(c)) for r, c in coords if int(grid[int(r), int(c)]) == int(marker_color)]
    if len(marker_pixels) != 1:
        return None
    return marker_color, marker_pixels[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each sprite to the isolated border anchor of the same marker color and
    slide the sprite along the axis implied by the anchor edge.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    anchors_by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            color = int(g[r, c])
            if color == int(bg):
                continue
            if all(
                not (0 <= nr < h and 0 <= nc < w) or int(g[nr, nc]) == int(bg)
                for nr, nc in neighbors4(int(r), int(c))
            ):
                anchors_by_color[color].append((int(r), int(c)))

    anchors = {
        int(color): pts[0]
        for color, pts in anchors_by_color.items()
        if len(pts) == 1
    }
    anchor_cells = {pos for pos in anchors.values()}

    moves: list[tuple[np.ndarray, int, int]] = []
    for comp in find_components(g, bg=bg, connectivity=8):
        coords = comp["coords"]
        if int(coords.shape[0]) == 1 and tuple(map(int, coords[0])) in anchor_cells:
            continue

        marker = _unique_marker(g, coords)
        if marker is None:
            continue
        marker_color, (mr, mc) = marker
        if int(marker_color) not in anchors:
            continue

        ar, ac = anchors[int(marker_color)]
        horizontal_slide = ar in (0, h - 1) and ac not in (0, w - 1)
        dr = 0
        dc = 0
        if horizontal_slide:
            dc = int(ac - mc)
        else:
            dr = int(ar - mr)
        moves.append((coords, int(dr), int(dc)))

    out = g.copy()
    for coords, _dr, _dc in moves:
        out[coords[:, 0], coords[:, 1]] = int(bg)

    for coords, dr, dc in moves:
        for r, c in coords:
            rr = int(r) + int(dr)
            cc = int(c) + int(dc)
            if 0 <= rr < h and 0 <= cc < w and int(out[rr, cc]) == int(bg):
                out[rr, cc] = int(g[int(r), int(c)])

    return out


_COLORS = tuple(range(10))
_GRID_H_RANGE = (12, 20)
_GRID_W_RANGE = (12, 20)
_SPRITE_COUNT_RANGE = (2, 4)
_BODY_AREA_RANGE = (4, 9)
_BOX_H_RANGE = (2, 4)
_BOX_W_RANGE = (3, 5)
_ANCHOR_MARGIN = 4
_GEN_RETRIES = 400


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return {(int(r - r0), int(c - c0)) for r, c in cells}


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    return (
        min(r for r, _c in cells),
        min(c for _r, c in cells),
        max(r for r, _c in cells),
        max(c for _r, c in cells),
    )


def _cells_connected(cells: set[tuple[int, int]]) -> bool:
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in _NEIGHBORS4:
            nxt = (int(r + dr), int(c + dc))
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _random_transform(cells: set[tuple[int, int]], rng: random.Random) -> set[tuple[int, int]]:
    cur = set(cells)
    if rng.random() < 0.5:
        mr = max(r for r, _c in cur)
        cur = {(int(mr - r), int(c)) for r, c in cur}
    if rng.random() < 0.5:
        mc = max(c for _r, c in cur)
        cur = {(int(r), int(mc - c)) for r, c in cur}
    if rng.random() < 0.35:
        cur = {(int(c), int(r)) for r, c in cur}
    return _normalize(cur)


def _frontier(cells: set[tuple[int, int]], height: int, width: int) -> list[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in _NEIGHBORS4:
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                out.add((int(rr), int(cc)))
    return list(out)


def _blob_body(rng: random.Random, area: int) -> set[tuple[int, int]]:
    dims = [
        (h, w)
        for h in range(_BOX_H_RANGE[0], _BOX_H_RANGE[1] + 1)
        for w in range(_BOX_W_RANGE[0], _BOX_W_RANGE[1] + 1)
        if h * w >= int(area)
    ]
    h, w = rng.choice(dims)
    seeds = [
        {(0, 0), (0, 1), (1, 0)},
        {(0, 0), (0, 1), (1, 1)},
        {(0, 1), (1, 0), (1, 1)},
    ]
    cells = set(rng.choice(seeds))
    while len(cells) < int(area):
        fringe = _frontier(cells, h, w)
        if not fringe:
            return _blob_body(rng, area)
        cells.add(rng.choice(fringe))
    return _random_transform(cells, rng)


def _bar_body(rng: random.Random, area: int) -> set[tuple[int, int]]:
    width = rng.randint(3, min(_BOX_W_RANGE[1], area))
    dims = [
        h
        for h in range(_BOX_H_RANGE[0], _BOX_H_RANGE[1] + 1)
        if h * width >= int(area)
    ]
    if not dims:
        return _blob_body(rng, area)
    height = rng.choice(dims)
    cells = {(0, c) for c in range(width)}
    cells.add((1, rng.randrange(width)))
    while len(cells) < int(area):
        cells.add(rng.choice(_frontier(cells, height, width)))
    return _random_transform(cells, rng)


def _frame_body(rng: random.Random, area: int) -> set[tuple[int, int]]:
    choices = []
    for h in range(3, _BOX_H_RANGE[1] + 2):
        for w in range(3, _BOX_W_RANGE[1] + 1):
            perimeter = 2 * h + 2 * w - 4
            excess = perimeter - int(area)
            if 0 <= excess <= 2:
                choices.append((h, w))
    if not choices:
        return _blob_body(rng, area)

    h, w = rng.choice(choices)
    cells = {
        (r, c)
        for r in range(h)
        for c in range(w)
        if r in (0, h - 1) or c in (0, w - 1)
    }
    excess = len(cells) - int(area)
    removable = [
        (r, c)
        for r, c in cells
        if not (r in (0, h - 1) and c in (0, w - 1))
    ]
    for _ in range(20):
        trial = set(cells)
        if excess:
            for cell in rng.sample(removable, excess):
                trial.remove(cell)
        if _cells_connected(trial):
            return _random_transform(trial, rng)
    return _blob_body(rng, area)


def _make_body(rng: random.Random, area: int) -> set[tuple[int, int]]:
    pick = rng.random()
    if pick < 0.4:
        return _blob_body(rng, area)
    if pick < 0.75:
        return _bar_body(rng, area)
    return _frame_body(rng, area)


def _marker_cell(body: set[tuple[int, int]], rng: random.Random) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(body)
    interior: set[tuple[int, int]] = set()
    exterior: set[tuple[int, int]] = set()
    for r, c in body:
        for dr, dc in _NEIGHBORS4:
            rr, cc = int(r + dr), int(c + dc)
            if (rr, cc) in body:
                continue
            if r0 <= rr <= r1 and c0 <= cc <= c1:
                interior.add((int(rr), int(cc)))
            else:
                exterior.add((int(rr), int(cc)))
    pool = list(interior) if interior and rng.random() < 0.65 else list(interior | exterior)
    return rng.choice(pool)


def _make_sprite(
    rng: random.Random,
    area: int,
    body_color: int,
    marker_color: int,
    bg: int,
) -> dict[str, object]:
    body = _make_body(rng, area)
    marker = _marker_cell(body, rng)
    all_cells = set(body) | {marker}
    r0, c0, r1, c1 = _bbox(all_cells)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in body:
        patch[int(r - r0), int(c - c0)] = int(body_color)
    patch[int(marker[0] - r0), int(marker[1] - c0)] = int(marker_color)
    return {
        "patch": patch,
        "offset": (int(marker[0] - r0), int(marker[1] - c0)),
    }


def _top_left(sprite: dict[str, object], marker_r: int, marker_c: int) -> tuple[int, int]:
    off_r, off_c = sprite["offset"]
    return int(marker_r - off_r), int(marker_c - off_c)


def _anchor_candidates(height: int, width: int, axis: str) -> list[tuple[int, int]]:
    if axis == "h":
        cols = list(range(_ANCHOR_MARGIN, width - _ANCHOR_MARGIN))
        return [(0, c) for c in cols] + [(height - 1, c) for c in cols]
    rows = list(range(_ANCHOR_MARGIN, height - _ANCHOR_MARGIN))
    return [(r, 0) for r in rows] + [(r, width - 1) for r in rows]


def _anchor_ok(canvas: np.ndarray, r: int, c: int, bg: int) -> bool:
    h, w = canvas.shape
    if r not in (0, h - 1) and c not in (0, w - 1):
        return False
    if r in (0, h - 1) and c in (0, w - 1):
        return False
    r0, r1 = max(0, r - 1), min(h, r + 2)
    c0, c1 = max(0, c - 1), min(w, c + 2)
    return not bool((canvas[r0:r1, c0:c1] != int(bg)).any())


def _can_place(canvas: np.ndarray, sprite: dict[str, object], marker_r: int, marker_c: int, bg: int) -> bool:
    patch = sprite["patch"]
    top, left = _top_left(sprite, marker_r, marker_c)
    h, w = patch.shape
    H, W = canvas.shape
    if top < 1 or left < 1 or top + h > H - 1 or left + w > W - 1:
        return False

    mask = patch != int(bg)
    coords = np.argwhere(mask)
    for pr, pc in coords:
        rr = int(top + pr)
        cc = int(left + pc)
        if int(canvas[rr, cc]) != int(bg):
            return False
        for nr in range(max(0, rr - 1), min(H, rr + 2)):
            for nc in range(max(0, cc - 1), min(W, cc + 2)):
                if int(canvas[nr, nc]) != int(bg):
                    return False
    return True


def _place(canvas: np.ndarray, sprite: dict[str, object], marker_r: int, marker_c: int, bg: int) -> None:
    patch = sprite["patch"]
    top, left = _top_left(sprite, marker_r, marker_c)
    mask = patch != int(bg)
    coords = np.argwhere(mask)
    for pr, pc in coords:
        canvas[int(top + pr), int(left + pc)] = int(patch[int(pr), int(pc)])


def _aligned_positions(
    canvas: np.ndarray,
    sprite: dict[str, object],
    anchor: dict[str, int | str],
    bg: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    H, W = canvas.shape
    if anchor["axis"] == "h":
        rows = list(range(1, H - 1))
        rng.shuffle(rows)
        return [
            (int(r), int(anchor["c"]))
            for r in rows
            if _can_place(canvas, sprite, int(r), int(anchor["c"]), bg)
        ]
    cols = list(range(1, W - 1))
    rng.shuffle(cols)
    return [
        (int(anchor["r"]), int(c))
        for c in cols
        if _can_place(canvas, sprite, int(anchor["r"]), int(c), bg)
    ]


def _scatter_positions(
    canvas: np.ndarray,
    sprite: dict[str, object],
    anchor: dict[str, int | str],
    solved_marker: tuple[int, int],
    bg: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    H, W = canvas.shape
    mr, mc = solved_marker
    if anchor["axis"] == "h":
        cols = list(range(1, W - 1))
        rng.shuffle(cols)
        return [
            (int(mr), int(c))
            for c in cols
            if int(c) != int(anchor["c"]) and _can_place(canvas, sprite, int(mr), int(c), bg)
        ]
    rows = list(range(1, H - 1))
    rng.shuffle(rows)
    return [
        (int(r), int(mc))
        for r in rows
        if int(r) != int(anchor["r"]) and _can_place(canvas, sprite, int(r), int(mc), bg)
    ]


def _bg_is_mode(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.ravel()))
    return counts[int(bg)] > max((count for color, count in counts.items() if color != int(bg)), default=0)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    H = rng.randint(*_GRID_H_RANGE)
    W = rng.randint(*_GRID_W_RANGE)
    bg = rng.choice(_COLORS)
    out = np.full((H, W), int(bg), dtype=int)
    inp = np.full((H, W), int(bg), dtype=int)

    n_sprites = rng.randint(*_SPRITE_COUNT_RANGE)
    branches = ["h", "v"] + [rng.choice(["h", "v"]) for _ in range(max(0, n_sprites - 2))]
    rng.shuffle(branches)

    marker_colors = rng.sample([c for c in _COLORS if c != int(bg)], n_sprites)
    body_pool = [c for c in _COLORS if c != int(bg) and c not in marker_colors]
    if len(body_pool) < n_sprites:
        return None
    rng.shuffle(body_pool)

    anchors: list[dict[str, int | str]] = []
    for branch, marker_color in zip(branches, marker_colors):
        candidates = _anchor_candidates(H, W, branch)
        rng.shuffle(candidates)
        placed = None
        for r, c in candidates:
            if _anchor_ok(out, int(r), int(c), bg):
                placed = {"axis": branch, "r": int(r), "c": int(c), "color": int(marker_color)}
                break
        if placed is None:
            return None
        out[int(placed["r"]), int(placed["c"])] = int(marker_color)
        inp[int(placed["r"]), int(placed["c"])] = int(marker_color)
        anchors.append(placed)

    sprites: list[tuple[dict[str, object], dict[str, int | str], tuple[int, int]]] = []
    for anchor, body_color in zip(anchors, body_pool):
        sprite = _make_sprite(rng, rng.randint(*_BODY_AREA_RANGE), int(body_color), int(anchor["color"]), bg)
        solved_positions = _aligned_positions(out, sprite, anchor, bg, rng)
        if not solved_positions:
            return None
        solved_marker = rng.choice(solved_positions)
        _place(out, sprite, solved_marker[0], solved_marker[1], bg)
        sprites.append((sprite, anchor, solved_marker))

    rng.shuffle(sprites)
    for sprite, anchor, solved_marker in sprites:
        scattered = _scatter_positions(inp, sprite, anchor, solved_marker, bg, rng)
        if not scattered:
            return None
        input_marker = rng.choice(scattered)
        _place(inp, sprite, input_marker[0], input_marker[1], bg)

    if not _bg_is_mode(inp, bg):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for task 2")
