import numpy as np


AZURE = 8
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 9)
MAX_ATTEMPTS = 400
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _color_order(objects: list[dict]) -> list[int]:
    first_seen: dict[int, tuple[int, int]] = {}
    for obj in objects:
        color = obj["color"]
        key = min((c, r) for r, c in obj["antennas"])
        if color not in first_seen or key < first_seen[color]:
            first_seen[color] = key
    return [color for color, _ in sorted(first_seen.items(), key=lambda kv: kv[1])]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Place one grey marker for each foreground object using two independent attributes:
    antenna count chooses the row, and antenna color chooses the column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    fg_mask = g != AZURE
    comps = _components(fg_mask)
    assert comps, "expected at least one foreground object"

    objects: list[dict] = []
    used_slots: set[tuple[int, int]] = set()

    for comp in comps:
        pixels = set(comp)
        antennas: list[tuple[int, int]] = []
        antenna_colors: set[int] = set()

        for r, c in comp:
            nbrs = [
                (r + dr, c + dc)
                for dr, dc in NEIGHBORS4
                if (r + dr, c + dc) in pixels
            ]
            color = int(g[r, c])
            if color != GREY:
                assert len(nbrs) == 1, "antenna must have exactly one foreground neighbor"
                nr, nc = nbrs[0]
                assert int(g[nr, nc]) == GREY, "antenna must attach to the grey body"
                antennas.append((r, c))
                antenna_colors.add(color)

        assert 1 <= len(antennas) <= 4, "each object must have 1 to 4 antennas"
        assert len(antenna_colors) == 1, "all antennas of one object must share a color"

        body = pixels - set(antennas)
        assert body, "object must have a grey body"
        assert all(int(g[r, c]) == GREY for r, c in body), "non-antenna pixels must be grey"

        color = next(iter(antenna_colors))
        slot = (len(antennas), color)
        assert slot not in used_slots, "each antenna-count/color pair must be unique"
        used_slots.add(slot)
        objects.append({"count": len(antennas), "color": color, "antennas": antennas})

    color_order = _color_order(objects)
    color_to_col = {color: idx for idx, color in enumerate(color_order)}

    out = np.full((4, len(color_order)), AZURE, dtype=int)
    for obj in objects:
        out[obj["count"] - 1, color_to_col[obj["color"]]] = GREY
    return out


def _rectangle_cells(top: int, left: int, height: int, width: int) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r in range(top, top + height)
        for c in range(left, left + width)
    }


def _antenna_options(top: int, left: int, height: int, width: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    options: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for c in range(left + 1, left + width - 1):
        options.append(((top, c), (top - 1, c)))
        options.append(((top + height - 1, c), (top + height, c)))
    for r in range(top + 1, top + height - 1):
        options.append(((r, left), (r, left - 1)))
        options.append(((r, left + width - 1), (r, left + width)))
    return options


def _clear_with_margin(grid: np.ndarray, cells: set[tuple[int, int]], margin: int = 1) -> bool:
    h, w = grid.shape
    for r, c in cells:
        for rr in range(r - margin, r + margin + 1):
            for cc in range(c - margin, c + margin + 1):
                if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] != AZURE:
                    return False
    return True


def _build_object(
    top: int,
    left: int,
    height: int,
    width: int,
    antenna_count: int,
    color: int,
    rng: np.random.Generator,
) -> tuple[dict, set[tuple[int, int]]]:
    body = _rectangle_cells(top, left, height, width)
    options = _antenna_options(top, left, height, width)
    assert len(options) >= antenna_count
    rng.shuffle(options)

    chosen: list[tuple[int, int]] = []
    used_anchors: set[tuple[int, int]] = set()
    used_tips: set[tuple[int, int]] = set()
    for anchor, tip in options:
        if anchor in used_anchors or tip in used_tips:
            continue
        chosen.append(tip)
        used_anchors.add(anchor)
        used_tips.add(tip)
        if len(chosen) == antenna_count:
            break
    assert len(chosen) == antenna_count

    cells = set(body) | set(chosen)
    return {"body": body, "antennas": chosen, "color": color}, cells


def _paint_object(grid: np.ndarray, obj: dict) -> None:
    for r, c in obj["body"]:
        grid[r, c] = GREY
    for r, c in obj["antennas"]:
        grid[r, c] = obj["color"]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(16, 27))
    width = int(rng.integers(16, 27))
    grid = np.full((height, width), AZURE, dtype=int)

    color_count = int(rng.integers(2, min(6, len(PALETTE)) + 1))
    colors = list(rng.choice(PALETTE, size=color_count, replace=False))

    combos = [(count, color) for count in range(1, 5) for color in colors]
    chosen_set: set[tuple[int, int]] = set()
    for count in range(1, 5):
        chosen_set.add((count, int(rng.choice(colors))))
    for color in colors:
        chosen_set.add((int(rng.integers(1, 5)), color))

    max_objects = min(max(len(chosen_set), 8), len(combos))
    extras_pool = [combo for combo in combos if combo not in chosen_set]
    extra_cap = max(0, min(max_objects - len(chosen_set), len(extras_pool)))
    extra_count = int(rng.integers(0, extra_cap + 1))
    if extra_count:
        extra_ids = rng.choice(len(extras_pool), size=extra_count, replace=False)
        for idx in np.atleast_1d(extra_ids):
            chosen_set.add(extras_pool[int(idx)])

    chosen = list(chosen_set)
    rng.shuffle(chosen)

    for count, color in chosen:
        placed = False
        for _ in range(80):
            obj_h = int(rng.integers(3, 6))
            obj_w = int(rng.integers(3, 6))
            if 2 * (obj_h - 2) + 2 * (obj_w - 2) < count:
                continue
            top = int(rng.integers(1, height - obj_h))
            left = int(rng.integers(1, width - obj_w))
            obj, cells = _build_object(top, left, obj_h, obj_w, count, color, rng)
            if not _clear_with_margin(grid, cells, margin=1):
                continue
            _paint_object(grid, obj)
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    expected_cols = len({color for _, color in chosen})
    if out.shape != (4, expected_cols):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
