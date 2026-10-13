import random
import numpy as np


BG = 0
DIRECTIONS = ("up", "right", "down", "left")
ROTATION_STEPS = {"up": 0, "right": 3, "down": 2, "left": 1}
MAX_GENERATION_ATTEMPTS = 400
MAX_PLACEMENT_ATTEMPTS = 500


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                if rr > 0 and mask[rr - 1, cc] and not seen[rr - 1, cc]:
                    seen[rr - 1, cc] = True
                    stack.append((rr - 1, cc))
                if rr + 1 < h and mask[rr + 1, cc] and not seen[rr + 1, cc]:
                    seen[rr + 1, cc] = True
                    stack.append((rr + 1, cc))
                if cc > 0 and mask[rr, cc - 1] and not seen[rr, cc - 1]:
                    seen[rr, cc - 1] = True
                    stack.append((rr, cc - 1))
                if cc + 1 < w and mask[rr, cc + 1] and not seen[rr, cc + 1]:
                    seen[rr, cc + 1] = True
                    stack.append((rr, cc + 1))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop_non_bg(arr: np.ndarray) -> np.ndarray:
    coords = np.argwhere(arr != BG)
    if coords.size == 0:
        return arr.copy()
    r0, c0, r1, c1 = _bbox(coords)
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _ring_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _gap_direction(
    gap_r: int, gap_c: int, r0: int, c0: int, r1: int, c1: int
) -> str | None:
    if gap_r == r0 and c0 < gap_c < c1:
        return "up"
    if gap_r == r1 and c0 < gap_c < c1:
        return "down"
    if gap_c == c0 and r0 < gap_r < r1:
        return "left"
    if gap_c == c1 and r0 < gap_r < r1:
        return "right"
    return None


def _classify_frame_component(
    grid: np.ndarray, color: int, coords: np.ndarray
) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None

    border = _ring_cells(r0, c0, r1, c1)
    comp_cells = {tuple(map(int, rc)) for rc in coords}
    if not comp_cells.issubset(border):
        return None

    missing = list(border - comp_cells)
    if len(missing) > 1:
        return None
    if len(comp_cells) != len(border) - len(missing):
        return None

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    if not missing:
        inner_mask = inner != BG
        if not bool(inner_mask.any()):
            return None
        inner_colors = {int(v) for v in inner[inner_mask]}
        if len(inner_colors) != 1 or color in inner_colors:
            return None
        pattern = _crop_non_bg(inner)
        return {
            "kind": "stamp",
            "color": int(color),
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "pattern": pattern,
        }

    if bool((inner != BG).any()):
        return None
    gap_r, gap_c = map(int, missing[0])
    direction = _gap_direction(gap_r, gap_c, r0, c0, r1, c1)
    if direction is None:
        return None
    return {
        "kind": "canvas",
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "direction": direction,
    }


def _extract_objects(grid: np.ndarray) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    stamps: list[dict[str, object]] = []
    canvases: list[dict[str, object]] = []
    colors = sorted({int(v) for v in grid.ravel()} - {BG})
    for color in colors:
        for coords in _components(grid == color):
            obj = _classify_frame_component(grid, color, coords)
            if obj is None:
                continue
            if obj["kind"] == "stamp":
                stamps.append(obj)
            else:
                canvases.append(obj)
    return stamps, canvases


def _rotate_pattern(pattern: np.ndarray, direction: str) -> np.ndarray:
    return np.rot90(pattern, ROTATION_STEPS[direction]).copy()


def _pattern_top_left(
    bbox: tuple[int, int, int, int], pattern_shape: tuple[int, int], direction: str
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    inner_h = int(r1 - r0 - 1)
    inner_w = int(c1 - c0 - 1)
    ph, pw = map(int, pattern_shape)
    if direction == "up":
        return int(r1 - ph), int(c0 + 1 + (inner_w - pw) // 2)
    if direction == "down":
        return int(r0 + 1), int(c0 + 1 + (inner_w - pw) // 2)
    if direction == "left":
        return int(r0 + 1 + (inner_h - ph) // 2), int(c1 - pw)
    return int(r0 + 1 + (inner_h - ph) // 2), int(c0 + 1)


def _paste_non_bg(dst: np.ndarray, src: np.ndarray, top: int, left: int) -> None:
    h, w = src.shape
    patch = dst[top : top + h, left : left + w]
    mask = src != BG
    patch[mask] = src[mask]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    stamps, canvases = _extract_objects(g)
    stamp_by_color: dict[int, dict[str, object]] = {}
    canvas_by_color: dict[int, dict[str, object]] = {}

    for stamp in stamps:
        stamp_by_color[int(stamp["color"])] = stamp
    for canvas in canvases:
        canvas_by_color[int(canvas["color"])] = canvas

    for stamp in stamp_by_color.values():
        r0, c0, r1, c1 = stamp["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BG

    for color, canvas in canvas_by_color.items():
        if color not in stamp_by_color:
            continue
        bbox = canvas["bbox"]
        r0, c0, r1, c1 = bbox
        out[r0 + 1 : r1, c0 + 1 : c1] = BG
        rotated = _rotate_pattern(stamp_by_color[color]["pattern"], str(canvas["direction"]))
        top, left = _pattern_top_left(bbox, rotated.shape, str(canvas["direction"]))
        _paste_non_bg(out, rotated, top, left)

    return out


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_connected(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _is_asymmetric(mask: np.ndarray) -> bool:
    for k in (1, 2, 3):
        rot = np.rot90(mask, k)
        if rot.shape == mask.shape and np.array_equal(rot, mask):
            return False
    if np.array_equal(mask, np.flipud(mask)):
        return False
    if np.array_equal(mask, np.fliplr(mask)):
        return False
    if mask.shape[0] == mask.shape[1]:
        if np.array_equal(mask, mask.T):
            return False
        if np.array_equal(mask, np.fliplr(mask).T):
            return False
    return True


def _random_pattern(color: int, rng: random.Random) -> np.ndarray:
    for _ in range(200):
        h = rng.randint(2, 3)
        w = rng.randint(2, 3)
        target = rng.randint(2, h * w - 1)
        mask = np.zeros((h, w), dtype=bool)
        start = (rng.randrange(h), rng.randrange(w))
        mask[start] = True
        frontier: list[tuple[int, int]] = []

        def add_neighbors(r: int, c: int) -> None:
            if r > 0 and not mask[r - 1, c]:
                frontier.append((r - 1, c))
            if r + 1 < h and not mask[r + 1, c]:
                frontier.append((r + 1, c))
            if c > 0 and not mask[r, c - 1]:
                frontier.append((r, c - 1))
            if c + 1 < w and not mask[r, c + 1]:
                frontier.append((r, c + 1))

        add_neighbors(*start)
        while int(mask.sum()) < target and frontier:
            idx = rng.randrange(len(frontier))
            r, c = frontier.pop(idx)
            if mask[r, c]:
                continue
            mask[r, c] = True
            add_neighbors(r, c)

        if int(mask.sum()) != target:
            continue
        cropped = _crop_bool(mask)
        if not _is_connected(cropped):
            continue
        if not _is_asymmetric(cropped):
            continue
        if bool(cropped.all()):
            continue
        pattern = np.zeros(cropped.shape, dtype=int)
        pattern[cropped] = int(color)
        return pattern
    raise RuntimeError("failed to build asymmetric pattern")


def _make_stamp(pattern: np.ndarray, border_color: int) -> np.ndarray:
    h, w = pattern.shape
    stamp = np.full((h + 2, w + 2), int(border_color), dtype=int)
    stamp[1:-1, 1:-1] = BG
    stamp[1:-1, 1:-1][pattern != BG] = pattern[pattern != BG]
    return stamp


def _make_canvas(
    border_color: int, direction: str, pattern_shape: tuple[int, int], rng: random.Random
) -> np.ndarray | None:
    ph, pw = map(int, pattern_shape)
    if direction in ("up", "down"):
        h_choices = list(range(max(5, ph + 1), 9))
        w_choices = [w for w in range(max(5, pw + 2), 9) if (w - pw) % 2 == 0]
    else:
        h_choices = [h for h in range(max(5, ph + 2), 9) if (h - ph) % 2 == 0]
        w_choices = list(range(max(5, pw + 1), 9))
    if not h_choices or not w_choices:
        return None

    inner_h = rng.choice(h_choices)
    inner_w = rng.choice(w_choices)
    canvas = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    canvas[0, :] = int(border_color)
    canvas[-1, :] = int(border_color)
    canvas[:, 0] = int(border_color)
    canvas[:, -1] = int(border_color)

    if direction == "up":
        canvas[0, rng.randint(1, inner_w)] = BG
    elif direction == "down":
        canvas[-1, rng.randint(1, inner_w)] = BG
    elif direction == "left":
        canvas[rng.randint(1, inner_h), 0] = BG
    else:
        canvas[rng.randint(1, inner_h), -1] = BG
    return canvas


def _place_rectangles(
    items: list[dict[str, object]], height: int, width: int, rng: random.Random
) -> dict[tuple[str, int], tuple[int, int]] | None:
    blocked = np.zeros((height, width), dtype=bool)
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True
    placements: dict[tuple[str, int], tuple[int, int]] = {}

    ordered = sorted(items, key=lambda item: -int(np.prod(item["array"].shape)))
    for item in ordered:
        arr = item["array"]
        h, w = map(int, arr.shape)
        if h + 2 >= height or w + 2 >= width:
            return None
        placed = False
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            r = rng.randint(1, height - h - 1)
            c = rng.randint(1, width - w - 1)
            rr0 = max(0, r - 1)
            rr1 = min(height, r + h + 1)
            cc0 = max(0, c - 1)
            cc1 = min(width, c + w + 1)
            if bool(blocked[rr0:rr1, cc0:cc1].any()):
                continue
            placements[item["key"]] = (int(r), int(c))
            blocked[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return None
    return placements


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    n_pairs = rng.randint(2, 4)
    palette = rng.sample(list(range(1, 10)), 2 * n_pairs)
    border_colors = palette[:n_pairs]
    pattern_colors = palette[n_pairs:]
    directions = rng.sample(list(DIRECTIONS), n_pairs)

    pairs: list[dict[str, object]] = []
    for idx in range(n_pairs):
        pattern = _random_pattern(pattern_colors[idx], rng)
        stamp = _make_stamp(pattern, border_colors[idx])
        rotated = _rotate_pattern(pattern, directions[idx])
        canvas = _make_canvas(border_colors[idx], directions[idx], rotated.shape, rng)
        if canvas is None:
            return None
        pairs.append(
            {
                "pattern": pattern,
                "stamp": stamp,
                "canvas": canvas,
                "direction": directions[idx],
            }
        )

    items: list[dict[str, object]] = []
    for idx, pair in enumerate(pairs):
        items.append({"key": ("stamp", idx), "array": pair["stamp"]})
        items.append({"key": ("canvas", idx), "array": pair["canvas"]})
    rng.shuffle(items)

    height = rng.randint(14 + 4 * n_pairs, 20 + 4 * n_pairs)
    width = rng.randint(14 + 4 * n_pairs, 20 + 4 * n_pairs)
    placements = _place_rectangles(items, height, width, rng)
    if placements is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    for item in items:
        r, c = placements[item["key"]]
        arr = item["array"]
        h, w = arr.shape
        grid[r : r + h, c : c + w] = arr

    expected = grid.copy()
    for idx, pair in enumerate(pairs):
        sr, sc = placements[("stamp", idx)]
        sh, sw = pair["stamp"].shape
        expected[sr : sr + sh, sc : sc + sw] = BG

        cr, cc = placements[("canvas", idx)]
        ch, cw = pair["canvas"].shape
        expected[cr + 1 : cr + ch - 1, cc + 1 : cc + cw - 1] = BG
        rotated = _rotate_pattern(pair["pattern"], str(pair["direction"]))
        bbox = (int(cr), int(cc), int(cr + ch - 1), int(cc + cw - 1))
        top, left = _pattern_top_left(bbox, rotated.shape, str(pair["direction"]))
        _paste_non_bg(expected, rotated, top, left)

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
