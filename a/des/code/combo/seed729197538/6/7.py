from __future__ import annotations

import numpy as np


BG = 0
KEY_COLOR = 8
FRAME_COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 9)
GEN_ATTEMPTS = 400

DIR_TO_DELTA = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}

# The azure key is a 2x2 L tromino. Its "opening" direction is the side
# associated with the missing corner under this fixed cyclic mapping.
KEY_PATTERNS = {
    "up": ((0, 0), (1, 0), (1, 1)),
    "right": ((0, 0), (0, 1), (1, 0)),
    "down": ((0, 0), (0, 1), (1, 1)),
    "left": ((0, 1), (1, 0), (1, 1)),
}
PATTERN_TO_DIR = {frozenset(cells): direction for direction, cells in KEY_PATTERNS.items()}

DECOR_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _line_status(line: np.ndarray, color: int) -> str:
    missing = [idx for idx, value in enumerate(line) if int(value) == BG]
    if not missing:
        return "solid" if all(int(value) == color for value in line) else "bad"
    if len(missing) != 2 or missing[1] != missing[0] + 1:
        return "bad"
    if missing[0] == 0 or missing[1] == len(line) - 1:
        return "bad"
    for idx, value in enumerate(line):
        if idx in missing:
            continue
        if int(value) != color:
            return "bad"
    return "gap"


def _frame_from_component(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None
    if len(coords) != 2 * (h + w) - 6:
        return None
    for r, c in coords:
        if r not in (r0, r1) and c not in (c0, c1):
            return None

    sides = {
        "up": _line_status(grid[r0, c0 : c1 + 1], color),
        "down": _line_status(grid[r1, c0 : c1 + 1], color),
        "left": _line_status(grid[r0 : r1 + 1, c0], color),
        "right": _line_status(grid[r0 : r1 + 1, c1], color),
    }
    gap_sides = [side for side, status in sides.items() if status == "gap"]
    if len(gap_sides) != 1:
        return None
    if any(status == "bad" for status in sides.values()):
        return None
    return {
        "bbox": (r0, c0, r1, c1),
        "color": int(color),
        "gap_side": gap_sides[0],
        "inner_h": h - 2,
        "inner_w": w - 2,
    }


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(value) for value in np.unique(grid) if int(value) != BG):
        for coords in _components(grid == color):
            frame = _frame_from_component(grid, coords, color)
            if frame is not None:
                frames.append(frame)
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _key_direction(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> str | None:
    r0, c0, r1, c1 = bbox
    sub = grid[r0 + 1 : r1, c0 + 1 : c1] == KEY_COLOR
    matches: list[str] = []
    for comp in _components(sub):
        if len(comp) != 3:
            continue
        kr0, kc0, kr1, kc1 = _bbox(comp)
        if kr1 - kr0 != 1 or kc1 - kc0 != 1:
            continue
        norm = frozenset((r - kr0, c - kc0) for r, c in comp)
        direction = PATTERN_TO_DIR.get(norm)
        if direction is not None:
            matches.append(direction)
    if len(matches) == 1:
        return matches[0]
    return None


def _interior_pixels(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = bbox
    pixels: list[tuple[int, int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            color = int(grid[r, c])
            if color != BG:
                pixels.append((r, c, color))
    return pixels


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for frame in _find_frames(g):
        direction = _key_direction(g, frame["bbox"])
        if direction is None or direction != frame["gap_side"]:
            continue

        r0, c0, r1, c1 = frame["bbox"]
        span = frame["inner_h"] if direction in ("up", "down") else frame["inner_w"]
        dr, dc = DIR_TO_DELTA[direction]
        shift_r = dr * span
        shift_c = dc * span
        contents = _interior_pixels(g, frame["bbox"])

        out[r0 + 1 : r1, c0 + 1 : c1] = BG
        for r, c, color in contents:
            nr = r + shift_r
            nc = c + shift_c
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                out[nr, nc] = color

    return out


def _rng_choice(rng: np.random.Generator, seq):
    return seq[int(rng.integers(0, len(seq)))]


def _touches_occupied(occupied: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        for nr in range(max(0, r - 1), min(h, r + 2)):
            for nc in range(max(0, c - 1), min(w, c + 2)):
                if occupied[nr, nc]:
                    return True
    return False


def _place_shape(
    rng: np.random.Generator,
    occupied: np.ndarray,
    inner: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    color: int,
) -> bool:
    inner_h, inner_w = inner.shape
    shape_h = 1 + max(r for r, _ in shape)
    shape_w = 1 + max(c for _, c in shape)
    max_r = inner_h - shape_h - 1
    max_c = inner_w - shape_w - 1
    if max_r < 1 or max_c < 1:
        return False

    positions = [(r, c) for r in range(1, max_r + 1) for c in range(1, max_c + 1)]
    rng.shuffle(positions)
    for base_r, base_c in positions:
        cells = [(base_r + dr, base_c + dc) for dr, dc in shape]
        if _touches_occupied(occupied, cells):
            continue
        for r, c in cells:
            occupied[r, c] = True
            inner[r, c] = color
        return True
    return False


def _gap_start(rng: np.random.Generator, side_len: int) -> int:
    return int(rng.integers(1, side_len - 2))


def _active_flags(rng: np.random.Generator, n_frames: int) -> list[bool]:
    flags = [bool(rng.integers(0, 2)) for _ in range(n_frames)]
    if n_frames > 1:
        if all(flags):
            flags[int(rng.integers(0, n_frames))] = False
        if not any(flags):
            flags[int(rng.integers(0, n_frames))] = True
    return flags


def _frame_spec(
    rng: np.random.Generator,
    frame_color: int,
    decor_colors: tuple[int, ...],
    active: bool,
) -> dict | None:
    inner_h = int(rng.integers(4, 7))
    inner_w = int(rng.integers(4, 7))
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    direction = _rng_choice(rng, ("up", "right", "down", "left"))
    if active:
        gap_side = direction
    else:
        gap_side = _rng_choice(rng, tuple(side for side in DIR_TO_DELTA if side != direction))

    inner = np.zeros((inner_h, inner_w), dtype=int)
    occupied = np.zeros((inner_h, inner_w), dtype=bool)

    key_shape = KEY_PATTERNS[direction]
    key_r = int(rng.integers(1, inner_h - 1))
    key_c = int(rng.integers(1, inner_w - 1))
    key_cells = [(key_r + dr, key_c + dc) for dr, dc in key_shape]
    if any(r >= inner_h - 1 or c >= inner_w - 1 for r, c in key_cells):
        return None
    for r, c in key_cells:
        occupied[r, c] = True
        inner[r, c] = KEY_COLOR

    n_decor = int(rng.integers(0, 4))
    for _ in range(n_decor):
        shape = _rng_choice(rng, DECOR_SHAPES)
        color = int(_rng_choice(rng, decor_colors))
        _place_shape(rng, occupied, inner, shape, color)

    patch = np.zeros((frame_h, frame_w), dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    patch[1:-1, 1:-1] = inner

    if gap_side == "up":
        start = _gap_start(rng, frame_w)
        patch[0, start : start + 2] = BG
    elif gap_side == "down":
        start = _gap_start(rng, frame_w)
        patch[-1, start : start + 2] = BG
    elif gap_side == "left":
        start = _gap_start(rng, frame_h)
        patch[start : start + 2, 0] = BG
    else:
        start = _gap_start(rng, frame_h)
        patch[start : start + 2, -1] = BG

    if active and direction == "right":
        reserve = (-1, -1, frame_h, frame_w + inner_w)
    elif active and direction == "left":
        reserve = (-1, -inner_w - 1, frame_h, frame_w)
    elif active and direction == "up":
        reserve = (-inner_h - 1, -1, frame_h, frame_w)
    elif active and direction == "down":
        reserve = (-1, -1, frame_h + inner_h, frame_w)
    else:
        reserve = (-1, -1, frame_h, frame_w)

    contents_local = [
        (r, c, int(inner[r, c]))
        for r in range(inner_h)
        for c in range(inner_w)
        if int(inner[r, c]) != BG
    ]

    return {
        "frame_color": int(frame_color),
        "direction": direction,
        "gap_side": gap_side,
        "active": bool(active),
        "inner_h": inner_h,
        "inner_w": inner_w,
        "patch": patch,
        "reserve": reserve,
        "contents_local": contents_local,
    }


def _place_specs(rng: np.random.Generator, specs: list[dict], h: int, w: int) -> list[dict] | None:
    occupancy = np.zeros((h, w), dtype=bool)
    placed = [dict(spec) for spec in specs]
    order = sorted(
        range(len(placed)),
        key=lambda idx: (placed[idx]["reserve"][2] - placed[idx]["reserve"][0] + 1)
        * (placed[idx]["reserve"][3] - placed[idx]["reserve"][1] + 1),
        reverse=True,
    )

    for idx in order:
        spec = placed[idx]
        rr0, cc0, rr1, cc1 = map(int, spec["reserve"])
        r_min = max(0, -rr0)
        r_max = h - 1 - rr1
        c_min = max(0, -cc0)
        c_max = w - 1 - cc1
        if r_min > r_max or c_min > c_max:
            return None

        found = False
        for _ in range(200):
            top = int(rng.integers(r_min, r_max + 1))
            left = int(rng.integers(c_min, c_max + 1))
            region = occupancy[top + rr0 : top + rr1 + 1, left + cc0 : left + cc1 + 1]
            if np.any(region):
                continue
            occupancy[top + rr0 : top + rr1 + 1, left + cc0 : left + cc1 + 1] = True
            spec["top"] = top
            spec["left"] = left
            found = True
            break
        if not found:
            return None
    return placed


def _render_grid(h: int, w: int, specs: list[dict]) -> tuple[np.ndarray, list[dict]]:
    grid = np.zeros((h, w), dtype=int)
    meta: list[dict] = []
    for spec in specs:
        patch = spec["patch"]
        top = int(spec["top"])
        left = int(spec["left"])
        ph, pw = patch.shape
        region = grid[top : top + ph, left : left + pw]
        mask = patch != BG
        region[mask] = patch[mask]

        contents = [
            (top + 1 + r, left + 1 + c, color)
            for r, c, color in spec["contents_local"]
        ]
        meta.append(
            {
                "bbox": (top, left, top + ph - 1, left + pw - 1),
                "active": bool(spec["active"]),
                "direction": spec["direction"],
                "contents": contents,
            }
        )
    return grid, meta


def _expected_from_meta(grid: np.ndarray, meta: list[dict]) -> np.ndarray:
    out = grid.copy()
    for frame in meta:
        if not frame["active"]:
            continue
        r0, c0, r1, c1 = frame["bbox"]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        span = inner_h if frame["direction"] in ("up", "down") else inner_w
        dr, dc = DIR_TO_DELTA[frame["direction"]]
        out[r0 + 1 : r1, c0 + 1 : c1] = BG
        for r, c, color in frame["contents"]:
            out[r + dr * span, c + dc * span] = color
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_frames = int(rng.integers(2, 5))
    frame_colors = list(rng.choice(FRAME_COLOR_POOL, size=n_frames, replace=False))
    decor_palette = tuple(color for color in FRAME_COLOR_POOL if color not in frame_colors)
    flags = _active_flags(rng, n_frames)

    specs: list[dict] = []
    for color, active in zip(frame_colors, flags):
        spec = _frame_spec(rng, int(color), decor_palette, active)
        if spec is None:
            return None
        specs.append(spec)

    h = int(rng.integers(22 + n_frames, 31))
    w = int(rng.integers(22 + n_frames, 31))
    placed = _place_specs(rng, specs, h, w)
    if placed is None:
        return None

    grid, meta = _render_grid(h, w, placed)
    expected = _expected_from_meta(grid, meta)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
