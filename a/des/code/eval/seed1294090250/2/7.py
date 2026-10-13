import random
from collections import Counter

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
OPENINGS = ("up", "down", "left", "right")
OPENING_TO_TARGET = {
    "up": "bottom",
    "down": "top",
    "left": "right",
    "right": "left",
}
STAMP_GAP = {
    "up": (0, 1),
    "down": (2, 1),
    "left": (1, 0),
    "right": (1, 2),
}
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid).flat))
    return int(max(counts, key=counts.get))


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
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_envelope(grid: np.ndarray, color: int, coords: list[tuple[int, int]], bg: int) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return False
    if len(coords) != 2 * h + 2 * w - 4:
        return False

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(sub[0, :] == color):
        return False
    if not np.all(sub[-1, :] == color):
        return False
    if not np.all(sub[:, 0] == color):
        return False
    if not np.all(sub[:, -1] == color):
        return False
    if not np.all(sub[1:-1, 1:-1] == bg):
        return False
    return True


def _detect_envelopes(grid: np.ndarray, bg: int) -> dict[int, tuple[int, int, int, int]]:
    found: dict[int, list[tuple[int, int, int, int]]] = {}
    for color in sorted(set(map(int, grid.flat)) - {bg}):
        for comp in _components(grid == color):
            if _is_envelope(grid, color, comp, bg):
                found.setdefault(color, []).append(_bbox(comp))
    return {color: boxes[0] for color, boxes in found.items() if len(boxes) == 1}


def _detect_stamps(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    stamps: list[dict[str, object]] = []
    for r in range(h - 2):
        for c in range(w - 2):
            sub = grid[r : r + 3, c : c + 3]
            mark = int(sub[1, 1])
            if mark == bg:
                continue

            corners = [int(sub[0, 0]), int(sub[0, 2]), int(sub[2, 0]), int(sub[2, 2])]
            if len(set(corners)) != 1:
                continue
            border = corners[0]
            if border == bg or border == mark:
                continue

            edges = {
                "up": int(sub[0, 1]),
                "down": int(sub[2, 1]),
                "left": int(sub[1, 0]),
                "right": int(sub[1, 2]),
            }
            missing = [name for name, value in edges.items() if value == bg]
            if len(missing) != 1:
                continue
            if any(value != border for name, value in edges.items() if name != missing[0]):
                continue

            stamps.append(
                {
                    "bbox": (r, c, r + 2, c + 2),
                    "border": border,
                    "marker": mark,
                    "opening": missing[0],
                }
            )
    return stamps


def _target_cell(bbox: tuple[int, int, int, int], opening: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    target = OPENING_TO_TARGET[opening]
    if target == "top":
        return r0 + 1, mid_c
    if target == "bottom":
        return r1 - 1, mid_c
    if target == "left":
        return mid_r, c0 + 1
    return mid_r, c1 - 1


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    envelopes = _detect_envelopes(g, bg)
    stamps = _detect_stamps(g, bg)

    out = g.copy()
    for stamp in stamps:
        r0, c0, r1, c1 = stamp["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = bg

    for stamp in stamps:
        env_bbox = envelopes.get(int(stamp["border"]))
        if env_bbox is None:
            continue
        tr, tc = _target_cell(env_bbox, str(stamp["opening"]))
        out[tr, tc] = int(stamp["marker"])

    return out


def _draw_envelope(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    grid[top, left : left + width] = color
    grid[top + height - 1, left : left + width] = color
    grid[top : top + height, left] = color
    grid[top : top + height, left + width - 1] = color


def _draw_stamp(grid: np.ndarray, top: int, left: int, border: int, marker: int, opening: str) -> None:
    grid[top : top + 3, left : left + 3] = border
    gap_r, gap_c = STAMP_GAP[opening]
    grid[top + gap_r, left + gap_c] = BG
    grid[top + 1, left + 1] = marker


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = occupied.shape
    if top < 1 or left < 1 or top + height >= h or left + width >= w:
        return False
    return not occupied[top - 1 : top + height + 1, left - 1 : left + width + 1].any()


def _place_box(
    occupied: np.ndarray,
    height: int,
    width: int,
    rng: random.Random,
) -> tuple[int, int] | None:
    h, w = occupied.shape
    candidates = []
    for top in range(1, h - height):
        for left in range(1, w - width):
            if _can_place(occupied, top, left, height, width):
                candidates.append((top, left))
    if not candidates:
        return None
    top, left = rng.choice(candidates)
    occupied[top - 1 : top + height + 1, left - 1 : left + width + 1] = True
    return top, left


def _sample_sizes(rng: random.Random, pair_count: int) -> list[tuple[int, int]]:
    max_dim = {1: 13, 2: 11, 3: 9, 4: 7}[pair_count]
    choices = list(range(5, max_dim + 1, 2))
    sizes = []
    for _ in range(pair_count):
        sizes.append((rng.choice(choices), rng.choice(choices)))
    return sizes


def _sample_directions(rng: random.Random, pair_count: int) -> list[str]:
    dirs = list(OPENINGS)
    rng.shuffle(dirs)
    return dirs[:pair_count]


def _build_expected(
    grid: np.ndarray,
    specs: list[dict[str, object]],
) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        sr, sc = spec["stamp_top_left"]
        out[sr : sr + 3, sc : sc + 3] = BG
    for spec in specs:
        tr, tc = _target_cell(spec["env_bbox"], str(spec["opening"]))
        out[tr, tc] = int(spec["marker"])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.choice([1, 2, 3, 4])
    side_min, side_max = {
        1: (14, 24),
        2: (18, 28),
        3: (22, 30),
        4: (24, 32),
    }[pair_count]
    height = rng.randint(side_min, side_max)
    width = rng.randint(side_min, side_max)

    colors = list(COLOR_POOL)
    rng.shuffle(colors)
    border_colors = colors[:pair_count]
    marker_colors = colors[pair_count : 2 * pair_count]
    directions = _sample_directions(rng, pair_count)
    sizes = _sample_sizes(rng, pair_count)

    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    specs: list[dict[str, object]] = []

    order = list(range(pair_count))
    order.sort(key=lambda idx: sizes[idx][0] * sizes[idx][1], reverse=True)

    placed_envs: dict[int, tuple[int, int, int, int]] = {}
    for idx in order:
        env_h, env_w = sizes[idx]
        pos = _place_box(occupied, env_h, env_w, rng)
        if pos is None:
            return None
        top, left = pos
        _draw_envelope(grid, top, left, env_h, env_w, border_colors[idx])
        placed_envs[idx] = (top, left, top + env_h - 1, left + env_w - 1)

    for idx in range(pair_count):
        pos = _place_box(occupied, 3, 3, rng)
        if pos is None:
            return None
        top, left = pos
        _draw_stamp(grid, top, left, border_colors[idx], marker_colors[idx], directions[idx])
        specs.append(
            {
                "border": border_colors[idx],
                "marker": marker_colors[idx],
                "opening": directions[idx],
                "env_bbox": placed_envs[idx],
                "stamp_top_left": (top, left),
            }
        )

    detected_envs = _detect_envelopes(grid, BG)
    detected_stamps = _detect_stamps(grid, BG)
    if set(detected_envs.keys()) != {int(spec["border"]) for spec in specs}:
        return None
    if len(detected_stamps) != pair_count:
        return None

    expected = _build_expected(grid, specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid puzzle instance")
