from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
MAX_GEN_ATTEMPTS = 300
FRAME_COLOR = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
BASE_SHAPES = (
    ("10", "11"),
    ("111", "010"),
    ("010", "111", "010"),
    ("110", "011"),
    ("101", "111"),
    ("111", "100", "100"),
    ("011", "110", "010"),
    ("111", "001", "111"),
)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _scale_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    src_h, src_w = mask.shape
    row_idx = np.minimum((np.arange(target_h) * src_h) // target_h, src_h - 1)
    col_idx = np.minimum((np.arange(target_w) * src_w) // target_w, src_w - 1)
    return mask[row_idx[:, None], col_idx[None, :]]


def _repeat_scale(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(mask, factor, axis=0), factor, axis=1)


def _random_shape(rng: random.Random) -> np.ndarray:
    rows = random.choice(BASE_SHAPES)
    mask = np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)
    mask = np.rot90(mask, rng.randrange(4))
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    return mask


def _frame_components(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color0 in np.unique(grid):
        color = int(color0)
        if color == bg:
            continue
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            border_area = 2 * (h + w) - 4
            if len(coords) != border_area:
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros((h, w), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.all(patch[border] == color):
                continue
            if np.any(patch[~border] == color):
                continue
            inner = patch[1:-1, 1:-1]
            marker_coords = np.argwhere((inner != bg) & (inner != color))
            if marker_coords.shape[0] != 1:
                continue
            marker_r, marker_c = map(int, marker_coords[0])
            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "inner_shape": inner.shape,
                    "marker_pos": (r0 + 1 + marker_r, c0 + 1 + marker_c),
                    "marker_color": int(inner[marker_r, marker_c]),
                }
            )
    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each fenced interior with the same-colored outside template scaled to that fence's inner rectangle.

    Input structure:
    - The background is black.
    - One or more one-pixel-thick closed rectangular fences sit on the grid, and each fence interior contains exactly one non-black singleton marker pixel.
    - Away from the fences, there are separate monochrome solid template shapes; each marker color matches exactly one outside template of the same color.

    Rule:
    - Detect every closed rectangular fence and read the color of its unique interior marker.
    - Find the outside template with the same color, take its tight binary mask, and scale that mask so its bounding box matches the fence interior size.
    - Clear the fence interior to black, then draw the scaled template there in the marker color while leaving the fence border and all outside templates unchanged.

    Key insight:
    - The singleton inside a fence is only a color key; the real geometry lives in the same-colored outside template, and the fence interior determines the target scale.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    bg = BG
    frames = _frame_components(g, bg)
    assert frames, "expected at least one closed fence"

    marker_positions = {frame["marker_pos"] for frame in frames}
    required_colors = {int(frame["marker_color"]) for frame in frames}
    templates: dict[int, np.ndarray] = {}

    for color in required_colors:
        for coords in _components(g == color):
            cells = {tuple(map(int, coord)) for coord in coords}
            if len(cells) == 1 and next(iter(cells)) in marker_positions:
                continue
            assert color not in templates, "each marker color should have a unique outside template"
            templates[color] = _tight_mask(coords)

    assert required_colors <= templates.keys(), "missing outside template for at least one marker color"

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inner_h, inner_w = frame["inner_shape"]
        color = int(frame["marker_color"])
        scaled = _scale_mask(templates[color], inner_h, inner_w)
        out[r0 + 1 : r1, c0 + 1 : c1] = bg
        out[r0 + 1 : r1, c0 + 1 : c1][scaled] = color

    return out


def _choose_pair_count(rng: random.Random, h: int, w: int) -> int:
    if min(h, w) >= 26:
        return rng.choice((2, 3))
    return rng.choice((1, 2))


def _pair_scale_options(pair_count: int, h: int, w: int) -> list[tuple[int, int]]:
    options = [(1, 2)]
    if pair_count == 1 and min(h, w) >= 24:
        options.extend([(1, 3), (2, 3)])
    elif pair_count == 2 and min(h, w) >= 27:
        options.append((2, 3))
    return options


def _can_place(occupied: np.ndarray, top: int, left: int, box_h: int, box_w: int, gap: int = 1) -> bool:
    h, w = occupied.shape
    if top < 0 or left < 0 or top + box_h > h or left + box_w > w:
        return False
    r0 = max(0, top - gap)
    c0 = max(0, left - gap)
    r1 = min(h, top + box_h + gap)
    c1 = min(w, left + box_w + gap)
    return not occupied[r0:r1, c0:c1].any()


def _place_box(occupied: np.ndarray, box_h: int, box_w: int, rng: random.Random, retries: int = 300) -> tuple[int, int] | None:
    h, w = occupied.shape
    for _ in range(retries):
        top = rng.randrange(h - box_h + 1)
        left = rng.randrange(w - box_w + 1)
        if _can_place(occupied, top, left, box_h, box_w):
            occupied[top : top + box_h, left : left + box_w] = True
            return top, left
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 30)
    w = rng.randint(18, 30)
    pair_count = _choose_pair_count(rng, h, w)
    colors = rng.sample(COLOR_POOL, pair_count)
    scale_options = _pair_scale_options(pair_count, h, w)

    pairs: list[dict] = []
    jobs: list[tuple[str, int, int, int]] = []

    for idx, color in enumerate(colors):
        base = _random_shape(rng)
        small, large = rng.choice(scale_options)
        mode = rng.choice(("up", "down"))
        if mode == "up":
            template_scale, inner_scale = small, large
        else:
            template_scale, inner_scale = large, small

        template_mask = _repeat_scale(base, template_scale)
        inner_h = base.shape[0] * inner_scale
        inner_w = base.shape[1] * inner_scale
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        pairs.append(
            {
                "color": color,
                "template_mask": template_mask,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": outer_h,
                "outer_w": outer_w,
            }
        )
        jobs.append(("frame", idx, outer_h, outer_w))
        jobs.append(("template", idx, template_mask.shape[0], template_mask.shape[1]))

    occupied = np.zeros((h, w), dtype=bool)
    positions: dict[tuple[str, int], tuple[int, int]] = {}
    jobs.sort(key=lambda item: item[2] * item[3], reverse=True)

    for kind, idx, box_h, box_w in jobs:
        pos = _place_box(occupied, box_h, box_w, rng)
        if pos is None:
            return None
        positions[(kind, idx)] = pos

    grid = np.full((h, w), BG, dtype=int)

    for idx, pair in enumerate(pairs):
        fr, fc = positions[("frame", idx)]
        tr, tc = positions[("template", idx)]
        outer_h = int(pair["outer_h"])
        outer_w = int(pair["outer_w"])
        inner_h = int(pair["inner_h"])
        inner_w = int(pair["inner_w"])
        color = int(pair["color"])

        grid[fr, fc : fc + outer_w] = FRAME_COLOR
        grid[fr + outer_h - 1, fc : fc + outer_w] = FRAME_COLOR
        grid[fr : fr + outer_h, fc] = FRAME_COLOR
        grid[fr : fr + outer_h, fc + outer_w - 1] = FRAME_COLOR

        mr = fr + 1 + rng.randrange(inner_h)
        mc = fc + 1 + rng.randrange(inner_w)
        grid[mr, mc] = color

        template_mask = pair["template_mask"]
        patch = grid[tr : tr + template_mask.shape[0], tc : tc + template_mask.shape[1]]
        patch[template_mask] = color

    try:
        out = main(grid)
    except AssertionError:
        return None

    for idx, pair in enumerate(pairs):
        fr, fc = positions[("frame", idx)]
        inner_h = int(pair["inner_h"])
        inner_w = int(pair["inner_w"])
        color = int(pair["color"])
        inner = out[fr + 1 : fr + 1 + inner_h, fc + 1 : fc + 1 + inner_w]
        if not np.any(inner == color):
            return None
        if not np.all((inner == BG) | (inner == color)):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
