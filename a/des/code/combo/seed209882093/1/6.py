from __future__ import annotations

import random

import numpy as np


_NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_INSTRUCTION_ROTATIONS = {
    ((0, 0), (1, 0), (1, 1)): 3,  # opening right -> clockwise 90
    ((0, 1), (1, 0), (1, 1)): 2,  # opening up -> 180
    ((0, 0), (0, 1), (1, 1)): 1,  # opening left -> counterclockwise 90
}
_ALL_COLORS = tuple(range(1, 10))
_GEN_ATTEMPTS = 400


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
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _rect_contains(box: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    ir0, ic0, ir1, ic1 = inner
    return r0 <= ir0 and c0 <= ic0 and r1 >= ir1 and c1 >= ic1


def _frame_candidate(coords: np.ndarray, color: int) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    side_h = r1 - r0 + 1
    side_w = c1 - c0 + 1
    if side_h != side_w or side_h < 4:
        return None

    mask = np.zeros((side_h, side_w), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = np.zeros_like(mask)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.array_equal(mask, border):
        return None

    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "side": int(side_h),
    }


def _scan_frame_candidates(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    found: list[dict] = []
    seen = set()
    for r0 in range(h):
        for c0 in range(w):
            color = int(grid[r0, c0])
            if color == 0:
                continue
            max_side = min(h - r0, w - c0)
            for side in range(4, max_side + 1):
                r1 = r0 + side - 1
                c1 = c0 + side - 1
                if not (
                    np.all(grid[r0, c0 : c1 + 1] == color)
                    and np.all(grid[r1, c0 : c1 + 1] == color)
                    and np.all(grid[r0 : r1 + 1, c0] == color)
                    and np.all(grid[r0 : r1 + 1, c1] == color)
                ):
                    continue
                key = (color, r0, c0, r1, c1)
                if key in seen:
                    continue
                seen.add(key)
                found.append(
                    {
                        "color": color,
                        "bbox": (r0, c0, r1, c1),
                        "side": side,
                    }
                )
    return found


def _find_frames(grid: np.ndarray) -> list[dict]:
    candidates: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color == 0:
            continue
        for coords in _components(grid == color):
            cand = _frame_candidate(coords, color)
            if cand is not None:
                candidates.append(cand)
    candidates.extend(_scan_frame_candidates(grid))

    candidates.sort(key=lambda frame: (-frame["side"], frame["bbox"]))
    frames: list[dict] = []
    for cand in candidates:
        if any(cand["bbox"] == frame["bbox"] for frame in frames):
            continue
        if any(_rect_contains(frame["bbox"], cand["bbox"]) for frame in frames):
            continue
        frames.append(cand)
    frames.sort(key=lambda frame: frame["bbox"])
    return frames


def _instruction_candidate(coords: np.ndarray, color: int) -> dict | None:
    if int(coords.shape[0]) != 3:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 + 1 != 2 or c1 - c0 + 1 != 2:
        return None

    norm = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    k = _INSTRUCTION_ROTATIONS.get(norm)
    if k is None:
        return None
    return {
        "color": int(color),
        "coords": coords.copy(),
        "rotation": int(k),
        "bbox": (r0, c0, r1, c1),
    }


def _find_instructions(grid: np.ndarray, frames: list[dict]) -> dict[int, dict]:
    instructions: dict[int, dict] = {}
    for color in map(int, np.unique(grid)):
        if color == 0:
            continue
        for coords in _components(grid == color):
            cand = _instruction_candidate(coords, color)
            if cand is None:
                continue
            if any(_rect_contains(frame["bbox"], cand["bbox"]) for frame in frames):
                continue
            instructions.setdefault(int(color), cand)
    return instructions


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each square frame's interior according to the matched same-color L instruction, then erase the instructions.

    Input structure
    - On a black background, there are isolated objects of two kinds.
    - Instruction objects are monochrome 3-cell L shapes in a 2x2 box.
    - Target objects are square 1-pixel-thick monochrome frames containing an interior pattern.
    - Matching is by color: at most one instruction and one target frame share a color.

    Rule
    - Decode each L orientation as a rotation: opening right -> clockwise 90, opening up -> 180, opening left -> counterclockwise 90.
    - Rotate only the matched frame's interior pattern in place; leave unmatched frames unchanged.
    - Erase every instruction block to black.

    Key insight
    - The tiny L sprites are not content to move: they are direction-encoding instructions, and color links each instruction to its target frame.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frames = _find_frames(g)
    instructions = _find_instructions(g, frames)

    for inst in instructions.values():
        for r, c in inst["coords"]:
            out[int(r), int(c)] = 0

    for frame in frames:
        inst = instructions.get(int(frame["color"]))
        if inst is None:
            continue
        r0, c0, r1, c1 = frame["bbox"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        out[r0 + 1 : r1, c0 + 1 : c1] = np.rot90(inner, inst["rotation"])

    return out


def _random_pattern(size: int, palette: list[int], must_change: bool) -> np.ndarray | None:
    for _ in range(80):
        seed_h = random.randint(2, min(4, size))
        seed_w = random.randint(2, min(4, size))
        color_count = random.randint(2, min(4, len(palette)))
        colors = random.sample(palette, color_count)

        seed = np.asarray(random.choices(colors, k=seed_h * seed_w), dtype=int).reshape(seed_h, seed_w)
        if len(set(map(int, seed.flat))) < 2:
            continue

        row_idx = np.linspace(0, seed_h - 1, size).astype(int)
        col_idx = np.linspace(0, seed_w - 1, size).astype(int)
        pattern = seed[row_idx][:, col_idx].copy()

        accent_count = random.randint(1, min(3, size * size))
        for _ in range(accent_count):
            rr = random.randrange(size)
            cc = random.randrange(size)
            pattern[rr, cc] = random.choice(colors)

        if len(set(map(int, pattern.flat))) < 2:
            continue
        if must_change and any(np.array_equal(pattern, np.rot90(pattern, k)) for k in (1, 2, 3)):
            continue
        return pattern
    return None


def _draw_frame(grid: np.ndarray, top: int, left: int, border_color: int, inner: np.ndarray) -> None:
    side = int(inner.shape[0] + 2)
    grid[top : top + side, left] = border_color
    grid[top : top + side, left + side - 1] = border_color
    grid[top, left : left + side] = border_color
    grid[top + side - 1, left : left + side] = border_color
    grid[top + 1 : top + side - 1, left + 1 : left + side - 1] = inner


def _draw_instruction(grid: np.ndarray, top: int, left: int, color: int, rotation: int) -> None:
    cells = {
        3: ((0, 0), (1, 0), (1, 1)),
        2: ((0, 1), (1, 0), (1, 1)),
        1: ((0, 0), (0, 1), (1, 1)),
    }[int(rotation)]
    for dr, dc in cells:
        grid[top + dr, left + dc] = color


def _place_rectangles(shapes: list[tuple[int, int]], height: int, width: int) -> list[tuple[int, int]] | None:
    occupied = np.zeros((height, width), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(shapes)
    order = sorted(range(len(shapes)), key=lambda i: shapes[i][0] * shapes[i][1], reverse=True)

    for idx in order:
        h, w = shapes[idx]
        choices = [
            (r, c)
            for r in range(1, height - h)
            for c in range(1, width - w)
            if not occupied[r - 1 : r + h + 1, c - 1 : c + w + 1].any()
        ]
        if not choices:
            return None
        top, left = random.choice(choices)
        positions[idx] = (top, left)
        occupied[top : top + h, left : left + w] = True

    return [(int(r), int(c)) for r, c in positions if r is not None and c is not None]


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    frame_count = random.randint(2, 5)
    paired_count = random.randint(1, frame_count)
    border_colors = random.sample(list(_ALL_COLORS), frame_count)
    interior_palette = [color for color in _ALL_COLORS if color not in border_colors]
    if len(interior_palette) < 2:
        return None

    paired_ids = set(random.sample(range(frame_count), paired_count))
    frames: list[dict] = []
    rotations_by_color: dict[int, int] = {}
    for idx, border_color in enumerate(border_colors):
        inner_size = random.randint(2, 6)
        paired = idx in paired_ids
        inner = _random_pattern(inner_size, interior_palette, must_change=paired)
        if inner is None:
            return None
        frames.append(
            {
                "color": int(border_color),
                "inner": inner,
                "outer_side": int(inner_size + 2),
                "paired": bool(paired),
            }
        )
        if paired:
            rotations_by_color[int(border_color)] = random.choice((1, 2, 3))

    instructions = [
        {"color": int(frame["color"]), "rotation": int(rotations_by_color[int(frame["color"])])}
        for frame in frames
        if frame["paired"]
    ]

    shapes = [(frame["outer_side"], frame["outer_side"]) for frame in frames] + [(2, 2)] * len(instructions)
    padded_area = sum((h + 2) * (w + 2) for h, w in shapes)
    base = max(max(h for h, _ in shapes) + 3, int(np.ceil(np.sqrt(padded_area))) + random.randint(1, 4))

    for _ in range(20):
        height = random.randint(base, base + 6)
        width = random.randint(base, base + 6)
        positions = _place_rectangles(shapes, height, width)
        if positions is None:
            continue

        grid = np.zeros((height, width), dtype=int)
        expected = np.zeros((height, width), dtype=int)

        for frame, (top, left) in zip(frames, positions[:frame_count], strict=False):
            _draw_frame(grid, top, left, frame["color"], frame["inner"])
            rotated = np.rot90(frame["inner"], rotations_by_color[frame["color"]]) if frame["paired"] else frame["inner"]
            _draw_frame(expected, top, left, frame["color"], rotated)

        for inst, (top, left) in zip(instructions, positions[frame_count:], strict=False):
            _draw_instruction(grid, top, left, inst["color"], inst["rotation"])

        found_frames = _find_frames(grid)
        found_instructions = _find_instructions(grid, found_frames)
        if len(found_frames) != frame_count:
            continue
        if {frame["color"] for frame in found_frames} != {frame["color"] for frame in frames}:
            continue
        if set(found_instructions) != {inst["color"] for inst in instructions}:
            continue
        if not np.array_equal(main(grid), expected):
            continue
        return grid, expected

    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        built = _build_case()
        if built is not None:
            grid, _expected = built
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
