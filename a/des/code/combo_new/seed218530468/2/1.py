from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
FRAME_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRECTIONS = ("up", "down", "left", "right")
CANVAS_SIZE = 30
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 500


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((int(r0), int(c)))
        coords.append((int(r1), int(c)))
    for r in range(r0 + 1, r1):
        coords.append((int(r), int(c0)))
        coords.append((int(r), int(c1)))
    return coords


def _direction_from_gap(
    gap: tuple[int, int], box: tuple[int, int, int, int]
) -> str | None:
    r0, c0, r1, c1 = box
    gr, gc = gap
    side_hits = int(gr == r0) + int(gr == r1) + int(gc == c0) + int(gc == c1)
    if side_hits != 1:
        return None
    if gr == r0:
        return "up"
    if gr == r1:
        return "down"
    if gc == c0:
        return "left"
    return "right"


def _seed_sort_key(
    seed: tuple[int, int, int], frame: dict[str, object]
) -> tuple[int, int, int]:
    r, c, _color = seed
    r0, c0, r1, c1 = frame["box"]
    direction = frame["direction"]
    if direction == "up":
        return (int(r - r0), int(c), int(r))
    if direction == "down":
        return (int(r1 - r), int(c), int(r))
    if direction == "left":
        return (int(c - c0), int(r), int(c))
    return (int(c1 - c), int(r), int(c))


def _node_sort_key(
    node: tuple[int, int], frame: dict[str, object]
) -> tuple[int, int, int]:
    r, c = node
    gap_r, gap_c = frame["gap"]
    direction = frame["direction"]
    if direction == "up":
        return (int(gap_r - r), int(c), int(r))
    if direction == "down":
        return (int(r - gap_r), int(c), int(r))
    if direction == "left":
        return (int(gap_c - c), int(r), int(c))
    return (int(c - gap_c), int(r), int(c))


def _side_span(seed: tuple[int, int, int], frame: dict[str, object]) -> int:
    r, c, _color = seed
    r0, c0, r1, c1 = frame["box"]
    direction = frame["direction"]
    if direction in ("up", "down"):
        return int(min(c - c0, c1 - c))
    return int(min(r - r0, r1 - r))


def _branch_cells(
    node: tuple[int, int], direction: str, span: int
) -> list[tuple[int, int]]:
    r, c = node
    cells = [(int(r), int(c))]
    if direction in ("up", "down"):
        for dc in range(1, span + 1):
            cells.append((int(r), int(c - dc)))
            cells.append((int(r), int(c + dc)))
    else:
        for dr in range(1, span + 1):
            cells.append((int(r - dr), int(c)))
            cells.append((int(r + dr), int(c)))
    return cells


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict[str, object]] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != BG):
        for coords in _components(g == int(color)):
            if coords.shape[0] < 7:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            box = (int(r0), int(c0), int(r1), int(c1))
            border = _border_coords(r0, c0, r1, c1)
            pts = {(int(r), int(c)) for r, c in coords}
            if len(pts) != len(border) - 1:
                continue
            if any((r, c) not in pts and int(g[r, c]) != BG for r, c in border):
                continue
            if any(not (r == r0 or r == r1 or c == c0 or c == c1) for r, c in pts):
                continue
            if int((g[r0 : r1 + 1, c0 : c1 + 1] == int(color)).sum()) != len(pts):
                continue
            missing = [(int(r), int(c)) for r, c in border if (int(r), int(c)) not in pts]
            if len(missing) != 1:
                continue
            gap = missing[0]
            direction = _direction_from_gap(gap, box)
            if direction is None:
                continue
            seeds: list[tuple[int, int, int]] = []
            for r in range(r0 + 1, r1):
                for c in range(c0 + 1, c1):
                    value = int(g[r, c])
                    if value not in (BG, int(color)):
                        seeds.append((int(r), int(c), int(value)))
            if not seeds:
                continue
            frames.append(
                {
                    "box": box,
                    "color": int(color),
                    "gap": gap,
                    "direction": direction,
                    "seeds": seeds,
                }
            )
    frames.sort(key=lambda frame: tuple(frame["box"]))
    return frames


def _collect_nodes(grid: np.ndarray, frame: dict[str, object]) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    gap_r, gap_c = frame["gap"]
    direction = frame["direction"]
    nodes: list[tuple[int, int]] = []
    if direction == "up":
        for r in range(gap_r - 1, -1, -1):
            value = int(g[r, gap_c])
            if value == GREY:
                nodes.append((int(r), int(gap_c)))
            elif value != BG:
                break
    elif direction == "down":
        for r in range(gap_r + 1, h):
            value = int(g[r, gap_c])
            if value == GREY:
                nodes.append((int(r), int(gap_c)))
            elif value != BG:
                break
    elif direction == "left":
        for c in range(gap_c - 1, -1, -1):
            value = int(g[gap_r, c])
            if value == GREY:
                nodes.append((int(gap_r), int(c)))
            elif value != BG:
                break
    else:
        for c in range(gap_c + 1, w):
            value = int(g[gap_r, c])
            if value == GREY:
                nodes.append((int(gap_r), int(c)))
            elif value != BG:
                break
    nodes.sort(key=lambda node: _node_sort_key(node, frame))
    return nodes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow colored side branches from grey nodes matched to seeds inside gapped rectangular frames.

    Input structure:
    - The background is black.
    - One or more same-colored hollow rectangles appear, each with exactly one missing border pixel on one side.
    - Each frame contains one or more colored seed pixels, and grey singleton nodes lie outside the frame on the ray facing the gap.

    Rule:
    - Detect each frame, its single gap, and the gap-facing direction.
    - Sort the seeds in that frame by distance to the gap side, nearest first, and pair them with the grey nodes ordered from the gap outward.
    - Recolor each paired node with the seed color, then draw rays in the perpendicular axis on both sides of that node.
    - Each ray length equals the seed's distance to the nearer side wall inside the frame; the frame and the original seeds stay unchanged.

    Key insight:
    - The gap side gives the outward axis, seed depth inside the frame gives the node matching order, and seed offset from the side walls gives the branch span.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()

    frames = _find_frames(g)
    assert frames, "expected at least one gapped frame"

    for frame in frames:
        seeds = sorted(frame["seeds"], key=lambda seed: _seed_sort_key(seed, frame))
        nodes = _collect_nodes(g, frame)
        assert len(nodes) == len(seeds), "grey node count must match the seed count"
        for seed, node in zip(seeds, nodes):
            color = int(seed[2])
            span = _side_span(seed, frame)
            for r, c in _branch_cells(node, frame["direction"], span):
                assert 0 <= r < out.shape[0] and 0 <= c < out.shape[1]
                out[r, c] = color

    return out


def _frame_pixels(
    outer_h: int, outer_w: int, gap: tuple[int, int], frame_color: int
) -> dict[tuple[int, int], int]:
    pixels: dict[tuple[int, int], int] = {}
    for r, c in _border_coords(0, 0, outer_h - 1, outer_w - 1):
        if (int(r), int(c)) != gap:
            pixels[(int(r), int(c))] = int(frame_color)
    return pixels


def _build_object(frame_color: int) -> dict[str, object]:
    direction = random.choice(DIRECTIONS)
    seed_count = random.randint(1, 4)

    if direction in ("up", "down"):
        outer_h = random.randint(max(5, seed_count + 2), 8)
        outer_w = random.randint(5, 8)
        gap = (0, random.randint(1, outer_w - 2)) if direction == "up" else (outer_h - 1, random.randint(1, outer_w - 2))
        seed_rows = random.sample(range(1, outer_h - 1), seed_count)
        seeds = [
            (int(r), int(random.randint(1, outer_w - 2)))
            for r in seed_rows
        ]
    else:
        outer_h = random.randint(5, 8)
        outer_w = random.randint(max(5, seed_count + 2), 8)
        gap = (random.randint(1, outer_h - 2), 0) if direction == "left" else (random.randint(1, outer_h - 2), outer_w - 1)
        seed_cols = random.sample(range(1, outer_w - 1), seed_count)
        seeds = [
            (int(random.randint(1, outer_h - 2)), int(c))
            for c in seed_cols
        ]

    seed_colors = random.sample([c for c in FRAME_COLORS if c != frame_color], seed_count)
    seed_records = [(int(r), int(c), int(color)) for (r, c), color in zip(seeds, seed_colors)]
    frame = {
        "box": (0, 0, outer_h - 1, outer_w - 1),
        "gap": gap,
        "direction": direction,
    }

    input_pixels = _frame_pixels(outer_h, outer_w, gap, frame_color)
    output_pixels = dict(input_pixels)
    for seed in seed_records:
        input_pixels[(int(seed[0]), int(seed[1]))] = int(seed[2])
        output_pixels[(int(seed[0]), int(seed[1]))] = int(seed[2])

    start = random.choice((1, 2))
    step = random.choice((2, 3))
    node_offsets = [int(start + i * step) for i in range(seed_count)]
    if direction == "up":
        nodes = [(-offset, gap[1]) for offset in node_offsets]
    elif direction == "down":
        nodes = [(outer_h - 1 + offset, gap[1]) for offset in node_offsets]
    elif direction == "left":
        nodes = [(gap[0], -offset) for offset in node_offsets]
    else:
        nodes = [(gap[0], outer_w - 1 + offset) for offset in node_offsets]

    for node in nodes:
        input_pixels[(int(node[0]), int(node[1]))] = GREY

    ordered_seeds = sorted(seed_records, key=lambda seed: _seed_sort_key(seed, frame))
    for node, seed in zip(nodes, ordered_seeds):
        span = _side_span(seed, frame)
        for cell in _branch_cells((int(node[0]), int(node[1])), direction, span):
            output_pixels[(int(cell[0]), int(cell[1]))] = int(seed[2])

    return {"input": input_pixels, "output": output_pixels}


def _dilate(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                expanded.add((int(r + dr), int(c + dc)))
    return expanded


def _place_objects(
    objects: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.full((CANVAS_SIZE, CANVAS_SIZE), BG, dtype=int)
    out = np.full((CANVAS_SIZE, CANVAS_SIZE), BG, dtype=int)
    reserved: set[tuple[int, int]] = set()

    ordered = sorted(
        objects,
        key=lambda obj: len(set(obj["output"].keys())),
        reverse=True,
    )
    for obj in ordered:
        out_coords = list(obj["output"].keys())
        min_r = min(r for r, _c in out_coords)
        max_r = max(r for r, _c in out_coords)
        min_c = min(c for _r, c in out_coords)
        max_c = max(c for _r, c in out_coords)
        row_lo = -int(min_r)
        row_hi = CANVAS_SIZE - 1 - int(max_r)
        col_lo = -int(min_c)
        col_hi = CANVAS_SIZE - 1 - int(max_c)
        if row_lo > row_hi or col_lo > col_hi:
            return None

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            dr = random.randint(row_lo, row_hi)
            dc = random.randint(col_lo, col_hi)
            shifted_out = {
                (int(r + dr), int(c + dc))
                for r, c in out_coords
            }
            if any(cell in reserved for cell in shifted_out):
                continue
            shifted_in = {
                (int(r + dr), int(c + dc)): int(color)
                for (r, c), color in obj["input"].items()
            }
            shifted_out_map = {
                (int(r + dr), int(c + dc)): int(color)
                for (r, c), color in obj["output"].items()
            }
            for (r, c), color in shifted_in.items():
                inp[r, c] = int(color)
            for (r, c), color in shifted_out_map.items():
                out[r, c] = int(color)
            reserved.update(
                {
                    cell
                    for cell in _dilate(shifted_out)
                    if 0 <= cell[0] < CANVAS_SIZE and 0 <= cell[1] < CANVAS_SIZE
                }
            )
            placed = True
            break
        if not placed:
            return None

    return inp, out


def _sample_margin(space: int) -> int:
    if space <= 0:
        return 0
    upper = min(3, int(space))
    lower = 0 if upper == 0 else int(random.random() < 0.25)
    return random.randint(lower, upper)


def _crop_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(inp != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top = _sample_margin(int(r0))
    left = _sample_margin(int(c0))
    bottom = _sample_margin(int(inp.shape[0] - 1 - r1))
    right = _sample_margin(int(inp.shape[1] - 1 - c1))
    rs = slice(int(r0 - top), int(r1 + bottom + 1))
    cs = slice(int(c0 - left), int(c1 + right + 1))
    return inp[rs, cs].copy(), out[rs, cs].copy()


def _attempt() -> np.ndarray | None:
    frame_color = random.choice(FRAME_COLORS)
    object_count = random.randint(1, 3)
    objects = [_build_object(frame_color) for _ in range(object_count)]
    placed = _place_objects(objects)
    if placed is None:
        return None

    inp, expected = _crop_pair(*placed)
    if inp.ndim != 2 or expected.ndim != 2:
        return None
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
