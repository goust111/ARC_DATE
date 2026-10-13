from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
COLOR_ORDER = (1, 2, 3, 4, 6, 7, 8, 9)  # Blue, Red, Green, Yellow, Magenta, Orange, Azure, Maroon
DIRECTIONS = ("up", "right", "down", "left")
DIR_TO_ROT = {"up": 0, "right": 1, "down": 2, "left": 3}

MAX_GENERATION_ATTEMPTS = 300
GRID_SIZE_RANGE = (18, 39)
ENVELOPE_COUNT_RANGE = (2, 6)
PAPER_DIM_RANGE = (1, 6)
OPEN_GAP_RANGE = (1, 3)
CLOSED_GAP_RANGE = (2, 5)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    mask = grid == color
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(pts, dtype=int))
    return components


def _frame_cells(
    height: int,
    width: int,
    missing_side: str,
    row_offset: int = 0,
    col_offset: int = 0,
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    if missing_side != "top":
        for c in range(width):
            cells.add((row_offset, col_offset + c))
    if missing_side != "bottom":
        for c in range(width):
            cells.add((row_offset + height - 1, col_offset + c))
    if missing_side != "left":
        for r in range(height):
            cells.add((row_offset + r, col_offset))
    if missing_side != "right":
        for r in range(height):
            cells.add((row_offset + r, col_offset + width - 1))
    return cells


def _detect_envelope_component(coords: np.ndarray) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 3 or width < 3:
        return None

    pts = {(int(r), int(c)) for r, c in coords}
    matches: list[tuple[str, str]] = []
    for missing_side, opening in (
        ("top", "up"),
        ("right", "right"),
        ("bottom", "down"),
        ("left", "left"),
    ):
        expected = _frame_cells(height, width, missing_side, r0, c0)
        if pts == expected:
            matches.append((missing_side, opening))

    if len(matches) != 1:
        return None

    missing_side, opening = matches[0]
    return {
        "bbox": (r0, c0, r1, c1),
        "opening": opening,
        "missing_side": missing_side,
    }


def _crop_nonzero_pattern(arr: np.ndarray, excluded_color: int) -> np.ndarray:
    mask = (arr != BLACK) & (arr != excluded_color)
    assert bool(mask.any()), "envelope interior must contain a paper pattern"
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _collect_envelopes(grid: np.ndarray) -> list[dict[str, object]]:
    envelopes: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BLACK):
        for comp in _components_of_color(grid, color):
            info = _detect_envelope_component(comp)
            if info is None:
                continue
            r0, c0, r1, c1 = info["bbox"]
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            paper = _crop_nonzero_pattern(sub, color)
            envelopes.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "opening": str(info["opening"]),
                    "paper": paper,
                }
            )
    return envelopes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Disassemble open rectangular envelopes, normalize their interior paper orientation,
    then concatenate papers by a fixed visible color order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    envelopes = _collect_envelopes(g)
    assert envelopes, "expected at least one envelope"

    colors = [int(env["color"]) for env in envelopes]
    assert len(colors) == len(set(colors)), "envelope border colors must be unique"
    assert all(color in COLOR_ORDER for color in colors), "unexpected envelope border color"

    normalized: list[dict[str, object]] = []
    for env in envelopes:
        opening = str(env["opening"])
        paper = np.asarray(env["paper"], dtype=int)
        normalized.append(
            {
                "color": int(env["color"]),
                "paper": np.rot90(paper, k=DIR_TO_ROT[opening]),
            }
        )

    normalized.sort(key=lambda env: COLOR_ORDER.index(int(env["color"])))

    out_h = max(int(env["paper"].shape[0]) for env in normalized)
    out_w = sum(int(env["paper"].shape[1]) for env in normalized) + len(normalized) - 1
    out = np.full((out_h, out_w), BLACK, dtype=int)

    col = 0
    for env in normalized:
        paper = np.asarray(env["paper"], dtype=int)
        h, w = map(int, paper.shape)
        out[:h, col : col + w] = paper
        col += w + 1

    return out


def _boxes_too_close(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    separated = ar1 + 2 < br0 or br1 + 2 < ar0 or ac1 + 2 < bc0 or bc1 + 2 < ac0
    return not separated


def _staircase_seed(rng: np.random.Generator, height: int, width: int) -> set[tuple[int, int]]:
    if height == 1:
        return {(0, c) for c in range(width)}
    if width == 1:
        return {(r, 0) for r in range(height)}

    if bool(rng.integers(2)):
        r, c = 0, 0
        steps = [(1, 0)] * (height - 1) + [(0, 1)] * (width - 1)
    else:
        r, c = 0, width - 1
        steps = [(1, 0)] * (height - 1) + [(0, -1)] * (width - 1)
    steps = [steps[i] for i in rng.permutation(len(steps))]

    cells = {(r, c)}
    for dr, dc in steps:
        r += dr
        c += dc
        cells.add((r, c))
    return cells


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                cells.add((nr, nc))
    return list(cells)


def _looks_like_open_frame(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 3 or width < 3:
        return False
    shifted = {(int(r - r0), int(c - c0)) for r, c in coords}
    for missing_side in ("top", "right", "bottom", "left"):
        if shifted == _frame_cells(height, width, missing_side):
            return True
    return False


def _make_support_mask(
    rng: np.random.Generator,
    height: int,
    width: int,
) -> np.ndarray | None:
    seed = _staircase_seed(rng, height, width)
    min_area = len(seed)
    for _ in range(80):
        target = int(rng.integers(min_area, height * width + 1))
        mask = np.zeros((height, width), dtype=bool)
        for r, c in seed:
            mask[r, c] = True
        while int(mask.sum()) < target:
            fringe = _frontier(mask)
            if not fringe:
                break
            fr, fc = fringe[int(rng.integers(len(fringe)))]
            mask[fr, fc] = True
        if int(mask.sum()) != target:
            continue
        if _looks_like_open_frame(mask):
            continue
        return mask
    return None


def _colorize_patch(
    rng: np.random.Generator,
    mask: np.ndarray,
    available_colors: list[int],
    rotation_k: int,
) -> np.ndarray | None:
    occupied = np.argwhere(mask)
    if occupied.size == 0:
        return None

    max_colors = min(3, len(available_colors), int(occupied.shape[0]))
    if max_colors <= 0:
        return None

    for _ in range(60):
        count = int(rng.integers(1, max_colors + 1))
        chosen = [int(available_colors[i]) for i in rng.choice(len(available_colors), size=count, replace=False)]
        patch = np.zeros(mask.shape, dtype=int)
        order = rng.permutation(len(occupied))

        for idx, color in enumerate(chosen):
            r, c = occupied[int(order[idx])]
            patch[int(r), int(c)] = int(color)
        for idx in order[count:]:
            r, c = occupied[int(idx)]
            patch[int(r), int(c)] = int(chosen[int(rng.integers(count))])

        if rotation_k and np.array_equal(patch, np.rot90(patch, k=rotation_k)):
            continue
        return patch
    return None


def _opening_gap(rng: np.random.Generator, opening: str, side: str) -> int:
    if opening == side:
        return int(rng.integers(*OPEN_GAP_RANGE))
    return int(rng.integers(*CLOSED_GAP_RANGE))


def _make_envelope(
    rng: np.random.Generator,
    border_color: int,
    opening: str,
    available_paper_colors: list[int],
) -> np.ndarray | None:
    for _ in range(100):
        paper_h = int(rng.integers(*PAPER_DIM_RANGE))
        paper_w = int(rng.integers(*PAPER_DIM_RANGE))
        if paper_h == 1 and paper_w == 1:
            continue

        support = _make_support_mask(rng, paper_h, paper_w)
        if support is None:
            continue

        rotation_k = DIR_TO_ROT[opening]
        patch = _colorize_patch(rng, support, available_paper_colors, rotation_k)
        if patch is None:
            continue

        gap_top = _opening_gap(rng, opening, "up")
        gap_right = _opening_gap(rng, opening, "right")
        gap_bottom = _opening_gap(rng, opening, "down")
        gap_left = _opening_gap(rng, opening, "left")

        height = int(gap_top + paper_h + gap_bottom)
        width = int(gap_left + paper_w + gap_right)
        env = np.zeros((height, width), dtype=int)

        if opening != "up":
            env[0, :] = int(border_color)
        if opening != "down":
            env[-1, :] = int(border_color)
        if opening != "left":
            env[:, 0] = int(border_color)
        if opening != "right":
            env[:, -1] = int(border_color)

        window = env[gap_top : gap_top + paper_h, gap_left : gap_left + paper_w]
        window[patch != BLACK] = patch[patch != BLACK]
        return env
    return None


def _sample_directions(rng: np.random.Generator, count: int) -> list[str]:
    permuted = [DIRECTIONS[int(i)] for i in rng.permutation(len(DIRECTIONS))]
    if count <= len(DIRECTIONS):
        return permuted[:count]
    return permuted + [DIRECTIONS[int(rng.integers(len(DIRECTIONS)))]]


def _place_patch(
    rng: np.random.Generator,
    grid: np.ndarray,
    patch: np.ndarray,
    placed_boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    gh, gw = grid.shape
    ph, pw = patch.shape
    candidates: list[tuple[int, int]] = []
    for r in range(gh - ph + 1):
        for c in range(gw - pw + 1):
            box = (int(r), int(c), int(r + ph - 1), int(c + pw - 1))
            if any(_boxes_too_close(box, other) for other in placed_boxes):
                continue
            candidates.append((int(r), int(c)))
    if not candidates:
        return None

    r, c = candidates[int(rng.integers(len(candidates)))]
    box = (int(r), int(c), int(r + ph - 1), int(c + pw - 1))
    window = grid[r : r + ph, c : c + pw]
    window[patch != BLACK] = patch[patch != BLACK]
    return box


def _position_order(boxes_by_color: dict[int, tuple[int, int, int, int]]) -> list[int]:
    items = list(boxes_by_color.items())
    items.sort(key=lambda item: (item[1][1], item[1][0], item[0]))
    return [int(color) for color, _box in items]


def _row_major_order(boxes_by_color: dict[int, tuple[int, int, int, int]]) -> list[int]:
    items = list(boxes_by_color.items())
    items.sort(key=lambda item: (item[1][0], item[1][1], item[0]))
    return [int(color) for color, _box in items]


def _generate_candidate(rng: np.random.Generator) -> np.ndarray | None:
    envelope_count = int(rng.integers(*ENVELOPE_COUNT_RANGE))
    chosen_colors = [int(COLOR_ORDER[i]) for i in rng.choice(len(COLOR_ORDER), size=envelope_count, replace=False)]
    directions = _sample_directions(rng, envelope_count)
    shuffled_indices = list(rng.permutation(envelope_count))

    paper_colors = [c for c in range(1, 10) if c not in chosen_colors]
    if not paper_colors:
        return None

    patches: dict[int, np.ndarray] = {}
    openings: dict[int, str] = {}
    for idx in range(envelope_count):
        color = int(chosen_colors[idx])
        opening = str(directions[idx])
        patch = _make_envelope(rng, color, opening, paper_colors)
        if patch is None:
            return None
        patches[color] = patch
        openings[color] = opening

    max_h = max(int(patch.shape[0]) for patch in patches.values())
    max_w = max(int(patch.shape[1]) for patch in patches.values())
    min_grid = max(GRID_SIZE_RANGE[0], max_h + 2, max_w + 2)
    grid_h = int(rng.integers(min_grid, GRID_SIZE_RANGE[1]))
    grid_w = int(rng.integers(min_grid, GRID_SIZE_RANGE[1]))
    grid = np.zeros((grid_h, grid_w), dtype=int)

    placed_boxes: list[tuple[int, int, int, int]] = []
    boxes_by_color: dict[int, tuple[int, int, int, int]] = {}
    for idx in shuffled_indices:
        color = int(chosen_colors[idx])
        box = _place_patch(rng, grid, patches[color], placed_boxes)
        if box is None:
            return None
        placed_boxes.append(box)
        boxes_by_color[color] = box

    color_order = sorted(chosen_colors, key=COLOR_ORDER.index)
    if _position_order(boxes_by_color) == color_order:
        return None
    if _row_major_order(boxes_by_color) == color_order:
        return None

    envelopes = _collect_envelopes(grid)
    if len(envelopes) != envelope_count:
        return None

    detected_colors = sorted(int(env["color"]) for env in envelopes)
    if detected_colors != sorted(chosen_colors):
        return None

    for env in envelopes:
        color = int(env["color"])
        if str(env["opening"]) != openings[color]:
            return None

    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _generate_candidate(rng)
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid envelope puzzle input")
