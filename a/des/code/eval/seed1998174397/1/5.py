from __future__ import annotations

import numpy as np


DIRS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
OPPOSITE: dict[str, str] = {
    "up": "down",
    "right": "left",
    "down": "up",
    "left": "right",
}
ROTATE_TO_UP: dict[str, int] = {
    "up": 0,
    "left": 3,
    "down": 2,
    "right": 1,
}
ALL_DIRS = ("up", "right", "down", "left")
PIECE_SIZE = 5
CHAIN_STEP = 5
MAX_GENERATION_ATTEMPTS = 400


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


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
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIRS.values():
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _canonical_piece_mask(notch_dir: str) -> np.ndarray:
    canvas = np.zeros((7, 7), dtype=bool)
    canvas[1:6, 1:6] = True
    if notch_dir == "up":
        canvas[1, 3] = False
        canvas[6, 3] = True
    elif notch_dir == "down":
        canvas[5, 3] = False
        canvas[0, 3] = True
    elif notch_dir == "left":
        canvas[3, 1] = False
        canvas[3, 6] = True
    elif notch_dir == "right":
        canvas[3, 5] = False
        canvas[3, 0] = True
    else:
        raise ValueError(f"unknown direction: {notch_dir}")

    coords = np.argwhere(canvas)
    r0, c0, r1, c1 = _bbox(coords)
    return canvas[r0 : r1 + 1, c0 : c1 + 1]


CANONICAL_MASKS = {direction: _canonical_piece_mask(direction) for direction in ALL_DIRS}


def _piece_patch(color: int, notch_dir: str, bg: int) -> np.ndarray:
    mask = CANONICAL_MASKS[notch_dir]
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(color)
    return patch


def _infer_notch_dir(mask: np.ndarray) -> str | None:
    for direction, canonical in CANONICAL_MASKS.items():
        if mask.shape == canonical.shape and np.array_equal(mask, canonical):
            return direction
    return None


def _extract_pieces(grid: np.ndarray, bg: int) -> list[dict]:
    pieces: list[dict] = []
    colors = [int(color) for color in np.unique(grid) if int(color) != int(bg)]
    for color in colors:
        for coords in _components(grid == int(color)):
            r0, c0, r1, c1 = _bbox(coords)
            local_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            local_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            notch_dir = _infer_notch_dir(local_mask)
            if notch_dir is None:
                continue
            pieces.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "center": coords.mean(axis=0),
                    "patch": _piece_patch(int(color), notch_dir, bg),
                    "notch": notch_dir,
                    "knob": OPPOSITE[notch_dir],
                }
            )
    return pieces


def _forward_key(delta: np.ndarray, direction: str) -> tuple[float, float, float] | None:
    dr, dc = float(delta[0]), float(delta[1])
    vr, vc = DIRS[direction]
    proj = dr * vr + dc * vc
    if proj <= 0:
        return None
    perp = dr * vc - dc * vr
    dist2 = dr * dr + dc * dc
    return float(proj), abs(float(perp)), float(dist2)


def _graph_order(pieces: list[dict]) -> list[int]:
    n = len(pieces)
    succ: list[int | None] = [None] * n
    pred: list[int | None] = [None] * n

    for i, piece in enumerate(pieces):
        best_succ: tuple[tuple[float, float, float], int] | None = None
        best_pred: tuple[tuple[float, float, float], int] | None = None
        for j, other in enumerate(pieces):
            if i == j:
                continue
            delta = other["center"] - piece["center"]

            if other["notch"] == OPPOSITE[piece["knob"]]:
                key = _forward_key(delta, piece["knob"])
                if key is not None and (best_succ is None or key < best_succ[0]):
                    best_succ = (key, j)

            if other["knob"] == OPPOSITE[piece["notch"]]:
                key = _forward_key(delta, piece["notch"])
                if key is not None and (best_pred is None or key < best_pred[0]):
                    best_pred = (key, j)

        succ[i] = None if best_succ is None else best_succ[1]
        pred[i] = None if best_pred is None else best_pred[1]

    starts = [i for i, prev in enumerate(pred) if prev is None]
    if len(starts) != 1:
        raise ValueError("failed to identify a unique chain start")

    order: list[int] = []
    seen: set[int] = set()
    cur = starts[0]
    while cur is not None:
        if cur in seen:
            raise ValueError("cycle detected in chain reconstruction")
        seen.add(cur)
        order.append(cur)
        cur = succ[cur]

    if len(order) != n:
        raise ValueError("chain reconstruction missed some pieces")
    return order


def _sorted_order(pieces: list[dict]) -> list[int] | None:
    if not pieces:
        return []
    knob_dirs = {piece["knob"] for piece in pieces}
    notch_dirs = {piece["notch"] for piece in pieces}
    if len(knob_dirs) != 1 or len(notch_dirs) != 1:
        return None

    step_dir = pieces[0]["knob"]
    axis = 1 if step_dir in {"left", "right"} else 0
    other = 1 - axis
    reverse = step_dir in {"left", "up"}
    return sorted(
        range(len(pieces)),
        key=lambda idx: (float(pieces[idx]["center"][axis]), float(pieces[idx]["center"][other])),
        reverse=reverse,
    )


def _recover_order(pieces: list[dict]) -> list[int]:
    try:
        return _graph_order(pieces)
    except ValueError:
        order = _sorted_order(pieces)
        if order is not None:
            return order
        raise


def _compose_chain(pieces: list[dict], order: list[int], bg: int) -> np.ndarray:
    positions: dict[int, tuple[int, int]] = {order[0]: (0, 0)}
    for current, nxt in zip(order, order[1:]):
        dr, dc = DIRS[pieces[current]["knob"]]
        pr, pc = positions[current]
        positions[nxt] = (pr + CHAIN_STEP * dr, pc + CHAIN_STEP * dc)

    min_r = min(pos[0] for pos in positions.values())
    min_c = min(pos[1] for pos in positions.values())
    max_r = max(positions[idx][0] + pieces[idx]["patch"].shape[0] for idx in order)
    max_c = max(positions[idx][1] + pieces[idx]["patch"].shape[1] for idx in order)

    out = np.full((max_r - min_r, max_c - min_c), int(bg), dtype=int)
    for idx in order:
        top = positions[idx][0] - min_r
        left = positions[idx][1] - min_c
        patch = pieces[idx]["patch"]
        mask = patch != int(bg)
        h, w = patch.shape
        target = out[top : top + h, left : left + w]
        target[mask] = patch[mask]
    return out


def _crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(grid != int(bg))
    if coords.size == 0:
        return np.zeros((0, 0), dtype=int)
    r0, c0, r1, c1 = _bbox(coords)
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the scattered puzzle pieces into one interlocked chain and rotate
    the whole chain so the first piece's free notch points upward.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    pieces = _extract_pieces(g, bg)
    if not pieces:
        return np.zeros((0, 0), dtype=int)

    order = _recover_order(pieces)
    assembled = _compose_chain(pieces, order, bg)
    rotated = np.rot90(assembled, k=ROTATE_TO_UP[pieces[order[0]]["notch"]])
    return _crop_non_bg(rotated, bg)


def _build_expected(colors: list[int], notch_dir: str, bg: int) -> np.ndarray:
    pieces = [
        {
            "patch": _piece_patch(color, notch_dir, bg),
            "notch": notch_dir,
            "knob": OPPOSITE[notch_dir],
        }
        for color in colors
    ]
    order = list(range(len(colors)))
    assembled = _compose_chain(pieces, order, bg)
    rotated = np.rot90(assembled, k=ROTATE_TO_UP[notch_dir])
    return _crop_non_bg(rotated, bg)


def _scatter_positions(
    n: int, step_dir: str, patch_shape: tuple[int, int], rng: np.random.Generator
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    patch_h, patch_w = map(int, patch_shape)
    gap_min = 2
    gap_max = 8

    axis_positions = [0]
    axis_extent = patch_w if step_dir in {"left", "right"} else patch_h
    for _ in range(1, n):
        gap = int(rng.integers(gap_min, gap_max + 1))
        axis_positions.append(axis_positions[-1] + axis_extent + gap)

    if step_dir in {"left", "up"}:
        max_axis = axis_positions[-1]
        axis_positions = [max_axis - pos for pos in axis_positions]

    side_span = int(rng.integers(max(4, n), max(8, 3 * n + 4)))
    orth_positions = [int(rng.integers(0, side_span + 1)) for _ in range(n)]

    positions: list[tuple[int, int]] = []
    for axis_pos, orth_pos in zip(axis_positions, orth_positions):
        if step_dir in {"left", "right"}:
            positions.append((orth_pos, axis_pos))
        else:
            positions.append((axis_pos, orth_pos))

    top_margin = int(rng.integers(1, 5))
    left_margin = int(rng.integers(1, 5))
    bottom_margin = int(rng.integers(1, 5))
    right_margin = int(rng.integers(1, 5))
    extra_h = int(rng.integers(0, 5))
    extra_w = int(rng.integers(0, 5))

    shifted = [(r + top_margin, c + left_margin) for r, c in positions]
    height = max(r + patch_h for r, _ in shifted) + bottom_margin + extra_h
    width = max(c + patch_w for _, c in shifted) + right_margin + extra_w
    return shifted, (height, width)


def _render_input(
    colors: list[int], notch_dir: str, positions: list[tuple[int, int]], shape: tuple[int, int], bg: int
) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for color, (top, left) in zip(colors, positions):
        patch = _piece_patch(color, notch_dir, bg)
        h, w = patch.shape
        mask = patch != int(bg)
        view = grid[top : top + h, left : left + w]
        view[mask] = patch[mask]
    return grid


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    bg = 0
    n = int(rng.integers(3, 8))
    notch_dir = str(rng.choice(ALL_DIRS))
    colors = [int(color) for color in rng.choice(np.arange(1, 10), size=n, replace=False)]
    patch_shape = CANONICAL_MASKS[notch_dir].shape
    positions, grid_shape = _scatter_positions(n, OPPOSITE[notch_dir], patch_shape, rng)
    grid = _render_input(colors, notch_dir, positions, grid_shape, bg)
    expected = _build_expected(colors, notch_dir, bg)

    pieces = _extract_pieces(grid, bg)
    if len(pieces) != n:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
