from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Iterable

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    if not path.is_file():
        return None
    try:
        spec = spec_from_file_location("_shared_arc_utils_fable_2_029_4", str(path))
        if spec is None or spec.loader is None:
            return None
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


_ARC_UTILS = _load_arc_utils()


if _ARC_UTILS is not None:
    Color = _ARC_UTILS.Color
    bbox_coords = _ARC_UTILS.bbox_coords
    components_mask = _ARC_UTILS.components_mask
else:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        GRAY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9
        BACKGROUND = 0

    def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
        mask = np.asarray(mask, dtype=bool)
        seen = np.zeros_like(mask, dtype=bool)
        if int(connectivity) == 4:
            steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            steps = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1),
            ]
        out: list[np.ndarray] = []
        height, width = mask.shape
        for r in range(height):
            for c in range(width):
                if not mask[r, c] or seen[r, c]:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                coords = []
                while stack:
                    rr, cc = stack.pop()
                    coords.append((rr, cc))
                    for dr, dc in steps:
                        nr = rr + dr
                        nc = cc + dc
                        if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                out.append(np.array(coords, dtype=int))
        return out


_RNG = np.random.default_rng()

BG = Color.BLACK
SEPARATOR = Color.GREY
INPUT_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]

WELL_HEIGHT_RANGE = (11, 16)
WELL_WIDTH_RANGE = (6, 8)
PIECE_COUNT_OPTIONS = (4, 5)
GENERATION_RETRIES = 600
PACK_RETRIES = 80

BASE_SHAPES = [
    np.array([[1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
]

ABSTRACT_TEMPLATES = [
    (
        6,
        [
            (np.array([[1, 1]], dtype=bool), 0),
            (np.array([[1, 1]], dtype=bool), 0),
            (np.array([[1, 1]], dtype=bool), 2),
            (np.array([[1, 1, 1, 1]], dtype=bool), 2),
            (np.array([[1], [1]], dtype=bool), 4),
        ],
    ),
    (
        6,
        [
            (np.array([[1, 0], [1, 1], [1, 0]], dtype=bool), 1),
            (np.array([[1, 1, 1, 1]], dtype=bool), 2),
            (np.array([[0, 1], [1, 1], [1, 0]], dtype=bool), 0),
            (np.array([[1, 1], [1, 0]], dtype=bool), 0),
            (np.array([[1, 1], [1, 0]], dtype=bool), 4),
        ],
    ),
    (
        6,
        [
            (np.array([[0, 1], [1, 1], [1, 0]], dtype=bool), 4),
            (np.array([[1, 0], [1, 1]], dtype=bool), 0),
            (np.array([[1, 0, 0], [1, 1, 1]], dtype=bool), 1),
            (np.array([[1, 1]], dtype=bool), 2),
            (np.array([[1, 0, 0], [1, 1, 1]], dtype=bool), 3),
        ],
    ),
    (
        8,
        [
            (np.array([[1, 1, 1], [0, 1, 0]], dtype=bool), 5),
            (np.array([[1, 1, 1], [1, 0, 0]], dtype=bool), 0),
            (np.array([[1, 1, 0], [0, 1, 1]], dtype=bool), 2),
            (np.array([[1, 0], [1, 1], [0, 1]], dtype=bool), 4),
            (np.array([[1, 1, 1], [0, 1, 0]], dtype=bool), 1),
        ],
    ),
]


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _shape_cells(mask: np.ndarray, r0: int = 0, c0: int = 0) -> np.ndarray:
    coords = np.argwhere(mask)
    return coords + np.array([int(r0), int(c0)])


def _fits(board: np.ndarray, mask: np.ndarray, r0: int, c0: int, bg: int) -> bool:
    h, w = mask.shape
    if r0 < 0 or c0 < 0 or r0 + h > board.shape[0] or c0 + w > board.shape[1]:
        return False
    region = board[r0 : r0 + h, c0 : c0 + w]
    return not np.any(region[mask] != int(bg))


def _stamp(board: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    region = board[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
    region[mask] = int(color)


def _clear_full_rows(board: np.ndarray, bg: int) -> tuple[np.ndarray, int, int]:
    total_lines = 0
    rounds = 0
    current = board.copy()
    while True:
        full = np.all(current != int(bg), axis=1)
        if not np.any(full):
            return current, total_lines, rounds
        rounds += 1
        total_lines += int(np.count_nonzero(full))
        kept = current[~full]
        next_board = np.full_like(current, int(bg))
        next_board[-kept.shape[0] :] = kept
        current = next_board


def _simulate(
    pieces: dict[int, dict],
    order: list[int],
    well_shape: tuple[int, int],
    *,
    bg: int = BG,
    clear_mode: str = "step",
    use_input_rows: bool = False,
) -> tuple[np.ndarray, dict]:
    board = np.full(well_shape, int(bg), dtype=int)
    landing_rows: dict[int, int] = {}
    clear_events: list[dict] = []

    for piece_index, color in enumerate(order):
        piece = pieces[int(color)]
        start_r = int(piece["start_r"]) if use_input_rows else 0
        if not _fits(board, piece["mask"], start_r, int(piece["c0"]), bg):
            raise AssertionError("piece starts inside settled terrain")

        r = start_r
        while _fits(board, piece["mask"], r + 1, int(piece["c0"]), bg):
            r += 1

        landing_rows[int(color)] = int(r)
        _stamp(board, piece["mask"], r, int(piece["c0"]), int(color))

        if clear_mode == "step":
            board, cleared_lines, cleared_rounds = _clear_full_rows(board, bg)
            if cleared_lines:
                clear_events.append(
                    {
                        "piece_index": int(piece_index),
                        "color": int(color),
                        "lines": int(cleared_lines),
                        "rounds": int(cleared_rounds),
                    }
                )
        elif clear_mode not in {"end", "none"}:
            raise ValueError(f"unknown clear_mode: {clear_mode}")

    if clear_mode == "end":
        board, cleared_lines, cleared_rounds = _clear_full_rows(board, bg)
        if cleared_lines:
            clear_events.append(
                {
                    "piece_index": len(order),
                    "color": -1,
                    "lines": int(cleared_lines),
                    "rounds": int(cleared_rounds),
                }
            )

    return board, {
        "landing_rows": landing_rows,
        "clear_events": clear_events,
        "total_lines": int(sum(event["lines"] for event in clear_events)),
        "max_rounds": int(max((event["rounds"] for event in clear_events), default=0)),
    }


def _parse_input(grid: np.ndarray) -> tuple[int, int, list[int], dict[int, dict], tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3 and g.shape[1] >= 4

    top_row = g[0]
    sep_row = g[1]
    assert np.all(sep_row == sep_row[0]), "expected a full separator row"

    separator = int(sep_row[0])
    well = g[2:]
    bg = _dominant_color(well)
    assert separator != bg, "separator must differ from the well background"

    schedule = [int(value) for value in top_row if int(value) != bg]
    assert schedule and len(schedule) == len(set(schedule)), "schedule colors must be unique"
    assert separator not in schedule, "separator color cannot appear in the schedule"
    assert set(map(int, top_row)).issubset({bg, *schedule}), "top row can only contain background or schedule colors"
    assert separator not in set(map(int, np.unique(well))), "separator cannot appear in the well"

    non_bg_components = components_mask(well != bg, connectivity=4)
    assert len(non_bg_components) == len(schedule), "expected one floating object per schedule color"

    pieces: dict[int, dict] = {}
    for coords in non_bg_components:
        colors = {int(well[r, c]) for r, c in coords}
        assert len(colors) == 1, "each floating object must be monochromatic"
        color = colors.pop()
        assert color in schedule and color not in pieces, "well colors must match the schedule one-to-one"

        r0, c0, r1, c1 = bbox_coords(coords)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        shifted = coords - np.array([r0, c0])
        mask[shifted[:, 0], shifted[:, 1]] = True
        pieces[color] = {
            "color": int(color),
            "mask": mask,
            "c0": int(c0),
            "start_r": int(r0),
        }

    assert sorted(pieces) == sorted(schedule)
    return int(bg), separator, schedule, pieces, tuple(map(int, well.shape))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the scheduled floating objects one by one, clearing every full well row immediately after each landing.

    Input structure:
    - The top row contains a left-to-right sequence of colored schedule pixels on the background.
    - The second row is one full gray separator row.
    - Below the separator, each schedule color appears exactly once as a single floating monochrome object inside the well.

    Rule:
    - Read the schedule colors from left to right.
    - For each scheduled color, take its matching floating object, keep its shape and absolute columns fixed, and move it straight down until one more step would hit the bottom or an already settled pixel.
    - After that object lands, repeatedly clear every completely filled well row and let everything above fall down until no full row remains; then continue with the next scheduled object.

    Key insight:
    - The top row is an execution timetable, not a legend: order changes the terrain, and row clears rewrite that terrain again before the next object falls.
    """
    g = np.asarray(grid, dtype=int)
    bg, _separator, schedule, pieces, well_shape = _parse_input(g)
    settled, _stats = _simulate(pieces, schedule, well_shape, bg=bg, clear_mode="step", use_input_rows=True)

    out = np.full_like(g, int(bg))
    out[0] = g[0]
    out[1] = g[1]
    out[2:] = settled
    return out


def _random_shape(max_width: int) -> np.ndarray:
    for _ in range(50):
        mask = BASE_SHAPES[int(_RNG.integers(len(BASE_SHAPES)))]
        transformed = mask.copy()
        if bool(_RNG.integers(2)):
            transformed = np.fliplr(transformed)
        transformed = np.rot90(transformed, int(_RNG.integers(4)))
        transformed = _normalize_mask(transformed)
        if transformed.shape[1] <= max_width:
            return transformed
    raise RuntimeError("failed to sample a fitting shape")


def _piece_span(piece: dict) -> tuple[int, int]:
    left = int(piece["c0"])
    right = left + int(piece["mask"].shape[1]) - 1
    return left, right


def _overlap_degree(pieces: dict[int, dict]) -> dict[int, int]:
    degrees = {int(color): 0 for color in pieces}
    colors = list(pieces)
    for i, color_a in enumerate(colors):
        a0, a1 = _piece_span(pieces[color_a])
        for color_b in colors[i + 1 :]:
            b0, b1 = _piece_span(pieces[color_b])
            if max(a0, b0) <= min(a1, b1):
                degrees[color_a] += 1
                degrees[color_b] += 1
    return degrees


def _order_by_left(pieces: dict[int, dict]) -> list[int]:
    def key(color: int) -> tuple[int, int, int, int]:
        piece = pieces[int(color)]
        left, right = _piece_span(piece)
        return (int(left), int(right), int(piece["mask"].shape[0]), int(piece["start_r"]))

    return sorted((int(color) for color in pieces), key=key)


def _order_by_top(pieces: dict[int, dict]) -> list[int]:
    def key(color: int) -> tuple[int, int, int, int]:
        piece = pieces[int(color)]
        left, right = _piece_span(piece)
        return (int(piece["start_r"]), int(left), int(right), int(piece["mask"].shape[0]))

    return sorted((int(color) for color in pieces), key=key)


def _simulate_safe(
    pieces: dict[int, dict],
    order: list[int],
    well_shape: tuple[int, int],
    *,
    bg: int,
    clear_mode: str,
    use_input_rows: bool,
) -> np.ndarray | None:
    try:
        board, _stats = _simulate(
            pieces,
            order,
            well_shape,
            bg=bg,
            clear_mode=clear_mode,
            use_input_rows=use_input_rows,
        )
        return board
    except AssertionError:
        return None


def _touching_cells(coords: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
    cells = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                cells.add((int(r) + dr, int(c) + dc))
    return cells


def _pack_start_rows(
    pieces: dict[int, dict],
    landing_rows: dict[int, int],
    well_shape: tuple[int, int],
    schedule: list[int],
) -> dict[int, int] | None:
    colors = sorted(schedule, key=lambda color: landing_rows[int(color)])

    def try_with_cap(cap: int) -> dict[int, int] | None:
        assignment: dict[int, int] = {}
        blocked: set[tuple[int, int]] = set()

        def dfs(index: int) -> bool:
            if index == len(colors):
                return True
            color = int(colors[index])
            piece = pieces[color]
            max_start = min(int(landing_rows[color]) - 1, cap)
            if max_start < 0:
                return False

            rows = list(range(max_start + 1))
            _RNG.shuffle(rows)
            for start_r in rows:
                coords = [tuple(map(int, rc)) for rc in _shape_cells(piece["mask"], start_r, int(piece["c0"]))]
                if any(rc in blocked for rc in coords):
                    continue
                if any(not (0 <= rc[0] < well_shape[0] and 0 <= rc[1] < well_shape[1]) for rc in coords):
                    continue

                assignment[color] = int(start_r)
                prev_blocked = blocked.copy()
                blocked.update(_touching_cells(coords))
                if dfs(index + 1):
                    return True
                blocked.clear()
                blocked.update(prev_blocked)
                del assignment[color]
            return False

        if dfs(0):
            return assignment
        return None

    for cap in (max(2, well_shape[0] // 2), well_shape[0] - 1):
        for _ in range(PACK_RETRIES):
            packed = try_with_cap(cap)
            if packed is not None:
                return packed
    return None


def _instantiate_template(
    template: tuple[int, list[tuple[np.ndarray, int]]],
    schedule: list[int],
) -> tuple[dict[int, dict], tuple[int, int]]:
    well_w, specs = template
    mirror = bool(_RNG.integers(2))
    pieces: dict[int, dict] = {}

    for color, (base_mask, base_c0) in zip(schedule, specs):
        mask = base_mask.copy()
        c0 = int(base_c0)
        if mirror:
            mask = np.fliplr(mask)
            c0 = int(well_w - mask.shape[1] - c0)
        pieces[int(color)] = {
            "color": int(color),
            "mask": mask,
            "c0": int(c0),
            "start_r": 0,
        }

    well_h = int(_RNG.integers(WELL_HEIGHT_RANGE[0], WELL_HEIGHT_RANGE[1] + 1))
    return pieces, (well_h, int(well_w))


def _build_input_grid(
    schedule: list[int],
    pieces: dict[int, dict],
    well_shape: tuple[int, int],
) -> np.ndarray:
    well_h, well_w = map(int, well_shape)
    grid = np.full((well_h + 2, well_w), BG, dtype=int)

    slot_cols = np.sort(_RNG.choice(well_w, size=len(schedule), replace=False))
    for col, color in zip(slot_cols, schedule):
        grid[0, int(col)] = int(color)
    grid[1, :] = SEPARATOR

    well = grid[2:]
    for color, piece in pieces.items():
        _stamp(well, piece["mask"], int(piece["start_r"]), int(piece["c0"]), int(color))
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        bg, _separator, schedule, pieces, well_shape = _parse_input(grid)
        target = main(grid)[2:]
        simulated, stats = _simulate(pieces, schedule, well_shape, bg=bg, clear_mode="step", use_input_rows=True)
        if not np.array_equal(target, simulated):
            return False
        if stats["total_lines"] < 1:
            return False
        if not stats["clear_events"] or stats["clear_events"][0]["piece_index"] >= len(schedule) - 1:
            return False
        if any(int(pieces[color]["start_r"]) >= int(stats["landing_rows"][color]) for color in schedule):
            return False

        overlap_degree = _overlap_degree(pieces)
        if any(degree == 0 for degree in overlap_degree.values()):
            return False

        no_clear = _simulate_safe(
            pieces,
            schedule,
            well_shape,
            bg=bg,
            clear_mode="none",
            use_input_rows=True,
        )
        if no_clear is None or np.array_equal(no_clear, target):
            return False

        clear_at_end = _simulate_safe(
            pieces,
            schedule,
            well_shape,
            bg=bg,
            clear_mode="end",
            use_input_rows=True,
        )
        if clear_at_end is None or np.array_equal(clear_at_end, target):
            return False

        left_order = _order_by_left(pieces)
        if left_order == schedule:
            return False
        left_variant = _simulate_safe(
            pieces,
            left_order,
            well_shape,
            bg=bg,
            clear_mode="step",
            use_input_rows=True,
        )
        if left_variant is not None and np.array_equal(left_variant, target):
            return False

        top_order = _order_by_top(pieces)
        if top_order == schedule:
            return False
        top_variant = _simulate_safe(
            pieces,
            top_order,
            well_shape,
            bg=bg,
            clear_mode="step",
            use_input_rows=True,
        )
        if top_variant is not None and np.array_equal(top_variant, target):
            return False

        reverse_order = list(reversed(schedule))
        if reverse_order == schedule:
            return False
        reverse_variant = _simulate_safe(
            pieces,
            reverse_order,
            well_shape,
            bg=bg,
            clear_mode="step",
            use_input_rows=True,
        )
        if reverse_variant is not None and np.array_equal(reverse_variant, target):
            return False

        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        template = ABSTRACT_TEMPLATES[int(_RNG.integers(len(ABSTRACT_TEMPLATES)))]
        schedule = _RNG.choice(INPUT_COLORS, size=len(template[1]), replace=False).astype(int).tolist()
        pieces, well_shape = _instantiate_template(template, schedule)

        try:
            _abstract_board, stats = _simulate(
                pieces,
                schedule,
                well_shape,
                bg=BG,
                clear_mode="step",
                use_input_rows=False,
            )
        except AssertionError:
            continue

        packed_rows = _pack_start_rows(pieces, stats["landing_rows"], well_shape, schedule)
        if packed_rows is None:
            continue

        input_pieces = {
            color: {
                "color": piece["color"],
                "mask": piece["mask"].copy(),
                "c0": int(piece["c0"]),
                "start_r": int(packed_rows[color]),
            }
            for color, piece in pieces.items()
        }
        if _order_by_top(input_pieces) == schedule:
            continue

        grid = _build_input_grid(schedule, input_pieces, well_shape)
        if _semantic_checks(grid):
            return grid

    raise RuntimeError("failed to generate a non-degenerate timetable-drop ARC input")
