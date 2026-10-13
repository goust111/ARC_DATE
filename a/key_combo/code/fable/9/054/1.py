import random

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9


INPUT_COLORS = [
    Color.BLUE,
    Color.RED,
    Color.YELLOW,
    Color.GREY,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]

MAX_GENERATION_ATTEMPTS = 600
_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()

_GEN_BRANCHES = (
    {"height": (10, 22), "width": (4, 7), "required_bounces": (1, 1), "require_multi_hit": True},
    {"height": (8, 18), "width": (6, 12), "required_bounces": (1, 0), "require_multi_hit": False},
    {"height": (12, 24), "width": (4, 8), "required_bounces": (2, 1), "require_multi_hit": True},
    {"height": (9, 19), "width": (8, 15), "required_bounces": (0, 1), "require_multi_hit": False},
)


def _bounce_path_info(shape: tuple[int, int], start: tuple[int, int]) -> tuple[list[tuple[int, int]], int]:
    """Trace one hidden up-right bouncing path, including its start cell."""
    height, width = map(int, shape)
    r, c = map(int, start)
    dc = 1
    bounces = 0
    path: list[tuple[int, int]] = []

    while 0 <= r < height and 0 <= c < width:
        path.append((r, c))
        nr = r - 1
        nc = c + dc
        if nr < 0:
            break
        if nc < 0 or nc >= width:
            dc = -dc
            bounces += 1
            nc = c + dc
        r, c = nr, nc

    return path, bounces


def _extract_starts(grid: np.ndarray) -> list[tuple[int, int]]:
    coords = [tuple(map(int, coord)) for coord in np.argwhere(grid != Color.BLACK)]
    assert len(coords) == 2, "expected exactly two isolated foreground pixels"
    colors = [int(grid[r, c]) for r, c in coords]
    assert len(set(colors)) == 2, "the two start pixels must have different colors"
    assert all(r > 0 for r, _c in coords), "start pixels must not already have exited from the top"
    assert ((coords[0][0] + coords[0][1]) - (coords[1][0] + coords[1][1])) % 2 == 0, "start parities must match"
    return coords


def _paint(shape: tuple[int, int], coords: set[tuple[int, int]], color: int) -> np.ndarray:
    out = np.full(shape, Color.BLACK, dtype=int)
    for r, c in coords:
        out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark only the shared cells of two hidden bouncing up-right rays.

    Input structure:
    - The grid is black except for exactly two isolated pixels of different colors.
    - Each pixel is a start generator for an unseen diagonal path.
    - The two starts have matching row-plus-column parity, and the side borders are reflecting walls.

    Rule:
    - From each start, move diagonally up-right one cell per step.
    - Whenever the path would leave through the left or right side, reverse only the horizontal direction and continue upward.
    - Stop each path when its next step would leave above the top edge, then output a black grid that colors only the cells visited by both paths in green, excluding the two start cells themselves.

    Key insight:
    - Nothing visible in the input is a drawn line; the answer is the intersection of two independently reconstructed bounce trajectories.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 2 and width >= 2, "grid must be at least 2x2"

    start_a, start_b = _extract_starts(g)
    path_a, _ = _bounce_path_info(g.shape, start_a)
    path_b, _ = _bounce_path_info(g.shape, start_b)

    shared = set(path_a) & set(path_b)
    shared.discard(start_a)
    shared.discard(start_b)
    return _paint(g.shape, shared, Color.GREEN)


def _satisfies_bounce_profile(bounces: tuple[int, int], required: tuple[int, int]) -> bool:
    actual = sorted(map(int, bounces))
    needed = sorted(map(int, required))
    return all(a >= b for a, b in zip(actual, needed))


def _gen_candidate_pair(shape: tuple[int, int], branch: dict) -> tuple[np.ndarray, np.ndarray] | None:
    height, width = map(int, shape)
    infos = []
    for r in range(2, height):
        for c in range(width):
            path, bounce_count = _bounce_path_info(shape, (r, c))
            infos.append(
                {
                    "start": (r, c),
                    "path": path,
                    "path_set": set(path),
                    "bounces": bounce_count,
                }
            )

    random.shuffle(infos)
    candidates = []
    for i, info_a in enumerate(infos):
        start_a = info_a["start"]
        for info_b in infos[i + 1 :]:
            start_b = info_b["start"]
            if (start_a[0] + start_a[1]) % 2 != (start_b[0] + start_b[1]) % 2:
                continue
            if abs(start_a[0] - start_b[0]) + abs(start_a[1] - start_b[1]) < 4:
                continue
            if start_a in info_b["path_set"] or start_b in info_a["path_set"]:
                continue
            if not _satisfies_bounce_profile((info_a["bounces"], info_b["bounces"]), branch["required_bounces"]):
                continue

            shared = (info_a["path_set"] & info_b["path_set"]) - {start_a, start_b}
            if not shared:
                continue
            if branch["require_multi_hit"] and len(shared) < 2:
                continue
            shorter = min(len(info_a["path"]), len(info_b["path"]))
            if len(shared) >= shorter - 1:
                continue
            if len(shared) > max(1, shorter // 2 + 1):
                continue

            out = _paint(shape, shared, Color.GREEN)
            union = _paint(shape, info_a["path_set"] | info_b["path_set"], Color.GREEN)
            if np.array_equal(out, union):
                continue
            if np.array_equal(out, _paint(shape, info_a["path_set"], Color.GREEN)):
                continue
            if np.array_equal(out, _paint(shape, info_b["path_set"], Color.GREEN)):
                continue

            candidates.append((info_a, info_b, out))

    if not candidates:
        return None

    info_a, info_b, expected = random.choice(candidates)
    colors = random.sample(INPUT_COLORS, 2)
    inp = np.full(shape, Color.BLACK, dtype=int)
    inp[info_a["start"]] = colors[0]
    inp[info_b["start"]] = colors[1]
    return inp, expected


def _gen_attempt() -> np.ndarray | None:
    branch = random.choice(_GEN_BRANCHES)
    height = random.randint(*branch["height"])
    width = random.randint(*branch["width"])
    result = _gen_candidate_pair((height, width), branch)
    if result is None:
        return None

    inp, expected = result
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    output_sig = (expected.shape, expected.tobytes())
    if output_sig in _SEEN_OUTPUTS:
        return None
    _SEEN_OUTPUTS.add(output_sig)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-double-bounce intersection input")
