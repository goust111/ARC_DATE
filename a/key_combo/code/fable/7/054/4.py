from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_TOOLBOX_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_TOOLBOX_DIR) not in sys.path:
    sys.path.append(str(_TOOLBOX_DIR))

from arc_utils import Color, colors_present, find_components, get_bg


BG_COLOR = Color.BLACK
COLOR_POOL = [c for c in Color.NOT_BLACK if c != Color.GREY]
GENERATION_RETRIES = 500
BRANCH_SEQUENCE = ["count_area_trap", "far_trap", "tie_trap"]
_branch_index = 0


def _cmp(a: int, b: int) -> int:
    return int(a > b) - int(a < b)


def _interval_sum(near: int, width: int) -> int:
    return int(sum(range(near, near + width)))


def _copy_pieces(pieces: list[dict]) -> list[dict]:
    return [dict(piece) for piece in pieces]


def _choose_intervals(
    center: int,
    count: int,
    dist_lo: int,
    dist_hi: int,
    width_lo: int,
    width_hi: int,
) -> list[tuple[int, int]] | None:
    dist_hi = min(dist_hi, center)
    candidates: list[tuple[int, int]] = []
    for near in range(dist_lo, dist_hi + 1):
        max_width = min(width_hi, center - near + 1)
        for width in range(width_lo, max_width + 1):
            candidates.append((near, width))
    random.shuffle(candidates)

    picked: list[tuple[int, int]] = []
    for near, width in candidates:
        far = near + width - 1
        if all(far < old_near - 1 or near > old_near + old_width for old_near, old_width in picked):
            picked.append((near, width))
            if len(picked) == count:
                return picked
    return None


def _abstract_metrics(pieces: list[dict]) -> dict:
    torque = 0
    area = 0
    farthest = 0
    for piece in pieces:
        coeff = _interval_sum(piece["near"], piece["width"])
        torque += piece["height"] * coeff
        area += piece["height"] * piece["width"]
        farthest = max(farthest, piece["near"] + piece["width"] - 1)
    return {
        "torque": int(torque),
        "area": int(area),
        "farthest": int(farthest),
        "components": len(pieces),
    }


def _assign_orientations(left: list[dict], right: list[dict]) -> None:
    pieces = left + right
    orientations = ["up", "down"] + [random.choice(["up", "down"]) for _ in range(len(pieces) - 2)]
    random.shuffle(orientations)
    for piece, orientation in zip(pieces, orientations, strict=False):
        piece["orientation"] = orientation


def _add_caps(pieces: list[dict], allow_caps: bool) -> None:
    for piece in pieces:
        piece["cap_height"] = 0
        piece["cap_width"] = 0
        piece["cap_offset"] = 0
    if not allow_caps:
        return

    cap_candidates = pieces[:]
    random.shuffle(cap_candidates)
    n_caps = 1 if cap_candidates else 0
    if len(cap_candidates) > 1 and random.random() < 0.5:
        n_caps = 2
    for piece in cap_candidates[:n_caps]:
        piece["cap_height"] = random.randint(1, 2)
        piece["cap_width"] = random.randint(1, piece["width"])
        piece["cap_offset"] = random.randint(0, piece["width"] - piece["cap_width"])


def _piece_cols(center: int, side: str, near: int, width: int) -> tuple[int, int]:
    far = near + width - 1
    if side == "left":
        return center - far, center - near
    return center + near, center + far


def _render_piece(grid: np.ndarray, beam_row: int, center: int, side: str, color: int, piece: dict) -> None:
    c0, c1 = _piece_cols(center, side, piece["near"], piece["width"])
    if piece["orientation"] == "up":
        base_h = piece["height"]
        r0 = beam_row - base_h
        grid[r0:beam_row, c0 : c1 + 1] = int(color)
        if piece["cap_height"] > 0:
            cap_c0 = c0 + piece["cap_offset"]
            cap_c1 = cap_c0 + piece["cap_width"] - 1
            cap_r0 = r0 - piece["cap_height"]
            grid[cap_r0:r0, cap_c0 : cap_c1 + 1] = int(color)
    else:
        base_h = piece["height"]
        r1 = beam_row + 1 + base_h
        grid[beam_row + 1 : r1, c0 : c1 + 1] = int(color)
        if piece["cap_height"] > 0:
            cap_c0 = c0 + piece["cap_offset"]
            cap_c1 = cap_c0 + piece["cap_width"] - 1
            grid[r1 : r1 + piece["cap_height"], cap_c0 : cap_c1 + 1] = int(color)


def _build_grid(left: list[dict], right: list[dict], left_color: int, right_color: int, center: int) -> np.ndarray:
    max_up = max(
        [piece["height"] + piece["cap_height"] for piece in left + right if piece["orientation"] == "up"] or [1]
    )
    max_down = max(
        [piece["height"] + piece["cap_height"] for piece in left + right if piece["orientation"] == "down"] or [1]
    )
    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    beam_row = top_margin + max_up
    height = beam_row + max_down + bottom_margin + 1
    width = 2 * center + 1
    assert height <= 30 and width <= 30

    grid = np.full((height, width), int(BG_COLOR), dtype=int)
    grid[beam_row, :] = int(Color.GREY)
    grid[beam_row + 1, center] = int(Color.GREY)

    for piece in left:
        _render_piece(grid, beam_row, center, "left", left_color, piece)
    for piece in right:
        _render_piece(grid, beam_row, center, "right", right_color, piece)
    return grid


def _find_beam(grid: np.ndarray) -> tuple[int, int]:
    grey_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == int(Color.GREY))]
    assert len(grey_rows) == 1, "expected exactly one full grey beam row"
    beam_row = grey_rows[0]
    width = int(grid.shape[1])
    assert width % 2 == 1, "beam length must be odd"
    center = width // 2
    assert beam_row + 1 < grid.shape[0], "pivot row must exist"
    expected_grey = {(beam_row, c) for c in range(width)}
    expected_grey.add((beam_row + 1, center))
    actual_grey = {tuple(map(int, rc)) for rc in np.argwhere(grid == int(Color.GREY))}
    assert actual_grey == expected_grey, "grey pixels must be exactly the beam and its pivot"
    return beam_row, center


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    beam_row, center = _find_beam(g)
    assert not np.any((g[:, center] != bg) & (g[:, center] != int(Color.GREY))), "blocks may not cross the center"

    left_colors = colors_present(g[:, :center], exclude={bg, int(Color.GREY)})
    right_colors = colors_present(g[:, center + 1 :], exclude={bg, int(Color.GREY)})
    assert len(left_colors) == 1 and len(right_colors) == 1, "each side must use exactly one color"
    assert left_colors[0] != right_colors[0], "left and right colors must differ"

    metrics = {
        "left": {"color": int(left_colors[0]), "components": 0},
        "right": {"color": int(right_colors[0]), "components": 0},
    }

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    for comp in comps:
        color = int(comp["color"])
        if color == int(Color.GREY):
            continue
        coords = comp["coords"]
        rows = coords[:, 0]
        cols = coords[:, 1]
        assert np.all(rows != beam_row), "colored blocks may not overwrite the beam"
        if np.all(cols < center):
            side = "left"
        elif np.all(cols > center):
            side = "right"
        else:
            raise AssertionError("a block component crossed the pivot column")
        assert color == metrics[side]["color"], "all blocks on one side must share one color"
        metrics[side]["components"] += 1

    for side in ("left", "right"):
        coords = np.argwhere(g == metrics[side]["color"])
        assert coords.size > 0, "both sides must contain load pixels"
        dists = np.abs(coords[:, 1] - center)
        metrics[side]["area"] = int(coords.shape[0])
        metrics[side]["torque"] = int(dists.sum())
        metrics[side]["farthest"] = int(dists.max())

    return {"bg": bg, "beam_row": beam_row, "center": center, **metrics}


def _expected_output(info: dict) -> np.ndarray:
    left = info["left"]
    right = info["right"]
    if left["torque"] == right["torque"]:
        color = int(Color.GREY)
    elif left["torque"] > right["torque"]:
        color = int(left["color"])
    else:
        color = int(right["color"])
    return np.array([[color]], dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decide which side of the grey beam exerts the larger torque and output that side's color.

    Input structure:
    - A single full-width grey beam occupies one entire row, and one grey pivot pixel sits
      directly below the beam's center column.
    - Colored solid rectangles are attached above the beam or hanging below it, optionally
      with stacked extensions, and no colored block crosses the center column.
    - All load pixels on the left side share one color, and all load pixels on the right side
      share another color.

    Rule:
    - For each side, sum the horizontal distance from every colored pixel to the center column;
      pixels above and below the beam contribute equally.
    - Compare the two side totals as torques around the pivot.
    - Return a 1x1 grid containing the heavier side's color, or grey when the torques tie.

    Key insight:
    - Vertical placement is a decoy: the only stable quantity is the per-pixel horizontal
      distance sum to the center column on each side.
    """
    return _expected_output(_analyze_grid(np.asarray(grid, dtype=int)))


def _make_count_area_branch(center: int) -> tuple[list[dict], list[dict]] | None:
    winner = [
        {
            "near": random.randint(max(3, center - 4), center - 1),
            "width": random.randint(2, min(3, center)),
            "height": random.randint(2, 3),
        }
    ]
    if winner[0]["near"] + winner[0]["width"] - 1 > center:
        return None

    loser_intervals = _choose_intervals(center, 3, 1, min(5, center - 1), 1, 2)
    if loser_intervals is None:
        return None
    loser = [
        {"near": near, "width": width, "height": random.randint(1, 2)}
        for near, width in loser_intervals
    ]

    win_metrics = _abstract_metrics(winner)
    lose_metrics = _abstract_metrics(loser)
    if not (
        win_metrics["torque"] > lose_metrics["torque"]
        and win_metrics["area"] < lose_metrics["area"]
        and win_metrics["components"] < lose_metrics["components"]
    ):
        return None

    if random.random() < 0.5:
        return winner, loser
    return loser, winner


def _make_far_branch(center: int) -> tuple[list[dict], list[dict]] | None:
    winner = [
        {
            "near": random.randint(2, min(4, center - 2)),
            "width": random.randint(2, min(3, center - 1)),
            "height": random.randint(2, 3),
        }
    ]
    if winner[0]["near"] + winner[0]["width"] - 1 >= center:
        return None

    loser = [
        {"near": random.randint(1, 2), "width": random.randint(1, 2), "height": random.randint(2, 3)},
        {"near": random.randint(center - 1, center), "width": 1, "height": random.randint(1, 2)},
    ]
    if loser[0]["near"] + loser[0]["width"] >= loser[1]["near"]:
        return None

    win_metrics = _abstract_metrics(winner)
    lose_metrics = _abstract_metrics(loser)
    if not (
        win_metrics["torque"] > lose_metrics["torque"]
        and win_metrics["farthest"] < lose_metrics["farthest"]
        and win_metrics["area"] <= lose_metrics["area"]
    ):
        return None

    if random.random() < 0.5:
        return winner, loser
    return loser, winner


def _make_tie_branch(center: int) -> tuple[list[dict], list[dict]] | None:
    near_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    candidates = _choose_intervals(center, 2, 1, min(3, center - 2), 1, 2)
    if candidates is None:
        return None
    base_pool: list[tuple[int, int]] = []
    for near in range(1, min(3, center - 2) + 1):
        for width in range(1, min(2, center - near + 1) + 1):
            base_pool.append((near, width))
    for i, first in enumerate(base_pool):
        for second in base_pool[i + 1 :]:
            far_first = first[0] + first[1] - 1
            far_second = second[0] + second[1] - 1
            if far_first < second[0] - 1 or far_second < first[0] - 1:
                near_pairs.append((first, second))
    random.shuffle(near_pairs)

    far_options: list[tuple[int, int]] = []
    for near in range(max(3, center // 2), center + 1):
        for width in range(1, min(2, center - near + 1) + 1):
            far_options.append((near, width))
    random.shuffle(far_options)

    for far_near, far_width in far_options:
        far_coeff = _interval_sum(far_near, far_width)
        for far_height in range(1, 4):
            target = far_coeff * far_height
            for (near1, width1), (near2, width2) in near_pairs:
                coeff1 = _interval_sum(near1, width1)
                coeff2 = _interval_sum(near2, width2)
                for h1 in range(1, 5):
                    for h2 in range(1, 5):
                        if coeff1 * h1 + coeff2 * h2 != target:
                            continue
                        near_side = [
                            {"near": near1, "width": width1, "height": h1},
                            {"near": near2, "width": width2, "height": h2},
                        ]
                        far_side = [{"near": far_near, "width": far_width, "height": far_height}]
                        near_metrics = _abstract_metrics(near_side)
                        far_metrics = _abstract_metrics(far_side)
                        if not (
                            near_metrics["torque"] == far_metrics["torque"]
                            and near_metrics["area"] != far_metrics["area"]
                            and near_metrics["farthest"] != far_metrics["farthest"]
                            and near_metrics["components"] != far_metrics["components"]
                        ):
                            continue
                        if random.random() < 0.5:
                            return near_side, far_side
                        return far_side, near_side
    return None


def _branch_spec(branch: str, center: int) -> tuple[list[dict], list[dict], bool] | None:
    if branch == "count_area_trap":
        result = _make_count_area_branch(center)
        return None if result is None else (result[0], result[1], True)
    if branch == "far_trap":
        result = _make_far_branch(center)
        return None if result is None else (result[0], result[1], True)
    result = _make_tie_branch(center)
    return None if result is None else (result[0], result[1], False)


def _semantic_branch_ok(info: dict, branch: str) -> bool:
    torque_cmp = _cmp(info["left"]["torque"], info["right"]["torque"])
    area_cmp = _cmp(info["left"]["area"], info["right"]["area"])
    count_cmp = _cmp(info["left"]["components"], info["right"]["components"])
    far_cmp = _cmp(info["left"]["farthest"], info["right"]["farthest"])

    if branch == "count_area_trap":
        return torque_cmp != 0 and area_cmp == -torque_cmp and count_cmp == -torque_cmp
    if branch == "far_trap":
        return torque_cmp != 0 and far_cmp == -torque_cmp and area_cmp in {-torque_cmp, 0}
    return torque_cmp == 0 and area_cmp != 0 and count_cmp != 0 and far_cmp != 0


def _next_branch() -> str:
    global _branch_index
    branch = BRANCH_SEQUENCE[_branch_index % len(BRANCH_SEQUENCE)]
    _branch_index += 1
    return branch


def _gen_attempt(branch: str) -> np.ndarray | None:
    center = random.randint(6, 10)
    spec = _branch_spec(branch, center)
    if spec is None:
        return None
    left, right, allow_caps = spec
    left = _copy_pieces(left)
    right = _copy_pieces(right)

    _assign_orientations(left, right)
    _add_caps(left + right, allow_caps=allow_caps)

    left_color, right_color = random.sample(COLOR_POOL, 2)
    grid = _build_grid(left, right, left_color, right_color, center)

    try:
        info = _analyze_grid(grid)
    except AssertionError:
        return None

    if not _semantic_branch_ok(info, branch):
        return None

    out = main(grid)
    expected = _expected_output(info)
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    branch = _next_branch()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(branch)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid '{branch}' sample after {GENERATION_RETRIES} attempts")
