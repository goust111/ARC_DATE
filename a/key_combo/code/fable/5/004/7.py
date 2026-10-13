from __future__ import annotations

from importlib.machinery import SourceFileLoader
from itertools import combinations
from pathlib import Path
import random

import numpy as np


# Keep the task directly loadable with SourceFileLoader even when arc_utils is not
# on sys.path (the normal situation for generated puzzle files).
try:
    from arc_utils import bbox_coords, components_mask, get_bg
except ModuleNotFoundError:
    _UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    _arc_utils = SourceFileLoader("_fable_5_004_7_arc_utils", str(_UTILS_PATH)).load_module()
    bbox_coords = _arc_utils.bbox_coords
    components_mask = _arc_utils.components_mask
    get_bg = _arc_utils.get_bg


RED = 2
GREY = 5
TEAL = 8
FULCRUM = 9
BACKGROUND = 0
WEIGHT_COLORS = (1, 3, 4, 6, 7)
MAX_GENERATION_ATTEMPTS = 1200


def _longest_horizontal_run(g: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Return (row, first_col, last_col, color) for the unique longest run."""
    runs: list[tuple[int, int, int, int]] = []
    H, W = g.shape
    for r in range(H):
        c = 0
        while c < W:
            color = int(g[r, c])
            if color == bg:
                c += 1
                continue
            c0 = c
            while c + 1 < W and int(g[r, c + 1]) == color:
                c += 1
            runs.append((r, c0, c, color))
            c += 1

    assert runs, "expected a foreground beam"
    longest = max(c1 - c0 + 1 for _r, c0, c1, _color in runs)
    winners = [run for run in runs if run[2] - run[1] + 1 == longest]
    assert len(winners) == 1, "the horizontal beam must be uniquely longest"
    assert longest >= W // 2, "the beam must span most of the scene"
    return winners[0]


def _parse_balance(g: np.ndarray) -> tuple[int, int, list[dict[str, object]]]:
    """Recover the fulcrum column and every visibly connected hanging block."""
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    beam_r, beam_c0, beam_c1, structure_color = _longest_horizontal_run(g, bg)
    assert (beam_c0 + beam_c1) % 2 == 0, "beam midpoint must be an integer column"
    fulcrum_c = (beam_c0 + beam_c1) // 2
    assert beam_r + 1 < g.shape[0], "fulcrum must fit below the beam"

    fulcrum_color = int(g[beam_r + 1, fulcrum_c])
    assert fulcrum_color not in (bg, structure_color), "expected a marked fulcrum below the midpoint"
    assert int(np.count_nonzero(g == fulcrum_color)) == 1, "fulcrum marker must be unique"

    role_colors = set(map(int, np.unique(g))) - {bg, structure_color, fulcrum_color}
    assert len(role_colors) == 1, "all hanging blocks must share one input color"
    weight_color = role_colors.pop()

    blocks: list[dict[str, object]] = []
    for coords in components_mask(g == weight_color, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(coords)
        area = int(coords.shape[0])
        assert area == (r1 - r0 + 1) * (c1 - c0 + 1), "hanging blocks must be solid rectangles"
        assert r0 > beam_r + 1, "hanging blocks must lie below the beam"

        top_columns = {int(c) for r, c in coords if int(r) == r0}
        attachment_candidates = []
        for c in sorted(top_columns):
            if not (beam_c0 <= c <= beam_c1) or c == fulcrum_c:
                continue
            if int(g[beam_r, c]) != structure_color:
                continue
            if np.all(g[beam_r + 1 : r0, c] == structure_color):
                attachment_candidates.append(c)
        assert len(attachment_candidates) == 1, "each block needs one unambiguous vertical rod"

        attachment_c = attachment_candidates[0]
        side = -1 if attachment_c < fulcrum_c else 1
        blocks.append(
            {
                "coords": coords,
                "area": area,
                "attachment_c": attachment_c,
                "distance": abs(attachment_c - fulcrum_c),
                "side": side,
            }
        )

    assert len(blocks) >= 2, "expected hanging blocks"
    assert {-1, 1} == {int(block["side"]) for block in blocks}, "both beam sides need weights"
    return fulcrum_c, weight_color, blocks


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor hanging blocks by comparing total weight-times-lever-arm torque.

    Input structure:
    - A uniquely longest horizontal beam crosses the scene, with a single distinct
      fulcrum pixel immediately below its midpoint.
    - Solid same-colored rectangular weights hang below both sides.  Each has one
      thin vertical rod whose beam column is its attachment point.

    Rule:
    - Treat a weight's pixel count as its mass and its attachment-column distance
      from the fulcrum column as its lever arm.  Sum mass times lever arm separately
      on the left and right.
    - Recolor every weight on the larger-torque side teal and every weight on the
      smaller-torque side red.  Preserve the background, beam, fulcrum, and rods.

    Key insight:
    - Neither total colored area nor distance alone decides the result: each block's
      area must be multiplied by its distance in the beam/fulcrum coordinate frame.
    """
    g = np.asarray(grid, dtype=int)
    _fulcrum_c, _weight_color, blocks = _parse_balance(g)

    torques = {
        side: sum(
            int(block["area"]) * int(block["distance"])
            for block in blocks
            if int(block["side"]) == side
        )
        for side in (-1, 1)
    }
    assert torques[-1] != torques[1], "left and right torques must not tie"
    winning_side = -1 if torques[-1] > torques[1] else 1

    out = g.copy()
    for block in blocks:
        color = TEAL if int(block["side"]) == winning_side else RED
        coords = block["coords"]
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _side_interval_choices(side: int, fulcrum_c: int, width: int) -> list[tuple[int, int]]:
    """List (attachment_col, block_width) options entirely on one side."""
    choices = []
    for block_w in (1, 3):
        half = block_w // 2
        if side == -1:
            centers = range(3 + half, fulcrum_c - 1 - half)
        else:
            centers = range(fulcrum_c + 2 + half, width - 3 - half)
        choices.extend((c, block_w) for c in centers)
    return choices


def _sample_side_blocks(
    side: int,
    count: int,
    fulcrum_c: int,
    H: int,
    W: int,
    beam_r: int,
) -> list[dict[str, int]] | None:
    """Choose separated horizontal footprints, then vary mass and hanging height."""
    choices = _side_interval_choices(side, fulcrum_c, W)
    compatible: list[tuple[tuple[int, int], ...]] = []
    for combo in combinations(choices, count):
        ordered = sorted(combo)
        if len({c for c, _w in ordered}) != count:
            continue
        spans = [(c - w // 2, c + w // 2) for c, w in ordered]
        if all(spans[i + 1][0] - spans[i][1] >= 2 for i in range(count - 1)):
            compatible.append(tuple(ordered))
    if not compatible:
        return None

    selected = random.choice(compatible)
    blocks = []
    for attachment_c, block_w in selected:
        block_h = random.randint(2, 5)
        top_min = beam_r + 4
        top_max = H - block_h - 2
        if top_min > top_max:
            return None
        top = random.randint(top_min, top_max)
        blocks.append(
            {
                "side": side,
                "attachment_c": attachment_c,
                "width": block_w,
                "height": block_h,
                "top": top,
                "left": attachment_c - block_w // 2,
                "area": block_w * block_h,
                "distance": abs(attachment_c - fulcrum_c),
            }
        )
    return blocks


def _sign(right_minus_left: int) -> int:
    return (right_minus_left > 0) - (right_minus_left < 0)


def _semantic_stats(blocks: list[dict[str, int]]) -> tuple[int, dict[int, int], dict[int, int], dict[int, int]]:
    masses = {s: sum(b["area"] for b in blocks if b["side"] == s) for s in (-1, 1)}
    arms = {s: sum(b["distance"] for b in blocks if b["side"] == s) for s in (-1, 1)}
    torques = {
        s: sum(b["area"] * b["distance"] for b in blocks if b["side"] == s)
        for s in (-1, 1)
    }
    torque_side = _sign(torques[1] - torques[-1])
    return torque_side, masses, arms, torques


def _render_input(
    H: int,
    W: int,
    beam_r: int,
    fulcrum_c: int,
    weight_color: int,
    blocks: list[dict[str, int]],
) -> np.ndarray:
    grid = np.full((H, W), BACKGROUND, dtype=int)
    grid[beam_r, 1 : W - 1] = GREY
    grid[beam_r + 1, fulcrum_c] = FULCRUM
    for block in blocks:
        c = block["attachment_c"]
        top = block["top"]
        grid[beam_r + 1 : top, c] = GREY
        r1 = top + block["height"]
        c0 = block["left"]
        c1 = c0 + block["width"]
        grid[top:r1, c0:c1] = weight_color
    return grid


def _expected_output(inp: np.ndarray, blocks: list[dict[str, int]], winning_side: int) -> np.ndarray:
    out = inp.copy()
    for block in blocks:
        color = TEAL if block["side"] == winning_side else RED
        r0 = block["top"]
        r1 = r0 + block["height"]
        c0 = block["left"]
        c1 = c0 + block["width"]
        out[r0:r1, c0:c1] = color
    return out


def _validate_generated(
    inp: np.ndarray,
    blocks: list[dict[str, int]],
    expected: np.ndarray,
    torque_side: int,
    masses: dict[int, int],
    arms: dict[int, int],
    torques: dict[int, int],
) -> bool:
    """Check the stated semantics and reject shortcuts, not merely round-trip output."""
    if inp.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if len([b for b in blocks if b["side"] == -1]) < 2:
        return False
    if len([b for b in blocks if b["side"] == 1]) < 2:
        return False
    if torque_side == 0 or torques[-1] == torques[1]:
        return False

    # Strong counterexamples: replacing all lever arms by one (mass-only) and
    # replacing all masses by one (distance-only) must both choose the wrong side.
    mass_side = _sign(masses[1] - masses[-1])
    arm_side = _sign(arms[1] - arms[-1])
    if mass_side != -torque_side or arm_side != -torque_side:
        return False
    if len({b["area"] for b in blocks}) < 3 or len({b["distance"] for b in blocks}) < 3:
        return False

    changed = inp != expected
    intended = np.zeros_like(inp, dtype=bool)
    for block in blocks:
        r0, c0 = block["top"], block["left"]
        intended[r0 : r0 + block["height"], c0 : c0 + block["width"]] = True
    if not np.array_equal(changed, intended):
        return False
    if not ({RED, TEAL} <= set(map(int, np.unique(expected)))):
        return False

    try:
        solved = main(inp)
        _pivot, _input_color, parsed = _parse_balance(inp)
    except (AssertionError, ValueError):
        return False
    if len(parsed) != len(blocks) or not np.array_equal(solved, expected):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    W = random.choice((21, 23, 25, 27, 29))
    H = random.randint(16, 24)
    fulcrum_c = W // 2
    beam_r = random.randint(2, 4)
    max_count = 3 if W >= 27 else 2
    left_count = random.randint(2, max_count)
    right_count = random.randint(2, max_count)

    left = _sample_side_blocks(-1, left_count, fulcrum_c, H, W, beam_r)
    right = _sample_side_blocks(1, right_count, fulcrum_c, H, W, beam_r)
    if left is None or right is None:
        return None
    blocks = left + right

    torque_side, masses, arms, torques = _semantic_stats(blocks)
    if torque_side == 0:
        return None
    weight_color = random.choice(WEIGHT_COLORS)
    inp = _render_input(H, W, beam_r, fulcrum_c, weight_color, blocks)
    expected = _expected_output(inp, blocks, torque_side)
    if not _validate_generated(inp, blocks, expected, torque_side, masses, arms, torques):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a varied, unambiguous balance with anti-shortcut counterexamples."""
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid balance after {MAX_GENERATION_ATTEMPTS} attempts")
