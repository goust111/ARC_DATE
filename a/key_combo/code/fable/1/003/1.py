from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np

try:
    _ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
    if str(_ARC_UTILS_DIR) not in sys.path:
        sys.path.append(str(_ARC_UTILS_DIR))
    from arc_utils import find_components, get_bg  # type: ignore
except Exception:
    def get_bg(grid: np.ndarray) -> int:
        values, counts = np.unique(grid, return_counts=True)
        return int(values[np.argmax(counts)])

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        colors: list[int] | None = None,
        min_area: int = 1,
        return_format: str = "dict",
    ) -> list[dict[str, Any]]:
        if not monochromatic or return_format != "dict":
            raise ValueError("fallback find_components only supports monochromatic dict output")

        if bg is None:
            bg = get_bg(grid)
        bg = int(bg)

        if colors is None:
            palette = sorted(int(v) for v in np.unique(grid) if int(v) != bg)
        else:
            palette = [int(v) for v in colors if int(v) != bg]

        if connectivity == 4:
            steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

        h, w = grid.shape
        out: list[dict[str, Any]] = []
        for color in palette:
            mask = grid == color
            seen = np.zeros_like(mask, dtype=bool)
            coords = np.argwhere(mask)
            for r0, c0 in coords:
                r0 = int(r0)
                c0 = int(c0)
                if seen[r0, c0]:
                    continue
                stack = [(r0, c0)]
                seen[r0, c0] = True
                comp: list[tuple[int, int]] = []
                while stack:
                    r, c = stack.pop()
                    comp.append((r, c))
                    for dr, dc in steps:
                        rr = r + dr
                        cc = c + dc
                        if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                            seen[rr, cc] = True
                            stack.append((rr, cc))
                if len(comp) < int(min_area):
                    continue
                arr = np.asarray(comp, dtype=int)
                r1, c1 = arr.min(axis=0)
                r2, c2 = arr.max(axis=0)
                out.append(
                    {
                        "color": int(color),
                        "coords": arr,
                        "bbox": (int(r1), int(c1), int(r2), int(c2)),
                        "area": int(arr.shape[0]),
                    }
                )
        return out


SIDE_TO_VEC = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GEN_RETRIES = 400
CHAIN_LENGTHS = np.array([0, 1, 2, 3], dtype=int)
CHAIN_PROBS = np.array([0.18, 0.32, 0.30, 0.20], dtype=float)
GRID_HEIGHT_RANGE = (14, 25)
GRID_WIDTH_RANGE = (14, 25)
BLOCK_SIZE_RANGE = (1, 5)
GAP_RANGE = (2, 5)
END_GAP_RANGE = (2, 6)
DISTRACTOR_MAX = 3
PALETTE = np.arange(10, dtype=int)


def _make_block(r0: int, c0: int, h: int, w: int, color: int) -> dict[str, int]:
    return {"r0": int(r0), "c0": int(c0), "h": int(h), "w": int(w), "color": int(color)}


def _copy_block(block: dict[str, int]) -> dict[str, int]:
    return dict(block)


def _block_signature(block: dict[str, int]) -> tuple[int, int, int, int, int]:
    return (int(block["color"]), int(block["r0"]), int(block["c0"]), int(block["h"]), int(block["w"]))


def _r1(block: dict[str, int]) -> int:
    return int(block["r0"] + block["h"] - 1)


def _c1(block: dict[str, int]) -> int:
    return int(block["c0"] + block["w"] - 1)


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _rects_too_close(a: dict[str, int], b: dict[str, int], pad: int = 1) -> bool:
    pad = int(pad)
    if _r1(a) + pad < int(b["r0"]):
        return False
    if _r1(b) + pad < int(a["r0"]):
        return False
    if _c1(a) + pad < int(b["c0"]):
        return False
    if _c1(b) + pad < int(a["c0"]):
        return False
    return True


def _shift_block(block: dict[str, int], steps: int, direction: tuple[int, int]) -> dict[str, int]:
    dr, dc = map(int, direction)
    out = _copy_block(block)
    out["r0"] = int(out["r0"] + steps * dr)
    out["c0"] = int(out["c0"] + steps * dc)
    return out


def _border_gap(block: dict[str, int], direction: tuple[int, int], shape: tuple[int, int]) -> int:
    h, w = map(int, shape)
    dr, dc = map(int, direction)
    if (dr, dc) == (-1, 0):
        return int(block["r0"] - 1)
    if (dr, dc) == (1, 0):
        return int((h - 2) - _r1(block))
    if (dr, dc) == (0, -1):
        return int(block["c0"] - 1)
    if (dr, dc) == (0, 1):
        return int((w - 2) - _c1(block))
    raise ValueError("invalid direction")


def _forward_gap(current: dict[str, int], other: dict[str, int], direction: tuple[int, int]) -> int | None:
    dr, dc = map(int, direction)
    if dr != 0:
        if not _interval_overlap(int(current["c0"]), _c1(current), int(other["c0"]), _c1(other)):
            return None
        if dr < 0:
            if _r1(other) >= int(current["r0"]):
                return None
            return int(current["r0"] - _r1(other) - 1)
        if int(other["r0"]) <= _r1(current):
            return None
        return int(other["r0"] - _r1(current) - 1)

    if not _interval_overlap(int(current["r0"]), _r1(current), int(other["r0"]), _r1(other)):
        return None
    if dc < 0:
        if _c1(other) >= int(current["c0"]):
            return None
        return int(current["c0"] - _c1(other) - 1)
    if int(other["c0"]) <= _c1(current):
        return None
    return int(other["c0"] - _c1(current) - 1)


def _first_hit(
    blocks: list[dict[str, int]],
    moving_idx: int,
    direction: tuple[int, int],
) -> tuple[int, int] | None:
    current = blocks[int(moving_idx)]
    candidates: list[tuple[int, int]] = []
    for idx, other in enumerate(blocks):
        if idx == int(moving_idx):
            continue
        gap = _forward_gap(current, other, direction)
        if gap is not None:
            candidates.append((int(gap), int(idx)))
    if not candidates:
        return None
    min_gap = min(gap for gap, _ in candidates)
    hits = [idx for gap, idx in candidates if gap == min_gap]
    assert min_gap >= 1, "blocks must start with at least one empty cell between them"
    assert len(hits) == 1, "the visible layout must determine a unique next collision"
    return int(min_gap), int(hits[0])


def _simulate_relay(
    blocks: list[dict[str, int]],
    driver_idx: int,
    direction: tuple[int, int],
    shape: tuple[int, int],
) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    state = [_copy_block(block) for block in blocks]
    moving_idx = int(driver_idx)
    trace: list[dict[str, int]] = []
    while True:
        border_steps = _border_gap(state[moving_idx], direction, shape)
        assert border_steps >= 1, "the moving block must start away from its target border"
        hit = _first_hit(state, moving_idx, direction)
        if hit is None:
            state[moving_idx] = _shift_block(state[moving_idx], border_steps, direction)
            trace.append({"kind": 1, "moving": int(moving_idx), "steps": int(border_steps), "target": -1})
            break
        hit_steps, hit_idx = hit
        assert hit_steps < border_steps, "a real collision must occur before the border"
        state[moving_idx] = _shift_block(state[moving_idx], hit_steps, direction)
        trace.append({"kind": 0, "moving": int(moving_idx), "steps": int(hit_steps), "target": int(hit_idx)})
        moving_idx = int(hit_idx)
    return state, trace


def _render_grid(
    shape: tuple[int, int],
    bg: int,
    side_colors: dict[str, int],
    blocks: list[dict[str, int]],
) -> np.ndarray:
    h, w = map(int, shape)
    out = np.full((h, w), int(bg), dtype=int)
    out[0, 1:-1] = int(side_colors["top"])
    out[h - 1, 1:-1] = int(side_colors["bottom"])
    out[1:-1, 0] = int(side_colors["left"])
    out[1:-1, w - 1] = int(side_colors["right"])
    for block in blocks:
        r0 = int(block["r0"])
        c0 = int(block["c0"])
        out[r0 : r0 + int(block["h"]), c0 : c0 + int(block["w"])] = int(block["color"])
    return out


def _parse_grid(grid: np.ndarray) -> dict[str, Any]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert 5 <= h <= 30 and 5 <= w <= 30, "grid must fit ARC bounds"

    bg = int(g[0, 0])
    assert int(g[0, w - 1]) == bg and int(g[h - 1, 0]) == bg and int(g[h - 1, w - 1]) == bg

    side_colors = {
        "top": int(g[0, 1]),
        "bottom": int(g[h - 1, 1]),
        "left": int(g[1, 0]),
        "right": int(g[1, w - 1]),
    }
    assert len(set(side_colors.values())) == 4, "each border side must have a distinct color"
    assert bg not in set(side_colors.values()), "border sides must differ from the corner background"
    assert np.all(g[0, 1:-1] == side_colors["top"])
    assert np.all(g[h - 1, 1:-1] == side_colors["bottom"])
    assert np.all(g[1:-1, 0] == side_colors["left"])
    assert np.all(g[1:-1, w - 1] == side_colors["right"])

    inner = g.copy()
    inner[0, :] = bg
    inner[h - 1, :] = bg
    inner[:, 0] = bg
    inner[:, w - 1] = bg

    components = find_components(inner, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert components, "expected at least one interior block"

    blocks: list[dict[str, int]] = []
    driver_candidates: list[int] = []
    neutral_colors: set[int] = set()
    side_to_color = {name: int(color) for name, color in side_colors.items()}

    for comp in components:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        color = int(comp["color"])
        area = int(comp["area"])
        assert area == (r1 - r0 + 1) * (c1 - c0 + 1), "every block must be a solid rectangle"
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle footprint must be monochrome"
        block = _make_block(r0, c0, r1 - r0 + 1, c1 - c0 + 1, color)
        for other in blocks:
            assert not _rects_too_close(block, other, pad=1), "blocks must remain visually separated"
        blocks.append(block)
        if color in side_to_color.values():
            driver_candidates.append(len(blocks) - 1)
        else:
            neutral_colors.add(color)

    assert len(driver_candidates) == 1, "exactly one block must match a border-side color"
    assert len(neutral_colors) <= 1, "all non-driver blocks must share one neutral color"
    driver_idx = int(driver_candidates[0])
    driver_color = int(blocks[driver_idx]["color"])
    driver_side = [name for name, color in side_to_color.items() if color == driver_color]
    assert len(driver_side) == 1
    direction = SIDE_TO_VEC[driver_side[0]]

    return {
        "bg": bg,
        "shape": (h, w),
        "side_colors": side_colors,
        "blocks": blocks,
        "driver_idx": driver_idx,
        "direction": direction,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Relay one sliding rectangle toward its same-colored border until the last hit block reaches the frame.

    Input structure:
    - The outer ring is a one-cell border whose four sides are solid, pairwise distinct colors and whose corners are background.
    - Inside the border are separated solid rectangular blocks; exactly one block shares its color with exactly one border side.
    - Every other block is the same neutral color, and only some of them may lie on the driver's straight path to that matching side.

    Rule:
    - Move the unique border-colored driver block straight toward the border side of the same color.
    - If the moving block would overlap another rectangle on the next step, stop it in the last position where it is flush against that rectangle, then transfer all motion to the struck rectangle in the same direction.
    - Repeat this one-block-at-a-time relay until the current moving rectangle becomes flush with the border; untouched rectangles stay fixed.

    Key insight:
    - Color matching between one interior block and one border side gives the only direction, while collisions do not form a pushed stack: each moving block freezes at contact and the struck block alone continues the slide.
    """
    parsed = _parse_grid(grid)
    final_blocks, _trace = _simulate_relay(
        parsed["blocks"],
        parsed["driver_idx"],
        parsed["direction"],
        parsed["shape"],
    )
    return _render_grid(parsed["shape"], parsed["bg"], parsed["side_colors"], final_blocks)


def _sample_axis_layout(rng: np.random.Generator, axis_limit: int, n_blocks: int) -> tuple[list[int], list[int], int] | None:
    for _ in range(120):
        sizes = [int(rng.integers(BLOCK_SIZE_RANGE[0], BLOCK_SIZE_RANGE[1])) for _ in range(n_blocks)]
        gaps = [int(rng.integers(GAP_RANGE[0], GAP_RANGE[1])) for _ in range(max(0, n_blocks - 1))]
        end_gap = int(rng.integers(END_GAP_RANGE[0], END_GAP_RANGE[1]))
        total = end_gap + sum(sizes) + sum(gaps)
        if total <= int(axis_limit):
            return sizes, gaps, end_gap
    return None


def _build_chain(
    shape: tuple[int, int],
    direction_name: str,
    driver_color: int,
    neutral_color: int,
    chain_length: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, int]], tuple[int, int, int, int]] | None:
    h, w = map(int, shape)
    direction = SIDE_TO_VEC[direction_name]
    n_blocks = int(chain_length) + 1

    if direction_name in ("top", "bottom"):
        axis_limit = h - 3
        layout = _sample_axis_layout(rng, axis_limit, n_blocks)
        if layout is None:
            return None
        sizes, gaps, end_gap = layout
        lane_w_max = min(4, w - 5)
        if lane_w_max < 1:
            return None
        lane_w = int(rng.integers(1, lane_w_max + 1))
        lane_c0 = int(rng.integers(2, w - lane_w - 1))
        starts = [0] * n_blocks
        if direction_name == "top":
            starts[-1] = 1 + end_gap
            for idx in range(n_blocks - 2, -1, -1):
                starts[idx] = starts[idx + 1] + sizes[idx + 1] + gaps[idx]
        else:
            starts[-1] = (h - 2) - end_gap - sizes[-1] + 1
            for idx in range(n_blocks - 2, -1, -1):
                starts[idx] = starts[idx + 1] - gaps[idx] - sizes[idx]
        blocks = []
        for idx in range(n_blocks):
            color = int(driver_color if idx == 0 else neutral_color)
            blocks.append(_make_block(starts[idx], lane_c0, sizes[idx], lane_w, color))
        lane_bbox = (min(starts), lane_c0, max(starts[idx] + sizes[idx] - 1 for idx in range(n_blocks)), lane_c0 + lane_w - 1)
        return blocks, lane_bbox

    axis_limit = w - 3
    layout = _sample_axis_layout(rng, axis_limit, n_blocks)
    if layout is None:
        return None
    sizes, gaps, end_gap = layout
    lane_h_max = min(4, h - 5)
    if lane_h_max < 1:
        return None
    lane_h = int(rng.integers(1, lane_h_max + 1))
    lane_r0 = int(rng.integers(2, h - lane_h - 1))
    starts = [0] * n_blocks
    if direction_name == "left":
        starts[-1] = 1 + end_gap
        for idx in range(n_blocks - 2, -1, -1):
            starts[idx] = starts[idx + 1] + sizes[idx + 1] + gaps[idx]
    else:
        starts[-1] = (w - 2) - end_gap - sizes[-1] + 1
        for idx in range(n_blocks - 2, -1, -1):
            starts[idx] = starts[idx + 1] - gaps[idx] - sizes[idx]
    blocks = []
    for idx in range(n_blocks):
        color = int(driver_color if idx == 0 else neutral_color)
        blocks.append(_make_block(lane_r0, starts[idx], lane_h, sizes[idx], color))
    lane_bbox = (lane_r0, min(starts), lane_r0 + lane_h - 1, max(starts[idx] + sizes[idx] - 1 for idx in range(n_blocks)))
    return blocks, lane_bbox


def _random_distractor(
    shape: tuple[int, int],
    direction_name: str,
    lane_bbox: tuple[int, int, int, int],
    existing: list[dict[str, int]],
    neutral_color: int,
    rng: np.random.Generator,
) -> dict[str, int] | None:
    h, w = map(int, shape)
    lane_r0, lane_c0, lane_r1, lane_c1 = map(int, lane_bbox)
    vertical = direction_name in ("top", "bottom")

    for attempt in range(180):
        if attempt < 80:
            if vertical:
                band_choices: list[tuple[int, int]] = []
                if lane_c0 - 2 >= 2:
                    band_choices.append((2, lane_c0 - 2))
                if lane_c1 + 2 <= w - 3:
                    band_choices.append((lane_c1 + 2, w - 3))
                if not band_choices:
                    continue
                h_rect = int(rng.integers(1, min(5, h - 3)))
                w_band0, w_band1 = band_choices[int(rng.integers(0, len(band_choices)))]
                w_max = min(4, w_band1 - w_band0 + 1)
                if w_max < 1:
                    continue
                w_rect = int(rng.integers(1, w_max + 1))
                r_low = 2
                r_high = h - h_rect - 2
                if r_low > r_high:
                    continue
                c_low = w_band0
                c_high = w_band1 - w_rect + 1
                if c_low > c_high:
                    continue
                r0 = int(rng.integers(r_low, r_high + 1))
                c0 = int(rng.integers(c_low, c_high + 1))
            else:
                band_choices = []
                if lane_r0 - 2 >= 2:
                    band_choices.append((2, lane_r0 - 2))
                if lane_r1 + 2 <= h - 3:
                    band_choices.append((lane_r1 + 2, h - 3))
                if not band_choices:
                    continue
                w_rect = int(rng.integers(1, min(5, w - 3)))
                h_band0, h_band1 = band_choices[int(rng.integers(0, len(band_choices)))]
                h_max = min(4, h_band1 - h_band0 + 1)
                if h_max < 1:
                    continue
                h_rect = int(rng.integers(1, h_max + 1))
                r_low = h_band0
                r_high = h_band1 - h_rect + 1
                c_low = 2
                c_high = w - w_rect - 2
                if r_low > r_high or c_low > c_high:
                    continue
                r0 = int(rng.integers(r_low, r_high + 1))
                c0 = int(rng.integers(c_low, c_high + 1))
        else:
            h_rect = int(rng.integers(1, min(5, h - 3)))
            w_rect = int(rng.integers(1, min(5, w - 3)))
            r_low = 2
            r_high = h - h_rect - 2
            c_low = 2
            c_high = w - w_rect - 2
            if r_low > r_high or c_low > c_high:
                continue
            r0 = int(rng.integers(r_low, r_high + 1))
            c0 = int(rng.integers(c_low, c_high + 1))

        block = _make_block(r0, c0, h_rect, w_rect, neutral_color)
        if any(_rects_too_close(block, other, pad=1) for other in existing):
            continue
        if vertical and _interval_overlap(int(block["c0"]), _c1(block), lane_c0, lane_c1):
            continue
        if (not vertical) and _interval_overlap(int(block["r0"]), _r1(block), lane_r0, lane_r1):
            continue
        return block
    return None


def _naive_driver_only(
    blocks: list[dict[str, int]],
    driver_idx: int,
    direction: tuple[int, int],
    shape: tuple[int, int],
) -> list[dict[str, int]]:
    out = [_copy_block(block) for block in blocks]
    out[driver_idx] = _shift_block(out[driver_idx], _border_gap(out[driver_idx], direction, shape), direction)
    return out


def _validate_generated_case(
    grid: np.ndarray,
    side_colors: dict[str, int],
    blocks: list[dict[str, int]],
    driver_idx: int,
    direction_name: str,
    chain_length: int,
    distractor_count: int,
) -> bool:
    parsed = _parse_grid(grid)
    if parsed["bg"] != int(grid[0, 0]):
        return False
    if get_bg(grid) != int(grid[0, 0]):
        return False
    if parsed["side_colors"] != side_colors:
        return False
    if parsed["direction"] != SIDE_TO_VEC[direction_name]:
        return False

    expected_in = sorted(_block_signature(block) for block in blocks)
    parsed_in = sorted(_block_signature(block) for block in parsed["blocks"])
    if expected_in != parsed_in:
        return False
    if int(parsed["blocks"][parsed["driver_idx"]]["color"]) != int(blocks[driver_idx]["color"]):
        return False

    expected_out_blocks, trace = _simulate_relay(blocks, driver_idx, SIDE_TO_VEC[direction_name], grid.shape)
    expected_out = _render_grid(grid.shape, int(grid[0, 0]), side_colors, expected_out_blocks)
    actual_out = main(grid)
    if not np.array_equal(actual_out, expected_out):
        return False
    if np.array_equal(actual_out, grid):
        return False

    moved = [expected_out_blocks[idx]["r0"] != blocks[idx]["r0"] or expected_out_blocks[idx]["c0"] != blocks[idx]["c0"] for idx in range(len(blocks))]
    if sum(moved) != int(chain_length) + 1:
        return False
    if distractor_count > 0 and sum(not flag for flag in moved[chain_length + 1 :]) == 0:
        return False

    if chain_length > 0:
        if not any(step["kind"] == 0 for step in trace):
            return False
        naive_out = _render_grid(grid.shape, int(grid[0, 0]), side_colors, _naive_driver_only(blocks, driver_idx, SIDE_TO_VEC[direction_name], grid.shape))
        if np.array_equal(naive_out, expected_out):
            return False
    else:
        if any(step["kind"] == 0 for step in trace):
            return False

    final_mover = trace[-1]["moving"]
    if _border_gap(expected_out_blocks[final_mover], SIDE_TO_VEC[direction_name], grid.shape) != 0:
        return False
    if int(grid[0, 0]) in set(side_colors.values()):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_HEIGHT_RANGE[0], GRID_HEIGHT_RANGE[1]))
    w = int(rng.integers(GRID_WIDTH_RANGE[0], GRID_WIDTH_RANGE[1]))
    chosen = rng.choice(PALETTE, size=6, replace=False)
    bg = int(chosen[0])
    neutral_color = int(chosen[1])
    side_colors = {
        "top": int(chosen[2]),
        "bottom": int(chosen[3]),
        "left": int(chosen[4]),
        "right": int(chosen[5]),
    }
    direction_name = str(rng.choice(np.array(["top", "bottom", "left", "right"], dtype=object)))
    driver_color = int(side_colors[direction_name])
    chain_length = int(rng.choice(CHAIN_LENGTHS, p=CHAIN_PROBS))

    chain_result = _build_chain((h, w), direction_name, driver_color, neutral_color, chain_length, rng)
    if chain_result is None:
        return None
    blocks, lane_bbox = chain_result

    distractor_min = 1 if chain_length == 0 else 0
    distractor_count = int(rng.integers(distractor_min, DISTRACTOR_MAX + 1))
    for _ in range(distractor_count):
        distractor = _random_distractor((h, w), direction_name, lane_bbox, blocks, neutral_color, rng)
        if distractor is None:
            return None
        blocks.append(distractor)

    grid = _render_grid((h, w), bg, side_colors, blocks)
    if not _validate_generated_case(grid, side_colors, blocks, 0, direction_name, chain_length, distractor_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid momentum-relay sliding input")
