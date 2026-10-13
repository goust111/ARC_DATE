from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, find_components


BG = Color.BLACK
TRAIL = Color.ORANGE
RECT_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.TEAL, Color.MAROON]
MODES = ("first_border", "stop_mid", "exact_touch", "relay", "later_border")


def _is_corner(coord: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = map(int, coord)
    h, w = map(int, shape)
    return (r, c) in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}


def _to_forward_interval(start: int, end: int, line_len: int, direction: int) -> tuple[int, int]:
    start = int(start)
    end = int(end)
    line_len = int(line_len)
    direction = int(direction)
    if direction == 1:
        return start, end
    return line_len - 1 - end, line_len - 1 - start


def _from_forward_interval(start: int, end: int, line_len: int, direction: int) -> tuple[int, int]:
    start = int(start)
    end = int(end)
    line_len = int(line_len)
    direction = int(direction)
    if direction == 1:
        return start, end
    return line_len - 1 - end, line_len - 1 - start


def _block_bbox(block: dict, orientation: str, line_len: int, direction: int) -> tuple[int, int, int, int]:
    start, end = _from_forward_interval(block["f0"], block["f1"], line_len, direction)
    if orientation == "horizontal":
        return block["band0"], start, block["band1"], end
    return start, block["band0"], end, block["band1"]


def _paint_trail(out: np.ndarray, before: tuple[int, int, int, int], after: tuple[int, int, int, int]) -> None:
    br0, bc0, br1, bc1 = before
    ar0, ac0, ar1, ac1 = after
    r0 = min(br0, ar0)
    c0 = min(bc0, ac0)
    r1 = max(br1, ar1)
    c1 = max(bc1, ac1)
    out[r0 : r1 + 1, c0 : c1 + 1] = TRAIL


def _simulate_forward_chain(blocks: list[dict], selected_idx: int, line_len: int) -> tuple[list[dict], list[tuple[int, int, int, int, int]]]:
    final_blocks = [dict(block) for block in blocks]
    moves: list[tuple[int, int, int, int, int]] = []
    idx = int(selected_idx)
    quota: int | None = None

    while True:
        block = final_blocks[idx]
        old0 = int(block["f0"])
        old1 = int(block["f1"])
        next_exists = idx + 1 < len(final_blocks)

        if quota is None:
            if next_exists:
                step = int(final_blocks[idx + 1]["f0"]) - old1 - 1
                next_quota = step
                continue_chain = next_quota > 0
                border_contact = False
            else:
                step = int(line_len) - 1 - old1
                next_quota = 0
                continue_chain = False
                border_contact = True
        else:
            if next_exists:
                gap = int(final_blocks[idx + 1]["f0"]) - old1 - 1
                if gap < quota:
                    step = gap
                    next_quota = quota - gap
                    continue_chain = next_quota > 0
                    border_contact = False
                else:
                    step = quota
                    next_quota = 0
                    continue_chain = False
                    border_contact = False
            else:
                border_dist = int(line_len) - 1 - old1
                if border_dist <= quota:
                    step = border_dist
                    next_quota = 0
                    continue_chain = False
                    border_contact = True
                else:
                    step = quota
                    next_quota = 0
                    continue_chain = False
                    border_contact = False

        new0 = old0 + step
        new1 = old1 + step
        block["f0"] = new0
        block["f1"] = new1
        if step > 0:
            moves.append((idx, old0, old1, new0, new1))

        if border_contact or not continue_chain:
            break

        idx += 1
        quota = next_quota

    return final_blocks, moves


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert int(np.min(g)) >= 0

    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")
    guides = [comp for comp in comps if int(comp["area"]) == 1 and _is_corner(tuple(map(int, comp["coords"][0])), g.shape)]
    assert len(guides) == 1, "expected exactly one corner guide pixel"
    guide = guides[0]
    guide_r, guide_c = map(int, guide["coords"][0])
    guide_color = int(guide["color"])

    rects = [comp for comp in comps if comp is not guide]
    assert len(rects) >= 2, "expected at least two line rectangles"
    for rect in rects:
        r0, c0, r1, c1 = map(int, rect["bbox"])
        assert int(rect["area"]) == (r1 - r0 + 1) * (c1 - c0 + 1), "each foreground object must be a solid rectangle"
        assert int(rect["area"]) >= 2, "guide must be the only 1-cell object"

    same_rows = all(tuple(map(int, rect["bbox"][::2])) == tuple(map(int, rects[0]["bbox"][::2])) for rect in rects)
    same_cols = all(tuple(map(int, rect["bbox"][1::2])) == tuple(map(int, rects[0]["bbox"][1::2])) for rect in rects)
    assert same_rows ^ same_cols, "rectangles must align on one shared row band or one shared column band"

    if same_rows:
        orientation = "horizontal"
        band0, band1 = map(int, rects[0]["bbox"][::2])
        direction = 1 if guide_c == 0 else -1
        line_len = int(g.shape[1])
    else:
        orientation = "vertical"
        band0, band1 = map(int, rects[0]["bbox"][1::2])
        direction = 1 if guide_r == 0 else -1
        line_len = int(g.shape[0])

    blocks = []
    for rect in rects:
        r0, c0, r1, c1 = map(int, rect["bbox"])
        if orientation == "horizontal":
            assert (r0, r1) == (band0, band1)
            start, end = c0, c1
        else:
            assert (c0, c1) == (band0, band1)
            start, end = r0, r1
        f0, f1 = _to_forward_interval(start, end, line_len, direction)
        blocks.append(
            {
                "color": int(rect["color"]),
                "band0": band0,
                "band1": band1,
                "f0": int(f0),
                "f1": int(f1),
            }
        )

    blocks.sort(key=lambda block: block["f0"])
    for left, right in zip(blocks, blocks[1:]):
        assert int(left["f1"]) < int(right["f0"]) - 0, "rectangles must be separated"
        assert int(right["f0"]) - int(left["f1"]) > 1, "rectangles must have a nonzero gap"

    selected = [i for i, block in enumerate(blocks) if int(block["color"]) == guide_color]
    assert len(selected) == 1, "guide color must identify exactly one rectangle"

    return {
        "orientation": orientation,
        "direction": direction,
        "line_len": line_len,
        "band0": band0,
        "band1": band1,
        "guide": (guide_r, guide_c, guide_color),
        "blocks": blocks,
        "selected_idx": selected[0],
    }


def _render_scene_from_blocks(shape: tuple[int, int], orientation: str, direction: int, line_len: int, final_blocks: list[dict], moves: list[tuple[int, int, int, int, int]]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for idx, old0, old1, new0, new1 in moves:
        before = _block_bbox({**final_blocks[idx], "f0": old0, "f1": old1}, orientation, line_len, direction)
        after = _block_bbox({**final_blocks[idx], "f0": new0, "f1": new1}, orientation, line_len, direction)
        _paint_trail(out, before, after)
    for block in final_blocks:
        r0, c0, r1, c1 = _block_bbox(block, orientation, line_len, direction)
        out[r0 : r1 + 1, c0 : c1 + 1] = int(block["color"])
    return out


def _solve(grid: np.ndarray) -> np.ndarray:
    scene = _extract_scene(grid)
    final_blocks, moves = _simulate_forward_chain(scene["blocks"], scene["selected_idx"], scene["line_len"])
    return _render_scene_from_blocks(grid.shape, scene["orientation"], scene["direction"], scene["line_len"], final_blocks, moves)


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Relay a displacement quota along an aligned rectangle chain and paint every swept stripe orange.

    Input structure:
    - The grid has a black background, one corner guide pixel, and at least two separated solid colored rectangles.
    - All rectangles share one row band or one column band, with empty gaps between neighboring rectangles.
    - The guide pixel color matches exactly one rectangle color.

    Rule:
    - The guide color selects the first moving rectangle, and the motion direction is along the shared line away from the guide-side end.
    - The first rectangle slides until it touches the next rectangle; if there is no next rectangle, it slides to the border instead, and that slide distance becomes the initial quota when a collision happened.
    - Each later rectangle spends the received quota moving in the same direction: if it hits the next rectangle before using all quota, the unused remainder is passed on; if it uses the full quota first, the chain stops there; any rectangle that reaches the border ends the whole chain immediately.
    - Every cell swept by a moving rectangle becomes orange, final rectangle positions keep their original colors, and the guide pixel disappears.

    Key insight:
    - What propagates is a finite distance budget, not a global push-to-pack instruction, so later rectangles may stop inside empty gaps instead of compressing to the edge.
    """
    g = np.asarray(input_grid, dtype=int)
    return _solve(g)


def _random_partition(total: int, count: int, lo: int, hi: int) -> list[int] | None:
    if count * lo > total or count * hi < total:
        return None
    parts = [lo] * count
    extra = total - count * lo
    indices = list(range(count))
    while extra > 0:
        random.shuffle(indices)
        changed = False
        for idx in indices:
            if parts[idx] < hi:
                parts[idx] += 1
                extra -= 1
                changed = True
                if extra == 0:
                    break
        if not changed:
            return None
    random.shuffle(parts)
    return parts


def _relay_gap_sequence(start_quota: int, max_hops: int) -> tuple[list[int], int]:
    quota = int(start_quota)
    hops = random.randint(1, max_hops)
    gaps = []
    for _ in range(hops):
        if quota <= 1:
            break
        gap = random.randint(1, quota - 1)
        gaps.append(gap)
        quota -= gap
    return gaps, quota


def _choose_layout_parameters(mode: str) -> tuple[int, int, list[int], list[int], int, int]:
    mode = str(mode)
    if mode == "first_border":
        n = random.randint(2, 5)
        selected = n - 1
        sizes = [random.randint(2, 5) for _ in range(n)]
        gaps = [random.randint(1, 5) for _ in range(n - 1)]
        margin_end = random.randint(1, 6)
        return n, selected, sizes, gaps, random.randint(1, 5), margin_end

    if mode == "stop_mid":
        n = random.randint(2, 5)
        selected = random.randint(0, n - 2)
        sizes = [random.randint(2, 5) for _ in range(n)]
        gaps = [random.randint(1, 6) for _ in range(n - 1)]
        first_quota = random.randint(1, 5)
        gaps[selected] = first_quota
        if selected == n - 2 or random.random() < 0.5:
            selected = n - 2
            margin_end = random.randint(first_quota + 1, first_quota + 6)
        else:
            gaps[selected + 1] = random.randint(first_quota + 1, first_quota + 6)
            margin_end = random.randint(1, 6)
        return n, selected, sizes, gaps, random.randint(1, 5), margin_end

    if mode == "exact_touch":
        n = random.randint(3, 5)
        selected = random.randint(0, n - 3)
        sizes = [random.randint(2, 5) for _ in range(n)]
        gaps = [random.randint(1, 6) for _ in range(n - 1)]
        first_quota = random.randint(1, 5)
        gaps[selected] = first_quota
        gaps[selected + 1] = first_quota
        margin_end = random.randint(1, 6)
        return n, selected, sizes, gaps, random.randint(1, 5), margin_end

    if mode == "relay":
        n = random.randint(3, 5)
        selected = random.randint(0, n - 3)
        sizes = [random.randint(2, 5) for _ in range(n)]
        gaps = [random.randint(1, 6) for _ in range(n - 1)]
        first_quota = random.randint(2, 7)
        gaps[selected] = first_quota
        max_hops = min(2, n - selected - 2)
        relay_gaps, remaining = _relay_gap_sequence(first_quota, max_hops)
        for offset, gap in enumerate(relay_gaps, start=1):
            gaps[selected + offset] = gap
        stop_idx = selected + len(relay_gaps) + 1
        if stop_idx == n - 1:
            margin_end = random.randint(remaining + 1, remaining + 6)
        else:
            gaps[stop_idx] = random.randint(remaining + 1, remaining + 6)
            margin_end = random.randint(1, 6)
        return n, selected, sizes, gaps, random.randint(1, 5), margin_end

    assert mode == "later_border"
    n = random.randint(2, 5)
    selected = n - 2
    sizes = [random.randint(2, 5) for _ in range(n)]
    gaps = [random.randint(1, 6) for _ in range(n - 1)]
    first_quota = random.randint(2, 7)
    gaps[selected] = first_quota
    margin_end = random.randint(1, first_quota)
    return n, selected, sizes, gaps, random.randint(1, 5), margin_end


def _build_case_from_mode(mode: str) -> tuple[np.ndarray, np.ndarray, dict]:
    for _ in range(300):
        n, selected, sizes, gaps, margin0, margin_end = _choose_layout_parameters(mode)
        line_len = margin0 + sum(sizes) + sum(gaps) + margin_end
        if not (10 <= line_len <= 26):
            continue

        orientation = random.choice(("horizontal", "vertical"))
        thickness = random.randint(1, 3)
        if thickness == 1 and max(sizes) < 2:
            continue

        if orientation == "horizontal":
            h = random.randint(max(thickness + 3, 7), 18)
            w = line_len
            band0 = random.randint(1, h - thickness - 1)
            band1 = band0 + thickness - 1
            guide_r = random.choice((0, h - 1))
            guide_c = random.choice((0, w - 1))
            direction = 1 if guide_c == 0 else -1
        else:
            h = line_len
            w = random.randint(max(thickness + 3, 7), 18)
            band0 = random.randint(1, w - thickness - 1)
            band1 = band0 + thickness - 1
            guide_r = random.choice((0, h - 1))
            guide_c = random.choice((0, w - 1))
            direction = 1 if guide_r == 0 else -1

        if max(h, w) > 30:
            continue

        available_colors = random.sample(RECT_COLORS, n)
        selected_color = available_colors[selected]
        blocks = []
        pos = margin0
        for color, size in zip(available_colors, sizes):
            blocks.append({"color": int(color), "band0": band0, "band1": band1, "f0": pos, "f1": pos + size - 1})
            pos += size
            if len(blocks) < n:
                pos += gaps[len(blocks) - 1]

        inp = np.full((h, w), BG, dtype=int)
        for block in blocks:
            r0, c0, r1, c1 = _block_bbox(block, orientation, line_len, direction)
            inp[r0 : r1 + 1, c0 : c1 + 1] = int(block["color"])
        inp[guide_r, guide_c] = int(selected_color)

        final_blocks, moves = _simulate_forward_chain(blocks, selected, line_len)
        expected = _render_scene_from_blocks(inp.shape, orientation, direction, line_len, final_blocks, moves)

        metadata = {
            "mode": mode,
            "orientation": orientation,
            "direction": direction,
            "selected_idx": selected,
            "guide": (guide_r, guide_c),
            "blocks": [dict(block) for block in blocks],
            "moves": moves,
            "final_blocks": final_blocks,
        }
        if _validate_generated_case(inp, expected, metadata):
            return inp, expected, metadata
    raise RuntimeError(f"failed to build a valid {mode} case")


def _validate_generated_case(inp: np.ndarray, expected: np.ndarray, metadata: dict) -> bool:
    if not (isinstance(inp, np.ndarray) and isinstance(expected, np.ndarray)):
        return False
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if np.array_equal(inp, expected):
        return False
    if int(inp[metadata["guide"]]) != int(metadata["blocks"][metadata["selected_idx"]]["color"]):
        return False
    if int(expected[metadata["guide"]]) != BG:
        return False

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return False

    scene = _extract_scene(inp)
    if scene["orientation"] != metadata["orientation"] or scene["direction"] != metadata["direction"]:
        return False
    if scene["selected_idx"] != metadata["selected_idx"]:
        return False

    moved_count = len(metadata["moves"])
    if moved_count < 1:
        return False

    mode = metadata["mode"]
    final_blocks = metadata["final_blocks"]
    selected = metadata["selected_idx"]
    if mode == "first_border":
        if selected != len(final_blocks) - 1:
            return False
    elif mode == "stop_mid":
        last_move = metadata["moves"][-1]
        idx, _old0, old1, _new0, new1 = last_move
        if idx == len(final_blocks) - 1 and new1 == scene["line_len"] - 1:
            return False
        if idx + 1 < len(final_blocks) and new1 + 1 >= final_blocks[idx + 1]["f0"]:
            return False
    elif mode == "exact_touch":
        if moved_count < 2:
            return False
        idx, _old0, _old1, _new0, new1 = metadata["moves"][-1]
        if idx + 1 >= len(final_blocks):
            return False
        if new1 + 1 != final_blocks[idx + 1]["f0"]:
            return False
    elif mode == "relay":
        if moved_count < 2:
            return False
        quotas = []
        initial = scene["blocks"][selected + 1]["f0"] - scene["blocks"][selected]["f1"] - 1
        quotas.append(initial)
        for idx, old0, old1, new0, new1 in metadata["moves"][1:]:
            step = new0 - old0
            quotas.append(step)
            if step >= quotas[-2]:
                return False
    elif mode == "later_border":
        idx, _old0, _old1, _new0, new1 = metadata["moves"][-1]
        if idx != len(final_blocks) - 1 or new1 != scene["line_len"] - 1:
            return False

    return True


def generate_input() -> np.ndarray:
    mode = random.choices(MODES, weights=[1, 2, 2, 3, 2], k=1)[0]
    inp, _expected, _metadata = _build_case_from_mode(mode)
    return inp


if __name__ == "__main__":
    seen = set()
    mode_counts = {mode: 0 for mode in MODES}
    for _ in range(100):
        case_mode = random.choices(MODES, weights=[1, 2, 2, 3, 2], k=1)[0]
        inp, expected, metadata = _build_case_from_mode(case_mode)
        out = main(inp)
        assert isinstance(out, np.ndarray)
        assert out.ndim == 2 and max(out.shape) <= 30
        assert np.array_equal(out, expected)
        seen.add((inp.shape, inp.tobytes()))
        mode_counts[metadata["mode"]] += 1
    assert len(seen) >= 20
    assert sum(v > 0 for v in mode_counts.values()) >= 4
