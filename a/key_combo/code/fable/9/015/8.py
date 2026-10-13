from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random

import numpy as np


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

PALETTE = (BLUE, RED, GREEN, YELLOW, GREY, PINK, ORANGE, TEAL, MAROON)
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATION_ATTEMPTS = 1200


@dataclass
class Block:
    r: int
    c: int
    h: int
    w: int
    body_color: int
    port_color: int
    port_offset: tuple[int, int]
    direction: tuple[int, int]

    def shifted(self, dr: int, dc: int) -> "Block":
        return Block(
            r=self.r + int(dr),
            c=self.c + int(dc),
            h=self.h,
            w=self.w,
            body_color=self.body_color,
            port_color=self.port_color,
            port_offset=self.port_offset,
            direction=self.direction,
        )


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _rectangles_overlap(a: Block, b: Block, *, a_shift: tuple[int, int] = (0, 0), b_shift: tuple[int, int] = (0, 0)) -> bool:
    ar0 = a.r + int(a_shift[0])
    ac0 = a.c + int(a_shift[1])
    ar1 = ar0 + a.h - 1
    ac1 = ac0 + a.w - 1
    br0 = b.r + int(b_shift[0])
    bc0 = b.c + int(b_shift[1])
    br1 = br0 + b.h - 1
    bc1 = bc0 + b.w - 1
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _render(blocks: list[Block], shape: tuple[int, int], bg: int) -> np.ndarray:
    h, w = map(int, shape)
    out = np.full((h, w), int(bg), dtype=int)
    for block in blocks:
        assert 0 <= block.r and 0 <= block.c
        assert block.r + block.h <= h and block.c + block.w <= w
        patch = out[block.r : block.r + block.h, block.c : block.c + block.w]
        assert np.all(patch == bg), "blocks must not overlap"
        patch[:, :] = int(block.body_color)
    for block in blocks:
        pr, pc = block.port_offset
        out[block.r + pr, block.c + pc] = int(block.port_color)
    return out


def _parse_blocks(grid: np.ndarray) -> tuple[list[Block], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _get_bg(g)
    comps = _components(g != bg)
    assert comps, "expected at least one foreground block"

    blocks: list[Block] = []
    port_colors: set[int] = set()
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch != bg), "each block bbox must be fully filled"
        colors, counts = np.unique(patch, return_counts=True)
        assert len(colors) == 2, "each block must contain one body color and one port color"
        order = np.argsort(counts)
        port_color = int(colors[order[0]])
        body_color = int(colors[order[1]])
        assert int(counts[order[0]]) == 1, "each block must have exactly one port pixel"
        port_pos = np.argwhere(patch == port_color)
        assert port_pos.shape == (1, 2)
        pr, pc = map(int, port_pos[0])
        side_dirs = []
        if pr == 0:
            side_dirs.append((-1, 0))
            assert 0 < pc < patch.shape[1] - 1
        if pr == patch.shape[0] - 1:
            side_dirs.append((1, 0))
            assert 0 < pc < patch.shape[1] - 1
        if pc == 0:
            side_dirs.append((0, -1))
            assert 0 < pr < patch.shape[0] - 1
        if pc == patch.shape[1] - 1:
            side_dirs.append((0, 1))
            assert 0 < pr < patch.shape[0] - 1
        assert len(side_dirs) == 1, "port must lie on exactly one side, not a corner"
        direction = side_dirs[0]
        port_colors.add(port_color)
        blocks.append(
            Block(
                r=r0,
                c=c0,
                h=patch.shape[0],
                w=patch.shape[1],
                body_color=body_color,
                port_color=port_color,
                port_offset=(pr, pc),
                direction=direction,
            )
        )

    assert len(port_colors) == 1, "all blocks must share the same port color"
    return blocks, bg


def _closure_for_step(blocks: list[Block], start_idx: int, shape: tuple[int, int]) -> set[int] | None:
    h, w = map(int, shape)
    dr, dc = blocks[start_idx].direction
    chain = {int(start_idx)}
    while True:
        for idx in chain:
            block = blocks[idx]
            nr = block.r + dr
            nc = block.c + dc
            if nr < 0 or nc < 0 or nr + block.h > h or nc + block.w > w:
                return None
        changed = False
        for idx in tuple(chain):
            for other_idx, other in enumerate(blocks):
                if other_idx in chain:
                    continue
                if _rectangles_overlap(blocks[idx], other, a_shift=(dr, dc)):
                    chain.add(other_idx)
                    changed = True
        if not changed:
            return chain


def _move_one_turn(blocks: list[Block], idx: int, shape: tuple[int, int], stats: dict | None = None) -> bool:
    dr, dc = blocks[idx].direction
    moved = False
    while True:
        chain = _closure_for_step(blocks, idx, shape)
        if chain is None:
            if stats is not None:
                stats["blocked_turns"] += 1
            return moved
        if stats is not None and len(chain) > 1:
            stats["push_steps"] += 1
            stats["max_chain"] = max(int(stats["max_chain"]), len(chain))
            stats["passive_objects"].update(chain - {idx})
        for member in chain:
            blocks[member] = blocks[member].shifted(dr, dc)
            if stats is not None:
                stats["net_displacement"][member] = (
                    stats["net_displacement"][member][0] + dr,
                    stats["net_displacement"][member][1] + dc,
                )
        if stats is not None:
            stats["step_count"] += 1
        moved = True


def _simulate(blocks: list[Block], shape: tuple[int, int], bg: int, *, round_limit: int | None = None, collect: bool = False) -> tuple[list[Block], np.ndarray, dict]:
    current = [block.shifted(0, 0) for block in blocks]
    stats = {
        "step_count": 0,
        "push_steps": 0,
        "blocked_turns": 0,
        "moving_rounds": 0,
        "executed_rounds": 0,
        "passive_objects": set(),
        "max_chain": 1,
        "net_displacement": [(0, 0) for _ in current],
    }
    max_rounds = max(1, shape[0] * shape[1] * max(1, len(current)))
    for _ in range(max_rounds):
        order = sorted(range(len(current)), key=lambda i: (current[i].r, current[i].c, i))
        round_moved = False
        for idx in order:
            if _move_one_turn(current, idx, shape, stats if collect else None):
                round_moved = True
        stats["executed_rounds"] += 1
        if round_moved:
            stats["moving_rounds"] += 1
        if round_limit is not None and stats["executed_rounds"] >= round_limit:
            break
        if not round_moved:
            break
    else:
        raise AssertionError("simulation did not reach a fixpoint")
    return current, _render(current, shape, bg), stats


def _move_one_turn_without_push(blocks: list[Block], idx: int, shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    dr, dc = blocks[idx].direction
    moved = False
    while True:
        block = blocks[idx]
        nr = block.r + dr
        nc = block.c + dc
        if nr < 0 or nc < 0 or nr + block.h > h or nc + block.w > w:
            return moved
        blocked = False
        trial = block.shifted(dr, dc)
        for other_idx, other in enumerate(blocks):
            if other_idx == idx:
                continue
            if _rectangles_overlap(trial, other):
                blocked = True
                break
        if blocked:
            return moved
        blocks[idx] = trial
        moved = True


def _simulate_without_push(blocks: list[Block], shape: tuple[int, int], bg: int) -> np.ndarray:
    current = [block.shifted(0, 0) for block in blocks]
    max_rounds = max(1, shape[0] * shape[1] * max(1, len(current)))
    for _ in range(max_rounds):
        order = sorted(range(len(current)), key=lambda i: (current[i].r, current[i].c, i))
        round_moved = False
        for idx in order:
            if _move_one_turn_without_push(current, idx, shape):
                round_moved = True
        if not round_moved:
            return _render(current, shape, bg)
    raise AssertionError("no-push simulation did not settle")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each port-marked rectangle in its own port direction, letting collisions transmit motion through push chains until the whole canvas settles.

    Input structure:
    - The background is the most common color.
    - Each foreground object is a solid monochrome rectangle whose bounding box is fully filled.
    - Every rectangle contains exactly one shared-color port pixel on one non-corner edge cell, and disconnected rectangles do not overlap.

    Rule:
    - The side containing the port sets that rectangle's driving direction: top means up, bottom means down, left means left, and right means right.
    - Repeatedly scan the rectangles in top-to-bottom then left-to-right order of their current top-left corners.
    - On its turn, a rectangle advances one cell at a time in its own direction; if that step would enter another rectangle, it pushes that rectangle and every recursively contacted rectangle one cell the same way, and the whole chain stops only when its front would leave the grid.
    - Pushed rectangles keep their own ports and future driving directions. Output the first state after a full scan round in which nothing moves.

    Key insight:
    - A collision is not a blocker by itself: it grows a same-direction push chain, so the final arrangement depends on both per-block port directions and repeated round-by-round re-evaluation after earlier pushes change later opportunities.
    """
    blocks, bg = _parse_blocks(np.asarray(grid, dtype=int))
    _settled, out, _stats = _simulate(blocks, grid.shape, bg, collect=False)
    return out


def _direction_name(direction: tuple[int, int]) -> str:
    return {
        (-1, 0): "up",
        (1, 0): "down",
        (0, -1): "left",
        (0, 1): "right",
    }[direction]


def _random_dimensions() -> tuple[int, int]:
    return random.randint(3, 4), random.randint(3, 4)


def _sample_port_offset(h: int, w: int, direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    if dr == -1:
        return 0, random.randint(1, w - 2)
    if dr == 1:
        return h - 1, random.randint(1, w - 2)
    if dc == -1:
        return random.randint(1, h - 2), 0
    return random.randint(1, h - 2), w - 1


def _make_block(r: int, c: int, h: int, w: int, body_color: int, port_color: int, direction: tuple[int, int]) -> Block:
    return Block(
        r=int(r),
        c=int(c),
        h=int(h),
        w=int(w),
        body_color=int(body_color),
        port_color=int(port_color),
        port_offset=_sample_port_offset(int(h), int(w), direction),
        direction=direction,
    )


def _normalize_group(blocks: list[Block]) -> list[Block]:
    min_r = min(block.r for block in blocks)
    min_c = min(block.c for block in blocks)
    return [block.shifted(-min_r, -min_c) for block in blocks]


def _group_bbox(blocks: list[Block]) -> tuple[int, int]:
    max_r = max(block.r + block.h for block in blocks)
    max_c = max(block.c + block.w for block in blocks)
    return max_r, max_c


def _build_lane(count: int, orientation: str, colors: list[int], port_color: int, *, emphasize_revisit: bool) -> tuple[list[Block], tuple[int, int]]:
    lane_direction = random.choice(((0, 1), (0, -1))) if orientation == "horizontal" else random.choice(((1, 0), (-1, 0)))
    active_idx = 0 if lane_direction in ((0, 1), (1, 0)) else count - 1
    heights: list[int] = []
    widths: list[int] = []
    for _ in range(count):
        h, w = _random_dimensions()
        heights.append(h)
        widths.append(w)

    blocks: list[Block] = []
    shared_coord = 1
    if orientation == "horizontal":
        tops = []
        for idx in range(count):
            low = shared_coord - heights[idx] + 1
            high = shared_coord
            if emphasize_revisit and idx == active_idx:
                top = high
            elif emphasize_revisit and idx != active_idx and idx == (active_idx + 1 if active_idx + 1 < count else active_idx - 1):
                top = low
            else:
                top = random.randint(low, high)
            tops.append(top)
        lefts = [0]
        for idx in range(1, count):
            lefts.append(lefts[-1] + widths[idx - 1] + random.randint(1, 2))
        if lane_direction == (0, -1):
            lefts = list(lefts)
        other_dirs = []
        for idx in range(count):
            if idx == active_idx:
                other_dirs.append(lane_direction)
                continue
            choices = list(DIRECTIONS)
            if random.random() < 0.8:
                choices = [d for d in choices if d != lane_direction]
            other_dirs.append(random.choice(choices))
        if all(direction == lane_direction for direction in other_dirs):
            swap_idx = 0 if active_idx != 0 else 1
            other_dirs[swap_idx] = random.choice([d for d in DIRECTIONS if d != lane_direction])
        for idx in range(count):
            blocks.append(_make_block(tops[idx], lefts[idx], heights[idx], widths[idx], colors[idx], port_color, other_dirs[idx]))
    else:
        lefts = []
        for idx in range(count):
            low = shared_coord - widths[idx] + 1
            high = shared_coord
            if emphasize_revisit and idx == active_idx:
                left = high
            elif emphasize_revisit and idx != active_idx and idx == (active_idx + 1 if active_idx + 1 < count else active_idx - 1):
                left = low
            else:
                left = random.randint(low, high)
            lefts.append(left)
        tops = [0]
        for idx in range(1, count):
            tops.append(tops[-1] + heights[idx - 1] + random.randint(1, 2))
        other_dirs = []
        for idx in range(count):
            if idx == active_idx:
                other_dirs.append(lane_direction)
                continue
            choices = list(DIRECTIONS)
            if random.random() < 0.8:
                choices = [d for d in choices if d != lane_direction]
            other_dirs.append(random.choice(choices))
        if all(direction == lane_direction for direction in other_dirs):
            swap_idx = 0 if active_idx != 0 else 1
            other_dirs[swap_idx] = random.choice([d for d in DIRECTIONS if d != lane_direction])
        for idx in range(count):
            blocks.append(_make_block(tops[idx], lefts[idx], heights[idx], widths[idx], colors[idx], port_color, other_dirs[idx]))

    blocks = _normalize_group(blocks)
    return blocks, lane_direction


def _build_singleton(color: int, port_color: int) -> list[Block]:
    h, w = _random_dimensions()
    direction = random.choice(DIRECTIONS)
    return [_make_block(0, 0, h, w, color, port_color, direction)]


def _bboxes_have_gap(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], gap: int) -> bool:
    return box_a[2] + gap < box_b[0] or box_b[2] + gap < box_a[0] or box_a[3] + gap < box_b[1] or box_b[3] + gap < box_a[1]


def _shift_group(blocks: list[Block], dr: int, dc: int) -> list[Block]:
    return [block.shifted(dr, dc) for block in blocks]


def _blocks_bbox(blocks: list[Block]) -> tuple[int, int, int, int]:
    r0 = min(block.r for block in blocks)
    c0 = min(block.c for block in blocks)
    r1 = max(block.r + block.h - 1 for block in blocks)
    c1 = max(block.c + block.w - 1 for block in blocks)
    return r0, c0, r1, c1


def _place_groups(groups: list[dict], shape: tuple[int, int]) -> list[Block] | None:
    h, w = map(int, shape)
    placed: list[list[Block]] = []
    placed_boxes: list[tuple[int, int, int, int]] = []
    for group in groups:
        local = group["blocks"]
        gh, gw = _group_bbox(local)
        attempts = 200
        for _ in range(attempts):
            if group.get("mandatory", False):
                dr, dc = group["lane_direction"]
                if dr == -1:
                    top = random.randint(0, min(2, h - gh))
                    left = random.randint(0, w - gw)
                elif dr == 1:
                    top = random.randint(max(0, h - gh - 2), h - gh)
                    left = random.randint(0, w - gw)
                elif dc == -1:
                    top = random.randint(0, h - gh)
                    left = random.randint(0, min(2, w - gw))
                else:
                    top = random.randint(0, h - gh)
                    left = random.randint(max(0, w - gw - 2), w - gw)
            else:
                top = random.randint(0, h - gh)
                left = random.randint(0, w - gw)

            shifted = _shift_group(local, top, left)
            bbox = _blocks_bbox(shifted)
            if all(_bboxes_have_gap(bbox, other, 2) for other in placed_boxes):
                placed.append(shifted)
                placed_boxes.append(bbox)
                break
        else:
            return None

    merged: list[Block] = []
    for group_blocks in placed:
        merged.extend(group_blocks)
    return merged


def _grid_signature(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(grid, dtype=np.int8)
    return tuple(map(int, arr.shape)), arr.tobytes()


def _validate_generated_case(blocks: list[Block], shape: tuple[int, int], bg: int) -> np.ndarray | None:
    inp = _render(blocks, shape, bg)
    parsed, parsed_bg = _parse_blocks(inp)
    assert parsed_bg == bg
    try:
        _, final_out, stats = _simulate(parsed, shape, bg, collect=True)
    except AssertionError:
        return None

    if np.array_equal(inp, final_out):
        return None
    if int(stats["push_steps"]) == 0:
        return None
    if not stats["passive_objects"]:
        return None
    if int(stats["moving_rounds"]) < 2:
        return None
    if int(stats["max_chain"]) < 2:
        return None
    if len({_direction_name(block.direction) for block in parsed}) < 2:
        return None

    displacements = [tuple(map(int, delta)) for delta in stats["net_displacement"]]
    if not any(delta != (0, 0) for delta in displacements):
        return None
    if not any(delta not in {(0, 0), block.direction} for block, delta in zip(parsed, displacements)):
        return None
    if len({_grid_signature(inp), _grid_signature(final_out)}) < 2:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        port_color = random.choice(PALETTE)
        colors = [color for color in PALETTE if color != port_color]
        random.shuffle(colors)

        mandatory_count = 2
        extra_kind = "single" if random.random() < 0.7 else "lane"
        extra_count = 2 if extra_kind == "lane" else 1
        total = mandatory_count + extra_count
        body_colors = colors[:total]

        mandatory_blocks, lane_direction = _build_lane(
            mandatory_count,
            random.choice(("horizontal", "vertical")),
            body_colors[:mandatory_count],
            port_color,
            emphasize_revisit=True,
        )

        groups = [{"blocks": mandatory_blocks, "lane_direction": lane_direction, "mandatory": True}]
        if extra_kind == "lane":
            extra_blocks, extra_lane_direction = _build_lane(
                2,
                random.choice(("horizontal", "vertical")),
                body_colors[mandatory_count:mandatory_count + 2],
                port_color,
                emphasize_revisit=False,
            )
            groups.append({"blocks": extra_blocks, "lane_direction": extra_lane_direction, "mandatory": False})
        else:
            groups.append(
                {
                    "blocks": _build_singleton(body_colors[mandatory_count], port_color),
                    "lane_direction": random.choice(DIRECTIONS),
                    "mandatory": False,
                }
            )

        max_h = max(_group_bbox(group["blocks"])[0] for group in groups)
        max_w = max(_group_bbox(group["blocks"])[1] for group in groups)
        height_low = max(14, max_h + 3)
        width_low = max(14, max_w + 3)
        if height_low > 17 or width_low > 17:
            continue
        height = random.randint(height_low, 17)
        width = random.randint(width_low, 17)

        placed = _place_groups(groups, (height, width))
        if placed is None:
            continue

        candidate = _validate_generated_case(placed, (height, width), BLACK)
        if candidate is not None:
            return candidate

    raise RuntimeError("failed to generate a valid port-push-chain input")
