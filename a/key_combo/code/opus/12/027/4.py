from __future__ import annotations

import itertools
import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

FRAME_COLORS = (BLUE, GREEN, YELLOW, ORANGE)
FRAME_SIZE = 4
GAP_OFFSETS = (1, 2)
PARTITIONS = ("tl", "tr", "bl", "br")
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ALLOWED_DIRS = {
    "tl": ("down", "right"),
    "tr": ("down", "left"),
    "bl": ("up", "right"),
    "br": ("up", "left"),
}
TARGET_PARTITION = {
    ("tl", "down"): "bl",
    ("tl", "right"): "tr",
    ("tr", "down"): "br",
    ("tr", "left"): "tl",
    ("bl", "up"): "tl",
    ("bl", "right"): "br",
    ("br", "up"): "tr",
    ("br", "left"): "bl",
}
MAX_GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _partition_bounds(height: int, width: int, sep_r: int, sep_c: int) -> dict[str, tuple[int, int, int, int]]:
    return {
        "tl": (0, sep_r - 1, 0, sep_c - 1),
        "tr": (0, sep_r - 1, sep_c + 1, width - 1),
        "bl": (sep_r + 1, height - 1, 0, sep_c - 1),
        "br": (sep_r + 1, height - 1, sep_c + 1, width - 1),
    }


def _frame_cells(r0: int, c0: int, gap: tuple[int, int]) -> set[tuple[int, int]]:
    cells = set()
    for r in range(r0, r0 + FRAME_SIZE):
        for c in range(c0, c0 + FRAME_SIZE):
            if r in (r0, r0 + FRAME_SIZE - 1) or c in (c0, c0 + FRAME_SIZE - 1):
                cells.add((r, c))
    cells.remove(gap)
    return cells


def _gap_from_bbox(r0: int, c0: int, direction: str, offset: int) -> tuple[int, int]:
    if direction == "up":
        return r0, c0 + offset
    if direction == "down":
        return r0 + FRAME_SIZE - 1, c0 + offset
    if direction == "left":
        return r0 + offset, c0
    return r0 + offset, c0 + FRAME_SIZE - 1


def _partition_id_for_bbox(bbox: tuple[int, int, int, int], sep_r: int, sep_c: int) -> str | None:
    r0, c0, r1, c1 = bbox
    if r1 < sep_r:
        row_half = "t"
    elif r0 > sep_r:
        row_half = "b"
    else:
        return None
    if c1 < sep_c:
        col_half = "l"
    elif c0 > sep_c:
        col_half = "r"
    else:
        return None
    return row_half + col_half


def _component_cells(grid: np.ndarray, mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_frames(grid: np.ndarray, sep_r: int, sep_c: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict] = []
    colors = sorted(int(color) for color in np.unique(g) if int(color) not in (BLACK, GREY))
    for color in colors:
        for comp in _component_cells(g, g == color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            bbox = (min(rows), min(cols), max(rows), max(cols))
            r0, c0, r1, c1 = bbox
            if (r1 - r0 + 1, c1 - c0 + 1) != (FRAME_SIZE, FRAME_SIZE):
                raise AssertionError("expected 4x4 frame bbox")
            if len(comp) != 11:
                raise AssertionError("expected a one-gap frame border")

            border = {
                (r, c)
                for r in range(r0, r1 + 1)
                for c in range(c0, c1 + 1)
                if r in (r0, r1) or c in (c0, c1)
            }
            coords = set(comp)
            gaps = list(border - coords)
            if len(gaps) != 1:
                raise AssertionError("expected exactly one frame gap")
            gap = gaps[0]
            if gap[0] == r0 and c0 < gap[1] < c1:
                direction = "up"
            elif gap[0] == r1 and c0 < gap[1] < c1:
                direction = "down"
            elif gap[1] == c0 and r0 < gap[0] < r1:
                direction = "left"
            elif gap[1] == c1 and r0 < gap[0] < r1:
                direction = "right"
            else:
                raise AssertionError("gap must be a non-corner border cell")

            partition = _partition_id_for_bbox(bbox, sep_r, sep_c)
            if partition is None:
                raise AssertionError("frame must lie fully inside one partition")

            dr, dc = DIRS[direction]
            rr, cc = gap[0] + dr, gap[1] + dc
            hit_separator = False
            while _in_bounds(g.shape, rr, cc):
                if g[rr, cc] == GREY:
                    hit_separator = True
                    break
                rr += dr
                cc += dc
            if not hit_separator:
                raise AssertionError("gap ray must reach a separator before leaving the grid")

            frames.append(
                {
                    "bbox": bbox,
                    "color": color,
                    "gap": gap,
                    "direction": direction,
                    "partition": partition,
                }
            )
    frames.sort(key=lambda frame: frame["bbox"])
    return frames


def _apply_transform(grid: np.ndarray, frames: list[dict]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        gap_r, gap_c = frame["gap"]
        dr, dc = DIRS[frame["direction"]]

        for rr in range(r0 + 1, r1):
            for cc in range(c0 + 1, c1):
                out[rr, cc] = RED
        out[gap_r, gap_c] = RED

        ray_len = 0
        rr, cc = gap_r + dr, gap_c + dc
        while _in_bounds(g.shape, rr, cc) and int(g[rr, cc]) != GREY:
            out[rr, cc] = RED
            ray_len += 1
            rr += dr
            cc += dc

        while _in_bounds(g.shape, rr, cc) and int(g[rr, cc]) == GREY:
            rr += dr
            cc += dc

        for _ in range(ray_len):
            if not _in_bounds(g.shape, rr, cc):
                break
            if int(out[rr, cc]) != BLACK:
                break
            out[rr, cc] = int(frame["color"])
            rr += dr
            cc += dc

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Measure each gap-to-separator ray and replay that length as a colored echo across the partition boundary.

    Input structure
    - Black background is split into four rectangular partitions by one full grey row and one full grey column.
    - Two or three partitions may contain a single 4x4 one-gap rectangular frame, each in a distinct non-grey color.
    - Every frame's missing border cell faces a separator, not the outer grid edge.

    Rule
    - Detect every one-gap frame, fill its 2x2 interior and the missing gap cell with red.
    - Extend a red ray outward from the gap through black cells until the next cell would be grey or outside the grid; the number of red cells drawn is the measured length.
    - Cross the separator into the adjacent partition and draw a same-direction echo stripe of that measured length in the frame's original color, stopping early if a non-black cell blocks it.

    Key insight
    - The gap gives a direction, the separator truncates that ray into a distance code, and that code is re-emitted as a same-axis stripe in the neighboring partition.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == GREY)]
    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    assert len(sep_rows) == 1 and len(sep_cols) == 1, "expected one horizontal and one vertical separator"
    frames = _find_frames(g, sep_rows[0], sep_cols[0])
    assert 1 <= len(frames) <= 3, "expected one to three frames"
    return _apply_transform(g, frames)


def _all_valid_configs() -> dict[str, list[tuple[tuple[str, str], ...]]]:
    groups = {"shared": [], "split": []}
    for size in (2, 3):
        for sources in itertools.combinations(PARTITIONS, size):
            for dirs in itertools.product(*(ALLOWED_DIRS[source] for source in sources)):
                config = tuple((source, direction) for source, direction in zip(sources, dirs))
                targets = [TARGET_PARTITION[item] for item in config]
                if any(target in sources for target in targets):
                    continue
                kind = "shared" if len(set(targets)) == 1 else "split"
                groups[kind].append(config)
    return groups


CONFIGS = _all_valid_configs()


def _enumerate_candidates(
    source_bounds: tuple[int, int, int, int],
    target_bounds: tuple[int, int, int, int],
    direction: str,
    color: int,
) -> list[dict]:
    rs, re, cs, ce = source_bounds
    trs, tre, tcs, tce = target_bounds
    candidates: list[dict] = []
    max_c0 = ce - FRAME_SIZE + 1
    max_r0 = re - FRAME_SIZE + 1

    for offset in GAP_OFFSETS:
        if direction == "up":
            max_len = re - rs - (FRAME_SIZE - 1)
            for ray_len in range(1, max_len + 1):
                r0 = rs + ray_len
                for c0 in range(cs, max_c0 + 1):
                    gap = (r0, c0 + offset)
                    ray_cells = [(gap[0] - step, gap[1]) for step in range(1, ray_len + 1)]
                    echo_cells = []
                    rr = tre
                    for _ in range(ray_len):
                        if rr < trs:
                            break
                        echo_cells.append((rr, gap[1]))
                        rr -= 1
                    candidates.append(
                        {
                            "bbox": (r0, c0, r0 + FRAME_SIZE - 1, c0 + FRAME_SIZE - 1),
                            "gap": gap,
                            "direction": direction,
                            "color": color,
                            "offset": offset,
                            "ray_len": ray_len,
                            "ray_cells": tuple(ray_cells),
                            "echo_cells": tuple(echo_cells),
                        }
                    )
        elif direction == "down":
            max_len = re - rs - (FRAME_SIZE - 1)
            for ray_len in range(1, max_len + 1):
                r0 = re - (FRAME_SIZE - 1) - ray_len
                for c0 in range(cs, max_c0 + 1):
                    gap = (r0 + FRAME_SIZE - 1, c0 + offset)
                    ray_cells = [(gap[0] + step, gap[1]) for step in range(1, ray_len + 1)]
                    echo_cells = []
                    rr = trs
                    for _ in range(ray_len):
                        if rr > tre:
                            break
                        echo_cells.append((rr, gap[1]))
                        rr += 1
                    candidates.append(
                        {
                            "bbox": (r0, c0, r0 + FRAME_SIZE - 1, c0 + FRAME_SIZE - 1),
                            "gap": gap,
                            "direction": direction,
                            "color": color,
                            "offset": offset,
                            "ray_len": ray_len,
                            "ray_cells": tuple(ray_cells),
                            "echo_cells": tuple(echo_cells),
                        }
                    )
        elif direction == "left":
            max_len = ce - cs - (FRAME_SIZE - 1)
            for ray_len in range(1, max_len + 1):
                c0 = cs + ray_len
                for r0 in range(rs, max_r0 + 1):
                    gap = (r0 + offset, c0)
                    ray_cells = [(gap[0], gap[1] - step) for step in range(1, ray_len + 1)]
                    echo_cells = []
                    cc = tce
                    for _ in range(ray_len):
                        if cc < tcs:
                            break
                        echo_cells.append((gap[0], cc))
                        cc -= 1
                    candidates.append(
                        {
                            "bbox": (r0, c0, r0 + FRAME_SIZE - 1, c0 + FRAME_SIZE - 1),
                            "gap": gap,
                            "direction": direction,
                            "color": color,
                            "offset": offset,
                            "ray_len": ray_len,
                            "ray_cells": tuple(ray_cells),
                            "echo_cells": tuple(echo_cells),
                        }
                    )
        else:
            max_len = ce - cs - (FRAME_SIZE - 1)
            for ray_len in range(1, max_len + 1):
                c0 = ce - (FRAME_SIZE - 1) - ray_len
                for r0 in range(rs, max_r0 + 1):
                    gap = (r0 + offset, c0 + FRAME_SIZE - 1)
                    ray_cells = [(gap[0], gap[1] + step) for step in range(1, ray_len + 1)]
                    echo_cells = []
                    cc = tcs
                    for _ in range(ray_len):
                        if cc > tce:
                            break
                        echo_cells.append((gap[0], cc))
                        cc += 1
                    candidates.append(
                        {
                            "bbox": (r0, c0, r0 + FRAME_SIZE - 1, c0 + FRAME_SIZE - 1),
                            "gap": gap,
                            "direction": direction,
                            "color": color,
                            "offset": offset,
                            "ray_len": ray_len,
                            "ray_cells": tuple(ray_cells),
                            "echo_cells": tuple(echo_cells),
                        }
                    )
    return candidates


def _render_expected(base: np.ndarray, frame_specs: list[dict]) -> np.ndarray:
    out = base.copy()
    for spec in sorted(frame_specs, key=lambda item: item["bbox"]):
        r0, c0, r1, c1 = spec["bbox"]
        gap_r, gap_c = spec["gap"]
        for rr in range(r0 + 1, r1):
            for cc in range(c0 + 1, c1):
                out[rr, cc] = RED
        out[gap_r, gap_c] = RED
        for rr, cc in spec["ray_cells"]:
            out[rr, cc] = RED
        for rr, cc in spec["echo_cells"]:
            if out[rr, cc] != BLACK:
                break
            out[rr, cc] = int(spec["color"])
    return out


def _semantic_check(grid: np.ndarray, expected: np.ndarray, frame_specs: list[dict], sep_r: int, sep_c: int) -> bool:
    if np.array_equal(grid, expected):
        return False

    bounds = _partition_bounds(grid.shape[0], grid.shape[1], sep_r, sep_c)
    occupied_sources = {spec["partition"] for spec in frame_specs}
    discovered = _find_frames(grid, sep_r, sep_c)
    if len(discovered) != len(frame_specs):
        return False

    discovered_by_partition = {frame["partition"]: frame for frame in discovered}
    for spec in frame_specs:
        target = spec["target"]
        if target in occupied_sources:
            return False
        if spec["partition"] not in discovered_by_partition:
            return False
        if discovered_by_partition[spec["partition"]]["color"] != spec["color"]:
            return False
        trs, tre, tcs, tce = bounds[target]
        region = grid[trs : tre + 1, tcs : tce + 1]
        if np.any((region != BLACK) & (region != GREY)):
            return False
        if spec["ray_len"] != len(spec["ray_cells"]):
            return False
        if len(spec["echo_cells"]) > spec["ray_len"]:
            return False

    used = set()
    for spec in frame_specs:
        echo = set(spec["echo_cells"])
        if echo & used:
            return False
        used |= echo
    return True


def _gen_attempt() -> np.ndarray | None:
    top_h = random.randint(7, 11)
    bot_h = random.randint(7, 11)
    left_w = random.randint(7, 11)
    right_w = random.randint(7, 11)
    sep_r = top_h
    sep_c = left_w
    height = top_h + 1 + bot_h
    width = left_w + 1 + right_w

    grid = np.full((height, width), BLACK, dtype=int)
    grid[sep_r, :] = GREY
    grid[:, sep_c] = GREY
    bounds = _partition_bounds(height, width, sep_r, sep_c)

    branch = random.choice(["shared", "split"])
    configs = CONFIGS[branch] if CONFIGS[branch] else CONFIGS["shared"] + CONFIGS["split"]
    config = random.choice(configs)
    colors = random.sample(FRAME_COLORS, len(config))
    require_truncation = random.random() < 0.45

    frame_specs: list[dict] = []
    used_echo_cells: set[tuple[int, int]] = set()
    horizontal_rows: dict[str, set[int]] = {part: set() for part in PARTITIONS}
    vertical_cols: dict[str, set[int]] = {part: set() for part in PARTITIONS}
    any_truncated = False

    for (partition, direction), color in zip(config, colors):
        target = TARGET_PARTITION[(partition, direction)]
        candidates = _enumerate_candidates(bounds[partition], bounds[target], direction, color)
        filtered = []
        for cand in candidates:
            echo_cells = set(cand["echo_cells"])
            if echo_cells & used_echo_cells:
                continue
            gap_r, gap_c = cand["gap"]
            if direction in ("left", "right") and gap_r in horizontal_rows[target]:
                continue
            if direction in ("up", "down") and gap_c in vertical_cols[target]:
                continue
            filtered.append(cand)

        if not filtered:
            return None

        truncated = [cand for cand in filtered if len(cand["echo_cells"]) < cand["ray_len"]]
        non_truncated = [cand for cand in filtered if len(cand["echo_cells"]) == cand["ray_len"]]
        if require_truncation and not any_truncated and truncated:
            choice_pool = truncated
        elif non_truncated and random.random() < 0.65:
            choice_pool = non_truncated
        else:
            choice_pool = filtered
        spec = random.choice(choice_pool).copy()
        spec["partition"] = partition
        spec["target"] = target
        spec["frame_cells"] = _frame_cells(spec["bbox"][0], spec["bbox"][1], spec["gap"])

        for rr, cc in spec["frame_cells"]:
            if grid[rr, cc] != BLACK:
                return None
            grid[rr, cc] = color

        frame_specs.append(spec)
        used_echo_cells.update(spec["echo_cells"])
        if direction in ("left", "right"):
            horizontal_rows[target].add(spec["gap"][0])
        else:
            vertical_cols[target].add(spec["gap"][1])
        any_truncated = any_truncated or len(spec["echo_cells"]) < spec["ray_len"]

    if require_truncation and not any_truncated:
        return None

    expected = _render_expected(grid, frame_specs)
    if not _semantic_check(grid, expected, frame_specs, sep_r, sep_c):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gap-ray echo input")
