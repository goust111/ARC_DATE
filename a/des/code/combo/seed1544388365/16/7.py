from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
FRAME_SIZE = 4
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (16, 24)
GRID_W_RANGE = (16, 24)
SOURCE_COUNT_RANGE = (2, 4)
FRAME_COUNT_RANGE = (4, 7)
SOURCE_SPACING = 2
GEN_RETRIES = 500

PERIMETER_RELS = tuple(
    (r, c)
    for r in range(FRAME_SIZE)
    for c in range(FRAME_SIZE)
    if r in (0, FRAME_SIZE - 1) or c in (0, FRAME_SIZE - 1)
)
GAP_RELS_BY_DIR = {
    "up": ((0, 1), (0, 2)),
    "down": ((3, 1), (3, 2)),
    "left": ((1, 0), (2, 0)),
    "right": ((1, 3), (2, 3)),
}
GAP_DIR_BY_REL = {rel: direction for direction, rels in GAP_RELS_BY_DIR.items() for rel in rels}
STEP_BY_DIR = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
HIT_RELS_BY_VIEW_DIR = {
    "up": tuple((3, c) for c in range(FRAME_SIZE)),
    "down": tuple((0, c) for c in range(FRAME_SIZE)),
    "left": tuple((r, 3) for r in range(FRAME_SIZE)),
    "right": tuple((r, 0) for r in range(FRAME_SIZE)),
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _frame_bbox_cells(r0: int, c0: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + FRAME_SIZE) for c in range(c0, c0 + FRAME_SIZE)}


def _expanded_bbox_cells(r0: int, c0: int, shape: tuple[int, int], pad: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    cells: set[tuple[int, int]] = set()
    for r in range(max(0, r0 - pad), min(h, r0 + FRAME_SIZE + pad)):
        for c in range(max(0, c0 - pad), min(w, c0 + FRAME_SIZE + pad)):
            cells.add((r, c))
    return cells


def _border_cells(r0: int, c0: int, gap_rel: tuple[int, int]) -> set[tuple[int, int]]:
    return {(r0 + dr, c0 + dc) for dr, dc in PERIMETER_RELS if (dr, dc) != gap_rel}


def _ray_cells(gap: tuple[int, int], direction: str, shape: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = STEP_BY_DIR[direction]
    r, c = gap[0] + dr, gap[1] + dc
    out: list[tuple[int, int]] = []
    while _in_bounds(shape, r, c):
        out.append((r, c))
        r += dr
        c += dc
    return out


def _ray_prefix_to_target(
    gap: tuple[int, int], direction: str, target: tuple[int, int], shape: tuple[int, int]
) -> list[tuple[int, int]] | None:
    dr, dc = STEP_BY_DIR[direction]
    r, c = gap[0] + dr, gap[1] + dc
    out: list[tuple[int, int]] = []
    while _in_bounds(shape, r, c):
        if (r, c) == target:
            return out
        out.append((r, c))
        r += dr
        c += dc
    return None


def _scan_first_nonblack(grid: np.ndarray, gap: tuple[int, int], direction: str) -> tuple[int, int] | None:
    dr, dc = STEP_BY_DIR[direction]
    r, c = gap[0] + dr, gap[1] + dc
    h, w = grid.shape
    while 0 <= r < h and 0 <= c < w:
        if int(grid[r, c]) != BLACK:
            return int(r), int(c)
        r += dr
        c += dc
    return None


def _make_frame(
    r0: int,
    c0: int,
    gap_dir: str,
    gap_rel: tuple[int, int],
    target_kind: str,
    target_ref: int | None,
    target_pos: tuple[int, int] | None,
    shape: tuple[int, int],
) -> dict:
    gap = (r0 + gap_rel[0], c0 + gap_rel[1])
    return {
        "r0": int(r0),
        "c0": int(c0),
        "gap_dir": gap_dir,
        "gap_rel": gap_rel,
        "gap": gap,
        "border": _border_cells(r0, c0, gap_rel),
        "bbox": _frame_bbox_cells(r0, c0),
        "bbox_block": _expanded_bbox_cells(r0, c0, shape),
        "target_kind": target_kind,
        "target_ref": target_ref,
        "target_pos": target_pos,
    }


def _find_frames(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int]]:
    h, w = grid.shape
    frames: list[dict] = []
    border_lookup: dict[tuple[int, int], int] = {}

    for r0 in range(h - FRAME_SIZE + 1):
        for c0 in range(w - FRAME_SIZE + 1):
            block = grid[r0 : r0 + FRAME_SIZE, c0 : c0 + FRAME_SIZE]
            if not np.all(block[1:3, 1:3] == BLACK):
                continue
            if np.any((block != BLACK) & (block != GREY)):
                continue

            black_perimeter = [(dr, dc) for dr, dc in PERIMETER_RELS if int(block[dr, dc]) == BLACK]
            if len(black_perimeter) != 1:
                continue
            gap_rel = black_perimeter[0]
            if gap_rel not in GAP_DIR_BY_REL:
                continue
            if any(int(block[dr, dc]) != GREY for dr, dc in PERIMETER_RELS if (dr, dc) != gap_rel):
                continue

            border = _border_cells(r0, c0, gap_rel)
            if any(cell in border_lookup for cell in border):
                continue

            idx = len(frames)
            for cell in border:
                border_lookup[cell] = idx
            frames.append(
                {
                    "r0": int(r0),
                    "c0": int(c0),
                    "gap_rel": gap_rel,
                    "gap_dir": GAP_DIR_BY_REL[gap_rel],
                    "gap": (int(r0 + gap_rel[0]), int(c0 + gap_rel[1])),
                }
            )

    return frames, border_lookup


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow each frame's gap direction; the first seen object or source determines its interior color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames, border_lookup = _find_frames(g)
    out = g.copy()
    if not frames:
        return out

    state = [0] * len(frames)
    resolved = [BLACK] * len(frames)

    def resolve(idx: int) -> int:
        if state[idx] == 2:
            return int(resolved[idx])
        if state[idx] == 1:
            return BLACK

        state[idx] = 1
        frame = frames[idx]
        hit = _scan_first_nonblack(g, frame["gap"], frame["gap_dir"])
        color = BLACK
        if hit is not None:
            value = int(g[hit])
            if value == GREY:
                target = border_lookup.get(hit)
                color = resolve(target) if target is not None else BLACK
            else:
                color = value

        resolved[idx] = int(color)
        state[idx] = 2
        return int(color)

    for idx, frame in enumerate(frames):
        color = resolve(idx)
        if color != BLACK:
            r0 = int(frame["r0"])
            c0 = int(frame["c0"])
            out[r0 + 1 : r0 + 3, c0 + 1 : c0 + 3] = color

    return out


def _valid_source_positions(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    cells = [(0, c) for c in range(2, w - 2)]
    cells += [(h - 1, c) for c in range(2, w - 2)]
    cells += [(r, 0) for r in range(2, h - 2)]
    cells += [(r, w - 1) for r in range(2, h - 2)]
    return cells


def _source_positions(shape: tuple[int, int], count: int, rng: random.Random) -> list[tuple[int, int]] | None:
    candidates = _valid_source_positions(shape)
    rng.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for pos in candidates:
        if all(max(abs(pos[0] - r), abs(pos[1] - c)) >= SOURCE_SPACING for r, c in chosen):
            chosen.append(pos)
            if len(chosen) == count:
                return chosen
    return None


def _frame_hit_pixels(frame: dict, view_dir: str) -> list[tuple[int, int]]:
    gap_rel = frame["gap_rel"]
    r0 = int(frame["r0"])
    c0 = int(frame["c0"])
    pixels = []
    for rel in HIT_RELS_BY_VIEW_DIR[view_dir]:
        if rel != gap_rel:
            pixels.append((r0 + rel[0], c0 + rel[1]))
    return pixels


def _bbox_ok(r0: int, c0: int, shape: tuple[int, int], bbox_blocked: set[tuple[int, int]], ray_blocked: set[tuple[int, int]]) -> bool:
    h, w = shape
    if not (1 <= r0 <= h - FRAME_SIZE - 1 and 1 <= c0 <= w - FRAME_SIZE - 1):
        return False
    bbox = _frame_bbox_cells(r0, c0)
    return bbox.isdisjoint(bbox_blocked) and bbox.isdisjoint(ray_blocked)


def _path_clear(path: list[tuple[int, int]], occupied: set[tuple[int, int]], ray_blocked: set[tuple[int, int]]) -> bool:
    return set(path).isdisjoint(occupied) and set(path).isdisjoint(ray_blocked)


def _source_candidates(
    shape: tuple[int, int],
    source_idx: int,
    source: dict,
    bbox_blocked: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    ray_blocked: set[tuple[int, int]],
) -> list[dict]:
    h, w = shape
    tr, tc = source["pos"]
    candidates: list[dict] = []

    if tr == 0:
        direction = "up"
        for gap_rel in GAP_RELS_BY_DIR[direction]:
            c0 = tc - gap_rel[1]
            for r0 in range(1, h - FRAME_SIZE):
                if not _bbox_ok(r0, c0, shape, bbox_blocked, ray_blocked):
                    continue
                gap = (r0 + gap_rel[0], c0 + gap_rel[1])
                path = _ray_prefix_to_target(gap, direction, source["pos"], shape)
                if path is None or not _path_clear(path, occupied, ray_blocked):
                    continue
                candidates.append(
                    {
                        "r0": r0,
                        "c0": c0,
                        "gap_dir": direction,
                        "gap_rel": gap_rel,
                        "target_kind": "source",
                        "target_ref": source_idx,
                        "target_pos": source["pos"],
                    }
                )

    if tr == h - 1:
        direction = "down"
        for gap_rel in GAP_RELS_BY_DIR[direction]:
            c0 = tc - gap_rel[1]
            for r0 in range(1, h - FRAME_SIZE):
                if not _bbox_ok(r0, c0, shape, bbox_blocked, ray_blocked):
                    continue
                gap = (r0 + gap_rel[0], c0 + gap_rel[1])
                path = _ray_prefix_to_target(gap, direction, source["pos"], shape)
                if path is None or not _path_clear(path, occupied, ray_blocked):
                    continue
                candidates.append(
                    {
                        "r0": r0,
                        "c0": c0,
                        "gap_dir": direction,
                        "gap_rel": gap_rel,
                        "target_kind": "source",
                        "target_ref": source_idx,
                        "target_pos": source["pos"],
                    }
                )

    if tc == 0:
        direction = "left"
        for gap_rel in GAP_RELS_BY_DIR[direction]:
            r0 = tr - gap_rel[0]
            for c0 in range(1, w - FRAME_SIZE):
                if not _bbox_ok(r0, c0, shape, bbox_blocked, ray_blocked):
                    continue
                gap = (r0 + gap_rel[0], c0 + gap_rel[1])
                path = _ray_prefix_to_target(gap, direction, source["pos"], shape)
                if path is None or not _path_clear(path, occupied, ray_blocked):
                    continue
                candidates.append(
                    {
                        "r0": r0,
                        "c0": c0,
                        "gap_dir": direction,
                        "gap_rel": gap_rel,
                        "target_kind": "source",
                        "target_ref": source_idx,
                        "target_pos": source["pos"],
                    }
                )

    if tc == w - 1:
        direction = "right"
        for gap_rel in GAP_RELS_BY_DIR[direction]:
            r0 = tr - gap_rel[0]
            for c0 in range(1, w - FRAME_SIZE):
                if not _bbox_ok(r0, c0, shape, bbox_blocked, ray_blocked):
                    continue
                gap = (r0 + gap_rel[0], c0 + gap_rel[1])
                path = _ray_prefix_to_target(gap, direction, source["pos"], shape)
                if path is None or not _path_clear(path, occupied, ray_blocked):
                    continue
                candidates.append(
                    {
                        "r0": r0,
                        "c0": c0,
                        "gap_dir": direction,
                        "gap_rel": gap_rel,
                        "target_kind": "source",
                        "target_ref": source_idx,
                        "target_pos": source["pos"],
                    }
                )

    return candidates


def _frame_candidates(
    shape: tuple[int, int],
    target_idx: int,
    target_frame: dict,
    bbox_blocked: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    ray_blocked: set[tuple[int, int]],
) -> list[dict]:
    h, w = shape
    candidates: list[dict] = []

    for direction in ("up", "down", "left", "right"):
        for target_pos in _frame_hit_pixels(target_frame, direction):
            tr, tc = target_pos
            for gap_rel in GAP_RELS_BY_DIR[direction]:
                if direction == "up":
                    c0 = tc - gap_rel[1]
                    for r0 in range(1, h - FRAME_SIZE):
                        if not _bbox_ok(r0, c0, shape, bbox_blocked, ray_blocked):
                            continue
                        gap = (r0 + gap_rel[0], c0 + gap_rel[1])
                        path = _ray_prefix_to_target(gap, direction, target_pos, shape)
                        if path is None or not _path_clear(path, occupied, ray_blocked):
                            continue
                        candidates.append(
                            {
                                "r0": r0,
                                "c0": c0,
                                "gap_dir": direction,
                                "gap_rel": gap_rel,
                                "target_kind": "frame",
                                "target_ref": target_idx,
                                "target_pos": target_pos,
                            }
                        )

                elif direction == "down":
                    c0 = tc - gap_rel[1]
                    for r0 in range(1, h - FRAME_SIZE):
                        if not _bbox_ok(r0, c0, shape, bbox_blocked, ray_blocked):
                            continue
                        gap = (r0 + gap_rel[0], c0 + gap_rel[1])
                        path = _ray_prefix_to_target(gap, direction, target_pos, shape)
                        if path is None or not _path_clear(path, occupied, ray_blocked):
                            continue
                        candidates.append(
                            {
                                "r0": r0,
                                "c0": c0,
                                "gap_dir": direction,
                                "gap_rel": gap_rel,
                                "target_kind": "frame",
                                "target_ref": target_idx,
                                "target_pos": target_pos,
                            }
                        )

                elif direction == "left":
                    r0 = tr - gap_rel[0]
                    for c0 in range(1, w - FRAME_SIZE):
                        if not _bbox_ok(r0, c0, shape, bbox_blocked, ray_blocked):
                            continue
                        gap = (r0 + gap_rel[0], c0 + gap_rel[1])
                        path = _ray_prefix_to_target(gap, direction, target_pos, shape)
                        if path is None or not _path_clear(path, occupied, ray_blocked):
                            continue
                        candidates.append(
                            {
                                "r0": r0,
                                "c0": c0,
                                "gap_dir": direction,
                                "gap_rel": gap_rel,
                                "target_kind": "frame",
                                "target_ref": target_idx,
                                "target_pos": target_pos,
                            }
                        )

                else:
                    r0 = tr - gap_rel[0]
                    for c0 in range(1, w - FRAME_SIZE):
                        if not _bbox_ok(r0, c0, shape, bbox_blocked, ray_blocked):
                            continue
                        gap = (r0 + gap_rel[0], c0 + gap_rel[1])
                        path = _ray_prefix_to_target(gap, direction, target_pos, shape)
                        if path is None or not _path_clear(path, occupied, ray_blocked):
                            continue
                        candidates.append(
                            {
                                "r0": r0,
                                "c0": c0,
                                "gap_dir": direction,
                                "gap_rel": gap_rel,
                                "target_kind": "frame",
                                "target_ref": target_idx,
                                "target_pos": target_pos,
                            }
                        )

    return candidates


def _none_candidates(
    shape: tuple[int, int],
    bbox_blocked: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    ray_blocked: set[tuple[int, int]],
) -> list[dict]:
    h, w = shape
    candidates: list[dict] = []
    for r0 in range(1, h - FRAME_SIZE):
        for c0 in range(1, w - FRAME_SIZE):
            if not _bbox_ok(r0, c0, shape, bbox_blocked, ray_blocked):
                continue
            for direction, rels in GAP_RELS_BY_DIR.items():
                for gap_rel in rels:
                    gap = (r0 + gap_rel[0], c0 + gap_rel[1])
                    path = _ray_cells(gap, direction, shape)
                    if not path:
                        continue
                    if not _path_clear(path, occupied, ray_blocked):
                        continue
                    candidates.append(
                        {
                            "r0": r0,
                            "c0": c0,
                            "gap_dir": direction,
                            "gap_rel": gap_rel,
                            "target_kind": "none",
                            "target_ref": None,
                            "target_pos": None,
                        }
                    )
    return candidates


def _collect_candidates(
    shape: tuple[int, int],
    sources: list[dict],
    frames: list[dict],
    bbox_blocked: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    ray_blocked: set[tuple[int, int]],
) -> dict[str, list[dict]]:
    out = {"source": [], "frame": [], "none": []}
    for source_idx, source in enumerate(sources):
        out["source"].extend(_source_candidates(shape, source_idx, source, bbox_blocked, occupied, ray_blocked))
    for target_idx, target_frame in enumerate(frames):
        out["frame"].extend(_frame_candidates(shape, target_idx, target_frame, bbox_blocked, occupied, ray_blocked))
    out["none"].extend(_none_candidates(shape, bbox_blocked, occupied, ray_blocked))
    return out


def _apply_candidate(
    candidate: dict,
    shape: tuple[int, int],
    frames: list[dict],
    bbox_blocked: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    ray_blocked: set[tuple[int, int]],
) -> None:
    frame = _make_frame(
        candidate["r0"],
        candidate["c0"],
        candidate["gap_dir"],
        candidate["gap_rel"],
        candidate["target_kind"],
        candidate["target_ref"],
        candidate["target_pos"],
        shape,
    )
    frames.append(frame)
    occupied.update(frame["border"])
    bbox_blocked.update(frame["bbox_block"])
    ray_blocked.add(frame["gap"])
    if frame["target_kind"] == "none":
        ray_blocked.update(_ray_cells(frame["gap"], frame["gap_dir"], shape))
    else:
        prefix = _ray_prefix_to_target(frame["gap"], frame["gap_dir"], frame["target_pos"], shape)
        assert prefix is not None
        ray_blocked.update(prefix)


def _scene_input(shape: tuple[int, int], sources: list[dict], frames: list[dict]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for source in sources:
        r, c = source["pos"]
        grid[r, c] = int(source["color"])
    for frame in frames:
        for r, c in frame["border"]:
            grid[r, c] = GREY
    return grid


def _expected_output(grid: np.ndarray, sources: list[dict], frames: list[dict]) -> np.ndarray:
    source_colors = {idx: int(source["color"]) for idx, source in enumerate(sources)}
    memo: dict[int, int] = {}

    def resolve(idx: int) -> int:
        if idx in memo:
            return memo[idx]
        frame = frames[idx]
        if frame["target_kind"] == "source":
            color = source_colors[int(frame["target_ref"])]
        elif frame["target_kind"] == "frame":
            color = resolve(int(frame["target_ref"]))
        else:
            color = BLACK
        memo[idx] = int(color)
        return int(color)

    out = grid.copy()
    for idx, frame in enumerate(frames):
        color = resolve(idx)
        if color == BLACK:
            continue
        r0 = int(frame["r0"])
        c0 = int(frame["c0"])
        out[r0 + 1 : r0 + 3, c0 + 1 : c0 + 3] = color
    return out


def _resolved_color(idx: int, sources: list[dict], frames: list[dict], memo: dict[int, int]) -> int:
    if idx in memo:
        return memo[idx]
    frame = frames[idx]
    if frame["target_kind"] == "source":
        color = int(sources[int(frame["target_ref"])]["color"])
    elif frame["target_kind"] == "frame":
        color = _resolved_color(int(frame["target_ref"]), sources, frames, memo)
    else:
        color = BLACK
    memo[idx] = color
    return color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    shape = (rng.randint(*GRID_H_RANGE), rng.randint(*GRID_W_RANGE))
    source_count = rng.randint(*SOURCE_COUNT_RANGE)
    frame_count = rng.randint(*FRAME_COUNT_RANGE)

    source_positions = _source_positions(shape, source_count, rng)
    if source_positions is None:
        return None

    source_colors = rng.sample(COLOR_POOL, source_count)
    sources = [{"pos": pos, "color": color} for pos, color in zip(source_positions, source_colors, strict=False)]

    bbox_blocked: set[tuple[int, int]] = set()
    occupied = {tuple(source["pos"]) for source in sources}
    ray_blocked: set[tuple[int, int]] = set()
    frames: list[dict] = []

    root_count = min(frame_count, rng.randint(1, min(2, source_count)))
    require_chain = frame_count >= 3 and rng.random() < 0.75
    require_none = frame_count >= 4 and rng.random() < 0.6

    for _ in range(root_count):
        source_choices = _collect_candidates(shape, sources, frames, bbox_blocked, occupied, ray_blocked)["source"]
        if not source_choices:
            return None
        _apply_candidate(rng.choice(source_choices), shape, frames, bbox_blocked, occupied, ray_blocked)

    forced_types: list[str] = []
    if require_chain and len(frames) < frame_count:
        forced_types.append("frame")
    if require_none and len(frames) < frame_count:
        forced_types.append("none")
    rng.shuffle(forced_types)

    for target_type in forced_types:
        choices = _collect_candidates(shape, sources, frames, bbox_blocked, occupied, ray_blocked)[target_type]
        if not choices:
            return None
        _apply_candidate(rng.choice(choices), shape, frames, bbox_blocked, occupied, ray_blocked)

    while len(frames) < frame_count:
        by_type = _collect_candidates(shape, sources, frames, bbox_blocked, occupied, ray_blocked)
        available_types = [kind for kind, choices in by_type.items() if choices]
        if not available_types:
            return None
        target_type = rng.choice(available_types)
        _apply_candidate(rng.choice(by_type[target_type]), shape, frames, bbox_blocked, occupied, ray_blocked)

    resolved_memo: dict[int, int] = {}
    resolved_colors = [_resolved_color(idx, sources, frames, resolved_memo) for idx in range(len(frames))]
    has_colored_chain = any(
        frame["target_kind"] == "frame" and resolved_colors[idx] != BLACK for idx, frame in enumerate(frames)
    )
    if require_chain and not has_colored_chain:
        return None
    if require_none and not any(frame["target_kind"] == "none" for frame in frames):
        return None

    grid = _scene_input(shape, sources, frames)
    found_frames, _ = _find_frames(grid)
    if len(found_frames) != len(frames):
        return None

    expected = _expected_output(grid, sources, frames)
    got = main(grid)
    if not np.array_equal(got, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(random)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
