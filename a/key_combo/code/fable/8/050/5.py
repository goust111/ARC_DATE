import random
from typing import Any

import numpy as np


BACKGROUND = 0
PUSHER = 2
SLIDER_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]

MIN_WIDTH = 22
MAX_WIDTH = 30
MIN_ACTIVE_ROWS = 4
MAX_ACTIVE_ROWS = 6
MAX_GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _parse_row(row: np.ndarray, bg: int) -> tuple[int, list[dict[str, int]]] | None:
    if np.all(row == bg):
        return None

    width = row.shape[0]
    pusher_len = 0
    while pusher_len < width and int(row[pusher_len]) == PUSHER:
        pusher_len += 1
    assert pusher_len > 0, "every non-empty row must start with a left-edge red pusher"
    assert not np.any(row[pusher_len:] == PUSHER), "red is reserved for the pusher"

    segments: list[dict[str, int]] = []
    c = pusher_len
    saw_gap_after_pusher = False
    while c < width:
        if int(row[c]) == bg:
            saw_gap_after_pusher = True
            c += 1
            continue

        assert saw_gap_after_pusher, "first slider must be separated from the pusher"
        color = int(row[c])
        start = c
        while c < width and int(row[c]) == color:
            c += 1
        end = c - 1
        segments.append({"start": start, "end": end, "color": color})
        if c < width:
            assert int(row[c]) == bg, "adjacent sliders are not allowed"
            saw_gap_after_pusher = True

    assert segments, "every active row must contain at least one slider"
    return pusher_len, segments


def _simulate_segments(
    width: int, pusher_len: int, segments: list[dict[str, int]]
) -> tuple[list[dict[str, int]], dict[str, Any]]:
    moved = [seg.copy() for seg in segments]
    momentum = int(pusher_len)
    current = 0
    info: dict[str, Any] = {
        "transfers": 0,
        "exact_contact_stop": False,
        "open_stop": False,
        "edge_annihilation": False,
        "moved_distances": [0] * len(moved),
    }

    while momentum > 0 and current < len(moved):
        seg = moved[current]
        seg_len = seg["end"] - seg["start"] + 1
        assert seg_len >= 1

        if current + 1 < len(moved):
            next_start = moved[current + 1]["start"]
            gap = next_start - seg["end"] - 1
            assert gap >= 1, "sliders must be strictly separated in the input"
            if momentum > gap:
                seg["start"] += gap
                seg["end"] += gap
                info["moved_distances"][current] += gap
                momentum -= gap
                info["transfers"] += 1
                current += 1
                continue

            seg["start"] += momentum
            seg["end"] += momentum
            info["moved_distances"][current] += momentum
            if momentum == gap:
                info["exact_contact_stop"] = True
            else:
                info["open_stop"] = True
            momentum = 0
            break

        edge_dist = width - 1 - seg["end"]
        if momentum > edge_dist:
            seg["start"] += edge_dist
            seg["end"] += edge_dist
            info["moved_distances"][current] += edge_dist
            info["edge_annihilation"] = True
            momentum = 0
            break

        seg["start"] += momentum
        seg["end"] += momentum
        info["moved_distances"][current] += momentum
        if momentum < edge_dist:
            info["open_stop"] = True
        momentum = 0
        break

    return moved, info


def _render_row(width: int, pusher_len: int, segments: list[dict[str, int]]) -> np.ndarray:
    row = np.full(width, BACKGROUND, dtype=int)
    row[:pusher_len] = PUSHER
    for seg in segments:
        row[seg["start"] : seg["end"] + 1] = seg["color"]
    return row


def _step_oracle(
    width: int, pusher_len: int, segments: list[dict[str, int]]
) -> tuple[list[dict[str, int]], dict[str, Any]]:
    moved = [seg.copy() for seg in segments]
    momentum = int(pusher_len)
    current = 0
    transfers = 0
    exact_contact_stop = False
    edge_annihilation = False

    while momentum > 0 and current < len(moved):
        seg = moved[current]
        if current + 1 < len(moved) and seg["end"] + 1 == moved[current + 1]["start"]:
            current += 1
            continue

        if current + 1 == len(moved) and seg["end"] == width - 1:
            edge_annihilation = True
            break

        seg["start"] += 1
        seg["end"] += 1
        momentum -= 1

        if current + 1 < len(moved) and seg["end"] + 1 == moved[current + 1]["start"]:
            if momentum > 0:
                transfers += 1
                current += 1
            else:
                exact_contact_stop = True
                break
        elif current + 1 == len(moved) and seg["end"] == width - 1:
            if momentum > 0:
                edge_annihilation = True
            break

    open_stop = False
    if not exact_contact_stop and not edge_annihilation:
        if current < len(moved):
            if current + 1 < len(moved):
                open_stop = moved[current]["end"] + 1 < moved[current + 1]["start"]
            else:
                open_stop = moved[current]["end"] < width - 1

    info = {
        "transfers": transfers,
        "exact_contact_stop": exact_contact_stop,
        "open_stop": open_stop,
        "edge_annihilation": edge_annihilation,
    }
    return moved, info


def main(grid: np.ndarray) -> np.ndarray:
    """Propagate each red pusher's finite rightward momentum through a row-wise collision chain.

    Input structure:
    - The grid has a dominant background and several otherwise empty rows.
    - Each active row contains one red horizontal pusher bar flush with the left edge.
    - To its right are one or more non-touching monochrome horizontal slider segments.

    Rule:
    - Process each active row independently; the pusher stays fixed and carries momentum equal to its length.
    - The nearest slider moves right, spending one momentum per cell.
    - If it becomes adjacent to the next slider before momentum is exhausted, it stops immediately and transfers all remaining momentum to that next slider; if momentum reaches zero exactly on contact, the chain stops there.
    - If the moving slider reaches the right boundary, it stops and any leftover momentum disappears.

    Key insight:
    - The whole row shares one conserved packet of momentum that is gradually spent across gaps and motion, so final positions depend on both pusher length and the sequence of slider separations.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _background_color(g)

    out = np.full_like(g, bg)
    for r in range(g.shape[0]):
        parsed = _parse_row(g[r], bg)
        if parsed is None:
            continue
        pusher_len, segments = parsed
        moved, _info = _simulate_segments(g.shape[1], pusher_len, segments)
        out[r] = _render_row(g.shape[1], pusher_len, moved)
    return out


def _sample_colors(rng: random.Random, count: int) -> list[int]:
    return rng.sample(SLIDER_COLORS, count)


def _segments_from_lengths(
    pusher_len: int, gap0: int, lengths: list[int], gaps: list[int], colors: list[int]
) -> list[dict[str, int]]:
    start = pusher_len + gap0
    segments = []
    for i, length in enumerate(lengths):
        segments.append({"start": start, "end": start + length - 1, "color": colors[i]})
        if i < len(gaps):
            start += length + gaps[i]
    return segments


def _build_exact_contact_row(width: int, rng: random.Random) -> dict[str, Any] | None:
    for _ in range(80):
        pusher_len = rng.randint(2, min(8, width - 10))
        gap0 = rng.randint(1, 4)
        lengths = [rng.randint(1, 4), rng.randint(1, 4)]
        gaps = [pusher_len]
        if rng.random() < 0.7:
            lengths.append(rng.randint(1, 4))
            gaps.append(rng.randint(1, 5))
        colors = _sample_colors(rng, len(lengths))
        segments = _segments_from_lengths(pusher_len, gap0, lengths, gaps, colors)
        if segments[-1]["end"] < width:
            moved, info = _simulate_segments(width, pusher_len, segments)
            if info["exact_contact_stop"] and info["transfers"] == 0 and not info["edge_annihilation"]:
                return {"tag": "exact_contact", "pusher_len": pusher_len, "segments": segments, "moved": moved}
    return None


def _build_partial_chain_row(width: int, rng: random.Random) -> dict[str, Any] | None:
    for _ in range(100):
        pusher_len = rng.randint(3, min(8, width - 14))
        gap0 = rng.randint(1, 4)
        first_gap = rng.randint(1, pusher_len - 1)
        remaining = pusher_len - first_gap
        second_gap = rng.randint(remaining + 1, remaining + 5)
        lengths = [rng.randint(1, 4), rng.randint(1, 4), rng.randint(1, 4)]
        gaps = [first_gap, second_gap]
        colors = _sample_colors(rng, len(lengths))
        segments = _segments_from_lengths(pusher_len, gap0, lengths, gaps, colors)
        if segments[-1]["end"] < width:
            moved, info = _simulate_segments(width, pusher_len, segments)
            if info["transfers"] == 1 and info["open_stop"] and not info["exact_contact_stop"]:
                return {"tag": "partial_chain", "pusher_len": pusher_len, "segments": segments, "moved": moved}
    return None


def _build_edge_row(width: int, rng: random.Random) -> dict[str, Any] | None:
    for _ in range(100):
        pusher_len = rng.randint(4, min(8, width - 9))
        gap0 = rng.randint(1, 4)
        first_gap = rng.randint(1, pusher_len - 1)
        lengths = [rng.randint(1, 4), rng.randint(1, 4)]
        gaps = [first_gap]
        colors = _sample_colors(rng, len(lengths))
        segments = _segments_from_lengths(pusher_len, gap0, lengths, gaps, colors)
        if segments[-1]["end"] >= width:
            continue
        moved, info = _simulate_segments(width, pusher_len, segments)
        if info["transfers"] >= 1 and info["edge_annihilation"]:
            return {"tag": "edge", "pusher_len": pusher_len, "segments": segments, "moved": moved}
    return None


def _build_multi_transfer_row(width: int, rng: random.Random) -> dict[str, Any] | None:
    for _ in range(120):
        pusher_len = rng.randint(5, min(9, width - 16))
        gap0 = rng.randint(1, 4)
        gap1 = rng.randint(1, pusher_len - 2)
        remaining = pusher_len - gap1
        gap2 = rng.randint(1, remaining - 1)
        remaining2 = remaining - gap2
        gap3 = rng.randint(remaining2 + 1, remaining2 + 5)
        lengths = [rng.randint(1, 3), rng.randint(1, 3), rng.randint(1, 3), rng.randint(1, 3)]
        gaps = [gap1, gap2, gap3]
        colors = _sample_colors(rng, len(lengths))
        segments = _segments_from_lengths(pusher_len, gap0, lengths, gaps, colors)
        if segments[-1]["end"] >= width:
            continue
        moved, info = _simulate_segments(width, pusher_len, segments)
        if info["transfers"] >= 2 and info["open_stop"] and not info["edge_annihilation"]:
            return {"tag": "multi_transfer", "pusher_len": pusher_len, "segments": segments, "moved": moved}
    return None


def _build_random_row(width: int, rng: random.Random) -> dict[str, Any] | None:
    builders = [
        _build_partial_chain_row,
        _build_exact_contact_row,
        _build_edge_row,
        _build_multi_transfer_row,
    ]
    for _ in range(40):
        row = rng.choice(builders)(width, rng)
        if row is not None:
            return row
    return None


def _row_signature(row: np.ndarray) -> bytes:
    return row.astype(np.int8, copy=False).tobytes()


def _grid_from_rows(height: int, width: int, active_rows: list[int], rows: list[dict[str, Any]]) -> np.ndarray:
    grid = np.full((height, width), BACKGROUND, dtype=int)
    for r, row in zip(active_rows, rows):
        grid[r] = _render_row(width, row["pusher_len"], row["segments"])
    return grid


def _choose_active_rows(height: int, count: int, rng: random.Random) -> list[int] | None:
    available = list(range(1, height - 1))
    rng.shuffle(available)
    chosen: list[int] = []
    for r in available:
        if all(abs(r - prev) >= 2 for prev in chosen):
            chosen.append(r)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _validate_semantics(inp: np.ndarray, rows: list[dict[str, Any]], active_rows: list[int]) -> bool:
    out = main(inp)
    if out.shape != inp.shape:
        return False

    seen_input_rows: set[bytes] = set()
    seen_output_rows: set[bytes] = set()
    branch_tags = {row["tag"] for row in rows}
    if branch_tags != {"partial_chain", "exact_contact", "edge", "multi_transfer"}:
        return False

    for row_idx, row_spec in zip(active_rows, rows):
        parsed = _parse_row(inp[row_idx], BACKGROUND)
        if parsed is None:
            return False
        pusher_len, segments = parsed
        if pusher_len != row_spec["pusher_len"]:
            return False

        moved_fast, info_fast = _simulate_segments(inp.shape[1], pusher_len, segments)
        moved_step, info_step = _step_oracle(inp.shape[1], pusher_len, segments)
        fast_row = _render_row(inp.shape[1], pusher_len, moved_fast)
        step_row = _render_row(inp.shape[1], pusher_len, moved_step)
        if not np.array_equal(fast_row, step_row):
            return False
        if info_fast["transfers"] != info_step["transfers"]:
            return False
        if info_fast["exact_contact_stop"] != info_step["exact_contact_stop"]:
            return False
        if info_fast["open_stop"] != info_step["open_stop"]:
            return False
        if info_fast["edge_annihilation"] != info_step["edge_annihilation"]:
            return False
        if not np.array_equal(out[row_idx], fast_row):
            return False

        input_sig = _row_signature(inp[row_idx])
        output_sig = _row_signature(out[row_idx])
        if input_sig in seen_input_rows or output_sig in seen_output_rows:
            return False
        seen_input_rows.add(input_sig)
        seen_output_rows.add(output_sig)

    inactive_rows = set(range(inp.shape[0])) - set(active_rows)
    for r in inactive_rows:
        if not np.all(inp[r] == BACKGROUND):
            return False
        if not np.all(out[r] == BACKGROUND):
            return False

    output_changes = sum(int(not np.array_equal(inp[r], out[r])) for r in active_rows)
    if output_changes != len(active_rows):
        return False

    return True


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    width = rng.randint(MIN_WIDTH, MAX_WIDTH)
    required_rows = [
        _build_partial_chain_row(width, rng),
        _build_exact_contact_row(width, rng),
        _build_edge_row(width, rng),
        _build_multi_transfer_row(width, rng),
    ]
    if any(row is None for row in required_rows):
        return None

    row_count = rng.randint(MIN_ACTIVE_ROWS, MAX_ACTIVE_ROWS)
    rows = [row for row in required_rows if row is not None]
    while len(rows) < row_count:
        extra = _build_random_row(width, rng)
        if extra is None:
            return None
        rows.append(extra)
    rng.shuffle(rows)

    height = rng.randint(max(10, row_count * 2 + 1), min(30, row_count * 4 + 6))
    active_rows = _choose_active_rows(height, row_count, rng)
    if active_rows is None:
        return None

    inp = _grid_from_rows(height, width, active_rows, rows)
    if _validate_semantics(inp, rows, active_rows):
        return inp
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid momentum-chain input")
