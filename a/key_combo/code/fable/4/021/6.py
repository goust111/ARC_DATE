import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
SOURCE_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

MAX_GEN_ATTEMPTS = 400
HEIGHT_RANGE = (12, 21)
TOP_ROW_RANGE = (6, 12)
CONTAINER_HEIGHT_RANGE = (4, 7)
CONTAINER_COUNT_RANGE = (2, 4)
CONTAINER_WIDTH_RANGE = (3, 7)
GAP_RANGE = (1, 3)
MARGIN_RANGE = (1, 4)
SOURCE_COUNT_RANGE = (2, 4)
MAX_NET_SHIFT = 2
MAX_DEFLECTORS_PER_SOURCE = 3


def _component_bboxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    bboxes: list[tuple[int, int, int, int]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            bboxes.append((min(rows), min(cols), max(rows), max(cols)))
    return bboxes


def _find_containers(grid: np.ndarray) -> list[dict[str, int]]:
    containers = []
    for r0, c0, r1, c1 in _component_bboxes(grid == GREY):
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 3 or width < 3:
            continue
        border = np.zeros_like(grid, dtype=bool)
        border[r0, c0 : c1 + 1] = True
        border[r1, c0 : c1 + 1] = True
        border[r0 : r1 + 1, c0] = True
        border[r0 : r1 + 1, c1] = True
        if not np.all(grid[border] == GREY):
            continue
        if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] != BLACK):
            continue
        if int(np.count_nonzero(border)) != int(np.count_nonzero(grid[r0 : r1 + 1, c0 : c1 + 1] == GREY)):
            continue
        containers.append({"r0": r0, "c0": c0, "r1": r1, "c1": c1})
    containers.sort(key=lambda item: (item["r0"], item["c0"]))
    assert containers, "expected at least one hollow grey container"
    top_rows = {item["r0"] for item in containers}
    assert len(top_rows) == 1, "all containers should share one top row"
    return containers


def _collect_sources(grid: np.ndarray) -> list[tuple[int, int]]:
    sources = [(c, int(grid[0, c])) for c in range(grid.shape[1]) if int(grid[0, c]) not in (BLACK, GREY)]
    assert sources, "expected top-row source pixels"
    colors = [color for _, color in sources]
    assert len(set(colors)) == len(colors), "source colors must be distinct"
    return sources


def _collect_deflectors(
    grid: np.ndarray,
    top_row: int,
    source_colors: set[int],
    source_positions: set[tuple[int, int]],
) -> dict[int, tuple[str, int]]:
    _, w = grid.shape
    deflectors: dict[int, tuple[str, int]] = {}
    for r in range(top_row):
        left_active = (r, 0) not in source_positions and int(grid[r, 0]) in source_colors
        right_active = (r, w - 1) not in source_positions and int(grid[r, w - 1]) in source_colors
        assert not (left_active and right_active), "each row may host at most one edge deflector"
        if left_active:
            deflectors[r] = ("left", int(grid[r, 0]))
        elif right_active:
            deflectors[r] = ("right", int(grid[r, w - 1]))
    return deflectors


def _validate_structure(
    grid: np.ndarray,
    top_row: int,
    source_positions: set[tuple[int, int]],
    deflector_positions: set[tuple[int, int]],
) -> None:
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            value = int(grid[r, c])
            if value in (BLACK, GREY):
                continue
            pos = (r, c)
            if pos in source_positions:
                assert r == 0
                continue
            if pos in deflector_positions:
                assert 0 <= r < top_row and c in (0, w - 1)
                continue
            raise AssertionError(f"unexpected colored cell at {(r, c)}")


def _landing_container(containers: list[dict[str, int]], col: int) -> int | None:
    matches = [i for i, item in enumerate(containers) if item["c0"] <= col <= item["c1"]]
    assert len(matches) <= 1, "container top edges should not overlap"
    return matches[0] if matches else None


def _trace_stream(
    source_col: int,
    source_color: int,
    top_row: int,
    width: int,
    deflectors: dict[int, tuple[str, int]],
    containers: list[dict[str, int]],
) -> int | None:
    col = int(source_col)
    for r in range(top_row):
        side_and_color = deflectors.get(r)
        if side_and_color is None:
            continue
        side, color = side_and_color
        if color != source_color:
            continue
        col += 1 if side == "left" else -1
        assert 0 <= col < width, "deflected stream left the grid"
    return _landing_container(containers, col)


def _paint_container(out: np.ndarray, container: dict[str, int], fill_color: int) -> None:
    r0, c0, r1, c1 = container["r0"], container["c0"], container["r1"], container["c1"]
    out[r0, c0 : c1 + 1] = GREY
    out[r1, c0 : c1 + 1] = GREY
    out[r0 : r1 + 1, c0] = GREY
    out[r0 : r1 + 1, c1] = GREY
    out[r0 + 1 : r1, c0 + 1 : c1] = fill_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route each top-edge stream through same-color edge deflectors and keep only the containers that end up claimed.

    Input structure:
    - The background is black, with several distinct-colored source pixels on the top row.
    - Grey hollow rectangular containers sit side by side near the bottom and share one top-edge row.
    - Above that row, some left-edge or right-edge pixels reuse source colors as per-row deflectors.

    Rule:
    - Each source falls straight downward from its own column.
    - Whenever the stream reaches a row whose edge deflector has the same color, shift one column away from that edge and continue downward.
    - If the final column reaches a container's grey top edge, keep that container's grey frame and fill its interior with the source color; otherwise discard the stream. Containers that catch nothing become entirely black.

    Key insight:
    - Edge deflectors are private instructions keyed by color, so every stream must be simulated against the same rows but only react to its own colored deflectors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    containers = _find_containers(g)
    top_row = containers[0]["r0"]
    sources = _collect_sources(g)
    source_positions = {(0, c) for c, _ in sources}
    source_colors = {color for _, color in sources}
    deflectors = _collect_deflectors(g, top_row, source_colors, source_positions)
    deflector_positions = {
        (r, 0 if side == "left" else g.shape[1] - 1)
        for r, (side, _color) in deflectors.items()
    }
    _validate_structure(g, top_row, source_positions, deflector_positions)

    claimed: dict[int, int] = {}
    for source_col, source_color in sources:
        container_idx = _trace_stream(source_col, source_color, top_row, g.shape[1], deflectors, containers)
        if container_idx is None:
            continue
        assert container_idx not in claimed, "each container should receive at most one stream"
        claimed[container_idx] = source_color

    out = np.zeros_like(g)
    for idx, fill_color in claimed.items():
        _paint_container(out, containers[idx], fill_color)
    return out


def _choose_container_layout() -> tuple[int, int, list[dict[str, int]]]:
    for _ in range(200):
        top_row = random.randint(*TOP_ROW_RANGE)
        container_height = random.randint(*CONTAINER_HEIGHT_RANGE)
        bottom_margin = random.randint(0, 1)
        height = top_row + container_height + bottom_margin
        if not (HEIGHT_RANGE[0] <= height <= HEIGHT_RANGE[1]):
            continue

        count = random.randint(*CONTAINER_COUNT_RANGE)
        widths = [random.randint(*CONTAINER_WIDTH_RANGE) for _ in range(count)]
        gaps = [random.randint(*GAP_RANGE) for _ in range(count - 1)]
        left_margin = random.randint(*MARGIN_RANGE)
        right_margin = random.randint(*MARGIN_RANGE)
        width = left_margin + sum(widths) + sum(gaps) + right_margin
        if width > 30:
            continue

        boxes = []
        c = left_margin
        r0 = top_row
        r1 = top_row + container_height - 1
        for i, box_width in enumerate(widths):
            boxes.append({"r0": r0, "c0": c, "r1": r1, "c1": c + box_width - 1})
            c += box_width
            if i < len(gaps):
                c += gaps[i]
        return height, width, boxes
    raise RuntimeError("failed to choose a valid container layout")


def _column_hits_container(col: int, containers: list[dict[str, int]]) -> int | None:
    return _landing_container(containers, col)


def _build_sequence(source_col: int, final_col: int, width: int, force_deflection: bool) -> list[int] | None:
    delta = final_col - source_col
    if abs(delta) > MAX_NET_SHIFT:
        return None

    base = [1] * max(delta, 0) + [-1] * max(-delta, 0)
    candidates: list[list[int]] = [base]
    if force_deflection and not base:
        candidates.extend(([1, -1], [-1, 1]))
    elif base and len(base) + 2 <= MAX_DEFLECTORS_PER_SOURCE:
        if 0 < source_col < width - 1:
            candidates.extend((base + [1, -1], base + [-1, 1]))

    random.shuffle(candidates)
    for dirs in candidates:
        if len(dirs) > MAX_DEFLECTORS_PER_SOURCE:
            continue
        dirs = list(dirs)
        for _ in range(24):
            random.shuffle(dirs)
            col = source_col
            ok = True
            for step in dirs:
                col += step
                if not (0 <= col < width):
                    ok = False
                    break
            if ok and col == final_col and (dirs or not force_deflection):
                return list(dirs)
    return None


def _expected_output(shape: tuple[int, int], containers: list[dict[str, int]], claims: dict[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for idx, fill_color in claims.items():
        _paint_container(out, containers[idx], fill_color)
    return out


def _gen_attempt() -> np.ndarray | None:
    height, width, containers = _choose_container_layout()
    top_row = containers[0]["r0"]

    n_sources = random.randint(*SOURCE_COUNT_RANGE)
    n_sources = min(n_sources, len(SOURCE_PALETTE), max(2, width - 2))
    n_hits_max = min(len(containers) - 1, n_sources - 1)
    if n_hits_max < 1:
        return None
    n_hits = random.randint(1, n_hits_max)

    container_indices = list(range(len(containers)))
    random.shuffle(container_indices)
    hit_containers = container_indices[:n_hits]

    hit_final_cols = []
    for idx in hit_containers:
        box = containers[idx]
        hit_final_cols.append(random.randint(box["c0"], box["c1"]))

    miss_cols = [c for c in range(width) if _column_hits_container(c, containers) is None]
    if len(miss_cols) < n_sources - n_hits:
        return None
    miss_final_cols = random.sample(miss_cols, n_sources - n_hits)

    desired = []
    for idx, final_col in zip(hit_containers, hit_final_cols):
        desired.append({"final_col": final_col, "target": idx})
    for final_col in miss_final_cols:
        desired.append({"final_col": final_col, "target": None})
    random.shuffle(desired)

    source_cols: list[int] = []
    sequences: list[list[int]] = []
    must_change_projection = True
    deflected_any = False
    interior_cols = list(range(1, width - 1))
    random.shuffle(interior_cols)

    for i, item in enumerate(desired):
        target = item["target"]
        final_col = item["final_col"]
        force_deflection = must_change_projection or (i == 0 and random.random() < 0.5)

        candidates = []
        for source_col in interior_cols:
            if source_col in source_cols:
                continue
            direct_target = _column_hits_container(source_col, containers)
            if force_deflection and direct_target == target:
                continue
            seq = _build_sequence(source_col, final_col, width, force_deflection=False)
            if seq is not None:
                candidates.append((source_col, seq, direct_target))
            if force_deflection:
                seq = _build_sequence(source_col, final_col, width, force_deflection=True)
                if seq is not None:
                    candidates.append((source_col, seq, direct_target))

        if not candidates:
            return None
        source_col, seq, direct_target = random.choice(candidates)
        source_cols.append(source_col)
        sequences.append(seq)
        if seq:
            deflected_any = True
        if direct_target != target:
            must_change_projection = False

    if must_change_projection or not deflected_any:
        return None

    total_deflectors = sum(len(seq) for seq in sequences)
    if total_deflectors > top_row - 1:
        return None

    colors = random.sample(list(SOURCE_PALETTE), n_sources)
    grid = np.zeros((height, width), dtype=int)
    for box in containers:
        _paint_container(grid, box, BLACK)
        grid[box["r0"] + 1 : box["r1"], box["c0"] + 1 : box["c1"]] = BLACK

    available_rows = list(range(1, top_row))
    random.shuffle(available_rows)
    row_blocks = []
    cursor = 0
    for seq in sequences:
        rows = sorted(available_rows[cursor : cursor + len(seq)])
        cursor += len(seq)
        row_blocks.append(rows)

    expected_claims: dict[int, int] = {}
    for source_col, color, seq, rows in zip(source_cols, colors, sequences, row_blocks):
        grid[0, source_col] = color
        for row, step in zip(rows, seq):
            grid[row, 0 if step == 1 else width - 1] = color
        target_idx = _trace_stream(source_col, color, top_row, width, _collect_deflector_map_from_rows(rows, seq, color), containers)
        if target_idx is not None:
            if target_idx in expected_claims:
                return None
            expected_claims[target_idx] = color

    deflectors = _collect_deflectors(grid, top_row, set(colors), {(0, col) for col in source_cols})
    if len(deflectors) != total_deflectors:
        return None

    if len(expected_claims) != n_hits:
        return None

    if not any(target is None for target in (_trace_stream(col, color, top_row, width, deflectors, containers) for col, color in zip(source_cols, colors))):
        return None

    actual = main(grid)
    expected = _expected_output((height, width), containers, expected_claims)
    if not np.array_equal(actual, expected):
        return None
    return grid


def _collect_deflector_map_from_rows(
    rows: list[int],
    seq: list[int],
    color: int,
) -> dict[int, tuple[str, int]]:
    return {row: ("left" if step == 1 else "right", color) for row, step in zip(rows, seq)}


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 6")
