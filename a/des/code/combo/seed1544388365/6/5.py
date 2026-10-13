from __future__ import annotations

import numpy as np


BG = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 300


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _strict_contains(
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
) -> bool:
    return (
        outer[0] < inner[0]
        and outer[1] < inner[1]
        and outer[2] > inner[2]
        and outer[3] > inner[3]
    )


def _shrink_bbox(
    bbox: tuple[int, int, int, int], delta: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return (r0 + delta, c0 + delta, r1 - delta, c1 - delta)


def _find_components(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []

            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc]:
                        if int(grid[nr, nc]) == color:
                            seen[nr, nc] = True
                            stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(coords.shape[0]),
                }
            )

    return components


def _classify_component(component: dict[str, object]) -> str | None:
    coords = np.asarray(component["coords"], dtype=int)
    r0, c0, r1, c1 = component["bbox"]  # type: ignore[assignment]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    area = int(component["area"])

    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

    if h >= 3 and w >= 3 and area == 2 * (h + w) - 4:
        if (
            bool(mask[0, :].all())
            and bool(mask[-1, :].all())
            and bool(mask[:, 0].all())
            and bool(mask[:, -1].all())
            and not bool(mask[1:-1, 1:-1].any())
        ):
            return "frame"

    if h == w and h in (1, 2) and area == h * w and bool(mask.all()):
        return "core"

    return None


def _extract_sequences(grid: np.ndarray) -> list[list[int]]:
    components = _find_components(grid)
    assert components, "expected at least one nested structure"

    objects: list[dict[str, object]] = []
    for component in components:
        kind = _classify_component(component)
        assert kind is not None, "all foreground objects must be frames or small cores"
        obj = dict(component)
        obj["kind"] = kind
        objects.append(obj)

    parents: list[int | None] = [None] * len(objects)
    children: list[list[int]] = [[] for _ in objects]

    for i, obj in enumerate(objects):
        containers = [
            j
            for j, other in enumerate(objects)
            if i != j
            and other["kind"] == "frame"
            and _strict_contains(
                other["bbox"],  # type: ignore[arg-type]
                obj["bbox"],  # type: ignore[arg-type]
            )
        ]
        if containers:
            parent = min(containers, key=lambda j: _bbox_area(objects[j]["bbox"]))  # type: ignore[arg-type]
            parents[i] = parent
            children[parent].append(i)

    roots: list[int] = []
    for i, obj in enumerate(objects):
        kind = obj["kind"]
        if parents[i] is None:
            assert kind == "frame", "each root must be an outer frame"
            roots.append(i)
        if kind == "core":
            assert not children[i], "core objects cannot contain children"
        else:
            assert len(children[i]) <= 1, "nested structures must form a single chain"
            if children[i]:
                child = objects[children[i][0]]
                assert _shrink_bbox(obj["bbox"], 2) == child["bbox"]  # type: ignore[arg-type]

    assert 2 <= len(roots) <= 3, "expected two or three independent structures"
    roots.sort(key=lambda i: (objects[i]["bbox"][0], objects[i]["bbox"][1]))  # type: ignore[index]

    sequences: list[list[int]] = []
    for root in roots:
        seq: list[int] = []
        current = root
        while True:
            seq.append(int(objects[current]["color"]))
            if not children[current]:
                break
            current = children[current][0]
        sequences.append(seq)

    return sequences


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flatten each nested border structure into one output row of colors.

    Groups are ordered by the top-left corner of their outermost frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert int(g.min()) >= 0

    sequences = _extract_sequences(g)
    width = max(len(seq) for seq in sequences)
    out = np.full((len(sequences), width), BG, dtype=int)
    for r, seq in enumerate(sequences):
        out[r, : len(seq)] = seq
    return out


def _paint_frame(
    grid: np.ndarray, bbox: tuple[int, int, int, int], color: int
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _make_group(
    rng: np.random.Generator, max_frames: int, max_void: int
) -> tuple[np.ndarray, list[int]]:
    frame_count = int(rng.integers(1, max_frames + 1))
    has_core = bool(rng.integers(0, 2))

    if has_core:
        core_size = int(rng.choice([1, 2]))
        height = core_size + 4 * frame_count
        width = core_size + 4 * frame_count
        seq_len = frame_count + 1
    else:
        void_h = int(rng.integers(1, max_void + 1))
        void_w = int(rng.integers(1, max_void + 2))
        height = void_h + 2 + 4 * (frame_count - 1)
        width = void_w + 2 + 4 * (frame_count - 1)
        seq_len = frame_count

    colors = list(map(int, rng.choice(np.arange(1, 10), size=seq_len, replace=False)))
    patch = np.full((height, width), BG, dtype=int)

    sequence: list[int] = []
    for layer in range(frame_count):
        bbox = (2 * layer, 2 * layer, height - 1 - 2 * layer, width - 1 - 2 * layer)
        _paint_frame(patch, bbox, colors[layer])
        sequence.append(colors[layer])

    if has_core:
        start = 2 * frame_count
        patch[start : start + core_size, start : start + core_size] = colors[-1]
        sequence.append(colors[-1])

    return patch, sequence


def _render_expected(sequences: list[list[int]]) -> np.ndarray:
    width = max(len(seq) for seq in sequences)
    out = np.full((len(sequences), width), BG, dtype=int)
    for r, seq in enumerate(sequences):
        out[r, : len(seq)] = seq
    return out


def _generate_case(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray] | None:
    group_count = int(rng.integers(2, 4))
    max_frames = 3 if group_count == 2 else 2
    max_void = 4 if group_count == 2 else 3

    groups = [_make_group(rng, max_frames=max_frames, max_void=max_void) for _ in range(group_count)]
    heights = [patch.shape[0] for patch, _ in groups]
    widths = [patch.shape[1] for patch, _ in groups]

    top_margin = int(rng.integers(1, 4))
    bottom_margin = int(rng.integers(1, 4))
    row_gaps = [int(rng.integers(2, 5)) for _ in range(group_count - 1)]
    height = top_margin + bottom_margin + sum(heights) + sum(row_gaps)
    if height > 36:
        return None

    width_low = max(max(widths) + 2, 14)
    width_high = min(37, max(width_low + 1, max(widths) + 13))
    if width_low >= width_high:
        return None
    width = int(rng.integers(width_low, width_high))

    grid = np.full((height, width), BG, dtype=int)
    positions: list[tuple[int, int]] = []
    current_row = top_margin

    for i, (patch, _sequence) in enumerate(groups):
        patch_h, patch_w = patch.shape
        min_col = 1
        max_col = width - patch_w - 1
        if max_col < min_col:
            return None
        col = int(rng.integers(min_col, max_col + 1))
        grid[current_row : current_row + patch_h, col : col + patch_w] = patch
        positions.append((current_row, col))
        if i < len(row_gaps):
            current_row += patch_h + row_gaps[i]

    order = sorted(range(group_count), key=lambda i: positions[i])
    expected = _render_expected([groups[i][1] for i in order])

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _generate_case(rng)
        if case is not None:
            grid, _expected = case
            return grid
    raise RuntimeError("failed to generate a valid nested-frame input")
